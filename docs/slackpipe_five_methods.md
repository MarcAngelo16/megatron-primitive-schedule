# Five-method Evaluation Contract

The current public method IDs are exactly:

| Method ID | Executor | Topology | Search |
| --- | --- | --- | --- |
| `1f1b` | `megatron-native` | N=PP, no VPP | None |
| `interleaved-1f1b` | `megatron-native` | PP>1, explicit N>PP, N divisible by PP, B divisible by PP | None |
| `octopipe-tuning` | `slackpipe` | Explicit N, cyclic placement | Restricted fixed-stage F/B OctoPipe Algorithm 1 |
| `cpsat-no-hint` | `slackpipe` | Explicit N, cyclic placement | Joint partition/order CP-SAT without external initialization |
| `slack-heuristic` | `slackpipe` | Explicit N, cyclic placement | Standalone total-float heuristic, coupled refinement |

`tools/slackpipe_eval_config.py` is the single Python registry. CLI parsing,
worker dispatch, receipts, campaign tables and plots consume it. The C++ registry
describes these contracts but executes only the three plan generators. Native
methods run the existing Megatron implementations, which this refactor does not
modify. The internal runtime flag remains `pipeline_schedule="slackpipe"`;
it is **not** a method ID.

Retired names (`slackpipe`, `interleaved`, `1f1b-interleave`, `octopipe`,
`slackpipe-refined`, `optimized_interleaved`, and old C++ algorithm names) are
rejected, not silently aliased. Changing N is an experiment parameter, not a new
optimization algorithm. Historical results retain their original labels.

## Search Semantics

All three methods use the same explicit B/N/W/L, cost profile and cyclic mapping.
Use the same wall-clock budget and one search worker for comparisons.
`--solver-workers` is distinct from `OMP_NUM_THREADS`; multi-worker CP-SAT is an
explicit, separately recorded configuration. Heuristic and OctoPipe reject
search worker counts other than one.

* CP-SAT retains legal partitions, actual per-operation duration variables
  (including profiles), data dependencies, FIFO, NoOverlap, worker/global
  workload bounds and requested supported activation caps. It builds a
  conservative horizon without evaluating any external schedule. There are no
  external hints, incumbent horizon bounds, pressure/balance restrictions,
  fixed/local partition restrictions, or fallback schedule. Model construction
  and all solve attempts share the deadline. Extracted orders undergo DAG replay
  and the separate result validator. The existing exact P/order no-good for a
  replay activation-cap violation remains. A model proof is not a GPU proof.
* The heuristic preserves structured initialization, total float, candidate
  ranking, perturbation, coupled refinement and comparison rules, including its
  valid lower-bound certificate. It is not called by CP-SAT. The one public
  time budget is passed directly to its search (no hidden 60-second cap).
  Fixed-evaluation mode is a deterministic regression diagnostic, not a
  wall-clock performance comparison.
* OctoPipe preserves phase selection, neighborhoods, validation and strict
  makespan acceptance; there is no phase fallback. Formal comparisons disable
  placement moves, keeping cyclic placement fixed. The timer now includes
  initial-state evaluation. Fixed-iteration diagnostics retain their exact
  trajectory. This is not full OctoPipe: no stage dispersion, B/W split or W ops.

Finite budgets are cooperative: a final candidate evaluation, extraction or
independent validation can finish after the nominal deadline. Actual elapsed
time is recorded, never reported as exactly the requested budget. Validation
and JSON/CSV/SVG/plan serialization are not GPU execution time.

Heuristic and OctoPipe **reject enforced activation caps**. Their post-hoc
activation analysis does not constitute a constrained search. Only CP-SAT
supports enforcement, subject to the build's cumulative-demand capabilities.
Explicit capacity units are model units, not measured CUDA peak bytes.

## Build and Minimal Checks

Run Megatron Python and tests inside the existing `slackpipe-dev` container.
Its checkout is `/workspace/Megatron-LM` and Python is `/opt/venv/bin/python`.
No host dependency installation is needed.

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev \
  cmake -S slackpipe -B slackpipe/build/no-or -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DSLACKPIPE_ENABLE_ORTOOLS=OFF
docker exec -w /workspace/Megatron-LM slackpipe-dev \
  cmake --build slackpipe/build/no-or -j 4
docker exec -w /workspace/Megatron-LM slackpipe-dev \
  ctest --test-dir slackpipe/build/no-or --output-on-failure

# From the host repository root, using the existing OR-Tools image:
docker run --rm --entrypoint /bin/bash \
  -v "$PWD:/workspace/Megatron-LM" -w /workspace/Megatron-LM \
  -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory \
  -e GIT_CONFIG_VALUE_0=/workspace/Megatron-LM \
  -e SLACKPIPE_OR_TEST_FILTER='.*' -e BUILD_JOBS=4 \
  slackpipe-ortools-runtime:local slackpipe/scripts/validate_ortools_evaluation.sh

# Immutable solver -> independent validator -> real Python plan parser smoke.
docker exec -w /workspace/Megatron-LM slackpipe-dev /opt/venv/bin/python \
  -m tools.slackpipe_artifact_smoke --output /tmp/slackpipe-smoke-new \
  --method slack-heuristic
```

Use a fresh output directory for each smoke. Repeat with `octopipe-tuning`.
For `cpsat-no-hint`, pass `--solver` pointing to the OR-enabled binary in an
environment with its OR-Tools runtime libraries. Add `--gpu` only when two GPUs
are available; this runs the existing PP=2 numerical-equivalence fixture and
checks the generated plan's runtime order. CPU parsing is not GPU equivalence.

Direct solver commands (inside the appropriate build/runtime container):

```bash
slackpipe/build/no-or/slackpipe_cli --algorithm slack-heuristic \
  --B 4 --N 4 --J 2 --L 8 --ratio-num 2 --ratio-den 1 --communication 0 \
  --time-limit 1 --num-workers 1 --output-prefix /tmp/heuristic \
  --emit-plan /tmp/heuristic.plan.json
slackpipe/build/no-or/slackpipe_cli --algorithm octopipe-tuning \
  --B 4 --N 4 --J 2 --L 8 --ratio-num 2 --ratio-den 1 --communication 0 \
  --time-limit 1 --num-workers 1 --output-prefix /tmp/octopipe \
  --emit-plan /tmp/octopipe.plan.json
slackpipe/build/ortools/slackpipe_cli --algorithm cpsat-no-hint \
  --B 4 --N 4 --J 2 --L 8 --ratio-num 2 --ratio-den 1 --communication 0 \
  --time-limit 1 --num-workers 1 --output-prefix /tmp/cpsat \
  --emit-plan /tmp/cpsat.plan.json
slackpipe/build/ortools/slackpipe_cli --validate-result /tmp/cpsat.json
```

These are abstract-cost correctness examples. Real experiments add the same
`--cost-profile PROFILE` to each optimizer. `--build-info` reports OR availability.
No-OR CP-SAT produces an `UNAVAILABLE` result and no plan (exit 3).
Invalid requests/export failures exit nonzero. A completed search without a
replay-valid solution returns a recorded `NO_VALID_SOLUTION` outcome (exit 0),
raw solver status/bound, null replay makespan, and no plan. Existing plan paths
are refused so a failed attempt cannot accidentally reuse a stale plan.

## Real-system Entry Point

For all commands below use, inside the prepared container:

```bash
python -m tools.run_slackpipe_eval --phase full --model-config CONFIG --output OUTPUT \
  --method 1f1b --pp 2 --num-stages 2 --microbatches 4
python -m tools.run_slackpipe_eval --phase full --model-config CONFIG --output OUTPUT \
  --method interleaved-1f1b --pp 2 --num-stages 4 --microbatches 4
python -m tools.run_slackpipe_eval --phase full --model-config CONFIG --output OUTPUT \
  --method octopipe-tuning --pp 2 --num-stages 4 --microbatches 4 \
  --profiling-method both --solver SOLVER --solver-seconds 300 --solver-workers 1
python -m tools.run_slackpipe_eval --phase full --model-config CONFIG --output OUTPUT \
  --method cpsat-no-hint --pp 2 --num-stages 4 --microbatches 4 \
  --profiling-method both --solver OR_ENABLED_SOLVER --solver-seconds 300 --solver-workers 1
python -m tools.run_slackpipe_eval --phase full --model-config CONFIG --output OUTPUT \
  --method slack-heuristic --pp 2 --num-stages 4 --microbatches 4 \
  --profiling-method both --solver SOLVER --solver-seconds 300 --solver-workers 1
```

`CONFIG` is a JSON in `configs/slackpipe_eval` or a matching custom model spec.
Do not attempt a large preset on insufficient memory; the preflight records a
skip instead of shrinking the model. The public names are `--phase`,
`--method` and `--num-stages`; the old positional phase, `--schedule`,
`--logical-stages` and `--calibration-estimator` are not public aliases.
Select phases with `--phase env|correctness|calibrate|solve|smoke|benchmark|trace|inspect`.
Native standalone benchmarks need neither profiling nor a solver. Plan methods
require the selected structurally valid profile, solver validation, provenance
checks and GPU smoke before timing or tracing. See [Profiling contract](slackpipe_profiling.md)
for a native benchmark, both-backend collection, layer-only solving and explicit
artifact commands.

```bash
python -m tools.run_slackpipe_real_system_campaign \
  --families llama,nemotron_h --sizes 4b --configs configs/slackpipe_eval \
  --output /tmp/five-method-campaign --pp 4 --num-stages 8 \
  --microbatches 8 --solver OR_ENABLED_SOLVER --solver-seconds 300 \
  --solver-workers 1 --warmups 30 --iterations 50 --repetitions 3 --resume
```

This is a full-size example, **not** the refactor acceptance test or an instruction
to run a sweep on the local two-GPU host. The thin campaign rotates methods,
calls `Experiment`, records failures, shares only byte/context-compatible
prerequisites, and aggregates independent runs. All three plan methods reuse one
accepted profile at the same N. Native 1F1B legitimately has N=PP.

Profiling defaults to `--profiling-method both`, sequential stage then layer with
separate artifacts. Omitted `--solver-profile` means stage for both/stage and layer
for layer-only collection. Stage alone applies the single-pass affine-residual
rule; layer retains all legal samples. Statistical stability gates and automatic
retries are removed. Nonselected backend failure is reported as partial; selected
backend failure blocks dependencies without fallback.
Benchmark mode has no profiler. Trace runs are separate captures. The ordinary
native-VPP calibration path requires a valid native topology; PP=1 multi-stage
plan execution remains tested by its dedicated runtime fixtures, not by claiming
native PP=1 interleaving exists.

## Receipts, Results and Plots

* C++ result schema and method contract version: **2**. `method_id` and
  `executor` are separate. The result contains the actual duration model,
  profile/manifest hashes, independent validation, split and native operations.
  `solver_objective_ticks`, `best_bound_ticks`, `solver_status_raw` are CP-SAT
  fields; they are null for heuristic/OctoPipe. Heuristic certificate bounds are
  explicitly separate. `makespan_ticks` is earliest deterministic replay.
* `lower_bound_report` retains raw solver bound, independent all-partition
  workload bound, effective bound and `(replay - effective_bound) / replay`.
  Partition-dependent work uses a safe global relaxation, never incumbent work.
  Status is not upgraded by reporting. No-solution results have no gap.
* Real-system receipt/context: **v4**. Method contracts bind backend, topology,
  budget, workers, cap, source and binary hashes. Earlier v1/v2/v3 receipts are
  stale; no migration can turn an old hinted solution into `cpsat-no-hint`.
  Original receipts are archived before explicit replacement. Untracked source
  hashes are recorded in addition to commit/dirty diff. Artifact tampering,
  changed parent bytes or a changed solver invalidate dependent stages.
* Successful profiles/plans are content bound. Plan v1 uses an external binding
  sidecar where needed; plan v2 retains its existing profile/manifest/range hashes.
  Native results have null solver metadata. No-solution solver receipts are
  resumable but never accepted as a prerequisite for GPU work.
* Benchmark summary v2 preserves per-rank/per-step samples. Main statistics and
  the normal 95% confidence half-width use independent continuous wall-time run
  observations. CUDA-event statistics are separate under `cuda_event_summary`;
  a single run has a null interval. Modeled F/B makespan, optimizer/communication
  costs, search time and real GPU step time are not interchangeable.
* Receipts propagate selected `profile_identity` (backend, file SHA, cost hash,
  policy) and transport into solve/benchmark/trace results. Actual measurement
  parallelism and target parallelism are recorded separately.
* Tables compare registry methods, not historical partition/refinement
  contributions. Trace figure report v2 stores a `panels` list. Select the same
  global iteration and capture cycle across all ranks/methods. The automatic
  rule is earliest common complete-rank iteration; retain all candidates.

```bash
python -m tools.plot_schedule_trace \
  --panel interleaved-1f1b=TRACE_NATIVE --panel slack-heuristic=TRACE_HEURISTIC \
  --panel cpsat-no-hint=TRACE_CPSAT --cycle 0 --iteration 34 \
  --output-png /tmp/timeline.png --output-pdf /tmp/timeline.pdf \
  --report /tmp/timeline.json
python -m tools.slackpipe_eval_tables --output CAMPAIGN --configs configs/slackpipe_eval
```

White gaps are not certified hardware idle. Operation envelopes, GPU activity
union, communication and CPU allocation calls retain their distinct meanings.
The plotter rejects old method labels instead of relabelling historical results.

## Validation and Scope

The focused Python suite includes registry/topology/budget/thread rejection,
same-profile/different-solver receipts, no-solution skip/resume, no stability retry,
artifact integrity, source identity, model/plan/cost parsing and plot alignment.
For example:

```bash
python -m pytest -q \
  tests/unit_tests/pipeline_parallel/test_slackpipe_methods.py \
  tests/unit_tests/pipeline_parallel/test_slackpipe_eval_receipts.py \
  tests/unit_tests/pipeline_parallel/test_slackpipe_campaign_plot.py \
  tests/unit_tests/pipeline_parallel/test_slackpipe_plan.py \
  tests/unit_tests/pipeline_parallel/test_slackpipe_cost_profile.py
```

GPU numerical-equivalence tests remain under `test_slackpipe_model_construction.py`,
`test_slackpipe_hybrid.py`, `test_slackpipe_heterogeneous.py`, and transport/lifecycle
fixtures. Run in the existing container with the specified GPU count. Do not
count an unavailable/skip as a pass.

This refactor adds no TP/DP/CP, overlap, recomputation, CUDA graphs or optimization
strategy. It replaces old quality filters with stage-only one-pass residual
filtering and all-sample layer medians. Existing runtime capabilities and their
validation limits remain. No full-model throughput claim follows from a unit
test, solver optimum or this cleanup. Historical Nemotron profile accuracy and
paper results are not revalidated or rewritten.

The C++ `one_f_one_b`, `interleaving_stats`, internal BFS, and
`octopipe_phase_diagnostic` remain for diagnostics/regression. Their offline
orders are **not** native Megatron baselines. Historical campaign/ablation tools
were removed rather than relocated into an active legacy tree; use the recorded
historical commit for those experiments.
