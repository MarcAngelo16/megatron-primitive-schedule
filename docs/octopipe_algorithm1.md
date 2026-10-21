# OctoPipe Algorithm-1 Fixed-Stage Baseline

> Historical implementation/experiment record. The current public method is
> `octopipe-tuning`, with cyclic placement fixed in formal comparisons. The
> commands, aliases, registry entries and test counts below describe the earlier
> checkout; they are not current reproduction instructions. Use
> [the five-method guide](slackpipe_five_methods.md) for current commands.
> The forced-phase diagnostic remains available as a diagnostic only; its
> offline interleaved initializer is not a native Megatron baseline.

**This is NOT a full reproduction of OctoPipe.** It implements the phase-selection
control logic of [Algorithm 1 in OctoPipe, arXiv:2509.23722v3](https://arxiv.org/html/2509.23722v3)
using SlackPipe's C++ representation, cost model, and evaluator. Makespans are
simulated ticks, not measured GPU performance.

| OctoPipe paper feature | This baseline |
| --- | --- |
| Iterative Algorithm 1 and bubble-aware phase selection | Yes |
| Partition tuning | Adjacent contiguous one-layer boundary moves |
| Placement tuning | Whole-stage swaps only, optionally disabled |
| Stage dispersion / stage-count refinement | No; initial N always equals final N |
| Scheduling tuning | Local F/B-only earlier insertions, including adjacent swaps |
| B/W split / W delay / selective-W advancement | No; B is complete backward |
| Separate graph simulator / memory simulator | No |
| OctoPipe executor / communication reordering | No |
| Common SlackPipe evaluator, C++ implementation, plan format | Yes |

## Architecture and Inspection Findings

`Instance` owns problem dimensions and homogeneous, affine-v1, or range-v2 costs.
`OperationId` encodes a microbatch's F chain followed by its reverse-stage B chain.
`MachineOrders` and `MachinePredecessors` are the existing worker-local schedule
representations. `EvaluateSchedule`/`EvaluateScheduleWithPredecessors` combine
data, FIFO, and worker edges, reject cycles, and compute earliest operation
start/end times and the global makespan. `ValidateScheduleSolutionIndependent`
also checks complete operation coverage, ordering, serialization, and FIFO.

The tuner in `slackpipe/src/octopipe_tuner.cc` uses these APIs unchanged. Each
proposal is evaluated and independently validated, then replayed through the
predecessor evaluator. There is no second simulator. Invalid proposals never
become tuning states. Plan export uses the existing atomic v1/v2 emitter after
the same predecessor replay; operations are native structured F/B records.

Placement was implicit `stage % workers`. `Instance::stage_to_worker` now optionally
specifies it, with one neutral lookup used by the evaluator, independent validator,
exporter, result metadata, and existing diagnostic helpers. Empty means the original
cyclic behavior. No SlackPipe CP-SAT search or Megatron execution code was changed.
The new CLI placement flag is rejected for other algorithms; the existing CP-SAT
solvers still support cyclic placement only.

## Exact Metrics and Control Rule

For each worker d, using its validated compute intervals and global makespan T:

```text
leading[d]  = first compute start
trailing[d] = T - last compute finish
boundary[d] = leading[d] + trailing[d]
residual[d] = sum of internal gaps between consecutive compute intervals
bubble[d]   = boundary[d] + residual[d]
delta_b     = max(bubble) - min(bubble)
b_bd        = sum(boundary)
b_res       = sum(residual)
```

An empty worker has leading=T, trailing=0, residual=0, bubble=T. Communication
delay remains the common evaluator's edge delay; these are compute occupancy
metrics, not claims of hardware idleness. All quantities use evaluator ticks.
The tuner uses actual evaluated intervals, not the legacy ratio-only
`TotalUsefulWork` helper, for heterogeneous bubble metrics.

For range-v2 costs, `t_layer = min_l(F_l + Backward_l)` from consecutive prefix
differences. Backward is unsplit. Fixed first/middle/last stage biases are excluded
from this **layer** threshold, but included in every candidate evaluation. Uniform
costs use `ratio_den + ratio_num`. Affine-v1 profiles lack individual layer
observations: the threshold uses the minimum stage-indexed F+B slope, excluding
fixed biases. This is an explicit approximation for v1, not synthetic W work.

```text
if delta_b > t_layer: partition
else if b_bd > b_res: fixed-stage placement
else: F/B scheduling
```

Both comparisons are strict. Equality proceeds to the next branch. Bubble metrics
select the phase; only the shared evaluator's makespan selects improvements.

## Neighborhoods and Stopping

* **Partition:** all legal +/-1 layer transfers at every adjacent stage boundary.
  Respect `min_layers`, fixed N, sum L, and contiguous global ranges. Prioritize
  transfers from more occupied workers to less occupied workers (the latter have
  more bubble time). Recompute actual range costs, not layer-count approximations.
* **Fixed-stage placement:** all cross-worker stage swaps, prioritized by boundary
  bubble difference. P and each worker's stage count remain fixed. Regroup the
  current evaluated global operation order `(start,end,id)` onto the new workers,
  then validate it. Invalid equal-time projections are rejected, not repaired by
  an alternate simulator. No relocations, layer exchanges, or stage dispersion.
* **F/B scheduling:** prioritize large gaps, with deterministic worker/slot ties.
  Try advancing each of the next four operations into a slot; at most 64 proposals
  by default (`--octopipe-candidates-per-iteration`). Includes adjacent swaps and
  leading slots as well as residual gaps. Shared validators determine legality;
  edits cannot relax data or FIFO constraints. No exhaustive permutation search.

Select the best legal neighbor; equal-cost neighbors use lexicographic split,
placement, then native orders. Accept only a strict improvement over the current
state, so current is also best-so-far. No plateau moves, random tie breaks, CP-SAT
calls, or fallback to another phase when the selected phase cannot improve.

`--time-limit-seconds` covers the tuning loop, including metrics, candidate
generation, evaluation, and progress callbacks. Initialization and output are
outside that loop. Deadline checks occur during proposal generation and before
each evaluation; an in-flight evaluator call is not preempted. A wall-clock-only
stalled iteration sleeps 1 ms, then repeats the same phase until the deadline.
`--octopipe-max-iterations` bounds iterations without early convergence exit.
Set time limit to 0 for deterministic iteration-only experiments; at least one
limit must be positive. With both limits, the first reached stops tuning.

The API accepts an exact initial split, placement, and native worker orders.
The CLI defaults to the same `UniformSplit` and `BreadthFirstOrders` routines used
by SlackPipe's `--bfs-method uniform`; `--split` and `--stage-to-worker` override
the initial partition and mapping. There is no existing SlackPipe CLI input for
arbitrary initial worker orders, and this change does not add one.

## Fair Comparisons and Runtime Compatibility

For shared **cyclic placement** experiments, use `--octopipe-fixed-placement true`
and SlackPipe `--bfs-method uniform`. The placement phase then does nothing if
selected; it does not fall through. Without this switch the baseline explores
whole-stage placement that current SlackPipe does not search, so label that result
as an expanded-placement experiment, not an identical-search-space comparison.
The algorithms also intentionally have different schedule neighborhoods: SlackPipe
retains its existing predecessor-candidate restrictions; OctoPipe checks local
edits against the common data/FIFO/serialization DAG without adding that heuristic
restriction. Both evaluate the same representation with the same objective.

Both cyclic and noncyclic plans load in the production Megatron **plan parser**.
The Megatron **execution path still requires cyclic placement**; noncyclic plans
are simulation-only. This patch does not add a new runtime or communication path.
Activation-cap enforcement and pressure/worker-balance search restrictions are
unsupported for this baseline and rejected. Existing activation diagnostics are
still emitted; this is not OctoPipe's memory simulator. The tuner reports FEASIBLE,
never an optimality certificate (`--require-optimal` is a CP-SAT control).

## Commands

Run from the monorepo root. Build without OR-Tools:

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev \
  cmake -S slackpipe -B slackpipe/build/no-or -G Ninja -DSLACKPIPE_ENABLE_ORTOOLS=OFF
docker exec -w /workspace/Megatron-LM slackpipe-dev \
  cmake --build slackpipe/build/no-or -j2
docker exec -w /workspace/Megatron-LM slackpipe-dev \
  ctest --test-dir slackpipe/build/no-or --output-on-failure
```

Use the [README OR-Tools validation command](../README.md#build-the-optimizer) to build
`slackpipe/build/release`. Run the OR-enabled binary in the same solver image,
so the example does not depend on untracked copied libraries. These commands share B/N/W/L, costs, cyclic
placement, uniform-BFS initialization, and a two-second limit:

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace/Megatron-LM" -w /workspace/Megatron-LM \
  --entrypoint ./slackpipe/build/release/slackpipe_cli slackpipe-ortools-runtime:local \
  --algorithm slackpipe --B 4 --N 4 --J 2 --L 8 \
  --ratio-num 1 --ratio-den 1 --bfs-method uniform \
  --time-limit-seconds 2 --num-workers 1 --random-seed 1 --require-optimal false \
  --output-prefix slackpipe/build/slackpipe-homo \
  --emit-plan slackpipe/build/slackpipe-homo.plan.json

docker exec -w /workspace/Megatron-LM slackpipe-dev slackpipe/build/no-or/slackpipe_cli \
  --algorithm octopipe-algorithm1-fixed-stage --B 4 --N 4 --J 2 --L 8 \
  --ratio-num 1 --ratio-den 1 --bfs-method uniform --octopipe-fixed-placement true \
  --time-limit-seconds 2 --num-workers 1 --random-seed 1 --require-optimal false \
  --output-prefix slackpipe/build/octopipe-homo \
  --emit-plan slackpipe/build/octopipe-homo.plan.json
```

For both heterogeneous runs add
`--cost-profile slackpipe/tests/fixtures/octopipe_heterogeneous.cost_profile.json`
and change `homo` to `hetero` in output paths. This fixture is explicitly synthetic,
not a measured performance profile. For deterministic diagnostics replace the
OctoPipe time limit by `--time-limit-seconds 0 --octopipe-max-iterations 20` and add
`--log-search-progress true`. Remove `--octopipe-fixed-placement true` to exercise
the full restricted tuner's placement phase.

The normal result JSON/CSV/orders/SVG and `--emit-plan` formats are reused.
Result metadata identifies `octopipe-algorithm1-fixed-stage` and records the command,
mapping, and method contract. Progress goes to stderr and reports iteration,
elapsed time, input makespan, all phase metrics, threshold, phase, attempted/valid
candidate counts, acceptance, and best makespan. Search-stat accepted/rejected
counts describe candidate **validity**; progress `accepted` means a strict state
improvement. No W operation or extra node type exists in the output.

## Validation Examples

The implementation tests cover phase equality boundaries, exact known bubbles,
partition and fixed-stage invariants, swap placement, F/B-only counts, bounded
edits, combined-DAG and reversed-FIFO rejection, monotonicity, deterministic
iteration budgets, no phase fallback, range-cost boundaries, invalid inputs, and
noncyclic result-JSON round-trip validation. Both no-OR and OR builds run these
tests through the existing `slackpipe_core_tests` target.

Example common-evaluator results (B=4, N=4, W=2, L=8):

| Costs | Uniform-BFS initial | SlackPipe final | OctoPipe final | Final splits (SlackPipe; OctoPipe) |
| --- | ---: | ---: | ---: | --- |
| Unit F=B | 36 | 34 | 36 | `[1,2,3,2]`; `[2,2,2,2]` |
| Synthetic range-v2 fixture | 358 | 306 | 349 | `[1,2,4,1]`; `[2,1,3,2]` |

These are tiny correctness examples, not evidence of general relative performance.
All four exported plans have N=4 and exactly 32 distinct native F/B operations,
and load in the production v1/v2 parser. The heterogeneous OctoPipe run first
improves the partition, then remains in the non-improving partition phase, as
Algorithm 1 requires; it does not silently try scheduling instead.

Historical implementation validation in `slackpipe-dev` /
`slackpipe-ortools-runtime:local` (not a claim that the current full suite was rerun):

* No-OR CTest: 4/4 targets passed, including all 11 new C++ tests.
* OR-Tools 9.15.6755 CTest: 4/4 targets passed, followed by 14 validated canonical
  capped/uncapped CLI cases. Existing C++ tests were not modified.
* Production parser: both homogeneous v1 and heterogeneous v2 plans from both
  algorithms accepted; an explicit noncyclic mapping also accepted by the parser.
* Two-GPU PP=2 execution of each algorithm's exported homogeneous plan: exact
  initial parameters, loss, every gradient, and post-SGD parameters (maximum
  absolute differences all 0).
* Focused `test_slackpipe_plan.py`, `test_slackpipe_cost_profile.py`,
  `test_slackpipe_heterogeneous.py`, and `test_slackpipe_model_construction.py`:
  46 passed and 4 skipped per rank (two-rank launch).
* Full `test_slackpipe_*.py` run: 170 passed, 10 skipped, 2 failed per rank.
  The failures are `test_legacy_source_fix_boundary` and
  `test_legacy_across_receipt_only_source_fix`. They assume the live checkout
  differs from commit `5f000dcc` only by the audited receipt-only fix. The new
  solver changes correctly invalidate that assumption in
  `legacy_source_compatible`; its provenance guard was not weakened. This is a
  test-isolation issue at that checkpoint, not a green full Python suite.
  The current focused publication regression includes both tests with isolated
  source checks, and they pass; this does not retroactively change that older run.

Reproduce the OR build and complete C++ validation:

```bash
docker run --rm -v "$PWD:/workspace/Megatron-LM" -w /workspace/Megatron-LM \
  --entrypoint /bin/bash \
  -e BUILD_DIR=/workspace/Megatron-LM/slackpipe/build/release \
  -e BUILD_JOBS=2 -e SLACKPIPE_OR_TEST_FILTER='.*' \
  slackpipe-ortools-runtime:local slackpipe/scripts/validate_ortools_evaluation.sh
```

To run the full Python regression suite, beyond the focused publication checks:

```bash
docker exec -w /workspace/Megatron-LM \
  -e OMP_NUM_THREADS=1 -e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 \
  -e MAMBA_DETERMINISTIC=1 -e TRITON_CACHE_AUTOTUNING=0 \
  -e NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 -e SLACKPIPE_TEST_TRANSPORT=nccl-p2p \
  slackpipe-dev /opt/venv/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node 2 -m pytest \
  tests/unit_tests/pipeline_parallel/test_slackpipe_*.py \
  -q --tb=short --capture=fd --disable-warnings -o addopts=
```

For each exported plan's execution check, use the same command/environment,
add `-e SLACKPIPE_EXTERNAL_PP2_PLAN=/workspace/Megatron-LM/slackpipe/build/octopipe-homo.plan.json`
to `docker exec`, and select
`tests/unit_tests/pipeline_parallel/test_slackpipe_model_construction.py::test_slackpipe_pp2_numerical_equivalence`.
Repeat with `slackpipe-homo.plan.json`. The test compares against ordinary PP=1
Megatron and asserts all four maximum absolute differences are exactly zero.

All generated plans/results stay under ignored `slackpipe/build/`; no recorded
experiment results were modified.

## Plotting Evaluated Schedules

`slackpipe/scripts/plot_schedule_comparison.py` consumes the existing solver
`.csv` and `.plan.json` pair for each prefix, not profiler traces or a new
simulator. It checks operation coverage, worker serialization, exported order,
placement, and makespan before plotting actual intervals on a common tick axis.
Stage colors are shared between panels; hatching denotes complete backward;
labels are direction plus microbatch (omitted on blocks too narrow to label).

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev /opt/venv/bin/python \
  slackpipe/scripts/plot_schedule_comparison.py \
  --left-prefix slackpipe/build/comparison_b8_n8_w4_l64_r2_30s/slackpipe \
  --right-prefix slackpipe/build/comparison_b8_n8_w4_l64_r2_30s/octopipe \
  --output-dir slackpipe/build/comparison_b8_n8_w4_l64_r2_30s

docker exec -w /workspace/Megatron-LM slackpipe-dev /opt/venv/bin/python \
  slackpipe/tests/plot_schedule_comparison_test.py
```

Outputs are `slackpipe_gantt.png`, `octopipe_gantt.png`, `comparison_gantt.png`,
and `comparison_metrics.json`, including per-worker bubble/busy metrics and
input checksums. The comparison run directory preserves its exact `run.sh`,
stdout/stderr, source provenance, C++ replay/metric cross-check adapter, and
`verification.json`. No timestamps are averaged or rescaled. Wall-clock-limited
search can return different incumbents on rerun; the saved artifacts reproduce
these particular figures exactly.

## Diagnostic-Only Forced Phases

`octopipe_phase_diagnostic` is a separate executable, not an algorithm option or
fallback in `slackpipe_cli`. It evaluates the uniform-BFS initializer and clones
that exact state independently for **one invocation per phase**. It never enters
`TuneOctoPipeAlgorithm1` or chains phase results. `DiagnoseOctoPipePhase` calls the
production `OctoPipeNeighborProposals`, `EvaluateOctoPipeCandidate` (including the
common independent validator and predecessor replay), and the exact production
`Better` tie-breaker. No candidate generation or performance model is duplicated.

Placement is required to be enabled. Scheduling uses the production default
64-proposal bound and four-slot insertion window; partition and placement use
their existing complete one-layer-transfer and whole-stage-swap neighborhoods.
There is no wall-clock cutoff because this diagnostic completes just one bounded
invocation, not a convergence search. This is not a stronger neighborhood than
production; it simply bypasses phase selection to inspect each neighborhood once.

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev \
  cmake -S slackpipe -B slackpipe/build/no-or -G Ninja -DSLACKPIPE_ENABLE_ORTOOLS=OFF
docker exec -w /workspace/Megatron-LM slackpipe-dev \
  cmake --build slackpipe/build/no-or --target octopipe_phase_diagnostic slackpipe_core_tests -j2
docker exec -w /workspace/Megatron-LM slackpipe-dev \
  slackpipe/build/no-or/octopipe_phase_diagnostic \
  --B 8 --N 8 --J 4 --L 64 --min-layers 1 \
  --ratio-num 2 --ratio-den 1 --communication 0 \
  --octopipe-candidates-per-iteration 64 \
  --output-prefix slackpipe/build/forced_phases_b8_n8_w4_l64_r2
```

Output is structured JSON on stdout and at `<output-prefix>.json`, not a new
execution-plan format. It records generated/valid/improving counts, the best
valid state's split, placement and native operation IDs, predecessor/order-change
flags, and both candidate and accepted-result improvements. A `null` best valid
state means none passed validation; a `null` best improving makespan means no
strict improvement, even if a valid candidate exists. Worse valid candidates
have negative `best_valid_improvement_ticks/percent`; the resulting diagnostic
makespan still stays at the initializer. Invalid proposals are counted but never
reported as valid states.

The requested B=8/N=8/W=4/L=64, F=1/B=2/communication=0 case gives these measured
evaluator results, with every phase starting from the same 456-tick state:

| Phase | Generated | Valid | Improving | Best valid | Best improving | Resulting makespan |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Partition | 14 | 14 | 1 | 447 | 447 | 447 |
| Fixed-stage placement | 24 | 24 | 0 | 472 | none | 456 |
| F/B scheduling | 64 | 13 | 0 | 456 | none | 456 |

* Partition: best split `[8,8,8,7,9,8,8,8]`, cyclic placement unchanged,
  predecessor/order unchanged. Gain: 9 ticks (1.97368%).
* Placement: best valid split remains `[8,8,8,8,8,8,8,8]`; swapping stages 5 and 7
  gives placement `[0,1,2,3,0,3,2,1]`. Worker order and predecessors change. This
  candidate is 16 ticks worse (-3.50877% improvement) and is **not accepted**.
* Scheduling: best valid candidate retains uniform split/cyclic placement, but
  advances `B4(b0)` before `F4(b6)` on worker 0. Its 456-tick tie is **not accepted**.
  The other 51 generated scheduling proposals fail common validation.

The production metric functions give delta_b=0, t_layer=3, boundary=144,
residual=144; the production selector returns `schedule-fb-only` because both
strict comparisons are false. Thus this is **Case A, a phase-selection limitation
at this initializer**: there is a one-step partition improvement that the selected
scheduling phase cannot reach. This does not prove that any alternative selector
would solve the broader optimization problem, and the production selector was
not changed. No forced-phase convergence experiment was run.

Focused `OctoPipeDiagnostic.*` C++ tests verify independent/immutable initial
states, repeatable best states, all valid candidates' common-DAG/FIFO validity,
N=8 and 128 unique F/B operations, the observed phase results, and exact agreement
between the selected forced phase and one normal production iteration. Existing
normal tuner function bodies and CLI are unchanged; only the diagnostic helper
was appended to the tuner source.

## Initialization Sensitivity Diagnostic

The diagnostic executable also accepts `--initialization`, resolved through the
existing evaluation-method registry. Its default remains `uniform-breadth-first`.
This selects only an existing initializer, never a new scheduling implementation.
The JSON adds the generator name, per-worker busy/boundary/residual/total-bubble
arrays, equality checks, and analysis-only minimum-operation threshold and phase.
Production `t_layer` remains the minimum per-layer forward plus complete-backward
cost. Busy time is summed from evaluator intervals and checked against makespan
minus the production bubble metric for each worker.

### Generator Audit

Repository-wide searches covered C++ generators, Megatron runtime/integration,
Python experiment helpers, tests, and the legacy `slackpipe/eval_checkouts` copy.

* BF-uniform uses `BreadthFirstOrders` in `slackpipe/src/breadth_first.cc`.
* The C++ interleaved baseline uses `InterleavedOneFOneBOrders` in
  `slackpipe/src/one_f_one_b.cc`: stage-local warmup/alternation/cooldown streams,
  merged by its existing ready-operation priority. No generator was rewritten.
* `uniform-1f1b` and `interleaved-1f1b` are registry aliases for
  `uniform-interleaved-1f1b`. Both CLI commands were actually executed. Worker-local
  orders, their predecessor edges, all evaluated operation intervals, and makespan
  agree exactly: **TRUE ALIAS**. Only one row counts toward the experiment.
* Megatron's `forward_backward_pipelining_without_interleaving` in
  `megatron/core/pipeline_parallel/schedules.py` is a distinct standard runtime,
  but explicitly rejects multiple model chunks per rank. The experiment helper
  `tools/slackpipe_eval_config.py::schedule_topology` requires `N=PP` for it.
  Multi-module tests do not supply a cyclic two-chunks-per-worker order generator.
  CUDA-graph schedule-table conversion and combined-1F1B helpers implement
  interleaved or intra-model execution, not a compatible non-interleaved generator.
* The legacy C++ checkout has the same interleaved generator, not a third one.

Consequently there are **two unique compatible initializations**, not three.
Using native non-interleaved Megatron at PP=4 would change N to 4; projecting eight
independent pipeline ranks onto four workers would introduce a new merge policy.
Neither was done. The C++ interleaved baseline is its repository-defined order,
not a claim of identical ordering or performance to Megatron's VPP runtime.

### Reproduction

After building the diagnostic as above, run these commands inside `slackpipe-dev`
at `/workspace/Megatron-LM` (for example using `docker exec ... bash -lc`):

```bash
out=slackpipe/build/initialization_diagnostic_b8_n8_w4_l64
mkdir -p "$out"
for method in uniform-breadth-first uniform-1f1b interleaved-1f1b; do
  name=$method
  if [ "$method" = uniform-breadth-first ]; then name=bf; fi
  slackpipe/build/no-or/slackpipe_cli --algorithm "$method" \
    --B 8 --N 8 --J 4 --L 64 --ratio-num 2 --ratio-den 1 --communication 0 \
    --output-prefix "$out/$name" --emit-plan "$out/$name.plan.json"
done
for name in bf interleaved; do
  method=uniform-breadth-first
  if [ "$name" = interleaved ]; then method=uniform-interleaved-1f1b; fi
  slackpipe/build/no-or/octopipe_phase_diagnostic --initialization "$method" \
    --B 8 --N 8 --J 4 --L 64 --ratio-num 2 --ratio-den 1 --communication 0 \
    --octopipe-candidates-per-iteration 64 --output-prefix "$out/$name.diagnostic"
done
/opt/venv/bin/python slackpipe/scripts/plot_schedule_comparison.py \
  --left-prefix "$out/bf" --right-prefix "$out/interleaved-1f1b" \
  --left-label BF-uniform --right-label C++-InterleavedOneFOneBOrders \
  --output-dir "$out"
```

Both initial states have split `[8,8,8,8,8,8,8,8]`, placement
`[0,1,2,3,0,1,2,3]`, B=8, N=8, W=4, L=64, F=1, complete B=2, communication=0.
All numbers below are abstract evaluator ticks, not GPU measurements.

| Initialization | Makespan | delta_b | Boundary | Residual | t_layer | Selected phase | Forced partition best |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| BF-uniform | 456 | 0 | 144 | 144 | 3 | scheduling | 447 |
| C++ interleaved 1F1B | 840 | 0 | 144 | 1680 | 3 | scheduling | 837 |

Per-worker arrays below are ordered W0, W1, W2, W3:

| Initialization | Busy | Boundary | Residual | Total bubble |
| --- | --- | --- | --- | --- |
| BF-uniform | [384,384,384,384] | [0,24,48,72] | [72,48,24,0] | [72,72,72,72] |
| C++ interleaved 1F1B | [384,384,384,384] | [0,24,48,72] | [456,432,408,384] | [456,456,456,456] |

For each initializer, `0 > 3` is false and analysis-only `0 > 1` is false.
The second test is respectively `144 > 144` and `144 > 1680`, both false.
Both threshold interpretations therefore select scheduling. The analysis-only
threshold never enters the production tuner.

| Initialization | Phase | Generated | Valid | Improving | Best valid | Best improving | Best-valid gain ticks (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF-uniform | Partition | 14 | 14 | 1 | 447 | 447 | 9 (1.97368%) |
| BF-uniform | Placement | 24 | 24 | 0 | 472 | none | -16 (-3.50877%) |
| BF-uniform | Scheduling | 64 | 13 | 0 | 456 | none | 0 (0%) |
| C++ interleaved | Partition | 14 | 14 | 3 | 837 | 837 | 3 (0.357143%) |
| C++ interleaved | Placement | 24 | 24 | 16 | 728 | 728 | 112 (13.3333%) |
| C++ interleaved | Scheduling | 64 | 21 | 6 | 832 | 832 | 8 (0.952381%) |

All forced invocations start from independent copies; none consumes another's
result. BF placement/scheduling have no improving candidate and leave the
diagnostic state at 456, notwithstanding the worse 472-tick placement neighbor.

Best states for BF-uniform are documented above. For C++ interleaved:

* Partition: split `[8,8,9,7,8,8,8,8]`, original cyclic placement, unchanged
  worker order/predecessors, makespan 837.
* Placement: original uniform split, mapping `[0,0,2,3,1,1,2,3]` (swap stages 1
  and 4), worker orders/predecessors changed, makespan 728.
* Scheduling: original split/mapping; on W3 advance `F7(b7)` before `B3(b6)`
  (native IDs 119 and 108); predecessors change, makespan 832. A normal production
  Algorithm-1 iteration accepts this exact candidate, so this is not an
  acceptance bug.

### Interpretation and Artifacts

`delta_b=0` persists because each worker owns 16 layers and executes
`8 * 16 * (1+2) = 384` ticks of compute. Within each run the common makespan T
gives `total_bubble[w] = T - 384` on every worker, independently of order.
Changing order changes T and hence total bubble magnitude, not its cross-worker
imbalance. It does **not** change only the boundary/residual decomposition:
boundary is unchanged here, residual increases, and total bubbles also increase.

The first phase is insensitive to these two initializations and to thresholds 3
versus 1. Both miss an available partition improvement. However, the selected
scheduling neighborhood has no improving BF neighbor but six improving
interleaved neighbors. Thus the first decision is the same while its outcome is
initialization-sensitive. This does not generalize to arbitrary initial orders.

The ignored output directory retains both CLI alias runs, two diagnostic JSONs,
`alias_verification.json` (actual-state comparisons, predecessor edges and input
checksums), and shared-scale plots. Existing plotting filenames are
`slackpipe_gantt.png` (BF), `octopipe_gantt.png` (C++ interleaved), and
`comparison_gantt.png`; titles identify the initializers. `comparison_metrics.json`
is independently computed from raw evaluator CSV intervals. No alias-only chart
is generated; no timestamps are averaged or rescaled.

Two additional C++ tests cover distinct initial orders, registry aliases,
per-worker metrics, both thresholds, all valid neighbors' common validation,
immutable interleaved initialization, and exact normal-iteration acceptance.
No production source, threshold, bubble definition, neighborhood, or algorithm
was changed for this experiment. No convergence search was performed.

Validation completed in Docker: no-OR and OR-enabled CTest each passed all four
targets; all 18 `OctoPipe*` tests passed in both builds; the OR validation script
validated 14 canonical CLI result files; all five plotting tests passed. The two
initializers' diagnostic JSONs agree across builds, and CSV-derived metrics agree
with every reported per-worker metric. Production tuner, CLI, BFS/1F1B generators,
and method-registry source hashes are unchanged from the start of this task.
