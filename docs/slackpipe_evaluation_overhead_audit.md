> Historical scope: this document records pre-five-method code/results. Commands and method IDs below are not current entry points. Use the [current workflow](slackpipe_five_methods.md); historical results keep their recorded commit and labels.

# Real-system evaluation overhead audit

Audit date: 2026-09-30. Branch: `slackpipe/mvp`, base commit
`a6f2d985e07d87ba22a27c49cb0610555a753927`, with pre-existing uncommitted
OctoPipe/five-method work. This audit changes orchestration, not scheduling.
The original audit did not request a commit or push; publication of the
five-method integration was requested subsequently. Existing experiment data
was not overwritten.

Inspected orchestration: `tools/run_slackpipe_real_system_campaign.py`,
`run_slackpipe_eval.py`, `slackpipe_eval_config.py`, `slackpipe_eval_receipts.py`,
`slackpipe_eval_tables.py`, `run_slackpipe_nemotron_h8b_pp4.py`, and the older
`slackpipe_scale_sweep.sh`. Inspected execution/profiling:
`tools/slackpipe_eval_worker.py`, `slackpipe_nemotron_worker.py`,
`slackpipe_profile_quality.py`, and core SlackPipe `collection.py`,
`profile_quality.py`, `schedule.py`, `communication.py`, `communication_rma.py`.
Inspected plotting: `tools/plot_schedule_trace.py` and its compact-trace
validation path. Relevant existing tests are listed below.

Changed in this audit: the campaign and Experiment orchestration modules,
`test_slackpipe_eval_receipts.py`, new `test_slackpipe_campaign_plot.py`, and this
report. Other worktree changes predate this audit and were preserved.

## Execution map and scope

The current entry point is `tools/run_slackpipe_real_system_campaign.py`, not
the older scale-sweep wrapper. The campaign runs inside an already provisioned
container; it does not build binaries or start Docker for individual methods.

1. **Campaign:** parse/canonicalize methods, expand family/size configurations,
   validate them, choose seeded rotation across independent repetitions.
2. **Model/config:** establish output identity and write execution order.
   Refined SlackPipe has effective `N = base_N + PP`; it is a different topology.
3. **Method/repetition:** construct `Experiment`, derive stage contexts, assess
   receipts recursively before launching GPU work. Context binds model, effective
   topology, precision, sequence/batch sizes, transport, source/environment,
   stage policy and parent artifacts. Artifact checksums remain mandatory.
4. **Prerequisites:** `env -> correctness -> smoke` for native methods;
   `env -> correctness -> native-smoke -> calibrate -> solve -> smoke` for
   plan-driven methods. Correctness uses its own torchrun numerical test.
   Native 1F1B/interleaved never require a solver or cost profile for benchmarking.
5. **Calibration attempt:** fresh torchrun, native Megatron execution, one model
   per calibration partition, full-step warmup, stage-wall measurements, raw
   events with group IDs, fit, quality diagnostics and profile publication.
   Heterogeneous partitions/classes remain distinct. At most one complete retry
   is launched in a new process; neither attempt is deleted. A second failure
   blocks solving. An unrelated collection failure is not a quality retry.
6. **Solve:** validate accepted profile, invoke existing C++ algorithm, export
   evaluator-validated plan; parse/check dimensions and provenance, including
   the v1 binding sidecar. OctoPipe and SlackPipe retain different algorithms.
7. **Each GPU launch/rank:** validate inputs, initialize distributed state,
   construct chunks/parameters/SGD/resident mock batches, select the scheduler.
   Independent repetitions and traces construct fresh models/processes.
8. **Warmup/iteration:** the same `iteration()` performs zero_grad, iterator
   creation/consumption, forward/backward, communication and optimizer.step.
   Benchmark uses CUDA events and a synchronized continuous window, without a
   profiler. No plan JSON parsing, transport construction, result serialization
   or per-step Python printing occurs in the steady-state benchmark loop.
9. **Run completion:** finite checks, per-rank results and trace compaction,
   collective transport shutdown, cleanup. Aggregation and receipt hashing are
   outside the measured window. Timeline exports occur at profiler cycle
   boundaries, outside step events but inside the diagnostic continuous window.
10. **Campaign completion:** trace each successful method once; select the
    earliest common complete-rank iteration in cycle 0; render; export tables.

Native and SlackPipe runtime validation are intentionally separate trust
boundaries. `_get_slackpipe_runtime` checks path/stat/distributed context each
iteration, but returns the cached compiled plan/communicator before JSON loading.
P2P groups and compatible RMA mailboxes persist within the run. Teardown occurs
after the partition/run, not between warmup and measurement. RMA's put drains,
receive-stream joins and slot-reuse synchronization are correctness operations.

## Findings

| Finding | Audited behavior | Change or decision | Impact | Risk |
|---|---|---|---|---|
| Fresh campaign prerequisites | Cross-method reuse existed only with `--resume`; OctoPipe/SlackPipe recalibrated identical inputs | Share only receipt paths accepted earlier in this invocation, with unchanged context/hash validation | HIGH: one calibration, one native smoke, two correctness launches avoided in the default five-method case | Safe; no implicit adoption of old campaigns |
| Refined profile | Effective N differs; v1 stage-role fit/topology differs | Keep separate calibration and solver | HIGH potential cost, requires caution | Unsafe to blindly reuse base-N costs |
| Duplicate config expansion | Repeated family/size, symlinks, copies or output collisions detected late | Reject before any GPU work; normalize CSV whitespace; retain alias collision rejection | HIGH when duplication occurs | Safe; explicit error, no silent merging |
| Completed figures | Trace JSON scanned before checking existing destination | Check PNG, PDF and final report first | MEDIUM for large traces | Safe after trace receipts are validated |
| Interrupted figures | Any existing destination directory suppressed regeneration | Require all three nonempty output files; regenerate incomplete derived figures | MEDIUM correctness/recovery | Safe; raw captures untouched |
| Repeated source/environment/hash work | New Experiment per repetition; recursive artifact verification | Leave intact | LOW for tiny runs, potentially MEDIUM for large artifacts | Requires caution; caching could hide changes/corruption |
| Model/config parsing | Small JSON loaded by normalization, Experiment and worker | Reuse preflight model only in campaign loop; leave process-boundary validation | LOW | Further abstraction not worth it |
| Plan validation | Export, driver, worker and runtime each validate | Leave trust boundaries; runtime already caches compiled operations | LOW (parser measured below) | Removing guards requires caution |
| Model/process setup | Fresh model and torchrun per independent run | Leave intact | HIGH absolute overhead on tiny models | Reuse risks state/timing contamination |
| Warmup | One full-step warmup per launch/partition; profiler warmup additionally intentional | Leave all counts/policies unchanged | MEDIUM | Requires caution; not duplicate training warmup |
| Synchronization | Calibration stage timing and runtime communication have necessary waits; benchmark window has boundary synchronization | Leave intact | Potentially HIGH | Requires correctness evidence before changing |
| Docker/builds | No Docker or compilation in generic campaign loop | No change | LOW/no redundancy | Older explicit scale-sweep wrapper has per-job Docker, not this path |
| Logging/artifacts | Captured subprocess logs, retained failed profiling attempts, no hot-loop JSON/prints | Leave intact | LOW outside GPU window | Diagnostics/provenance outweigh savings |
| Empty event gathering | Non-calibration run still gathers empty calibration event lists at completion | Leave intact | LOW, once per run | Not worth touching collective lifecycle |

Equivalent configurations with deliberately different metadata/output identities
are not automatically merged: semantic equivalence must not be inferred by
dropping arbitrary model fields. Checks cover normalized full model identity,
resolved config path, and resolved output identity.

For one successful fresh five-method configuration, three repetitions, distinct
physical/base/refined topologies, and no quality retry:

| Launch/task | Before | After |
|---|---:|---:|
| Correctness torchrun | 5 | 3 |
| Native calibration smoke torchrun | 3 | 2 |
| Calibration torchrun | 3 | 2 |
| Per-method execution smoke torchrun | 5 | 5 |
| Independent benchmark torchrun | 15 | 15 |
| Timeline torchrun | 5 | 5 |
| Total torchrun | 36 | 32 |
| C++ solver subprocess | 3 | 3 |

Each torchrun creates PP rank processes in addition to its launcher. There are
zero nested Docker calls; normally one external `docker exec` launches the
campaign. Each failed quality attempt adds at most one calibration torchrun
per distinct profile. Counts were checked with the real campaign/receipt code
and synthetic workers, not an invented replacement control flow. Force/resume
can intentionally change these counts. Existing receipt schema is unchanged.

## Small real-GPU measurements

New ignored output: `slackpipe_experiments/orchestration_audit_20260930/`.
Two RTX A2000 GPUs; FP32, PP=2, base N=4, L=8, B=4, hidden=128, sequence=32,
microbatch size=1, untied output, SGD; native interleaved and NCCL-P2P SlackPipe.
Warmup=20 complete steps; benchmark=8 steps. Timeline uses
wait=2/warmup=2/active=2/repeat=2 after training warmup. These are overhead
diagnostics, not statistically powered method-performance comparisons.

| Phase | Observation | Boundary |
|---|---|---|
| Profiling | 21.972 s, accepted attempt 0 | Complete torchrun, 20 warmups + 10 calibration iterations, quality gate |
| C++ solver | 0.14 s process wall; lifecycle CLI 0.057 s | Existing OR-enabled binary, 5 s budget, includes export; excludes Docker startup |
| Plan parsing/validation | 0.487 ms first call; 0.158 ms mean of 100 repeats | Python imports excluded; parser only, no receipt hashing |
| Native benchmark launch | 20.971 s | Complete subprocess lifetime |
| Native training warmup | 2.164 s | Maximum rank sum of 20 CUDA event spans, not exact wall time |
| Native measured window | 0.708 s | Synchronized continuous 8-step window |
| Native setup/other residual | Approximately 18.100 s | Total minus above spans: imports, startup, construction, validation, gaps, IO and teardown; NOT isolated startup latency |
| SlackPipe benchmark launch | 22.024 s | Complete subprocess lifetime |
| SlackPipe warmup / measured window | 1.958 s / 0.672 s | Same definitions as above |
| SlackPipe setup/other residual | Approximately 19.395 s | Same nonexclusive estimate, not a pure startup measurement |
| Native / SlackPipe trace launches | 23.278 s / 23.979 s | Includes profiling, serialization and compaction, not benchmark samples |
| Rank aggregation | 3.45 ms native / 0.46 ms SlackPipe | Read results + benchmark_summary; first-call import/cache effects apply |
| Initial two-method plot | 1.405 s | Parse/validate compact traces and write PNG/PDF/report |
| Old completed-figure scan / new resume | 182.705 ms / 0.163 ms | Same saved captures; old scan reproduced offline, new real early return |
| Synthetic campaign orchestration | 0.223 s before / 0.248 s after | Actual campaign IO with mocked workers; excludes real GPU/solver work |

The optimization saves launches, not Python loop time. Avoiding a duplicate
calibration would save roughly its observed 22 s on this tiny case, in addition
to the avoided smoke/correctness work; a full before/after GPU campaign timing
was NOT performed. No throughput improvement is claimed. Solver exported
split `[1,2,3,2]`; predicted makespan is not substituted for measured GPU time.
Existing prebuilt solver was used, not rebuilt or optimized by this task.

Raw commands, elapsed subprocess times and return codes are in
`phase_measurements.json` and `slackpipe_phase_measurements.json`; all rank
results/logs, quality attempts, accepted profile, plan/binding, raw/compact
captures and derived figures are retained. `analysis.json` and the two
`*_synthetic.json` files contain the reported aggregation/count evidence.
`audit_provenance.json` records artifact and solver-binary SHA-256 hashes.
Metadata records source identity; orchestration edits occurred between native
and SlackPipe diagnostic launches, so these runs are not a controlled speedup
experiment. The training/runtime code was unchanged.

## Validation

Run inside the existing container (no host packages installed):

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev /opt/venv/bin/python -m pytest \
  tests/unit_tests/pipeline_parallel/test_slackpipe_eval.py \
  tests/unit_tests/pipeline_parallel/test_slackpipe_eval_receipts.py \
  tests/unit_tests/pipeline_parallel/test_slackpipe_campaign_plot.py \
  tests/unit_tests/pipeline_parallel/test_slackpipe_collection.py \
  tests/unit_tests/pipeline_parallel/test_slackpipe_figure_trace.py \
  tests/unit_tests/pipeline_parallel/test_slackpipe_plan.py \
  tests/unit_tests/pipeline_parallel/test_slackpipe_cost_profile.py \
  tests/unit_tests/pipeline_parallel/test_slackpipe_profile_quality.py \
  tests/unit_tests/pipeline_parallel/test_slackpipe_lifecycle.py \
  -q --capture=fd -r f --disable-warnings
```

Result: **164 passed**, no skips. Coverage includes five-method fresh and resume
flows, topology/source/policy invalidation, checksum corruption, refusing old
unlisted receipts, alias/config duplication, partial plot repair, profiling
retry exhaustion and solver blocking, trace correlation/interval unions and
runtime cache lifecycle. Formatting and `git diff --check` passed.

Real capture checks: both methods preserve two cycles x two ranks; global
iterations are `[24,25]` and `[30,31]`, with four raw and four compact files per
method. Real plots validate rank/capture alignment and select iteration 24.
No synthetic averaged/rescaled timeline is used. Raw rank timings feed the
unchanged summary aggregator; all eight samples are retained.

The solver-produced plan also passed PP=2 numerical equivalence: one test
passed on each rank; initial parameters, loss, every gradient and post-SGD
parameters all have maximum absolute difference **0**. Executed operation
sequences match the exported plan. Evidence is in
`equivalence-traces/equivalence.json` and the rank-local operation traces.

```bash
docker exec -w /workspace/Megatron-LM \
  -e CUDA_VISIBLE_DEVICES=0,1 -e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 \
  -e SLACKPIPE_EXTERNAL_PP2_PLAN=/workspace/Megatron-LM/slackpipe_experiments/orchestration_audit_20260930/solver.plan.json \
  -e SLACKPIPE_EXTERNAL_TRACE_DIR=/workspace/Megatron-LM/slackpipe_experiments/orchestration_audit_20260930/equivalence-traces \
  -e SLACKPIPE_TEST_TRANSPORT=nccl-p2p slackpipe-dev \
  /opt/venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=2 \
  -m pytest tests/unit_tests/pipeline_parallel/test_slackpipe_model_construction.py::test_slackpipe_pp2_numerical_equivalence \
  -q --capture=fd -r f --disable-warnings
```

## Unchanged semantics and limits

- No changes to scheduling algorithms, optimization budgets, cost fitting,
  quality thresholds, retry limits, sample selection or result schemas.
- No profiler in benchmark mode; no new synchronization/timers in GPU steps;
  no model/optimizer/communicator reuse across independent measurements.
- CUDA events still cover the documented supported stream-join contract;
  the continuous metric still includes dispatch/bookkeeping and end barriers.
- Warmup defaults remain five in this driver; this audit explicitly used twenty.
  No warmup was removed, no allocator policy changed, and no slow samples dropped.
- PP=4/full-size five-method GPU campaigns and real RMA hardware lifecycle were
  not rerun. RMA evidence here is source inspection plus the cache unit test.
  Two-method small GPU execution and five-method mocked orchestration are
  distinct evidence. Earlier Nemotron data/prediction accuracy was not revalidated.
