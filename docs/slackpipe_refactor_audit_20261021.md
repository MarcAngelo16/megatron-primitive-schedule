# Refactor Audit and Publication Checks

The original review and limited GPU closeout below were performed without
committing or publishing. Their results are historical and are preserved.
See **Publication Checks** for the subsequent user-authorized cleanup commit
preparation; that section does not retroactively mark GPU acceptance as passed.

Review date: 2026-10-21 (Asia/Tokyo; container logs use UTC).
No commit, push, PR, history rewrite, dependency installation or experiment-data
rewrite was performed.

The historical review below is unchanged. See **Limited GPU-Preflight Closeout**
at the end for the subsequent narrow fix and separately counted test batches.

## Scope and Environment

- Checkout: `/home/rui/Megatron-LM`, branch `slackpipe/mvp`.
- HEAD: `322fdf68cfcc146ec23d44b9ed2dd6ffdfc7bcaf`.
- Index empty at entry. The starting tracked diff contained 121 paths,
  3,322 insertions and 44,167 deletions; this excludes new untracked source.
  It includes the earlier method pruning and profiling-policy refactor, not
  just this review's edits. Relevant new sources were read directly.
- `slackpipe/` is part of the outer repository. Personal remote is
  `git@github.com:RioReal/megatron-primitive-schedule.git`; NVIDIA remains origin.
  Neither was contacted for publication.
- Existing `slackpipe-dev`, image `megatron-slackpipe:0.18.2`, bind mount
  `/home/rui/Megatron-LM` -> `/workspace/Megatron-LM`, `/opt/venv/bin/python`.
- PyTorch `2.12.0a0+0291f960b6.nv26.04.48445190`, CUDA toolkit 13.2,
  Transformer Engine `2.16.0+b9d690e0`, Mamba `2.3.2.post1`.
- CUDA available=False, device_count=0. Container `nvidia-smi` exits 255:
  `Failed to initialize NVML: Unknown Error`. This establishes lack of usable
  container GPU access, **not** its driver/runtime root cause.
- Relevant testing, container/dependency and formatting skills were read.
  Generated review outputs are under ignored
  `slackpipe/build/refactor-audit-20261021/`. `source-before.tar.gz` snapshots
  the initial versions of the principal edited modules/tests, not the entire
  already-dirty tree. Historical archives, datasets and results were untouched.

## Findings and Fixes

### 1. Stage capability rejection was too late

The launcher previously entered `ensure()` prerequisites before checking whether
new stage collection could run. Rejection existed only in `run_worker()`.
Calibration also derived its schedule from N==PP, choosing unsupported ordinary
1F1B in that case. A campaign could execute other methods before encountering
this predictable failure.

Fixed in `tools/slackpipe_profiling.py`, `slackpipe_eval_args.py`,
`run_slackpipe_eval.py`, `run_slackpipe_real_system_campaign.py`, and the shared
parser call in `slackpipe_eval_worker.py`:

- Check new stage collection before prerequisite workers, including correctness,
  and preflight every requested campaign configuration/method before the first
  method starts.
- Require supported native interleaved topology: PP=2/4, N>PP, N%PP=0,
  VPP>=2, B%PP=0. No values are corrected silently.
- Record calibration schedule separately from experiment method: stage uses
  `interleaved-1f1b`, layer uses `isolated-unit`, explicit imports are labelled
  `artifact-import`. Solver selection remains a separate field.
- Existing artifacts and execution-only workers do not inherit restrictions on
  *new* collection. Read-only inspection is also not a collection.
- Native 1F1B and native interleaved benchmarks remain independent of profile
  and solver prerequisites, including with the default `both`.
- A known unsupported stage/both configuration fails up front even if layer is
  selected. Runtime failures still produce the existing partial outcome:
  failed optional backend is recorded, failed selected backend stops, no retry
  or fallback.

Evidence: configuration/API tests for PP=1, N=PP and nondivisible B; campaign
preflight test; real CLI rejection before an output directory is created;
native benchmark dependency tests with default both; sharing/receipt tests for
all three optimized methods; explicit import and single-attempt failure matrix.
The CLI help and `slackpipe_profiling.md` now describe the same rules.

### 2. Declared stage strategy was not verified on load

The previous integrity loader checked schema, costs, units, prefixes and a
present hash, but did not check that a stage artifact's declared current policy
matched its raw measurements, initial predictions, masks and final model.
Rehashing a changed mask or copying a strategy label onto unrelated summaries
could pass. Layer already reconstructed its current median policy.

`profile_integrity.py` now reconstructs declared stage policies from their
preserved inputs and compares policy, raw/final observations, initial/final
models, candidate/applied masks and emitted costs. This checks integrity; it
does not filter the retained observations a second time. Floating model values
allow last-bit numerical-library differences (1e-10 relative/absolute), while
booleans, identities, indices and masks must match. Backend labels must agree
with isolated evidence. Legacy artifacts without a current-policy declaration
keep their identity and are not rejected solely for a failed quality flag.

This exposed mock profiles that merely attached a new strategy label and a test
that relabelled an affine v1 fit as v2. The fixtures now use the real respective
builders; the loader was not weakened to accept the mocks. Receipt-level group
tests now use a supported four-stage calibration topology.

Evidence: rehashed mask/prediction/policy/cost/raw corruption rejected;
stage/layer publication and C++ export/parser integration; high-variance and
shifted-group fixtures accepted without statistical approval; historical bytes
and policy retained; malformed optional backend still reports partial failure.

### 3. Residual bookkeeping and raw identity checks

Core single-pass behavior was already implemented correctly:

1. Inventory validation, then initial median of per-iteration means using all
   legal operation measurements.
2. Homogeneous shared-slope/stage-bias fit or heterogeneous class/role fit.
3. Per **raw operation**, separately for F/B:
   `q=(y-p)/p`, `m=median(q)`, `s=1.4826*median(abs(q-m))`.
4. Exactly one candidate mask:
   `abs(q-m)>k*s AND abs(q)>rho`, defaults k=6, rho=0.5.
5. Recompute means using retained counts, then at most one refit. Empty iteration
   means are null. A required empty observation or explicit structural fit
   failure rolls the entire direction back to all samples and its initial fit.
   Only `FitStructureError` is caught; unexpected errors propagate.

Two small correctness fixes: final `backward_forward_ratio` now reflects final
F/B summaries rather than stale raw summaries, and refit prediction inventory
is checked. Heterogeneous raw input now also rejects wrong phase/stage,
boolean indices and duplicate sample IDs. Layer timings reject nonnumeric/
boolean values and boolean indices that Python could otherwise equate with
integer sample IDs. These are structural checks, not stability thresholds.

Added checks exercise zero aggregate residual with a raw spike, strict rho and
scaled-MAD equality, zero MAD, empty iteration, real heterogeneous required-row
rollback, restoration of the complete direction, independent successful F,
unexpected refit errors, operation/B/unit scaling and raw corruption.
No estimator, extra intercept, iterative mask or new filtering algorithm added.

### 4. No-hint implementation retained; deterministic failure tests added

Reviewed actual path:

```text
eval registry / solver_arguments
 -> Experiment.ensure("solve") / fresh immutable output directory
 -> slackpipe_cli option whitelist and method dispatch
 -> JointOptimizerOptions
 -> OptimizeJointSplitAndScheduleCpSat
 -> CP model / SolveCpModel / extraction
 -> EvaluateSchedule / independent validator / ToJson / plan export
```

The options/API accept no plan, external incumbent, checkpoint or hint.
Only the separate OctoPipe branch builds BFS initialization; only the separate
heuristic branch calls its search. CP-SAT model construction has no `AddHint`;
every solve checks `solution_hint().vars_size()==0`. Model horizon comes from
problem dimensions/duration limits and communication, not an incumbent.
Worker-load/global-capacity expressions sum actual per-operation duration
variables in both profile branches. FIFO, dependency, serial-resource and
activation constraints remain. Internal feasible-solution callbacks are legal
CP-SAT incumbents, not external seeds. Activation replay rejection uses the
existing exact no-good, not a heuristic fallback.

No-solution returns no schedule; CLI exports no plan; Python records the
no-solution receipt and blocks execution. Existing requested output plans are
refused, never loaded as seeds. Compatible reuse of a prior **completed solve
receipt** is explicitly recorded reuse, not a new CP-SAT search or seed injection;
source/method/binary/profile/byte identity must match. Historical hinted-method
receipts cannot be relabelled as this method.

Replaced nanosecond-timeout tests with deterministic infeasible zero activation
capacity in both API and real CLI tests. Also test rejected injection/other-method
flags and refusal to consume/overwrite an existing plan. Existing real OR tests
check optimum=18, structural horizon=24 (not the feasible objective), zero hints,
workload/FIFO constraint counts, profile branches, and independent enumeration.
Solver/search/model implementation itself was not changed in this review.

### 5. Stability gates and consumers

Reviewed stage and layer producers, aggregator/fit, publication, explicit import,
receipt/cached-artifact checks, `validate_profile`, standalone integrity/fit/
iteration-inspection tools, diagnostic stage-fit exports, Python solver preflight,
C++ profile parser and evaluator. Searched their consumers for removed quality,
consensus, spike-filter, retry and `allow_unstable` symbols.

No active statistical admission decision found: no CV/drift/consensus selection,
exclusion-rate cap, RMSE/R2/condition cutoff or old `quality.status` gate.
Diagnostic errors/condition estimates are descriptive. Class-fit rank and positive
predictions remain constructibility checks. Layer uses every legal sample's median.
Schema/manifest/config/hash/prefix/raw inventory and independent replay checks
remain. The C++ parser still consumes the established numeric duration format;
it does not implement Python policy reconstruction or authenticate content hashes.
The unified workflow performs Python integrity checks before invoking it. Direct
standalone C++ users should first run the integrity CLI on imported profiles.

No production training scheduler, transport or default Megatron 1F1B hot path
was changed by this review.

## Validation Commands and Results

Existing containers only; tests are not summed across overlapping runs.

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev bash -c \
  '/opt/venv/bin/python -m pytest tests/unit_tests/pipeline_parallel/test_slackpipe_*.py \
  -q --capture=fd --tb=short -o addopts= --disable-warnings \
  --junitxml=slackpipe/build/refactor-audit-20261021/final-all.xml'
```

Full execution: **451 passed, 22 skipped, 6 failed, 0 deselected**, 167.25 s.
All six failures below are preserved as failures. The four final layer type/
index checks and malformed-policy partial-failure check were added afterwards
and validated with their actual consumers:

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev /opt/venv/bin/python -m pytest \
  tests/unit_tests/pipeline_parallel/test_slackpipe_isolated_profile.py \
  tests/unit_tests/pipeline_parallel/test_slackpipe_profile_producers.py \
  tests/unit_tests/pipeline_parallel/test_slackpipe_profiling_contract.py \
  -q --capture=fd --tb=short -o addopts= --disable-warnings \
  --junitxml=slackpipe/build/refactor-audit-20261021/final-integrity.xml
```

Final focused execution: **141 passed, 1 skipped**, 4.81 s. The skip is the same
real isolated-unit GPU test listed below, not an additional validation.
The explicit-artifact test calls the real target configuration/hash loader,
not just generic integrity validation. Its individual rerun also passed
(0.15 s, `artifact-import.xml`), using the same pytest options and node
`test_slackpipe_profiling_contract.py::test_existing_stage_import_does_not_require_collection_capability`.
That run and the earlier `final-focused.xml` (140 passed, one skipped) overlap
the final focused run and must not be added to the totals.

Intermediate review logs are retained: `targeted.xml` (114 passed, 28 fixture
failures caught by the stronger preflight/policy checks), `full.xml` (445 passed,
22 skipped, the six GPU failures plus two old PP=N context fixtures).
Fixtures were corrected with real builders/supported topology; no skips or
weakened assertions were introduced. These intermediate counts are not successes.

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev cmake \
  -S slackpipe -B slackpipe/build/no-or -G Ninja -DSLACKPIPE_ENABLE_ORTOOLS=OFF
docker exec -w /workspace/Megatron-LM slackpipe-dev cmake \
  --build slackpipe/build/no-or -j4
docker exec -w /workspace/Megatron-LM slackpipe-dev ctest \
  --test-dir slackpipe/build/no-or --output-on-failure

docker run --rm --entrypoint /bin/bash \
  -v /home/rui/Megatron-LM:/workspace/Megatron-LM -w /workspace/Megatron-LM \
  -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory \
  -e GIT_CONFIG_VALUE_0=/workspace/Megatron-LM \
  -e SLACKPIPE_OR_TEST_FILTER=.* -e BUILD_JOBS=4 \
  -e BUILD_DIR=/workspace/Megatron-LM/slackpipe/build/cpp-rename-or \
  slackpipe-ortools-runtime:local slackpipe/scripts/validate_ortools_evaluation.sh
```

Both builds: **4/4 CTest targets pass** (3.12 s / 3.82 s), OR-Tools 9.15.6755.
The no-OR API explicitly reports UNAVAILABLE; its OR-labelled smoke target is
not evidence of an OR solve. Real no-hint optimal/infeasible solves were executed
in the OR-enabled build. CTest `Testing/Temporary/LastTest.log` files contain
per-target commands/results: the core runner reports 92 cases and the smoke
filter reports six overlapping cases, not 98 distinct tests. No solver-speed
improvement is claimed.

Also executed: container `uv run --no-project --python /opt/venv/bin/python isort`
on edited imports, scoped Black, `ruff check --select F` (passed), CLI `--help`,
and `git diff --check` (passed).
Real rejected-configuration check:

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev /opt/venv/bin/python \
  -m tools.run_slackpipe_eval --phase calibrate --method slack-heuristic \
  --model-config configs/slackpipe_eval/llama_8b.json \
  --pp 1 --num-stages 4 --microbatches 4 \
  --output /tmp/slackpipe-unsupported-audit-20261021
docker exec slackpipe-dev test ! -e /tmp/slackpipe-unsupported-audit-20261021
```

First command exits 1 with the specific interleaved capability error in
normalization, before workers; second exits 0. Help lists exactly five methods,
stage/layer/both, selection defaults and supported new-stage topology.

## Previous Report Reconciliation

Read actual previous JUnit files in `slackpipe/build/profiling-contract-review/`:
`final-slackpipe-cpu.xml` contains 444 executed cases: 427 passed and 17 skipped.
Its documented selection expression was:

```text
not heterogeneous_pp1_exact_construction and not heterogeneous_numerical_equivalence
and not hybrid_numerical_equivalence and not exact_count_matches_instantiated_model
```

The exact node list was recollected with the inverse expression. The previous
JUnit does not itself contain deselected nodes or the entire shell command.
The prior broader 197/10/4 run has only the prior written report here, not a
retained standalone JUnit/log. It is not used as new evidence; its four GPU
failures were actually rerun in this review. `profiling-tests.xml` retains the
earlier 33 failures, including the other two exact-model CUDA setup failures.
`all-slackpipe-cpu.xml` retains two subsequent mock `num_layers` failures,
absent from the prior final XML. Prior CTest logs were not separately archived
before rebuilding; this review's builds provide fresh evidence.

All following node IDs are relative to `tests/unit_tests/pipeline_parallel/`.
No item became inapplicable because a scheduling method was removed.

### Former 11 Deselections

The old command excluded all of these by name, not by an explicit skip marker.
All were restored to execution this time:

| Node | This review |
| --- | --- |
| `test_slackpipe_eval.py::test_exact_count_matches_instantiated_model[llama]` | Failure G1 |
| `test_slackpipe_eval.py::test_exact_count_matches_instantiated_model[nemotron_h]` | Failure G1 |
| `test_slackpipe_heterogeneous.py::test_heterogeneous_pp1_exact_construction` | Failure G1 |
| `test_slackpipe_heterogeneous.py::test_heterogeneous_numerical_equivalence[1]` | Failure G2 |
| `test_slackpipe_heterogeneous.py::test_heterogeneous_numerical_equivalence[2]` | WORLD_SIZE != 2 |
| `test_slackpipe_hybrid.py::test_hybrid_numerical_equivalence[fp32-1]` | Failure G2 |
| `test_slackpipe_hybrid.py::test_hybrid_numerical_equivalence[bf16-1]` | Failure G2 |
| `test_slackpipe_hybrid.py::test_hybrid_numerical_equivalence[fp32-2]` | WORLD_SIZE != 2 |
| `test_slackpipe_hybrid.py::test_hybrid_numerical_equivalence[bf16-2]` | WORLD_SIZE != 2 |
| `test_slackpipe_hybrid.py::test_hybrid_numerical_equivalence[fp32-4]` | device_count < 4 |
| `test_slackpipe_hybrid.py::test_hybrid_numerical_equivalence[bf16-4]` | device_count < 4 |

These remain applicable model/construction/numerical checks. PP=2 needs torchrun;
PP=4 needs four functioning GPUs and is unavailable on this two-GPU host.

### Original 17 Skips, Still Skipped

Single-device availability guards (seven cases):

- `test_slackpipe_compute_timing.py::test_real_cuda_events_resolve_after_full_step`
- `test_slackpipe_isolated_profile.py::test_actual_units_finite_repeatable_and_no_pipeline`
- `test_slackpipe_model_construction.py::test_slackpipe_pp1_constructs_logical_vpp_chunks`
- `test_slackpipe_model_construction.py::test_slackpipe_pp1_numerical_equivalence`
- `test_slackpipe_overlap_diagnostic.py::test_cuda_serialization_preserves_gradients_and_update[False]`
- `test_slackpipe_overlap_diagnostic.py::test_cuda_serialization_preserves_gradients_and_update[True]`
- `test_slackpipe_stage_measurement.py::test_events_and_node_labels_preserve_gradients_and_optimizer`

Each tests actual GPU behavior and needs functioning CUDA; no CPU mock can replace it.

Two-device availability guards (two cases):

- `test_slackpipe_communication.py::test_slackpipe_logical_edge_communicators_are_isolated`
- `test_slackpipe_model_construction.py::test_slackpipe_pp2_numerical_equivalence`

The first checks logical communicator isolation, the second distributed numerical
equivalence. Both also require WORLD_SIZE=2 after their availability guard.

WORLD_SIZE != 2 guards (eight cases):

- `test_slackpipe_rma.py::test_rma_puts_complete_before_any_consumer_waits[dtype0]`
- `test_slackpipe_rma.py::test_rma_puts_complete_before_any_consumer_waits[dtype1]`
- `test_slackpipe_rma.py::test_rma_numerical_equivalence[pp2-equivalence]`
- `test_slackpipe_rma.py::test_rma_numerical_equivalence[m1-delayed-match]`
- `test_slackpipe_rma.py::test_rma_existing_solver_plans[M0_equal_joint]`
- `test_slackpipe_rma.py::test_rma_existing_solver_plans[M2_shared_slopes_joint]`
- `test_slackpipe_rma.py::test_rma_existing_solver_plans[M3_full_joint]`
- `test_slackpipe_rma.py::test_rma_existing_solver_plans[measured]`

dtype0/dtype1 are FP32/BF16. RMA needs working CUDA, two ranks and the existing
NCCL RMA-capable environment. The six wrapper cases inherit the model-construction
function body and hit its WORLD_SIZE guard. The four historical plan names are
still meaningful legal-order/lifecycle regressions, not retired-method launches.
These test communication completion, persistent buffers and numerical execution.
The five restored rank/device skips above plus these 17 explain all 22 current
skips; none are new skip decorators.

### Actual GPU Failures

Command: the unrestricted full pytest invocation above, not a mock or renamed
skip. `final-all.xml` preserves each stack trace.

- **G1 (three named nodes above):** `Utils.initialize_model_parallel` ->
  `Utils.initialize_distributed`, `tests/unit_tests/test_utilities.py:55`,
  `Utils.rank % torch.cuda.device_count()`: `ZeroDivisionError: integer modulo
  by zero`. Observed count=0 confirms this immediate cause. No model was built.
- **G2 (three named nodes above):** heterogeneous/hybrid test
  `torch.cuda.set_device` -> `_lazy_init` -> `_cuda_init`:
  `RuntimeError: No CUDA GPUs are available`. No F/B/numerical comparison ran.

The six failures are not evidence of a numerical regression or its absence.
The environment's underlying NVML failure is unresolved. Ordinary CPU,
CLI, fit and C++ replay successes are separate evidence levels.

## Minimum GPU Follow-up

Run only after the existing container exposes CUDA again. Use fresh output
paths. These are runnable follow-ups, **not** commands successfully GPU-validated
in this review:

```bash
# PP=1 construction, counts, homogeneous/heterogeneous/hybrid checks.
docker exec -w /workspace/Megatron-LM slackpipe-dev /opt/venv/bin/python \
  -m torch.distributed.run --standalone --nproc-per-node=1 -m pytest \
  tests/unit_tests/pipeline_parallel/test_slackpipe_model_construction.py \
  tests/unit_tests/pipeline_parallel/test_slackpipe_eval.py \
  tests/unit_tests/pipeline_parallel/test_slackpipe_heterogeneous.py \
  tests/unit_tests/pipeline_parallel/test_slackpipe_hybrid.py \
  -q -o addopts=

# Generate a fresh legal solver plan, then exercise PP=2 with it.
docker exec -w /workspace/Megatron-LM slackpipe-dev \
  slackpipe/build/no-or/slackpipe_cli --algorithm slack-heuristic \
  --B 4 --N 4 --J 2 --L 8 --time-limit 1 \
  --emit-plan /tmp/slackpipe-review-pp2.plan.json
docker exec -w /workspace/Megatron-LM \
  -e SLACKPIPE_EXTERNAL_PP2_PLAN=/tmp/slackpipe-review-pp2.plan.json slackpipe-dev \
  /opt/venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=2 \
  -m pytest tests/unit_tests/pipeline_parallel/test_slackpipe_model_construction.py::test_slackpipe_pp2_numerical_equivalence \
  tests/unit_tests/pipeline_parallel/test_slackpipe_communication.py \
  tests/unit_tests/pipeline_parallel/test_slackpipe_rma.py \
  tests/unit_tests/pipeline_parallel/test_slackpipe_heterogeneous.py \
  tests/unit_tests/pipeline_parallel/test_slackpipe_hybrid.py -q -o addopts=

# Real unit/events/diagnostic measurements, separately from the above run.
docker exec -w /workspace/Megatron-LM slackpipe-dev /opt/venv/bin/python -m pytest \
  tests/unit_tests/pipeline_parallel/test_slackpipe_isolated_profile.py::test_actual_units_finite_repeatable_and_no_pipeline \
  tests/unit_tests/pipeline_parallel/test_slackpipe_compute_timing.py::test_real_cuda_events_resolve_after_full_step \
  tests/unit_tests/pipeline_parallel/test_slackpipe_overlap_diagnostic.py \
  tests/unit_tests/pipeline_parallel/test_slackpipe_stage_measurement.py \
  -q -o addopts=
```

For four-rank hybrid cases use the corresponding hybrid nodes with
`--nproc-per-node=4` on a four-GPU system. Native stage/layer collection itself
still needs a real appropriately sized model run; see the profiling command forms,
not the 8B preset as a promise that it fits a 6GB card.

## Files Changed by This Review

Not a claim that all pre-existing dirty changes belong to this review:

- `tools/slackpipe_profiling.py`: preflight/capability and calibration schedule identity.
- `tools/slackpipe_eval_args.py`: early launcher validation, worker distinction, CLI help.
- `tools/run_slackpipe_eval.py`: guard dependencies/worker creation; distinct calibration context.
- `tools/run_slackpipe_real_system_campaign.py`: validate all requested methods before work.
- `tools/slackpipe_eval_worker.py`: explicit execution-worker normalization.
- `megatron/core/pipeline_parallel/slackpipe/profile_integrity.py`: stage policy/evidence reconstruction and backend consistency.
- `megatron/core/pipeline_parallel/slackpipe/stage_residual.py`: final ratio and refit inventory.
- `megatron/core/pipeline_parallel/slackpipe/cost_profile.py`: raw heterogeneous identity checks.
- `megatron/core/pipeline_parallel/slackpipe/isolated_profile.py`: nonnumeric/boolean sample integrity.
- `tests/unit_tests/pipeline_parallel/test_slackpipe_profiling_contract.py`: preflight, masks, rollback, import and policy-corruption tests.
- `tests/unit_tests/pipeline_parallel/test_slackpipe_eval_receipts.py`: real builder fixtures, default-both native independence, campaign preflight.
- `tests/unit_tests/pipeline_parallel/test_slackpipe_group_quality.py`: valid four-stage fixture option and full-direction rollback.
- `tests/unit_tests/pipeline_parallel/test_slackpipe_profile_quality.py`: actual v2 builder fixture instead of relabelling v1.
- `tests/unit_tests/pipeline_parallel/test_slackpipe_isolated_profile.py`: supported context fixture, schedule identity, malformed samples.
- `slackpipe/tests/joint_tests.cc`: deterministic no-solution API test.
- `slackpipe/tests/result_validator_cli_test.py`: deterministic no-solution CLI, no-seed flags/output checks.
- `docs/slackpipe_profiling.md`: preflight versus runtime partial failure; policy integrity.
- `docs/slackpipe_method_simplification.md`: link to this newer review, preserving old results.
- This report.

Shared Python parser, profiling and integrity consumers are affected and covered
by receipt, producer, loader, offline-fit, plan/parser and export tests. No common
C++ production module was edited; both C++ builds were nevertheless run to
validate the no-hint behavior and changed tests.

**Verdict:** source/call-path review and applicable CPU/build checks completed.
GPU hardware acceptance is **not passed**. No fresh profiling, training,
PP=2 end-to-end numerical acceptance or RMA hardware acceptance is claimed.

## Limited GPU-Preflight Closeout

Same branch/HEAD, existing dirty work preserved, no staging/commit/push/PR.
No algorithm, estimator, residual rule, supported method set, transport or
production training schedule changed. This section supersedes only the
environment-error handling findings, not the historical full-suite result.

### Actual failure paths and fixes

Read the six tracebacks in `refactor-audit-20261021/final-all.xml`:

| Previous failed node (under `tests/unit_tests/pipeline_parallel/`) | Current result |
| --- | --- |
| `test_slackpipe_eval.py::test_exact_count_matches_instantiated_model[llama]` | skip before binding, requires 1 GPU, observed 0 |
| `test_slackpipe_eval.py::test_exact_count_matches_instantiated_model[nemotron_h]` | same |
| `test_slackpipe_heterogeneous.py::test_heterogeneous_pp1_exact_construction` | same |
| `test_slackpipe_heterogeneous.py::test_heterogeneous_numerical_equivalence[1]` | same |
| `test_slackpipe_hybrid.py::test_hybrid_numerical_equivalence[fp32-1]` | same |
| `test_slackpipe_hybrid.py::test_hybrid_numerical_equivalence[bf16-1]` | same |

All six were actually re-executed in `final-focused.xml`. None has passed its
GPU assertions; none now fails setup in that run.

The first three share `Utils.initialize_model_parallel` (wrapped by
`managed_model_parallel_groups`) -> `Utils.initialize_distributed` ->
`Utils.rank % torch.cuda.device_count()`. Evaluation of the argument raised
before `set_device` ran; distributed rendezvous and model construction had
not started. Earlier cleanup on these paths only cleared empty group/runtime
references and environment settings. The latter three called `set_device`
directly before their helper and failed in CUDA lazy initialization.

Changes in this closeout:

- `tests/unit_tests/test_utilities.py`: existing public GPU test helper now
  probes availability/count before modulo, binding or rendezvous. It checks
  local process count (`LOCAL_WORLD_SIZE`, else fixture world size), not just
  PP, so TP/DP/CP workers are not omitted. Negative inactive fixture ranks
  retain their old behavior. Out-of-range active local ranks fail explicitly.
- `test_slackpipe_heterogeneous.py`, `test_slackpipe_hybrid.py`: GPU-only
  numerical tests call that check before direct binding. Target TP=DP=CP=1,
  one process per physical PP rank; required count is respectively PP=1/2
  and PP=1/2/4. CPU tests in the same files are unaffected.
- `tools/slackpipe_eval_common.py`: small explicit capability check, with
  no exception swallowing, CPU fallback, device remapping or new framework.
- `tools/run_slackpipe_eval.py`: check the single-node launcher's requested PP
  before environment context/device-property queries, receipt reuse or workers.
  Read-only context inspection does not query properties when CUDA is
  unavailable. CPU `inspect`, native solve/not-applicable, parser and profile
  integrity paths do not acquire GPU execution prerequisites. The pure
  advisory `memory_preflight` helper is unchanged; GPU execution no longer
  reaches its hardware-capacity skip when GPUs are absent/insufficient.
- `tools/slackpipe_eval_worker.py`: validate WORLD_SIZE=PP (TP=DP=CP=1),
  local process count, global/local ranks and visible capacity before explicit
  binding, transformer configuration/model construction or distributed init.
  Count local workers rather than imposing a new multi-node scheduling policy.
  Existing torchrun failure propagation remains unchanged.
- `tools/slackpipe_isolated_profile.py`: check one GPU before output creation,
  binding, gloo/model-parallel initialization or unit construction. Its target
  profile PP does not imply that isolated collection needs PP GPUs.
- `test_slackpipe_eval.py`, `test_slackpipe_eval_receipts.py`: receipt-only
  fixtures now explicitly represent capable synthetic GPU environments;
  production capacity checks are not bypassed.
- New `test_slackpipe_gpu_preflight.py`: zero/unavailable/insufficient devices,
  invalid rank config, no downstream binding/init, CLI nonzero exit, explicit
  fixture skips, unknown probe/binding failures propagating, CPU-only path,
  schedule preflight precedence and torchrun peer termination.

New skip conditions are **only** in the GPU test helper/call sites:
zero visible devices; nonzero count but CUDA unavailable; count below the
test's required local workers. Messages include required count, availability
and observed count. No probe exceptions are caught. Existing rank-count guards
remain. The PP2/PP4 heterogeneous/hybrid cases now report the device prerequisite
first when both rank and hardware requirements are missing. No entire module
is skipped. The five CPU heterogeneous tests in the focused batch still ran.
Production errors raise, never pytest-skip, retry profiling, or change transport.
Example:

```text
RuntimeError: No visible CUDA devices (required=2, cuda_available=False,
visible_devices=0); GPU execution cannot start
```

The mock CLI test imports dependency libraries before replacing capability
probes: installed TE queries compute capability at import when CUDA advertises
availability. This is a test-fixture boundary, not proof about dependency import
side effects. The tested guards precede explicit execution initialization.

### Environment evidence

Rechecked the existing container; same mount, image and software versions above.
`torch.cuda.is_available() == False`, `device_count() == 0`.
Container `nvidia-smi`: exit 255, `Failed to initialize NVML: Unknown Error`.
Host `nvidia-smi`: exit 18, `Driver/library version mismatch`, NVML library
615.71. `/proc/driver/nvidia/version` reports loaded kernel module 595.71.05.
This provides a concrete host mismatch diagnostic, but does not prove a single
complete cause of every container symptom. No driver/library/device changes
were attempted. Hardware runs were not retried.

NCCL `(2, 29, 7)` and `_symmetric_memory.put_signal`, `wait_signal`, `rendezvous`
exist. These satisfy software feature checks only, not RMA hardware acceptance.
PP2 needs two working CUDA devices; PP4 fixtures cannot run on the two-GPU host.

Commands used for these observations:

```bash
docker inspect slackpipe-dev --format '{{json .Mounts}} {{json .State.Status}} {{json .Config.Image}}'
docker exec slackpipe-dev nvidia-smi
nvidia-smi
cat /proc/driver/nvidia/version
docker exec slackpipe-dev /opt/venv/bin/python -c \
 'import torch, importlib.metadata as m; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count()); print({n:m.version(n) for n in ("transformer-engine","mamba-ssm")})'
docker exec slackpipe-dev /opt/venv/bin/python -c \
 'import torch; import torch.distributed._symmetric_memory as s; print(torch.cuda.nccl.version()); print({k:hasattr(s,k) for k in ("put_signal","wait_signal","rendezvous")})'
```

### Validation actually executed

Evidence: ignored `slackpipe/build/gpu-preflight-review-20261021/`, including
`source-before.tar.gz` (principal touched files before this closeout).
Counts below overlap and must **not** be added:

| Batch | Result | Scope |
| --- | --- | --- |
| `final-focused.xml` | 108 passed, 11 skipped; 50.78 s | entry tests, eval, heterogeneous, hybrid, receipts; all six original failures revisited |
| `final-cpu-contracts.xml` | 182 passed, 1 skipped; 18.91 s | entry, profiling contract, isolated producer, plan parser; sole skip is actual isolated GPU test |
| `final-entry.xml` | 25 passed; 16.25 s | final entry checks after local-rank/count refinement, including waiting-peer termination |

First focused attempt (`focused.xml`): 105 passed, 11 skipped, 3 failed:
premature mock availability triggered TE import initialization, one expected
message regex said interleaved instead of interleaving, and a legacy receipt
fixture lacked explicit synthetic device prerequisites. All three fixtures
were fixed without weakening the production checks. An attempted CPU batch
referenced nonexistent `test_slackpipe_manifest.py`, exited 4 and ran no tests
(`cpu-contracts.xml`); corrected command below. Manifest CPU coverage is in the
heterogeneous file of the focused batch.

Actual pytest commands (inside the existing container):

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev /opt/venv/bin/python -m pytest \
 tests/unit_tests/pipeline_parallel/test_slackpipe_gpu_preflight.py \
 tests/unit_tests/pipeline_parallel/test_slackpipe_eval.py \
 tests/unit_tests/pipeline_parallel/test_slackpipe_heterogeneous.py \
 tests/unit_tests/pipeline_parallel/test_slackpipe_hybrid.py \
 tests/unit_tests/pipeline_parallel/test_slackpipe_eval_receipts.py \
 -q --capture=fd --tb=short -o addopts= --disable-warnings \
 --junitxml=slackpipe/build/gpu-preflight-review-20261021/final-focused.xml
docker exec -w /workspace/Megatron-LM slackpipe-dev /opt/venv/bin/python -m pytest \
 tests/unit_tests/pipeline_parallel/test_slackpipe_gpu_preflight.py \
 tests/unit_tests/pipeline_parallel/test_slackpipe_profiling_contract.py \
 tests/unit_tests/pipeline_parallel/test_slackpipe_isolated_profile.py \
 tests/unit_tests/pipeline_parallel/test_slackpipe_plan.py \
 -q --capture=fd --tb=short -o addopts= --disable-warnings \
 --junitxml=slackpipe/build/gpu-preflight-review-20261021/final-cpu-contracts.xml
docker exec -w /workspace/Megatron-LM slackpipe-dev /opt/venv/bin/python -m pytest \
 tests/unit_tests/pipeline_parallel/test_slackpipe_gpu_preflight.py \
 -q --capture=fd --tb=short -o addopts= --disable-warnings \
 --junitxml=slackpipe/build/gpu-preflight-review-20261021/final-entry.xml
```

Real CLI command below was executed once through the existing `run()` subprocess
wrapper with a 60 s timeout. It exited 1 with the explicit prerequisite error,
without creating its output directory. Log: `production-no-gpu.log`.
No torchrun/model/profile job was started by this invocation.

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev /opt/venv/bin/python \
 -m tools.run_slackpipe_eval --phase benchmark --method interleaved-1f1b \
 --model-config configs/slackpipe_eval/llama_8b.json \
 --pp 2 --num-stages 4 --microbatches 4 \
 --output /tmp/slackpipe-preflight-no-gpu-20261021
docker exec slackpipe-dev test ! -e /tmp/slackpipe-preflight-no-gpu-20261021
```

The two-process test deliberately hides CUDA, makes rank 0 fail preflight and
leaves rank 1 sleeping. The existing torchrun plus `run()` timeout/cleanup
returns nonzero and the peer PID no longer exists. This validates process
failure handling without pretending to test NCCL or GPU communication.

Scoped container `uv run --no-project --python /opt/venv/bin/python isort`,
Black checks on all ten changed Python files (preserving upstream single quotes
in `test_utilities.py` with `--skip-string-normalization`), and
`ruff check --select F` passed. `git diff --check` passed.
No C++ sources changed in this closeout; neither C++ build was rerun.
The earlier 4/4 no-OR and 4/4 OR results remain historical, as does the full
**451 passed, 22 skipped, 6 failed** result. The earlier 141/1 focused result
also remains a separate historical batch.

### GPU acceptance matrix

| Item | This closeout | Evidence / outstanding requirement |
| --- | --- | --- |
| PP2 native 1F1B | Not executed | CUDA unavailable; numerical acceptance still pending |
| PP2 native interleaved | Not executed | Same; native benchmark independence covered only by CPU orchestration tests |
| New stage profiling | Not executed | Requires functioning two-GPU native interleaving |
| New layer profiling | Not executed | Requires at least one functioning CUDA GPU |
| Both, stage-default selection -> solve/validate/parse/PP2 | Not executed on GPU | Selection/integrity tested on CPU only; no new measured profile |
| Explicit layer selection -> solve/validate/parse/PP2 | Not executed on GPU | Same |
| NCCL P2P completion/numerical correctness | Not executed | No usable CUDA device |
| NCCL RMA lifecycle/completion/numerical correctness | Not executed | API/version present, GPU capability unverified; no fallback attempted |

### Runnable minimum follow-up after GPU repair

Use the **existing** container only after its CUDA probe reports availability
and at least two devices. Do not run these commands repeatedly while that check
fails. These commands are provided for pending work, **not** new GPU evidence.
The existing small model fixture is used instead of allocating an 8B preset.
All output is fresh; no historical receipts are overwritten.

```bash
docker exec -it -w /workspace/Megatron-LM slackpipe-dev bash
# All following commands are inside this existing container.
set -eu
export PYTHONPATH=/workspace/Megatron-LM
PY=/opt/venv/bin/python
$PY -c 'from tools.slackpipe_eval_common import require_cuda_devices; require_cuda_devices(2)'
R=$(mktemp -d /tmp/slackpipe-gpu-acceptance.XXXXXX)
export R
$PY -c 'import os; from pathlib import Path; from tests.unit_tests.pipeline_parallel.test_slackpipe_eval import tiny_model; from tools.slackpipe_eval_common import write_json; write_json(Path(os.environ["R"])/"model.json", tiny_model("llama"))'
COMMON=(--model-config "$R/model.json" --pp 2 --microbatches 4
 --precision fp32 --seq-length 32 --micro-batch-size 1
 --warmups 2 --iterations 2 --repetitions 1
 --smoke-warmups 2 --smoke-iterations 2 --timeout 240)

# Native benchmark dependency graph must not invoke calibration with default both.
for method in 1f1b interleaved-1f1b; do
 N=4; if [ "$method" = 1f1b ]; then N=2; fi
 $PY -m tools.run_slackpipe_eval "${COMMON[@]}" --phase benchmark \
  --method "$method" --num-stages "$N" --output "$R/native-$method"
done

# Sequential stage then layer collection, with default solver selection=stage.
$PY -m tools.run_slackpipe_eval "${COMMON[@]}" --phase calibrate \
 --method interleaved-1f1b --num-stages 4 --profiling-method both \
 --calibration-warmups 2 --calibration-iterations 3 \
 --isolated-profile-warmups 2 --isolated-profile-iterations 3 \
 --output "$R/both"
S=$(find "$R/both/calibration" -path '*/stage/worker/cost_profile.json')
L=$(find "$R/both/calibration" -path '*/layer/worker/cost_profile.json')
test -f "$S"; test -f "$L"

# Reuse those exact fresh profiles; no calibration rerun. Fresh receipt namespaces.
for backend in stage layer; do
 SELECT=()
 if [ "$backend" = layer ]; then SELECT=(--solver-profile layer); fi
 $PY -m tools.run_slackpipe_eval "${COMMON[@]}" --phase smoke \
  --method slack-heuristic --num-stages 4 --profiling-method both \
  --stage-profile "$S" --layer-profile "$L" "${SELECT[@]}" \
  --solver slackpipe/build/no-or/slackpipe_cli --solver-seconds 1 \
  --transport nccl-p2p --output "$R/solve-$backend"
 PLAN=$(find "$R/solve-$backend/solver" -name slackpipe.plan.json)
 test -f "$PLAN"
 SLACKPIPE_EXTERNAL_PP2_PLAN="$PLAN" \
 SLACKPIPE_EXTERNAL_TRACE_DIR="$R/equivalence-$backend" \
 SLACKPIPE_TEST_TRANSPORT=nccl-p2p \
 timeout --kill-after=15s 240s $PY -m torch.distributed.run --standalone \
  --nproc-per-node=2 -m pytest \
  tests/unit_tests/pipeline_parallel/test_slackpipe_model_construction.py::test_slackpipe_pp2_numerical_equivalence \
  -q --capture=fd -o addopts=
done

# Communication isolation and explicit RMA lifecycle/equivalence, separate evidence.
timeout --kill-after=15s 240s $PY -m torch.distributed.run --standalone \
 --nproc-per-node=2 -m pytest \
 tests/unit_tests/pipeline_parallel/test_slackpipe_communication.py \
 tests/unit_tests/pipeline_parallel/test_slackpipe_rma.py \
 -q --capture=fd -o addopts=
```

Check each `collection.json` plus backend `receipt.json` and raw worker files,
and each solve receipt's `profile_identity`, solver result, independent validator
output, parsed topology and runtime trace. The external-plan equivalence fixture
requires exact zero initial/loss/gradient/post-SGD differences against its local
GPT reference and checks the runtime operation order. It validates the selected
plan's execution, **not** target-model prediction accuracy. The target smoke
separately checks finite loss/gradients and a representable SGD update.

The launcher's correctness prerequisite is the existing small hybrid
SlackPipe-vs-reference test, not a native 1F1B-vs-interleaved equality proof.
Native benchmark/smoke exit success or that prerequisite alone must not be
reported as native cross-schedule numerical equivalence. This comparison is
still pending in the GPU acceptance work; no new numerical tolerance or
comparator was invented in this environment-only closeout.

RMA tests explicitly request RMA and require actual completion/lifecycle
assertions; API presence or P2P success cannot substitute. Preserve logs on
failure and check for leftover workers before another GPU run. No statistical
resampling/quality gate is added to these follow-ups.

## Publication Checks

On 2026-10-21, the user explicitly requested publication of the broader cleanup,
not just the already published NVSkills removal. Starting HEAD was
`182ee4ca1d86fce8a82e3a40b63f25290b80f393`, on `slackpipe/mvp` tracking
`fork/slackpipe/mvp`. The reviewed change includes the five-method simplification,
shared experiment/profiling entry points, their supporting diagnostics/tests,
and the GPU preflight closeout. It does not change upstream native scheduler
implementations or add a new estimator/solver modification during publication.

The pre-publication full SlackPipe pytest run found **479 passed, 28 skipped,
2 failed**, 181.54 s (`slackpipe/build/publication-cleanup-20261021/all.xml`).
Both failures were parameterizations of the synthetic CPU publication test
`test_isolated_collect_publishes_all_samples_once`: it mocked every GPU action
but had omitted the newly added capacity precondition. The fixture now explicitly
mocks that prerequisite and asserts the one-device/local-rank-zero request.
Production preflight remains unchanged; no numerical failure is converted into
a skip. Twelve unused imports left by the refactor were also removed.

After those fixes, the complete SlackPipe selection was rerun: **481 passed,
28 skipped, 0 failed, 0 deselected**, 184.07 s (`final-all.xml`). This is a new
batch, not a replacement for the older reports or a sum of focused reruns:

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev bash -c \
  '/opt/venv/bin/python -m pytest tests/unit_tests/pipeline_parallel/test_slackpipe_*.py \
  -q --capture=fd --tb=short -o addopts= --disable-warnings \
  --junitxml=slackpipe/build/publication-cleanup-20261021/final-all.xml'
```

Separate final focused batches (overlapping; not additive):

- Producer plus GPU-preflight contracts: **37 passed**, 18.54 s,
  `producer-preflight.xml`.
- Files affected by import cleanup and diagnosis imports: **39 passed,
  4 skipped**, 1.47 s, `import-cleanup.xml`.
- Existing-container no-OR build and CTest: **4/4 targets passed**, 3.12 s.
- Existing OR-Tools 9.15.6755 image build and full CTest filter: **4/4 targets
  passed**, 3.81 s. The validation script also ran each retained plan backend
  and independent result validation, plus the capped CP-SAT case.
- `tools.slackpipe_artifact_smoke --method slack-heuristic` passed the CPU
  solver/export/independent-validator/production-Python-parser path. Its fresh
  receipt is in `slackpipe/build/publication-cleanup-20261021/artifact-smoke/`.
  This is abstract-cost correctness, not GPU execution or performance evidence.

The full pytest command is the earlier audit command with JUnit output under
`slackpipe/build/publication-cleanup-20261021/`. Builds use the documented
no-OR commands and `SLACKPIPE_OR_TEST_FILTER=.*` with
`slackpipe/scripts/validate_ortools_evaluation.sh` in the existing OR image.
No dependencies were installed. All 68 changed/new Python files pass isort,
Black (preserving upstream string style in `test_utilities.py`) and Ruff F
checks; six changed/new shell scripts pass `bash -n`. Diff whitespace checks
also pass.

CUDA was probed again in `slackpipe-dev`: `is_available=False`, visible devices
`0`, with an NVML initialization warning. The GPU acceptance matrix and runnable
follow-ups above remain outstanding. No new GPU measurements or equivalence
claims are made. Historical Nemotron data and prediction accuracy were not
revalidated in this publication pass.

Only explicitly reviewed source, tests, scripts and documentation are staged.
Raw dataset JSON, Nemotron ZIPs, reproduction archives/directories, generated
experiment/profiler data, build trees and local environments stay outside the
commit. They are neither deleted nor rewritten. NVIDIA `origin` and the separate
`feature/schedule` branch are not publication destinations.
