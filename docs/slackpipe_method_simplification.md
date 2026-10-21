# Five-method Simplification Review

## Scope and State

Reviewed on 2026-10-20 on `slackpipe/mvp`, starting from
`322fdf68cfcc146ec23d44b9ed2dd6ffdfc7bcaf`. This is a reviewable working-tree
change, **not a commit or publication**. No checkout reset, staging, push, PR,
merge, dependency installation or paper-result modification was performed.

The working tree already contained solver compatibility fixes, coupled-heuristic
work, Puzzle experiments and profiler diagnostics. Before pruning, their source
and dirty diff were saved under the ignored
`slackpipe/build/method-pruning-review/` directory (`source-before.tar.gz`,
`preexisting.patch`, `before-cli`). In-scope dependencies were migrated rather
than reverted. Historical datasets, archives, receipts and experiments were
not relabelled or overwritten. The previously modified `timing_diagnostic.py`
and `docs/slackpipe_compute_timing.md` were not part of this implementation.

The profiling follow-up is part of the current working-tree change. The earlier
method-only validation record below is historical, not acceptance evidence for
the new profiling policy. Historical paper results are unchanged.

The current supported entry points and exact five-method commands are in
[slackpipe_five_methods.md](slackpipe_five_methods.md). Public IDs are
`1f1b`, `interleaved-1f1b`, `octopipe-tuning`, `cpsat-no-hint`,
`slack-heuristic`; `slackpipe` is an executor, not an alias for an algorithm.

## Implemented Contracts

* Removed old joint initialization, external hints, incumbent horizon bounds,
  fallback, pressure/balance and partition-restriction paths from the actual
  CP-SAT implementation/API, not just the CLI. The conservative horizon,
  variable profile durations, FIFO, NoOverlap, capacity bounds and supported
  activation constraints remain. No-solution records have null plans/objectives
  where inapplicable and block GPU stages.
* Kept coupled-heuristic search logic and OctoPipe phase/neighborhood/acceptance
  logic. The formal OctoPipe backend fixes cyclic placement. Its timer now starts
  before initial-state evaluation, so initialization is charged to its budget.
* The registry determines topology, solver requirement, backend, threads,
  budgets and cap capability. Native 1F1B has N=PP. All other public methods
  require explicit N; native interleaving additionally requires PP>1, N>PP and
  B divisible by PP. All plan methods share the accepted profile at the same N.
* One public wall budget reaches both the runner and heuristic search; one
  worker is the default. Explicit CP-SAT parallel workers are recorded;
  heuristic/OctoPipe reject them and reject enforced activation caps.
* C++ result/method schema v2 and Python receipt/context v4 invalidate old
  contracts. Native solver metadata is null. Raw solver status/objective/bound,
  independent bound, replay, proof scope, requested time and actual time remain
  separate. Internal CP-SAT solution progress is not an external hint.
* Current run identity includes untracked source hashes as well as commit/diff.
  Resume rejects changed source, artifacts, parent contexts and old hinted-method
  receipts. Historical receipts are archived on explicit replacement.
* Tables and trace plots use method lists; five panels are supported. Old
  refinement/contribution comparisons are not relabelled. Raw JSON/CSV, independent
  run samples, trace capture/iteration identity and failure records remain.
* A report-only bug found during migration was corrected: profile utilization
  now sums actual replay operation durations, not the uniform abstract ratio.
  Search decisions are unaffected. Control characters in serialized CLI
  provenance are escaped, including newlines in paths.

## Profiling Contract

The public parser uses `--phase`, `--method`, `--num-stages`,
`--profiling-method {stage,layer,both}` (default both), and optional
`--solver-profile {stage,layer}`. Omitted selection resolves to stage for
both/stage, layer for layer-only. Explicit artifacts use `--stage-profile` and
`--layer-profile`. Native standalone benchmarks need neither profiling nor a solver.

Both backends execute sequentially once each with separate artifacts and receipts.
Nonselected failure is reported as partial; selected failure blocks dependencies
without fallback or retry. Actual measurement parallelism and downstream target
parallelism are separate. Stage collection currently requires native interleaving:
PP>1, N>PP, N divisible by PP and B divisible by PP. Layer supports target N=PP;
unsupported stage collection can be partial when layer is selected. No native
noninterleaved timing hot path was added.

Only stage profiling can exclude samples: initial raw-data fit, one phase-local
affine-relative-residual mask, at most one refit. Layer uses all legal samples.
CV/drift/consensus/discard-rate/fit-error stability gates are removed, not bypassed.
Structural, provenance, raw-inventory, prefix and plan checks remain.

For stage, `q=(y-p)/p`, `m=median(q)`, `s=1.4826*median(abs(q-m))`.
Exclude only when `abs(q-m) > k*s` AND `abs(q) > rho`, with finite positive
defaults `k=6.0`, `rho=0.5`. The mask uses initial predictions at operation
scale, not fitted summary residuals or iteration totals. Empty iterations have
no mean; empty required observations/structural loss roll back the whole phase.
This is one initial-fit mask, not iterative clipping or a paper guarantee.

Summary-only heterogeneous fits are labelled `stage-summary-fit-v1`, with
`filter_not_applied`, never falsely labelled `stage-affine-residual-v1`.
Existing explicitly supplied historical artifacts keep their bytes and policy;
old failed quality metadata is not a statistical admission rule.

`profile_identity` propagates backend, file SHA, cost hash and policy through
solve/benchmark/trace, separately from method and transport. Benchmark summary v2
computes main statistics and normal 95% confidence half-width from independent
continuous wall-time run observations. CUDA-event statistics are separate; a
single run has no confidence interval.

The [profiling guide](slackpipe_profiling.md) supplies native, both-backend and
layer-only minimal commands, formulas, measurement/communication scope, explicit
artifact reuse and structural rollback. The unified solver still uses zero
communication delay; neither compute backend adds a communication profiler or
link contention model.

## File Inventory

The grouped names below are literal files, with paired `.h`/`.cc` names denoting
both files. Generated files are not part of the source deliverable.

### C++ Retained and Migrated

* `slackpipe/include/slackpipe/search_result.h`: new method-neutral result.
* `slackpipe/include/slackpipe/instance.h`, `slackpipe/src/instance.cc`:
  `UniformSplit`, `WorkerLayerTotals`, capped partition counting moved here.
* `slackpipe/include/slackpipe/joint_solver.h`,
  `slackpipe/src/joint_solver.cc`, `slackpipe/src/joint_solver_cpsat.cc`:
  pure joint CP-SAT and explicit no-OR availability result.
* `slackpipe/include/slackpipe/evaluation_method.h`,
  `slackpipe/src/evaluation_method.cc`: five-method contract.
* `slackpipe/include/slackpipe/io.h`, `slackpipe/src/io.cc`,
  `slackpipe/include/slackpipe/result_schema.h`,
  `slackpipe/src/result_schema.cc`: neutral outputs, raw bounds/status and
  provenance; legacy algorithm-specific serialization removed.
* `slackpipe/include/slackpipe/slack_heuristic.h`,
  `slackpipe/src/slack_heuristic.cc`: renamed from `heuristic_hint`; standalone
  heuristic naming and neutral types/metadata, with search logic retained.
* `slackpipe/include/slackpipe/deadline.h`: removed retired budget fractions.
* `slackpipe/src/activation_analyzer.cc`: shared helper/internal BFS identity.
* `slackpipe/src/dag_evaluator.cc`: actual profile utilization reporting;
  pre-existing FIFO validation fixes preserved.
* `slackpipe/src/octopipe_tuner.cc`: helper include and inclusive budget timer.
* `slackpipe/apps/slackpipe_cli.cc`: three plan backends, native-method rejection,
  common budget/worker flags, no-solution/export handling.
* `slackpipe/apps/octopipe_phase_diagnostic.cc`: internal initializer names,
  not the public native-method registry.
* `slackpipe/CMakeLists.txt`, `slackpipe/scripts/validate_ortools_evaluation.sh`:
  retired targets removed, retained backends built/tested.

### C++ Removed

From `slackpipe/include/slackpipe/` and `slackpipe/src/`, paired `.h`/`.cc`:
`alternating_solver`, `fixed_order_partition_solver`, `hybrid_slack`,
`pressure_pruning`, `bfs_solver`, `slackpipe_solver`, `benchmark`, `puzzle_hint`.
Also removed `src/bfs_solver_cpsat.cc`, `src/slackpipe_solver_cpsat.cc`,
`src/partition_costs_cpsat.h`, `include/slackpipe/partition_restriction.h`,
`include/slackpipe/worker_balance.h`.

Removed apps: `slackpipe/apps/slackpipe_benchmark.cc`,
`slackpipe/apps/puzzle_hint_benchmark.cc`,
`slackpipe/apps/heuristic_hint_benchmark.cc`.
Puzzle and the standalone historical benchmark adapter were untracked at task
start; their exact bytes remain in the pre-task source snapshot.

Kept common `operation`, `types`, `schedule`, `breadth_first`, `cost_profile`,
`plan_export`, `workload_bound`, `activation_cpsat`, `timing`, `lifecycle`,
`search_stats`, independent `result_validator`, `heuristic_internal.h`,
`one_f_one_b` and `interleaving_stats`. DAG replay and independent validation
remain distinct implementations.

### Python Migrated or Updated

New shared modules:

* `tools/slackpipe_eval_common.py`: digest/source identity/process cleanup,
  deterministic environment, sample summaries and atomic JSON.
* `tools/slackpipe_eval_args.py`: reusable argument parser/normalization.
* `tools/slackpipe_model_support.py`: model construction, initialization,
  deterministic batches, forward adapter and global parameter names, shared
  with tests instead of imported from tests.
* `tools/slackpipe_training.py`: generic partition execution/construction and
  calibration partitions, without a duplicate launcher.

Updated active chain:
`tools/run_slackpipe_eval.py`, `tools/slackpipe_eval_config.py`,
`tools/slackpipe_eval_worker.py`, `tools/slackpipe_eval_receipts.py`,
`tools/slackpipe_eval_tables.py`, `tools/run_slackpipe_real_system_campaign.py`,
`tools/plot_schedule_trace.py`, `tools/slackpipe_hybrid.py`,
`tools/slackpipe_isolated_profile.py`, `tools/slackpipe_artifact_smoke.py`.

Updated thin wrappers:
`tools/runpod_nemotron_baseline.sh`, `tools/runpod_nemotron_slackpipe.sh`,
`tools/runpod_slackpipe_correctness.sh`, `tools/runpod_slackpipe_verify.sh`.

Preserved diagnostic callers with migrated imports:
`tools/slackpipe_calibration_transfer.py`,
`tools/slackpipe_diagnose_composition.py`,
`tools/slackpipe_linear_identification.py`,
`tools/slackpipe_linear_worker.py`, `tools/slackpipe_transfer_worker.py`.
`tools/slackpipe_scale_sweep.sh` remains a fixture diagnostic; new solver calls
use `cpsat-no-hint`, not historical hinted results.

### Profiling Modules

New:
* `megatron/core/pipeline_parallel/slackpipe/stage_residual.py`: residual policy,
  raw-operation summaries, auditable initial masks and structural phase rollback.
* `megatron/core/pipeline_parallel/slackpipe/profile_integrity.py`: schema/cost
  structure, identity, event provenance and immutable publication.
* `tools/slackpipe_profiling.py`: sequential collection, explicit artifact
  selection and per-backend outcomes without retry/fallback.
* `tools/slackpipe_profile_integrity.py`: read-only artifact validation CLI.

Migrated:
* Core `cost_profile.py`: retained homogeneous slope/bias and heterogeneous
  nonnegative class/role fits; all groups participate, raw residual evidence
  remains inspectable.
* Core `isolated_profile.py`: retained class mapping, boundary expansion and
  prefix reconstruction; spike/CV filtering replaced with all-sample medians.
* `tools/slackpipe_eval_args.py`, `run_slackpipe_eval.py`,
  `slackpipe_eval_worker.py`, `slackpipe_eval_receipts.py`,
  `run_slackpipe_real_system_campaign.py`: new public flags and v4 identity.
* `tools/slackpipe_isolated_profile.py`, `slackpipe_hybrid.py`,
  `slackpipe_stage_calibration.py`, `slackpipe_training.py`,
  `slackpipe_compare_isolated.py`, `slackpipe_inspect_iterations.py`,
  `slackpipe_stage_trace.py`, `slackpipe_linear_analysis.py`,
  `slackpipe_diagnose_residuals.py`: producers/diagnostics use migrated contracts.

Removed after extracting integrity/publication:
`megatron/core/pipeline_parallel/slackpipe/profile_quality.py` and
`tools/slackpipe_profile_quality.py`. Active workflows no longer use
`collect_with_retry`, `require_profile_quality`, group consensus or layer
spike/CV admission.

### Python and Campaigns Removed

Removed from `tools/`: `run_slackpipe_nemotron_h8b_pp4.py`,
`slackpipe_nemotron_worker.py`, `slackpipe_experiment_driver.py`,
`slackpipe_heterogeneous_experiment.py`, `slackpipe_plot_experiments.py`.
Manifest/fit stays in `slackpipe_hybrid`; generic execution uses the shared
modules. Fixture-only heterogeneous calibration moved to
`tests/unit_tests/pipeline_parallel/slackpipe_heterogeneous_calibration.py`.
Native numerical assertions remain in the test suite.

Removed from `slackpipe/scripts/`: `focused_eval.py`, `run_primary_ablation.py`,
`analyze_primary_ablation.py`, `run_worker_balance_study.py`, `cal_manifest.py`,
`generate_cal_manifest.py`, `run_cal_manifest.py`, `analyze_cal_results.py`,
`run_heuristic_comparison.py`.
Removed `slackpipe/configs/primary_ablation.json` and
`slackpipe/configs/worker_balance_easier.csv`.

Identity, resume, artifact validation, budget/timeout failure accounting,
summaries and figures are supplied by the unified runner/receipts/tables/plotter.
Retired ablation plots are not applicable to the new five-method contract.
No active legacy tree was created.

### Tests

Split `slackpipe/tests/core_tests.cc` into `common_tests.cc` and `joint_tests.cc`;
adapted the existing local `profile_solver_tests.cc`.
Updated `heuristic_tests.cc`, `heuristic_cli_test.py`, `octopipe_tests.cc`,
`octopipe_diagnostic_tests.cc`, `workload_bound_tests.cc`,
`result_validator_cli_test.py`.
Removed retired-only `benchmark_tests.cc`, `focused_eval_runner_test.py`,
`puzzle_tests.cc`.

In `tests/unit_tests/pipeline_parallel/`, added `test_slackpipe_methods.py` and
the fixture calibration tool above. Updated:
`slackpipe_perf_benchmark.py`, `test_slackpipe_campaign_plot.py`,
`test_slackpipe_collection.py`, `test_slackpipe_eval.py`,
`test_slackpipe_eval_receipts.py`, `test_slackpipe_group_quality.py`,
`test_slackpipe_heterogeneous.py`, `test_slackpipe_isolated_profile.py`,
`test_slackpipe_model_construction.py`, `test_slackpipe_nemotron_workflow.py`,
`test_slackpipe_profile_quality.py`, `test_slackpipe_topology.py`.
Removed assertions cover retired entry points, not retained numerical behavior.

Profiling follow-up adds `test_slackpipe_profiling_contract.py` for residual
boundaries, one-pass behavior, structural rollback, scaling, selection and partial
failure, plus `test_slackpipe_profile_producers.py` for peripheral collectors.
The existing `test_slackpipe_{profile_quality,group_quality,outliers}.py` retain
integrity, provenance, raw preservation and C++ loading regressions while replacing
obsolete stability rejection/retry expectations. Eval/receipt/method tests retain
configuration, sharing, invalidation, corruption and native routing coverage with
canonical flags, v4 receipts and wall-time summary v2. Layer tests retain
class/boundary/prefix reconstruction with all-sample medians. Cost and
iteration-inspection tests no longer require obsolete statistical approval.

### Documentation

Updated `README.md`, `slackpipe/README.md`, `slackpipe/EVALUATION.md`,
`docs/slackpipe_real_system_eval.md`, `slackpipe/docs/heuristic.md`, and
`slackpipe/docs/evaluation_{methods,budget_policy,result_schema,validation}.md`.
Added `docs/slackpipe_five_methods.md` and this report.
Historical-scope notices were added to `docs/octopipe_algorithm1.md`,
`docs/slackpipe_{artifact_audit,evaluation_overhead_audit,pp4_hybrid,validation_matrix}.md`
and `slackpipe/docs/{evaluation_activation_memory,evaluation_analysis,
evaluation_manifests,profile_solver_compatibility,puzzle}.md`.
Historical experiment numbers were not changed.

Added `docs/slackpipe_profiling.md`; updated README, five-method and real-system
guidance. Historical notices on `docs/slackpipe_profile_quality.md` and
`docs/slackpipe_isolated_quality_audit.md` retain original measurements and audit
narratives without presenting old gates/retries as current behavior.

## Profiling Validation Before Audit

The subsequent [2026-10-21 review](slackpipe_refactor_audit_20261021.md) fixes
pre-launch capability/policy-integrity gaps and lists each skipped, deselected
and failed GPU check. Counts below describe the earlier tree, not the latest run.

All checks use existing containers. XML/logs remain under ignored
`slackpipe/build/profiling-contract-review/` or temporary directories. The final
consolidated CPU run is the authoritative count, not a sum of overlapping runs.

* 4/4 no-OR and 4/4 OR-enabled CTest targets pass after the
  heuristic rename, including 92 C++ cases. Do not add repeated run counts.
* Initial broader Python run: 197 passed, 10 skipped, four failed at GPU setup:
  heterogeneous PP=1 exact construction/equivalence and hybrid PP=1 FP32/BF16
  equivalence. These were attempted failures, not GPU validation.
* Initial profiling run: 241 passed, one skipped, 33 failed, recorded in
  `profiling-tests.xml`. Thirty-one failures were synthetic fixture dimensions;
  two exact-model construction cases were blocked by CUDA availability.
* After fixture migration, the focused eval/receipt/method rerun passed 92 tests,
  with two GPU exact-construction cases deselected. The quality/group/outlier
  regressions passed in the earlier focused run. These overlap later consolidated
  tests and must not be added to their total.
* Final consolidated CPU selection: **427 passed, 17 skipped, 11 deselected**
  in 99.85 seconds, recorded in `final-slackpipe-cpu.xml`. Skips/deselections are
  environment-specific GPU/distributed checks, not passes. The preceding full
  selection's two failures from a producer mock missing `num_layers` were fixed.
  Residual-contract and layer-suite focused results overlap this consolidated
  suite and are not additional tests. The last added regression also audits a
  historical layer artifact's recorded mask/expansion without invoking its old
  filter or using its failed quality flag as an admission rule.
* The six migrated stage/integrity/evaluation test files were formatted using
  container `uv run --no-sync isort` and Black. Scoped Ruff F checks and
  `git diff --check` pass. No dependency installation, commit or push was performed.

Container observations: CUDA availability is false and device count is zero.
`nvidia-smi` exits 255 with `Failed to initialize NVML: Unknown Error`. The cause
is not diagnosed here. PP=2 solver-to-Megatron equivalence, GPU execution and fresh
timing measurements remain outstanding. Excluding GPU-only checks from CPU runs
does not turn their attempted failures into passes. Historical GPU/paper evidence
is preserved, not revalidated.

Reproduce the final CPU selection in the existing container:

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev bash -c \
  '/opt/venv/bin/python -m pytest tests/unit_tests/pipeline_parallel/test_slackpipe_*.py \
  -k "not heterogeneous_pp1_exact_construction and not heterogeneous_numerical_equivalence and not hybrid_numerical_equivalence and not exact_count_matches_instantiated_model" \
  -q --capture=fd --tb=short -o addopts= --disable-warnings'
```

The C++ export regressions feed newly constructed stage/layer fixtures through
the no-OR executable, independent result validator and production Python plan
parser, checking actual duration-model units/prefixes as well as legality.
These are synthetic-data integration checks, not fresh GPU calibration results.

## Earlier Method-only Validation

This record predates the profiling-policy migration. Its quality/retry checks
and counts describe that earlier tree only.


All Python/Megatron work used existing `slackpipe-dev`, mounted at
`/workspace/Megatron-LM`, Python `/opt/venv/bin/python`. No-OR C++ used the same
container. OR C++ used the existing `slackpipe-ortools-runtime:local` image,
OR-Tools 9.15.6755 in `/opt/or-tools`. No replacement environment was created.
Installed training versions: PyTorch 2.12 development build, CUDA toolkit 13.2,
Transformer Engine 2.16.0 development build, Mamba 2.3.2.post1.

| Check | Result and evidence scope |
| --- | --- |
| No-OR configure/build/CTest | Four CTest targets pass, including explicit CP-SAT unavailable/no-plan handling |
| OR configure/build/CTest | Four targets pass; script validates all three plan generators and activation-cap example |
| Pure joint model | Zero external hints, conservative horizon, actual profile-duration capacity bounds, FIFO, no fallback, deadline/no-solution cases |
| Tiny joint optimum | Uniform optimum 18 ticks; asymmetric affine + communication case agrees with independent exhaustive partition/order enumeration |
| Search behavior preservation | Fixed 40,000 heuristic evaluations: split, orders, all operation times, non-timing events identical, 129 ticks; fixed four OctoPipe iterations: same split/orders/times, 147 ticks |
| Registry/runner/receipts/plots | 106 pass; two GPU exact-count cases deliberately deselected |
| Plan/cost/quality/collection/diagnostic suite | 235 pass, five GPU skips; retry/filter policy, v1/v2 parser compatibility, trace correlation and imported diagnostic helpers covered |
| Runtime/manifest test collection | Seven CPU checks pass; four GPU cases fail at device initialization, 17 skip for CUDA/rank requirements. Not claimed as GPU validation |
| Abstract B4/N4/W2/L8 export smoke | All three methods pass independent C++ validation and production Python plan parsing: 32 operations each, cyclic placement |
| CPU five-panel plotting | Real PNG/PDF generation and JSON report, capture/rank/iteration alignment, wrong-label rejection |
| Formatting/imports | Container `uv run ... isort`, Black; direct normal-chain import regression; no imports of retired launchers or test implementations |
| Static checks | `git diff --check` and Ruff F checks on the normal Python execution chain pass |

These are distinct suites, not a summed pass count inflated by repeat runs.
Migration failures were fixed and suites rerun: retired BFS CLI in profile-quality
tests, implicit N in isolated-profile tests, old receipt/method expectations and
five-panel plotting assumptions. That earlier method-only step did not change
quality thresholds or filtering; the current profiling follow-up explicitly
removes those rules as described above.

The smoke profiles use abstract costs, not new calibrated GPU measurements.
The pure CP-SAT smoke reached raw model OPTIMAL=51 ticks with zero hints, replay
51, independent workload bound 48 and no fallback. Heuristic replay was 51 and
OctoPipe replay 54. No performance claim follows from these values.
Behavior comparison uses the captured pre-refactor binary; its embedded older
build-commit field is not a replacement for the saved dirty-source snapshot.

Reproduction commands for builds, CTest and each public method are in the
five-method guide. The main Python groups were:

```bash
python -m pytest -q --capture=fd --tb=short -o addopts= \
  tests/unit_tests/pipeline_parallel/test_slackpipe_methods.py \
  tests/unit_tests/pipeline_parallel/test_slackpipe_eval.py \
  tests/unit_tests/pipeline_parallel/test_slackpipe_eval_receipts.py \
  tests/unit_tests/pipeline_parallel/test_slackpipe_campaign_plot.py \
  tests/unit_tests/pipeline_parallel/test_slackpipe_nemotron_workflow.py \
  -k 'not exact_count_matches_instantiated_model'
```

The 235-pass group contains `test_slackpipe_{plan,cost_profile,profile_quality,
group_quality,outliers,collection,figure_trace,topology,isolated_profile,
iteration_inspection,lifecycle,stage_measurement,transfer,linear_identification,
profiler_diagnosis,overlap_diagnostic,compute_timing}.py` with the same pytest
capture/addopts settings. Execute through `docker exec` as in the guide.

Local ignored evidence includes `behavior_comparison.json`, old/new solver
outputs, `cpsat-final.json`, `cpsat-final.validation.json`,
`cpsat-final.plan.json`, `parser_smoke.json` under the review directory.
The two no-OR artifact smoke receipts are in container
`/tmp/slackpipe-pruning-{heuristic,octopipe}-20261020/receipt.json`.
These paths identify this review, not durable published experiment artifacts.

## Limits and Deliberate Differences

The existing container currently reports CUDA unavailable/device_count=0 and
cannot initialize NVML. Accordingly PP=1/PP=2 numerical equivalence, P2P/RMA
execution/lifecycle and GPU execution of the three newly generated plans are
**outstanding**. No throughput sweep, CUDA experiment or historical Nemotron
prediction revalidation was performed. CPU parser/replay success is not a
substitute for these checks.

Native offline `one_f_one_b`, `interleaving_stats`, BFS and forced-phase OctoPipe
remain diagnostic utilities, never extra public methods. Diagnostic fixture
tools may deliberately import test fixtures; normal production model/training
imports do not. The heuristic retains internal coupled/PD-control APIs and
generic predicate/deadline hooks for regression, with no CP-SAT caller.
The offline CSV plotter `slackpipe/scripts/plot_schedule_comparison.py` and the
historical result readers `activation_diagnostics.py` and
`analyze_interleaving_heatmap.py` also remain outside the active campaign.
The latter readers retain their legacy input schemas; current activation
information is the structured `activation_analysis` in result v2, not a
relabelled legacy canonical-result record.

Old names and receipt contracts are rejected rather than migrated. This is
intentional to prevent relabelling old hinted CP-SAT runs. Historical records
must use their original code snapshots. The narrowed interface does not add
GPU capabilities, alter runtime scheduling, or certify current large-model
performance.
