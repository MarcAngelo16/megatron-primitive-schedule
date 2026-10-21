> Historical scope: this document records pre-five-method code/results. Commands and method IDs below are not current entry points. Use the [current workflow](slackpipe_five_methods.md); historical results keep their recorded commit and labels.

# Artifact Readiness Audit

Audit date: 2026-09-29. **Publication scope: software fixes and audit checkpoint**,
not a claim that the complete empirical paper artifact has been reproduced.
The owner excluded Nemotron historical-data verification and GPU reruns from
this commit/push gate. They are skipped, not passed; no Nemotron evidence package
is requested or inspected in this continuation. Other software regressions are
rechecked against the exact proposed commit tree, without unrelated local work.
No paper numbers or original measurements were edited. This report does not
promise acceptance by an artifact committee.

## Checkout and Scope

- Base revision: `5d33ac46c2e72fb85a11b976bfd205e69ec7dfd5`, branch `slackpipe/mvp`.
  Initial audit checks used a working tree containing earlier uncommitted
  OctoPipe, workload-bound and profiling-quality work plus the audit fixes.
  Publication excludes unrelated OctoPipe/placement/plotting/scale-sweep work.
  The base SHA alone does not identify the initial tested tree; its minimal-run
  receipt fingerprints selected source files and records dirty status,
  build information and binary SHA-256.
- `fork/slackpipe/mvp` initially matched this base. `git fsck --full --no-reflogs`
  found dangling objects but no corruption. Existing changes were preserved.
- `slackpipe/` is tracked monorepo source, not a nested repository. Generated
  results, builds and audit logs remain ignored. Only the reviewed software
  fixes, regression tests, tools and documentation are in the publication
  scope. The destination is `fork/slackpipe/mvp`; origin and other branches
  are not modified. Historical evidence limitations below remain explicit.

## Fixes and Contracts

1. `alternating_solver.cc` previously accepted equal-cost lexicographic swaps.
   It now accepts only strictly smaller makespans between feasible incumbents.
   Obtaining the first cap-feasible solution is initialization, not a tie move.
   Stopping is after the first **complete round** without improvement, not the
   first unsuccessful phase; deadline and round limits still apply. Acceptance
   and stopping rules are exported, and the method contract hash changes.
2. Profile and abstract joint models use the actual `2*B*N` duration variables
   for W worker-load constraints and one global capacity constraint. They do not
   multiply their already complete operation sum by B again. Dependencies,
   FIFO, resource serialization and early-incumbent policy are retained.
3. Independent workload bounds follow the loader's integer prefix rounding,
   stage-role bias and zero-duration semantics. Partition-dependent affine
   costs use an all-partitions minimum, not incumbent work. Conditional
   fixed-order/fixed-partition bounds cannot strengthen the global joint bound.
   Raw solver bounds/statuses remain separate; unavailable bounds remain null
   in canonical reports and their top-level copies. Effective gap is
   `(feasible_makespan - effective_bound) / feasible_makespan`, or null without
   a valid incumbent. A smaller gap never promotes FEASIBLE to OPTIMAL.
4. Existing quality-gate work was inspected and exercised: population CV >10%
   (the existing project convention), comparable-group median ratio minus one
   >30%, or phase RMSE / mean observation >15% rejects a calibration. At least
   three iterations per group/stage/phase are required. Comparability requires
   the same layer range, role, worker and run context, not just the same stage
   number. All thresholds are configurable heuristics. Fresh-process collection
   retries the complete partition set once; a second failure blocks automatic
   solver use. Events keep group/attempt IDs, both candidates and diagnostics
   are retained, and no slow samples are removed. Direct manual C++ use of old
   profiles is not an automatic quality certification.
5. Added an immutable minimal solver-to-runtime receipt and a read-only
   historical evidence inventory. README now warns about evidence limitations
   and uses the quality-gated calibration entry. The new reproduction commands
   do not depend on a personal username, UID or host checkout path.

See [bounds](slackpipe_workload_bounds.md), [quality gate](slackpipe_profile_quality.md)
and [method contracts](../slackpipe/docs/evaluation_methods.md) for details.

## Publication Recheck

The proposed index was exported to an independent detached worktree, excluding
all unrelated local additions. Its 2,921 regular files matched the index by Git
blob hash. Tested tree: `67929deb3034cbbdcef9a74367d46e3bee663445`; only this
report changed after testing. This is a tree identifier, not a release commit.

- No-OR Release build and CTest: **4/4 targets passed**.
- OR-enabled Release build and CTest: **4/4 targets passed**, plus the validation
  script's **14/14** evaluator-validated tiny canonical CLI cases. This covers
  costs, profile capacity constraints, strict alternation, bounds and fallback.
- Python: **137 passed, zero skipped**, covering `test_slackpipe_profile_quality`,
  `test_slackpipe_cost_profile`, `test_slackpipe_plan`, `test_slackpipe_eval`,
  `test_slackpipe_eval_receipts`, `test_slackpipe_nemotron_workflow` and
  `test_slackpipe_collection`. The Nemotron-named tests exercise configuration
  and workflow fixtures, not the historical evidence or an 8B training run.
  The retry tests use real fresh CPU subprocesses and synthetic measurements;
  campaign tests verify that a second failure prevents solver invocation.
- No-GPU minimal reproduction: C++ exporter -> evaluator -> production Megatron
  plan parser passed, with B=4/N=4/W=2/L=8 and 32 operations. All recorded
  artifact SHA-256 hashes were rechecked. No new GPU training/profiling run was
  performed; earlier GPU results below are not reclassified as this recheck.
- Import sorting, Black checks on all 16 Python files and staged diff checks
  passed. No generated data, build products or unrelated solver features are
  included in the commit.

Local receipts: `slackpipe_experiments/artifact_publication_20260929/`, including
`python-regressions-final.xml` and `minimal_final/receipt.json` (SHA-256
`d459a1ef639a7897664c07bde7b0d01a1313e96e1b627519ecea3774e580dd69`). Initial
worktree setup attempts encountered container UID/cache/temp-file ownership
issues; the successful runs used the prepared container's normal user and a
process-local Git trust setting for this worktree. The failed minimal attempt
was retained, not overwritten. These receipts are local validation records,
not redistributed historical performance data.

## Initial Audit Checks

All execution used Docker, without host dependency installation.

| Check | Result / scope |
| --- | --- |
| No-OR Release build and CTest | 4/4 targets passed; does not validate CP-SAT |
| OR-enabled Release build and CTest | 4/4 targets passed; OR-Tools 9.15.6755, variable cumulative demands supported |
| OR validation script | 14 validated tiny CLI cases: seven canonical methods with/without memory cap |
| Bound regression | Uniform/v1/v2 known optima 18/42/54 unchanged; profile constraint count W+1; bound <= optimum; partition dependence, rounding, overflow and unavailable cases |
| Strict-alternating regression | B=1/N=2/W=2/L=4: equal-cost [1,3] proposal rejected, [2,2] retained, both phases run before no-improvement stop |
| Python targeted suite | 137 passed: quality/retry, profile, plan, generic receipts/campaign, Nemotron workflow, collection |
| Manifest/heterogeneous PP=1 suite | Seven passed, including exact PP=1 numerical equivalence; the PP=2 case was skipped in this single-rank invocation and tested separately with two ranks |
| Minimal C++ -> plan -> PP=2 P2P | B=4/N=4/W=2/L=8; actual two-GPU execution, rank orders match plan; max absolute differences for initial parameters/loss/all gradients/post-SGD parameters all 0 |
| Heterogeneous PP=2 P2P | Separate two-rank invocation passed on both ranks, covering the case skipped by the single-rank suite |
| PP=2 RMA equivalence | Homogeneous and heterogeneous two-rank tests passed, two tests per rank; no claim of multi-node or 8B validation |
| Deadline CLI | 1e-6-second joint budget: FEASIBLE/NOT_RUN fallback, makespan 54 ticks, independent/effective bound 48 ticks, gap 11.1111%, not proven optimal |
| Historical strict reanalysis | 2,592 main rows and four oracle rows passed validation; no solver or GPU rerun implied |
| Repository checks | `git diff --check`, shell syntax check and personal-path scan passed |

The GPU host has two RTX A2000 devices (6,082,068,480 bytes each), not the
four-GPU Nemotron-H 8B environment. PyTorch is
`2.12.0a0+0291f960b6.nv26.04.48445190`, CUDA 13.2, NCCL 2.29.7.
The local images used were:

- `megatron-slackpipe:0.18.2`:
  `sha256:c2712411d7b00471551cbe0c1980fcfb66dd86d33ff4afab9904e009d1e1308c`
- `slackpipe-ortools-runtime:local`:
  `sha256:971dddbe40c7a8d25a53dbbd8f46c19afa96c06b506c4ed0286d6eb2fffa946c`

These are local image IDs, not downloadable registry digests. Environment
provisioning remains an external prerequisite described in the README.
No controlled before/after solver-speed comparison was performed. No new
Nemotron profiling, 8B training benchmark or full main CAL campaign was run.

Local evidence is under `slackpipe_experiments/artifact_audit_20260929/`:
`minimal_release_check/receipt.json` records the final successful GPU run
(SHA-256 `6b91df6b856408b5e29fc209e91d8a34e833189ed8b7512d410253dacce32a01`;
all artifact hashes in the receipt were independently rechecked);
`minimal_final/receipt.json` retains an earlier failed build attempt caused by
the test shim's unsupported vector assertion (subsequently fixed). Failed
attempts were not overwritten. CTest logs are in the respective build trees.

## Historical Evidence and Limitations

The main historical run is
`slackpipe/results/cal_main_71489e6_20260811T062130Z` (about 1.1 GB).
It contains 2,592 native result JSONs, configuration manifests, solver operation
timelines, logs, binary and derived tables. These are **simulation** results;
there is no GPU-time/profile/plan chain to infer from abstract operation ticks.

- All main rows record source `71489e682cb9007d3713d2119ed3f0d0eea87b07`.
  That commit is absent from this monorepo's object database. The import does
  not by itself reconstruct the exact historical source tree.
- Original main manifest hash:
  `50968370d9612a5aeceee891e1c25555be35677d4a6f38884949de3d9096cce4`.
- Reanalysis retained 1,296 uncapped and 1,296 equal-memory rows, 96 fallbacks,
  zero reported cap violations and 1,665 feasible-but-not-proven-optimal rows.
  Shared historical summary values agree numerically with the saved summaries;
  extra fields added by the current analyzer prevent byte-identical JSON.
  Aggregation is median across seeds per workload/method, then geometric mean
  across workloads; the old joint/alternating ratios are not GPU speedups.
- The four-row oracle run
  `slackpipe/results/cal_oracle_340dbf9_20260811T073223Z` records missing source
  `340dbf99aa56ca15b4edaea372e1020ce727d80a`. Its oracle is fixed-order partition
  optimality, not proof of the globally optimal joint schedule.
- Read-only inventories fingerprint JSON/JSONL/CSV/log/Markdown files. Their
  `records_sha256` values are
  `b363c278048f844da36505fb28785ecc1e99c1f7d707408485e8f11d2532582e` (main) and
  `66066ca07e6da0d7c4a486e1f48fca65a056b0b73938a4f6ba9a4b1dd44c2516` (oracle).
  These are inventory-list hashes, not source or binary hashes and not a
  semantic certificate. Original data were opened read-only.

Neither run has tracked raw data or a recorded immutable public archive location.
Thus another researcher cannot obtain all evidence from this repository alone.
Historical alternating rows may contain equal-cost exchanges; their old method
contract cannot be relabeled as the new strict policy. A current strict-policy
comparison must be rerun before associating those numbers with this version.

**Nemotron historical data and prediction accuracy were not revalidated in this
audit.** Their verification and GPU reruns are explicitly excluded from this
software publication gate, not marked as successful. The following records the
limitation already established by the initial audit; no additional evidence
package was received or inspected for publication.

The original six-group Nemotron-H 8B calibration, ~42% fit errors and
818.391-ms prediction could not be located with their raw events/profile/source
provenance in this checkout. The reported 762.110-ms bound and 93.12% utilization
are conditional arithmetic from user-supplied aggregates, **not** a replay of
that profile. Synthetic shifted-group tests establish rejection behavior, not
that the physical cause of the historical stalls was found or removed. No
actual GPU profiling retry was executed in this audit.

Before separately declaring the full empirical artifact complete, supply the
historical source/data
archives with hashes and accessible locations (or rerun an explicitly new
campaign), regenerate the strict-alternating comparison, and supply a
quality-passing real-system calibration/execution chain for any retained 8B
claim. These are not claims made by this software checkpoint. Four-GPU/large-memory
checks remain hardware-gated. Do not change paper
numbers to fit this audit, or interpret successful unit tests as those results.

## Minimal Reproduction

From the monorepo root, with the prepared `slackpipe-dev` container mounted at
`/workspace/Megatron-LM`:

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev /opt/venv/bin/python \
  -m tools.slackpipe_artifact_smoke \
  --output slackpipe_experiments/minimal-artifact
```

Choose a fresh output directory. This runs the no-OR C++ build, evaluator
and production Python parser path without GPU execution. Adding `--gpu` requires two
visible GPUs and fails rather than accepting a skipped equivalence test.
The `slackpipe.artifact_smoke.v1` receipt includes configuration, command/log
records, source/build/binary hashes, generated plan/result hashes, runtime
trace hashes and numerical differences. Solver makespan is abstract **ticks**;
command wall seconds include subprocess/setup work and are not GPU iteration
measurements. Each runtime trace must match its rank's generated plan order.

The README provides the no-OR and full OR CTest commands. To repeat historical
analysis without editing its inputs:

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev /opt/venv/bin/python \
  slackpipe/scripts/analyze_cal_results.py \
  --manifest slackpipe/results/cal_main_71489e6_20260811T062130Z/manifests/cal_main.jsonl \
  --results-root slackpipe/results/cal_main_71489e6_20260811T062130Z/results \
  --output-dir slackpipe_experiments/historical-reanalysis --strict --format csv,json
docker exec -w /workspace/Megatron-LM slackpipe-dev /opt/venv/bin/python \
  -m tools.slackpipe_artifact_inventory \
  --run slackpipe/results/cal_main_71489e6_20260811T062130Z \
  --output slackpipe_experiments/historical-inventory.json
```

These commands require the original run directory, which is currently local
only. An inventory refuses missing evidence and existing output paths.

## Full Current-Policy Campaign

Build and test the OR-enabled binary as in the README, then run the following
in the OR-Tools container (same checkout mount). This is a **new** experiment,
not a command to recreate unavailable historical source. The explicit 30-second
budget matches the historical protocol; the current default is different.

```bash
python3 slackpipe/scripts/generate_cal_manifest.py --profile main \
  --binary slackpipe/build/release/slackpipe_cli \
  --output slackpipe_experiments/strict-main/manifests/main.jsonl \
  --time-limit-seconds 30 --solver-threads 1 --seeds 0,1,2
python3 slackpipe/scripts/run_cal_manifest.py \
  --manifest slackpipe_experiments/strict-main/manifests/main.jsonl \
  --binary slackpipe/build/release/slackpipe_cli \
  --output-root slackpipe_experiments/strict-main/results --jobs 1 --fail-fast
```

Use a fresh campaign directory; preserve raw rows, logs, manifest metadata,
build-info, binary, exact source snapshot and method contract hashes. Then run
`analyze_cal_results.py --strict` in `slackpipe-dev`, pointing at the new manifest
and results with a new analysis output. Do not mix old/new contracts or replace
original tables. This full campaign was **not executed** during the audit.

For measured workloads, follow the [generic campaign guide](slackpipe_real_system_eval.md)
and [quality-gated collection](slackpipe_profile_quality.md): collect raw grouped
events, retain both attempts, fit a passing profile, solve with that profile,
export/validate the plan, execute that exact plan, and preserve independent
benchmark samples and trace captures. Keep profile/model/configuration/plan
hashes in stage receipts and final summaries; compare like timing definitions,
units, warmup and aggregation. Model weights, compatible hybrid dependencies,
four suitable GPUs for PP=4/8B and the original evidence archive are external
requirements, not artifacts supplied by the small two-GPU correctness fixture.
