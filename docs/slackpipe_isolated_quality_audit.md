# Historical Isolated Quality Dispatch Audit

> Superseded by the [current profiling contract](slackpipe_profiling.md).
> The old isolated spike/CV gates and stability retries below have been removed.
> Current layer profiling uses all legal samples and structural reconstruction.
> Historical GPU measurements and paper-related evidence below are preserved
> under their original policy; this refactor does not revalidate them.

## Findings Before Editing

Inspected baseline: `da48a88501a72ea401a27b7003f19b30ab9d2a5d`.
The requested `calibrate-e3d718041352/profiling_attempts.json` was not available
in the checkout, attachments, or container `/workspace` and `/tmp`. Consequently
the exact failing classes/rules for its attempt0 and attempt1 remain **unknown**.
Neither A (isolated CV failures) nor B (legacy pipeline failures) can be asserted
for that external run. The supplied command and exception alone are insufficient.
No historical receipt or experiment data was modified.

The inspected code already dispatched correctly labeled isolated profiles away
from the legacy statistical evaluator:

1. `Experiment.ensure("calibrate")` called the shared `collect_with_retry` runner.
2. `_worker` replaced torchrun with the single-process isolated collector, but
   unnecessarily constructed/forwarded every legacy `--quality-*` flag.
3. `collect` used only `IsolatedPolicy(max_cv=args.isolated_profile_max_cv)` and
   passed isolated warmups/iterations directly to `measure_unit`.
4. The collector reported isolated sample count/CV failure through
   `cost_profile.quality.json`. The runner retried once if it said rerun_required.
5. Passing profiles reached `require_profile_quality`, which dispatched on the
   estimator to `isolated_profile.require_quality` and returned before
   `assess_profile`, cross-group shifts or `group_consensus`.
6. Two reported failures produced the same generic `ProfilingQualityError` used
   by legacy profiling, without including the per-class causes in the exception.

Thus the confirmed defects were confusing legacy flag forwarding and inadequate
failure diagnostics, not evidence that the isolated CV threshold was 0.10 or
that the requested external run failed pipeline consensus.

## Changes

- `tools/run_slackpipe_eval.py`: builds legacy policy/flags only in the legacy
  branch. Isolated context identity and its 0.15 threshold are unchanged.
- `tools/slackpipe_profile_quality.py`: explicit isolated attempt validation,
  recomputed raw-data report for either outcome, direct isolated selected-profile
  validation and candidate/selected hash agreement. Legacy statistics unchanged.
  Retry remains shared and limited to one additional complete subprocess.
- `isolated_profile.py`: additive per-unit/phase report and human-readable failure
  description. Validating an unstable candidate is allowed only for diagnosis;
  the default and selected-profile validators still reject it.
- `tools/slackpipe_isolated_profile.py`: writes the expanded quality sidecar and
  records executed F/B counts in every raw unit. No timing/runtime changes.
- Tests and the calibration guide cover dispatch isolation, diagnostics, actual
  measurement-loop counts and unchanged policy behavior.

No solver code, quality thresholds, group filtering, default estimator, sample
selection rule or runtime communication path was changed. The unchanged rule
allows at most one >10x-median sample exclusion with at least 30 samples, so the
maximum possible rejected fraction is 1/30. A genuine CV above 0.15 still fails.

Quality sidecars and each attempt's `quality.units` contain class signature,
member indices, representative/config signature, independent raw/accepted F/B
arrays and statistics, counts/discarded indices/fraction, limits, status and
failed rules. `raw_data_path` identifies the original timing data. There are no
cross-group or consensus metrics. Old valid isolated profile hashes remain
readable; reports are derived without rewriting those profiles.

## Fresh Nemotron-H 4B Collection

Executed on one RTX A2000 6 GiB in `slackpipe-dev`, seq=1024, micro batch=1,
BF16, TP=DP=CP=1. Target metadata: PP4/N8/B8, interleaved, nccl-p2p. Actual
execution: one unit at a time, single-rank Gloo metadata groups, no pipeline
schedule/VPP/P2P/NCCL work, no full model construction.

The command deliberately included compatibility `--warmups 5 --iterations 10`,
plus `--isolated-profile-warmups 20 --isolated-profile-iterations 30
--isolated-profile-max-cv 0.15`. All five units recorded 20 warmup pairs and 30
measured pairs, with sample IDs 20..49, plus one untimed initialization forward.
Attempt 0 passed; no retry was needed, and **no samples were rejected**.

| Unit | F median ms | F CV % | B median ms | B CV % |
|---|---:|---:|---:|---:|
| Mamba | 5.115776 | 0.329719 | 11.327488 | 0.335396 |
| MLP | 5.978912 | 0.798981 | 12.103248 | 0.812245 |
| Attention | 2.198880 | 0.682496 | 5.272016 | 0.435196 |
| Embedding | 0.111536 | 6.231180 | 3.189760 | 0.269944 |
| Final norm/head/loss | 44.543776 | 0.437990 | 57.423872 | 0.690589 |

Three unique partitionable classes, plus two boundary units. Collection time
16.469 seconds; peak allocated 3,354,677,760 bytes, reserved 3,802,136,576 bytes.
These are A2000 isolated measurements, not A100 or PP4 iteration results.

Selected profile (relative to repository root):

```text
slackpipe_experiments/isolated_quality_20261001/nemotron4b/attempt0/worker/cost_profile.json
cost_profile_hash: 48a11302ac8e7b77353d046bcc3f6b7287d8173bf2292ddb15b5b67705c855f0
```

`profiling_attempts.json` in the same collection root selects that exact file;
raw timings, summary and detailed quality sidecar remain beside it. Generated
artifacts are ignored, not bundled into the source commit.

The unchanged OR-enabled SlackPipe CLI successfully consumed that selected path
with B8/N8/W4/L52, communication=0, five-second limit and require-optimal=false.
It returned **FEASIBLE**, source `joint_cpsat`, no fallback, not globally optimal.
Predicted makespan was 2212.028 ms (not GPU-measured iteration time). The exported
v2 plan passed the production parser and profile/manifest provenance validation.

## Checks and Remaining Limits

- 157 quality/receipt/outlier/group tests passed in the existing container.
  Tests make the legacy evaluator and consensus function raise if an isolated
  profile reaches them; both stable and unstable isolated retry paths pass.
  The final focused isolated-profile/receipt rerun passed 94 tests (overlapping
  coverage, not an additional independent total).
- Independent F/B rejection, CV 0.12 pass versus 0.16 fail, exactly 20/30 loop
  counts, no legacy flag forwarding, estimator switch, receipt reuse and at-most
  one retry are covered. Failed isolated candidates cannot select a profile.
- 45 plan/cost/manifest/construction tests passed; one two-rank case skipped in
  the single-process invocation. The small PP1 heterogeneous test was exactly
  equivalent for loss, gradients and parameter updates.
- A separate two-GPU PP2 FP32 hybrid test passed on both ranks using the existing
  solver-exported synthetic isolated-profile fixture. Maximum absolute differences
  for initial parameters, loss, gradients and post-step parameters were all zero.
  This checks solver-plan/runtime compatibility, not Nemotron prediction accuracy.
- The missing external receipt remains the blocker to diagnosing its two failed
  attempts. This local passing run does not explain or invalidate those failures.
- No four-GPU training, A100 rerun or pipeline prediction-accuracy claim. Existing
  unrelated overlap-diagnostic and scale-sweep changes are preserved and excluded.
