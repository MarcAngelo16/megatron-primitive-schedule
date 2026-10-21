# Historical Calibration Quality Gate

> Superseded by the [current profiling contract](slackpipe_profiling.md).
> The stability gates, group consensus, raw spike rules, quality receipts and
> automatic retries described below are historical and are not current behavior.
> Current stage profiling uses one affine-residual mask; layer profiling retains
> all legal samples. The original measurements, paths, counts and audit narrative
> below are preserved as historical evidence, not new validation or instructions.

Calibration is a separate synchronized diagnostic workload, not benchmark timing.
Its stage-call wall time includes CPU dispatch, allocation/cache stalls, compute,
and the completion wait; explicit P2P calls and optimizer time are excluded from
the fitted stage costs. Training warmup still executes the complete training step.
Only rare catastrophic raw timing spikes can be excluded from aggregation as
described below. This does not substitute kernel sums, change the fitter, or
establish that allocation stalls are absent. Raw event files are never edited.

## Isolated Timing Spikes

Before averaging microbatches, analyze each calibration group/context, logical
stage and phase independently. With median `m`, `MAD = median(abs(x - m))` and
`sigma = 1.4826 * MAD`, a raw sample is a candidate only if **both**
`x > m + K * sigma` and `x > R * m`. These are strict upper-tail tests; `MAD=0`
requires no division and still requires the ratio test. There is no percentile
clipping, fixed-fraction trimming, or removal of moderately slow samples.

At least 20 raw samples are required by default; below that count nothing is
filtered, and the existing quality checks apply to the unfiltered data.
Accepted microbatches are averaged within each iteration, then the median across
iterations gives the per-operation stage cost, preserving the existing estimator.
For compatibility, diagnostics `samples_ms` store these means multiplied by B;
`raw_samples_ms` store original iteration totals. A partial accepted sum is never
divided by the original B. An iteration with no accepted samples fails quality.

Filtering is not permission to accept an unstable run. For **each stage/phase**,
exceeding any frequency, affected-iteration or consecutive-sample limit below
requires a retry even if cleaned CV is small. Consecutive means adjacent samples
in `(global iteration, microbatch)` order, including iteration boundaries.
Cleaned CV, cross-group consistency and fit error must also pass unchanged.
Isolated spikes alone do not cause a retry or `ProfilingQualityError`.

## Repeated Calibration Groups

The heterogeneous (cost-profile v2) builder now performs a second, **phase-local**
analysis before fitting. Comparable observations require the same logical stage,
exact contiguous range, stage role, worker, measurement context, manifest hash,
ordered layer composition and, when recorded, class counts. The manifest supplies
the composition; supplied composition/counts inconsistent with it are rejected.
The context already binds model/hardware/software/timing configuration. Merely
sharing a numeric stage index is not sufficient. Different ranges or Mamba,
Attention and FFN compositions never vote in the same comparison.

Each distinct calibration-group ID supplies one estimate (median if duplicate
rows exist). For group estimates `x`, compute median `m` and
`sigma = 1.4826 * median(abs(x - m))`. A **two-sided** candidate must satisfy both
`abs(x - m) > K_group * sigma` and `abs(x - m) > D_group * m`.
The detector is applied once to all original groups, not iterated until a tight
cluster appears. It treats high and low anomalies symmetrically.

Candidates are excluded only if at least three groups and a strict majority
remain, the excluded fraction is at most 40%, and the proposed survivors have
`max / min - 1 <= 0.10`. Otherwise no group is excluded and the consensus gate
fails. Two equally plausible clusters or broadly dispersed groups are not
arbitrarily pruned: the unchanged cross-group check can still reject them.
Single-partition observations remain usable under the ordinary sample checks;
they provide no evidence of repeat-group robustness and cannot authorize group
exclusion. The homogeneous v1 collector still measures a single partition.

For each comparable set and phase, the fitter consumes **one median of accepted
group estimates**, with explicit source-row indices. This intentionally gives
repeats one consensus vote instead of weighting an exact range by how many
partitions happened to repeat it. The class/role least-squares fitter itself is
unchanged. Forward and backward use separate accepted sets; rejecting backward
does not remove valid forward measurements. Prefix costs and role biases are
refitted from this data, not merely relabeled as passing. Every original stage
row and its diagnostics remain in `observed_stages`.

Within-group CV is still checked at 10% for **every accepted group**, never on
averaged timestamps. CV alone cannot trigger group exclusion. Sample-level
frequency/run-length guards and missing-provenance checks apply even to excluded
groups. Fit RMSE is checked against all accepted original observations, not just
the median fit inputs. Thus a majority with noisy accepted measurements still
fails, even if its group medians are close. Old profiles without group selection
metadata retain the strict all-observation checks; a receipt alone never removes
previously fitted data.

## Empirical Checks

All limits are configurable heuristics, not theoretical guarantees:

| Flag | Default | Definition |
| --- | --- | --- |
| `--quality-cv` | 0.10 | Population standard deviation / mean of accepted iteration per-operation means (equivalently B-normalized totals). This retains the existing 10% CV convention. |
| `--quality-median-shift` | 0.30 | Maximum / minimum comparable group median minus one. Each group cost is the median iteration total divided by its fixed microbatch count. |
| `--quality-relative-rmse` | 0.15 | `sqrt(mean((prediction - observation)^2)) / mean(observation)`, independently for forward and backward, over all stage/group rows. |
| `--quality-min-samples` | 3 | Minimum complete measured iterations per stage/group/phase; cannot be reduced below three. |
| `--quality-outlier-mad-multiplier` | 10 | K in the robust upper-tail test. |
| `--quality-outlier-median-multiplier` | 3 | R in the simultaneous median-ratio test. |
| `--quality-min-outlier-samples` | 20 | Minimum raw sample count for detection in one stage/group/phase. |
| `--quality-max-outlier-fraction` | 0.02 | Reject when discarded / raw sample count exceeds this fraction. |
| `--quality-max-outlier-iteration-fraction` | 0.20 | Reject when iterations with any rejected sample / all measured iterations exceeds this fraction. |
| `--quality-max-consecutive-outliers` | 1 | Reject when the longest consecutive rejected-sample run exceeds this count. |
| `--quality-group-mad-multiplier` | 3 | K_group, applied to absolute deviation from the group median. |
| `--quality-group-relative-deviation` | 0.15 | D_group, simultaneous relative-deviation floor. |
| `--quality-group-min-survivors` | 3 | Minimum accepted distinct group IDs for any exclusion; cannot be below three. |
| `--quality-group-max-discarded-fraction` | 0.40 | Maximum excluded groups / original comparable groups; must remain below 0.5. |
| `--quality-group-consensus-median-shift` | 0.10 | Maximum proposed survivor max/min minus one to authorize exclusion, separate from the unchanged 0.30 final consistency limit. |

Cross-group comparisons require at least two groups with identical contiguous
layer ranges, stage roles, physical workers, and measurement-context hashes.
The context binds model, microbatches, precision, input shape, warmup/measurement
settings, seed, optimizer learning rate, software, GPU identity, allocator, and
timing definition. Different partitions are compared only where those keys match.
Missing provenance or inadequate samples fail closed. If there are no comparable
groups (for example, the single-partition v1 collector), diagnostics explicitly
record `not_assessed_no_comparable_groups`; no cross-group consistency is claimed.
Fit checks require at least two observations per phase. For v2, predictions use
the actual heterogeneous prefix costs and role biases; for v1, the shared slope
and stage biases. A v1 exact fit is not independent evidence of generalization.

## Retry and Artifacts

`tools.run_slackpipe_eval calibrate` and its campaign callers launch the entire
partition set in fresh torchrun processes. On a quality failure, they repeat the
entire calibration **once**, with the same configuration and thresholds. A second
failure stops the pipeline before solver invocation. Retry configuration and
hardware identity must match the first attempt. There is no fastest-attempt
selection, raw-sample deletion, threshold relaxation, or unbounded retry.

Each calibration stage stores:

```text
profiling_attempts.json             # configuration, timestamps, attempts, selection
attempt0/worker/
  calibration_events.json           # all raw events, with explicit group/attempt IDs
  observations.json                # stage/group rows and all iteration samples
  cost_profile.candidate.json       # retained even when rejected
  cost_profile.quality.json         # metrics, issues, thresholds, rerun command
  cost_profile.json                 # created only when this attempt passes
attempt1/worker/                    # created only after a quality failure
```

The quality receipt is `slackpipe.profile_quality.v3`. Its status is `passed` or
`rerun_required`; issues carry rule, threshold, value, phase, group, stage, layer
range, worker, observed costs, context hash and original data path. Profiles embed
the receipt and are rehashed. The attempt ledger identifies the only selected
profile. Original attempt files are never replaced. Group IDs are attached to
events before gathering/aggregation, not inferred from event order.
The embedded receipt uses `quality_schema_version` instead of `schema_version`
to remain compatible with the existing C++ cost reader's first-key lookup.

New fields include global `raw_sample_count`, `accepted_sample_count`,
`samples_discarded`, `discarded_fraction`, and per-stage/phase `sample_groups`
with the same counts plus `raw_cv`, `cleaned_cv`, affected iterations and longest
outlier run. CV here is across iteration means, **not** across raw microbatches.
`rejected_samples` retain group/attempt/context, stage, layer range/role, worker,
rank, global iteration, microbatch, original `elapsed_ms`, median, MAD,
robust sigma, both thresholds, rejection reason and `raw_data_path`.
The source event JSON retains every sample, including rejected ones.

Each observation records `outlier_filter.policy` and accepted counts per global
iteration. Quality policy changes require reaggregation, not simply reapproving
previously filtered observations. All policy flags and the quality schema enter
the generic calibration context/hash; downstream receipts are invalidated when
they change. The existing receipt/context-v2 schema is unchanged. Legacy quality
v1/v2 profiles can still be explicitly validated under their stored policy but are
not silently reused in a new v3 calibration context. Reports from old aggregated
rows mark `raw_sample_status=unavailable_or_partial_legacy_aggregation`; zero
reported discards there does not certify that original raw samples were checked.

Synthetic example: 239 samples of 10 ms and one of 356 ms, B=8, 30 iterations,
give 240 raw / 239 accepted / 1 discarded (0.4167%) for that stage/phase.
Raw iteration CV is about 0.679 and cleaned CV is zero; stage cost remains
10 ms. The profile passes without retry and can continue to solver/benchmark.
This is a regression fixture, not an 8B GPU result.

`group_filter` records the policy, comparable-set `comparisons`, phase-local
`accepted_row_indices`, and exact `fit_observations`. `rejected_groups` records
group ID, stage/phase/range/composition/context/worker, original estimate,
consensus median, original median/MAD/sigma, absolute threshold, reason and raw
data paths. Counts `raw_group_count`, `accepted_group_count`,
`rejected_group_count`, `group_discarded_fraction` count **stage/phase group
estimates**, not unique whole partition runs. Every comparison contains raw and
cleaned CV **across group estimates**, raw/cleaned median shift and consensus
cost in ms. These are separate from iteration CV in `sample_groups`.

Sample counts retain their level-1 meaning. `group_excluded_sample_count` counts
level-1 accepted samples excluded at level 2; `effective_accepted_sample_count`
counts samples supporting accepted groups. A sample is not counted as rejected
twice. Every sample-group report also marks `included_in_group_consensus`.
Revalidation recomputes selection and fit inputs, checks the stored policy and
row lineage, and applies the original fit-residual test to the actual prefix/
role model. Changing a threshold or mask requires rebuilding the profile.

Synthetic backward example `[19, 19.2, 18.9, 19.1, 13.5]` excludes only 13.5,
retains four groups and uses their 19.05 ms median as the fit observation.
`[19, 19.2, 18.9, 13.5, 13.8]` retains a strict 3/5 majority under defaults.
`[14, 16, 18, 20, 22]` and `[14, 14.2, 20, 20.1]` fail without invented consensus.
The reported `[19.053, 18.799, 20.045, 13.755, 15.995]` is not promised to pass:
the broad original MAD can prevent exclusion, and accepted-group CV failures
remain blockers. These heuristics are not a guarantee of physical contamination.

Example warning (synthetic illustration, not a new GPU result):

```text
Profiling measurements are inconsistent. Rerun profiling before using this cost model.
rule=cross_group_median_shift phase=backward threshold=0.30 value=1.21774
comparable group medians: 24.8 ms, 55.0 ms
```

## Usage

Run the generic calibration inside the existing container; thresholds also enter
the immutable calibration receipt context:

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev /opt/venv/bin/python \
  -m tools.run_slackpipe_eval calibrate \
  --model-config configs/slackpipe_eval/llama_8b.json --pp 2 --logical-stages 4 \
  --output slackpipe_experiments/quality-run \
  --calibration-warmups 20 --calibration-iterations 10 \
  --quality-cv 0.10 --quality-median-shift 0.30 --quality-relative-rmse 0.15
```

Apply the same two-attempt wrapper to a standalone collector (small heterogeneous
fixture shown; full Nemotron requires the hardware specified by its own runner):

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev /opt/venv/bin/python \
  -m tools.slackpipe_profile_quality --collect-output slackpipe_experiments/quality-fixture -- \
  /opt/venv/bin/python -m torch.distributed.run --standalone --nproc-per-node 2 \
  tools/slackpipe_heterogeneous_experiment.py \
  --output-dir slackpipe_experiments/quality-fixture \
  --warmup-iterations 20 --iterations 10
```

The legacy Nemotron driver and homogeneous experiment driver also use this
wrapper and gate their solver calls. Intentional M1/M2 ablations in
the homogeneous driver validate the source calibration and its digest; they are
not relabeled as accurate measured fits.

Standalone workers do not recursively spawn torchrun. They retain rejected
candidates, emit `rerun_required` diagnostics and a fresh-output collection
command, withhold the default profile, and exit nonzero after cleanup. Offline `tools.slackpipe_hybrid fit`
requires `--profiling-rerun-command '<original collection command with fresh output>'`:
refitting the same observations is not a profiling retry. Old profiles without
explicit event provenance/quality receipts are not automatically authorized.
The gate does not rewrite historical measurements or constrain manual direct
C++ solver usage outside these Python/shell experiment entry points.

The checks diagnose inconsistency, not its physical cause. GPU contention,
allocation/cache effects, thermal/clock changes and other stalls still require
separate allocation/timeline diagnostics.

The generic collector's `run_partition(..., collect="calibrate")` does not
construct a torch profiler: profiler windows and trace serialization run only
in the separate timeline/memory path. In `_run_with_slackpipe_cost_calibration`,
the pre-call CUDA synchronization precedes the timer; the completion
synchronization is inside it; event bookkeeping follows the stop timestamp.
Thus no trace flush or profiler transition is evident in this calibration path.
Without the failing run's raw traces, the reported 280-360 ms spikes cannot be
attributed to those mechanisms or ruled out as GPU/system stalls. Timing
boundaries and profiler functionality are unchanged.

For backward, the measured closure is `backward_step`; pipeline input/gradient
preprocessing and post-step gradient-sync bookkeeping are outside that closure.
The calibration configuration disables P2P overlap and does not enable activation
recomputation. Each partition is rebuilt and independently warmed up with full
steps; cache clearing occurs between partitions, not between timed stage calls.
Per-partition allocator/state/clock effects may still change backward costs.
Without raw traces we cannot correlate the reported group deviations with those
effects, pipeline fill/drain, or communication. No timing-boundary change is
justified by the currently available evidence.

The supplied Nemotron-H 4B attempt-ledger path was absent from both this checkout's
host and `slackpipe-dev` during the 2026-10-01 validation. The two 6 GiB A2000 GPUs
cannot validate the full target here. Synthetic regression cases are explicitly
separate from a rerun or a reanalysis of that historical calibration.

Regression checks (existing container, no 8B model allocation):

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev /opt/venv/bin/python -m pytest \
  tests/unit_tests/pipeline_parallel/test_slackpipe_outliers.py \
  tests/unit_tests/pipeline_parallel/test_slackpipe_group_quality.py \
  tests/unit_tests/pipeline_parallel/test_slackpipe_profile_quality.py \
  tests/unit_tests/pipeline_parallel/test_slackpipe_eval_receipts.py -q
```

## Validation Scope (2026-10-01)

- Main regression suite: 217 passed; collection suite: 11 passed. Four PP=2
  numerical-equivalence tests passed separately, covering homogeneous,
  heterogeneous and hybrid models. FP32 differences were zero; BF16 used the
  existing tolerances. Two PP=4 tests remain unexecuted on the two-GPU machine.
- Synthetic two-level/campaign tests verify that accepted consensus reaches the
  real C++ exporter and the mocked benchmark orchestration without retry;
  broad/ambiguous/noisy survivors stop after at most one retry. The runtime
  solver-to-PP=2 smoke uses an abstract-cost plan, not a rejected measured profile.
- A tiny hybrid FP32 calibration ran at PP=2, N=4, B=8, sequence length 32,
  20 warmup and 10 measured iterations per partition. Both attempts had 3840
  raw samples, zero sample exclusions, and respectively two and three group
  exclusions. Remaining quality failures correctly withheld the default profile.
  Accepted-original-observation forward/backward relative RMSE was 38.86%/29.98%
  initially and 37.02%/31.46% on retry, still above the unchanged 15% limit.
  No solver/benchmark was authorized by these measurements. Their cause is not
  established; this is not a successful measured-cost calibration or a 4B rerun.
- Local diagnostics are retained under the ignored directory
  `slackpipe_experiments/group_quality_20261001_fixed/calibration/`, including
  both attempts and `profiling_attempts.json` (`selected_profile: null`). All
  partitions/attempts in this validation share one source-diff hash. An earlier
  development-time collection under `group_quality_20261001/calibration/` was
  invalidated by concurrent source edits and blocked by the context guard; it
  remains separate and is not counted as a validation success.

The actual Nemotron-H 4B raw events and affected-stage consensus costs cannot be
reported without the missing ledger/data. The two supplied raw spike values and
the supplied group-estimate patterns are covered by explicitly synthetic tests,
not presented as analysis of those unavailable files.

## Evidence-First Iteration Inspection (2026-10-01)

For the subsequent repeatable cross-partition anomalies, see
[compute timing diagnostics](slackpipe_compute_timing.md). That probe is
independent of filtering and cannot publish a solver cost profile.

The subsequent request identifies `calibrate-691347ced1ab`, attempt 1, with
eight surviving within-group CV failures. Its raw events are not present in
this checkout or the existing container; the requested external workspace is
not mounted. The quoted CVs and group counts alone cannot distinguish rare
iteration contamination from drift, two timing populations, or sustained
instability. No iteration filter or empirically justified iteration thresholds
have been enabled. CV remains 0.10; sample/group policy, retries, cost fitting,
quality schema v3, and receipt/context semantics are unchanged. This is a
diagnostic checkpoint, not completion of the requested three-level filter.

Use the read-only replay tool once the attempt directory is accessible inside
the container (`WORKER` is the directory containing the three input files):

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev /opt/venv/bin/python \
  -m tools.slackpipe_inspect_iterations \
  --profile WORKER/cost_profile.candidate.json \
  --quality-report WORKER/cost_profile.quality.json \
  --events WORKER/calibration_events.json \
  --output WORKER/iteration-inspection.json
```

The output must not exist. The tool accepts the existing partition-bundle or
flat raw-event JSON, requires explicit event provenance, replays the saved
policy, and inspects **every remaining** `within_group_cv` failure. It verifies
the quality issues against the candidate, exact microbatch coverage, the raw
sample detector's decisions, per-iteration accepted counts, raw/cleaned totals,
and the reported CV. Mismatched evidence fails before any output is written.
It does not read group-excluded CVs as remaining failures or infer a group's
identity from file/event order. Missing legacy provenance is an error.

Output schema `slackpipe.iteration_inspection.v1` contains:

- `inputs`: actual input paths and SHA-256 hashes; `diagnostic_only: true`;
  unchanged profile status and recorded selected profile.
- `failures`: group/attempt, stage/phase, exact range, role, worker, timing
  context, manifest/composition where present, and original raw paths.
- Each failure's `iterations`: global ID, ranks, raw and accepted microbatch
  counts, raw total, B-normalized sample-cleaned total, accepted microbatch
  mean/median/min/max. No iteration is discarded.
- Raw and sample-cleaned iteration CV, median, MAD, robust sigma, relative and
  signed deviations; half-window medians, adjacent changes and largest sorted
  gap. These are descriptive evidence, **not** automated cause classifications.
  Robust z is null when MAD is zero; that does not authorize exclusion.
- Existing sample and group rejections separately, and explicitly unavailable
  boundary attribution. Elapsed-only calibration events cannot establish clock,
  allocator, CUDA/NCCL or pipeline-fill/drain causality.

Exit 0 means diagnostic replay succeeded, **not** that the profile passed or the
solver may run. This tool neither selects a profile nor launches solver/training.
Original files are never modified. It cannot silently repair an inconsistent
quality report or select the faster part of a distribution.

Offline replay on the **different, previously recorded tiny-hybrid** attempt 1
under `slackpipe_experiments/group_quality_20261001_fixed/calibration/` verified
all eight of that run's remaining CV failures against raw events. For example,
partition5/stage2/backward per-operation means (global iterations 20-29) are
`[8.8070, 6.6080, 5.9826, 5.9951, 5.9902, 5.9869, 5.9839, 5.9929, 5.9755, 5.9905]`
ms, CV 0.133585. Several stages have consecutive deviations at the beginning
of the measured window, not evidence of isolated <=5% contamination. With only
10 measured iterations even one exclusion is 10%. No samples/iterations were
newly discarded, the three prior group exclusions remain, and the ledger still
has `selected_profile: null`. The derived inspection JSON is retained locally
alongside the original attempt and is excluded from Git.

The saved metadata confirms calibration mode, profiler disabled, 20 full-step
warmups and measurement IDs 20-29. Source inspection confirms no profiler
wait/active/repeat or flush transition in this path; pre-call CUDA sync is
outside the timer, completion sync inside, and cache clearing between
partitions. This does not establish the physical cause of the initial shifts.
No timing boundaries were changed based on this unrelated diagnostic run.

The new regression suite tests stable, isolated slow/fast, two rare, 20%,
balanced-cluster, and drift distributions as **unchanged diagnostic inputs**;
sample-spike replay; exact group/range/context separation; missing/duplicate
microbatches; stale evidence; input hashes; and refusal to overwrite files.
These are not pass/fail tests of a third filter that has not been enabled:

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev /opt/venv/bin/python -m pytest \
  tests/unit_tests/pipeline_parallel/test_slackpipe_iteration_inspection.py -q
```

The Nemotron-H 4B raw-distribution analysis, data-derived iteration thresholds,
third-level filtering and real profiling/solver continuation remain blocked on
the specified raw data and an appropriate GPU environment. No 4B profiling was
rerun, and no claim is made that its eight CV failures are fixed.

This diagnostic checkpoint passed 18 new inspection tests and 154 existing
tests in `slackpipe-dev` (`test_slackpipe_outliers`, `test_slackpipe_group_quality`,
`test_slackpipe_profile_quality`, `test_slackpipe_eval_receipts`,
`test_slackpipe_cost_profile`, `test_slackpipe_collection`). The latter include
fresh-process retry and campaign gating regressions. Black/isort checks and
`git diff --check` passed. No new GPU training/calibration was performed; the
raw-event replay above uses previously recorded data, not a new measurement.
