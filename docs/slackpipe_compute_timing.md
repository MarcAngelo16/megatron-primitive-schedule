# Compute Timing Diagnostics

This diagnostic does not change the solver estimator or any quality threshold.
The supplied 30-iteration Nemotron-H summary has no raw data mounted in this
environment. Its recurring differences have not been explained or fixed here.

## Actual Boundaries

`existing-stage-wall-time-v1` in the generic campaign names the synchronized
wall-time collector in `schedules._run_with_slackpipe_cost_calibration`:

1. Device-wide CUDA synchronization **before** starting the timer.
2. `forward_step` or `backward_step` on the current compute stream.
3. Device-wide CUDA synchronization **inside** the interval, then stop the timer.

It is not unsynchronized CPU launch timing. Forward includes last-stage loss
handling. Backward includes autograd/custom Mamba backward launches. Pipeline
receives/sends, input selection and explicit pre/post gradient-sync bookkeeping
are outside these closures. Autograd-internal dependencies, allocations, host
dispatch and any work caught by the completion boundary are not identifiable
separately from the original elapsed duration.

Legacy records now also expose `cpu_wall_ms`, `pre_compute_sync_ms`, and
`completion_sync_ms`; `existing_stage_wall_ms` aliases `elapsed_ms`, whose meaning
and use as the ordinary calibration cost remain unchanged. Two extra host clock
reads add small, unquantified overhead, not new synchronization boundaries.

Native P2P waits on Work handles and, when `batch_p2p_comm && batch_p2p_sync`,
performs its own device synchronization. Those existing behaviors are unchanged.
Default calibration constructs no torch profiler. There is no profiler wait/active/repeat
or flush transition in this path. Each partition receives full-step warmup;
cache clearing occurs between partitions, outside timed compute closures.

## Opt-In Probe

`--diagnose-compute-timing` is available on `tools.slackpipe_eval_worker` and
`tools.slackpipe_nemotron_worker`, only for native interleaved calibration
(VPP >= 2). It shares model construction, partition generation, data, schedule,
backward and SGD with ordinary collection.

- Preallocate and initialize one CUDA event pair per local F/B operation before
  training warmup; reuse pairs across iterations.
- Record on the current compute stream around the compute closure. The probe
  adds **no per-operation device synchronization or event wait**.
- Drain the device once after the full step including SGD, then resolve events.
  No new layer/microbatch barriers are inserted.
- Observe inclusive native P2P host-call duration with a communicator subclass.
  Requests, return objects and native wait/sync policy are unchanged.
- Keep global iteration, microbatch, stage, worker, exact range and ordered
  composition. `compute_position` counts compute calls; diagnostic
  `schedule_position` also includes P2P calls, exposing adjacency.

An event envelope is **not proven intrinsic compute time**: it can include host
launch gaps and waits inserted inside the region. Auxiliary-stream work is
included only if it joins the compute stream before the end event. The final
device drain does not retroactively extend individual event intervals.
P2P host duration, GPU envelopes and drain time cannot be added as disjoint work.
Fine-grained NCCL wait, allocator and kernel attribution still requires a trace
or additional focused instrumentation.

## Overlap and Serialized Control

Two worker-only flags extend the diagnostic, without changing the estimator,
filtering, quality thresholds, ordinary calibration or benchmark/runtime:

- `--diagnostic-timeline`: CPU and CUDA activity, shapes/stack/memory profiling
  disabled. One continuous raw Kineto capture **per partition and rank**, covering
  all training warmup and measured iterations. No scheduled cycles are dropped;
  serialization happens after the partition's entire measurement window. This
  can produce large files and perturb CPU dispatch. Compare both modes with the
  same setting; optionally repeat both without tracing as a perturbation check.
- `--diagnostic-execution serialized`: intentionally synchronize the device
  before recording each compute start event and after its end event. Complete
  native P2P before proceeding. This supports **blocking batch P2P with native
  batch synchronization only**; asynchronous unmatched receives are rejected
  before launch to avoid deadlock. There are no cross-rank per-operation barriers.
  This isolates preceding pipeline work on the local GPU, not other processes,
  kernel-internal concurrency, clocks or thermal/power state.
- `--diagnostic-nvtx`: independent NVTX operation/step/P2P labels and native
  timing-event handles, for an **external** Nsight Systems capture. It does not
  construct a torch profiler. Do not combine CUPTI collectors; omit
  `--diagnostic-timeline` when using Nsight.

All require `--diagnose-compute-timing`. Normal execution adds no per-operation
sync. Full-step warmup uses the selected diagnostic policy, including optimizer
and the same profiler. No profile is fitted or passed to a solver in either mode.
Native batch P2P already synchronizes in the current configuration: the name
"normal" does **not** assert that communication overlap was enabled.

Using the four-GPU `launcher` and `common` arrays from Target Reproduction:

```bash
"${launcher[@]}" "${common[@]}" --diagnose-compute-timing \
  --diagnostic-timeline --diagnostic-execution normal --run-id overlap-normal \
  --output slackpipe_experiments/overlap-pair/normal
"${launcher[@]}" "${common[@]}" --diagnose-compute-timing \
  --diagnostic-timeline --diagnostic-execution serialized --run-id overlap-serial \
  --output slackpipe_experiments/overlap-pair/serialized
docker exec -w /workspace/Megatron-LM slackpipe-dev /opt/venv/bin/python \
  -m tools.slackpipe_analyze_overlap \
  --normal slackpipe_experiments/overlap-pair/normal \
  --serialized slackpipe_experiments/overlap-pair/serialized \
  --output slackpipe_experiments/overlap-pair/comparison.json
```

Use 30 measured iterations and 20 full-step warmups, identical hardware,
configuration, partitions and checkout. Preserve original runs. The report
rejects differing source/configuration, missing ranks or operations, mismatched
ranges/composition and mismatched trace settings. Inputs are SHA256-hashed.

Raw `overlap.<run-id>.<execution>.rank<rank>.json` files retain CUDA runtime/driver
APIs, kernels, NCCL, allocation calls, stream/correlation IDs and Kineto clock
metadata. Operation labels use `ScheduleTrace/F|B/i<iteration>/s<stage>/b<batch>`;
P2P labels include iteration and dispatch position. These are torch profiler
annotations, not a claim that separate Nsight NVTX capture was performed.
For a follow-up dependency capture, the installed `nsys profile --help` supports
`--trace=cuda,nvtx --cuda-event-trace=true --sample=none`. Place `nsys profile`
inside Docker before the torchrun Python command, select `--diagnostic-nvtx`,
and use a fresh output directory/report name. Device event tracing requires
CUDA driver 12.8 or newer according to that CLI and can itself introduce overhead
and false stream dependencies. Check the target's help/driver and interpret this
as another controlled diagnostic, not a benchmark or automatic causality test.
No Nsight dependency report for the four-GPU target has been collected here.

`slackpipe.overlap_diagnostic.v1` contains:

- `groups`: every partition/stage/phase, median and CV of all per-iteration
  microbatch means for normal/serialized `wall_ms` and `compute_gpu_ms`.
- `cross_partition`: max(median)/min(median)-1 for **identical** stage, ordered
  composition/range, role, worker and configuration only; null for one group.
- `normal_operations`, `serialized_operations`: all samples, co-resident stages
  and ranges, neighboring compute/P2P operations and per-operation evidence.
- `overlap_evidence`: correlated GPU envelope and union, internal gaps, overlapping
  NCCL union (no double counting), earlier P2P launch attribution, co-resident
  stage GPU activity, individual compute activities and allocation API totals.
  `nccl_simultaneous_compute_union_us` separately counts intersections with
  associated compute activities, rather than overlap with envelope gaps.
  Raw trace hashes preserve access to all other events and metadata.

**Boundary caveats:** `wall_ms` in normal mode is asynchronous host closure time;
serialized `wall_ms` includes completion (pre-drain excluded). They are labeled
explicitly and must not be compared as identical latency estimators. Normal
`existing_stage_wall_ms` remains null: obtaining synchronous wall latency in the
same operation would destroy the overlap being investigated. CUDA events in both
modes remain stream envelopes, not pure compute time.

Kineto does not expose the device timestamps of our timing event pair. NCCL and
co-resident overlap are therefore reported against the **correlated GPU activity
envelope**, not fabricated event bounds. Missing overlap there does not prove no
overlap in the unobserved boundary gaps. `cuda_event_minus_gpu_envelope_ms`
exposes this discrepancy without rescaling. `stream_wait_api_candidates` retain
calls launched during the CPU operation (including engine threads); they do not
prove the compute stream blocked or provide its blocking duration. CUDA stream
handles and Kineto stream IDs are not assumed interchangeable. Wait attribution
and root cause remain `unknown`/`unresolved` when the trace cannot establish them.
Use a focused Nsight Systems CUDA-event/dependency capture on the target if those
ambiguities persist, instead of interpreting CPU API duration as GPU wait time.

If reproducible cross-partition variation disappears under serialization, that
supports an execution-context effect, not a choice of estimator by itself. If it
persists, compare the retained Mamba backward kernel names/durations, allocation
calls and launch gaps. Autograd-internal state/recomputation, clocks and allocator
cache behavior can still matter; overlapping activity alone proves no causal
resource contention. No estimator change is made by this diagnostic.

### Local Overlap Validation (2026-10-01)

The four-GPU Nemotron-H 4B run was **not available**: this machine exposes two
6 GiB RTX A2000s. None of partition5/s2/[13,19), partition4/s4/[26,32),
partition1/s6/[39,45), or partition2/s1/[6,13) has been remeasured here.
Their target wall/event/serialized times, overlap, waits and causal diagnosis
remain unavailable. The following is a separate tiny hybrid harness check.

Retained, ignored artifacts:
`slackpipe_experiments/overlap_diagnostic_20261001/{normal-v2,serialized-v2}`
and `comparison.json` (`slackpipe.overlap_diagnostic.v1`). Both runs completed
PP=2/N=4/L=12/B=8, FP32, sequence length 32, SGD, six partitions, 20 full-step
warmups and 30 measurements. Each has 11,520 measured F/B samples, 12 raw trace
files, and every captured global iteration 0..49. The comparison validated
52 input hashes and produced 48 partition/stage/phase comparisons, with zero
samples discarded. No solver cost profile was produced.

Collection provenance: base commit `8d528c69b18ef8e3179b01ddd6d22b190df2a704`,
tracked-diff SHA256
`0841a62d393a852827545846043633d425967d7cda9528acf9f9cfbefc7181ad`.
Per-rank metadata retains untracked-source hashes, hardware, software, native
allocator settings and timing configuration. The paired collection used one
frozen source tree; subsequent offline-analysis/tests/docs and the optional
NVTX extension are not represented as the source of those measurements.

Same stage 1, worker 1, range [3,6), backward. Values are medians of 30 raw
iteration means over all eight microbatches, in milliseconds:

| Partition | Normal host wall | Normal event | Serialized wall | Serialized event | Normal GPU activity union | Serialized GPU activity union |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 7.1492 | 7.1474 | 7.1886 | 7.1560 | 0.5310 | 0.5304 |
| 1 | 6.9910 | 6.9892 | 7.0787 | 7.0467 | 0.5300 | 0.5299 |
| 3 | 16.0924 | 16.0897 | 16.1569 | 16.1152 | 0.5307 | 0.5306 |

Normal wall is asynchronous host time, not a completion-latency measurement.
Across all measured operations in both modes, no NCCL/co-resident-stage activity
overlapped the correlated GPU envelopes; no earlier-P2P overlaps or in-operation
stream-wait API candidates were observed. Native batch P2P synchronization was
already enabled, with `overlap_p2p_comm=false`. This does not certify absence of
unobserved waits or external-process contention.

For the rows above, normal envelope internal gaps were 6.4101, 6.2345 and
15.1917 ms; serialized gaps were 6.4212, 6.2964 and 15.2178 ms. Each group had
36 distinct associated activity names and the same median per-iteration activity count; no
in-operation CUDA allocation API calls were attributed. Thus this local variation
persisted under serialization and appeared predominantly in gaps, not the union
of GPU activity. CPU dispatch/launch behavior deserves further investigation;
the precise cause, Mamba state/recomputation contribution and relevance to the
large target remain unresolved. No claim of intrinsic compute timing follows.

Validation in `slackpipe-dev`: 196 regression tests passed (including quality,
filtering and retry policies); the final expanded focused suite passed 11 tests.
Homogeneous and hybrid PP=2 FP32 numerical equivalence each passed on both ranks.
An independent scalar/matrix CUDA test preserved gradients and SGD updates
exactly with normal/serialized instrumentation. A separate single-GPU Nsight
smoke (`nvtx-smoke.nsys-rep`) captured the expected F/B labels and native event
handles; it was not a warmed performance measurement or target dependency study.

Probe records contain `compute_gpu_ms`, `cpu_wall_ms`, stream ID and operation
positions. `existing_stage_wall_ms` and per-operation `wait_synchronization_ms`
are **null**, not zero: recreating the old device-wide boundary in the same
operation would perturb the probe. Compare a separate matched legacy run.
Inclusive P2P host duration is not NCCL GPU service time; its wait component is
not separately measured. Full-step `completion_wait_ms` is measured separately.

## Target Reproduction

Run both commands on the same fixed checkout/container and idle GPUs, with the
original precision, seed, optimizer, sequence and batch settings. Do not edit or
stage source or run competing GPU tests during the pair. This four-GPU example
requires adequately sized target GPUs, not this host's two 6 GiB A2000s. Adjust
the common arguments consistently to match the original experiment. Bash:

```bash
launcher=(docker exec -w /workspace/Megatron-LM slackpipe-dev env
  OMP_NUM_THREADS=1 MAMBA_DETERMINISTIC=1 TRITON_CACHE_AUTOTUNING=0
  NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
  /opt/venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=4
  -m tools.slackpipe_eval_worker calibrate)
common=(--model-config configs/slackpipe_eval/nemotron_h_4b.json
  --schedule interleaved --pp 4 --logical-stages 8 --microbatches 8
  --micro-batch-size 1 --seq-length 1024 --precision bf16
  --seed 1234 --learning-rate 1e-6 --warmups 20 --iterations 30)

"${launcher[@]}" "${common[@]}" --run-id wall-reference \
  --output slackpipe_experiments/timing-pair/legacy
"${launcher[@]}" "${common[@]}" --run-id event-probe \
  --output slackpipe_experiments/timing-pair/events --diagnose-compute-timing

docker exec -w /workspace/Megatron-LM slackpipe-dev /opt/venv/bin/python \
  -m tools.slackpipe_compare_compute_timing \
  --legacy slackpipe_experiments/timing-pair/legacy \
  --diagnostic slackpipe_experiments/timing-pair/events \
  --output slackpipe_experiments/timing-pair/comparison.json
```

A legacy quality failure returns worker exit 2 (torchrun reports failure).
Keep the raw data and receipt; never bypass it to launch the solver. The separate
event diagnostic writes **no cost profile**, does no fitting/filtering, and
cannot authorize the campaign. Exit 0 only means diagnostic collection completed;
it does not trigger quality retries. Always use fresh output directories.

For a local mechanism check, use `tools.slackpipe_nemotron_worker calibrate
--tiny --precision fp32 --seq-length 32 --warmups 20 --iterations 30` under
two-rank torchrun with the same environment above. Run without/with
`--diagnose-compute-timing` using different fresh `--output` directories. This is
a tiny hybrid with Mamba, attention and MLP blocks, not Nemotron-H 4B.

## Output and Comparison

- `timing_events.rank*.json` per partition and merged `timing_events.json`: all
  raw diagnostic operations. They deliberately lack `elapsed_ms`, preventing
  the current cost aggregator from interpreting them as calibrated costs.
- `result.rank*.json`: configuration, source, hardware/software/allocator,
  profiler settings (disabled), warmup/measurement IDs and boundary memory
  samples. `timing_diagnostic` uses `slackpipe.compute_timing_diagnostic.v1` and
  includes iteration-level P2P calls and full-step completion waits.
- `slackpipe.compute_timing_comparison.v1`: input hashes, raw records, all
  stage/group/phase iteration means and CVs, cross-partition median shifts.
  CV = population stddev/mean; shift = max(group medians)/min(group medians)-1,
  or null for fewer than two groups. Nothing is filtered or rescaled.

Comparison requires identical manifests, complete ranks/operation grids,
configuration including hardware/software/source, partition sets and iteration
IDs. It compares identical stage/range/ordered-composition/role/worker/phase, not
different workloads. Old references without comparable metadata must be
recollected rather than silently matched.

Inspect backward ranges [13,19], [26,32], [39,45], [6,13], [34,39] and [19,28]
in `groups` and `comparisons` on the target run. Lower event CV alone is not a
root-cause diagnosis. The report deliberately sets `estimator_validated: false`;
any permanent estimator change requires subsequent empirical review.

## Validation on 2026-10-01

The two-rank tiny-hybrid pair above completed six partitions, 20 warmup and 30
measured iterations per partition, B=8, N=4, FP32, sequence length 32, SGD
learning rate 1e-6. Each run retained 11,520 F/B records. The event probe retained
7,020 P2P call records and emitted no cost profile. Collection was sequential,
with no source edits or other GPU tests during the pair. The comparison verified
identical per-worker configuration/source/hardware across both runs and all
partitions, plus complete operation grids and event timing-context hashes.

Raw data, logs and `comparison-context-verified.json` are retained locally under
the ignored `slackpipe_experiments/compute_timing_20261001/`. Both runs used the
pre-commit worktree based on `30032681d0b510dd5853ae41089e797af445047a`, with tracked
diff SHA-256 `fe00811d2b42f1da95b0f81ad14ba5863f19df0d358cb4dcb3dcf68e870c6bdd`.
Per-file untracked source hashes are retained in each result's environment
metadata. The comparison records hashes of all input evidence. These are new
**tiny-hybrid diagnostic measurements**, not Nemotron-H 4B results.

Backward examples, same exact range/stage/worker/composition across partitions:

| Range | Groups | Wall medians (ms) | Event medians (ms) | Wall shift | Event shift |
| --- | --- | --- | --- | --- | --- |
| [3,6] | 0,1,3 | 12.0651, 9.2245, 12.0390 | 11.8397, 9.1358, 11.9864 | 30.79% | 31.20% |
| [6,9] | 0,2,4 | 7.9368, 5.8270, 3.5200 | 7.8252, 5.8116, 3.5358 | 125.48% | 121.31% |
| [0,3] | 0,1,3,5 | 8.0467, 8.1127, 7.6749, 3.5116 | 7.8498, 8.1330, 7.6977, 3.5131 | 131.02% | 131.50% |

For [3,6], the wall CVs are 0.0082/0.0899/0.0161 and event CVs are
0.0422/0.0504/0.0054 in the same group order. Across all 48 stage/phase/group
estimates (no diagnostic filtering), five wall CVs and ten event CVs exceed 0.1.
This run **does not support switching to events as a stable intrinsic cost**.
The separate protocols alter synchronization, so these are not paired samples
from identical GPU execution, nor a controlled throughput comparison.

The legacy reference's existing quality gate returned `rerun_required`: zero
raw sample exclusions, two group exclusions, five surviving within-group CV
failures, plus consensus/cross-group/fit failures. `selected_profile` remains
null; its worker exit 2 was expected quality rejection after complete collection.
No solver or benchmark was launched from these measurements. No iteration filter
or threshold adjustment was made, and no diagnostic retry was mislabeled as a
successful cost calibration.

Runtime inspection followed `MambaMixer` into the installed mamba-ssm 2.3.2.post1
`MambaSplitConv1dScanCombinedFn.backward`: it recomputes convolution intermediates,
allocates scratch tensors and launches the scan/convolution backward kernels.
There is no explicit device synchronization or stream-switch call in that Python
function. This is existing kernel-internal work, not newly enabled Megatron
activation recomputation. The native communication calls in the probe all retain
batch synchronization. Host closure durations closely track the observed event
envelopes, but that does not separate launch gaps, allocator/cache behavior,
kernel selection, or device execution. The precise cause remains unresolved;
kernel/runtime traces are needed before changing an estimator or buffer lifetime.

Regression validation: 165 profiling/cost/receipt/collection/timing tests passed
before the pair, including one real-CUDA event test. Two distributed PP=2 FP32
numerical-equivalence tests (homogeneous and hybrid) passed on both ranks after
the pair. The generic eval-worker diagnostic CLI also completed a 30-iteration,
PP=2/N=4/B=4 tiny GPT smoke, retaining 960 operations and no cost profile under
`generic-events/` in the same ignored root. This checks the generic entry point,
not 4B behavior. No target 4B or PP=4 run was possible on the two 6 GiB GPUs.

Final combined regression: **184 passed**, including the added context-mismatch
check and real-CUDA gradient/SGD equivalence with and without the event probe.
Black, isort and `git diff --check` passed. Generated measurement files and logs
are not part of the source commit.
