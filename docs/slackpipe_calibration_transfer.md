# Complete-Stage Calibration Transfer Diagnostic

This opt-in experiment does not modify production calibration, filtering,
quality thresholds, native scheduling, or solver semantics. It never publishes a
trusted cost profile. Use the existing `slackpipe-dev` container; no host install.

## Frozen Study

The entry checks the existing 12-layer `M*-M*-M*-M*-` native hybrid H64 model.
Sequence32, microbatch1, FP32, PP2/VPP2, B4, TP/DP/CP1, SGD lr0.001, seed1234.
The six partitions and train/run split are defined in `FROZEN.json` before any
GPU measurements. Run0/1 of P1-P3 only train the four-column F/B models. P0 is
always control; P4/P5 and run2 are held out. No rows are filtered.

Native `get_schedule_table` and `get_pp_rank_microbatches` define two interleaved
orders: microbatch group2 (training) and group4 (order holdout). Both retain the
same native scheduler, P2P path, cyclic placement, and full backward. Orders are
checked by the production plan validator before execution, then compared exactly
with recorded execution. There is no solver-derived order selection.

Composition keys count complete manifest class signatures. A lexicographically
first real contiguous range represents each count vector, preserving its order.
This groups order permutations deliberately: their transferability is a hypothesis,
not an assumption of measured equivalence. Each repetition is ONE complete body
F/B with ONE autograd entry, then SGD, using matched-shape random input/upstream
gradients. These are not captured pipeline activations. Parameter names/seeds use
global layer IDs. The seed5678 standard-normal upstream is divided by B*S*H=8192,
an averaged linear surrogate, not an unnormalized random loss. Gradients reset
once per repetition. No boundary modules/P2P
are included. Runs alternate device0/1/0; device and order are recorded.

Pipeline repetitions are complete four-microbatch SGD steps. All main conditions
use 30 exact-path warmups and 30 measured repetitions, with no object recreation
or cache clearing at the warmup/measurement boundary. Partitions rebuild models
between windows. Fresh processes are used per run/mode/order; conditions rotate
across runs. No GPU experiments are run concurrently.

## Collection and Models

- `off`: no F/B or per-step CUDA events and no profiler. Synchronized continuous
  full-step wall window, max across ranks divided by 30. Per-step host dispatch
  samples are context, not synchronized individual GPU completion times.
- `pipeline_events`: existing preallocated AND first-record-initialized event
  collector, no added per-F/B or per-step drain. Read all events after the window.
- `pipeline_sync`: same full step, adding the old profiler's pre/post device
  synchronization around each whole F/B. Event and synchronized host wall times
  are stored separately. Pre-sync is excluded from stage wall. No added P2P drain.
- `composition_stage`: whole-body events, auxiliary host dispatch times, no
  per-F/B synchronization. Optimizer is outside each body event interval.

CUDA-event intervals include gaps and stream waits; they are not intrinsic kernel
costs. Coverage of unjoined side-stream computation is not proven. The end-to-end
window drains the device on each rank. GPU timestamp origins are never compared.

M1 fits synchronous GPU events and M2 fits normal GPU events, separately for F/B:
`a * composition + b0 + b_first * first + b_last * last`. Arithmetic means of all
120 pipeline operations per stage/run and all30 composition repetitions are used;
the two training composition runs receive equal weight. Unconstrained OLS has a
rank check; negative predictions are reported, not clipped. Standard errors assume
iid stage observations and are descriptive ONLY; per-training-run coefficients
expose instability, but two run clusters cannot support reliable confidence bounds.

M0 fixes a=1, b0=0 and estimates first/last mean residuals ONLY from training
pipeline-events data. These are boundary AND context increments, not pure measured
embedding/head/loss costs. Thus M0 is an unscaled training-boundary baseline, not
a wholly pipeline-independent oracle. No historical Llama values are used.

Matching the sync profiler is not matching unperturbed execution. MAPE is reported
against both reference modes. The events reference is independently checked using
paired profiler-off full-step windows with a three-run Student-t interval (df2),
not by treating 30 consecutive steps as independent runs. An interval outside
[-1%,1%] does not establish the <=1% perturbation goal.

## Commands

Run from the checkout root, using a new output directory each time:

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev env \
  OMP_NUM_THREADS=1 MAMBA_DETERMINISTIC=1 TRITON_CACHE_AUTOTUNING=0 \
  NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 \
  /opt/venv/bin/python -m tools.slackpipe_calibration_transfer prepare \
  --model slackpipe_experiments/stage_calibration_20261002/model.json \
  --output slackpipe_experiments/new-transfer

docker exec -w /workspace/Megatron-LM slackpipe-dev env \
  OMP_NUM_THREADS=1 MAMBA_DETERMINISTIC=1 TRITON_CACHE_AUTOTUNING=0 \
  NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 \
  /opt/venv/bin/python -m tools.slackpipe_calibration_transfer run \
  --output slackpipe_experiments/new-transfer

docker exec -w /workspace/Megatron-LM slackpipe-dev /opt/venv/bin/python \
  -m tools.slackpipe_transfer_analysis --input slackpipe_experiments/new-transfer

docker exec -w /workspace/Megatron-LM slackpipe-dev /opt/venv/bin/python -m pytest \
  tests/unit_tests/pipeline_parallel/test_slackpipe_transfer.py -q
```

`prepare` stores commit, dirty diff, source snapshot, configs, signatures, execution
order and hashes. `run` records every subprocess command/log/exit status and first
requires exact FP32 numerical checks for every partition/mode/order. If the
default order fails, no measurements are admitted. An alternate order failing
numerical equivalence is explicitly rejected before timing and reported as an
unvalidated order holdout; the default-order partition study can continue.
`MEASUREMENT_ADMISSION.json` binds that decision to correctness/source hashes.
Correctness
runs separately clone operation outputs and returned input gradients, losses,
parameter gradients and updated parameters; their timings are excluded. They use
one warmup plus one checked full step, NOT performance samples.

Outputs: `composition/run*/C*.json` raw body events, `pipeline/run*.g*.*/P*/`
rank-local raw operations and window metadata, `correctness.json`, `analysis.json`
(fits, uncertainty, every stage prediction, heldout metrics, residual grouping,
same-[0,2) contrasts and paired perturbation), `FINAL_MANIFEST.json` hashes.
All collection is diagnostic; no profile or plan is automatically adopted.

Failures stop the campaign with logs retained. Nothing retries until a fast run
is found. Stability statistics describe warmup tail and measured thirds, using
host dispatch context only; a completed warmup is not proof of stable timing.
These H64/A2000 results do not validate A100/16B performance or prediction accuracy.
