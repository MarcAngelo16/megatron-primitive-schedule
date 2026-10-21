# Profiler Transfer Diagnosis

These read-only/offline and opt-in GPU controls do not export a cost profile,
invoke the solver, change quality policy, or instrument normal training.
Use the existing `slackpipe-dev` container. Use fresh output directories; never
overwrite calibration data or receipts.

## Evidence Boundaries

The earlier Llama results used real stage calibration whereas the Nemotron
results used isolated unit sums. Architecture and measurement method therefore
changed together. A comparison of their prediction errors alone cannot identify
an architectural cause. F/B compute makespan, profiled GPU envelope, and full
profiler-off optimizer-step wall time are different quantities.

`slackpipe_diagnose_residuals` verifies the historical file manifest and analyzes
only training partitions of the training runs listed by the profile. It reports
per-run fits using the existing feature set, exact-range cross-partition shifts,
raw/iteration statistics, pipeline positions and residual groupings. Per-run
fits are in-sample diagnostics, not independent validation or new coefficients
for production. It does not read heldout observations into a corrective fit.

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev /opt/venv/bin/python \
  -m tools.slackpipe_diagnose_residuals \
  --source slackpipe_experiments/stage_calibration_20261002 \
  --output slackpipe_experiments/new-diagnosis/residuals.json
```

## Matched Composition Control

`slackpipe_diagnose_composition` uses the actual native `HybridStack.forward`
versus independent execution of those exact same layer objects. It captures
actual intermediate activations and upstream gradients before measurement,
checks shape/stride preservation, and compares final outputs, input gradients
and every parameter gradient. These are fixed-weight mechanism tests, not full
pipeline SGD throughput measurements.

- H/FFN/Mamba dimensions come from the supplied tiny model configuration;
  sequence=32, microbatch=1, FP32, TP/DP/CP/PP=1.
- Both modes reset gradients once per step and accumulate four microbatches.
  All layer inputs require gradients. Isolated replay has additional leaf
  gradient accumulators/autograd entry calls, unlike one connected stage graph.
- Both use preallocated, first-record-initialized events. There are no per-layer
  or per-F/B synchronizations. Warmup/window boundaries synchronize both modes.
- No inference, communication, embedding, final norm/head/loss, optimizer or
  new compile/graph configuration enters the measured control.
- Internal library compilation/fusion is left enabled. A disabled outer CUDA
  graph configuration does not imply that no internally compiled functions exist.
- Three fresh processes, randomized case/mode order with recorded fixed seeds,
  20 warmups + 30 measurements. All warmup and measurement samples are retained.
- Sum isolated layer envelopes per microbatch, average the four microbatches
  per iteration, then report all 30 values, their median/mean/population stddev,
  CV and quantiles. Run-level dispersion is separate.
- A full TransformerLayer (attention + MLP, ReLU2/no RoPE) is a homogeneous
  control on the same path, not a reproduction of the earlier Llama benchmark.
  A hybrid `M`, `*`, `-` is respectively a Mamba residual layer, attention-only
  residual layer, or MLP-only residual layer, not a full Transformer block.

Match the existing tiny pipeline harness's environment explicitly:

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev env \
  OMP_NUM_THREADS=1 MAMBA_DETERMINISTIC=1 TRITON_CACHE_AUTOTUNING=0 \
  NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 \
  /opt/venv/bin/python -m tools.slackpipe_diagnose_composition \
  --model slackpipe_experiments/stage_calibration_20261002/model.json \
  --output slackpipe_experiments/new-diagnosis/matched \
  --runs 3 --warmups 20 --iterations 30
```

`design.json`, `run*.command.json`, logs, per-run environment/order and per-case
JSON retain provenance and all samples. Optional `--trace-steps 3` adds a separate
post-measurement CPU/CUDA capture with shapes/stack/memory disabled. Trace data
is diagnostic and never replaces event costs. Enable it in a separate short run;
trace serialization can affect subsequent cases' environment.
The trace probe resets gradients after each single F/B pair (one microbatch),
unlike the preceding four-microbatch timing windows. It checks dispatch and
activity provenance, not accumulation timing or production throughput.

The backward-entry factor has two additional, diagnostic-only values for
`--isolated-backward`: `separate-entries` and `single-entry`. Both use ONE event
envelope around all disconnected layer backwards, so changing the number of
event pairs is not confused with changing autograd entry count. The first calls
autograd once per root; the second calls it once with the same list of roots and
upstream gradients. Compare these to each other, not directly to the default
`per-layer-events` boundary. Keep the environment, cases, warmup and sample count
identical, alternate method order across independent repeats, and preserve all
outcomes, including ratios below one. `--trace-steps` deliberately rejects these
entry-count variants; the normal composition trace uses explicit autograd-node
hooks to correlate backward launches.

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev /opt/venv/bin/python \
  -m tools.slackpipe_diagnose_trace \
  --trace slackpipe_experiments/new-diagnosis/traces/run0/mamba6.trace.json \
  --output slackpipe_experiments/new-diagnosis/mamba6.attribution.json
```

This analyzer requires same-thread launch correlation to an executed backward
node, not merely a CPU timestamp within an outer backward annotation. It retains
unassigned work, overlapping-stream interval unions, individual kernel durations,
allocation/wait API calls, stream IDs and the original clock metadata. It does
not interpret CPU wait insertion duration as GPU blocking time.

Run the focused aggregation/manifest and existing native-stage tests inside the
container:

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev /opt/venv/bin/python -m pytest \
  tests/unit_tests/pipeline_parallel/test_slackpipe_profiler_diagnosis.py \
  tests/unit_tests/pipeline_parallel/test_slackpipe_stage_measurement.py -q
```

Do not infer kernel causality from a compiled graph name, numerical agreement,
or an event envelope. Validate actual dispatch/layout and correlate GPU launches
before attributing a difference to a specific operator. The local H64/A2000
controls cannot validate Nemotron-H 16B/BF16/A100 performance or its older library
versions. A failure of additive isolated sums is not permission to scale costs,
loosen thresholds or select faster runs.
