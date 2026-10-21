# Native Stage Calibration

This opt-in research workflow preserves the existing isolated and synchronized
stage profilers. Omitting `--stage-collection` leaves the old worker behavior
unchanged. It does not change either native pipeline scheduling algorithm or
SlackPipe's optimizer. All examples run from the checkout in `slackpipe-dev`;
do not install host dependencies. Four-GPU examples need the actual target
hardware, not this development machine's two 6 GiB RTX A2000s.

## Actual Entrypoints

- `tools.slackpipe_isolated_profile`: legacy isolated module measurement.
- `tools.run_slackpipe_eval` and `tools.slackpipe_profile_quality`: existing
  campaign, quality checking and one-retry policy. Their default estimator and
  filtering are unchanged; the new experimental workflow is not automatically
  admitted to campaign receipts.
- `tools.slackpipe_eval_worker`: torchrun worker; `run_partition()` in
  `tools.slackpipe_nemotron_worker` supplies the shared complete SGD step.
- `StageEventCollector`: consumes the existing native interleaved scheduler's
  `_run_with_slackpipe_cost_calibration` boundaries. Native PP>1/VPP>1 only for
  now; no hooks are inserted into normal runs or custom SlackPipe execution.
- `tools.slackpipe_stage_calibration`: bounded experimental design, aggregation,
  independent-run holdout and v2 exporter. `tools.slackpipe_stage_trace` analyzes
  raw layer diagnostics. Existing `figure_trace.py`/`plot_schedule_trace.py`
  remain available for pipeline trace visualization, not cost substitution.
- `slackpipe/src/cost_profile.cc::ApplyCostProfileFile`: reads v2 prefixes and
  biases; `Instance::ForwardDuration/BackwardDuration` use rounded prefix
  differences and stage-role bias. No C++ format/algorithm changes are required.

## Evidence About Execution Differences

Both paths use the strict model adapter's `transformer_config`, TE layer specs
and `hybrid_stack_spec`, precision, and manifest layer order. This is not enough
to conclude that isolated and pipeline timings measure the same quantity:

| Property | Existing isolated path | Native stage path |
|---|---|---|
| Input | Random contiguous `[S,B,H]` per unit; fixed random upstream gradient | Actual embedding/preceding stage output and downstream gradient |
| F/B | Real autograd on one module; separate embedding/head units | Native `forward_step/backward_step`, all layers and wrapper/loss work |
| Gradients | Reset per F/B pair | Reset once per step, accumulate across microbatches |
| Optimizer | None | Same SGD step in off/events/layers modes |
| Synchronization | Device-wide before/after each F and B | Only setup and full-window boundaries added |
| State | Module-by-module initialization/cache cleanup | Entire partition remains resident through warmup/measurement |
| Communication | No pipeline; scoped identity TP=1 reductions | Native Megatron P2P requests and waits unchanged |

The current adapter has native batch P2P synchronization and no P2P overlap
optimization. Internal kernel streams/waits may still exist. Do not infer their
absence from these flags. Metadata records deterministic settings, allocator,
TF32, package/source/hardware identifiers, module classes and fusion/recompute/
CUDA-graph configuration. Layer traces record actual input/output shapes and
strides, including keyword inputs. No setting is changed to improve timings.
Layer diagnostics reject compiled outer modules, graph execution and configured
recomputation instead of silently disabling them. Coverage of arbitrary fused
or internally compiled modules is not certified.

## Timing Boundaries

`events` uses event pairs initialized before warmup. F/B events bracket the
existing native compute closures after explicit input availability and outside
P2P/optimizer. They include internal gaps, launch delays and stream dependencies.
The current helper obtains/clones its batch inside F: this wrapper cost remains
inside that envelope. Events are not intrinsic kernel time. Off-stream work is
covered only when an existing dependency joins it before the end event; otherwise
coverage and dependency wait duration remain unknown, never zero.

No per-operation or per-step synchronization is added. Pairs for the entire
warmup/measurement sequence are preallocated and read after window completion.
Optimizer events are separate. P2P CPU call durations are inclusive host time,
not GPU communication time and not additive to F/B event durations. Continuous
wall timing includes batch preparation, zero_grad, F/B, native communication and
SGD; boundary validation, event extraction, reductions and output are outside.
Per-step compute-stream envelopes are retained but are not proof of completion
on independent communication streams. Use synchronized continuous wall time for
throughput, not sums of operation costs.

`layers` is a separate short CUPTI run. Forward module hooks and real autograd
Node pre/post hooks carry layer/stage/worker/microbatch/iteration/phase identities.
Node traversal stops at input graph ancestors, including keyword inputs, and
includes parameter-gradient nodes. Reused AccumulateGrad nodes take the current
B microbatch identity. This does not assume reversed F order. Parameter sharing
and activity without unique launch correlation are `shared/unattributed`.
Hex-encoded JSON labels avoid unescaped quotes in the installed Kineto exporter.
The offline correlator retains every GPU activity, stream/correlation ID, raw
CPU allocation/wait API call and clock origin. It distinguishes GPU envelope,
activity interval union and internal gaps. It never calls gaps hardware idle or
uses CPU range duration as GPU time. GPU wait duration and cross-rank clock
alignment are not established by this tool.

No profiler object exists in `off` or `events`. `layers` uses one explicitly short
capture window; export follows the entire window, with all global iteration IDs
recorded. Shape/stack/memory collection defaults off. This follows the
[PyTorch profiler overhead guidance](https://docs.pytorch.org/docs/2.12/profiler.html)
and the installed API, not an assumed newer release. For multi-cycle pipeline
traces, the existing collection workflow retains its `on_trace_ready` exports.

## Reproduce

Freeze partitions before collecting data. The original and new 16B splits are
checks only. A separate partition is reserved for validation. Eight training
partitions cover short/long stages and full-rank class/first/last features; the
default 20-layer bound controls calibration size, NOT the solver search space.
It does not guarantee memory feasibility on a particular GPU.

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev /opt/venv/bin/python \
  -m tools.slackpipe_stage_calibration design \
  --model-config configs/slackpipe_eval/nemotron_h_16b.json --stages 8 \
  --check-split 2,4,7,9,17,17,14,10 --check-split 3,8,9,9,19,14,12,6 \
  --output slackpipe_profiles/stage-study/design.json
```

Run the following three times, replacing `run0` by `run1` and `run2`. Every
invocation is a fresh torchrun process with the same seed/configuration. Explicit
deterministic/autotune environment values must be identical across compared
methods; record the target's settings rather than treating a local setting as
universally appropriate.

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev env OMP_NUM_THREADS=1 \
  /opt/venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m tools.slackpipe_eval_worker calibrate \
  --model-config configs/slackpipe_eval/nemotron_h_16b.json \
  --schedule interleaved --pp 4 --logical-stages 8 --microbatches 8 \
  --seq-length 1024 --micro-batch-size 1 --precision bf16 \
  --warmups 30 --iterations 30 --stage-collection events \
  --calibration-design slackpipe_profiles/stage-study/design.json \
  --run-id run0 --output slackpipe_profiles/stage-study/run0
```

For the profiler-off reference, use `--stage-collection off`, a fresh output and
`--iterations 50`. Compare off/events with three independent runs each, 30 full
step warmups and 50 measured steps. Assess last-warmup CV/drift and timing series;
extend warmup in a NEW run if unstable. Nothing is silently discarded.
Use `--stage-collection layers --iterations 3` in another output for short layer
diagnosis, then compare against equally short off/events windows. Never claim
the diagnostic's timed window is profiler-off throughput.

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev /opt/venv/bin/python \
  -m tools.slackpipe_stage_calibration fit \
  --model-config configs/slackpipe_eval/nemotron_h_16b.json --stages 8 \
  --design slackpipe_profiles/stage-study/design.json \
  --runs slackpipe_profiles/stage-study/run0 slackpipe_profiles/stage-study/run1 \
         slackpipe_profiles/stage-study/run2 \
  --output slackpipe_profiles/stage-study/fit

docker exec -w /workspace/Megatron-LM slackpipe-dev /opt/venv/bin/python \
  -m tools.slackpipe_stage_trace --trace PATH_TO_CYCLE_JSON --output NEW_ANALYSIS_JSON
```

## Estimator and Output

All raw samples remain. Per stage/run/phase: average all microbatches within each
iteration, then take the median of those iteration means. Equal-row-weight NNLS
fits class counts plus first/last biases independently for F/B; middle bias is
zero. Costs expanded to individual layers are **inferred**, never called measured
single-layer costs. Units are microseconds; v2 prefixes, biases and content hash
are exported unchanged in meaning.

The last supplied run is wholly excluded from fitting; only its predeclared
validation partition supplies independent validation metrics. Training uses only
training partitions of preceding runs. Check partitions never fit. Reusing a
holdout for model selection requires a new design and new independent holdout.
Mixed execution/transport/software/hardware/allocator/source contexts fail
closed. Production quality defaults are reused (currently CV 10%, comparable
median spread 30%, relative RMSE 15%); at least ten iteration means are required.
No new outlier/group filtering is applied and existing policies are not relaxed.
Raw CV uses population standard deviation/mean; comparable spread is max/min-1
for identical range/role/worker/config; fit RMSE is normalized by target mean.

- `stage_samples.rank*.json`, `stage_samples.json`: all warmup and measured F/B
  event durations and identities; pipeline warmup/steady/cooldown position is
  distinct from training warmup. Null wait/communication GPU times mean unknown.
- `result.rank*.json`: environment/source, configuration, explicit profiler
  settings, optimizer samples, P2P calls, wall window, step samples and memory.
- `failed.rank*.json`: incomplete/failed records, without a potentially unsafe
  device drain. They are never fit.
- `cost_profile.experimental.json`: v2, raw data hashes, design, run split,
  environment fingerprints and inferred coefficients. No `cost_profile.json`
  is published and no solver is automatically launched by this command.
- `fit_validation.json`: every stage/run row, statistics/quantiles, CV, phase/
  role/worker/length residual groups, independent validation and actual-worker-
  order duration DAG replay. Replay is NOT an observed GPU-wall makespan.

Acceptance targets are <=1% collection perturbation, <=5% heldout stage MAPE,
and <=3% heldout F/B makespan absolute relative error. They are targets, not
guarantees. Clock-aligned real makespan and perturbation evidence remain external
requirements; even a good fit is experimental until they are established.
The exporter cannot safely upgrade a profile to passed on its own. Solver-only
integration testing with an experimental profile must be explicitly labeled;
its OPTIMAL status proves only the supplied model, not real-system optimality.

## Checks

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev /opt/venv/bin/python -m pytest \
  tests/unit_tests/pipeline_parallel/test_slackpipe_stage_measurement.py \
  tests/unit_tests/pipeline_parallel/test_slackpipe_plan.py \
  tests/unit_tests/pipeline_parallel/test_slackpipe_cost_profile.py -q
```

Most new tests are CPU-only; the explicitly marked event/Node gradient test uses
one GPU. Full PP=2 hybrid correctness and perturbation collection use both GPUs.
The target 16B/PP=4 run, clock alignment and old/new plan memory/throughput
comparison require separate target hardware. The local test model cannot certify
these. See the fresh experiment directory's audit report for actual results,
failed attempts and exact commands; historical data and receipts are untouched.
