# Stage and Layer Profiling

This is the current contract for `tools/run_slackpipe_eval.py`. Historical
[stage-quality](slackpipe_profile_quality.md) and
[isolated-quality](slackpipe_isolated_quality_audit.md) reports retain their
original evidence, not current acceptance rules.

## CLI and Selection

The workflow selector is `--phase`; the scheduling algorithm is `--method`.
Logical stages use `--num-stages`, independently of physical `--pp` and solver
CPU `--solver-workers`. The five canonical methods are documented in
[Five-method evaluation](slackpipe_five_methods.md).

| `--profiling-method` | Backends collected | Omitted `--solver-profile` |
| --- | --- | --- |
| `both` (default) | stage, then layer | stage |
| `stage` | stage | stage |
| `layer` | layer | layer |

The parser leaves `solver_profile=None` until normalization, preserving explicit
user selection. `--stage-profile PATH` and `--layer-profile PATH` supply existing
artifacts. A selected backend must be collected or explicitly supplied; there is
no implicit collection or fallback to the other backend.

Inside `slackpipe-dev`, from `/workspace/Megatron-LM`:

```bash
# No profiling or solver prerequisite for a native benchmark.
/opt/venv/bin/python -m tools.run_slackpipe_eval --phase benchmark \
  --method 1f1b --model-config configs/slackpipe_eval/llama_4b.json \
  --pp 2 --num-stages 2 --microbatches 4 --output /tmp/native-benchmark

# Separate stage/layer collection; downstream selection defaults to stage.
/opt/venv/bin/python -m tools.run_slackpipe_eval --phase calibrate \
  --method slack-heuristic --model-config configs/slackpipe_eval/llama_4b.json \
  --pp 2 --num-stages 4 --microbatches 4 --profiling-method both \
  --output /tmp/both-profiles

# Layer-only collection, explicitly selected for the solver.
/opt/venv/bin/python -m tools.run_slackpipe_eval --phase solve \
  --method slack-heuristic --model-config configs/slackpipe_eval/llama_4b.json \
  --pp 2 --num-stages 4 --microbatches 4 \
  --profiling-method layer --solver-profile layer \
  --solver slackpipe/build/no-or/slackpipe_cli --solver-workers 1 \
  --output /tmp/layer-solve
```

These are minimal command forms using a supplied model preset, not a claim that
the preset fits local memory. Use an appropriately sized compatible model config
and fresh output directories. GPU phases require a functioning CUDA environment.
Current stage timing hooks require native interleaving: supported PP=2 or 4,
N>PP, N divisible by PP (VPP>=2), and B divisible by PP. The launcher checks
this before **any prerequisite GPU worker**, including correctness, for a
request that needs new stage collection. An unsupported `stage`/`both` request
is rejected even when layer is selected; it does not first collect layer.
No topology, backend or downstream selection is silently changed. This refactor
does not add timing hooks to the native noninterleaved hot path.
Layer-only profiling supports target N=PP and PP=1. An explicit stage artifact
does not require the target to support new stage collection. Both native
standalone benchmarks are independent of profiling/solver prerequisites, and
execution workers consume their bound plan/profile without collecting again.
For artifact reuse, add `--stage-profile /absolute/stage/cost_profile.json` or
`--layer-profile /absolute/layer/cost_profile.json`; identity, dimensions,
configuration and cost structure must match the target. Explicit legacy artifacts
keep their recorded policy and hash; old `quality.status` does not authorize or
reject reuse. Historical masks are not presented as current-policy measurements.
Rebuilding from saved raw data does not authorize a new GPU collection.

## Separate Artifacts and Failures

`tools/slackpipe_profiling.py` invokes each requested collector once, sequentially.
The collection directory contains separate `stage/` and `layer/` directories,
each with `receipt.json` and its raw/profile artifacts, plus `collection.json`.
No raw file is overwritten and no costs are mixed or averaged. `both` does not
run the solver or benchmark twice.

A runtime-failed nonselected backend leaves a partial collection and the selected
successful profile can proceed. A failed selected backend records
`selected_backend_failed` and blocks dependencies without fallback or retry.
This does not waive pre-launch capability errors for a known unsupported topology.
Fixed warmups and configured sampling counts remain; there is no sampling loop
that waits for statistical stability.

Receipt/context v4 bind backend, explicit selection, configuration, source,
artifacts, sampling policy and relevant parameters. The profile records actual
`measurement_parallel_config` separately from `target_parallel_config`.
Layer measurements use PP=1, VPP=None, TP=DP=CP=1 even when the target has PP/VPP.
The backend policy and calibration context explicitly record
`interleaved-1f1b` for new stage collection, `isolated-unit` for layer collection,
and `artifact-import` for imports, separately from the experiment method.

## Stage Model and Residuals

Implementation: `megatron/core/pipeline_parallel/slackpipe/cost_profile.py` and
`stage_residual.py`, policy `stage-affine-residual-v1`.

After inventory validation and the fixed warmup, every legal F/B operation enters
the initial summary. Each observation is the median of iteration means, each
mean being the mean time per operation, not an iteration total. Groups, stage
ranges, roles, raw paths and stable operation IDs remain traceable.

Forward and backward are fitted independently. For homogeneous layers the
default deterministic construction is:

```text
a = min(center(stage) / layer_count(stage))
bias(stage) = center(stage) - a * layer_count(stage)
prediction(stage) = a * layer_count(stage) + bias(stage)
```

A shared slope plus independent stage intercepts at one fixed partition is not
generally uniquely identifiable regression. This construction is retained rather
than adding partitions or regularization. Explicit core `median` or `percentile`
slope options retain their recorded estimator and nonnegative clipped biases.

For heterogeneous layers the existing nonnegative class/role fit is:

```text
prediction = sum(class_count[c] * class_cost[c]) + role_bias
```

All calibration groups participate directly; there is no consensus vote or
per-observation free intercept. Rank/constructibility checks remain structural.
Fit errors and condition estimates are descriptive, not stability gates.

For every raw operation, using its initial model prediction in the same units:

```text
q_i = (y_i - p_i) / p_i
m = median(q_i)
s = 1.4826 * median(abs(q_i - m))
candidate_i = abs(q_i - m) > k * s AND abs(q_i) > rho
```

The median and scaled MAD are phase-local across the profile. Defaults are
`--stage-residual-k 6.0` and `--stage-residual-rho 0.5`, both finite and positive.
These are project engineering defaults, not a paper's prescription or statistical
guarantee. Predictions must be positive and finite; invalid denominators are not
substituted. Both comparisons are strict, including when MAD is zero.

Compute the mask once from the initial fit. Recompute summaries using the actual
retained count in each iteration and refit at most once. An empty iteration is
recorded with no mean and excluded from the median, never assigned zero. If a
required observation is emptied or the model loses necessary structure, undo the
entire phase mask, retain the initial model/all samples and record
`filter_not_applied` with its structural reason. Unexpected errors propagate.
Initial constructibility failures are errors, not reasons to retry.

Evidence includes raw observations, initial/final models, per-sample initial
predictions/residuals, candidate/applied masks, policy parameters, counts and
rollback reasons. Final residuals do not overwrite the initial mask evidence.
Systematic shifts absorbed by the model may remain undetected.

Summary-only heterogeneous input cannot support operation-level masking. It is
labelled `stage-summary-fit-v1`, with `filter_not_applied` and the raw-unavailable
reason, never `stage-affine-residual-v1`. Explicit historical artifacts retain
their bytes and policy rather than acquiring a new filtering claim.

## Layer Medians and Integrity

`isolated_profile.py` uses policy `layer-median-v1`: measure real layer F/B plus
embedding and final norm/head/loss boundary units, take the median of **all**
legal samples, then expand classes to ordered layers and prefix/range costs.
No spike filter, residual mask, CV gate, cross-stage comparison or retry is used.
Class/execution identity, sample inventory, raw-to-summary reconstruction,
boundary costs and prefix consistency remain checked.

`profile_integrity.validate_profile_integrity` and `publish_profile` replace
statistical admission. Schema, units, dimensions, finite legal costs, hash,
model/configuration, raw inventory and plan correctness checks remain. Publication
refuses to overwrite an existing profile. There is no minimum statistical sample
count, discarded-fraction cap, consecutive-outlier cap, group drift/consensus
gate, RMSE/R-squared threshold or requirement for `quality.status == passed`.
For declared current stage policies the loader reconstructs the initial fit,
single mask and final fit from saved raw observations (or explicitly labelled
summary-only inputs). It checks policy, sample identities, masks and resulting
costs, even if the file has been rehashed. A layer label without isolated evidence,
or a legacy profile merely relabelled as the current strategy, is not accepted.
Reconstruction is an integrity check, not another filtering pass on retained data.

## Measurement and Result Scope

Stage measures synchronized wall time around the full stage F/B call, including
dispatch, compute and completion wait within that scope. Explicit P2P outside
the call and optimizer work are excluded. Layer measures isolated synchronized
current-stream CUDA-event implementation time, including possible launch gaps;
it excludes pipeline waits, P2P and optimizer work. Neither is communication
profiling or a pure FLOP-time estimate.

No communication profiler, bandwidth fit, contention model or transport is added.
The existing model's constant cross-worker dependency latency is not an explicit
communication task or link-competition model. The unified solve path currently
passes zero communication delay; report that zero-cost assumption. Record actual
runtime transport (`nccl-p2p` or experimental `nccl-rma`) without inferring overlap
from its name.

Keep modeled/replay makespan, lower bounds, raw solver status, solver wall time
and real GPU iteration time separate. `FEASIBLE` is not `OPTIMAL`; proof scope is
the configured model/search space. v1 costs use milliseconds; v2 prefixes use
microseconds; solver ticks must retain their declared conversion.
Solve, benchmark and trace receipts propagate `profile_identity` with backend,
file SHA, cost-profile hash and policy, separately from method and transport.

Benchmark summary v2 bases main statistics and `ci95_ms` on independent run
observations of max-rank continuous window wall time divided by measured steps.
`ci95_ms` is a normal 95% **half-width**, not interval endpoints. CUDA-event
statistics are under `cuda_event_summary`; they do not supply the wall-time
confidence interval. One run has a null interval. Benchmark runs do not include
profiling's per-operation synchronization instrumentation.

## Validation Limits

The current container reports `torch.cuda.is_available() == False`, device count
zero, and `nvidia-smi` exit 255 with `Failed to initialize NVML: Unknown Error`.
This does not establish a driver/library mismatch. CPU/CLI/parser tests and C++
replay are useful but do not replace PP=2 solver-to-Megatron numerical equivalence.
See [refactor review](slackpipe_refactor_audit_20261021.md) for actual tests,
attempted GPU failures and remaining checks. Historical paper/GPU evidence is
preserved, not rerun or relabelled by this migration.
