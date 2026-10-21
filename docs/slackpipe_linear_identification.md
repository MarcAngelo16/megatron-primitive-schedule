# Linear Profiler Identification Diagnostic

This opt-in study tests linear structure rather than fitting a scalar to a
previous whole-stage composition table. It does not change native scheduling,
production estimators, quality thresholds, filters, or solver behavior. No output
is an automatically approved cost profile.

## Frozen Design

Use the existing `slackpipe-dev` container and `/opt/venv/bin/python`. The local
experiment targets two 6 GiB RTX A2000 GPUs, not A100/16B performance. The model
is exactly `M*-M*-M*-M*-`, twelve native hybrid residual units, H64, FFN128,
sequence32, microbatch1, FP32, TP/DP/CP1. Here `M` is a full Mamba residual unit,
`*` a native attention residual unit, and `-` a native MLP residual unit; these
are not three full Transformer blocks. Complete manifest class signatures are
validated before simplifying their count columns to Mamba/MLP/attention.

Pipeline execution uses PP2/VPP2, B4, cyclic placement, native interleaving with
microbatch group2, full backward, and SGD lr0.001. Parameters use the existing
global-name/seed1234 initializer. Inputs and labels use the unchanged native
harness generator. Dropout is zero. Model construction, initialization and
communication are not modified by the main collection entry.

Metadata-only partition design:

| Role | Splits |
| --- | --- |
| T0-T4 train | `[1,1,4,6]`, `[2,2,6,2]`, `[4,1,1,6]`, `[6,2,2,2]`, `[2,4,4,2]` |
| V0-V4 heldout | `[1,3,3,5]`, `[2,5,1,4]`, `[1,7,2,2]`, `[1,8,1,2]`, `[2,3,5,2]` |
| U0 control | `[3,3,3,3]`, never fitted |

Controlled bodies use lengths 1-8 and starts 0,1,2, preserving actual global
layer IDs and original order. Lengths 1,2,4,6 train; 3,5,7,8 are held out.
Balanced three/six-unit permutations test equal counts with different order.
Only training cases from run0/1 fit coefficients. Run2 is independently held
out. Both devices participate in every run, avoiding GPU/run confounding.
Partitions and modes rotate deterministically before seeing measurements.

Before sampling, `prepare` checks ranks and stores matrices, singular values,
condition numbers, correlations, class signatures, native worker orders,
train/heldout identities, source snapshots, dirty diff and hashes. H1 uses
length as a metadata-only proxy until the measured positive unit costs are
available; the actual scaled matrix is checked again at analysis.

## Measurement Objects

- **Unit:** actual single native residual module F/B, not a divided stage
  measurement. Measure each type directly on both devices in all three runs.
- **Controlled body:** one complete continuous native body forward and one
  autograd backward entry. No embedding/head/loss. Fixed parameters and a
  seed5678 input/upstream with the upstream scaled by 8192; no timed optimizer.
  Exact saved input/upstream replay checks every layer's gradients against the
  connected body, then compares identical SGD operations separately.
- **Pipeline sync:** unchanged complete B4 SGD step, with the old measurement
  semantics: two extra device drains per complete F/B, outside P2P. Pre-drain
  is excluded from stage wall. GPU event and synchronized host wall are separate.
- **Pipeline events:** same complete step, preallocated AND first-recorded CUDA
  events, no extra F/B or per-step device drain. Read at window completion.
- **Off:** no F/B events or profiler. Continuous full-step wall window with
  rank barrier/device completion at its boundaries; use max rank window / 30.

Every successful condition retains 30 warmups and all30 measured repetitions.
A unit/body repetition is one F/B, whereas a pipeline repetition is a complete
four-microbatch optimizer step. Controls use fixed weights, not the evolving
pipeline's SGD state: they establish a deliberately narrower hypothesis.
No cache clearing or event initialization occurs between warmup and measurement.
Models are reconstructed and cache may be cleared BETWEEN controlled cases.

CUDA events include gaps/waits in the measured compute stream, not pure kernel
time. Coverage of unjoined side-stream computation is not established. Enqueue
wall, synchronized wall, and device intervals are never mixed in a fit.
Internal compiled activations/fusion remain enabled; no CUDA Graph is introduced.
All jobs run sequentially, never concurrently on a device.

## Identification and Admission

The fixed unit table is the arithmetic mean of direct unit measurements on both
GPUs in run0/1. It is never adapted using a validation run. For each phase:

```text
H1: Y = a * sum(n_t*u_t) + b
H2: Y = sum(a_t*n_t*u_t) + b
H2-boundary: H2 + b_first*I_first + b_last*I_last
```

First fit H1/H2 to interior bodies/stages; only then fit boundary terms using
all training pipeline stages. No stage/partition/test-run free parameters.
`theta_t = a_t*u_t` is the equivalent count parameterization, not an independently
identified intrinsic kernel cost. Measured `u_t` anchors the scale. Two arbitrary
unknown affine transformations of two profilers would not recover absolute cost.
With free per-type slopes, H2 predictions are invariant to rescaling positive
`u_t`: H2 is exactly count regression. A better H2 fit alone does not establish
that paired unit measurements supplied extra independent equations.

Use overdetermined SVD least squares, all samples and arithmetic means. Report
actual rank/conditioning, positivity without clipping, coefficients by training
run and leave-one-training-case-out. IID standard errors are descriptive only:
two independent training runs cannot support strong population-level intervals.
MAPE, signed relative bias, maximum relative error and every heldout prediction
are retained. The 5% MAPE goal does not alter selection or estimation.

An independent audit records actual tokens/labels/microbatch, global-layer
inputs/outputs/input gradients, pre-SGD parameter gradients and post-SGD weights
at step0 and steps29/30 (after full warmup). Exact FP32 equality is the declared
gate. If it fails, pipeline fits are descriptive diagnostics, not equivalent-
training validation. Controlled bodies may still provide valid narrower evidence.
No new order is tested before this gate succeeds.

The q-entry control reuses identical disconnected graphs, exact saved inputs
and upstreams, zeroed gradient buffers, and one outer B interval. It requests
`retain_graph=True` and changes only root grouping. The installed TE may clear
backward state despite retained autograd graphs; such a failure is saved, and
**no h coefficient is fitted**. Do not substitute different graph semantics
silently or infer entry overhead from the other fits' bias.

The recorded study adds an explicitly separate `SUPPLEMENT_FREEZE.json` after
that API failure. `slackpipe_linear_entries` rebuilds isomorphic disconnected
graphs before each B event, consumes each graph once, and varies only the grouping
of its backward roots. Exact gradient checks gate its separate five-column fit.
All original training cases, both GPUs, and three runs are retained. Run0/1 fit;
run2 is held out. Its `analysis/entry_supplement.json` is NOT a replacement for
the original retained-object experiment and does not recalibrate the main fits.
An estimated h describes this graph protocol, not pure CPU or pipeline overhead.

`slackpipe_linear_boundary_check` is a separate, post-discovery audit entry that
switches only to existing non-batched P2P. It must never relabel or pool the
main frozen batched-P2P measurements. It does not patch the native scheduler.

## Reproduction

Use a fresh output directory; existing files are never overwritten. Repeat the
environment prefix for `prepare`, `audit`, and `run`:

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev env \
  OMP_NUM_THREADS=1 MAMBA_DETERMINISTIC=1 TRITON_CACHE_AUTOTUNING=0 \
  NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 \
  /opt/venv/bin/python -m tools.slackpipe_linear_identification prepare \
  --model slackpipe_experiments/calibration_transfer_20261002/study-v2/model.json \
  --output slackpipe_experiments/new-linear-study
```

Then run `audit --output ...` with the same entry/environment; summarize it using:

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev /opt/venv/bin/python \
  -m tools.slackpipe_linear_analysis audit --root slackpipe_experiments/new-linear-study
```

Inspect the gate, then use `run --output ...` with the collection entry and
environment. This takes the documented diagnostic-only fallback if equivalence
failed; it does not certify pipeline predictions. Analyze with:

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev /opt/venv/bin/python \
  -m tools.slackpipe_linear_analysis fit --root slackpipe_experiments/new-linear-study
docker exec -w /workspace/Megatron-LM slackpipe-dev /opt/venv/bin/python -m pytest \
  tests/unit_tests/pipeline_parallel/test_slackpipe_linear_identification.py -q
```

The model JSON path refers to the existing local archive, not a shipped model
checkpoint. A new checkout must supply an equivalent model configuration.
Exact executed commands, logs and exit statuses are in each result's `commands/`.

The supplemental entry command, repeated sequentially for run0/1/2 on both GPUs
with the same environment prefix, is:

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev env \
  OMP_NUM_THREADS=1 MAMBA_DETERMINISTIC=1 TRITON_CACHE_AUTOTUNING=0 \
  NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 \
  /opt/venv/bin/python -m tools.slackpipe_linear_entries \
  --root slackpipe_experiments/new-linear-study \
  --output slackpipe_experiments/new-linear-study/q-fresh/run0.gpu0 \
  --run 0 --device 0
docker exec -w /workspace/Megatron-LM slackpipe-dev /opt/venv/bin/python \
  -m tools.slackpipe_linear_analysis entries --root slackpipe_experiments/new-linear-study
```

Outputs include `FROZEN.json`, `design*.json`, `numerical_audit.json`,
`controls-v2/run*.gpu*/units.json`, per-body raw samples and correctness records,
`pipeline/run*.mode/partition/` raw operations/window metadata, and `analysis/`
with fixed units, coefficients, predictions, grouped residuals, plots, numerical
admission and independent-run perturbation intervals. All original failed
attempts and amendments remain separate. `FINAL_MANIFEST.json` binds final
analysis/code/config/data hashes. No duration replay is called measured makespan.
