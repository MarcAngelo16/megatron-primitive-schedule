> Historical scope: this document records pre-five-method code/results. Commands and method IDs below are not current entry points. Use the [current workflow](slackpipe_five_methods.md); historical results keep their recorded commit and labels.

# PP4 and Hybrid Readiness

## Nemotron-H 8B Preparation

**PP=4 implementation ready; PP=4 hardware validation pending.** Preparation
does not establish four-rank NCCL progress, 8B memory capacity, or throughput.
No pod was rented, no weights downloaded, and no 8B run performed locally.

### Architecture Authority

`megatron/core/pipeline_parallel/slackpipe/hybrid.py::nemotron_h_8b_config`
is the single architecture helper. Its source is NVIDIA's
[released Base-8K config at revision 253e002](https://huggingface.co/nvidia/Nemotron-H-8B-Base-8K/blob/253e00241ff77421b6d811c971e9cee1b2d824ad/config.json).
The native Megatron adapter supplies 52 hybrid blocks, hidden size 4096, FFN
21504, 32 attention heads, 8 query groups, head dimension 128, RMSNorm epsilon
1e-5, squared-ReLU MLPs, vocabulary 131072, maximum sequence length 8192, no
position embeddings, and untied input/output weights. Mamba uses state 128,
head dimension 64, 8 groups and 128 heads; pinned native defaults supply
convolution 4, expansion 2 and chunk size 256. This is random initialization,
not a pretrained checkpoint conversion.

The exact sequence is
`M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-`.
Its SHA-256 is
`e0cf75d03b16c79cac23da4c217a5d3243d45043f07cb445ab358dad42fc4cf5`.
The unit test locks both this fingerprint and the layer count. Manifests come
from the actual native config; architecture constants are not duplicated in
shell scripts. Precision remains cost-profile execution metadata, not a change
to partition-independent model identity.

### BF16 Audit

- Runtime previously rejected every dtype except FP32. It now accepts FP32 or
  BF16 and rejects mismatched pipeline/parameter dtypes. Missing pipeline dtype
  falls back to `config.params_dtype`, not FP32.
- P2P payload receives already used runtime dtype. Its FP32 communicator warmup
  is only a control exchange, not model data.
- RMA allocation uses the configured dtype, send validation checks that same
  dtype, and page-aligned slot strides/byte statistics use `element_size()`.
  BF16 and FP32 mailboxes cannot share a cached transport.
- Runtime and transport keys already include dtype, shape, B, topology, device
  and group identity. Local detached inputs and backward gradients retain their
  actual dtype; no extra boundary casts were introduced.
- v2 profiles with precision metadata must match model dtype. The pod driver
  also checks sequence length, profile fingerprint and model fingerprint.
- Stage calibration measures actual BF16 native chunks. FP64 regression algebra
  and FP32 loss/metric accumulation are intentional, not payload conversions.
- Numerical helpers unwrap native Megatron BF16 modules and subtract FP32 views
  of parameters/gradients. They do not round differences back to BF16.

Small native hybrid PP1, PP2-P2P and PP2-RMA tests compare the same full ordinary
model against plan-partitioned chunks. Seed/data are fixed, dropout is zero,
and embeddings are untied. Initial values are copied by global logical name;
all ranks verify ownership coverage. The reference accumulates eight ordinary
microbatch losses/backwards and uses the same SGD update. All four local max
differences are **0** in FP32 and BF16. FP32 continues to require exact equality.
BF16 absolute limits are initial=0, loss=2e-4, gradient=2e-3, post-step=5e-4.
The PP4 test executes FP32 first, then BF16, with one result file per rank and
precision. Missing results (including skipped tests) cannot certify correctness.

Structural tests cross B={1,4,8,16} with W=4/N={4,8,12}. They verify total
operations `2*B*N`, FIFO coverage, `stage % W`, `stage // W` and runtime compiled
chunk mapping. W4/N8/B8 has 128 operations; N12 is structural coverage only.

## Pod Setup

Start from `nvcr.io/nvidia/pytorch:26.01-py3` on four A100 SXM GPUs. Commands below
are for the container, never the host. The checkout and output paths are
configurable; the first sequence length is 1024. Setup on a fresh pod has not
been hardware-validated locally. Preserve the container's CUDA/PyTorch stack
when installing the repository's locked dependencies:

```bash
export ROOT=${ROOT:-/workspace/Megatron-LM}
git clone --branch slackpipe/mvp https://github.com/RioReal/megatron-primitive-schedule.git "$ROOT"
cd "$ROOT"
apt-get update
apt-get install -y build-essential cmake ninja-build curl python3-venv
unset PIP_CONSTRAINT
python -m pip install uv==0.7.2
export UV_PROJECT_ENVIRONMENT=/opt/slackpipe-venv
uv venv "$UV_PROJECT_ENVIRONMENT" --system-site-packages
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
uv sync --locked --only-group build --no-install-package torch --no-install-package torchvision --no-install-package triton
MAX_JOBS=2 uv sync --locked --extra training --extra ssm --extra te --group test \
  --no-install-package torch --no-install-package torchvision --no-install-package triton
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export MAMBA_DETERMINISTIC=1 TRITON_CACHE_AUTOTUNING=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
```

The lock pins Mamba 2.3.2.post1 and causal-conv1d 1.6.2.post1. Deterministic
Mamba support must be importable before accepting correctness. Do not use
unversioned dependency upgrades to make a failing environment appear supported.

Build the monorepo's joint solver using the official
[OR-Tools 9.15 release](https://github.com/google/or-tools/releases/tag/v9.15):

```bash
export ORTOOLS_PREFIX=/opt/or-tools
curl -fL https://github.com/google/or-tools/releases/download/v9.15/or-tools_amd64_ubuntu-24.04_cpp_v9.15.6755.tar.gz \
  -o /tmp/or-tools.tar.gz
mkdir -p "$ORTOOLS_PREFIX"
tar -xzf /tmp/or-tools.tar.gz --strip-components=1 -C "$ORTOOLS_PREFIX"
BUILD_JOBS=2 SLACKPIPE_OR_TEST_FILTER='.*' bash slackpipe/scripts/validate_ortools_evaluation.sh
```

**Known stock-image RMA blocker:** direct inspection of the locally available
26.01 image reports PyTorch `2.10.0a0+a36e1d39eb.nv26.01.42222806`, NCCL 2.29.2,
`rendezvous=True`, but `put_signal=False` and `wait_signal=False`. NCCL's version
alone is insufficient. The local validated RMA stack uses PyTorch
`2.12.0a0+0291f960b6.nv26.04.48445190` with NCCL 2.29.7. **Do not rent a pod for
the RMA campaign assuming stock 26.01 will work.** A separately prepared,
compatible PyTorch/TE/Mamba overlay with those APIs must pass preflight and
the transport tests first. Such an overlay was not built or validated here;
the workflow deliberately refuses RMA instead of patching private APIs or
silently changing the requested base image. P2P setup is provided above, but
arbitrary joint-solver orders may require RMA to make progress.

## Explicit Pod Stages

The existing four scripts are retained, with no duplicate role scripts:

| Exact path | Purpose |
| --- | --- |
| `tools/runpod_slackpipe_verify.sh` | Metadata, hardware/dependency checks, four-rank NCCL sanity |
| `tools/runpod_slackpipe_correctness.sh` | Tiny PP4/N8/B8 FP32 followed by BF16 equivalence |
| `tools/runpod_nemotron_baseline.sh` | Ordinary native Megatron random-init BF16 8B smoke |
| `tools/runpod_nemotron_slackpipe.sh` | Plan-derived native Megatron SlackPipe BF16 8B smoke |

They delegate to `tools/run_slackpipe_nemotron_h8b_pp4.py`; all share the same
prerequisites. The master can be invoked directly for every stage:

```bash
export OUT=/workspace/slackpipe-runs/nemotron-p2p
python -m tools.run_slackpipe_nemotron_h8b_pp4 env --output "$OUT"
python -m tools.run_slackpipe_nemotron_h8b_pp4 pp4-correctness --output "$OUT"
python -m tools.run_slackpipe_nemotron_h8b_pp4 baseline-smoke --output "$OUT"
python -m tools.run_slackpipe_nemotron_h8b_pp4 calibrate --output "$OUT"
python -m tools.run_slackpipe_nemotron_h8b_pp4 solve --output "$OUT"
python -m tools.run_slackpipe_nemotron_h8b_pp4 slackpipe-smoke --output "$OUT"
python -m tools.run_slackpipe_nemotron_h8b_pp4 benchmark --output "$OUT"
python -m tools.run_slackpipe_nemotron_h8b_pp4 trace --output "$OUT"
```

For a capable RMA image, repeat in a new output directory with
`--transport nccl-rma` on **every** command. `--slackpipe-transport` is an alias.
Use `--seq-length 2048`, `4096`, or `8192` consistently in a separate campaign.
`--solver PATH`, `--solver-seconds 300`, `--warmups 5`, `--iterations 20` and
`--timeout 1800` are configurable. No source/layout editing is needed.
`--dry-run` reports prerequisites/settings without allocation or success receipt.

`env` requires at least four visible A100 SXM GPUs, CUDA/NCCL/BF16 support,
TE/Mamba imports, and by default 35 GiB free **on each selected GPU**. This is
a conservative preflight floor, not a memory-capacity guarantee. It records
GPU UUIDs, free/total memory, package versions and Git identity, then runs a
four-rank FP32/BF16 NCCL all-reduce. Select the four devices with
`CUDA_VISIBLE_DEVICES`; changing their assignment invalidates environment reuse.

Every stage requires successful predecessors with matching code/config and
artifact hashes. Failed reruns remove downstream receipts; stale files cannot
authorize a run. Child process groups are terminated on timeout or interrupt.
Failures in NCCL, small FP32/BF16 correctness, finite checks, resource cleanup,
8B baseline allocation (including OOM at sequence 1024), regression rank,
parser validation, or hashes stop the workflow. There is no automatic retry,
smaller sequence fallback, precision change or schedule substitution.

## Calibration, Timing and Traces

`tools/slackpipe_nemotron_worker.py` supplies the shared native hybrid worker.
Real runs are fixed W4/N8/B8, BF16; `--tiny` is a separate PP2/PP4 test harness
and cannot create real-pod success receipts. Calibration uses ordinary Megatron
interleaved 1F1B with six near-uniform partitions (boundary shifts at most two
layers), at least ten measured iterations, and synchronized **stage-level**
F/B events. The 8B design has three class columns and three role-bias columns,
rank 6/6. Actual observations are fitted through the existing nonnegative
class-cost regression and exact prefix/range cost builder. No synthetic values
are substituted for measured costs.

`solve` invokes the unchanged joint CP-SAT path with B8/N8/J4 and the fitted v2
profile. Existing C++ evaluation precedes export. The Python validator then
requires plan.v2, 128 operations, placement `[0,1,2,3,0,1,2,3]`, nonempty exact
ranges, the authoritative manifest hash and the actual profile hash/path. The
runtime derives native pipe-separated segments and VPP=2 from that plan.

Benchmark modes are uniform ordinary, optimized-partition ordinary, and
optimized-plan SlackPipe. They use identical per-logical-parameter mock seeds,
stable Mamba state initialization, identical generated tokens and the same
unwrapped-parameter SGD harness. This controlled harness is not a claim about
end-to-end data-loader/checkpoint/optimizer throughput. Full CLI smoke uses
Megatron's native BF16 optimizer. Warmups are excluded; CUDA events are placed
only at iteration boundaries. Calibration, per-operation profiling and trace
capture are disabled. Finite checks run outside the timing window. The summary
reports each step's maximum rank time and mean/median across measured steps.

Trace capture runs separately after benchmark success, with 20 training warmups
and three profiler cycles (wait=2, warmup=2, active=3) by default. It reuses `figure_trace.label_logical_operations`,
`compact_profiler_trace`, `validate_compact_trace`, and `tools.plot_schedule_trace`.
There are per-cycle per-rank raw torch traces, compact activity/envelope JSON,
validated SlackPipe worker order, and a paired baseline/SlackPipe PDF/PNG. No multi-node clock
alignment is attempted.
See the [collection guide](slackpipe_collection.md) for timing boundaries, memory
diagnosis, output schemas and retained independent benchmark runs (`--runs 3`).

Expected artifacts below `--output`:

- `run_metadata.json`, `logs/`, `receipts/`.
- `correctness/{fp32,bf16}.rank{0,1,2,3}.json` (differences, layers and operations).
- `calibration/model_manifest.json`, `calibration_events.json`, `observations.json`,
  `cost_profile.json`, `fit_diagnostics.json`, per-partition layer/result files.
- `solve/slackpipe.plan.json` and the solver's existing result/orders/CSV files.
- `benchmark/{baseline,partition,slackpipe}/runNNN/result.rank*.json`, `benchmark/summary.json`.
- `trace/{baseline,slackpipe}/RUN/METHOD/*.cycleCCC.{torch,compact}.json`,
  per-rank collection metadata/samples, `trace/timeline.{pdf,png}`, `figure_report.json`.

Generated outputs, builds and dependency environments are not committed.
All local Megatron tests run in `slackpipe-dev`; the stock-image API probe above
only inspected its PyTorch installation, not Megatron execution.

### Preparation Validation Record

The BF16 worker was exercised locally using its tiny PP2 fixture: six ordinary
calibration partitions produced a measured v2 profile; the unchanged C++ joint
solver exported an OPTIMAL plan.v2; production parsing, RMA execution and
BF16 numerical equivalence passed with all four maximum differences zero.
Separate ordinary/SlackPipe captures passed envelope/order validation and
produced a paired PDF/PNG. These are harness correctness checks, not an 8B
performance experiment.

The final 105-case SlackPipe suite reports 90 passed / 15 expected skips at
PP1 and 97 passed / 8 expected skips per rank for each PP2 transport selection.
Tiny BF16 full native `pretrain_hybrid.py` runs also take a finite, non-skipped
SGD step: ordinary PP2/N2 and SlackPipe PP2/N4 RMA. The ordinary CLI retains its
upstream prohibition on non-overlapped PP2 interleaving; it is not bypassed for
this smoke test. Ordinary interleaved calibration is separately tested through
the existing core scheduler harness. Full target PP4/N8 CLI arguments pass
parsing/config validation without allocating the 8B model.

RMA delayed-consumer tests exercise both element sizes and three persistent
iterations. Ten reuse cycles and ten Megatron-context recreation cycles with
the solver plan ended with zero live contexts, mailboxes and windows on both
ranks (only WORLD remained before final process-group destruction). Both C++
CTest configurations pass 4/4; OR validation also passes 14 CLI smoke exports.
Hardware preflight on the local two-GPU machine correctly writes its failure
metadata and refuses certification; an uncertified baseline invocation fails
before model allocation. No four-GPU result is represented as a local pass.

During harness testing, eager initialization of unused Megatron communicators
exhausted the small local GPUs. The worker now binds each rank's device, warms
WORLD, and retains lazy subgroup initialization, as the established test
harness does. Rank-local exceptions are printed immediately and propagated to
torchrun, instead of attempting collective teardown while a peer is still in
model code. Successful runs still explicitly verify transport shutdown.

### Files In This Preparation

Paths below are relative to the repository root. `runtime/` abbreviates
`megatron/core/pipeline_parallel/slackpipe/` and `tests/` abbreviates
`tests/unit_tests/pipeline_parallel/`.

| File | Change |
| --- | --- |
| `runtime/communication_rma.py` | BF16 allocation, metadata checks and byte-correct aligned mailboxes |
| `runtime/schedule.py` | Config-derived dtype and FP32/BF16 guard |
| `runtime/manifest.py` | Reject cost-profile/model precision mismatch |
| `runtime/hybrid.py` | Clarify the existing authoritative architecture helper |
| `runtime/README.rma.md` | Precision support and validation boundary |
| `tests/test_slackpipe_hybrid.py` | Native BF16 wrappers, tolerances, finite checks and rank acceptance artifacts |
| `tests/test_slackpipe_model_construction.py` | Shared wrapped-model parameter mapping and FP32 difference metrics |
| `tests/test_slackpipe_rma.py` | FP32/BF16 delayed consumption, alignment and persistent reuse |
| `tests/test_slackpipe_topology.py` | B/N matrix, sequence fingerprint, BF16 cache separation and launcher arguments |
| `tests/test_slackpipe_nemotron_workflow.py` | Calibration rank, target-plan provenance and fail-closed gate tests |
| `tools/slackpipe_hybrid.py` | BF16/1024 native CLI launch, atomic JSON helper, no unconditional tracing |
| `tools/slackpipe_nemotron_worker.py` | Shared native calibration, controlled measurement and trace worker |
| `tools/run_slackpipe_nemotron_h8b_pp4.py` | Explicit master stages, preflight, receipts, solver and plotting integration |
| `tools/runpod_slackpipe_verify.sh` | Delegate environment verification to the master |
| `tools/runpod_slackpipe_correctness.sh` | Delegate ordered FP32/BF16 correctness to the master |
| `tools/runpod_nemotron_baseline.sh` | Delegate gated ordinary smoke to the master |
| `tools/runpod_nemotron_slackpipe.sh` | Delegate gated plan-derived smoke to the master |
| `README.md` | Target workflow, BF16 status and stock-image RMA blocker |
| `docs/slackpipe_validation_matrix.md` | Current local passes, tolerances and hardware gates |
| `docs/slackpipe_pp4_hybrid.md` | Setup, methodology, artifacts, audit and file map |

## Previous PP4 Extension Record

The following audit records the preceding FP32 milestone. Its historical test
counts and validation boundaries are not the BF16 preparation results above.

## Audit Before Edits

- `schedule._validate_supported_parallelism` rejected PP outside {1,2}; two
  SlackPipe-only guards in `training.arguments.validate_args` did the same.
  These now accept positive PP while keeping TP/DP/CP=1 restrictions.
- RMA required exactly two world ranks, created every NCCL group with `[0,1]`,
  and addressed all puts/waits with `1-rank`. Channels now derive their ordered
  source/destination and sorted subgroup ranks from each logical edge.
- RMA initialization used a world barrier per channel. Nonmembers now skip
  allocations; warmup synchronizes the subgroup. The persistent world Gloo
  completion barrier remains necessary before mailbox reuse.
- P2P already created distinct logical-edge groups, but warmup and destruction
  assumed membership in every group. Every rank still creates every group in
  order; only members retain, warm and destroy them.
- Runtime local-stage enumeration and message extraction were already generic.
  Compiled operations now also expose previous/next workers and edge IDs.
- RMA cache keys already include world/PP group identities, world size, plan W,
  rank, all logical edges/placement, B, shape, dtype, transport and device. N is
  determined by the full topology. Compatible splits/orders intentionally reuse
  mailboxes. Compiled-plan keys additionally include resolved path and file
  mtime/size, PP rank, execution mode, shape and dtype. New tests distinguish
  PP2/PP4, B, device, rank, shape and dtype contexts.
- Plan validation already handled arbitrary positive W, exact ranges, DAG and
  FIFO dependencies. The existing cyclic/divisibility validator is reused.
- Existing GPU tests intentionally gate on PP1/PP2 and the isolation fixture
  expects three edges. They remain regressions, not generalized coverage claims.
- The full training CLI had duplicate dataclass/manual registration of three
  SlackPipe flags. They are now excluded from automatic registration and both
  future launch commands pass the real argument parser and validator.
- CLI-to-core config conversion enabled pseudo-deallocation unconditionally,
  but SlackPipe retains full outputs for arbitrary backward order. The config
  adapter now disables that optimization only for SlackPipe. The native hybrid
  pretraining entry explicitly closes persistent SlackPipe resources on success.

## Four-Worker Topology

Rank 0 owns stages 0,4; rank 1 owns 1,5; rank 2 owns 2,6; rank 3 owns 3,7.
Local chunk is `stage // W`; worker is `stage % W`. Seven logical edges map to
`(0,1), (1,2), (2,3), (3,0), (0,1), (1,2), (2,3)`. Repeated rank pairs retain
distinct communicators. RMA has fourteen directional channels globally and
6,8,8,6 local channels on ranks 0,1,2,3. Peer indices are positions in the sorted
two-member subgroup, including the wraparound edge. Groups/windows are created
outside the operation loop and released by collective explicit shutdown.

## Pinned Hybrid Mechanism

- `pretrain_hybrid.py` uses `hybrid_builders.hybrid_builder` and the legacy
  `training.get_model` VPP construction path.
- `core/models/hybrid/hybrid_model.py` parses the hybrid pattern and calls
  `hybrid_layer_allocation.select_pipeline_segment`. Segment index is
  `vp_stage * pp_size + pp_rank`; its global offset sums all preceding segments.
- `hybrid_block.HybridStack` instantiates layers with one-based global
  `layer_number`, without adding a second PP offset.
- `hybrid_layer_specs.hybrid_stack_spec` maps `M` to `MambaLayer`, `*` to a
  `TransformerLayer` with `SelfAttention` and an identity MLP, and `-` to
  `MLPLayer`. Pinned symbols also include GDN, DSA and MoE, but these and MTP
  are deliberately rejected by the SlackPipe hybrid adapter.
- The newer `training.models.hybrid.HybridModelBuilder` still rejects VPP;
  these launchers do not use that path. No upstream HybridModel or ordinary
  scheduler algorithm was changed.

Hybrid manifests derive from the actual config and unpartitioned pattern.
Partition boundaries are not part of model identity. Stage IDs, types and class
counts describe exact half-open global ranges. Runtime checks unwrap Megatron
wrappers before inspecting native layers. The tiny fixture is `M-*MM-*-M-*-`,
12 blocks, hidden size 128, sequence length 64, eight microbatches. Its PP4 cuts
are `[0,1,3,4,6,7,9,10,12]`. PP1/PP2 use four nonuniform chunks.

## Solver Proof

A synthetic (not measured) class-cost profile was fitted through the existing
`build_heterogeneous_cost_profile` regression and expanded to all twelve global
layers. The unchanged OR-enabled CLI ran `joint-unrestricted-no-overlap` with
`--B 8 --N 8 --J 4 --L 12 --ratio-num 1 --ratio-den 1
--time-limit-seconds 15 --num-workers 2 --random-seed 1 --require-optimal false`.
The evaluator validated the result before plan export. Result: OPTIMAL, split
`[1,2,1,2,2,1,2,1]`, 128 operations, predicted abstract makespan 2520.
The production parser accepted plan.v2 and validated model/profile fingerprints.
This proves schema/solver compatibility, not GPU execution or measured speed.

Actual tiny-hybrid stage calibration also completed on PP2 with the ordinary
interleaved schedule: six partitions, 24 aggregated stage observations, all
three native block classes, and regression rank 6/6. Its measured profile was
consumed by the unchanged joint solver, then the exported plan.v2 executed with
RMA and all four numerical differences zero. Calibration and solver artifacts
remain under `/tmp`, not in the repository.

## Local Regression Record

- The expanded suite contains 83 tests: PP1 runs 71 and skips 12; each PP2
  transport selection runs 77 and skips 6 per rank. All applicable cases pass
  with deterministic hybrid kernels. Skips retain explicit world/device reasons.
- Native hybrid initial parameters, loss, every gradient and post-SGD parameters
  match exactly at PP1 and PP2. Strict comparisons were not loosened. Enable
  `MAMBA_DETERMINISTIC=1 TRITON_CACHE_AUTOTUNING=0
  NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` before imports: nondeterministic kernel
  runs showed repeat-dependent differences, documented in the validation matrix.
- C++ no-OR and OR-enabled CTest pass 4/4 each; OR validation additionally passes
  its 14 tiny CLI export/evaluator smoke cases. No C++ semantics changed.
- Ten persistent-cache reuse cycles and ten Megatron-context recreation cycles
  pass with the measured solver plan. Both ranks end with zero live contexts,
  mailboxes and windows; the only remaining process group is WORLD before exit.
- Tiny full `pretrain_hybrid.py` CLI training completed on P2P and RMA, using
  mock tokens and native three-type blocks, never 8B weights. The RMA launch
  takes a nonzero-learning-rate SGD step and closes its SlackPipe resources.
  The upstream CLI still warns that WORLD is not explicitly destroyed at exit;
  targeted lifecycle harnesses explicitly destroy their process groups.
- Full baseline/SlackPipe Nemotron argument parsing, config/manifest identity,
  Python import ordering, formatting of new helpers, shell syntax, and
  `git diff --check` pass. Four-GPU tests were intentionally not executed.

## RunPod Acceptance

The real four-rank hybrid test checks ownership, VPP indices, pre/post roles,
global IDs/types, complete logical parameter coverage, initial values, loss,
every gradient, one SGD step, exact traces and empty transport state after
shutdown. Run it separately with P2P and RMA via the documented scripts.
On fewer than four visible devices it explicitly skips, never reports PASS.
The two-GPU host cannot establish PP4 NCCL progress, subgroup initialization,
RMA addressing or numerical correctness. Run those checks on A100 SXM x4 before
attempting the offline adapter's full 8B one-step smoke launch.

No weights were downloaded. The Base-8K architecture adapter records the exact
released source revision. Full 8B allocation/activation capacity, eight-stage 8B
execution, long sequences and multi-node remain unvalidated. P2P's known
delayed-matching progress limitation is unchanged; arbitrary optimized orders
must also be validated with the experimental RMA path.

## Changed Files

Paths below are relative to the repository root; `runtime` denotes
`megatron/core/pipeline_parallel/slackpipe/`.

| Files | Change |
| --- | --- |
| `runtime/topology.py` | Pure logical edges, directional channel identities, cyclic local chunks |
| `runtime/communication.py` | Deterministic group creation with nonmember-safe retention |
| `runtime/communication_rma.py` | Generic endpoint groups and subgroup-local RMA peers |
| `runtime/hybrid.py` | Pattern/range helpers and sourced offline 8B configuration |
| `runtime/manifest.py` | Hybrid identity, native type/config/shape validation, wrapper unwrapping |
| `runtime/schedule.py` | Positive PP, compiled neighbors/edges, hybrid validation and feature guards |
| `megatron/training/arguments.py` | Generic PP guards, hybrid plan layout, duplicate CLI registration fix |
| `megatron/training/argument_utils.py` | SlackPipe-specific output retention configuration |
| `hybrid_builders.py`, `pretrain_hybrid.py` | Bind the global pattern and explicitly close SlackPipe after training |
| `tests/unit_tests/pipeline_parallel/test_slackpipe_topology.py` | Structural topology/cache/range/profile and full CLI tests |
| `tests/unit_tests/pipeline_parallel/test_slackpipe_hybrid.py` | Native strict PP1/PP2 and gated PP4 equivalence fixture |
| `tools/slackpipe_heterogeneous_experiment.py` | Ordinary tiny-hybrid stage calibration option |
| `tools/slackpipe_hybrid.py` | Offline manifest, generic class fitting and validated launch commands |
| `tools/runpod_slackpipe_verify.sh`, `tools/runpod_slackpipe_correctness.sh` | Pod capability checks and gated four-rank correctness launch |
| `tools/runpod_nemotron_baseline.sh`, `tools/runpod_nemotron_slackpipe.sh` | Future random-init 8B one-step launchers |
| `README.md`, `runtime/README.rma.md` | Workflow, topology, prerequisites and limitations |
| `docs/slackpipe_validation_matrix.md`, this report | Validation boundary, audit and results |

The C++ source tree and ordinary Megatron schedules are unchanged. The two
untracked campaign scripts are preserved, and generated artifacts are excluded.
