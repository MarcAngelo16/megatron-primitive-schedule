> Historical scope: this document records pre-five-method code/results. Commands and method IDs below are not current entry points. Use the [current workflow](slackpipe_five_methods.md); historical results keep their recorded commit and labels.

# SlackPipe Validation Coverage

## Nemotron-H BF16 Preparation

**PP=4 implementation ready; PP=4 hardware validation pending.** The local
machine has only two GPUs. No full 8B model, RunPod allocation, or real benchmark
was run for this preparation milestone.

| Check | Local result / boundary |
| --- | --- |
| Native small hybrid BF16 PP1 | PASS: initial/loss/gradient/post-step max differences all 0 |
| Native small hybrid BF16 PP2 P2P and RMA | PASS: all four max differences 0 |
| Existing FP32 homogeneous, heterogeneous and hybrid equivalence | PASS: exact-zero assertions retained |
| BF16 RMA delayed consumption, aligned slots and reuse | PASS; allocation/send/receive dtype checked |
| W4/N8 with B=1,4,8,16 | Structural coverage; B8 has 128 operations |
| W4 with N=4,8,12 | Structural coverage; VPP=1,2,3 and cyclic ownership |
| Nemotron sequence identity | 52 global blocks; fixed SHA-256 fingerprint |
| 8B calibration partition design | Offline full rank 6/6, six near-uniform partitions |
| PP4 FP32/BF16 equivalence, either transport | Expected skip: requires 4 CUDA GPUs |
| Real8B allocation, memory capacity, throughput and pod setup | Hardware validation pending |

BF16 limits are absolute: initial parameters 0, normalized loss 2e-4, every
gradient 2e-3, post-SGD parameters 5e-4. Comparisons cast to FP32 before
subtracting. Observed local differences are zero; these limits are not a claim
about arbitrary BF16 models. Native Megatron BF16 wrappers are exercised.
The full regression suite, including dedicated transport/lifecycle cases, is
run in the existing container: **90 passed / 15 expected skips at PP1**, and
**97 passed / 8 expected skips per rank** for each PP2 transport selection.
The 105 collected cases include two gated PP4 precision cases. C++ no-OR and
OR-enabled CTest each pass 4/4; OR also passes 14 tiny CLI exports.
See the [workflow and audit](slackpipe_pp4_hybrid.md)
for commands, exact methodology, prerequisites and fail-closed receipts.

## PP4/Hybrid Extension

PP4 structure is implemented; this host still exposes only two GPUs. The table
below distinguishes local execution from future four-GPU acceptance tests.

| Feature | Current status |
| --- | --- |
| Hybrid manifest and exact type/global ID/range validation | PASS |
| Native Mamba/attention/MLP PP1 equivalence | PASS, all four max differences 0 |
| Native Mamba/attention/MLP PP2 P2P and RMA equivalence | PASS in deterministic mode; see caveat below |
| PP4/N8/B8 stage/chunk/worker mapping | PASS, structural only |
| Seven P2P groups and fourteen directional RMA channel specifications | PASS, structural only |
| PP4 cache separation, exact ranges and full CLI argument validation | PASS, structural only |
| W4/N8/B8 cost_profile.v2 -> CP-SAT -> production plan.v2 parser | PASS, OPTIMAL; not GPU execution |
| PP4 P2P runtime/numerical equivalence | NOT YET RUN - requires 4 GPUs |
| PP4 RMA runtime/numerical equivalence | NOT YET RUN - requires 4 GPUs |
| Nemotron-H 8B allocation/training/benchmark | NOT RUN |

The four-rank test skips with `requires 4 CUDA GPUs`; this is an intentionally
uncovered hardware requirement, not a passing result. The Base-8K offline
architecture adapter and launch scripts need no weight download.

Strict hybrid tests require deterministic Mamba reductions/autotuning, with
`MAMBA_DETERMINISTIC=1 TRITON_CACHE_AUTOTUNING=0
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` set before imports. Unconstrained runs sometimes
matched exactly but also produced gradient differences up to `3.052409738302231e-6`
and post-step differences `5.960464477539063e-8`, despite identical loss.
The equality assertion was not relaxed; arbitrary nondeterministic kernel runs
are not claimed to be bitwise reproducible. See the [extension audit](slackpipe_pp4_hybrid.md).

## Original Monorepo Audit

This audit starts at published monorepo checkpoint
`5078391de5b76ad510c1faf24eab09549b560aad`. The supported validation environment is
one node, two visible GPUs, NCCL 2.29.7 with working symmetric-memory RMA,
FP32, TP=DP=CP=1, fixed shapes, dropout=0, and untied embeddings/output.
All Megatron runs use `slackpipe-dev`. No runtime or solver behavior is changed
by this audit.

## Observed Coverage

| Feature | PP1 | PP2-P2P | PP2-RMA |
| --- | --- | --- | --- |
| plan.v1 parsing/validation | PASS | PASS | PASS |
| plan.v2 and exact global ranges | PASS | PASS | PASS |
| cost_profile.v1 | PASS | PASS | PASS |
| cost_profile.v2, class fitting and prefix costs | PASS | PASS | PASS |
| model_manifest.v1 and model/profile hashes | PASS | PASS | PASS |
| Homogeneous numerical equivalence | PASS | PASS | PASS |
| Heterogeneous equivalence and global layer/class mapping | PASS | PASS | PASS |
| P2P communicator / logical edge isolation | N/A | PASS | N/A |
| RMA put/wait and persistent mailboxes | N/A | N/A | PASS |
| RMA cached reuse, shutdown and recreation | N/A | N/A | PASS (10 cycles per mode) |
| External solver plan / end-to-end plan.v2 | N/A | PASS (deterministic baseline) | PASS (CP-SAT) |

N/A means the feature belongs to the indicated other configuration, not an
unexercised claimed feature. Schema/cost/manifest unit tests are transport
independent and run in all three invocations. C++ no-OR and OR-enabled CTest
both pass 4/4; the OR build executes the 142-test harness and 14-test OR smoke
subset. No-OR counts alone do not prove CP-SAT coverage: OR-only test bodies
return early in that build. Solver Python analysis/manifest and CLI tests pass
125 and 6 cases respectively, with no skips.

Range-duration arithmetic and solver/evaluator consistency are additionally
asserted by C++ `CostProfile.AppliesRangePrefixAndRoleBiasDurations` and
`JointCpSat.RangeCostProfileDurationsReplayInDeterministicEvaluator`. The latter
executes in the OR-enabled run rather than returning early as it does without OR.

The final whole-suite counts are 61 passed / 10 expected skips at PP1 and
67 passed / 4 expected skips **per rank** in each PP2 transport-selected run.
Transport selection controls both general homogeneous and heterogeneous PP2
equivalence. Dedicated P2P isolation and RMA-only tests retain their explicit
transport in both full-suite runs; a transport-selected run is not exclusively
using that transport for every test.

Every executed equivalence compares initial logical parameters, all expected
losses, every logical parameter gradient (including consistent missing-gradient
checks), and post-SGD parameters. Maximum absolute differences for all four
metrics are zero. PP2 reduces differences across ranks; loss-bearing ranks must
return exactly the expected microbatch count. Trace order must equal plan order.
The same default hand plans execute with both transports. Arbitrary optimized
orders, especially delayed matching, are **not** claimed to make P2P progress.

## Every Original Skip

All 14 original skip cases below are **A: expected / configuration-specific**,
and every one passes in an alternate invocation. There were no observed B
(optional feature unavailable) or C (unintended core) skips on the audited host.
Rank-identical PP2 reports are deduplicated here.

Files are under `tests/unit_tests/pipeline_parallel/`: C =
`test_slackpipe_communication.py`, H = `test_slackpipe_heterogeneous.py`, M =
`test_slackpipe_model_construction.py`, R = `test_slackpipe_rma.py`.
Each row covers a current claimed feature; reason codes quote the exact original
pytest message below the table.

| Original run | File | Test | Reason | Alternate PASS |
| --- | --- | --- | --- | --- |
| PP1 | C | `test_slackpipe_logical_edge_communicators_are_isolated` | W2 | PP2 P2P |
| PP1 | H | `test_heterogeneous_numerical_equivalence[2]` | H2 | PP2 RMA; now also P2P |
| PP1 | M | `test_slackpipe_pp2_numerical_equivalence` | W2 | PP2 P2P and RMA |
| PP1 | R | `test_rma_puts_complete_before_any_consumer_waits` | R2 | PP2 RMA |
| PP1 | R | `test_rma_numerical_equivalence[pp2-equivalence]` | W2 | PP2 RMA |
| PP1 | R | `test_rma_numerical_equivalence[m1-delayed-match]` | W2 | PP2 RMA |
| PP1 | R | `test_rma_existing_solver_plans[M0_equal_joint]` | W2 | PP2 RMA |
| PP1 | R | `test_rma_existing_solver_plans[M2_shared_slopes_joint]` | W2 | PP2 RMA |
| PP1 | R | `test_rma_existing_solver_plans[M3_full_joint]` | W2 | PP2 RMA |
| PP1 | R | `test_rma_existing_solver_plans[measured]` | W2 | PP2 RMA |
| PP2 | H | `test_heterogeneous_pp1_exact_construction` | HC1 | PP1 |
| PP2 | H | `test_heterogeneous_numerical_equivalence[1]` | H1 | PP1 |
| PP2 | M | `test_slackpipe_pp1_constructs_logical_vpp_chunks` | MC1 | PP1 |
| PP2 | M | `test_slackpipe_pp1_numerical_equivalence` | ME1 | PP1 |

- W2: `run with torchrun --nproc-per-node 2` (`WORLD_SIZE != 2`).
- H2: `requires 2 ranks` (`WORLD_SIZE != 2`).
- R2: `requires torchrun with two GPUs` (`WORLD_SIZE != 2`).
- HC1: `PP1 construction requires one rank` (`WORLD_SIZE != 1`).
- H1: `requires 1 ranks` (`WORLD_SIZE != 1`).
- MC1: `run PP=1 SlackPipe construction test without torchrun`.
- ME1: `run PP=1 SlackPipe equivalence test without torchrun`.

The last two messages mean `WORLD_SIZE == 1`, not a prohibition on single-rank
torchrun. PP1 tests cannot represent a single physical worker inside a two-rank
test world; distributed communication tests require both participating ranks.

## Other Gates and Fixes

CUDA guards on the homogeneous construction/equivalence tests and the two-GPU
guard on P2P isolation did not fire. There is no silent NCCL/RMA capability skip:
the real transport tests initialize it and fail on an incompatible stack.
Supported-environment CI must not treat a CUDA/two-GPU skip as coverage.

Four historical RMA plan tests previously skipped with `requires generated
solver-plan artifacts` on a clean clone. This was a latent C-class reproducibility
gap, hidden locally by old campaign output. Their exact native orders and splits
are now small curated [regression fixtures](../tests/unit_tests/pipeline_parallel/slackpipe_fixtures/README.md).
No campaign results or cost profiles are checked in; no solver runs during these
unit tests. Missing fixtures are errors. The existing M1 delayed-match fixture
already made the primary external-plan RMA regression self-contained.

`SLACKPIPE_EXTERNAL_PP2_PLAN` and `SLACKPIPE_HETERO_PLAN` are optional overrides,
not skip gates: without them tests construct deterministic plans. Artifact-path
environment variables only control output locations. The heterogeneous PP2 test
previously hardcoded RMA; it now honors `SLACKPIPE_TEST_TRANSPORT` (retaining RMA
as its default), so a P2P-selected run really exercises heterogeneous P2P.

Solver `test_primary_ablation_runner.py` can skip if its release CLI is absent or
predates the budget schema. Neither condition applies in the OR-enabled run.
No other SlackPipe integration skip sites or inherited skip fixtures were found.

## Reproduction and Evidence

Run all three [README test commands](../README.md#targeted-tests) with `-ra`.
For rank-separated evidence, torchrun can launch `--no-python bash -c` with
`pytest --junitxml="$AUDIT/run.rank$RANK.xml" > "$AUDIT/run.rank$RANK.log" 2>&1`.
Set `AUDIT` to a container-visible directory under ignored
`slackpipe_experiments/`; do not commit raw logs or generated profiles/plans.

The end-to-end audit runs fresh ordinary Megatron heterogeneous calibration
(five warmups, ten measured iterations), exports the resulting cost_profile.v2
with the C++ CLI, and executes `test_heterogeneous_numerical_equivalence[2]`
with `SLACKPIPE_HETERO_PLAN` and the selected transport. P2P uses
`--algorithm uniform-breadth-first`; RMA uses `--algorithm slackpipe --split-mode
global`. Both use B=4, N=4, W=2, L=12. The production parser checks v2 ranges and
model/profile provenance; the runtime validates chunk layer/class identity and
exact worker operation order. This is correctness evidence, not a benchmark.

Run real lifecycle probes separately with two-rank torchrun on
`tests/unit_tests/pipeline_parallel/slackpipe_rma_lifecycle_stress.py`: use
`--cycles 10 --managed-groups --plans PLAN --output OUTPUT`, once with `--reuse`
and once with `--recreate-megatron`. This exercises actual cached runtime
contexts and plan-cache eviction, not just the mocked cache-contract unit test.
Final contexts, mailboxes, and windows must all be zero.

The audit's ignored `audit_summary.json` and `audit_report.md` retain every
per-rank testcase outcome, exact skip reason, alternate passing invocation,
per-test numerical maximum, and feature-to-test mapping. No currently claimed
feature is accepted as covered solely because a test was skipped.
