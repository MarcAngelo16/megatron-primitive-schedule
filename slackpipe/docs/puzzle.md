# Puzzle Early Hints

> Historical record. Puzzle and external CP-SAT hint paths are not part of the
> current five-method implementation. Use the original experiment source
> snapshots for these results, not a relabelled `slack-heuristic` run. Current
> commands are in [the five-method guide](../../docs/slackpipe_five_methods.md).

## Source and Naming

`src/puzzle_hint.cc` implements the supplied 2026-10-05 puzzle-anytime strategy
under the public name `puzzle`. The subsequently supplied `anytime_hint.cc`
matches the specification's SHA256
`6be918ca7c9593bd3bb31765f060a6f61810ebf7d38d92d17837f11a973cac17`.
Source comparison corrected transfer enumeration, composite directions,
archive seeds, local/global quota separation, and phase-stream edge cases.
The backward-completion tie-break accumulates in operation-ID order using
`long double`, as in the reference, rather than an overflowing Tick sum.

An 18-case, evaluation-limited comparison against that supplied source matched
partitions, every worker order, makespans, strict improvement values, generated
and replay counts, and local-round counts. It covered asymmetric affine costs,
communication, N=24 sparse search and B=65 streams. The test-only reference
adapter set every option explicitly: dependency clusters, seed=1, structural
bound and block seeds enabled, long-partition expert disabled. Its missing
header defaults were not inferred. A checked-in regression pins the nonuniform
40,000-replay trajectory. Wall-clock budgets do not promise identical results
across implementations, machines or compiler modes.

The reference header and old deterministic `puzzle_hint.cc` were not supplied.
The internal expert remains reconstructed from section 14, shares implementation
with the new search, and cannot recurse through the public entry point. Its
source-level parity is not established. `hybrid-slack` is a different, existing
initializer; it must not be mislabeled `legacy-puzzle`. Current v2 range costs,
explicit placement, caller predicates, shared deadlines and independent final
validation remain repository-specific adaptations, not blindly copied from
the reference's cyclic/affine implementation.

## Algorithm and Accounting

A state comprises a contiguous positive partition and every worker's complete
ordered F/B list. A cached, linear-time DAG replay includes data edges with
communication lag, phase FIFO, and worker serialization. Costs are recomputed
in batches of 2N, preserving independent forward/backward functions, affine
biases and range-profile prefix sums. Transferring layers does not imply
transferring an equal compute cost. The generic evaluator and independent
schedule validator check the final schedule. Diagnostic mode additionally
compares every completed cached replay (including cycle rejection) with the
generic evaluator.

Feasible states are compared by `(makespan, sum_backward_end, partition,
worker_orders)`; only strictly decreasing makespans generate convergence
events, following an initial baseline point. Every successful replay can
update the global incumbent, including candidates not accepted by a local step.

The search starts with existing uniform BFS, or an admissible caller reference.
The ordinary warm phase ends at `min(2, 0.2*H)` seconds from search creation.
Wavefront, fill/drain and large-N coordinated block seeds feed partition
descent and joint refinement. Actions include multiscale transfers, critical
operation insertions and phase streams. Small-N refinement also tries
dependency-ranked action pairs/triples and coordinated same-worker transfers.
Feasible individually non-improving actions remain eligible for combinations.

After warm start, an archive of at most 12 states feeds four seeded perturbation
families and shared repair quotas (3072 replays for N<=16; 1024 otherwise).
Reaching a local round/neighborhood limit does not stop the anytime loop.
The RNG is the specified uint64 xorshift initialized by `20261005 + seed`.
Large-N search skips dense stage matrices and composites, and uses sparse
partition proposals. Distinct-phase streams are rebuilt in one pass. Fill/drain
uses a priority queue with stage-ID tie breaking; this also preserves the old
expert's scan-selection rule while avoiding its repeated scans.

The partition-cost cache clears on insertion at
`max(16, floor(4194304/(3*N*sizeof(Tick))))` entries. This is an approximate
payload target, not a process-memory bound. The internal legacy expert retains
its unbounded cache. At N<=16 and floor(L/N)>=32 that expert can consume 10% of
the remaining search allowance, including its final validation, and shares
the parent's replay cap. Ordinary warm start does not get a new clock afterward.

## Budgets and Certificates

`PuzzleHintOptions` defaults to 60 seconds and 100,000,000 complete replays.
Standalone time must be in (0,60]. Search uses
`H = allowance - min(1, 0.05*allowance)` to reserve final verification time.
Inside original SlackPipe and direct joint, the existing 10% preparation
allowance and parent deadline clamp this budget. Local deadlines and quotas
cannot extend it. Checks are cooperative: one replay or final verification can
overrun a tiny budget. A safe baseline is still constructed if the parent has
just expired. Parsing and serialization are outside hint search time.

The structural lower bound is available only for common per-stage forward and
backward slopes in the supported affine model (F and B may differ). Integer
capacity bisection minimizes the largest F, B, or terminal F+B phase duration.
The bound combines chain/spacing and total work per active worker. Range costs
and stage-dependent slopes explicitly leave this certificate unavailable.
Equality with a fully validated feasible schedule permits native `OPTIMAL`;
otherwise `FEASIBLE` means deadline or evaluation-cap exhaustion, not proof of
neighborhood or global optimality. This never upgrades CP-SAT's raw status.

## Output and Reproduction

Standalone JSON keeps existing schedule/result fields and adds puzzle budget,
termination and replay accounting. `incumbent_trace` preserves every strict
improvement. `duration_model_ticks` is a self-contained snapshot of evaluator
input arrays, not a new calibration profile or changed v1/v2 cost semantics.
It allows `validate-result` to replay nonuniform-profile output without needing
the original profile path. Costs and makespan are solver ticks; wall time is
CPU search elapsed seconds, not measured GPU iteration time.

Integrated SlackPipe/joint JSON includes the full puzzle result under
`early_hint`, preserving its trace separately from CP-SAT's convergence events.
Existing external-incumbent validation and activation-cap rejection remain in
place. Worker-balance restrictions also gate hints and fallback availability:
an inadmissible early solution cannot tighten a restricted model's horizon.

Build/test commands are in the root README. Focused commands inside the same
development container are:

```bash
cmake --build slackpipe/build/no-or -j2
slackpipe/build/no-or/slackpipe_core_tests --gtest_filter='Puzzle.*'
python3 slackpipe/tests/result_validator_cli_test.py slackpipe/build/no-or/slackpipe_cli
slackpipe/build/no-or/puzzle_hint_benchmark /tmp/puzzle-comparison-new 1
```

The benchmark refuses an existing directory. Its fixed matrix is B=8,W=4,
F=1,B=2,communication=0 with (N,L)=(8,64),(8,512),(24,192). It compares uniform
BFS, existing hybrid-slack and puzzle, assigning the two search methods the
same wall allowance. Uniform BFS is construction only; hybrid-slack may finish
its finite neighborhoods early and does not expose an improvement trace.
Every row saves native output and an independent validation result. Invalid
legacy schedules must be reported as invalid, not as successful baselines.

CSV stdout columns are case, method, B, N, W, L, allowance_seconds,
measured_cpu_search_seconds, modeled_makespan_ticks, independent_validation,
complete_replays, termination_reason. Comparisons are CPU solver diagnostics,
not GPU performance measurements or claims of speedup under 60-second budgets.
