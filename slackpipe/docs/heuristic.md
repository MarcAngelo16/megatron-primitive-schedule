# Heuristic Algorithm

`heuristic` is an independent C++ early-hint method. It does not rename Puzzle,
call Puzzle as an initializer, or change the defaults of existing algorithms.
No external reproduction package is required.

## Search Contract

The selectable state is a complete contiguous partition and all worker-local
orders `(P,O)`. Placement is fixed cyclic; each operation is a complete F or B,
encoded `id=b*(2*N)+q`, with backward phases in reverse stage order.
Costs are the existing integer-tick `Instance` costs: uniform, affine, or
heterogeneous prefix sums with role bias. Every changed partition recomputes
all affected F/B ranges, including intermediate boundaries.

Replay includes the data chain and its communication lag, phase FIFO, and all
worker-order edges. Invalid inventory, placement, FIFO, or a combined-DAG cycle
is infeasible. The final schedule is checked by both the general evaluator and
the independent result validator. Debug mode cross-checks every search replay,
including operation start/end times.

All accepted local moves use the strict integer lexicographic comparator:

```text
(makespan, sum of every backward completion time, partition, worker orders)
```

Equal-makespan secondary improvements are allowed. They do not reset outer
staleness. There is no epsilon or implicit best update inside replay.

Initialization is, in order:

1. Uniform-by-layer-count BFS, sorting each worker by `(b+q,-q,b)`.
2. For large N, the specified first/last-block partition starts.
3. Both diagonal shapes with strides `0,.25,.5,.75,1,1.5,2,4`.
4. Fill/drain partitions with beta `.5,1,1.5`, strides `0,.5,1,2`.
5. PartitionDescent on all feasible deduplicated seeds, refinement of up to
   four optimized seeds, refinement of the best, and worker-local restarts.

Initialization uses `K=min(32768,E)` evaluations and at most
`min(2s,.2*T_search)`. It explicitly updates its best from all fully replayed
feasible candidates. The seed collection is destroyed on exit.

After initialization there is exactly one parent, one xorshift64 RNG, and only
Combined perturbation. Partition moves use the modified trial partition;
order actions are generated once from the original parent's total float.
The first sampled action is forced to stream form; END is never interpreted
as moving every microbatch to the end. Failed actions roll back atomically.
A feasible trial enters refinement even if temporarily worse.

Coupled refinement considers P and O from the same current pair, not forced
alternation. It commits the round's best fully replayed candidate, including
when a budget expires mid-round. There are at most 48 rounds; partition-entry
quotas are 192/64 for N<=16/>16, followed by 96 order entries. Failures consume
entry quota but only actual replay calls consume E.

Only `coupled` refinement is supported. PartitionDescent remains part of
structured initialization; the outer search does not call it. Each feasible
outer trial has a 3072/1024 additional-evaluation cutoff for N<=16/>16.

## Timing And Certificates

The default requested budget is 60 seconds, one search thread, E=100,000,000.
Search starts before baseline construction. It reserves `min(1s,.05*T)`,
includes generation, scoring, replay and recording, and checks deadlines at
candidate and round boundaries. Final independent validation is always done.
Actual measured elapsed time, not the requested budget, is returned. Finalization
can exceed the reserve on very large inputs; the deadline is cooperative.

The optional affine structural bound is **not implemented**. The applicable
certificate is the existing global minimum-work capacity bound, strengthened
by the minimum-work single-microbatch chain plus fixed communication. For
partition-dependent affine costs this uses minimum work over all partitions,
not the current solution's work. Any bound above a replayed feasible objective
raises an error. `OPTIMAL` is returned only when the bound is reached; otherwise
the result is `FEASIBLE`. These are cost-model statements, not GPU optima.

`--heuristic-fixed-evaluations true` is a dedicated deterministic **test** mode:
all wall-clock truncation is disabled, while evaluation quotas, candidate
ordering and RNG calls remain unchanged. It cannot share a CP-SAT wall deadline.
Use real wall budgets for comparisons.

## Implementation

- `include/slackpipe/heuristic_hint.h`: options and incumbent API.
- `src/heuristic_hint.cc`: replay, total float, moves, structured initialization
  and single-parent search. `src/heuristic_internal.h` exposes primitives to tests.
- Full-partition duration cache: bounded to approximately 4 MiB of payload
  (container overhead is additional); it is not an archive of selectable plans.
- DAG/topological order is reused when O is unchanged; changed orders recompile.
- Different-phase streams use linear rebuilds and linear permutation/FIFO checks.
- Candidate/graph containers reserve capacity; completion sums use checked ticks.

## Build And Run

From the monorepo root, using the existing development container:

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev \
  cmake -S slackpipe -B slackpipe/build/no-or -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DSLACKPIPE_ENABLE_ORTOOLS=OFF
docker exec -w /workspace/Megatron-LM slackpipe-dev \
  cmake --build slackpipe/build/no-or -j4
docker exec -w /workspace/Megatron-LM slackpipe-dev \
  ctest --test-dir slackpipe/build/no-or --output-on-failure
```

Standalone modeled search, validation and Megatron plan export:

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev \
  slackpipe/build/no-or/slackpipe_cli --algorithm heuristic \
  --B 8 --N 8 --J 4 --L 64 --ratio-num 2 --ratio-den 1 --communication 0 \
  --random-seed 2 --heuristic-time-limit-seconds 60 \
  --heuristic-refinement coupled --heuristic-trace /tmp/heuristic.trace.json \
  --output-prefix /tmp/heuristic --emit-plan /tmp/heuristic.plan.json
docker exec -w /workspace/Megatron-LM slackpipe-dev \
  slackpipe/build/no-or/slackpipe_cli validate-result --input /tmp/heuristic.json
```

`--cost-profile PATH` uses the existing v1/v2 loader and exporter. Standalone
starts from uniform BFS; explicit starting splits, activation-cap optimization,
pressure pruning, noncyclic placement and multiple search threads are rejected.
Split-mode restrictions use uniform as their reference; worker-load constraints
remain active. A forbidden uniform baseline is an explicit failure, not a
fabricated hint.

With an OR-enabled binary, select `--incumbent-method heuristic` for direct
joint search, or `--bfs-method heuristic` for SlackPipe. Existing preparation
phase caps and the global deadline limit the heuristic; CP-SAT gets only the
remaining budget. Final activation/memory and other feasibility screening is
preserved. The native incumbent path supplies partition, orders and replayed
start/end times to the existing model hint encoder and horizon calculation.

## Diagnostics

Standalone CLI outputs:

| File | Content |
|---|---|
| `PREFIX.json` | Native P/O, complete timings, costs, strict improvement trace, elapsed time and status |
| `PREFIX.csv`, `PREFIX.orders.txt`, `PREFIX.svg` | Operation timings, worker orders and schedule visualization |
| `--heuristic-trace PATH` | Initialization plan/time/evaluations; outer before/trial/child values; acceptance reasons; action counts and initialization/refinement costs |
| `--emit-plan PATH` | Validated Megatron-consumable plan |

Native result JSON retains integer duration arrays so `validate-result` can
replay uniform, affine and prefix-profile results without substituting abstract
ratio costs. This is evaluator input metadata, not a new calibration schema.

Search events distinguish strict makespan improvement from secondary-only
acceptance. `accepted_*_actions` count committed local steps, even in a trial
eventually rejected by the outer comparator; they are not independent globally
accepted parents. Debug/final evaluator checks are counted separately from E.
Native legacy utilization fields are not used for profile comparisons because
some still use the abstract forward/backward ratio.

## Validation Scope

Tests directly cover DAG cycles, inventory/placement/FIFO rejection, prefix
cost transfers, signed proposal scores, insertion/stream rollback, critical
intervals and rightward ranking, total float, every tie-break, overflow,
large-B stream rebuilding, large-N initialization, certificates and final replay.
The CLI regression compares entire deterministic trajectories (excluding
timestamps), proves worse trials reach refinement, checks zero outer PD in
coupled mode, and checks the refinement allowance.

Run the OR-enabled suite with the existing runtime image and
`SLACKPIPE_OR_TEST_FILTER=.*`. Production Megatron plan/profile parser checks and
a solver-generated PP2 numerical smoke are separate evidence levels. GPU
runtime unavailability must be reported, not treated as a passed numerical test.
