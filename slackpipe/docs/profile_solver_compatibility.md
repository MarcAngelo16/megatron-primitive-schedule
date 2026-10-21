# FIFO Safety and Profile-Aware Partition Solvers

> Historical record, retained without rewriting results. The BFS, fixed-order
> and hybrid solvers described below have been retired. Current profile-aware
> joint optimization is `cpsat-no-hint`; it has no external incumbent or fallback.
> See [the current guide](../../docs/slackpipe_five_methods.md).

Hybrid's order neighborhood is unchanged. With FIFO enabled, the common
evaluator rejects decreasing microbatch IDs at each operation position before
extracting worker predecessors. Same-position edges may only be omitted when
they agree with FIFO. Both adjacent and nonadjacent reversals are rejected;
explicit FIFO-disabled evaluation retains its previous semantics. Hybrid also
independently validates the final result before returning it.

The BFS CP-SAT and fixed-worker-order partition CP-SAT backends support:

- abstract forward/backward ratios;
- affine profile slopes and per-stage biases (v1);
- heterogeneous prefix costs with first/middle/last role biases (v2).

For v2, partition cuts are decision variables. Element constraints look up
forward/backward prefix values at those cuts. Every operation uses
`prefix[end] - prefix[begin] + role_bias`; no uniform-cost approximation is made.
These expressions are shared across microbatches. The conservative horizon
includes compute and communication. This is a compatibility/correctness fix,
not a new search algorithm or a change to incumbent/fallback policy.

For example, in the existing OR-Tools container, from the monorepo root:

```bash
./slackpipe/build/fifo-profile-or/slackpipe_cli \
  --algorithm partition-only-fixed-order --fixed-order-partition-backend cpsat \
  --B 4 --N 4 --J 2 --L 8 --communication 0 \
  --time-limit-seconds 5 --num-workers 1 --require-optimal false \
  --cost-profile slackpipe/tests/fixtures/octopipe_heterogeneous.cost_profile.json \
  --output-prefix /tmp/profile-partition \
  --emit-plan /tmp/profile-partition.plan.json
```

The fixture is synthetic, not a measured profile. For the legacy BFS optimizer,
use `--algorithm optimize-bfs --method cpsat` with the same cost-profile option.
The standard OR build procedure is
`slackpipe/scripts/validate_ortools_evaluation.sh`; `BUILD_DIR` selects the path.

`OPTIMAL` only certifies the specified fixed-order partition problem. At short
deadlines BFS CP-SAT can return `UNKNOWN` without a solution. The fixed-order
backend may instead return a validated incumbent with `fallback_used=true` and
`solver_status_raw=UNKNOWN`; this is not an optimality proof. The BFS time limit
covers solving, while fixed-order optimization uses its existing global phase
deadline. No silent change to either budget definition is made here.

Regression tests in `tests/profile_solver_tests.cc` compare both CP-SAT
backends against exhaustive small-instance optima, check exact operation
durations, asymmetric forward/backward profiles and role biases, single-stage
behavior, large communication delays, and timeout/fallback handling. The common
evaluator tests explicitly retain FIFO-off behavior. Existing joint-incumbent
tests cover acceptance of the repaired Hybrid hint.

Known separate limitation: the legacy `TotalUsefulWork`/utilization summary
still uses abstract ratios. For real-profile result analysis use the sum of
actual operation durations, not that legacy summary field. This patch does not
change workload reporting, cost-profile schemas, or Megatron runtime behavior.
