# Slack-based Heuristic

The public method is `slack-heuristic`, displayed as `Slack-based Heuristic`.
The implementation is in `src/slack_heuristic.cc` and
`include/slackpipe/slack_heuristic.h`, exposed as `SolveSlackHeuristic` with
`SlackHeuristicOptions`. It is a standalone optimizer, not a CP-SAT hint provider.
CP-SAT never calls this method. There is no old SlackPipe reference combination
or Puzzle initializer.

## Preserved Search

A state is a complete contiguous partition and every worker-local F/B order.
Placement is fixed cyclic. Costs use the existing integer-tick Instance model,
including affine and heterogeneous prefix profiles. All changed layer ranges
are recomputed. Replay checks data dependencies, communication lag, phase FIFO,
worker serialization and the combined DAG; final independent validation remains.

The comparator is the exact integer lexicographic tuple:

```
(makespan, sum of backward completion times, partition, worker orders)
```

Secondary improvements at equal makespan remain permitted. They do not reset
outer staleness. This differs deliberately from OctoPipe's strict makespan
acceptance; the refactor does not make their search rules the same.

Initialization retains uniform BFS, structured block/diagonal/fill-drain seeds,
partition descent, optimized-seed refinement and worker-local restarts. The
single-parent search retains total float, candidate ranking, combined
perturbation, atomic rollback and coupled P/order refinement. Existing quotas,
RNG calls, tie-breaks and independent replay are unchanged. The internal
`pd_then_refine` control is retained only for regression, not as another public
method or experimental launcher.

The valid certificate is the global minimum-work capacity bound strengthened
by the minimum-work single-microbatch chain and fixed communication. It never
uses incumbent partition-dependent work as a global lower bound. Reaching the
certificate may prove modeled optimality, never GPU optimality.

## Budget and Invocation

`--time-limit` is the single public wall-clock budget (default 300 seconds),
passed unchanged to the heuristic. Search starts before baseline construction
and reserves `min(1s, .05*T)` for finalization. Deadlines are cooperative; the
actual measured elapsed time is recorded. There is no hidden 60-second limit.
Only one search worker is supported. Activation-cap enforcement, noncyclic
placement and additional partition restrictions are not supported.

Inside the existing `slackpipe-dev` container, from `/workspace/Megatron-LM`:

```bash
slackpipe/build/no-or/slackpipe_cli --algorithm slack-heuristic \
  --B 8 --N 8 --J 4 --L 64 --ratio-num 2 --ratio-den 1 --communication 0 \
  --seed 2 --time-limit 1 --num-workers 1 \
  --heuristic-trace true \
  --output-prefix /tmp/heuristic --emit-plan /tmp/heuristic.plan.json
slackpipe/build/no-or/slackpipe_cli --validate-result /tmp/heuristic.json
```

Add `--cost-profile PATH` for real v1/v2 costs. Use fresh output paths. See the
[five-method guide](../../docs/slackpipe_five_methods.md) for builds, the unified
real-system runner, result schema and hardware requirements.

`--heuristic-evaluations 40000 --heuristic-fixed-evaluations true` is a
deterministic regression mode: wall truncation is disabled but replay quotas and
search ordering are preserved. It is not a fair wall-clock benchmark mode.

## Regression Evidence

The pruning audit compares the old and new binary on B=4, N=4, W=2, L=12,
F=2, B=3, communication=2 and 40,000 evaluations. The final 129-tick makespan,
partition, every worker order, operation timings and all non-timing trace
events match exactly. The captured baseline binary and source snapshot are
retained in the ignored review directory; this is one representative
regression, not a proof for every input.

Unit tests retain FIFO/cycle rejection, range costs, total float, rollback,
tie-breaks, coupled refinement, large-B/N behavior, overflow, certificates and
final evaluator cross-checks. CLI tests cover the unified budget, worker
rejection, structured plans and independent validation. Historical comparison
campaigns and their hint-combination launchers were removed, not moved into an
active legacy tree. Their recorded results remain unchanged.
