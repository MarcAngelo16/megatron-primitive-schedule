# SlackPipe Optimizers and Plan Tools

Tracked C++20 sources in the Megatron research monorepo.
The public plan generators are `slack-heuristic`, `octopipe-tuning`, and
`cpsat-no-hint`. The other two comparison methods, `1f1b` and
`interleaved-1f1b`, execute in Megatron, not in an offline C++ approximation.

See [the current five-method guide](../docs/slackpipe_five_methods.md) for
build commands, API/CLI contracts, budgets, results, activation caps, tests and
real-system examples. [EVALUATION.md](EVALUATION.md) is the protocol summary.

## Source Structure

`instance`, `operation`, `schedule`, `types` hold problem data and shared
partition helpers. `dag_evaluator` and `result_validator` retain independent
validation paths. `cost_profile` loads affine/range costs; `plan_export`
serializes validated native operations atomically for Megatron.

`slack_heuristic`/`heuristic_internal.h` preserve the standalone coupled
total-float search, exposed as `SolveSlackHeuristic` with `SlackHeuristicOptions`.
`octopipe_tuner` preserves fixed-stage F/B phase selection
and neighborhoods; the formal CLI fixes cyclic placement. `joint_solver` and
`joint_solver_cpsat` contain only the unhinted joint model, no external incumbent
constructor or fallback. `SearchResult` replaces the BFS-specific shared type.
`instance.*` owns `UniformSplit`, `WorkerLayerTotals`, `CountValidSplitsCapped`.

`activation_analyzer`/`activation_cpsat`, `deadline`, `workload_bound`, timing,
`io`, `result_schema`, and `evaluation_method` support the retained methods.
Internal BFS, offline 1F1B and `octopipe_phase_diagnostic` remain diagnostic
tools, not formal native baseline implementations.

## Build

Inside the prepared container:

```bash
cmake -S slackpipe -B slackpipe/build/no-or -G Ninja -DSLACKPIPE_ENABLE_ORTOOLS=OFF
cmake --build slackpipe/build/no-or -j 4
ctest --test-dir slackpipe/build/no-or --output-on-failure
slackpipe/build/no-or/slackpipe_cli --help
slackpipe/build/no-or/slackpipe_cli --build-info
```

For OR-Tools use `scripts/validate_ortools_evaluation.sh` in the existing
`slackpipe-ortools-runtime:local` image with the whole repository mounted;
the guide includes the full command. Do not silently use an OR-disabled binary.
Only CP-SAT supports enforced activation caps.

## Historical Experiments

The previous BFS optimizer, hybrid-slack, Puzzle hint integration,
partition-only/schedule-only/alternating, pressure/balance restrictions, old
SlackPipe reference solver and benchmark/CAL campaign implementations are no
longer active modules. Their recorded outputs retain the original method names
and source commit. Reproduce them from that commit, not by relabelling current
results. No solver optimum certifies real-system optimality.
