# SlackPipe

SlackPipe is a C++ optimizer, evaluator, and plan exporter for fixed-shape
virtualized pipeline micro-batches. This directory is part of the Megatron
monorepo, not a separate Git checkout. See the [root README](../README.md) for
calibration, runtime integration, Docker commands, and supported configurations.
See [monorepo development](docs/monorepo.md) for the source-import scope.

## Current Scheduling Model

Operations are indexed by micro-batch `b` and operation position `n`, where
`n in [0, 2N)`:

```text
0, ..., N-1       = F_0, ..., F_{N-1}
N, ..., 2N-1      = B_{N-1}, ..., B_0
```

The logical stage and worker for operation position `n` are:

```text
s(n)     = n if n < N, otherwise 2N - 1 - n
omega(n) = worker_of_stage(s(n))
```

FIFO is a canonicalization rule over exchangeable micro-batches. It is applied
per operation position:

```text
(b - 1, n) -> (b, n), for b > 0
```

The worker-local `N` predecessor is a selected serialization predecessor from a
different operation position on the same physical worker. The deterministic
evaluator starts each operation at:

```text
StartTime[b,n] = max(R_data(b,n), R_B(b,n), R_N(b,n))
```

`pred_N` candidate generation exists for explicit predecessor evaluation and
tests. The production joint CP-SAT model currently uses unrestricted per-worker
`NoOverlap`; it does not enforce a predecessor-candidate window.

## Build And Test

```bash
cd slackpipe
cmake -S . -B build/no-or -G Ninja -DCMAKE_BUILD_TYPE=Release -DSLACKPIPE_ENABLE_ORTOOLS=OFF
cmake --build build/no-or --target slackpipe_core_tests slackpipe_cli slackpipe_benchmark -j2
ctest --test-dir build/no-or --output-on-failure
```

`SLACKPIPE_ENABLE_ORTOOLS=ON` requires a discoverable OR-Tools C++ package and
fails at configure time when it is missing. CMake probes the discovered
OR-Tools headers for cumulative-resource support and variable cumulative
demands. Use `slackpipe_cli build-info` to record git/build/schema provenance,
whether OR-Tools was compiled in, and `activation_cap_solver_support`.

For the OR-enabled evaluation smoke suite, use:

```bash
ORTOOLS_PREFIX=/opt/or-tools scripts/validate_ortools_evaluation.sh
```

The script configures `SLACKPIPE_ENABLE_ORTOOLS=ON`, passes `ortools_DIR`, and
passes bundled `absl_DIR`/`Protobuf_DIR` when those package configs exist. It
runs only OR-gated solver smoke tests plus tiny canonical CLI runs; it does not
launch the production benchmark sweep. `activation_cap_solver_support` is
`fixed_demands_only` when fixed-partition/count/explicit activation caps can be
enforced in CP-SAT, and `variable_demands` when optimized linear-in-layers
demands can also be enforced without freezing the incumbent partition.

The core C++ tests cover operation-position indexing, per-position FIFO,
worker-local `R_N` predecessor validation, pruned `pred_N` candidate helper
scope, and canonical evaluation-result schema serialization.

## Coupled Heuristic Early Hints

`--algorithm heuristic` runs the independent Heuristic Algorithm without
OR-Tools. It uses structured initialization followed by single-parent coupled
partition/order refinement. Only `coupled` refinement is supported.
Select it explicitly with `--incumbent-method heuristic` for joint CP-SAT
or `--bfs-method heuristic` for SlackPipe; existing defaults are unchanged.

See [Heuristic Algorithm](docs/heuristic.md) for search semantics, budget
accounting, diagnostics, tests and executable Docker examples. A returned
`FEASIBLE` schedule is not an optimality proof or a GPU performance measurement.

## Megatron Plan Export

`--emit-plan PATH` exports the evaluator-validated native worker orders as
`slackpipe.plan.v1`, or `slackpipe.plan.v2` for heterogeneous range-cost input.
`--cost-profile PATH` accepts `slackpipe.cost_profile.v1` and `.v2`; measured
costs, exact contiguous global-layer ranges, and model/profile hashes accompany
the corresponding plan. Existing canonical result JSON/CSV/orders/SVG outputs
are independent of plan export. No OR-Tools dependency is needed for the
deterministic `uniform-breadth-first` exporter tests.

The authoritative runtime contracts live in
[`../megatron/core/pipeline_parallel/slackpipe/`](../megatron/core/pipeline_parallel/slackpipe/).

## Search Statistics Diagnostics

Search statistics are opt-in diagnostic metadata for reproducibility. Normal CLI
runs do not print them. Enable them with `--emit-search-stats true` and/or write
full JSON with `--search-stats-json path.json`.

The JSON separates several concepts:

- `theoretical` is the mathematical size of a space when known. For positive
  stage layer partitions with `min_layers = 1`, this is `C(L - 1, N - 1)`.
- `enumerated` says whether SlackPipe explicitly generated that category in an
  outer C++ loop. `visited`, `pruned`, and `kept` are numeric only when
  `enumerated` is true; otherwise they are `null`.
- CP-SAT modes such as `optimize-joint` encode layer partitions and worker-local
  operation ordering as solver variables and constraints. They do not enumerate
  those spaces in outer loops, so their `stage_level_partition` and
  `interleave_reordering` sections report `enumerated: false` and `visited:
  null`.
- `candidate_schedules` counts complete schedules extracted or generated by the
  program and replayed through the deterministic evaluator.
- `cp_sat` reports native OR-Tools search metadata such as status, objective,
  best bound, branches, conflicts, wall time, and deterministic time when a
  CP-SAT solver path is used.

Worker-level partition search is currently disabled in the production model:
stage-to-worker mapping is fixed by `StageOfOp(n,N)%J`, so that section reports
`enabled: false` unless a future algorithm explicitly searches worker placement.

## Activation Memory Diagnostics

Canonical results include independent retained-activation memory metrics by
default. The analyzer uses half-open lifetimes from forward completion to the
matching backward start, controlled activation-unit models, and optional caps.
Equal `B`, `N`, `J`, and `L` alone do not imply equal activation memory because
worker-local order changes how many forward activations are retained.

When an explicit or uniform-baseline cap is requested with
`--activation-cap-enforcement solver`, OR-Tools CP-SAT paths add per-worker
cumulative constraints over retained-activation intervals. Post-hoc cap
checking remains available for diagnostics, but solver-backed equal-memory
comparisons require `activation_cap_enforced_in_solver = true` and
`activation_cap_constraints_added = true`. OR-Tools-disabled builds report
solver-backed capped methods as `UNAVAILABLE`.

See [`docs/evaluation_activation_memory.md`](docs/evaluation_activation_memory.md).

## Evaluation Result Schema

Standalone CLI JSON and benchmark JSONL outputs include a canonical
`canonical_result` object plus top-level `schema_version`, `canonical_method`,
and `actual_solver_path` fields. The schema records requested versus canonical
method names, fixed and optimized decision variables, active restrictions,
fallback behavior, derived worker predecessor edges, and build/git provenance.

See [`docs/evaluation_result_schema.md`](docs/evaluation_result_schema.md).
See [`docs/evaluation_methods.md`](docs/evaluation_methods.md) for the
canonical evaluation method contracts.
See [`docs/evaluation_manifests.md`](docs/evaluation_manifests.md) for the
JSONL CAL manifest generator and resumable runner.
See [`docs/evaluation_analysis.md`](docs/evaluation_analysis.md) for the
deterministic analysis pipeline that turns manifest-backed results into CAL
figures, tables, preflight reports, and paper-ready summaries.
See [`EVALUATION.md`](EVALUATION.md) for the frozen top-level CAL evaluation
protocol.

## CAL Manifest Smoke

```bash
python3 scripts/generate_cal_manifest.py --profile smoke \
  --output /tmp/slackpipe_cal_smoke.jsonl \
  --binary build/ortools/slackpipe_cli

python3 scripts/run_cal_manifest.py \
  --manifest /tmp/slackpipe_cal_smoke.jsonl \
  --output-root /tmp/slackpipe_cal_smoke \
  --binary build/ortools/slackpipe_cli \
  --jobs 1

python3 scripts/analyze_cal_results.py \
  --manifest /tmp/slackpipe_cal_smoke.jsonl \
  --results-root /tmp/slackpipe_cal_smoke \
  --output-dir /tmp/slackpipe_cal_smoke_analysis \
  --mode both \
  --include-groups smoke \
  --strict
```

The smoke profile declares 12 rows before execution: six canonical methods,
uncapped plus equal-memory, on one tiny configuration. The main profile declares
2592 rows and is intentionally not run by default. The opt-in
`main_big_5min_1f1b` profile adds `uniform-interleaved-1f1b`, uses a 300-second
row budget with 8 solver threads by default, includes both uncapped and
equal-memory rows, and declares 2754 rows. The
`main_big_5min_1f1b_uncapped` profile uses the same methods and big run settings
but contains only uncapped rows, declaring 1377 rows. The analysis command writes
preflight CSV/JSON, normalized makespan CSV/JSON plus PDF/PNG figures when
matplotlib is available, the compact CAL summary table, and Markdown/JSON
provenance summaries.

## Canonical Evaluation Commands

```bash
slackpipe_cli describe-method alternating-partition-schedule
slackpipe_cli build-info

slackpipe_cli --algorithm uniform-breadth-first --B 8 --N 8 --J 4 --L 64 --ratio-num 2 --ratio-den 1 --output-prefix out/uniform
slackpipe_cli --algorithm uniform-interleaved-1f1b --B 8 --N 8 --J 4 --L 64 --ratio-num 2 --ratio-den 1 --output-prefix out/1f1b
slackpipe_cli --algorithm partition-only-fixed-order --fixed-order-partition-backend cpsat --B 8 --N 8 --J 4 --L 64 --ratio-num 2 --ratio-den 1 --require-optimal false --output-prefix out/partition
slackpipe_cli --algorithm schedule-only-uniform --B 8 --N 8 --J 4 --L 64 --ratio-num 2 --ratio-den 1 --require-optimal false --output-prefix out/schedule
slackpipe_cli --algorithm sequential-partition-then-schedule --B 8 --N 8 --J 4 --L 64 --ratio-num 2 --ratio-den 1 --require-optimal false --output-prefix out/sequential
slackpipe_cli --algorithm alternating-partition-schedule --alternating-max-rounds 4 --B 8 --N 8 --J 4 --L 64 --ratio-num 2 --ratio-den 1 --require-optimal false --output-prefix out/alternating
slackpipe_cli --algorithm joint-unrestricted-no-overlap --B 8 --N 8 --J 4 --L 64 --ratio-num 2 --ratio-den 1 --require-optimal false --output-prefix out/joint
```

The original fixed baseline is `uniform-breadth-first`, the repository's
breadth-first wavefront worker order. `uniform-interleaved-1f1b` is a separate
deterministic fixed-order baseline and does not rename or replace
`uniform-breadth-first`.

## SlackPipe Split Modes

`slackpipe --split-mode MODE` uses the optimized-BFS partition as its reference
split. If optimized BFS cannot produce a replay-valid schedule, SlackPipe falls
back to the uniform split. The reference split is also used as the CP-SAT
incumbent and makespan horizon, so restricted runs are bounded by a partition
that satisfies the active restriction.

The split mode restricts only the partition search. Worker-local operation order
remains a CP-SAT optimization variable in every mode, including `fixed`.

| Mode | Enforced partition constraint |
| --- | --- |
| `fixed` | Every stage size must equal the reference stage size. |
| `local` | Stage-level movement is bounded by `sum_s abs(t_s - ref_s) <= 2 * move_budget`; `per_stage_delta`, when set, also requires each stage deviation to be at most that value. |
| `global` | No mode-specific partition restriction beyond validity, minimum stage size, and total layer sum. |
| `worker-fixed` | Each worker's aggregate layer load must equal its reference aggregate load. |
| `worker-local` | Worker-level movement is bounded by `sum_w abs(load_w - ref_load_w) <= 2 * worker_move_budget`; `per_worker_delta`, when set, also requires each worker aggregate deviation to be at most that value. |

Budgets are half-L1 movement budgets because all valid splits have the same
total layer count. For example, moving one layer from one stage to another
changes the stage-level L1 distance by two. `OPTIMAL` for a restricted run means
CP-SAT proved optimality within that restricted feasible set, not necessarily in
the unrestricted global partition space.

Mode-inapplicable explicit budget options are rejected instead of ignored. For
example, `fixed` does not accept movement budgets, `local` accepts only
stage-local budgets, and `worker-local` accepts only worker-aggregate budgets.
The benchmark runner's virtualized default comparison labels map directly to
these modes: `worker-fixed`, `worker-local-1`, `worker-local-2`,
`worker-local-4`, and `global`.
