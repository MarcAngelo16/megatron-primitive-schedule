> Historical scope: this document records pre-five-method code/results. Commands and method IDs below are not current entry points. Use the [current workflow](../../docs/slackpipe_five_methods.md); historical results keep their recorded commit and labels.

# Activation Memory Evaluation

SlackPipe now reports retained activation memory for every independently
validated schedule. The analyzer is separate from CP-SAT: it consumes the
validated instance, selected partition, and replayed operation timings.

## Lifetime Semantics

For each micro-batch `b` and logical stage `s`, the retained activation lifetime
is the half-open interval:

```text
[forward_completion(b,s), backward_start(b,s))
```

An activation is not live during its forward operation. It is released when the
matching backward operation begins. Zero-length lifetimes consume no retained
activation capacity. At equal timestamps, release events are processed before
acquire events, with equal-type events ordered by worker, stage, then
micro-batch for deterministic diagnostics.

Each lifetime records a stable identity: micro-batch, stage, worker, forward
operation id, and backward operation id. The analyzer fails schedules with
missing operation records, backward-before-forward lifetime intervals,
unexpected worker mismatches, or negative activation demand.

## Controlled Models

The default model is `linear_in_stage_layers` with
`activation_units_per_layer = 1`.

| Model | Demand per live activation |
| --- | --- |
| `count` | `1` |
| `linear_in_stage_layers` | `activation_units_per_layer * selected_partition[s]` |
| `explicit_stage_units` | supplied N-vector value for stage `s` |

These are controlled activation units, not physical GPU bytes. Byte-valued
fields are `null` unless `activation_bytes_per_unit` is supplied; then bytes are
reported as `activation_units * activation_bytes_per_unit`.

## Metrics

Per-worker metrics include peak live activation count, peak activation units,
optional peak bytes, earliest unit-peak timestamp, average live count, average
units, unit-time area, and cap satisfaction. Global metrics include maximum
worker peak, global simultaneous activation usage, earliest global unit-peak
timestamp, and total activation unit-time area.

The analyzer also derives ratios to the deterministic uniform-breadth-first
baseline under the same instance and activation model. Dividing by a zero
baseline peak yields `null` plus a warning.

## Caps

`activation_cap_mode` supports:

- `none`: metrics only; cap satisfaction is `null`.
- `explicit`: a scalar cap expanded to all workers, or a W-length vector.
- `uniform_baseline`: per-worker caps derived from the deterministic
  uniform-breadth-first baseline.

Uniform-baseline derivation uses no CP-SAT. The result records the baseline
partition, method contract hash, deterministic derivation hash, run id, and
derivation runtime.

Manifest-based CAL runs materialize the uniform-baseline cap vector once per
configuration before executing method rows. The runner passes that frozen vector
with `--activation-cap-mode uniform-baseline`, `--activation-cap-units`, and
`--activation-cap-derivation-hash`; the CLI records it as a uniform-baseline
cap, not an explicit-cap row. This prevents per-row cap drift in equal-memory
evaluation.

If a cap is supplied without solver enforcement, the schedule remains
schedule-feasible and `activation_cap_satisfied = false` marks it as
memory-infeasible for diagnostics. These rows use
`activation_cap_enforcement_mode = "posthoc_only"` and are excluded from
equal-memory makespan aggregation by the CAL analyzer documented in
[`evaluation_analysis.md`](evaluation_analysis.md).

## Solver-Side Enforcement

When `--activation-cap-enforcement solver` or `--enforce-activation-cap` is
used with an explicit or uniform-baseline cap, solver-backed methods add
retained-activation constraints to the CP-SAT model. For every micro-batch
`b` and stage `s`, the model creates an interval:

```text
start = forward_end(b,s)
end   = backward_start(b,s)
size  = end - start, with size >= 0
```

These intervals represent only retained activations. They exclude forward and
backward execution, parameters, gradients, optimizer state, communication
buffers, and temporary workspaces.

The solver adds one cumulative resource per physical worker with capacity from
the resolved `activation_cap_units_per_worker` vector serialized in the
canonical result. Demands are exact integers:

| Model | Solver demand |
| --- | --- |
| `count` | fixed `1` |
| `explicit_stage_units` | fixed stage value |
| `linear_in_stage_layers`, fixed partition | fixed `activation_units_per_layer * selected_partition[s]` |
| `linear_in_stage_layers`, optimized partition | variable `activation_units_per_layer * t_s` |

Variable linear demands require an OR-Tools C++ API that accepts
`CumulativeConstraint::AddDemand(IntervalVar, LinearExpr)`. CMake compiles a
small capability probe against the discovered headers and exposes the result in
`slackpipe_cli build-info`:

- `cumulative_constraint_supported`
- `variable_cumulative_demand_supported`
- `activation_cap_solver_support`: `none`, `fixed_demands_only`, or
  `variable_demands`

An OR-Tools-disabled build reports solver-side cap enforcement as
`UNAVAILABLE`; it does not run an uncapped optimization and relabel post-hoc
checking as solver enforcement. A fixed-demand-only OR-Tools build supports
`count`, `explicit_stage_units`, and `linear_in_stage_layers` only when the
partition is fixed. Optimized-partition linear demand requires
`variable_demands`.

Solver-enforced successful results must still pass the independent analyzer.
If CP-SAT proves no cap-feasible solution exists, the canonical status is
`INFEASIBLE`, not `INVALID_RESULT`. If a solver-enforced solution is returned
but independent analysis finds a cap violation, the row is marked
`INVALID_RESULT` and records `activation_model_validation_agreement = false`.

## CLI

```bash
slackpipe_cli --algorithm uniform-breadth-first \
  --B 8 --N 8 --J 4 --L 64 \
  --activation-model linear-in-stage-layers \
  --activation-cap-mode none \
  --output-prefix out/uniform

slackpipe_cli validate-result --input out/uniform.json \
  --activation-summary \
  --activation-model explicit-stage-units \
  --activation-stage-units 4,4,8,8,4,4,8,8

slackpipe_cli --algorithm schedule-only-uniform \
  --B 8 --N 8 --J 4 --L 64 \
  --activation-cap-mode uniform-baseline \
  --activation-cap-enforcement solver \
  --output-prefix out/schedule-capped
```

For compact inspection across JSON or JSONL outputs:

```bash
scripts/activation_diagnostics.py results.jsonl
```

## Limitations

The analyzer does not recompute tensors and does not model allocator
fragmentation, parameters, optimizer state, gradients, workspaces, or
communication buffers. Equal B, N, W, and L do not imply equal activation
memory because worker-local order changes retained activation lifetimes.
Uncapped activation ratios in the CAL summary are diagnostic; equal-memory
claims use only cap-satisfied rows with solver-side enforcement for
solver-backed methods.
