# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""SlackPipe cost-profile aggregation and shared-slope fitting."""

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy

from megatron.core.pipeline_parallel.slackpipe.manifest import canonical_json
from megatron.core.pipeline_parallel.slackpipe.stage_residual import (
    FitStructureError,
    ResidualPolicy,
    fit_with_residuals,
    summarize_operations,
)

SLACKPIPE_COST_PROFILE_SCHEMA_VERSION = "slackpipe.cost_profile.v1"
SLACKPIPE_COST_PROFILE_SCHEMA_VERSION_V2 = "slackpipe.cost_profile.v2"

_STAGE_ROLES = ("first", "middle", "last")


def percentile(values: Sequence[float], percentile_value: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of an empty list")
    if not 0.0 <= percentile_value <= 100.0:
        raise ValueError("percentile must be in [0, 100]")
    sorted_values = sorted(float(value) for value in values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percentile_value / 100.0
    lower = int(math.floor(position))
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def fit_shared_slope_stage_bias(
    layers: Sequence[int],
    observed_costs: Sequence[float],
    estimator: str = "min",
    percentile_value: float = 20.0,
) -> tuple[float, list[float]]:
    """Fit observed_cost[s] ~= a * layers[s] + bias[s]."""

    if len(layers) != len(observed_costs):
        raise ValueError("layers and observed_costs must have the same length")
    if any(int(layer_count) <= 0 for layer_count in layers):
        raise ValueError("layer counts must be positive")
    ratios = [float(cost) / int(layer_count) for layer_count, cost in zip(layers, observed_costs)]
    if estimator == "min":
        slope = min(ratios)
    elif estimator == "median":
        slope = statistics.median(ratios)
    elif estimator == "percentile":
        slope = percentile(ratios, percentile_value)
    else:
        raise ValueError(f"unsupported shared slope estimator {estimator!r}")
    bias = [
        max(0.0, float(cost) - slope * int(layer_count))
        for layer_count, cost in zip(layers, observed_costs)
    ]
    return slope, bias


def aggregate_stage_costs(
    events: Iterable[Mapping[str, object]],
    *,
    layer_split: Sequence[int],
    num_microbatches: int,
    iteration_start: int,
    iteration_end: int,
) -> list[dict[str, object]]:
    """Validate raw inventory, then aggregate all operation timings without filtering."""

    if type(num_microbatches) is not int or num_microbatches <= 0:
        raise ValueError("num_microbatches must be positive")
    if not layer_split or any(type(n) is not int or n < 1 for n in layer_split):
        raise ValueError("Layer split must contain positive integer counts")
    if type(iteration_start) is not int or type(iteration_end) is not int or iteration_start < 0:
        raise ValueError("Iteration bounds must be nonnegative integers")
    iterations = list(range(iteration_start, iteration_end + 1))
    if not iterations:
        raise ValueError("iteration range must be non-empty")
    num_stages = len(layer_split)
    samples = defaultdict(list)
    provenance = {}

    seen = set()
    for event in events:
        phase = event.get("phase")
        if phase not in {"forward_compute", "backward_compute"}:
            continue
        stage = event["logical_stage"]
        iteration = event["iteration"]
        if type(stage) is not int or type(iteration) is not int:
            raise ValueError("Calibration stage and iteration must be integers")
        if not 0 <= stage < num_stages:
            raise ValueError(f"logical stage {stage} is out of range")
        if iteration not in iterations:
            continue
        microbatch = event.get("microbatch")
        if type(microbatch) is not int or not 0 <= microbatch < num_microbatches:
            raise ValueError("Calibration operation requires a valid microbatch ID")
        key = (iteration, stage, phase, microbatch)
        if key in seen:
            raise ValueError(f"Duplicate calibration operation: {key}")
        seen.add(key)
        identity = {
            k: event[k]
            for k in (
                "calibration_group_id",
                "attempt",
                "worker",
                "stage_layer_range",
                "stage_role",
                "measurement_context",
                "raw_data_path",
            )
            if k in event
        }
        if stage in provenance and provenance[stage] != identity:
            raise ValueError("Cannot aggregate different calibration groups or contexts together")
        provenance[stage] = identity
        sample = dict(event)
        sample["sample_id"] = (
            f"{event.get('calibration_group_id', 'group0')}/"
            f"i{iteration}/s{stage}/{phase}/b{microbatch}"
        )
        samples[(stage, phase.removesuffix("_compute"))].append(sample)

    rows = []
    for stage, layer_count in enumerate(layer_split):
        phase_diagnostics = {}
        for phase in ("forward", "backward"):
            raw = samples[(stage, phase)]
            raw_by_iteration = defaultdict(list)
            for event in raw:
                raw_by_iteration[int(event["iteration"])].append(float(event["elapsed_ms"]))
            for iteration in iterations:
                values = raw_by_iteration[iteration]
                if len(values) != num_microbatches:
                    raise ValueError(
                        f"Expected {num_microbatches} calibration events for iteration {iteration}, stage {stage}, phase {phase}; got {len(values)}"
                    )
            phase_diagnostics[phase] = summarize_operations(raw, iterations, num_microbatches)
        forward_totals = phase_diagnostics["forward"]["samples_ms"]
        backward_totals = phase_diagnostics["backward"]["samples_ms"]
        median_forward = statistics.median(forward_totals)
        median_backward = statistics.median(backward_totals)
        forward_op = median_forward / num_microbatches
        backward_op = median_backward / num_microbatches

        rows.append(
            {
                "stage": stage,
                "layer_count": int(layer_count),
                "median_forward_total_ms": median_forward,
                "median_backward_total_ms": median_backward,
                "forward_ms_per_op": forward_op,
                "backward_ms_per_op": backward_op,
                "backward_forward_ratio": backward_op / forward_op if forward_op else None,
                "forward_diagnostics": phase_diagnostics["forward"],
                "backward_diagnostics": phase_diagnostics["backward"],
                **provenance.get(stage, {}),
            }
        )
    return rows


def build_cost_profile(
    *,
    events: Iterable[Mapping[str, object]],
    model_config: Mapping[str, object],
    parallel_config: Mapping[str, object],
    layer_split: Sequence[int],
    num_microbatches: int,
    iteration_start: int,
    iteration_end: int,
    estimator: str = "min",
    percentile_value: float = 20.0,
    residual_policy: ResidualPolicy = ResidualPolicy(),
) -> dict[str, object]:
    observed = aggregate_stage_costs(
        events,
        layer_split=layer_split,
        num_microbatches=num_microbatches,
        iteration_start=iteration_start,
        iteration_end=iteration_end,
    )

    def fit(rows, phase):
        layers = [row["layer_count"] for row in rows]
        slope, bias = fit_shared_slope_stage_bias(
            layers,
            [row[f"{phase}_ms_per_op"] for row in rows],
            estimator=estimator,
            percentile_value=percentile_value,
        )
        return dict(
            slope=slope,
            bias=bias,
            estimator=estimator,
            predicted=[slope * n + b for n, b in zip(layers, bias)],
        )

    models, final_rows, filtering = fit_with_residuals(observed, fit, residual_policy)
    a_fwd, bias_fwd = models["forward"]["slope"], models["forward"]["bias"]
    a_bwd, bias_bwd = models["backward"]["slope"], models["backward"]["bias"]
    return {
        "schema_version": SLACKPIPE_COST_PROFILE_SCHEMA_VERSION,
        "model_config": dict(model_config),
        "parallel_config": dict(parallel_config),
        "microbatch_config": {
            "num_microbatches": num_microbatches,
            "calibration_iterations": iteration_end - iteration_start + 1,
            "iteration_start": iteration_start,
            "iteration_end": iteration_end,
        },
        "calibration_partition": list(int(value) for value in layer_split),
        "estimator": estimator,
        "percentile": percentile_value if estimator == "percentile" else None,
        "a_fwd": a_fwd,
        "a_bwd": a_bwd,
        "measured_backward_forward_ratio": a_bwd / a_fwd if a_fwd else None,
        "bias_fwd": bias_fwd,
        "bias_bwd": bias_bwd,
        "profiling_backend": "stage",
        "profiling_policy": residual_policy.identity(),
        "measurement_parallel_config": dict(parallel_config),
        "target_parallel_config": dict(parallel_config),
        "raw_observed_stages": observed,
        "observed_stages": final_rows,
        "residual_filter": filtering,
        "measurement_definition": calibration_measurement_definition(),
        "units": "milliseconds",
    }


def calibration_measurement_definition() -> dict:
    return dict(
        source="ordinary Megatron synchronized stage-call wall time, not profiler operation envelopes",
        includes="CPU dispatch, compute, allocation/cache stalls inside the call and completion wait",
        excludes="explicit pipeline P2P calls outside stage compute and optimizer step",
        synchronization="device synchronization before/after each calibrated stage call; separate diagnostic run",
        estimator="median of nonempty iteration means; each mean uses its actual retained operation count",
        filtering="one initial-model relative-residual median/MAD mask per phase, then one refit; raw samples preserved",
        kernel_union_substitution=False,
    )


def stage_role(stage: int, num_stages: int) -> str:
    """Return the placement role used for stage fixed-cost bias."""

    if num_stages <= 0:
        raise ValueError("num_stages must be positive")
    if stage < 0 or stage >= num_stages:
        raise ValueError("stage out of range")
    if stage == 0:
        return "first"
    if stage == num_stages - 1:
        return "last"
    return "middle"


def _manifest_class_ids(model_manifest: Mapping[str, object]) -> list[str]:
    layers = model_manifest.get("layers")
    if not isinstance(layers, Sequence) or isinstance(layers, (str, bytes)):
        raise ValueError("model manifest must contain a layers list")
    class_ids = []
    seen = set()
    for layer in layers:
        if not isinstance(layer, Mapping):
            raise ValueError("model manifest layers must be objects")
        class_id = str(layer.get("config_class"))
        if class_id not in seen:
            seen.add(class_id)
            class_ids.append(class_id)
    return class_ids


def _layer_classes(model_manifest: Mapping[str, object]) -> list[str]:
    layers = model_manifest.get("layers")
    if not isinstance(layers, Sequence) or isinstance(layers, (str, bytes)):
        raise ValueError("model manifest must contain a layers list")
    layer_classes = []
    for expected_id, layer in enumerate(layers):
        if not isinstance(layer, Mapping):
            raise ValueError("model manifest layers must be objects")
        layer_id = int(layer.get("layer_id"))
        if layer_id != expected_id:
            raise ValueError("model manifest layer_id values must be contiguous from zero")
        layer_classes.append(str(layer.get("config_class")))
    return layer_classes


def _stage_range(row: Mapping[str, object]) -> tuple[int, int]:
    if "begin" in row and "end" in row:
        return int(row["begin"]), int(row["end"])
    if "stage_layer_range" in row:
        stage_range = row["stage_layer_range"]
        if isinstance(stage_range, Mapping):
            return int(stage_range["begin"]), int(stage_range["end"])
        if isinstance(stage_range, Sequence) and not isinstance(stage_range, (str, bytes)):
            if len(stage_range) != 2:
                raise ValueError("stage_layer_range must have two entries")
            return int(stage_range[0]), int(stage_range[1])
    raise ValueError("observed stage row must contain begin/end or stage_layer_range")


def _counts_for_range(
    layer_classes: Sequence[str], class_to_index: Mapping[str, int], begin: int, end: int
) -> list[int]:
    if begin < 0 or end <= begin or end > len(layer_classes):
        raise ValueError(f"invalid stage layer range [{begin}, {end})")
    counts = [0] * len(class_to_index)
    for class_id in layer_classes[begin:end]:
        counts[class_to_index[class_id]] += 1
    return counts


def fit_heterogeneous_class_costs(
    observed_stage_rows: Sequence[Mapping[str, object]],
    *,
    model_manifest: Mapping[str, object],
    value_field: str,
) -> dict[str, object]:
    """Fit per-class layer costs plus stage-role bias from stage-level samples."""

    class_ids = _manifest_class_ids(model_manifest)
    if not class_ids:
        raise ValueError("model manifest must contain at least one class")
    class_to_index = {class_id: index for index, class_id in enumerate(class_ids)}
    layer_classes = _layer_classes(model_manifest)

    rows = []
    values = []
    for observed in observed_stage_rows:
        begin, end = _stage_range(observed)
        counts = _counts_for_range(layer_classes, class_to_index, begin, end)
        role = str(observed.get("stage_role", observed.get("role", "middle")))
        if role not in _STAGE_ROLES:
            raise ValueError(f"unsupported stage role {role!r}")
        feature_row = [float(value) for value in counts]
        feature_row.extend(1.0 if role == candidate else 0.0 for candidate in _STAGE_ROLES)
        rows.append(feature_row)
        values.append(float(observed[value_field]))

    if len(rows) < len(class_ids):
        raise FitStructureError("not enough observed stage rows to fit class costs")

    design = numpy.asarray(rows, dtype=numpy.float64)
    target = numpy.asarray(values, dtype=numpy.float64)
    if not numpy.all(numpy.isfinite(target)) or numpy.any(target < 0):
        raise ValueError("heterogeneous calibration costs must be finite and nonnegative")
    solution, residuals, rank, singular_values = numpy.linalg.lstsq(design, target, rcond=None)
    if rank != design.shape[1]:
        raise FitStructureError(
            f"rank-deficient heterogeneous calibration ({rank}/{design.shape[1]}); add partitions"
        )
    if numpy.any(solution < -1.0e-12):
        from scipy.optimize import nnls

        solution, _ = nnls(design, target)

    solution = numpy.where(numpy.abs(solution) < 1.0e-12, 0.0, solution)
    if numpy.any(solution[: len(class_ids)] < -1.0e-9):
        raise ValueError("fitted class costs contain negative values")

    predicted = design @ solution
    residual_vector = target - predicted
    rmse = float(numpy.sqrt(numpy.mean(residual_vector * residual_vector)))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values.size and singular_values[-1] > 0.0
        else math.inf
    )
    class_costs = {
        class_id: max(0.0, float(solution[index])) for index, class_id in enumerate(class_ids)
    }
    role_bias = {
        role: max(0.0, float(solution[len(class_ids) + index]))
        for index, role in enumerate(_STAGE_ROLES)
    }
    return {
        "class_costs": class_costs,
        "stage_role_bias": role_bias,
        "rank": int(rank),
        "condition": condition,
        "rmse": rmse,
        "max_abs_error": float(numpy.max(numpy.abs(residual_vector))) if values else 0.0,
        "residuals": [float(value) for value in residual_vector],
        "relative_rmse": rmse / float(numpy.mean(target)) if numpy.mean(target) else 0.0,
        "predicted": predicted.tolist(),
        "method": "nonnegative_least_squares",
    }


def _prefix(values: Sequence[float]) -> list[float]:
    total = 0.0
    result = [0.0]
    for value in values:
        total += float(value)
        result.append(total)
    return result


def build_heterogeneous_cost_profile(
    *,
    model_manifest: Mapping[str, object],
    observed_stage_rows: Sequence[Mapping[str, object]],
    model_config: Mapping[str, object],
    parallel_config: Mapping[str, object],
    units: str = "milliseconds",
    residual_policy: ResidualPolicy = ResidualPolicy(),
) -> dict[str, object]:
    """Build slackpipe.cost_profile.v2 from stage-level heterogeneous samples."""

    if units != "milliseconds":
        raise ValueError("only millisecond input samples are currently supported")
    layer_classes = _layer_classes(model_manifest)
    observations = []
    for row in observed_stage_rows:
        begin, end = _stage_range(row)
        _counts_for_range(
            layer_classes, {c: i for i, c in enumerate(dict.fromkeys(layer_classes))}, begin, end
        )
        for phase in ("forward", "backward"):
            diagnostics = row.get(f"{phase}_diagnostics", {})
            if "raw_events" not in diagnostics:
                continue
            raw = diagnostics["raw_events"]
            iterations = diagnostics["global_iterations"]
            batches = diagnostics["num_microbatches"]
            if (
                type(batches) is not int
                or batches < 1
                or not iterations
                or any(type(i) is not int or i < 0 for i in iterations)
                or len(set(iterations)) != len(iterations)
            ):
                raise ValueError("Invalid raw observation inventory")
            actual = [(e["iteration"], e["microbatch"]) for e in raw]
            expected = {(i, b) for i in iterations for b in range(batches)}
            if len(actual) != len(expected) or set(actual) != expected:
                raise ValueError("Missing/duplicate raw observation operations")
            if any(
                type(e["iteration"]) is not int
                or type(e["microbatch"]) is not int
                or e.get("logical_stage", row["stage"]) != row["stage"]
                or e.get("phase", phase + "_compute") != phase + "_compute"
                for e in raw
            ):
                raise ValueError("Raw operation stage/phase/index differs from observation")
            if len({e["sample_id"] for e in raw}) != len(raw):
                raise ValueError("Duplicate raw operation sample identity")
            summary = summarize_operations(raw, iterations, batches)
            if not math.isclose(summary["center_ms"], row[f"{phase}_ms_per_op"], rel_tol=1e-12):
                raise ValueError("Stage summary does not match unfiltered raw operations")
        composition = layer_classes[begin:end]
        expected_counts = {c: composition.count(c) for c in set(composition)}
        if (
            ("layer_composition" in row and row["layer_composition"] != composition)
            or ("class_counts" in row and row["class_counts"] != expected_counts)
            or (
                "model_manifest_hash" in row
                and row["model_manifest_hash"] != model_manifest.get("manifest_hash")
            )
        ):
            raise ValueError("Observed layer composition differs from the model manifest")
        observations.append(
            dict(
                row,
                stage_layer_range=[begin, end],
                layer_composition=composition,
                model_manifest_hash=model_manifest.get("manifest_hash"),
            )
        )

    def fit(rows, phase):
        return fit_heterogeneous_class_costs(
            rows, model_manifest=model_manifest, value_field=f"{phase}_ms_per_op"
        )

    models, final_rows, filtering = fit_with_residuals(observations, fit, residual_policy)
    policy_identity = residual_policy.identity()
    if any(report["mask_passes"] == 0 for report in filtering["phases"].values()):
        policy_identity = dict(
            policy_version="stage-summary-fit-v1",
            input_observation_basis="provided summaries; prior sample filtering unknown",
            residual_filter="not applied without complete raw operations",
        )
    forward_fit, backward_fit = models["forward"], models["backward"]
    forward_ms = [forward_fit["class_costs"][class_id] for class_id in layer_classes]
    backward_ms = [backward_fit["class_costs"][class_id] for class_id in layer_classes]
    layer_costs_us = [
        {
            "layer_id": layer_id,
            "config_class": layer_classes[layer_id],
            "forward_us": forward_ms[layer_id] * 1000.0,
            "backward_us": backward_ms[layer_id] * 1000.0,
        }
        for layer_id in range(len(layer_classes))
    ]
    profile = {
        "schema_version": SLACKPIPE_COST_PROFILE_SCHEMA_VERSION_V2,
        "model_manifest_hash": model_manifest.get("manifest_hash"),
        "measurement_definition": {
            **calibration_measurement_definition(),
            "group_estimator": "every calibration group participates; no consensus or group selection",
        },
        "profiling_backend": "stage",
        "profiling_policy": policy_identity,
        "measurement_parallel_config": dict(parallel_config),
        "target_parallel_config": dict(parallel_config),
        "raw_observed_stages": observations,
        "observed_stages": final_rows,
        "residual_filter": filtering,
        "model_config": dict(model_config),
        "parallel_config": dict(parallel_config),
        "class_costs_us": {
            class_id: {
                "forward": forward_fit["class_costs"][class_id] * 1000.0,
                "backward": backward_fit["class_costs"][class_id] * 1000.0,
            }
            for class_id in _manifest_class_ids(model_manifest)
        },
        "layer_costs_us": layer_costs_us,
        "prefix_forward_us": [value * 1000.0 for value in _prefix(forward_ms)],
        "prefix_backward_us": [value * 1000.0 for value in _prefix(backward_ms)],
        "stage_role_bias_us": {
            role: {
                "forward": forward_fit["stage_role_bias"][role] * 1000.0,
                "backward": backward_fit["stage_role_bias"][role] * 1000.0,
            }
            for role in _STAGE_ROLES
        },
        "fit": {
            "forward": {
                "observation_basis": "all_group_iteration_mean_medians_after_one_residual_mask",
                "rank": forward_fit["rank"],
                "condition": forward_fit["condition"],
                "rmse": forward_fit["rmse"],
                "max_abs_error": forward_fit["max_abs_error"],
                "residuals": forward_fit["residuals"],
                "relative_rmse": forward_fit["relative_rmse"],
            },
            "backward": {
                "observation_basis": "all_group_iteration_mean_medians_after_one_residual_mask",
                "rank": backward_fit["rank"],
                "condition": backward_fit["condition"],
                "rmse": backward_fit["rmse"],
                "max_abs_error": backward_fit["max_abs_error"],
                "residuals": backward_fit["residuals"],
                "relative_rmse": backward_fit["relative_rmse"],
            },
        },
        "units": "microseconds",
    }
    profile["cost_profile_hash"] = profile_fingerprint(profile)
    return profile


def profile_fingerprint(profile: Mapping[str, object]) -> str:
    payload = dict(profile)
    payload.pop("cost_profile_hash", None)
    import hashlib

    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def write_cost_profile(path: str | Path, profile: Mapping[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.tmp")
    tmp.write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(target)
