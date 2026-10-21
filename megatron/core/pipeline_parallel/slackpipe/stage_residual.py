# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""One initial fit, one operation-residual mask, at most one refit per phase."""

import copy
import math
import statistics
from dataclasses import asdict, dataclass

POLICY_VERSION = "stage-affine-residual-v1"


class FitStructureError(ValueError):
    """The observations cannot construct the required cost model."""


@dataclass(frozen=True)
class ResidualPolicy:
    k: float = 6.0
    rho: float = 0.5

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in (self.k, self.rho)):
            raise ValueError("Residual k and rho must be finite and positive")

    def identity(self):
        return dict(policy_version=POLICY_VERSION, **asdict(self))


def add_residual_arguments(parser):
    parser.add_argument("--stage-residual-k", type=float, default=6.0)
    parser.add_argument("--stage-residual-rho", type=float, default=0.5)


def residual_policy_from_args(args):
    return ResidualPolicy(args.stage_residual_k, args.stage_residual_rho)


def summarize_operations(events, iterations, num_microbatches, *, excluded=()):
    """Median of nonempty iteration means, with no stability acceptance test."""
    excluded = set(excluded)
    by_iteration = {i: [] for i in iterations}
    for index, event in enumerate(events):
        value = event["elapsed_ms"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Operation timings must be numeric")
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Operation timings must be finite and positive")
        if index not in excluded:
            by_iteration[int(event["iteration"])].append(value)
    means = [statistics.fmean(by_iteration[i]) if by_iteration[i] else None for i in iterations]
    nonempty = [v for v in means if v is not None]
    if not nonempty:
        raise FitStructureError("required_observation_has_no_samples")
    center = statistics.median(nonempty)
    return dict(
        center_ms=center,
        global_iterations=list(iterations),
        iteration_means_ms=means,
        # Compatibility totals are B-normalized means, never partial sum / B.
        samples_ms=[v * num_microbatches if v is not None else None for v in means],
        samples_ms_definition="B times retained per-operation iteration mean, not a partial sum",
        accepted_counts_by_iteration=[len(by_iteration[i]) for i in iterations],
        raw_sample_count=len(events),
        accepted_sample_count=len(events) - len(excluded),
        samples_discarded=len(excluded),
        raw_events=copy.deepcopy(events),
        num_microbatches=num_microbatches,
    )


def fit_with_residuals(rows, fit_phase, policy=ResidualPolicy()):
    """fit_phase(rows, phase) returns a model with one positive prediction per row.

    Only explicit post-mask structural failures roll back a phase. Unexpected
    numerical/programming failures propagate; no retry or threshold tuning.
    """
    final_rows = copy.deepcopy(rows)
    models, reports = {}, {}
    for phase in ("forward", "backward"):
        initial = fit_phase(rows, phase)
        predictions = list(initial["predicted"])
        if len(predictions) != len(rows) or any(
            not math.isfinite(p) or p <= 0 for p in predictions
        ):
            raise FitStructureError("Initial model must predict positive finite operation times")
        records = []
        has_raw = all(
            all(
                k in r.get(f"{phase}_diagnostics", {})
                for k in ("raw_events", "global_iterations", "num_microbatches")
            )
            for r in rows
        )
        if has_raw:
            for row_index, (row, prediction) in enumerate(zip(rows, predictions)):
                for index, event in enumerate(row[f"{phase}_diagnostics"]["raw_events"]):
                    records.append(
                        dict(
                            row_index=row_index,
                            sample_index=index,
                            sample_id=event["sample_id"],
                            iteration=event["iteration"],
                            microbatch=event.get("microbatch"),
                            calibration_group_id=event.get("calibration_group_id"),
                            stage=row["stage"],
                            stage_layer_range=row.get("stage_layer_range"),
                            raw_data_path=event.get("raw_data_path"),
                            observed_ms=float(event["elapsed_ms"]),
                            prediction_ms=prediction,
                            relative_residual=(float(event["elapsed_ms"]) - prediction)
                            / prediction,
                        )
                    )
        residuals = [r["relative_residual"] for r in records]
        median = statistics.median(residuals) if residuals else None
        scale = (
            1.4826 * statistics.median(abs(q - median) for q in residuals) if residuals else None
        )
        candidate = [abs(q - median) > policy.k * scale and abs(q) > policy.rho for q in residuals]
        applied = list(candidate)
        final = initial
        status, reason = "applied", None
        if not has_raw:
            status, reason = "filter_not_applied", "raw_operation_samples_unavailable"
        elif any(candidate):
            trial = copy.deepcopy(rows)
            try:
                for row_index, row in enumerate(trial):
                    diagnostics = row[f"{phase}_diagnostics"]
                    excluded = [
                        record["sample_index"]
                        for record, remove in zip(records, candidate)
                        if remove and record["row_index"] == row_index
                    ]
                    updated = summarize_operations(
                        diagnostics["raw_events"],
                        diagnostics["global_iterations"],
                        diagnostics["num_microbatches"],
                        excluded=excluded,
                    )
                    row[f"{phase}_diagnostics"] = updated
                    row[f"{phase}_ms_per_op"] = updated["center_ms"]
                    row[f"median_{phase}_total_ms"] = (
                        updated["center_ms"] * diagnostics["num_microbatches"]
                    )
                final = fit_phase(trial, phase)
                if len(final["predicted"]) != len(rows) or any(
                    not math.isfinite(p) or p <= 0 for p in final["predicted"]
                ):
                    raise FitStructureError("refit_has_nonpositive_prediction")
            except FitStructureError as exc:
                status, reason = "filter_not_applied", str(exc)
                applied = [False] * len(candidate)
                final = initial
            else:
                for original, updated in zip(final_rows, trial):
                    for key in (
                        f"{phase}_diagnostics",
                        f"{phase}_ms_per_op",
                        f"median_{phase}_total_ms",
                    ):
                        original[key] = updated[key]
        for record, proposed, used in zip(records, candidate, applied):
            record.update(candidate_excluded=proposed, applied_excluded=used)
        models[phase] = final
        reports[phase] = dict(
            status=status,
            reason=reason,
            initial_model=initial,
            final_model=final,
            initial_residual_median=median,
            initial_residual_scaled_mad=scale,
            samples=records,
            raw_sample_count=len(records),
            candidate_excluded_count=sum(candidate),
            applied_excluded_count=sum(applied),
            retained_sample_count=len(records) - sum(applied),
            mask_passes=1 if has_raw else 0,
            refit_attempts=int(any(candidate)),
        )
    for row in final_rows:
        if "backward_forward_ratio" in row:
            row["backward_forward_ratio"] = row["backward_ms_per_op"] / row["forward_ms_per_op"]
    return models, final_rows, dict(**policy.identity(), phases=reports)
