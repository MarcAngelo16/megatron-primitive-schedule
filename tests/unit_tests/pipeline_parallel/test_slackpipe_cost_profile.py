# Copyright (c) 2026 NVIDIA CORPORATION. All rights reserved.

import pytest

from megatron.core.pipeline_parallel.slackpipe.cost_profile import (
    SLACKPIPE_COST_PROFILE_SCHEMA_VERSION_V2,
    aggregate_stage_costs,
    build_cost_profile,
    build_heterogeneous_cost_profile,
    fit_heterogeneous_class_costs,
    fit_shared_slope_stage_bias,
    percentile,
    stage_role,
)


def _event(iteration, stage, phase, elapsed_ms):
    return {
        "iteration": iteration,
        "rank": stage % 2,
        "pp_rank": stage % 2,
        "vp_rank": stage // 2,
        "model_chunk_id": stage // 2,
        "logical_stage": stage,
        "microbatch": 0,
        "phase": phase,
        "elapsed_ms": elapsed_ms,
    }


def test_percentile_interpolates_reference_style():
    assert percentile([1.0, 3.0, 5.0], 25.0) == 2.0
    assert percentile([1.0, 3.0, 5.0], 50.0) == 3.0


def test_fit_shared_slope_stage_bias_estimators():
    layers = [1, 2, 4]
    costs = [3.0, 7.0, 10.0]

    slope, bias = fit_shared_slope_stage_bias(layers, costs, estimator="min")
    assert slope == pytest.approx(2.5)
    assert bias == pytest.approx([0.5, 2.0, 0.0])

    slope, bias = fit_shared_slope_stage_bias(layers, costs, estimator="median")
    assert slope == pytest.approx(3.0)
    assert bias == pytest.approx([0.0, 1.0, 0.0])

    slope, bias = fit_shared_slope_stage_bias(
        layers, costs, estimator="percentile", percentile_value=50.0
    )
    assert slope == pytest.approx(3.0)
    assert bias == pytest.approx([0.0, 1.0, 0.0])


def test_aggregate_stage_costs_uses_iteration_medians_and_normalizes_by_microbatch():
    events = []
    for iteration, scale in [(2, 1.0), (3, 3.0), (4, 5.0)]:
        for stage in range(2):
            for microbatch in range(4):
                event = _event(iteration, stage, "forward_compute", scale * (stage + 1))
                event["microbatch"] = microbatch
                events.append(event)
                event = _event(iteration, stage, "backward_compute", scale * (stage + 2))
                event["microbatch"] = microbatch
                events.append(event)

    rows = aggregate_stage_costs(
        events, layer_split=[1, 2], num_microbatches=4, iteration_start=2, iteration_end=4
    )

    assert rows[0]["median_forward_total_ms"] == pytest.approx(12.0)
    assert rows[0]["median_backward_total_ms"] == pytest.approx(24.0)
    assert rows[0]["forward_ms_per_op"] == pytest.approx(3.0)
    assert rows[0]["backward_ms_per_op"] == pytest.approx(6.0)
    assert rows[1]["forward_ms_per_op"] == pytest.approx(6.0)
    assert rows[1]["backward_ms_per_op"] == pytest.approx(9.0)
    assert rows[0]["forward_diagnostics"]["samples_ms"] == [4, 12, 20]
    assert rows[0]["forward_diagnostics"]["global_iterations"] == [2, 3, 4]
    assert rows[0]["forward_diagnostics"]["samples_discarded"] == 0
    assert "review_required" not in rows[0]["forward_diagnostics"]


def test_build_cost_profile_schema_and_biases():
    events = [
        _event(0, 0, "forward_compute", 4.0),
        _event(0, 0, "backward_compute", 8.0),
        _event(0, 1, "forward_compute", 8.0),
        _event(0, 1, "backward_compute", 20.0),
    ]

    profile = build_cost_profile(
        events=events,
        model_config={"num_layers": 3},
        parallel_config={"pp": 2, "vpp": 1, "tp": 1, "cp": 1},
        layer_split=[1, 2],
        num_microbatches=1,
        iteration_start=0,
        iteration_end=0,
        estimator="min",
    )

    assert profile["schema_version"] == "slackpipe.cost_profile.v1"
    assert profile["a_fwd"] == pytest.approx(4.0)
    assert profile["a_bwd"] == pytest.approx(8.0)
    assert profile["bias_fwd"] == pytest.approx([0.0, 0.0])
    assert profile["bias_bwd"] == pytest.approx([0.0, 4.0])
    assert profile["measured_backward_forward_ratio"] == pytest.approx(2.0)


def _heterogeneous_manifest():
    classes = ["A", "A", "B", "C", "A", "B"]
    manifest = {
        "schema_version": "slackpipe.model_manifest.v1",
        "num_layers": len(classes),
        "layers": [
            {"layer_id": layer_id, "layer_type": "decoder", "config_class": class_id}
            for layer_id, class_id in enumerate(classes)
        ],
        "manifest_hash": "manifest-hash",
    }
    return manifest


def _heterogeneous_observations():
    fwd = {"A": 1.0, "B": 2.0, "C": 4.0}
    bwd = {"A": 3.0, "B": 5.0, "C": 7.0}
    bias_fwd = {"first": 0.5, "middle": 0.25, "last": 0.75}
    bias_bwd = {"first": 1.5, "middle": 1.25, "last": 1.75}
    classes = ["A", "A", "B", "C", "A", "B"]
    ranges = [
        (0, 1, "first"),
        (1, 3, "middle"),
        (3, 6, "last"),
        (0, 2, "first"),
        (2, 4, "middle"),
        (4, 6, "last"),
        (0, 3, "first"),
        (3, 5, "middle"),
        (5, 6, "last"),
    ]
    rows = []
    for begin, end, role in ranges:
        rows.append(
            {
                "begin": begin,
                "end": end,
                "stage_role": role,
                "forward_ms_per_op": sum(fwd[class_id] for class_id in classes[begin:end])
                + bias_fwd[role],
                "backward_ms_per_op": sum(bwd[class_id] for class_id in classes[begin:end])
                + bias_bwd[role],
            }
        )
    return rows


def test_stage_role_labels_pipeline_positions():
    assert stage_role(0, 4) == "first"
    assert stage_role(1, 4) == "middle"
    assert stage_role(2, 4) == "middle"
    assert stage_role(3, 4) == "last"


def test_summary_only_input_does_not_claim_raw_residual_policy():
    result = build_heterogeneous_cost_profile(
        model_manifest=_heterogeneous_manifest(),
        observed_stage_rows=_heterogeneous_observations(),
        model_config={},
        parallel_config={},
    )
    assert result["profiling_policy"]["policy_version"] == "stage-summary-fit-v1"
    for phase in ("forward", "backward"):
        report = result["residual_filter"]["phases"][phase]
        assert report["status"] == "filter_not_applied"
        assert report["reason"] == "raw_operation_samples_unavailable"


def test_fit_heterogeneous_class_costs_from_stage_level_observations():
    fit = fit_heterogeneous_class_costs(
        _heterogeneous_observations(),
        model_manifest=_heterogeneous_manifest(),
        value_field="forward_ms_per_op",
    )

    assert fit["class_costs"] == pytest.approx({"A": 1.0, "B": 2.0, "C": 4.0})
    assert fit["stage_role_bias"] == pytest.approx({"first": 0.5, "middle": 0.25, "last": 0.75})
    assert fit["rank"] == 6
    assert fit["max_abs_error"] == pytest.approx(0.0, abs=1.0e-10)


def test_build_heterogeneous_cost_profile_v2_expands_prefix_layer_costs():
    profile = build_heterogeneous_cost_profile(
        model_manifest=_heterogeneous_manifest(),
        observed_stage_rows=_heterogeneous_observations(),
        model_config={"num_layers": 6},
        parallel_config={"pp": 2, "vpp": 2, "tp": 1, "cp": 1},
    )

    assert profile["schema_version"] == SLACKPIPE_COST_PROFILE_SCHEMA_VERSION_V2
    assert profile["model_manifest_hash"] == "manifest-hash"
    assert profile["class_costs_us"]["A"]["forward"] == pytest.approx(1000.0)
    assert profile["class_costs_us"]["B"]["backward"] == pytest.approx(5000.0)
    assert [row["config_class"] for row in profile["layer_costs_us"]] == [
        "A",
        "A",
        "B",
        "C",
        "A",
        "B",
    ]
    assert profile["prefix_forward_us"] == pytest.approx(
        [0.0, 1000.0, 2000.0, 4000.0, 8000.0, 9000.0, 11000.0]
    )
    assert profile["prefix_backward_us"] == pytest.approx(
        [0.0, 3000.0, 6000.0, 11000.0, 18000.0, 21000.0, 26000.0]
    )
    assert profile["stage_role_bias_us"]["last"]["forward"] == pytest.approx(750.0)
    assert profile["cost_profile_hash"]


def test_heterogeneous_rank_deficiency_is_rejected():
    with pytest.raises(ValueError, match="rank-deficient"):
        fit_heterogeneous_class_costs(
            _heterogeneous_observations()[:3],
            model_manifest=_heterogeneous_manifest(),
            value_field="forward_ms_per_op",
        )


def test_constrained_fit_reports_actual_exported_prediction():
    rows = _heterogeneous_observations()
    # Positive observations whose unconstrained intercept would be negative.
    for row in rows:
        row["forward_ms_per_op"] -= 0.75
    fit = fit_heterogeneous_class_costs(
        rows, model_manifest=_heterogeneous_manifest(), value_field="forward_ms_per_op"
    )
    classes = [r["config_class"] for r in _heterogeneous_manifest()["layers"]]
    for i, row in enumerate(rows):
        prediction = (
            sum(fit["class_costs"][c] for c in classes[row["begin"] : row["end"]])
            + fit["stage_role_bias"][row["stage_role"]]
        )
        assert fit["predicted"][i] == pytest.approx(prediction)
        assert fit["residuals"][i] == pytest.approx(row["forward_ms_per_op"] - prediction)
    assert all(c >= 0 for c in fit["class_costs"].values())
    assert all(c >= 0 for c in fit["stage_role_bias"].values())
