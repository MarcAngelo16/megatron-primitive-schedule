# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Synthetic operation residuals, not GPU measurements or stability acceptance tests."""

import copy
import json

import pytest

from megatron.core.pipeline_parallel.slackpipe.cost_profile import build_cost_profile
from megatron.core.pipeline_parallel.slackpipe.profile_integrity import (
    publish_profile,
    tag_calibration_events,
    validate_profile_integrity,
)
from megatron.core.pipeline_parallel.slackpipe.stage_residual import ResidualPolicy


def raw_profile(
    root, spikes=(), *, stages=8, iterations=30, microbatches=8, policy=None, normal=None
):
    """Generate one tagged calibration group with independent stage/phase clocks."""
    events = []
    overrides = {(s, p, i): v for s, p, i, v in spikes}
    for stage in range(stages):
        local = []
        for phase in ("forward", "backward"):
            for i in range(iterations * microbatches):
                value = (
                    normal(stage, phase, i) if normal else (10.0 if phase == "forward" else 20.0)
                )
                local.append(
                    dict(
                        logical_stage=stage,
                        phase=phase + "_compute",
                        iteration=5 + i // microbatches,
                        microbatch=i % microbatches,
                        rank=stage % 4,
                        elapsed_ms=overrides.get((stage, phase, i), value),
                    )
                )
        tag_calibration_events(
            local,
            group="partition0",
            ranges=[[s, s + 1] for s in range(stages)],
            worker=stage % 4,
            configuration={"model": "fixture"},
            raw_path=root / "raw.json",
        )
        events.extend(local)
    original = copy.deepcopy(events)
    profile = build_cost_profile(
        events=events,
        model_config={},
        parallel_config={},
        layer_split=[1] * stages,
        num_microbatches=microbatches,
        iteration_start=5,
        iteration_end=5 + iterations - 1,
        residual_policy=policy or ResidualPolicy(),
    )
    assert events == original
    return profile, events


def test_stable_data_preserves_estimator(tmp_path):
    profile, _ = raw_profile(tmp_path)
    validate_profile_integrity(profile)
    assert profile["residual_filter"]["phases"]["forward"]["applied_excluded_count"] == 0
    assert profile["a_fwd"] == 10 and profile["a_bwd"] == 20
    assert profile["bias_fwd"] == [0] * 8
    assert profile["observed_stages"][0]["forward_diagnostics"]["samples_ms"] == [80] * 30


def test_one_spike_before_iteration_mean_and_raw_unchanged(tmp_path):
    profile, events = raw_profile(tmp_path, [(4, "forward", 13, 356)])
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps(events))
    original = raw.read_bytes()
    publish_profile(tmp_path / "cost_profile.json", profile)
    assert raw.read_bytes() == original
    phase = profile["residual_filter"]["phases"]["forward"]
    assert phase["applied_excluded_count"] == 1
    rejected = next(s for s in phase["samples"] if s["applied_excluded"])
    assert (rejected["iteration"], rejected["microbatch"], rejected["observed_ms"]) == (6, 5, 356)
    assert rejected["prediction_ms"] == 10
    assert rejected["relative_residual"] == pytest.approx(34.6)
    assert phase["initial_residual_scaled_mad"] == 0
    assert rejected["raw_data_path"] == str(raw.resolve())
    row = profile["observed_stages"][4]
    assert row["forward_ms_per_op"] == 10
    assert row["forward_diagnostics"]["samples_ms"][1] == 80
    assert row["forward_diagnostics"]["accepted_counts_by_iteration"][1] == 7
    assert row["forward_diagnostics"]["raw_sample_count"] == 240
    assert row["forward_diagnostics"]["accepted_sample_count"] == 239
    assert (
        profile["raw_observed_stages"][4]["forward_diagnostics"]["raw_events"][13]["elapsed_ms"]
        == 356
    )


def test_reported_synthetic_pattern_is_filtered_only_by_residual(tmp_path):
    spikes = [
        (4, "forward", 13, 356),
        (2, "forward", 50, 313),
        (5, "forward", 57, 306),
        (3, "forward", 98, 286),
    ]
    profile, _ = raw_profile(tmp_path, spikes)
    phases = profile["residual_filter"]["phases"]
    assert phases["forward"]["applied_excluded_count"] == 4
    assert phases["backward"]["applied_excluded_count"] == 0
    assert "quality" not in profile
    validate_profile_integrity(profile)


def test_rare_multiple_spikes_and_nonzero_mad(tmp_path):
    profile, _ = raw_profile(
        tmp_path,
        [(0, "forward", i, 310) for i in (13, 50, 98, 170)],
        normal=lambda s, p, i: (10 if p == "forward" else 20) + (i % 3 - 1) * 0.1,
    )
    phase = profile["residual_filter"]["phases"]["forward"]
    assert phase["applied_excluded_count"] == 4
    assert phase["initial_residual_scaled_mad"] > 0


def test_moderate_variation_is_not_trimmed_or_rejected(tmp_path):
    profile, _ = raw_profile(tmp_path, normal=lambda s, p, i: 9 + (i // 8 % 5))
    assert all(
        p["applied_excluded_count"] == 0 for p in profile["residual_filter"]["phases"].values()
    )
    publish_profile(tmp_path / "cost_profile.json", profile)
    assert profile["a_fwd"] == 11


@pytest.mark.parametrize("spike,spread", [(15, 0), (25, 2)])
def test_both_residual_conditions_must_be_exceeded(tmp_path, spike, spread):
    profile, _ = raw_profile(
        tmp_path,
        [(0, "forward", 13, spike)],
        normal=lambda s, p, i: 10 + (spread if i % 2 else -spread),
    )
    # rho equality fails the strict test; in the other case the MAD condition fails.
    assert profile["residual_filter"]["phases"]["forward"]["applied_excluded_count"] == 0


@pytest.mark.parametrize(
    "indices,kwargs",
    [
        ([1, 25, 49, 73, 97], {}),
        ([13, 14], {}),
        ([7, 8], {}),
        ([0, 100, 200, 300, 400, 500, 600], {"microbatches": 100}),
    ],
)
def test_systemic_pattern_limits_are_not_acceptance_gates(tmp_path, indices, kwargs):
    profile, _ = raw_profile(tmp_path, [(0, "forward", i, 310) for i in indices], **kwargs)
    phase = profile["residual_filter"]["phases"]["forward"]
    assert phase["applied_excluded_count"] == len(indices)
    assert phase["mask_passes"] == phase["refit_attempts"] == 1
    publish_profile(tmp_path / "cost_profile.json", profile)


def test_stage_and_phase_isolation(tmp_path):
    profile, _ = raw_profile(
        tmp_path,
        [(0, "forward", 13, 310)],
        normal=lambda s, p, i: 100 if s == 7 or p == "backward" else 10,
    )
    assert profile["residual_filter"]["phases"]["forward"]["applied_excluded_count"] == 1
    assert profile["residual_filter"]["phases"]["backward"]["applied_excluded_count"] == 0
    assert profile["observed_stages"][7]["forward_ms_per_op"] == 100
    assert profile["observed_stages"][0]["backward_ms_per_op"] == 100


def test_changed_policy_is_explicit_and_does_not_rewrite_raw_samples(tmp_path):
    first, events = raw_profile(tmp_path, [(0, "forward", 13, 20)])
    second, same_events = raw_profile(
        tmp_path, [(0, "forward", 13, 20)], policy=ResidualPolicy(rho=1.1)
    )
    assert events == same_events
    assert first["profiling_policy"] != second["profiling_policy"]
    assert first["residual_filter"]["phases"]["forward"]["applied_excluded_count"] == 1
    assert second["residual_filter"]["phases"]["forward"]["applied_excluded_count"] == 0


def test_few_raw_samples_have_no_minimum_stability_gate(tmp_path):
    profile, _ = raw_profile(tmp_path, [(0, "forward", 1, 310)], iterations=3, microbatches=4)
    assert profile["residual_filter"]["phases"]["forward"]["applied_excluded_count"] == 1
    validate_profile_integrity(profile)


@pytest.mark.parametrize(
    "policy", [{"k": 0}, {"k": float("nan")}, {"rho": -0.1}, {"rho": float("inf")}]
)
def test_invalid_policy(policy):
    with pytest.raises(ValueError):
        ResidualPolicy(**policy)
