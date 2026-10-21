# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Synthetic repeated partitions; not measurements of the reported 4B run."""

import copy
import json
import subprocess
from pathlib import Path

import pytest

from megatron.core.pipeline_parallel.slackpipe.cost_profile import (
    aggregate_stage_costs,
    build_heterogeneous_cost_profile,
)
from megatron.core.pipeline_parallel.slackpipe.profile_integrity import (
    publish_profile,
    tag_calibration_events,
    validate_profile_integrity,
)
from megatron.core.pipeline_parallel.slackpipe.stage_residual import ResidualPolicy


def group_profile(
    root, values, *, raw_spike=False, noisy_group=None, policy=None, four_stages=False
):
    partitions = (
        ((0, 1, 3, 5, 8), (0, 2, 4, 6, 8), (0, 3, 5, 7, 8))
        if four_stages
        else ((0, 1, 3, 6), (0, 2, 4, 6), (0, 3, 5, 6))
    )
    manifest = dict(
        manifest_hash="one-class-fixture",
        layers=[dict(layer_id=i, config_class="A") for i in range(partitions[0][-1])],
    )
    rows = []
    for g, value in enumerate(values):
        for p, cuts in enumerate(partitions):
            events = []
            for s in range(len(cuts) - 1):
                local = []
                for phase in ("forward", "backward"):
                    v = (cuts[s + 1] - cuts[s]) * (1 if phase == "forward" else 2)
                    role = 0 if s == 0 else (2 if s == len(cuts) - 2 else 1)
                    v += ([0.5, 0.25, 0.75] if phase == "forward" else [17, 1, 2])[role]
                    if phase == "backward" and s == p == 0:
                        v = value
                    for i in range(30):
                        for b in range(8):
                            elapsed = v
                            if raw_spike and (g, p, s, phase, i, b) == (0, 0, 0, "forward", 6, 5):
                                elapsed = 310
                            if g == noisy_group and p == s == 0 and phase == "backward":
                                elapsed *= 0.7 if i % 2 else 1.3
                            local.append(
                                dict(
                                    logical_stage=s,
                                    phase=phase + "_compute",
                                    iteration=i,
                                    microbatch=b,
                                    elapsed_ms=elapsed,
                                    rank=s % 2,
                                )
                            )
                tag_calibration_events(
                    local,
                    group=f"g{g}-p{p}",
                    worker=s % 2,
                    ranges=list(zip(cuts, cuts[1:])),
                    configuration={"fixture": 1},
                    raw_path=root / f"g{g}-p{p}.raw.json",
                )
                events.extend(local)
            rows.extend(
                aggregate_stage_costs(
                    events,
                    layer_split=[b - a for a, b in zip(cuts, cuts[1:])],
                    num_microbatches=8,
                    iteration_start=0,
                    iteration_end=29,
                )
            )
    original = copy.deepcopy(rows)
    profile = build_heterogeneous_cost_profile(
        model_manifest=manifest,
        observed_stage_rows=rows,
        model_config={},
        parallel_config={},
        residual_policy=policy or ResidualPolicy(),
    )
    assert rows == original
    return profile


@pytest.mark.parametrize(
    "values",
    [
        [19, 19.2, 18.9, 19.1, 13.5],
        [19, 19.2, 18.9, 13.5, 13.8],
        [19, 19.1, 18.9, 19],
        [19.05, 18.8, 20.04, 13.75],
        [19, 19.2, 18.9, 19.1, 26],
        [14, 16, 18, 20, 22],
        [14, 14.2, 20, 20.1],
        [19.053, 18.799, 20.045, 13.755, 15.995],
    ],
)
def test_all_groups_participate_without_consensus_gate(tmp_path, values):
    profile = group_profile(tmp_path, values)
    validate_profile_integrity(profile)
    assert "group_filter" not in profile and "quality" not in profile
    rows = profile["observed_stages"]
    assert len(rows) == len(values) * 9
    assert {r["calibration_group_id"] for r in rows} == {
        f"g{g}-p{p}" for g in range(len(values)) for p in range(3)
    }
    report = profile["residual_filter"]["phases"]["backward"]
    assert len(report["initial_model"]["predicted"]) == len(rows)
    assert len(report["final_model"]["predicted"]) == len(rows)
    assert all(r["backward_diagnostics"]["accepted_sample_count"] > 0 for r in rows)
    path = publish_profile(tmp_path / "cost_profile.json", profile)
    validate_profile_integrity(json.loads(path.read_text()))


def test_raw_spike_filtering_does_not_require_surviving_cv(tmp_path):
    profile = group_profile(tmp_path, [19, 19.2, 18.9, 19.1, 13.5], raw_spike=True, noisy_group=0)
    phase = profile["residual_filter"]["phases"]["forward"]
    assert phase["applied_excluded_count"] == 1
    rejected = next(s for s in phase["samples"] if s["applied_excluded"])
    assert rejected["observed_ms"] == 310
    assert rejected["calibration_group_id"] == "g0-p0"
    assert profile["observed_stages"][0]["backward_diagnostics"]["accepted_sample_count"] == 240
    publish_profile(tmp_path / "cost_profile.json", profile)


def test_required_raw_observation_empty_rolls_back_whole_phase(tmp_path):
    profile = group_profile(tmp_path, [19, 19, 19, 19, 1000], raw_spike=True)
    forward = profile["residual_filter"]["phases"]["forward"]
    backward = profile["residual_filter"]["phases"]["backward"]
    assert forward["applied_excluded_count"] == 1
    assert backward["candidate_excluded_count"] > 0
    assert backward["applied_excluded_count"] == 0
    assert backward["reason"] == "required_observation_has_no_samples"
    assert backward["initial_model"] == backward["final_model"]
    for before, after in zip(profile["raw_observed_stages"], profile["observed_stages"]):
        assert before["backward_diagnostics"] == after["backward_diagnostics"]
        assert before["backward_ms_per_op"] == after["backward_ms_per_op"]
    validate_profile_integrity(profile)


@pytest.mark.parametrize(
    "key,value",
    [
        ("layer_composition", ["Mamba"]),
        ("model_manifest_hash", "other-model"),
        ("class_counts", {"A": 99}),
    ],
)
def test_cannot_launder_manifest_or_composition(tmp_path, key, value):
    profile = group_profile(tmp_path, [19, 19.2, 18.9])
    rows = copy.deepcopy(profile["raw_observed_stages"])
    rows[0][key] = value
    with pytest.raises(ValueError, match="composition"):
        build_heterogeneous_cost_profile(
            model_manifest=dict(
                manifest_hash="one-class-fixture",
                layers=[dict(layer_id=i, config_class="A") for i in range(6)],
            ),
            observed_stage_rows=rows,
            model_config={},
            parallel_config={},
        )


def test_revalidation_rejects_changed_costs_not_fit_error(tmp_path):
    profile = group_profile(tmp_path, [14, 16, 18, 20, 22])
    path = publish_profile(tmp_path / "profile.json", profile)
    saved = json.loads(path.read_text())
    saved["prefix_backward_us"][1] += 10
    with pytest.raises(ValueError, match="hash"):
        validate_profile_integrity(saved)
    del saved["cost_profile_hash"]
    with pytest.raises(ValueError, match="Prefix"):
        validate_profile_integrity(saved)


def test_summary_only_history_is_not_relabelled_as_filtered(tmp_path):
    profile = group_profile(tmp_path, [19, 19.2, 18.9])
    rows = copy.deepcopy(profile["raw_observed_stages"])
    for row in rows:
        for phase in ("forward", "backward"):
            del row[f"{phase}_diagnostics"]["raw_events"]
    rebuilt = build_heterogeneous_cost_profile(
        model_manifest=dict(
            manifest_hash="one-class-fixture",
            layers=[dict(layer_id=i, config_class="A") for i in range(6)],
        ),
        observed_stage_rows=rows,
        model_config={},
        parallel_config={},
    )
    assert all(
        p["status"] == "filter_not_applied"
        and p["reason"] == "raw_operation_samples_unavailable"
        and p["mask_passes"] == 0
        for p in rebuilt["residual_filter"]["phases"].values()
    )


@pytest.mark.parametrize("values", [[19, 19.2, 18.9, 19.1, 13.5], [14, 16, 18, 20, 22]])
def test_group_profile_solver_export_without_stability_approval(tmp_path, values):
    binary = Path(__file__).resolve().parents[3] / "slackpipe/build/no-or/slackpipe_cli"
    if not binary.exists():
        pytest.skip("requires no-OR C++ build")
    selected = publish_profile(tmp_path / "cost_profile.json", group_profile(tmp_path, values))
    result = subprocess.run(
        [
            str(binary),
            "--algorithm",
            "octopipe-tuning",
            "--time-limit",
            "1",
            "--octopipe-max-iterations",
            "1",
            "--B",
            "4",
            "--N",
            "3",
            "--J",
            "1",
            "--L",
            "6",
            "--cost-profile",
            str(selected),
            "--output-prefix",
            str(tmp_path / "solver"),
            "--emit-plan",
            str(tmp_path / "plan.json"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "plan.json").exists()
