# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Artifact integrity regressions; synthetic timings are not GPU measurements."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from megatron.core.pipeline_parallel.slackpipe.cost_profile import aggregate_stage_costs
from megatron.core.pipeline_parallel.slackpipe.profile_integrity import (
    publish_profile,
    tag_calibration_events,
    validate_profile_integrity,
)


def stable_profile():
    rows = []
    for group in range(6):
        for stage in range(2):
            rows.append(
                dict(
                    calibration_group_id=f"partition{group}",
                    stage=stage,
                    stage_layer_range=[stage * 8, (stage + 1) * 8],
                    stage_role="first" if stage == 0 else "last",
                    worker=stage,
                    measurement_context="same-hardware-model-and-timing",
                    raw_data_path=f"/fixture/partition{group}/events.rank{stage}.json",
                    layer_count=8,
                    forward_ms_per_op=8.0,
                    backward_ms_per_op=24.8,
                    forward_diagnostics=dict(samples_ms=[64.0, 64.0, 64.0]),
                    backward_diagnostics=dict(samples_ms=[198.4, 198.4, 198.4]),
                )
            )
    return dict(
        schema_version="slackpipe.cost_profile.v1",
        units="milliseconds",
        profiling_backend="stage",
        observed_stages=rows,
        a_fwd=1.0,
        a_bwd=3.1,
        bias_fwd=[0.0, 0.0],
        bias_bwd=[0.0, 0.0],
    )


def shifted_profile():
    profile = stable_profile()
    for row in profile["observed_stages"]:
        if row["calibration_group_id"] in ("partition4", "partition5"):
            row["backward_ms_per_op"] = 55.0
            row["backward_diagnostics"]["samples_ms"] = [440.0] * 3
    return profile


@pytest.mark.parametrize("factory", [stable_profile, shifted_profile])
@pytest.mark.parametrize("quality", [None, {"status": "failed"}, {"status": "rerun_required"}])
def test_valid_artifact_does_not_require_statistical_approval(tmp_path, factory, quality):
    profile = factory()
    if quality is not None:
        profile["quality"] = quality
    original = copy.deepcopy(profile)
    path = publish_profile(tmp_path / "cost_profile.json", profile)
    saved = json.loads(path.read_text())
    validate_profile_integrity(saved)
    assert profile == original
    assert saved.get("quality") == quality
    assert saved["observed_stages"] == profile["observed_stages"]
    assert not list(tmp_path.glob("*.quality.json"))


def test_variance_fit_error_and_sample_count_are_not_gates(tmp_path):
    profile = stable_profile()
    profile["observed_stages"][0]["forward_diagnostics"]["samples_ms"] = [1.0, 127.0]
    profile["a_fwd"] = 20.0
    profile["fit"] = {"forward": {"relative_rmse": 10.0, "condition": 1e20}}
    validate_profile_integrity(profile)
    publish_profile(tmp_path / "cost_profile.json", profile)


@pytest.mark.parametrize(
    "key,value",
    [
        ("units", "microseconds"),
        ("schema_version", "unknown"),
        ("a_fwd", float("nan")),
        ("a_bwd", -1.0),
        ("bias_fwd", [0.0]),
        ("bias_bwd", [0.0, float("inf")]),
        ("calibration_partition", [8, 0]),
    ],
)
def test_structural_corruption_is_still_rejected(key, value):
    profile = stable_profile()
    profile[key] = value
    with pytest.raises(ValueError):
        validate_profile_integrity(profile)


def test_publication_revalidates_hash_and_never_overwrites(tmp_path):
    path = publish_profile(tmp_path / "cost_profile.json", stable_profile())
    original = path.read_bytes()
    saved = json.loads(original)
    saved["a_fwd"] *= 10
    with pytest.raises(ValueError, match="hash"):
        validate_profile_integrity(saved)
    with pytest.raises(FileExistsError):
        publish_profile(path, shifted_profile())
    assert path.read_bytes() == original


def test_raw_group_identity_survives_aggregation(tmp_path):
    events = [
        dict(logical_stage=s, phase=p, iteration=i, microbatch=0, elapsed_ms=1.0)
        for s in range(2)
        for p in ("forward_compute", "backward_compute")
        for i in range(3)
    ]
    tag_calibration_events(
        events,
        group="group7",
        ranges=[[0, 1], [1, 2]],
        worker=0,
        configuration={"model": "fixture"},
        raw_path=tmp_path / "raw.json",
        attempt=1,
    )
    rows = aggregate_stage_costs(
        events, layer_split=[1, 1], num_microbatches=1, iteration_start=0, iteration_end=2
    )
    assert all(r["calibration_group_id"] == "group7" and r["attempt"] == 1 for r in rows)
    samples = [
        e
        for row in rows
        for phase in ("forward", "backward")
        for e in row[f"{phase}_diagnostics"]["raw_events"]
    ]
    assert len({e["sample_id"] for e in samples}) == len(events)
    assert all(e["raw_data_path"] == str((tmp_path / "raw.json").resolve()) for e in samples)
    events[0]["calibration_group_id"] = "other"
    with pytest.raises(ValueError, match="different calibration groups"):
        aggregate_stage_costs(
            events, layer_split=[1, 1], num_microbatches=1, iteration_start=0, iteration_end=2
        )


def test_integrity_cli_checks_existing_artifact_without_collection(tmp_path):
    path = publish_profile(tmp_path / "cost_profile.json", shifted_profile())
    original = path.read_bytes()
    command = [sys.executable, "-m", "tools.slackpipe_profile_integrity", "--profile", str(path)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert path.read_bytes() == original
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cost_profile.json"]
    profile = json.loads(original)
    profile["a_fwd"] = -1
    path.write_text(json.dumps(profile))
    failed = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert failed.returncode != 0


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("filtered", [False, True])
def test_profile_cpp_compatibility(tmp_path, version, filtered):
    binary = Path(__file__).resolve().parents[3] / "slackpipe/build/no-or/slackpipe_cli"
    if not binary.is_file():
        pytest.skip("requires the no-OR C++ build")
    profile = stable_profile()
    if filtered:
        from tests.unit_tests.pipeline_parallel.test_slackpipe_outliers import raw_profile

        profile, _ = raw_profile(tmp_path, [(0, "forward", 13, 310)], stages=2)
    profile["calibration_partition"] = [1, 1] if filtered else [8, 8]
    layers = sum(profile["calibration_partition"])
    stages = 2
    if version == 2:
        from tests.unit_tests.pipeline_parallel.test_slackpipe_group_quality import group_profile

        # v2 has a different model, so generate it rather than relabel an affine fit.
        profile = group_profile(tmp_path, [19], raw_spike=filtered)
        layers, stages = 6, 3
    path = publish_profile(tmp_path / "cost_profile.json", profile)
    assert path.read_text().count('"schema_version"') == 1
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
            str(stages),
            "--J",
            "2",
            "--L",
            str(layers),
            "--cost-profile",
            str(path),
            "--output-prefix",
            str(tmp_path / "solver"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
