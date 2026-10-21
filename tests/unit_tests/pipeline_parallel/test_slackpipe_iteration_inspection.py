# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Raw evidence replay does not select samples or gate profile usage."""

import copy
import hashlib
import json
import sys

import pytest

from megatron.core.pipeline_parallel.slackpipe.cost_profile import (
    build_cost_profile,
    profile_fingerprint,
)
from megatron.core.pipeline_parallel.slackpipe.profile_integrity import tag_calibration_events
from tools.slackpipe_inspect_iterations import inspect_iterations, main


def fixture(root, pattern, *, spike=False):
    events = [
        dict(
            iteration=5 + i,
            logical_stage=s,
            phase=phase + "_compute",
            microbatch=b,
            elapsed_ms=(pattern[i] if s == 0 and phase == "forward" else 20.0),
        )
        for i in range(len(pattern))
        for s in range(2)
        for phase in ("forward", "backward")
        for b in range(8)
    ]
    if spike:
        events[3]["elapsed_ms"] = 400.0
    tag_calibration_events(
        events,
        group="partition0",
        ranges=[(0, 2), (2, 4)],
        worker=0,
        configuration=dict(model="fixture"),
        raw_path=root / "events.json",
    )
    profile = build_cost_profile(
        events=events,
        model_config=dict(num_layers=4),
        parallel_config=dict(pp=1, vpp=2, tp=1, dp=1, cp=1),
        layer_split=[2, 2],
        num_microbatches=8,
        iteration_start=5,
        iteration_end=4 + len(pattern),
    )
    profile["cost_profile_hash"] = profile_fingerprint(profile)
    return profile, events


def test_all_groups_are_inspected_without_mutation(tmp_path):
    inputs = fixture(tmp_path, [10.0] * 3)
    before = copy.deepcopy(inputs)
    result = inspect_iterations(*inputs)
    assert inputs == before
    assert result["stage_phase_count"] == 4
    assert not result["iteration_filter_enabled"]
    assert "profile_status" not in result and "thresholds" not in result


@pytest.mark.parametrize(
    "pattern",
    [
        [10.0],
        [10.0, 100.0, 10.0],
        [10.0, 0.1, 10.0],
        [10.0] * 3 + [20.0] * 3,
        [10.0 + i for i in range(6)],
    ],
    ids=["single", "slow", "fast", "clusters", "drift"],
)
def test_all_raw_iterations_remain_in_time_order(tmp_path, pattern):
    profile, events = fixture(tmp_path, pattern)
    profile["quality"] = dict(status="rerun_required", issues=[dict(rule="within_group_cv")])
    profile["cost_profile_hash"] = profile_fingerprint(profile)
    result = inspect_iterations(profile, events)
    group = result["groups"][0]
    assert [i["mean_ms"] for i in group["iterations"]] == pattern
    assert [i["iteration"] for i in group["iterations"]] == list(range(5, 5 + len(pattern)))
    assert all(i["raw_microbatch_count"] == 8 for i in group["iterations"])
    assert "quality" not in result and "failures" not in result


def test_residual_exclusions_do_not_filter_diagnostic_replay(tmp_path):
    profile, events = fixture(tmp_path, [10.0] * 5, spike=True)
    raw = [
        dict(events=list(reversed(events[len(events) // 2 :]))),
        dict(events=events[: len(events) // 2]),
    ]
    result = inspect_iterations(profile, raw)
    first = result["groups"][0]["iterations"][0]
    assert first["raw_microbatch_count"] == 8
    assert first["aggregate_ms"] == 470
    assert first["mean_ms"] == 470 / 8
    assert first["max_ms"] == 400
    assert profile["observed_stages"][0]["forward_diagnostics"]["samples_discarded"] == 1


def test_unrelated_range_group_and_context_never_enter_iteration_mean(tmp_path):
    profile, events = fixture(tmp_path, [10.0, 20.0, 10.0])
    expected = inspect_iterations(profile, events)
    for field, value in (
        ("calibration_group_id", "other-group"),
        ("measurement_context", "other-configuration"),
        ("stage_layer_range", [8, 9]),
        ("attempt", 1),
        ("worker", 3),
    ):
        other = copy.deepcopy(events)
        for event in other:
            event[field] = value
            event["elapsed_ms"] *= 100
        assert inspect_iterations(profile, other + events) == expected


@pytest.mark.parametrize(
    "corruption", ["duplicate", "missing", "context", "range", "worker", "timing", "nan", "profile"]
)
def test_mismatched_evidence_fails_closed(tmp_path, corruption):
    profile, events = fixture(tmp_path, [10.0, 20.0, 10.0])
    if corruption == "duplicate":
        events[0]["microbatch"] = 1
    elif corruption == "missing":
        events.pop(0)
    elif corruption in ("context", "range", "worker"):
        field = dict(context="measurement_context", range="stage_layer_range", worker="worker")[
            corruption
        ]
        events[0][field] = dict(context="different", range=[1, 2], worker=99)[corruption]
    elif corruption in ("timing", "nan"):
        events[0]["elapsed_ms"] = float("nan") if corruption == "nan" else 11
    else:
        profile["a_fwd"] += 1
    with pytest.raises(ValueError):
        inspect_iterations(profile, events)


def cli_inputs(tmp_path, monkeypatch, *, corrupt=False):
    inputs = fixture(tmp_path, [10.0, 100.0, 10.0])
    if corrupt:
        inputs[1].pop(0)
    paths = [tmp_path / f"{name}.json" for name in ("profile", "events")]
    for path, value in zip(paths, inputs):
        path.write_text(json.dumps(value))
    output = tmp_path / "inspection.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["inspect", "--profile", str(paths[0]), "--events", str(paths[1]), "--output", str(output)],
    )
    return paths, output


def test_cli_hashes_inputs_and_refuses_overwrite(tmp_path, monkeypatch):
    paths, output = cli_inputs(tmp_path, monkeypatch)
    before = [p.read_bytes() for p in paths]
    assert main() == 0
    report = json.loads(output.read_text())
    assert report["diagnostic_only"] and report["stage_phase_count"] == 4
    for key, data in zip(("profile", "events"), before):
        assert report["inputs"][key]["sha256"] == hashlib.sha256(data).hexdigest()
    saved = output.read_bytes()
    assert main() == 2
    assert output.read_bytes() == saved
    assert [p.read_bytes() for p in paths] == before


def test_no_output_on_failed_replay(tmp_path, monkeypatch):
    _, output = cli_inputs(tmp_path, monkeypatch, corrupt=True)
    assert main() == 2
    assert not output.exists()
