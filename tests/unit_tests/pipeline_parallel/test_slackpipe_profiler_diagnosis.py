# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Diagnostic aggregation, not a claim about prediction accuracy."""

import json

import pytest

from tools.slackpipe_diagnose_composition import relative_summary
from tools.slackpipe_diagnose_residuals import diagnose
from tools.slackpipe_diagnose_trace import analyze
from tools.slackpipe_stage_calibration import digest


def test_sum_layers_then_mean_microbatches_preserves_slow_sample():
    rows = [
        dict(
            iteration=i,
            microbatch=b,
            mode="isolated",
            phase="B",
            measured=i > 0,
            event_ms=(100 if i == 2 else 1) * (layer + 1),
        )
        for i in range(3)
        for b in range(4)
        for layer in range(2)
    ]
    summary = relative_summary(rows, "isolated", "B")
    assert summary["ordered_iteration_mean_ms"] == [3, 300]
    assert summary["statistics"]["mean"] == 151.5
    assert summary["statistics"]["count"] == 2


def test_historical_manifest_tampering_fails_before_fit(tmp_path):
    raw = tmp_path / "raw.json"
    raw.write_text("[]")
    manifest = dict(files=[dict(path=raw.name, sha256=digest(raw))])
    (tmp_path / "FINAL_MANIFEST.json").write_text(json.dumps(manifest))
    raw.write_text("[1]")
    with pytest.raises(ValueError, match="manifest mismatch"):
        diagnose(tmp_path)


def test_trace_requires_real_backward_node_and_deduplicates_activity():
    def event(name, cat, ts, dur, tid=1, correlation=None, stream=None):
        return dict(
            name=name,
            cat=cat,
            ts=ts,
            dur=dur,
            tid=tid,
            pid=1,
            ph="X",
            args=dict(correlation=correlation, stream=stream),
        )

    events = [
        event("Composition/segment/0/B", "user_annotation", 0, 100),
        event("CompositionNode/segment/0/B/Node", "user_annotation", 10, 50, tid=2),
        event("launch", "cuda_runtime", 20, 1, tid=2, correlation=1),
        event("launch", "cuda_runtime", 22, 1, tid=2, correlation=2),
        event("kernel1", "kernel", 40, 20, tid=9, correlation=1, stream=7),
        event("kernel2", "kernel", 50, 30, tid=9, correlation=2, stream=8),
        event("launch", "cuda_runtime", 25, 1, tid=3, correlation=3),
        event("unrelated", "kernel", 90, 2, tid=9, correlation=3, stream=9),
    ]
    result = analyze(dict(traceEvents=events))
    assert result["rows"][0]["gpu_activity_union_us"] == 40
    assert result["rows"][0]["identity"] == ["segment", "0", "B"]
    assert len(result["assignments"]) == 2
    assert len(result["shared_unattributed"]) == 1
