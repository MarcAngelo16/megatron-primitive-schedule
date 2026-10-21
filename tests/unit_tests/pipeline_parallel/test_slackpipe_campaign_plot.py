# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Resume avoids reparsing completed captures, but repairs interrupted plots."""

import json

import pytest

from tools import plot_schedule_trace
from tools.run_slackpipe_real_system_campaign import plot_campaign
from tools.slackpipe_eval_config import fingerprint


@pytest.mark.parametrize("complete", [True, False])
def test_campaign_plot_completion(tmp_path, monkeypatch, complete):
    receipts = {
        s: dict(status="passed", directory=f"traces/{s}")
        for s in ("interleaved-1f1b", "slack-heuristic")
    }
    destination = (
        tmp_path
        / "traces"
        / ("figure-" + fingerprint({s: r["directory"] for s, r in receipts.items()})[:12])
    )
    destination.mkdir(parents=True)
    (destination / "timeline.png").write_bytes(b"old interrupted PNG")
    if complete:
        (destination / "timeline.pdf").write_bytes(b"PDF")
        (destination / "figure_report.json").write_text('{"selection": {"iteration": 23}}')
    parsed = []

    def iterations(path, cycle):
        assert not complete, "A completed figure must be skipped before trace parsing"
        parsed.append(path)
        assert cycle == 0
        return {23, 24} if path.name == "interleaved-1f1b" else {24, 25}

    def panel(path, iteration, cycle):
        assert iteration == 24 and cycle == 0
        return path

    def render(*args, png, pdf, **kwargs):
        png.write_bytes(b"new PNG")
        pdf.write_bytes(b"new PDF")
        return dict(panels=2)

    monkeypatch.setattr(plot_schedule_trace, "available_iterations", iterations)
    monkeypatch.setattr(plot_schedule_trace, "load_panel", panel)
    monkeypatch.setattr(plot_schedule_trace, "render", render)
    plot_campaign(tmp_path, receipts)
    assert len(parsed) == (0 if complete else 2)
    if not complete:
        report = json.loads((destination / "figure_report.json").read_text())
        assert report["selection"]["iteration"] == 24
        assert report["selection"]["candidates"] == [24]
        assert (destination / "timeline.png").read_bytes() == b"new PNG"


def test_five_panel_plot_checks_iteration_capture_and_method(tmp_path):
    import copy

    from tools.slackpipe_eval_config import METHODS

    config = dict.fromkeys(
        (
            "pp num_stages num_layers num_microbatches hidden_size heads seq_length micro_batch_size "
            "vocab_size seed learning_rate dtype tf32 dropout tp dp cp model model_sha256"
        ).split(),
        1,
    )
    config.update(pp=2, num_stages=4, num_layers=8, num_microbatches=4)
    collection = dict(
        profiler={"shape": False},
        active_iterations=[24],
        cycle=0,
        environment=dict.fromkeys(
            ("allocator_environment", "allocator_backend", "torch", "cuda", "nccl"), "same"
        ),
    )
    panels = [
        dict(
            mode=m,
            config=dict(config, num_stages=2 if m == "1f1b" else 4),
            collection=copy.deepcopy(collection),
            iteration=24,
            rank_count=2,
            duration_ms=12.0,
            records=[dict(kind="F", microbatch=0, rank=0, start_ms=0, end_ms=3)],
            gpu_activities=[],
            cuda_allocation_calls=[],
        )
        for m in METHODS
    ]
    options = dict(
        png=tmp_path / "five.png",
        pdf=tmp_path / "five.pdf",
        title="Fixture",
        caption="Synthetic test, not measurements",
        color_mode="microbatch",
    )
    report = plot_schedule_trace.render(panels, **options)
    assert len(report["panels"]) == 5
    assert options["png"].stat().st_size > 1000 and options["pdf"].stat().st_size > 1000
    bad = copy.deepcopy(panels)
    bad[-1]["iteration"] = 25
    with pytest.raises(ValueError, match="iteration"):
        plot_schedule_trace.render(bad, **options)
    bad = copy.deepcopy(panels)
    bad[-1]["collection"]["cycle"] = 1
    with pytest.raises(ValueError, match="Capture"):
        plot_schedule_trace.render(bad, **options)
    bad[-1]["mode"] = "slackpipe"
    with pytest.raises(ValueError, match="Historical"):
        plot_schedule_trace.render(bad, **options)
