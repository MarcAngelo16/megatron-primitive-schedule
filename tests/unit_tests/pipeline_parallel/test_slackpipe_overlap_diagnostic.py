# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Conservative CUDA correlation and isolated serialized-control boundaries."""

import copy
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch

from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.slackpipe.timing_diagnostic import (
    ComputeTimingDiagnostic,
    DiagnosticP2PCommunicator,
)
from tests.unit_tests.pipeline_parallel.test_slackpipe_figure_trace import raw_trace
from tools.slackpipe_analyze_overlap import compact_partition, compare, overlap_evidence


def test_stepwise_correlation_preserves_capture_and_ignores_gpu_mirrors():
    raw = raw_trace()
    second = copy.deepcopy(raw["traceEvents"])
    for event in second:
        event["ts"] += 200
        event["name"] = event["name"].replace("/i5/", "/i6/")
    raw["traceEvents"].extend(second)
    raw["traceEvents"].append(
        dict(ph="X", cat="gpu_user_annotation", name="ScheduleTrace/step/i5/s-1/b-1", ts=10, dur=75)
    )
    metadata = dict(
        rank=0,
        profiler=dict(captured_iterations=[5, 6]),
        calibration_configuration=dict(pp=2, stages=2, microbatches=1),
        timing_diagnostic=dict(
            iterations=[dict(iteration=i, cuda_elapsed_ms=0.08) for i in (5, 6)]
        ),
    )
    result = compact_partition(raw, metadata, "normal")
    assert {e["iteration"] for e in result["records"]} == {5, 6}
    assert len({e["activity_id"] for e in result["gpu_activities"]}) == 6
    for op in result["records"]:
        assert all(
            result["gpu_activities"][i]["iteration"] == op["iteration"] for i in op["activity_ids"]
        )
    metadata["profiler"]["captured_iterations"].append(7)
    with pytest.raises(ValueError, match="coverage"):
        compact_partition(raw, metadata, "normal")


def test_overlap_union_prior_p2p_and_co_resident_work():
    def activity(i, start, end, comm=False, correlation=None):
        return dict(
            activity_id=i,
            start_us=start,
            end_us=end,
            communication=comm,
            name="nccl" if comm else "compute",
            args=dict(correlation=correlation),
        )

    def operation(stage, ids, start, end):
        return dict(
            stage=stage,
            iteration=20,
            microbatch=0,
            kind="B",
            activity_ids=ids,
            start_us=start,
            end_us=end,
            cpu_start_us=10,
            cpu_end_us=20,
            gpu_activity_union_us=end - start,
            internal_gap_us=0,
            cuda_malloc_count=0,
            cuda_malloc_cpu_union_us=0,
        )

    raw = dict(
        traceEvents=[
            dict(
                cat="cuda_runtime",
                ph="X",
                name="cudaLaunchKernel",
                ts=2,
                dur=1,
                pid=1,
                tid=1,
                args=dict(correlation=9),
            ),
            dict(
                cat="user_annotation",
                ph="X",
                name="SlackPipeDiagnostic/P2P/i20/p0",
                ts=1,
                dur=5,
                pid=1,
                tid=1,
            ),
            dict(cat="cuda_runtime", ph="X", name="cudaStreamWaitEvent", ts=12, dur=0.1, args={}),
        ]
    )
    compact = dict(
        gpu_activities=[
            activity(0, 100, 120),
            activity(1, 95, 110, True, 9),
            activity(2, 105, 115, True),
            activity(3, 113, 118),
        ],
        records=[operation(0, [0], 100, 120), operation(2, [3], 113, 118)],
    )
    evidence = overlap_evidence(raw, compact)[(20, 0, 0, "B")]
    assert evidence["nccl_overlap_union_us"] == 15  # Not the overlapping sum 20.
    assert evidence["earlier_p2p_activity_ids"] == [1]
    assert evidence["co_resident_activity"][0]["stage"] == 2
    assert len(evidence["stream_wait_api_candidates"]) == 1
    assert evidence["compute_stream_wait"].startswith("unknown")
    assert evidence["cuda_event_interval_overlap"].startswith("unknown")
    assert evidence["inferred_cause"].startswith("unresolved")


def test_serialized_only_drains_around_compute(monkeypatch):
    order = []
    event = lambda **kw: SimpleNamespace(
        record=lambda *a: order.append("event"), elapsed_time=lambda e: 1
    )
    monkeypatch.setattr(torch.cuda, "Event", event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=7))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: order.append("sync"))
    probe = ComputeTimingDiagnostic(1, serialized=True)
    probe.begin(20)
    order.clear()
    probe.compute(lambda: order.append("compute"), dict(phase="backward_compute"))
    assert order == ["sync", "event", "compute", "event", "sync"]
    row = probe.finish()[0]
    assert row["diagnostic_execution"] == "serialized"
    assert row["wall_ms"] >= row["cpu_wall_ms"]


@pytest.mark.parametrize(
    "batch,sync,overlap", [(False, True, False), (True, False, False), (True, True, True)]
)
def test_unsafe_serialized_communicator_rejected(batch, sync, overlap):
    with pytest.raises(ValueError, match="blocking batch P2P"):
        DiagnosticP2PCommunicator(
            None,
            SimpleNamespace(batch_p2p_comm=batch, batch_p2p_sync=sync, overlap_p2p_comm=overlap),
            SimpleNamespace(serialized=True),
        )


def test_unmatched_receive_rejected_before_launch(monkeypatch):
    native = Mock()
    monkeypatch.setattr(P2PCommunicator, "_communicate", native)
    communicator = object.__new__(DiagnosticP2PCommunicator)
    communicator.diagnostic = SimpleNamespace(active=True, position=0, serialized=True)
    with pytest.raises(ValueError, match="unmatched asynchronous"):
        communicator._communicate(wait_on_reqs=False)
    native.assert_not_called()


def test_configuration_mismatch_not_accepted():
    normal = dict(
        manifest="m",
        metadata={
            ('partition0', 0): dict(
                measurement_iterations=[20],
                warmup_iterations=[0],
                stage_layer_ranges=[[0, 2]],
                calibration_configuration=dict(precision="fp32"),
            )
        },
    )
    serial = dict(
        manifest="m",
        metadata={
            ('partition0', 0): dict(
                measurement_iterations=[20],
                warmup_iterations=[0],
                stage_layer_ranges=[[0, 2]],
                calibration_configuration=dict(precision="bf16"),
            )
        },
    )
    with pytest.raises(ValueError, match="Configuration"):
        compare(normal, serial)


def test_capture_is_optional_and_exports_once_after_window(monkeypatch, tmp_path):
    profiler = MagicMock()
    factory = Mock(return_value=profiler)
    monkeypatch.setattr(torch.profiler, "profile", factory)
    probe = object.__new__(ComputeTimingDiagnostic)
    probe.timeline = False
    with probe.capture(tmp_path / "off.json"):
        pass
    factory.assert_not_called()
    probe.timeline = True
    with probe.capture(tmp_path / "on.json"):
        profiler.__enter__.return_value.export_chrome_trace.assert_not_called()
    profiler.__enter__.return_value.export_chrome_trace.assert_called_once_with(
        str(tmp_path / "on.json")
    )
    assert factory.call_args.kwargs["record_shapes"] is False
    assert factory.call_args.kwargs["profile_memory"] is False


@pytest.mark.parametrize("serialized", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_serialization_preserves_gradients_and_update(serialized):
    x = torch.ones(32, 32, device="cuda", requires_grad=True)
    ref = torch.ones_like(x, requires_grad=True)
    (ref @ ref).sum().backward()
    with torch.no_grad():
        ref.add_(ref.grad, alpha=-0.01)
    probe = ComputeTimingDiagnostic(2, serialized=serialized, nvtx=True)
    probe.begin(0)
    loss = probe.compute(lambda: (x @ x).sum(), dict(phase="forward_compute"))
    probe.compute(loss.backward, dict(phase="backward_compute"))
    with torch.no_grad():
        x.add_(x.grad, alpha=-0.01)
    rows = probe.finish()
    torch.testing.assert_close(x, ref, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, ref.grad, rtol=0, atol=0)
    assert all(r["compute_gpu_ms"] > 0 for r in rows)
    assert all(r["start_event_handle"] and r["end_event_handle"] for r in rows)
