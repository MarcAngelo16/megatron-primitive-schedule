"""Collection contracts, including profiler cycles without depending on CUPTI."""

import copy
import json
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import pytest

from megatron.core.pipeline_parallel.slackpipe import collection as c
from megatron.core.pipeline_parallel.slackpipe.figure_trace import (
    activity_metrics,
    compact_profiler_trace,
    interval_union,
)
from tests.unit_tests.pipeline_parallel.test_slackpipe_figure_trace import raw_trace
from tools.plot_schedule_trace import available_iterations, load_panel, plot_time_bounds
from tools.slackpipe_eval_common import summarize_samples


def test_union_not_envelope_or_sum():
    intervals = [(0, 10), (3, 7), (5, 15), (20, 22), (22, 25), (20, 22)]
    assert interval_union(intervals) == [[0, 15], [20, 25]]
    assert activity_metrics([dict(ts=a, dur=b - a) for a, b in intervals]) == dict(
        gpu_activity_union_us=20,
        gpu_envelope_us=25,
        internal_gap_us=5,
        gpu_union_intervals_us=[[0, 15], [20, 25]],
    )
    with pytest.raises(ValueError):
        interval_union([(2, 1)])


def test_plot_retains_allocation_calls_outside_gpu_envelope():
    panels = [dict(duration_ms=10, cuda_allocation_calls=[dict(start_ms=-5, end_ms=12)])]
    assert plot_time_bounds(panels, detail=True) == (-5, 12)
    assert plot_time_bounds(panels, detail=False) == (0, 10)


def test_profiler_defaults_and_global_iteration_mapping():
    options = c.CollectionOptions(mode="timeline")
    assert c.capture_windows(options) == [
        list(range(24, 27)),
        list(range(31, 34)),
        list(range(38, 41)),
    ]
    assert options.steps == 21
    settings = c.profiler_settings(options)
    assert settings["activities"] == ["CPU", "CUDA"]
    assert all(
        not settings[k] for k in ("record_shapes", "with_stack", "profile_memory", "acc_events")
    )
    assert c.profiler_settings(replace(options, mode="memory"))["profile_memory"]
    assert not c.profiler_settings(replace(options, mode="benchmark"))["enabled"]
    with pytest.raises(ValueError):
        replace(options, run_id="../overwrite").validate()


def test_allocation_and_communication_metrics():
    raw = raw_trace()
    raw["traceEvents"].extend(
        [
            dict(ph="X", cat="cuda_runtime", name="cudaMalloc", ts=7, dur=9),
            dict(ph="X", cat="cuda_driver", name="cuMemAlloc", ts=8, dur=6),
            dict(ph="X", cat="kernel", name="ncclSend", ts=12, dur=10, args={"External id": 1}),
        ]
    )
    trace = compact_profiler_trace(
        raw,
        rank=0,
        mode="baseline",
        transport="p2p",
        measured_steps={"5": dict(cuda_elapsed_ms=0.08)},
        config=dict(pp=2, num_stages=2, num_microbatches=1),
    )
    forward = trace["records"][0]
    assert forward["gpu_activity_union_us"] == 15
    assert forward["communication_union_us"] == 10
    assert forward["cuda_malloc_count"] == 2
    assert forward["cuda_malloc_cpu_union_us"] == 9
    assert forward["cuda_malloc_cpu_sum_us"] == 15


def test_backward_engine_thread_correlation_and_cross_process_exclusion():
    raw = raw_trace()
    for event in raw["traceEvents"]:
        event.update(pid=10, tid=1)
        if event["cat"] == "cpu_op" and event.get("args", {}).get("External id") == 2:
            event["tid"] = 2
    raw["traceEvents"].extend(
        [
            dict(
                ph="X",
                cat="cpu_op",
                name="autograd::engine::evaluate_function: Backward",
                ts=40,
                dur=25,
                pid=10,
                tid=2,
            ),
            dict(
                ph="X",
                cat="cuda_runtime",
                name="cudaLaunchKernel",
                ts=42,
                dur=1,
                pid=10,
                tid=2,
                args={"correlation": 12},
            ),
            dict(
                ph="X",
                cat="kernel",
                name="engine-kernel",
                ts=53,
                dur=8,
                args={"correlation": 12, "stream": 9},
            ),
            dict(ph="X", cat="cuda_runtime", name="cudaMalloc", ts=43, dur=2, pid=10, tid=2),
            dict(
                ph="X",
                cat="cpu_op",
                name="other-process",
                ts=44,
                dur=1,
                pid=20,
                tid=1,
                args={"External id": 50},
            ),
            dict(
                ph="X",
                cat="kernel",
                name="other-process-kernel",
                ts=54,
                dur=50,
                args={"External id": 50},
            ),
            dict(
                ph="X",
                cat="cpu_op",
                name="unrelated-thread",
                ts=44,
                dur=1,
                pid=10,
                tid=3,
                args={"External id": 51},
            ),
            dict(
                ph="X",
                cat="kernel",
                name="unrelated-kernel",
                ts=54,
                dur=2,
                args={"External id": 51},
            ),
        ]
    )
    trace = compact_profiler_trace(
        raw,
        rank=0,
        mode="baseline",
        transport="p2p",
        measured_steps={"5": dict(cuda_elapsed_ms=0.08)},
        config=dict(pp=2, num_stages=2, num_microbatches=1),
    )
    backward = next(r for r in trace["records"] if r["kind"] == "B")
    assert backward["gpu_activity_count"] == 2
    assert backward["gpu_activity_union_us"] == 15
    assert backward["cuda_malloc_count"] == trace["steps"][0]["cuda_malloc_count"] == 1
    assert trace["cuda_allocation_calls"][0]["iteration"] == 5
    assert trace["steps"][0]["end_us"] == 85
    assert trace["unassigned_gpu_activity_count"] == 1


def test_run_variability_is_separate_from_iteration_variability():
    summary = summarize_samples([[1, 3], [10, 10]])
    assert summary["run_means_ms"] == [2, 10]
    assert summary["within_run_stddev_ms"] == [1, 0]
    assert summary["all_samples_by_run"] == [[1, 3], [10, 10]]
    assert summary["rep_mean_stddev_ms"] != summary["stddev_ms"]
    with pytest.raises(ValueError):
        summarize_samples([[1], []])


@pytest.mark.parametrize("mode", ["benchmark", "timeline", "memory"])
def test_collector_preserves_cycles_and_does_not_profile_benchmark(tmp_path, monkeypatch, mode):
    options = c.CollectionOptions(mode=mode, warmup=20, iterations=5, run_id="test-run")
    calls, profiles, history = [], [], []
    tick = [0.0]

    class Event:
        def __init__(self, **kwargs):
            self.timestamp = 0

        def record(self):
            self.timestamp = tick[0]

        def elapsed_time(self, other):
            return other.timestamp - self.timestamp

    def make_raw(iterations):
        raw = raw_trace()
        raw["traceEvents"] = []
        for i in iterations:
            for event in copy.deepcopy(raw_trace()["traceEvents"]):
                event["name"] = event["name"].replace("/i5/", f"/i{i}/")
                event["ts"] += i * 200
                if "args" in event:
                    event["args"]["External id"] += i * 100
                raw["traceEvents"].append(event)
        return raw

    class Profiler:
        def __init__(self, **kwargs):
            profiles.append(kwargs)
            self.callback = kwargs["on_trace_ready"]
            self.position, self.cycle = 0, 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def step(self):
            self.position += 1
            if self.position % 7 == 0:
                self.callback(self)
                self.cycle += 1

        def export_chrome_trace(self, path):
            Path(path).write_text(json.dumps(make_raw(c.capture_windows(options)[self.cycle])))

    monkeypatch.setattr(c.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(c.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(c.dist, "broadcast_object_list", lambda *a, **k: None)
    monkeypatch.setattr(
        c.dist, "all_gather_object", lambda result, value: result.__setitem__(0, value)
    )
    monkeypatch.setattr(c.dist, "barrier", lambda: None)
    monkeypatch.setattr(c, "environment_metadata", lambda: {})
    monkeypatch.setattr(c, "memory_sample", lambda: dict(allocated_bytes=1, reserved_bytes=2))
    monkeypatch.setattr(c, "label_logical_operations", lambda *a: nullcontext())
    monkeypatch.setattr(c.torch.cuda, "Event", Event)
    monkeypatch.setattr(c.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(c.torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(
        c.torch.cuda.memory, "_record_memory_history", lambda **k: history.append(k)
    )
    monkeypatch.setattr(
        c.torch.cuda.memory, "_dump_snapshot", lambda path: Path(path).write_bytes(b"test")
    )
    monkeypatch.setattr(c.torch.profiler, "profile", Profiler)
    monkeypatch.setattr(c.torch.profiler, "record_function", lambda *a: nullcontext())

    def step(i):
        calls.append(i)
        tick[0] += 0.08

    result = c.collect_steps(
        step=step,
        optimizer=None,
        options=options,
        output=tmp_path,
        method="baseline",
        transport="p2p",
        config=dict(pp=1, num_stages=1, num_microbatches=1),
    )
    assert calls == list(range(options.warmup + options.steps))
    assert len(result["samples"]) == options.steps
    assert len(result["warmup_samples"]) == 20
    assert result["summary"]["mean_ms"] == pytest.approx(0.08)
    assert c.summarize_rank_samples([result])["mean_ms"] == pytest.approx(0.08)
    if mode == "benchmark":
        assert not profiles and not result["captures"]
    else:
        assert len(list(tmp_path.rglob("*.torch.json"))) == 3
        assert len(list(tmp_path.rglob("*.compact.json"))) == 3
        assert [x["active_iterations"] for x in result["captures"]] == c.capture_windows(options)
    assert bool(history) == (mode == "memory")
    if history:
        assert history[-1] == dict(enabled=None)
    with pytest.raises(FileExistsError):
        c.collect_steps(
            step=step,
            optimizer=None,
            options=options,
            output=tmp_path,
            method="baseline",
            transport="p2p",
            config=dict(pp=1, num_stages=1, num_microbatches=1),
        )


def test_sample_summaries_preserve_slow_iterations_and_reject_truncation():
    assert c.sample_summary([1, 1, 100])["mean_ms"] == 34
    result = dict(
        rank=0,
        run_id="run",
        method="A",
        options={},
        config=dict(pp=2),
        samples=[dict(iteration=20, cuda_elapsed_ms=1)],
        continuous_wall_ms=2,
    )
    other = dict(result, rank=1, samples=[])
    with pytest.raises(ValueError, match="alignment"):
        c.summarize_rank_samples([result, other])


def test_mixed_capture_rank_selection_rejected(tmp_path):
    from tests.unit_tests.pipeline_parallel.test_slackpipe_figure_trace import compact

    for rank in range(2):
        trace = compact(rank)
        trace["collection"] = dict(run_id="run", capture_id=f"capture{rank}", cycle=rank)
        (tmp_path / f"rank{rank}_trace.json").write_text(json.dumps(trace))
    with pytest.raises(ValueError, match="Mixed"):
        load_panel(tmp_path, 5)
    assert available_iterations(tmp_path) == set()
    for rank in range(2):
        path = tmp_path / f"rank{rank}_trace.json"
        trace = json.loads(path.read_text())
        trace["collection"] = dict(run_id="run", capture_id="capture1", cycle=1)
        path.write_text(json.dumps(trace))
    assert available_iterations(tmp_path, cycle=1) == {5}
    assert available_iterations(tmp_path, cycle=0) == set()
