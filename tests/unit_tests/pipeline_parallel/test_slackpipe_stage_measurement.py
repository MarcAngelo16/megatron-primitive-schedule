# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from megatron.core.pipeline_parallel.slackpipe.cost_profile import profile_fingerprint
from megatron.core.pipeline_parallel.slackpipe.stage_measurement import (
    LayerGraphLabels,
    StageEventCollector,
    graph_nodes,
)
from tools.slackpipe_stage_calibration import (
    features,
    fit_runs,
    make_design,
    read_design,
    validate_samples,
)


def manifest(length=24):
    return dict(
        manifest_hash="test-only",
        layers=[
            dict(layer_id=i, config_class=("mamba", "mlp", "attention")[i % 3])
            for i in range(length)
        ],
    )


def test_window_deferred_no_operation_or_step_sync(monkeypatch):
    sync = Mock()
    monkeypatch.setattr(
        torch.cuda, "Event", lambda **k: Mock(query=lambda: True, elapsed_time=lambda end: 2.5)
    )
    monkeypatch.setattr(torch.cuda, "synchronize", sync)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=12))
    collector = StageEventCollector(6)
    assert sync.call_count == 1
    for i in range(2):
        collector.begin(i)
        for phase in ("forward_compute", "backward_compute"):
            assert (
                collector.compute(lambda: 42, dict(phase=phase, microbatch=0, logical_stage=0))
                == 42
            )
        collector.optimizer(lambda: None)
        assert collector.finish() == []
    assert sync.call_count == 1
    rows = collector.resolve()
    assert [r["elapsed_ms"] for r in rows] == [2.5] * 6
    assert all(r["dependency_wait_ms"] is None for r in rows)
    assert [r["phase"] for r in rows].count("optimizer") == 2
    collector.begin(2)
    with pytest.raises(RuntimeError, match="exhausted"):
        collector.compute(lambda: None, dict(phase="forward_compute"))


def test_failed_pair_cannot_be_fitted(monkeypatch):
    monkeypatch.setattr(torch.cuda, "Event", lambda **k: Mock(query=lambda: True))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=12))
    c = StageEventCollector(1)
    c.begin(0)
    with pytest.raises(ZeroDivisionError):
        c.compute(lambda: 1 / 0, dict(phase="forward_compute"))
    c.finish()
    with pytest.raises(RuntimeError, match="failed"):
        c.resolve()


def test_design_identifiability_and_no_holdout_leakage(tmp_path):
    m = manifest(80)
    d = make_design(m, 8, [[10] * 8], 8)
    assert d["rank"] == 5 and d["condition"] > 1
    assert {2, 3, 4, 6, 17, 18, 19, 20} <= set(d["training_lengths"])
    assert len({tuple(p["cuts"]) for p in d["partitions"]}) == len(d["partitions"])
    path = tmp_path / "design.json"
    path.write_text(json.dumps(d))
    assert read_design(path, m, 8) == d
    d["partitions"].append(d["partitions"][0])
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="Duplicate"):
        read_design(path, m, 8)


def synthetic_run(path, m, d, run):
    path.mkdir()
    (path / "calibration_design.json").write_text(json.dumps(d))
    raw = []
    for part in d["partitions"]:
        folder = path / part["id"]
        folder.mkdir()
        for worker in range(2):
            c = dict(pp=2, vpp=2, tp=1, dp=1, cp=1, num_microbatches=2, transport="native")
            metadata = dict(
                mode="events",
                run_id=run,
                partition_id=part["id"],
                rank=worker,
                configuration=c,
                measurement_iterations=list(range(20, 30)),
            )
            (folder / f"result.rank{worker}.json").write_text(json.dumps(metadata))
        x = features(m, part["cuts"])
        for i in range(20, 30):
            position = 0
            for b in range(2):
                for phase, order in (
                    ("forward_compute", range(4)),
                    ("backward_compute", reversed(range(4))),
                ):
                    for s in order:
                        cost = float(x[s] @ np.array([1, 2, 3, 4, 5])) * (
                            2 if phase.startswith("backward") else 1
                        )
                        raw.append(
                            dict(
                                run_id=run,
                                partition_id=part["id"],
                                iteration=i,
                                microbatch=b,
                                logical_stage=s,
                                phase=phase,
                                worker=s % 2,
                                stage_layer_range=part["cuts"][s : s + 2],
                                schedule_position=position,
                                elapsed_ms=cost,
                                status="recorded",
                            )
                        )
                        position += 1
    (path / "stage_samples.json").write_text(json.dumps(raw))
    return raw


def test_fit_units_prefix_hash_and_independent_holdout(tmp_path):
    m = manifest()
    d = make_design(m, 4, train_count=5)
    runs = [tmp_path / "run0", tmp_path / "run1", tmp_path / "run2"]
    for i, p in enumerate(runs):
        synthetic_run(p, m, d, str(i))
    profile, report = fit_runs(runs, m, d)
    assert report["heldout_mape"] < 1e-12
    assert max(abs(r["relative_error"]) for r in report["fixed_worker_order_replay"]) < 1e-12
    assert profile["class_costs_us"]["attention"]["forward"] == pytest.approx(1000)
    assert profile["stage_role_bias_us"]["first"]["forward"] == pytest.approx(4000)
    assert profile["prefix_backward_us"][-1] == pytest.approx(
        sum(r["backward_us"] for r in profile["layer_costs_us"])
    )
    assert profile_fingerprint(profile) == profile["cost_profile_hash"]
    assert not profile["validation"]["clock_aligned_makespan_verified"]
    assert report["diagnostic_only"]
    assert "quality" not in profile and "passed" not in report and "issues" not in report
    # Corrupt only holdout: fitted coefficients must remain unchanged.
    p = runs[2] / "stage_samples.json"
    raw = json.loads(p.read_text())
    for r in raw:
        if r["partition_id"] == "validation0":
            r["elapsed_ms"] *= 2
    p.write_text(json.dumps(raw))
    changed, bad = fit_runs(runs, m, d)
    assert changed["class_costs_us"] == profile["class_costs_us"]
    assert bad["heldout_mape"] > 0.2
    assert "issues" not in bad


def test_fit_accepts_short_variable_samples_without_quality_verdict(tmp_path):
    m = manifest()
    d = make_design(m, 4, train_count=5)
    runs = [tmp_path / "run0", tmp_path / "run1"]
    for i, path in enumerate(runs):
        raw = synthetic_run(path, m, d, str(i))
        for metadata_path in path.glob("*/result.rank*.json"):
            metadata = json.loads(metadata_path.read_text())
            metadata["measurement_iterations"] = [20, 21]
            metadata_path.write_text(json.dumps(metadata))
        for row in raw:
            row["elapsed_ms"] *= (50 if row["iteration"] == 21 else 1) * (i + 1)
        (path / "stage_samples.json").write_text(json.dumps(raw))
    profile, report = fit_runs(runs, m, d)
    assert all(row["statistics"]["count"] == 2 for row in report["rows"])
    assert all(row["raw"]["count"] == 4 for row in report["rows"])
    assert "quality" not in profile and "passed" not in report and "issues" not in report


@pytest.mark.parametrize("damage", ["duplicate", "missing", "range", "failed", "nan"])
def test_invalid_samples_fail_closed(tmp_path, damage):
    m = manifest()
    d = make_design(m, 4, train_count=3)
    raw = synthetic_run(tmp_path / "run", m, d, "r")
    p = d["partitions"][0]
    rows = [r for r in raw if r["partition_id"] == p["id"]]
    if damage == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    if damage == "missing":
        rows.pop()
    if damage == "range":
        rows[0]["stage_layer_range"] = [1, 2]
    if damage == "failed":
        rows[0]["status"] = "failed"
    if damage == "nan":
        rows[0]["elapsed_ms"] = float("nan")
    with pytest.raises(ValueError):
        validate_samples(rows, iterations=range(20, 30), microbatches=2, cuts=p["cuts"], workers=2)


def test_native_node_graph_includes_parameter_and_input_gradients():
    module = torch.nn.Linear(4, 4)
    x = torch.ones(2, 4, requires_grad=True)
    output = module(x)
    nodes = graph_nodes(output)
    variables = {id(n.variable) for n in nodes if hasattr(n, "variable")}
    assert {id(x), id(module.weight), id(module.bias)} <= variables


def test_kwargs_graph_does_not_assign_upstream_embedding_to_layer():
    class Layer(torch.nn.Linear):
        layer_number = 1

        def forward(self, *, hidden_states):
            return super().forward(hidden_states)

    layer = Layer(4, 4)
    chunk = SimpleNamespace(decoder=SimpleNamespace(layers=[layer]), modules=lambda: [layer])
    c = SimpleNamespace(current=dict(iteration=0, microbatch=0, phase="forward_compute"))
    labels = LayerGraphLabels(
        c, [chunk], SimpleNamespace(stage_layer_ranges=[(0, 1)]), manifest(1), 0
    )
    labels.module_handles = len(labels.handles)
    embedding = torch.nn.Embedding(8, 4)
    x = embedding(torch.tensor([1, 2]))
    y = layer(hidden_states=x)
    assert not any("EmbeddingBackward" in name for r in labels.metadata for name in r["nodes"])
    assert labels.metadata[0]["input"]["kwargs"]["hidden_states"]["shape"] == [2, 4]
    c.current = dict(iteration=0, microbatch=0, phase="backward_compute")
    y.sum().backward()
    assert embedding.weight.grad is not None and layer.weight.grad is not None
    labels.clear_graph()
    labels.close()


def test_trace_correlation_separates_wait_communication_and_unknown_stream():
    from tools.slackpipe_stage_trace import correlate

    fields = dict(
        iteration=5,
        microbatch=1,
        stage=2,
        global_layer_id=7,
        layer_type="mamba",
        phase="backward_compute",
        attribution="unique",
    )
    label = dict(
        ph="X",
        cat="user_annotation",
        name="SlackPipeLayer/B/" + json.dumps(fields).encode().hex(),
        ts=0,
        dur=10,
        pid=1,
        tid=1,
    )
    launch = dict(
        ph="X",
        cat="cuda_runtime",
        name="cudaLaunchKernel",
        ts=2,
        dur=1,
        pid=1,
        tid=1,
        args={"correlation": 3},
    )
    kernel = dict(
        ph="X",
        cat="kernel",
        name="anonymous",
        ts=20,
        dur=5,
        pid=0,
        tid=7,
        args={"correlation": 3, "stream": 7},
    )
    other = dict(kernel, name="ncclKernel", ts=21, args={"correlation": 9, "stream": 9})
    wait = dict(launch, name="cudaStreamWaitEvent", args={"correlation": 10})
    unknown = dict(kernel, name="unknown", args={"stream": 11})
    out = correlate(dict(traceEvents=[label, launch, kernel, other, wait, unknown]))
    assert [r["attribution"] for r in out["activities"]] == [
        "unique",
        "shared/unattributed",
        "shared/unattributed",
    ]
    assert out["dependency_wait_gpu_ms"] is None
    assert len(out["cpu_allocations_and_wait_insertions"]) == 1
    assert out["activities"][0]["owners"][0]["global_layer_id"] == 7


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_events_and_node_labels_preserve_gradients_and_optimizer():
    class Layer(torch.nn.Linear):
        layer_number = 1

    torch.manual_seed(7)
    layer = Layer(8, 8).cuda()
    chunk = SimpleNamespace(decoder=SimpleNamespace(layers=[layer]), modules=lambda: [layer])
    plan = SimpleNamespace(stage_layer_ranges=[(0, 1)])
    original = copy.deepcopy(layer.state_dict())
    results = []
    for mode in ("off", "events", "layers"):
        layer.load_state_dict(original)
        layer.zero_grad(set_to_none=True)
        x = torch.ones(4, 8, device="cuda", requires_grad=True)
        opt = torch.optim.SGD(layer.parameters(), lr=0.01)
        c = StageEventCollector(3, timeline=mode == "layers") if mode != "off" else None
        labels = LayerGraphLabels(c, [chunk], plan, manifest(1), 0) if mode == "layers" else None
        if labels:
            labels.module_handles = len(labels.handles)
            c.layer_labels = labels
        if c:
            c.begin(0)
        identity = dict(iteration=0, microbatch=0, logical_stage=0, phase="forward_compute")
        y = c.compute(lambda: layer(x), identity) if c else layer(x)
        identity["phase"] = "backward_compute"
        if c:
            c.compute(lambda: y.sum().backward(), identity)
        else:
            y.sum().backward()
        gradients = [x.grad.clone()] + [p.grad.clone() for p in layer.parameters()]
        if c:
            c.optimizer(opt.step)
            c.finish()
        else:
            opt.step()
        torch.cuda.synchronize()
        if c:
            assert len(c.resolve()) == 3
        if labels:
            assert any("AccumulateGrad" in n for r in labels.metadata for n in r["nodes"])
            labels.close()
        results.append((gradients, [p.detach().clone() for p in layer.parameters()]))
    for result in results[1:]:
        for actual, expected in zip(sum(result, []), sum(results[0], [])):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
