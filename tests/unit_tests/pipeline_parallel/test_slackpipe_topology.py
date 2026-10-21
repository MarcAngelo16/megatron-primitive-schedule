# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Four-worker structure, without four-rank or CUDA execution."""

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from megatron.core.pipeline_parallel.slackpipe import communication, communication_rma, schedule
from megatron.core.pipeline_parallel.slackpipe.cost_profile import build_heterogeneous_cost_profile
from megatron.core.pipeline_parallel.slackpipe.hybrid import (
    NEMOTRON_H_8B_PATTERN,
    nemotron_h_8b_config,
    partition_hybrid_pattern,
    stage_class_counts,
    stage_layer_types,
)
from megatron.core.pipeline_parallel.slackpipe.manifest import build_model_manifest
from megatron.core.pipeline_parallel.slackpipe.plan import (
    parse_slackpipe_plan,
    validate_plan_parallel_layout,
)
from megatron.core.pipeline_parallel.slackpipe.topology import local_chunk, logical_edges
from megatron.training.arguments import _configure_slackpipe_plan_args
from tests.unit_tests.pipeline_parallel.test_slackpipe_hybrid import (
    PATTERN,
    hybrid_plan,
    tiny_config,
)


def synthetic_hybrid_profile():
    manifest = build_model_manifest(tiny_config())
    # Synthetic stage observations, not hardware measurements. Vary lengths and
    # class mixtures so class columns and role intercepts are identifiable.
    rows = []
    for begin in range(len(PATTERN)):
        for end in range(begin + 1, len(PATTERN) + 1):
            role = "first" if begin == 0 else "last" if end == len(PATTERN) else "middle"
            costs = {"M": 0.03, "*": 0.02, "-": 0.01}
            cost = sum(costs[s] for s in PATTERN[begin:end]) + 0.005
            rows.append(
                dict(
                    begin=begin,
                    end=end,
                    stage_role=role,
                    forward_ms_per_op=cost,
                    backward_ms_per_op=3 * cost,
                )
            )
    return build_heterogeneous_cost_profile(
        model_manifest=manifest,
        observed_stage_rows=rows,
        model_config=manifest["model_config"],
        parallel_config={"pipeline_model_parallel_size": 4},
    )


def test_pp4_mapping_and_ranges(monkeypatch):
    plan = parse_slackpipe_plan(hybrid_plan(), pipeline_model_parallel_size=4)
    assert validate_plan_parallel_layout(plan, 4) == 2
    assert plan.stage_to_worker == (0, 1, 2, 3, 0, 1, 2, 3)
    assert [local_chunk(plan, s) for s in range(8)] == [0, 0, 0, 0, 1, 1, 1, 1]
    monkeypatch.setattr(
        schedule.parallel_state, "get_pipeline_model_parallel_world_size", lambda: 1
    )
    manifest = build_model_manifest(tiny_config())
    for rank in range(4):
        runtime = schedule._SlackPipeRuntime(
            plan=plan,
            pp_rank=rank,
            pipeline_tensor_shape=(64, 1, 128),
            pipeline_tensor_dtype=torch.float32,
            pipeline_tensor_device=torch.device("cpu"),
            forward_only=False,
            enable_fast_path=True,
        )
        assert runtime.local_stages == (rank, rank + 4)
        assert len(runtime.expected_operations) == 32
        for op in runtime.compiled_operations:
            s = op.operation.stage
            assert op.local_index == s // 4
            assert op.previous_worker == ((s - 1) % 4 if s else None)
            assert op.next_worker == ((s + 1) % 4 if s < 7 else None)
            assert op.previous_edge == ((s - 1, s) if s else None)
            assert op.next_edge == ((s, s + 1) if s < 7 else None)
            assert op.operation in plan.worker_operations(rank)
    assert partition_hybrid_pattern(PATTERN, plan).split("|") == [
        PATTERN[b:e] for b, e in plan.stage_layer_ranges
    ]
    for s in range(8):
        assert len(stage_layer_types(plan, manifest, s)) == plan.layer_split[s]
        assert sum(stage_class_counts(plan, manifest, s).values()) == plan.layer_split[s]
    with pytest.raises(ValueError, match="match"):
        partition_hybrid_pattern("M|" + PATTERN[1:], plan)
    with pytest.raises(ValueError):
        validate_plan_parallel_layout(replace(plan, num_workers=3), 3)
    with pytest.raises(ValueError):
        parse_slackpipe_plan(hybrid_plan(), pipeline_model_parallel_size=2)


@pytest.mark.parametrize("rank", range(4))
def test_pp4_edge_creation_and_mailbox_spec(monkeypatch, rank):
    plan = parse_slackpipe_plan(hybrid_plan())
    edges = logical_edges(plan)
    assert [e.workers for e in edges] == [(0, 1), (1, 2), (2, 3), (3, 0), (0, 1), (1, 2), (2, 3)]
    channels = [e.channel(d) for e in edges for d in ("forward", "backward")]
    assert len(channels) == len(set(channels)) == 14
    assert [sum(r in channel[2:] for channel in channels) for r in range(4)] == [6, 8, 8, 6]
    for e in edges:
        for d in ("forward", "backward"):
            _, _, source, destination = e.channel(d)
            assert e.ranks[e.ranks.index(source)] == source
            assert e.ranks[e.ranks.index(destination)] == destination
    created, destroyed = [], []
    monkeypatch.setattr(communication.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(communication.dist, "get_rank", lambda: rank)
    monkeypatch.setattr(communication.dist, "get_world_size", lambda: 4)

    def new_group(ranks, backend):
        created.append(tuple(ranks))
        return object()

    monkeypatch.setattr(communication.dist, "new_group", new_group)
    monkeypatch.setattr(communication.dist, "destroy_process_group", destroyed.append)
    monkeypatch.setattr(
        communication.SlackPipeCommunicator, "_warm_up_edge_groups", lambda self: None
    )
    comm = communication.SlackPipeCommunicator(plan)
    assert created == [e.ranks for e in edges]
    assert list(comm.edge_groups) == [e.stages for e in edges if rank in e.ranks]
    assert len(set(comm.edge_groups.values())) == len(comm.edge_groups)
    comm.close()
    assert len(destroyed) == sum(rank in e.ranks for e in edges)


def test_pp4_cache_separation(monkeypatch):
    class Fake:
        def __init__(self, plan):
            pass

        def assert_no_outstanding_work(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(communication_rma, "SlackPipeRMACommunicator", Fake)
    monkeypatch.setattr(schedule, "_distributed_context", lambda: (None, None))
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 4)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    def get(plan, device=0, shape=(64, 1, 128), dtype=torch.float32):
        return schedule._get_rma_transport(plan, shape, dtype, torch.device("cuda", device))

    plan = parse_slackpipe_plan(hybrid_plan())
    try:
        a = get(plan)
        assert get(plan) is a
        for p in (replace(plan, num_microbatches=4), parse_slackpipe_plan(hybrid_plan(2))):
            assert get(p) is not a
        assert get(plan, device=1) is not a
        assert get(plan, shape=(32, 1, 128)) is not a
        assert get(plan, dtype=torch.bfloat16) is not a
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
        assert get(plan) is not a
    finally:
        schedule.shutdown_slackpipe_runtime()


def test_hybrid_args_manifest_and_prefix_costs(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(hybrid_plan()))
    args = SimpleNamespace(
        pipeline_schedule="slackpipe",
        slackpipe_plan=str(path),
        num_layers=12,
        pipeline_model_parallel_size=4,
        pipeline_model_parallel_layout=None,
        hybrid_layer_pattern=PATTERN,
    )
    _configure_slackpipe_plan_args(args)
    assert args.virtual_pipeline_model_parallel_size == 2
    assert args.pipeline_model_parallel_layout is None
    assert len(args.hybrid_layer_pattern.split("|")) == 8
    profile = synthetic_hybrid_profile()
    assert len(profile["layer_costs_us"]) == 12
    for direction in ("forward", "backward"):
        prefix = profile[f"prefix_{direction}_us"]
        assert len(prefix) == 13
        for i, row in enumerate(profile["layer_costs_us"]):
            assert prefix[i + 1] - prefix[i] == pytest.approx(row[f"{direction}_us"])
    config = nemotron_h_8b_config()
    manifest = build_model_manifest(config)
    assert config.num_layers == len(NEMOTRON_H_8B_PATTERN) == 52
    assert (
        hashlib.sha256(NEMOTRON_H_8B_PATTERN.encode()).hexdigest()
        == "e0cf75d03b16c79cac23da4c217a5d3243d45043f07cb445ab358dad42fc4cf5"
    )
    assert (
        config.hidden_size,
        config.ffn_hidden_size,
        config.num_attention_heads,
        config.num_query_groups,
    ) == (4096, 21504, 32, 8)
    assert [l["global_layer_id"] for l in manifest["layers"]] == list(range(52))
    assert {l["layer_type"] for l in manifest["layers"]} == {"mamba", "attention", "mlp"}


@pytest.mark.parametrize("microbatches", [1, 4, 8, 16])
@pytest.mark.parametrize("stages", [4, 8, 12])
def test_generic_microbatches_and_chunks(microbatches, stages, monkeypatch):
    payload = hybrid_plan()
    payload.update(
        num_microbatches=microbatches,
        num_stages=stages,
        layer_cuts=[12 * s // stages for s in range(stages + 1)],
        stage_to_worker=[s % 4 for s in range(stages)],
    )
    payload["operations"] = [[] for _ in range(4)]
    for kind, order in (("F", range(stages)), ("B", reversed(range(stages)))):
        order = list(order)
        for b in range(microbatches):
            for s in order:
                payload["operations"][s % 4].append(dict(kind=kind, microbatch=b, stage=s))
    plan = parse_slackpipe_plan(payload, pipeline_model_parallel_size=4)
    assert validate_plan_parallel_layout(plan, 4) == stages // 4
    assert sum(len(plan.worker_operations(r)) for r in range(4)) == 2 * microbatches * stages
    monkeypatch.setattr(
        schedule.parallel_state, "get_pipeline_model_parallel_world_size", lambda: 1
    )
    for rank in range(4):
        runtime = schedule._SlackPipeRuntime(
            plan=plan,
            pp_rank=rank,
            pipeline_tensor_shape=(64, 1, 128),
            pipeline_tensor_dtype=torch.bfloat16,
            pipeline_tensor_device=torch.device("cpu"),
            forward_only=False,
            enable_fast_path=True,
        )
        assert runtime.local_stages == tuple(range(rank, stages, 4))
        assert len(runtime.compiled_operations) == 2 * microbatches * stages // 4
        for op in runtime.compiled_operations:
            assert op.operation.stage % 4 == rank
            assert op.local_index == op.operation.stage // 4
