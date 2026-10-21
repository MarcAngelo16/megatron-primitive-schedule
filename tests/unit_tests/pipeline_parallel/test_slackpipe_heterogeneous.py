# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Real heterogeneous decoder construction and strict numerical regressions."""

import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs import (
    get_gpt_heterogeneous_layer_spec,
)
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.pipeline_parallel.slackpipe.cost_profile import profile_fingerprint
from megatron.core.pipeline_parallel.slackpipe.manifest import (
    build_model_manifest,
    validate_chunk_layers,
    validate_plan_model,
)
from megatron.core.pipeline_parallel.slackpipe.plan import (
    derive_pipeline_model_parallel_layout,
    load_slackpipe_plan,
    parse_slackpipe_plan,
)
from megatron.core.pipeline_parallel.slackpipe.schedule import (
    clear_slackpipe_runtime_cache,
    slackpipe_transport_statistics,
)
from megatron.core.transformer.heterogeneous.heterogeneous_config import (
    HeterogeneousTransformerConfig,
)
from tests.unit_tests.pipeline_parallel.test_slackpipe_model_construction import (
    _assert_parameters_and_gradients_finite,
    _batch_iterator,
    _build_model,
    _copy_parameters,
    _forward_step_func,
    _logical_named_parameters,
    _max_abs_gradient_diff,
    _max_abs_parameter_diff,
)
from tests.unit_tests.test_utilities import Utils, clear_nvte_env_vars


def block_config_payload(hidden_size=128, num_attention_heads=4):
    """The sole synthetic architecture definition; manifests come from the config."""
    classes = {
        "A": {
            "attention": {"num_query_groups": num_attention_heads},
            "mlp": {"ffn_hidden_size": 4 * hidden_size},
        },
        "B": {
            "attention": {"num_query_groups": num_attention_heads},
            "mlp": {"ffn_hidden_size": 16 * hidden_size},
        },
        "C": {
            "attention": {"num_query_groups": None, "no_op": True},
            "mlp": {"ffn_hidden_size": 2 * hidden_size},
        },
    }
    return {"block_configs": [classes[name] for name in "AABCABCABCAA"]}


def make_config(
    pp=1, vpp=None, layout=None, schedule="default", hidden_size=128, num_attention_heads=4
):
    return HeterogeneousTransformerConfig(
        num_layers=12,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        heterogeneous_layers_config_encoded_json=json.dumps(
            block_config_payload(hidden_size, num_attention_heads)
        ),
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
        pipeline_model_parallel_size=pp,
        virtual_pipeline_model_parallel_size=vpp,
        pipeline_model_parallel_layout=layout,
        pipeline_schedule=schedule,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        batch_p2p_comm=True,
        overlap_p2p_comm=False,
    )


def provider(pre_process=True, post_process=True, vp_stage=None, config=None, pg_collection=None):
    return GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_heterogeneous_layer_spec(
            config, use_te=False, vp_stage=vp_stage
        ),
        vocab_size=128,
        max_sequence_length=64,
        pre_process=pre_process,
        post_process=post_process,
        position_embedding_type="rope",
        vp_stage=vp_stage,
        pg_collection=pg_collection,
        share_embeddings_and_output_weights=False,
    )


def hand_plan(pp=1, cuts=(0, 2, 5, 9, 12), microbatches=4):
    operations = [[] for _ in range(pp)]
    # All forwards then all backwards, FIFO within each logical direction.
    for kind, stages in (("F", range(4)), ("B", range(3, -1, -1))):
        for b in range(microbatches):
            for s in stages:
                operations[s % pp].append({"kind": kind, "microbatch": b, "stage": s})
    return dict(
        schema_version="slackpipe.plan.v2",
        num_microbatches=microbatches,
        num_stages=4,
        num_workers=pp,
        num_layers=12,
        layer_cuts=list(cuts),
        stage_to_worker=[s % pp for s in range(4)],
        operations=operations,
        model_manifest_hash=build_model_manifest(make_config())["manifest_hash"],
    )


def test_manifest_and_provenance(tmp_path):
    config = make_config()
    manifest = build_model_manifest(config)
    assert [r["config_class"] for r in manifest["layers"]] == [
        f"class_{'ABC'.index(c)}" for c in "AABCABCABCAA"
    ]
    assert [r["block_config"] for r in manifest["layers"]] == [
        asdict(b) for b in config.per_block_parameters
    ]
    assert [r["layer_id"] for r in manifest["layers"]] == list(range(12))
    plan = parse_slackpipe_plan(hand_plan())
    validate_plan_model(plan, config)
    with pytest.raises(ValueError, match="model_manifest_hash"):
        validate_plan_model(replace(plan, model_manifest_hash="wrong"), config)
    changed = make_config()
    changed.per_block_parameters[4].mlp.ffn_hidden_size = 1024
    with pytest.raises(ValueError, match="model_manifest_hash"):
        validate_plan_model(plan, changed)
    profile = {
        "schema_version": "slackpipe.cost_profile.v2",
        "model_manifest_hash": manifest["manifest_hash"],
    }
    profile["cost_profile_hash"] = profile_fingerprint(profile)
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    linked = replace(
        plan,
        cost_profile_hash=profile["cost_profile_hash"],
        cost_profile_version=profile["schema_version"],
        cost_profile_path=str(path),
    )
    validate_plan_model(linked, config)
    raw = hand_plan()
    raw.update(
        cost_profile_hash=profile["cost_profile_hash"],
        cost_profile_version=profile["schema_version"],
        cost_model={"path": "profile.json"},
    )
    plan_path = tmp_path / "linked.plan.json"
    plan_path.write_text(json.dumps(raw))
    loaded = load_slackpipe_plan(plan_path)
    assert loaded.cost_profile_path == str(path)
    validate_plan_model(loaded, config)
    with pytest.raises(ValueError, match="requires cost_model.path"):
        validate_plan_model(replace(linked, cost_profile_path=None), config)
    with pytest.raises(ValueError, match="version"):
        validate_plan_model(replace(linked, cost_profile_version="wrong"), config)
    profile["modified"] = True
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="cost_profile_hash"):
        validate_plan_model(linked, config)


def test_calibration_partitions_are_identifiable():

    cuts, diagnostics = select_partitions(build_model_manifest(make_config()))
    assert len(cuts) >= 6
    assert len(set(cuts)) == len(cuts)
    assert diagnostics["rank"] == diagnostics["columns"] == 6
    assert diagnostics["condition"] < 100


@pytest.mark.parametrize("hidden,heads", [(128, 4), (256, 8), (512, 16)])
def test_scaled_heterogeneous_manifest(hidden, heads):
    config = make_config(hidden_size=hidden, num_attention_heads=heads)
    manifest = build_model_manifest(config)
    assert manifest["model_config"]["hidden_size"] == hidden
    assert manifest["model_config"]["num_attention_heads"] == heads
    assert [row["config_class"] for row in manifest["layers"]] == [
        f"class_{'ABC'.index(c)}" for c in "AABCABCABCAA"
    ]
    payload = block_config_payload(hidden, heads)["block_configs"]
    assert [payload[i]["mlp"]["ffn_hidden_size"] for i in (0, 2, 3)] == [
        4 * hidden,
        16 * hidden,
        2 * hidden,
    ]
    assert (manifest["manifest_hash"] == build_model_manifest(make_config())["manifest_hash"]) == (
        hidden == 128
    )


def test_heterogeneous_pp1_exact_construction(tmp_path):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("PP1 construction requires one rank")
    Utils.initialize_model_parallel(1, 1)
    parallel_state.set_virtual_pipeline_model_parallel_world_size(4)
    plan = parse_slackpipe_plan(hand_plan())
    layout = derive_pipeline_model_parallel_layout(plan)
    try:
        chunks = _build_model(
            make_config(1, 4, layout, "slackpipe"),
            pipeline_schedule="slackpipe",
            vpp=4,
            layout=layout,
            provider=provider,
        )
        rows = validate_chunk_layers(plan, chunks, 0)
        assert [r["layer_id"] for r in rows] == list(range(12))
        assert [len(c.decoder.layers) for c in chunks] == [2, 3, 4, 3]
        assert [c.pre_process for c in chunks] == [True, False, False, False]
        assert [c.post_process for c in chunks] == [False, False, False, True]
        chunks[1].decoder.layers[0].config.ffn_hidden_size += 1
        with pytest.raises(ValueError, match="MLP configuration"):
            validate_chunk_layers(plan, chunks, 0)
        chunks[1].decoder.layers[0].config.ffn_hidden_size -= 1
        bad = hand_plan()
        bad["model_manifest_hash"] = "wrong"
        path = tmp_path / "wrong.plan.json"
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="model_manifest_hash"):
            get_forward_backward_func(pipeline_schedule="slackpipe", slackpipe_plan_path=str(path))(
                forward_step_func=lambda *a: pytest.fail("model execution before hash validation"),
                data_iterator=[iter(())] * 4,
                model=chunks,
                num_microbatches=4,
                seq_length=8,
                micro_batch_size=1,
                forward_only=False,
            )
    finally:
        clear_slackpipe_runtime_cache()
        Utils.destroy_model_parallel()
        parallel_state.set_virtual_pipeline_model_parallel_world_size(None)


@pytest.mark.parametrize("pp", [1, 2])
def test_heterogeneous_numerical_equivalence(tmp_path, pp):
    # Single-node fixture: TP=DP=CP=1, one process/device per physical PP rank.
    Utils.require_cuda_devices(pp)
    if int(os.environ.get("WORLD_SIZE", "1")) != pp:
        pytest.skip(f"requires {pp} ranks")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    clear_nvte_env_vars()
    external = os.environ.get("SLACKPIPE_HETERO_PLAN") if pp == 2 else None
    path = Path(external) if external else tmp_path / "hand.plan.json"
    if not external:
        path.write_text(json.dumps(hand_plan(pp)))
    plan = load_slackpipe_plan(path, pipeline_model_parallel_size=pp)
    Utils.initialize_model_parallel(1, 1)
    baseline = _build_model(make_config(), provider=provider)
    params = _logical_named_parameters(baseline)
    from tests.unit_tests.pipeline_parallel.slackpipe_perf_benchmark import _make_batches

    batches = _make_batches(
        SimpleNamespace(
            seed=1234,
            num_microbatches=plan.num_microbatches,
            vocab_size=128,
            micro_batch_size=1,
            seq_length=64,
        )
    )
    opt_ref = torch.optim.SGD(params.values(), lr=0.01)
    losses_ref = []
    for b in batches:
        out = baseline[0](b["tokens"], b["position_ids"], None, labels=b["labels"])
        loss = out.float().mean() / plan.num_microbatches
        losses_ref.append(loss.detach())
        loss.backward()
    Utils.initialize_model_parallel(
        1, pp, virtual_pipeline_model_parallel_size=2 if pp == 2 else None
    )
    parallel_state.set_virtual_pipeline_model_parallel_world_size(4 // pp)
    layout = derive_pipeline_model_parallel_layout(plan)
    try:
        chunks = _build_model(
            make_config(pp, 4 // pp, layout, "slackpipe"),
            pipeline_schedule="slackpipe",
            pp_size=pp,
            vpp=4 // pp,
            layout=layout,
            provider=provider,
        )
        rank = parallel_state.get_pipeline_model_parallel_rank()
        rows = validate_chunk_layers(plan, chunks, rank)
        actual = _logical_named_parameters(chunks)
        names = [None] * pp
        dist.all_gather_object(names, list(actual))
        flat = [name for ns in names for name in ns]
        assert len(flat) == len(set(flat)) and set(flat) == set(params)
        _copy_parameters(params, actual)
        expected = {k: params[k] for k in actual}
        differences = {"initial": _max_abs_parameter_diff(expected, actual)}
        optimizer = torch.optim.SGD(actual.values(), lr=0.01)
        root = Path(os.environ.get("SLACKPIPE_HETERO_ARTIFACTS", tmp_path)) / f"pp{pp}"
        root.mkdir(parents=True, exist_ok=True)
        losses = get_forward_backward_func(
            pipeline_schedule="slackpipe",
            slackpipe_plan_path=str(path),
            slackpipe_trace_path=str(root / "trace.json"),
            slackpipe_transport=(
                os.environ.get("SLACKPIPE_TEST_TRANSPORT", "nccl-rma") if pp == 2 else "nccl-p2p"
            ),
        )(
            forward_step_func=_forward_step_func,
            data_iterator=[_batch_iterator(batches) for _ in chunks],
            model=chunks,
            num_microbatches=plan.num_microbatches,
            seq_length=64,
            micro_batch_size=1,
            forward_only=False,
        )
        assert len(losses) == (plan.num_microbatches if rank == plan.stage_to_worker[-1] else 0)
        transport = slackpipe_transport_statistics()
        assert all(r["outstanding_puts"] == 0 and not r["active_iteration"] for r in transport)
        (root / f"transport.rank{rank}.json").write_text(json.dumps(transport, indent=2))
        differences["loss"] = max(
            [(a["loss"] - b).abs().item() for a, b in zip(losses, losses_ref)] or [0.0]
        )
        _assert_parameters_and_gradients_finite(actual)
        differences["gradient"] = _max_abs_gradient_diff(expected, actual)
        optimizer.step()
        opt_ref.step()
        differences["post_step"] = _max_abs_parameter_diff(expected, actual)
        for key in differences:
            t = torch.tensor(differences[key], device="cuda")
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            differences[key] = t.item()
        trace = json.loads((root / f"trace.rank{rank}.json").read_text())
        assert trace["matched_plan"]
        assert trace["operations"] == [asdict(op) for op in plan.worker_operations(rank)]
        (root / f"layers.rank{rank}.json").write_text(json.dumps(rows, indent=2))
        if rank == 0:
            (root / "differences.json").write_text(json.dumps(differences, indent=2))
            print(f"Heterogeneous PP{pp} max absolute differences: {differences}")
        assert all(v == 0.0 for v in differences.values())
    finally:
        clear_slackpipe_runtime_cache()
        Utils.destroy_model_parallel()
        parallel_state.set_virtual_pipeline_model_parallel_world_size(None)


def select_partitions(manifest, minimum=6):
    """Add partitions until all class and role coefficients are identifiable."""
    import itertools

    import numpy as np

    from megatron.core.pipeline_parallel.slackpipe.cost_profile import stage_role

    classes = [r["config_class"] for r in manifest["layers"]]
    keys = sorted(set(classes))

    def design(cuts):
        return np.array(
            [
                [classes[a:b].count(k) for k in keys]
                + [int(stage_role(s, 4) == r) for r in ("first", "middle", "last")]
                for s, (a, b) in enumerate(zip(cuts, cuts[1:]))
            ],
            dtype=float,
        )

    selected = [(0, 2, 5, 9, 12)]
    candidates = [(0, *inner, 12) for inner in itertools.combinations(range(1, 12), 3)]
    columns = len(keys) + 3
    while True:
        matrix = np.concatenate([design(c) for c in selected])
        rank = np.linalg.matrix_rank(matrix)
        if len(selected) >= minimum and rank == columns:
            return selected, {
                "rank": int(rank),
                "columns": columns,
                "condition": float(np.linalg.cond(matrix)),
            }
        remaining = [c for c in candidates if c not in selected]
        if not remaining:
            raise ValueError("no identifiable calibration design")

        def score(c):
            augmented = np.concatenate([matrix, design(c)])
            return (-np.linalg.matrix_rank(augmented), np.linalg.cond(augmented), c)

        selected.append(min(remaining, key=score))
