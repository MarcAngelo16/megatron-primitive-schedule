# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Native Mamba/attention/MLP equivalence, with explicit four-device gating."""

import json
import os
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.pipeline_parallel.slackpipe.hybrid import (
    bind_hybrid_config,
    partition_hybrid_pattern,
)
from megatron.core.pipeline_parallel.slackpipe.manifest import (
    build_model_manifest,
    validate_chunk_layers,
)
from megatron.core.pipeline_parallel.slackpipe.plan import load_slackpipe_plan
from megatron.core.pipeline_parallel.slackpipe.schedule import (
    shutdown_slackpipe_runtime,
    slackpipe_transport_statistics,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import unwrap_model
from tests.unit_tests.pipeline_parallel.slackpipe_perf_benchmark import _make_batches
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

PATTERN = "M-*MM-*-M-*-"
CUTS = (0, 1, 3, 4, 6, 7, 9, 10, 12)


def tiny_config(pp=1, vpp=None, schedule="default", dtype=torch.float32):
    return bind_hybrid_config(
        TransformerConfig(
            num_layers=len(PATTERN),
            hidden_size=128,
            num_attention_heads=4,
            ffn_hidden_size=256,
            mamba_state_dim=16,
            mamba_head_dim=32,
            mamba_num_groups=1,
            normalization="RMSNorm",
            use_cpu_initialization=True,
            pipeline_dtype=dtype,
            params_dtype=dtype,
            bf16=dtype == torch.bfloat16,
            pipeline_model_parallel_size=pp,
            virtual_pipeline_model_parallel_size=vpp,
            pipeline_schedule=schedule,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            batch_p2p_comm=True,
            overlap_p2p_comm=False,
        ),
        PATTERN,
    )


def hybrid_plan(pp=4):
    cuts = CUTS if pp == 4 else (0, 2, 5, 9, 12)
    stages = len(cuts) - 1
    operations = [[] for _ in range(pp)]
    for kind, order in (("F", range(stages)), ("B", range(stages - 1, -1, -1))):
        for b in range(8):
            for s in order:
                operations[s % pp].append(dict(kind=kind, microbatch=b, stage=s))
    return dict(
        schema_version="slackpipe.plan.v2",
        num_microbatches=8,
        num_stages=stages,
        num_workers=pp,
        num_layers=len(PATTERN),
        layer_cuts=list(cuts),
        stage_to_worker=[s % pp for s in range(stages)],
        operations=operations,
        model_manifest_hash=build_model_manifest(tiny_config())["manifest_hash"],
    )


def hybrid_provider(pattern):
    def provider(
        pre_process=True, post_process=True, vp_stage=None, config=None, pg_collection=None
    ):
        return HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=128,
            max_sequence_length=64,
            hybrid_layer_pattern=pattern,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
            pg_collection=pg_collection,
            position_embedding_type="none",
            share_embeddings_and_output_weights=False,
        )

    return provider


@pytest.mark.parametrize("pp", [1, 2, 4])
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_hybrid_numerical_equivalence(tmp_path, pp, monkeypatch, precision):
    # Single-node fixture: TP=DP=CP=1, one process/device per physical PP rank.
    Utils.require_cuda_devices(pp)
    if int(os.environ.get("WORLD_SIZE", "1")) != pp:
        pytest.skip(f"requires {pp} ranks")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    clear_nvte_env_vars()
    monkeypatch.setenv("MAMBA_DETERMINISTIC", "1")
    monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    path = Path(os.environ.get("SLACKPIPE_HYBRID_PLAN", tmp_path / "hybrid.plan.json"))
    if "SLACKPIPE_HYBRID_PLAN" not in os.environ:
        path.write_text(json.dumps(hybrid_plan(pp)))
    plan = load_slackpipe_plan(path, pipeline_model_parallel_size=pp)
    Utils.initialize_model_parallel(1, 1)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    baseline = _build_model(tiny_config(dtype=dtype), provider=hybrid_provider(PATTERN))
    reference = _logical_named_parameters(baseline)
    batches = _make_batches(
        SimpleNamespace(
            seed=1234,
            num_microbatches=plan.num_microbatches,
            vocab_size=128,
            micro_batch_size=1,
            seq_length=64,
        )
    )
    ref_optimizer = torch.optim.SGD(reference.values(), lr=0.01)
    ref_losses = []
    for batch in batches:
        loss = baseline[0](batch["tokens"], batch["position_ids"], None, labels=batch["labels"])
        loss = loss.float().mean() / plan.num_microbatches
        ref_losses.append(loss.detach())
        loss.backward()
    vpp = plan.num_stages // pp
    Utils.initialize_model_parallel(
        1, pp, virtual_pipeline_model_parallel_size=vpp if pp > 1 else None
    )
    parallel_state.set_virtual_pipeline_model_parallel_world_size(vpp)
    try:
        chunks = _build_model(
            tiny_config(pp, vpp, "slackpipe", dtype),
            pipeline_schedule="slackpipe",
            pp_size=pp,
            vpp=vpp,
            layout=None,
            provider=hybrid_provider(partition_hybrid_pattern(PATTERN, plan)),
        )
        rank = parallel_state.get_pipeline_model_parallel_rank()
        raw_chunks = unwrap_model(chunks)
        records = validate_chunk_layers(plan, chunks, rank)
        assert [r["layer_id"] for r in records] == [
            i for s in range(rank, plan.num_stages, pp) for i in plan.stage_layer_ids(s)
        ]
        assert [c.vp_stage for c in chunks] == list(range(vpp))
        assert [c.pre_process for c in raw_chunks] == [rank == 0 and i == 0 for i in range(vpp)]
        assert [c.post_process for c in raw_chunks] == [
            rank == pp - 1 and i == vpp - 1 for i in range(vpp)
        ]
        actual = _logical_named_parameters(chunks)
        owners = [None] * pp
        dist.all_gather_object(owners, list(actual))
        names = [n for owner in owners for n in owner]
        assert len(names) == len(set(names)) and set(names) == set(reference)
        _copy_parameters(reference, actual)
        expected = {n: reference[n] for n in actual}
        differences = {"initial": _max_abs_parameter_diff(expected, actual)}
        optimizer = torch.optim.SGD(actual.values(), lr=0.01)
        losses = get_forward_backward_func(
            pipeline_schedule="slackpipe",
            slackpipe_plan_path=str(path),
            slackpipe_trace_path=str(tmp_path / "trace.json"),
            slackpipe_transport=os.environ.get("SLACKPIPE_TEST_TRANSPORT", "nccl-p2p"),
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
        assert all(torch.isfinite(loss["loss"]).all() for loss in losses)
        assert all(torch.isfinite(loss).all() for loss in ref_losses)
        differences["loss"] = max(
            [abs((a["loss"] - b).item()) for a, b in zip(losses, ref_losses)] or [0.0]
        )
        _assert_parameters_and_gradients_finite(actual)
        differences["gradient"] = _max_abs_gradient_diff(expected, actual)
        if differences["gradient"]:
            print(
                "Nonzero hybrid gradients:",
                {
                    n: (p.grad - expected[n].grad).abs().max().item()
                    for n, p in actual.items()
                    if p.grad is not None and not torch.equal(p.grad, expected[n].grad)
                },
            )
        optimizer.step()
        ref_optimizer.step()
        differences["post_step"] = _max_abs_parameter_diff(expected, actual)
        for name, value in differences.items():
            value = torch.tensor(value, device="cuda")
            dist.all_reduce(value, op=dist.ReduceOp.MAX)
            differences[name] = value.item()
        trace = json.loads((tmp_path / f"trace.rank{rank}.json").read_text())
        assert trace["operations"] == [asdict(op) for op in plan.worker_operations(rank)]
        assert trace["matched_plan"]
        assert all(
            not s["active_iteration"] and s["outstanding_puts"] == 0
            for s in slackpipe_transport_statistics()
        )
        print(f"Hybrid PP{pp} {precision} max absolute differences: {differences}")
        # BF16 has 7 mantissa bits. Bound accumulated-gradient error separately
        # from the FP32 loss and the rounded BF16 SGD update; never relax FP32.
        limits = dict(initial=0.0, loss=2e-4, gradient=2e-3, post_step=5e-4)
        assert all(
            value <= (limits[name] if precision == "bf16" else 0.0)
            for name, value in differences.items()
        )
        if os.environ.get("SLACKPIPE_CORRECTNESS_OUTPUT"):
            destination = Path(os.environ["SLACKPIPE_CORRECTNESS_OUTPUT"])
            destination.mkdir(parents=True, exist_ok=True)
            (destination / f"{precision}.rank{rank}.json").write_text(
                json.dumps(
                    dict(
                        pp=pp,
                        precision=precision,
                        differences=differences,
                        matched_plan=True,
                        layer_records=records,
                        operations=trace["operations"],
                    )
                )
            )
        config = chunks[0].config
        original = config.slackpipe_hybrid_pattern
        first_id = raw_chunks[0].decoder.layers[0].layer_number - 1
        replacement = "-" if original[first_id] != "-" else "M"
        config.slackpipe_hybrid_pattern = (
            original[:first_id] + replacement + original[first_id + 1 :]
        )
        try:
            with pytest.raises(ValueError, match="type mismatch"):
                validate_chunk_layers(plan, chunks, rank)
        finally:
            config.slackpipe_hybrid_pattern = original
    finally:
        shutdown_slackpipe_runtime()
        assert not slackpipe_transport_statistics()
        Utils.destroy_model_parallel()
        parallel_state.set_virtual_pipeline_model_parallel_world_size(None)
