import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.pipeline_parallel.slackpipe.plan import (
    SLACKPIPE_PLAN_SCHEMA_VERSION,
    derive_pipeline_model_parallel_layout,
    load_slackpipe_plan,
)
from megatron.core.pipeline_parallel.slackpipe.schedule import clear_slackpipe_runtime_cache
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.arguments import _configure_slackpipe_plan_args
from megatron.training.training import get_model
from tests.unit_tests.test_utilities import Utils, clear_nvte_env_vars
from tools.slackpipe_model_support import (
    _batch_iterator,
    _build_model,
    _forward_step_func,
    _gpt_model_provider,
    _logical_named_parameters,
    _set_minimal_training_args,
)

# Copyright (c) 2026 NVIDIA CORPORATION. All rights reserved.


NUM_LAYERS = 8
SLACKPIPE_LAYOUT = "Et|t*2|t|t*4L"
SLACKPIPE_SPLIT = [1, 2, 1, 4]
SLACKPIPE_PP2_LAYOUT = "Et|t*2|t*2|t*3L"
SLACKPIPE_PP2_SPLIT = [1, 2, 2, 3]


def _gpt_local_model_provider(
    pre_process=True, post_process=True, vp_stage=None, config=None, pg_collection=None
):
    return GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=128,
        max_sequence_length=8,
        pre_process=pre_process,
        post_process=post_process,
        position_embedding_type="rope",
        vp_stage=vp_stage,
        pg_collection=pg_collection,
        share_embeddings_and_output_weights=False,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_slackpipe_pp1_constructs_logical_vpp_chunks():
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("run PP=1 SlackPipe construction test without torchrun")

    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    parallel_state.set_virtual_pipeline_model_parallel_world_size(4)
    _set_minimal_training_args()
    torch.manual_seed(123)
    model_parallel_cuda_manual_seed(123)
    config = TransformerConfig(
        num_layers=NUM_LAYERS,
        hidden_size=128,
        num_attention_heads=8,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=4,
        pipeline_model_parallel_layout=SLACKPIPE_LAYOUT,
        pipeline_schedule="slackpipe",
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )

    model = get_model(
        _gpt_model_provider,
        model_type=ModelType.encoder_or_decoder,
        wrap_with_ddp=False,
        config=config,
    )

    assert len(model) == 4
    assert [chunk.vp_stage for chunk in model] == [0, 1, 2, 3]
    assert [len(chunk.decoder.layers) for chunk in model] == [1, 2, 1, 4]
    assert [[layer.layer_number for layer in chunk.decoder.layers] for chunk in model] == [
        [1],
        [2, 3],
        [4],
        [5, 6, 7, 8],
    ]
    assert [chunk.pre_process for chunk in model] == [True, False, False, False]
    assert [chunk.post_process for chunk in model] == [False, False, False, True]

    Utils.destroy_model_parallel()
    parallel_state.set_virtual_pipeline_model_parallel_world_size(None)


def _make_equivalence_config(
    pipeline_schedule="default", vpp=None, layout=None, num_layers=NUM_LAYERS
):
    return TransformerConfig(
        num_layers=num_layers,
        hidden_size=64,
        num_attention_heads=4,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=vpp,
        pipeline_model_parallel_layout=layout,
        pipeline_schedule=pipeline_schedule,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )


def _make_pp2_equivalence_config(
    pipeline_schedule="default", vpp=2, layout=SLACKPIPE_PP2_LAYOUT, num_layers=NUM_LAYERS
):
    return TransformerConfig(
        num_layers=num_layers,
        hidden_size=64,
        num_attention_heads=4,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=vpp,
        pipeline_model_parallel_layout=layout,
        pipeline_schedule=pipeline_schedule,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        batch_p2p_comm=True,
        overlap_p2p_comm=False,
    )


def _make_batches(num_microbatches=4):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(789)
    batches = []
    for _ in range(num_microbatches):
        tokens = torch.randint(0, 128, (1, 8), generator=generator, device="cpu").cuda()
        labels = torch.randint(0, 128, (1, 8), generator=generator, device="cpu").cuda()
        position_ids = torch.arange(8, device="cuda").view(1, -1)
        batches.append({"tokens": tokens, "labels": labels, "position_ids": position_ids})
    return batches


def _write_slackpipe_plan(path):
    operations = []
    for microbatch in range(4):
        operations.extend(
            {"kind": "F", "microbatch": microbatch, "stage": stage} for stage in range(4)
        )
        operations.extend(
            {"kind": "B", "microbatch": microbatch, "stage": stage} for stage in reversed(range(4))
        )
    path.write_text(
        json.dumps(
            {
                "num_microbatches": 4,
                "num_stages": 4,
                "num_workers": 1,
                "num_layers": NUM_LAYERS,
                "layer_split": SLACKPIPE_SPLIT,
                "stage_to_worker": [0, 0, 0, 0],
                "operations": [operations],
            }
        ),
        encoding="utf-8",
    )


def _write_slackpipe_pp2_plan(path):
    worker_operations = [[], []]
    for microbatch in range(4):
        worker_operations[0].extend(
            [
                {"kind": "F", "microbatch": microbatch, "stage": 0},
                {"kind": "F", "microbatch": microbatch, "stage": 2},
                {"kind": "B", "microbatch": microbatch, "stage": 2},
                {"kind": "B", "microbatch": microbatch, "stage": 0},
            ]
        )
        worker_operations[1].extend(
            [
                {"kind": "F", "microbatch": microbatch, "stage": 1},
                {"kind": "F", "microbatch": microbatch, "stage": 3},
                {"kind": "B", "microbatch": microbatch, "stage": 3},
                {"kind": "B", "microbatch": microbatch, "stage": 1},
            ]
        )
    path.write_text(
        json.dumps(
            {
                "schema_version": SLACKPIPE_PLAN_SCHEMA_VERSION,
                "num_microbatches": 4,
                "num_stages": 4,
                "num_workers": 2,
                "num_layers": NUM_LAYERS,
                "layer_split": SLACKPIPE_PP2_SPLIT,
                "stage_to_worker": [0, 1, 0, 1],
                "operations": worker_operations,
                "solver_status": "optimal",
                "predicted_makespan": 16.0,
                "forward_costs": [1.0, 2.0, 2.0, 3.0],
                "backward_costs": [1.0, 2.0, 2.0, 3.0],
            }
        ),
        encoding="utf-8",
    )


def _minimal_slackpipe_args(plan_path, explicit_layout=None):
    return SimpleNamespace(
        pipeline_schedule="slackpipe",
        slackpipe_plan=str(plan_path),
        num_layers=NUM_LAYERS,
        pipeline_model_parallel_size=2,
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        data_parallel_size=1,
        pipeline_model_parallel_layout=explicit_layout,
        virtual_pipeline_model_parallel_size=None,
        overlap_p2p_comm=True,
        align_param_gather=True,
    )


def _max_abs_parameter_diff(left, right):
    assert set(left) == set(right)
    return max(
        (left[name].detach().float() - right[name].detach().float()).abs().max().item()
        for name in left
    )


def _max_abs_gradient_diff(left, right):
    assert set(left) == set(right)
    max_diff = 0.0
    for name in left:
        left_grad = left[name].grad
        right_grad = right[name].grad
        assert (left_grad is None) == (right_grad is None), name
        if left_grad is not None:
            max_diff = max(max_diff, (left_grad.float() - right_grad.float()).abs().max().item())
    return max_diff


def _copy_parameters(source, destination):
    assert set(destination).issubset(set(source))
    for name, param in destination.items():
        param.data.copy_(source[name].data)


def _assert_parameters_and_gradients_finite(params):
    for name, param in params.items():
        assert torch.isfinite(param).all(), name
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), name


def _max_abs_distributed(value):
    tensor = torch.tensor([value], dtype=torch.float32, device="cuda")
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
    return tensor.item()


def test_slackpipe_args_derive_layout_from_plan(tmp_path):
    plan_path = tmp_path / "slackpipe_pp2_plan.json"
    _write_slackpipe_pp2_plan(plan_path)

    args = _minimal_slackpipe_args(plan_path)
    _configure_slackpipe_plan_args(args)

    assert args.pipeline_model_parallel_layout == SLACKPIPE_PP2_LAYOUT
    assert args.virtual_pipeline_model_parallel_size == 2
    assert args.overlap_p2p_comm is False
    assert args.align_param_gather is False

    args = _minimal_slackpipe_args(plan_path, explicit_layout="Et*4|t*4L")
    with pytest.raises(AssertionError, match="must match the layout derived"):
        _configure_slackpipe_plan_args(args)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_slackpipe_pp1_numerical_equivalence(tmp_path):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("run PP=1 SlackPipe equivalence test without torchrun")

    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    parallel_state.set_virtual_pipeline_model_parallel_world_size(None)
    baseline_model = _build_model(_make_equivalence_config(), provider=_gpt_local_model_provider)

    parallel_state.set_virtual_pipeline_model_parallel_world_size(4)
    slackpipe_model = _build_model(
        _make_equivalence_config("slackpipe", vpp=4, layout=SLACKPIPE_LAYOUT),
        pipeline_schedule="slackpipe",
        vpp=4,
        layout=SLACKPIPE_LAYOUT,
        provider=_gpt_local_model_provider,
    )

    baseline_params = _logical_named_parameters(baseline_model)
    slackpipe_params = _logical_named_parameters(slackpipe_model)
    initial_param_diff = _max_abs_parameter_diff(baseline_params, slackpipe_params)

    baseline_optimizer = torch.optim.SGD(baseline_params.values(), lr=0.01)
    slackpipe_optimizer = torch.optim.SGD(slackpipe_params.values(), lr=0.01)
    baseline_optimizer.zero_grad(set_to_none=True)
    slackpipe_optimizer.zero_grad(set_to_none=True)

    batches = _make_batches()
    baseline_losses = get_forward_backward_func()(
        forward_step_func=_forward_step_func,
        data_iterator=_batch_iterator(batches),
        model=baseline_model,
        num_microbatches=4,
        seq_length=8,
        micro_batch_size=1,
        forward_only=False,
    )

    plan_path = tmp_path / "slackpipe_plan.json"
    _write_slackpipe_plan(plan_path)
    slackpipe_losses = get_forward_backward_func(
        pipeline_schedule="slackpipe", slackpipe_plan_path=str(plan_path)
    )(
        forward_step_func=_forward_step_func,
        data_iterator=[_batch_iterator(batches) for _ in range(4)],
        model=slackpipe_model,
        num_microbatches=4,
        seq_length=8,
        micro_batch_size=1,
        forward_only=False,
    )

    assert len(baseline_losses) == len(slackpipe_losses) == 4
    loss_diff = max(
        (baseline_loss["loss"] - slackpipe_loss["loss"]).abs().max().item()
        for baseline_loss, slackpipe_loss in zip(baseline_losses, slackpipe_losses)
    )
    grad_diff = _max_abs_gradient_diff(baseline_params, slackpipe_params)

    baseline_optimizer.step()
    slackpipe_optimizer.step()
    post_step_param_diff = _max_abs_parameter_diff(baseline_params, slackpipe_params)

    print(
        "SlackPipe PP=1 equivalence max abs diffs: "
        f"initial_params={initial_param_diff:.8e}, "
        f"loss={loss_diff:.8e}, "
        f"grads={grad_diff:.8e}, "
        f"post_step_params={post_step_param_diff:.8e}"
    )

    assert initial_param_diff == 0.0
    assert loss_diff == 0.0
    assert grad_diff == 0.0
    assert post_step_param_diff == 0.0

    Utils.destroy_model_parallel()
    parallel_state.set_virtual_pipeline_model_parallel_world_size(None)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_slackpipe_pp2_numerical_equivalence(tmp_path):
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        pytest.skip("run with torchrun --nproc-per-node 2")

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    clear_nvte_env_vars()
    external_plan_path = os.environ.get("SLACKPIPE_EXTERNAL_PP2_PLAN")
    if external_plan_path:
        plan_path = Path(external_plan_path)
    else:
        plan_path = tmp_path / "slackpipe_pp2_plan.json"
        _write_slackpipe_pp2_plan(plan_path)
    parsed_plan = load_slackpipe_plan(plan_path, pipeline_model_parallel_size=2)
    num_microbatches = parsed_plan.num_microbatches
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    baseline_model = _build_model(
        _make_equivalence_config(num_layers=parsed_plan.num_layers),
        provider=_gpt_local_model_provider,
    )
    baseline_params = _logical_named_parameters(baseline_model)
    baseline_optimizer = torch.optim.SGD(baseline_params.values(), lr=0.01)
    baseline_optimizer.zero_grad(set_to_none=True)
    batches = _make_batches(num_microbatches)
    baseline_losses = []
    baseline_module = baseline_model[0] if isinstance(baseline_model, list) else baseline_model
    for batch in batches:
        output_tensor = baseline_module(
            batch["tokens"], batch["position_ids"], None, labels=batch["labels"]
        )
        loss = output_tensor.float().mean()
        baseline_losses.append({"loss": (loss / num_microbatches).detach()})
        (loss / num_microbatches).backward()

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=parsed_plan.num_stages // 2,
    )

    try:
        slackpipe_args = _minimal_slackpipe_args(plan_path)
        slackpipe_args.num_layers = parsed_plan.num_layers
        _configure_slackpipe_plan_args(slackpipe_args)
        assert (
            slackpipe_args.pipeline_model_parallel_layout
            == derive_pipeline_model_parallel_layout(parsed_plan)
        )

        slackpipe_model = _build_model(
            _make_pp2_equivalence_config(
                "slackpipe",
                vpp=slackpipe_args.virtual_pipeline_model_parallel_size,
                layout=slackpipe_args.pipeline_model_parallel_layout,
                num_layers=parsed_plan.num_layers,
            ),
            pipeline_schedule="slackpipe",
            pp_size=2,
            vpp=slackpipe_args.virtual_pipeline_model_parallel_size,
            layout=slackpipe_args.pipeline_model_parallel_layout,
            provider=_gpt_local_model_provider,
        )

        local_layer_counts = [len(chunk.decoder.layers) for chunk in slackpipe_model]
        expected_local_layer_counts = [
            layer_count
            for stage, layer_count in enumerate(parsed_plan.layer_split)
            if parsed_plan.stage_to_worker[stage]
            == parallel_state.get_pipeline_model_parallel_rank()
        ]
        assert local_layer_counts == expected_local_layer_counts

        slackpipe_params = _logical_named_parameters(slackpipe_model)
        parameter_names = [None, None]
        torch.distributed.all_gather_object(parameter_names, list(slackpipe_params))
        all_names = [name for names in parameter_names for name in names]
        assert len(all_names) == len(set(all_names))
        assert set(all_names) == set(baseline_params)
        _copy_parameters(baseline_params, slackpipe_params)
        baseline_subset = {name: baseline_params[name] for name in slackpipe_params}
        initial_param_diff = _max_abs_parameter_diff(baseline_subset, slackpipe_params)
        slackpipe_optimizer = torch.optim.SGD(slackpipe_params.values(), lr=0.01)
        slackpipe_optimizer.zero_grad(set_to_none=True)

        trace_dir = Path(os.environ.get("SLACKPIPE_EXTERNAL_TRACE_DIR", tmp_path))
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / "slackpipe_trace.json"
        slackpipe_losses = get_forward_backward_func(
            pipeline_schedule="slackpipe",
            slackpipe_plan_path=str(plan_path),
            slackpipe_trace_path=str(trace_path),
            slackpipe_transport=os.environ.get("SLACKPIPE_TEST_TRANSPORT", "nccl-p2p"),
        )(
            forward_step_func=_forward_step_func,
            data_iterator=[_batch_iterator(batches) for _ in slackpipe_model],
            model=slackpipe_model,
            num_microbatches=num_microbatches,
            seq_length=8,
            micro_batch_size=1,
            forward_only=False,
        )

        is_last_worker = (
            parallel_state.get_pipeline_model_parallel_rank() == parsed_plan.stage_to_worker[-1]
        )
        assert len(slackpipe_losses) == (num_microbatches if is_last_worker else 0)
        local_loss_diff = 0.0
        for baseline_loss in baseline_losses:
            assert torch.isfinite(baseline_loss["loss"]).all()
        if slackpipe_losses:
            assert len(baseline_losses) == len(slackpipe_losses)
            for slackpipe_loss in slackpipe_losses:
                assert torch.isfinite(slackpipe_loss["loss"]).all()
            local_loss_diff = max(
                (baseline_loss["loss"] - slackpipe_loss["loss"]).abs().max().item()
                for baseline_loss, slackpipe_loss in zip(baseline_losses, slackpipe_losses)
            )
        loss_diff = _max_abs_distributed(local_loss_diff)
        _assert_parameters_and_gradients_finite(baseline_subset)
        _assert_parameters_and_gradients_finite(slackpipe_params)
        grad_diff = _max_abs_distributed(_max_abs_gradient_diff(baseline_subset, slackpipe_params))

        baseline_optimizer.step()
        slackpipe_optimizer.step()
        _assert_parameters_and_gradients_finite(baseline_subset)
        _assert_parameters_and_gradients_finite(slackpipe_params)
        post_step_param_diff = _max_abs_distributed(
            _max_abs_parameter_diff(baseline_subset, slackpipe_params)
        )
        initial_param_diff = _max_abs_distributed(initial_param_diff)

        if torch.distributed.get_rank() == 0:
            print(
                "SlackPipe PP=2 equivalence max abs diffs: "
                f"initial_params={initial_param_diff:.8e}, "
                f"loss={loss_diff:.8e}, "
                f"grads={grad_diff:.8e}, "
                f"post_step_params={post_step_param_diff:.8e}"
            )

        assert initial_param_diff == 0.0
        assert loss_diff == 0.0
        assert grad_diff == 0.0
        assert post_step_param_diff == 0.0

        rank = parallel_state.get_pipeline_model_parallel_rank()
        rank_trace_path = trace_dir / f"slackpipe_trace.rank{rank}.json"
        trace = json.loads(rank_trace_path.read_text(encoding="utf-8"))
        assert trace["matched_plan"] is True
        assert trace["operations"] == [
            {"kind": op.kind, "microbatch": op.microbatch, "stage": op.stage}
            for op in parsed_plan.worker_operations(rank)
        ]
        if rank == 0:
            (trace_dir / "equivalence.json").write_text(
                json.dumps(
                    {
                        "plan": str(plan_path),
                        "transport": os.environ.get("SLACKPIPE_TEST_TRANSPORT", "nccl-p2p"),
                        "layer_split": list(parsed_plan.layer_split),
                        "operation_count": sum(len(ops) for ops in parsed_plan.operations),
                        "logical_parameter_count": len(all_names),
                        "initial_params_max_abs_diff": initial_param_diff,
                        "loss_max_abs_diff": loss_diff,
                        "gradients_max_abs_diff": grad_diff,
                        "post_step_params_max_abs_diff": post_step_param_diff,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
    finally:
        clear_slackpipe_runtime_cache()
        parallel_state.destroy_model_parallel()
        Utils.inited = False
        parallel_state.set_virtual_pipeline_model_parallel_world_size(None)
