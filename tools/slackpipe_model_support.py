# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Shared model construction and deterministic training inputs for workers and tests."""
import re
from types import SimpleNamespace

import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec as gpt_te_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import ModelType
from megatron.core.utils import unwrap_model
from megatron.training.global_vars import set_args
from megatron.training.training import get_model

SLACKPIPE_LAYOUT = "Et|t*2|t|t*4L"


def _set_minimal_training_args(**overrides):
    args = SimpleNamespace(
        create_all_gather_group=False,
        distributed_timeout_minutes=30,
        load=None,
        init_model_with_meta_device=False,
        use_torch_fsdp2=False,
        use_cpu_initialization=True,
        use_megatron_fsdp=False,
        fp16=False,
        bf16=False,
        pipeline_schedule="slackpipe",
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=4,
        pipeline_model_parallel_layout=SLACKPIPE_LAYOUT,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    set_args(args)
    return args


def _gpt_model_provider(
    pre_process=True, post_process=True, vp_stage=None, config=None, pg_collection=None
):
    return GPTModel(
        config=config,
        transformer_layer_spec=gpt_te_spec(),
        vocab_size=128,
        max_sequence_length=8,
        pre_process=pre_process,
        post_process=post_process,
        position_embedding_type="rope",
        vp_stage=vp_stage,
        pg_collection=pg_collection,
        share_embeddings_and_output_weights=False,
    )


def _build_model(
    config, pipeline_schedule="default", pp_size=1, vpp=None, layout=None, provider=None
):
    _set_minimal_training_args(
        bf16=config.bf16,
        pipeline_schedule=pipeline_schedule,
        pipeline_model_parallel_size=pp_size,
        virtual_pipeline_model_parallel_size=vpp,
        pipeline_model_parallel_layout=layout,
    )
    torch.manual_seed(456)
    model_parallel_cuda_manual_seed(456)
    return get_model(
        provider or _gpt_model_provider,
        model_type=ModelType.encoder_or_decoder,
        wrap_with_ddp=False,
        config=config,
    )


def _batch_iterator(batches):
    for batch in batches:
        yield {key: value.clone() for key, value in batch.items()}


def _forward_step_func(data_iterator, model):
    batch = next(data_iterator)
    output_tensor = model(batch["tokens"], batch["position_ids"], None, labels=batch["labels"])

    def loss_func(output_tensor):
        loss = output_tensor.float().mean()
        return loss, {"loss": loss.detach()}

    return output_tensor, loss_func


def _logical_named_parameters(model):
    chunks = model if isinstance(model, list) else [model]
    logical_params = {}
    decoder_layer_pattern = re.compile(r"^decoder\.layers\.(\d+)\.(.*)$")
    for chunk in unwrap_model(chunks):
        local_to_global = {
            local_idx: layer.layer_number - 1
            for local_idx, layer in enumerate(chunk.decoder.layers)
        }
        for name, param in chunk.named_parameters():
            match = decoder_layer_pattern.match(name)
            if match:
                local_idx = int(match.group(1))
                name = f"decoder.layers.{local_to_global[local_idx]}.{match.group(2)}"
            assert name not in logical_params, f"duplicate logical parameter {name}"
            logical_params[name] = param
    return logical_params


def _make_batches(args):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 17)
    batches = []
    for _ in range(args.num_microbatches):
        tokens = torch.randint(
            0,
            args.vocab_size,
            (args.micro_batch_size, args.seq_length),
            generator=generator,
            device="cpu",
        ).cuda()
        labels = torch.randint(
            0,
            args.vocab_size,
            (args.micro_batch_size, args.seq_length),
            generator=generator,
            device="cpu",
        ).cuda()
        position_ids = torch.arange(args.seq_length, device="cuda").view(1, -1)
        position_ids = position_ids.expand(args.micro_batch_size, -1).contiguous()
        batches.append({"tokens": tokens, "labels": labels, "position_ids": position_ids})
    return batches
