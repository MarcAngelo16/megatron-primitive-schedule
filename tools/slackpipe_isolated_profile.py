# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""One-GPU isolated real-Megatron unit calibration. Never invokes a pipeline scheduler."""

import gc
import tempfile
import time
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from unittest.mock import patch

import torch
import torch.distributed as dist

from megatron.core import parallel_state, tensor_parallel
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.pipeline_parallel.slackpipe.collection import environment_metadata
from megatron.core.pipeline_parallel.slackpipe.cost_profile import profile_fingerprint
from megatron.core.pipeline_parallel.slackpipe.isolated_profile import (
    ESTIMATOR,
    POLICY_VERSION,
    build_profile,
    class_mapping,
    execution_signature,
)
from megatron.core.pipeline_parallel.slackpipe.manifest import build_model_manifest
from megatron.core.pipeline_parallel.slackpipe.profile_integrity import publish_profile
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.spec_utils import build_module
from tools.slackpipe_eval_common import write_json
from tools.slackpipe_eval_config import (
    fingerprint,
    load_model,
    schedule_topology,
    transformer_config,
)


@contextmanager
def no_communication():
    """Elide only identity TP=1 reductions in Megatron's unchanged native CE.

    Gloo groups provide rank/size metadata. No collective or P2P is dispatched.
    Other unexpected communication is a hard error, not a timed cost.
    """

    def identity_reduce(tensor, op=None, group=None, async_op=False):
        if dist.get_world_size(group) != 1 or async_op:
            raise RuntimeError("Isolated profiling supports synchronous identity reductions only")
        return None

    def forbidden(*args, **kwargs):
        raise RuntimeError("Communication is forbidden during isolated profiling")

    with ExitStack() as stack:
        stack.enter_context(patch.object(dist, "all_reduce", identity_reduce))
        for name in (
            "send",
            "recv",
            "isend",
            "irecv",
            "batch_isend_irecv",
            "all_gather",
            "all_gather_into_tensor",
            "reduce_scatter_tensor",
            "broadcast",
            "barrier",
        ):
            stack.enter_context(patch.object(dist, name, forbidden))
        yield


def make_unit(model: dict, config, unit: dict, seq: int, batch: int):
    """Instantiate one actual partitionable layer OR one independent boundary module."""
    pg = ProcessGroupCollection.use_mpu_process_groups()
    dtype = config.params_dtype
    hidden = torch.randn(
        seq, batch, config.hidden_size, device="cuda", dtype=dtype, requires_grad=True
    )
    key = unit["class_signature"]
    if key == "first":
        module = (
            LanguageModelEmbedding(
                config,
                model["vocab_size"],
                seq,
                position_embedding_type=model["position_embedding"],
                tp_group=pg.tp,
            )
            .cuda()
            .to(dtype)
        )
        tokens = torch.randint(model["vocab_size"], (batch, seq), device="cuda")
        positions = torch.arange(seq, device="cuda").expand(batch, -1)
        forward = lambda: module(tokens, positions)
        hidden = None
    elif key == "last":
        module = (
            torch.nn.ModuleDict(
                dict(
                    norm=TENorm(config, config.hidden_size, eps=config.layernorm_epsilon),
                    head=tensor_parallel.ColumnParallelLinear(
                        config.hidden_size,
                        model["vocab_size"],
                        config=config,
                        init_method=config.init_method,
                        bias=False,
                        skip_bias_add=False,
                        gather_output=False,
                        tp_group=pg.tp,
                    ),
                )
            )
            .cuda()
            .to(dtype)
        )
        labels = torch.randint(model["vocab_size"], (seq, batch), device="cuda")

        def forward():
            logits, _ = module["head"](module["norm"](hidden))
            # Same native unfused vocabulary CE and float mean as the campaign.
            return (
                tensor_parallel.vocab_parallel_cross_entropy(logits.float(), labels)
                .transpose(0, 1)
                .contiguous()
                .float()
                .mean()
            )

    else:
        kind = unit["layer_type"]
        spec = (
            get_gpt_layer_with_transformer_engine_spec()
            if kind == "decoder"
            else getattr(
                hybrid_stack_spec.submodules,
                {"mamba": "mamba_layer", "mlp": "mlp_layer", "attention": "attention_layer"}[kind],
            )
        )
        module = (
            build_module(
                spec, config=config, layer_number=unit["representative_layer"] + 1, pg_collection=pg
            )
            .cuda()
            .to(dtype)
        )
        rotary = None
        if model["position_embedding"] == "rope":
            rotary = RotaryEmbedding(
                config.kv_channels, rotary_percent=1.0, use_cpu_initialization=True, cp_group=pg.cp
            ).cuda()(seq)

        def forward():
            value = module(hidden, attention_mask=None, rotary_pos_emb=rotary)
            return value[0] if isinstance(value, tuple) else value

    module.train()
    return module, hidden, forward


def measure_unit(module, hidden, forward, *, warmups: int, iterations: int) -> dict:
    """Time F and real B separately; keep input/gradient setup and validation outside events."""
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    end.record()
    torch.cuda.synchronize()
    output = forward()
    upstream = torch.ones_like(output) if output.ndim == 0 else torch.randn_like(output)
    del output
    rows = []

    def timed(call):
        torch.cuda.synchronize()
        start.record()
        before = time.perf_counter()
        value = call()
        end.record()
        torch.cuda.synchronize()
        return value, start.elapsed_time(end), (time.perf_counter() - before) * 1000

    for i in range(warmups + iterations):
        module.zero_grad(set_to_none=True)
        if hidden is not None:
            hidden.grad = None
        output, f, fw = timed(forward)
        _, b, bw = timed(lambda output=output: torch.autograd.backward(output, upstream))
        tensors = [output] + [p.grad for p in module.parameters() if p.requires_grad]
        if hidden is not None:
            tensors.append(hidden.grad)
        if any(t is None or not torch.isfinite(t).all().item() for t in tensors):
            raise RuntimeError("Invalid isolated forward/parameter/input gradient")
        del tensors
        if i >= warmups:
            rows.append(
                dict(
                    iteration=i,
                    forward_gpu_ms=f,
                    backward_gpu_ms=b,
                    forward_wall_ms=fw,
                    backward_wall_ms=bw,
                )
            )
        del output
    return dict(
        executed=dict(warmup_pairs=warmups, measured_pairs=len(rows), initialization_forwards=1),
        samples=rows,
        **{key: [r[key] for r in rows] for key in ("forward_gpu_ms", "backward_gpu_ms")},
        finite_outputs_and_gradients=True,
    )


def collect(args) -> None:
    if dist.is_initialized():
        raise ValueError("Isolated collector must run in a fresh single-process worker")
    if args.output.exists():
        raise FileExistsError("Use a fresh isolated output directory")
    if args.isolated_profile_warmups < 1 or args.isolated_profile_iterations < 1:
        raise ValueError("Isolated profiling requires positive warmup and measurement counts")
    model = load_model(args.model_config)
    topology = schedule_topology(args.schedule, args.pp, args.logical_stages, args.microbatches)
    config = transformer_config(model, dict(pp=1, vpp=None), args.precision)
    if (
        config.recompute_granularity
        or config.cross_entropy_loss_fusion
        or config.apply_query_key_layer_scaling
    ):
        raise ValueError(
            "This isolated adapter does not support recompute/fused loss/layer-dependent attention scaling"
        )
    manifest = build_model_manifest(config)
    execution = execution_signature(
        model,
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        precision=args.precision,
    )
    classes = class_mapping(manifest, execution)
    from tools.slackpipe_eval_common import require_cuda_devices

    # Target PP is profile metadata; isolated collection uses one GPU/process.
    require_cuda_devices(1, local_rank=0)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Rank/size APIs required by real layers, without any NCCL process group.
    with tempfile.TemporaryDirectory() as rendezvous:
        dist.init_process_group(
            "gloo", init_method="file://" + rendezvous + "/store", rank=0, world_size=1
        )
        try:
            parallel_state.initialize_model_parallel()
            environment = environment_metadata()
            context = dict(
                estimator=ESTIMATOR,
                profiling_backend="layer",
                execution=execution,
                model_config_hash=fingerprint(model),
                model_manifest_hash=manifest["manifest_hash"],
                warmups=args.isolated_profile_warmups,
                iterations=args.isolated_profile_iterations,
                statistic="median",
                policy_version=POLICY_VERSION,
                class_signatures=[c["class_signature"] for c in classes],
                environment=environment,
                seed=args.seed,
            )
            units = []
            for unit in classes + [
                dict(class_signature="first", layer_type="embedding"),
                dict(class_signature="last", layer_type="final_norm_head_loss"),
            ]:
                torch.manual_seed(args.seed)
                model_parallel_cuda_manual_seed(args.seed)
                torch.cuda.reset_peak_memory_stats()
                before = time.perf_counter()
                with no_communication():
                    module, hidden, forward = make_unit(
                        model, config, unit, args.seq_length, args.micro_batch_size
                    )
                    row = measure_unit(
                        module,
                        hidden,
                        forward,
                        warmups=args.isolated_profile_warmups,
                        iterations=args.isolated_profile_iterations,
                    )
                row.update(
                    **unit,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    elapsed_seconds=time.perf_counter() - before,
                )
                units.append(row)
                write_json(
                    args.output / "isolated_profile_raw.json",
                    dict(
                        schema_version="slackpipe.isolated_profile_raw.v1",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        context=context,
                        units=units,
                    ),
                )
                del module, hidden, forward
                gc.collect()
                torch.cuda.empty_cache()
            profile = build_profile(
                manifest=manifest,
                classes=classes,
                units=units,
                context=context,
                model_config=dict(
                    **manifest["model_config"],
                    model_config_hash=fingerprint(model),
                    dtype=args.precision,
                    sequence_length=args.seq_length,
                    micro_batch_size=args.micro_batch_size,
                    vocab_size=model["vocab_size"],
                ),
                parallel_config={k: topology[k] for k in ("pp", "vpp", "tp", "dp", "cp")},
            )
            profile["collection"] = dict(
                measurement_mode="isolated",
                full_model_constructed=False,
                unique_layer_classes=len(classes),
                boundary_units=2,
                elapsed_seconds=time.perf_counter() - started,
                peak_allocated_bytes=max(u["peak_allocated_bytes"] for u in units),
                peak_reserved_bytes=max(u["peak_reserved_bytes"] for u in units),
                environment=environment,
            )
            profile["target_instance"] = dict(
                num_microbatches=args.microbatches,
                num_stages=topology["num_stages"],
                num_layers=model["num_layers"],
            )
            profile["cost_profile_hash"] = profile_fingerprint(profile)
            write_json(args.output / "model_manifest.json", manifest)
            write_json(
                args.output / "isolated_profile_summary.json",
                dict(
                    classes=classes,
                    summaries=profile["isolated"]["summaries"],
                    collection=profile["collection"],
                ),
            )
            publish_profile(args.output / "cost_profile.json", profile)
        finally:
            parallel_state.destroy_model_parallel()
            dist.destroy_process_group()


def main() -> None:
    from tools.slackpipe_eval_args import argument_parser, normalize_args

    parser = argument_parser()
    parser.add_argument("--run-id")
    args = normalize_args(parser.parse_args())
    if args.stage != "calibrate":
        parser.error("This worker only implements --phase calibrate")
    collect(args)


if __name__ == "__main__":
    main()
