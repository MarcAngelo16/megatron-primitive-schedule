from tools.slackpipe_model_support import _make_batches

# Copyright (c) 2026 NVIDIA CORPORATION. All rights reserved.

"""Small controlled SlackPipe PP=2 benchmark.

This is intentionally a narrow development benchmark, not a CI performance
contract. It compares uniform interleaved 1F1B, partition-controlled
interleaved 1F1B, and SlackPipe using one generated plan.
"""

import argparse
import csv
import gc
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs import (
    get_gpt_heterogeneous_layer_spec,
)
from megatron.core.pipeline_parallel.schedules import (
    begin_slackpipe_cost_calibration_iteration,
    end_slackpipe_cost_calibration_iteration,
    get_forward_backward_func,
)
from megatron.core.pipeline_parallel.slackpipe.collection import (
    CollectionOptions,
    collect_steps,
    environment_metadata,
    memory_sample,
    profiler_settings,
    summarize_rank_samples,
)
from megatron.core.pipeline_parallel.slackpipe.cost_profile import build_cost_profile
from megatron.core.pipeline_parallel.slackpipe.plan import (
    derive_pipeline_model_parallel_layout,
    load_slackpipe_plan,
)
from megatron.core.pipeline_parallel.slackpipe.schedule import (
    _get_slackpipe_runtime,
    clear_slackpipe_plan_cache,
    shutdown_slackpipe_runtime,
    slackpipe_transport_statistics,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.heterogeneous.heterogeneous_config import (
    HeterogeneousTransformerConfig,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.global_vars import set_args
from megatron.training.training import get_model
from tests.unit_tests.test_utilities import Utils, clear_nvte_env_vars


def _layout_from_split(split):
    specs = []
    for stage, layer_count in enumerate(split):
        spec = ""
        if stage == 0:
            spec += "E"
        if layer_count == 1:
            spec += "t"
        elif layer_count > 1:
            spec += f"t*{layer_count}"
        if stage == len(split) - 1:
            spec += "L"
        specs.append(spec)
    return "|".join(specs)


def _set_minimal_args(schedule, pp_size, vpp, layout):
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
        pipeline_schedule=schedule,
        pipeline_model_parallel_size=pp_size,
        virtual_pipeline_model_parallel_size=vpp,
        pipeline_model_parallel_layout=layout,
    )
    set_args(args)


def _model_provider_factory(vocab_size, seq_length):
    def provider(
        pre_process=True, post_process=True, vp_stage=None, config=None, pg_collection=None
    ):
        return GPTModel(
            config=config,
            transformer_layer_spec=(
                get_gpt_heterogeneous_layer_spec(config, use_te=False, vp_stage=vp_stage)
                if isinstance(config, HeterogeneousTransformerConfig)
                else get_gpt_layer_local_spec()
            ),
            vocab_size=vocab_size,
            max_sequence_length=seq_length,
            pre_process=pre_process,
            post_process=post_process,
            position_embedding_type="rope",
            vp_stage=vp_stage,
            pg_collection=pg_collection,
            share_embeddings_and_output_weights=False,
        )

    return provider


def _make_config(args, layout, schedule):
    config_type = TransformerConfig
    extra = {}
    if getattr(args, "heterogeneous_config", None):
        config_type = HeterogeneousTransformerConfig
        extra["heterogeneous_layers_config_path"] = str(args.heterogeneous_config)
    return config_type(
        **extra,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2,
        pipeline_model_parallel_layout=layout,
        pipeline_schedule=schedule,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        batch_p2p_comm=True,
        overlap_p2p_comm=False,
    )


def _build_model(args, mode):
    _set_minimal_args(mode["schedule"], 2, 2, mode["layout"])
    torch.manual_seed(args.seed)
    model_parallel_cuda_manual_seed(args.seed)
    model = get_model(
        _model_provider_factory(args.vocab_size, args.seq_length),
        model_type=ModelType.encoder_or_decoder,
        wrap_with_ddp=False,
        config=_make_config(args, mode["layout"], mode["schedule"]),
    )
    for chunk in model:
        chunk.train()
    # Same global parameter initialization for homogeneous and heterogeneous cuts.
    from tests.unit_tests.pipeline_parallel.test_slackpipe_model_construction import (
        _logical_named_parameters,
    )

    with torch.no_grad():
        for name, param in _logical_named_parameters(model).items():
            generator = torch.Generator(device=param.device)
            generator.manual_seed(
                int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little") + args.seed
            )
            if "layer_norm" in name or "layernorm" in name:
                param.fill_(0.0 if name.endswith("bias") else 1.0)
            elif name.endswith("bias"):
                param.zero_()
            else:
                std = 0.02
                if name.endswith(("linear_proj.weight", "linear_fc2.weight")):
                    std /= (2 * args.num_layers) ** 0.5
                param.normal_(0.0, std, generator=generator)
    return model


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


def _logical_params(model):
    params = {}
    for chunk in model:
        for name, param in chunk.named_parameters():
            params[f"vp{getattr(chunk, 'vp_stage', 0)}.{name}"] = param
    return params


def _assert_finite(losses, params):
    finite = torch.tensor([1], dtype=torch.int32, device="cuda")
    failures = []
    for loss in losses:
        if not torch.isfinite(loss["loss"]).all():
            finite.zero_()
            failures.append("loss")
    for name, param in params.items():
        if not torch.isfinite(param).all():
            finite.zero_()
            failures.append(f"parameter {name}")
        if param.grad is not None and not torch.isfinite(param.grad).all():
            finite.zero_()
            failures.append(f"gradient {name}")
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if finite.item() != 1:
        raise RuntimeError(
            f"non-finite loss, parameter, or gradient detected on rank {dist.get_rank()}: {failures}"
        )


def make_training_step(args, mode, model, optimizer, batches, profile_path=None, trace_path=None):
    """Build once; benchmark, profiler and calibration execute the same full step."""
    if mode["schedule"] == "slackpipe":
        forward_backward = get_forward_backward_func(
            pipeline_schedule="slackpipe",
            slackpipe_plan_path=str(args.plan),
            slackpipe_trace_path=str(trace_path) if trace_path is not None else None,
            slackpipe_profile_path=str(profile_path) if profile_path is not None else None,
            slackpipe_runtime=args.slackpipe_runtime,
            slackpipe_transport=args.slackpipe_transport,
            slackpipe_enable_nvtx=not args.disable_slackpipe_nvtx,
        )
    else:
        forward_backward = get_forward_backward_func(pp_size=2, vp_size=2)

    def step(iteration):
        optimizer.zero_grad(set_to_none=True)
        losses = forward_backward(
            forward_step_func=_forward_step_func,
            data_iterator=[_batch_iterator(batches) for _ in range(len(model))],
            model=model,
            num_microbatches=args.num_microbatches,
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            forward_only=False,
        )
        optimizer.step()
        return losses

    return step


def _run_iteration(args, mode, model, optimizer, batches, profile_path=None, trace_path=None):
    losses = make_training_step(args, mode, model, optimizer, batches, profile_path, trace_path)(0)
    _assert_finite(losses, _logical_params(model))
    return losses


def _stats(values):
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "stddev_ms": statistics.pstdev(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _parameter_count(model):
    local = torch.tensor(
        [sum(param.numel() for param in _logical_params(model).values())],
        dtype=torch.long,
        device="cuda",
    )
    dist.all_reduce(local, op=dist.ReduceOp.SUM)
    return int(local.item())


def _benchmark_mode(args, mode, rank):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    dist.barrier()
    model = _build_model(args, mode)
    optimizer = torch.optim.SGD(_logical_params(model).values(), lr=args.learning_rate)
    batches = _make_batches(args)
    param_count = _parameter_count(model)
    transport_construction_seconds = 0.0
    transport_construction_device_bytes = 0
    if mode["schedule"] == "slackpipe":
        torch.cuda.synchronize()
        free_before = torch.cuda.mem_get_info()[0]
        started = time.perf_counter()
        _get_slackpipe_runtime(
            str(args.plan),
            pp_rank=rank,
            pipeline_tensor_shape=(args.seq_length, args.micro_batch_size, args.hidden_size),
            pipeline_tensor_dtype=torch.float32,
            pipeline_tensor_device=torch.device("cuda"),
            num_microbatches=args.num_microbatches,
            forward_only=False,
            enable_fast_path=args.slackpipe_runtime == "fast",
            transport=args.slackpipe_transport,
        )
        torch.cuda.synchronize()
        transport_construction_seconds = time.perf_counter() - started
        transport_construction_device_bytes = free_before - torch.cuda.mem_get_info()[0]

    params = _logical_params(model)
    collection = collect_steps(
        step=make_training_step(args, mode, model, optimizer, batches),
        optimizer=optimizer,
        options=CollectionOptions(
            warmup=args.warmup_iterations, iterations=args.iterations, run_id=args.run_id
        ),
        output=args.output_dir,
        method=mode["name"],
        transport=args.slackpipe_transport if mode["schedule"] == "slackpipe" else "megatron-p2p",
        config=dict(
            pp=2,
            num_layers=args.num_layers,
            num_microbatches=args.num_microbatches,
            layout=mode["layout"],
            hidden_size=args.hidden_size,
            heads=args.num_attention_heads,
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            vocab_size=args.vocab_size,
            seed=args.seed,
            learning_rate=args.learning_rate,
            model_sha256=(
                hashlib.sha256(args.heterogeneous_config.read_bytes()).hexdigest()
                if args.heterogeneous_config
                else "homogeneous"
            ),
            initialization="per-global-logical-parameter seed; partition independent",
            dtype="fp32",
            plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        ),
        validate=lambda losses: _assert_finite(losses, params),
    )
    cuda_times_ms = [s["cuda_elapsed_ms"] for s in collection["samples"]]
    cpu_times_ms = [s["cpu_dispatch_ms"] for s in collection["samples"]]
    total_cpu_ms = collection["continuous_wall_ms"]
    free_device_bytes, total_device_bytes = torch.cuda.mem_get_info()

    result = {
        "mode": mode["name"],
        "collection": collection,
        "rank": rank,
        "layout": mode["layout"],
        "schedule": mode["schedule"],
        "transport": args.slackpipe_transport,
        "transport_statistics": slackpipe_transport_statistics(),
        "transport_construction_seconds": transport_construction_seconds,
        "transport_construction_device_bytes": transport_construction_device_bytes,
        "local_layer_counts": [len(chunk.decoder.layers) for chunk in model],
        "parameter_count": param_count,
        "iteration_cuda_times_ms": cuda_times_ms,
        "iteration_cpu_times_ms": cpu_times_ms,
        "total_cpu_wall_time_ms": total_cpu_ms,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "cuda_device_used_bytes": total_device_bytes - free_device_bytes,
    }
    del batches
    del optimizer
    del model
    gc.collect()
    clear_slackpipe_plan_cache()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    dist.barrier()
    return result


def _calibrate_cost_profile(args, plan, rank):
    from megatron.core.pipeline_parallel.slackpipe.profile_integrity import (
        publish_profile,
        tag_calibration_events,
    )
    from megatron.core.pipeline_parallel.slackpipe.stage_residual import residual_policy_from_args

    mode = {
        "name": "cost_calibration_solver_split_interleaved_1f1b",
        "schedule": "default",
        "layout": derive_pipeline_model_parallel_layout(plan),
    }
    torch.cuda.empty_cache()
    dist.barrier()
    model = _build_model(args, mode)
    optimizer = torch.optim.SGD(_logical_params(model).values(), lr=args.learning_rate)
    batches = _make_batches(args)
    step = make_training_step(args, mode, model, optimizer, batches)
    before_warmup = memory_sample()
    for _ in range(args.calibration_warmup_iterations):
        losses = step(0)
    _assert_finite(losses, _logical_params(model))
    provenance = dict(
        collection_mode="calibration",
        rank=rank,
        run_id=args.run_id or str(args.output_dir),
        seed=args.seed,
        optimizer=dict(name="SGD", learning_rate=args.learning_rate),
        environment=environment_metadata(),
        profiler=profiler_settings(CollectionOptions()),
        memory_before_warmup=before_warmup,
        memory_after_warmup=memory_sample(),
        warmup_iterations=list(range(args.calibration_warmup_iterations)),
        measurement_iterations=list(
            range(
                args.calibration_warmup_iterations,
                args.calibration_warmup_iterations + args.calibration_iterations,
            )
        ),
    )

    events = []
    for iteration in provenance["measurement_iterations"]:
        begin_slackpipe_cost_calibration_iteration(iteration)
        try:
            losses = step(iteration)
        finally:
            events.extend(end_slackpipe_cost_calibration_iteration())
    _assert_finite(losses, _logical_params(model))

    tag_calibration_events(
        events,
        group="partition0",
        ranges=plan.stage_layer_ranges,
        worker=rank,
        raw_path=args.output_dir / "calibration_events.json",
        configuration=dict(
            model=dict(
                num_layers=args.num_layers,
                heterogeneous_config=(
                    json.loads(args.heterogeneous_config.read_text())
                    if args.heterogeneous_config
                    else None
                ),
                hidden_size=args.hidden_size,
                heads=args.num_attention_heads,
                seq_length=args.seq_length,
                vocab_size=args.vocab_size,
                micro_batch_size=args.micro_batch_size,
                microbatches=args.num_microbatches,
            ),
            seed=args.seed,
            learning_rate=args.learning_rate,
            warmups=args.calibration_warmup_iterations,
            iterations=args.calibration_iterations,
            precision="fp32",
            pp=2,
            vpp=2,
            environment=provenance["environment"],
            timing="synchronized-stage-wall-time-v1",
        ),
    )
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, events)
    provenance["memory_end"] = memory_sample()
    all_provenance = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(all_provenance, provenance)
    if rank == 0:
        all_events = [event for rank_events in gathered for event in rank_events]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "calibration_events.json").write_text(
            json.dumps(all_events, indent=2, sort_keys=True), encoding="utf-8"
        )
        profile = build_cost_profile(
            events=all_events,
            model_config={
                "num_layers": args.num_layers,
                "hidden_size": args.hidden_size,
                "num_attention_heads": args.num_attention_heads,
                "sequence_length": args.seq_length,
                "micro_batch_size": args.micro_batch_size,
                "vocab_size": args.vocab_size,
                "dropout": 0.0,
                "dtype": "fp32",
            },
            parallel_config={"pp": 2, "vpp": 2, "tp": 1, "dp": 1, "cp": 1},
            layer_split=list(plan.layer_split),
            num_microbatches=args.num_microbatches,
            iteration_start=args.calibration_warmup_iterations,
            iteration_end=args.calibration_warmup_iterations + args.calibration_iterations - 1,
            estimator=args.shared_slope_estimator,
            percentile_value=args.shared_slope_percentile,
            residual_policy=residual_policy_from_args(args),
        )
        profile["collection"] = dict(provenance, rank_metadata=all_provenance)
        (args.output_dir / "observations.json").write_text(
            json.dumps(profile["observed_stages"], indent=2)
        )
        publish_profile(args.emit_cost_profile, profile)
    del batches
    del optimizer
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    dist.barrier()
    if rank == 0 and not args.emit_cost_profile.is_file():
        raise SystemExit(2)


def _write_summary(args, all_results, plan):
    by_mode = {}
    for rank_result in all_results:
        by_mode.setdefault(rank_result["mode"], []).append(rank_result)

    rows = []
    summary = {
        "model_config": {
            "num_layers": args.num_layers,
            "hidden_size": args.hidden_size,
            "num_attention_heads": args.num_attention_heads,
            "sequence_length": args.seq_length,
            "micro_batch_size": args.micro_batch_size,
            "num_microbatches": args.num_microbatches,
            "global_batch_size": args.micro_batch_size * args.num_microbatches,
            "vocab_size": args.vocab_size,
        },
        "plan": {
            "path": str(args.plan),
            "solver_status": plan.solver_status,
            "layer_split": list(plan.layer_split),
            "stage_to_worker": list(plan.stage_to_worker),
            "predicted_makespan": plan.predicted_makespan,
            "forward_costs": list(plan.forward_costs),
            "backward_costs": list(plan.backward_costs),
        },
        "modes": {},
    }
    samples_per_iteration = args.micro_batch_size * args.num_microbatches
    tokens_per_iteration = samples_per_iteration * args.seq_length
    for mode_name, rank_results in by_mode.items():
        per_rank_times = [r["iteration_cuda_times_ms"] for r in rank_results]
        collected_summary = summarize_rank_samples([r["collection"] for r in rank_results])
        global_times = [max(values) for values in zip(*per_rank_times)]
        assert global_times == collected_summary["max_rank_iteration_ms"]
        stats = _stats(global_times)
        mean_seconds = stats["mean_ms"] / 1000.0
        stats["samples_per_second"] = samples_per_iteration / mean_seconds
        stats["tokens_per_second"] = tokens_per_iteration / mean_seconds
        stats["peak_allocated_bytes"] = max(r["cuda_peak_allocated_bytes"] for r in rank_results)
        stats["peak_reserved_bytes"] = max(r["cuda_peak_reserved_bytes"] for r in rank_results)
        stats["parameter_count"] = rank_results[0]["parameter_count"]
        stats["rank_results"] = rank_results
        summary["modes"][mode_name] = stats
        rows.append({"mode": mode_name, **{k: v for k, v in stats.items() if k != "rank_results"}})

    summary["improvements"] = {}
    required_modes = {
        "A_uniform_interleaved_1f1b",
        "B_solver_split_interleaved_1f1b",
        "C_slackpipe_solver_order",
    }
    if required_modes.issubset(summary["modes"]):
        a = summary["modes"]["A_uniform_interleaved_1f1b"]["mean_ms"]
        b = summary["modes"]["B_solver_split_interleaved_1f1b"]["mean_ms"]
        c = summary["modes"]["C_slackpipe_solver_order"]["mean_ms"]
        predicted_ratio = None
        if args.uniform_predicted_makespan is not None and plan.predicted_makespan is not None:
            predicted_ratio = plan.predicted_makespan / args.uniform_predicted_makespan
        summary["improvements"] = {
            "overall_slackpipe": (a - c) / a,
            "partition_only": (a - b) / a,
            "schedule_with_partition_controlled": (b - c) / b,
            "solver_predicted_makespan_ratio_vs_uniform": predicted_ratio,
            "measured_time_ratio_slackpipe_vs_uniform": c / a,
            "measured_time_ratio_slackpipe_vs_partition_controlled": c / b,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    with open(args.output_dir / "benchmark_summary.csv", "w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(
            out,
            fieldnames=[
                "mode",
                "mean_ms",
                "median_ms",
                "stddev_ms",
                "min_ms",
                "max_ms",
                "samples_per_second",
                "tokens_per_second",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
                "parameter_count",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    from megatron.core.pipeline_parallel.slackpipe.stage_residual import add_residual_arguments

    parser = argparse.ArgumentParser()
    add_residual_arguments(parser)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--heterogeneous-config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup-iterations", type=int, default=20)
    parser.add_argument("--run-id")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-attention-heads", type=int, default=8)
    parser.add_argument("--seq-length", type=int, default=128)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--uniform-predicted-makespan", type=float, default=None)
    parser.add_argument("--slackpipe-runtime", choices=["debug", "fast"], default="fast")
    parser.add_argument(
        "--slackpipe-transport", choices=["nccl-p2p", "nccl-rma"], default="nccl-p2p"
    )
    parser.add_argument("--disable-slackpipe-nvtx", action="store_true")
    parser.add_argument("--order-rotation", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--strict-fp32", action="store_true")
    parser.add_argument("--profile-measured-iterations", action="store_true")
    parser.add_argument("--trace-measured-iterations", action="store_true")
    parser.add_argument("--emit-cost-profile", type=Path, default=None)
    parser.add_argument("--calibration-warmup-iterations", type=int, default=20)
    parser.add_argument("--calibration-iterations", type=int, default=10)
    parser.add_argument(
        "--shared-slope-estimator", choices=["min", "median", "percentile"], default="min"
    )
    parser.add_argument("--shared-slope-percentile", type=float, default=20.0)
    parser.add_argument(
        "--only-mode",
        choices=[
            "A_uniform_interleaved_1f1b",
            "B_solver_split_interleaved_1f1b",
            "C_slackpipe_solver_order",
        ],
        default=None,
        help="Run only one benchmark mode. Intended for short profiler captures.",
    )
    args = parser.parse_args()
    if (args.output_dir / "benchmark_summary.json").exists() or (
        args.emit_cost_profile is not None and args.emit_cost_profile.exists()
    ):
        parser.error("Refusing to overwrite recorded results; use a fresh output directory")
    if args.profile_measured_iterations or args.trace_measured_iterations:
        parser.error(
            "Benchmark is unprofiled; use tools.capture_schedule_trace --collection-mode timeline or memory"
        )
    if (
        min(
            args.warmup_iterations,
            args.iterations,
            args.calibration_warmup_iterations,
            args.calibration_iterations,
        )
        < 1
    ):
        parser.error("Warmup and measurement lengths must be positive")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if args.strict_fp32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    clear_nvte_env_vars()
    plan = load_slackpipe_plan(args.plan, pipeline_model_parallel_size=2)
    args.num_layers = plan.num_layers
    args.num_microbatches = plan.num_microbatches

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2,
    )
    parallel_state.set_virtual_pipeline_model_parallel_world_size(2)

    uniform_split = [args.num_layers // 4] * 4
    for i in range(args.num_layers % 4):
        uniform_split[i] += 1
    modes = [
        {
            "name": "A_uniform_interleaved_1f1b",
            "schedule": "default",
            "layout": _layout_from_split(uniform_split),
        },
        {
            "name": "B_solver_split_interleaved_1f1b",
            "schedule": "default",
            "layout": derive_pipeline_model_parallel_layout(plan),
        },
        {
            "name": "C_slackpipe_solver_order",
            "schedule": "slackpipe",
            "layout": derive_pipeline_model_parallel_layout(plan),
        },
    ]
    if args.only_mode is not None:
        modes = [mode for mode in modes if mode["name"] == args.only_mode]
    else:
        modes = modes[args.order_rotation :] + modes[: args.order_rotation]

    try:
        if args.emit_cost_profile is not None:
            _calibrate_cost_profile(args, plan, rank)
            return
        local_results = []
        for mode in modes:
            local_results.append(_benchmark_mode(args, mode, rank))
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, local_results)
        if rank == 0:
            all_results = [item for rank_items in gathered for item in rank_items]
            _write_summary(args, all_results, plan)
    finally:
        shutdown_slackpipe_runtime()
        parallel_state.destroy_model_parallel()
        Utils.inited = False
        parallel_state.set_virtual_pipeline_model_parallel_world_size(None)
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
