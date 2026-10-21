# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Native hybrid calibration, benchmark and trace worker (launched by torchrun).

The default architecture is the released random-init 8B model. --tiny is solely
for local harness tests; it cannot produce a certificate for a real pod run.
"""

import gc
import hashlib
import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
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
)
from megatron.core.pipeline_parallel.slackpipe.cost_profile import stage_role
from megatron.core.pipeline_parallel.slackpipe.hybrid import partition_hybrid_pattern
from megatron.core.pipeline_parallel.slackpipe.manifest import validate_chunk_layers
from megatron.core.pipeline_parallel.slackpipe.plan import parse_slackpipe_plan
from megatron.core.pipeline_parallel.slackpipe.schedule import (
    shutdown_slackpipe_runtime,
    slackpipe_transport_statistics,
)
from megatron.core.pipeline_parallel.slackpipe.timing_diagnostic import (
    ComputeTimingDiagnostic,
    DiagnosticP2PCommunicator,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from tools.slackpipe_eval_common import write_json
from tools.slackpipe_model_support import (
    _batch_iterator,
    _build_model,
    _forward_step_func,
    _logical_named_parameters,
    _make_batches,
)


def uniform_cuts(layers: int, stages: int) -> tuple:
    if layers < stages:
        raise ValueError("At least one layer per stage is required")
    return tuple(layers * s // stages for s in range(stages + 1))


def calibration_partitions(manifest: dict, stages: int, minimum: int = 6) -> tuple:
    """Choose full-rank, near-uniform designs without enumerating L choose N.

    Moving each boundary by at most two layers keeps calibration memory close
    to the smoke-tested uniform partition. Refuse unidentifiable architectures.
    """
    classes = [r["config_class"] for r in manifest["layers"]]
    keys = sorted(set(classes))
    base = uniform_cuts(len(classes), stages)
    candidates = {base}
    for s in range(1, stages):
        for shift in (-2, -1, 1, 2):
            cuts = list(base)
            cuts[s] += shift
            if all(a < b for a, b in zip(cuts, cuts[1:])):
                candidates.add(tuple(cuts))

    def design(cuts):
        return np.asarray(
            [
                [classes[a:b].count(k) for k in keys]
                + [int(stage_role(s, stages) == r) for r in ("first", "middle", "last")]
                for s, (a, b) in enumerate(zip(cuts, cuts[1:]))
            ],
            dtype=float,
        )

    selected = [base]
    columns = len(keys) + 3
    while True:
        matrix = np.concatenate([design(c) for c in selected])
        rank = np.linalg.matrix_rank(matrix)
        if rank == columns and len(selected) >= minimum:
            return selected, dict(
                rank=int(rank), columns=columns, condition=float(np.linalg.cond(matrix))
            )
        remaining = candidates - set(selected)
        if not remaining:
            raise ValueError("Rank-deficient calibration design; refusing to fit")

        def score(c):
            aug = np.concatenate([matrix, design(c)])
            return (-np.linalg.matrix_rank(aug), np.linalg.cond(aug), c)

        selected.append(min(remaining, key=score))


def construction_plan(cuts: tuple, pp: int, microbatches: int, manifest: dict):
    """Structural construction fixture, never a replacement for solver ordering."""
    stages = len(cuts) - 1
    operations = [[] for _ in range(pp)]
    for kind, order in (("F", range(stages)), ("B", range(stages - 1, -1, -1))):
        for b in range(microbatches):
            for s in order:
                operations[s % pp].append(dict(kind=kind, microbatch=b, stage=s))
    return parse_slackpipe_plan(
        dict(
            schema_version="slackpipe.plan.v2",
            num_layers=cuts[-1],
            num_workers=pp,
            num_stages=stages,
            num_microbatches=microbatches,
            layer_cuts=cuts,
            stage_to_worker=[s % pp for s in range(stages)],
            operations=operations,
            model_manifest_hash=manifest["manifest_hash"],
        )
    )


def initialize_parameters(parameters: dict, seed: int, layers: int) -> None:
    """Partition-independent mock weights; preserve stable Mamba state scales."""
    with torch.no_grad():
        for name, param in parameters.items():
            generator = torch.Generator(device=param.device)
            generator.manual_seed(
                seed + int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
            )
            if name.endswith("A_log"):
                param.uniform_(1, 16, generator=generator).log_()
            elif name.endswith("dt_bias"):
                dt = (
                    torch.empty_like(param, dtype=torch.float32)
                    .uniform_(-6.907755, -2.302585, generator=generator)
                    .exp_()
                )
                param.copy_(dt + torch.log(-torch.expm1(-dt)))
            elif name.endswith(".D") or (param.ndim == 1 and name.endswith("weight")):
                param.fill_(1)
            elif name.endswith("bias"):
                param.zero_()
            else:
                std = 0.02 / (
                    (2 * layers) ** 0.5
                    if name.endswith(("out_proj.weight", "linear_proj.weight", "linear_fc2.weight"))
                    else 1
                )
                param.normal_(0, std, generator=generator)


def run_partition(args, config, manifest, plan, output: Path, collect: str) -> list:
    rank = dist.get_rank()
    schedule = "slackpipe" if args.mode == "slackpipe" else "default"
    config.pipeline_schedule = schedule
    config.deallocate_pipeline_outputs = schedule == "default"
    pattern = (
        partition_hybrid_pattern(config.slackpipe_hybrid_pattern, plan)
        if getattr(config, "slackpipe_hybrid_pattern", None)
        else None
    )
    vpp = getattr(args, "vpp", args.stages // args.pp)
    diagnose = getattr(args, "diagnose_compute_timing", False)
    stage_collection = getattr(args, "stage_collection", None)
    if stage_collection and (
        diagnose or collect != "calibrate" or schedule != "default" or vpp < 2
    ):
        raise ValueError(
            "Stage collection requires native interleaved calibration, without legacy diagnosis"
        )
    if stage_collection == "layers" and (
        config.recompute_granularity
        or getattr(config, "cuda_graph_impl", "none") != "none"
        or getattr(config, "enable_cuda_graph", False)
    ):
        raise ValueError("Layer graph diagnosis cannot preserve recompute/CUDA graph execution")
    serialized = getattr(args, "diagnostic_execution", "normal") == "serialized"
    timeline = getattr(args, "diagnostic_timeline", False)
    nvtx = getattr(args, "diagnostic_nvtx", False)
    if (serialized or timeline or nvtx) and not diagnose:
        raise ValueError("Overlap controls require --diagnose-compute-timing")
    if diagnose and (collect != "calibrate" or schedule != "default" or vpp < 2):
        raise ValueError("Timing diagnosis requires native interleaved calibration (VPP >= 2)")

    def provider(
        pre_process=True, post_process=True, vp_stage=None, config=None, pg_collection=None
    ):
        if pattern is None:
            return GPTModel(
                config=config,
                transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
                vocab_size=args.vocab_size,
                max_sequence_length=args.seq_length,
                pre_process=pre_process,
                post_process=post_process,
                vp_stage=vp_stage,
                pg_collection=pg_collection,
                position_embedding_type="rope",
                share_embeddings_and_output_weights=False,
            )
        return HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=args.vocab_size,
            max_sequence_length=args.seq_length,
            hybrid_layer_pattern=pattern,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
            pg_collection=pg_collection,
            position_embedding_type="none",
            share_embeddings_and_output_weights=False,
        )

    model = _build_model(
        config,
        pipeline_schedule=schedule,
        pp_size=args.pp,
        vpp=vpp,
        layout=getattr(args, "layout", None),
        provider=provider,
    )
    records = validate_chunk_layers(plan, model, rank)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / f"layers.rank{rank}.json", records)
    params = _logical_named_parameters(model)
    if hasattr(args, "exact_parameter_count"):
        actual = torch.tensor(
            sum(p.numel() for p in params.values()), device="cuda", dtype=torch.int64
        )
        dist.all_reduce(actual)
        if actual.item() != args.exact_parameter_count:
            raise ValueError(
                f"Actual parameter count {actual.item()} != specification {args.exact_parameter_count}"
            )
    initialize_parameters(params, args.seed, config.num_layers)
    optimizer = torch.optim.SGD(params.values(), lr=args.learning_rate)
    batches = _make_batches(args)
    fb = (
        get_forward_backward_func(
            pipeline_schedule="slackpipe",
            slackpipe_plan_path=str(args.plan),
            slackpipe_transport=args.transport,
            slackpipe_runtime="fast",
            slackpipe_enable_nvtx=collect in ("trace", "timeline", "memory"),
        )
        if schedule == "slackpipe"
        else get_forward_backward_func(pp_size=args.pp, vp_size=vpp)
    )

    smoke_updates = []
    diagnostic = None
    schedule_kwargs = {}
    if stage_collection in ("events", "layers"):
        from megatron.core.pipeline_parallel.slackpipe.stage_measurement import StageEventCollector

        diagnostic = StageEventCollector(
            (args.warmups + args.iterations) * (2 * args.num_microbatches * len(model) + 1),
            timeline=stage_collection == "layers",
        )
        schedule_kwargs = dict(
            p2p_communicator=DiagnosticP2PCommunicator(
                parallel_state.get_pipeline_model_parallel_group(), config, diagnostic
            ),
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(
                ["tp", "cp", "embd", "pos_embd", "pp", "dp_cp", "tp_dp_cp"]
            ),
        )
    if diagnose:
        diagnostic = ComputeTimingDiagnostic(
            2 * args.num_microbatches * len(model),
            serialized=serialized,
            timeline=timeline,
            nvtx=nvtx,
        )
        schedule_kwargs = dict(
            p2p_communicator=DiagnosticP2PCommunicator(
                parallel_state.get_pipeline_model_parallel_group(), config, diagnostic
            ),
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(
                ["tp", "cp", "embd", "pos_embd", "pp", "dp_cp", "tp_dp_cp"]
            ),
        )

    def iteration():
        optimizer.zero_grad(set_to_none=True)
        losses = fb(
            forward_step_func=_forward_step_func,
            data_iterator=[_batch_iterator(batches) for _ in model],
            model=model,
            num_microbatches=args.num_microbatches,
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            forward_only=False,
            **schedule_kwargs,
        )
        probes = []
        if getattr(args, "verify_update", False):
            for p in params.values():
                if p.grad is not None:
                    index = p.grad.detach().abs().reshape(-1).argmax()
                    probes.append((p, index, p.detach().reshape(-1)[index].clone()))
        if stage_collection in ("events", "layers"):
            diagnostic.optimizer(optimizer.step)
        else:
            optimizer.step()
        if probes:
            smoke_updates.append(
                torch.stack([p.detach().reshape(-1)[i] != before for p, i, before in probes]).any()
            )
        return losses

    def assert_finite(losses):
        flags = [torch.isfinite(p).all() for p in params.values()]
        flags += [torch.isfinite(p.grad).all() for p in params.values() if p.grad is not None]
        flags += [torch.isfinite(l["loss"]).all() for l in losses]
        if hasattr(args, "eval_metadata"):
            expected_losses = args.num_microbatches if rank == (args.stages - 1) % args.pp else 0
            flags.append(torch.tensor(len(losses) == expected_losses, device="cuda"))
            flags.append(
                torch.tensor(all(p.grad is not None for p in params.values()), device="cuda")
            )
        finite = torch.stack(flags).all().to(torch.int32)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise RuntimeError("Non-finite loss/gradient/parameter")
        if getattr(args, "verify_update", False):
            changed = torch.stack(smoke_updates).any().to(torch.int32)
            dist.all_reduce(changed, op=dist.ReduceOp.MIN)
            if not changed.item():
                raise RuntimeError(
                    "Smoke requires a representable SGD parameter update on every rank"
                )

    events, measured = [], {}
    if stage_collection:
        from megatron.core.pipeline_parallel.slackpipe.stage_measurement import collect_stage_window

        events, measured = collect_stage_window(
            step=iteration,
            collector=diagnostic,
            args=args,
            output=output,
            validate=assert_finite,
            model=model,
            plan=plan,
            manifest=manifest,
        )
        for row in events:
            row.update(
                run_id=args.run_id,
                partition_id=output.name,
                worker=rank,
                stage_layer_range=list(plan.stage_layer_ranges[row["logical_stage"]]),
            )
        write_json(output / f"stage_samples.rank{rank}.json", events)
    elif collect == "calibrate":
        trace_path = (
            output
            / f"overlap.{args.run_id or 'run'}.{('serialized' if serialized else 'normal')}.rank{rank}.json"
        )

        def observed_iteration(i):
            region = (
                diagnostic.region(f"ScheduleTrace/step/i{i}/s-1/b-1") if diagnose else nullcontext()
            )
            with region:
                begin_slackpipe_cost_calibration_iteration(i, diagnostic=diagnostic)
                try:
                    result = iteration()
                finally:
                    rows = end_slackpipe_cost_calibration_iteration()
            return result, rows

        before_warmup = memory_sample()
        with diagnostic.capture(trace_path) if diagnose else nullcontext():
            for i in range(args.warmups):
                # Warm the exact diagnostic execution policy, including optimizer.
                losses = observed_iteration(i)[0] if diagnose else iteration()
            assert_finite(losses)
            measured = dict(
                collection_mode="calibration",
                run_id=args.run_id or str(args.output),
                rank=rank,
                partition_output=str(output.relative_to(args.output)),
                stage_layer_ranges=list(plan.stage_layer_ranges),
                seed=args.seed,
                optimizer=dict(name="SGD", learning_rate=args.learning_rate),
                environment=environment_metadata(),
                profiler=profiler_settings(CollectionOptions()),
                warmup_iterations=list(range(args.warmups)),
                measurement_iterations=list(range(args.warmups, args.warmups + args.iterations)),
                memory_before_warmup=before_warmup,
                memory_after_warmup=memory_sample(),
            )
            for i in measured["measurement_iterations"]:
                losses, rows = observed_iteration(i)
                events.extend(rows)
                assert_finite(losses)
        if timeline:
            measured["profiler"] = dict(
                enabled=True,
                activities=["CPU", "CUDA"],
                record_shapes=False,
                with_stack=False,
                profile_memory=False,
                schedule="continuous single partition capture",
                path=str(trace_path),
                captured_iterations=list(range(args.warmups + args.iterations)),
                serialization="after the complete partition measurement window",
            )
        measured["memory_end"] = memory_sample()
        if diagnose and len(events) != args.iterations * 2 * args.num_microbatches * len(model):
            raise RuntimeError("Incomplete diagnostic operation coverage")
        measured["timing_diagnostic"] = (
            dict(
                schema_version="slackpipe.compute_timing_diagnostic.v1",
                diagnostic_only=True,
                execution="serialized" if serialized else "normal",
                timeline=timeline,
                nvtx=nvtx,
                estimator="compute-stream-events-diagnostic-v1",
                compute_gpu_ms="current-stream event envelope; includes in-region waits and launch gaps; not kernel sum",
                cpu_wall_ms="host compute closure without added completion wait; not existing stage wall time",
                existing_stage_wall_ms="unavailable in this run: legacy device-wide synchronization would perturb the probe",
                p2p_wall_ms="inclusive native P2P call CPU time; contains native waits/sync; not additive to GPU envelopes",
                wait_synchronization_ms="per-operation decomposition unavailable; completion_wait_ms measured once after optimizer",
                side_stream_coverage="only work joined to compute stream is covered; full-step drain does not extend per-op intervals",
                iterations=diagnostic.iterations,
            )
            if diagnose
            else None
        )
    else:
        metadata = dict(
            pp=args.pp,
            num_stages=args.stages,
            num_layers=config.num_layers,
            num_microbatches=args.num_microbatches,
            layer_split=list(plan.layer_split),
            hidden_size=config.hidden_size,
            heads=config.num_attention_heads,
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            vocab_size=args.vocab_size,
            seed=args.seed,
            learning_rate=args.learning_rate,
            dtype=args.precision,
            tf32=False,
            dropout=0,
            tp=1,
            dp=1,
            cp=1,
            model="Nemotron-H-8B-Base-8K" if not args.tiny else "tiny-hybrid",
            model_sha256=manifest["manifest_hash"],
            host=os.uname().nodename,
            plan_sha256=(
                hashlib.sha256(args.plan.read_bytes()).hexdigest()
                if args.mode != "baseline"
                else None
            ),
        )
        metadata.update(getattr(args, "eval_metadata", {}))
        measured = collect_steps(
            step=lambda i: iteration(),
            optimizer=optimizer,
            options=CollectionOptions(
                mode="timeline" if collect == "trace" else collect,
                warmup=args.warmups,
                iterations=args.iterations,
                wait=args.profiler_wait,
                profiler_warmup=args.profiler_warmup,
                active=args.profiler_active,
                repeat=args.profiler_repeat,
                run_id=args.run_id,
                history_entries=args.memory_history_entries,
            ),
            output=output,
            method=getattr(args, "schedule", args.mode),
            transport=args.transport if schedule == "slackpipe" else "megatron-p2p",
            config=metadata,
            plan=plan if schedule == "slackpipe" else None,
            validate=assert_finite,
        )
        measured["iteration_ms"] = [s["cuda_elapsed_ms"] for s in measured["samples"]]
        measured["wall_ms"] = measured["continuous_wall_ms"]
    if getattr(args, "verify_update", False):
        measured["smoke_checks"] = dict(
            finite_loss_gradients_parameters=True, optimizer_update=True
        )
    if collect == "calibrate" and not stage_collection:
        from megatron.core.pipeline_parallel.slackpipe.profile_integrity import (
            tag_calibration_events,
        )

        configuration = dict(
            manifest=manifest["manifest_hash"],
            pp=args.pp,
            stages=args.stages,
            microbatches=args.num_microbatches,
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            precision=args.precision,
            warmups=args.warmups,
            iterations=args.iterations,
            seed=args.seed,
            learning_rate=args.learning_rate,
            environment=measured["environment"],
            schedule="native",
            batch_p2p_comm=config.batch_p2p_comm,
            batch_p2p_sync=config.batch_p2p_sync,
            overlap_p2p_comm=config.overlap_p2p_comm,
        )
        measured["calibration_configuration"] = configuration
        if serialized or timeline or nvtx:
            configuration["overlap_diagnostic"] = dict(
                execution="serialized" if serialized else "normal", timeline=timeline
            )
            if nvtx:
                configuration["overlap_diagnostic"]["nvtx"] = True
        basename = "timing_events" if diagnose else "calibration_events"
        tag_calibration_events(
            events,
            group=output.name,
            ranges=plan.stage_layer_ranges,
            worker=rank,
            attempt=getattr(args, "profiling_attempt", 0),
            raw_path=output / f"{basename}.rank{rank}.json",
            configuration=dict(
                **configuration,
                timing=(
                    "compute-stream-events-diagnostic-v1"
                    if diagnose
                    else "synchronized-stage-wall-time-v1"
                ),
            ),
        )
        for event in events:
            begin, end = event["stage_layer_range"]
            event["layer_composition"] = [
                layer["config_class"] for layer in manifest["layers"][begin:end]
            ]
            if diagnose:
                event["co_resident_stages"] = [
                    dict(stage=s, layer_range=list(plan.stage_layer_ranges[s]))
                    for s in range(args.stages)
                    if s % args.pp == rank and s != event["logical_stage"]
                ]
        write_json(output / f"{basename}.rank{rank}.json", events)
    write_json(output / f"result.rank{rank}.json", measured)
    if stage_collection and getattr(args, "stage_save_state", False):
        torch.save(
            dict(
                parameters={k: v.detach().cpu() for k, v in params.items()},
                gradients={
                    k: v.grad.detach().cpu() if v.grad is not None else None
                    for k, v in params.items()
                },
            ),
            output / f"state.rank{rank}.pt",
        )
    gathered = [None] * args.pp
    dist.all_gather_object(gathered, events)
    shutdown_slackpipe_runtime()
    if slackpipe_transport_statistics():
        raise RuntimeError("Transport state remains after shutdown")
    optimizer = params = model = batches = fb = None
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    return [event for worker in gathered for event in worker]
