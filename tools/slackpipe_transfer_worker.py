# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Explicit diagnostic entry; production worker and estimators remain unchanged."""

import argparse
import gc
import json
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.pipeline_parallel import schedules
from megatron.core.pipeline_parallel.slackpipe.collection import environment_metadata, memory_sample
from megatron.core.pipeline_parallel.slackpipe.stage_measurement import (
    StageEventCollector,
    tensor_description,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.spec_utils import build_module
from tools.slackpipe_diagnose_composition import context_snapshot
from tools.slackpipe_eval_config import load_model, transformer_config
from tools.slackpipe_isolated_profile import no_communication
from tools.slackpipe_stage_calibration import write_new
from tools.slackpipe_training import initialize_parameters


class SynchronousStageCollector(StageEventCollector):
    """Legacy pre/post-device-drain boundaries plus distinct GPU event intervals.

    serialized stays False: do NOT change native P2P communication policy.
    Only F/B, not optimizer, gets the legacy synchronization treatment.
    """

    def compute(self, compute, identity):
        before = time.perf_counter()
        torch.cuda.synchronize()
        pre_ms = (time.perf_counter() - before) * 1000
        before = time.perf_counter()
        result = super().compute(compute, identity)
        torch.cuda.synchronize()
        self.pending[-1].update(
            synchronized_wall_ms=(time.perf_counter() - before) * 1000,
            pre_sync_wall_ms=pre_ms,
            added_device_syncs=2,
        )
        return result


def clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, (tuple, list)):
        return [clone_tree(v) for v in value]
    if isinstance(value, dict):
        return {k: clone_tree(v) for k, v in value.items()}
    return value


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.cpu()
    if isinstance(value, (tuple, list)):
        return [cpu_tree(v) for v in value]
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    return value


def transfer_window(*, step, collector, args, output, validate, model, plan, manifest):
    """Same complete SGD step in all modes, no per-step completion synchronization."""
    rank = dist.get_rank()
    numerical = {}
    original = schedules._run_with_slackpipe_cost_calibration
    current_iteration = -1

    def capture(**kw):
        result = original(**kw)
        if args.transfer_correctness and current_iteration == args.warmups + args.iterations - 1:
            stage = kw["model_chunk_id"] * 2 + rank
            key = f"{kw['phase']}/s{stage}/b{kw['microbatch']}"
            if key in numerical:
                raise ValueError("Duplicate correctness identity")
            numerical[key] = clone_tree(result)
        return result

    def observed(i):
        nonlocal current_iteration
        current_iteration = i
        if collector is None:
            return step()
        schedules.begin_slackpipe_cost_calibration_iteration(i, diagnostic=collector)
        try:
            return step()
        finally:
            schedules.end_slackpipe_cost_calibration_iteration()

    meta = dict(
        schema_version="slackpipe.transfer_window.v1",
        condition=args.transfer_condition,
        group_size=args.transfer_group,
        run_id=args.run_id,
        rank=rank,
        split=list(plan.layer_split),
        environment=environment_metadata(),
        context_before=context_snapshot(),
        configuration=args.eval_metadata,
        profiler=dict(enabled=False),
        correctness_only=args.transfer_correctness,
        timing_definition="continuous data clones + zero_grad + native F/B/P2P + SGD; final device drain and barrier included; aggregation/serialization excluded",
        cuda_event_definition="current-compute-stream interval, including internal gaps/waits; unjoined side-stream compute not proven covered; not pure kernel time",
        synchronous_definition="two device synchronizations per F/B, pre-sync excluded from wall; no extra P2P synchronizations",
        memory_before=memory_sample(),
        warmup_iterations=list(range(args.warmups)),
        measurement_iterations=list(range(args.warmups, args.warmups + args.iterations)),
    )
    with (
        patch.object(schedules, "_run_with_slackpipe_cost_calibration", capture)
        if args.transfer_correctness
        else nullcontext()
    ):
        warm = []
        for i in range(args.warmups):
            start = time.perf_counter()
            losses = observed(i)
            warm.append(dict(iteration=i, host_dispatch_ms=(time.perf_counter() - start) * 1000))
        torch.cuda.synchronize()
        validate(losses)
        meta["memory_after_warmup"] = memory_sample()
        dist.barrier()
        torch.cuda.synchronize()
        before = time.perf_counter()
        samples = []
        for i in meta["measurement_iterations"]:
            start = time.perf_counter()
            losses = observed(i)
            samples.append(dict(iteration=i, host_dispatch_ms=(time.perf_counter() - start) * 1000))
        torch.cuda.synchronize()
        dist.barrier()
        meta["continuous_wall_ms"] = (time.perf_counter() - before) * 1000
    validate(losses)
    if args.transfer_correctness:
        numerical["losses"] = clone_tree(losses)
        torch.save(cpu_tree(numerical), output / f"numerical.rank{rank}.pt")
    meta.update(
        samples=samples,
        warmup_samples=warm,
        memory_end=memory_sample(),
        context_after=context_snapshot(),
    )
    rows = collector.resolve() if collector else []
    meta["optimizer_samples"] = [r for r in rows if r["phase"] == "optimizer"]
    meta["steps"] = collector.steps if collector else []
    for row in rows:
        row["measured"] = row["iteration"] >= args.warmups
    return [r for r in rows if r["phase"] != "optimizer"], meta


def pipeline(args, remaining):
    from megatron.core.pipeline_parallel.slackpipe import stage_measurement
    from tools import slackpipe_eval_worker as worker
    from tools.slackpipe_eval_args import argument_parser, normalize_args

    parser = argument_parser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--calibration-design", type=Path, required=True)
    parsed = normalize_args(parser.parse_args(remaining))
    parsed.stage_collection = "off" if args.condition == "off" else "events"
    parsed.stage_save_state = args.correctness
    parsed.transfer_condition, parsed.transfer_group = args.condition, args.group
    parsed.transfer_correctness = args.correctness
    parsed.action, parsed.memory_history_entries = parsed.stage, 100000
    base_config = worker.transformer_config

    def configuration(*a, **kw):
        result = base_config(*a, **kw)
        result.microbatch_group_size_per_vp_stage = args.group
        return result

    collector = (
        SynchronousStageCollector if args.condition == "pipeline_sync" else StageEventCollector
    )
    with (
        patch.object(worker, "transformer_config", configuration),
        patch.object(stage_measurement, "StageEventCollector", collector),
        patch.object(stage_measurement, "collect_stage_window", transfer_window),
    ):
        worker.run_worker(parsed)


def composition(args):
    design = json.loads(args.design.read_text())
    model = load_model(args.model)
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with tempfile.NamedTemporaryFile() as rendezvous:
        dist.init_process_group(
            "gloo", init_method=f"file://{rendezvous.name}", rank=0, world_size=1
        )
        parallel_state.initialize_model_parallel()
        config = transformer_config(model, dict(pp=1, vpp=None), "fp32")
        results = []
        entries = design["composition_cases"]
        entries = entries[args.run % len(entries) :] + entries[: args.run % len(entries)]
        for case in entries:
            torch.manual_seed(1234)
            model_parallel_cuda_manual_seed(1234)
            start, end = case["range"]
            stack = (
                build_module(
                    hybrid_stack_spec,
                    config=config,
                    layer_type_list=list(model["hybrid_pattern"][start:end]),
                    pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
                    pre_process=True,
                    post_process=False,
                    post_layer_norm=False,
                )
                .cuda()
                .train()
            )
            # Global identities match pipeline initialization, not segment-local numbering.
            parameters = {}
            for name, p in stack.named_parameters():
                pieces = name.split(".")
                if pieces[0] == "layers":
                    pieces[1] = str(int(pieces[1]) + start)
                parameters["decoder." + ".".join(pieces)] = p
            for i, layer in enumerate(stack.layers):
                layer.layer_number = start + i + 1
            initialize_parameters(parameters, 1234, 12)
            optimizer = torch.optim.SGD(stack.parameters(), lr=0.001)
            generator = torch.Generator(device="cuda").manual_seed(5678)
            hidden = torch.randn(32, 1, 64, generator=generator, device="cuda", requires_grad=True)
            # An averaged linear surrogate, not an unnormalized random linear loss.
            # Keep SGD stable without selecting timings or changing model weights.
            upstream = torch.randn(32, 1, 64, generator=generator, device="cuda") / (4 * 32 * 64)
            pairs = [
                (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                for _ in range(120)
            ]
            for pair in pairs:
                for event in pair:
                    event.record()
            torch.cuda.synchronize()
            rows = []
            before_context = context_snapshot()
            with no_communication():
                for i in range(60):
                    optimizer.zero_grad(set_to_none=True)
                    hidden.grad = None
                    for offset, phase in enumerate(("F", "B")):
                        first, last = pairs[2 * i + offset]
                        first.record()
                        before = time.perf_counter()
                        if phase == "F":
                            output = stack(hidden, attention_mask=None)
                        else:
                            torch.autograd.backward(output, upstream)
                        last.record()
                        rows.append(
                            dict(
                                iteration=i,
                                phase=phase,
                                measured=i >= 30,
                                host_dispatch_ms=(time.perf_counter() - before) * 1000,
                            )
                        )
                    optimizer.step()
                    if i == 29:
                        torch.cuda.synchronize()
                torch.cuda.synchronize()
            for row, (first, last) in zip(rows, pairs):
                row["event_ms"] = first.elapsed_time(last)
            finite = all(torch.isfinite(p).all().item() for p in stack.parameters())
            result = dict(
                case=case,
                run=args.run,
                device=args.device,
                samples=rows,
                input=tensor_description(hidden),
                output=tensor_description(output),
                upstream=tensor_description(upstream),
                context_before=before_context,
                context_after=context_snapshot(),
                environment=environment_metadata(),
                finite_parameters=finite,
                upstream_rule="seed5678 standard normal divided by B*S*H=8192; averaged linear surrogate, not pipeline loss gradient",
                modules=[f"{type(m).__module__}.{type(m).__qualname__}" for m in stack.layers],
                semantics="one body F and single-entry B per repetition; SGD after each; no boundaries/P2P; random matched-shape input/upstream, not pipeline activations; gradient reset once per repetition; no per-F/B sync",
            )
            write_new(args.result / f"{case['id']}.json", result)
            if not finite:
                raise ValueError("Nonfinite composition parameters; raw samples preserved")
            results.append(case["id"])
            del stack, optimizer, parameters, hidden, output, upstream, pairs
            gc.collect()
            torch.cuda.empty_cache()
        write_new(args.result / "order.json", results)
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition",
        choices=("off", "pipeline_events", "pipeline_sync", "composition_stage"),
        required=True,
    )
    parser.add_argument("--group", type=int, choices=(2, 4), default=2)
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--design", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    args, remaining = parser.parse_known_args()
    if args.condition == "composition_stage":
        if remaining:
            parser.error(str(remaining))
        composition(args)
    else:
        pipeline(args, remaining)


if __name__ == "__main__":
    main()
