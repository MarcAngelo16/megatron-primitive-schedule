# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Opt-in native-stage envelopes. No synchronization inside a training step."""

import json
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.utils._pytree import tree_leaves

from megatron.core.pipeline_parallel.slackpipe.collection import environment_metadata, memory_sample


def tensor_description(value):
    if isinstance(value, torch.Tensor):
        return dict(shape=list(value.shape), stride=list(value.stride()), dtype=str(value.dtype))
    if isinstance(value, (tuple, list)):
        return [tensor_description(v) for v in value]
    if isinstance(value, dict):
        return {k: tensor_description(v) for k, v in value.items()}
    return None


def graph_nodes(value):
    """Native graph identities, not kernel names or forward/backward ordering."""
    values = tree_leaves(value)
    stack = [v.grad_fn for v in values if isinstance(v, torch.Tensor) and v.grad_fn is not None]
    nodes = set()
    while stack:
        node = stack.pop()
        if node in nodes:
            continue
        nodes.add(node)
        stack.extend(n for n, _ in node.next_functions if n is not None)
    return nodes


class LayerGraphLabels:
    """Diagnostic only: label real autograd nodes, including parameter gradients.

    Hooks do not insert autograd identity/view nodes. Shared leaf accumulators,
    wrappers outside layers, and activity lacking runtime correlation remain
    shared/unattributed. A label is launch provenance, not a GPU duration.
    """

    def __init__(self, collector, model, plan, manifest, rank):
        self.collector = collector
        self.handles = []
        self.nodes = {}
        self.metadata = []
        self.executions = []
        self.regions = {}
        for chunk in model:
            for module in chunk.modules():
                if not hasattr(module, "layer_number"):
                    continue
                # Only partitionable top-level decoder layers, not nested attention.
                if not any(module is layer for c in model for layer in c.decoder.layers):
                    continue
                layer = module.layer_number - 1
                stage = next(
                    s for s, (a, b) in enumerate(plan.stage_layer_ranges) if a <= layer < b
                )
                info = dict(
                    global_layer_id=layer,
                    layer_type=manifest["layers"][layer]["config_class"],
                    stage=stage,
                    worker=rank,
                    qualified_class=f"{type(module).__module__}.{type(module).__qualname__}",
                )
                self.handles.append(
                    module.register_forward_pre_hook(self._pre(info), with_kwargs=True)
                )
                self.handles.append(
                    module.register_forward_hook(self._post(info), with_kwargs=True)
                )

    def _pre(self, info):
        def hook(module, inputs, kwargs):
            if self.collector.current is None:
                return
            identity = dict(self.collector.current, **info)
            region = torch.profiler.record_function(
                "SlackPipeLayer/F/" + json.dumps(identity, sort_keys=True).encode().hex()
            )
            region.__enter__()
            self.regions[id(module)] = (
                region,
                graph_nodes((inputs, kwargs)),
                identity,
                tensor_description(dict(args=inputs, kwargs=kwargs)),
            )

        return hook

    def _post(self, info):
        def hook(module, inputs, kwargs, output):
            if id(module) not in self.regions:
                return
            region, ancestors, identity, description = self.regions.pop(id(module))
            region.__exit__(None, None, None)
            record = dict(identity, input=description, output=tensor_description(output), nodes=[])
            self.metadata.append(record)
            for node in graph_nodes(output) - ancestors:
                if node in self.nodes:
                    self.nodes[node]["owners"].add(info["global_layer_id"])
                    continue
                entry = dict(owners={info["global_layer_id"]}, identity=identity, active=None)
                self.nodes[node] = entry
                record["nodes"].append(node.name())

                def pre(grad_outputs, entry=entry, name=node.name()):
                    owners = sorted(entry["owners"])
                    # AccumulateGrad nodes can be reused by several microbatches.
                    # Identity comes from the currently executing native B, not
                    # the first F that discovered this node.
                    label = {
                        **entry["identity"],
                        **(self.collector.current or {}),
                        "phase": "backward_compute",
                        "owners": owners,
                        "attribution": "unique" if len(owners) == 1 else "shared",
                        "node": name,
                    }
                    self.executions.append(label)
                    region = torch.profiler.record_function(
                        "SlackPipeLayer/B/" + json.dumps(label, sort_keys=True).encode().hex()
                    )
                    entry["active"] = region
                    region.__enter__()

                def post(grad_inputs, grad_outputs, entry=entry):
                    entry["active"].__exit__(None, None, None)
                    entry["active"] = None

                self.handles.extend((node.register_prehook(pre), node.register_hook(post)))

        return hook

    def clear_graph(self):
        # Called only after full F/B: avoid retaining completed autograd graphs.
        for node in self.nodes.values():
            if node["active"] is not None:
                raise RuntimeError("Unclosed backward node range")
        self.nodes.clear()
        # Module hooks remain for the next step; node handles may now be removed.
        while len(self.handles) > self.module_handles:
            self.handles.pop().remove()

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.nodes.clear()


class StageEventCollector:
    serialized = False
    nvtx = False

    def __init__(self, capacity, *, timeline=False):
        self.timeline = timeline
        self.pairs = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(capacity)
        ]
        for pair in self.pairs:
            for event in pair:
                event.record()
        torch.cuda.synchronize()  # Lazy initialization, before any training warmup.
        self.pending = []
        self.steps = []
        self.current = None
        self.active = False
        self.layer_labels = None

    def region(self, name):
        return torch.profiler.record_function(name) if self.timeline else nullcontext()

    def begin(self, iteration):
        if self.active:
            raise RuntimeError("Unfinished stage iteration")
        self.active, self.iteration, self.position = True, iteration, 0
        self.p2p = []

    def compute(self, compute, identity):
        return self._measure(compute, dict(identity, iteration=self.iteration))

    def _measure(self, call, identity):
        if not self.active or len(self.pending) >= len(self.pairs):
            raise RuntimeError("Inactive collector or exhausted event pool")
        start, end = self.pairs[len(self.pending)]
        stream = torch.cuda.current_stream()
        row = dict(
            identity,
            schedule_position=self.position,
            compute_stream_id=stream.cuda_stream,
            dependency_wait_ms=None,
            communication_gpu_ms=None,
            status="pending",
            side_stream_coverage="only work joined before end event; unjoined work unknown",
        )
        self.position += 1
        self.pending.append(row)
        self.current = identity
        kind = {"forward_compute": "F", "backward_compute": "B"}.get(identity["phase"], "optimizer")
        label = f"ScheduleTrace/{kind}/i{self.iteration}/s{identity.get('logical_stage', -1)}/b{identity.get('microbatch', -1)}"
        try:
            with self.region(label):
                start.record(stream)
                before = time.perf_counter()
                result = call()
                row["cpu_dispatch_ms"] = (time.perf_counter() - before) * 1000
                end.record(stream)
            row["status"] = "recorded"
            return result
        except BaseException:
            row["status"] = "failed"
            raise
        finally:
            self.current = None

    def optimizer(self, call):
        return self._measure(call, dict(iteration=self.iteration, phase="optimizer"))

    def finish(self):
        if not self.active:
            raise RuntimeError("No active iteration")
        self.steps.append(dict(iteration=self.iteration, p2p_calls=self.p2p))
        self.active = False
        if self.layer_labels:
            self.layer_labels.clear_graph()
        return []  # Deliberately no synchronization, event read, or file I/O.

    def resolve(self):
        if self.active:
            raise RuntimeError("Cannot resolve inside a step")
        for row, (start, end) in zip(self.pending, self.pairs):
            if row["status"] != "recorded" or not end.query():
                raise RuntimeError("Incomplete/failed event pair; caller must finish window first")
            row["elapsed_ms"] = start.elapsed_time(end)
        return self.pending


def collect_stage_window(*, step, collector, args, output, validate, model, plan, manifest):
    """One continuous window; trace export and event resolution outside wall timing."""
    from megatron.core.pipeline_parallel.schedules import (
        begin_slackpipe_cost_calibration_iteration,
        end_slackpipe_cost_calibration_iteration,
    )

    mode = args.stage_collection
    timeline = mode == "layers"
    step_events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(args.warmups + args.iterations)
    ]
    for pair in step_events:
        for event in pair:
            event.record()
    torch.cuda.synchronize()
    metadata = dict(
        schema_version="slackpipe.stage_measurement.v1",
        mode=mode,
        run_id=args.run_id,
        partition_id=output.name,
        rank=dist.get_rank(),
        split=list(plan.layer_split),
        manifest_hash=manifest["manifest_hash"],
        configuration=dict(
            args.eval_metadata,
            sequence_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            hidden_size=args.hidden_size,
            dtype=args.precision,
            learning_rate=args.learning_rate,
            seed=args.seed,
        ),
        environment=environment_metadata(),
        warmup_iterations=list(range(args.warmups)),
        measurement_iterations=list(range(args.warmups, args.warmups + args.iterations)),
        profiler=dict(
            enabled=timeline,
            activities=["CPU", "CUDA"] if timeline else [],
            record_shapes=False,
            with_stack=False,
            profile_memory=False,
        ),
        clocks="CUDA event durations only; no cross-device timestamp origin",
        timing_definition="data iterator + zero_grad + F/B + native P2P + SGD; validation/export excluded",
        memory={"before_warmup": memory_sample()},
        execution_settings=dict(
            torch_deterministic=torch.are_deterministic_algorithms_enabled(),
            gradient_reset="SGD.zero_grad(set_to_none=True) once per full step",
            backward="Megatron backward_step; accumulate across all microbatches",
            optimizer="unwrapped SGD",
            compiled_modules=[type(m).__qualname__ for m in model if hasattr(m, "_orig_mod")],
            module_implementations=sorted(
                {
                    f"{type(layer).__module__}.{type(layer).__qualname__}"
                    for m in model
                    for layer in m.decoder.layers
                }
            ),
            config={
                name: str(getattr(model[0].config, name, None))
                for name in (
                    "cuda_graph_impl",
                    "cuda_graph_modules",
                    "recompute_granularity",
                    "recompute_method",
                    "gradient_accumulation_fusion",
                    "bias_activation_fusion",
                    "bias_dropout_fusion",
                    "cross_entropy_loss_fusion",
                    "batch_p2p_comm",
                    "batch_p2p_sync",
                    "overlap_p2p_comm",
                )
            },
        ),
    )
    if timeline:
        if metadata["execution_settings"]["compiled_modules"]:
            raise ValueError("Refusing node hooks on compiled model: use events mode")
        labels = LayerGraphLabels(collector, model, plan, manifest, dist.get_rank())
        labels.module_handles = len(labels.handles)
        collector.layer_labels = labels

    def observed(i):
        if collector is None:
            return step()
        begin_slackpipe_cost_calibration_iteration(i, diagnostic=collector)
        try:
            return step()
        finally:
            end_slackpipe_cost_calibration_iteration()

    try:
        warmup_samples = []
        for i in metadata["warmup_iterations"]:
            step_events[i][0].record()
            before = time.perf_counter()
            losses = observed(i)
            step_events[i][1].record()
            warmup_samples.append(
                dict(iteration=i, cpu_dispatch_ms=(time.perf_counter() - before) * 1000)
            )
        torch.cuda.synchronize()
        validate(losses)
        metadata["memory"]["after_warmup"] = memory_sample()
        torch.cuda.reset_peak_memory_stats()
        # Short diagnostic capture only. No profiler object in off/events mode.
        profiler = (
            torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
                with_stack=False,
                profile_memory=False,
            )
            if timeline
            else None
        )
        samples = []
        with profiler if profiler is not None else nullcontext():
            dist.barrier()
            torch.cuda.synchronize()
            before = time.perf_counter()
            for i in metadata["measurement_iterations"]:
                step_events[i][0].record()
                start = time.perf_counter()
                losses = observed(i)
                step_events[i][1].record()
                samples.append(
                    dict(iteration=i, cpu_dispatch_ms=(time.perf_counter() - start) * 1000)
                )
                if profiler is not None:
                    profiler.step()
            torch.cuda.synchronize()
            dist.barrier()
            metadata["continuous_wall_ms"] = (time.perf_counter() - before) * 1000
        validate(losses)
        metadata["samples"] = samples
        metadata["warmup_samples"] = warmup_samples
        for sample in samples + warmup_samples:
            start, end = step_events[sample["iteration"]]
            sample["compute_stream_step_ms"] = start.elapsed_time(end)
        metadata["step_event_definition"] = (
            "current-stream envelope, not completion proof for unjoined communication; use synchronized continuous wall window for throughput"
        )
        metadata["memory"]["end"] = memory_sample()
        rows = collector.resolve() if collector is not None else []
        for i in range(args.warmups + args.iterations):
            operations = [r for r in rows if r["iteration"] == i and r["phase"] != "optimizer"]
            if operations:
                first_b = min(
                    r["schedule_position"] for r in operations if r["phase"] == "backward_compute"
                )
                last_f = max(
                    r["schedule_position"] for r in operations if r["phase"] == "forward_compute"
                )
                for row in operations:
                    position = row["schedule_position"]
                    row["pipeline_position"] = (
                        "warmup"
                        if position < first_b
                        else "cooldown" if position > last_f else "steady"
                    )
                    row["co_resident_stages"] = [
                        s
                        for s, w in enumerate(plan.stage_to_worker)
                        if w == dist.get_rank() and s != row["logical_stage"]
                    ]
        metadata["steps"] = collector.steps if collector is not None else []
        metadata["optimizer_samples"] = [r for r in rows if r["phase"] == "optimizer"]
        if timeline:
            path = output / f"{args.run_id}.layers.rank{dist.get_rank()}.cycle000.json"
            profiler.export_chrome_trace(str(path))
            metadata["capture"] = dict(
                path=str(path),
                global_iterations=metadata["measurement_iterations"],
                serialization="after entire timed window",
                clock_alignment="unverified",
            )
            metadata["layer_graph"] = labels.metadata
            metadata["backward_nodes"] = labels.executions
            metadata["layer_coverage"] = (
                "node launch provenance; fused/shared/unjoined GPU work not uniquely decomposed"
            )
        return [r for r in rows if r["phase"] != "optimizer"], metadata
    except BaseException as error:
        # Preserve incomplete records; do not drain a potentially unmatched P2P.
        path = output / f"failed.rank{dist.get_rank()}.json"
        with path.open("x") as stream:
            json.dump(
                dict(
                    metadata=metadata,
                    error=repr(error),
                    pending=collector.pending if collector else [],
                ),
                stream,
                indent=2,
            )
        raise
    finally:
        if timeline:
            labels.close()
