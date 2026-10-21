# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Opt-in mechanistic control, not a cost exporter or pipeline throughput benchmark.

Compare a native HybridStack with isolated replay of the SAME layer objects,
inputs and upstream gradients. No embedding/head/optimizer/communication is timed.
Both paths accumulate B microbatches, reset once per step, and use the same event
policy. Extra leaf AccumulateGrad nodes in isolated replay are intentionally
reported: numerical equivalence is not equality of autograd execution paths.
"""

import argparse
import dataclasses
import os
import random
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.pipeline_parallel.slackpipe.collection import environment_metadata
from megatron.core.pipeline_parallel.slackpipe.stage_measurement import (
    graph_nodes,
    tensor_description,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.spec_utils import build_module
from tools.slackpipe_eval_config import load_model, transformer_config
from tools.slackpipe_isolated_profile import no_communication
from tools.slackpipe_stage_calibration import digest, statistics, write_new
from tools.slackpipe_training import initialize_parameters

CASES = {
    "mamba2": "MM",
    "mamba6": "MMMMMM",
    "mlp2": "--",
    "mlp6": "------",
    "attention2": "**",
    "attention6": "******",
    "mixed3": "M*-",
    "mixed6": "M*-M*-",
    "rich6": "MMMM*-",
    "poor6": "M*--*-",
    "transformer2": "**",
    "transformer6": "******",
}


def context_snapshot() -> dict:
    return dict(
        utc=datetime.now(timezone.utc).isoformat(),
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        torch_threads=torch.get_num_threads(),
        omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
        gpu=subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,clocks.sm,temperature.gpu,power.draw,utilization.gpu",
                "--format=csv",
            ],
            text=True,
        ),
    )


def traced_backward(output, upstream, identity: str) -> None:
    """Launch provenance from actual autograd nodes, not CPU nesting guesses."""
    handles = []
    regions = []
    for node in graph_nodes(output):
        entry = {}

        def pre(grads, entry=entry, name=node.name()):
            region = torch.profiler.record_function(f"CompositionNode/{identity}/{name}")
            entry["region"] = region
            regions.append(entry)
            region.__enter__()

        def post(inputs, outputs, entry=entry):
            entry.pop("region").__exit__(None, None, None)

        handles.extend((node.register_prehook(pre), node.register_hook(post)))
    try:
        output.backward(upstream)
    finally:
        for entry in regions:
            if "region" in entry:
                entry.pop("region").__exit__(None, None, None)
        for handle in handles:
            handle.remove()


def tensor(value) -> torch.Tensor:
    return value[0] if isinstance(value, tuple) else value


def relative_summary(rows: list[dict], mode: str, phase: str) -> dict:
    iterations = sorted({r["iteration"] for r in rows if r["measured"]})
    # Sum individual layer envelopes per microbatch, then mean B, then median steps.
    values = [
        sum(
            r["event_ms"]
            for r in rows
            if r["mode"] == mode and r["phase"] == phase and r["iteration"] == i
        )
        / len({r["microbatch"] for r in rows if r["iteration"] == i})
        for i in iterations
    ]
    return dict(statistics=statistics(values), ordered_iteration_mean_ms=values)


def control_case(args: argparse.Namespace, name: str, order: list[str]) -> dict:
    model = load_model(args.model)
    config = transformer_config(model, dict(pp=1, vpp=None), "fp32")
    torch.manual_seed(1234)
    model_parallel_cuda_manual_seed(1234)
    spec = hybrid_stack_spec
    if name.startswith("transformer"):
        spec = dataclasses.replace(
            spec,
            submodules=dataclasses.replace(
                spec.submodules, attention_layer=get_gpt_layer_with_transformer_engine_spec()
            ),
        )
    stack = (
        build_module(
            spec,
            config=config,
            layer_type_list=list(CASES[name]),
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
            pre_process=True,
            post_process=False,
            post_layer_norm=False,
        )
        .cuda()
        .train()
    )
    initialize_parameters(dict(stack.named_parameters()), 1234, config.num_layers)
    hidden = torch.randn(32, 1, config.hidden_size, device="cuda", requires_grad=True)
    upstream = torch.randn_like(hidden)
    captured = {}
    handles = []

    def hook(index):
        def capture(module, inputs, kwargs, output):
            inp = inputs[0] if inputs else kwargs["hidden_states"]
            out = tensor(output)
            captured[index] = dict(
                input=inp.detach().clone().requires_grad_(True),
                description=dict(input=tensor_description(inp), output=tensor_description(out)),
            )

            def gradient(grad):
                captured[index]["upstream"] = grad.detach().clone()
                captured[index]["description"]["upstream"] = tensor_description(grad)

            out.register_hook(gradient)

        return capture

    for i, layer in enumerate(stack.layers):
        handles.append(layer.register_forward_hook(hook(i), with_kwargs=True))
    output = stack(hidden, attention_mask=None)
    output.backward(upstream)
    expected = {k: p.grad.detach().clone() * args.microbatches for k, p in stack.named_parameters()}
    expected_input = hidden.grad.detach().clone() * args.microbatches
    expected_output = output.detach().clone()
    for h in handles:
        h.remove()
    del output
    for c in captured.values():
        if (
            tensor_description(c["input"]) != c["description"]["input"]
            or tensor_description(c["upstream"]) != c["description"]["upstream"]
        ):
            raise ValueError("Replay clone changed tensor layout")
    total = (args.warmups + args.iterations) * args.microbatches * 2 * (len(stack.layers) + 1)
    pairs = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(total)
    ]
    for a, b in pairs:
        a.record()
        b.record()
    torch.cuda.synchronize()
    rows, checks, windows = [], {}, {}

    def timed(call, **identity):
        start, end = pairs[len(rows)]
        start.record()
        before = time.perf_counter()
        value = call()
        end.record()
        rows.append(dict(identity, cpu_dispatch_ms=1000 * (time.perf_counter() - before)))
        return value

    for mode in order:
        torch.cuda.synchronize()
        before = time.perf_counter()
        for iteration in range(args.warmups + args.iterations):
            stack.zero_grad(set_to_none=True)
            hidden.grad = None
            for c in captured.values():
                c["input"].grad = None
            for mb in range(args.microbatches):
                identity = dict(
                    mode=mode,
                    iteration=iteration,
                    measured=iteration >= args.warmups,
                    microbatch=mb,
                )
                if mode == "segment":
                    output = timed(
                        lambda: stack(hidden, attention_mask=None),
                        phase="F",
                        layer=None,
                        **identity,
                    )
                    timed(lambda: output.backward(upstream), phase="B", layer=None, **identity)
                else:
                    outputs = [
                        timed(
                            lambda i=i, layer=layer: tensor(
                                layer(captured[i]["input"], attention_mask=None)
                            ),
                            phase="F",
                            layer=i,
                            **identity,
                        )
                        for i, layer in enumerate(stack.layers)
                    ]
                    if args.isolated_backward == "per-layer-events":
                        for i in reversed(range(len(outputs))):
                            timed(
                                lambda i=i: outputs[i].backward(captured[i]["upstream"]),
                                phase="B",
                                layer=i,
                                **identity,
                            )
                    else:

                        def backward_group():
                            indices = list(reversed(range(len(outputs))))
                            if args.isolated_backward == "single-entry":
                                torch.autograd.backward(
                                    [outputs[i] for i in indices],
                                    [captured[i]["upstream"] for i in indices],
                                )
                            else:
                                for i in indices:
                                    torch.autograd.backward(outputs[i], captured[i]["upstream"])

                        # Same ONE event envelope for both entry-count controls.
                        timed(backward_group, phase="B", layer=None, **identity)
                    output = outputs[-1]
                    del outputs
            if iteration == args.warmups - 1:
                torch.cuda.synchronize()  # Common full-window boundary, never per layer/F/B.
                before = time.perf_counter()
        torch.cuda.synchronize()
        windows[mode] = 1000 * (time.perf_counter() - before)
        gradients = [p.grad for p in stack.parameters()]
        input_gradient = hidden.grad if mode == "segment" else captured[0]["input"].grad
        if not all(torch.isfinite(g).all().item() for g in gradients + [input_gradient, output]):
            raise ValueError("Nonfinite gradients/output")
        checks[mode] = dict(
            output_max_abs=float((output.detach() - expected_output).abs().max()),
            input_grad_max_abs=float((input_gradient - expected_input).abs().max()),
            param_grad_max_abs=max(
                float((p.grad - expected[k]).abs().max()) for k, p in stack.named_parameters()
            ),
        )
        del output
    for row, (a, b) in zip(rows, pairs):
        row["event_ms"] = a.elapsed_time(b)
    metadata = []
    for i, layer in enumerate(stack.layers):
        mixer = getattr(layer, "mixer", None)
        metadata.append(
            dict(
                layer=i,
                symbol=CASES[name][i],
                qualified_class=f"{type(layer).__module__}.{type(layer).__qualname__}",
                tensors=captured[i]["description"],
                mamba_mem_eff_path=getattr(mixer, "use_mem_eff_path", None),
                mamba_chunk_size=getattr(mixer, "chunk_size", None),
            )
        )
    summaries = {
        mode: {phase: relative_summary(rows, mode, phase) for phase in ("F", "B")} for mode in order
    }
    if args.trace_steps:
        # Separate, post-measurement capture: never used as calibration samples.
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
            profile_memory=False,
        ) as prof:
            for mode in order:
                for i in range(args.trace_steps):
                    stack.zero_grad(set_to_none=True)
                    hidden.grad = None
                    for c in captured.values():
                        c["input"].grad = None
                    if mode == "segment":
                        with torch.profiler.record_function(f"Composition/{mode}/{i}/F"):
                            output = stack(hidden, attention_mask=None)
                        with torch.profiler.record_function(f"Composition/{mode}/{i}/B"):
                            traced_backward(output, upstream, f"{mode}/{i}/B")
                    else:
                        with torch.profiler.record_function(f"Composition/{mode}/{i}/F"):
                            outputs = [
                                tensor(layer(captured[j]["input"], attention_mask=None))
                                for j, layer in enumerate(stack.layers)
                            ]
                        with torch.profiler.record_function(f"Composition/{mode}/{i}/B"):
                            for j in reversed(range(len(outputs))):
                                traced_backward(
                                    outputs[j], captured[j]["upstream"], f"{mode}/{i}/B"
                                )
                        del outputs
                    prof.step()
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(args.output / f"{name}.trace.json"))
    return dict(
        case=name,
        isolated_backward=args.isolated_backward,
        order=order,
        samples=rows,
        summaries=summaries,
        numerical_checks=checks,
        isolated_over_segment={
            p: summaries["isolated"][p]["statistics"]["median"]
            / summaries["segment"][p]["statistics"]["median"]
            for p in ("F", "B")
        },
        layers=metadata,
        measured_window_wall_ms=windows,
        config=dict(
            dtype=str(config.params_dtype),
            cuda_graph_impl=config.cuda_graph_impl,
            recompute=config.recompute_granularity,
            deterministic=torch.are_deterministic_algorithms_enabled(),
            bias_dropout_fusion=config.bias_dropout_fusion,
            gradient_accumulation_fusion=config.gradient_accumulation_fusion,
        ),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--warmups", type=int, default=20)
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--microbatches", type=int, default=4)
    p.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    p.add_argument("--worker-run", type=int)
    p.add_argument("--trace-steps", type=int, default=0)
    p.add_argument(
        "--isolated-backward",
        choices=("per-layer-events", "separate-entries", "single-entry"),
        default="per-layer-events",
    )
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Fresh diagnostic directory required")
    if min(args.runs, args.warmups, args.iterations, args.microbatches) < 1:
        raise ValueError("Positive repetition counts required")
    if args.trace_steps < 0:
        raise ValueError("Trace steps must be nonnegative")
    if args.trace_steps and args.isolated_backward != "per-layer-events":
        raise ValueError(
            "Trace capture currently supports the original per-layer-events control only"
        )
    args.output.mkdir(parents=True)
    if args.worker_run is None:
        design = dict(
            schema_version="slackpipe.composition_control.v1",
            model=str(args.model),
            model_sha256=digest(args.model),
            warmups=args.warmups,
            iterations=args.iterations,
            microbatches=args.microbatches,
            runs=args.runs,
            isolated_backward=args.isolated_backward,
            profiler=dict(
                trace_steps=args.trace_steps,
                trace_microbatches_per_step=1,
                activities=["CPU", "CUDA"] if args.trace_steps else [],
                record_shapes=False,
                with_stack=False,
                profile_memory=False,
            ),
            cases=args.cases,
            order_seed=8100,
            scope="H from model, S32/mb1/FP32, PP1 mechanistic control; not a 16B scale proxy or full optimizer step.",
            policy="All samples retained. Population stddev. Median of per-step mean microbatch envelopes. No per-region synchronize; initialized event pool, synchronization at warmup/window boundaries. Same fixed weights and actual intermediate inputs/upstream gradients for both modes; parameter/input gradients computed, B-way accumulation, no optimizer.",
            transformer_control="Full attention+MLP TransformerLayer, ReLU2/no RoPE at matching H, not an original Llama result.",
        )
        write_new(args.output / "design.json", design)
        for run in range(args.runs):
            cmd = [
                sys.executable,
                "-m",
                "tools.slackpipe_diagnose_composition",
                "--model",
                str(args.model),
                "--output",
                str(args.output / f"run{run}"),
                "--worker-run",
                str(run),
                "--warmups",
                str(args.warmups),
                "--iterations",
                str(args.iterations),
                "--microbatches",
                str(args.microbatches),
                "--trace-steps",
                str(args.trace_steps),
                "--isolated-backward",
                args.isolated_backward,
                "--cases",
                *args.cases,
            ]
            write_new(args.output / f"run{run}.command.json", cmd)
            with (args.output / f"run{run}.log").open("x") as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        return
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cases = list(args.cases)
    rng = random.Random(8100 + args.worker_run)
    rng.shuffle(cases)
    write_new(args.output / "environment.json", environment_metadata())
    write_new(args.output / "order.json", cases)
    with tempfile.TemporaryDirectory() as rdv:
        dist.init_process_group("gloo", init_method=f"file://{rdv}/store", rank=0, world_size=1)
        try:
            parallel_state.initialize_model_parallel()
            with no_communication():
                for name in cases:
                    order = ["segment", "isolated"]
                    rng.shuffle(order)
                    context_before = context_snapshot()
                    result = control_case(args, name, order)
                    result["context_before"] = context_before
                    result["context_after"] = context_snapshot()
                    write_new(args.output / f"{name}.json", result)
                    print(
                        name,
                        result["isolated_over_segment"],
                        result["numerical_checks"],
                        flush=True,
                    )
        finally:
            parallel_state.destroy_model_parallel()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
