# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Diagnostic-only unit/body/entry probes and native layer-state audit."""

import argparse
import gc
import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.pipeline_parallel import schedules
from megatron.core.pipeline_parallel.slackpipe.collection import environment_metadata
from megatron.core.pipeline_parallel.slackpipe.stage_measurement import tensor_description
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.spec_utils import build_module
from tools import slackpipe_transfer_worker as transfer
from tools.slackpipe_diagnose_composition import context_snapshot, tensor
from tools.slackpipe_eval_config import load_model, transformer_config
from tools.slackpipe_isolated_profile import make_unit, no_communication
from tools.slackpipe_model_support import _logical_named_parameters
from tools.slackpipe_stage_calibration import write_new
from tools.slackpipe_training import initialize_parameters


def tensor_difference(a, b):
    if a is None or b is None:
        if a is not b:
            return dict(max_abs=None, max_relative=None, equal=False)
        return dict(max_abs=0.0, max_relative=0.0, equal=True)
    error = (a.detach() - b.detach()).abs().float()
    return dict(
        max_abs=float(error.max()),
        max_relative=float((error / b.detach().abs().float().clamp_min(1e-12)).max()),
        equal=bool(torch.equal(a, b)),
    )


def audit_window(**kw):
    """Full SGD warmup; snapshot before updates and layer/microbatch identities."""
    from tools import slackpipe_training as native

    params = _logical_named_parameters(kw["model"])
    record = dict(
        initial={k: v.detach().cpu().clone() for k, v in params.items()},
        layers={},
        batches={},
        steps={},
        configuration=kw["args"].eval_metadata,
    )
    selected = {0, 29, 30}
    state = dict(iteration=-1, current=None)
    original_compute = schedules._run_with_slackpipe_cost_calibration
    original_forward = native._forward_step_func
    original_sgd = torch.optim.SGD.step
    original_step = kw["step"]
    handles = []

    def compute(**call):
        state["current"] = dict(
            iteration=state["iteration"],
            phase=call["phase"],
            microbatch=call["microbatch"],
            stage=call["model_chunk_id"] * 2 + dist.get_rank(),
        )
        try:
            return original_compute(**call)
        finally:
            state["current"] = None

    def forward(iterator, model):
        batch = next(iterator)
        c = state["current"]
        if c["iteration"] in selected:
            key = f"i{c['iteration']}/s{c['stage']}/b{c['microbatch']}"
            record["batches"][key] = transfer.clone_tree(batch)
        return original_forward(iter([batch]), model)

    def layer_hook(layer_id):
        def hook(module, inputs, kwargs, output):
            c = state["current"]
            if c is None or c["iteration"] not in selected:
                return
            hidden = inputs[0] if inputs else kwargs["hidden_states"]
            out = tensor(output)
            key = f"i{c['iteration']}/l{layer_id}/b{c['microbatch']}"
            if key in record["layers"]:
                raise ValueError("Duplicate layer/microbatch identity")
            row = dict(
                **c,
                layer=layer_id,
                input=hidden.detach().clone(),
                output=out.detach().clone(),
                input_layout=tensor_description(hidden),
                output_layout=tensor_description(out),
                module=f"{type(module).__module__}.{type(module).__qualname__}",
            )
            record["layers"][key] = row
            hidden.register_hook(lambda g: row.update(input_grad=g.detach().clone()))
            out.register_hook(lambda g: row.update(output_grad=g.detach().clone()))

        return hook

    for model in kw["model"]:
        for layer in model.decoder.layers:
            handles.append(
                layer.register_forward_hook(layer_hook(layer.layer_number - 1), with_kwargs=True)
            )

    def sgd(optimizer, *a, **k):
        i = state["iteration"]
        if i in selected:
            record["steps"][str(i)] = dict(
                before={k: v.detach().clone() for k, v in params.items()},
                gradients={
                    k: None if v.grad is None else v.grad.detach().clone()
                    for k, v in params.items()
                },
            )
        result = original_sgd(optimizer, *a, **k)
        if i in selected:
            record["steps"][str(i)]["after"] = {k: v.detach().clone() for k, v in params.items()}
        return result

    def step():
        state["iteration"] += 1
        result = original_step()
        if state["iteration"] in selected:
            record["steps"][str(state["iteration"])]["losses"] = transfer.clone_tree(result)
        return result

    kw["step"] = step
    try:
        with (
            patch.object(schedules, "_run_with_slackpipe_cost_calibration", compute),
            patch.object(native, "_forward_step_func", forward),
            patch.object(torch.optim.SGD, "step", sgd),
        ):
            return transfer.transfer_window(**kw)
    finally:
        for handle in handles:
            handle.remove()
        torch.save(
            transfer.cpu_tree(record), kw["output"] / f"layers-audit.rank{dist.get_rank()}.pt"
        )


class Timer:
    """One preinitialized event interval; synchronous or window-deferred reading."""

    def __init__(self, n, sync):
        self.pairs = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(n)
        ]
        for a, b in self.pairs:
            a.record()
            b.record()
        torch.cuda.synchronize()
        self.sync = sync
        self.rows = []

    def call(self, fn, **identity):
        if self.sync:
            torch.cuda.synchronize()
        a, b = self.pairs[len(self.rows)]
        a.record()
        start = time.perf_counter()
        result = fn()
        b.record()
        dispatch = (time.perf_counter() - start) * 1000
        if self.sync:
            torch.cuda.synchronize()
        self.rows.append(
            dict(
                identity,
                host_dispatch_ms=dispatch,
                synchronized_wall_ms=(time.perf_counter() - start) * 1000 if self.sync else None,
            )
        )
        return result

    def finish(self):
        torch.cuda.synchronize()
        for row, (a, b) in zip(self.rows, self.pairs):
            row["event_ms"] = a.elapsed_time(b)
        return self.rows


def gradients(module, hidden):
    return dict(
        parameters={k: v.grad.detach().clone() for k, v in module.named_parameters()},
        input=hidden.grad.detach().clone(),
    )


def compare_gradients(a, b):
    pairs = [tensor_difference(a["parameters"][k], b["parameters"][k]) for k in a["parameters"]] + [
        tensor_difference(a["input"], b["input"])
    ]
    return dict(
        equal=all(p["equal"] for p in pairs),
        max_abs=max(p["max_abs"] for p in pairs),
        max_relative=max(p["max_relative"] for p in pairs),
    )


def measure(module, hidden, forward, upstream, sync):
    timer = Timer(120, sync)
    for i in range(60):
        module.zero_grad(set_to_none=True)
        hidden.grad = None
        out = timer.call(forward, phase="F", iteration=i, measured=i >= 30)
        timer.call(
            lambda: torch.autograd.backward(out, upstream), phase="B", iteration=i, measured=i >= 30
        )
        if i == 29:
            torch.cuda.synchronize()
    rows = timer.finish()
    return rows, gradients(module, hidden), out.detach().clone()


def stack_for(model, case):
    config = transformer_config(model, dict(pp=1, vpp=None), "fp32")
    start, end = case["range"]
    stack = (
        build_module(
            hybrid_stack_spec,
            config=config,
            layer_type_list=list(case["pattern"]),
            pp_layer_offset=start,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
            pre_process=True,
            post_process=False,
            post_layer_norm=False,
        )
        .cuda()
        .train()
    )
    names = {}
    for name, p in stack.named_parameters():
        parts = name.split(".")
        if parts[0] == "layers":
            parts[1] = str(int(parts[1]) + start)
        names["decoder." + ".".join(parts)] = p
    initialize_parameters(names, 1234, 12)
    torch.manual_seed(5678)
    hidden = torch.randn(32, 1, 64, device="cuda", requires_grad=True)
    upstream = torch.randn_like(hidden) / 8192
    return stack, hidden, upstream


def controlled_case(model, case, run):
    stack, hidden, upstream = stack_for(model, case)
    captured = {}
    handles = []
    for i, layer in enumerate(stack.layers):

        def hook(m, a, k, o, i=i):
            inp = a[0] if a else k["hidden_states"]
            captured[i] = dict(input=inp.detach().clone().requires_grad_(True))
            tensor(o).register_hook(lambda g: captured[i].update(upstream=g.detach().clone()))

        handles.append(layer.register_forward_hook(hook, with_kwargs=True))
    out = stack(hidden, attention_mask=None)
    torch.autograd.backward(out, upstream)
    reference = gradients(stack, hidden)
    expected = out.detach().clone()
    for h in handles:
        h.remove()
    del out
    rows = []
    checks = {}
    modes = [True, False] if run % 2 == 0 else [False, True]
    for sync in modes:
        samples, g, o = measure(
            stack, hidden, lambda: stack(hidden, attention_mask=None), upstream, sync
        )
        check = compare_gradients(reference, g)
        check["output"] = tensor_difference(o, expected)
        checks["sync" if sync else "events"] = check
        rows.extend(dict(s, mode="sync" if sync else "events") for s in samples)
    # Independent native layers replay exact saved input/upstream; numerical check only.
    stack.zero_grad(set_to_none=True)
    for i, layer in enumerate(stack.layers):
        value = tensor(layer(captured[i]["input"], attention_mask=None))
        torch.autograd.backward(value, captured[i]["upstream"])
    replay = gradients(stack, captured[0]["input"])
    checks["layer_replay"] = compare_gradients(reference, replay)
    # Check one identical SGD update without inserting it into fixed-state timing.
    initial = {k: p.detach().clone() for k, p in stack.named_parameters()}
    optimizer = torch.optim.SGD(stack.parameters(), lr=0.001)
    optimizer.step()
    replay_updated = {k: p.detach().clone() for k, p in stack.named_parameters()}
    with torch.no_grad():
        for k, p in stack.named_parameters():
            p.copy_(initial[k])
            p.grad = reference["parameters"][k].clone()
    optimizer.step()
    checks["sgd_max_abs"] = max(
        float((p.detach() - replay_updated[k]).abs().max()) for k, p in stack.named_parameters()
    )
    with torch.no_grad():
        for k, p in stack.named_parameters():
            p.copy_(initial[k])
    qrows = []
    qchecks = {}
    q_failure = None
    if case["role"] == "train":
        # These exact disconnected graphs are reused for ALL q; no topology change.
        outputs = [
            tensor(layer(captured[i]["input"], attention_mask=None))
            for i, layer in enumerate(stack.layers)
        ]
        qs = sorted({1, 2, len(outputs)} & set(range(1, len(outputs) + 1)))
        if run % 2:
            qs.reverse()
        timer = Timer(60 * len(qs), False)
        for q in qs:
            indices = list(reversed(range(len(outputs))))
            groups = [
                indices[j * len(indices) // q : (j + 1) * len(indices) // q] for j in range(q)
            ]
            for iteration in range(60):
                stack.zero_grad(set_to_none=True)
                for c in captured.values():
                    c["input"].grad = None

                def backward():
                    for group in groups:
                        torch.autograd.backward(
                            [outputs[i] for i in group],
                            [captured[i]["upstream"] for i in group],
                            retain_graph=True,
                        )

                try:
                    timer.call(
                        backward, q=q, iteration=iteration, measured=iteration >= 30, phase="B"
                    )
                except AttributeError as error:
                    if "grad_output" not in str(error):
                        raise
                    q_failure = dict(
                        error=repr(error),
                        q=q,
                        iteration=iteration,
                        status="unavailable: installed TE does not support repeated backward on these retained graphs",
                    )
                    break
                if iteration == 29:
                    torch.cuda.synchronize()
            if q_failure:
                break
            qchecks[str(q)] = compare_gradients(reference, gradients(stack, captured[0]["input"]))
        qrows = timer.finish()
    return dict(
        case=case,
        samples=rows,
        checks=checks,
        q_samples=qrows,
        q_checks=qchecks,
        q_failure=q_failure,
        input=tensor_description(hidden),
        upstream=tensor_description(upstream),
        modules=[f"{type(m).__module__}.{type(m).__qualname__}" for m in stack.layers],
    )


def units(model, frozen):
    config = transformer_config(model, dict(pp=1, vpp=None), "fp32")
    result = []
    for c in frozen["classes"]:
        torch.manual_seed(1234)
        model_parallel_cuda_manual_seed(1234)
        module, hidden, forward = make_unit(model, config, c, 32, 1)
        initialize_parameters(
            {
                f"decoder.layers.{c['representative_layer']}." + k: v
                for k, v in module.named_parameters()
            },
            1234,
            12,
        )
        out = forward()
        upstream = torch.randn_like(out) / 8192
        samples, g, _ = measure(module, hidden, forward, upstream, True)
        result.append(
            dict(
                class_info=c,
                samples=samples,
                input=tensor_description(hidden),
                upstream=tensor_description(upstream),
                module=f"{type(module).__module__}.{type(module).__qualname__}",
                finite=all(torch.isfinite(v).all().item() for v in g["parameters"].values()),
            )
        )
    return result


def controls(args):
    model = load_model(args.root / "model.json")
    frozen = json.loads((args.root / "FROZEN.json").read_text())
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with tempfile.NamedTemporaryFile() as rendezvous:
        dist.init_process_group(
            "gloo", init_method=f"file://{rendezvous.name}", rank=0, world_size=1
        )
        parallel_state.initialize_model_parallel()
        torch.manual_seed(1234)
        model_parallel_cuda_manual_seed(1234)
        cases = frozen["segments"]
        cases = cases[args.run * 3 :] + cases[: args.run * 3]
        write_new(
            args.result / "context.json",
            dict(
                before=context_snapshot(),
                environment=environment_metadata(),
                run=args.run,
                device=args.device,
                order=[c["id"] for c in cases],
            ),
        )
        with no_communication():
            write_new(args.result / "units.json", units(model, frozen))
            for case in cases:
                before = context_snapshot()
                value = controlled_case(model, case, args.run)
                write_new(
                    args.result / (case["id"] + ".json"),
                    dict(
                        value,
                        run=args.run,
                        device=args.device,
                        context_before=before,
                        context_after=context_snapshot(),
                    ),
                )
                gc.collect()
                torch.cuda.empty_cache()
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=("audit", "pipeline", "controls"), required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    args, remaining = parser.parse_known_args()
    if args.action == "controls":
        if remaining:
            parser.error(str(remaining))
        controls(args)
        return
    sys.argv = [sys.argv[0]] + remaining
    if args.action == "audit":
        original = transfer.transfer_window
        # Avoid recursion: audit_window calls the saved original via this binding.
        with patch.object(
            transfer, "transfer_window", lambda **kw: audit_window_original(original, kw)
        ):
            transfer.main()
    else:
        transfer.main()


def audit_window_original(original, kw):
    with patch.object(transfer, "transfer_window", original):
        return audit_window(**kw)


if __name__ == "__main__":
    main()
