# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Supplemental q control: fresh isomorphic graphs, construction outside B events."""

import argparse
import gc
import json
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.pipeline_parallel.slackpipe.collection import environment_metadata
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tools.slackpipe_diagnose_composition import context_snapshot, tensor
from tools.slackpipe_isolated_profile import no_communication
from tools.slackpipe_linear_worker import Timer, compare_gradients, gradients, stack_for
from tools.slackpipe_stage_calibration import write_new


def measure_case(model, case, run):
    stack, hidden, upstream = stack_for(model, case)
    captured, handles = {}, []
    for i, layer in enumerate(stack.layers):

        def hook(module, inputs, kwargs, output, i=i):
            value = inputs[0] if inputs else kwargs["hidden_states"]
            captured[i] = dict(input=value.detach().clone().requires_grad_(True))
            tensor(output).register_hook(lambda g: captured[i].update(upstream=g.detach().clone()))

        handles.append(layer.register_forward_hook(hook, with_kwargs=True))
    out = stack(hidden, attention_mask=None)
    torch.autograd.backward(out, upstream)
    reference = gradients(stack, hidden)
    for handle in handles:
        handle.remove()
    del out
    qs = sorted({1, 2, len(stack.layers)} & set(range(1, len(stack.layers) + 1)))
    if run % 2:
        qs.reverse()
    timer = Timer(60 * len(qs), False)
    checks = {}
    for q in qs:
        indices = list(reversed(range(len(stack.layers))))
        groups = [indices[j * len(indices) // q : (j + 1) * len(indices) // q] for j in range(q)]
        for iteration in range(60):
            stack.zero_grad(set_to_none=True)
            for c in captured.values():
                c["input"].grad = None
            # Identical disconnected topology for every q; each graph consumed once.
            outputs = [
                tensor(layer(captured[i]["input"], attention_mask=None))
                for i, layer in enumerate(stack.layers)
            ]

            def backward():
                for group in groups:
                    torch.autograd.backward(
                        [outputs[i] for i in group], [captured[i]["upstream"] for i in group]
                    )

            timer.call(backward, q=q, iteration=iteration, measured=iteration >= 30, phase="B")
            if iteration == 29:
                torch.cuda.synchronize()
        checks[str(q)] = compare_gradients(reference, gradients(stack, captured[0]["input"]))
    return dict(
        case=case,
        samples=timer.finish(),
        checks=checks,
        q_order=qs,
        definition="Fresh isomorphic disconnected graphs before every B interval; same saved exact inputs/upstreams/weights, set-to-None gradients; one outer event; no retained backward graph; NOT pipeline cost",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--device", type=int, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = json.loads((args.root / "FROZEN.json").read_text())
    model = json.loads((args.root / "model.json").read_text())
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(1234)
    with tempfile.NamedTemporaryFile() as rendezvous:
        dist.init_process_group(
            "gloo", init_method=f"file://{rendezvous.name}", rank=0, world_size=1
        )
        parallel_state.initialize_model_parallel()
        model_parallel_cuda_manual_seed(1234)
        cases = [c for c in frozen["segments"] if c["role"] == "train"]
        cases = cases[args.run * 3 :] + cases[: args.run * 3]
        write_new(
            args.output / "context.json",
            dict(
                environment=environment_metadata(),
                before=context_snapshot(),
                run=args.run,
                worker=args.device,
                order=[c["id"] for c in cases],
            ),
        )
        with no_communication():
            for case in cases:
                value = measure_case(model, case, args.run)
                write_new(
                    args.output / (case["id"] + ".json"),
                    dict(value, run=args.run, worker=args.device, context=context_snapshot()),
                )
                gc.collect()
                torch.cuda.empty_cache()
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
