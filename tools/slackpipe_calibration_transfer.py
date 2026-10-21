# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Frozen, opt-in A2000 transfer study. Never exports trusted costs or runs a solver."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from megatron.core.pipeline_parallel.schedules import get_pp_rank_microbatches, get_schedule_table
from megatron.core.pipeline_parallel.slackpipe.isolated_profile import (
    class_mapping,
    execution_signature,
)
from megatron.core.pipeline_parallel.slackpipe.manifest import build_model_manifest
from megatron.core.pipeline_parallel.slackpipe.plan import parse_slackpipe_plan
from tools.slackpipe_eval_config import load_model, transformer_config
from tools.slackpipe_stage_calibration import digest, write_new

SPLITS = [[3, 3, 3, 3], [2, 4, 2, 4], [4, 2, 4, 2], [2, 2, 4, 4], [4, 4, 2, 2], [2, 3, 4, 3]]
CONDITIONS = ["off", "pipeline_sync", "pipeline_events"]


def native_order(group):
    table = get_schedule_table(4, 2, group)
    orders = []
    for rank in range(2):
        communicator = SimpleNamespace(
            pp_group=SimpleNamespace(size=lambda: 2, rank=lambda: rank),
            virtual_pipeline_model_parallel_size=2,
        )
        _, _, warmup, _ = get_pp_rank_microbatches(4, 2, group, p2p_communicator=communicator)
        forward = [dict(kind="F", microbatch=b, stage=2 * c + rank) for b, c in table]
        backward = [dict(kind="B", microbatch=b, stage=2 * (1 - c) + rank) for b, c in table]
        operations = forward[:warmup]
        for f, b in zip(forward[warmup:], backward):
            operations.extend((f, b))
        operations.extend(backward[len(forward) - warmup :])
        orders.append(operations)
    parse_slackpipe_plan(
        dict(
            num_layers=12,
            num_workers=2,
            num_stages=4,
            num_microbatches=4,
            layer_split=[3] * 4,
            stage_to_worker=[0, 1, 0, 1],
            operations=orders,
        )
    )
    return orders


def prepare(root, model_path):
    root.mkdir(parents=True, exist_ok=False)
    model = load_model(model_path)
    expected = dict(num_layers=12, hidden_size=64, hybrid_pattern="M*-M*-M*-M*-")
    if any(model[k] != v for k, v in expected.items()):
        raise ValueError("This diagnostic is predeclared for the existing H64/12-layer harness")
    config = transformer_config(model, dict(pp=2, vpp=2), "fp32")
    manifest = build_model_manifest(config)
    classes = class_mapping(
        manifest, execution_signature(model, seq_length=32, micro_batch_size=1, precision="fp32")
    )
    signatures = sorted(c["class_signature"] for c in classes)
    layer_class = {i: c["class_signature"] for c in classes for i in c["member_layers"]}
    cases, parts = {}, []
    for index, split in enumerate(SPLITS):
        cuts = [0] + np.cumsum(split).tolist()
        parts.append(
            dict(
                id=f"P{index}",
                cuts=cuts,
                role="train" if index in (1, 2, 3) else "check" if index == 0 else "validation",
            )
        )
        for a, b in zip(cuts, cuts[1:]):
            count = Counter(layer_class[i] for i in range(a, b))
            key = tuple(count[s] for s in signatures)
            if key not in cases or [a, b] < cases[key]["range"]:
                cases[key] = dict(
                    range=[a, b], signature_counts=list(key), pattern=model["hybrid_pattern"][a:b]
                )
    composition_cases = [dict(id=f"C{i}", **v) for i, (_, v) in enumerate(sorted(cases.items()))]
    design = dict(
        schema_version="slackpipe.stage_design.v1",
        manifest_hash=manifest["manifest_hash"],
        partitions=parts,
    )
    write_new(root / "model.json", model)
    write_new(root / "model_manifest.json", manifest)
    write_new(root / "design.json", design)
    for run in range(3):
        # Same partition order within each paired mode/order block; rotations across runs.
        ordered = parts[run * 2 :] + parts[: run * 2]
        write_new(root / f"design.run{run}.json", dict(design, partitions=ordered))
    frozen = dict(
        schema_version="slackpipe.calibration_transfer.v1",
        utc=datetime.now(timezone.utc).isoformat(),
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        classes=classes,
        signatures=signatures,
        composition_cases=composition_cases,
        orders={str(g): native_order(g) for g in (2, 4)},
        train="P1-P3/run0,1/group2 only",
        control="P0 never fitted",
        validation="P1-P3/run2; P4-P5/run0,1; P4-P5/run2; group4 separately, no refit",
        warmup=30,
        measured=30,
        runs=3,
        seed=1234,
        learning_rate=0.001,
        composition="whole native real segment, one B entry, SGD per repetition; mean of all samples then equal mean of run0,1; run2 separately diagnostic",
        composition_upstream="seed5678 normal / (B*S*H=8192), averaged linear surrogate; no pipeline observations",
        boundary="M0 first/last mean residual, a=1 b0=0, from pipeline_events training only; contains boundary AND context residual, not pure embedding/head cost; negative increments retained",
        models="F/B separate unweighted OLS on per-stage/run arithmetic means; M1 sync GPU events; M2 normal GPU events; [composition,1,first,last]; no clipping/filtering",
        uncertainty="OLS iid-stage standard errors descriptive only; coefficients refit separately on each training run; n=2 clusters insufficient for reliable inference",
        perturbation="paired independent-run window rank-max /30; Student-t df2 95% CI; no iid-step CI; target interval entirely within +/-1%",
        accuracy="heldout mean absolute relative stage error <=5%; signed bias and maximum also retained",
        correctness=dict(
            warmup=1,
            measured=1,
            atol=0,
            rtol=0,
            description="separate diagnostic-only cloning run, not timed evidence; outputs/loss/input gradients/parameter gradients/updated parameters",
        ),
        timing="CUDA intervals not pure compute; no cross-GPU timestamp makespan; wall/dispatch/event quantities separate",
        environment={
            k: os.environ.get(k)
            for k in [
                "OMP_NUM_THREADS",
                "MAMBA_DETERMINISTIC",
                "TRITON_CACHE_AUTOTUNING",
                "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
                "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
                "PYTORCH_CUDA_ALLOC_CONF",
                "PYTORCH_ALLOC_CONF",
            ]
        },
        sampling_order=[
            dict(
                run=r,
                groups=[2, 4] if r % 2 == 0 else [4, 2],
                conditions=CONDITIONS[r:] + CONDITIONS[:r],
            )
            for r in range(3)
        ],
    )
    write_new(root / "FROZEN.json", frozen)
    source = root / "source"
    source.mkdir()
    for name in (
        "tools/slackpipe_transfer_worker.py",
        "tools/slackpipe_calibration_transfer.py",
        "tools/slackpipe_transfer_analysis.py",
        "tools/slackpipe_training.py",
        "tools/slackpipe_eval_worker.py",
        "megatron/core/pipeline_parallel/slackpipe/stage_measurement.py",
    ):
        path = Path(name)
        if path.exists():
            shutil.copyfile(path, source / path.name)
    (source / "dirty.diff").write_bytes(subprocess.check_output(["git", "diff", "--binary"]))
    (source / "status.txt").write_bytes(subprocess.check_output(["git", "status", "--short"]))
    write_new(
        root / "FREEZE_HASHES.json",
        {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*")) if p.is_file()},
    )


def pipeline_command(root, run, group, mode, correctness=False):
    output = root / ("correctness" if correctness else "pipeline") / f"run{run}.g{group}.{mode}"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        "-m",
        "tools.slackpipe_transfer_worker",
        "--condition",
        mode,
        "--group",
        str(group),
    ]
    if correctness:
        command.append("--correctness")
    command += [
        "--phase",
        "calibrate",
        "--model-config",
        str(root / "model.json"),
        "--method",
        "interleaved-1f1b",
        "--pp",
        "2",
        "--num-stages",
        "4",
        "--microbatches",
        "4",
        "--seq-length",
        "32",
        "--precision",
        "fp32",
        "--warmups",
        "1" if correctness else "30",
        "--iterations",
        "1" if correctness else "30",
        "--calibration-design",
        str(root / f"design.run{run}.json"),
        "--run-id",
        f"run{run}",
        "--output",
        str(output),
    ]
    return output, command


def execute(root, output, command):
    key = output.name
    logs = root / "commands"
    logs.mkdir(exist_ok=True)
    path = logs / f"{output.parent.name}.{key}.json"
    start = time.time()
    write_new(
        path,
        dict(command=command, cwd=os.getcwd(), start_utc=datetime.now(timezone.utc).isoformat()),
    )
    with path.with_suffix(".log").open("x") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    write_new(
        path.with_suffix(".exit.json"),
        dict(returncode=result.returncode, elapsed_seconds=time.time() - start),
    )
    if result.returncode:
        raise RuntimeError(f"Failed, preserved raw output: {path}")
    print(f"Completed {output} ({time.time() - start:.1f}s)", flush=True)


def compare_tree(a, b):
    if isinstance(a, torch.Tensor):
        if (
            a.shape != b.shape
            or a.dtype != b.dtype
            or not torch.isfinite(a).all()
            or not torch.isfinite(b).all()
        ):
            raise ValueError("Invalid numerical tensor")
        return float((a - b).abs().max()) if a.numel() else 0.0
    if isinstance(a, dict):
        if a.keys() != b.keys():
            raise ValueError("Mismatched numerical identities")
        return max([compare_tree(a[k], b[k]) for k in a] or [0.0])
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise ValueError("Mismatched tensor tree")
        return max([compare_tree(x, y) for x, y in zip(a, b)] or [0.0])
    if a != b:
        raise ValueError("Mismatched numerical scalar")
    return 0.0


def verify_correctness(root):
    rows = []
    expected = json.loads((root / "FROZEN.json").read_text())["orders"]
    for group in (2, 4):
        for mode in CONDITIONS:
            base = root / "correctness" / "run0.g2.off"
            other = root / "correctness" / f"run0.g{group}.{mode}"
            for partition in range(6):
                for rank in range(2):
                    differences = {}
                    for kind in ("state", "numerical"):
                        path = f"P{partition}/{kind}.rank{rank}.pt"
                        a = torch.load(base / path, weights_only=True)
                        b = torch.load(other / path, weights_only=True)
                        differences[kind] = compare_tree(a, b)
                    if mode != "off":
                        samples = json.loads(
                            (other / f"P{partition}/stage_samples.rank{rank}.json").read_text()
                        )
                        actual = [
                            dict(
                                kind="F" if s["phase"] == "forward_compute" else "B",
                                microbatch=s["microbatch"],
                                stage=s["logical_stage"],
                            )
                            for s in samples
                            if s["iteration"] == 1
                        ]
                        if actual != expected[str(group)][rank]:
                            raise ValueError("Native order differs from frozen validated order")
                    rows.append(
                        dict(
                            group=group,
                            mode=mode,
                            partition=partition,
                            rank=rank,
                            max_abs=differences,
                        )
                    )
    eligible = [
        g for g in (2, 4) if all(max(r["max_abs"].values()) == 0 for r in rows if r["group"] == g)
    ]
    write_new(
        root / "correctness.json",
        dict(
            passed=eligible == [2, 4],
            eligible_groups=eligible,
            atol=0,
            rtol=0,
            comparisons=rows,
            policy="Never relax numerical tolerance. Failed order excluded before timing, retained as failed correctness evidence; no order generalization claim.",
        ),
    )
    if 2 not in eligible:
        raise ValueError(
            "Default order collection changes numerical results; refusing all measurements"
        )
    if 4 not in eligible:
        print(
            "WARNING: alternate order failed exact numerical equivalence; retain failure data and do not claim order generalization.",
            flush=True,
        )


def run(root):
    for name, sha in json.loads((root / "FREEZE_HASHES.json").read_text()).items():
        if digest(root / name) != sha:
            raise ValueError(f"Frozen input changed: {name}")
    for group in (2, 4):
        for mode in CONDITIONS:
            execute(root, *pipeline_command(root, 0, group, mode, True))
    verify_correctness(root)
    measure(root)


def measure(root):
    frozen = json.loads((root / "FROZEN.json").read_text())
    correctness = json.loads((root / "correctness.json").read_text())
    eligible = correctness["eligible_groups"]
    if 2 not in eligible:
        raise ValueError("Default order failed correctness")
    write_new(
        root / "MEASUREMENT_ADMISSION.json",
        dict(
            eligible_groups=eligible,
            correctness_sha256=digest(root / "correctness.json"),
            utc=datetime.now(timezone.utc).isoformat(),
            sources={
                name: digest(Path(name))
                for name in (
                    "tools/slackpipe_calibration_transfer.py",
                    "tools/slackpipe_transfer_worker.py",
                    "tools/slackpipe_transfer_analysis.py",
                )
            },
            amendment="Failed numerical order is not measured. No performance data used for this decision; all original frozen artifacts and failure tensors retained.",
        ),
    )
    for entry in frozen["sampling_order"]:
        r = entry["run"]
        composition_output = root / "composition" / f"run{r}"
        execute(
            root,
            composition_output,
            [
                sys.executable,
                "-m",
                "tools.slackpipe_transfer_worker",
                "--condition",
                "composition_stage",
                "--model",
                str(root / "model.json"),
                "--design",
                str(root / "FROZEN.json"),
                "--result",
                str(composition_output),
                "--run",
                str(r),
                "--device",
                str(r % 2),
            ],
        )
        for group in entry["groups"]:
            if group not in eligible:
                continue
            for mode in entry["conditions"]:
                execute(root, *pipeline_command(root, r, group, mode))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "verify", "measure"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.output, args.model)
    elif args.action == "run":
        run(args.output)
    elif args.action == "verify":
        verify_correctness(args.output)
    else:
        measure(args.output)


if __name__ == "__main__":
    main()
