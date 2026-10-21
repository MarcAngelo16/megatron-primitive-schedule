# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Metadata-frozen linear identification study; no production estimator changes."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from megatron.core.pipeline_parallel.slackpipe.isolated_profile import (
    class_mapping,
    execution_signature,
)
from megatron.core.pipeline_parallel.slackpipe.manifest import build_model_manifest
from tools.slackpipe_calibration_transfer import execute, native_order
from tools.slackpipe_eval_config import load_model, transformer_config
from tools.slackpipe_stage_calibration import digest, write_new

TYPES = ("mamba", "mlp", "attention")
SYMBOLS = "M-*"
TRAIN = [[1, 1, 4, 6], [2, 2, 6, 2], [4, 1, 1, 6], [6, 2, 2, 2], [2, 4, 4, 2]]
HELDOUT = [[1, 3, 3, 5], [2, 5, 1, 4], [1, 7, 2, 2], [1, 8, 1, 2], [2, 3, 5, 2]]


def counts(pattern):
    return [pattern.count(c) for c in SYMBOLS]


def matrix_report(x):
    x = np.asarray(x, dtype=float)
    singular = np.linalg.svd(x, compute_uv=False)
    rank = int(np.linalg.matrix_rank(x))
    correlations = []
    for i in range(x.shape[1]):
        for j in range(i):
            if x[:, i].std() > 0 and x[:, j].std() > 0:
                correlations.append(
                    dict(columns=[j, i], correlation=float(np.corrcoef(x[:, j], x[:, i])[0, 1]))
                )
    return dict(
        rows=len(x),
        unknowns=x.shape[1],
        independent_equations=rank,
        rank=rank,
        singular_values=singular.tolist(),
        condition=float(singular[0] / singular[-1]) if rank == x.shape[1] else None,
        correlations=correlations,
        matrix=x.tolist(),
    )


def prepare(root, model_path):
    root.mkdir(parents=True, exist_ok=False)
    model = load_model(model_path)
    assert (
        model["num_layers"] == 12
        and model["hidden_size"] == 64
        and model["hybrid_pattern"] == "M*-M*-M*-M*-"
    )
    manifest = build_model_manifest(transformer_config(model, dict(pp=2, vpp=2), "fp32"))
    classes = class_mapping(
        manifest, execution_signature(model, seq_length=32, micro_batch_size=1, precision="fp32")
    )
    assert sorted(c["layer_type"] for c in classes) == sorted(TYPES)
    partitions = []
    for prefix, splits, role in (
        ("T", TRAIN, "train"),
        ("V", HELDOUT, "validation"),
        ("U", [[3, 3, 3, 3]], "check"),
    ):
        for i, split in enumerate(splits):
            partitions.append(
                dict(id=f"{prefix}{i}", role=role, cuts=[0] + np.cumsum(split).tolist())
            )
    segments = [
        dict(
            id=f"L{n}.o{start}",
            range=[start, start + n],
            pattern=model["hybrid_pattern"][start : start + n],
            counts=counts(model["hybrid_pattern"][start : start + n]),
            role="train" if n in (1, 2, 4, 6) else "validation",
        )
        for n in range(1, 9)
        for start in range(3)
    ]
    train_segments = [s for s in segments if s["role"] == "train"]
    body = []
    boundary = []
    for p in partitions:
        if p["role"] != "train":
            continue
        for s, (a, b) in enumerate(zip(p["cuts"], p["cuts"][1:])):
            row = counts(model["hybrid_pattern"][a:b]) + [1]
            boundary.append(row + [int(s == 0), int(s == 3)])
            if s in (1, 2):
                body.append(row)
    reports = dict(
        control_H1_proxy=matrix_report([[sum(s["counts"]), 1] for s in train_segments]),
        control_H2=matrix_report([s["counts"] + [1] for s in train_segments]),
        pipeline_body_H2=matrix_report(body),
        pipeline_boundary_H2=matrix_report(boundary),
        entry=matrix_report(
            [
                s["counts"] + [1, q - 1]
                for s in train_segments
                for q in sorted({1, 2, len(s["pattern"])})
                if q <= len(s["pattern"])
            ]
        ),
    )
    if any(r["rank"] != r["unknowns"] for r in reports.values()):
        raise ValueError(f"Metadata-only design unidentifiable: {reports}")
    design = dict(
        schema_version="slackpipe.stage_design.v1",
        manifest_hash=manifest["manifest_hash"],
        partitions=partitions,
    )
    write_new(root / "model.json", model)
    write_new(root / "model_manifest.json", manifest)
    write_new(root / "design.json", design)
    for run in range(3):
        ordered = partitions[run * 3 :] + partitions[: run * 3]
        write_new(root / f"design.run{run}.json", dict(design, partitions=ordered))
    frozen = dict(
        schema_version="slackpipe.linear_identification.v1",
        date_jst="2026-10-03",
        classes=classes,
        segments=segments,
        types=list(TYPES),
        partitions=partitions,
        design_reports=reports,
        order=native_order(2),
        warmups=30,
        measured=30,
        runs=[0, 1, 2],
        devices=[0, 1],
        anchor="Measured unit GPU-event means fix u_t; no free transformation of unit profiler; effective theta=a_t*u_t is not intrinsic kernel cost",
        train="Only train ranges/partitions and run0,1; freeze pooled unit table from both devices/run0,1; never remeasure u on validation run for predictions",
        models="F/B separately: H1=[sum(n*u),1]; H2=[n_m*u_m,n_mlp*u_mlp,n_a*u_a,1]; H2-boundary adds first,last using only training pipeline data",
        policy="Arithmetic mean, all 30 measured repetitions retained; no filters or rescaling; unweighted SVD least squares on stage/run/device means; report rank, positivity, coefficients by run and leave-one-training-case out",
        h="Same disconnected graphs reused via retain_graph=True; q=1,2,n roots grouped without changing graphs/inputs/gradient reset; same ONE B event interval; only fit h if all gradient checks pass",
        correctness=dict(
            atol=0,
            rtol=0,
            steps=[0, 29, 30],
            description="Layer inputs/outputs/input grads, global param grads before SGD and updated params; tokens/labels/microbatch IDs; initial and post30-warmup state",
        ),
        fallback="If native partitions differ, pipeline fits are descriptive only; controlled body tests at fixed weights with exact layer replay establish narrower evidence",
        unit_semantics="One actual native unit F/B per repetition, no SGD; zero once per pair; same shape/precision/config; archived initial state",
        control_semantics="One actual contiguous body F/B per repetition at fixed weights, no SGD; exact replayed inputs/upstreams checks; not full pipeline training",
        pipeline_semantics="Full B4 native training step with SGD, 30+30; off/sync/events rotated by run; no optimizer cost in F/B fit",
        timing="GPU events primary; synchronized host wall for sync, enqueue wall for async, never mix; no cross-GPU timestamp makespan",
        hypothesis="Stage MAPE <=5% is a target, not enforced by data selection; no order experiment until numerical issue resolved",
        h1_design_note="Before measuring u, total layer count is only a structural proxy. Positive u and lengths including six balanced layers guarantee nonconstant X; actual scaled SVD reported after collection.",
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        environment={
            k: os.environ.get(k)
            for k in [
                "OMP_NUM_THREADS",
                "MAMBA_DETERMINISTIC",
                "TRITON_CACHE_AUTOTUNING",
                "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
                "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
                "PYTORCH_ALLOC_CONF",
                "PYTORCH_CUDA_ALLOC_CONF",
            ]
        },
    )
    write_new(root / "FROZEN.json", frozen)
    source = root / "source"
    source.mkdir()
    for path in list(Path("tools").glob("slackpipe_linear*.py")) + [
        Path("tools/slackpipe_transfer_worker.py"),
        Path("tools/slackpipe_training.py"),
        Path("tools/slackpipe_eval_worker.py"),
        Path("megatron/core/pipeline_parallel/slackpipe/stage_measurement.py"),
    ]:
        shutil.copyfile(path, source / path.name)
    (source / "dirty.diff").write_bytes(subprocess.check_output(["git", "diff", "--binary"]))
    (source / "status.txt").write_bytes(subprocess.check_output(["git", "status", "--short"]))
    write_new(
        root / "FREEZE_HASHES.json",
        {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*")) if p.is_file()},
    )


def pipeline_command(root, run, mode, audit=False):
    output = root / ("audit" if audit else "pipeline") / f"run{run}.{mode}"
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        "-m",
        "tools.slackpipe_linear_worker",
        "--action",
        "audit" if audit else "pipeline",
        "--condition",
        mode,
        "--group",
        "2",
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
        "30",
        "--iterations",
        "1" if audit else "30",
        "--calibration-design",
        str(root / f"design.run{run}.json"),
        "--run-id",
        f"run{run}",
        "--output",
        str(output),
    ]
    return output, cmd


def run(root, action):
    if action == "audit":
        execute(root, *pipeline_command(root, 0, "off", True))
        return
    if not (root / "numerical_audit.json").exists():
        raise ValueError("Inspect and summarize initial/warmup numerical audit before timing")
    for name, sha in json.loads((root / "FREEZE_HASHES.json").read_text()).items():
        if digest(root / name) != sha:
            raise ValueError("Frozen input changed")
    for r in range(3):
        modes = ["pipeline_sync", "pipeline_events", "off"]
        modes = modes[r:] + modes[:r]
        # Cross device with run, rather than confounding GPU1 with run1.
        for d in [0, 1] if r % 2 == 0 else [1, 0]:
            output = root / "controls-v2" / f"run{r}.gpu{d}"
            execute(
                root,
                output,
                [
                    sys.executable,
                    "-m",
                    "tools.slackpipe_linear_worker",
                    "--action",
                    "controls",
                    "--root",
                    str(root),
                    "--result",
                    str(output),
                    "--run",
                    str(r),
                    "--device",
                    str(d),
                ],
            )
            if d == ([0, 1] if r % 2 == 0 else [1, 0])[0]:
                execute(root, *pipeline_command(root, r, modes[0]))
        for mode in modes[1:]:
            execute(root, *pipeline_command(root, r, mode))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "audit", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.output, args.model)
    else:
        run(args.output, args.action)


if __name__ == "__main__":
    main()
