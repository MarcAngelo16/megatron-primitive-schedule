# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Shared artifact I/O, subprocess lifecycle and independent-run statistics."""
import hashlib
import json
import math
import os
import shlex
import signal
import statistics
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DETERMINISTIC = dict(
    OMP_NUM_THREADS="1",
    TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="0",
    MAMBA_DETERMINISTIC="1",
    TRITON_CACHE_AUTOTUNING="0",
    NVTE_ALLOW_NONDETERMINISTIC_ALGO="0",
)


def require_cuda_devices(required_devices: int, *, local_rank: int | None = None) -> int:
    """Fail a GPU entry before binding; CPU artifact consumers never call this."""
    import torch

    if required_devices < 1:
        raise ValueError(f"Invalid required CUDA device count: {required_devices}")
    available = torch.cuda.is_available()
    count = torch.cuda.device_count()
    observed = f"required={required_devices}, cuda_available={available}, visible_devices={count}"
    # Do not catch probe/initialization exceptions or interpret compute errors as capacity.
    if count == 0:
        raise RuntimeError(f"No visible CUDA devices ({observed}); GPU execution cannot start")
    if not available:
        raise RuntimeError(f"CUDA unavailable ({observed}); GPU execution cannot start")
    if count < required_devices:
        raise RuntimeError(f"Insufficient CUDA devices ({observed}); GPU execution cannot start")
    if local_rank is not None and not 0 <= local_rank < min(required_devices, count):
        raise ValueError(f"Invalid LOCAL_RANK={local_rank} ({observed})")
    return count


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_identity() -> dict:
    untracked = (
        subprocess.check_output(
            [
                "git",
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
                "tools",
                "megatron",
                "slackpipe/src",
                "slackpipe/include",
                "slackpipe/apps",
                "tests/unit_tests/pipeline_parallel",
            ],
            cwd=ROOT,
        )
        .decode()
        .split("\0")
    )
    return dict(
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        diff=hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD", "--"], cwd=ROOT)
        ).hexdigest(),
        untracked_source_sha256={
            name: digest(ROOT / name)
            for name in sorted(untracked)
            if name and Path(name).suffix in (".py", ".cc", ".h", ".sh")
        },
    )


def run(command: list, log: Path, timeout: int, env: dict) -> None:
    print(shlex.join(map(str, command)), flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        process = subprocess.Popen(
            list(map(str, command)),
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if code:
        raise RuntimeError(f"Command failed ({code}); stopped. See {log}")


def summarize_samples(samples_by_rep: list[list[float]]) -> dict[str, Any]:
    if not samples_by_rep or any(not rep for rep in samples_by_rep):
        raise ValueError("Every independent run must contain samples")
    samples = [value for rep in samples_by_rep for value in rep]
    rep_means = [statistics.fmean(rep) for rep in samples_by_rep if rep]
    mean = statistics.fmean(samples)
    stddev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    rep_stddev = statistics.stdev(rep_means) if len(rep_means) > 1 else 0.0
    ci95 = 1.96 * rep_stddev / math.sqrt(len(rep_means)) if len(rep_means) > 1 else 0.0
    return {
        "mean_ms": mean,
        "median_ms": statistics.median(samples),
        "stddev_ms": stddev,
        "rep_mean_stddev_ms": rep_stddev,
        "run_means_ms": rep_means,
        "within_run_stddev_ms": [statistics.pstdev(rep) for rep in samples_by_rep],
        "all_samples_by_run": samples_by_rep,
        "ci95_ms": ci95,
        "cv": stddev / mean if mean else 0.0,
        "min_ms": min(samples),
        "max_ms": max(samples),
        "num_iterations": len(samples),
        "num_repetitions": len(samples_by_rep),
    }


def write_json(path: Path, payload: object) -> None:
    """Atomically publish a completed workflow artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)
