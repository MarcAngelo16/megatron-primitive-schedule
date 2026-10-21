# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Immutable no-OR solver -> plan -> optional two-GPU correctness reproduction.

Run in the prepared Megatron container. This is an abstract-cost correctness
fixture, not calibration, a performance benchmark, or a paper result.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true", help="Also execute PP=2 FP32 equivalence")
    parser.add_argument(
        "--method",
        choices=("slack-heuristic", "octopipe-tuning", "cpsat-no-hint"),
        default="slack-heuristic",
    )
    parser.add_argument("--solver", type=Path, help="Existing OR-enabled binary for cpsat-no-hint")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(
        schema_version="slackpipe.artifact_smoke.v1",
        started_at=datetime.now(timezone.utc).isoformat(),
        scope="abstract-cost correctness only; not a paper performance result",
        configuration=dict(B=4, N=4, W=2, L=8, forward=1, backward=2, communication=0),
        commands=[],
        status="running",
        gpu_requested=args.gpu,
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        source_status=subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ),
        source_sha256={},
    )
    for base in (
        "slackpipe/src",
        "slackpipe/include",
        "slackpipe/apps",
        "megatron/core/pipeline_parallel/slackpipe",
    ):
        for path in sorted((ROOT / base).rglob("*")):
            if path.suffix in (".py", ".cc", ".h"):
                metadata["source_sha256"][str(path.relative_to(ROOT))] = sha256(path)
    for name in (
        "slackpipe/CMakeLists.txt",
        "tools/slackpipe_artifact_smoke.py",
        "tests/unit_tests/pipeline_parallel/test_slackpipe_model_construction.py",
    ):
        metadata["source_sha256"][name] = sha256(ROOT / name)

    def run(command, name, env=None):
        command = list(map(str, command))
        started = time.monotonic()
        with (output / f"{name}.log").open("w") as log:
            result = subprocess.run(
                command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        metadata["commands"].append(
            dict(
                argv=command,
                returncode=result.returncode,
                wall_seconds=time.monotonic() - started,
                log=f"{name}.log",
            )
        )
        result.check_returncode()

    try:
        build = ROOT / "slackpipe/build/no-or"
        if args.method == "cpsat-no-hint" and args.solver is None:
            raise ValueError("cpsat-no-hint requires --solver with an OR-enabled binary")
        run(
            [
                "cmake",
                "-S",
                "slackpipe",
                "-B",
                build,
                "-G",
                "Ninja",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DSLACKPIPE_ENABLE_ORTOOLS=OFF",
            ],
            "configure",
        )
        run(["cmake", "--build", build, "-j2"], "build")
        binary = args.solver.resolve() if args.solver else build / "slackpipe_cli"
        metadata["binary_sha256"] = sha256(binary)
        metadata["build_info"] = json.loads(
            subprocess.check_output([binary, "--build-info"], text=True)
        )
        plan = output / "plan.json"
        run(
            [
                binary,
                "--algorithm",
                args.method,
                "--time-limit",
                1,
                "--B",
                4,
                "--N",
                4,
                "--J",
                2,
                "--L",
                8,
                "--ratio-num",
                2,
                "--ratio-den",
                1,
                "--communication",
                0,
                "--output-prefix",
                output / "solver",
                "--emit-plan",
                plan,
            ],
            "solver",
        )
        run([binary, "--validate-result", output / "solver.json"], "validate")
        from megatron.core.pipeline_parallel.slackpipe.plan import load_slackpipe_plan

        parsed = load_slackpipe_plan(plan, pipeline_model_parallel_size=2)
        assert (
            parsed.num_microbatches,
            parsed.num_stages,
            parsed.num_workers,
            parsed.num_layers,
        ) == (4, 4, 2, 8)
        assert sum(map(len, parsed.operations)) == 32
        metadata["plan_validation"] = dict(
            passed=True, layer_split=parsed.layer_split, operations=32
        )
        metadata["predicted_makespan_ticks"] = parsed.predicted_makespan
        if args.gpu:
            import torch

            if torch.cuda.device_count() < 2:
                raise RuntimeError(
                    "--gpu requires two visible CUDA GPUs; do not accept a skipped test"
                )
            metadata["hardware"] = [str(torch.cuda.get_device_properties(i)) for i in range(2)]
            metadata["software"] = dict(
                torch=torch.__version__, cuda=torch.version.cuda, nccl=torch.cuda.nccl.version()
            )
            env = dict(
                os.environ,
                SLACKPIPE_EXTERNAL_PP2_PLAN=str(plan),
                SLACKPIPE_EXTERNAL_TRACE_DIR=str(output / "runtime"),
                SLACKPIPE_TEST_TRANSPORT="nccl-p2p",
                CUDA_VISIBLE_DEVICES="0,1",
                TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="0",
            )
            run(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc-per-node",
                    2,
                    "-m",
                    "pytest",
                    "tests/unit_tests/pipeline_parallel/test_slackpipe_model_construction.py::test_slackpipe_pp2_numerical_equivalence",
                    "-q",
                    "--capture=fd",
                ],
                "gpu",
                env,
            )
            equivalence = json.loads((output / "runtime/equivalence.json").read_text())
            for key in (
                "initial_params_max_abs_diff",
                "loss_max_abs_diff",
                "gradients_max_abs_diff",
                "post_step_params_max_abs_diff",
            ):
                assert equivalence[key] == 0
            for rank in range(2):
                trace = json.loads(
                    (output / f"runtime/slackpipe_trace.rank{rank}.json").read_text()
                )
                assert trace["matched_plan"] is True
            metadata["gpu_equivalence"] = equivalence
        metadata["status"] = "passed"
    except Exception as exc:
        metadata.update(status="failed", error=str(exc))
        raise
    finally:
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        metadata["artifacts_sha256"] = {
            str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()
        }
        (output / "receipt.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Passed {metadata['scope']}; provenance: {output / 'receipt.json'}")


if __name__ == "__main__":
    main()
