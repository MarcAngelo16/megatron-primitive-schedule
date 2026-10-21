# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Shared launcher/worker parser. No launcher imports in the worker."""
import argparse
from pathlib import Path

from megatron.core.pipeline_parallel.slackpipe.stage_residual import (
    add_residual_arguments,
    residual_policy_from_args,
)
from tools.slackpipe_eval_common import ROOT
from tools.slackpipe_eval_config import (
    METHODS,
    SCHEDULES,
    canonical_schedule,
    load_model,
    schedule_topology,
    solver_arguments,
)
from tools.slackpipe_profiling import validate_collection_capabilities

STAGES = (
    "env",
    "correctness",
    "smoke",
    "calibrate",
    "solve",
    "benchmark",
    "trace",
    "full",
    "inspect",
)


def argument_parser() -> argparse.ArgumentParser:
    from megatron.core.pipeline_parallel.slackpipe.isolated_profile import add_arguments

    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    add_residual_arguments(parser)
    parser.add_argument("--phase", dest="stage", choices=STAGES, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--method", dest="schedule", choices=SCHEDULES, default="slack-heuristic")
    parser.add_argument(
        "--profiling-method",
        choices=("stage", "layer", "both"),
        default="both",
        help="Default both; new stage collection uses native interleaved PP=2/4, N>PP, N%%PP=0, B%%PP=0",
    )
    parser.add_argument(
        "--solver-profile",
        choices=("stage", "layer"),
        help="Omitted: both/stage select stage, layer selects layer; never falls back",
    )
    parser.add_argument("--stage-profile", type=Path, help="Explicit existing stage artifact")
    parser.add_argument("--layer-profile", type=Path, help="Explicit existing layer artifact")
    parser.add_argument("--pp", type=int, default=4)
    parser.add_argument("--num-stages", dest="logical_stages", type=int)
    parser.add_argument("--microbatches", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--seq-length", type=int)
    parser.add_argument("--precision", choices=("fp32", "bf16"))
    parser.add_argument("--transport", choices=("nccl-p2p", "nccl-rma"), default="nccl-p2p")
    parser.add_argument(
        "--solver", type=Path, default=ROOT / "slackpipe/build/release/slackpipe_cli"
    )
    parser.add_argument("--solver-seconds", type=float, default=300)
    parser.add_argument("--solver-workers", type=int, default=1)
    parser.add_argument("--activation-cap-units", type=int, nargs="+")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--calibration-iterations", type=int, default=10)
    parser.add_argument("--calibration-warmups", type=int, default=5)
    parser.add_argument("--smoke-warmups", type=int, default=5)
    parser.add_argument("--smoke-iterations", type=int, default=2)
    parser.add_argument("--trace-warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--run-index", type=int, help="Campaign repetition identity; does not change the data seed"
    )
    parser.add_argument("--profiler-wait", type=int, default=2)
    parser.add_argument("--profiler-warmup", type=int, default=2)
    parser.add_argument("--profiler-active", type=int, default=3)
    parser.add_argument("--profiler-repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--explain-receipt", action="store_true")
    return parser


def normalize_args(args, *, worker=False):
    if not hasattr(args, "solver_profile_explicit"):
        args.solver_profile_explicit = args.solver_profile is not None
    if args.solver_profile is None:
        args.solver_profile = "layer" if args.profiling_method == "layer" else "stage"
    if args.profiling_method not in ("both", args.solver_profile) and not getattr(
        args, f"{args.solver_profile}_profile"
    ):
        raise ValueError("Selected solver profile is not collected; supply its explicit artifact")
    residual_policy_from_args(args)
    args.schedule = canonical_schedule(args.schedule)
    if METHODS[args.schedule].requires_solve:
        solver_arguments(
            args.schedule, args.solver_seconds, args.solver_workers, args.activation_cap_units
        )
    elif args.activation_cap_units is not None:
        raise ValueError("Native methods do not support solver-enforced activation caps")
    model = load_model(args.model_config)
    for field in ("seq_length", "precision", "micro_batch_size"):
        if getattr(args, field) is None:
            setattr(args, field, model["defaults"][field])
    topology = schedule_topology(args.schedule, args.pp, args.logical_stages, args.microbatches)
    args.logical_stages = topology["num_stages"]
    if (
        topology["num_stages"] > model["num_layers"]
        or not 1 <= args.seq_length <= model["max_sequence_length"]
    ):
        raise ValueError("Invalid stages/sequence for model")
    if (
        min(
            args.warmups,
            args.iterations,
            args.repetitions,
            args.micro_batch_size,
            args.timeout,
            args.solver_seconds,
            args.calibration_warmups,
            args.smoke_warmups,
            args.smoke_iterations,
            args.trace_warmups,
        )
        < 1
    ):
        raise ValueError("Positive warmup/measurement/run/batch/timeout required")
    if (
        args.calibration_iterations < 1
        or min(args.profiler_wait, args.profiler_warmup) < 0
        or min(args.profiler_active, args.profiler_repeat) < 1
    ):
        raise ValueError("Invalid calibration/profiler schedule")
    if args.run_index is not None and args.run_index < 0:
        raise ValueError("run-index must be non-negative")
    if args.isolated_profile_warmups < 1 or args.isolated_profile_iterations < 1:
        raise ValueError("Positive layer warmup/sample count required")
    args.output, args.model_config, args.solver = (
        p.resolve() for p in (args.output, args.model_config, args.solver)
    )
    for backend in ("stage", "layer"):
        path = getattr(args, f"{backend}_profile")
        if path is not None:
            setattr(args, f"{backend}_profile", path.resolve())
    # Execution workers consume an already bound plan/profile; they never resolve
    # the launcher's profiling prerequisites for smoke/benchmark/trace.
    if not worker or args.stage == "calibrate":
        validate_collection_capabilities(args)
    return args
