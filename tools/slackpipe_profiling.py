# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Sequential stage/layer collection and explicit artifact selection, without retries."""

import json
import shutil
import subprocess
from pathlib import Path

from megatron.core.pipeline_parallel.slackpipe.isolated_profile import (
    POLICY_VERSION as LAYER_POLICY,
)
from megatron.core.pipeline_parallel.slackpipe.stage_residual import residual_policy_from_args
from tools.slackpipe_eval_common import digest, write_json
from tools.slackpipe_eval_config import METHODS, schedule_topology


def validate_collection_capabilities(args, phase=None):
    """Reject unsupported *new* stage collection before any prerequisite worker."""
    phase = args.stage if phase is None else phase
    needs_profiles = phase == "calibrate" or (
        METHODS[args.schedule].requires_solve
        and phase in ("solve", "smoke", "benchmark", "trace", "full")
    )
    if not needs_profiles or args.profiling_method == "layer" or args.stage_profile is not None:
        return
    stage_calibration_schedule(args)


def stage_calibration_schedule(args):
    """The sole schedule supported by the current stage timing hooks."""
    try:
        schedule_topology("interleaved-1f1b", args.pp, args.logical_stages, args.microbatches)
    except ValueError as exc:
        raise ValueError(
            "New stage profiling requires native interleaved calibration: "
            "PP in {2,4}, N>PP, N%PP=0 (VPP>=2), B%PP=0. "
            f"Received PP={args.pp}, N={args.logical_stages}, B={args.microbatches}. "
            "Use a supported topology, explicitly select --profiling-method layer, "
            "or supply --stage-profile for import; no topology/backend is changed."
        ) from exc
    return "interleaved-1f1b"


def backend_identity(profile: dict) -> str:
    backend = profile.get("profiling_backend")
    if backend in ("stage", "layer"):
        return backend
    estimator = profile.get("estimator", profile.get("measurement", {}).get("estimator"))
    if estimator == "isolated-layer-compute-v1":
        return "layer"
    # Historical layer artifacts embed the estimator in their context.
    if profile.get("context", {}).get("estimator") == "isolated-layer-compute-v1":
        return "layer"
    if profile.get("observed_stages") is not None:
        return "stage"
    raise ValueError("Artifact has no identifiable profiling backend")


def profiling_policy(args) -> dict:
    requested = ("stage", "layer") if args.profiling_method == "both" else (args.profiling_method,)
    backends = {}
    for backend in ("stage", "layer"):
        path = getattr(args, f"{backend}_profile")
        if backend not in requested and path is None:
            continue
        if path:
            try:
                checksum = digest(path)
            except OSError:
                checksum = None  # The backend records the read failure; no implicit collection.
            backends[backend] = dict(source="explicit_artifact", path=str(path), sha256=checksum)
        elif backend == "stage":
            backends[backend] = dict(
                source="collect",
                warmups=args.calibration_warmups,
                iterations=args.calibration_iterations,
                policy=residual_policy_from_args(args).identity(),
                measurement_parallelism="target-native-pipeline",
                calibration_schedule="interleaved-1f1b",
            )
        else:
            backends[backend] = dict(
                source="collect",
                warmups=args.isolated_profile_warmups,
                iterations=args.isolated_profile_iterations,
                policy=dict(policy_version=LAYER_POLICY),
                measurement_parallelism=dict(pp=1, vpp=None, tp=1, dp=1, cp=1),
                calibration_schedule="isolated-unit",
            )
    return dict(
        profiling_method=args.profiling_method,
        solver_profile=args.solver_profile,
        solver_profile_explicit=args.solver_profile_explicit,
        backends=backends,
    )


def collect_profiles(directory: Path, args, collect, validate) -> dict:
    """Call each requested producer at most once, retaining failed and partial outputs."""
    validate_collection_capabilities(args, "calibrate")
    policy = profiling_policy(args)
    outcomes = {}
    for backend, identity in policy["backends"].items():
        folder = directory / backend
        folder.mkdir(parents=True)
        outcome = dict(backend=backend, identity=identity, status="failed")
        try:
            if identity["source"] == "explicit_artifact":
                source = Path(identity["path"])
                profile_path = folder / "cost_profile.json"
                shutil.copyfile(source, profile_path)
                if identity["sha256"] is None or digest(profile_path) != identity["sha256"]:
                    raise ValueError("Explicit artifact changed during collection")
            else:
                collect(backend, folder / "worker")
                profile_path = folder / "worker" / "cost_profile.json"
            profile = json.loads(profile_path.read_text())
            if backend_identity(profile) != backend:
                raise ValueError(f"Explicit/collected profile is not {backend}")
            validate(profile)
            outcome.update(
                status="completed",
                profile=str(profile_path),
                sha256=digest(profile_path),
                cost_profile_hash=profile.get("cost_profile_hash"),
                policy=profile.get("profiling_policy", {"policy_version": "legacy-unspecified"}),
                measurement_parallel_config=profile.get("measurement_parallel_config"),
                target_parallel_config=profile.get(
                    "target_parallel_config", profile.get("parallel_config")
                ),
            )
        except (
            ValueError,
            KeyError,
            TypeError,
            OSError,
            RuntimeError,
            subprocess.TimeoutExpired,
        ) as exc:
            # A failed nonselected backend is a reported partial collection, not a fallback.
            outcome["error"] = str(exc)
            outcome["error_type"] = type(exc).__name__
        outcomes[backend] = outcome
        write_json(folder / "receipt.json", outcome)
    selected = outcomes[args.solver_profile]
    result = dict(
        solver_profile=args.solver_profile,
        backends=outcomes,
        status=(
            "completed" if all(o["status"] == "completed" for o in outcomes.values()) else "partial"
        ),
        selected_profile=selected.get("profile"),
    )
    if selected["status"] != "completed":
        result["status"] = "selected_backend_failed"
    write_json(directory / "collection.json", result)
    if result["status"] == "selected_backend_failed":
        raise RuntimeError(
            f"Selected {args.solver_profile} profiling failed; no fallback: {selected.get('error')}"
        )
    return result
