# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Artifact structure, identity and publication. No statistical quality gate."""

import hashlib
import json
import math
from pathlib import Path


def validate_profile_integrity(profile):
    from .cost_profile import profile_fingerprint

    if "profiling_policy" in profile and not isinstance(profile["profiling_policy"], dict):
        raise ValueError("Profiling policy must be an object")
    if profile.get("profiling_backend") == "layer" and "isolated" not in profile:
        raise ValueError("Layer profile requires isolated measurement evidence")
    if "isolated" in profile and profile.get("profiling_backend") not in (None, "layer"):
        raise ValueError("Profiling backend differs from isolated measurement evidence")
    version = profile.get("schema_version")
    if version not in ("slackpipe.cost_profile.v1", "slackpipe.cost_profile.v2"):
        raise ValueError("Unsupported cost profile schema")
    if "cost_profile_hash" in profile and profile["cost_profile_hash"] != profile_fingerprint(
        profile
    ):
        raise ValueError("Cost profile hash mismatch")

    def numbers(values):
        if not values or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
            for v in values
        ):
            raise ValueError("Cost values must be finite and nonnegative")

    if version.endswith("v1"):
        if profile.get("units") != "milliseconds":
            raise ValueError("v1 costs must use milliseconds")
        numbers([profile["a_fwd"], profile["a_bwd"]])
        for phase in ("fwd", "bwd"):
            numbers(profile["bias_" + phase])
        if len(profile["bias_fwd"]) != len(profile["bias_bwd"]):
            raise ValueError("Forward/backward stage counts differ")
        if "calibration_partition" in profile and (
            len(profile["calibration_partition"]) != len(profile["bias_fwd"])
            or any(type(v) is not int or v <= 0 for v in profile["calibration_partition"])
        ):
            raise ValueError("Invalid calibration partition")
    else:
        if profile.get("units") != "microseconds":
            raise ValueError("v2 costs must use microseconds")
        layers = profile["layer_costs_us"]
        if not layers or [l["layer_id"] for l in layers] != list(range(len(layers))):
            raise ValueError("Cost layers must be contiguous and nonempty")
        for phase in ("forward", "backward"):
            values = [l[phase + "_us"] for l in layers]
            numbers(values)
            prefix = profile["prefix_" + phase + "_us"]
            numbers(prefix)
            expected = [0.0]
            for value in values:
                expected.append(expected[-1] + value)
            if len(prefix) != len(expected) or any(
                not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-7)
                for a, b in zip(prefix, expected)
            ):
                raise ValueError("Prefix costs do not match layer costs")
            numbers([profile["stage_role_bias_us"][r][phase] for r in ("first", "middle", "last")])
    if "isolated" in profile:
        from .isolated_profile import validate_layer_profile

        validate_layer_profile(profile)
    elif "profiling_policy" in profile:
        _validate_stage_policy(profile)


def _validate_stage_policy(profile):
    """Reproduce declared stage selection from preserved raw inputs, not quality flags."""
    from .cost_profile import build_cost_profile, build_heterogeneous_cost_profile
    from .stage_residual import POLICY_VERSION, ResidualPolicy

    identity = profile["profiling_policy"]
    version = identity.get("policy_version")
    if version not in (POLICY_VERSION, "stage-summary-fit-v1"):
        raise ValueError("Unknown stage profiling strategy identity")
    try:
        filtering = profile["residual_filter"]
        policy = ResidualPolicy(filtering["k"], filtering["rho"])
        raw = profile["raw_observed_stages"]
        if profile["schema_version"].endswith("v1"):
            config = profile["microbatch_config"]
            events = [
                event
                for row in raw
                for phase in ("forward", "backward")
                for event in row[f"{phase}_diagnostics"]["raw_events"]
            ]
            rebuilt = build_cost_profile(
                events=events,
                model_config=profile["model_config"],
                parallel_config=profile["parallel_config"],
                layer_split=profile["calibration_partition"],
                num_microbatches=config["num_microbatches"],
                iteration_start=config["iteration_start"],
                iteration_end=config["iteration_end"],
                estimator=profile["estimator"],
                percentile_value=profile.get("percentile"),
                residual_policy=policy,
            )
            costs = ("a_fwd", "a_bwd", "bias_fwd", "bias_bwd")
        else:
            rebuilt = build_heterogeneous_cost_profile(
                model_manifest=dict(
                    manifest_hash=profile["model_manifest_hash"],
                    layers=[
                        dict(layer_id=l["layer_id"], config_class=l["config_class"])
                        for l in profile["layer_costs_us"]
                    ],
                ),
                observed_stage_rows=raw,
                model_config=profile["model_config"],
                parallel_config=profile["parallel_config"],
                residual_policy=policy,
            )
            costs = ("class_costs_us", "layer_costs_us", "stage_role_bias_us")

        # SVD/NNLS may differ in last-bit rounding across numerical libraries.
        def equivalent(a, b):
            if type(a) in (int, float) and type(b) in (int, float):
                return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)
            if isinstance(a, dict) and isinstance(b, dict):
                return a.keys() == b.keys() and all(equivalent(a[k], b[k]) for k in a)
            if isinstance(a, list) and isinstance(b, list):
                return len(a) == len(b) and all(equivalent(x, y) for x, y in zip(a, b))
            return type(a) is type(b) and a == b

        for key in (
            "profiling_policy",
            "raw_observed_stages",
            "observed_stages",
            "residual_filter",
            *costs,
        ):
            if not equivalent(profile[key], rebuilt[key]):
                raise ValueError(f"Stage {key} differs from declared policy/raw operations")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Incomplete stage profiling strategy evidence: {exc}") from exc


def publish_profile(path, profile):
    from .cost_profile import profile_fingerprint, write_cost_profile

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite profile: {path}")
    profile = dict(profile)
    profile["cost_profile_hash"] = profile_fingerprint(profile)
    validate_profile_integrity(profile)
    write_cost_profile(path, profile)
    return path


def tag_calibration_events(events, *, group, ranges, worker, configuration, raw_path, attempt=0):
    """Persist group identity; never infer group membership from ordering."""
    context = hashlib.sha256(
        json.dumps(configuration, sort_keys=True, default=str).encode()
    ).hexdigest()
    for event in events:
        stage = int(event["logical_stage"])
        event.update(
            calibration_group_id=str(group),
            attempt=attempt,
            worker=worker,
            stage_layer_range=list(ranges[stage]),
            stage_role=(
                "first" if stage == 0 else ("last" if stage == len(ranges) - 1 else "middle")
            ),
            measurement_context=context,
            raw_data_path=str(Path(raw_path).resolve()),
        )
