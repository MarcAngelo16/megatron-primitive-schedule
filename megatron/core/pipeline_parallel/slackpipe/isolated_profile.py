# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Isolated measured-unit costs, deliberately independent of partition consensus."""

import hashlib
import math
import statistics

from .cost_profile import profile_fingerprint
from .manifest import canonical_json, manifest_fingerprint

ESTIMATOR = "isolated-layer-compute-v1"
LEGACY_ESTIMATOR = "existing-stage-wall-time-v1"
POLICY_VERSION = "layer-median-v1"


def class_mapping(manifest: dict, execution: dict) -> list[dict]:
    """Group only identical manifest units under the complete execution signature."""
    if manifest.get("manifest_hash") != manifest_fingerprint(manifest):
        raise ValueError("Invalid model manifest hash")
    classes = {}
    for i, layer in enumerate(manifest["layers"]):
        if layer["layer_id"] != i:
            raise ValueError("Manifest layers must be contiguous and ordered")
        signature = dict(
            execution=execution,
            manifest_model_config=manifest["model_config"],
            unit={k: v for k, v in layer.items() if k not in ("layer_id", "global_layer_id")},
        )
        key = hashlib.sha256(canonical_json(signature).encode()).hexdigest()
        classes.setdefault(
            key,
            dict(
                class_signature=key,
                signature=signature,
                layer_type=layer["layer_type"],
                representative_layer=i,
                member_layers=[],
            ),
        )["member_layers"].append(i)
    return list(classes.values())


def summarize(values: list[float]) -> dict:
    """Median of every legal measured sample; no exclusion or stability gate."""
    if not values or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
        for v in values
    ):
        raise ValueError("Isolated timings must be positive finite samples")
    return dict(
        mean_ms=statistics.fmean(values),
        median_ms=statistics.median(values),
        stddev_ms=statistics.pstdev(values),
        min_ms=min(values),
        max_ms=max(values),
        count=len(values),
        selected_ms=statistics.median(values),
    )


def build_profile(
    *,
    manifest: dict,
    classes: list[dict],
    units: list[dict],
    context: dict,
    model_config: dict,
    parallel_config: dict,
) -> dict:
    """Expand measured medians to ordered layers and the existing [i,j) v2 prefixes."""
    measured = {u["class_signature"]: u for u in units}
    if len(measured) != len(units) or set(measured) != {c["class_signature"] for c in classes} | {
        "first",
        "last",
    }:
        raise ValueError("Missing/duplicate/unexpected profiling units")
    summaries = {
        key: {phase: summarize(unit[phase + "_gpu_ms"]) for phase in ("forward", "backward")}
        for key, unit in measured.items()
    }
    layers = [None] * manifest["num_layers"]
    for c in classes:
        for i in c["member_layers"]:
            if not 0 <= i < len(layers) or layers[i] is not None:
                raise ValueError("Invalid class expansion")
            layers[i] = dict(
                layer_id=i,
                config_class=manifest["layers"][i]["config_class"],
                class_signature=c["class_signature"],
                **{
                    p + "_us": 1000 * summaries[c["class_signature"]][p]["selected_ms"]
                    for p in ("forward", "backward")
                },
            )
    if None in layers:
        raise ValueError("Class expansion does not cover every layer")
    prefixes = {}
    for phase in ("forward", "backward"):
        values = [0.0]
        for layer in layers:
            values.append(values[-1] + layer[phase + "_us"])
        prefixes["prefix_" + phase + "_us"] = values
    profile = dict(
        schema_version="slackpipe.cost_profile.v2",
        estimator=ESTIMATOR,
        profiling_backend="layer",
        profiling_policy=dict(policy_version=POLICY_VERSION),
        units="microseconds",
        model_manifest_hash=manifest["manifest_hash"],
        model_config=model_config,
        parallel_config=parallel_config,
        target_parallel_config=parallel_config,
        measurement_parallel_config=dict(pp=1, vpp=None, tp=1, dp=1, cp=1),
        measurement_definition=dict(
            measurement_mode="isolated",
            includes_pipeline_wait=False,
            includes_p2p=False,
            includes_communication=False,
            statistic="median of all legal GPU event samples; no exclusions",
            timing="synchronized current-stream events; isolated implementation time including launch gaps, not pure FLOP time",
            backward="real autograd with representative upstream gradient; parameter/input grads reset before each F/B pair",
            optimizer_included=False,
        ),
        isolated=dict(
            context=context, manifest=manifest, classes=classes, units=units, summaries=summaries
        ),
        layer_costs_us=layers,
        class_costs_us={
            c["class_signature"]: {
                p: 1000 * summaries[c["class_signature"]][p]["selected_ms"]
                for p in ("forward", "backward")
            }
            for c in classes
        },
        stage_role_bias_us={
            r: {
                p: 0.0 if r == "middle" else 1000 * summaries[r][p]["selected_ms"]
                for p in ("forward", "backward")
            }
            for r in ("first", "middle", "last")
        },
        **prefixes,
    )
    # Nested manifest must not expose a second schema_version to the C++ reader.
    profile["isolated"]["manifest"] = {k: v for k, v in manifest.items() if k != "schema_version"}
    if parallel_config["pp"] * (parallel_config.get("vpp") or 1) == 1:
        for phase in ("forward", "backward"):
            profile["stage_role_bias_us"]["first"][phase] += profile["stage_role_bias_us"]["last"][
                phase
            ]
    profile["cost_profile_hash"] = profile_fingerprint(profile)
    return profile


def validate_layer_profile(profile: dict) -> None:
    """Rebuild the current all-sample median profile, checking data identity only."""
    if profile.get("cost_profile_hash") != profile_fingerprint(profile):
        raise ValueError("Layer profile hash mismatch")
    try:
        data = profile["isolated"]
        current_policy = profile.get("profiling_policy", {}).get("policy_version")
        if current_policy is not None and (
            current_policy != POLICY_VERSION
            or data["context"].get("policy_version") != POLICY_VERSION
        ):
            raise ValueError("Layer profiling strategy identity mismatch")
        for unit in data["units"]:
            samples = unit["samples"]
            context = data["context"]
            if (
                any(
                    type(context[k]) is not int or context[k] < 0 for k in ("warmups", "iterations")
                )
                or context["iterations"] == 0
                or any(type(s["iteration"]) is not int for s in samples)
            ):
                raise ValueError(
                    "Isolated sampling requires integer bounds/indices and nonempty samples"
                )
            if [s["iteration"] for s in samples] != list(
                range(context["warmups"], context["warmups"] + context["iterations"])
            ):
                raise ValueError("Incomplete isolated samples")
            for phase in ("forward", "backward"):
                if unit[phase + "_gpu_ms"] != [s[phase + "_gpu_ms"] for s in samples]:
                    raise ValueError("Isolated raw samples and selected input differ")
        manifest = dict(data["manifest"], schema_version="slackpipe.model_manifest.v1")
        expected_classes = class_mapping(manifest, data["context"]["execution"])
        if data["classes"] != expected_classes:
            raise ValueError("Class mapping differs from execution/manifest")
        if current_policy is None:
            # Audit a historical artifact's recorded selection, never run the old filter.
            # Its policy and bytes stay historical, even when the old quality flag failed.
            selected = {}
            for unit in data["units"]:
                key = unit["class_signature"]
                if key in selected:
                    raise ValueError("Duplicate isolated unit")
                selected[key] = {}
                for phase in ("forward", "backward"):
                    values = unit[phase + "_gpu_ms"]
                    summarize(values)
                    summary = data["summaries"][key][phase]
                    excluded = summary["excluded_indices"]
                    if len(set(excluded)) != len(excluded) or any(
                        type(i) is not int or not 0 <= i < len(values) for i in excluded
                    ):
                        raise ValueError("Invalid historical selection indices")
                    retained = [v for i, v in enumerate(values) if i not in excluded]
                    if not retained or statistics.median(retained) != summary["selected_ms"]:
                        raise ValueError("Historical layer median differs from recorded selection")
                    selected[key][phase] = 1000 * summary["selected_ms"]
            if set(selected) != {c["class_signature"] for c in expected_classes} | {
                "first",
                "last",
            }:
                raise ValueError("Missing/unexpected historical layer units")
            if profile["model_manifest_hash"] != manifest["manifest_hash"]:
                raise ValueError("Historical layer manifest mismatch")
            for cls in expected_classes:
                key = cls["class_signature"]
                if profile["class_costs_us"][key] != selected[key]:
                    raise ValueError("Historical class expansion mismatch")
                for i in cls["member_layers"]:
                    layer = profile["layer_costs_us"][i]
                    if layer["class_signature"] != key or any(
                        layer[p + "_us"] != selected[key][p] for p in ("forward", "backward")
                    ):
                        raise ValueError("Historical layer expansion mismatch")
            single = (
                profile["parallel_config"]["pp"] * (profile["parallel_config"].get("vpp") or 1) == 1
            )
            for role in ("first", "middle", "last"):
                for phase in ("forward", "backward"):
                    expected = 0 if role == "middle" else selected[role][phase]
                    if single and role == "first":
                        expected += selected["last"][phase]
                    if profile["stage_role_bias_us"][role][phase] != expected:
                        raise ValueError("Historical boundary expansion mismatch")
            return
        rebuilt = build_profile(
            manifest=manifest,
            classes=expected_classes,
            units=data["units"],
            context=data["context"],
            model_config=profile["model_config"],
            parallel_config=profile["parallel_config"],
        )
        if rebuilt["isolated"]["summaries"] != data["summaries"]:
            raise ValueError("Isolated summaries differ from raw samples")
        for key in (
            "model_manifest_hash",
            "layer_costs_us",
            "class_costs_us",
            "prefix_forward_us",
            "prefix_backward_us",
            "stage_role_bias_us",
            "profiling_policy",
        ):
            if rebuilt[key] != profile[key]:
                raise ValueError(f"Invalid isolated {key}")
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"Invalid layer profile: {e}") from e


def add_arguments(parser) -> None:
    parser.add_argument("--isolated-profile-warmups", type=int, default=20)
    parser.add_argument("--isolated-profile-iterations", type=int, default=30)


def execution_signature(
    model: dict, *, seq_length: int, micro_batch_size: int, precision: str
) -> dict:
    """Conservative key for the strict eval-model adapter; unknown models are rejected by it."""
    return dict(
        model={k: v for k, v in model.items() if k != "schema_version"},
        sequence_length=seq_length,
        micro_batch_size=micro_batch_size,
        precision=precision,
        tp=1,
        dp=1,
        cp=1,
        recompute=None,
        implementation="Megatron TE GPT/hybrid_stack_spec; native untied head/loss; adapter-v1",
    )
