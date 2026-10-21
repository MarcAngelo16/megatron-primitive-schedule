# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Diagnostic comparison only: isolated range predictions need not equal pipeline latency."""

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

from megatron.core.pipeline_parallel.slackpipe.isolated_profile import ESTIMATOR
from megatron.core.pipeline_parallel.slackpipe.profile_integrity import validate_profile_integrity
from tools.slackpipe_compare_compute_timing import load_run


def compare(profile: dict, pipeline: dict) -> list[dict]:
    validate_profile_integrity(profile)
    if (
        profile.get("estimator") != ESTIMATOR
        or profile["model_manifest_hash"] != pipeline["manifest"]
    ):
        raise ValueError("Different model manifest or non-isolated profile")
    for item in pipeline["metadata"].values():
        c = item["calibration_configuration"]
        for old, new in (
            ("seq_length", "sequence_length"),
            ("micro_batch_size", "micro_batch_size"),
            ("precision", "dtype"),
        ):
            if c[old] != profile["model_config"][new]:
                raise ValueError(f"Incomparable {old}")
    groups = defaultdict(lambda: defaultdict(list))
    for e in pipeline["events"]:
        key = (
            e["calibration_group_id"],
            e["worker"],
            e["logical_stage"],
            tuple(e["stage_layer_range"]),
            e["stage_role"],
            e["phase"],
        )
        groups[key][e["iteration"]].append(e["elapsed_ms"])
    rows = []
    for (group, worker, stage, bounds, role, phase), samples in sorted(groups.items()):
        direction = phase.removesuffix("_compute")
        a, b = bounds
        prefix = profile["prefix_" + direction + "_us"]
        compute = (prefix[b] - prefix[a]) / 1000
        rows.append(
            dict(
                partition=group,
                worker=worker,
                stage=stage,
                layer_range=bounds,
                phase=direction,
                isolated_layers_ms=compute,
                isolated_stage_ms=compute + profile["stage_role_bias_us"][role][direction] / 1000,
                pipeline_wall_ms=statistics.median([statistics.fmean(v) for v in samples.values()]),
                iteration_means_ms={str(i): statistics.fmean(v) for i, v in samples.items()},
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = args.profile.read_bytes()
    pipeline = load_run(args.pipeline, diagnostic=False)
    result = dict(
        schema_version="slackpipe.isolated_pipeline_comparison.v1",
        diagnostic_only=True,
        definition="median of all raw per-iteration microbatch means; no equality acceptance gate",
        inputs={
            str(args.profile.resolve()): hashlib.sha256(data).hexdigest(),
            **pipeline["hashes"],
        },
        rows=compare(json.loads(data), pipeline),
    )
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
