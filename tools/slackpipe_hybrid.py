# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Offline manifest/class-fit utility; execution uses run_slackpipe_eval.py."""

import argparse
import json
from pathlib import Path

from megatron.core.pipeline_parallel.slackpipe.cost_profile import build_heterogeneous_cost_profile
from megatron.core.pipeline_parallel.slackpipe.hybrid import (
    NEMOTRON_H_8B_SOURCE,
    nemotron_h_8b_config,
)
from megatron.core.pipeline_parallel.slackpipe.manifest import build_model_manifest
from megatron.core.pipeline_parallel.slackpipe.profile_integrity import publish_profile
from megatron.core.pipeline_parallel.slackpipe.stage_residual import (
    add_residual_arguments,
    residual_policy_from_args,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_residual_arguments(parser)
    parser.add_argument("action", choices=("manifest", "fit"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--observations", type=Path, help="stage-level rows from ordinary Megatron calibration"
    )
    parser.add_argument(
        "--manifest", type=Path, help="actual model_manifest.v1, defaults to Nemotron-H Base-8K"
    )
    parser.add_argument(
        "--calibration-pp", type=int, help="physical PP used to collect fit observations"
    )
    parser.add_argument(
        "--calibration-vpp", type=int, help="chunks per rank used to collect fit observations"
    )
    args = parser.parse_args()
    manifest = (
        json.loads(args.manifest.read_text())
        if args.manifest
        else build_model_manifest(nemotron_h_8b_config())
    )
    if args.action in ("manifest", "fit"):
        payload = manifest
        if args.action == "fit":
            if not args.observations:
                parser.error("fit requires --observations")
            if (
                not args.calibration_pp
                or args.calibration_pp < 1
                or not args.calibration_vpp
                or args.calibration_vpp < 1
            ):
                parser.error("fit requires positive --calibration-pp and --calibration-vpp")
            payload = build_heterogeneous_cost_profile(
                model_manifest=manifest,
                observed_stage_rows=json.loads(args.observations.read_text()),
                residual_policy=residual_policy_from_args(args),
                model_config=manifest["model_config"],
                parallel_config={
                    "pp": args.calibration_pp,
                    "vpp": args.calibration_vpp,
                    "tp": 1,
                    "dp": 1,
                    "cp": 1,
                },
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.action == "fit":
            publish_profile(args.output, payload)
        else:
            args.output.write_text(json.dumps(payload, indent=2) + "\n")
        source = args.manifest or NEMOTRON_H_8B_SOURCE
        print(f"Wrote {args.output}; architecture source: {source}")
        return


if __name__ == "__main__":
    main()
