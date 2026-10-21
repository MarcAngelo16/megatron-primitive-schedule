# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fixture-only heterogeneous calibration using ordinary interleaved Megatron.

Run with two-rank torchrun in slackpipe-dev. Solver and benchmark artifacts use
the same model.json; the benchmark accepts --heterogeneous-config model.json.
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import (
    begin_slackpipe_cost_calibration_iteration,
    end_slackpipe_cost_calibration_iteration,
)
from megatron.core.pipeline_parallel.slackpipe.cost_profile import (
    aggregate_stage_costs,
    build_heterogeneous_cost_profile,
    profile_fingerprint,
    stage_role,
)
from megatron.core.pipeline_parallel.slackpipe.manifest import (
    build_model_manifest,
    validate_chunk_layers,
)
from megatron.core.pipeline_parallel.slackpipe.plan import parse_slackpipe_plan
from tests.unit_tests.pipeline_parallel import slackpipe_perf_benchmark as bench
from tests.unit_tests.pipeline_parallel.test_slackpipe_heterogeneous import (
    block_config_payload,
    hand_plan,
    make_config,
    select_partitions,
)
from tests.unit_tests.test_utilities import Utils, clear_nvte_env_vars


def main():
    from megatron.core.pipeline_parallel.slackpipe.profile_integrity import (
        publish_profile,
        tag_calibration_events,
    )
    from megatron.core.pipeline_parallel.slackpipe.stage_residual import (
        add_residual_arguments,
        residual_policy_from_args,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    add_residual_arguments(parser)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup-iterations", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--seq-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.000001)
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="calibrate the tiny native Mamba/attention/MLP fixture",
    )
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Refusing to overwrite calibration data; use a fresh --output-dir")
    if args.warmup_iterations < 1:
        parser.error("at least one training warmup is required")
    if args.iterations < 1:
        parser.error("at least one calibration iteration is required")
    args.num_layers, args.micro_batch_size, args.vocab_size = 12, 1, 128
    args.num_microbatches, args.seed = 4, 1234
    args.heterogeneous_config = args.output_dir / "model.json"
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    clear_nvte_env_vars()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.hybrid:
        from megatron.core.pipeline_parallel.slackpipe.hybrid import partition_hybrid_pattern
        from tests.unit_tests.pipeline_parallel.test_slackpipe_hybrid import (
            PATTERN,
            hybrid_plan,
            hybrid_provider,
            tiny_config,
        )
        from tests.unit_tests.pipeline_parallel.test_slackpipe_model_construction import (
            _build_model,
        )

        if (args.hidden_size, args.num_attention_heads) != (128, 4) or args.seq_length > 64:
            parser.error("tiny hybrid calibration requires H=128, heads=4, sequence length <=64")
        manifest = build_model_manifest(tiny_config())
    else:
        manifest = build_model_manifest(
            make_config(hidden_size=args.hidden_size, num_attention_heads=args.num_attention_heads)
        )
    partitions, diagnostics = select_partitions(manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        args.heterogeneous_config.write_text(
            json.dumps(
                (
                    {"hybrid_layer_pattern": PATTERN}
                    if args.hybrid
                    else block_config_payload(args.hidden_size, args.num_attention_heads)
                ),
                indent=2,
            )
        )
        (args.output_dir / "model_manifest.json").write_text(json.dumps(manifest, indent=2))
    Utils.initialize_model_parallel(1, 2, virtual_pipeline_model_parallel_size=2)
    parallel_state.set_virtual_pipeline_model_parallel_world_size(2)
    dist.barrier()
    observed, raw_events, actual_layers, collection_metadata = [], [], [], []
    try:
        for partition_id, cuts in enumerate(partitions):
            split = [b - a for a, b in zip(cuts, cuts[1:])]
            mode = {
                "name": f"partition_{partition_id}",
                "schedule": "default",
                "layout": bench._layout_from_split(split),
            }
            if args.hybrid:
                construction_plan = hybrid_plan(2)
                construction_plan["layer_cuts"] = cuts
                pattern = partition_hybrid_pattern(PATTERN, parse_slackpipe_plan(construction_plan))
                model = _build_model(
                    tiny_config(2, 2), pp_size=2, vpp=2, provider=hybrid_provider(pattern)
                )
            else:
                model = bench._build_model(args, mode)
                construction_plan = hand_plan(2, cuts)
            construction_plan["model_manifest_hash"] = manifest["manifest_hash"]
            rows = validate_chunk_layers(parse_slackpipe_plan(construction_plan), model, rank)
            actual_layers.append({"cuts": cuts, "layers": rows})
            optimizer = torch.optim.SGD(
                bench._logical_params(model).values(), lr=args.learning_rate
            )
            batches = bench._make_batches(args)
            step = bench.make_training_step(args, mode, model, optimizer, batches)
            before_warmup = bench.memory_sample()
            for _ in range(args.warmup_iterations):
                losses = step(0)
            bench._assert_finite(losses, bench._logical_params(model))
            after_warmup = bench.memory_sample()
            events = []
            for iteration in range(
                args.warmup_iterations, args.warmup_iterations + args.iterations
            ):
                begin_slackpipe_cost_calibration_iteration(iteration)
                try:
                    losses = step(iteration)
                finally:
                    events.extend(end_slackpipe_cost_calibration_iteration())
            bench._assert_finite(losses, bench._logical_params(model))
            tag_calibration_events(
                events,
                group=str(partition_id),
                ranges=list(zip(cuts, cuts[1:])),
                worker=rank,
                raw_path=args.output_dir / "calibration_events.json",
                configuration=dict(
                    manifest=manifest["manifest_hash"],
                    seq_length=args.seq_length,
                    dtype="fp32",
                    microbatches=4,
                    seed=args.seed,
                    learning_rate=args.learning_rate,
                    warmups=args.warmup_iterations,
                    iterations=args.iterations,
                    environment=bench.environment_metadata(),
                    timing="synchronized-stage-wall-time-v1",
                ),
            )
            gathered = [None] * 2
            dist.all_gather_object(gathered, events)
            rank_metadata = [None] * 2
            dist.all_gather_object(
                rank_metadata,
                dict(
                    rank=rank,
                    partition=partition_id,
                    cuts=cuts,
                    environment=bench.environment_metadata(),
                    memory_before_warmup=before_warmup,
                    memory_after_warmup=after_warmup,
                    memory_end=bench.memory_sample(),
                ),
            )
            collection_metadata.extend(rank_metadata)
            if rank == 0:
                events = [e for worker in gathered for e in worker]
                raw_events.append({"cuts": cuts, "events": events})
                rows = aggregate_stage_costs(
                    events,
                    layer_split=split,
                    num_microbatches=4,
                    iteration_start=args.warmup_iterations,
                    iteration_end=args.warmup_iterations + args.iterations - 1,
                )
                for s, row in enumerate(rows):
                    layer_classes = [
                        x["config_class"] for x in manifest["layers"][cuts[s] : cuts[s + 1]]
                    ]
                    row.update(
                        partition=partition_id,
                        stage_layer_range=[cuts[s], cuts[s + 1]],
                        stage_role=stage_role(s, 4),
                        class_counts={
                            k: layer_classes.count(k) for k in sorted(set(layer_classes))
                        },
                    )
                observed.extend(rows)
                print(f"CALIBRATED {cuts}: {rows}", flush=True)
            del model, optimizer, batches, step
            gc.collect()
            torch.cuda.empty_cache()
            dist.barrier()
        (args.output_dir / f"calibration_layers.rank{rank}.json").write_text(
            json.dumps(actual_layers, indent=2)
        )
        if rank == 0:
            (args.output_dir / "calibration_events.json").write_text(
                json.dumps(raw_events, indent=2)
            )
            (args.output_dir / "observations.json").write_text(json.dumps(observed, indent=2))
            profile = build_heterogeneous_cost_profile(
                model_manifest=manifest,
                observed_stage_rows=observed,
                residual_policy=residual_policy_from_args(args),
                model_config={
                    **manifest["model_config"],
                    "num_layers": 12,
                    "sequence_length": args.seq_length,
                    "vocab_size": 128,
                    "micro_batch_size": 1,
                    "dtype": "fp32",
                },
                parallel_config={"pp": 2, "vpp": 2, "tp": 1, "dp": 1, "cp": 1},
            )
            profile["calibration"] = {
                "collection_mode": "calibration",
                "run_id": str(args.output_dir),
                "rank_partition_metadata": collection_metadata,
                "environment": bench.environment_metadata(),
                "profiler": bench.profiler_settings(bench.CollectionOptions()),
                "measurement_global_iterations": list(
                    range(args.warmup_iterations, args.warmup_iterations + args.iterations)
                ),
                "schedule": "megatron_interleaved_1f1b",
                "partitions": partitions,
                "warmup_iterations": args.warmup_iterations,
                "measured_iterations": args.iterations,
                "num_microbatches": 4,
                "learning_rate": args.learning_rate,
                "seed": args.seed,
                "initialization": "per-logical-parameter seed; normal 0.02 with residual projections scaled by sqrt(2L)",
                "device": torch.cuda.get_device_name(),
                "torch_version": torch.__version__,
                "estimator": "median iteration compute total / microbatches",
                "design": diagnostics,
                "observed_stages": observed,
            }
            profile["cost_profile_hash"] = profile_fingerprint(profile)
            for phase in ("forward", "backward"):
                np.testing.assert_allclose(
                    np.diff(profile[f"prefix_{phase}_us"]),
                    [r[f"{phase}_us"] for r in profile["layer_costs_us"]],
                    rtol=1e-12,
                )
            publish_profile(args.output_dir / "cost_profile.json", profile)
            print(
                json.dumps(
                    {
                        "fit": profile["fit"],
                        "class_costs_us": profile["class_costs_us"],
                        "stage_role_bias_us": profile["stage_role_bias_us"],
                    },
                    indent=2,
                ),
                flush=True,
            )
    finally:
        parallel_state.destroy_model_parallel()
        parallel_state.set_virtual_pipeline_model_parallel_world_size(None)
        if dist.is_initialized():
            dist.destroy_process_group()
    if rank == 0 and not (args.output_dir / "cost_profile.json").is_file():
        raise SystemExit(2)


if __name__ == "__main__":
    main()
