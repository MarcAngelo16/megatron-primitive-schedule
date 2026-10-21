# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Read-only raw-event replay for every recorded stage/phase observation.

This is evidence collection, not an iteration filter or a solver authorization.
Run with ``python -m tools.slackpipe_inspect_iterations --help``.
"""

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from megatron.core.pipeline_parallel.slackpipe.profile_integrity import validate_profile_integrity

IDENTITY_FIELDS = (
    "calibration_group_id",
    "attempt",
    "worker",
    "stage_layer_range",
    "stage_role",
    "measurement_context",
    "raw_data_path",
)


def _key(record: dict, *, event: bool = False) -> str:
    identity = {key: record.get(key) for key in IDENTITY_FIELDS}
    if any(value is None for value in identity.values()):
        raise ValueError(
            "Raw replay requires explicit group/attempt/range/worker/context provenance"
        )
    identity["stage"] = record["logical_stage" if event else "stage"]
    return json.dumps(identity, sort_keys=True)


def _check_values(actual: list, expected: list, name: str) -> None:
    if len(actual) != len(expected) or any(
        not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) for a, b in zip(actual, expected)
    ):
        raise ValueError(f"Raw replay disagrees with saved {name}")


def inspect_iterations(profile: dict, raw: list) -> dict:
    """Replay all raw observations without filtering or changing any input.

    Accept either rank-local event lists or the collector's list of partition
    bundles. Group identity always comes from events, never bundle order. Require
    exact microbatch coverage and agreement with the saved raw sample evidence.
    """
    validate_profile_integrity(profile)
    events = []
    for item in raw:
        events.extend(item["events"] if "events" in item else [item])
    buckets = defaultdict(list)
    for event in events:
        if event.get("phase") in ("forward_compute", "backward_compute"):
            buckets[(_key(event, event=True), event["phase"].removesuffix("_compute"))].append(
                event
            )
    groups, seen = [], set()
    for row in profile["observed_stages"]:
        identity = _key(row)
        if identity in seen:
            raise ValueError("Duplicate profile observation identity")
        seen.add(identity)
        for phase in ("forward", "backward"):
            diagnostic = row[f"{phase}_diagnostics"]
            iterations = diagnostic["global_iterations"]
            if not iterations or iterations != sorted(set(iterations)):
                raise ValueError("Expected unique ordered global iteration IDs")
            raw_events = buckets[(identity, phase)]
            selected = [e for e in raw_events if e["iteration"] in iterations]
            if not selected:
                raise ValueError("Missing raw events for stage/phase/context")
            microbatches = diagnostic["num_microbatches"]
            if type(microbatches) is not int or microbatches < 1:
                raise ValueError("Invalid microbatch count")
            expected = diagnostic["raw_events"]
            order = lambda e: (e["iteration"], e["microbatch"])
            selected = sorted(selected, key=order)
            expected = sorted(expected, key=order)
            if (
                len(selected) != diagnostic["raw_sample_count"]
                or [order(e) for e in selected] != [order(e) for e in expected]
                or any(
                    _key(e, event=True) != identity or e["phase"] != phase + "_compute"
                    for e in expected
                )
            ):
                raise ValueError("Raw replay disagrees with saved operation inventory")
            _check_values(
                [float(e["elapsed_ms"]) for e in selected],
                [float(e["elapsed_ms"]) for e in expected],
                "raw operation times",
            )
            by_iteration = defaultdict(list)
            for event in selected:
                if not math.isfinite(float(event["elapsed_ms"])) or float(event["elapsed_ms"]) <= 0:
                    raise ValueError("Operation timings must be finite and positive")
                by_iteration[event["iteration"]].append(event)
            details = []
            for iteration in iterations:
                group = by_iteration[iteration]
                ids = [e.get("microbatch") for e in group]
                if len(ids) != microbatches or set(ids) != set(range(microbatches)):
                    raise ValueError("Missing or duplicate microbatch operation in raw events")
                values = [float(e["elapsed_ms"]) for e in group]
                details.append(
                    dict(
                        iteration=iteration,
                        ranks=sorted({e["rank"] for e in group if "rank" in e}),
                        raw_microbatch_count=len(group),
                        aggregate_ms=sum(values),
                        mean_ms=statistics.fmean(values),
                        median_ms=statistics.median(values),
                        min_ms=min(values),
                        max_ms=max(values),
                    )
                )
            values = [d["mean_ms"] for d in details]
            groups.append(
                dict(
                    **{k: row[k] for k in IDENTITY_FIELDS},
                    stage=row["stage"],
                    phase=phase,
                    layer_composition=row.get("layer_composition"),
                    model_manifest_hash=row.get("model_manifest_hash"),
                    median_iteration_value_ms=statistics.median(values),
                    raw_iteration_count=len(iterations),
                    ignored_events_outside_measurement_window=len(raw_events) - len(selected),
                    iterations=details,
                )
            )
    return dict(
        schema_version="slackpipe.iteration_inspection.v2",
        diagnostic_only=True,
        iteration_filter_enabled=False,
        definitions=dict(
            aggregate_ms="sum of all raw microbatch operation times within the iteration",
            mean_ms="per-operation iteration mean, including any samples excluded by the stage fit",
        ),
        stage_phase_count=len(groups),
        groups=groups,
        boundary_attribution="unavailable from elapsed-only raw events; no clock/allocator/communication attribution inferred",
    )


def main() -> int:
    """Export a new diagnostic file; never overwrite profiles or original events."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        paths = dict(profile=args.profile, events=args.events)
        inputs = {name: path.read_bytes() for name, path in paths.items()}
        report = inspect_iterations(*(json.loads(inputs[k]) for k in ("profile", "events")))
        report["inputs"] = {
            name: dict(path=str(path.resolve()), sha256=hashlib.sha256(inputs[name]).hexdigest())
            for name, path in paths.items()
        }
        serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
        with args.output.open("x") as stream:
            stream.write(serialized)
        print(f"Diagnostic only: {report['stage_phase_count']} stage/phase groups; all raw samples")
        for group in report["groups"]:
            print(f"{group['calibration_group_id']} s{group['stage']} {group['phase']}")
            for item in group["iterations"]:
                print(
                    f"  i={item['iteration']} aggregate={item['aggregate_ms']:.6f} "
                    f"mean={item['mean_ms']:.6f} median={item['median_ms']:.6f} "
                    f"min={item['min_ms']:.6f} max={item['max_ms']:.6f} "
                    f"samples={item['raw_microbatch_count']}"
                )
        print(f"Saved {args.output}; input artifacts unchanged")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Iteration inspection failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
