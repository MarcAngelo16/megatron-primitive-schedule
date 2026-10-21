# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Read-only normal/serialized diagnostic comparison; never emits solver costs."""

import argparse
import bisect
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

from megatron.core.pipeline_parallel.slackpipe.figure_trace import (
    LABEL,
    compact_profiler_trace,
    interval_union,
)
from tools.slackpipe_compare_compute_timing import load_run


def compact_partition(raw, metadata, execution):
    """Reuse the common correlator one completed step at a time (no sampling).

    Each diagnostic step drains the device before its CPU annotation ends, so
    the complete local step's launches and device activity are inside this range.
    """
    events = sorted((e for e in raw["traceEvents"] if "ts" in e), key=lambda e: e["ts"])
    times = [e["ts"] for e in events]
    labels = []
    for event in events:
        match = LABEL.fullmatch(event.get("name", ""))
        if (
            match
            and match[1] == "step"
            and event.get("ph") == "X"
            and event.get("cat") == "user_annotation"
        ):
            labels.append((event, int(match[2])))
    if [i for _, i in labels] != metadata["profiler"]["captured_iterations"]:
        raise ValueError("Trace capture iteration coverage mismatch")
    c = metadata["calibration_configuration"]
    measured = {
        str(e["iteration"]): dict(cuda_elapsed_ms=e["cuda_elapsed_ms"])
        for e in metadata["timing_diagnostic"]["iterations"]
    }
    merged = dict(gpu_activities=[], records=[])
    for label, iteration in labels:
        lo = bisect.bisect_left(times, label["ts"])
        hi = bisect.bisect_left(times, label["ts"] + label["dur"])
        part = compact_profiler_trace(
            dict(raw, traceEvents=events[lo:hi]),
            rank=metadata["rank"],
            mode=execution,
            transport="native-p2p",
            measured_steps=measured,
            config=dict(num_microbatches=c["microbatches"], num_stages=c["stages"], pp=c["pp"]),
        )
        offset = len(merged["gpu_activities"])
        for activity in part["gpu_activities"]:
            activity["activity_id"] += offset
        for record in part["records"]:
            record["activity_ids"] = [i + offset for i in record["activity_ids"]]
        merged["gpu_activities"].extend(part["gpu_activities"])
        merged["records"].extend(part["records"])
    return merged


def overlap_evidence(raw, compact):
    """GPU envelope evidence, not invented absolute CUDA-event timestamps.

    Kineto does not expose the event-pair device timestamps. Positive overlap
    inside a correlated GPU envelope is evidence; absence outside that envelope
    and the duration of an inserted stream dependency remain unknown.
    """
    runtime = {
        e.get("args", {}).get("correlation"): e
        for e in raw["traceEvents"]
        if e.get("cat") in ("cuda_runtime", "cuda_driver") and e.get("ph") == "X"
    }
    p2p = [
        e
        for e in raw["traceEvents"]
        if e.get("name", "").startswith("SlackPipeDiagnostic/P2P/") and e.get("ph") == "X"
    ]
    waits = [
        e
        for e in raw["traceEvents"]
        if e.get("ph") == "X"
        and e.get("cat") in ("cuda_runtime", "cuda_driver")
        and "StreamWaitEvent" in e.get("name", "")
    ]
    activities = {e["activity_id"]: e for e in compact["gpu_activities"]}
    gpu_by_start = sorted(activities.values(), key=lambda e: e["start_us"])
    gpu_starts = [e["start_us"] for e in gpu_by_start]
    longest = max((e["end_us"] - e["start_us"] for e in gpu_by_start), default=0)
    ops = [r for r in compact["records"] if r["kind"] in ("F", "B")]
    result = {}

    def intersects(e, start, end):
        return e["start_us"] < end and e["end_us"] > start

    def union_duration(items, start, end):
        return sum(
            b - a
            for a, b in interval_union(
                (max(start, e["start_us"]), min(end, e["end_us"])) for e in items
            )
        )

    for op in ops:
        start, end = op["start_us"], op["end_us"]
        candidates = gpu_by_start[
            bisect.bisect_left(gpu_starts, start - longest) : bisect.bisect_left(gpu_starts, end)
        ]
        concurrent = [e for e in candidates if intersects(e, start, end)]
        nccl = [e for e in concurrent if e["communication"]]
        early_p2p = []
        for e in nccl:
            launch = runtime.get(e["args"].get("correlation"))
            if (
                launch
                and launch["ts"] < op["cpu_start_us"]
                and any(
                    p["tid"] == launch["tid"]
                    and p["pid"] == launch["pid"]
                    and p["ts"] <= launch["ts"] < p["ts"] + p["dur"]
                    for p in p2p
                )
            ):
                early_p2p.append(e)
        co_resident = []
        for other in ops:
            if other["stage"] == op["stage"] or not intersects(other, start, end):
                continue
            shared = [
                activities[i]
                for i in other["activity_ids"]
                if intersects(activities[i], start, end) and not activities[i]["communication"]
            ]
            if shared:
                co_resident.append(
                    dict(
                        stage=other["stage"],
                        iteration=other["iteration"],
                        microbatch=other["microbatch"],
                        kind=other["kind"],
                        activity_ids=[e["activity_id"] for e in shared],
                        overlap_union_us=union_duration(shared, start, end),
                    )
                )
        own = [activities[i] for i in op["activity_ids"] if not activities[i]["communication"]]
        simultaneous_nccl = interval_union(
            (max(c["start_us"], n["start_us"]), min(c["end_us"], n["end_us"]))
            for c in own
            for n in nccl
            if intersects(n, c["start_us"], c["end_us"])
        )
        candidate_waits = [e for e in waits if op["cpu_start_us"] <= e["ts"] < op["cpu_end_us"]]
        result[(op["iteration"], op["stage"], op["microbatch"], op["kind"])] = dict(
            gpu_envelope_start_us=start,
            gpu_envelope_end_us=end,
            gpu_activity_union_us=op["gpu_activity_union_us"],
            internal_gap_us=op["internal_gap_us"],
            overlapping_nccl_in_gpu_envelope=bool(nccl),
            nccl_overlap_union_us=union_duration(nccl, start, end),
            nccl_simultaneous_compute_union_us=sum(b - a for a, b in simultaneous_nccl),
            nccl_activity_ids=[e["activity_id"] for e in nccl],
            earlier_p2p_activity_ids=[e["activity_id"] for e in early_p2p],
            co_resident_activity=co_resident,
            stream_wait_api_candidates=candidate_waits,
            compute_stream_wait="unknown: API insertion is not GPU blocking duration; raw stream identifiers require inspection",
            cuda_event_interval_overlap="unknown outside correlated GPU activity envelope",
            compute_activities=own,
            cuda_malloc_count=op["cuda_malloc_count"],
            cuda_malloc_cpu_union_us=op["cuda_malloc_cpu_union_us"],
            inferred_cause=(
                "unresolved; overlap is not proof of contention"
                if nccl or co_resident or candidate_waits
                else "unresolved; inspect kernels, launch gaps and unobserved event-boundary waits"
            ),
        )
    return result


def annotate_run(root, execution):
    run = load_run(root, diagnostic=True)
    evidence = {}
    for (group, worker), metadata in run["metadata"].items():
        diagnostic = metadata["timing_diagnostic"]
        if diagnostic.get("execution") != execution:
            raise ValueError("Unexpected diagnostic execution policy")
        sequence = {}
        by_iteration = defaultdict(list)
        for event in run["events"]:
            if event["calibration_group_id"] == group and event["worker"] == worker:
                by_iteration[event["iteration"]].append(event)
        for iteration, events in by_iteration.items():
            sequence[iteration] = sorted(
                [
                    dict(
                        type="compute",
                        **{
                            k: e[k]
                            for k in ("schedule_position", "logical_stage", "microbatch", "phase")
                        },
                    )
                    for e in events
                ]
                + [
                    dict(type="p2p", **e)
                    for item in diagnostic["iterations"]
                    if item["iteration"] == iteration
                    for e in item["p2p_calls"]
                ],
                key=lambda e: e["schedule_position"],
            )
        if diagnostic.get("timeline"):
            path = root / group / Path(metadata["profiler"]["path"]).name
            data = path.read_bytes()
            run["hashes"][str(path.resolve())] = hashlib.sha256(data).hexdigest()
            raw = json.loads(data)
            compact = compact_partition(raw, metadata, execution)
            evidence[(group, worker)] = overlap_evidence(raw, compact)
        for iteration, events in by_iteration.items():
            ordered = sequence[iteration]
            for event in events:
                index = next(
                    i
                    for i, e in enumerate(ordered)
                    if e["schedule_position"] == event["schedule_position"]
                )
                event["neighboring_pipeline_operations"] = ordered[max(0, index - 2) : index + 3]
                key = (
                    iteration,
                    event["logical_stage"],
                    event["microbatch"],
                    "F" if event["phase"] == "forward_compute" else "B",
                )
                event["overlap_evidence"] = (
                    evidence[(group, worker)][key] if diagnostic.get("timeline") else None
                )
                if event["overlap_evidence"] is not None:
                    e = event["overlap_evidence"]
                    e["cuda_event_minus_gpu_envelope_ms"] = (
                        event["compute_gpu_ms"]
                        - (e["gpu_envelope_end_us"] - e["gpu_envelope_start_us"]) / 1000
                    )
    return run


def compare(normal, serialized):
    if (
        normal["manifest"] != serialized["manifest"]
        or normal["metadata"].keys() != serialized["metadata"].keys()
    ):
        raise ValueError("Model/partition/rank mismatch")
    for key, first in normal["metadata"].items():
        second = serialized["metadata"][key]
        for field in ("measurement_iterations", "warmup_iterations", "stage_layer_ranges"):
            if first[field] != second[field]:
                raise ValueError(f"Incomparable {key}: {field}")
        a, b = (dict(item["calibration_configuration"]) for item in (first, second))
        da, db = a.pop("overlap_diagnostic", {}), b.pop("overlap_diagnostic", {})
        da.pop("execution", None)
        db.pop("execution", None)
        da.setdefault("timeline", False)
        db.setdefault("timeline", False)
        if a != b or da != db:
            raise ValueError(
                "Configuration/source/hardware/profiler differs beyond serialization policy"
            )
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    identities = {}
    for mode, run in (("normal", normal), ("serialized", serialized)):
        for event in run["events"]:
            identity = {
                k: event[k]
                for k in (
                    "logical_stage",
                    "stage_layer_range",
                    "layer_composition",
                    "stage_role",
                    "worker",
                    "phase",
                )
            }
            key = json.dumps(identity, sort_keys=True)
            identities[key] = identity
            for field in ("wall_ms", "compute_gpu_ms"):
                groups[(key, event["calibration_group_id"])][f"{mode}_{field}"][
                    event["iteration"]
                ].append(event[field])
    rows, comparisons = [], defaultdict(list)
    for (key, group), metrics in sorted(groups.items()):
        summaries = {}
        for name, values in metrics.items():
            means = [statistics.fmean(v) for _, v in sorted(values.items())]
            summaries[name] = dict(
                median_ms=statistics.median(means),
                iteration_means_ms=means,
                global_iterations=sorted(values),
                cv=statistics.pstdev(means) / statistics.fmean(means),
            )
        row = dict(**identities[key], partition=group, metrics=summaries)
        rows.append(row)
        comparisons[key].append(row)
    return dict(
        schema_version="slackpipe.overlap_diagnostic.v1",
        diagnostic_only=True,
        estimator_validated=False,
        samples_discarded=0,
        definitions=dict(
            wall_ms="normal: asynchronous host closure; serialized: host closure through device completion (pre-drain excluded); NOT equivalent latency boundaries",
            cuda_event_ms="compute-stream envelope including waits, launch gaps and contention, not intrinsic compute",
            overlap="clipped union of GPU activity inside correlated kernel envelope, NOT exact CUDA-event interval",
            wait="API calls are candidates; no claim of GPU blocking from host API duration",
            comparison="median of per-iteration raw microbatch means, no filtering; exact range/composition/role/worker/context only",
        ),
        inputs={**normal["hashes"], **serialized["hashes"]},
        groups=rows,
        cross_partition=[
            dict(
                **identities[key],
                partitions=[r["partition"] for r in members],
                median_shift={
                    name: (
                        max(r["metrics"][name]["median_ms"] for r in members)
                        / min(r["metrics"][name]["median_ms"] for r in members)
                        - 1
                        if len(members) > 1
                        else None
                    )
                    for name in members[0]["metrics"]
                },
            )
            for key, members in comparisons.items()
        ],
        normal_operations=normal["events"],
        serialized_operations=serialized["events"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal", type=Path, required=True)
    parser.add_argument("--serialized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        annotate_run(args.normal, "normal"), annotate_run(args.serialized, "serialized")
    )
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved {len(result['groups'])} comparisons; no estimator or root cause certified")


if __name__ == "__main__":
    main()
