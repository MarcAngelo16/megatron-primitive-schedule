# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Correlate composition-control launches; never use kernel sums as cost inputs."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from megatron.core.pipeline_parallel.slackpipe.figure_trace import activity_metrics
from tools.slackpipe_stage_calibration import digest, write_new


def analyze(raw: dict) -> dict:
    labels = defaultdict(list)
    runtime = defaultdict(list)
    events = raw["traceEvents"]
    for event in events:
        if event.get("ph") != "X":
            continue
        name = event.get("name", "")
        if event.get("cat") == "user_annotation":
            parts = name.split("/")
            # F is an explicit call on the dispatch thread. B attribution must
            # come from hooks on executed autograd nodes, not the outer B range.
            if (name.startswith("Composition/") and parts[-1] == "F") or name.startswith(
                "CompositionNode/"
            ):
                identity = tuple(parts[1:4])
                labels[event.get("pid"), event.get("tid")].append((event, identity))
        if event.get("cat") in ("cuda_runtime", "cuda_driver"):
            correlation = event.get("args", {}).get("correlation")
            if correlation is not None:
                runtime[correlation].append(event)
    groups = defaultdict(list)
    unassigned = []
    assignments = []
    for index, event in enumerate(events):
        if event.get("ph") != "X" or event.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        owners = set()
        for launch in runtime.get(event.get("args", {}).get("correlation"), []):
            for label, identity in labels[launch.get("pid"), launch.get("tid")]:
                if label["ts"] <= launch["ts"] < label["ts"] + label["dur"]:
                    owners.add(identity)
        if len(owners) == 1:
            owner = next(iter(owners))
            groups[owner].append(event)
            assignments.append(dict(raw_index=index, identity=list(owner)))
        else:
            unassigned.append(dict(raw_index=index, owners=sorted(owners)))
    rows = []
    for identity, group in sorted(groups.items()):
        kernels = defaultdict(list)
        for event in group:
            if event["cat"] == "kernel":
                kernels[event["name"]].append(event["dur"])
        rows.append(
            dict(
                identity=list(identity),
                **activity_metrics(group),
                streams=sorted({str(e.get("args", {}).get("stream")) for e in group}),
                kernels={k: dict(count=len(v), ordered_duration_us=v) for k, v in kernels.items()},
            )
        )
    return dict(
        schema_version="slackpipe.composition_trace_diagnosis.v1",
        rows=rows,
        assignments=assignments,
        shared_unattributed=unassigned,
        cpu_allocation_wait_calls=[
            e
            for e in events
            if e.get("cat") in ("cuda_runtime", "cuda_driver")
            and any(k in e.get("name", "") for k in ("Malloc", "MemAlloc", "StreamWaitEvent"))
        ],
        nccl_activity_count=sum(
            "nccl" in e.get("name", "").lower() for e in events if e.get("cat") == "kernel"
        ),
        baseTimeNanoseconds=raw.get("baseTimeNanoseconds"),
        caveat="Separate instrumented execution, not cost fitting. B uses actual Node launch provenance. Internal gaps are not asserted hardware idle. CPU wait insertion is not GPU blocked duration. No cross-device clock comparison.",
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = analyze(json.loads(args.trace.read_text()))
    result["source"] = dict(path=str(args.trace), sha256=digest(args.trace))
    write_new(args.output, result)


if __name__ == "__main__":
    main()
