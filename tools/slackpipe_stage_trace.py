# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Correlate diagnostic Node/module labels to GPU launches, retaining ambiguity."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from megatron.core.pipeline_parallel.slackpipe.figure_trace import activity_metrics
from tools.slackpipe_stage_calibration import digest, write_new


def correlate(raw):
    events = raw["traceEvents"]
    labels = []
    runtime = defaultdict(list)
    for event in events:
        if event.get("ph") != "X":
            continue
        name = event.get("name", "")
        if event.get("cat") == "user_annotation" and name.startswith(
            ("SlackPipeLayer/F/", "SlackPipeLayer/B/")
        ):
            fields = json.loads(bytes.fromhex(name.split("/", 2)[2]))
            labels.append((event, fields))
        if event.get("cat") in ("cuda_runtime", "cuda_driver"):
            runtime[event.get("args", {}).get("correlation")].append(event)
    if not labels:
        raise ValueError("No structured native layer/Node labels")
    result = []
    for index, event in enumerate(events):
        if event.get("ph") != "X" or event.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        correlation = event.get("args", {}).get("correlation")
        launches = runtime.get(correlation, []) if correlation is not None else []
        owners = []
        for launch in launches:
            for label, fields in labels:
                if (
                    launch.get("tid") == label.get("tid")
                    and launch.get("pid") == label.get("pid")
                    and label["ts"] <= launch["ts"] < label["ts"] + label["dur"]
                ):
                    if fields not in owners:
                        owners.append(fields)
        communication = "nccl" in event.get("name", "").lower()
        unique = (
            len(owners) == 1
            and owners[0].get("attribution", "unique") == "unique"
            and not communication
        )
        result.append(
            dict(
                raw_event_index=index,
                event=event,
                owners=owners,
                attribution="unique" if unique else "shared/unattributed",
                communication=communication,
            )
        )
    groups = defaultdict(list)
    for r in result:
        owner = r["owners"][0] if r["attribution"] == "unique" else {}
        key = (
            owner.get("iteration"),
            owner.get("microbatch"),
            owner.get("stage"),
            owner.get("global_layer_id"),
            owner.get("layer_type"),
            owner.get("phase"),
            r["communication"],
            r["attribution"],
        )
        groups[key].append(r["event"])
    # These are GPU activity metrics, never substituted for event-envelope costs.
    metrics = [dict(identity=list(k), **activity_metrics(v)) for k, v in groups.items()]
    cpu = [
        e
        for e in events
        if e.get("ph") == "X"
        and e.get("cat") in ("cuda_runtime", "cuda_driver")
        and any(n in e.get("name", "") for n in ("Malloc", "MemAlloc", "StreamWaitEvent"))
    ]
    return dict(
        schema_version="slackpipe.layer_attribution.v1",
        activities=result,
        metrics=metrics,
        cpu_allocations_and_wait_insertions=cpu,
        baseTimeNanoseconds=raw.get("baseTimeNanoseconds"),
        cross_rank_clock_alignment="unverified; no cross-device event origin comparison",
        dependency_wait_gpu_ms=None,
        definition="Native layer/Node identity -> same-thread CUDA launch correlation -> GPU activity; ambiguous or unjoined work retained. CPU wait API duration is not GPU blocking duration.",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = correlate(json.loads(args.trace.read_text()))
    result["source"] = dict(path=str(args.trace), sha256=digest(args.trace))
    write_new(args.output, result)


if __name__ == "__main__":
    main()
