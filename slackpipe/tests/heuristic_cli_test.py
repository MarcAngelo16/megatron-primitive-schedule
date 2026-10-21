"""Standalone heuristic contracts, deterministic trajectories and budget wiring."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

binary = sys.argv[1]
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    command = [
        binary,
        "--algorithm",
        "slack-heuristic",
        "--B",
        "4",
        "--N",
        "4",
        "--J",
        "2",
        "--L",
        "12",
        "--ratio-num",
        "3",
        "--ratio-den",
        "2",
        "--communication",
        "2",
        "--time-limit",
        "300",
        "--heuristic-fixed-evaluations",
        "true",
        "--heuristic-evaluations",
        "40000",
        "--heuristic-trace",
        "true",
    ]
    results = [json.loads(subprocess.check_output(command, text=True)) for _ in range(2)]
    for result in results:
        assert result["makespan_ticks"] == 129  # pre-refactor fixed-evaluation checkpoint
        assert result["method_id"] == "slack-heuristic"
        assert result["display_name"] == "Slack-based Heuristic"
        assert result["solution_source"] == "slack-heuristic"
        assert result["heuristic"]["method"] == "slack-heuristic"
        assert result["diagnostic"].startswith("Slack-based Heuristic:")
        assert result["heuristic"]["requested_seconds"] == 300
        assert result["heuristic"]["refinement_mode"] == "coupled"
        assert not result["cp_sat_launched"]
        assert result["solver_status_raw"] is None
    for key in ("split", "operations", "worker_local_operation_order"):
        assert results[0][key] == results[1][key]
    events = [
        [{k: v for k, v in e.items() if k != "seconds"} for e in r["heuristic"]["events"]]
        for r in results
    ]
    assert events[0] == events[1]
    for extra in (
        ["--num-workers", "2"],
        ["--enforce-activation-cap", "true", "--activation-cap-units", "20"],
    ):
        proc = subprocess.run(command + extra, capture_output=True)
        assert proc.returncode != 0
print("standalone heuristic CLI contracts passed")
