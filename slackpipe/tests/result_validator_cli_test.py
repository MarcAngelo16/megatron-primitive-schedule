"""Independent result replay, exports and unavailable/no-solution failure records."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

binary = sys.argv[1]
info = json.loads(subprocess.check_output([binary, "--build-info"]))
display_names = {
    "1f1b": "1F1B",
    "interleaved-1f1b": "Interleaved 1F1B",
    "octopipe-tuning": "OctoPipe-Tuning",
    "cpsat-no-hint": "CP-SAT without hints",
    "slack-heuristic": "Slack-based Heuristic",
}
for method, display_name in display_names.items():
    description = json.loads(subprocess.check_output([binary, "--describe-method", method]))
    assert description["method_id"] == method
    assert description["display_name"] == display_name
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    methods = ["slack-heuristic", "octopipe-tuning", "cpsat-no-hint"]
    for method in methods:
        prefix, plan = root / method, root / f"{method}.plan.json"
        command = [
            binary,
            "--algorithm",
            method,
            "--B",
            "2",
            "--N",
            "2",
            "--J",
            "2",
            "--L",
            "4",
            "--time-limit",
            "1",
            "--output-prefix",
            str(prefix),
            "--emit-plan",
            str(plan),
        ]
        process = subprocess.run(command, capture_output=True, text=True)
        result = json.loads(prefix.with_suffix(".json").read_text())
        assert result["method_id"] == method
        assert result["display_name"] == display_names[method]
        if method == "cpsat-no-hint" and not info["ortools_enabled"]:
            assert process.returncode == 3
            assert result["status"] == "UNAVAILABLE"
            assert not plan.exists() and result["makespan_ticks"] is None
            continue
        assert process.returncode == 0, process.stderr
        assert result["final_solution_available"]
        subprocess.run(
            [binary, "--validate-result", str(prefix.with_suffix(".json"))],
            check=True,
            capture_output=True,
        )
        exported = json.loads(plan.read_text())
        operations = [op for order in exported["operations"] for op in order]
        assert len(operations) == 8
        assert len({(op["kind"], op["microbatch"], op["stage"]) for op in operations}) == 8
        # Independent validator must reject tampered timings, even if embedded verdict is "passed".
        result["operations"][0]["duration"] += 1
        bad = root / "bad.json"
        bad.write_text(json.dumps(result))
        assert (
            subprocess.run([binary, "--validate-result", str(bad)], capture_output=True).returncode
            != 0
        )
    if info["ortools_enabled"]:
        plan = root / "infeasible.plan.json"
        result = json.loads(
            subprocess.check_output(
                [
                    binary,
                    "--algorithm",
                    "cpsat-no-hint",
                    "--time-limit",
                    "10",
                    "--enforce-activation-cap",
                    "true",
                    "--activation-cap-units",
                    "0",
                    "--emit-plan",
                    str(plan),
                ]
            )
        )
        assert not result["final_solution_available"] and not plan.exists()
        assert result["display_name"] == display_names["cpsat-no-hint"]
        assert result["external_hint_count"] == 0 and not result["fallback_used"]
        assert result["solver_status_raw"] == "INFEASIBLE"
        assert result["cp_sat_launched"] and result["solution_source"] == "none"
    # Neither accepted optimizer flags nor historical plan paths may inject a seed.
    for option in (
        "--hint",
        "--initial-plan",
        "--checkpoint",
        "--incumbent",
        "--heuristic-evaluations",
        "--octopipe-max-iterations",
    ):
        process = subprocess.run(
            [binary, "--algorithm", "cpsat-no-hint", option, "1"], capture_output=True
        )
        assert process.returncode == 2
    existing = root / "existing.plan.json"
    existing.write_text('{"historical": true}')
    process = subprocess.run(
        [binary, "--algorithm", "cpsat-no-hint", "--emit-plan", str(existing)], capture_output=True
    )
    assert process.returncode == 2
    assert existing.read_text() == '{"historical": true}'
    for method in ("uniform-breadth-first", "heuristic", "slackpipe", "joint", "interleaved-1f1b"):
        assert subprocess.run([binary, "--algorithm", method], capture_output=True).returncode != 0
print("independent validation/export/no-solution CLI contracts passed")
