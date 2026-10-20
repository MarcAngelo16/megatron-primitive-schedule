"""Observable search semantics and replay/export checks, without OR-Tools."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

CLI = sys.argv[1]


def audit_trajectory(data: dict) -> None:
    """Check every outer parent transition, not just the final objective."""
    trace = data["heuristic"]
    assert not data["cp_sat_launched"]
    current = trace["initialization"]["makespan_ticks"]
    backward = sum(
        op["end"] for op in trace["initialization"]["plan"]["operations"] if op["kind"] == "B"
    )
    feasible = worse = strict = equal = 0
    for event in trace["events"]:
        if event["kind"] != "outer":
            continue
        assert event["before_ticks"] == current
        assert event["before_backward_sum"] == backward
        assert event["trial_refinement_evaluations"] <= (3072 if data["stages"] <= 16 else 1024)
        if event["perturbation_ticks"] >= 0:
            feasible += 1
            worse += event["perturbation_ticks"] > current
            assert event["refined"]
            assert event["child_ticks"] <= event["perturbation_ticks"]
        else:
            assert not event["refined"] and not event["accepted"]
        if event["accepted"]:
            assert event["child_ticks"] <= current
            if event["child_ticks"] < current:
                strict += 1
                assert event["reason"] == "makespan" and event["stale_after"] == 0
            else:
                equal += 1
                assert event["reason"] in ("backward_completion_sum", "partition_lex", "order_lex")
                assert event["child_backward_sum"] <= backward
                assert event["stale_after"] == event["stale_before"] + 1
            current, backward = event["child_ticks"], event["child_backward_sum"]
        else:
            if event["refined"]:
                assert event["child_ticks"] >= current
                if event["child_ticks"] == current:
                    assert event["child_backward_sum"] >= backward
            assert event["stale_after"] == event["stale_before"] + 1
    assert current == data["makespan_ticks"]
    assert backward == sum(op["end"] for op in data["operations"] if op["kind"] == "B")
    assert feasible == trace["feasible_outer_trials"] == trace["outer_refine_calls"]
    assert worse == trace["worse_feasible_outer_trials"]
    assert strict == trace["strict_outer_accepts"] and equal == trace["equal_outer_accepts"]
    assert trace["outer_pd_calls"] == 0
    assert trace["refinement_mode"] == "coupled"
    assert trace["evaluations"] <= trace["evaluation_limit"]
    points = data["incumbent_trace"]
    assert points[0]["objective"] == data["first_feasible_objective"]
    assert points[-1]["objective"] == current
    assert all(
        a["objective"] > b["objective"] and a["time_seconds"] <= b["time_seconds"]
        for a, b in zip(points, points[1:])
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        def run(name, seconds=60):
            prefix = root / name
            command = [
                CLI,
                "--algorithm",
                "heuristic",
                "--B",
                "8",
                "--N",
                "8",
                "--J",
                "4",
                "--L",
                "64",
                "--heuristic-evaluation-limit",
                "50000",
                "--heuristic-fixed-evaluations",
                "true",
                "--heuristic-refinement",
                "coupled",
                "--heuristic-time-limit-seconds",
                str(seconds),
                "--heuristic-trace",
                str(prefix) + ".trace.json",
                "--output-prefix",
                str(prefix),
                "--emit-plan",
                str(prefix) + ".plan.json",
            ]
            subprocess.run(command, check=True, capture_output=True)
            return json.loads(Path(str(prefix) + ".json").read_text())

        a, b = run("a"), run("b", seconds=0.000001)
        ta, tb = a["heuristic"], b["heuristic"]
        audit_trajectory(a)
        assert a["split"] == b["split"]
        assert a["operations"] == b["operations"]

        # Compare the complete event trajectory, excluding only actual wall time.
        def events(trace):
            return [{k: v for k, v in e.items() if k != "seconds"} for e in trace["events"]]

        assert events(ta) == events(tb)
        assert ta["evaluations"] == tb["evaluations"] == 50000
        assert ta["initialization_pd_calls"] > 0
        assert ta["outer_pd_calls"] == 0
        assert ta["worse_feasible_outer_trials"] > 0
        assert ta["equal_outer_accepts"] > 0
        for e in ta["events"]:
            if e["kind"] == "outer" and e["perturbation_ticks"] >= 0:
                assert e["refined"]
                assert e["trial_refinement_evaluations"] <= 3072
                if e["accepted"] and e["child_ticks"] == e["before_ticks"]:
                    assert e["stale_after"] == e["stale_before"] + 1
        corrupted = json.loads(json.dumps(a))
        next(e for e in corrupted["heuristic"]["events"] if e["kind"] == "outer")[
            "before_ticks"
        ] += 1
        try:
            audit_trajectory(corrupted)
        except AssertionError:
            pass
        else:
            raise AssertionError("trajectory audit accepted a hidden parent change")
        subprocess.run(
            [CLI, "validate-result", "--input", str(root / "a.json")],
            check=True,
            capture_output=True,
        )
        plan = json.loads((root / "a.plan.json").read_text())
        assert plan["layer_split"] == a["split"]
        assert json.loads((root / "a.trace.json").read_text()) == ta
        print(
            "Heuristic CLI: deterministic full trajectories, worse-trial refinement, "
            "coupled-only outer refinement, quota and independent export validation passed."
        )


if __name__ == "__main__":
    main()
