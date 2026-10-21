# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Read-only training-data diagnosis; never refit on the final holdout or export costs."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

from tools.slackpipe_stage_calibration import digest, statistics, write_new


def diagnose(root: Path) -> dict:
    manifest = json.loads((root / "FINAL_MANIFEST.json").read_text())
    failures = [r["path"] for r in manifest["files"] if digest(root / r["path"]) != r["sha256"]]
    if failures:
        raise ValueError(f"Historical manifest mismatch: {failures}")
    report = json.loads((root / "fit-final/fit_validation.json").read_text())
    profile = json.loads((root / "fit-final/cost_profile.experimental.json").read_text())
    runs = profile["independent_run_split"]["training"]
    rows = [r for r in report["rows"] if r["role"] == "train" and r["run_id"] in runs]
    fits = []
    for phase in ("forward", "backward"):
        for run in runs:
            group = [r for r in rows if r["phase"] == phase and r["run_id"] == run]
            x = np.array(
                [
                    [r["class_counts"][k] for k in sorted(r["class_counts"])]
                    + [int(r["stage_role"] == "first"), int(r["stage_role"] == "last")]
                    for r in group
                ]
            )
            y = np.array([r["observed_ms"] for r in group])
            coef, _ = nnls(x, y)
            fits.append(
                dict(
                    run=run,
                    phase=phase,
                    rows=len(y),
                    rank=int(np.linalg.matrix_rank(x)),
                    relative_rmse=float(np.sqrt(np.mean((x @ coef - y) ** 2)) / y.mean()),
                    observed_ms=y.tolist(),
                    fitted_ms=(x @ coef).tolist(),
                    coefficients_diagnostic_only=coef.tolist(),
                )
            )
    comparisons = []
    groups = defaultdict(list)
    for r in rows:
        groups[r["partition"], r["stage"], r["phase"]].append(r)
    for (partition, stage, phase), group in groups.items():
        a, b = sorted(group, key=lambda r: r["run_id"])
        comparisons.append(
            dict(
                partition=partition,
                stage=stage,
                phase=phase,
                worker=a["worker"],
                layer_range=a["layer_range"],
                class_counts=a["class_counts"],
                stage_role=a["stage_role"],
                run1_over_run0=b["observed_ms"] / a["observed_ms"],
                runs=[
                    dict(
                        run=r["run_id"],
                        statistics=r["statistics"],
                        raw=r["raw"],
                        iteration_means=r["iteration_means"],
                        pooled_relative_error=r["relative_error"],
                    )
                    for r in (a, b)
                ],
            )
        )
    raw_groups = defaultdict(list)
    for run in runs:
        raw = json.loads((root / run / "stage_samples.json").read_text())
        iterations = {}
        for part in {r["partition"] for r in rows if r["run_id"] == run}:
            metadata = [
                json.loads(p.read_text())
                for p in sorted((root / run / part).glob("result.rank*.json"))
            ]
            if not metadata or any(
                m["measurement_iterations"] != metadata[0]["measurement_iterations"]
                for m in metadata
            ):
                raise ValueError("Missing or misaligned measurement metadata")
            iterations[part] = metadata[0]["measurement_iterations"]
        for fitted in [r for r in rows if r["run_id"] == run]:
            selected = [
                r
                for r in raw
                if r["partition_id"] == fitted["partition"]
                and r["logical_stage"] == fitted["stage"]
                and r["phase"] == fitted["phase"] + "_compute"
                and r["iteration"] in iterations[fitted["partition"]]
            ]
            means = [
                np.mean([r["elapsed_ms"] for r in selected if r["iteration"] == i])
                for i in iterations[fitted["partition"]]
            ]
            if len(selected) != fitted["raw"]["count"] or not np.allclose(
                means, fitted["iteration_means"], rtol=1e-12, atol=1e-12
            ):
                raise ValueError("Raw samples do not reproduce fitted iteration means")
            if not np.isclose(np.median(means), fitted["observed_ms"], rtol=1e-12, atol=1e-12):
                raise ValueError("Raw samples do not reproduce stage estimate")
        for row in raw:
            if (
                row["partition_id"] not in iterations
                or row["iteration"] not in iterations[row["partition_id"]]
            ):
                continue
            raw_groups[
                run,
                row["partition_id"],
                row["logical_stage"],
                row["phase"],
                row["pipeline_position"],
            ].append(row)
    positions = []
    for key, group in sorted(raw_groups.items()):
        elapsed = np.array([r["elapsed_ms"] for r in group])
        cpu = np.array([r["cpu_dispatch_ms"] for r in group])
        positions.append(
            dict(
                identity=list(key),
                elapsed=statistics(elapsed),
                cpu_dispatch=statistics(cpu),
                median_cpu_over_event=float(np.median(cpu / elapsed)),
            )
        )
    grouped = {}
    for field in ("run_id", "stage_role", "worker", "length", "mamba_count", "composition"):
        g = defaultdict(list)
        for row in rows:
            key = {
                "length": row["layer_range"][1] - row["layer_range"][0],
                "mamba_count": row["class_counts"].get("mamba", 0),
                "composition": json.dumps(row["class_counts"], sort_keys=True),
            }.get(field, row.get(field))
            g[row["phase"], str(key)].append(row)
        grouped[field] = [
            dict(
                phase=k[0],
                value=k[1],
                stage_run_rows=len(v),
                signed_mean_error=float(np.mean([r["relative_error"] for r in v])),
                mape=float(np.mean([abs(r["relative_error"]) for r in v])),
            )
            for k, v in sorted(g.items())
        ]
    comparable = defaultdict(list)
    for r in rows:
        comparable[
            r["run_id"], tuple(r["layer_range"]), r["stage_role"], r["worker"], r["phase"]
        ].append(r)
    exact_range_context = [
        dict(
            identity=list(key),
            max_over_min=max(r["observed_ms"] for r in group)
            / min(r["observed_ms"] for r in group),
            rows=group,
        )
        for key, group in sorted(comparable.items())
        if len(group) > 1
    ]
    return dict(
        schema_version="slackpipe.residual_diagnosis.v1",
        manifest_files_verified=len(manifest["files"]),
        raw_stage_run_phase_groups_recomputed=len(rows),
        source_manifest_sha256=digest(root / "FINAL_MANIFEST.json"),
        scope="Training partitions of training runs only; final holdout untouched. Microbatches are not independent runs.",
        statistic="Population stddev/CV; stage estimate=median of 30 iteration means of four microbatches; all samples retained.",
        per_run_fits=fits,
        exact_range_cross_partition=exact_range_context,
        same_partition_comparisons=comparisons,
        pipeline_position=positions,
        grouped_pooled_residuals=grouped,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    write_new(args.output, diagnose(args.source))


if __name__ == "__main__":
    main()
