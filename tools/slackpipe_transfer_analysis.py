# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Predeclared transfer analysis. All samples retained, no trusted profile output."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from tools.slackpipe_stage_calibration import digest, statistics, write_new


def training(row):
    return row["group"] == 2 and row["partition"] in ("P1", "P2", "P3") and row["run"] in (0, 1)


def subset(row):
    if row["partition"] == "P0":
        value = "uniform_control"
    elif row["partition"] in ("P1", "P2", "P3"):
        value = "independent_run" if row["run"] == 2 else "train"
    else:
        value = "partition_and_run" if row["run"] == 2 else "unseen_partition"
    return ("order_holdout/" if row["group"] == 4 else "native/") + value


def fit_ols(rows):
    if not rows or not all(training(r) for r in rows):
        raise ValueError("Refusing heldout observations in fitting")
    x = np.asarray(
        [[r["composition_ms"], 1, int(r["stage"] == 0), int(r["stage"] == 3)] for r in rows]
    )
    y = np.asarray([r["mean_ms"] for r in rows])
    rank = int(np.linalg.matrix_rank(x))
    if rank != 4:
        raise ValueError(f"Unidentifiable model: {rank}/4")
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = x @ beta - y
    covariance = np.linalg.inv(x.T @ x) * np.sum(residual**2) / (len(y) - 4)
    per_run = {}
    for run in (0, 1):
        keep = [i for i, r in enumerate(rows) if r["run"] == run]
        if keep:
            per_run[str(run)] = dict(
                rank=int(np.linalg.matrix_rank(x[keep])),
                coefficients=np.linalg.lstsq(x[keep], y[keep], rcond=None)[0].tolist(),
            )
    return dict(
        coefficients=beta.tolist(),
        rank=rank,
        columns=4,
        condition=float(np.linalg.cond(x)),
        samples=len(rows),
        iid_standard_errors=np.sqrt(np.diag(covariance)).tolist(),
        uncertainty_warning="Stage rows correlated within run; iid SE descriptive, NOT cluster-valid CI; only two independent training runs",
        coefficients_by_training_run=per_run,
        relative_rmse=float(np.sqrt(np.mean(residual**2)) / np.mean(y)),
    )


def error_metrics(rows):
    errors = np.array([r["relative_error"] for r in rows])
    return dict(
        count=len(rows),
        mape=float(np.abs(errors).mean()),
        signed_bias=float(errors.mean()),
        maximum_absolute_relative_error=float(np.abs(errors).max()),
        target_passed=bool(np.abs(errors).mean() <= 0.05),
    )


def paired_interval(values):
    x = np.asarray(values, dtype=float)
    if len(x) != 3:
        raise ValueError("Three independent paired runs required")
    half = 4.302652729911275 * x.std(ddof=1) / np.sqrt(3)
    return dict(
        values=x.tolist(),
        mean=float(x.mean()),
        ci95=[float(x.mean() - half), float(x.mean() + half)],
        method="paired Student t, df=2; exploratory with n=3; not iid steps",
        within_one_percent=bool(x.mean() - half >= -0.01 and x.mean() + half <= 0.01),
    )


def validate_operations(rows, order):
    for iteration in range(60):
        selected = [r for r in rows if r["iteration"] == iteration]
        actual = [
            dict(
                kind="F" if r["phase"] == "forward_compute" else "B",
                stage=r["logical_stage"],
                microbatch=r["microbatch"],
            )
            for r in selected
        ]
        if actual != order or any(r["status"] != "recorded" for r in selected):
            raise ValueError("Missing, duplicate, reordered or failed operations")
    if len(rows) != 60 * len(order):
        raise ValueError("Unexpected iteration records")


def analyze(root):
    frozen = json.loads((root / "FROZEN.json").read_text())
    eligible = json.loads((root / "MEASUREMENT_ADMISSION.json").read_text())["eligible_groups"]
    hashes = json.loads((root / "FREEZE_HASHES.json").read_text())
    for name, expected in hashes.items():
        if digest(root / name) != expected:
            raise ValueError("Modified frozen input " + name)
    cuts = {
        p["id"]: p["cuts"] for p in json.loads((root / "design.json").read_text())["partitions"]
    }
    layer_class = {i: c["class_signature"] for c in frozen["classes"] for i in c["member_layers"]}
    composition, composition_stats = {}, []
    for case in frozen["composition_cases"]:
        for phase in ("F", "B"):
            run_means = []
            for run in range(3):
                path = root / "composition" / f"run{run}" / f"{case['id']}.json"
                data = json.loads(path.read_text())
                if data["case"] != case or data.get("finite_parameters") is not True:
                    raise ValueError("Failed or mismatched composition case")
                samples = [r for r in data["samples"] if r["phase"] == phase and r["measured"]]
                if [r["iteration"] for r in samples] != list(range(30, 60)):
                    raise ValueError("Incomplete composition samples")
                stats = statistics([r["event_ms"] for r in samples])
                composition_stats.append(
                    dict(
                        case=case,
                        phase=phase,
                        run=run,
                        device=data["device"],
                        statistics=stats,
                        raw=str(path),
                    )
                )
                if run in (0, 1):
                    run_means.append(stats["mean"])
            composition[tuple(case["signature_counts"]), phase] = float(np.mean(run_means))
    stages, windows, position_stats = [], [], []
    for run in range(3):
        for group in eligible:
            for mode in ("off", "pipeline_sync", "pipeline_events"):
                folder = root / "pipeline" / f"run{run}.g{group}.{mode}"
                for partition in cuts:
                    meta = [
                        json.loads((folder / partition / f"result.rank{rank}.json").read_text())
                        for rank in range(2)
                    ]
                    windows.append(
                        dict(
                            run=run,
                            group=group,
                            mode=mode,
                            partition=partition,
                            window_ms_by_rank=[m["continuous_wall_ms"] for m in meta],
                            step_ms=max(m["continuous_wall_ms"] for m in meta) / 30,
                            host_dispatch_by_rank=[
                                statistics([s["host_dispatch_ms"] for s in m["samples"]])
                                for m in meta
                            ],
                            warmup_tail_by_rank=[
                                statistics(
                                    [s["host_dispatch_ms"] for s in m["warmup_samples"][-10:]]
                                )
                                for m in meta
                            ],
                            measured_thirds_by_rank=[
                                [
                                    statistics(
                                        [s["host_dispatch_ms"] for s in m["samples"][i : i + 10]]
                                    )
                                    for i in (0, 10, 20)
                                ]
                                for m in meta
                            ],
                            peak_memory_by_rank=[m["memory_end"] for m in meta],
                        )
                    )
                    if mode == "off":
                        continue
                    for rank in range(2):
                        path = folder / partition / f"stage_samples.rank{rank}.json"
                        rows = json.loads(path.read_text())
                        if any(
                            r["worker"] != rank
                            or r["partition_id"] != partition
                            or r["run_id"] != f"run{run}"
                            or r["stage_layer_range"]
                            != cuts[partition][r["logical_stage"] : r["logical_stage"] + 2]
                            for r in rows
                        ):
                            raise ValueError("Raw operation provenance/range mismatch")
                        order = frozen["orders"][str(group)][rank]
                        validate_operations(rows, order)
                        first_b = next(i for i, op in enumerate(order) if op["kind"] == "B")
                        last_f = max(i for i, op in enumerate(order) if op["kind"] == "F")
                        for stage in range(rank, 4, 2):
                            a, b = cuts[partition][stage : stage + 2]
                            count = Counter(layer_class[i] for i in range(a, b))
                            key = tuple(count[s] for s in frozen["signatures"])
                            for phase in ("F", "B"):
                                selected = [
                                    r
                                    for r in rows
                                    if r["iteration"] >= 30
                                    and r["logical_stage"] == stage
                                    and r["phase"]
                                    == ("forward_compute" if phase == "F" else "backward_compute")
                                ]
                                stats = statistics([r["elapsed_ms"] for r in selected])
                                record = dict(
                                    run=run,
                                    group=group,
                                    mode=mode,
                                    partition=partition,
                                    stage=stage,
                                    worker=rank,
                                    phase=phase,
                                    range=[a, b],
                                    length=b - a,
                                    composition=list(key),
                                    composition_ms=composition[key, phase],
                                    mean_ms=stats["mean"],
                                    statistics=stats,
                                    raw=str(path),
                                    role=(
                                        "first"
                                        if stage == 0
                                        else "last" if stage == 3 else "interior"
                                    ),
                                )
                                if mode == "pipeline_sync":
                                    record["synchronized_wall_statistics"] = statistics(
                                        [r["synchronized_wall_ms"] for r in selected]
                                    )
                                stages.append(record)
                                for position in ("warmup", "steady", "cooldown"):

                                    def pos(r):
                                        p = r["schedule_position"]
                                        return (
                                            "warmup"
                                            if p < first_b
                                            else "cooldown" if p > last_f else "steady"
                                        )

                                    values = [
                                        r["elapsed_ms"] for r in selected if pos(r) == position
                                    ]
                                    if values:
                                        position_stats.append(
                                            dict(
                                                record,
                                                pipeline_position=position,
                                                statistics=statistics(values),
                                            )
                                        )
    fits, predictions = {}, []
    for phase in ("F", "B"):
        train = [r for r in stages if training(r) and r["phase"] == phase]
        event_train = [r for r in train if r["mode"] == "pipeline_events"]
        boundary = [
            float(
                np.mean(
                    [r["mean_ms"] - r["composition_ms"] for r in event_train if r["stage"] == s]
                )
            )
            for s in (0, 3)
        ]
        fits["M0/" + phase] = dict(coefficients=[1, 0] + boundary, warning=frozen["boundary"])
        for name, mode in (("M1", "pipeline_sync"), ("M2", "pipeline_events")):
            fits[name + "/" + phase] = fit_ols([r for r in train if r["mode"] == mode])
        for row in [r for r in stages if r["phase"] == phase]:
            x = np.array([row["composition_ms"], 1, int(row["stage"] == 0), int(row["stage"] == 3)])
            for name in ("M0", "M1", "M2"):
                prediction = float(x @ fits[name + "/" + phase]["coefficients"])
                predictions.append(
                    dict(
                        row,
                        model=name,
                        subset=subset(row),
                        prediction_ms=prediction,
                        relative_error=prediction / row["mean_ms"] - 1,
                    )
                )
    grouped = defaultdict(list)
    for r in predictions:
        grouped[r["model"], r["mode"], r["subset"], r["phase"]].append(r)
    metrics = [
        dict(model=k[0], reference=k[1], subset=k[2], phase=k[3], **error_metrics(v))
        for k, v in sorted(grouped.items())
    ]
    residual_groups = []
    for field in ("length", "role", "worker", "partition", "run", "group", "composition"):
        groups = defaultdict(list)
        for r in predictions:
            groups[r["model"], r["mode"], r["phase"], r["subset"], str(r[field])].append(r)
        residual_groups.extend(
            dict(
                field=field,
                model=k[0],
                reference=k[1],
                phase=k[2],
                subset=k[3],
                value=k[4],
                **error_metrics(v),
            )
            for k, v in sorted(groups.items())
        )
    perturbations = []
    for group in eligible:
        for partition in cuts:
            values = []
            details = []
            for run in range(3):
                pair = {
                    r["mode"]: r
                    for r in windows
                    if r["group"] == group and r["partition"] == partition and r["run"] == run
                }
                values.append(pair["pipeline_events"]["step_ms"] / pair["off"]["step_ms"] - 1)
                details.append(
                    dict(
                        run=run,
                        off_ms=pair["off"]["step_ms"],
                        events_ms=pair["pipeline_events"]["step_ms"],
                    )
                )
            perturbations.append(
                dict(group=group, partition=partition, runs=details, **paired_interval(values))
            )
    position_predictions = []
    for row in position_stats:
        x = np.array([row["composition_ms"], 1, int(row["stage"] == 0), int(row["stage"] == 3)])
        for name in ("M0", "M1", "M2"):
            prediction = float(x @ fits[name + "/" + row["phase"]]["coefficients"])
            position_predictions.append(
                dict(
                    row,
                    model=name,
                    prediction_ms=prediction,
                    relative_error=prediction / row["statistics"]["mean"] - 1,
                    interpretation="Subgroup diagnostic using unchanged predeclared model, not refit",
                )
            )
    run_dispersion = []
    for group in eligible:
        for mode in ("off", "pipeline_sync", "pipeline_events"):
            for partition in cuts:
                selected = [
                    r
                    for r in windows
                    if r["group"] == group and r["mode"] == mode and r["partition"] == partition
                ]
                run_dispersion.append(
                    dict(
                        group=group,
                        mode=mode,
                        partition=partition,
                        independent_run_step_means_ms=[r["step_ms"] for r in selected],
                        statistics=statistics([r["step_ms"] for r in selected]),
                    )
                )
    result = dict(
        schema_version="slackpipe.transfer_analysis.v1",
        analysis_source_sha256=digest(Path(__file__)),
        status="diagnostic_not_trusted_profile",
        units="milliseconds",
        sample_policy="all raw samples, arithmetic mean; steps/microbatches not independent runs",
        fits=fits,
        metrics=metrics,
        composition_statistics=composition_stats,
        stages=stages,
        predictions=predictions,
        residual_groups=residual_groups,
        pipeline_position_statistics=position_stats,
        pipeline_position_predictions=position_predictions,
        windows=windows,
        independent_run_dispersion=run_dispersion,
        perturbation=perturbations,
        same_first_range=[
            r for r in predictions if r["stage"] == 0 and r["partition"] in ("P1", "P3", "P5")
        ],
        interpretation="GPU-event reference not ground truth; M1 matching sync != normal execution prediction; no measured cross-GPU makespan; M0 boundary residual not pure boundary measurement",
    )
    write_new(root / "analysis.json", result)
    write_new(
        root / "FINAL_MANIFEST.json",
        {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*")) if p.is_file()},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    analyze(parser.parse_args().input)
