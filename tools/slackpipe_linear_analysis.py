# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Explicit numerical admission and SVD identification, never a cost exporter."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from tools.slackpipe_linear_identification import TYPES, counts, matrix_report
from tools.slackpipe_linear_worker import tensor_difference
from tools.slackpipe_stage_calibration import digest, statistics, write_new
from tools.slackpipe_transfer_analysis import error_metrics, paired_interval, validate_operations


def compare_dict(a, b):
    if a.keys() != b.keys():
        raise ValueError("Different logical parameter identities")
    errors = {k: tensor_difference(v, b[k]) for k, v in a.items()}
    return dict(
        equal=all(e["equal"] for e in errors.values()),
        max_abs=max(e["max_abs"] for e in errors.values()),
        max_relative=max(e["max_relative"] for e in errors.values()),
        worst_parameter=max(errors, key=lambda k: errors[k]["max_abs"]),
    )


def audit(root, folder="audit", filename="numerical_audit.json"):
    design = json.loads((root / "design.json").read_text())
    parts = {}
    for p in design["partitions"]:
        raw = [
            torch.load(
                root / f"{folder}/run0.off/{p['id']}/layers-audit.rank{r}.pt", weights_only=True
            )
            for r in (0, 1)
        ]
        parts[p["id"]] = dict(
            initial={k: v for d in raw for k, v in d["initial"].items()},
            layers={k: v for d in raw for k, v in d["layers"].items()},
            batches={k: v for d in raw for k, v in d["batches"].items()},
            steps={
                str(i): {
                    field: {k: v for d in raw for k, v in d["steps"][str(i)][field].items()}
                    for field in ("before", "gradients", "after")
                }
                for i in (0, 29, 30)
            },
        )
    ref = parts["U0"]
    results = []
    for name, p in parts.items():
        row = dict(
            partition=name,
            initial=compare_dict(p["initial"], ref["initial"]),
            steps={},
            layer_errors=[],
        )
        batch_errors = []
        for key, batch in p["batches"].items():
            iteration, _, mb = key.split("/")
            batch_errors.append(compare_dict(batch, ref["batches"][f"{iteration}/s0/{mb}"]))
        row["batches_equal"] = all(e["equal"] for e in batch_errors)
        for i in (0, 29, 30):
            row["steps"][str(i)] = {
                field: compare_dict(p["steps"][str(i)][field], ref["steps"][str(i)][field])
                for field in ("before", "gradients", "after")
            }
            if i == 0:
                row["parameter_mutation_during_first_FB"] = compare_dict(
                    p["steps"]["0"]["before"], p["initial"]
                )
            first = None
            for layer in range(12):
                for mb in range(4):
                    key = f"i{i}/l{layer}/b{mb}"
                    actual = p["layers"][key]
                    expected = ref["layers"][key]
                    errors = {
                        field: tensor_difference(actual[field], expected[field])
                        for field in ("input", "output", "input_grad", "output_grad")
                    }
                    row["layer_errors"].append(
                        dict(
                            iteration=i, layer=layer, microbatch=mb, stage=actual["stage"], **errors
                        )
                    )
                    if first is None and not errors["output"]["equal"]:
                        first = dict(
                            layer=layer,
                            microbatch=mb,
                            errors=errors,
                            input_nearest_reference_microbatch=[
                                tensor_difference(
                                    actual["input"], ref["layers"][f"i{i}/l{layer}/b{b}"]["input"]
                                )
                                for b in range(4)
                            ],
                        )
            row["steps"][str(i)]["first_forward_divergence"] = first
        row["equivalent"] = (
            row["initial"]["equal"]
            and row["batches_equal"]
            and all(
                all(step[field]["equal"] for field in ("before", "gradients", "after"))
                for step in row["steps"].values()
            )
            and all(
                all(e[field]["equal"] for field in ("input", "output", "input_grad", "output_grad"))
                for e in row["layer_errors"]
            )
        )
        results.append(row)
    write_new(
        root / filename,
        dict(
            atol=0,
            rtol=0,
            relative_denominator_floor=1e-12,
            reference="U0",
            pipeline_equivalent=all(r["equivalent"] for r in results),
            results=results,
            policy="If false, pipeline observations are diagnostic only, excluded from equivalent-training prediction claims; continue fixed-state controlled bodies",
        ),
    )
    for r in results:
        print(
            r["partition"],
            "equal",
            r["equivalent"],
            "first",
            r["steps"]["0"]["first_forward_divergence"],
        )


def training(row):
    return row["design_role"] == "train" and row["run"] in (0, 1)


def boundary_diagnosis(root, folder, filename):
    design = json.loads((root / "design.json").read_text())
    records, losses = [], []
    reference = None
    # Start with U0 only to define the comparison, never to fit a cost model.
    parts = sorted(design["partitions"], key=lambda p: p["id"] != "U0")
    for p in parts:
        raw = [
            torch.load(
                root / f"{folder}/run0.off/{p['id']}/layers-audit.rank{r}.pt", weights_only=True
            )
            for r in (0, 1)
        ]
        layers = {k: v for d in raw for k, v in d["layers"].items()}
        if reference is None:
            reference = raw[1]["steps"]
        for iteration in (0, 29, 30):
            actual_loss = raw[1]["steps"][str(iteration)]["losses"]
            expected_loss = reference[str(iteration)]["losses"]
            losses.append(
                dict(
                    partition=p["id"],
                    iteration=iteration,
                    errors=[compare_dict(a, b) for a, b in zip(actual_loss, expected_loss)],
                    actual=[{k: float(v) for k, v in d.items()} for d in actual_loss],
                    expected=[{k: float(v) for k, v in d.items()} for d in expected_loss],
                )
            )
            for boundary in p["cuts"][1:-1]:
                for mb in range(4):
                    x = layers[f"i{iteration}/l{boundary}/b{mb}"]["input"]
                    expected = layers[f"i{iteration}/l{boundary-1}/b{mb}"]["output"]
                    difference = tensor_difference(x, expected)
                    matches = []
                    if not difference["equal"]:
                        for key, row in layers.items():
                            if not key.startswith(f"i{iteration}/"):
                                continue
                            for field in ("output", "input_grad", "output_grad"):
                                if torch.equal(x, row[field]):
                                    matches.append(dict(identity=key, field=field))
                    records.append(
                        dict(
                            partition=p["id"],
                            iteration=iteration,
                            boundary=boundary,
                            microbatch=mb,
                            expected_predecessor=difference,
                            exact_other_tensor_matches=matches,
                        )
                    )
    write_new(
        root / filename,
        dict(
            audit_folder=folder,
            records=records,
            losses=losses,
            failed_boundary_count=sum(not r["expected_predecessor"]["equal"] for r in records),
        ),
    )
    print(
        "Boundary mismatches:",
        sum(not r["expected_predecessor"]["equal"] for r in records),
        "/",
        len(records),
    )


def subset(row):
    if row["design_role"] == "check":
        return "uniform_control"
    if row["design_role"] == "train":
        return "independent_run" if row["run"] == 2 else "train"
    return "unseen_case_and_run" if row["run"] == 2 else "unseen_case"


def features(row, unit, model):
    x = np.asarray(row["counts"]) * np.asarray(unit)
    if model == "H1":
        return [float(x.sum()), 1.0]
    value = x.tolist() + [1.0]
    if model == "H2-boundary":
        value += [int(row["role"] == "first"), int(row["role"] == "last")]
    if model == "H2-entry":
        value += [row["q"] - 1]
    return value


def solve(rows, unit, model):
    if not rows or not all(training(r) for r in rows):
        raise ValueError("Heldout data cannot enter identification")
    x = np.asarray([features(r, unit, model) for r in rows])
    y = np.asarray([r["statistics"]["mean"] for r in rows])
    report = matrix_report(x)
    if report["rank"] != report["unknowns"] or len(x) <= x.shape[1]:
        return dict(design=report, status="unidentifiable or not overdetermined")
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = x @ beta - y
    _, sv, vt = np.linalg.svd(x, full_matrices=False)
    covariance = (vt.T * (1 / sv**2)) @ vt * (residual @ residual / (len(x) - x.shape[1]))
    theta = (beta[0] * np.asarray(unit)).tolist() if model == "H1" else (beta[:3] * unit).tolist()
    return dict(
        status="identified",
        design=report,
        coefficients=beta.tolist(),
        effective_theta=theta,
        count_parameterization=(
            theta + beta[1:].tolist() if model == "H1" else theta + beta[3:].tolist()
        ),
        iid_standard_errors=np.sqrt(np.maximum(0, np.diag(covariance))).tolist(),
        uncertainty="IID errors descriptive only: observations share runs/devices; two training runs cannot establish reliable run-population confidence intervals",
        training_rmse_ms=float(np.sqrt(np.mean(residual**2))),
        unit_anchor_ms=list(unit),
    )


def paired_samples(rows, phase, field):
    chosen = [s for s in rows if s["measured"] and s["phase"] == phase]
    if [s["iteration"] for s in chosen] != list(range(30, 60)):
        raise ValueError("Missing/duplicate/incorrectly ordered measurements")
    return statistics([s[field] for s in chosen])


def load_observations(root):
    frozen = json.loads((root / "FROZEN.json").read_text())
    model = json.loads((root / "model.json").read_text())
    observations, units, checks, windows, positions, qrecords = [], [], [], [], [], []
    for run in range(3):
        for device in range(2):
            folder = root / "controls-v2" / f"run{run}.gpu{device}"
            for data in json.loads((folder / "units.json").read_text()):
                if not data["finite"]:
                    raise ValueError("Nonfinite unit gradients")
                for phase in ("F", "B"):
                    for metric, field in (("event", "event_ms"), ("wall", "synchronized_wall_ms")):
                        units.append(
                            dict(
                                run=run,
                                worker=device,
                                phase=phase,
                                metric=metric,
                                layer_type=data["class_info"]["layer_type"],
                                statistics=paired_samples(data["samples"], phase, field),
                                raw=str(folder / "units.json"),
                            )
                        )
            for case in frozen["segments"]:
                path = folder / (case["id"] + ".json")
                data = json.loads(path.read_text())
                if data["case"] != case:
                    raise ValueError("Controlled-case metadata mismatch")
                checks.append(
                    dict(
                        case=case["id"],
                        run=run,
                        worker=device,
                        checks=data["checks"],
                        q_checks=data["q_checks"],
                        q_failure=data["q_failure"],
                        raw=str(path),
                    )
                )
                qrecords.append(
                    dict(
                        case=case,
                        run=run,
                        worker=device,
                        samples=data["q_samples"],
                        failure=data["q_failure"],
                        checks=data["q_checks"],
                    )
                )
                for mode in ("sync", "events"):
                    samples = [s for s in data["samples"] if s["mode"] == mode]
                    for phase in ("F", "B"):
                        metrics = [("event", "event_ms")]
                        if mode == "sync":
                            metrics.append(("wall", "synchronized_wall_ms"))
                        for metric, field in metrics:
                            observations.append(
                                dict(
                                    source="controlled_body",
                                    mode=mode,
                                    metric=metric,
                                    case=case["id"],
                                    partition=None,
                                    run=run,
                                    worker=device,
                                    phase=phase,
                                    counts=case["counts"],
                                    pattern=case["pattern"],
                                    length=len(case["pattern"]),
                                    range=case["range"],
                                    role="interior",
                                    design_role=case["role"],
                                    statistics=paired_samples(samples, phase, field),
                                    raw=str(path),
                                )
                            )
        for mode in ("off", "pipeline_sync", "pipeline_events"):
            for part in frozen["partitions"]:
                folder = root / "pipeline" / f"run{run}.{mode}" / part["id"]
                meta = [json.loads((folder / f"result.rank{r}.json").read_text()) for r in range(2)]
                windows.append(
                    dict(
                        run=run,
                        mode=mode,
                        partition=part["id"],
                        step_ms=max(d["continuous_wall_ms"] for d in meta) / 30,
                        windows_ms=[d["continuous_wall_ms"] for d in meta],
                        host_dispatch=[
                            statistics([s["host_dispatch_ms"] for s in d["samples"]]) for d in meta
                        ],
                        warmup_tail=[
                            statistics([s["host_dispatch_ms"] for s in d["warmup_samples"][-10:]])
                            for d in meta
                        ],
                        measured_thirds=[
                            [
                                statistics(
                                    [s["host_dispatch_ms"] for s in d["samples"][i : i + 10]]
                                )
                                for i in (0, 10, 20)
                            ]
                            for d in meta
                        ],
                    )
                )
                if mode == "off":
                    continue
                for rank in range(2):
                    path = folder / f"stage_samples.rank{rank}.json"
                    raw = json.loads(path.read_text())
                    order = frozen["order"][rank]
                    validate_operations(raw, order)
                    if any(
                        r["worker"] != rank
                        or r["partition_id"] != part["id"]
                        or r["run_id"] != f"run{run}"
                        or r["stage_layer_range"]
                        != part["cuts"][r["logical_stage"] : r["logical_stage"] + 2]
                        for r in raw
                    ):
                        raise ValueError("Pipeline provenance mismatch")
                    first_b = next(i for i, op in enumerate(order) if op["kind"] == "B")
                    last_f = max(i for i, op in enumerate(order) if op["kind"] == "F")
                    for stage in range(rank, 4, 2):
                        a, b = part["cuts"][stage : stage + 2]
                        pattern = model["hybrid_pattern"][a:b]
                        for phase, name in (("F", "forward_compute"), ("B", "backward_compute")):
                            selected = [
                                s
                                for s in raw
                                if s["measured"]
                                and s["logical_stage"] == stage
                                and s["phase"] == name
                            ]
                            if len(selected) != 120:
                                raise ValueError("Pipeline stage must have all 30*4 operations")
                            metrics = [("event", "elapsed_ms")]
                            if mode == "pipeline_sync":
                                metrics.append(("wall", "synchronized_wall_ms"))
                            for metric, field in metrics:
                                row = dict(
                                    source="pipeline",
                                    mode=mode,
                                    metric=metric,
                                    case=part["id"],
                                    partition=part["id"],
                                    stage=stage,
                                    run=run,
                                    worker=rank,
                                    phase=phase,
                                    counts=counts(pattern),
                                    pattern=pattern,
                                    length=b - a,
                                    range=[a, b],
                                    role=(
                                        "first"
                                        if stage == 0
                                        else "last" if stage == 3 else "interior"
                                    ),
                                    design_role=part["role"],
                                    statistics=statistics([s[field] for s in selected]),
                                    raw=str(path),
                                )
                                observations.append(row)
                                if metric == "event":
                                    for pos in ("warmup", "steady", "cooldown"):
                                        samples = [
                                            s
                                            for s in selected
                                            if (
                                                "warmup"
                                                if s["schedule_position"] < first_b
                                                else (
                                                    "cooldown"
                                                    if s["schedule_position"] > last_f
                                                    else "steady"
                                                )
                                            )
                                            == pos
                                        ]
                                        if samples:
                                            positions.append(
                                                dict(
                                                    row,
                                                    pipeline_position=pos,
                                                    statistics=statistics(
                                                        [s[field] for s in samples]
                                                    ),
                                                )
                                            )
    return observations, units, checks, windows, positions, qrecords


def identify(observations, units):
    tables = {}
    for phase in ("F", "B"):
        for metric in ("event", "wall"):
            tables[phase, metric] = [
                float(
                    np.mean(
                        [
                            u["statistics"]["mean"]
                            for u in units
                            if u["run"] in (0, 1)
                            and u["phase"] == phase
                            and u["metric"] == metric
                            and u["layer_type"] == t
                        ]
                    )
                )
                for t in TYPES
            ]
    fits, predictions = [], []
    groups = sorted({(r["source"], r["mode"], r["metric"], r["phase"]) for r in observations})
    for source, mode, metric, phase in groups:
        rows = [
            r
            for r in observations
            if (r["source"], r["mode"], r["metric"], r["phase"]) == (source, mode, metric, phase)
        ]
        for model in ("H1", "H2", "H2-boundary") if source == "pipeline" else ("H1", "H2"):
            selected = [r for r in rows if model == "H2-boundary" or r["role"] == "interior"]
            train = [r for r in selected if training(r)]
            unit = tables[phase, metric]
            fit = solve(train, unit, model)
            identity = dict(source=source, mode=mode, metric=metric, phase=phase, model=model)
            if fit["status"] != "identified":
                fits.append(dict(identity, **fit))
                continue
            beta = np.asarray(fit["coefficients"])
            fit["by_training_run"] = {
                str(run): solve([r for r in train if r["run"] == run], unit, model)
                for run in (0, 1)
            }
            fit["leave_one_training_case_out"] = {
                case: solve([r for r in train if r["case"] != case], unit, model)
                for case in sorted({r["case"] for r in train})
            }
            pred = []
            for row in selected:
                p = float(np.dot(features(row, unit, model), beta))
                pred.append(
                    dict(
                        row,
                        model=model,
                        prediction_ms=p,
                        relative_error=p / row["statistics"]["mean"] - 1,
                        subset=subset(row),
                    )
                )
            fit["positive_all_candidates"] = all(r["prediction_ms"] > 0 for r in pred)
            fit["minimum_prediction_ms"] = min(r["prediction_ms"] for r in pred)
            fit["metrics"] = {
                s: error_metrics([r for r in pred if r["subset"] == s])
                for s in sorted({r["subset"] for r in pred})
            }
            # Effective coefficients in count basis must reproduce the anchored fit.
            for row in pred:
                x = row["counts"] + [1]
                if model == "H2-boundary":
                    x += [int(row["role"] == "first"), int(row["role"] == "last")]
                if not np.isclose(
                    np.dot(x, fit["count_parameterization"]),
                    row["prediction_ms"],
                    atol=1e-10,
                    rtol=1e-10,
                ):
                    raise ValueError("Count/scaled parameterizations disagree")
            fits.append(dict(identity, **fit))
            predictions.extend(pred)
    return fits, predictions, tables


def entries(root):
    rows, checks = [], []
    for run in range(3):
        for worker in range(2):
            folder = root / "q-fresh" / f"run{run}.gpu{worker}"
            for case in json.loads((root / "FROZEN.json").read_text())["segments"]:
                if case["role"] != "train":
                    continue
                path = folder / (case["id"] + ".json")
                data = json.loads(path.read_text())
                if data["case"] != case:
                    raise ValueError("q case mismatch")
                checks.append(dict(case=case["id"], run=run, worker=worker, checks=data["checks"]))
                for q in data["q_order"]:
                    samples = [s for s in data["samples"] if s["q"] == q]
                    rows.append(
                        dict(
                            run=run,
                            worker=worker,
                            case=case["id"],
                            q=q,
                            counts=case["counts"],
                            design_role="train",
                            role="interior",
                            statistics=paired_samples(samples, "B", "event_ms"),
                            raw=str(path),
                        )
                    )
    admitted = all(x["equal"] for c in checks for x in c["checks"].values())
    result = dict(
        scope="Supplement only: fresh isomorphic disconnected graphs, F construction outside B event; not the failed retained-object experiment; not a pipeline h",
        checks=checks,
        rows=rows,
        admitted=admitted,
    )
    if admitted:
        train = [r for r in rows if training(r)]
        fit = solve(train, [1, 1, 1], "H2-entry")
        if fit["status"] != "identified":
            raise ValueError("q design unidentifiable")
        fit["by_training_run"] = {
            str(run): solve([r for r in train if r["run"] == run], [1, 1, 1], "H2-entry")
            for run in (0, 1)
        }
        pred = [
            dict(
                r,
                prediction_ms=float(
                    np.dot(features(r, [1, 1, 1], "H2-entry"), fit["coefficients"])
                ),
            )
            for r in rows
        ]
        for r in pred:
            r["relative_error"] = r["prediction_ms"] / r["statistics"]["mean"] - 1
        fit["metrics"] = {
            s: error_metrics([r for r in pred if subset(r) == s])
            for s in ("train", "independent_run")
        }
        fit["positive_all_candidates"] = all(r["prediction_ms"] > 0 for r in pred)
        paired = []
        for r in rows:
            if r["q"] == 1:
                continue
            base = next(
                b
                for b in rows
                if b["q"] == 1 and all(b[k] == r[k] for k in ("run", "worker", "case"))
            )
            paired.append(
                dict(
                    run=r["run"],
                    worker=r["worker"],
                    case=r["case"],
                    q=r["q"],
                    incremental_event_ms_per_entry=(
                        r["statistics"]["mean"] - base["statistics"]["mean"]
                    )
                    / (r["q"] - 1),
                )
            )
        result.update(fit=fit, predictions=pred, paired_entry_deltas=paired)
    write_new(root / "analysis/entry_supplement.json", result)
    print(json.dumps({k: v for k, v in result.items() if k in ("admitted", "fit")}, indent=2))


def analyze(root):
    for name, sha in json.loads((root / "FREEZE_HASHES.json").read_text()).items():
        if digest(root / name) != sha:
            raise ValueError("Frozen artifact changed: " + name)
    observations, units, checks, windows, positions, qrecords = load_observations(root)
    fits, predictions, tables = identify(observations, units)
    output = root / "analysis"
    output.mkdir(exist_ok=False)
    audit_data = json.loads((root / "numerical_audit.json").read_text())
    admitted = all(
        c["checks"][m]["equal"] and c["checks"][m]["output"]["equal"]
        for c in checks
        for m in ("sync", "events")
    )
    replay = all(
        c["checks"]["layer_replay"]["equal"] and c["checks"]["sgd_max_abs"] == 0 for c in checks
    )
    gate = dict(
        controlled_body_numerically_equal=admitted,
        exact_layer_replay_and_sgd=replay,
        pipeline_numerically_equal=audit_data["pipeline_equivalent"],
        pipeline_prediction_scope=(
            "descriptive diagnostic, NOT equivalent-training validation"
            if not audit_data["pipeline_equivalent"]
            else "numerically admitted"
        ),
        q_status="unavailable" if any(q["failure"] for q in qrecords) else "requires explicit fit",
        q_failure_count=sum(q["failure"] is not None for q in qrecords),
        q_fit_performed=False,
    )
    for fit in fits:
        fit["numerical_admission"] = (
            gate["pipeline_prediction_scope"]
            if fit["source"] == "pipeline"
            else (
                "fixed-state body checks passed"
                if admitted and replay
                else "numerical checks failed"
            )
        )
    for prediction in predictions:
        prediction["equivalent_training_claim_allowed"] = (
            audit_data["pipeline_equivalent"] if prediction["source"] == "pipeline" else False
        )
    write_new(output / "admission.json", gate)
    for name, value in (
        ("observations", observations),
        ("units", units),
        ("numerical_checks", checks),
        ("fits", fits),
        ("predictions", predictions),
        ("windows", windows),
        ("pipeline_positions", positions),
        ("q_records", qrecords),
    ):
        write_new(output / (name + ".json"), value)
    write_new(
        output / "fixed_unit_table.json",
        {
            phase: {metric: tables[phase, metric] for metric in ("event", "wall")}
            for phase in ("F", "B")
        },
    )
    grouped = defaultdict(list)
    for p in predictions:
        for dimension in ("length", "pattern", "role", "worker", "partition", "run"):
            key = tuple(p[k] for k in ("source", "mode", "metric", "phase", "model", "subset")) + (
                dimension,
                str(p[dimension]),
            )
            grouped[key].append(p)
    write_new(
        output / "grouped_residuals.json",
        [dict(key=list(k), **error_metrics(v)) for k, v in grouped.items()],
    )
    perturb = []
    for part in sorted({w["partition"] for w in windows}):
        pairs = []
        for run in range(3):
            d = {
                w["mode"]: w["step_ms"]
                for w in windows
                if w["partition"] == part and w["run"] == run
            }
            pairs.append(
                dict(run=run, **d, relative_difference=d["pipeline_events"] / d["off"] - 1)
            )
        perturb.append(
            dict(
                partition=part,
                runs=pairs,
                interval=paired_interval([p["relative_difference"] for p in pairs]),
                off_run_statistics=statistics([p["off"] for p in pairs]),
                events_run_statistics=statistics([p["pipeline_events"] for p in pairs]),
            )
        )
    write_new(output / "perturbation.json", perturb)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for row, phase in enumerate(("F", "B")):
        for col, source in enumerate(("controlled_body", "pipeline")):
            ax = axes[row, col]
            for model in ("H1", "H2"):
                p = [
                    p
                    for p in predictions
                    if p["source"] == source
                    and p["phase"] == phase
                    and p["model"] == model
                    and p["metric"] == "event"
                    and p["mode"] in ("sync", "pipeline_sync")
                    and p["subset"] != "train"
                ]
                ax.scatter(
                    [r["length"] for r in p],
                    [r["relative_error"] * 100 for r in p],
                    s=12,
                    alpha=0.5,
                    label=model,
                )
            ax.axhline(0, color="black", lw=0.7)
            ax.set_title(source + " " + phase)
            ax.set_ylabel("Heldout signed error (%)")
            ax.set_xlabel("Residual units per stage")
            ax.legend()
    fig.tight_layout()
    fig.savefig(output / "heldout_residuals.png", dpi=160)
    plt.close(fig)
    write_new(
        output / "FILES.json",
        {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*")) if p.is_file()},
    )
    print(json.dumps(gate, indent=2))
    for f in fits:
        if f["metric"] == "event":
            print(f["source"], f["mode"], f["phase"], f["model"], f.get("metrics"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("audit", "fit", "boundary", "entries"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit-folder", default="audit")
    parser.add_argument("--audit-output", default="numerical_audit.json")
    args = parser.parse_args()
    if args.action == "audit":
        audit(args.root, args.audit_folder, args.audit_output)
    elif args.action == "boundary":
        boundary_diagnosis(args.root, args.audit_folder, args.audit_output)
    elif args.action == "entries":
        entries(args.root)
    else:
        analyze(args.root)
