# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tables derived only from model specifications and accepted experiment receipts."""

import csv
import json
from pathlib import Path

from tools.slackpipe_eval_common import write_json
from tools.slackpipe_eval_config import METHODS, load_model, parameter_breakdown


def write_csv(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) or ["status"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {
                k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                for k, v in row.items()
            }
            for row in rows
        )


def latex_model_table(models: list) -> str:
    def escape(s):
        return str(s).replace("_", r"\_").replace("%", r"\%")

    lines = [
        r"\begin{tabular}{llrrrll}",
        r"Model & Size (actual B) & L & V & H & FFN Type & Attn. Type \\",
        r"\hline",
    ]
    for family, label in (
        ("llama", "LLaMA-style (Homo.)"),
        ("nemotron_h", "Nemotron-H-style (Hetero.)"),
    ):
        lines.append(r"\multicolumn{7}{l}{" + label + r"} \\")
        for model in sorted(models, key=lambda m: m["expected_parameter_scale"]):
            if model["model_family"] == family:
                count = parameter_breakdown(model)["exact_parameter_count"]
                fields = [
                    model["name"],
                    f"{count / 1e9:.3f}",
                    model["num_layers"],
                    model["vocab_size"],
                    model["hidden_size"],
                    model["ffn_type"],
                    model["attention_type"] + (" + Mamba2" if family == "nemotron_h" else ""),
                ]
                lines.append(" & ".join(map(escape, fields)) + r" \\")
    return "\n".join(lines + [r"\end{tabular}", ""])


def benefits(times: dict) -> dict:
    return {
        f"{b}_vs_{a}": (times[a] - times[b]) / times[a]
        for i, a in enumerate(METHODS)
        for b in tuple(METHODS)[i + 1 :]
        if a in times and b in times and times[a] > 0
    }


def comparison_identity(summary: dict) -> dict:
    """Allow partition/N/schedule differences, not model, data or collection changes."""
    keys = (
        "model_config_hash",
        "exact_parameter_count",
        "pp",
        "tp",
        "dp",
        "cp",
        "num_microbatches",
        "micro_batch_size",
        "global_batch_size",
        "seq_length",
        "precision",
        "seed",
        "learning_rate",
        "tf32",
        "dropout",
    )
    provenance = []
    for record in summary["collection_provenance"]:
        provenance.append(
            dict(
                environment=record["environment"],
                timing_definitions=record["timing_definitions"],
                options={k: v for k, v in record["options"].items() if k != "run_id"},
            )
        )
    if any(p != provenance[0] for p in provenance):
        raise ValueError("Independent runs use different environment/measurement policies")
    return dict(config={k: summary["config"][k] for k in keys}, collection=provenance[0])


def export_tables(output: Path, config_paths: list) -> None:
    from tools.slackpipe_eval_common import summarize_samples
    from tools.slackpipe_eval_receipts import receipt_valid

    models = [load_model(p) for p in config_paths]
    model_rows = [
        {**m, **{k: v for k, v in parameter_breakdown(m).items() if k != "layers"}} for m in models
    ]
    write_json(output / "model_configs.json", model_rows)
    write_csv(output / "model_configs.csv", model_rows)
    (output / "model_configs.tex").write_text(latex_model_table(models))
    rows, calibrations, solvers, memories = [], [], [], []
    grouped = {}
    for path in sorted(output.glob("*/*/receipts/*.json")):
        receipt = json.loads(path.read_text())
        root = path.parent.parent
        stage = receipt["stage"]
        valid = receipt_valid(receipt, receipt.get("context"), root)
        base = dict(
            model=f"{root.parent.name}_{root.name}",
            stage=stage,
            method_id=receipt.get("method_id"),
            executor=receipt.get("executor"),
            transport=receipt.get("transport"),
            solver_profile=receipt.get("solver_profile"),
            profile_identity=receipt.get("profile_identity"),
            status=receipt["status"],
            receipt=str(path.relative_to(output)),
        )
        if stage == "env" and valid:
            memory = json.loads((root / receipt["directory"] / "memory_preflight.json").read_text())
            memories.append({**base, **memory})
        if not valid or receipt["status"] != "passed":
            rows.append(
                {
                    **base,
                    "status": (
                        "invalid_artifacts" if receipt["status"] == "passed" else receipt["status"]
                    ),
                }
            )
            continue
        context = receipt["context"]
        base.update(
            schedule=receipt.get("schedule", path.name.split(".")[0]), **context["topology"]
        )
        if stage == "benchmark":
            summary = json.loads((root / receipt["summary"]).read_text())
            grouped.setdefault((base["model"], base["schedule"]), []).append((base, summary))
        elif stage == "calibrate":
            profile = json.loads((root / receipt["profile"]).read_text())
            calibrations.append(
                {
                    **base,
                    "schema": profile["schema_version"],
                    "profile_hash": profile["cost_profile_hash"],
                    "model_config": profile["model_config"],
                }
            )
        elif stage == "solve":
            plan = json.loads((root / receipt["plan"]).read_text())
            solver = json.loads((root / receipt["solver_result"]).read_text())
            solvers.append(
                {
                    **base,
                    "solver_status": solver.get("solver_status_raw"),
                    "solver": solver,
                    "predicted_makespan": plan.get("predicted_makespan"),
                    "predicted_makespan_units": plan.get("cost_model", {}).get("units"),
                    "layer_split": plan.get("layer_split"),
                    "layer_cuts": plan.get("layer_cuts"),
                }
            )
    benchmarks, times, identities = [], {}, {}
    for (model, schedule), entries in grouped.items():
        base = entries[0][0]
        summaries = [e[1] for e in entries]
        identity = comparison_identity(summaries[0])
        if any(comparison_identity(s) != identity for s in summaries):
            raise ValueError("Refusing to pool incomparable benchmark provenance")
        if model in identities and identities[model] != identity:
            raise ValueError(
                "Refusing cross-method comparison with different model/data/environment/policy"
            )
        identities[model] = identity
        walls = [w for s in summaries for w in s["continuous_step_ms_by_run"]]
        combined = summarize_samples([[w] for w in walls])
        if combined["num_repetitions"] == 1:
            combined["ci95_ms"] = combined["rep_mean_stddev_ms"] = None
        mean = sum(walls) / len(walls)
        config = summaries[0]["config"]
        if any(s["config"] != config for s in summaries):
            raise ValueError("Refusing to pool incomparable benchmark configs")
        row = {
            **base,
            **combined,
            "config": config,
            "continuous_mean_ms": mean,
            "continuous_ci95_ms": combined["ci95_ms"],
            "summary_metric": "max-rank continuous window wall time / measured steps",
            "samples_per_second": config["global_batch_size"] * 1000 / mean,
            "tokens_per_second": config["global_batch_size"] * config["seq_length"] * 1000 / mean,
            "peak_allocated_bytes": max(s["peak_allocated_bytes"] for s in summaries),
            "peak_reserved_bytes": max(s["peak_reserved_bytes"] for s in summaries),
        }
        benchmarks.append(row)
        times.setdefault(model, {})[schedule] = mean
    for row in benchmarks:
        row.update(benefits(times[row["model"]]))
    rows.extend(benchmarks)
    write_json(output / "experiment_summary.json", rows)
    for name, records in (
        ("experiment_summary", rows),
        ("benchmark_summary", benchmarks),
        ("calibration_summary", calibrations),
        ("solver_summary", solvers),
        ("memory_summary", memories),
    ):
        write_csv(output / f"{name}.csv", records)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, default=Path("configs/slackpipe_eval"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_tables(args.output, sorted(args.configs.glob("*.json")))


if __name__ == "__main__":
    main()
