# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Predeclare stage experiments and fit v2 costs without touching legacy estimators."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from megatron.core.pipeline_parallel.slackpipe.cost_profile import profile_fingerprint
from megatron.core.pipeline_parallel.slackpipe.manifest import build_model_manifest
from megatron.core.pipeline_parallel.slackpipe.profile_integrity import validate_profile_integrity
from tools.slackpipe_eval_config import load_model, transformer_config


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def features(manifest, cuts):
    kinds = [r["config_class"] for r in manifest["layers"]]
    keys = sorted(set(kinds))
    return np.asarray(
        [
            [kinds[a:b].count(k) for k in keys] + [int(s == 0), int(s == len(cuts) - 2)]
            for s, (a, b) in enumerate(zip(cuts, cuts[1:]))
        ],
        dtype=float,
    )


def read_design(path, manifest, stages):
    value = json.loads(path.read_text())
    if (
        value["schema_version"] != "slackpipe.stage_design.v1"
        or value["manifest_hash"] != manifest["manifest_hash"]
    ):
        raise ValueError("Design schema/manifest mismatch")
    seen, names = set(), set()
    for p in value["partitions"]:
        cuts = tuple(p["cuts"])
        if (
            len(cuts) != stages + 1
            or cuts[0] != 0
            or cuts[-1] != len(manifest["layers"])
            or any(type(c) is not int for c in cuts)
            or any(a >= b for a, b in zip(cuts, cuts[1:]))
        ):
            raise ValueError("Invalid contiguous partition")
        if cuts in seen or p["id"] in names or p["role"] not in ("train", "validation", "check"):
            raise ValueError("Duplicate partition/identity or invalid role")
        if Path(p["id"]).name != p["id"] or p["id"] in (".", ".."):
            raise ValueError("Unsafe partition ID")
        seen.add(cuts)
        names.add(p["id"])
    if not {"train", "validation"} <= {p["role"] for p in value["partitions"]}:
        raise ValueError("Disjoint train/validation partitions required")
    return value


def make_design(manifest, stages, checks=(), train_count=8, seed=1234, max_stage_layers=20):
    length = len(manifest["layers"])
    if stages < 2 or length <= stages or train_count < 1:
        raise ValueError("Need multiple nonempty stages and training partitions")
    rng = np.random.default_rng(seed)
    candidates = {
        tuple(
            [0]
            + sorted(rng.choice(np.arange(1, length), stages - 1, replace=False).tolist())
            + [length]
        )
        for _ in range(1500)
    }
    # Include short and long stages at every logical position, not just near-uniform cuts.
    for n in (2, 3, 4, 6, 17, 18, 19, 20):
        if length - n < stages - 1:
            continue
        rest = [
            (length - n) * (s + 1) // (stages - 1) - (length - n) * s // (stages - 1)
            for s in range(stages - 1)
        ]
        for position in range(stages):
            split = rest.copy()
            split.insert(position, n)
            candidates.add(tuple([0] + np.cumsum(split).tolist()))
    candidates = {c for c in candidates if max(np.diff(c)) <= max_stage_layers}
    if len(candidates) <= train_count:
        raise ValueError(
            "Not enough distinct bounded partitions; increase --max-stage-layers or reduce --train-partitions"
        )
    check_cuts = []
    for split in checks:
        if len(split) != stages or sum(split) != length or min(split) < 1:
            raise ValueError("Check split has wrong dimensions")
        check_cuts.append(tuple([0] + np.cumsum(split).tolist()))
    check_cuts = list(dict.fromkeys(check_cuts))
    candidates.difference_update(check_cuts)
    # Freeze validation before design optimization or any measurements.
    validation = sorted(candidates)[len(candidates) // 2]
    candidates.remove(validation)
    width = features(manifest, validation).shape[1]
    gram = np.eye(width) * 1e-6
    chosen = []
    covered = set()
    for _ in range(train_count):

        def score(c):
            x = features(manifest, c)
            lengths = set(np.diff(c).tolist())
            return (
                len((lengths & {2, 3, 4, 6, 17, 18, 19, 20}) - covered),
                np.linalg.slogdet(gram + x.T @ x)[1],
            )

        best = max(sorted(candidates), key=score)
        candidates.remove(best)
        chosen.append(best)
        x = features(manifest, best)
        gram += x.T @ x
        covered.update(np.diff(best).tolist())
    matrix = np.concatenate([features(manifest, c) for c in chosen])
    rank = int(np.linalg.matrix_rank(matrix))
    if rank != width:
        raise ValueError(f"Unidentifiable design {rank}/{width}")
    parts = [dict(id=f"train{i}", role="train", cuts=list(c)) for i, c in enumerate(chosen)]
    parts += [dict(id="validation0", role="validation", cuts=list(validation))]
    parts += [dict(id=f"check{i}", role="check", cuts=list(c)) for i, c in enumerate(check_cuts)]
    return dict(
        schema_version="slackpipe.stage_design.v1",
        manifest_hash=manifest["manifest_hash"],
        seed=seed,
        partitions=parts,
        features=sorted({r["config_class"] for r in manifest["layers"]}) + ["first", "last"],
        rank=rank,
        columns=width,
        condition=float(np.linalg.cond(matrix)),
        training_lengths=sorted(covered),
        max_stage_layers=max_stage_layers,
        split_policy="partition holdout frozen before fitting; independent processes/runs required",
    )


def statistics(values):
    a = np.asarray(values, dtype=float)
    if not len(a) or not np.all(np.isfinite(a)) or np.any(a <= 0):
        raise ValueError("Missing, failed or nonpositive timing samples")
    mean = float(a.mean())
    return dict(
        count=len(a),
        mean=mean,
        median=float(np.median(a)),
        stddev=float(a.std()),
        min=float(a.min()),
        max=float(a.max()),
        p05=float(np.quantile(a, 0.05)),
        p95=float(np.quantile(a, 0.95)),
    )


def validate_samples(rows, *, iterations, microbatches, cuts, workers):
    expected = {
        (i, b, s, p)
        for i in iterations
        for b in range(microbatches)
        for s in range(len(cuts) - 1)
        for p in ("forward_compute", "backward_compute")
    }
    seen = set()
    for r in rows:
        key = (r["iteration"], r["microbatch"], r["logical_stage"], r["phase"])
        if key in seen or key not in expected:
            raise ValueError("Duplicate or unexpected F/B event")
        s = r["logical_stage"]
        if (
            r["worker"] != s % workers
            or r["stage_layer_range"] != cuts[s : s + 2]
            or r["status"] != "recorded"
        ):
            raise ValueError("Incorrect layer/worker ownership or failed event")
        statistics([r["elapsed_ms"]])
        seen.add(key)
    if seen != expected:
        raise ValueError("Missing F/B events")


def replay(rows, durations):
    """Replay actual worker order with F/B/FIFO dependencies, no GPU clock fiction."""
    keys = {(r["microbatch"], r["logical_stage"], r["phase"]): r for r in rows}
    stages = 1 + max(k[1] for k in keys)
    deps = {k: set() for k in keys}
    for b, s, p in keys:
        if b:
            deps[b, s, p].add((b - 1, s, p))
        if p == "forward_compute" and s:
            deps[b, s, p].add((b, s - 1, p))
        if p == "backward_compute":
            deps[b, s, p].add((b, s, "forward_compute"))
            if s < stages - 1:
                deps[b, s, p].add((b, s + 1, p))
    by_worker = defaultdict(list)
    for k, r in keys.items():
        by_worker[r["worker"]].append(k)
    for order in by_worker.values():
        order.sort(key=lambda k: keys[k]["schedule_position"])
        for a, b in zip(order, order[1:]):
            deps[b].add(a)
    ends = {}
    while deps:
        ready = [k for k, d in deps.items() if d <= ends.keys()]
        if not ready:
            raise ValueError("Cyclic operation order")
        for k in ready:
            ends[k] = max((ends[d] for d in deps.pop(k)), default=0) + durations[k]
    return max(ends.values())


def fit_runs(paths, manifest, design):
    from scipy.optimize import nnls

    rows, sources, configurations, run_ids, all_raw = [], [], [], set(), []
    environments = {}
    validation_run = None
    partition_map = {p["id"]: p for p in design["partitions"]}
    for path in paths:
        actual = json.loads((path / "calibration_design.json").read_text())
        if actual != design:
            raise ValueError("Run design mismatch")
        raw = json.loads((path / "stage_samples.json").read_text())
        metadata = [json.loads(p.read_text()) for p in sorted(path.glob("*/result.rank*.json"))]
        if not metadata or any(m["mode"] != "events" for m in metadata):
            raise ValueError("Only low-perturbation events runs can fit costs")
        run = metadata[0]["run_id"]
        if run in run_ids:
            raise ValueError("Duplicate independent run identity")
        run_ids.add(run)
        validation_run = run  # Last explicitly supplied run is held out in full.
        for name, part in partition_map.items():
            records = [m for m in metadata if m["partition_id"] == name]
            if not records:
                raise ValueError("Missing partition metadata")
            c = records[0]["configuration"]
            pp = c["pp"]
            if {m["rank"] for m in records} != set(range(pp)):
                raise ValueError("Missing rank")
            for record in records:
                env = record.get("environment", {})
                identity = {
                    k: env.get(k)
                    for k in (
                        "torch",
                        "cuda",
                        "nccl",
                        "gpu",
                        "gpu_uuid",
                        "packages",
                        "allocator_backend",
                        "allocator_environment",
                        "deterministic_environment",
                        "tf32_matmul",
                        "tf32_cudnn",
                        "source",
                    )
                }
                worker = record["rank"]
                if worker in environments and environments[worker] != identity:
                    raise ValueError(
                        "Mixed software/hardware/allocator/source contexts; fit separately"
                    )
                environments[worker] = identity
            iterations = records[0]["measurement_iterations"]
            if any(
                m["measurement_iterations"] != iterations or m["run_id"] != run for m in records
            ):
                raise ValueError("Rank/run/iteration misalignment")
            # Ranges differ intentionally; execution/hardware/software must not.
            configurations.append({k: v for k, v in c.items() if k != "stage_layer_ranges"})
            selected = [
                r for r in raw if r["partition_id"] == name and r["iteration"] in iterations
            ]
            if any(r.get("run_id") != run for r in selected):
                raise ValueError("Raw sample run identity mismatch")
            validate_samples(
                selected,
                iterations=iterations,
                microbatches=c["num_microbatches"],
                cuts=part["cuts"],
                workers=pp,
            )
            all_raw.extend((run, name, r) for r in selected)
            for s in range(len(part["cuts"]) - 1):
                for phase in ("forward", "backward"):
                    data = [
                        r
                        for r in selected
                        if r["logical_stage"] == s and r["phase"] == phase + "_compute"
                    ]
                    means = [
                        np.mean([r["elapsed_ms"] for r in data if r["iteration"] == i])
                        for i in iterations
                    ]
                    rows.append(
                        dict(
                            run_id=run,
                            partition=name,
                            role=part["role"],
                            stage=s,
                            worker=s % pp,
                            layer_range=part["cuts"][s : s + 2],
                            class_counts={
                                k: int(v)
                                for k, v in zip(
                                    sorted({l["config_class"] for l in manifest["layers"]}),
                                    features(manifest, part["cuts"])[s][:-2],
                                )
                            },
                            phase=phase,
                            stage_role=(
                                "first"
                                if s == 0
                                else "last" if s == len(part["cuts"]) - 2 else "middle"
                            ),
                            raw=statistics([r["elapsed_ms"] for r in data]),
                            iteration_means=means,
                            statistics=statistics(means),
                            observed_ms=float(np.median(means)),
                            raw_path=str(path / "stage_samples.json"),
                        )
                    )
        for p in sorted(path.rglob("*.json")):
            sources.append(dict(path=str(p), sha256=digest(p)))
    if any(c != configurations[0] for c in configurations):
        raise ValueError("Mixed execution/transport configurations; fit separately")
    if len(run_ids) < 2:
        raise ValueError("Separate training and heldout runs required")
    coefficients, fit = {}, {}
    for phase in ("forward", "backward"):
        train = [
            r
            for r in rows
            if r["role"] == "train" and r["phase"] == phase and r["run_id"] != validation_run
        ]
        matrix = np.asarray(
            [features(manifest, partition_map[r["partition"]]["cuts"])[r["stage"]] for r in train]
        )
        y = np.asarray([r["observed_ms"] for r in train])
        if np.linalg.matrix_rank(matrix) != matrix.shape[1]:
            raise ValueError("Rank-deficient fit")
        coef, _ = nnls(matrix, y)
        coefficients[phase] = coef
        fit[phase] = dict(
            rank=int(np.linalg.matrix_rank(matrix)),
            condition=float(np.linalg.cond(matrix)),
            relative_rmse=float(np.sqrt(np.mean((matrix @ coef - y) ** 2)) / np.mean(y)),
        )
        for r in rows:
            if r["phase"] != phase:
                continue
            r["predicted_ms"] = float(
                features(manifest, partition_map[r["partition"]]["cuts"])[r["stage"]] @ coef
            )
            r["relative_error"] = (r["predicted_ms"] - r["observed_ms"]) / r["observed_ms"]
    keys = sorted({r["config_class"] for r in manifest["layers"]})
    costs = [
        dict(
            layer_id=i,
            config_class=r["config_class"],
            source="inferred_class_coefficient",
            **{
                p + "_us": float(coefficients[p][keys.index(r["config_class"])]) * 1000
                for p in coefficients
            },
        )
        for i, r in enumerate(manifest["layers"])
    ]
    validation = [r for r in rows if r["role"] == "validation" and r["run_id"] == validation_run]
    mape = float(np.mean([abs(r["relative_error"]) for r in validation]))
    grouped = {}
    for field in ("phase", "stage_role", "worker"):
        grouped[field] = {
            str(k): dict(
                mape=float(np.mean([abs(r["relative_error"]) for r in validation if r[field] == k]))
            )
            for k in {r[field] for r in validation}
        }
    grouped["length"] = {
        str(n): dict(
            mape=float(
                np.mean(
                    [
                        abs(r["relative_error"])
                        for r in validation
                        if r["layer_range"][1] - r["layer_range"][0] == n
                    ]
                )
            )
        )
        for n in {r["layer_range"][1] - r["layer_range"][0] for r in validation}
    }
    grouped["composition"] = {}
    for row in validation:
        key = json.dumps(row["class_counts"], sort_keys=True)
        grouped["composition"].setdefault(key, []).append(abs(row["relative_error"]))
    grouped["composition"] = {
        k: dict(mape=float(np.mean(v)), count=len(v)) for k, v in grouped["composition"].items()
    }
    position_errors = defaultdict(list)
    for run, name, row in all_raw:
        if run == validation_run and partition_map[name]["role"] == "validation":
            prediction = float(
                features(manifest, partition_map[name]["cuts"])[row["logical_stage"]]
                @ coefficients[row["phase"].removesuffix("_compute")]
            )
            position_errors[row.get("pipeline_position", "unrecorded")].append(
                abs(prediction - row["elapsed_ms"]) / row["elapsed_ms"]
            )
    grouped["pipeline_position"] = {
        k: dict(mape=float(np.mean(v)), count=len(v)) for k, v in position_errors.items()
    }
    replay_rows = []
    raw_groups = defaultdict(list)
    for run, name, r in all_raw:
        raw_groups[run, name, r["iteration"]].append(r)
    for (run, name, i), group in raw_groups.items():
        if partition_map[name]["role"] != "validation" or run != validation_run:
            continue
        observed = {
            (r["microbatch"], r["logical_stage"], r["phase"]): r["elapsed_ms"] for r in group
        }
        predicted = {
            k: float(
                features(manifest, partition_map[name]["cuts"])[k[1]]
                @ coefficients[k[2].removesuffix("_compute")]
            )
            for k in observed
        }
        a, b = replay(group, observed), replay(group, predicted)
        replay_rows.append(
            dict(
                run_id=run,
                partition=name,
                iteration=i,
                observed_duration_replay_ms=a,
                predicted_duration_replay_ms=b,
                relative_error=(b - a) / a,
            )
        )
    profile = dict(
        schema_version="slackpipe.cost_profile.v2",
        units="microseconds",
        estimator="native-stage-events-nnls-v1",
        model_manifest_hash=manifest["manifest_hash"],
        model_config=configurations[0],
        parallel_config={k: configurations[0][k] for k in ("pp", "vpp", "tp", "dp", "cp")},
        layer_costs_us=costs,
        class_costs_us={
            k: {p: float(coefficients[p][i]) * 1000 for p in coefficients}
            for i, k in enumerate(keys)
        },
        stage_role_bias_us={
            role: {
                p: (
                    float(coefficients[p][-2 if role == "first" else -1]) * 1000
                    if role != "middle"
                    else 0
                )
                for p in coefficients
            }
            for role in ("first", "middle", "last")
        },
        **{
            "prefix_" + p + "_us": [0.0] + np.cumsum([r[p + "_us"] for r in costs]).tolist()
            for p in coefficients
        },
        measurement_definition=dict(
            boundary="native F/B closure current-stream event envelope",
            includes_internal_gaps=True,
            explicit_p2p_optimizer_excluded=True,
            internal_dependency_wait="unresolved",
            unjoined_stream_work="unresolved",
            estimator="median of iteration means per stage/run; equal row-weight NNLS; no samples removed",
        ),
        calibration_design=design,
        independent_run_split=dict(
            training=sorted(run_ids - {validation_run}), validation=validation_run
        ),
        environment_fingerprints=environments,
        fit=fit,
        sources=sources,
        validation=dict(
            scope="experimental heldout prediction and duration-only replay",
            stage_mape=mape,
            perturbation_verified=False,
            clock_aligned_makespan_verified=False,
        ),
    )
    profile["cost_profile_hash"] = profile_fingerprint(profile)
    validate_profile_integrity(profile)
    report = dict(
        rows=rows,
        heldout_mape=mape,
        heldout_max_abs_relative_error=max(abs(r["relative_error"]) for r in validation),
        grouped_errors=grouped,
        fit=fit,
        fixed_worker_order_replay=replay_rows,
        makespan_definition="duration-only DAG replay, NOT observed cross-GPU wall makespan",
        diagnostic_only=True,
        limitations="No clock-aligned runtime or perturbation evidence; errors are descriptive only",
    )
    return profile, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("design", "fit"))
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--stages", type=int, required=True)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--check-split", action="append", type=lambda s: [int(n) for n in s.split(",")], default=[]
    )
    parser.add_argument("--train-partitions", type=int, default=8)
    parser.add_argument(
        "--max-stage-layers",
        type=int,
        default=20,
        help="Bound calibration partition size only; does not restrict solver search",
    )
    parser.add_argument("--design", type=Path)
    parser.add_argument("--runs", nargs="+", type=Path)
    args = parser.parse_args()
    manifest = build_model_manifest(
        transformer_config(load_model(args.model_config), dict(pp=1, vpp=None), args.precision)
    )
    if args.action == "design":
        write_new(
            args.output,
            make_design(
                manifest,
                args.stages,
                args.check_split,
                args.train_partitions,
                max_stage_layers=args.max_stage_layers,
            ),
        )
    else:
        if not args.design or not args.runs:
            parser.error("fit requires --design and --runs")
        design = read_design(args.design, manifest, args.stages)
        profile, report = fit_runs(args.runs, manifest, design)
        write_new(args.output / "cost_profile.experimental.json", profile)
        write_new(args.output / "fit_validation.json", report)
        print(json.dumps(profile["validation"], indent=2))


if __name__ == "__main__":
    main()
