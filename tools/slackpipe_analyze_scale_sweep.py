"""Regenerate SlackPipe scale/lifecycle tables and plots from measured artifacts."""

import argparse
import csv
import hashlib
import json
import math
import statistics as st
import sys
from graphlib import TopologicalSorter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scipy.stats import t

from megatron.core.pipeline_parallel.slackpipe.plan import load_slackpipe_plan, parse_slackpipe_plan

MODES = {
    "A": "A_uniform_interleaved_1f1b",
    "B": "B_solver_split_interleaved_1f1b",
    "C": "C_slackpipe_solver_order",
}


def read(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {
                key: json.dumps(value) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
            for row in rows
        )


def confidence(values):
    mean = st.fmean(values)
    margin = t.ppf(0.975, len(values) - 1) * st.stdev(values) / math.sqrt(len(values))
    return [float(mean - margin), float(mean + margin)]


def aggregate(paths, mode):
    assert len(paths) == 3, paths
    repetitions, samples, ranks = [], [], []
    for path in paths:
        data = read(path)["modes"][MODES[mode]]
        local = data["rank_results"]
        assert len(local) == 2
        times = [max(pair) for pair in zip(*(r["iteration_cuda_times_ms"] for r in local))]
        assert len(times) >= 50
        repetitions.append(st.fmean(times))
        samples.extend(times)
        ranks.extend(local)
        for rank in local:
            assert all(
                not s["active_iteration"] and s["outstanding_puts"] == 0
                for s in rank["transport_statistics"]
            )
    return dict(
        mean_ms=st.fmean(samples),
        median_ms=st.median(samples),
        stddev_ms=st.pstdev(samples),
        cv=st.pstdev(samples) / st.fmean(samples),
        ci95_ms=confidence(repetitions),
        repetition_means_ms=repetitions,
        measured_iterations=len(samples),
        peak_allocated_bytes=max(r["cuda_peak_allocated_bytes"] for r in ranks),
        peak_reserved_bytes=max(r["cuda_peak_reserved_bytes"] for r in ranks),
        device_used_bytes=max(r["cuda_device_used_bytes"] for r in ranks),
        parameter_count=ranks[0]["parameter_count"],
        rank_results=ranks,
    )


def validate_execution(paths, plan):
    for path in paths:
        for mode, data in read(path)["modes"].items():
            for rank in data["rank_results"]:
                split = [3] * 4 if mode == MODES["A"] else list(plan.layer_split)
                assert rank["local_layer_counts"] == split[rank["rank"] :: 2]
        if MODES["C"] in read(path)["modes"]:
            for rank in range(2):
                trace = read(path.parent / f"slackpipe_trace.rank{rank}.json")
                expected = [
                    dict(kind=op.kind, microbatch=op.microbatch, stage=op.stage)
                    for op in plan.worker_operations(rank)
                ]
                assert trace["matched_plan"] and trace["operations"] == expected


def replay(profile, cuts, orders):
    key = lambda op: (op.kind, op.microbatch, op.stage)
    deps = {key(op): set() for worker in orders for op in worker}
    for (kind, b, s), incoming in deps.items():
        if b:
            incoming.add((kind, b - 1, s))
        if kind == "F" and s:
            incoming.add((kind, b, s - 1))
        if kind == "B":
            incoming.add(("F", b, s))
            if s < 3:
                incoming.add((kind, b, s + 1))
    for worker in orders:
        for previous, op in zip(worker, worker[1:]):
            deps[key(op)].add(key(previous))
    end = {}
    for op in TopologicalSorter(deps).static_order():
        kind, _, s = op
        phase = "forward" if kind == "F" else "backward"
        prefix = profile[f"prefix_{phase}_us"]
        role = "first" if s == 0 else "last" if s == 3 else "middle"
        bias = profile["stage_role_bias_us"][role][phase]
        cost = (
            math.floor(prefix[cuts[s + 1]] + 0.5)
            - math.floor(prefix[cuts[s]] + 0.5)
            + math.floor(bias + 0.5)
        )
        end[op] = max((end[p] for p in deps[op]), default=0) + cost
    return max(end.values())


def lifecycle(root):
    all_rows, summaries = [], []
    names = (
        "lifecycle_megatron_before",
        "lifecycle_recreate",
        "lifecycle_reuse",
        "lifecycle_cached_recreate",
    )
    for name in names:
        for rank in range(2):
            rows = read(root / f"{name}.rank{rank}.json")
            all_rows.extend(dict(test=name, rank=rank, **r) for r in rows)
            phase = "idle" if name == "lifecycle_reuse" else "closed"
            band = [r for r in rows if r["phase"] == phase]
            summary = dict(
                test=name,
                rank=rank,
                cycles=len(band),
                steady_state_phase=phase,
                start=rows[0],
                steady_start=band[0],
                end=rows[-1],
            )
            for metric in (
                "allocated_bytes",
                "reserved_bytes",
                "device_used_bytes",
                "registered_process_groups",
                "live_contexts",
                "live_mailboxes",
            ):
                values = [r[metric] for r in band]
                summary[metric + "_band_delta"] = max(values) - min(values)
            summaries.append(summary)
    write_csv(root / "rma_lifecycle_stress.csv", all_rows)
    write_json(
        root / "rma_lifecycle_stress.json",
        {
            "window_counts": "SlackPipe-owned counts, not native allocator counters",
            "native_allocation_counters": "not exposed by installed symmetric-memory API",
            "summaries": summaries,
            "measurements": all_rows,
        },
    )
    return summaries


def analyze(root):
    checkpoint = read(root / "runtime_checkpoint.json")
    for filename, digest in checkpoint["sha256"].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == digest, filename
    life = lifecycle(root)
    scales, calibration, solver, memory, transport = [], [], [], [], []
    for name in ("SMALL", "MEDIUM", "LARGE"):
        folder = root / name
        profile = read(folder / "cost_profile.json")
        plan = load_slackpipe_plan(folder / "joint.plan.json", pipeline_model_parallel_size=2)
        cuts = [a for a, _ in plan.stage_layer_ranges] + [plan.num_layers]
        assert replay(profile, cuts, plan.operations) == plan.predicted_makespan
        events = read(folder / "calibration_events.json")[0]["events"]
        orders = [
            [
                dict(
                    kind="F" if e["phase"] == "forward_compute" else "B",
                    microbatch=e["microbatch"],
                    stage=e["logical_stage"],
                )
                for e in events
                if e["rank"] == rank and e["iteration"] == 0
            ]
            for rank in range(2)
        ]
        baseline = parse_slackpipe_plan(
            dict(
                schema_version="slackpipe.plan.v2",
                num_microbatches=4,
                num_stages=4,
                num_workers=2,
                num_layers=12,
                layer_cuts=[0, 3, 6, 9, 12],
                stage_to_worker=[0, 1, 0, 1],
                operations=orders,
            )
        )
        predicted = dict(
            A=replay(profile, [0, 3, 6, 9, 12], baseline.operations),
            B=replay(profile, cuts, baseline.operations),
            C=plan.predicted_makespan,
        )
        paths = sorted((folder / "benchmark").glob("rep*/benchmark_summary.json"))
        validate_execution(paths, plan)
        config = read(paths[0])["model_config"]
        config["learning_rate"] = profile["calibration"]["learning_rate"]
        modes = {mode: aggregate(paths, mode) for mode in MODES}
        payload = config["sequence_length"] * config["hidden_size"] * 4
        benefits = {}
        for label, first, second in (
            ("partition", "A", "B"),
            ("schedule", "B", "C"),
            ("overall", "A", "C"),
        ):
            values = [
                100 * (a - b) / a
                for a, b in zip(
                    modes[first]["repetition_means_ms"], modes[second]["repetition_means_ms"]
                )
            ]
            benefits[label] = dict(
                percent=100
                * (modes[first]["mean_ms"] - modes[second]["mean_ms"])
                / modes[first]["mean_ms"],
                repetition_mean_percent=st.fmean(values),
                ci95_percent=confidence(values),
                repetition_values_percent=values,
            )
        for mode, data in modes.items():
            data["samples_per_second"] = 4000 / data["mean_ms"]
            data["tokens_per_second"] = 4000 * config["sequence_length"] / data["mean_ms"]
            memory.append(
                dict(
                    scale=name,
                    mode=mode,
                    activation_payload_bytes=payload,
                    peak_allocated_bytes=data["peak_allocated_bytes"],
                    peak_reserved_bytes=data["peak_reserved_bytes"],
                    device_used_bytes=data["device_used_bytes"],
                    rma_retained_in_some_rotations=mode != "C",
                )
            )
        clean_a = read(paths[0])["modes"][MODES["A"]]["rank_results"]
        clean_b = read(paths[1])["modes"][MODES["B"]]["rank_results"]
        for mode, ranks in (("A", clean_a), ("B", clean_b)):
            assert not any(r["transport_statistics"] for r in ranks)
            memory[-3 + "ABC".index(mode)]["clean_device_used_bytes"] = max(
                r["cuda_device_used_bytes"] for r in ranks
            )
        cal = dict(
            scale=name,
            design=profile["calibration"]["design"],
            forward_relative_rmse=profile["fit"]["forward"]["relative_rmse"],
            backward_relative_rmse=profile["fit"]["backward"]["relative_rmse"],
            class_costs_us={
                k: dict(**v, backward_forward_ratio=v["backward"] / v["forward"])
                for k, v in profile["class_costs_us"].items()
            },
            stage_role_bias_us=profile["stage_role_bias_us"],
        )
        assert cal["design"]["rank"] == cal["design"]["columns"] == 6
        calibration.append(cal)
        solver.append(
            dict(
                scale=name,
                status=plan.solver_status,
                cuts=cuts,
                split=list(plan.layer_split),
                predicted_makespan_us=plan.predicted_makespan,
                uniform_predicted_makespan_us=predicted["A"],
                optimized_interleaved_predicted_makespan_us=predicted["B"],
            )
        )
        scale = dict(
            scale=name,
            config=config,
            parameter_count=modes["A"]["parameter_count"],
            activation_payload_bytes=payload,
            cuts=cuts,
            predicted_us=predicted,
            predicted_C_A=predicted["C"] / predicted["A"],
            measured_C_A=modes["C"]["mean_ms"] / modes["A"]["mean_ms"],
            benefits=benefits,
            calibration=cal,
            modes=modes,
        )
        scale["rma_extra_device_bytes_vs_clean_interleaved"] = modes["C"][
            "device_used_bytes"
        ] - max(r["cuda_device_used_bytes"] for r in clean_a)
        scales.append(scale)
        for backend in ("nccl-p2p", "nccl-rma"):
            paths = sorted((folder / "transport" / backend).glob("rep*/benchmark_summary.json"))
            if name == "MEDIUM":
                continue
            validate_execution(paths, plan)
            data = aggregate(paths, "C")
            ranks = data.pop("rank_results")
            row = dict(
                scale=name,
                transport=backend,
                **data,
                samples_per_second=4000 / data["mean_ms"],
                tokens_per_second=4000 * config["sequence_length"] / data["mean_ms"],
                plan_sha256=hashlib.sha256((folder / "joint.plan.json").read_bytes()).hexdigest(),
                activation_payload_bytes=payload,
                initialization_seconds=st.fmean(
                    r["transport_construction_seconds"]
                    + sum(s["initialization_seconds"] for s in r["transport_statistics"])
                    for r in ranks
                ),
                persistent_device_delta_bytes=max(
                    r["transport_construction_device_bytes"]
                    + sum(s["device_memory_delta_bytes"] for s in r["transport_statistics"])
                    for r in ranks
                ),
                mailbox_bytes=max(
                    sum(s["mailbox_bytes"] for s in r["transport_statistics"]) for r in ranks
                ),
            )
            transport.append(row)
        rma = modes["C"]["rank_results"][0]["transport_statistics"][0]
        scale["rma"] = rma
        memory[-1].update(
            mailbox_bytes=rma["mailbox_bytes"],
            data_window_count=rma["data_window_count"],
            signal_window_count=rma["signal_window_count"],
            signal_window_bytes=rma["signal_window_bytes"],
            extra_device_bytes_vs_clean_interleaved=scale[
                "rma_extra_device_bytes_vs_clean_interleaved"
            ],
        )
        assert rma["mailbox_bytes"] == 6 * 4 * math.ceil(payload / 4096) * 4096
        for mode in modes.values():
            del mode["rank_results"]
    transport_benefits = []
    for name in ("SMALL", "LARGE"):
        p2p, rma = [
            next(r for r in transport if r["scale"] == name and r["transport"] == backend)
            for backend in ("nccl-p2p", "nccl-rma")
        ]
        values = [
            100 * (a - b) / a
            for a, b in zip(p2p["repetition_means_ms"], rma["repetition_means_ms"])
        ]
        rma["extra_total_device_bytes_vs_p2p"] = rma["device_used_bytes"] - p2p["device_used_bytes"]
        transport_benefits.append(
            dict(
                scale=name,
                rma_reduction_mean_percent=st.fmean(values),
                ci95_percent=confidence(values),
                extra_total_device_bytes_vs_p2p=rma["extra_total_device_bytes_vs_p2p"],
            )
        )
    write_json(root / "transport_benefits.json", transport_benefits)
    for name, rows in (
        ("calibration_summary", calibration),
        ("solver_summary", solver),
        ("memory_summary", memory),
        ("transport_comparison", transport),
    ):
        write_csv(root / f"{name}.csv", rows)
        write_json(root / f"{name}.json", rows)
    write_json(root / "scale_summary.json", scales)
    write_csv(
        root / "class_costs.csv",
        [
            dict(scale=c["scale"], config_class=k, **v)
            for c in calibration
            for k, v in c["class_costs_us"].items()
        ],
    )
    write_csv(
        root / "role_biases.csv",
        [
            dict(scale=c["scale"], role=k, **v)
            for c in calibration
            for k, v in c["stage_role_bias_us"].items()
        ],
    )
    write_csv(
        root / "scale_summary.csv",
        [
            dict(
                scale=s["scale"],
                **s["config"],
                activation_payload_bytes=s["activation_payload_bytes"],
                cuts=s["cuts"],
                predicted_C_A=s["predicted_C_A"],
                measured_C_A=s["measured_C_A"],
                benefits=s["benefits"],
                mode=m,
                **data,
            )
            for s in scales
            for m, data in s["modes"].items()
        ],
    )
    plots(root, scales)
    report(root, checkpoint, life, scales, transport, transport_benefits)


def plots(root, scales):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        (root / "plots_unavailable.txt").write_text("matplotlib unavailable; nothing installed\n")
        return
    labels = [s["scale"] for s in scales]
    x = np.arange(3)

    def save(name):
        plt.tight_layout()
        plt.savefig(root / f"{name}.png", dpi=200)
        plt.savefig(root / f"{name}.pdf")
        plt.close()

    for metric, name in (("overall", "speedup_vs_scale"),):
        y = [s["benefits"][metric]["repetition_mean_percent"] for s in scales]
        ci = [s["benefits"][metric]["ci95_percent"] for s in scales]
        plt.errorbar(
            x,
            y,
            yerr=np.array([[v - c[0], c[1] - v] for v, c in zip(y, ci)]).T,
            marker="o",
            capsize=4,
        )
        plt.xticks(x, labels)
        plt.axhline(0, color="gray", linewidth=0.8)
        plt.ylabel("Overall step-time reduction (%)\n95% CI across 3 repetitions")
        save(name)
    for m in MODES:
        y = [s["modes"][m]["mean_ms"] for s in scales]
        ci = [s["modes"][m]["ci95_ms"] for s in scales]
        plt.errorbar(
            x,
            y,
            yerr=np.array([[v - c[0], c[1] - v] for v, c in zip(y, ci)]).T,
            marker="o",
            capsize=4,
            label=m,
        )
    plt.xticks(x, labels)
    plt.ylabel("Step time (ms)")
    plt.legend()
    save("abc_step_time")
    for key, label in (
        ("predicted_C_A", "Predicted compute C/A"),
        ("measured_C_A", "Measured C/A"),
    ):
        plt.plot(x, [s[key] for s in scales], marker="o", label=label)
    plt.xticks(x, labels)
    plt.ylabel("Runtime ratio")
    plt.legend()
    save("predicted_vs_measured")
    plt.plot(
        [s["activation_payload_bytes"] / 1024 for s in scales],
        [s["rma"]["device_memory_delta_bytes"] / 2**20 for s in scales],
        marker="o",
    )
    plt.xlabel("Boundary activation payload (KiB)")
    plt.ylabel("RMA initialization device-memory delta (MiB)")
    save("rma_memory_vs_activation")
    for k in range(3):
        for phase, style in (("forward", "-"), ("backward", "--")):
            plt.plot(
                x,
                [s["calibration"]["class_costs_us"][f"class_{k}"][phase] / 1000 for s in scales],
                style,
                marker="o",
                label=f"{'ABC'[k]} {phase}",
            )
    plt.xticks(x, labels)
    plt.ylabel("Fitted compute per layer (ms)")
    plt.legend(ncol=2)
    save("class_costs")
    for phase in ("forward", "backward"):
        plt.plot(
            x,
            [100 * s["calibration"][f"{phase}_relative_rmse"] for s in scales],
            marker="o",
            label=phase,
        )
    plt.xticks(x, labels)
    plt.ylabel("Calibration relative RMSE (%)")
    plt.legend()
    save("fit_error")


def report(root, checkpoint, life, scales, transport, transport_benefits):
    final_results_path = root / "final_regressions.json"
    final_results = read(final_results_path) if final_results_path.exists() else None
    lines = [
        "# Heterogeneous SlackPipe scale sweep",
        "",
        "## Lifecycle",
        "",
        "The reproduced GPU-memory growth came from test-owned Megatron NCCL process groups, "
        "not retained RMA windows. Eight recreate cycles retained 136 extra registered groups. "
        "RMA-only close returned to a stable baseline even before the cache change.",
        "",
        "The SlackPipe-only fixture now destroys groups it created. Production caches separate "
        "compiled plans from RMA resources. Keys include default/PP group identity, world/PP sizes, "
        "rank, logical edges/placement, shape, dtype, B, transport, and device. Compatible plans "
        "reuse one context. Shutdown rejects active work, frees handles/slots/storage before groups, "
        "drops cache references, and must precede parent-group destruction.",
        "",
        "| Probe (rank 0) | Cycles | Device-memory band delta, MiB | Ending allocated/reserved, MiB |",
        "|---|---:|---:|---:|",
    ]
    for row in life:
        if row["rank"] == 0:
            lines.append(
                f"| {row['test']} | {row['cycles']} | {row['device_used_bytes_band_delta']/2**20:.4f} | "
                f"{row['end']['allocated_bytes']/2**20:.4f} / {row['end']['reserved_bytes']/2**20:.2f} |"
            )
    lines += [
        "",
        "GPU memory plateaus under 120 exact-order M0/M1/M2/M3 iterations and "
        "50 cached shutdown/recreate cycles. Final cache/context/mailbox counts are zero. "
        "The installed API exposes no native allocation counters; window counts describe "
        "SlackPipe ownership. A bounded stress test is evidence against growth, not a proof "
        "for every driver/version or arbitrarily long execution.",
        "",
        "## Ownership audit",
        "",
        "- RMA owns six NCCL groups, one Gloo group, six symmetric allocations, rebased "
        "DLPack slots, native rendezvous handles, and six channel streams.",
        "- Native NCCL allocation destruction erases the installed allocator's cached "
        "handles, deregisters data/signal windows, and frees backing memory. DLPack slots "
        "retain parent allocation ownership until slots are cleared.",
        "- Pending source tensors/events survive until put completion; receives return "
        "ordinary cloned tensors so autograd/model references cannot retain symmetric storage.",
        "- Per-step expected sets must be empty at completion/teardown. No closure or "
        "lru_cache owns RMA tensors. Persistent transport no longer retains a whole plan.",
        "- PyTorch's process-group registry strongly owns groups until explicit destruction. "
        "Only the test harness inspects that private registry, preserving enclosing groups.",
        "- The installed native NCCL manager has a process-lifetime name/raw-communicator "
        "registry with no exposed unregister API. Put/wait use host NCCL APIs, not its "
        "device-communicator creation path. No proportional GPU growth was observed; "
        "this does not certify zero native CPU metadata retention.",
        "",
        "## Correctness and checkpoint",
        "",
        f"Pre-freeze tests: `{json.dumps(checkpoint['tests'])}`.",
        (
            f"Final post-sweep regressions: {final_results['pp1_contracts']['passed']} passed "
            f"({final_results['pp1_contracts']['skipped']} skipped) in PP1/contracts; "
            f"{final_results['pp2_per_rank']['passed']} passed per rank "
            f"({final_results['pp2_per_rank']['skipped']} skipped) in combined PP2, including "
            "the newly calibrated SMALL solver plan. Formatting/compile/diff checks and "
            "all six zero-difference RMA result records are in `final_regressions.json`."
            if final_results
            else "Final post-sweep regressions not yet recorded."
        ),
        "PP1/PP2 homogeneous, heterogeneous, M0/M1/M2/M3 and measured-cost initial/loss/gradient/"
        "post-SGD maximum absolute differences remain exactly zero. The formerly failing "
        "combined PP2 suite now completes. `runtime_checkpoint.json` pins source hashes; "
        "analysis verifies communication/runtime files stayed unchanged during scaling.",
        "",
        "## Method",
        "",
        "Two RTX A2000 6-GiB GPUs; PP2/N4, TP=DP=CP=1, FP32 (TF32 disabled), dropout zero, "
        "untied embeddings, fixed seed 1234. SGD lr is 1e-6 for SMALL/MEDIUM and 1e-9 "
        "for LARGE (identical within each scale's calibration/A/B/C/transport runs). Twelve layers with pattern "
        "AABCABCABCAA: A full attention/FFN4H, B full attention/FFN16H, C no attention/FFN2H. "
        "Head dimension stays 32; B=4, microbatch size 1, vocabulary 128. No OOM search.",
        "",
        "LARGE's original ordinary-interleaved A run at lr=1e-6 developed non-finite values. "
        "A diagnostic reproduced gradient growth through step 21 and failure at step 22, "
        "before any SlackPipe operation. Those failed runs are excluded and preserved under "
        "LARGE/failed_lr1e-6. LARGE was recalibrated and re-solved at the smaller positive "
        "learning rate; no model/communication implementation or finite check was changed. "
        "This is a synthetic-training stability limitation, not evidence of a transport failure "
        "or an investigation establishing the underlying gradient pathology's cause.",
        "",
        "Each scale has independent six-partition ordinary-interleaved calibration (5 warmups, "
        "10 measured iterations/partition), full-rank class/role fit and joint CP-SAT export. "
        "A/B/C run in fresh distributed processes for 3 repetitions, rotated ABC/BCA/CAB, "
        "5 warmups and 50 measured steps per mode. Per-step time is max rank CUDA-event time "
        "including finite checks and SGD. Calibration/NVTX are off in measured steps; C has "
        "one additional diagnostic step before measurement. CI uses Student-t across 3 "
        "repetition means (df=2), not 150 independent steps. Benefit CIs use paired per-"
        "repetition fractional reductions. Sample stddev/CV describe all 150 steps.",
        "",
        "| Scale | H / heads / sequence | Parameters | Payload, KiB | Cuts | F/B fit error, % |",
        "|---|---|---:|---:|---|---|",
    ]
    for s in scales:
        c = s["config"]
        lines.append(
            f"| {s['scale']} | {c['hidden_size']} / {c['num_attention_heads']} / {c['sequence_length']} | "
            f"{s['parameter_count']} | {s['activation_payload_bytes']/1024:g} | {s['cuts']} | "
            f"{100*s['calibration']['forward_relative_rmse']:.2f} / {100*s['calibration']['backward_relative_rmse']:.2f} |"
        )
    lines += [
        "",
        "All calibration designs have rank 6/6 and condition 8.94594. Class costs, "
        "B/F ratios, role biases, solver status and predicted makespans are in the calibration "
        "and solver CSV/JSON tables.",
        "",
        "## Step times",
        "",
        "| Scale | Mode | Mean [95% CI], ms | Median / stddev, ms | CV, % | Tokens/s | Peak allocated/reserved, MiB |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for s in scales:
        for mode, data in s["modes"].items():
            lo, hi = data["ci95_ms"]
            lines.append(
                f"| {s['scale']} | {mode} | {data['mean_ms']:.3f} [{lo:.3f}, {hi:.3f}] | "
                f"{data['median_ms']:.3f} / {data['stddev_ms']:.3f} | {100*data['cv']:.2f} | "
                f"{data['tokens_per_second']:.1f} | {data['peak_allocated_bytes']/2**20:.2f} / "
                f"{data['peak_reserved_bytes']/2**20:.2f} |"
            )
    lines += [
        "",
        "A: uniform interleaved. B: optimized cuts with interleaved. C: same cuts, "
        "SlackPipe order and persistent RMA.",
        "",
        "## Benefits",
        "",
        "Entries show paired repetition mean [95% CI], %. Ratio-of-pooled-mean benefits "
        "are also retained in `scale_summary.json`.",
        "",
        "| Scale | Partition | Schedule | Overall | Predicted C/A | Measured C/A |",
        "|---|---|---|---|---:|---:|",
    ]
    for s in scales:
        cells = []
        for key in ("partition", "schedule", "overall"):
            b = s["benefits"][key]
            cells.append(
                f"{b['repetition_mean_percent']:.2f} [{b['ci95_percent'][0]:.2f}, {b['ci95_percent'][1]:.2f}]"
            )
        lines.append(
            f"| {s['scale']} | "
            + " | ".join(cells)
            + f" | {s['predicted_C_A']:.4f} | {s['measured_C_A']:.4f} |"
        )
    lines += [
        "",
        "Predictions are compute-DAG replays, not end-to-end runtime models. The ordinary "
        "worker order comes from actual calibration events; prefix endpoint/role rounding "
        "matches C++, and replay of C must equal exported makespan exactly.",
        "",
        "## Same-plan transport controls",
        "",
        "| Scale | Transport | Step, ms | Init, s | Persistent device delta, MiB | Mailbox, MiB | Tokens/s |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in transport:
        lines.append(
            f"| {row['scale']} | {row['transport']} | {row['mean_ms']:.3f} | "
            f"{row['initialization_seconds']:.4f} | {row['persistent_device_delta_bytes']/2**20:.2f} | "
            f"{row['mailbox_bytes']/2**20:.2f} | {row['tokens_per_second']:.1f} |"
        )
    lines += ["", "RMA step-time reductions versus P2P (paired repetition 95% CI):"]
    for row in transport_benefits:
        lo, hi = row["ci95_percent"]
        lines.append(
            f"- {row['scale']}: {row['rma_reduction_mean_percent']:.2f}% "
            f"[{lo:.2f}, {hi:.2f}]; total extra device memory "
            f"{row['extra_total_device_bytes_vs_p2p']/2**20:.2f} MiB. "
            + (
                "Interval crosses zero; no transport advantage established."
                if lo <= 0 <= hi
                else "Interval excludes zero in this small repeated control."
            )
        )
    lines += [
        "",
        "Controls use the exact selected plan for both transports, three fresh-process "
        "repetitions each, alternating transport order, 5 warmups/50 steps. Setup is excluded "
        "from step time. Init includes runtime/group construction and RMA's lazy registration "
        "warmup; it excludes model construction. P2P mailboxes are per-iteration, not persistent.",
        "",
        "## Memory scaling",
        "",
        "Requested RMA mailbox bytes per GPU are `2 * edges * B * align_up(payload,4096)`; "
        "six data and six signal windows stay fixed. The measured payload sweep tests "
        "activation-size scaling; it does not independently vary B or edge count. Their "
        "linear factors follow allocation structure, not an empirical B/edge sweep. Native "
        "NCCL overhead and VMM granularity dominate small payloads. PyTorch allocator peaks "
        "omit this native footprint. A/B device-used measurements can include retained RMA "
        "from earlier C in rotated runs; isolated transport controls provide clean baselines.",
        "",
        "| Scale | Payload, KiB | RMA mailboxes, MiB | RMA init device delta, MiB | Total extra vs clean A, MiB |",
        "|---|---:|---:|---:|---:|",
    ]
    for s in scales:
        lines.append(
            f"| {s['scale']} | {s['activation_payload_bytes']/1024:g} | {s['rma']['mailbox_bytes']/2**20:.2f} | "
            f"{s['rma']['device_memory_delta_bytes']/2**20:.2f} | {s['rma_extra_device_bytes_vs_clean_interleaved']/2**20:.2f} |"
        )
    lines += [
        "",
        "## Conclusions",
        "",
        "1. No proportional GPU-memory growth remains in the tested persistent reuse and "
        "explicit-teardown paths. The root cause of the combined-suite growth was fixed in "
        "test process-group ownership, with production transport caching/teardown made explicit.",
    ]
    values = [s["benefits"]["overall"]["repetition_mean_percent"] for s in scales]
    trend = (
        "increases"
        if values[0] < values[1] < values[2]
        else "decreases" if values[0] > values[1] > values[2] else "is not monotonic"
    )
    supported = [s["scale"] for s in scales if s["benefits"]["overall"]["ci95_percent"][0] > 0]
    work_ratio = (
        scales[-1]["parameter_count"]
        * scales[-1]["config"]["sequence_length"]
        / (scales[0]["parameter_count"] * scales[0]["config"]["sequence_length"])
    )
    time_ratio = scales[-1]["modes"]["A"]["mean_ms"] / scales[0]["modes"]["A"]["mean_ms"]
    lines += [
        f"2. Overall benefit {trend} in point estimates ({', '.join(f'{v:.2f}%' for v in values)}). "
        f"Positive overall benefit excludes zero only for: {', '.join(supported) or 'none'}. "
        "These three scales/repetitions on one two-GPU machine do not establish a general "
        "scaling law. Inspect the schedule-only control separately from partition effects.",
        "",
        f"The parameter-times-tokens work proxy grows {work_ratio:.1f}x from SMALL to LARGE, "
        f"while ordinary A step time grows {time_ratio:.2f}x. This is consistent with substantial "
        "fixed launch/control costs at SMALL and better kernel utilization at larger dimensions; "
        "it is not a direct measurement separating useful arithmetic from communication overhead. "
        "Calibration event spans also include dispatch gaps. The changed LARGE learning rate "
        "and only three repetitions limit causal claims about compute scaling.",
        "",
        "Reproduce with `bash tools/slackpipe_scale_sweep.sh all`, then inside slackpipe-dev "
        "run `python tools/slackpipe_analyze_scale_sweep.py`. Raw events, plans, model manifests, "
        "profiles, per-rank timings/traces, CSV/JSON tables, and six PNG/PDF plots are retained.",
        "",
        "Unsupported: PP>2, TP/DP/CP>1, BF16/FP16, recomputation, CUDA graphs, distributed "
        "optimizer, new solver algorithms, variable boundaries, multi-node, Mamba/MoE/MTP, "
        "and overlap optimization. No communication tuning followed the lifecycle checkpoint.",
        "",
        "## Changed files in this milestone",
        "",
        "| File | Change |",
        "|---|---|",
        "| megatron/core/pipeline_parallel/slackpipe/schedule.py | Separate RMA transport cache, context/shape/B key, plan invalidation, idle-only shutdown. |",
        "| megatron/core/pipeline_parallel/slackpipe/communication_rma.py | Retain dimensions instead of plan; complete idle checks and explicit reference cleanup. |",
        "| megatron/core/pipeline_parallel/slackpipe/__init__.py | Lazy public shutdown export. |",
        "| megatron/core/pipeline_parallel/slackpipe/README.rma.md | Cache/lifetime/ownership documentation. |",
        "| tests/unit_tests/pipeline_parallel/conftest.py | Scope group ownership fixture to SlackPipe tests. |",
        "| tests/unit_tests/pipeline_parallel/slackpipe_test_utils.py | Track and destroy only test-owned process groups. |",
        "| tests/unit_tests/pipeline_parallel/slackpipe_rma_lifecycle_stress.py | Reuse/recreate/exact-plan-order stress, per-rank CSV/JSON and plateau assertions. |",
        "| tests/unit_tests/pipeline_parallel/test_slackpipe_lifecycle.py | Compatible/incompatible keys, active teardown rejection, idempotent shutdown. |",
        "| tests/unit_tests/pipeline_parallel/test_slackpipe_heterogeneous.py | Parameterize synthetic architecture scale; test scale-specific manifest identities. |",
        "| tests/unit_tests/pipeline_parallel/slackpipe_perf_benchmark.py | Retain compatible RMA resources across modes, rotated order, strict FP32, setup-only measurements. |",
        "| tests/unit_tests/pipeline_parallel/slackpipe_heterogeneous_calibration.py | Diagnostic fixture scale/LR controls, scaled manifests and metadata. |",
        "| tools/slackpipe_scale_sweep.sh | Container-only calibration/solver/rotated benchmarks and same-plan transport controls. |",
        "| tools/slackpipe_analyze_scale_sweep.py | Reproducible tables, CIs, exact trace/layout/replay/hash checks, six plots and report. |",
        "",
        "No nested solver source or search behavior changed. No ordinary 1F1B/interleaved "
        "implementation changed in this milestone. Existing worktree changes were preserved; "
        "nothing was staged or committed. Full separate worktree stats (including earlier "
        "milestones, excluding untracked sources) are in `outer_diff_stat.txt` and "
        "`nested_diff_stat.txt`. Generated artifact recipes are in `COMMANDS.md`.",
    ]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("slackpipe_experiments/heterogeneous_scale_sweep")
    )
    analyze(parser.parse_args().root)
