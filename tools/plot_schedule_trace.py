"""Publication-style GPU logical-envelope figure from rank-local compact JSON."""

import argparse
import json
from pathlib import Path

from megatron.core.pipeline_parallel.slackpipe.figure_trace import validate_compact_trace


def _traces(directory: Path) -> list[dict]:
    paths = list(directory.glob("rank*_trace.json")) + list(directory.rglob("*.compact.json"))
    return [json.loads(path.read_text()) for path in sorted(paths)]


def available_iterations(directory: Path, run_id=None, cycle=None) -> set[int]:
    traces = [
        t
        for t in _traces(directory)
        if run_id is None or t.get("collection", {}).get("run_id") == run_id
    ]
    groups = {}
    for t in traces:
        capture = t.get("collection", {})
        if cycle is not None and capture.get("cycle") != cycle:
            continue
        key = (capture.get("run_id"), capture.get("capture_id"), t["mode"], t["config"]["pp"])
        ranks = groups.setdefault(key, {})
        ranks.setdefault(t["rank"], set()).update(s["iteration"] for s in t["steps"])
    available = set()
    for key, ranks in groups.items():
        if sorted(ranks) == list(range(key[-1])):
            available.update(set.intersection(*ranks.values()))
    return available


def load_panel(directory: Path, iteration: int, run_id=None, cycle=None) -> dict:
    traces = [
        t
        for t in _traces(directory)
        if (run_id is None or t.get("collection", {}).get("run_id") == run_id)
        and (cycle is None or t.get("collection", {}).get("cycle") == cycle)
        and any(s["iteration"] == iteration for s in t["steps"])
    ]
    if not traces:
        raise ValueError(f"No compact traces in {directory}")
    traces.sort(key=lambda trace: trace["rank"])
    if [t["rank"] for t in traces] != list(range(traces[0]["config"]["pp"])):
        raise ValueError("Missing or duplicate ranks")
    if len({t["config"]["host"] for t in traces}) != 1:
        raise ValueError("Cross-host clock alignment is unsupported")
    if any(t.get("collection") != traces[0].get("collection") for t in traces[1:]):
        # Environment GPU UUID legitimately differs by rank; compare capture identity/options.
        fields = ("run_id", "capture_id", "cycle", "profiler", "active_iterations")
        if any(
            any(
                t.get("collection", {}).get(k) != traces[0].get("collection", {}).get(k)
                for k in fields
            )
            for t in traces[1:]
        ):
            raise ValueError("Mixed run/capture/config across ranks")
        env_keys = ("host", "torch", "cuda", "nccl", "allocator_backend", "allocator_environment")
        for key in env_keys:
            first = traces[0].get("collection", {}).get("environment", {}).get(key)
            if any(
                t.get("collection", {}).get("environment", {}).get(key) != first for t in traces[1:]
            ):
                raise ValueError(f"Rank environment differs: {key}")
    if any(
        t["config"] != traces[0]["config"]
        or t["mode"] != traces[0]["mode"]
        or t["transport"] != traces[0]["transport"]
        for t in traces[1:]
    ):
        raise ValueError("Rank model/method metadata differs")

    def ns(trace, record, key):
        return trace["base_time_ns"] + round(record[key] * 1000)

    steps = []
    for trace in traces:
        validate_compact_trace(trace)
        selected = [step for step in trace["steps"] if step["iteration"] == iteration]
        if len(selected) != 1:
            raise ValueError(f"Missing selected iteration {iteration}")
        steps.append(selected[0])
    start = min(ns(trace, step, "start_us") for trace, step in zip(traces, steps))
    end = max(ns(trace, step, "end_us") for trace, step in zip(traces, steps))
    event_end = max(
        ns(trace, step, "start_us") + round(step["cuda_elapsed_ms"] * 1e6)
        for trace, step in zip(traces, steps)
    )
    rows, activities, allocations = [], [], []
    for trace in traces:
        for record in trace["records"]:
            if record["iteration"] == iteration:
                rows.append(
                    dict(
                        **record,
                        start_ms=(ns(trace, record, "start_us") - start) / 1e6,
                        end_ms=(ns(trace, record, "end_us") - start) / 1e6,
                    )
                )
        for activity in trace.get("gpu_activities", []):
            if activity["iteration"] == iteration:
                activities.append(
                    dict(
                        **activity,
                        rank=trace["rank"],
                        start_ms=(ns(trace, activity, "start_us") - start) / 1e6,
                        end_ms=(ns(trace, activity, "end_us") - start) / 1e6,
                    )
                )
        for call in trace.get("cuda_allocation_calls", []):
            if call["iteration"] == iteration:
                offset = trace["base_time_ns"] + round(call["ts"] * 1000) - start
                allocations.append(
                    dict(
                        **call,
                        rank=trace["rank"],
                        start_ms=offset / 1e6,
                        end_ms=offset / 1e6 + call["dur"] / 1000,
                    )
                )
    return dict(
        duration_ms=(end - start) / 1e6,
        rank_count=len(traces),
        records=rows,
        gpu_activities=activities,
        cuda_allocation_calls=allocations,
        collection=traces[0].get("collection"),
        iteration=iteration,
        logical_ops=sum(r["kind"] in ("F", "B") for r in rows),
        config=traces[0]["config"],
        steps=steps,
        origin_ns=start,
        end_ns=end,
        mode=traces[0]["mode"],
        cuda_event_panel_estimate_ms=(event_end - start) / 1e6,
        cuda_event_panel_discrepancy_ms=abs(event_end - end) / 1e6,
        transport=traces[0]["transport"],
    )


def plot_time_bounds(panels: list[dict], detail: bool) -> tuple[float, float]:
    """Keep CPU allocations before/after the GPU envelope visible without rescaling."""
    left, right = 0.0, max(p["duration_ms"] for p in panels)
    if detail:
        for panel in panels:
            for event in panel["cuda_allocation_calls"]:
                left = min(left, event["start_ms"])
                right = max(right, event["end_ms"])
    return left, right


def render(
    panels: list[dict],
    *,
    png: Path,
    pdf: Path,
    title: str,
    caption: str,
    color_mode: str,
    annotate_saved: bool = False,
    detail: bool = False,
) -> dict:
    from tools.slackpipe_eval_config import METHODS

    if not panels or len({p["mode"] for p in panels}) != len(panels):
        raise ValueError("Expected a nonempty set of distinct method panels")
    if any(p["mode"] not in METHODS for p in panels):
        raise ValueError(
            "Historical method labels require the historical plotter; not current methods"
        )
    baseline = panels[0]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    fields = (
        "pp",
        "num_stages",
        "num_layers",
        "num_microbatches",
        "hidden_size",
        "heads",
        "seq_length",
        "micro_batch_size",
        "vocab_size",
        "seed",
        "learning_rate",
        "dtype",
        "tf32",
        "dropout",
        "tp",
        "dp",
        "cp",
        "model",
        "model_sha256",
    )
    for panel in panels:
        if any(panel["config"][k] != baseline["config"][k] for k in fields if k != "num_stages"):
            raise ValueError("Panels use different model/data settings")
        if panel["iteration"] != baseline["iteration"]:
            raise ValueError("Panels must show the same global iteration")
        if panel["mode"] == "1f1b" and panel["config"]["num_stages"] != panel["config"]["pp"]:
            raise ValueError("Native 1F1B must have N=PP")
        if bool(panel["collection"]) != bool(baseline["collection"]):
            raise ValueError("Cannot mix collection provenance versions")
        if panel["collection"]:
            for key in ("profiler", "active_iterations", "cycle"):
                if panel["collection"][key] != baseline["collection"][key]:
                    raise ValueError(f"Capture policy differs: {key}")
            for key in ("allocator_environment", "allocator_backend", "torch", "cuda", "nccl"):
                if (
                    panel["collection"]["environment"][key]
                    != baseline["collection"]["environment"][key]
                ):
                    raise ValueError(f"Environment differs: {key}")
    optimized_n = {p["config"]["num_stages"] for p in panels if p["mode"] != "1f1b"}
    if len(optimized_n) > 1:
        raise ValueError("Plan/interleaved methods must use the same explicit N")
    titles = [METHODS[p["mode"]].display_name for p in panels]
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42}
    )
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(11, 2.5 * len(panels) + 0.45 * (baseline["rank_count"] - 2)),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]
    left_bound, right_bound = plot_time_bounds(panels, detail)
    padding = max(0.001, (right_bound - left_bound) * 0.025)
    for panel_index, (axis, panel) in enumerate(zip(axes, panels)):
        n = panel["rank_count"]
        for record in panel["records"]:
            kind = record["kind"]
            if kind in ("F", "B"):
                fraction = record["microbatch"] / max(1, panel["config"]["num_microbatches"] - 1)
                level = 0.42 + fraction * 0.40 if color_mode == "microbatch" else 0.65
                color = plt.get_cmap("Blues" if kind == "F" else "Oranges")(level)
            else:
                color = "#858b91" if kind == "optimizer" else "#54a88b"
            width = record["end_ms"] - record["start_ms"]
            axis.add_patch(
                Rectangle(
                    (record["start_ms"], n - 1 - record["rank"] - 0.30),
                    width,
                    0.6,
                    facecolor=color,
                    alpha=0.15 if detail else 1.0,
                    edgecolor="white",
                    linewidth=0.35,
                )
            )
        if detail:
            if not panel["gpu_activities"]:
                raise ValueError("Detailed plots require figure_trace.v2 activity records")
            for activity in panel["gpu_activities"] + panel["cuda_allocation_calls"]:
                allocation = "cat" in activity
                communication = activity.get("communication", False)
                offset = -0.24 if allocation else (0 if communication else 0.24)
                color = "#c23b40" if allocation else ("#238b65" if communication else "#3579ae")
                axis.add_patch(
                    Rectangle(
                        (activity["start_ms"], n - 1 - activity["rank"] + offset - 0.08),
                        activity["end_ms"] - activity["start_ms"],
                        0.16,
                        facecolor=color,
                        edgecolor="none",
                    )
                )
        axis.set_yticks(list(range(n)), [f"GPU {rank}" for rank in reversed(range(n))])
        axis.set_ylim(-1.2, n - 0.3)
        axis.set_xlim(left_bound - padding, right_bound + padding)
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)
        axis.tick_params(axis="y", length=0)
        axis.annotate(
            "",
            xy=(panel["duration_ms"], -0.65),
            xytext=(0, -0.65),
            arrowprops=dict(arrowstyle="<->", linestyle="--", color="#303030", linewidth=1),
        )
        axis.text(
            panel["duration_ms"] / 2,
            -0.86,
            f"{panel['duration_ms']:.2f} ms",
            ha="center",
            va="top",
            fontsize=10,
        )
        axis.set_title(
            f"({chr(97 + panel_index)}) {titles[panel_index]}: PP={panel['config']['pp']}, N={panel['config']['num_stages']}",
            loc="left",
            fontsize=11,
            pad=9,
        )
    saved = baseline["duration_ms"] - panels[-1]["duration_ms"]
    if annotate_saved and len(panels) > 1:
        axes[-1].text(
            0.99,
            1.02,
            f"Profiled span difference from first panel: {saved:+.2f} ms",
            transform=axes[-1].transAxes,
            ha="right",
            fontsize=9,
        )
    axes[-1].set_xlabel("Time from panel step start (ms)")
    fig.suptitle(title, fontsize=12, y=0.98)
    fig.legend(
        handles=(
            [
                Patch(color="#3579ae", label="GPU activity"),
                Patch(color="#238b65", label="NCCL activity"),
                Patch(color="#c23b40", label="CPU allocation API"),
            ]
            if detail
            else [
                Patch(color=plt.get_cmap("Blues")(0.65), label="Forward"),
                Patch(color=plt.get_cmap("Oranges")(0.65), label="Backward"),
                Patch(color="#858b91", label="Optimizer"),
            ]
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=3,
        frameon=False,
    )
    fig.text(0.5, 0.015, caption, ha="center", va="bottom", fontsize=8)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.83, bottom=0.13, hspace=0.56)
    for path in (png, pdf):
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return dict(
        schema_version="slackpipe.figure_report.v2",
        panels=panels,
        profiled_span_difference_ms=saved,
        plot_time_bounds_ms=[left_bound, right_bound],
        png=str(png),
        pdf=str(pdf),
        title=title,
        caption=caption,
        timing="GPU activity envelopes correlated to profiler ranges; step spans include unplotted work",
        warning="One profiled step; profiler perturbation means this is not benchmark evidence",
    )


def main():
    from tools.slackpipe_eval_config import METHODS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel", action="append", required=True, help="METHOD=trace-directory; repeat per method"
    )
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--cycle", type=int, default=0)
    parser.add_argument("--detail", action="store_true")
    parser.add_argument("--title", default="Pipeline schedule traces")
    parser.add_argument(
        "--caption",
        default="Profiled GPU envelopes; gaps are not guaranteed hardware idle. Not benchmark evidence.",
    )
    parser.add_argument("--color-mode", choices=("direction", "microbatch"), default="microbatch")
    args = parser.parse_args()
    directories = dict(item.split("=", 1) for item in args.panel)
    if set(directories) - METHODS.keys() or len(directories) != len(args.panel):
        parser.error("Unknown or duplicate method")
    candidates = sorted(
        set.intersection(
            *(available_iterations(Path(path), cycle=args.cycle) for path in directories.values())
        )
    )
    if not candidates:
        parser.error("No common complete-rank global iteration")
    iteration = args.iteration if args.iteration is not None else candidates[0]
    if iteration not in candidates:
        parser.error("Requested iteration is not present on all ranks/captures")
    panels = [load_panel(Path(path), iteration, cycle=args.cycle) for path in directories.values()]
    if any(panel["mode"] != method for panel, method in zip(panels, directories)):
        parser.error("Panel method provenance does not match its label")
    report = render(
        panels,
        png=args.output_png,
        pdf=args.output_pdf,
        title=args.title,
        caption=args.caption,
        color_mode=args.color_mode,
        detail=args.detail,
    )
    report["selection"] = dict(
        iteration=iteration,
        candidates=candidates,
        rule=(
            "explicit"
            if args.iteration is not None
            else "earliest common complete-rank global iteration"
        ),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
