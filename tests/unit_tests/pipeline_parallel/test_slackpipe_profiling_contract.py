# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Operation-scale residuals and explicit, single-attempt backend selection."""

import copy
import json
import subprocess
from pathlib import Path

import pytest

from megatron.core.pipeline_parallel.slackpipe.cost_profile import (
    aggregate_stage_costs,
    build_cost_profile,
    profile_fingerprint,
)
from megatron.core.pipeline_parallel.slackpipe.profile_integrity import validate_profile_integrity
from megatron.core.pipeline_parallel.slackpipe.stage_residual import (
    FitStructureError,
    ResidualPolicy,
    fit_with_residuals,
)
from tools.run_slackpipe_eval import Experiment
from tools.slackpipe_eval_args import argument_parser, normalize_args
from tools.slackpipe_profiling import collect_profiles, profiling_policy


def events(values, *, batches=2, scale=1):
    return [
        dict(
            iteration=i,
            microbatch=b,
            logical_stage=s,
            phase=phase + "_compute",
            elapsed_ms=value * scale * (1 if phase == "forward" else 2),
            calibration_group_id="group0",
            stage_layer_range=[s, s + 1],
        )
        for s in range(2)
        for i, per_mb in enumerate(values)
        for b, value in enumerate(per_mb)
        for phase in ("forward", "backward")
    ]


def profile(values, *, scale=1):
    return build_cost_profile(
        events=events(values, scale=scale),
        layer_split=[1, 1],
        model_config={},
        parallel_config=dict(pp=2, vpp=None, tp=1, dp=1, cp=1),
        num_microbatches=len(values[0]),
        iteration_start=0,
        iteration_end=len(values) - 1,
    )


def backend_profile(backend):
    from tests.unit_tests.pipeline_parallel.test_slackpipe_isolated_profile import fixture_profile

    p = profile([[1, 1]] * 3) if backend == "stage" else fixture_profile()
    p["quality"] = {"status": "failed"}
    p["cost_profile_hash"] = profile_fingerprint(p)
    return p


def test_zero_aggregate_residual_still_filters_raw_operation_once():
    p = profile([[1, 1], [1, 100], [1, 1]])
    for phase in ("forward", "backward"):
        report = p["residual_filter"]["phases"][phase]
        assert report["initial_residual_scaled_mad"] == 0
        assert report["candidate_excluded_count"] == report["applied_excluded_count"] == 2
        assert report["mask_passes"] == report["refit_attempts"] == 1
        assert report["initial_model"]["predicted"] == report["final_model"]["predicted"]
        row = p["observed_stages"][0][phase + "_diagnostics"]
        assert row["accepted_counts_by_iteration"] == [2, 1, 2]
        assert row["iteration_means_ms"] == [1 if phase == "forward" else 2] * 3
        assert len(row["raw_events"]) == 6
        assert all(r["sample_id"] for r in report["samples"])


def test_empty_iteration_is_recorded_not_zero_and_no_second_mask():
    p = profile([[1, 1], [100, 100], [1, 1]])
    d = p["observed_stages"][0]["forward_diagnostics"]
    assert d["iteration_means_ms"] == [1, None, 1]
    assert d["samples_ms"] == [2, None, 2]
    assert p["a_fwd"] == 1
    assert p["residual_filter"]["phases"]["forward"]["refit_attempts"] == 1


def test_strict_rho_and_mad_boundaries():
    p = profile([[1, 1], [1, 1.5], [1, 1]])
    report = p["residual_filter"]["phases"]["forward"]
    assert report["candidate_excluded_count"] == 0  # q == rho is retained.
    assert report["refit_attempts"] == 0


def test_strict_scaled_mad_boundary_is_not_excluded():
    rows = aggregate_stage_costs(
        events([[1, 1, 2, 2, 2, 3, 3, 4]]),
        layer_split=[1, 1],
        num_microbatches=8,
        iteration_start=0,
        iteration_end=0,
    )
    policy = ResidualPolicy(k=2 / 1.4826)
    _, _, report = fit_with_residuals(rows, lambda data, phase: {"predicted": [2, 2]}, policy)
    f = report["phases"]["forward"]
    extreme = f["samples"][7]
    assert abs(extreme["relative_residual"] - f["initial_residual_median"]) == (
        policy.k * f["initial_residual_scaled_mad"]
    )
    assert abs(extreme["relative_residual"]) > policy.rho
    assert not extreme["candidate_excluded"]


@pytest.mark.parametrize("reason", ["empty", "rank"])
def test_entire_phase_rollback_and_unexpected_errors_propagate(reason):
    values = [[1, 1], [1, 1], [1, 1]]
    rows = aggregate_stage_costs(
        events(values), layer_split=[1, 1], num_microbatches=2, iteration_start=0, iteration_end=2
    )
    # One required observation is far from the constrained model; all its samples
    # are candidates, while the other observation supplies a zero MAD majority.
    rows[1]["forward_diagnostics"]["raw_events"] = [
        dict(rows[1]["forward_diagnostics"]["raw_events"][0], elapsed_ms=100)
    ]
    calls = []

    def fit(data, phase):
        calls.append(phase)
        if reason == "rank" and calls.count(phase) == 2:
            raise FitStructureError("lost_required_rank")
        return {"predicted": [1, 1] if phase == "forward" else [2, 2]}

    if reason == "rank":
        rows[1]["forward_diagnostics"]["raw_events"].append(
            dict(rows[1]["forward_diagnostics"]["raw_events"][0], elapsed_ms=1)
        )
    models, final, report = fit_with_residuals(rows, fit)
    f = report["phases"]["forward"]
    assert f["status"] == "filter_not_applied"
    assert f["candidate_excluded_count"] == 1 and f["applied_excluded_count"] == 0
    assert final == rows and models["forward"] == f["initial_model"]
    assert f["reason"] == (
        "required_observation_has_no_samples" if reason == "empty" else "lost_required_rank"
    )

    def broken(data, phase):
        raise RuntimeError("programming error")

    with pytest.raises(RuntimeError, match="programming"):
        fit_with_residuals(rows, broken)


@pytest.mark.parametrize("batches", [1, 4])
@pytest.mark.parametrize("scale", [1, 1000])
def test_operation_cost_not_multiplied_by_microbatches_or_units(batches, scale):
    p = profile([[2] * batches] * 3, scale=scale)
    assert p["a_fwd"] == 2 * scale and p["a_bwd"] == 4 * scale
    assert p["units"] == "milliseconds"


@pytest.mark.parametrize("bad", ["missing", "duplicate", "nan", "index"])
def test_raw_integrity_precedes_any_fitting(bad):
    raw = events([[1, 1]] * 3)
    if bad == "missing":
        raw.pop()
    elif bad == "duplicate":
        raw.append(raw[0])
    elif bad == "nan":
        raw[0]["elapsed_ms"] = float("nan")
    else:
        raw[0]["microbatch"] = 9
    with pytest.raises(ValueError):
        aggregate_stage_costs(
            raw, layer_split=[1, 1], num_microbatches=2, iteration_start=0, iteration_end=2
        )


def args_for(tmp_path, *extra):
    return normalize_args(
        argument_parser().parse_args(
            [
                "--phase",
                "calibrate",
                "--model-config",
                "configs/slackpipe_eval/llama_8b.json",
                "--output",
                str(tmp_path),
                "--num-stages",
                "4",
                "--pp",
                "2",
                *extra,
            ]
        )
    )


@pytest.mark.parametrize(
    "mode,selected", [("both", "stage"), ("stage", "stage"), ("layer", "layer")]
)
def test_defaults_preserve_explicit_selection(tmp_path, mode, selected):
    args = args_for(tmp_path, "--profiling-method", mode)
    assert args.solver_profile == selected and not args.solver_profile_explicit
    explicit = args_for(tmp_path, "--solver-profile", "layer")
    assert explicit.solver_profile == "layer" and explicit.solver_profile_explicit
    with pytest.raises(ValueError, match="explicit artifact"):
        args_for(tmp_path, "--profiling-method", "stage", "--solver-profile", "layer")


@pytest.mark.parametrize("selected", ["stage", "layer"])
@pytest.mark.parametrize("failed", [None, "stage", "layer"])
def test_both_sequential_partial_no_retry_no_fallback(tmp_path, selected, failed):
    args = args_for(tmp_path, "--solver-profile", selected)
    calls = []

    def collect(backend, folder):
        calls.append(backend)
        folder.mkdir()
        (folder / "raw.json").write_text('{"preserved": true}')
        if backend == failed:
            raise RuntimeError("producer failed")
        p = backend_profile(backend)
        (folder / "cost_profile.json").write_text(json.dumps(p))

    if selected == failed:
        with pytest.raises(RuntimeError, match="no fallback"):
            collect_profiles(tmp_path, args, collect, validate_profile_integrity)
    else:
        result = collect_profiles(tmp_path, args, collect, validate_profile_integrity)
        assert Path(result["selected_profile"]).parent.parent.name == selected
        assert result["status"] == ("partial" if failed else "completed")
    assert calls == ["stage", "layer"]
    assert (tmp_path / "stage/worker/raw.json").exists()
    assert (tmp_path / "layer/worker/raw.json").exists()


def test_explicit_artifact_preserves_legacy_identity_and_bytes(tmp_path):
    p = profile([[1, 100], [1, 1], [1, 1]])
    p.pop("profiling_policy")
    p["quality"] = {"status": "rerun_required"}
    p["cost_profile_hash"] = profile_fingerprint(p)
    original = tmp_path / "legacy.json"
    original.write_text(json.dumps(p))
    args = args_for(tmp_path, "--profiling-method", "stage", "--stage-profile", str(original))
    result = collect_profiles(
        tmp_path / "import",
        args,
        lambda *unused: pytest.fail("Explicit artifact must not launch GPU collection"),
        validate_profile_integrity,
    )
    assert Path(result["selected_profile"]).read_bytes() == original.read_bytes()
    assert result["backends"]["stage"]["policy"]["policy_version"] == "legacy-unspecified"
    damaged = copy.deepcopy(p)
    damaged["a_fwd"] += 1
    with pytest.raises(ValueError, match="hash"):
        validate_profile_integrity(damaged)


@pytest.mark.parametrize("corruption,error", [("cost", "KeyError"), ("policy", "ValueError")])
def test_malformed_optional_backend_keeps_selected_profile(tmp_path, corruption, error):
    args = args_for(tmp_path)

    def collect(backend, folder):
        folder.mkdir()
        p = backend_profile(backend)
        if backend == "layer":
            if corruption == "cost":
                del p["layer_costs_us"]
            else:
                p["profiling_policy"] = []
            p["cost_profile_hash"] = profile_fingerprint(p)
        (folder / "cost_profile.json").write_text(json.dumps(p))

    result = collect_profiles(tmp_path, args, collect, validate_profile_integrity)
    assert result["status"] == "partial"
    assert result["backends"]["layer"]["error_type"] == error
    assert result["backends"]["stage"]["status"] == "completed"


def test_missing_explicit_optional_artifact_is_partial_not_implicit_collection(tmp_path):
    args = args_for(tmp_path, "--layer-profile", str(tmp_path / "missing.json"))
    calls = []

    def collect(backend, folder):
        calls.append(backend)
        folder.mkdir()
        (folder / "cost_profile.json").write_text(json.dumps(profile([[1, 1]] * 3)))

    result = collect_profiles(tmp_path, args, collect, validate_profile_integrity)
    assert result["status"] == "partial" and calls == ["stage"]
    assert result["backends"]["layer"]["error_type"] == "FileNotFoundError"


def test_stage_hook_topology_limit_is_explicit_before_gpu_start(tmp_path, monkeypatch):
    import torch

    from tools.slackpipe_eval_worker import run_worker

    args = args_for(
        tmp_path, "--phase", "env", "--method", "1f1b", "--pp", "1", "--num-stages", "1"
    )
    args.action = "calibrate"
    monkeypatch.setattr(torch.cuda, "set_device", lambda *unused: pytest.fail("No GPU launch"))
    with pytest.raises(ValueError, match="Stage calibration hooks"):
        run_worker(args)


@pytest.mark.parametrize("pp,n,b", [(1, 4, 4), (2, 2, 4), (2, 4, 3), (4, 4, 8), (4, 8, 6)])
@pytest.mark.parametrize("selected", ["stage", "layer"])
def test_collection_preflight_blocks_before_prerequisites(
    tmp_path, monkeypatch, pp, n, b, selected
):
    monkeypatch.setattr(
        Experiment, "_execute", lambda *a, **kw: pytest.fail("Must not launch any prerequisite")
    )
    options = [
        "--pp",
        str(pp),
        "--num-stages",
        str(n),
        "--microbatches",
        str(b),
        "--solver-profile",
        selected,
    ]
    with pytest.raises(ValueError, match="New stage profiling requires native interleaved"):
        Experiment(args_for(tmp_path, *options))
    # Also guard direct API users who start with a non-collection phase.
    exp = Experiment(args_for(tmp_path, "--phase", "env", *options))
    with pytest.raises(ValueError, match="New stage profiling"):
        exp.ensure("solve")
    assert not exp.completed and not list(tmp_path.iterdir())


@pytest.mark.parametrize("method", ["octopipe-tuning", "cpsat-no-hint", "slack-heuristic"])
def test_calibration_schedule_distinct_from_method_and_selection(tmp_path, monkeypatch, method):
    args = args_for(tmp_path, "--method", method, "--solver-profile", "layer")
    exp = Experiment(args)
    policy = profiling_policy(args)
    assert policy["backends"]["stage"]["calibration_schedule"] == "interleaved-1f1b"
    assert policy["backends"]["layer"]["calibration_schedule"] == "isolated-unit"
    before = (args.pp, args.logical_stages, args.microbatches, args.solver_profile, args.schedule)
    commands = []
    monkeypatch.setattr(exp, "_execute", lambda command, *unused: commands.append(command))
    exp._worker("calibrate", tmp_path / "stage", schedule="interleaved-1f1b")
    command = commands[0]
    assert command[command.index("--method") + 1] == "interleaved-1f1b"
    assert (
        args.pp,
        args.logical_stages,
        args.microbatches,
        args.solver_profile,
        args.schedule,
    ) == before


def test_existing_stage_import_does_not_require_collection_capability(tmp_path):
    from tools.slackpipe_eval_config import fingerprint, load_model, schedule_topology
    from tools.slackpipe_eval_worker import validate_profile

    original = tmp_path / "profile.json"
    args = args_for(
        tmp_path, "--pp", "1", "--profiling-method", "stage", "--stage-profile", str(original)
    )
    model = load_model(args.model_config)
    topology = schedule_topology(args.schedule, args.pp, args.logical_stages, args.microbatches)
    split = [model["num_layers"] // 4] * 4
    data = build_cost_profile(
        events=[
            dict(
                logical_stage=s,
                iteration=i,
                microbatch=b,
                phase=phase + "_compute",
                elapsed_ms=n * multiplier,
            )
            for s, n in enumerate(split)
            for i in range(3)
            for b in range(args.microbatches)
            for phase, multiplier in (("forward", 1), ("backward", 2))
        ],
        model_config=dict(
            model_config_hash=fingerprint(model),
            dtype=args.precision,
            sequence_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
        ),
        parallel_config={k: topology[k] for k in ("pp", "vpp", "tp", "dp", "cp")},
        layer_split=split,
        num_microbatches=args.microbatches,
        iteration_start=0,
        iteration_end=2,
    )
    data["cost_profile_hash"] = profile_fingerprint(data)
    original.write_text(json.dumps(data))
    result = collect_profiles(
        tmp_path / "import",
        args,
        lambda *unused: pytest.fail("An explicit artifact must not be re-collected"),
        lambda p: validate_profile(
            p, model, topology, args.precision, args.seq_length, args.micro_batch_size
        ),
    )
    assert result["status"] == "completed"
    assert Path(result["selected_profile"]).read_bytes() == original.read_bytes()


def test_execution_worker_has_no_collection_prerequisites(tmp_path):
    raw = argument_parser().parse_args(
        [
            "--phase",
            "benchmark",
            "--method",
            "slack-heuristic",
            "--pp",
            "1",
            "--num-stages",
            "4",
            "--model-config",
            "configs/slackpipe_eval/llama_8b.json",
            "--output",
            str(tmp_path),
        ]
    )
    normalized = normalize_args(copy.deepcopy(raw), worker=True)
    assert normalized.pp == 1 and normalized.logical_stages == 4
    with pytest.raises(ValueError, match="New stage profiling"):
        normalize_args(raw)  # A launcher would need a new stage profile.


@pytest.mark.parametrize("change", ["mask", "prediction", "policy", "cost", "raw"])
def test_rehashed_stage_profile_cannot_launder_strategy(change):
    p = profile([[1, 1], [100, 100], [1, 1]])
    if change == "mask":
        p["residual_filter"]["phases"]["forward"]["samples"][0]["applied_excluded"] = True
    elif change == "prediction":
        p["residual_filter"]["phases"]["forward"]["samples"][0]["prediction_ms"] *= 2
    elif change == "policy":
        p["profiling_policy"]["rho"] = 100
    elif change == "cost":
        p["a_fwd"] += 1
    else:
        p["raw_observed_stages"][0]["forward_diagnostics"]["raw_events"].pop()
    p["cost_profile_hash"] = profile_fingerprint(p)
    with pytest.raises(ValueError):
        validate_profile_integrity(p)


def test_phase_rollback_does_not_undo_other_phase_and_refit_errors_propagate():
    rows = aggregate_stage_costs(
        events([[1, 1], [1, 100], [1, 1]]),
        layer_split=[1, 1],
        num_microbatches=2,
        iteration_start=0,
        iteration_end=2,
    )
    calls = []

    def fit(data, phase):
        calls.append(phase)
        if phase == "forward" and calls.count(phase) == 2:
            raise FitStructureError("lost_rank")
        return {"predicted": [1, 1] if phase == "forward" else [2, 2]}

    models, final, report = fit_with_residuals(rows, fit)
    assert calls == ["forward", "forward", "backward", "backward"]
    assert models["forward"] == report["phases"]["forward"]["initial_model"]
    for before, after in zip(rows, final):
        assert after["forward_diagnostics"] == before["forward_diagnostics"]
        assert after["backward_diagnostics"]["accepted_counts_by_iteration"] == [2, 1, 2]
    assert report["phases"]["forward"]["applied_excluded_count"] == 0
    assert report["phases"]["backward"]["applied_excluded_count"] == 2
    calls.clear()

    def broken(data, phase):
        calls.append(phase)
        if len(calls) == 2:
            raise RuntimeError("unexpected refit failure")
        return {"predicted": [1, 1]}

    with pytest.raises(RuntimeError, match="unexpected refit"):
        fit_with_residuals(rows, broken)


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_residual_policy_requires_finite_positive_values(value):
    with pytest.raises(ValueError):
        ResidualPolicy(k=value)
    with pytest.raises(ValueError):
        ResidualPolicy(rho=value)


@pytest.mark.parametrize("backend", ["stage", "layer"])
def test_generated_profile_cpp_solver_independent_validator_python_parser(tmp_path, backend):
    from megatron.core.pipeline_parallel.slackpipe.plan import load_slackpipe_plan
    from megatron.core.pipeline_parallel.slackpipe.profile_integrity import publish_profile
    from tests.unit_tests.pipeline_parallel.test_slackpipe_isolated_profile import fixture_profile

    binary = Path("slackpipe/build/no-or/slackpipe_cli").resolve()
    if not binary.is_file():
        pytest.skip("Build the no-OR CLI to run exporter/parser integration")
    if backend == "stage":
        data = profile([[1, 1], [1, 100], [1, 1]])
    else:
        data = fixture_profile(kinds=("decoder", "decoder"))
    data["quality"] = {"status": "failed"}
    path = publish_profile(tmp_path / "cost_profile.json", data)
    plan = tmp_path / "plan.json"
    subprocess.run(
        [
            str(binary),
            "--algorithm",
            "slack-heuristic",
            "--B",
            "2",
            "--N",
            "2",
            "--J",
            "2",
            "--L",
            "2",
            "--ratio-num",
            "1",
            "--ratio-den",
            "1",
            "--communication",
            "0",
            "--time-limit",
            "1",
            "--num-workers",
            "1",
            "--cost-profile",
            str(path),
            "--output-prefix",
            str(tmp_path / "solver"),
            "--emit-plan",
            str(plan),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            str(binary),
            "--validate-result",
            str(tmp_path / "solver.json"),
            "--validation-output",
            str(tmp_path / "validation.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    parsed = load_slackpipe_plan(plan, pipeline_model_parallel_size=2)
    assert parsed.num_layers == parsed.num_stages == parsed.num_microbatches == 2
    result = json.loads((tmp_path / "solver.json").read_text())
    assert result["method_id"] == "slack-heuristic" and result["final_solution_available"]
    durations = result["duration_model_ticks"]
    assert result["cost_units"] == "microseconds"
    if backend == "stage":
        assert durations["forward_slope"] == [1000, 1000]
        assert durations["backward_slope"] == [2000, 2000]
    else:
        assert durations["prefix_forward"] == data["prefix_forward_us"]
        assert durations["prefix_backward"] == data["prefix_backward_us"]
        assert durations["role_forward_bias"] == [
            data["stage_role_bias_us"][role]["forward"] for role in ("first", "middle", "last")
        ]
