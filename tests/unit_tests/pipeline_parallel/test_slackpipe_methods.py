# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Five-method contracts, not five names for the same hinted optimizer."""
import ast
from pathlib import Path

import pytest

from tools.slackpipe_eval_config import (
    METHODS,
    PLAN_SCHEDULES,
    execution_mode,
    schedule_topology,
    solver_arguments,
)


def test_registry_and_native_runtime_routing():
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

    assert tuple(METHODS) == (
        "1f1b",
        "interleaved-1f1b",
        "octopipe-tuning",
        "cpsat-no-hint",
        "slack-heuristic",
    )
    assert (
        get_forward_backward_func(pp_size=2, vp_size=None).__name__
        == "forward_backward_pipelining_without_interleaving"
    )
    assert (
        get_forward_backward_func(pp_size=2, vp_size=2).__name__
        == "forward_backward_pipelining_with_interleaving"
    )
    for method in PLAN_SCHEDULES:
        assert execution_mode(method) == "slackpipe"
        assert METHODS[method].executor == "slackpipe"
        assert schedule_topology(method, 2, 4, 4)["num_stages"] == 4
    assert METHODS["cpsat-no-hint"].requires_ortools
    assert not METHODS["slack-heuristic"].requires_ortools


@pytest.mark.parametrize("method", PLAN_SCHEDULES)
def test_budget_and_solver_threads(method):
    flags = solver_arguments(method, 300)
    assert flags == ["--algorithm", method, "--time-limit", "300", "--num-workers", "1"]
    if method == "cpsat-no-hint":
        assert solver_arguments(method, 300, 4)[-1] == "4"
        assert "--activation-cap-units" in solver_arguments(method, 300, 1, [8, 8])
    else:
        with pytest.raises(ValueError):
            solver_arguments(method, 300, 2)
        with pytest.raises(ValueError):
            solver_arguments(method, 300, 1, [8])


@pytest.mark.parametrize(
    "method",
    (
        "slackpipe",
        "slackpipe-refined",
        "optimized_interleaved",
        "interleaved",
        "1f1b-interleave",
        "octopipe",
    ),
)
def test_retired_names_fail_closed(method):
    with pytest.raises(ValueError, match="retired"):
        execution_mode(method)


def test_topology_and_budget_rejections():
    for method, pp, n, b in (
        ("1f1b", 2, 4, 4),
        ("interleaved-1f1b", 1, 4, 4),
        ("interleaved-1f1b", 2, 4, 3),
        ("slack-heuristic", 2, 3, 4),
        ("slack-heuristic", 2, None, 4),
    ):
        with pytest.raises(ValueError):
            schedule_topology(method, pp, n, b)
    for budget in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            solver_arguments("cpsat-no-hint", budget)


def test_public_cli_names_and_unspecified_solver_profile(tmp_path):
    from tools.slackpipe_eval_args import argument_parser, normalize_args

    config = Path(__file__).resolve().parents[3] / "configs/slackpipe_eval/llama_4b.json"
    flags = [
        "--phase",
        "benchmark",
        "--method",
        "1f1b",
        "--num-stages",
        "2",
        "--pp",
        "2",
        "--model-config",
        str(config),
        "--output",
        str(tmp_path),
    ]
    args = argument_parser().parse_args(flags)
    assert (args.stage, args.schedule, args.logical_stages) == ("benchmark", "1f1b", 2)
    assert args.profiling_method == "both" and args.solver_profile is None
    normalized = normalize_args(args)
    assert normalized.solver_profile == "stage"
    assert normalized.solver_profile_explicit is False
    for retired in (
        ["--schedule", "1f1b"],
        ["--logical-stages", "2"],
        ["--calibration-estimator", "existing-stage-wall-time-v1"],
        ["--quality-cv", "0.1"],
        ["--isolated-profile-max-cv", "0.15"],
    ):
        with pytest.raises(SystemExit):
            argument_parser().parse_args(flags + retired)


def test_normal_execution_chain_has_no_test_or_retired_imports():
    root = Path(__file__).resolve().parents[3]
    modules = (
        "run_slackpipe_eval",
        "slackpipe_eval_worker",
        "slackpipe_training",
        "slackpipe_model_support",
        "slackpipe_eval_args",
        "slackpipe_eval_common",
        "slackpipe_eval_receipts",
        "slackpipe_eval_tables",
        "slackpipe_profiling",
        "slackpipe_profile_integrity",
        "run_slackpipe_real_system_campaign",
    )
    retired = (
        "slackpipe_nemotron_worker",
        "run_slackpipe_nemotron_h8b_pp4",
        "slackpipe_experiment_driver",
        "slackpipe_heterogeneous_experiment",
        "slackpipe_profile_quality",
    )
    for module in modules:
        for node in ast.walk(ast.parse((root / "tools" / f"{module}.py").read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("tests.")
                assert not any(name in node.module for name in retired)
                if module != "run_slackpipe_real_system_campaign":
                    assert node.module != "tools.run_slackpipe_eval"
