# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real orchestration/receipt IO with synthetic workers, never an 8B allocation."""

import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import run_slackpipe_real_system_campaign as campaign
from tools.run_slackpipe_eval import Experiment, argument_parser
from tools.slackpipe_eval_common import digest, write_json
from tools.slackpipe_eval_config import COMPARISON_SCHEDULES, fingerprint, parameter_breakdown
from tools.slackpipe_eval_receipts import assess_receipt

EMPTY_DIFF = __import__("hashlib").sha256(b"").hexdigest()

CONFIG = Path(__file__).resolve().parents[3] / "configs/slackpipe_eval/llama_8b.json"


def args_for(root, stage="solve", **updates):
    args = argument_parser().parse_args(
        [
            "--phase",
            stage,
            "--num-stages",
            "8",
            "--model-config",
            str(CONFIG),
            "--output",
            str(root),
            "--resume",
            "--profiling-method",
            "stage",
        ]
    )
    for key, value in updates.items():
        setattr(args, key, value)
    if args.schedule == "1f1b" and "logical_stages" not in updates:
        args.logical_stages = args.pp
    return args


@pytest.fixture
def synthetic(monkeypatch):
    calls = Counter()

    def base(self):
        a = self.args
        return dict(
            model_config_hash=fingerprint(self.model),
            topology=self.topology,
            precision=a.precision,
            seq_length=a.seq_length,
            micro_batch_size=a.micro_batch_size,
            transport=a.transport,
            source=dict(commit="fixture", diff=EMPTY_DIFF),
            environment={"fixture": True},
        )

    monkeypatch.setattr(Experiment, "_base_context", base)

    def execute(self, command, directory, name):
        calls[name] += 1
        if name == "correctness":
            (directory / "correctness.xml").write_text("<testsuite><testcase/></testsuite>")
        elif name == "solve":
            a = self.args
            n, b, pp = self.topology["num_stages"], a.microbatches, a.pp
            cuts = [self.model["num_layers"] * s // n for s in range(n + 1)]
            ops = [[] for _ in range(pp)]
            for mb in range(b):
                for kind, stages in (("F", range(n)), ("B", reversed(range(n)))):
                    for s in stages:
                        ops[s % pp].append(dict(kind=kind, microbatch=mb, stage=s))
            write_json(
                directory / "slackpipe.plan.json",
                dict(
                    schema_version="slackpipe.plan.v1",
                    num_microbatches=b,
                    num_stages=n,
                    num_workers=pp,
                    num_layers=self.model["num_layers"],
                    layer_split=[y - x for x, y in zip(cuts, cuts[1:])],
                    stage_to_worker=[s % pp for s in range(n)],
                    operations=ops,
                    cost_model=dict(path=str(self.output / self.completed["calibrate"]["profile"])),
                ),
            )
            write_json(
                directory / "solver.json",
                dict(
                    schema_version=2,
                    method_id=a.schedule,
                    final_solution_available=True,
                    solver_status_raw="FEASIBLE",
                ),
            )
        elif name == "validate":
            write_json(directory / "independent_validation.json", dict(passed=True))
        else:
            pytest.fail(f"unexpected command: {command}")

    def worker(self, stage, directory, **kwargs):
        calls[f"{self.args.schedule}.{stage}"] += 1
        calls[(self.args.schedule, stage, self.args.run_index)] += 1
        directory.mkdir(parents=True)
        a = self.args
        if stage == "calibrate":
            assert kwargs["schedule"] == (
                "interleaved-1f1b" if kwargs["backend"] == "stage" else self.args.schedule
            )
            from megatron.core.pipeline_parallel.slackpipe.profile_integrity import publish_profile
            from tests.unit_tests.pipeline_parallel.test_slackpipe_profile_quality import (
                stable_profile,
            )

            profile = dict(
                schema_version="slackpipe.cost_profile.v1",
                model_config=dict(
                    model_config_hash=fingerprint(self.model),
                    dtype=a.precision,
                    sequence_length=a.seq_length,
                    micro_batch_size=a.micro_batch_size,
                ),
                parallel_config={k: self.topology[k] for k in ("pp", "vpp", "tp", "dp", "cp")},
            )
            profile.update(stable_profile())
            n = self.topology["num_stages"]
            cuts = [self.model["num_layers"] * s // n for s in range(n + 1)]
            profile["calibration_partition"] = [y - x for x, y in zip(cuts, cuts[1:])]
            profile["bias_fwd"] = profile["bias_bwd"] = [0.0] * n
            profile["measurement_parallel_config"] = profile["parallel_config"]
            profile["target_parallel_config"] = profile["parallel_config"]
            if kwargs.get("backend") == "layer":
                from megatron.core.pipeline_parallel.slackpipe.isolated_profile import (
                    execution_signature,
                )
                from megatron.core.pipeline_parallel.slackpipe.manifest import build_model_manifest
                from tests.unit_tests.pipeline_parallel.test_slackpipe_isolated_profile import (
                    fixture_profile,
                )
                from tools.slackpipe_eval_config import transformer_config

                isolated = fixture_profile(
                    warmups=a.isolated_profile_warmups,
                    iterations=a.isolated_profile_iterations,
                    manifest=build_model_manifest(
                        transformer_config(self.model, self.topology, a.precision)
                    ),
                    execution=execution_signature(
                        self.model,
                        seq_length=a.seq_length,
                        micro_batch_size=a.micro_batch_size,
                        precision=a.precision,
                    ),
                )
                isolated.update(
                    model_config=profile["model_config"],
                    parallel_config=profile["parallel_config"],
                    target_parallel_config=profile["parallel_config"],
                )
                from megatron.core.pipeline_parallel.slackpipe.cost_profile import (
                    profile_fingerprint,
                )

                isolated["cost_profile_hash"] = profile_fingerprint(isolated)
                write_json(directory / "cost_profile.json", isolated)
                return
            from megatron.core.pipeline_parallel.slackpipe.cost_profile import build_cost_profile
            from megatron.core.pipeline_parallel.slackpipe.stage_residual import (
                residual_policy_from_args,
            )

            profile = build_cost_profile(
                events=[
                    dict(
                        logical_stage=s,
                        iteration=i,
                        microbatch=b,
                        phase=phase + "_compute",
                        elapsed_ms=(end - begin) * factor,
                        calibration_group_id="fixture",
                    )
                    for s, (begin, end) in enumerate(zip(cuts, cuts[1:]))
                    for i in range(
                        a.calibration_warmups, a.calibration_warmups + a.calibration_iterations
                    )
                    for b in range(a.microbatches)
                    for phase, factor in (("forward", 1), ("backward", 2))
                ],
                model_config=profile["model_config"],
                parallel_config=profile["parallel_config"],
                layer_split=profile["calibration_partition"],
                num_microbatches=a.microbatches,
                iteration_start=a.calibration_warmups,
                iteration_end=a.calibration_warmups + a.calibration_iterations - 1,
                residual_policy=residual_policy_from_args(a),
            )
            publish_profile(directory / "cost_profile.json", profile)
        else:
            config = dict(
                self.topology,
                model_config_hash=fingerprint(self.model),
                exact_parameter_count=parameter_breakdown(self.model)["exact_parameter_count"],
                micro_batch_size=a.micro_batch_size,
                global_batch_size=a.microbatches * a.micro_batch_size,
                seq_length=a.seq_length,
                precision=a.precision,
                seed=a.seed,
                learning_rate=a.learning_rate,
                tf32=False,
                dropout=0,
                schedule=a.schedule,
            )
            for rank in range(a.pp):
                write_json(
                    directory / f"result.rank{rank}.json",
                    dict(
                        rank=rank,
                        run_id=directory.name,
                        method=a.schedule,
                        config=config,
                        options=dict(warmup=a.warmups, iterations=a.iterations),
                        samples=[
                            dict(iteration=a.warmups + i, cuda_elapsed_ms=10 + rank + i)
                            for i in range(a.iterations)
                        ],
                        continuous_wall_ms=100 * a.iterations,
                        memory_boundaries={
                            "end": dict(peak_allocated_bytes=100, peak_reserved_bytes=200)
                        },
                    ),
                )

    monkeypatch.setattr(Experiment, "_execute", execute)
    monkeypatch.setattr(Experiment, "_worker", worker)
    import torch

    # Synthetic orchestration represents a capable node, not a zero-GPU bypass.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda i: (10**12, 10**12))
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda i: SimpleNamespace(name="fixture", uuid=str(i))
    )
    monkeypatch.setattr(
        "tools.run_slackpipe_eval.memory_preflight", lambda *a: dict(status="passed")
    )
    monkeypatch.setattr(campaign, "plot_campaign", lambda *a: None)
    return calls


def execute(root, stage="trace", **kwargs):
    exp = Experiment(args_for(root, stage, **kwargs))
    assert exp.execute()["status"] == "passed"
    return exp


def directories(exp):
    return {k: r["directory"] for k, r in exp.completed.items()}


def test_exact_standalone_three_campaign_one_reproduction(tmp_path, synthetic):
    first = execute(tmp_path, "solve", repetitions=3, warmups=5, iterations=50)
    before = synthetic.copy()
    second = execute(tmp_path, "solve", repetitions=1, run_index=0)
    assert directories(first) == directories(second)
    assert synthetic == before


@pytest.mark.parametrize(
    "field,value,changed",
    [
        ("iterations", 100, {"benchmark"}),
        ("warmups", 10, {"benchmark"}),
        ("repetitions", 5, {"benchmark"}),
        ("calibration_iterations", 20, {"calibrate", "solve", "smoke", "benchmark", "trace"}),
        ("calibration_warmups", 8, {"calibrate", "solve", "smoke", "benchmark", "trace"}),
        ("stage_residual_k", 12, {"calibrate", "solve", "smoke", "benchmark", "trace"}),
        ("stage_residual_rho", 0.8, {"calibrate", "solve", "smoke", "benchmark", "trace"}),
        ("solver_seconds", 600, {"solve", "smoke", "benchmark", "trace"}),
        ("profiler_active", 4, {"trace"}),
        ("trace_warmups", 9, {"trace"}),
    ],
)
def test_stage_invalidation(tmp_path, synthetic, field, value, changed):
    first = execute(tmp_path, "full")
    old = directories(first)
    second = execute(tmp_path, "full", **{field: value})
    new = directories(second)
    assert {s for s in old if old[s] != new[s]} == changed
    assert all((tmp_path / p).is_dir() for p in old.values())
    assert list((tmp_path / "receipts/history").glob("*.json"))


@pytest.mark.parametrize("systemic", [False, True])
@pytest.mark.parametrize("group_level", [False, True])
def test_raw_spikes_campaign_continuation(tmp_path, synthetic, monkeypatch, systemic, group_level):
    from megatron.core.pipeline_parallel.slackpipe.profile_integrity import publish_profile
    from megatron.core.pipeline_parallel.slackpipe.stage_residual import residual_policy_from_args
    from tests.unit_tests.pipeline_parallel.test_slackpipe_group_quality import group_profile
    from tests.unit_tests.pipeline_parallel.test_slackpipe_outliers import raw_profile

    original = Experiment._worker
    collections = []

    def worker(self, stage, directory, **kwargs):
        if stage != "calibrate":
            return original(self, stage, directory, **kwargs)
        directory.mkdir(parents=True)
        collections.append(kwargs["backend"])
        spikes = [
            (4, "forward", 13, 356),
            (2, "forward", 50, 313),
            (5, "forward", 57, 306),
            (3, "forward", 98, 286),
        ]
        if systemic:
            spikes.append((4, "forward", 14, 356))
        policy = residual_policy_from_args(self.args)
        if group_level:
            profile = group_profile(
                directory,
                [14, 16, 18, 20, 22] if systemic else [19, 19.2, 18.9, 19.1, 13.5],
                policy=policy,
                four_stages=True,
            )
        else:
            profile, events = raw_profile(directory, spikes, policy=policy)
            write_json(directory / "raw.json", events)
        profile["model_config"] = dict(
            model_config_hash=fingerprint(self.model),
            dtype=self.args.precision,
            sequence_length=self.args.seq_length,
            micro_batch_size=self.args.micro_batch_size,
        )
        profile["parallel_config"] = {k: self.topology[k] for k in ("pp", "vpp", "tp", "dp", "cp")}
        profile["measurement_parallel_config"] = profile["parallel_config"]
        profile["target_parallel_config"] = profile["parallel_config"]
        if profile["schema_version"] == "slackpipe.cost_profile.v2":
            from megatron.core.pipeline_parallel.slackpipe.manifest import build_model_manifest
            from tools.slackpipe_eval_config import transformer_config

            profile["model_manifest_hash"] = build_model_manifest(
                transformer_config(self.model, self.topology, self.args.precision)
            )["manifest_hash"]
            for key in ("raw_observed_stages", "observed_stages"):
                for row in profile[key]:
                    row["model_manifest_hash"] = profile["model_manifest_hash"]
        profile["quality"] = {"status": "failed"}  # Historical metadata is not an admission rule.
        publish_profile(directory / "cost_profile.json", profile)

    monkeypatch.setattr(Experiment, "_worker", worker)
    from tests.unit_tests.pipeline_parallel.test_slackpipe_eval import tiny_model

    model = tiny_model("llama")
    model["num_layers"] = 8
    config = tmp_path / "fixture-model.json"
    write_json(config, model)
    exp = Experiment(
        args_for(
            tmp_path,
            "benchmark",
            model_config=config,
            logical_stages=4 if group_level else 8,
            pp=2 if group_level else 4,
        )
    )
    assert exp.execute()["status"] == "passed"
    assert collections == ["stage"]
    assert synthetic["solve"] == 1
    assert synthetic["slack-heuristic.benchmark"] == exp.args.repetitions
    profile = json.loads((tmp_path / exp.completed["calibrate"]["profile"]).read_text())
    assert profile["quality"]["status"] == "failed"
    if group_level:
        assert "group_filter" not in profile
    else:
        assert profile["residual_filter"]["phases"]["forward"]["applied_excluded_count"] == (
            5 if systemic else 4
        )


@pytest.mark.parametrize("schedule", ["1f1b", "interleaved-1f1b"])
def test_native_benchmark_ignores_calibration_solver_trace(tmp_path, synthetic, schedule):
    first = execute(tmp_path, "benchmark", schedule=schedule, profiling_method="both")
    before = synthetic.copy()
    second = execute(
        tmp_path,
        "benchmark",
        schedule=schedule,
        profiling_method="both",
        calibration_iterations=20,
        calibration_warmups=12,
        solver_seconds=600,
        profiler_active=4,
    )
    assert directories(first) == directories(second)
    assert before == synthetic
    assert "solve" not in second.completed
    assert "calibrate" not in second.completed
    assert synthetic[f"{schedule}.calibrate"] == synthetic["solve"] == 0


def test_campaign_preflights_all_methods_before_first_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "campaign",
            "--families",
            "llama",
            "--sizes",
            "8b",
            "--methods",
            "1f1b,slack-heuristic",
            "--pp",
            "4",
            "--num-stages",
            "4",
            "--output",
            str(tmp_path),
        ],
    )
    monkeypatch.setattr(
        Experiment, "execute", lambda *args: pytest.fail("No method may launch yet")
    )
    with pytest.raises(ValueError, match="New stage profiling"):
        campaign.main()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "field,value",
    [
        ("precision", "fp32"),
        ("seq_length", 512),
        ("pp", 2),
        ("microbatches", 16),
        ("logical_stages", 12),
        ("micro_batch_size", 2),
        ("transport", "nccl-rma"),
    ],
)
def test_real_inputs_remain_bound(tmp_path, synthetic, field, value):
    first = Experiment(args_for(tmp_path))._context("env", {})
    second = Experiment(args_for(tmp_path, **{field: value}))._context("env", {})
    assert fingerprint(first) != fingerprint(second)


def test_schedule_namespaces_and_shared_native_calibration(tmp_path, synthetic):
    inter = execute(tmp_path, "calibrate", schedule="interleaved-1f1b")
    before = synthetic.copy()
    slack = execute(tmp_path, "solve", schedule="slack-heuristic")
    assert all(directories(inter)[stage] == directories(slack)[stage] for stage in inter.completed)
    assert synthetic["solve"] == before["solve"] + 1
    for schedule in ("1f1b", "interleaved-1f1b", "slack-heuristic"):
        execute(tmp_path, "benchmark", schedule=schedule, run_index=0)
        assert (tmp_path / f"receipts/{schedule}.benchmark.run000.json").exists()


def test_three_plan_methods_share_profile_not_solver_results(tmp_path, synthetic):
    from tools.slackpipe_eval_config import PLAN_SCHEDULES

    experiments = [execute(tmp_path, "solve", schedule=method) for method in PLAN_SCHEDULES]
    assert len({e.completed["calibrate"]["profile"] for e in experiments}) == 1
    assert len({e.completed["solve"]["plan"] for e in experiments}) == 3
    assert synthetic["solve"] == 3
    for e in experiments:
        calibration = e.completed["calibrate"]["context"]
        assert "schedule" not in calibration
        assert calibration["calibration_schedules"] == {"stage": "interleaved-1f1b"}
        contract = e.completed["solve"]["context"]["method_contract"]
        assert contract["method_id"] == e.args.schedule
        assert e.completed["solve"]["context"]["solver_workers"] == 1


def test_no_solution_skips_gpu_and_is_resumable(tmp_path, synthetic, monkeypatch):
    original = Experiment._execute

    def no_solution(self, command, directory, name):
        if name == "solve":
            synthetic["solve"] += 1
            write_json(
                directory / "solver.json",
                dict(
                    schema_version=2,
                    method_id="cpsat-no-hint",
                    final_solution_available=False,
                    solver_status_raw="UNKNOWN",
                    best_bound_ticks=7,
                    makespan_ticks=None,
                ),
            )
        else:
            original(self, command, directory, name)

    monkeypatch.setattr(Experiment, "_execute", no_solution)
    exp = Experiment(args_for(tmp_path, "benchmark", schedule="cpsat-no-hint"))
    assert exp.execute()["status"] == "skipped_prerequisite"
    receipt = exp.completed["solve"]
    assert receipt["status"] == "no_valid_solution"
    assert "plan" not in receipt
    assert synthetic["cpsat-no-hint.benchmark"] == 0
    before = synthetic.copy()
    resumed = Experiment(args_for(tmp_path, "benchmark", schedule="cpsat-no-hint"))
    assert resumed.execute()["status"] == "skipped_prerequisite"
    assert synthetic == before


def test_old_hinted_receipt_cannot_be_relabelled_as_no_hint(tmp_path, synthetic):
    exp = execute(tmp_path, "solve", schedule="cpsat-no-hint")
    receipt = exp.completed["solve"]
    old = dict(receipt, schema_version="slackpipe.eval_receipt.v2")
    state, reason, accepted = assess_receipt(old, receipt["context"], tmp_path, {})
    assert state == "stale" and accepted is None
    changed = dict(receipt["context"], method_contract=dict(version=2, method_id="joint"))
    old = dict(receipt, context=changed, context_hash=fingerprint(changed))
    assert assess_receipt(old, receipt["context"], tmp_path, {})[0] == "stale"


def downgrade_to_v1(exp):
    """Reproduce the original global context and full-receipt parent hashes exactly."""
    a = exp.args
    old = {}
    for stage, receipt in exp.completed.items():
        context = dict(
            exp._base_context(),
            schedule=a.schedule,
            seed=a.seed,
            learning_rate=a.learning_rate,
            policy={
                k: getattr(a, k)
                for k in (
                    "warmups",
                    "iterations",
                    "calibration_iterations",
                    "repetitions",
                    "profiler_wait",
                    "profiler_warmup",
                    "profiler_active",
                    "profiler_repeat",
                )
            },
            parents={p: fingerprint(old[p]) for p in exp.dependencies(stage)},
            solver_sha256=digest(a.solver) if stage == "solve" and a.solver.is_file() else None,
            solver_seconds=a.solver_seconds if stage == "solve" else None,
            run_index=a.run_index if stage in ("benchmark", "trace") else None,
        )
        result = dict(receipt, schema_version="slackpipe.eval_receipt.v1", context=context)
        result.pop("context_hash")
        old[stage] = result
        write_json(exp.receipt_path(stage), result)
    return old


def test_octopipe_distinct_solver_shared_calibration(tmp_path, synthetic, monkeypatch):
    commands = []
    original = Experiment._execute

    def record(self, command, directory, name):
        commands.append((self.args.schedule, name, command))
        return original(self, command, directory, name)

    monkeypatch.setattr(Experiment, "_execute", record)
    slack = execute(tmp_path, "solve", schedule="slack-heuristic").completed["solve"]
    octo = execute(tmp_path, "solve", schedule="octopipe-tuning").completed["solve"]
    assert octo["context"]["solver_algorithm"] == "octopipe-tuning"
    assert octo["context"]["octopipe_fixed_placement"] is True
    assert slack["plan"] != octo["plan"]
    assert slack["context"]["parents"]["calibrate"] == octo["context"]["parents"]["calibrate"]
    assert synthetic["octopipe-tuning.calibrate"] == 0
    command = next(
        c for method, name, c in commands if method == "octopipe-tuning" and name == "solve"
    )
    assert command[command.index("--algorithm") + 1] == "octopipe-tuning"
    assert "--octopipe-fixed-placement" not in command  # fixed by the backend contract
    assert Experiment(args_for(tmp_path, schedule="octopipe-tuning")).dependencies("smoke") == (
        "solve",
    )
    before = synthetic.copy()
    execute(tmp_path, "solve", schedule="octopipe-tuning")
    assert before == synthetic


@pytest.mark.parametrize("schedule", ["slack-heuristic", "octopipe-tuning", "cpsat-no-hint"])
def test_corrupt_selected_profile_prevents_campaign_solver_without_retry(
    tmp_path, synthetic, monkeypatch, schedule
):
    original = Experiment._worker
    collections = []

    def worker(self, stage, directory, **kwargs):
        if stage != "calibrate":
            return original(self, stage, directory, **kwargs)
        collections.append(kwargs["backend"])
        directory.mkdir(parents=True)
        write_json(
            directory / "cost_profile.json",
            {"profiling_backend": "stage", "schema_version": "corrupt"},
        )

    monkeypatch.setattr(Experiment, "_worker", worker)
    experiment = Experiment(args_for(tmp_path, "solve", schedule=schedule))
    with pytest.raises(RuntimeError, match="Selected stage profiling failed; no fallback"):
        experiment.execute()
    assert collections == ["stage"]
    assert synthetic["solve"] == 0
    receipt = json.loads(experiment.receipt_path("calibrate").read_text())
    assert receipt["status"] == "failed"
    assert any("collection.json" in path for path in receipt["artifacts"])


def test_inspect_is_read_only_and_explains(tmp_path, synthetic):
    execute(tmp_path, "solve")
    original = {p: digest(p) for p in tmp_path.rglob("*") if p.is_file()}
    before = synthetic.copy()
    report = Experiment(args_for(tmp_path, "inspect", repetitions=1)).execute()
    by_stage = {r["stage"]: r for r in report["inspection"]}
    assert by_stage["solve"]["compatibility"] == "compatible"
    assert by_stage["benchmark"]["status"] == "missing"
    assert original == {p: digest(p) for p in tmp_path.rglob("*") if p.is_file()}
    assert before == synthetic


def test_worker_uses_stage_warmups(tmp_path, monkeypatch):
    exp = Experiment(
        args_for(tmp_path, warmups=30, calibration_warmups=7, smoke_warmups=2, trace_warmups=9)
    )
    commands = []
    monkeypatch.setattr(exp, "_execute", lambda cmd, *args: commands.append(cmd))
    for stage in ("calibrate", "smoke", "benchmark", "trace"):
        exp._worker(stage, tmp_path / stage)
    assert [c[c.index("--warmups") + 1] for c in commands] == [7, 2, 30, 9]
    exp.args.profiler_active = 7
    exp.args.iterations = 100
    commands.clear()
    for stage in ("calibrate", "smoke", "benchmark", "trace"):
        exp._worker(stage, tmp_path / stage)
    assert [c[c.index("--profiler-active") + 1] for c in commands] == [3, 3, 3, 7]
    assert commands[-1][commands[-1].index("--iterations") + 1] == (2 + 2 + 7) * 3


def test_real_campaign_partial_resume_three_repetitions(tmp_path, synthetic, monkeypatch):
    root = tmp_path / "llama/8b"
    upstream = execute(root, "solve", repetitions=3)
    # Complete exactly one campaign repetition, then resume the real campaign entry point.
    for schedule in COMPARISON_SCHEDULES:
        execute(root, "benchmark", schedule=schedule, run_index=0)
    before = synthetic.copy()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "campaign",
            "--num-stages",
            "8",
            "--families",
            "llama",
            "--sizes",
            "8b",
            "--output",
            str(tmp_path),
            "--repetitions",
            "3",
            "--resume",
            "--profiling-method",
            "stage",
        ],
    )
    campaign.main()
    assert synthetic["solve"] == before["solve"]
    assert synthetic["slack-heuristic.calibrate"] == before["slack-heuristic.calibrate"]
    for schedule in COMPARISON_SCHEDULES:
        assert synthetic[(schedule, "benchmark", 0)] == before[(schedule, "benchmark", 0)]
        assert synthetic[(schedule, "benchmark", 1)] == 1
        assert synthetic[(schedule, "benchmark", 2)] == 1
    summaries = json.loads((tmp_path / "experiment_summary.json").read_text())
    assert [s["num_repetitions"] for s in summaries if s["stage"] == "benchmark"] == [3] * 5
    before = synthetic.copy()
    campaign.main()
    assert synthetic == before
    assert (
        execute(root, "solve").completed["solve"]["directory"]
        == upstream.completed["solve"]["directory"]
    )


def test_indexed_repetition_count_does_not_change_identity(tmp_path, synthetic):
    first = execute(tmp_path, "benchmark", run_index=0, repetitions=3)
    before = synthetic.copy()
    second = execute(tmp_path, "benchmark", run_index=0, repetitions=6)
    assert directories(first) == directories(second)
    assert synthetic == before


def test_isolated_receipt_resume_and_policy_invalidation(tmp_path, synthetic):
    first = execute(tmp_path, "solve", profiling_method="layer")
    before = synthetic.copy()
    second = execute(tmp_path, "solve", profiling_method="layer")
    assert directories(first) == directories(second) and synthetic == before
    third = execute(tmp_path, "solve", profiling_method="layer", isolated_profile_iterations=40)
    assert third.completed["calibrate"]["directory"] != first.completed["calibrate"]["directory"]
    assert synthetic["slack-heuristic.calibrate"] == before["slack-heuristic.calibrate"] + 1


def test_force_preserves_history_and_full_overrides_resume(tmp_path, synthetic):
    first = execute(tmp_path, "full")
    selective = execute(tmp_path, "benchmark", force=True)
    assert (
        selective.completed["benchmark"]["directory"] != first.completed["benchmark"]["directory"]
    )
    assert selective.completed["solve"]["directory"] == first.completed["solve"]["directory"]
    full = execute(tmp_path, "full", force=True, resume=True)
    assert all(full.completed[s]["directory"] != r["directory"] for s, r in first.completed.items())
    assert all((tmp_path / r["directory"]).exists() for r in first.completed.values())


def test_solver_binary_hash_invalidates_only_solve_downstream(tmp_path, synthetic):
    binary = tmp_path / "solver-binary"
    binary.write_text("build1")
    first = execute(tmp_path / "experiment", "full", solver=binary)
    binary.write_text("build2")
    second = execute(tmp_path / "experiment", "full", solver=binary)
    assert {
        s
        for s in first.completed
        if first.completed[s]["directory"] != second.completed[s]["directory"]
    } == {"solve", "smoke", "benchmark", "trace"}


def test_fresh_campaign_shares_only_compatible_upstream_work(tmp_path, synthetic, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "campaign",
            "--num-stages",
            "8",
            "--families",
            "llama",
            "--sizes",
            "8b",
            "--output",
            str(tmp_path),
            "--profiling-method",
            "stage",
        ],
    )
    campaign.main()
    assert synthetic["correctness"] == 2  # Native N=PP and explicit logical/VPP topology.
    assert sum(synthetic[f"{s}.calibrate"] for s in COMPARISON_SCHEDULES) == 1
    assert sum(synthetic[f"{s}.smoke"] for s in COMPARISON_SCHEDULES) == 5
    assert synthetic["solve"] == 3  # Never share different method contracts.
    for schedule in COMPARISON_SCHEDULES:
        assert synthetic[f"{schedule}.benchmark"] == 3
        assert synthetic[f"{schedule}.trace"] == 1
    root = tmp_path / "llama/8b/receipts"
    calibration = {
        s: json.loads((root / f"{s}.calibrate.json").read_text())
        for s in ("octopipe-tuning", "slack-heuristic", "cpsat-no-hint")
    }
    assert calibration["octopipe-tuning"]["profile"] == calibration["slack-heuristic"]["profile"]
    assert calibration["cpsat-no-hint"]["profile"] == calibration["slack-heuristic"]["profile"]
    before = synthetic.copy()
    with pytest.raises(RuntimeError, match="Existing/stale receipt"):
        campaign.main()  # Fresh sharing is not implicit resume of a previous campaign.
    assert synthetic == before


def test_campaign_sharing_checks_artifact_integrity(tmp_path, synthetic):
    first = execute(tmp_path, "solve", schedule="octopipe-tuning")
    shared = [first.receipt_path(s) for s in first.completed]
    calibration = first.completed["calibrate"]
    (tmp_path / calibration["profile"]).write_text("corrupted")
    second = Experiment(args_for(tmp_path, resume=False), shared_receipts=shared)
    second.execute()
    assert synthetic["slack-heuristic.calibrate"] == 1
    assert second.completed["calibrate"]["directory"] != calibration["directory"]


def test_campaign_sharing_does_not_adopt_unlisted_receipts(tmp_path, synthetic):
    first = execute(tmp_path, "solve", schedule="octopipe-tuning")
    second = Experiment(
        args_for(tmp_path, resume=False), shared_receipts=[first.receipt_path("env")]
    )
    second.execute()
    assert synthetic["correctness"] == 2
    assert synthetic["slack-heuristic.calibrate"] == 1
    assert second.completed["env"]["directory"] == first.completed["env"]["directory"]


@pytest.mark.parametrize("duplicate", ["family", "size", "symlink", "model", "output", "alias"])
def test_campaign_rejects_duplicates_before_launch(tmp_path, synthetic, monkeypatch, duplicate):
    configs = tmp_path / "configs"
    configs.mkdir()
    model = json.loads(CONFIG.read_text())
    write_json(configs / "llama_8b.json", model)
    families, sizes, schedules = "llama", "8b", "slack-heuristic"
    if duplicate == "family":
        families = "llama, llama"
    elif duplicate == "size":
        sizes = "8b,8b"
    elif duplicate in ("symlink", "model", "output"):
        sizes = "8b,copy"
        other = configs / "llama_copy.json"
        if duplicate == "symlink":
            other.symlink_to(configs / "llama_8b.json")
        else:
            if duplicate == "output":
                model["name"] = "different-model-same-output"
            write_json(other, model)
    else:
        schedules = "interleaved,1f1b-interleave"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "campaign",
            "--num-stages",
            "8",
            "--configs",
            str(configs),
            "--output",
            str(tmp_path / "results"),
            "--families",
            families,
            "--sizes",
            sizes,
            "--methods",
            schedules,
        ],
    )
    with pytest.raises(SystemExit) as exc:
        campaign.main()
    assert exc.value.code == 2
    assert not synthetic
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize(
    "schedule,n",
    [("slack-heuristic", 8), ("cpsat-no-hint", 8), ("interleaved-1f1b", 8), ("octopipe-tuning", 8)],
)
def test_worker_effective_n_is_applied_once(tmp_path, monkeypatch, schedule, n):
    exp = Experiment(args_for(tmp_path, schedule=schedule))
    commands = []
    monkeypatch.setattr(exp, "_execute", lambda cmd, *rest: commands.append(cmd))
    exp._worker("calibrate", tmp_path / "calibration", schedule="interleaved-1f1b")
    exp._worker("smoke", tmp_path / "execution", schedule=schedule)
    assert commands[0][commands[0].index("--num-stages") + 1] == n
    assert commands[1][commands[1].index("--num-stages") + 1] == 8
    assert Experiment(exp.args).topology["num_stages"] == n


def test_v4_integrity_and_detailed_diagnostics(tmp_path, synthetic):
    exp = execute(tmp_path, "benchmark")
    receipt = exp.completed["benchmark"]
    new = Experiment(args_for(tmp_path, "benchmark", iterations=100))
    expected = new._context("benchmark", {"smoke": exp.completed["smoke"]})
    state, reason, _ = assess_receipt(receipt, expected, tmp_path, {})
    assert state == "stale" and "policy.iterations" in reason
    damaged = dict(receipt, context_hash="wrong")
    assert assess_receipt(damaged, expected, tmp_path, {})[0] == "invalid"
    damaged = dict(receipt, context=dict(receipt["context"], context_schema_version=99))
    damaged["context_hash"] = fingerprint(damaged["context"])
    assert assess_receipt(damaged, expected, tmp_path, {})[0] == "invalid"
