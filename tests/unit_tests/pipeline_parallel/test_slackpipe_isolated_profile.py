# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""All-sample layer medians, structural integrity and no-communication autograd."""

import copy
import json
import statistics
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.pipeline_parallel.slackpipe.cost_profile import profile_fingerprint
from megatron.core.pipeline_parallel.slackpipe.isolated_profile import (
    ESTIMATOR,
    POLICY_VERSION,
    build_profile,
    class_mapping,
    summarize,
)
from megatron.core.pipeline_parallel.slackpipe.manifest import manifest_fingerprint
from megatron.core.pipeline_parallel.slackpipe.profile_integrity import validate_profile_integrity
from tools.run_slackpipe_eval import Experiment, argument_parser
from tools.slackpipe_eval_config import transformer_config
from tools.slackpipe_isolated_profile import make_unit, measure_unit, no_communication
from tools.slackpipe_profiling import profiling_policy


def fixture_profile(
    kinds=("mamba", "mlp", "mamba", "attention"),
    unstable=False,
    manifest=None,
    execution=None,
    warmups=20,
    iterations=30,
):
    manifest = manifest or dict(
        schema_version="slackpipe.model_manifest.v1",
        num_layers=len(kinds),
        model_config=dict(hidden_size=64),
        layers=[dict(layer_id=i, layer_type=k, config_class=k) for i, k in enumerate(kinds)],
    )
    manifest["manifest_hash"] = manifest_fingerprint(manifest)
    context = dict(
        execution=execution or dict(precision="fp32", hidden_size=64),
        policy_version=POLICY_VERSION,
        warmups=warmups,
        iterations=iterations,
    )
    classes = class_mapping(manifest, context["execution"])
    units = []
    for i, key in enumerate([c["class_signature"] for c in classes] + ["first", "last"]):
        f = [float(i + 1)] * iterations
        b = [float(11 + i * i)] * iterations
        if unstable:
            b = [1.0 if j % 2 == 0 else 5.0 for j in range(iterations)]
        units.append(
            dict(
                class_signature=key,
                forward_gpu_ms=f,
                backward_gpu_ms=b,
                samples=[
                    dict(iteration=warmups + j, forward_gpu_ms=x, backward_gpu_ms=y)
                    for j, (x, y) in enumerate(zip(f, b))
                ],
            )
        )
    return build_profile(
        manifest=manifest,
        classes=classes,
        units=units,
        context=context,
        model_config={},
        parallel_config=dict(pp=2, vpp=2, tp=1, cp=1, dp=1),
    )


def test_homogeneous_and_heterogeneous_expansion():
    homogeneous = fixture_profile(("decoder",) * 8)
    assert len(homogeneous["class_costs_us"]) == 1
    assert homogeneous["prefix_forward_us"] == [1000.0 * i for i in range(9)]
    p = fixture_profile()
    assert [x["forward_us"] for x in p["layer_costs_us"]] == [1000, 2000, 1000, 3000]
    assert [x["backward_us"] for x in p["layer_costs_us"]] == [11000, 12000, 11000, 15000]
    for i, j in ((0, 1), (0, 4), (1, 3), (2, 4)):
        for phase in ("forward", "backward"):
            assert p[f"prefix_{phase}_us"][j] - p[f"prefix_{phase}_us"][i] == sum(
                x[phase + "_us"] for x in p["layer_costs_us"][i:j]
            )
    assert p["stage_role_bias_us"]["middle"] == dict(forward=0, backward=0)
    assert p["stage_role_bias_us"]["first"]["forward"] == 4000
    assert p["stage_role_bias_us"]["last"]["forward"] == 5000
    assert p["estimator"] == ESTIMATOR
    assert p["profiling_backend"] == "layer"
    assert p["profiling_policy"] == dict(policy_version="layer-median-v1")
    assert p["measurement_parallel_config"] == dict(pp=1, vpp=None, tp=1, cp=1, dp=1)
    assert p["target_parallel_config"] == p["parallel_config"]
    assert p["target_parallel_config"]["pp"] == 2
    for field in ("includes_pipeline_wait", "includes_p2p", "includes_communication"):
        assert p["measurement_definition"][field] is False
    validate_profile_integrity(json.loads(json.dumps(p)))


@pytest.mark.parametrize("value", [True, "1", None])
def test_non_numeric_layer_samples_are_not_medians(value):
    with pytest.raises(ValueError, match="finite samples"):
        summarize([1.0, value])


def test_boolean_iteration_cannot_alias_integer_inventory():
    p = fixture_profile(warmups=1)
    p["isolated"]["units"][0]["samples"][0]["iteration"] = True
    p["cost_profile_hash"] = profile_fingerprint(p)
    with pytest.raises(ValueError, match="integer bounds"):
        validate_profile_integrity(p)


def test_signature_is_complete_stable_and_instance_sensitive():
    p = fixture_profile(("decoder",) * 4)
    manifest = dict(p["isolated"]["manifest"], schema_version="slackpipe.model_manifest.v1")
    execution = p["isolated"]["context"]["execution"]
    first = class_mapping(manifest, execution)
    assert class_mapping(copy.deepcopy(manifest), copy.deepcopy(execution)) == first
    assert class_mapping(manifest, dict(execution, precision="bf16")) != first
    manifest["layers"][2]["block_config"] = dict(ffn_hidden_size=128)
    manifest["manifest_hash"] = manifest_fingerprint(manifest)
    assert len(class_mapping(manifest, execution)) == 2


def test_single_stage_charges_both_boundaries_once():
    p = fixture_profile()
    data = p["isolated"]
    single = build_profile(
        manifest=dict(data["manifest"], schema_version="slackpipe.model_manifest.v1"),
        classes=data["classes"],
        units=data["units"],
        context=data["context"],
        model_config={},
        parallel_config=dict(pp=1, vpp=None),
    )
    for phase in ("forward", "backward"):
        assert single["stage_role_bias_us"]["first"][phase] == sum(
            p["stage_role_bias_us"][role][phase] for role in ("first", "last")
        )
        assert single[f"prefix_{phase}_us"] == p[f"prefix_{phase}_us"]
    validate_profile_integrity(single)


def test_optional_pipeline_comparison_is_not_an_equality_gate():
    from tools.slackpipe_compare_isolated import compare

    p = fixture_profile()
    pipeline = dict(
        manifest=p["model_manifest_hash"],
        metadata={},
        events=[
            dict(
                calibration_group_id=f"partition{g}",
                worker=0,
                logical_stage=0,
                stage_layer_range=[0, 2],
                stage_role="middle",
                phase="backward_compute",
                iteration=i,
                elapsed_ms=value,
            )
            for g, value in enumerate((46, 92))
            for i in range(3)
        ],
    )
    rows = compare(p, pipeline)
    assert [r["isolated_layers_ms"] for r in rows] == [23, 23]
    assert [r["pipeline_wall_ms"] for r in rows] == [46, 92]
    assert all(len(r["iteration_means_ms"]) == 3 for r in rows)
    pipeline["manifest"] = "different"
    with pytest.raises(ValueError, match="manifest"):
        compare(p, pipeline)


@pytest.mark.parametrize(
    "values",
    [[2.0], [1.0, 100.0], [1.0] * 8, [1.0, 5.0] * 15, [1.0] * 29 + [100.0], [1.0, 3.0, 5.0, 100.0]],
)
def test_summary_uses_every_legal_sample(values):
    original = list(values)
    assert summarize(values) == dict(
        mean_ms=statistics.fmean(values),
        median_ms=statistics.median(values),
        stddev_ms=statistics.pstdev(values),
        min_ms=min(values),
        max_ms=max(values),
        count=len(values),
        selected_ms=statistics.median(values),
    )
    assert values == original


@pytest.mark.parametrize(
    "values", [[], [0.0], [-1.0], [float("nan")], [float("inf")], [float("-inf")]]
)
def test_summary_rejects_missing_or_illegal_timings(values):
    with pytest.raises(ValueError, match="positive finite samples"):
        summarize(values)


@pytest.mark.parametrize("iterations", [1, 2, 8, 30])
def test_noisy_profile_has_no_stability_or_minimum_sample_gate(iterations):
    p = fixture_profile(unstable=True, iterations=iterations)
    assert "quality" not in p
    expected = [1.0 if i % 2 == 0 else 5.0 for i in range(iterations)]
    for unit in p["isolated"]["units"]:
        assert unit["backward_gpu_ms"] == expected
        summary = p["isolated"]["summaries"][unit["class_signature"]]["backward"]
        assert summary["count"] == iterations
        assert summary["selected_ms"] == statistics.median(expected)
    assert all(
        row["backward_us"] == 1000 * statistics.median(expected) for row in p["layer_costs_us"]
    )
    validate_profile_integrity(json.loads(json.dumps(p)))


def test_historical_quality_failure_does_not_block_integrity_validation():
    p = fixture_profile(unstable=True)
    p["quality"] = dict(status="failed", failed_rules=["historical-instability"])
    p["cost_profile_hash"] = profile_fingerprint(p)
    original = copy.deepcopy(p)
    validate_profile_integrity(p)
    assert p == original


def test_legacy_recorded_selection_is_audited_without_running_old_filter():
    p = fixture_profile()
    p.pop("profiling_policy")
    p["isolated"]["context"].pop("policy_version")
    p["quality"] = dict(status="rerun_required")
    for summary in p["isolated"]["summaries"].values():
        for phase in ("forward", "backward"):
            summary[phase]["excluded_indices"] = []
    unit = p["isolated"]["units"][0]
    unit["forward_gpu_ms"][0] = 100.0
    unit["samples"][0]["forward_gpu_ms"] = 100.0
    p["isolated"]["summaries"][unit["class_signature"]]["forward"]["excluded_indices"] = [0]
    p["cost_profile_hash"] = profile_fingerprint(p)
    original = copy.deepcopy(p)
    validate_profile_integrity(p)
    assert p == original and "profiling_policy" not in p
    p["class_costs_us"][unit["class_signature"]]["forward"] += 1
    p["cost_profile_hash"] = profile_fingerprint(p)
    with pytest.raises(ValueError, match="Historical class expansion"):
        validate_profile_integrity(p)


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("prefix", "Prefix costs"),
        ("raw", "raw samples"),
        ("policy", "strategy identity"),
        ("advertised_policy", "strategy identity"),
        ("class", "Class mapping"),
        ("execution", "Class mapping"),
        ("manifest", "manifest hash"),
        ("model_manifest_hash", "model_manifest_hash"),
        ("summary", "summaries"),
        ("class_cost", "class_costs_us"),
        ("layer_cost", "layer_costs_us"),
        ("boundary", "stage_role_bias_us"),
        ("units", "microseconds"),
    ],
)
def test_integrity_rebuild_rejects_corruption_even_with_updated_hash(corruption, message):
    p = fixture_profile()
    if corruption == "prefix":
        p["prefix_backward_us"][2] += 100
    elif corruption == "raw":
        p["isolated"]["units"][0]["samples"][0]["forward_gpu_ms"] += 1
    elif corruption == "policy":
        p["isolated"]["context"]["policy_version"] = "obsolete-layer-policy"
    elif corruption == "advertised_policy":
        p["profiling_policy"]["policy_version"] = "obsolete-layer-policy"
    elif corruption == "class":
        p["isolated"]["classes"][0]["member_layers"] = [0]
    elif corruption == "execution":
        p["isolated"]["context"]["execution"] = dict(precision="bf16", hidden_size=64)
    elif corruption == "manifest":
        p["isolated"]["manifest"]["manifest_hash"] = "different"
    elif corruption == "model_manifest_hash":
        p["model_manifest_hash"] = "different"
    elif corruption == "summary":
        p["isolated"]["summaries"]["first"]["forward"]["selected_ms"] += 1
    elif corruption == "class_cost":
        signature = p["isolated"]["classes"][0]["class_signature"]
        p["class_costs_us"][signature]["forward"] += 1
    elif corruption == "layer_cost":
        p["layer_costs_us"][0]["forward_us"] += 1
        p["prefix_forward_us"] = [0.0]
        for layer in p["layer_costs_us"]:
            p["prefix_forward_us"].append(p["prefix_forward_us"][-1] + layer["forward_us"])
    elif corruption == "boundary":
        p["stage_role_bias_us"]["last"]["backward"] += 1
    elif corruption == "units":
        p["units"] = "milliseconds"
    p["cost_profile_hash"] = profile_fingerprint(p)
    with pytest.raises(ValueError, match=message):
        validate_profile_integrity(p)


def test_integrity_rejects_stale_hash():
    p = fixture_profile()
    p["isolated"]["context"]["iterations"] += 1
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_profile_integrity(p)


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "reordered", "count", "warmups"])
def test_integrity_rejects_incomplete_or_misidentified_samples(corruption):
    p = fixture_profile()
    unit = p["isolated"]["units"][0]
    if corruption == "missing":
        unit["samples"].pop()
        unit["forward_gpu_ms"].pop()
        unit["backward_gpu_ms"].pop()
    elif corruption == "duplicate":
        unit["samples"][1] = copy.deepcopy(unit["samples"][0])
    elif corruption == "reordered":
        unit["samples"].reverse()
    elif corruption == "count":
        p["isolated"]["context"]["iterations"] += 1
    elif corruption == "warmups":
        p["isolated"]["context"]["warmups"] += 1
    p["cost_profile_hash"] = profile_fingerprint(p)
    with pytest.raises(ValueError, match="Incomplete isolated samples"):
        validate_profile_integrity(p)


@pytest.mark.parametrize("phase", ["forward", "backward"])
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_integrity_rejects_illegal_raw_timings(phase, value):
    p = fixture_profile()
    unit = p["isolated"]["units"][0]
    unit[phase + "_gpu_ms"][0] = value
    unit["samples"][0][phase + "_gpu_ms"] = value
    p["cost_profile_hash"] = profile_fingerprint(p)
    with pytest.raises(ValueError, match="positive finite samples"):
        validate_profile_integrity(p)


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "unexpected"])
def test_build_rejects_invalid_measured_units(corruption):
    p = fixture_profile()
    data = p["isolated"]
    if corruption == "missing":
        data["units"].pop()
    elif corruption == "duplicate":
        data["units"].append(copy.deepcopy(data["units"][0]))
    elif corruption == "unexpected":
        data["units"][0]["class_signature"] = "unknown"
    with pytest.raises(ValueError, match="Missing/duplicate/unexpected profiling units"):
        build_profile(
            manifest=dict(data["manifest"], schema_version="slackpipe.model_manifest.v1"),
            classes=data["classes"],
            units=data["units"],
            context=data["context"],
            model_config={},
            parallel_config=p["parallel_config"],
        )


@pytest.mark.parametrize("members", [[0], [0, 0, 2], [0, 4], [-1, 2]])
def test_build_rejects_invalid_class_expansion(members):
    p = fixture_profile()
    data = p["isolated"]
    data["classes"][0]["member_layers"] = members
    with pytest.raises(ValueError, match="class expansion|Class expansion"):
        build_profile(
            manifest=dict(data["manifest"], schema_version="slackpipe.model_manifest.v1"),
            classes=data["classes"],
            units=data["units"],
            context=data["context"],
            model_config={},
            parallel_config=p["parallel_config"],
        )


@pytest.mark.parametrize("phase", ["forward", "backward"])
@pytest.mark.parametrize("unit_index", [0, -2, -1], ids=["layer", "first", "last"])
def test_all_sample_medians_expand_independently_by_phase_and_unit(phase, unit_index):
    original = fixture_profile(iterations=4)
    data = copy.deepcopy(original["isolated"])
    unit = data["units"][unit_index]
    unit[phase + "_gpu_ms"] = [1.0, 3.0, 5.0, 100.0]
    for row, value in zip(unit["samples"], unit[phase + "_gpu_ms"]):
        row[phase + "_gpu_ms"] = value
    raw_units = copy.deepcopy(data["units"])
    p = build_profile(
        manifest=dict(data["manifest"], schema_version="slackpipe.model_manifest.v1"),
        classes=data["classes"],
        units=data["units"],
        context=data["context"],
        model_config={},
        parallel_config=original["parallel_config"],
    )
    key = unit["class_signature"]
    assert p["isolated"]["units"] == raw_units
    assert p["isolated"]["summaries"][key][phase] == summarize([1.0, 3.0, 5.0, 100.0])
    other = "backward" if phase == "forward" else "forward"
    assert p["prefix_" + other + "_us"] == original["prefix_" + other + "_us"]
    for role in ("first", "middle", "last"):
        for direction in ("forward", "backward"):
            assert p["stage_role_bias_us"][role][direction] == (
                4000.0
                if role == key and direction == phase
                else original["stage_role_bias_us"][role][direction]
            )
    for layer, previous in zip(p["layer_costs_us"], original["layer_costs_us"]):
        assert layer[phase + "_us"] == (
            4000.0 if layer["class_signature"] == key else previous[phase + "_us"]
        )
        assert layer[other + "_us"] == previous[other + "_us"]
    validate_profile_integrity(p)


@pytest.mark.parametrize("sampling", ["warmups", "iterations"])
def test_layer_policy_context_scoped_to_calibration(tmp_path, monkeypatch, sampling):
    args = argument_parser().parse_args(
        [
            "--phase",
            "calibrate",
            "--method",
            "slack-heuristic",
            "--model-config",
            "configs/slackpipe_eval/llama_8b.json",
            "--output",
            str(tmp_path),
            "--num-stages",
            "4",
            "--pp",
            "2",
        ]
    )
    assert args.profiling_method == "both"
    assert args.solver_profile is None
    monkeypatch.setattr(Experiment, "_base_context", lambda self: {})
    exp = Experiment(args)
    first = exp._context("calibrate", {})
    benchmark = exp._context("benchmark", {})
    assert first["policy"] == dict(profiling_policy(exp.args), stage_partitions="uniform")
    assert first["policy"]["solver_profile"] == "stage"
    assert first["policy"]["solver_profile_explicit"] is False
    assert first["policy"]["backends"]["layer"] == dict(
        source="collect",
        warmups=20,
        iterations=30,
        policy=dict(policy_version="layer-median-v1"),
        measurement_parallelism=dict(pp=1, vpp=None, tp=1, dp=1, cp=1),
        calibration_schedule="isolated-unit",
    )
    assert first == exp._context("calibrate", {})  # Compatible resume identity.
    field = "isolated_profile_" + sampling
    setattr(exp.args, field, getattr(exp.args, field) + 1)
    changed = exp._context("calibrate", {})
    assert changed != first
    assert changed["policy"]["backends"]["stage"] == first["policy"]["backends"]["stage"]
    assert exp._context("benchmark", {}) == benchmark


def test_layer_worker_dispatch_is_single_process(tmp_path, monkeypatch):
    args = argument_parser().parse_args(
        [
            "--phase",
            "calibrate",
            "--method",
            "slack-heuristic",
            "--model-config",
            "configs/slackpipe_eval/llama_8b.json",
            "--output",
            str(tmp_path),
            "--num-stages",
            "4",
            "--profiling-method",
            "layer",
            "--isolated-profile-warmups",
            "2",
            "--isolated-profile-iterations",
            "3",
        ]
    )
    exp = Experiment(args)
    commands = []
    monkeypatch.setattr(exp, "_execute", lambda command, *unused: commands.append(command))
    exp._worker("calibrate", tmp_path / "layer" / "worker", backend="layer")
    assert len(commands) == 1
    command = commands[0]
    assert command[1:3] == ["-m", "tools.slackpipe_isolated_profile"]
    assert "torch.distributed.run" not in command
    assert command[command.index("--profiling-method") + 1] == "layer"
    assert command[command.index("--isolated-profile-warmups") + 1] == 2
    assert command[command.index("--isolated-profile-iterations") + 1] == 3
    assert not any(
        str(x).startswith(("--quality-", "--stage-residual-", "--calibration-estimator"))
        for x in command
    )


@pytest.mark.parametrize(("warmups", "iterations"), [(1, 1), (2, 8), (20, 30)])
def test_actual_measurement_loop_counts_isolated_pairs(tmp_path, monkeypatch, warmups, iterations):
    class Event:
        def __init__(self, **kwargs):
            self.calls = 0

        def record(self):
            pass

        def elapsed_time(self, other):
            self.calls += 1
            return 1.0 if self.calls % 4 < 2 else 100.0

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    args = argument_parser().parse_args(
        [
            "--phase",
            "calibrate",
            "--model-config",
            "configs/slackpipe_eval/llama_8b.json",
            "--output",
            str(tmp_path),
            "--warmups",
            "5",
            "--iterations",
            "10",
            "--isolated-profile-warmups",
            str(warmups),
            "--isolated-profile-iterations",
            str(iterations),
        ]
    )
    module = torch.nn.Linear(2, 2)
    hidden = torch.ones(1, 2, requires_grad=True)
    calls = dict(forward=0, backward=0)

    def forward():
        calls["forward"] += 1
        out = module(hidden)
        out.register_hook(lambda grad: calls.__setitem__("backward", calls["backward"] + 1))
        return out

    measured = measure_unit(
        module,
        hidden,
        forward,
        warmups=args.isolated_profile_warmups,
        iterations=args.isolated_profile_iterations,
    )
    # One extra untimed forward prepares the upstream gradient; noisy timings add no retries.
    assert calls == dict(forward=warmups + iterations + 1, backward=warmups + iterations)
    assert measured["executed"] == dict(
        warmup_pairs=warmups, measured_pairs=iterations, initialization_forwards=1
    )
    assert measured["finite_outputs_and_gradients"] is True
    assert [s["iteration"] for s in measured["samples"]] == list(
        range(warmups, warmups + iterations)
    )
    for phase in ("forward", "backward"):
        values = measured[phase + "_gpu_ms"]
        assert len(values) == iterations
        assert values == [row[phase + "_gpu_ms"] for row in measured["samples"]]
        assert summarize(values)["count"] == iterations


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_actual_units_finite_repeatable_and_no_pipeline(tmp_path):
    from unittest.mock import patch

    from megatron.core.pipeline_parallel import schedules
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    model = json.loads(Path("configs/slackpipe_eval/nemotron_h_4b.json").read_text())
    model.update(
        num_layers=6,
        hidden_size=64,
        ffn_hidden_size=128,
        num_attention_heads=4,
        num_query_groups=2,
        vocab_size=128,
        hybrid_pattern="M*-M*-",
        mamba_state_dim=16,
        mamba_head_dim=16,
        mamba_num_groups=1,
        mamba_num_heads=8,
    )
    assert not dist.is_initialized()
    dist.init_process_group(
        "gloo", init_method="file://" + str(tmp_path / "store"), rank=0, world_size=1
    )
    try:
        parallel_state.initialize_model_parallel()
        config = transformer_config(model, dict(pp=1, vpp=None), "fp32")
        model_parallel_cuda_manual_seed(1234)
        with (
            patch.object(
                schedules, "get_forward_backward_func", side_effect=AssertionError("No pipeline")
            ),
            no_communication(),
        ):
            for kind in ("mamba", "attention", "mlp", "decoder", "first", "last"):
                if kind == "decoder":
                    model = dict(model, model_family="llama", position_embedding="rope")
                    config = transformer_config(
                        {k: v for k, v in model.items() if not k.startswith("mamba_")},
                        dict(pp=1, vpp=None),
                        "fp32",
                    )
                unit = dict(class_signature=kind, layer_type=kind, representative_layer=0)
                module, hidden, forward = make_unit(model, config, unit, 32, 1)
                out = forward()
                grad = torch.ones_like(out)
                out.backward(grad)
                reference = out.detach().clone()
                grads = [p.grad.clone() for p in module.parameters()]
                module.zero_grad(set_to_none=True)
                if hidden is not None:
                    hidden.grad = None
                out = forward()
                out.backward(grad)
                torch.testing.assert_close(out, reference, rtol=0, atol=0)
                for p, g in zip(module.parameters(), grads):
                    assert torch.isfinite(p.grad).all()
                    torch.testing.assert_close(p.grad, g, rtol=0, atol=0)
                measured = measure_unit(module, hidden, forward, warmups=1, iterations=2)
                assert measured["finite_outputs_and_gradients"] is True
                assert measured["executed"] == dict(
                    warmup_pairs=1, measured_pairs=2, initialization_forwards=1
                )
                for phase in ("forward", "backward"):
                    assert summarize(measured[phase + "_gpu_ms"])["count"] == 2
                del module, hidden, forward, out, reference, grads, grad
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()
