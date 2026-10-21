# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Standalone producers and peripheral commands use structural publication only."""

import copy
import json
import os
import shlex
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from megatron.core.pipeline_parallel.slackpipe.cost_profile import (
    aggregate_stage_costs,
    profile_fingerprint,
)
from megatron.core.pipeline_parallel.slackpipe.isolated_profile import POLICY_VERSION
from megatron.core.pipeline_parallel.slackpipe.manifest import manifest_fingerprint
from megatron.core.pipeline_parallel.slackpipe.profile_integrity import (
    tag_calibration_events,
    validate_profile_integrity,
)
from tools import slackpipe_hybrid, slackpipe_isolated_profile, slackpipe_profile_integrity
from tools.slackpipe_eval_args import argument_parser

ROOT = Path(__file__).resolve().parents[3]


def manifest():
    value = dict(
        schema_version="slackpipe.model_manifest.v1",
        num_layers=6,
        model_config=dict(hidden_size=64),
        layers=[dict(layer_id=i, layer_type="decoder", config_class="decoder") for i in range(6)],
    )
    value["manifest_hash"] = manifest_fingerprint(value)
    return value


@pytest.mark.parametrize("values", [[2.0], [1.0, 100.0, 3.0]])
def test_isolated_collect_publishes_all_samples_once(tmp_path, monkeypatch, values):
    worker = slackpipe_isolated_profile
    # This CPU publication fixture mocks all GPU work; capacity is a prerequisite,
    # not part of the sample-retention behavior under test.
    require_devices = Mock(return_value=1)
    monkeypatch.setattr("tools.slackpipe_eval_common.require_cuda_devices", require_devices)
    args = SimpleNamespace(
        output=tmp_path / "layer",
        model_config=tmp_path / "model.json",
        schedule="interleaved-1f1b",
        pp=2,
        logical_stages=4,
        microbatches=4,
        seq_length=8,
        micro_batch_size=1,
        precision="fp32",
        seed=1,
        isolated_profile_warmups=1,
        isolated_profile_iterations=len(values),
    )
    monkeypatch.setattr(worker, "load_model", lambda _: dict(vocab_size=128, num_layers=6))
    monkeypatch.setattr(
        worker,
        "transformer_config",
        lambda *a: SimpleNamespace(
            recompute_granularity=None,
            cross_entropy_loss_fusion=False,
            apply_query_key_layer_scaling=False,
        ),
    )
    monkeypatch.setattr(worker, "build_model_manifest", lambda _: manifest())
    monkeypatch.setattr(worker, "environment_metadata", lambda: dict(fixture=True))
    monkeypatch.setattr(worker.dist, "is_initialized", lambda: False)
    for name in ("init_process_group", "destroy_process_group"):
        monkeypatch.setattr(worker.dist, name, Mock())
    for name in ("initialize_model_parallel", "destroy_model_parallel"):
        monkeypatch.setattr(worker.parallel_state, name, Mock())
    for name in ("set_device", "reset_peak_memory_stats", "empty_cache"):
        monkeypatch.setattr(worker.torch.cuda, name, Mock())
    for name in ("max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(worker.torch.cuda, name, lambda: 0)
    monkeypatch.setattr(worker.torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(worker.torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(worker, "model_parallel_cuda_manual_seed", Mock())
    monkeypatch.setattr(worker, "no_communication", nullcontext)
    monkeypatch.setattr(worker, "make_unit", lambda *a: (object(), None, None))
    measured = dict(
        forward_gpu_ms=values,
        backward_gpu_ms=values,
        samples=[
            dict(iteration=i + 1, forward_gpu_ms=v, backward_gpu_ms=v) for i, v in enumerate(values)
        ],
    )
    measure = Mock(side_effect=lambda *a, **k: copy.deepcopy(measured))
    monkeypatch.setattr(worker, "measure_unit", measure)
    worker.collect(args)
    require_devices.assert_called_once_with(1, local_rank=0)
    assert measure.call_count == 3  # One class, embedding, and final norm/head/loss.
    profile = json.loads((args.output / "cost_profile.json").read_text())
    raw = json.loads((args.output / "isolated_profile_raw.json").read_text())
    validate_profile_integrity(profile)
    assert raw["context"]["policy_version"] == POLICY_VERSION
    assert all(u["forward_gpu_ms"] == values for u in raw["units"])
    assert profile["prefix_forward_us"][-1] == 6 * (2000 if len(values) == 1 else 3000)
    assert profile["measurement_parallel_config"]["pp"] == 1
    assert profile["target_parallel_config"]["pp"] == 2
    assert "quality" not in profile
    assert not list(args.output.glob("*.quality.json"))
    assert not list(args.output.glob("*.candidate.json"))
    with pytest.raises(FileExistsError):
        worker.collect(args)
    assert measure.call_count == 3


@pytest.mark.parametrize("phase", ["calibrate", "benchmark"])
def test_isolated_main_uses_shared_phase_without_estimator_flag(tmp_path, monkeypatch, phase):
    collect = Mock()
    monkeypatch.setattr(slackpipe_isolated_profile, "collect", collect)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "isolated",
            "--phase",
            phase,
            "--model-config",
            str(ROOT / "configs/slackpipe_eval/llama_4b.json"),
            "--pp",
            "2",
            "--num-stages",
            "4",
            "--output",
            str(tmp_path / "layer"),
            "--profiling-method",
            "layer",
        ],
    )
    if phase == "calibrate":
        slackpipe_isolated_profile.main()
        assert collect.call_count == 1
        assert collect.call_args.args[0].solver_profile == "layer"
    else:
        with pytest.raises(SystemExit) as exc:
            slackpipe_isolated_profile.main()
        assert exc.value.code == 2
        collect.assert_not_called()


def hybrid_inputs(tmp_path):
    model = manifest()
    observed = []
    for group, split in enumerate(([1, 1, 4], [2, 2, 2], [3, 1, 2])):
        cuts = [0]
        for length in split:
            cuts.append(cuts[-1] + length)
        events = [
            dict(
                iteration=i,
                logical_stage=s,
                phase=phase + "_compute",
                microbatch=b,
                elapsed_ms=(split[s] + s + 1) * factor * (1 + group),
            )
            for i, factor in enumerate((1.0, 10.0, 2.0))
            for s in range(3)
            for phase in ("forward", "backward")
            for b in range(2)
        ]
        tag_calibration_events(
            events,
            group=str(group),
            ranges=list(zip(cuts, cuts[1:])),
            worker=0,
            configuration=dict(fixture=True),
            raw_path=tmp_path / f"raw{group}.json",
        )
        observed.extend(
            aggregate_stage_costs(
                events, layer_split=split, num_microbatches=2, iteration_start=0, iteration_end=2
            )
        )
    manifest_path, observations = tmp_path / "manifest.json", tmp_path / "observations.json"
    manifest_path.write_text(json.dumps(model))
    observations.write_text(json.dumps(observed))
    return manifest_path, observations


def test_hybrid_fit_uses_residual_policy_and_no_quality_receipt(tmp_path, monkeypatch):
    model, observations = hybrid_inputs(tmp_path)
    before = observations.read_bytes()
    output = tmp_path / "cost_profile.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hybrid",
            "fit",
            "--manifest",
            str(model),
            "--observations",
            str(observations),
            "--output",
            str(output),
            "--calibration-pp",
            "1",
            "--calibration-vpp",
            "3",
            "--stage-residual-k",
            "7",
            "--stage-residual-rho",
            "0.75",
        ],
    )
    slackpipe_hybrid.main()
    profile = json.loads(output.read_text())
    validate_profile_integrity(profile)
    assert profile["profiling_policy"] == dict(
        policy_version="stage-affine-residual-v1", k=7, rho=0.75
    )
    assert "quality" not in profile
    assert observations.read_bytes() == before
    assert not list(tmp_path.glob("*.quality.json"))
    with pytest.raises(FileExistsError):
        slackpipe_hybrid.main()


def test_integrity_cli_ignores_old_quality_but_rejects_corrupt_costs(tmp_path, monkeypatch):
    output = tmp_path / "profile.json"
    profile = dict(
        schema_version="slackpipe.cost_profile.v1",
        units="milliseconds",
        a_fwd=1.0,
        a_bwd=2.0,
        bias_fwd=[0.0],
        bias_bwd=[0.0],
        quality=dict(status="failed"),
    )
    output.write_text(json.dumps(profile))
    monkeypatch.setattr(sys, "argv", ["integrity", "--profile", str(output)])
    before = output.read_bytes()
    slackpipe_profile_integrity.main()
    assert output.read_bytes() == before
    profile["a_fwd"] = -1
    profile["cost_profile_hash"] = profile_fingerprint(profile)
    output.write_text(json.dumps(profile))
    with pytest.raises(ValueError):
        slackpipe_profile_integrity.main()


@pytest.mark.parametrize(
    "script,phase,method",
    [
        ("runpod_nemotron_baseline.sh", "smoke", "interleaved-1f1b"),
        ("runpod_nemotron_slackpipe.sh", "smoke", "slack-heuristic"),
        ("runpod_slackpipe_correctness.sh", "correctness", "slack-heuristic"),
        ("runpod_slackpipe_verify.sh", "env", "slack-heuristic"),
    ],
)
def test_shell_wrappers_emit_shared_cli(tmp_path, script, phase, method):
    result = subprocess.run(
        ["bash", str(ROOT / "tools" / script), "--output", str(tmp_path / "run")],
        env=dict(os.environ, PYTHON="/bin/echo"),
        check=True,
        text=True,
        capture_output=True,
    )
    command = shlex.split(result.stdout)
    assert command[:2] == ["-m", "tools.run_slackpipe_eval"]
    args = argument_parser().parse_args(command[2:])
    assert (args.stage, args.schedule, args.logical_stages) == (phase, method, 8)


@pytest.mark.parametrize(
    "module", ["slackpipe_calibration_transfer", "slackpipe_linear_identification"]
)
def test_diagnostic_pipeline_commands_use_shared_cli(tmp_path, module):
    from importlib import import_module

    producer = import_module("tools." + module)
    if module == "slackpipe_calibration_transfer":
        _, command = producer.pipeline_command(tmp_path, 0, 2, "off")
    else:
        _, command = producer.pipeline_command(tmp_path, 0, "off")
    parser = argument_parser()
    parser.add_argument("--calibration-design")
    parser.add_argument("--run-id")
    args = parser.parse_args(command[command.index("--phase") :])
    assert (args.stage, args.schedule, args.logical_stages) == ("calibrate", "interleaved-1f1b", 4)
