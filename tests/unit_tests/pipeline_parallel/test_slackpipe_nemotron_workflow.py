# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Offline preparation and fail-closed gates; no full 8B allocation."""

from dataclasses import replace

import pytest
import torch

from megatron.core.pipeline_parallel.slackpipe.cost_profile import profile_fingerprint
from megatron.core.pipeline_parallel.slackpipe.hybrid import nemotron_h_8b_config
from megatron.core.pipeline_parallel.slackpipe.manifest import (
    build_model_manifest,
    validate_plan_model,
)
from megatron.core.pipeline_parallel.slackpipe.schedule import _validate_unsupported_features
from tests.unit_tests.pipeline_parallel.test_slackpipe_hybrid import tiny_config
from tools.slackpipe_training import calibration_partitions, construction_plan, write_json


def test_nemotron_calibration_design():
    manifest = build_model_manifest(nemotron_h_8b_config())
    partitions, diagnostics = calibration_partitions(manifest, 8)
    assert diagnostics["rank"] == diagnostics["columns"] == 6
    assert len(partitions) >= 6
    for cuts in partitions:
        plan = construction_plan(cuts, 4, 8, manifest)
        assert plan.num_layers == 52
        assert sum(map(len, plan.operations)) == 128
        assert list(range(52)) == [i for s in range(8) for i in plan.stage_layer_ids(s)]
    bad = {"layers": [{"config_class": "only"}] * 8}
    with pytest.raises(ValueError, match="Rank-deficient"):
        calibration_partitions(bad, 8)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_runtime_dtype_and_profile_validation(tmp_path, dtype):
    config = tiny_config(dtype=dtype)
    _validate_unsupported_features(config, False)
    manifest = build_model_manifest(config)
    plan = construction_plan((0, 2, 5, 9, 12), 2, 8, manifest)
    profile = dict(
        schema_version="slackpipe.cost_profile.v2",
        model_manifest_hash=manifest["manifest_hash"],
        model_config={"dtype": "bf16" if dtype == torch.bfloat16 else "fp32"},
    )
    profile["cost_profile_hash"] = profile_fingerprint(profile)
    path = tmp_path / "profile.json"
    write_json(path, profile)
    linked = replace(
        plan,
        cost_profile_hash=profile["cost_profile_hash"],
        cost_profile_path=str(path),
        cost_profile_version=profile["schema_version"],
    )
    validate_plan_model(linked, config)
    config.params_dtype = torch.float32 if dtype == torch.bfloat16 else torch.bfloat16
    with pytest.raises(ValueError, match="dtype"):
        validate_plan_model(linked, config)
    with pytest.raises(ValueError, match="dtype"):
        _validate_unsupported_features(config, False)
    config.params_dtype = torch.float16
    with pytest.raises(ValueError, match="FP32 and BF16"):
        _validate_unsupported_features(config, False)
