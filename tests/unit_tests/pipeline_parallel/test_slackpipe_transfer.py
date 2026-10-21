# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Transfer-study leakage, ordering, aggregation and synchronization guards."""

from unittest.mock import patch

import numpy as np
import pytest
import torch

from tools.slackpipe_calibration_transfer import compare_tree, native_order
from tools.slackpipe_transfer_analysis import fit_ols, paired_interval, subset, validate_operations
from tools.slackpipe_transfer_worker import SynchronousStageCollector


def test_two_native_orders_are_distinct_fifo_validated():
    a, b = native_order(2), native_order(4)
    assert a != b
    for order in (a, b):
        for worker in order:
            assert len(worker) == 16
            for stage in {op["stage"] for op in worker}:
                for kind in ("F", "B"):
                    assert [
                        op["microbatch"]
                        for op in worker
                        if op["stage"] == stage and op["kind"] == kind
                    ] == list(range(4))


def test_numerical_failure_is_measured_not_relaxed():
    a = dict(parameters=torch.tensor([1.0, 2.0]))
    b = dict(parameters=torch.tensor([1.0, 2.25]))
    assert compare_tree(a, b) == 0.25
    assert compare_tree(a, a) == 0
    with pytest.raises(ValueError, match="Invalid numerical"):
        compare_tree(a, dict(parameters=torch.tensor([1.0, float("nan")])))


def test_fitter_recovers_known_model_and_rejects_holdout():
    rows = []
    for run in (0, 1):
        for p in (1, 2, 3):
            for s in range(4):
                cost = 1 + p * (s + 1) / 10
                rows.append(
                    dict(
                        run=run,
                        partition=f"P{p}",
                        group=2,
                        stage=s,
                        composition_ms=cost,
                        mean_ms=2 * cost + 0.5 + 0.2 * (s == 0) + 0.4 * (s == 3),
                    )
                )
    result = fit_ols(rows)
    np.testing.assert_allclose(result["coefficients"], [2, 0.5, 0.2, 0.4], atol=1e-12)
    for changed in (dict(partition="P0"), dict(partition="P4"), dict(run=2), dict(group=4)):
        with pytest.raises(ValueError, match="heldout"):
            fit_ols([dict(r, **changed) for r in rows])


def test_rank_deficiency_is_explicit():
    rows = [
        dict(run=r, partition="P1", group=2, stage=s, composition_ms=1, mean_ms=2)
        for r in (0, 1)
        for s in range(4)
    ]
    with pytest.raises(ValueError, match="Unidentifiable"):
        fit_ols(rows)


def test_near_zero_mean_does_not_prove_low_perturbation():
    result = paired_interval([-0.2, 0, 0.2])
    assert result["mean"] == 0
    assert not result["within_one_percent"]
    with pytest.raises(ValueError, match="independent"):
        paired_interval([0] * 30)


def test_complete_operation_coverage_not_just_count():
    order = native_order(2)[0]
    rows = [
        dict(
            iteration=i,
            phase="forward_compute" if op["kind"] == "F" else "backward_compute",
            logical_stage=op["stage"],
            microbatch=op["microbatch"],
            status="recorded",
        )
        for i in range(60)
        for op in order
    ]
    validate_operations(rows, order)
    rows[1] = dict(rows[0])
    with pytest.raises(ValueError, match="duplicate"):
        validate_operations(rows, order)


def test_holdout_categories_do_not_mix_training():
    assert subset(dict(partition="P0", run=0, group=2)) == "native/uniform_control"
    assert subset(dict(partition="P3", run=2, group=2)) == "native/independent_run"
    assert subset(dict(partition="P5", run=2, group=4)) == "order_holdout/partition_and_run"


def test_sync_probe_has_two_drains_only_for_compute():
    collector = object.__new__(SynchronousStageCollector)
    collector.pending = [{}]
    calls = []

    def compute(self, fn, identity):
        calls.append("event_compute")
        return fn()

    with (
        patch("torch.cuda.synchronize", side_effect=lambda: calls.append("sync")),
        patch(
            "megatron.core.pipeline_parallel.slackpipe.stage_measurement.StageEventCollector.compute",
            compute,
        ),
    ):
        assert collector.compute(lambda: 42, {}) == 42
    assert calls == ["sync", "event_compute", "sync"]
    assert collector.pending[0]["added_device_syncs"] == 2
    assert not collector.serialized
