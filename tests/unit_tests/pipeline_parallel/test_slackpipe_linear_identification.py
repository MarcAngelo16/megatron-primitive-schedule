# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Offline checks for frozen identification and observation integrity."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from tools import slackpipe_linear_boundary_check
from tools.slackpipe_linear_analysis import features, paired_samples, solve, subset, training
from tools.slackpipe_linear_identification import counts, matrix_report


def rows():
    result = []
    for run in (0, 1):
        for pattern in ("M", "*", "-", "M*", "*-", "-M", "M*-M", "M*-M*-"):
            c = counts(pattern)
            result.append(
                dict(
                    counts=c,
                    role="interior",
                    run=run,
                    design_role="train",
                    statistics={"mean": float(np.dot(c, [2, 3, 4]) + 0.5)},
                )
            )
    return result


def test_scaled_h2_and_count_basis_are_identical():
    unit = [7, 11, 13]
    fit = solve(rows(), unit, "H2")
    assert fit["status"] == "identified"
    np.testing.assert_allclose(fit["effective_theta"], [2, 3, 4])
    assert fit["coefficients"][-1] == pytest.approx(0.5)
    for row in rows():
        assert np.dot(features(row, unit, "H2"), fit["coefficients"]) == pytest.approx(
            np.dot(row["counts"] + [1], fit["count_parameterization"])
        )
    reanchored = solve(rows(), [1, 2, 1], "H2")
    np.testing.assert_allclose(reanchored["effective_theta"], fit["effective_theta"])
    assert not np.allclose(reanchored["coefficients"], fit["coefficients"])


@pytest.mark.parametrize(
    "change", [{"run": 2}, {"design_role": "validation"}, {"design_role": "check"}]
)
def test_no_heldout_leakage(change):
    samples = rows()
    samples[-1].update(change)
    assert not training(samples[-1])
    with pytest.raises(ValueError, match="Heldout"):
        solve(samples, [1, 2, 3], "H2")


def test_same_proportions_do_not_identify_type_slopes():
    samples = [
        dict(
            counts=[n, n, n],
            role="interior",
            run=0,
            design_role="train",
            statistics={"mean": 9 * n + 0.5},
        )
        for n in range(1, 9)
    ]
    fit = solve(samples, [1, 2, 3], "H2")
    assert fit["design"]["rank"] == 2
    assert fit["status"].startswith("unidentifiable")
    # Two profiler values are not two independent composition equations.
    assert matrix_report([[1, 1, 1, 1], [2, 2, 2, 1]])["rank"] == 2


def test_raw_samples_retain_spike_and_reject_duplicate():
    data = [dict(iteration=i, phase="B", measured=True, event_ms=1.0) for i in range(30, 60)]
    data[3]["event_ms"] = 100
    result = paired_samples(data, "B", "event_ms")
    assert result["mean"] == pytest.approx(4.3)
    assert result["max"] == 100
    data[-1]["iteration"] = 58
    with pytest.raises(ValueError, match="Missing/duplicate"):
        paired_samples(data, "B", "event_ms")


def test_independent_run_and_uniform_stay_heldout():
    assert subset(dict(run=2, design_role="train")) == "independent_run"
    assert subset(dict(run=2, design_role="validation")) == "unseen_case_and_run"
    assert subset(dict(run=0, design_role="check")) == "uniform_control"


def test_boundary_terms_require_distinct_roles():
    samples = rows()
    assert solve(samples, [1, 2, 3], "H2-boundary")["status"].startswith("unidentifiable")
    expanded = []
    for row in samples:
        for role, extra in (("interior", 0), ("first", 5), ("last", 7)):
            expanded.append(
                dict(row, role=role, statistics={"mean": row["statistics"]["mean"] + extra})
            )
    fit = solve(expanded, [1, 2, 3], "H2-boundary")
    np.testing.assert_allclose(fit["coefficients"][-3:], [0.5, 5, 7])


def test_nonbatch_diagnostic_does_not_leak_into_default_configuration():
    module = slackpipe_linear_boundary_check.slackpipe_eval_worker
    original = lambda: SimpleNamespace(batch_p2p_comm=True, overlap_p2p_comm=False)
    observed = []
    with (
        patch.object(module, "transformer_config", original),
        patch.object(
            slackpipe_linear_boundary_check.slackpipe_linear_worker,
            "main",
            side_effect=lambda: observed.append(module.transformer_config()),
        ),
    ):
        slackpipe_linear_boundary_check.main()
        assert module.transformer_config().batch_p2p_comm is True
    assert observed[0].batch_p2p_comm is False
    assert observed[0].overlap_p2p_comm is False


def test_entry_term_requires_q_variation_at_fixed_composition():
    base = [dict(r, q=1) for r in rows()]
    assert solve(base, [1, 1, 1], "H2-entry")["status"].startswith("unidentifiable")
    varied = []
    for row in rows():
        for q in range(1, min(2, sum(row["counts"])) + 1):
            varied.append(
                dict(row, q=q, statistics={"mean": row["statistics"]["mean"] + 0.3 * (q - 1)})
            )
    fit = solve(varied, [1, 1, 1], "H2-entry")
    assert fit["design"]["rank"] == 5
    np.testing.assert_allclose(fit["coefficients"], [2, 3, 4, 0.5, 0.3])
