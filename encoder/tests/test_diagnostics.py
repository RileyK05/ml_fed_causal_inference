"""Diagnostics on saved predictions: MSE split, b grades, persistence. No fitting."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fedcore.q3.synthetic import make_synthetic_panel
from fedenc import diagnostics as dg


@pytest.fixture
def frame():
    panel = make_synthetic_panel(n_firms=80, n_meetings=24, seed=0)
    shock, traits = dg.panel_frames(panel)
    rng = np.random.default_rng(0)
    meeting = pd.DatetimeIndex(panel.meeting)
    target = 0.3 + panel.b_true * panel.shock + rng.normal(0, 0.05, panel.target.shape)
    df = pd.DataFrame({"meeting": meeting, "firm_id": panel.firm_id, "seed": 0,
                       "fold": (np.searchsorted(np.unique(meeting), meeting) // 6),
                       "a_hat": 0.3, "b_hat": panel.b_true, "target": target})
    df["mu"] = df["a_hat"] + df["b_hat"] * meeting.map(shock).to_numpy()
    df["s"] = df["meeting"].map(shock)
    return df.merge(traits, on=["meeting", "firm_id"])


def test_mse_split_adds_up_and_constant_predictions_have_no_cross_firm_gain(frame):
    parts = dg.meeting_parts(frame)
    np.testing.assert_allclose(parts["total"], parts["mean_part"] + parts["cs"], atol=1e-9)
    flat = frame.assign(b_hat=frame.groupby("meeting")["b_hat"].transform("mean"))
    flat["mu"] = flat["a_hat"] + flat["b_hat"] * flat["s"]
    flat_parts = dg.meeting_parts(flat)
    np.testing.assert_allclose(flat_parts["cs"], flat_parts["cs_zero"], atol=1e-9)
    # The true b explains the cross-firm part; the a-only version cannot.
    assert parts["cs"].mean() < parts["cs_a_only"].mean()


def test_oracle_b_grades_one_and_persists(frame):
    grades = dg.b_grades(frame)
    est = float(grades["raw"].split()[0])
    assert est == pytest.approx(1.0, abs=0.05)
    assert float(grades["minus_a"].split()[0]) == pytest.approx(1.0, abs=0.05)
    corr, flips = dg.persistence(frame.assign(b_hat=frame.groupby("firm_id")["b_hat"].transform("first")))
    assert corr == pytest.approx(1.0) and flips == 0.0


def test_paired_range_of_a_constant_difference_is_exact():
    left = pd.Series([3.0, 4.0, 5.0])
    est, lo, hi = dg.paired_range(left, left - 0.5, n=200)
    assert est == lo == hi == pytest.approx(0.5)


def test_clustered_ols_recovers_coefficients():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 2))
    y = X @ np.array([2.0, -1.0]) + rng.normal(0, 0.01, 400)
    coef, se = dg.ols_clustered(X, y, np.repeat(np.arange(40), 10))
    np.testing.assert_allclose(coef, [2.0, -1.0], atol=0.01)
    assert (se > 0).all()
