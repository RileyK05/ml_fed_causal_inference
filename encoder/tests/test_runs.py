"""Shared run saving and the b grade. Tabular baselines only, nothing heavy."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fedcore.q3.baselines import BASELINES
from fedcore.q3.synthetic import make_synthetic_panel
from fedcore.results import ResultStore
from fedenc.runs import b_grade, save_runs


def test_b_grade_is_one_for_oracle_and_nan_for_constant_b():
    panel = make_synthetic_panel(n_firms=60, n_meetings=40, seed=0)
    meeting = pd.DatetimeIndex(panel.meeting)
    shock = pd.Series(panel.shock, index=meeting).groupby(level=0).first()
    noise_free = 0.3 + panel.b_true * panel.shock  # synthetic a_true is correlated with b_true
    df = pd.DataFrame({"seed": 0, "meeting": meeting, "target": noise_free, "b_hat": panel.b_true})
    assert b_grade(df, shock)["b_slope"] == pytest.approx(1.0, abs=1e-6)
    flat = df.assign(b_hat=1.0)
    assert np.isnan(b_grade(flat, shock)["b_slope"])


def test_save_runs_saves_one_run_per_arm_and_a_comparison(tmp_path):
    panel = make_synthetic_panel(n_firms=12, n_meetings=20, seed=0)
    store = ResultStore(tmp_path)
    makers = {"capm_pooled": BASELINES["capm_pooled"], "ridge_interact": BASELINES["ridge_interact"]}
    summary = save_runs(panel, makers, prefix="t", params=dict(note="test"), min_train=12, test_size=4,
                        seeds=(0,), store=store, datasets=[], role="robustness")
    assert list(summary["arm"]) == ["capm_pooled", "ridge_interact"]
    assert {"run_id", "path", "mse", "rank_ic", "b_slope", "b_t"} <= set(summary.columns)
    manifests = [json.loads(p.read_text()) for p in tmp_path.rglob("manifest.json")]
    assert sorted(m["name"] for m in manifests) == ["t-capm_pooled", "t-comparison", "t-ridge_interact"]
    assert {m["role"] for m in manifests} == {"robustness"}
    arm = next(m for m in manifests if m["name"] == "t-capm_pooled")
    assert arm["params"]["arm"] == "capm_pooled" and arm["params"]["note"] == "test"
    assert np.isfinite(arm["metrics"]["mse_per_meeting"])
