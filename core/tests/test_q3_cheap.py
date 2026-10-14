"""Cheap-test library on a small synthetic stand-in (no licensed data, tiny fits only)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fedcore.q3.cheap import REGISTRY, Config, synthetic_data, window_ols
from fedcore.q3.cheap.data import _earnings_permnos
from fedcore.q3.cheap.linear import boot_mean, demean, ols_cluster, ridge_fit_predict

SMALL = Config(
    dev_meetings=40, inner_min_train=20, inner_test=5, outer_min_train=40, outer_test=5, val_meetings=8,
    n_boot=50, n_perm=5, min_obs=20, horizons=(21,), r1_effects=(0.0, 2.0), r2_min_meetings=20,
    r8_min_meetings=8, synthetic_firms=30, synthetic_meetings=40,
)


@pytest.fixture(scope="module")
def data():
    return synthetic_data(n_firms=30, n_meetings=60, seed=0)


def test_window_ols_recovers_known_loadings(data):
    d = synthetic_data(n_firms=3, n_meetings=10, seed=1)
    ret = 0.5 + 1.2 * d.mkt - 0.3 * d.dy
    d.channels[:, :, 0] = ret[None, :]
    d.observed[:] = True
    end = np.array([300, 400])
    fit = window_ols(d, np.array([0, 2]), end, 63, ("mkt", "dy"), min_obs=20)
    assert np.allclose(fit["coef"], [[0.5, 1.2, -0.3]] * 2, atol=1e-6)
    assert (fit["nobs"] == 63).all()
    few = window_ols(d, np.array([0]), np.array([300]), 63, ("mkt", "dy"), min_obs=100)
    assert np.isnan(few["coef"]).all()


def test_window_ols_respects_day_mask_and_window_end(data):
    d = synthetic_data(n_firms=2, n_meetings=10, seed=2)
    ok = np.ones(d.calendar.size, bool)
    ok[250:260] = False
    fit = window_ols(d, np.array([0]), np.array([270]), 63, ("mkt",), day_ok=ok, min_obs=10)
    assert fit["nobs"][0] <= 53  # ten masked days inside [208, 270]


def test_ols_cluster_matches_lstsq_and_boot_covers_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 2))
    y = x @ [1.0, -2.0] + rng.normal(size=200)
    fit = ols_cluster(x, y, np.repeat(np.arange(20), 10))
    assert np.allclose(fit["coef"], np.linalg.lstsq(x, y, rcond=None)[0])
    b = boot_mean(rng.normal(1.0, 1.0, 100), n=200)
    assert b["ci_low"] < b["estimate"] < b["ci_high"]


def test_demean_removes_group_means():
    g = np.array([0, 0, 1, 1])
    out = demean(g, np.array([1.0, 3.0, 10.0, 14.0]))
    assert np.allclose(out, [-1, 1, -2, 2])


def test_ridge_kinds_decompose_prediction(data):
    from fedcore.q3.cheap.linear import blocks

    z, _ = blocks(data)
    y, s = data.panel.target.astype(float), data.panel.shock.astype(float)
    tr, te = np.arange(600), np.arange(600, 900)
    for kind in ("a", "ab0", "abz"):
        p = ridge_fit_predict(kind, z, s, y, tr, te)
        assert np.allclose(p["mu"], p["a"] + p["b"] * s[te])
    assert np.allclose(ridge_fit_predict("a", z, s, y, tr, te)["b"], 0.0)


def test_earnings_map_to_linked_permno():
    fq = pd.DataFrame({"gvkey": ["1", "1", "2"], "rdq": pd.to_datetime(["2000-01-10", "2005-01-10", "2000-02-01"])})
    link = pd.DataFrame({"gvkey": ["1", "1", "2"], "lpermno": [10, 11, 20], "linkprim": ["P", "P", "C"],
                         "linkdt": pd.to_datetime(["1990-01-01", "2003-01-01", "1990-01-01"]),
                         "linkenddt": pd.to_datetime(["2002-12-31", None, None])})
    out = _earnings_permnos(fq, link).sort_values("rdq")
    assert out["permno"].tolist() == [10, 20, 11]


def test_synthetic_data_is_internally_consistent(data):
    from fedcore.q3.cheap.audits import a6

    res = a6(data, SMALL)
    failed = res.tables["checks"].query("not passed")
    assert failed.empty, failed.to_string()


def test_a7_labels_never_cross_cutoff_when_usable(data):
    from fedcore.q3.cheap.audits import a7

    t = a7(data, SMALL).tables["windows"]
    assert (t["usable"] <= t["windows"]).all()
    assert (t.loc[t.label == "history_only", "label_crosses_cutoff"] == 0).all()


def test_r1_detects_a_large_planted_effect(data):
    from fedcore.q3.cheap.premise import r1

    t = r1(data, SMALL).tables["power"]
    big = t[(t.planted_sd_b == 2.0) & (t.design == "outer")]
    assert big["b_hat_corr_with_planted"].min() > 0.3


@pytest.mark.parametrize("check_id", sorted(c for c in REGISTRY if c[0] in "ACRS"))
def test_every_check_runs_on_the_stand_in(data, check_id):
    chk = REGISTRY[check_id]
    res = chk.fn(data if chk.uses_real else None, SMALL)
    assert isinstance(res.metrics, dict)
    for name, table in res.tables.items():
        assert isinstance(table, pd.DataFrame), name
