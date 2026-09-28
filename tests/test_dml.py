import numpy as np
import pandas as pd
import pytest

from fedci.eval import (leave_one_out, make_nuisance, ols_event_study, partially_linear_dml)


def _synthetic(n=600, theta=2.0, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    s = X @ np.array([0.5, -0.3, 0.2]) + rng.normal(size=n)
    y = theta * s + X @ np.array([1.0, -0.7, 0.4]) + rng.normal(size=n)
    return pd.DataFrame(dict(
        announcement_date=pd.date_range("1990-01-01", periods=n, freq="45D"),
        y=y, s=s, x0=X[:, 0], x1=X[:, 1], x2=X[:, 2]))


def _fit(df=None, **kw):
    df = _synthetic() if df is None else df
    args = dict(y="y", s="s", x=["x0", "x1", "x2"], min_train=200, test_size=100,
                n_boot=100, seed=0)
    args.update(kw)
    return partially_linear_dml(df, **args)


def test_plr_recovers_theta_on_synthetic():
    fit = _fit()
    assert abs(fit.estimate.theta - 2.0) < 0.2
    assert fit.estimate.n_meetings == 400
    assert fit.n_meetings_discarded == 200
    assert fit.n_meetings_total == 600
    assert fit.estimate.ci_low < 2.0 < fit.estimate.ci_high


def test_theta_is_the_residual_moment():
    fit = _fit()
    r = fit.residuals
    manual = float(r.s_resid @ r.y_resid) / float(r.s_resid @ r.s_resid)
    assert manual == pytest.approx(fit.estimate.theta)


def test_cross_fit_is_strictly_temporal():
    fit = _fit()
    assert (fit.folds.train_end < fit.folds.test_start).all()
    assert fit.estimate.n_meetings + fit.n_meetings_discarded == fit.n_meetings_total
    assert fit.residuals.announcement_date.min() == fit.folds.test_start.min()


def test_inference_and_bootstrap_bracket_theta():
    fit = _fit(n_boot=200)
    assert fit.estimate.se > 0
    assert fit.estimate.se_type == "HC1"
    b = fit.bootstrap
    assert b["n_meetings"] == fit.estimate.n_meetings
    assert b["se"] > 0
    assert b["ci_low"] < fit.estimate.theta < b["ci_high"]


def test_hac_se_over_meeting_order():
    fit = _fit(se="HAC", hac_lag=3, n_boot=0)
    assert fit.estimate.se_type == "HAC(3)"
    assert fit.estimate.se > 0
    with pytest.raises(ValueError):
        _fit(se="HAC", hac_lag=0, n_boot=0)


def test_incomplete_rows_are_dropped_and_counted():
    df = _synthetic()
    df.loc[[5, 50, 300], "x1"] = np.nan
    fit = _fit(df)
    assert fit.n_meetings_dropped == 3
    assert fit.n_meetings_total == 597
    assert fit.estimate.n_meetings + fit.n_meetings_discarded == 597


def test_ols_baseline_recovers_theta():
    df = _synthetic()
    est = ols_event_study(df, name="ols", y="y", s="s", x=("x0", "x1", "x2"))
    assert est.theta == pytest.approx(2.0, abs=0.1)
    assert est.se > 0


def test_leave_one_out_matches_manual_refit():
    fit = _fit()
    loo = leave_one_out(fit.residuals)
    r = fit.residuals.drop(index=0)
    manual = float(r.s_resid @ r.y_resid) / float(r.s_resid @ r.s_resid)
    assert manual == pytest.approx(loo.theta_loo.iloc[0])


def test_make_nuisance_validates_name():
    with pytest.raises(ValueError):
        make_nuisance("gpr")
