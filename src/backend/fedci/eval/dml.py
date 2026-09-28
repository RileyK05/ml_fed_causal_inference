"""Partially linear DML (PLR) for continuous treatments -- the Q1 causal estimator.

Design reference: docs/methods.md "Q1 - Keep the causal estimator focused"; spec in
docs/question.md. Model:

    Y_t = theta * s_t + g(X_t) + eps_t

Nuisances l(X) = E[Y|X] and m(X) = E[s|X] are scored out of sample and theta solves the
orthogonal moment

    sum_t s_resid_t * (y_resid_t - theta * s_resid_t) = 0.

The moment is written out here rather than delegated to a DML library so the inference
assumptions stay auditable (methods.md). Project rules encoded here:

- Cross-fitting is chronological via eval.walk_forward: never random folds, never a
  meeting on both sides of a split. Meetings in the initial training window are never
  scored -- they are the price of forward-only nuisance fits and must be reported.
- Inference is over meetings: HC1/HAC on the cross-fitted score plus a whole-meeting
  pairs bootstrap (eval.meeting_bootstrap). Classical random-fold DML standard errors
  do not automatically justify a temporal procedure.
- Nuisance learners start simple (ridge/elastic net, optional shallow trees).
  sklearn/statsmodels are the optional `model` extra and are imported lazily so the
  base install can still import this module.

Units are the caller's: theta is units of Y per unit of s.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np
import pandas as pd

from fedci.eval.protocol import Fold, meeting_bootstrap, walk_forward

NUISANCES = ("ridge", "enet", "rf")


@dataclass(frozen=True)
class Estimate:
    name: str
    theta: float
    se: float
    ci_low: float
    ci_high: float
    se_type: str
    n_meetings: int
    extras: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        return dict(estimator=self.name, theta=self.theta, se=self.se, ci_low=self.ci_low,
                    ci_high=self.ci_high, se_type=self.se_type, n_meetings=self.n_meetings)


@dataclass(frozen=True)
class PLRResult:
    estimate: Estimate
    n_meetings_total: int        # complete cases seen by the split
    n_meetings_discarded: int    # initial training window, never scored
    n_meetings_dropped: int      # rows missing y, s or X before splitting
    sd_s_resid: float            # residual treatment variation
    sd_y_resid: float
    r2_y: float                  # out-of-fold R^2 of E[Y|X] on the scored rows
    r2_s: float                  # out-of-fold R^2 of E[s|X] on the scored rows
    residuals: pd.DataFrame      # scored rows with l_hat, m_hat, y_resid, s_resid
    folds: pd.DataFrame
    bootstrap: dict | None = None


def make_nuisance(name: str = "ridge", *, seed: int = 0, **kwargs):
    """Sklearn regressor for one nuisance function; linear models get standardized features."""
    kwargs = dict(kwargs)
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=kwargs.pop("alpha", 1.0), **kwargs))
    if name == "enet":
        alpha = kwargs.pop("alpha", 0.01)
        l1_ratio = kwargs.pop("l1_ratio", 0.5)
        max_iter = kwargs.pop("max_iter", 10_000)
        return make_pipeline(StandardScaler(), ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                                                          max_iter=max_iter, **kwargs))
    if name == "rf":
        kwargs.setdefault("n_estimators", 300)
        kwargs.setdefault("min_samples_leaf", 5)
        return RandomForestRegressor(random_state=seed, **kwargs)
    raise ValueError(f"nuisance must be one of {NUISANCES}, got {name!r}")


def cross_fit_nuisances(df: pd.DataFrame, *, y: str, s: str, x: list[str], folds: list[Fold],
                        nuisance_y, nuisance_s, date: str = "announcement_date"):
    """Out-of-fold l(X), m(X) on the rows of a temporal cross-fit.

    Rows that never fall in a test fold (the initial training window) are not returned.
    `df` must have a unique index. Returns (scored, fold_table): `scored` keeps the input
    columns and adds l_hat, m_hat, y_resid, s_resid.
    """
    x = list(x)
    l_hat = pd.Series(np.nan, index=df.index, dtype=float)
    m_hat = pd.Series(np.nan, index=df.index, dtype=float)
    rows = []
    for f in folds:
        tr, te = f.split(df, meeting_col=date)
        if not len(te):
            continue
        nuisance_y.fit(tr[x], tr[y])
        nuisance_s.fit(tr[x], tr[s])
        l_hat.loc[te.index] = nuisance_y.predict(te[x])
        m_hat.loc[te.index] = nuisance_s.predict(te[x])
        rows.append(dict(fold=f.k, n_train=len(tr), train_start=f.train.min(), train_end=f.train.max(),
                         n_test=len(te), test_start=f.test.min(), test_end=f.test.max()))
    scored = df.loc[l_hat.notna()].copy()
    scored["l_hat"] = l_hat.loc[scored.index]
    scored["m_hat"] = m_hat.loc[scored.index]
    scored["y_resid"] = scored[y] - scored["l_hat"]
    scored["s_resid"] = scored[s] - scored["m_hat"]
    return scored.reset_index(drop=True), pd.DataFrame(rows)


def _theta(y_resid: np.ndarray, s_resid: np.ndarray) -> float:
    denom = float(s_resid @ s_resid)
    if denom <= 0:
        raise ValueError("no residual treatment variation: s_resid is constant zero")
    return float(s_resid @ y_resid) / denom


def _score_inference(y_resid: np.ndarray, s_resid: np.ndarray, theta: float,
                     se_type: str = "HC1", hac_lag: int = 0) -> float:
    """SE of theta from the cross-fitted score psi_t = s_resid_t (y_resid_t - theta s_resid_t).

    se = sqrt(LRV(psi)) / sum(s_resid^2), where LRV is the long-run variance of the score
    sum: HC1 scales the contemporaneous variance by n/(n-1); HAC uses a Bartlett kernel
    over meeting order with `hac_lag` lags.
    """
    n = len(s_resid)
    psi = s_resid * (y_resid - theta * s_resid)
    if se_type == "HC1":
        long_run = (n / (n - 1)) * float(psi @ psi)
    elif se_type == "HAC":
        if hac_lag < 1:
            raise ValueError("HAC needs hac_lag >= 1")
        long_run = float(psi @ psi)
        for h in range(1, hac_lag + 1):
            long_run += 2 * (1 - h / (hac_lag + 1)) * float(psi[h:] @ psi[:-h])
        long_run = max(long_run, 0.0)
    else:
        raise ValueError(f"se must be 'HC1' or 'HAC', got {se_type!r}")
    return float(np.sqrt(long_run)) / float(s_resid @ s_resid)


def ols_event_study(df: pd.DataFrame, *, name: str, y: str, s: str, x: tuple[str, ...] = (),
                    date: str = "announcement_date", se: str = "HC1", hac_lag: int = 0,
                    level: float = 0.95) -> Estimate:
    """Event-study baseline: OLS of Y on s (plus X), robust covariance over meetings.

    Give it the same frame the DML scored for a like-for-like comparison.
    """
    import statsmodels.api as sm

    cols = [s, *x]
    design = sm.add_constant(df[cols].astype(float), has_constant="add")
    cov_kwds = dict(maxlags=hac_lag) if se == "HAC" else {}
    res = sm.OLS(df[y].astype(float), design).fit(cov_type=se, cov_kwds=cov_kwds)
    lo, hi = res.conf_int(alpha=1 - level).loc[s]
    return Estimate(name=name, theta=float(res.params[s]), se=float(res.bse[s]),
                    ci_low=float(lo), ci_high=float(hi), se_type=se,
                    n_meetings=int(df[date].nunique()), extras=dict(r2=float(res.rsquared)))


def partially_linear_dml(df: pd.DataFrame, *, y: str, s: str, x: list[str],
                         date: str = "announcement_date", min_train: int, test_size: int,
                         step: int | None = None, embargo: int = 0,
                         nuisance: str = "ridge", nuisance_kwargs: dict | None = None,
                         se: str = "HC1", hac_lag: int = 0, n_boot: int = 400, seed: int = 0,
                         level: float = 0.95) -> PLRResult:
    """Fit Y = theta*s + g(X) + e by cross-fitted partially linear DML on whole meetings.

    Complete cases only: rows missing y, s or any x are dropped and counted (a gap stays a
    gap; nothing is filled). The first `min_train` meetings train the nuisances and are
    never scored -- n_meetings_discarded documents the resulting estimation population.

    Inference: normal CI on the cross-fitted score (HC1, or HAC over meeting order when
    se="HAC" and hac_lag>=1). n_boot>0 additionally runs a whole-meeting pairs bootstrap
    of the same score (nuisance fits are not refit per draw).
    """
    cols = [date, y, s, *x]
    full = df[cols].reset_index(drop=True)
    complete = full.dropna().reset_index(drop=True)
    n_dropped = len(full) - len(complete)
    meetings = pd.DatetimeIndex(sorted(complete[date].unique()))
    folds = walk_forward(meetings, min_train=min_train, test_size=test_size, step=step, embargo=embargo)
    scored, fold_table = cross_fit_nuisances(
        complete, y=y, s=s, x=list(x), folds=folds, date=date,
        nuisance_y=make_nuisance(nuisance, seed=seed, **(nuisance_kwargs or {})),
        nuisance_s=make_nuisance(nuisance, seed=seed, **(nuisance_kwargs or {})),
    )
    if len(scored) < 3:
        raise ValueError(f"only {len(scored)} scored meetings; lower min_train or check the sample")

    y_resid = scored["y_resid"].to_numpy()
    s_resid = scored["s_resid"].to_numpy()
    theta = _theta(y_resid, s_resid)
    se_v = _score_inference(y_resid, s_resid, theta, se_type=se, hac_lag=hac_lag)
    z = NormalDist().inv_cdf((1 + level) / 2)
    se_label = se if se == "HC1" else f"HAC({hac_lag})"

    boot = None
    if n_boot and n_boot > 0:
        boot = meeting_bootstrap(
            scored, lambda d: _theta(d["y_resid"].to_numpy(), d["s_resid"].to_numpy()),
            meeting_col=date, n=n_boot, seed=seed, level=level)

    def _r2(actual: str, pred: str) -> float:
        a = scored[actual].to_numpy()
        return float(1 - ((a - scored[pred].to_numpy()) ** 2).sum() / ((a - a.mean()) ** 2).sum())

    estimate = Estimate(
        name=f"plr_dml({nuisance})", theta=theta, se=se_v,
        ci_low=theta - z * se_v, ci_high=theta + z * se_v, se_type=se_label,
        n_meetings=len(scored),
        extras=dict(sd_s_resid=float(s_resid.std(ddof=1)), r2_y=_r2(y, "l_hat"), r2_s=_r2(s, "m_hat")),
    )
    return PLRResult(
        estimate=estimate,
        n_meetings_total=len(complete),
        n_meetings_discarded=len(complete) - len(scored),
        n_meetings_dropped=n_dropped,
        sd_s_resid=float(s_resid.std(ddof=1)),
        sd_y_resid=float(y_resid.std(ddof=1)),
        r2_y=_r2(y, "l_hat"),
        r2_s=_r2(s, "m_hat"),
        residuals=scored,
        folds=fold_table,
        bootstrap=boot,
    )


def leave_one_out(residuals: pd.DataFrame, *, y: str = "y_resid", s: str = "s_resid",
                  date: str = "announcement_date") -> pd.DataFrame:
    """Sensitivity of theta to dropping each meeting, from the cross-fitted score."""
    yv = residuals[y].to_numpy()
    sv = residuals[s].to_numpy()
    s2 = sv @ sv
    num = sv @ yv
    theta = num / s2
    with np.errstate(divide="ignore", invalid="ignore"):
        loo = (num - sv * yv) / (s2 - sv ** 2)
    return pd.DataFrame({date: residuals[date].to_numpy(), "theta": theta,
                         "theta_loo": loo, "delta_loo": loo - theta})


__all__ = ["NUISANCES", "Estimate", "PLRResult", "make_nuisance", "cross_fit_nuisances",
           "ols_event_study", "partially_linear_dml", "leave_one_out"]
