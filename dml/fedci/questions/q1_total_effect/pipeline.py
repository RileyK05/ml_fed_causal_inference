"""Q1 pipeline: aggregate equity response to a monetary-policy surprise (spec.py).

    fedci run q1                                 # primary: announcement-window SP500 on STMT
    fedci run q1 treatment=MP1 role=robustness   # shock-decomposition variant
    fedci run q1 outcome=etf_day0 role=robustness  # day-0 ETF window, 2023+ meetings

`estimate(**params)` fits without saving (used by fedci.api and by run()); `run()` only
assembles the ResultStore folder: estimates (OLS baselines beside PLR-DML), a sample
audit, the temporal folds, per-meeting residuals with leave-one-meeting-out sensitivity,
and the residualized fit. Samples and units come from samples.py.

Params (CLI values arrive as strings; all are coerced here):
    outcome        usmpd_sp500 (default) | etf_day0
    treatment      STMT (default) | MP1 | ME        -- surprise used as s
    outcome_ticker SPY (default)                    -- ETF for outcome=etf_day0
    features       comma-separated control subset (default: all sample controls)
    nuisance       ridge (default) | enet | rf
    alpha          nuisance regularization (default: learner's own)
    min_train      meetings in the first training window (default: max(10, n/3))
    test_size      meetings per test block (default: max(5, n/12))
    step           fold advance (default: test_size)
    embargo        meetings dropped between train and test (default 0; same-day outcome
                   windows cannot reach the next meeting)
    se             HC1 (default) | HAC              -- analytic inference on the score
    hac_lag        HAC lags in meeting order (default 4)
    n_boot         whole-meeting bootstrap draws (default 400; 0 disables)
    level          CI level (default 0.95)
    role           primary (default) | robustness | exploratory
    name           run name override
"""
from __future__ import annotations

import pandas as pd

from fedci.api.figures import residualized_fit_figure
from fedci.eval import (leave_one_out, ols_event_study, partially_linear_dml)
from fedci.questions.q1_total_effect.samples import OUTCOMES, SURPRISES, etf_day0_sample, usmpd_window_sample
from fedcore.questions.q1_total_effect import SPEC
from fedcore.results import ResultStore

_INT = ("min_train", "test_size", "step", "embargo", "hac_lag", "n_boot", "seed")
_FLOAT = ("alpha", "level")
_STR = ("outcome", "treatment", "outcome_ticker", "nuisance", "se", "role", "name")
_LIST = ("features",)

_SOURCES = {
    "usmpd_sp500": ["usmpd_statements", "mps_surprises", "treasury_1y"],
    "etf_day0": ["event_study_table"],
}


def _parse(params: dict) -> dict:
    unknown = set(params) - set(_INT) - set(_FLOAT) - set(_STR) - set(_LIST)
    if unknown:
        raise ValueError(f"unknown params {sorted(unknown)}; allowed: {sorted(_INT + _FLOAT + _STR + _LIST)}")
    p = dict(outcome="usmpd_sp500", treatment="STMT", outcome_ticker="SPY", features=None,
             nuisance="ridge", alpha=None, min_train=None, test_size=None, step=None, embargo=0,
             se="HC1", hac_lag=4, n_boot=400, seed=0, level=0.95, role="primary", name=None)
    for k, v in params.items():
        if v is None:
            continue
        if k in _LIST:
            p[k] = [t for t in (v.split(",") if isinstance(v, str) else [str(f) for f in v]) if t]
        else:
            p[k] = int(v) if k in _INT else float(v) if k in _FLOAT else str(v)
    if p["outcome"] not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES}")
    if p["treatment"] not in SURPRISES:
        raise ValueError(f"treatment must be one of {SURPRISES}")
    if p["se"] not in ("HC1", "HAC"):
        raise ValueError("se must be 'HC1' or 'HAC'")
    if p["features"] is not None and not p["features"]:
        raise ValueError("features must be a non-empty subset of the sample controls")
    return p


def estimate(**params) -> dict:
    """Fit Q1 without saving. Returns resolved (parsed params incl. computed
    min_train/test_size/step), sources, metrics, estimates/audit/folds/residuals/influence
    (DataFrames), figure (residualized fit) and notes."""
    p = _parse(params)
    if p["outcome"] == "usmpd_sp500":
        df = usmpd_window_sample(p["treatment"])
    else:
        df = etf_day0_sample(p["treatment"], p["outcome_ticker"])
    controls = [c for c in df.columns if c not in ("announcement_date", "y", "s")]
    if p["features"] is None:
        x = controls
    else:
        bad = [f for f in p["features"] if f not in controls]
        if bad:
            raise ValueError(f"unknown features {bad}; allowed: {controls}")
        x = list(p["features"])

    n_est = int(len(df.dropna(subset=["y", "s", *x])))
    min_train = p["min_train"] if p["min_train"] is not None else max(10, round(n_est / 3))
    test_size = p["test_size"] if p["test_size"] is not None else max(5, round(n_est / 12))
    resolved = dict(p, min_train=min_train, test_size=test_size,
                    step=p["step"] if p["step"] is not None else test_size)

    nuisance_kwargs = {"alpha": resolved["alpha"]} if resolved["alpha"] is not None else None
    fit = partially_linear_dml(
        df, y="y", s="s", x=x, min_train=min_train, test_size=test_size,
        step=resolved["step"], embargo=resolved["embargo"], nuisance=resolved["nuisance"],
        nuisance_kwargs=nuisance_kwargs, se=resolved["se"], hac_lag=resolved["hac_lag"],
        n_boot=resolved["n_boot"], seed=resolved["seed"], level=resolved["level"],
    )
    scored = fit.residuals
    ols_uni = ols_event_study(scored, name="ols y~s", y="y", s="s",
                              se=resolved["se"], hac_lag=resolved["hac_lag"], level=resolved["level"])
    ols_ctl = ols_event_study(scored, name="ols y~s+X", y="y", s="s", x=tuple(x),
                              se=resolved["se"], hac_lag=resolved["hac_lag"], level=resolved["level"])

    rows = [ols_uni.as_row(), ols_ctl.as_row(), fit.estimate.as_row()]
    if fit.bootstrap:
        b = fit.bootstrap
        rows.append(dict(estimator=f"{fit.estimate.name} (bootstrap)", theta=fit.estimate.theta,
                         se=b["se"], ci_low=b["ci_low"], ci_high=b["ci_high"],
                         se_type="meeting bootstrap", n_meetings=b["n_meetings"]))
    estimates = pd.DataFrame(rows)

    loo = leave_one_out(scored)
    resid = scored[["announcement_date", "y", "s", "l_hat", "m_hat", "y_resid", "s_resid"]] \
        .merge(loo.drop(columns=["theta"]), on="announcement_date")
    resid["score"] = resid["s_resid"] * (resid["y_resid"] - fit.estimate.theta * resid["s_resid"])
    influence = resid.reindex(resid["delta_loo"].abs().sort_values(ascending=False).index) \
        .head(10)[["announcement_date", "theta_loo", "delta_loo", "y_resid", "s_resid"]]

    missing_dates = df.loc[df[["y", "s", *x]].isna().any(axis=1), "announcement_date"]
    zero_dates = df.loc[df["y"] == 0, "announcement_date"]
    audit = pd.DataFrame([
        dict(item="outcome", value=f"{p['outcome']} (y in percent)"),
        dict(item="treatment", value=f"{p['treatment']} (s per 10bp of 1y-yield-equivalent surprise)"),
        dict(item="controls", value=", ".join(x)),
        dict(item="n_meetings_in_sample", value=len(df)),
        dict(item="n_meetings_complete", value=fit.n_meetings_total),
        dict(item="n_meetings_dropped_incomplete", value=fit.n_meetings_dropped),
        dict(item="n_meetings_discarded_training", value=fit.n_meetings_discarded),
        dict(item="n_meetings_scored", value=fit.estimate.n_meetings),
        dict(item="scored_date_range", value=f"{scored.announcement_date.min().date()} .. {scored.announcement_date.max().date()}"),
        dict(item="mean_y_scored", value=round(float(scored.y.mean()), 4)),
        dict(item="sd_y_scored", value=round(float(scored.y.std(ddof=1)), 4)),
        dict(item="mean_s_scored", value=round(float(scored.s.mean()), 4)),
        dict(item="sd_s_scored", value=round(float(scored.s.std(ddof=1)), 4)),
        dict(item="sd_s_resid", value=round(fit.sd_s_resid, 4)),
        dict(item="r2_y_nuisance_oof", value=round(fit.r2_y, 4)),
        dict(item="r2_s_nuisance_oof", value=round(fit.r2_s, 4)),
        dict(item="zero_outcome_meetings", value=", ".join(str(d.date()) for d in zero_dates) or "(none)"),
        dict(item="incomplete_meetings", value=", ".join(str(d.date()) for d in missing_dates) or "(none)"),
    ])

    fig = residualized_fit_figure(
        scored, fit.estimate.theta,
        title=f"{SPEC.id}: residualized Y vs residualized surprise ({p['outcome']})",
        x_title="s residual (per 10bp)", y_title="y residual (percent)")

    e = fit.estimate
    metrics = dict(
        theta=e.theta, se=e.se, ci_low=e.ci_low, ci_high=e.ci_high, se_type=e.se_type,
        n_meetings=e.n_meetings, n_meetings_discarded=fit.n_meetings_discarded,
        sd_s_resid=fit.sd_s_resid, r2_y=fit.r2_y, r2_s=fit.r2_s,
        boot_se=(fit.bootstrap or {}).get("se"), boot_ci_low=(fit.bootstrap or {}).get("ci_low"),
        boot_ci_high=(fit.bootstrap or {}).get("ci_high"),
        ols_theta=ols_uni.theta, ols_controls_theta=ols_ctl.theta,
    )
    notes = (
        f"theta: percent return per 10bp of the {p['treatment']} surprise. "
        f"Complete cases only; {fit.n_meetings_dropped} meeting(s) dropped for missing inputs. "
        f"Nuisances fit on the first {fit.n_meetings_discarded} meetings and never scored on them. "
        f"Inference over meetings on the cross-fitted score ({e.se_type}); the bootstrap resamples "
        f"whole meetings without refitting nuisances. "
        f"OLS baselines use the same scored meetings as the DML."
    )
    if len(zero_dates):
        notes += f" Zero-outcome meetings present ({len(zero_dates)}) -- audit before interpreting."

    return dict(resolved=resolved, sources=list(_SOURCES[p["outcome"]]), metrics=metrics,
                estimates=estimates, audit=audit, folds=fit.folds, residuals=resid,
                influence=influence, figure=fig, notes=notes)


def run(store: ResultStore, **params):
    e = estimate(**params)
    r = e["resolved"]
    return store.save(
        SPEC, r["name"] or f"PLR-DML {r['outcome']} {r['treatment']} {r['nuisance']}",
        role=r["role"], params=r, metrics=e["metrics"], datasets=e["sources"],
        tables=dict(estimates=e["estimates"], audit=e["audit"], folds=e["folds"],
                    residuals=e["residuals"], influence=e["influence"]),
        figures=dict(residualized_fit=e["figure"]), notes=e["notes"],
    )
