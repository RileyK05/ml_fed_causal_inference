"""C1-C5: small fitted diagnostics with linear models, inside the development period."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fedcore.q3.cheap.base import Result, check
from fedcore.q3.cheap.data import CheapData, Config, window_ols
from fedcore.q3.cheap.linear import (
    blocks,
    dev_meetings,
    inner_folds,
    meeting_mse,
    paired,
    ridge_fit_predict,
    rows_of,
)

LOADING_FACTORS = ("mkt", "dy", "smb", "hml")


def _inner_scores(data: CheapData, cfg: Config, kinds) -> dict:
    """Per-meeting test MSE of each ridge kind over the inner folds."""
    z, _ = blocks(data)
    y = data.panel.target.astype(np.float64)
    s = data.panel.shock.astype(np.float64)
    out = {k: [] for k in kinds}
    for fold in inner_folds(data, cfg):
        tr, te = rows_of(data, fold.train), rows_of(data, fold.test)
        for kind in kinds:
            pred = ridge_fit_predict(kind, z, s, y, tr, te, alpha=cfg.ridge_alpha)
            out[kind].append(meeting_mse(data.panel.meeting[te], y[te], pred["mu"]))
    return {k: pd.concat(v) for k, v in out.items()}


@check("C1", "Does the shock help, and does state-dependent b help?", "fitted")
def c1(data: CheapData, cfg: Config) -> Result:
    mse = _inner_scores(data, cfg, ("a", "ab0", "abz"))
    rows = [{"model": k, "mse_per_meeting": float(v.mean()), "n_meetings": int(v.size)} for k, v in mse.items()]
    comp = []
    for left, right, q in (("ab0", "a", "does knowing s help?"), ("abz", "ab0", "does b(z) beat one common b?"),
                           ("abz", "a", "full vs no-shock")):
        comp.append({"comparison": f"{left} - {right}", "question": q, **paired(mse[left], mse[right], cfg)})
    comp = pd.DataFrame(comp)
    metrics = {f"{r['comparison'].replace(' - ', '_minus_')}": r["estimate"] for r in comp.to_dict("records")}
    metrics.update({f"{r['comparison'].replace(' - ', '_minus_')}_ci_high": r["ci_high"] for r in comp.to_dict("records")})
    return Result(metrics, {"models": pd.DataFrame(rows), "comparisons": comp},
                  "Ridge on summary features + fundamentals + context; inner walk-forward inside the development "
                  "period. Negative differences favor the left model.")


@check("C2", "Ordinary-day rate model vs market-plus-rate model", "fitted")
def c2(data: CheapData, cfg: Config) -> Result:
    meetings = dev_meetings(data, cfg)
    val_start = data.calendar.searchsorted(meetings[-cfg.val_meetings])
    dev_end = data.dev_end_pos(cfg)
    first = data.calendar.searchsorted(meetings[0])
    ordinary = data.ordinary_days()
    f = np.arange(data.permnos.size)
    width = int(val_start - first)
    rows, loadings = [], {}
    for name, factors in (("rate_only", ("dy",)), ("market_plus_rate", ("mkt", "dy"))):
        fit = window_ols(data, f, np.full(f.size, val_start - 1), width, factors, ordinary, min_obs=250)
        half = width // 2
        early = window_ols(data, f, np.full(f.size, first + half - 1), half, factors, ordinary, min_obs=120)
        late = window_ols(data, f, np.full(f.size, val_start - 1), width - half, factors, ordinary, min_obs=120)
        k_dy = 1 + factors.index("dy")
        both = np.isfinite(early["coef"][:, k_dy]) & np.isfinite(late["coef"][:, k_dy])
        stability = float(np.corrcoef(early["coef"][both, k_dy], late["coef"][both, k_dy])[0, 1]) if both.sum() > 2 else np.nan
        x = data.factors(factors)
        days = np.arange(val_start, dev_end)
        ok_day = ordinary[days] & np.isfinite(x[days]).all(axis=1)
        days = days[ok_day]
        pred = fit["coef"][:, :1] + fit["coef"][:, 1:] @ x[days].T          # (F, days)
        actual = data.ret[:, days].astype(np.float64)
        valid = data.observed[:, days] & np.isfinite(pred)
        err = np.where(valid, (actual - pred) ** 2, np.nan)
        loadings[name] = pd.Series(np.nanmean(err, axis=0), index=data.calendar[days])
        rows.append({"model": name, "firms_fitted": int(np.isfinite(fit["coef"][:, k_dy]).sum()),
                     "val_days": int(days.size), "val_mse": float(np.nanmean(err)),
                     "rate_loading_half_split_corr": stability,
                     "median_rate_loading": float(np.nanmedian(fit["coef"][:, k_dy]))})
    diff = (loadings["market_plus_rate"] - loadings["rate_only"]).dropna()
    boot = {"comparison": "market_plus_rate - rate_only (per-day MSE)",
            **_boot_days(diff.to_numpy(), cfg)}
    return Result({"val_mse_diff": boot["estimate"], "val_mse_diff_ci_high": boot["ci_high"]},
                  {"models": pd.DataFrame(rows), "comparison": pd.DataFrame([boot])},
                  "Per-firm OLS on ordinary days before the development validation block; scored on ordinary days "
                  "inside it, using that day's realized factors. These are ordinary-day associations.")


def _boot_days(v: np.ndarray, cfg: Config) -> dict:
    from fedcore.q3.cheap.linear import boot_mean

    out = boot_mean(v, cfg.n_boot, cfg.seed)
    out["n_days"] = out.pop("n_meetings")
    return out


def _sample_firm_dates(data: CheapData, cfg: Config, last_label_pos: int, horizon: int, past: int = 0):
    """(firm, t) pairs every ``sample_every`` days with a past window and a label inside bounds."""
    first = data.calendar.searchsorted(data.meetings()[0])
    t = np.arange(max(first, past), last_label_pos - horizon, cfg.sample_every)
    firm = np.repeat(np.arange(data.permnos.size), t.size)
    tt = np.tile(t, data.permnos.size)
    keep = data.observed[firm, tt]
    return firm[keep], tt[keep]


@check("C3", "Future-loading label reliability", "fitted")
def c3(data: CheapData, cfg: Config) -> Result:
    bound = data.calendar.searchsorted(dev_meetings(data, cfg)[-cfg.val_meetings])
    ordinary = data.ordinary_days()
    rows = []
    for h in cfg.horizons:
        firm, t = _sample_firm_dates(data, cfg, bound, h)
        full = window_ols(data, firm, t + h, h, LOADING_FACTORS, ordinary, min_obs=max(15, h // 2))
        odd = ordinary & (np.arange(data.calendar.size) % 2 == 1)
        even = ordinary & (np.arange(data.calendar.size) % 2 == 0)
        mo = max(10, h // 4)
        a = window_ols(data, firm, t + h, h, LOADING_FACTORS, odd, min_obs=mo)
        b = window_ols(data, firm, t + h, h, LOADING_FACTORS, even, min_obs=mo)
        dates = data.calendar[t]
        for j, name in enumerate(LOADING_FACTORS, start=1):
            coef, se = full["coef"][:, j], full["se"][:, j]
            ok = np.isfinite(coef)
            reliab, half = [], []
            for d in np.unique(dates[ok]):
                sel = ok & (dates == d)
                if sel.sum() < 10:
                    continue
                var = np.var(coef[sel])
                reliab.append(1.0 - np.mean(se[sel] ** 2) / var if var > 0 else np.nan)
                both = sel & np.isfinite(a["coef"][:, j]) & np.isfinite(b["coef"][:, j])
                if both.sum() >= 10:
                    r = np.corrcoef(a["coef"][both, j], b["coef"][both, j])[0, 1]
                    half.append(2 * r / (1 + r) if r > -1 else np.nan)
            rows.append({
                "horizon_days": h, "factor": name, "labels": int(ok.sum()),
                "median_abs_loading": float(np.nanmedian(np.abs(coef))), "median_se": float(np.nanmedian(se)),
                "reliability_1_minus_noise_share": float(np.nanmean(reliab)) if reliab else np.nan,
                "split_half_reliability": float(np.nanmean(half)) if half else np.nan,
                "median_design_cond": float(np.nanmedian(full["cond"])),
            })
    table = pd.DataFrame(rows)
    metrics = {f"reliability_{r.factor}_{r.horizon_days}d": r.reliability_1_minus_noise_share for r in table.itertuples()}
    return Result(metrics, {"by_horizon_factor": table},
                  "Loadings on market, 1-year yield change, SMB and HML over the next h ordinary days, sampled "
                  "every 21 days, labels ending before the development validation block. Reliability near 0 means "
                  "the label is mostly estimation noise. Overlapping windows are not independent.")


@check("C4", "Leave-one-large-meeting-out", "fitted")
def c4(data: CheapData, cfg: Config) -> Result:
    z, _ = blocks(data)
    y, s = data.panel.target.astype(np.float64), data.panel.shock.astype(np.float64)
    meetings = dev_meetings(data, cfg)
    train_m, val_m = meetings[: -cfg.val_meetings], meetings[-cfg.val_meetings :]
    sm = data.meeting_table().set_index("meeting")["s"].loc[train_m]
    top = ((sm - sm.mean()) ** 2).sort_values(ascending=False).head(5).index
    va = rows_of(data, val_m)

    def fit(drop=None):
        keep = train_m if drop is None else train_m.drop(drop)
        tr = rows_of(data, keep)
        x = np.column_stack((np.ones(tr.size), s[tr]))
        pooled = np.linalg.lstsq(x, y[tr], rcond=None)[0]
        pred = ridge_fit_predict("abz", z, s, y, tr, va, alpha=cfg.ridge_alpha)
        mse = meeting_mse(data.panel.meeting[va], y[va], pred["mu"]).mean()
        return pooled[1], pred, mse

    b_full, p_full, mse_full = fit()
    rows = [{"dropped": "none", "s_bp": np.nan, "pooled_b": b_full, "val_mse": mse_full,
             "mean_abs_mu_change": 0.0, "mean_b_change": 0.0}]
    for d in top:
        b, p, mse = fit(d)
        rows.append({"dropped": d.date().isoformat(), "s_bp": 10.0 * float(sm.loc[d]), "pooled_b": b,
                     "val_mse": mse, "mean_abs_mu_change": float(np.mean(np.abs(p["mu"] - p_full["mu"]))),
                     "mean_b_change": float(np.mean(p["b"] - p_full["b"]))})
    table = pd.DataFrame(rows)
    return Result({"max_pooled_b_shift": float((table["pooled_b"] - b_full).abs().max()),
                   "max_val_mse_shift": float((table["val_mse"] - mse_full).abs().max())},
                  {"leave_one_out": table},
                  "Top 5 training meetings by centered s^2 (chosen from training data only), dropped one at a time. "
                  "Pooled b is a + b*s OLS; predictions are the abz ridge scored on the development validation block.")


@check("C5", "Global shock centering", "fitted")
def c5(data: CheapData, cfg: Config) -> Result:
    z, _ = blocks(data)
    y, s = data.panel.target.astype(np.float64), data.panel.shock.astype(np.float64)
    raw, cen, max_gap = [], [], 0.0
    for fold in inner_folds(data, cfg):
        tr, te = rows_of(data, fold.train), rows_of(data, fold.test)
        c = s[tr].mean()
        p_raw = ridge_fit_predict("abz", z, s, y, tr, te, alpha=cfg.ridge_alpha)
        p_cen = ridge_fit_predict("abz", z, s - c, y, tr, te, alpha=cfg.ridge_alpha)
        # mu is in original units either way (a_orig = a_c - b*c); only the ridge penalty differs
        max_gap = max(max_gap, float(np.max(np.abs(p_raw["mu"] - p_cen["mu"]))))
        raw.append(meeting_mse(data.panel.meeting[te], y[te], p_raw["mu"]))
        cen.append(meeting_mse(data.panel.meeting[te], y[te], p_cen["mu"]))
    comp = paired(pd.concat(cen), pd.concat(raw), cfg)
    return Result({"centered_minus_raw": comp["estimate"], "centered_minus_raw_ci_low": comp["ci_low"],
                   "centered_minus_raw_ci_high": comp["ci_high"], "max_abs_prediction_gap": max_gap},
                  {"comparison": pd.DataFrame([{"comparison": "centered - raw", **comp}])},
                  "abz ridge with s vs s minus its training mean. Ridge shrinks a different intercept-slope "
                  "split, so predictions can differ slightly; a neural optimizer may differ more.")
