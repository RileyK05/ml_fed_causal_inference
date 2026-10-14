"""R1-R11: real-data linear premise tests (docs/encoder/cheap-test-menu.md, section 3a).

Everything fitted runs inside the development period with inner folds, except R1, which
also scores the outer design because its readout is whether a *planted* effect is
detectable, not which model wins.

Within-meeting tests (R3, R4, R7-R10) regress the return minus its meeting mean on
``s * (x - meeting mean of x)``: meeting fixed effects absorb the common slope, so only
the interactions are identified, and they are scored with centered MSE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fedcore.q3.contracts import BETA63, RET, RET_REL, RVOL21
from fedcore.q3.features import last_observed

from fedcore.q3.cheap.base import Result, check
from fedcore.q3.cheap.data import CheapData, Config, window_ols
from fedcore.q3.cheap.linear import (
    Standardizer,
    blocks,
    boot_mean,
    demean,
    dev_meetings,
    inner_folds,
    meeting_mse,
    ols_cluster,
    outer_folds,
    paired,
    ridge_fit_predict,
    rows_of,
    state_features,
    wald_p,
)

LOADING_FACTORS = ("mkt", "dy", "smb", "hml")


# ---------------------------------------------------------------------- helpers

def _sector_dummies(data: CheapData) -> pd.DataFrame:
    if "gsector" not in data.rows:
        return pd.DataFrame(index=range(len(data.rows)))
    d = pd.get_dummies(data.rows["gsector"].fillna("none").astype(str), prefix="sector", dtype=float)
    return d.iloc[:, 1:].reset_index(drop=True)  # drop one level


def _within_fit(data, rows, feats, std, s, y):
    meet = data.panel.meeting[rows]
    x = s[rows, None] * demean(meet, std.transform(feats[rows]))
    yt = demean(meet, y[rows])
    return x, yt, meet


def within_gain(data: CheapData, cfg: Config, feats: np.ndarray, *, s=None, y=None, folds=None) -> pd.Series:
    """Per test meeting: centered MSE of the interaction model minus centered MSE of 'no
    interactions' (which predicts 0 deviation). Negative means the features carry b."""
    s = data.panel.shock.astype(np.float64) if s is None else s
    y = data.panel.target.astype(np.float64) if y is None else y
    feats = np.asarray(feats, dtype=np.float64)
    if feats.ndim == 1:
        feats = feats[:, None]
    out = []
    for fold in folds or inner_folds(data, cfg):
        tr, te = rows_of(data, fold.train), rows_of(data, fold.test)
        tr, te = tr[np.isfinite(y[tr])], te[np.isfinite(y[te])]
        std = Standardizer().fit(feats[tr])
        xt, yt, _ = _within_fit(data, tr, feats, std, s, y)
        coef = np.linalg.lstsq(xt, yt, rcond=None)[0]
        xe, ye, me = _within_fit(data, te, feats, std, s, y)
        pred = xe @ coef
        model = meeting_mse(me, ye, pred)
        zero = meeting_mse(me, ye, np.zeros_like(ye))
        out.append(model - zero)
    return pd.concat(out)


def within_insample(data: CheapData, rows: np.ndarray, feats: np.ndarray, names, *, s=None, y=None) -> pd.DataFrame:
    """Full-sample within-meeting interaction regression with meeting-clustered SEs."""
    s = data.panel.shock.astype(np.float64) if s is None else s
    y = data.panel.target.astype(np.float64) if y is None else y
    feats = np.asarray(feats, dtype=np.float64)
    if feats.ndim == 1:
        feats = feats[:, None]
    rows = rows[np.isfinite(y[rows])]
    std = Standardizer().fit(feats[rows])
    x, yt, meet = _within_fit(data, rows, feats, std, s, y)
    fit = ols_cluster(x, yt, meet)
    table = pd.DataFrame({"term": [f"s x {n}" for n in names], "coef_per_sd": fit["coef"], "se": fit["se"], "t": fit["t"]})
    table.attrs["wald_p"] = wald_p(fit["coef"], fit["vcov"])
    table.attrs["n_meetings"] = fit["n_groups"]
    return table


def _dev_rows_idx(data: CheapData, cfg: Config) -> np.ndarray:
    return np.flatnonzero(data.dev_rows(cfg))


def _joint_features(data: CheapData) -> tuple[np.ndarray, list[str]]:
    st = state_features(data)
    sec = _sector_dummies(data)
    x = pd.concat([st.reset_index(drop=True), sec], axis=1)
    return x.to_numpy(dtype=np.float64), list(x.columns)


def rate_loadings(data: CheapData, cfg: Config, day_ok=None, width=None, min_obs=None, key="rate") -> pd.DataFrame:
    """Per panel row: market and 1-year-yield loadings over the ``width`` ordinary days that end
    the day before the meeting (ret ~ 1 + mkt + dy). Rate loading is % per bp."""
    memo = f"loadings_{key}"
    if memo not in data._memo:
        firm, pos = data.row_index()
        ok = data.ordinary_days() if day_ok is None else day_ok
        fit = window_ols(data, firm, pos - 1, width or cfg.window, ("mkt", "dy"), ok, min_obs or cfg.min_obs)
        data._memo[memo] = pd.DataFrame({"mkt_loading": fit["coef"][:, 1], "rate_loading": fit["coef"][:, 2],
                                         "rate_se": fit["se"][:, 2], "nobs": fit["nobs"]})
    return data._memo[memo]


def _firm_slopes(firm: np.ndarray, meeting: np.ndarray, y: np.ndarray, s_row: np.ndarray, min_n: int):
    """Per-firm OLS y = alpha + b*s. Returns firms, b, se (firms with at least min_n meetings)."""
    df = pd.DataFrame({"f": firm, "m": meeting, "y": y, "s": s_row}).dropna()
    out = []
    for f, g in df.groupby("f"):
        if len(g) < min_n:
            continue
        sc = g["s"] - g["s"].mean()
        sxx = float((sc ** 2).sum())
        if sxx <= 0:
            continue
        b = float((sc * (g["y"] - g["y"].mean())).sum() / sxx)
        resid = g["y"] - g["y"].mean() - b * sc
        se = float(np.sqrt((resid ** 2).sum() / max(len(g) - 2, 1) / sxx))
        out.append((f, b, se, len(g)))
    return pd.DataFrame(out, columns=["firm", "b", "se", "n"])


def _firm_slope_matrix(firm, meeting, y, min_n):
    """Firm x meeting matrix of y (NaN missing), its meeting dates and firm ids, for fast permutations."""
    df = pd.DataFrame({"f": firm, "m": pd.to_datetime(meeting), "y": y})
    wide = df.pivot_table(index="f", columns="m", values="y", aggfunc="first")
    wide = wide[wide.notna().sum(axis=1) >= min_n]
    return wide.to_numpy(dtype=np.float64), wide.columns, wide.index


def _slopes_from_matrix(ymat: np.ndarray, s: np.ndarray) -> np.ndarray:
    m = np.isfinite(ymat)
    n = m.sum(axis=1)
    y0 = np.where(m, ymat, 0.0)
    s_b = np.where(m, s[None, :], 0.0)
    sbar = s_b.sum(axis=1) / n
    ybar = y0.sum(axis=1) / n
    sc = np.where(m, s[None, :] - sbar[:, None], 0.0)
    sxx = (sc ** 2).sum(axis=1)
    return (sc * (y0 - ybar[:, None])).sum(axis=1) / np.where(sxx > 0, sxx, np.nan)


# ---------------------------------------------------------------------- R1

@check("R1", "Power on real noise (planted b heterogeneity)", "fitted")
def r1(data: CheapData, cfg: Config) -> Result:
    z, _ = blocks(data)
    y0, s = data.panel.target.astype(np.float64), data.panel.shock.astype(np.float64)
    st = state_features(data)
    drivers = {"rvol21": st["rvol21"].to_numpy(), "leverage": st[data.fund_names[0]].to_numpy()}
    rows = []
    for dname, raw in drivers.items():
        x = Standardizer().fit(raw[:, None]).transform(raw[:, None])[:, 0]
        for kappa in cfg.r1_effects:
            y = y0 + kappa * x * s
            for design, folds in (("outer", outer_folds(data, cfg)), ("inner_dev", inner_folds(data, cfg))):
                m_ab0, m_abz, corr = [], [], []
                for fold in folds:
                    tr, te = rows_of(data, fold.train), rows_of(data, fold.test)
                    p0 = ridge_fit_predict("ab0", z, s, y, tr, te, alpha=cfg.ridge_alpha)
                    p1 = ridge_fit_predict("abz", z, s, y, tr, te, alpha=cfg.ridge_alpha)
                    meet = data.panel.meeting[te]
                    m_ab0.append(meeting_mse(meet, y[te], p0["mu"]))
                    m_abz.append(meeting_mse(meet, y[te], p1["mu"]))
                    if kappa > 0:
                        corr.append(np.corrcoef(p1["b"], kappa * x[te])[0, 1])
                comp = paired(pd.concat(m_abz), pd.concat(m_ab0), cfg)
                rows.append({"driver": dname, "planted_sd_b": kappa, "design": design,
                             "abz_minus_ab0": comp["estimate"], "ci_low": comp["ci_low"], "ci_high": comp["ci_high"],
                             "detected": bool(comp["ci_high"] < 0), "n_test_meetings": comp["n_meetings"],
                             "b_hat_corr_with_planted": float(np.nanmean(corr)) if corr else np.nan})
    table = pd.DataFrame(rows)
    metrics = {}
    for (d, design), g in table.groupby(["driver", "design"]):
        hit = g.loc[g["detected"] & (g["planted_sd_b"] > 0), "planted_sd_b"]
        metrics[f"min_detected_sd_b_{d}_{design}"] = float(hit.min()) if len(hit) else None
    return Result(metrics, {"power": table},
                  "Real Fed-day returns plus kappa * x * s, x = standardized driver. abz vs ab0 ridge; detected = "
                  "the paired bootstrap CI is entirely below 0. One deterministic plant per size, so this is a "
                  "detectability read, not a false-positive rate. kappa = 0 shows whether real heterogeneity is "
                  "already detectable. The outer design is used only for this power read.")


# ---------------------------------------------------------------------- R2

@check("R2", "Firm-level b dispersion vs sampling noise", "fitted")
def r2(data: CheapData, cfg: Config) -> Result:
    from scipy.stats import chi2

    dev = _dev_rows_idx(data, cfg)
    p = data.panel
    y, s = p.target[dev].astype(np.float64), p.shock[dev].astype(np.float64)
    slopes = _firm_slopes(p.firm_id[dev], p.meeting[dev], y, s, cfg.r2_min_meetings)
    if len(slopes) < 3:
        return Result({"n_firms": len(slopes)}, {}, "Too few firms with enough meetings.")
    var_b, mean_se2 = float(slopes["b"].var(ddof=1)), float((slopes["se"] ** 2).mean())
    w = 1.0 / slopes["se"] ** 2
    bw = float((w * slopes["b"]).sum() / w.sum())
    q = float((w * (slopes["b"] - bw) ** 2).sum())
    ymat, mcols, _ = _firm_slope_matrix(p.firm_id[dev], p.meeting[dev], y, cfg.r2_min_meetings)
    s_m = data.meeting_table().set_index("meeting")["s"].reindex(mcols).to_numpy()
    obs = float(np.nanvar(_slopes_from_matrix(ymat, s_m), ddof=1))
    rng = np.random.default_rng(cfg.seed)
    perm = np.array([np.nanvar(_slopes_from_matrix(ymat, rng.permutation(s_m)), ddof=1) for _ in range(cfg.n_perm)])
    metrics = {
        "n_firms": int(len(slopes)), "var_b": var_b, "mean_se2": mean_se2, "excess_var_b": var_b - mean_se2,
        "var_ratio": var_b / mean_se2, "cochran_q": q, "cochran_p": float(chi2.sf(q, len(slopes) - 1)),
        "perm_p": float((perm >= obs).mean()), "pooled_b_weighted": bw,
    }
    perm_t = pd.DataFrame({"perm_var_b": perm})
    return Result(metrics, {"firm_slopes": slopes, "permutations": perm_t},
                  "Per-firm OLS of the Fed-day return on s, development period, firms with enough meetings "
                  "(a survivor sample of long-lived index members). Cochran's Q treats firms as independent; "
                  "shared meeting noise breaks that, so the meeting-permutation p-value is the honest one.")


# ---------------------------------------------------------------------- R3

@check("R3", "Which state features carry b", "fitted")
def r3(data: CheapData, cfg: Config) -> Result:
    st = state_features(data)
    sec = _sector_dummies(data)
    groups = {c: st[[c]].to_numpy(np.float64) for c in st.columns}
    if sec.shape[1]:
        groups["sector"] = sec.to_numpy(np.float64)
    joint, names = _joint_features(data)
    groups["joint"] = joint
    rows = []
    for name, x in groups.items():
        g = boot_mean(within_gain(data, cfg, x).to_numpy(), cfg.n_boot, cfg.seed)
        rows.append({"features": name, "centered_gain": g["estimate"], "ci_low": g["ci_low"], "ci_high": g["ci_high"],
                     "helps": bool(g["ci_high"] < 0), "n_test_meetings": g["n_meetings"]})
    insample = within_insample(data, _dev_rows_idx(data, cfg), joint, names)
    gains = pd.DataFrame(rows)
    return Result({"joint_gain": float(gains.loc[gains.features == "joint", "centered_gain"].iloc[0]),
                   "joint_wald_p_insample": insample.attrs["wald_p"],
                   "features_that_help": int(gains["helps"].sum())},
                  {"inner_fold_gains": gains, "insample_joint": insample},
                  "Within-meeting interaction model, development period. centered_gain = centered MSE minus the "
                  "no-interaction model (negative = helps). Features standardized on training rows.")


# ---------------------------------------------------------------------- R4

@check("R4", "Ordinary-day rate loading predicts Fed-day response?", "fitted")
def r4(data: CheapData, cfg: Config) -> Result:
    lo = rate_loadings(data, cfg)
    rate = lo["rate_loading"].to_numpy()
    joint, names = _joint_features(data)
    with_rate = np.column_stack((joint, rate))
    g_rate = within_gain(data, cfg, rate)
    g_joint = within_gain(data, cfg, joint)
    g_both = within_gain(data, cfg, with_rate)
    rows = [
        {"model": "rate_loading only", **boot_mean(g_rate.to_numpy(), cfg.n_boot, cfg.seed)},
        {"model": "joint state", **boot_mean(g_joint.to_numpy(), cfg.n_boot, cfg.seed)},
        {"model": "joint state + rate_loading", **boot_mean(g_both.to_numpy(), cfg.n_boot, cfg.seed)},
        {"model": "(joint + rate) - joint", **paired(g_both, g_joint, cfg)},
    ]
    dev = _dev_rows_idx(data, cfg)
    dev = dev[np.isfinite(rate[dev])]
    lo_q, hi_q = np.nanquantile(rate[dev], [0.01, 0.99])
    r = np.clip(rate, lo_q, hi_q)
    meet = data.panel.meeting[dev]
    s = data.panel.shock.astype(np.float64)
    fit = ols_cluster(s[dev] * demean(meet, r[dev]), demean(meet, data.panel.target[dev].astype(np.float64)), meet)
    transfer = {"coef_raw_units": float(fit["coef"][0]), "se": float(fit["se"][0]),
                "transfer_ratio": float(fit["coef"][0] / 10.0), "transfer_ratio_se": float(fit["se"][0] / 10.0),
                "rows": fit["n"], "meetings": fit["n_groups"],
                "median_rate_loading_pct_per_bp": float(np.nanmedian(rate[dev])),
                "coverage": float(np.isfinite(rate[data.dev_rows(cfg)]).mean())}
    table = pd.DataFrame(rows)
    return Result({"rate_only_gain": rows[0]["estimate"], "rate_adds_to_joint": rows[3]["estimate"],
                   "rate_adds_to_joint_ci_high": rows[3]["ci_high"], "transfer_ratio": transfer["transfer_ratio"]},
                  {"inner_fold_gains": table, "transfer": pd.DataFrame([transfer])},
                  "Rate loading: 63 ordinary days ending the day before each meeting, ret ~ 1 + mkt + dy, % per bp. "
                  "transfer_ratio = Fed-day interaction coefficient / 10: 1 means a firm's Fed sensitivity moves "
                  "one-for-one with its ordinary-day rate loading (s is in 10bp). Loading winsorized at 1/99%.")


# ---------------------------------------------------------------------- R5

@check("R5", "Persistence of factor loadings (past -> future)", "fitted")
def r5(data: CheapData, cfg: Config) -> Result:
    bound = data.calendar.searchsorted(dev_meetings(data, cfg)[-cfg.val_meetings])
    first = data.calendar.searchsorted(data.meetings()[0])
    ordinary = data.ordinary_days()
    rows = []
    for h in cfg.horizons:
        t = np.arange(max(first, cfg.window), bound - h, cfg.sample_every)
        firm = np.repeat(np.arange(data.permnos.size), t.size)
        tt = np.tile(t, data.permnos.size)
        keep = data.observed[firm, tt]
        firm, tt = firm[keep], tt[keep]
        past = window_ols(data, firm, tt, cfg.window, LOADING_FACTORS, ordinary, cfg.min_obs)
        fut = window_ols(data, firm, tt + h, h, LOADING_FACTORS, ordinary, max(15, h // 2))
        dates = data.calendar[tt]
        era = (dates.year // 5) * 5
        for j, name in enumerate(LOADING_FACTORS, start=1):
            a, b = past["coef"][:, j], fut["coef"][:, j]
            ok = np.isfinite(a) & np.isfinite(b)
            corrs = []
            for d in np.unique(dates[ok]):
                sel = ok & (dates == d)
                if sel.sum() >= 10:
                    corrs.append(pd.Series(a[sel]).corr(pd.Series(b[sel]), method="spearman"))
            lo, hi = np.nanquantile(a[ok], [0.01, 0.99]) if ok.any() else (0, 0)
            ac = np.clip(a[ok], lo, hi)
            slope = float(np.polyfit(ac, b[ok], 1)[0]) if ok.sum() > 10 else np.nan
            by_era = {f"rank_corr_era_{int(e)}": float(np.nanmean([
                pd.Series(a[ok & (dates == d)]).corr(pd.Series(b[ok & (dates == d)]), method="spearman")
                for d in np.unique(dates[ok & (era == e)]) if (ok & (dates == d)).sum() >= 10
            ] or [np.nan])) for e in np.unique(era[ok])}
            rows.append({"horizon_days": h, "factor": name, "pairs": int(ok.sum()),
                         "mean_cross_sectional_rank_corr": float(np.nanmean(corrs)) if corrs else np.nan,
                         "pooled_slope_future_on_past": slope, **by_era})
    table = pd.DataFrame(rows)
    return Result({f"persistence_{r.factor}_{r.horizon_days}d": r.mean_cross_sectional_rank_corr for r in table.itertuples()},
                  {"persistence": table},
                  "Past loading: 63 ordinary days ending at t. Future loading: the next h ordinary days. Factors "
                  "jointly: market, 1-year yield change, SMB, HML. Labels end before the development validation block.")


# ---------------------------------------------------------------------- R6

def earnings_reaction(data: CheapData, n_last: int = 4, min_events: int = 2) -> np.ndarray:
    """Per panel row: mean |2-day market-adjusted return| over the firm's last ``n_last``
    earnings events that finished before the meeting day (NaN with fewer than ``min_events``)."""
    if "earn" in data._memo:
        return data._memo["earn"]
    ev = data.earnings
    fidx = np.searchsorted(data.permnos, ev["permno"].to_numpy(dtype=np.int64))
    inside = (fidx < data.permnos.size) & (data.permnos[np.minimum(fidx, data.permnos.size - 1)] == ev["permno"].to_numpy())
    fidx = fidx[inside]
    p = data.calendar.searchsorted(pd.to_datetime(ev["rdq"].to_numpy()[inside]))
    ok = p + 1 < data.calendar.size
    fidx, p = fidx[ok], p[ok]
    rel = data.channels[:, :, RET_REL]
    both = data.observed[fidx, p] & data.observed[fidx, p + 1]
    reac = np.where(both, np.abs(rel[fidx, p].astype(np.float64) + rel[fidx, p + 1]), np.nan)
    events = pd.DataFrame({"f": fidx, "p": p, "r": reac}).dropna().sort_values(["f", "p"])
    firm, pos = data.row_index()
    out = np.full(firm.size, np.nan)
    by_firm = {f: (g["p"].to_numpy(), g["r"].to_numpy()) for f, g in events.groupby("f")}
    for f in np.unique(firm):
        if f not in by_firm:
            continue
        ep, er = by_firm[f]
        rows = np.flatnonzero(firm == f)
        k = np.searchsorted(ep + 1, pos[rows], side="left")  # events whose 2nd day is before the meeting
        cs = np.concatenate(([0.0], np.cumsum(er)))
        lo = np.maximum(k - n_last, 0)
        cnt = k - lo
        out[rows] = np.where(cnt >= min_events, (cs[k] - cs[lo]) / np.maximum(cnt, 1), np.nan)
    data._memo["earn"] = out
    return out


@check("R6", "Earnings-day reaction size vs Fed-day reaction size", "fitted")
def r6(data: CheapData, cfg: Config) -> Result:
    earn = earnings_reaction(data)
    rvol = state_features(data)["rvol21"].to_numpy()
    dev = _dev_rows_idx(data, cfg)
    meet = data.panel.meeting[dev]
    y = data.panel.target[dev].astype(np.float64)
    absdev = np.abs(demean(meet, y))
    e, r = earn[dev], rvol[dev]
    ok = np.isfinite(e) & np.isfinite(r)
    x = np.column_stack((demean(meet[ok], e[ok]), demean(meet[ok], r[ok])))
    fit = ols_cluster(x, demean(meet[ok], absdev[ok]), meet[ok])
    df = pd.DataFrame({"m": meet[ok], "e": e[ok], "r": r[ok], "a": absdev[ok]})
    resid_e = df["e"] - np.polyval(np.polyfit(df["r"], df["e"], 1), df["r"])
    df["e_res"] = resid_e
    ic = df.groupby("m").apply(lambda g: g["e"].corr(g["a"], method="spearman") if len(g) > 2 else np.nan,
                               include_groups=False)
    ic_res = df.groupby("m").apply(lambda g: g["e_res"].corr(g["a"], method="spearman") if len(g) > 2 else np.nan,
                                   include_groups=False)
    table = pd.DataFrame([
        {"term": "earnings_reaction", "coef": fit["coef"][0], "se": fit["se"][0], "t": fit["t"][0]},
        {"term": "rvol21 (control)", "coef": fit["coef"][1], "se": fit["se"][1], "t": fit["t"][1]},
    ])
    metrics = {"coverage": float(ok.mean()), "earn_t": float(fit["t"][0]),
               "rank_ic_mean": float(ic.mean()), "rank_ic_resid_on_rvol_mean": float(ic_res.mean()),
               "rank_ic_resid_boot_ci_low": boot_mean(ic_res.to_numpy(), cfg.n_boot, cfg.seed)["ci_low"]}
    return Result(metrics, {"regression": table},
                  "Outcome: |Fed-day return minus meeting mean|. Earnings reaction: mean |2-day market-adjusted "
                  "return| over the last 4 report dates before the meeting. Within-meeting regression with "
                  "trailing volatility as control; rank IC of the vol-residualized reaction is the clean read.")


# ---------------------------------------------------------------------- R7

@check("R7", "Which ordinary days carry transferable rate sensitivity", "fitted")
def r7(data: CheapData, cfg: Config) -> Result:
    ordinary = data.ordinary_days()
    dev_m = dev_meetings(data, cfg)
    span = (data.calendar >= dev_m.min() - pd.Timedelta(days=400)) & (data.calendar <= dev_m.max())
    rel = data.calendar.isin(data.releases) if data.releases is not None else np.zeros(data.calendar.size, bool)
    sets = {"all_ordinary": (ordinary, cfg.window, cfg.min_obs),
            "release_days": (ordinary & rel, 756, 20),
            "non_release_days": (ordinary & ~rel, cfg.window, cfg.min_obs)}
    rows = []
    for name, (mask, width, min_obs) in sets.items():
        n_in_dev = int((mask & span).sum())
        if name != "all_ordinary" and int((rel & span).sum()) < 20:
            rows.append({"day_set": name, "status": "skipped: fewer than 20 release dates in the development span",
                         "days_in_span": n_in_dev})
            continue
        lo = rate_loadings(data, cfg, day_ok=mask, width=width, min_obs=min_obs, key=name)
        g = boot_mean(within_gain(data, cfg, lo["rate_loading"].to_numpy()).to_numpy(), cfg.n_boot, cfg.seed)
        rows.append({"day_set": name, "status": "ok", "days_in_span": n_in_dev, "window": width,
                     "centered_gain": g["estimate"], "ci_low": g["ci_low"], "ci_high": g["ci_high"]})
    table = pd.DataFrame(rows)
    return Result({"sets_run": int((table["status"] == "ok").sum())}, {"day_sets": table},
                  "Rate loading estimated on each day set, scored like R4 (rate loading only). Release dates: CPI "
                  "from bls_cpi_release_dates (2023 onward only) and no payroll dates yet, so the release split is "
                  "skipped for the development period until earlier dates are sourced.")


# ---------------------------------------------------------------------- R8

@check("R8", "Stability of sensitivity across time and regimes", "fitted")
def r8(data: CheapData, cfg: Config) -> Result:
    dev_m = dev_meetings(data, cfg)
    half = len(dev_m) // 2
    halves = {"first_half": dev_m[:half], "second_half": dev_m[half:]}
    p = data.panel
    y = p.target.astype(np.float64)
    s_all = data.meeting_table().set_index("meeting")["s"]
    mats = {}
    for name, ms in halves.items():
        r = rows_of(data, ms)
        mats[name] = _firm_slope_matrix(p.firm_id[r], p.meeting[r], y[r], cfg.r8_min_meetings)
    slopes = {name: pd.Series(_slopes_from_matrix(ymat, s_all.reindex(cols).to_numpy()), index=firms)
              for name, (ymat, cols, firms) in mats.items()}
    common = slopes["first_half"].index.intersection(slopes["second_half"].index)
    corr = float(slopes["first_half"].loc[common].corr(slopes["second_half"].loc[common])) if len(common) > 3 else np.nan
    rng = np.random.default_rng(cfg.seed)
    perm = []
    if len(common) > 3:
        (y1, c1, f1), (y2, c2, f2) = mats["first_half"], mats["second_half"]
        s1, s2 = s_all.reindex(c1).to_numpy(), s_all.reindex(c2).to_numpy()
        for _ in range(cfg.n_perm):
            b1 = pd.Series(_slopes_from_matrix(y1, rng.permutation(s1)), index=f1)
            b2 = pd.Series(_slopes_from_matrix(y2, rng.permutation(s2)), index=f2)
            perm.append(b1.loc[common].corr(b2.loc[common]))
    # regime: sign of the trailing 63-day ordinary-day correlation of market and 1-year yield change
    f = pd.DataFrame({"mkt": data.mkt, "dy": data.dy}, index=data.calendar)
    f = f[data.ordinary_days()]
    roll = f["mkt"].rolling(63, min_periods=40).corr(f["dy"])
    regime = {d: ("positive" if roll[roll.index < d].dropna().iloc[-1] > 0 else "negative")
              if (roll.index < d).any() and roll[roll.index < d].notna().any() else "unknown" for d in dev_m}
    joint, names = _joint_features(data)
    rows = []
    for label, ms in [*halves.items(),
                      *((f"regime_{g}", pd.DatetimeIndex([d for d in dev_m if regime[d] == g])) for g in ("positive", "negative"))]:
        if len(ms) < 10:
            continue
        t = within_insample(data, rows_of(data, ms), joint, names)
        for rec in t.to_dict("records"):
            rows.append({"subset": label, "meetings": len(ms), **rec})
    metrics = {"firm_b_half_corr": corr, "firm_b_half_corr_perm_p": float(np.mean(np.array(perm) >= corr)) if perm else None,
               "n_firms_both_halves": int(len(common)),
               "meetings_positive_regime": sum(v == "positive" for v in regime.values()),
               "meetings_negative_regime": sum(v == "negative" for v in regime.values())}
    return Result(metrics, {"interactions_by_subset": pd.DataFrame(rows)},
                  "Inside the development period only (the 2009-2025 outer period stays untouched): first vs second "
                  "half of the development meetings, and stock-bond correlation regimes (sign of the trailing "
                  "63-day corr of market and 1-year yield change). Firm b correlation across halves vs a "
                  "meeting-permutation baseline.")


# ---------------------------------------------------------------------- R9

@check("R9", "Placebo date: the trading day before each meeting", "fitted")
def r9(data: CheapData, cfg: Config) -> Result:
    p = data.panel
    y_p = p.history[:, -1, RET].astype(np.float64)     # the prior day's return
    shifted_mask = p.history_mask[:, :-1]
    st = pd.DataFrame({"rvol21": last_observed(p.history[:, :-1, RVOL21], shifted_mask),
                       "beta63": last_observed(p.history[:, :-1, BETA63], shifted_mask)})
    for k, name in enumerate(data.fund_names):
        st[name] = p.fundamentals[:, k].astype(np.float64)
    x = pd.concat([st, _sector_dummies(data)], axis=1)
    names = list(x.columns)
    x = x.to_numpy(np.float64)
    dev = _dev_rows_idx(data, cfg)
    s = p.shock.astype(np.float64)
    out = []
    for label, y in (("placebo_prior_day", y_p), ("real_meeting_day", p.target.astype(np.float64))):
        r = dev[np.isfinite(y[dev])]
        pooled = ols_cluster(s[r], y[r], p.meeting[r], intercept=True)
        inter = within_insample(data, dev, x, names, y=y)
        gain = boot_mean(within_gain(data, cfg, x, y=y).to_numpy(), cfg.n_boot, cfg.seed)
        out.append({"target": label, "mean_return": float(np.mean(y[r])), "pooled_slope_on_s": float(pooled["coef"][1]),
                    "pooled_slope_t": float(pooled["t"][1]), "interactions_wald_p": inter.attrs["wald_p"],
                    "inner_centered_gain": gain["estimate"], "gain_ci_low": gain["ci_low"], "gain_ci_high": gain["ci_high"]})
    table = pd.DataFrame(out)
    plac = table.iloc[0]
    return Result({"placebo_slope_t": float(plac["pooled_slope_t"]), "placebo_interactions_wald_p": float(plac["interactions_wald_p"]),
                   "placebo_gain_ci_high": float(plac["gain_ci_high"]), "placebo_mean_return": float(plac["mean_return"])},
                  {"placebo_vs_real": table},
                  "The prior-day return should not load on the surprise: |t| well below 2, interactions not "
                  "jointly significant, no inner-fold gain. A non-zero mean is the known pre-FOMC drift and is "
                  "not a failure. Features are shifted back one day; fundamentals as of the meeting.")


# ---------------------------------------------------------------------- R10

@check("R10", "Placebo shock: shuffled s across meetings", "fitted")
def r10(data: CheapData, cfg: Config) -> Result:
    joint, _ = _joint_features(data)
    rate = rate_loadings(data, cfg)["rate_loading"].to_numpy()
    m = data.meeting_table()
    real = {"joint": float(within_gain(data, cfg, joint).mean()), "rate_loading": float(within_gain(data, cfg, rate).mean())}
    rng = np.random.default_rng(cfg.seed)
    meet = pd.to_datetime(data.panel.meeting)
    perms = {k: [] for k in real}
    for _ in range(cfg.n_perm):
        s_new = m["s"].to_numpy().copy()
        for era in m["era"].unique():
            sel = (m["era"] == era).to_numpy()
            s_new[sel] = rng.permutation(s_new[sel])
        s_row = pd.Series(s_new, index=m["meeting"]).reindex(meet).to_numpy()
        perms["joint"].append(float(within_gain(data, cfg, joint, s=s_row).mean()))
        perms["rate_loading"].append(float(within_gain(data, cfg, rate, s=s_row).mean()))
    rows = [{"features": k, "real_gain": v, "perm_median": float(np.median(perms[k])),
             "perm_p05": float(np.quantile(perms[k], 0.05)), "p_value": float(np.mean(np.array(perms[k]) <= v))}
            for k, v in real.items()]
    table = pd.DataFrame(rows)
    return Result({f"p_{r.features}": r.p_value for r in table.itertuples()}, {"permutation": table},
                  "s shuffled across whole meetings within 5-year eras, then the R3-joint and R4 rate-loading models "
                  "refit on inner folds. p = share of shuffles with a gain at least as good as the real one. "
                  "Exchangeability within eras is an assumption.")


# ---------------------------------------------------------------------- R11

@check("R11", "Residual dispersion vs surprise size and firm state", "fitted")
def r11(data: CheapData, cfg: Config) -> Result:
    z, _ = blocks(data)
    p = data.panel
    dev = _dev_rows_idx(data, cfg)
    y, s = p.target.astype(np.float64), p.shock.astype(np.float64)
    pred = ridge_fit_predict("abz", z, s, y, dev, dev, alpha=cfg.ridge_alpha)
    e2 = (y[dev] - pred["mu"]) ** 2
    rv = Standardizer().fit(state_features(data)[["rvol21"]].to_numpy()[dev]).transform(
        state_features(data)[["rvol21"]].to_numpy()[dev])[:, 0]
    a_s = np.abs(s[dev])
    meet = p.meeting[dev]
    spec1 = ols_cluster(np.column_stack((a_s, rv, a_s * rv)), e2, meet, intercept=True)
    joint, names = _joint_features(data)
    std = Standardizer().fit(joint[dev])
    xw = s[dev, None] * demean(meet, std.transform(joint[dev]))
    yw = demean(meet, y[dev])
    fit = ols_cluster(xw, yw, meet)
    ew2 = fit["resid"] ** 2
    m2 = meet[fit["rows"]]
    spec2 = ols_cluster(np.column_stack((demean(m2, rv[fit["rows"]]), a_s[fit["rows"]] * demean(m2, rv[fit["rows"]]))),
                        demean(m2, ew2), m2)
    rows = [
        *({"spec": "absolute residual^2 (no meeting FE)", "term": t, "coef": c, "se": se, "t": tt}
          for t, c, se, tt in zip(("intercept", "|s|", "rvol21_std", "|s| x rvol21_std"), spec1["coef"], spec1["se"], spec1["t"])),
        *({"spec": "within-meeting residual^2 (meeting FE)", "term": t, "coef": c, "se": se, "t": tt}
          for t, c, se, tt in zip(("rvol21_std", "|s| x rvol21_std"), spec2["coef"], spec2["se"], spec2["t"])),
    ]
    return Result({"abs_s_t": float(spec1["t"][1]), "rvol_t": float(spec1["t"][2]),
                   "abs_s_x_rvol_t_within": float(spec2["t"][1])},
                  {"dispersion": pd.DataFrame(rows)},
                  "Squared residuals of in-sample development fits. With meeting fixed effects |s| is absorbed (it "
                  "is constant within a meeting), so the within spec only identifies |s| x state. A positive "
                  "|s| or |s| x rvol term supports the surprise-dependent volatility output (section 7).")
