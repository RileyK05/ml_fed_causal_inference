"""Q3 v2 descriptive gates G1-G3 (docs/encoder/q3-v2-plan.md, section 6). No model is fit.

Development meetings only (through 2021-12-31), top-1500 panel. Target per firm and window:
the market-adjusted return ``R_i(window) - beta_i * R_mkt(window)``, beta_i the last 63-day beta
before the meeting. A window with a missing firm day is dropped for that firm (never filled).

Each test is the v1 group test (encoder report, sections 3.3-3.4): firms are split into groups by
a trait; the firm's return minus the meeting mean is regressed on group dummies, group x market
move, and group x each shock term; a meeting-clustered Wald test asks whether one term's
coefficient differs across groups. Non-sector traits run raw and within sector.

* G1: group x ``s_policy`` and group x ``s_info`` (policy/information split), and group x
  ``slope`` (10y-2y window change, with group x s alongside).
* G2: group x s x regime, with group x regime and group x market x regime as controls.
* G3: ``|abnormal return|`` on group, group x large-surprise (\\|s\\| >= 2.5bp), pre-meeting
  volatility and beta (each also x large); tests whether groups' extra movement on big days differs.

Pass bar for each gate: some trait (within sector, or sector itself) has p < 0.01 in both
1994-2008 and 2009-2021 for the same window and term.

    python -m fedenc.gates            # saves one exploratory Q3 run with every test
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm

WINDOWS = {"day 0": (0, 0), "days 0-2": (0, 2)}
PERIODS = (("1994-2021", "1900-01-01", "2021-12-31"), ("1994-2008", "1900-01-01", "2008-12-31"),
           ("2009-2021", "2009-01-01", "2021-12-31"))
SUBPERIODS = ("1994-2008", "2009-2021")
LARGE = 0.25  # |s| in 10bp units
P_PASS = 0.01
MIN_ACTIVE = 10  # meetings per group where the tested term is nonzero
REGIMES = ("high_vol", "drawdown", "easing", "tightening", "zlb")
SECT = {"10": "Energy", "15": "Materials", "20": "Industrials", "25": "Cons. Disc.", "30": "Staples",
        "35": "Health Care", "40": "Financials", "45": "Info Tech", "50": "Comm. Svcs", "55": "Utilities",
        "60": "Real Estate"}
TRAITS = {"sector_name": "sector", "rating": "credit rating", "fcf_group": "free cash flow",
          "gap_group": "funding gap", "maturity": "debt maturity", "style": "growth vs value",
          "size": "size", "rate_beta_group": "rate beta (63d)", "hml_beta_group": "value beta (63d)"}


# ----------------------------------------------------------------------------- tests

def _fit_and_test(full: pd.DataFrame, yy: np.ndarray, meetings: pd.Series, groups, suffix: str):
    """Clustered OLS, then a Wald test that the ``group|suffix`` coefficients are all equal.

    All-zero columns are dropped first (a group the term never touches), and a test whose
    constraint covariance is rank-deficient returns NaN rather than an unreliable p."""
    full = full.loc[:, (full != 0).any(axis=0)]
    groups = [g for g in groups if f"{g}|{suffix}" in full.columns]
    if len(groups) < 2:
        return {}, np.nan, "fewer than 2 groups"
    fit = sm.OLS(yy, full.to_numpy()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(meetings)[0]})
    idx = [full.columns.get_loc(f"{g}|{suffix}") for g in groups]
    per = {g: (float(fit.params[i]), float(fit.tvalues[i])) for g, i in zip(groups, idx)}
    R = np.zeros((len(idx) - 1, full.shape[1]))
    for r, i in enumerate(idx[1:]):
        R[r, i], R[r, idx[0]] = 1.0, -1.0
    rank = np.linalg.matrix_rank(R @ np.asarray(fit.cov_params()) @ R.T)
    if rank < R.shape[0]:
        return per, np.nan, f"constraint rank {rank} of {R.shape[0]}"
    return per, float(np.squeeze(fit.wald_test(R, scalar=True, use_f=True).pvalue)), ""


def group_test(d: pd.DataFrame, col: str, y: str, terms: list[str], test: str):
    """Joint meeting-clustered test that group x ``test`` coefficients are equal across groups.

    Regressors: group dummies and group x each of ``terms`` (which must include ``test``).
    Returns (per-group {group: (coef, t)}, p-value, meetings, note)."""
    d = d.dropna(subset=[y, *terms])
    d = d[(d[col] != "unknown") & d[col].notna()]
    counts = d.groupby(col)["meeting"].nunique()
    active = d[d[test] != 0].groupby(col)["meeting"].nunique()  # e.g. meetings in the regime
    keep = counts.index[(counts >= 30) & (active.reindex(counts.index).fillna(0) >= MIN_ACTIVE)]
    d = d[d[col].isin(keep)]
    if d[col].nunique() < 2:
        return {}, np.nan, d["meeting"].nunique(), "fewer than 2 groups"
    X = pd.get_dummies(d[col], dtype=float)
    blocks = [X] + [X.mul(d[t].astype(float), axis=0).add_suffix(f"|{t}") for t in terms]
    full = pd.concat(blocks, axis=1)
    yy = (d[y] - d.groupby("meeting")[y].transform("mean")).to_numpy(dtype=float)
    per, p, note = _fit_and_test(full, yy, d["meeting"], list(X.columns), test)
    return per, p, d["meeting"].nunique(), note


def magnitude_test(d: pd.DataFrame, col: str, y: str):
    """G3: is a group's extra |abnormal return| on large-surprise meetings different across groups,
    beyond pre-meeting volatility and beta (both also interacted with large)?"""
    d = d.dropna(subset=[y, "vol21", "beta"]).copy()
    d = d[(d[col] != "unknown") & d[col].notna()]
    d["absy"] = d[y].abs()
    counts = d[d["large"] == 1].groupby(col)["meeting"].nunique()
    d = d[d[col].isin(counts.index[counts >= 10])]
    if d[col].nunique() < 2:
        return {}, np.nan, d["meeting"].nunique(), "fewer than 2 groups"
    X = pd.get_dummies(d[col], dtype=float)
    common = d[["vol21", "beta"]].astype(float)
    common = pd.concat([common, common.mul(d["large"], axis=0).add_suffix("|large")], axis=1)
    full = pd.concat([X, X.mul(d["large"], axis=0).add_suffix("|large"), common], axis=1)
    yy = (d["absy"] - d.groupby("meeting")["absy"].transform("mean")).to_numpy(dtype=float)
    per, p, note = _fit_and_test(full, yy, d["meeting"], list(X.columns), "large")
    return per, p, int(d.loc[d["large"] == 1, "meeting"].nunique()), note


# ----------------------------------------------------------------------------- data

def _terciles(base: pd.DataFrame, col: str, labels) -> pd.Series:
    out = base.groupby("meeting")[col].transform(
        lambda x: pd.qcut(x.rank(method="first"), len(labels), labels=labels) if x.notna().sum() >= 30 else np.nan
    ).astype(object)
    out[base[col].isna()] = "missing"
    return out


def _trailing_betas(panel, macro) -> tuple[np.ndarray, np.ndarray]:
    """63-day OLS slope of the firm's market-relative return on the 1y yield change and on HML."""
    from fedcore.q3.contracts import RET_REL

    rr = panel.history[:, -63:, RET_REL].astype(np.float64)
    obs = panel.history_mask[:, -63:] & np.isfinite(rr)
    meetings = pd.DatetimeIndex(panel.meeting)
    pos = macro.calendar.searchsorted(meetings, side="left")
    win = pos[:, None] - 63 + np.arange(63)[None, :]
    out = []
    for x in (macro.dy1, macro.hml):
        xx = np.asarray(x, dtype=np.float64)[win]
        cnt = obs.sum(axis=1)
        xm = np.where(obs, xx, 0.0).sum(axis=1) / np.maximum(cnt, 1)
        rm = np.where(obs, rr, 0.0).sum(axis=1) / np.maximum(cnt, 1)
        xd = np.where(obs, xx - xm[:, None], 0.0)
        cov = (xd * np.where(obs, rr - rm[:, None], 0.0)).sum(axis=1)
        var = (xd ** 2).sum(axis=1)
        out.append(np.where((cnt >= 40) & (var > 1e-10), cov / np.where(var > 0, var, 1.0), np.nan))
    return out[0], out[1]


def regime_table(meetings: pd.DatetimeIndex) -> pd.DataFrame:
    """Binary regimes per meeting, all from the prior close. Thresholds: high_vol above the
    development median of 21-day realized market volatility; drawdown at least 10% below the
    252-day high; easing / tightening a 126-day 1y-yield change of at most -0.25 / at least
    +0.25 points; zlb a 1y yield below 0.5%."""
    from fedcore.data import load

    ff = load("ff_factors_daily").assign(date=lambda d: pd.to_datetime(d["date"])).set_index("date").sort_index()
    mkt = 100.0 * (ff["mktrf"] + ff["rf"])
    vol21 = mkt.rolling(21, min_periods=15).std() * np.sqrt(252)
    level = (1 + mkt / 100).cumprod()
    dd = 100 * (level / level.rolling(252, min_periods=126).max() - 1)
    y1 = load("treasury_1y").rename(columns={"Date": "date", "SVENY01": "y1"})
    y1 = y1.assign(date=pd.to_datetime(y1["date"])).set_index("date")["y1"].sort_index().dropna()
    chg = y1 - y1.shift(126)

    def prior(series):
        s = series.dropna()
        pos = s.index.searchsorted(meetings, side="left") - 1
        return pd.Series(np.where(pos >= 0, s.to_numpy()[np.maximum(pos, 0)], np.nan), index=meetings)

    v, d, lv, c = prior(vol21), prior(dd), prior(y1), prior(chg)
    return pd.DataFrame({
        "vol21_mkt": v, "drawdown_pct": d, "y1": lv, "y1_chg126": c,
        "high_vol": (v > v.median()).astype(float), "drawdown": (d <= -10).astype(float),
        "easing": (c <= -0.25).astype(float), "tightening": (c >= 0.25).astype(float),
        "zlb": (lv < 0.5).astype(float),
    })


def build_base(through: str = "2021-12-31") -> pd.DataFrame:
    """One row per firm and development meeting: windows, shocks, traits, regimes."""
    from fedcore.data import load
    from fedcore.q3.contracts import BETA63, RVOL21
    from fedcore.q3.features import last_observed
    from fedcore.q3.real import BETA_FALLBACK, FUNDAMENTAL_NAMES
    from fedcore.q3.shocks import split_shock
    from fedenc.v2 import load_macro_tables
    from fedenc.workflow import bounded_panel

    panel, _, rows = bounded_panel(through, universe="top1500", with_rows=True)
    beta = last_observed(panel.history[:, :, BETA63], panel.history_mask)
    pol_inf = split_shock(panel, "policy_info")
    slope = split_shock(panel, "target_slope")[:, 1]
    rate_beta, hml_beta = _trailing_betas(panel, load_macro_tables())
    base = pd.DataFrame({
        "meeting": pd.DatetimeIndex(panel.meeting).astype("datetime64[ns]"),
        "permno": rows["permno"].to_numpy(dtype=np.int64), "s": panel.shock.astype(float),
        "s_policy": pol_inf[:, 0], "s_info": pol_inf[:, 1], "slope": slope,
        "beta": np.where(np.isfinite(beta), beta, BETA_FALLBACK),
        "vol21": last_observed(panel.history[:, :, RVOL21], panel.history_mask),
        "gvkey": rows["gvkey"].astype("string"), "sector": rows["gsector"].astype("string").str[:2].to_numpy(),
        "btm": panel.fundamentals[:, FUNDAMENTAL_NAMES.index("book_to_market")].astype(float),
        "log_size": panel.fundamentals[:, FUNDAMENTAL_NAMES.index("log_size")].astype(float),
        "rate_beta": rate_beta, "hml_beta": hml_beta, "day0_panel": panel.target.astype(float),
    })
    base["row"] = np.arange(len(base))
    base["large"] = (base["s"].abs() >= LARGE).astype(float)
    del panel

    # windows: firm returns from CRSP daily, market from Fama-French (percent)
    ff = load("ff_factors_daily", columns=["mktrf", "rf"])
    cal = pd.DatetimeIndex(pd.to_datetime(ff["date"])).astype("datetime64[ns]")
    mkt_day = pd.Series(100.0 * (ff["mktrf"] + ff["rf"]).to_numpy(float), index=cal)
    um = pd.DatetimeIndex(base["meeting"].unique())
    pos0 = cal.get_indexer(um)
    assert (pos0 >= 0).all(), "a meeting is not a trading day"
    span = max(hi for _, hi in WINDOWS.values()) + 1
    need = cal[np.unique((pos0[:, None] + np.arange(span)[None, :]).ravel())]
    daily = load("crsp_daily_top1500", start=need.min(), end=need.max(), columns=["permno", "dlyret"])
    daily = daily[daily["dlycaldt"].isin(need)]
    ret = daily.assign(dlyret=100.0 * pd.to_numeric(daily["dlyret"], errors="coerce")).pivot_table(
        index="permno", columns="dlycaldt", values="dlyret", aggfunc="first")
    ret.columns = pd.DatetimeIndex(ret.columns).astype("datetime64[ns]")
    meet_pos = dict(zip(um, pos0))
    fi = ret.index.get_indexer(base["permno"])
    R = ret.to_numpy()
    col_of = {d: j for j, d in enumerate(ret.columns)}
    for name, (lo, hi) in WINDOWS.items():
        days = {m: [cal[meet_pos[m] + k] for k in range(lo, hi + 1)] for m in um}
        cols = np.array([[col_of.get(d, -1) for d in days[m]] for m in base["meeting"]])
        r = np.where((fi[:, None] >= 0) & (cols >= 0), R[np.maximum(fi, 0)[:, None], np.maximum(cols, 0)], np.nan)
        firm = 100.0 * (np.prod(1 + r / 100.0, axis=1) - 1)
        mk = {m: 100.0 * (np.prod([1 + mkt_day[d] / 100.0 for d in days[m]]) - 1) for m in um}
        market = base["meeting"].map(mk).to_numpy(dtype=float)
        base[f"y|{name}"] = firm - base["beta"] * market
        base[f"mkt|{name}"] = market

    # traits as of the last report before the meeting (same definitions as the v1 tests)
    rat = load("comp_ratings_top1500").dropna(subset=["splticrm"])
    rat = rat.assign(gvkey=rat["gvkey"].astype("string"),
                     datadate=pd.to_datetime(rat["datadate"]).astype("datetime64[ns]")).sort_values("datadate")
    g = pd.merge_asof(base.dropna(subset=["gvkey"]).sort_values("meeting"), rat[["gvkey", "datadate", "splticrm"]],
                      left_on="meeting", right_on="datadate", by="gvkey", allow_exact_matches=False,
                      tolerance=pd.Timedelta(days=100)).set_index("row")

    def bucket(x):
        if not isinstance(x, str):
            return "unrated"
        x = x.strip()
        return "BBB" if x.startswith("BBB") else "A- or better" if x.startswith("A") else "junk (BB+ and below)"

    base["rating"] = [bucket(x) for x in g["splticrm"].reindex(base["row"])]
    key = ["gvkey", "datadate", "fyearq", "fqtr"]
    cash = load("comp_fundq_cash_top1500")[key + ["rdq", "oancfy", "capxy"]]
    fund = load("comp_fundq_funding_top1500")[key + ["dltry", "ppentq"]]
    fq = load("comp_fundq_top1500")[key + ["atq", "dlcq", "dlttq"]]
    f = cash.merge(fund, on=key, how="left").merge(fq, on=key, how="left")
    f = f.assign(gvkey=f["gvkey"].astype("string"), rdq=pd.to_datetime(f["rdq"]).astype("datetime64[ns]"))
    for c in ("fyearq", "fqtr", "oancfy", "capxy", "dltry", "ppentq", "atq", "dlcq", "dlttq"):
        f[c] = pd.to_numeric(f[c], errors="coerce").astype("float64")
    f = f.dropna(subset=["rdq", "fyearq", "fqtr"]).sort_values(["gvkey", "fyearq", "fqtr"]).drop_duplicates(
        ["gvkey", "fyearq", "fqtr"])
    for c in ("oancfy", "capxy", "dltry"):  # year-to-date -> quarter
        prev = f.groupby(["gvkey", "fyearq"])[c].shift(1)
        f[c[:-1] + "_q"] = np.where(f["fqtr"] > 1, f[c] - prev, f[c])
    at = f["atq"].where(f["atq"] > 0)
    f["fcf"] = (f["oancf_q"] - f["capx_q"]) / at
    f["gap4"] = (f["capx_q"] + f["dltr_q"].fillna(0.0) - f["oancf_q"]).groupby(f["gvkey"]).transform(
        lambda x: x.rolling(4, min_periods=4).sum())
    f["funding_gap"] = f["gap4"] / at
    tot = f["dlcq"] + f["dlttq"]
    f["st_share"] = f["dlcq"] / tot.where(tot > 0)
    g = pd.merge_asof(base.dropna(subset=["gvkey"]).sort_values("meeting"),
                      f.sort_values("rdq")[["gvkey", "rdq", "fcf", "funding_gap", "st_share"]], left_on="meeting",
                      right_on="rdq", by="gvkey", allow_exact_matches=False,
                      tolerance=pd.Timedelta(days=365)).set_index("row")
    for c in ("fcf", "funding_gap", "st_share"):
        base[c] = g[c].reindex(base["row"]).to_numpy()

    base["sector_name"] = base["sector"].map(SECT).fillna("unknown")
    base["fcf_group"] = _terciles(base, "fcf", ["cash burners", "mid FCF", "cash generators"])
    base["gap_group"] = _terciles(base, "funding_gap", ["self-funded", "middle", "needs outside money"])
    base["maturity"] = _terciles(base, "st_share", ["mostly long-term", "middle", "mostly short-term"])
    base["style"] = _terciles(base, "btm", ["growth", "middle", "value"])
    base["size"] = _terciles(base, "log_size", ["small", "mid", "large"])
    base["rate_beta_group"] = _terciles(base, "rate_beta", ["low rate beta", "mid", "high rate beta"])
    base["hml_beta_group"] = _terciles(base, "hml_beta", ["growth-like", "mid", "value-like"])

    reg = regime_table(um)
    for c in reg.columns:
        base[c] = base["meeting"].map(reg[c]).to_numpy(dtype=float)
    return base


# ----------------------------------------------------------------------------- gates

def run_gates(base: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(gate, period, w, label, how, term, per, p, n, note):
        if per:
            g, (coef, t) = max(per.items(), key=lambda kv: abs(kv[1][1]))
        else:
            g, coef, t = None, np.nan, np.nan
        rows.append(dict(gate=gate, period=period, window=w, trait=label, how=how, term=term, meetings=n,
                         p=p, strongest_group=g, coef=coef, t=t, note=note))

    for period, lo, hi in PERIODS:
        part = base[(base.meeting >= lo) & (base.meeting <= hi)]
        for w in WINDOWS:
            y, mk = f"y|{w}", f"mkt|{w}"
            within = part.assign(**{y: part[y] - part.groupby(["meeting", "sector"])[y].transform("mean")})
            for col, label in TRAITS.items():
                for how, d in (("raw", part),) + ((("within sector", within),) if col != "sector_name" else ()):
                    for term in ("s_policy", "s_info"):
                        add("G1", period, w, label, how, term, *group_test(d, col, y, [mk, "s_policy", "s_info"], term))
                    add("G1", period, w, label, how, "slope", *group_test(d, col, y, [mk, "s", "slope"], "slope"))
                    for r in REGIMES:
                        dd = d.assign(**{f"s_x_{r}": d["s"] * d[r], f"mkt_x_{r}": d[mk] * d[r]})
                        add("G2", period, w, label, how, f"s x {r}",
                            *group_test(dd, col, y, [mk, "s", r, f"mkt_x_{r}", f"s_x_{r}"], f"s_x_{r}"))
                    add("G3", period, w, label, how, "|y| x large", *magnitude_test(d, col, y))
            print(f"  {period} {w}: {len(rows)} tests so far", flush=True)
    return pd.DataFrame(rows)


def run_signed_gate(base: pd.DataFrame) -> pd.DataFrame:
    """G4 (v3): do groups differ in their response to hawkish (s > 0) or dovish (s < 0) surprises,
    with both halves of the surprise in the regression? Same pass bar as G1-G3."""
    base = base.assign(s_hawk=base["s"].clip(lower=0.0), s_dove=base["s"].clip(upper=0.0))
    rows = []
    for period, lo, hi in PERIODS:
        part = base[(base.meeting >= lo) & (base.meeting <= hi)]
        for w in WINDOWS:
            y, mk = f"y|{w}", f"mkt|{w}"
            within = part.assign(**{y: part[y] - part.groupby(["meeting", "sector"])[y].transform("mean")})
            for col, label in TRAITS.items():
                for how, d in (("raw", part),) + ((("within sector", within),) if col != "sector_name" else ()):
                    for term in ("s_hawk", "s_dove"):
                        per, p, n, note = group_test(d, col, y, [mk, "s_hawk", "s_dove"], term)
                        g, (coef, tt) = max(per.items(), key=lambda kv: abs(kv[1][1])) if per else (None, (np.nan, np.nan))
                        rows.append(dict(gate="G4", period=period, window=w, trait=label, how=how, term=term,
                                         meetings=n, p=p, strongest_group=g, coef=coef, t=tt, note=note))
    return pd.DataFrame(rows)


def verdict(tests: pd.DataFrame) -> pd.DataFrame:
    """Per gate: candidates passing p < 0.01 in both subperiods (within sector, or sector raw)."""
    eligible = tests[(tests["how"] == "within sector") | (tests["trait"] == "sector")]
    sub = eligible[eligible["period"].isin(SUBPERIODS)]
    wide = sub.pivot_table(index=["gate", "window", "trait", "how", "term"], columns="period", values="p")
    wide = wide.dropna(subset=list(SUBPERIODS))
    wide["passes"] = (wide[list(SUBPERIODS)] < P_PASS).all(axis=1)
    out = []
    for gate, g in wide.groupby(level="gate"):
        n_all = int(tests.loc[tests["gate"] == gate, "p"].notna().sum())
        n_p01 = int((tests.loc[tests["gate"] == gate, "p"] < P_PASS).sum())
        passing = g[g["passes"]].reset_index()
        out.append(dict(gate=gate, tests=n_all, p_below_01=n_p01, expected_by_chance=round(P_PASS * n_all, 1),
                        candidates=len(g), passing=len(passing),
                        passing_list="; ".join(f"{r.trait}/{r.how}/{r.term}/{r.window}" for r in passing.itertuples()),
                        result="PASS" if len(passing) else "FAIL"))
    return pd.DataFrame(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--through", default="2021-12-31")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--base-cache", help="parquet path: reuse the built rows if present, else write them")
    p.add_argument("--signed", action="store_true", help="run only the signed-shock gate G4 (v3)")
    args = p.parse_args(argv)
    if pd.Timestamp(args.through) > pd.Timestamp("2021-12-31"):
        raise SystemExit("gates use development meetings only (through 2021-12-31)")
    from pathlib import Path

    if args.base_cache and Path(args.base_cache).exists():
        base = pd.read_parquet(args.base_cache)
    else:
        base = build_base(args.through)
        if args.base_cache:
            base.to_parquet(args.base_cache)
    check = (base["y|day 0"] + base["beta"] * base["mkt|day 0"] - base["day0_panel"]).abs()
    print(f"{base.meeting.nunique()} meetings, {len(base):,} rows; day-0 rebuild max |diff| {check.max():.4f}; "
          f"information meetings {base.groupby('meeting')['s_info'].first().ne(0).mean():.0%}; "
          f"large meetings {int(base.groupby('meeting')['large'].first().sum())}", flush=True)
    if args.signed:
        tests = run_signed_gate(base)
        summary = verdict(tests)
        with pd.option_context("display.width", 250, "display.max_colwidth", 200):
            print("\n== verdict\n" + summary.to_string(index=False))
            print("\n== smallest p, 1994-2021\n" + tests[tests.period == "1994-2021"].nsmallest(15, "p").to_string(index=False))
        if not args.no_save:
            from fedcore.questions import specs
            from fedcore.results import ResultStore
            from fedenc.config import RESULTS

            ResultStore(RESULTS).save(specs()["Q3"], "v3-signed-gate", role="exploratory",
                                      params=dict(through=args.through, universe="top1500", p_pass=P_PASS),
                                      metrics={"G4_passing": int(summary["passing"].sum())},
                                      tables={"verdict": summary, "tests": tests},
                                      notes="Signed-shock gate G4 (docs/encoder/q3-v3-results.md); no model fit.")
        return summary, tests
    tests = run_gates(base)
    summary = verdict(tests)
    regimes = base.groupby("meeting")[list(REGIMES)].first().mean().rename("share_of_meetings").reset_index()
    with pd.option_context("display.width", 250, "display.max_colwidth", 200, "display.max_rows", 200):
        print("\n== verdict\n" + summary.to_string(index=False))
        print("\n== regime shares\n" + regimes.to_string(index=False))
        top = tests[tests["period"] == "1994-2021"].nsmallest(25, "p")
        print("\n== smallest p, 1994-2021\n" + top.to_string(index=False))
    if not args.no_save:
        from fedcore.questions import specs
        from fedcore.results import ResultStore
        from fedenc.config import RESULTS

        ResultStore(RESULTS).save(
            specs()["Q3"], "v2-gates", role="exploratory",
            params=dict(through=args.through, universe="top1500", large=LARGE, p_pass=P_PASS,
                        windows=list(WINDOWS), regimes=list(REGIMES), traits=list(TRAITS.values())),
            metrics={f"{r.gate}_passing": int(r.passing) for r in summary.itertuples()},
            tables={"verdict": summary, "tests": tests, "regime_shares": regimes},
            notes="Descriptive gates G1-G3 from docs/encoder/q3-v2-plan.md; no model fit.")
    return summary, tests


if __name__ == "__main__":
    main()
