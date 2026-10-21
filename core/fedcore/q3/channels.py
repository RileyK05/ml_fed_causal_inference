"""Firm-meeting channels beyond beta, from data on disk and free public sources (no new WRDS pull).

One row per row of the real Q3 panel (meeting, permno). Every value uses only information from
before the meeting: prices to the prior close, Compustat quarters reported strictly before the
meeting (and no older than ``STALE_DAYS``), ratings at most ``RATING_DAYS`` old, TNIC links from
April after their fiscal year. Missing stays NaN; nothing is filled.

Channels, grouped by what they stand in for (docs/encoder/q3-v5-plan.md section 2):

Market-implied (CRSP, GSW curve, release dates)
- ``rate_beta_2y``: % return per 1pp move in the 2-year zero yield, controlling for the market,
  over the prior 252 days (at least ``MIN_DAYS``).
- ``fed_beta_past``: slope of the firm's cross-firm-centered abnormal Fed-day return
  (return - beta63 x market, beta from the day before) on s, over the scheduled meetings in the
  prior ``FED_YEARS`` years (at least ``MIN_MEETINGS``). The firm's own history of beyond-beta
  Fed response.
- ``announce_premium``: log ratio of mean |abnormal return| on FOMC, CPI and jobs-report days to
  other days, prior ``PREMIUM_DAYS`` trading days. Stands in for the option-implied Fed-day move.
- ``vol_runup``: log ratio of return volatility over the last 10 days to the 63 days before them.

Credit and funding (Compustat x CRSP)
- ``distance_to_default``: Bharath-Shumway naive Merton distance to default. Stands in for CDS.
- ``int_coverage``: trailing 4-quarter operating income over interest expense, clipped to
  [-10, 50].
- ``rate_paid``: trailing 4-quarter interest expense over average debt (debt at least 1% of
  assets; rates outside [0, 30%] are treated as data errors).
- ``pass_through``: slope of the quarterly change in annualized interest paid per dollar of debt
  on the change in the 1-year yield, last 12 quarters (at least 8). Floating-rate debt reprices
  with short rates, so this stands in for the floating-rate share.
- ``st_debt_share``: debt in current liabilities over total debt (refinancing need).
- ``debt_issue_4q``, ``debt_repay_4q``: trailing 4-quarter long-term debt issuance and repayment
  over assets.
- ``fcf_4q``: trailing 4-quarter operating cash flow minus capex, over assets.
- ``rating_num``: S&P long-term issuer rating, AAA = 1 to D = 22 (Compustat stops in 2017).

Duration
- ``implied_duration``: Dechow-Sloan-Soliman (2004) implied equity duration in years.

Peers (Hoberg-Phillips TNIC-3)
- ``tnic_n_peers``: TNIC-3 peers in the same meeting's panel.
- ``tnic_peer_fed_beta``: mean ``fed_beta_past`` of those peers, and ``fed_beta_vs_peers`` the
  firm's own minus it.
- ``tnic_peer_ret63``: peers' mean return relative to the market over the prior 63 days
  (text-based peer stress).

Also ``earnings_day``: a Compustat report date on the meeting day or the trading day before.

The table is derived from licensed data, so its cache (``processed/q3_channels/``) is gitignored.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from fedcore.config import PROCESSED
from fedcore.data import get, load
from fedcore.data.loaders import fingerprint
from fedcore.q3.contracts import BETA63
from fedcore.q3.synthetic import _window_sum

STALE_DAYS = 365
RATING_DAYS = 100
MIN_DAYS = 126
FED_YEARS = 8
MIN_MEETINGS = 16
PREMIUM_DAYS = 756
MIN_MACRO_DAYS = 12
PASS_QUARTERS, PASS_MIN = 12, 8
MAX_RATE = 0.30  # interest paid per dollar of debt above this is a data error
# Dechow-Sloan-Soliman (2004) parameters: ROE and sales growth mean-revert to these levels.
DSS_R, DSS_ROE_BAR, DSS_ROE_RHO, DSS_G_BAR, DSS_G_RHO, DSS_T = 0.12, 0.12, 0.57, 0.06, 0.24, 10
RATING_SCALE = {r: k + 1 for k, r in enumerate(
    ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-",
     "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC", "C", "D"])}
RATING_SCALE["SD"] = RATING_SCALE["D"]
BUILD_VERSION = 1
CACHE = PROCESSED / "q3_channels"
INPUTS = ("crsp_daily", "ff_factors_daily", "gsw_yields", "treasury_1y", "bls_cpi_release_dates",
          "bls_macro_release_dates", "comp_fundq", "comp_fundq_top1500", "comp_fundq_cash_top1500",
          "comp_fundq_funding_top1500", "comp_ratings_top1500", "ccm_link", "tnic3")

CHANNEL_NAMES = (
    "rate_beta_2y", "fed_beta_past", "announce_premium", "vol_runup",
    "distance_to_default", "int_coverage", "rate_paid", "pass_through", "st_debt_share",
    "debt_issue_4q", "debt_repay_4q", "fcf_4q", "rating_num", "implied_duration",
    "tnic_n_peers", "tnic_peer_fed_beta", "fed_beta_vs_peers", "tnic_peer_ret63", "earnings_day",
)


@dataclass(frozen=True)
class ChannelInputs:
    """Everything the build reads, already loaded; tests pass small frames."""

    grid: dict                 # fedcore.q3.real._grid output: calendar, permnos, channels, observed, ret, mktcap, mkt
    rows: pd.DataFrame         # panel rows: meeting, permno, gvkey (and firm, pos into the grid)
    meetings: pd.DataFrame     # all scheduled statements: announcement_date, s
    macro_dates: pd.DatetimeIndex
    y2: pd.Series              # 2-year zero yield, percent, by date
    y1: pd.Series              # 1-year zero yield, percent, by date
    comp: pd.DataFrame         # quarterly Compustat (see compustat_quarters)
    ratings: pd.DataFrame      # gvkey, datadate, splticrm
    tnic: pd.DataFrame | None  # gvkey1, gvkey2, usable_from, year


# ----------------------------------------------------------------------------- market-implied

def rate_beta(grid: dict, rows: pd.DataFrame, y2: pd.Series, window: int = 252) -> np.ndarray:
    """Coefficient on the 2y yield change in ret ~ 1 + mkt + dy2, per row, prior ``window`` days."""
    cal = grid["calendar"]
    dy = y2.reindex(cal).to_numpy(dtype=np.float64)
    dy = np.r_[np.nan, np.diff(dy)]  # NaN where either day is missing: never bridged
    out = np.full(len(rows), np.nan)
    for p, g in rows.groupby("pos"):
        sl = slice(p - window, p)
        X = np.column_stack((np.ones(window), grid["mkt"][sl], dy[sl]))
        firm = g["firm"].to_numpy()
        y = grid["ret"][firm, sl]
        w = (grid["observed"][firm, sl] & np.isfinite(X).all(axis=1)[None, :]).astype(np.float64)
        Xz, yz = np.nan_to_num(X), np.nan_to_num(y)
        XtX = np.einsum("fd,dk,dl->fkl", w, Xz, Xz)
        Xty = np.einsum("fd,dk,fd->fk", w, Xz, yz)
        ok = (w.sum(axis=1) >= MIN_DAYS) & (np.abs(np.linalg.det(XtX)) > 1e-10)
        coef = np.full((len(firm), 3), np.nan)
        if ok.any():
            coef[ok] = np.linalg.solve(XtX[ok], Xty[ok][..., None])[..., 0]
        out[g.index.to_numpy()] = coef[:, 2]
    return out


def abnormal_returns(grid: dict) -> np.ndarray:
    """Firm return minus beta63 (as of the day before) times the market, percent; NaN when unobserved."""
    beta_prev = np.full_like(grid["ret"], np.nan)
    beta_prev[:, 1:] = grid["channels"][:, :-1, BETA63]
    abn = grid["ret"] - beta_prev * grid["mkt"][None, :]
    return np.where(grid["observed"], abn, np.nan)


def fed_beta_past(grid: dict, rows: pd.DataFrame, meetings: pd.DataFrame, abn: np.ndarray) -> np.ndarray:
    """Slope of centered abnormal Fed-day returns on s over the prior ``FED_YEARS`` years."""
    cal = grid["calendar"]
    m = meetings.dropna(subset=["s"])
    pos = cal.get_indexer(pd.DatetimeIndex(m["announcement_date"]))
    m, pos = m[pos >= 0], pos[pos >= 0]
    s = m["s"].to_numpy(dtype=np.float64)
    dates = pd.DatetimeIndex(m["announcement_date"])
    A = abn[:, pos]
    A = A - np.nanmean(A, axis=0, keepdims=True)  # cross-firm centering removes the common move
    out = np.full(len(rows), np.nan)
    for d, g in rows.groupby("meeting"):
        d = pd.Timestamp(d)
        past = (dates < d) & (dates >= d - pd.DateOffset(years=FED_YEARS))
        a = A[g["firm"].to_numpy()][:, past]
        x = s[past]
        w = np.isfinite(a)
        n = w.sum(axis=1)
        xm = np.where(w, x[None, :], 0.0).sum(axis=1) / np.maximum(n, 1)
        am = np.where(w, a, 0.0).sum(axis=1) / np.maximum(n, 1)
        xc = np.where(w, x[None, :] - xm[:, None], 0.0)
        ac = np.where(w, a - am[:, None], 0.0)
        sxx = (xc * xc).sum(axis=1)
        slope = np.divide((xc * ac).sum(axis=1), sxx, out=np.full(len(n), np.nan), where=sxx > 1e-12)
        out[g.index.to_numpy()] = np.where(n >= MIN_MEETINGS, slope, np.nan)
    return out


def announce_premium(grid: dict, rows: pd.DataFrame, macro_days: np.ndarray, abn: np.ndarray) -> np.ndarray:
    """log(mean |abn| on macro days / mean |abn| on other days), prior ``PREMIUM_DAYS`` days."""
    obs = np.isfinite(abn)
    a = np.where(obs, np.abs(abn), 0.0)
    is_macro = macro_days[None, :]
    sum_m = _window_sum(a * is_macro, PREMIUM_DAYS)
    cnt_m = _window_sum((obs & is_macro).astype(np.float64), PREMIUM_DAYS)
    sum_o = _window_sum(a * ~is_macro, PREMIUM_DAYS)
    cnt_o = _window_sum((obs & ~is_macro).astype(np.float64), PREMIUM_DAYS)
    f, p = rows["firm"].to_numpy(), rows["pos"].to_numpy() - 1  # window ends at the prior close
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = (sum_m[f, p] / cnt_m[f, p]) / (sum_o[f, p] / cnt_o[f, p])
        out = np.log(ratio)
    return np.where((cnt_m[f, p] >= MIN_MACRO_DAYS) & np.isfinite(out), out, np.nan)


def vol_runup(grid: dict, rows: pd.DataFrame) -> np.ndarray:
    f, p = rows["firm"].to_numpy(), rows["pos"].to_numpy()
    ret = np.where(grid["observed"], grid["ret"], np.nan)
    recent = np.stack([ret[f, p - k] for k in range(1, 11)], axis=1)
    before = np.stack([ret[f, p - k] for k in range(11, 74)], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r = np.nanstd(recent, axis=1, ddof=1)
        b = np.nanstd(before, axis=1, ddof=1)
        out = np.log(r / b)
    ok = (np.isfinite(recent).sum(axis=1) >= 8) & (np.isfinite(before).sum(axis=1) >= 40)
    return np.where(ok & np.isfinite(out), out, np.nan)


# ----------------------------------------------------------------------------- Compustat

_KEY = ["gvkey", "datadate", "fyearq", "fqtr"]


def compustat_quarters(fundq: pd.DataFrame, cash: pd.DataFrame, funding: pd.DataFrame,
                       y1: pd.Series) -> pd.DataFrame:
    """One row per firm-quarter with the trailing-4-quarter and pass-through measures, keyed by rdq."""
    f = fundq.merge(cash.drop(columns=["rdq"], errors="ignore"), on=_KEY, how="left").merge(
        funding.drop(columns=["rdq"], errors="ignore"), on=_KEY, how="left")
    f = f.assign(gvkey=f["gvkey"].astype("string"), datadate=pd.to_datetime(f["datadate"]),
                 rdq=pd.to_datetime(f["rdq"]))
    num = [c for c in f.columns if c not in ("gvkey", "datadate", "rdq")]
    f[num] = f[num].apply(pd.to_numeric, errors="coerce").astype("float64")
    f = f.dropna(subset=["fyearq", "fqtr"]).sort_values(["gvkey", "fyearq", "fqtr"]).drop_duplicates(
        ["gvkey", "fyearq", "fqtr"], keep="last").reset_index(drop=True)
    for c in ("oancfy", "capxy", "dltry", "dltisy"):  # year-to-date -> quarter
        if c in f:
            prev = f.groupby(["gvkey", "fyearq"])[c].shift(1)
            f[c[:-1] + "_q"] = np.where(f["fqtr"] > 1, f[c] - prev, f[c])
    g = f.groupby("gvkey")

    def trail(col: str, n: int = 4) -> pd.Series:
        return g[col].transform(lambda x: x.rolling(n, min_periods=n).sum())

    at = f["atq"].where(f["atq"] > 0)
    debt = f[["dlttq", "dlcq"]].sum(axis=1, min_count=1)
    # Rates paid on a sliver of debt are noise (interest over near-zero debt): need debt >= 1% of assets.
    debt = debt.where((f["dlttq"].fillna(0) >= 0) & (f["dlcq"].fillna(0) >= 0))
    meaningful = debt >= 0.01 * at
    f["debt"] = debt
    debt_avg4 = g["debt"].transform(lambda x: x.rolling(4, min_periods=4).mean())
    xint4, oiadp4 = trail("xintq"), trail("oiadpq")
    with np.errstate(invalid="ignore", divide="ignore"):
        f["int_coverage"] = (oiadp4 / xint4.where(xint4 > 0)).clip(-10, 50)
        f["rate_paid"] = (xint4 / debt_avg4.where(debt_avg4 > 0)).where(meaningful)
        f["rate_paid"] = f["rate_paid"].where(f["rate_paid"].between(0, MAX_RATE))
        f["st_debt_share"] = f["dlcq"] / debt.where(debt > 0)
        f["debt_issue_4q"] = trail("dltis_q") / at
        f["debt_repay_4q"] = trail("dltr_q") / at
        f["fcf_4q"] = (trail("oancf_q") - trail("capx_q")) / at
        f["sale_4q"], f["ni_4q"] = trail("saleq"), trail("niq")

    # pass-through: d(4 * xintq / average debt) on d(average 1y yield over the fiscal quarter)
    y1 = y1.sort_index()
    y1_q = y1.rolling("91D").mean()
    f["y1_q"] = y1_q.reindex(f["datadate"], method="ffill").to_numpy() if len(y1_q) else np.nan
    prev_debt = g["debt"].shift(1)
    avg_debt = (f["debt"] + prev_debt) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        f["paid_q"] = 4.0 * f["xintq"] / avg_debt.where(avg_debt > 0) * 100.0  # percent
    f["paid_q"] = f["paid_q"].where(meaningful & f["paid_q"].between(0, 100 * MAX_RATE))
    f["d_paid"] = f["paid_q"] - g["paid_q"].shift(1)
    f["d_y1"] = f["y1_q"] - g["y1_q"].shift(1)
    f["pass_through"] = _rolling_slope(f, "d_y1", "d_paid", PASS_QUARTERS, PASS_MIN)
    return f


def _rolling_slope(f: pd.DataFrame, x: str, y: str, n: int, min_n: int) -> pd.Series:
    out = np.full(len(f), np.nan)
    for _, idx in f.groupby("gvkey").indices.items():
        xv, yv = f[x].to_numpy()[idx], f[y].to_numpy()[idx]
        for k in range(len(idx)):
            lo = max(0, k - n + 1)
            xs, ys = xv[lo:k + 1], yv[lo:k + 1]
            ok = np.isfinite(xs) & np.isfinite(ys)
            if ok.sum() >= min_n:
                xc = xs[ok] - xs[ok].mean()
                sxx = (xc * xc).sum()
                if sxx > 1e-12:
                    out[idx[k]] = (xc * (ys[ok] - ys[ok].mean())).sum() / sxx
    return pd.Series(out, index=f.index)


def implied_duration(price: np.ndarray, book: np.ndarray, roe: np.ndarray, growth: np.ndarray) -> np.ndarray:
    """Dechow-Sloan-Soliman (2004) equity duration from price, book equity, ROE and sales growth."""
    r, T = DSS_R, DSS_T
    roe = np.clip(roe, -1.0, 1.0)
    growth = np.clip(growth, -0.5, 1.0)
    bv = book.astype(np.float64).copy()
    pv = np.zeros_like(bv)
    tpv = np.zeros_like(bv)
    for t in range(1, T + 1):
        roe_t = DSS_ROE_BAR + DSS_ROE_RHO ** t * (roe - DSS_ROE_BAR)
        g_t = DSS_G_BAR + DSS_G_RHO ** t * (growth - DSS_G_BAR)
        cf = bv * (roe_t - g_t)
        bv = bv * (1 + g_t)
        disc = cf / (1 + r) ** t
        pv += disc
        tpv += t * disc
    with np.errstate(invalid="ignore", divide="ignore"):
        dur = tpv / price + (T + (1 + r) / r) * (price - pv) / price
    return np.where((price > 0) & (book > 0), dur, np.nan)


def _asof(rows: pd.DataFrame, table: pd.DataFrame, on: str, cols: list[str], days: int) -> pd.DataFrame:
    """Latest ``table`` row per gvkey with ``on`` strictly before the meeting, at most ``days`` old."""
    linked = rows.dropna(subset=["gvkey"]).reset_index()
    t = table.dropna(subset=[on]).sort_values(on)
    merged = pd.merge_asof(linked.sort_values("meeting"), t[["gvkey", on] + cols], left_on="meeting",
                           right_on=on, by="gvkey", allow_exact_matches=False,
                           tolerance=pd.Timedelta(days=days))
    return merged.set_index("index").reindex(rows.index)


def rating_number(code: pd.Series) -> pd.Series:
    return code.astype("string").str.strip().map(RATING_SCALE).astype("float64")


# ----------------------------------------------------------------------------- peers

def tnic_peers(rows: pd.DataFrame, tnic: pd.DataFrame) -> pd.DataFrame:
    """(row, peer_row) pairs: TNIC-3 peers of each firm present in the same meeting's panel.

    Links of fiscal year Y are used for meetings on or after ``usable_from`` (April 1 of Y+1),
    the latest such year per meeting."""
    years = np.sort(tnic["year"].unique())
    out = []
    for d, g in rows.dropna(subset=["gvkey"]).groupby("meeting"):
        usable = years[pd.to_datetime([f"{y + 1}-04-01" for y in years]) <= pd.Timestamp(d)]
        if usable.size == 0:
            continue
        links = tnic[tnic["year"] == usable[-1]]
        here = pd.Series(g.index, index=g["gvkey"].astype("string").to_numpy())
        here = here[~here.index.duplicated()]
        j = links[links["gvkey1"].isin(here.index) & links["gvkey2"].isin(here.index)]
        out.append(pd.DataFrame({"row": here.loc[j["gvkey1"]].to_numpy(), "peer": here.loc[j["gvkey2"]].to_numpy()}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame({"row": [], "peer": []}, dtype="int64")


def peer_means(pairs: pd.DataFrame, values: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(number of peers with a value, mean over them) per row."""
    v = values[pairs["peer"].to_numpy()]
    ok = np.isfinite(v)
    rows = pairs["row"].to_numpy()[ok]
    cnt = np.bincount(rows, minlength=n).astype(np.float64)
    tot = np.bincount(rows, weights=v[ok], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return cnt, np.where(cnt > 0, tot / cnt, np.nan)


# ----------------------------------------------------------------------------- build

def build_channels(inp: ChannelInputs) -> pd.DataFrame:
    rows, grid = inp.rows, inp.grid
    out = pd.DataFrame({"meeting": rows["meeting"].to_numpy(), "permno": rows["permno"].to_numpy()})
    n = len(rows)

    abn = abnormal_returns(grid)
    macro = np.isin(grid["calendar"], inp.macro_dates.union(pd.DatetimeIndex(inp.meetings["announcement_date"])))
    out["rate_beta_2y"] = rate_beta(grid, rows, inp.y2)
    out["fed_beta_past"] = fed_beta_past(grid, rows, inp.meetings, abn)
    out["announce_premium"] = announce_premium(grid, rows, macro, abn)
    out["vol_runup"] = vol_runup(grid, rows)

    q = _asof(rows, inp.comp, "rdq",
              ["int_coverage", "rate_paid", "pass_through", "st_debt_share", "debt_issue_4q", "debt_repay_4q",
               "fcf_4q", "dlcq", "dlttq", "ceqq", "sale_4q", "ni_4q"], STALE_DAYS)
    for c in ("int_coverage", "rate_paid", "pass_through", "st_debt_share", "debt_issue_4q", "debt_repay_4q", "fcf_4q"):
        out[c] = q[c].to_numpy(dtype=np.float64, na_value=np.nan)

    # distance to default (Bharath-Shumway 2008, naive)
    f, p = rows["firm"].to_numpy(), rows["pos"].to_numpy()
    ret = np.where(grid["observed"], grid["ret"], np.nan)
    window = np.stack([ret[f, p - k] for k in range(1, 253)], axis=1) / 100.0
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sigma_e = np.nanstd(window, axis=1, ddof=1) * np.sqrt(252.0)
        mu = np.nansum(np.log1p(window), axis=1)
        equity = grid["mktcap"][f, p - 1]
        face = (q["dlcq"] + 0.5 * q["dlttq"]).to_numpy(dtype=np.float64, na_value=np.nan)
        value = equity + face
        sigma_v = equity / value * sigma_e + face / value * (0.05 + 0.25 * sigma_e)
        dd = (np.log(value / face) + mu - 0.5 * sigma_v ** 2) / sigma_v
    enough = np.isfinite(window).sum(axis=1) >= MIN_DAYS
    out["distance_to_default"] = np.where(enough & (face > 0) & (equity > 0) & np.isfinite(dd), dd, np.nan)

    # implied duration: book equity, trailing ROE and sales growth from the as-of quarter
    comp = inp.comp
    lag = comp.groupby("gvkey")["sale_4q"].shift(4)
    growth_q = comp.assign(growth=comp["sale_4q"] / lag.where(lag > 0) - 1.0)
    gq = _asof(rows, growth_q, "rdq", ["growth"], STALE_DAYS)["growth"].to_numpy(dtype=np.float64, na_value=np.nan)
    book = q["ceqq"].to_numpy(dtype=np.float64, na_value=np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        roe = q["ni_4q"].to_numpy(dtype=np.float64, na_value=np.nan) / np.where(book > 0, book, np.nan)
    out["implied_duration"] = implied_duration(equity, book, roe, gq)

    rat = inp.ratings.dropna(subset=["splticrm"]).assign(
        gvkey=lambda d: d["gvkey"].astype("string"), datadate=lambda d: pd.to_datetime(d["datadate"]))
    rat = rat.assign(rating_num=rating_number(rat["splticrm"]))
    out["rating_num"] = _asof(rows, rat, "datadate", ["rating_num"], RATING_DAYS)["rating_num"].to_numpy(
        dtype=np.float64, na_value=np.nan)

    if inp.tnic is not None and len(inp.tnic):
        pairs = tnic_peers(rows, inp.tnic)
        cnt, peer_fb = peer_means(pairs, out["fed_beta_past"].to_numpy(), n)
        _, peer_ret = peer_means(pairs, _ret_rel_63(grid, rows), n)
        n_peers = np.bincount(pairs["row"].to_numpy(dtype=np.int64), minlength=n).astype(np.float64)
        has_tnic = rows["gvkey"].notna().to_numpy()
        out["tnic_n_peers"] = np.where(has_tnic, n_peers, np.nan)
        out["tnic_peer_fed_beta"] = peer_fb
        out["fed_beta_vs_peers"] = out["fed_beta_past"] - peer_fb
        out["tnic_peer_ret63"] = peer_ret
    else:
        for c in ("tnic_n_peers", "tnic_peer_fed_beta", "fed_beta_vs_peers", "tnic_peer_ret63"):
            out[c] = np.nan

    rdq = inp.comp.dropna(subset=["rdq"])
    cal = grid["calendar"]
    prev_day = pd.Series(cal[np.maximum(p - 1, 0)], index=rows.index)
    keys = set(zip(rdq["gvkey"].astype("string"), pd.DatetimeIndex(rdq["rdq"])))
    gv = rows["gvkey"].astype("string")
    out["earnings_day"] = [
        float((g, m) in keys or (g, d) in keys) if isinstance(g, str) else np.nan
        for g, m, d in zip(gv, pd.DatetimeIndex(rows["meeting"]), prev_day)
    ]
    return out[["meeting", "permno", *CHANNEL_NAMES]]


def _ret_rel_63(grid: dict, rows: pd.DataFrame) -> np.ndarray:
    f, p = rows["firm"].to_numpy(), rows["pos"].to_numpy()
    rel = np.where(grid["observed"], grid["ret"] - grid["mkt"][None, :], np.nan)
    w = np.stack([rel[f, p - k] for k in range(1, 64)], axis=1)
    ok = np.isfinite(w).sum(axis=1) >= 40
    with np.errstate(invalid="ignore"):
        return np.where(ok, np.nansum(w, axis=1), np.nan)


# ----------------------------------------------------------------------------- real data

def read_channel_inputs() -> ChannelInputs:
    from fedcore.q3.real import _grid, load_real_panel, read_inputs

    base = read_inputs()
    grid = _grid(base.daily, base.market)
    _, rows, _ = load_real_panel()
    rows = rows[["meeting", "permno", "gvkey"]].copy()
    rows["meeting"] = pd.to_datetime(rows["meeting"])
    rows["firm"] = np.searchsorted(grid["permnos"], rows["permno"].to_numpy(dtype=np.int64))
    rows["pos"] = grid["calendar"].get_indexer(pd.DatetimeIndex(rows["meeting"]))
    rows["gvkey"] = rows["gvkey"].astype("string")

    meetings = base.meetings[["announcement_date", "s"]]  # scheduled statements only (meeting_table)
    macro_dates = pd.DatetimeIndex(pd.concat([load("bls_cpi_release_dates")["release_date"],
                                              load("bls_macro_release_dates")["release_date"]]).unique())
    gsw = load("gsw_yields").set_index("Date")
    key = ["gvkey", "datadate", "fyearq", "fqtr"]
    fundq = pd.concat([load("comp_fundq_top1500"), load("comp_fundq")], ignore_index=True)
    fundq = fundq.assign(gvkey=fundq["gvkey"].astype("string"), datadate=pd.to_datetime(fundq["datadate"]))
    fundq = fundq.drop_duplicates(key)
    cash, funding = load("comp_fundq_cash_top1500"), load("comp_fundq_funding_top1500")
    for t in (cash, funding):
        t["gvkey"], t["datadate"] = t["gvkey"].astype("string"), pd.to_datetime(t["datadate"])
    y1 = gsw["SVENY01"].dropna()
    comp = compustat_quarters(fundq, cash, funding, y1)
    tnic = load("tnic3") if get("tnic3").exists else None
    return ChannelInputs(grid=grid, rows=rows.reset_index(drop=True), meetings=meetings, macro_dates=macro_dates,
                         y2=gsw["SVENY02"], y1=y1, comp=comp, ratings=load("comp_ratings_top1500"), tnic=tnic)


def load_channels(rebuild: bool = False) -> pd.DataFrame:
    """The cached channel table, rebuilt when missing, forced, or when an input changed."""
    meta_file = CACHE / "meta.json"
    prints = {name: fingerprint(name) for name in INPUTS if get(name).exists}
    if not rebuild and meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if meta.get("inputs") == prints and meta.get("build_version") == BUILD_VERSION:
            return pd.read_parquet(CACHE / "channels.parquet")
    out = build_channels(read_channel_inputs())
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CACHE / "channels.parquet", index=False)
    meta = {"built": date.today().isoformat(), "build_version": BUILD_VERSION, "inputs": prints,
            "n_rows": len(out), "observed_share": out[list(CHANNEL_NAMES)].notna().mean().round(4).to_dict()}
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out
