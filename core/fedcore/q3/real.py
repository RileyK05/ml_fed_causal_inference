"""Real Q3 firm-meeting panel: S&P 500 members at each scheduled FOMC statement, 1994-2025.

Built from the registered WRDS datasets (core/data/PROVENANCE.md section 7) with the
synthetic panel's channel definitions, so every arm runs on it unchanged.

- Rows: every firm in the S&P 500 on the meeting date (point-in-time membership, so firms
  that later left or delisted are included) with a meeting-day return and at least one
  observed history day. The ``top1500`` universe instead takes the 1,500 largest firms by
  market cap at the prior close from a broader pull, flagging S&P members in ``in_sp500``.
- History: the 252 trading days before the meeting; ``history[:, -1]`` is the prior close.
  A day is observed when the firm has a return and positive volume and the market factor
  exists. Gaps stay NaN: nothing is filled.
- Target: meeting-day close-to-close return in percent (CRSP CIZ, delisting-inclusive).
- Shock: ``s = 10 * STMT`` (units of 10bp), scheduled statements only -- the Q1 definition.
- Fundamentals: the latest Compustat quarter whose report date ``rdq`` is strictly before
  the meeting, treated as missing once older than ``STALE_DAYS``.
- Context: the Q1 state controls plus trailing 21-day market volatility, as of the prior day.

The inputs are licensed, so the cached build lives in ``core/data/processed/q3_panel/``,
which is gitignored. ``load_real_panel`` rebuilds it when any input file changes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from fedcore.config import PROCESSED
from fedcore.data import get, load
from fedcore.data.loaders import fingerprint
from fedcore.q3.contracts import BETA63, CHANNELS, DLOGVOL, L, Q3Panel, RET, RET_REL, RVOL21
from fedcore.q3.synthetic import (
    RVOL_FALLBACK,
    RVOL_WINDOW,
    _dlog_volume,
    _rolling_beta,
    _rolling_std,
)

FUNDAMENTAL_NAMES = ("leverage", "log_size", "profitability", "cash", "book_to_market", "fund_age")
CONTEXT_NAMES = ("y1_level", "y1_chg_21d", "y1_chg_63d", "st_prev", "me_prev", "st_mean3", "mkt_rvol21")
STALE_DAYS = 365
# Bump whenever build logic changes: cached panels from another version are rebuilt.
BUILD_VERSION = 2
# The synthetic panel falls back to the firm's true beta; real data has none, so use the market's.
BETA_FALLBACK = 1.0
CACHE = PROCESSED / "q3_panel"


@dataclass(frozen=True)
class Universe:
    """Which firms are rows, and which raw tables feed them."""

    name: str
    top_n: int | None  # None: S&P 500 members on the meeting date; else the top_n by prior-close market cap
    daily: str
    link: str
    fundq: str
    gics: str
    cache: Path | None  # None: the module-level CACHE

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.daily, "sp500_members", "ff_factors_daily", self.link, self.fundq, self.gics,
                "usmpd_statements", "mps_surprises", "treasury_1y")


UNIVERSES = {
    "sp500": Universe("sp500", None, "crsp_daily", "ccm_link", "comp_fundq", "comp_gics", None),
    "top1500": Universe("top1500", 1500, "crsp_daily_top1500", "ccm_link_top1500", "comp_fundq_top1500",
                        "comp_gics_top1500", PROCESSED / "q3_panel_top1500"),
}
INPUTS = UNIVERSES["sp500"].inputs
_ARRAYS = (
    "meeting", "firm_id", "history", "history_mask", "fundamentals", "fundamentals_mask",
    "context", "shock", "target",
)


@dataclass(frozen=True)
class Inputs:
    """Raw tables, already loaded. Tests pass small frames; ``read_inputs`` loads the real ones."""

    daily: pd.DataFrame      # permno, dlycaldt, dlyret, dlyvol, dlyprc, shrout
    members: pd.DataFrame    # permno, mbrstartdt, mbrenddt
    market: pd.DataFrame     # date, mktrf, rf (decimals)
    link: pd.DataFrame       # gvkey, lpermno, linkprim, linkdt, linkenddt
    fundq: pd.DataFrame      # gvkey, datadate, rdq, atq, ceqq, cheq, dlttq, dlcq, niq
    gics: pd.DataFrame       # gvkey, gsector, indfrom, indthru
    meetings: pd.DataFrame   # announcement_date, s, and the Q1 context columns


def meeting_table() -> pd.DataFrame:
    """Scheduled statements with ``s = 10 * STMT`` and the Q1 state controls (prior day)."""
    stmt = load("usmpd_statements").rename(columns={"Date": "announcement_date"})
    stmt = stmt.loc[stmt["Unscheduled"] == 0, ["announcement_date"]]
    mps = load("mps_surprises").rename(columns={"Date": "announcement_date"}).sort_values("announcement_date")
    hist = mps[["announcement_date", "STMT", "ME"]].copy()
    hist["st_prev"] = hist["STMT"].shift(1)  # previous statement event, any type
    hist["me_prev"] = hist["ME"].shift(1)
    hist["st_mean3"] = hist["STMT"].shift(1).rolling(3).mean()

    y1 = load("treasury_1y").rename(columns={"Date": "date", "SVENY01": "y1_level"}).sort_values("date")
    y1["y1_chg_21d"] = y1["y1_level"] - y1["y1_level"].shift(21)
    y1["y1_chg_63d"] = y1["y1_level"] - y1["y1_level"].shift(63)

    df = stmt.merge(hist, on="announcement_date", how="left").sort_values("announcement_date")
    df = pd.merge_asof(
        df, y1[["date", "y1_level", "y1_chg_21d", "y1_chg_63d"]],
        left_on="announcement_date", right_on="date", allow_exact_matches=False,
    ).drop(columns="date")
    df["s"] = 10.0 * df["STMT"]
    return df.reset_index(drop=True)


def read_inputs(universe: str = "sp500") -> Inputs:
    u = UNIVERSES[universe]

    def dates(df: pd.DataFrame, *cols: str) -> pd.DataFrame:
        for col in cols:
            df[col] = pd.to_datetime(df[col])
        return df

    def gvkey(df: pd.DataFrame) -> pd.DataFrame:
        df["gvkey"] = df["gvkey"].astype(str)  # one plain type so the as-of joins match
        return df

    link = gvkey(dates(load(u.link), "linkenddt"))
    link["lpermno"] = link["lpermno"].astype("int64")
    return Inputs(
        daily=load(u.daily, columns=["permno", "dlyret", "dlyvol", "dlyprc", "shrout"]),
        members=dates(load("sp500_members"), "mbrenddt"),
        market=load("ff_factors_daily", columns=["mktrf", "rf"]),
        link=link,
        fundq=gvkey(dates(load(u.fundq), "rdq")),
        gics=gvkey(dates(load(u.gics), "indthru")),
        meetings=meeting_table(),
    )


def _grid(daily: pd.DataFrame, market: pd.DataFrame) -> dict:
    """Dense firm x trading-day arrays and the five channels."""
    daily = daily.assign(dlycaldt=pd.to_datetime(daily["dlycaldt"]))
    calendar = pd.DatetimeIndex(np.sort(daily["dlycaldt"].unique()))
    permnos = np.sort(daily["permno"].unique()).astype(np.int64)
    fi = np.searchsorted(permnos, daily["permno"].to_numpy(dtype=np.int64))
    di = calendar.get_indexer(daily["dlycaldt"])
    shape = (permnos.size, calendar.size)

    def dense(col: str, scale: float = 1.0) -> np.ndarray:
        out = np.full(shape, np.nan)
        out[fi, di] = daily[col].to_numpy(dtype=np.float64, na_value=np.nan) * scale
        return out

    ret = dense("dlyret", 100.0)
    vol = dense("dlyvol")
    mktcap = np.abs(dense("dlyprc")) * dense("shrout") / 1000.0  # shrout is in thousands -> $ millions

    m = market.assign(date=pd.to_datetime(market["date"])).set_index("date")
    mkt = (100.0 * (m["mktrf"] + m["rf"])).reindex(calendar).to_numpy(dtype=np.float64, na_value=np.nan)

    observed = np.isfinite(ret) & (vol > 0) & np.isfinite(mkt)[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        log_vol = np.log(np.where(vol > 0, vol, np.nan))
        channels = np.empty(shape + (len(CHANNELS),), dtype=np.float32)
        channels[:, :, RET] = ret
        channels[:, :, DLOGVOL] = _dlog_volume(log_vol, observed)
        channels[:, :, RVOL21] = _rolling_std(ret, observed, 21, RVOL_FALLBACK)
        channels[:, :, RET_REL] = ret - mkt[None, :]
        channels[:, :, BETA63] = _rolling_beta(
            ret, mkt, observed, RVOL_WINDOW, np.full(permnos.size, BETA_FALLBACK)
        )
    channels[~observed] = np.nan
    return {
        "calendar": calendar, "permnos": permnos, "channels": channels, "observed": observed,
        "ret": ret, "mktcap": mktcap, "mkt": mkt,
    }


def _rows(grid: dict, members: pd.DataFrame, meetings: pd.DataFrame,
          top_n: int | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """One row per (meeting, firm) that has a target and some history.

    Firms are the S&P 500 members on the meeting date, or with ``top_n`` the ``top_n`` largest
    by market cap at the prior close. ``in_sp500`` flags members either way.
    """
    calendar, permnos = grid["calendar"], grid["permnos"]
    end = members["mbrenddt"].fillna(pd.Timestamp.max)
    rows, dropped = [], []
    for m in meetings.itertuples(index=False):
        d = pd.Timestamp(m.announcement_date)
        p = calendar.get_indexer([d])[0]
        if p < L:
            dropped.append({"announcement_date": d.date().isoformat(),
                            "reason": "not a trading day in the data" if p < 0 else "under 252 prior days"})
            continue
        in_index = members.loc[(members["mbrstartdt"] <= d) & (end >= d), "permno"].to_numpy(dtype=np.int64)
        if top_n is None:
            idx = np.searchsorted(permnos, in_index)
            idx = idx[(idx < permnos.size) & (permnos[np.minimum(idx, permnos.size - 1)] == in_index)]
            idx = np.unique(idx)
        else:
            cap = grid["mktcap"][:, p - 1]
            ok = np.flatnonzero(np.isfinite(cap) & (cap > 0))
            idx = np.sort(ok[np.argsort(-cap[ok], kind="stable")[:top_n]])
        keep = np.isfinite(grid["ret"][idx, p]) & grid["observed"][idx, p - L : p].any(axis=1)
        idx = idx[keep]
        if idx.size == 0:
            dropped.append({"announcement_date": d.date().isoformat(), "reason": "no member firm with data"})
            continue
        rows.append(pd.DataFrame({"meeting": d, "pos": p, "firm": idx, "permno": permnos[idx],
                                  "in_sp500": np.isin(permnos[idx], in_index)}))
    return pd.concat(rows, ignore_index=True), dropped


def _asof_link(rows: pd.DataFrame, link: pd.DataFrame) -> pd.Series:
    """gvkey valid on the meeting date; primary links (P) win over C."""
    j = rows[["meeting", "permno"]].reset_index().merge(link, left_on="permno", right_on="lpermno")
    ok = (j["linkdt"] <= j["meeting"]) & (j["linkenddt"].fillna(pd.Timestamp.max) >= j["meeting"])
    j = j[ok].sort_values(["index", "linkprim"], ascending=[True, False])  # 'P' > 'C'
    return j.drop_duplicates("index").set_index("index")["gvkey"].reindex(rows.index)


def _asof_fundamentals(rows: pd.DataFrame, fundq: pd.DataFrame, mktcap: np.ndarray) -> np.ndarray:
    """(N, F) fundamentals from the latest quarter reported strictly before the meeting."""
    out = np.full((len(rows), len(FUNDAMENTAL_NAMES)), np.nan)
    size = mktcap[rows["firm"].to_numpy(), rows["pos"].to_numpy() - 1]  # prior close, $ millions
    with np.errstate(invalid="ignore", divide="ignore"):
        out[:, 1] = np.where(size > 0, np.log(size), np.nan)

    linked = rows.dropna(subset=["gvkey"]).reset_index()
    fq = fundq.dropna(subset=["rdq"]).sort_values(["rdq", "datadate"])
    if linked.empty or fq.empty:
        return out
    merged = pd.merge_asof(
        linked.sort_values("meeting"), fq, left_on="meeting", right_on="rdq",
        by="gvkey", allow_exact_matches=False,
    ).set_index("index").reindex(rows.index)
    age = (merged["meeting"] - merged["rdq"]).dt.days
    fresh = (age <= STALE_DAYS).to_numpy()
    at = merged["atq"].where(merged["atq"] > 0)
    debt = merged[["dlttq", "dlcq"]].sum(axis=1, min_count=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cols = {
            0: debt / at,
            2: merged["niq"] / at,
            3: merged["cheq"] / at,
            4: merged["ceqq"] / pd.Series(np.where(size > 0, size, np.nan), index=rows.index),
            5: age / 365.25,
        }
    for k, values in cols.items():
        v = values.to_numpy(dtype=np.float64, na_value=np.nan)
        out[:, k] = np.where(fresh, v, np.nan)
    out[~np.isfinite(out)] = np.nan
    return out


def _asof_sector(rows: pd.DataFrame, gics: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """GICS sector valid on the meeting date, else the firm's earliest known sector.

    Compustat's GICS history starts 1999-06-30. Sectors almost never change, so earlier
    rows take the first recorded sector; ``source`` marks those as ``"backfill"`` (a
    labeled look-ahead), ``"asof"`` otherwise, and NaN when the firm has no GICS at all.
    """
    j = rows[["meeting", "gvkey"]].dropna().reset_index().merge(gics, on="gvkey")
    ok = (j["indfrom"] <= j["meeting"]) & (j["indthru"].fillna(pd.Timestamp.max) >= j["meeting"])
    asof = j[ok].sort_values(["index", "indfrom"]).drop_duplicates("index", keep="last")
    sector = asof.set_index("index")["gsector"].reindex(rows.index)
    source = pd.Series(np.where(sector.notna(), "asof", None), index=rows.index, dtype=object)
    first = gics.sort_values("indfrom").drop_duplicates("gvkey").set_index("gvkey")["gsector"]
    fill = sector.isna() & rows["gvkey"].isin(first.index)
    sector[fill] = rows.loc[fill, "gvkey"].map(first)
    source[fill] = "backfill"
    return sector, source


def build_panel(inputs: Inputs, top_n: int | None = None) -> tuple[Q3Panel, pd.DataFrame, dict]:
    """Pure build from loaded tables: (panel, row metadata, build notes). See ``_rows`` for ``top_n``."""
    grid = _grid(inputs.daily, inputs.market)
    meetings = inputs.meetings.dropna(subset=["s"])
    meetings = meetings[meetings["announcement_date"] <= grid["calendar"][-1]]
    rows, dropped = _rows(grid, inputs.members, meetings, top_n)

    # Context per meeting: Q1 controls + annualized 21-day market volatility to the prior close.
    ctx = meetings.set_index("announcement_date")
    pos = rows.groupby("meeting")["pos"].first()
    mkt = grid["mkt"]
    ctx = ctx.loc[pos.index].assign(
        mkt_rvol21=[float(np.nanstd(mkt[p - 21 : p]) * np.sqrt(252.0)) for p in pos.to_numpy()]
    )
    bad = ctx.index[~np.isfinite(ctx[list(CONTEXT_NAMES)].to_numpy(dtype=np.float64)).all(axis=1)]
    dropped += [{"announcement_date": d.date().isoformat(), "reason": "missing context"} for d in bad]
    rows = rows[~rows["meeting"].isin(bad)].reset_index(drop=True)

    rows["gvkey"] = _asof_link(rows, inputs.link)
    fundamentals = _asof_fundamentals(rows, inputs.fundq, grid["mktcap"])
    rows["gsector"], rows["gsector_source"] = _asof_sector(rows, inputs.gics)

    firm, p = rows["firm"].to_numpy(), rows["pos"].to_numpy()
    history = np.empty((len(rows), L, len(CHANNELS)), dtype=np.float32)
    history_mask = np.empty((len(rows), L), dtype=bool)
    for k in range(len(rows)):
        history[k] = grid["channels"][firm[k], p[k] - L : p[k]]
        history_mask[k] = grid["observed"][firm[k], p[k] - L : p[k]]

    meeting_ctx = ctx.loc[rows["meeting"], list(CONTEXT_NAMES)].to_numpy(dtype=np.float32)
    fundamentals = fundamentals.astype(np.float32)
    fundamentals[~np.isfinite(fundamentals)] = np.nan
    panel = Q3Panel(
        meeting=rows["meeting"].to_numpy(dtype="datetime64[ns]"),
        firm_id=rows["permno"].to_numpy(dtype=np.int64),
        history=history,
        history_mask=history_mask,
        fundamentals=fundamentals,
        fundamentals_mask=np.isfinite(fundamentals),
        context=meeting_ctx,
        shock=ctx.loc[rows["meeting"], "s"].to_numpy(dtype=np.float32),
        target=grid["ret"][firm, p].astype(np.float32),
    )
    panel.validate()
    meta_rows = rows[["meeting", "permno", "gvkey", "gsector", "gsector_source", "in_sp500"]].assign(source="crsp")
    notes = {
        "top_n": top_n,
        "n_rows": len(rows),
        "in_sp500_share": round(float(rows["in_sp500"].mean()), 4),
        "n_meetings": int(rows["meeting"].nunique()),
        "n_firms": int(rows["permno"].nunique()),
        "first_meeting": rows["meeting"].min().date().isoformat(),
        "last_meeting": rows["meeting"].max().date().isoformat(),
        "dropped_meetings": dropped,
        "gsector_source_share": rows["gsector_source"].value_counts(normalize=True, dropna=False)
        .round(4).rename(lambda k: k if isinstance(k, str) else "none").to_dict(),
        "fundamentals_observed_share": dict(
            zip(FUNDAMENTAL_NAMES, np.isfinite(fundamentals).mean(axis=0).round(4).tolist())
        ),
        "fundamental_names": list(FUNDAMENTAL_NAMES),
        "context_names": list(CONTEXT_NAMES),
    }
    return panel, meta_rows, notes


# ---------------------------------------------------------------- 2026 holdout (yfinance)

HOLDOUT_CACHE = PROCESSED / "q3_holdout"
HOLDOUT_INPUTS = INPUTS + ("yf_firm_daily", "yf_spy_daily")


def _yf_returns() -> tuple[pd.DataFrame, pd.DataFrame]:
    """yfinance firm bars with daily return (adjusted close) and SPY total return, decimals."""
    bars = load("yf_firm_daily").sort_values(["permno", "date"])
    bars["ret"] = bars.groupby("permno")["adj_close"].pct_change()
    spy = load("yf_spy_daily").sort_values("date")
    spy["ret"] = spy["adj_close"].pct_change()
    return bars, spy


def compare_overlap() -> dict:
    """How closely yfinance reproduces CRSP on the overlap year (2025): returns, prices,
    volume, and SPY vs the Fama-French market. Run before trusting the 2026 extension."""
    bars, spy = _yf_returns()
    crsp = load("crsp_daily", start=bars["date"].min(), columns=["permno", "dlyret", "dlyprc", "dlyvol"])
    j = bars.merge(crsp, left_on=["permno", "date"], right_on=["permno", "dlycaldt"]).dropna(
        subset=["ret", "dlyret"]
    )
    diff = 100.0 * (j["ret"] - j["dlyret"])
    per_firm = j.groupby("permno").apply(lambda g: g["ret"].corr(g["dlyret"]), include_groups=False)
    vol_ratio = np.log(j["volume"].where(j["volume"] > 0) / j["dlyvol"].where(j["dlyvol"] > 0))
    ff = load("ff_factors_daily", start=spy["date"].min())
    m = spy.merge(ff, on="date").dropna(subset=["ret"])
    mdiff = 100.0 * (m["ret"] - (m["mktrf"] + m["rf"]))
    return {
        "firm_days": int(len(j)),
        "firms": int(j["permno"].nunique()),
        "ret_corr": round(float(j["ret"].corr(j["dlyret"])), 5),
        "ret_abs_diff_pp_median": round(float(diff.abs().median()), 4),
        "ret_abs_diff_pp_p99": round(float(diff.abs().quantile(0.99)), 4),
        "share_days_diff_over_0.1pp": round(float((diff.abs() > 0.1).mean()), 4),
        "share_days_diff_over_1pp": round(float((diff.abs() > 1.0).mean()), 4),
        "firm_corr_min": round(float(per_firm.min()), 4),
        "firms_corr_below_0.99": int((per_firm < 0.99).sum()),
        "close_vs_crsp_price_abs_rel_median": round(float((j["close"] / j["dlyprc"].abs() - 1).abs().median()), 5),
        "log_volume_ratio_median": round(float(vol_ratio.median()), 4),
        "log_volume_ratio_iqr": round(float(vol_ratio.quantile(0.75) - vol_ratio.quantile(0.25)), 4),
        "spy_vs_ff_market_corr": round(float(m["ret"].corr(m["mktrf"] + m["rf"])), 5),
        "spy_vs_ff_abs_diff_pp_median": round(float(mdiff.abs().median()), 4),
    }


def holdout_inputs(base: Inputs | None = None) -> Inputs:
    """CRSP inputs extended past the CRSP vintage with yfinance bars.

    After the last CRSP day: firm returns from yfinance adjusted closes, price from the
    close, shares held at the last CRSP value; the market is Fama-French while it lasts,
    then SPY. Firms in the yfinance universe stay S&P members; other end-of-vintage members
    drop out (removed in 2026, no dated change list).
    """
    base = base or read_inputs()
    crsp_end = pd.Timestamp(base.daily["dlycaldt"].max())
    bars, spy = _yf_returns()
    shares = base.daily.sort_values("dlycaldt").groupby("permno")["shrout"].last()
    ext = bars[bars["date"] > crsp_end].dropna(subset=["ret"])
    ext = pd.DataFrame({
        "permno": ext["permno"].astype("int64"), "dlycaldt": ext["date"], "dlyret": ext["ret"],
        "dlyvol": ext["volume"], "dlyprc": ext["close"], "shrout": ext["permno"].map(shares),
    })
    daily = pd.concat([base.daily, ext], ignore_index=True)

    ff = base.market[pd.to_datetime(base.market["date"]) <= crsp_end]
    ff_after = base.market[pd.to_datetime(base.market["date"]) > crsp_end]
    spy_after = spy[(spy["date"] > crsp_end) & ~spy["date"].isin(pd.to_datetime(ff_after["date"]))]
    market = pd.concat(
        [ff, ff_after, pd.DataFrame({"date": spy_after["date"], "mktrf": spy_after["ret"], "rf": 0.0})],
        ignore_index=True,
    ).dropna(subset=["mktrf"])

    members = base.members.copy()
    stays = members["permno"].isin(bars["permno"].unique()) & (members["mbrenddt"] >= crsp_end)
    members.loc[stays, "mbrenddt"] = pd.NaT
    return Inputs(daily, members, market, base.link, base.fundq, base.gics, base.meetings)


def load_holdout_panel(rebuild: bool = False) -> tuple[Q3Panel, pd.DataFrame, dict]:
    """Meetings after the CRSP vintage (2026), from CRSP history spliced with yfinance.

    A bonus holdout, scored separately from the CRSP panel: never pooled into training or
    the headline evaluation. History windows mix CRSP (2025) and yfinance (2026) days.
    """
    meta_file = HOLDOUT_CACHE / "meta.json"
    prints = _fingerprints(HOLDOUT_INPUTS)
    if prints is None:
        if not meta_file.exists() or rebuild:
            raise FileNotFoundError(f"no raw inputs to build from and no cached panel in {HOLDOUT_CACHE}")
        return _load_cache(HOLDOUT_CACHE)
    if not rebuild and meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if meta.get("inputs") == prints and meta.get("build_version") == BUILD_VERSION:
            return _load_cache(HOLDOUT_CACHE)
    base = read_inputs()
    crsp_end = pd.Timestamp(base.daily["dlycaldt"].max())
    full, rows, notes = build_panel(holdout_inputs(base))
    keep = (rows["meeting"] > crsp_end).to_numpy()
    panel, rows = full.subset(keep), rows[keep].reset_index(drop=True).assign(source="crsp+yfinance")
    panel.validate()
    HOLDOUT_CACHE.mkdir(parents=True, exist_ok=True)
    for name in _ARRAYS:
        np.save(HOLDOUT_CACHE / f"{name}.npy", getattr(panel, name))
    rows.to_parquet(HOLDOUT_CACHE / "rows.parquet", index=False)
    meta = {
        "built": date.today().isoformat(), "build_version": BUILD_VERSION, "inputs": prints,
        "crsp_end": crsp_end.date().isoformat(),
        "n_rows": int(len(rows)), "n_meetings": int(rows["meeting"].nunique()),
        "meetings": sorted(d.date().isoformat() for d in rows["meeting"].unique()),
        "n_firms": int(rows["permno"].nunique()), "overlap_check": compare_overlap(),
    }
    meta_file.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return panel, rows, meta


def _fingerprints(names: tuple[str, ...]) -> dict[str, str] | None:
    """Input fingerprints, or None when the raw inputs are absent (e.g. on a remote GPU box
    that only received the cached arrays)."""
    if not all(get(name).exists for name in names):
        return None
    return {name: fingerprint(name) for name in names}


def _load_cache(folder) -> tuple[Q3Panel, pd.DataFrame, dict]:
    panel = Q3Panel(**{name: np.load(folder / f"{name}.npy") for name in _ARRAYS})
    panel.validate()
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    return panel, pd.read_parquet(folder / "rows.parquet"), meta


def load_real_panel(rebuild: bool = False, universe: str = "sp500") -> tuple[Q3Panel, pd.DataFrame, dict]:
    """The cached real panel, rebuilt when missing, forced, or when any input file changed.

    ``universe`` is a key of ``UNIVERSES``; each has its own raw tables and cache folder.
    Without the raw inputs (a remote GPU box given only the cache folder) the cache is used
    as is; its ``meta["inputs"]`` still records the data it was built from.
    """
    u = UNIVERSES[universe]
    cache = u.cache or CACHE
    meta_file = cache / "meta.json"
    prints = _fingerprints(u.inputs)
    if prints is None:
        if not meta_file.exists() or rebuild:
            raise FileNotFoundError(f"no raw inputs to build from and no cached panel in {cache}")
        return _load_cache(cache)
    if not rebuild and meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if meta.get("inputs") == prints and meta.get("build_version") == BUILD_VERSION:
            return _load_cache(cache)
    panel, rows, notes = build_panel(read_inputs(universe), top_n=u.top_n)
    cache.mkdir(parents=True, exist_ok=True)
    for name in _ARRAYS:
        np.save(cache / f"{name}.npy", getattr(panel, name))
    rows.to_parquet(cache / "rows.parquet", index=False)
    meta = {"built": date.today().isoformat(), "build_version": BUILD_VERSION, "inputs": prints, **notes}
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return panel, rows, meta
