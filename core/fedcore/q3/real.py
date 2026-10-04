"""Real Q3 firm-meeting panel: S&P 500 members at each scheduled FOMC statement, 1994-2025.

Built from the registered WRDS datasets (core/data/PROVENANCE.md section 7) with the
synthetic panel's channel definitions, so every arm runs on it unchanged.

- Rows: every firm in the S&P 500 on the meeting date (point-in-time membership, so firms
  that later left or delisted are included) with a meeting-day return and at least one
  observed history day.
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

import numpy as np
import pandas as pd

from fedcore.config import PROCESSED
from fedcore.data import load
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
# The synthetic panel falls back to the firm's true beta; real data has none, so use the market's.
BETA_FALLBACK = 1.0
CACHE = PROCESSED / "q3_panel"
INPUTS = (
    "crsp_daily", "sp500_members", "ff_factors_daily", "ccm_link", "comp_fundq", "comp_gics",
    "usmpd_statements", "mps_surprises", "treasury_1y",
)
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


def read_inputs() -> Inputs:
    def dates(df: pd.DataFrame, *cols: str) -> pd.DataFrame:
        for col in cols:
            df[col] = pd.to_datetime(df[col])
        return df

    def gvkey(df: pd.DataFrame) -> pd.DataFrame:
        df["gvkey"] = df["gvkey"].astype(str)  # one plain type so the as-of joins match
        return df

    link = gvkey(dates(load("ccm_link"), "linkenddt"))
    link["lpermno"] = link["lpermno"].astype("int64")
    return Inputs(
        daily=load("crsp_daily", columns=["permno", "dlyret", "dlyvol", "dlyprc", "shrout"]),
        members=dates(load("sp500_members"), "mbrenddt"),
        market=load("ff_factors_daily", columns=["mktrf", "rf"]),
        link=link,
        fundq=gvkey(dates(load("comp_fundq"), "rdq")),
        gics=gvkey(dates(load("comp_gics"), "indthru")),
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


def _rows(grid: dict, members: pd.DataFrame, meetings: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """One row per (meeting, member firm) that has a target and some history."""
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
        idx = np.searchsorted(permnos, in_index)
        idx = idx[(idx < permnos.size) & (permnos[np.minimum(idx, permnos.size - 1)] == in_index)]
        idx = np.unique(idx)
        keep = np.isfinite(grid["ret"][idx, p]) & grid["observed"][idx, p - L : p].any(axis=1)
        idx = idx[keep]
        if idx.size == 0:
            dropped.append({"announcement_date": d.date().isoformat(), "reason": "no member firm with data"})
            continue
        rows.append(pd.DataFrame({"meeting": d, "pos": p, "firm": idx, "permno": permnos[idx]}))
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


def _asof_sector(rows: pd.DataFrame, gics: pd.DataFrame) -> pd.Series:
    j = rows[["meeting", "gvkey"]].dropna().reset_index().merge(gics, on="gvkey")
    ok = (j["indfrom"] <= j["meeting"]) & (j["indthru"].fillna(pd.Timestamp.max) >= j["meeting"])
    j = j[ok].sort_values(["index", "indfrom"]).drop_duplicates("index", keep="last")
    return j.set_index("index")["gsector"].reindex(rows.index)


def build_panel(inputs: Inputs) -> tuple[Q3Panel, pd.DataFrame, dict]:
    """Pure build from loaded tables: (panel, row metadata, build notes)."""
    grid = _grid(inputs.daily, inputs.market)
    meetings = inputs.meetings.dropna(subset=["s"])
    meetings = meetings[meetings["announcement_date"] <= grid["calendar"][-1]]
    rows, dropped = _rows(grid, inputs.members, meetings)

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
    rows["gsector"] = _asof_sector(rows, inputs.gics)

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
    meta_rows = rows[["meeting", "permno", "gvkey", "gsector"]].assign(source="crsp")
    notes = {
        "n_rows": len(rows),
        "n_meetings": int(rows["meeting"].nunique()),
        "n_firms": int(rows["permno"].nunique()),
        "first_meeting": rows["meeting"].min().date().isoformat(),
        "last_meeting": rows["meeting"].max().date().isoformat(),
        "dropped_meetings": dropped,
        "fundamentals_observed_share": dict(
            zip(FUNDAMENTAL_NAMES, np.isfinite(fundamentals).mean(axis=0).round(4).tolist())
        ),
        "fundamental_names": list(FUNDAMENTAL_NAMES),
        "context_names": list(CONTEXT_NAMES),
    }
    return panel, meta_rows, notes


def _fingerprints() -> dict[str, str]:
    return {name: fingerprint(name) for name in INPUTS}


def load_real_panel(rebuild: bool = False) -> tuple[Q3Panel, pd.DataFrame, dict]:
    """The cached real panel, rebuilt when missing, forced, or when any input file changed."""
    meta_file = CACHE / "meta.json"
    prints = _fingerprints()
    if not rebuild and meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if meta.get("inputs") == prints:
            arrays = {name: np.load(CACHE / f"{name}.npy") for name in _ARRAYS}
            panel = Q3Panel(**arrays)
            panel.validate()
            return panel, pd.read_parquet(CACHE / "rows.parquet"), meta
    panel, rows, notes = build_panel(read_inputs())
    CACHE.mkdir(parents=True, exist_ok=True)
    for name in _ARRAYS:
        np.save(CACHE / f"{name}.npy", getattr(panel, name))
    rows.to_parquet(CACHE / "rows.parquet", index=False)
    meta = {"built": date.today().isoformat(), "inputs": prints, **notes}
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return panel, rows, meta
