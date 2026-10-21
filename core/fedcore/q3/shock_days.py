"""Macro shock days: the Q3 panel built on non-FOMC days with a big 2-year Treasury move.

Pretraining data for v6 (docs/encoder/q3-v6-plan.md section 3). Every Fed arm starves on about
40 big meetings. Track A found that distress amplifies any big market move, so the encoder first
learns beyond-beta responses on these days and is then fine-tuned and scored on Fed days only.

- Events: trading days with |change in the GSW 2-year zero yield| >= ``THRESHOLD_BP``, excluding
  every FOMC statement day (scheduled or not) and the trading day after.
- Shock: the 2-year change in 10bp units, the Fed panel's unit. It mixes rate and growth news, so
  it is a noisier proxy than the Fed surprise.
- Rows, history, target, fundamentals: exactly as the Fed panel (``fedcore.q3.real.build_panel``),
  with the event day in place of the meeting day.
- Context: the Fed panel's columns as of the event. The statement terms come from the latest
  FOMC statement strictly before the day; 1-year yield terms and market volatility to the prior
  close.

Cached in gitignored ``core/data/processed/q3_shock_panel/`` (licensed inputs).
"""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd

from fedcore.config import PROCESSED
from fedcore.data import load
from fedcore.q3 import real
from fedcore.q3.contracts import Q3Panel

THRESHOLD_BP = 5.0
START, END = pd.Timestamp("1994-01-01"), pd.Timestamp("2021-12-31")
CACHE = PROCESSED / "q3_shock_panel"
INPUTS = real.INPUTS + ("gsw_yields",)
BUILD_VERSION = 1


def shock_days(y2: pd.Series, fomc: pd.DatetimeIndex, threshold_bp: float = THRESHOLD_BP,
               start=START, end=END) -> pd.Series:
    """2-year change (bp) on qualifying days. ``y2``: yield in percent indexed by trading date."""
    y2 = y2.dropna().sort_index()
    chg = (y2.diff() * 100.0).dropna()
    cal = y2.index
    fomc = pd.DatetimeIndex(fomc)
    nxt = cal.searchsorted(fomc, side="right")             # first trading day after each statement
    excluded = fomc.union(cal[nxt[nxt < len(cal)]])
    keep = (chg.index >= start) & (chg.index <= end) & ~chg.index.isin(excluded) & (chg.abs() >= threshold_bp)
    return chg[keep]


def event_table(threshold_bp: float = THRESHOLD_BP, start=START, end=END) -> pd.DataFrame:
    """``real.meeting_table`` columns for shock days: announcement_date, s, and the context terms."""
    gsw = load("gsw_yields").rename(columns={"Date": "date"})
    gsw["date"] = pd.to_datetime(gsw["date"])
    stmt = load("usmpd_statements")
    fomc = pd.DatetimeIndex(pd.to_datetime(stmt["Date"]))
    chg = shock_days(gsw.set_index("date")["SVENY02"], fomc, threshold_bp, start, end)
    ev = pd.DataFrame({"announcement_date": chg.index, "s": chg.to_numpy() / 10.0})

    mps = load("mps_surprises").rename(columns={"Date": "announcement_date"}).sort_values("announcement_date")
    hist = mps[["announcement_date", "STMT", "ME"]].copy()
    hist["st_prev"], hist["me_prev"] = hist["STMT"], hist["ME"]          # latest statement before the day
    hist["st_mean3"] = hist["STMT"].rolling(3).mean()
    ev = pd.merge_asof(ev, hist[["announcement_date", "st_prev", "me_prev", "st_mean3"]].rename(
        columns={"announcement_date": "stmt_date"}), left_on="announcement_date", right_on="stmt_date",
        allow_exact_matches=False).drop(columns="stmt_date")

    y1 = load("treasury_1y").rename(columns={"Date": "date", "SVENY01": "y1_level"}).sort_values("date")
    y1["y1_chg_21d"] = y1["y1_level"] - y1["y1_level"].shift(21)
    y1["y1_chg_63d"] = y1["y1_level"] - y1["y1_level"].shift(63)
    ev = pd.merge_asof(ev, y1[["date", "y1_level", "y1_chg_21d", "y1_chg_63d"]], left_on="announcement_date",
                       right_on="date", allow_exact_matches=False).drop(columns="date")
    return ev.reset_index(drop=True)


def load_shock_panel(rebuild: bool = False, threshold_bp: float = THRESHOLD_BP) -> tuple[Q3Panel, pd.DataFrame, dict]:
    """The cached shock-day panel, rebuilt when missing, forced, or when an input changed."""
    meta_file = CACHE / "meta.json"
    prints = real._fingerprints(INPUTS)
    key = dict(build_version=BUILD_VERSION, panel_build_version=real.BUILD_VERSION, threshold_bp=threshold_bp)
    if prints is None:
        if not meta_file.exists() or rebuild:
            raise FileNotFoundError(f"no raw inputs to build from and no cached panel in {CACHE}")
        return real._load_cache(CACHE)
    if not rebuild and meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if meta.get("inputs") == prints and all(meta.get(k) == v for k, v in key.items()):
            return real._load_cache(CACHE)
    base = real.read_inputs()
    inputs = real.Inputs(base.daily, base.members, base.market, base.link, base.fundq, base.gics,
                         event_table(threshold_bp))
    panel, rows, notes = real.build_panel(inputs)
    CACHE.mkdir(parents=True, exist_ok=True)
    for name in real._ARRAYS:
        np.save(CACHE / f"{name}.npy", getattr(panel, name))
    rows.to_parquet(CACHE / "rows.parquet", index=False)
    meta = {"built": date.today().isoformat(), "inputs": prints, **key,
            **{k: v for k, v in notes.items() if k != "dropped_meetings"},
            "dropped_events": len(notes["dropped_meetings"])}
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return panel, rows, meta
