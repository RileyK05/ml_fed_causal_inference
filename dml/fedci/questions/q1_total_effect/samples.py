"""Estimation samples for Q1 (docs/question.md, "Q1 - Aggregate shock response").

Both samples are one row per scheduled FOMC statement event. Units matter because the
sources disagree, so they are normalized here and nowhere else:

- y is an equity return in percent. USMPD's SP500 window return is already percent;
  event_study_table's ETF simple daily returns are decimal and are converted x100.
- s is the surprise in units of 10bp of the 1y-yield-equivalent surprise. USMPD/ACM
  surprises are percentage points (STMT = 0.15 means 15bp), so s = 10 * STMT. theta is
  therefore "percent return per 10bp surprise".

Nothing is filled or interpolated: rows with missing outcomes, treatments or controls
keep NaN and are dropped (and counted) by fedci.eval.partially_linear_dml.

Pre-event controls only (question.md: contemporaneous channel variables are
post-treatment and are never Q1 controls):

- usmpd_window_sample: 1y Treasury state and lagged surprise history -- the only
  pre-event series with coverage back to 1996 (FRED and Cleveland Fed controls start in
  2023; they belong to the daily-ETF sample and to Q3 state). The 1y yield is read from
  the last close strictly before the announcement, so it is never contaminated by the
  same-day move.
- etf_day0_sample: the lagged FRED controls and Cleveland Fed inflation expectations
  that event_study_table already carries. The credit-OAS series are excluded there:
  they start 2023-09-22 and would drop 6 of 30 meetings for a control set already too
  rich for the sample (the compact set below is predeclared instead). The term spread
  T10Y2Y is excluded too: it equals DGS10 - DGS2 exactly and would make the controlled
  OLS rank-deficient.

Unscheduled actions are a separate treatment and are never pooled (question.md):
usmpd_window_sample filters on USMPD's Unscheduled flag; event_study_table is already
restricted to use_for_surprise meetings.
"""
from __future__ import annotations

import pandas as pd

from fedcore.data import load

SURPRISES = ("STMT", "MP1", "ME")
OUTCOMES = ("usmpd_sp500", "etf_day0")

_CONTROL_X = ("y1_level", "y1_chg_21d", "y1_chg_63d", "st_prev", "me_prev", "st_mean3")
_ETF_X = ("lagged_VIXCLS", "lagged_DGS2", "lagged_DGS10",
          "val_1yr", "val_5yr", "val_10yr")

FEATURE_INFO = {
    "usmpd_sp500": {
        "y1_level": "1-year Treasury yield at the last close strictly before the announcement",
        "y1_chg_21d": "1-year yield change over the ~1 month of trading days before the announcement",
        "y1_chg_63d": "1-year yield change over the ~1 quarter of trading days before the announcement",
        "st_prev": "STMT surprise at the previous statement event",
        "me_prev": "ME surprise at the previous statement event",
        "st_mean3": "mean STMT surprise over the three statement events before this one",
    },
    "etf_day0": {
        "lagged_VIXCLS": "VIX level on the prior FRED calendar day (pre-event)",
        "lagged_DGS2": "2-year Treasury yield on the prior FRED calendar day (pre-event)",
        "lagged_DGS10": "10-year Treasury yield on the prior FRED calendar day (pre-event)",
        "val_1yr": "1-year inflation expectation published as of the meeting (Cleveland Fed, CPI-date corrected)",
        "val_5yr": "5-year inflation expectation published as of the meeting (Cleveland Fed, CPI-date corrected)",
        "val_10yr": "10-year inflation expectation published as of the meeting (Cleveland Fed, CPI-date corrected)",
    },
}


def usmpd_window_sample(treatment: str = "STMT") -> pd.DataFrame:
    """Primary sample: announcement-window SP500 return on the statement surprise.

    Columns: announcement_date, y (percent), s (per 10bp), then _CONTROL_X.
    MP1 comes from the USMPD statements sheet; STMT/ME from mps_surprises.
    """
    if treatment not in SURPRISES:
        raise ValueError(f"treatment must be one of {SURPRISES}, got {treatment!r}")
    stmt = load("usmpd_statements").rename(columns={"Date": "announcement_date"})
    stmt = stmt[stmt["Unscheduled"] == 0].sort_values("announcement_date")
    mps = load("mps_surprises").rename(columns={"Date": "announcement_date"}).sort_values("announcement_date")

    y1 = load("treasury_1y").rename(columns={"Date": "date", "SVENY01": "y1_level"}).sort_values("date")
    y1["y1_chg_21d"] = y1["y1_level"] - y1["y1_level"].shift(21)   # ~1 month of trading days
    y1["y1_chg_63d"] = y1["y1_level"] - y1["y1_level"].shift(63)   # ~1 quarter
    state = y1[["date", "y1_level", "y1_chg_21d", "y1_chg_63d"]]

    hist = mps[["announcement_date", "STMT", "ME"]].copy()
    hist["st_prev"] = hist["STMT"].shift(1)                 # previous statement event, any type
    hist["me_prev"] = hist["ME"].shift(1)
    hist["st_mean3"] = hist["STMT"].shift(1).rolling(3).mean()

    df = stmt[["announcement_date", "SP500", "MP1"]].rename(columns={"SP500": "y"})
    df = df.merge(mps[["announcement_date", "STMT", "ME"]], on="announcement_date", how="left")
    df = df.merge(hist[["announcement_date", "st_prev", "me_prev", "st_mean3"]],
                  on="announcement_date", how="left")
    df = pd.merge_asof(df.sort_values("announcement_date"), state,
                       left_on="announcement_date", right_on="date", allow_exact_matches=False)
    df["s"] = 10.0 * df.pop(treatment)
    return df[["announcement_date", "y", "s", *_CONTROL_X]].reset_index(drop=True)


def etf_day0_sample(treatment: str = "STMT", outcome: str = "SPY") -> pd.DataFrame:
    """Robustness sample: day-0 ETF return on the statement surprise, 2023+ (30 meetings).

    Columns: announcement_date, y (percent), s (per 10bp), then _ETF_X from
    event_study_table (NaN kept).
    """
    if treatment not in SURPRISES:
        raise ValueError(f"treatment must be one of {SURPRISES}, got {treatment!r}")
    est = load("event_study_table")
    est = est[est["use_for_surprise"]].sort_values("announcement_date")
    y_col = f"ret_{outcome}"
    if y_col not in est.columns:
        raise ValueError(f"no column {y_col!r} in event_study_table")
    df = est[["announcement_date", y_col, treatment, *_ETF_X]].copy()
    df["y"] = 100.0 * df.pop(y_col)
    df["s"] = 10.0 * df.pop(treatment)
    return df[["announcement_date", "y", "s", *_ETF_X]].reset_index(drop=True)
