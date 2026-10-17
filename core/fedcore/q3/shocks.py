"""Two-shock splits of the Q3 surprise ``s = 10 * STMT`` (10bp units), one value per meeting.

``policy_info``: a Jarocinski-Karadi style sign split. A meeting is an *information* meeting
when the S&P 500 moved the same way as rates in the USMPD announcement window
(``s * SP500 > 0``), otherwise a *policy* meeting. Columns are ``[s_policy, s_info]``; exactly
one is nonzero and they sum to ``s``.

``target_slope``: ``[s, slope]`` with ``slope = 10 * (UST10Y - UST2Y)``, the change in the
10y-2y spread in the same window (10bp units).

Meetings with no S&P 500 window move (all 14 are in 1994-95, before USMPD's S&P series starts)
are counted as policy meetings, the majority type. With the walk-forward's 60-meeting minimum
training window they only ever sit in training folds. ``UNKNOWN_AS_POLICY`` records the choice.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from fedcore.q3.contracts import Q3Panel

SPLITS = ("policy_info", "target_slope")
UNKNOWN_AS_POLICY = True


@lru_cache(maxsize=1)
def _usmpd_table() -> pd.DataFrame:
    from fedcore.data import load

    st = load("usmpd_statements").rename(columns={"Date": "meeting"})
    st["meeting"] = pd.to_datetime(st["meeting"])
    st = st.loc[st["Unscheduled"] == 0].drop_duplicates("meeting").set_index("meeting").sort_index()
    return pd.DataFrame({"sp500": st["SP500"], "slope": 10.0 * (st["UST10Y"] - st["UST2Y"])})


def meeting_table() -> pd.DataFrame:
    """Scheduled meetings: ``sp500`` (window S&P 500 move, NaN before 1996) and ``slope``."""
    return _usmpd_table().copy()


def info_flags(panel: Q3Panel, table: pd.DataFrame | None = None) -> np.ndarray:
    """Per row: 1.0 information meeting, 0.0 policy meeting, NaN no S&P window move."""
    table = meeting_table() if table is None else table
    sp = table["sp500"].reindex(pd.DatetimeIndex(panel.meeting)).to_numpy(dtype=np.float64)
    s = np.asarray(panel.shock, dtype=np.float64)
    return np.where(np.isfinite(sp), (s * sp > 0.0).astype(np.float64), np.nan)


def split_shock(panel: Q3Panel, kind: str, table: pd.DataFrame | None = None) -> np.ndarray:
    """(n, 2) shock matrix for ``kind`` in ``SPLITS``."""
    table = meeting_table() if table is None else table
    s = np.asarray(panel.shock, dtype=np.float64)
    if kind == "policy_info":
        flags = info_flags(panel, table)
        if not UNKNOWN_AS_POLICY and np.isnan(flags).any():
            raise ValueError("meetings without an S&P 500 window move")
        info = flags == 1.0
        return np.column_stack((np.where(info, 0.0, s), np.where(info, s, 0.0)))
    if kind == "target_slope":
        meetings = pd.DatetimeIndex(panel.meeting)
        slope = table["slope"].reindex(meetings).to_numpy(dtype=np.float64)
        if not np.isfinite(slope).all():
            missing = sorted({m.date() for m in meetings[~np.isfinite(slope)]})
            raise ValueError(f"no USMPD slope change for {missing[:5]}")
        return np.column_stack((s, slope))
    raise ValueError(f"kind must be one of {SPLITS}, got {kind!r}")
