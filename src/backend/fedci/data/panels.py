"""Event-time retrieval: the meeting spine, and any daily dataset lined up around each
meeting in trading-day time. These are the shapes every question consumes."""
from __future__ import annotations

import warnings

import pandas as pd

from fedci.data.catalog import get
from fedci.data.loaders import load

TRADING_CALENDAR = "etf_returns"  # dataset whose date column defines trading days


def meetings(scheduled_only: bool = True) -> pd.DataFrame:
    """The meeting spine: one row per FOMC announcement date. scheduled_only keeps rows
    with use_for_surprise = True (question.md: unscheduled actions are a separate
    treatment and are never pooled)."""
    m = load("fomc_dates")
    if scheduled_only:
        m = m[m["use_for_surprise"]]
    return m.reset_index(drop=True)


def trading_days(calendar: str = TRADING_CALENDAR) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(load(calendar)[get(calendar).date_column])


def event_panel(
    name: str,
    pre: int = 0,
    post: int = 0,
    columns: list[str] | None = None,
    events: pd.Series | None = None,
    calendar: str = TRADING_CALENDAR,
    long: bool = True,
) -> pd.DataFrame:
    """Line up a daily dataset around each event date in trading-day time.

    rel_day = 0 is the announcement day; rel_day = -1 the prior trading day, +1 the next.
    The data is reindexed onto `calendar` without filling, so a missing value stays NaN
    and never borrows a neighbouring day.

    long=True  -> columns announcement_date, rel_day, date, series, value
    long=False -> columns announcement_date, rel_day, date, <one per series>
    """
    ds = get(name)
    if ds.grain != "daily":
        raise ValueError(f"event_panel needs a daily dataset; {name!r} is {ds.grain}")
    df = load(name, columns=columns).set_index(ds.date_column)
    cal = trading_days(calendar)
    df = df.reindex(cal)

    ev = pd.to_datetime(meetings()["announcement_date"] if events is None else events)
    pos = cal.get_indexer(ev)
    off = [str(d.date()) for d, p in zip(ev, pos) if p < 0]
    if off:
        warnings.warn(f"{len(off)} event dates are not trading days in {calendar!r} and were dropped: {off}")

    rows = [
        (d, k, cal[p + k])
        for d, p in zip(ev, pos) if p >= 0
        for k in range(-pre, post + 1) if 0 <= p + k < len(cal)
    ]
    idx = pd.DataFrame(rows, columns=["announcement_date", "rel_day", "date"])
    wide = idx.join(df, on="date")
    if not long:
        return wide
    return wide.melt(id_vars=["announcement_date", "rel_day", "date"], var_name="series", value_name="value")
