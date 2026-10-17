"""Utility helpers for fedviz."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable

import numpy as np
import pandas as pd


def format_pct(value: float, digits: int = 2) -> str:
    """Format a float as a percentage string."""
    if pd.isna(value):
        return "—"
    return f"{value * 100:.{digits}f}%"


def format_bp(value: float, digits: int = 1) -> str:
    """Format a float as basis points."""
    if pd.isna(value):
        return "—"
    return f"{value * 10000:.{digits}f} bp"


def format_number(value: float, digits: int = 2) -> str:
    """Format a float with commas and fixed digits."""
    if pd.isna(value):
        return "—"
    return f"{value:,.{digits}f}"


def format_date(date: pd.Timestamp | datetime | str) -> str:
    """Format a date for display."""
    if pd.isna(date):
        return "—"
    ts = pd.Timestamp(date)
    return ts.strftime("%Y-%m-%d")


def truncate_df(df: pd.DataFrame, max_rows: int = 10_000) -> pd.DataFrame:
    """Truncate a DataFrame to max_rows, adding a flag column."""
    if len(df) <= max_rows:
        return df
    return df.head(max_rows).copy()


def hash_dataframe(df: pd.DataFrame) -> str:
    """Compute a stable hash of a DataFrame's content."""
    content = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.md5(content).hexdigest()[:12]


def rolling_window(
    series: pd.Series,
    window: int,
    func: str = "mean",
) -> pd.Series:
    """Apply a rolling window function to a series."""
    roll = series.rolling(window=window, min_periods=max(1, window // 4))
    if func == "mean":
        return roll.mean()
    elif func == "std":
        return roll.std()
    elif func == "sum":
        return roll.sum()
    elif func == "min":
        return roll.min()
    elif func == "max":
        return roll.max()
    elif func == "median":
        return roll.median()
    else:
        raise ValueError(f"Unknown rolling function: {func}")


def cumulative_returns(returns: pd.Series) -> pd.Series:
    """Compute cumulative returns from a return series."""
    return (1 + returns.fillna(0)).cumprod() - 1


def drawdown(returns: pd.Series) -> pd.Series:
    """Compute drawdown series from returns."""
    cum = (1 + returns.fillna(0)).cumprod()
    running_max = cum.expanding().max()
    return cum / running_max - 1


def safe_divide(a: pd.Series, b: pd.Series, fill: float = np.nan) -> pd.Series:
    """Divide two series, handling division by zero."""
    result = a / b.replace(0, np.nan)
    return result.fillna(fill)


def date_range_str(start: pd.Timestamp, end: pd.Timestamp) -> str:
    """Format a date range as a string."""
    return f"{start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')}"


def memoize_with_ttl(ttl_seconds: int = 300):
    """Simple TTL memoization decorator."""
    def decorator(func: Callable) -> Callable:
        cache: dict[str, tuple[float, Any]] = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(sorted(kwargs.items()))
            now = datetime.now().timestamp()
            if key in cache:
                ts, result = cache[key]
                if now - ts < ttl_seconds:
                    return result
            result = func(*args, **kwargs)
            cache[key] = (now, result)
            return result

        return wrapper
    return decorator


def detect_frequency(dates: pd.DatetimeIndex) -> str:
    """Detect the frequency of a datetime index."""
    if len(dates) < 2:
        return "unknown"
    diffs = dates.to_series().diff().dropna()
    median_diff = diffs.median()
    if median_diff <= pd.Timedelta(days=1):
        return "daily"
    elif median_diff <= pd.Timedelta(days=7):
        return "weekly"
    elif median_diff <= pd.Timedelta(days=31):
        return "monthly"
    else:
        return "irregular"


def align_event_dates(
    dates: pd.DatetimeIndex,
    events: pd.DatetimeIndex,
    window: int = 5,
) -> pd.DataFrame:
    """Align event dates to a window of trading days around each event."""
    records = []
    for event in events:
        idx = dates.searchsorted(event)
        if idx < 0 or idx >= len(dates):
            continue
        start_idx = max(0, idx - window)
        end_idx = min(len(dates), idx + window + 1)
        for i in range(start_idx, end_idx):
            records.append({
                "event_date": event,
                "date": dates[i],
                "rel_day": i - idx,
            })
    return pd.DataFrame(records)
