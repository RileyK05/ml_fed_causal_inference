"""Interactive filter components for fedviz.

These components provide consistent, reusable filter widgets that
all pages can use. They return filtered data or filter parameters.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


def date_range_filter(
    df: pd.DataFrame,
    date_col: str = "date",
    label: str = "Date Range",
    default_years: int = 5,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Date range slider filter."""
    dates = pd.to_datetime(df[date_col])
    min_date = dates.min()
    max_date = dates.max()
    default_start = max(min_date, max_date - pd.DateOffset(years=default_years))

    start, end = st.date_input(
        label,
        value=(default_start, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    return pd.Timestamp(start), pd.Timestamp(end)


def multi_select_filter(
    df: pd.DataFrame,
    column: str,
    label: str = "",
    default: list[str] | None = None,
    max_selections: int | None = None,
) -> list[str]:
    """Multi-select filter for categorical columns."""
    options = sorted(df[column].dropna().unique().tolist())
    if not options:
        return []
    selected = st.multiselect(
        label or column,
        options=options,
        default=default or options[:10],
        max_selections=max_selections,
    )
    return selected


def numeric_range_filter(
    df: pd.DataFrame,
    column: str,
    label: str = "",
    step: float | None = None,
) -> tuple[float, float]:
    """Range slider filter for numeric columns."""
    values = df[column].dropna()
    if len(values) == 0:
        return (0.0, 0.0)
    min_val = float(values.min())
    max_val = float(values.max())
    if min_val == max_val:
        return (min_val, max_val)
    if step is None:
        step = (max_val - min_val) / 100
    selected = st.slider(
        label or column,
        min_value=min_val,
        max_value=max_val,
        value=(min_val, max_val),
        step=step,
    )
    return selected


def categorical_filter(
    df: pd.DataFrame,
    column: str,
    label: str = "",
    default: str | None = None,
) -> str | None:
    """Single-select filter for categorical columns."""
    options = sorted(df[column].dropna().unique().tolist())
    if not options:
        return None
    selected = st.selectbox(label or column, options=options, index=0)
    return selected


def meeting_selector(
    meetings_df: pd.DataFrame,
    date_col: str = "announcement_date",
    label: str = "Select Meetings",
    max_meetings: int = 20,
) -> list[pd.Timestamp]:
    """Multi-select filter for FOMC meetings."""
    dates = sorted(meetings_df[date_col].dropna().unique())
    if not dates:
        return []
    selected = st.multiselect(
        label,
        options=dates,
        default=dates[-max_meetings:],
        max_selections=max_meetings,
        format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"),
    )
    return [pd.Timestamp(d) for d in selected]


def sector_selector(
    label: str = "Select Sectors",
    default: list[str] | None = None,
) -> list[str]:
    """Multi-select filter for ETF sectors."""
    from fedviz.themes import SECTOR_COLORS
    options = list(SECTOR_COLORS.keys())
    selected = st.multiselect(
        label,
        options=options,
        default=default or options,
    )
    return selected


def run_selector(
    runs_df: pd.DataFrame,
    label: str = "Select Run",
    project_col: str = "project",
    question_col: str = "question",
    name_col: str = "name",
) -> str | None:
    """Select a saved run from the results browser."""
    if runs_df.empty:
        return None
    options = []
    for _, row in runs_df.iterrows():
        label_text = f"{row[project_col]} / {row[question_col]} / {row[name_col]}"
        options.append((label_text, row["run_dir"]))
    labels = [o[0] for o in options]
    selected_label = st.selectbox(label, options=labels)
    for label_text, run_dir in options:
        if label_text == selected_label:
            return run_dir
    return None


def apply_filters(
    df: pd.DataFrame,
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    date_col: str = "date",
    categories: dict[str, list[str]] | None = None,
    numeric_ranges: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Apply multiple filters to a DataFrame."""
    mask = pd.Series(True, index=df.index)

    if date_range:
        dates = pd.to_datetime(df[date_col])
        mask &= (dates >= date_range[0]) & (dates <= date_range[1])

    if categories:
        for col, values in categories.items():
            if values:
                mask &= df[col].isin(values)

    if numeric_ranges:
        for col, (low, high) in numeric_ranges.items():
            mask &= (df[col] >= low) & (df[col] <= high)

    return df[mask].copy()
