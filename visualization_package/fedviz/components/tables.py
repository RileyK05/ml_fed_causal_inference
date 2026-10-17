"""Interactive data table components for fedviz."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


def data_table(
    df: pd.DataFrame,
    height: int = 400,
    max_rows: int = 10_000,
    column_config: dict[str, Any] | None = None,
    use_container_width: bool = True,
) -> None:
    """Render an interactive data table."""
    display_df = df.head(max_rows)
    st.dataframe(
        display_df,
        height=height,
        width="stretch" if use_container_width else "content",
        column_config=column_config,
        hide_index=True,
    )
    if len(df) > max_rows:
        st.caption(f"Showing first {max_rows:,} of {len(df):,} rows")


def metrics_table(
    metrics: dict[str, Any],
    title: str = "Metrics",
    comparison: dict[str, Any] | None = None,
) -> None:
    """Render a metrics comparison table."""
    records = []
    for key, value in metrics.items():
        record = {"Metric": key, "Value": value}
        if comparison and key in comparison:
            record["Comparison"] = comparison[key]
            if isinstance(value, (int, float)) and isinstance(comparison[key], (int, float)):
                record["Delta"] = value - comparison[key]
        records.append(record)

    df = pd.DataFrame(records)
    st.markdown(f"**{title}**")
    st.dataframe(df, width="stretch", hide_index=True)


def summary_table(
    df: pd.DataFrame,
    groupby: str | None = None,
    agg_funcs: list[str] | None = None,
    title: str = "Summary",
) -> pd.DataFrame:
    """Render a summary statistics table."""
    if agg_funcs is None:
        agg_funcs = ["mean", "std", "min", "max", "count"]

    if groupby:
        summary = df.groupby(groupby).agg(agg_funcs)
    else:
        summary = df.describe()

    st.markdown(f"**{title}**")
    st.dataframe(summary, width="stretch")
    return summary


def styled_table(
    df: pd.DataFrame,
    color_map: dict[str, str] | None = None,
    number_fmt: str = ".4f",
) -> None:
    """Render a styled dataframe with conditional formatting."""
    def _color(val: Any, col: str) -> str:
        if color_map and col in color_map:
            return f"color: {color_map[col]}"
        return ""

    styled = df.style.map(lambda v: "", subset=None)
    st.dataframe(styled, width="stretch", hide_index=True)


def pivot_table(
    df: pd.DataFrame,
    index: str,
    columns: str,
    values: str,
    aggfunc: str = "mean",
    title: str = "",
) -> pd.DataFrame:
    """Render a pivot table."""
    pivot = df.pivot_table(
        index=index, columns=columns, values=values, aggfunc=aggfunc,
    )
    if title:
        st.markdown(f"**{title}**")
    st.dataframe(pivot, width="stretch")
    return pivot
