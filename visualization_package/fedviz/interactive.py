"""Interactive layer for fedviz — cross-filtering, linked charts, drill-down.

This module provides the interactivity that makes the visualization suite
feel alive. It enables:
- Cross-filtering: click on a chart element to filter other charts
- Linked brushing: select a range on one chart to highlight on others
- Drill-down: click on a data point to see more detail
- Hover context: rich tooltips with related data
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


@dataclass
class FilterState:
    """Tracks the current filter state across all linked charts."""

    date_range: tuple[pd.Timestamp, pd.Timestamp] | None = None
    selected_categories: list[str] = field(default_factory=list)
    selected_points: list[int] = field(default_factory=list)
    highlight_indices: list[int] = field(default_factory=list)
    drill_down_id: str | None = None
    custom_filters: dict[str, Any] = field(default_factory=dict)

    def clear(self) -> None:
        self.date_range = None
        self.selected_categories = []
        self.selected_points = []
        self.highlight_indices = []
        self.drill_down_id = None
        self.custom_filters.clear()

    def to_dict(self) -> dict:
        return {
            "date_range": self.date_range,
            "selected_categories": self.selected_categories,
            "selected_points": self.selected_points,
            "highlight_indices": self.highlight_indices,
            "drill_down_id": self.drill_down_id,
            "custom_filters": self.custom_filters,
        }


def get_filter_state() -> FilterState:
    """Get or create the global filter state."""
    if "filter_state" not in st.session_state:
        st.session_state["filter_state"] = FilterState()
    return st.session_state["filter_state"]


def clear_filters() -> None:
    """Clear all active filters."""
    state = get_filter_state()
    state.clear()
    st.rerun()


def apply_filter(
    category: str,
    value: Any,
) -> None:
    """Apply a filter to the global state."""
    state = get_filter_state()
    state.custom_filters[category] = value
    st.rerun()


def linked_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
    size: str | None = None,
    text: str | None = None,
    title: str = "",
    height: int = 500,
    theme: str | None = None,
    selection_mode: str = "box",
    on_select: Callable | None = None,
    key: str = "",
) -> go.Figure:
    """Scatter plot with box/lasso selection linked to global filter state."""
    fig = go.Figure()

    if color:
        groups = df[color].unique()
        for i, grp in enumerate(groups):
            mask = df[color] == grp
            sub = df[mask]
            fig.add_trace(go.Scatter(
                x=sub[x], y=sub[y],
                mode="markers",
                name=str(grp),
                marker=dict(size=8, opacity=0.7, line=dict(width=0.5, color="rgba(255,255,255,0.45)")),
                text=sub[text] if text else None,
                customdata=sub.index.tolist(),
                selectedpoints=[],
                unselected=dict(marker=dict(opacity=0.2)),
            ))
    else:
        fig.add_trace(go.Scatter(
            x=df[x], y=df[y],
            mode="markers",
            marker=dict(size=8, opacity=0.7, color="#5b8def",
                        line=dict(width=0.5, color="rgba(255,255,255,0.45)")),
            text=df[text] if text else None,
            customdata=df.index.tolist(),
            selectedpoints=[],
            unselected=dict(marker=dict(opacity=0.2)),
        ))

    fig.update_layout(
        title=title,
        height=height,
        dragmode="select" if selection_mode == "box" else selection_mode,
        clickmode="event+select",
    )

    if theme:
        from fedviz.themes import get_theme
        fig = get_theme(theme).apply(fig)

    return fig


def linked_time_series(
    df: pd.DataFrame,
    x: str = "date",
    y: str | list[str] | None = None,
    color: str | None = None,
    title: str = "",
    height: int = 450,
    theme: str | None = None,
    brush_range: bool = True,
    key: str = "",
) -> go.Figure:
    """Time series with range brush selection linked to global filter state."""
    fig = go.Figure()

    if y is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        y = [c for c in numeric_cols if c != x and c != color]
    elif isinstance(y, str):
        y = [y]
    elif not isinstance(y, list):
        y = list(y)

    if color:
        groups = df[color].unique()
        for i, grp in enumerate(groups):
            mask = df[color] == grp
            sub = df[mask]
            for col in y:
                fig.add_trace(go.Scatter(
                    x=sub[x], y=sub[col],
                    mode="lines",
                    name=f"{col} ({grp})",
                    line=dict(width=2),
                ))
    else:
        for i, col in enumerate(y):
            fig.add_trace(go.Scatter(
                x=df[x], y=df[col],
                mode="lines",
                name=col,
                line=dict(width=2, color=f"hsl({i * 60}, 70%, 50%)"),
            ))

    fig.update_layout(
        title=title,
        height=height,
        hovermode="x unified",
    )

    if brush_range:
        fig.update_xaxes(rangeslider_visible=True, rangeslider_thickness=0.05)

    if theme:
        from fedviz.themes import get_theme
        fig = get_theme(theme).apply(fig)

    return fig


def drill_down_table(
    df: pd.DataFrame,
    group_cols: list[str],
    value_cols: list[str],
    agg_funcs: list[str] | None = None,
    title: str = "",
    height: int = 400,
) -> pd.DataFrame:
    """Interactive drill-down table with expandable groups."""
    if agg_funcs is None:
        agg_funcs = ["mean", "std", "count"]

    summary = df.groupby(group_cols)[value_cols].agg(agg_funcs)
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()

    st.markdown(f"**{title}**")
    st.dataframe(
        summary,
        height=height,
        width="stretch",
        hide_index=True,
    )
    return summary


def cross_filter_panel(
    df: pd.DataFrame,
    filter_cols: list[str],
    title: str = "Filters",
    expanded: bool = False,
) -> dict[str, Any]:
    """Render a cross-filter panel that updates the global filter state."""
    state = get_filter_state()
    if state is None:
        state = FilterState()
    active_filters = {}

    with st.expander(title, expanded=expanded, icon=":material/filter_alt:"):
        for col in filter_cols:
            if df[col].dtype in ["object", "category"]:
                options = sorted(df[col].dropna().unique().tolist())
                default = state.custom_filters.get(col, options[:5])
                selected = st.multiselect(col, options, default=default)
                active_filters[col] = selected
                state.custom_filters[col] = selected
            elif pd.api.types.is_numeric_dtype(df[col]):
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                if min_val != max_val:
                    default = state.custom_filters.get(col, (min_val, max_val))
                    selected = st.slider(col, min_val, max_val, default)
                    active_filters[col] = selected
                    state.custom_filters[col] = selected

        if st.button("Clear All Filters", type="secondary"):
            clear_filters()

    return active_filters


def filter_df_by_state(
    df: pd.DataFrame,
    state: FilterState | None = None,
) -> pd.DataFrame:
    """Filter a DataFrame based on the current global filter state."""
    if state is None:
        state = get_filter_state()

    mask = pd.Series(True, index=df.index)

    if state.date_range and "date" in df.columns:
        dates = pd.to_datetime(df["date"])
        mask &= (dates >= state.date_range[0]) & (dates <= state.date_range[1])

    for col, values in state.custom_filters.items():
        if col not in df.columns:
            continue
        if isinstance(values, list):
            if values:
                mask &= df[col].isin(values)
        elif isinstance(values, tuple) and len(values) == 2:
            mask &= (df[col] >= values[0]) & (df[col] <= values[1])

    return df[mask].copy()


def linked_heatmap_with_drill(
    df: pd.DataFrame,
    x: str,
    y: str,
    z: str,
    title: str = "",
    height: int = 500,
    theme: str | None = None,
    on_click: Callable | None = None,
) -> go.Figure:
    """Heatmap with click-to-drill-down functionality."""
    pivot = df.pivot_table(index=y, columns=x, values=z, aggfunc="mean")

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[[0, "#b2384d"], [0.125, "#d4626f"], [0.25, "#ea9a9f"], [0.375, "#f4cfd0"], [0.5, "#eef1f5"], [0.625, "#c7dbf3"], [0.75, "#8fb6ea"], [0.875, "#5b8def"], [1, "#2f5fb3"]],
        zmid=0,
        text=np.vectorize(lambda v: "—" if pd.isna(v) else f"{v:.2f}")(pivot.values),
        texttemplate="%{text}",
        hovertemplate=(
            f"{y}: %{{y}}<br>{x}: %{{x}}<br>{z}: %{{z:.4f}}<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=title,
        height=height,
        clickmode="event",
    )

    if theme:
        from fedviz.themes import get_theme
        fig = get_theme(theme).apply(fig)

    return fig


def comparison_overlay(
    datasets: dict[str, pd.DataFrame],
    x: str,
    y: str,
    title: str = "",
    height: int = 500,
    theme: str | None = None,
    mode: str = "lines",
) -> go.Figure:
    """Overlay multiple datasets on the same chart for comparison."""
    fig = go.Figure()

    for i, (name, df) in enumerate(datasets.items()):
        if x in df.columns and y in df.columns:
            fig.add_trace(go.Scatter(
                x=df[x], y=df[y],
                mode=mode,
                name=name,
                line=dict(width=2, color=f"hsl({i * 60}, 70%, 50%)"),
                opacity=0.8,
            ))

    fig.update_layout(
        title=title,
        height=height,
        hovermode="x unified",
    )

    if theme:
        from fedviz.themes import get_theme
        fig = get_theme(theme).apply(fig)

    return fig


def animated_time_series(
    df: pd.DataFrame,
    x: str = "date",
    y: str = "value",
    frame_col: str = "date",
    title: str = "",
    height: int = 500,
    theme: str | None = None,
    animation_speed: int = 100,
) -> go.Figure:
    """Animated time series that evolves over time."""
    fig = go.Figure()

    frames = []
    dates = sorted(df[frame_col].unique())

    for i, date in enumerate(dates):
        sub = df[df[frame_col] <= date]
        frame = go.Frame(
            data=[go.Scatter(
                x=sub[x], y=sub[y],
                mode="lines",
                line=dict(width=2, color="#5b8def"),
                fill="tozeroy",
                fillcolor="rgba(91, 141, 239, 0.18)",
            )],
            name=str(date),
        )
        frames.append(frame)

    fig.add_trace(go.Scatter(
        x=df[df[frame_col] == dates[0]][x],
        y=df[df[frame_col] == dates[0]][y],
        mode="lines",
        line=dict(width=2, color="#5b8def"),
        fill="tozeroy",
        fillcolor="rgba(91, 141, 239, 0.18)",
    ))

    fig.frames = frames

    fig.update_layout(
        title=title,
        height=height,
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {
                        "frame": {"duration": animation_speed, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 0},
                    }],
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    }],
                },
            ],
        }],
        sliders=[{
            "active": 0,
            "steps": [
                {
                    "args": [[str(d)], {
                        "frame": {"duration": animation_speed, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    }],
                    "label": str(d),
                    "method": "animate",
                }
                for d in dates
            ],
        }],
    )

    if theme:
        from fedviz.themes import get_theme
        fig = get_theme(theme).apply(fig)

    return fig


def rich_tooltip_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    tooltip_cols: list[str],
    color: str | None = None,
    title: str = "",
    height: int = 500,
    theme: str | None = None,
) -> go.Figure:
    """Scatter plot with rich multi-column tooltips."""
    fig = go.Figure()

    hover_template = f"<b>{x}</b>: %{{x:.4f}}<br><b>{y}</b>: %{{y:.4f}}<br>"
    for col in tooltip_cols:
        hover_template += f"<b>{col}</b>: %{{customdata[{tooltip_cols.index(col)}]:.4f}}<br>"
    hover_template += "<extra></extra>"

    if color:
        groups = df[color].unique()
        for i, grp in enumerate(groups):
            mask = df[color] == grp
            sub = df[mask]
            customdata = sub[tooltip_cols].values.tolist() if tooltip_cols else None
            fig.add_trace(go.Scatter(
                x=sub[x], y=sub[y],
                mode="markers",
                name=str(grp),
                marker=dict(size=8, opacity=0.7),
                customdata=customdata,
                hovertemplate=hover_template,
            ))
    else:
        customdata = df[tooltip_cols].values.tolist() if tooltip_cols else None
        fig.add_trace(go.Scatter(
            x=df[x], y=df[y],
            mode="markers",
            marker=dict(size=8, opacity=0.7, color="#5b8def"),
            customdata=customdata,
            hovertemplate=hover_template,
        ))

    fig.update_layout(title=title, height=height, hovermode="closest")

    if theme:
        from fedviz.themes import get_theme
        fig = get_theme(theme).apply(fig)

    return fig


def brush_range_filter(
    df: pd.DataFrame,
    date_col: str = "date",
    title: str = "Select Date Range",
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Render a brush range filter using Plotly's range slider."""
    dates = pd.to_datetime(df[date_col])
    min_date = dates.min()
    max_date = dates.max()

    start, end = st.date_input(
        title,
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if start and end:
        return pd.Timestamp(start), pd.Timestamp(end)
    return None


def selection_info_panel(
    selected_count: int,
    total_count: int,
    filter_summary: dict[str, Any] | None = None,
) -> None:
    """Display information about the current selection."""
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Selected", f"{selected_count:,}")
    with col2:
        st.metric("Total", f"{total_count:,}")

    if filter_summary:
        st.markdown("**Active Filters:**")
        for key, value in filter_summary.items():
            st.caption(f"- **{key}**: {value}")
