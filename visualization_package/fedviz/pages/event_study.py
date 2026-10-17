"""Event Study page — analyze market behavior around FOMC meetings.

Provides event-time alignment, cross-sectional analysis, surprise
vs response visualization, and cumulative average abnormal returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fedviz.components.layout import show_chart
from plotly.subplots import make_subplots

from fedviz.components.charts import (
    bar_chart,
    box_plot,
    cumulative_path,
    distribution,
    heatmap,
    scatter,
    time_series,
    violin_plot,
)
from fedviz.components.filters import (
    date_range_filter,
    meeting_selector,
    multi_select_filter,
    numeric_range_filter,
    sector_selector,
)
from fedviz.components.layout import (
    columns,
    metric_card,
    page_header,
    section_header,
    tabs,
)
from fedviz.data import VizData
from fedviz.export import chart_export_menu
from fedviz.interactive import (
    cross_filter_panel,
    filter_df_by_state,
    get_filter_state,
    linked_scatter,
    rich_tooltip_scatter,
    selection_info_panel,
)
from fedviz.session import get_state, init_session_state, set_state
from fedviz.themes import ACCENT, EVENT_COLOR, REFERENCE, SECTOR_COLORS, WARNING
from fedviz.utils import cumulative_returns


def render(data: VizData | None = None) -> None:
    """Render the Event Study page."""
    init_session_state()
    data = data or VizData()
    page_header(
        "Event Study",
        "How sector returns move in the days around each FOMC announcement.",
        icon="",
    )

    with st.sidebar:
        st.markdown("**Event Window**")
        pre_window = st.slider("Pre-event days", 1, 20, get_state("pre_window", 5))
        post_window = st.slider("Post-event days", 1, 20, get_state("post_window", 5))
        set_state("pre_window", pre_window)
        set_state("post_window", post_window)
        st.markdown("**Filters**")
        sectors = sector_selector(default=get_state("selected_sectors", ["SPY", "XLF", "XLK", "XLE", "XLV"]))
        set_state("selected_sectors", sectors)

    etf_returns = data.get_etf_returns()
    meetings = data.get_meetings()
    surprises = data.get_surprises()

    if etf_returns.empty or meetings.empty:
        st.warning("Insufficient data for event study.")
        return

    event_panel = data.event_panel("etf_returns", pre=pre_window, post=post_window)

    tab_timeline, tab_cross_section, tab_surprise, tab_distribution = tabs([
        "Paths", "By event day", "Surprise vs response", "Distribution",
    ])

    with tab_timeline:
        _render_event_timeline(event_panel, sectors, pre_window, post_window)

    with tab_cross_section:
        _render_cross_section(event_panel, sectors, pre_window, post_window)

    with tab_surprise:
        _render_surprise_vs_response(event_panel, surprises, sectors)

    with tab_distribution:
        _render_distribution(event_panel, sectors, pre_window, post_window)


def _paths(event_panel: pd.DataFrame, sector: str) -> pd.DataFrame:
    """One sector's cumulative return path per meeting, in event time."""
    d = event_panel[["rel_day", "announcement_date", sector]].rename(columns={sector: "return"})
    d = d.sort_values(["announcement_date", "rel_day"])
    d["cum_return"] = d.groupby("announcement_date")["return"].transform(lambda r: (1 + r.fillna(0)).cumprod() - 1)
    return d


def _render_event_timeline(
    event_panel: pd.DataFrame,
    sectors: list[str],
    pre_window: int,
    post_window: int,
) -> None:
    """Render event-time cumulative return paths."""
    if "rel_day" not in event_panel.columns:
        st.warning("Event panel missing rel_day column.")
        return
    sectors = [s for s in sectors if s in event_panel.columns]
    if not sectors:
        st.info("None of the selected sectors are in the event panel.")
        return

    avg_df = pd.concat([_paths(event_panel, s).groupby("rel_day")["cum_return"].mean().rename(s)
                        for s in sectors], axis=1).reset_index()
    fig = time_series(
        avg_df, x="rel_day", y=sectors,
        title=f"Average cumulative return around FOMC meetings ({event_panel['announcement_date'].nunique()} meetings)",
        height=420,
    )
    fig.add_vline(x=0, line=dict(color=EVENT_COLOR, width=1))
    fig.add_hline(y=0, line=dict(color=REFERENCE, dash="dot", width=1))
    fig.update_layout(yaxis_tickformat=".1%", xaxis_title="Trading days from the announcement", xaxis_dtick=1)
    show_chart(fig)

    st.markdown("### Meeting by meeting")
    st.caption("Each faint line is one meeting; the bold line is the average, the band the middle half.")
    cols = st.columns(2, gap="medium")
    for i, sector in enumerate(sectors):
        sector_data = _paths(event_panel, sector)
        fig = cumulative_path(
            sector_data,
            x="rel_day",
            y="cum_return",
            group="announcement_date",
            title=sector,
            height=340,
            show_mean=True,
            confidence_band=True,
            color=SECTOR_COLORS.get(sector),
        )
        fig.update_layout(xaxis_title=None, yaxis_title=None, showlegend=False)
        with cols[i % 2]:
            show_chart(fig)


def _render_cross_section(
    event_panel: pd.DataFrame,
    sectors: list[str],
    pre_window: int,
    post_window: int,
) -> None:
    """Mean return by sector on chosen event days, and how spread out returns are each day."""
    sectors = [s for s in sectors if s in event_panel.columns]
    all_days = list(range(-pre_window, post_window + 1))
    default_days = [d for d in [-1, 0, 1, 5] if d in all_days]
    key_days = st.pills("Event days", all_days, default=default_days, selection_mode="multi",
                        format_func=lambda d: f"day {d:+d}" if d else "day 0")
    if not key_days or not sectors:
        st.info("Select at least one event day.")
        return

    rows = [{"sector": s, "day": f"day {d:+d}" if d else "day 0", "mean_return": event_panel.loc[
        event_panel["rel_day"] == d, s].mean()} for d in sorted(key_days) for s in sectors]
    fig = bar_chart(pd.DataFrame(rows), x="sector", y="mean_return", color="day", sort=False,
                    title="Mean return by sector on each event day", height=380, text_fmt=".2%")
    fig.update_layout(barmode="group", yaxis_tickformat=".2%", bargap=0.25)
    fig.update_traces(textposition="none")
    show_chart(fig)

    stats = []
    for day in all_days:
        vals = event_panel.loc[event_panel["rel_day"] == day, sectors].to_numpy().ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size:
            stats.append({"rel_day": day, "mean": vals.mean(), "median": np.median(vals),
                          "q25": np.percentile(vals, 25), "q75": np.percentile(vals, 75), "std": vals.std()})
    if stats:
        d = pd.DataFrame(stats)
        fig = go.Figure([
            go.Scatter(x=d.rel_day, y=d.q75, mode="lines", line_width=0, showlegend=False, hoverinfo="skip"),
            go.Scatter(x=d.rel_day, y=d.q25, mode="lines", line_width=0, fill="tonexty",
                       fillcolor="rgba(91,141,239,0.16)", name="Middle 50%", hoverinfo="skip"),
            go.Scatter(x=d.rel_day, y=d["mean"], mode="lines+markers", name="Mean", line=dict(width=2.5)),
            go.Scatter(x=d.rel_day, y=d["median"], mode="lines", name="Median", line=dict(dash="dot")),
        ])
        fig.add_vline(x=0, line=dict(color=EVENT_COLOR, width=1))
        fig.update_layout(title="Spread of daily returns across sectors and meetings", height=380,
                          yaxis_tickformat=".2%", xaxis_title="Trading days from the announcement",
                          xaxis_dtick=1, hovermode="x unified")
        show_chart(fig)


def _render_surprise_vs_response(
    event_panel: pd.DataFrame,
    surprises: pd.DataFrame,
    sectors: list[str],
) -> None:
    """Announcement-day return against the policy surprise, one panel per sector."""
    if surprises.empty:
        st.warning("No surprise data available.")
        return

    surprise_col = next((c for c in ["STMT", "MP1", "ME", "surprise"] if c in surprises.columns), None)
    if surprise_col is None:
        st.warning("No recognized surprise column found.")
        return

    merged = event_panel.merge(
        surprises[["Date", surprise_col]].rename(columns={"Date": "announcement_date"}),
        on="announcement_date",
        how="inner",
    )
    day0 = merged[merged["rel_day"] == 0].copy()
    if day0.empty:
        st.info("No meetings with both a surprise and returns.")
        return
    day0["meeting"] = pd.to_datetime(day0["announcement_date"]).dt.strftime("%Y-%m-%d")
    st.caption(f"{day0['meeting'].nunique()} meetings. Surprise: USMPD `{surprise_col}`, percentage points; "
               "positive = hawkish. Dashed line: least-squares fit.")

    sectors = [s for s in sectors if s in day0.columns]
    cols = st.columns(2, gap="medium")
    for i, sector in enumerate(sectors):
        fig = scatter(day0.dropna(subset=[surprise_col, sector]), x=surprise_col, y=sector, text="meeting",
                      title=sector, height=320, add_regression=True,
                      xaxis_title="Surprise (pp)", yaxis_title="Announcement-day return")
        fig.update_traces(marker=dict(size=9, color=SECTOR_COLORS.get(sector, ACCENT)), selector=dict(mode="markers"))
        fig.update_traces(hovertemplate="%{text}<br>surprise %{x:.3f}<br>return %{y:.2%}<extra></extra>",
                          selector=dict(mode="markers"))
        fig.add_hline(y=0, line=dict(color=REFERENCE, dash="dot", width=1))
        fig.add_vline(x=0, line=dict(color=REFERENCE, dash="dot", width=1))
        fig.update_layout(yaxis_tickformat=".1%", legend=dict(y=1.0))
        with cols[i % 2]:
            show_chart(fig)

    fig = distribution(surprises, surprise_col, title="Distribution of FOMC surprises (all statements)",
                       xaxis_title="Surprise (pp)", height=320, show_kde=True, show_rug=True)
    fig.update_layout(showlegend=False)
    show_chart(fig)


def _render_distribution(
    event_panel: pd.DataFrame,
    sectors: list[str],
    pre_window: int,
    post_window: int,
) -> None:
    """Return distributions in the event window: pooled, then by event day."""
    sectors = [s for s in sectors if s in event_panel.columns]
    long = event_panel.melt(id_vars=["rel_day"], value_vars=sectors, var_name="sector",
                            value_name="return").dropna(subset=["return"])
    if long.empty:
        st.info("No returns in the event window.")
        return

    fig = distribution(long, "return", title="All event-window daily returns, pooled",
                       xaxis_title="Daily return", height=340, show_kde=True)
    fig.update_layout(xaxis_tickformat=".1%", showlegend=False)
    show_chart(fig)

    fig = violin_plot(long.sort_values("rel_day"), y="return", x="rel_day", title="Daily returns by event day",
                      height=420, show_box=True)
    for trace in fig.data:  # one colour; the announcement day stands out
        c = WARNING if trace.name == "0" else ACCENT
        trace.update(line_color=c, fillcolor=c, opacity=0.55 if trace.name != "0" else 0.85, meanline_visible=True)
    fig.update_layout(yaxis_tickformat=".1%", yaxis_title="Daily return",
                      xaxis_title="Trading days from the announcement", showlegend=False)
    show_chart(fig)
