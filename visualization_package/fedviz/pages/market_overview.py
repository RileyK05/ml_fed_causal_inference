"""Market Overview page — see the market evolve through the FOMC lens.

This page provides a comprehensive view of how markets have evolved
around FOMC meetings, with interactive charts for sector performance,
cumulative returns, volatility, and yield curve dynamics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fedviz.components.layout import show_chart
from plotly.subplots import make_subplots

from fedviz.animation import animated_sector_evolution, meeting_by_meeting_animation
from fedviz.components.charts import (
    bar_chart,
    distribution,
    cumulative_path,
    drawdown_chart,
    heatmap,
    rolling_metric,
    time_series,
    time_series_with_events,
)
from fedviz.components.filters import (
    date_range_filter,
    multi_select_filter,
    sector_selector,
)
from fedviz.components.layout import (
    columns,
    guarded,
    metric_card,
    page_header,
    section_header,
    tabs,
)
from fedviz.data import VizData
from fedviz.export import chart_export_menu
from fedviz.interactive import (
    FilterState,
    clear_filters,
    cross_filter_panel,
    filter_df_by_state,
    get_filter_state,
    linked_heatmap_with_drill,
    linked_time_series,
    selection_info_panel,
)
from fedviz.session import get_state, init_session_state, set_state
from fedviz.themes import SECTOR_COLORS, get_theme


def render(data: VizData | None = None) -> None:
    """Render the Market Overview page."""
    init_session_state()
    data = data or VizData()
    page_header(
        "Market Overview",
        "Sector performance, volatility and rates, with every FOMC meeting marked.",
        icon="",
    )

    etf_returns = data.get_etf_returns()
    meetings = data.get_meetings()
    treasury = data.get_treasury_1y()
    fred = data.get_fred_controls()

    if etf_returns.empty:
        st.warning("No ETF returns data available.")
        return

    with st.sidebar:
        st.markdown("**Filters**")
        sectors = sector_selector(default=get_state("selected_sectors", ["SPY", "XLF", "XLK", "XLE", "XLV"]))
        set_state("selected_sectors", sectors)
        span_years = (etf_returns["date"].max() - etf_returns["date"].min()).days / 365.25
        windows = {"6M": 0.5, "1Y": 1, "2Y": 2, "3Y": 3, "5Y": 5, "10Y": 10}
        options = [k for k, v in windows.items() if v < span_years] + ["All"]
        window = st.segmented_control("Window", options, default="All") or "All"

        st.markdown("**Animation**")
        anim_speed = st.slider("Frame time (ms)", 50, 500, 100, step=25)
        set_state("animation_speed", anim_speed)

    if window == "All":
        etf_filtered = etf_returns.copy()
    else:
        start_date = etf_returns["date"].max() - pd.DateOffset(months=int(12 * windows[window]))
        etf_filtered = etf_returns[etf_returns["date"] >= start_date].copy()

    available_sectors = [s for s in sectors if s in etf_filtered.columns]
    if not available_sectors:
        st.warning("No selected sectors available in data.")
        return

    tab_overview, tab_sectors, tab_volatility, tab_yields, tab_drawdown, tab_animated = tabs([
        "Overview", "Sector Performance", "Volatility", "Yields & Rates", "Drawdown", "Animated",
    ])

    with tab_overview:
        guarded(_render_overview, etf_filtered, available_sectors, meetings, treasury, label="Overview")

    with tab_sectors:
        guarded(_render_sector_performance, etf_filtered, available_sectors, meetings, label="Sector performance")

    with tab_volatility:
        guarded(_render_volatility, etf_filtered, available_sectors, fred, label="Volatility")

    with tab_yields:
        guarded(_render_yields, treasury, fred, meetings, label="Yields & rates")

    with tab_drawdown:
        guarded(_render_drawdown, etf_filtered, available_sectors, label="Drawdown")

    with tab_animated:
        guarded(_render_animated, etf_filtered, available_sectors, meetings, anim_speed, label="Animation")


def _render_overview(
    etf_returns: pd.DataFrame,
    sectors: list[str],
    meetings: pd.DataFrame,
    treasury: pd.DataFrame,
) -> None:
    """Render the overview tab with key metrics and main chart."""
    col1, col2, col3, col4 = columns(4)

    spy_returns = etf_returns["SPY"] if "SPY" in etf_returns.columns else etf_returns[sectors[0]]
    total_return = (1 + spy_returns.fillna(0)).prod() - 1
    annualized = (1 + total_return) ** (252 / len(spy_returns)) - 1
    volatility = spy_returns.std() * (252 ** 0.5)
    sharpe = annualized / volatility if volatility > 0 else 0

    first, last = etf_returns["date"].min(), etf_returns["date"].max()
    span = f"SPY, {first:%b %Y} to {last:%b %Y}"
    with col1:
        metric_card("Total return", f"{total_return * 100:.1f}%", help=span)
    with col2:
        metric_card("Annualized", f"{annualized * 100:.1f}%", help=span)
    with col3:
        metric_card("Volatility", f"{volatility * 100:.1f}%", help="Annualized standard deviation of daily returns")
    with col4:
        metric_card("Sharpe", f"{sharpe:.2f}", help="Annualized return / volatility (no risk-free rate)")

    cum = etf_returns[["date"]].copy()
    for sector in sectors[:6]:
        cum[sector] = (1 + etf_returns[sector].fillna(0)).cumprod() - 1
    fig = time_series_with_events(
        cum, x="date", y=sectors[:6],
        events=meetings,
        title="Cumulative sector returns, FOMC meetings marked",
        height=460,
    )
    fig.update_layout(yaxis_tickformat=".0%")
    show_chart(fig)
    chart_export_menu(fig, cum, prefix="overview_cumulative")

    if not treasury.empty:
        y_col = "value" if "value" in treasury.columns else treasury.columns[-1]
        recent = treasury[pd.to_datetime(treasury["Date"]) >= etf_returns["date"].min()]
        fig = time_series_with_events(
            recent if len(recent) else treasury, x="Date", y=y_col,
            events=meetings,
            title="1-year Treasury yield",
            yaxis_title="Yield (%)",
            height=340,
        )
        show_chart(fig)


def _render_sector_performance(
    etf_returns: pd.DataFrame,
    sectors: list[str],
    meetings: pd.DataFrame,
) -> None:
    """Render sector performance heatmap and bar charts."""
    if not meetings.empty and "announcement_date" in meetings.columns:
        meeting_dates = meetings["announcement_date"].tolist()
        records = []
        for date in meeting_dates:
            idx = etf_returns["date"].searchsorted(date)
            if idx < 0 or idx >= len(etf_returns):
                continue
            row = {"meeting": pd.Timestamp(date).strftime("%Y-%m-%d")}
            for sector in sectors:
                if sector in etf_returns.columns:
                    row[sector] = etf_returns[sector].iloc[idx]
            records.append(row)

        if records:
            meeting_returns = pd.DataFrame(records)
            meeting_returns = meeting_returns.set_index("meeting")

            grid = meeting_returns.T
            fig = heatmap(grid, title="Announcement-day return by sector and meeting",
                          height=80 + 52 * len(grid), text_fmt=".1%")
            fig.update_traces(
                hovertemplate="%{y} on %{x}: %{z:.2%}<extra></extra>",
                colorbar=dict(tickformat=".0%", title=dict(text="")),
                xgap=2, ygap=2, textfont=dict(size=9),
            )
            fig.update_layout(xaxis=dict(type="category", tickangle=-45, showgrid=False),
                              yaxis=dict(showgrid=False))
            show_chart(fig)

            avg_returns = meeting_returns.mean().sort_values(ascending=False)
            fig = bar_chart(
                avg_returns.rename("avg_return").rename_axis("sector").reset_index(),
                x="sector", y="avg_return",
                title="Average announcement-day return",
                height=360,
                text_fmt=".2%",
                color_scale="sign",
                sort=False,
            )
            fig.update_layout(yaxis_tickformat=".2%")
            fig.update_traces(textposition="outside", cliponaxis=False)
            show_chart(fig)

    cum_returns = pd.DataFrame()
    cum_returns["date"] = etf_returns["date"]
    for sector in sectors:
        if sector in etf_returns.columns:
            cum_returns[sector] = (1 + etf_returns[sector].fillna(0)).cumprod() - 1

    fig = time_series_with_events(
        cum_returns, x="date", y=sectors,
        events=meetings,
        title="Cumulative return by sector",
        height=460,
    )
    fig.update_layout(yaxis_tickformat=".0%")
    show_chart(fig)
    chart_export_menu(fig, cum_returns, prefix="cumulative_returns")


def _render_volatility(
    etf_returns: pd.DataFrame,
    sectors: list[str],
    fred: pd.DataFrame,
) -> None:
    """Render volatility analysis charts."""
    window = st.segmented_control("Rolling window (trading days)", [21, 63, 126, 252], default=63) or 63

    vol_df = pd.DataFrame()
    vol_df["date"] = etf_returns["date"]
    for sector in sectors:
        if sector in etf_returns.columns:
            vol_df[sector] = etf_returns[sector].rolling(window).std() * (252 ** 0.5)

    fig = time_series(
        vol_df, x="date", y=sectors,
        title=f"Rolling {window}-day annualized volatility",
        height=420,
    )
    fig.update_layout(yaxis_tickformat=".0%")
    show_chart(fig)

    if not fred.empty and "VIX" in fred.columns:
        fig = time_series(
            fred, x="date", y="VIX",
            title="VIX (Volatility Index)",
            yaxis_title="VIX",
            height=350,
        )
        show_chart(fig)

    section_header("Return distribution")
    if sectors:
        fig = distribution(
            etf_returns, column=sectors[0],
            title=f"{sectors[0]} daily returns",
            xaxis_title="Daily return",
            height=320,
        )
        fig.update_layout(xaxis_tickformat=".1%", showlegend=False)
        show_chart(fig)


def _render_yields(
    treasury: pd.DataFrame,
    fred: pd.DataFrame,
    meetings: pd.DataFrame,
) -> None:
    """Render yield and rates charts."""
    if not treasury.empty:
        y_col = "value" if "value" in treasury.columns else treasury.columns[-1]
        fig = time_series_with_events(
            treasury, x="Date", y=y_col,
            events=meetings,
            title="1-year Treasury yield",
            yaxis_title="Yield (%)",
            height=400,
        )
        show_chart(fig)

    if not fred.empty:
        rate_cols = [c for c in ["DGS2", "DGS10", "T10Y2Y", "DGS3MO"] if c in fred.columns]
        if rate_cols:
            fig = time_series_with_events(
                fred, x="date", y=rate_cols,
                events=meetings,
                title="Treasury rates",
                yaxis_title="Rate (%)",
                height=400,
            )
            show_chart(fig)

        if "T10Y2Y" in fred.columns:
            fig = time_series(
                fred, x="date", y="T10Y2Y",
                title="Term spread (10Y minus 2Y)",
                yaxis_title="Spread (%)",
                height=300,
            )
            show_chart(fig)


def _render_drawdown(
    etf_returns: pd.DataFrame,
    sectors: list[str],
) -> None:
    """Render drawdown analysis."""
    cols = st.columns(min(3, len(sectors)) or 1)
    for i, sector in enumerate(sectors[:6]):
        if sector in etf_returns.columns:
            fig = drawdown_chart(
                etf_returns.set_index("date")[sector],
                title=f"{sector} drawdown from peak",
                height=260,
            )
            fig.update_layout(yaxis_title=None, showlegend=False, yaxis_tickformat=".0%")
            with cols[i % len(cols)]:
                show_chart(fig)


def _render_animated(
    etf_returns: pd.DataFrame,
    sectors: list[str],
    meetings: pd.DataFrame,
    speed: int,
) -> None:
    """Render animated visualizations."""
    fig = animated_sector_evolution(
        etf_returns,
        date_col="date",
        sector_cols=sectors[:8],
        title="Cumulative returns, played forward",
        height=520,
        speed=speed,
    )
    show_chart(fig)


    event_panel = None
    try:
        from fedcore.data.panels import event_panel as make_event_panel
        from fedviz.data import wide_event_panel

        event_panel = wide_event_panel(make_event_panel("etf_returns", pre=5, post=5))
    except Exception:
        pass

    if event_panel is not None and not event_panel.empty:
        fig = meeting_by_meeting_animation(
            event_panel,
            sector_cols=sectors[:6],
            title="Returns around each FOMC meeting, one meeting per frame",
            height=600,
            speed=max(200, speed * 2),
        )
        show_chart(fig)
    else:
        st.info("Event panel data not available for animation.")
