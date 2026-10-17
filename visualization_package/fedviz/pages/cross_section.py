"""Cross-Section page — analyze cross-sectional patterns in the data.

Provides distribution analysis, scatter matrices, ranking visualizations,
and group-based comparisons across sectors, firms, or any categorical
variable.
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
    distribution,
    heatmap,
    scatter,
    violin_plot,
)
from fedviz.components.filters import (
    categorical_filter,
    date_range_filter,
    multi_select_filter,
    numeric_range_filter,
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
    cross_filter_panel,
    filter_df_by_state,
    linked_scatter,
    rich_tooltip_scatter,
    selection_info_panel,
)
from fedviz.session import get_state, init_session_state, set_state
from fedviz.themes import ACCENT, CATEGORICAL_PALETTE, SECTOR_COLORS


def render(data: VizData | None = None) -> None:
    """Render the Cross-Section page."""
    init_session_state()
    data = data or VizData()
    page_header(
        "Cross-Section",
        "Distributions, co-movement and rankings across sectors (or the columns of any dataset).",
    )

    with st.sidebar:
        st.markdown("**Data selection**")
        dataset_name = st.selectbox(
            "Dataset",
            options=["etf_returns", "fred_controls", "event_study_table", "mps_surprises"],
            index=0,
        )
        sectors = sector_selector(default=get_state("selected_sectors", ["SPY", "XLF", "XLK", "XLE", "XLV"]))
        set_state("selected_sectors", sectors)

    df = data.load(dataset_name)
    if df.empty:
        st.warning(f"No data available for {dataset_name}.")
        return

    is_returns = dataset_name == "etf_returns"
    if is_returns:
        cols = [s for s in sectors if s in df.columns]
    else:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()[:8]
    if not cols:
        st.info("Pick at least one sector.")
        return

    tab_dist, tab_scatter, tab_rank, tab_heatmap = tabs([
        "Distributions", "Scatter matrix", "Rankings", "Correlation",
    ])

    with tab_dist:
        guarded(_render_distributions, df, cols, is_returns, label="Distributions")

    with tab_scatter:
        guarded(_render_scatter_matrix, df, cols, is_returns, label="Scatter matrix")

    with tab_rank:
        if is_returns:
            guarded(_render_rankings, df, cols, label="Rankings")
        else:
            st.info("Rankings are computed from daily returns; choose the etf_returns dataset.")

    with tab_heatmap:
        guarded(_render_correlation_heatmap, df, cols, label="Correlation")


def _render_distributions(df: pd.DataFrame, cols: list[str], is_returns: bool) -> None:
    """One compact histogram per column, then all of them side by side as box plots."""
    grid = st.columns(3, gap="medium")
    for i, col in enumerate(cols):
        fig = distribution(df, col, title=col, height=250, show_kde=True)
        fig.update_traces(marker_color=SECTOR_COLORS.get(col, CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)]),
                          selector=dict(type="histogram"))
        fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title=None, yaxis_showticklabels=False,
                          margin=dict(t=40), xaxis_tickformat=".0%" if is_returns else None)
        with grid[i % 3]:
            show_chart(fig)

    long = df[cols].melt(var_name="series", value_name="value").dropna()
    fig = box_plot(long, y="value", x="series", title="Side by side (box: middle half, whiskers: 1.5 IQR)",
                   height=380)
    fig.update_traces(boxpoints="outliers", marker=dict(size=3, opacity=0.5))
    fig.update_layout(yaxis_title="Daily return" if is_returns else None,
                      yaxis_tickformat=".1%" if is_returns else None)
    show_chart(fig)
    chart_export_menu(fig, df[cols], prefix="distributions")


def _render_scatter_matrix(df: pd.DataFrame, cols: list[str], is_returns: bool) -> None:
    """Every pair of columns at once, then one pair in detail."""
    cols = cols[:6]
    if len(cols) >= 2:
        fmt = ".0%" if is_returns else None
        fig = go.Figure(go.Splom(
            dimensions=[dict(label=c, values=df[c]) for c in cols],
            showupperhalf=False, diagonal_visible=False,
            marker=dict(size=3.5, color=ACCENT, opacity=0.45, line_width=0),
        ))
        axes = {}
        for k in range(1, len(cols) + 1):
            suffix = str(k) if k > 1 else ""
            axes[f"xaxis{suffix}"] = dict(tickformat=fmt, showgrid=True)
            axes[f"yaxis{suffix}"] = dict(tickformat=fmt, showgrid=True)
        fig.update_layout(title="Daily co-movement, every pair" if is_returns else "Every pair",
                          height=130 * len(cols) + 80, hovermode="closest", dragmode="select", **axes)
        show_chart(fig)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        st.markdown("### One pair in detail")
        col_x, col_y = st.columns(2)
        x_default = cols[0] if cols[0] in numeric_cols else numeric_cols[0]
        y_default = cols[1] if len(cols) > 1 and cols[1] in numeric_cols else numeric_cols[1]
        with col_x:
            x_col = st.selectbox("X axis", numeric_cols, index=numeric_cols.index(x_default))
        with col_y:
            y_col = st.selectbox("Y axis", numeric_cols, index=numeric_cols.index(y_default))
        fig = scatter(df, x=x_col, y=y_col, title=f"{y_col} against {x_col}", height=460, add_regression=True,
                      opacity=0.55)
        if is_returns:
            fig.update_layout(xaxis_tickformat=".1%", yaxis_tickformat=".1%")
        show_chart(fig)
        chart_export_menu(fig, df[[x_col, y_col]], prefix=f"{x_col}_vs_{y_col}")


def _render_rankings(df: pd.DataFrame, sectors: list[str]) -> None:
    """Return, risk and tail statistics per sector over the whole sample."""
    metrics = {}
    for sector in sectors:
        r = df[sector].dropna()
        path = (1 + r).cumprod()
        metrics[sector] = {
            "Total return": path.iloc[-1] - 1 if len(path) else np.nan,
            "Annualized vol": r.std() * np.sqrt(252),
            "Sharpe": (r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0,
            "Max drawdown": (path / path.cummax() - 1).min(),
            "Skew": r.skew(),
            "Excess kurtosis": r.kurtosis(),
        }
    metrics_df = pd.DataFrame(metrics).T.sort_values("Total return", ascending=False)

    fig = bar_chart(metrics_df.rename_axis("sector").reset_index(), x="sector", y="Total return",
                    title="Total return over the sample", height=360, text_fmt=".1%", color_scale="sign",
                    sort=False)
    fig.update_layout(yaxis_tickformat=".0%")
    fig.update_traces(textposition="outside", cliponaxis=False)
    show_chart(fig)

    st.dataframe(
        metrics_df,
        width="stretch",
        column_config={
            "Total return": st.column_config.NumberColumn(format="percent"),
            "Annualized vol": st.column_config.NumberColumn(format="percent"),
            "Sharpe": st.column_config.NumberColumn(format="%.2f"),
            "Max drawdown": st.column_config.NumberColumn(format="percent"),
            "Skew": st.column_config.NumberColumn(format="%.2f"),
            "Excess kurtosis": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    chart_export_menu(fig, metrics_df.reset_index(), prefix="sector_rankings")


def _render_correlation_heatmap(df: pd.DataFrame, cols: list[str]) -> None:
    """Correlation matrix of the chosen columns (lower triangle)."""
    corr = df[cols].corr()
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    fig = heatmap(corr.mask(mask), title="Correlation of daily changes", height=120 + 60 * len(cols),
                  text_fmt=".2f", center_zero=True)
    fig.update_traces(zmin=-1, zmax=1, zmid=0, xgap=3, ygap=3, colorbar=dict(title=dict(text="")),
                      hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>")
    fig.update_layout(xaxis=dict(showgrid=False, side="bottom"), yaxis=dict(showgrid=False, autorange="reversed"))
    show_chart(fig)
