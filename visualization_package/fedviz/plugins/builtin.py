"""Built-in plugins for fedviz.

These plugins provide model-specific visualizations for the DML,
encoder, and CDE projects. They demonstrate how to extend the
generalized visualization suite with custom views.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fedviz.plugins.base import ChartPlugin, MetricPlugin, ModelPlugin, PagePlugin
from fedviz.themes import CATEGORICAL_PALETTE


def _dml_overview_page(data=None) -> None:
    """DML-specific overview page."""
    from fedviz.data import VizData
    data = data or VizData()
    st.markdown("## DML Model Overview")
    st.markdown("Partially Linear DML estimation results and diagnostics.")

    runs_df = data.list_results(project="dml")
    if runs_df.empty:
        st.info("No DML runs found.")
        return

    for _, row in runs_df.iterrows():
        with st.expander(f"{row['name']} ({row['role']})"):
            run_data = data.load_run(row["run_dir"])
            manifest = run_data.get("manifest", {})
            metrics = manifest.get("metrics", {})
            if metrics:
                cols = st.columns(min(4, len(metrics)))
                for i, (k, v) in enumerate(metrics.items()):
                    with cols[i % 4]:
                        st.metric(k, f"{v:.4f}" if isinstance(v, float) else str(v))


def _encoder_overview_page(data=None) -> None:
    """Encoder-specific overview page."""
    from fedviz.data import VizData
    data = data or VizData()
    st.markdown("## Encoder Model Overview")
    st.markdown("Patch-transformer encoder results and attention diagnostics.")

    runs_df = data.list_results(project="encoder")
    if runs_df.empty:
        st.info("No encoder runs found.")
        return

    for _, row in runs_df.iterrows():
        with st.expander(f"{row['name']} ({row['role']})"):
            run_data = data.load_run(row["run_dir"])
            manifest = run_data.get("manifest", {})
            metrics = manifest.get("metrics", {})
            if metrics:
                cols = st.columns(min(4, len(metrics)))
                for i, (k, v) in enumerate(metrics.items()):
                    with cols[i % 4]:
                        st.metric(k, f"{v:.4f}" if isinstance(v, float) else str(v))


def _cde_overview_page(data=None) -> None:
    """CDE-specific overview page."""
    from fedviz.data import VizData
    data = data or VizData()
    st.markdown("## Neural CDE Model Overview")
    st.markdown("Neural CDE encoder results and signature diagnostics.")

    runs_df = data.list_results(project="cde")
    if runs_df.empty:
        st.info("No CDE runs found.")
        return

    for _, row in runs_df.iterrows():
        with st.expander(f"{row['name']} ({row['role']})"):
            run_data = data.load_run(row["run_dir"])
            manifest = run_data.get("manifest", {})
            metrics = manifest.get("metrics", {})
            if metrics:
                cols = st.columns(min(4, len(metrics)))
                for i, (k, v) in enumerate(metrics.items()):
                    with cols[i % 4]:
                        st.metric(k, f"{v:.4f}" if isinstance(v, float) else str(v))


def _coefficient_chart(df: pd.DataFrame, **kwargs) -> go.Figure:
    """Custom coefficient chart for DML results."""
    fig = go.Figure()
    if "name" in df.columns and "estimate" in df.columns:
        fig.add_trace(go.Bar(
            x=df["name"],
            y=df["estimate"],
            marker_color=CATEGORICAL_PALETTE[0],
        ))
    fig.update_layout(title="Coefficients", **kwargs)
    return fig


def _attention_heatmap(df: pd.DataFrame, **kwargs) -> go.Figure:
    """Custom attention heatmap for encoder results."""
    fig = go.Figure(data=go.Heatmap(
        z=df.values,
        x=df.columns.tolist(),
        y=df.index.tolist(),
        colorscale=[[0, "#1b2f55"], [0.35, "#2f5fb3"], [0.7, "#5b8def"], [1, "#bcd3fa"]],
    ))
    fig.update_layout(title="Attention Weights", **kwargs)
    return fig


def _compute_sharpe(returns: pd.Series) -> float:
    """Compute Sharpe ratio."""
    if returns.std() == 0:
        return 0.0
    return returns.mean() / returns.std() * (252 ** 0.5)


def _compute_max_drawdown(returns: pd.Series) -> float:
    """Compute maximum drawdown."""
    cum = (1 + returns.fillna(0)).cumprod()
    running_max = cum.expanding().max()
    return (cum / running_max - 1).min()


dml_plugin = ModelPlugin(
    name="dml",
    display_name="DML (Q1)",
    description="Partially Linear DML model for total effect estimation",
    pages={
        "dml_overview": PagePlugin(
            name="dml_overview",
            icon="",
            render_fn=_dml_overview_page,
            description="DML model overview",
            category="model",
            order=10,
        ),
    },
    charts={
        "coefficient_chart": ChartPlugin(
            name="coefficient_chart",
            render_fn=_coefficient_chart,
            description="Coefficient bar chart",
        ),
    },
    metrics={
        "sharpe": MetricPlugin(
            name="sharpe",
            compute_fn=_compute_sharpe,
            description="Sharpe ratio",
            format="number",
        ),
        "max_drawdown": MetricPlugin(
            name="max_drawdown",
            compute_fn=_compute_max_drawdown,
            description="Maximum drawdown",
            format="percent",
        ),
    },
)

encoder_plugin = ModelPlugin(
    name="encoder",
    display_name="Encoder (Q3)",
    description="Patch-transformer encoder for firm response prediction",
    pages={
        "encoder_overview": PagePlugin(
            name="encoder_overview",
            icon="",
            render_fn=_encoder_overview_page,
            description="Encoder model overview",
            category="model",
            order=20,
        ),
    },
    charts={
        "attention_heatmap": ChartPlugin(
            name="attention_heatmap",
            render_fn=_attention_heatmap,
            description="Attention weight heatmap",
        ),
    },
)

cde_plugin = ModelPlugin(
    name="cde",
    display_name="Neural CDE (Q3)",
    description="Neural CDE encoder for firm response prediction",
    pages={
        "cde_overview": PagePlugin(
            name="cde_overview",
            icon="",
            render_fn=_cde_overview_page,
            description="CDE model overview",
            category="model",
            order=30,
        ),
    },
)
