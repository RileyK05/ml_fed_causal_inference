"""Export capabilities for fedviz.

Provides functions to export charts as PNG/SVG and data as CSV/JSON.
All exports are triggered through Streamlit download buttons.
"""

from __future__ import annotations

import io
import json
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def export_chart_png(
    fig: go.Figure,
    filename: str = "chart.png",
    width: int = 1200,
    height: int = 800,
    scale: int = 2,
) -> None:
    """Export a Plotly figure as PNG."""
    try:
        img_bytes = fig.to_image(format="png", width=width, height=height, scale=scale)
        st.download_button(
            label="Download PNG",
            data=img_bytes,
            file_name=filename,
            mime="image/png",
        )
    except Exception as e:
        st.error(f"PNG export failed: {e}")
        st.info("Install kaleido: pip install kaleido")


def export_chart_svg(
    fig: go.Figure,
    filename: str = "chart.svg",
    width: int = 1200,
    height: int = 800,
) -> None:
    """Export a Plotly figure as SVG."""
    try:
        svg_bytes = fig.to_image(format="svg", width=width, height=height)
        st.download_button(
            label="Download SVG",
            data=svg_bytes,
            file_name=filename,
            mime="image/svg+xml",
        )
    except Exception as e:
        st.error(f"SVG export failed: {e}")


def export_chart_html(
    fig: go.Figure,
    filename: str = "chart.html",
) -> None:
    """Export a Plotly figure as interactive HTML."""
    html_str = fig.to_html(include_plotlyjs="cdn", full_html=True)
    st.download_button(
        label="Download HTML",
        data=html_str,
        file_name=filename,
        mime="text/html",
    )


def export_df_csv(
    df: pd.DataFrame,
    filename: str = "data.csv",
) -> None:
    """Export a DataFrame as CSV."""
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=filename,
        mime="text/csv",
    )


def export_df_json(
    df: pd.DataFrame,
    filename: str = "data.json",
) -> None:
    """Export a DataFrame as JSON."""
    json_str = df.to_json(orient="records", date_format="iso")
    st.download_button(
        label="Download JSON",
        data=json_str,
        file_name=filename,
        mime="application/json",
    )


def export_df_excel(
    df: pd.DataFrame,
    filename: str = "data.xlsx",
) -> None:
    """Export a DataFrame as Excel."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Data")
    st.download_button(
        label="Download Excel",
        data=output.getvalue(),
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def chart_export_menu(
    fig: go.Figure,
    df: pd.DataFrame | None = None,
    prefix: str = "chart",
) -> None:
    """Render a complete export menu for a chart and optional data."""
    with st.expander("Export"):
        col1, col2, col3 = st.columns(3)
        with col1:
            export_chart_png(fig, f"{prefix}.png")
        with col2:
            export_chart_svg(fig, f"{prefix}.svg")
        with col3:
            export_chart_html(fig, f"{prefix}.html")

        if df is not None:
            st.markdown("**Export Data**")
            col4, col5, col6 = st.columns(3)
            with col4:
                export_df_csv(df, f"{prefix}_data.csv")
            with col5:
                export_df_json(df, f"{prefix}_data.json")
            with col6:
                export_df_excel(df, f"{prefix}_data.xlsx")


def export_summary_report(
    title: str,
    metrics: dict[str, Any],
    figures: list[go.Figure],
    dataframes: list[pd.DataFrame],
) -> None:
    """Export a complete summary report as HTML."""
    html_parts = [
        "<html><head>",
        f"<title>{title}</title>",
        "<style>",
        "body { font-family: sans-serif; margin: 40px; }",
        "h1 { color: #333; }",
        "h2 { color: #666; }",
        "table { border-collapse: collapse; width: 100%; }",
        "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "th { background-color: #f2f2f2; }",
        "</style></head><body>",
        f"<h1>{title}</h1>",
    ]

    if metrics:
        html_parts.append("<h2>Metrics</h2><table>")
        html_parts.append("<tr><th>Metric</th><th>Value</th></tr>")
        for k, v in metrics.items():
            html_parts.append(f"<tr><td>{k}</td><td>{v}</td></tr>")
        html_parts.append("</table>")

    for i, fig in enumerate(figures):
        html_parts.append(f"<h2>Figure {i + 1}</h2>")
        html_parts.append(fig.to_html(include_plotlyjs="cdn", full_html=False))

    for i, df in enumerate(dataframes):
        html_parts.append(f"<h2>Data {i + 1}</h2>")
        html_parts.append(df.to_html(index=False))

    html_parts.append("</body></html>")

    html_str = "\n".join(html_parts)
    st.download_button(
        label="Download Report (HTML)",
        data=html_str,
        file_name=f"{title.replace(' ', '_')}_report.html",
        mime="text/html",
    )
