"""Shared plotly figure builders for Q1 results (fedci/questions/q1_total_effect + the API).

Palette constants are copied from the legacy viewer (dml/frontend/streamlit/_common.py)
so backend figures do not import from the Streamlit folder.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

CATEGORICAL = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "dark":  ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
DIVERGING = {"light": ("#2a78d6", "#e34948"), "dark": ("#3987e5", "#e66767")}  # (positive, negative)
INK = {
    "light": dict(primary="#0b0b0b", secondary="#52514e", grid="#e6e5e1", axis="#c9c8c3"),
    "dark":  dict(primary="#ffffff", secondary="#c3c2b7", grid="#2e2e2c", axis="#4a4a47"),
}


def residualized_fit_figure(scored: pd.DataFrame, theta: float, title: str,
                            x_title: str, y_title: str) -> go.Figure:
    """Residualized Y against residualized surprise with the fitted slope through the origin."""
    fig = go.Figure()
    fig.add_scatter(x=scored.s_resid, y=scored.y_resid, mode="markers", name="meetings",
                    text=[str(d.date()) for d in scored.announcement_date],
                    marker=dict(color=CATEGORICAL["light"][0]))
    grid = [float(scored.s_resid.min()), float(scored.s_resid.max())]
    fig.add_scatter(x=grid, y=[theta * g for g in grid], mode="lines",
                    name=f"theta={theta:.3g}", line=dict(color=CATEGORICAL["light"][1]))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, template="plotly_white")
    return fig


def influence_figure(influence: pd.DataFrame) -> go.Figure:
    """Horizontal bar of leave-one-meeting-out theta deltas (top-N rows as passed in)."""
    pos, neg = DIVERGING["light"]
    rows = influence.iloc[::-1]  # biggest influence on top, matching the static export
    dates = [str(d.date()) if hasattr(d, "date") else str(d) for d in rows.announcement_date]
    deltas = [float(v) for v in rows.delta_loo]
    fig = go.Figure()
    fig.add_bar(y=dates, x=deltas, orientation="h", name="delta_loo",
                marker=dict(color=[pos if v >= 0 else neg for v in deltas]))
    fig.update_layout(title="Leave-one-meeting-out sensitivity (theta_loo - theta)",
                      xaxis_title="delta theta", yaxis_title="meeting",
                      template="plotly_white", bargap=0.25)
    return fig


def coefficients_figure(estimates: pd.DataFrame, level: float) -> go.Figure:
    """Forest plot of the estimate rows: theta with CI whiskers, zero reference line."""
    blue, orange = CATEGORICAL["light"][0], CATEGORICAL["light"][1]
    rows = estimates.iloc[::-1]
    names = [str(n) for n in rows.estimator]
    theta = [float(v) for v in rows.theta]
    lows = [float(v) for v in rows.ci_low]
    highs = [float(v) for v in rows.ci_high]
    colors = [orange if "bootstrap" in n else blue for n in names]
    fig = go.Figure()
    fig.add_scatter(
        x=theta, y=names, mode="markers", name="theta",
        marker=dict(color=colors, size=10),
        error_x=dict(type="data", symmetric=False,
                     array=[hi - t for t, hi in zip(theta, highs)],
                     arrayminus=[t - lo for t, lo in zip(theta, lows)],
                     color=INK["light"]["secondary"], thickness=1.6, width=5),
    )
    fig.add_vline(x=0, line=dict(color=INK["light"]["axis"], dash="dash"))
    fig.update_layout(title=f"Estimates with {level:.0%} CI",
                      xaxis_title="theta: percent return per 10bp surprise",
                      template="plotly_white")
    return fig


__all__ = ["CATEGORICAL", "DIVERGING", "INK", "residualized_fit_figure", "influence_figure",
           "coefficients_figure"]
