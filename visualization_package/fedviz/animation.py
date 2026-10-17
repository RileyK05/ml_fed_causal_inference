"""Animation system for fedviz — bring market evolution to life.

Provides time-based animations that show how markets, sectors, and
indicators evolve over time. Supports play/pause, speed control,
and frame-by-frame scrubbing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


def _play_controls(speed: int, redraw: bool = False) -> list[dict]:
    """Play/pause buttons, placed under the chart on the left."""
    return [{
        "type": "buttons", "direction": "left", "showactive": False,
        "x": 0, "xanchor": "left", "y": -0.12, "yanchor": "top", "pad": {"r": 8, "t": 0},
        "bgcolor": "rgba(139,148,167,0.12)", "bordercolor": "rgba(139,148,167,0.3)",
        "font": {"size": 12},
        "buttons": [
            {"label": "▶  Play", "method": "animate",
             "args": [None, {"frame": {"duration": speed, "redraw": redraw}, "fromcurrent": True,
                             "transition": {"duration": 0}}]},
            {"label": "❚❚  Pause", "method": "animate",
             "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate",
                               "transition": {"duration": 0}}]},
        ],
    }]


def _slider(names: list[str], labels: list[str], speed: int, redraw: bool = False) -> list[dict]:
    return [{
        "active": 0, "x": 0.16, "len": 0.84, "y": -0.06, "yanchor": "top",
        "pad": {"t": 0, "b": 0}, "ticklen": 0, "minorticklen": 0,
        "bgcolor": "rgba(139,148,167,0.25)", "bordercolor": "rgba(0,0,0,0)",
        "activebgcolor": "#5b8def", "font": {"size": 10},
        "currentvalue": {"prefix": "", "visible": True, "xanchor": "right", "font": {"size": 12}},
        "steps": [
            {"args": [[n], {"frame": {"duration": speed, "redraw": redraw}, "mode": "immediate",
                            "transition": {"duration": 0}}],
             "label": lab, "method": "animate"}
            for n, lab in zip(names, labels)
        ],
    }]


def animated_sector_evolution(
    df: pd.DataFrame,
    date_col: str = "date",
    sector_cols: list[str] | None = None,
    title: str = "Sector Evolution Over Time",
    height: int = 600,
    theme: str | None = None,
    speed: int = 100,
    show_cumulative: bool = True,
    max_frames: int = 60,
) -> go.Figure:
    """Animated sector performance evolution with play/pause controls.

    Lines grow from the first date to the last over at most ``max_frames`` frames; the
    axes are fixed to the full range so the picture does not rescale while it plays.
    """
    from fedviz.themes import CATEGORICAL_PALETTE, SECTOR_COLORS

    if sector_cols is None:
        sector_cols = [c for c in df.columns if c != date_col
                       and pd.api.types.is_numeric_dtype(df[c])][:10]
    sector_cols = [c for c in sector_cols if c in df.columns]
    data = df.sort_values(date_col)
    if data.empty or not sector_cols:
        return go.Figure()

    x = pd.to_datetime(data[date_col]).reset_index(drop=True)
    values = data[sector_cols].astype(float).reset_index(drop=True)
    if show_cumulative:
        values = (1 + values.fillna(0)).cumprod() - 1
    n = len(x)
    ends = np.unique(np.linspace(1, n, num=min(max_frames, n)).round().astype(int))

    def color(i, sector):
        return SECTOR_COLORS.get(sector, CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)])

    def traces(k):
        return [go.Scatter(x=x[:k], y=values[s][:k], mode="lines", name=s,
                           line=dict(width=2, color=color(i, s)))
                for i, s in enumerate(sector_cols)]

    fig = go.Figure(data=traces(int(ends[0])))
    names = [x[k - 1].strftime("%Y-%m-%d") for k in ends]
    fig.frames = [go.Frame(data=traces(int(k)), name=nm) for k, nm in zip(ends, names)]

    lo, hi = float(np.nanmin(values.to_numpy())), float(np.nanmax(values.to_numpy()))
    pad = 0.05 * (hi - lo or 1.0)
    fig.update_layout(
        title=title,
        height=height,
        yaxis_title="Cumulative return" if show_cumulative else "Value",
        xaxis=dict(range=[x.iloc[0], x.iloc[-1]]),
        yaxis=dict(range=[lo - pad, hi + pad], tickformat=".0%" if show_cumulative else None),
        margin=dict(b=90),
        updatemenus=_play_controls(speed),
        sliders=_slider(names, [x[k - 1].strftime("%b %Y") for k in ends], speed),
    )

    if theme:
        from fedviz.themes import get_theme
        fig = get_theme(theme).apply(fig)

    return fig


def animated_heatmap_evolution(
    df: pd.DataFrame,
    date_col: str = "date",
    value_col: str = "value",
    category_col: str = "category",
    title: str = "Heatmap Evolution",
    height: int = 500,
    theme: str | None = None,
    speed: int = 200,
) -> go.Figure:
    """Animated heatmap showing how values evolve over time."""
    dates = sorted(df[date_col].unique())
    categories = sorted(df[category_col].unique())

    if not dates or not categories:
        return go.Figure()

    fig = go.Figure()

    first_date = dates[0]
    first_data = df[df[date_col] == first_date]
    pivot = first_data.pivot_table(index=category_col, columns=date_col,
                                   values=value_col, aggfunc="mean")

    fig.add_trace(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[[0, "#b2384d"], [0.125, "#d4626f"], [0.25, "#ea9a9f"], [0.375, "#f4cfd0"], [0.5, "#eef1f5"], [0.625, "#c7dbf3"], [0.75, "#8fb6ea"], [0.875, "#5b8def"], [1, "#2f5fb3"]],
        zmid=0,
    ))

    frames = []
    for date in dates:
        sub = df[df[date_col] <= date]
        pivot = sub.pivot_table(index=category_col, columns=date_col,
                                values=value_col, aggfunc="mean")
        frames.append(go.Frame(
            data=[go.Heatmap(
                z=pivot.values,
                x=pivot.columns.tolist(),
                y=pivot.index.tolist(),
                colorscale=[[0, "#b2384d"], [0.125, "#d4626f"], [0.25, "#ea9a9f"], [0.375, "#f4cfd0"], [0.5, "#eef1f5"], [0.625, "#c7dbf3"], [0.75, "#8fb6ea"], [0.875, "#5b8def"], [1, "#2f5fb3"]],
                zmid=0,
            )],
            name=str(date),
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
                        "frame": {"duration": speed, "redraw": True},
                        "fromcurrent": True,
                    }],
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                    }],
                },
            ],
        }],
    )

    if theme:
        from fedviz.themes import get_theme
        fig = get_theme(theme).apply(fig)

    return fig


def animated_bubble_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    size: str,
    color: str,
    frame_col: str,
    text_col: str | None = None,
    title: str = "",
    height: int = 600,
    theme: str | None = None,
    speed: int = 200,
) -> go.Figure:
    """Animated bubble chart showing evolution of x, y, size, and color."""
    frames_data = sorted(df[frame_col].unique())
    if not frames_data:
        return go.Figure()

    fig = go.Figure()

    first = df[df[frame_col] == frames_data[0]]
    fig.add_trace(go.Scatter(
        x=first[x], y=first[y],
        mode="markers",
        marker=dict(
            size=first[size],
            color=first[color],
            colorscale=[[0, "#1b2f55"], [0.35, "#2f5fb3"], [0.7, "#5b8def"], [1, "#bcd3fa"]],
            showscale=True,
            sizemode="diameter",
            sizeref=2.0,
        ),
        text=first[text_col] if text_col else None,
        hovertemplate=(
            f"%{{text}}<br>{x}: %{{x:.2f}}<br>{y}: %{{y:.2f}}<br>"
            f"{size}: %{{marker.size:.0f}}<extra></extra>"
        ) if text_col else None,
    ))

    frames = []
    for frame_val in frames_data:
        sub = df[df[frame_col] == frame_val]
        frames.append(go.Frame(
            data=[go.Scatter(
                x=sub[x], y=sub[y],
                mode="markers",
                marker=dict(
                    size=sub[size],
                    color=sub[color],
                    colorscale=[[0, "#1b2f55"], [0.35, "#2f5fb3"], [0.7, "#5b8def"], [1, "#bcd3fa"]],
                    showscale=True,
                    sizemode="diameter",
                    sizeref=2.0,
                ),
                text=sub[text_col] if text_col else None,
            )],
            name=str(frame_val),
        ))

    fig.frames = frames

    fig.update_layout(
        title=title,
        height=height,
        xaxis_title=x,
        yaxis_title=y,
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {
                        "frame": {"duration": speed, "redraw": True},
                        "fromcurrent": True,
                    }],
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                    }],
                },
            ],
        }],
    )

    if theme:
        from fedviz.themes import get_theme
        fig = get_theme(theme).apply(fig)

    return fig


def meeting_by_meeting_animation(
    event_panel: pd.DataFrame,
    sector_cols: list[str],
    title: str = "Market Evolution Meeting by Meeting",
    height: int = 600,
    theme: str | None = None,
    speed: int = 500,
) -> go.Figure:
    """Animate market evolution from one FOMC meeting to the next."""
    if "rel_day" not in event_panel.columns:
        return go.Figure()

    meetings = sorted(event_panel["announcement_date"].unique())
    if not meetings:
        return go.Figure()

    from fedviz.themes import CATEGORICAL_PALETTE, SECTOR_COLORS

    cols = [c for c in sector_cols if c in event_panel.columns]
    if not cols:
        return go.Figure()
    panel = event_panel.sort_values(["announcement_date", "rel_day"])
    values = panel[cols].to_numpy(dtype=float)
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    pad = 0.08 * (hi - lo or 1.0)

    def traces(meeting):
        sub = panel[panel["announcement_date"] == meeting]
        return [go.Scatter(x=sub["rel_day"], y=sub[c], mode="lines+markers", name=c,
                           line=dict(width=2, color=SECTOR_COLORS.get(c, CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)])),
                           marker=dict(size=5))
                for i, c in enumerate(cols)]

    names = [pd.Timestamp(m).strftime("%Y-%m-%d") for m in meetings]
    fig = go.Figure(data=traces(meetings[0]))
    fig.frames = [go.Frame(data=traces(m), name=nm) for m, nm in zip(meetings, names)]
    fig.add_vline(x=0, line=dict(color="rgba(242,165,65,0.6)", width=1, dash="dot"))
    fig.update_layout(
        title=title,
        height=height,
        xaxis_title="Trading days from the announcement",
        yaxis_title="Daily return",
        yaxis=dict(range=[lo - pad, hi + pad], tickformat=".1%"),
        xaxis=dict(dtick=1),
        margin=dict(b=90),
        updatemenus=_play_controls(speed),
        sliders=_slider(names, [pd.Timestamp(m).strftime("%b %Y") for m in meetings], speed),
    )

    if theme:
        from fedviz.themes import get_theme
        fig = get_theme(theme).apply(fig)

    return fig
