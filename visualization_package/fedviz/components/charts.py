"""Reusable Plotly chart builders for fedviz.

All chart functions return plotly.graph_objects.Figure and accept a
theme parameter for consistent styling. These are model-agnostic —
any model can use them to visualize its outputs.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fedviz.themes import (
    ACCENT,
    CATEGORICAL_PALETTE,
    DIVERGING_PALETTE_CENTERED,
    EVENT_COLOR,
    NEGATIVE,
    POSITIVE,
    REFERENCE,
    SECTOR_COLORS,
    WARNING,
    Theme,
    get_theme,
)

DIVERGING_SCALE = [[i / (len(DIVERGING_PALETTE_CENTERED) - 1), c]
                   for i, c in enumerate(DIVERGING_PALETTE_CENTERED)]


def _series_color(name, i: int) -> str:
    """Sector tickers keep their own colour everywhere; anything else uses the palette."""
    return SECTOR_COLORS.get(str(name), CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)])


def _apply_theme(fig: go.Figure, theme: str | Theme | None = None) -> go.Figure:
    if theme is None:
        return fig
    if isinstance(theme, str):
        theme = get_theme(theme)
    return theme.apply(fig)


def time_series(
    df: pd.DataFrame,
    x: str = "date",
    y: str | list[str] | None = None,
    color: str | None = None,
    title: str = "",
    yaxis_title: str = "",
    xaxis_title: str = "",
    height: int = 400,
    theme: str | Theme | None = None,
    show_legend: bool = True,
    line_width: int = 2,
    opacity: float = 1.0,
) -> go.Figure:
    """Multi-line time series chart."""
    fig = go.Figure()
    if y is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        y = [c for c in numeric_cols if c != x and c != color]
    elif isinstance(y, str):
        y = [y]

    if color:
        groups = df[color].unique()
        for i, grp in enumerate(groups):
            mask = df[color] == grp
            sub = df[mask]
            for col in y:
                fig.add_trace(go.Scatter(
                    x=sub[x], y=sub[col],
                    mode="lines",
                    name=f"{col} ({grp})" if len(y) > 1 else str(grp),
                    line=dict(
                        color=CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)],
                        width=line_width,
                    ),
                    opacity=opacity,
                ))
    else:
        for i, col in enumerate(y):
            fig.add_trace(go.Scatter(
                x=df[x], y=df[col],
                mode="lines",
                name=col,
                line=dict(
                    color=_series_color(col, i),
                    width=line_width,
                ),
                opacity=opacity,
            ))

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=height,
        showlegend=show_legend,
        hovermode="x unified",
    )
    return _apply_theme(fig, theme)


def time_series_with_events(
    df: pd.DataFrame,
    x: str = "date",
    y: str = "value",
    events: pd.DataFrame | None = None,
    event_date_col: str = "announcement_date",
    event_label_col: str | None = None,
    title: str = "",
    yaxis_title: str = "",
    height: int = 450,
    theme: str | Theme | None = None,
    event_line_color: str | None = None,
    event_line_dash: str = "solid",
    band: tuple[float, float] | None = None,
) -> go.Figure:
    """Time series with vertical event markers and optional band."""
    fig = time_series(df, x=x, y=y, title=title, yaxis_title=yaxis_title,
                      height=height, theme=theme)

    if band:
        fig.add_hrect(y0=band[0], y1=band[1], fillcolor=REFERENCE, opacity=0.1,
                      line_width=0, annotation_text="Band")

    if events is not None and len(events) > 0 and event_date_col in events.columns:
        dates = pd.to_datetime(events[event_date_col]).dropna()
        if x in df.columns and len(df):
            xs = pd.to_datetime(df[x], errors="coerce")
            dates = dates[(dates >= xs.min()) & (dates <= xs.max())]
        shapes = list(fig.layout.shapes or ()) + [
            dict(type="line", xref="x", yref="paper", x0=d, x1=d, y0=0, y1=1, layer="below",
                 line=dict(color=event_line_color or EVENT_COLOR, width=1))
            for d in dates
        ]
        fig.update_layout(shapes=shapes)
        if event_label_col:
            for d, label in zip(dates, events.loc[dates.index, event_label_col]):
                fig.add_annotation(x=d, y=1, yref="paper", text=str(label), showarrow=False,
                                   font=dict(size=9, color=WARNING), yanchor="bottom")
        if len(dates):
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="FOMC meeting   ",  # room for the web font
                                     line=dict(color=WARNING, width=1)))
    return fig


def heatmap(
    df: pd.DataFrame,
    x: str | None = None,
    y: str | None = None,
    z: str | None = None,
    title: str = "",
    colorscale: str | list | None = None,
    height: int = 500,
    theme: str | Theme | None = None,
    text_fmt: str = ".2f",
    center_zero: bool = True,
) -> go.Figure:
    """Annotated heatmap."""
    if x and y and z:
        pivot = df.pivot_table(index=y, columns=x, values=z, aggfunc="mean")
    else:
        pivot = df.select_dtypes(include=[np.number])

    z_values = pivot.values
    z_mid = 0 if center_zero and np.nanmin(z_values) < 0 < np.nanmax(z_values) else None

    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=colorscale or DIVERGING_SCALE,
        zmid=z_mid,
        text=np.vectorize(lambda v: "" if pd.isna(v) else f"{v:{text_fmt}}")(z_values),
        texttemplate="%{text}" if z_values.shape[1] <= 24 else None,  # crowded grids: hover only
        textfont={"size": 10},
        hoverongaps=False,
    ))
    fig.update_layout(title=title, height=height)
    return _apply_theme(fig, theme)


def scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
    size: str | None = None,
    text: str | None = None,
    title: str = "",
    xaxis_title: str = "",
    yaxis_title: str = "",
    height: int = 450,
    theme: str | Theme | None = None,
    add_regression: bool = False,
    add_trendline: bool = False,
    opacity: float = 0.7,
    symbol: str | None = None,
) -> go.Figure:
    """Scatter plot with optional regression line and color/size encoding."""
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
                marker=dict(
                    color=CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)],
                    size=sub[size] if size else 8,
                    opacity=opacity,
                    symbol=symbol if symbol else "circle",
                    line=dict(width=0.5, color="rgba(255,255,255,0.45)"),
                ),
                text=sub[text] if text else None,
                hovertemplate=(
                    f"%{{text}}<br>{x}: %{{x:.4f}}<br>{y}: %{{y:.4f}}<extra></extra>"
                    if text else f"{x}: %{{x:.4f}}<br>{y}: %{{y:.4f}}<extra></extra>"
                ),
            ))
    else:
        fig.add_trace(go.Scatter(
            x=df[x], y=df[y],
            mode="markers", name=y, showlegend=False,
            marker=dict(
                color=CATEGORICAL_PALETTE[0],
                size=df[size] if size else 8,
                opacity=opacity,
                symbol=symbol if symbol else "circle",
                line=dict(width=0.5, color="rgba(255,255,255,0.45)"),
            ),
            text=df[text] if text else None,
        ))

    if add_regression or add_trendline:
        from scipy import stats
        mask = df[[x, y]].dropna()
        if len(mask) > 2:
            slope, intercept, r, p, se = stats.linregress(mask[x], mask[y])
            x_range = np.linspace(mask[x].min(), mask[x].max(), 100)
            fig.add_trace(go.Scatter(
                x=x_range, y=slope * x_range + intercept,
                mode="lines", name=f"Fit (R²={r**2:.3f})",
                line=dict(color=WARNING, dash="dash", width=2),
            ))

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title or x,
        yaxis_title=yaxis_title or y,
        height=height,
        hovermode="closest",
    )
    return _apply_theme(fig, theme)


def distribution(
    df: pd.DataFrame,
    column: str,
    group: str | None = None,
    title: str = "",
    xaxis_title: str = "",
    height: int = 400,
    theme: str | Theme | None = None,
    nbins: int = 50,
    show_kde: bool = True,
    show_rug: bool = False,
) -> go.Figure:
    """Histogram with optional KDE overlay and grouping."""
    fig = go.Figure()

    if group:
        groups = df[group].unique()
        for i, grp in enumerate(groups):
            sub = df[df[group] == grp][column].dropna()
            fig.add_trace(go.Histogram(
                x=sub, name=str(grp),
                nbinsx=nbins, histnorm="probability density",
                opacity=0.6,
                marker_color=CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)],
            ))
            if show_kde and len(sub) > 1:
                from scipy import stats
                kde = stats.gaussian_kde(sub)
                x_range = np.linspace(sub.min(), sub.max(), 200)
                fig.add_trace(go.Scatter(
                    x=x_range, y=kde(x_range),
                    mode="lines", name=f"{grp} KDE",
                    line=dict(color=CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)], width=2),
                ))
    else:
        data = df[column].dropna()
        fig.add_trace(go.Histogram(
            x=data, nbinsx=nbins, histnorm="probability density", name=column,
            marker_color=CATEGORICAL_PALETTE[0], opacity=0.75,
        ))
        if show_kde and len(data) > 1:
            from scipy import stats
            kde = stats.gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            fig.add_trace(go.Scatter(
                x=x_range, y=kde(x_range),
                mode="lines", name="KDE",
                line=dict(color=WARNING, width=2),
            ))
        if show_rug:
            fig.add_trace(go.Scatter(
                x=data, y=np.zeros_like(data),
                mode="markers", name="Rug",
                marker=dict(symbol="line-ns", size=10, color=REFERENCE, opacity=0.4),
            ))

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title or column,
        yaxis_title="Density",
        height=height,
        barmode="overlay" if group else "relative",
    )
    return _apply_theme(fig, theme)


def box_plot(
    df: pd.DataFrame,
    y: str,
    x: str | None = None,
    color: str | None = None,
    title: str = "",
    height: int = 400,
    theme: str | Theme | None = None,
    show_points: bool = False,
    points_jitter: float = 0.3,
) -> go.Figure:
    """Box plot with optional grouping and point overlay."""
    fig = go.Figure()

    if x:
        groups = df[x].unique()
        for i, grp in enumerate(groups):
            sub = df[df[x] == grp][y].dropna()
            fig.add_trace(go.Box(
                y=sub, name=str(grp),
                marker_color=_series_color(grp, i),
                boxpoints="all" if show_points else False,
                jitter=points_jitter,
                pointpos=-1.8,
            ))
    else:
        fig.add_trace(go.Box(
            y=df[y].dropna(), name=y,
            marker_color=CATEGORICAL_PALETTE[0],
            boxpoints="all" if show_points else False,
            jitter=points_jitter,
        ))

    fig.update_layout(title=title, yaxis_title=y, height=height, showlegend=False)
    return _apply_theme(fig, theme)


def violin_plot(
    df: pd.DataFrame,
    y: str,
    x: str | None = None,
    title: str = "",
    height: int = 400,
    theme: str | Theme | None = None,
    show_box: bool = True,
    show_points: bool = False,
) -> go.Figure:
    """Violin plot with optional box overlay."""
    fig = go.Figure()

    if x:
        groups = df[x].unique()
        for i, grp in enumerate(groups):
            sub = df[df[x] == grp][y].dropna()
            fig.add_trace(go.Violin(
                y=sub, name=str(grp),
                box_visible=show_box,
                points="all" if show_points else False,
                marker_color=_series_color(grp, i),
                line_color=_series_color(grp, i),
            ))
    else:
        fig.add_trace(go.Violin(
            y=df[y].dropna(), name=y,
            box_visible=show_box,
            points="all" if show_points else False,
        ))

    fig.update_layout(title=title, yaxis_title=y, height=height, showlegend=False)
    return _apply_theme(fig, theme)


def bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
    orientation: str = "v",
    title: str = "",
    height: int = 400,
    theme: str | Theme | None = None,
    text_fmt: str = ".2f",
    color_scale: str | None = None,
    sort: bool = True,
) -> go.Figure:
    """Bar chart with optional color encoding and sorting."""
    if sort:
        df = df.sort_values(y, ascending=(orientation == "h"))

    fig = go.Figure()

    if color:
        groups = df[color].unique()
        for i, grp in enumerate(groups):
            sub = df[df[color] == grp]
            fig.add_trace(go.Bar(
                x=sub[x] if orientation == "v" else sub[y],
                y=sub[y] if orientation == "v" else sub[x],
                name=str(grp),
                orientation=orientation,
                marker_color=CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)],
                text=[f"{v:{text_fmt}}" for v in sub[y]],
                textposition="auto",
            ))
    else:
        colors = None
        if color_scale:
            vals = df[y].values
            vmin, vmax = np.nanmin(vals), np.nanmax(vals)
            norm = (vals - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(vals)
            colors = [POSITIVE if v >= 0 else NEGATIVE for v in vals] if (
                vmin < 0 < vmax or color_scale == "sign") else [
                f"rgba(91,141,239,{0.35 + 0.65 * n:.2f})" for n in norm]

        fig.add_trace(go.Bar(
            x=df[x] if orientation == "v" else df[y],
            y=df[y] if orientation == "v" else df[x],
            orientation=orientation,
            marker_color=colors or CATEGORICAL_PALETTE[0],
            text=[f"{v:{text_fmt}}" for v in df[y]],
            textposition="auto",
        ))

    n_bars = len(df) if not color else df[x].nunique()
    fig.update_layout(title=title, height=height, showlegend=color is not None,
                      bargap=0.55 if n_bars <= 6 else 0.25)
    return _apply_theme(fig, theme)


def cumulative_path(
    df: pd.DataFrame,
    x: str = "rel_day",
    y: str = "cum_return",
    group: str | None = None,
    title: str = "",
    height: int = 450,
    theme: str | Theme | None = None,
    show_mean: bool = True,
    show_median: bool = True,
    confidence_band: bool = False,
    band_alpha: float = 0.2,
    color: str | None = None,
) -> go.Figure:
    """Cumulative return paths in event time.

    With ``group`` (one path per meeting): each path is a faint line (hover shows which), the
    mean is drawn bold on top, and ``confidence_band`` shades the 25th-75th percentile.
    """
    fig = go.Figure()
    accent = color or ACCENT

    if group:
        for grp, sub in df.groupby(group, sort=True):
            label = pd.Timestamp(grp).strftime("%Y-%m-%d") if isinstance(grp, (pd.Timestamp, np.datetime64)) else str(grp)
            fig.add_trace(go.Scatter(
                x=sub[x], y=sub[y], mode="lines", name=label, showlegend=False,
                line=dict(color=REFERENCE, width=1), opacity=0.35,
                hovertemplate=f"{label}<br>day %{{x}}: %{{y:.2%}}<extra></extra>",
            ))
        grouped = df.groupby(x)[y]
        if confidence_band and len(df) > 1:
            lo, hi = grouped.quantile(0.25), grouped.quantile(0.75)
            fig.add_trace(go.Scatter(x=hi.index, y=hi.values, mode="lines", line_width=0,
                                     showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=lo.index, y=lo.values, mode="lines", line_width=0,
                                     fill="tonexty", fillcolor="rgba(91,141,239,0.16)",
                                     name="Middle 50%", hoverinfo="skip"))
        if show_mean:
            mean_path = grouped.mean()
            fig.add_trace(go.Scatter(
                x=mean_path.index, y=mean_path.values, mode="lines+markers", name="Mean",
                line=dict(color=accent, width=3), marker=dict(size=5),
                hovertemplate="mean, day %{x}: %{y:.2%}<extra></extra>",
            ))
    else:
        fig.add_trace(go.Scatter(
            x=df[x], y=df[y],
            mode="lines", name=y,
            line=dict(color=accent, width=2),
        ))
        if confidence_band and len(df) > 1:
            grouped = df.groupby(x)[y]
            mean = grouped.mean()
            std = grouped.std()
            fig.add_trace(go.Scatter(
                x=mean.index, y=mean + 1.96 * std,
                mode="lines", line_width=0, showlegend=False,
                hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=mean.index, y=mean - 1.96 * std,
                mode="lines", line_width=0, showlegend=False,
                fill="tonexty", fillcolor="rgba(91,141,239,0.18)",
                hoverinfo="skip",
            ))

    fig.add_hline(y=0, line=dict(color=REFERENCE, dash="dot", width=1))
    fig.add_vline(x=0, line=dict(color=EVENT_COLOR, width=1))
    fig.update_layout(
        title=title,
        xaxis_title="Trading days from the announcement",
        yaxis_title="Cumulative return",
        yaxis_tickformat=".1%",
        xaxis_dtick=1,
        height=height,
        hovermode="closest",
    )
    return _apply_theme(fig, theme)


def rolling_metric(
    series: pd.Series,
    window: int = 63,
    metric: str = "mean",
    title: str = "",
    height: int = 350,
    theme: str | Theme | None = None,
    show_raw: bool = True,
    raw_alpha: float = 0.3,
) -> go.Figure:
    """Rolling window metric chart."""
    fig = go.Figure()

    if show_raw:
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values,
            mode="lines", name="Raw",
            line=dict(color=REFERENCE, width=1),
            opacity=raw_alpha,
        ))

    roll = series.rolling(window=window, min_periods=max(1, window // 4))
    if metric == "mean":
        values = roll.mean()
    elif metric == "std":
        values = roll.std()
    elif metric == "min":
        values = roll.min()
    elif metric == "max":
        values = roll.max()
    elif metric == "median":
        values = roll.median()
    else:
        values = roll.mean()

    fig.add_trace(go.Scatter(
        x=values.index, y=values.values,
        mode="lines", name=f"{window}d {metric}",
        line=dict(color=CATEGORICAL_PALETTE[0], width=2),
    ))

    fig.update_layout(title=title, height=height, hovermode="x unified")
    return _apply_theme(fig, theme)


def drawdown_chart(
    returns: pd.Series,
    title: str = "Drawdown",
    height: int = 350,
    theme: str | Theme | None = None,
) -> go.Figure:
    """Drawdown chart from a return series."""
    cum = (1 + returns.fillna(0)).cumprod()
    running_max = cum.expanding().max()
    dd = cum / running_max - 1

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        mode="lines", fill="tozeroy",
        fillcolor="rgba(229,89,107,0.22)",
        line=dict(color=NEGATIVE, width=1.5),
        name="Drawdown",
    ))
    fig.update_layout(
        title=title,
        yaxis_title="Drawdown",
        yaxis_tickformat=".1%",
        height=height,
        hovermode="x unified",
    )
    return _apply_theme(fig, theme)


def candlestick(
    df: pd.DataFrame,
    date: str = "date",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    title: str = "",
    height: int = 500,
    theme: str | Theme | None = None,
    show_volume: bool = False,
    volume_col: str = "volume",
) -> go.Figure:
    """OHLC candlestick chart with optional volume subplot."""
    if show_volume:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.7, 0.3], vertical_spacing=0.05)
    else:
        fig = go.Figure()

    candle = go.Candlestick(
        x=df[date],
        open=df[open_col], high=df[high_col],
        low=df[low_col], close=df[close_col],
        increasing=dict(line=dict(color=POSITIVE), fillcolor=POSITIVE),
        decreasing=dict(line=dict(color=NEGATIVE), fillcolor=NEGATIVE),
        name="OHLC",
    )

    if show_volume:
        fig.add_trace(candle, row=1, col=1)
        colors = [POSITIVE if c >= o else NEGATIVE
                  for c, o in zip(df[close_col], df[open_col])]
        fig.add_trace(go.Bar(
            x=df[date], y=df[volume_col],
            marker_color=colors, name="Volume", opacity=0.5,
        ), row=2, col=1)
        fig.update_layout(title=title, height=height, xaxis_rangeslider_visible=False)
    else:
        fig.add_trace(candle)
        fig.update_layout(title=title, height=height, xaxis_rangeslider_visible=False)

    return _apply_theme(fig, theme)


def forest_plot(
    df: pd.DataFrame,
    estimate: str = "estimate",
    low: str = "ci_low",
    high: str = "ci_high",
    label: str = "name",
    title: str = "",
    height: int = 400,
    theme: str | Theme | None = None,
    show_zero: bool = True,
    color_by: str | None = None,
    sort: bool = True,
) -> go.Figure:
    """Forest plot with confidence intervals."""
    if sort:
        df = df.sort_values(estimate)

    fig = go.Figure()

    colors = None
    if color_by:
        groups = df[color_by].unique()
        color_map = {g: CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)]
                     for i, g in enumerate(groups)}
        colors = [color_map[g] for g in df[color_by]]

    for i, (_, row) in enumerate(df.iterrows()):
        color = colors[i] if colors else CATEGORICAL_PALETTE[0]
        fig.add_trace(go.Scatter(
            x=[row[low], row[high]],
            y=[i, i],
            mode="lines",
            line=dict(color=color, width=2),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=[row[estimate]],
            y=[i],
            mode="markers",
            marker=dict(color=color, size=10, symbol="diamond"),
            name=str(row[label]),
            hovertemplate=f"{row[label]}: {row[estimate]:.4f}<br>[{row[low]:.4f}, {row[high]:.4f}]<extra></extra>",
        ))

    if show_zero:
        fig.add_vline(x=0, line=dict(color=REFERENCE, dash="dash", width=1))

    fig.update_layout(
        title=title,
        yaxis=dict(tickmode="array", tickvals=list(range(len(df))),
                   ticktext=df[label].tolist()),
        height=max(200, len(df) * 30),
        xaxis_title="Estimate",
    )
    return _apply_theme(fig, theme)


def qq_plot(
    series: pd.Series,
    title: str = "Q-Q Plot",
    height: int = 400,
    theme: str | Theme | None = None,
    dist: str = "norm",
) -> go.Figure:
    """Q-Q plot against a theoretical distribution."""
    from scipy import stats

    data = series.dropna().values
    data_sorted = np.sort(data)
    n = len(data_sorted)
    theoretical = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=theoretical, y=data_sorted,
        mode="markers",
        marker=dict(color=CATEGORICAL_PALETTE[0], size=6, opacity=0.6),
        name="Data",
    ))

    slope, intercept, r, _, _ = stats.linregress(theoretical, data_sorted)
    x_range = np.linspace(theoretical.min(), theoretical.max(), 100)
    fig.add_trace(go.Scatter(
        x=x_range, y=slope * x_range + intercept,
        mode="lines", name=f"Reference (R²={r**2:.4f})",
        line=dict(color=WARNING, dash="dash", width=2),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Sample Quantiles",
        height=height,
    )
    return _apply_theme(fig, theme)


def residual_plot(
    residuals: pd.Series,
    fitted: pd.Series | None = None,
    title: str = "Residual Diagnostics",
    height: int = 600,
    theme: str | Theme | None = None,
) -> go.Figure:
    """Multi-panel residual diagnostics plot."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Residuals vs Fitted", "Q-Q Plot",
                        "Residual Distribution", "Residuals vs Order"),
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    if fitted is not None:
        fig.add_trace(go.Scatter(
            x=fitted, y=residuals,
            mode="markers",
            marker=dict(color=CATEGORICAL_PALETTE[0], size=5, opacity=0.5),
            name="Residuals",
        ), row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color=REFERENCE, row=1, col=1)

    from scipy import stats
    data = residuals.dropna().values
    data_sorted = np.sort(data)
    n = len(data_sorted)
    theoretical = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    fig.add_trace(go.Scatter(
        x=theoretical, y=data_sorted,
        mode="markers",
        marker=dict(color=CATEGORICAL_PALETTE[1], size=5, opacity=0.5),
        name="Q-Q",
    ), row=1, col=2)

    fig.add_trace(go.Histogram(
        x=data, nbinsx=50,
        marker_color=CATEGORICAL_PALETTE[2], opacity=0.7,
        name="Distribution",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=list(range(len(residuals))), y=residuals,
        mode="lines",
        line=dict(color=CATEGORICAL_PALETTE[3], width=1),
        name="Order",
    ), row=2, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color=REFERENCE, row=2, col=2)

    fig.update_layout(title=title, height=height, showlegend=False)
    return _apply_theme(fig, theme)


def paired_difference(
    df: pd.DataFrame,
    a: str,
    b: str,
    label: str | None = None,
    title: str = "Paired Difference",
    height: int = 450,
    theme: str | Theme | None = None,
    show_bland_altman: bool = True,
) -> go.Figure:
    """Paired difference plot with optional Bland-Altman overlay."""
    if show_bland_altman:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Paired Comparison", "Bland-Altman"),
            horizontal_spacing=0.12,
        )
    else:
        fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df[a], y=df[b],
        mode="markers",
        marker=dict(color=CATEGORICAL_PALETTE[0], size=8, opacity=0.6),
        text=df[label] if label else None,
        name="Pairs",
    ), row=1, col=1 if show_bland_altman else None)

    max_val = max(df[a].max(), df[b].max())
    min_val = min(df[a].min(), df[b].min())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines", line=dict(dash="dash", color=REFERENCE),
        name="y=x",
    ), row=1, col=1 if show_bland_altman else None)

    if show_bland_altman:
        mean = (df[a] + df[b]) / 2
        diff = df[a] - df[b]
        fig.add_trace(go.Scatter(
            x=mean, y=diff,
            mode="markers",
            marker=dict(color=CATEGORICAL_PALETTE[1], size=8, opacity=0.6),
            text=df[label] if label else None,
            name="Difference",
        ), row=1, col=2)
        mean_diff = diff.mean()
        std_diff = diff.std()
        fig.add_hline(y=mean_diff, line_dash="solid", line_color=ACCENT,
                      annotation_text=f"Mean: {mean_diff:.4f}", row=1, col=2)
        fig.add_hline(y=mean_diff + 1.96 * std_diff, line_dash="dash", line_color=NEGATIVE,
                      annotation_text=f"+1.96σ: {mean_diff + 1.96*std_diff:.4f}", row=1, col=2)
        fig.add_hline(y=mean_diff - 1.96 * std_diff, line_dash="dash", line_color=NEGATIVE,
                      annotation_text=f"-1.96σ: {mean_diff - 1.96*std_diff:.4f}", row=1, col=2)

    fig.update_layout(title=title, height=height, showlegend=False)
    return _apply_theme(fig, theme)
