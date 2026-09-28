"""Static presentation figures: seaborn/matplotlib renders exported as PNG or SVG.

The interactive plotly charts stay in the UI; these are the professional static versions
for slides (docs/methods.md Q1 reporting). Same data as the API responses (records) and
the same palette as fedci.api.figures. Nothing is written to disk -- the API returns
bytes; playground results remain ephemeral.
"""
from __future__ import annotations

import io
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

from fedci.api.figures import CATEGORICAL, DIVERGING, INK

EXPORT_KINDS = ("residualized_fit", "influence", "coefficients")
ExportFormat = Literal["png", "svg"]


def _theme() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.frameon": False,
    })


def _num(v: float, digits: int = 3) -> str:
    return f"{v:.{digits}f}".replace("-", "\u2212")


def _estimator_label(name: str) -> str:
    pretty = {"ols y~s": "OLS  y ~ s", "ols y~s+X": "OLS  y ~ s + X"}
    if name in pretty:
        return pretty[name]
    return name.replace("plr_dml", "PLR-DML").replace(" (bootstrap)", " (bootstrap)").replace("_", " ")


def _residualized_fit(metrics: dict, residuals: list[dict]) -> plt.Figure:
    _theme()
    x = [float(r["s_resid"]) for r in residuals]
    y = [float(r["y_resid"]) for r in residuals]
    theta = float(metrics["theta"])
    se = metrics.get("se")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    sns.scatterplot(x=x, y=y, ax=ax, color=CATEGORICAL["light"][0], s=56,
                    edgecolor="white", linewidth=0.6)
    grid = np.linspace(min(x), max(x), 100)
    ax.plot(grid, theta * grid, color=CATEGORICAL["light"][1], linewidth=2.2,
            label=f"PLR-DML slope: \u03b8 = {_num(theta)}"
                  + (f"  (se {_num(float(se))})" if se is not None else ""))
    ax.axhline(0, color=INK["light"]["axis"], linewidth=0.8, zorder=0)
    ax.axvline(0, color=INK["light"]["axis"], linewidth=0.8, zorder=0)
    ax.set_xlabel("s residual (per 10bp surprise)")
    ax.set_ylabel("y residual (percent)")
    ax.set_title(f"Q1 residualized fit  (n = {metrics['n_meetings']} meetings)")
    ax.legend(loc="best")
    sns.despine(ax=ax)
    return fig


def _influence(influence: list[dict]) -> plt.Figure:
    _theme()
    rows = list(influence)[::-1]
    labels = [str(r["announcement_date"])[:10] for r in rows]
    deltas = [float(r["delta_loo"]) for r in rows]
    pos, neg = DIVERGING["light"]
    fig, ax = plt.subplots(figsize=(7.2, max(3.2, 0.34 * len(rows) + 1.2)))
    ax.barh(labels, deltas, color=[pos if d >= 0 else neg for d in deltas], height=0.62)
    ax.axvline(0, color=INK["light"]["axis"], linewidth=0.8)
    ax.set_xlabel("\u03b8 (drop meeting) \u2212 \u03b8 (all meetings)")
    ax.set_title("Leave-one-meeting-out sensitivity")
    sns.despine(ax=ax)
    return fig


def _coefficients(estimates: list[dict], level: float) -> plt.Figure:
    _theme()
    rows = list(estimates)[::-1]
    names = [_estimator_label(str(r["estimator"])) for r in rows]
    theta = [float(r["theta"]) for r in rows]
    lows = [float(r["ci_low"]) for r in rows]
    highs = [float(r["ci_high"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, max(3.2, 0.7 * len(rows) + 1.4)))
    ypos = np.arange(len(rows))
    xerr = [[t - lo for t, lo in zip(theta, lows)], [hi - t for t, hi in zip(theta, highs)]]
    ax.errorbar(theta, ypos, xerr=xerr, fmt="o", color=CATEGORICAL["light"][0],
                ecolor=INK["light"]["secondary"], elinewidth=2, capsize=5, markersize=8)
    for t, y in zip(theta, ypos):
        ax.annotate(_num(t), (t, y), textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=9, color=INK["light"]["secondary"])
    ax.axvline(0, color=INK["light"]["axis"], linewidth=1, linestyle="--")
    ax.set_yticks(ypos)
    ax.set_yticklabels(names)
    ax.set_xlabel("theta: percent return per 10bp surprise".replace("theta", "\u03b8"))
    ax.set_title(f"Q1 estimates with {level:.0%} CI")
    sns.despine(ax=ax)
    return fig


def render_export(kind: str, *, metrics: dict, estimates: list[dict], residuals: list[dict],
                  influence: list[dict], fmt: ExportFormat = "png", level: float = 0.95,
                  dpi: int = 200) -> bytes:
    """Render one export figure and return its bytes (PNG or SVG)."""
    if kind not in EXPORT_KINDS:
        raise ValueError(f"figure must be one of {EXPORT_KINDS}, got {kind!r}")
    if fmt not in ("png", "svg"):
        raise ValueError(f"format must be 'png' or 'svg', got {fmt!r}")
    if kind == "residualized_fit":
        fig = _residualized_fit(metrics, residuals)
    elif kind == "influence":
        fig = _influence(influence)
    else:
        fig = _coefficients(estimates, level)
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


__all__ = ["EXPORT_KINDS", "ExportFormat", "render_export"]
