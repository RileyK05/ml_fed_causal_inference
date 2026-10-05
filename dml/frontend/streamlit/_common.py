"""Shared helpers for the viewer: import path, cached data access, chart theme."""
from __future__ import annotations

import sys
from pathlib import Path

_DML = Path(__file__).resolve().parents[2]          # dml/
for _pkg_root in (_DML, _DML.parent / "core"):       # works even if not pip-installed
    if str(_pkg_root) not in sys.path:
        sys.path.insert(0, str(_pkg_root))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from fedcore import data  # noqa: E402

# Reference palette (validated): categorical slots in fixed order, light and dark steps.
CATEGORICAL = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "dark":  ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
DIVERGING = {"light": ("#2a78d6", "#e34948"), "dark": ("#3987e5", "#e66767")}  # (positive, negative)
INK = {
    "light": dict(primary="#0b0b0b", secondary="#52514e", grid="#e6e5e1", axis="#c9c8c3"),
    "dark":  dict(primary="#ffffff", secondary="#c3c2b7", grid="#2e2e2c", axis="#4a4a47"),
}
LAYER_ORDER = ["raw", "interim", "processed"]


def mode() -> str:
    """Chart palette follows the app theme pinned in .streamlit/config.toml ([theme] base)."""
    return "dark" if st.get_option("theme.base") == "dark" else "light"


def series_colors() -> list[str]:
    return CATEGORICAL[mode()]


def style(fig: go.Figure, height: int = 360, legend: bool = True, left: int = 72) -> go.Figure:
    """Apply the project chart style: recessive grid, 2px lines, text in ink not series color."""
    ink = INK[mode()]
    fig.update_layout(
        height=height, margin=dict(l=left, r=16, t=40, b=56),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=ink["secondary"], size=12), title_font=dict(color=ink["primary"], size=14),
        colorway=series_colors(), showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
        hovermode="x unified", hoverlabel=dict(font_size=12),
    )
    fig.update_xaxes(gridcolor=ink["grid"], linecolor=ink["axis"], zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor=ink["grid"], linecolor=ink["axis"], zerolinecolor=ink["axis"], automargin=True)
    fig.update_traces(selector=dict(type="scatter"), line=dict(width=2))
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", theme=None)


# ---- cached data access (cache clears when a file changes via the fingerprint) ----

@st.cache_data(show_spinner=False)
def _load(name: str, fp: str) -> pd.DataFrame:
    return data.load(name)


def _stamp(name: str) -> str:
    """Cheap change detector (size + mtime per file). The content hash is for run records only."""
    return "|".join(f"{p.name}:{p.stat().st_size}:{p.stat().st_mtime_ns}" for p in data.get(name).parts)


def load(name: str) -> pd.DataFrame:
    return _load(name, _stamp(name))


@st.cache_data(show_spinner=False)
def _coverage(fps: tuple) -> pd.DataFrame:
    return data.coverage()


def coverage() -> pd.DataFrame:
    return _coverage(tuple(_stamp(n) for n, d in data.catalog().items() if d.exists))


def page(title: str, icon: str = ":material/insights:") -> None:
    st.set_page_config(page_title=f"{title} - FOMC causal inference", page_icon=icon, layout="wide")
