"""Color palettes, chart templates, and styling for fedviz.

One design system for the app and its charts. Streamlit's own light and dark themes live in
``.streamlit/config.toml``; the Plotly template here follows whichever mode the viewer has
(``current_mode``), with transparent backgrounds so charts sit on the page's own surface.
"""

from __future__ import annotations

from dataclasses import dataclass

import plotly.graph_objects as go
import plotly.io as pio

FONT_FAMILY = "Inter, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"
MONO_FAMILY = "JetBrains Mono, ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

# Readable on both light and dark backgrounds; ordered so neighbours contrast.
CATEGORICAL_PALETTE = [
    "#5b8def",  # blue
    "#f2a541",  # amber
    "#34b38a",  # green
    "#e5596b",  # red
    "#9b7be8",  # violet
    "#3fb6d3",  # cyan
    "#e57ab5",  # pink
    "#9cc54b",  # lime
    "#f08a4b",  # orange
    "#8a94a8",  # slate
    "#2f5fb3",  # navy
    "#c9893a",  # bronze
    "#23806a",  # teal
    "#b2384d",  # crimson
    "#6a51b8",  # indigo
]

SECTOR_COLORS = {
    "SPY": "#8a94a8",   # the market: neutral slate
    "XLF": "#5b8def",   # financials
    "XLK": "#9b7be8",   # technology
    "XLE": "#f08a4b",   # energy
    "XLI": "#3fb6d3",   # industrials
    "XLP": "#9cc54b",   # staples
    "XLV": "#34b38a",   # health care
    "XLB": "#c9893a",   # materials
    "XLRE": "#e57ab5",  # real estate
    "XLU": "#f2a541",   # utilities
    "XLY": "#e5596b",   # discretionary
    "XLC": "#2f5fb3",   # communication
}

SEQUENTIAL_PALETTE = [
    "#0b1f3a", "#12325e", "#1b4a85", "#2763aa", "#3b7fcb",
    "#5b9be0", "#82b6ec", "#abcff4", "#d2e6fa", "#eef6fd",
][::-1]

DIVERGING_PALETTE = [
    "#b2384d", "#d4626f", "#ea9a9f", "#f4cfd0", "#eef1f5",
    "#c7dbf3", "#8fb6ea", "#5b8def", "#2f5fb3",
]

# Negative red -> neutral -> positive blue; the midpoint is a soft grey, not pure white,
# so centred heatmaps don't glare in dark mode.
DIVERGING_PALETTE_CENTERED = DIVERGING_PALETTE

HEATMAP_COLORS = {
    "positive": "#2f5fb3",
    "negative": "#b2384d",
    "neutral": "#eef1f5",
}

# Semantic colours that work in both modes.
POSITIVE = "#34b38a"
NEGATIVE = "#e5596b"
WARNING = "#f2a541"
ACCENT = "#5b8def"
REFERENCE = "#8a94a8"          # zero lines, 45-degree lines, fitted references
EVENT_COLOR = "rgba(242, 165, 65, 0.28)"  # FOMC meeting markers


@dataclass
class Theme:
    """Chart theme with colors, fonts, and layout defaults."""

    name: str = "dark"
    bg_color: str = "#0d1117"
    plot_bg_color: str = "rgba(0,0,0,0)"
    paper_bg_color: str = "rgba(0,0,0,0)"
    surface_color: str = "#151b26"
    font_color: str = "#d5dae3"
    muted_color: str = "#8b94a7"
    font_family: str = FONT_FAMILY
    grid_color: str = "rgba(139,148,167,0.14)"
    line_color: str = "rgba(139,148,167,0.35)"
    accent_color: str = ACCENT
    secondary_accent: str = "#9b7be8"
    positive_color: str = POSITIVE
    negative_color: str = NEGATIVE
    warning_color: str = WARNING
    info_color: str = ACCENT

    @classmethod
    def dark(cls) -> "Theme":
        return cls()

    @classmethod
    def light(cls) -> "Theme":
        return cls(
            name="light",
            bg_color="#f7f8fb",
            surface_color="#ffffff",
            font_color="#1f2633",
            muted_color="#5d6678",
            grid_color="rgba(93,102,120,0.13)",
            line_color="rgba(93,102,120,0.35)",
            accent_color="#3a6fd8",
            secondary_accent="#7a5bd6",
        )

    def _axis(self) -> dict:
        return {
            "gridcolor": self.grid_color,
            "linecolor": self.line_color,
            "zerolinecolor": self.line_color,
            "zerolinewidth": 1,
            "showline": False,
            "ticks": "",
            "tickfont": {"color": self.muted_color, "size": 11},
            "title": {"font": {"color": self.muted_color, "size": 12}, "standoff": 8},
            "automargin": True,
        }

    def to_plotly_layout(self, **kwargs) -> dict:
        """Convert theme to Plotly layout dict."""
        layout = {
            "paper_bgcolor": self.paper_bg_color,
            "plot_bgcolor": self.plot_bg_color,
            "font": {"color": self.font_color, "family": self.font_family, "size": 12},
            "title": {
                "font": {"size": 15, "color": self.font_color, "family": self.font_family},
                "x": 0.005, "xanchor": "left", "xref": "container", "y": 0.98, "yanchor": "top",
                "pad": {"b": 12},
            },
            "xaxis": self._axis(),
            "yaxis": self._axis(),
            "legend": {
                "orientation": "h", "x": 0, "xanchor": "left", "y": 1.0, "yanchor": "bottom",
                "bgcolor": "rgba(0,0,0,0)", "font": {"size": 11, "color": self.muted_color},
                "title": {"text": ""},
            },
            "margin": {"l": 8, "r": 8, "t": 64, "b": 8},
            "hoverlabel": {
                "bgcolor": self.surface_color,
                "bordercolor": self.line_color,
                "font": {"color": self.font_color, "family": self.font_family, "size": 12},
            },
            "colorway": CATEGORICAL_PALETTE,
            "coloraxis": {"colorbar": {"outlinewidth": 0, "thickness": 12,
                                       "tickfont": {"color": self.muted_color, "size": 10}}},
            "bargap": 0.25,
        }
        layout.update(kwargs)
        return layout

    def template(self) -> go.layout.Template:
        """A full Plotly template: layout defaults plus per-trace styling."""
        tpl = go.layout.Template(layout=self.to_plotly_layout())
        tpl.data.scatter = [go.Scatter(line={"width": 2})]
        tpl.data.bar = [go.Bar(marker={"line": {"width": 0}})]
        tpl.data.histogram = [go.Histogram(marker={"line": {"width": 0}})]
        tpl.data.heatmap = [go.Heatmap(colorbar={"outlinewidth": 0, "thickness": 12})]
        return tpl

    def apply(self, fig: go.Figure) -> go.Figure:
        """Apply theme to a Plotly figure."""
        fig.update_layout(template=self.template())
        return fig


def get_theme(name: str | None = "dark") -> Theme:
    """Get a theme by name; ``None`` means the viewer's current mode."""
    name = current_mode() if name is None else name
    return Theme.light() if name == "light" else Theme.dark()


def current_mode(default: str = "dark") -> str:
    """'light' or 'dark', following the viewer's Streamlit theme when it is known."""
    try:
        import streamlit as st

        mode = st.context.theme.type
    except Exception:
        mode = None
    return mode if mode in ("light", "dark") else default


def apply_theme(fig: go.Figure, theme_name: str | None = None) -> go.Figure:
    """Apply a named theme (default: the viewer's mode) to a figure."""
    return get_theme(theme_name).apply(fig)


def set_default_template(theme_name: str | None = None) -> None:
    """Make the fedviz template (for the viewer's mode by default) Plotly's default."""
    pio.templates["fedviz"] = get_theme(theme_name).template()
    pio.templates.default = "fedviz"
