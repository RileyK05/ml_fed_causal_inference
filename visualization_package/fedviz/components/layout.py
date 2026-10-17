"""Layout helpers for fedviz pages."""

from __future__ import annotations

import html
from typing import Any

import streamlit as st

from fedviz.themes import ACCENT, get_theme

PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d", "toggleSpikelines"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}

_CSS = f"""
<style>
.block-container {{ padding-top: 2.4rem; padding-bottom: 4rem; max-width: 1480px; }}
h1, h2, h3 {{ letter-spacing: -0.015em; }}
h3 {{ font-size: 1.12rem !important; margin-top: 0.6rem; }}

.fv-header {{ margin: 0 0 1.4rem 0; }}
.fv-eyebrow {{ font-size: 0.72rem; font-weight: 600; letter-spacing: 0.09em;
              text-transform: uppercase; color: {ACCENT}; margin-bottom: 0.35rem; }}
.fv-title {{ font-size: 2.05rem; font-weight: 700; letter-spacing: -0.025em;
            line-height: 1.15; margin: 0; }}
.fv-subtitle {{ font-size: 0.98rem; opacity: 0.66; max-width: 78ch; margin-top: 0.45rem;
               line-height: 1.5; }}

[data-testid="stMetric"] {{ padding: 0.9rem 1.05rem 0.85rem 1.05rem; }}
[data-testid="stMetricLabel"] p {{ font-size: 0.72rem !important; font-weight: 600;
                                  letter-spacing: 0.06em; text-transform: uppercase; opacity: 0.7; }}
[data-testid="stMetricValue"] {{ font-size: 1.6rem; font-weight: 650;
                                 font-variant-numeric: tabular-nums; }}

.stTabs [data-baseweb="tab-list"] {{ gap: 1.4rem; }}
.stTabs [data-baseweb="tab"] {{ padding-left: 0; padding-right: 0; font-weight: 500; }}
.stTabs [data-baseweb="tab-panel"] {{ padding-top: 1.1rem; }}

[data-testid="stExpander"] summary p {{ font-weight: 600; }}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p strong {{
    font-size: 0.72rem; letter-spacing: 0.07em; text-transform: uppercase; opacity: 0.72; }}
[data-testid="stSidebarNavSeparator"] {{ margin: 0.4rem 0; }}

[data-testid="stDataFrame"] {{ border-radius: 0.6rem; overflow: hidden; }}
.fv-footer {{ font-size: 0.72rem; opacity: 0.5; margin-top: 0.6rem; }}
</style>
"""


def inject_css() -> None:
    """Add the fedviz stylesheet to the page (call once per rerun, before content)."""
    st.html(_CSS)


def page_header(
    title: str,
    subtitle: str = "",
    icon: str = "",
    eyebrow: str = "",
) -> None:
    """Render a consistent page header: small eyebrow, title, one-line subtitle."""
    eyebrow = eyebrow or st.session_state.get("_fv_section", "")
    parts = ['<div class="fv-header">']
    if eyebrow:
        parts.append(f'<div class="fv-eyebrow">{html.escape(eyebrow)}</div>')
    text = f"{icon} {title}" if icon else title
    parts.append(f'<div class="fv-title">{html.escape(text)}</div>')
    if subtitle:
        parts.append(f'<div class="fv-subtitle">{html.escape(subtitle)}</div>')
    parts.append("</div>")
    st.html("".join(parts))


def section_header(title: str, level: int = 3) -> None:
    """Render a section header."""
    st.markdown(f"{'#' * level} {title}")


def metric_card(
    label: str,
    value: str,
    delta: str | None = None,
    delta_color: str = "normal",
    help: str | None = None,
) -> None:
    """Render a bordered metric card."""
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color, help=help, border=True)


def show_chart(fig: Any, card: bool = False, key: str | None = None, **kwargs) -> Any:
    """Render a Plotly figure with the fedviz template for the viewer's light/dark mode.

    ``card`` wraps the chart in a bordered container.
    """
    theme = get_theme(None)
    theme.apply(fig)
    # Streamlit's frontend fills unset backgrounds with its own colours; keep charts transparent.
    if fig.layout.paper_bgcolor is None:
        fig.update_layout(paper_bgcolor=theme.paper_bg_color)
    if fig.layout.plot_bgcolor is None:
        fig.update_layout(plot_bgcolor=theme.plot_bg_color)
    kwargs.pop("use_container_width", None)
    kwargs.setdefault("width", "stretch")
    kwargs.setdefault("config", PLOTLY_CONFIG)
    if card:
        with st.container(border=True):
            return st.plotly_chart(fig, theme=None, key=key, **kwargs)
    return st.plotly_chart(fig, theme=None, key=key, **kwargs)


def guarded(fn, *args, label: str = "This section", **kwargs) -> Any:
    """Run one page section; if it fails, show a compact error card and keep the page going."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # a broken chart should not take the whole page down
        import traceback

        with st.container(border=True):
            st.markdown(f":material/error: **{label} could not be drawn.** {type(exc).__name__}: {exc}")
            with st.expander("Details"):
                st.code(traceback.format_exc(), language="text")
        return None


def card(
    title: str = "",
    content: Any = None,
) -> None:
    """Render content inside a bordered card."""
    with st.container(border=True):
        if title:
            st.markdown(f"**{title}**")
        if content is not None:
            if callable(content):
                content()
            else:
                st.write(content)


def columns(
    widths: list[int] | int | None = None,
    gap: str = "medium",
) -> list[Any]:
    """Create responsive columns."""
    return st.columns(widths or [1], gap=gap)


def sidebar_info(
    title: str = "About",
    content: str = "",
) -> None:
    """Render info in the sidebar."""
    with st.sidebar:
        st.markdown(f"**{title}**")
        if content:
            st.markdown(content)


def tabs(
    tab_names: list[str],
) -> list[Any]:
    """Create tabs and return tab containers."""
    return st.tabs(tab_names)


def info_box(
    message: str,
    type: str = "info",
) -> None:
    """Render an info/warning/error box."""
    if type == "info":
        st.info(message)
    elif type == "warning":
        st.warning(message)
    elif type == "error":
        st.error(message)
    elif type == "success":
        st.success(message)


def progress_bar(
    value: float,
    label: str = "",
    max_value: float = 1.0,
) -> None:
    """Render a progress bar."""
    st.progress(min(value / max_value, 1.0), text=label)


def download_button(
    data: Any,
    filename: str,
    mime: str = "text/csv",
    label: str = "Download",
) -> None:
    """Render a download button."""
    st.download_button(label=label, data=data, file_name=filename, mime=mime)
