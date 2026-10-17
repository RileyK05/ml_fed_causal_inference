"""Main entry point for the fedviz Streamlit app.

Run with (from anywhere; uses the package's theme):
    python -m fedviz            # or: fedviz

Plain Streamlit also works, but then the theme in visualization_package/.streamlit only
applies when you start it from that folder:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Add repo root to path so fedcore is importable
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from fedviz import __version__
from fedviz.components.layout import guarded, inject_css
from fedviz.config import VizConfig
from fedviz.data import VizData
from fedviz.pages import (
    render_comparison,
    render_cross_section,
    render_data_explorer,
    render_event_study,
    render_market_overview,
    render_model_diagnostics,
    render_results_browser,
)
from fedviz.registry import get_registry
from fedviz.session import (
    clear_all_filters,
    get_active_filters,
    get_saved_views,
    init_session_state,
    load_view_state,
    save_view_state,
    set_state,
)
from fedviz.themes import current_mode, set_default_template

ASSETS = Path(__file__).resolve().parent / "fedviz" / "assets"

# (section, title, Material icon, url path, renderer)
PAGES = [
    ("Markets", "Market Overview", ":material/monitoring:", "market", render_market_overview),
    ("Markets", "Event Study", ":material/event:", "event-study", render_event_study),
    ("Markets", "Cross-Section", ":material/scatter_plot:", "cross-section", render_cross_section),
    ("Models", "Model Diagnostics", ":material/troubleshoot:", "diagnostics", render_model_diagnostics),
    ("Models", "Results Browser", ":material/folder_open:", "results", render_results_browser),
    ("Models", "Comparison", ":material/compare_arrows:", "comparison", render_comparison),
    ("Data", "Data Explorer", ":material/database:", "data", render_data_explorer),
]


def _data() -> VizData:
    return VizData(VizConfig())  # cheap; not cached so code reloads take effect


def _page(section: str, title: str, icon: str, url: str, render, data: VizData) -> st.Page:
    def run() -> None:
        st.session_state["_fv_section"] = section
        set_state("active_page", title)
        guarded(render, data=data, label=title)

    return st.Page(run, title=title, icon=icon, url_path=url, default=url == "market")


def _shown_filters(filters: dict) -> dict:
    """Only filters a viewer actually set: no empty values, no defaults, no sector picker."""
    labels = {"date_range": "Dates", "selected_meetings": "Meetings"}
    out = {labels[k]: v for k, v in filters.items() if k in labels and v not in (None, [], ())}
    pre, post = filters.get("pre_window", 5), filters.get("post_window", 5)
    if (pre, post) != (5, 5):
        out["Event window"] = f"-{pre} to +{post} days"
    return out


def _sidebar_tools() -> None:
    with st.sidebar:
        active_filters = _shown_filters(get_active_filters())
        if active_filters:
            with st.container(border=True):
                st.markdown("**Active filters**")
                for k, v in active_filters.items():
                    st.caption(f"{k}: {v}")
                if st.button("Clear all", type="tertiary", icon=":material/filter_alt_off:"):
                    clear_all_filters()

        with st.expander("Saved views", icon=":material/bookmark:"):
            saved = get_saved_views()
            if saved:
                selected_view = st.selectbox("Load view", ["—"] + saved)
                if selected_view != "—":
                    load_view_state(selected_view)
            view_name = st.text_input("Save current view as", placeholder="view name")
            if st.button("Save view", width="stretch") and view_name:
                save_view_state(view_name)
                st.toast(f"View '{view_name}' saved", icon=":material/bookmark_added:")

        st.html(f'<div class="fv-footer">FedViz v{__version__} · FOMC causal inference</div>')


def main() -> None:
    """Render the main fedviz application."""
    st.set_page_config(
        page_title="FedViz",
        page_icon=str(ASSETS / "icon.svg"),
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state()
    mode = current_mode()
    set_state("theme", mode)
    set_default_template(mode)
    inject_css()
    st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "icon.svg"), size="large")

    data = _data()
    nav: dict[str, list[st.Page]] = {}
    for section, title, icon, url, render in PAGES:
        nav.setdefault(section, []).append(_page(section, title, icon, url, render, data))

    registry = get_registry()
    for mp in registry.list_pages(category="model"):
        def run_plugin(name=mp.name) -> None:
            st.session_state["_fv_section"] = "Plugins"
            registry.render_page(name, data=data)

        nav.setdefault("Plugins", []).append(
            st.Page(run_plugin, title=mp.description or mp.name, icon=mp.icon or ":material/extension:",
                    url_path=f"plugin-{mp.name}"))

    page = st.navigation(nav)
    page.run()
    _sidebar_tools()  # after the page, so its own filters sit above


if __name__ == "__main__":
    main()
