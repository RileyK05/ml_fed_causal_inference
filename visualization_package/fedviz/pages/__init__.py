"""Streamlit pages for the fedviz visualization suite."""

from fedviz.pages.market_overview import render as render_market_overview
from fedviz.pages.event_study import render as render_event_study
from fedviz.pages.cross_section import render as render_cross_section
from fedviz.pages.model_diagnostics import render as render_model_diagnostics
from fedviz.pages.results_browser import render as render_results_browser
from fedviz.pages.comparison import render as render_comparison
from fedviz.pages.data_explorer import render as render_data_explorer

__all__ = [
    "render_market_overview",
    "render_event_study",
    "render_cross_section",
    "render_model_diagnostics",
    "render_results_browser",
    "render_comparison",
    "render_data_explorer",
]
