"""Tests for fedviz page renderers."""

from __future__ import annotations

import pandas as pd
import pytest

from fedviz.data import VizData
from fedviz.pages.market_overview import render as render_market_overview
from fedviz.pages.event_study import render as render_event_study
from fedviz.pages.cross_section import render as render_cross_section
from fedviz.pages.model_diagnostics import render as render_model_diagnostics
from fedviz.pages.results_browser import render as render_results_browser
from fedviz.pages.comparison import render as render_comparison
from fedviz.pages.data_explorer import render as render_data_explorer


@pytest.fixture
def viz_data():
    from fedviz.config import VizConfig
    return VizData(VizConfig())


def test_market_overview_render(viz_data):
    try:
        render_market_overview(data=viz_data)
    except Exception as e:
        if "missing ScriptRunContext" not in str(e):
            raise


def test_event_study_render(viz_data):
    try:
        render_event_study(data=viz_data)
    except Exception as e:
        if "missing ScriptRunContext" not in str(e):
            raise


def test_cross_section_render(viz_data):
    try:
        render_cross_section(data=viz_data)
    except Exception as e:
        if "missing ScriptRunContext" not in str(e):
            raise


def test_model_diagnostics_render(viz_data):
    try:
        render_model_diagnostics(data=viz_data)
    except Exception as e:
        if "missing ScriptRunContext" not in str(e):
            raise


def test_results_browser_render(viz_data):
    try:
        render_results_browser(data=viz_data)
    except Exception as e:
        if "missing ScriptRunContext" not in str(e):
            raise


def test_comparison_render(viz_data):
    try:
        render_comparison(data=viz_data)
    except Exception as e:
        if "missing ScriptRunContext" not in str(e):
            raise


def test_data_explorer_render(viz_data):
    try:
        render_data_explorer(data=viz_data)
    except Exception as e:
        if "missing ScriptRunContext" not in str(e):
            raise
