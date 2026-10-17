"""Tests for the data fixes, theme and page smoke-rendering added with the visual refresh."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from fedviz.config import VizConfig
from fedviz.data import VizData, wide_event_panel
from fedviz.themes import Theme, current_mode, get_theme


def test_wide_event_panel_pivots_long_and_keeps_gaps():
    long = pd.DataFrame({
        "announcement_date": pd.to_datetime(["2024-01-31"] * 4),
        "rel_day": [0, 0, 1, 1],
        "date": pd.to_datetime(["2024-01-31", "2024-01-31", "2024-02-01", "2024-02-01"]),
        "series": ["SPY", "XLF", "SPY", "XLF"],
        "value": [0.01, -0.02, np.nan, 0.03],
    })
    wide = wide_event_panel(long)
    assert list(wide.columns) == ["announcement_date", "rel_day", "date", "SPY", "XLF"]
    assert len(wide) == 2
    assert np.isnan(wide.loc[wide.rel_day == 1, "SPY"]).all()  # gaps stay NaN
    already_wide = wide.copy()
    assert wide_event_panel(already_wide) is already_wide


def _write_run(root, name, created, tables):
    run = root / "q3_susceptibility" / f"{created.replace(':', '')}_{name}"
    (run / "tables").mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({
        "question": "Q3", "name": name, "run_id": run.name, "role": "exploratory",
        "created": created, "metrics": {"mse_per_meeting": 1.0},
    }))
    for tname, df in tables.items():
        df.to_csv(run / "tables" / f"{tname}.csv", index=False)
    return run


def test_runs_are_found_in_question_folders_and_big_tables_are_deferred(tmp_path):
    results = tmp_path / "encoder" / "results"
    _write_run(results, "old", "2026-01-01T00:00:00", {"folds": pd.DataFrame({"fold": [0]})})
    big = pd.DataFrame({"x": np.arange(50_000)})
    run = _write_run(results, "new", "2026-02-01T00:00:00", {"predictions": big, "folds": pd.DataFrame({"fold": [0]})})
    data = VizData(VizConfig(results_dirs=[results], cache_dir=tmp_path / "cache"))

    runs = data.list_results()
    assert list(runs["name"]) == ["new", "old"]  # newest first
    assert (runs["project"] == "encoder").all() and (runs["n_metrics"] == 1).all()

    loaded = data.load_run(run, max_table_mb=0.1)
    assert "folds" in loaded["tables"] and "predictions" not in loaded["tables"]
    assert loaded["large_tables"]["predictions"].endswith("predictions.csv")


def test_theme_template_is_transparent_and_mode_falls_back():
    for theme in (Theme.dark(), Theme.light()):
        layout = theme.template().layout
        assert layout.paper_bgcolor == "rgba(0,0,0,0)" and layout.plot_bgcolor == "rgba(0,0,0,0)"
    assert current_mode() in ("light", "dark")  # outside a Streamlit session: the default
    assert get_theme("light").name == "light" and get_theme(None).name in ("light", "dark")
    fig = get_theme("dark").apply(go.Figure(go.Scatter(x=[1, 2], y=[1, 2])))
    assert fig.layout.template.layout.font.family.startswith("Inter")


def test_guarded_returns_value_and_swallows_errors():
    from fedviz.components.layout import guarded

    assert guarded(lambda a, b=0: a + b, 1, b=2) == 3

    def boom():
        raise ValueError("nope")

    assert guarded(boom, label="Test") is None


PAGES = ["render_market_overview", "render_event_study", "render_cross_section", "render_model_diagnostics",
         "render_results_browser", "render_comparison", "render_data_explorer"]


def _page_script(name):
    from fedviz import pages
    from fedviz.components.layout import inject_css
    from fedviz.themes import set_default_template

    set_default_template("dark")
    inject_css()
    getattr(pages, name)()


@pytest.mark.parametrize("name", PAGES)
def test_page_renders_without_exceptions(name):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_page_script, args=(name,), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert not [m.value for m in at.markdown if "could not be drawn" in m.value]
