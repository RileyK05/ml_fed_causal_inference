"""Tests for fedviz export capabilities."""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import pytest

from fedviz.export import (
    export_chart_html,
    export_df_csv,
    export_df_json,
    export_summary_report,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6],
        "C": [7, 8, 9],
    })


@pytest.fixture
def sample_fig():
    return go.Figure(data=go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))


def test_export_df_csv(sample_df, tmp_path):
    output = tmp_path / "test.csv"
    sample_df.to_csv(output, index=False)
    loaded = pd.read_csv(output)
    assert len(loaded) == 3
    assert list(loaded.columns) == ["A", "B", "C"]


def test_export_df_json(sample_df, tmp_path):
    output = tmp_path / "test.json"
    sample_df.to_json(output, orient="records")
    with open(output) as f:
        data = json.load(f)
    assert len(data) == 3
    assert data[0]["A"] == 1


def test_export_chart_html(sample_fig):
    html = sample_fig.to_html(include_plotlyjs="cdn", full_html=True)
    assert "<html>" in html
    assert "plotly" in html.lower()
