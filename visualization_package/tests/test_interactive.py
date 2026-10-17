"""Tests for the fedviz interactive layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from fedviz.interactive import (
    FilterState,
    animated_time_series,
    comparison_overlay,
    cross_filter_panel,
    filter_df_by_state,
    get_filter_state,
    linked_heatmap_with_drill,
    linked_scatter,
    linked_time_series,
    rich_tooltip_scatter,
    selection_info_panel,
)


@pytest.fixture
def sample_df():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    return pd.DataFrame({
        "date": dates,
        "A": np.random.randn(100).cumsum() * 0.01,
        "B": np.random.randn(100).cumsum() * 0.01,
        "C": np.random.randn(100).cumsum() * 0.01,
        "category": np.random.choice(["X", "Y", "Z"], 100),
    })


def test_filter_state():
    state = FilterState()
    assert state.date_range is None
    assert state.selected_categories == []
    assert state.selected_points == []
    state.clear()
    assert state.date_range is None


def test_filter_state_to_dict():
    state = FilterState()
    state.selected_categories = ["A", "B"]
    d = state.to_dict()
    assert "selected_categories" in d
    assert d["selected_categories"] == ["A", "B"]


def test_linked_scatter(sample_df):
    fig = linked_scatter(sample_df, x="A", y="B")
    assert isinstance(fig, go.Figure)


def test_linked_time_series(sample_df):
    fig = linked_time_series(sample_df, x="date", y=["A", "B"])
    assert isinstance(fig, go.Figure)


def test_linked_heatmap_with_drill(sample_df):
    pivot = sample_df.pivot_table(index="category", columns="date", values="A", aggfunc="mean")
    fig = linked_heatmap_with_drill(pivot.reset_index(), x="category", y=None, z=None)
    assert isinstance(fig, go.Figure)


def test_rich_tooltip_scatter(sample_df):
    fig = rich_tooltip_scatter(sample_df, x="A", y="B", tooltip_cols=["C"])
    assert isinstance(fig, go.Figure)


def test_comparison_overlay(sample_df):
    df1 = sample_df[["date", "A"]].rename(columns={"A": "value"})
    df2 = sample_df[["date", "B"]].rename(columns={"B": "value"})
    fig = comparison_overlay({"Run 1": df1, "Run 2": df2}, x="date", y="value")
    assert isinstance(fig, go.Figure)


def test_animated_time_series(sample_df):
    fig = animated_time_series(sample_df, x="date", y="A")
    assert isinstance(fig, go.Figure)
    assert len(fig.frames) > 0


def test_filter_df_by_state(sample_df):
    state = FilterState()
    filtered = filter_df_by_state(sample_df, state)
    assert len(filtered) == len(sample_df)

    state.custom_filters = {"category": ["X", "Y"]}
    filtered = filter_df_by_state(sample_df, state)
    assert len(filtered) < len(sample_df)
    assert set(filtered["category"].unique()) == {"X", "Y"}
