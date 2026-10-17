"""Tests for fedviz chart builders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from fedviz.components.charts import (
    bar_chart,
    box_plot,
    distribution,
    forest_plot,
    heatmap,
    qq_plot,
    residual_plot,
    scatter,
    time_series,
    time_series_with_events,
    violin_plot,
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


def test_time_series(sample_df):
    fig = time_series(sample_df, x="date", y=["A", "B"])
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_time_series_with_events(sample_df):
    events = pd.DataFrame({"announcement_date": pd.to_datetime(["2020-02-01", "2020-04-01"])})
    fig = time_series_with_events(sample_df, x="date", y="A", events=events)
    assert isinstance(fig, go.Figure)


def test_heatmap(sample_df):
    pivot = sample_df.pivot_table(index="category", columns="date", values="A", aggfunc="mean")
    fig = heatmap(pivot.reset_index(), x="category", y=None, z=None)
    assert isinstance(fig, go.Figure)


def test_scatter(sample_df):
    fig = scatter(sample_df, x="A", y="B")
    assert isinstance(fig, go.Figure)


def test_scatter_with_regression(sample_df):
    fig = scatter(sample_df, x="A", y="B", add_regression=True)
    assert isinstance(fig, go.Figure)


def test_distribution(sample_df):
    fig = distribution(sample_df, "A")
    assert isinstance(fig, go.Figure)


def test_box_plot(sample_df):
    fig = box_plot(sample_df, y="A", x="category")
    assert isinstance(fig, go.Figure)


def test_violin_plot(sample_df):
    fig = violin_plot(sample_df, y="A", x="category")
    assert isinstance(fig, go.Figure)


def test_bar_chart(sample_df):
    agg = sample_df.groupby("category")["A"].mean().reset_index()
    fig = bar_chart(agg, x="category", y="A")
    assert isinstance(fig, go.Figure)


def test_forest_plot(sample_df):
    est_df = pd.DataFrame({
        "name": ["A", "B", "C"],
        "estimate": [0.1, 0.2, 0.3],
        "ci_low": [0.05, 0.15, 0.25],
        "ci_high": [0.15, 0.25, 0.35],
    })
    fig = forest_plot(est_df)
    assert isinstance(fig, go.Figure)


def test_qq_plot(sample_df):
    fig = qq_plot(sample_df["A"])
    assert isinstance(fig, go.Figure)


def test_residual_plot(sample_df):
    fig = residual_plot(sample_df["A"], sample_df["B"])
    assert isinstance(fig, go.Figure)
