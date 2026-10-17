"""Tests for the fedviz animation system."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from fedviz.animation import (
    animated_bubble_chart,
    animated_heatmap_evolution,
    animated_sector_evolution,
    meeting_by_meeting_animation,
)


@pytest.fixture
def sample_df():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=50, freq="B")
    return pd.DataFrame({
        "date": dates,
        "SPY": np.random.randn(50).cumsum() * 0.01,
        "XLF": np.random.randn(50).cumsum() * 0.01,
        "XLK": np.random.randn(50).cumsum() * 0.01,
        "XLE": np.random.randn(50).cumsum() * 0.01,
    })


@pytest.fixture
def event_panel():
    np.random.seed(42)
    records = []
    for meeting in pd.date_range("2020-01-01", periods=10, freq="30D"):
        for day in range(-5, 6):
            records.append({
                "announcement_date": meeting,
                "rel_day": day,
                "SPY": np.random.randn() * 0.01,
                "XLF": np.random.randn() * 0.01,
                "XLK": np.random.randn() * 0.01,
            })
    return pd.DataFrame(records)


def test_animated_sector_evolution(sample_df):
    fig = animated_sector_evolution(
        sample_df, date_col="date", sector_cols=["SPY", "XLF", "XLK"],
        speed=100,
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.frames) > 0


def test_meeting_by_meeting_animation(event_panel):
    fig = meeting_by_meeting_animation(
        event_panel, sector_cols=["SPY", "XLF"],
        speed=200,
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.frames) > 0


def test_animated_bubble_chart(sample_df):
    df = sample_df.copy()
    df["size"] = np.abs(df["SPY"]) * 100
    df["color"] = df["XLF"]
    fig = animated_bubble_chart(
        df, x="SPY", y="XLF", size="size", color="color",
        frame_col="date", speed=100,
    )
    assert isinstance(fig, go.Figure)


def test_animated_heatmap_evolution(sample_df):
    df = sample_df.copy()
    df["category"] = "A"
    fig = animated_heatmap_evolution(
        df, date_col="date", value_col="SPY", category_col="category",
        speed=100,
    )
    assert isinstance(fig, go.Figure)
