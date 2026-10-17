"""Tests for the fedviz data layer."""

from __future__ import annotations

import pandas as pd
import pytest

from fedviz.config import VizConfig
from fedviz.data import VizData


@pytest.fixture
def viz_data():
    config = VizConfig()
    return VizData(config)


def test_catalog_loading(viz_data):
    catalog = viz_data.catalog
    assert isinstance(catalog, pd.DataFrame)
    assert len(catalog) > 0
    assert "name" in catalog.columns
    assert "layer" in catalog.columns
    assert "grain" in catalog.columns


def test_list_datasets(viz_data):
    datasets = viz_data.list_datasets()
    assert isinstance(datasets, list)
    assert len(datasets) > 0
    assert "etf_returns" in datasets


def test_list_datasets_with_filter(viz_data):
    daily = viz_data.list_datasets(grain="daily")
    assert isinstance(daily, list)
    assert len(daily) > 0


def test_load_etf_returns(viz_data):
    df = viz_data.load("etf_returns")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "date" in df.columns


def test_load_surprises(viz_data):
    df = viz_data.get_surprises()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_meetings(viz_data):
    df = viz_data.get_meetings()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_question_specs(viz_data):
    specs = viz_data.question_specs()
    assert isinstance(specs, dict)
    assert len(specs) > 0


def test_fingerprint(viz_data):
    fp = viz_data.fingerprint("etf_returns")
    assert isinstance(fp, str)
    assert len(fp) > 0
