"""Tests for the fedviz plugin registry."""

from __future__ import annotations

import pytest

from fedviz.registry import (
    ChartPlugin,
    MetricPlugin,
    ModelPlugin,
    PagePlugin,
    VizRegistry,
    get_registry,
    register_chart,
    register_metric,
    register_model,
    register_page,
)


def test_registry_singleton():
    r1 = get_registry()
    r2 = get_registry()
    assert r1 is r2


def test_register_model_plugin():
    registry = VizRegistry()

    def dummy_render(data=None):
        pass

    plugin = ModelPlugin(
        name="test_model",
        display_name="Test Model",
        description="A test model",
    )
    plugin.register_page(PagePlugin(
        name="test_page",
        icon="",
        render_fn=dummy_render,
        description="Test page",
    ))
    plugin.register_chart(ChartPlugin(
        name="test_chart",
        render_fn=lambda: None,
        description="Test chart",
    ))
    plugin.register_metric(MetricPlugin(
        name="test_metric",
        compute_fn=lambda: 42.0,
        description="Test metric",
    ))

    registry.register_model(plugin)

    assert "test_model" in registry.list_models()
    assert "test_page" in [p.name for p in registry.list_pages()]
    assert "test_chart" in [c.name for c in registry.list_charts()]
    assert "test_metric" in [m.name for m in registry.list_metrics()]


def test_register_standalone_plugins():
    registry = VizRegistry()

    def dummy_render(data=None):
        pass

    registry.register_page(PagePlugin(
        name="standalone_page",
        icon="",
        render_fn=dummy_render,
    ))
    registry.register_chart(ChartPlugin(
        name="standalone_chart",
        render_fn=lambda: None,
    ))
    registry.register_metric(MetricPlugin(
        name="standalone_metric",
        compute_fn=lambda: 0.0,
    ))

    assert registry.get_page("standalone_page") is not None
    assert registry.get_chart("standalone_chart") is not None
    assert registry.get_metric("standalone_metric") is not None


def test_model_plugin_attributes():
    plugin = ModelPlugin(
        name="test",
        display_name="Test",
        description="Test model",
        version="1.0.0",
    )
    assert plugin.name == "test"
    assert plugin.display_name == "Test"
    assert plugin.version == "1.0.0"
    assert plugin.pages == {}
    assert plugin.charts == {}
    assert plugin.metrics == {}


def test_page_plugin_order():
    registry = VizRegistry()

    def dummy_render(data=None):
        pass

    for i, name in enumerate(["page_a", "page_b", "page_c"]):
        registry.register_page(PagePlugin(
            name=name,
            icon="",
            render_fn=dummy_render,
            order=i * 10,
        ))

    pages = registry.list_pages()
    assert len(pages) == 3
    assert [p.name for p in pages] == ["page_a", "page_b", "page_c"]


def test_discover_plugins():
    registry = VizRegistry()
    registry.discover_plugins()
    models = registry.list_models()
    assert "dml" in models
    assert "encoder" in models
    assert "cde" in models
