"""Plugin registry for the fedviz visualization suite.

Any market model can register custom visualizers, chart types, and metrics
through this registry. The registry is the backbone of the generalized
visualization framework — it decouples the visualization infrastructure
from any specific model implementation.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from dataclasses import dataclass, field
from typing import Any, Callable

import streamlit as st


@dataclass
class PagePlugin:
    """A custom page that a model can register."""

    name: str
    icon: str
    render_fn: Callable[..., None]
    description: str = ""
    category: str = "model"
    order: int = 100


@dataclass
class ChartPlugin:
    """A custom chart type that a model can register."""

    name: str
    render_fn: Callable[..., Any]
    description: str = ""
    category: str = "model"


@dataclass
class MetricPlugin:
    """A custom metric that a model can register."""

    name: str
    compute_fn: Callable[..., float]
    description: str = ""
    format: str = "number"
    category: str = "model"


@dataclass
class ModelPlugin:
    """A complete model plugin that bundles pages, charts, and metrics."""

    name: str
    display_name: str
    description: str
    version: str = "0.1.0"
    pages: dict[str, PagePlugin] = field(default_factory=dict)
    charts: dict[str, ChartPlugin] = field(default_factory=dict)
    metrics: dict[str, MetricPlugin] = field(default_factory=dict)

    def register_page(self, page: PagePlugin) -> None:
        self.pages[page.name] = page

    def register_chart(self, chart: ChartPlugin) -> None:
        self.charts[chart.name] = chart

    def register_metric(self, metric: MetricPlugin) -> None:
        self.metrics[metric.name] = metric


class VizRegistry:
    """Central registry for all visualization plugins.

    The registry discovers plugins from the built-in plugin package and
    any external packages that register themselves. It provides lookup
    methods for pages, charts, and metrics.
    """

    def __init__(self) -> None:
        self._models: dict[str, ModelPlugin] = {}
        self._pages: dict[str, PagePlugin] = {}
        self._charts: dict[str, ChartPlugin] = {}
        self._metrics: dict[str, MetricPlugin] = {}
        self._discovered = False

    def register_model(self, plugin: ModelPlugin) -> None:
        """Register a complete model plugin."""
        self._models[plugin.name] = plugin
        for page in plugin.pages.values():
            self._pages[page.name] = page
        for chart in plugin.charts.values():
            self._charts[chart.name] = chart
        for metric in plugin.metrics.values():
            self._metrics[metric.name] = metric

    def register_page(self, page: PagePlugin) -> None:
        """Register a standalone page plugin."""
        self._pages[page.name] = page

    def register_chart(self, chart: ChartPlugin) -> None:
        """Register a standalone chart plugin."""
        self._charts[chart.name] = chart

    def register_metric(self, metric: MetricPlugin) -> None:
        """Register a standalone metric plugin."""
        self._metrics[metric.name] = metric

    def discover_plugins(self) -> None:
        """Auto-discover plugins from the fedviz.plugins package."""
        if self._discovered:
            return
        try:
            from fedviz.plugins import builtin
            for name, obj in inspect.getmembers(builtin):
                if isinstance(obj, ModelPlugin):
                    self.register_model(obj)
        except ImportError:
            pass
        self._discovered = True

    def list_models(self) -> list[str]:
        """List all registered model names."""
        self.discover_plugins()
        return sorted(self._models.keys())

    def get_model(self, name: str) -> ModelPlugin | None:
        """Get a model plugin by name."""
        self.discover_plugins()
        return self._models.get(name)

    def list_pages(self, category: str | None = None) -> list[PagePlugin]:
        """List all registered pages, optionally filtered by category."""
        pages = list(self._pages.values())
        if category:
            pages = [p for p in pages if p.category == category]
        return sorted(pages, key=lambda p: p.order)

    def get_page(self, name: str) -> PagePlugin | None:
        """Get a page plugin by name."""
        self._discovered = True
        return self._pages.get(name)

    def list_charts(self, category: str | None = None) -> list[ChartPlugin]:
        """List all registered chart types."""
        self.discover_plugins()
        charts = list(self._charts.values())
        if category:
            charts = [c for c in charts if c.category == category]
        return sorted(charts, key=lambda c: c.name)

    def get_chart(self, name: str) -> ChartPlugin | None:
        """Get a chart plugin by name."""
        self.discover_plugins()
        return self._charts.get(name)

    def list_metrics(self, category: str | None = None) -> list[MetricPlugin]:
        """List all registered metrics."""
        self.discover_plugins()
        metrics = list(self._metrics.values())
        if category:
            metrics = [m for m in metrics if m.category == category]
        return sorted(metrics, key=lambda m: m.name)

    def get_metric(self, name: str) -> MetricPlugin | None:
        """Get a metric plugin by name."""
        self.discover_plugins()
        return self._metrics.get(name)

    def render_page(self, name: str, **kwargs) -> None:
        """Render a registered page by name."""
        page = self.get_page(name)
        if page is None:
            st.error(f"Page '{name}' not found in registry.")
            return
        page.render_fn(**kwargs)

    def render_chart(self, name: str, **kwargs) -> Any:
        """Render a registered chart by name."""
        chart = self.get_chart(name)
        if chart is None:
            st.error(f"Chart '{name}' not found in registry.")
            return None
        return chart.render_fn(**kwargs)

    def compute_metric(self, name: str, **kwargs) -> float:
        """Compute a registered metric by name."""
        metric = self.get_metric(name)
        if metric is None:
            st.error(f"Metric '{name}' not found in registry.")
            return float("nan")
        return metric.compute_fn(**kwargs)


_registry: VizRegistry | None = None


def get_registry() -> VizRegistry:
    """Get the global registry instance."""
    global _registry
    if _registry is None:
        _registry = VizRegistry()
    return _registry


def register_model(plugin: ModelPlugin) -> None:
    """Register a model plugin with the global registry."""
    get_registry().register_model(plugin)


def register_page(page: PagePlugin) -> None:
    """Register a page plugin with the global registry."""
    get_registry().register_page(page)


def register_chart(chart: ChartPlugin) -> None:
    """Register a chart plugin with the global registry."""
    get_registry().register_chart(chart)


def register_metric(metric: MetricPlugin) -> None:
    """Register a metric plugin with the global registry."""
    get_registry().register_metric(metric)
