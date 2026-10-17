"""Plugin system for fedviz.

Models can register custom visualizers by creating a ModelPlugin and
registering it with the global registry. The registry auto-discovers
plugins from this package.
"""

from fedviz.plugins.base import ModelPlugin, PagePlugin, ChartPlugin, MetricPlugin

__all__ = ["ModelPlugin", "PagePlugin", "ChartPlugin", "MetricPlugin"]
