"""Base classes for fedviz plugins.

To create a custom plugin for your model:

    from fedviz.plugins.base import ModelPlugin, PagePlugin
    from fedviz.registry import register_model

    plugin = ModelPlugin(
        name="my_model",
        display_name="My Model",
        description="Custom model visualizations",
    )

    plugin.register_page(PagePlugin(
        name="my_page",
        icon="",
        render_fn=my_render_function,
        description="My custom page",
    ))

    register_model(plugin)
"""

from __future__ import annotations

from fedviz.registry import ChartPlugin, MetricPlugin, ModelPlugin, PagePlugin

__all__ = ["ModelPlugin", "PagePlugin", "ChartPlugin", "MetricPlugin"]
