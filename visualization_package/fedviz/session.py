"""Session state management for fedviz.

Preserves user interactions, filters, and preferences across page
switches and reruns. This makes the app feel like a persistent
application rather than a stateless script.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


def init_session_state() -> None:
    """Initialize all session state variables with defaults."""
    defaults = {
        "theme": "dark",
        "active_page": "Market Overview",
        "selected_run": None,
        "selected_meetings": [],
        "selected_sectors": ["SPY", "XLF", "XLK", "XLE", "XLV"],
        "date_range": None,
        "pre_window": 5,
        "post_window": 5,
        "rolling_window": 63,
        "animation_speed": 100,
        "show_confidence_bands": True,
        "show_points": False,
        "color_by": None,
        "chart_height": 500,
        "max_rows": 10000,
        "sql_query": "SELECT * FROM etf_returns LIMIT 100",
        "sql_results": None,
        "filter_state": None,
        "comparison_runs": [],
        "drill_down": None,
        "sidebar_expanded": True,
        "show_advanced": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_state(key: str, default: Any = None) -> Any:
    """Get a session state value."""
    return st.session_state.get(key, default)


def set_state(key: str, value: Any) -> None:
    """Set a session state value."""
    st.session_state[key] = value


def toggle_state(key: str) -> None:
    """Toggle a boolean session state value."""
    st.session_state[key] = not st.session_state.get(key, False)


def add_to_list(key: str, value: Any, max_items: int = 10) -> None:
    """Add an item to a list in session state."""
    if key not in st.session_state:
        st.session_state[key] = []
    if value not in st.session_state[key]:
        st.session_state[key].append(value)
        if len(st.session_state[key]) > max_items:
            st.session_state[key] = st.session_state[key][-max_items:]


def remove_from_list(key: str, value: Any) -> None:
    """Remove an item from a list in session state."""
    if key in st.session_state and value in st.session_state[key]:
        st.session_state[key].remove(value)


def clear_list(key: str) -> None:
    """Clear a list in session state."""
    st.session_state[key] = []


def persist_filter(key: str, value: Any) -> None:
    """Persist a filter value and trigger rerun."""
    st.session_state[key] = value
    st.rerun()


def get_filter(key: str, default: Any = None) -> Any:
    """Get a filter value from session state."""
    return st.session_state.get(key, default)


def set_filter(key: str, value: Any) -> None:
    """Set a filter value and trigger rerun."""
    st.session_state[key] = value
    st.rerun()


def clear_all_filters() -> None:
    """Clear all filter-related session state."""
    filter_keys = [
        "date_range", "selected_meetings", "selected_sectors",
        "pre_window", "post_window", "filter_state",
    ]
    for key in filter_keys:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()


def get_active_filters() -> dict[str, Any]:
    """Get all active filters as a dict."""
    filter_keys = [
        "date_range", "selected_meetings", "selected_sectors",
        "pre_window", "post_window",
    ]
    return {k: st.session_state[k] for k in filter_keys if k in st.session_state}


def save_view_state(name: str) -> None:
    """Save the current view state with a name."""
    if "saved_views" not in st.session_state:
        st.session_state["saved_views"] = {}
    st.session_state["saved_views"][name] = {
        "theme": st.session_state.get("theme"),
        "active_page": st.session_state.get("active_page"),
        "selected_sectors": st.session_state.get("selected_sectors"),
        "date_range": st.session_state.get("date_range"),
        "pre_window": st.session_state.get("pre_window"),
        "post_window": st.session_state.get("post_window"),
    }


def load_view_state(name: str) -> bool:
    """Load a saved view state. Returns True if found."""
    if "saved_views" not in st.session_state:
        return False
    if name not in st.session_state["saved_views"]:
        return False
    view = st.session_state["saved_views"][name]
    for key, value in view.items():
        st.session_state[key] = value
    st.rerun()
    return True


def get_saved_views() -> list[str]:
    """Get list of saved view names."""
    if "saved_views" not in st.session_state:
        return []
    return list(st.session_state["saved_views"].keys())
