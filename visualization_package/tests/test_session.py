"""Tests for fedviz session state management."""

from __future__ import annotations

import pytest

from fedviz.session import (
    add_to_list,
    clear_all_filters,
    clear_list,
    get_active_filters,
    get_filter,
    get_saved_views,
    get_state,
    init_session_state,
    load_view_state,
    remove_from_list,
    save_view_state,
    set_filter,
    set_state,
    toggle_state,
)


def test_init_session_state():
    init_session_state()
    assert get_state("theme") == "dark"
    assert get_state("active_page") == "Market Overview"


def test_set_and_get_state():
    set_state("test_key", "test_value")
    assert get_state("test_key") == "test_value"


def test_get_state_default():
    assert get_state("nonexistent", "default") == "default"


def test_toggle_state():
    set_state("toggle_test", False)
    toggle_state("toggle_test")
    assert get_state("toggle_test") is True
    toggle_state("toggle_test")
    assert get_state("toggle_test") is False


def test_list_operations():
    clear_list("test_list")
    add_to_list("test_list", "a")
    add_to_list("test_list", "b")
    add_to_list("test_list", "c")
    assert get_state("test_list") == ["a", "b", "c"]
    remove_from_list("test_list", "b")
    assert get_state("test_list") == ["a", "c"]


def test_list_max_items():
    clear_list("max_test")
    for i in range(15):
        add_to_list("max_test", i, max_items=5)
    assert len(get_state("max_test")) == 5


def test_save_and_load_view():
    init_session_state()
    set_state("selected_sectors", ["SPY", "XLF"])
    save_view_state("test_view")
    assert "test_view" in get_saved_views()
    set_state("selected_sectors", ["XLK"])
    load_view_state("test_view")
    assert get_state("selected_sectors") == ["SPY", "XLF"]


def test_get_active_filters():
    init_session_state()
    set_state("date_range", ("2020-01-01", "2020-12-31"))
    set_state("selected_sectors", ["SPY"])
    filters = get_active_filters()
    assert "date_range" in filters
    assert "selected_sectors" in filters
