"""Data retrieval layer. Two ways in, same catalog underneath:

    from fedci.data import load, query
    ret = load("etf_returns", start="2024-01-01")            # pandas
    df  = query("SELECT * FROM fomc_meetings")                # SQL (DuckDB)
"""
from fedci.data.catalog import Dataset, catalog, for_question, get
from fedci.data.db import connect, query, tables
from fedci.data.loaders import check, coverage, fingerprint, load
from fedci.data.panels import event_panel, meetings

__all__ = [
    "Dataset", "catalog", "get", "for_question",
    "load", "coverage", "check", "fingerprint",
    "connect", "query", "tables",
    "meetings", "event_panel",
]
