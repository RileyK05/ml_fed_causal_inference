"""Data retrieval layer. Two ways in, same catalog underneath:

    from fedcore.data import load, query
    ret = load("etf_returns", start="2024-01-01")            # pandas
    df  = query("SELECT * FROM fomc_meetings")                # SQL (DuckDB)
"""
from fedcore.data.catalog import Dataset, catalog, for_question, get
from fedcore.data.db import connect, query, tables
from fedcore.data.loaders import check, coverage, fingerprint, load
from fedcore.data.panels import event_panel, meetings

__all__ = [
    "Dataset", "catalog", "get", "for_question",
    "load", "coverage", "check", "fingerprint",
    "connect", "query", "tables",
    "meetings", "event_panel",
]
