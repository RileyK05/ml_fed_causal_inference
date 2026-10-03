"""fedcore: the shared layer every project builds on.

    fedcore.data        dataset catalog, load(), query() (DuckDB), meetings(), event_panel()
    fedcore.protocol    walk_forward() and meeting_bootstrap(): splits by whole meetings
    fedcore.results     ResultStore: append-only runs with data fingerprints
    fedcore.questions   the five question definitions (QuestionSpec), metadata only

The projects (dml/, encoder/, cde/) import fedcore and never import each other.
"""
__version__ = "0.1.0"
