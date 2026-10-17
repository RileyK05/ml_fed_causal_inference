"""Data loading and caching layer for fedviz.

Wraps fedcore.data to provide a clean, cached interface for the
visualization suite. All data access goes through this module.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from fedviz.config import VizConfig


@lru_cache(maxsize=128)
def _read_table(path: str, mtime: float) -> pd.DataFrame:
    """Saved run tables are append-only, so (path, mtime) is a safe cache key."""
    return pd.read_csv(path)


def wide_event_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Pivot a long event panel (``series``/``value``) to one column per series; wide passes through."""
    if not {"series", "value"} <= set(panel.columns):
        return panel
    keys = [c for c in ("announcement_date", "rel_day", "date") if c in panel.columns]
    wide = panel.pivot_table(index=keys, columns="series", values="value", aggfunc="first").reset_index()
    wide.columns.name = None
    return wide


class VizData:
    """Cached data access layer for the visualization suite.

    Provides methods to load datasets, event panels, results, and
    metadata from the shared data layer. All results are cached for
    performance.
    """

    def __init__(self, config: VizConfig | None = None) -> None:
        self.config = config or VizConfig()
        self._catalog: pd.DataFrame | None = None
        self._question_specs: dict[str, Any] | None = None

    @property
    def catalog(self) -> pd.DataFrame:
        """Load and cache the dataset catalog."""
        if self._catalog is None:
            import yaml
            with open(self.config.catalog_file) as f:
                raw = yaml.safe_load(f)
            records = []
            for name, info in raw.get("datasets", {}).items():
                key = info.get("key", "")
                records.append({
                    "name": name,
                    "layer": info.get("layer", ""),
                    "grain": info.get("grain", ""),
                    "description": info.get("description", ""),
                    "questions": ",".join(info.get("questions", [])),
                    "source": info.get("source", ""),
                    "date_col": info.get("date_column", ""),
                    "key_col": ", ".join(key) if isinstance(key, list) else str(key),
                })
            self._catalog = pd.DataFrame(records)
        return self._catalog

    def list_datasets(
        self,
        layer: str | None = None,
        grain: str | None = None,
        question: str | None = None,
    ) -> list[str]:
        """List available datasets with optional filters."""
        df = self.catalog
        if layer:
            df = df[df["layer"] == layer]
        if grain:
            df = df[df["grain"] == grain]
        if question:
            df = df[df["questions"].str.contains(question, na=False)]
        return df["name"].tolist()

    def load(self, name: str, **kwargs) -> pd.DataFrame:
        """Load a dataset by name using fedcore.data.load."""
        from fedcore.data.loaders import load
        return load(name, **kwargs)

    def load_many(self, names: list[str], **kwargs) -> dict[str, pd.DataFrame]:
        """Load multiple datasets at once."""
        return {name: self.load(name, **kwargs) for name in names}

    @lru_cache(maxsize=32)
    def load_cached(self, name: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        """Load a dataset with caching based on name and date range."""
        kwargs: dict[str, Any] = {}
        if start:
            kwargs["start"] = start
        if end:
            kwargs["end"] = end
        return self.load(name, **kwargs)

    def event_panel(self, name: str, pre: int = 5, post: int = 5) -> pd.DataFrame:
        """Event panel for a dataset, wide: one row per (meeting, rel_day), one column per series.

        fedcore returns it long (``series``/``value`` columns); every fedviz chart wants wide.
        Gaps stay NaN (never filled).
        """
        from fedcore.data.panels import event_panel as make_event_panel
        return wide_event_panel(make_event_panel(name, pre=pre, post=post))

    def meetings(self, scheduled_only: bool = True) -> pd.DataFrame:
        """Load FOMC meeting dates."""
        from fedcore.data.panels import meetings
        return meetings(scheduled_only=scheduled_only)

    def sql(self, query: str) -> pd.DataFrame:
        """Execute a SQL query against the DuckDB views (read-only connection)."""
        from fedcore.data.db import query as run_query
        return run_query(query)

    def coverage(self) -> pd.DataFrame:
        """Get dataset coverage information (loads every dataset; slow)."""
        from fedcore.data.loaders import coverage
        return coverage()

    def quick_coverage(self) -> pd.DataFrame:
        """Rows, columns and date range per dataset from DuckDB (one connection, no full loads)."""
        from fedcore.data import catalog
        from fedcore.data.db import connect

        rows = []
        con = connect()
        try:
            for ds in catalog().values():
                row = {"dataset": ds.name, "layer": ds.layer, "grain": ds.grain, "exists": bool(ds.exists)}
                if ds.exists:
                    try:
                        n, s, e = con.execute(f'SELECT COUNT(*), MIN("{ds.date_column}"), MAX("{ds.date_column}") '
                                              f'FROM "{ds.name}"').fetchone()
                        ncol = len(con.execute(f'DESCRIBE "{ds.name}"').fetchall())
                        row.update(rows=int(n), columns=ncol, start=pd.to_datetime(s, errors="coerce"),
                                   end=pd.to_datetime(e, errors="coerce"))
                    except Exception:  # an unreadable view should not hide the rest
                        row.update(exists=False)
                rows.append(row)
        finally:
            con.close()
        return pd.DataFrame(rows)

    def row_count(self, name: str) -> int | None:
        """Row count via DuckDB, without loading the dataset."""
        from fedcore.data.db import query
        try:
            return int(query(f'SELECT COUNT(*) AS n FROM "{name}"').iloc[0]["n"])
        except Exception:
            return None

    def preview(self, name: str, limit: int = 200_000) -> pd.DataFrame:
        """The first ``limit`` rows of a dataset via DuckDB."""
        from fedcore.data.db import query
        return query(f'SELECT * FROM "{name}" LIMIT {int(limit)}')

    def fingerprint(self, name: str) -> str:
        """Get the content fingerprint of a dataset."""
        from fedcore.data.loaders import fingerprint
        return fingerprint(name)

    def question_specs(self) -> dict[str, Any]:
        """Load question specifications."""
        if self._question_specs is None:
            from fedcore.questions import specs
            self._question_specs = specs()
        return self._question_specs

    def list_results(self, project: str | None = None) -> pd.DataFrame:
        """All saved runs across projects, newest first.

        Runs live at ``<project>/results/<question key>/<run id>/manifest.json``.
        """
        import json

        records = []
        dirs = self.config.results_dirs
        if project:
            dirs = [d for d in dirs if d.parent.name == project]
        for results_dir in dirs:
            if not results_dir.exists():
                continue
            for manifest_path in results_dir.rglob("manifest.json"):
                run_dir = manifest_path.parent
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                records.append({
                    "project": results_dir.parent.name,
                    "question": manifest.get("question", ""),
                    "run_id": manifest.get("run_id", run_dir.name),
                    "name": manifest.get("name", ""),
                    "role": manifest.get("role", ""),
                    "created": manifest.get("created", ""),
                    "n_metrics": len(manifest.get("metrics") or {}),
                    "run_dir": str(run_dir),
                })
        columns = ["project", "question", "run_id", "name", "role", "created", "n_metrics", "run_dir"]
        out = pd.DataFrame(records, columns=columns)
        return out.sort_values("created", ascending=False, kind="stable").reset_index(drop=True)

    def load_run(self, run_dir: str | Path, max_table_mb: float = 8.0) -> dict[str, Any]:
        """A run's manifest, tables and figures.

        Tables larger than ``max_table_mb`` are not read (some prediction tables are tens of
        MB); their paths are listed under ``large_tables`` so a page can load one on request.
        """
        import json
        run_dir = Path(run_dir)
        result: dict[str, Any] = {"run_dir": str(run_dir), "tables": {}, "large_tables": {}}
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            result["manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
        tables_dir = run_dir / "tables"
        if tables_dir.exists():
            for table_file in sorted(tables_dir.glob("*.csv")):
                size_mb = table_file.stat().st_size / 1e6
                if size_mb > max_table_mb:
                    result["large_tables"][table_file.stem] = str(table_file)
                else:
                    result["tables"][table_file.stem] = _read_table(str(table_file), table_file.stat().st_mtime)
        figures_dir = run_dir / "figures"
        if figures_dir.exists():
            result["figures"] = {}
            for fig_file in figures_dir.glob("*.json"):
                with open(fig_file, encoding="utf-8") as f:
                    result["figures"][fig_file.stem] = json.load(f)
        return result

    def get_etf_returns(self, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        """Convenience method to load ETF returns."""
        return self.load("etf_returns", start=start, end=end)

    def get_surprises(self) -> pd.DataFrame:
        """Convenience method to load FOMC surprises."""
        return self.load("mps_surprises")

    def get_meetings(self) -> pd.DataFrame:
        """Convenience method to load FOMC meetings."""
        return self.meetings()

    def get_event_study_table(self) -> pd.DataFrame:
        """Convenience method to load the event study table."""
        return self.load("event_study_table")

    def get_fred_controls(self) -> pd.DataFrame:
        """Convenience method to load FRED controls."""
        return self.load("fred_controls")

    def get_treasury_1y(self) -> pd.DataFrame:
        """Convenience method to load 1y treasury yields."""
        return self.load("treasury_1y")
