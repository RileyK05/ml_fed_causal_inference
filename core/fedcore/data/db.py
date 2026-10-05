"""SQL access to every catalog dataset through DuckDB.

Each dataset is exposed as a view with its catalog name, reading the file in place (no
copy, no server). Example:

    from fedcore.data import query
    query("SELECT announcement_date, MP1, ret_XLF FROM event_study_table WHERE ABS(MP1) > 0.05")
"""
from __future__ import annotations

import duckdb
import pandas as pd

from fedcore.config import DATA
from fedcore.data.catalog import catalog


def connect() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB connection with one view per existing catalog dataset.

    Locked down after the views exist: files are readable only under ``core/data`` (where the
    views point), nothing can be written, attached or installed, and the settings cannot be
    changed. The SQL console and CLI pass user text straight to this connection.
    """
    con = duckdb.connect(":memory:")
    for ds in catalog().values():
        if not ds.exists:
            continue
        path = ds.file.as_posix().replace("'", "''")
        if ds.is_parquet:
            source = f"{path}/*.parquet" if ds.file.is_dir() else path
            con.execute(f'CREATE VIEW "{ds.name}" AS SELECT * FROM read_parquet(\'{source}\')')
            continue
        nulls = ", ".join(f"'{v}'" for v in ("", *ds.na_values))
        con.execute(
            f'CREATE VIEW "{ds.name}" AS SELECT * FROM read_csv(\'{path}\', header=true, '
            f"nullstr=[{nulls}], sample_size=-1)"
        )
    con.execute(f"SET allowed_directories=['{DATA.as_posix().replace(chr(39), chr(39) * 2)}']")
    con.execute("SET enable_external_access=false")
    con.execute("SET lock_configuration=true")
    return con


def query(sql: str, params: list | None = None) -> pd.DataFrame:
    with connect() as con:
        return con.execute(sql, params or []).df()


def tables() -> list[str]:
    with connect() as con:
        return [r[0] for r in con.execute("SELECT view_name FROM duckdb_views() WHERE NOT internal ORDER BY 1").fetchall()]
