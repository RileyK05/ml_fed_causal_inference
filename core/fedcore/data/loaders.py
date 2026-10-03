"""Load registered datasets as DataFrames, plus coverage and integrity checks."""
from __future__ import annotations

import hashlib

import pandas as pd

from fedcore.data.catalog import Dataset, catalog, get


def load(name: str, start=None, end=None, columns: list[str] | None = None) -> pd.DataFrame:
    """Load a dataset by catalog name. The date column is parsed; start/end filter on it
    (inclusive). Nothing is filled or interpolated -- gaps stay NaN."""
    ds = get(name)
    df = pd.read_csv(ds.file, na_values=list(ds.na_values) or None)
    df[ds.date_column] = pd.to_datetime(df[ds.date_column])
    if start is not None:
        df = df[df[ds.date_column] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df[ds.date_column] <= pd.Timestamp(end)]
    if columns is not None:
        keep = [ds.date_column] + [c for c in columns if c != ds.date_column]
        df = df[keep]
    return df.sort_values(ds.date_column).reset_index(drop=True)


def fingerprint(name: str) -> str:
    """Short content hash of a dataset file, recorded with every result run so a result
    can always be traced back to the exact data it was computed on."""
    return hashlib.md5(get(name).file.read_bytes()).hexdigest()[:12]


def coverage(names: list[str] | None = None) -> pd.DataFrame:
    """One row per dataset: rows, date range, column count, overall missing share."""
    rows = []
    for ds in catalog().values():
        if names is not None and ds.name not in names:
            continue
        if not ds.exists:
            rows.append(dict(dataset=ds.name, layer=ds.layer, grain=ds.grain, exists=False))
            continue
        df = load(ds.name)
        d = df[ds.date_column]
        rows.append(dict(
            dataset=ds.name, layer=ds.layer, grain=ds.grain, exists=True,
            rows=len(df), columns=df.shape[1], start=d.min().date(), end=d.max().date(),
            missing_share=round(float(df.drop(columns=ds.date_column).isna().mean().mean()), 4),
        ))
    return pd.DataFrame(rows)


def check(ds: Dataset) -> list[str]:
    """Integrity problems for one dataset (empty list = passes)."""
    if not ds.exists:
        return [f"file missing: {ds.file}"]
    problems = []
    df = pd.read_csv(ds.file, na_values=list(ds.na_values) or None)
    for col in (ds.date_column, *ds.key):
        if col not in df.columns:
            problems.append(f"column {col!r} not in file")
    if problems:
        return problems
    if pd.to_datetime(df[ds.date_column], errors="coerce").isna().any():
        problems.append(f"unparseable or missing values in date column {ds.date_column!r}")
    dupes = int(df.duplicated(subset=list(ds.key)).sum())
    if dupes:
        problems.append(f"{dupes} duplicate rows on key {list(ds.key)}")
    return problems
