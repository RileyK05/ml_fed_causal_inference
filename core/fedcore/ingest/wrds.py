"""Pull the Q3 firm panel inputs from WRDS into core/data/raw/wrds/.

    python -m fedcore.ingest.wrds discover --user <wrds username>
    python -m fedcore.ingest.wrds pull --user <wrds username>

Log in once yourself first so the password is saved to pgpass (this script never asks for it):

    python -c "import wrds; wrds.Connection(wrds_username='<name>').create_pgpass_file()"

Every dataset lists candidate (library, table) sources in preference order: the CRSP CIZ
"_v2" tables first (the legacy SIZ tables stopped updating after 2024), legacy second.
``discover`` reports which candidate exists with the needed columns; ``pull`` uses the
first one that does and fails before downloading anything if a dataset has none.

Raw files keep WRDS's native column names (raw is untouched downloads); renaming and joins
happen in interim. Universe: every PERMNO in the S&P 500 at any point since START.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from fedcore.config import RAW

OUT = RAW / "wrds"
# 252 trading days of history before the first USMPD statement (1994-02-04).
START = "1993-01-01"


@dataclass(frozen=True)
class Source:
    library: str
    table: str
    columns: tuple[str, ...]
    where: str = ""  # SQL predicate; {start} and {members} are filled in


@dataclass(frozen=True)
class Dataset:
    name: str
    description: str
    sources: tuple[Source, ...]
    by_year: str | None = None  # date column to chunk the pull by calendar year
    needs: tuple[str, ...] = field(default_factory=tuple)  # notes recorded in the manifest


def _members_sql(src: Source) -> str:
    """Server-side subquery: PERMNOs in the S&P 500 at any point on or after START."""
    end_col = src.columns[2]
    return f"SELECT DISTINCT permno FROM {src.library}.{src.table} WHERE {end_col} >= '{START}' OR {end_col} IS NULL"


SP500 = Dataset(
    "sp500_members",
    "Historical S&P 500 membership spells (permno, start, end).",
    (
        Source("crsp", "dsp500list_v2", ("permno", "mbrstartdt", "mbrenddt")),
        Source("crsp", "dsp500list", ("permno", "start", "ending")),
        Source("crsp", "msp500list", ("permno", "start", "ending")),
    ),
)

DATASETS = (
    SP500,
    Dataset(
        "crsp_daily",
        "CRSP daily stock file for S&P 500 members (returns, price, volume, shares).",
        (
            Source(
                "crsp", "dsf_v2",
                ("permno", "dlycaldt", "dlyret", "dlyretx", "dlyprc", "dlyvol", "shrout"),
                "dlycaldt >= '{start}' AND permno IN ({members})",
            ),
            Source(
                "crsp", "dsf",
                ("permno", "date", "ret", "retx", "prc", "vol", "shrout", "cfacpr", "cfacshr"),
                "date >= '{start}' AND permno IN ({members})",
            ),
        ),
        by_year="date",
        needs=("CIZ dlyret includes delisting returns; legacy dsf needs crsp_delist merged in",),
    ),
    Dataset(
        "crsp_delist",
        "CRSP delisting returns (needed only if crsp_daily came from legacy dsf).",
        (
            Source("crsp", "stkdelists", ("permno", "delistingdt", "delret", "delactiontype"),
                   "permno IN ({members})"),
            Source("crsp", "dsedelist", ("permno", "dlstdt", "dlret", "dlstcd"),
                   "permno IN ({members})"),
        ),
    ),
    Dataset(
        "ff_factors_daily",
        "Fama-French daily factors: market excess return and risk-free rate (market channel).",
        (Source("ff", "factors_daily", ("date", "mktrf", "smb", "hml", "rf"), "date >= '{start}'"),),
    ),
    Dataset(
        "ccm_link",
        "CRSP-Compustat link history (gvkey <-> permno with validity dates).",
        (
            Source("crsp_a_ccm", "ccmxpf_lnkhist",
                   ("gvkey", "lpermno", "linktype", "linkprim", "linkdt", "linkenddt"),
                   "linktype IN ('LU','LC') AND linkprim IN ('P','C') AND lpermno IN ({members})"),
            Source("crsp", "ccmxpf_lnkhist",
                   ("gvkey", "lpermno", "linktype", "linkprim", "linkdt", "linkenddt"),
                   "linktype IN ('LU','LC') AND linkprim IN ('P','C') AND lpermno IN ({members})"),
        ),
    ),
    Dataset(
        "comp_fundq",
        "Compustat quarterly fundamentals with report date rdq (the point-in-time join key).",
        (
            Source(
                "comp", "fundq",
                ("gvkey", "datadate", "rdq", "fyearq", "fqtr", "atq", "ltq", "ceqq", "cheq",
                 "dlttq", "dlcq", "saleq", "niq", "oibdpq", "cshoq", "prccq"),
                "indfmt = 'INDL' AND datafmt = 'STD' AND popsrc = 'D' AND consol = 'C' "
                "AND datadate >= '{start}' AND gvkey IN (SELECT gvkey FROM {link})",
            ),
        ),
    ),
    Dataset(
        "comp_gics",
        "Historical GICS codes with validity dates (point-in-time sector).",
        (
            Source("comp", "co_hgic",
                   ("gvkey", "gsector", "ggroup", "gind", "gsubind", "indfrom", "indthru"),
                   "gvkey IN (SELECT gvkey FROM {link})"),
        ),
    ),
)


def connect(user: str | None):
    try:
        import wrds
    except ImportError:
        sys.exit('wrds is not installed: pip install -e "core[ingest]" then pip install --no-deps wrds')
    user = user or os.environ.get("WRDS_USERNAME")
    if not user:
        sys.exit("pass --user <wrds username> or set WRDS_USERNAME")
    return wrds.Connection(wrds_username=user)


def _columns(db, src: Source) -> set[str] | None:
    try:
        if src.table not in db.list_tables(library=src.library):
            return None
        return set(db.describe_table(src.library, src.table)["name"].str.lower())
    except Exception:
        return None


def _readable(db, src: Source) -> bool:
    """A table can be listed but not subscribed; only a real SELECT proves access."""
    try:
        db.raw_sql(f"SELECT {', '.join(src.columns)} FROM {src.library}.{src.table} LIMIT 1")
        return True
    except Exception:
        return False


def resolve(db) -> dict[str, Source | None]:
    """First candidate per dataset that exists, has every needed column, and is readable."""
    chosen: dict[str, Source | None] = {}
    for ds in DATASETS:
        chosen[ds.name] = None
        for src in ds.sources:
            cols = _columns(db, src)
            if cols is None:
                status = "missing table"
            elif not set(src.columns) <= cols:
                status = f"missing columns {sorted(set(src.columns) - cols)}"
            elif not _readable(db, src):
                status = "no access (not in your subscription)"
            else:
                status = "ok"
            print(f"  {ds.name:18s} {src.library}.{src.table:18s} {status}")
            if status == "ok" and chosen[ds.name] is None:
                chosen[ds.name] = src
    return chosen


def _where(src: Source, chosen: dict[str, Source | None]) -> str:
    members = _members_sql(chosen["sp500_members"])
    link = chosen.get("ccm_link")
    link_sql = f"{link.library}.{link.table} WHERE {link.where.format(members=members)}" if link else "(SELECT NULL AS gvkey) x"
    return src.where.format(start=START, members=members, link=link_sql)


def pull(db, chosen: dict[str, Source | None]) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"pulled": date.today().isoformat(), "start": START, "datasets": {}}
    for ds in DATASETS:
        src = chosen[ds.name]
        if src is None:
            continue
        where = _where(src, chosen)
        base = f"SELECT {', '.join(src.columns)} FROM {src.library}.{src.table}"
        sql = f"{base} WHERE {where}" if where else base
        if ds.by_year:
            date_col = src.columns[1]
            folder = OUT / ds.name
            folder.mkdir(exist_ok=True)
            rows = 0
            for year in range(int(START[:4]), date.today().year + 1):
                part = db.raw_sql(f"{sql} AND {date_col} BETWEEN '{year}-01-01' AND '{year}-12-31'")
                if part.empty:  # the vintage ends before this year
                    continue
                part.to_parquet(folder / f"{year}.parquet", index=False)
                rows += len(part)
                print(f"  {ds.name} {year}: {len(part):,} rows", flush=True)
            path = f"{ds.name}/"
        else:
            frame = db.raw_sql(sql)
            frame.to_parquet(OUT / f"{ds.name}.parquet", index=False)
            rows = len(frame)
            path = f"{ds.name}.parquet"
            print(f"  {ds.name}: {rows:,} rows", flush=True)
        manifest["datasets"][ds.name] = {
            "path": path, "source": f"{src.library}.{src.table}", "rows": rows,
            "query": sql, "description": ds.description, "notes": list(ds.needs),
        }
    (OUT / "_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m fedcore.ingest.wrds")
    parser.add_argument("command", choices=("discover", "pull"))
    parser.add_argument("--user", help="WRDS username (or set WRDS_USERNAME)")
    args = parser.parse_args(argv)

    db = connect(args.user)
    try:
        print("Checking WRDS tables:")
        chosen = resolve(db)
        missing = [name for name, src in chosen.items() if src is None and name != "crsp_delist"]
        if missing:
            sys.exit(f"no usable source for: {', '.join(missing)}. Nothing was downloaded.")
        if args.command == "pull":
            pull(db, chosen)
            print(f"Done. Files in {OUT}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
