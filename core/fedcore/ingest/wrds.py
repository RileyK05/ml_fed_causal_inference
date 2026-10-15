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

``--universe top1500`` pulls a broader universe into ``raw/wrds_top1500/`` instead: every
US common stock (NYSE/AMEX/Nasdaq) ranked in the top 1,500 by market cap at any month-end
since START. Only the tables the panel needs are pulled; the month-end ranks are saved as
``top1500_months`` so the universe is auditable. ``sp500_members`` and the factors stay in
``raw/wrds/``.

    python -m fedcore.ingest.wrds pull --user <wrds username> --universe top1500
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
    query: str = ""  # full SELECT instead of columns + where (the universe ranking)


@dataclass(frozen=True)
class Dataset:
    name: str
    description: str
    sources: tuple[Source, ...]
    by_year: str | None = None  # date column to chunk the pull by calendar year
    needs: tuple[str, ...] = field(default_factory=tuple)  # notes recorded in the manifest


def _members_sql(chosen: dict) -> str:
    """Server-side subquery: the universe's PERMNOs. S&P 500 members at any point on or after
    START, or every stock in the month-end top ranks when the universe is top1500."""
    top = chosen.get("top1500_months")
    if top is not None:
        return f"SELECT DISTINCT permno FROM ({top.query.format(start=START)}) universe"
    src = chosen["sp500_members"]
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
    Dataset(
        "crsp_names",
        "CRSP ticker history per permno (share-class exact; maps firms to current tickers).",
        (
            Source("crsp", "stksecurityinfohist",
                   ("permno", "secinfostartdt", "secinfoenddt", "ticker", "tradingsymbol", "shareclass"),
                   "permno IN ({members})"),
            Source("crsp", "dsenames", ("permno", "namedt", "nameendt", "ticker", "tsymbol"),
                   "permno IN ({members})"),
        ),
    ),
    Dataset(
        "comp_company",
        "Compustat company header: name and SEC CIK (CIK maps firms to current tickers).",
        (Source("comp", "company", ("gvkey", "conm", "cik"), "gvkey IN (SELECT gvkey FROM {link})"),),
    ),
)


TOP_N = 1500
_CIZ_COMMON = ("sharetype = 'NS' AND securitytype = 'EQTY' AND securitysubtype = 'COM' AND usincflg = 'Y' "
               "AND issuertype IN ('ACOR', 'CORP') AND primaryexch IN ('N', 'A', 'Q')")
TOP_MONTHS = Dataset(
    "top1500_months",
    f"Month-end market-cap rank (1 = largest) of US common stocks, top {TOP_N} only. Defines the universe.",
    (
        Source(
            "crsp", "msf_v2",
            ("permno", "mthcaldt", "mthprc", "shrout", "sharetype", "securitytype", "securitysubtype",
             "usincflg", "issuertype", "primaryexch"),
            query=(f"SELECT permno, mthcaldt, mktcap_rank FROM (SELECT permno, mthcaldt, RANK() OVER "
                   f"(PARTITION BY mthcaldt ORDER BY ABS(mthprc) * shrout DESC) AS mktcap_rank FROM crsp.msf_v2 "
                   f"WHERE mthcaldt >= '{{start}}' AND {_CIZ_COMMON} AND mthprc IS NOT NULL AND shrout > 0) ranked "
                   f"WHERE mktcap_rank <= {TOP_N}"),
        ),
    ),
)
_LINKED = "gvkey IN (SELECT gvkey FROM {link})"
CREDIT = (
    Dataset(
        "comp_ratings",
        "S&P ratings by month: long-term issuer, subordinated debt, short-term. Compustat stopped "
        "carrying S&P ratings in 2017.",
        (Source("comp", "adsprate", ("gvkey", "datadate", "splticrm", "spsdrm", "spsticrm"), _LINKED),),
    ),
    Dataset(
        "comp_fundq_cash",
        "Quarterly cash-flow and interest items (year-to-date cash flows) for funding-dependence measures.",
        (
            Source(
                "comp", "fundq",
                ("gvkey", "datadate", "rdq", "fyearq", "fqtr", "oancfy", "capxy", "xintq", "oiadpq", "dvy"),
                "indfmt = 'INDL' AND datafmt = 'STD' AND popsrc = 'D' AND consol = 'C' "
                "AND datadate >= '{start}' AND gvkey IN (SELECT gvkey FROM {link})",
            ),
        ),
    ),
)
_BY_NAME = {ds.name: ds for ds in DATASETS}
UNIVERSES = {
    "sp500": (OUT, DATASETS),
    "top1500": (RAW / "wrds_top1500", (TOP_MONTHS, *(_BY_NAME[n] for n in ("crsp_daily", "ccm_link",
                                                                           "comp_fundq", "comp_gics")),
                                       *CREDIT)),
}


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


def resolve(db, datasets=DATASETS) -> dict[str, Source | None]:
    """First candidate per dataset that exists, has every needed column, and is readable."""
    chosen: dict[str, Source | None] = {}
    for ds in datasets:
        chosen[ds.name] = None
        for src in ds.sources:
            cols = _columns(db, src)
            if cols is None:
                status = "missing table (or library not in your subscription)"
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
    members = _members_sql(chosen)
    link = chosen.get("ccm_link")
    link_sql = f"{link.library}.{link.table} WHERE {link.where.format(members=members)}" if link else "(SELECT NULL AS gvkey) x"
    return src.where.format(start=START, members=members, link=link_sql)


def _write_atomic(frame, path) -> None:
    """Write beside the target, then rename: a crash never leaves a half-written parquet part."""
    tmp = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def pull(db, chosen: dict[str, Source | None], only: set[str] | None = None, out=OUT,
         datasets=DATASETS, resume: bool = False) -> dict:
    """Download every dataset (or just ``only``) into ``out``; the manifest keeps earlier entries.

    ``resume`` keeps yearly parts already on disk (an interrupted pull) instead of re-downloading."""
    import pandas as pd

    out.mkdir(parents=True, exist_ok=True)
    manifest_file = out / "_manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.exists() else {"datasets": {}}
    manifest.update({"start": START})
    for ds in datasets:
        src = chosen[ds.name]
        if src is None or (only and ds.name not in only):
            continue
        if src.query:
            sql = src.query.format(start=START)
        else:
            where = _where(src, chosen)
            base = f"SELECT {', '.join(src.columns)} FROM {src.library}.{src.table}"
            sql = f"{base} WHERE {where}" if where else base
        if ds.by_year:
            date_col = src.columns[1]
            folder = out / ds.name
            folder.mkdir(exist_ok=True)
            rows = 0
            for year in range(int(START[:4]), date.today().year + 1):
                done = folder / f"{year}.parquet"
                if resume and done.exists():  # written atomically, so an existing part is complete
                    rows += len(pd.read_parquet(done, columns=[src.columns[0]]))
                    print(f"  {ds.name} {year}: kept from an earlier pull", flush=True)
                    continue
                part = db.raw_sql(f"{sql} AND {date_col} BETWEEN '{year}-01-01' AND '{year}-12-31'")
                if part.empty:  # the vintage ends before this year
                    continue
                _write_atomic(part, folder / f"{year}.parquet")
                rows += len(part)
                print(f"  {ds.name} {year}: {len(part):,} rows", flush=True)
            path = f"{ds.name}/"
        else:
            frame = db.raw_sql(sql)
            _write_atomic(frame, out / f"{ds.name}.parquet")
            rows = len(frame)
            path = f"{ds.name}.parquet"
            print(f"  {ds.name}: {rows:,} rows", flush=True)
        manifest["datasets"][ds.name] = {
            "pulled": date.today().isoformat(),
            "path": path, "source": f"{src.library}.{src.table}", "rows": rows,
            "query": sql, "description": ds.description, "notes": list(ds.needs),
        }
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m fedcore.ingest.wrds")
    parser.add_argument("command", choices=("discover", "pull"))
    parser.add_argument("--user", help="WRDS username (or set WRDS_USERNAME)")
    parser.add_argument("--only", help="comma-separated dataset names to pull (default: all)")
    parser.add_argument("--universe", choices=tuple(UNIVERSES), default="sp500")
    parser.add_argument("--resume", action="store_true", help="keep yearly parts already downloaded")
    args = parser.parse_args(argv)
    out, datasets = UNIVERSES[args.universe]
    only = set(args.only.split(",")) if args.only else None
    unknown = (only or set()) - {ds.name for ds in datasets}
    if unknown:
        sys.exit(f"unknown dataset(s): {sorted(unknown)}")

    db = connect(args.user)
    try:
        print("Checking WRDS tables:")
        chosen = resolve(db, datasets)
        missing = [name for name, src in chosen.items() if src is None and name != "crsp_delist"]
        if missing:
            sys.exit(f"no usable source for: {', '.join(missing)}. Nothing was downloaded.")
        unresolved = sorted(name for name in (only or ()) if chosen[name] is None)
        if unresolved:
            sys.exit(f"no usable source for requested: {', '.join(unresolved)}. Nothing was downloaded.")
        if args.command == "pull":
            pull(db, chosen, only, out, datasets, resume=args.resume)
            print(f"Done. Files in {out}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
