"""Command line entry point for the data layer.

    fedcore check               validate every catalog dataset (exists, dates parse, keys unique)
    fedcore catalog             coverage table for every dataset
    fedcore sql "SELECT ..."    run SQL against the catalog views
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd


def _print(df: pd.DataFrame) -> None:
    with pd.option_context("display.max_rows", 200, "display.max_columns", 50, "display.width", 200):
        print(df.to_string(index=False) if len(df) else "(no rows)")


def cmd_check(_) -> int:
    from fedcore.data import catalog, check
    bad = 0
    for ds in catalog().values():
        problems = check(ds)
        print(f"{'FAIL' if problems else 'ok  '}  {ds.name:28s} {ds.path}")
        for p in problems:
            print(f"        - {p}")
        bad += bool(problems)
    print(f"\n{len(catalog()) - bad}/{len(catalog())} datasets pass")
    return 1 if bad else 0


def cmd_catalog(_) -> int:
    from fedcore.data import coverage
    _print(coverage())
    return 0


def cmd_sql(a) -> int:
    from fedcore.data import query
    _print(query(a.query))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="fedcore", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check").set_defaults(fn=cmd_check)
    sub.add_parser("catalog").set_defaults(fn=cmd_catalog)
    s = sub.add_parser("sql"); s.add_argument("query"); s.set_defaults(fn=cmd_sql)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
