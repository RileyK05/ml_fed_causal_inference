"""Command line entry point.

    fedci check                 validate every catalog dataset (exists, dates parse, keys unique)
    fedci catalog               coverage table for every dataset
    fedci sql "SELECT ..."      run SQL against the catalog views
    fedci run q1 [k=v ...]      run a question's pipeline, saving to results/
    fedci runs [q1]             list saved runs
    fedci app                   launch the research viewer
    fedci serve [--no-build] [--no-open] [--rebuild] [--port N]
                                build the web app if stale, serve API + web app, open browser
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import pandas as pd

from fedci.config import ROOT


def _print(df: pd.DataFrame) -> None:
    with pd.option_context("display.max_rows", 200, "display.max_columns", 50, "display.width", 200):
        print(df.to_string(index=False) if len(df) else "(no rows)")


def cmd_check(_) -> int:
    from fedci.data import catalog, check
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
    from fedci.data import coverage
    _print(coverage())
    return 0


def cmd_sql(a) -> int:
    from fedci.data import query
    _print(query(a.query))
    return 0


def cmd_run(a) -> int:
    from fedci.questions import pipeline
    from fedci.results import ResultStore
    params = dict(p.split("=", 1) for p in a.params)
    run = pipeline(a.question).run(ResultStore(), **params)
    if run is not None:
        print(f"saved {run.path}")
    return 0


def cmd_runs(a) -> int:
    from fedci.results import ResultStore
    _print(ResultStore().summary(a.question))
    return 0


def cmd_app(_) -> int:
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(ROOT / "src" / "frontend" / "streamlit" / "Home.py")])


def cmd_serve(a) -> int:
    from fedci.serve import launch
    return launch(host=a.host, port=a.port, build=not a.no_build,
                  open_browser=not a.no_open, rebuild=a.rebuild)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="fedci", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check").set_defaults(fn=cmd_check)
    sub.add_parser("catalog").set_defaults(fn=cmd_catalog)
    s = sub.add_parser("sql"); s.add_argument("query"); s.set_defaults(fn=cmd_sql)
    r = sub.add_parser("run"); r.add_argument("question"); r.add_argument("params", nargs="*", help="key=value"); r.set_defaults(fn=cmd_run)
    rs = sub.add_parser("runs"); rs.add_argument("question", nargs="?"); rs.set_defaults(fn=cmd_runs)
    sub.add_parser("app").set_defaults(fn=cmd_app)
    s = sub.add_parser("serve", help="build if needed, then serve API + web app")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--no-build", action="store_true", help="do not build the web app")
    s.add_argument("--no-open", action="store_true", help="do not open the browser")
    s.add_argument("--rebuild", action="store_true", help="force a web app rebuild")
    s.set_defaults(fn=cmd_serve)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
