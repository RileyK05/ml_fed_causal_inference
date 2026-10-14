"""Run the cheap tests from docs/encoder/cheap-test-menu.md.

    python -m fedenc.cheap list
    python -m fedenc.cheap run --batch overnight --dry-run    # print the plan, load nothing
    python -m fedenc.cheap run --batch overnight              # A, B, C, R and S checks
    python -m fedenc.cheap run --ids A1,R1

Options: ``--synthetic`` uses a small simulated stand-in instead of the WRDS panel (smoke
run), ``--no-save`` prints metrics without writing runs, ``--quick`` cuts bootstrap and
permutation budgets. Fitted checks (C, R, S) are training under AGENTS.md: run them only
after the user's go-ahead. Every check is saved as an exploratory Q3 run in
``encoder/results/``; a final ``cheap-batch`` run records status and timing for all of them.
A failing check is logged and the batch moves on.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import sys
import time
import traceback

import pandas as pd

from fedcore.q3.cheap import REGISTRY, Config, load_real, synthetic_data
from fedcore.q3.evaluate import _jsonable
from fedcore.questions import specs as question_specs
from fedcore.results import ResultStore

import fedenc.cheap_gradients  # noqa: F401  (registers B1-B4)
from fedenc.config import RESULTS

BATCHES = {
    "overnight": ("A", "B", "C", "R", "S"),
    "no_fit": ("A", "B"),
    "go_no_go": ("R1", "C1", "R2", "R9"),
}


def _order(check_id: str) -> tuple:
    return (check_id[0], int(check_id[1:]))


def select(ids: str | None = None, batch: str | None = None) -> list[str]:
    if bool(ids) == bool(batch):
        raise SystemExit("pass exactly one of --ids or --batch")
    if batch:
        if batch not in BATCHES:
            raise SystemExit(f"unknown batch {batch!r}; choose from {sorted(BATCHES)}")
        keys = BATCHES[batch]
        chosen = [c for c in REGISTRY if c in keys or c[0] in keys]
    else:
        chosen = [c.strip().upper() for c in ids.split(",") if c.strip()]
        unknown = [c for c in chosen if c not in REGISTRY]
        if unknown:
            raise SystemExit(f"unknown check ids: {unknown}")
    return sorted(set(chosen), key=_order)


def run(ids: list[str], *, cfg: Config, synthetic: bool = False, save: bool = True,
        store: ResultStore | None = None, data=None) -> pd.DataFrame:
    """Run the checks in order; return one status row per check."""
    store = store or ResultStore(RESULTS)
    spec = question_specs()["Q3"]
    status = []
    load_error = None
    for cid in ids:
        chk = REGISTRY[cid]
        t0 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] {cid} {chk.title} ...", flush=True)
        row = {"check": cid, "title": chk.title, "kind": chk.kind}
        try:
            if chk.uses_real and data is None:
                if load_error:
                    raise RuntimeError(f"data load failed earlier: {load_error}")
                print("  loading data (once) ...", flush=True)
                try:
                    data = synthetic_data() if synthetic else load_real()
                except Exception as exc:
                    load_error = f"{type(exc).__name__}: {exc}"
                    raise
            res = chk.fn(data if chk.uses_real else None, cfg)
            row.update(status="ok", seconds=round(time.time() - t0, 1))
            if save:
                source = ("synthetic" if synthetic else "real") if chk.uses_real else "simulated"
                run_ = store.save(
                    spec, f"cheap-{cid}", role="exploratory",
                    params=_jsonable({"check": cid, "title": chk.title, "kind": chk.kind, "data": source,
                                      "config": dataclasses.asdict(cfg), **res.params}),
                    metrics=_jsonable(res.metrics), tables=res.tables,
                    datasets=list(data.datasets) if chk.uses_real and data is not None else [],
                    notes=res.notes,
                )
                row["run_id"] = run_.run_id
            for k, v in list(res.metrics.items())[:6]:
                print(f"  {k} = {v}", flush=True)
        except Exception as exc:  # keep the batch going; record the failure
            row.update(status="failed", seconds=round(time.time() - t0, 1), error=f"{type(exc).__name__}: {exc}")
            print(traceback.format_exc(), flush=True)
        print(f"  -> {row['status']} in {row['seconds']}s", flush=True)
        status.append(row)
        gc.collect()
    table = pd.DataFrame(status)
    if save and len(table):
        store.save(spec, "cheap-batch", role="exploratory",
                   params=_jsonable({"checks": ids, "synthetic": synthetic, "config": dataclasses.asdict(cfg)}),
                   metrics={"n_ok": int((table["status"] == "ok").sum()), "n_failed": int((table["status"] != "ok").sum())},
                   tables={"status": table}, notes="Status and timing of one cheap-test batch.")
    return table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m fedenc.cheap")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    r = sub.add_parser("run")
    r.add_argument("--ids")
    r.add_argument("--batch")
    r.add_argument("--synthetic", action="store_true")
    r.add_argument("--no-save", action="store_true")
    r.add_argument("--quick", action="store_true")
    r.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "list":
        for cid in sorted(REGISTRY, key=_order):
            c = REGISTRY[cid]
            gate = "needs go-ahead" if c.needs_approval else "no fitting"
            print(f"{cid:4} {c.kind:12} {gate:15} {c.title}")
        return
    ids = select(args.ids, args.batch)
    cfg = Config(n_boot=200, n_perm=20) if args.quick else Config()
    print(f"checks: {', '.join(ids)}")
    print(f"data: {'synthetic stand-in' if args.synthetic else 'real WRDS panel'}; save: {not args.no_save}")
    if any(REGISTRY[c].needs_approval for c in ids):
        print("note: this batch includes fitted checks (training under AGENTS.md).")
    if args.dry_run:
        return
    table = run(ids, cfg=cfg, synthetic=args.synthetic, save=not args.no_save)
    print(table.to_string(index=False))
    if (table["status"] != "ok").any():
        sys.exit(1)


if __name__ == "__main__":
    main()
