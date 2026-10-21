"""Gurkaynak-Sack-Wright daily Treasury zero-coupon curve (Federal Reserve Board, feds200628).

    python -m fedcore.ingest.gsw pull     # download raw/frb_gsw/feds200628.csv, then parse
    python -m fedcore.ingest.gsw parse    # re-parse the saved raw file only

The raw file is kept as downloaded. The parsed table keeps the date and the zero-coupon yields
at the maturities Q3 uses, in percent, continuously compounded. Days the Board left blank stay NaN.
"""
from __future__ import annotations

import argparse
import io

import pandas as pd

from fedcore.config import INTERIM, RAW

URL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"
RAW_FILE = RAW / "frb_gsw" / "feds200628.csv"
OUT = INTERIM / "frb_gsw" / "gsw_yields.csv"
MATURITIES = (1, 2, 5, 10, 30)
USER_AGENT = "fedcore-research/0.1 (academic research project)"


def parse(text: str) -> pd.DataFrame:
    """Date and SVENY01..SVENY30 at ``MATURITIES`` from the published CSV (notes above the header)."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("Date,"))
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])), na_values=["NA", "-999.99"])
    cols = [f"SVENY{m:02d}" for m in MATURITIES]
    out = df[["Date"] + cols].copy()
    out["Date"] = pd.to_datetime(out["Date"])
    return out.sort_values("Date").reset_index(drop=True)


def pull() -> pd.DataFrame:
    import requests

    resp = requests.get(URL, headers={"User-Agent": USER_AGENT}, timeout=120)
    resp.raise_for_status()
    RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    RAW_FILE.write_bytes(resp.content)
    return write()


def write() -> pd.DataFrame:
    out = parse(RAW_FILE.read_text(encoding="utf-8"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, date_format="%Y-%m-%d")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m fedcore.ingest.gsw")
    parser.add_argument("command", choices=["pull", "parse"])
    args = parser.parse_args(argv)
    out = pull() if args.command == "pull" else write()
    print(f"{len(out)} days, {out['Date'].min().date()} to {out['Date'].max().date()} -> {OUT}")


if __name__ == "__main__":
    main()
