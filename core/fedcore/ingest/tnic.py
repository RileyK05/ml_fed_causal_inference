"""Hoberg-Phillips TNIC-3 text-based industry network, cut to the Q3 firm universe.

    python -m fedcore.ingest.tnic pull     # download raw/hoberg_phillips/tnic3_data.zip, then cut
    python -m fedcore.ingest.tnic cut      # re-cut the saved zip only

Source: https://hobergphillips.tuck.dartmouth.edu/industryclass.htm (TNIC-3, free for research).
One row per (year, gvkey1, gvkey2) whose 10-K product descriptions are similar enough to be in
the same TNIC-3 industry; ``score`` is how far the pair's similarity clears the threshold.

Timing (README technical note 3): ``year`` is the fiscal year of the 10-K, not a release date,
and the file is not lagged. ``usable_from`` puts each year's links in force from April 1 of the
next calendar year, after the 10-K for a December year-end is filed. Fiscal years ending earlier
are usable sooner; this rule is conservative for them.

The cut keeps pairs where both firms are in the CRSP/Compustat link tables on disk, so every peer
has returns. It is gitignored: gvkeys index licensed Compustat data, and the raw zip is 151MB.
"""
from __future__ import annotations

import argparse
import io
import zipfile

import pandas as pd

from fedcore.config import INTERIM, RAW
from fedcore.data import catalog, load

URL = "https://hobergphillips.tuck.dartmouth.edu/idata/tnic3_data.zip"
RAW_ZIP = RAW / "hoberg_phillips" / "tnic3_data.zip"
OUT = INTERIM / "hoberg_phillips" / "tnic3.parquet"
USER_AGENT = "fedcore-research/0.1 (academic research project)"
LINK_TABLES = ("ccm_link", "ccm_link_top1500")


def gvkey_text(col: pd.Series) -> pd.Series:
    """Compustat's six-character zero-padded gvkey."""
    return col.astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(6)


def universe() -> set[str]:
    have = catalog()
    keys: set[str] = set()
    for name in LINK_TABLES:
        if name in have and have[name].exists:
            keys |= set(gvkey_text(load(name)["gvkey"]).dropna())
    if not keys:
        raise FileNotFoundError("no CCM link table on disk to define the universe")
    return keys


def cut(raw: io.BufferedIOBase, keep: set[str], chunksize: int = 5_000_000) -> pd.DataFrame:
    """Pairs with both gvkeys in ``keep``, self-pairs dropped, plus ``usable_from``."""
    parts = []
    for chunk in pd.read_csv(raw, sep="\t", dtype={"year": "int64", "gvkey1": "string", "gvkey2": "string",
                                                   "score": "float64"}, chunksize=chunksize):
        chunk["gvkey1"], chunk["gvkey2"] = gvkey_text(chunk["gvkey1"]), gvkey_text(chunk["gvkey2"])
        chunk = chunk[chunk["gvkey1"].isin(keep) & chunk["gvkey2"].isin(keep) & (chunk["gvkey1"] != chunk["gvkey2"])]
        parts.append(chunk)
    out = pd.concat(parts, ignore_index=True)
    out["usable_from"] = pd.to_datetime((out["year"] + 1).astype(str) + "-04-01")
    return out.sort_values(["year", "gvkey1", "score"], ascending=[True, True, False]).reset_index(drop=True)


def pull() -> pd.DataFrame:
    import requests

    RAW_ZIP.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(URL, headers={"User-Agent": USER_AGENT}, timeout=1200, stream=True) as resp:
        resp.raise_for_status()
        with open(RAW_ZIP, "wb") as fh:
            for block in resp.iter_content(1 << 20):
                fh.write(block)
    return write()


def write() -> pd.DataFrame:
    with zipfile.ZipFile(RAW_ZIP) as z, z.open("tnic3_data.txt") as fh:
        out = cut(fh, universe())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m fedcore.ingest.tnic")
    parser.add_argument("command", choices=["pull", "cut"])
    args = parser.parse_args(argv)
    out = pull() if args.command == "pull" else write()
    print(f"{len(out)} pairs, {out['gvkey1'].nunique()} firms, years {out['year'].min()}-{out['year'].max()} -> {OUT}")


if __name__ == "__main__":
    main()
