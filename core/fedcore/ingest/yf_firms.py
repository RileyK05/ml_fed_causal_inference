"""yfinance daily bars for the 2026 holdout extension (CRSP's annual vintage ends 2025-12-31).

    python -m fedcore.ingest.yf_firms pull

Universe: S&P 500 members on the last CRSP day that are still in today's Wikipedia
constituent list, matched by CRSP ticker (share-class exact), else by SEC CIK. Firms removed during 2026 and 2026 additions are not
covered: Wikipedia no longer publishes dated changes, and additions have no CRSP history.

Bars start a year before the CRSP end so the overlap can be checked against CRSP
(``fedcore.q3.real.compare_overlap``). Files land in ``core/data/raw/yfinance_firms/``,
gitignored because the permno mapping comes from licensed CRSP data. The Wikipedia snapshot
goes to ``core/data/raw/wikipedia/``.
"""
from __future__ import annotations

import argparse
import io
import json
import time
from datetime import date

import numpy as np
import pandas as pd

from fedcore.config import RAW
from fedcore.data import load

OUT = RAW / "yfinance_firms"
WIKI = RAW / "wikipedia"
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
USER_AGENT = "fedcore-research/0.1 (academic research project)"
START = "2025-01-01"
MARKET_TICKER = "SPY"
RETRIES = 3
RETRY_PAUSE = 20  # seconds


def wiki_constituents() -> pd.DataFrame:
    """Today's S&P 500 list from Wikipedia, saved as a dated raw snapshot."""
    import requests

    resp = requests.get(WIKI_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    table = next(t for t in pd.read_html(io.StringIO(resp.text)) if {"Symbol", "CIK"} <= set(t.columns))
    table["CIK"] = table["CIK"].astype(str).str.zfill(10)
    WIKI.mkdir(parents=True, exist_ok=True)
    table.to_csv(WIKI / "sp500_constituents.csv", index=False)
    return table


def holdout_universe(wiki: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(matched firms: permno, gvkey, cik, conm, ticker, match; unmatched end-of-vintage members).

    Pass 1 matches the CRSP ticker on the last CRSP day (share-class exact: GOOG vs GOOGL,
    BRK + class B -> BRK.B) to the Wikipedia symbol. Pass 2 matches the rest by SEC CIK when
    the CIK has a single Wikipedia symbol (catches ticker changes in 2026).
    """
    crsp_end = load("crsp_daily", columns=["permno"])["dlycaldt"].max()
    members = load("sp500_members")
    members["mbrenddt"] = pd.to_datetime(members["mbrenddt"])
    last = members.loc[(members["mbrstartdt"] <= crsp_end) & (members["mbrenddt"].fillna(crsp_end) >= crsp_end)]
    firms = last[["permno"]].astype("int64").drop_duplicates()

    link = load("ccm_link")
    link["linkenddt"] = pd.to_datetime(link["linkenddt"])
    link = link[(link["linkdt"] <= crsp_end) & (link["linkenddt"].fillna(crsp_end) >= crsp_end)]
    link = link.sort_values("linkprim", ascending=False).drop_duplicates("lpermno")  # P before C
    link = link.assign(permno=link["lpermno"].astype("int64"), gvkey=link["gvkey"].astype(str))
    company = pd.read_parquet(RAW / "wrds" / "comp_company.parquet").assign(
        gvkey=lambda d: d["gvkey"].astype(str), cik=lambda d: d["cik"].astype(str).str.zfill(10)
    )
    firms = firms.merge(link[["permno", "gvkey"]], on="permno", how="left").merge(
        company[["gvkey", "conm", "cik"]], on="gvkey", how="left"
    )

    names = pd.read_parquet(RAW / "wrds" / "crsp_names.parquet")
    names = names[pd.to_datetime(names["secinfoenddt"]) >= crsp_end].drop_duplicates("permno", keep="last")
    symbols = set(wiki["Symbol"])

    def crsp_symbol(row) -> str | None:
        for cand in (row["ticker"], f"{row['ticker']}.{row['shareclass']}", row["tradingsymbol"]):
            if isinstance(cand, str) and cand in symbols:
                return cand
        return None

    names["Symbol"] = names.apply(crsp_symbol, axis=1)
    firms = firms.merge(names[["permno", "Symbol"]], on="permno", how="left")
    firms["match"] = np.where(firms["Symbol"].notna(), "ticker", None)

    by_cik = wiki.groupby("CIK")["Symbol"].agg(lambda s: s.iloc[0] if len(s) == 1 else None).dropna()
    taken = set(firms["Symbol"].dropna())
    fill = firms["Symbol"].isna() & firms["cik"].isin(by_cik.index)
    firms.loc[fill, "Symbol"] = firms.loc[fill, "cik"].map(by_cik)
    firms.loc[fill, "match"] = "cik"
    firms.loc[fill & firms["Symbol"].isin(taken), ["Symbol", "match"]] = None  # never reuse a symbol

    firms["ticker"] = firms["Symbol"].str.replace(".", "-", regex=False)  # yfinance: BRK.B -> BRK-B
    matched = firms.dropna(subset=["ticker"]).drop_duplicates("permno")
    unmatched = firms[~firms["permno"].isin(matched["permno"])]
    cols = ["permno", "gvkey", "cik", "conm", "ticker", "match"]
    return matched[cols].reset_index(drop=True), unmatched.reset_index(drop=True)


def pull(start: str = START) -> dict:
    import yfinance as yf

    wiki = wiki_constituents()
    firms, unmatched = holdout_universe(wiki)
    def fetch(tickers: list[str]) -> pd.DataFrame:
        raw = yf.download(tickers, start=start, auto_adjust=False, progress=False, threads=True, group_by="column")
        return (
            raw[["Adj Close", "Close", "Volume"]].stack(level=1, future_stack=True)
            .rename_axis(["date", "ticker"]).reset_index()
            .rename(columns={"Adj Close": "adj_close", "Close": "close", "Volume": "volume"})
            .dropna(subset=["adj_close"])
        )

    wanted = sorted(set(firms["ticker"]) | {MARKET_TICKER})
    parts, missing = [], wanted
    for attempt in range(RETRIES + 1):  # Yahoo drops some tickers under load; retry just those
        if not missing:
            break
        if attempt:
            time.sleep(RETRY_PAUSE)
        got = fetch(missing if len(missing) > 1 else missing * 2)  # one ticker returns a flat frame
        parts.append(got)
        missing = sorted(set(missing) - set(got["ticker"]))
    bars = pd.concat(parts, ignore_index=True).drop_duplicates(["date", "ticker"])
    OUT.mkdir(parents=True, exist_ok=True)
    firm_bars = bars[bars["ticker"] != MARKET_TICKER].merge(firms[["ticker", "permno"]], on="ticker")
    firm_bars.to_parquet(OUT / "firm_daily.parquet", index=False)
    bars[bars["ticker"] == MARKET_TICKER].drop(columns="ticker").to_parquet(OUT / "spy_daily.parquet", index=False)
    firms.to_csv(OUT / "universe.csv", index=False)
    unmatched.to_csv(OUT / "unmatched.csv", index=False)
    got = set(firm_bars["ticker"])
    manifest = {
        "pulled": date.today().isoformat(), "start": start, "source": "Yahoo Finance via yfinance "
        + yf.__version__, "wikipedia": WIKI_URL, "n_end_members": int(len(firms) + len(unmatched)),
        "n_matched": int(len(firms)), "n_with_bars": len(got),
        "no_bars": sorted(set(firms["ticker"]) - got), "rows": int(len(firm_bars)),
        "last_date": str(pd.Timestamp(bars["date"].max()).date()),
    }
    (OUT / "_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m fedcore.ingest.yf_firms")
    parser.add_argument("command", choices=("pull",))
    parser.add_argument("--start", default=START)
    args = parser.parse_args(argv)
    print(json.dumps(pull(args.start), indent=2))


if __name__ == "__main__":
    main()
