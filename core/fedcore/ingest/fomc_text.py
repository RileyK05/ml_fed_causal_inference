"""FOMC statement texts from federalreserve.gov, 1994 onward.

    python -m fedcore.ingest.fomc_text pull     # fetch index pages and statements, then parse
    python -m fedcore.ingest.fomc_text parse    # re-parse the saved HTML only

Statement links come from the yearly historical pages (``fomchistoricalYYYY.htm``, through the
last year the Board has archived) and from the current calendar page for later years. Each
statement's HTML is saved untouched under ``raw/fomc_statements/``; the parsed table has one row
per statement: the release date (from the URL), the URL and the plain text. Several statements on
one date (for example an intermeeting action and a scheduled one) are separate rows.

Statements before 1999 exist only for policy changes. Matching to the USMPD events is left to the
user of the table: unscheduled releases are kept here and flagged nowhere (D1 keeps them out of
the pooled sample downstream).
"""
from __future__ import annotations

import argparse
import re
import time

import pandas as pd

from fedcore.config import INTERIM, RAW

BASE = "https://www.federalreserve.gov"
HISTORICAL = BASE + "/monetarypolicy/fomchistorical{year}.htm"
CALENDAR = BASE + "/monetarypolicy/fomccalendars.htm"
RAW_DIR = RAW / "fomc_statements"
OUT = INTERIM / "fomc" / "fomc_statement_text.csv"
FIRST_YEAR = 1994
USER_AGENT = "fedcore-research/0.1 (academic research project)"
PAUSE = 1.0  # seconds between requests
DATE_IN_URL = re.compile(r"(19|20)\d{6}")
CALENDAR_LINK = re.compile(r'href="(/newsevents/pressreleases/monetary(\d{8})a\.htm)"')


def statement_links(html: str) -> list[str]:
    """Hrefs of anchors whose text is exactly 'Statement' on a historical-year page."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    return [a["href"] for a in soup.find_all("a", href=True) if a.get_text(strip=True) == "Statement"]


def calendar_links(html: str) -> list[str]:
    return sorted({m.group(1) for m in CALENDAR_LINK.finditer(html)})


def release_date(url: str) -> pd.Timestamp:
    m = DATE_IN_URL.search(url)
    if m is None:
        raise ValueError(f"no date in {url}")
    return pd.Timestamp(m.group(0))


def statement_text(html: str) -> str:
    """Plain text of a statement page: the article block when there is one, else the body,
    without scripts, navigation and the page footer."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()
    node = soup.find(id="article") or soup.body or soup
    lines = [ln.strip() for ln in node.get_text("\n").splitlines()]
    lines = [ln for ln in lines if ln]
    # Old boarddocs pages end with site links and an update stamp.
    cut = next((i for i, ln in enumerate(lines) if ln.startswith(("Home |", "Last update:", "Last Update:"))), len(lines))
    return " ".join(lines[:cut]).replace("\xa0", " ")


def raw_name(url: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", url.strip("/")) + ".html"


def _get(session, url: str) -> str:
    resp = session.get(url if url.startswith("http") else BASE + url, timeout=60)
    resp.raise_for_status()
    time.sleep(PAUSE)
    return resp.text


def pull(last_year: int | None = None) -> pd.DataFrame:
    import requests

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    links: list[str] = []
    year = FIRST_YEAR
    while last_year is None or year <= last_year:
        resp = session.get(HISTORICAL.format(year=year), timeout=60)
        time.sleep(PAUSE)
        if resp.status_code == 404:
            break  # not archived yet: later years are on the calendar page
        resp.raise_for_status()
        links += statement_links(resp.text)
        year += 1
    links += [u for u in calendar_links(_get(session, CALENDAR)) if release_date(u).year >= year]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for url in dict.fromkeys(links):
        path = RAW_DIR / raw_name(url)
        if not path.exists():
            path.write_text(_get(session, url), encoding="utf-8")
    return write()


def write() -> pd.DataFrame:
    rows = []
    for path in sorted(RAW_DIR.glob("*.html")):
        text = statement_text(path.read_text(encoding="utf-8"))
        rows.append({"date": release_date(path.stem), "source_file": path.name, "text": text,
                     "n_words": len(text.split())})
    out = pd.DataFrame(rows).sort_values(["date", "source_file"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, date_format="%Y-%m-%d")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m fedcore.ingest.fomc_text")
    parser.add_argument("command", choices=["pull", "parse"])
    args = parser.parse_args(argv)
    out = pull() if args.command == "pull" else write()
    print(f"{len(out)} statements, {out['date'].min().date()} to {out['date'].max().date()} -> {OUT}")


if __name__ == "__main__":
    main()
