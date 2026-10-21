"""SEC EDGAR: filing index, 8-K item codes, and the market-risk sections of 10-Ks and 10-Qs.

    python -m fedcore.ingest.edgar index    --user-agent "Name email"   # submissions JSON per firm
    python -m fedcore.ingest.edgar sections --user-agent "Name email"   # 10-K Item 7A, 10-Q Part I Item 3
    python -m fedcore.ingest.edgar parse                                 # rebuild tables from saved files

The SEC requires a User-Agent naming the requester and a contact address, and allows at most 10
requests per second; this module sends ``RATE`` per second. Nothing is fetched without
``--user-agent`` (or ``SEC_USER_AGENT``).

Universe: CIKs of the Compustat firms on disk (``comp_company``). Compustat carries the current
CIK; EDGAR files a company's history under one CIK, so this covers the history in nearly all cases.

Outputs (``interim/edgar/``, gitignored with the raw cache because the CIK list comes from
licensed Compustat):

- ``filings.parquet``: one row per 10-K, 10-Q (and amendments) and 8-K: cik, form, filing date,
  acceptance time, accession, primary document, and the 8-K item codes the SEC records
  (structured codes such as 2.02 start in August 2004; earlier 8-Ks have none).
- ``market_risk.parquet``: one row per 10-K or 10-Q fetched, with the extracted market-risk
  section, how it was found (``method``: ``item``, ``item_short``, ``reference`` when the item
  points to an MD&A passage, ``rate_sentences`` when there is no item and the filing's
  interest-rate and debt sentences are kept instead, ``none``), and simple text measures.

Point in time: a filing is usable from the trading day after its acceptance time. The market-risk
item was phased in over 1997-1998; earlier filings have none and stay missing. A 10-Q that says
"no material change" is recorded as ``no_change = True``, not as a copy of the 10-K.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from fedcore.config import INTERIM, RAW

RAW_DIR = RAW / "edgar"
OUT_DIR = INTERIM / "edgar"
SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
SUBMISSIONS_PAGE = "https://data.sec.gov/submissions/{name}"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
RATE = 8.0  # requests per second (SEC limit: 10)
WORKERS = 6
FORMS = ("10-K", "10-K405", "10-KSB", "10-K/A", "10-Q", "10-Q/A", "8-K", "8-K/A")
PERIODIC = ("10-K", "10-K405", "10-KSB", "10-Q")
FIRST_YEAR = 1996
SECTION_WORDS = 20_000  # cap on stored section length
FALLBACK_WORDS = 3_000  # cap on the rate-sentence fallback

# Section headings. Filings vary ("Item 7A.", "ITEM 7(a)", "Item 3 -"), so match loosely.
_ITEM_10K_START = re.compile(r"item\s*7\s*\(?a\)?[\s.:\-–—]*quantitative\s+and\s+qualitative\s+disclosures?",
                             re.I)
_ITEM_10K_END = re.compile(r"item\s*8[\s.:\-–—]*(consolidated\s+)?financial\s+statements", re.I)
_ITEM_10Q_START = re.compile(r"item\s*3[\s.:\-–—]*quantitative\s+and\s+qualitative\s+disclosures?", re.I)
_ITEM_10Q_END = re.compile(r"item\s*4[\s.:\-–—]*(controls|submission)", re.I)
_MARKET_RISK_HEADING = re.compile(r"(interest\s+rate\s+risk|market\s+risk)", re.I)
_NO_CHANGE = re.compile(r"(no|not\s+been\s+any|were\s+no)\s+(material|significant)\s+changes?", re.I)
_SENTENCE_SPLIT = re.compile(r"(?<=[.;])\s+(?=[A-Z(])")
_RATE_SENTENCE = re.compile(r"interest\s+rate|variable[-\s]+rate|floating[-\s]+rate|fixed[-\s]+rate|basis\s+points?|"
                            r"interest\s+expense|swap|libor|sofr|commercial\s+paper|credit\s+facilit|matur", re.I)


class Fetcher:
    """Rate-limited GET with the SEC's required User-Agent."""

    def __init__(self, user_agent: str):
        import requests

        if not user_agent or "@" not in user_agent:
            raise SystemExit("SEC requires a User-Agent with a contact email: --user-agent 'Name email@domain'")
        self.headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
        self.local = threading.local()  # one session per thread
        self.lock = threading.Lock()
        self.last = 0.0

    def _session(self):
        import requests

        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
            self.local.session.headers.update(self.headers)
        return self.local.session

    def _wait_turn(self) -> None:
        """Global rate limit shared by all threads."""
        with self.lock:
            wait = 1.0 / RATE - (time.monotonic() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.monotonic()

    def get(self, url: str) -> bytes | None:
        import requests

        for attempt in range(6):
            self._wait_turn()
            try:
                resp = self._session().get(url, timeout=60)
            except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError):
                time.sleep(5 * (attempt + 1))  # dropped connection: back off and retry
                continue
            if resp.status_code == 404:
                return None
            if resp.status_code in (429, 503):
                time.sleep(10 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.content
        raise RuntimeError(f"gave up on {url}")


def cik_text(col: pd.Series) -> pd.Series:
    return col.astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(10)


def universe() -> pd.DataFrame:
    """gvkey, CIK and name of every Compustat firm on disk (the WRDS pull's company table)."""
    co = pd.read_parquet(RAW / "wrds" / "comp_company.parquet").dropna(subset=["cik"])
    return co.assign(cik=cik_text(co["cik"]))[["gvkey", "cik", "conm"]].drop_duplicates("cik")


# ----------------------------------------------------------------------------- filing index

def _table(block: dict) -> pd.DataFrame:
    cols = ["form", "filingDate", "acceptanceDateTime", "accessionNumber", "primaryDocument", "items"]
    return pd.DataFrame({c: block.get(c, [None] * len(block.get("form", []))) for c in cols})


def filings_from_submissions(cik: str, pages: list[dict]) -> pd.DataFrame:
    """All filings in a firm's submissions JSON (main file's ``recent`` block plus older pages)."""
    frames = [_table(p) for p in pages]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        return df
    df = df[df["form"].isin(FORMS)].copy()
    df.insert(0, "cik", cik)
    df["filing_date"] = pd.to_datetime(df.pop("filingDate"))
    df["acceptance"] = pd.to_datetime(df.pop("acceptanceDateTime"), errors="coerce", utc=True).dt.tz_convert(
        "America/New_York").dt.tz_localize(None)
    df = df.rename(columns={"accessionNumber": "accession", "primaryDocument": "primary_doc"})
    df["items"] = df["items"].fillna("").astype(str)
    return df[df["filing_date"].dt.year >= FIRST_YEAR].reset_index(drop=True)


def index(fetcher: Fetcher, ciks: list[str]) -> pd.DataFrame:
    sub_dir = RAW_DIR / "submissions"
    sub_dir.mkdir(parents=True, exist_ok=True)
    for k, cik in enumerate(ciks):
        main_file = sub_dir / f"CIK{cik}.json"
        if not main_file.exists():
            body = fetcher.get(SUBMISSIONS.format(cik=cik))
            if body is None:
                continue
            main_file.write_bytes(body)
        main = json.loads(main_file.read_text(encoding="utf-8"))
        for page in main.get("filings", {}).get("files", []):
            path = sub_dir / page["name"]
            if not path.exists():
                body = fetcher.get(SUBMISSIONS_PAGE.format(name=page["name"]))
                if body is not None:
                    path.write_bytes(body)
        if k % 100 == 0:
            print(f"  submissions {k}/{len(ciks)}", flush=True)
    return write_index()


def write_index() -> pd.DataFrame:
    sub_dir = RAW_DIR / "submissions"
    frames = []
    for main_file in sorted(sub_dir.glob("CIK??????????.json")):
        cik = main_file.stem[3:]
        main = json.loads(main_file.read_text(encoding="utf-8"))
        pages = [main.get("filings", {}).get("recent", {})]
        for page in main.get("filings", {}).get("files", []):
            path = sub_dir / page["name"]
            if path.exists():
                pages.append(json.loads(path.read_text(encoding="utf-8")))
        frames.append(filings_from_submissions(cik, pages))
    out = pd.concat(frames, ignore_index=True).drop_duplicates("accession")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_DIR / "filings.parquet", index=False)
    return out


# ----------------------------------------------------------------------------- sections

def to_text(raw: bytes) -> str:
    """Plain text of an HTML or SGML/plain-text filing document (lxml; text nodes joined by spaces)."""
    s = raw.decode("utf-8", errors="replace")
    if re.search(r"<(html|body|div|p|table)", s[:20000], re.I):
        import lxml.html

        try:
            doc = lxml.html.document_fromstring(re.sub(r"^\s*<\?xml[^>]*>", "", s))
            for bad in doc.xpath("//script|//style"):
                bad.drop_tree()
            s = " ".join(doc.itertext())
        except (ValueError, lxml.etree.ParserError):
            pass
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"&(nbsp|#160|#xa0);", " ", s, flags=re.I)
    return re.sub(r"\s+", " ", s.replace(" ", " ")).strip()


def extract_section(text: str, form: str) -> tuple[str, str]:
    """(section text, method). The table of contents also names the item, so take the start
    match that is followed by the longest stretch before the next item; fall back to the
    paragraph under a 'market risk' heading when the item only points elsewhere."""
    quarterly = form.startswith("10-Q")
    start, end = (_ITEM_10Q_START, _ITEM_10Q_END) if quarterly else (_ITEM_10K_START, _ITEM_10K_END)
    best, best_at = "", -1
    for m in start.finditer(text):
        e = end.search(text, m.end())
        chunk = text[m.start(): e.start() if e else m.start() + 60_000]
        if len(chunk) > len(best):
            best, best_at = chunk, m.start()
    words = best.split()
    if len(words) >= 60 or (best and _NO_CHANGE.search(best)):
        return " ".join(words[:SECTION_WORDS]), "item"
    if len(words) < 60:  # a pointer ("see MD&A", "provided under the caption") or nothing
        for m in _MARKET_RISK_HEADING.finditer(text):
            if best_at <= m.start() < best_at + len(best):
                continue  # the pointer itself, not the passage it points to
            chunk = " ".join(text[m.start(): m.start() + 15_000].split()[:1500])
            if re.search(r"basis\s+point|variable|floating|hypothetical|\d+\s*%", chunk, re.I):
                return chunk, "reference"
    # No usable item (folded into MD&A, a bare pointer, or omitted from 10-Qs): keep the
    # filing's interest-rate and debt sentences instead, in order.
    picked = [x for x in _SENTENCE_SPLIT.split(text) if _RATE_SENTENCE.search(x)]
    picked = " ".join(" ".join(picked).split()[:FALLBACK_WORDS])
    if picked:
        return picked, "rate_sentences"
    return (" ".join(words), "item_short") if words else ("", "none")


_FLOAT = re.compile(r"\b(variable|floating|adjustable)[-\s]+rate", re.I)
_FIXED = re.compile(r"\bfixed[-\s]+rate", re.I)
_SWAP = re.compile(r"interest[-\s]+rate\s+(swap|cap|collar|hedg)", re.I)
_SHOCK = re.compile(r"(\d{2,3})\s*(basis[-\s]+points?|bps?)\b|\b(1|one|10|ten)\s*(%|percent(age)?\s+point)", re.I)
_FLOAT_SHARE = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%\s+of\s+(?:our|the\s+company.s|its|total)?\s*(?:total\s+)?(?:outstanding\s+)?"
                          r"(?:debt|borrowings|indebtedness)[^.]{0,80}?(?:variable|floating)", re.I)


def risk_measures(section: str) -> dict:
    """Text measures of rate exposure in a market-risk section. Counts are per 1,000 words."""
    words = max(len(section.split()), 1)
    share = _FLOAT_SHARE.search(section)
    return {
        "n_words": len(section.split()),
        "no_change": bool(_NO_CHANGE.search(section)) and len(section.split()) < 400,
        "floating_per_k": 1000.0 * len(_FLOAT.findall(section)) / words,
        "fixed_per_k": 1000.0 * len(_FIXED.findall(section)) / words,
        "swap_per_k": 1000.0 * len(_SWAP.findall(section)) / words,
        "has_rate_shock_table": bool(_SHOCK.search(section)),
        "floating_share_stated": float(share.group(1)) / 100.0 if share else float("nan"),
    }


def _doc_url(cik: str, accession: str, primary_doc: str | None) -> str:
    acc = accession.replace("-", "")
    doc = primary_doc if primary_doc else f"{accession}.txt"
    return ARCHIVE.format(cik=int(cik), acc=acc, doc=doc)


def panel_windows(lead_days: int = 400) -> pd.DataFrame:
    """Per CIK, the span of filing dates a Q3 panel row could use: from ``lead_days`` before the
    firm's first panel meeting to its last one."""
    from fedcore.q3.real import load_real_panel

    _, rows, _ = load_real_panel()
    co = universe()
    r = rows.dropna(subset=["gvkey"]).assign(meeting=lambda d: pd.to_datetime(d["meeting"]),
                                             gvkey=lambda d: d["gvkey"].astype(str))
    r = r.merge(co[["gvkey", "cik"]], on="gvkey")
    span = r.groupby("cik")["meeting"].agg(["min", "max"])
    span["min"] -= pd.Timedelta(days=lead_days)
    return span.rename(columns={"min": "first", "max": "last"})


def sections(fetcher: Fetcher, start_year: int = 1997, panel_only: bool = True) -> pd.DataFrame:
    """Fetch every periodic filing not yet in the section store; the store is append-only so
    the job can stop and resume. ``panel_only`` keeps filings a panel row could use."""
    filings = pd.read_parquet(OUT_DIR / "filings.parquet")
    todo = filings[filings["form"].isin(PERIODIC) & (filings["filing_date"].dt.year >= start_year)]
    if panel_only:
        span = panel_windows()
        todo = todo.join(span, on="cik", how="inner")
        todo = todo[(todo["filing_date"] >= todo["first"]) & (todo["filing_date"] <= todo["last"])]
    store = OUT_DIR / "market_risk_parts"
    store.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in store.glob("*.json")}
    todo = todo[~todo["accession"].isin(done)]
    print(f"  {len(todo)} filings to fetch", flush=True)
    def one(f) -> None:
        body = fetcher.get(_doc_url(f.cik, f.accession, f.primary_doc))
        if body is None and f.primary_doc:
            body = fetcher.get(_doc_url(f.cik, f.accession, None))
        section, method = extract_section(to_text(body), f.form) if body else ("", "missing")
        rec = {"accession": f.accession, "cik": f.cik, "form": f.form,
               "filing_date": str(f.filing_date.date()), "acceptance": str(f.acceptance),
               "method": method, "section": section}
        tmp = store / f"{f.accession}.json.part"
        tmp.write_text(json.dumps(rec), encoding="utf-8")
        tmp.replace(store / f"{f.accession}.json")  # atomic: an interrupted write never looks done

    # Several threads overlap download latency and parsing; the shared limiter keeps the
    # combined request rate at RATE per second.
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for k, _ in enumerate(pool.map(one, todo.itertuples(index=False))):
            if k % 500 == 0:
                print(f"  sections {k}/{len(todo)}", flush=True)
    return write_sections()


def write_sections() -> pd.DataFrame:
    store = OUT_DIR / "market_risk_parts"
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(store.glob("*.json"))]
    out = pd.DataFrame(recs)
    out = pd.concat([out, pd.DataFrame([risk_measures(s) for s in out["section"]])], axis=1)
    out["filing_date"] = pd.to_datetime(out["filing_date"])
    out.to_parquet(OUT_DIR / "market_risk.parquet", index=False)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m fedcore.ingest.edgar")
    parser.add_argument("command", choices=["index", "sections", "parse"])
    parser.add_argument("--user-agent", default=os.environ.get("SEC_USER_AGENT"))
    parser.add_argument("--start-year", type=int, default=1997)
    args = parser.parse_args(argv)
    if args.command == "parse":
        write_index()
        out = write_sections() if (OUT_DIR / "market_risk_parts").exists() else None
    else:
        fetcher = Fetcher(args.user_agent)
        if args.command == "index":
            out = index(fetcher, universe()["cik"].tolist())
        else:
            out = sections(fetcher, args.start_year)
    if out is not None:
        print(f"{len(out)} rows")


if __name__ == "__main__":
    main()
