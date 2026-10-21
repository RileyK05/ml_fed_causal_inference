"""Parsers for the free outside sources: GSW curve, TNIC-3, FOMC statements, SEC EDGAR."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from fedcore.ingest import edgar, fomc_text, gsw, tnic


def test_gsw_parse_skips_notes_and_keeps_missing():
    text = ('"Note: not an official release."\n\nSeries,Compounding Convention,Mnemonic(s)\n'
            "Date,BETA0,SVENY01,SVENY02,SVENY05,SVENY10,SVENY30\n"
            "2001-01-03,1,5.1,5.2,5.3,5.4,NA\n2001-01-02,1,5.0,5.1,5.2,5.3,5.5\n")
    out = gsw.parse(text)
    assert list(out.columns) == ["Date", "SVENY01", "SVENY02", "SVENY05", "SVENY10", "SVENY30"]
    assert out["Date"].is_monotonic_increasing and np.isnan(out["SVENY30"].iloc[1])


def test_tnic_cut_keeps_universe_pairs_and_dates_them():
    raw = io.BytesIO(b"year\tgvkey1\tgvkey2\tscore\n1999\t1011\t1011\t1\n1999\t1011\t6282\t0.2\n"
                     b"1999\t1011\t9999\t0.3\n2000\t6282\t1011\t0.1\n")
    out = tnic.cut(raw, {"001011", "006282"})
    assert len(out) == 2 and set(out["gvkey1"]) == {"001011", "006282"}
    assert (out.loc[out["year"] == 1999, "usable_from"] == pd.Timestamp("2000-04-01")).all()


def test_fomc_statement_parsing():
    page = '<a href="/fomc/19940204default.htm">Statement</a><a href="/x.htm">Minutes</a>'
    assert fomc_text.statement_links(page) == ["/fomc/19940204default.htm"]
    cal = 'href="/newsevents/pressreleases/monetary20260128a.htm" href="/newsevents/pressreleases/monetary20260128a.htm"'
    assert fomc_text.calendar_links(cal) == ["/newsevents/pressreleases/monetary20260128a.htm"]
    assert fomc_text.release_date("fomc_19940204default_htm") == pd.Timestamp("1994-02-04")
    html = ("<html><body><script>x()</script><div id='article'><p>The Committee decided to raise"
            " the target.</p></div><p>Home | Press releases</p></body></html>")
    assert fomc_text.statement_text(html) == "The Committee decided to raise the target."


def test_edgar_section_extraction_skips_the_table_of_contents():
    body = ("Item 7A. Quantitative and Qualitative Disclosures About Market Risk 45 Item 8. Financial Statements 46 "
            + "filler " * 50
            + "Item 7A. Quantitative and Qualitative Disclosures About Market Risk. "
            + "A hypothetical 100 basis point increase in rates on our variable rate debt would raise interest "
            + "expense by $4 million. " * 10
            + "Item 8. Financial Statements and Supplementary Data")
    section, method = edgar.extract_section(body, "10-K")
    assert method == "item" and "hypothetical 100 basis point" in section and "filler" not in section


def test_edgar_no_change_10q_is_flagged():
    body = ("Item 3. Quantitative and Qualitative Disclosures About Market Risk. There have been no material "
            "changes in our market risk since our annual report. Item 4. Controls and Procedures")
    section, method = edgar.extract_section(body, "10-Q")
    m = edgar.risk_measures(section)
    assert method == "item" and m["no_change"]


def test_edgar_risk_measures():
    s = ("Approximately 35% of our total debt bears interest at variable rates. We use interest rate swaps. "
         "A 100 basis point change would affect interest expense. Our fixed rate notes are unaffected.")
    m = edgar.risk_measures(s)
    assert m["floating_share_stated"] == 0.35
    assert m["has_rate_shock_table"] and m["swap_per_k"] > 0 and m["fixed_per_k"] > 0 and not m["no_change"]


def test_edgar_filings_from_submissions():
    page = {"form": ["10-K", "4", "8-K", "10-Q"], "filingDate": ["2005-03-01", "2005-03-02", "2005-03-03", "1995-05-01"],
            "acceptanceDateTime": ["2005-03-01T16:30:00.000Z"] * 4, "accessionNumber": ["a", "b", "c", "d"],
            "primaryDocument": ["k.htm", "f.xml", "e.htm", ""], "items": ["", "", "2.02,9.01", ""]}
    out = edgar.filings_from_submissions("0000000001", [page])
    assert out["form"].tolist() == ["10-K", "8-K"]  # Form 4 dropped; 1995 is before FIRST_YEAR
    assert out.loc[out["form"] == "8-K", "items"].item() == "2.02,9.01"
    assert out["acceptance"].iloc[0] == pd.Timestamp("2005-03-01 11:30:00")
