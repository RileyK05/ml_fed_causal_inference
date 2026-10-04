# Data provenance

> **Layout note (2026-09-24).** This log was written against the original `eeif_data/` layout. Files have
> since moved (contents unchanged, byte-for-byte). The one-off build scripts (`build_*.py`) mentioned below
> were retired; the data they produced is kept as-is. Every dataset is registered in `data/catalog.yaml`.
>
> | Old location | New location |
> |---|---|
> | `01_fomc_dates/fomc_dates.csv` | `interim/fomc/fomc_dates.csv` |
> | `01_fomc_dates/fomc_bloomberg_pull_list.csv` | `interim/fomc/fomc_meetings.csv` (renamed; Bloomberg pull replaced by USMPD) |
> | `01_fomc_dates/raw_fomccalendars*.html` | `raw/fomc_calendar/` |
> | `02_sector_etf_returns/*.csv` | `interim/etf/` |
> | `03_fred_controls/fred_controls.csv` | `interim/fred/` |
> | `04_cleveland_fed_inflation_expectations/inflation-expectations.xlsx` | `raw/cleveland_fed/` |
> | `04_cleveland_fed_inflation_expectations/bls_cpi_release_dates.csv` | `raw/bls/` |
> | `04_cleveland_fed_inflation_expectations/cfed_*.csv` | `interim/cleveland_fed/` |
> | `05_sf_fed_usmpd/USMPD.xlsx`, `monetary-policy-surprises.zip` | `raw/sf_fed_usmpd/` |
> | `05_sf_fed_usmpd/surprises_extracted/` | `raw/sf_fed_usmpd/monetary-policy-surprises/` |
> | `05_sf_fed_usmpd/usmpd_mps_surprises.csv`, `usmpd_mps_minutes_surprises.csv` | removed: byte-identical copies of `mps.csv` / `mps_minutes.csv` in the folder above |
> | `05_sf_fed_usmpd/usmpd_{statements,press_conferences,monetary_events,minutes}.csv` | `interim/usmpd/` |
> | `06_merged/event_study_table.csv` | `processed/event_study_table.csv` |
> | `06_merged/Data Dictionary_ event_study_table.csv.docx` | `processed/event_study_table_dictionary.docx` |


Data for the EEIF causal-inference project on sector rate sensitivity around Fed announcement surprises.
Date range: 2023-01-01 to the pull date. API keys are read from environment variables only (never stored here).
Python env: project `.venv`, see the root README.

| # | Dataset | Folder | Source | Series / URL | Date pulled |
|---|---------|--------|--------|--------------|-------------|
| 1 | FOMC announcement dates | `01_fomc_dates/` | Federal Reserve Board, FOMC Calendars | https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm | 2026-09-21 |
| 2 | Sector ETF daily prices and returns | `02_sector_etf_returns/` | Yahoo Finance via yfinance 1.7.0 | Tickers: XLF, XLK, XLU, XLY, XLI, XLE, XLV, XLP, XLB, XLRE, XLC, SPY; field `Adj Close` | 2026-09-21 |
| 3 | FRED macro controls | `03_fred_controls/` | FRED (Federal Reserve Bank of St. Louis) API | Series: VIXCLS, DGS2, DGS10, T10Y2Y, BAMLC0A4CBBB, BAMLC0A1CAAA, DCOILWTICO, DTWEXBGS, DTB3 | 2026-09-22 |
| 4 | Cleveland Fed inflation expectations | `04_cleveland_fed_inflation_expectations/` | Federal Reserve Bank of Cleveland | https://www.clevelandfed.org/-/media/files/webcharts/inflationexpectations/inflation-expectations.xlsx?sc_lang=en (linked from https://www.clevelandfed.org/indicators-and-data/inflation-expectations) | 2026-09-22 |
| 5 | SF Fed U.S. Monetary Policy Event-Study Database (USMPD) | `05_sf_fed_usmpd/` | Federal Reserve Bank of San Francisco, Center for Monetary Research | https://www.frbsf.org/wp-content/uploads/USMPD.xlsx and https://www.frbsf.org/wp-content/uploads/monetary-policy-surprises.zip (linked from https://www.frbsf.org/research-and-insights/data-and-indicators/us-monetary-policy-event-study-database/) | 2026-09-22 |
| 6 | Merged event-study table | `06_merged/` | Built in-project from datasets 1-5 | n/a (derived) | 2026-09-22 |
| 7 | WRDS firm panel (CRSP, Compustat, CCM, Fama-French) | `core/data/raw/wrds/` (gitignored, licensed) | Wharton Research Data Services, user rileykehoe05 | crsp.dsp500list_v2, crsp.dsf_v2, crsp.stkdelists, ff.factors_daily, crsp_a_ccm.ccmxpf_lnkhist, comp.fundq, comp.co_hgic | 2026-10-04 |
| 8 | 2026 holdout extension | `core/data/raw/yfinance_firms/` (gitignored), `core/data/raw/wikipedia/` | Yahoo Finance via yfinance 1.7.0; Wikipedia | 486 end-2025 S&P members still listed, plus SPY, from 2025-01-01; https://en.wikipedia.org/wiki/List_of_S%26P_500_companies | 2026-10-04 |

## 1. FOMC dates
- Source: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm. First pulled 2026-09-21; 2026 rows re-fetched and re-verified 2026-09-21 (snapshot in `raw_fomccalendars_recheck.html`).
- Files in `01_fomc_dates/`:
  - `fomc_dates.csv`: 31 rows, 2023-02-01 to 2026-09-16.
  - `fomc_meetings.csv`: 30 rows, for the separate futures/Bloomberg pull.
  - `raw_fomccalendars.html` and `raw_fomccalendars_recheck.html`: page snapshots.
  - `build_fomc.py`: parser that regenerates both CSVs.
- `fomc_dates.csv` columns: `announcement_date`, `scheduled_or_unscheduled`, `notes`, `use_for_surprise`. Two-day meetings use the second day.
- `use_for_surprise` is True for all 30 scheduled meetings and False for the 2025-08-22 notation vote (longer-run goals statement, no rate decision). That row is kept and flagged, not deleted.
- `fomc_meetings.csv` columns: `announcement_date`, `meeting_year`. Only rows with `use_for_surprise` True, sorted oldest first (8 in 2023, 8 in 2024, 8 in 2025, 6 in 2026).
- 2026 verification (2026-09-21): the six 2026 dates (01-28, 03-18, 04-29, 06-17, 07-29, 09-16) each matched the second day of the meeting on the Fed page. Result: 6 of 6 PASS. The October 27-28 and December 8-9 meetings are on the page but not yet held, so they are excluded, as are all 2027 meetings.
- The page lists no unscheduled or emergency meetings for 2023+. It is a schedule, not a log of policy actions, so this file cannot prove no unscheduled rate move occurred.

## 2. Sector ETF daily returns
- Source: Yahoo Finance via `yfinance` 1.7.0 (`yf.download`, `auto_adjust=False`, field `Adj Close`, which is split- and dividend-adjusted). Pulled 2026-09-21, about 17:15 EDT, after the close.
- Files in `02_sector_etf_returns/`: `etf_adj_close_prices.csv`, `etf_daily_returns.csv` (simple returns, `pct_change` with no gap filling), `build_etf.py` (pull, checks, FOMC cross-check).
- Coverage: 932 trading days per ticker, 2023-01-03 to 2026-09-21, 931 returns each, 0 missing prices. 2023-01-01 was a market holiday, so 2023-01-03 is the first possible date. Its return is NaN because no earlier price is in range. All 12 tickers start on 2023-01-03, including XLRE and XLC.
- FOMC cross-check against `fomc_meetings.csv`: all 30 dates are trading days and every ticker has a return on every one of them.
- Returns above 10% in absolute value: 3, all on 2025-04-09 (XLK +13.4%, XLY +10.9%, SPY +10.5%). Raw and adjusted returns are identical, so this is not a data or split artifact. It is the tariff-pause rally.
- Splits in the window: XLK, XLU, XLY, XLE and XLB each had a 2-for-1 split on 2025-12-05. Adjusted prices are continuous across it (no return above 1.2% that day), so no correction is needed. Do not use unadjusted `Close` for these tickers.
- Caveat: the 2026-09-21 row was pulled shortly after the close. Yahoo occasionally revises the latest bar, so refresh before final analysis.

## 3. FRED controls
- Source: FRED API (`api.stlouisfed.org/fred`), key read from `FRED_API_KEY` (never hardcoded or printed; confirmed present via a length check only). Pulled 2026-09-22.
- Files in `03_fred_controls/`: `fred_controls.csv`, `build_fred.py` (existence check, pull, `BBB_minus_AAA_OAS` construction, checks, FOMC cross-check).
- **Series ID confirmation (done before pulling any data):** all 9 IDs exist and are all daily frequency. Titles/frequencies:
  - `VIXCLS` — CBOE Volatility Index: VIX (Daily, Close)
  - `DGS2` — 2-Year Treasury constant-maturity yield (Daily)
  - `DGS10` — 10-Year Treasury constant-maturity yield (Daily)
  - `T10Y2Y` — 10Y minus 2Y term spread (Daily)
  - `BAMLC0A4CBBB` — ICE BofA BBB US Corporate Index OAS (Daily, Close)
  - `BAMLC0A1CAAA` — ICE BofA AAA US Corporate Index OAS (Daily, Close)
  - `DCOILWTICO` — WTI crude oil price, Cushing OK (Daily)
  - `DTWEXBGS` — Nominal broad US dollar index (Daily)
  - `DTB3` — 3-Month T-bill secondary market rate (Daily)
  - None are wrong and none are non-daily. **One flag:** `BAMLC0A4CBBB` and `BAMLC0A1CAAA` both have `observation_start = 2023-09-22` in FRED's live data (confirmed independently, not a request error), about 8.7 months after this project's 2023-01-01 start. There is no BBB/AAA OAS (or spread) data for 2023-01-01 through 2023-09-21.
- Coverage: 983 rows, 2023-01-02 to 2026-09-21. Not forward-filled; a missing value is left as `NaN`.
- `BBB_minus_AAA_OAS` = `BAMLC0A4CBBB` − `BAMLC0A1CAAA`, so it is `NaN` wherever either input is `NaN` (same 198 missing rows as the OAS series).
- **Missing-value counts and why:**
  - `VIXCLS` 23, `DGS2`/`DTB3` 54, `DGS10` 54, `T10Y2Y` 53, `DCOILWTICO` 61, `DTWEXBGS` 53 — almost all line up with US federal holidays (New Year's, MLK, Presidents' Day, Memorial Day, Juneteenth, July 4th, Labor Day, Thanksgiving, Christmas). A handful do not: Good Friday (2024-03-29, 2025-04-18, 2026-04-03 — bond/equity markets close but it isn't a federal holiday), the 2025-01-09 national day of mourning, and the most recent 1-3 days near the pull date (2026-09-16 to 2026-09-21) that FRED hasn't posted yet — a reporting lag, not a real gap.
  - `BAMLC0A4CBBB` / `BAMLC0A1CAAA` / `BBB_minus_AAA_OAS` 198 each — 185 of these are the pre-2023-09-22 start-date gap above; the remaining 13 are ordinary holiday gaps within the series' actual coverage.
  - 12 rows in the file fall on a Saturday or Sunday (all month-end dates). This is only because the ICE BofA OAS series occasionally timestamps a month-end value on a weekend date; every other series is `NaN` on those 12 rows, as expected.
  - One business day, 2026-09-22, has no row at all yet in FRED's own calendar (today's data not posted at pull time).
- **FOMC cross-check** against `fomc_meetings.csv` (30 dates): every date is present as a row.
  - `BAMLC0A4CBBB`, `BAMLC0A1CAAA`, `BBB_minus_AAA_OAS` have no value on 6 of the 30 dates (2023-02-01, 2023-03-22, 2023-05-03, 2023-06-14, 2023-07-26, 2023-09-20) — all before the series' 2023-09-22 start.
  - `DCOILWTICO` has no value on 2026-09-16 (the most recent FOMC date), from the reporting lag above — likely to backfill on a later pull.
  - All other series/date combinations have a value.

## 4. Cleveland Fed inflation expectations
- Source: Cleveland Fed "Inflation Expectations" indicator page, public spreadsheet download. Page: https://www.clevelandfed.org/indicators-and-data/inflation-expectations . File: `inflation-expectations.xlsx`, linked from that page. Pulled 2026-09-22.
- Files in `04_cleveland_fed_inflation_expectations/`: `inflation-expectations.xlsx` (raw download), `cfed_inflation_expectations_monthly.csv` (extracted, project-window output), `build_cfed_inflation.py` (download, extraction, checks, FOMC preview).
- **Frequency: monthly**, one row per calendar month (`model_output_date`, always the 1st of the month).
- **Horizons available: 30**, every integer year from 1-year through 30-year expected inflation, all in the `Expected Inflation` sheet used here.
- The workbook also has two sheets not extracted (available on request, same source): `Ten-year Expected Chart` (10-year expected inflation plus real and inflation risk premia) and `Real Interest Rate` (real rate at 1-month, 1-year, and 10-year horizons).
- Values are decimal fractions as published (e.g. `0.0261` = 2.61%), not converted to percent.
- Full native history in the file: 537 rows, 1982-01-01 to 2026-09-01. The saved CSV is trimmed to the project window: 45 rows, 2023-01-01 to 2026-09-01, 0 missing values in any horizon column. Not forward-filled or interpolated — it is monthly data saved as its own file, kept separate from the daily datasets.
- **Publication-lag caveat**, from the page's own FAQ (quoted): *"The inflation expectations model is run on the day of the month that the CPI is released. The model results are released before 4 pm on that day."* CPI for month M-1 is typically released by the BLS roughly 10-15 days into month M, so a row labeled `model_output_date` = the 1st of month M is usually not actually published until partway through month M, not on the 1st as the date label implies.
- **FOMC preview mapping (shown, not saved, not merged):** for each of the 30 dates in `fomc_meetings.csv`, the most recent `model_output_date` on or before that date, using the naive month-start convention. 10 of the 30 FOMC dates fall on or before the 15th of their own month, where that convention is least reliable per the caveat above — for those, the true most-recently-published value may actually be one month earlier than shown. Those 10 dates: 2023-02-01, 2023-05-03, 2023-06-14, 2023-11-01, 2023-12-13, 2024-05-01, 2024-06-12, 2024-11-07, 2025-05-07, 2025-12-10.
- This mapping is not in any CSV and not joined to the FOMC, ETF, or FRED files — it is a preview only, pending your call on how to resolve the early-in-month cases above.

## 4a. FOMC -> Cleveland Fed inflation expectations mapping, corrected for true CPI publication dates
- Source for release dates: BLS "Schedule of Selected Releases" year pages, `https://www.bls.gov/schedule/2023/home.htm`, `.../2024/home.htm`, `.../2025/home.htm`, plus `https://www.bls.gov/schedule/news_release/cpi.htm` for 2026 (that page only shows a rolling ~1-year window, so the year pages were needed for 2023-2025). Pulled via the browser pane 2026-09-22 (`curl` to bls.gov returns HTTP 403 -- BLS blocks non-browser requests -- so this could not be scripted with `requests`/`curl`; the compiled dates are checked into `bls_cpi_release_dates.csv` for reproducibility instead of being re-scraped by `build_cfed_fomc_mapping.py`).
- Files: `bls_cpi_release_dates.csv` (45 rows: `reference_month`, `release_date`, 2022-12 through 2026-08), `cfed_fomc_mapping.csv` (the corrected mapping, 30 rows), `build_cfed_fomc_mapping.py`.
- Method: Cleveland Fed's own FAQ (quoted on its page): *"The inflation expectations model is run on the day of the month that the CPI is released. The model results are released before 4 pm on that day."* So the CF row labeled `model_output_date` = 1st of month M is truly published on the date BLS released the CPI for reference month M-1, not on the 1st of month M. `true_publish_date` = that BLS release date, joined from `bls_cpi_release_dates.csv`.
- **One row has no BLS-anchored publish date: the CF row for `model_output_date = 2025-11-01`.** BLS never released a standalone October 2025 CPI report (2025 government-shutdown disruption) -- confirmed both by its absence from the BLS schedule pages and by the Cleveland Fed page's own note that it used its Nowcasting estimate for October 2025 CPI in place of actual BLS data for that row. There is no public BLS release event to anchor this row's true publish date to, so it is excluded from the corrected mapping (its `true_publish_date` is left blank) rather than guessed.
- **7 of the 30 FOMC dates change** between the naive (1st-of-month) mapping and the corrected (true-publish-date) mapping: 2023-02-01, 2023-05-03, 2023-11-01, 2024-05-01, 2024-11-07, 2025-05-07, 2025-12-10.
  - The first six shift back exactly one month, as expected (the "current month" row had not actually posted yet on those FOMC dates).
  - **2025-12-10 is the interesting case:** the naive mapping picks the 2025-12 row, but that row's true publish date is 2026-01-13 (well after the FOMC date). Because the 2025-11 row has no confirmed publish date (see above), the corrected mapping falls back to the last row with a *confirmed* publish date: 2025-10, published 2025-10-24 (itself a late release, delayed by the shutdown from its normal mid-October slot). It is possible the Nowcast-based 2025-11 row was actually published before 2025-12-10 too, which would make that the true answer instead -- this is flagged as unresolved, not silently assumed either way.
- `cfed_fomc_mapping.csv` columns: `fomc_date`, `naive_month`, `naive_val_1yr`, `naive_val_10yr`, `corrected_month`, `corrected_publish_date`, `corrected_val_1yr`, `corrected_val_10yr`, `changed`. Not merged into `fomc_dates.csv`, `fomc_meetings.csv`, or any other file.

## 5. SF Fed U.S. Monetary Policy Event-Study Database (USMPD) -- candidate Bloomberg-futures replacement
- Source: https://www.frbsf.org/research-and-insights/data-and-indicators/us-monetary-policy-event-study-database/ . The page's own chart-data CSVs are small illustrative excerpts; the full underlying files, linked at the bottom of the page, are `USMPD.xlsx` (raw high-frequency changes) and `monetary-policy-surprises.zip` (constructed surprise indices + R code). Pulled via the browser pane 2026-09-22, then downloaded directly with `curl` (unlike BLS, frbsf.org does not block non-browser requests). Reference: Acosta, Ajello, Bauer, Loria, and Miranda-Agrippino (2025), SF Fed Working Paper 2025-30.
- Files in `05_sf_fed_usmpd/`: `USMPD.xlsx` and `monetary-policy-surprises.zip` (raw downloads), `usmpd_statements.csv` / `usmpd_press_conferences.csv` / `usmpd_monetary_events.csv` / `usmpd_minutes.csv` (the four USMPD.xlsx sheets, extracted), `usmpd_mps_surprises.csv` / `usmpd_mps_minutes_surprises.csv` (copies of `mps.csv` / `mps_minutes.csv` from the zip), `surprises_extracted/` (full zip contents: the two CSVs above plus `y1.csv`, `mps.R`, `gss.R`, `README.md`), `build_usmpd.py`.
- **Surprise measures included:**
  - Raw high-frequency changes, one column each, in every one of the four event sheets: `MP1` (surprise change in fed funds target for the current meeting), `MP2` (implied change for the next meeting), `FF1`-`FF6` (fed funds futures, 6 monthly contracts), `ED1`-`ED8` (Eurodollar futures through 2021 / SOFR futures from 2022, 8 quarterly contracts -- covers and exceeds the requested ED1-ED4), plus `OIS1Y`, `OIS2Y`, Treasury yields (`UST3M` through `UST30Y`), TIPS yields, and equity/dollar/FX returns (`SP500`, `SPFUT`, `DXY`, `EURUSD`, `USDJPY`). These are all **raw**, not orthogonalized.
  - Constructed single-number surprise indices in `usmpd_mps_surprises.csv`: `STMT`, `PC`, `ME` -- the first-principal-component monetary policy surprise (Acosta et al. 2025 methodology, extending Nakamura-Steinsson 2018), one per FOMC statement/press-conference/monetary-event. `usmpd_mps_minutes_surprises.csv` has the equivalent `MIN` surprise for minutes releases.
  - **No precomputed orthogonalized (target/path) series is included.** `gss.R` in the zip computes Gürkaynak-Sack-Swanson (2005) target and path factors from the Statements sheet, but only as an R script -- running it is a separate step this pull did not do. If you need Jarociński-Karadi or Miranda-Agrippino-Ricco decompositions, the page says the raw USMPD columns are sufficient to build them, but neither is precomputed here either.
- **Date range:** `Statements` and `Monetary Events`: 279 rows, 1994-02-04 to 2026-09-16. `Press Conferences`: 95 rows, 2011-04-27 to 2026-09-16 (press conferences didn't start until 2011). `Minutes`: 206 rows, 2001-02-01 to 2026-08-19. `usmpd_mps_surprises.csv`: 279 rows, same range as Statements. `usmpd_mps_minutes_surprises.csv`: 174 rows, 2005-01-04 to 2026-08-19.
- **Frequency: one row per FOMC communication event, not a continuous daily series.** Each row is a discrete event (a statement release, a press conference, a minutes release, etc.), identified by `Date` and a precise `date_time` timestamp -- there is one row per FOMC meeting per event type, not one row per calendar day. (`y1.csv` in the zip is the one exception: a genuinely continuous daily series, the 1-year Treasury yield used as a normalization input, not a surprise measure itself.)
- **Scheduled vs. unscheduled: yes, explicitly.** An `Unscheduled` 0/1 column in the `Statements` and `Monetary Events` sheets flags conference-call/emergency meetings. 18 of the 279 Statements-sheet rows are flagged unscheduled (1994-2020, including the four March/April 2020 COVID-period calls); none of the 30 dates in `fomc_meetings.csv` are unscheduled, consistent with `fomc_dates.csv`. There is also a `SEP` flag (Summary of Economic Projections meeting) that lines up with the `* meeting` notes already in `fomc_dates.csv`, and a `PC` flag for meetings with a press conference.
- **FOMC cross-check** against `fomc_meetings.csv` (30 dates), matched to the `Statements` sheet: **all 30 have a row and a non-missing `MP1`/`MP2` value.** None are missing. The `SEP` flag on the matched rows also agrees with `fomc_dates.csv`'s SEP notes for every date checked.
- **Assessment: yes, USMPD looks like a workable replacement for the planned Bloomberg futures pull** for FOMC-meeting-level surprises -- it has raw fed funds/Eurodollar/SOFR futures changes plus a ready-made principal-component surprise index, covers the full 2023-present window with no gaps against your 30 dates, and explicitly flags scheduled vs. unscheduled meetings. The one gap versus a from-scratch Bloomberg pull is that orthogonalized target/path factors are not precomputed -- you'd run `gss.R` (or your own code on the raw columns) if you need that decomposition.
- **Not merged into any other project file** (not joined to `fomc_dates.csv`, `fomc_meetings.csv`, the ETF, FRED, or Cleveland Fed files) -- saved as its own dataset, per your instruction, pending your decision on whether to formally replace the Bloomberg futures pull with this.

## 6. Merged event-study table
- **File:** `06_merged/event_study_table.csv`, built by `06_merged/build_event_study_table.py`. One row per FOMC announcement date, keyed on the 30 dates in `01_fomc_dates/fomc_meetings.csv`. **A data table only -- no regression or model was run.**
- **Amendment to dataset 4a:** `cfed_fomc_mapping.csv` was regenerated to add `corrected_val_5yr` / `naive_val_5yr` (it previously had only 1yr/10yr). Same method as before, no other change; see section 4a.

### Column groups
- **Keys / meta:** `announcement_date`, `meeting_year`, `use_for_surprise` (carried over from `fomc_dates.csv` -- always True for these 30 rows, since `fomc_meetings.csv` already excludes the one unscheduled notation vote; kept here as an explicit, checkable column rather than an assumption).
- **Treatment (dataset 5, USMPD, matched on the Statements-sheet event date):**
  - **Primary surprise measures: `MP1`** (raw, fed-funds-futures-implied surprise for the current meeting) and **`STMT`** (the constructed first-principal-component surprise for the statement window, Acosta et al. 2025). Use one of these two as the headline regressor.
  - Secondary / robustness columns, all included, none dropped: `MP2` (next-meeting-implied surprise), `PC` and `ME` (press-conference-window and full-monetary-event-window versions of the constructed surprise), and the raw `FF1`-`FF6` (fed funds futures) / `ED1`-`ED8` (Eurodollar/SOFR futures) term-structure changes.
  - `SEP` and `Unscheduled` flags, carried through unchanged from USMPD (both 0/1).
- **Outcome (dataset 2, sector ETFs):** `ret_XLF`, `ret_XLK`, `ret_XLU`, `ret_XLY`, `ret_XLI`, `ret_XLE`, `ret_XLV`, `ret_XLP`, `ret_XLB`, `ret_XLRE`, `ret_XLC`, `ret_SPY` -- same-day simple daily return for each ticker on the announcement date.
- **Controls (dataset 3, FRED):** same-day `VIXCLS`, `DGS2`, `DGS10`, `T10Y2Y`, `DCOILWTICO`, `DTWEXBGS`, `DTB3`, `BAMLC0A4CBBB`, `BAMLC0A1CAAA`, `BBB_minus_AAA_OAS`, plus a `lagged_` version of each. `lagged_*` = that series' value on the prior row of the FRED daily calendar (the calendar day immediately before the FOMC date in `fred_controls.csv`, not the next available non-null day -- if that prior row itself is NaN, `lagged_*` is left NaN, nothing is filled or searched further back). `lagged_date` is included as an audit column showing exactly which calendar date each `lagged_*` value was read from.
- **Controls (dataset 4, Cleveland Fed):** `val_1yr`, `val_5yr`, `val_10yr` from the **corrected** mapping (`corrected_val_1yr/5yr/10yr` in `cfed_fomc_mapping.csv`, true-CPI-publication-date based) -- the naive month-start columns were not used here.

### Checks
- **Row count: 30 of 30, confirmed** (the build script asserts this and would fail loudly otherwise).
- **NaN cells, by column, and why:**
  - `DCOILWTICO`: 1 NaN, on 2026-09-16 -- the WTI reporting lag already flagged in section 3 (not yet posted at pull time).
  - `BAMLC0A4CBBB`, `BAMLC0A1CAAA`, `BBB_minus_AAA_OAS`: 6 NaN each, on the 6 FOMC dates before the series' true 2023-09-22 start (2023-02-01, 2023-03-22, 2023-05-03, 2023-06-14, 2023-07-26, 2023-09-20) -- the gap already flagged in section 3.
  - `lagged_BAMLC0A4CBBB`, `lagged_BAMLC0A1CAAA`, `lagged_BBB_minus_AAA_OAS`: 6 NaN each, the *same* 6 dates -- for all 6, the prior trading day is also before 2023-09-22, so no extra rows are affected.
  - Every other column (all USMPD treatment columns, all 12 ticker returns, VIX and the other 8 FRED series and their lags, and val_1yr/5yr/10yr): **0 NaN.**
- **Core-field non-null check: passes.** All 12 `ret_*` columns, `MP1`, `STMT`, and `VIXCLS` are non-null for all 30 rows.
- **Outlier flags (values far outside their normal range for that column), reviewed and judged plausible, not data errors:**
  - No ticker return exceeds 10% in absolute value on any of the 30 FOMC dates (the two >10% ETF moves found in section 2, on 2025-04-09, are not FOMC dates and don't appear here).
  - `MP1` exceeds 0.05 percentage points on two dates: 2024-09-18 (-0.1187) and 2026-07-29 (-0.065). 2024-09-18 was the first 50-basis-point cut, a genuinely large and well-known surprise (the market had been split 25bp vs. 50bp going in). Both dates are internally consistent -- `MP2`, `STMT`, `PC`, and `ME` all move in the same direction with plausible relative magnitudes, and the equity/rate-sensitive-sector returns on those two dates move the same direction too -- so these read as real surprises, not data glitches.
  - `STMT` exceeds 0.05 in absolute value on four dates (2023-06-14, 2024-09-18, 2024-12-18, 2026-06-17); same pattern, cross-checked against `MP1`/`ME` and judged plausible.
  - No VIX value in the table exceeds 30.
  - `DCOILWTICO` same-day values range $58.67-$110.47 across the 30 dates -- within the full-sample range reported in section 3, nothing out of bounds.
- **Not run:** any regression, model fit, or statistical test. This is a merge and a set of data-quality checks only.

## 7. WRDS firm panel (Q3)
- **Pulled by:** `python -m fedcore.ingest.wrds pull --user <name>` on 2026-10-04. Login is saved once in the user's pgpass file; the script never handles a password. `core/data/raw/wrds/_manifest.json` records the exact SQL, source table and row count for every file.
- **Licensed data:** WRDS terms apply. The folder is gitignored and must never be committed or uploaded raw; ship only derived arrays to any external compute.
- **Universe:** every PERMNO in the S&P 500 at any point with a spell ending on or after 1993-01-01 (1,295 firms). Start date 1993-01-01 gives 252 trading days of history before the first USMPD statement (1994-02-04). Survivor-free: departed and delisted members are included.
- **Vintage:** CRSP CIZ (`_v2`) format, annual update: daily data ends **2025-12-31**. The legacy SIZ tables stopped at 2024-12-31 and were not used. 2026 meetings have no firm data until the next annual update.

| Dataset | Source table | Rows | Key | Notes |
|---|---|---|---|---|
| `sp500_members` | crsp.dsp500list_v2 | 2,084 | permno, mbrstartdt | Membership spells; current members end 2025-12-31 (file end, not exit) |
| `crsp_daily` | crsp.dsf_v2 | 6,541,906 | permno, dlycaldt | One parquet per year, 1993-2025. `dlyret` is delisting-inclusive (CIZ); 0.19% of returns missing |
| `crsp_delist` | crsp.stkdelists | 637 | permno, delistingdt | Reference only; already folded into `dlyret` |
| `ff_factors_daily` | ff.factors_daily | 8,473 | date | `mktrf`, `smb`, `hml`, `rf`; runs to 2026-08-31 |
| `ccm_link` | crsp_a_ccm.ccmxpf_lnkhist | 1,590 | gvkey, lpermno, linkdt | linktype LU/LC, linkprim P/C only |
| `comp_fundq` | comp.fundq | 113,450 | gvkey, datadate, fyearq, fqtr | INDL/STD/D/C. 28 (gvkey, datadate) repeats are fiscal-year changes, hence the 4-column key. 3.6% have no `rdq` and cannot be joined point-in-time |
| `comp_gics` | comp.co_hgic | 1,958 | gvkey, indfrom | Historical GICS with validity dates |

- **Raw is untouched:** WRDS native column names, dates stored as ISO strings. Renaming, type conversion and the point-in-time joins (fundamentals on `rdq`, sector and link on their validity dates) belong in interim.
- **Not run:** any model or statistical test.

### 7a. Derived Q3 panel (`processed/q3_panel/`, gitignored)
- **Built by:** `fedcore.q3.real.load_real_panel()` from the section-7 tables plus `usmpd_statements`, `mps_surprises` and `treasury_1y`; cached as `.npy` arrays, `rows.parquet` (meeting, permno, gvkey, GICS sector) and `meta.json` (input fingerprints, counts, dropped meetings). Rebuilt automatically when any input file or `BUILD_VERSION` changes; a build takes about 15 seconds.
- **Build of 2026-10-04:** 126,854 firm-meeting rows, 253 scheduled statements (1994-05-17 to 2025-12-10), 1,252 firms, 499-505 firms per meeting. 1994-02-04 and 1994-03-22 dropped: the 63-day 1-year-yield change needs history the Treasury file does not have.
- **Definitions:** shock `s = 10 * STMT` and context = the Q1 state controls (identical to `dml` Q1) plus annualized 21-day market volatility. Target = meeting-day CRSP return in percent. Channels use the synthetic panel's own functions; where the synthetic panel falls back to a firm's true beta, the real panel uses 1.0.
- **Point-in-time:** membership on the meeting date; fundamentals from the latest quarter with `rdq` strictly before the meeting (stale after 365 days); link and GICS valid on the meeting date.
- **Sector:** GICS history starts 1999-06-30. Rows before it take the firm's earliest recorded sector, labeled `gsector_source = backfill` (a deliberate, labeled look-ahead: sectors almost never change). Build 2: 83.3% as-of, 14.6% backfill, 2.2% none. Sector is row metadata, not a model input.
- **Known gaps:** Book-to-market reaches -78/+20 for firms with tiny or negative book equity; the trainer's median/IQR scaling clips at +/-5.
- **Sanity check (not a model):** the meeting-average return regressed on `s` gives -1.05% per 10bp over the 253 meetings, in line with the Q1 DML estimate (-0.96).

## 8. 2026 holdout extension (yfinance)
- **Why:** the school's CRSP subscription is the annual vintage, ending 2025-12-31. The six 2026 statements (01-28, 03-18, 04-29, 06-17, 07-29, 09-16) get firm data from yfinance until the next vintage replaces it.
- **Pulled by:** `python -m fedcore.ingest.yf_firms pull` (retries tickers Yahoo drops under load). Also pulled two small WRDS lookups for the mapping: `crsp_names` (crsp.stksecurityinfohist) and `comp_company` (comp.company: name and CIK; a lookup table with no date, so read directly rather than registered in the catalog).
- **Universe:** the 503 S&P members on 2025-12-31, kept if still in the Wikipedia list on the pull date. Matched by CRSP ticker, share-class exact (GOOG/GOOGL, NWS/NWSA, FOX/FOXA, BRK.B, BF.B): 484, plus 2 by CIK = **486**. The 17 unmatched all left the index during 2026 (e.g. Campbell's, Electronic Arts, Hologic, Trade Desk). Wikipedia no longer publishes dated changes, so 2026 removals are excluded rather than dated and 2026 additions are not covered (no CRSP history): a small, labeled survivorship bias.
- **Overlap check, 2025 (`fedcore.q3.real.compare_overlap`):** 120,622 firm-days. Return correlation 0.99998; median absolute difference 0.0pp, 99th percentile 0.003pp; 0.02% of days differ by more than 0.1pp; worst firm correlation 0.991. Close and volume match CRSP (median log volume ratio 0.0). SPY vs the Fama-French market: correlation 0.995, median gap 0.056pp (SPY is the S&P 500, FF is the whole market).
- **Splice (`holdout_inputs`):** after 2025-12-31, returns from adjusted closes, price from the close, shares held at the last CRSP value, market = Fama-French through 2026-08-31 then SPY. CRSP rows are never replaced.
- **Holdout panel (`load_holdout_panel`, cached in gitignored `processed/q3_holdout/`):** 6 meetings x 486 firms = 2,916 rows. Scored separately; never pooled into training or the headline evaluation.

