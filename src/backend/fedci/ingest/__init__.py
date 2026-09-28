"""Data ingestion: one module per external source (e.g. firm_bars.py, edgar.py).

Contract for an ingest module:
  - downloads write untouched copies to data/raw/<source>/
  - cleaning writes to data/interim/<source>/ (no forward-fill, no interpolation: gaps stay NaN)
  - register every output in data/catalog.yaml and document provenance in data/PROVENANCE.md
  - API keys come from environment variables only

Pending pulls (docs/question.md, "Data moves"): sector ETF returns back to 1998, firm daily
bars (S&P 100 -> 500), SEC EDGAR as-filed fundamentals, auxiliary event calendar (CPI,
payrolls, ECB, minutes, press conferences), earnings dates + actuals.
"""
