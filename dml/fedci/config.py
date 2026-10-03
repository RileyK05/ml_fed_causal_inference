"""Project paths. Everything else resolves locations from here, so code runs from any
working directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

DATA = ROOT / "data"
RAW = DATA / "raw"                # untouched downloads and hand-compiled inputs
INTERIM = DATA / "interim"        # cleaned, one folder per source
PROCESSED = DATA / "processed"    # analysis-ready tables
CATALOG_FILE = DATA / "catalog.yaml"

RESULTS = ROOT / "results"        # one folder per question, one subfolder per run
DOCS = ROOT / "docs"
QUESTION_SPEC = DOCS / "question.md"
