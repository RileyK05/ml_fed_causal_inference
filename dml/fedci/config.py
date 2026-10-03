"""DML project paths. Data paths live in fedcore.config."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # dml/

RESULTS = ROOT / "results"        # one folder per question, one subfolder per run
FRONTEND = ROOT / "frontend"      # React web app (dist/ = build) and the Streamlit viewer
