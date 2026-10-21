from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # trading/
RESULTS = ROOT / "results"                   # this project's runs
# Frozen Q3 predictions are read as files from the encoder's runs (never imported as code).
Q3_RUNS = ROOT.parent / "encoder" / "results" / "q3_susceptibility"
