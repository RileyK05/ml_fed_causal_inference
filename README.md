# FOMC causal inference

How do firms respond to unanticipated Fed news, and do deviations from their predicted
responses forecast subsequent returns? The active project is **Q1 -> Q3 -> Q4 -> Q5**:
DML establishes the aggregate baseline, Q3 is the main firm-response modeling contribution,
Q4 tests out-of-sample prediction gaps, and Q5 evaluates trading usefulness if Q4 succeeds.
Q2 (mechanisms) is parked. See [docs/question.md](docs/question.md) for the specification
and [docs/methods.md](docs/methods.md) for model candidates. The detailed Q3 encoder design
is in [docs/transformer.md](docs/transformer.md).

This repo is the framework: data access, the question scaffolds, a results store and a viewer.
The analysis inside each question is built on top of it.

## Setup

**One-click launch (Windows):** double-click `start.cmd` at the repo root. It creates the
venv on first run, builds the web app if it is stale, serves the estimation playground, and
opens the browser at http://localhost:8000.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev,model]"
.venv\Scripts\fedci check      # every dataset present, dates parse, keys unique
.venv\Scripts\fedci serve      # build if stale, serve, open browser (flags: --no-build --no-open --rebuild --port)
.venv\Scripts\fedci app        # open the legacy Streamlit viewer
```

## Layout

```
docs/question.md             the spec: active chain, parked Q2, evaluation protocol
data/
  catalog.yaml               registry of every dataset (the one place to add data)
  PROVENANCE.md              where each file came from, pull dates, known gaps
  raw/                       untouched downloads, never edited
  interim/                   cleaned, one folder per source
  processed/                 analysis-ready tables
src/
  backend/
    fedci/                   the Python package
      data/                    retrieval layer: load(), query() (SQL), meetings(), event_panel()
      questions/               one package per question
        q1_total_effect/         spec.py (what is asked) + pipeline.py (how it is answered)
        q2_channels/            parked; retained for continuity
        q3_susceptibility/
        q4_gap/
        q5_strategy/
      eval/                    shared protocol: walk-forward by meeting, meeting bootstrap
      results/                 ResultStore: append-only runs with data fingerprints
      ingest/                  future data pulls (1998 extension, firm bars, EDGAR, ...)
      cli.py                   the `fedci` command
  frontend/
    streamlit/               legacy Streamlit viewer (Home.py + pages/)
    dist/                    built web app (fedci serve)
results/                     saved runs, results/<question>/<run_id>/
tests/
```

## Getting data

```python
from fedci.data import load, query, meetings, event_panel

ret = load("etf_returns", start="2024-01-01")               # DataFrame, date parsed
sql = query("SELECT Date, STMT FROM mps_surprises WHERE ABS(STMT) > 0.05")
spine = meetings()                                           # scheduled FOMC meetings
panel = event_panel("etf_returns", pre=1, post=20)          # long: meeting x rel_day x series
```

Dataset names are the keys in [data/catalog.yaml](data/catalog.yaml), and every one is also a
SQL view. Nothing is ever forward-filled or interpolated: a gap stays NaN.

**Adding a dataset:** put the file under `data/raw|interim|processed/`, add an entry to
`catalog.yaml`, log where it came from in `PROVENANCE.md`, and run `fedci check`. It then appears
in `load()`, SQL, the viewer and the tests.

## Building a question

Each `questions/qN_*/pipeline.py` has one function to implement:

```python
def run(store: ResultStore, **params):
    df = load("event_study_table")
    ...
    return store.save(SPEC, name="OLS day-0, STMT", role="primary",
                      params=params, datasets=["event_study_table"],
                      metrics={"beta": b, "n_meetings": n},
                      tables={"coefficients": coef_df}, figures={"scatter": fig})
```

Then run `fedci run q1 window=day0`, and the result appears on the Results and Questions pages.
Each run records the content hash of every dataset it read, and runs are never overwritten. Tag
a run's `role` as `primary`, `robustness` or `exploratory`. The spec allows one primary
specification per active question, and every other run adds to the multiple-testing count.

Code used by only one question stays inside that question's package. Code shared by two or more
moves to `fedci/eval` or `fedci/data`.

## Evaluation rules built into the framework

- `eval.walk_forward` splits by whole meetings in time order. Folds are never random, and one
  meeting never lands on both sides of a split.
- `eval.meeting_bootstrap` resamples whole meetings and reports `n_meetings`, not row count.
- `data.meetings()` excludes unscheduled actions by default, because they are a separate treatment.

## Commands

| | |
|---|---|
| `fedci check` | integrity check of every catalog dataset |
| `fedci catalog` | rows and date coverage per dataset |
| `fedci sql "SELECT ..."` | SQL over the catalog |
| `fedci run q1 k=v` | run a question's pipeline |
| `fedci runs [q1]` | list saved runs |
| `fedci app` | launch the viewer |
| `fedci serve` | serve the estimation playground (API + web app at localhost:8000) |
| `pytest` | tests, including a headless render of every viewer page |
