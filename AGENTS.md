# AGENTS.md

Research framework (`fedci`) for FOMC causal inference. Spec lives in `docs/question.md`
(Q-spec v1.1); `docs/methods.md` and `docs/transformer.md` are the model design notes.
Verify docs against code — the framework is the source of truth.

## Commands

- Install: `.venv\Scripts\python -m pip install -e ".[dev,model]"` (Windows venv, README style; `ingest` extra exists but is unused so far)
- Verify (in order): `fedci check` (data integrity) -> `pytest` (includes headless renders of every viewer page)
- If `pytest` dies with `PermissionError` on `Temp\pytest-of-moomi`, rerun with `--basetemp=<writable dir>`; the default temp root is broken and needs admin to delete.
- There is no lint/typecheck/CI config. Do not invent one unless asked.
- CLI: `fedci check | catalog | sql "SELECT ..." | run q1 k=v | runs [q1] | app | serve`
- `start.cmd` (repo root) = double-click launcher: creates the venv on first run, then `fedci serve`, which builds the SPA if `src/frontend/dist` is stale, reuses an already-running server, and opens the browser (`--no-build/--no-open/--rebuild/--port`).

## Data layer

- Every dataset is registered in `data/catalog.yaml` and exposed as a DuckDB SQL view with the same name. Adding data = file under `data/{raw,interim,processed}/` + catalog entry + provenance note in `data/PROVENANCE.md` + `fedci check`. It then appears in `load()`, SQL, viewer, and tests automatically.
- `data/raw/` is untouched downloads — never edit or regenerate. `data/PROVENANCE.md` still cites old `eeif_data/` paths; use its mapping table to locate files.
- Never forward-fill or interpolate: a gap stays NaN. `fedci.data.event_panel` reindexes to the trading-day calendar (`etf_returns`) without filling.
- `data.meetings()` excludes unscheduled actions by default (separate treatment, never pooled). `rel_day = 0` is the announcement day.

## Questions and results

- Question IDs Q1–Q5 are permanent (Q2 is parked but kept for run continuity) — never renumber, rename, or delete `q2_channels`. `tests/test_questions.py` enforces the registry order and that every `spec.datasets` entry exists in the catalog.
- Each `questions/qN_*/` package = `spec.py` (what is asked) + `pipeline.py` (`run(store, **params)`). All output goes through `ResultStore.save(...)`; pipelines write nothing else anywhere. Code shared by 2+ questions moves to `fedci/eval` or `fedci/data`; everything else stays in the question package.
- Q2–Q5 pipelines raise `NotImplementedError` and `test_parked_pipelines_are_stubs_until_built` asserts that. Q1 is implemented (PLR-DML in `fedci/eval/dml.py`, estimation samples in `q1_total_effect/samples.py`); when you implement another pipeline, update that test in the same change.
- Runs are append-only under `results/<question key>/<run_id>/` with data fingerprints; never overwrite or delete runs. Tag `role` as `primary`/`robustness`/`exploratory` — one primary spec per active question, every other run adds to the multiple-testing count.

## Evaluation rules (enforced in `fedci/eval`)

- `walk_forward` splits by whole meetings in chronological order — never random folds, never one meeting on both sides. Use `embargo` when the outcome window (e.g. t+20) could reach the next meeting.
- `meeting_bootstrap` resamples whole meetings; report `n_meetings`, not row count.
- Select Q3 models on response-prediction performance, not downstream trading profits (docs/question.md).

## Gotchas

- Paths resolve from `fedci/config.py`, so code runs from any working directory; the CLI entry is `fedci` -> `fedci.cli:main`.
- Viewer pages import `src/frontend/streamlit/_common.py`; `streamlit run src/frontend/streamlit/Home.py` puts `src/frontend/streamlit/` on `sys.path` — `tests/test_app.py` mirrors this. Keep page imports compatible with both.
- `fedci run` params are `k=v` pairs; a run's datasets record content hashes so results are traceable to exact data versions.
