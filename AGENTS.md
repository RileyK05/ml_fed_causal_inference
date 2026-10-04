# AGENTS.md

FOMC research repo: three isolated projects on one shared data layer. The spec lives in
`docs/question.md` (Q-spec v1.1) and `docs/methods.md`. Verify docs against code: the code is
the source of truth.

**No training without the user's explicit go-ahead in chat.** Agents build code and unit
tests only; a smoke fit inside a test is fine, but no pre-training, benchmarks, `run_arms`,
or anything that saves a run. A handoff doc asking for training does not count as the go-ahead.

## Layout

| Folder | Package | What |
|---|---|---|
| `core/` | `fedcore` | Shared layer: `data/` (catalog, raw/interim/processed), data access, `protocol.py` (walk-forward, meeting bootstrap), `results/` (ResultStore), `questions/` (the five question definitions) |
| `dml/` | `fedci` | Q1 DML project: pipelines, `eval/dml.py`, FastAPI backend, `frontend/` (React app + Streamlit viewer), `results/` |
| `encoder/` | `fedenc` | Q3 patch-transformer project. Build spec: `docs/encoder/HANDOFF.md`; design: `docs/encoder/transformer.md` |
| `cde/` | `fedcde` | Q3 Neural CDE project. Build spec: `docs/cde/HANDOFF.md` |
| `docs/` | | **All docs.** Project-wide spec at the top (`question.md`, `methods.md`, `agent-guide.md`); one subfolder per project (`docs/core/`, `docs/dml/`, `docs/encoder/`, `docs/cde/`) with its HANDOFF brief and design notes; agent reports in `docs/reports/`. Project folders hold only a README |

**Isolation rule:** projects import `fedcore` and never each other; `fedcore` imports no
project. `core/tests/test_isolation.py` enforces it. Code needed by 2+ projects goes in
`core/`. Each project keeps its own `results/`.

## Commands

- One venv at the repo root. Install: `.venv\Scripts\python -m pip install -e "core[dev]" -e "dml[dev,model]" -e "encoder[dev]" -e "cde[dev]"` (add `[dl]` to encoder/cde for torch).
- Verify (in order): `fedcore check` (data integrity) -> `pytest core` -> `pytest dml` (includes headless renders of every viewer page) -> `pytest encoder` -> `pytest cde`.
- If `pytest` dies with `PermissionError` on `Temp\pytest-of-moomi`, rerun with `--basetemp=<short writable dir>`; the default temp root is broken and needs admin to delete. Keep the path short: Windows' 260-character limit breaks `ResultStore` tests under deep temp paths.
- There is no lint/typecheck/CI config. Do not invent one unless asked.
- CLIs: `fedcore check | catalog | sql "SELECT ..."`; `fedci check | catalog | sql | run q1 k=v | runs [q1] | app | serve`.
- `dml/start.cmd` = double-click launcher: creates the root venv on first run, then `fedci serve`, which builds the SPA if `dml/frontend/dist` is stale, reuses an already-running server, and opens the browser (`--no-build/--no-open/--rebuild/--port`).

## Data layer (core/)

- Every dataset is registered in `core/data/catalog.yaml` and exposed as a DuckDB SQL view with the same name. Adding data = file under `core/data/{raw,interim,processed}/` + catalog entry + provenance note in `core/data/PROVENANCE.md` + `fedcore check`. It then appears in `load()`, SQL, the viewer and the tests automatically.
- `core/data/raw/` is untouched downloads: never edit or regenerate. `PROVENANCE.md` still cites old `eeif_data/` paths; use its mapping table to locate files.
- Never forward-fill or interpolate: a gap stays NaN. `fedcore.data.event_panel` reindexes to the trading-day calendar (`etf_returns`) without filling.
- WRDS data (`core/data/raw/wrds/`, Q3 firm panel) is licensed and gitignored: never commit it or upload it raw. Re-pull with `python -m fedcore.ingest.wrds pull --user <name>` (login saved in pgpass by the user; agents never handle the password). Install `wrds` with `pip install --no-deps wrds`: it pins pandas<2.3, but the repo runs on pandas 3.
- The real Q3 panel is `fedcore.q3.real.load_real_panel()` (cached in gitignored `core/data/processed/q3_panel/`, rebuilt when inputs change); `run_arms(data="real")` uses it. Building the panel is fine; fitting arms on it is training and needs the user's go-ahead.
- `data.meetings()` excludes unscheduled actions by default (separate treatment, never pooled). `rel_day = 0` is the announcement day.

## Questions and results

- Question IDs Q1–Q5 are permanent (Q2 is parked but kept for run continuity): never renumber or rename them. Definitions live in `core/fedcore/questions/`; `core/tests/test_registry.py` enforces the order and that every `spec.datasets` entry exists in the catalog.
- Pipelines live in the project that answers the question: Q1 (built) and Q2 (parked stub) in `dml/fedci/questions/`; Q3 in `encoder/` and `cde/`. Q4/Q5 have definitions only. Pipeline contract: `run(store, **params)`; all output goes through `ResultStore.save(...)`, nothing is written anywhere else.
- Runs are append-only under `<project>/results/<question key>/<run_id>/` with data fingerprints; never overwrite or delete runs. Tag `role` as `primary`/`robustness`/`exploratory`: one primary spec per active question, every other run adds to the multiple-testing count.

## Evaluation rules (enforced in `fedcore.protocol`)

- `walk_forward` splits by whole meetings in chronological order: never random folds, never one meeting on both sides. Use `embargo` when the outcome window (e.g. t+20) could reach the next meeting.
- `meeting_bootstrap` resamples whole meetings; report `n_meetings`, not row count.
- Select Q3 models on response-prediction performance, not downstream trading profits (docs/question.md).

## Gotchas

- Paths resolve from each package's `config.py` (`fedcore.config` for data, `fedci.config` for dml results and frontend), so code runs from any working directory.
- Viewer pages import `dml/frontend/streamlit/_common.py`; `streamlit run dml/frontend/streamlit/Home.py` puts that folder on `sys.path`, and `dml/tests/test_app.py` mirrors this. Keep page imports compatible with both.
- `fedci run` params are `k=v` pairs; a run's datasets record content hashes so results are traceable to exact data versions.
