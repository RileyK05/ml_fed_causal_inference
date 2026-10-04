# Implementation prompt: fedci estimation playground (hand to coding agent)

You are implementing a planned feature in an existing research codebase. Your job is to follow
the plan exactly, verify it end-to-end, and report honestly. Do not redesign anything.

## Context

Repo: `c:\Users\moomi\personal_projects\ml_fed_causal_inference` (Windows, Git Bash available,
NOT a git repo — use plain `mv`, never `git mv`). Python venv at `.venv\` — always invoke as
`.venv/Scripts/python` and `.venv/Scripts/pip`. Node/npm must be used from Git Bash for the
frontend build.

The backend package is `fedci`, physically at `src/backend/fedci/`, installed editable with
setuptools `where = ["src/backend"]`. The CLI entry point is `fedci = fedci.cli:main`. A legacy
Streamlit viewer currently lives at `src/frontend/` and must keep working throughout — you are
moving it, not deleting it.

## Read these files first, in this order

1. `AGENTS.md` (repo rules — follow them; they are binding)
2. `docs/plans/frontend-playground.md` (the plan — your specification; implement it section by section)
3. `src/backend/fedci/eval/dml.py`, `src/backend/fedci/eval/protocol.py` (the estimator)
4. `src/backend/fedci/questions/q1_total_effect/pipeline.py` and `samples.py` (what you will refactor)
5. `src/backend/fedci/results/store.py` (the runs API you will expose read-only)
6. `src/backend/fedci/cli.py` and `pyproject.toml`

## What you are building

A FastAPI backend (`fedci/api/`) exposing: Q1 config, on-demand Q1 DML estimation with adjustable
parameters (treatment, outcome, nuisance learner, fold geometry, inference method, bootstrap,
and a user-selected feature subset), and read-only access to the existing results store. Plus a
React + TypeScript + Vite SPA in `src/frontend/` (two tabs: an estimation playground with plots
and CSV downloads, and a runs browser). The Streamlit app moves to `src/frontend/streamlit/`
with its three references updated (cli.py, tests/test_app.py, AGENTS.md).

## Ground rules — what NOT to do

- Do NOT modify anything under `data/` or `results/`. Never write playground/estimation output
  into `results/` — only `fedci run` writes runs.
- Do NOT delete or disable the Streamlit pages, tests, or any existing behavior. The only
  behavioral change permitted is the `pipeline.py` extraction, and existing tests must pass
  unmodified.
- Do NOT add Python dependencies beyond `fastapi` and `uvicorn` (to `pyproject.toml`
  `dependencies`), and do NOT add npm packages beyond: `react`, `react-dom`, `plotly.js-dist-min`
  (+ dev: `vite`, `@vitejs/plugin-react`, `typescript`, `@types/react`, `@types/react-dom`).
- Do NOT add linting, formatting, CI, Docker, or any config the repo doesn't already have.
  Do NOT reorganize directories beyond what the plan specifies.
- Do NOT commit anything (there is no git repo).
- Do NOT paper over failures. If a plan step turns out to be wrong or blocked (missing manifest
  key, npm failure, estimator raising on valid input), STOP and report the exact error and what
  you tried. One reasonable alternate fix is fine; two failed attempts means stop and report.
- Match the existing code style: module docstrings citing `docs/question.md`, type hints,
  `from __future__ import annotations`, no decorative comments.

## Execution order

Follow the plan's sections in order: (1) move Streamlit + fix its three references, (2) pyproject
deps + editable reinstall, (3) the `pipeline.py` `estimate()` extraction with the new `features`
param, (4) `samples.py` FEATURE_INFO, (5) `fedci/api/` package (serialize, schemas, figures, q1,
runs, `create_app()`), (6) `fedci serve` CLI subcommand, (7) `tests/test_api.py`, (8) the SPA
(scaffold, api client, components, styles), (9) prose updates (README command table + layout,
AGENTS.md CLI line). Build backend fully and test it before touching the frontend.

## Verification — all of this must pass before you declare done

1. `.venv/Scripts/python -m pip install -e ".[dev,model]"` succeeds, then `fedci check` prints
   `16/16 datasets pass`.
2. `.venv/Scripts/python -m pytest -q --basetemp=$TMPDIR/fedci-pytest` — full suite green,
   including the pre-existing Streamlit render tests. (The default pytest temp root
   `Temp\pytest-of-moomi` is broken on this machine; if you see PermissionError there, that
   basetemp flag is mandatory, not optional.)
3. `fedci serve` starts; `curl -s localhost:8000/api/q1/config` returns JSON;
   a POST to `/api/q1/estimate` (usmpd_sp500, STMT, ridge, features=["y1_level","st_prev"],
   min_train=30, test_size=30, n_boot=50) returns 200 with all response keys; `curl -s
   localhost:8000/api/runs` returns the existing runs.
4. `cd src/frontend && npm install && npm run build` produces `dist/`.
5. `fedci serve` again: `http://localhost:8000/` serves the built SPA; the config populates the
   controls; clicking Run renders the estimate, both charts, and the tables; CSV downloads work;
   the Runs tab lists the three existing runs and shows run detail. Also confirm `fedci app`
   still launches the Streamlit viewer.

## Technical traps you will hit (handle these proactively)

- After editing `pyproject.toml` you MUST reinstall editable or imports will silently use stale
  metadata; a new `fedci/api/` subpackage is auto-discovered, no packaging config needed.
- Pandas Timestamps and NaNs are not JSON-serializable — use the plan's `serialize.records()`
  (to_json with `date_format="iso"`) everywhere a DataFrame crosses the API.
- Plotly figures cross as `json.loads(fig.to_json())`; in React use `Plotly.react` with a ref div
  and `Plotly.purge` on unmount (wrapper component is in the plan).
- The Vite dev server proxies `/api` to port 8000; the built app is served by FastAPI's static
  mount — test both paths.
- The Streamlit pages use a bare `import _common`; it only works because `_common.py` sits next
  to `Home.py` — move all Streamlit files together or every render test breaks.
- `partially_linear_dml` raises `ValueError` on bad fold geometry (e.g. embargo ≥ min_train,
  fewer than 3 scored meetings) — catch and return HTTP 422 with the message, never a 500
  traceback, and the frontend must display that message.
- `rf` nuisance can take ~1 minute per estimate — that is expected, not a hang.

## Final report format

When done (or blocked), report: (a) files created/moved/modified as a tree diff, (b) verbatim
output of the pytest summary line and `fedci check` tail, (c) confirmation each verification
step passed or the exact step where you stopped, with the full error, (d) how to run it
(`fedci serve` + optional `npm run dev`), (e) any place where you deviated from the plan and why
(should be "none"). Do not claim steps passed that you did not run.
