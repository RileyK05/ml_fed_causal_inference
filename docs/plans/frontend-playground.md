# Plan: FastAPI + SPA estimation playground for Q1

**Goal:** a browser UI where every Q1 DML knob is adjustable in real time — treatment,
outcome, nuisance learner, fold geometry, inference method, and the control feature set
via checkboxes — returning graphs (plotly) and CSV-downloadable tables. Plus a read-only
Runs browser over the existing ResultStore.

**Execution notes for the implementing agent**
- Repo root: `c:\Users\moomi\personal_projects\ml_fed_causal_inference`. Windows; Git Bash for
  shell; venv at `.venv\` (invoke as `.venv/Scripts/python`, `.venv/Scripts/pip`).
- Verify with: `fedci check` then `.venv/Scripts/python -m pytest -q --basetemp=$TMPDIR/fedci-pytest`
  (the default `Temp\pytest-of-moomi` root is broken; the `--basetemp` flag is mandatory if you
  see PermissionError).
- The backend package is `src/backend/fedci/` (installed editable, `where = ["src/backend"]`).
- Follow existing code style: module docstrings that cite `docs/question.md`/`docs/methods.md`,
  type hints, `from __future__ import annotations`, no comments restating code.
- **Do not** delete or modify the Streamlit pages, **do not** write playground results into
  `results/` (playground is ephemeral; only `fedci run q1 ...` saves runs).

---

## 1. Current state (read before writing code)

- Estimator: `src/backend/fedci/eval/dml.py` — `partially_linear_dml(df, y, s, x, date, min_train,
  test_size, step, embargo, nuisance, nuisance_kwargs, se, hac_lag, n_boot, seed, level) -> PLRResult`
  (dataclass with `.estimate`, `.residuals`, `.folds`, `.bootstrap`, `.r2_y`, `.r2_s`, `.sd_s_resid`,
  `.n_meetings_total/discarded/dropped`). Also `ols_event_study(...) -> Estimate`,
  `leave_one_out(residuals) -> DataFrame`.
- Q1 pipeline: `src/backend/fedci/questions/q1_total_effect/pipeline.py` — `_parse(params)` (defaults
  + validation), `run(store, **params)` (builds sample → fits DML → OLS baselines → LOO → audit df →
  residualized-fit figure → `store.save(...)`). Samples: `samples.py` — `SURPRISES=("STMT","MP1","ME")`,
  `OUTCOMES=("usmpd_sp500","etf_day0")`, `usmpd_window_sample(treatment)`, `etf_day0_sample(treatment, outcome)`.
- Control features (checkbox source of truth, in `samples.py`):
  - `usmpd_sp500`: `y1_level, y1_chg_21d, y1_chg_63d, st_prev, me_prev, st_mean3`
  - `etf_day0`: `lagged_VIXCLS, lagged_DGS2, lagged_DGS10, val_1yr, val_5yr, val_10yr`
  (module constants `_CONTROL_X`, `_ETF_X`).
- Results store: `src/backend/fedci/results/store.py` — `ResultStore().summary(question=None) ->
  DataFrame`, `.runs(question=None) -> list[Run]`; `Run` has `.run_id`, `.manifest`, `.table(name) -> path`,
  `.figure(name) -> path`. Tables are CSV under `<run>/tables/`, figures are plotly JSON under `<run>/figures/`.
- Chart palette/theme constants currently live in `src/frontend/_common.py` (Streamlit) — copy the
  palette hex values into the new backend figures module; do not import from the Streamlit folder.
- CLI: `src/backend/fedci/cli.py`, `cmd_app` at line ~68 launches Streamlit. Add a `serve` subcommand.

## 2. Target layout changes

```
src/
  backend/fedci/
    api/                      NEW
      __init__.py             create_app() factory
      schemas.py              pydantic request/response models
      q1.py                   /api/q1/config + /api/q1/estimate
      runs.py                 /api/runs routes
      figures.py              shared plotly figure builders (also used by pipeline)
      serialize.py            DataFrame -> JSON-safe records helper
    questions/q1_total_effect/
      pipeline.py             refactored: estimate(**params) extracted from run()
      samples.py              + FEATURE_INFO dict
    cli.py                    + cmd_serve
  frontend/
    index.html                NEW (Vite entry)
    package.json  vite.config.ts  tsconfig.json  src/...   NEW (React SPA)
    streamlit/                MOVED from src/frontend/{Home.py,_common.py,pages/}
docs/plans/frontend-playground.md   (this file)
tests/test_api.py             NEW
pyproject.toml                + fastapi, uvicorn deps
```

Move first: `mkdir src/frontend/streamlit && git mv`-equivalent (no git repo — plain `mv`)
`src/frontend/Home.py src/frontend/_common.py src/frontend/pages src/frontend/streamlit/`.
Then fix the three references:
- `src/backend/fedci/cli.py` (`cmd_app`): `... / "src" / "frontend" / "Home.py"` → `... / "streamlit" / "Home.py"`.
- `tests/test_app.py`: `APP = ... / "src" / "frontend"` → `... / "src" / "frontend" / "streamlit"`.
- `AGENTS.md` gotcha line: `src/frontend/_common.py` → `src/frontend/streamlit/_common.py`, same for Home.py.
  README layout tree: `frontend/` line becomes two lines (`streamlit/` legacy viewer + web app).

Bare `import _common` keeps working because all Streamlit files move together (same-dir import).

## 3. Backend

### 3.1 Dependencies — `pyproject.toml`
Add to `dependencies`: `"fastapi>=0.115"`, `"uvicorn>=0.30"`. Reinstall:
`.venv/Scripts/python -m pip install -e ".[dev,model]"`.

### 3.2 Refactor `pipeline.py`: extract `estimate(**params)`
Split `run()` so the computation is reusable without saving:

```python
def estimate(**params) -> dict:
    """Fit Q1 without saving. Returns dict with keys:
    resolved (dict of parsed params incl. computed min_train/test_size/step),
    sources (list[str]), metrics (dict), estimates/audit/folds/residuals/influence (DataFrames),
    figure (go.Figure), notes (str)."""
```

`run(store, **params)` becomes: `e = estimate(**params)` then the existing `store.save(...)`
call using `e["resolved"]`, `e["sources"]`, `e["metrics"]`, `e["tables"]` etc. Behavior must be
identical — `tests/test_questions.py` roundtrip and `tests/test_results.py` must pass unmodified.

**New param:** `features: list[str] | None = None` (accepted by both `_parse` and `estimate`).
When `None` → all sample controls (current behavior). Otherwise must be a non-empty subset of the
sample's control columns; `x = features` in the `partially_linear_dml` and OLS-control calls, and
`audit` reflects the actual `x` used (it already does via the `x` variable).

### 3.3 `samples.py`: feature metadata
Add (descriptions one line each, consistent with the module docstring):

```python
FEATURE_INFO = {
    "usmpd_sp500": {"y1_level": "...", "y1_chg_21d": "...", "y1_chg_63d": "...",
                    "st_prev": "...", "me_prev": "...", "st_mean3": "..."},
    "etf_day0": {"lagged_VIXCLS": "...", "lagged_DGS2": "...", "lagged_DGS10": "...",
                 "val_1yr": "...", "val_5yr": "...", "val_10yr": "..."},
}
```

### 3.4 `figures.py` (shared plotly builders)
- `residualized_fit_figure(scored: pd.DataFrame, theta: float, title: str, x_title: str, y_title: str) -> go.Figure`
  — port the exact figure `pipeline.py` currently builds (markers with date hover text, theta line,
  `template="plotly_white"`). Pipeline switches to calling this.
- `influence_figure(influence: pd.DataFrame) -> go.Figure` — horizontal bar of `delta_loo` by
  `announcement_date` (top-N already passed in), `template="plotly_white"`.
- Palette constants copied from `src/frontend/streamlit/_common.py` (`CATEGORICAL["light"]` etc.),
  used for the marker/line colors.

### 3.5 `serialize.py`
```python
def records(df: pd.DataFrame) -> list[dict]:
    """JSON-safe records: ISO dates, NaN -> None, numbers stay numbers.
    Implement via json.loads(df.to_json(orient="records", date_format="iso"))."""
def figure_json(fig: go.Figure) -> dict:
    """json.loads(fig.to_json())"""
```

### 3.6 `schemas.py` (pydantic)
```python
class EstimateRequest(BaseModel):
    outcome: Literal["usmpd_sp500", "etf_day0"] = "usmpd_sp500"
    treatment: Literal["STMT", "MP1", "ME"] = "STMT"
    outcome_ticker: str = "SPY"
    features: list[str] = Field(min_length=1)
    nuisance: Literal["ridge", "enet", "rf"] = "ridge"
    alpha: float | None = None
    min_train: int | None = Field(default=None, ge=2)
    test_size: int | None = Field(default=None, ge=1)
    step: int | None = Field(default=None, ge=1)
    embargo: int = Field(default=0, ge=0)
    se: Literal["HC1", "HAC"] = "HC1"
    hac_lag: int = Field(default=4, ge=1)
    n_boot: int = Field(default=400, ge=0)
    seed: int = 0
    level: float = Field(default=0.95, gt=0.5, lt=1.0)
```
Response models can be loose (`dict` / `list[dict]`) — do not over-model; keep one `EstimateResponse`
typed as a pydantic model with `params: dict, metrics: dict, estimates: list[dict], audit: list[dict],
folds: list[dict], residuals: list[dict], influence: list[dict], figures: dict[str, dict]`.

### 3.7 `q1.py` routes
- `GET /api/q1/config` →
  ```json
  {"surprises": ["STMT","MP1","ME"], "outcomes": ["usmpd_sp500","etf_day0"],
   "nuisances": ["ridge","enet","rf"], "se_types": ["HC1","HAC"],
   "defaults": {"outcome":"usmpd_sp500","treatment":"STMT","outcome_ticker":"SPY","nuisance":"ridge",
                "alpha":null,"min_train":null,"test_size":null,"step":null,"embargo":0,
                "se":"HC1","hac_lag":4,"n_boot":400,"seed":0,"level":0.95},
   "features": {"usmpd_sp500": [...], "etf_day0": [...]},
   "feature_info": {...}, "tickers": ["SPY", ...]}
  ```
  `tickers`: column names of `event_study_table` starting with `ret_`, prefix stripped
  (compute once, `functools.lru_cache`). Wrap data loading failures in `HTTPException(500)`.
- `POST /api/q1/estimate` — body `EstimateRequest`. Steps:
  1. Load the sample (`usmpd_window_sample(treatment)` / `etf_day0_sample(treatment, outcome_ticker)`).
  2. Validate `features ⊆ sample control columns`, else `HTTPException(422, f"unknown features {...}; allowed: ...")`.
  3. Call `pipeline.estimate(**req.model_dump())` (pass `features` through; `_parse` validates the rest).
  4. Catch `ValueError` from the estimator → `HTTPException(422, str(e))`.
  5. Build figures: `residualized_fit_figure(...)`, `influence_figure(...)`.
  6. Return `EstimateResponse` with `records(...)` on every DataFrame and `figure_json(...)` on figures.
  7. **Cache:** module-level `OrderedDict` keyed by `req.model_dump_json()` (canonical), max 32 entries,
     FIFO eviction. Estimation is seconds for ridge, up to ~1 min for `rf` — the cache makes knob
     flips that were already run instant.

### 3.8 `runs.py` routes
- `GET /api/runs` → `{"runs": records(ResultStore().summary())}` (may be empty list).
- `GET /api/runs/{question}/{run_id}` → find via `ResultStore().runs(question=question.upper())`
  matching `run_id`; 404 if absent. Response:
  ```json
  {"manifest": <manifest.json contents>, "tables": {name: records(read csv)},
   "figures": {name: figure_json(read json file)}}
  ```
  Table/figure names from `manifest["tables"]` / `manifest["figures"]` keys (check store.py for exact
  manifest key names; fall back to globbing `tables/*.csv`, `figures/*.json`).

### 3.9 `api/__init__.py`
```python
def create_app() -> FastAPI:
    app = FastAPI(title="fedci", ...)
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"], allow_headers=["*"])
    app.include_router(q1.router, prefix="/api")
    app.include_router(runs.router, prefix="/api")
    dist = ROOT / "src" / "frontend" / "dist"
    if dist.exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="web")
    return app
```
`ROOT` from `fedci.config`. Endpoints as plain `def` (threadpool) — estimation blocks, that's fine.

### 3.10 CLI — `cli.py`
Add `cmd_serve(a)`: `import uvicorn; from fedci.api import create_app; uvicorn.run(create_app(), host=a.host, port=a.port)`.
Register: `s = sub.add_parser("serve"); s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8000)`.
Add to the module docstring/CLI list and to README's command table + AGENTS.md CLI line:
`fedci check | catalog | sql ... | run q1 k=v | runs [q1] | app | serve`.

## 4. Frontend (React + TypeScript + Vite)

Location: `src/frontend/` (alongside `streamlit/`). Keep dependencies minimal —
`react`, `react-dom`, `plotly.js-dist-min`; dev: `vite`, `@vitejs/plugin-react`, `typescript`,
`@types/react`, `@types/react-dom`. **No** tailwind/router/state libraries.

- `package.json` (`name: "fedci-frontend"`, `private: true`): scripts
  `dev: "vite"`, `build: "tsc --noEmit && vite build"`, `preview: "vite preview"`.
- `vite.config.ts`: `server.proxy = { "/api": "http://localhost:8000" }`, `build.outDir: "dist"`.
- `index.html`: single `<div id="root">`, title "fedci".
- `src/main.tsx`, `src/App.tsx` — two tabs: **Playground** | **Runs** (plain `useState`, no router).
- `src/api.ts` — typed fetch wrappers: `getConfig()`, `estimate(req)`, `listRuns()`, `getRun(q, id)`;
  on non-2xx parse `{detail}` and throw `Error(detail)`.
- `src/csv.ts` — `downloadCsv(filename, rows)`: rows→CSV string (quote fields containing `,`/`"`/newline;
  `null`→empty), Blob + `URL.createObjectURL` + `<a download>` click.
- `src/components/Plot.tsx` — wrapper around `plotly.js-dist-min`:
  `useRef` div + `useEffect(() => { Plotly.react(el, fig.data, fig.layout, {responsive: true}) }, [fig])`
  + `Plotly.purge` on unmount. ~30 lines.
- `src/components/Playground.tsx`:
  - Left control panel (form state initialized from `/api/q1/config`):
    - selects: treatment, outcome, outcome_ticker (visible only when outcome=etf_day0), nuisance, se
    - number inputs: min_train, test_size, step (each with an "auto" checkbox that nulls the value),
      embargo, hac_lag (disabled unless se=HAC), n_boot, alpha (blank = learner default), seed, level
    - feature checkboxes per current outcome (with Select all / Clear); submitting with zero checked
      → client-side error message
    - **Run** button → POST; while pending show "estimating…" and disable the button.
  - Right results panel (rendered from the response):
    - headline card: theta (se), CI, se_type, n_meetings scored/discarded, r2_y, r2_s, sd_s_resid
    - estimates table (PLR-DML, bootstrap row, both OLS rows)
    - `Plot` residualized fit + `Plot` influence bar
    - collapsible folds table and audit table
    - download buttons: `estimates.csv`, `residuals.csv`, `folds.csv`, `influence.csv`, `audit.csv`
      (from response JSON via `downloadCsv`)
    - 422 errors rendered as a dismissible red box showing the backend message.
- `src/components/Runs.tsx`: table from `/api/runs` (run_id, name, role, created, theta columns when
  present); row click → detail view calling `/api/runs/{question}/{run_id}`: metrics/manifest table,
  each saved figure via `Plot`, each saved table rendered as an HTML `<table>` (cap at 500 rows,
  "download CSV" per table).
- `src/styles.css`: single small stylesheet, dark-on-white, system font; a `.btn`, `.card`, `.table`,
  `.error` classes. Keep it ~100 lines, plain CSS.

Types: define `Config`, `EstimateRequest`, `EstimateResponse` interfaces in `src/types.ts` mirroring
§3.6/§3.7 field names exactly.

## 5. Tests — `tests/test_api.py`

Use `from fastapi.testclient import TestClient` with `create_app()` (module-scoped client).
Do **not** hit the static mount or streamlit. Cases:
1. `GET /api/q1/config` → 200; `features["usmpd_sp500"]` has 6 entries; `tickers` non-empty.
2. `POST /api/q1/estimate` with `{outcome:"usmpd_sp500", treatment:"STMT", features:["y1_level","st_prev"],
   nuisance:"ridge", min_train:30, test_size:30, n_boot:50, se:"HC1"}` → 200; response has all 8
   top-level keys; `metrics.theta` finite; `len(residuals) == metrics.n_meetings`;
   `len(estimates) == 4` (2 OLS + DML + bootstrap).
3. Unknown feature → 422. Empty `features` → 422 (pydantic min_length). `features=["y1_level"]`
   (single control) → 200, works.
4. `se="HAC", hac_lag=2` → 200 and `metrics.se_type == "HAC(2)"`.
5. `n_boot=0` → 200 and bootstrap row absent from `estimates`.
6. `embargo=999` (or min_train=2 with tiny data) → 422 with the estimator's message surfaced.
7. `GET /api/runs` → 200, `runs` is a list with ≥1 entry in this repo.
8. `GET /api/runs/Q1/<first run_id>` → 200; has `manifest`, `tables`, `figures`;
   `figures.residualized_fit` has `data` and `layout`.
Keep total added runtime under ~60s (ridge + n_boot=50 is a few seconds).

## 6. Verification checklist (run in order)

1. `.venv/Scripts/python -m pip install -e ".[dev,model]"` then `fedci check` — 16/16 datasets.
2. `.venv/Scripts/python -m pytest -q --basetemp=$TMPDIR/fedci-pytest` — all green, including old
   Streamlit render tests (they now point at `src/frontend/streamlit/`).
3. `fedci serve` (background); `curl localhost:8000/api/q1/config | head`; POST an estimate via curl;
   `curl localhost:8000/api/runs`. Stop the server.
4. `cd src/frontend && npm install && npm run build` → `dist/` exists.
5. Re-run `fedci serve`; open `http://localhost:8000/` → the built SPA loads, config populates,
   a ridge estimate renders figures + tables, CSV downloads work. (`fedci app` still launches the
   legacy Streamlit viewer — parity check during transition.)
6. Dev loop: `fedci serve --port 8000` + `npm run dev` (port 5173, proxies /api).

## 7. Rollout / rollback

- Streamlit stays fully working until the SPA reaches parity; deleting it is a separate step.
- The only behavioral change to existing code is the `pipeline.py` extraction — covered by the
  existing Q1 roundtrip test; if anything else regresses, revert `pipeline.py` only.

## 8. Out of scope (do not build)

- Saving playground configs as runs (no writes to `results/` from the API).
- The other Streamlit pages' SPA equivalents beyond the Runs browser (data catalog, SQL console,
  event explorer) — later increments behind the same `/api` seam.
- Auth, multi-user, anything networked beyond localhost.
- Tauri packaging.
