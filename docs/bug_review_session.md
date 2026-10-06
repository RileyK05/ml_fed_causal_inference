# Bug review session

Full read of the codebase plus targeted verification. The full test suite passes
(116 tests: `fedcore check` and `pytest core dml encoder cde`), but several issues
are not covered by tests. Findings are ordered by severity. Nothing was changed.

Verification performed: full suite run; reproduced the `_fill_nan` behavior against
`torchcde.linear_interpolation_coeffs`; confirmed `Run` is unhashable; confirmed
`git check-ignore` on `core/results/`; confirmed no secrets/licensed data tracked.

---

## Critical / correctness

### 1. `_fill_nan` never fills — every masked path step is forced to 0
`cde/fedcde/encoder_cde.py:78-115`

The final line `out = torch.where(m.unsqueeze(-1), out, torch.zeros_like(out))`
(line 114) overwrites the computed leading/interior/trailing interpolation with
zeros at every masked position.

Reproduced: path `[0, 1, NaN, 3, NaN]`, mask `[T,T,F,T,F]` -> output `[0,1,0,3,0]`;
`torchcde.linear_interpolation_coeffs` gives `[0,1,2,3,3]`.

The docstring (lines 78-82) and the module header (lines 12-15) explicitly claim
equivalence with torchcde; that claim is false. Mask invariance still holds (the
result is deterministic), which is why tests pass. Impact: `NeuralCDEEncoder` feeds
the CDE a path with zeros at every missing day, and the Hermite cubic then oscillates
at those steps. Currently latent because `ARMS` only registers `cde_logsig` (which
uses `_window_logsig`, not `_fill_nan`), but it is a real defect in a shipped encoder.

### 2. "Pre-trained" encoder arms always pre-train on synthetic data — even for the real panel
`encoder/fedenc/arms.py:38-70, 128-134`

`get_pretrained` unconditionally calls `make_pretrain_windows(...)` (the synthetic
DGP). The docstring at lines 6-7 states "On real data a pre-trained checkpoint may
only serve test folds that start after its corpus ends," but no such guard exists and
there is no path to load a real, pre-cutoff checkpoint. So running
`run_arms({**ARMS}, data="real")` produces "pre-trained" arms whose encoder saw a
synthetic ordinary-day universe, not the real pre-test period.

`docs/training-plan.md` D3 confirms these pieces ("streaming pre-training from the
corpus, checkpoint loading in the arms") are *not built*. This is the most important
methodological misalignment: an arm named pretrained is not doing what the plan/spec
claims.

### 3. SQL console is not read-only (arbitrary file/network access)
`dml/frontend/streamlit/pages/3_SQL_Console.py:1,37` -> `core/fedcore/data/db.py:36-38`

User SQL is executed unparameterized on a DuckDB connection. DuckDB allows
`COPY ... TO`, `ATTACH`, `INSTALL/LOAD`, `read_csv('/abs/path')`, httpfs, etc. The page
text calls it "read-only views", which is false. `fedci app` runs Streamlit with no
auth, reachable on the LAN by default. Highest-impact security issue.

### 4. CSV formula injection in exports
`dml/frontend/src/csv.ts:4-6,26-28`

`escapeField` only quotes fields containing `,`/`"`/`\n`/`\r`. Cells beginning with
`=`, `+`, `-`, `@` (dataset descriptions, run `notes`, audit text pulled from the API)
execute as formulas when opened in Excel. No guard/prefix.

---

## Data leakage / point-in-time alignment

### 5. `yf_firms.py` CIK fallback breaks on float-typed CIK
`core/fedcore/ingest/yf_firms.py:69-71, 90-95`

`cik.astype(str).str.zfill(10)` turns float `12345.0` into `"0000012345.0"`, which
never matches Wikipedia's `"0000012345"`. PROVENANCE §8 claims 2 CIK matches, so
today's parquet must have string CIK; latent but real if re-pulled with NaNs.

Also ticker merge can fan out: `firm_bars.merge(firms[["ticker","permno"]], on="ticker")`
(line 130) assumes ticker->permno is 1:1, but `firms` is only `drop_duplicates("permno")`;
the "never reuse a symbol" guard (line 95) applies only to the CIK path.

### 6. `crsp_names` active window not bounded at the start
`yf_firms.py:77`

`names = names[pd.to_datetime(names["secinfoenddt"]) >= crsp_end]` never checks
`secinfostartdt <= crsp_end`, so a ticker whose validity begins after 2025-12-31 could
label a 2025 firm. (There is also a latent NaT risk: if `secinfoenddt` is NaT for
current names, this filter drops exactly the current tickers — PROVENANCE says it
didn't, so the column is populated in this vintage.)

### 7. 2025 holdout universe defined by today's Wikipedia list
`yf_firms.py:51-61`

Disclosed in PROVENANCE §8 as "a small, labeled survivorship bias," but it remains a
point-in-time violation if anyone treats the holdout as representative. Consistent
with docs, so flag-not-fix.

### 8. `walk_forward` has no default embargo
`core/fedcore/protocol.py:28`, used at `core/fedcore/q3/evaluate.py:60`

Safe for Q3 (same-day target), but any future t+k outcome would silently train through
overlapping windows. Q1 passes `embargo` explicitly; Q3 doesn't.

### 9. `_asof_link` ignores CRSP `linktype`
`core/fedcore/q3/real.py:184-188`

Prefers `linkprim` 'P' over 'C' but doesn't distinguish LU/LC. Minor link-selection risk.

### 10. `_asof_sector` backfill is a labeled look-ahead
`real.py:225-241`

Correctly flagged `gsector_source="backfill"` and kept out of model inputs. OK, but
confirm nothing downstream consumes `gsector` as a feature (currently it doesn't).

---

## Bugs / robustness

### 11. WRDS by-year pull is non-atomic and drops interior gap years
`core/fedcore/ingest/wrds.py:226-250`

Year files are written incrementally and the manifest only at the end; a crash leaves
orphaned parts with a stale manifest. `if part.empty: continue` (line 233) treats an
interior empty year as "vintage ended," silently skipping it.

### 12. WRDS conflates "unsubscribed library" with "missing table"
`wrds.py:167-203`

`_columns` swallows every exception, so a library you can't list reports as
`missing table`; `main` then aborts with "no usable source" (lines 269-271),
contradicting the docstring's discover contract.

### 13. `--only crsp_delist` silently no-ops
`wrds.py:219-222,269`

`crsp_delist` is exempt from the "missing" check, but if explicitly requested and
unresolved it prints "Done" with nothing written.

### 14. Streamlit: `fingerprint()` of every dataset on every rerun
`dml/frontend/streamlit/_common.py:77-78`

`coverage()` reads the full bytes of every dataset (all parquet parts of `crsp_daily`)
to build the `@st.cache_data` key *before* consulting the cache, defeating it.

### 15. Streamlit: `Run` objects passed to `selectbox`
`dml/frontend/streamlit/pages/5_Results.py:27`

Confirmed `Run` is unhashable (`manifest` is a dict). It renders today but will crash
any code path that hashes session/options state. Pass `run_id` strings.

### 16. Frontend: `localStorage` in a `useState` initializer with no try/catch
`dml/frontend/src/theme.tsx:12`

Throws `SecurityError` in sandboxed/incognito contexts and blanks the SPA on first
render.

### 17. Frontend: FastAPI 422 `detail` is an array, not a string
`dml/frontend/src/api.ts:6`

`errorDetail` only renders string details, so validation errors show raw JSON. Also no
`encodeURIComponent` on `question`/`runId` in URLs (api.ts:47,51).

### 18. Streamlit empty-state crashes
`1_Data_Catalog.py:13-16` and `2_Event_Explorer.py:18-23`

A filter with no datasets / a calendar covering no meetings yields
`st.selectbox(...) is None`, then `cat[name]` / `event_panel(events=[None])` raises
`KeyError` instead of a friendly message.

### 19. `_metrics_one` seed averaging then `compare`
`core/fedcore/q3/evaluate.py:296-360`

Logic is correct, but `_paired_meeting_diff` averages MSE across seeds per meeting
before bootstrapping, so the CI reflects meeting variation only, not seed variation.
Consistent with the report; just be aware the seed spread is not in the interval.

---

## Documentation vs code (all verified)

- `docs/agent-guide.md:18` — "Synthetic data only. No real firm data exists yet."
  False: `core/fedcore/q3/real.py`, `load_real_panel`, and the 2026 holdout are built.
- `docs/core/HANDOFF.md:207` and `docs/reports/core-report.md:172` — claim
  `run_arms(data="real")` raises `NotImplementedError("real Q3 panel not built yet")`.
  The code implements it (`evaluate.py:170-177`).
- `docs/encoder/HANDOFF.md:26,85` and `docs/cde/HANDOFF.md:41,107` — reference
  `fedenc/run.py` / `fedcde/run.py`, which don't exist (`arms.py`).
- `docs/cde/HANDOFF.md:107` — says `ARMS` contains `sig_ridge`, `cde` (v1),
  `cde_logsig`; actual `ARMS` is only `sig_ridge`, `cde_logsig`
  (`cde/fedcde/arms.py:78-81`).
- `docs/cde/HANDOFF.md:59-64` — describes torchcde interpolating NaNs; the code does
  its own (buggy) fill.
- `docs/reports/core-report.md:24` — "16/16 datasets pass"; catalog now has 24.
  Point-in-time artifact but presented as current.
- `dml/frontend/src/components/Explain.tsx:326` — says `r2_s` "≈ 0.007" in the primary
  run; the saved manifest has `r2_s = -0.0027`.
- `docs/agent-guide.md:64` — names interface `encode(history, mask)`; actual contract
  is `HistoryEncoder.forward(x, mask)`.
- `docs/question.md` / `docs/methods.md` are consistent with the code's estimands and
  protocol. Q1 docstrings match.

---

## Process / policy

- `AGENTS.md` says "no ... benchmarks, `run_arms`, or anything that saves a run"
  without explicit go-ahead. `docs/reports/core-report.md:38-74` documents running
  `run_arms` 8 times (`core/results/q3_susceptibility/...`, gitignored/untracked).
  Synthetic-only, but it directly contradicts the current policy. Either the policy
  postdates the runs or it was violated; worth reconciling the record.
- `run_arms` does not enforce project isolation. The store root is the caller's
  responsibility; the core report used `ResultStore(ROOT/"results")` (= `core/results/`),
  which `AGENTS.md` forbids ("Each project keeps its own `results/`"). Nothing in the
  harness prevents a Q3 run from landing under `core/`.

---

## Verified-clean areas

- Full suite passes (116): `fedcore check` + all four projects.
- Isolation rule is enforced and holds (`core/tests/test_isolation.py`).
- No secrets, `.env`, pgpass, or WRDS/yfinance data are tracked; licensed paths are
  correctly gitignored.
- WRDS credential handling is clean (pgpass only; password never logged/asked).
- Mask invariance holds in `PatchTransformerEncoder.patchify`, `SummaryEncoder`,
  `build_path`, and `_window_logsig` (each reproduced). No masked-value leakage into
  model inputs.
- Point-in-time joins in `real.py` (`rdq` with `allow_exact_matches=False` + 365-day
  staleness; 1y yield as-of with `allow_exact_matches=False`) are correct.
- React hook ordering is correct; no `dangerouslySetInnerHTML`/`eval`; backend API
  field names match the TS types exactly.
- `walk_forward` splits whole meetings chronologically and `meeting_bootstrap`
  resamples meetings, per spec.

---

## Suggested fix order

1. #1 `_fill_nan` (genuine numerical bug).
2. #2 pretrained arm not matching its documented contract.
3. #3 SQL console exposure.
4. Docs drift (#agent-guide, HANDOFFs, core-report, Explain text).
