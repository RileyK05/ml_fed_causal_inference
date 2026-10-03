# Package 02: signature baseline and Neural CDE history encoder

**Goal:** build the continuous-time alternative to the transformer. Two arms, both plugging into
package 00's contract:

1. **Signature + ridge:** the linear, no-training version (truncated path signature features).
2. **Neural CDE:** a learned hidden state driven by the history path, as a `HistoryEncoder`.

Then test on the synthetic panel whether either beats the baselines and `summary_nn`, and
whether the neural version beats its own linear cousin.

Prerequisite: package 00 is merged. Read first: `AGENTS.md`, `encoder/docs/transformer.md` sections 1,
6, 9-12, `contracts.py`, `model.py`, `train.py` and `evaluate.py` in `core/fedcore/q3/`, and
`core/docs/reports/q3-foundation-report.md` (the baseline numbers to beat). Background: Kidger et al.
(2020), *Neural Controlled Differential Equations for Irregular Time Series*, and the
`torchcde` README.

## Background in one paragraph

A Neural CDE evolves a hidden state h along the data path X:
dh = f_θ(h) dX, so h(T) = h(0) + ∫ f_θ(h(t)) dX(t). Here f_θ is an MLP returning a matrix
(hidden × channels). Discretized, it's a residual RNN driven by the increments ΔX. Missing days
simply widen an increment, so irregular data is handled naturally. The signature of a path (its
iterated integrals) is the basis in which CDE solutions expand, so **ridge on truncated
signature features is the "linear CDE"** and the right first baseline.

## Files you own

Everything lives in the **cde/** project. It imports `fedcore` (including `fedcore.q3`) and
never imports `dml/` or `encoder/` (`core/tests/test_isolation.py` enforces this).

```text
cde/fedcde/signature.py
cde/fedcde/encoder_cde.py
cde/fedcde/run.py              # run_arms({**BASELINES, **ARMS}, ResultStore(RESULTS))
cde/tests/test_cde.py
cde/pyproject.toml             # add libraries to the dl extra if needed (torchcde is already there)
```

Never edit `core/` (including `fedcore/q3/`), `dml/` or `encoder/`. Request contract changes in
your report. Install with `pip install -e "core[dev,q3]" -e "cde[dev,dl]"`.
**A Neural SDE is out of scope.**

Dependencies: `torchcde` (pure Python on top of torch). For signatures, try `iisignature`. If
it won't install on Windows, implement a truncated signature up to depth 3 yourself in numpy,
with a unit test against a hand-computed 2-D example (depth-2 terms = increments plus Lévy
area). Record which one you used.

## 1. Path construction (shared by both arms)

`build_path(x, mask, *, window=252, add_time=True, add_obs_count=True) -> tensor (B, T, C')`:
- Use the scaled history, taking the last `window` days.
- **Channels:** the C history channels, cumulated where it makes sense for a path
  (cumulative return for `ret` and `ret_rel`; levels for `rvol21` and `beta63`; cumulative
  sum for `dlogvol`). Plus a **time channel** t ∈ [0, 1] (`add_time`), and a **cumulative
  observation count** channel (`add_obs_count`, the "observational intensity" trick from
  Kidger et al.) so that missingness carries information.
- Unobserved days are NaN in the path. `torchcde` coefficient builders interpolate across NaNs.
  Leading padding before a firm's first observed day: start the path at its first observed day
  (pass per-row start indices, or forward-fill the first observation backwards **only for the
  path's initial value** and document it). Test this case explicitly.

Note: without the time channel a CDE is invariant to time reparametrization (it sees only the
shape of the path, not its speed). Keep the time channel on by default; ablate it in M4.

## 2. Arm `sig_ridge` (`signature.py`)

- Path → depth-3 truncated signature, with a **lead-lag** transform as an option, over the full
  window and over the last 63 days (two feature blocks). Standardize the features on training
  rows only.
- Model: ridge on [sig, fundamentals, context, s·sig, s·fundamentals, s·context, s]. b_hat is
  the s-interaction part evaluated per row, the same construction as `ridge_interact` in
  package 00.
- Choose the ridge alpha on the trainer's validation slice (last 15% of training meetings).
- Wrap it in the same fit/predict adapter the baselines use so `run_walk_forward` scores it.

## 3. Arm `cde` (`encoder_cde.py`)

`NeuralCDEEncoder(HistoryEncoder)`:

| Part | v1 setting |
|---|---|
| Interpolation | `torchcde.hermite_cubic_coefficients_with_backward_differences` (ablate linear in M4) |
| Initial state | h(0) = `Linear(C', hidden)` applied to X(t₀) |
| Vector field f_θ | MLP: hidden → 128 → 128 → hidden·C', ReLU between layers, **tanh** on the output, reshaped to (hidden, C') |
| Hidden size | 32 |
| Solver | `torchcde.cdeint(..., method="rk4", options={"step_size": 1.0})` over the day grid; no adjoint (the model is small) |
| Readout | `Linear(hidden, 64)` on h(T), then LayerNorm → `out_dim=64` |

The forward pass must satisfy the contract: `forward(x, mask) -> (B, 64)`, with no dependence on
values where mask=False. Build the path inside `forward` from (x, mask).

**Speed:** 252 RK4 steps per row is the cost driver. For development use `window=126` and a
small panel. If full-size epochs are too slow on CPU, the documented fallback is the
**log-signature / "log-ODE" method**: take log-signatures over windows of about 10 days so the
solver takes about 25 big steps instead of 252 (see `torchcde` docs and Morrill et al. 2021).
`torchcde.logsig_windows` needs the `signatory` package, which often won't install on Windows.
If it doesn't, compute depth-2 log-signatures per window yourself (increments plus Lévy areas)
and feed those as the path. Report wall-clock time per epoch for each setting.

Arms (in `fedcde/run.py` as `ARMS`): `sig_ridge`, `cde` (v1), and `cde_logsig` if you implement the fallback.

## 4. Milestones (in order; stop and report if one fails)

**M1: the modules work.** Tests in `cde/tests/test_cde.py`:
- signature correctness on a hand-computed tiny path;
- output shapes; no NaN with missing days and left padding;
- **mask invariance:** changing values where mask=False leaves h unchanged (atol 1e-5);
- path construction: the time channel is monotone, the observation count increases only on
  observed days.

**M2: linear first.** `sig_ridge` vs `ridge_interact` vs `summary_nn` on `noise="easy"` and
`noise="realistic"`, 3 seeds. Do signatures pick up the planted sequence pattern better than
the hand-built summary features?

**M3: neural.** `cde` vs `sig_ridge` vs the baselines, both noise settings, 3 seeds. Key
questions:
- Does the Neural CDE beat its linear cousin `sig_ridge`? If not, the nonlinearity isn't paying
  for itself on this task. That's a useful answer.
- Split `b_corr` by rows with and without the planted sequence pattern.

**M4: small ablations** (pre-declare the grid in the report, about 8 runs): hidden ∈ {16, 32, 64};
interpolation ∈ {cubic, linear}; time channel on/off; window ∈ {126, 252}.

**M5: practical profile.** Parameter count, seconds per epoch, peak memory and
stability notes (did the solver ever blow up? Was gradient clipping needed beyond the trainer's?).

## Exit criteria

- M1-M3 complete; M4-M5 attempted.
- Tests pass; `fedcore check`, `pytest core` and `pytest cde` are green.
- Runs saved to `cde/results/` through `ResultStore` (`role="exploratory"`).
- `cde/docs/report.md` contains: the results table (both arms + baselines, easy and
  realistic, mean ± sd over 3 seeds), the ablation table, the speed profile, deviations,
  requested contract changes, and an honest verdict: does the CDE earn its cost relative to
  `sig_ridge` and, at review time, relative to the transformer?
