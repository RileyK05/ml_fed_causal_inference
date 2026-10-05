# Agent guide: running the Q3 bots

Three briefs, one per agent. The shared contract goes first; then the encoder and CDE build
in parallel against it, and the results are reviewed together.

| Brief | What it builds | Where | Depends on |
|---|---|---|---|
| [docs/core/HANDOFF.md](core/HANDOFF.md) | Panel contract, synthetic data, heads, trainer, scoring harness, baselines | `core/fedcore/q3/` | Nothing |
| [docs/encoder/HANDOFF.md](encoder/HANDOFF.md) | Patch-transformer encoder + masked-patch pre-training | `encoder/` | Foundation merged |
| [docs/cde/HANDOFF.md](cde/HANDOFF.md) | Signature + ridge, Neural CDE encoder | `cde/` | Foundation merged |

**The foundation must be merged before the encoder and CDE agents start.** It defines the data
shapes, the encoder interface, the loss and the scoring, so both models are judged by the same
code.

## Scope

- **Synthetic panel for validation, real panel for the study.** Models are validated on a
  synthetic panel where the true sensitivity b is known, so "did it work" has an exact answer.
  The real WRDS panel is built (`fedcore.q3.real.load_real_panel`; the 2026 holdout is
  `load_holdout_panel`), but fitting any model on it is training and needs the user's go-ahead
  (see `AGENTS.md`).
- Models accept the real panel unchanged (that's what `Q3Panel` is for). Real data has no true
  b, so `b_corr` and `b_rmse` are undefined there (see `docs/training-plan.md`, D2).
- No claims about real markets come out of synthetic runs.

## Launching an agent

Give it, in its first message: `AGENTS.md`, this file, and its brief. Then: "Implement your
brief. Follow AGENTS.md. Stop and write your report when the exit criteria are met or you are
blocked."

One branch per brief so agents can't collide: `q3/foundation`, `q3/encoder`, `q3/cde` (use
git worktrees if they run at the same time).

## Rules for every agent

- Read `AGENTS.md` first. **Work only inside your own project folder** (the foundation agent
  only inside `core/fedcore/q3/`, `core/tests/` and the `core/pyproject.toml` extra). Never
  edit another project, `core/data/raw/`, or tests you didn't write. Request shared-contract
  changes in your report instead.
- Projects import `fedcore`, never each other (`core/tests/test_isolation.py`).
- All output goes through `ResultStore(<your project>/results).save(...)` with
  `role="exploratory"`. Never write results anywhere else.
- Splits come from `fedcore.protocol.walk_forward` (whole meetings, chronological).
- Never tune on test folds; use the trainer's validation slice. Declare any search grid in the
  report *before* running it.
- Fix seeds; report at least 3 seeds for any headline number.
- Windows venv: `.venv\Scripts\python`. If `pytest` fails on the temp directory, rerun with a
  short `--basetemp` path.
- Done means: `fedcore check` and `pytest` for `core` plus your project are green.

## The report every agent writes

Location given in its brief. Contents:

1. **What was built:** files, one line each
2. **How to reproduce:** exact commands
3. **Results table:** per-meeting MSE, b-recovery correlation and b RMSE vs the baselines,
   mean ± spread over seeds
4. **Deviations from the brief**, and why
5. **Requested contract changes**, if any
6. **Open questions and known weaknesses:** be blunt

## Review checklist (human + Claude)

- [ ] Interface respected: `HistoryEncoder.forward(x, mask) -> (B, out_dim)` and nothing else crosses the boundary
- [ ] No leakage: scalers fit on training meetings only; no test-fold tuning; masked/padded
      values provably ignored (the tests exist and pass)
- [ ] Synthetic recovery: does the model recover the planted b better than the baselines?
      How does it behave in the realistic-noise setting?
- [ ] Seeds: is the gap over baselines larger than the seed-to-seed spread?
- [ ] Runs saved to the project's own `results/` with honest roles and complete params
- [ ] Isolation test green; `fedcore check` and the project's tests green
- [ ] Report is candid about what didn't work
