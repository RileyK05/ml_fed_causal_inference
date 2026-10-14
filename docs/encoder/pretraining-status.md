# Pretraining status (2026-10-13)

Handoff for a fresh session. Full debate history: `pretraining-debate.md`. Test list: `cheap-test-menu.md`.
Cheap-test results: `encoder/results/q3_susceptibility/*cheap-*` (all 32 ran, exploratory).

## Unchanged

- Model: encoder reads firm history -> z -> heads a(z), b(z). Fed-day prediction mu = a + b*s,
  target = firm's raw Fed-day return. s in 10bp units.
- Training stages: **pretraining** (ordinary days) -> **fine-tuning** (Fed days) -> **evaluation** (held-out meetings).

## User's question

What pretraining makes a and b properly represent the firm?

## Decision (2026-10-14, agreed with the user)

Pretraining = the user's original plan: **predict the firm's raw return on the next ordinary
trading day from its history through today. No shock input.**
- Pretraining teaches firm state only (question.md: "Ordinary-day data teaches state
  representations, not the policy response itself"). a and b train on Fed days.
- Temporary head: expected return m and variance v. Loss: log v + (r - m)^2 / v.
- Afterwards: drop the head, keep the encoder + history scaler, train a/b on Fed days.
- Trial: this vs no pretraining vs masked reconstruction (`fedenc.workflow evaluate`).
- Built in `fedenc/return_corpus.py`, `pretrain_return.py`, `checkpoint.py`, `workflow.py`.
  Setup and commands: `next-return-pretraining.md`.
- v1 pretrains the history transformer only; fundamentals/context branches train on Fed days.

Rejected in the 2026-10-14 session: putting a shock (market or rate move) in pretraining.
It turns pretraining into response learning, and with the market as the shock it mostly
re-learns the beta63 input.

## First trial result (2026-10-14, vast RTX 3090, ~$0.44)

Dev period only: pretraining cutoff 2000-10-03, Fed evaluation through 2009-03-18, 60 test
meetings, 3 seeds, full fine-tuning (encoder lr = head lr / 10). Runs in
`encoder/results/q3_susceptibility/20261014-*` (exploratory).

| Arm | MSE per meeting | vs scratch (95% CI) | RankIC |
|---|---|---|---|
| No pretraining | 8.49 | | 0.086 |
| Next-day return | 8.89 | +0.40 (+0.06, +0.98) worse | 0.008 |
| Masked reconstruction | 8.82 | +0.33 (-0.03, +0.83) | 0.060 |

References on the same rows: predict zero 9.10; knowing each meeting's realized average
(unattainable) 6.11. Both pretrained arms were worse than scratch and less stable across
seeds; most of their extra error is in the meeting-wide level, not cross-firm differences.
Not yet tested: frozen encoder (the plan's first fine-tuning step).

## What was wrong with the first trial, and the fixes (2026-10-14, code only)

1. **Pretraining saw only 1994-1999.** One checkpoint (cutoff 2000-10-03) served every fold,
   including tests in 2007-2009. Fix: `pretrain-folds`, one checkpoint per fold (cutoffs
   2000-10-03, 2002-06-26, 2004-01-28, 2005-11-01), each on all ordinary days before it.
2. **Pretraining barely learned, and likely only volatility.** Validation loss 3.346 vs
   3.401 for "mean 0, trailing 63-day variance" on the same dates (no fitting needed for
   that check). Fix: objective `next_return_scaled` (mean only, error divided by the naive
   variance), plus a logged mean/variance split for every return objective.
3. **Post-training scrambled the encoder.** Random a/b heads updated the pretrained encoder
   from step one; fits stopped after 1-8 epochs. Fix: `adapt="staged"` (`tx_pre_staged`,
   `tx_masked_staged`): heads first with the encoder frozen, then everything at lr/10.
4. **No baseline.** Scratch was only compared with the pretrained arms. Fix:
   `evaluate --baselines pooled ridge_interact` on the same folds; the comparison reports
   every arm against scratch and pooled.

Next run (needs the go-ahead): `vast_train.sh`. Not run yet.

## Evidence from cheap tests (real data, linear proxies, development period only)

| Finding | Number | Implication |
|---|---|---|
| Ordinary-day rate beta is unstable | reliability 0.007 (21d), 0.05 (63d) (C3) | Don't pretrain on rate betas |
| Ordinary-day rate loading doesn't transfer to Fed days | transfer ratio -0.08, perm p 0.84 (R4, R10) | Don't pretrain on rate-day a + b*dy |
| Firm volatility predicts Fed-day residual size | t = 6.9; \|s\| t = -0.9 (R11) | Move size is a firm trait that carries over |
| Earnings-day jump size predicts Fed-day reaction size | rank IC 0.14, 0.06 after vol (R6) | Same: jumpiness carries over |
| MSE can't detect even a large planted b effect | b_hat corr 0.8-0.88, MSE gain not significant (R1) | Need a b-focused evaluation metric |
| Real firm-level b spread is small | ~16% of measured b variance is real (R2) | Expect weak signal |
| a pushes shared layers ~3x harder than b (untrained model) | B1 | lambda/detach question stays open for the trial |

## Rejected during the session (don't re-propose)

- Three pretraining targets at once (overbuilt).
- 3-month-ahead vol target (doesn't match "current day" task).
- Any beta adjustment in the pretraining target (mismatch with raw fine-tuning target).

## Parked

- Beta-adjusted fine-tuning target (market part + firm part split): user rejected the framing; not pursued.
- Earnings-jump target as an add-on.

## Notes for the next session

- The user designed the encoder. Answer the exact question asked, with concrete specs (data, input, target, loss).
- Make one recommendation tied to evidence and hold it. Don't swap plans on pushback without naming the flaw.
- No training without explicit go-ahead in chat.
- The first trial's code is committed (5c96bb4); the fixes above are committed after it.
