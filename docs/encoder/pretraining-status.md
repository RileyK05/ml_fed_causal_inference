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
- Nothing from this session is committed yet.
