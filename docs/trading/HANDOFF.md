# Trading project (`trading/`, `fedtrade`): handoff and plan

Written 2026-10-21. **No training without the user's explicit go-ahead in chat** (AGENTS.md). v1
below fits nothing; v2 fits a model and needs the go-ahead.

## What carries over from Q3

The Q3 encoder work is frozen at git tag `fed_pred_encoder`. What it taught us
(`docs/encoder/q3-v6-drift.md`):
- **What the ranking predicts.** The summary MLP (F0s) ranks firms by sensitivity to a Fed
  surprise beyond beta. That ranking barely moves day-0 returns, but it predicts the next 20
  days: +374bp per 10bp surprise, placebo p 0.003, about half left after characteristic
  controls.
- **The ranking is a credit tilt.** Most sensitive: distressed, levered firms. Least sensitive:
  cash-rich tech.
- **The edge is the sensitive leg, in both directions** (section 9). It falls for a month after
  hawkish surprises and rises after dovish ones, about +1.6% per trade either way. The
  cash-rich leg adds nothing and carries the sector bets behind most losses.
- **Reverting to the model's fair value doesn't work** (sections 6-8). At firm level the gap
  between actual and predicted is firm news, not Fed mispricing.

## Structure

| Stage | What | Status |
|---|---|---|
| 1. Rate-sensitivity ranking | Q3's saved out-of-sample predictions, delta_b = (mu - mu_base)/s per firm (`signals.py`). Read as files from `encoder/results`; never re-selected on trading P&L (docs/question.md) | Frozen |
| 2. Trade | Which firms, which way, how long. Entered at the day-0 close (the surprise's sign is known); held 20 trading days; returns beta-hedged | v1 built, v2 proposed |

## v1: rule books (no fitting; `books.py`)

Each meeting with |s| >= 2.5bp; legs are deciles of delta_b; direction = -sign(s).
- **`decile`:** sensitive minus insensitive leg. The v6 drift trade.
- **`sensitive`:** sensitive leg minus the average firm. Found after the fact (section 9).
- **`sensitive_sector`:** sensitive leg minus each firm's own GICS sector average. No sector bet.

Development, F0s predictions, 42 big meetings, 20-day CAR, % per trade before costs
(`python -m fedtrade.backtest dev --no-save`):

| Book | Mean (95% CI) | Win | t | Dovish / hawkish | Worst | With 0.3% per leg cost |
|---|---|---|---|---|---|---|
| decile | +1.62 (+0.50, +2.81) | 57% | 2.7 | +2.15 / +1.13 | -5.5 | +1.02 |
| sensitive | +1.55 (+0.67, +2.49) | 69% | 3.3 | +1.54 / +1.57 | -5.2 | +1.25 |
| sensitive_sector | +1.14 (+0.43, +1.87) | 67% | 3.0 | +1.39 / +0.91 | -4.1 | +0.84 |

`sensitive` and `sensitive_sector` were chosen after seeing development results. Their
development numbers are not evidence.

## v2: a fitted drift model (proposal, needs the go-ahead)

Predict each firm's 20-day post-meeting CAR directly:

drift_i = s x g(delta_b_i, distance to default, leverage, cash, beta63, sector)

- **Inputs:** stage 1's ranking plus the traits that section 9 found carry or break the edge,
  e.g. high beta63 hurts the long leg, and energy names in the sensitive leg get swamped by oil.
- **First version:** ridge, walk-forward on Q3's folds, all meetings weighted by s².
- **Judged by:** the same per-meeting trade P&L as v1, against `sensitive`.
- **Small model only.** One data point per meeting (161 development meetings), so nothing
  deep.

## Evaluation and holdouts (decision needed)

- **Development:** Q3's walk-forward test meetings 2001-2021. Each delta_b is out of sample for
  Q3, but v1's books were designed on these meetings.
- **2022-2025:** untouched, and needed by both the Q3 final test and this project.
  - **Recommendation:** freeze v1 now (this doc and `books.py`). Run the Q3 final test, which
    saves F0s predictions for 2022-2025. Then score the three v1 books once, before anyone tunes
    anything.
  - The trading leads in `q3-v6-final-plan.md` move here.
- **After that, for v2:** no clean holdout remains except the 2026 yfinance holdout (8
  meetings) and live meetings. So v2 would be judged by paper-trading future meetings with
  its spec frozen in advance.
- **Reported alongside v1 on 2022-2025, not deciding:**
  - value-weighted versions;
  - the market-level fade from `q3-v6-drift.md` section 7: average actual minus average
    predicted day-0 move, fade it for days 1-5.

## Questions

- **Q5 (strategy):** this project's question.
- **Q4 (gap):** its literal estimand, actual minus predicted predicting future returns, is null
  on development data (`q3-v6-drift.md` sections 6-8). The continuation found here comes from
  the prediction itself, not the gap.

## Constraints

- Licensed data: per-firm tables (CRSP permnos, returns) are never committed. Saved runs hold
  per-meeting tables only.
- No training, no saved runs, without the user's go-ahead.
- 2022-2025 is scored once per frozen spec.
- Costs are reported, with liquidity in mind. The sensitive leg is small, distressed names,
  and shorting them after hawkish surprises may be expensive or impossible.
