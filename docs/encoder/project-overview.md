# Q3 transformer: project overview and results to date

Written 2026-10-15. Covers the work through the inputs experiment that day. Everything here
is **exploratory** and uses the development period only (meetings through 2009-03-18). No
model has been selected, and nothing has touched the 2010-2025 outer test or the 2026 holdout.

Other docs: spec in [question.md](../question.md) (Q3 section), design menu in
[transformer.md](transformer.md), pretraining setup in
[next-return-pretraining.md](next-return-pretraining.md), running log in
[pretraining-status.md](pretraining-status.md), cheap tests in
[cheap-test-menu.md](cheap-test-menu.md), proposals not yet adopted in
[plan-b-metric-and-rerun.md](plan-b-metric-and-rerun.md) and
[next-experiment-proposals.md](next-experiment-proposals.md).

---

## 1. What the project is

**Question (Q3):** given what we know about a firm before a Fed announcement, and the size of
the policy surprise, how will that firm's stock react?

The model splits every prediction into two parts:

```text
predicted return = a(firm state) + b(firm state) x shock
```

| Part | Meaning | Analogy |
|---|---|---|
| **a** | What the firm does on the day regardless of the surprise | the intercept |
| **b** | How sensitive the firm is to the surprise | the slope on the shock |
| **shock** | The Fed surprise, measured in a 30-minute window around the statement, in 10bp units | the X variable |
| **firm state** | Learned from 252 days of price history, plus fundamentals and market context | what the firm "is" right now |

b is the point of Q3 ("susceptibility"): which firms react more to Fed news. The transformer's
job is to read a firm's recent history and produce a state from which a and b follow. The bet
behind the design is that a learned reading of history beats hand-made characteristics.

**Target:** the firm's raw return on the Fed day (previous close to the announcement-day close),
in percent. The spec fixes this target; changing it needs a spec revision.

---

## 2. Data

| Item | Value |
|---|---|
| Firms | S&P 500 members at each meeting, from CRSP (WRDS); about 500 per meeting, 1,252 distinct |
| Meetings | 253 scheduled FOMC statements, 1994-05 to 2025-12; 126,854 firm-meeting rows |
| Development period | First 120 meetings, through 2009-03-18 (about 60,000 rows) |
| Outer test | 2009-2025 meetings: reserved, never used for selection |
| Holdout | 2026 meetings (yfinance-spliced): separate, never pooled |
| Shock | 10 x the statement surprise (USMPD). Std 0.39 in the development period, i.e. about 4bp |

**Inputs per firm-meeting** (all known by the previous close):

| Group | Contents | Width into the model |
|---|---|---|
| Price history | 252 trading days x 5 channels: return, volume change, 21-day volatility, market-relative return, 63-day beta | 64 (transformer) |
| Fundamentals | Leverage, log size, profitability, cash, book-to-market, age of last report (point-in-time, ~99% observed), plus missing flags | 16 |
| Context | 1-year yield level and 21/63-day changes, last three Fed surprises, 21-day market volatility | 16 |

---

## 3. Model

```text
history (252 x 5) --> patch transformer --> 64 ┐
fundamentals (6 + flags) --> small MLP ---> 16 ├─> 96 --> fusion MLP --> z (64) --> a(z), b(z)
context (7) --> small MLP ----------------> 16 ┘
prediction = a(z) + b(z) x shock
```

- **Transformer:** 5-day patches (about 50 tokens), width 64, 3 layers, 4 heads, feed-forward 256,
  dropout 0.1, mean pooling. Small on purpose: the number of Fed meetings is small.
- The shock enters only at the end. It never feeds the encoder or the fusion network, so the
  firm's state is the same whatever the surprise turns out to be.

---

## 4. How models are judged

**Walk-forward by whole meetings.** Train on the first 60 meetings, test on the next 16, then
grow the training set by 16 and repeat. That gives 4 test windows and 60 test meetings:

| Window | Test meetings | Training meetings | Fold pretraining cutoff |
|---|---|---|---|
| 0 | 2001-11 to 2003-09 | 60 | 2000-10-03 |
| 1 | 2003-10 to 2005-09 | 76 | 2002-06-26 |
| 2 | 2005-11 to 2007-09 | 92 | 2004-01-28 |
| 3 | 2007-10 to 2009-03 | 108 | 2005-11-01 |

Inside each training set, the last 15% of meetings are held back for early stopping. Anything
pretrained must finish before that block starts (the "cutoff" column).

**Scores:**

| Score | What it measures | Notes |
|---|---|---|
| **MSE per meeting** (primary, per spec) | Average squared miss, each meeting weighted equally | In %²; lower is better |
| RankIC | Does the model order firms correctly within a meeting? | Higher is better |
| **b grade** (added 2026-10-14) | Within each meeting: regress firms' returns on (their predicted b minus the meeting average) x shock. 1 = b differences have the right size, 0 = noise, negative = backwards | Meeting-clustered SE. Biased if a and b are correlated across firms |
| Paired difference vs a reference | Mean per-meeting MSE gap with a meeting-bootstrap 95% range | A range that excludes 0 counts as a real difference |

**How to read MSE** (development test meetings):

| Reference | MSE |
|---|---|
| Predict 0% for every stock | 9.10 |
| Typical model in this project | about 8.5 |
| Knowing each meeting's actual average move in advance (impossible) | 6.11 |

About 6.1 of the 9.1 is firm-specific noise no model can reach. The reachable part (about 3.0)
is mostly the whole market's move that day, and the shock explains only about 9% of that.
A realistic floor is around 8.3-8.4. **Firm-to-firm differences in b × shock explain roughly
0.05% of the variance across firms within a meeting, so MSE is close to blind to b.** This is
why the b grade was added.

**One window dominates.** Scratch's MSE by window: about 8.0, 3.1, 3.2 and **23.5**
(2007-2009). Any MSE comparison is largely a contest over the financial crisis.

---

## 5. What we did, in order

### 5.1 Cheap tests (2026-10-13): 32 linear checks before any training

Quick regressions on real data to see where signal could come from. Full list in
[cheap-test-menu.md](cheap-test-menu.md); results under `encoder/results/q3_susceptibility/*cheap-*`.

| Finding | Number | Implication |
|---|---|---|
| Ordinary-day rate beta is unstable | Reliability 0.007 (21-day), 0.05 (63-day) (C3) | Don't pretrain on rate betas |
| Ordinary-day rate loading doesn't carry over to Fed days | Transfer ratio −0.08, permutation p 0.84 (R4, R10) | Ordinary-day "a + b × rate move" is not Fed-day b |
| Firm volatility predicts Fed-day move size | t = 6.9 (R11) | Size of moves is a firm trait that carries over |
| Earnings-day jump size predicts Fed-day reaction size | RankIC 0.14; 0.06 after volatility (R6) | Same, from company news |
| MSE can't detect even a large planted b effect | b correlation 0.8-0.88, MSE gain not significant (R1) | MSE alone can't judge b |
| Real cross-firm b spread is small | ~16% of measured b variance is real (R2) | Expect weak b signal |
| In ridge, flexible b hurts | MSE 8.91 with one common b vs 9.69 with state-dependent b (C1) | Noisy b estimates cost accuracy |

**R7, release days (2026-10-14).** Does a firm's sensitivity to rate moves on jobs-report and CPI
days (BLS release dates, sourced that day) predict its Fed-day sensitivity? No. Controlling for
the market, the link is about zero (transfer −0.02, t −0.13). Using the yield move only, it
leans the wrong way (transfer −0.36, t −1.05) and is not significant. Training b on release
days was therefore dropped.

### 5.2 Choosing a pretraining task

After a long debate ([pretraining-debate.md](pretraining-debate.md)), the agreed plan was the
user's original one: on ordinary days, predict the firm's next-day return from its history,
**with no shock**, then train a and b on Fed days. Putting a shock (market or rate move) into
pretraining was rejected: it turns pretraining into response learning, and with the market as
the shock it mostly re-learns beta.

### 5.3 First trial (2026-10-14, RTX 3090 on vast.ai, about $0.44)

Three models, same folds, 3 seeds:

| Model | What it is | MSE | vs scratch (95% range) | RankIC | b grade |
|---|---|---|---|---|---|
| Scratch | Random transformer, trained on Fed days only | 8.49 | – | 0.086 | +2.3 (t 1.5) |
| Next-day pretrained | Pretrained to predict tomorrow's return (mean and variance), then fine-tuned | 8.89 | **+0.40** (+0.06, +0.98), worse | 0.008 | −4.1 (t −1.2) |
| Masked pretrained | Pretrained to fill in hidden patches of history, then fine-tuned | 8.82 | +0.33 (−0.03, +0.83) | 0.060 | +1.2 (t 0.6) |

Pretraining made things worse. The pretrained model's b was almost flat across firms (spread
0.04 vs 0.13 for scratch), which matches the user's concern: pretraining fed a and starved b.

### 5.4 What was wrong with the first trial

| Flaw | Evidence | Fix (coded, commit b634f2d) |
|---|---|---|
| Pretraining only saw 1994-1999 | One checkpoint (cutoff 2000-10-03) served all four windows, up to 2009 | One checkpoint per window (`pretrain-folds`) |
| Pretraining learned volatility, not returns | Validation loss 3.346 vs 3.401 for "mean 0, trailing volatility": that was its whole edge | Mean-only objective `next_return_scaled`, plus a logged mean/variance split |
| Fine-tuning scrambled the encoder | Random a/b heads updated the pretrained encoder from step one; fits stopped after 1-8 epochs | Two-stage fine-tuning (`adapt="staged"`): heads first with the encoder frozen |
| No baseline | Scratch was only compared with the pretrained models | `--baselines pooled ridge_interact` on the same folds |

These fixes are coded and tested but **were never run**. The ordinary-day experiment below
replaced them.

### 5.5 Ordinary-day pretraining with the same formula (2026-10-15, about $1)

The user's simplification: pretrain on the same task the model is used for. Same model, same
formula, same target every day. On ordinary days there is no Fed shock, so shock = 0 and the
prediction is just a. On Fed days b turns on. Nothing is thrown away between stages.

- Ordinary days: every 10th trading day, 1994-05 to 2009-03 (362 days, 181k firm-days), built
  with the Fed panel's own builder (same firms, inputs and point-in-time rules). All statement
  dates excluded. Each window pretrains only on days before its cutoff (157, 199, 238, 281 days).
- Fine-tuning on Fed days keeps every weight and the pretraining scaler. The b head, which never
  trained, gets the full learning rate; everything else gets a tenth of it.
- Code: `encoder/fedenc/daily.py` (`python -m fedenc.daily run`).

| Model | MSE | vs scratch (95% range) | RankIC | b grade | b spread across firms |
|---|---|---|---|---|---|
| Daily-pretrained | 8.54 | +0.03 (−0.04, +0.12) | 0.087 | 1.13 (t 0.9) | 0.133 |
| Scratch | 8.51 | – | 0.079 | 1.75 (t 1.2) | 0.117 |
| One b for everyone (pooled) | 8.59 | +0.08 (−0.02, +0.19) | – | – | 0 |

**Result: a tie.** Pretraining with the same formula no longer damages b (spread 0.133, not
0.04, right sign), so the user's diagnosis of the first trial holds. But it adds nothing over
scratch. And no transformer clearly beats one b for everyone.

The fine-tuning epochs were telling. In window 0 (about 50 training meetings) fine-tuning stopped
after 1 epoch every time; in windows 1-3 (65-90 meetings) it kept improving for 16-39 epochs.
The model finds more to learn about b as Fed meetings accumulate.

### 5.6 Does the model use fundamentals? (2026-10-15)

**Shuffle check, no training.** For each saved scratch model from the first trial, fundamentals
were shuffled across firms within each meeting, and the change in predicted a and b measured
(share of their cross-firm variance):

| Shuffled | Change in a | Change in b |
|---|---|---|
| Fundamentals | 4% | 3% |
| Price history | 194% | 200% |

The trained model essentially ignores fundamentals; it runs on price history.

**Inputs experiment** (GPU, about $0.25): is that a width problem or a signal problem?

| Model | MSE | vs scratch (95% range) | RankIC | b grade |
|---|---|---|---|---|
| Scratch (branches 16 wide) | 8.51 | – | 0.079 | 1.75 (t 1.2) |
| Branches 32 wide | 9.63 | +1.13 (+0.02, +3.14) | 0.072 | 0.39 (t 0.2) |
| No price history (fundamentals + context only) | 8.78 | +0.27 (+0.04, +0.64) | 0.023 | 0.30 (t 0.4) |
| Pooled | 8.59 | +0.08 (−0.02, +0.19) | – | – |

- **Width:** the 32-wide loss is almost all one seed that blew up in window 3 (MSE 38.7 vs 23-24
  for the other two). Without it, 32-wide is about equal to scratch. More width adds no accuracy
  and less stability.
- **Signal:** without price history the model is clearly worse and ranks firms almost at random.
  Fundamentals were not squeezed out. Price history already carries what is useful in them
  (volatility and beta reflect leverage, size and industry).

---

## 6. What we know now

1. **The limit is the Fed data, not the architecture.** Every variant lands near 8.5, and none
   beats one-b-for-everyone with confidence. b explains a tiny share of returns, shocks average
   about 4bp, and there are only 60 test meetings.
2. **MSE cannot judge b.** It is dominated by the market-wide move and by the 2008 window.
   b needs its own score; the b grade exists but is noisy (t around 1 for every model).
3. **Pretraining:** a throwaway-head objective (next-day mean and variance) fed a and flattened b.
   Pretraining with the real formula on ordinary days (shock = 0) is harmless but adds nothing yet.
4. **Inputs:** price history carries the signal. Fundamentals add little, and width doesn't change that.
5. **Carry-over traits are about size, not direction:** volatility and earnings-day jumpiness
   predict how big a firm's Fed-day move is, not which way.
6. **Instability shows up in the crisis window.** Most per-seed blow-ups happen in 2007-2009.

---

## 7. Open decisions (the user's)

| Decision | Options | Why it matters |
|---|---|---|
| **How a winner is chosen** | Keep MSE as the only rule; or require "beats scratch on the b grade without being meaningfully worse on MSE"; or another rule, written into `question.md` before the next run | MSE can't reward b, so under the current rule a better b can never win |
| **Development period length** | Keep 120 meetings, or move the cutoff later (more meetings to learn from, smaller final test) | Window-by-window results suggest b improves with more meetings |
| **Next experiment** | Earnings-reaction input (proposal 2, ridge first on CPU); b-grade metric in the shared evaluator (plan-b Task 1); or none until the winner rule is set | Proposal 1 (shrinking b) conflicts with b grades above 1, which say the predicted differences are too small, not too big |
| **Cleanup** | Delete the old pretraining machinery (Gaussian, scaled and masked objectives, per-fold checkpoints, staged fine-tuning) and keep `daily.py`; or keep everything | The ordinary-day version matches or beats it on every score with far less code |

---

## 8. Ideas rejected along the way

| Idea | Why |
|---|---|
| Shock (market or rate move) in pretraining | Becomes response learning; with the market it re-learns beta |
| Training b on BLS release days | R7: release-day rate sensitivity doesn't predict Fed-day sensitivity |
| Full-day 1-year yield change as the shock | Correlation with the Fed-day market move is 0.00 |
| Three pretraining targets at once; 3-month volatility target; beta-adjusted targets | Overbuilt, or mismatched with the raw-return fine-tuning target |
| Shorter (intraday) return window to cut noise | Changes the question; the user rejected it |
| Portfolio-style b scoring | Creative scoring rather than fixing the model; set aside |
| 32-wide fundamentals branch | Tested: no gain, less stable |

---

## 9. Code, runs and costs

**Code** (`encoder/fedenc/`):

| File | Role |
|---|---|
| `encoder_transformer.py` | Patch transformer |
| `arms.py` | `TxArm`: scratch / pretrained, adapt modes, `no_history`, `branch_dim` |
| `daily.py` | Ordinary-day experiment, inputs experiment, b grade, chunked evaluation |
| `pretrain_return.py`, `return_corpus.py`, `checkpoint.py`, `train.py`, `workflow.py` | First-trial pretraining machinery and its fixes (candidates for deletion) |
| `core/fedcore/q3/` | Shared panel, trainer, evaluator and baselines |

**Commits:** 5c96bb4 (next-day pretraining), ead731e, b634f2d (fixes), 599f8ef (ordinary-day
experiment), 5013427 (GPU memory fix), 1bf92a5 (inputs experiment). **Not committed:** the R7
and BLS release-date changes (`core/data/...`, `core/fedcore/q3/cheap/...`) and the vast scripts
(`vast_*.sh`, which hardcode hosts).

**Runs** (`encoder/results/q3_susceptibility/`, gitignored, all exploratory):

| Experiment | Run folders |
|---|---|
| First trial | `20261014-151845` and `-152742` (pretraining), `-153152` scratch, `-153526` next-day, `-154006` masked, `-154009` comparison |
| Ordinary-day | `20261015-030832` daily, `-031256` scratch, `-031258` pooled, `-031300` comparison |
| Inputs | `20261015-033807` scratch, `-034246` 32-wide, `-034817` no history, `-034818` pooled, `-034822` comparison |
| R7 | `20261014-122402_cheap-r7` |

**Rerunning on a GPU:** `bash vast_upload.sh HOST PORT`, then
`bash vast_run.sh HOST PORT NAME <args for python -m fedenc.daily>` (for example
`inputs inputs --seeds 0 1 2 --device cuda`). Download the result folders with `scp`, then
destroy the instance.

**GPU spend to date:** about $1.70 on vast.ai (RTX 3090 at $0.20-0.30/hour), including one
host that stalled while starting.
