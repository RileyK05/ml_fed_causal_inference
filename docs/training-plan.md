# Q3 training plan: decisions before we spend compute

Everything up to training is built. Changing anything above the line below is cheap; changing it
after runs exist is not, because **every run is part of the record** (append-only results, a
multiple-testing count, and no tuning on test folds). So the real cost of training isn't GPU
dollars, which are small. It's that the choices get locked in.

Nothing in this doc has been run. Speed figures marked *measured* come from a CPU smoke test;
GPU figures are estimates until the Colab speed cell runs.

## 1. What exists now

| Piece | Where | Status |
|---|---|---|
| Real panel: 126,854 firm-meetings, 253 scheduled statements (1994-05 to 2025-12), ~500 firms each | `fedcore.q3.real.load_real_panel` | built, tested, cached |
| 2026 holdout: 6 meetings x 486 firms, CRSP history + yfinance (overlap corr 0.99998) | `load_holdout_panel` | built, separate, never pooled |
| Pre-training corpus: real 252-day windows with end dates and a cutoff | `pretrain_corpus(cutoff)` | built, tested |
| Transformer (166,770 params) and log-signature CDE (159,186) on the real panel's 6 fundamentals / 7 context | `encoder/`, `cde/` | build-tested, never fit on real data |
| GPU: arms default to CUDA when present | `resolve_device` | CPU-tested |
| Colab notebook: setup + a 1-minute speed cell (off by default) | `core/colab/q3_gpu_check.ipynb` | smoke-tested on CPU |

## 2. Decisions

Each has a recommendation. The defaults go into code only once you agree.

### D1. Folds (how we score on real data)

Walk-forward by whole meetings, expanding window. 253 meetings:

| min_train | test block | first test meeting | folds |
|---|---|---|---|
| 100 | 24 | 2006-10-25 | 7 |
| **120** | **16** | **2009-04-29** | **9** |
| 140 | 16 | 2011-11-02 | 8 |

**Recommend 120 / 16.** The test period then covers 2009 to 2025: ZIRP, the hiking cycles and 2020.
The target is the meeting-day return and features end the day before, so no embargo is needed.

### D2. What counts as "better" (the real data has no true b)

On fake data we scored how well the model recovered the planted b. Real data has no planted b, so:

- **Primary:** out-of-sample `mse_per_meeting` minus `ridge_interact`'s, with a paired
  meeting-bootstrap CI (already in the harness). This is the rule in `question.md`: select on
  response prediction.
- **Key secondary, the real-data stand-in for b recovery:** within each test meeting, sort firms
  by predicted b into quintiles. Then fit realized return on s inside each quintile across meetings.
  If b(z) means anything, the realized slopes should rise from Q1 to Q5. *(Needs a small function;
  not built yet.)*
- Also report: `centered_mse` (within-meeting differentiation only) and `rank_ic`.

**Expect small numbers.** A typical surprise is |s| of about 0.3-0.5 (3-5bp), so b·s is about 0.4pp,
against about 2.4pp of daily return noise. Most of the MSE is noise, so the gaps between models
will be tiny and the CIs carry the conclusion.

### D3. Pre-training

| Option | Data | Validity | Cost |
|---|---|---|---|
| **A. One checkpoint, cutoff = first test meeting (2009-04-29)** | ~620k windows, S&P ever-members | valid for every fold | 1 pre-training run |
| B. One checkpoint per fold | grows to 1.27M windows | uses more data in later folds | 9 runs |
| C. All of CRSP (~7,000 firms) instead of S&P members | ~5x the windows | as A or B | new WRDS pull (~1 GB, ~30 min) + longer runs |

**Recommend A.** It's the cleanest version that is still honest, and later folds can be refreshed
if A shows promise. On fake data pre-training mostly learned volatility and beta state, not
return direction (encoder bot, M3). Real data may differ.

Two pieces of code are still needed for A: pre-training has to read from the corpus without
materializing it (620k windows would be ~3 GB as arrays), and the arms must load a checkpoint
from disk instead of from the synthetic `get_pretrained`.

### D4. Mid-training on other macro events (CPI, jobs releases)

**Recommend skip for v1.** We only have CPI release dates from 2023, and it adds a stage to defend.
Revisit if pre-training helps.

### D5. Which arms, how many seeds

| Arm | What | Keep? |
|---|---|---|
| pooled, ridge_interact, lgbm, summary_nn | baselines | **yes**: ridge_interact is the bar |
| tx_scratch | transformer, no pre-training | **yes** |
| tx_pre_frozen | pre-trained encoder frozen, heads trained | **yes** (cheapest pre-trained arm) |
| tx_pre_top | top block unfrozen | optional: cut first if budget is tight |
| tx_pre_full | everything fine-tuned, smaller encoder lr | **yes** |
| sig_ridge | signature + ridge ("linear CDE") | **yes** (cheap) |
| cde_logsig | Neural CDE (log-ODE) | **yes** |

**Recommend 3 seeds.** Fixed before running: lr 1e-3, max 40 epochs, patience 10 (the encoder
bot's budget), and model size as in `transformer.md`.

### D6. Budget (estimates)

Laptop CPU, *measured*: 0.76 s per 500-row transformer step, 0.46 s for the CDE. A fine-tuning fit
averages ~150 training meetings, so ~38 steps of 2,000 rows per epoch, about 2 min/epoch on CPU.
At ~25 epochs that's **~50 min per fit on the laptop**.

| | Fits | Laptop CPU | One T4 / 4090 (est. 15-30x) |
|---|---|---|---|
| Pre-training, option A (~620k windows x ~10 epochs) | 1 | ~2-3 h | ~5-15 min |
| Neural arms: 5 arms x 9 folds x 3 seeds | 135 | ~110 h (no) | ~4-8 GPU-h; ~2-3 h wall with 3 jobs at once |
| Baselines (ridge, lgbm, pooled); summary_nn | 36; 27 | minutes; ~20 h | minutes; ~1 h |

On Vast a 4090 is about $0.40-0.80/h, so the whole plan is roughly **$5-20**. Colab free can handle
pre-training and spot checks, but session limits make the 135-fit grid painful there.

### D7. Pre-register the primary spec

`AGENTS.md`: one `primary` run per question; everything else is `robustness` or `exploratory` and
adds to the multiple-testing count. **Before the first real fit**, write down the primary spec
(D1 folds, D2 metric, D3 option, D5 arms and seeds) in `docs/question.md`, then don't change it.

## 3. Suggested order

1. Agree D1-D7 (this doc).
2. Build the three missing pieces: the b-quintile metric, streaming pre-training from the corpus,
   and checkpoint loading in the arms. Code + tests only.
3. Colab: run the speed cell (~1 min) and replace the estimates above with measured numbers.
4. Baselines on the real panel. **First real training; needs your go-ahead.**
5. Pre-train one checkpoint (option A).
6. Neural arms on Vast.
7. Score the 2026 holdout once, at the very end.

## 4. Known limits to say out loud

- The target is the full meeting-day return, not the 30-minute window (no firm-level intraday
  data; TAQ is a later option). The information effect and the rest of the day's news are inside it.
- Universe: S&P 500 members only. Fundamentals: 6 ratios, report-date point-in-time. Sector is
  metadata only (pre-1999 backfilled, labeled).
- The 2026 holdout drops firms removed from the index during 2026 (no dated change list) and
  2026 additions: a small, labeled survivorship bias.
- On fake data, hand-built summary features beat the transformer from scratch. That says the fake
  data was too easy for summary stats, not that the transformer is wrong, but it sets the bar
  the real data has to clear.
