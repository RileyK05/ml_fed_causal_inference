# Q3 encoder report: what works, what didn't, what next

Written 2026-10-16. Covers all Q3 transformer work from 2026-10-13 to the one-shot final test on
2026-10-16. Background, data and model details through 2026-10-15 are in
[project-overview.md](../encoder/project-overview.md); this report summarizes those and adds
everything since.

---

## Summary

**Question (Q3):** given what we know about a firm before an FOMC statement and the size of the
policy surprise, how does its stock react? The model predicts
`return = a(firm) + b(firm) x shock`, where **b**, the firm's sensitivity to the surprise, is the
point of Q3.

**Answer:**

1. **b beyond beta is not detectable.** On Fed days, firms' different reactions to the surprise
   are explained by their beta times the market's reaction. Every angle tried, eleven descriptive
   tests, four model variants and a 3- and 5-day return check, found no stable firm-specific
   surprise sensitivity beyond that (sections 3 and 4). Longer windows don't reveal a delayed
   reaction: the one day-0 signal (sector) fades within days (section 3.4).
2. **a is mostly beta x the Fed-day drift.** Which firms do better on Fed days is predictable,
   but almost entirely because the market drifts up on FOMC days (about +0.37% x beta in our
   fit; Lucca and Moench 2015) and high-beta firms drift more. The transformer's firm
   differences correlate 0.62-0.70 with pre-meeting beta.
3. **The transformer does not beat a three-number CAPM baseline** (`c + beta x (drift + slope x
   shock)`, added 2026-10-16 after an outside critique). The locked two-part model beat
   one-b-for-everyone (pooled) in every development comparison, but against the CAPM baseline
   its edge is -0.020 (-0.050, +0.012) in development and -0.002 in the final test. Its win over
   pooled came from beta, which pooled cannot see.
4. **Most of the error is out of reach.** Each meeting's market-wide move, especially on
   big-surprise days and in 2008, is the largest part of the error, and the surprise explains
   only about 17% of it.

**Final test** (role `primary`; trained on every meeting through 2021, scored once on the 32
meetings from 2022-01-26 to 2025-12-10):

| Model | MSE | vs pooled (95% range) | vs CAPM baseline | RankIC | a grade | b grade |
|---|---|---|---|---|---|---|
| **Two-part (locked)** | **5.723** | -0.045 (-0.110, +0.018) | **-0.002 (-0.085, +0.082)** | 0.109 | +1.28 (t 2.5) | -0.09 (t -0.1) |
| CAPM baseline (computed afterwards) | 5.725 | -0.043 (-0.121, +0.040) | – | – | – | – |
| Pooled (one b for everyone) | 5.768 | – | – | – | – | – |

The CAPM baseline was not part of the locked comparison; it was fitted afterwards on the same
training meetings (3 parameters, nothing tuned on the test meetings). The final test was also
underpowered: the range implies a per-meeting standard error of about 0.033, so a true -0.045
edge had only about a 1-in-4 chance of reaching significance on 32 meetings.

---

## 1. Setup in brief

| Item | Value |
|---|---|
| Firms | S&P 500 members at each meeting (CRSP via WRDS), about 500 per meeting |
| Meetings | 253 scheduled FOMC statements, 1994-05 to 2025-12 |
| Target | Firm's raw return on the Fed day, in percent (fixed by the spec) |
| Shock | 10 x the USMPD statement surprise (STMT), in 10bp units |
| Inputs | 252 days x 5 price channels (return, volume change, 21-day vol, market-relative return, 63-day beta); 6 fundamentals; 7 meeting-context variables |
| Model | Patch transformer (width 64, 3 layers) + fundamentals and context branches, then heads for a and b |
| Evaluation | Walk-forward by whole meetings; MSE per meeting (primary score), paired meeting-bootstrap 95% ranges, RankIC, b grade (within-meeting slope of returns on predicted b differences x shock, meeting-clustered) |

**Periods** (decided 2026-10-15; see the memory note and section 6):

| Period | Meetings | Use |
|---|---|---|
| Original development | 1994-05 to 2009-03 (120) | All work through 2026-10-15 |
| Extended development | 1994-05 to 2021-12 (221) | Later comparisons |
| Final test | 2022-01 to 2025-12 (32) | Used once, 2026-10-16 |
| 2026 holdout (yfinance) | 2026 meetings | Untouched |

**MSE split.** Every meeting's MSE splits exactly into a **meeting-wide part** (miss on the
average move) and a **firm-to-firm part** (miss on firms' differences around it). a and b
compete only in the firm-to-firm part. This split drives most of the conclusions below.

---

## 2. What works

### 2.1 The two-part model

**What it is.** Pooled OLS (`a + b x shock`, one a and b for all firms) predicts each meeting's
average move. A transformer with a **centered a** (a cannot move the meeting average) trained
with a **cross-firm loss** (the meeting-wide part weighted at 0.05) predicts each firm's
difference around that average. Code: `TwoPartArm` in `encoder/fedenc/decouple.py`.

**Why it works.** Run-to-run noise in the transformer is almost all in its guess of the
meeting-wide move during 2008 (that part's spread across runs and seeds: 0.153; the
firm-to-firm part: 0.035). The two-part model stops the network from guessing that part, so it
keeps the network's useful, stable part and drops the noisy one.

**Every test it got:**

| Test | Window | Two-part | Pooled | Difference (95% range) |
|---|---|---|---|---|
| Built from saved predictions (picked after seeing results) | dev to 2009, seeds 0-2 | 8.532 | 8.587 | -0.055 (-0.099, -0.015) |
| Average of all six transformer arms (no picking) | dev to 2009 | 8.551 | 8.587 | -0.036 (-0.072, -0.003) |
| Fresh seeds 3-5, built as a model | dev to 2009 | 8.532 | 8.587 | **-0.055 (-0.102, -0.010)** |
| Longer window, 161 test meetings | dev to 2021 | 5.927 | 5.954 | **-0.027 (-0.057, -0.004)** |
| Same, press-conference surprise (ME) | dev to 2021 | 5.934 | 5.954 | -0.020 (-0.050, +0.003) |
| **Final test, used once** | 2022-2025 | 5.723 | 5.768 | -0.045 (-0.110, +0.018) |

The edge is small (0.5-1% of MSE) but it points the same way in every test. Its significance
in development doesn't carry to the 32-meeting final test.

**Against the CAPM baseline** (`c + beta x (drift + slope x shock)`, beta = last 63-day beta
before the meeting, OLS on each fold's training meetings):

| Window | Two-part | CAPM baseline | Two-part vs CAPM (95% range) | CAPM vs pooled |
|---|---|---|---|---|
| Dev to 2021 (161 meetings) | 5.927 | 5.947 | -0.020 (-0.050, +0.012) | -0.007 (-0.043, +0.025) |
| Final test (32 meetings) | 5.723 | 5.725 | -0.002 (-0.085, +0.082) | -0.043 (-0.121, +0.040) |

So what the two-part model adds over pooled is mostly beta. In development it may add a little
beyond beta (point estimate -0.020), but not significantly, and not in the final test.

### 2.2 a: mostly beta x the Fed-day drift

Almost all of the two-part model's gain comes from a, not b:

| Firm-to-firm error | Dev to 2009 (fresh seeds) | Dev to 2021 |
|---|---|---|
| No firm differences | 6.110 | 4.147 |
| a's differences only | 6.061 | 4.122 |
| b x shock's differences only | 6.105 | 4.146 |

The a grade is significant in development (t 3.4 over 161 meetings) and in the final test
(t 2.5), and the gain also appears on tiny-surprise meetings, where b x shock is close to zero.
But a is not firm-specific alpha. The market rises on FOMC days on average (the announcement
drift; the CAPM fit puts beta x drift at about +0.37%), high-beta firms rise more, and pooled
gives every firm the same drift. The transformer's centered a correlates 0.63-0.70 with
pre-meeting beta, and on the market-adjusted target (beta x market removed) its edge
disappears (-0.001). a is mostly beta x the Fed-day drift.

### 2.3 Things that are solid knowledge, not models

| Finding | Evidence |
|---|---|
| Beta is the main firm-level channel | High-beta firms amplify the day's market move: per-meeting slope on beta vs market move t +7; b grade loading on beta x market t 9-14 in every run |
| Big surprises are the main error point | Surprises of 5bp or more: 15% of meetings, 25% of error; error per meeting about double (14.4 vs 7.6), also outside the crisis (9.1 vs 4.2) |
| The meeting-wide move is close to its floor | Predicting zero: 2.99. Shock only (pooled): 2.48. Adding context hurt (2.85). Robust fit: 2.40, not significant. The shock explains about 17% of it |
| The model leans on the history channels | Shuffling fundamentals moves a and b by 3-4%; shuffling history moves them about 200%; without history the model is clearly worse (+0.27). Two of the five history channels are pre-computed 21-day volatility and 63-day beta, so this shows those features dominate, not that the transformer learned sequential patterns |
| Volatility and earnings-day jumpiness predict move size | Cheap tests R11 (t 6.9) and R6 (RankIC 0.14) |

---

## 3. What didn't work

### 3.1 Model and training changes

| Attempt | Result | Why it failed |
|---|---|---|
| Next-day-return pretraining, then fine-tuning | +0.40 worse than scratch; b nearly flat | Throwaway heads fed a and starved b; pretraining mostly learned volatility |
| Masked-patch pretraining | +0.33 worse | Same structure problem |
| Ordinary-day pretraining with the real formula (shock = 0) | Tie (+0.03) | Harmless, adds nothing |
| Plain transformer (scratch) | 8.51-8.76 across identical reruns; never beat pooled reliably | Its meeting-wide guess is noisy, worst in 2008 |
| Two-part transformer vs a CAPM baseline | -0.020 (dev, not significant), -0.002 (final) | What it learned is mostly beta |
| Centered a alone | 8.88 | Gives up the meeting-wide move without a replacement |
| Wider branches (32) | 9.63; one seed blew up | More capacity, less stability |
| Ridge with interactions | 9.88 | Flexible b estimates are noise |
| Averaging 3 seeds' predictions | Saves only 0.01-0.03 | Doesn't fix the crisis-window guesses |
| Better models of the meeting-wide move (context, volatility interaction) | Worse than pooled | About 100 meetings, few crises |

### 3.2 More data and more inputs

| Attempt | Result |
|---|---|
| **Top 1,500 firms** instead of S&P 500 | Slightly worse on S&P rows (two-part 8.188 vs 8.130 trained on S&P only). Small caps add noise, not signal |
| **Sector** as an input (11 GICS flags) | Worse: two-part 5.940 vs 5.927 without it (dev to 2021). Zero effect on the market-adjusted target |
| **Market-adjusted target** (return - pre-meeting beta x day's market) | Two-part ties pooled exactly (-0.001, range ±0.005); every b grade below |t| 1 |
| **Press-conference surprise (ME)** instead of the statement window | Slightly weaker edge (-0.020); b unchanged |
| **Longer development window** (to 2021, 161 test meetings) | Two-part edge holds (-0.027); b still undetectable |

### 3.3 Searching for b directly (descriptive, no models)

Each test asks: does this group of firms beat or lag the market in line with the surprise,
beyond what its beta explains? Rows are firm groups; numbers are excess return per 10bp of
hawkish surprise.

| Angle | Data | Result |
|---|---|---|
| Big surprises: do firms spread out more? | S&P, dev to 2009 | Spread rises on big days (sd 2.35 vs 1.8), but it is all pre-meeting volatility: s^2 term t -0.1 once vol is controlled |
| Regime (calm vs turbulent markets; easing vs tightening) | S&P, dev to 2009 | Same average reaction (-0.88 vs -1.00 per 10bp); interaction t -0.17. Turbulence adds noise, not a different response |
| Size terciles | Top 1,500, dev to 2009 | Identical: -0.90, -0.82, -0.87 |
| **Sector** | Top 1,500, dev to 2009 | **The one real signal:** consumer discretionary -0.32 (t -3.4), materials -0.47 (t -3.0), health care +0.29 (t +2.3); joint p 0.006; explains about 0.15% of firm-to-firm variance. **Hurt as a model input** (3.2); fades within days (3.4) |
| Credit rating (S&P, comp.adsprate) | Top 1,500, dev to 2009 | Groups differ (p 0.047): BBB -0.18 (t -2.3), unrated +0.17 (t +2.2), junk -0.02. Within sector, nothing (all \|t\| < 1.8). Junk shows up as higher beta (1.10 vs 0.89) |
| Interest coverage; free cash flow | Top 1,500, dev to 2009 | Nothing (p 0.22, 0.63) |
| Funding gap (capex + debt repaid - operating cash) and collateral (PP&E / assets) | Top 1,500, dev to 2009 | Nothing within sector. "Needs money, no collateral" firms lean the wrong way (+0.15, t 1.9) and have the highest beta (1.24) |
| Debt maturity (short-term share) | Top 1,500, dev to 2009 | Nothing; flat zero within sector |
| Growth vs value | Top 1,500, dev to 2009 | Growth slightly *better* on hawkish surprises (+0.14, t 1.4); nothing within sector. "Hawkish hurts growth" is not visible before 2009 |
| GSS target vs path factors | S&P, dev to 2021 | Market falls 0.21% per 10bp of path news (t -2.5). Sector responses are mixed and run against the duration story (info tech +0.14, t +2.4); growth vs value nothing in either 1994-2008 or 2009-2021 |
| Release days (jobs, CPI) as a b teacher (R7) | S&P | Release-day rate sensitivity doesn't predict Fed-day sensitivity (t -0.13) |

**Reading.** Every channel through which firms' reactions differ turns out to run through beta
(and, before 2009, sector). That fits the user's view that traders react within hours to a few
widely watched factors, not to balance-sheet detail. One caveat: everything here is the
**same-day** return. Statements come out at 2:00 or 2:15 pm, so about two hours of trading
remain, dominated by index futures and ETFs that move stocks in proportion to beta. Credit and
funding effects could take days to show, and this design can't see them. Section 3.4 tests
that: they don't show up over 3 or 5 days either.

### 3.4 Multi-day pre-check (2026-10-16)

Same group tests, top 1,500, all 221 development meetings (1994-2021), on the market-adjusted
cumulative return `R(window) - beta x Rmkt(window)` (beta = last 63-day beta before the meeting).
Windows in trading days, 0 = Fed day. Non-sector traits are also run within sector. Cells are
p-values for "the groups react differently to the surprise" (meeting-clustered):

| Trait | Day 0 | Days 0-2 | Days 0-4 | Days 1-4 (after-day only) |
|---|---|---|---|---|
| Sector | **<0.001** | **<0.001** | 0.36 | 0.34 |
| Credit rating (within sector) | 0.002 (0.13) | 0.42 (0.19) | 0.56 (0.60) | 0.08 (0.07) |
| Free cash flow (within sector) | 0.54 (0.24) | 0.15 (0.29) | 0.11 (0.44) | 0.65 (0.31) |
| Funding gap (within sector) | 0.49 (0.78) | 0.78 (0.55) | 0.90 (0.42) | 0.96 (0.17) |
| Debt maturity (within sector) | 0.73 (0.88) | 0.58 (0.85) | 0.13 (0.36) | 0.02 (0.21) |
| Growth vs value (within sector) | 0.21 (0.11) | 0.18 (0.14) | 0.15 (0.18) | 0.03 (0.07) |

**Reading.** Nothing gets stronger with a longer window. The sector signal is a Fed-day effect
that fades by day 4 and partly reverses: consumer discretionary goes from t -3.5 on day 0 to
+0.5 over days 1-4, and info tech from +1.6 to -2.9. The few after-day p-values below 0.05 (debt
maturity, growth vs value) come from the firms with missing data, not the trait groups, and
don't survive within sector. Across the 132 tests in all three periods (1994-2021, 1994-2008,
2009-2021), 16 have p < 0.05, against 7 expected by chance. The excess is the known day-0 sector
and rating results; the rest is chance level. Caveat: noise grows with the window (five days of
news), so these tests have less power than the day-0 ones. But the point estimates don't grow
either. Script: `scratchpad/multiday.py` (not committed).

---

## 4. Why b is out of reach

1. **The signal is tiny relative to noise.** Firm-to-firm differences in b x shock explain about
   0.05% of within-meeting return variance; the typical surprise is about 3-4bp.
2. **It runs through beta.** Firms move with the market in proportion to beta, and the market's
   reaction to the surprise is itself unreliable. A dovish surprise in 2002-08 came with a 2.3%
   market fall, and a hawkish one in 2008-09 with a 1.5% rise (the Fed-information effect).
3. **Big surprises are rare and clustered in turbulent periods.** There are 25 big-surprise
   meetings (5bp or more) in 32 years, 19 of them in the development period. Of the 17 before
   2009, 13 came in volatile markets.
4. **What the market cares about changes.** The 1994-2009 reflexes (growth holding up on
   hawkish news) are not today's consensus. A model trained on one era learns that era's
   reflexes.
5. **When beta is removed, nothing is left** (market-adjusted target: exact tie with pooled).

---

## 5. Lessons

| Lesson | What happened |
|---|---|
| Benchmark against the domain's own null | Every comparison used pooled (beta-blind) or ridge with interactions. The finance null, a CAPM baseline with beta x drift and beta x shock, was only added after an outside critique, and it erased the transformer's edge |
| Compute power before spending a test set | The 32-meeting final test could detect a -0.045 edge only about 1 time in 4. Extending development to 2021 helped diagnosis but left a small final test |
| Read input-importance checks carefully | Price history "carrying the signal" meant pre-computed beta and volatility channels, not learned sequential patterns |
| Crisis meetings dominate unweighted MSE | 2007-2009 meetings carry about 8x the weight of calm ones; any choice made on 1994-2009 MSE is mostly a choice about 2008 |

---

## 6. What next

Ordered by value for effort.

| # | Next step | Why | Cost |
|---|---|---|---|
| ~~1~~ | ~~Multi-day pre-check on CPU~~ **Done 2026-10-16: flat** (section 3.4). Q3 is closed at the one-day horizon, and a longer target horizon is not justified | | |
| ~~2~~ | ~~Add the CAPM baseline to the shared evaluator~~ **Done 2026-10-16**: `BASELINES["capm_pooled"]` (`CAPMPooled` in `core/fedcore/q3/baselines.py`), first reference in `run_arms` tables and in `decouple.confirm`. Reproduces the final-test CAPM score exactly (5.725) | | |
| ~~3~~ | ~~Write the Q3 result into the spec docs~~ **Done 2026-10-16**: result paragraph in `docs/question.md` Q3, outcome note in `docs/methods.md` | | |
| 4 | **Clean up the code** | Delete the first-trial pretraining machinery (`pretrain_return.py`, `return_corpus.py`, `checkpoint.py`, staged fine-tuning in `train.py`, `pretrain-folds` in `workflow.py`). Keep `decouple.py`, `diagnostics.py` and the `workflow.py` panel options | A few hours; tests guard it |
| 5 | **Score the locked model and the CAPM baseline once on the 2026 holdout**, and report 2022-2026 pooled (about 38 meetings) | Completes the record; it won't change the conclusion on its own | About $0.10 GPU; needs the user's go-ahead |
| 6 | **Settle the winner rule** in `question.md` | Only matters if a future b model is tried; it should name the CAPM baseline as the reference | Decision |

**Not recommended:** bigger networks; more firms (top 1,500 tested); more balance-sheet
features (ratings, coverage, FCF, funding gap, collateral and maturity all tested); sector as a
model input (tested; it hurt); FOMC minutes as extra events (weaker surprises, same same-day
bottleneck); point-in-time GSS factors (no signal even with the full-sample version).

---

## 7. Decisions on record

| Date | Decision |
|---|---|
| 2026-10-14 | Pretrain on the same task as prediction (shock = 0 on ordinary days); rejected shock-in-pretraining, intraday windows, portfolio scoring |
| 2026-10-15 | Development period extended from 2009-03-18 to 2021-12-31; 2022-2025 kept as the final test |
| 2026-10-15 | Universe for the locked model: S&P 500 (top 1,500 tested and rejected) |
| 2026-10-16 | Locked spec: two-part model, S&P 500, raw returns, original 6 fundamentals, statement surprise, 3 networks per model; final test run once with seeds 6, 7, 8 |
| 2026-10-16 | After an outside critique, a CAPM baseline was fitted on the same folds (dev and final, nothing tuned on test meetings). It matches the locked model; the report's conclusion on a was revised to "mostly beta x Fed-day drift" |
| 2026-10-16 | Multi-day pre-check (development meetings only) was flat: no trait gains surprise sensitivity over 3 or 5 days. Q3 closed at the one-day horizon; no spec revision of the target horizon. CAPM baseline added to the shared baselines as the first reference |

---

## 8. Reproducibility

**Code** (`encoder/fedenc/`, all committed):

| File | Role |
|---|---|
| `decouple.py` | `StructModel` (centered a), `DecoupleArm`, `TwoPartArm`; commands `run`, `confirm`, `final` |
| `diagnostics.py` | MSE split, window and surprise tables, a/b anatomy, b grades; `keep=` scores a row subset |
| `workflow.py` | `bounded_panel` with `universe`, `sector`, `target` (`market_adjusted`), `shock` (`ME`) |
| `daily.py` | Ordinary-day pretraining, inputs experiment, shared run saving |
| `core/fedcore/q3/baselines.py` | `CAPMPooled` (`capm_pooled`): the CAPM baseline, first reference in Q3 tables |
| `core/fedcore/q3/real.py` | Panel builder; `UNIVERSES` (`sp500`, `top1500`) |
| `core/fedcore/ingest/wrds.py` | WRDS pulls; `--universe top1500`, `--resume`, credit and funding tables |

**Commits:** b634f2d, 599f8ef, 5013427, 1bf92a5, d11951d (diagnostics, decoupling), 3d2c107
(two-part model), 5adc317 (top 1,500), c50304a (credit pull, sector), b673d60 (market-adjusted
target, longer windows), 2dcc6a2 (final test), dc8f7d4 (ME shock).

**Runs** (`encoder/results/q3_susceptibility/`, gitignored):

| Experiment | Run folders (prefix) |
|---|---|
| Decoupling 2x2 | `20261015-1330..1346_decouple-exp-*`, diagnostics `-134713` |
| Two-part, fresh seeds | `20261015-161051..163147_twopart-confirm-*`, diagnostics `-163210` |
| Top 1,500 | `20261015-192804..201129_twopart-top1500-*`, diagnostics `-201141`, `-201151` (S&P rows) |
| Dev to 2021, raw / market-adjusted | `20261015-2053..2107_twopart-confirm-to2021-*` |
| Dev to 2021, ME shock | `20261016-0248.._twopart-confirm-to2021-me-*`, `-0300.._-adj-me-*` |
| **Final test (primary)** | `20261016-024736_q3-final-two-part`, `-024737_q3-final-pooled`, `-024737_q3-final-comparison`, `-024739_q3-final-diagnostics` |

**Data added:** `raw/wrds_top1500/` (top 1,500 universe, ratings, cash-flow and funding items;
licensed, gitignored) with catalog entries and PROVENANCE section 7b.

**Rerun the final test:** `bash vast_upload_dev.sh HOST PORT 2025`, then
`bash vast_run.sh HOST PORT final fedenc.decouple final --device cuda`. Don't rerun it for
selection: the 2022-2025 meetings are used.

**GPU spend:** about $3.50 in total on vast.ai (RTX 3090s at about $0.20/hour).
