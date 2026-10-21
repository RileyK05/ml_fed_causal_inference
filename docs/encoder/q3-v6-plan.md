# Q3 v6 plan: score on departures from beta, learn fragility from more shocks

Written 2026-10-20, before any v6 run. Development data only (scheduled meetings through
2021-12-31); 2022-2025 stays untouched. Builds on Track A (`q3-v5-track-a.md`), the v5 text
test (`q3-v5-text-results.md`) and the diagnosis notes (`q3-v5-mse-diagnosis.md`).

## 1. Headline metric: the beyond-beta spread

MSE is a do-no-harm guardrail only. The ~5.9 floor is daily news no Fed model can see, and a
real t = 2.9 effect moves it in the 3rd-4th decimal.

Code: `fedcore.q3.beyond_beta` (tests: `core/tests/test_q3_beyond_beta.py`). Per meeting, with
the base = the two-beta CAPM (`capm_beta_both`):

- **dev** = model prediction minus base prediction, demeaned within the meeting;
- **miss** = realized return minus base prediction, demeaned within the meeting.

Metrics:

| Metric | Definition | Unit |
|---|---|---|
| **Spread (headline)** | Firms ranked by predicted extra sensitivity (dev / s). Realized miss of the top decile minus the bottom decile, pooled as the through-origin slope of that gap on s. A meeting with no ranking counts as a zero gap. | bp of extra response per 10bp surprise |
| Claimed spread | The same with dev in place of miss | same |
| Rank IC | Mean within-meeting Spearman correlation of dev and miss | correlation |
| Calibration | Pooled within-meeting slope of miss on dev | 1 = sizes right, 0 = noise |

- Intervals resample whole meetings.
- Slices:
  - headline: |s| >= 2.5bp without 2004-01-28 and 2011-09-21;
  - also all |s| >= 2.5bp and |s| >= 5bp.
- Guardrail: all-meeting MSE vs B0 at most +0.005 (as v4).
- Kept as readable checks: the top-decile hit rate and the vulnerable and low-distance-to-default
  slices.

**Pass bar for a v6 model arm:**
1. headline spread lower bound > 0;
2. the paired meeting-bootstrap difference vs its control arm has lower bound > 0;
3. guardrail met.

**Context, the old runs under the new metric** (scratchpad `bb_context.py`; read-only, 12 arms,
no multiple-testing correction):
- On |s| >= 5bp (10 meetings), several v4 arms show a positive spread: B4 summary MLP +100bp
  (+40, +161), B3c +74, B5 +56. Rank ICs there are about +0.08 to +0.10.
- Without the standout meetings, no v4 arm's interval excludes zero.
- The v5 text arms are all within noise of zero (T1 +10bp (-29, +39) on the headline slice). T0
  barely ranks at all (claimed spread 3bp).

So the v5 setup (24 firms per step, no fundamentals) lost the v4 arms' ranking ability, weak as
it was. That points at v6 needing the full v4 inputs.

## 2. Quick checks (regressions, CPU, minutes)

Same machinery as Track A (`track_a_single.py`):
- within-meeting z-scored channels;
- both betas x s and GICS sector x s held fixed;
- channel main effects;
- meeting fixed effects;
- standard errors clustered by meeting;
- earnings-day firms dropped.

The channels are distance to default (DD), interest coverage and leverage: Track A's leaders.
Units as Track A: pp per SD per 10bp. Eleven tests across C1-C3, so **p < 0.0045 (0.05/11)**,
plus the same sign with p < 0.05 in 1994-2008 and in 2009-2021.

**C1. Hikes vs cuts (Merton convexity).**
- Model: channel x s+ and channel x s- as separate slopes (s+ = max(s, 0), s- = min(s, 0)).
- Test (3): slope(s+) - slope(s-) for each channel.
- Big dovish surprises bunch in 2001 and 2008, so a second version adds channel x s x
  recession (NBER recession months 2001-03 to 2001-11, 2007-12 to 2009-06, 2020-02 to 2020-04).
  The asymmetry counts only if it survives that version.

**C2. Text as a gate on fragility.**
- Text measures from the latest usable 10-K/10-Q market-risk section, point in time (as
  `filing_for_rows`):
  - floating tilt = floating / (floating + fixed) mentions per thousand words, NaN when both are
    zero;
  - hedging = swap mentions per thousand words.
- Tests (4): DD x floating tilt x s; leverage x floating tilt x s; DD x hedging x s; and the
  Compustat version, DD x pass-through x s.
- All lower-order terms included. Expected signs: floating debt makes distress hurt more on
  hawkish surprises; hedging makes it hurt less.

**C3. Does the regime route the shock?**
- Regime variables, known the day before the meeting from GSW yields:
  - curve slope = 10y minus 2y;
  - tightening = the 2y yield change over the prior 126 trading days.
- Tests (4): DD x s x slope, coverage x s x slope, DD x s x tightening, coverage x s x
  tightening. Regime variables are z-scored across meetings.

If a check passes it shapes the model in section 3 (an asymmetric head, a text gate, a regime
input). If not, it is reported as is and the model keeps the plain form.

## 3. The model: learn fragility from macro shock days, score on Fed days (training, needs the go-ahead)

Every arm so far starves on about 40 big Fed meetings. Track A's placebo found that distress
amplifies any big market move, not only Fed ones. So:

1. **Shock days.**
   - Events: non-FOMC trading days 1994-2021 where |change in the GSW 2y yield| >= 5bp
     (CPI, jobs, other macro news).
   - Surprise: that change in 10bp units. It mixes rate and growth news, so it is a noisier
     proxy than the Fed surprise.
   - FOMC days and the day after are excluded. Pretraining events never come from a Fed test
     block's dates or later.
2. **Pretraining** (`fedenc.v6.V6Arm`, data `fedcore.q3.shock_days`).
   - Inputs and setup: the v4 B4/B5 setup with all ~500 firms per event and the full v4 inputs
     (history, fundamentals, context). It learns beyond-beta responses on shock days with its
     own two-beta base.
   - Which days: only shock days dated before the fold's last training meeting.
   - Early stopping: on the last 15% of those days (20 epochs max, patience 3); the best weights
     are kept.
   - One pretraining per fit, shared by its members. Pretrained arms scale inputs with the shock
     days' inner-block statistics throughout.
   - Both Fed stages start from the pretrained weights, so a stage-1 choice of epoch 0 means
     "the pretrained model as is". Without pretraining, epoch 0 means the base exactly.
3. **Fine-tuning and evaluation** on Fed days, on the v4 walk-forward folds (3 seeds x 3
   members, as v4), scored with section 1.
4. **Arms:**
   - F0: transformer, as v4 B5 with the new selection;
   - F1: F0 pretrained on shock days;
   - F2: F1 plus a bounded gate from the cached frozen EmbeddingGemma 2 filing vector:
     delta_b x (1 + 0.5 tanh(g(vector))), gate off for rows without a filing. Text can scale a
     correction but cannot create one.
   - F3: F2 with the year-over-year change in the vector (minus the same firm's same-form filing
     300-430 days earlier) instead of its level;
   - F0s / F1s: the summary-MLP pair (v4 B4), robustness for F1 vs F0.

   Filing vectors (`fedenc.filing_vectors`): the 768-d mean-pooled last hidden state of frozen
   EmbeddingGemma 2 on the market-risk section (v5 prompt, rate sentences first, up to 2,048
   tokens). The row gets the latest usable filing accepted strictly before the day.
5. **Comparisons:** F1 vs F0 (do shock days help?), F2 vs F1 (does text add?), F3 vs F2
   (level vs change), F1s vs F0s.
6. **Model selection on big-surprise validation meetings only** (|s| >= 2.5bp). If a fold's
   validation block has none, every validation meeting is used, and `folds.csv` records which.

Cost: one GPU pass to cache the filing vectors (about 30 min on a 3090), then the six arms.
