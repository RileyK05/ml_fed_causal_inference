# Q3 v5 B3 first test: the EmbeddingGemma 2 fundamentals tower

Run 2026-10-19/20, exploratory, development data only (161 walk-forward test meetings 2001-2021;
2022-2025 untouched). Design, arms and criterion: `q3-v5-plan.md`, "B3 first test" (written
before any text run). Code: `fedenc.text_tower`, `fedenc.v5text`. One seed, one member per fold,
RTX 3090s on vast.ai.

| Arm | Run |
|---|---|
| T0 (no text) | `20261020-014915_v5text-to2021-t0` |
| T1 (Gemma, top 4 layers fine-tuned, attention pooling) | `20261020-105039_v5text-to2021-t1` |
| T1m (same, mean pooling) | `20261020-052509_v5text-to2021-t1m` |
| T1f (Gemma frozen, attention pooling) | `20261020-045059_v5text-to2021-t1f` |
| Conditional score | `20261020-065548_v5text-to2021-score` |

Text input: the latest 10-K/10-Q market-risk section accepted before the meeting day, 1,024
tokens, rate sentences first. 84% of firm-meetings have a filing (90%+ from 2002).

## Verdict

**Text adds nothing. No arm passes.** The criterion was T1 or T1m beating T0 on the top-decile
hit rate without the two standout meetings, with the interval excluding T0's rate, and not losing
on firm-vs-firm error. Both fail on both counts.

## Conditional view (|s| >= 2.5bp, 40 meetings without 2004-01-28 and 2011-09-21)

Hit = the firm's realized move, net of the base, goes the way the model's departure from beta
said. 95% meeting-clustered intervals.

| Arm | Top-decile hit | Mean departure (pt) | Realized toward (pt) | Vulnerable hit (n) | Low-DD hit (n) | Firm-vs-firm error vs B0 |
|---|---|---|---|---|---|---|
| T0 | 0.485 (0.454, 0.516) | 0.014 | +0.036 | 0.71 (7) | 0.41 (68) | **-0.0013 (-0.0030, -0.0001)** |
| T1 | 0.500 (0.465, 0.535) | 0.073 | +0.020 | 0.50 (82) | 0.48 (128) | +0.0014 (-0.0006, +0.0037) |
| T1m | 0.493 (0.467, 0.519) | 0.051 | -0.003 | 0.58 (127) | 0.53 (202) | +0.0009 (-0.0008, +0.0026) |
| T1f | 0.502 (0.472, 0.531) | 0.056 | -0.048 | 0.51 (39) | 0.52 (71) | +0.0008 (-0.0005, +0.0022) |

Error: negative = smaller squared error than B0 on the within-meeting (firm-vs-firm) part.

Context (not the criterion): firm-vs-firm error on |s| >= 5bp and all-meeting error vs B0 are
within noise of zero for every arm. The spread of the correction to b (delta_b sd) is 0.067 (T0),
0.084 (T1), 0.066 (T1m) and 0.054 (T1f).

## Reading

1. **The text makes the model move more, not better.** With text, the top decile departs from
   beta 3.5-5x as far as T0's, but those firms' realized moves go the predicted way about half the
   time, and on average by roughly nothing. T0 barely departs from beta and is the only arm that
   beats B0 on firm-vs-firm error (by a small amount).
2. **Fine-tuning and pooling make no difference.** Frozen, fine-tuned, attention pooling and mean
   pooling all land at a coin flip. A gain in only one variant could have been a training issue.
   With no gain in any of them, the problem is either upstream (what the text holds) or the shared
   training setup (see the caveats below).
3. **The vulnerable-firm hint doesn't hold up.** T1m's 0.58 on vulnerable firms was the one
   promising number in the preview. It has an interval reaching below 0.5, and T1, the arm the
   criterion is about, is at 0.50 on the same slice. Distress remains the Track A finding. The
   text doesn't add to it.

## Caveats found after the write-up (checked against the run tables)

These weaken point 2 above: the null is a null for this training setup, not yet for the text.

1. **Some folds never trained.** Early stopping picked epoch 0, so stage 2 took zero steps and
   delta_b = 0:
   - T1 on folds 0 and 6 (32 of 161 test meetings);
   - T1f on folds 0 and 2;
   - T1m on fold 4.

   The validation loss moves in the 4th decimal across epochs, because validation blocks are
   mostly small-|s| meetings. So epoch selection is close to random for every arm.
2. **Every v5 arm is undertrained next to v4.** Each step sees 24 of ~500 firms per meeting, so
   at the selected 1-3 epochs a firm-meeting is seen 0.05-0.15 times. T0 shares this, so T1 vs T0
   is a fair comparison, but both are weaker than the v4 arms.
3. **The v5 inputs drop the tabular fundamentals and context** (`V5TextArm._data` keeps only
   history plus the document). Again, T0 has the same inputs. A fairer "does text add" test puts
   text on top of the v4 inputs.
4. **No sign of a firm-ID fingerprint** (scratchpad `fingerprint.py`, folds with zero steps
   dropped):
   - the text arms' corrections are no more firm-fixed than T0's: the firm-mean share of variance
     is 0.11-0.13 vs 0.14 for T0;
   - next-meeting persistence is 0.81-0.85 vs 0.77;
   - rank correlation with past own Fed response is +0.00 to +0.03, the same as T0.

   They lean slightly more toward distance to default than T0 does (+0.05 to +0.08 vs +0.00).

## What this does and doesn't rule out

- **Rules out (on this data):** a market-risk section read by EmbeddingGemma 2 and fed into b
  predicting which firms beat their beta on big surprises.
- **Doesn't rule out:**
  - other text, such as Fed statements as a FiLM condition, or 8-K news;
  - other uses of the text, such as finding twins for B2 rather than predicting b directly;
  - a tower that could learn a real text signal but sees too few big-surprise meetings (40) to
    do it.

  The last case is what the Track C synthetic check is for: plant a known text-carried channel
  and see whether this pipeline recovers it.

## Run notes

- About 3.2 hours per arm on one RTX 3090.
- T1's first box went offline (host outage at 01:38 on 2026-10-20), and T1 was re-run on a
  fresh box with identical code and data.
- Total rental across the T-arms was roughly $5.
