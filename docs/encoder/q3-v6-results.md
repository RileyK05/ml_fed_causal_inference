# Q3 v6 results: macro-day pretraining and filing gates

Run 2026-10-20, exploratory, development data only (161 walk-forward test meetings 2001-2021;
2022-2025 untouched). Plan and pass bar: `q3-v6-plan.md`, written before the runs. 3 seeds x 3
members per arm, four RTX 3090s, about $5. Score run: `20261020-180616_v6-to2021-score`.

## Verdict

**No arm passes.** Neither idea helps:
- **Macro-day pretraining:** F1 vs F0 is -7bp (-99, +57), and on the summary model F1s vs F0s is
  -106bp (-229, +20).
- **The filing gate:** F2 vs F1 is -5bp (-51, +30). Its year-over-year change version: F3 vs F2
  is +1bp (-45, +54).

The plain summary MLP, **F0s**, is the strongest arm. It reproduces v4's B4 almost exactly.

## Beyond-beta spread (bp of extra response per 10bp surprise, top vs bottom decile)

| Arm | Headline: \|s\| >= 2.5bp, no standouts (40) | All \|s\| >= 2.5bp (42) | \|s\| >= 5bp (10) | Rank IC (headline) | MSE vs B0, all meetings |
|---|---|---|---|---|---|
| F0 transformer | +18 (-32, +75) | +27 (-20, +81) | **+55 (+15, +131)** | +0.015 | -0.0006 |
| F1 F0 + macro-day pretraining | +11 (-70, +69) | +25 (-51, +80) | +50 (-35, +117) | +0.019 | -0.0007 |
| F2 F1 + filing gate | +6 (-97, +68) | +20 (-73, +79) | +48 (-56, +118) | +0.024 | -0.0007 |
| F3 F2, filing change | +7 (-85, +65) | +18 (-65, +72) | +48 (-63, +114) | +0.020 | -0.0002 |
| **F0s summary MLP** | **+48 (-20, +105)** | +62 (-1, +117) | **+97 (+28, +167)** | +0.011 | -0.0005 |
| F1s F0s + macro-day pretraining | -57 (-153, +24) | -34 (-127, +47) | -11 (-111, +104) | +0.001 | -0.0002 |

- Every arm meets the MSE guardrail.
- For comparison, v4's B4 (the same summary MLP, older selection rule) gave +48 (-18, +99) on the
  headline slice and +100 (+40, +161) on |s| >= 5bp.

## Reading

1. **The one steady result is on the largest surprises.** On the 10 meetings with |s| >= 5bp,
   the plain models rank firms beyond beta:
   - F0s +97bp per 10bp, F0 +55, v4 B4 +100;
   - intervals exclude zero for all three.

   On the broader headline slice the same arms are positive but not significant. The 5bp
   meetings are mostly 2001-2008.
2. **Pretraining on macro shock days hurts, especially the summary model.**
   - Pretraining stops after about 1 epoch on its own validation days: what predicts beyond-beta
     moves on CPI and jobs days doesn't carry over to the next stretch of such days.
   - Starting the Fed fit from those weights pulls the model away from what works on Fed days.
   - Track A had found fragility amplifies any big day, but a learned beyond-beta map from rate
     news days doesn't transfer.
3. **Text: the fifth null.** It follows the three v5 Gemma arms and F2. As a gate on the
   correction, the filing vector, or its change, adds nothing.

## Next step (proposed, needs the user's go-ahead)

One confirmatory run of F0s on the untouched 2022-2025 meetings: `q3-v6-final-plan.md`.
