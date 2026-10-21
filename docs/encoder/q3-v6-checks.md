# Q3 v6 quick checks (C1-C3): results

Run 2026-10-20, exploratory, development data only. Setup and pass bar are in `q3-v6-plan.md`
section 2, written before the run.
- Script: scratchpad `v6_cpu.py`; nothing saved to a ResultStore.
- Sample: 105,198 firm-meetings, 221 scheduled meetings 1994-2021, earnings-day firms dropped.
- Units: pp of day-0 return per SD of the channel, per 10bp surprise, beyond both betas and
  sector.

## Verdict

**None of the 11 tests passes.** Distress stays a plain linear amplifier:
- the same on hikes and cuts;
- not gated by what the filing says about floating or hedged debt;
- not routed by the yield curve or the tightening cycle.

Section 3 keeps the plain form, so no asymmetric head, no text gate from these measures and no
regime input are built in up front.

| Check | Test | All (t) | 1994-2008 (t) | 2009-2021 (t) |
|---|---|---|---|---|
| C1 | DD: hike slope minus cut slope | -0.19 (-0.61) | -0.61 (-2.10) | +1.05 (0.95) |
| C1 | Coverage: hike minus cut | +0.06 (0.41) | -0.03 (-0.20) | +0.01 (0.02) |
| C1 | Leverage: hike minus cut | -0.02 (-0.14) | -0.00 (-0.02) | +0.27 (0.98) |
| C1r | DD, recession control | +0.21 (0.62) | -0.54 (-1.87) | +0.99 (1.20) |
| C1r | Coverage, recession control | +0.22 (1.37) | +0.07 (0.45) | +0.01 (0.04) |
| C1r | Leverage, recession control | -0.07 (-0.61) | -0.14 (-1.12) | +0.25 (0.93) |
| C2 | DD x floating tilt | -0.06 (-0.74) | -0.06 (-0.66) | +0.04 (0.65) |
| C2 | Leverage x floating tilt | +0.03 (0.67) | +0.05 (1.05) | -0.12 (-1.29) |
| C2 | DD x hedging (swap mentions) | +0.07 (2.01) | +0.09 (2.47) | -0.00 (-0.01) |
| C2 | DD x Compustat pass-through | +0.03 (0.51) | +0.02 (0.33) | +0.06 (0.86) |
| C3 | DD x curve slope | +0.11 (1.04) | +0.14 (1.20) | +0.01 (0.01) |
| C3 | DD x tightening | -0.03 (-0.35) | -0.06 (-0.61) | -1.72 (-1.53) |
| C3 | Coverage x curve slope | +0.05 (1.04) | +0.05 (1.09) | -0.02 (-0.16) |
| C3 | Coverage x tightening | +0.02 (0.57) | +0.00 (0.04) | -0.34 (-0.84) |

Bar: p < 0.0045 on all meetings and the same sign with p < 0.05 in both eras.

## Notes

1. **Hikes vs cuts.**
   - Without the recession control, the DD effect looks larger on cuts (slope 0.27) than on hikes
     (0.08).
   - With the control, it reverses (0.28 hikes, 0.07 cuts).
   - The apparent asymmetry is the 2001 and 2008 easing in recessions, which the plan expected.
2. **Hedging.**
   - The only test near the bar is DD x hedging (t 2.0, all of it 1994-2008).
   - Its sign is the opposite of the hedging story: distressed firms that mention swaps more do
     slightly worse on hawkish surprises.
   - One possible reading: swap language marks firms with more debt to hedge.
   - It's a lead at best.
3. **Floating tilt covers only 38% of firm-meetings**, because most market-risk sections mention
   neither floating nor fixed rates. Swap mentions cover 81%. Word counts are a crude read of the
   text; the cached Gemma vector in section 3 (arm F2) is the better test of the text.
4. **Yield-curve regime.** The 2009-2021 tightening terms are large but imprecise: there was little
   tightening in that era outside 2015-2018 and 2022 (2022 is out of sample).
