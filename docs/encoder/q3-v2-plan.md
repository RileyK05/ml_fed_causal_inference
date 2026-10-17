# Plan: Q3 v2, a regime-aware, two-shock susceptibility model

Written 2026-10-16 for review. **Plan only: no training, no `run_arms`, nothing that saves a
run until the user gives the go-ahead in chat** (AGENTS.md). Background:
[encoder-report.md](../reports/encoder-report.md) (v1 result) and
[question.md](../question.md) (Q3 spec).

## 1. Why a v2

v1 answered its question cleanly: at the one-day horizon, firms' different FOMC-day returns
are beta times the market's move, and the locked transformer tied a three-number CAPM
baseline in the 2022-2025 final test (-0.002). The user's critique, which this plan adopts:

1. **The objective rewarded the ordinary meeting.** Every meeting weighed the same in the loss
   and in the score, and most meetings carry a 1-3bp surprise. The model learned the average
   case well and says nothing reliable about the high-stakes meetings. A car that survives
   every 20 mph crash and few 35 mph ones scores 95% on an average-crash test.
2. **The model could not tell what a surprise meant.** The same rate cut reads as rescue in
   one regime (markets rallied on post-2008 easing) and as bad news in another (the Fed sees
   weakness). The literature calls this the Fed information effect. v1 fed one surprise
   series, so b averaged two shocks with opposite stock effects. The market's response to
   the surprise flipped sign across eras (dovish surprises with a 2.3% market fall in
   2002-08, hawkish ones with a 1.5% rise in 2008-09), which fits.
3. **The model saw almost nothing about the regime.** Its 7 context inputs were the 1-year
   yield level and changes, recent surprises and 21-day market volatility: no curve, credit,
   cycle position or policy-communication regime.

v1's null stands for its question. v2 asks a different one: **does firm-level sensitivity
beyond beta appear once the surprise is split by meaning, the regime is visible, and the
model is graded on the meetings that matter?**

## 2. Hypotheses (fixed before any data work)

| ID | Hypothesis | Main test |
|---|---|---|
| H1 | Split into a policy shock and an information shock, at least one shock has firm-level sensitivity beyond beta | Group tests and b grade per shock, beyond `beta x shock` |
| H2 | The regime changes which firms are sensitive (not just how much the market moves) | Firm trait x shock x regime interaction, beyond `beta x shock x regime` |
| H3 | On large surprises, firms differ in how *much* they move (abnormal-return magnitude) beyond pre-meeting volatility and beta | Magnitude model vs a volatility-only baseline |

Prior evidence to respect: v1 found the cross-firm spread widens on big days, but all of it
was pre-meeting volatility (s^2 term t -0.1 once volatility is controlled). H3 starts as the
weakest hypothesis.

## 3. Step 0: literature pass (CPU, reading only)

Deliverable: `docs/encoder/literature-v2.md`, one row per paper: question, data, shock
measure, finding, and what it implies for our design. Every citation verified before it is
used in a slide. Candidate reading (from memory, **to verify**):

| Topic | Candidates | What we need from it |
|---|---|---|
| Information effect and shock splits | Jarocinski & Karadi (2020); Nakamura & Steinsson (2018); Bauer & Swanson (2023) | How to split the surprise; whether published series cover our sample |
| Multi-dimensional shocks | Gurkaynak, Sack & Swanson (2005); Swanson (2021) | Target / path / asset-purchase factors (v1 tried full-sample GSS only) |
| Heterogeneous stock responses | Bernanke & Kuttner (2005); Ehrmann & Fratzscher (2004); Ozdagli (2018); Ippolito, Ozdagli & Perez-Orive (2018); Gurkaynak, Karasoy-Can & Lee (2022); Ozdagli & Velikov (2020) | Which firm traits matter and in which regime; floating-rate and bank debt; tightening vs easing asymmetry |
| Volatility, not mean | Gorodnichenko & Weber (2016) | Magnitude targets and how they measured them (H3) |
| FOMC-cycle returns | Lucca & Moench (2015); Cieslak, Morse & Vissing-Jorgensen (2019) | The drift in our a; cycle-position regime variables |
| Regime / state dependence | State-dependent monetary transmission papers (to find) | Which regime variables are standard and point-in-time |

Exit: a short list of regime variables and firm traits with published support, and the
shock-split method we will use.

## 4. Step 1: data (CPU)

| Item | Source | Status | Notes |
|---|---|---|---|
| Policy / information split | USMPD `SP500` window change (in `usmpd_statements`, 262 events 1996-2026) with `MP1` / `STMT` | **Have** | 23% of events have rates and stocks moving the same way (information-type). Method from Step 0: simple sign split first, then a published series if one covers the sample |
| Regime variables, daily from 1994 | FRED: VIX, 10y-2y and 10y-3m slope, BAA/AAA or BBB spreads, 3m bill, recession indicator; Chicago Fed NFCI; Gilchrist-Zakrajsek excess bond premium | **Pull** (public) | `fred_controls` covers 2023-2026 only. Every value as of the prior close |
| Cycle position | Fed funds target history | **Derive** | Easing / tightening / hold, meetings since the first move, zero-lower-bound flag |
| Statement text (optional) | FOMC statements | Later | A hawkishness score per statement; only if Step 0 shows it adds over the split |
| Unscheduled meetings | `usmpd_statements` (`Unscheduled = 1`: 18 events, e.g. 2001-01-03 cut with S&P +4.4% in the window, March 2020) | **Have**, spec decision D1 | AGENTS.md keeps them out of the pooled sample. They are the high-stakes events this plan is about |

Data follows the repo rules: catalog entry, PROVENANCE note, `fedcore check`, no filling.

## 5. Step 2: baselines before any network

The v1 lesson: the reference has to be the field's own null, from day one.

| Baseline | Formula | Role |
|---|---|---|
| `capm_pooled` (exists) | `c + beta (d + k s)` | v1 reference |
| `capm_two_shock` | `c + beta (d + k_p s_p + k_i s_i)` | **Primary reference for v2** |
| `capm_two_shock_regime` | adds `beta x s_p x regime` and `beta x s_i x regime` (few regime terms) | Reference for H2: the market's response varies by regime, scaled by beta |
| `ridge_traits_two_shock` | ridge on traits x `s_p`, traits x `s_i` | Tabular check: is there anything a linear model finds? |
| `vol_only` (H3) | `|abnormal return| ~ pre-meeting vol x |s|` | Reference for the magnitude model |

A network is only worth building if Step 3 finds something these do not already explain.

## 6. Step 3: descriptive gates (CPU, development meetings only)

The same group tests as v1 section 3.3 (top 1,500, day 0 and days 0-2, raw and within
sector), rerun on the new pieces. Each gate is a joint, meeting-clustered test with the count
of tests reported.

| Gate | Test | Pass |
|---|---|---|
| G1 | Group excess response to `s_p` and to `s_i` separately (sector, rating, funding, maturity, growth/value, size) | Joint p < 0.01 for some trait on either shock, holding within sector, in both 1996-2008 and 2009-2021 |
| G2 | Trait x shock x regime, beyond `beta x shock x regime` | Same bar |
| G3 | On large-surprise meetings (\|s\| >= 2.5bp, plus unscheduled if D1 allows), does any trait predict \|abnormal return\| beyond pre-meeting volatility and beta? | Same bar |

**Kill rule:** if all three gates fail, stop. Write the result into the report and do not
build a network. This keeps to "build cheap, trash it if it doesn't work".

## 7. Step 4: model (only if a gate passes)

Built on v1's two-part structure, changed in four ways:

1. **Beta is in the base, not the network's job.** `mu = capm_two_shock_regime(beta, s_p, s_i,
   regime) + centered network deviations`. The networks can only add what beta and the regime
   terms miss, so a tie with the baseline means exactly "nothing beyond beta".
2. **Two shocks, two b's.** The head outputs `b_p(z)` and `b_i(z)`;
   `mu = a(z) + b_p(z) s_p + b_i(z) s_i`.
3. **A regime branch.** Regime variables pass through a small MLP that scales and shifts the
   firm embedding before the b heads (FiLM). Regime enters only b, keeping centered a.
4. **A tail-weighted fine-tuning loss.** Meeting weights `w = min(1 + alpha s^2 / mean(s^2),
   w_max)`, alpha in {0, 1}, w_max fixed in advance so a few meetings cannot dominate. Only if
   G3 passes: a variance head (Gaussian NLL, `log v + (r - mu)^2 / v`) for the magnitude
   target.

Size stays near v1's (d_model 64, 3 members). No pretraining in the first pass: v1's
pretraining never beat training from scratch. Unscheduled meetings in training only if D1
says so.

## 8. Step 5: evaluation

- **Folds:** v1's walk-forward by meeting, development through 2021-12-31, the same test windows.
- **Metrics, all against `capm_two_shock(_regime)`:**
  - Overall per-meeting MSE (kept for continuity).
  - **Large-surprise MSE:** meetings with \|s\| >= 2.5bp (and unscheduled if D1).
  - **Tail error:** mean of the worst 10% of per-meeting errors (the 35 mph crash).
  - b grade per shock (within-meeting slope; 1 = calibrated), on all and on large meetings.
  - RankIC on large-surprise meetings.
- Meeting-bootstrap ranges throughout; N is meetings, not rows.
- **Power first:** before any confirmatory test, compute the detectable effect on the
  large-surprise subset. If the confirmatory test cannot detect a plausible effect, say so
  rather than run it.

## 9. Final-test status (decision D2)

2022-2025 has been used once, for v1's spec. It has not been used to tune v2, but its results
are known, so reusing it is weaker evidence. Options for the reviewer:

| Option | For | Against |
|---|---|---|
| A. Write the v2 spec down, then test once on 2022-2026 (2022-25 plus the 2026 holdout) | Most meetings (about 38) | 2022-25 already seen |
| B. Test only on 2026 and forward | Clean | 6-8 meetings: no power |
| C. Treat v2 as development-only evidence; no confirmatory claim yet | Honest | No headline number |

Recommendation: C now, plus A as a clearly labelled secondary check once the spec is fixed.

## 10. Risks

| Risk | Mitigation |
|---|---|
| Few large-surprise meetings (about 25 at 5bp or more, 19 in development) | Power calculation first; a 2.5bp threshold; unscheduled meetings if D1 |
| Few regime episodes (perhaps 5-8 in 32 years), so few independent observations | Two or three regime variables, not twenty; regime enters through a small branch |
| Many knobs means overfitting the development data | Hypotheses, gates and metrics fixed here; report every test run |
| Sign-split noise (a 30-minute stock move can carry other news) | Compare with a published decomposition if Step 0 finds one covering 1996-2021 |
| Look-ahead in regime data | All values as of the prior close; revised macro series avoided or vintage-dated |
| External validity: a Warsh-era Fed that leans away from forward guidance may not behave like the 2008-2021 guidance era | State it as a limitation; regime variables for communication style if Step 0 finds a measure |

## 11. Decisions for the reviewer

| ID | Decision | Default if not decided |
|---|---|---|
| D1 | Pool unscheduled meetings into training and evaluation, or keep them as a separate evaluation set | Separate evaluation set (keeps AGENTS.md's rule) |
| D2 | Final-test strategy (section 9) | Option C, plus A as a labelled secondary |
| D3 | Shock split: simple sign split, or a published series | Sign split first, published series as robustness |
| D4 | Tail weight alpha and cap w_max | alpha in {0, 1}, w_max = 5 |
| D5 | Is H3 (magnitude) in scope for v2 | In scope only if G3 passes |

## 12. Order, cost, sign-off

| Step | What | Compute | Needs go-ahead |
|---|---|---|---|
| 0 | Literature pass | None | No |
| 1 | FRED pull, shock split, cycle variables | CPU, minutes | No (public data, no training) |
| 2 | New baselines in `core/fedcore/q3/baselines.py`, with tests | CPU | No (code and tests) |
| 3 | Descriptive gates G1-G3 | CPU, under an hour | No (descriptive, development only) |
| 4 | Model code and unit tests | None | No |
| 4b | Training runs | GPU, about $2-5 | **Yes** |
| 5 | Confirmatory test | GPU, under $1 | **Yes**, after D2 |

Steps 0-3 are done and reported before any model code. If the gates fail, v2 ends at Step 3
with a written null.
