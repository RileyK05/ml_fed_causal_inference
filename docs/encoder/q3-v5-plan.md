# Q3 v5 plan: what makes a firm Fed-sensitive, and what the encoder can do that ridge can't

Draft 2026-10-18, from the chat after the v4 results. Tower comparisons and review notes added
by Astra on 2026-10-19. Track A exploratory results are in [q3-v5-track-a.md](q3-v5-track-a.md);
the tower experiments below remain proposals, not completed runs or training authorization.

## 1. The question, restated

Beta is a firm's average co-movement with the market on all days. It stacks duration, leverage,
refinancing need, sector, size and whatever drove the market in the window. v1-v4 asked "does
anything beat beta?" and scored on average error. v5 asks:

1. **Which channels make a firm respond more than others when the Fed moves the market**, holding
   the rest fixed (Track A)?
2. **What explains the outliers**: the firms that move far from their beta and from firms that
   look like them (Tracks B and C)?

Scoring is conditional throughout (section 6). Average MSE is reported as context only.

## 2. Data

WRDS is unavailable for about two weeks (Duo re-enrollment), so v5 uses what is on disk plus free
sources. Each WRDS dataset the plan wanted has a stand-in built from on-disk data
(`fedcore.q3.channels`, one row per panel firm-meeting, cached in gitignored
`processed/q3_channels/`):

| Wanted (WRDS) | Stand-in | Channel(s) |
|---|---|---|
| OptionMetrics implied Fed-day move | The firm's abnormal moves on past FOMC, CPI and jobs days vs other days; pre-meeting volatility run-up | `announce_premium`, `vol_runup` |
| Markit CDS | Naive Merton distance to default (CRSP equity, Compustat debt) | `distance_to_default` |
| Capital IQ floating-rate debt | Interest-cost pass-through from the 1-year yield, last 12 quarters | `pass_through`, `rate_paid` |
| Refinancing exposure | Short-term debt share, debt issuance and repayment | `st_debt_share`, `debt_issue_4q`, `debt_repay_4q` |
| Duration | Dechow-Sloan-Soliman implied equity duration | `implied_duration` |
| 13F / peer networks | Hoberg-Phillips TNIC-3 text peers | `tnic_*`, `fed_beta_vs_peers` |
| (none needed) | 2-year rate beta (GSW curve); the firm's own past beyond-beta Fed response; interest coverage; FCF; rating | `rate_beta_2y`, `fed_beta_past`, `int_coverage`, `fcf_4q`, `rating_num` |

| Source | What | Status |
|---|---|---|
| On disk | CRSP daily, Compustat quarterly (debt, interest, cash flow, issuance), ratings to 2017, 1-year yield, CPI and jobs dates | Channels built (`fedcore.q3.channels`) |
| Federal Reserve Board (free) | GSW zero-coupon curve, 1961-present | Pulled (`fedcore.ingest.gsw`) |
| federalreserve.gov (free) | FOMC statement texts, 247 statements 1994-2026 | Pulled (`fedcore.ingest.fomc_text`) |
| Hoberg-Phillips TNIC-3 (free) | Yearly text-based product-market peers, 1988-2023 | Pulled (`fedcore.ingest.tnic`) |
| SEC EDGAR (free) | 10-K Item 7A and 10-Q Part I Item 3 market-risk sections; 8-K item codes (news flags) | Code ready (`fedcore.ingest.edgar`); needs the requester's name and email for the SEC User-Agent |
| WRDS | OptionMetrics, Markit CDS, TRACE, IBES, 13F, RavenPack | After Duo is restored, if subscribed |

EDGAR notes:

- Item 7A was phased in from 1997 to 1998. Before that, the disclosure does not exist, so it stays
  NaN; it is never back-filled.
- Many 10-Qs say "no material change since the 10-K". That is a statement, not a gap. Store it as a
  flag (`no_change = 1`, pointing at the filing it refers to), not as a copied value.
- A filing is usable only from the day after its EDGAR acceptance date: no look-ahead.
- Keep only the parsed market-risk section and a small set of extracted numbers (rate-shock
  sensitivity, floating-rate share, hedged share), plus a text embedding. Not the full filings.

## 3. Track A: controlled tests (no model; CPU)

Each candidate channel (leverage, cash, interest coverage, rating, floating-rate share, CDS spread,
option-implied Fed-day move, the firm's past response to rate moves) gets three tests.

1. **Matched pairs.** Pair firms in the same meeting with the same beta (both betas), the same
   sector and similar size, differing on the channel (top vs bottom third). Statistic: the pair's
   return gap per unit of s, on meetings with |s| >= 2.5bp. Meeting-clustered intervals.
2. **Fed days vs ordinary big days (placebo).** The same pair gap per unit of market move on
   non-FOMC days with a market move as large as the FOMC-day move (ordinary days, CPI and jobs
   days separately). A channel is **monetary-specific** only if its Fed-day effect exceeds its
   ordinary-day effect. Equal effects mean it is beta in another form.
3. **DML with channels as each other's controls.** Partially linear model of the firm's Fed-day
   response, with channel x s as the target and every other channel, both betas, sector and size
   as controls, using the Q1 DML machinery from `fedcore` (no import from `dml/`; anything shared
   moves to core).

Reading: the surprise is as good as random within the statement window, so the response to it is
identified. Which firms carry which traits is not random. Tests 1-3 show what *contributes* with
the rest held fixed; a causal reading of a channel needs it to pass test 2.

## 4. Track B: encoder designs ridge cannot express

**Evidence entering this track.** [v4](q3-v4-results.md) found no supported improvement on its
primary test, including no advantage for b-only over joint training. Track A reports that no
channel clears its stated bar. Distress is a lead, not an established predictor; null results
for past-response summaries do not establish that firm sensitivity is inherently unstable or
that richer histories cannot help. The new hypothesis is that filings contain changing exposure
information missing from the existing inputs, and that this information helps interpret history.

### B1. Event-history encoder (each firm's sensitivity read from its own record)

- **Tokens.** One token per past rate event for this firm, from all available data before the
  meeting:
  - every trading day's 2y yield change, CPI and jobs-report days, past FOMC meetings;
  - per token: event type, shock size, days before the meeting, the firm's response, and the
    firm's state *at that time* (fundamentals, CDS, implied vol, filing values in force), plus
    market state (rate level, credit spread, VIX).
  - Thousands of events per firm instead of about 250 meetings.
- **Time variation (the Amazon point).** Amazon in 2000 (unprofitable, long-duration, levered) was
  far more rate-sensitive than in 2016. A firm's whole-history average mixes the two. The design
  handles it three ways:
  - each token carries the firm's state at that event, so the model can learn "state ->
    sensitivity" across firms rather than memorize firm identity;
  - each token carries its age, so attention can weight recent events more;
  - lookback-length ablation (1 year, 3 years, all history).

  Explicit check: firms whose split-sample Fed sensitivity changed most; does the model track the
  change, or lag it like a long-window beta?
- **Stages.**
  - *Pretraining:* predict a firm's response to a held-out ordinary-day rate event from its
    earlier events, across all firms and days. Daily yield moves are not clean shocks (they mix
    in other news), which is acceptable for learning how to read a response history.
  - *Fine-tuning:* FOMC meetings, with the clean high-frequency surprise s as the shock.
  - *Prediction:* no weight updates. The firm's recent event history is the input, so a change in
    behavior shows up as soon as it is in the record (in-context).
  - Walk-forward by meeting as before. Pretraining for a test meeting uses only days before that
    window's training cutoff.

### B2. Twins and outliers (who breaks from firms like them)

1. Embed each firm from **pre-meeting inputs only** (event history, fundamentals, text, holdings).
   Never the Fed-day response, or the clustering is circular.
2. For each firm, find its nearest neighbors in that embedding: its "twins", including twins from
   other industries and sizes.
3. On big meetings, score each firm's response against the average of its twins (a learned
   synthetic control). Flag the biggest departures.
4. Keep the full eligible sample for the main score. Repeat with retrospective earnings/news
   exclusions as a sensitivity analysis. The news pass found important contamination, not that
   most departures have a demonstrated earnings explanation. Departures identified using actual
   meeting-day returns are retrospective diagnostics, not flags available for prediction.
5. Probe each flagged firm (Track C, step 2): which inputs separate it from its twins? Then check
   those inputs with the Track A tests across all firms.

Also report the reverse: firms that cluster with twins *outside* their GICS industry and size
bucket, and whether those clusters predict joint Fed-day moves better than sector does (the
insurers-and-REITs case in December 2008).

### B3. Fundamentals tower: a trained text encoder (user design, 2026-10-19)

Two towers, each 64-d, like the market tower already is:

```
market tower:       return history (existing patch transformer)                      -> 64-d
fundamentals tower: the firm's filings text (10-K/10-Q market risk and liquidity,
                    8-Ks before the meeting), read by a text encoder                  -> 64-d
                    [market 64 | fundamentals 64] = 128 -> heads -> a_i, b_i ; mu = a_i + b_i s
```

The fundamentals tower is an encoder trained through the a + b·s loss, not a fixed lookup of
precomputed vectors.

- **Start from a pretrained text model and fine-tune it; do not train from scratch.** About 50,000
  filings are far too few to learn English, and plenty to adapt a model that already reads it.
  Candidates, run as separate arms rather than picked in advance:
  - *Lead arm (user's pick): EmbeddingGemma 2* (Google, released 2026-10-06). Text backbone 130M
    parameters plus a 140M embedder (load the text path only; the vision and audio encoders are
    not needed). 8,192-token context with 1,024-token sliding-window local layers and periodic
    global layers. Mean pooling into 768-d, Matryoshka-truncatable to 512/256/128. Card says
    Apache 2.0 but links Gemma terms: check before use. Run in bfloat16 or float32, never float16
    (the card warns of NaNs). Small enough to fine-tune fully on one 3090.
  - *Decoder, last-token pooling:* Qwen3-Embedding-0.6B (Apache 2.0, 32K-token context, so most
    market-risk sections fit whole), LoRA fine-tuned; the 4B model if 0.6B shows signal. The
    representation is built left to right and read off the final token.
  - *Encoder, finance-adapted:* FinModernBERT-embed-large (ModernBERT-large adapted on SEC filings
    and financial news; trained on 1,024-token windows).
  - *Bag-of-words control:* TF-IDF of the same sections into the same heads. FinMTEB (EMNLP 2025)
    found bag-of-words beat dense models on financial similarity tasks, so the neural arms must beat it.
- **Long filings: hierarchical reading** (encoder arm; the 32K decoder mostly avoids it). Split a filing into chunks, encode each chunk, then a small
  transformer over the chunk vectors pools them into the 64-d output. Sections longer than the
  chunk budget keep their first and most rate-related chunks (those mentioning rates, debt,
  floating, hedging).
- **Cost control.** Filings change quarterly, not daily: about 50,000 distinct documents, each reused
  for every meeting until the next filing. Fine-tune the top layers only (or LoRA) and cache the
  frozen lower layers' outputs once, so each walk-forward window retrains only the upper layers,
  pooling and heads.
- **Two-stage training, as in B1.** Pretraining: predict the firm's response to ordinary-day rate
  moves (2y yield) from its filings, thousands of days. Fine-tuning: FOMC days with s.
- **Control arm.** The same model with the encoder frozen (no fine-tuning). No improvement would
  mean no demonstrated benefit from this fine-tuning recipe on the tested sample, not that the
  pretrained model already knows everything useful. Compare both with the market-only model and
  the same-document TF-IDF control under the scoring rules below.
- **Timing.** Firm inputs must be available by the previous trading-day close. Same-day 8-Ks
  are never inputs; they flag rows for secondary sensitivity analyses, not primary deletion.
- **Fed statement text** is meeting-level (the same for every firm), so it cannot sit in the firm
  tower. It is a later extension: a meeting-level encoder whose output modulates the heads (FiLM),
  with only about 224 statements to learn from.
- **Attribution.** Track C can point at the chunks and sentences behind a flagged firm.

### B4. What the fundy and movers towers add (Astra proposal, 2026-10-19)

**Main question: do fundamentals explain when the historical market response becomes unreliable?**
Two firms could have similar return histories but different newly disclosed refinancing needs
or hedging arrangements. The hypothesis is that those disclosures change how to interpret the
history. This is a prediction hypothesis, not a causal claim about refinancing or hedging.

Here, *fundy* means the filings-text tower, and *movers* means the market-history tower. Start
with the existing patch transformer for movers. Replacing it with B1's event-history encoder is
a separate experiment, so new text and new history architectures are not introduced together.

#### First experiment: separate information from interaction

Use a common two-beta baseline and learn corrections to it:

`mu = mu_two_beta + delta_a + delta_b * s`.

Keep the baseline fit, projection policy, eligible firms, chronological folds, training budget,
and joint a/b objective common across comparisons. If retaining v4's projection, apply the same
projection off `[1, beta63, beta_smoothed]` to each arm's corrections. Projection removes linear
exposure to those columns; it does not establish a monetary-policy mechanism or remove nonlinear
beta proxies. Do not rerun b-only versus joint training as part of this comparison.

| Arm | Correction inputs and combination | Question answered |
|---|---|---|
| T0 | No correction | Does anything improve on the shared baseline? |
| TM | Movers tower only | What can the existing history representation contribute? |
| TF | Fundy tower only | Does filings text predict anything beyond the baseline? |
| TADD | Separate tower heads, summed: `g_M(h_M) + g_F(h_F)` | Do the two sources contribute independently? |
| TJOINT | Nonlinear head on `[h_M, h_F]` | Does interpreting the sources together help? |

Each head outputs `(delta_a, delta_b)`. In TADD the two heads may be trained jointly but never
receive the other tower's representation. Match head capacity approximately and give each arm
the same validation search budget; otherwise extra head capacity is an alternative explanation.
The key contrast is **TJOINT versus TADD**, alongside TADD versus TM and TF. Keep the frozen-text
and TF-IDF controls from B3 on the same documents, masks and evaluation rows.

An improvement over TM alone supports adding text; an improvement over TADD supports the tested
interaction architecture. Neither establishes why the model works. If TADD matches TJOINT,
prefer TADD. If text does not help, report that result before enlarging the text backbone.

#### Second experiment: does the latest disclosure matter?

After the first comparison, test the same selected architecture with the latest eligible filing
versus the preceding eligible filing. Restrict this paired diagnostic to firms with both
documents and report its coverage. Select documents entirely from information available at the
prediction cutoff. Predefine an additional slice where disclosures change in refinancing,
floating-rate debt or hedging content, using a fixed text rule rather than subsequent returns.

This tests whether updating the text improves prediction over retaining older information.
It does not isolate a particular economic mechanism: document age, changes in other content and
measurement quality remain alternative explanations. A later extension could encode the latest
filing together with its change from the preceding filing; that should earn its place against
the latest-filing-only control first.

#### Data and training details to settle before these experiments

- The current `fedcore.ingest.edgar` section extractor covers market-risk sections. B3 also
  promises liquidity text and earlier 8-K content; those extraction paths still need building
  and coverage checks. Start with the verified section set or implement the extra sections
  explicitly. Save accession, acceptance time, section type, extraction status and document age.
- Missing text is not evidence of no exposure. Preserve missingness and extraction failures;
  compare arms on the same firm-meetings and report results by text coverage.
- Thousands of ordinary-day labels reuse the same filings and dates. They provide training
  examples, not thousands of independent FOMC shocks. Pretraining, vocabulary fitting and
  supervised representation updates must respect each fold's training cutoff.
- Cache activations only below a genuinely frozen boundary. LoRA inside cached layers changes
  their outputs, so those activations cannot be reused as if the layers were frozen.
- Historical filing timestamps alone do not establish that a modern pretrained text backbone
  has no knowledge of later events. Record backbone version and known training-data coverage;
  describe this limitation of retrospective tests and retain the temporally fitted TF-IDF control.

### B3 first test: arms and criteria (written before any text run)

Code: `fedenc.text_tower` (tower, two-tower model), `fedenc.v5text` (token store, arm, runner).
Same base, beta-span projection, two-stage step-matched refit and walk-forward folds as v4
(161 development test meetings, 2001-2021), one seed, one member per fold.

| Arm | Market tower | Fundamentals tower | Question |
|---|---|---|---|
| T0 | patch transformer | none (every row "no filing") | the baseline the text must beat |
| T1 | patch transformer | EmbeddingGemma 2, top 4 layers fine-tuned, learned attention pooling | the user's design |
| T1m | patch transformer | same, mean pooling | embedding pooling vs learned pooling |
| T1f | patch transformer | EmbeddingGemma 2 frozen, learned attention pooling | does fine-tuning learn anything? |

Text input: the latest 10-K or 10-Q market-risk section (or its interest-rate and debt sentences
when the filing has no usable item) accepted before the meeting day, up to 1,024 tokens, rate
sentences first.

Scoring, conditional first (v4 conditional view), each arm against T0 and against B0:
1. Hit rate and realized-vs-predicted move on the top decile of departures from beta, meetings
   with |s| >= 2.5bp, with and without 2004-01-28 and 2011-09-21, meeting-clustered intervals.
2. Vulnerable firms (low cash, high leverage) and the distance-to-default bottom third: hit rate
   of departures over 0.1pt.
3. Firm-vs-firm squared error on |s| >= 2.5bp and >= 5bp; all-meeting MSE as context only.

Text "adds something" if T1 (or T1m) beats T0 on item 1 without the standout meetings, with the
interval excluding T0's rate, and does not lose on item 3. Anything less is reported as is.

## 5. Track C: probing, checked on synthetic data first

1. **Synthetic check.** Extend `fedcore.q3.synthetic` with planted channels:
   - a true channel (for example leverage x s, only when credit is tight);
   - a decoy correlated with beta but with no effect;
   - a channel whose strength drifts over time (the Amazon case);
   - a planted outlier: one firm in a twin cluster carrying an extra trait that changes its
     response.

   Pass criteria, fixed before the run:
   - the probe ranks the true channel above the decoy;
   - it tracks the drift;
   - B2 flags the planted outlier and names its trait.

   A probe that fails here is not used on real data.
2. **On real data.** For each flagged outlier: change one input at a time (CDS, leverage, response
   history, filing text) and record how the predicted response moves, plus gradient attribution.
   The output is a ranked list of explanations per outlier. Each is a hypothesis for Track A, not a
   finding.

## 6. Scoring

- Conditional response scores on the model's biggest predicted departures from both betas, on
  meetings with |s| >= 2.5bp and >= 5bp, with and without 2004-01-28 and 2011-09-21. Freeze the
  flag rule and threshold using training/validation data only. The realized surprise is permitted
  under Q3's conditional-response task; this does not make the flags pre-announcement forecasts.
- For B4 comparisons, use a common evaluation population, including common predefined slices;
  report each arm's own flagged population separately with coverage. Define directional success
  as agreement between the predicted correction and `actual return - common baseline`, not the
  sign of raw return. Report conditional squared error and calibration alongside hit rate, with
  equal meeting weights and meeting-bootstrap intervals. Fix one primary contrast and metric
  before training; the conditional scoring proposal changes the earlier Q3 selection rule and
  must be reconciled with `docs/question.md`, not silently replace it.
- Long-short spreads are secondary descriptive outcomes, not the Q3 model-selection criterion.
- Vulnerable-firm and outlier slices; earnings-day firms excluded as a secondary score.
- Track A statistics as above.
- Average MSE vs the two-beta CAPM as context, not as the criterion.
- Criteria are written and committed before any new run. 2022-2025 was already evaluated in
  earlier versions, so it is a reused temporal evaluation period, not an untouched confirmatory
  holdout. Keep it out of v5 tuning and describe any later check accordingly; 2026 stays closed.

## 7. Order and permissions

1. Now, no permission needed: wire the on-disk Compustat fields into the panel; Track A on on-disk
   channels; the synthetic extension and its tests.
2. After the user's OK: EDGAR 10-K and 10-Q market-risk sections, TNIC, FOMC texts.
3. After the WRDS subscription list: OptionMetrics, CDS, TRACE, IBES, 13F pulls.
4. After the user's go-ahead on this plan: B1 pretraining and fine-tuning, B2, Track C on real data
   (GPU on vast, with approval).
