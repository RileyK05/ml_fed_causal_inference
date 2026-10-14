# The pre-training debate

A record of the back-and-forth on how to pre-train the Q3 encoder, written up on 2026-10-12. Nothing in this
doc has been built or run. It's the argument so far: proposals, counters, concessions, and what is
still open. Each round is labelled with who raised it.

---

## 0. The setup everyone agrees on

The model predicts a firm's Fed-day return as

```
return = a(z) + b(z) × s
```

- **z**: the firm's state, built by the encoder from 252 days of trading history plus fundamentals and
  macro context.
- **s**: the Fed surprise. s = 1 means a 10bp hawkish surprise.
- **b**: the firm's sensitivity, in % move per 10bp of surprise. This is the thing Q3 is about.
- **a**: what the firm does on a Fed day regardless of the surprise.

Three facts drive the whole debate:

1. **There are only 253 Fed meetings.** Every firm in a meeting sees the same s, so for learning b the
   effective sample is closer to 253 than to the 127k firm-meeting rows.
2. **The signal is small next to the noise.** The median surprise is 1.4bp (s ≈ 0.14), and the
   90th percentile is 4.8bp (see [cheap-test-menu.md](cheap-test-menu.md)). With b near −1, b·s is
   a few tenths of a percent at most, against roughly 2.4pp of ordinary daily noise. *(An earlier
   draft said "typically 3–5bp", which was too high.)*
3. **So pre-training exists to give the encoder a head start.** It runs on a large pool of non-Fed data
   so that the 253 meetings only have to *adjust* the model, not build it from nothing.

The open question is **what that large pool should teach.**

---

## 1. Starting point: masked reconstruction (current build)

**What it is.** Hide about 30% of the week-patches in a 252-day window and train the encoder to fill them
back in. It's BERT-style and borrowed from PatchTST. This is what `encoder/fedenc/pretrain.py` does
today. The training plan (D3) recommended one checkpoint with a 2009-04-29 cutoff, about 620k real
windows.

**Problem raised (me).** On the synthetic data, masked pre-training mostly learned volatility and
market-beta state, not anything about rate sensitivity. Hand-built summary features also beat the
transformer trained from scratch. Filling in charts is not obviously the same skill as reading
sensitivity.

**Status.** Kept only as a baseline to compare against.

---

## 2. Side debate: is the encoder too small?

**User.** Three layers and a 64→256→64 feed-forward seems small.

**Me.** The bottleneck is data, not capacity: 253 shocks and low signal-to-noise. Fine-tuning a bigger
network would mostly fit noise, and on the fake data simpler already beat bigger. The one place size
could pay off is **pre-training**, which has plenty of data. Proposed test: pre-train at a few sizes
(3 vs 6 layers, width 64 vs 128, feed-forward 128 vs 256) and keep the bigger one only if held-out
reconstruction loss keeps improving.

**Status.** Agreed in principle. It's a pre-training-time check, not a design change.

---

## 3. Proposal: pre-train on ordinary rate-move days

**Me.** Use the same regression shape on ordinary days, with that day's 1-year yield move as the shock:

```
firm return on day t (vs market) = a(z) + b(z) × Δyield on day t      (z from history through t−1)
```

There are about 8,000 trading days with real rate moves, and the b head gets practice before ever seeing
a Fed meeting. The 1-year yield is already in the repo, and s is defined in 1-year-yield units, so the
units line up.

**User (key intuition).** Vol and beta fit "whatever the market is doing", because about 90% of the time
everything moves together. The model is being asked for an expected value while the informative 10% is
rare: *"you're trying to give it the E(X) when it only wants to weight P(X)."*

**Me.** Agreed, and restated mechanically. Squared-error loss is an average, so it's dominated by the
frequent common moves, and the rate-sensitive signal barely registers. Proposed fixes: a market-relative
target (`ret_rel`), and weighting the informative days.

### 3a. Why pre-training and not post-training?

**User.** Why is this *pre*-training and not post?

**Me.** You end on the task you're graded on. Training on 8,000 ordinary days *after* the Fed days would
overwrite what the 253 meetings taught (catastrophic forgetting). In finance terms, ordinary days give
the **prior** on b and Fed days give the **update**.

### 3b. Isn't this the mid-training from before?

**User.** How is this different from mid-training?

**Me.** It mostly replaces it. Mid-training on CPI and jobs days existed because masked reconstruction
never trained the b head. Rate-day pre-training trains b from the start, and the big rate days are
already mostly CPI and jobs days. A three-stage version (masked → rate → Fed) was floated as an
ablation, then dropped as a variant without a clear question behind it.

### 3c. Confusion: "two pre-trainings"

**User.** Do we train a, then b, then post-train? And what are the two pre-training things?

**Me.** No. a and b always train together, because they come from one error. There is **one**
pre-training. Masked reconstruction was the old one, mentioned only as something to compare against.
*(My fault: I presented baselines and ablations as if they were stages.)*

---

## 4. Proposal: only train on the top 20% of rate-move days

**Me.** On quiet days Δy ≈ 0, so a + b·0 = a and b learns nothing. Keep only the biggest-moving days.

**User (counter 1).** Then a becomes an under- or over-estimate to compensate for b.

**Me (counter).** Selecting on the *shock* (x), not the outcome (y), leaves the slope unbiased. a shifts
toward a "news-day average", but a symmetric cut (big up *and* big down days) keeps that out of b, and a
is re-learned on Fed days anyway.

**User (counter 2, the decisive one).** Training only on crazy days teaches the model that only crazy
things happen. What about a Fed day where nothing happens? It won't get it.

**Me (conceded).** Right, and the filter was never needed. Regression already weights informative days
automatically: a day's pull on the slope grows with Δy², so a 10bp day counts about 100× a 1bp day.
Quiet days barely move b, and they *do* teach a the "nothing happened" baseline. Cutting them threw
away a's best data to solve a problem the math already handles.

**Status.** All ordinary days, no filter. Before 2009 that's about 3,800 days × 500 firms ≈ 1.9M rows.

---

## 5. Proposal: stop a from shaping the encoder (λ / detach)

**User.** Should we add a λ to a and b, if a dominates because real shocks are rare? *(Section 3's
E(X)-vs-P(X) concern again.)*

**Me.** Yes: this is **gradient imbalance**, a named problem in multi-task learning. a and b read the
same z. On each row, a's push on the encoder is proportional to the error, and b's is proportional to
error × s. With s usually near zero, a gets almost all the say in what the encoder learns. Fix:

```python
a = head_a(z.detach())                       # a reads z but can't reshape the encoder
z_a = z.detach() + lam * (z - z.detach())    # or a dial: lam=1 today, lam=0 fully detached
```

Predictions are unchanged; only the backward pass differs. Stop-gradient is common (SimSiam, BYOL,
GANs, RL target networks).

**User.** So we're holding one arm still while training the other, one at a time?

**Me.** No, everything trains at once. Only b's feedback reaches the encoder. Analogy: two PMs both use
one analyst's report, but only the b-PM gives the analyst feedback.

**User.** Then the transformer is only trained on b?

**Me.** Yes, and that's the point. The trade-off is less total signal reaching the encoder.

**User (counter, the decisive one).** That's not the point. The goal is "what reaction do we expect from
this firm, given the shock", and that includes its baseline. Blocking a means the firm's base does
nothing. *"You don't train your regression model on houses where the data is an outlier."*

**Me (conceded).** Right. The deliverable is the full expected move, a + b·s, not b alone. Under that
goal, a getting most of the signal is accurate, not a hijack. Detach dropped from v1.

### 5a. My market-split proposal, and the process critique

**Me.** Split each return into the meeting average (fit with a simple α + β·s on 253 points) plus the
firm's deviation (the transformer, trained on returns with the meeting average removed). That gives
both an absolute and a relative answer.

**User (process critique).** I keep dropping plans because none of them do what's needed, and I'm
reaching for high-probability, off-the-shelf methods instead of working out how to train *this*
architecture.

**User (on the split).** It's more Q5 ("can we trade on it") than Q3.

**Status.** Parked as a Q5 idea. The critique stands: proposals were patching one symptom at a time
instead of starting from the objective.

---

## 6. User's proposal: a shock-dependent dial on a

**User.** On normal days, train normally. When rates move a lot, make a contribute less, so b is more
present on big shock days.

Two fears:
1. The model learns normal days are the norm, so a Fed day looks like any other day, and then "bam",
   it can't adapt.
2. Or the model learns that tiny 1-year moves are massively important.

And a framing point: b is the firm's *current state* (vol, fundamentals, …), which is fundamentally a
different kind of thing from a, what the firm normally does.

**Me.** Coherent, and different from detach. Normal days teach the encoder the firm's baseline; big days
teach it reaction. A *smooth* dial (λ falling with |Δy|) avoids the cliff of a hard cutoff. On fear 1:
fine-tuning on Fed days comes last and teaches "Fed days are their own thing". Also, Fed surprises
may be about the size of an ordinary day's move, so Fed days may differ in *kind* (policy vs growth
news) rather than size. **Checked since:** they're actually slightly *smaller*. The median |surprise|
is 1.4bp vs 1.6bp for an ordinary day's 1-year move, and the 90th percentiles are 4.8bp vs 6.1bp. So
the "bam" fear is not about size. On fear 2: low risk, because tiny
moves have little pull on a slope.

**User (follow-up).** But a still counts on fundamental days.

**Me (wrong turn, then corrected).** I said detach "activates on big days by itself". Imprecise: detach
is always on. The size-dependence comes from b's feedback being error × s, which shrinks on quiet days.

**Status.** Superseded by section 7, but the dial is still a valid option.

---

## 7. User's proposal: add a vol response to the output

**User.** We've never added vol. Something like a normal response plus a vol response, on top of
a + b·s.

**Me.** Correct: vol is an *input* (`rvol21`) but never an *output*. Squared-error loss assumes every
row is equally noisy, which is false. Proposed form:

```
return = a(z) + b(z)·s + noise,     noise size = v0(z) + v1(z)·|s|
```

The head would have four outputs: a (normal response), b (direction response), v0 (normal vol) and v1
(vol response to surprise size). Benefits:
- Noisy rows get trusted less automatically. This is weighted least squares, with no hand-set dial.
- v1 is a finding in itself: which firms get more uncertain when the Fed surprises.

Catches:
- Rows are weighted by signal vs noise, not by shock size alone. If big-shock days are also very noisy,
  they get trusted *less*, the opposite of the user's instinct, though statistically correct.
- The model can "give up" on hard rows by calling them noisy. β-NLL is the standard fix.

**Status.** Recommended. It would replace the λ dial. Not decided.

---

## 8. Overswing and overweighting

**User.** If a is held constant on surprise days, b grows and grows, and shock days become overswings.

**Me.** It would only happen if a's own weights stopped learning, which no proposal does. Even then, b
can only soak up the part of a that lines up with the *direction* of s. An effect that is the same for
hawkish and dovish days can't be absorbed, unless the sample is lopsided (e.g. mostly hawkish, as in
2022). Rule: never freeze a's own weights.

**User.** So we're not worried about overweighting shocks?

**Me.** We are, just about something narrower: **a few extreme meetings dominating b.** Pull grows
with s², so a 30bp meeting pulls about 100× harder than a 3bp one, and with 253 meetings two or three
monsters can set b. Options:
- check influence first (the DML project already computes influence tables);
- a Huber loss;
- a robustness run without the ~5 biggest meetings;
- capping s (not recommended).

There's also a separate, real risk that holds in any version: linear extrapolation overshoots if
responses saturate on huge surprises.

**Status.** Influence check first; Huber is a candidate for the main spec; the no-extremes run as
robustness.

---

## 9. Literature pass

Options surveyed, mapped to the test "does it teach sensitivity-relevant state without predicting pure
noise?":

| # | Task | Notes |
|---|---|---|
| 1 | Predict the firm's **future 3-month betas** (market, rates, SMB, HML) and future vol | Low-noise targets, sensitivity-shaped, data in hand. My top pick |
| 2 | Conditional factor model every day: return = betas(z) × factor moves (Gu, Kelly & Xiu style) | a + b·s generalised to ~4 shocks; noisier than #1 |
| 3 | Earnings-day reaction size, from Compustat report dates | The only option trained on real *event days*; matches the vol-response idea |
| 4 | Rate-day regression (section 3) | Superseded by #1 and #2 |
| 5 | JEPA (predict latent summaries of hidden weeks) | Noise-robust in principle; more engineering |
| 6 | Co-movement contrastive learning | Teaches "which firms behave alike"; indirect |
| 7 | Synthetic-world pre-training (TabPFN-style) | Creative, but fake data already misled us once |
| 8 | Off-the-shelf foundation model (Chronos, MOMENT, Kronos) | Benchmark only; finance results are mixed |
| 9 | Masked reconstruction (current) | Baseline |

**User (counter on #5).** JEPA is interesting but hard, and probably not useful.

**User (counter on #1).** It might change the thing we're predicting entirely.

**Me (response, not yet discussed).** This is the right worry for any pre-training target that isn't
the final target. For: only the encoder carries over, and the pre-training head is thrown away, so the
fine-tuning target is unchanged. Against: if the encoder specialises in "describe future betas", its
description of the firm may not contain what Fed-day *a* needs. That is exactly the user's section 5
objection again, that the firm's baseline gets lost. The frozen vs top vs full comparison partly
measures this: if frozen loses badly to full, the pre-trained description was missing something.
Mitigation if needed: combine #1 with a task that keeps the baseline in, such as the rate-day
regression with its a term.

---

## 10. Where it stands

**Agreed:**
- One pre-training stage, then fine-tuning on Fed days. Pre-training comes first because it's the
  prior, and the graded task comes last.
- Use all ordinary days. No top-X% filter.
- a's own weights always learn. No detach in v1, because the deliverable is the full expected move.
- Cutoff before the first test fold (2009-04-29), so one checkpoint is honest for every fold.
- Masked reconstruction stays as a comparison baseline, not a stage.
- JEPA parked.

**Open:**
1. **The pre-training target itself.** Rate-day regression (keeps the a + b·s shape and the baseline),
   future betas (low noise, but may shift what the encoder describes), or a combination.
2. **Vol head** (v0, v1) on the output, with β-NLL.
3. **Shock-dependent λ dial on a.** Probably redundant if the vol head goes in.
4. **Handling extreme meetings.** Influence check, then Huber or plain MSE.
5. **Data checks**, seconds each, no training:
   - Fed surprise sizes vs ordinary-day 1-year yield moves.
   - How many meetings dominate the slope.
6. **Encoder size for pre-training** (section 2).

**Lesson from the process.** Proposals went wrong when they fixed one symptom (noise, a dominating,
quiet days) without checking against the actual deliverable: **the full expected move of each firm on
a Fed day, given the surprise.** Every candidate should be tested against that sentence first.

---

## Sources from the literature pass

- [Autoencoder Asset Pricing Models (Gu, Kelly, Xiu, 2021)](https://papers.ssrn.com/abstract=3335536)
- [Deep Learning in Asset Pricing](https://arxiv.org/pdf/1904.00745)
- [Contrastive Learning of Asset Embeddings](https://arxiv.org/html/2407.18645v1)
- [Trading through Earnings Seasons using Self-Supervised Contrastive Representation Learning](https://arxiv.org/pdf/2409.17392)
- [Cross-Sectional Asset Retrieval via Future-Aligned Soft Contrastive Learning](https://arxiv.org/pdf/2602.10711)
- [Joint Embeddings Go Temporal (TS-JEPA)](https://arxiv.org/html/2509.25449v1)
- [TabPFN (prior-fitted networks)](https://arxiv.org/pdf/2207.01848v4)
- [Re(Visiting) Time Series Foundation Models in Finance](https://arxiv.org/pdf/2511.18578)
- [Kronos: A Foundation Model for the Language of Financial Markets](https://arxiv.org/pdf/2508.02739)
- [Pretrained Time-Series Foundation Models for Financial Return Forecasting](https://arxiv.org/pdf/2606.27100)
- [A Survey of Financial AI](https://arxiv.org/pdf/2411.12747)
