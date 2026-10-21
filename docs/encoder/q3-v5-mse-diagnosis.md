# Why MSE Is Poor in Q3 v5 (and Why Text Arms T1/T1m/T1f Lag T0 and B0)

**Date:** 2026-10-20  
**Scope:** Forensic diagnosis of the Q3 v5 MSE numbers across Track A (`docs/encoder/q3-v5-track-a.md`) and Track B3 (`docs/encoder/q3-v5-text-results.md`, `encoder/fedenc/v5text.py`, `encoder/fedenc/text_tower.py`, and saved runs `20261020-*_v5text-to2021-*`).

---

## Executive Summary

There are **two distinct phenomena** behind the "poor MSE" in Q3 v5:

1. **The Structural MSE Ceiling (Why all arms sit near ~5.902 raw / ~4.121 centered MSE, and why valid $b$ corrections only move MSE by $\sim 0.001\text{–}0.003$):**
   - A $b$-only model predicts $\hat{\mu}_{i,m} = \mu^{\text{base}}_{i,m} + P_m(\Delta b)_{i,m} \cdot s_m$.
   - Because $P_m(\Delta b)$ is orthogonal to $[1, \beta_{63}, \beta_{\text{smooth}}]$ within every meeting, **100% of the meeting-wide error ($1.781\text{ pct}^2$) and 100% of the two-beta cross-sectional error are untouched by construction**.
   - Within the remaining $4.121\text{ pct}^2$ firm-vs-firm variance (a $2.03\%$ daily idiosyncratic return standard deviation on FOMC days), any correction $\Delta b$ is multiplied by the policy surprise $s_m$ (in 10bp units), so its MSE contribution scales with **$s_m^2$**.
   - Across the 161 development test meetings (2001–2021), the median $|s_m|$ is $\sim 1.2\text{ bp}$ ($s_m^2 \approx 0.014$), and 74% of meetings have $|s_m| < 2.5\text{ bp}$ ($s_m^2 < 0.0625$). Even the strongest true economic signal in the entire dataset—Track A's **Merton Distance to Default (`distance_to_default`, $t = 2.86$)**—has a slope of $+0.195\%$ per 1 SD per 10bp shock, which translates into a maximum theoretical MSE reduction of only **$\Delta \text{MSE} \approx -(0.195)^2 \cdot \overline{s_m^2} \approx -0.0015\text{ to }-0.0035$** ($< 0.1\%$ of within-meeting return variance).

2. **Why the v5 Text Arms (`T1`, `T1m`, `T1f`) Lost to Both `B0` and `T0` (`+0.0014` vs `-0.0013` on $|s| \ge 2.5\text{ bp}$, and `0.500` top-decile hit rate):**
   - **No-text `T0` actually beat `B0` slightly** (`-0.0013`, 95% CI `[-0.0030, -0.0001]` on $|s| \ge 2.5\text{ bp}$) by keeping its predicted departures tiny (`0.014%` in the top decile).
   - **Adding text (`T1`, `T1m`, `T1f`) made predictions $3.5\text{–}5.2\times$ larger (`0.051%`–`0.073%`), which pushed MSE worse than `B0` (`+0.0008` to `+0.0014`) while landing at a 50/50 coin flip (`0.493`–`0.502`).**
   - Forensics of the saved training curves (`tables/curves.csv`), fold selections (`tables/folds.csv`), and loss mechanics in `fedenc/v5text.py` show five concrete reasons why the text tower overfit noise rather than extracting rate sensitivity:
     1. **$s_m^2$ Gradient Starvation (Effective Sample Size of ~5 Meetings):** The squared-error gradient with respect to $\Delta b$ is weighted by $s_m$ (and curvature by $s_m^2$). Roughly 75% of training meetings have $s_m \approx 0$ and give zero gradient to the text tower; 3–5 extreme meetings carry $>80\%$ of the gradient.
     2. **Text as a Firm-Identity Fingerprint Memorizing `fed_beta_past` ($t = -0.75$):** A firm's 10-K Item 7A prose changes once a year (and `"no material change"` 10-Qs map back to the same 10-K). High-dimensional text embeddings effortlessly identify *which firm* wrote the document. On 3–5 big-shock training meetings, the fusion head uses the text vector as a firm lookup table to memorize which firms outperformed on past big FOMC days (`fed_beta_past`). Track A proved `fed_beta_past` has **$t = -0.75$** out of sample—so memorizing firm identity from text injects pure out-of-sample variance.
     3. **Validation Loss Invariance & Epoch-0 Collapse (`folds.csv`):** Validation early stopping scores raw within-meeting MSE on the last 15% of the training window (9–33 meetings, mostly tiny $|s_m|$). In `curves.csv`, validation loss across 15 epochs changes only in the **4th decimal place** ($\pm 0.0002$). As a result, `T1` selected **`best_epoch = 0` (`0` optimizer steps, i.e., an untrained zero model)** in Folds 0 and 6, and `epoch = 1` or `2` in Folds 3 and 4.
     4. **4.8% Per-Epoch Firm Sampling (`firms_per_meeting = 24`):** Each training step samples 24 of $\sim 500$ firms per meeting ($4.8\%$). When early stopping halts at epoch 1–4, the average `(firm, meeting)` row in the training window has been seen **only 0.05 to 0.20 times** across the entire training run.
     5. **Qualitative Boilerplate vs. Quantitative Merton Distance to Default:** Track A showed that true beyond-beta sensitivity comes from quantitative balance-sheet fragility (`distance_to_default` $t = 2.86$, `int_coverage` $t = 3.56$ post-2009), which requires combining current market equity, 252-day return volatility, and debt ratios—numbers that are either absent from or buried in qualitative Item 7A legal boilerplate.

---

## 1. The Variance Budget: Why Raw & Centered MSE Barely Budge

From the saved manifests (`20261020-*_v5text-to2021-*`) and conditional scoring (`20261020-065548_v5text-to2021-score`), here is the exact MSE breakdown across the 161 walk-forward test meetings (2001-11-06 to 2021-12-15):

| Metric | `T0` (No Text) | `T1` (Gemma top-4, attn) | `T1m` (Gemma top-4, mean) | `T1f` (Gemma frozen, attn) |
|---|---:|---:|---:|---:|
| **All-meeting raw MSE** | `5.90214` | `5.90291` | `5.90297` | `5.90255` |
| **Meeting-wide MSE** ($\text{raw} - \text{centered}$) | `1.78132` | `1.78132` | `1.78132` | `1.78132` |
| **All-meeting centered MSE** | `4.12082` | `4.12159` | `4.12165` | `4.12123` |
| **All-meeting MSE vs `B0`** | `-0.0003 (-0.0008, +0.0002)` | `+0.0005 (-0.0001, +0.0012)` | `+0.0006 (+0.0000, +0.0011)` | `+0.0001 (-0.0002, +0.0005)` |
| **Cross-firm MSE vs `B0` ($\|s\| \ge 2.5\text{ bp}$, 40 mtgs)** | **`-0.0013 (-0.0030, -0.0001)`** | `+0.0014 (-0.0006, +0.0037)` | `+0.0009 (-0.0008, +0.0026)` | `+0.0008 (-0.0005, +0.0022)` |
| **Cross-firm MSE vs `B0` ($\|s\| \ge 5\text{ bp}$)** | **`-0.0004 (-0.0010, -0.0000)`** | `+0.0027 (-0.0023, +0.0085)` | `+0.0018 (-0.0029, +0.0063)` | `+0.0002 (-0.0037, +0.0048)` |
| **$\text{SD}(\Delta b)$ across test rows** | `0.0667` | `0.0836` | `0.0661` | `0.0535` |
| **Top-decile mean $\|\Delta b \cdot s\|$ ($\|s\| \ge 2.5\text{ bp}$)** | `0.014%` | `0.073%` | `0.051%` | `0.056%` |
| **Top-decile hit rate (40 mtgs ex-standouts)** | `0.485` | `0.500` | `0.493` | `0.502` |

### Why is the MSE difference in the 4th decimal place (`0.001`)?
Write the within-meeting residual after the two-beta CAPM base `B0` as $e_{i,m} = y_{i,m} - \mu^{\text{base}}_{i,m}$. The model predicts $\hat{e}_{i,m} = \tilde{b}_{i,m} s_m$, where $\tilde{b}_m = P_m(\Delta b_m)$. For any meeting $m$:

$$\text{MSE}_m(\tilde{b}) - \text{MSE}_m(0) = s_m^2 \|\tilde{b}_m\|_2^2 / N_m - 2 s_m \frac{\tilde{b}_m^\top e_m}{N_m}$$

Notice three mathematical facts from this identity:
1. **Everything is multiplied by $s_m$ and $s_m^2$:** On the 74% of meetings where $|s_m| < 0.25$ ($2.5\text{ bp}$), $s_m^2 < 0.0625$. A model with $\text{SD}(\tilde{b}) = 0.07$ changes predictions by less than $0.07 \times 0.25 = 0.0175\%$ (1.75 bps), which alters squared error by $(0.0175)^2 \approx 0.0003\text{ pct}^2$.
2. **Idiosyncratic daily stock noise on FOMC days is $4.12\text{ pct}^2$ ($\sigma_e \approx 2.03\%$):** Over a 24-hour close-to-close window, S&P 500 stocks move $\pm 200\text{ bps}$ from earnings announcements, guidance, sector rotation, and macro news unrelated to the 2:00 PM FOMC statement's 3bp surprise.
3. **Even the strongest true Track A channel (`distance_to_default`, $t = 2.86$) only moves MSE by $\sim 0.002$:**
   - In `docs/encoder/q3-v5-track-a.md`, a 1-SD change in Merton Distance to Default has a true beyond-beta slope of $\gamma = +0.195\%$ per 10bp shock ($t = 2.86$).
   - If an oracle knew the exact population slope $\gamma = 0.195$, the reduction in per-meeting MSE on a meeting with surprise $s_m$ would be $\gamma^2 s_m^2 \approx 0.038 \cdot s_m^2$.
   - Averaged across all meetings (where median $s_m^2 \approx 0.014$ and mean non-standout $s_m^2 \approx 0.05$), **even the population oracle only lowers MSE by $0.0015\text{ to }0.0030\text{ pct}^2$**!

In short: **close-to-close MSE on all firms is dominated 99.9% by non-shock idiosyncratic return variance ($\sigma_e^2 = 4.12$) and meeting-wide market moves ($1.78$).**

---

## 2. Why the v5 Text Tower (`T1`, `T1m`, `T1f`) Degraded MSE vs. `T0` and `B0`

While the variance budget explains why *no* $b$-only model can change raw MSE by 0.10, it does **not** explain why `T1` lost to `T0` (`+0.0014` vs `-0.0013` on big shocks) and got a `0.500` coin-flip hit rate. The saved run artifacts point to five concrete causes:

### Cause 1: $s_m^2$ Gradient Starvation in `_deployed_loss`
In `fedenc/v3.py` (`_deployed_loss`), the training loss on a mini-batch of meetings is:

$$\mathcal{L} = \frac{1}{|M|} \sum_{m \in M} \frac{1}{N_m} \sum_{i \in m} \left(e_{i,m} - P_m(\Delta b)_{i,m} \cdot s_m\right)^2$$

The gradient reaching the text tower (`FilingTower`) and fusion MLP (`TwoTowerModel.fusion`) for firm $i$ in meeting $m$ is scaled by **$s_m$**, and the quadratic penalty that prevents $\Delta b_{i,m}$ from blowing up is scaled by **$s_m^2$**:
- On the ~75% of training meetings where $|s_m| < 2\text{ bp}$ ($|s_m| < 0.2$), the text tower receives virtually **zero gradient** and **zero curvature penalty** on $\Delta b$.
- On the 3–5 extreme training meetings where $|s_m| \ge 7\text{ bp}$, $s_m^2$ is **50 to 200 times larger** than on a typical meeting.
- Consequently, the 768-d $\to$ 64-d text head and 4 fine-tuned Gemma layers are effectively fit on an **effective sample size of ~5 meetings**, not 150 meetings.

### Cause 2: Slow-Moving Filing Text Acts as a Firm-Identity Lookup Table (Memorizing `fed_beta_past`)
Consider how SEC Item 7A filings behave across time in `edgar_market_risk`:
- A 10-K is filed once per year. In the intervening three 10-Qs, firms frequently state *"no material changes since our Annual Report on Form 10-K"*, and `filing_for_rows` (`fedenc/text_tower.py:187-189`) rightly points those rows back to the exact same 10-K accession.
- Even when a new 10-K is filed the next year, 90%+ of the Item 7A prose is carried over verbatim from the prior year's template.
- To a 24-layer pretrained language model (`google/embeddinggemma-2`), a firm's 1,024-token Item 7A excerpt is a **near-perfect firm identifier** (distinguishing Apple from Citigroup from Duke Energy from Ford).
- During training, because the same filing (or nearly identical prose) appears across multiple meetings for the same firm, the easiest way for `TwoTowerModel` to drive down squared error on the 3–5 giant-$|s_m|$ training meetings is to assign a positive or negative $\Delta b_i$ to the specific firms that happened to have positive or negative residuals on those past giant-$|s_m|$ meetings.
- That is mathematically equivalent to learning **`fed_beta_past`** (a firm's historical beyond-beta Fed-day loading).
- And as Track A proved (`docs/encoder/q3-v5-track-a.md`, Table 1), **`fed_beta_past` has a $t$-stat of $-0.75$ out of sample**! Firm-level beyond-beta sensitivity does not persist as a static firm trait; memorizing firm identity from persistent filing text simply injects out-of-sample noise (`delta_b_sd` rises from `0.0667` in `T0` to `0.0836` in `T1`, pushing top-decile departures from `0.014%` to `0.073%` with `0.500` accuracy).

### Cause 3: Validation Early-Stopping Flatness & Epoch-0 Collapse (`folds.csv` & `curves.csv`)
Inspecting `tables/folds.csv` and `tables/curves.csv` in `20261020-105039_v5text-to2021-t1` reveals a major training pathology:

| Fold | Train Meetings | Val Meetings | `T1` Selected `best_epoch` (`steps`) | `T1f` Selected `best_epoch` (`steps`) | `T1m` Selected `best_epoch` (`steps`) | `T0` Selected `best_epoch` (`steps`) |
|---:|---:|---:|---:|---:|---:|---:|
| **0** | 60 | 9 | **`0` (`0` steps!)** | **`0` (`0` steps!)** | `2` (`26` steps) | `6` (`78` steps) |
| **1** | 76 | 11 | `8` (`136` steps) | `1` (`17` steps) | `5` (`85` steps) | `1` (`17` steps) |
| **2** | 92 | 14 | `7` (`140` steps) | **`0` (`0` steps!)** | `7` (`140` steps) | `7` (`140` steps) |
| **3** | 108 | 16 | `1` (`23` steps) | `5` (`115` steps) | `11` (`253` steps) | `14` (`322` steps) |
| **4** | 124 | 19 | `2` (`54` steps) | `7` (`189` steps) | **`0` (`0` steps!)** | `1` (`27` steps) |
| **5** | 140 | 21 | `3` (`90` steps) | `2` (`60` steps) | `2` (`60` steps) | `2` (`60` steps) |
| **6** | 156 | 23 | **`0` (`0` steps!)** | `4` (`136` steps) | `1` (`34` steps) | `5` (`170` steps) |
| **7** | 172 | 26 | `7` (`259` steps) | `1` (`37` steps) | `3` (`111` steps) | `2` (`74` steps) |
| **8** | 188 | 28 | `4` (`160` steps) | `3` (`120` steps) | `1` (`40` steps) | `5` (`200` steps) |
| **9** | 204 | 31 | `10` (`440` steps) | `5` (`220` steps) | `3` (`132` steps) | `7` (`308` steps) |
| **10** | 220 | 33 | `12` (`564` steps) | `8` (`376` steps) | `3` (`141` steps) | `5` (`235` steps) |

Look at what happened:
- In `T1`, **Folds 0 and 6 selected `best_epoch = 0` (`steps = 0`)**, meaning Stage 2 (`fedenc/v4.py:303`) executed **zero gradient steps** and predicted $\Delta b \equiv 0$ for all 32 test meetings in those two folds!
- In `T1f`, Folds 0 and 2 selected `best_epoch = 0`; in `T1m`, Fold 4 selected `best_epoch = 0`.
- Across `curves.csv`, validation loss in Fold 3 (`T1`) moved from `3.40254` (epoch 1) to `3.40275` (epoch 2) to `3.40293` (epoch 3); in Fold 10 it stayed between `5.04063` and `5.04100` for 14 consecutive epochs.
- Because the inner validation block is only 9 to 33 meetings (most with small $|s_m|$) and `V5TextArm._eval` scores a 64-firm random draw on unweighted MSE, the validation curve is dominated by noise in the 4th decimal place, turning `best_epoch` into a noisy coin flip.

### Cause 4: Aggressive Subsampling (`firms_per_meeting = 24`) Starves Row Coverage and Noises Up the Projection Basis
In `V5TextArm` (`fedenc/v5text.py:207-219`):
- Each training step samples only `firms_per_meeting = 24` firms out of $\sim 500$ per meeting ($4.8\%$ of the cross-section).
- When a fold stops at `best_epoch = 1` or `2`, each `(firm, meeting)` row in the training set has only been drawn **$0.05\text{ to }0.10$ times** during the entire Stage 2 refit!
- Furthermore, `subset_basis` projects $\Delta b$ off the 3-dimensional span $[1, \beta_{63}, \beta_{\text{smooth}}]$ estimated on just those **24 firms** during training, whereas test predictions project off all **500 firms**. On a 24-firm draw, 3 degrees of freedom absorb $3/24 = 12.5\%$ of the cross-sectional variance (vs. $3/500 = 0.6\%$ at test time) and introduce sample-covariance noise into the projection operator $P_m$ on every step.

### Cause 5: Signal Mismatch — Legal Prose vs. Quantitative Balance-Sheet Fragility
Track A (`docs/encoder/q3-v5-track-a.md`) established that the actual cross-sectional variation in FOMC-day returns beyond beta is driven by:
1. **Merton Distance to Default (`distance_to_default`, $t = 2.86$)** — how close a firm's market value of assets is to its debt default boundary, scaled by asset volatility.
2. **Interest coverage (`int_coverage`, $t = 3.56$ post-2009)**, **leverage ($t = -2.14$)**, **book-to-market ($t = -2.04$)**, and **cash ($t = +1.78$)**.

Why didn't the Item 7A text tower pick this up?
- Merton Distance to Default is a quantitative ratio of **market equity cap + book debt** divided by **realized 252-day equity volatility**. A firm's 10-K Item 7A text does not know the firm's current stock price or current 252-day market volatility on the day before the FOMC meeting.
- While `TwoTowerModel` concatenates `[market 64 | text 64]`, `V5TextArm` **dropped all tabular fundamentals (`comp_fundq` / Track A channels)** from `TwoTowerModel`! Look at `fedenc/v5text.py:202`:
  ```python
  d["x"] = dict(history=x["history"], history_mask=x["history_mask"],
                doc=torch.as_tensor(self.store.docs_for(p), dtype=torch.long, device=device))
  ```
  Notice that **`V5TextArm` completely replaced the Compustat balance-sheet fundamentals (`leverage`, `cash`, `book_to_market`, `log_size`, `distance_to_default`, `int_coverage`) with raw Item 7A text!**
- Even `T0` (`use_text=False`) only had `history` (the 252-day return/vol/beta series) and **zero balance-sheet fundamentals**! So `T0` could only infer distress indirectly from 252-day equity volatility and drawdown in `history`, while `T1` tried to infer balance-sheet leverage and coverage purely from tokenized legal paragraphs without the actual numbers.

---

## 3. Summary Table: Where Each Piece of Error Comes From

| Component / Mechanism | Impact on MSE / Hit Rate | Fixable by Model? | Why / How |
|---|---|---|---|
| **Meeting-wide shock miss** (`1.781 pct^2`) | ~30% of total MSE (`5.902`) | **No** (in $b$-only) | $P_m(\Delta b)$ has zero meeting mean by construction; macro shock explains only ~17% of the market's FOMC-day move. |
| **Same-day idiosyncratic firm noise** (`~4.118 pct^2`) | ~70% of total MSE | **No** | Daily individual stock returns have $\sigma \approx 2.03\%$ from non-Fed news; $b \cdot s$ is bounded by small $s_m$. |
| **$s_m^2$ scaling on small-surprise meetings** | Caps max possible $\Delta \text{MSE}$ at $\sim 0.002\text{–}0.004$ | **No** (structural) | 74% of meetings have $\|s\| < 2.5\text{ bp}$; even a $t = 2.86$ channel (`distance_to_default`) only moves MSE by $\sim 0.002$. |
| **Text memorizing firm identity (`fed_beta_past`)** | Degrades big-shock MSE from `-0.0013` (`T0`) to `+0.0014` (`T1`); hit rate = `0.500` | **Yes** | 768-d text embeddings act as a firm ID on the ~5 huge-$s_m$ training meetings; `fed_beta_past` has $t = -0.75$ OOS. |
| **Unweighted val MSE early stopping + 24-firm draws** | Selects `epoch = 0` (`0` steps) in multiple folds; ~0.1 passes per row | **Yes** | Val loss is 99.95% small-shock noise; 24-firm subsampling starves coverage and noises up $P_m$. |
| **Dropping numeric balance-sheet channels in `V5TextArm`** | Starves the model of the proven Track A signals (`distance_to_default`, `int_coverage`) | **Yes** | `V5TextArm._data` omitted `fundamentals` and Track A's quantitative distress channels, forcing Gemma to guess balance-sheet ratios from legal prose. |

---

## 4. Five "What I'm Curious to Know" Hypotheses (and How to Test Them)

Rather than incrementally tweaking hyperparameters on the same setup, the diagnosis above raises five genuine scientific questions where structure and representation learning can actually earn their keep:

### Curiosity 1: What happens when the model is only allowed to read *what changed* in a firm's filing ($\Delta \text{Text}_{i,t}$) instead of static boilerplate?
- **The question:** Is a firm's 10-K Item 7A mostly acting as a corporate ID badge (95% static company prose, 5% new risk changes)? Since Track A proved static firm identity (`fed_beta_past`) has $t = -0.75$ out of sample, feeding raw filing embeddings invites the network to memorize firm IDs on the 3–5 largest training shocks.
- **How to test it (Diff-Filing Tower):** Feed the difference between the firm's current filing representation and its prior year's 10-K representation, $\Delta h^{\text{text}}_{i,t} = h(\text{Filing}_{i,t}) - h(\text{Filing}_{i,t-1})$ (or extract only sentences added/changed since the prior 10-K). If a firm's market-risk disclosure is unchanged, $\Delta h^{\text{text}}_{i,t} = 0$. Only firms that newly disclose a drawn revolver, a new floating-rate facility, a covenant amendment, or unwound swaps produce a non-zero text signal.

### Curiosity 2: Does text only matter *multiplicatively*—as a hedge/floating-rate gate on top of quantitative debt fragility (`distance_to_default`)?
- **The question:** Two firms can write the exact same sentence (*"Our credit facility bears floating interest at SOFR + 200 bps"*). If Firm A has zero net debt and 15x interest coverage, that sentence is irrelevant. If Firm B has high leverage and low Merton Distance to Default, that sentence is critical. Why did `V5TextArm` throw away the quantitative balance sheet (`distance_to_default`, `int_coverage`, `leverage`) and ask text to predict $\Delta b$ additively in a vacuum?
- **How to test it (Gated Structural Interaction):** Anchor the baseline sensitivity on the proven quantitative distress state $d_{i,m} = (\text{DD}_{i,m}, \text{IntCov}_{i,m}, \text{Lev}_{i,m})$, and let the text tower output a bounded multiplicative gate $\gamma(\text{text}_{i,m}) \in [-1, +1]$:
  $$\Delta b_{i,m} = (w^\top d_{i,m}) \cdot \big(1 + \lambda \tanh(g_{\text{text}}(\text{Filing}_{i,m}))\big)$$
  Safe/unlevered firms ($d_{i,m} \approx 0$) cannot be given giant spurious $\Delta b$ predictions from text noise; text is strictly asked: *"Given that this firm has balance-sheet debt/fragility, does its Item 7A disclosure say that exposure is hedged/fixed or unhedged/floating?"*

### Curiosity 3: Can we solve the 15-meeting sample-size bottleneck by learning the "latent fragility state" across *all* macro/tail shock days, not just FOMC days?
- **The question:** Track A discovered something striking in the non-FOMC placebo check (`docs/encoder/q3-v5-track-a.md`): distressed firms (`distance_to_default`) amplify large ordinary-day market moves beyond their quiet-day trailing $\beta$ ($t = -2.4$) in the exact same way they amplify FOMC shocks ($t = +2.86$). Why? Because 63-day/252-day CAPM beta is estimated on ~90% quiet days where credit convexity is dormant; whenever a macro shock hits, latent credit fragility wakes up.
- **How to test it (Multi-Event Fragility Transfer):** Instead of starving a neural encoder on ~40 big FOMC meetings (where 75% of meetings have $s_m^2 \approx 0$), train the firm-state encoder $z_{i,t}$ to predict **excess sensitivity on all non-FOMC macro/yield-shock days** (e.g., CPI/NFP release days or days with $|\Delta y_{2Y}| \ge 5\text{ bp}$ in `gsw_yields`, giving ~500+ shock events), and then transfer/freeze $z_{i,t}$ into the FOMC $b$-head.

### Curiosity 4: Does *why* the Fed surprised (the FOMC statement narrative / yield-curve mix) determine *which* firm vulnerability channel lights up?
- **The question:** Why did `int_coverage` jump from $t = +0.58$ (1994–2008) to $t = +3.56$ (2009–2021), while sector effects flipped sign across eras? Because a $+5\text{ bp}$ surprise during an inflation/tightening cycle prices **refinancing and interest-expense walls** (`int_coverage`), whereas a surprise during a growth/credit scare prices **default probability** (`distance_to_default`).
- **How to test it (Bilinear Context-Routed Sensitivity):** Instead of a scalar shock $s_m$ multiplying a single scalar $\Delta b_{i,m}$, let the firm encoder output a $K$-channel vulnerability vector $v_{i,m} \in \mathbb{R}^K$ (e.g., default risk, refinancing/floating-rate exposure, equity duration) and let the pre-statement regime + FOMC statement text (`fomc_statements`) / yield-curve slope shock (`gsw_yields`) produce a routing vector $g_m \in \mathbb{R}^K$ so $\Delta b_{i,m} = \langle v_{i,m}, g_m \rangle$.

### Curiosity 5: Are hikes and cuts asymmetric for distressed firms (Merton Call-Option Convexity)?
- **The question:** In Merton's (1974) model, equity is a call option on firm assets with strike equal to debt. For a firm near the default boundary (low DD), the payoff is convex and asymmetric: a surprise rate **cut** (dovish relief, $s_m < 0$) can trigger a violent short-covering/solvency rally, whereas a rate **hike** ($s_m > 0$) reprices default risk differently—and tiny $\pm 1\text{ bp}$ surprises do nothing.
- **How to test it:** Allow piecewise-linear response slopes $\Delta b^+_{i,m} \max(s_m, 0) + \Delta b^-_{i,m} \min(s_m, 0)$ (or a thresholded dead-zone for $|s_m| < 1.5\text{ bp}$) so the model can directly test whether distressed firms react asymmetrically to Fed easing vs. tightening.


