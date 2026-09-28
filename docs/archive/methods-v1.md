> Historical catalog, superseded by ../methods.md and Q-spec v1.1. Feasibility labels and methodological claims below were not validated and are not the active plan.

# Methods Catalog — FOMC Causal Inference Project

**Companion to:** `question.md` (Q-spec v1.0)
**Purpose:** Exhaustive map of *what exists* for each question, not a ranking.
Hybrids of 2–3 methods are expected to be the final answer; you can't hybridize
what you haven't listed.

**Feasibility tags (our data regime: ~230 meetings, daily close-to-close,
liquid-US-large-cap universe):**
- ✅ viable at our n
- ⚠️ needs cross-section width, long history, or regime variation
- 🧪 stretch / novel for a club project

Cross-cutting rule from the spec: *firms multiply the cross-section; only the
calendar multiplies the treatment.* Methods below inherit the currency of the
question they serve.

---

## Q1 — Total effect of the surprise on returns

### Classical / econometric
- ✅ **Event-study OLS** — return on surprise, meeting-clustered SEs. The baseline everything else must beat.
- ✅ **Instrumental variables** — futures surprise as instrument for the rate change; cleans measurement error in the policy action.
- ✅ **Local projections (Jordà)** — horizon-by-horizon response without VAR dynamics; handles daily data naturally.
- ✅ **Two-way fixed effects panel** — meeting + firm/sector dummies; absorbs common shocks per meeting.
- ⚠️ **DiD family** — staggered adoption with heterogeneity-robust estimators (Callaway–Sant'Anna, Sun–Abraham), treated = high-sensitivity units. Club's existing workstream.
- ⚠️ **Synthetic control / synthetic DiD (Arkhangelsky et al.)** — construct no-shock counterfactual market from rate-insensitive assets.
- ⚠️ **FAVAR (Bernanke–Boivin–Eliasz)** — surprise as external instrument in a factor-augmented VAR; prices the transmission to many series jointly.
- ✅ **Interrupted time series / ARIMA intervention (Box–Tiao)** — pre/post event-window mean shift with explicit autocorrelation.
- ⚠️ **Bayesian structural time series (CausalImpact)** — counterfactual from posterior predictive; honest UQ at small n.
- ⚠️ **Regression discontinuity in time** — known to be biased at the cutoff; list for completeness, prefer not.
- ✅ **Nonparametric dose-response** — LOESS / local polynomial / Nadaraya–Watson of return on surprise; checks linearity of the "one number" claim.
- ✅ **Gaussian process regression** — surprise → return with full predictive uncertainty; excellent at n≈230, gives a principled error bar on the effect curve.
- ✅ **Hierarchical regression with partial pooling** — sector/firm betas shrunk toward market beta; the right answer to "per-sector effect" at small n.

### Identification refinements (treatment definition, run all, compare)
- ✅ **GSS two-factor (target/path)** and **Swanson three-factor (+LSAP)** — run `gss.R` on raw USMPD columns.
- ✅ **Jarociński–Karadi MP/INFO split** — sign restrictions on stock-rate co-movement; the information-effect correction.
- ✅ **Bauer–Swanson orthogonalization** — purge the surprise of pre-meeting public information; different premise than J-K, agreement = evidence.
- ✅ **Acosta et al. PC surprise (`STMT`)** — already in hand; principal-component alternative.
- ✅ **Miranda-Agrippino–Ricco pure shocks / Cieslak–Schrimpf non-monetary news** — further robustness series.

### ML-flavored causal estimation
- ✅ **Double machine learning (PLR/interactive)** — club's assigned starter; GBM/RF nuisances, chronologically cross-fitted, meeting-clustered.
- ⚠️ **Generalized random forests** — including IV-GRF; treatment-effect heterogeneity in the effect itself; likely underpowered at 230 but citable.
- ✅ **TMLE (targeted maximum likelihood)** — doubly robust, efficient-influence-curve estimation; rarely used in finance, worth having.
- ✅ **AIPW / doubly robust estimators** — simple plug-in version of the same idea.
- ✅ **Kernel regularized least squares (KRLS)** — flexible, interpretable, small-sample-friendly.

### Inference machinery (applies to everything above)
- ✅ Cluster bootstrap / wild cluster bootstrap by meeting.
- ✅ Randomization/permutation inference — placebo-meeting distribution of the effect.
- ✅ Pre-specified power calculation at meeting-level n.

---

## Q2 — Channels / mechanism

### Decomposition / mediation
- ✅ **2SLS channel-by-channel** — surprise → Δchannel (first stage), Δchannel → returns (second stage); rates, dollar, oil, credit separately.
- ✅ **Causal mediation (Imai–Keele–Tingley)** — direct vs indirect effects with bootstrap; formalizes "through what."
- ⚠️ **Sequential g-estimation** — mediation robust to exposure-mediator interaction.
- ⚠️ **Front-door style analysis** — if a channel is the only observed route, its response identifies the effect through it.
- ✅ **Path analysis / SEM** — classical, deprecated for causal claims but a fine descriptive scaffold.

### Attribution / variable-importance over channels
- ✅ **Relative weight analysis (Johnson)** / **dominance analysis** / **Shapley decomposition of R²** — how much of the explained response does each channel carry.
- ✅ **Bayesian model averaging over channel sets** — posterior probability that each channel belongs in the model.
- ✅ **Group lasso by channel block** — sparse attribution.
- ✅ **Leave-one-channel-out attribution** — effect surviving exclusion of each channel; sensitivity table.

### Time-series structural approaches
- ⚠️ **External-instrument VAR impulse responses** — Mertens–Ravn proxy-SVAR: surprise as instrument for the monetary shock; read off responses of DGS2, DTWEXBGS, oil, OAS jointly.
- ⚠️ **Dynamic factor model** — one latent "transmission factor" extracted from all channel responses.
- ⚠️ **Sign-restricted multi-asset decomposition** — extend J-K logic across channel covariance.

### Discovery-flavored (the ML angle on Q2)
- 🧪 **Causal discovery on event windows** — PC algorithm / NOTEARS / PCMCI on the channel+return panel around events; hypothesis-generating graph, not identification.
- ✅ **Gaussian graphical model of channel partial correlations** — what moves independently of what.
- ✅ **Quantile/tail mediation** — do channels matter more in the tails than the mean? (Connects to Q4's tail focus.)

### Robustness to "it's not really the channel"
- ✅ **Rosenbaum bounds / E-values** — how strong would an unobserved confounder of the channel attribution have to be to kill it.
- ✅ **Negative outcome controls** — outcomes the Fed shouldn't move (e.g., non-US names' idiosyncratic components) as falsification.
- ✅ **Austen plots / Oster bounds** — coefficient stability under selection on unobservables.

---

## Q3 — Susceptibility map μ̂(firm state, shock)

*The question where width is real information. Most methods here are viable once
the firm pull exists.*

### Linear / pooled baselines (the V0–V1 gatekeepers)
- ✅ **Pooled interactions (characteristics × shock)** — the load-bearing V0 rung.
- ✅ **Fama–MacBeth cross-sectional regressions** — per-meeting cross-sections, average slopes; the classic finance format for exactly this problem.
- ✅ **Ridge / lasso / elastic net on interactions** — sparse heterogeneity; the trivial-but-strong baseline.
- ✅ **Group lasso** — shrink whole characteristic families together.
- ✅ **GAM / interpretable boosting (EBM)** — smooth, auditable nonlinearity in V1.

### Hierarchical / shrinkage (the small-n-currency discipline)
- ✅ **Bayesian hierarchical model with partial pooling** — firm slopes shrunk toward sector mean shrunk toward market; *the* method for per-firm Fed betas at 230 meetings.
- ✅ **Horseshoe / spike-and-slab priors** — which characteristics moderate, with shrinkage.
- ✅ **Mixed-effects models (random slopes on the surprise)** — frequentist version of the same.
- ✅ **IPCA (Kelly–Pruitt–Su instrumental PCA)** — latent factors whose loadings are functions of characteristics; tailor-made for "sensitivity is driven by observable state." Strongly recommended rung.

### Trees / ensembles
- ✅ **Gradient boosting (LightGBM/XGBoost/CatBoost), quantile objective** — V1; pinball loss doubles as the distributional head Q4 needs.
- ✅ **Quantile regression forests** — conditional distribution directly; feeds the frozen benchmark or the susceptibility model.
- ✅ **BART (Bayesian additive regression trees)** — flexible fit *with* uncertainty; honest UQ where boosting gives none.
- ⚠️ **Distributional forests / transformation forests (Hothorn)** — distribution-on-trees formalism.

### Sequence / representation learning (the V2+ stack)
- ⚠️ **Masked-patch temporal transformer encoder + FiLM shock conditioning** — the draft's architecture; pretrain on ordinary days, freeze, thin adapter on shock.
- ⚠️ **PatchTST / TCN / LSTM-GRU encoders** — alternative sequence backbones; ablate.
- 🧪 **Mixture density network / deep distributional head** — predicts full conditional response distribution; pairs with PIT scoring.
- 🧪 **VAE / β-VAE of firm state** — latent z with disentanglement pressure; interpretable state dimensions as a bonus.
- 🧪 **Contrastive learning on return sequences** — firm-state embeddings via augmentation-invariance.
- 🧪 **Set transformer / DeepSets cross-sectional context** — "everything around a firm" as an explicit permutation-invariant layer.
- 🧪 **Graph neural nets (GCN/GAT/relational)** — supply-chain, credit, ownership edges; V4 rung.
- ⚠️ **Neural hierarchical / conditional autoencoder factor models** — per-firm embeddings inside a global pricing model (deep-factor family, Gu–Kelly–Xiu lineage).

### Time-varying / regime-aware susceptibility
- ✅ **Kalman-filter time-varying betas** — sensitivity as a drifting state; also usable as *features* into the encoder.
- ✅ **Markov-switching / HMM regimes** — "tightening-regime sensitivity ≠ easing-regime sensitivity."
- ⚠️ **Bayesian online changepoint detection (BOCPD)** — discover when the sensitivity map breaks instead of assuming constancy; produces the regime-shift figure.
- ✅ **Exponentially weighted / Bayesian updating of per-firm sensitivity** — online-learning framing of the same idea.

### Transfer / multi-task (attacks the treatment-direction scarcity)
- ✅ **Multi-task learning across event types** — CPI, payrolls, ECB, minutes, press conferences as auxiliary tasks; Fed statements sole test set.
- ⚠️ **Meta-learning (MAML-style) few-shot per-firm adaptation** — global map + rapid per-firm adaptation.
- ⚠️ **Pretrain-finetune (masked reconstruction → auxiliary events → FiLM adapter)** — the draft's three-stage story made explicit.

### Structural priors (finance knowledge injected into ML)
- ✅ **Equity-duration prior** — Dechow–Sloan–Solonnikov duration as a structured feature/prior; ML learns the residual map. Semi-structural hybrid.
- ✅ **Theoretical sensitivity as offset** — model learns deviation from DCF-implied rate sensitivity, not sensitivity from scratch.

### Feature selection with guarantees (which moderators are real)
- ✅ **Model-X knockoffs** — FDR-controlled characteristic selection; the rigorous answer to cherry-picking.
- ✅ **Stability selection** — resample-and-count selection frequencies.
- ⚠️ **Conditional permutation / LOCO inference** — feature importance with honest nulls.
- ⚠️ **SHAP + cluster-robust caution** — fine for explanation, not for inference.

### Uncertainty quantification / calibration
- ✅ **Conformal prediction** (split / CV+ / jackknife+; conformalized quantile regression) — distribution-free intervals on μ̂.
- ✅ **Deep ensembles / MC dropout** — cheap predictive variance.
- ✅ **Simulation-based calibration** — simulate a known-DGP return panel (GARCH + factor + event effect), run the whole pipeline, check recovery. End-to-end validation of *all* Q3 machinery before touching real events.

---

## Q4 — The gap: frozen scoring + drift test

### 4a — Extreme-response scoring (frozen, model-free right branch)
- ✅ **Market model / factor-model abnormal returns** — CAPM, FF3/FF5/Carhart residuals as the simplest frozen benchmark.
- ✅ **PCA/ICA statistical factors from non-event days** — data-driven factor benchmark.
- ✅ **GARCH/EWMA volatility-standardized residuals** — handle the systematic event-day vol expansion.
- ✅ **Quantile / expectile regression benchmark** — center + tail of the conditional distribution.
- ✅ **Conditional density estimators** — kernel conditional density, k-NN conditional distribution, normalizing flows (MAF), distributional regression (GAMLSS).
- ✅ **PIT percentile + within-meeting rank (the spec's S̃)** — robust to the event-day stretch/shift; primary score.
- ✅ **Gaussianized rank (Blom scores)** — recover magnitudes within ranks if needed.
- ✅ **EVT / peaks-over-threshold tail scoring** — extremeness in the far tail specifically.
- ⚠️ **Copula joint (market, firm) model** — residual through the dependence structure rather than linear beta.
- 🧪 **Autoencoder reconstruction error** on event windows as anomaly score — unsupervised cross-check.
- 🧪 **Isolation forest / local outlier factor / deep SVDD** — purely unsupervised anomaly framing of "weird response."
- ✅ **Mahalanobis distance in factor space** — multivariate extremeness.

### 4b — Drift / reversal testing
- ✅ **Per-meeting cross-sectional predictive regression** — future abnormal return on S̃, slopes averaged across meetings (Fama–MacBeth form), meeting-clustered. The primary drift statistic.
- ✅ **Portfolio sorts** — decile long-short spread on S̃; IC/RankIC time series.
- ✅ **Post-event CAR paths** — cumulative abnormal returns at h = 1…20 days; the picture that sells the result.
- ✅ **Matched-placebo comparison** — identical pipeline on non-FOMC matched days; difference-in-differences of the reversal relation. The primary falsification.
- ⚠️ **State-space permanent/transitory decomposition** — is the post-event move a correction of a transitory component? The formal "overreaction" object.
- ⚠️ **Variance-ratio tests** — event-day deviations from random-walk behavior vs normal days.
- ⚠️ **Hazard/survival model of reversal timing** — when, not just whether.
- ⚠️ **Long-horizon reversal (De Bondt–Thaler framing)** at event level — slow correction as the alternative hypothesis.
- ✅ **Block-bootstrap the entire signal→return pipeline** — uncertainty on the *pipeline output*, not just one regression.
- 🧪 **FDR control on the event×firm "extremes" grid** — how many called extremes are real before testing drift on them.
- ✅ **Randomization inference over meetings** — permutation distribution of the drift slope.

---

## Q5 — Trading layer (only if Q4b survives placebos)

### Signal → decision
- ✅ **Policy learning (Athey–Wager / outcome weighted learning / doubly robust policy learning)** — learn the *rule* directly, not via return prediction.
- ⚠️ **Contextual bandits (LinUCB / Thompson / neural)** — the sequential-decision framing with regret logic.
- ✅ **Meta-labeling (López de Prado)** — primary model takes direction, secondary model sizes the bet; a pragmatic hybrid worth knowing.
- ✅ **Constrained portfolio construction** — rank-weighted vs thresholded entry; vol targeting; Kelly sizing.
- ✅ **Regime-gated strategies** — trade the signal only when conditioning state says it's live.

### Honest evaluation
- ✅ **Doubly robust / self-normalized IPS / FQE (fitted Q evaluation) / MAGIC** — off-policy evaluation family; the defensible backtest.
- ✅ **Combinatorial purged cross-validation + embargo** — purge overlapping reversal windows.
- ✅ **Deflated Sharpe ratio, probability of backtest overfitting (PBO)** — multiple-strategy-selection discounts.
- ✅ **White's reality check / Hansen SPA** — family-wise robustness across strategy variants.
- ✅ **Transaction-cost and spread model** — event-day spreads widen around FOMC; a real cost, not a footnote.

---

## Cross-cutting machinery (applies to every question)

- ✅ **Simulation-first validation** — known-DGP synthetic panel; the cheapest way to find pipeline bugs before they become results. *Build first.*
- ✅ **Meeting-clustered everything** — cluster bootstrap, wild bootstrap, permutation inference.
- ✅ **Multiple-testing discipline** — Benjamini–Hochberg, Romano–Wolf, knockoffs; count specifications, pre-designate primary.
- ✅ **Sensitivity to unobserved confounding** — Rosenbaum bounds, E-values, Oster bounds.
- ✅ **Gate/ablation protocol** — the V-ladder is itself a method: no rung advances without beating the previous on held-out meetings under the frozen score.

---

## Hybrid combinations worth pre-registering as candidates

Hybrids are where the results usually live. Strong initial candidates:

1. **Hierarchical pooling + GBM** — GBM captures nonlinearity; hierarchical layer shrinks per-firm estimates toward sector structure. (Q3)
2. **Kalman time-varying betas as encoder features** — regime-aware state feeding the sequence model. (Q3)
3. **Knockoff selection → hierarchical estimation → conformal intervals** — selection, estimation, and UQ each by the right tool. (Q3)
4. **Structural duration prior + ML residual learning** — finance theory supplies the baseline map, ML learns the deviation. (Q3)
5. **DML headline + causal forest second stage on residual heterogeneity** — the modern standard combo. (Q1→Q3 bridge)
6. **Frozen quantile-RF benchmark + conformal calibration + within-meeting rank** — distributional scoring with a coverage guarantee. (Q4a)
7. **Placebo-matched DiD of the drift relation** — scoring machinery + DiD logic; the strongest version of the falsification. (Q4b)
8. **Multi-task auxiliary events + FiLM conditioning** — representation learning that respects the treatment-direction scarcity. (Q3)
9. **GP dose-response with J-K / Bauer–Swanson robustness rotations** — full uncertainty band on the Q1 effect curve under three identification variants. (Q1)
10. **Policy learning on conformal-gapped signals with DR-OPE evaluation** — decision, signal, and evaluation from three different toolboxes. (Q5)

---

## Deliberately omitted (with reasons)

- **Dragonnet / TARNet / deep treatment-effect nets** — data-hungry; at 230 meetings they add fragility, not signal. Cite, don't run.
- **Regression discontinuity in time** — known bias at the cutoff; weak identification at daily granularity.
- **Full end-to-end FOMC transformers** — the draft's V1 sin; the score must stay frozen and model-free.
- **VARs without external instruments** — recursive ordering assumptions are exactly the "third party" confound the surprise instrument exists to kill.
