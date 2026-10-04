# Package 01: patch-transformer history encoder

**Goal:** implement the history encoder from `docs/encoder/transformer.md` as a `HistoryEncoder`
(package 00's contract), add masked-patch pre-training, and test on the synthetic panel
whether it beats the baselines and `summary_nn`. Find out whether it actually learns the
sequence-dependent part of the planted b.

Prerequisite: package 00 is merged. Read first: `AGENTS.md`, all of `docs/encoder/transformer.md`,
`contracts.py`, `model.py`, `train.py` and `evaluate.py` in `core/fedcore/q3/`, and
`docs/reports/core-report.md` (the baseline numbers to beat).

## Files you own

Everything lives in the **encoder/** project. It imports `fedcore` (including `fedcore.q3`)
and never imports `dml/` or `cde/` (`core/tests/test_isolation.py` enforces this).

```text
encoder/fedenc/encoder_transformer.py
encoder/fedenc/pretrain.py
encoder/fedenc/run.py          # run_arms({**BASELINES, **ARMS}, ResultStore(RESULTS))
encoder/tests/test_encoder.py
encoder/pyproject.toml         # add libraries to the dl extra if needed (only edit outside fedenc/)
```

Never edit `core/` (including `fedcore/q3/`), `dml/` or `cde/`. If the shared contract needs a
change, request it in your report. Install with
`pip install -e "core[dev,q3]" -e "encoder[dev,dl]"`.

## 1. The encoder (`encoder_transformer.py`)

`PatchTransformerEncoder(HistoryEncoder)`, with the v1 configuration from
`docs/encoder/transformer.md` section 2:

| Part | v1 setting |
|---|---|
| Patching | Non-overlapping, `patch_len=5`. Left-pad L=252 to 255, giving **51 patches** |
| Patch input | Flatten the 5×C values **plus the 5 day-mask bits**: 5·C + 5 = 30 inputs per patch. Unobserved values are already 0; the mask bits tell the model which zeros are real |
| Patch embedding | `nn.Linear(30, d_model)`, `d_model=64` |
| Position | Learned embedding, 51 × 64, added |
| Patch validity | A patch is valid if any of its days is observed. Fully padded patches are excluded from attention (`src_key_padding_mask`) and from pooling |
| Blocks | 3 × pre-norm `nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True, norm_first=True)` |
| Attention | **Bidirectional** (no causal mask). Every token is already before the cutoff |
| Final norm | LayerNorm |
| Pool | Masked mean over valid patches → h (64) |
| `out_dim` | 64 |

Keep `patch_len`, `d_model`, `n_layers`, `n_heads` and `pool` ("mean" | "last" | "cls") as
constructor arguments for the ablations. Expose `forward_tokens(x, mask) -> (tokens, patch_valid)`
returning the per-patch outputs before pooling; pre-training needs it.

A defensive detail: if a row has zero valid patches (it shouldn't happen after the synthetic
min-history rule), return zeros rather than NaN, and assert in tests that it doesn't occur.

## 2. Pre-training (`pretrain.py`)

Masked patch reconstruction (`docs/encoder/transformer.md` section 8):

- **Data:** `synthetic.make_pretrain_windows(n_windows=20000, seed)`. These are ordinary-day
  windows, with no meetings or targets. Use the scaler fitted on these windows (it's a separate
  corpus).
- **Masking:** for each window, choose 30% of the *valid* patches at random. Replace their
  embedded tokens with a learned `[MASK]` vector **before** the transformer, so the hidden
  values are never seen.
- **Decoder:** a small MLP (64 → 128 → 5·C) applied to the output token at each masked
  position, predicting the original scaled 5×C values.
- **Loss:** MSE on **observed** entries of masked patches only (use the day mask). Average per
  observed entry. Balance across channels (mean of per-channel MSE) so no channel dominates.
- **Validation:** the last 10% of windows by window end date (a calendar block, not random).
  Early-stop on it.
- **Output:** the encoder `state_dict`. The decoder is discarded.
- The function signature `pretrain(encoder, windows, mask, *, epochs, lr, seed) -> PretrainResult`
  returns the encoder plus its curves.

Chronology note for later real data: a pre-training checkpoint may only be used for test folds
that start after the last day in its corpus. On synthetic data, generate the pre-training
windows from a separate seed and note in the report that real data will need per-fold
checkpoints (`docs/encoder/transformer.md` section 11).

## 3. Arms (defined in `fedenc/run.py` as `ARMS`)

| Arm name | What |
|---|---|
| `tx_scratch` | `ResponseModel(PatchTransformerEncoder())`, trained end to end from random init |
| `tx_pre_frozen` | pre-trained encoder, `freeze_encoder=True` |
| `tx_pre_top` | pre-trained, then unfreeze the top block only (freeze the embedding, positions and blocks 0-1) |
| `tx_pre_full` | pre-trained, all weights trainable, `encoder_lr = lr / 10` |

## 4. Milestones (do them in order; stop and report if one fails)

**M1: the module works.** Unit tests in `encoder/tests/test_encoder.py`:
- output shape (B, 64); 51 patches for L=252;
- **mask invariance:** randomizing values where mask=False leaves h unchanged (atol 1e-5);
- **padding invariance:** adding extra left padding to a short history leaves h unchanged;
- no NaN with partially observed patches;
- pre-training masks only valid patches and the loss ignores unobserved entries.

**M2: learns on easy data.** On `noise="easy"`, `tx_scratch` beats `summary_nn` on `b_corr`
and `mse_per_meeting`. If it doesn't, debug before going further: check the learning rate,
the scaling and the mask handling. Plot train and validation curves.

**M3: realistic data.** All four arms plus the package-00 baselines on `noise="realistic"`, 3
seeds, via `run_walk_forward`. The questions to answer:
- Does any transformer arm beat `ridge_interact`, `lgbm` and `summary_nn` on
  `mse_per_meeting` (paired CI) and on `b_corr`?
- **Does pre-training help (E2), and does adaptation depth matter (E3)?**
- Where does the gain come from? Split `b_corr` into rows where the planted sequence pattern
  is present vs absent.

**M4: small ablations** (pre-declare the grid in the report before running; keep it to about 8
runs): `patch_len` ∈ {5, 10}; `n_layers` ∈ {1, 3}; `pool` ∈ {mean, last}; history-only vs
full fusion (fundamentals and context zeroed).

**M5: diagnostics, cheap.** Linear probes from h to trailing volatility, beta63 and drawdown
(fit on train meetings, score on test), plus one attention-map figure for a row with the planted
pattern. State plainly in the report that attention maps are not explanations
(`docs/encoder/transformer.md` section 13).

## 5. Budget and practicalities

- CPU must work. The v1 model is about 150k parameters. If an epoch on the full synthetic panel
  takes more than about 2 minutes on CPU, reduce `n_firms` for development and say so.
- `torch.manual_seed`, numpy and Python seeds are all set from the arm seed. Use deterministic
  algorithms where cheap.
- Don't add PyTorch Lightning or other frameworks. Plain PyTorch, using the shared trainer.

## Exit criteria

- M1-M3 complete; M4-M5 attempted.
- Tests pass; `fedcore check`, `pytest core` and `pytest encoder` are green.
- Runs saved to `encoder/results/` through `ResultStore` (`role="exploratory"`).
- `docs/reports/encoder-report.md` contains: the results table (all arms + baselines, easy and
  realistic, mean ± sd over 3 seeds), the pre-training curves, the ablation table, the probe
  results, deviations, requested contract changes, and an honest verdict on whether the
  transformer earns its complexity on this synthetic task.
