# Encoder setup: next-day return pretraining

Implemented 2026-10-14. This is the selected engineering setup, not an empirical result or
an authorization to train. Commands below that fit models still require the user's explicit
go-ahead in chat. Changes are confined to `encoder/` and this project's documentation.

## Objective and transfer

The history transformer takes the existing five channels (`ret`, `dlogvol`, `rvol21`,
`ret_rel`, `beta63`) for 252 trading days ending at close t. A temporary two-layer head
predicts the next day's raw return in two parts: its expected value `m(h)` (percent) and
its variance `v(h) = softplus(head(h)) + 1e-6` (percent squared). The score is the Gaussian
negative log-likelihood without its constant:

```text
loss_i,t = log(v_i,t) + (r_i,t+1 - m_i,t)² / v_i,t
```

The population optimum is the conditional mean and variance of the next raw return given
history. No shock enters: pretraining teaches firm state, not a response (question.md, Q3
Training). Ordinary-day rate loading and Fed-surprise sensitivity are not pretraining labels.
This task is a candidate initialization for useful firm state; improvement on Q3 remains
to be tested. The head is discarded after pretraining.

Fed adaptation uses the existing model: history + point-in-time fundamentals + context
feed the fusion network, and `mu = a(z) + b(z)*s` predicts the raw Fed-day return.
Both a and b train normally. The default transfer comparison adapts all encoder weights,
using encoder lr = head lr / 10. Frozen and top-block transfer remain available.
The scratch comparator receives the same ordinary-day history scaler and architecture,
but random encoder weights. All comparison arms therefore share history normalization;
the three-way runner rejects checkpoints with different history scalers. A standalone
`TxArm(init="scratch")` without a checkpoint still uses the shared training-meeting scaler.

## Data and chronology

- `fedenc.return_corpus` wraps the shared real `PretrainCorpus`; it never changes raw data.
- Default stride 1 retains every eligible ordinary target day, quiet or volatile.
- Minimum observed history is 126 days; gaps remain NaN, with the existing day mask.
- A label is the next **trading calendar** day's observed raw return, not the firm's next
  available return. A missing target is dropped; it is never filled or shifted forward.
- All `usmpd_statements` target dates are excluded, including unscheduled statements.
  Historical FOMC observations inside the 252-day context remain allowed.
- Both history and target dates must precede the exclusive cutoff. The 2026 holdout is
  excluded by the CLI and is never loaded by this workflow.
- Pretraining validation contains the last 10% of unique target dates; all firms on a
  date remain together. Training labels must end before the first validation history
  begins, so reconstruction values cannot occur on both sides. The same conservative
  252-day history purge applies to both objectives for a matched comparison. It removes
  approximately a trading year of examples adjacent to validation; past context overlap
  would be acceptable for the next-day task alone, but not for its reconstruction comparator.
  Tiny synthetic unit-test corpora explicitly disable the purge; real CLI commands do not.
- The scaler uses a deterministic bounded sample of at most 4,096 training histories:
  median/IQR per channel, clipping at ±5. Validation does not fit the scaler. Sampling
  bounds memory; increase `--scaler-windows` if needed and record it as a specification.
- Windows are fetched in batches. The shared firm/day channel grid remains resident;
  this is not a disk-backed grid implementation, and no hardware/runtime claim is made.
- Training and evaluation losses give dates equal weight and firms equal weight within
  date. Date groups are shuffled for optimization after the chronological split.
- Checkpoint `available_through` includes every pretraining training/validation target.
  An arm checks it against its first **inner early-stopping validation meeting** before
  fitting, and against prediction dates. Checking only the outer test is insufficient.

The history median/IQR travels with the encoder and remains fixed during adaptation.
Fundamental and context scalers fit on adaptation training meetings only. An encoder-local
adapter reuses shared meeting loss/optimizer helpers; `core/` is unchanged.

## Defaults

Architecture remains 5-day patches, width 64, three blocks, four heads, FFN 256,
dropout 0.1, masked mean pooling. Pretraining: AdamW, lr 1e-3, weight decay 1e-4,
30-epoch maximum, patience 5, batch size 256, four target dates per optimizer step,
gradient norm cap 1. The train-loss curve scores a fixed sample of 250 training dates
(`--train-eval-dates`); validation scores every validation date. Validation masks for the reconstruction comparator repeat across
epochs. Encoder initialization and training seeds are set explicitly by the workflow.

## Commands

Run from the repo root with its installed venv. `plan` loads no datasets and takes no
optimizer steps; `audit` loads the ordinary-day corpus and reports coverage without fitting
or saving a run.

```powershell
.venv\Scripts\python -m fedenc.workflow plan --cutoff 2000-10-03
.venv\Scripts\python -m fedenc.workflow audit --cutoff 2000-10-03
```

The following are **training commands**, provided for later authorized execution:

```powershell
# Candidate objective: one stage, followed by Fed adaptation.
.venv\Scripts\python -m fedenc.workflow pretrain --cutoff 2000-10-03 --device cpu

# Comparator on the same dated histories, not another stage in the candidate.
.venv\Scripts\python -m fedenc.workflow pretrain --cutoff 2000-10-03 --objective masked_reconstruction --device cpu

# Replace MOMENT_RUN and MASKED_RUN with the saved run directories printed above.
.venv\Scripts\python -m fedenc.workflow evaluate --checkpoint MOMENT_RUN --masked-checkpoint MASKED_RUN --through 2009-03-18 --min-train 60 --test-size 16 --arms tx_scratch tx_pre_full tx_masked_full --seeds 0 1 2 --device cpu
```

For the currently cached meetings, 60 initial training meetings place the first inner
validation meeting at 2000-10-03, and the first scored meeting at 2001-11-06.
The first 120 meetings end on 2009-03-18. These commands keep selection inside that
development period. Guards calculate the actual fold boundaries instead of trusting
these example dates. A checkpoint through 2009 would be rejected for this comparison.

For a later **fixed-specification** outer evaluation with 120 initial meetings, the first
inner validation meeting is 2007-01-31 and the first outer test is 2009-04-29 in the current
cache. Either reuse the earlier checkpoint or build a new pretraining checkpoint with an
appropriate cutoff. The originally suggested 2009-04-29 checkpoint is too late for the
trainer's existing inner validation. Any later refreshed checkpoints require their own
recorded specifications; the workflow does not automatically pretrain per fold.

No architecture or pretraining objective is selected using 2009–2025 outer test results,
the 2026 holdout, reversal outcomes or trading profits. Mean per-meeting Fed-return MSE is
the primary transfer metric; RankIC and centered MSE remain in saved results. The runner's
paired comparison uses the shared meeting bootstrap, which does not itself account for
serial dependence. This setup does not add a new b-identification or b-calibration metric.

## Saved artifacts and API

All writes go through `ResultStore.save`, under `encoder/results/q3_susceptibility/`.
Pretraining creates one append-only exploratory run with curves, dataset fingerprints,
architecture, objective, seeds, date boundaries, weights and scaler. Checkpoints store
tensor arrays in CSV/JSON tables because the shared store has no binary-artifact API.
They do not pickle objects or contain the temporary head or raw training windows.
The loader verifies tensor shapes, scaler channel order and a weights/scaler SHA-256 digest.

Evaluation saves each arm's predictions, curves, fold/seed metrics, response weights and
scalers, then an exploratory comparison run. Saved fit ordinals map to the folds table's
row order. Checkpoint fingerprints must match the current registered inputs before fits.

```python
from fedenc.checkpoint import load_checkpoint
from fedenc.arms import TxArm, make_arms

checkpoint = load_checkpoint("encoder/results/q3_susceptibility/PRETRAIN_RUN")
arm = TxArm(init="pretrained", adapt="full", checkpoint=checkpoint)
# arm.fit(train_panel) is training; only call after chat authorization.
# arm.predict(test_panel) returns mu, a, b.
factories = make_arms(checkpoint)
```

`ARMS` retains the historical arm names, but pretrained factories require an explicit
checkpoint. Use `make_arms(checkpoint)` to bind it. `pretrain_kwargs` no longer starts
training from inside `TxArm.fit`. The old `get_pretrained` function remains an explicit
synthetic diagnostic only, without a disk cache. The older array-based `pretrain.py`
remains for synthetic tests; the real workflow never infers dates by overlap chaining.

The reconstruction comparator preserves patch positions when hiding values. The earlier
implementation replaced positional information with the mask token too; that is corrected.

## Verification

Unit tests cover label timing, missing targets, FOMC exclusions, calendar splits,
training-only scaling, equal date weights across batches, Gaussian-score gradients,
checkpoint integrity, normalization transfer, checkpoint availability, frozen layers,
and absence of implicit training. Tiny synthetic smoke fits never save trained runs.
Real training and benchmarks have not been launched for this implementation.
