# Plan: clean up the Q3 encoder code after the final test

Written 2026-10-16, branch `cleanup/q3-encoder`. Code and tests only: **no training, no
`run_arms`, nothing that saves a run.** Read `AGENTS.md` and
[encoder-report.md](../reports/encoder-report.md) first.

## Goal

Keep only the code behind the reported result: the two-part model, its diagnostics, the panel
options and the CAPM baseline. Delete the first-trial pretraining machinery (masked-patch and
next-return pretraining, checkpoints, staged fine-tuning, the ordinary-day arm, the inputs
experiment). Every number in the report must still reproduce from HEAD, from saved
predictions, with no refitting.

## Ground rules

- **Tag first:** `git tag q3-pretraining-archive` on the last master commit (6b21ecd) before
  deleting anything, so the deleted code stays one checkout away. Saved runs that the deleted
  code produced (`encoder/results/q3_susceptibility/`, gitignored) are append-only and stay.
- **Never touch:** the saved runs, `decouple.final` (the locked final-test command, run once),
  the WRDS data, the 2026 holdout, or `fedcore.q3.train`'s public `fit`/`predict` (CDE and
  `SummaryNN` use them).
- **One commit per step** below, each with `pytest core` and `pytest encoder` passing.

## Keep / delete map

### encoder/fedenc

| File | Action | Reason |
|---|---|---|
| `decouple.py` | **Keep.** Import `save_runs` from the new `runs.py` instead of `daily._compare_and_save` | The locked model, `run` / `confirm` / `final` |
| `diagnostics.py` | **Keep.** Fix the docstring that points at `daily.b_grade` | Scores every reported table |
| `encoder_transformer.py` | **Keep** | `PatchTransformerEncoder`, used by `decouple` and `cheap_gradients` |
| `workflow.py` | **Keep the panel half:** `GICS_SECTORS`, `with_sector`, `market_adjusted`, `SHOCKS`, `with_shock`, `TARGETS`, `bounded_panel`. **Delete:** `PRETRAIN_DATASETS`, `ADAPTS`, `VAL_FRAC`, `corpus_audit`, `fold_cutoffs`, `_checkpoint_set`, `evaluate_checkpoint`, `pretrain_one`, `_fold_args`, `_pretrain_args`, `parser`, `main` (the `plan` / `audit` / `pretrain` / `folds` / `pretrain-folds` / `evaluate` commands) | Name kept to avoid churn in docs and scripts |
| `daily.py` | **Delete.** Move `_compare_and_save` to a new `runs.py` as public `save_runs` (`_eval_chunked` goes with the deleted training loop) | Ordinary-day pretraining, `DailyArm`, `run_inputs`; `b_grade` duplicates `diagnostics.b_grades` |
| `arms.py` | **Delete** | `TxArm`, `make_arms`, pretrained/frozen/top adaptation |
| `train.py` | **Delete** | `fit_transferred` (staged fine-tuning) |
| `pretrain.py`, `pretrain_return.py`, `return_corpus.py`, `checkpoint.py` | **Delete** | Pretraining and checkpoints |
| `cheap.py`, `cheap_gradients.py` | **Keep** | The cheap-test runner (R and B series in `cheap-test-menu.md`); not pretraining |

### encoder/tests

| File | Action |
|---|---|
| `test_decouple.py`, `test_diagnostics.py`, `test_cheap.py`, `test_fedenc_import.py` | Keep; update imports |
| `test_encoder.py` | Keep the encoder tests (shape, mask, padding, NaN, attention, pool/patch variants, masked-token positions). Delete the pretraining and `TxArm` tests (`test_pretrain_choice_only_valid_patches`, `test_loss_ignores_unobserved_entries`, both `window_end_order` tests, `test_pretrain_smoke`, `test_txarm_fit_predict_smoke`, `test_pretrained_top_freezes_the_lower_stack`, `test_pretrained_arm_refuses_missing_checkpoint`). Rewrite `test_response_model_integration` without `TxArm` if it uses it |
| `test_daily.py` | Delete (everything it tests is deleted; `b_grade` is covered by `diagnostics.b_grades` tests) |
| `test_return_pretrain.py` | Delete |
| New `test_runs.py` | `save_runs` on a tiny synthetic panel into a tmp `ResultStore` (pooled + capm_pooled only): one run per arm, summary columns, role passed through |

### core/fedcore/q3

| Item | Action | Reason |
|---|---|---|
| `real.py`: `PretrainCorpus`, `pretrain_corpus` | Delete, with its two tests in `test_q3_real.py` | Only the encoder pretraining used them |
| `synthetic.py`: `make_pretrain_windows` | Delete, with its export in `__init__.py` and its test | Same |
| `train.py`: `freeze_encoder`, `encoder_lr`, `_param_groups` | **Keep** | Part of the shared trainer's contract; CDE may want them. Removing them buys little |
| `baselines.py`, `evaluate.py` | Keep | `capm_pooled` and the reference tables |

`core/tests/test_isolation.py` must still pass: nothing in `core` may import `fedenc`.

### Docs

| File | Action |
|---|---|
| `docs/encoder/next-return-pretraining.md`, `pretraining-debate.md`, `pretraining-status.md`, `next-experiment-proposals.md`, `plan-b-metric-and-rerun.md` | Move to `docs/archive/encoder/`, with one header line each: "Archived 2026-10-16; code removed, see tag `q3-pretraining-archive`" |
| `docs/encoder/HANDOFF.md`, `transformer.md` | Add a status line at the top pointing to the report; strike or archive the pretraining build steps |
| `encoder/README.md` | Rewrite: what the package holds now, the `decouple` and `diagnostics` commands, a link to the report |
| `docs/encoder/project-overview.md` | Add a note that the pretraining sections describe archived code |
| `docs/reports/encoder-report.md` | Section 8 (Reproducibility): file table matches the new layout; name the archive tag; mark next-step 4 done |
| `docs/bug_review_session.md` | Leave (historical) |
| `AGENTS.md` | Encoder row: "Q3 two-part transformer (final test done; see `docs/reports/encoder-report.md`)" |

## Steps

1. Tag `q3-pretraining-archive`. Record the baseline: `pytest core`, `pytest encoder`, `pytest cde`, test counts.
2. Add `runs.py` (`save_runs`) and its test; point `decouple.py` at it. Tests pass with `daily.py` still present.
3. Delete `daily.py`, `arms.py`, `train.py`, `pretrain.py`, `pretrain_return.py`, `return_corpus.py`, `checkpoint.py`, and the pretraining half of `workflow.py`. Trim the tests as above. `rg "fedenc\.(daily|arms|train|pretrain|pretrain_return|return_corpus|checkpoint)"` returns nothing in code, tests or live docs.
4. Delete `pretrain_corpus` / `PretrainCorpus` / `make_pretrain_windows` from core with their tests and exports.
5. Docs: archive, README, HANDOFF and transformer status lines, report section 8, AGENTS row.
6. Verify (below), then a code review of the branch diff.

## Verification

- `fedcore check`, `pytest core`, `pytest dml`, `pytest encoder`, `pytest cde` all pass. Counts drop only by the deleted tests (list them in the final summary).
- `python -m fedenc.decouple --help` and `python -m fedenc.diagnostics --help` work.
- **Reproduce the report from saved predictions, no fitting, nothing saved:** `python -m fedenc.diagnostics --no-save two_part=<q3-final-two-part run> pooled=<q3-final-pooled run> --reference pooled`. It must print MSE 5.723 / 5.768 and the -0.045 edge. The dev-to-2021 confirm runs must print 5.927 / 5.954.
- `python -c "import fedenc.decouple, fedenc.diagnostics, fedenc.workflow, fedenc.cheap"` imports cleanly in a fresh process.
- `git diff --stat master` touches nothing under `core/data/` or `encoder/results/`.

## Out of scope

Changing any model, retraining, the 2026 holdout, the winner rule, and the CDE project.
