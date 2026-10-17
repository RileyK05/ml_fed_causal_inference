Saved runs for this project: results/<question key>/<run_id>/ via fedcore.results.ResultStore(RESULTS).

Tracked in git: each run's manifest (params, metrics, data fingerprints), its aggregate tables
(comparisons, MSE splits, b grades, folds, training curves) and saved model state.

Not tracked: `tables/predictions.csv`. It has one row per firm and meeting with the CRSP id and
the firm's actual return, which is licensed WRDS data and this repository is public. Views that
need firm-level predictions (residuals, per-firm scatter) only work on a machine with the WRDS
data, where rerunning the run's spec regenerates them.
