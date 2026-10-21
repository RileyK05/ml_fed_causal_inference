import numpy as np
import pandas as pd

from fedcore.q3 import beyond_beta as bb


def _panel(n_meet=60, n_firm=400, noise=2.0, seed=0):
    rng = np.random.default_rng(seed)
    meetings = pd.date_range("2000-01-01", periods=n_meet, freq="45D")
    s = pd.Series(rng.choice([-1, 1], n_meet) * rng.uniform(0.3, 1.5, n_meet), index=meetings)
    idx = pd.MultiIndex.from_product([meetings, range(n_firm)], names=["meeting", "firm"])
    true_db = np.tile(rng.normal(0, 0.2, n_firm), n_meet)          # pct per 10bp, beyond beta
    sv = s.reindex(idx.get_level_values(0)).to_numpy()
    base = rng.normal(0, 1, len(idx))
    target = base + true_db * sv + rng.normal(0, noise, len(idx))
    return pd.DataFrame(dict(target=target, mu_base=base), index=idx), true_db, sv, s


def test_oracle_recovers_true_spread_and_unit_calibration():
    pred, true_db, sv, s = _panel()
    pred["mu"] = pred["mu_base"] + true_db * sv
    r = bb.score(pred, s, min_abs_s=0.0, n_boot=300)
    q = np.quantile(true_db[:400], [0.1, 0.9])
    expected = 100 * (true_db[:400][true_db[:400] >= q[1]].mean() - true_db[:400][true_db[:400] <= q[0]].mean())
    assert abs(r["spread"] - expected) < 0.25 * expected
    assert r["spread_lo"] > 0
    assert abs(r["predicted_spread"] - expected) < 1e-6 * expected + 1e-9
    assert 0.8 < r["calibration"] < 1.2
    assert r["rank_ic"] > 0


def test_noise_model_has_no_spread_and_shrunk_calibration():
    pred, true_db, sv, s = _panel(seed=1)
    pred["mu"] = pred["mu_base"] + np.random.default_rng(9).normal(0, 0.2, len(pred)) * sv
    r = bb.score(pred, s, min_abs_s=0.0, n_boot=300)
    assert r["spread_lo"] < 0 < r["spread_hi"]
    assert abs(r["calibration"]) < 0.2


def test_sign_of_surprise_does_not_flip_the_metric():
    pred, true_db, sv, s = _panel(seed=2)
    pred["mu"] = pred["mu_base"] + true_db * sv
    r = bb.score(pred, s, min_abs_s=0.0, n_boot=100)
    r_neg = bb.score(pred, s[s < 0], min_abs_s=0.0, n_boot=100)
    assert r["spread"] > 0 and r_neg["spread"] > 0


def test_no_ranking_counts_as_zero_gap():
    pred, _, _, s = _panel(n_meet=20, seed=3)
    pred["mu"] = pred["mu_base"]
    t = bb.meeting_table(pred, s)
    assert (t["gap"] == 0).all() and t["ic"].isna().all()
    r = bb.summarize(t, n_boot=50)
    assert r["spread"] == 0 and np.isnan(r["rank_ic"]) and np.isnan(r["calibration"])


def test_filters_on_surprise_size_and_exclusions():
    pred, true_db, sv, s = _panel(n_meet=30, seed=4)
    pred["mu"] = pred["mu_base"] + true_db * sv
    big = s.abs() >= 1.0
    r = bb.score(pred, s, min_abs_s=1.0, exclude=[s.index[big][0]], n_boot=50)
    assert r["meetings"] == int(big.sum()) - 1


def test_paired_difference_favours_the_oracle():
    pred, true_db, sv, s = _panel(seed=5)
    oracle = pred.assign(mu=pred["mu_base"] + true_db * sv)
    noise = pred.assign(mu=pred["mu_base"] + np.random.default_rng(3).normal(0, 0.2, len(pred)) * sv)
    r = bb.paired(oracle, noise, s, min_abs_s=0.0, n_boot=300)
    assert r["meetings"] == 60 and r["spread_lo"] > 0 and r["rank_ic_lo"] > 0
    same = bb.paired(oracle, oracle, s, min_abs_s=0.0, n_boot=50)
    assert same["spread"] == 0 and same["spread_lo"] == 0
