import numpy as np
import pandas as pd

from fedcore.q3 import drift


def _setup(n_firm=200, n_meet=30, seed=0, follows_s=True):
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("2000-01-03", periods=n_meet * 30 + 40)
    meetings = cal[10:10 + n_meet * 30:30]
    s = pd.Series(rng.choice([-1, 1], n_meet) * rng.uniform(0.3, 1.2, n_meet), index=meetings)
    db = rng.normal(0, 0.3, n_firm)
    abn = rng.normal(0, 1.0, (n_firm, len(cal)))
    for m in meetings:
        p = cal.get_loc(m)
        push = db * (s[m] if follows_s else 0.8)              # drift with or without the surprise's sign
        abn[:, p + 1: p + 6] += push[:, None] / 5
    rows = pd.DataFrame({"meeting": np.repeat(meetings, n_firm), "permno": np.tile(np.arange(n_firm), n_meet)})
    car = drift.car_table(rows, abn, cal, np.arange(n_firm), 5, cal[-1])
    pred = pd.DataFrame({"mu_base": 0.0, "target": 0.0}, index=car.index)
    pred["mu"] = np.tile(db, n_meet) * s.reindex(car.index.get_level_values(0)).to_numpy()
    return pred, car, s, abn, cal, rows


def test_car_sums_days_after_and_respects_end():
    pred, car, s, abn, cal, rows = _setup(n_meet=3)
    m = s.index[0]
    p = cal.get_loc(m)
    np.testing.assert_allclose(car.loc[(m, 7)], abn[7, p + 1: p + 6].sum())
    cut = drift.car_table(rows, abn, cal, np.arange(200), 5, cal[cal.get_loc(s.index[-1]) + 3])
    assert cut.loc[s.index[-1]].isna().all() and cut.loc[s.index[0]].notna().all()


def test_surprise_linked_drift_beats_sign_flips():
    pred, car, s, *_ = _setup(follows_s=True)
    t = drift.drift_table(pred, car, s)
    p, real = drift.sign_flip_p(t, n=500)
    assert real > 0 and p < 0.05


def test_static_characteristic_drift_does_not():
    pred, car, s, *_ = _setup(follows_s=False, seed=2)
    t = drift.drift_table(pred, car, s)
    p, _ = drift.sign_flip_p(t, n=500)
    assert p > 0.05
