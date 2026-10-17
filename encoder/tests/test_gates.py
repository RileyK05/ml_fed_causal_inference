"""The v2 gate tests find a planted group effect and stay quiet without one (synthetic data)."""
import numpy as np
import pandas as pd

from fedenc import gates


def _frame(effect: float, seed: int = 0, n_meetings: int = 120, n_firms: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    m = np.repeat(pd.date_range("2000-01-01", periods=n_meetings, freq="45D"), n_firms)
    s = np.repeat(rng.normal(0, 0.3, n_meetings), n_firms)
    mkt = np.repeat(rng.normal(0, 1.0, n_meetings), n_firms)
    group = np.tile(np.array(["a", "b", "c"])[np.arange(n_firms) % 3], n_meetings)
    y = rng.normal(0, 1.0, len(m)) + effect * s * (group == "c") + rng.normal(0, 1.0, n_meetings).repeat(n_firms)
    return pd.DataFrame({"meeting": m, "s": s, "mkt": mkt, "g": group, "y": y,
                         "vol21": rng.uniform(10, 40, len(m)), "beta": rng.uniform(0.5, 1.5, len(m)),
                         "large": (np.abs(s) >= 0.25).astype(float)})


def test_group_test_detects_a_planted_shock_effect():
    per, p, n, _ = gates.group_test(_frame(2.0), "g", "y", ["mkt", "s"], "s")
    assert n == 120 and p < 1e-6 and max(per, key=lambda g: per[g][0]) == "c"
    _, p_null, _, _ = gates.group_test(_frame(0.0, seed=1), "g", "y", ["mkt", "s"], "s")
    assert p_null > 0.01


def test_magnitude_test_detects_bigger_moves_on_large_days():
    d = _frame(0.0, seed=2)
    d.loc[(d["g"] == "c") & (d["large"] == 1), "y"] *= 3.0
    per, p, n_large, _ = gates.magnitude_test(d, "g", "y")
    assert p < 1e-6 and max(per, key=lambda g: per[g][0]) == "c" and n_large > 10
