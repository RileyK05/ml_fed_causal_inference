import numpy as np
import pandas as pd
import pytest

from fedtrade.backtest import summary, trades
from fedtrade.books import legs, trade_returns
from fedtrade.signals import load_q3, rate_sensitivity


def _meeting(n=100, seed=0):
    rng = np.random.default_rng(seed)
    firms = pd.Index(np.arange(n), name="firm_id")
    db = pd.Series(np.linspace(-1, 1, n), index=firms)          # firm 0 most sensitive
    noise = pd.Series(rng.normal(0, 0.01, n), index=firms)
    return firms, db, noise


def test_legs_pick_the_extremes():
    firms, db, _ = _meeting()
    sens, insens = legs(db)
    assert set(sens) == set(range(10)) and set(insens) == set(range(90, 100))
    assert all(len(x) == 0 for x in legs(pd.Series(0.0, index=firms)))


@pytest.mark.parametrize("s", [-0.5, 0.5])
def test_signs_follow_the_model(s):
    # realized drift = delta_b x s: the model is right, every book earns > 0 in both directions
    firms, db, noise = _meeting()
    car = db * s * 10 + noise
    r = trade_returns(db, car, s, sector=pd.Series("a", index=firms))
    assert r["decile"] > 0 and r["sensitive"] > 0 and r["sensitive_sector"] > 0
    # the model is exactly wrong: every book loses
    r = trade_returns(db, -car, s, sector=pd.Series("a", index=firms))
    assert r["decile"] < 0 and r["sensitive"] < 0


def test_sector_neutral_removes_a_sector_bet():
    # the sensitive leg is all in sector "x", and sector x drifts up for reasons unrelated to firms
    firms, db, noise = _meeting()
    sector = pd.Series(np.where(firms < 30, "x", "y"), index=firms)
    car = pd.Series(np.where(sector == "x", 5.0, 0.0), index=firms) + noise
    r = trade_returns(db, car, -0.5, sector)
    assert r["sensitive"] > 3                       # a pure sector windfall
    assert abs(r["sensitive_sector"]) < 0.05        # gone once measured within sector


def test_trades_and_summary_end_to_end(tmp_path):
    rows = []
    for i, m in enumerate(pd.date_range("2005-01-01", periods=6, freq="90D")):
        firms, db, noise = _meeting(seed=i)
        s = 0.5 if i % 2 else -0.5
        if i == 5:
            s = 0.1                                   # small meeting: skipped
        for f in firms:
            rows.append(dict(meeting=m, firm_id=f, mu_base=0.1, mu=0.1 + db[f] * s, target=0.0, car=db[f] * s + noise[f], s=s))
    df = pd.DataFrame(rows).set_index(["meeting", "firm_id"])
    shock = df["s"].groupby(level=0).first()
    pred = df[["mu", "mu_base", "target"]]
    assert np.allclose(rate_sensitivity(pred, shock).loc[df.index[0][0]].to_numpy(), np.linspace(-1, 1, 100))
    t = trades(pred, shock, df["car"])
    assert len(t) == 5 and (t["decile"] > 0).all() and t["sensitive_sector"].isna().all()
    res = summary(t, cost_pct=0.1, n_boot=200).set_index(["strategy", "meetings"])
    assert res.loc[("decile", "all"), "trades"] == 5
    assert res.loc[("decile", "dovish"), "trades"] + res.loc[("decile", "hawkish"), "trades"] == 5
    gross = summary(t, n_boot=200).set_index(["strategy", "meetings"])
    assert np.isclose(gross.loc[("decile", "all"), "mean"] - res.loc[("decile", "all"), "mean"], 0.2)   # 2 legs x 0.1


def test_load_q3_averages_members(tmp_path):
    run = tmp_path / "20260101-000000_v9-x-f0s" / "tables"
    run.mkdir(parents=True)
    pd.DataFrame(dict(meeting=["2010-01-27"] * 2, firm_id=[1, 1], mu=[1.0, 3.0], mu_base=[0.5, 0.5],
                      target=[2.0, 2.0], member=[0, 1])).to_csv(run / "predictions.csv", index=False)
    p, name = load_q3("v9-x", "F0s", runs=tmp_path)
    assert name.endswith("v9-x-f0s") and p["mu"].iloc[0] == 2.0 and len(p) == 1
    with pytest.raises(FileNotFoundError):
        load_q3("nope", "F0s", runs=tmp_path)
