"""Q3 channels beyond beta: each measure recovers a planted value and uses only pre-meeting data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fedcore.q3 import channels as ch
from fedcore.q3.contracts import BETA63, CHANNELS


def make_grid(n_firms: int = 3, n_days: int = 900, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    calendar = pd.bdate_range("2000-01-03", periods=n_days)
    mkt = rng.normal(0, 1.0, n_days)
    ret = 1.0 * mkt[None, :] + rng.normal(0, 0.5, (n_firms, n_days))
    channels = np.zeros((n_firms, n_days, len(CHANNELS)), dtype=np.float32)
    channels[:, :, BETA63] = 1.0
    return {"calendar": calendar, "permnos": np.arange(n_firms) + 10, "channels": channels,
            "observed": np.ones((n_firms, n_days), dtype=bool), "ret": ret,
            "mktcap": np.full((n_firms, n_days), 1000.0), "mkt": mkt}


def rows_at(grid: dict, positions: list[int]) -> pd.DataFrame:
    n = grid["ret"].shape[0]
    out = [pd.DataFrame({"meeting": grid["calendar"][p], "pos": p, "firm": np.arange(n),
                         "permno": grid["permnos"], "gvkey": [f"{k:06d}" for k in range(n)]}) for p in positions]
    return pd.concat(out, ignore_index=True)


def test_rate_beta_recovers_planted_coefficient():
    grid = make_grid()
    rng = np.random.default_rng(1)
    y2 = pd.Series(np.cumsum(rng.normal(0, 0.2, len(grid["calendar"]))) + 3.0, index=grid["calendar"])
    dy = np.r_[np.nan, np.diff(y2.to_numpy())]
    grid["ret"][1] += -4.0 * np.nan_to_num(dy)  # firm 1 loses 4% per 1pp rise in the 2y yield
    rows = rows_at(grid, [600])
    beta = ch.rate_beta(grid, rows, y2)
    assert beta[1] == pytest.approx(-4.0, abs=0.6)
    assert abs(beta[0]) < 0.8


def test_fed_beta_past_uses_only_past_meetings():
    grid = make_grid(n_firms=4, n_days=2600)
    cal = grid["calendar"]
    pos = np.arange(300, 2600, 30)
    rng = np.random.default_rng(2)
    meetings = pd.DataFrame({"announcement_date": cal[pos], "s": rng.normal(0, 1, pos.size)})
    for p, s in zip(pos, meetings["s"]):
        grid["ret"][0, p] += 0.5 * s   # firm 0 responds beyond beta
        grid["ret"][1, p] -= 0.5 * s
    abn = ch.abnormal_returns(grid)
    rows = rows_at(grid, [int(pos[-1])])
    before = ch.fed_beta_past(grid, rows, meetings, abn)
    assert before[0] == pytest.approx(0.5, abs=0.15)
    assert before[1] == pytest.approx(-0.5, abs=0.15)
    grid["ret"][0, pos[-1]] += 50.0  # the meeting itself must not enter
    after = ch.fed_beta_past(grid, rows, meetings, ch.abnormal_returns(grid))
    np.testing.assert_allclose(before, after)


def test_announce_premium_is_log_ratio_of_event_day_moves():
    grid = make_grid(n_firms=2, n_days=1000)
    macro = np.zeros(1000, dtype=bool)
    macro[::20] = True
    grid["ret"][0, macro] = grid["mkt"][macro] + 2.0 * (grid["ret"][0, macro] - grid["mkt"][macro])
    rows = rows_at(grid, [990])
    prem = ch.announce_premium(grid, rows, macro, ch.abnormal_returns(grid))
    assert prem[0] == pytest.approx(np.log(2.0), abs=0.3)
    assert abs(prem[1]) < 0.3


def quarters(n: int, gvkey: str = "000001") -> pd.DataFrame:
    fyear = 2000 + np.arange(n) // 4
    fqtr = np.arange(n) % 4 + 1
    datadate = pd.date_range("2000-03-31", periods=n, freq="QE")
    return pd.DataFrame({"gvkey": gvkey, "datadate": datadate, "fyearq": fyear, "fqtr": fqtr,
                         "rdq": datadate + pd.Timedelta(days=30)})


def test_compustat_quarters_decumulates_ytd_and_recovers_pass_through():
    n = 16
    q = quarters(n)
    y1_days = pd.date_range("1999-10-01", "2004-12-31", freq="D")
    rng = np.random.default_rng(3)
    y1 = pd.Series(4.0 + np.repeat(rng.normal(0, 1.0, len(y1_days) // 91 + 1), 91)[: len(y1_days)], index=y1_days)
    y1_q = y1.rolling("91D").mean().reindex(q["datadate"]).to_numpy()
    debt = 1000.0
    paid = 1.0 + 0.6 * y1_q  # percent per year: 60% of debt floats
    fund = q.assign(atq=5000.0, dlttq=debt * 0.8, dlcq=debt * 0.2, xintq=paid / 100 * debt / 4, oiadpq=200.0,
                    saleq=500.0, niq=50.0, ceqq=2000.0)
    ytd_q = np.full(n, 30.0)
    ytd = pd.Series(ytd_q).groupby(q["fyearq"]).cumsum().to_numpy()
    cash = q[ch._KEY].assign(oancfy=ytd, capxy=ytd / 3, xintq_cash=0.0)
    funding = q[ch._KEY].assign(dltry=ytd / 10, dltisy=ytd / 5)
    f = ch.compustat_quarters(fund, cash.drop(columns="xintq_cash"), funding, y1)
    np.testing.assert_allclose(f["oancf_q"], 30.0)
    np.testing.assert_allclose(f["dltis_q"], 6.0)
    assert f["pass_through"].iloc[-1] == pytest.approx(0.6, abs=1e-6)
    assert f["st_debt_share"].iloc[-1] == pytest.approx(0.2)
    assert f["fcf_4q"].iloc[-1] == pytest.approx(4 * 20.0 / 5000.0)


def test_asof_skips_reports_on_the_meeting_day_and_stale_ones():
    table = pd.DataFrame({"gvkey": ["000001"] * 2, "rdq": pd.to_datetime(["2001-01-10", "2001-03-20"]),
                          "x": [1.0, 2.0]})
    rows = pd.DataFrame({"meeting": pd.to_datetime(["2001-03-20", "2001-03-21", "2003-01-01"]),
                         "gvkey": ["000001"] * 3})
    got = ch._asof(rows, table, "rdq", ["x"], days=365)["x"].tolist()
    assert got[0] == 1.0 and got[1] == 2.0 and np.isnan(got[2])


def test_implied_duration_properties():
    price = np.array([100.0, 100.0, 100.0, -1.0])
    book = np.array([50.0, 50.0, 50.0, 50.0])
    roe = np.array([0.20, 0.05, 0.20, 0.1])
    growth = np.array([0.02, 0.30, 0.02, 0.0])
    d = ch.implied_duration(price, book, roe, growth)
    assert np.isfinite(d[:3]).all() and np.isnan(d[3])
    assert d[1] > d[0]  # low ROE, high growth: cash flows come later
    assert 5 < d[0] < 30


def test_rating_scale():
    got = ch.rating_number(pd.Series(["AAA", "BBB", " D", "SD", "N.M.", None]))
    assert got.tolist()[:4] == [1.0, 9.0, 22.0, 22.0]
    assert got.iloc[4:].isna().all()


def test_tnic_peers_respect_the_usable_date_and_the_meeting_panel():
    rows = pd.DataFrame({"meeting": pd.to_datetime(["2001-03-01"] * 3 + ["2001-05-01"] * 3),
                         "gvkey": pd.array(["000001", "000002", "000003"] * 2, dtype="string")})
    tnic = pd.DataFrame({"year": [1999, 1999, 2000, 2000], "gvkey1": ["000001", "000002", "000001", "000003"],
                         "gvkey2": ["000002", "000001", "000003", "000001"]})
    pairs = ch.tnic_peers(rows, tnic)
    march = pairs[pairs["row"] < 3]
    may = pairs[pairs["row"] >= 3]
    assert set(map(tuple, march[["row", "peer"]].to_numpy())) == {(0, 1), (1, 0)}      # 1999 links
    assert set(map(tuple, may[["row", "peer"]].to_numpy())) == {(3, 5), (5, 3)}        # 2000 links from April 2001
    cnt, mean = ch.peer_means(pairs, np.arange(6, dtype=float), 6)
    assert cnt[0] == 1 and mean[0] == 1.0 and np.isnan(mean[4])
