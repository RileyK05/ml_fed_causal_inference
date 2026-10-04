"""Real-panel builder on tiny hand-made tables (no licensed data needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fedcore.q3 import BASELINES, make_synthetic_panel, run_arms
from fedcore.q3.contracts import CHANNELS, L, RET
from fedcore.q3.real import CONTEXT_NAMES, FUNDAMENTAL_NAMES, Inputs, build_panel, pretrain_corpus
from fedcore.results import ResultStore

DAYS = pd.bdate_range("2000-01-03", periods=300)
M1, M2 = DAYS[260], DAYS[280]
LEV = FUNDAMENTAL_NAMES.index("leverage")
SIZE = FUNDAMENTAL_NAMES.index("log_size")


def _inputs() -> Inputs:
    rng = np.random.default_rng(0)
    daily = pd.concat(
        [
            pd.DataFrame({
                "permno": permno, "dlycaldt": DAYS, "dlyret": rng.normal(0, 0.01, DAYS.size),
                "dlyvol": 1e5, "dlyprc": 10.0, "shrout": 1000,
            })
            for permno in (1, 2, 3)
        ],
        ignore_index=True,
    )
    # permno 1 has zero volume two days before M1: that day must be masked.
    daily.loc[(daily["permno"] == 1) & (daily["dlycaldt"] == DAYS[258]), "dlyvol"] = 0.0
    members = pd.DataFrame({
        "permno": [1, 2, 3],
        "mbrstartdt": [DAYS[0], DAYS[0], DAYS[270]],  # permno 3 joins between the meetings
        "mbrenddt": [pd.NaT, pd.NaT, pd.NaT],
    })
    market = pd.DataFrame({"date": DAYS, "mktrf": 0.001, "rf": 0.0})
    link = pd.DataFrame({
        "gvkey": ["A", "B", "C"], "lpermno": [1, 2, 3], "linkprim": ["P", "P", "P"],
        "linkdt": [DAYS[0]] * 3,
        "linkenddt": [pd.NaT, DAYS[100], pd.NaT],  # permno 2's link has ended
    })
    q = dict(atq=100.0, ceqq=50.0, cheq=10.0, dlcq=0.0, niq=2.0)
    fundq = pd.DataFrame([
        dict(gvkey="A", datadate=DAYS[200], rdq=DAYS[230], dlttq=20.0, **q),  # usable for M1
        dict(gvkey="A", datadate=DAYS[240], rdq=M1, dlttq=90.0, **q),  # reported on M1: not for M1
        dict(gvkey="C", datadate=DAYS[0], rdq=DAYS[0] - pd.Timedelta(days=400), dlttq=5.0, **q),
    ])
    gics = pd.DataFrame({"gvkey": ["A"], "gsector": ["45"], "indfrom": [DAYS[0]], "indthru": [pd.NaT]})
    meetings = pd.DataFrame({"announcement_date": [M1, M2], "s": [0.5, -0.2]})
    for k, name in enumerate(CONTEXT_NAMES[:-1]):  # mkt_rvol21 is computed by the builder
        meetings[name] = float(k)
    return Inputs(daily, members, market, link, fundq, gics, meetings)


@pytest.fixture(scope="module")
def built():
    return build_panel(_inputs())


def test_rows_follow_point_in_time_membership(built):
    panel, rows, notes = built
    got = {d: sorted(g["permno"]) for d, g in rows.groupby("meeting")}
    assert got == {M1: [1, 2], M2: [1, 2, 3]}
    assert notes["n_meetings"] == 2 and notes["dropped_meetings"] == []


def test_history_ends_the_day_before_and_target_is_meeting_day(built):
    panel, rows, _ = built
    inp = _inputs().daily.set_index(["permno", "dlycaldt"])["dlyret"]
    for k, r in rows.iterrows():
        p = DAYS.get_loc(r["meeting"])
        assert panel.target[k] == pytest.approx(100 * inp[(r["permno"], DAYS[p])], rel=1e-5)
        assert panel.history[k, -1, RET] == pytest.approx(100 * inp[(r["permno"], DAYS[p - 1])], rel=1e-5)
    assert panel.history.shape[1] == L


def test_zero_volume_day_is_masked(built):
    panel, rows, _ = built
    k = rows.index[(rows["meeting"] == M1) & (rows["permno"] == 1)][0]
    assert not panel.history_mask[k, -2]
    assert np.isnan(panel.history[k, -2]).all()
    assert panel.history_mask[k, -1]


def test_fundamentals_are_strictly_before_the_meeting(built):
    panel, rows, _ = built

    def row(meeting, permno):
        return panel.fundamentals[rows.index[(rows["meeting"] == meeting) & (rows["permno"] == permno)][0]]

    assert row(M1, 1)[LEV] == pytest.approx(0.20)  # the quarter reported on M1 is not used
    assert row(M2, 1)[LEV] == pytest.approx(0.90)  # ...but it is by the next meeting
    for meeting, permno in ((M1, 2), (M2, 3)):  # ended link; stale quarter
        f = row(meeting, permno)
        assert np.isnan(np.delete(f, SIZE)).all()
        assert np.isfinite(f[SIZE])  # size comes from CRSP, not Compustat


def test_shock_context_and_sector(built):
    panel, rows, _ = built
    assert set(panel.shock[rows["meeting"] == M1]) == {np.float32(0.5)}
    assert np.isfinite(panel.context).all() and panel.context.shape[1] == len(CONTEXT_NAMES)
    assert rows.loc[rows["permno"] == 1, "gsector"].eq("45").all()
    assert rows.loc[rows["permno"] == 1, "gsector_source"].eq("asof").all()
    assert rows.loc[rows["permno"] != 1, "gsector"].isna().all()


def test_sector_before_gics_history_is_backfilled_and_labeled():
    inp = _inputs()
    late = inp.gics.assign(indfrom=DAYS[275])  # GICS history starts after M1
    _, rows, _ = build_panel(Inputs(**{**inp.__dict__, "gics": late}))
    r = rows[rows["permno"] == 1].set_index("meeting")
    assert r.loc[M1, "gsector"] == "45" and r.loc[M1, "gsector_source"] == "backfill"
    assert r.loc[M2, "gsector_source"] == "asof"


def test_run_arms_real_uses_the_cached_panel(tmp_path, monkeypatch):
    tiny = make_synthetic_panel(n_firms=8, n_meetings=12, seed=0)
    meta = {"inputs": {"crsp_daily": "abc123"}}
    monkeypatch.setattr("fedcore.q3.real.load_real_panel", lambda: (tiny, None, meta))
    store = ResultStore(tmp_path)
    run_arms({"pooled": BASELINES["pooled"]}, store, data="real", seeds=(0,))
    params = store.runs()[-1].manifest["params"]
    assert params["data"] == "real" and params["panel_inputs"] == meta["inputs"]
    assert params["noise"] is None
    with pytest.raises(ValueError, match="synthetic data only"):
        run_arms({"pooled": BASELINES["pooled"]}, store, data="real", n_firms=5)


def test_pretrain_corpus_ends_before_cutoff_and_carries_end_dates():
    corpus = pretrain_corpus(M1, stride=1, min_observed=200, inputs=_inputs())
    assert len(corpus) > 0
    assert (corpus.end_dates < M1).all()
    assert corpus.end_dates.max() == DAYS[259]  # last day before the cutoff
    history, mask, ends = corpus.sample(5, seed=0)
    assert history.shape == (5, L, len(CHANNELS)) and mask.shape == (5, L)
    assert np.array_equal(np.isfinite(history).all(axis=-1), mask)
    assert ends.is_monotonic_increasing
    # window i's last row is that firm's return on its end date
    h, _ = corpus.windows(np.array([0]))
    f, e = corpus.firm[0], corpus.end[0]
    assert np.allclose(h[0, -1], corpus.channels[f, e], equal_nan=True)


def test_pretrain_corpus_respects_min_observed_and_stride():
    corpus = pretrain_corpus(DAYS[-1], stride=5, min_observed=252, inputs=_inputs())
    assert ((corpus.end - (L - 1)) % 5 == 0).all()
    assert corpus.observed[corpus.firm[:, None], corpus.end[:, None] + np.arange(-L + 1, 1)].all()


def test_holdout_inputs_splice_yfinance_after_the_crsp_end(monkeypatch):
    from fedcore.q3 import real

    base = _inputs()
    end = DAYS[289]
    base = Inputs(
        daily=base.daily[base.daily["dlycaldt"] <= end],
        members=base.members.assign(mbrenddt=end),
        market=base.market[base.market["date"] <= DAYS[294]],  # FF outlasts CRSP by 5 days
        link=base.link, fundq=base.fundq, gics=base.gics, meetings=base.meetings,
    )
    after = DAYS[285:]
    bars = pd.concat(
        [pd.DataFrame({"permno": p, "date": after, "adj_close": 1.0, "close": 20.0,
                       "volume": 5e4, "ret": 0.002}) for p in (1, 2)],  # permno 3 left the index
        ignore_index=True,
    )
    spy = pd.DataFrame({"date": after, "ret": 0.003})
    monkeypatch.setattr(real, "_yf_returns", lambda: (bars, spy))
    ext = real.holdout_inputs(base)

    new = ext.daily[ext.daily["dlycaldt"] > end]
    assert set(new["permno"]) == {1, 2} and new["dlycaldt"].min() == DAYS[290]
    assert new["shrout"].eq(1000).all()  # shares held at the last CRSP value
    old = ext.daily[ext.daily["dlycaldt"] <= end]
    assert len(old) == len(base.daily)  # CRSP rows untouched, no yfinance rows before the end

    mkt = ext.market.set_index("date")
    assert mkt.loc[DAYS[294], "mktrf"] == pytest.approx(0.001)  # Fama-French while it lasts
    assert mkt.loc[DAYS[295], "mktrf"] == pytest.approx(0.003) and mkt.loc[DAYS[295], "rf"] == 0.0  # then SPY
    assert mkt.index.is_unique

    ends = ext.members.set_index("permno")["mbrenddt"]
    assert ends[[1, 2]].isna().all() and ends[3] == end


def test_cached_panel_loads_without_raw_inputs(tmp_path, monkeypatch):
    """A remote GPU box gets only processed/q3_panel/: the cache must load as is."""
    import json

    from fedcore.q3 import real

    panel, rows, _ = build_panel(_inputs())
    for name in real._ARRAYS:
        np.save(tmp_path / f"{name}.npy", getattr(panel, name))
    rows.to_parquet(tmp_path / "rows.parquet", index=False)
    (tmp_path / "meta.json").write_text(json.dumps({"inputs": {"crsp_daily": "abc"}}), encoding="utf-8")
    monkeypatch.setattr(real, "CACHE", tmp_path)
    monkeypatch.setattr(real, "_fingerprints", lambda names: None)
    got, got_rows, meta = real.load_real_panel()
    assert np.array_equal(got.target, panel.target) and len(got_rows) == len(rows)
    assert meta["inputs"] == {"crsp_daily": "abc"}
    with pytest.raises(FileNotFoundError):
        real.load_real_panel(rebuild=True)
