"""Ordinary-day pretraining experiment: dates, context timing, leakage, b grade. Smoke fits only."""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from fedcore.q3.synthetic import make_synthetic_panel
from fedenc import daily
from fedenc.daily import DailyArm, b_grade, ordinary_dates


def test_ordinary_dates_skip_statements_and_respect_bounds():
    market = pd.bdate_range("2000-01-03", periods=30)
    statements = [market[4], market[10]]
    got = ordinary_dates(market, statements, start=market[2], end=market[20], every=3)
    assert got.min() >= market[2] and got.max() < market[20]
    assert not got.isin(statements).any()
    assert len(got) == len(np.arange(0, 18 - 2, 3))  # 16 eligible days, every 3rd


def test_ordinary_context_uses_only_events_and_yields_before_the_day(monkeypatch):
    events = pd.DataFrame({"Date": pd.to_datetime(["2000-01-05", "2000-01-10", "2000-01-12", "2000-01-14"]),
                           "STMT": [1.0, 2.0, 3.0, 4.0], "ME": [0.1, 0.2, 0.3, 0.4]})
    days = pd.bdate_range("1999-09-01", "2000-01-31")
    yields = pd.DataFrame({"Date": days, "SVENY01": np.arange(len(days), dtype=float)})
    monkeypatch.setattr(daily, "load", lambda name: {"mps_surprises": events, "treasury_1y": yields}[name])
    table = daily.ordinary_day_table(pd.to_datetime(["2000-01-14", "2000-01-18"])).set_index("announcement_date")
    # On the 14th (an event day) the previous event is the 12th; afterwards it is the 14th.
    assert table.loc["2000-01-14", "st_prev"] == 3.0
    assert table.loc["2000-01-14", "st_mean3"] == pytest.approx(2.0)
    assert table.loc["2000-01-18", "st_prev"] == 4.0
    assert table.loc["2000-01-18", "st_mean3"] == pytest.approx(3.0)
    # Yield level is the prior trading day's.
    assert table.loc["2000-01-18", "y1_level"] == days.get_loc(pd.Timestamp("2000-01-17"))
    assert (table["s"] == 0).all()


def _daily_panel(n_days=24):
    base = make_synthetic_panel(n_firms=4, n_meetings=n_days, seed=1)
    dates = pd.bdate_range("1997-01-02", periods=n_days)
    remap = dict(zip(base.meetings(), dates))
    return dataclasses.replace(
        base, meeting=np.array([remap[m] for m in pd.DatetimeIndex(base.meeting)], dtype="datetime64[ns]"),
        shock=np.zeros_like(base.shock),
    )


def test_daily_arm_pretrains_only_before_inner_validation_and_trains_b_after():
    fed = make_synthetic_panel(n_firms=4, n_meetings=8, seed=0)
    ordinary = _daily_panel()
    late = dataclasses.replace(ordinary, meeting=ordinary.meeting + np.timedelta64(3 * 365, "D"))
    both = ordinary.subset(np.arange(len(ordinary.target)))
    both = dataclasses.replace(
        both, **{f.name: np.concatenate([getattr(ordinary, f.name), getattr(late, f.name)])
                 for f in dataclasses.fields(ordinary) if getattr(ordinary, f.name) is not None})
    arm = DailyArm(both, device="cpu", pretrain_epochs=1, pretrain_patience=1, max_epochs=1, patience=1)
    b_before = None

    def spy(model, panel, scaler, **kw):
        nonlocal b_before
        b_before = {k: v.clone() for k, v in model.head.b.state_dict().items()}
        return original(model, panel, scaler, **kw)

    original = daily.post_train
    daily.post_train = spy
    try:
        arm.fit(fed)
    finally:
        daily.post_train = original
    assert pd.Timestamp(arm.pretrain_result.train_meetings.max()) < arm.cutoff
    assert pd.Timestamp(arm.pretrain_result.val_meetings.max()) < arm.cutoff
    assert any(not torch.equal(b_before[k], v) for k, v in arm.model.head.b.state_dict().items())
    mu, a, b = arm.predict(fed)
    np.testing.assert_allclose(mu, a + b * fed.shock, atol=1e-5)


def test_b_grade_is_one_for_oracle_and_nan_for_constant_b():
    panel = make_synthetic_panel(n_firms=60, n_meetings=40, seed=0)
    meeting = pd.DatetimeIndex(panel.meeting)
    shock = pd.Series(panel.shock, index=meeting).groupby(level=0).first()
    noise_free = 0.3 + panel.b_true * panel.shock  # synthetic a_true is correlated with b_true
    df = pd.DataFrame({"seed": 0, "meeting": meeting, "target": noise_free, "b_hat": panel.b_true})
    assert b_grade(df, shock)["b_slope"] == pytest.approx(1.0, abs=1e-6)
    flat = df.assign(b_hat=1.0)
    assert np.isnan(b_grade(flat, shock)["b_slope"])


def test_chunked_evaluation_matches_shared_meeting_mse():
    from fedcore.q3 import ResponseModel
    from fedcore.q3.train import ChannelScaler, _eval_loss, _tensors
    from fedenc.encoder_transformer import PatchTransformerEncoder

    panel = make_synthetic_panel(n_firms=5, n_meetings=6, seed=2)
    torch.manual_seed(0)
    model = ResponseModel(PatchTransformerEncoder(d_model=8, n_layers=1, n_heads=2, dim_feedforward=16),
                          n_fund=panel.fundamentals.shape[1], n_ctx=panel.context.shape[1])
    batch = _tensors(panel, ChannelScaler.fit(panel), torch.device("cpu"))
    assert daily._eval_chunked(model, batch, chunk=7) == pytest.approx(_eval_loss(model, batch, False), rel=1e-5)
