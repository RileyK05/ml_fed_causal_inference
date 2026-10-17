"""Tests for the Q3 shared foundation (docs/core/HANDOFF.md)."""
import pytest

pytest.importorskip("torch")
pytest.importorskip("lightgbm")
pytest.importorskip("sklearn")

from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from fedcore.protocol import walk_forward
from fedcore.q3 import (
    BASELINES,
    L,
    ChannelScaler,
    SummaryEncoder,
    make_synthetic_panel,
    per_meeting_mse,
    run_arms,
    run_walk_forward,
    sequence_pattern,
    summary_features,
    validation_meetings,
)
from fedcore.q3.contracts import BETA63, RET, RET_REL, RVOL21
from fedcore.q3.features import channel_window_vol
from fedcore.q3.model import ResponseModel
from fedcore.q3.train import fit
from fedcore.q3.synthetic import RVOL_FALLBACK
from fedcore.results import ResultStore


def test_validate_catches_shock_mask_and_shape():
    panel = make_synthetic_panel(n_firms=8, n_meetings=4, seed=0)
    panel.validate()
    assert (~panel.history_mask).any()

    shock = panel.shock.copy()
    same = np.flatnonzero(panel.meeting == panel.meeting[0])
    assert len(same) >= 2
    shock[same[1]] = shock[same[0]] + 1
    with pytest.raises(ValueError, match="shock"):
        replace(panel, shock=shock).validate()

    history = panel.history.copy()
    observed = np.argwhere(panel.history_mask)[0]
    history[observed[0], observed[1], 0] = np.nan
    with pytest.raises(ValueError, match="mask"):
        replace(panel, history=history).validate()

    history = panel.history.copy()
    missing = np.argwhere(~panel.history_mask)[0]
    history[missing[0], missing[1], :] = 0
    with pytest.raises(ValueError, match="mask"):
        replace(panel, history=history).validate()

    with pytest.raises(ValueError, match="shape"):
        replace(panel, history=panel.history[:, :10, :]).validate()


def test_synthetic_is_deterministic_and_noise_only_moves_the_target():
    easy = make_synthetic_panel(n_firms=10, n_meetings=5, noise="easy", seed=1)
    again = make_synthetic_panel(n_firms=10, n_meetings=5, noise="easy", seed=1)
    other = make_synthetic_panel(n_firms=10, n_meetings=5, noise="easy", seed=2)
    realistic = make_synthetic_panel(n_firms=10, n_meetings=5, noise="realistic", seed=1)
    assert np.array_equal(easy.history, again.history, equal_nan=True)
    assert np.array_equal(easy.target, again.target)
    assert np.array_equal(easy.b_true, again.b_true)
    assert not np.array_equal(easy.target, other.target)
    assert np.array_equal(easy.history, realistic.history, equal_nan=True)
    assert np.array_equal(easy.shock, realistic.shock)
    assert not np.allclose(easy.target, realistic.target)
    gap = np.diff(easy.meetings().values).astype("timedelta64[D]").astype(int)
    assert set(gap.tolist()) == {42}
    assert easy.history.shape[1:] == (L, 5)


def test_channels_match_their_definitions():
    panel = make_synthetic_panel(n_firms=6, n_meetings=4, seed=1)
    row = next(i for i in range(len(panel.target)) if panel.history_mask[i, -63:].all())
    window = panel.history[row, -63:].astype(np.float64)
    ret = window[:, RET]
    market = ret - window[:, RET_REL]
    ret_c = ret - ret.mean()
    mkt_c = market - market.mean()
    beta = float(np.dot(ret_c, mkt_c) / np.dot(mkt_c, mkt_c))
    hand_vol = float(np.std(panel.history[row, -21:, RET].astype(np.float64), ddof=0))
    assert abs(beta - float(window[-1, BETA63])) < 1e-3
    assert abs(hand_vol - float(panel.history[row, -1, RVOL21])) < 1e-3


def test_planted_response_matches_the_documented_formula():
    panel = make_synthetic_panel(n_firms=12, n_meetings=6, noise="easy", seed=0)
    rvol = channel_window_vol(panel.history, panel.history_mask, RET, 63)
    rvol = np.where(np.isfinite(rvol), rvol, RVOL_FALLBACK)
    pattern = sequence_pattern(panel.history, panel.history_mask)
    leverage = panel.fundamentals[:, 0].astype(np.float64)
    b = 0.5 + 2.0 * leverage + 0.9 * rvol + 1.5 * pattern
    a = (
        0.30 * (leverage - 0.45)
        + 0.04 * (panel.context[:, 0].astype(np.float64) - 2.0)
        + 0.03 * (panel.context[:, 1].astype(np.float64) - 18.0)
    )
    assert np.allclose(b, panel.b_true, atol=1e-4)
    assert np.allclose(a, panel.a_true, atol=1e-4)
    residual = panel.target - (panel.a_true + panel.b_true * panel.shock)
    assert float(np.std(residual)) > 0.05


def test_sequence_pattern_is_sometimes_on():
    panel = make_synthetic_panel(n_firms=30, n_meetings=20, seed=0)
    rate = float(sequence_pattern(panel.history, panel.history_mask).mean())
    assert 0.05 < rate < 0.80


def test_scaler_ignores_rows_it_was_not_fit_on():
    panel = make_synthetic_panel(n_firms=10, n_meetings=8, seed=0)
    meetings = panel.meetings()
    train_rows = np.flatnonzero(np.isin(panel.meeting, meetings[:5].to_numpy()))
    test_rows = np.flatnonzero(np.isin(panel.meeting, meetings[5:].to_numpy()))
    scaler = ChannelScaler.fit(panel, train_rows)
    changed = panel.history.copy()
    changed[test_rows] = 1e6
    panel2 = replace(panel, history=changed)
    refit = ChannelScaler.fit(panel2, train_rows)
    assert np.allclose(scaler.hist_median, refit.hist_median)
    assert np.allclose(scaler.hist_iqr, refit.hist_iqr)
    leaked = ChannelScaler.fit(panel2, test_rows)
    assert not np.allclose(scaler.hist_median, leaked.hist_median)
    scaled = scaler.transform(panel)
    assert np.all(scaled["history"][~panel.history_mask] == 0)
    assert np.nanmax(np.abs(scaled["history"])) <= 5 + 1e-5


def test_summary_encoder_ignores_masked_values():
    torch.manual_seed(0)
    encoder = SummaryEncoder()
    encoder.eval()
    x = torch.randn(4, L, 5)
    mask = torch.rand(4, L) > 0.3
    mask[:, :40] = False
    mask[0] = False
    first = encoder(x, mask)
    altered = x.clone()
    altered[~mask] = torch.randn_like(altered[~mask]) * 100
    second = encoder(altered, mask)
    assert torch.allclose(first, second, atol=1e-5, rtol=1e-5)
    assert torch.isfinite(first).all()
    assert first.shape == (4, encoder.out_dim)


def test_summary_features_ignore_masked_values():
    panel = make_synthetic_panel(n_firms=6, n_meetings=3, seed=0)
    original = summary_features(panel)
    history = panel.history.copy()
    history[~panel.history_mask] = 12345
    altered = summary_features(history, panel.history_mask)
    assert np.allclose(original, altered, equal_nan=True)


def test_shock_enters_only_the_head():
    torch.manual_seed(0)
    model = ResponseModel(SummaryEncoder(), n_fund=3, n_ctx=4)
    model.eval()
    rows = 5
    history = torch.randn(rows, L, 5)
    mask = torch.ones(rows, L, dtype=torch.bool)
    fundamentals = torch.randn(rows, 3)
    fund_mask = torch.ones(rows, 3, dtype=torch.bool)
    context = torch.randn(rows, 4)
    shock = torch.randn(rows)
    other = shock + 1.5
    mu, a, b = model(history, mask, fundamentals, fund_mask, context, shock)
    mu2, a2, b2 = model(history, mask, fundamentals, fund_mask, context, other)
    assert torch.allclose(a, a2) and torch.allclose(b, b2)
    assert torch.allclose(mu, a + b * shock)
    assert not torch.allclose(mu, mu2)


def test_frozen_encoder_and_best_checkpoint():
    panel = make_synthetic_panel(n_firms=8, n_meetings=12, noise="easy", seed=0)
    model = ResponseModel(SummaryEncoder(), n_fund=3, n_ctx=4)
    before = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    result = fit(model, panel, max_epochs=3, patience=3, freeze_encoder=True, seed=0)
    after = model.encoder.state_dict()
    for key, value in before.items():
        assert torch.equal(value, after[key])
    assert result.best_epoch == int(np.argmin(result.val_loss)) + 1
    assert len(result.train_loss) == len(result.val_loss)
    assert result.val_meetings.min() > result.train_meetings.max()


def test_meeting_loss_weights_meetings_equally():
    mu = torch.zeros(5)
    target = torch.tensor([0.0, 0.0, 0.0, 0.0, 2.0])
    codes = torch.tensor([0, 0, 0, 0, 1])
    loss = per_meeting_mse(mu, target, codes)
    assert torch.allclose(loss, torch.tensor(2.0))


def test_walk_forward_keeps_a_meeting_on_one_side():
    panel = make_synthetic_panel(n_firms=8, n_meetings=24, noise="easy", seed=0)
    fitted, predicted = [], []

    class Spy:
        def fit(self, part, **kwargs):
            del kwargs
            fitted.append(set(pd.DatetimeIndex(part.meeting)))

        def predict(self, part):
            predicted.append(set(pd.DatetimeIndex(part.meeting)))
            zeros = np.zeros(len(part.target))
            return zeros, zeros.copy(), zeros.copy()

    result = run_walk_forward(panel, lambda seed: Spy(), min_train=12, test_size=4, seeds=(0, 1))
    folds = walk_forward(panel.meetings(), 12, 4)
    assert len(fitted) == len(folds) * 2
    for i, fold in enumerate(folds):
        assert set(fold.train).isdisjoint(set(fold.test))
        train_panel = panel.subset(
            np.isin(
                np.asarray(panel.meeting, dtype="datetime64[ns]"),
                fold.train.to_numpy(dtype="datetime64[ns]"),
            )
        )
        val = validation_meetings(train_panel, 0.15)
        assert val.max() < fold.test.min()
        assert set(val).issubset(set(fold.train))
        for s in range(2):
            assert fitted[i * 2 + s] == set(fold.train)
            assert predicted[i * 2 + s] == set(fold.test)
    assert len(result.predictions) == 8 * 4 * len(folds) * 2


def test_ridge_recovers_planted_sensitivity_on_easy_noise():
    panel = make_synthetic_panel(n_firms=40, n_meetings=48, noise="easy", seed=0)
    result = run_walk_forward(
        panel,
        BASELINES["ridge_interact"],
        min_train=24,
        test_size=8,
        seeds=(0,),
    )
    assert result.metrics["b_corr"] > 0.5


def test_capm_pooled_recovers_beta_scaled_drift_and_shock():
    panel = make_synthetic_panel(n_firms=30, n_meetings=20, noise="easy", seed=1)
    hist, mask = panel.history.copy(), panel.history_mask.copy()
    rng = np.random.default_rng(0)
    beta = rng.uniform(0.5, 1.5, len(panel.target))
    hist[:, :, BETA63] = beta[:, None]
    hist[0, -1, BETA63] = 9.0  # masked: the last observed beta is used instead
    mask[0, -1] = False
    hist[1, :, BETA63] = np.nan  # never observed: beta falls back to 1
    mask[1] = False
    beta[1] = 1.0
    s = np.asarray(panel.shock, dtype=np.float64)
    target = 0.1 + beta * (0.4 - 0.9 * s)
    panel = replace(panel, history=hist, history_mask=mask, target=target.astype(np.float32))
    model = BASELINES["capm_pooled"](0)
    model.fit(panel)
    mu, a, b = model.predict(panel)
    assert (model.c_, model.d_, model.k_) == pytest.approx((0.1, 0.4, -0.9), abs=1e-4)
    np.testing.assert_allclose(mu, a + b * s, atol=1e-9)
    np.testing.assert_allclose(b, -0.9 * beta, atol=1e-4)


def _two_shock_panel(seed=2):
    from fedcore.q3.shocks import info_flags

    panel = make_synthetic_panel(n_firms=30, n_meetings=24, noise="easy", seed=seed)
    meetings = panel.meetings()
    rng = np.random.default_rng(seed)
    sp500 = rng.normal(0.0, 0.5, len(meetings))
    sp500[0] = np.nan  # no S&P window move: counted as a policy meeting
    table = pd.DataFrame({"sp500": sp500, "slope": rng.normal(0.0, 0.4, len(meetings))}, index=meetings)
    beta = rng.uniform(0.5, 1.5, len(panel.target))
    hist = panel.history.copy()
    hist[:, :, BETA63] = beta[:, None]
    flags = info_flags(panel, table)
    assert np.isnan(flags).any() and (flags == 1.0).any() and (flags == 0.0).any()
    return replace(panel, history=hist), table, beta


def test_shock_splits_sum_to_s_and_unknown_meetings_are_policy():
    from fedcore.q3.shocks import split_shock

    panel, table, _ = _two_shock_panel()
    s = np.asarray(panel.shock, dtype=np.float64)
    S = split_shock(panel, "policy_info", table)
    np.testing.assert_allclose(S.sum(axis=1), s)
    assert ((S[:, 0] == 0) | (S[:, 1] == 0)).all()
    first = pd.DatetimeIndex(panel.meeting) == panel.meetings()[0]
    np.testing.assert_allclose(S[first, 0], s[first])  # unknown sign -> policy
    T = split_shock(panel, "target_slope", table)
    np.testing.assert_allclose(T[:, 0], s)
    with pytest.raises(ValueError, match="slope"):
        split_shock(panel, "target_slope", table.iloc[1:])


@pytest.mark.parametrize("kind", ["policy_info", "target_slope"])
def test_capm_two_shock_recovers_loadings_and_keeps_the_b_contract(kind):
    from fedcore.q3.baselines import CAPMTwoShock
    from fedcore.q3.shocks import split_shock

    panel, table, beta = _two_shock_panel()
    S = split_shock(panel, kind, table)
    target = 0.1 + beta * (0.4 - 0.9 * S[:, 0] + 1.7 * S[:, 1])
    panel = replace(panel, target=target.astype(np.float32))
    model = CAPMTwoShock(kind, table=table)
    model.fit(panel)
    mu, a, b = model.predict(panel)
    assert (model.c_, model.d_, model.k1_, model.k2_) == pytest.approx((0.1, 0.4, -0.9, 1.7), abs=1e-3)
    np.testing.assert_allclose(mu, target, atol=1e-4)
    np.testing.assert_allclose(mu, a + b * np.asarray(panel.shock, dtype=np.float64), atol=1e-9)


def test_run_arms_saves_a_run(tmp_path):
    run_arms({"pooled": BASELINES["pooled"]}, ResultStore(tmp_path), n_firms=20, n_meetings=40)
    runs = ResultStore(tmp_path).runs("Q3")
    assert len(runs) == 1
    manifest = runs[0].manifest
    assert manifest["role"] == "exploratory"
    assert manifest["question"] == "Q3"
    assert "mse_per_meeting" in manifest["metrics"]
    pred = runs[0].table("predictions")
    assert {"meeting", "firm_id", "fold", "seed", "mu", "a_hat", "b_hat", "target", "gap", "a_true", "b_true"} <= set(pred.columns)
    assert np.allclose(pred["gap"], pred["target"] - pred["mu"])


def test_unknown_data_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown data"):
        run_arms({"pooled": BASELINES["pooled"]}, ResultStore(tmp_path), data="nope")
