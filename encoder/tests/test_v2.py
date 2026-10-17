"""Unit tests for modular Q3 v2 mechanisms (encoder/fedenc/v2.py). Smoke fits only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from fedcore.q3.baselines import CAPMPooled, PooledOLS
from fedcore.q3.synthetic import make_synthetic_panel
from fedenc import v2
from fedenc.encoder_transformer import PatchTransformerEncoder


def tiny_encoder(n_channels: int = 5):
    return PatchTransformerEncoder(
        n_channels=n_channels, d_model=8, n_layers=1, n_heads=2, dim_feedforward=16, dropout=0.0
    )


@pytest.fixture
def panel():
    return make_synthetic_panel(n_firms=6, n_meetings=8, seed=0)


@pytest.fixture
def macro_tables(panel):
    return v2.make_synthetic_macro_tables(panel, seed=42)


def test_v2_config_validation():
    cfg = v2.V2Config(
        base="capm",
        history_macro=True,
        regime=True,
        shock_split="policy_info",
        shock_weight=2.0,
        b_only=True,
    )
    assert cfg.base == "capm"
    with pytest.raises(ValueError, match="base"):
        v2.V2Config(base="invalid")
    with pytest.raises(ValueError, match="shock_split"):
        v2.V2Config(shock_split="invalid")
    with pytest.raises(ValueError, match="shock_weight"):
        v2.V2Config(shock_weight=-1.0)
    with pytest.raises(ValueError, match="w_max"):
        v2.V2Config(w_max=0.5)


def test_augment_inputs_shapes_and_masks(panel, macro_tables):
    n = len(panel.target)
    f0 = panel.fundamentals.shape[1]
    c0 = panel.context.shape[1]

    # 1. Default config: no change in channel/feature counts, 1D shock
    aug0 = v2._augment_inputs(panel, v2.V2Config(), macro_tables)
    assert aug0["history"].shape == (n, 252, 5)
    assert aug0["fundamentals"].shape == (n, f0)
    assert aug0["context"].shape == (n, c0)
    assert aug0["shock_mat"].shape == (n, 1)

    # 2. All switches on: +3 history channels, +2 fundamentals, +4 context, 2D shock
    cfg_all = v2.V2Config(history_macro=True, regime=True, shock_split="policy_info")
    aug1 = v2._augment_inputs(panel, cfg_all, macro_tables)
    assert aug1["history"].shape == (n, 252, 8)
    assert aug1["fundamentals"].shape == (n, f0 + 2)
    assert aug1["fundamentals_mask"].shape == (n, f0 + 2)
    assert aug1["context"].shape == (n, c0 + 4)
    assert aug1["shock_mat"].shape == (n, 2)

    # Unobserved history days stay NaN across all 8 channels
    assert np.array_equal(np.isfinite(aug1["history"]).all(axis=-1), panel.history_mask)
    assert np.array_equal(np.isfinite(aug1["fundamentals"]), aug1["fundamentals_mask"])

    # In policy_info split, s_policy + s_info == s identically, and one of the two is 0 on each row
    np.testing.assert_allclose(aug1["shock_mat"].sum(axis=1), panel.shock, atol=1e-6)
    assert np.all((aug1["shock_mat"][:, 0] == 0.0) | (aug1["shock_mat"][:, 1] == 0.0))

    # 3. target_slope split: first column is s, second is slope_shock
    cfg_ts = v2.V2Config(shock_split="target_slope")
    aug_ts = v2._augment_inputs(panel, cfg_ts, macro_tables)
    assert aug_ts["shock_mat"].shape == (n, 2)
    np.testing.assert_allclose(aug_ts["shock_mat"][:, 0], panel.shock, atol=1e-6)


@pytest.mark.parametrize("base", ["pooled", "capm"])
@pytest.mark.parametrize("shock_split", ["none", "policy_info", "target_slope"])
@pytest.mark.parametrize("b_only", [False, True])
def test_v2_two_part_arm_preserves_base_meeting_mean_and_identity(
    panel, macro_tables, base, shock_split, b_only
):
    cfg = v2.V2Config(
        base=base,
        history_macro=True,
        regime=True,
        shock_split=shock_split,
        shock_weight=1.5,
        b_only=b_only,
        members=2,
        max_epochs=2,
        patience=1,
    )
    arm = v2.V2TwoPartArm(
        0,
        cfg,
        device="cpu",
        macro_tables=macro_tables,
        encoder_fn=tiny_encoder,
    )
    arm.fit(panel)
    mu, a, b = arm.predict(panel)
    assert np.isfinite(mu).all() and np.isfinite(a).all() and np.isfinite(b).all()

    ref_base = CAPMPooled() if base == "capm" else PooledOLS()
    ref_base.fit(panel)
    mu_base, a_base, b_base = ref_base.predict(panel)

    # Meeting-average prediction matches the base model's meeting-average prediction exactly
    meeting = pd.DatetimeIndex(panel.meeting)
    np.testing.assert_allclose(
        pd.Series(mu).groupby(meeting).mean().to_numpy(),
        pd.Series(mu_base).groupby(meeting).mean().to_numpy(),
        atol=1e-6,
    )

    if b_only and shock_split != "target_slope":  # target_slope: a carries the slope response
        np.testing.assert_allclose(a, a_base, atol=1e-7)

    # The output contract holds for every split (target_slope folds the slope response into a)
    np.testing.assert_allclose(mu, a + b * panel.shock, atol=1e-5)


def test_shock_weights_increase_with_shock_size_and_respect_cap(panel, macro_tables):
    cfg = v2.V2Config(shock_weight=2.0, w_max=3.5)
    aug = v2._augment_inputs(panel, cfg, macro_tables)
    scaler = v2._V2Scaler.fit(aug)
    batch = scaler.transform(aug, panel, panel.target, torch.device("cpu"))
    rows = torch.arange(len(panel.target))
    inverse, width, count = v2._segments(batch["codes"][rows])
    w = v2._meeting_weights(batch, rows, inverse, width, count, cfg, scaler.s2_mean)
    assert w.shape == (width,)
    assert float(w.min()) >= 1.0 and float(w.max()) <= 3.5 + 1e-6

    s_abs = v2._segment_mean(batch["shock_1d"].double(), inverse, width, count).abs()
    order = torch.argsort(s_abs)
    assert torch.all(w[order][1:] >= w[order][:-1] - 1e-9)



@pytest.mark.parametrize("slope", [False, True])
def test_sector_two_shock_recovers_sector_loadings_and_keeps_the_b_contract(slope):
    import dataclasses

    from fedcore.q3.contracts import BETA63
    from fedcore.q3.shocks import split_shock
    from fedenc.workflow import GICS_SECTORS

    panel = make_synthetic_panel(n_firms=44, n_meetings=24, noise="easy", seed=3)
    meetings = panel.meetings()
    rng = np.random.default_rng(3)
    table = pd.DataFrame({"sp500": rng.normal(0, 0.5, len(meetings)), "slope": rng.normal(0, 0.4, len(meetings))},
                         index=meetings)
    n, k = len(panel.target), len(GICS_SECTORS)
    code = rng.integers(0, k, n)
    D = np.eye(k, dtype=np.float32)[code]
    beta = rng.uniform(0.5, 1.5, n)
    hist = panel.history.copy()
    hist[:, :, BETA63] = beta[:, None]
    panel = dataclasses.replace(panel, history=hist, fundamentals=np.concatenate([panel.fundamentals, D], axis=1),
                                fundamentals_mask=np.concatenate([panel.fundamentals_mask, D == D], axis=1))
    S = split_shock(panel, "policy_info", table)
    sl = split_shock(panel, "target_slope", table)[:, 1]
    g_p = np.linspace(-0.5, 0.5, k)
    target = 0.1 + beta * (0.3 - 0.9 * S[:, 0] + 0.6 * S[:, 1]) + g_p[code] * S[:, 0] + (0.4 * sl if slope else 0)
    panel = dataclasses.replace(panel, target=target.astype(np.float32))
    model = v2.SectorTwoShock(slope=slope, table=table)
    model.fit(panel)
    mu, a, b = model.predict(panel)
    np.testing.assert_allclose(mu, target, atol=1e-3)
    np.testing.assert_allclose(mu, a + b * panel.shock, atol=1e-6)


@pytest.mark.parametrize("vol", [False, True])
def test_capm_tilt_recovers_a_context_scaled_beta_drift(vol):
    import dataclasses

    from fedcore.q3.contracts import BETA63

    panel = make_synthetic_panel(n_firms=40, n_meetings=30, noise="easy", seed=4)
    rng = np.random.default_rng(4)
    beta = rng.uniform(0.5, 1.5, len(panel.target))
    hist = panel.history.copy()
    hist[:, :, BETA63] = beta[:, None]
    panel = dataclasses.replace(panel, history=hist)
    model = v2.CAPMTilt(vol=vol)
    model._design(panel, fit=True)
    z = np.clip(np.nan_to_num((panel.context - model.med_) / model.iqr_), -5, 5)
    s = np.asarray(panel.shock, dtype=np.float64)
    target = 0.1 + beta * (0.3 + 0.5 * z[:, 0] - 0.8 * s)
    panel = dataclasses.replace(panel, target=target.astype(np.float32))
    model.fit(panel)
    mu, a, b = model.predict(panel)
    np.testing.assert_allclose(mu, target, atol=1e-4)
    np.testing.assert_allclose(mu, a + b * s, atol=1e-9)
    np.testing.assert_allclose(b, -0.8 * beta, atol=1e-4)


def test_capm_tilt_ridge_shrinks_to_capm_when_the_tilt_is_noise():
    from fedcore.q3.baselines import CAPMPooled

    panel = make_synthetic_panel(n_firms=40, n_meetings=30, seed=5)
    tilt, capm = v2.CAPMTilt(ridge=True), CAPMPooled()
    tilt.fit(panel)
    capm.fit(panel)
    mu, a, b = tilt.predict(panel)
    np.testing.assert_allclose(mu, a + b * panel.shock, atol=1e-9)
    assert tilt.alpha_ in v2.CAPMTilt.ALPHAS and abs(tilt.k_ - capm.k_) < 1e-9


@pytest.mark.parametrize("both", [False, True])
def test_capm_betas_uses_the_window_average_beta_and_keeps_the_b_contract(both):
    import dataclasses

    from fedcore.q3.contracts import BETA63

    panel = make_synthetic_panel(n_firms=30, n_meetings=20, noise="easy", seed=6)
    rng = np.random.default_rng(6)
    n = len(panel.target)
    true = rng.uniform(0.5, 1.5, n)
    hist = panel.history.copy()
    hist[:, :, BETA63] = true[:, None] + rng.normal(0, 0.3, (n, hist.shape[1]))  # noisy daily estimates
    hist[:, -1, BETA63] = true + rng.normal(0, 0.5, n)                           # an extra-noisy last value
    s = np.asarray(panel.shock, dtype=np.float64)
    panel = dataclasses.replace(panel, history=hist, target=(0.2 + true * (0.3 - 0.9 * s)).astype(np.float32))
    model = v2.CAPMBetas(both=both)
    model.fit(panel)
    mu, a, b = model.predict(panel)
    np.testing.assert_allclose(mu, a + b * s, atol=1e-9)
    from fedcore.q3.baselines import CAPMPooled
    capm = CAPMPooled()
    capm.fit(panel)
    t = np.asarray(panel.target, dtype=np.float64)
    assert np.mean((t - mu) ** 2) < 0.5 * np.mean((t - capm.predict(panel)[0]) ** 2)
