"""Tests for the signature arm and the Neural CDE encoders.

The signature test is the hand-computed 2-D example from the brief: depth-2 terms are the
increments plus the Levy areas. The encoder tests check shapes, NaN safety with missing
days and left padding, and mask invariance.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from fedcore.q3 import make_synthetic_panel

from fedcde.encoder_cde import CDELogSigEncoder, NeuralCDEEncoder, build_path
from fedcde.signature import SigRidge, truncated_signature

pytest.importorskip("torch")


def _tiny_panel(n_firms=8, n_meetings=4, window=None):
    panel = make_synthetic_panel(n_firms=n_firms, n_meetings=n_meetings, noise="easy", seed=0)
    x = torch.from_numpy(np.ascontiguousarray(panel.history)).float()
    mask = torch.from_numpy(panel.history_mask)
    return panel, x, mask


def test_signature_hand_computed():
    # Path (0,0)->(1,0)->(1,2)->(3,2). Increments dX0=[0,1,0,2], dX1=[0,0,2,0].
    # depth1 = [3, 2]; depth2 = [S00, S01, S10, S11] = [2, 2, 4, 0];
    # depth3 (i,j,k) k fastest = [S000, S001, S010, S011, S100, S101, S110, S111]
    #                 = [0, 0, 4, 0, 0, 0, 0, 0].
    X = np.array([[[0.0, 0.0], [1.0, 0.0], [1.0, 2.0], [3.0, 2.0]]])
    sig = truncated_signature(X, depth=3)
    expected = np.array([[3.0, 2.0, 2.0, 2.0, 4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    np.testing.assert_allclose(sig, expected, rtol=1e-10, atol=1e-10)


def test_signature_levy_area_identity():
    # For a 2-D path, S^{i,j} + S^{j,i} = dX^i dX^j - sum_t dX^i_t dX^j_t.
    rng = np.random.default_rng(0)
    X = np.cumsum(rng.normal(size=(5, 20, 2)), axis=1)
    sig = truncated_signature(X, depth=2)
    d = np.diff(X, axis=1, prepend=X[:, :1, :])
    for b in range(5):
        dx0, dx1 = d[b, :, 0].sum(), d[b, :, 1].sum()
        diag = (d[b, :, 0] * d[b, :, 1]).sum()
        s01, s10 = sig[b, 3], sig[b, 4]
        assert s01 + s10 == pytest.approx(dx0 * dx1 - diag, abs=1e-8)


def test_signature_nan_increments_zero():
    # A NaN in the path contributes no increment; the signature stays finite.
    X = np.array([[[0.0, 0.0], [1.0, np.nan], [1.0, 2.0], [3.0, 2.0]]])
    sig = truncated_signature(X, depth=3)
    assert np.isfinite(sig).all()


def test_build_path_time_and_obs():
    _, x, mask = _tiny_panel()
    path = build_path(x, mask, window=126)
    assert path.shape == (32, 126, 7)
    time_ch = path[:, :, -2]
    obs_ch = path[:, :, -1]
    assert bool((time_ch[:, 1:] >= time_ch[:, :-1] - 1e-6).all())
    inc = obs_ch[:, 1:] - obs_ch[:, :-1]
    assert bool((inc >= -1e-6).all())
    # obs count increases only on observed days
    assert bool(((inc > 1e-6) <= mask[:, -126:][:, 1:]).all())
    # data channels are NaN exactly where unobserved
    data = path[:, :, :5]
    assert bool((torch.isnan(data) == (~mask[:, -126:].unsqueeze(-1))).all())


def test_encoder_shapes_and_finite():
    _, x, mask = _tiny_panel()
    for enc in (
        NeuralCDEEncoder(n_channels=5, window=126),
        CDELogSigEncoder(n_channels=5, window=126, window_size=10),
    ):
        enc.eval()
        with torch.no_grad():
            h = enc(x, mask)
        assert h.shape == (32, 64)
        assert bool(torch.isfinite(h).all())


def test_encoder_left_padding_finite():
    # Early meetings are left-padded (firms listed late); h must stay finite.
    panel, x, mask = _tiny_panel(n_firms=6, n_meetings=3)
    enc = NeuralCDEEncoder(n_channels=5, window=126)
    enc.eval()
    with torch.no_grad():
        h = enc(x, mask)
    assert bool(torch.isfinite(h).all())
    # the first meeting has the most padding
    assert not bool(mask[:6].all())


def test_mask_invariance():
    _, x, mask = _tiny_panel()
    for enc in (
        NeuralCDEEncoder(n_channels=5, window=126),
        CDELogSigEncoder(n_channels=5, window=126, window_size=10),
    ):
        enc.eval()
        with torch.no_grad():
            h1 = enc(x, mask)
        x2 = x.clone()
        x2[~mask] = torch.randn_like(x2[~mask]) * 1000.0
        with torch.no_grad():
            h2 = enc(x2, mask)
        assert bool((h1 - h2).abs().max() <= 1e-5)


@pytest.mark.parametrize(
    "encoder",
    [
        NeuralCDEEncoder(n_channels=5, window=126, add_time=False),
        NeuralCDEEncoder(n_channels=5, window=126, add_obs_count=False),
        NeuralCDEEncoder(n_channels=5, window=126, add_time=False, add_obs_count=False),
        NeuralCDEEncoder(n_channels=5, window=126, interpolation="linear"),
        NeuralCDEEncoder(n_channels=5, window=126, hidden=16),
        CDELogSigEncoder(n_channels=5, window=126, add_time=False),
        CDELogSigEncoder(n_channels=5, window=126, add_obs_count=False),
        CDELogSigEncoder(n_channels=5, window=126, add_time=False, add_obs_count=False),
        CDELogSigEncoder(n_channels=5, window=126, window_size=21),
        CDELogSigEncoder(n_channels=5, window=126, hidden=16),
    ],
)
def test_encoder_ablation_configs_mask_invariant(encoder):
    _, x, mask = _tiny_panel()
    encoder.eval()
    with torch.no_grad():
        h1 = encoder(x, mask)
    assert h1.shape == (32, 64) and bool(torch.isfinite(h1).all())
    x2 = x.clone()
    x2[~mask] = torch.randn_like(x2[~mask]) * 1000.0
    with torch.no_grad():
        h2 = encoder(x2, mask)
    assert bool((h1 - h2).abs().max() <= 1e-5)


def test_sig_ridge_fit_predict():
    panel = make_synthetic_panel(n_firms=20, n_meetings=30, noise="easy", seed=1)
    arm = SigRidge(seed=0, window=126, short_window=63)
    arm.fit(panel)
    mu, a, b = arm.predict(panel)
    assert mu.shape == a.shape == b.shape == (len(panel.target),)
    assert np.isfinite(mu).all() and np.isfinite(b).all()
    # a_hat + b_hat * s must equal mu
    s = np.asarray(panel.shock, dtype=np.float64)
    np.testing.assert_allclose(mu, a + b * s, rtol=1e-8, atol=1e-8)


def test_sig_ridge_recovers_b():
    # On easy noise the signature ridge should recover the planted b better than chance.
    panel = make_synthetic_panel(n_firms=40, n_meetings=60, noise="easy", seed=2)
    arm = SigRidge(seed=0, window=126, short_window=63)
    arm.fit(panel)
    _, _, b = arm.predict(panel)
    b_true = np.asarray(panel.b_true, dtype=np.float64)
    corr = np.corrcoef(b, b_true)[0, 1]
    assert corr > 0.5


def test_fill_nan_matches_torchcde():
    import torchcde

    from fedcde.encoder_cde import _fill_nan

    torch.manual_seed(0)
    x = torch.randn(6, 40, 5)
    mask = torch.rand(6, 40) > 0.4
    mask[0] = False  # all-masked row
    mask[1, :5] = False  # leading gap
    mask[2, -5:] = False  # trailing gap
    path = build_path(x, mask, window=40)
    got = _fill_nan(path, mask)
    assert torch.isfinite(got).all()
    ref = torchcde.linear_interpolation_coeffs(path[1:])
    assert torch.allclose(got[1:], ref, atol=1e-5)
    # time channel passes through on unobserved days; hand-checked small case
    assert torch.equal(got[1:, :, -2], path[1:, :, -2])
    p = torch.tensor([[[5.0], [1.0], [float("nan")], [3.0], [float("nan")]]])
    assert _fill_nan(p, torch.tensor([[True, True, False, True, False]])).flatten().tolist() == [5, 1, 2, 3, 3]
    lead = torch.tensor([[[float("nan")], [2.0], [4.0]]])
    assert _fill_nan(lead, torch.tensor([[False, True, True]])).flatten().tolist() == [2, 2, 4]
