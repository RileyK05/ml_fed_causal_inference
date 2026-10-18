"""Unit tests for fedenc.v3 (projection, review fixes, smoke fits). Tiny synthetic fits only."""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from fedcore.q3.contracts import BETA63
from fedcore.q3.synthetic import make_synthetic_panel
from fedenc import v3
from fedenc.encoder_transformer import PatchTransformerEncoder


def tiny(n_channels=5):
    return PatchTransformerEncoder(n_channels=n_channels, d_model=8, n_layers=1, n_heads=2, dim_feedforward=16,
                                   dropout=0.0)


@pytest.fixture
def panel():
    p = make_synthetic_panel(n_firms=12, n_meetings=10, seed=0)
    rng = np.random.default_rng(0)
    hist = p.history.copy()
    hist[:, :, BETA63] = rng.uniform(0.5, 1.5, (len(p.target), 1)) + rng.normal(0, 0.2, hist.shape[:2])
    return dataclasses.replace(p, history=hist)


def _project_np(panel, kind, v):
    q = torch.as_tensor(v3.basis_rows(panel, kind))
    codes = torch.as_tensor(pd.factorize(pd.DatetimeIndex(panel.meeting))[0])
    inverse, width = codes, int(codes.max()) + 1
    return v3.project(torch.as_tensor(v, dtype=torch.float64), q, inverse, width).numpy(), codes.numpy()


@pytest.mark.parametrize("kind", ["centered", "beta"])
def test_projection_is_exact_and_idempotent(panel, kind):
    rng = np.random.default_rng(1)
    v = rng.normal(size=len(panel.target))
    pv, codes = _project_np(panel, kind, v)
    ppv, _ = _project_np(panel, kind, pv)
    np.testing.assert_allclose(ppv, pv, atol=1e-10)
    b63, bw = v3.base_betas(panel)
    for m in np.unique(codes):
        r = codes == m
        assert abs(pv[r].sum()) < 1e-9  # the meeting mean is always removed
        if kind == "beta":
            assert abs(pv[r] @ b63[r]) < 1e-9 and abs(pv[r] @ bw[r]) < 1e-9


def test_projection_handles_collinear_betas(panel):
    hist = panel.history.copy()
    hist[:, :, BETA63] = hist[:, -1:, BETA63]  # every day the same: beta_smooth == beta63
    p = dataclasses.replace(panel, history=hist)
    q = v3.basis_rows(p, "beta")
    for rows in [np.flatnonzero(pd.DatetimeIndex(p.meeting) == m) for m in p.meetings()]:
        assert np.linalg.matrix_rank(q[rows]) == 2  # one direction dropped, the rest orthonormal
        np.testing.assert_allclose(q[rows].T @ q[rows], np.diag([1.0, 1.0, 0.0]), atol=1e-10)


def test_zero_correction_predicts_the_inner_training_base(panel):
    arm = v3.V3Arm(0, basis="beta", members=1, max_epochs=0, encoder_fn=tiny, device="cpu")
    arm.fit(panel)
    mu, a, b = arm.predict(panel)
    mu0, a0, b0 = arm.base.predict(panel)
    np.testing.assert_allclose(mu, mu0, atol=1e-12)
    np.testing.assert_allclose(mu, a + b * panel.shock, atol=1e-9)


def test_inner_validation_labels_do_not_move_the_base(panel):
    from fedcore.q3.train import validation_meetings

    val = validation_meetings(panel, 0.15)
    y = np.asarray(panel.target, dtype=np.float64).copy()
    y[np.isin(pd.DatetimeIndex(panel.meeting), val)] += 50.0
    arms = []
    for target in (panel.target, y):
        arm = v3.V3Arm(0, members=1, max_epochs=0, encoder_fn=tiny, device="cpu")
        arm.fit(dataclasses.replace(panel, target=np.asarray(target, dtype=np.float32)))
        arms.append(arm)
    np.testing.assert_allclose(arms[0].base.coef_, arms[1].base.coef_, atol=1e-12)


def test_deployed_loss_ignores_outputs_inside_the_projected_span(panel):
    arm = v3.V3Arm(0, basis="beta", members=1, max_epochs=0, encoder_fn=tiny, device="cpu")
    arm.fit(panel)
    d = arm._data(panel, torch.device("cpu"))
    model = arm.nets[0]
    rows = torch.arange(len(panel.target))
    with torch.no_grad():
        model.head_a.bias.fill_(0.3)
        model.head_b.bias.fill_(-0.2)
        base = v3._deployed_loss(model, d["x"], rows, d["q"], d["codes"], d["s"], d["e"]).item()
        model.head_a.bias.fill_(5.0)  # a meeting-wide shift of a: removed by every projection
        moved = v3._deployed_loss(model, d["x"], rows, d["q"], d["codes"], d["s"], d["e"]).item()
    assert abs(base - moved) < 1e-9


@pytest.mark.parametrize("encoder,basis", [("summary", "beta"), ("transformer", "centered"), ("transformer", "beta")])
def test_v3_arms_smoke_fit(panel, encoder, basis):
    arm = v3.V3Arm(0, encoder=encoder, basis=basis, members=2, max_epochs=2, patience=1, device="cpu",
                   encoder_fn=None if encoder == "summary" else tiny)
    arm.fit(panel)
    mu, a, b = arm.predict(panel)
    assert np.isfinite(mu).all() and len(arm.fit_result.member_best_epochs) == 2
    np.testing.assert_allclose(mu, a + b * panel.shock, atol=1e-9)
    mu0, _, _ = arm.base.predict(panel)
    m = pd.DatetimeIndex(panel.meeting)
    np.testing.assert_allclose(pd.Series(mu - mu0).groupby(m).mean().to_numpy(), 0.0, atol=1e-9)


@pytest.mark.parametrize("ranks,recency", [(False, False), (False, True), (True, False)])
def test_residual_ridge_contract_and_centering(panel, ranks, recency):
    model = v3.ResidualRidge(ranks=ranks, recency=recency)
    model.fit(panel)
    mu, a, b = model.predict(panel)
    np.testing.assert_allclose(mu, a + b * panel.shock, atol=1e-9)
    mu0, _, _ = model.base.predict(panel)
    m = pd.DatetimeIndex(panel.meeting)
    np.testing.assert_allclose(pd.Series(mu - mu0).groupby(m).mean().to_numpy(), 0.0, atol=1e-9)
    assert model.alpha_ in v3.ResidualRidge.ALPHAS


def test_recency_weights_halve_every_five_years():
    m = pd.DatetimeIndex(["2011-01-01", "2016-01-01", "2021-01-01"])
    w = v3.recency_weights(m, pd.Timestamp("2021-01-01"))
    np.testing.assert_allclose(w / w[-1], [0.25, 0.5, 1.0], atol=2e-3)
