"""Unit tests for fedenc.v4 (projection, two-stage fit, saved components). Tiny synthetic fits only."""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from fedcore.q3.contracts import BETA63
from fedcore.q3.evaluate import run_walk_forward
from fedcore.q3.synthetic import make_synthetic_panel
from fedenc import v3, v4
from fedenc.encoder_transformer import PatchTransformerEncoder
from fedenc.v2 import CAPMBetas


def tiny(n_channels=5):
    return PatchTransformerEncoder(n_channels=n_channels, d_model=8, n_layers=1, n_heads=2, dim_feedforward=16,
                                   dropout=0.0)


@pytest.fixture
def panel():
    p = make_synthetic_panel(n_firms=14, n_meetings=12, seed=0)
    rng = np.random.default_rng(0)
    hist = p.history.copy()
    hist[:, :, BETA63] = rng.uniform(0.5, 1.5, (len(p.target), 1)) + rng.normal(0, 0.2, hist.shape[:2])
    s = np.asarray(p.shock, dtype=np.float32).copy()
    s[pd.DatetimeIndex(p.meeting) == p.meetings()[-2]] = 0.0  # one test-side meeting with no surprise
    from fedcore.q3.real import FUNDAMENTAL_NAMES

    k = len(FUNDAMENTAL_NAMES) - p.fundamentals.shape[1]  # pad to the real panel's six fundamentals
    fund = np.column_stack((p.fundamentals, rng.normal(size=(len(p.target), k)))).astype(np.float32)
    mask = np.column_stack((p.fundamentals_mask, np.ones((len(p.target), k), dtype=bool)))
    return dataclasses.replace(p, history=hist, shock=s, fundamentals=fund, fundamentals_mask=mask)


@pytest.fixture
def sector_map(panel):
    rng = np.random.default_rng(1)
    firms = np.unique(panel.firm_id)
    sec = dict(zip(firms, rng.choice(["10", "20", "40"], len(firms))))
    rows = pd.DataFrame({"meeting": pd.DatetimeIndex(panel.meeting), "permno": panel.firm_id.astype(int),
                         "gsector": [sec[f] for f in panel.firm_id]})
    return v4.SectorMap(rows)


def _check_projected(panel, v):
    q = v3.basis_rows(panel, "beta")
    b63, bw = v3.base_betas(panel)
    for m in panel.meetings():
        r = pd.DatetimeIndex(panel.meeting) == m
        for basis in (np.ones(r.sum()), b63[r], bw[r]):
            assert abs(v[r] @ basis) < 1e-8
    return q


def test_project_columns_matches_the_v3_projection(panel):
    rng = np.random.default_rng(2)
    V = rng.normal(size=(len(panel.target), 3))
    codes = v4._codes(panel)
    q = v3.basis_rows(panel, "beta")
    PV = v4.project_columns(V, q, codes)
    qt, ct = torch.as_tensor(q), torch.as_tensor(codes)
    for k in range(3):
        ref = v3.project(torch.as_tensor(V[:, k]), qt, ct, int(codes.max()) + 1).numpy()
        np.testing.assert_allclose(PV[:, k], ref, atol=1e-12)
        _check_projected(panel, PV[:, k])


def test_peer_stress_is_leave_one_out():
    own = np.array([1.0, 2.0, 6.0, 4.0, np.nan])
    sec = np.array(["a", "a", "a", "b", "a"], dtype=object)
    out = v4.peer_stress(own, sec, np.zeros(5, dtype=int))
    np.testing.assert_allclose(out[:3], [4.0, 3.5, 1.5])
    assert np.isnan(out[3]) and np.isnan(out[4])  # alone in its sector / own value missing


@pytest.mark.parametrize("kw", [dict(features="all"), dict(), dict(sector=True), dict(sector=True, peer=True)])
def test_ridge_b_only_contract(panel, sector_map, kw):
    model = v4.V4Ridge(sector_map=sector_map, **kw)
    model.fit(panel)
    mu, a, b = model.predict(panel)
    ex = model.prediction_extras(panel)
    s = np.asarray(panel.shock, dtype=np.float64)
    np.testing.assert_allclose(mu, a + b * s, atol=1e-9)
    np.testing.assert_allclose(mu, ex["mu_base"] + ex["delta_a"] + ex["delta_b"] * s, atol=1e-9)
    _check_projected(panel, ex["delta_b"])
    zero = s == 0
    np.testing.assert_allclose(mu[zero], ex["mu_base"][zero], atol=1e-12)  # no surprise: exactly the base
    base = CAPMBetas(252, both=True)
    base.fit(panel)
    np.testing.assert_allclose(ex["mu_base"], base.predict(panel)[0], atol=1e-10)  # stage 2 refits the base


@pytest.mark.parametrize("b_only", [True, False])
def test_neural_two_stage_contract(panel, b_only):
    arm = v4.V4Arm(0, encoder="transformer", b_only=b_only, members=2, max_epochs=2, patience=1, device="cpu",
                   encoder_fn=tiny)
    arm.fit(panel)
    mu, a, b = arm.predict(panel)
    ex = arm.prediction_extras(panel)
    s = np.asarray(panel.shock, dtype=np.float64)
    np.testing.assert_allclose(mu, a + b * s, atol=1e-9)
    np.testing.assert_allclose(mu, ex["mu_base"] + ex["delta_a"] + ex["delta_b"] * s, atol=1e-9)
    _check_projected(panel, ex["delta_b"])
    if b_only:
        assert np.all(ex["delta_a"] == 0.0)
        np.testing.assert_allclose(mu[s == 0], ex["mu_base"][s == 0], atol=1e-12)
    sel = arm.fit_result.member_selection
    steps_per_epoch = int(np.ceil(len(arm.fit_result.val_meetings) and
                                  (len(panel.meetings()) - len(arm.fit_result.val_meetings)) / arm.meetings_per_batch))
    assert len(sel) == 2 and all(m["steps"] == m["epoch"] * steps_per_epoch for m in sel)
    base = CAPMBetas(252, both=True)
    base.fit(panel)
    np.testing.assert_allclose(ex["mu_base"], base.predict(panel)[0], atol=1e-10)


def test_zero_selected_epochs_reproduce_the_base(panel):
    arm = v4.V4Arm(0, members=1, max_epochs=0, device="cpu", encoder_fn=tiny)
    arm.fit(panel)
    mu, _, _ = arm.predict(panel)
    np.testing.assert_allclose(mu, arm.base.predict(panel)[0], atol=1e-12)


def test_inner_validation_labels_do_not_move_the_stage_one_base(panel):
    from fedcore.q3.train import validation_meetings

    val = validation_meetings(panel, 0.15)
    y = np.asarray(panel.target, dtype=np.float64).copy()
    y[np.isin(pd.DatetimeIndex(panel.meeting), val)] += 50.0
    coefs = []
    for target in (panel.target, y):
        arm = v4.V4Arm(0, members=1, max_epochs=0, device="cpu", encoder_fn=tiny)
        arm.fit(dataclasses.replace(panel, target=np.asarray(target, dtype=np.float32)))
        coefs.append(arm.base_inner_.coef_)
    np.testing.assert_allclose(coefs[0], coefs[1], atol=1e-12)


def test_walk_forward_saves_components_and_member_selection(panel):
    res = run_walk_forward(panel, lambda seed: v4.V4Arm(seed, members=2, max_epochs=1, patience=1, device="cpu",
                                                         encoder_fn=tiny), min_train=8, test_size=2, seeds=(0,))
    for c in ("mu_base", "delta_a", "delta_b"):
        assert c in res.predictions.columns
    assert res.folds["member_selection"].str.contains("steps").all()
    res0 = run_walk_forward(panel, lambda seed: CAPMBetas(252, both=True), min_train=8, test_size=2, seeds=(0,))
    assert "mu_base" not in res0.predictions.columns  # models without extras are unchanged


def test_panel_fingerprint_changes_with_the_target(panel):
    a = v4.panel_fingerprint(panel)
    b = v4.panel_fingerprint(dataclasses.replace(panel, target=panel.target + 1))
    assert a != b and a == v4.panel_fingerprint(panel)
