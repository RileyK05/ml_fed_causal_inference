"""Mean/cross-firm loss split and the centered-a model. Smoke fits only."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fedcore.q3.synthetic import make_synthetic_panel
from fedcore.q3.train import ChannelScaler, _meeting_index_groups, _tensors, per_meeting_mse
from fedenc import decouple as dc
from fedenc.encoder_transformer import PatchTransformerEncoder


def tiny_encoder():
    return PatchTransformerEncoder(d_model=8, n_layers=1, n_heads=2, dim_feedforward=16, dropout=0)


@pytest.fixture
def panel():
    return make_synthetic_panel(n_firms=6, n_meetings=8, seed=0)


def test_loss_parts_add_up_to_per_meeting_mse():
    g = torch.Generator().manual_seed(0)
    mu, y = torch.randn(30, generator=g), torch.randn(30, generator=g)
    codes = torch.tensor([0] * 10 + [1] * 12 + [2] * 8)
    m, c = dc.meeting_losses(mu, y, codes)
    assert (m + c).mean().item() == pytest.approx(per_meeting_mse(mu, y, codes).item(), rel=1e-6)
    # A prediction off by a constant within a meeting only costs the mean part.
    m2, c2 = dc.meeting_losses(y + 0.7, y, codes)
    assert torch.allclose(c2, torch.zeros(3), atol=1e-6) and torch.allclose(m2, torch.full((3,), 0.49), atol=1e-5)


def test_chunks_keep_meetings_whole():
    groups = [np.arange(k * 5, k * 5 + 5) for k in range(7)]
    chunks = dc._chunks(groups, limit=12)
    assert np.array_equal(np.concatenate(chunks), np.arange(35))
    assert all(len(ch) % 5 == 0 for ch in chunks)


def test_centered_a_cannot_move_the_meeting_average_and_ignores_context(panel):
    torch.manual_seed(0)
    model = dc.StructModel(tiny_encoder(), n_fund=panel.fundamentals.shape[1], n_ctx=panel.context.shape[1])
    with torch.no_grad():
        model.a0.fill_(0.25)
    model.eval()
    batch = _tensors(panel, ChannelScaler.fit(panel), torch.device("cpu"))
    rows = torch.arange(len(panel.target))
    mu, a, b = dc.outputs(model, batch, rows)
    for idx in _meeting_index_groups(panel.meeting):
        assert a[idx].mean().item() == pytest.approx(0.25, abs=1e-5)
    shifted = dict(batch, context=batch["context"] + 3.0)
    _, a2, b2 = dc.outputs(model, shifted, rows)
    assert torch.allclose(a, a2) and not torch.allclose(b, b2)
    assert torch.allclose(mu, a + b * batch["shock"])


@pytest.mark.parametrize("centered_a,lambda_mean", [(False, 1.0), (False, 0.05), (True, 1.0), (True, 0.05)])
def test_decouple_arm_smoke(panel, monkeypatch, centered_a, lambda_mean):
    monkeypatch.setattr(dc, "PatchTransformerEncoder", lambda: tiny_encoder())
    arm = dc.DecoupleArm(0, centered_a=centered_a, lambda_mean=lambda_mean, device="cpu", max_epochs=2, patience=1)
    arm.fit(panel)
    mu, a, b = arm.predict(panel)
    assert np.isfinite(mu).all()
    np.testing.assert_allclose(mu, a + b * panel.shock, atol=1e-5)
    assert 1 <= arm.fit_result.best_epoch <= 2


def test_two_part_keeps_pooled_meeting_mean_and_network_deviations(panel, monkeypatch):
    import pandas as pd
    from fedcore.q3.baselines import PooledOLS

    monkeypatch.setattr(dc, "PatchTransformerEncoder", lambda: tiny_encoder())
    arm = dc.TwoPartArm(0, members=2, device="cpu", max_epochs=2, patience=1)
    arm.fit(panel)
    mu, a, b = arm.predict(panel)
    np.testing.assert_allclose(mu, a + b * panel.shock, atol=1e-9)
    pooled = PooledOLS()
    pooled.fit(panel)
    meeting = pd.DatetimeIndex(panel.meeting)
    np.testing.assert_allclose(pd.Series(mu).groupby(meeting).mean().to_numpy(),
                               pd.Series(pooled.predict(panel)[0]).groupby(meeting).mean().to_numpy(), atol=1e-9)
    nets = [n.predict(panel)[0] for n in arm.nets]
    dev = np.mean([m - pd.Series(m).groupby(meeting).transform("mean").to_numpy() for m in nets], axis=0)
    np.testing.assert_allclose(mu - pd.Series(mu).groupby(meeting).transform("mean").to_numpy(), dev, atol=1e-6)
    assert [n.seed for n in arm.nets] == [0, 1]


def test_reference_objective_is_the_shared_meeting_mse(panel):
    from fedcore.q3 import ResponseModel
    from fedcore.q3.train import _eval_loss

    torch.manual_seed(0)
    model = ResponseModel(tiny_encoder(), n_fund=panel.fundamentals.shape[1], n_ctx=panel.context.shape[1])
    batch = _tensors(panel, ChannelScaler.fit(panel), torch.device("cpu"))
    objective, m, c = dc.evaluate(model, batch, _meeting_index_groups(panel.meeting), 1.0)
    assert objective == pytest.approx(_eval_loss(model, batch, False), rel=1e-5)
    assert objective == pytest.approx(m + c)
