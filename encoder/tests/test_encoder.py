"""Unit tests for the patch-transformer encoder."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fedcore.q3 import ResponseModel
from fedcore.q3.contracts import L
from fedcore.q3.synthetic import make_synthetic_panel

from fedenc.encoder_transformer import PatchTransformerEncoder


def _random_history(n=3, t=L, c=5, p_obs=0.85, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x = torch.randn(n, t, c)
    mask = torch.rand(n, t) < p_obs
    mask[:, -5:] = True
    return x, mask


def test_output_shape_and_patch_count():
    enc = PatchTransformerEncoder()
    x, mask = _random_history()
    enc.eval()
    with torch.no_grad():
        h = enc(x, mask)
        tokens, valid = enc.forward_tokens(x, mask)
    assert enc.out_dim == 64
    assert h.shape == (3, 64)
    assert tokens.shape == (3, 51, 64)
    assert valid.shape == (3, 51)
    assert bool(valid[:, -1].all())  # the last patch covers the forced-observed tail
    assert bool(valid.any(dim=1).all())
    patches, pvalid = enc.patchify(x, mask)
    assert patches.shape == (3, 51, 30)
    assert pvalid.shape == (3, 51)


def test_mask_invariance():
    enc = PatchTransformerEncoder()
    enc.eval()
    x, mask = _random_history(n=4, seed=1)
    rng = np.random.default_rng(2)
    x2 = x.clone()
    garbage = torch.from_numpy(rng.normal(0.0, 50.0, size=x.shape)).to(x.dtype)
    x2 = torch.where(mask.unsqueeze(-1), x2, garbage)
    x2[~mask] = torch.nan
    with torch.no_grad():
        h1 = enc(x, mask)
        h2 = enc(x2, mask)
    assert torch.allclose(h1, h2, atol=1e-5)
    assert torch.isfinite(h1).all()


def test_padding_invariance():
    enc = PatchTransformerEncoder()
    enc.eval()
    rng = np.random.default_rng(3)
    t = 200
    x = torch.randn(2, t, 5)
    mask = torch.ones(2, t, dtype=torch.bool)
    with torch.no_grad():
        h_short = enc(x, mask)
        for extra in (52, 55, 53):
            pad_x = torch.from_numpy(rng.normal(0.0, 25.0, size=(2, t + extra, 5))).float()
            pad_x[:, extra:] = x
            pad_mask = torch.zeros(2, t + extra, dtype=torch.bool)
            pad_mask[:, extra:] = True
            h_pad = enc(pad_x, pad_mask)
            assert torch.allclose(h_short, h_pad, atol=1e-5), f"extra={extra}"


def test_no_nan_with_partial_patches_and_zero_rows():
    enc = PatchTransformerEncoder()
    enc.eval()
    x, mask = _random_history(n=6, p_obs=0.5, seed=4)
    with torch.no_grad():
        h = enc(x, mask)
    assert torch.isfinite(h).all()
    empty_mask = torch.zeros(2, L, dtype=torch.bool)
    with torch.no_grad():
        h0 = enc(torch.randn(2, L, 5), empty_mask)
    assert torch.isfinite(h0).all()
    assert torch.equal(h0, torch.zeros_like(h0))


def test_synthetic_rows_always_have_a_valid_patch():
    panel = make_synthetic_panel(n_firms=20, n_meetings=25, noise="easy", seed=0)
    enc = PatchTransformerEncoder()
    x = torch.from_numpy(np.nan_to_num(panel.history))
    mask = torch.from_numpy(panel.history_mask)
    _, valid = enc.patchify(x, mask)
    assert bool(valid.any(dim=1).all())


def test_response_model_integration():
    enc = PatchTransformerEncoder()
    model = ResponseModel(enc, n_fund=3, n_ctx=4)
    x, mask = _random_history(n=5, seed=6)
    shock = torch.randn(5)
    mu, a, b = model(
        x,
        mask,
        torch.randn(5, 3),
        torch.ones(5, 3, dtype=torch.bool),
        torch.randn(5, 4),
        shock,
    )
    for part in (mu, a, b):
        assert part.shape == (5,)
        assert torch.isfinite(part).all()
    assert torch.allclose(mu, a + b * shock, atol=1e-6)


def test_attention_is_a_distribution():
    enc = PatchTransformerEncoder()
    enc.eval()
    x, mask = _random_history(n=2, seed=7)
    with torch.no_grad():
        att = enc.attention(x, mask, layer=-1)
    assert att.shape == (2, 4, 51, 51)
    assert torch.allclose(att.sum(dim=-1), torch.ones(2, 4, 51), atol=1e-4)


def test_pool_and_patch_len_variants():
    for pool in ("mean", "last", "cls"):
        enc = PatchTransformerEncoder(pool=pool)
        x, mask = _random_history(n=2, seed=8)
        enc.eval()
        with torch.no_grad():
            h = enc(x, mask)
            tokens, valid = enc.forward_tokens(x, mask)
        assert h.shape == (2, 64)
        assert tokens.shape == (2, 51, 64)
    enc = PatchTransformerEncoder(patch_len=10)
    x, mask = _random_history(n=2, seed=9)
    with torch.no_grad():
        h = enc(x, mask)
    assert h.shape == (2, 64)
    assert enc.n_patches == 26


def test_masked_tokens_retain_positions_without_exposing_hidden_values():
    torch.manual_seed(0)
    enc = PatchTransformerEncoder(dropout=0)
    enc.eval()
    x, mask = _random_history(n=1)
    choice = torch.zeros(1, enc.n_patches, dtype=torch.bool)
    choice[:, -2:] = True
    changed = x.clone()
    changed[:, -10:] = 10000
    with torch.no_grad():
        tokens, _ = enc.forward_tokens(x, mask, choice)
        other, _ = enc.forward_tokens(changed, mask, choice)
    assert torch.allclose(tokens, other, atol=1e-6)
    assert not torch.allclose(tokens[:, -1], tokens[:, -2], atol=1e-6)
