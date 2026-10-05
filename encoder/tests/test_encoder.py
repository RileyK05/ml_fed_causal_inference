"""Unit tests for the patch-transformer encoder and its pre-training (brief M1)."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fedcore.q3 import ResponseModel
from fedcore.q3.contracts import L
from fedcore.q3.synthetic import make_pretrain_windows, make_synthetic_panel

from fedenc.encoder_transformer import PatchTransformerEncoder
from fedenc.pretrain import _choose_masks, _masked_loss, pretrain, window_end_order
from fedenc.arms import TxArm, get_pretrained


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


def test_pretrain_choice_only_valid_patches():
    enc = PatchTransformerEncoder()
    windows, mask = make_pretrain_windows(60, seed=1)
    rng = np.random.default_rng(0)
    choice = _choose_masks(enc, mask, np.arange(60), 0.3, rng)
    x = torch.from_numpy(np.nan_to_num(windows))
    m = torch.from_numpy(mask)
    _, valid = enc.patchify(x, m)
    valid_np = valid.numpy()
    assert not (choice & ~valid_np).any()
    counts = valid_np.sum(axis=1)
    for i in range(60):
        if counts[i] > 0:
            assert 1 <= choice[i].sum() <= counts[i]
        else:
            assert choice[i].sum() == 0


def test_loss_ignores_unobserved_entries():
    enc = PatchTransformerEncoder()
    decoder = torch.nn.Linear(enc.d_model, enc.patch_len * enc.n_channels)
    windows, mask = make_pretrain_windows(24, seed=2)
    scaled = np.nan_to_num(windows).astype(np.float32)
    rows = np.arange(24)
    rng = np.random.default_rng(3)
    choice = _choose_masks(enc, mask, rows, 0.3, rng)
    batch = {
        "history": torch.from_numpy(np.ascontiguousarray(scaled)),
        "mask": torch.from_numpy(np.ascontiguousarray(mask)),
    }
    enc.eval()
    with torch.no_grad():
        loss, count = _masked_loss(enc, decoder, batch, choice)
        polluted = scaled.copy()
        polluted[~np.asarray(mask)] = 12345.0
        batch2 = {
            "history": torch.from_numpy(np.ascontiguousarray(polluted)),
            "mask": batch["mask"],
        }
        loss2, count2 = _masked_loss(enc, decoder, batch2, choice)
    assert count == count2 > 0
    assert torch.allclose(loss, loss2, atol=1e-5)
    _, pvalid = enc.patchify(batch["history"], batch["mask"])
    padded = np.pad(np.asarray(mask), ((0, 0), (enc.total_days - L, 0)))
    bits = padded.reshape(24, -1, enc.patch_len)
    sel = choice & pvalid.numpy()
    expected = int(bits[sel].sum())
    assert count == expected


def test_window_end_order_recovers_true_order():
    rng = np.random.default_rng(0)
    n_firms, n_days, t = 3, 400, 120
    paths = rng.normal(0, 1, size=(n_firms, n_days, 5)).astype(np.float32).cumsum(axis=1)
    ends = np.arange(n_days - t - 20, n_days - 20, 20)
    windows, masks, truth = [], [], []
    for firm in range(n_firms):
        for end in ends:
            windows.append(paths[firm, end - t : end])
            masks.append(np.ones(t, dtype=bool))
            truth.append(int(end))
    windows = np.stack(windows).astype(np.float32)
    masks = np.stack(masks)
    truth = np.asarray(truth)
    perm = rng.permutation(truth.size)
    order = window_end_order(windows[perm], masks[perm])
    assert order.method == "overlap_chain"
    got = np.argsort(order.key, kind="stable")
    assert np.array_equal(truth[perm][got], np.sort(truth[perm]))


def test_window_end_order_falls_back_on_tiny_corpus():
    windows = np.zeros((4, L, 5), dtype=np.float32)
    mask = np.ones((4, L), dtype=bool)
    order = window_end_order(windows, mask)
    assert order.method in ("draw_order", "overlap_chain")


def test_pretrain_smoke():
    enc = PatchTransformerEncoder()
    windows, mask = make_pretrain_windows(80, seed=5)
    result = pretrain(enc, windows, mask, epochs=2, seed=0, batch_size=32, patience=2)
    assert len(result.train_loss) <= 2
    assert len(result.val_loss) == len(result.train_loss)
    assert np.isfinite(result.val_loss).all()
    assert result.best_epoch >= 1
    assert result.state_dict.keys() == enc.state_dict().keys()
    assert set(result.curves.columns) == {"epoch", "train_loss", "val_loss"}


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


def test_txarm_fit_predict_smoke():
    panel = make_synthetic_panel(n_firms=8, n_meetings=12, noise="easy", seed=0)
    arm = TxArm(seed=0, max_epochs=2, patience=2)
    arm.fit(panel, seed=0)
    mu, a, b = arm.predict(panel)
    assert mu.shape == panel.target.shape
    assert np.isfinite(mu).all() and np.isfinite(a).all() and np.isfinite(b).all()
    assert arm.fit_result.best_epoch >= 1


def test_pretrained_top_freezes_the_lower_stack():
    result = get_pretrained(n_windows=60, seed=1, epochs=1)
    panel = make_synthetic_panel(n_firms=8, n_meetings=12, noise="easy", seed=0)
    arm = TxArm(
        seed=0,
        init="pretrained",
        adapt="top",
        pretrain_kwargs={"n_windows": 60, "seed": 1, "epochs": 1},
        max_epochs=2,
        patience=2,
    )
    arm.fit(panel, seed=0)
    enc = arm.model.encoder
    assert torch.equal(enc.proj.weight, result.state_dict["proj.weight"])
    assert torch.equal(enc.pos, result.state_dict["pos"])
    assert torch.equal(enc.blocks.layers[0].linear1.weight, result.state_dict["blocks.layers.0.linear1.weight"])
    assert not enc.proj.weight.requires_grad
    assert not enc.blocks.layers[0].linear1.weight.requires_grad
    assert enc.blocks.layers[-1].linear1.weight.requires_grad
    assert not enc.blocks.layers[0].linear1.weight.grad


def test_pretrained_arm_refuses_real_panels():
    """Only a synthetic checkpoint exists: a pretrained arm on real data (no b_true) must not run."""
    import dataclasses

    from fedcore.q3 import make_synthetic_panel

    panel = dataclasses.replace(make_synthetic_panel(n_firms=6, n_meetings=6, seed=0), a_true=None, b_true=None)
    with pytest.raises(NotImplementedError, match="pre-cutoff"):
        TxArm(init="pretrained", adapt="frozen", max_epochs=1).fit(panel)
