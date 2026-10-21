"""Unit tests for fedenc.v6 (shock-day pretraining, filing gate, big-surprise selection) and
fedenc.filing_vectors. Tiny synthetic fits only."""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from fedcore.q3.contracts import BETA63
from fedcore.q3.real import FUNDAMENTAL_NAMES
from fedcore.q3.synthetic import make_synthetic_panel
from fedenc import v3, v6
from fedenc.encoder_transformer import PatchTransformerEncoder
from fedenc.filing_vectors import FilingVectors, prior_year


def tiny(n_channels=5):
    return PatchTransformerEncoder(n_channels=n_channels, d_model=8, n_layers=1, n_heads=2, dim_feedforward=16,
                                   dropout=0.0)


def _realistic(p, seed):
    rng = np.random.default_rng(seed)
    hist = p.history.copy()
    hist[:, :, BETA63] = rng.uniform(0.5, 1.5, (len(p.target), 1)) + rng.normal(0, 0.2, hist.shape[:2])
    k = len(FUNDAMENTAL_NAMES) - p.fundamentals.shape[1]
    fund = np.column_stack((p.fundamentals, rng.normal(size=(len(p.target), k)))).astype(np.float32)
    mask = np.column_stack((p.fundamentals_mask, np.ones((len(p.target), k), dtype=bool)))
    return dataclasses.replace(p, history=hist, fundamentals=fund, fundamentals_mask=mask)


@pytest.fixture
def panel():
    return _realistic(make_synthetic_panel(n_firms=14, n_meetings=12, seed=0), 0)


@pytest.fixture
def shock(panel):
    """Shock days interleaved with the Fed meetings, some after the last one."""
    p = _realistic(make_synthetic_panel(n_firms=14, n_meetings=16, seed=3), 3)
    fed = panel.meetings()
    days = pd.DatetimeIndex(np.concatenate([fed + pd.Timedelta(days=10), fed[-4:] + pd.Timedelta(days=20)]))
    remap = dict(zip(p.meetings(), days.sort_values()))
    return dataclasses.replace(p, meeting=pd.DatetimeIndex(p.meeting).map(remap).to_numpy(dtype="datetime64[ns]"))


@pytest.fixture
def vectors(panel):
    rng = np.random.default_rng(5)
    idx = pd.MultiIndex.from_arrays([pd.DatetimeIndex(panel.meeting), panel.firm_id.astype(np.int64)],
                                    names=["date", "permno"])
    keep = rng.random(len(idx)) < 0.8
    accs = [f"a{k}" for k in range(int(keep.sum()))]
    row_acc = pd.Series(accs, index=idx[keep])
    prior = pd.Series(accs[:-5], index=accs[5:])
    return FilingVectors(rng.normal(size=(len(accs), 6)).astype(np.float16), pd.Series(accs), prior, row_acc)


def _check_projected(panel, v):
    b63, bw = v3.base_betas(panel)
    for m in panel.meetings():
        r = pd.DatetimeIndex(panel.meeting) == m
        for basis in (np.ones(r.sum()), b63[r], bw[r]):
            assert abs(v[r] @ basis) < 1e-8


@pytest.mark.parametrize("kind", ["plain", "pretrained", "gated", "change"])
def test_arms_fit_and_predict_projected(panel, shock, vectors, kind):
    kw = dict(members=2, max_epochs=2, patience=1, device="cpu", encoder_fn=tiny, pretrain_epochs=2, pretrain_patience=1)
    if kind != "plain":
        kw["pretrain"] = shock
    if kind in ("gated", "change"):
        kw |= dict(vectors=vectors, change=kind == "change")
    arm = v6.V6Arm(0, **kw)
    arm.fit(panel)
    mu, a, b = arm.predict(panel)
    ex = arm.prediction_extras(panel)
    s = np.asarray(panel.shock, dtype=np.float64)
    np.testing.assert_allclose(mu, ex["mu_base"] + ex["delta_b"] * s, atol=1e-9)
    assert np.all(ex["delta_a"] == 0.0)
    _check_projected(panel, ex["delta_b"])
    rec = arm.fit_result.member_selection
    assert len(rec["members"]) == 2 and "selected_on_big" in rec
    if kind == "plain":
        assert rec["pretrain"] is None
    else:
        last = panel.meetings()[-1]
        assert rec["pretrain"]["events"] == int((shock.meetings() < last).sum())


def test_pretraining_never_sees_shock_days_after_the_window(panel, shock):
    last = panel.meetings()[-1]
    late = pd.DatetimeIndex(shock.meeting) >= last
    y = np.asarray(shock.target, dtype=np.float32).copy()
    y[late] += 100.0
    states = []
    for target in (shock.target, y):
        arm = v6.V6Arm(0, members=1, max_epochs=0, device="cpu", encoder_fn=tiny, pretrain_epochs=2,
                       pretrain=dataclasses.replace(shock, target=target))
        states.append(arm._pretrain(panel, "cpu", 7)[0])
    for k in states[0]:
        torch.testing.assert_close(states[0][k], states[1][k])


def test_epoch_zero_keeps_the_pretrained_correction(panel, shock):
    plain = v6.V6Arm(0, members=1, max_epochs=0, device="cpu", encoder_fn=tiny)
    plain.fit(panel)
    assert np.all(plain.prediction_extras(panel)["delta_b"] == 0.0)  # no pretraining: the base exactly
    pre = v6.V6Arm(0, members=1, max_epochs=0, device="cpu", encoder_fn=tiny, pretrain=shock)
    model = pre._new_model(panel, "cpu")
    torch.nn.init.normal_(model.head_b.weight)                        # stands in for a pretrained model
    scaler = v3._Scaler(panel)
    pre._pretrain = lambda *_: (model.state_dict(), scaler, dict(events=0, epoch=1, val=0.0, curve=[]))
    pre.fit(panel)
    got = pre.prediction_extras(panel)["delta_b"]
    model.eval()
    d = pre._data(panel, pre.base, scaler, "cpu")
    with torch.no_grad():
        idx = torch.arange(len(panel.target))
        _, want, _, _ = v3._correction(model, d["x"], idx, d["q"], d["codes"], d["s"])
    assert np.abs(got).max() > 0
    np.testing.assert_allclose(got, want.numpy(), atol=1e-6)        # the pretrained model, as is


def test_selection_uses_big_meetings_or_falls_back(panel):
    arm = v6.V6Arm(0, device="cpu", encoder_fn=tiny)
    groups = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]
    p = dataclasses.replace(panel, shock=np.r_[0.3, 0.3, 0.01, 0.01, -0.5, -0.5, np.zeros(len(panel.shock) - 6)]
                            .astype(np.float32))
    got, big = arm._selection_groups(p, groups)
    assert big and [g[0] for g in got] == [0, 4]
    small = dataclasses.replace(panel, shock=np.full(len(panel.shock), 0.01, dtype=np.float32))
    got, big = arm._selection_groups(small, groups)
    assert not big and len(got) == 3


def test_gate_is_identity_at_start_and_for_missing_filings(panel):
    arm = v6.V6Arm(0, device="cpu", encoder_fn=tiny)
    base = arm._new_model(panel, "cpu")
    nn = torch.nn
    nn.init.normal_(base.head_b.weight)
    gated = v6.GatedCorrection(base, width=6)
    x = v3._Scaler(panel).tensors(panel, "cpu")
    args = [x[k] for k in v3.INPUT_KEYS]
    filing = torch.randn(len(panel.target), 6)
    gated.eval(), base.eval()
    _, b0 = base(*args)
    _, b1 = gated(*args, filing, torch.ones(len(panel.target), dtype=torch.bool))
    torch.testing.assert_close(b0, b1)                                  # zero-init gate
    nn.init.normal_(gated.gate[-1].weight)
    _, b2 = gated(*args, filing, torch.zeros(len(panel.target), dtype=torch.bool))
    torch.testing.assert_close(b0, b2)                                  # no filing: gate off
    _, b3 = gated(*args, filing, torch.ones(len(panel.target), dtype=torch.bool))
    ratio = (b3 / b0)[b0.abs() > 1e-6]
    assert (ratio >= 1 - v6.GATE_LAMBDA - 1e-6).all() and (ratio <= 1 + v6.GATE_LAMBDA + 1e-6).all()


def test_filing_vectors_level_change_and_missing(panel, vectors):
    v, ok = vectors.for_panel(panel)
    assert v.shape == (len(panel.target), 6) and 0 < ok.mean() < 1
    assert np.all(v[~ok] == 0)
    c, ok_c = vectors.for_panel(panel, change=True)
    assert ok_c.sum() < ok.sum() and np.all(c[~ok_c] == 0)
    k = int(np.flatnonzero(ok_c)[0])
    idx = (pd.Timestamp(panel.meeting[k]), int(panel.firm_id[k]))
    acc = vectors.row_acc[idx]
    expect = vectors.vectors[vectors.position[acc]].astype(np.float32) - \
        vectors.vectors[vectors.position[vectors.prior[acc]]].astype(np.float32)
    np.testing.assert_allclose(c[k], expect)


def test_filing_vectors_round_trip(vectors, panel, tmp_path):
    vectors.save(tmp_path)
    back = FilingVectors.load(tmp_path)
    np.testing.assert_array_equal(back.for_panel(panel)[0], vectors.for_panel(panel)[0])
    np.testing.assert_array_equal(back.for_panel(panel, change=True)[0], vectors.for_panel(panel, change=True)[0])


def test_prior_year_picks_same_form_nearest_a_year():
    s = pd.DataFrame({"accession": ["k20", "k21", "q21", "k22", "k22b"], "cik": ["1"] * 5,
                      "form": ["10-K", "10-K", "10-Q", "10-K", "10-K"],
                      "acceptance": pd.to_datetime(["2020-03-01", "2021-03-03", "2021-05-01", "2022-02-20", "2022-09-01"])})
    p = prior_year(s)
    assert p["k21"] == "k20" and p["k22"] == "k21"
    assert "q21" not in p and "k20" not in p           # no same-form filing a year earlier
    assert p.get("k22b") is None                       # 547 days after k21: outside the window


def test_embed_is_the_masked_mean_of_the_last_hidden_state():
    pytest.importorskip("transformers")
    from transformers.models.embedding_gemma2.configuration_embedding_gemma2 import EmbeddingGemma2TextConfig
    from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import EmbeddingGemma2TextModel
    from fedenc.filing_vectors import embed

    torch.manual_seed(0)
    cfg = EmbeddingGemma2TextConfig(vocab_size=300, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                    num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                                    hidden_size_per_layer_input=16, embedding_dim=24, sliding_window=8,
                                    max_position_embeddings=256)
    bb = EmbeddingGemma2TextModel(cfg).eval()

    class Tok:
        pad_token_id = 0

        def __call__(self, text, truncation=True, max_length=None):
            ids = [3 + (ord(c) % 290) for c in text][:max_length]
            return {"input_ids": ids}

    texts = ["short text", "a somewhat longer piece of filing text", "mid length one"]
    got = embed(texts, Tok(), bb, "cpu", batch=2)
    for k, t in enumerate(texts):
        ids = torch.as_tensor([Tok()(t)["input_ids"]])
        want = bb(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state.mean(1)[0]
        np.testing.assert_allclose(got[k].astype(np.float32), want.detach().numpy(), atol=2e-3, rtol=1e-2)
