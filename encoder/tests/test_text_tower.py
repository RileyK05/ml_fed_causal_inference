"""Fundamentals tower on a tiny randomly initialised EmbeddingGemma 2 (no download)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers.models.embedding_gemma2.configuration_embedding_gemma2 import EmbeddingGemma2TextConfig  # noqa: E402
from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import EmbeddingGemma2TextModel  # noqa: E402

from fedenc import text_tower as tt  # noqa: E402


@pytest.fixture(scope="module")
def backbone():
    torch.manual_seed(0)
    cfg = EmbeddingGemma2TextConfig(vocab_size=300, hidden_size=32, intermediate_size=64, num_hidden_layers=6,
                                    num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                                    hidden_size_per_layer_input=16, embedding_dim=24, sliding_window=8,
                                    max_position_embeddings=256)
    return EmbeddingGemma2TextModel(cfg).eval()


def batch(n=3, length=20, seed=1):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(3, 300, (n, length), generator=g)
    mask = torch.ones(n, length, dtype=torch.long)
    mask[1, 12:] = 0
    ids[1, 12:] = 0
    return ids, mask


def test_lower_plus_upper_matches_the_backbone(backbone):
    tower = tt.FilingTower(backbone, train_top=2).eval()
    ids, mask = batch()
    full = backbone(input_ids=ids, attention_mask=mask).last_hidden_state
    split = tower.upper(tower.lower(ids, mask), ids, mask)
    torch.testing.assert_close(split, full, atol=1e-5, rtol=1e-5)


def test_only_top_layers_pooling_and_head_train(backbone):
    tower = tt.FilingTower(backbone, train_top=2)
    trainable = {n for n, p in tower.named_parameters() if p.requires_grad}
    assert any(n.startswith("backbone.layers.5.") for n in trainable)
    assert not any(n.startswith("backbone.layers.3.") for n in trainable)
    assert not any("embed_tokens" in n for n in trainable)
    assert {"score.weight", "out.2.weight"} <= trainable
    frozen = tt.FilingTower(backbone, train_top=0, pooling="mean")
    assert {n for n, p in frozen.named_parameters() if p.requires_grad and n.startswith("backbone")} == set()


def test_output_shape_padding_and_gradients(backbone):
    tower = tt.FilingTower(backbone, train_top=1)
    ids, mask = batch()
    out = tower(ids, mask)
    assert out.shape == (3, tt.OUT_DIM) and torch.isfinite(out).all()
    out.sum().backward()
    assert backbone.layers[5].mlp.down_proj.weight.grad is not None
    assert backbone.layers[0].mlp.down_proj.weight.grad is None
    # padding never gets pooling weight
    w = tower.token_weights(tower.upper(tower.lower(ids, mask), ids, mask), mask)
    assert float(w[1, 12:].detach().abs().sum()) == 0.0
    torch.testing.assert_close(w.sum(1), torch.ones(3))
    for p in backbone.parameters():
        p.grad = None


def test_float16_is_refused():
    with pytest.raises(ValueError):
        tt.load_backbone(dtype=torch.float16)


def test_select_text_keeps_rate_sentences_in_order():
    section = "Our stores did well. Rates on our variable rate debt float with SOFR. We sell shoes. A 100 basis point rise costs $5 million."
    count = lambda s: len(s.split())
    assert tt.select_text(section, count, max_tokens=100) == section
    short = tt.select_text(section, count, max_tokens=18)
    assert "variable rate debt" in short and "basis point" in short and "shoes" not in short
    assert short.index("variable") < short.index("basis point")


def test_filing_for_rows_uses_only_filings_before_the_meeting_day():
    rows = pd.DataFrame({"meeting": pd.to_datetime(["2005-03-22", "2005-03-22", "2007-01-31"]),
                         "gvkey": ["000001", "000002", "000001"]})
    link = pd.DataFrame({"gvkey": ["000001", "000002"], "cik": ["0000000001", "0000000002"]})
    sections = pd.DataFrame({
        "accession": ["a-old", "a-sameday", "b-1", "a-none"],
        "cik": ["0000000001", "0000000001", "0000000002", "0000000001"],
        "acceptance": pd.to_datetime(["2005-02-10 16:00", "2005-03-22 08:00", "2005-03-21 17:30", "2006-12-01 09:00"]),
        "method": ["item", "item", "item", "none"],
        "section": ["x", "y", "z", ""],
    })
    got = tt.filing_for_rows(rows, link, sections)
    # a later "no material change" 10-Q is skipped in favour of the substantive filing
    later = pd.DataFrame({"accession": ["a-nochange"], "cik": ["0000000001"],
                          "acceptance": pd.to_datetime(["2005-03-01 16:00"]), "method": ["item"], "section": ["none"],
                          "no_change": [True]})
    with_nc = pd.concat([sections.assign(no_change=False), later], ignore_index=True)
    assert tt.filing_for_rows(rows, link, with_nc).iloc[0] == "a-old"
    assert got.iloc[0] == "a-old"            # the same-day filing is skipped
    assert got.iloc[1] == "b-1"              # accepted the evening before: usable
    assert pd.isna(got.iloc[2])              # only an empty section within 400 days


def test_two_tower_starts_at_zero_and_uses_text(backbone):
    from fedenc.encoder_transformer import PatchTransformerEncoder
    from fedcore.q3.contracts import CHANNELS, L

    torch.manual_seed(0)
    tower = tt.FilingTower(backbone, train_top=1)
    model = tt.TwoTowerModel(PatchTransformerEncoder(d_model=64, n_layers=1), b_only=True)
    ids, mask = batch(n=2)
    row_doc = np.array([0, 1, 1, -1])
    text, has = tt.gather_filings(tower, ids, mask, row_doc)
    assert text.shape == (4, tt.OUT_DIM) and has.tolist() == [True, True, True, False]
    torch.testing.assert_close(text[1], text[2])
    hist = torch.randn(4, L, len(CHANNELS))
    hmask = torch.ones(4, L, dtype=torch.bool)
    a, b = model(hist, hmask, text, has)
    assert torch.all(a == 0) and torch.all(b == 0)  # zero-init heads: predicts the base exactly
    b.sum().backward()
    assert model.head_b.weight.grad is not None and model.head_b.weight.grad.abs().sum() > 0
