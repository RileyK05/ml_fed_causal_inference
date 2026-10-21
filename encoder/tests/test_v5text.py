"""v5 text arm on a synthetic panel and a tiny randomly initialised EmbeddingGemma 2 (no download)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers.models.embedding_gemma2.configuration_embedding_gemma2 import EmbeddingGemma2TextConfig  # noqa: E402
from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import EmbeddingGemma2TextModel  # noqa: E402

from fedcore.q3.evaluate import run_walk_forward  # noqa: E402
from fedcore.q3.synthetic import make_synthetic_panel  # noqa: E402
from fedenc import v3, v5text  # noqa: E402


@pytest.fixture(scope="module")
def backbone():
    torch.manual_seed(0)
    cfg = EmbeddingGemma2TextConfig(vocab_size=50, hidden_size=16, intermediate_size=32, num_hidden_layers=6,
                                    num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                                    hidden_size_per_layer_input=8, embedding_dim=16, sliding_window=8,
                                    max_position_embeddings=64)
    return EmbeddingGemma2TextModel(cfg).eval()


@pytest.fixture(scope="module")
def panel():
    return make_synthetic_panel(n_firms=10, n_meetings=10, seed=0)


def store_for(panel, with_text=lambda firm: True, seed=0):
    rng = np.random.default_rng(seed)
    firms = np.unique(panel.firm_id)
    ids = [rng.integers(3, 50, size=rng.integers(5, 12)) for _ in firms]
    keys = [(m, int(f)) for m, f in zip(pd.DatetimeIndex(panel.meeting), panel.firm_id) if with_text(int(f))]
    row_doc = pd.Series([int(np.searchsorted(firms, f)) for _, f in keys],
                        index=pd.MultiIndex.from_tuples(keys), dtype="int64")
    return v5text.TextStore(ids, row_doc)


def test_store_round_trip_and_lookup(panel, tmp_path):
    store = store_for(panel, with_text=lambda f: f % 2 == 0)
    store.save(tmp_path)
    back = v5text.TextStore.load(tmp_path)
    docs = back.docs_for(panel)
    assert (docs[np.asarray(panel.firm_id) % 2 == 1] == -1).all() and (docs[np.asarray(panel.firm_id) % 2 == 0] >= 0).all()
    ids, mask = back.batch(np.array([0, 1]), "cpu", max_tokens=6)
    assert ids.shape[0] == 2 and ids.shape[1] <= 6 and mask.sum() == min(len(back.ids[0]), 6) + min(len(back.ids[1]), 6)


def test_subset_basis_spans_the_draws_betas():
    rng = np.random.default_rng(0)
    span = torch.as_tensor(np.column_stack((np.ones(9), rng.normal(1, .3, 9), rng.normal(1, .3, 9))))
    inverse = torch.as_tensor([0] * 5 + [1] * 4)
    q = v5text.subset_basis(span, inverse, 2)
    v = torch.as_tensor(rng.normal(size=9))
    pv = v3.project(v, q, inverse, 2)
    for m in (0, 1):
        r = (inverse == m).numpy()
        for col in range(3):
            assert abs(float(pv[r] @ span[r, col])) < 1e-8


@pytest.mark.parametrize("arm", ["T0", "T1", "T1f"])
def test_arms_fit_and_predict_projected(panel, backbone, arm):
    store = store_for(panel, with_text=lambda f: f != int(np.unique(panel.firm_id)[0]))  # one firm has no filing
    make = v5text.makers(store, backbone, device="cpu")[arm]

    def small(seed):
        a = make(seed)
        a.max_epochs, a.patience, a.firms_per_meeting, a.doc_batch = 2, 1, 6, 4
        a.lr = 1e-2
        return a

    result = run_walk_forward(panel, small, min_train=7, test_size=3, seeds=(0,))
    p = result.predictions
    assert np.isfinite(p["mu"]).all() and len(p) == (pd.DatetimeIndex(panel.meeting) > panel.meetings()[6]).sum()
    test = panel.subset(np.flatnonzero(pd.DatetimeIndex(panel.meeting) > panel.meetings()[6]))
    arm_ = small(0)
    arm_.fit(panel.subset(np.flatnonzero(pd.DatetimeIndex(panel.meeting) <= panel.meetings()[6])))
    db = arm_.prediction_extras(test)["delta_b"]
    b63, bw = v3.base_betas(test)
    for m in test.meetings():
        r = pd.DatetimeIndex(test.meeting) == m
        for basis in (np.ones(r.sum()), b63[r], bw[r]):
            assert abs(db[r] @ basis) < 1e-6
    if arm == "T0":
        assert arm_.nets[0].tower is None
    if arm == "T1f":
        assert not any(p.requires_grad for p in arm_.nets[0].tower.backbone.parameters())


def test_new_model_is_in_training_mode_even_from_an_eval_backbone(panel, backbone):
    backbone.eval()  # as from_pretrained leaves it
    arm = v5text.V5TextArm(0, store=store_for(panel), backbone=backbone, device="cpu", train_top=2)
    model = arm._new_model(panel, "cpu")
    assert model.training and model.tower.backbone.layers[-1].training
    assert model.tower.backbone.layers[-1].gradient_checkpointing
