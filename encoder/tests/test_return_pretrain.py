"""Timing, weighting, checkpoint transfer and a tiny synthetic-only smoke fit."""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from fedcore.q3.real import PretrainCorpus
from fedcore.q3.synthetic import make_synthetic_panel
from fedcore.results import ResultStore
from fedenc.arms import TxArm, make_arms
from fedenc.checkpoint import load_checkpoint, save_checkpoint
from fedenc.encoder_transformer import PatchTransformerEncoder
from fedenc.pretrain_return import (
    OBJECTIVES, GaussianHead, ReturnPretrainResult, date_groups,
    evaluate, fit_scaler, gaussian_loss, pretrain_return,
)
from fedenc.return_corpus import from_history_corpus


def tiny_corpus():
    rng = np.random.default_rng(3)
    calendar = pd.bdate_range("1990-01-01", periods=280)
    channels = rng.normal(size=(2, len(calendar), 5)).astype(np.float32)
    observed = np.ones(channels.shape[:2], bool)
    firm = np.repeat(np.arange(2), 16)
    end = np.tile(np.arange(251, 267), 2)
    source = PretrainCorpus(channels, observed, calendar, np.array([10, 20]),
                           firm, end, calendar[270])
    return from_history_corpus(source, calendar[[255, 259]])


def fake_result():
    torch.manual_seed(0)
    encoder = PatchTransformerEncoder(d_model=8, n_layers=1, n_heads=2, dim_feedforward=16, dropout=0)
    params = dict(objective=OBJECTIVES[0], return_units="percent", cutoff="1991-01-01",
                  available_through="1990-12-31", train_target_end="1990-11-30",
                  val_target_end="1990-12-31")
    return ReturnPretrainResult(
        encoder, {k: v.clone() for k, v in encoder.state_dict().items()},
        {"hist_median": np.array([0.01, -1.23456789012345, 2, 3, 4]),
         "hist_iqr": np.array([2, 3, 4, 5, 6], dtype=float)},
        pd.DataFrame({"epoch": [1], "train_loss": [1.0], "val_loss": [2.0]}), 1, 2.0, params,
    )


@pytest.fixture
def checkpoint(tmp_path):
    # Serialize fabricated weights only. No trained run is written in this test.
    return load_checkpoint(save_checkpoint(fake_result(), ResultStore(tmp_path)).path)


def test_next_day_target_timing_exclusions_and_missingness():
    corpus = tiny_corpus()
    source = corpus.source
    history, mask, target = corpus.batch(np.arange(len(corpus)))
    assert (corpus.history_end_dates < corpus.target_dates).all()
    assert (corpus.target_dates < source.cutoff).all()
    assert not corpus.target_dates.isin(corpus.excluded_dates).any()
    np.testing.assert_equal(history[:, -1, 0], source.channels[source.firm[corpus.rows], source.end[corpus.rows], 0])
    np.testing.assert_equal(target, source.channels[source.firm[corpus.rows], source.end[corpus.rows] + 1, 0])
    assert mask.all()
    source.observed[0, 253] = False
    source.channels[0, 253] = np.nan
    filtered = from_history_corpus(source, corpus.excluded_dates)
    assert not ((source.firm[filtered.rows] == 0) & (source.end[filtered.rows] == 252)).any()
    # A history gap remains missing, while later valid targets can still be used.
    h, m, _ = filtered.batch(np.arange(len(filtered)))
    assert (~m).any()
    assert np.isnan(h[~m]).all()


def test_target_cutoff_checks_label_not_just_history():
    corpus = tiny_corpus()
    cutoff = corpus.source.calendar[261]
    source = dataclasses.replace(corpus.source, cutoff=cutoff)
    filtered = from_history_corpus(source, [])
    assert filtered.target_dates.max() < cutoff
    assert not (source.end[filtered.rows] == 260).any()


def test_calendar_split_and_train_only_scaler():
    corpus = tiny_corpus()
    train, val = corpus.split(0.25)
    assert corpus.target_dates[train].max() < corpus.target_dates[val].min()
    before = fit_scaler(corpus, train, seed=0)
    source = corpus.source
    boundary = source.calendar.get_loc(corpus.target_dates[val].min())
    source.channels[:, boundary:] = 100000
    after = fit_scaler(corpus, train, seed=0)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])


def test_gaussian_score_targets_mean_and_variance():
    mean = torch.zeros(2, requires_grad=True)
    var = torch.ones(2, requires_grad=True)
    returns = torch.tensor([2.0, -2.0])
    losses = gaussian_loss(mean, var, returns)
    assert torch.equal(losses[0], losses[1])
    losses.sum().backward()
    assert (var.grad < 0).all()  # optimizer increases the variance toward (r - m)^2 = 4
    assert mean.grad[0] < 0 < mean.grad[1]  # and moves each mean toward its return
    # At the sample mean and variance both gradients vanish.
    sample = torch.tensor([1.0, 3.0])
    m = torch.tensor([2.0, 2.0], requires_grad=True)
    v = torch.tensor([1.0, 1.0], requires_grad=True)
    gaussian_loss(m, v, sample).sum().backward()
    assert m.grad.sum().item() == 0 and v.grad.sum().item() == 0
    head_mean, head_var = GaussianHead(8)(torch.zeros(2, 8))
    assert head_mean.shape == head_var.shape == (2,)
    assert (head_var > 0).all()


def test_evaluation_weights_dates_equally_across_fragmented_batches():
    corpus = tiny_corpus()
    # Different firm counts on the two target dates.
    rows = np.array([0, 14, 1])
    groups = date_groups(corpus, rows)
    assert sorted(map(len, groups)) == [1, 2]

    class Encoder(torch.nn.Module):
        def forward(self, x, mask):
            return torch.ones(len(x), 1)

    class Head(torch.nn.Module):
        def forward(self, h):
            return torch.zeros(len(h)), torch.ones(len(h))

    expected = np.mean([np.mean(corpus.batch(group)[2] ** 2) for group in groups])
    for batch_size in (1, 8):
        got = evaluate(Encoder(), Head(), corpus, groups,
                       {"hist_median": np.zeros(5), "hist_iqr": np.ones(5)},
                       objective=OBJECTIVES[0], mask_frac=0.3, batch_size=batch_size,
                       seed=0, device="cpu")
        assert got == pytest.approx(expected)


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_streaming_pretraining_smoke(objective):
    torch.manual_seed(0)
    encoder = PatchTransformerEncoder(d_model=8, n_layers=1, n_heads=2, dim_feedforward=16, dropout=0)
    corpus = tiny_corpus()
    result = pretrain_return(encoder, corpus, objective=objective, epochs=1, batch_size=8,
                             dates_per_step=4, scaler_windows=16, purge_history=False, device="cpu")
    assert result.best_epoch == 1
    assert np.isfinite(result.curves[["train_loss", "val_loss"]]).all().all()
    assert result.params["available_through"] == str(corpus.target_dates.max().date())
    assert result.params["train_target_end"] < result.params["val_target_start"]
    assert result.state_dict.keys() == encoder.state_dict().keys()
    # No checkpoint/run is saved by a smoke fit.


def test_purge_prevents_shared_reconstruction_targets():
    rng = np.random.default_rng(1)
    calendar = pd.bdate_range("1990-01-01", periods=800)
    source = PretrainCorpus(rng.normal(size=(1, 800, 5)).astype(np.float32),
                           np.ones((1, 800), bool), calendar, np.array([1]),
                           np.zeros(500, dtype=int), np.arange(251, 751), calendar[-1])
    corpus = from_history_corpus(source, [])
    train, val = corpus.split(0.1, purge_history=True)
    first_val_context = source.calendar[source.end[corpus.rows[val]].min() - 251]
    assert corpus.target_dates[train].max() < first_val_context
    assert corpus.target_dates[val].min() > corpus.target_dates[train].max()
    with pytest.raises(ValueError, match="purging"):
        tiny_corpus().split(purge_history=True)


def test_checkpoint_roundtrip_retains_weights_scaler_config_and_no_head(checkpoint):
    expected = fake_result()
    assert checkpoint.config == expected.encoder.get_config()
    for key, tensor in expected.state_dict.items():
        assert torch.equal(checkpoint.state_dict[key], tensor)
    for key, value in expected.scaler.items():
        np.testing.assert_array_equal(checkpoint.scaler[key], value)
    assert not any("head" in key for key in checkpoint.state_dict)
    checkpoint.require_before("1991-01-01")
    with pytest.raises(ValueError, match="must precede"):
        checkpoint.require_before("1990-12-31")


def test_checkpoint_rejects_changed_scaler(checkpoint):
    path = checkpoint.path / "tables" / "history-scaler.csv"
    table = pd.read_csv(path)
    table.loc[0, "median"] += 1
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="digest"):
        load_checkpoint(checkpoint.path)


def test_pretrained_fit_preserves_history_scaling_and_trains_response(checkpoint):
    panel = make_synthetic_panel(n_firms=4, n_meetings=6, seed=0)
    arm = TxArm(init="pretrained", checkpoint=checkpoint, max_epochs=1, patience=1,
                device="cpu", adapt="full")
    arm.fit(panel)
    np.testing.assert_equal(arm.scaler.hist_median, checkpoint.scaler["hist_median"])
    np.testing.assert_equal(arm.scaler.hist_iqr, checkpoint.scaler["hist_iqr"])
    mu, a, b = arm.predict(panel)
    assert np.isfinite(mu).all()
    np.testing.assert_allclose(mu, a + b * panel.shock, atol=1e-5)
    assert arm.model.head.a.weight.requires_grad and arm.model.head.b.weight.requires_grad


def test_checkpoint_after_inner_validation_is_rejected_before_fit(checkpoint):
    panel = make_synthetic_panel(n_firms=4, n_meetings=6, seed=0)
    late = dataclasses.replace(checkpoint, params={**checkpoint.params, "available_through": "2099-01-01"})
    arm = TxArm(init="pretrained", checkpoint=late, max_epochs=1)
    with pytest.raises(ValueError, match="must precede"):
        arm.fit(panel)
    assert arm.model is None


def test_missing_checkpoint_never_implicitly_trains():
    with pytest.raises(ValueError, match="explicit pre-cutoff"):
        TxArm(init="pretrained")._encoder()


def test_scratch_comparator_shares_normalization_but_uses_random_weights(checkpoint):
    torch.manual_seed(123)
    arm = make_arms(checkpoint)["tx_scratch"](123)
    encoder = arm._encoder()
    assert arm.loaded_checkpoint is checkpoint
    assert encoder.get_config() == checkpoint.config
    assert not torch.equal(encoder.proj.weight, checkpoint.state_dict["proj.weight"])


def test_plan_loads_no_data_or_fits(monkeypatch, capsys):
    import fedenc.workflow as workflow

    def forbidden(*args, **kwargs):
        raise AssertionError("plan must not load data or train")

    monkeypatch.setattr(workflow, "real_return_corpus", forbidden)
    monkeypatch.setattr(workflow, "pretrain_return", forbidden)
    workflow.main(["plan", "--cutoff", "2007-01-01"])
    output = json.loads(capsys.readouterr().out)
    assert output["fits"] is False and output["loads_data"] is False


def test_evaluation_preflight_rejects_late_checkpoint_without_fitting(checkpoint, monkeypatch):
    import fedcore.q3.real as real
    import fedenc.workflow as workflow

    panel = make_synthetic_panel(n_firms=3, n_meetings=8, seed=0)
    late = dataclasses.replace(checkpoint, params={**checkpoint.params,
                                                 "available_through": "2025-12-31"})
    monkeypatch.setattr(workflow, "load_checkpoint", lambda path: late)
    monkeypatch.setattr(real, "load_real_panel", lambda **_: (panel, None, {}))
    monkeypatch.setattr(workflow, "run_walk_forward", lambda *a, **k: pytest.fail("must not fit"))
    with pytest.raises(ValueError, match="must precede"):
        workflow.evaluate_checkpoint("unused", through="2009-03-18", min_train=4, test_size=2)
    with pytest.raises(ValueError, match="settings"):
        workflow.evaluate_checkpoint("unused", through="2009-03-18", test_size=0)


def test_three_way_runner_saves_traceable_artifacts_without_optimizer_steps(checkpoint, tmp_path, monkeypatch):
    import fedcore.q3.real as real
    import fedenc.workflow as workflow
    from fedcore.q3 import ResponseModel
    from fedcore.q3.evaluate import EvalResult
    from fedcore.q3.train import ChannelScaler

    baseline = fake_result()
    baseline.params["objective"] = OBJECTIVES[1]
    masked_run = save_checkpoint(baseline, ResultStore(tmp_path / "masked"))
    panel = make_synthetic_panel(n_firms=3, n_meetings=8, seed=0)
    monkeypatch.setattr(real, "load_real_panel", lambda **_: (panel, None, {}))

    def fake_walk_forward(panel, factory, **kwargs):
        # Fabricate fitted artifacts without fitting anything or saving trained weights.
        arm = factory(0)
        arm.model = ResponseModel(arm._encoder(), n_fund=3, n_ctx=4)
        arm.scaler = ChannelScaler.fit(panel)
        return EvalResult(pd.DataFrame({"meeting": [panel.meeting[-1]]}),
                          {"mse_per_meeting": 1.0}, pd.DataFrame({"seed": [0]}),
                          pd.DataFrame({"fold": [0], "seed": [0]}),
                          pd.DataFrame({"epoch": [1], "train_loss": [1.0], "val_loss": [1.0]}))

    monkeypatch.setattr(workflow, "run_walk_forward", fake_walk_forward)
    monkeypatch.setattr(workflow, "compare", lambda *args: {"estimate": 0.0, "n_meetings": 1})

    class FakeDataStore(ResultStore):
        def save(self, *args, **kwargs):
            kwargs["datasets"] = []  # fabricated test panel has no registered raw inputs
            return super().save(*args, **kwargs)

    store = FakeDataStore(tmp_path / "eval")
    summary = workflow.evaluate_checkpoint(
        checkpoint.path, masked_checkpoint=masked_run.path, through="2009-03-18",
        min_train=4, test_size=2, seeds=(0,),
        arms=("tx_scratch", "tx_pre_full", "tx_masked_full"), store=store,
    )
    assert summary.arm.tolist() == ["tx_scratch", "tx_pre_full", "tx_masked_full"]
    assert len(store.runs("Q3")) == 4
    runs = {run.manifest["name"]: run for run in store.runs("Q3")}
    assert runs["tx_masked_full"].manifest["params"]["objective"] == OBJECTIVES[1]
    assert runs["tx_pre_full"].manifest["params"]["objective"] == OBJECTIVES[0]
    assert len(runs["tx_pre_full"].table("response-state")) > 0
    assert len(runs["tx_pre_full"].table("response-scalers")) == 6


# ---------------------------------------------------------------- per-fold checkpoints,
# mean/variance split and staged adaptation. Fabricated weights and smoke fits only.

def _checkpoint_at(tmp_path, through, name, objective=OBJECTIVES[0]):
    result = fake_result()
    end = pd.Timestamp(through)
    result.params.update(objective=objective, available_through=str(end.date()),
                         cutoff=str((end + pd.Timedelta(days=1)).date()),
                         train_target_end=str((end - pd.Timedelta(days=30)).date()),
                         val_target_end=str(end.date()))
    return load_checkpoint(save_checkpoint(result, ResultStore(tmp_path / name)).path)


def test_pick_checkpoint_uses_latest_available_before_each_fold(tmp_path):
    from fedenc.checkpoint import pick_checkpoint

    early = _checkpoint_at(tmp_path, "1990-12-31", "early")
    late = _checkpoint_at(tmp_path, "1999-01-31", "late")
    assert pick_checkpoint([late, early], "1999-02-01") is late
    assert pick_checkpoint([late, early], "1999-01-31") is early
    assert pick_checkpoint([early, late]) is late
    with pytest.raises(ValueError, match="must precede"):
        pick_checkpoint([early, late], "1990-12-31")


def test_checkpoint_list_fits_with_latest_qualifying_checkpoint(tmp_path):
    panel = make_synthetic_panel(n_firms=4, n_meetings=6, seed=0)
    # Inner validation is the last meeting, 1999-07-21.
    early = _checkpoint_at(tmp_path, "1990-12-31", "early")
    usable = _checkpoint_at(tmp_path, "1999-06-30", "usable")
    too_late = _checkpoint_at(tmp_path, "1999-07-21", "late")
    arm = TxArm(init="pretrained", adapt="full", checkpoint=[early, too_late, usable],
                max_epochs=1, patience=1, device="cpu")
    arm.fit(panel)
    assert arm.loaded_checkpoint is usable


def test_staged_stage_one_trains_heads_and_leaves_encoder_untouched(checkpoint):
    from fedcore.q3 import ResponseModel
    from fedenc.train import fit_transferred

    panel = make_synthetic_panel(n_firms=4, n_meetings=6, seed=0)
    model = ResponseModel(checkpoint.encoder(), n_fund=panel.fundamentals.shape[1], n_ctx=panel.context.shape[1])
    head_before = {k: v.clone() for k, v in model.head.state_dict().items()}
    # Stage 2 at encoder_lr=0 cannot move the encoder either, so any change came from stage 1.
    result = fit_transferred(model, panel, checkpoint.scaler, warmup_epochs=1, max_epochs=1,
                             patience=1, encoder_lr=0.0)
    assert result.stage_epochs == [1, 1]
    assert len(result.val_loss) == 2
    for key, tensor in checkpoint.state_dict.items():
        assert torch.equal(model.encoder.state_dict()[key].cpu(), tensor)
    assert any(not torch.equal(head_before[k], v) for k, v in model.head.state_dict().items())
    assert all(p.requires_grad for p in model.encoder.parameters())  # stage 2 unfroze it


def test_staged_arm_runs_both_stages_and_needs_pretrained_weights(checkpoint):
    panel = make_synthetic_panel(n_firms=4, n_meetings=6, seed=0)
    arm = TxArm(init="pretrained", adapt="staged", checkpoint=checkpoint, max_epochs=1, patience=1, device="cpu")
    arm.fit(panel)
    assert arm.fit_result.stage_epochs == [1, 1]
    with pytest.raises(ValueError, match="pretrained"):
        TxArm(init="scratch", adapt="staged")


def test_naive_variance_and_zero_mean_split():
    import math

    from fedenc.pretrain_return import diagnose, naive_variance

    corpus = tiny_corpus()
    rows = np.arange(len(corpus))
    history, mask, _ = corpus.batch(rows)
    expected = np.mean(history[:, -63:, 0].astype(np.float64) ** 2, axis=1)
    np.testing.assert_allclose(naive_variance(history, mask), np.maximum(expected, 0.01), rtol=1e-5)

    class Encoder(torch.nn.Module):
        def forward(self, x, mask):
            return torch.ones(len(x), 1)

    class Head(torch.nn.Module):
        def forward(self, h):
            return torch.zeros(len(h)), torch.ones(len(h))

    parts = diagnose(Encoder(), Head(), corpus, date_groups(corpus, rows),
                     {"hist_median": np.zeros(5), "hist_iqr": np.ones(5)}, batch_size=4, device="cpu")
    assert parts["nll_mean_only"] == pytest.approx(parts["nll_naive"])  # a zero mean adds nothing
    assert parts["r2"] == pytest.approx(0.0)
    assert parts["xs_r2"] == pytest.approx(0.0)
    assert math.isnan(parts["xs_corr"])  # two firms per date is too few


@pytest.mark.parametrize("objective", [OBJECTIVES[0], OBJECTIVES[2]])
def test_return_objectives_log_the_mean_variance_split(objective):
    torch.manual_seed(0)
    encoder = PatchTransformerEncoder(d_model=8, n_layers=1, n_heads=2, dim_feedforward=16, dropout=0)
    result = pretrain_return(encoder, tiny_corpus(), objective=objective, epochs=1, batch_size=8,
                             dates_per_step=4, scaler_windows=16, purge_history=False, device="cpu")
    split = {"val_nll_naive", "val_nll_mean_only", "val_r2", "val_xs_r2", "val_xs_corr"}
    assert split <= set(result.curves.columns)
    assert np.isfinite(result.curves[["val_nll_naive", "val_nll_mean_only", "val_r2"]]).all().all()


def test_fold_cutoffs_are_each_folds_first_inner_validation_meeting():
    import fedenc.workflow as workflow
    from fedcore.protocol import walk_forward
    from fedcore.q3.train import validation_meetings

    panel = make_synthetic_panel(n_firms=3, n_meetings=12, seed=0)
    table = workflow.fold_cutoffs(panel, 6, 2)
    folds = walk_forward(panel.meetings(), 6, 2)
    assert len(table) == len(folds) > 1
    for row, fold in zip(table.itertuples(), folds):
        part = panel.subset(np.flatnonzero(np.isin(panel.meeting, fold.train)))
        assert row.cutoff == validation_meetings(part, 0.15).min()
        assert row.cutoff < row.first_test
    assert table.cutoff.is_monotonic_increasing


def test_per_fold_runner_records_which_checkpoint_each_fold_used(tmp_path, monkeypatch):
    import fedcore.q3.real as real
    import fedenc.workflow as workflow
    from fedcore.q3 import ResponseModel
    from fedcore.q3.evaluate import EvalResult
    from fedcore.q3.train import ChannelScaler

    # 8 meetings, min_train 4, test 2: fold cutoffs 1999-04-28 and 1999-07-21.
    dates = ["1999-03-31", "1999-06-30"]
    nxt = [_checkpoint_at(tmp_path, d, f"n{i}") for i, d in enumerate(dates)]
    msk = [_checkpoint_at(tmp_path, d, f"m{i}", OBJECTIVES[1]) for i, d in enumerate(dates)]
    panel = make_synthetic_panel(n_firms=3, n_meetings=8, seed=0)
    monkeypatch.setattr(real, "load_real_panel", lambda **_: (panel, None, {}))

    def fake_walk_forward(panel, factory, **kwargs):
        arm = factory(0)
        arm.model = ResponseModel(arm._encoder(), n_fund=3, n_ctx=4)
        arm.scaler = ChannelScaler.fit(panel)
        return EvalResult(pd.DataFrame({"meeting": [panel.meeting[-1]]}),
                          {"mse_per_meeting": 1.0}, pd.DataFrame({"seed": [0]}),
                          pd.DataFrame({"fold": [0], "seed": [0]}),
                          pd.DataFrame({"epoch": [1], "train_loss": [1.0], "val_loss": [1.0]}))

    monkeypatch.setattr(workflow, "run_walk_forward", fake_walk_forward)
    monkeypatch.setattr(workflow, "compare", lambda *args: {"estimate": 0.0, "n_meetings": 1})

    class FakeDataStore(ResultStore):
        def save(self, *args, **kwargs):
            kwargs["datasets"] = []
            return super().save(*args, **kwargs)

    store = FakeDataStore(tmp_path / "eval")
    summary = workflow.evaluate_checkpoint(
        [c.path for c in nxt], masked_checkpoint=[c.path for c in msk], through="2009-03-18",
        min_train=4, test_size=2, seeds=(0,), arms=("tx_scratch", "tx_pre_staged", "tx_masked_staged"),
        store=store,
    )
    assert summary.arm.tolist() == ["tx_scratch", "tx_pre_staged", "tx_masked_staged"]
    runs = {run.manifest["name"]: run for run in store.runs("Q3")}
    used = runs["encoder-transfer-comparison"].table("checkpoint-by-fold")
    assert used.checkpoint.tolist() == [str(c.path) for c in nxt]
    assert used.masked_checkpoint.tolist() == [str(c.path) for c in msk]
    assert len(runs["tx_pre_staged"].manifest["params"]["checkpoint_runs"]) == 2
