"""Streaming pretraining of history state: predict the next raw return's mean and size.

The head estimates E[r_next | history] (percent) and Var[r_next | history] (percent
squared) under a Gaussian score. It is temporary; Fed adaptation retains the encoder
and its history scaler only. Masked reconstruction is an alternative objective on the
same dated examples.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from fedcore.q3.train import resolve_device, seed_everything
from fedenc.pretrain import _choose_masks, _fit_hist_scaler, _scale_windows
from fedenc.return_corpus import ReturnCorpus

OBJECTIVES = ("next_return_gaussian", "masked_reconstruction")


class GaussianHead(nn.Module):
    """Expected next-day return and its variance, both from the pooled history state."""

    def __init__(self, width, floor=1e-6):
        super().__init__()
        if not math.isfinite(floor) or floor <= 0:
            raise ValueError("variance floor must be finite and positive")
        self.floor = float(floor)
        self.net = nn.Sequential(nn.Linear(width, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, h):
        out = self.net(h)
        return out[..., 0], F.softplus(out[..., 1]) + self.floor


def gaussian_loss(mean, var, returns):
    """Gaussian NLL without its constant; optimum is the conditional mean and variance."""
    return torch.log(var) + (returns - mean).square() / var


@dataclass
class ReturnPretrainResult:
    encoder: nn.Module
    state_dict: dict
    scaler: dict
    curves: pd.DataFrame
    best_epoch: int
    best_val_loss: float
    params: dict


def date_groups(corpus, rows):
    rows = np.asarray(rows, dtype=np.int64)
    dates = corpus.target_dates.to_numpy()[rows]
    order = np.argsort(dates, kind="stable")
    cuts = np.flatnonzero(dates[order][1:] != dates[order][:-1]) + 1
    return list(np.split(rows[order], cuts))


def fit_scaler(corpus, rows, *, max_windows=4096, seed=0):
    """Bounded, deterministic sample of training histories only for robust quantiles."""
    if max_windows < 1:
        raise ValueError("max_windows must be positive")
    rows = np.asarray(rows)
    if len(rows) > max_windows:
        rows = np.random.default_rng(seed).choice(rows, max_windows, replace=False)
    history, mask, _ = corpus.batch(rows)
    median, iqr = _fit_hist_scaler(history, mask)
    return {"hist_median": median, "hist_iqr": iqr}


def _losses(encoder, head, corpus, rows, scaler, objective, mask_frac, rng, device):
    history, mask, returns = corpus.batch(rows)
    scaled = _scale_windows(history, mask, scaler["hist_median"], scaler["hist_iqr"])
    x = torch.as_tensor(scaled, device=device)
    m = torch.as_tensor(mask, device=device)
    if objective == OBJECTIVES[0]:
        mean, var = head(encoder(x, m))
        return gaussian_loss(mean, var, torch.as_tensor(returns, device=device))
    choice = _choose_masks(encoder, mask, np.arange(len(rows)), mask_frac, rng)
    choice = torch.as_tensor(choice, device=device)
    tokens, valid = encoder.forward_tokens(x, m, patch_mask=choice)
    patches, _ = encoder.patchify(x, m)
    p, c = encoder.patch_len, encoder.n_channels
    target = patches[..., :p*c].reshape(len(rows), encoder.n_patches, p, c)
    bits = patches[..., p*c:].bool()
    observed = bits & (choice & valid).unsqueeze(-1)
    pred = head(tokens).reshape_as(target)
    error = (pred - target).square() * observed.unsqueeze(-1)
    denominator = observed.sum(dim=(1, 2)).clamp(min=1).unsqueeze(-1)
    return (error.sum(dim=(1, 2)) / denominator).mean(dim=-1)


def evaluate(encoder, head, corpus, groups, scaler, *, objective, mask_frac, batch_size, seed, device):
    """Exact mean of per-date means; validation masks repeat across epochs."""
    encoder.eval()
    head.eval()
    rng = np.random.default_rng(seed)
    total = 0.0
    with torch.inference_mode():
        for group in groups:
            date_sum = 0.0
            for start in range(0, len(group), batch_size):
                rows = group[start:start + batch_size]
                losses = _losses(encoder, head, corpus, rows, scaler, objective, mask_frac, rng, device)
                date_sum += float(losses.double().sum())
            total += date_sum / len(group)
    return total / len(groups)


def pretrain_return(
    encoder, corpus: ReturnCorpus, *, objective=OBJECTIVES[0], epochs=30, lr=1e-3,
    seed=0, val_frac=0.1, batch_size=256, dates_per_step=4, weight_decay=1e-4,
    patience=5, scaler_windows=4096, mask_frac=0.3, purge_history=True, train_eval_dates=250,
    device=None, verbose=False,
) -> ReturnPretrainResult:
    """One epoch visits every eligible row; optimizer steps give each target date equal weight.

    Windows are materialized a batch at a time. No checkpoint or run is saved here.
    Encoder initialization must be seeded by the caller before constructing it.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    if min(epochs, batch_size, dates_per_step, patience, scaler_windows, train_eval_dates) < 1:
        raise ValueError("epochs, batch_size, dates_per_step, patience, scaler_windows and train_eval_dates must be positive")
    if not math.isfinite(lr) or lr <= 0 or not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("invalid optimizer settings")
    if not 0 < mask_frac < 1:
        raise ValueError("mask_frac must be between zero and one")
    seed_everything(seed)
    device = resolve_device(device)
    encoder.to(device)
    train_rows, val_rows = corpus.split(val_frac, purge_history=purge_history)
    scaler = fit_scaler(corpus, train_rows, max_windows=scaler_windows, seed=seed)
    train_groups, val_groups = date_groups(corpus, train_rows), date_groups(corpus, val_rows)
    # Train loss is only a monitor: score a fixed sample of training dates, not the whole corpus.
    pick = np.random.default_rng(seed).choice(len(train_groups), min(train_eval_dates, len(train_groups)), replace=False)
    train_eval_groups = [train_groups[i] for i in np.sort(pick)]
    head = (GaussianHead(encoder.out_dim) if objective == OBJECTIVES[0] else nn.Sequential(
        nn.Linear(encoder.d_model, 128), nn.ReLU(),
        nn.Linear(128, encoder.patch_len * encoder.n_channels),
    )).to(device)
    parameters = [p for p in encoder.parameters() if p.requires_grad] + list(head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    curves, best_state, best_val, best_epoch, stale = [], None, math.inf, 0, 0
    for epoch in range(1, epochs + 1):
        encoder.train()
        head.train()
        order = rng.permutation(len(train_groups))
        for offset in range(0, len(order), dates_per_step):
            optimizer.zero_grad(set_to_none=True)
            for group_index in order[offset:offset + dates_per_step]:
                group = rng.permutation(train_groups[group_index])
                for start in range(0, len(group), batch_size):
                    rows = group[start:start + batch_size]
                    losses = _losses(encoder, head, corpus, rows, scaler, objective, mask_frac, rng, device)
                    loss = losses.sum() / (len(group) * dates_per_step)
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError("nonfinite pretraining loss")
                    loss.backward()
            nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
        args = dict(objective=objective, mask_frac=mask_frac, batch_size=batch_size,
                    seed=seed + 1, device=device)
        tr = evaluate(encoder, head, corpus, train_eval_groups, scaler, **args)
        va = evaluate(encoder, head, corpus, val_groups, scaler, **args)
        if not math.isfinite(tr) or not math.isfinite(va):
            raise RuntimeError("nonfinite pretraining evaluation loss")
        curves.append({"epoch": epoch, "train_loss": tr, "val_loss": va})
        if verbose:
            print(f"epoch={epoch} train={tr:.6g} val={va:.6g}", flush=True)
        if va < best_val:
            best_val, best_epoch, stale = va, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    encoder.load_state_dict(best_state)
    encoder.eval()
    params = dict(
        objective=objective, epochs=epochs, lr=lr, seed=seed, val_frac=val_frac,
        batch_size=batch_size, dates_per_step=dates_per_step, weight_decay=weight_decay,
        patience=patience, scaler_windows=scaler_windows, mask_frac=mask_frac,
        train_eval_dates=len(train_eval_groups),
        cutoff=str(corpus.source.cutoff.date()),
        available_through=str(corpus.target_dates.max().date()),
        train_target_start=str(corpus.target_dates[train_rows].min().date()),
        train_target_end=str(corpus.target_dates[train_rows].max().date()),
        val_target_start=str(corpus.target_dates[val_rows].min().date()),
        val_target_end=str(corpus.target_dates[val_rows].max().date()),
        n_train_rows=len(train_rows), n_val_rows=len(val_rows),
        n_purged_rows=len(corpus) - len(train_rows) - len(val_rows),
        n_train_dates=len(train_groups), n_val_dates=len(val_groups),
        return_units="percent", weighting="equal target dates, equal firms within date",
        purge_history=purge_history,
        history_overlap="purged" if purge_history else "allowed past context; disjoint target dates",
        excluded_fomc_dates=len(corpus.excluded_dates),
        device=str(device), torch_version=str(torch.__version__),
    )
    return ReturnPretrainResult(encoder, best_state, scaler, pd.DataFrame(curves), best_epoch, best_val, params)
