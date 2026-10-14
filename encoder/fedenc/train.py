"""Encoder-local adaptation using an immutable pretrained history scaler.

Reuses the shared meeting optimizer/loss/evaluation helpers. Only history scaling
differs: fundamentals and context still fit exclusively on adaptation training meetings.

``warmup_epochs > 0`` adapts in two stages. Stage 1 trains the response heads with the
encoder frozen, so random heads cannot scramble pretrained features. Stage 2 starts from
the best stage-1 weights and unfreezes the encoder (at ``encoder_lr``). Early stopping
runs per stage on the same validation meetings; the best weights across both stages win.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch

from fedcore.q3.train import (
    ChannelScaler, FitResult, _columns, _eval_loss, _meeting_index_groups, _meeting_rows,
    _param_groups, _tensors, _train_epoch, seed_everything, validation_meetings,
)


def fit_transferred(model, panel, history_scaler, *, val_frac=0.15, lr=1e-3,
                    encoder_lr=None, weight_decay=1e-4, max_epochs=40, patience=10,
                    seed=0, freeze_encoder=False, warmup_epochs=0, meetings_per_batch=4):
    if max_epochs < 1 or patience < 1 or warmup_epochs < 0 or not 1 <= meetings_per_batch <= 4:
        raise ValueError("invalid adaptation epoch/patience/batch settings")
    if not 0 < val_frac < 1:
        raise ValueError("val_frac must be between zero and one")
    panel.validate()
    seed_everything(seed)
    val_m = validation_meetings(panel, val_frac)
    train_m = panel.meetings()[:-len(val_m)]
    train_part = panel.subset(_meeting_rows(panel, train_m))
    val_part = panel.subset(_meeting_rows(panel, val_m))
    fund_median, fund_iqr = _columns(train_part.fundamentals, train_part.fundamentals_mask)
    ctx_median, ctx_iqr = _columns(train_part.context, np.ones(train_part.context.shape, bool))
    scaler = ChannelScaler(np.asarray(history_scaler["hist_median"]).copy(),
                           np.asarray(history_scaler["hist_iqr"]).copy(),
                           fund_median, fund_iqr, ctx_median, ctx_iqr)
    device = next(model.parameters()).device
    train_batch = _tensors(train_part, scaler, device)
    val_batch = _tensors(val_part, scaler, device)
    rng = np.random.default_rng(seed)
    groups = _meeting_index_groups(train_part.meeting)
    # Remember what the arm left trainable (e.g. "top" freezes the lower stack) so stage 2
    # unfreezes only what stage 1 froze.
    trainable = {name: p.requires_grad for name, p in model.named_parameters()}
    stages = [(True, warmup_epochs)] if warmup_epochs and not freeze_encoder else []
    stages.append((freeze_encoder, max_epochs))
    best_val, best_epoch, best_state = math.inf, 0, None
    train_curve, val_curve, stage_epochs = [], [], []
    for frozen, epochs in stages:
        for name, param in model.named_parameters():
            param.requires_grad_(trainable[name])
        if best_state is not None:
            model.load_state_dict(best_state)
        optimizer = torch.optim.AdamW(_param_groups(model, lr, encoder_lr, weight_decay, frozen))
        stale, ran = 0, 0
        for _ in range(epochs):
            _train_epoch(model, optimizer, train_batch, groups, rng, meetings_per_batch, len(groups), frozen)
            tr = _eval_loss(model, train_batch, frozen)
            va = _eval_loss(model, val_batch, frozen)
            if not math.isfinite(tr) or not math.isfinite(va):
                raise RuntimeError("nonfinite adaptation loss")
            train_curve.append(tr)
            val_curve.append(va)
            ran += 1
            if va < best_val:
                best_val, best_epoch, stale = va, len(val_curve), 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break
        stage_epochs.append(ran)
    model.load_state_dict(best_state)
    model.eval()
    result = FitResult(model, scaler, train_curve, val_curve, best_epoch, best_val,
                       pd.DatetimeIndex(train_m), val_m)
    result.stage_epochs = stage_epochs  # epochs run per stage; best_epoch counts across stages
    return result
