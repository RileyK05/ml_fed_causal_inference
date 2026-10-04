"""Meeting-batched trainer for the Q3 response model.

Scaling (median / IQR, clipped at ±5) is fit on the early-stopping training
meetings only, then reused for validation and test. Unobserved entries are set
to 0 after scaling. The loss is the mean across meetings of within-meeting MSE,
so a meeting with more firms does not count more. Validation is the
chronologically last slice of the meetings passed to ``fit``, never a test fold.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from fedcore.q3.contracts import Q3Panel

MEETINGS_PER_BATCH = 4


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """``None`` picks CUDA when available, else CPU. The trainer follows the model's device."""
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def validation_meetings(panel: Q3Panel, val_frac: float = 0.15) -> pd.DatetimeIndex:
    """Chronologically last slice of ``panel``'s meetings. At least one, and not all of them."""
    meetings = panel.meetings()
    if len(meetings) < 2:
        raise ValueError("need at least two meetings to hold out a validation slice")
    n_val = int(round(float(val_frac) * len(meetings)))
    n_val = min(max(n_val, 1), len(meetings) - 1)
    return pd.DatetimeIndex(meetings[-n_val:])


def per_meeting_mse(mu: torch.Tensor, target: torch.Tensor, meeting_codes: torch.Tensor) -> torch.Tensor:
    """Mean over meetings of the within-meeting MSE. Every meeting weighs the same."""
    err = (mu.reshape(-1) - target.reshape(-1)) ** 2
    codes = meeting_codes.reshape(-1).long()
    _, inverse = torch.unique(codes, return_inverse=True)
    width = int(inverse.max().item()) + 1
    sse = err.new_zeros(width)
    count = err.new_zeros(width)
    sse.scatter_add_(0, inverse, err)
    count.scatter_add_(0, inverse, torch.ones_like(err))
    return (sse / count.clamp(min=1)).mean()


@dataclass(frozen=True)
class ChannelScaler:
    """Per-channel median and IQR for history, fundamentals and context."""

    hist_median: np.ndarray
    hist_iqr: np.ndarray
    fund_median: np.ndarray
    fund_iqr: np.ndarray
    ctx_median: np.ndarray
    ctx_iqr: np.ndarray

    @classmethod
    def fit(cls, panel: Q3Panel, rows: np.ndarray | None = None) -> "ChannelScaler":
        if rows is not None:
            panel = panel.subset(np.asarray(rows))
        hist_median, hist_iqr = _columns(panel.history, panel.history_mask)
        fund_median, fund_iqr = _columns(panel.fundamentals, panel.fundamentals_mask)
        ctx_median, ctx_iqr = _columns(panel.context, np.ones(panel.context.shape, dtype=bool))
        return cls(hist_median, hist_iqr, fund_median, fund_iqr, ctx_median, ctx_iqr)

    def transform(self, panel: Q3Panel) -> dict[str, np.ndarray]:
        history = _scale(panel.history, self.hist_median, self.hist_iqr)
        history = np.where(panel.history_mask[..., None], history, np.float32(0.0))
        fundamentals = _scale(panel.fundamentals, self.fund_median, self.fund_iqr)
        fundamentals = np.where(panel.fundamentals_mask, fundamentals, np.float32(0.0))
        context = _scale(panel.context, self.ctx_median, self.ctx_iqr)
        return {
            "history": np.nan_to_num(history, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False),
            "fundamentals": np.nan_to_num(fundamentals, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False),
            "context": np.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False),
        }


@dataclass
class FitResult:
    model: nn.Module
    scaler: ChannelScaler
    train_loss: list[float]
    val_loss: list[float]
    best_epoch: int
    best_val_loss: float
    train_meetings: pd.DatetimeIndex
    val_meetings: pd.DatetimeIndex


def fit(
    model: nn.Module,
    train_panel: Q3Panel,
    *,
    val_frac: float = 0.15,
    lr: float = 1e-3,
    encoder_lr: float | None = None,
    weight_decay: float = 1e-4,
    max_epochs: int = 100,
    patience: int = 10,
    seed: int = 0,
    freeze_encoder: bool = False,
    meetings_per_batch: int = MEETINGS_PER_BATCH,
) -> FitResult:
    """Train on whole-meeting batches. Restore the weights with the lowest validation loss.

    ``meetings_per_batch`` is 1 to 4. The loss of a shorter tail batch is scaled by
    ``n_meetings / meetings_per_batch`` so each meeting's gradient has the same weight.
    Curves are the eval-mode meeting-MSE (dropout off), which is what early stopping uses.
    """
    if max_epochs < 1:
        raise ValueError("max_epochs must be positive")
    if not 1 <= int(meetings_per_batch) <= 4:
        raise ValueError("meetings_per_batch must be from 1 to 4")
    train_panel.validate()
    seed_everything(seed)
    meetings = train_panel.meetings()
    val_m = validation_meetings(train_panel, val_frac)
    train_m = pd.DatetimeIndex(meetings[:-len(val_m)])
    train_rows = _meeting_rows(train_panel, train_m)
    val_rows = _meeting_rows(train_panel, val_m)
    train_part = train_panel.subset(train_rows)
    val_part = train_panel.subset(val_rows)
    scaler = ChannelScaler.fit(train_part)
    device = next(model.parameters()).device
    train_batch = _tensors(train_part, scaler, device)
    val_batch = _tensors(val_part, scaler, device)

    groups = _param_groups(model, lr, encoder_lr, weight_decay, freeze_encoder)
    opt = torch.optim.AdamW(groups)
    rng = np.random.default_rng(seed)
    groups_idx = _meeting_index_groups(train_part.meeting)
    n_train = len(groups_idx)

    best_val = math.inf
    best_epoch = 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    stale = 0
    train_curve: list[float] = []
    val_curve: list[float] = []
    for epoch in range(1, max_epochs + 1):
        _train_epoch(
            model, opt, train_batch, groups_idx, rng, meetings_per_batch, n_train, freeze_encoder
        )
        tr_loss = _eval_loss(model, train_batch, freeze_encoder)
        va_loss = _eval_loss(model, val_batch, freeze_encoder)
        if not math.isfinite(va_loss):
            raise RuntimeError(f"validation loss is {va_loss} at epoch {epoch}")
        train_curve.append(tr_loss)
        val_curve.append(va_loss)
        if va_loss < best_val:
            best_val = va_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return FitResult(
        model=model,
        scaler=scaler,
        train_loss=train_curve,
        val_loss=val_curve,
        best_epoch=best_epoch,
        best_val_loss=float(best_val),
        train_meetings=train_m,
        val_meetings=val_m,
    )


def predict(model: nn.Module, panel: Q3Panel, scaler: ChannelScaler) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``mu, a, b`` as float64 arrays aligned with ``panel`` rows."""
    model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    scaled = scaler.transform(panel)
    n = panel.target.shape[0]
    mu_out = np.empty(n, dtype=np.float64)
    a_out = np.empty(n, dtype=np.float64)
    b_out = np.empty(n, dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, n, 2048):
            stop = min(start + 2048, n)
            mu, a, b = model(
                _chunk(scaled["history"], start, stop, device, dtype),
                _chunk(panel.history_mask, start, stop, device, torch.bool),
                _chunk(scaled["fundamentals"], start, stop, device, dtype),
                _chunk(panel.fundamentals_mask, start, stop, device, torch.bool),
                _chunk(scaled["context"], start, stop, device, dtype),
                _chunk(panel.shock, start, stop, device, dtype),
            )
            mu_out[start:stop] = mu.detach().cpu().numpy()
            a_out[start:stop] = a.detach().cpu().numpy()
            b_out[start:stop] = b.detach().cpu().numpy()
    return mu_out, a_out, b_out


def _train_epoch(model, opt, batch, groups_idx, rng, meetings_per_batch, n_train, freeze_encoder) -> None:
    model.train()
    if freeze_encoder:
        model.encoder.eval()
    order = rng.permutation(n_train)
    nominal = float(meetings_per_batch)
    for start in range(0, n_train, meetings_per_batch):
        chosen = order[start : start + meetings_per_batch]
        idx = np.concatenate([groups_idx[int(j)] for j in chosen])
        index = torch.as_tensor(idx, device=batch["history"].device)
        opt.zero_grad(set_to_none=True)
        mu, _, _ = _forward(model, batch, index)
        n_m = len(chosen)
        loss = per_meeting_mse(mu, batch["target"][index], batch["codes"][index]) * (n_m / nominal)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


def _eval_loss(model, batch, freeze_encoder) -> float:
    model.eval()
    with torch.no_grad():
        mu, _, _ = _forward(model, batch, slice(None))
        value = float(per_meeting_mse(mu, batch["target"], batch["codes"]).item())
    model.train()
    if freeze_encoder:
        model.encoder.eval()
    return value


def _forward(model, batch, index):
    return model(
        batch["history"][index],
        batch["history_mask"][index],
        batch["fundamentals"][index],
        batch["fundamentals_mask"][index],
        batch["context"][index],
        batch["shock"][index],
    )


def _param_groups(model, lr, encoder_lr, weight_decay, freeze_encoder):
    if (freeze_encoder or encoder_lr is not None) and not hasattr(model, "encoder"):
        raise ValueError("encoder_lr and freeze_encoder require model.encoder")
    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad_(False)
    enc_ids = {id(p) for p in model.encoder.parameters()} if hasattr(model, "encoder") else set()
    enc_params = []
    others = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if id(param) in enc_ids:
            enc_params.append(param)
        else:
            others.append(param)
    groups = []
    if others:
        groups.append({"params": others, "lr": lr, "weight_decay": weight_decay})
    if enc_params:
        groups.append({
            "params": enc_params,
            "lr": lr if encoder_lr is None else encoder_lr,
            "weight_decay": weight_decay,
        })
    if not groups:
        raise ValueError("model has no trainable parameters")
    return groups


def _meeting_rows(panel: Q3Panel, meetings: pd.DatetimeIndex) -> np.ndarray:
    selected = pd.DatetimeIndex(meetings).to_numpy(dtype="datetime64[ns]")
    own = np.asarray(panel.meeting, dtype="datetime64[ns]")
    return np.flatnonzero(np.isin(own, selected))


def _meeting_index_groups(meeting: np.ndarray) -> list[np.ndarray]:
    values = np.asarray(meeting, dtype="datetime64[ns]")
    groups = []
    for day in np.unique(values):
        groups.append(np.flatnonzero(values == day))
    return groups


def _tensors(panel: Q3Panel, scaler: ChannelScaler, device: torch.device) -> dict:
    scaled = scaler.transform(panel)
    codes = pd.factorize(np.asarray(panel.meeting, dtype="datetime64[ns]"), sort=True)[0]

    def as_tensor(array, dtype):
        tensor = torch.from_numpy(np.ascontiguousarray(array))
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        if device.type != "cpu":
            tensor = tensor.to(device)
        return tensor

    return {
        "history": as_tensor(scaled["history"], torch.float32),
        "history_mask": as_tensor(panel.history_mask, torch.bool),
        "fundamentals": as_tensor(scaled["fundamentals"], torch.float32),
        "fundamentals_mask": as_tensor(panel.fundamentals_mask, torch.bool),
        "context": as_tensor(scaled["context"], torch.float32),
        "shock": as_tensor(np.asarray(panel.shock, dtype=np.float32), torch.float32),
        "target": as_tensor(np.asarray(panel.target, dtype=np.float32), torch.float32),
        "codes": as_tensor(codes.astype(np.int64), torch.long),
    }


def _chunk(array: np.ndarray, start: int, stop: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(array[start:stop]))
    return tensor.to(device=device, dtype=dtype)


def _columns(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median and IQR of observed entries, one pair per trailing axis.

    ``mask`` either matches ``values`` or is missing the channel axis (history).
    """
    arr = np.asarray(values, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if valid.shape != arr.shape:
        if valid.shape == arr.shape[:-1]:
            valid = np.broadcast_to(valid[..., None], arr.shape)
        else:
            raise ValueError(f"mask shape {mask.shape} does not match values {arr.shape}")
    valid = valid & np.isfinite(arr)
    n_cols = arr.shape[-1]
    median = np.zeros(n_cols, dtype=np.float64)
    iqr = np.ones(n_cols, dtype=np.float64)
    flat = arr.reshape(-1, n_cols)
    flat_mask = valid.reshape(-1, n_cols)
    for c in range(n_cols):
        observed = flat[flat_mask[:, c], c]
        median[c], iqr[c] = _med_iqr(observed)
    return median, iqr


def _med_iqr(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 1.0
    q25, med, q75 = np.quantile(values, [0.25, 0.5, 0.75])
    spread = float(q75 - q25)
    if spread < 1e-6:
        spread = 1.0
    return float(med), spread


def _scale(values: np.ndarray, median: np.ndarray, iqr: np.ndarray) -> np.ndarray:
    z = (np.asarray(values, dtype=np.float64) - median) / iqr
    return np.clip(z, -5.0, 5.0).astype(np.float32)
