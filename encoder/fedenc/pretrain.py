"""Masked-patch pre-training for the patch transformer (docs/encoder/transformer.md section 8).

Hide 30% of the valid patches of ordinary-day windows behind a learned [MASK]
token, reconstruct the observed scaled values of the hidden patches with a small
decoder, and throw the decoder away. The loss is the mean of per-channel MSE over
observed entries of masked patches only.

Chronology: the validation split is a calendar block, the last ``val_frac`` of
windows by end date. ``make_pretrain_windows`` does not return end dates, so they
are rebuilt by chaining windows that overlap exactly (two windows of one firm
whose ends differ by ``d`` days share ``L - d`` exact day records). On real data
the checkpoint itself must also respect chronology: one checkpoint per fold,
trained only on windows ending before that fold's test meetings.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import os

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from fedcore.q3.train import seed_everything


@dataclass
class EndOrder:
    """Sort keys for sampled windows: larger means a later end date."""

    key: np.ndarray
    method: str
    n_chains: int
    n_chained: int


@dataclass
class PretrainResult:
    encoder: nn.Module
    state_dict: dict
    train_loss: list[float]
    val_loss: list[float]
    best_epoch: int
    best_val_loss: float
    curves: pd.DataFrame
    end_order: EndOrder
    scaler: dict = field(default_factory=dict)
    n_masked: int = 0

    @property
    def split(self) -> str:
        return self.end_order.method


def window_end_order(history: np.ndarray, mask: np.ndarray, max_gap: int = 64) -> EndOrder:
    """Reconstruct each sampled window's end-date order by exact overlap chaining.

    Windows are slices of a few firms' paths, so two windows of the same firm
    whose end dates differ by ``d`` days share ``L - d`` exact day records
    (masked days all carry the same record and are handled through each window's
    first observed day). Chaining those links orders every window relative to its
    firm's path; chains are aligned on their normalized span, which is enough for
    a calendar-block split. Falls back to draw order when the corpus is too small
    or too fragmented to chain -- ``EndOrder.method`` reports which was used.
    """
    n, t = mask.shape
    if history.shape[:2] != (n, t):
        raise ValueError(f"history {history.shape} does not match mask {mask.shape}")
    if n < 8:
        return EndOrder(np.arange(n, dtype=np.float64), "draw_order", n, 0)

    day_mask = np.asarray(mask, dtype=bool)
    vals = np.where(day_mask[..., None], np.nan_to_num(history, nan=0.0), np.float32(0.0)) + np.float32(0.0)
    raw = np.concatenate(
        (
            np.ascontiguousarray(vals.astype(np.float32)).view(np.uint8).reshape(n, t, -1),
            day_mask.astype(np.uint8)[..., None],
        ),
        axis=2,
    )
    item = raw.shape[-1]
    if os.environ.get("FEDENC_VERBOSE"):
        print(f"[end-order] uniquing {n * t} day records", flush=True)
    flat = np.ascontiguousarray(raw.reshape(n * t, item)).view(f"V{item}").reshape(n * t)
    day_id = np.unique(flat, return_inverse=True)[1].reshape(n, t).astype(np.int64)

    if os.environ.get("FEDENC_VERBOSE"):
        print(f"[end-order] uniquing {n} windows", flush=True)
    win_view = np.ascontiguousarray(day_id).view(f"V{8 * t}").reshape(n)
    _, first_idx, win_id = np.unique(win_view, return_index=True, return_inverse=True)
    win_days = day_id[first_idx]
    u = win_days.shape[0]

    if os.environ.get("FEDENC_VERBOSE"):
        print(f"[end-order] chaining {u} unique windows", flush=True)

    obs = day_mask[first_idx]
    has_obs = obs.any(axis=1)
    first_obs = np.argmax(obs, axis=1)
    lead = np.where(obs, np.arange(t)[None, :], t)
    next_obs = np.minimum.accumulate(lead[:, ::-1], axis=1)[:, ::-1]

    start_map: dict[int, list[tuple[int, int]]] = {}
    for v in range(u):
        if has_obs[v]:
            o = int(first_obs[v])
            start_map.setdefault(int(win_days[v, o]), []).append((v, o))

    succ: dict[int, tuple[int, int]] = {}
    for w in range(u):
        if not has_obs[w]:
            continue
        for gap in range(1, min(max_gap, t - 1) + 1):
            p = int(next_obs[w, gap])
            if p >= t:
                continue
            for v, o in start_map.get(int(win_days[w, p]), ()):
                if v == w or o != p - gap:
                    continue
                if np.array_equal(win_days[v, : t - gap], win_days[w, gap:]):
                    succ[w] = (v, gap)
                    break
            if w in succ:
                break

    has_pred = {v for v, _ in succ.values()}
    rel = np.zeros(u, dtype=np.float64)
    key_u = np.zeros(u, dtype=np.float64)
    seen = np.zeros(u, dtype=bool)
    n_chains = 0
    n_chained = 0
    for w in range(u):
        if seen[w] or w in has_pred:
            continue
        chain = [w]
        seen[w] = True
        cur = w
        while cur in succ:
            nxt, gap = succ[cur]
            if seen[nxt]:
                break
            rel[nxt] = rel[cur] + gap
            seen[nxt] = True
            chain.append(nxt)
            cur = nxt
        lo = rel[chain].min()
        hi = rel[chain].max()
        span = hi - lo
        for node in chain:
            key_u[node] = 0.5 if span <= 0 else (rel[node] - lo) / span
        n_chains += 1
        if len(chain) > 1:
            n_chained += len(chain)
    for w in range(u):  # defensive: any node still unseen is its own chain
        if not seen[w]:
            key_u[w] = 0.5
            seen[w] = True
            n_chains += 1

    key = key_u[win_id]
    method = "overlap_chain" if n_chains <= max(8, n // 50) else "draw_order"
    if method == "draw_order":
        key = np.arange(n, dtype=np.float64)
    return EndOrder(key, method, n_chains, n_chained)


def pretrain(
    encoder: nn.Module,
    windows: np.ndarray,
    mask: np.ndarray,
    *,
    epochs: int = 30,
    lr: float = 1e-3,
    seed: int = 0,
    val_frac: float = 0.1,
    mask_frac: float = 0.3,
    batch_size: int = 256,
    weight_decay: float = 1e-4,
    patience: int = 5,
    device: torch.device | str | None = None,
) -> PretrainResult:
    """Masked-patch reconstruction. The encoder is trained in place; the decoder is local and discarded.

    ``windows`` is (N, L, C) float with NaN at unobserved days and ``mask`` is
    (N, L) bool. The scaler is fit on the training windows only (the pre-training
    corpus is its own universe). The validation block is the last ``val_frac`` of
    windows by reconstructed end date. ``train_loss`` is the epoch's mean loss per
    observed entry; ``val_loss`` is the full validation block under one fixed
    mask draw.
    """
    windows = np.asarray(windows, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if windows.ndim != 3 or mask.shape != windows.shape[:2]:
        raise ValueError(f"expected windows (N, L, C) and mask (N, L), got {windows.shape} {mask.shape}")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    n = windows.shape[0]
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device(device) if device is not None else next(encoder.parameters()).device

    verbose = bool(os.environ.get("FEDENC_VERBOSE"))
    if verbose:
        print(f"[pretrain] windows={n} ordering end dates", flush=True)
    order = window_end_order(windows, mask)
    if verbose:
        print(f"[pretrain] end-order method={order.method} chains={order.n_chains}", flush=True)
    # Cut by key value so duplicate windows (identical keys) never straddle the split.
    thr = float(np.quantile(order.key, 1.0 - val_frac))
    val_rows = np.flatnonzero(order.key >= thr)
    train_rows = np.flatnonzero(order.key < thr)
    if train_rows.size == 0 or val_rows.size == 0:
        rank = np.argsort(order.key, kind="stable")
        n_val = min(max(1, int(round(val_frac * n))), max(1, n - 1))
        val_rows = rank[-n_val:]
        train_rows = rank[:-n_val] if n > n_val else rank

    median, iqr = _fit_hist_scaler(windows[train_rows], mask[train_rows])
    scaled = _scale_windows(windows, mask, median, iqr)

    patch_len = int(getattr(encoder, "patch_len", 5))
    n_channels = int(getattr(encoder, "n_channels", windows.shape[-1]))
    d_model = int(getattr(encoder, "d_model", 64))
    decoder = nn.Sequential(nn.Linear(d_model, 128), nn.ReLU(), nn.Linear(128, patch_len * n_channels)).to(dev)
    encoder.to(dev)

    params = [p for p in encoder.parameters() if p.requires_grad] + list(decoder.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    val_in = _batch_tensors(scaled, mask, val_rows, dev)
    val_choice = _choose_masks(encoder, mask, val_rows, mask_frac, np.random.default_rng(seed + 1))

    train_curve: list[float] = []
    val_curve: list[float] = []
    best_val = float("inf")
    best_epoch = 0
    best_state = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
    stale = 0
    n_masked = 0
    for epoch in range(1, epochs + 1):
        encoder.train()
        order_rng = rng.permutation(train_rows.size)
        total = 0.0
        seen = 0
        n_masked = 0
        for start in range(0, train_rows.size, batch_size):
            idx = train_rows[order_rng[start : start + batch_size]]
            batch = _batch_tensors(scaled, mask, idx, dev)
            choice = _choose_masks(encoder, mask, idx, mask_frac, rng)
            loss, count = _masked_loss(encoder, decoder, batch, choice)
            if count == 0:
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 1.0)
            opt.step()
            total += float(loss.item()) * count
            seen += count
            n_masked += int(choice.sum())
        train_curve.append(total / max(seen, 1))
        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"[pretrain] epoch {epoch} train={train_curve[-1]:.5f}", flush=True)

        encoder.eval()
        with torch.no_grad():
            val_loss, val_count = _masked_loss(encoder, decoder, val_in, val_choice)
        if val_count == 0:
            raise RuntimeError("no observed entry in any validation mask")
        val_curve.append(float(val_loss.item()))
        if val_curve[-1] < best_val:
            best_val = val_curve[-1]
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    encoder.load_state_dict(best_state)
    encoder.eval()
    curves = pd.DataFrame(
        {"epoch": np.arange(1, len(train_curve) + 1), "train_loss": train_curve, "val_loss": val_curve}
    )
    return PretrainResult(
        encoder=encoder,
        state_dict={k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
        train_loss=train_curve,
        val_loss=val_curve,
        best_epoch=best_epoch,
        best_val_loss=float(best_val),
        curves=curves,
        end_order=order,
        scaler={"hist_median": median, "hist_iqr": iqr},
        n_masked=n_masked,
    )


def _fit_hist_scaler(windows: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel median and IQR of observed entries. Mirrors ChannelScaler for history only."""
    arr = np.asarray(windows, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)[..., None] & np.isfinite(arr)
    n_cols = arr.shape[-1]
    median = np.zeros(n_cols, dtype=np.float64)
    iqr = np.ones(n_cols, dtype=np.float64)
    flat = arr.reshape(-1, n_cols)
    flat_mask = valid.reshape(-1, n_cols)
    for ch in range(n_cols):
        observed = flat[flat_mask[:, ch], ch]
        if observed.size == 0:
            continue
        q25, med, q75 = np.quantile(observed, [0.25, 0.5, 0.75])
        spread = float(q75 - q25)
        median[ch] = float(med)
        iqr[ch] = spread if spread >= 1e-6 else 1.0
    return median, iqr


def _scale_windows(windows: np.ndarray, mask: np.ndarray, median: np.ndarray, iqr: np.ndarray) -> np.ndarray:
    z = (np.asarray(windows, dtype=np.float64) - median) / iqr
    z = np.clip(z, -5.0, 5.0)
    z = np.where(np.asarray(mask, bool)[..., None], z, 0.0)
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _batch_tensors(scaled: np.ndarray, mask: np.ndarray, rows: np.ndarray, device: torch.device) -> dict:
    def as_tensor(array, dtype):
        tensor = torch.from_numpy(np.ascontiguousarray(array[rows]))
        if device.type != "cpu":
            tensor = tensor.to(device)
        return tensor.to(dtype=dtype)

    return {"history": as_tensor(scaled, torch.float32), "mask": as_tensor(mask, torch.bool)}


def _choose_masks(
    encoder: nn.Module, mask: np.ndarray, rows: np.ndarray, mask_frac: float, rng: np.random.Generator
) -> np.ndarray:
    """One pre-training patch mask per window: ``mask_frac`` of the valid patches, at random."""
    sub = np.asarray(mask[rows], dtype=bool)
    b, t = sub.shape
    p = int(getattr(encoder, "patch_len", 5))
    n_patches = int(getattr(encoder, "n_patches", -(-t // p)))
    pad = n_patches * p - t
    if pad > 0:
        sub = np.pad(sub, ((0, 0), (pad, 0)))
    patch_valid = sub.reshape(b, n_patches, p).any(axis=-1)
    choice = np.zeros((b, n_patches), dtype=bool)
    counts = patch_valid.sum(axis=1)
    for i in range(b):
        if counts[i] == 0:
            continue
        k = min(max(int(round(mask_frac * counts[i])), 1), int(counts[i]))
        pick = rng.choice(np.flatnonzero(patch_valid[i]), size=k, replace=False)
        choice[i, pick] = True
    return choice


def _masked_loss(
    encoder: nn.Module, decoder: nn.Module, batch: dict, choice: Tensor | np.ndarray
) -> tuple[Tensor, int]:
    """Per-channel MSE on observed entries of the masked patches. Count is the number of observed days."""
    if isinstance(choice, np.ndarray):
        choice = torch.from_numpy(choice)
    choice = choice.to(device=batch["history"].device, dtype=torch.bool)
    tokens, valid = encoder.forward_tokens(batch["history"], batch["mask"], patch_mask=choice)
    sel = choice & valid
    if not bool(sel.any()):
        return tokens.new_zeros(()), 0
    patches, _ = encoder.patchify(batch["history"], batch["mask"])
    p = int(getattr(encoder, "patch_len", 5))
    c = int(getattr(encoder, "n_channels", batch["history"].shape[-1]))
    vals = patches[..., : p * c].reshape(patches.shape[0], patches.shape[1], p, c)
    bits = patches[..., p * c :].reshape(patches.shape[0], patches.shape[1], p).bool()
    pred = decoder(tokens[sel]).reshape(-1, p, c)
    tgt = vals[sel]
    obs = bits[sel].unsqueeze(-1)
    err = (pred - tgt) ** 2 * obs.to(pred.dtype)
    denom = obs.to(pred.dtype).sum().clamp(min=1.0)
    per_channel = err.sum(dim=(0, 1)) / denom
    return per_channel.mean(), int(obs.sum().item())
