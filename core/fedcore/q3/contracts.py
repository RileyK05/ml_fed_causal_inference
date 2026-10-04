"""Q3 panel contract and the reference history encoder.

One row is one (firm, meeting). History is unscaled: the trainer fits scaling on
training rows only. ``history[:, -1]`` is the last trading day before the meeting.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

CHANNELS = ("ret", "dlogvol", "rvol21", "ret_rel", "beta63")
RET, DLOGVOL, RVOL21, RET_REL, BETA63 = range(len(CHANNELS))
L = 252

_FLOAT32 = np.dtype(np.float32)
_DT = np.dtype("datetime64[ns]")


def _varies_within(meeting: np.ndarray, values: np.ndarray, tol: float = 1e-5) -> bool:
    """True when `values` is not constant inside some meeting. values is (N,) or (N, K)."""
    n = meeting.shape[0]
    if n <= 1:
        return False
    order = np.argsort(meeting, kind="mergesort")
    grouped = meeting[order]
    v = np.asarray(values, dtype=np.float64)[order]
    if v.ndim == 1:
        v = v[:, None]
    cuts = np.flatnonzero(grouped[1:] != grouped[:-1]) + 1
    starts = np.concatenate((np.array([0], dtype=int), cuts))
    hi = np.maximum.reduceat(v, starts, axis=0)
    lo = np.minimum.reduceat(v, starts, axis=0)
    return bool(np.any(np.abs(hi - lo) > tol))


@dataclass(frozen=True)
class Q3Panel:
    """Aligned firm-meeting arrays. ``validate`` is explicit so a broken panel can be built and caught."""

    meeting: np.ndarray
    firm_id: np.ndarray
    history: np.ndarray
    history_mask: np.ndarray
    fundamentals: np.ndarray
    fundamentals_mask: np.ndarray
    context: np.ndarray
    shock: np.ndarray
    target: np.ndarray
    a_true: np.ndarray | None = None
    b_true: np.ndarray | None = None

    def subset(self, rows: np.ndarray) -> "Q3Panel":
        """Return a copy of the selected rows. `rows` is an integer index or a boolean mask."""
        idx = np.asarray(rows)
        n = self.meeting.shape[0]
        if idx.dtype == bool:
            if idx.shape != (n,):
                raise ValueError(f"boolean row mask has shape {idx.shape}, expected {(n,)}")
        else:
            idx = idx.astype(np.int64, copy=False)

        def take(array):
            if array is None:
                return None
            return array[idx]

        return Q3Panel(
            meeting=take(self.meeting),
            firm_id=take(self.firm_id),
            history=take(self.history),
            history_mask=take(self.history_mask),
            fundamentals=take(self.fundamentals),
            fundamentals_mask=take(self.fundamentals_mask),
            context=take(self.context),
            shock=take(self.shock),
            target=take(self.target),
            a_true=take(self.a_true),
            b_true=take(self.b_true),
        )

    def meetings(self) -> pd.DatetimeIndex:
        """Sorted unique announcement dates."""
        return pd.DatetimeIndex(np.unique(np.asarray(self.meeting, dtype=_DT)))

    def validate(self) -> None:
        """Check shapes, dtypes, within-meeting shock and context, and mask/NaN agreement.

        On a masked step every channel is NaN. On an observed step every channel is finite.
        The same rule applies to fundamentals. Shock and context are constant within a meeting.
        """
        n = int(self.meeting.shape[0])
        arrays = {
            "meeting": self.meeting,
            "firm_id": self.firm_id,
            "history": self.history,
            "history_mask": self.history_mask,
            "fundamentals": self.fundamentals,
            "fundamentals_mask": self.fundamentals_mask,
            "context": self.context,
            "shock": self.shock,
            "target": self.target,
        }
        for name, array in arrays.items():
            if array.shape[0] != n:
                raise ValueError(f"shape mismatch: {name} has length {array.shape[0]}, expected {n}")
        if self.history.ndim != 3 or self.history.shape[1] != L or self.history.shape[2] != len(CHANNELS):
            raise ValueError(
                f"shape mismatch: history {self.history.shape}, expected (N, {L}, {len(CHANNELS)})"
            )
        if self.history_mask.shape != (n, L):
            raise ValueError(f"shape mismatch: history_mask {self.history_mask.shape}, expected {(n, L)}")
        if self.fundamentals.ndim != 2 or self.fundamentals.shape[1] < 1:
            raise ValueError(f"shape mismatch: fundamentals {self.fundamentals.shape}")
        f = self.fundamentals.shape[1]
        if self.fundamentals_mask.shape != (n, f):
            raise ValueError(f"shape mismatch: fundamentals_mask {self.fundamentals_mask.shape}")
        if self.context.ndim != 2 or self.context.shape[1] < 1:
            raise ValueError(f"shape mismatch: context {self.context.shape}")
        for name in ("shock", "target"):
            if arrays[name].ndim != 1:
                raise ValueError(f"shape mismatch: {name} {arrays[name].shape}")
        for name, array in (("a_true", self.a_true), ("b_true", self.b_true)):
            if array is not None and array.shape != (n,):
                raise ValueError(f"shape mismatch: {name} {array.shape}, expected {(n,)}")

        if self.meeting.dtype != _DT:
            raise ValueError(f"dtype mismatch: meeting is {self.meeting.dtype}, expected {_DT}")
        id_ok = (
            np.issubdtype(self.firm_id.dtype, np.integer)
            or np.issubdtype(self.firm_id.dtype, np.str_)
            or self.firm_id.dtype == object
        )
        if not id_ok:
            raise ValueError(f"dtype mismatch: firm_id is {self.firm_id.dtype}")
        for name in ("history", "fundamentals", "context", "shock", "target"):
            if arrays[name].dtype != _FLOAT32:
                raise ValueError(f"dtype mismatch: {name} is {arrays[name].dtype}, expected float32")
        if self.history_mask.dtype != bool or self.fundamentals_mask.dtype != bool:
            raise ValueError("dtype mismatch: masks must be bool")
        for name, array in (("a_true", self.a_true), ("b_true", self.b_true)):
            if array is not None and array.dtype != _FLOAT32:
                raise ValueError(f"dtype mismatch: {name} is {array.dtype}, expected float32")

        if np.isnat(self.meeting).any():
            raise ValueError("meeting contains NaT")
        for name in ("shock", "target", "context"):
            if not np.isfinite(arrays[name]).all():
                raise ValueError(f"{name} contains a non-finite value")
        for name, array in (("a_true", self.a_true), ("b_true", self.b_true)):
            if array is not None and not np.isfinite(array).all():
                raise ValueError(f"{name} contains a non-finite value")

        observed = np.isfinite(self.history).all(axis=-1)
        if not np.array_equal(observed, self.history_mask):
            raise ValueError("history mask and NaN do not agree")
        fund_observed = np.isfinite(self.fundamentals)
        if not np.array_equal(fund_observed, self.fundamentals_mask):
            raise ValueError("fundamentals mask and NaN do not agree")
        if _varies_within(self.meeting, self.shock):
            raise ValueError("shock varies within a meeting")
        if _varies_within(self.meeting, self.context):
            raise ValueError("context varies within a meeting")


class HistoryEncoder(nn.Module):
    """Map a scaled history window to a vector.

    ``forward(x, mask) -> h`` with ``x`` shaped (B, L, C). Entries where ``mask``
    is false are unobserved. ``h`` must not depend on those entries. ``out_dim`` is
    the width of ``h``.
    """

    out_dim: int

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        raise NotImplementedError


class SummaryEncoder(HistoryEncoder):
    """Masked mean and standard deviation of each channel over 21, 63 and 252 days, then a linear layer.

    This is the no-sequence control. Windows shorter than the tensor use the days that exist.
    A channel with fewer than two observed days contributes a standard deviation of 0.
    """

    windows = (21, 63, 252)

    def __init__(self, n_channels: int = len(CHANNELS), out_dim: int = 32):
        super().__init__()
        self.out_dim = int(out_dim)
        self.n_channels = int(n_channels)
        n_in = self.n_channels * 2 * len(self.windows)
        self.proj = nn.Linear(n_in, self.out_dim)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = torch.nan_to_num(x.to(dtype=self.proj.weight.dtype), nan=0.0, posinf=0.0, neginf=0.0)
        mask = mask.bool()
        parts: list[Tensor] = []
        length = x.shape[1]
        for window in self.windows:
            width = min(window, length)
            mean, std = _masked_mean_std(x[:, -width:, :], mask[:, -width:])
            parts.append(mean)
            parts.append(std)
        return self.proj(torch.cat(parts, dim=-1))


def _masked_mean_std(x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """Mean and population std over time. Values with mask False are ignored. x is (B, W, C)."""
    observed = mask.unsqueeze(-1)
    clean = torch.where(observed, x, torch.zeros_like(x))
    count = observed.to(dtype=x.dtype).sum(dim=1)  # (B, 1)
    safe = count.clamp(min=1.0)
    mean = clean.sum(dim=1) / safe
    centered = torch.where(observed, clean - mean.unsqueeze(1), torch.zeros_like(clean))
    var = (centered * centered).sum(dim=1) / safe
    std = torch.sqrt(var.clamp(min=0.0))
    present = count > 0
    enough = count >= 2
    mean = torch.where(present, mean, torch.zeros_like(mean))
    std = torch.where(enough, std, torch.zeros_like(std))
    return mean, std
