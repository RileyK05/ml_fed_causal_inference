"""Neural CDE history encoders for Q3, plus the shared path construction.

Two encoders, both implementing ``forward(x, mask) -> (B, 64)``:

* ``NeuralCDEEncoder``: the full-length CDE of the brief. The hidden state evolves along
  the data path with ``dh = f_theta(h) dX``, solved by ``torchcde.cdeint`` (rk4, step 1).
* ``CDELogSigEncoder``: the log-signature / "log-ODE" fallback. Depth-2 log-signatures
  (increments plus Levy areas) over windows of ~10 days give the solver ~25 big steps
  instead of 252, which is the only way the neural arm fits the CPU budget.

``iisignature``/``signatory`` do not build on Windows, so the log-signatures are computed
here. ``torchcde``'s own NaN interpolation is a Python loop and too slow for training, so
NaNs are filled with an equivalent vectorised scheme (leading -> first observation,
interior -> linear, trailing -> last observation) before the coefficients are built. The
result is identical to what ``torchcde`` produces; only the speed differs.
"""
from __future__ import annotations

import numpy as np
import torch
import torchcde
from torch import Tensor, nn

from fedcore.q3.contracts import BETA63, DLOGVOL, RET, RET_REL, RVOL21, HistoryEncoder

CUMSUM_CHANNELS = (RET, DLOGVOL, RET_REL)
LEVEL_CHANNELS = (RVOL21, BETA63)


def build_path(
    x: Tensor,
    mask: Tensor,
    *,
    window: int = 252,
    add_time: bool = True,
    add_obs_count: bool = True,
) -> Tensor:
    """Scaled history to a path tensor ``(B, T, C')`` with NaN on unobserved days.

    ``x`` is ``(B, L, C)`` already scaled, with unobserved entries set to 0; ``mask`` is
    ``(B, L)`` bool. The mask puts the NaN back. Channels are cumulated where that makes
    sense for a path: cumulative sum for ``ret``, ``dlogvol`` and ``ret_rel``; levels for
    ``rvol21`` and ``beta63``. A time channel in ``[0, 1]`` stays flat during leading
    padding (it measures elapsed time since the first observed day) and a cumulative
    observation-count channel records the observational intensity.

    The returned path depends on ``x`` only where ``mask`` is true, so it is mask
    invariant by construction.
    """
    x = x[:, -window:, :]
    mask = mask[:, -window:]
    b, t, c = x.shape
    x = torch.where(mask.unsqueeze(-1), x, torch.full_like(x, float("nan")))
    zero = torch.nan_to_num(x, nan=0.0)

    channels: list[Tensor] = []
    for ch in range(c):
        if ch in CUMSUM_CHANNELS:
            cum = torch.cumsum(zero[:, :, ch], dim=1)
            channels.append(torch.where(mask, cum, torch.full_like(cum, float("nan"))).unsqueeze(-1))
        else:
            channels.append(x[:, :, ch].unsqueeze(-1))

    any_obs = mask.any(dim=1)
    if add_time:
        first_obs = mask.to(x.dtype).argmax(dim=1)
        idx = torch.arange(t, device=x.device, dtype=x.dtype).unsqueeze(0)
        time = ((idx - first_obs.unsqueeze(1)) / max(t - 1, 1)).clamp(min=0.0)
        time = torch.where(any_obs.unsqueeze(1), time, torch.zeros_like(time))
        channels.append(time.unsqueeze(-1).expand(b, t, 1))
    if add_obs_count:
        obs = mask.to(x.dtype).cumsum(dim=1) / max(t, 1)
        channels.append(obs.unsqueeze(-1))

    return torch.cat(channels, dim=-1)


def _fill_nan(path: Tensor, mask: Tensor) -> Tensor:
    """Vectorised NaN fill: leading -> first obs, interior -> linear, trailing -> last obs.

    Equivalent to ``torchcde``'s own missing-value handling but vectorised. ``path`` is
    ``(B, T, C')``; ``mask`` is the ``(B, T)`` observed mask. Only NaN entries are filled:
    the time and obs-count channels are finite on unobserved days and pass through. A row
    with no observation at all becomes zeros.
    """
    b, t, c = path.shape
    idx = torch.arange(t, device=path.device, dtype=path.dtype).expand(b, t)

    last_idx = torch.where(mask, idx, torch.full_like(idx, -1.0)).cummax(dim=1).values
    next_idx = (
        torch.where(mask, idx, torch.full_like(idx, float(t)))
        .flip(1)
        .cummin(dim=1)
        .values.flip(1)
    )
    has_prev = (last_idx >= 0).unsqueeze(-1)
    has_next = (next_idx <= t - 1).unsqueeze(-1)
    li = last_idx.clamp(min=0).long()
    ni = next_idx.clamp(max=t - 1).long()

    prev_val = torch.gather(path, 1, li.unsqueeze(-1).expand(-1, -1, c))
    next_val = torch.gather(path, 1, ni.unsqueeze(-1).expand(-1, -1, c))

    denom = (ni - li).clamp(min=1).to(path.dtype)
    ratio = ((idx - li.to(path.dtype)) / denom).unsqueeze(-1)
    interp = prev_val + ratio * (next_val - prev_val)

    zeros = torch.zeros_like(path)
    filled = torch.where(
        has_prev & has_next, interp, torch.where(has_prev, prev_val, torch.where(has_next, next_val, zeros))
    )
    return torch.where(torch.isfinite(path), path, filled)


class _VectorField(nn.Module):
    """``f_theta(t, z)`` for torchcde: an MLP reshaped to ``(B, hidden, C')``."""

    def __init__(self, hidden: int, c_prime: int, width: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, hidden * c_prime),
            nn.Tanh(),
        )
        self.c_prime = c_prime

    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        return self.mlp(z).reshape(z.shape[0], -1, self.c_prime)


class NeuralCDEEncoder(HistoryEncoder):
    """Full-length Neural CDE: ``dh = f_theta(h) dX`` over the day grid."""

    def __init__(
        self,
        n_channels: int = 5,
        hidden: int = 32,
        vf_width: int = 128,
        out_dim: int = 64,
        window: int = 252,
        add_time: bool = True,
        add_obs_count: bool = True,
        interpolation: str = "cubic",
    ):
        super().__init__()
        if interpolation not in ("cubic", "linear"):
            raise ValueError("interpolation must be 'cubic' or 'linear'")
        self.out_dim = int(out_dim)
        self.hidden = int(hidden)
        self.window = int(window)
        self.interpolation = interpolation
        self.add_time = bool(add_time)
        self.add_obs_count = bool(add_obs_count)
        self.c_prime = int(n_channels) + int(add_time) + int(add_obs_count)
        self.initial = nn.Linear(self.c_prime, hidden)
        self.vector_field = _VectorField(hidden, self.c_prime, vf_width)
        self.readout = nn.Sequential(nn.Linear(hidden, out_dim), nn.LayerNorm(out_dim))

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        path = build_path(
            x, mask, window=self.window, add_time=self.add_time, add_obs_count=self.add_obs_count
        )
        b, t, c_prime = path.shape
        filled = _fill_nan(path, mask[:, -self.window :])
        grid = torch.arange(t, device=x.device, dtype=x.dtype)
        if self.interpolation == "cubic":
            coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(filled, grid)
            control = torchcde.CubicSpline(coeffs, grid)
        else:
            coeffs = torchcde.linear_interpolation_coeffs(filled, grid)
            control = torchcde.LinearInterpolation(coeffs, grid)
        h0 = self.initial(self._initial_value(path, mask[:, -self.window :]))
        z = torchcde.cdeint(
            control,
            self.vector_field,
            h0,
            grid,
            method="rk4",
            options={"step_size": 1.0},
            adjoint=False,
        )
        return self.readout(z[:, -1, :])

    @staticmethod
    def _initial_value(path: Tensor, mask: Tensor) -> Tensor:
        """``X(t_0)`` per row: the first observed value, or 0 for an all-masked row."""
        b, t, c = path.shape
        first_obs = mask.to(path.dtype).argmax(dim=1)
        x0 = path[torch.arange(b, device=path.device), first_obs]
        return torch.where(torch.isfinite(x0), x0, torch.zeros_like(x0))


def _window_logsig(path: Tensor, mask: Tensor, window_size: int, time_ch=None, obs_ch=None) -> Tensor:
    """Depth-2 log-signatures over windows: increments plus Levy areas, plus time/obs.

    ``path`` is ``(B, T, C')`` with NaN on unobserved days. Returns ``(B, W, C' + C'(C'-1)/2 + k)``
    where ``k`` is the number of optional channels among ``time_ch``/``obs_ch``: per window
    the increment, the upper-triangle Levy area, and the time and observation-count
    increments. Increments touching a NaN are zero (hold flat).
    """
    b, t, c = path.shape
    n_win = t // window_size
    use = n_win * window_size
    m = mask[:, :use]
    seg = path[:, :use, :]

    zero = torch.nan_to_num(seg, nan=0.0)
    d = zero[:, 1:, :] - zero[:, :-1, :]
    d = torch.cat((torch.zeros(b, 1, c, dtype=path.dtype, device=path.device), d), dim=1)
    d = torch.where(m.unsqueeze(-1), d, torch.zeros_like(d))
    d = d.reshape(b, n_win, window_size, c)

    inc = d.sum(dim=2)  # (B, W, C)

    a_prev = d.cumsum(dim=2) - d  # exclusive cumsum within the window
    s2 = torch.einsum("bwsi,bwsj->bwij", a_prev, d)  # (B, W, C, C)
    area = 0.5 * (s2 - s2.transpose(-1, -2))
    tri = torch.triu_indices(c, c, offset=1, device=path.device)
    area_vec = area[:, :, tri[0], tri[1]]  # (B, W, C(C-1)/2)

    parts = [inc, area_vec]
    for ch in (time_ch, obs_ch):
        if ch is None:
            continue
        ends = path[:, window_size - 1 : use : window_size, ch]
        starts = path[:, 0 : use - window_size + 1 : window_size, ch]
        parts.append((ends - starts).unsqueeze(-1))
    return torch.cat(parts, dim=-1)


class CDELogSigEncoder(HistoryEncoder):
    """Neural CDE driven by depth-2 log-signatures over ~10-day windows.

    The control path is the running log-signature (cumulative sum of the per-window
    log-signatures), so the solver takes ~T/10 steps instead of T. Same vector field,
    initial state and readout as ``NeuralCDEEncoder``.
    """

    def __init__(
        self,
        n_channels: int = 5,
        hidden: int = 32,
        vf_width: int = 128,
        out_dim: int = 64,
        window: int = 252,
        window_size: int = 10,
        add_time: bool = True,
        add_obs_count: bool = True,
        interpolation: str = "cubic",
    ):
        super().__init__()
        if interpolation not in ("cubic", "linear"):
            raise ValueError("interpolation must be 'cubic' or 'linear'")
        self.out_dim = int(out_dim)
        self.hidden = int(hidden)
        self.window = int(window)
        self.window_size = int(window_size)
        self.interpolation = interpolation
        self.add_time = bool(add_time)
        self.add_obs_count = bool(add_obs_count)
        fine_c = int(n_channels) + int(add_time) + int(add_obs_count)
        c_prime = fine_c + fine_c * (fine_c - 1) // 2 + int(add_time) + int(add_obs_count)
        self.c_prime = c_prime
        self.initial = nn.Linear(c_prime, hidden)
        self.vector_field = _VectorField(hidden, c_prime, vf_width)
        self.readout = nn.Sequential(nn.Linear(hidden, out_dim), nn.LayerNorm(out_dim))

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        path = build_path(
            x, mask, window=self.window, add_time=self.add_time, add_obs_count=self.add_obs_count
        )
        c = path.shape[-1]
        time_ch = c - 2 if (self.add_time and self.add_obs_count) else (c - 1 if self.add_time else None)
        obs_ch = c - 1 if self.add_obs_count else None
        logsig = _window_logsig(path, mask[:, -self.window :], self.window_size, time_ch, obs_ch)
        coarse = torch.cumsum(logsig, dim=1)  # (B, W, C') running log-signature
        b, w, c_prime = coarse.shape
        grid = torch.arange(w, device=x.device, dtype=x.dtype)
        if self.interpolation == "cubic":
            coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(coarse, grid)
            control = torchcde.CubicSpline(coeffs, grid)
        else:
            coeffs = torchcde.linear_interpolation_coeffs(coarse, grid)
            control = torchcde.LinearInterpolation(coeffs, grid)
        h0 = self.initial(coarse[:, 0, :])
        z = torchcde.cdeint(
            control,
            self.vector_field,
            h0,
            grid,
            method="rk4",
            options={"step_size": 1.0},
            adjoint=False,
        )
        return self.readout(z[:, -1, :])
