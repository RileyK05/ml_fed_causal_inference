"""State-dependent linear shock head: mu(z, s) = a(z) + b(z) * s.

Shock units stay as they arrive (synthetic shocks are in units of 10bp). The
encoder and the fusion never see s; only this head does.
"""
from __future__ import annotations

from torch import Tensor, nn


class ShockHead(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()
        self.a = nn.Linear(z_dim, 1)
        self.b = nn.Linear(z_dim, 1)

    def forward(self, z: Tensor, s: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if s.ndim == 2 and s.shape[-1] == 1:
            s = s.squeeze(-1)
        a = self.a(z).squeeze(-1)
        b = self.b(z).squeeze(-1)
        return a + b * s, a, b
