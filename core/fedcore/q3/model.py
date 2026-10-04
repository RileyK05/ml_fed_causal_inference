"""Response model: history encoder, fundamentals branch, context branch, fusion, shock head.

    h = encoder(history, mask)
    f = MLP(fundamentals ++ fundamentals_mask)
    c = MLP(context)
    z = MLP(concat(h, f, c))          # z_dim defaults to 64
    mu, a, b = ShockHead(z, s)

s is not an input to the encoder or the fusion.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from fedcore.q3.contracts import HistoryEncoder
from fedcore.q3.heads import ShockHead


class _MLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_out, d_out),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ResponseModel(nn.Module):
    def __init__(
        self,
        encoder: HistoryEncoder,
        n_fund: int,
        n_ctx: int,
        z_dim: int = 64,
        branch_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not hasattr(encoder, "out_dim"):
            raise ValueError("encoder must expose out_dim")
        self.encoder = encoder
        self.fund = _MLP(n_fund * 2, branch_dim, dropout)
        self.context_mlp = _MLP(n_ctx, branch_dim, dropout)
        fused = int(encoder.out_dim) + branch_dim + branch_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused, z_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(z_dim, z_dim),
            nn.ReLU(),
        )
        self.head = ShockHead(z_dim)

    def forward(
        self,
        history: Tensor,
        history_mask: Tensor,
        fundamentals: Tensor,
        fundamentals_mask: Tensor,
        context: Tensor,
        shock: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        dtype = self.head.a.weight.dtype
        history = history.to(dtype=dtype)
        fundamentals = fundamentals.to(dtype=dtype)
        context = context.to(dtype=dtype)
        shock = shock.to(dtype=dtype)
        h = self.encoder(history, history_mask)
        fund_in = torch.cat((fundamentals, fundamentals_mask.to(dtype=dtype)), dim=-1)
        f = self.fund(fund_in)
        c = self.context_mlp(context)
        z = self.fusion(torch.cat((h, f, c), dim=-1))
        return self.head(z, shock)
