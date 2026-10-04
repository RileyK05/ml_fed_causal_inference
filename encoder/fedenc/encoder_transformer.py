"""Patch-transformer history encoder (docs/encoder/transformer.md section 2, v1).

A history window of at most 252 trading days is right-aligned and left-padded with
masked days to ``n_patches * patch_len`` days (255 for the v1 patch length of 5),
then cut into non-overlapping patches. Each patch flattens its ``patch_len x C``
values plus the ``patch_len`` day-mask bits; the bits are how the model tells a
real zero from an unobserved day. Unobserved values are zeroed inside the encoder,
so ``forward`` cannot depend on them. Fully padded patches are excluded from
attention and from pooling.

Padding invariance falls out of the right-alignment: adding extra left padding to
a short history only extends the invalid region in front of the same patches.
"""
from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from fedcore.q3.contracts import CHANNELS, L, HistoryEncoder

POOLS = ("mean", "last", "cls")


class PatchTransformerEncoder(HistoryEncoder):
    """Non-overlapping patch tokens, pre-norm transformer blocks, masked pooling.

    Constructor knobs used by the ablations: ``patch_len``, ``d_model``,
    ``n_layers``, ``n_heads`` and ``pool`` (``"mean"``, ``"last"``, ``"cls"``).
    ``forward_tokens`` exposes the per-patch outputs before pooling for
    pre-training and diagnostics.
    """

    def __init__(
        self,
        n_channels: int = len(CHANNELS),
        patch_len: int = 5,
        d_model: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        pool: str = "mean",
    ):
        super().__init__()
        if patch_len < 1:
            raise ValueError("patch_len must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if pool not in POOLS:
            raise ValueError(f"pool must be one of {POOLS}, got {pool!r}")
        self.n_channels = int(n_channels)
        self.patch_len = int(patch_len)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.pool = pool
        self.out_dim = self.d_model
        self.n_patches = -(-L // self.patch_len)  # ceil(252 / patch_len): 51 for patch_len 5
        self.total_days = self.n_patches * self.patch_len
        patch_in = self.patch_len * (self.n_channels + 1)

        self.proj = nn.Linear(patch_in, self.d_model)
        self.pos = nn.Parameter(torch.zeros(self.n_patches, self.d_model))
        self.mask_token = nn.Parameter(torch.zeros(self.d_model))
        self.cls_token = nn.Parameter(torch.zeros(self.d_model)) if pool == "cls" else None
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=self.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(self.d_model)
        self._frozen: list[nn.Module] = []
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=0.02)

    # ------------------------------------------------------------------ patching

    def patchify(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Right-aligned non-overlapping patches.

        ``x`` is (B, T, C) with T <= total_days and ``mask`` is (B, T) bool, true
        where the day is observed. Returns the flat patch input
        (B, n_patches, patch_len*(C+1)) -- values then day-mask bits, values
        zeroed where the day is masked -- and ``valid`` (B, n_patches), true when
        at least one day of the patch is observed.
        """
        if x.ndim != 3 or mask.ndim != 2:
            raise ValueError(f"expected x (B, T, C) and mask (B, T), got {tuple(x.shape)} {tuple(mask.shape)}")
        if x.shape[:2] != mask.shape:
            raise ValueError(f"mask shape {tuple(mask.shape)} does not match x {tuple(x.shape)}")
        if x.shape[2] != self.n_channels:
            raise ValueError(f"x has {x.shape[2]} channels, expected {self.n_channels}")
        b, t, c = x.shape
        if t > self.total_days:
            raise ValueError(f"history of {t} days exceeds the {self.total_days} patch slots")
        x = torch.nan_to_num(x.to(dtype=self.proj.weight.dtype), nan=0.0, posinf=0.0, neginf=0.0)
        day_mask = mask.bool()
        pad = self.total_days - t
        if pad:
            x = F.pad(x, (0, 0, pad, 0))
            day_mask = F.pad(day_mask, (pad, 0), value=False)
        x = torch.where(day_mask.unsqueeze(-1), x, torch.zeros_like(x))
        bits = day_mask.to(dtype=x.dtype)
        p = self.patch_len
        vals = x.reshape(b, self.n_patches, p * c)
        bits_flat = bits.reshape(b, self.n_patches, p)
        patches = torch.cat((vals, bits_flat), dim=-1)
        valid = bits_flat.bool().any(dim=-1)
        return patches, valid

    # ------------------------------------------------------------------ encoding

    def _encode(self, x: Tensor, mask: Tensor, patch_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Full token sequence (with the cls token first when pooling is cls) and patch validity."""
        patches, valid = self.patchify(x, mask)
        tokens = self.proj(patches) + self.pos.unsqueeze(0).to(dtype=self.proj.weight.dtype)
        if patch_mask is not None:
            pm = patch_mask.bool() & valid
            tokens = torch.where(pm.unsqueeze(-1), self.mask_token.to(dtype=tokens.dtype), tokens)
        key_pad = ~valid
        if self.cls_token is not None:
            cls = self.cls_token.to(dtype=tokens.dtype).expand(tokens.shape[0], 1, -1)
            tokens = torch.cat((cls, tokens), dim=1)
            key_pad = F.pad(key_pad, (1, 0), value=False)
        # A row with no valid patch would make softmax over an all-masked row NaN.
        # Let it attend to its dummy first slot instead; pooling then returns zeros.
        empty = ~valid.any(dim=1)
        if bool(empty.any()):
            key_pad = key_pad.clone()
            key_pad[empty, 0] = False
        out = self.blocks(tokens, src_key_padding_mask=key_pad)
        return self.norm(out), valid

    def forward_tokens(self, x: Tensor, mask: Tensor, patch_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Per-patch outputs before pooling: (B, n_patches, d_model) and (B, n_patches) validity.

        ``patch_mask`` (B, n_patches) replaces the embedded tokens of the selected
        valid patches with the learned [MASK] token before the transformer, which
        is how pre-training hides patch content.
        """
        out, valid = self._encode(x, mask, patch_mask)
        if self.cls_token is not None:
            out = out[:, 1:]
        return out, valid

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        out, valid = self._encode(x, mask)
        empty = ~valid.any(dim=1)
        if self.pool == "cls":
            h = out[:, 0]
        elif self.pool == "last":
            pos = torch.arange(out.shape[1] - (1 if self.cls_token is not None else 0), device=out.device)
            tokens = out[:, 1:] if self.cls_token is not None else out
            idx = torch.where(valid, pos.unsqueeze(0), torch.full_like(pos.unsqueeze(0), -1)).max(dim=1).values
            h = tokens.gather(1, idx.clamp(min=0).view(-1, 1, 1).expand(-1, 1, tokens.shape[-1])).squeeze(1)
        else:
            tokens = out[:, 1:] if self.cls_token is not None else out
            w = valid.to(dtype=tokens.dtype).unsqueeze(-1)
            h = (tokens * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
        return torch.where(empty.unsqueeze(-1), torch.zeros_like(h), h)

    # ------------------------------------------------------------------ freezing

    def freeze(self, modules: Iterable[nn.Module | Tensor]) -> None:
        """Keep ``modules`` (modules or parameters) fixed and in eval mode."""
        self._frozen = list(modules)
        for module in self._frozen:
            if isinstance(module, nn.Parameter):
                module.requires_grad_(False)
                continue
            for param in module.parameters():
                param.requires_grad_(False)
        self.mask_token.requires_grad_(False)
        self._apply_frozen_eval()
        if self.cls_token is not None:
            self.cls_token.requires_grad_(False)

    def _apply_frozen_eval(self) -> None:
        for module in self._frozen:
            if isinstance(module, nn.Module):
                module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self._apply_frozen_eval()
        return self

    # ------------------------------------------------------------------ diagnostics

    def attention(self, x: Tensor, mask: Tensor, layer: int = -1) -> Tensor:
        """Attention probabilities (B, heads, S, S) at one block. A diagnostic, not an explanation."""
        patches, valid = self.patchify(x, mask)
        tokens = self.proj(patches) + self.pos.unsqueeze(0).to(dtype=self.proj.weight.dtype)
        key_pad = ~valid
        if self.cls_token is not None:
            cls = self.cls_token.to(dtype=tokens.dtype).expand(tokens.shape[0], 1, -1)
            tokens = torch.cat((cls, tokens), dim=1)
            key_pad = F.pad(key_pad, (1, 0), value=False)
        empty = ~valid.any(dim=1)
        if bool(empty.any()):
            key_pad = key_pad.clone()
            key_pad[empty, 0] = False
        layers = self.blocks.layers
        idx = layer if layer >= 0 else len(layers) + layer
        if not 0 <= idx < len(layers):
            raise ValueError(f"layer {layer} out of range for {len(layers)} blocks")
        h = tokens
        for i, block in enumerate(layers):
            if i == idx:
                return _attention_weights(block, h, key_pad)
            h = block(h, src_key_padding_mask=key_pad)
        raise ValueError(f"layer {layer} out of range for {len(layers)} blocks")


def _attention_weights(block: nn.TransformerEncoderLayer, h: Tensor, key_pad: Tensor) -> Tensor:
    x = block.norm1(h)
    qkv = F.linear(x, block.self_attn.in_proj_weight, block.self_attn.in_proj_bias)
    q, k, _ = qkv.chunk(3, dim=-1)
    b, s, _ = q.shape
    heads = block.self_attn.num_heads
    dim = q.shape[-1] // heads
    q = q.view(b, s, heads, dim).transpose(1, 2)
    k = k.view(b, s, heads, dim).transpose(1, 2)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(dim)
    scores = scores.masked_fill(key_pad[:, None, None, :], float("-inf"))
    return scores.softmax(dim=-1)
