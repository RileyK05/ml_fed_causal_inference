"""Fundamentals tower: EmbeddingGemma 2's text encoder reads a firm's filings into a 64-d vector.

docs/encoder/q3-v5-plan.md section B3. The market tower (return history) gives 64 dims, this tower
gives 64, and the two are joined into the a + b·s heads.

- Backbone: ``google/embeddinggemma-2`` (Apache 2.0), text path only, pinned to ``REVISION``. It is
  a bidirectional encoder: 24 layers, hidden 512, sliding-window (512) local layers with a global
  layer every sixth, then a projection to 768 per token. Run in float32 or bfloat16, never float16
  (the model card warns of NaNs).
- Fine-tuning: the lower ``n_layers - train_top`` layers are frozen, so their output can be computed
  once per filing and cached (``lower``); only the top ``train_top`` layers, the pooling and the
  64-d projection train (``upper``). ``train_top = 0`` is the frozen control arm.
- Pooling: ``"attention"`` learns which tokens matter (a query scores every token; softmax over the
  filing); ``"mean"`` is the model's own embedding pooling, the "use the embedding" control.
- Text: the filing's market-risk section, with the model card's document prompt. Sections longer
  than ``max_tokens`` keep their most rate-related sentences, in original order.
- Timing: a row uses the latest 10-K or 10-Q accepted on a day strictly before the meeting and at
  most ``STALE_DAYS`` old. Filings accepted on the meeting day are never inputs.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import torch
from torch import nn

MODEL_ID = "google/embeddinggemma-2"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
OUT_DIM = 64
MAX_TOKENS = 4096
STALE_DAYS = 400
RATE_TERMS = re.compile(
    r"interest|rate|floating|variable|fixed|swap|hedg|basis point|libor|sofr|prime|debt|borrow|credit|"
    r"refinanc|matur|liquidity|covenant|yield|duration",
    re.I,
)
_SENTENCE = re.compile(r"(?<=[.;:])\s+(?=[A-Z(])")


def load_backbone(dtype: torch.dtype = torch.float32, device: str | torch.device = "cpu"):
    """(tokenizer, text model) for EmbeddingGemma 2 without the vision and audio encoders."""
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    if dtype == torch.float16:
        raise ValueError("EmbeddingGemma 2 must not run in float16; use bfloat16 or float32")
    config = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION)
    config.vision_config = None
    config.audio_config = None
    model = AutoModel.from_pretrained(MODEL_ID, revision=REVISION, config=config, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    return tokenizer, model.language_model.to(device)


def document_text(form: str, section: str) -> str:
    """The model card's document prompt."""
    return f"title: {form} market risk | text: {section}"


def select_text(section: str, n_tokens, max_tokens: int = MAX_TOKENS) -> str:
    """The whole section if it fits, else its most rate-related sentences in their original order.

    ``n_tokens(text) -> int`` counts tokens (the tokenizer's length)."""
    if n_tokens(section) <= max_tokens:
        return section
    sentences = _SENTENCE.split(section)
    lengths = np.array([n_tokens(s) for s in sentences])
    score = np.array([len(RATE_TERMS.findall(s)) / max(n, 1) for s, n in zip(sentences, lengths)])
    keep, used = [], 0
    for k in np.argsort(-score, kind="stable"):
        if used + lengths[k] > max_tokens:
            continue
        keep.append(k)
        used += lengths[k]
    return " ".join(sentences[k] for k in sorted(keep))


class FilingTower(nn.Module):
    """Filing tokens -> 64-d vector. ``lower`` (frozen layers) can be cached; ``upper`` trains."""

    def __init__(self, backbone: nn.Module, train_top: int = 4, pooling: str = "attention",
                 out_dim: int = OUT_DIM, dropout: float = 0.1):
        super().__init__()
        if pooling not in ("attention", "mean"):
            raise ValueError(f"pooling must be 'attention' or 'mean', got {pooling!r}")
        self.backbone = backbone
        n_layers = len(backbone.layers)
        if not 0 <= train_top <= n_layers:
            raise ValueError(f"train_top must be in [0, {n_layers}]")
        self.split = n_layers - train_top
        for p in backbone.parameters():
            p.requires_grad_(False)
        if train_top:
            backbone.gradient_checkpointing_enable()  # recompute top-layer activations in backward
            for layer in backbone.layers[self.split:]:
                for p in layer.parameters():
                    p.requires_grad_(True)
            for p in (*backbone.norm.parameters(), *backbone.embedding_projection.parameters()):
                p.requires_grad_(True)
        width = backbone.config.embedding_dim
        self.pooling = pooling
        self.score = nn.Linear(width, 1) if pooling == "attention" else None
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Dropout(dropout), nn.Linear(width, out_dim))
        self.out_dim = out_dim

    def _setup(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Token embeddings, per-layer inputs (frozen, so no graph), masks and rotary tables."""
        from transformers.masking_utils import create_bidirectional_mask, create_bidirectional_sliding_window_mask

        bb = self.backbone
        with torch.no_grad():
            embeds = bb.embed_tokens(input_ids)
            per_layer = bb.ple(embeds)
        kw = {"config": bb.config, "inputs_embeds": embeds, "attention_mask": attention_mask}
        masks = {"full_attention": create_bidirectional_mask(**kw),
                 "sliding_attention": create_bidirectional_sliding_window_mask(**kw)}
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        rope = {t: bb.rotary_emb(embeds, position_ids, t) for t in bb.unique_layer_types}
        return embeds, per_layer, masks, rope

    def _layers(self, hidden, per_layer, masks, rope, lo: int, hi: int, offset: int = 0):
        """Layers lo..hi-1; ``per_layer`` holds layer inputs from index ``offset`` on."""
        types = self.backbone.config.layer_types
        for i in range(lo, hi):
            hidden = self.backbone.layers[i](hidden, per_layer[:, :, i - offset, :], attention_mask=masks[types[i]],
                                             position_embeddings=rope[types[i]])
        return hidden

    def _top(self, hidden, per_layer_top, masks, rope) -> torch.Tensor:
        hidden = self._layers(hidden, per_layer_top, masks, rope, self.split, len(self.backbone.layers), self.split)
        return self.backbone.embedding_projection(self.backbone.norm(hidden))

    @torch.no_grad()
    def lower(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Hidden states after the frozen layers: the cacheable part."""
        embeds, per_layer, masks, rope = self._setup(input_ids, attention_mask)
        return self._layers(embeds, per_layer, masks, rope, 0, self.split)

    def upper(self, hidden: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """768-d token states from cached lower hidden states (trainable layers, norm, projection)."""
        _, per_layer, masks, rope = self._setup(input_ids, attention_mask)
        return self._top(hidden, per_layer[:, :, self.split:, :].contiguous(), masks, rope)

    def pool(self, states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.to(states.dtype)
        if self.pooling == "mean":
            return (states * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1.0)
        logits = self.score(states).squeeze(-1).masked_fill(attention_mask == 0, torch.finfo(states.dtype).min)
        return (torch.softmax(logits, dim=1)[..., None] * states).sum(1)

    def token_weights(self, states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Per-token pooling weights (for attribution: which sentences drove the vector)."""
        if self.pooling == "mean":
            m = attention_mask.to(states.dtype)
            return m / m.sum(1, keepdim=True).clamp(min=1.0)
        logits = self.score(states).squeeze(-1).masked_fill(attention_mask == 0, torch.finfo(states.dtype).min)
        return torch.softmax(logits, dim=1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                cached: torch.Tensor | None = None) -> torch.Tensor:
        if cached is not None:
            states = self.upper(cached, input_ids, attention_mask)
        else:
            # one setup for both halves; keep only the trainable layers' slice of the (large)
            # per-layer inputs while the frozen layers run
            embeds, per_layer, masks, rope = self._setup(input_ids, attention_mask)
            top = per_layer[:, :, self.split:, :].contiguous()
            with torch.no_grad():
                hidden = self._layers(embeds, per_layer, masks, rope, 0, self.split)
            del per_layer
            states = self._top(hidden, top, masks, rope)
        return self.out(self.pool(states, attention_mask))


# ----------------------------------------------------------------------------- rows -> filings

def filing_for_rows(rows: pd.DataFrame, gvkey_cik: pd.DataFrame, sections: pd.DataFrame) -> pd.Series:
    """Accession of the latest usable 10-K/10-Q market-risk section per panel row (NaN if none).

    ``rows``: meeting, gvkey. ``gvkey_cik``: gvkey, cik. ``sections``: accession, cik, acceptance,
    method, section (optionally no_change). A filing is usable when accepted on a day strictly
    before the meeting, found (method != 'missing'/'none'), not a bare "no material change"
    10-Q (the row then reads the latest substantive filing, usually the 10-K), and at most
    ``STALE_DAYS`` old at the meeting."""
    keep = ~sections["method"].isin(["missing", "none"]) & (sections["section"].str.len() > 0)
    if "no_change" in sections:  # "no material change since the 10-K": read the filing it points back to
        keep &= ~sections["no_change"].fillna(False).astype(bool)
    sec = sections[keep].copy()
    sec["usable"] = pd.to_datetime(sec["acceptance"]).dt.normalize() + pd.Timedelta(days=1)
    sec = sec.dropna(subset=["usable"]).sort_values("usable")
    sec["cik"] = sec["cik"].astype(str)  # one plain key type on both sides of the as-of join
    link = gvkey_cik.dropna().astype(str).drop_duplicates("gvkey").set_index("gvkey")["cik"]
    r = rows[["meeting", "gvkey"]].reset_index()
    r["cik"] = r["gvkey"].astype(object).map(link)
    r = r.dropna(subset=["cik"]).astype({"cik": str}).sort_values("meeting")
    merged = pd.merge_asof(r, sec[["cik", "usable", "accession"]], left_on="meeting", right_on="usable", by="cik",
                           tolerance=pd.Timedelta(days=STALE_DAYS))
    return merged.set_index("index")["accession"].reindex(rows.index)


# ----------------------------------------------------------------------------- two towers

class TwoTowerModel(nn.Module):
    """[market 64 | fundamentals 64] = 128 -> a_i, b_i, so mu_i = a_i + b_i * s.

    ``market`` is the return-history encoder (``PatchTransformerEncoder``, out_dim 64). The
    fundamentals vector comes from ``FilingTower`` per filing and is gathered to rows by the caller,
    so a filing shared by several rows is read once. Rows with no usable filing get a learned
    "no filing" vector and a 1 in the missing flag, never zeros passed off as text. Heads start at
    zero, so the untrained model predicts exactly the base it corrects."""

    def __init__(self, market: nn.Module, text_dim: int = OUT_DIM, b_only: bool = True, z_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.market = market
        self.b_only = bool(b_only)
        self.no_filing = nn.Parameter(torch.zeros(text_dim))
        self.fusion = nn.Sequential(
            nn.Linear(int(market.out_dim) + text_dim + 1, z_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(z_dim, z_dim), nn.ReLU(),
        )
        self.head_b = nn.Linear(z_dim, 1)
        self.head_a = None if self.b_only else nn.Linear(z_dim, 1)
        for head in (self.head_b, self.head_a):
            if head is not None:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

    def forward(self, history: torch.Tensor, history_mask: torch.Tensor, text: torch.Tensor,
                has_text: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = self.head_b.weight.dtype
        h = self.market(history.to(dtype), history_mask)
        flag = has_text.to(dtype)[:, None]
        t = torch.where(flag > 0, text.to(dtype), self.no_filing.to(dtype).expand_as(text))
        z = self.fusion(torch.cat((h, t, 1.0 - flag), dim=-1))
        b = self.head_b(z).squeeze(-1)
        a = torch.zeros_like(b) if self.head_a is None else self.head_a(z).squeeze(-1)
        return a, b


def gather_filings(tower: FilingTower, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                   row_doc: np.ndarray, cached: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """(text vectors per row, has_text per row). ``row_doc[i]`` indexes the batch's unique filings
    (``input_ids`` rows), -1 when row i has none; each filing goes through the tower once."""
    row_doc = np.asarray(row_doc)
    has = torch.as_tensor(row_doc >= 0, device=input_ids.device)
    if input_ids.shape[0] == 0:
        return torch.zeros(len(row_doc), tower.out_dim, device=input_ids.device), has
    docs = tower(input_ids, attention_mask, cached)
    idx = torch.as_tensor(np.maximum(row_doc, 0), device=docs.device)
    return docs[idx] * has[:, None].to(docs.dtype), has
