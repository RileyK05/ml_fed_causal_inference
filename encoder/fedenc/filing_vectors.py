"""Cached frozen EmbeddingGemma 2 vectors, one per filing, for the v6 text arms (q3-v6-plan.md section 3).

The v5 test found that frozen and fine-tuned Gemma behave alike, so v6 reads each filing's
market-risk section once with the frozen model and keeps the 768-d mean-pooled vector (the
model's own embedding pooling). Training then costs nothing extra for text, which lets every
arm see all ~500 firms per event and the full v4 inputs.

- Text: as v5 (``document_text`` prompt, rate sentences first), up to ``MAX_TOKENS`` tokens.
- Rows -> filing: ``filing_for_rows`` (latest usable 10-K/10-Q accepted strictly before the
  day), for the Fed panel and the shock-day panel alike, keyed by (date, permno). Shock days
  never fall on a Fed day, so one map holds both.
- Change: each filing's prior-year filing is the same firm's same-form filing accepted
  ``PRIOR_DAYS`` before it, nearest to 365 days. ``change=True`` gives vector minus prior
  vector, missing when either is missing.

Cached in gitignored ``core/data/processed/q3_filing_vectors/``: the row maps carry CRSP and
Compustat keys.

    python -m fedenc.filing_vectors ids      # local: the gvkey -> CIK table the boxes need
    python -m fedenc.filing_vectors build    # GPU box: embed every filing a row uses
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from fedcore.config import PROCESSED
from fedcore.q3.contracts import Q3Panel
from fedenc.text_tower import document_text, filing_for_rows, select_text

FOLDER = PROCESSED / "q3_filing_vectors"
MAX_TOKENS = 2048
PRIOR_DAYS = (300, 430)


def prior_year(sections: pd.DataFrame) -> pd.Series:
    """accession -> the same CIK and form's filing accepted ``PRIOR_DAYS`` earlier (nearest 365 days)."""
    s = sections[["accession", "cik", "form", "acceptance"]].dropna().copy()
    s["cik"], s["day"] = s["cik"].astype(str), pd.to_datetime(s["acceptance"]).dt.normalize()
    m = s.merge(s, on=["cik", "form"], suffixes=("", "_prior"))
    gap = (m["day"] - m["day_prior"]).dt.days
    m = m[(gap >= PRIOR_DAYS[0]) & (gap <= PRIOR_DAYS[1])].assign(off=(gap - 365).abs())
    m = m.sort_values(["accession", "off"]).drop_duplicates("accession")
    return m.set_index("accession")["accession_prior"]


def usable(sections: pd.DataFrame) -> pd.DataFrame:
    """Sections ``filing_for_rows`` may pick: found, non-empty, not a bare no-change 10-Q."""
    keep = ~sections["method"].isin(["missing", "none"]) & (sections["section"].str.len() > 0)
    if "no_change" in sections:
        keep &= ~sections["no_change"].fillna(False).astype(bool)
    return sections[keep]


@torch.no_grad()
def embed(texts: list[str], tokenizer, backbone, device, batch: int = 16, max_tokens: int = MAX_TOKENS) -> np.ndarray:
    """Mean-pooled last hidden states (768-d) in float16, longest texts batched together."""
    enc = [tokenizer(t, truncation=True, max_length=max_tokens)["input_ids"] for t in texts]
    order = np.argsort([-len(e) for e in enc], kind="stable")
    out = np.zeros((len(texts), backbone.config.embedding_dim), dtype=np.float16)
    cuda = torch.device(device).type == "cuda"
    pad = tokenizer.pad_token_id or 0
    for lo in range(0, len(order), batch):
        idx = order[lo: lo + batch]
        width = max(len(enc[k]) for k in idx)
        ids = torch.full((len(idx), width), pad, dtype=torch.long)
        mask = torch.zeros((len(idx), width), dtype=torch.long)
        for j, k in enumerate(idx):
            ids[j, : len(enc[k])], mask[j, : len(enc[k])] = torch.as_tensor(enc[k]), 1
        ids, mask = ids.to(device), mask.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
            h = backbone(input_ids=ids, attention_mask=mask).last_hidden_state.float()
        m = mask[..., None].float()
        out[idx] = ((h * m).sum(1) / m.sum(1).clamp(min=1.0)).cpu().numpy().astype(np.float16)
    return out


class FilingVectors:
    """Vectors by accession and the (date, permno) -> accession map."""

    def __init__(self, vectors: np.ndarray, accessions: pd.Series, prior: pd.Series, row_acc: pd.Series):
        self.vectors = vectors                                  # (n, 768) float16
        self.position = pd.Series(np.arange(len(accessions)), index=accessions.to_numpy())
        self.prior = prior                                      # accession -> prior-year accession
        self.row_acc = row_acc                                  # (date, permno) -> accession

    @property
    def width(self) -> int:
        return int(self.vectors.shape[1])

    def _rows(self, acc: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        pos = self.position.reindex(acc.to_numpy()).to_numpy(dtype=np.float64)
        ok = np.isfinite(pos)
        out = np.zeros((len(acc), self.width), dtype=np.float32)
        out[ok] = self.vectors[pos[ok].astype(np.int64)]
        return out, ok

    def for_panel(self, panel: Q3Panel, change: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """(N, 768) float32 vectors and an observed mask for the panel's rows; zeros where missing."""
        idx = pd.MultiIndex.from_arrays([pd.DatetimeIndex(panel.meeting), np.asarray(panel.firm_id).astype(np.int64)])
        acc = self.row_acc.reindex(idx)
        v, ok = self._rows(acc)
        if not change:
            return v, ok
        p, ok_p = self._rows(acc.map(self.prior))
        both = ok & ok_p
        return np.where(both[:, None], v - p, np.float32(0.0)), both

    def save(self, folder=FOLDER) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "vectors.npy", self.vectors)
        pd.DataFrame({"accession": self.position.index}).to_parquet(folder / "accessions.parquet", index=False)
        self.prior.rename("prior").rename_axis("accession").reset_index().to_parquet(folder / "prior.parquet", index=False)
        self.row_acc.rename("accession").reset_index().to_parquet(folder / "row_acc.parquet", index=False)

    @classmethod
    def load(cls, folder=FOLDER) -> "FilingVectors":
        acc = pd.read_parquet(folder / "accessions.parquet")["accession"]
        pr = pd.read_parquet(folder / "prior.parquet")
        r = pd.read_parquet(folder / "row_acc.parquet")
        row_acc = pd.Series(r["accession"].to_numpy(), index=pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex(r["date"]), r["permno"].astype(np.int64)], names=["date", "permno"]))
        return cls(np.load(folder / "vectors.npy"), acc, pr.set_index("accession")["prior"], row_acc)


def row_map(rows: pd.DataFrame, gvkey_cik: pd.DataFrame, sections: pd.DataFrame) -> pd.Series:
    """(date, permno) -> accession for panel ``rows`` (meeting, permno, gvkey)."""
    r = rows.assign(meeting=pd.to_datetime(rows["meeting"]), gvkey=rows["gvkey"].astype("string"))
    acc = filing_for_rows(r, gvkey_cik, sections)
    keep = acc.notna().to_numpy()
    return pd.Series(acc[keep].to_numpy(), index=pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex(r["meeting"])[keep], r["permno"].astype(np.int64).to_numpy()[keep]], names=["date", "permno"]))


def write_ids() -> None:
    """Local step: the gvkey -> CIK table (from the unregistered WRDS company table)."""
    from fedcore.ingest.edgar import universe

    FOLDER.mkdir(parents=True, exist_ok=True)
    universe()[["gvkey", "cik"]].astype(str).to_parquet(FOLDER / "gvkey_cik.parquet", index=False)


def build(device="cuda", batch: int = 16, max_tokens: int = MAX_TOKENS) -> FilingVectors:
    """Embed every filing a Fed-panel or shock-day row uses, plus their prior-year filings."""
    from fedcore.data import load
    from fedcore.q3.real import load_real_panel
    from fedcore.q3.shock_days import load_shock_panel
    from fedenc.text_tower import load_backbone

    gk = pd.read_parquet(FOLDER / "gvkey_cik.parquet")
    sections = usable(load("edgar_market_risk"))
    maps = [row_map(load_real_panel()[1], gk, sections), row_map(load_shock_panel()[1], gk, sections)]
    row_acc = pd.concat(maps)
    row_acc = row_acc[~row_acc.index.duplicated()]
    prior = prior_year(sections)
    used = pd.Index(row_acc.unique())
    prior = prior[prior.index.isin(used)]
    need = used.union(pd.Index(prior.unique()))
    sec = sections.drop_duplicates("accession").set_index("accession").loc[need]
    tokenizer, backbone = load_backbone(torch.bfloat16 if device != "cpu" else torch.float32, device)
    backbone.eval()
    count = lambda t: len(tokenizer(t, add_special_tokens=False)["input_ids"])
    texts = [document_text(f, select_text(s, count, max_tokens - 8)) for f, s in zip(sec["form"], sec["section"])]
    vectors = embed(texts, tokenizer, backbone, device, batch=batch, max_tokens=max_tokens)
    fv = FilingVectors(vectors, pd.Series(need), prior, row_acc)
    fv.save()
    meta = dict(n_filings=len(need), n_rows=int(len(row_acc)), with_prior=int(len(prior)), max_tokens=max_tokens,
                width=fv.width)
    (FOLDER / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(meta)
    return fv


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m fedenc.filing_vectors")
    ap.add_argument("step", choices=["ids", "build"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args(argv)
    write_ids() if a.step == "ids" else build(a.device, a.batch)


if __name__ == "__main__":
    main()
