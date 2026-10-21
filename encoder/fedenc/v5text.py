"""Q3 v5 text arm: market tower (returns, 64-d) + fundamentals tower (filings text, 64-d) -> b.

docs/encoder/q3-v5-plan.md section B3. Same base, projection, two-stage step-matched refit and
walk-forward as v4 (``fedenc.v4.V4Arm``), so results are directly comparable to B0-B5:

    mu = mu_base + P_m(delta_b) * s,   delta_b = head([market(history) | filings(text)])

Arms:
- ``T1``: EmbeddingGemma 2 fine-tuned (top ``train_top`` layers), learned attention pooling.
- ``T1m``: the same with mean pooling (the model's own embedding pooling).
- ``T1f``: EmbeddingGemma 2 frozen, learned attention pooling (does fine-tuning learn anything?).
- ``T0``: market tower alone, every row marked "no filing" (does text add anything?).

Training cost: each optimizer step reads ``firms_per_meeting`` randomly drawn firms from each of
``meetings_per_batch`` meetings (an unbiased draw of the within-meeting loss) and projects off the
beta span of that draw. Early stopping scores a fixed draw of ``eval_firms`` firms per validation
meeting, the same draw every epoch. Predictions use whole meetings. Filings are read in
micro-batches of ``doc_batch``.

    python -m fedenc.v5text tokenize                         # build the token store (CPU)
    python -m fedenc.v5text run --arms T0 T1 --device cuda   # training: needs the user's go-ahead
"""
from __future__ import annotations

import argparse
import copy
import json

import numpy as np
import pandas as pd
import torch
from torch import nn

from fedcore.config import PROCESSED
from fedcore.q3.contracts import Q3Panel
from fedcore.q3.train import _meeting_index_groups
from fedenc.decouple import _segments
from fedenc.encoder_transformer import PatchTransformerEncoder
from fedenc.text_tower import (MAX_TOKENS, FilingTower, TwoTowerModel, document_text, filing_for_rows,
                               load_backbone, select_text)
from fedenc.v3 import _deployed_loss, base_betas, project
from fedenc.v4 import V4Arm, _codes

STORE = PROCESSED / "q3_text"
TRAIN_TOKENS = 1024


# ----------------------------------------------------------------------------- token store

class TextStore:
    """Tokenized filing sections and the (meeting, permno) -> filing map."""

    def __init__(self, ids: list[np.ndarray], row_doc: pd.Series, pad_id: int = 0):
        self.ids = ids
        self.row_doc = row_doc.rename_axis(["meeting", "permno"])  # (meeting, permno) -> doc index
        self.pad_id = int(pad_id)

    def docs_for(self, panel: Q3Panel) -> np.ndarray:
        idx = pd.MultiIndex.from_arrays([pd.DatetimeIndex(panel.meeting), np.asarray(panel.firm_id).astype(np.int64)])
        return self.row_doc.reindex(idx).fillna(-1).to_numpy(dtype=np.int64)

    def batch(self, docs: np.ndarray, device, max_tokens: int = TRAIN_TOKENS) -> tuple[torch.Tensor, torch.Tensor]:
        seqs = [self.ids[d][:max_tokens] for d in docs]
        width = max((len(s) for s in seqs), default=1)
        ids = np.full((len(seqs), width), self.pad_id, dtype=np.int64)
        mask = np.zeros((len(seqs), width), dtype=np.int64)
        for k, s in enumerate(seqs):
            ids[k, : len(s)], mask[k, : len(s)] = s, 1
        return torch.as_tensor(ids, device=device), torch.as_tensor(mask, device=device)

    def save(self, folder=STORE) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        lengths = np.array([len(x) for x in self.ids], dtype=np.int64)
        np.savez(folder / "tokens.npz", flat=np.concatenate(self.ids) if self.ids else np.zeros(0, np.int64),
                 lengths=lengths, pad_id=self.pad_id)
        self.row_doc.rename("doc").reset_index().to_parquet(folder / "row_doc.parquet", index=False)

    @classmethod
    def load(cls, folder=STORE) -> "TextStore":
        z = np.load(folder / "tokens.npz")
        ids = np.split(z["flat"], np.cumsum(z["lengths"])[:-1]) if len(z["lengths"]) else []
        r = pd.read_parquet(folder / "row_doc.parquet")
        row_doc = pd.Series(r["doc"].to_numpy(), index=pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex(r["meeting"]), r["permno"].astype(np.int64)]))
        return cls(ids, row_doc, int(z["pad_id"]))


def build_store(tokenizer, rows: pd.DataFrame, gvkey_cik: pd.DataFrame, sections: pd.DataFrame,
                max_tokens: int = MAX_TOKENS) -> TextStore:
    """Token ids for every filing some panel row uses (rate-selected to ``max_tokens``)."""
    acc = filing_for_rows(rows, gvkey_cik, sections)
    used = sections[sections["accession"].isin(acc.dropna().unique())].reset_index(drop=True)
    count = lambda t: len(tokenizer(t, add_special_tokens=False)["input_ids"])
    ids = []
    for f in used.itertuples(index=False):
        text = document_text(f.form, select_text(f.section, count, max_tokens - 8))
        ids.append(np.asarray(tokenizer(text, truncation=True, max_length=max_tokens)["input_ids"], dtype=np.int64))
    position = pd.Series(np.arange(len(used)), index=used["accession"])
    doc = acc.map(position)
    keep = doc.notna().to_numpy()
    row_doc = pd.Series(doc[keep].astype(np.int64).to_numpy(), index=pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex(rows["meeting"])[keep], rows["permno"].astype(np.int64).to_numpy()[keep]]))
    return TextStore(ids, row_doc, tokenizer.pad_token_id or 0)


# ----------------------------------------------------------------------------- model

class TextCorrection(nn.Module):
    """(a_raw, b) from history and filing ids, in the shape v3's projection code expects."""

    input_keys = ("history", "history_mask", "doc")

    def __init__(self, market: nn.Module, tower: FilingTower | None, store: TextStore, b_only: bool = True,
                 doc_batch: int = 32, max_tokens: int = TRAIN_TOKENS):
        super().__init__()
        self.two = TwoTowerModel(market, b_only=b_only)
        self.tower = tower
        self.store = store
        self.doc_batch, self.max_tokens = int(doc_batch), int(max_tokens)
        self.head_b, self.head_a = self.two.head_b, self.two.head_a  # V4Arm zero-inits these

    def _text(self, doc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        device = doc.device
        d = doc.cpu().numpy()
        has = torch.as_tensor(d >= 0, device=device)
        text = torch.zeros(len(d), self.two.no_filing.shape[0], device=device)
        if self.tower is None:
            return text, torch.zeros_like(has)
        if not has.any():
            return text, has
        uniq, inv = np.unique(d[d >= 0], return_inverse=True)
        cuda = device.type == "cuda"
        parts = []
        for lo in range(0, len(uniq), self.doc_batch):
            ids, mask = self.store.batch(uniq[lo: lo + self.doc_batch], device, self.max_tokens)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
                parts.append(self.tower(ids, mask).float())
        vec = torch.cat(parts)
        text[torch.as_tensor(np.flatnonzero(d >= 0), device=device)] = vec[torch.as_tensor(inv, device=device)]
        return text, has

    def forward(self, history, history_mask, doc):
        text, has = self._text(doc)
        a, b = self.two(history, history_mask, text, has)
        return a, b[:, None]


# ----------------------------------------------------------------------------- arm

def subset_basis(span: torch.Tensor, inverse: torch.Tensor, width: int, tol: float = 1e-8) -> torch.Tensor:
    """Orthonormal basis rows of span[1, beta63, beta_smooth] within each meeting of a row subset."""
    q = torch.zeros_like(span)
    for m in range(width):
        rows = torch.nonzero(inverse == m).squeeze(-1)
        U, S, _ = torch.linalg.svd(span[rows], full_matrices=False)
        keep = S > tol * S.max()
        q[rows, : int(keep.sum())] = U[:, keep]
    return q


class V5TextArm(V4Arm):
    """V4Arm with the market tower plus a filings tower (or the market tower alone for ``T0``)."""

    def __init__(self, seed: int = 0, *, store: TextStore, backbone=None, use_text: bool = True,
                 train_top: int = 4, pooling: str = "attention", members: int = 1, max_epochs: int = 15,
                 patience: int = 4, lr: float = 1e-3, text_lr: float = 2e-5, firms_per_meeting: int = 24,
                 doc_batch: int = 16, max_tokens: int = TRAIN_TOKENS, eval_firms: int = 64, device=None, **kw):
        super().__init__(seed, encoder="transformer", b_only=True, members=members, lr=lr, max_epochs=max_epochs,
                         patience=patience, device=device, **kw)
        if use_text and backbone is None:
            raise ValueError("a text arm needs the EmbeddingGemma 2 backbone (fedenc.text_tower.load_backbone)")
        self.store, self.backbone, self.use_text = store, backbone, bool(use_text)
        self.train_top, self.pooling, self.text_lr = int(train_top), pooling, float(text_lr)
        self.firms_per_meeting, self.doc_batch, self.max_tokens = int(firms_per_meeting), int(doc_batch), int(max_tokens)
        self.eval_firms = int(eval_firms)
        self.track_train = False

    def config(self) -> dict:
        return dict(super().config(), model="text_two_tower", text="embeddinggemma-2" if self.use_text else None,
                    train_top=self.train_top, pooling=self.pooling, text_lr=self.text_lr,
                    firms_per_meeting=self.firms_per_meeting, eval_firms=self.eval_firms, max_tokens=self.max_tokens)

    def _new_model(self, panel: Q3Panel, device):
        market = PatchTransformerEncoder(n_channels=panel.history.shape[-1])
        tower = (FilingTower(copy.deepcopy(self.backbone), train_top=self.train_top, pooling=self.pooling)
                 if self.use_text else None)
        model = TextCorrection(market, tower, self.store, doc_batch=self.doc_batch, max_tokens=self.max_tokens)
        # from_pretrained leaves the backbone in eval mode, which also switches off gradient
        # checkpointing; V4Arm's stage-2 refit never calls train(), so set it here
        return model.to(device).train()

    def _optimizer(self, model):
        text = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("tower.backbone.")]
        rest = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("tower.backbone.")]
        groups = [dict(params=rest, lr=self.lr)] + ([dict(params=text, lr=self.text_lr)] if text else [])
        return torch.optim.AdamW(groups, weight_decay=self.weight_decay)

    def _data(self, p: Q3Panel, base, scaler, device) -> dict:
        d = V4Arm._data(p, base, scaler, device)
        x = d["x"]
        b63, bw = base_betas(p)
        d["x"] = dict(history=x["history"], history_mask=x["history_mask"],
                      doc=torch.as_tensor(self.store.docs_for(p), dtype=torch.long, device=device))
        d["span"] = torch.as_tensor(np.column_stack((np.ones(len(b63)), b63, bw)), dtype=torch.float64, device=device)
        return d

    def _batches(self, groups, rng, device):
        order = rng.permutation(len(groups))
        for start in range(0, len(order), self.meetings_per_batch):
            picked = [rng.choice(groups[k], size=min(self.firms_per_meeting, len(groups[k])), replace=False)
                      for k in order[start:start + self.meetings_per_batch]]
            yield torch.as_tensor(np.sort(np.concatenate(picked)), device=device)

    @staticmethod
    def _draw_loss(model, d, rows):
        """Deployed loss on a within-meeting draw, projected off that draw's own beta span."""
        inverse, width, _ = _segments(d["codes"][rows])
        q = torch.zeros_like(d["q"]).index_copy_(0, rows, subset_basis(d["span"][rows], inverse, width))
        return _deployed_loss(model, d["x"], rows, q, d["codes"], d["s"], d["e"])

    def _eval(self, model, d, groups) -> float:
        """Validation loss on a fixed draw of ``eval_firms`` firms per meeting (reading every
        filing of every validation meeting each epoch would dominate the cost)."""
        rng = np.random.default_rng(12345)
        picked = [np.sort(rng.choice(g, size=min(self.eval_firms, len(g)), replace=False)) for g in groups]
        model.eval()
        with torch.no_grad():
            tot = sum(self._draw_loss(model, d, torch.as_tensor(np.concatenate(picked[k:k + 4]), device=d["e"].device)).item()
                      * len(picked[k:k + 4]) for k in range(0, len(picked), 4))
        model.train()
        return tot / len(groups)

    def _step(self, model, opt, d, rows):
        opt.zero_grad(set_to_none=True)
        loss = self._draw_loss(model, d, rows)
        loss.backward()
        nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()


# ----------------------------------------------------------------------------- runner

def makers(store: TextStore, backbone, device=None, members: int = 1) -> dict:
    common = dict(store=store, members=members, device=device)
    return {
        "T0": lambda seed: V5TextArm(seed, use_text=False, **common),
        "T1": lambda seed: V5TextArm(seed, backbone=backbone, train_top=4, pooling="attention", **common),
        "T1m": lambda seed: V5TextArm(seed, backbone=backbone, train_top=4, pooling="mean", **common),
        "T1f": lambda seed: V5TextArm(seed, backbone=backbone, train_top=0, pooling="attention", **common),
    }


def tokenize() -> TextStore:
    from fedcore.data import load
    from fedcore.q3.real import load_real_panel
    from fedcore.ingest.edgar import universe

    _, rows, _ = load_real_panel()
    rows = rows.assign(meeting=pd.to_datetime(rows["meeting"]), gvkey=rows["gvkey"].astype("string"))
    sections = load("edgar_market_risk")
    tokenizer, _ = load_backbone()
    store = build_store(tokenizer, rows, universe()[["gvkey", "cik"]], sections)
    store.save()
    lengths = np.array([len(x) for x in store.ids])
    meta = dict(n_docs=len(store.ids), rows_with_text=int(len(store.row_doc)), rows=int(len(rows)),
                tokens_median=float(np.median(lengths)) if len(lengths) else 0.0,
                tokens_p90=float(np.percentile(lengths, 90)) if len(lengths) else 0.0)
    (STORE / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(meta)
    return store


def run(arms, through="2021-12-31", seeds=(0,), members=1, device=None, test_size=16, store=None):
    """Development walk-forward on the v4 folds (min_train 60, test_size 16). Training."""
    from fedcore.q3.real import UNIVERSES
    from fedcore.results import ResultStore
    from fedenc.config import RESULTS
    from fedenc.runs import save_runs
    from fedenc.v4 import panel_fingerprint
    from fedenc.workflow import bounded_panel

    if pd.Timestamp(through) > pd.Timestamp("2021-12-31"):
        raise ValueError("v5 development runs stop at 2021-12-31")
    panel, through_ts, _ = bounded_panel(through, with_rows=True)
    text = TextStore.load()
    backbone = None if set(arms) <= {"T0"} else load_backbone()[1]
    every = makers(text, backbone, device=device, members=members)
    chosen = {a: every[a] for a in arms}
    return save_runs(panel, chosen, prefix=f"v5text-to{through_ts.year}", role="exploratory",
                     params=dict(through=str(through_ts.date()), panel_fingerprint=panel_fingerprint(panel),
                                 configs={a: m(0).config() for a, m in chosen.items()},
                                 plan="docs/encoder/q3-v5-plan.md"),
                     min_train=60, test_size=test_size, seeds=seeds, store=store or ResultStore(RESULTS),
                     datasets=list(UNIVERSES["sp500"].inputs) + ["edgar_market_risk"])


# ----------------------------------------------------------------------------- scoring

def _clustered(values: pd.Series, n: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    """Pooled mean of a row-level 0/1 or numeric series, meeting-clustered bootstrap interval."""
    g = values.dropna().groupby(level=0)
    tot, cnt = g.sum(), g.count()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(tot), (n, len(tot)))
    draws = tot.to_numpy()[idx].sum(1) / cnt.to_numpy()[idx].sum(1)
    return float(tot.sum() / cnt.sum()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def score(prefix: str = "v5text-to2021", arms=("T0", "T1", "T1m", "T1f"), base_prefix: str = "v4-to2021",
          save: bool = True):
    """The conditional scoring pre-stated in q3-v5-plan.md (B3 first test). Read-only apart from the score run."""
    from fedcore.q3.channels import load_channels
    from fedcore.q3.real import FUNDAMENTAL_NAMES
    from fedenc.v4 import PRIMARY, STANDOUT, _boot, _fmt, _load
    from fedenc.workflow import bounded_panel

    loaded = {a: _load(prefix, a) for a in arms}
    loaded = {a: v for a, v in loaded.items() if v is not None}
    b0 = _load(base_prefix, "B0")
    if not loaded or b0 is None:
        raise RuntimeError("need saved v5 text runs and the v4 B0 run")
    P = {a: v[0] for a, v in loaded.items()}
    panel, _, rows = bounded_panel("2021-12-31", with_rows=True)
    s = pd.Series(panel.shock, index=pd.DatetimeIndex(panel.meeting)).groupby(level=0).first()
    idx = pd.MultiIndex.from_arrays([pd.DatetimeIndex(panel.meeting), panel.firm_id.astype(int)])
    fund = pd.DataFrame(panel.fundamentals, columns=list(FUNDAMENTAL_NAMES), index=idx)
    ch = load_channels().set_index(["meeting", "permno"])
    ch.index = pd.MultiIndex.from_arrays([pd.DatetimeIndex(ch.index.get_level_values(0)),
                                          ch.index.get_level_values(1).astype(int)])

    base = b0[0]
    out, cond = [], []
    for a, p in P.items():
        d = p.join(base[["mu"]].rename(columns={"mu": "mu_b0"}), how="inner")
        d["s"] = s.reindex(d.index.get_level_values(0)).to_numpy()
        dm = lambda c: d[c] - d.groupby(level=0)[c].transform("mean")
        d["y_c"], d["base_c"], d["mu_c"], d["b0_c"] = dm("target"), dm("mu_base"), dm("mu"), dm("mu_b0")
        d["dev"] = d["mu_c"] - d["base_c"]               # departure from the two-beta base
        d["miss"] = d["y_c"] - d["base_c"]               # what the base got wrong
        big = d[d["s"].abs() >= PRIMARY]
        for label, sub in (("all big", big), ("without standouts", big[~big.index.get_level_values(0).isin(STANDOUT)])):
            q = sub.groupby(level=0)["dev"].transform(lambda x: x.abs().rank(pct=True))
            top = sub[q > 0.9]
            right = (np.sign(top["dev"]) == np.sign(top["miss"])).astype(float)
            toward = np.sign(top["dev"]) * top["miss"]
            v = fund.reindex(sub.index)
            r = v[["cash", "leverage"]].groupby(level=0).rank(pct=True)
            vuln = sub[((r["cash"] <= 1 / 3) & (r["leverage"] >= 2 / 3)).to_numpy() & (sub["dev"].abs() > 0.1).to_numpy()]
            dd = ch["distance_to_default"].reindex(sub.index)
            low_dd = sub[(dd.groupby(level=0).rank(pct=True) <= 1 / 3).to_numpy() & (sub["dev"].abs() > 0.1).to_numpy()]
            hit = lambda x: _clustered((np.sign(x["dev"]) == np.sign(x["miss"])).astype(float)) if len(x) else (np.nan,) * 3
            cs = ((sub["y_c"] - sub["mu_c"]) ** 2 - (sub["y_c"] - sub["b0_c"]) ** 2).groupby(level=0).mean()
            cond.append(dict(arm=a, slice=label, meetings=sub.index.get_level_values(0).nunique(),
                             top_decile_hit=_fmt(_clustered(right)), top_decile_realized_toward=round(float(toward.mean()), 3),
                             top_decile_predicted=round(float(top["dev"].abs().mean()), 3),
                             vulnerable_hit=_fmt(hit(vuln)), vulnerable_n=len(vuln),
                             low_dd_hit=_fmt(hit(low_dd)), low_dd_n=len(low_dd),
                             cross_firm_vs_b0=_fmt(_boot(cs))))
        ge5 = d[d["s"].abs() >= 0.5]
        cs5 = ((ge5["y_c"] - ge5["mu_c"]) ** 2 - (ge5["y_c"] - ge5["b0_c"]) ** 2).groupby(level=0).mean()
        tot = ((d["target"] - d["mu"]) ** 2 - (d["target"] - d["mu_b0"]) ** 2).groupby(level=0).mean()
        out.append(dict(arm=a, cross_firm_ge5bp_vs_b0=_fmt(_boot(cs5)), all_meetings_raw_vs_b0=_fmt(_boot(tot)),
                        delta_b_sd=round(float(d["delta_b"].std()), 4) if "delta_b" in d else np.nan, run=loaded[a][1]))
    cond_t, out_t = pd.DataFrame(cond), pd.DataFrame(out)
    with pd.option_context("display.width", 300, "display.max_columns", 40, "display.max_colwidth", 40):
        print("== conditional view (|s| >= 2.5bp)")
        print(cond_t.to_string(index=False))
        print("\n== context")
        print(out_t.drop(columns=["run"]).to_string(index=False))
    if save:
        from fedcore.questions import specs
        from fedcore.results import ResultStore
        from fedenc.config import RESULTS

        ResultStore(RESULTS).save(specs()["Q3"], f"{prefix}-score", role="exploratory",
                                  params=dict(prefix=prefix, base_prefix=base_prefix, primary=PRIMARY,
                                              runs={a: v[1] for a, v in loaded.items()}),
                                  tables=dict(conditional=cond_t, context=out_t),
                                  notes="Read-only conditional scoring of v5 text runs (docs/encoder/q3-v5-plan.md B3).")
    return cond_t, out_t


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m fedenc.v5text")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tokenize")
    sc = sub.add_parser("score")
    sc.add_argument("--no-save", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--arms", nargs="+", default=["T0", "T1"])
    r.add_argument("--seeds", nargs="+", type=int, default=[0])
    r.add_argument("--members", type=int, default=1)
    r.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    if args.cmd == "tokenize":
        tokenize()
    elif args.cmd == "score":
        score(save=not args.no_save)
    else:
        print(run(args.arms, seeds=tuple(args.seeds), members=args.members, device=args.device).to_string(index=False))


if __name__ == "__main__":
    main()
