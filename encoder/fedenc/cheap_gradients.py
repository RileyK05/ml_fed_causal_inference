"""B1-B4: gradient checks on the transformer response model. No optimizer steps.

One fixed batch from the development period's first inner training fold: up to 64 firms
from each of the last 16 training meetings, scaled with statistics from those training
rows. The model is freshly initialized (eval mode, so dropout is off). Initialization
alone cannot establish the gradient balance of a trained model.

The "trunk" is everything that builds z (encoder, fundamentals and context branches,
fusion); the "encoder" is the patch transformer alone.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch

from fedcore.q3 import ResponseModel
from fedcore.q3.cheap.base import Result, check
from fedcore.q3.cheap.data import CheapData, Config
from fedcore.q3.cheap.linear import inner_folds, rows_of
from fedcore.q3.train import ChannelScaler, _tensors, seed_everything

from fedenc.encoder_transformer import PatchTransformerEncoder

BATCH_MEETINGS = 16
FIRMS_PER_MEETING = 64


def _setup(data: CheapData, cfg: Config):
    if "b_setup" in data._memo:
        return data._memo["b_setup"]
    fold = inner_folds(data, cfg)[0]
    train_rows = rows_of(data, fold.train)
    scaler = ChannelScaler.fit(data.panel, train_rows)
    rng = np.random.default_rng(cfg.seed)
    pick = []
    for m in fold.train[-BATCH_MEETINGS:]:
        r = rows_of(data, [m])
        pick.append(rng.choice(r, size=min(FIRMS_PER_MEETING, r.size), replace=False))
    batch = _tensors(data.panel.subset(np.sort(np.concatenate(pick))), scaler, torch.device("cpu"))
    seed_everything(cfg.seed)
    model = ResponseModel(PatchTransformerEncoder(), n_fund=data.panel.fundamentals.shape[1],
                          n_ctx=data.panel.context.shape[1]).eval()
    data._memo["b_setup"] = (model, batch)
    return model, batch


def _z(model, batch):
    h = model.encoder(batch["history"], batch["history_mask"])
    f = model.fund(torch.cat((batch["fundamentals"], batch["fundamentals_mask"].float()), dim=-1))
    c = model.context_mlp(batch["context"])
    return model.fusion(torch.cat((h, f, c), dim=-1))


def _trunk_params(model):
    head = {id(p) for p in model.head.parameters()}
    return [p for p in model.parameters() if id(p) not in head]


def _encoder_mask(model, params):
    enc = {id(p) for p in model.encoder.parameters()}
    return [id(p) in enc for p in params]


def _flat(grads, keep=None):
    parts = [g.reshape(-1) for g, k in zip(grads, keep or [True] * len(grads)) if k and g is not None]
    return torch.cat(parts) if parts else torch.zeros(1)


def path_gradients(model, batch, shock=None, weights=None, rows=None) -> dict:
    """Separate trunk gradients through a and through b, from the same residual.

    dL/dmu is computed once; the a path back-propagates it through a, the b path
    back-propagates dL/dmu * s through b. ``rows`` restricts the residual to a subset
    (shock-size bins); ``weights`` gives a per-row loss weight (variance weighting).
    """
    s = batch["shock"] if shock is None else shock
    z = _z(model, batch)
    a = model.head.a(z).squeeze(-1)
    b = model.head.b(z).squeeze(-1)
    mu = a + b * s
    err = (mu - batch["target"]) ** 2
    if weights is not None:
        err = err * weights
    loss = _weighted_meeting_mse(err, batch["codes"])
    g = torch.autograd.grad(loss, mu, retain_graph=True)[0]
    if rows is not None:
        keep = torch.zeros_like(g)
        keep[rows] = 1.0
        g = g * keep
    params = _trunk_params(model)
    ga = torch.autograd.grad(a, params, grad_outputs=g, retain_graph=True, allow_unused=True)
    gb = torch.autograd.grad(b, params, grad_outputs=g * s, retain_graph=True, allow_unused=True)
    enc = _encoder_mask(model, params)
    return {"a_trunk": _flat(ga), "b_trunk": _flat(gb), "a_encoder": _flat(ga, enc), "b_encoder": _flat(gb, enc),
            "mu": mu.detach()}


def _weighted_meeting_mse(err, codes):
    """``per_meeting_mse`` for already-squared (optionally weighted) errors."""
    n = int(codes.max().item()) + 1
    sse = torch.zeros(n, dtype=err.dtype).scatter_add_(0, codes, err)
    cnt = torch.zeros(n, dtype=err.dtype).scatter_add_(0, codes, torch.ones_like(err))
    return (sse / cnt.clamp(min=1)).mean()


def _summary(gr: dict) -> dict:
    out = {}
    for part in ("trunk", "encoder"):
        na, nb = float(gr[f"a_{part}"].norm()), float(gr[f"b_{part}"].norm())
        cos = float(torch.nn.functional.cosine_similarity(gr[f"a_{part}"], gr[f"b_{part}"], dim=0)) if na and nb else np.nan
        out.update({f"{part}_a_norm": na, f"{part}_b_norm": nb,
                    f"{part}_a_over_b": na / nb if nb else np.inf, f"{part}_cosine": cos})
    return out


@check("B1", "Encoder gradients through a vs through b", "gradient")
def b1(data: CheapData, cfg: Config) -> Result:
    model, batch = _setup(data, cfg)
    rows = [{"bin": "all", **_summary(path_gradients(model, batch))}]
    abs_s = batch["shock"].abs().numpy()
    edges = np.quantile(np.unique(abs_s), [1 / 3, 2 / 3])
    for name, sel in (("quiet", abs_s <= edges[0]), ("middle", (abs_s > edges[0]) & (abs_s <= edges[1])),
                      ("large", abs_s > edges[1])):
        if sel.any():
            r = _summary(path_gradients(model, batch, rows=torch.as_tensor(np.flatnonzero(sel))))
            rows.append({"bin": name, "mean_abs_s_bp": float(10 * abs_s[sel].mean()), **r})
    table = pd.DataFrame(rows)
    return Result({"trunk_a_over_b": float(table.iloc[0]["trunk_a_over_b"]),
                   "encoder_a_over_b": float(table.iloc[0]["encoder_a_over_b"])},
                  {"by_shock_bin": table},
                  "Norm of each path's contribution to the trunk/encoder gradient at initialization, by meeting "
                  "|s| tercile. a_over_b >> 1 is the claimed imbalance.")


@check("B2", "Agreement of the a and b gradient paths", "gradient")
def b2(data: CheapData, cfg: Config) -> Result:
    model, batch = _setup(data, cfg)
    r = _summary(path_gradients(model, batch))
    return Result({"trunk_cosine": r["trunk_cosine"], "encoder_cosine": r["encoder_cosine"]},
                  {"cosine": pd.DataFrame([r])},
                  "Cosine similarity of the two paths' gradients. Near +1: a's larger gradient pushes the "
                  "representation the way b wants anyway; near 0 or negative: they compete.")


@check("B3", "Shock-unit rescaling (10bp vs 1bp)", "gradient")
def b3(data: CheapData, cfg: Config) -> Result:
    model, batch = _setup(data, cfg)
    base = path_gradients(model, batch)
    m2 = copy.deepcopy(model)
    with torch.no_grad():
        m2.head.b.weight /= 10.0
        m2.head.b.bias /= 10.0
    resc = path_gradients(m2, batch, shock=batch["shock"] * 10.0)
    gap = float((base["mu"] - resc["mu"]).abs().max())
    rows = [{"units": "10bp (current)", **_summary(base)}, {"units": "1bp, b rescaled", **_summary(resc)}]
    table = pd.DataFrame(rows)
    return Result({"prediction_gap": gap, "a_over_b_10bp": float(table.iloc[0]["trunk_a_over_b"]),
                   "a_over_b_1bp": float(table.iloc[1]["trunk_a_over_b"])},
                  {"units": table},
                  "Same predictions, different shock units. If a_over_b changes a lot, the apparent imbalance is "
                  "a parameterization effect, which optimizer dynamics and weight decay also see.")


@check("B4", "Backward-only dial and variance weighting", "gradient")
def b4(data: CheapData, cfg: Config) -> Result:
    model, batch = _setup(data, cfg)
    params = _trunk_params(model)
    s = batch["shock"]
    ref_mu, rows = None, []
    for lam in (1.0, 0.5, 0.0):
        model.zero_grad(set_to_none=True)
        z = _z(model, batch)
        z_a = z.detach() + lam * (z - z.detach())
        mu = model.head.a(z_a).squeeze(-1) + model.head.b(z).squeeze(-1) * s
        loss = _weighted_meeting_mse((mu - batch["target"]) ** 2, batch["codes"])
        loss.backward()
        ref_mu = mu.detach() if ref_mu is None else ref_mu
        trunk = torch.cat([p.grad.reshape(-1) for p in params if p.grad is not None])
        a_head = torch.cat([p.grad.reshape(-1) for p in model.head.a.parameters()])
        rows.append({"lambda": lam, "prediction_gap": float((mu.detach() - ref_mu).abs().max()),
                     "trunk_grad_norm": float(trunk.norm()), "a_head_grad_norm": float(a_head.norm())})
    model.zero_grad(set_to_none=True)
    rvol = batch["history"][:, -21:, 2].abs().mean(dim=1) + 0.5          # scaled proxy, positive
    w = 1.0 / rvol ** 2
    w = w / w.mean()
    plain = _summary(path_gradients(model, batch))
    weighted = _summary(path_gradients(model, batch, weights=w))
    table = pd.DataFrame(rows)
    wt = pd.DataFrame([{"loss": "unweighted", **plain}, {"loss": "1/variance weighted", **weighted}])
    return Result({"max_prediction_gap": float(table["prediction_gap"].max()),
                   "a_head_grad_constant": bool(np.allclose(table["a_head_grad_norm"], table["a_head_grad_norm"].iloc[0], rtol=1e-4)),
                   "a_over_b_weighted": weighted["trunk_a_over_b"], "a_over_b_unweighted": plain["trunk_a_over_b"]},
                  {"dial": table, "variance_weighting": wt},
                  "The dial scales only a's backward path: predictions and a's own gradients must not change. "
                  "Variance weighting rescales both paths row by row, so it does not by itself favor b.")
