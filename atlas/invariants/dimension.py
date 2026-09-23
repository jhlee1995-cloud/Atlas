"""
dimension.py -- how many directions the reference distribution actually uses, per layer.

  pca_spectrum : participation ratio, entropy effective rank, dim95 (linear dimension)
  twonn_id     : TwoNN intrinsic dimension (Facco et al. 2017), nonlinear / local

Known shape to sanity-check against (Ansuini et al. 2019): ID rises through early layers,
peaks mid-network, then falls toward the output ("hunchback"), while the linear dimension
(PR / dim95) can stay high. A profile that is flat or monotone is worth a second look.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors

from ..registry import invariant
from ._util import pca_frame, subsample


@invariant("pca_spectrum", needs=("ref",), cost="cheap")
def pca_spectrum(ctx, cfg):
    """Eigen-spectrum summaries of the reference covariance at this layer."""
    n = int(cfg.get("n_ref", 10000))
    X = subsample(ctx.ref, n, ctx.rng)
    _, w, _ = pca_frame(X)
    tot = w.sum() + 1e-12
    p = w / tot
    pr = float(tot ** 2 / ((w ** 2).sum() + 1e-12))
    ent = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))
    cum = np.cumsum(p)
    dim95 = int(np.searchsorted(cum, 0.95) + 1)
    dim99 = int(np.searchsorted(cum, 0.99) + 1)
    return {
        "dim": int(X.shape[1]),
        "participation_ratio": pr,
        "effective_rank_entropy": ent,
        "dim95": dim95,
        "dim99": dim99,
        "top_eig_frac": float(p[0]),
        "eig_top20": w[:20].tolist(),
    }


def twonn(X, discard_frac=0.1):
    """TwoNN estimator: mu = r2/r1; F(mu) = 1 - mu^-d ; fit -log(1-F) = d log(mu) through origin.
    Discards the top `discard_frac` of mu (outliers) as in the original method."""
    nn = NearestNeighbors(n_neighbors=3).fit(X)
    d, _ = nn.kneighbors(X)
    r1, r2 = d[:, 1], d[:, 2]
    ok = r1 > 0
    mu = np.sort(r2[ok] / r1[ok])
    n = len(mu)
    if n < 20:
        return float("nan")
    F = np.arange(1, n + 1) / n
    keep = int(n * (1 - discard_frac))
    x = np.log(mu[:keep])
    y = -np.log(1 - F[:keep])
    return float((x * y).sum() / ((x * x).sum() + 1e-12))


@invariant("twonn_id", needs=("ref",), cost="medium")
def twonn_id(ctx, cfg):
    """TwoNN intrinsic dimension on a reference subsample, with a bootstrap spread."""
    n = int(cfg.get("n_ref", 5000))
    boots = int(cfg.get("bootstrap", 3))
    vals = []
    for b in range(boots):
        X = subsample(ctx.ref, n, ctx.rng)
        vals.append(twonn(X))
    vals = np.array(vals)
    out = {"id": float(np.nanmean(vals)), "id_std": float(np.nanstd(vals)), "n": n}
    # per-class ID on the reference (cheap extra: is the manifold class-uniform?)
    if cfg.get("per_class", True):
        pc = []
        for c in range(ctx.n_classes):
            Xc = ctx.ref[ctx.ref_labels == c]
            if len(Xc) >= 200:
                pc.append(twonn(subsample(Xc, n, ctx.rng)))
        if pc:
            out["id_per_class_mean"] = float(np.nanmean(pc))
            out["id_per_class_std"] = float(np.nanstd(pc))
    return out
