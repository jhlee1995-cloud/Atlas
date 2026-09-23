"""
density.py -- where the reference distribution is dense and where it thins out.

  knn_density : distance to the k-th nearest reference neighbour (k=10) as a density proxy.
                Records reference quantiles of log-radius, the sparse fraction of every split
                (radius above the reference 95th percentile), and the median log-radius shift.

This is the "familiarity" field the compute-reduction lever reads (dense + high margin =>
cheap path). The sparse fraction per corruption set is also the cheapest first look at
"which input types does the model not know" -- without any labels.
"""
import numpy as np

from ..registry import invariant
from ._util import knn_radii, subsample


@invariant("knn_density", needs=("ref",), cost="medium")
def knn_density(ctx, cfg):
    """kNN-radius density field on the reference; sparse fraction per split."""
    n = int(cfg.get("n_ref", 10000))
    k = int(cfg.get("k", 10))
    n_query = int(cfg.get("n_query", 3000))
    Xr = subsample(ctx.ref, n, ctx.rng)
    # self-queries come from the fit set, so drop each point's zero distance to itself; otherwise the
    # reference radius is the (k-1)-th neighbour and every held-out split looks too sparse
    r_ref = knn_radii(Xr, subsample(Xr, n_query, ctx.rng), k=k, exclude_self=True)
    lr = np.log(r_ref + 1e-12)
    q = np.quantile(lr, [0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    thresh95 = np.exp(q[4])
    out = {
        "k": k, "n_fit": int(len(Xr)),
        "ref_log_radius_quantiles": {"q05": q[0], "q25": q[1], "q50": q[2], "q75": q[3], "q95": q[4], "q99": q[5]},
        "ref_sparse_frac_self": float((r_ref > thresh95).mean()),   # ~0.05 by construction
        "splits": {},
    }

    def measure(name, X):
        r = knn_radii(Xr, subsample(X, n_query, ctx.rng), k=k)
        out["splits"][name] = {
            "sparse_frac": float((r > thresh95).mean()),
            "median_log_radius_shift": float(np.median(np.log(r + 1e-12)) - q[2]),
        }

    if ctx.test is not None:
        measure("test", ctx.test)
    if ctx.panel is not None:
        measure("panel", ctx.panel)
    for s, X in ctx.corrupt.items():
        measure(s, X)
    for s, X in ctx.ood.items():
        measure(s, X)
    return out
