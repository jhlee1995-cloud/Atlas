"""
landmarks.py -- the fixed points of the map: class centers, their arrangement, hubs.

  class_centers  : centers, RMS within-class radius, between/within separation ratio,
                   nearest-center accuracy on ref/test (how far the centers explain the labels)
  neural_collapse: NC1 = tr(Sigma_W Sigma_B^+)/K, simplex-ETF cosine deviation, norm CV
                   (Papyan/Han/Donoho 2020). The penult of a converged classifier should sit
                   near the ETF; the valley structure the 7 axes read is the deviation from it.
  hubness        : skewness of the k-occurrence distribution (Radovanovic et al. 2010). High
                   skew = a few points are everyone's nearest neighbour, which distorts every
                   nearest-center / kNN measurement at that layer.
"""
import numpy as np
from scipy.stats import skew
from sklearn.neighbors import NearestNeighbors

from ..registry import invariant
from ._util import class_stats, nearest_center, subsample


@invariant("class_centers", needs=("ref", "labels"), cost="cheap")
def class_centers(ctx, cfg):
    """Centers, within-class radii, between/within separation, nearest-center accuracy."""
    K = ctx.n_classes
    C, r, counts, mu = class_stats(ctx.ref, ctx.ref_labels, K)
    present = counts > 0
    D = np.linalg.norm(C[:, None] - C[None], axis=2)
    np.fill_diagonal(D, np.inf)
    nearest_other = D.min(1)
    sep_ratio = float((nearest_other[present] / (r[present] + 1e-12)).mean())
    # pairwise separation normalized by RMS radii: sep_ij = d_ij / sqrt(0.5 (r_i^2 + r_j^2))
    R2 = 0.5 * (r[:, None] ** 2 + r[None, :] ** 2)
    sep = np.where(np.isfinite(D), D / np.sqrt(R2 + 1e-12), np.inf)
    out = {
        "n_classes_present": int(present.sum()),
        "within_radius_mean": float(r[present].mean()),
        "within_radius_cv": float(r[present].std() / (r[present].mean() + 1e-12)),
        "nearest_other_center_mean": float(nearest_other[present].mean()),
        "sep_ratio": sep_ratio,
        "pair_sep_min": float(sep[np.isfinite(sep)].min()),
        "pair_sep_median": float(np.median(sep[np.isfinite(sep)])),
        "centers": C.tolist(),
        "radius": r.tolist(),
    }
    # nearest-center classification accuracy = how much of the label structure the
    # landmarks alone carry at this layer
    idx = ctx.rng.choice(len(ctx.ref), size=min(5000, len(ctx.ref)), replace=False)
    nc_ref, _ = nearest_center(ctx.ref[idx], C)
    out["nearest_center_acc_ref"] = float((nc_ref == ctx.ref_labels[idx]).mean())
    if ctx.test is not None:
        nc_t, _ = nearest_center(ctx.test, C)
        out["nearest_center_acc_test"] = float((nc_t == ctx.test_labels).mean())
        if ctx.test_preds is not None and "argmax" in ctx.test_preds:
            out["nearest_center_agrees_with_model"] = float((nc_t == ctx.test_preds["argmax"]).mean())
    return out


@invariant("neural_collapse", needs=("ref", "labels"), cost="cheap")
def neural_collapse(ctx, cfg):
    """NC1 (within/between collapse), ETF cosine deviation, class-mean norm CV."""
    K = ctx.n_classes
    X = ctx.ref.astype(np.float64)
    y = ctx.ref_labels
    C, r, counts, mu = class_stats(X, y, K)
    present = counts > 0
    M = C[present] - mu                              # centered class means
    Kp = int(present.sum())
    Sigma_B = M.T @ M / Kp
    W = X - C[y]
    Sigma_W = W.T @ W / len(X)
    nc1 = float(np.trace(Sigma_W @ np.linalg.pinv(Sigma_B)) / Kp)
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    cos = Mn @ Mn.T
    off = cos[~np.eye(Kp, dtype=bool)]
    ideal = -1.0 / (Kp - 1)
    norms = np.linalg.norm(M, axis=1)
    return {
        "nc1": nc1,
        "etf_cos_mean_offdiag": float(off.mean()),
        "etf_cos_ideal": float(ideal),
        "etf_deviation": float(np.abs(off - ideal).mean()),
        "etf_cos_std_offdiag": float(off.std()),
        "class_mean_norm_cv": float(norms.std() / (norms.mean() + 1e-12)),
        "global_mean_norm_over_class_mean_norm": float(np.linalg.norm(mu) / (norms.mean() + 1e-12)),
    }


@invariant("hubness", needs=("ref",), cost="medium")
def hubness(ctx, cfg):
    """k-occurrence skewness on a reference subsample (k=10). >1 means strong hubs."""
    n = int(cfg.get("n_ref", 5000))
    k = int(cfg.get("k", 10))
    X = subsample(ctx.ref, n, ctx.rng)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nn.kneighbors(X)
    occ = np.bincount(idx[:, 1:].ravel(), minlength=len(X))
    return {
        "k": k,
        "n": int(len(X)),
        "k_occurrence_skew": float(skew(occ)),
        "k_occurrence_max": int(occ.max()),
        "frac_antihubs": float((occ == 0).mean()),
        "top1pct_share": float(np.sort(occ)[::-1][: max(1, len(occ) // 100)].sum() / occ.sum()),
    }
