"""
adjacency.py -- which clusters touch which, and in what order they merge as scale grows.

  class_adjacency : normalized center distances (sep_ij), per-pair valley ratio (kNN radius at
                    the midpoint over the radius at the centers; > 1 = sparse gap = a valley),
                    nearest-center confusion matrix on test (row-normalized)
  merge_order     : single-linkage over class centers using sep_ij -> the sequence of class
                    pairs that merge and the height at which they do (multiscale H0 of the
                    landmark set). Stable across seeds if the map is real; the critic compares
                    merge-height ranks with Kendall tau.

Only H0 is tracked. H1 on sample clouds was already tried (TRAJ_LOOP) and is not stable at
this scale; the merge order is the topological summary that survives.
"""
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from ..registry import invariant
from ._util import class_stats, knn_radii, nearest_center, subsample


def _sep_matrix(C, r, present):
    D = np.linalg.norm(C[:, None] - C[None], axis=2)
    R2 = 0.5 * (r[:, None] ** 2 + r[None, :] ** 2)
    S = D / np.sqrt(R2 + 1e-12)
    S[~present, :] = np.nan
    S[:, ~present] = np.nan
    np.fill_diagonal(S, 0.0)
    return S


@invariant("class_adjacency", needs=("ref", "labels"), cost="medium")
def class_adjacency(ctx, cfg):
    """Pairwise separation, valley ratio per pair, nearest-center confusion on test."""
    K = ctx.n_classes
    C, r, counts, _ = class_stats(ctx.ref, ctx.ref_labels, K)
    present = counts > 0
    S = _sep_matrix(C, r, present)

    # valley ratio: kNN radius (k=10) at the pair midpoint relative to the centers' radii,
    # measured against a reference subsample. > 1 => the midpoint is sparser than the
    # centers (a valley exists between the two clusters)
    n = int(cfg.get("n_ref", 8000))
    k = int(cfg.get("k", 10))
    Xs = subsample(ctx.ref, n, ctx.rng)
    r_center = knn_radii(Xs, C[present], k=k)
    pres_idx = np.where(present)[0]
    V = np.full((K, K), np.nan)
    mids, pairs = [], []
    for a_i, a in enumerate(pres_idx):
        for b_i, b in enumerate(pres_idx):
            if b <= a:
                continue
            mids.append(0.5 * (C[a] + C[b]))
            pairs.append((a, b, a_i, b_i))
    if mids:
        r_mid = knn_radii(Xs, np.array(mids), k=k)
        for (a, b, a_i, b_i), rm in zip(pairs, r_mid):
            V[a, b] = V[b, a] = rm / (0.5 * (r_center[a_i] + r_center[b_i]) + 1e-12)
    iu = np.triu_indices(K, 1)
    vals = V[iu]
    vals = vals[np.isfinite(vals)]
    out = {
        "sep_matrix": np.nan_to_num(S, nan=-1.0).tolist(),
        "valley_ratio_matrix": np.nan_to_num(V, nan=-1.0).tolist(),
        "valley_ratio_mean": float(vals.mean()) if len(vals) else None,
        "valley_ratio_min": float(vals.min()) if len(vals) else None,
        "frac_pairs_with_valley": float((vals > 1.0).mean()) if len(vals) else None,
    }
    # closest pairs (the edges of the adjacency graph)
    Sf = np.where(np.isfinite(S) & (S > 0), S, np.inf)
    order = np.argsort(Sf[iu])
    out["closest_pairs"] = [
        [int(iu[0][o]), int(iu[1][o]), float(Sf[iu][o])] for o in order[:10] if np.isfinite(Sf[iu][o])
    ]
    if ctx.test is not None:
        nc, _ = nearest_center(ctx.test, C)
        conf = np.zeros((K, K))
        for t, p in zip(ctx.test_labels, nc):
            conf[t, p] += 1
        conf = conf / np.maximum(conf.sum(1, keepdims=True), 1)
        out["nearest_center_confusion"] = conf.tolist()
        off = conf.copy()
        np.fill_diagonal(off, 0)
        top = np.dstack(np.unravel_index(np.argsort(off.ravel())[::-1][:10], off.shape))[0]
        out["top_confusions"] = [[int(a), int(b), float(off[a, b])] for a, b in top if off[a, b] > 0]
    return out


@invariant("merge_order", needs=("ref", "labels"), cost="cheap")
def merge_order(ctx, cfg):
    """Single-linkage merge sequence of the class centers (multiscale H0 of the landmarks)."""
    K = ctx.n_classes
    C, r, counts, _ = class_stats(ctx.ref, ctx.ref_labels, K)
    present = np.where(counts > 0)[0]
    if len(present) < 3:
        return {"error": "fewer than 3 classes present"}
    S = _sep_matrix(C, r, counts > 0)[np.ix_(present, present)]
    Z = linkage(squareform(S, checks=False), method="single")
    # decode merges into class-id sets
    clusters = {i: [int(present[i])] for i in range(len(present))}
    merges = []
    for step, (a, b, h, _) in enumerate(Z):
        a, b = int(a), int(b)
        merged = clusters.pop(a) + clusters.pop(b)
        clusters[len(present) + step] = merged
        merges.append({"height": float(h), "members": sorted(merged)})
    heights = np.array([m["height"] for m in merges])
    # H0 count as a function of scale: number of components at each merge height (step curve)
    scales = np.linspace(0, heights.max() * 1.05, 20)
    h0 = [int(len(present) - (heights <= s).sum()) for s in scales]
    # pairwise cophenetic (merge) height between classes: the scale at which i and j join
    coph = np.full((K, K), np.nan)
    for m in merges:
        mem = m["members"]
        for i in mem:
            for j in mem:
                if i != j and np.isnan(coph[i, j]):
                    coph[i, j] = m["height"]
    np.fill_diagonal(coph, 0.0)
    return {
        "merges": merges,
        "first_merge": merges[0]["members"],
        "first_merge_height": float(heights[0]),
        "last_merge_height": float(heights[-1]),
        "h0_scales": scales.tolist(),
        "h0_counts": h0,
        "cophenetic_matrix": np.nan_to_num(coph, nan=-1.0).tolist(),
    }
