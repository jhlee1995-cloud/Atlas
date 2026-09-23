"""
sensitivity.py -- the tangent structure of the map: where does each corruption push?

  corruption_displacement : for every corrupt split paired with the clean test rows,
      delta_i = f(x_i^c) - f(x_i). Records per (corruption, severity):
        magnitude      ||mean delta|| / mean within-class radius (in reference units)
        top5_frac      energy of mean delta inside the top-5 reference PCs
        class_sub_frac energy of mean delta inside span(centered class means)   [semantic axis]
        coherence      mean cosine(delta_i, mean delta)  [1 = every sample pushed the same way]
        class_consistency  mean pairwise cosine between per-class mean deltas
        norm_ratio     mean ||f(x^c)|| / mean ||f(x)||   (energy: the DEVIATION story)
      plus the unit direction of mean delta in the PCA frame (top-20 coords) for map plots.

Row-6 reading: a corruption with LOW coherence and LOW class_sub_frac is "non-directional"
in this layer (spreads points instead of pushing them), which is exactly what the temporal
axes cannot see. High coherence + high class_sub_frac is directional drift (DRIFT_COH's case).
"""
import numpy as np

from ..context import parse_corrupt_split
from ..registry import invariant
from ._util import class_stats, pca_frame, subsample


@invariant("corruption_displacement", needs=("test", "corrupt"), cost="medium")
def corruption_displacement(ctx, cfg):
    """Paired displacement statistics per corrupt split (see module doc)."""
    K = ctx.n_classes
    C, r, counts, mu = class_stats(ctx.ref, ctx.ref_labels, K)
    within = float(r[counts > 0].mean())
    _, w, V = pca_frame(subsample(ctx.ref, int(cfg.get("n_ref", 10000)), ctx.rng))
    Vk = V[:, : int(cfg.get("top_k", 5))]
    M = (C[counts > 0] - mu)
    # orthonormal basis of the class-mean subspace. The centered means have rank K-1, so a QR of
    # M.T would add one arbitrary column; keep only the numerically non-zero singular directions.
    U, sv, _ = np.linalg.svd(M.T, full_matrices=False)
    Q = U[:, : int((sv > sv[0] * 1e-8).sum())]
    out = {"within_radius": within, "splits": {}}
    for s, Xc in ctx.corrupt.items():
        n = min(len(Xc), len(ctx.test))
        delta = (Xc[:n] - ctx.test[:n]).astype(np.float64)
        md = delta.mean(0)
        nm = np.linalg.norm(md) + 1e-12
        u = md / nm
        dn = delta / (np.linalg.norm(delta, axis=1, keepdims=True) + 1e-12)
        coh = float((dn @ u).mean())
        # per-class mean deltas
        y = ctx.test_labels[:n]
        pcs = []
        for c in range(K):
            m = y == c
            if m.sum() >= 5:
                v = delta[m].mean(0)
                pcs.append(v / (np.linalg.norm(v) + 1e-12))
        if len(pcs) >= 2:
            P = np.array(pcs)
            cosm = P @ P.T
            cc = float(cosm[~np.eye(len(P), dtype=bool)].mean())
        else:
            cc = None
        corr, sev = parse_corrupt_split(s)
        out["splits"][s] = {
            "corruption": corr, "severity": sev, "n": int(n),
            "magnitude": float(nm / (within + 1e-12)),
            "per_sample_magnitude": float(np.linalg.norm(delta, axis=1).mean() / (within + 1e-12)),
            "top5_frac": float(np.linalg.norm(Vk.T @ md) ** 2 / nm ** 2),
            "class_sub_frac": float(np.linalg.norm(Q.T @ md) ** 2 / nm ** 2),
            "coherence": coh,
            "class_consistency": cc,
            "norm_ratio": float(np.linalg.norm(Xc[:n], axis=1).mean() / (np.linalg.norm(ctx.test[:n], axis=1).mean() + 1e-12)),
            "direction_pca20": (V[:, :20].T @ u).tolist(),
        }
    return out
