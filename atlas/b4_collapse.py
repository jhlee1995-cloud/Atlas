"""
atlas/b4_collapse.py -- batch-4 collapse coordinates (docs/plans/B4_INTEGRATION.md D5, D2, D12). Owner: T2. numpy only
(tests/test_b4_core.py pins that importing it imports neither torch nor scipy nor sklearn). T1's nc_metrics and T3's
nc_suite are deleted: every track computes collapse through this module.

INTERFACE (stable at P1)
  nc_block(X, y, K=10, W=None) -> (rec | None, C (K, d) | None, Sb (d, d) | None)
      Labelled collapse of (X, y). rec: n, k_present, nc1 (= tr(Sigma_W pinv(Sigma_B)) / K', atlas/invariants/
      landmarks.py:71 VERBATIM: sample-mean centring, numpy-default pinv), nc1_trim (Sigma_B inverted on its top K'-1
      eigenpairs above 1e-10 lambda_1), rank_sb, nc1_per_rank (T3 review M6), vci (Xu & Liu 2023 Def. 5.3), rank_st, bt,
      etf_dev, etf_std, norm_cv, center_dist_min / _median / _cv (pairwise class-centre distances, Gram form: T3 review
      B2); with a head W (K, d): head_center_cos (NC3; extract_imagenet._head_center_cos) and nc3_dist (Papyan et al.).
      (None, None, None) when fewer than 2 classes have 2+ rows (random-init nulls under pseudo-labels).
  nc1_around(Xte, yte, C_tr, Sb_tr, K=10) -> float     held-out scatter around the TRAIN means over the train Sigma_B
  nc1_pseudo(X, argmax, K=10) -> rec | None            nc_block with the head's argmax as labels (LABEL-FREE)
  spectrum_block(Xn, K=10, Xh1=None, Xh2=None) -> dict  label-free spectrum of held-out rows: dim, pr, top_eig_frac,
      erank (RankMe), khat, g_raw = lambda_{K-1}/lambda_K, topk_frac = top-(K-1) variance share, alpha (alpha-ReQ, INFO),
      g_cv = the same ratio of CROSS-VALIDATED eigenvalues (eigenvectors from one half, variances on the other,
      averaged both ways; removes the bulk inflation of lambda_K that makes d = 64 and d = 2048 incomparable), lam_cv_top
  label_free_coords(X, argmax, K=10, halves=None) -> {plnc1, log10_plnc1, g_cv, log10_g_cv, topk_frac}   the PRIMARY's
      candidate label-free coordinates (D12); halves default: split_halves(len(X))
  split_halves(n) -> (even rows, odd rows)             T2's split-half rule on test rows 0-4999 (D4)
  center_dists(X, C) -> (N, K);  center_pair_dists(C) -> (K, K)   Gram form, float64, never an (N, K, d) tensor
  ncc_agree(X, C, argmax) -> (agree, ncc)              NC4: nearest-train-centre class vs the head's argmax
  dup_of_penult(X_tap, X_pen, tol=1e-3) -> bool        a tap that is a (float16) copy of the penult (GAP of the last block)
  pre_tap(taps, spatial, dups) -> str | None           D2 (every track): the LAST spatial tap that is not a copy of the
      penult; taps in forward order. Gives layer3.1/3.2/3.3/3.5 (resnet20/32/44/56), pool5 (every VGG), features.17
      (MobileNetV2), stage4 (ShuffleNetV2), stage3 (RepVGG). scripts/b4_extract.py records the same choice in meta.
"""
import math

import numpy as np

from .b4_core import EIG_FLOOR

K_DEFAULT = 10


def _f(v):
    return None if v is None or not np.isfinite(v) else float(v)


def _eig_all(S):
    """All eigenpairs of a symmetric matrix, descending, eigenvalues clipped at 0."""
    w, V = np.linalg.eigh((S + S.T) / 2.0)
    o = np.argsort(w)[::-1]
    return np.clip(w[o], 0.0, None), V[:, o]


def _cov(X):
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(0)
    return Xc.T @ Xc / max(1, len(X) - 1)


def pinv_floor(S, floor=EIG_FLOOR, rank_max=None):
    """Pseudo-inverse on the eigenpairs above floor * lambda_1 (at most rank_max); returns (pinv, rank)."""
    w, V = _eig_all(S)
    keep = w > floor * max(w[0], 1e-300)
    if rank_max is not None:
        keep &= np.arange(len(w)) < rank_max
    r = int(keep.sum())
    return (V[:, :r] / w[:r]) @ V[:, :r].T, r


def class_stats(X, y, K=K_DEFAULT):
    """As atlas/invariants/_util.class_stats: centres (K, d), counts, global SAMPLE mean."""
    X = np.asarray(X, dtype=np.float64)
    C = np.zeros((K, X.shape[1]))
    n = np.zeros(K, dtype=np.int64)
    for c in range(K):
        m = y == c
        n[c] = int(m.sum())
        if n[c]:
            C[c] = X[m].mean(0)
    return C, n, X.mean(0)


def center_pair_dists(C):
    """(K, K) Euclidean distances between the rows of C, Gram form (T3 review B2)."""
    C = np.asarray(C, dtype=np.float64)
    G = C @ C.T
    sq = np.diag(G)
    return np.sqrt(np.clip(sq[:, None] + sq[None, :] - 2.0 * G, 0.0, None))


def center_dists(X, C, rows=4096):
    """(N, K) Euclidean distances from the rows of X to the rows of C, Gram form in row blocks."""
    X = np.asarray(X, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    c2 = np.einsum("kd,kd->k", C, C)
    out = np.empty((len(X), len(C)))
    for i in range(0, len(X), rows):
        x = X[i:i + rows]
        out[i:i + rows] = np.sqrt(np.maximum(np.einsum("nd,nd->n", x, x)[:, None] + c2[None, :] - 2.0 * (x @ C.T), 0.0))
    return out


def nc_block(X, y, K=K_DEFAULT, W=None):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    C, n, mu = class_stats(X, y, K)
    present = n > 1
    Kp = int(present.sum())
    if Kp < 2:
        return None, None, None
    M = C[present] - mu                                                          # centred class means
    Sb = M.T @ M / Kp
    Wd = X - C[y]
    Sw = Wd.T @ Wd / len(X)
    rec = {"n": int(len(X)), "k_present": Kp}
    rec["nc1"] = _f(np.trace(Sw @ np.linalg.pinv(Sb)) / Kp)                    # landmarks.py:71, verbatim
    Sb_p, rb = pinv_floor(Sb, floor=1e-10, rank_max=Kp - 1)
    tr_trim = np.trace(Sw @ Sb_p)
    rec["nc1_trim"] = _f(tr_trim / Kp)
    rec["rank_sb"] = rb
    rec["nc1_per_rank"] = _f(tr_trim / max(1, rb))
    St = (X - mu).T @ (X - mu) / len(X)                                          # total scatter around the sample mean
    St_p, rt = pinv_floor(St)
    rec["vci"] = _f(1.0 - np.trace(St_p @ Sb) / max(1, rb))
    rec["rank_st"] = rt
    rec["bt"] = _f(np.trace(Sb) / np.trace(St)) if np.trace(St) > 0 else None
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    cos = Mn @ Mn.T
    off = cos[~np.eye(Kp, dtype=bool)]
    norms = np.linalg.norm(M, axis=1)
    rec["etf_dev"] = _f(np.abs(off + 1.0 / (Kp - 1)).mean())
    rec["etf_std"] = _f(off.std())
    rec["norm_cv"] = _f(norms.std() / (norms.mean() + 1e-12))
    D = center_pair_dists(C[present])
    dd = D[~np.eye(Kp, dtype=bool)]
    rec["center_dist_min"] = _f(dd.min())
    rec["center_dist_median"] = _f(np.median(dd))
    rec["center_dist_cv"] = _f(dd.std() / (dd.mean() + 1e-12))
    if W is not None and Kp == K and np.asarray(W).shape[0] == K:
        Wc = np.asarray(W, dtype=np.float64)
        mu_c = C - C.mean(0)
        wc = Wc - Wc.mean(0)
        cs = (mu_c * wc).sum(1) / (np.linalg.norm(mu_c, axis=1) * np.linalg.norm(wc, axis=1) + 1e-12)
        rec["head_center_cos"] = _f(cs.mean())
        rec["nc3_dist"] = _f(np.linalg.norm(wc / np.linalg.norm(wc) - mu_c / np.linalg.norm(mu_c)) ** 2)
    return rec, C, Sb


def nc1_around(Xte, yte, C_tr, Sb_tr, K=K_DEFAULT):
    """Held-out within-class scatter around the TRAIN class means, over the train Sigma_B (numpy-default pinv)."""
    Xte = np.asarray(Xte, dtype=np.float64)
    Wd = Xte - C_tr[np.asarray(yte, dtype=np.int64)]
    Sw = Wd.T @ Wd / len(Xte)
    return _f(np.trace(Sw @ np.linalg.pinv(Sb_tr)) / K)


def nc1_pseudo(X, argmax, K=K_DEFAULT):
    """Label-free NC1: the head's own argmax as labels (None when it predicts fewer than 2 classes)."""
    return nc_block(X, np.asarray(argmax, dtype=np.int64), K)[0]


def _ols(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 3 or x.std() == 0:
        return None
    b = float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())
    return y.mean() - b * x.mean(), b


def spectrum_block(Xn, K=K_DEFAULT, Xh1=None, Xh2=None):
    """Label-free spectral coordinates of held-out rows Xn; cross-validated gap from the halves Xh1 / Xh2."""
    w, _ = _eig_all(_cov(Xn))
    d = len(w)
    tot = w.sum() + 1e-300
    out = {"dim": int(d), "pr": _f(tot ** 2 / ((w ** 2).sum() + 1e-300)), "top_eig_frac": _f(w[0] / tot)}
    sv = np.sqrt(w)
    p = sv / (sv.sum() + 1e-300)
    p = p[p > 0]
    out["erank"] = _f(np.exp(-(p * np.log(p)).sum()))
    m = min(50, d - 1)
    if m >= 1:
        out["khat"] = int(np.argmax(w[:m] / np.maximum(w[1:m + 1], 1e-300))) + 1
    else:
        out["khat"] = None
    out["g_raw"] = out["g_cv"] = out["topk_frac"] = out["alpha"] = None
    if d >= K + 1:
        out["g_raw"] = _f(w[K - 2] / max(w[K - 1], 1e-300))
        out["topk_frac"] = _f(w[:K - 1].sum() / tot)
        hi = min(d, 10 * K)
        if hi - K >= 5:
            i = np.arange(K, hi + 1)
            fit = _ols(np.log(i), np.log(np.maximum(w[K - 1:hi], 1e-300)))
            out["alpha"] = _f(-fit[1]) if fit else None
        if Xh1 is not None and Xh2 is not None:
            lam = []
            for a, b in ((Xh1, Xh2), (Xh2, Xh1)):
                _, Va = _eig_all(_cov(a))
                Sb_ = _cov(b)
                lam.append(np.einsum("ij,ij->j", Va[:, :K], Sb_ @ Va[:, :K]))
            lc = np.mean(lam, axis=0)
            out["g_cv"] = _f(lc[K - 2] / max(lc[K - 1], 1e-300))
            out["lam_cv_top"] = [_f(v) for v in lc]
    out["eig_top"] = [_f(v) for v in w[:min(d, max(20, K + 5))]]
    return out


def split_halves(n):
    """T2's split-half rule (D4): even and odd rows."""
    idx = np.arange(int(n))
    return idx[0::2], idx[1::2]


def label_free_coords(X, argmax, K=K_DEFAULT, halves=None):
    """The PRIMARY's label-free candidates (D12): pseudo-label nc1, cross-validated lambda_{K-1}/lambda_K, top-(K-1)
    variance share, with the log10 forms the laws use."""
    X = np.asarray(X, dtype=np.float64)
    h1, h2 = split_halves(len(X)) if halves is None else halves
    pl = nc1_pseudo(X, argmax, K)
    sp = spectrum_block(X, K, X[h1], X[h2])
    plnc1 = pl["nc1"] if pl else None

    def lg(v):
        return math.log10(v) if v is not None and v > 0 else None
    return {"plnc1": plnc1, "log10_plnc1": lg(plnc1), "g_cv": sp.get("g_cv"), "log10_g_cv": lg(sp.get("g_cv")),
            "topk_frac": sp.get("topk_frac"), "g_raw": sp.get("g_raw")}


def ncc_agree(X, C, argmax):
    """NC4 on held-out rows: share of rows whose nearest train class centre equals the head's argmax."""
    ncc = center_dists(X, C).argmin(1)
    return _f((ncc == np.asarray(argmax)).mean()), ncc


def dup_of_penult(X_tap, X_pen, tol=1e-3):
    """True when a tap is a copy of the penult (relative RMS difference below tol; float16 copies differ by ~1e-4)."""
    a, b = np.asarray(X_tap, dtype=np.float64), np.asarray(X_pen, dtype=np.float64)
    if a.shape != b.shape:
        return False
    return bool(np.sqrt(((a - b) ** 2).mean() / ((b ** 2).mean() + 1e-30)) < tol)


def pre_tap(taps, spatial, dups):
    """D2: the last spatial tap that is not a copy of the penult (taps in forward order)."""
    cand = [t for t in taps if t != "penult" and t not in set(dups) and bool((spatial or {}).get(t, False))]
    return cand[-1] if cand else None
