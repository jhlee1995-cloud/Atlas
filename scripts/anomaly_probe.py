#!/usr/bin/env python3
"""
anomaly_probe.py -- ANOMALY_H1 missing-axis probes AX-1..AX-4 (docs/plans/ANOMALY_H1.md). Stage B, CPU, numpy only.

Reads CIFAR atlas dumps only: acts/<layer>/<split>.npy, labels/<split>.npy and preds/<split>.npz (argmax, maxprob). It
never reads logits, factors, the panel or ood__svhn, never reads a B1 / ImageNet / ViT dump (refused from the path before
any file is opened), and reads the confirmation corruptions (holdout block of experiments/queue/atlas_v1_resnet20_s1.yaml)
only on the two confirmation dumps (atlas_v1_resnet56_s1, _s2). It writes one NEW directory
results/anomaly_probe_<tag>/probe.json and refuses to write into an existing non-empty directory (append-only).
It measures; it applies no threshold (scripts/anomaly_eval.js decides), except the AX-3 batch-64 summary, which is INFO.
It changes no existing file and imports nothing from atlas/, so no existing behaviour changes.

  python scripts/anomaly_probe.py --tag resnet56_s1 --dumps results/atlas_v1_resnet56_s1/dump
  python scripts/anomaly_probe.py --selftest [--selftest-out results/instrument_check_anomaly/selftest.json]

Rows (every dump; CIFAR-10 test rows 0..4999 in order, corrupt splits paired with test rows 0..1999):
  calibration half A = test rows 2000-3499 (held-out clean mean, covariance, radius median)
  clean half B       = test rows 3500-4999 (clean batches, single-class stress batches, AX-4 clean samples)
  corrupt            = rows 0-1999 of each corrupt split (images disjoint from A and B)
Definitions (see ANOMALY_H1.md for the hypotheses and thresholds):
  AX-1 at the pre-collapse tap (r20 layer3.1, r56 layer3.5) and at penult: c_k = train-reference class means, c_bar their
       unweighted mean, Q = orthonormal basis of span{c_k - c_bar} (SVD, singular values > 1e-8 s_max; rank 9), P_perp =
       I - Q Q^T. T_perp(B) = n zbar^T Lambda^-1 zbar, zbar = batch mean of U_r^T (x - m_A), (U_r, Lambda) = top r <= 20
       eigenpairs (> 1e-8 lambda_1) of the half-A covariance of P_perp (x - m_A). T_par: the same on the Q coordinates
       (whitened by their half-A covariance). T_full (penult): top 20 eigenpairs of the half-A covariance, no projection.
       V_cov (INFO, H-W1 arm, pre tap): Stein divergence tr(S) - log det(S) - r of the batch covariance S of the
       half-A-whitened top-20 full-space coordinates. Batch 64, 500 seeded batches per condition; single-class stress
       batches: 50 per class from half B by true label (labels build the stream only, never the statistic).
       tau95 = 95th percentile of the statistic over the 500 clean half-B batches.
  AX-2 (scored as AX-2a, the tap profile, and AX-2b, the router) at 6 taps (stem, layer1.0, end of stage 1, end of
       stage 2, pre-collapse, penult): Hotelling T^2 of the batch mean
       against m_A in the top r <= 10 half-A principal components (eigenvalues <= 1e-8 lambda_1 dropped); per tap AUC at
       batch 16 (500 batches). Router at the pre tap: whitened (top 20 of half A) unit mean shifts of the 8 family
       corruptions at s3 from corrupt rows 0-999 are the templates; a batch of 64 from rows 1000-1999 goes to the family
       of the template with the highest cosine. Families N = {gaussian, shot}, B = {defocus, motion, fog, contrast},
       L = {brightness}, P = {pixelate}.
  AX-3 at penult: r10(x) = distance to the 10th nearest train-reference row (all reference rows; self excluded for the
       reference itself); h(B) = [median_B log r10 - median_A log r10] / (q95 - q50 of the reference self log r10).
       loss(B) = 100 x (clean accuracy - corrupt accuracy) of the same paired images (argmax vs label). e(B) = mean_B ||x||
       / mean_A ||x|| (the norm_ratio of corruption_displacement, batch version); e2 the same with ||x||^2 (INFO).
       Scored batch 256 (200 seeded batches per split); batch 64 summarised as INFO.
  AX-4 per sample at penult: e_perp(x) = ||P_perp (x - c_a)||^2 / ||x - c_a||^2, a = argmax (label-free). Baselines: d1 =
       nearest-center distance; d1 / r_a (RMS within-class radius of the predicted class); dens2 = |log r10(x) - median_A
       log r10| at the end of stage 2 (r20 layer2.2, r56 layer2.8). AUROC corrupt (positive, all rows of the split) vs
       clean half B; every score oriented so that larger = more corrupt-like (no direction-free AUCs).
Determinism: every batch set is drawn from numpy default_rng([SEED, crc32(<purpose>|<split>|<batch>)]), independent of
the dump, so every dump sees the same image rows; all arithmetic is float64.
"""
import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import tempfile
import time
import zlib

import numpy as np

SCHEMA = "anomaly_probe/1"
SEED = 20260923
DISC = ("gaussian_noise", "shot_noise", "defocus_blur", "motion_blur", "snow", "fog", "brightness", "contrast",
        "pixelate", "jpeg_compression")
HOLDOUT = ("impulse_noise", "glass_blur", "zoom_blur", "frost", "elastic_transform")
SEVS = (1, 3, 5)
HOLDOUT_SEVS = (3, 5)
CONFIRMATION_DUMPS = ("atlas_v1_resnet56_s1", "atlas_v1_resnet56_s2")
NULL_DUMPS = ("atlas_v1_resnet20_rand", "atlas_v1_resnet56_rand")
TAPS = {
    "cifar10_resnet20": {"stem": "stem", "l10": "layer1.0", "s1end": "layer1.2", "s2end": "layer2.2", "pre": "layer3.1",
                         "penult": "penult"},
    "cifar10_resnet56": {"stem": "stem", "l10": "layer1.0", "s1end": "layer1.8", "s2end": "layer2.8", "pre": "layer3.5",
                         "penult": "penult"},
}
TAPKEYS = ("stem", "l10", "s1end", "s2end", "pre", "penult")
FORBIDDEN = re.compile(r"(margin_b1_|imagenet|vitb16|vit_b_16|deitb|deit_)")
N_TEST, N_CLASSES = 5000, 10
ROWS_A, ROWS_B = (2000, 3500), (3500, 5000)
EIG_FLOOR = 1e-8
AX1 = {"batch": 64, "n_batches": 500, "single_per_class": 50, "r_perp": 20, "r_full": 20}
AX2 = {"batch": 16, "n_batches": 500, "r": 10, "router_batch": 64, "router_batches": 500, "router_r": 20,
       "template_rows": (0, 1000), "route_rows": (1000, 2000)}
FAMILIES = {"N": ("gaussian_noise", "shot_noise"), "B": ("defocus_blur", "motion_blur", "fog", "contrast"),
            "L": ("brightness",), "P": ("pixelate",)}
AX3 = {"batch": 256, "n_batches": 200, "k": 10, "info_batch": 64, "info_hold": 0.25, "info_loss": 10.0}
AX4 = {"k": 10}
AX1_HOLDOUT = ("corrupt__zoom_blur__s3", "corrupt__glass_blur__s3")
AX4_HOLDOUT = ("corrupt__zoom_blur__s3",)
CONSTANTS = {"seed": SEED, "rows_A": ROWS_A, "rows_B": ROWS_B, "eig_floor": EIG_FLOOR, "AX1": AX1, "AX2": AX2, "AX3": AX3,
             "AX4": AX4, "families": FAMILIES, "taps": TAPS, "confirmation_dumps": CONFIRMATION_DUMPS,
             "null_dumps": NULL_DUMPS, "holdout_read_on_confirmation": [f"{c} s{s}" for c in HOLDOUT for s in HOLDOUT_SEVS]}


def csplit(c, s):
    return f"corrupt__{c}__s{int(s)}"


def parse_csplit(split):
    if not split.startswith("corrupt__"):
        return None
    _, c, s = split.split("__")
    return c, int(s[1:])


DISC_SPLITS = tuple(csplit(c, s) for c in DISC for s in SEVS)
HOLDOUT_SPLITS = tuple(csplit(c, s) for c in HOLDOUT for s in HOLDOUT_SEVS)


# ---------------------------------------------------------------------------------------------------------------------
# numerics
# ---------------------------------------------------------------------------------------------------------------------
def rank_avg(x):
    """1-based ranks with ties averaged."""
    x = np.asarray(x, dtype=np.float64).ravel()
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    _, first, counts = np.unique(xs, return_index=True, return_counts=True)
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.repeat(first + (counts - 1) / 2.0 + 1.0, counts)
    return r


def auc(pos, neg):
    """P(pos > neg) + 0.5 P(tie) (Mann-Whitney); None if a side is empty or non-finite."""
    pos = np.asarray(pos, dtype=np.float64).ravel()
    neg = np.asarray(neg, dtype=np.float64).ravel()
    if len(pos) == 0 or len(neg) == 0 or not (np.isfinite(pos).all() and np.isfinite(neg).all()):
        return None
    r = rank_avg(np.concatenate([pos, neg]))
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def spearman(x, y):
    rx, ry = rank_avg(x), rank_avg(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return None if d <= 0 else float((rx * ry).sum() / d)


def knn_kth(fit, query, k, exclude_self=False, chunk=1024):
    """Distance from each query row to its k-th nearest row of `fit` (exact, float64). exclude_self: the queries are
    the fit rows themselves, so the zero self-distance is skipped (the (k+1)-th smallest), as knn_density does."""
    fit = np.asarray(fit, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    kk = k if exclude_self else k - 1
    if kk >= len(fit):
        raise ValueError(f"knn_kth: k={k} needs more than {len(fit)} fit rows")
    f2 = np.einsum("ij,ij->i", fit, fit)
    out = np.empty(len(query), dtype=np.float64)
    for i in range(0, len(query), chunk):
        q = query[i:i + chunk]
        d2 = np.einsum("ij,ij->i", q, q)[:, None] + f2[None, :] - 2.0 * (q @ fit.T)
        np.maximum(d2, 0.0, out=d2)
        out[i:i + chunk] = np.sqrt(np.partition(d2, kk, axis=1)[:, kk])
    return out


def _eig_desc(C, r):
    w, V = np.linalg.eigh(C)
    o = np.argsort(w)[::-1]
    w, V = w[o], V[:, o]
    if len(w) == 0 or not w[0] > 0:
        return w[:0], V[:, :0]
    k = int(min(r, np.sum(w > EIG_FLOOR * w[0])))
    return w[:k], V[:, :k]


def pca_frame(X, r):
    """(mean, top-r eigenvalues > 1e-8 lambda_1, eigenvectors) of the covariance of X."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(0)
    Xc = X - mu
    w, V = _eig_desc(Xc.T @ Xc / max(1, len(X) - 1), r)
    return mu, w, V


def class_means(ref, y, K=N_CLASSES):
    C = np.zeros((K, ref.shape[1]))
    for k in range(K):
        m = y == k
        if not m.any():
            raise ValueError(f"class {k} absent from the reference")
        C[k] = ref[m].mean(0)
    return C


def class_radii(ref, y, C):
    return np.array([np.sqrt(((ref[y == k] - C[k]) ** 2).sum(1).mean()) for k in range(len(C))])


def class_span(C):
    """Orthonormal basis (D, q) of span{c_k - c_bar}, c_bar the unweighted mean of the class means (q = K - 1 = 9)."""
    M = (C - C.mean(0)).T
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    return U[:, : int((s > s[0] * 1e-8).sum())]


def perp_frame(Xa, Q, r):
    mu = Xa.mean(0)
    Xc = Xa - mu
    Xp = Xc - (Xc @ Q) @ Q.T
    w, V = _eig_desc(Xp.T @ Xp / max(1, len(Xa) - 1), r)
    return mu, w, V


def par_frame(Xa, Q):
    mu = Xa.mean(0)
    Zq = (Xa - mu) @ Q
    w, W = _eig_desc(Zq.T @ Zq / max(1, len(Xa) - 1), Q.shape[1])
    return mu, w, Q @ W                       # whitening basis inside span(Q)


def whiten(X, mu, w, V):
    return (np.asarray(X, dtype=np.float64) - mu) @ V / np.sqrt(w)


def batch_idx(key, n_rows, batch, nb, offset=0):
    """(nb, batch) row indices, each batch without replacement; seeded by the key only (dump-independent)."""
    if n_rows < batch:
        return None
    rng = np.random.default_rng([SEED, zlib.crc32(key.encode("utf-8"))])
    return np.argsort(rng.random((nb, n_rows)), axis=1)[:, :batch] + offset


def t2(Z, idx):
    """Hotelling T^2 of each batch mean in whitened coordinates: n * ||mean z||^2."""
    return idx.shape[1] * (Z[idx].mean(axis=1) ** 2).sum(axis=1)


def stein(Z, idx):
    B = Z[idx]
    Bc = B - B.mean(axis=1, keepdims=True)
    S = np.einsum("bij,bik->bjk", Bc, Bc) / (idx.shape[1] - 1)
    sign, logdet = np.linalg.slogdet(S)
    v = np.trace(S, axis1=1, axis2=2) - logdet - S.shape[1]
    v[sign <= 0] = np.inf
    return v


def e_perp(X, C, Q, am):
    V = np.asarray(X, dtype=np.float64) - C[am]
    vv = (V * V).sum(1)
    par = V @ Q
    out = np.full(len(X), np.nan)
    ok = vv > 0
    out[ok] = (vv[ok] - (par[ok] * par[ok]).sum(1)) / vv[ok]
    return out


def center_dists(X, C):
    X = np.asarray(X, dtype=np.float64)
    d2 = (X * X).sum(1)[:, None] + (C * C).sum(1)[None, :] - 2.0 * X @ C.T
    return np.sqrt(np.maximum(d2, 0.0))


# ---------------------------------------------------------------------------------------------------------------------
# dump reader (role-enforced)
# ---------------------------------------------------------------------------------------------------------------------
class Reader:
    def __init__(self, root, role):
        self.root, self.role = root, role
        self._acts, self.read = {}, set()

    def allowed(self, split):
        if split in ("ref", "test", "ood__cifar100"):
            return True
        cs = parse_csplit(split)
        if cs is None:
            return False                      # panel, ood__svhn and anything else are never read
        if cs[0] in DISC:
            return True
        return cs[0] in HOLDOUT and self.role == "confirmation"

    def _check(self, split):
        if not self.allowed(split):
            raise PermissionError(f"split {split} is not readable in role {self.role}")

    def has(self, layer, split):
        return self.allowed(split) and os.path.isfile(os.path.join(self.root, "acts", layer, f"{split}.npy"))

    def acts(self, layer, split):
        self._check(split)
        key = (layer, split)
        if key not in self._acts:
            self._acts[key] = np.load(os.path.join(self.root, "acts", layer, f"{split}.npy")).astype(np.float64)
            self.read.add(split)
        return self._acts[key]

    def labels(self, split):
        self._check(split)
        self.read.add(split)
        return np.load(os.path.join(self.root, "labels", f"{split}.npy")).astype(np.int64)

    def argmax(self, split):
        self._check(split)
        self.read.add(split)
        with np.load(os.path.join(self.root, "preds", f"{split}.npz")) as z:
            return np.asarray(z["argmax"]).astype(np.int64)    # only argmax (maxprob is not needed; no logits exist)

    def drop(self):
        self._acts.clear()


def _r(a, d=6):
    return [None if not np.isfinite(v) else round(float(v), d) for v in np.asarray(a, dtype=np.float64).ravel()]


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ---------------------------------------------------------------------------------------------------------------------
# AX-1 .. AX-4
# ---------------------------------------------------------------------------------------------------------------------
def ax1(rd, taps, disc, hold):
    n, nb = AX1["batch"], AX1["n_batches"]
    y = rd.labels("test")
    idx_clean = batch_idx(f"ax1|clean|{n}", ROWS_B[1] - ROWS_B[0], n, nb, ROWS_B[0])
    idx_single = {}
    for k in range(N_CLASSES):
        rows = np.flatnonzero(y[ROWS_B[0]:ROWS_B[1]] == k) + ROWS_B[0]
        b = batch_idx(f"ax1|single|{k}|{n}", len(rows), n, AX1["single_per_class"])
        idx_single[k] = None if b is None else rows[b]
    splits = list(disc) + [s for s in hold if s in AX1_HOLDOUT]
    idx_split = {s: batch_idx(f"ax1|{s}|{n}", len(rd.acts(taps["penult"], s)), n, nb) for s in splits}
    out = {"tap_pre": taps["pre"], "batch": n, "n_batches": nb, "single_per_class": AX1["single_per_class"],
           "q_rank": {}, "r_perp": {}, "tau95": {}, "single_class": {}, "single_vs_mixed_auc": {}}
    frames = {}
    for tk in ("pre", "penult"):
        L = taps[tk]
        ref, yr = rd.acts(L, "ref"), rd.labels("ref")
        Q = class_span(class_means(ref, yr))
        Xa = rd.acts(L, "test")[ROWS_A[0]:ROWS_A[1]]
        out["q_rank"][tk], frames[f"perp_{tk}"] = Q.shape[1], perp_frame(Xa, Q, AX1["r_perp"])
        out["r_perp"][tk] = len(frames[f"perp_{tk}"][1])
        if tk == "pre":
            frames["par_pre"] = par_frame(Xa, Q)
            frames["cov_pre"] = pca_frame(Xa, AX1["r_full"])
            out["r_par"], out["r_cov_pre"] = len(frames["par_pre"][1]), len(frames["cov_pre"][1])
        else:
            frames["full_penult"] = pca_frame(Xa, AX1["r_full"])
            out["r_full_penult"] = len(frames["full_penult"][1])
    for name, fr in frames.items():
        L = taps["pre"] if name.endswith("_pre") else taps["penult"]
        stat = stein if name == "cov_pre" else t2
        Zt = whiten(rd.acts(L, "test"), *fr)
        clean = stat(Zt, idx_clean)
        out["tau95"][name] = float(np.quantile(clean, 0.95))
        out[f"auc_{name}"] = {s: auc(stat(whiten(rd.acts(L, s), *fr), idx_split[s]), clean)
                              for s in splits if idx_split[s] is not None}
        single = {k: stat(Zt, b) for k, b in idx_single.items() if b is not None}
        fpr = [float((single[k] > out["tau95"][name]).mean()) if k in single else None for k in range(N_CLASSES)]
        vals = [v for v in fpr if v is not None]
        out["single_class"][name] = {"fpr": fpr, "fpr_mean": float(np.mean(vals)) if vals else None}
        if name in ("par_pre", "perp_pre", "full_penult") and single:
            out["single_vs_mixed_auc"][name] = auc(np.concatenate(list(single.values())), clean)
    return out


def ax2(rd, taps, disc, hold):
    n, nb = AX2["batch"], AX2["n_batches"]
    idx_clean = batch_idx(f"ax2|clean|{n}", ROWS_B[1] - ROWS_B[0], n, nb, ROWS_B[0])
    out = {"taps": dict(taps), "batch": n, "n_batches": nb, "r": {}, "auc": {s: {} for s in disc}}
    for tk in TAPKEYS:
        L = taps[tk]
        test = rd.acts(L, "test")
        fr = pca_frame(test[ROWS_A[0]:ROWS_A[1]], AX2["r"])
        out["r"][tk] = len(fr[1])
        clean = t2(whiten(test, *fr), idx_clean)
        for s in disc:
            X = rd.acts(L, s)
            b = batch_idx(f"ax2|{s}|{n}", len(X), n, nb)
            out["auc"][s][tk] = None if b is None else auc(t2(whiten(X, *fr), b), clean)
        if tk not in ("pre", "penult"):
            rd.drop()
    # router at the pre tap
    L = taps["pre"]
    fr = pca_frame(rd.acts(L, "test")[ROWS_A[0]:ROWS_A[1]], AX2["router_r"])
    t0, t1 = AX2["template_rows"]
    r0, r1 = AX2["route_rows"]
    members = [(fam, c) for fam, cs in FAMILIES.items() for c in cs]
    tmpl, tfam = [], []
    for fam, c in members:
        s = csplit(c, 3)
        if s in disc:
            v = whiten(rd.acts(L, s)[t0:t1], *fr).mean(0)
            tmpl.append(v / (np.linalg.norm(v) + 1e-300))
            tfam.append(fam)
    router = {"batch": AX2["router_batch"], "n_batches": AX2["router_batches"], "r": len(fr[1]),
              "templates": [f"{f}:{c}" for f, c in members if csplit(c, 3) in disc], "by_split": {},
              "accuracy_by_family": {}, "holdout": {} if hold else None}
    if tmpl:
        Tm = np.stack(tmpl)

        def route(s):
            X = rd.acts(L, s)[r0:r1]
            b = batch_idx(f"ax2r|{s}|{AX2['router_batch']}", len(X), AX2["router_batch"], AX2["router_batches"])
            if b is None:
                return None
            m = whiten(X, *fr)[b].mean(axis=1)
            m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-300
            pick = np.argmax(m @ Tm.T, axis=1)
            fams = np.array(tfam)[pick]
            return {f: float((fams == f).mean()) for f in FAMILIES}

        for fam, c in members:
            for sv in (3, 5):
                s = csplit(c, sv)
                if s in disc:
                    router["by_split"][s] = {"true": fam, "assigned": route(s)}
        for fam in FAMILIES:
            v = [d["assigned"][fam] for d in router["by_split"].values() if d["true"] == fam and d["assigned"]]
            router["accuracy_by_family"][fam] = float(np.mean(v)) if v else None
        for s in hold:
            router["holdout"][s] = {"assigned": route(s)}
    out["router"] = router
    return out


def ax3(rd, taps, disc, hold):
    L = taps["penult"]
    k, n, nb = AX3["k"], AX3["batch"], AX3["n_batches"]
    ref = rd.acts(L, "ref")
    lself = np.log(knn_kth(ref, ref, k, exclude_self=True) + 1e-12)
    q50, q95 = (float(v) for v in np.quantile(lself, [0.5, 0.95]))
    scale = q95 - q50
    test, y = rd.acts(L, "test"), rd.labels("test")
    ltest = np.log(knn_kth(ref, test[ROWS_A[0]:ROWS_B[1]], k) + 1e-12)          # rows 2000..4999
    med_a = float(np.median(ltest[: ROWS_A[1] - ROWS_A[0]]))
    xa = test[ROWS_A[0]:ROWS_A[1]]
    na, na2 = float(np.linalg.norm(xa, axis=1).mean()), float((xa * xa).sum(1).mean())
    clean_ok = rd.argmax("test") == y
    out = {"tap": L, "k": k, "batch": n, "n_batches": nb, "ref_self_q50": q50, "ref_self_q95": q95, "scale": scale,
           "median_A": med_a, "splits": {}, "ood": {}, "info_b64": {}}

    def grade(lr, b):
        return (np.median(lr[b], axis=1) - med_a) / scale

    b64 = {"h": [], "loss": []}
    for s in disc:
        X = rd.acts(L, s)
        ys = rd.labels(s)
        m = len(X)
        if not np.array_equal(ys, y[:m]):
            raise ValueError(f"pairing broken: labels of {s} != test[:{m}]")
        lr = np.log(knn_kth(ref, X, k) + 1e-12)
        ok = rd.argmax(s) == ys
        nrm, nrm2 = np.linalg.norm(X, axis=1), (X * X).sum(1)
        b = batch_idx(f"ax3|{s}|{n}", m, n, nb)
        out["splits"][s] = {"h": _r(grade(lr, b)), "loss": _r(100.0 * (clean_ok[b].mean(1) - ok[b].mean(1))),
                            "e": _r(nrm[b].mean(1) / na), "e2": _r(nrm2[b].mean(1) / na2)}
        b = batch_idx(f"ax3|{s}|{AX3['info_batch']}", m, AX3["info_batch"], nb)
        b64["h"].append(grade(lr, b))
        b64["loss"].append(100.0 * (clean_ok[b].mean(1) - ok[b].mean(1)))
    bc = batch_idx(f"ax3|clean|{n}", ROWS_B[1] - ROWS_B[0], n, nb, ROWS_B[0] - ROWS_A[0])
    xb = test[ROWS_A[0]:ROWS_B[1]]
    out["clean_B"] = {"h": _r(grade(ltest, bc)), "e": _r(np.linalg.norm(xb, axis=1)[bc].mean(1) / na)}
    if rd.has(L, "ood__cifar100"):
        X = rd.acts(L, "ood__cifar100")
        lr = np.log(knn_kth(ref, X, k) + 1e-12)
        out["ood"]["ood__cifar100"] = {"h": _r(grade(lr, batch_idx(f"ax3|ood__cifar100|{n}", len(X), n, nb)))}
    if b64["h"]:
        h, lo = np.concatenate(b64["h"]), np.concatenate(b64["loss"])
        band = h <= AX3["info_hold"]
        out["info_b64"] = {"spearman_pooled": spearman(h, lo), "n_in_band": int(band.sum()),
                           "frac_loss_gt10_in_band": float((lo[band] > AX3["info_loss"]).mean()) if band.any() else None}
    return out


def ax4(rd, taps, disc, hold):
    L, L2, k = taps["penult"], taps["s2end"], AX4["k"]
    ref, yr = rd.acts(L, "ref"), rd.labels("ref")
    C = class_means(ref, yr)
    Q, rad = class_span(C), class_radii(ref, yr, C)
    ref2, test2 = rd.acts(L2, "ref"), rd.acts(L2, "test")
    med_a = float(np.median(np.log(knn_kth(ref2, test2[ROWS_A[0]:ROWS_A[1]], k) + 1e-12)))

    def scores(X, am, X2):
        d1 = center_dists(X, C).min(1)
        return {"e_perp": e_perp(X, C, Q, am), "d1": d1, "d1_over_r": d1 / rad[am],
                "dens2": np.abs(np.log(knn_kth(ref2, X2, k) + 1e-12) - med_a)}

    test = rd.acts(L, "test")
    clean = scores(test[ROWS_B[0]:ROWS_B[1]], rd.argmax("test")[ROWS_B[0]:ROWS_B[1]], test2[ROWS_B[0]:ROWS_B[1]])
    out = {"tap": L, "dens_tap": L2, "clean_rows": ROWS_B, "median_A_dens2": med_a, "auroc": {}}
    for s in list(disc) + [x for x in hold if x in AX4_HOLDOUT]:
        cs = scores(rd.acts(L, s), rd.argmax(s), rd.acts(L2, s))
        out["auroc"][s] = {**{name: auc(cs[name], clean[name]) for name in cs},
                           "n_pos": int(len(cs["d1"])), "n_neg": int(len(clean["d1"]))}
    return out


# ---------------------------------------------------------------------------------------------------------------------
# one dump
# ---------------------------------------------------------------------------------------------------------------------
def dump_root(path):
    p = os.path.normpath(path)
    if os.path.basename(p) != "dump" and os.path.isdir(os.path.join(p, "dump")):
        p = os.path.join(p, "dump")
    return p


def dump_name(root):
    p = os.path.normpath(root)
    return os.path.basename(os.path.dirname(p)) if os.path.basename(p) == "dump" else os.path.basename(p)


def process_dump(path, role_override=None, allow_synthetic=False):
    """Returns (name, record). Never raises for a bad dump: the record says REFUSED / ERROR / PARTIAL / OK."""
    t_start = time.time()
    root = dump_root(path)
    name = dump_name(root)
    rec = {"status": "OK", "error": None, "dump": path, "realpath": os.path.realpath(root)}
    low = os.path.abspath(root).replace("\\", "/").lower() + "/" + os.path.realpath(root).replace("\\", "/").lower()
    if FORBIDDEN.search(low):                                          # before any file is opened
        rec.update(status="REFUSED", error="B1 / ImageNet / ViT path: never read by this probe")
        return name, rec
    if os.path.isdir(os.path.join(root, "logits")):
        rec.update(status="REFUSED", error="dump stores logits: not a CIFAR atlas dump")
        return name, rec
    try:
        with open(os.path.join(root, "meta.json")) as f:
            meta = json.load(f)
    except (OSError, ValueError) as e:
        rec.update(status="ERROR", error=f"meta.json unreadable: {e}")
        return name, rec
    arch, src = meta.get("arch"), meta.get("source")
    rec["meta"] = {k: meta.get(k) for k in ("arch", "seed_tag", "weights", "git_commit", "created", "exp_id", "n_test",
                                             "source", "dtype", "pooling", "n_classes")}
    problems = []
    if arch not in TAPS:
        problems.append(f"arch {arch} is not a CIFAR ResNet")
    if int(meta.get("n_classes", 10)) != N_CLASSES:
        problems.append(f"n_classes {meta.get('n_classes')}")
    if src != "real" and not (allow_synthetic and src == "synthetic"):
        problems.append(f"source {src}")
    if int(meta.get("n_test", 0)) < N_TEST:
        problems.append(f"n_test {meta.get('n_test')} < {N_TEST}")
    if role_override is not None and src != "synthetic":
        problems.append("a role override is allowed on synthetic dumps only")
    if not problems:
        missing_taps = [t for t in TAPS[arch].values() if t not in meta.get("layers", [])]
        if missing_taps:
            problems.append(f"taps missing: {missing_taps}")
    if problems:
        rec.update(status="REFUSED", error="; ".join(problems))
        return name, rec
    role = role_override or ("confirmation" if name in CONFIRMATION_DUMPS else "null" if name in NULL_DUMPS else "discovery")
    taps = TAPS[arch]
    rd = Reader(root, role)
    disc = [s for s in DISC_SPLITS if rd.has(taps["penult"], s)]
    hold = [s for s in HOLDOUT_SPLITS if rd.has(taps["penult"], s)] if role == "confirmation" else []
    rec.update(role=role, taps=dict(taps), discovery_splits=disc, holdout_splits=hold,
               missing_discovery_splits=[s for s in DISC_SPLITS if s not in disc],
               missing_holdout_splits=[s for s in HOLDOUT_SPLITS if s not in hold] if role == "confirmation" else None)
    timing, errors = {}, []
    for ax, fn in (("AX2", ax2), ("AX1", ax1), ("AX3", ax3), ("AX4", ax4)):
        t0 = time.time()
        try:
            rec[ax] = fn(rd, taps, disc, hold)
        except Exception as e:                                         # one AX failing never loses the others
            rec[ax] = {"error": f"{type(e).__name__}: {e}"}
            errors.append(ax)
        timing[ax] = round(time.time() - t0, 2)
        rd.drop()
    rec["splits_read"] = sorted(rd.read)
    rec["timing_s"] = {**timing, "total": round(time.time() - t_start, 2)}
    rec["status"] = "OK" if not errors else ("ERROR" if len(errors) == 4 else "PARTIAL")
    rec["error"] = None if not errors else f"failed: {errors}"
    return name, rec


# ---------------------------------------------------------------------------------------------------------------------
# provenance and output
# ---------------------------------------------------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def code_sha256():
    with open(os.path.abspath(__file__), "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def repo_commit(repo=REPO_ROOT):
    """HEAD commit read from .git files (no git process runs while pod_atlas.sh runs)."""
    g = os.path.join(repo, ".git")
    try:
        if os.path.isfile(g):
            with open(g) as f:
                g = os.path.join(repo, f.read().split("gitdir:", 1)[1].strip())
        with open(os.path.join(g, "HEAD")) as f:
            head = f.read().strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:]
        if os.path.isfile(os.path.join(g, ref)):
            with open(os.path.join(g, ref)) as f:
                return f.read().strip()
        with open(os.path.join(g, "packed-refs")) as f:
            for line in f:
                if line.strip().endswith(" " + ref):
                    return line.split()[0]
    except (OSError, IndexError):
        pass
    return None


def out_dir(out_root, tag):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", tag or ""):
        raise ValueError(f"bad tag {tag!r}")
    return os.path.join(out_root, f"anomaly_probe_{tag}")


def refuse_existing(d):
    if os.path.exists(d) and (not os.path.isdir(d) or os.listdir(d)):
        raise FileExistsError(f"{d} exists and is not empty: results are append-only (use a new --tag)")


def write_probe(out_root, tag, payload):
    d = out_dir(out_root, tag)
    refuse_existing(d)
    os.makedirs(d, exist_ok=True)
    tmp, dst = os.path.join(d, "probe.json.tmp"), os.path.join(d, "probe.json")
    with open(tmp, "w") as f:
        json.dump(_clean(payload), f, indent=1, sort_keys=True, allow_nan=False)
    os.replace(tmp, dst)
    return dst


def env_record():
    return {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
            "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}


def run(tag, dumps, out_root="results", role_override=None, allow_synthetic=False):
    d = out_dir(out_root, tag)
    refuse_existing(d)                                                 # fail fast, before any compute
    payload = {"schema": SCHEMA, "tag": tag, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "code": {"path": "scripts/anomaly_probe.py", "sha256": code_sha256(), "repo_commit": repo_commit()},
               "env": env_record(), "constants": CONSTANTS, "dumps": {}}
    for p in dumps:
        name, rec = process_dump(p, role_override=role_override, allow_synthetic=allow_synthetic)
        if name in payload["dumps"]:
            rec = {"status": "ERROR", "error": f"duplicate dump name {name}", "dump": p}
            name = f"{name}#dup{len(payload['dumps'])}"
        payload["dumps"][name] = rec
        print(f"[anomaly_probe] {name}: {rec['status']}{' (' + rec['error'] + ')' if rec.get('error') else ''} "
              f"{rec.get('timing_s', {}).get('total', '')}s", flush=True)
    payload["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return write_probe(out_root, tag, payload), payload


# ---------------------------------------------------------------------------------------------------------------------
# synthetic dump and self-test (known answers)
# ---------------------------------------------------------------------------------------------------------------------
SYN_LAYERS = ("stem", "layer1.0", "layer1.2", "layer2.2", "layer3.1", "penult")      # the resnet20 tap names


def make_synth_dump(root, n_ref=10000, seed=0):
    """Synthetic resnet20-named dump, D = 16: class means 3 e_k (k < 9) and 0, isotropic in-span noise, complement
    dims 9..15 with SDs 1.6..1.0. Shifts per split and tap (unit vectors e_j) define the known answers."""
    rng = np.random.default_rng(seed)
    D = 16
    sd = np.ones(D)
    sd[9:] = [1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0]
    C = np.zeros((N_CLASSES, D))
    for k in range(9):
        C[k, k] = 3.0
    e = np.eye(D)

    def draw(n):
        y = rng.integers(0, N_CLASSES, n)
        return C[y] + rng.standard_normal((n, D)) * sd, y

    ref, yr = draw(n_ref)
    test, yt = draw(N_TEST)
    shifts = {
        csplit("motion_blur", 1): {"layer3.1": 1.0 * e[10]},                        # AX-1 complement, AX-2 -> pre
        csplit("brightness", 1): {"stem": 1.0 * e[9]},                              # AX-2 -> stem
        csplit("snow", 1): {"layer3.1": (e[0] - e[1]) / np.sqrt(2.0)},             # AX-1 in-span: T_perp blind
        csplit("jpeg_compression", 1): {},                                          # identity copy (AX-3: h 0, loss 0)
        csplit("jpeg_compression", 3): {"penult": 6.0 * e[13]},                   # AX-4 off-simplex
        csplit("snow", 5): "noise",                                                 # AX-3: large h, large loss
        csplit("impulse_noise", 3): {"layer3.1": 1.0 * e[9]}, csplit("impulse_noise", 5): {"layer3.1": 1.2 * e[9]},
        csplit("zoom_blur", 3): {"layer3.1": 1.0 * e[10]}, csplit("zoom_blur", 5): {"layer3.1": 1.2 * e[10]},
        csplit("glass_blur", 3): {"layer3.1": 1.0 * e[11]},
    }
    fam_dir = {"N": e[9], "B": e[10], "L": e[11], "P": e[12]}
    for fam, cs in FAMILIES.items():
        for c in cs:
            for sv in (3, 5):
                shifts.setdefault(csplit(c, sv), {"layer3.1": (1.0 if sv == 3 else 1.2) * fam_dir[fam]})
    base = test[:2000]
    splits = {"ref": (ref, yr), "test": (test, yt)}
    for s, sh in shifts.items():
        splits[s] = (base, yt[:2000], sh)
    ood = rng.standard_normal((2000, D)) * 3.0
    splits["ood__cifar100"] = (ood, np.zeros(2000, dtype=np.int64))
    os.makedirs(root, exist_ok=True)
    for sub in ("acts", "labels", "preds"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    noise = rng.standard_normal((2000, D)) * 3.0
    for s, v in splits.items():
        X0, y = v[0], v[1]
        sh = v[2] if len(v) > 2 else {}
        pen = None
        for L in SYN_LAYERS:
            X = X0.copy()
            if sh == "noise":
                X = X + noise
            elif L in sh:
                X = X + sh[L]
            os.makedirs(os.path.join(root, "acts", L), exist_ok=True)
            np.save(os.path.join(root, "acts", L, f"{s}.npy"), X.astype(np.float16))
            if L == "penult":
                pen = X.astype(np.float16).astype(np.float64)
        am = center_dists(pen, C).argmin(1)
        np.save(os.path.join(root, "labels", f"{s}.npy"), np.asarray(y, dtype=np.int64))
        np.savez(os.path.join(root, "preds", f"{s}.npz"), argmax=am, maxprob=np.full(len(am), 0.9, dtype=np.float32))
    meta = {"source": "synthetic", "arch": "cifar10_resnet20", "layers": list(SYN_LAYERS), "splits": list(splits),
            "n_classes": N_CLASSES, "n_test": N_TEST, "dtype": "float16", "pooling": "gap", "git_commit": None,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "exp_id": "anomaly_synth", "seed_tag": "synth"}
    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump(meta, f)
    return root


def selftest(n_ref=10000, workdir=None):
    """Known answers on tiny arrays and on a synthetic dump. Returns {"status": PASS|FAIL, "checks": [...]}."""
    checks = []

    def ck(name, ok, detail=""):
        checks.append({"name": name, "pass": bool(ok), "detail": detail})

    ck("auc separated", auc([2, 3], [0, 1]) == 1.0)
    ck("auc reversed", auc([0, 1], [2, 3]) == 0.0)
    ck("auc all ties", auc([1, 1], [1, 1]) == 0.5)
    ck("auc half tie", auc([1, 2], [1, 0]) == 0.875, str(auc([1, 2], [1, 0])))
    ck("rank ties averaged", list(rank_avg([3, 1, 3, 2])) == [3.5, 1.0, 3.5, 2.0])
    line = np.arange(10, dtype=np.float64)[:, None]
    ck("knn k=1", abs(knn_kth(line, [[0.4]], 1)[0] - 0.4) < 1e-12)
    ck("knn k=3", abs(knn_kth(line, [[0.4]], 3)[0] - 1.6) < 1e-12)
    ck("knn exclude_self", np.allclose(knn_kth(line, line, 2, exclude_self=True)[[0, 5, 9]], [2.0, 1.0, 2.0]))
    Cq = np.zeros((10, 16))
    for k in range(9):
        Cq[k, k] = 3.0
    Qq = class_span(Cq)
    ck("class span rank 9", Qq.shape[1] == 9)
    x = np.zeros((1, 16))
    x[0, 0], x[0, 12] = 3.0 + 3.0, 4.0                                  # 3 e0 in-span + 4 e12 off-span from c_0
    ck("e_perp hand value 16/25", abs(e_perp(x, Cq, Qq, np.array([0]))[0] - 16.0 / 25.0) < 1e-12)
    b1, b2 = batch_idx("t|x|8", 50, 8, 5), batch_idx("t|x|8", 50, 8, 5)
    ck("batch_idx deterministic, no replacement", np.array_equal(b1, b2) and all(len(set(r)) == 8 for r in b1))
    tmp = workdir or tempfile.mkdtemp(prefix="anomaly_selftest_")
    try:
        name, rec = process_dump(os.path.join(tmp, "margin_b1_vitb16", "dump"))
        ck("refuses a B1 path before reading", rec["status"] == "REFUSED" and "never read" in rec["error"], rec["error"])
        root = make_synth_dump(os.path.join(tmp, "synth_r20", "dump"), n_ref=n_ref)
        name, rec = process_dump(root)
        ck("refuses a synthetic dump without the self-test flag", rec["status"] == "REFUSED", rec.get("error"))
        name, dis = process_dump(root, role_override="discovery", allow_synthetic=True)
        ck("discovery role never reads a confirmation corruption",
           dis["status"] == "OK" and not any(parse_csplit(s) and parse_csplit(s)[0] in HOLDOUT for s in dis["splits_read"]),
           str(dis.get("error")))
        name, con = process_dump(root, role_override="confirmation", allow_synthetic=True)
        ck("confirmation role reads the holdout", con["status"] == "OK" and "corrupt__zoom_blur__s3" in con["splits_read"],
           str(con.get("error")))
        if con["status"] == "OK":
            a1, a2, a3, a4 = con["AX1"], con["AX2"], con["AX3"], con["AX4"]
            ms = csplit("motion_blur", 1)
            # known-answer margins (review item A6): a node port of this synthetic dump over 12 seeds gave 0.9935-0.9998
            # for both complement shifts below; 0.97 keeps the check meaningful without a knife-edge on a frozen file
            ck("AX-1 complement shift detected (AUC >= 0.97)", a1["auc_perp_pre"][ms] >= 0.97, str(a1["auc_perp_pre"][ms]))
            ck("AX-1 in-span shift invisible to T_perp (0.35..0.65)", 0.35 <= a1["auc_perp_pre"][csplit("snow", 1)] <= 0.65,
               str(a1["auc_perp_pre"][csplit("snow", 1)]))
            ck("AX-1 pre-only shift: AUC pre - penult >= 0.3", a1["auc_perp_pre"][ms] - a1["auc_perp_penult"][ms] >= 0.3)
            ck("AX-1 single-class FPR of T_perp small (<= 0.30)", a1["single_class"]["perp_pre"]["fpr_mean"] <= 0.30,
               str(a1["single_class"]["perp_pre"]["fpr_mean"]))
            ck("AX-1 single-class FPR of penult T_full large (>= 0.90)", a1["single_class"]["full_penult"]["fpr_mean"] >= 0.90,
               str(a1["single_class"]["full_penult"]["fpr_mean"]))
            ck("AX-1 T_par separates single-class batches (>= 0.95), T_perp does not (<= 0.70)",
               a1["single_vs_mixed_auc"]["par_pre"] >= 0.95 and a1["single_vs_mixed_auc"]["perp_pre"] <= 0.70,
               str(a1["single_vs_mixed_auc"]))
            ck("AX-1 holdout glass_blur s3 detected (AUC >= 0.97)", a1["auc_perp_pre"][csplit("glass_blur", 3)] >= 0.97,
               str(a1["auc_perp_pre"][csplit("glass_blur", 3)]))
            am = a2["auc"][csplit("brightness", 1)]
            ck("AX-2 stem-only shift peaks at stem", max(am, key=am.get) == "stem", str(am))
            am = a2["auc"][ms]
            ck("AX-2 pre-only shift peaks at pre", max(am, key=am.get) == "pre", str(am))
            acc = a2["router"]["accuracy_by_family"]
            ck("AX-2 router families (>= 0.99)", all(v is not None and v >= 0.99 for v in acc.values()), str(acc))
            ho = a2["router"]["holdout"]
            ck("AX-2 holdout routing (impulse -> N, zoom -> B)",
               ho[csplit("impulse_noise", 3)]["assigned"]["N"] >= 0.99 and ho[csplit("zoom_blur", 5)]["assigned"]["B"] >= 0.99)
            idn = a3["splits"][csplit("jpeg_compression", 1)]
            ck("AX-3 identity split: loss 0 everywhere, |median h| < 0.15",
               max(abs(v) for v in idn["loss"]) == 0.0 and abs(float(np.median(idn["h"]))) < 0.15,
               f"median h {np.median(idn['h']):.3f}")
            nz = a3["splits"][csplit("snow", 5)]
            ck("AX-3 heavy noise: median h > 1 and mean loss > 10",
               float(np.median(nz["h"])) > 1.0 and float(np.mean(nz["loss"])) > 10.0,
               f"median h {np.median(nz['h']):.2f}, loss {np.mean(nz['loss']):.1f}")
            ck("AX-4 off-simplex shift: AUROC(e_perp) >= 0.90", a4["auroc"][csplit("jpeg_compression", 3)]["e_perp"] >= 0.90,
               str(a4["auroc"][csplit("jpeg_compression", 3)]))
            ck("AX-4 identity split: AUROC(e_perp) near 0.5",
               abs(a4["auroc"][csplit("jpeg_compression", 1)]["e_perp"] - 0.5) <= 0.06)
            _, con2 = process_dump(root, role_override="confirmation", allow_synthetic=True)
            strip = lambda r: {k: v for k, v in r.items() if k != "timing_s"}   # noqa: E731
            ck("deterministic (two runs identical)",
               json.dumps(_clean(strip(con)), sort_keys=True) == json.dumps(_clean(strip(con2)), sort_keys=True))
        outroot = os.path.join(tmp, "out")
        p, _ = run("selftest", [root], out_root=outroot, role_override="confirmation", allow_synthetic=True)
        ok = False
        try:
            run("selftest", [root], out_root=outroot, role_override="confirmation", allow_synthetic=True)
        except FileExistsError:
            ok = True
        ck("refuses to overwrite results/anomaly_probe_<tag>", ok and os.path.isfile(p))
    finally:
        if workdir is None:
            shutil.rmtree(tmp, ignore_errors=True)
    bad = [c["name"] for c in checks if not c["pass"]]
    return {"status": "PASS" if not bad else "FAIL", "failed": bad, "checks": checks, "code_sha256": code_sha256(),
            "env": env_record(), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def main(argv=None):
    ap = argparse.ArgumentParser(description="ANOMALY_H1 probes AX-1..AX-4 (CPU, numpy only)")
    ap.add_argument("--tag", help="output goes to <out-root>/anomaly_probe_<tag>/probe.json (must not exist)")
    ap.add_argument("--dumps", nargs="+", help="CIFAR dump dirs (results/<run>/dump)")
    ap.add_argument("--out-root", default="results")
    ap.add_argument("--selftest", action="store_true", help="known answers on tiny arrays and a synthetic dump")
    ap.add_argument("--selftest-out", help="--selftest: write the record here (must not exist)")
    a = ap.parse_args(argv)
    if a.selftest:
        rep = selftest()
        for c in rep["checks"]:
            print(f"[selftest] {'PASS' if c['pass'] else 'FAIL'} {c['name']}{' (' + c['detail'] + ')' if c['detail'] and not c['pass'] else ''}")
        if a.selftest_out:
            if os.path.exists(a.selftest_out):
                print(f"[selftest] {a.selftest_out} exists: not overwritten", file=sys.stderr)
                return 2
            os.makedirs(os.path.dirname(os.path.abspath(a.selftest_out)), exist_ok=True)
            with open(a.selftest_out, "w") as f:
                json.dump(_clean(rep), f, indent=1)
        print(f"[selftest] {rep['status']}")
        return 0 if rep["status"] == "PASS" else 1
    if not a.tag or not a.dumps:
        ap.error("--tag and --dumps are required (or --selftest)")
    try:
        path, payload = run(a.tag, a.dumps, out_root=a.out_root)
    except (FileExistsError, ValueError) as e:
        print(f"[anomaly_probe] refused: {e}", file=sys.stderr)
        return 2
    st = {k: v["status"] for k, v in payload["dumps"].items()}
    print(f"[anomaly_probe] wrote {path}: {st}")
    return 0 if all(v == "OK" for v in st.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
