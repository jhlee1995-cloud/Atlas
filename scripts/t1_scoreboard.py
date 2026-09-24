#!/usr/bin/env python3
"""
t1_scoreboard.py -- Track 1 (batch 4): the head-conditional scoreboard (docs/plans/T1_SCOREBOARD.md; EXTRACTION_PROPOSALS
X1). For one registry unit and one layout it measures, per sample, what intermediate geometry adds BEYOND THE FULL OUTPUT
HEAD, with the standard detectors adopted as components and as baselines:
  X4  the calibrated multi-tap conformal flag (L2-normalised kNN per X4 tap -> conformal p -> Cauchy fusion -> conformal p)
  X8  Trust Score, relative Mahalanobis, kNN purity, LID
  X7  late-depth trajectories (prediction depth, cross-tap agreement, margin area)
  X2  the head-null residual (whitened energy off the head's row space)
  X6  the batch arm: harm against AC / DoC / ATC, BBSE / BBSD label-skew tests, the BBSE-explained covariate residual R,
      covariate / prior / novelty typing
and it writes the per-model outcomes O1 and O5 that T2's PRIMARY reads (experiments/b4/model_outcomes.schema.json; fit
layout only). It applies no pre-registered threshold: scripts/t1_eval.js decides. CPU, numpy only; the shared numerics
(AUC, DeLong, the ridge-logistic joint model, the HEAD-ADDITIVE rule, conformal p) are atlas/b4_core.py.

  python scripts/t1_scoreboard.py --registry experiments/b4/models.json --unit resnet56_s31 --phase confirmation \
      --layout eval --out results/b4_t1/resnet56_s31_eval
  python scripts/t1_scoreboard.py --selftest [--selftest-out results/instrument_check_b4s1/selftest_t1_scoreboard.json]

Dumps are opened ONLY through atlas/b4_core.open_unit (the D7 seal): a sealed dump opens only in the confirmation phase
after P2, and in the discovery phase every confirmation-only split (holdout corruptions, CIFAR-10-C extras, the
exposure_global fault) is invisible (Dump.list_splits) and refused if named. The head path is the stored float32 logits.

Rows (global CIFAR-10 test indices; corrupt and fault splits pair by row with the clean test row; D4):
  fit   test = rows 0-4999 of the unit's fit dump; corrupt rows 0-1999 (all positives, POS); OOD rows 0-1999; T1 faults of
        rows 3500-4999 (salt 0). B = 3500-4999: clean negatives. CAL = A = 2000-3499: temperature, BBSE confusion matrix,
        ATC thresholds, univariate thresholds, batch frames, stream calibration; X4 per-tap p on A1 = 2000-2749, fusion
        on A2 = 2750-3499 (n_cal 750).
  eval  (F, F20, R2; never-read rows) test = rows 5000-9999 of the eval dump; corrupt rows 5000-9999, positives of the
        detection targets 5000-8499 (POS); OOD 2000-6999; faults of rows 8500-9999 (salt 1); B = 8500-9999. Calibration
        comes from the SAME unit's fit dump, whole fit split (T1 review A2): CAL = fit rows 0-4999, X4 per-tap p on fit
        rows 0-2499, fusion on 2500-4999 (n_cal 2500).
  Both layouts: the best head statistic is chosen on the fit dump's rows 2000-3499 (HEAD_SEL, minimum AURC on clean
  errors) before any other decision, so the eval run reproduces the fit run's choice exactly.
Per-sample geometric signals are referenced to the TRAIN reference rows of the fit dump only (class means, kNN banks,
whitening, Mahalanobis), so the error targets may use every test row without held-out leakage.

Output: results/b4_t1/<id>_<fit|eval><tag>/scoreboard.json through atlas/b4_core.ProbeRun (append-only; confirmation
units are touched once), with timing_s per tap ('tap:<t>'), per target ('target:<t>') and per block ('block:<b>'),
max_rss_mb, code.sha256 and repo_commit. Functional units (D14): dtpr5 (TPR at 5% FPR), daurc, and the 'cost' block
(reference bytes; CPU ms per 1000 queries under cost.timing_s, which the S2 replay compare ignores).
"""
import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from atlas import b4_core as C              # noqa: E402  numpy + stdlib only (D5)
from atlas import b4_collapse as NC         # noqa: E402  numpy only: nc1 of the DO-3 replication, Gram-form distances

SCHEMA = "b4_t1_scoreboard/1"
OUTCOMES_SCHEMA = "b4_model_outcomes/1"
OUTCOMES_SCHEMA_PATH = os.path.join(REPO_ROOT, "experiments", "b4", "model_outcomes.schema.json")
K = C.K
TAPKEYS = ("stem", "s1end", "s2end", "pre", "penult")
BATCH_KEYS = ("penult", "pre", "s2end")
FAULT_KINDS = ("deadpix", "occlusion_disc", "exposure_global")
FAULT_LEVELS = (1, 2, 3)
FAMS = ("N", "B", "W", "D")
OOD_GROUP = {"cifar100": 1_000_000, "svhn": 2_000_000}       # OOD rows never share a fold group with CIFAR-10 rows
LAYOUT = {   # CAL, X4_1, X4_2, HEAD_SEL index rows of the FIT dump's test split in both layouts
    "fit": {"test": (0, 5000), "B": (3500, 5000), "POS": (0, 2000), "CAL": (2000, 3500), "X4_1": (2000, 2750),
            "X4_2": (2750, 3500), "HEAD_SEL": (2000, 3500)},
    "eval": {"test": (5000, 10000), "B": (8500, 10000), "POS": (5000, 8500), "CAL": (0, 5000), "X4_1": (0, 2500),
             "X4_2": (2500, 5000), "HEAD_SEL": (2000, 3500)},
}
CFG = {"k_knn": 10, "k_purity": (5, 10, 25, 50), "k_lid": 20, "trust_alpha": 0.1, "r_white": 10, "r_batch": 20,
       "r_null": 20, "alpha": 0.05, "conf_stratum": 0.5, "min_n": 20, "batch_sizes": (16, 64, 256), "n_batches": 200,
       "n_single": 50, "dirichlet": (0.1, 1.0, 10.0), "markov": (0.9, 0.99), "vim_dim_frac": 0.5, "neco_dim": 9,
       "maha_floor": 1e-6, "n_cost": 1000, "typing_n": 64, "typing_cov_batches": 20, "typing_batches": 200,
       "nov_p": 0.01, "add_pix": 0.005}
# orientation: a LARGER oriented score is more error-, shift- or novelty-like (head statistics: atlas/b4_core.ERR_SIGN)
SIGN = dict(C.ERR_SIGN)
SIGN.update({"d1": 1.0, "margin": -1.0, "d1rel": 1.0, "nccdis": 1.0, "ncchead": 1.0, "wnorm": 1.0, "knnL2": 1.0,
             "knn": 1.0, "logr10": 1.0, "purity5": -1.0, "purity10": -1.0, "purity25": -1.0, "purity50": -1.0,
             "lid": 1.0, "trust": -1.0, "maha": 1.0, "relmaha": 1.0, "neco": -1.0, "snull": 1.0, "srow": 1.0,
             "nc3p": -1.0, "vim": 1.0, "x4": 1.0, "x4tap": 1.0, "pd": 1.0, "agree": -1.0, "lastdis": 1.0,
             "marea": -1.0})
HEAD_STATS = C.HEAD_STATS
O5_HEADS = ("msp", "maxlogit", "gap", "energy", "entropy")
# O1's penult bundle is fixed by experiments/b4/model_outcomes.schema.json (the PRIMARY reads it; integration decision 9)
O1_BUNDLE = ("margin_penult", "d1_penult", "knnL2_penult", "trust_penult", "relmaha_penult", "purity10_penult",
             "purity50_penult", "lid_penult")
BUNDLES = {
    "margin_pen": ("margin_penult",),
    "d1_pen": ("d1_penult",),
    "knnL2_pen": ("knnL2_penult",),
    "penult": O1_BUNDLE,
    "pre": ("margin_pre", "d1rel_pre", "nccdis_pre", "knnL2_pre", "purity10_pre"),
    "s2": ("knnL2_s2end", "wnorm_s2end"),
    "early": ("knnL2_stem", "wnorm_stem", "knnL2_s1end", "wnorm_s1end", "knnL2_s2end", "wnorm_s2end"),
    "x4": ("x4",),
    "trust_pen": ("trust_penult",),
    "relmaha_pen": ("relmaha_penult",),
    "local_pen": ("purity10_penult", "purity50_penult", "lid_penult", "trust_penult"),
    "nc3p_pen": ("nc3p_penult", "ncchead_penult"),
    "null_pen": ("snull_penult", "srow_penult"),
    "ood_std": ("neco_penult", "vim_penult", "maha_penult"),
    "traj": ("pd", "agree", "marea"),
    "all_geom": ("margin_penult", "d1_penult", "knnL2_penult", "margin_pre", "d1rel_pre", "nccdis_pre", "knnL2_pre",
                 "purity10_pre", "knnL2_s2end", "wnorm_s2end", "knnL2_stem", "knnL2_s1end", "x4", "trust_penult",
                 "relmaha_penult", "snull_penult", "pd", "agree", "marea"),
}
PIX_BUNDLES = ("early", "s2", "x4", "pre", "all_geom")      # these also get the pixel twin (beyond head AND pixels)
FAST_BUNDLES = ("margin_pen", "penult", "pre", "early", "x4", "local_pen", "traj", "null_pen", "all_geom")


def say(msg, quiet=False):
    if not quiet:
        print(f"[t1] {msg}", flush=True)


def base(name):
    return name.split("_")[0]


def in_range(rows, lo_hi):
    lo, hi = lo_hi
    rows = np.asarray(rows)
    return np.flatnonzero((rows >= lo) & (rows < hi))


# =====================================================================================================================
# numerics specific to T1 (the shared ones are atlas/b4_core.py)
# =====================================================================================================================
def auc_in_deciles(y, s, gap):
    """AUROC inside logit-gap deciles, pooled with weights n_pos * n_neg (deciles with >= 10 of each class)."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    q = np.quantile(gap, np.linspace(0, 1, 11))
    idx = np.clip(np.searchsorted(q[1:-1], gap, side="right"), 0, 9)
    num = den = 0.0
    used = 0
    for j in range(10):
        m = idx == j
        npos, nneg = int((y & m).sum()), int((~y & m).sum())
        if npos >= 10 and nneg >= 10:
            a = C.auc(s[y & m], s[~y & m])
            num += a * npos * nneg
            den += npos * nneg
            used += 1
    return (float(num / den) if den > 0 else None), used


def batch_idx(key, n_rows, batch, nb):
    """nb seeded batches of `batch` distinct local rows out of n_rows (None if too few rows)."""
    if n_rows < batch:
        return None
    rng = C.rng_for("t1_batch", key)
    return np.argsort(rng.random((nb, n_rows)), axis=1)[:, :batch]


def ks_stat_sorted(a, b_sorted):
    a = np.sort(np.asarray(a, dtype=np.float64))
    x = np.concatenate([a, b_sorted])
    return float(np.max(np.abs(np.searchsorted(a, x, side="right") / len(a)
                               - np.searchsorted(b_sorted, x, side="right") / len(b_sorted))))


def ks_stat(a, b):
    return ks_stat_sorted(a, np.sort(np.asarray(b, dtype=np.float64)))


def ks_pvalue(d, n, m):
    """Asymptotic two-sample KS p-value (Kolmogorov series, Stephens' correction); d may be an array."""
    d = np.asarray(d, dtype=np.float64)
    en = math.sqrt(n * m / (n + m))
    lam = (en + 0.12 + 0.11 / en) * d
    j = np.arange(1, 101, dtype=np.float64).reshape((-1,) + (1,) * lam.ndim)
    s = (2.0 * (-1.0) ** (j - 1) * np.exp(-2.0 * j * j * lam * lam)).sum(0)
    return np.where(lam < 1e-3, 1.0, np.clip(s, 0.0, 1.0))


def project_simplex(v):
    """Euclidean projection onto the probability simplex (Duchi et al. 2008)."""
    v = np.asarray(v, dtype=np.float64)
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (css - 1))[0][-1]
    theta = (css[rho] - 1) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def isotonic_fit(x, y, increasing=True):
    """Pool-adjacent-violators; returns a step function f(x_new) (clamped at the ends)."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    o = np.argsort(x, kind="mergesort")
    xs, ys = x[o], (y[o] if increasing else -y[o])
    vals, wts, ends = [], [], []
    for i in range(len(xs)):
        vals.append(ys[i])
        wts.append(1.0)
        ends.append(i)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = wts[-2] + wts[-1]
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            vals[-2:] = [v]
            wts[-2:] = [w]
            ends[-2:] = [ends[-1]]
    xb = xs[np.array(ends, dtype=np.int64)]
    vb = np.array(vals) if increasing else -np.array(vals)

    def f(xn):
        j = np.clip(np.searchsorted(xb, np.asarray(xn, dtype=np.float64), side="left"), 0, len(xb) - 1)
        return vb[j]
    return f


def call_pix(dauc_pix, dauc_pix_ci, dI_pix_ci):
    """Beyond head AND pixel statistics (frozen, T1 plan section 3): ADDS when dAUC_pix >= +0.005 and the 95% CI of
    dI_pix lies above 0; BOUNDED when the upper bound of dAUC_pix is below EQ_BOUND = +0.02; else INCONCLUSIVE."""
    if dauc_pix is None or dauc_pix_ci is None or dI_pix_ci is None or None in dauc_pix_ci or None in dI_pix_ci:
        return "INCONCLUSIVE"
    if dauc_pix >= CFG["add_pix"] and dI_pix_ci[0] > 0:
        return "ADDS"
    if dauc_pix_ci[1] < C.EQ_BOUND:
        return "BOUNDED"
    return "INCONCLUSIVE"


def validate_schema(obj, schema, root=None, path="$"):
    """The JSON-schema subset experiments/b4/model_outcomes.schema.json uses ($ref, type, const, enum, required,
    properties, additionalProperties, items, minItems, maxItems, minimum). Returns a list of error strings."""
    root = schema if root is None else root
    if "$ref" in schema:
        node = root
        for part in schema["$ref"].lstrip("#/").split("/"):
            node = node[part]
        return validate_schema(obj, node, root, path)
    errs = []
    if "type" in schema:
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]

        def is_type(t):
            if t == "null":
                return obj is None
            if t == "boolean":
                return isinstance(obj, bool)
            if t == "integer":
                return isinstance(obj, int) and not isinstance(obj, bool)
            if t == "number":
                return isinstance(obj, (int, float)) and not isinstance(obj, bool)
            if t == "string":
                return isinstance(obj, str)
            if t == "array":
                return isinstance(obj, list)
            if t == "object":
                return isinstance(obj, dict)
            return False
        if not any(is_type(t) for t in types):
            return [f"{path}: type {type(obj).__name__} is not {types}"]
    if "const" in schema and obj != schema["const"]:
        errs.append(f"{path}: {obj!r} != const {schema['const']!r}")
    if "enum" in schema and obj not in schema["enum"]:
        errs.append(f"{path}: {obj!r} not in {schema['enum']}")
    if isinstance(obj, (int, float)) and not isinstance(obj, bool) and "minimum" in schema and obj < schema["minimum"]:
        errs.append(f"{path}: {obj} < minimum {schema['minimum']}")
    if isinstance(obj, dict):
        for k in schema.get("required", []):
            if k not in obj:
                errs.append(f"{path}: missing {k}")
        props = schema.get("properties", {})
        addl = schema.get("additionalProperties", True)
        for k, v in obj.items():
            if k in props:
                errs += validate_schema(v, props[k], root, f"{path}.{k}")
            elif addl is False:
                errs.append(f"{path}: unexpected key {k}")
            elif isinstance(addl, dict):
                errs += validate_schema(v, addl, root, f"{path}.{k}")
    if isinstance(obj, list):
        if "minItems" in schema and len(obj) < schema["minItems"]:
            errs.append(f"{path}: {len(obj)} items < {schema['minItems']}")
        if "maxItems" in schema and len(obj) > schema["maxItems"]:
            errs.append(f"{path}: {len(obj)} items > {schema['maxItems']}")
        if "items" in schema:
            for i, v in enumerate(obj):
                errs += validate_schema(v, schema["items"], root, f"{path}[{i}]")
    return errs


# =====================================================================================================================
# per-tap reference structures (TRAIN reference rows only) and per-sample geometric signals
# =====================================================================================================================
def _pinv_sqrt(S):
    """(V, sqrt(w)) on the eigenpairs of S above maha_floor * lambda_1: pseudo-inverse semantics (T1 review E1)."""
    w, V = C.eig_desc(S)
    if len(w) == 0:
        return np.zeros((S.shape[0], 0)), np.zeros(0)
    keep = w > CFG["maha_floor"] * w[0]
    return V[:, keep], np.sqrt(w[keep])


def _sub_frame(Z, r, P):
    """Top-r principal frame of Z (already projected by P): returns (P @ V, sqrt(w)) so a score is ((Qc @ PV) / s)^2
    summed, without ever forming Qc @ P for a wide penult (T1 review E2)."""
    if r < 1:
        return np.zeros((P.shape[0], 0)), np.zeros(0)
    w, V = C.eig_desc(np.cov(Z, rowvar=False), r)
    return P @ V, np.sqrt(w)


def _resid_norm(Xo, Vp):
    """|| Xo - Xo Vp Vp^T || for an orthonormal Vp, as sqrt(|Xo|^2 - |Xo Vp|^2) (ViM residual; T1 review E2)."""
    a = np.einsum("ij,ij->i", Xo, Xo)
    p = Xo @ Vp
    return np.sqrt(np.maximum(a - np.einsum("ij,ij->i", p, p), 0.0))


class TapRef:
    """Reference structures of one tap. level 'centres': class means and radii only (trajectory taps); 'light': + the
    L2-normalised kNN bank, the raw bank and a top-10 PCA frame; 'full': + Mahalanobis, Trust Score, NECO and, with a
    head (W, b) of matching width, the head-null / head-row frames, ViM and the NC3+ alignment."""

    def __init__(self, X, y, W=None, b=None, level="full"):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self.level, self.n, self.d = level, int(len(X)), int(X.shape[1])
        self.C = C.class_means(X, y)
        self.rad = np.array([math.sqrt(float(((X[y == k] - self.C[k]) ** 2).sum(1).mean())) for k in range(K)])
        self.W = self.b = None
        self.ts_X = None
        if level == "centres":
            return
        self.y, self.X, self.Xn = y, X, C.l2n(X)
        self.mu = X.mean(0)
        self.frame = C.pca_frame(X, CFG["r_white"])
        if level != "full":
            return
        Wc = X - self.C[y]
        self.mw = _pinv_sqrt(Wc.T @ Wc / len(X))                  # class-conditional, shared within-class covariance
        self.mg = _pinv_sqrt(np.cov(X, rowvar=False))              # background Gaussian (relative Mahalanobis)
        keep = []
        for k in range(K):                                         # Trust Score: the (1 - alpha) densest points per class
            Xk = X[y == k]
            kk = max(1, min(CFG["k_knn"], len(Xk) - 1))
            rk = C.knn_kth(Xk, Xk, kk, exclude_self=True)
            keep.append(Xk[rk <= np.quantile(rk, 1.0 - CFG["trust_alpha"])])
        self.ts_X = np.vstack(keep)
        self.ts_bounds = np.cumsum([0] + [len(a) for a in keep])
        self.neco_V = C.pca_frame(X, CFG["neco_dim"])[2]
        if W is not None and np.asarray(W).shape[1] == self.d:
            W = np.asarray(W, dtype=np.float64)
            b = np.asarray(b, dtype=np.float64)
            self.W, self.b = W, b
            Wp = np.linalg.pinv(W)                                  # (d, K)
            Pr = Wp @ W                                             # orthogonal projector onto the head's row space
            Pn = np.eye(self.d) - Pr
            Xc = X - self.mu
            self.null_frame = _sub_frame(Xc @ Pn, CFG["r_null"], Pn)
            self.row_frame = _sub_frame(Xc @ Pr, int(np.linalg.matrix_rank(W)), Pr)
            self.vim_o = -Wp @ b                                    # ViM (Wang et al. 2022): origin -W^+ b
            Xo = X - self.vim_o
            _, Vv = C.eig_desc(Xo.T @ Xo / len(Xo))
            dp = max(1, min(int(CFG["vim_dim_frac"] * self.d), Vv.shape[1]))
            self.vim_V = Vv[:, :dp]
            self.vim_alpha = float((X @ W.T + b).max(1).mean() / (_resid_norm(Xo, self.vim_V).mean() + 1e-12))
            self.nc3_W = W - W.mean(0)

    def maha(self, Q):
        V, s = self.mw
        md = C.sqdist((Q @ V) / s, (self.C @ V) / s)
        Vg, sg = self.mg
        md0 = ((((Q - self.mu) @ Vg) / sg) ** 2).sum(1)
        return md.min(1), md.min(1) - md0

    def trust(self, Q, am):
        D = np.full((len(Q), K), np.inf)
        f2 = np.einsum("ij,ij->i", self.ts_X, self.ts_X)
        for i in range(0, len(Q), 512):
            d2 = C.sqdist(Q[i:i + 512], self.ts_X, f2)
            for k in range(K):
                a, e = int(self.ts_bounds[k]), int(self.ts_bounds[k + 1])
                if e > a:
                    D[i:i + 512, k] = np.sqrt(d2[:, a:e].min(1))
        idx = np.arange(len(Q))
        dp = D[idx, am]
        Do = D.copy()
        Do[idx, am] = np.inf
        return Do.min(1) / (dp + 1e-12)

    def nbytes(self):
        """Bytes a deployed float32 reference needs (functional units, D14)."""
        out = {"centres": int(self.C.size * 4)}
        if self.level != "centres":
            out["knn_bank"] = int(self.n * self.d * 4)
        if self.level == "full":
            out["maha"] = int((self.mw[0].size + self.mg[0].size + self.C.size) * 4)
            out["trust_bank"] = int(self.ts_X.size * 4)
        return out


def tap_signals(tr, Q, am, key, logits=None):
    """Per-sample signals of query rows Q (head argmax am) at the functional tap `key`; names <signal>_<key>."""
    Q = np.asarray(Q, dtype=np.float64)
    out = {}
    D = NC.center_dists(Q, tr.C)
    Ds = np.sort(D, axis=1)
    nearest = D.argmin(1)
    idx = np.arange(len(Q))
    out[f"d1_{key}"] = Ds[:, 0]
    out[f"margin_{key}"] = Ds[:, 1] - Ds[:, 0]
    out[f"d1rel_{key}"] = Ds[:, 0] / (tr.rad[nearest] + 1e-12)
    out[f"nccdis_{key}"] = (nearest != am).astype(np.float64)
    out[f"ncchead_{key}"] = D[idx, am] - Ds[:, 0]
    out[f"wnorm_{key}"] = (C.whiten(Q, tr.frame) ** 2).sum(1)
    full = tr.level == "full"
    kq = max(CFG["k_purity"]) if full else CFG["k_knn"]
    dL, iL = C.knn(tr.Xn, C.l2n(Q), kq)
    out[f"knnL2_{key}"] = dL[:, CFG["k_knn"] - 1]
    if key == "penult":                                           # raw 10-NN distance: DO-3 density and the H grade
        r = C.knn_kth(tr.X, Q, CFG["k_knn"])
        out["knn_penult"] = r
        out["logr10_penult"] = np.log(r + 1e-12)
    if not full:
        return out
    nl = tr.y[iL]
    for kp in CFG["k_purity"]:
        out[f"purity{kp}_{key}"] = (nl[:, :kp] == am[:, None]).mean(1)
    dd = np.maximum(dL[:, :CFG["k_lid"]], 1e-12)
    out[f"lid_{key}"] = -1.0 / np.minimum(np.log(dd / dd[:, -1:]).mean(1), -1e-12)
    out[f"maha_{key}"], out[f"relmaha_{key}"] = tr.maha(Q)
    out[f"trust_{key}"] = tr.trust(Q, am)
    Qc = Q - tr.mu
    qn = np.linalg.norm(Qc, axis=1)
    out[f"neco_{key}"] = np.linalg.norm(Qc @ tr.neco_V, axis=1) / (qn + 1e-12)
    if tr.W is not None:
        for nm, (PV, s) in (("snull", tr.null_frame), ("srow", tr.row_frame)):
            out[f"{nm}_{key}"] = (((Qc @ PV) / s) ** 2).sum(1) if PV.shape[1] else np.zeros(len(Q))
        wa = tr.nc3_W[am]                                         # NC3+ alignment (Chen et al., CVPR 2026)
        out[f"nc3p_{key}"] = (Qc * wa).sum(1) / (qn * np.linalg.norm(wa, axis=1) + 1e-12)
        z = np.asarray(logits, dtype=np.float64) if logits is not None else Q @ tr.W.T + tr.b
        m = z.max(1)
        lse = m + np.log(np.exp(z - m[:, None]).sum(1))
        out[f"vim_{key}"] = tr.vim_alpha * _resid_norm(Q - tr.vim_o, tr.vim_V) - lse
    return out


def trajectory(refs, taps, get_acts, am):
    """X7: nearest-train-centre label per tap (forward order, penult last) against the head's argmax."""
    T = len(taps)
    n = len(am)
    idx = np.arange(n)
    agree = np.zeros((n, T), dtype=bool)
    marea = np.zeros(n)
    n_area = 0
    for t, tap in enumerate(taps):
        tr = refs[tap]
        D = NC.center_dists(get_acts(tap), tr.C)
        agree[:, t] = D.argmin(1) == am
        if t >= T // 2:
            da = D[idx, am]
            Do = D.copy()
            Do[idx, am] = np.inf
            marea += (Do.min(1) - da) / (tr.rad.mean() + 1e-12)
            n_area += 1
    suffix = np.flip(np.cumprod(np.flip(agree, 1), axis=1), 1).astype(bool)   # agrees at t and at every later tap
    pd = np.where(suffix.any(1), T - suffix.sum(1), T).astype(np.float64)
    last = np.where((~agree).any(1), T - 1 - np.argmax(np.flip(~agree, 1), axis=1), -1).astype(np.float64)
    return {"pd": pd / T, "agree": agree.sum(1) / T, "lastdis": (last + 1) / T, "marea": marea / max(1, n_area)}


# =====================================================================================================================
# the unit: every per-split table of one model and one layout
# =====================================================================================================================
class Unit:
    """level 'board': every signal of the scoreboard; 'streams': the light set scripts/t1_streams.py needs (head
    statistics, L2-kNN at the five functional taps, penult d1 and raw 10-NN, X4, CAL-whitened 10-d frames, pixels)."""

    def __init__(self, reg, uid, layout, phase, timed=None, level="board", quiet=False):
        if layout not in LAYOUT:
            raise SystemExit(f"[t1] unknown layout {layout}")
        if level not in ("board", "streams"):
            raise ValueError(level)
        self.reg, self.uid, self.layout, self.phase, self.level, self.quiet = reg, uid, layout, phase, level, quiet
        self.spec = C.unit_spec(reg, uid)
        self.timed = timed if timed is not None else C.Timing()
        self.L = LAYOUT[layout]
        self.fitd = C.open_unit(reg, uid, "fit", phase)                  # the D7 seal decides what may open
        self.dump = self.fitd if layout == "fit" else C.open_unit(reg, uid, "eval", phase)
        self.tables = {}
        self._tap_meta()

    def say(self, msg):
        say(f"{self.uid} {self.layout}: {msg}", self.quiet)

    def dumps(self):
        return [self.fitd] if self.dump is self.fitd else [self.fitd, self.dump]

    def _tap_meta(self):
        tm = self.fitd.b4.get("taps") or {}
        self.ft = self.fitd.functional_taps()
        for k in TAPKEYS:
            if not self.ft.get(k):
                raise SystemExit(f"[t1] {self.uid}: functional tap {k} missing in meta.b4.taps.functional")
        if self.dump is not self.fitd and self.dump.functional_taps() != self.ft:
            raise SystemExit(f"[t1] {self.uid}: the eval dump's functional taps differ from the fit dump's")
        dups = set(tm.get("dup_of_penult") or [])
        self.taps_all = [t for t in (tm.get("all") or self.fitd.taps) if t not in dups]
        if "penult" not in self.taps_all:
            self.taps_all.append("penult")
        self.key_of = {}
        for k in TAPKEYS:
            self.key_of.setdefault(self.ft[k], k)
        self.x4t = list(self.fitd.x4_taps())
        bad = [t for t in self.x4t if t not in self.key_of]
        if not self.x4t or bad:
            raise SystemExit(f"[t1] {self.uid}: X4 taps {self.x4t} are not functional taps {self.ft}")
        self.x4keys = [self.key_of[t] for t in self.x4t]

    def plan(self):
        """The layout dump's splits this unit reads (readable in the phase; extraction order)."""
        sp = [s for s in self.dump.list_splits() if s != "ref"]
        if self.level == "streams":                                  # scenarios use the discovery corruptions only
            sp = [s for s in sp if not s.startswith("corrupt__") or C.parse_split(s)[1] in C.DISC]
        return sp

    # ---- build ------------------------------------------------------------------------------------------------------
    def build(self):
        t0 = time.time()
        fd = self.fitd
        fz, fy, fr = fd.logits("test"), fd.labels("test"), fd.rows("test")
        self.cal_idx = in_range(fr, self.L["CAL"])
        self.x41_idx = in_range(fr, self.L["X4_1"])
        self.x42_idx = in_range(fr, self.L["X4_2"])
        self.headsel_idx = in_range(fr, self.L["HEAD_SEL"])
        for nm in ("cal_idx", "x41_idx", "x42_idx", "headsel_idx"):
            if len(getattr(self, nm)) < 20:
                raise SystemExit(f"[t1] {self.uid}: the fit dump's test split has no rows for {nm}")
        cal = self.cal_idx
        self.T = C.fit_temperature(fz[cal], fy[cal])
        zc, yc = fz[cal], fy[cal]
        amc = zc.argmax(1)
        hc = C.head_stats(zc)
        Cm = np.zeros((K, K))
        np.add.at(Cm, (amc, yc), 1.0)
        self.cal = {"acc": float((amc == yc).mean()), "msp": hc["msp"], "negent": -hc["entropy"],
                    "pred_hist": np.bincount(amc, minlength=K) / len(amc), "probs": C.softmax(zc),
                    "bbse_C": Cm / np.maximum(Cm.sum(0, keepdims=True), 1.0), "n": int(len(cal))}
        self.W, self.b = fd.head()
        ref_y = fd.labels("ref")
        self.refs = {}
        taps = [self.ft[k] for k in TAPKEYS] if self.level == "streams" else self.taps_all
        for tap in taps:
            key = self.key_of.get(tap)
            if self.level == "streams":
                lvl = "light"
            else:
                lvl = "full" if key in ("pre", "penult") else ("light" if key else "centres")
            with self.timed(f"tap:{tap}"):
                X = fd.acts(tap, "ref")
                self.refs[tap] = TapRef(X, ref_y, self.W if key == "penult" else None,
                                        self.b if key == "penult" else None, lvl)
                if key == "penult" and self.level == "board":
                    self.do3_tau = float(np.quantile(C.knn_kth(X, X, CFG["k_knn"], exclude_self=True), 0.95))
                    rec, _, _ = NC.nc_block(X, ref_y, K, self.W)
                    self.nc1_train = rec["nc1"] if rec else None
                    self.nc3_train = rec.get("head_center_cos") if rec else None
                del X
        self.say(f"reference structures {time.time() - t0:.0f}s ({len(self.refs)} taps)")
        # CAL-row frames: the batch arm's T^2 / residual frames, or the streams' 10-d whitening frames
        self.bframes, self.sframes = {}, {}
        for key in (BATCH_KEYS if self.level == "board" else TAPKEYS):
            tap = self.ft[key]
            XC = fd.acts(tap, "test")[cal]
            if self.level == "streams":
                self.sframes[key] = C.pca_frame(XC, CFG["r_white"])
                continue
            mk = C.class_means(XC, yc)
            Wd = XC - mk[yc]
            w, V = C.eig_desc(Wd.T @ Wd / len(XC), CFG["r_batch"])
            s = np.sqrt(w)
            self.bframes[key] = {"frame": C.pca_frame(XC, CFG["r_batch"]), "V": V, "s": s, "mkV": (mk @ V) / s}
        px = fd.factors("test")
        self.pix_frame = C.pca_frame(px[0][cal], None) if px is not None else None
        # per-split tables: the fit dump's test split first (calibration rows), then the layout's splits
        self.caltab = self._table(fd, "test")
        if self.layout == "fit":
            self.tables["test"] = self.caltab
        for sp in self.plan():
            if sp not in self.tables:
                self.tables[sp] = self._table(self.dump, sp)
        self._x4()
        hs = self.headsel_idx
        yw = self.caltab["wrong"][hs]
        # layout-invariant choice: the selection's mspT uses a temperature fitted on the selection rows themselves (fit
        # rows 2000-3499 in both layouts), not self.T (CAL rows, which differ between the fit and eval layouts)
        self.T_sel = C.fit_temperature(fz[hs], fy[hs])
        sel = {s: self.caltab[s][hs] for s in HEAD_STATS}
        sel["mspT"] = C.softmax(fz[hs], self.T_sel).max(1)
        self.aurc = {s: C.risk_coverage(yw, -SIGN[s] * sel[s])[0] for s in HEAD_STATS}
        self.best_head = min(HEAD_STATS, key=lambda s: (self.aurc[s], HEAD_STATS.index(s)))
        self.say(f"tables for {len(self.tables)} splits {time.time() - t0:.0f}s; best head statistic {self.best_head}")

    def _table(self, dump, sp):
        z = dump.logits(sp)
        y = dump.labels(sp)
        rows = dump.rows(sp).astype(np.int64)
        grp, name, _ = C.parse_split(sp)
        if grp == "ood":
            rows = rows + OOD_GROUP.get(name, 3_000_000)
        am = z.argmax(1)
        tb = {"split": sp, "y": y, "am": am, "rows": rows, "wrong": (am != y).astype(np.float64), "logits": z}
        hs = C.head_stats(z, self.T)
        hs.pop("argmax")
        tb.update(hs)
        acts = {}
        for key in TAPKEYS:
            tap = self.ft[key]
            if not dump.has(tap, sp):
                continue
            with self.timed(f"tap:{tap}"):
                X = dump.acts(tap, sp)
                tb.update(tap_signals(self.refs[tap], X, am, key, z))
                if self.level == "board" and key in BATCH_KEYS:
                    bf = self.bframes[key]
                    tb[f"bw_{key}"] = C.whiten(X, bf["frame"])
                    tb[f"br_{key}"] = (X @ bf["V"]) / bf["s"]
                if self.level == "streams":
                    tb[f"frame_{key}"] = C.whiten(X, self.sframes[key])
            acts[tap] = X
        if self.level == "board" and all(dump.has(t, sp) for t in self.taps_all):
            def get(tap):
                return acts[tap] if tap in acts else dump.acts(tap, sp)
            with self.timed("block:trajectory"):
                tb.update(trajectory(self.refs, self.taps_all, get, am))
        fx = dump.factors(sp)
        if fx is not None:
            tb["pix"] = fx[0]
            if self.pix_frame is not None:
                tb["bw_pix"] = C.whiten(fx[0], self.pix_frame)
        return tb

    def all_tables(self):
        out = list(self.tables.values())
        if not any(t is self.caltab for t in out):
            out.append(self.caltab)
        return out

    def _x4(self):
        """X4: conformal p per X4 tap (L2-kNN, calibrated on X4_1), Cauchy fusion recalibrated on X4_2 (fit rows)."""
        ct = self.caltab
        keys = self.x4keys
        cal1 = {k: ct[f"knnL2_{k}"][self.x41_idx] for k in keys}

        def ptap(tb):
            if not all(f"knnL2_{k}" in tb for k in keys):
                return None
            return np.column_stack([C.conformal_p(cal1[k], tb[f"knnL2_{k}"]) for k in keys])
        self.x4_cal = C.cauchy_stat(ptap(ct)[self.x42_idx])
        for tb in self.all_tables():
            P = ptap(tb)
            if P is None:
                continue
            p = C.conformal_p(self.x4_cal, C.cauchy_stat(P))
            tb["x4_p"] = p
            tb["x4"] = -np.log(p)
            for j, k in enumerate(keys):
                tb[f"x4tap_{k}"] = -np.log(P[:, j])

    def test_local(self, rows):
        """Local index in tables['test'] of global test rows (-1 if absent)."""
        tr = self.tables["test"]["rows"]
        lo = int(tr.min())
        pos = np.full(int(tr.max()) - lo + 1, -1, dtype=np.int64)
        pos[tr - lo] = np.arange(len(tr))
        r = np.asarray(rows, dtype=np.int64) - lo
        ok = (r >= 0) & (r < len(pos))
        out = np.full(r.shape, -1, dtype=np.int64)
        out[ok] = pos[r[ok]]
        return out

    def gather(self, parts, names):
        """parts: [(split, local index array)]; returns {name: concatenated array | None (a split lacks it)}."""
        out = {}
        for nm in names:
            vals = []
            for sp, ix in parts:
                v = self.tables[sp].get(nm)
                if v is None:
                    vals = None
                    break
                vals.append(np.asarray(v)[ix])
            out[nm] = None if vals is None else np.concatenate(vals, axis=0)
        return out


# =====================================================================================================================
# targets
# =====================================================================================================================
def local(unit, split, lo_hi):
    return in_range(unit.tables[split]["rows"], lo_hi)


def targets(unit):
    """name -> {parts [(split, idx)], y, kind, groups, fam, [train_ok, eval_mask, lofo]}."""
    T, L = unit.tables, unit.L
    have = set(T)
    out = {}
    nte = len(T["test"]["y"])
    out["err_clean"] = {"parts": [("test", np.arange(nte))], "kind": "error"}
    for s in C.SEVS:
        parts = [(C.csplit(c, s), np.arange(len(T[C.csplit(c, s)]["y"]))) for c in C.DISC if C.csplit(c, s) in have]
        if not parts:
            continue
        out[f"err_shift_s{s}"] = {"parts": parts, "kind": "error"}
        if s in (3, 5):
            fp = []
            for sp, ix in parts:
                tl = unit.test_local(T[sp]["rows"][ix])
                ok = tl >= 0
                ok[ok] = T["test"]["wrong"][tl[ok]] == 0
                fp.append((sp, ix[ok]))
            out[f"flip_s{s}"] = {"parts": fp, "kind": "error"}
    for grp, names in (("holdout", C.HOLDOUT), ("extra", C.EXTRA)):
        parts = [(C.csplit(c, 3), np.arange(len(T[C.csplit(c, 3)]["y"]))) for c in names if C.csplit(c, 3) in have]
        if parts:
            out[f"err_shift_{grp}_s3"] = {"parts": parts, "kind": "error"}
    negB = ("test", local(unit, "test", L["B"]))
    for fam in FAMS:
        for s in (1, 3):
            cs = [c for c in C.DISC if C.FAMILY[c] == fam and C.csplit(c, s) in have]
            if cs:
                out[f"corrupt_{fam}_s{s}"] = {"parts": [(C.csplit(c, s), local(unit, C.csplit(c, s), L["POS"]))
                                                        for c in cs] + [negB], "kind": "detect"}
    cs3 = [c for c in C.DISC if C.csplit(c, 3) in have]
    if cs3:
        out["corrupt_any_s3_lofo"] = {"parts": [(C.csplit(c, 3), local(unit, C.csplit(c, 3), L["POS"])) for c in cs3]
                                      + [negB], "kind": "detect", "lofo": True}
    for grp, names in (("holdout", C.HOLDOUT), ("extra", C.EXTRA)):
        cs = [c for c in names if C.csplit(c, 3) in have]
        if cs and cs3:
            out[f"corrupt_{grp}_s3_transfer"] = {
                "parts": [(C.csplit(c, 3), local(unit, C.csplit(c, 3), L["POS"])) for c in cs3 + cs] + [negB],
                "kind": "detect", "transfer_from": tuple(cs3)}
    fams3 = sorted({C.FAMILY[c] for c in cs3})
    if len(fams3) >= 2:                                   # per-sample family identification among corrupt rows (s3)
        for F in fams3:
            out[f"famid_{F}_s3"] = {"parts": [(C.csplit(c, 3), local(unit, C.csplit(c, 3), L["POS"])) for c in cs3],
                                    "kind": "family", "family": F}
    if all(C.csplit(c, s) in have for c in C.DISC for s in (1, 5)):
        out["sev_5v1"] = {"parts": [(C.csplit(c, 5), np.arange(len(T[C.csplit(c, 5)]["y"]))) for c in C.DISC]
                          + [(C.csplit(c, 1), np.arange(len(T[C.csplit(c, 1)]["y"]))) for c in C.DISC],
                          "kind": "severity"}
    for o in ("cifar100", "svhn"):
        sp = f"ood__{o}"
        if sp in have:
            out[f"ood_{o}"] = {"parts": [(sp, np.arange(len(T[sp]["y"]))), negB], "kind": "detect"}
    for f in FAULT_KINDS:
        fs = [C.fsplit(f, lv) for lv in FAULT_LEVELS if C.fsplit(f, lv) in have]
        if fs:
            out[f"fault_{f}"] = {"parts": [(x, np.arange(len(T[x]["y"]))) for x in fs] + [negB], "kind": "detect"}
    for name, t in out.items():
        ys, fams = [], []
        for sp, ix in t["parts"]:
            p = C.parse_split(sp)
            if t["kind"] == "error":
                ys.append(T[sp]["wrong"][ix])
            elif t["kind"] == "family":
                ys.append(np.full(len(ix), 1.0 if C.FAMILY[p[1]] == t["family"] else 0.0))
            elif t["kind"] == "severity":
                ys.append(np.full(len(ix), 1.0 if p[2] == 5 else 0.0))
            else:
                ys.append(np.full(len(ix), 0.0 if sp == "test" else 1.0))
            fams.append(np.full(len(ix), C.FAMILY.get(p[1], "-") if p[0] == "corrupt" else "-"))
        t["y"] = np.concatenate(ys).astype(np.float64)
        t["fam"] = np.concatenate(fams)
        t["groups"] = np.concatenate([T[sp]["rows"][ix] for sp, ix in t["parts"]])
        if "transfer_from" in t:
            src = np.concatenate([np.full(len(ix), sp == "test" or C.parse_split(sp)[1] in t["transfer_from"])
                                  for sp, ix in t["parts"]])
            t["train_ok"] = src                          # trained on the discovery families and the clean negatives
            t["eval_mask"] = (~src) | (t["y"] == 0)      # scored on the unseen families against the clean negatives
    bh = unit.best_head
    for b0 in ("err_clean", "err_shift_s3"):
        if b0 not in out:
            continue
        t = out[b0]
        s = -SIGN[bh] * unit.gather(t["parts"], [bh])[bh]          # confidence of the unit's best head statistic
        keep = s >= np.quantile(s, CFG["conf_stratum"])
        parts, off = [], 0
        for sp, ix in t["parts"]:
            parts.append((sp, ix[keep[off:off + len(ix)]]))
            off += len(ix)
        out[f"conf_{b0}"] = {"parts": parts, "kind": "error", "y": t["y"][keep], "fam": t["fam"][keep],
                             "groups": t["groups"][keep]}
    return out


# =====================================================================================================================
# the joint model: head-only vs head + bundle, cross-fitted on the same group folds (atlas/b4_core)
# =====================================================================================================================
def head_blocks(unit, parts):
    """T1's full head summary: sorted logits (linear) + splines of gap, max logit, energy, entropy (O1 schema)."""
    g = unit.gather(parts, ["logits", "gap", "maxlogit", "energy", "entropy"])
    sl = C.sorted_logits(g["logits"], min(10, g["logits"].shape[1]))
    return [("sorted_logits", sl, "linear")] + [(k, g[k], "spline") for k in ("gap", "maxlogit", "energy", "entropy")]


def sig_blocks(unit, parts, names):
    g = unit.gather(parts, list(names))
    blocks, nbad = [], 0
    for n in names:
        if g[n] is None:
            return None, 0
        v, bad = C.finite_fix(g[n])
        nbad += bad
        blocks.append((n, v, "spline"))
    return blocks, nbad


def pix_blocks(unit, parts):
    g = unit.gather(parts, ["pix"])
    return None if g["pix"] is None else [("pix", g["pix"], "spline")]


def summarise(y, p, kind):
    out = {"auc": C.auc_y(y, p), "ce_bits": float(C.ce_bits(y, p).mean()), "fpr_at_95tpr": C.fpr_at_tpr(y, p),
           "tpr_at_5fpr": C.tpr_at_fpr(y, p)}
    if kind == "error":
        a, r = C.risk_coverage(y, -p)
        out["aurc"], out["risk_at"] = a, r
    return out


def univariate_names(unit):
    names = list(HEAD_STATS) + ["x4", "pd", "agree", "lastdis", "marea"]
    for key in TAPKEYS:
        names += [f"{b}_{key}" for b in ("d1", "margin", "d1rel", "nccdis", "ncchead", "wnorm", "knnL2", "x4tap")]
        if key in ("pre", "penult"):
            names += [f"{b}_{key}" for b in ("purity10", "purity50", "lid", "trust", "maha", "relmaha", "neco")]
    names += ["knn_penult", "snull_penult", "srow_penult", "nc3p_penult", "vim_penult"]
    t = unit.caltab
    return [n for n in names if n in t]


def univariate_block(unit, parts, em, ye):
    gap = unit.gather(parts, ["gap"])["gap"][em]
    out = {}
    for nm in univariate_names(unit):
        v = unit.gather(parts, [nm])[nm]
        if v is None:
            continue
        v, _ = C.finite_fix(v)
        s = SIGN[base(nm)] * v[em]
        ad, used = auc_in_deciles(ye, s, gap)
        out[nm] = {"auc": C.auc_y(ye, s), "fpr_at_95tpr": C.fpr_at_tpr(ye, s), "auc_gap_deciles": ad,
                   "deciles_used": used}
    return out


def joint_eval(unit, tname, t, nboot, bundles=None):
    """Cross-fitted head-only vs head + bundle on the same folds (the atlas/b4_core.head_increment arithmetic,
    identical by construction: the same crossfit calls, boot_ci key '<unit>|<target>|<bundle>', DeLong, dI).
    Predictions are made for every sample (a transfer target trains only on train_ok); metrics are read on eval_mask."""
    bundles = list(BUNDLES) if bundles is None else [b for b in bundles if b in BUNDLES]
    y = t["y"]
    parts = t["parts"]
    groups = np.asarray(t["groups"])
    em = np.asarray(t.get("eval_mask", np.ones(len(y), dtype=bool)), dtype=bool)
    ye, ge = y[em], groups[em]
    npos, nneg = int(ye.sum()), int(len(ye) - ye.sum())
    res = {"n_pos": npos, "n_neg": nneg, "n": int(len(ye)), "kind": t["kind"]}
    if npos < CFG["min_n"] or nneg < CFG["min_n"]:
        res["skipped"] = "too few samples"
        return res
    folds = C.folds_for(groups)
    hb = head_blocks(unit, parts)
    if t.get("lofo"):
        return lofo_eval(unit, t, hb, folds, bundles, res)
    train_ok = t.get("train_ok")
    ph = C.crossfit(hb, y, folds, train_ok)[em]
    res["head_only"] = summarise(ye, ph, t["kind"])
    res["univariate"] = univariate_block(unit, parts, em, ye)
    pix = pix_blocks(unit, parts)
    php = None
    if pix is not None:
        php = C.crossfit(hb + pix, y, folds, train_ok)[em]
        res["head_pix"] = summarise(ye, php, t["kind"])
        res["pixel_only"] = summarise(ye, C.crossfit(pix, y, folds, train_ok)[em], t["kind"])
    res["joint"] = {}
    ce_h = C.ce_bits(ye, ph)
    for bname in bundles:
        gb, nbad = sig_blocks(unit, parts, BUNDLES[bname])
        if gb is None:
            continue
        pj = C.crossfit(hb + gb, y, folds, train_ok)[em]
        r = summarise(ye, pj, t["kind"])
        key = f"{unit.uid}|{tname}|{bname}"
        d, se, p = C.delong_diff(ye, pj, ph)
        ca, ci = C.boot_ci(ye, ge, pj, ph, nboot, key)
        r.update({"dauc": r["auc"] - res["head_only"]["auc"], "dauc_ci": ca, "delong_se": se, "delong_p": p,
                  "dI_bits": float(ce_h.mean() - C.ce_bits(ye, pj).mean()), "dI_ci": ci, "n_nonfinite_fixed": nbad})
        r["call"] = C.head_additive_call(r["dauc"], ca, ci)
        th, tj = res["head_only"]["tpr_at_5fpr"], r["tpr_at_5fpr"]
        r["dtpr5"] = (tj - th) if (th is not None and tj is not None) else None
        if t["kind"] == "error":
            r["daurc"] = r["aurc"] - res["head_only"]["aurc"]
        if php is not None and bname in PIX_BUNDLES:
            pjp = C.crossfit(hb + pix + gb, y, folds, train_ok)[em]
            cap, cip = C.boot_ci(ye, ge, pjp, php, nboot, key + "|pix")
            r["dauc_pix"] = C.auc_y(ye, pjp) - C.auc_y(ye, php)
            r["dauc_pix_ci"] = cap
            r["dI_pix_bits"] = float(C.ce_bits(ye, php).mean() - C.ce_bits(ye, pjp).mean())
            r["dI_pix_ci"] = cip
            r["call_pix"] = call_pix(r["dauc_pix"], cap, cip)
        res["joint"][bname] = r
    return res


def lofo_eval(unit, t, hb, folds, bundles, res):
    """Leave-one-family-out: trained without family F's positives, scored on F's positives against the negatives."""
    y, fam = t["y"], t["fam"]
    res["lofo"] = {}
    fams = [f for f in FAMS if ((fam == f) & (y == 1)).any()]
    per = {}
    for bname in ["head_only"] + list(bundles):
        if bname == "head_only":
            blocks = hb
        else:
            gb, _ = sig_blocks(unit, t["parts"], BUNDLES[bname])
            if gb is None:
                continue
            blocks = hb + gb
        per[bname] = {}
        for f in fams:
            p = C.crossfit(blocks, y, folds, train_ok=~((fam == f) & (y == 1)))
            m = (y == 0) | (fam == f)
            per[bname][f] = C.auc_y(y[m], p[m])
    for bname, d in per.items():
        v = [x for x in d.values() if x is not None]
        res["lofo"][bname] = {"auc_by_family": d, "auc_mean": float(np.mean(v)) if v else None}
        hm = res["lofo"].get("head_only", {}).get("auc_mean")
        if bname != "head_only" and hm is not None and res["lofo"][bname]["auc_mean"] is not None:
            res["lofo"][bname]["dauc_mean"] = res["lofo"][bname]["auc_mean"] - hm
    return res


# =====================================================================================================================
# calibrated false alarms, X4, the train-referenced density (DO-3 replication)
# =====================================================================================================================
def calibration(unit):
    ct, cal = unit.caltab, unit.cal_idx
    tB = unit.tables["test"]
    B = local(unit, "test", unit.L["B"])
    out = {}
    for nm in univariate_names(unit):
        if nm not in tB:
            continue
        sg = SIGN[base(nm)]
        tau = C.cal_threshold(sg * C.finite_fix(ct[nm][cal])[0], CFG["alpha"])
        sb = sg * C.finite_fix(tB[nm][B])[0]
        k = int((sb > tau).sum())
        rec = {"tau": tau, "fpr_B": k / len(sb), "fpr_B_ci": C.wilson(k, len(sb)), "tpr": {}}
        for sp, tb in unit.tables.items():
            if sp != "test" and nm in tb:
                rec["tpr"][sp] = float((sg * C.finite_fix(tb[nm])[0] > tau).mean())
        out[nm] = rec
    return out


def x4_record(unit):
    tB = unit.tables["test"]
    B = local(unit, "test", unit.L["B"])
    pB = tB["x4_p"][B]
    k = int((pB <= CFG["alpha"]).sum())
    return {"taps": unit.x4t, "keys": unit.x4keys, "alpha": CFG["alpha"], "n_cal_tap": int(len(unit.x41_idx)),
            "n_cal_fusion": int(len(unit.x42_idx)), "n_B": int(len(pB)), "k_B": k, "fpr_B": k / len(pB),
            "fpr_B_ci": C.wilson(k, len(pB)), "rows_cal_tap": list(unit.L["X4_1"]),
            "rows_cal_fusion": list(unit.L["X4_2"]), "rows_B": list(unit.L["B"]),
            "fpr_curve": {f"{a:.2f}": float((pB <= a).mean()) for a in (0.02, 0.03, 0.04, 0.05)},   # rule 5 (D14)
            "tpr": {sp: float((tb["x4_p"] <= CFG["alpha"]).mean()) for sp, tb in unit.tables.items()
                    if sp != "test" and "x4_p" in tb}}


def do3_record(unit):
    """DO-3 (AH-1a replication): the fraction of the layout's clean test rows whose raw penult 10-NN distance exceeds
    the q95 of the reference's self-excluded 10-NN distances (atlas/invariants/density.py, all rows instead of 3000)."""
    tB = unit.tables["test"]
    B = local(unit, "test", unit.L["B"])
    r = tB["knn_penult"]
    return {"tau_q95": unit.do3_tau, "sparse_frac_test": float((r > unit.do3_tau).mean()),
            "fpr_B": float((r[B] > unit.do3_tau).mean()), "n_test": int(len(r)), "nc1_train": unit.nc1_train,
            "head_center_cos_train": unit.nc3_train,
            "law": {"a": 0.0224, "b": -0.108, "source": "results/anomaly_h1/SESSION.md:149 (AH-1)"}}


# =====================================================================================================================
# batch arm (X6)
# =====================================================================================================================
class BatchStats:
    def __init__(self, unit):
        u = unit
        ct, cal = u.caltab, u.cal_idx
        self.acc_A = u.cal["acc"]
        self.msp_A = ct["msp"][cal]
        self.negent_A = -ct["entropy"][cal]
        self.t_mc = float(np.quantile(self.msp_A, 1 - self.acc_A))           # ATC (Garg et al. 2022)
        self.t_ne = float(np.quantile(self.negent_A, 1 - self.acc_A))
        self.hist_A = u.cal["pred_hist"]
        self.probs_A_sorted = np.sort(u.cal["probs"], axis=0)
        self.C_inv = np.linalg.pinv(u.cal["bbse_C"])
        self.pi_A = project_simplex(self.C_inv @ self.hist_A)
        lA = ct["logr10_penult"][cal]
        self.medA = float(np.median(lA))
        self.scaleA = max(float(np.quantile(lA, 0.95) - np.quantile(lA, 0.5)), 1e-12)   # H recalibrated on CAL
        self.knn_tauA = float(np.quantile(ct["knn_penult"][cal], 0.95))
        self.bf = u.bframes

    def stats(self, tb, idx):
        """idx: (nb, n) local indices into table tb -> {statistic: (nb,)}."""
        nb, n = idx.shape
        msp = tb["msp"][idx]
        negent = -tb["entropy"][idx]
        out = {"AC": msp.mean(1), "DoC": self.acc_A - (self.msp_A.mean() - msp.mean(1)),
               "ATC_MC": (msp > self.t_mc).mean(1), "ATC_NE": (negent > self.t_ne).mean(1),
               "mean_gap": tb["gap"][idx].mean(1), "mean_energy": tb["energy"][idx].mean(1)}
        if "logr10_penult" in tb:
            out["H"] = (np.median(tb["logr10_penult"][idx], axis=1) - self.medA) / self.scaleA
            out["sparse"] = (tb["knn_penult"][idx] > self.knn_tauA).mean(1)
        am = tb["am"][idx]
        q = np.stack([np.bincount(r, minlength=K) for r in am]) / n
        pi = np.stack([project_simplex(self.C_inv @ qq) for qq in q])
        out["bbse_l1"] = np.abs(pi - self.pi_A).sum(1)
        ex = n * self.hist_A + 0.5
        out["bbsdh_chi2"] = (((q * n + 0.5) - ex) ** 2 / ex).sum(1)
        P = C.softmax(tb["logits"][idx.ravel()]).reshape(nb, n, -1)
        ks = np.array([[ks_stat_sorted(P[i, :, k], self.probs_A_sorted[:, k]) for k in range(K)] for i in range(nb)])
        pv = ks_pvalue(ks, n, len(self.probs_A_sorted))
        out["bbsds"] = -np.log10(np.clip(pv.min(1) * K, 1e-300, 1.0))
        for key in BATCH_KEYS:
            if f"bw_{key}" not in tb:
                continue
            Zb = tb[f"bw_{key}"][idx].mean(1)
            out[f"T2_{key}"] = n * (Zb ** 2).sum(1)
            out[f"disp_{key}"] = np.linalg.norm(Zb, axis=1)
            r = tb[f"br_{key}"][idx].mean(1) - pi @ self.bf[key]["mkV"]    # the BBSE-explained covariate residual
            out[f"R_{key}"] = n * (r * r).sum(1)
        if "bw_pix" in tb:
            out["T2_pix"] = n * (tb["bw_pix"][idx].mean(1) ** 2).sum(1)
        return out


ACC_EST = ("AC", "DoC", "ATC_MC", "ATC_NE")
MAPPED = ("H", "sparse", "disp_penult", "mean_gap", "mean_energy", "T2_pre", "T2_s2end")
HARM_SIGN = {"AC": -1, "DoC": -1, "ATC_MC": -1, "ATC_NE": -1, "mean_gap": -1, "mean_energy": -1, "H": 1, "sparse": 1,
             "disp_penult": 1, "T2_pre": 1, "T2_s2end": 1}
DETECTORS = ("H", "sparse", "T2_penult", "T2_pre", "T2_s2end", "R_pre", "R_s2end", "R_penult", "bbse_l1",
             "bbsdh_chi2", "bbsds", "T2_pix", "disp_penult")


def _rate(st, tau, d):
    return float((st[d] > tau[d]).mean())


def batch_arm(unit):
    bs = BatchStats(unit)
    T = unit.tables
    tB = T["test"]
    Bloc = local(unit, "test", unit.L["B"])
    ct, cal = unit.caltab, unit.cal_idx
    wrong_clean = tB["wrong"]
    pool = Bloc if unit.layout == "fit" else np.arange(len(tB["y"]))
    out = {"sizes": {}, "pool": "B" if unit.layout == "fit" else "eval test rows"}
    for n in CFG["batch_sizes"]:
        rec = {}
        calb = cal[batch_idx(f"cal|{n}", len(cal), n, CFG["n_batches"])]
        cst = bs.stats(ct, calb)
        tau = {d: C.cal_threshold(cst[d], 0.05) for d in DETECTORS if d in cst}
        rec["tau95_cal"] = tau
        cl = Bloc[batch_idx(f"cleanB|{n}", len(Bloc), n, CFG["n_batches"])]
        st = bs.stats(tB, cl)
        rec["clean_fpr"] = {d: _rate(st, tau, d) for d in tau}
        sh = {}
        pooled = {k: [] for k in ACC_EST + MAPPED + ("loss", "acc", "split")}
        for sp in [s for s in T if s.startswith("corrupt__")]:
            tb = T[sp]
            idx = batch_idx(f"shift|{sp}|{n}", len(tb["y"]), n, CFG["n_batches"])
            if idx is None:
                continue
            st = bs.stats(tb, idx)
            tl = unit.test_local(tb["rows"][idx])
            if (tl < 0).any():
                continue
            acc = 1.0 - tb["wrong"][idx].mean(1)
            loss = 100.0 * ((1.0 - wrong_clean[tl].mean(1)) - acc)
            rs = {"loss_mean": float(loss.mean()), "flag_rate": {d: _rate(st, tau, d) for d in tau if d in st},
                  "spearman_within": {k: C.spearman(HARM_SIGN[k] * st[k], loss) for k in ACC_EST + MAPPED
                                      if k in st}}
            if "H" in st:
                rs["partial_H_given_ATC"] = C.partial_spearman(st["H"], loss, st["ATC_MC"])
                rs["partial_H_given_ATC_AC"] = C.partial_spearman(st["H"], loss,
                                                                  np.column_stack([st["ATC_MC"], st["AC"]]))
            sh[sp] = rs
            if all(k in st for k in ACC_EST + MAPPED):
                for k in ACC_EST + MAPPED:
                    pooled[k].append(st[k])
                pooled["loss"].append(loss)
                pooled["acc"].append(acc)
                pooled["split"].append(np.full(len(loss), sp))
        rec["by_split"] = sh
        if pooled["loss"]:
            P = {k: np.concatenate(v) for k, v in pooled.items()}
            est = {}
            for k in ACC_EST:
                est[k] = {"mae_pp": float(100 * np.abs(P[k] - P["acc"]).mean())}
            loso = len(np.unique(P["split"])) >= 2
            for k in ACC_EST + MAPPED:                   # isotonic map to accuracy, fitted leave-one-split-out
                e = est.setdefault(k, {})
                e["spearman_pooled"] = C.spearman(HARM_SIGN[k] * P[k], P["loss"])
                wv = [v["spearman_within"].get(k) for v in sh.values()]
                wv = [x for x in wv if x is not None]
                e["spearman_within_mean"] = float(np.mean(wv)) if wv else None
                if not loso:
                    e["mae_pp_loso"] = None
                    continue
                pred = np.empty_like(P["acc"])
                for sp in np.unique(P["split"]):
                    m = P["split"] == sp
                    f = isotonic_fit(P[k][~m], P["acc"][~m], increasing=HARM_SIGN[k] < 0)
                    pred[m] = f(P[k][m])
                e["mae_pp_loso"] = float(100 * np.abs(pred - P["acc"]).mean())
            ph = [v.get("partial_H_given_ATC") for v in sh.values()]
            ph = [x for x in ph if x is not None]
            rec["partial_H_given_ATC_mean"] = float(np.mean(ph)) if ph else None
            rec["partial_H_given_ATC_frac_ge_0.3"] = float(np.mean(np.array(ph) >= 0.3)) if ph else None
            rec["estimators"] = est
        rec["skew"] = skew_batches(unit, bs, tB, pool, n, tau)
        if n == CFG["typing_n"]:
            rec["typing"] = typing(unit, bs, pool, n, tau)
        out["sizes"][str(n)] = rec
    return out


def _class_pools(pool, y):
    return [pool[y[pool] == k] for k in range(K)]


def _draw_by_label(rng, by, lab):
    """One local row per label, drawn with replacement from that class's pool."""
    out = np.empty(len(lab), dtype=np.int64)
    for c in np.unique(lab):
        m = lab == c
        out[m] = rng.choice(by[c], size=int(m.sum()), replace=True)
    return out


def skew_batches(unit, bs, tb, pool, n, tau):
    y = tb["y"]
    by = _class_pools(pool, y)
    out = {}
    single = {}
    for k in range(K):
        b = batch_idx(f"single|{k}|{n}", len(by[k]), n, CFG["n_single"])
        if b is None:
            continue
        st = bs.stats(tb, by[k][b])
        single[k] = {d: _rate(st, tau, d) for d in tau}
    out["single_class"] = ({d: {"fpr_mean": float(np.mean([v[d] for v in single.values()])),
                                "fpr_max": float(np.max([v[d] for v in single.values()]))} for d in tau}
                           if single else {})
    if any(len(b) == 0 for b in by):
        return out
    for a in CFG["dirichlet"]:
        rng = C.rng_for("t1_dirichlet", a, n)
        idx = [_draw_by_label(rng, by, rng.choice(K, size=n, p=rng.dirichlet(np.full(K, a))))
               for _ in range(CFG["n_batches"])]
        st = bs.stats(tb, np.stack(idx))
        out[f"dirichlet_{a}"] = {d: _rate(st, tau, d) for d in tau}
    for rho in CFG["markov"]:
        rng = C.rng_for("t1_markov", rho, n)
        idx = []
        for _ in range(CFG["n_batches"]):
            stay = rng.random(n) <= rho
            fresh = rng.integers(0, K, n)
            lab = np.empty(n, dtype=np.int64)
            c = int(fresh[0])
            for i in range(n):
                if i > 0 and not stay[i]:
                    c = int(fresh[i])
                lab[i] = c
            idx.append(_draw_by_label(rng, by, lab))
        st = bs.stats(tb, np.stack(idx))
        out[f"markov_{rho}"] = {d: _rate(st, tau, d) for d in tau}
    return out


def typing(unit, bs, pool, n, tau):
    """3-way typing of pure batches: covariate (discovery corruption s3), prior (clean, Dirichlet 0.1), novelty
    (CIFAR-100). Geometry rule (sensor change before the collapse, semantics at the penult): covariate if T2_s2end >
    tau95; else novelty if the fraction of penult per-tap conformal p <= 0.01 exceeds its clean tau95; else prior if
    bbse_l1 > tau95; else none. Head rule: covariate if bbsds > tau95; else novelty if the fraction of energy below the
    CAL 1% quantile exceeds its clean tau95; else prior if bbsdh_chi2 > tau95; else none."""
    T = unit.tables
    ct, cal = unit.caltab, unit.cal_idx
    need = ("T2_s2end", "bbse_l1", "bbsds", "bbsdh_chi2")
    if not all(d in tau for d in need) or "x4tap_penult" not in ct:
        return {"skipped": "missing statistics"}
    e1 = float(np.quantile(ct["energy"][cal], 0.01))
    lp = -math.log(CFG["nov_p"])
    calb = cal[batch_idx(f"typecal|{n}", len(cal), n, CFG["n_batches"])]
    nov_g_tau = C.cal_threshold((ct["x4tap_penult"][calb] >= lp).mean(1), 0.05)
    nov_h_tau = C.cal_threshold((ct["energy"][calb] < e1).mean(1), 0.05)
    y = T["test"]["y"]
    by = _class_pools(pool, y)
    sets = {"covariate": [(C.csplit(c, 3), batch_idx(f"typecov|{c}|{n}", len(T[C.csplit(c, 3)]["y"]), n,
                                                     CFG["typing_cov_batches"]))
                          for c in C.DISC if C.csplit(c, 3) in T]}
    if all(len(b) for b in by):
        rng = C.rng_for("t1_typing_prior", n)
        sets["prior"] = [("test", np.stack([_draw_by_label(rng, by, rng.choice(K, size=n, p=rng.dirichlet(
            np.full(K, 0.1)))) for _ in range(CFG["typing_batches"])]))]
    if "ood__cifar100" in T:
        sets["novelty"] = [("ood__cifar100", batch_idx(f"typenov|{n}", len(T["ood__cifar100"]["y"]), n,
                                                       CFG["typing_batches"]))]
    res = {}
    for truth, lst in sets.items():
        g_ok = h_ok = tot = 0
        for sp, idx in lst:
            if idx is None:
                continue
            tb = T[sp]
            st = bs.stats(tb, idx)
            if not all(d in st for d in need):
                continue
            nov_g = (tb["x4tap_penult"][idx] >= lp).mean(1) > nov_g_tau
            nov_h = (tb["energy"][idx] < e1).mean(1) > nov_h_tau
            gcall = np.where(st["T2_s2end"] > tau["T2_s2end"], "covariate",
                             np.where(nov_g, "novelty", np.where(st["bbse_l1"] > tau["bbse_l1"], "prior", "none")))
            hcall = np.where(st["bbsds"] > tau["bbsds"], "covariate",
                             np.where(nov_h, "novelty", np.where(st["bbsdh_chi2"] > tau["bbsdh_chi2"], "prior", "none")))
            g_ok += int((gcall == truth).sum())
            h_ok += int((hcall == truth).sum())
            tot += len(idx)
        res[truth] = {"geometry_acc": g_ok / tot if tot else None, "head_acc": h_ok / tot if tot else None, "n": tot}
    ok = [v for v in res.values() if v["n"]]
    res["macro"] = {"geometry_acc": float(np.mean([v["geometry_acc"] for v in ok])) if len(ok) == 3 else None,
                    "head_acc": float(np.mean([v["head_acc"] for v in ok])) if len(ok) == 3 else None}
    return res


# =====================================================================================================================
# per-model outcomes for T2 (experiments/b4/model_outcomes.schema.json; fit layout only)
# =====================================================================================================================
def o5_record(unit):
    """O5: mean over the 10 discovery corruptions at s3 of AUROC(X4, scored -p) minus the mean AUROC of the best head
    statistic (the head statistic with the largest mean AUROC over the same corruptions); positives the corrupt rows
    POS, negatives the clean rows B."""
    L = LAYOUT["fit"]
    tB = unit.tables["test"]
    B = local(unit, "test", L["B"])
    per, missing = {}, []
    for c in C.DISC:
        sp = C.csplit(c, 3)
        if sp not in unit.tables or "x4" not in unit.tables[sp]:
            per[c] = {"x4_auroc": None, "head_auroc": {s: None for s in O5_HEADS}}
            missing.append(sp)
            continue
        tb = unit.tables[sp]
        P = local(unit, sp, L["POS"])
        per[c] = {"x4_auroc": C.auc(tb["x4"][P], tB["x4"][B]),
                  "head_auroc": {s: C.auc(SIGN[s] * tb[s][P], SIGN[s] * tB[s][B]) for s in O5_HEADS}}
    rec = {"severity": 3, "corruptions": list(C.DISC), "negatives": list(L["B"]), "per_corruption": per,
           "value": None, "x4_auroc_mean": None, "best_head": O5_HEADS[0], "best_head_auroc_mean": None}
    ok = [c for c in C.DISC if per[c]["x4_auroc"] is not None and all(v is not None for v in per[c]["head_auroc"].values())]
    if len(ok) != len(C.DISC):
        rec["missing"] = missing
        return rec
    hm = {s: float(np.mean([per[c]["head_auroc"][s] for c in ok])) for s in O5_HEADS}
    best = max(O5_HEADS, key=lambda s: (hm[s], -O5_HEADS.index(s)))
    xm = float(np.mean([per[c]["x4_auroc"] for c in ok]))
    rec.update(value=xm - hm[best], x4_auroc_mean=xm, best_head=best, best_head_auroc_mean=hm[best],
               head_auroc_mean=hm)
    return rec


def model_outcomes(unit, board, nboot):
    ec = board["persample"].get("err_clean") or {}
    j = (ec.get("joint") or {}).get("penult") or {}
    o1 = {"bundle": "penult", "bundle_members": list(O1_BUNDLE), "target": "err_clean",
          "rows": list(LAYOUT["fit"]["test"]), "n_pos": int(ec.get("n_pos") or 0), "n_neg": int(ec.get("n_neg") or 0),
          "nboot": int(nboot)}
    if "dauc" in j:
        o1.update(value=j["dauc"], ci95=list(j["dauc_ci"]), dI_bits=j["dI_bits"], dI_ci95=list(j["dI_ci"]),
                  call=j["call"], auc_head=(ec.get("head_only") or {}).get("auc"), auc_joint=j["auc"])
    else:
        o1.update(value=None, ci95=[None, None], dI_bits=None, dI_ci95=[None, None], call="INCONCLUSIVE",
                  error=ec.get("error") or ec.get("skipped") or "penult bundle not evaluated")
    rec = {"schema": OUTCOMES_SCHEMA, "unit": unit.uid, "layout": "fit", "phase": unit.phase, "O1": o1,
           "O5": o5_record(unit)}
    try:
        with open(OUTCOMES_SCHEMA_PATH) as f:
            errs = validate_schema(C._clean(rec), json.load(f))
    except OSError as e:
        errs = [f"schema unreadable: {e}"]
    rec["schema_errors"] = errs
    return rec


# =====================================================================================================================
# functional cost (D14): reference bytes; CPU ms per 1000 queries under cost.timing_s (ignored by the replay compare)
# =====================================================================================================================
def cost_block(unit):
    ref_bytes = {}
    for key in TAPKEYS:
        for k, v in unit.refs[unit.ft[key]].nbytes().items():
            ref_bytes[f"{k}_{key}"] = v
    ref_bytes["x4"] = int(sum(unit.refs[t].n * unit.refs[t].d * 4 for t in unit.x4t)
                          + 8 * (len(unit.x41_idx) * len(unit.x4t) + len(unit.x42_idx)))
    ref_bytes["head"] = int((unit.W.size + unit.b.size) * 4)
    nq = min(CFG["n_cost"], len(unit.caltab["y"]))
    fd = unit.fitd
    z = fd.logits("test")[:nq]
    am = z.argmax(1)
    ms = {}

    def clock(name, fn):
        t = time.perf_counter()
        fn()
        ms[name] = round(1000.0 * (time.perf_counter() - t) * 1000.0 / max(1, nq), 3)
    clock("head_stats", lambda: C.head_stats(z))
    Q = {key: fd.acts(unit.ft[key], "test")[:nq] for key in TAPKEYS}
    for key in TAPKEYS:
        tr = unit.refs[unit.ft[key]]
        clock(f"knnL2_{key}", lambda tr=tr, key=key: C.knn(tr.Xn, C.l2n(Q[key]), CFG["k_knn"]))
        clock(f"centres_{key}", lambda tr=tr, key=key: NC.center_dists(Q[key], tr.C))
    tp = unit.refs[unit.ft["penult"]]
    clock("maha_penult", lambda: tp.maha(Q["penult"]))
    clock("trust_penult", lambda: tp.trust(Q["penult"], am))

    def x4():
        P = []
        for t, k in zip(unit.x4t, unit.x4keys):
            d = C.knn(unit.refs[t].Xn, C.l2n(Q[k]), CFG["k_knn"])[0][:, -1]
            P.append(C.conformal_p(unit.caltab[f"knnL2_{k}"][unit.x41_idx], d))
        C.conformal_p(unit.x4_cal, C.cauchy_stat(np.column_stack(P)))
    clock("x4", x4)
    return {"ref_bytes": ref_bytes, "n_queries": int(nq), "timing_s": {"ms_per_1000_queries": ms}}


# =====================================================================================================================
# one scoreboard
# =====================================================================================================================
def run_board(args, nboot=None, bundles=None, quiet=False):
    """The probe (D5 CLI contract). Returns (path, board, unit)."""
    run = C.ProbeRun("t1", __file__, args)               # the output guard runs before any read
    reg = C.load_registry(args.registry)
    unit = Unit(reg, args.unit, args.layout, args.phase, run.timed, "board", quiet)
    nb = run.nboot if nboot is None else int(nboot)
    with run.timed("block:build"):
        unit.build()
    spec = unit.spec
    board = {"schema": SCHEMA,
             "unit_info": {"id": unit.uid, "roles": list(spec.get("roles", [])), "family": spec.get("family"),
                           "arch": spec.get("arch"), "penult_dim": unit.refs[unit.ft["penult"]].d,
                           "functional_taps": unit.ft, "x4_taps": unit.x4t, "taps_trajectory": unit.taps_all,
                           "splits": sorted(unit.tables)},
             "rows": LAYOUT[unit.layout],
             "constants": {"cfg": CFG, "bundles": {k: list(v) for k, v in BUNDLES.items()}, "sign": SIGN,
                           "pix_bundles": list(PIX_BUNDLES), "nboot_used": nb,
                           "bundles_used": list(BUNDLES) if bundles is None else list(bundles)},
             "head": {"best_stat": unit.best_head, "aurc_headsel": unit.aurc, "rows_headsel": list(unit.L["HEAD_SEL"]),
                      "temperature": unit.T, "temperature_headsel": unit.T_sel, "acc_cal": unit.cal["acc"],
                      "n_cal": unit.cal["n"],
                      "rows_cal": list(unit.L["CAL"])},
             "persample": {}, "errors": {}}
    for name, t in targets(unit).items():
        with run.timed(f"target:{name}"):
            try:
                board["persample"][name] = joint_eval(unit, name, t, nb, bundles)
            except Exception as e:                          # one failing target never loses the others (ReadRefused
                board["persample"][name] = {"error": f"{type(e).__name__}: {e}"}     # is a SystemExit: not caught)
                board["errors"][f"target:{name}"] = f"{type(e).__name__}: {e}"
        unit.say(f"target {name}")
    for blk, fn in (("calibration", calibration), ("x4", x4_record), ("do3", do3_record), ("batch", batch_arm),
                    ("cost", cost_block)):
        with run.timed(f"block:{blk}"):
            try:
                board[blk] = fn(unit)
            except Exception as e:
                board[blk] = {"error": f"{type(e).__name__}: {e}"}
                board["errors"][f"block:{blk}"] = f"{type(e).__name__}: {e}"
    if unit.layout == "fit":
        with run.timed("block:model_outcomes"):
            board["model_outcomes"] = model_outcomes(unit, board, nb)
    path = run.finish(board, dumps=unit.dumps())
    unit.say(f"wrote {path}")
    return path, board, unit


# =====================================================================================================================
# synthetic unit (known answers) and the self-test
# =====================================================================================================================
SYN_TAPS = ("stem", "layer1.0", "layer2.0", "layer3.0", "layer3.1", "penult")
SYN_FUNC = {"stem": "stem", "s1end": "layer1.0", "s2end": "layer2.0", "pre": "layer3.0", "penult": "penult"}
SYN_X4 = ["stem", "layer3.0", "penult"]
SYN_HOLDOUT = ("impulse_noise", "frost")
SYN_EXTRA = ("saturate",)
SYN_ROWS = {"fit": {"test": (0, 5000), "c10c": (0, 600), "ood": (0, 600), "faults": (3500, 5000)},
            "eval": {"test": (5000, 10000), "c10c": (5000, 5400), "ood": (2000, 2400), "faults": (8500, 10000)}}


def _synth_model(D=16):
    Cm = np.zeros((K, D))
    for k in range(9):
        Cm[k, k] = 3.0
    return Cm, Cm.copy(), -0.5 * (Cm * Cm).sum(1)


def make_synth(root, uid="t1_synth", n_ref=4000, seed=0):
    """A synthetic 'ResNet' unit in the batch-4 dump format (fit and eval layouts, both open) and its registry.
    D = 16 at every tap; class means 3 e_k (k < 9) and 0; penult = C[y] + 1.1 eps; logits z = C x - |c|^2 / 2 (W = C),
    so the head is the Bayes nearest-centre rule and its sorted logits carry the full posterior. layer3.1 is a copy of
    the penult (dup_of_penult), layer3.0 (pre) = penult + 0.3 noise, layer2.0 / layer1.0 carry 0.8 / 0.5 of the class
    means, the stem is class-free 3 e15 + N(0, 0.3^2) (a fixed mean direction, 2.5x the 16-d noise norm: an isotropic
    or noise-dominated stem makes an L2-normalised kNN blind to a stem shift; S0 measured X4 AUROC 0.53 with e15 +
    N(0, 0.3^2) and a 1.5 shift). Planted effects (selftest() known answers):
      N family (gaussian_noise, shot_noise, holdout impulse_noise) at s3 shifts ONLY the stem by 3.0 (10 SD) along the
        all-ones direction with its e15 part removed (orthogonal to the stem mean): invisible to the head, visible to
        the early bundle and to X4 (the stem is an X4 tap);
      fog at s3 shifts the penult (and pre) by 3.0 along e13, a direction of the head's NULL space (the head is blind;
        the head-null residual sees it);
      every fault shifts the stem by 0.8 * level along e14; OOD rows are N(0, 3^2) (cifar100) and N(4, 3^2) (svhn);
      the 13 pixel factors are independent noise (the early bundle must add beyond them)."""
    rng = np.random.default_rng(seed)
    Cm, W, b = _synth_model()
    D = Cm.shape[1]
    ones = np.ones(D)
    ones[15] = 0.0
    ones /= np.linalg.norm(ones)                        # unit, orthogonal to the stem mean e15
    e13, e14, e15 = np.eye(D)[13], np.eye(D)[14], np.eye(D)[15]
    dumps = {}
    for layout, R in SYN_ROWS.items():
        d = os.path.join(root, f"b4d_{uid}" + ("" if layout == "fit" else "_eval"), "dump")
        dumps[layout] = d
        for sub in ("acts", "logits", "labels", "rows", "preds", "factors", "head"):
            os.makedirs(os.path.join(d, sub), exist_ok=True)
        np.savez(os.path.join(d, "head", "head.npz"), W=W.astype(np.float32), b=b.astype(np.float32))
        splits = {}
        lo, hi = R["test"]
        n = hi - lo
        yt = rng.integers(0, K, n)
        base_t = {"penult": Cm[yt] + 1.1 * rng.standard_normal((n, D))}
        base_t["pre"] = base_t["penult"] + 0.3 * rng.standard_normal((n, D))
        base_t["l2"] = 0.8 * Cm[yt] + rng.standard_normal((n, D))
        base_t["l1"] = 0.5 * Cm[yt] + rng.standard_normal((n, D))
        base_t["stem"] = 3.0 * e15 + 0.3 * rng.standard_normal((n, D))
        rows_t = np.arange(lo, hi)
        if layout == "fit":
            yr = rng.integers(0, K, n_ref)
            pr = Cm[yr] + 1.1 * rng.standard_normal((n_ref, D))
            splits["ref"] = ({"penult": pr, "pre": pr + 0.3 * rng.standard_normal((n_ref, D)),
                              "l2": 0.8 * Cm[yr] + rng.standard_normal((n_ref, D)),
                              "l1": 0.5 * Cm[yr] + rng.standard_normal((n_ref, D)),
                              "stem": 3.0 * e15 + 0.3 * rng.standard_normal((n_ref, D))}, yr, np.arange(n_ref), True, None)
        splits["test"] = (base_t, yt, rows_t, True, None)

        def paired(r0, r1, shift):
            ix = np.arange(r0 - lo, r1 - lo)
            a = {k: v[ix].copy() for k, v in base_t.items()}
            for k, v in shift.items():
                a[k] = a[k] + v
            return a, yt[ix], rows_t[ix]
        c0, c1 = R["c10c"]
        sevs = {"fit": {"disc": (1, 3, 5), "hold": (1, 3, 5), "extra": (3, 5)},
                "eval": {"disc": (1, 3, 5), "hold": (3, 5), "extra": (3, 5)}}[layout]
        for grp, names in (("disc", C.DISC), ("hold", SYN_HOLDOUT), ("extra", SYN_EXTRA)):
            for c in names:
                for s in sevs[grp]:
                    sh = {}
                    if C.FAMILY[c] == "N" and s == 3:
                        sh = {"stem": 3.0 * ones}
                    elif c == "fog" and s == 3:
                        sh = {"penult": 3.0 * e13, "pre": 3.0 * e13}
                    a, yy, rr = paired(c0, c1, sh)
                    splits[C.csplit(c, s)] = (a, yy, rr, False, SYN_X4 if grp != "disc" else None)
        o0, o1 = R["ood"]
        for nm, mu in (("cifar100", 0.0), ("svhn", 4.0)):
            m = o1 - o0
            a = {k: mu + 3.0 * rng.standard_normal((m, D)) for k in ("penult", "pre", "l2", "l1")}
            a["stem"] = mu + 3.0 * rng.standard_normal((m, D))
            splits[f"ood__{nm}"] = (a, np.zeros(m, dtype=np.int64), np.arange(o0, o1), True, None)
        f0, f1 = R["faults"]
        for kind in FAULT_KINDS:
            for lv in FAULT_LEVELS:
                a, yy, rr = paired(f0, f1, {"stem": 0.8 * lv * e14})
                splits[C.fsplit(kind, lv)] = (a, yy, rr, False, None)
        amap = {"stem": "stem", "layer1.0": "l1", "layer2.0": "l2", "layer3.0": "pre", "layer3.1": "penult",
                "penult": "penult"}
        for sp, (a, yy, rr, pen32, only) in splits.items():
            for tap in SYN_TAPS:
                if only is not None and tap not in only:
                    continue
                os.makedirs(os.path.join(d, "acts", tap), exist_ok=True)
                dt = np.float32 if (tap == "penult" and pen32) else np.float16
                np.save(os.path.join(d, "acts", tap, f"{sp}.npy"), a[amap[tap]].astype(dt))
            z = a["penult"] @ W.T + b
            np.save(os.path.join(d, "logits", f"{sp}.npy"), z.astype(np.float32))
            np.save(os.path.join(d, "labels", f"{sp}.npy"), np.asarray(yy, dtype=np.int64))
            np.save(os.path.join(d, "rows", f"{sp}.npy"), np.asarray(rr, dtype=np.int64))
            P = C.softmax(z)
            np.savez(os.path.join(d, "preds", f"{sp}.npz"), argmax=P.argmax(1), maxprob=P.max(1).astype(np.float32))
            if sp != "ref":
                np.savez(os.path.join(d, "factors", f"{sp}.npz"),
                         **{f"f{j:02d}": rng.standard_normal(len(yy)).astype(np.float32) for j in range(13)})
        meta = {"source": "synthetic", "arch": "synthetic_resnet", "layers": list(SYN_TAPS), "splits": list(splits),
                "n_classes": K, "n_test": n, "dims": {t: D for t in SYN_TAPS},
                "b4": {"schema": "b4_dump/1", "unit": uid, "layout": layout, "sealed": False, "roles": ["D"],
                       "taps": {"all": list(SYN_TAPS), "spatial": {t: t != "penult" for t in SYN_TAPS[:-1]},
                                "depth_frac": {"stem": 0.05, "layer1.0": 0.3, "layer2.0": 0.6, "layer3.0": 0.8,
                                               "layer3.1": 0.95, "penult": 1.0},
                                "dup_of_penult": ["layer3.1"], "pre": "layer3.0", "functional": dict(SYN_FUNC),
                                "x4": list(SYN_X4)},
                       "missing": []}}
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(meta, f)
    reg = {"schema": "b4_models/1", "models": [
        {"id": uid, "roles": ["D"], "arch": "synthetic_resnet", "family": "resnet", "penult": 16, "source": "synthetic",
         "layouts": {"fit": {"dump": dumps["fit"], "sealed": False}, "eval": {"dump": dumps["eval"], "sealed": False}}}]}
    rp = os.path.join(root, "models.json")
    with open(rp, "w") as f:
        json.dump(reg, f)
    return rp, uid


def selftest(workdir=None, fast=True, layouts=("fit", "eval"), quiet=True):
    """Known answers on the synthetic unit of make_synth. fast: nboot 50, 40 batches, the FAST_BUNDLES subset (the same
    code and definitions; only the counts shrink) so that it fits the pytest budget."""
    checks = []

    def ck(name, ok, detail=""):
        checks.append({"name": name, "pass": bool(ok), "detail": str(detail)[:300]})
        say(f"[selftest] {'PASS' if ok else 'FAIL'} {name} ({str(detail)[:120]})", quiet and ok)
    saved = dict(CFG)
    if fast:
        CFG.update({"n_batches": 40, "n_single": 10, "typing_batches": 40, "typing_cov_batches": 8})
    nboot, bundles = (50, FAST_BUNDLES) if fast else (None, None)
    tmp = workdir or tempfile.mkdtemp(prefix="t1_scoreboard_selftest_")
    try:
        rp, uid = make_synth(os.path.join(tmp, "synth"))
        mk = lambda layout, phase, tag="": argparse.Namespace(         # noqa: E731
            registry=rp, unit=uid, phase=phase, layout=layout, out=os.path.join(tmp, "out", f"{uid}_{layout}{tag}"))
        reg = C.load_registry(rp)
        u0 = Unit(reg, uid, "fit", "discovery", quiet=True)
        pl = u0.plan()
        ck("discovery plan holds no confirmation-only split (holdout, extras, exposure_global)",
           not any(C.is_confirmation_only(s) for s in pl) and "corrupt__fog__s3" in pl, [s for s in pl][:4])
        fit_board = None
        if "fit" in layouts:
            _, bd, unit = run_board(mk("fit", "confirmation"), nboot=nboot, bundles=bundles, quiet=quiet)
            fit_board = bd
            ps = bd["persample"]
            jm = ps["err_clean"]["joint"]["margin_pen"]["dauc"]
            ck("margin adds ~0 over the Bayes head on clean errors (|dAUC| <= 0.015)", abs(jm) <= 0.015, jm)
            o1 = bd["model_outcomes"]["O1"]
            ck("O1: the penult bundle does not ADD over the Bayes head (value <= 0.01, call != ADDS)",
               o1["value"] is not None and o1["value"] <= 0.01 and o1["call"] != "ADDS", o1)
            t = targets(unit)["err_clean"]
            hi = C.head_increment(head_blocks(unit, t["parts"]), sig_blocks(unit, t["parts"], O1_BUNDLE)[0], t["y"],
                                  t["groups"], nboot if nboot else 200, f"{uid}|err_clean|penult")
            ck("O1 equals atlas/b4_core.head_increment exactly (value, CIs, call)",
               hi["dauc"] == o1["value"] and hi["dauc_ci"] == o1["ci95"] and hi["dI_ci"] == o1["dI_ci95"]
               and hi["call"] == o1["call"], (hi.get("dauc"), o1["value"]))
            je = ps["corrupt_N_s3"]["joint"]["early"]
            ck("stem-only shift: the early bundle adds >= 0.3 AUROC over the head", je["dauc"] >= 0.3, je["dauc"])
            jp = ps["corrupt_N_s3"]["joint"]["margin_pen"]["dauc"]
            ck("stem-only shift: the penult margin adds ~0 (|dAUC| <= 0.03)", abs(jp) <= 0.03, jp)
            ck("pixel twin: the early bundle adds beyond uninformative pixel factors (dI_pix > 0.2 bits, ADDS)",
               (je.get("dI_pix_bits") or 0) > 0.2 and je.get("call_pix") == "ADDS", je.get("dI_pix_bits"))
            jn = ps["corrupt_W_s3"]["joint"]["null_pen"]
            ck("head-null shift (fog s3 along e13): head blind (AUC <= 0.6), null residual adds >= 0.1",
               ps["corrupt_W_s3"]["head_only"]["auc"] <= 0.6 and jn["dauc"] >= 0.1,
               (ps["corrupt_W_s3"]["head_only"]["auc"], jn["dauc"]))
            jt = ps["corrupt_holdout_s3_transfer"]["joint"]["x4"]["dauc"]
            ck("X4 transfers to an unseen family (impulse_noise shifts the stem): dAUC >= 0.1", jt >= 0.1, jt)
            x4 = bd["x4"]
            ck("X4 clean FPR at alpha 0.05 on B (n_cal 750) within [0.02, 0.09]", 0.02 <= x4["fpr_B"] <= 0.09
               and x4["n_cal_fusion"] == 750 and x4["n_B"] == 1500, x4["fpr_B"])
            o5 = bd["model_outcomes"]["O5"]
            ck("O5: X4 sees the stem-only noise shift (gaussian x4 AUROC >= 0.9, head <= 0.6), value > 0",
               o5["per_corruption"]["gaussian_noise"]["x4_auroc"] >= 0.9
               and max(o5["per_corruption"]["gaussian_noise"]["head_auroc"].values()) <= 0.6 and o5["value"] > 0, o5)
            ck("model_outcomes validates against experiments/b4/model_outcomes.schema.json",
               bd["model_outcomes"]["schema_errors"] == [], bd["model_outcomes"]["schema_errors"][:3])
            ck("head best statistic chosen", bd["head"]["best_stat"] in HEAD_STATS, bd["head"]["best_stat"])
            b64 = (bd.get("batch") or {}).get("sizes", {}).get("64", {})
            ck("batch arm ran (estimators, skew, typing, by_split)", all(k in b64 for k in
                                                                        ("estimators", "skew", "typing", "by_split")),
               sorted(b64)[:6])
            dl = b64.get("by_split", {}).get("corrupt__gaussian_noise__s3", {}).get("flag_rate", {})
            ck("batch arm: every detector has a CAL-calibrated flag rate on a corrupt split",
               all(isinstance(dl.get(k), float) for k in ("H", "sparse", "T2_penult", "R_pre", "bbsds")), dl)
            ck("DO-3 record: nc1 and sparse fraction are numbers", isinstance(bd["do3"].get("nc1_train"), float)
               and isinstance(bd["do3"].get("sparse_frac_test"), float), bd["do3"])
            ck("no target or block failed", not bd["errors"], bd["errors"])
            try:
                run_board(mk("fit", "confirmation"), nboot=nboot, bundles=bundles, quiet=True)
                ck("refuses an existing output (append-only)", False)
            except C.ReadRefused:
                ck("refuses an existing output (append-only)", True)
            try:
                run_board(mk("fit", "confirmation", "_r2"), nboot=nboot, bundles=bundles, quiet=True)
                ck("confirmation: a unit is scored once under any tag", False)
            except C.ReadRefused:
                ck("confirmation: a unit is scored once under any tag", True)
        if "eval" in layouts:
            _, be, _ = run_board(mk("eval", "confirmation"), nboot=nboot, bundles=bundles, quiet=quiet)
            x4 = be["x4"]
            ck("eval layout: X4 calibrated on fit rows 0-2499 / 2500-4999, FPR on B_eval within [0.02, 0.09]",
               0.02 <= x4["fpr_B"] <= 0.09 and x4["n_cal_fusion"] == 2500 and x4["n_B"] == 1500, x4["fpr_B"])
            if fit_board is not None:
                ck("eval layout reproduces the fit run's best head statistic", be["head"]["best_stat"]
                   == fit_board["head"]["best_stat"], (be["head"]["best_stat"], fit_board["head"]["best_stat"]))
            ck("eval layout writes no model_outcomes", "model_outcomes" not in be)
            ck("eval layout: stem-only shift still read by the early bundle (dAUC >= 0.3)",
               be["persample"]["corrupt_N_s3"]["joint"]["early"]["dauc"] >= 0.3,
               be["persample"]["corrupt_N_s3"]["joint"]["early"]["dauc"])
    except Exception as e:                                            # a crash is a failed self-test
        import traceback
        ck("self-test ran without an exception", False, f"{type(e).__name__}: {e} {traceback.format_exc()[-600:]}")
    finally:
        CFG.clear()
        CFG.update(saved)
        if workdir is None:
            shutil.rmtree(tmp, ignore_errors=True)
    bad = [c["name"] for c in checks if not c["pass"]]
    return {"status": "PASS" if not bad else "FAIL", "failed": bad, "checks": checks, "fast": bool(fast),
            "layouts": list(layouts), "code_sha256": C.code_sha256(os.path.abspath(__file__)),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def main(argv=None):
    ap = C.probe_argparser("t1", layout=True, description="T1 head-conditional scoreboard (docs/plans/T1_SCOREBOARD.md)")
    a = C.parse_probe_args(ap, argv)
    if a.selftest:
        if a.selftest_out and os.path.exists(a.selftest_out):
            print(f"[t1] {a.selftest_out} exists: not overwritten", file=sys.stderr)
            return 2
        rep = selftest(quiet=False)
        C.write_selftest(a.selftest_out, rep)
        print(f"[t1] selftest {rep['status']} ({len(rep['checks'])} checks)")
        return 0 if rep["status"] == "PASS" else 1
    run_board(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
