#!/usr/bin/env python3
"""
t3s_spatial_probe.py -- batch-4 lane S (X5 "beyond GAP") on the maps dumps (docs/plans/T3S_SPATIAL.md; docs/plans/
B4_INTEGRATION.md D1, D3, D4, D5, D7, D14). Owner: T3S. numpy + the batch-4 core only (no torch / scipy / sklearn): it runs
in the CPU lane of S1 (Sdisc units, discovery) and of S2 (Sconf units, confirmation: the sealed dumps opened once, after P2).

  python scripts/t3s_spatial_probe.py --registry experiments/b4/models.json --unit <id> --phase discovery|confirmation \
      --out results/b4_t3s/<id><tag>
  python scripts/t3s_spatial_probe.py --selftest [--selftest-out <json>]     (planted synthetic dumps, no torch, no volume)

Question: what can a controller read from the head-input MAP (and its spatial std) that the pooled head cannot see? A local
sensor fault (pasted CIFAR-10-C s5 pixels, occluder, glare, dead-pixel cluster; soiling held out) changes a few positions
of the 8 x 8 map; global average pooling dilutes it by its area before the head reads it.

Material (maps layout of scripts/b4_extract.py; every file read through atlas.b4_core.open_unit, the D7 seal):
  ref    every 2nd row of the atlas train reference draw (5000)   -> every per-image model is FITTED here
  cal    CIFAR-10 test rows 7500-8499 (clean)                    -> every threshold is CALIBRATED here (D4, D14)
  eval   test rows 8500-9499 (clean twins)                        -> the negatives of every AUROC
  fault__<cond>   atlas/faults.s_conditions on rows 8500-9499 (soiling only on Sconf units: confirmation-only, D7)
  global__<c>__s3 CIFAR-10-C s3 rows 8500-9499 (whole-image corruptions; local-vs-global typing)
Stored per split: acts/penult (= GAP of the head map), acts/headmap_std (spatial std of the head map), maps/headmap
(N, C, 8, 8), maps/stage2map (N, C, 16, 16; the two hubs only), logits (stored float32: the canonical head path, D2),
labels, rows, pixels (uint8) and masks (fault splits).

Per-image scores (orientation: larger = more anomalous):
  map_max, map_top, map_area, map_gini   PaDiM (Defard et al. 2020) on the head map: a Gaussian per position fitted on ref,
        Sigma_p = (1 - s) S_p + s tr(S_p)/C I (s = 0.1), squared Mahalanobis per position; the image scores are the
        maximum, the mean of the top 5% of positions, the fraction of positions above that position's cal q99, and the
        Gini of the position scores. map_max is THE frozen primary statistic.
  s2_map_max, s2_map_top                 the same on the stage-2 map (hubs only; INFO)
  std_half                               X5 arm (a): Gaussian (same shrinkage) on the 64-d spatial std of the head map
  d1, knn_l2, gap_maha                   the pooled vector (penult): nearest ref class centre; 10th-NN distance of the
                                         L2-normalised penult (Sun et al. 2022); Gaussian (same shrinkage) on the penult
  msp, maxlogit, gap, energy, entropy    the head, from the stored logits, oriented by atlas.b4_core.ERR_SIGN
  pix_padim, pix_sat, pix_lap_dev        pixels (no network): PaDiM over 4 x 4-pixel cells of [mean RGB, std RGB, log
                                         Laplacian variance] fitted on the ref pixels (max over cells); fraction of
                                         saturated / black pixels; |log Laplacian variance - its cal median|
Per condition (fault vs its clean twin, the same rows): AUROC of every score; TPR at the cal-calibrated 5% threshold;
pointing (the map-max position lies on a cell that holds a fault pixel; chance = mean fraction of such cells; also the
one-cell dilation, the pixel baseline and the stage-2 map); the "output still right" subset (fault argmax = clean argmax
= label); validity (rows, labels and the outside of the fault region equal to the clean twin; >= 95% of images changed).
Cross-fitted increments with the frozen HEAD-ADDITIVE rule (atlas.b4_core.head_increment; the head block is T1's: sorted
logits + splines of gap, max-logit, energy, entropy; folds and bootstrap by image row):
  SP2_map_over_pooled   map bundle over head + pooled geometry, F5 (a12) faults vs clean
  SP3_map_over_pixels   map bundle over head + pooled + pixel scores, the 4 paste families at a12 vs clean
  SP7_std_over_pooled   std_half over head + pooled, F5 vs clean
  SP8_typing            map bundle over head + pooled, local a12 faults vs global s3 corruptions
  INFO_*                soiling (Sconf), std + map, stage-2 over the head map (hubs)
Known answers written for the evaluator (scripts/t3s_eval.js decides): map-GAP identity (mean over positions of the
stored float16 map = stored penult; unbiased spatial std = stored headmap_std), pairing, mask areas, pixel hashes
(= meta.b4.image_sha256_first16), the clean false-alarm rate of every calibrated flag on eval.
The labels are applied by scripts/t3s_eval.js only; this probe writes numbers.
"""
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atlas import b4_core as C          # noqa: E402  numpy + stdlib only
from atlas import b4_collapse as NC     # noqa: E402  numpy only
from atlas import faults as FAULTS      # noqa: E402  numpy only

SCHEMA = "t3s_probe/1"
LAYOUT = "maps"
SHRINK = 0.1                 # PaDiM / Gaussian shrinkage (frozen)
ALPHA = 0.05                 # the calibrated operating point (D14)
ALPHAS = (0.02, 0.03, 0.04, 0.05)     # rule-5 curve
KNN_K = 10
TOP_FRAC = 0.05              # map_top = mean of the top 5% of positions (3 of 64, 13 of 256)
AREA_Q = 0.99                # map_area: per-position cal quantile
LAP_EPS = 1e-6
SAT_LO, SAT_HI = 5, 250
IDENT_TOL = 2e-3             # map-GAP identity: max |mean_hw(map16) - penult| / max |map| (float16 half-ulp ~4.9e-4)
CHANGED_MIN = 0.95           # a fault condition is valid when >= 95% of its images differ from the clean twin
MIN_KEEP = 50                # "output still right" subset: AUROCs only with >= 50 rows

SEV = FAULTS.S_GLOBAL_SEVERITY
F5 = ("paste__gaussian_noise__a12", "paste__defocus_blur__a12", "paste__pixelate__a12",
      "paste__jpeg_compression__a12", "occluder_sq__a12")
PASTE4 = F5[:4]
LOCAL_A12 = F5 + ("glare__a12", "dead_sq__a12")
GLOBAL = tuple(f"global__{c}__s{SEV}" for c in FAULTS.S_GLOBALS)
SOIL = ("soiling__a12", "soiling__a25")
SETS = {"F5": F5, "PASTE4": PASTE4, "LOCAL_A12": LOCAL_A12, "GLOBAL": GLOBAL, "SOIL": SOIL}

MAP_KEYS = ("map_max", "map_top", "map_area", "map_gini")
S2_KEYS = ("s2_map_max", "s2_map_top")
STD_KEYS = ("std_half",)
GEO_KEYS = ("d1", "knn_l2", "gap_maha")
HEAD_KEYS = ("msp", "maxlogit", "gap", "energy", "entropy")
POOLED_KEYS = GEO_KEYS + HEAD_KEYS
PIX_KEYS = ("pix_padim", "pix_sat", "pix_lap_dev")
ALL_KEYS = MAP_KEYS + S2_KEYS + STD_KEYS + GEO_KEYS + HEAD_KEYS + PIX_KEYS
STILL_KEYS = MAP_KEYS + STD_KEYS + GEO_KEYS + HEAD_KEYS
RAW_HEAD = ("gap", "maxlogit", "energy", "entropy")          # T1's head block: sorted logits (linear) + these (splines)
TPR_KEYS = ("map_max", "std_half", "d1", "knn_l2", "gap_maha", "msp", "energy", "pix_padim", "s2_map_max")
SQUARE_KINDS = ("paste", "occluder_sq", "glare", "dead_sq")
INCREMENTS = (   # name, positive conditions, negatives ('eval' = the clean twins), base keys (+ T1 head block), bundle
    ("SP2_map_over_pooled", F5, "eval", GEO_KEYS, MAP_KEYS),
    ("SP3_map_over_pixels", PASTE4, "eval", GEO_KEYS + PIX_KEYS, MAP_KEYS),
    ("SP7_std_over_pooled", F5, "eval", GEO_KEYS, STD_KEYS),
    ("SP8_typing", LOCAL_A12, GLOBAL, GEO_KEYS, MAP_KEYS),
    ("INFO_soiling_map_over_pooled", SOIL, "eval", GEO_KEYS, MAP_KEYS),
    ("INFO_std_map_over_pooled", F5, "eval", GEO_KEYS, MAP_KEYS + STD_KEYS),
    ("INFO_s2_over_map", F5, "eval", GEO_KEYS + MAP_KEYS, S2_KEYS),
)


# =====================================================================================================================
# per-image models
# =====================================================================================================================
class Gauss:
    """Gaussians per position (PaDiM) with shrinkage, fitted on X (N, C, P) (or (N, C): one position). Sigma_p =
    (1 - s) S_p + s tr(S_p)/C I; score(X) -> (n, P) squared Mahalanobis distances through the inverse Cholesky factor."""

    def __init__(self, X, shrink=SHRINK):
        X = np.asarray(X)
        if X.ndim == 2:
            X = X[:, :, None]
        n, c, p = X.shape
        self.C, self.P = int(c), int(p)
        self.mu = np.empty((p, c))
        self.A = np.empty((p, c, c))
        eye = np.eye(c)
        for j in range(p):
            Xj = X[:, :, j].astype(np.float64)
            mu = Xj.mean(0)
            Z = Xj - mu
            S = Z.T @ Z / max(1, n - 1)
            t = float(np.trace(S)) / c
            if not t > 0:
                t = 1.0
            L = np.linalg.cholesky((1.0 - shrink) * S + shrink * t * eye)
            self.mu[j] = mu
            self.A[j] = np.linalg.solve(L, eye)                       # L^-1, so Sigma^-1 = A^T A
        self.shrink = shrink

    def score(self, X):
        X = np.asarray(X)
        if X.ndim == 2:
            X = X[:, :, None]
        out = np.empty((X.shape[0], self.P))
        for j in range(self.P):
            Y = (X[:, :, j].astype(np.float64) - self.mu[j]) @ self.A[j].T
            out[:, j] = np.einsum("ij,ij->i", Y, Y)
        return out

    def nbytes32(self):
        """Deployable size in float32: a mean and a C x C factor per position."""
        return int(self.P * (self.C + self.C * self.C) * 4)


class Pooled:
    """Scores of the pooled vector (the penult = GAP of the head map), all fitted on ref."""

    def __init__(self, ref_pen, ref_y, K):
        ref_pen = np.asarray(ref_pen, dtype=np.float64)
        ref_y = np.asarray(ref_y)
        present = [c for c in range(int(K)) if (ref_y == c).any()]
        self.centres = np.stack([ref_pen[ref_y == c].mean(0) for c in present])
        self.ref_l2 = C.l2n(ref_pen)
        self.gauss = Gauss(ref_pen)

    def score(self, X):
        X = np.asarray(X, dtype=np.float64)
        return {"d1": NC.center_dists(X, self.centres).min(1),
                "knn_l2": C.knn_kth(self.ref_l2, C.l2n(X), KNN_K),
                "gap_maha": self.gauss.score(X)[:, 0]}

    def nbytes32(self):
        return {"class_centres": int(self.centres.size * 4), "knn_reference": int(self.ref_l2.size * 4),
                "gap_gauss": self.gauss.nbytes32()}


def n_top(P):
    return max(1, int(round(TOP_FRAC * P)))


def gini_rows(M):
    """Gini coefficient of each row of a non-negative matrix (0 = uniform, (P - 1)/P = one position)."""
    s = np.sort(np.asarray(M, dtype=np.float64), axis=1)
    p = s.shape[1]
    tot = s.sum(1)
    i = np.arange(1, p + 1, dtype=np.float64)
    g = 2.0 * (s * i).sum(1) / (p * np.where(tot > 0, tot, 1.0)) - (p + 1.0) / p
    return np.where(tot > 0, g, 0.0)


def map_stats(M, tau, prefix):
    s = np.sort(M, axis=1)
    out = {f"{prefix}_max": s[:, -1], f"{prefix}_top": s[:, -n_top(M.shape[1]):].mean(1)}
    if tau is not None:
        out[f"{prefix}_area"] = (M > tau[None, :]).mean(1)
        out[f"{prefix}_gini"] = gini_rows(M)
    return out


# =====================================================================================================================
# pixels, masks, pointing
# =====================================================================================================================
def _lap(g):
    """4-neighbour Laplacian of gray images (N, H, W), reflect border."""
    p = np.pad(g, ((0, 0), (1, 1), (1, 1)), mode="reflect")
    return p[:, :-2, 1:-1] + p[:, 2:, 1:-1] + p[:, 1:-1, :-2] + p[:, 1:-1, 2:] - 4.0 * g


def pixel_cells(imgs, cell=4, chunk=1000):
    """(N, 7, P): per cell x cell block [mean RGB, std RGB, log(Laplacian variance of gray + 1e-6)], P = (H / cell)^2,
    cell index = row * (W / cell) + column (the map's position order)."""
    imgs = np.asarray(imgs)
    N, H, W, _ = imgs.shape
    h, w = H // cell, W // cell
    out = np.empty((N, 7, h * w))
    for i in range(0, N, chunk):
        x = imgs[i:i + chunk].astype(np.float64) / 255.0
        n = len(x)
        c = x.reshape(n, h, cell, w, cell, 3).transpose(0, 1, 3, 2, 4, 5).reshape(n, h * w, cell * cell, 3)
        lap = _lap(x.mean(-1)).reshape(n, h, cell, w, cell).transpose(0, 1, 3, 2, 4).reshape(n, h * w, cell * cell)
        out[i:i + n, 0:3] = c.mean(2).transpose(0, 2, 1)
        out[i:i + n, 3:6] = c.std(2).transpose(0, 2, 1)
        out[i:i + n, 6] = np.log(lap.var(2) + LAP_EPS)
    return out


def lap_logvar(imgs, chunk=1000):
    """Whole-image log Laplacian variance of the gray image."""
    imgs = np.asarray(imgs)
    out = np.empty(len(imgs))
    for i in range(0, len(imgs), chunk):
        g = imgs[i:i + chunk].astype(np.float64).mean(-1) / 255.0
        out[i:i + len(g)] = np.log(_lap(g).reshape(len(g), -1).var(1) + LAP_EPS)
    return out


def sat_frac(imgs):
    x = np.asarray(imgs)
    return ((x <= SAT_LO).all(-1) | (x >= SAT_HI).any(-1)).mean((1, 2))


def cell_masks(masks, h, w):
    """(n, h * w) bool: map cells whose pixel block holds at least one fault pixel."""
    m = np.asarray(masks, dtype=bool)
    n, H, W = m.shape
    return m.reshape(n, h, H // h, w, W // w).any(axis=(2, 4)).reshape(n, h * w)


def dilate_cells(cm, h, w):
    """One-cell (3 x 3) dilation on the h x w grid, no wrap-around."""
    g = np.asarray(cm, dtype=bool).reshape(-1, h, w)
    p = np.pad(g, ((0, 0), (1, 1), (1, 1)))
    out = np.zeros_like(g)
    for dy in range(3):
        for dx in range(3):
            out |= p[:, dy:dy + h, dx:dx + w]
    return out.reshape(len(g), h * w)


def pointing(pos, cm):
    """(hit rate of the argmax positions on the fault cells, chance = mean fraction of fault cells)."""
    cm = np.asarray(cm, dtype=bool)
    hit = cm[np.arange(len(pos)), np.asarray(pos, dtype=np.int64)]
    return float(hit.mean()), float(cm.mean(1).mean())


def hash16(px):
    """= scripts/b4_extract.py image_sha256_first16 (sha256 of the first 16 uint8 images, 16 hex)."""
    return hashlib.sha256(np.ascontiguousarray(np.asarray(px)[:16]).tobytes()).hexdigest()[:16]


def identity_check(maps, pen, std):
    """The stored map is the head's input map: its spatial mean is the stored penult and its unbiased spatial std the
    stored headmap_std, to float16 precision relative to the map's scale."""
    m = np.asarray(maps, dtype=np.float64)
    flat = m.reshape(m.shape[0], m.shape[1], -1)
    scale = max(float(np.abs(flat).max()), 1e-12)
    e_mean = float(np.abs(flat.mean(2) - np.asarray(pen, dtype=np.float64)).max()) / scale
    e_std = float(np.abs(flat.std(2, ddof=1) - np.asarray(std, dtype=np.float64)).max()) / scale
    return {"mean_rel": e_mean, "std_rel": e_std, "pass": bool(e_mean <= IDENT_TOL and e_std <= IDENT_TOL)}


# =====================================================================================================================
# the probe
# =====================================================================================================================
class _Models:
    pooled = std = hm = s2 = pix = tau = lap_med = None
    shape = shape2 = None
    cell = 4


def score_split(d, s, M, timed, checks):
    """Every per-image score of split s. Returns (scores {key: (n,)}, aux {rows, labels, argmax, sl, raw, pos, ...})."""
    out, aux = {}, {}
    with timed("tap:penult"):
        pen = d.acts("penult", s)
        out.update(M.pooled.score(pen))
    with timed("tap:head"):
        Z = d.logits(s)
        h = C.head_stats(Z)
        for k in HEAD_KEYS:
            out[k] = C.ERR_SIGN[k] * h[k]
        aux["raw"] = {k: h[k] for k in RAW_HEAD}
        aux["sl"] = C.sorted_logits(Z, min(10, Z.shape[1]))
        aux["argmax"] = h["argmax"]
    with timed("tap:headmap_std"):
        std = d.acts("headmap_std", s)
        out["std_half"] = M.std.score(std)[:, 0]
    with timed("tap:headmap"):
        mp = d.maps("headmap", s)
        n, c, H, W = mp.shape
        P = M.hm.score(mp.reshape(n, c, H * W))
        if M.tau is None:                                       # the cal split sets the per-position q99 (D4: cal only)
            M.tau = np.quantile(P, AREA_Q, axis=0)
        out.update(map_stats(P, M.tau, "map"))
        aux["pos"] = P.argmax(1)
        checks["identity"][s] = identity_check(mp, pen, std)
        del mp, P
    if M.s2 is not None:
        with timed("tap:stage2map"):
            m2 = d.maps("stage2map", s)
            n2, c2, h2, w2 = m2.shape
            P2 = M.s2.score(m2.reshape(n2, c2, h2 * w2))
            out.update(map_stats(P2, None, "s2_map"))
            aux["pos2"] = P2.argmax(1)
            del m2, P2
    if M.pix is not None:
        with timed("tap:pixels"):
            px = d.pixels(s)
            Pp = M.pix.score(pixel_cells(px, M.cell))
            out["pix_padim"] = Pp.max(1)
            aux["pos_pix"] = Pp.argmax(1)
            out["pix_sat"] = sat_frac(px)
            lv = lap_logvar(px)
            if M.lap_med is None:                               # the cal split sets the median (calibration)
                M.lap_med = float(np.median(lv))
            out["pix_lap_dev"] = np.abs(lv - M.lap_med)
            checks["hash"][s] = hash16(px)
            aux["pixels"] = px
    aux["rows"] = d.rows(s)
    aux["labels"] = d.labels(s)
    return out, aux


def condition_record(d, c, s, f, a, e, ae, thr, M):
    """Per-condition record (fault / global vs the clean twins on eval), its pairing check and its mask record."""
    kind, _, area = ("global", None, None) if c.startswith("global__") else FAULTS.parse_condition(c)
    rows_eq = bool(np.array_equal(a["rows"], ae["rows"]))
    lab_eq = bool(np.array_equal(a["labels"], ae["labels"]))
    pe, pf = ae.get("pixels"), a.get("pixels")
    pix_ok = pe is not None and pf is not None and pe.shape == pf.shape
    pr = {"rows_equal": rows_eq, "labels_equal": lab_eq,
          "changed_frac": float((pf != pe).reshape(len(pf), -1).any(1).mean()) if pix_ok else None}
    rec = {"split": s, "kind": kind, "area": area, "n": int(len(a["rows"]))}
    rec["auroc"] = {k: C.auc(f[k], e[k]) for k in ALL_KEYS if k in f and k in e}
    rec["acc"] = float((a["argmax"] == a["labels"]).mean())
    rec["argmax_unchanged"] = float((a["argmax"] == ae["argmax"]).mean()) if rows_eq else None
    rec["tpr_cal05"] = {k: float((f[k] > thr[k]).mean()) for k in thr if k in f}
    sr = {"n": 0}
    if rows_eq and lab_eq:
        keep = (a["argmax"] == ae["argmax"]) & (ae["argmax"] == ae["labels"])
        sr["n"] = int(keep.sum())
        if sr["n"] >= MIN_KEEP:
            sr["auroc"] = {k: C.auc(f[k][keep], e[k][keep]) for k in STILL_KEYS if k in f and k in e}
    rec["still_right"] = sr
    mrec = None
    if kind != "global":
        mk = d.masks(s)
        _, h, w = M.shape
        cm = cell_masks(mk, h, w)
        cmd = dilate_cells(cm, h, w)
        pt = {}
        pt["hit"], pt["chance"] = pointing(a["pos"], cm)
        pt["hit_dil"], pt["chance_dil"] = pointing(a["pos"], cmd)
        if "pos_pix" in a:
            pt["pix_hit"] = pointing(a["pos_pix"], cm)[0]
        if "pos2" in a and M.shape2 is not None:
            pt["s2_hit"], pt["s2_chance"] = pointing(a["pos2"], cell_masks(mk, M.shape2[1], M.shape2[2]))
        rec["pointing"] = pt
        npx = mk.reshape(len(mk), -1).sum(1)
        exp = FAULTS.S_SIDES[area] ** 2 if kind in SQUARE_KINDS else None
        mrec = {"mean_px": float(npx.mean()), "min_px": int(npx.min()), "expected_px": exp,
                "exact": None if exp is None else bool((npx == exp).all())}
        if kind in SQUARE_KINDS and pix_ok and rows_eq:
            pr["outside_equal"] = bool(np.array_equal(pf[~mk], pe[~mk]))
    pr["pass"] = bool(rows_eq and lab_eq and pr.get("outside_equal", True))
    rec["valid"] = bool(pr["pass"] and (pr["changed_frac"] is None or pr["changed_frac"] >= CHANGED_MIN)
                        and (mrec is None or mrec["min_px"] > 0))
    return rec, pr, mrec


def increment(name, pos, neg, base, bundle, S, A, per, nboot, uid):
    """Cross-fitted head (+ base) vs head (+ base) + bundle, the frozen HEAD-ADDITIVE rule; folds and bootstrap by row."""
    negs = ["eval"] if neg == "eval" else list(neg)
    need = list(pos) + [c for c in negs if c != "eval"]
    absent = [c for c in need if c not in per]
    invalid = [c for c in need if c in per and not per[c]["valid"]]
    missing = [k for k in list(base) + list(bundle) if k not in S["eval"]]
    head = {"conds_pos": list(pos), "conds_neg": negs, "base": ["head_T1"] + list(base), "bundle": list(bundle)}
    if absent or invalid or missing:
        return {**head, "skipped": True, "absent": absent, "invalid": invalid, "missing_keys": missing}
    parts = list(pos) + negs
    y = np.concatenate([np.full(len(A[p]["rows"]), 1.0 if p in pos else 0.0) for p in parts])
    groups = np.concatenate([A[p]["rows"] for p in parts])
    hb = [("sorted_logits", np.concatenate([A[p]["sl"] for p in parts]), "linear")]
    hb += [(k, np.concatenate([A[p]["raw"][k] for p in parts]), "spline") for k in RAW_HEAD]
    hb += [(k, np.concatenate([S[p][k] for p in parts]), "spline") for k in base]
    bb = [(k, np.concatenate([S[p][k] for p in parts]), "spline") for k in bundle]
    r = dict(C.head_increment(hb, bb, y, groups, nboot, f"t3s|{uid}|{name}"))
    degenerate = r.pop("skipped", None)                         # head_increment: 'one class' (call INCONCLUSIVE)
    return {**head, **r, "skipped": False, "degenerate": degenerate}


def _get(rec, k):
    return None if not rec else rec.get(k)


def _log10(v):
    return math.log10(v) if isinstance(v, (int, float)) and v > 0 and math.isfinite(v) else None


def functional(d, M, conf):
    """D14 functional units: reference bytes (float32), false alarms per 1000 clean frames at the calibrated 5%, CPU ms
    per 1000 queries (single process; under 'timing_s' so the S2 replay compare ignores it)."""
    rb = {"map_padim": M.hm.nbytes32(), "std_gauss": M.std.nbytes32(), **M.pooled.nbytes32(),
          "stage2_padim": M.s2.nbytes32() if M.s2 is not None else None,
          "pixel_padim": M.pix.nbytes32() if M.pix is not None else None}
    fa = {k: 1000.0 * v for k, v in conf["fpr_eval"].items()}
    ms = {}
    mp = d.maps("headmap", "eval")
    n, c, H, W = mp.shape
    t = time.perf_counter()
    M.hm.score(mp.reshape(n, c, H * W)).max(1)
    ms["map_max_ms_per_1000"] = 1e6 * (time.perf_counter() - t) / n
    st = d.acts("headmap_std", "eval")
    t = time.perf_counter()
    M.std.score(st)
    ms["std_half_ms_per_1000"] = 1e6 * (time.perf_counter() - t) / n
    pen = d.acts("penult", "eval")
    t = time.perf_counter()
    M.pooled.score(pen)
    ms["pooled_ms_per_1000"] = 1e6 * (time.perf_counter() - t) / n
    if M.pix is not None:
        px = d.pixels("eval")
        t = time.perf_counter()
        M.pix.score(pixel_cells(px, M.cell)).max(1)
        ms["pix_padim_ms_per_1000"] = 1e6 * (time.perf_counter() - t) / n
    return {"reference_bytes_float32": rb, "false_alarms_per_1000_clean": fa, "timing_s": ms}


def probe_dump(d, timed, nboot, uid, spec=None):
    """Everything lane S computes from one maps dump (opened by the caller through atlas.b4_core.open_dump)."""
    if (d.layout or LAYOUT) != LAYOUT:
        raise SystemExit(f"[t3s] {d.path}: layout {d.layout!r}; lane S reads the maps layout only")
    splits = d.list_splits()                                    # readable only: soiling is invisible in discovery (D7)
    miss = [s for s in ("ref", "cal", "eval") if s not in splits]
    if miss:
        raise SystemExit(f"[t3s] {d.path}: splits {miss} missing")
    conds = [s[len("fault__"):] for s in splits if s.startswith("fault__")]
    globs = [s for s in splits if s.startswith("global__")]
    K = d.K
    has_s2 = "stage2map" in (d.b4.get("maps") or {})
    has_pix = os.path.isfile(os.path.join(d.path, "pixels", "ref.npy"))
    M = _Models()
    checks = {"identity": {}, "hash": {}}
    # ---- reference models: fitted on ref (train rows) only ----
    with timed("tap:penult"):
        ref_pen = d.acts("penult", "ref")
        ref_y = d.labels("ref")
        M.pooled = Pooled(ref_pen, ref_y, K)
    with timed("tap:headmap_std"):
        ref_std = d.acts("headmap_std", "ref")
        M.std = Gauss(ref_std)
    with timed("tap:headmap"):
        mp = d.maps("headmap", "ref")
        n_ref, c, H, W = mp.shape
        M.hm = Gauss(mp.reshape(n_ref, c, H * W))
        M.shape = [int(c), int(H), int(W)]
        checks["identity"]["ref"] = identity_check(mp, ref_pen, ref_std)
        del mp
    if has_s2:
        with timed("tap:stage2map"):
            m2 = d.maps("stage2map", "ref")
            n2, c2, h2, w2 = m2.shape
            M.s2 = Gauss(m2.reshape(n2, c2, h2 * w2))
            M.shape2 = [int(c2), int(h2), int(w2)]
            del m2
    if has_pix:
        with timed("tap:pixels"):
            px = d.pixels("ref")
            M.cell = int(px.shape[1] // H)
            M.pix = Gauss(pixel_cells(px, M.cell))
            checks["hash"]["ref"] = hash16(px)
            del px
    # ---- calibration (cal), clean twins (eval), conditions ----
    S, A = {}, {}
    S["cal"], A["cal"] = score_split(d, "cal", M, timed, checks)
    A["cal"].pop("pixels", None)
    S["eval"], A["eval"] = score_split(d, "eval", M, timed, checks)
    thr = {k: C.cal_threshold(S["cal"][k], ALPHA) for k in TPR_KEYS if k in S["cal"]}
    conf = {"alpha": ALPHA, "n_cal": int(len(A["cal"]["rows"])), "n_test": int(len(A["eval"]["rows"])),
            "threshold": thr, "fpr_eval": {k: float((S["eval"][k] > thr[k]).mean()) for k in thr},
            "curve_map_max": [{"alpha": al, "fpr": float((S["eval"]["map_max"]
                                                          > C.cal_threshold(S["cal"]["map_max"], al)).mean())}
                              for al in ALPHAS]}
    per, pairing, masks = {}, {}, {}
    for cnd in conds + globs:
        s = cnd if cnd.startswith("global__") else "fault__" + cnd
        with timed(f"target:{cnd}"):
            f, a = score_split(d, s, M, timed, checks)
            per[cnd], pairing[cnd], mrec = condition_record(d, cnd, s, f, a, S["eval"], A["eval"], thr, M)
        if mrec is not None:
            masks[cnd] = mrec
        a.pop("pixels", None)
        S[cnd], A[cnd] = f, a
    # ---- increments (frozen HEAD-ADDITIVE rule) ----
    incs = {}
    for name, pos, neg, base, bundle in INCREMENTS:
        with timed(f"target:{name}"):
            incs[name] = increment(name, pos, neg, base, bundle, S, A, per, nboot, uid)
    # ---- collapse of the unit (SP-10 INFO), clean accuracy, functional units ----
    with timed("block:collapse"):
        try:
            Wh = d.head()[0]
        except FileNotFoundError:
            Wh = None
        nc = NC.nc_block(ref_pen, ref_y, K, W=Wh)[0]
        pl = NC.nc1_pseudo(d.acts("penult", "cal"), A["cal"]["argmax"], K)
    coll = {"nc1_ref": _get(nc, "nc1"), "nc1_trim_ref": _get(nc, "nc1_trim"), "vci_ref": _get(nc, "vci"),
            "plnc1_cal": _get(pl, "nc1")}
    coll["log10_nc1_ref"] = _log10(coll["nc1_ref"])
    coll["log10_plnc1_cal"] = _log10(coll["plnc1_cal"])
    with timed("block:functional"):
        func = functional(d, M, conf)
    # ---- known answers (the evaluator decides) ----
    idn = checks["identity"]
    meta_h = d.b4.get("image_sha256_first16") or {}
    bad_h = sorted(s for s, v in checks["hash"].items() if meta_h.get(s) != v)
    ka = {"map_gap_identity": {"tol": IDENT_TOL, "per_split": idn,
                               "max_mean_rel": max(v["mean_rel"] for v in idn.values()),
                               "max_std_rel": max(v["std_rel"] for v in idn.values()),
                               "pass": all(v["pass"] for v in idn.values())},
          "pairing": {"per_condition": pairing, "bad": sorted(k for k, v in pairing.items() if not v["pass"]),
                      "pass": all(v["pass"] for v in pairing.values())},
          "mask_area": {"per_condition": masks,
                        "pass": all(v["exact"] is not False and v["min_px"] > 0 for v in masks.values())},
          "pixel_hash": {"n": len(checks["hash"]), "bad": bad_h, "pass": bool(has_pix and not bad_h)}}
    return {"schema": SCHEMA, "layout_read": LAYOUT, "arch": d.b4.get("arch"), "roles": d.b4.get("roles"),
            "trained": None if spec is None else spec.get("trained"), "sealed_dump": bool(d.sealed),
            "settings": {"shrink": SHRINK, "alpha": ALPHA, "alphas": list(ALPHAS), "knn_k": KNN_K, "top_frac": TOP_FRAC,
                         "area_q": AREA_Q, "ident_tol": IDENT_TOL, "changed_min": CHANGED_MIN, "min_keep": MIN_KEEP,
                         "nboot": int(nboot)},
            "sets": {k: list(v) for k, v in SETS.items()},
            "keys": {"map": list(MAP_KEYS), "s2": list(S2_KEYS) if has_s2 else [], "std": list(STD_KEYS),
                     "geo": list(GEO_KEYS), "head": list(HEAD_KEYS), "pix": list(PIX_KEYS) if has_pix else []},
            "maps_shape": {"headmap": M.shape, "stage2map": M.shape2},
            "n": {"ref": int(n_ref), "cal": conf["n_cal"], "eval": conf["n_test"]},
            "rows": {s: [int(A[s]["rows"].min()), int(A[s]["rows"].max()) + 1] for s in ("cal", "eval")},
            "conditions": conds + globs, "missing": list(d.b4.get("missing") or []),
            "clean": {"acc_cal": float((A["cal"]["argmax"] == A["cal"]["labels"]).mean()),
                      "acc_eval": float((A["eval"]["argmax"] == A["eval"]["labels"]).mean())},
            "collapse": coll, "known_answers": ka, "image_sha256_first16": dict(checks["hash"]),
            "conformal": conf, "per_condition": per, "increments": incs, "functional": func}


def run_probe(a):
    run = C.ProbeRun("t3s", __file__, a)                        # output guard before any read (D5)
    reg = C.load_registry(a.registry)
    spec = C.unit_spec(reg, a.unit)
    if LAYOUT not in spec["layouts"]:
        raise SystemExit(f"[t3s] unit {a.unit} has no maps layout (lane S units: roles Sdisc / Sconf)")
    d = C.open_unit(reg, a.unit, LAYOUT, a.phase)               # the D7 seal
    body = probe_dump(d, run.timed, run.nboot, a.unit, spec)
    path = run.finish(body, dumps=[d])
    print(f"[t3s] {a.unit} ({a.phase}): {len(body['per_condition'])} conditions, "
          f"{sum(not v.get('skipped') for v in body['increments'].values())} increments -> {path}", flush=True)
    return path


# =====================================================================================================================
# synthetic maps dumps with PLANTED answers (the self-test and tests/test_t3s_spatial.py; no torch, no volume)
# =====================================================================================================================
def _save(root, rel, arr):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.save(p, arr)


def _redistribute(m, cm, delta):
    """Add +delta to channel 0 on the fault cells and take the same total from the other cells: the spatial mean of every
    channel (= the penult) is unchanged, the positions change."""
    n, _, h, w = m.shape
    k = cm.sum(1).astype(np.float64)[:, None]
    add = np.where(cm, delta, -delta * k / np.maximum(h * w - k, 1.0))
    m[:, 0] += add.reshape(n, h, w)
    return m


def synth_maps_dump(path, uid="synth_s", sealed=False, holdout=False, n_ref=240, n_cal=120, n_eval=120, c=8, h=8,
                    s2=True, delta=8.0, glob_shift=1.0, spread=0.5, seed=0, roles=None):
    """A maps dump in the scripts/b4_extract.py layout, numpy only, with planted answers:
      local faults (every kind but glare) move activation of head-map channel 0 onto the fault cells while keeping every
        channel's spatial mean, and the stored penult and logits are the clean twin's: every pooled and head score equals
        the clean twin's bit for bit (AUROC exactly 0.5), the per-position scores and the spatial std see the fault;
      glare is planted INVISIBLE (maps equal to the clean twin): every network score ties (AUROC exactly 0.5);
      global corruptions shift channel 0 everywhere: the penult moves, so the pooled scores see them.
    Pixels are uint8 noise faulted by atlas/faults.py itself (real masks, real pairing)."""
    rng = np.random.default_rng(seed)
    K = 10
    mu = rng.normal(3.0, spread, (K, c))                        # class means shared by every position (+ unit noise)
    Wh, bh = 2.0 * mu, -(mu ** 2).sum(1)                        # logits = nearest class mean (up to a constant)
    c2, H2 = 4, 2 * h

    def clean(y):
        return mu[np.asarray(y)][:, :, None, None] + rng.normal(0.0, 1.0, (len(y), c, h, h))

    def pix(n):
        return rng.integers(0, 256, (n, 32, 32, 3)).astype(np.uint8)

    splits, meta_rows, hashes, acc = [], {}, {}, {}

    def put(s, maps, pen, logits, y, rows, px, m2=None, masks=None):
        _save(path, f"maps/headmap/{s}.npy", maps.astype(np.float16))
        _save(path, f"acts/headmap_std/{s}.npy", maps.reshape(len(maps), c, -1).std(2, ddof=1).astype(np.float16))
        _save(path, f"acts/penult/{s}.npy", np.asarray(pen, dtype=np.float32))
        if m2 is not None:
            _save(path, f"maps/stage2map/{s}.npy", m2.astype(np.float16))
        L = np.asarray(logits, dtype=np.float32)
        _save(path, f"logits/{s}.npy", L)
        _save(path, f"labels/{s}.npy", np.asarray(y, dtype=np.int64))
        _save(path, f"rows/{s}.npy", np.asarray(rows, dtype=np.int64))
        np.savez(os.path.join(path, "preds", f"{s}.npz"), argmax=L.argmax(1).astype(np.int64),
                 maxprob=np.ones(len(L), np.float32))
        _save(path, f"pixels/{s}.npy", px)
        if masks is not None:
            _save(path, f"masks/{s}.npy", np.asarray(masks, dtype=bool))
        splits.append(s)
        hashes[s] = hash16(px)
        acc[s] = float((L.argmax(1) == np.asarray(y)).mean())
        meta_rows[s] = [int(np.min(rows)), int(np.max(rows)) + 1]

    os.makedirs(os.path.join(path, "preds"), exist_ok=True)
    if sealed:
        with open(os.path.join(path, "SEALED.json"), "w") as fh:
            json.dump({"unit": uid, "layout": LAYOUT, "rule": "synthetic"}, fh)
    base = {}
    for s, n, lo, y in (("ref", n_ref, 0, np.arange(n_ref) % K), ("cal", n_cal, 7500, rng.integers(0, K, n_cal)),
                        ("eval", n_eval, 8500, rng.integers(0, K, n_eval))):
        m = clean(y)
        pen = m.mean((2, 3))
        rows = (2 * np.arange(n) if s == "ref" else lo + np.arange(n)).astype(np.int64)
        m2 = rng.normal(1.0, 1.0, (n, c2, H2, H2)) if s2 else None
        px = pix(n)
        put(s, m, pen, pen @ Wh.T + bh, y, rows, px, m2)
        base[s] = (m, pen, y, rows, px, m2)
    m_e, pen_e, y_e, rows_e, px_e, m2_e = base["eval"]
    L_e = pen_e @ Wh.T + bh
    for cond in FAULTS.s_conditions(bool(holdout)):
        kind, _, area = FAULTS.parse_condition(cond)
        src = pix(n_eval) if kind == "paste" else None
        fpx, fmask = FAULTS.s_fault_batch(kind, area, px_e, rows_e, FAULTS.SEED, src)
        m, m2 = m_e.copy(), (m2_e.copy() if s2 else None)
        if kind != "glare":
            _redistribute(m, cell_masks(fmask, h, h), delta)
            if s2:
                _redistribute(m2, cell_masks(fmask, H2, H2), delta)
        put(f"fault__{cond}", m, pen_e, L_e, y_e, rows_e, fpx, m2, fmask)
    for g in FAULTS.S_GLOBALS:
        m = m_e.copy()
        m[:, 0] += glob_shift
        pen = m.mean((2, 3))
        m2 = None
        if s2:
            m2 = m2_e.copy()
            m2[:, 0] += glob_shift
        gpx = np.clip(px_e.astype(np.int64) + 40, 0, 255).astype(np.uint8)
        put(f"global__{g}__s{SEV}", m, pen, pen @ Wh.T + bh, y_e, rows_e, gpx, m2)
    os.makedirs(os.path.join(path, "head"), exist_ok=True)
    np.savez(os.path.join(path, "head", "head.npz"), W=Wh.astype(np.float32), b=bh.astype(np.float32))
    b4 = {"schema": "b4_dump/1", "unit": uid, "layout": LAYOUT, "sealed": bool(sealed),
          "roles": list(roles or (["Sconf"] if holdout else ["Sdisc"])), "arch": "synthetic_resnet", "family": "resnet",
          "maps": {"headmap": "layer3", **({"stage2map": "layer2"} if s2 else {})}, "rows": meta_rows, "missing": [],
          "image_sha256_first16": hashes,
          "synthetic": {"planted": "local faults (not glare): channel-0 redistribution onto the fault cells, delta "
                        f"{delta}; glare invisible; globals channel-0 shift {glob_shift}", "seed": int(seed)}}
    meta = {"source": "synthetic", "arch": "synthetic_resnet", "seed_tag": uid, "layers": ["headmap_std", "penult"],
            "splits": splits, "n_classes": K, "accuracy": acc, "b4": b4}
    with open(os.path.join(path, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    return path


def synth_registry(path, units):
    """units: [(uid, dump, sealed, holdout, trained)] -> a registry file with maps layouts only."""
    models = [{"id": u, "roles": ["Sconf"] if ho else ["Sdisc"], "arch": "synthetic_resnet", "family": "resnet",
               "source": "synthetic", "trained": bool(tr), "penult": 8,
               "layouts": {LAYOUT: {"dump": dp, "sealed": bool(se), "maps": ["headmap", "stage2map"],
                                    "holdout_faults": bool(ho), "pixels": True}}}
              for u, dp, se, ho, tr in units]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"schema": "b4_models/1", "models": models}, fh, indent=1)
    return path


# =====================================================================================================================
# self-test (planted synthetic dumps; CPU; the S0 / S1 / S2 hard gate)
# =====================================================================================================================
class _GitOK:
    returncode, stdout = 0, "synthetic_head\n"


def _load_b4_reg():
    import importlib.util
    spec = importlib.util.spec_from_file_location("b4_reg", os.path.join(ROOT, "scripts", "b4_reg.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def selftest(workdir=None):
    import shutil
    import tempfile
    import traceback
    checks = []

    def ck(name, ok, detail=""):
        checks.append({"name": name, "pass": bool(ok), "detail": str(detail)[:300]})
        print(f"[t3s selftest] {'PASS' if ok else 'FAIL'} {name} {'' if ok else detail}", flush=True)

    tmp = workdir or tempfile.mkdtemp(prefix="t3s_selftest_")
    saved_git = C._git
    saved_env = {k: os.environ.get(k) for k in ("ATLAS_B4_UNSEAL", "ATLAS_B4_CHECK_DIR")}
    try:
        # ---- helpers: known answers ----
        ck("gini: uniform 0, one position (P-1)/P", abs(gini_rows(np.ones((1, 64)))[0]) < 1e-12
           and abs(gini_rows(np.eye(64)[:1])[0] - 63 / 64) < 1e-12)
        z = np.random.default_rng(1).normal(size=(20000, 6, 1))
        g = Gauss(z, shrink=0.0)
        ck("Gaussian: mean squared Mahalanobis of N(0, I) = C", abs(g.score(z).mean() - 6.0) < 0.1, g.score(z).mean())
        sq = np.zeros((1, 32, 32), bool)
        sq[0, 0:11, 0:11] = True
        cm = cell_masks(sq, 8, 8)
        ck("cell mask of an 11-px corner square: 3 x 3 cells; dilated 4 x 4", cm.sum() == 9
           and dilate_cells(cm, 8, 8).sum() == 16, (cm.sum(), dilate_cells(cm, 8, 8).sum()))
        ck("pointing: hit on a fault cell, chance = cell fraction", pointing([0], cm) == (1.0, 9 / 64)
           and pointing([63], cm)[0] == 0.0)
        flat = pixel_cells(np.full((2, 32, 32, 3), 77, np.uint8))
        ck("pixel cells of a flat image: mean 77/255, std 0, log(1e-6)", np.allclose(flat[:, 0:3], 77 / 255)
           and np.allclose(flat[:, 3:6], 0) and np.allclose(flat[:, 6], math.log(LAP_EPS)))
        ck("condition sets are frozen lane-S conditions", set(F5 + LOCAL_A12 + SOIL) <= set(FAULTS.S_CONDITIONS)
           and len(GLOBAL) == len(FAULTS.S_GLOBALS) and all(C.is_confirmation_only("fault__" + s) for s in SOIL)
           and not any(C.is_confirmation_only("fault__" + s) for s in LOCAL_A12 + PASTE4))
        # ---- planted dumps: an OPEN one that nevertheless holds soiling, and a SEALED confirmation one ----
        dd = synth_maps_dump(os.path.join(tmp, "b4d_synth_disc_maps", "dump"), "synth_disc", sealed=False,
                             holdout=True, roles=["Sdisc"])
        cd = synth_maps_dump(os.path.join(tmp, "b4c_synth_conf_maps", "dump"), "synth_conf", sealed=True, holdout=True,
                             seed=1)
        regp = synth_registry(os.path.join(tmp, "models.json"), [("synth_disc", dd, False, False, True),
                                                                 ("synth_conf", cd, True, True, True)])
        ap = C.probe_argparser("t3s")
        o1 = os.path.join(tmp, "out", "synth_disc")
        run_probe(C.parse_probe_args(ap, ["--registry", regp, "--unit", "synth_disc", "--phase", "discovery",
                                          "--out", o1]))
        r = json.load(open(os.path.join(o1, "probe.json")))
        pc, ka = r["per_condition"], r["known_answers"]
        ck("discovery never reads soiling (confirmation-only, D7)", not any("soiling" in k for k in pc)
           and not any("soiling" in s for s in r["dumps"][0]["splits_read"]))
        ck("known answers pass on the planted dump (identity, pairing, masks, hashes)",
           ka["map_gap_identity"]["pass"] and ka["pairing"]["pass"] and ka["mask_area"]["pass"]
           and ka["pixel_hash"]["pass"], {k: v.get("pass") for k, v in ka.items()})
        ck("every condition valid", all(v["valid"] for v in pc.values()),
           [k for k, v in pc.items() if not v["valid"]])
        mm = [pc[c]["auroc"]["map_max"] for c in F5]
        ck("planted local faults: map_max AUROC >= 0.99 on F5", min(mm) >= 0.99, mm)
        blind = [pc[c]["auroc"][k] for c in F5 for k in POOLED_KEYS]
        ck("planted local faults: every pooled and head AUROC is exactly 0.5", max(abs(v - 0.5) for v in blind) < 1e-12,
           blind[:8])
        gl = [pc["glare__a12"]["auroc"][k] for k in MAP_KEYS + STD_KEYS + POOLED_KEYS]
        ck("invisible fault (glare planted as nothing): every network AUROC exactly 0.5",
           max(abs(v - 0.5) for v in gl) < 1e-12, gl)
        ck("global shift: the pooled d1 sees it (AUROC >= 0.99)", min(pc[g]["auroc"]["d1"] for g in GLOBAL) >= 0.99)
        hit = np.mean([pc[c]["pointing"]["hit"] for c in F5])
        chn = np.mean([pc[c]["pointing"]["chance"] for c in F5])
        ck("pointing: the map max lies on the planted cells", hit >= 0.95 and chn < 0.4, (hit, chn))
        inc = r["increments"]
        ck("SP2 map over head + pooled: ADDS", inc["SP2_map_over_pooled"].get("call") == "ADDS",
           inc["SP2_map_over_pooled"].get("call"))
        ck("SP7 spatial std over head + pooled: ADDS", inc["SP7_std_over_pooled"].get("call") == "ADDS",
           inc["SP7_std_over_pooled"].get("call"))
        ck("SP3 and SP8 computed; soiling increment skipped in discovery",
           not inc["SP3_map_over_pixels"]["skipped"] and not inc["SP8_typing"]["skipped"]
           and inc["INFO_soiling_map_over_pooled"]["skipped"])
        fpr = r["conformal"]["fpr_eval"]["map_max"]
        ck("clean false-alarm rate at the cal-calibrated 5% is plausible (exchangeable cal / eval)", 0.0 <= fpr <= 0.15,
           fpr)
        ck("record carries timing per tap and target, max_rss_mb, code sha and commit",
           "tap:headmap" in r["timing_s"] and "target:SP2_map_over_pooled" in r["timing_s"] and "total" in r["timing_s"]
           and "max_rss_mb" in r and len(r["code"]["sha256"]) == 64 and "repo_commit" in r["code"])
        # ---- determinism: the S2 replay compare (scripts/b4_reg.py, rel 1e-6) on a second run ----
        o2 = os.path.join(tmp, "out", "synth_disc_s2replay")
        run_probe(C.parse_probe_args(ap, ["--registry", regp, "--unit", "synth_disc", "--phase", "discovery",
                                          "--out", o2]))
        bad = []
        _load_b4_reg().same(r, json.load(open(os.path.join(o2, "probe.json"))), "", bad)
        ck("a second run is identical under the replay tolerance", not bad, bad[:5])
        # ---- the seal ----
        try:
            run_probe(C.parse_probe_args(ap, ["--registry", regp, "--unit", "synth_conf", "--phase", "discovery",
                                              "--out", os.path.join(tmp, "out", "synth_conf_disc")]))
            ck("a sealed dump is refused in discovery", False)
        except C.ReadRefused:
            ck("a sealed dump is refused in discovery", True)
        os.environ["ATLAS_B4_UNSEAL"] = "p2_synthetic"
        os.environ["ATLAS_B4_CHECK_DIR"] = os.path.join(tmp, "check")
        C._git = lambda *a: _GitOK()
        o3 = os.path.join(tmp, "out", "synth_conf")
        run_probe(C.parse_probe_args(ap, ["--registry", regp, "--unit", "synth_conf", "--phase", "confirmation",
                                          "--out", o3]))
        rc = json.load(open(os.path.join(o3, "probe.json")))
        ck("confirmation after a (fake) P2 reads soiling; the soiling increment runs",
           "soiling__a12" in rc["per_condition"] and not rc["increments"]["INFO_soiling_map_over_pooled"]["skipped"]
           and rc["per_condition"]["soiling__a12"]["auroc"]["map_max"] >= 0.99 and rc["nboot"] == 1000)
        try:
            run_probe(C.parse_probe_args(ap, ["--registry", regp, "--unit", "synth_conf", "--phase", "confirmation",
                                              "--out", os.path.join(tmp, "out", "synth_conf_r2")]))
            ck("a confirmation unit is probed once (touched-once)", False)
        except C.ReadRefused:
            ck("a confirmation unit is probed once (touched-once)", True)
    except (Exception, SystemExit) as e:                        # a crash or an unexpected refusal is a failed self-test
        ck("self-test ran without an exception", False, f"{type(e).__name__}: {e} {traceback.format_exc()[-600:]}")
    finally:
        C._git = saved_git
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if workdir is None:
            shutil.rmtree(tmp, ignore_errors=True)
    bad = [c["name"] for c in checks if not c["pass"]]
    return {"program": "t3s_spatial_probe", "status": "PASS" if not bad else "FAIL", "failed": bad, "checks": checks,
            "code_sha256": C.code_sha256(os.path.abspath(__file__)),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def main(argv=None):
    ap = C.probe_argparser("t3s", description="batch-4 lane S (X5 beyond GAP) probe on the maps dumps "
                                              "(docs/plans/T3S_SPATIAL.md)")
    a = C.parse_probe_args(ap, argv)
    if a.selftest:
        rep = selftest()
        C.write_selftest(a.selftest_out, rep)
        print(f"[t3s] selftest {rep['status']} ({len(rep['checks'])} checks)", flush=True)
        return 0 if rep["status"] == "PASS" else 1
    run_probe(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
