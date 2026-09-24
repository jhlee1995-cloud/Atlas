#!/usr/bin/env python3
"""
collapse_probe.py -- T2, the collapse arm of batch 4: the Stage-B probe of ONE model (docs/plans/T2_COLLAPSE.md;
docs/plans/B4_INTEGRATION.md D4, D5, D7, D12). numpy + atlas.b4_core + atlas.b4_collapse only (no torch, scipy or
sklearn; tests/test_collapse_probe.py pins this). It measures
  (1) the model's COLLAPSE COORDINATES at every tap: labelled on the train reference, labelled on held-out clean rows,
      and LABEL-FREE (what a deployed controller can compute from unlabelled clean frames), and
  (2) the model-level TARGETS the laws predict, among them the PRIMARY outcomes O2, O3, O4 (O1 and O5 come from T1's
      scoreboard, experiments/b4/model_outcomes.schema.json).
It applies no threshold and fits no law: scripts/collapse_laws.js fits the laws on the discovery units (D21), freezes
them at P2 and scores them once on the sealed confirmation units.

  python scripts/collapse_probe.py --registry experiments/b4/models.json --unit <id> --phase discovery|confirmation \
         --out results/b4_t2/<id><tag>
  python scripts/collapse_probe.py --selftest [--selftest-out results/instrument_check_b4s0/selftest_collapse_probe.json]

Contract (D5): the output directory is guarded before anything is read (append-only; confirmation: touched once under any
tag); the unit's FIT-layout dump is opened only through atlas.b4_core.open_unit (the D7 seal: a sealed dump opens only in
the confirmation phase after P2; a confirmation-only split is never readable in discovery); probe.json carries timing_s
per tap ('tap:<t>') and per target ('target:<name>', 'target:taps:<t>'), timing_s.total, max_rss_mb, code.sha256,
code.core_sha256 and repo_commit (atlas.b4_core.ProbeRun), plus code.collapse_sha256 (atlas/b4_collapse.py; ProbeRunT2).
Deterministic (the only random draws are keyed by atlas.b4_core.rng_for), so the S2 replay of the anchors compares equal
(scripts/b4_reg.py replay-compare).
Units: discovery phase -> roles D, N, Dnew, ANCHOR; confirmation phase -> roles C, K, F, F20 (anything else is refused).

Rows (global CIFAR-10 test indices of the fit layout, D4; test = rows 0-4999 in order, so position = row):
  test 0-4999   every held-out coordinate (T2 never reads rows 5000-9999); split halves = even / odd rows
  H 0-1999      clean rows paired with corrupt__<c>__s<k> rows 0-1999 (harm, overconfidence)
  A 2000-3499   selection and calibration only (P3b head choice; IC-P2 conformal calibration)
  B 3500-4999   clean negatives (OOD, per-tap shift AUROC, IC-P2 evaluation)
  ref           the atlas train reference (10,000 train rows); the only reference of every kNN and centre

Coordinates (per tap; float64; K = 10; `trim` = Sigma_B inverted on its top K-1 eigenpairs above 1e-10 lambda_1):
  nc1_tr        tr(Sigma_W Sigma_B^+) / K on the reference, the TRIMMED inverse. It equals the atlas/invariants/
                landmarks.py:71 value (numpy-default pinv, kept as nc1_tr_verbatim) whenever that pinv resolves exactly the
                K-1 class directions; nc1_pinv_rel_dev records the difference (a spurious singular value above the 1e-15
                cutoff at a 512-2048-d penult would inflate the verbatim value; the laws use the trimmed one)
  nc1_te        the same on held-out test rows 0-4999 with their own class means (labels)
  nc1_tetr      held-out scatter around the TRAIN class means over the train Sigma_B (trimmed; labels)
  plnc1_te      nc1 on test rows 0-4999 with the head's argmax (stored logits) as labels: LABEL-FREE (trimmed)
  g_cv_te       lambda_{K-1}/lambda_K of cross-validated eigenvalues (eigenvectors from the even rows, variances on the odd
                rows, and the reverse): LABEL-FREE; g_raw_te the raw ratio (INFO)
  topk_frac_te  share of held-out variance in the top K-1 principal components: LABEL-FREE
  ncc_agree_te  NC4: nearest train class centre equals the head's argmax (test rows 0-4999)
  vci, etf_dev, norm_cv, bt, head_center_cos / nc3_dist (penult), erank, khat, alpha, pr (INFO)
  penult only: head-only coordinates sat999_te (share of rows with MSP >= 0.999), gap_mean_te (mean top-1 minus top-2
  logit), msp_def_te (1 - mean MSP); accuracy acc_te / err_te (labels); ece_te (15 bins); cdepth_pl = log10 plnc1(pre) -
  log10 plnc1(penult) (label-free collapse depth; pre = meta.b4.taps.pre, the D2 rule); equi-separation slope (INFO).
Targets (penult unless stated; scores oriented so that larger = more error-, shift- or OOD-like, atlas.b4_core.ERR_SIGN):
  do3        train-referenced kNN (k 10, raw features) false-alarm rate on clean test rows 0-4999 at the reference's
             self-excluded q95 of log r10 (AH-1's quantity, full rows) -> M-DO3, M-LF, P2(b), MP-1
  ic_p2      conformal p = (1 + #{cal >= s}) / (n + 1), cal = r10 of rows A, flagged at p <= alpha on rows B (IC-P2)
  margin     clean errors (argmax != label) on rows 0-4999: AUROC of margin (d2 - d1 to the train centres), d1 and the
             head statistics; lead_md = margin - d1 (P3a, MP-3); gap_minus_msp (MP-5)
  p3b        head statistic picked on rows A; AUROC(margin) - AUROC(picked head) on rows H + B with a 99% row-bootstrap
             CI (run.nboot draws) (P3b)
  ood        CIFAR-100 / SVHN rows 0-1999 vs clean rows B: AUROC of L2-normalised 10-NN radius (Sun et al. 2022), raw
             10-NN, and the head statistics; knn_l2_minus_besthead (O3 = CIFAR-100; P4 = SVHN)
  harm       per discovery corruption x severity (30 splits, rows 0-1999): H = (median log r10(split) - median log r10(clean
             paired rows 0-1999)) / (q95 - q50 of the reference self log r10); cost = paired accuracy loss in pt;
             OLS cost ~ H: slope (P5a), H10 = H at 10 pt (O2, P5b). harm_committed = the AH-2(b) form (clean median and
             accuracy over test rows 0-4999): the HOLD band H <= 0.25 -> cost <= 8 pt (P5c); holdout + extras only in the
             confirmation phase (INFO)
  overconf   mean over the 30 discovery splits of mean MSP - accuracy (O4, S7)
  taps       per non-duplicate tap: per-sample corrupt (rows 0-1999, 10 corruptions at s1 and s3) vs clean rows B AUROC of
             the 10-NN radius to the reference at that tap; the tap and the head statistic are picked on the even rows
             and scored on the odd rows; p6b = best tap - best head (P6b); penult_deficit (INFO)
  targets_se split-half noise |half1 - half2| / 2 of every law target (even / odd rows)
"""
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

from atlas import b4_collapse as NC     # noqa: E402  numpy only
from atlas import b4_core as C          # noqa: E402  numpy only

SCHEMA = "b4_t2_probe/1"
K = 10
KNN_K = 10
ALPHA = 0.05
ALPHAS_CURVE = (0.02, 0.03, 0.04, 0.05)          # rule 5: IC-P2 reported as a curve
HOLD = {"H": 0.25, "cost": 8.0, "loss10": 10.0}  # AH-2(b), docs/plans/ANOMALY_H1.md; H10 = H at 10 pt of loss
TAP_SEVS = (1, 3)
CI_LEVEL = 0.99
ROWS = {"TEST": (0, 5000), "H": (0, 2000), "A": (2000, 3500), "B": (3500, 5000)}   # D4 (fit layout)
HEADS = ("msp", "maxlogit", "gap", "energy", "entropy")                            # the head bar (O5's list)
ROLES = {"discovery": ("D", "N", "Dnew", "ANCHOR"), "confirmation": ("C", "K", "F", "F20")}
CONSTANTS = {"schema": SCHEMA, "k": K, "knn_k": KNN_K, "alpha": ALPHA, "alphas_curve": ALPHAS_CURVE, "hold": HOLD,
             "tap_sevs": TAP_SEVS, "ci_level": CI_LEVEL, "rows": ROWS, "heads": HEADS, "trim_floor": 1e-10,
             "disc": C.DISC, "holdout": C.HOLDOUT, "extra": C.EXTRA}
_QUIET = [False]


class ProbeRunT2(C.ProbeRun):
    """atlas.b4_core.ProbeRun plus code.collapse_sha256, the hash of atlas/b4_collapse.py that computes every NC
    coordinate (scripts/collapse_laws.js requires one probe instrument -- collapse_probe.py, b4_core.py, b4_collapse.py --
    across the fitted discovery probes and the confirmation probes; verifier T2-2)."""

    def header(self):
        h = super().header()
        h["code"]["collapse_sha256"] = C.code_sha256(os.path.abspath(NC.__file__))
        return h


def say(msg):
    if not _QUIET[0]:
        print(f"[t2] {msg}", flush=True)


# =====================================================================================================================
# small numerics
# =====================================================================================================================
def _f(v):
    """A finite Python float, or None."""
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _lg(v):
    v = _f(v)
    return math.log10(v) if v is not None and v > 0 else None


def _sub(a, b):
    return None if a is None or b is None else float(a - b)


def _half_se(a, b):
    return None if a is None or b is None else float(abs(a - b) / 2.0)


def _mean(vals):
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None


def _rel_dev(a, b):
    a, b = _f(a), _f(b)
    return None if a is None or b is None or b == 0 else float(abs(a / b - 1.0))


def _best_by(tab, order):
    """The key with the largest non-None value, ties to the first in `order` (deterministic)."""
    best = None
    for k in order:
        v = tab.get(k)
        if v is not None and (best is None or v > tab[best]):
            best = k
    return best


def _ols(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 3 or x.max() == x.min():
        return None
    b = float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())
    return float(y.mean() - b * x.mean()), b


def harm_fit(H, cost, loss10=HOLD["loss10"]):
    """OLS cost ~ H over the splits; H10 = the H at which the fitted loss reaches `loss10` pt (None unless slope > 0)."""
    pairs = [(h, c) for h, c in zip(H, cost) if _f(h) is not None and _f(c) is not None]
    out = {"n": len(pairs), "intercept": None, "slope": None, "H10": None,
           "spearman": C.spearman([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else None}
    fit = _ols([p[0] for p in pairs], [p[1] for p in pairs])
    if fit is not None:
        a, b = fit
        out.update(intercept=a, slope=b, H10=(loss10 - a) / b if b > 0 else None)
    return out


def hold_band(H, cost, hmax=HOLD["H"], cmax=HOLD["cost"]):
    """The committed HOLD band (AH-2(b)): splits with H <= 0.25 must cost <= 8 pt."""
    band = [c for h, c in zip(H, cost) if h is not None and c is not None and h <= hmax]
    return {"hold_n": len(band), "hold_viol": int(sum(c > cmax for c in band)),
            "max_cost_in_band": float(max(band)) if band else None}


def boot_auc_diff(sa, sb, y, nboot, key, level=CI_LEVEL):
    """Row-bootstrap percentile CI of AUROC(sa) - AUROC(sb) on the same rows (positives y); deterministic in key."""
    y = np.asarray(y).astype(bool)
    if nboot < 1 or y.all() or not y.any():
        return [None, None]
    A, B = C.AucBoot(sa, y), C.AucBoot(sb, y)
    rng = C.rng_for("t2_boot", *key)
    n = len(y)
    v = np.empty(nboot)
    for i in range(nboot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)
        v[i] = A.auc(w) - B.auc(w)
    if not np.isfinite(v).any():
        return [None, None]
    lo, hi = (1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0
    return [_f(np.nanquantile(v, lo)), _f(np.nanquantile(v, hi))]


def nc1_trim_around(Xte, yte, C_tr, Sb_tr, k=K):
    """nc1_tetr with the trimmed inverse of the train Sigma_B (the verbatim form is b4_collapse.nc1_around)."""
    Wd = np.asarray(Xte, dtype=np.float64) - C_tr[np.asarray(yte, dtype=np.int64)]
    Sw = Wd.T @ Wd / len(Wd)
    P, _ = NC.pinv_floor(Sb_tr, floor=CONSTANTS["trim_floor"], rank_max=k - 1)
    return _f(np.trace(Sw @ P) / k)


# =====================================================================================================================
# coordinates
# =====================================================================================================================
def tap_coords(Xtr, ytr, Xte, yte, am, W=None):
    """Collapse coordinates of one tap -> (record, train class centres, train Sigma_B); (None, None, None) if undefined."""
    tr, Ctr, Sbtr = NC.nc_block(Xtr, ytr, K, W)
    if tr is None:
        return None, None, None
    te = NC.nc_block(Xte, yte, K)[0] or {}
    pl = NC.nc1_pseudo(Xte, am, K) or {}
    h1, h2 = NC.split_halves(len(Xte))
    sp = NC.spectrum_block(Xte, K, Xte[h1], Xte[h2])
    agree, ncc = NC.ncc_agree(Xte, Ctr, am)
    c = {"dim": int(np.asarray(Xtr).shape[1]),
         "nc1_tr": tr.get("nc1_trim"), "nc1_tr_verbatim": tr.get("nc1"),
         "nc1_pinv_rel_dev": _rel_dev(tr.get("nc1"), tr.get("nc1_trim")),
         "vci_tr": tr.get("vci"), "bt_tr": tr.get("bt"), "etf_dev_tr": tr.get("etf_dev"), "norm_cv_tr": tr.get("norm_cv"),
         "center_dist_cv_tr": tr.get("center_dist_cv"), "rank_sb_tr": tr.get("rank_sb"),
         "nc1_te": te.get("nc1_trim"), "nc1_te_verbatim": te.get("nc1"), "vci_te": te.get("vci"),
         "etf_dev_te": te.get("etf_dev"),
         "nc1_tetr": nc1_trim_around(Xte, yte, Ctr, Sbtr), "nc1_tetr_verbatim": NC.nc1_around(Xte, yte, Ctr, Sbtr, K),
         "plnc1_te": pl.get("nc1_trim"), "plnc1_te_verbatim": pl.get("nc1"), "plvci_te": pl.get("vci"),
         "pl_k_present": pl.get("k_present"),
         "g_cv_te": sp.get("g_cv"), "g_raw_te": sp.get("g_raw"), "topk_frac_te": sp.get("topk_frac"),
         "erank_te": sp.get("erank"), "pr_te": sp.get("pr"), "khat_te": sp.get("khat"), "alpha_te": sp.get("alpha"),
         "lam_cv_top_te": sp.get("lam_cv_top"), "eig_top_te": sp.get("eig_top"),
         "ncc_agree_te": agree, "ncc_acc_te": _f((ncc == np.asarray(yte)).mean())}
    if W is not None:
        c["head_center_cos_tr"] = tr.get("head_center_cos")
        c["nc3_dist_tr"] = tr.get("nc3_dist")
    return c, Ctr, Sbtr


# =====================================================================================================================
# target blocks
# =====================================================================================================================
def ood_block(pos, neg, pm=slice(None), nm=slice(None)):
    """AUROCs of every score (positives = OOD rows, negatives = clean rows B); larger oriented score = more OOD-like."""
    a = {k: C.auc(pos[k][pm], neg[k][nm]) for k in pos}
    bh = _best_by({h: a.get(h) for h in HEADS}, HEADS)
    out = {f"auc_{k}": v for k, v in a.items()}
    out.update({"best_head": bh, "auc_best_head": a.get(bh) if bh else None,
                "knn_l2_minus_besthead": _sub(a.get("knn_l2"), a.get(bh)) if bh else None,
                "knn_minus_besthead": _sub(a.get("knn"), a.get(bh)) if bh else None,
                "knn_l2_minus_msp": _sub(a.get("knn_l2"), a.get("msp"))})
    return out


def split_pass(d, Xp_tr, splits, yH, run, keep=()):
    """Per corrupt split (penult, rows 0-1999 paired with clean test rows 0-1999): median log r10 to the train reference,
    accuracy and mean MSP (stored logits), for all rows and the even / odd halves. Unreadable or absent splits are skipped.
    The raw penult radii of the splits in `keep` are returned too (the per-tap target reuses them)."""
    ev, od = NC.split_halves(ROWS["H"][1] - ROWS["H"][0])
    out, cache = {}, {}
    for s in splits:
        if not d.has("penult", s):
            continue
        with run.timed("target:harm"):
            ys, rs = d.labels(s), d.rows(s)
            if not (np.array_equal(rs, np.arange(*ROWS["H"])) and np.array_equal(ys, yH)):
                raise SystemExit(f"[t2] {d.path} {s}: rows / labels do not pair with clean test rows 0-1999 (D4)")
            r = C.knn_kth(Xp_tr, d.acts("penult", s), KNN_K)
            lr = np.log(r + 1e-12)
            if s in keep:
                cache[s] = r
        with run.timed("target:overconf"):
            hs = C.head_stats(d.logits(s))
            ok = hs["argmax"] == ys
            msp = hs["msp"]
        out[s] = {tag: {"med_lr": float(np.median(lr[ix])), "acc": float(ok[ix].mean()), "msp": float(msp[ix].mean())}
                  for tag, ix in (("all", slice(None)), ("even", ev), ("odd", od))}
    return out, cache


def harm_rows(per, splits, tag, ref_med, ref_acc, spread):
    """[(split, H, cost_pt)] for the splits present in `per`."""
    rows = []
    for s in splits:
        if s in per and spread > 0:
            r = per[s][tag]
            rows.append((s, (r["med_lr"] - ref_med) / spread, 100.0 * (ref_acc - r["acc"])))
    return rows


def tap_shift(d, taps_c, Xp_tr, Xp_te, hs_te, run, pen_cache=None):
    """Per-sample shift information per tap (kNN radius) against the head statistics, corrupt rows 0-1999 at s1 / s3 vs
    clean rows B. Selection on the even rows, evaluation on the odd rows (and the reverse for the noise estimate).
    pen_cache: {split: penult radii} already computed by split_pass (identical numbers, computed once)."""
    tsplits = [C.csplit(c, s) for c in C.DISC for s in TAP_SEVS]
    ps, pe = NC.split_halves(ROWS["H"][1] - ROWS["H"][0])
    ns, ne = NC.split_halves(ROWS["B"][1] - ROWS["B"][0])
    B = slice(*ROWS["B"])
    pen_cache = pen_cache or {}
    tab = {}
    for t in taps_c:
        if not all(d.has(t, s) for s in tsplits):
            continue
        with run.timed(f"target:taps:{t}"):
            F = Xp_tr if t == "penult" else d.acts(t, "ref")
            neg = C.knn_kth(F, (Xp_te if t == "penult" else d.acts(t, "test"))[B], KNN_K)
            sel, evl = [], []
            for s in tsplits:
                pos = pen_cache[s] if (t == "penult" and s in pen_cache) else C.knn_kth(F, d.acts(t, s), KNN_K)
                sel.append(C.auc(pos[ps], neg[ns]))
                evl.append(C.auc(pos[pe], neg[ne]))
            tab[t] = {"sel": _mean(sel), "eval": _mean(evl)}
    head = {}
    with run.timed("target:taps:head"):
        negh = {h: C.ERR_SIGN[h] * hs_te[h][B] for h in HEADS}
        acc = {h: ([], []) for h in HEADS}
        for s in tsplits:
            if not d.readable(s) or s not in d.splits:
                continue
            hs = C.head_stats(d.logits(s))
            for h in HEADS:
                p = C.ERR_SIGN[h] * hs[h]
                acc[h][0].append(C.auc(p[ps], negh[h][ns]))
                acc[h][1].append(C.auc(p[pe], negh[h][ne]))
        head = {h: {"sel": _mean(v[0]), "eval": _mean(v[1])} for h, v in acc.items()}
    order = [t for t in taps_c if t in tab]
    out = {"n_splits": len(tsplits), "auc_by_tap": tab, "auc_head": head, "candidates": order}
    bt = _best_by({t: tab[t]["sel"] for t in order}, order)
    bh = _best_by({h: head[h]["sel"] for h in HEADS}, HEADS)
    out.update({"best_tap": bt, "best_head": bh,
                "p6b": _sub(tab[bt]["eval"], head[bh]["eval"]) if bt and bh else None})
    bt2 = _best_by({t: tab[t]["eval"] for t in order}, order)                   # roles swapped: the noise estimate
    bh2 = _best_by({h: head[h]["eval"] for h in HEADS}, HEADS)
    out["p6b_swapped"] = _sub(tab[bt2]["sel"], head[bh2]["sel"]) if bt2 and bh2 else None
    npen = [t for t in order if t != "penult"]
    bnp = _best_by({t: tab[t]["sel"] for t in npen}, npen)
    out["best_prepenult_tap"] = bnp
    out["penult_deficit"] = _sub(tab[bnp]["eval"], tab["penult"]["eval"]) if bnp and "penult" in tab else None
    return out


# =====================================================================================================================
# the probe
# =====================================================================================================================
def probe(d, spec, run):
    """Every coordinate and target of one fit-layout dump. Returns the probe.json body (ProbeRun.finish adds the header)."""
    uid = spec["id"]
    tp = d.b4.get("taps") or {}
    taps = list(tp.get("all") or d.taps)
    dups = [t for t in (tp.get("dup_of_penult") or []) if t != "penult"]
    spatial = dict(tp.get("spatial") or {})
    depth = dict(tp.get("depth_frac") or {})
    pre = tp.get("pre")
    if "penult" not in taps:
        raise SystemExit(f"[t2] {uid}: {d.path} has no penult tap")
    if not np.array_equal(d.rows("test"), np.arange(*ROWS["TEST"])):
        raise SystemExit(f"[t2] {uid}: test rows are not 0-4999 (the D4 fit layout)")
    ytr, yte = d.labels("ref"), d.labels("test")
    Lte = d.logits("test")
    hs = C.head_stats(Lte)
    am = hs["argmax"]
    W, _b = d.head()
    wrec, hc = d.b4.get("weights") or {}, d.b4.get("head_check") or {}
    inst = {"weights_status": wrec.get("status"), "head_check_status": hc.get("status"),
            "head_identity_rel_max": hc.get("identity_rel_max"), "pre_tap": pre, "pre_rule": NC.pre_tap(taps, spatial, dups),
            "dup_of_penult": dups, "missing": list(d.b4.get("missing") or []), "extras_present": d.b4.get("extras_present"),
            "n_ref": int(len(ytr)), "n_test": int(len(yte)), "layout": d.layout}
    inst["pre_agrees"] = inst["pre_rule"] == pre

    # ---- coordinates, every tap ----
    coords = {"taps": {}}
    Xp_tr = Xp_te = Ctr = None
    for t in taps:
        if not (d.has(t, "ref") and d.has(t, "test")):
            continue
        with run.timed(f"tap:{t}"):
            Xtr, Xte = d.acts(t, "ref"), d.acts(t, "test")
            c, Ct, _Sb = tap_coords(Xtr, ytr, Xte, yte, am, W if t == "penult" else None)
            if c is None:
                continue
            c.update({"depth_frac": _f(depth.get(t)), "spatial": bool(spatial.get(t, False)), "dup_of_penult": t in dups})
            coords["taps"][t] = c
            if t == "penult":
                Xp_tr, Xp_te, Ctr = Xtr, Xte, Ct
        say(f"{uid} {t:14s} d {c['dim']:5d} nc1_tr {c['nc1_tr']} plnc1 {c['plnc1_te']} g_cv {c['g_cv_te']}")
    if Xp_tr is None:
        raise SystemExit(f"[t2] {uid}: penult coordinates undefined")
    inst["nc1_pinv_rel_dev_max"] = max([v["nc1_pinv_rel_dev"] for v in coords["taps"].values()
                                        if v["nc1_pinv_rel_dev"] is not None] or [0.0])

    with run.timed("block:penult"):
        pen = dict(coords["taps"]["penult"])
        ok_te = am == yte
        pen.update({"msp_mean_te": _f(hs["msp"].mean()), "msp_def_te": _f(1.0 - hs["msp"].mean()),
                    "sat999_te": _f((hs["msp"] >= 0.999).mean()), "gap_mean_te": _f(hs["gap"].mean()),
                    "maxlogit_mean_te": _f(hs["maxlogit"].mean()), "ent_mean_te": _f(hs["entropy"].mean()),
                    "acc_te": _f(ok_te.mean()), "err_te": _f(1.0 - ok_te.mean()),
                    "acc_ref": _f((d.logits("ref").argmax(1) == ytr).mean()), "ece_te": C.ece(hs["msp"], ok_te),
                    "ncc_disagree_te": _sub(1.0, pen["ncc_agree_te"]) if pen["ncc_agree_te"] is not None else None,
                    "tt_gap_log10": _sub(_lg(pen["nc1_tetr"]), _lg(pen["nc1_tr"])),
                    "te_tr_gap_log10": _sub(_lg(pen["nc1_te"]), _lg(pen["nc1_tr"])),
                    "cdepth_pl": _sub(_lg((coords["taps"].get(pre) or {}).get("plnc1_te")), _lg(pen["plnc1_te"]))})
        coords["penult"] = pen
        coords["pre_tap"] = pre
        coords["pre"] = coords["taps"].get(pre)
        pts = [(c["depth_frac"], _lg(c["nc1_tr"])) for c in coords["taps"].values()
               if not c["dup_of_penult"] and c["depth_frac"] is not None and _lg(c["nc1_tr"]) is not None]
        fit = _ols([p[0] for p in pts], [p[1] for p in pts])
        coords["equisep"] = {"n_taps": len(pts), "intercept": fit[0] if fit else None, "slope": fit[1] if fit else None,
                             "pearson": C.pearson([p[0] for p in pts], [p[1] for p in pts]) if len(pts) >= 3 else None,
                             "note": "log10 nc1_tr against depth fraction over non-duplicate taps (He & Su 2023), INFO"}

    T, SE = {}, {}
    A, B, H = slice(*ROWS["A"]), slice(*ROWS["B"]), slice(*ROWS["H"])
    ev5, od5 = NC.split_halves(ROWS["TEST"][1])

    # ---- DO-3: the train-referenced density false alarm, and the held-out-calibrated control (IC-P2) ----
    with run.timed("target:do3"):
        r_ref = C.knn_kth(Xp_tr, Xp_tr, KNN_K, exclude_self=True)
        lr_ref = np.log(r_ref + 1e-12)
        q50, q95 = (float(v) for v in np.quantile(lr_ref, [0.5, 0.95]))
        r_te = C.knn_kth(Xp_tr, Xp_te, KNN_K)
        lr_te = np.log(r_te + 1e-12)
        flag = lr_te > q95
        T["do3"] = {"fpr_trainref": _f(flag.mean()), "fpr_trainref_even": _f(flag[ev5].mean()),
                    "fpr_trainref_odd": _f(flag[od5].mean()), "q50_ref": q50, "q95_ref": q95,
                    "median_shift_test": _f(np.median(lr_te) - q50), "n": int(len(flag)), "k": KNN_K}
        SE["fpr_trainref"] = _half_se(T["do3"]["fpr_trainref_even"], T["do3"]["fpr_trainref_odd"])
        p = C.conformal_p(r_te[A], r_te[B])
        T["ic_p2"] = {"alpha": ALPHA, "fpr": _f((p <= ALPHA).mean()), "n_cal": int(len(r_te[A])), "n_test": int(len(p)),
                      "curve": {f"{a:.2f}": _f((p <= a).mean()) for a in ALPHAS_CURVE},
                      "rows": {"cal": list(ROWS["A"]), "test": list(ROWS["B"])}}

    # ---- margin vs distance vs head on clean errors (rows 0-4999), and P3b ----
    with run.timed("target:margin"):
        Dc = NC.center_dists(Xp_te, Ctr)
        Ds = np.sort(Dc, axis=1)
        wrong = am != yte
        S = {"margin": -(Ds[:, 1] - Ds[:, 0]), "d1": Ds[:, 0], **{h: C.ERR_SIGN[h] * hs[h] for h in HEADS}}

        def aucs(mask):
            return {k: C.auc(v[mask & wrong], v[mask & ~wrong]) for k, v in S.items()}

        def mblock(mask):
            a = aucs(mask)
            return {"n_wrong": int((mask & wrong).sum()), "n_right": int((mask & ~wrong).sum()),
                    **{f"auc_{k}": v for k, v in a.items()}, "lead_md": _sub(a["margin"], a["d1"]),
                    "gap_minus_msp": _sub(a["gap"], a["msp"]), "margin_minus_msp": _sub(a["margin"], a["msp"])}
        allm = np.ones(len(yte), dtype=bool)
        mev, mod = np.zeros(len(yte), dtype=bool), np.zeros(len(yte), dtype=bool)
        mev[ev5], mod[od5] = True, True
        T["margin"] = mblock(allm)
        hv, ho = mblock(mev), mblock(mod)
        SE["lead_md"] = _half_se(hv["lead_md"], ho["lead_md"])
        SE["gap_minus_msp"] = _half_se(hv["gap_minus_msp"], ho["gap_minus_msp"])
    with run.timed("target:p3b"):
        selA = np.zeros(len(yte), dtype=bool)
        selA[A] = True
        aA = aucs(selA)
        pick = _best_by({h: aA[h] for h in HEADS}, HEADS)
        evE = ~selA
        yE = wrong[evE]
        rec = {"head_pick": pick, "pick_rows": list(ROWS["A"]), "eval_rows": [list(ROWS["H"]), list(ROWS["B"])],
               "auc_head_on_A": {h: aA[h] for h in HEADS}, "n_wrong_eval": int(yE.sum()), "ci_level": CI_LEVEL,
               "nboot": int(run.nboot), "d": None, "ci_lo": None, "ci_hi": None}
        if pick is not None:
            sm, sh = S["margin"][evE], S[pick][evE]
            rec["d"] = _sub(C.auc(sm[yE], sm[~yE]), C.auc(sh[yE], sh[~yE]))
            rec["ci_lo"], rec["ci_hi"] = boot_auc_diff(sm, sh, yE, run.nboot, (uid, "p3b"))
            rec["margin_minus_msp_eval"] = _sub(C.auc(sm[yE], sm[~yE]),
                                                C.auc(S["msp"][evE][yE], S["msp"][evE][~yE]))
        T["p3b"] = rec

    # ---- OOD: CIFAR-100 (O3) and SVHN (P4) vs clean rows B ----
    with run.timed("target:ood"):
        trl2 = C.l2n(Xp_tr)
        neg = {"knn_l2": C.knn_kth(trl2, C.l2n(Xp_te[B]), KNN_K), "knn": r_te[B],
               **{h: C.ERR_SIGN[h] * hs[h][B] for h in HEADS}}
        nev, nod = NC.split_halves(ROWS["B"][1] - ROWS["B"][0])
        T["ood"] = {}
        for name in ("cifar100", "svhn"):
            s = f"ood__{name}"
            if not d.has("penult", s):
                T["ood"][name] = None
                continue
            Xo = d.acts("penult", s)
            hso = C.head_stats(d.logits(s))
            pos = {"knn_l2": C.knn_kth(trl2, C.l2n(Xo), KNN_K), "knn": C.knn_kth(Xp_tr, Xo, KNN_K),
                   **{h: C.ERR_SIGN[h] * hso[h] for h in HEADS}}
            pev, pod = NC.split_halves(len(Xo))
            T["ood"][name] = {"n_pos": int(len(Xo)), "n_neg": int(ROWS["B"][1] - ROWS["B"][0]), **ood_block(pos, neg)}
            SE[f"ood_{name}"] = _half_se(ood_block(pos, neg, pev, nev)["knn_l2_minus_besthead"],
                                         ood_block(pos, neg, pod, nod)["knn_l2_minus_besthead"])

    # ---- harm (O2, P5a, P5c) and overconfidence under shift (O4, S7): one pass over the corrupt splits ----
    disc = [C.csplit(c, s) for c in C.DISC for s in C.SEVS]
    conf_only = [C.csplit(c, s) for c in C.HOLDOUT + C.EXTRA for s in C.SEVS]
    tsplits = {C.csplit(c, s) for c in C.DISC for s in TAP_SEVS}
    per, pen_cache = split_pass(d, Xp_tr, disc + conf_only, yte[H], run, keep=tsplits)
    with run.timed("target:harm"):
        spread = q95 - q50
        evH, odH = NC.split_halves(ROWS["H"][1] - ROWS["H"][0])
        lrH, okH = lr_te[H], ok_te[H]
        clean = {tag: (float(np.median(lrH[ix])), float(okH[ix].mean()))
                 for tag, ix in (("all", slice(None)), ("even", evH), ("odd", odH))}
        paired = {tag: harm_rows(per, disc, tag, *clean[tag], spread) for tag in clean}
        harm = harm_fit([r[1] for r in paired["all"]], [r[2] for r in paired["all"]])
        harm.update({"form": "paired: clean median log r10 and accuracy of test rows 0-1999 (the corrupt rows' sources)",
                     "spread_q95_q50": spread, "n_disc_splits": len(paired["all"]),
                     "per_split": {s: {"H": h, "cost": c} for s, h, c in paired["all"]}})
        halves = [harm_fit([r[1] for r in paired[t]], [r[2] for r in paired[t]]) for t in ("even", "odd")]
        SE["harm_slope"] = _half_se(halves[0]["slope"], halves[1]["slope"])
        SE["O2"] = _half_se(halves[0]["H10"], halves[1]["H10"])
        T["harm"] = harm
        med_t, acc_t = float(np.median(lr_te)), float(ok_te.mean())
        com = harm_rows(per, disc, "all", med_t, acc_t, spread)
        hcom = harm_rows(per, conf_only, "all", med_t, acc_t, spread)
        hc_ = harm_fit([r[1] for r in com], [r[2] for r in com])
        hc_.update(hold_band([r[1] for r in com], [r[2] for r in com]))
        hb = hold_band([r[1] for r in hcom], [r[2] for r in hcom])
        hc_.update({"form": "AH-2(b) as committed: clean median log r10 and accuracy of test rows 0-4999",
                    "holdout_n_splits": len(hcom), "holdout_hold_n": hb["hold_n"], "holdout_hold_viol": hb["hold_viol"],
                    "holdout_max_cost_in_band": hb["max_cost_in_band"],
                    "per_split": {s: {"H": h, "cost": c} for s, h, c in com + hcom}})
        T["harm_committed"] = hc_
    with run.timed("target:overconf"):
        oc = {tag: [per[s][tag]["msp"] - per[s][tag]["acc"] for s in disc if s in per] for tag in ("all", "even", "odd")}
        T["overconf"] = {"shift_mean": _mean(oc["all"]), "n_splits": len(oc["all"]),
                         "clean_te": _f(hs["msp"].mean() - ok_te.mean()),
                         "per_split": {s: per[s]["all"]["msp"] - per[s]["all"]["acc"] for s in disc if s in per}}
        SE["O4"] = _half_se(_mean(oc["even"]), _mean(oc["odd"]))
    say(f"{uid} do3 {T['do3']['fpr_trainref']} lead {T['margin']['lead_md']} H10 {harm['H10']} "
        f"O4 {T['overconf']['shift_mean']}")

    # ---- which tap carries per-sample shift information, against the head (P6b) ----
    taps_c = [t for t in coords["taps"] if t not in dups]
    T["taps"] = tap_shift(d, taps_c, Xp_tr, Xp_te, hs, run, pen_cache)
    del pen_cache
    SE["p6b"] = _half_se(T["taps"]["p6b"], T["taps"]["p6b_swapped"])
    SE["O3"] = SE.get("ood_cifar100")

    outcomes = {"O2": T["harm"]["H10"], "O3": (T["ood"].get("cifar100") or {}).get("knn_l2_minus_besthead"),
                "O4": T["overconf"]["shift_mean"],
                "definitions": {"O2": "targets.harm.H10", "O3": "targets.ood.cifar100.knn_l2_minus_besthead",
                                "O4": "targets.overconf.shift_mean"}}
    keep = ("id", "roles", "family", "depth_family", "run_group", "source", "full_recipe", "trained", "penult", "knob")
    return {"schema": SCHEMA, "constants": CONSTANTS, "spec": {k: spec.get(k) for k in keep},
            "instrument": inst, "coords": coords, "targets": T, "targets_se": SE, "outcomes": outcomes}


def check_role(spec, phase):
    roles = set(spec.get("roles") or [])
    if not roles & set(ROLES[phase]):
        raise SystemExit(f"[t2] {spec['id']} (roles {sorted(roles)}) is not a T2 unit of the {phase} phase "
                         f"(allowed roles: {ROLES[phase]}; docs/plans/B4_INTEGRATION.md D9)")


def main(argv=None):
    ap = C.probe_argparser("t2", description="T2 collapse probe (docs/plans/T2_COLLAPSE.md)")
    a = C.parse_probe_args(ap, argv)
    if a.selftest:
        if a.selftest_out and os.path.exists(a.selftest_out):
            raise SystemExit(f"[t2] {a.selftest_out} exists: a self-test record is never overwritten")
        rep = selftest()
        for c in rep["checks"]:
            print(f"  {c['status']}  {c['check']}")
        print(f"[t2] selftest {rep['status']} ({rep['n_pass']} pass, {rep['n_fail']} fail)")
        C.write_selftest(a.selftest_out, rep)
        return 0 if rep["status"] == "PASS" else 1
    run = ProbeRunT2("t2", __file__, a)                  # guards the output before anything is read
    reg = C.load_registry(a.registry)
    spec = C.unit_spec(reg, a.unit)
    check_role(spec, a.phase)
    d = C.open_unit(reg, a.unit, "fit", a.phase)          # the D7 seal
    body = probe(d, spec, run)
    path = run.finish(body, dumps=[d])
    say(f"{a.unit} {a.phase}: wrote {path}")
    return 0


# =====================================================================================================================
# self-test: synthetic fit-layout dumps with known answers (CPU, seconds)
# =====================================================================================================================
SYN = {"K": K, "dim": 64, "m": 4.0, "sigma": 1.0, "n_ref": 6000, "null_shift": 0.3, "noise_per_sev": 0.25}


def _simplex(k, dim, norm, rng):
    E = np.eye(k) - 1.0 / k
    U, _, _ = np.linalg.svd(rng.standard_normal((dim, k)), full_matrices=False)
    M = E @ U.T
    return norm * M / np.linalg.norm(M, axis=1, keepdims=True)


def make_synth_dump(root, rng, syn=SYN):
    """A fit-layout dump in the scripts/b4_extract.py format: Gaussian classes (sigma) at simplex-ETF means (norm m), head
    W = the means (b = 0). Taps: stem (a noisy 32-d linear view), mid (penult + 2 sigma noise), dupl (a float16 copy of
    the penult, listed in dup_of_penult), penult. Corrupt split (c, s): within-class noise sigma (1 + 0.25 s) plus a
    shift 0.3 s along a head-null direction u_c. One confirmation-only split (corrupt__frost__s3, X4 taps). Returns the
    closed-form answers."""
    k, dim, m, sig = syn["K"], syn["dim"], syn["m"], syn["sigma"]
    M = _simplex(k, dim, m, rng)
    Pn = np.eye(dim) - np.linalg.pinv(M) @ M
    U = Pn @ rng.standard_normal((dim, len(C.DISC) + 1))
    U /= np.linalg.norm(U, axis=0, keepdims=True)
    R = rng.standard_normal((dim, 32)) / np.sqrt(dim)
    W, b = M.copy(), np.zeros(k)
    taps = ["stem", "mid", "dupl", "penult"]
    x4 = ["stem", "mid", "penult"]
    splits, files = [], {}

    def save(split, Xp, y, rows, which):
        n = len(Xp)
        feats = {"stem": Xp @ R + 0.5 * rng.standard_normal((n, 32)), "mid": Xp + 2.0 * sig * rng.standard_normal((n, dim)),
                 "dupl": Xp, "penult": Xp}
        for t in which:
            pen32 = t == "penult" and (split in ("ref", "test") or split.startswith("ood__"))
            os.makedirs(os.path.join(root, "acts", t), exist_ok=True)
            np.save(os.path.join(root, "acts", t, f"{split}.npy"), feats[t].astype(np.float32 if pen32 else np.float16))
        L = Xp @ W.T + b
        for sub in ("logits", "labels", "rows", "preds"):
            os.makedirs(os.path.join(root, sub), exist_ok=True)
        np.save(os.path.join(root, "logits", f"{split}.npy"), L.astype(np.float32))
        np.save(os.path.join(root, "labels", f"{split}.npy"), np.asarray(y, dtype=np.int64))
        np.save(os.path.join(root, "rows", f"{split}.npy"), np.asarray(rows, dtype=np.int64))
        P = C.softmax(L)
        np.savez(os.path.join(root, "preds", f"{split}.npz"), argmax=L.argmax(1).astype(np.int64),
                 maxprob=P.max(1).astype(np.float32))
        splits.append(split)

    ytr = np.arange(syn["n_ref"]) % k
    save("ref", M[ytr] + sig * rng.standard_normal((len(ytr), dim)), ytr, np.arange(len(ytr)), taps)
    yte = rng.permutation(np.arange(5000) % k)          # shuffled: the even/odd split halves (D4) must each hold all K
    save("test", M[yte] + sig * rng.standard_normal((5000, dim)), yte, np.arange(5000), taps)
    yH = yte[:2000]
    for i, c in enumerate(C.DISC):
        for s in C.SEVS:
            sd = sig * (1.0 + syn["noise_per_sev"] * s)
            save(C.csplit(c, s), M[yH] + sd * rng.standard_normal((2000, dim)) + syn["null_shift"] * s * U[:, i], yH,
                 np.arange(2000), taps)
    save(C.csplit("frost", 3), M[yH] + 1.5 * sig * rng.standard_normal((2000, dim)) + 0.9 * U[:, -1], yH,
         np.arange(2000), x4)
    for name, keep, off in (("cifar100", 0.5, 1.1), ("svhn", 0.0, 1.6)):   # near: half a class mean, 1.1 sigma; far: noise
        yo = rng.integers(0, k, 2000)
        save(f"ood__{name}", keep * M[yo] + off * sig * rng.standard_normal((2000, dim)), yo, np.arange(2000), taps)
    os.makedirs(os.path.join(root, "head"), exist_ok=True)
    np.savez(os.path.join(root, "head", "head.npz"), W=W.astype(np.float32), b=b.astype(np.float32))
    b4 = {"schema": "b4_dump/1", "unit": "synthetic", "layout": "fit", "sealed": False, "roles": ["D"],
          "arch": "synthetic_resnet", "family": "resnet", "weights": {"source": "synthetic", "status": "PASS"},
          "taps": {"all": taps, "spatial": {"stem": True, "mid": True, "dupl": True},
                   "depth_frac": {"stem": 0.1, "mid": 0.6, "dupl": 0.95, "penult": 1.0}, "dup_of_penult": ["dupl"],
                   "pre": "mid", "functional": {"stem": "stem", "s1end": "stem", "s2end": "mid", "pre": "mid",
                                                "penult": "penult"}, "x4": x4},
          "head_check": {"status": "PASS", "identity_rel_max": 0.0}, "missing": [], "extras_present": None}
    meta = {"source": "synthetic", "layers": taps, "splits": splits, "n_classes": k, "n_test": 5000, "b4": b4}
    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump(meta, f)
    a = m * m / (k - 1)                                  # Sigma_B eigenvalue on each of the K-1 class directions
    return {"nc1": sig ** 2 * (k - 1) ** 2 / (k * m * m), "g": (a + sig ** 2) / sig ** 2}


def _strict_load(path):
    def bad(tok):
        raise ValueError(f"non-finite token {tok} in {path}")
    with open(path) as f:
        return json.load(f, parse_constant=bad)


def _strip(o, ignore=("timing_s", "max_rss_mb", "env", "code", "created", "created_utc", "host", "versions", "out",
                      "tag", "unseal")):
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if k not in ignore}
    if isinstance(o, list):
        return [_strip(v) for v in o]
    return o


def selftest(workdir=None):
    checks = []

    def chk(name, ok, val=None):
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "value": C._clean(val)})

    own = workdir is None
    workdir = workdir or tempfile.mkdtemp(prefix="t2_selftest_")
    _QUIET[0] = True
    try:
        rng = np.random.default_rng(20260925)
        dump = os.path.join(workdir, "b4d_synthetic", "dump")
        ans = make_synth_dump(dump, rng)
        sealed = os.path.join(workdir, "b4d_synthetic_sealed", "dump")
        shutil.copytree(dump, sealed)
        with open(os.path.join(sealed, "SEALED.json"), "w") as f:
            json.dump({"unit": "st_sealed"}, f)
        base = {"family": "synthetic", "depth_family": "synthetic", "run_group": "synthetic", "source": "synthetic",
                "full_recipe": True, "trained": True, "penult": SYN["dim"], "knob": None}
        reg = {"schema": "b4_models/1", "models": [
            dict(base, id="st_disc", roles=["D"], layouts={"fit": {"dump": dump, "sealed": False}}),
            dict(base, id="st_conf", roles=["C"], layouts={"fit": {"dump": dump, "sealed": False}}),
            dict(base, id="st_sealed", roles=["D"], layouts={"fit": {"dump": sealed, "sealed": False}})]}
        regp = os.path.join(workdir, "models.json")
        with open(regp, "w") as f:
            json.dump(reg, f)
        out1, out2, out3 = (os.path.join(workdir, "out", n) for n in ("st_disc", "st_disc_r1", "st_conf"))

        def run(unit, phase, out):
            try:
                return main(["--registry", regp, "--unit", unit, "--phase", phase, "--out", out])
            except SystemExit as e:
                return f"refused: {e}"

        rc = run("st_disc", "discovery", out1)
        p1 = os.path.join(out1, "probe.json")
        chk("discovery probe runs and writes probe.json", rc == 0 and os.path.isfile(p1), rc)
        rec = _strict_load(p1)
        chk("probe.json is strictly finite JSON (allow_nan=False)", True)
        pen, T = rec["coords"]["penult"], rec["targets"]
        chk("nc1_tr = sigma^2 (K-1)^2 / (K m^2) within 5% (simplex ETF, isotropic classes)",
            abs(pen["nc1_tr"] / ans["nc1"] - 1) < 0.05, [pen["nc1_tr"], ans["nc1"]])
        chk("trimmed nc1 = verbatim landmarks.py:71 nc1 when the pinv is stable (1e-6)",
            rec["instrument"]["nc1_pinv_rel_dev_max"] < 1e-6, rec["instrument"]["nc1_pinv_rel_dev_max"])
        chk("g_cv_te = (a + sigma^2) / sigma^2 within 10% (label-free spectral gap)",
            abs(pen["g_cv_te"] / ans["g"] - 1) < 0.10, [pen["g_cv_te"], ans["g"]])
        chk("khat_te = K - 1 (the class spike)", pen["khat_te"] == K - 1, pen["khat_te"])
        Xte = np.load(os.path.join(dump, "acts", "penult", "test.npy")).astype(np.float64)
        Lte = np.load(os.path.join(dump, "logits", "test.npy")).astype(np.float64)
        lf = NC.label_free_coords(Xte, Lte.argmax(1), K)
        chk("label-free candidates equal atlas.b4_collapse.label_free_coords (g_cv, topk_frac; plnc1 verbatim)",
            lf["g_cv"] == pen["g_cv_te"] and lf["topk_frac"] == pen["topk_frac_te"] and lf["plnc1"] == pen["plnc1_te_verbatim"],
            [lf["g_cv"], pen["g_cv_te"], lf["plnc1"], pen["plnc1_te_verbatim"]])
        chk("plnc1_te <= 1.02 nc1_te (argmax labels are the nearest-mean labels here)",
            pen["plnc1_te"] <= 1.02 * pen["nc1_te"], [pen["plnc1_te"], pen["nc1_te"]])
        chk("NC4: nearest train centre agrees with the head (> 0.97)", pen["ncc_agree_te"] > 0.97, pen["ncc_agree_te"])
        chk("DO-3: train-referenced FPR ~ 0.05 when train and test are exchangeable (+-0.03)",
            abs(T["do3"]["fpr_trainref"] - 0.05) < 0.03, T["do3"]["fpr_trainref"])
        chk("IC-P2: conformal FPR at alpha 0.05 (cal rows A, test rows B) ~ 0.05 (+-0.03)",
            abs(T["ic_p2"]["fpr"] - 0.05) < 0.03 and T["ic_p2"]["n_cal"] == 1500 and T["ic_p2"]["n_test"] == 1500,
            T["ic_p2"])
        hm = T["harm"]
        chk("harm: 30 discovery splits, slope > 0, Spearman(H, cost) > 0.5, H10 finite",
            hm["n_disc_splits"] == 30 and hm["slope"] > 0 and hm["spearman"] > 0.5 and hm["H10"] is not None,
            [hm["n_disc_splits"], hm["slope"], hm["spearman"], hm["H10"]])
        direct = []
        for c in C.DISC:
            for s in C.SEVS:
                L = np.load(os.path.join(dump, "logits", f"{C.csplit(c, s)}.npy")).astype(np.float64)
                y = np.load(os.path.join(dump, "labels", f"{C.csplit(c, s)}.npy"))
                direct.append(C.softmax(L).max(1).mean() - (L.argmax(1) == y).mean())
        chk("O4 = mean over the 30 splits of mean MSP - accuracy (direct recomputation, 1e-12)",
            abs(rec["outcomes"]["O4"] - float(np.mean(direct))) < 1e-12, [rec["outcomes"]["O4"], float(np.mean(direct))])
        chk("O2 = harm.H10 and O3 = ood.cifar100.knn_l2_minus_besthead",
            rec["outcomes"]["O2"] == hm["H10"] and rec["outcomes"]["O3"] == T["ood"]["cifar100"]["knn_l2_minus_besthead"])
        oo = T["ood"]
        chk("OOD: raw 10-NN AUROC far (SVHN) >= near (CIFAR-100) > 0.6; L2-kNN and best-head AUROCs defined",
            oo["svhn"]["auc_knn"] >= oo["cifar100"]["auc_knn"] > 0.6
            and all(oo[n]["auc_knn_l2"] is not None and oo[n]["best_head"] in HEADS for n in ("cifar100", "svhn")),
            [oo["cifar100"]["auc_knn"], oo["svhn"]["auc_knn"], oo["cifar100"]["auc_knn_l2"]])
        pb = T["p3b"]
        chk("P3b: head picked on rows A; the 99% bootstrap CI brackets d",
            pb["head_pick"] in HEADS and pb["ci_lo"] <= pb["d"] <= pb["ci_hi"] and pb["nboot"] == C.NBOOT["discovery"],
            [pb["head_pick"], pb["ci_lo"], pb["d"], pb["ci_hi"]])
        tt = T["taps"]
        chk("taps: the duplicate tap is excluded; P6b finite; the corruption is visible at every tap (AUROC > 0.6)",
            "dupl" not in tt["auc_by_tap"] and tt["p6b"] is not None
            and all(v["eval"] > 0.6 for v in tt["auc_by_tap"].values()), [tt["candidates"], tt["p6b"]])
        chk("pre tap = 'mid' = the D2 rule; cdepth_pl > 0 (mid is noisier than the penult)",
            rec["coords"]["pre_tap"] == "mid" and rec["instrument"]["pre_agrees"] and pen["cdepth_pl"] > 0,
            [rec["coords"]["pre_tap"], pen["cdepth_pl"]])
        read = rec["dumps"][0]["splits_read"]
        chk("discovery never read the confirmation-only split corrupt__frost__s3",
            "corrupt__frost__s3" not in read and T["harm_committed"]["holdout_n_splits"] == 0, read[-3:])
        ts = rec["timing_s"]
        chk("contract: timing_s per tap and target + total; max_rss_mb; code.sha256 / core_sha256 / collapse_sha256; "
            "repo_commit key",
            all(k in ts for k in ("tap:penult", "tap:mid", "target:do3", "target:harm", "target:taps:mid", "total"))
            and "max_rss_mb" in rec and all(len(rec["code"].get(k) or "") == 64 for k in ("sha256", "core_sha256",
                                                                                         "collapse_sha256"))
            and rec["code"]["collapse_sha256"] == C.code_sha256(os.path.abspath(NC.__file__)) and "repo_commit" in rec["code"],
            sorted(ts)[:6])
        rc_again = run("st_disc", "discovery", out1)
        chk("append-only: a second run into the same --out is refused", isinstance(rc_again, str), rc_again)
        rc2 = run("st_disc", "discovery", out2)
        same = rc2 == 0 and _strip(_strict_load(os.path.join(out2, "probe.json"))) == _strip(rec)
        chk("deterministic: a second discovery run equals the first (replay keys ignored)", same, rc2)
        rc3 = run("st_conf", "confirmation", out3)
        rec3 = _strict_load(os.path.join(out3, "probe.json")) if rc3 == 0 else {}
        hc3 = (rec3.get("targets") or {}).get("harm_committed") or {}
        chk("confirmation reads the holdout split (HOLD band on holdout splits reported)",
            rc3 == 0 and "corrupt__frost__s3" in rec3["dumps"][0]["splits_read"] and hc3.get("holdout_n_splits") == 1,
            [rc3, hc3.get("holdout_n_splits")])
        rc4 = run("st_sealed", "discovery", os.path.join(workdir, "out", "st_sealed"))
        chk("the D7 seal: a SEALED dump is refused in discovery", isinstance(rc4, str) and "SEALED" in rc4, rc4)
        rc5 = run("st_conf", "discovery", os.path.join(workdir, "out", "st_conf_disc"))
        chk("role guard: a confirmation unit is refused in the discovery phase", isinstance(rc5, str), rc5)
        H = [0.0, 0.1, 0.2, 0.3, 0.4]
        hf = harm_fit(H, [2.0 + 20.0 * h for h in H])
        chk("harm_fit known answer: cost = 2 + 20 H -> slope 20, H10 0.4", abs(hf["slope"] - 20) < 1e-12
            and abs(hf["H10"] - 0.4) < 1e-12 and harm_fit(H, [5.0 - h for h in H])["H10"] is None, hf)
        hb = hold_band([0.1, 0.2, 0.3, 0.25], [9.0, 3.0, 50.0, 8.0])
        chk("hold_band known answer: 3 splits in band, 1 violation, max 9", hb == {"hold_n": 3, "hold_viol": 1,
                                                                                  "max_cost_in_band": 9.0}, hb)
    except Exception as e:                                  # a crash is a FAIL, recorded with its message
        chk(f"self-test crashed: {type(e).__name__}: {e}", False)
    finally:
        _QUIET[0] = False
        if own:
            shutil.rmtree(workdir, ignore_errors=True)
    n_fail = sum(c["status"] == "FAIL" for c in checks)
    return {"status": "PASS" if checks and n_fail == 0 else "FAIL", "n_pass": len(checks) - n_fail, "n_fail": n_fail,
            "checks": checks, "code_sha256": C.code_sha256(os.path.abspath(__file__)),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "synthetic": SYN}


if __name__ == "__main__":
    sys.exit(main())
