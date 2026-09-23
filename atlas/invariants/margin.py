"""
margin.py -- where the backbone's confident mistakes sit in the map: on a ridge between two class
valleys (small top-2 margin) or deep inside a wrong valley (margin like a correct point).
ATLAS_STATUS row 9 (A3). The definition is frozen with its pre-registration in
experiments/queue/margin_v1_resnet20_s1.yaml; archaeology and design in docs/plans/A3_MARGIN.md.

  margin_typeb : per layer, on the clean test split, with class centers = class means of the TRAIN
      reference (true labels), as in class_centers. Per test sample, in the full pooled layer space
      (Euclidean, unnormalized):
        margin  = d(2) - d(1)   second-nearest minus nearest center distance
        dist    = d(1)          nearest-center distance ("absolute valley depth")
        maxprob = the backbone's max softmax probability (preds/test.npz; dumps store no logits)
        energy  = mean_j x_j^2  the Upgraded-Mod "energy" (mean squared activation, not logit energy)
      Groups: correct (argmax == label), wrong (argmax != label), type-b = wrong and maxprob > cut
      (strict; cut 0.7 as in Upgraded-Mod). Comparisons: correct vs type-b (headline), correct vs
      wrong, and confidence-matched (correct with maxprob > cut vs type-b, so maxprob cannot win
      through the selection alone). AUCs are ORIENTED: the error set is the positive class and each
      score is signed so that AUC > 0.5 is the pre-registered direction (small margin, large distance,
      low maxprob, low energy); DeLong standard errors and a paired DeLong test of margin minus dist.
      The cut is swept (rule 5).
      `legacy`: the Upgraded-Mod definitions as far as a dump allows (centers from the scored test
      split itself, true labels, in-sample; correct vs conf-wrong; direction-free AUC max(a, 1 - a);
      4-group subnet CLUSTER_DISTANCE; mean squared activation), for the M1 reproduction check only.

  Deterministic: nothing is drawn from ctx.rng, so adding it to a build leaves every other
  invariant's estimator draws unchanged (it is also registered last, cost "expensive"). A group that
  is empty or smaller than min_n gives None for the statistics that need it; it never raises for
  that (random-init null: no confident mistakes). Missing test predictions (a plumbing fault) raise.

Upgraded-Mod@d9683cd: session_experiments/sample_and_scale.py:21-24, 44-61 (0.895 / 0.763 / 0.757);
extract/axis_registry.py:33-57 (subnet CLUSTER_DISTANCE).
"""
import numpy as np
from scipy.stats import norm, rankdata, spearmanr

from ..registry import invariant
from ._util import class_stats

CUTS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
SIGN = {"margin": -1.0, "dist": 1.0, "maxprob": -1.0, "energy": -1.0}   # sign so that + = more error-like
QS = (0.1, 0.25, 0.5, 0.75, 0.9)
LEGACY_GROUPS = 4                                                         # axis_registry.py:19 N_GROUPS


def center_dists(X, C):
    """(N, K) Euclidean distances from the rows of X to the K rows of C."""
    X = np.asarray(X, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    return np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)


def top2_margin(D):
    """From an (N, K >= 2) distance matrix: (second-nearest minus nearest, nearest)."""
    Ds = np.sort(D, axis=1)
    return Ds[:, 1] - Ds[:, 0], Ds[:, 0]


def delong(pos, neg):
    """AUC = P(pos > neg) + P(tie)/2 for k scores on the same samples, and their DeLong covariance.
    pos (k, m), neg (k, n); midrank form (Sun & Xu 2014). Returns (auc (k,), cov (k, k))."""
    pos = np.atleast_2d(np.asarray(pos, dtype=np.float64))
    neg = np.atleast_2d(np.asarray(neg, dtype=np.float64))
    k, m = pos.shape
    n = neg.shape[1]
    auc, v10, v01 = np.zeros(k), np.zeros((k, m)), np.zeros((k, n))
    for i in range(k):
        tz = rankdata(np.concatenate([pos[i], neg[i]]))
        tx, ty = rankdata(pos[i]), rankdata(neg[i])
        auc[i] = (tz[:m].sum() - m * (m + 1) / 2.0) / (m * n)
        v10[i] = (tz[:m] - tx) / n            # share of negatives each positive beats
        v01[i] = 1.0 - (tz[m:] - ty) / m      # share of positives that beat each negative
    s10 = np.atleast_2d(np.cov(v10)) if m > 1 else np.zeros((k, k))
    s01 = np.atleast_2d(np.cov(v01)) if n > 1 else np.zeros((k, k))
    return auc, s10 / m + s01 / n


def _med(v):
    return float(np.median(v)) if len(v) else None


def _spearman(a, b):
    """Spearman rho, or None when it is undefined (fewer than 3 samples or a constant input)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 3 or a.max() == a.min() or b.max() == b.min():
        return None
    r = float(spearmanr(a, b).correlation)
    return r if np.isfinite(r) else None


def _scores(X, C, maxprob):
    X = np.asarray(X, dtype=np.float64)
    margin, dist = top2_margin(center_dists(X, C))
    return {"margin": margin, "dist": dist, "maxprob": np.asarray(maxprob, dtype=np.float64),
            "energy": (X ** 2).mean(1)}


def _aucs(S, pos, neg, min_n, tag):
    """Oriented AUC + DeLong SE of every score, errors (pos mask) vs correct (neg mask), and the paired
    DeLong test of margin minus dist. Every key is present; None when either group has < min_n rows."""
    m, n = int(pos.sum()), int(neg.sum())
    names = list(S)
    out = {f"n_pos_{tag}": m, f"n_neg_{tag}": n}
    if m < min_n or n < min_n:
        for s in names:
            out[f"auc_{s}_{tag}"] = None
            out[f"se_{s}_{tag}"] = None
        for sfx in ("", "_se", "_p"):
            out[f"margin_minus_dist_{tag}{sfx}"] = None
        return out
    auc, cov = delong(np.stack([SIGN[s] * S[s][pos] for s in names]),
                      np.stack([SIGN[s] * S[s][neg] for s in names]))
    for i, s in enumerate(names):
        out[f"auc_{s}_{tag}"] = float(auc[i])
        out[f"se_{s}_{tag}"] = float(np.sqrt(max(cov[i, i], 0.0)))
    i, j = names.index("margin"), names.index("dist")
    d = float(auc[i] - auc[j])
    se = float(np.sqrt(max(cov[i, i] + cov[j, j] - 2.0 * cov[i, j], 0.0)))
    out[f"margin_minus_dist_{tag}"] = d
    out[f"margin_minus_dist_{tag}_se"] = se
    out[f"margin_minus_dist_{tag}_p"] = float(2.0 * norm.sf(abs(d) / se)) if se > 0 else None
    return out


def _legacy(X, y, am, mp, cut, min_n, groups=LEGACY_GROUPS):
    """Upgraded-Mod definitions (sample_and_scale.py:44-61): centers = class means of the scored test split
    itself (true labels, in-sample, misclassified samples included), correct vs conf-wrong (wrong and
    maxprob > cut), per-sample AUC made direction-free, max(a, 1 - a). raw_auc_<s> = P(conf-wrong > correct).
    cluster_subnet (axis_registry.py:33-57): per channel group, nearest-center distance / its median over
    the same split, averaged over the groups; omitted when the layer width is not divisible by `groups`."""
    correct = am == y
    cw = ~correct & (mp > cut)
    out = {"centers": "scored test split, in-sample, true labels", "auc": "direction-free max(a, 1 - a)",
           "n": int(len(y)), "n_wrong": int((~correct).sum()), "n_cw": int(cw.sum())}
    cls = np.unique(y)
    if len(cls) < 2:
        out["note"] = "fewer than 2 classes in the test split"
        return out
    Ct = np.stack([X[y == c].mean(0) for c in cls])
    scores = {"margin": top2_margin(center_dists(X, Ct))[0],             # sample_and_scale.py:55-58
              "energy": (X ** 2).mean(1)}                                  # sample_and_scale.py:60
    if X.shape[1] % groups == 0:                                           # sample_and_scale.py:54
        gs = X.shape[1] // groups
        parts = []
        for g in range(groups):
            sl = slice(g * gs, (g + 1) * gs)
            dg = center_dists(X[:, sl], Ct[:, sl]).min(1)
            parts.append(dg / (np.median(dg) + 1e-9))
        scores["cluster_subnet"] = np.mean(parts, axis=0)
    enough = cw.sum() >= min_n and correct.sum() >= min_n
    for s, v in scores.items():
        if not enough:
            out[f"raw_auc_{s}"] = out[f"dir_auc_{s}"] = None
            continue
        a = float(delong(v[cw], v[correct])[0][0])
        out[f"raw_auc_{s}"], out[f"dir_auc_{s}"] = a, max(a, 1.0 - a)
    return out


@invariant("margin_typeb", needs=("ref", "labels", "test"), cost="expensive")
def margin_typeb(ctx, cfg):
    """Type-b geometry: top-2 center margin vs nearest-center distance, maxprob and energy; oriented
    per-sample AUCs with DeLong SEs; confidence-cut sweep; Upgraded-Mod legacy block. No ctx.rng draws."""
    cut = float(cfg.get("cut", 0.7))
    cuts = [float(c) for c in cfg.get("cuts", CUTS)]
    min_n = max(2, int(cfg.get("min_n", 10)))
    p = ctx.test_preds
    if not p or "argmax" not in p or "maxprob" not in p:
        raise ValueError("margin_typeb needs preds/test.npz with argmax and maxprob")
    X = np.asarray(ctx.test, dtype=np.float64)
    y = np.asarray(ctx.test_labels)
    am = np.asarray(p["argmax"])
    mp = np.asarray(p["maxprob"], dtype=np.float64)
    if not (len(X) == len(y) == len(am) == len(mp)):
        raise ValueError(f"margin_typeb: test rows {len(X)}, labels {len(y)}, preds {len(am)}/{len(mp)} differ")
    correct = am == y
    wrong = ~correct
    typeb = wrong & (mp > cut)
    n_wrong, n_typeb = int(wrong.sum()), int(typeb.sum())
    out = {
        "definition": {"typeb": f"argmax != label and maxprob > {cut}", "negatives": "all correct",
                       "centers": "ref split, true labels", "space": "full pooled layer, Euclidean",
                       "margin": "d2 - d1", "auc": "oriented (errors positive: -margin, +dist, -maxprob, -energy)"},
        "cut": cut, "min_n": min_n, "n_test": int(len(y)), "acc": float(correct.mean()) if len(y) else None,
        "n_correct": int(correct.sum()), "n_wrong": n_wrong, "n_typeb": n_typeb,
        "typeb_frac_of_wrong": n_typeb / n_wrong if n_wrong else None,
    }
    C, r, counts, _ = class_stats(ctx.ref, ctx.ref_labels, ctx.n_classes)
    present = counts > 0
    if present.sum() < 2:
        out["note"] = "fewer than 2 classes in the reference: no top-2 margin"
        return out
    C, rad = C[present], float(r[present].mean())
    S = _scores(X, C, mp)
    mg = S["margin"]
    groups = (("correct", correct), ("wrong", wrong), ("typeb", typeb))
    out["within_radius_mean"] = rad
    out["spearman_margin_maxprob"] = _spearman(mg, mp)
    out["spearman_margin_dist"] = _spearman(mg, S["dist"])
    out["margin_quantiles"] = {g: (np.quantile(mg[m], QS).tolist() if m.any() else None) for g, m in groups}
    for g, m in groups:
        med = _med(mg[m])
        out[f"median_margin_{g}"] = med
        out[f"median_margin_{g}_rad"] = None if med is None else med / (rad + 1e-12)
    mc = out["median_margin_correct"]
    for g in ("typeb", "wrong"):
        v = out[f"median_margin_{g}"]
        out[f"median_margin_ratio_{g}"] = v / mc if (v is not None and mc) else None
    out.update(_aucs(S, typeb, correct, min_n, "typeb"))
    out.update(_aucs(S, wrong, correct, min_n, "wrong"))
    out.update(_aucs(S, typeb, correct & (mp > cut), min_n, "confmatched"))
    sweep = []
    for c in cuts:
        tb = wrong & (mp > c)
        row = {"cut": c, "n_typeb": int(tb.sum()), "median_margin_typeb": _med(mg[tb])}
        for tag, neg in (("typeb", correct), ("confmatched", correct & (mp > c))):
            row.update({k: v for k, v in _aucs(S, tb, neg, min_n, tag).items()
                        if k.startswith(("auc_", "n_neg_"))})
        sweep.append(row)
    out["sweep"] = sweep
    out["legacy"] = _legacy(X, y, am, mp, cut, min_n)
    return out
