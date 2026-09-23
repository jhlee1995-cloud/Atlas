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

  B1 (docs/plans/B1_VIT_MARGIN.md; pre-registration experiments/queue/margin_b1_vitb16.yaml). Two opt-in
  config keys add NEW keys only; without them the function runs the A3 code path unchanged:
    b1: true          paired DeLong tests margin - maxprob, margin - logitgap (when preds carry logit_gap),
                      margin_norm - dist, for type-b / wrong / confidence-matched and at every sweep cut (with
                      margin - dist); margin_norm = (d2 - d1) / |c(1) - c(2)| (position across the top-2 ridge);
                      confidence-stratified AUCs; nearest-center vs model agreement; maxprob saturation; the
                      confidence-matched median ratio; class-center geometry of the reference split; when preds
                      carry real_ok (ImageNet ReaL labels, E11), per sweep cut the type-b keys restricted to the
                      type-b samples that ReaL also calls wrong (*_realwrong).
    legacy_imagenet   Upgraded-Mod session_experiments/imagenet_extract.py:66-83 on the scored split (gate G).
  center_dists keeps the A3 broadcast arithmetic up to GRAM_MIN elements (every CIFAR dump) and switches to a
  chunked float64 Gram form above it (ImageNet: 25,000 x 1,000 x 768 would be a 154 GB tensor).

Upgraded-Mod@d9683cd: session_experiments/sample_and_scale.py:21-24, 44-61 (0.895 / 0.763 / 0.757);
extract/axis_registry.py:33-57 (subnet CLUSTER_DISTANCE); session_experiments/imagenet_extract.py:21-24,
66-83 (ImageNet 0.800 / 0.636 / valley separation 1.14).
"""
import numpy as np
from scipy.stats import norm, rankdata, spearmanr

from ..registry import invariant
from ._util import class_stats

CUTS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
SIGN = {"margin": -1.0, "dist": 1.0, "maxprob": -1.0, "energy": -1.0}   # sign so that + = more error-like
QS = (0.1, 0.25, 0.5, 0.75, 0.9)
LEGACY_GROUPS = 4                                                         # axis_registry.py:19 N_GROUPS
GRAM_MIN = 200_000_000     # N*K*D above which center_dists uses the Gram form (CIFAR max: 5000*10*64 = 3.2e6)
GRAM_ROWS = 4096
SIGN_B1 = {**SIGN, "logitgap": -1.0, "margin_norm": -1.0}   # a small logit gap / normalized margin is error-like


def center_dists(X, C, gram_min=None):
    """(N, K) Euclidean distances from the rows of X to the K rows of C. Up to gram_min (default GRAM_MIN,
    read at call time) elements N*K*D: the exact broadcast form (the A3 arithmetic). Above it: float64
    sqrt(max(|x|^2 + |c|^2 - 2 x.c, 0)) in blocks of GRAM_ROWS rows, never the (N, K, D) tensor."""
    X = np.asarray(X, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    if X.shape[0] * C.shape[0] * X.shape[1] <= (GRAM_MIN if gram_min is None else gram_min):
        return np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
    c2 = np.einsum("kd,kd->k", C, C)
    out = np.empty((X.shape[0], C.shape[0]))
    for i in range(0, X.shape[0], GRAM_ROWS):
        x = X[i:i + GRAM_ROWS]
        d2 = np.einsum("nd,nd->n", x, x)[:, None] + c2[None, :] - 2.0 * (x @ C.T)
        out[i:i + GRAM_ROWS] = np.sqrt(np.maximum(d2, 0.0))
    return out


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


# ---- B1 (docs/plans/B1_VIT_MARGIN.md): new keys only ------------------------------------------------------------
def _pair(S, a, b, pos, neg, min_n, tag):
    """Paired DeLong test of AUC(a) - AUC(b) (both oriented by SIGN_B1), errors (pos) vs correct (neg), in its own
    two-score DeLong call, so the four-score call behind the A3 keys is untouched. Keys {a}_minus_{b}_{tag}[_se, _p]."""
    k = f"{a}_minus_{b}_{tag}"
    out = {k: None, f"{k}_se": None, f"{k}_p": None}
    if int(pos.sum()) < min_n or int(neg.sum()) < min_n:
        return out
    auc, cov = delong(np.stack([SIGN_B1[a] * S[a][pos], SIGN_B1[b] * S[b][pos]]),
                      np.stack([SIGN_B1[a] * S[a][neg], SIGN_B1[b] * S[b][neg]]))
    d = float(auc[0] - auc[1])
    se = float(np.sqrt(max(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1], 0.0)))
    out.update({k: d, f"{k}_se": se, f"{k}_p": float(2.0 * norm.sf(abs(d) / se)) if se > 0 else None})
    return out


def _b1_aucs(S, pos, neg, min_n, tag):
    """AUC + DeLong SE of the B1 scores (margin_norm; logitgap when the dump stores logit_gap) and the paired
    tests margin - maxprob, margin_norm - dist and, with logit_gap, margin - logitgap."""
    extra = [s for s in ("margin_norm", "logitgap") if s in S]
    out = {}
    if int(pos.sum()) >= min_n and int(neg.sum()) >= min_n:
        auc, cov = delong(np.stack([SIGN_B1[s] * S[s][pos] for s in extra]),
                          np.stack([SIGN_B1[s] * S[s][neg] for s in extra]))
        for i, s in enumerate(extra):
            out[f"auc_{s}_{tag}"] = float(auc[i])
            out[f"se_{s}_{tag}"] = float(np.sqrt(max(cov[i, i], 0.0)))
    else:
        for s in extra:
            out[f"auc_{s}_{tag}"] = out[f"se_{s}_{tag}"] = None
    pairs = [("margin", "maxprob"), ("margin_norm", "dist")] + ([("margin", "logitgap")] if "logitgap" in S else [])
    for a, b in pairs:
        out.update(_pair(S, a, b, pos, neg, min_n, tag))
    return out


def _strat(S, pos, neg, mp, bins, min_bin):
    """Confidence-stratified AUC (exploratory): equal-count maxprob bins over pos | neg; in every bin with >= min_bin
    errors and >= min_bin correct samples, the oriented AUC of each score; weighted mean over the used bins
    (weight = errors x correct). A score that is a monotone function of maxprob gets exactly auc_maxprob."""
    names = [s for s in ("margin", "dist", "maxprob", "margin_norm", "logitgap") if s in S]
    out = {"bins": int(bins), "min_bin": int(min_bin), "n_bins_used": 0}
    out.update({f"auc_{s}": None for s in names})
    pool = pos | neg
    if int(pool.sum()) < 2 * min_bin:
        return out
    edges = np.quantile(mp[pool], np.linspace(0.0, 1.0, bins + 1))
    b = np.clip(np.searchsorted(edges, mp, side="right") - 1, 0, bins - 1)
    tot, wsum = np.zeros(len(names)), 0.0
    for k in range(bins):
        p, n = pos & (b == k), neg & (b == k)
        m_, n_ = int(p.sum()), int(n.sum())
        if m_ < min_bin or n_ < min_bin:
            continue
        auc, _ = delong(np.stack([SIGN_B1[s] * S[s][p] for s in names]),
                        np.stack([SIGN_B1[s] * S[s][n] for s in names]))
        tot += (m_ * n_) * auc
        wsum += m_ * n_
        out["n_bins_used"] += 1
    if wsum:
        out.update({f"auc_{s}": float(tot[i] / wsum) for i, s in enumerate(names)})
    return out


def _b1(ctx, X, C, cls_ids, rad, S, preds, am, mp, correct, wrong, typeb, cut, sweep, min_n, cfg):
    """The B1 keys. S = the four A3 scores (unchanged); B = S + margin_norm (+ logitgap). Adds keys to the sweep
    rows in place (margin_minus_* and the B1 AUCs per cut); returns the new top-level keys."""
    D = center_dists(X, C)
    near = np.argpartition(D, 1, axis=1)[:, :2]                    # [:, 0] nearest, [:, 1] second-nearest center
    i1, i2 = near[:, 0], near[:, 1]
    spacing = np.linalg.norm(C[i1] - C[i2], axis=1)                # |c(1) - c(2)| >= d2 - d1 (triangle inequality)
    B = dict(S)
    B["margin_norm"] = S["margin"] / np.maximum(spacing, 1e-12)
    lg = preds.get("logit_gap")
    if lg is not None:
        lg = np.asarray(lg, dtype=np.float64)
        if len(lg) != len(X):
            raise ValueError(f"margin_typeb: logit_gap rows {len(lg)} != test rows {len(X)}")
        B["logitgap"] = lg
    ro = preds.get("real_ok")                                      # E11: 1 argmax in the ReaL set, 0 not, -1 empty set
    if ro is not None:
        ro = np.asarray(ro)
        if len(ro) != len(X):
            raise ValueError(f"margin_typeb: real_ok rows {len(ro)} != test rows {len(X)}")
    cm = correct & (mp > cut)
    mt, mc = _med(S["margin"][typeb]), _med(S["margin"][cm])
    d12 = D[np.arange(len(X)), i1] + D[np.arange(len(X)), i2]
    o = {"b1": {"logit_gap": lg is not None, "real_ok": ro is not None, "gram_min": GRAM_MIN,
                "strat_bins": int(cfg.get("strat_bins", 10)), "strat_min": int(cfg.get("strat_min", 5))},
         "nearest_center_agrees_with_model": float((cls_ids[i1] == am).mean()),
         "frac_maxprob_ge_0999": float((mp >= 0.999).mean()),
         "frac_maxprob_unique": float(len(np.unique(mp)) / len(mp)),
         "d1_plus_d2_cv": float(d12.std() / (d12.mean() + 1e-12)),
         "median_center_spacing_top2": float(np.median(spacing)),
         "median_margin_correct_confmatched": mc,
         "median_margin_ratio_typeb_confmatched": mt / mc if (mt is not None and mc) else None}
    if lg is not None:
        o["spearman_margin_logitgap"] = _spearman(S["margin"], lg)
        o["spearman_maxprob_logitgap"] = _spearman(mp, lg)
    for tag, pos, neg in (("typeb", typeb, correct), ("wrong", wrong, correct), ("confmatched", typeb, cm)):
        o.update(_b1_aucs(B, pos, neg, min_n, tag))
    bins, mb = o["b1"]["strat_bins"], o["b1"]["strat_min"]
    o["strat_confmatched"] = _strat(B, typeb, cm, mp, bins, mb)
    for row in sweep:
        c = float(row["cut"])
        tb, cmc = wrong & (mp > c), correct & (mp > c)
        for tag, neg in (("typeb", correct), ("confmatched", cmc)):
            row.update({k: v for k, v in _aucs(S, tb, neg, min_n, tag).items() if k.startswith("margin_minus_")})
            row.update(_b1_aucs(B, tb, neg, min_n, tag))
        row["median_margin_correct_confmatched"] = _med(S["margin"][cmc])
        row["strat_confmatched"] = _strat(B, tb, cmc, mp, bins, mb)
        if ro is not None:                                         # E11 / H9: type-b that ReaL also calls wrong
            rw = tb & (ro == 0)
            row.update({f"{k}_realwrong": v for k, v in _aucs(S, rw, correct, min_n, "typeb").items()
                        if k.startswith(("auc_", "margin_minus_", "n_pos_"))})
            row["median_margin_typeb_realwrong"] = _med(S["margin"][rw])
            row["n_typeb_real_ok"] = int((tb & (ro == 1)).sum())
    # reference-split center geometry (E3). sep_ratio_ref = class_centers.sep_ratio (landmarks.py:25-30);
    # sep_legacy_ref = the imagenet_extract.py:69-72 formula (mean pairwise center distance / mean per-class mean
    # within-distance), here with out-of-sample reference centers.
    Dc = center_dists(C, C)
    np.fill_diagonal(Dc, np.inf)
    ref, ry = np.asarray(ctx.ref, dtype=np.float64), np.asarray(ctx.ref_labels)
    within = np.array([np.linalg.norm(ref[ry == c] - C[j], axis=1).mean() for j, c in enumerate(cls_ids)])
    o["centers_geometry"] = {"n_centers": int(len(C)),
                             "sep_ratio_ref": float((Dc.min(1) / (rad + 1e-12)).mean()),
                             "sep_legacy_ref": float(Dc[np.isfinite(Dc)].mean() / (within.mean() + 1e-9)),
                             "nearest_other_center_mean": float(Dc.min(1).mean()),
                             "within_mean_dist_mean": float(within.mean())}
    return o


def _legacy_imagenet(X, y, am, mp, min_n, cfg):
    """Upgraded-Mod session_experiments/imagenet_extract.py:66-83 on the scored split (B1 gate G): centers =
    in-sample class means of the classes with >= min_class_n samples (:68); valley separation = mean pairwise
    center distance / mean per-class mean within-distance (:69-72); conf-wrong = wrong and maxprob > cut (0.5,
    :75); "cluster" = raw full-space nearest-center distance (:77); top-2 margin (:78-79); "energy" = mean
    squared activation (:83); direction-free AUC max(a, 1 - a) with conf-wrong positive (:21-24). The legacy
    scored conf-wrong against an unseeded random subsample of correct samples of the same size (:80); here the
    AUC against all correct samples (deterministic, `_full`) and `draws` seeded subsamples (`_sub_*`)."""
    cut = float(cfg.get("cut", 0.5))
    mcn = int(cfg.get("min_class_n", 3))
    draws = int(cfg.get("draws", 200))
    seed = int(cfg.get("seed", 0))
    correct = am == y
    cw = ~correct & (mp > cut)
    out = {"cut": cut, "min_class_n": mcn, "draws": draws, "seed": seed, "n": int(len(y)),
           "acc": float(correct.mean()) if len(y) else None, "n_correct": int(correct.sum()),
           "n_wrong": int((~correct).sum()), "n_cw": int(cw.sum())}
    labs, cnt = np.unique(y, return_counts=True)
    cls = labs[cnt >= mcn]
    out["n_classes_centered"] = int(len(cls))
    if len(cls) < 2:
        out["note"] = f"fewer than 2 classes with >= {mcn} samples"
        return out
    Ct = np.stack([X[y == c].mean(0) for c in cls])
    within = np.array([np.linalg.norm(X[y == c] - Ct[i], axis=1).mean() for i, c in enumerate(cls)])
    Dcc = center_dists(Ct, Ct)
    out["valley_sep_legacy"] = float(Dcc[~np.eye(len(cls), dtype=bool)].mean() / (within.mean() + 1e-9))
    mg, dist = top2_margin(center_dists(X, Ct))
    scores = {"margin": mg, "cluster": dist, "energy": (X ** 2).mean(1)}
    ci = np.flatnonzero(correct)
    m = int(cw.sum())
    enough = m >= min_n and len(ci) >= min_n
    rng = np.random.default_rng(seed)
    subs = [rng.choice(ci, size=min(m, len(ci)), replace=False) for _ in range(draws)] if enough else []
    for s, v in scores.items():
        keys = [f"raw_auc_{s}_full", f"dir_auc_{s}_full"] + [f"dir_auc_{s}_sub_{q}"
                                                             for q in ("mean", "sd", "q025", "q975")]
        out.update({k: None for k in keys})
        if not enough:
            continue
        a = float(delong(v[cw], v[correct])[0][0])
        out[f"raw_auc_{s}_full"], out[f"dir_auc_{s}_full"] = a, max(a, 1.0 - a)
        if subs:
            sub = np.array([delong(v[cw], v[idx])[0][0] for idx in subs])
            d = np.maximum(sub, 1.0 - sub)
            out[f"dir_auc_{s}_sub_mean"] = float(d.mean())
            out[f"dir_auc_{s}_sub_sd"] = float(d.std(ddof=1)) if len(d) > 1 else None
            lo, hi = np.quantile(d, [0.025, 0.975])
            out[f"dir_auc_{s}_sub_q025"], out[f"dir_auc_{s}_sub_q975"] = float(lo), float(hi)
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
    if cfg.get("b1"):                                              # B1: new keys only (docs/plans/B1_VIT_MARGIN.md)
        out.update(_b1(ctx, X, C, np.flatnonzero(present), r[present], S, p, am, mp, correct, wrong, typeb,
                       cut, sweep, min_n, cfg))
    li = cfg.get("legacy_imagenet")
    if li and ctx.layer in li.get("layers", [ctx.layer]):          # default: every layer; B1 manifests: penult
        out["legacy_imagenet"] = _legacy_imagenet(X, y, am, mp, min_n, li)
    return out
