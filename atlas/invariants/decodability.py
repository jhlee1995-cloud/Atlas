"""
decodability.py -- which input factors are linearly readable at each layer.

  linear_probes : for every registered factor, a 5-fold cross-validated linear probe on
                  standardized activations. Categorical -> logistic accuracy (+ chance level);
                  continuous -> ridge R^2 (alpha chosen by inner CV).

The pool a factor is probed on follows FactorSpec.pool:
  clean   : clean test only                (class, coarse)
  corrupt : clean test + every corrupt set (corruption family/type, severity)
  mixed   : clean test + every corrupt set (pixel factors; the corruptions supply the range)

Output is the layers x factors table the row-6 question reads directly: if luminance /
highfreq / anisotropy are decodable at stage-2 but washed out at penult, the gap was a
tap-placement problem, not a missing-axis problem.

Labels are used here, offline, to build the map. The runtime gate stays label-free.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..registry import invariant, FACTORS, select


def _gather(ctx, spec, n_train, rng):
    """Concatenate (X, y) across the splits this factor is probed on."""
    splits = ["test"] if spec.pool == "clean" else ["test"] + sorted(ctx.corrupt)
    Xs, ys = [], []
    for s in splits:
        X = ctx.test if s == "test" else ctx.corrupt.get(s)
        if X is None:
            continue
        labels = ctx.test_labels if s == "test" else ctx.corrupt_labels[s]
        if spec.source == "meta":
            y = spec.fn(s, labels)
        else:
            y = ctx.factors.get(s, {}).get(spec.name)
        if y is None:
            continue
        y = np.asarray(y)
        m = min(len(X), len(y))
        Xs.append(X[:m])
        ys.append(y[:m])
    if not Xs:
        return None, None
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    if len(X) > n_train:
        idx = rng.choice(len(X), size=n_train, replace=False)
        X, y = X[idx], y[idx]
    return X, y


def probe_one(X, y, kind, folds=5, seed=0):
    """Returns (score, baseline). Categorical: accuracy vs majority-class rate.
    Continuous: R^2 vs 0."""
    if kind == "categorical":
        y = y.astype(int)
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            return float("nan"), float("nan")
        folds_eff = int(min(folds, counts.min()))
        if folds_eff < 2:
            return float("nan"), float("nan")
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=1.0, max_iter=300))
        cv = StratifiedKFold(folds_eff, shuffle=True, random_state=seed)
        sc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        return float(sc.mean()), float(counts.max() / counts.sum())
    y = y.astype(np.float64)
    if y.std() < 1e-9:
        return float("nan"), 0.0
    reg = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 8)))
    cv = KFold(folds, shuffle=True, random_state=seed)
    sc = cross_val_score(reg, X, y, cv=cv, scoring="r2")
    return float(sc.mean()), 0.0


@invariant("linear_probes", needs=("test",), cost="expensive")
def linear_probes(ctx, cfg):
    """CV linear probe score per factor at this layer (see module doc for pools)."""
    n_train = int(cfg.get("n_train", 4000))
    folds = int(cfg.get("cv_folds", 5))
    wanted = select(FACTORS, cfg.get("factors", "all"))
    out = {"n_train": n_train, "cv_folds": folds, "factors": {}}
    for name, spec in wanted.items():
        X, y = _gather(ctx, spec, n_train, ctx.rng)
        if X is None or len(X) < 50:
            out["factors"][name] = {"score": None, "baseline": None, "n": 0, "note": "no data for pool"}
            continue
        score, base = probe_one(X, y, spec.kind, folds=folds, seed=int(ctx.rng.integers(1 << 30)))
        out["factors"][name] = {
            "kind": spec.kind, "pool": spec.pool, "n": int(len(X)),
            "score": None if np.isnan(score) else score,
            "baseline": None if np.isnan(base) else base,
            "excess": None if np.isnan(score) else float(score - (base if np.isfinite(base) else 0.0)),
        }
    return out
