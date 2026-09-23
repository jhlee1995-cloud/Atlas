"""
flow.py -- how the map at layer l relates to the map at layer l+1 (cross-layer).

  layer_cka    : linear CKA between consecutive layers and between each layer and penult,
                 on the same clean test inputs. The largest consecutive drop marks where the
                 representation is re-organized most.
  commit_layer : first layer whose class-probe accuracy reaches tau * (best over layers),
                 and the same for every other factor ("where does factor F become readable").
                 Reads linear_probes results; falls back to a quick nearest-center accuracy
                 if probes were not run.

Cross-layer invariants receive every LayerContext plus the per-layer results dict.
"""
import numpy as np

from ..registry import cross_layer
from ._util import linear_cka, subsample


@cross_layer("layer_cka", needs=("test",), cost="cheap")
def layer_cka(ctxs, per_layer, cfg):
    """Linear CKA between consecutive layers and vs the last layer, same inputs."""
    layers = list(ctxs)
    n = int(cfg.get("n", 2000))
    first = ctxs[layers[0]]
    if first.test is None:
        return {"error": "no test split"}
    rng = np.random.default_rng(0)
    idx = rng.choice(len(first.test), size=min(n, len(first.test)), replace=False)
    feats = {l: ctxs[l].test[idx] for l in layers}
    consec = {}
    for a, b in zip(layers[:-1], layers[1:]):
        consec[f"{a}->{b}"] = linear_cka(feats[a], feats[b])
    vs_last = {l: linear_cka(feats[l], feats[layers[-1]]) for l in layers}
    drops = {k: 1.0 - v for k, v in consec.items()}
    biggest = min(consec, key=consec.get) if consec else None
    return {
        "consecutive_cka": consec,
        "cka_vs_last": vs_last,
        "biggest_reorganization": biggest,
        "biggest_reorganization_drop": drops.get(biggest) if biggest else None,
    }


@cross_layer("commit_layer", needs=("test",), cost="cheap")
def commit_layer(ctxs, per_layer, cfg):
    """First layer where each factor's probe score reaches tau * best-over-layers."""
    tau = float(cfg.get("tau", 0.9))
    layers = list(ctxs)
    have = all("linear_probes" in per_layer.get(l, {}) for l in layers)
    out = {"tau": tau, "per_factor": {}}
    if not have:
        # fallback: nearest-center accuracy profile from class_centers
        accs = []
        for l in layers:
            cc = per_layer.get(l, {}).get("class_centers", {})
            accs.append(cc.get("nearest_center_acc_test", np.nan))
        accs = np.array(accs, dtype=float)
        if np.all(np.isnan(accs)):
            return {"error": "neither linear_probes nor class_centers available"}
        best = np.nanmax(accs)
        first = next(i for i, a in enumerate(accs) if a >= tau * best)
        out["per_factor"]["class(nearest_center)"] = {"commit_layer": layers[first], "profile": accs.tolist()}
        return out
    factor_names = list(per_layer[layers[0]]["linear_probes"]["factors"])
    for f in factor_names:
        prof = []
        for l in layers:
            r = per_layer[l]["linear_probes"]["factors"].get(f, {})
            v = r.get("excess") if r.get("kind") == "categorical" else r.get("score")
            prof.append(np.nan if v is None else v)
        prof = np.array(prof, dtype=float)
        if np.all(np.isnan(prof)) or np.nanmax(prof) <= 0:
            out["per_factor"][f] = {"commit_layer": None, "profile": prof.tolist()}
            continue
        best = np.nanmax(prof)
        first = next(i for i, a in enumerate(prof) if np.isfinite(a) and a >= tau * best)
        peak = int(np.nanargmax(prof))
        out["per_factor"][f] = {
            "commit_layer": layers[first],
            "peak_layer": layers[peak],
            "best": float(best),
            "at_last": None if np.isnan(prof[-1]) else float(prof[-1]),
            "washout": None if np.isnan(prof[-1]) else float(best - prof[-1]),
            "profile": prof.tolist(),
        }
    return out
