"""
compare.py -- deformation between two atlases (A = reference, B = current).

Three uses share this one tool:
  seed stability   A = seed0 atlas, B = seed1 atlas          (different weights, same task)
  scale transfer   A = resnet20,    B = resnet56              (layers matched by name/position)
  immune / TTA     A = before,      B = after adaptation      (same model, --same-space)

Per common layer:
  scalars        deltas of ID / PR / dim95 / sep_ratio / NC1 / ETF dev / hub skew
  landmarks      orthogonal-Procrustes disparity between class-center sets (rotation-invariant)
                 + raw center displacement in within-radius units when --same-space
  adjacency      Spearman of the pairwise separation matrices; Kendall tau of merge heights
  decodability   per-factor score deltas and profile-shape agreement
  panel          linear CKA between the two models' panel activations (same inputs)
                 relative-representation agreement: each panel point expressed as cosines to
                 the K class centers in its own model (Moschella et al. 2023); rows compared
                 across models. This is the cheapest test of the "embeddings mapped inside
                 the topology" idea: if rel-rep agreement is high across seeds, landmark
                 coordinates already travel between models.

Usage:
  python -m atlas.compare --a results/atlas_v0_resnet20_cifar10 --b results/atlas_v0_resnet20_seed1
  python -m atlas.compare --a <before> --b <after> --same-space
"""
import argparse
import json
import os

import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.stats import kendalltau, spearmanr

from .context import AtlasDump, to_jsonable
from .invariants._util import linear_cka

SCALARS = [
    ("twonn_id", "id"), ("pca_spectrum", "participation_ratio"), ("pca_spectrum", "dim95"),
    ("class_centers", "sep_ratio"), ("neural_collapse", "nc1"), ("neural_collapse", "etf_deviation"),
    ("hubness", "k_occurrence_skew"), ("class_centers", "nearest_center_acc_test"),
]


def _g(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _load(res_dir):
    atlas = json.load(open(os.path.join(res_dir, "atlas.json")))
    dump_dir = os.path.join(res_dir, "dump")
    dump = AtlasDump(dump_dir) if os.path.exists(os.path.join(dump_dir, "meta.json")) else None
    return atlas, dump


def match_layers(la, lb):
    """Common names first; otherwise match by relative depth position (scale transfer)."""
    common = [l for l in la if l in lb]
    if len(common) >= min(len(la), len(lb)) * 0.6:
        return [(l, l) for l in common]
    pairs = []
    for i, l in enumerate(la):
        j = int(round(i * (len(lb) - 1) / max(1, len(la) - 1)))
        pairs.append((l, lb[j]))
    return pairs


def procrustes_disparity(CA, CB):
    """Centered, unit-Frobenius, rotation-aligned residual in [0, 1]."""
    A = CA - CA.mean(0)
    B = CB - CB.mean(0)
    A = A / (np.linalg.norm(A) + 1e-12)
    B = B / (np.linalg.norm(B) + 1e-12)
    if A.shape[1] != B.shape[1]:
        return None
    R, _ = orthogonal_procrustes(A, B)
    return float(np.linalg.norm(A @ R - B) ** 2)


def upper(M):
    M = np.asarray(M, dtype=float)
    iu = np.triu_indices(len(M), 1)
    v = M[iu]
    return v


def rel_rep(X, C):
    """Cosine of each row of X to each class center (N, K)."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    return Xn @ Cn.T


def compare(res_a, res_b, same_space=False):
    A, dA = _load(res_a)
    B, dB = _load(res_b)
    pairs = match_layers(A["layers"], B["layers"])
    out = {"a": res_a, "b": res_b, "same_space": same_space,
           "source_a": A["source"], "source_b": B["source"], "layers": pairs, "per_layer": {}}
    for la, lb in pairs:
        pa, pb = A["per_layer"][la], B["per_layer"][lb]
        r = {"layer_b": lb, "scalars": {}}
        for inv, key in SCALARS:
            va, vb = _g(pa, inv, key), _g(pb, inv, key)
            if va is not None and vb is not None:
                r["scalars"][f"{inv}.{key}"] = {"a": va, "b": vb, "delta": vb - va,
                                                "rel": (vb - va) / (abs(va) + 1e-9)}
        # landmarks
        CA, CB = _g(pa, "class_centers", "centers"), _g(pb, "class_centers", "centers")
        if CA is not None and CB is not None:
            CA, CB = np.array(CA), np.array(CB)
            r["landmark_procrustes_disparity"] = procrustes_disparity(CA, CB)
            if same_space and CA.shape == CB.shape:
                wr = float(np.mean(_g(pa, "class_centers", "radius") or [1.0]))
                disp = np.linalg.norm(CB - CA, axis=1) / (wr + 1e-12)
                r["center_displacement_within_units"] = {"mean": float(disp.mean()), "max": float(disp.max()),
                                                         "per_class": disp.tolist()}
        # adjacency
        SA, SB = _g(pa, "class_adjacency", "sep_matrix"), _g(pb, "class_adjacency", "sep_matrix")
        if SA is not None and SB is not None:
            ua, ub = upper(SA), upper(SB)
            ok = (ua > 0) & (ub > 0)
            if ok.sum() > 3:
                r["adjacency_spearman"] = float(spearmanr(ua[ok], ub[ok]).correlation)
        HA, HB = _g(pa, "merge_order", "cophenetic_matrix"), _g(pb, "merge_order", "cophenetic_matrix")
        if HA is not None and HB is not None:
            ua, ub = upper(HA), upper(HB)
            ok = (ua >= 0) & (ub >= 0)
            if ok.sum() > 3:
                r["merge_height_kendall"] = float(kendalltau(ua[ok], ub[ok]).correlation)
            fa, fb = _g(pa, "merge_order", "first_merge"), _g(pb, "merge_order", "first_merge")
            r["first_merge_same"] = (fa == fb)
        # decodability
        FA, FB = _g(pa, "linear_probes", "factors"), _g(pb, "linear_probes", "factors")
        if FA and FB:
            dd, xa, xb = {}, [], []
            for f in FA:
                if f in FB:
                    ka = FA[f].get("excess") if FA[f].get("kind") == "categorical" else FA[f].get("score")
                    kb = FB[f].get("excess") if FB[f].get("kind") == "categorical" else FB[f].get("score")
                    if ka is not None and kb is not None:
                        dd[f] = {"a": ka, "b": kb, "delta": kb - ka}
                        xa.append(ka); xb.append(kb)
            r["decodability"] = dd
            if len(xa) > 3:
                r["decodability_profile_spearman"] = float(spearmanr(xa, xb).correlation)
                r["decodability_mean_abs_delta"] = float(np.mean(np.abs(np.array(xb) - np.array(xa))))
        # panel-based (needs both dumps)
        if dA is not None and dB is not None and dA.has(la, "panel") and dB.has(lb, "panel"):
            XA, XB = dA.acts(la, "panel"), dB.acts(lb, "panel")
            n = min(len(XA), len(XB))
            r["panel_cka"] = linear_cka(XA[:n], XB[:n])
            if CA is not None and CB is not None:
                RA, RB = rel_rep(XA[:n], CA), rel_rep(XB[:n], CB)
                corr = [np.corrcoef(RA[i], RB[i])[0, 1] for i in range(n)]
                r["relrep_row_corr_mean"] = float(np.nanmean(corr))
                r["relrep_argmax_agree"] = float((RA.argmax(1) == RB.argmax(1)).mean())
            if same_space:
                r["panel_mean_shift_within_units"] = float(
                    np.linalg.norm(XB[:n] - XA[:n], axis=1).mean() /
                    (float(np.mean(_g(pa, "class_centers", "radius") or [1.0])) + 1e-12))
        out["per_layer"][la] = r
    # cross-layer: commit layers
    ca, cb = _g(A, "cross_layer", "commit_layer", "per_factor"), _g(B, "cross_layer", "commit_layer", "per_factor")
    if ca and cb:
        out["commit_layer"] = {f: {"a": ca[f].get("commit_layer"), "b": cb.get(f, {}).get("commit_layer")}
                               for f in ca if f in cb}
    return to_jsonable(out)


def write_md(res, path):
    L = [f"# DEFORMATION  A=`{res['a']}`  B=`{res['b']}`  same_space={res['same_space']}",
         f"sources: {res['source_a']} vs {res['source_b']}", ""]
    if "synthetic" in (res["source_a"], res["source_b"]):
        L.append("> synthetic involved: pipeline check only\n")
    L += ["| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda v: "·" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
    for la, r in res["per_layer"].items():
        L.append("| " + " | ".join([la, r["layer_b"], f(r.get("landmark_procrustes_disparity")),
                                    f(r.get("adjacency_spearman")), f(r.get("merge_height_kendall")),
                                    f(r.get("first_merge_same")), f(r.get("decodability_profile_spearman")),
                                    f(r.get("decodability_mean_abs_delta")), f(r.get("panel_cka")),
                                    f(r.get("relrep_row_corr_mean")), f(r.get("relrep_argmax_agree"))]) + " |")
    L += ["", "## scalar deltas (B − A)", "| layer | " + " | ".join(k for k, _ in SCALARS) + " |",
          "|---|" + "---|" * len(SCALARS)]
    for la, r in res["per_layer"].items():
        row = []
        for inv, key in SCALARS:
            s = r["scalars"].get(f"{inv}.{key}")
            row.append("·" if s is None else f"{s['delta']:+.3f}")
        L.append(f"| {la} | " + " | ".join(row) + " |")
    if res.get("commit_layer"):
        L += ["", "## commit layer A → B", "| factor | A | B |", "|---|---|---|"]
        for fct, v in res["commit_layer"].items():
            L.append(f"| {fct} | {v['a']} | {v['b']} |")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


def _has_panel(res):
    return any("panel_cka" in r for r in res.get("per_layer", {}).values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="reference result dir (atlas.json [+ dump/])")
    ap.add_argument("--b", required=True, help="current result dir")
    ap.add_argument("--same-space", action="store_true", help="same model/coordinates (TTA before/after)")
    ap.add_argument("--out", help="output dir (default: <b>/compare_vs_<a-name>)")
    args = ap.parse_args()
    res = compare(args.a, args.b, args.same_space)
    out = args.out or os.path.join(args.b, f"compare_vs_{os.path.basename(os.path.normpath(args.a))}")
    tgt = os.path.join(out, "deformation.json")
    if not _has_panel(res):
        print("[compare] WARNING: a dump/ is missing; panel CKA and relrep metrics were not computed")
        if os.path.exists(tgt) and _has_panel(json.load(open(tgt))):
            raise SystemExit(f"[compare] refusing to overwrite {tgt}: it has panel metrics and this run "
                             f"has no dump/ (pass --out <new dir>)")
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(tgt, "w"), indent=1)
    write_md(res, os.path.join(out, "DEFORMATION.md"))
    print(f"[compare] wrote {out}/deformation.json and DEFORMATION.md")


if __name__ == "__main__":
    main()
