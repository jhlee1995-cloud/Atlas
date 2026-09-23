"""
critic.py -- the executable half of the Critic role. Given N result dirs (seeds, scales,
or before/after), decide which map entries replicate and flag methodology violations.

Checks (each is a function; add more in CHECKS):
  synthetic_refusal   any atlas with source=synthetic -> nothing can PASS; report only
  scalar_stability    per layer, per whitelisted scalar: relative spread across runs <= tol
  adjacency_stability pairwise Spearman of separation matrices >= tol, merge-height tau >= tol
  decodability_stability per factor: profile-shape Spearman >= tol and mean |delta| <= tol
  commit_agreement    commit layer identical (or within +/-1 position) across runs
  holdout_hygiene     manifest_used.yaml declares discovery/confirmation splits (warn if not)
  probe_hygiene       linear_probes used cross-validation with >= 3 folds (warn if not)
  norm_consistency    every run was extracted under the same input normalization
  alias_layers        consecutive layers identical in every run (penult == GAP(last block)) are counted once
  id_profile_stability TwoNN ID profile Spearman >= tol and peak position within commit_layer_slack
  panel_agreement     top-layer cka_test from the pod-computed compare_vs_*/deformation.json (+ INFO metrics)

Scalars listed in tolerances scalar_abs_tol ({"<inv>.<key>": max spread}, e.g. the A3 margin AUCs) are judged by
absolute spread only; every other scalar keeps the relative-spread / absolute-floor rule.
--align position (scale transfer, docs/plans/STAGE2.md): every run whose layer list differs from the first run's
is relabelled onto the first run's names by index, exactly as compare.match_layers pairs two equal-length lists
that share < 60% of their names (resnet56 block_stride 5 vs resnet20: 11 taps each). Unequal tap counts, name
pairs and pairs across stages are refused. The default (name) leaves every run as it is.

Output: CRITIC.md + critic.json with PASS / FAIL / WARN per item and an overall verdict.
The verdict is advisory: the agent reads it, a human promotes entries to ATLAS_STATUS.md.

Usage:
  python -m atlas.critic --results results/atlas_v0_resnet20_cifar10 results/atlas_v0_resnet20_seed1
  python -m atlas.critic --results ... --tol tolerances.yaml
  python -m atlas.critic --results <resnet20 dir> <resnet56 dir> --tol ... --align position --out ...
"""
import argparse
import copy
import itertools
import json
import os

import numpy as np
import yaml
from scipy.stats import kendalltau, spearmanr

from .compare import SCALARS, _g, match_layers, upper

DEFAULT_TOL = {
    "scalar_rel_spread": 0.15,       # (max-min)/|mean| across runs
    "scalar_abs_floor": 0.05,        # spreads below this pass regardless of rel (near-zero means)
    "adjacency_spearman": 0.70,
    "merge_kendall": 0.60,
    "decod_profile_spearman": 0.70,
    "decod_mean_abs_delta": 0.10,
    "commit_layer_slack": 1,         # positions
    "min_runs": 2,
    "id_profile_spearman": 0.90,     # TwoNN ID profile shape across runs (+ peak within commit_layer_slack)
    "cka_test_min": 0.80,            # top-layer linear CKA on test[:2000] (from pod-computed deformation.json)
    "scalar_abs_tol": {},            # {"<inv>.<key>": max abs spread}: replaces the rel/abs-floor rule for listed scalars
}


def _load_all(dirs):
    runs = []
    for d in dirs:
        a = json.load(open(os.path.join(d, "atlas.json")))
        mpath = os.path.join(d, "manifest_used.yaml")
        m = yaml.safe_load(open(mpath)) if os.path.exists(mpath) else {}
        runs.append({"dir": d, "atlas": a, "manifest": m})
    return runs


def align_runs(runs, mode="name"):
    """--align position (scale transfer): relabel every run whose layer list differs from run 0's onto
    run 0's names, pairing by index exactly as compare.match_layers does for two equal-length lists that
    share < 60% of their names (resnet56 block_stride 5 vs resnet20: 11 taps each). Refuses unequal tap
    counts, name-based pairs and pairs across stages. Mutates runs (the atlas is deep-copied first);
    returns {run dir name: [[run-0 layer, own layer], ...]} for every relabelled run."""
    if mode != "position":
        return {}
    ref = list(runs[0]["atlas"]["layers"])
    stage = lambda l: l.split(".")[0]
    out = {}
    for r in runs[1:]:
        lb = list(r["atlas"]["layers"])
        if lb == ref:
            continue
        pairs = [tuple(p) for p in match_layers(ref, lb)]
        if len(lb) != len(ref) or pairs != list(zip(ref, lb)) or any(stage(a) != stage(b) for a, b in pairs):
            raise SystemExit(f"[critic] --align position: {r['dir']} does not map 1:1 by position and stage "
                             f"onto {runs[0]['dir']}: {pairs}")
        m = {b: a for a, b in pairs}
        a = copy.deepcopy(r["atlas"])
        a["layers"] = [m[b] for b in lb]
        a["per_layer"] = {m[b]: a["per_layer"][b] for b in lb}
        for v in (_g(a, "cross_layer", "commit_layer", "per_factor") or {}).values():
            if not isinstance(v, dict):
                continue
            for k in ("commit_layer", "peak_layer"):
                if v.get(k) in m:
                    v[k] = m[v[k]]
        r["atlas"] = a
        out[os.path.basename(os.path.normpath(r["dir"]))] = [list(p) for p in pairs]
    return out


# per-layer signature for alias detection; margin_typeb lets margin-only atlases (A3 rebuilds) collapse
# layer3.2 == penult too. Atlases without it get one more None, so their aliases are unchanged.
_SIG = [("twonn_id", "id"), ("pca_spectrum", "participation_ratio"), ("class_centers", "sep_ratio"),
        ("neural_collapse", "nc1"), ("hubness", "k_occurrence_skew"), ("margin_typeb", "median_margin_correct")]


def _sig(run, layer):
    vals = [_g(run["atlas"]["per_layer"][layer], inv, key) for inv, key in _SIG]
    return tuple(None if v is None else float(f"{v:.6g}") for v in vals)


def _aliases(runs):
    """Collapse exact aliases between consecutive common layers (under pooling 'gap', penult is the
    GAP of the last block's output). The LATER name is kept, so 'penult' survives."""
    sets = [list(r["atlas"]["layers"]) for r in runs]
    names = [l for l in sets[0] if all(l in s for s in sets[1:])]
    alias = {}
    for a, b in zip(names, names[1:]):
        sa = [_sig(r, a) for r in runs]
        if all(s == _sig(r, b) and any(v is not None for v in s) for r, s in zip(runs, sa)):
            alias[a] = b
    for a in list(alias):                     # resolve chains a -> b -> c
        while alias[a] in alias:
            alias[a] = alias[alias[a]]
    return [l for l in names if l not in alias], alias


def common_layers(runs):
    return _aliases(runs)[0]


# ---- checks ------------------------------------------------------------------
def synthetic_refusal(runs, tol):
    items = []
    for r in runs:
        if r["atlas"]["source"] != "real":
            items.append({"name": f"source:{os.path.basename(r['dir'])}", "status": "FAIL",
                          "detail": f"source={r['atlas']['source']}; synthetic evidence is never promoted"})
    return items or [{"name": "sources", "status": "PASS", "detail": "all runs real"}]


def scalar_stability(runs, tol):
    items = []
    abs_tol = tol.get("scalar_abs_tol") or {}
    for l in common_layers(runs):
        for inv, key in SCALARS:
            vals = [_g(r["atlas"]["per_layer"][l], inv, key) for r in runs]
            vals = [v for v in vals if v is not None]
            if len(vals) < tol["min_runs"]:
                continue
            vals = np.array(vals, dtype=float)
            spread = float(vals.max() - vals.min())
            rel = spread / (abs(vals.mean()) + 1e-9)
            at = abs_tol.get(f"{inv}.{key}")
            if at is None:                  # the Stage 0/1 rule, unchanged
                ok = (rel <= tol["scalar_rel_spread"]) or (spread <= tol["scalar_abs_floor"])
                detail = f"values={np.round(vals, 3).tolist()} rel_spread={rel:.3f}"
            else:                           # AUC-like scalars: the relative rule would pass a 0.13 spread at 0.9
                ok = spread <= float(at)
                detail = f"values={np.round(vals, 3).tolist()} abs_spread={spread:.3f} abs_tol={at}"
            items.append({"name": f"{l}/{inv}.{key}", "status": "PASS" if ok else "FAIL", "detail": detail})
    return items


def adjacency_stability(runs, tol):
    items = []
    for l in common_layers(runs):
        for (i, ra), (j, rb) in itertools.combinations(enumerate(runs), 2):
            SA, SB = (_g(r["atlas"]["per_layer"][l], "class_adjacency", "sep_matrix") for r in (ra, rb))
            if SA is not None and SB is not None:
                ua, ub = upper(SA), upper(SB)
                ok = (ua > 0) & (ub > 0)
                if ok.sum() > 3:
                    rho = float(spearmanr(ua[ok], ub[ok]).correlation)
                    items.append({"name": f"{l}/adjacency_spearman[{i},{j}]",
                                  "status": "PASS" if rho >= tol["adjacency_spearman"] else "FAIL",
                                  "detail": f"rho={rho:.3f}"})
            HA, HB = (_g(r["atlas"]["per_layer"][l], "merge_order", "cophenetic_matrix") for r in (ra, rb))
            if HA is not None and HB is not None:
                ua, ub = upper(HA), upper(HB)
                ok = (ua >= 0) & (ub >= 0)
                if ok.sum() > 3:
                    tau = float(kendalltau(ua[ok], ub[ok]).correlation)
                    items.append({"name": f"{l}/merge_kendall[{i},{j}]",
                                  "status": "PASS" if tau >= tol["merge_kendall"] else "FAIL",
                                  "detail": f"tau={tau:.3f}"})
    return items


def _profile(run, factor, layers):
    out = []
    for l in layers:
        r = _g(run["atlas"]["per_layer"][l], "linear_probes", "factors", factor) or {}
        v = r.get("excess") if r.get("kind") == "categorical" else r.get("score")
        out.append(np.nan if v is None else v)
    return np.array(out, dtype=float)


def decodability_stability(runs, tol):
    items = []
    layers = common_layers(runs)
    f0 = _g(runs[0]["atlas"]["per_layer"][layers[0]], "linear_probes", "factors") if layers else None
    if not f0:
        return [{"name": "decodability", "status": "WARN", "detail": "linear_probes not present"}]
    for f in f0:
        profs = [_profile(r, f, layers) for r in runs]
        for (i, pa), (j, pb) in itertools.combinations(enumerate(profs), 2):
            ok = np.isfinite(pa) & np.isfinite(pb)
            if ok.sum() < 3:
                continue
            rho = float(spearmanr(pa[ok], pb[ok]).correlation) if pa[ok].std() > 1e-9 and pb[ok].std() > 1e-9 else 1.0
            mad = float(np.mean(np.abs(pa[ok] - pb[ok])))
            good = (rho >= tol["decod_profile_spearman"] or np.isnan(rho)) and mad <= tol["decod_mean_abs_delta"]
            items.append({"name": f"decod/{f}[{i},{j}]", "status": "PASS" if good else "FAIL",
                          "detail": f"profile_spearman={rho:.2f} mean_abs_delta={mad:.3f}"})
    return items


def commit_agreement(runs, tol):
    items = []
    layers, alias = _aliases(runs)
    pos = {l: i for i, l in enumerate(layers)}
    cls = [_g(r["atlas"], "cross_layer", "commit_layer", "per_factor") for r in runs]
    if not all(cls):
        return [{"name": "commit_layer", "status": "WARN", "detail": "commit_layer missing in some run"}]
    for f in cls[0]:
        cl = [c.get(f, {}).get("commit_layer") for c in cls]
        cl = [alias.get(c, c) for c in cl]
        if any(c is None or c not in pos for c in cl):
            items.append({"name": f"commit/{f}", "status": "WARN", "detail": f"{cl}"})
            continue
        p = [pos[c] for c in cl]
        ok = (max(p) - min(p)) <= tol["commit_layer_slack"]
        items.append({"name": f"commit/{f}", "status": "PASS" if ok else "FAIL", "detail": f"{cl}"})
    return items


def holdout_hygiene(runs, tol):
    items = []
    for r in runs:
        h = (r["manifest"] or {}).get("holdout") or {}
        declared = any(h.get(k) for k in ("discovery_corruptions", "confirmation_corruptions",
                                           "discovery_seeds", "confirmation_seeds"))
        items.append({"name": f"holdout:{os.path.basename(r['dir'])}", "status": "PASS" if declared else "WARN",
                      "detail": "declared" if declared else "no discovery/confirmation split declared in manifest"})
    return items


def probe_hygiene(runs, tol):
    items = []
    for r in runs:
        lp = None
        for l in r["atlas"]["layers"]:
            lp = _g(r["atlas"]["per_layer"][l], "linear_probes")
            if lp:
                break
        if not lp:
            items.append({"name": f"probes:{os.path.basename(r['dir'])}", "status": "WARN", "detail": "no probes"})
        elif int(lp.get("cv_folds", 0)) < 3:
            items.append({"name": f"probes:{os.path.basename(r['dir'])}", "status": "FAIL",
                          "detail": f"cv_folds={lp.get('cv_folds')} (<3): in-sample probe scores are not evidence"})
        else:
            items.append({"name": f"probes:{os.path.basename(r['dir'])}", "status": "PASS",
                          "detail": f"cv_folds={lp['cv_folds']} n_train={lp.get('n_train')}"})
    return items


def alias_layers(runs, tol):
    alias = _aliases(runs)[1]
    return [{"name": f"alias:{a}", "status": "WARN", "detail": f"identical to {b} in every run; counted once as {b}"}
            for a, b in alias.items()] or [{"name": "aliases", "status": "PASS", "detail": "none"}]


def norm_consistency(runs, tol):
    norms = {os.path.basename(os.path.normpath(r["dir"])): (r["atlas"].get("meta") or {}).get("norm", "cifar_true")
             for r in runs}
    return [{"name": "input_norm", "status": "PASS" if len(set(norms.values())) == 1 else "FAIL",
             "detail": str(norms)}]


def id_profile_stability(runs, tol):
    layers, items = common_layers(runs), []
    profs = [np.array([_g(r["atlas"]["per_layer"][l], "twonn_id", "id") for l in layers], dtype=float) for r in runs]
    for (i, pa), (j, pb) in itertools.combinations(enumerate(profs), 2):
        ok = np.isfinite(pa) & np.isfinite(pb)
        if ok.sum() < 4:
            continue
        rho = float(spearmanr(pa[ok], pb[ok]).correlation)
        ia, ib = int(np.nanargmax(pa)), int(np.nanargmax(pb))
        good = rho >= tol["id_profile_spearman"] and abs(ia - ib) <= tol["commit_layer_slack"]
        items.append({"name": f"id_profile[{i},{j}]", "status": "PASS" if good else "FAIL",
                      "detail": f"spearman={rho:.3f} peak={layers[ia]}/{layers[ib]}"})
    return items


# Read from the pod-computed deformation.json (needs both dumps). cka_test is judged; the rest are
# INFO: the panel is 64 train images (agrees trivially) and relrep argmax largely restates accuracy.
PANEL_JUDGED = [("cka_test", "cka_test_min")]
PANEL_INFO = ["panel_cka", "relrep_row_corr_mean", "relrep_argmax_agree", "relrep_argmax_agree_test",
              "relrep_argmax_chance_test", "relrep_offmax_corr_test", "error_consistency_test",
              "cka_ood_c100", "relrep_argmax_agree_ood_c100", "relrep_argmax_chance_ood_c100",
              "relrep_offmax_corr_ood_c100", "landmark_procrustes_disparity"]


def panel_agreement(runs, tol):
    items = []
    for (i, ra), (j, rb) in itertools.combinations(enumerate(runs), 2):
        dfm = None
        for x, y in ((ra, rb), (rb, ra)):
            p = os.path.join(y["dir"], f"compare_vs_{os.path.basename(os.path.normpath(x['dir']))}", "deformation.json")
            if os.path.exists(p):
                dfm = json.load(open(p))
                break
        if dfm is None or not dfm.get("per_layer"):
            items.append({"name": f"panel[{i},{j}]", "status": "WARN", "detail": "no compare_vs_* deformation.json"})
            continue
        top = list(dfm["per_layer"])[-1]
        r = dfm["per_layer"][top]
        for k, tk in PANEL_JUDGED:
            if r.get(k) is None:
                items.append({"name": f"{top}/{k}[{i},{j}]", "status": "WARN", "detail": "not computed (dump missing?)"})
            else:
                items.append({"name": f"{top}/{k}[{i},{j}]", "status": "PASS" if r[k] >= tol[tk] else "FAIL",
                              "detail": f"{k}={r[k]:.3f} min={tol[tk]}"})
        info = {k: round(r[k], 3) for k in PANEL_INFO if isinstance(r.get(k), (int, float))}
        items.append({"name": f"{top}/agreement_info[{i},{j}]", "status": "INFO", "detail": json.dumps(info)})
    return items


CHECKS = [synthetic_refusal, norm_consistency, alias_layers, holdout_hygiene, probe_hygiene, scalar_stability,
          id_profile_stability, adjacency_stability, decodability_stability, commit_agreement, panel_agreement]


def run_critic(dirs, tol=None, align="name"):
    tol = {**DEFAULT_TOL, **(tol or {})}
    runs = _load_all(dirs)
    report = {"runs": dirs, "tolerances": tol, "checks": {}}
    if align != "name":                         # name (default): runs and report exactly as before
        report["align"] = align
        report["alignment"] = align_runs(runs, align)
    any_synth = any(r["atlas"]["source"] != "real" for r in runs)
    for chk in CHECKS:
        try:
            items = chk(runs, tol)
        except Exception as e:
            items = [{"name": chk.__name__, "status": "WARN", "detail": f"check crashed: {repr(e)[:200]}"}]
        if any_synth and chk is not synthetic_refusal:
            for it in items:
                if it["status"] == "PASS":
                    it["status"] = "INFO"
                    it["detail"] += " (synthetic run: not promotable)"
        report["checks"][chk.__name__] = items
    allitems = [it for its in report["checks"].values() for it in its]
    counts = {s: sum(1 for it in allitems if it["status"] == s) for s in ("PASS", "FAIL", "WARN", "INFO")}
    report["counts"] = counts
    report["verdict"] = ("PIPELINE_CHECK_ONLY" if any_synth else
                         "REPLICATES" if counts["FAIL"] == 0 else
                         "PARTIAL" if counts["PASS"] > counts["FAIL"] else "DOES_NOT_REPLICATE")
    return report


def write_md(report, path):
    L = [f"# CRITIC  verdict: **{report['verdict']}**", f"runs: {report['runs']}",
         f"counts: {report['counts']}", ""]
    if report.get("align"):
        L.insert(3, f"layer alignment: {report['align']} {json.dumps(report.get('alignment') or {})}")
    for name, items in report["checks"].items():
        L.append(f"## {name}")
        fails = [it for it in items if it["status"] == "FAIL"]
        L.append(f"{len(items)} items, {len(fails)} FAIL")
        L += [f"- **{it['status']}** `{it['name']}` {it['detail']}" for it in items if it["status"] != "PASS"]
        if len(items) - len([it for it in items if it["status"] != "PASS"]) > 0:
            L.append(f"- PASS: {len([it for it in items if it['status'] == 'PASS'])} items")
        L.append("")
    with open(path, "w") as f:
        f.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--tol", help="yaml with tolerance overrides")
    ap.add_argument("--out", help="output dir (default: first result dir)")
    ap.add_argument("--align", choices=["name", "position"], default="name",
                    help="position: relabel runs onto the first run's layer names by index (scale transfer)")
    args = ap.parse_args()
    tol = yaml.safe_load(open(args.tol)) if args.tol else None
    rep = run_critic(args.results, tol, args.align)
    out = args.out or args.results[0]
    os.makedirs(out, exist_ok=True)
    json.dump(rep, open(os.path.join(out, "critic.json"), "w"), indent=1)
    write_md(rep, os.path.join(out, "CRITIC.md"))
    print(f"[critic] {rep['verdict']}  {rep['counts']}  -> {out}/CRITIC.md")


if __name__ == "__main__":
    main()
