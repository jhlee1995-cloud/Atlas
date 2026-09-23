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

Output: CRITIC.md + critic.json with PASS / FAIL / WARN per item and an overall verdict.
The verdict is advisory: the agent reads it, a human promotes entries to ATLAS_STATUS.md.

Usage:
  python -m atlas.critic --results results/atlas_v0_resnet20_cifar10 results/atlas_v0_resnet20_seed1
  python -m atlas.critic --results ... --tol tolerances.yaml
"""
import argparse
import itertools
import json
import os

import numpy as np
import yaml
from scipy.stats import kendalltau, spearmanr

from .compare import SCALARS, _g, upper

DEFAULT_TOL = {
    "scalar_rel_spread": 0.15,       # (max-min)/|mean| across runs
    "scalar_abs_floor": 0.05,        # spreads below this pass regardless of rel (near-zero means)
    "adjacency_spearman": 0.70,
    "merge_kendall": 0.60,
    "decod_profile_spearman": 0.70,
    "decod_mean_abs_delta": 0.10,
    "commit_layer_slack": 1,         # positions
    "min_runs": 2,
}


def _load_all(dirs):
    runs = []
    for d in dirs:
        a = json.load(open(os.path.join(d, "atlas.json")))
        mpath = os.path.join(d, "manifest_used.yaml")
        m = yaml.safe_load(open(mpath)) if os.path.exists(mpath) else {}
        runs.append({"dir": d, "atlas": a, "manifest": m})
    return runs


def common_layers(runs):
    sets = [list(r["atlas"]["layers"]) for r in runs]
    return [l for l in sets[0] if all(l in s for s in sets[1:])]


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
    for l in common_layers(runs):
        for inv, key in SCALARS:
            vals = [_g(r["atlas"]["per_layer"][l], inv, key) for r in runs]
            vals = [v for v in vals if v is not None]
            if len(vals) < tol["min_runs"]:
                continue
            vals = np.array(vals, dtype=float)
            spread = float(vals.max() - vals.min())
            rel = spread / (abs(vals.mean()) + 1e-9)
            ok = (rel <= tol["scalar_rel_spread"]) or (spread <= tol["scalar_abs_floor"])
            items.append({"name": f"{l}/{inv}.{key}", "status": "PASS" if ok else "FAIL",
                          "detail": f"values={np.round(vals, 3).tolist()} rel_spread={rel:.3f}"})
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
    layers = common_layers(runs)
    pos = {l: i for i, l in enumerate(layers)}
    cls = [_g(r["atlas"], "cross_layer", "commit_layer", "per_factor") for r in runs]
    if not all(cls):
        return [{"name": "commit_layer", "status": "WARN", "detail": "commit_layer missing in some run"}]
    for f in cls[0]:
        cl = [c.get(f, {}).get("commit_layer") for c in cls]
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


CHECKS = [synthetic_refusal, holdout_hygiene, probe_hygiene, scalar_stability,
          adjacency_stability, decodability_stability, commit_agreement]


def run_critic(dirs, tol=None):
    tol = {**DEFAULT_TOL, **(tol or {})}
    runs = _load_all(dirs)
    report = {"runs": dirs, "tolerances": tol, "checks": {}}
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
    args = ap.parse_args()
    tol = yaml.safe_load(open(args.tol)) if args.tol else None
    rep = run_critic(args.results, tol)
    out = args.out or args.results[0]
    os.makedirs(out, exist_ok=True)
    json.dump(rep, open(os.path.join(out, "critic.json"), "w"), indent=1)
    write_md(rep, os.path.join(out, "CRITIC.md"))
    print(f"[critic] {rep['verdict']}  {rep['counts']}  -> {out}/CRITIC.md")


if __name__ == "__main__":
    main()
