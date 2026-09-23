"""
ladder.py -- aggregate a deformation ladder run (scripts/tta_deform.py or any script that
writes dump_step<k>/ directories) into a dose-response table.

For each results/<exp>/dump_step<k>:
  1. build the atlas into results/<exp>/step<k>/ (Stage B, skipped if atlas.json exists)
  2. compare step<k> to step0 with --same-space
  3. pull ground truth from the dump meta (accuracies, pred_entropy)
Then write LADDER.md, ladder.json and plots/ladder.png, and compute the bar:
  onset_step[metric]   first step at which |metric| leaves its step-0 value by more than `tol`
  onset_step[collapse] first step at which pred_entropy < collapse_frac * pred_entropy(step 0)
  "earlier" = onset_step[metric] < onset_step[collapse]  (the atlas saw it before the histogram)

Usage:
  python -m atlas.ladder --exp results/tta_tent_resnet20_fog3 [--layer penult] [--tol 0.05]
"""
import argparse
import glob
import json
import os
import re
import shutil

import numpy as np

from .build import build_atlas
from .compare import compare
from .config import load_manifest

METRICS = [  # (name, source, extractor)
    ("panel_cka", "cmp", lambda r: r.get("panel_cka")),
    ("panel_shift", "cmp", lambda r: r.get("panel_mean_shift_within_units")),
    ("center_disp", "cmp", lambda r: (r.get("center_displacement_within_units") or {}).get("mean")),
    ("adjacency_rho", "cmp", lambda r: r.get("adjacency_spearman")),
    ("merge_tau", "cmp", lambda r: r.get("merge_height_kendall")),
    ("decod_abs_delta", "cmp", lambda r: r.get("decodability_mean_abs_delta")),
    ("d_twonn_id", "cmp", lambda r: (r["scalars"].get("twonn_id.id") or {}).get("delta")),
    ("d_sep_ratio", "cmp", lambda r: (r["scalars"].get("class_centers.sep_ratio") or {}).get("delta")),
    ("d_nc1", "cmp", lambda r: (r["scalars"].get("neural_collapse.nc1") or {}).get("delta")),
    ("d_etf_dev", "cmp", lambda r: (r["scalars"].get("neural_collapse.etf_deviation") or {}).get("delta")),
]
IDENTITY_VALUE = {"panel_cka": 1.0, "adjacency_rho": 1.0, "merge_tau": 1.0}   # value at zero deformation


def find_steps(exp):
    steps = []
    for d in glob.glob(os.path.join(exp, "dump_step*")):
        m = re.search(r"dump_step(\d+)$", d)
        if m and os.path.exists(os.path.join(d, "meta.json")):
            steps.append(int(m.group(1)))
    return sorted(steps)


def ensure_atlas(exp, step, cfg):
    out = os.path.join(exp, f"step{step}")
    if not os.path.exists(os.path.join(out, "atlas.json")):
        build_atlas(os.path.join(exp, f"dump_step{step}"), out, cfg, verbose=False)
        print(f"[ladder] built atlas for step {step}")
    # compare needs <dir>/dump for panel metrics: symlink the step dump
    link = os.path.join(out, "dump")
    if not os.path.exists(link):
        src = os.path.abspath(os.path.join(exp, f"dump_step{step}"))
        try:
            os.symlink(src, link, target_is_directory=True)
        except OSError:
            # Windows without Developer Mode (WinError 1314): copy instead. Only small local or
            # synthetic dumps reach this; on the pod the symlink path is taken.
            shutil.copytree(src, link)
    return out


def run_ladder(exp, layer="penult", tol=0.05, collapse_frac=0.8, manifest=None):
    cfg = load_manifest(manifest) if manifest else (
        load_manifest(os.path.join(exp, "manifest_used.yaml")) if os.path.exists(os.path.join(exp, "manifest_used.yaml")) else {})
    steps = find_steps(exp)
    if not steps or steps[0] != 0:
        raise SystemExit(f"need dump_step0 under {exp}; found steps {steps}")
    dirs = {s: ensure_atlas(exp, s, cfg) for s in steps}
    rows = []
    for s in steps:
        meta = json.load(open(os.path.join(exp, f"dump_step{s}", "meta.json")))
        gt = meta.get("ground_truth", {})
        row = {"step": s, **{k: gt.get(k) for k in ("acc_stream_heldout", "acc_clean", "acc_heldout_corr", "pred_entropy")}}
        if s == 0:
            for name, _, _ in METRICS:
                row[name] = IDENTITY_VALUE.get(name, 0.0)
        else:
            cmp = compare(dirs[0], dirs[s], same_space=True)
            r = cmp["per_layer"].get(layer) or cmp["per_layer"][list(cmp["per_layer"])[-1]]
            for name, _, fn in METRICS:
                try:
                    row[name] = fn(r)
                except Exception:
                    row[name] = None
            # mean over all layers for the two panel metrics (deformation can start early)
            cks = [v.get("panel_cka") for v in cmp["per_layer"].values() if v.get("panel_cka") is not None]
            row["panel_cka_min_layer"] = min(cks) if cks else None
            row["panel_cka_argmin_layer"] = (min(cmp["per_layer"], key=lambda l: cmp["per_layer"][l].get("panel_cka", 9))
                                             if cks else None)
        rows.append(row)

    # onset analysis
    pe0 = rows[0].get("pred_entropy")
    onset = {}
    if pe0:
        onset["collapse(pred_entropy)"] = next((r["step"] for r in rows if r.get("pred_entropy") is not None
                                                and r["pred_entropy"] < collapse_frac * pe0), None)
    for name, _, _ in METRICS + [("panel_cka_min_layer", None, None)]:
        base = IDENTITY_VALUE.get(name, IDENTITY_VALUE.get("panel_cka") if name == "panel_cka_min_layer" else 0.0)
        onset[name] = next((r["step"] for r in rows if r.get(name) is not None and abs(r[name] - base) > tol), None)
    c = onset.get("collapse(pred_entropy)")
    earlier = {k: (v is not None and (c is None or v < c)) for k, v in onset.items() if not k.startswith("collapse")}

    res = {"exp": exp, "layer": layer, "tol": tol, "collapse_frac": collapse_frac, "rows": rows,
           "onset_step": onset, "earlier_than_collapse": earlier}
    json.dump(res, open(os.path.join(exp, "ladder.json"), "w"), indent=1)
    write_md(res, os.path.join(exp, "LADDER.md"))
    try:
        plot(res, exp)
    except Exception as e:
        print("[ladder] plot failed:", repr(e)[:120])
    print(f"[ladder] {len(rows)} steps -> {exp}/LADDER.md")
    return res


def write_md(res, path):
    f = lambda v: "·" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
    cols = ["step", "acc_stream_heldout", "acc_clean", "acc_heldout_corr", "pred_entropy"] + [m[0] for m in METRICS] + ["panel_cka_min_layer", "panel_cka_argmin_layer"]
    L = [f"# LADDER  {res['exp']}  (layer {res['layer']}, tol {res['tol']})", "",
         "ground truth uses labels offline; every other column is label-free map deformation vs step 0", "",
         "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in res["rows"]:
        L.append("| " + " | ".join(f(r.get(c)) for c in cols) + " |")
    L += ["", "## onset (first step the quantity leaves its step-0 value by more than tol)", "",
          "| quantity | onset step | earlier than histogram collapse? |", "|---|---|---|"]
    for k, v in res["onset_step"].items():
        e = res["earlier_than_collapse"].get(k)
        L.append(f"| {k} | {v} | {'' if e is None else ('yes' if e else 'no')} |")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


def plot(res, exp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = res["rows"]
    x = [r["step"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for k in ("acc_stream_heldout", "acc_clean", "acc_heldout_corr", "pred_entropy"):
        axes[0].plot(x, [np.nan if r.get(k) is None else r[k] for r in rows], marker="o", label=k)
    axes[0].set_title("ground truth (offline labels)"); axes[0].legend(fontsize=7); axes[0].set_xlabel("step")
    for k in ("panel_cka", "panel_cka_min_layer", "adjacency_rho", "merge_tau"):
        axes[1].plot(x, [np.nan if r.get(k) is None else r[k] for r in rows], marker="o", label=k)
    axes[1].set_title("agreement with step 0 (1 = identical)"); axes[1].legend(fontsize=7); axes[1].set_xlabel("step")
    for k in ("panel_shift", "center_disp", "decod_abs_delta", "d_twonn_id", "d_sep_ratio"):
        axes[2].plot(x, [np.nan if r.get(k) is None else r[k] for r in rows], marker="o", label=k)
    axes[2].set_title("displacement / deltas"); axes[2].legend(fontsize=7); axes[2].set_xlabel("step")
    os.makedirs(os.path.join(exp, "plots"), exist_ok=True)
    fig.tight_layout(); fig.savefig(os.path.join(exp, "plots", "ladder.png"), dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--layer", default="penult")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--collapse-frac", type=float, default=0.8)
    ap.add_argument("--manifest")
    a = ap.parse_args()
    run_ladder(a.exp, a.layer, a.tol, a.collapse_frac, a.manifest)


if __name__ == "__main__":
    main()
