"""
report.py -- atlas.json -> ATLAS.md (tables) + PNG plots. The Recorder role of the loop.

Sections: layer table, decodability matrix, commit layers, sensitivity table, adjacency at
penult, density per split, skips/errors. Numbers only; interpretation is the agent's job and
goes in the session log, not here.
"""
import json
import os

import numpy as np


def _f(v, nd=3):
    if v is None:
        return "·"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _g(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def layer_table(atlas):
    rows = ["| layer | D | PR | eff.rank | dim95 | TwoNN ID | sep ratio | NC1 | ETF dev | hub skew | test sparse | NC acc(test) |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for l in atlas["layers"]:
        p = atlas["per_layer"][l]
        rows.append("| " + " | ".join([
            l, _f(p.get("dim")),
            _f(_g(p, "pca_spectrum", "participation_ratio"), 1),
            _f(_g(p, "pca_spectrum", "effective_rank_entropy"), 1),
            _f(_g(p, "pca_spectrum", "dim95"), 0),
            _f(_g(p, "twonn_id", "id"), 1),
            _f(_g(p, "class_centers", "sep_ratio"), 2),
            _f(_g(p, "neural_collapse", "nc1"), 2),
            _f(_g(p, "neural_collapse", "etf_deviation"), 3),
            _f(_g(p, "hubness", "k_occurrence_skew"), 2),
            _f(_g(p, "knn_density", "splits", "test", "sparse_frac"), 3),
            _f(_g(p, "class_centers", "nearest_center_acc_test"), 3),
        ]) + " |")
    return "\n".join(rows)


def decodability_table(atlas):
    layers = atlas["layers"]
    first = _g(atlas["per_layer"][layers[0]], "linear_probes", "factors")
    if not first:
        return "_linear_probes not run_"
    facs = list(first)
    rows = ["| factor | kind | " + " | ".join(layers) + " | commit | peak | washout |",
            "|---|---|" + "---|" * len(layers) + "---|---|---|"]
    commit = _g(atlas, "cross_layer", "commit_layer", "per_factor") or {}
    for f in facs:
        vals = []
        kind = None
        for l in layers:
            r = _g(atlas["per_layer"][l], "linear_probes", "factors", f) or {}
            kind = r.get("kind", kind)
            v = r.get("excess") if r.get("kind") == "categorical" else r.get("score")
            vals.append(_f(v, 2))
        c = commit.get(f, {})
        rows.append(f"| {f} | {kind or '·'} | " + " | ".join(vals) +
                    f" | {c.get('commit_layer') or '·'} | {c.get('peak_layer') or '·'} | {_f(c.get('washout'), 2)} |")
    return ("categorical = accuracy minus majority-class rate; continuous = CV R²\n\n" + "\n".join(rows))


def sensitivity_table(atlas, layer=None):
    layer = layer or atlas["layers"][-1]
    d = _g(atlas["per_layer"][layer], "corruption_displacement", "splits")
    if not d:
        return "_corruption_displacement not run_"
    rows = [f"layer `{layer}`; magnitude in within-class-radius units\n",
            "| corruption | sev | magnitude | per-sample | top5 frac | class-sub frac | coherence | class consistency | norm ratio |",
            "|---|---|---|---|---|---|---|---|---|"]
    for s in sorted(d, key=lambda k: (d[k]["corruption"], d[k]["severity"])):
        r = d[s]
        rows.append("| " + " | ".join([r["corruption"], str(r["severity"]), _f(r["magnitude"], 2),
                                       _f(r["per_sample_magnitude"], 2), _f(r["top5_frac"], 2),
                                       _f(r["class_sub_frac"], 2), _f(r["coherence"], 2),
                                       _f(r["class_consistency"], 2), _f(r["norm_ratio"], 3)]) + " |")
    return "\n".join(rows)


def density_table(atlas, layer=None):
    layer = layer or atlas["layers"][-1]
    d = _g(atlas["per_layer"][layer], "knn_density", "splits")
    if not d:
        return "_knn_density not run_"
    rows = [f"layer `{layer}`\n", "| split | sparse frac (ref q95) | median log-radius shift |", "|---|---|---|"]
    for s in sorted(d):
        rows.append(f"| {s} | {_f(d[s]['sparse_frac'])} | {_f(d[s]['median_log_radius_shift'])} |")
    return "\n".join(rows)


def adjacency_section(atlas, layer=None):
    layer = layer or atlas["layers"][-1]
    p = atlas["per_layer"][layer]
    out = [f"layer `{layer}`\n"]
    ca = p.get("class_adjacency") or {}
    if ca.get("closest_pairs"):
        out.append("closest class pairs (sep = center distance / RMS radius): " +
                   ", ".join(f"({a},{b}) {s:.2f}" for a, b, s in ca["closest_pairs"][:6]))
    if ca.get("valley_ratio_mean") is not None:
        out.append(f"valley ratio mean {ca['valley_ratio_mean']:.2f}, min {ca['valley_ratio_min']:.2f}, "
                   f"pairs with a valley {ca['frac_pairs_with_valley']:.2f}")
    if ca.get("top_confusions"):
        out.append("nearest-center confusions (true → nearest): " +
                   ", ".join(f"{a}→{b} {v:.2f}" for a, b, v in ca["top_confusions"][:6]))
    mo = p.get("merge_order") or {}
    if mo.get("merges"):
        seq = " ; ".join(f"{m['members']}@{m['height']:.2f}" for m in mo["merges"][:5])
        out.append(f"single-linkage merge order (first 5): {seq}")
    return "\n\n".join(out)


def flow_section(atlas):
    lc = _g(atlas, "cross_layer", "layer_cka")
    if not lc or "consecutive_cka" not in lc:
        return "_layer_cka not run_"
    rows = ["| pair | CKA |", "|---|---|"] + [f"| {k} | {v:.3f} |" for k, v in lc["consecutive_cka"].items()]
    rows.append(f"\nbiggest reorganization: `{lc.get('biggest_reorganization')}` (drop {_f(lc.get('biggest_reorganization_drop'))})")
    return "\n".join(rows)


def write_report(atlas, out_root, plots=True):
    md = [f"# ATLAS — {atlas.get('exp_id') or 'unnamed'}",
          f"built {atlas['built']} · source **{atlas['source']}** · arch `{atlas['meta'].get('arch')}` · "
          f"weights `{atlas['meta'].get('weights')}` · layers {len(atlas['layers'])}",
          "",
          "## 1. Per-layer invariants", layer_table(atlas), "",
          "## 2. Decodability profile (which factor lives where)", decodability_table(atlas), "",
          "## 3. Layer flow (CKA)", flow_section(atlas), "",
          "## 4. Sensitivity field (paired corruption displacement)", sensitivity_table(atlas), "",
          "## 5. Density per split", density_table(atlas), "",
          "## 6. Adjacency", adjacency_section(atlas), ""]
    if atlas.get("skipped"):
        md += ["## Skipped", "```", json.dumps(atlas["skipped"], indent=1), "```", ""]
    errs = {f"{l}/{k}": v["error"] for l, p in atlas["per_layer"].items() for k, v in p.items()
            if isinstance(v, dict) and "error" in v}
    errs.update({f"cross/{k}": v["error"] for k, v in atlas["cross_layer"].items() if isinstance(v, dict) and "error" in v})
    if errs:
        md += ["## Errors", "```", json.dumps(errs, indent=1), "```", ""]
    if atlas["source"] == "synthetic":
        md.insert(2, "> **SYNTHETIC dump.** Nothing here is evidence about a real model. Pipeline check only.")
    path = os.path.join(out_root, "ATLAS.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    if plots:
        try:
            make_plots(atlas, out_root)
        except Exception as e:  # plots are optional
            print("[report] plots failed:", repr(e)[:200])
    print(f"[report] wrote {path}")
    return path


def make_plots(atlas, out_root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    layers = atlas["layers"]
    pdir = os.path.join(out_root, "plots")
    os.makedirs(pdir, exist_ok=True)
    x = np.arange(len(layers))

    # 1. dimension profile
    ids = [_g(atlas["per_layer"][l], "twonn_id", "id") for l in layers]
    prs = [_g(atlas["per_layer"][l], "pca_spectrum", "participation_ratio") for l in layers]
    d95 = [_g(atlas["per_layer"][l], "pca_spectrum", "dim95") for l in layers]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for y, lab in [(ids, "TwoNN ID"), (prs, "participation ratio"), (d95, "dim95")]:
        if any(v is not None for v in y):
            ax.plot(x, [np.nan if v is None else v for v in y], marker="o", label=lab)
    ax.set_xticks(x); ax.set_xticklabels(layers, rotation=45, ha="right"); ax.legend(); ax.set_title("dimension profile")
    fig.tight_layout(); fig.savefig(os.path.join(pdir, "dimension_profile.png"), dpi=120); plt.close(fig)

    # 2. decodability heatmap
    first = _g(atlas["per_layer"][layers[0]], "linear_probes", "factors")
    if first:
        facs = list(first)
        M = np.full((len(facs), len(layers)), np.nan)
        for j, l in enumerate(layers):
            for i, f in enumerate(facs):
                r = _g(atlas["per_layer"][l], "linear_probes", "factors", f) or {}
                v = r.get("excess") if r.get("kind") == "categorical" else r.get("score")
                M[i, j] = np.nan if v is None else v
        fig, ax = plt.subplots(figsize=(1.2 + 0.7 * len(layers), 0.35 * len(facs) + 1.5))
        im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=max(1e-6, np.nanmax(M)))
        ax.set_xticks(range(len(layers))); ax.set_xticklabels(layers, rotation=45, ha="right")
        ax.set_yticks(range(len(facs))); ax.set_yticklabels(facs)
        fig.colorbar(im, ax=ax, label="probe score (excess acc / R²)"); ax.set_title("decodability profile")
        fig.tight_layout(); fig.savefig(os.path.join(pdir, "decodability.png"), dpi=120); plt.close(fig)

    # 3. sensitivity: coherence vs class-subspace fraction at the last layer, one dot per split
    d = _g(atlas["per_layer"][layers[-1]], "corruption_displacement", "splits")
    if d:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        for s, r in d.items():
            ax.scatter(r["coherence"], r["class_sub_frac"], s=20 + 40 * r["magnitude"], alpha=0.7)
            ax.annotate(f"{r['corruption'][:8]}{r['severity']}", (r["coherence"], r["class_sub_frac"]), fontsize=7)
        ax.set_xlabel("coherence (directional?)"); ax.set_ylabel("class-subspace fraction (semantic?)")
        ax.set_title(f"sensitivity field @ {layers[-1]} (size = magnitude)")
        fig.tight_layout(); fig.savefig(os.path.join(pdir, "sensitivity_field.png"), dpi=120); plt.close(fig)

    # 4. H0 count vs scale at the last layer
    mo = _g(atlas["per_layer"][layers[-1]], "merge_order")
    if mo and "h0_scales" in mo:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.step(mo["h0_scales"], mo["h0_counts"], where="post")
        ax.set_xlabel("scale (sep units)"); ax.set_ylabel("# components (H0)"); ax.set_title(f"landmark H0 @ {layers[-1]}")
        fig.tight_layout(); fig.savefig(os.path.join(pdir, "h0_scale.png"), dpi=120); plt.close(fig)
