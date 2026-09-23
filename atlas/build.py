"""
build.py -- STAGE B. Read a dump, run every selected invariant per layer, then the
cross-layer ones, write atlas.json (+ ATLAS.md and plots via report.py).

Usage:
  python -m atlas.build --dump results/<exp>/dump --out results/<exp> [--manifest m.yaml]
  python -m atlas.build --dump ... --only pca_spectrum,twonn_id       # subset
  python -m atlas.build --list                                        # show registry

CPU only. Runs unchanged on a synthetic dump (atlas.synth) and a real one; atlas.json
carries meta.source so the critic can refuse synthetic evidence.
"""
import argparse
import json
import os
import time
import traceback

import numpy as np

from . import factors as _factors  # noqa: F401  (registers factors)
from . import invariants as _inv    # noqa: F401  (registers invariants)
from .context import AtlasDump, build_context, to_jsonable
from .registry import INVARIANTS, CROSS_LAYER, select, describe


def build_atlas(dump_root, out_root, cfg=None, only=None, verbose=True):
    cfg = cfg or {}
    dump = AtlasDump(dump_root)
    wanted_inv = select(INVARIANTS, only.split(",") if only else cfg.get("invariants", "all"))
    wanted_cross = select(CROSS_LAYER, cfg.get("cross_layer", "all")) if not only else {
        k: v for k, v in CROSS_LAYER.items() if k in only.split(",")}
    inv_cfg = cfg.get("invariant_cfg", {})
    seed = int(cfg.get("seed", 0))

    # order: cheap -> medium -> expensive (kill-switch sequencing; cheap failures stop early)
    cost_rank = {"cheap": 0, "medium": 1, "expensive": 2}
    order = sorted(wanted_inv.values(), key=lambda s: cost_rank.get(s.cost, 9))

    atlas = {
        "exp_id": cfg.get("exp_id"),
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dump": os.path.abspath(dump_root),
        "source": dump.source,
        "meta": dump.meta,
        "layers": dump.layers,
        "per_layer": {},
        "cross_layer": {},
        "skipped": {},
        "timing_s": {},
    }
    ctxs = {}
    for layer in dump.layers:
        t0 = time.time()
        ctx = build_context(dump, layer, seed=seed)
        ctxs[layer] = ctx
        atlas["per_layer"][layer] = {"dim": ctx.dim, "n_ref": int(len(ctx.ref))}
        avail = ctx.available()
        for spec in order:
            missing = [p for p in spec.needs if p not in avail]
            if missing:
                atlas["skipped"].setdefault(spec.name, {})[layer] = f"missing {missing}"
                continue
            t1 = time.time()
            try:
                res = spec.fn(ctx, inv_cfg.get(spec.name, {}))
                atlas["per_layer"][layer][spec.name] = to_jsonable(res)
            except Exception as e:  # keep going; record the failure where the agent can see it
                atlas["per_layer"][layer][spec.name] = {"error": repr(e)[:300],
                                                        "trace": traceback.format_exc()[-800:]}
            atlas["timing_s"][f"{layer}/{spec.name}"] = round(time.time() - t1, 2)
        if verbose:
            print(f"[build] {layer:12s} D={ctx.dim:5d}  {time.time() - t0:6.1f}s  "
                  f"({', '.join(k for k in atlas['per_layer'][layer] if k not in ('dim', 'n_ref'))})")
        dump.drop_cache()

    for name, spec in wanted_cross.items():
        t1 = time.time()
        avail = set.intersection(*[c.available() for c in ctxs.values()]) if ctxs else set()
        missing = [p for p in spec.needs if p not in avail]
        if missing:
            atlas["skipped"][name] = f"missing {missing}"
            continue
        try:
            atlas["cross_layer"][name] = to_jsonable(spec.fn(ctxs, atlas["per_layer"], inv_cfg.get(name, {})))
        except Exception as e:
            atlas["cross_layer"][name] = {"error": repr(e)[:300], "trace": traceback.format_exc()[-800:]}
        atlas["timing_s"][f"cross/{name}"] = round(time.time() - t1, 2)
        if verbose:
            print(f"[build] cross-layer {name:14s} {time.time() - t1:6.1f}s")

    os.makedirs(out_root, exist_ok=True)
    path = os.path.join(out_root, "atlas.json")
    with open(path, "w") as f:
        json.dump(atlas, f, indent=1)
    if verbose:
        print(f"[build] wrote {path}")
    return atlas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", help="dump directory (Stage A output)")
    ap.add_argument("--out", help="where atlas.json / ATLAS.md go")
    ap.add_argument("--manifest", help="optional manifest yaml (invariant selection + cfg)")
    ap.add_argument("--only", help="comma-separated invariant names")
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--list", action="store_true", help="print registry and exit")
    args = ap.parse_args()
    if args.list:
        print(describe())
        return
    cfg = {}
    if args.manifest:
        from .config import load_manifest
        cfg = load_manifest(args.manifest)
    atlas = build_atlas(args.dump, args.out, cfg, only=args.only)
    if not args.no_report:
        from .report import write_report
        write_report(atlas, args.out, plots=cfg.get("outputs", {}).get("plots", True))


if __name__ == "__main__":
    main()
