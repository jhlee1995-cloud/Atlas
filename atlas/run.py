"""
run.py -- execute one manifest end to end. This is the Runner role's only entry point.

  python -m atlas.run --manifest experiments/queue/atlas_v0_resnet20_cifar10.yaml --volume /workspace
  python -m atlas.run --manifest ... --skip-extract          # dump exists; CPU only
  python -m atlas.run --manifest ... --weights /workspace/models/resnet20_seed1.pt

Steps: (1) copy manifest -> results/<exp_id>/manifest_used.yaml, (2) Stage A unless dump
exists or --skip-extract, (3) Stage B build + report, (4) write provenance.json.
Results directory is the only side effect. Commit results/<exp_id>/ (without dump/, which
is large and reproducible) to the repo so later sessions can read atlas.json from raw GitHub.
"""
import argparse
import json
import os
import time

from .config import load_manifest, dump_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--volume", default="/workspace")
    ap.add_argument("--weights")
    ap.add_argument("--device")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()

    cfg = load_manifest(args.manifest)
    root = cfg["outputs"]["root"]
    os.makedirs(root, exist_ok=True)
    dump_manifest(cfg, os.path.join(root, "manifest_used.yaml"))
    dump = os.path.join(root, "dump")
    t0 = time.time()

    if not args.skip_extract and not os.path.exists(os.path.join(dump, "meta.json")):
        from .extract_acts import extract
        extract(cfg, args.volume, args.weights, args.device, args.download, dump)
    elif not os.path.exists(os.path.join(dump, "meta.json")):
        raise SystemExit(f"--skip-extract but no dump at {dump}")
    else:
        print(f"[run] dump exists at {dump}; skipping Stage A")

    from .build import build_atlas
    atlas = build_atlas(dump, root, cfg)
    if not args.no_report:
        from .report import write_report
        write_report(atlas, root, plots=cfg["outputs"].get("plots", True))

    prov = {"exp_id": cfg["exp_id"], "manifest": cfg["_manifest_path"], "dump_meta": atlas["meta"],
            "wall_s": round(time.time() - t0, 1), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(prov, open(os.path.join(root, "provenance.json"), "w"), indent=2)
    print(f"[run] done in {prov['wall_s']}s -> {root}")


if __name__ == "__main__":
    main()
