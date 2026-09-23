"""
run_imagenet.py -- B1 entry point (docs/plans/B1_VIT_MARGIN.md): one ImageNet manifest end to end, as atlas.run does
for CIFAR, without touching atlas/extract_acts.py or atlas/run.py (A4b instrument files).

  python -m atlas.run_imagenet --manifest experiments/queue/margin_b1_resnet50.yaml --volume /workspace
  python -m atlas.run_imagenet --manifest experiments/queue/margin_b1_vitb16.yaml --volume /workspace \
      --selftest-random --out results/instrument_check_b1/selftest_margin_b1_vitb16.json

--selftest-random loads (downloads, hash-checks) the manifest's weights and runs random tensors through the hooks: 14 taps,
penult last, penult == final-norm token 0 (max |d| < 1e-5), head check <= 1e-3, logits == model(x). It reads no ImageNet
image and writes only --out (the ViTs' first data touch is after gate G). An existing dump is reused (Stage B only): a
dump is complete once its meta.json exists (written last); a dump dir without it (an interrupted Stage A) is extracted
again, once, and the log says so (integration D18).
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
    ap.add_argument("--device")
    ap.add_argument("--selftest-random", action="store_true")
    ap.add_argument("--out", help="--selftest-random: the JSON record")
    args = ap.parse_args()
    if args.selftest_random and not args.out:
        ap.error("--selftest-random needs --out")
    cfg = load_manifest(args.manifest)
    from .extract_imagenet import validate_cfg, selftest_random, extract_imagenet
    validate_cfg(cfg)                                     # before any download or data read
    if args.selftest_random:
        rep = selftest_random(cfg, args.device)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=1)
        print(f"[run_imagenet] selftest {rep['status']} {rep['checks']} -> {args.out}")
        raise SystemExit(0 if rep["status"] == "PASS" else 1)
    root = cfg["outputs"]["root"]
    os.makedirs(root, exist_ok=True)
    dump_manifest(cfg, os.path.join(root, "manifest_used.yaml"))
    dump, t0 = os.path.join(root, "dump"), time.time()
    if not os.path.exists(os.path.join(dump, "meta.json")):
        if os.path.isdir(dump):
            print(f"[run_imagenet] {dump} exists without meta.json (interrupted Stage A): extracting again (D18)")
        extract_imagenet(cfg, args.volume, device=args.device, dump=dump)
    else:
        print(f"[run_imagenet] dump exists at {dump}; skipping Stage A")
    from .build import build_atlas
    atlas = build_atlas(dump, root, cfg)
    from .report import write_report
    write_report(atlas, root, plots=cfg["outputs"].get("plots", False))
    from .extract_acts import git_commit
    prov = {"exp_id": cfg["exp_id"], "manifest": cfg["_manifest_path"], "dump_meta": atlas["meta"],
            "stage_b_git_commit": git_commit(), "dump_realpath": os.path.realpath(dump),
            "wall_s": round(time.time() - t0, 1), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(root, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)
    print(f"[run_imagenet] done in {prov['wall_s']}s -> {root}")


if __name__ == "__main__":
    main()
