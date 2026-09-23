"""
check_rebuild.py -- instrument checks around a confirmation run (Stage 1b: docs/plans/STAGE1.md amendment 2).
CPU only; `pod_atlas.sh --stage1b` calls every subcommand and every one exits 1 on FAIL.

  rebuild    I1: rebuild Stage B from a committed run's dump with the manifest that will measure the new
             seeds, and compare every leaf with the committed atlas.json -> <out>/check.json. PASS means the
             code AND the pod's library versions reproduce seed 1's committed atlas (so the new atlases are
             comparable with the committed s1, s2 and hub atlases), and that the manifest's explicit
             invariant / cross-layer / factor lists reproduce what `all` resolved to when s1 was built.
  replay     I1, the other half: compare and critic re-run with this commit's code on the committed Stage 1
             pair must reproduce the committed deformation.json (same leaf rule) and every critic item
             (status and detail) and tolerance -> <out>/replay.json. Covers compare.py, critic.py and
             experiments/tolerances_default.yaml, which compute every core item and which `rebuild` never runs.
  manifests  the new seeds' manifests agree in every key that defines the instrument (data, hooks, extract,
             invariants, cross_layer, invariant_cfg, probes, holdout, backbone arch and norm).
  dump-meta  I2: each new dump matches the reference dump in every Stage A contract field, names its own
             weights (a sha256 different from every reference run's), and its atlas has no error and no skip
             -> <out>/i2.json.

Leaf rule: identical key sets at every level; ints, strings, bools and None exact; floats within
ATOL + RTOL * |committed| (1e-3 each: about 1% of the tightest critic tolerance, far above float noise).
The top-level keys built, dump, timing_s and exp_id are not compared.

  python scripts/check_rebuild.py rebuild --dump results/atlas_v1_resnet20_s1/dump \
      --committed results/atlas_v1_resnet20_s1/atlas.json --manifest experiments/queue/atlas_v1_resnet20_s3.yaml \
      --work /workspace/scratch/stage1b_rebuild_s1 --out results/instrument_check_stage1b
  python scripts/check_rebuild.py replay \
      --deformation results/atlas_v1_resnet20_s2/compare_vs_atlas_v1_resnet20_s1/deformation.json \
      --deformation-new /workspace/scratch/stage1b_replay/compare_s1_s2/deformation.json \
      --critic results/critic_v1_resnet20_s1_s2/critic.json \
      --critic-new /workspace/scratch/stage1b_replay/critic_s1_s2/critic.json --out results/instrument_check_stage1b
  python scripts/check_rebuild.py manifests experiments/queue/atlas_v1_resnet20_s3.yaml \
      experiments/queue/atlas_v1_resnet20_s4.yaml --seed-tags s3 s4
  python scripts/check_rebuild.py dump-meta --ref results/atlas_v1_resnet20_s1 --others results/atlas_v1_resnet20_s2 \
      --new results/atlas_v1_resnet20_s3 results/atlas_v1_resnet20_s4 --out results/instrument_check_stage1b
"""
import argparse
import json
import os
import platform
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

SKIP_TOP = ("built", "dump", "timing_s", "exp_id")
ATOL = RTOL = 1e-3
MANIFEST_KEYS = ("data", "hooks", "extract", "invariants", "cross_layer", "invariant_cfg", "probes", "holdout")
BACKBONE_KEYS = ("arch", "norm")
I2_META_KEYS = ("layers", "dims", "splits", "n_test", "ref_indices", "panel_indices", "norm", "norm_values",
                "arch", "dtype", "pooling")


# ---- leaf-by-leaf comparison ---------------------------------------------------------------------------
def _family(path):
    p = path.split("/")
    if len(p) > 3 and p[1] == "per_layer":
        return p[3].split("[")[0]
    if len(p) > 2 and p[1] == "cross_layer":
        return "cross/" + p[2].split("[")[0]
    return p[1].split("[")[0] if len(p) > 1 else "(root)"


def _walk(a, b, path, acc):
    """a = committed, b = rebuilt."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if not path and k in SKIP_TOP:
                continue
            if k not in a or k not in b:
                acc["mismatch"].append([f"{path}/{k}", "new in rebuild" if k not in a else "missing in rebuild"])
            else:
                _walk(a[k], b[k], f"{path}/{k}", acc)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            acc["mismatch"].append([path, f"length {len(a)} vs {len(b)}"])
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _walk(x, y, f"{path}[{i}]", acc)
        return
    acc["n_leaves"] += 1
    num = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (a, b))
    if num and (isinstance(a, float) or isinstance(b, float)):
        d = abs(a - b)
        fam = _family(path)
        acc["max_abs_dev"][fam] = max(acc["max_abs_dev"].get(fam, 0.0), d)
        if d == 0:
            acc["n_exact"] += 1
        elif d <= ATOL + RTOL * abs(a):
            acc["n_within_tol"] += 1
        else:
            acc["mismatch"].append([path, f"{a!r} vs {b!r}"])
    elif type(a) is type(b) and a == b:
        acc["n_exact"] += 1
    else:
        acc["mismatch"].append([path, f"{a!r} vs {b!r}"])


def _errors(atlas):
    """'<layer>/<invariant>' and 'cross/<name>' entries that recorded an error."""
    errs = [f"{l}/{k}" for l, p in (atlas.get("per_layer") or {}).items() if isinstance(p, dict)
            for k, v in p.items() if isinstance(v, dict) and "error" in v]
    errs += [f"cross/{k}" for k, v in (atlas.get("cross_layer") or {}).items() if isinstance(v, dict) and "error" in v]
    return errs


def compare_atlases(committed, rebuilt):
    """Leaf-by-leaf comparison of two JSON dicts (atlas.json or deformation.json); a JSON-able report with
    status PASS / FAIL. FAIL also on any error entry or a non-empty `skipped` in the rebuilt dict."""
    acc = {"n_leaves": 0, "n_exact": 0, "n_within_tol": 0, "max_abs_dev": {}, "mismatch": []}
    _walk(committed, rebuilt, "", acc)
    errors = _errors(rebuilt)
    ok = not acc["mismatch"] and not errors and not rebuilt.get("skipped")
    return {"status": "PASS" if ok else "FAIL", "bitwise_identical": ok and acc["n_within_tol"] == 0,
            "rule": f"floats |a-b| <= {ATOL} + {RTOL}*|a|; all else exact; top-level {list(SKIP_TOP)} not compared",
            "n_leaves": acc["n_leaves"], "n_exact": acc["n_exact"], "n_within_tol": acc["n_within_tol"],
            "n_mismatch": len(acc["mismatch"]), "max_abs_dev": acc["max_abs_dev"], "errors": errors,
            "skipped": rebuilt.get("skipped"), "mismatch_first50": acc["mismatch"][:50]}


# ---- replay of compare + critic --------------------------------------------------------------------------
def replay_report(dfm_old, dfm_new, crit_old, crit_new):
    """Committed vs re-run deformation.json (leaf rule) and critic.json (every committed item's status and
    detail, every committed tolerance). Items only the re-run has are listed as INFO."""
    dc = compare_atlases(dfm_old, dfm_new)
    items = lambda c: {(k, i["name"]): [i["status"], i["detail"]] for k, v in c["checks"].items() for i in v}
    old, new = items(crit_old), items(crit_new)
    bad = [f"{k[0]}/{k[1]}: {old[k]} -> {new.get(k)}" for k in old if new.get(k) != old[k]]
    t_old, t_new = crit_old.get("tolerances") or {}, crit_new.get("tolerances") or {}
    bad += [f"tolerance {k}: {t_old[k]} -> {t_new.get(k)}" for k in t_old if t_new.get(k) != t_old[k]]
    ok = dc["status"] == "PASS" and not bad
    return {"status": "PASS" if ok else "FAIL", "compare": dc, "n_critic_items": len(old),
            "n_critic_mismatch": len(bad), "critic_mismatch_first50": bad[:50],
            "critic_new_items_INFO": sorted("/".join(k) for k in set(new) - set(old))}


# ---- manifests -------------------------------------------------------------------------------------------
def manifest_diff(a, b):
    """Instrument-defining keys in which two loaded manifests differ."""
    diff = [k for k in MANIFEST_KEYS if a.get(k) != b.get(k)]
    diff += [f"backbone.{k}" for k in BACKBONE_KEYS if (a.get("backbone") or {}).get(k) != (b.get("backbone") or {}).get(k)]
    return diff


# ---- I2: dump meta ---------------------------------------------------------------------------------------
def _sha(atlas):
    w = str((atlas.get("meta") or {}).get("weights"))
    return w.split("sha256:", 1)[1].strip() if "sha256:" in w else None


def dump_meta_report(ref, others, new, names):
    """ref, others, new: atlas.json dicts (meta = the dump's meta.json); names: the new runs' labels."""
    rm = ref.get("meta") or {}
    known = {s for s in [_sha(ref)] + [_sha(o) for o in others] if s}
    runs, seen = {}, set()
    for name, a in zip(names, new):
        m = a.get("meta") or {}
        problems = []
        diff = [k for k in I2_META_KEYS if m.get(k) != rm.get(k)]
        if diff:
            problems.append(f"meta differs from the reference in {diff}")
        w, tag, sha = str(m.get("weights")), m.get("seed_tag"), _sha(a)
        want = f"{str(m.get('arch', '')).split('_')[-1]}_{tag}_"
        if want not in w:
            problems.append(f"weights {w!r} do not name {want!r}")
        if sha is None:
            problems.append("no sha256 in meta.weights (hub or in-memory weights)")
        elif sha in known or sha in seen:
            problems.append(f"sha256 {sha} equals a reference or another new run")
        if sha:
            seen.add(sha)
        errs = _errors(a)
        if errs:
            problems.append(f"errors {errs[:10]}")
        if a.get("skipped"):
            problems.append(f"skipped {a.get('skipped')}")
        if a.get("source") != "real":
            problems.append(f"source {a.get('source')!r}")
        runs[name] = {"status": "FAIL" if problems else "PASS", "problems": problems, "weights": w,
                      "seed_tag": tag, "git_commit": m.get("git_commit"), "created": m.get("created"),
                      "accuracy_test": (m.get("accuracy") or {}).get("test")}
    ok = bool(runs) and all(r["status"] == "PASS" for r in runs.values())
    return {"status": "PASS" if ok else "FAIL", "keys_compared": list(I2_META_KEYS),
            "reference_git_commit": rm.get("git_commit"), "runs": runs}


# ---- CLI -------------------------------------------------------------------------------------------------
def _load(path):
    with open(path) as f:
        return json.load(f)


def _write(out, name, rep):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, name), "w") as f:
        json.dump(rep, f, indent=1)
    return os.path.join(out, name)


def cmd_rebuild(args):
    import numpy
    import scipy
    import sklearn
    from atlas.build import build_atlas
    from atlas.config import load_manifest
    from atlas.extract_acts import git_commit
    rebuilt = build_atlas(args.dump, args.work, load_manifest(args.manifest), verbose=False)
    rebuilt = json.loads(json.dumps(rebuilt))            # same serialization as the committed file
    rep = compare_atlases(_load(args.committed), rebuilt)
    rep.update({"dump": args.dump, "committed": args.committed, "manifest": args.manifest, "git_commit": git_commit(),
                "versions": {"python": platform.python_version(), "numpy": numpy.__version__,
                             "scipy": scipy.__version__, "sklearn": sklearn.__version__}})
    path = _write(args.out, "check.json", rep)
    print(f"[check_rebuild] I1 rebuild {rep['status']} (bitwise {rep['bitwise_identical']}): {rep['n_leaves']} leaves, "
          f"{rep['n_within_tol']} within tol, {rep['n_mismatch']} mismatches, errors {rep['errors']} -> {path}")
    return rep["status"] == "PASS"


def cmd_replay(args):
    from atlas.extract_acts import git_commit
    rep = replay_report(_load(args.deformation), _load(args.deformation_new), _load(args.critic), _load(args.critic_new))
    rep.update({"deformation": [args.deformation, args.deformation_new], "critic": [args.critic, args.critic_new],
                "git_commit": git_commit()})
    path = _write(args.out, "replay.json", rep)
    print(f"[check_rebuild] I1 replay {rep['status']}: compare {rep['compare']['status']} "
          f"({rep['compare']['n_mismatch']} mismatches), critic {rep['n_critic_mismatch']} of {rep['n_critic_items']} "
          f"items/tolerances differ -> {path}")
    return rep["status"] == "PASS"


def cmd_manifests(args):
    from atlas.config import load_manifest
    ms = [load_manifest(p) for p in args.manifests]
    bad = {p: d for p, m in zip(args.manifests[1:], ms[1:]) for d in [manifest_diff(ms[0], m)] if d}
    tags = [(m.get("backbone") or {}).get("seed_tag") for m in ms]
    if args.seed_tags and list(args.seed_tags) != tags:
        bad["seed_tags"] = f"expected {list(args.seed_tags)}, found {tags}"
    if len(set(tags)) != len(tags):
        bad["seed_tags_unique"] = tags
    print(f"[check_rebuild] manifests {'PASS' if not bad else 'FAIL'}: {args.manifests} seed_tags {tags}"
          + (f" differences {bad}" if bad else ""))
    return not bad


def cmd_dump_meta(args):
    new = [_load(os.path.join(d, "atlas.json")) for d in args.new]
    rep = dump_meta_report(_load(os.path.join(args.ref, "atlas.json")),
                           [_load(os.path.join(d, "atlas.json")) for d in args.others], new,
                           [os.path.basename(os.path.normpath(d)) for d in args.new])
    rep.update({"reference": args.ref, "others": args.others})
    path = _write(args.out, "i2.json", rep)
    print(f"[check_rebuild] I2 dump-meta {rep['status']}: "
          + "; ".join(f"{k} {v['status']} {v['problems']}" for k, v in rep["runs"].items()) + f" -> {path}")
    return rep["status"] == "PASS"


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("rebuild", help="I1: Stage B rebuild of a committed dump vs its committed atlas.json")
    p.add_argument("--dump", required=True, help="dump of the committed run (on the volume)")
    p.add_argument("--committed", required=True, help="that run's committed atlas.json")
    p.add_argument("--manifest", required=True, help="the manifest that will measure the new seeds")
    p.add_argument("--work", required=True, help="scratch dir for the rebuilt atlas.json (outside results/)")
    p.add_argument("--out", required=True, help="results dir for check.json")
    p = sub.add_parser("replay", help="I1: re-run compare / critic vs the committed Stage 1 files")
    p.add_argument("--deformation", required=True)
    p.add_argument("--deformation-new", required=True)
    p.add_argument("--critic", required=True)
    p.add_argument("--critic-new", required=True)
    p.add_argument("--out", required=True, help="results dir for replay.json")
    p = sub.add_parser("manifests", help="the new seeds' manifests define the same instrument")
    p.add_argument("manifests", nargs="+")
    p.add_argument("--seed-tags", nargs="*", help="expected backbone.seed_tag of each manifest, in order")
    p = sub.add_parser("dump-meta", help="I2: new dumps vs the reference dump")
    p.add_argument("--ref", required=True, help="reference result dir (atlas.json)")
    p.add_argument("--others", nargs="*", default=[], help="other committed runs whose weights must differ")
    p.add_argument("--new", nargs="+", required=True, help="new result dirs")
    p.add_argument("--out", required=True, help="results dir for i2.json")
    args = ap.parse_args()
    ok = {"rebuild": cmd_rebuild, "replay": cmd_replay, "manifests": cmd_manifests, "dump-meta": cmd_dump_meta}[args.cmd](args)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
