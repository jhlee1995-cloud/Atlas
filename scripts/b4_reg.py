#!/usr/bin/env python3
"""
b4_reg.py -- registry, job list, time limits, timing and replay helper for the batch-4 pod blocks (scripts/pod_b4.sh;
docs/plans/B4_INTEGRATION.md D3, D9, D10). Standard library only: it runs before and beside the numpy jobs.

  b4_reg.py --registry R [--cut "R2 Kwd"] ids ROLE...              ids with any of the roles (registry order, cuts applied)
  b4_reg.py --registry R get ID FIELD[.SUB]                         one field (weights, train.script, train.args, ...)
  b4_reg.py --registry R gb --group disc|C|F|K|R2|Sconf             GB a dump group writes (+15 %, ceil)
  b4_reg.py --registry R [--limits L] timeout PROG ID [--layout L]  seconds: experiments/b4/timeouts.json (P2) or limits
  b4_reg.py --registry R peak --phase PH                            GB per parallel job (timeouts.json peak_gb, else 4)
  b4_reg.py --registry R [--limits L] [--logdir D] jobs --phase discovery|replay|reprobe|confirmation [--tag T]
                                                                    one shell command per line, in launch order
  b4_reg.py --registry R timing --out F                             wall time and max RSS of every probe output
  b4_reg.py --registry R replay-compare --out F [--s1-tag T1] [--tag T]
                                                                    session-1 (tag T1) vs session-2 (tag T + _s2replay)
                                                                    anchor outputs (tolerance); the tags differ after a
                                                                    D10 relaunch of either session
  b4_reg.py --registry R versions-match --a F --b F --out F         python / numpy / scipy / sklearn equal?
  b4_reg.py --registry R validate                                   the D3 role counts and the layout contract
Limits (--limits, from scripts/pod_b4.sh, the only file whose limits may change between P1 and S1; D8):
  "t1=5400 t1_wide=7200 t2=1800 t2_wide=3600 t1s=3600 t3s=3600" (D10: 3x the upper design estimate, >= 1800 s).
  wide = penult width > 64; a T1 eval layout uses t1_wide.
Tags: '' | _r<k> (relaunch, same code) | _p2 (P2 re-probe) | _s2replay (S2 replay); anything else is refused.
"""
import argparse
import glob
import json
import math
import os
import re
import sys

TIMEOUTS = "experiments/b4/timeouts.json"
OUT = {  # program -> (script, output dir pattern, output file); = atlas/b4_core.OUTPUT_FILE
    "t1": ("scripts/t1_scoreboard.py", "results/b4_t1/{id}_{layout}{tag}", "scoreboard.json"),
    "t1s": ("scripts/t1_streams.py", "results/b4_t1s/{id}{tag}", "streams.json"),
    "t2": ("scripts/collapse_probe.py", "results/b4_t2/{id}{tag}", "probe.json"),
    "t3s": ("scripts/t3s_spatial_probe.py", "results/b4_t3s/{id}{tag}", "probe.json"),
}
GROUPS = {"disc": (("D", "N", "Dnew"), ("fit",), ("Sdisc",)), "C": (("C",), ("fit",), ()),
          "F": (("F", "F20"), ("fit", "eval"), ()), "K": (("K",), ("fit",), ()), "R2": (("R2",), ("eval",), ()),
          "Sconf": ((), (), ("Sconf",))}
KINDS = {"D", "N", "Dnew", "C", "F", "F20", "K", "Kls", "Kwd"}      # cutting a kind drops the unit; other roles (R2,
CUTTABLE = {"R2", "Kwd", "F20", "Sdisc", "Sconf", "STconf"}          # Sdisc, Sconf, ST*) drop only a layout / job
IGNORE = {"timing_s", "max_rss_mb", "env", "code", "created", "created_utc", "host", "versions", "out", "tag", "unseal"}
DEFAULT_LIMITS = "t1=5400 t1_wide=7200 t2=1800 t2_wide=3600 t1s=3600 t3s=3600"
TAG_RE = re.compile(r"(?:_r\d+|_p2|_s2replay)*")
TAG_END = re.compile(r"(?:_r\d+|_p2|_s2replay)*$")
ROLE_COUNTS = {"ANCHOR": 2, "D": 16, "N": 2, "Dnew": 5, "C": 12, "F": 2, "F20": 2, "K": 4, "Kls": 2, "Kwd": 2,
               "R2": 8, "Sdisc": 4, "Sconf": 4, "STdisc": 3, "STconf": 10}


def load(path, cut):
    with open(path) as f:
        reg = json.load(f)
    cut = set((cut or "").split())
    bad = cut - CUTTABLE - {m["id"] for m in reg["models"]}
    if bad:
        sys.exit(f"[b4_reg] not a registered cut: {sorted(bad)} (allowed: {sorted(CUTTABLE)} or a unit id)")
    for m in reg["models"]:
        drop = m["id"] in cut or bool(cut & KINDS & set(m["roles"]))
        m["eroles"] = [] if drop else [r for r in m["roles"] if r not in cut]
    reg["_path"] = path
    return reg


def ids(reg, roles):
    return [m["id"] for m in reg["models"] if any(r in m["eroles"] for r in roles)]


def model(reg, i):
    for m in reg["models"]:
        if m["id"] == i:
            return m
    sys.exit(f"[b4_reg] unknown unit {i}")


def parse_limits(s):
    out = {}
    for kv in (s or DEFAULT_LIMITS).split():
        k, v = kv.split("=", 1)
        out[k] = int(v)
    return out


def timeout(limits, prog, uid, reg, layout="fit"):
    key = f"{uid}_eval" if (prog == "t1" and layout == "eval") else uid
    if os.path.exists(TIMEOUTS):
        with open(TIMEOUTS) as f:
            t = (json.load(f).get(prog) or {}).get(key)
        if t:
            return int(t)
    wide = model(reg, uid)["penult"] > 64
    if prog == "t1":
        return limits["t1_wide"] if (wide or layout == "eval") else limits["t1"]
    if prog == "t2":
        return limits["t2_wide"] if wide else limits["t2"]
    return limits[prog]


def job(reg, limits, logdir, prog, i, phase, tag, layout="fit"):
    script, pat, _ = OUT[prog]
    out = pat.format(id=i, layout=layout, tag=tag)
    t = timeout(limits, prog, i, reg, layout)
    ph = "discovery" if phase in ("replay", "reprobe") else phase
    extra = f" --layout {layout}" if prog == "t1" else ""
    if prog == "t1s":
        extra = f" --frames /root/b4_frames/{i}{tag}"
    cmd = f"timeout {t} python {script} --registry {reg['_path']} --unit {i} --phase {ph}{extra} --out {out}"
    if logdir:
        cmd += f" > {logdir}/b4_{prog}_{i}_{layout}{tag}.log 2>&1"
    return cmd


def jobs(reg, limits, logdir, phase, tag):
    J = []
    has = (lambda i, r: r in model(reg, i)["eroles"])                     # noqa: E731
    anchors = ids(reg, ["ANCHOR"])
    mk = (lambda prog, i, tg, lay="fit": job(reg, limits, logdir, prog, i, phase, tg, lay))   # noqa: E731
    if phase in ("discovery", "reprobe", "replay"):
        tag = tag + {"discovery": "", "reprobe": "_p2", "replay": "_s2replay"}[phase]
        for i in anchors:                                                   # rule 6: resnet20 hub first
            J += [mk("t1", i, tag), mk("t2", i, tag)]
            if has(i, "Sdisc"):
                J.append(mk("t3s", i, tag))
        if phase == "replay":
            return J
        rest = [i for i in ids(reg, ["D", "N", "Dnew"]) if i not in anchors]
        J += [mk("t2", i, tag) for i in rest]
        J += [mk("t1", i, tag) for i in rest]
        J += [mk("t1s", i, tag) for i in ids(reg, ["STdisc"])]
        J += [mk("t3s", i, tag) for i in ids(reg, ["Sdisc"]) if i not in anchors]
        return J
    if phase != "confirmation":
        sys.exit(f"[b4_reg] unknown phase {phase}")
    # confirmation (D9): the owner's hypothesis first (cheap), then primary per-sample units, then breadth
    J += [mk("t2", i, tag) for i in ids(reg, ["C", "K", "F", "F20"])]
    for i in ids(reg, ["F", "F20"]):
        J += [mk("t1", i, tag, "fit"), mk("t1", i, tag, "eval")]
    J += [mk("t1", i, tag) for i in ids(reg, ["C", "K"])]
    J += [mk("t1s", i, tag) for i in ids(reg, ["STconf"]) if has(i, "F") or has(i, "F20") or has(i, "R2")]
    J += [mk("t1", i, tag, "eval") for i in ids(reg, ["R2"])]
    J += [mk("t3s", i, tag) for i in ids(reg, ["Sconf"])]
    return J


def outputs():
    for prog, (_, pat, fn) in OUT.items():
        root = pat.split("/{")[0]
        for f in sorted(glob.glob(os.path.join(root, "*", fn))):
            yield prog, os.path.basename(os.path.dirname(f)), f


def timing(reg, out):
    known = {m["id"] for m in reg["models"]}
    res = {p: {} for p in OUT}
    for prog, d, f in outputs():
        with open(f) as fh:
            j = json.load(fh)
        uid, layout = j.get("unit"), j.get("layout") or "fit"
        if uid is None:                                                     # fallback: parse the directory name
            base = TAG_END.sub("", d)
            uid, layout = (base.rsplit("_", 1) if prog == "t1" else (base, "fit"))
        if uid not in known:
            continue
        if prog == "t3s":
            layout = "maps"
        t = (j.get("timing_s") or {}).get("total")
        if t is None:
            continue
        cur = res[prog].get(uid)
        if cur is None or t > cur["wall_s"]:
            res[prog][uid] = {"wall_s": float(t), "max_rss_mb": j.get("max_rss_mb"), "layout": layout, "dir": d}
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"[b4_reg] timing of {sum(len(v) for v in res.values())} outputs -> {out}")


def same(x, y, path, bad):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)):
            if k in IGNORE:
                continue
            if k not in x or k not in y:
                bad.append(f"{path}/{k}: missing on one side")
            else:
                same(x[k], y[k], f"{path}/{k}", bad)
    elif isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            bad.append(f"{path}: length {len(x)} vs {len(y)}")
        for n, (a, b) in enumerate(zip(x, y)):
            same(a, b, f"{path}[{n}]", bad)
    elif (isinstance(x, (int, float)) and isinstance(y, (int, float))
          and not isinstance(x, bool) and not isinstance(y, bool)):
        if not math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-9):
            bad.append(f"{path}: {x} vs {y}")
    elif x != y:
        bad.append(f"{path}: {x!r} vs {y!r}")


def replay_compare(reg, out, tag, s1_tag=""):
    # a = the session-1 anchor output (tag s1_tag), b = the session-2 replay (tag + _s2replay): a D10 relaunch of S2
    # (ATLAS_B4_TAG=_r2) or of S1 (ATLAS_B4_S1_TAG) changes one side only
    rep = {"rule": "rel 1e-6, abs 1e-9; keys ignored: " + ", ".join(sorted(IGNORE)), "units": {},
           "tags": {"s1": s1_tag, "s2": tag}}
    for i in ids(reg, ["ANCHOR"]):
        for prog in ("t1", "t2", "t3s"):
            _, pat, fn = OUT[prog]
            a = os.path.join(pat.format(id=i, layout="fit", tag=s1_tag), fn)
            b = os.path.join(pat.format(id=i, layout="fit", tag=tag + "_s2replay"), fn)
            if not (os.path.exists(a) and os.path.exists(b)):
                rep["units"][f"{prog}:{i}"] = {"status": "NOT_EVALUABLE", "why": "missing output"}
                continue
            bad = []
            with open(a) as fa, open(b) as fb:
                same(json.load(fa), json.load(fb), "", bad)
            rep["units"][f"{prog}:{i}"] = {"status": "PASS" if not bad else "FAIL", "n_diff": len(bad),
                                           "first20": bad[:20]}
    st = [v["status"] for v in rep["units"].values()]
    rep["status"] = "PASS" if st and all(s == "PASS" for s in st) else "FAIL"
    with open(out, "w") as f:
        json.dump(rep, f, indent=1)
    print(f"[b4_reg] replay {rep['status']} -> {out}")
    return rep


def validate(reg):
    errs = []
    idl = [m["id"] for m in reg["models"]]
    if len(idl) != len(set(idl)):
        errs.append("duplicate ids")
    if len(idl) != 43:
        errs.append(f"{len(idl)} units, want 43 (D2)")
    for r, n in ROLE_COUNTS.items():
        got = sum(r in m["roles"] for m in reg["models"])
        if got != n:
            errs.append(f"role {r}: {got} units, want {n}")
    for m in reg["models"]:
        for lay, L in m["layouts"].items():
            want = "results/b4c_" if L.get("sealed") else "results/b4d_"
            if not L["dump"].startswith(want) or not L["dump"].endswith("/dump"):
                errs.append(f"{m['id']} {lay}: dump {L['dump']} does not match sealed={L.get('sealed')}")
        sealed_fit = any(r in m["roles"] for r in ("C", "F", "F20", "K"))
        if m["layouts"]["fit"].get("sealed") != sealed_fit:
            errs.append(f"{m['id']}: fit sealed must be {sealed_fit}")
        if ("eval" in m["layouts"]) != any(r in m["roles"] for r in ("F", "F20", "R2")):
            errs.append(f"{m['id']}: eval layout iff F, F20 or R2")
        if ("maps" in m["layouts"]) != any(r in m["roles"] for r in ("Sdisc", "Sconf")):
            errs.append(f"{m['id']}: maps layout iff Sdisc or Sconf")
        if m["source"] == "hub" and not (m.get("hub_asset") and m.get("readme_top1")):
            errs.append(f"{m['id']}: hub unit without hub_asset / readme_top1")
        if any(ord(ch) > 127 for ch in json.dumps(m)):
            errs.append(f"{m['id']}: non-ASCII")
    return errs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--cut", default="")
    ap.add_argument("--limits", default=DEFAULT_LIMITS)
    ap.add_argument("--logdir", default="")
    ap.add_argument("cmd")
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--group")
    ap.add_argument("--phase")
    ap.add_argument("--layout", default="fit")
    ap.add_argument("--tag", default="")
    ap.add_argument("--s1-tag", default="")                              # replay-compare: the S1 anchor outputs' tag
    ap.add_argument("--out")
    ap.add_argument("--a")
    ap.add_argument("--b")
    a = ap.parse_args(argv)
    for t in (a.tag, a.s1_tag):
        if not TAG_RE.fullmatch(t):
            sys.exit(f"[b4_reg] tag {t!r} is not '' / _r<k> / _p2 / _s2replay")
    reg = load(a.registry, a.cut)
    limits = parse_limits(a.limits)
    if a.cmd == "ids":
        print(" ".join(ids(reg, a.rest)))
    elif a.cmd == "get":
        v = model(reg, a.rest[0])
        for k in a.rest[1].split("."):
            v = v[k]
        print(v)
    elif a.cmd == "gb":
        roles, lays, maps = GROUPS[a.group]
        g = sum(m["layouts"][lay]["gb"] for m in reg["models"] if any(r in m["eroles"] for r in roles)
                for lay in lays if lay in m["layouts"])
        g += sum(m["layouts"]["maps"]["gb"] for m in reg["models"] if any(r in m["eroles"] for r in maps))
        print(int(math.ceil(g * 1.15)))                                       # +15 % for estimate error
    elif a.cmd == "timeout":
        print(timeout(limits, a.rest[0], a.rest[1], reg, a.layout))
    elif a.cmd == "peak":
        p = {}
        if os.path.exists(TIMEOUTS):
            with open(TIMEOUTS) as f:
                p = json.load(f).get("peak_gb", {}) or {}
        print(max(p.values()) if p else 4)
    elif a.cmd == "jobs":
        print("\n".join(jobs(reg, limits, a.logdir, a.phase, a.tag)))
    elif a.cmd == "timing":
        timing(reg, a.out)
    elif a.cmd == "replay-compare":
        replay_compare(reg, a.out, a.tag, a.s1_tag)
    elif a.cmd == "versions-match":
        with open(a.a) as fa, open(a.b) as fb:
            va, vb = json.load(fa), json.load(fb)
        keys = ("python", "numpy", "scipy", "sklearn")
        r = {k: [va.get(k), vb.get(k)] for k in keys}
        r["match"] = all(va.get(k) == vb.get(k) for k in keys)
        with open(a.out, "w") as f:
            json.dump(r, f, indent=1)
        print("[b4_reg] versions match" if r["match"] else f"[b4_reg] VERSION DRIFT {r}")
        return 0 if r["match"] else 1
    elif a.cmd == "validate":
        errs = validate(reg)
        print("[b4_reg] registry OK" if not errs else "[b4_reg] registry errors:\n  " + "\n  ".join(errs))
        return 1 if errs else 0
    else:
        sys.exit(f"[b4_reg] unknown command {a.cmd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
