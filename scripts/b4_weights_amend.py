#!/usr/bin/env python3
"""
b4_weights_amend.py -- B4a (docs/plans/B4A_AMENDMENT.md): the batch-4 weights gate of scripts/b4_weights.py with ONE
change, marked "B4a CHANGE": the README accuracy rule becomes one-sided. scripts/b4_weights.py (frozen at P1) is
imported, never edited, and its S1 record results/b4_weights/weights.json is never rewritten.

  B4  (D2, D15)  FAIL when |acc_10k - README top-1| > 0.003
  B4a            FAIL only when acc_10k < README top-1 - 0.003 (a loading, normalisation or architecture error lowers the
                 accuracy); acc_10k > README top-1 + 0.003 PASSES with the INFO flag README-EXCEEDED
Everything else is b4_weights.py unchanged: torch.hub at the pinned ref, the hub byte size (fatal), the full sha256, the
8-hex tag (else NOT_EVALUABLE), the head export and the full-10k accuracy (b4_weights.load_model, export_head,
test_accuracy). gate_unit below is b4_weights.gate_unit line for line outside the marked lines
(tests/test_b4a_amendment.py checks it).

  python scripts/b4_weights_amend.py --registry experiments/b4/models.json --volume /workspace \
      --s1-record results/b4_weights/weights.json --out results/b4_weights/weights_r2.json
                                          (scripts/b4a_supplement.sh, on the S1 pod after block_b4s1; GPU if present)
--units defaults to exactly the three B4a units (UNITS); another unit is refused unless --allow-other, and a unit that
is not a hub unit is always refused (only hub units have a README gate). Each B4a unit must be, in the S1 record, a
README-only FAIL above the README (status FAIL, readme status FAIL, delta > +0.003, sha256 recorded, no load error, the
hub byte size equal to the release asset, the 8-hex tag check passed: tag_prefix_ok true); otherwise the run is refused
(with --allow-other the check is recorded only). Every refusal happens before torch loads.
Output: the schema of weights.json (b4_extract.weights_gate reads units[<id>].status), plus "amendment": "B4a" at the top
and in each unit, readme.info / unit "info" = ["README-EXCEEDED"] when it applies, readme.status_b4 (what the frozen
two-sided rule says) and s1_crosscheck (sha256 and acc_10k equal to the S1 record; recorded, not a gate). Heads go to
<out dir>/heads<tag>/<id>/head.npz, <tag> from the --out name (weights_r2.json -> heads_r2), so the S1 heads are never
overwritten. Never overwrites --out or a head directory.
Exit: 0 when the record is written (a unit FAIL is per unit: the extractor refuses that unit); 2 on a refusal.

INTERFACE (pure, no torch; tests/test_b4a_amendment.py)
  readme_status(acc, want, tol) -> ('PASS' | 'FAIL', 'README-EXCEEDED' | None)
  readme_block(acc, want, tol) -> the rec['readme'] entry;  s1_readme_only_above(rec, tol) -> None | reason
  heads_dir(out) -> path | None;  counts(units) -> {status: n}
"""
import argparse
import json
import math
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b4_weights as BW  # noqa: E402  frozen at P1; torch is imported lazily inside its functions

AMENDMENT = "B4a"
DOC = "docs/plans/B4A_AMENDMENT.md"
UNITS = ("mobilenetv2_x0_75", "shufflenetv2_x0_5", "shufflenetv2_x1_0")     # the three S1 README-only FAILs (all above)
INFO_EXCEEDED = "README-EXCEEDED"
RULE = {"doc": DOC, "approved": "owner, 2026-09-24", "tol": BW.README_TOL,
        "b4": "FAIL when |acc_10k - readme_top1| > tol (scripts/b4_weights.py gate_unit)",
        "b4a": "FAIL only when acc_10k - readme_top1 < -tol; above +tol PASS with INFO README-EXCEEDED",
        "unchanged": "hub byte size (fatal), full sha256, 8-hex tag (NOT_EVALUABLE), head export, pinned hub ref"}
STATUSES = ("PASS", "NOT_EVALUABLE", "FAIL", "ABSENT")


def say(msg):
    print(f"[b4_weights_amend] {msg}", flush=True)


def readme_status(acc, want, tol=BW.README_TOL):
    """B4a CHANGE: the one-sided README rule. delta = acc - want is the frozen gate's expression, so the lower side is the
    frozen rule bit for bit (for delta < 0, abs(delta) > tol is exactly delta < -tol). A non-finite delta FAILS, as
    abs(nan) <= tol does in the frozen rule."""
    delta = acc - want
    if not math.isfinite(delta) or delta < -tol:
        return "FAIL", None
    return "PASS", (INFO_EXCEEDED if delta > tol else None)


def readme_block(acc, want, tol=BW.README_TOL):
    """rec['readme']: the frozen keys (acc_10k, readme_top1, delta, tol, status) with the B4a status, plus the INFO flag,
    the rule and what the frozen two-sided rule says (status_b4)."""
    st, info = readme_status(acc, want, tol)
    return {"acc_10k": acc, "readme_top1": want, "delta": acc - want, "tol": tol, "status": st, "info": info,
            "rule": "B4a one-sided: FAIL only when acc_10k < readme_top1 - tol",
            "status_b4": "PASS" if abs(acc - want) <= tol else "FAIL"}


def s1_readme_only_above(rec, tol=BW.README_TOL):
    """None when the S1 record of a unit is the B4a case (a README-only FAIL above the README), else the reason."""
    if not rec:
        return "no S1 record for the unit"
    r = rec.get("readme") or {}
    if rec.get("status") != "FAIL" or r.get("status") != "FAIL":
        return f"S1 status {rec.get('status')} (readme gate {r.get('status')}): not a README FAIL"
    if "why" in rec or not rec.get("sha256"):
        return "S1 failed before the README gate (load, byte size or download)"
    if rec.get("bytes") is None or rec.get("bytes") != rec.get("bytes_expected"):
        return f"S1 bytes {rec.get('bytes')} != bytes_expected {rec.get('bytes_expected')}: not README-only"
    if rec.get("tag_prefix_ok") is not True:
        return f"S1 tag_prefix_ok {rec.get('tag_prefix_ok')}: not README-only (the 8-hex tag check failed or is missing)"
    d = r.get("delta")
    if isinstance(d, bool) or not isinstance(d, (int, float)) or not d > tol:
        return f"S1 README delta {d} is not above +{tol}"
    return None


def heads_dir(out):
    """results/b4_weights/weights_r2.json -> <abs>/results/b4_weights/heads_r2; None when --out carries no relaunch tag."""
    m = re.fullmatch(r"weights((?:_r\d+)+)\.json", os.path.basename(out))
    return os.path.join(os.path.dirname(os.path.abspath(out)), "heads" + m.group(1)) if m else None


def counts(units):
    return {s: sum(1 for r in units.values() if r.get("status") == s) for s in STATUSES}


# b4_weights.gate_unit with the README decision replaced; every line outside the "B4a CHANGE" marks is the frozen one
# (with the b4_weights functions called through BW.).
def gate_unit(spec, volume, device, heads_root):
    t0 = time.time()
    try:
        model, rec = BW.load_model(spec, device)
    except FileNotFoundError:
        if spec["source"] == "file":
            return {"status": "ABSENT", "why": f"{spec.get('weights')} not on the volume (trained later in S1)"}
        raise
    except SystemExit as e:
        return {"status": "FAIL", "why": str(e)}
    except Exception as e:                                      # a hub download error fails this unit, not the session
        return {"status": "FAIL", "why": f"{type(e).__name__}: {e}"}
    rec["head"] = BW.export_head(model, os.path.join(heads_root, spec["id"]))
    if spec["source"] == "hub" and spec.get("readme_top1") is not None:
        acc = BW.test_accuracy(model, volume, spec["norm"], device)
        want = float(spec["readme_top1"]) / 100.0
        # B4a CHANGE begin: one-sided (was "status": "PASS" if abs(acc - want) <= README_TOL else "FAIL")
        rec["readme"] = readme_block(acc, want, BW.README_TOL)
        # B4a CHANGE end
        if rec["readme"]["status"] == "FAIL":
            rec["status"] = "FAIL"
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


def s1_crosscheck(rec, s1rec):
    """Recorded only (not a gate): the same file and the same 10k count as S1 measured."""
    s1rec = s1rec or {}
    a, b = rec.get("readme") or {}, s1rec.get("readme") or {}
    return {"sha256_equal": (rec.get("sha256") is not None and rec.get("sha256") == s1rec.get("sha256")),
            "acc_10k_equal": (a.get("acc_10k") is not None and a.get("acc_10k") == b.get("acc_10k")),
            "s1_status": s1rec.get("status"), "s1_readme_status": b.get("status")}


def main(argv=None):
    ap = argparse.ArgumentParser(description="B4a: the batch-4 weights gate with the one-sided README rule")
    ap.add_argument("--registry", required=True)
    ap.add_argument("--volume", required=True)
    ap.add_argument("--out", required=True, help="results/b4_weights/weights_r<k>.json (never overwritten)")
    ap.add_argument("--s1-record", default="results/b4_weights/weights.json", help="the S1 record (read only)")
    ap.add_argument("--units", default=",".join(UNITS), help="comma list (default: the three B4a units)")
    ap.add_argument("--heads", help="head export root (default: heads<tag> beside --out)")
    ap.add_argument("--allow-other", action="store_true", help="units outside UNITS; the S1 check is recorded only")
    ap.add_argument("--device")
    a = ap.parse_args(argv)

    def refuse(msg):
        print(f"[b4_weights_amend] REFUSED: {msg}", file=sys.stderr)
        return 2

    if os.path.exists(a.out):
        return refuse(f"{a.out} exists: never overwritten (relaunch with a new tag)")
    heads = os.path.abspath(a.heads) if a.heads else heads_dir(a.out)
    if not heads:
        return refuse(f"{a.out} is not weights_r<k>.json: give --heads (the S1 heads directory is never reused)")
    if heads == os.path.join(os.path.dirname(os.path.abspath(a.out)), "heads"):
        return refuse(f"{heads} is the S1 heads directory")
    with open(a.registry) as f:
        reg = json.load(f)
    byid = {m["id"]: m for m in reg["models"]}
    want = list(dict.fromkeys(u for u in a.units.split(",") if u))
    if not want:
        return refuse("--units is empty")
    bad = [u for u in want if u not in byid]
    if bad:
        return refuse(f"not in {a.registry}: {bad}")
    other = [u for u in want if u not in UNITS]
    if other and not a.allow_other:
        return refuse(f"B4a covers {list(UNITS)} only; {other} needs --allow-other")
    nonhub = [u for u in want if byid[u]["source"] != "hub" or byid[u].get("readme_top1") is None]
    if nonhub:
        return refuse(f"{nonhub}: B4a amends the README gate, which only hub units have")
    s1, s1_sha = None, None
    if os.path.isfile(a.s1_record):
        with open(a.s1_record) as f:
            s1 = json.load(f)
        s1_sha = BW.sha256_file(a.s1_record)
    elif not a.allow_other:
        return refuse(f"S1 record {a.s1_record} missing: B4a applies to units the S1 gate failed")
    s1u = (s1 or {}).get("units") or {}
    check = {u: s1_readme_only_above(s1u.get(u)) for u in want}
    notcase = {u: w for u, w in check.items() if w}
    if notcase and not a.allow_other:
        return refuse(f"not a README-only FAIL above the README in {a.s1_record}: {notcase}")
    taken = [u for u in want if os.path.exists(os.path.join(heads, u))]
    if taken:
        return refuse(f"{heads}/{taken} exists: a head export is never overwritten")

    units = [m for m in reg["models"] if m["id"] in want]                  # registry order, as b4_weights.py
    out = {"schema": BW.SCHEMA, "hub_ref": BW.HUB_REF, "registry": a.registry, "trained_only": False,
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "amendment": AMENDMENT,
           "amendment_rule": RULE, "heads": os.path.relpath(heads, BW.REPO_ROOT).replace(os.sep, "/"),
           "s1_record": {"path": a.s1_record, "sha256": s1_sha,
                         "check": {u: (w or "B4a case: README-only FAIL above the README") for u, w in check.items()}},
           "units": {}}
    import torch
    torch.set_grad_enabled(False)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out.update({"torch": torch.__version__, "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"})
    for m in units:
        rec = gate_unit(m, a.volume, device, heads)
        rec["amendment"] = AMENDMENT                                      # B4a record fields (every unit, early FAILs too)
        rec["info"] = [f for f in [(rec.get("readme") or {}).get("info")] if f]
        rec["s1_crosscheck"] = s1_crosscheck(rec, s1u.get(m["id"]))
        out["units"][m["id"]] = rec
        tail = ""
        if "readme" in rec:
            r = rec["readme"]
            tail = f"  (readme gate {r['status']} under B4a, {r['status_b4']} under B4"
            tail += (f"; INFO {r['info']})" if r["info"] else ")")
        say(f"{m['id']:22s} {rec['status']}{tail}")
        x = rec["s1_crosscheck"]
        if rec.get("sha256") and not (x["sha256_equal"] and x["acc_10k_equal"]):
            say(f"{m['id']}: WARNING sha256 or 10k count differs from the S1 record ({x}); recorded")
    out["counts"] = counts(out["units"])
    out["info_counts"] = {INFO_EXCEEDED: sum(1 for r in out["units"].values() if INFO_EXCEEDED in (r.get("info") or []))}
    if os.path.exists(a.out):
        return refuse(f"{a.out} appeared during the run: never overwritten")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1, allow_nan=False)
    os.replace(tmp, a.out)
    say(f"{out['counts']} info {out['info_counts']} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
