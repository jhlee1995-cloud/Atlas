"""
b1_gate.py -- gate G of B1 (docs/plans/B1_VIT_MARGIN.md; pre-registration experiments/queue/margin_b1_vitb16.yaml).
CPU only. Reads the two ResNet50 atlases built earlier in the same pod session, the B1 check dir (data.json, the ViT
self-tests) and exits 0 only if G0-G3 all hold. pod_atlas.sh --b1 builds the ViT manifests only on exit 0 (their first
and only margin touch). The thresholds are the ones in scripts/b1_verdicts.js (the local evaluator, which recomputes
this gate and checks it against gate.json); change neither without a new pre-registration.

  G0 plumbing: P0 of both ResNet50 dumps (decode_failures only outside the legacy block), legacy block
     n_test in [10048, 10111], lg.acc in [0.766, 0.776], lg.valley_sep_legacy in [1.12, 1.16]; both ViT self-tests
     (random inputs) PASS; data.json PASS and, when the ReaL labels verified, real.label_in_real_frac >= 0.5
  G1 lg.dir_auc_margin_full within max(0.035, 2.58 sd_sub + 0.005) of 0.800 and lg.raw_auc_margin_full < 0.5
  G2 legacy order margin > cluster > energy (direction-free, all correct); cluster within max(0.04, 2.58 sd + 0.005)
     of 0.636
  G3 atlas definition, penult: margin_minus_dist_typeb >= 0.01 and p < 0.05 at every cut in {0.7, 0.5} with
     n_typeb >= 300 (at least one such cut)
  B1-acc (INFO, integration D14): resnet50 official-232 minus legacy-256 accuracy on rows 0..L-1 (needs both dumps),
     mirror minus published acc@1, and the legacy cache <volume>/cache/imagenet/penult_10000.npz against the legacy
     dump's penult (rows, max |dF|) when that file exists.

  python scripts/b1_gate.py --out results/b1_gate --check-dir results/instrument_check_b1 [--volume /workspace]

<out>/gate.json is written once. A relaunch reuses it (exit 0 iff it was open) when both ResNet50 atlas.json files are
unchanged (sha256 recorded in "sources"); otherwise exit 1 (prereg review item 3).
"""
import argparse
import hashlib
import json
import os
import sys

SESOI, P, NMIN, CUTS = 0.01, 0.05, 300, (0.7, 0.5)
OFFICIAL = {"resnet50": 0.80858, "vitb16": 0.81072, "deitb": 0.8198}   # torchvision docs; timm results-imagenet.csv
BAND = {"resnet50": (-0.06, 0.01), "vitb16": (-0.10, 0.01), "deitb": (-0.10, 0.01)}   # P0 acc band (D14)
LEGACY_N, LEGACY_ACC, LEGACY_SEP = (10048, 10111), (0.766, 0.776), (1.12, 1.16)
G1_REF, G1_FLOOR, G2_REF, G2_FLOOR = 0.800, 0.035, 0.636, 0.04
SRC = ("margin_b1_resnet50_legacy10k", "margin_b1_resnet50")
SELFTESTS = ("margin_b1_vitb16", "margin_b1_deitb")


def _jload(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _load(root, exp):
    a = _jload(os.path.join(root, exp, "atlas.json"))
    if a is None:
        return None, None
    return a, ((a.get("per_layer") or {}).get("penult") or {}).get("margin_typeb") or {}


def _sha(root, exp):
    p = os.path.join(root, exp, "atlas.json")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _p0(name, a, pm, legacy, fails, model="resnet50", n_classes=1000, n_test=25000, per_class=25):
    """P0 plumbing of one run (notes P0). Appends '<name>: <check>' for every failed check."""
    if a is None:
        fails.append(f"{name}: no atlas.json")
        return
    m = a.get("meta") or {}
    hc = m.get("head_check_max_abs")
    checks = [("source real", m.get("source") == "real"), (f"n_classes {n_classes}", m.get("n_classes") == n_classes),
              ("head check <= 1e-3", _num(hc) and hc <= 1e-3), ("parquet sha256 ok", m.get("parquet_sha256_ok") is True),
              ("penult margin_typeb ok", bool(pm) and "error" not in pm)]
    if not legacy:                                   # the legacy code itself skipped undecodable rows (review item 17)
        c = m.get("per_class_counts") or {}
        acc = pm.get("acc") if pm else None
        lo, hi = OFFICIAL[model] + BAND[model][0], OFFICIAL[model] + BAND[model][1]
        checks += [("decode_failures 0", m.get("decode_failures") == 0), (f"n_test {n_test}", m.get("n_test") == n_test),
                   ("ref/test disjoint", m.get("ref_test_disjoint") is True),
                   (f"{per_class} per class", all(c.get(s) == [per_class, per_class] for s in ("ref", "test"))),
                   (f"acc band [{lo:.4f}, {hi:.4f}]", _num(acc) and lo <= acc <= hi)]
    fails += [f"{name}: {k}" for k, ok in checks if not ok]


def _b1_acc(root, La, Ma, volume):
    """INFO (D14): official-232 minus legacy-256 accuracy on rows 0..L-1, mirror minus published, legacy cache."""
    info = {"mirror_minus_published": None, "official_minus_legacy_rows_0_L": None, "legacy_cache": None}
    try:
        info["mirror_minus_published"] = {"resnet50": Ma["per_layer"]["penult"]["margin_typeb"]["acc"] - OFFICIAL["resnet50"]}
    except (TypeError, KeyError):
        pass
    try:
        import numpy as np
        L = int(La["meta"]["n_test"])
        dl, dm = os.path.join(root, SRC[0], "dump"), os.path.join(root, SRC[1], "dump")
        leg = np.load(os.path.join(dl, "preds", "test.npz"))["argmax"] == np.load(os.path.join(dl, "labels", "test.npy"))
        ok = {}
        for s, key in (("ref", "ref_indices"), ("test", "test_indices")):
            rows = np.asarray(Ma["meta"][key])
            c = np.load(os.path.join(dm, "preds", f"{s}.npz"))["argmax"] == np.load(os.path.join(dm, "labels", f"{s}.npy"))
            ok.update({int(r): bool(v) for r, v in zip(rows, c) if r < L})
        if len(ok) == L == len(leg):
            info["official_minus_legacy_rows_0_L"] = float(np.mean([ok[r] for r in range(L)]) - leg.mean())
        if volume:
            cp = os.path.join(volume, "cache", "imagenet", "penult_10000.npz")
            if os.path.exists(cp):
                Fc = np.load(cp)["F"]
                Fd = np.load(os.path.join(dl, "acts", "penult", "test.npy"))
                info["legacy_cache"] = {"path": cp, "rows": int(len(Fc)), "dump_rows": int(len(Fd)),
                                        "max_abs_dF": float(np.abs(Fc.astype(np.float64) - Fd).max()) if Fc.shape == Fd.shape else None}
    except Exception as e:                           # INFO only; never changes the gate
        info["note"] = repr(e)[:200]
    return info


def evaluate(root="results", check_dir="results/instrument_check_b1", volume=None):
    """The gate record (no file written): G0-G3, open, and their details."""
    La, lpm = _load(root, SRC[0])
    Ma, mpm = _load(root, SRC[1])
    fails, g = [], {}
    _p0("resnet50_legacy10k", La, lpm, True, fails)
    _p0("resnet50", Ma, mpm, False, fails)
    lg = (lpm or {}).get("legacy_imagenet") or {}
    n = ((La or {}).get("meta") or {}).get("n_test")
    num = lambda k: _num(lg.get(k))
    st = {e: (_jload(os.path.join(check_dir, f"selftest_{e}.json")) or {}).get("status") for e in SELFTESTS}
    data = _jload(os.path.join(check_dir, "data.json")) or {}
    real = data.get("real") or {}
    data_ok = data.get("status") == "PASS" and (not real.get("ok") or (_num(real.get("label_in_real_frac"))
                                                                        and real["label_in_real_frac"] >= 0.5))
    g["G0"] = bool(not fails and _num(n) and LEGACY_N[0] <= n <= LEGACY_N[1] and num("acc")
                   and LEGACY_ACC[0] <= lg["acc"] <= LEGACY_ACC[1] and num("valley_sep_legacy")
                   and LEGACY_SEP[0] <= lg["valley_sep_legacy"] <= LEGACY_SEP[1]
                   and all(v == "PASS" for v in st.values()) and data_ok)
    w = max(G1_FLOOR, 2.58 * (lg.get("dir_auc_margin_sub_sd") or 0.0) + 0.005)
    wc = max(G2_FLOOR, 2.58 * (lg.get("dir_auc_cluster_sub_sd") or 0.0) + 0.005)
    g["G1"] = bool(num("dir_auc_margin_full") and num("raw_auc_margin_full")
                   and abs(lg["dir_auc_margin_full"] - G1_REF) <= w and lg["raw_auc_margin_full"] < 0.5)
    g["G2"] = bool(all(num(f"dir_auc_{s}_full") for s in ("margin", "cluster", "energy"))
                   and lg["dir_auc_margin_full"] > lg["dir_auc_cluster_full"] > lg["dir_auc_energy_full"]
                   and abs(lg["dir_auc_cluster_full"] - G2_REF) <= wc)
    rows = {round(float(r["cut"]), 6): r for r in (mpm or {}).get("sweep") or []}
    g3, ev = {}, 0
    for c in CUTS:
        r = rows.get(c) or {}
        if (r.get("n_typeb") or 0) < NMIN:
            g3[str(c)] = {"n_typeb": r.get("n_typeb"), "evaluable": False}
            continue
        ev += 1
        d, p = r.get("margin_minus_dist_typeb"), r.get("margin_minus_dist_typeb_p")
        g3[str(c)] = {"n_typeb": r["n_typeb"], "d": d, "p": p,
                      "pass": _num(d) and _num(p) and d >= SESOI and p < P}
    g["G3"] = bool(ev > 0 and all(v.get("pass") for v in g3.values() if v.get("evaluable", True)))
    return {"gate": g, "open": all(g.values()), "plumbing_failures": fails, "check_dir": check_dir,
            "G0_detail": {"n_legacy": n, "acc": lg.get("acc"), "valley_sep_legacy": lg.get("valley_sep_legacy"),
                          "n_cw": lg.get("n_cw"), "n_classes_centered": lg.get("n_classes_centered"),
                          "legacy_decode_failures": ((La or {}).get("meta") or {}).get("decode_failures"),
                          "selftests": st, "data_status": data.get("status"), "real": real},
            "G3_detail": g3, "legacy_imagenet": lg, "band_halfwidth": {"G1": w, "G2": wc},
            "b1_acc_info": _b1_acc(root, La, Ma, volume) if La and Ma else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-dir", required=True, help="the B1 check dir (data.json, selftest_*.json)")
    ap.add_argument("--root", default="results", help="results tree holding the two ResNet50 runs")
    ap.add_argument("--volume", help="for the INFO legacy-cache comparison (<volume>/cache/imagenet)")
    args = ap.parse_args()
    path = os.path.join(args.out, "gate.json")
    src = {e: _sha(args.root, e) for e in SRC}
    if os.path.exists(path):                         # relaunch: reuse the record if its sources are unchanged
        old = _jload(path) or {}
        same = old.get("sources") == src
        print(f"[b1_gate] reusing {path}: open={old.get('open')} sources_unchanged={same}")
        sys.exit(0 if old.get("open") and same else 1)
    rep = evaluate(args.root, args.check_dir, args.volume)
    rep["sources"] = src
    os.makedirs(args.out, exist_ok=True)
    with open(path, "w") as f:
        json.dump(rep, f, indent=1)
    print(f"[b1_gate] {'OPEN' if rep['open'] else 'CLOSED'} {rep['gate']} plumbing={rep['plumbing_failures']} "
          f"G3={rep['G3_detail']} -> {path}")
    sys.exit(0 if rep["open"] else 1)


if __name__ == "__main__":
    main()
