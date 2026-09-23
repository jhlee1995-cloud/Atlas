"""
maxprob_ties.py -- A4b M56-c rule 8 (docs/plans/STAGE2B.md section M56): is margin's AUC lead over maxprob at depth 56
(A3 E5 on the hub: +0.0196 confidence-matched) a float32 saturation artifact? Dumps keep only the float32 softmax
maximum (atlas/extract_acts.py:172-174), so very confident samples collapse onto a few values near 1 and tie, and a
tie between a type-b and a correct sample scores 0.5 in the maxprob AUC while the continuous margin still orders it.
Per dump (clean test split): group sizes, the number of distinct maxprob values, the share of correct and of type-b
samples at the split's top value, and half the between-group tie probability, 0.5 * P(tie) = the most AUC the ties
can cost maxprob, for type-b vs all correct and vs confidence-matched correct (maxprob > cut). Groups as in
atlas/invariants/margin.py: type-b = wrong and maxprob > cut. A small tie mass excludes one artifact only
(TIES-EXCLUDED); it is not evidence that the lead is geometric. CPU only.

  python scripts/maxprob_ties.py --out results/instrument_check_a4b/maxprob_ties.json results/atlas_v1_resnet56_s1 ...
"""
import argparse
import json
import os

import numpy as np


def _half_tie(pos, neg):
    """0.5 * P(a positive and a negative sample have the same value); None when a group is empty."""
    if len(pos) == 0 or len(neg) == 0:
        return None
    vals, inv = np.unique(np.concatenate([pos, neg]), return_inverse=True)
    inv = np.asarray(inv).ravel()
    cp = np.bincount(inv[:len(pos)], minlength=len(vals)) / len(pos)
    cn = np.bincount(inv[len(pos):], minlength=len(vals)) / len(neg)
    return 0.5 * float(cp @ cn)


def tie_stats(maxprob, correct, cut=0.7):
    mp = np.asarray(maxprob, dtype=np.float64)
    ok = np.asarray(correct, dtype=bool)
    tb = ~ok & (mp > cut)
    top = float(mp.max())
    share = lambda m: float((mp[m] == top).mean()) if m.any() else None
    return {"n": int(len(mp)), "n_correct": int(ok.sum()), "n_typeb": int(tb.sum()), "cut": cut, "top_value": top,
            "n_distinct": int(len(np.unique(mp))), "top_share_correct": share(ok), "top_share_typeb": share(tb),
            "half_tie_typeb": _half_tie(mp[tb], mp[ok]),
            "half_tie_confmatched": _half_tie(mp[tb], mp[ok & (mp > cut)])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="result dirs with dump/preds/test.npz and dump/labels/test.npy")
    ap.add_argument("--cut", type=float, default=0.7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rep = {}
    for d in args.dirs:
        z = np.load(os.path.join(d, "dump", "preds", "test.npz"))
        y = np.load(os.path.join(d, "dump", "labels", "test.npy"))
        rep[os.path.basename(os.path.normpath(d))] = tie_stats(z["maxprob"], np.asarray(z["argmax"]) == y, args.cut)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=1)
    for k, v in rep.items():
        print(f"[maxprob_ties] {k}: top {v['top_value']!r} distinct {v['n_distinct']} share at top correct "
              f"{v['top_share_correct']} type-b {v['top_share_typeb']}; half-tie typeb {v['half_tie_typeb']} "
              f"confmatched {v['half_tie_confmatched']}")


if __name__ == "__main__":
    main()
