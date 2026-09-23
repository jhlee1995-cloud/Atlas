"""
matched_rung.py -- A4b (docs/plans/STAGE2B.md "Matched value"): the pre-registered matched-rung rule applied to the
seed-11 ladder's train.json files, so the pod knows whether (and for how many epochs) to train the seed-12 and
seed-13 replicates. The Evaluator re-applies the same rule from the pulled files (scripts/a4b_eval.js), where M11
must also have a built atlas; this script decides nothing else.

Rule: accuracy = train.json final_test_acc_10k (full 10k test set, last epoch); target 0.9259
(results/norm_check_resnet20/norm_check.json chenyaofo.acc_10k); a rung is in the window when |acc - target|, rounded
to 4 decimals (the 10k resolution), is <= 0.005. E* = the in-window rung with the smallest |acc - target|; a tie goes
to the longer E. Status: matched (E* exists) | interpolate (none in the window, target between the lowest and the
highest rung) | below (every rung below the window: the pod adds the e90 extension rung) | above (every rung above).

Prints "<status> <E* or none>" and writes the record to --out atomically (a temporary file, then os.replace): the
pod's CPU lane polls for that file, so it must never see a partial one. Unusable input (a train.json that is missing,
unreadable, or not a seed-11 resnet56 chenyaofo rung) writes {"status": "error", "e_star": null, "reason": ...} the
same way, prints nothing on stdout and exits 2; the pod treats it as no E*.

  python scripts/matched_rung.py --out results/instrument_check_a4b/ladder.json \
      results/train_resnet56_e40/train.json results/train_resnet56_e50/train.json ...
"""
import argparse
import json
import os
import sys

TARGET = 0.9259
WINDOW = 0.005


def pick(rungs, target=TARGET, window=WINDOW):
    """rungs: [(E, acc)] -> (status, E* or None, sorted in-window Es)."""
    if not rungs:
        raise ValueError("no rungs")
    d = {e: round(abs(a - target), 4) for e, a in rungs}
    inside = sorted(e for e in d if d[e] <= window)
    if inside:
        return "matched", min(inside, key=lambda e: (d[e], -e)), inside
    accs = [a for _, a in rungs]
    if max(accs) < target:
        return "below", None, []
    if min(accs) > target:
        return "above", None, []
    return "interpolate", None, []


def write_json(path, rep):
    """Atomic write: a reader polling for `path` sees either nothing or the complete file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rep, f, indent=1)
    os.replace(tmp, path)


def read_rungs(paths, target):
    """train.json paths -> ([(E, acc)], report rows); ValueError on anything that is not a seed-11 rung."""
    rungs, rows = [], []
    for p in paths:
        with open(p) as f:
            t = json.load(f)
        r = t["recipe"]
        if r["arch"] != "cifar10_resnet56" or int(r["seed"]) != 11 or r["norm"] != "chenyaofo":
            raise ValueError(f"{p}: not a seed-11 resnet56 chenyaofo rung ({r['arch']}, {r['seed']}, {r['norm']})")
        e, a = int(r["epochs"]), float(t["final_test_acc_10k"])
        rungs.append((e, a))
        rows.append({"E": e, "acc": a, "abs_delta": round(abs(a - target), 4), "train_json": p})
    return rungs, rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("train_json", nargs="+", help="train.json of the seed-11 rungs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=float, default=TARGET)
    ap.add_argument("--window", type=float, default=WINDOW)
    args = ap.parse_args(argv)
    try:
        rungs, rows = read_rungs(args.train_json, args.target)
        status, e_star, inside = pick(rungs, args.target, args.window)
    except (OSError, KeyError, TypeError, ValueError) as ex:
        write_json(args.out, {"target": args.target, "window": args.window, "status": "error", "e_star": None,
                              "in_window": [], "reason": repr(ex), "train_json": args.train_json})
        print(f"[matched_rung] error: {ex!r} -> {args.out}", file=sys.stderr)
        return 2
    rep = {"target": args.target, "window": args.window, "rule": "nearest in-window rung; tie -> longer E",
           "rungs": sorted(rows, key=lambda x: x["E"]), "status": status, "e_star": e_star, "in_window": inside}
    write_json(args.out, rep)
    print(f"{status} {e_star if e_star is not None else 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
