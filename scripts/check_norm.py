"""
check_norm.py -- which input normalization fits the hub checkpoint? Records evidence; never gates.

Evaluates the hub model on the full CIFAR-10 test set (10k) under every entry of
atlas.extract_acts.NORMS and reports top-1, test[:5000] accuracy (the dumps' n_test), argmax
agreement between the two normalizations, and an exact McNemar test on the discordant pairs.
The Stage 1 atlases use norm=chenyaofo because the upstream training log says so; this script
checks that the released file behaves like it (pre-registered in the v1 manifests as N1/N2).

  python scripts/check_norm.py --volume /workspace --out results/norm_check_resnet20/norm_check.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torchvision
from scipy.stats import binomtest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from atlas.extract_acts import NORMS, _resolve  # noqa: E402
from extract.backbone import load_backbone      # noqa: E402


def to_x(u8, mean, std):
    x = torch.from_numpy(u8).permute(0, 3, 1, 2).float().div(255)
    return (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", required=True)
    ap.add_argument("--arch", default="cifar10_resnet20")
    ap.add_argument("--out")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, src = load_backbone(args.arch, dev)
    te = torchvision.datasets.CIFAR10(_resolve(args.volume, "cifar10"), train=False)
    y = np.asarray(te.targets)
    out, correct, preds = {"arch": args.arch, "weights": src, "n": int(len(y))}, {}, {}
    with torch.no_grad():
        for name, (mean, std) in NORMS.items():
            p = torch.cat([model(to_x(te.data[i:i + 1000], mean, std).to(dev)).argmax(1).cpu()
                           for i in range(0, len(te.data), 1000)]).numpy()
            preds[name], correct[name] = p, p == y
            out[name] = {"std": list(std), "acc_10k": float(correct[name].mean()),
                         "acc_first5k": float(correct[name][:5000].mean())}
    a, b = correct["chenyaofo"], correct["cifar_true"]
    only_chen, only_true = int((a & ~b).sum()), int((~a & b).sum())
    n_disc = only_chen + only_true
    out["argmax_agree_10k"] = float((preds["chenyaofo"] == preds["cifar_true"]).mean())
    out["delta_acc_pt"] = 100 * (out["chenyaofo"]["acc_10k"] - out["cifar_true"]["acc_10k"])
    out["mcnemar"] = {"only_chenyaofo_correct": only_chen, "only_cifar_true_correct": only_true,
                      "p_two_sided": float(binomtest(only_chen, n_disc, 0.5).pvalue) if n_disc else 1.0}
    out["favours"] = ("chenyaofo" if out["delta_acc_pt"] > 0 and out["mcnemar"]["p_two_sided"] < 0.05 else
                      "cifar_true" if out["delta_acc_pt"] < 0 and out["mcnemar"]["p_two_sided"] < 0.05 else
                      "neither (not significant)")
    txt = json.dumps(out, indent=1)
    print(txt)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(txt)


if __name__ == "__main__":
    main()
