"""
train_second_seed.py -- train cifar10_resnet20 (chenyaofo architecture) from scratch with a
new seed, so the atlas has independent instances to test stability against.

Default recipe = the hub checkpoint's own training log (chenyaofo/pytorch-cifar-models, branch
`logs`, logs/cifar10/resnet20/default.log; image-classification-codebase conf/cifar10.conf):
SGD nesterov momentum 0.9, lr 0.1, wd 5e-4, batch 256 (partial last batch kept), cosine T_max =
epochs, 200 epochs, RandomCrop(32, pad 4) + flip. The input normalization is explicit (--norm)
and is saved in the checkpoint; atlas.extract_acts.load_model refuses a manifest whose
backbone.norm differs. The LAST epoch is saved (no test-set checkpoint selection).

--epochs 0 builds the random-init null control: weights at init, BN running stats re-estimated
on --bn-recal clean-train batches.

Usage (repo root, pod):
  python scripts/train_second_seed.py --volume /workspace --seed 1 --norm chenyaofo \
      --out /workspace/models/resnet20_s1_chenyaofo.pt --train-json results/train_resnet20_s1/train.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from atlas.extract_acts import NORMS  # noqa: E402


def _resolve(volume, name):
    try:
        from extract.populate_data import resolve_path
        return resolve_path(volume, name)
    except Exception:
        return os.path.join(volume, "datasets", name)


def evaluate(model, loader, device):
    model.eval()
    correct = n = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
            n += len(y)
    return correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-json", help="also write recipe + accuracy history here (e.g. under results/)")
    ap.add_argument("--arch", default="cifar10_resnet20")
    ap.add_argument("--norm", required=True, choices=sorted(NORMS))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--drop-last", action="store_true")
    ap.add_argument("--bn-recal", type=int, default=100, help="--epochs 0 only: BN recalibration batches")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    norm = T.Normalize(*NORMS[args.norm])
    tf_train = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), norm])
    tf_test = T.Compose([T.ToTensor(), norm])
    tr = torchvision.datasets.CIFAR10(_resolve(args.volume, "cifar10_train"), train=True,
                                      download=args.download, transform=tf_train)
    te = torchvision.datasets.CIFAR10(_resolve(args.volume, "cifar10"), train=False,
                                      download=args.download, transform=tf_test)
    persist = args.workers > 0
    tl = torch.utils.data.DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                                     pin_memory=True, drop_last=args.drop_last, persistent_workers=persist)
    vl = torch.utils.data.DataLoader(te, batch_size=512, shuffle=False, num_workers=args.workers,
                                     persistent_workers=persist)

    model = torch.hub.load("chenyaofo/pytorch-cifar-models", args.arch, pretrained=False).to(device)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t0 = time.time()
    history = []
    if args.epochs == 0:      # null control: random init, BN running stats from clean-train batches
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()
                m.momentum = None                     # cumulative average over the recal batches
        model.train()
        with torch.no_grad():
            for i, (x, _) in enumerate(tl):
                if i >= args.bn_recal:
                    break
                model(x.to(device, non_blocking=True))
    else:
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd, nesterov=True)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        lossf = nn.CrossEntropyLoss()
        for ep in range(args.epochs):
            model.train()
            for x, y in tl:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                loss = lossf(model(x), y)
                loss.backward()
                opt.step()
            sched.step()
            if ep % 10 == 9 and ep != args.epochs - 1:
                acc = evaluate(model, vl, device)
                history.append({"epoch": ep + 1, "loss": round(loss.item(), 4), "test_acc_10k": acc,
                                "elapsed_s": round(time.time() - t0, 1)})
                print(f"epoch {ep + 1:4d}  loss {loss.item():.3f}  test acc {acc:.4f}  ({time.time() - t0:.0f}s)",
                      flush=True)
    acc = evaluate(model, vl, device)                  # last epoch; no test-set selection
    wall = round(time.time() - t0, 1)
    tmp = args.out + ".tmp"
    torch.save({"state_dict": model.state_dict(), "arch": args.arch, "seed": args.seed, "epochs": args.epochs,
                "norm": args.norm, "test_acc": acc, "recipe": vars(args)}, tmp)
    os.replace(tmp, args.out)                          # atomic: a crash never leaves a truncated checkpoint
    print(f"saved {args.out}  final 10k acc {acc:.4f}  norm {args.norm}  ({wall:.0f}s)", flush=True)
    if args.train_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.train_json)), exist_ok=True)
        info = {"out": args.out, "recipe": vars(args), "final_test_acc_10k": acc, "wall_s": wall,
                "history": history, "torch": torch.__version__,
                "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"}
        with open(args.train_json, "w") as f:
            json.dump(info, f, indent=1)


if __name__ == "__main__":
    main()
