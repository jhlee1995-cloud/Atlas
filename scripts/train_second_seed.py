"""
train_second_seed.py -- train cifar10_resnet20 (chenyaofo architecture) from scratch with a
new seed, so the atlas has an independent instance to test stability against.

Recipe (standard CIFAR): SGD momentum 0.9, lr 0.1, wd 5e-4, batch 128, cosine schedule,
random crop + flip. 160 epochs reaches ~92% on a 4090 in well under an hour. The hub
checkpoint was trained with a similar recipe; small accuracy differences are fine, the
critic compares invariants, not accuracy.

Usage (repo root, pod):
  python scripts/train_second_seed.py --volume /workspace --seed 1 \
      --out /workspace/models/resnet20_seed1.pt [--epochs 160] [--arch cifar10_resnet20]
"""
import argparse
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


def _resolve(volume, name):
    try:
        from extract.populate_data import resolve_path
        return resolve_path(volume, name)
    except Exception:
        return os.path.join(volume, "datasets", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="cifar10_resnet20")
    ap.add_argument("--epochs", type=int, default=160)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    norm = T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    tf_train = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), norm])
    tf_test = T.Compose([T.ToTensor(), norm])
    tr = torchvision.datasets.CIFAR10(_resolve(args.volume, "cifar10_train"), train=True,
                                      download=args.download, transform=tf_train)
    te = torchvision.datasets.CIFAR10(_resolve(args.volume, "cifar10"), train=False,
                                      download=args.download, transform=tf_test)
    tl = torch.utils.data.DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                                     pin_memory=True, drop_last=True)
    vl = torch.utils.data.DataLoader(te, batch_size=512, shuffle=False, num_workers=args.workers)

    model = torch.hub.load("chenyaofo/pytorch-cifar-models", args.arch, pretrained=False).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.CrossEntropyLoss()
    t0 = time.time()
    best = 0.0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    for ep in range(args.epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(x), y)
            loss.backward()
            opt.step()
        sched.step()
        if ep % 10 == 9 or ep == args.epochs - 1:
            model.eval()
            correct = n = 0
            with torch.no_grad():
                for x, y in vl:
                    p = model(x.to(device)).argmax(1).cpu()
                    correct += (p == y).sum().item()
                    n += len(y)
            acc = correct / n
            best = max(best, acc)
            print(f"epoch {ep + 1:4d}  loss {loss.item():.3f}  test acc {acc:.4f}  ({time.time() - t0:.0f}s)")
    torch.save({"state_dict": model.state_dict(), "arch": args.arch, "seed": args.seed,
                "epochs": args.epochs, "test_acc": acc, "recipe": vars(args)}, args.out)
    print(f"saved {args.out}  final acc {acc:.4f} (best {best:.4f})")


if __name__ == "__main__":
    main()
