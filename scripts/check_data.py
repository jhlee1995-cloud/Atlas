"""
check_data.py -- hard gate: exit non-zero unless every dataset the queued manifests read is on
<volume>/datasets, resolved the same way the loaders resolve it, and loadable.

  python scripts/check_data.py /workspace              # everything
  python scripts/check_data.py /workspace cifar10c     # one group: torchvision | cifar10c

`populate_data --verify` is not enough: it always exits 0 and its recursive file count passes a
nested CIFAR-10-C layout (cifar10c/CIFAR-10-C/*.npy) that extract.data_loaders.CIFAR10C cannot read.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from atlas.extract_acts import _resolve                  # noqa: E402  same lookup as Stage A
from extract.data_loaders import CIFAR10C_CORRUPTIONS    # noqa: E402


def check_torchvision(volume):
    """torchvision md5-checks every archive member on construction when download=False."""
    import torchvision
    specs = {
        "cifar10": lambda d: torchvision.datasets.CIFAR10(d, train=False, download=False),
        "cifar10_train": lambda d: torchvision.datasets.CIFAR10(d, train=True, download=False),
        "cifar100": lambda d: torchvision.datasets.CIFAR100(d, train=False, download=False),
        "svhn": lambda d: torchvision.datasets.SVHN(d, split="test", download=False),
    }
    bad = []
    for name, ctor in specs.items():
        root = _resolve(volume, name)
        try:
            n = len(ctor(root))
            print(f"[data] {name:14s} ok  {n} images  ({root})")
        except Exception as e:
            bad.append(f"{name} at {root}: {repr(e)[:160]}")
    return bad


def check_cifar10c(volume):
    root = _resolve(volume, "cifar10c")
    nested = os.path.join(root, "CIFAR-10-C")
    if os.path.isdir(nested) and any(f.endswith(".npy") for f in os.listdir(nested)):
        return [f"cifar10c is nested at {nested}; move the .npy files up one level and remove that "
                f"directory (or re-extract with tar --strip-components=1)"]
    bad = []
    try:
        lab = np.load(os.path.join(root, "labels.npy"), mmap_mode="r")
        if lab.shape not in ((50000,), (10000,)):
            bad.append(f"cifar10c/labels.npy shape {lab.shape}")
    except Exception as e:
        bad.append(f"cifar10c/labels.npy: {repr(e)[:160]}")
    for c in CIFAR10C_CORRUPTIONS:
        try:
            a = np.load(os.path.join(root, f"{c}.npy"), mmap_mode="r")
            if a.shape != (50000, 32, 32, 3) or a.dtype != np.uint8:
                bad.append(f"cifar10c/{c}.npy shape {a.shape} dtype {a.dtype}")
        except Exception as e:
            bad.append(f"cifar10c/{c}.npy: {repr(e)[:160]}")
    if not bad:
        print(f"[data] cifar10c       ok  {len(CIFAR10C_CORRUPTIONS)} corruptions + labels  ({root})")
    return bad


def main():
    groups = {"torchvision", "cifar10c"}
    volume = sys.argv[1] if len(sys.argv) > 1 else "/workspace"
    only = set(sys.argv[2:]) or groups
    if only - groups:
        sys.exit(f"[data] unknown group(s) {sorted(only - groups)}; valid: {' | '.join(sorted(groups))}")
    bad = ((check_torchvision(volume) if "torchvision" in only else [])
           + (check_cifar10c(volume) if "cifar10c" in only else []))
    if bad:
        print("[data] NOT READY:\n  " + "\n  ".join(bad))
        sys.exit(1)
    print(f"[data] OK ({', '.join(sorted(only))}) under {volume}/datasets")


if __name__ == "__main__":
    main()
