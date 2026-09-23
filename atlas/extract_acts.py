"""
extract_acts.py -- STAGE A (GPU). Writes the activation DUMP that Stage B reads.

Reuses the repo's conventions: extract.backbone.load_backbone (chenyaofo hub), the
extract.data_loaders.CIFAR10C reader, extract.populate_data.resolve_path (mount-agnostic
dataset lookup). Adds: every residual block as a hook point, a fixed reference panel,
paired corrupt sets, and pixel factors computed from the raw uint8 arrays.

Usage (on the pod, from the repo root):
  python -m atlas.extract_acts --manifest experiments/queue/atlas_v0_resnet20_cifar10.yaml \
                               --volume /workspace
  # second seed:
  python -m atlas.extract_acts --manifest experiments/queue/atlas_v0_resnet20_seed1.yaml \
                               --volume /workspace --weights /workspace/models/resnet20_seed1.pt

Pairing contract: clean test = torchvision CIFAR-10 test indices 0..n_test-1 in order;
corrupt__X__sN = CIFAR-10-C rows 0..n_per_set-1 of severity N. Labels are asserted equal.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from .config import load_manifest, dump_manifest          # noqa: E402
from .context import META_NAME, corrupt_split             # noqa: E402
from .factors import compute_pixel_factors                # noqa: E402


# ---------------------------------------------------------------------------
# hooks
# ---------------------------------------------------------------------------
class BlockHooks:
    """Hook stem + residual blocks (layerN.i) + penult (input of the final Linear).
    mode: 'blocks' (every block, with stride) | 'stages' (layerN outputs only).
    pooling: 'gap' -> (B, C) ; 'gap_std' -> (B, 2C) mean and spatial std per channel."""

    def __init__(self, model, mode="blocks", block_stride=1, pooling="gap"):
        import torch.nn as nn
        self.model = model
        self.pooling = pooling
        self.acts = {}
        self.handles = []
        self.layer_names = []
        names = dict(model.named_modules())
        # stem: prefer the post-BN ReLU module, else bn1
        stem = names.get("relu") if isinstance(names.get("relu"), nn.ReLU) else names.get("bn1")
        if stem is not None:
            self._hook(stem, "stem")
        if mode == "stages":
            for n in ("layer1", "layer2", "layer3", "layer4"):
                if n in names:
                    self._hook(names[n], n)
        else:
            blocks = [n for n in names if re.fullmatch(r"layer\d+\.\d+", n)]
            blocks.sort(key=lambda n: (int(n[5:].split(".")[0]), int(n.split(".")[1])))
            # always keep the last block of each stage; stride the rest
            keep = []
            by_stage = {}
            for b in blocks:
                by_stage.setdefault(b.split(".")[0], []).append(b)
            for st, bl in by_stage.items():
                for i, b in enumerate(bl):
                    if i % max(1, block_stride) == 0 or i == len(bl) - 1:
                        keep.append(b)
            for b in keep:
                self._hook(names[b], b)
        final_fc = None
        for _, mod in model.named_modules():
            if isinstance(mod, nn.Linear):
                final_fc = mod
        if final_fc is not None:
            def _penult(_, inp, __):
                self.acts["penult"] = inp[0].detach()
            self.handles.append(final_fc.register_forward_hook(_penult))
            self.layer_names.append("penult")

    def _hook(self, mod, name):
        import torch
        def hook(_, __, out):
            if out.dim() == 4:
                m = out.mean(dim=(2, 3))
                if self.pooling == "gap_std":
                    s = out.std(dim=(2, 3))
                    m = torch.cat([m, s], dim=1)
                self.acts[name] = m.detach()
            else:
                self.acts[name] = out.detach()
        self.handles.append(mod.register_forward_hook(hook))
        self.layer_names.append(name)

    def forward(self, x):
        import torch
        self.acts = {}
        with torch.no_grad():
            logits = self.model(x)
        return {k: v.float() for k, v in self.acts.items()}, logits.float()

    def close(self):
        for h in self.handles:
            h.remove()


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def _resolve(volume, name):
    try:
        from extract.populate_data import resolve_path
        return resolve_path(volume, name)
    except Exception:
        return os.path.join(volume, "datasets", name)


def cifar_transform():
    import torchvision.transforms as T
    return T.Compose([T.ToTensor(),
                      T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])


def load_split_arrays(volume, dataset, cfg, download=False):
    """Return (images uint8 (N,32,32,3), labels (N,)) for a torchvision CIFAR split."""
    import torchvision
    if dataset == "cifar10_train":
        ds = torchvision.datasets.CIFAR10(_resolve(volume, "cifar10_train"), train=True, download=download)
    elif dataset == "cifar10":
        ds = torchvision.datasets.CIFAR10(_resolve(volume, "cifar10"), train=False, download=download)
    elif dataset == "cifar100":
        ds = torchvision.datasets.CIFAR100(_resolve(volume, "cifar100"), train=False, download=download)
    elif dataset == "svhn":
        ds = torchvision.datasets.SVHN(_resolve(volume, "svhn"), split="test", download=download)
        return np.transpose(ds.data, (0, 2, 3, 1)), np.asarray(ds.labels)
    else:
        raise ValueError(f"dataset {dataset} not wired for array loading")
    return np.asarray(ds.data), np.asarray(ds.targets)


def run_backbone(hooks, images_uint8, device, batch=256, transform=None):
    """images (N,H,W,3) uint8 -> {layer: (N,D) float16}, argmax, maxprob."""
    import torch
    from PIL import Image
    tf = transform or cifar_transform()
    feats, am, mp = {}, [], []
    for i in range(0, len(images_uint8), batch):
        blk = images_uint8[i:i + batch]
        x = torch.stack([tf(Image.fromarray(im)) for im in blk]).to(device)
        f, logits = hooks.forward(x)
        p = torch.softmax(logits, 1)
        am.append(p.argmax(1).cpu().numpy())
        mp.append(p.max(1).values.cpu().numpy())
        for k, v in f.items():
            feats.setdefault(k, []).append(v.cpu().numpy().astype(np.float16))
    return {k: np.concatenate(v) for k, v in feats.items()}, np.concatenate(am), np.concatenate(mp)


def write_split(dump, split, feats, labels, argmax, maxprob, images_uint8):
    for layer, arr in feats.items():
        os.makedirs(os.path.join(dump, "acts", layer), exist_ok=True)
        np.save(os.path.join(dump, "acts", layer, f"{split}.npy"), arr)
    os.makedirs(os.path.join(dump, "labels"), exist_ok=True)
    os.makedirs(os.path.join(dump, "preds"), exist_ok=True)
    os.makedirs(os.path.join(dump, "factors"), exist_ok=True)
    np.save(os.path.join(dump, "labels", f"{split}.npy"), np.asarray(labels).astype(np.int64))
    np.savez(os.path.join(dump, "preds", f"{split}.npz"), argmax=argmax, maxprob=maxprob)
    if images_uint8 is not None:
        np.savez(os.path.join(dump, "factors", f"{split}.npz"), **compute_pixel_factors(images_uint8))


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def load_model(cfg, weights=None, device="cuda"):
    """Backbone from the hub or from a .pt file (train_second_seed.py format). Returns (model, src)."""
    import torch
    from extract.backbone import load_backbone
    arch = cfg["backbone"]["arch"]
    weights = weights or cfg["backbone"].get("weights", "hub")
    if weights in (None, "hub"):
        return load_backbone(arch, device)
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", arch, pretrained=False)
    sd = torch.load(weights, map_location="cpu")
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    model.load_state_dict(sd)
    model.eval().to(device)
    return model, f"file:{weights}"


def extract(cfg, volume, weights=None, device=None, download=False, dump=None,
            model=None, model_src=None, extra_meta=None, quiet=False):
    """Write a dump. Pass `model` (already on device, in the mode you want measured) to
    re-dump an adapted model at a checkpoint; otherwise the backbone is loaded from cfg."""
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    arch = cfg["backbone"]["arch"]
    t0 = time.time()

    # --- model ---
    if model is None:
        model, src = load_model(cfg, weights, device)
    else:
        src = model_src or "in-memory model"
    hooks = BlockHooks(model, cfg["hooks"]["layers"], cfg["hooks"]["block_stride"], cfg["hooks"]["pooling"])
    if not quiet:
        print(f"[extract] {arch} from {src}; hooked {len(hooks.layer_names)} layers: {hooks.layer_names}")

    dump = dump or os.path.join(cfg["outputs"]["root"], "dump")
    os.makedirs(dump, exist_ok=True)
    batch = int(cfg["extract"]["batch"])
    splits, dims = [], {}

    def do(split, imgs, labels):
        feats, am, mp = run_backbone(hooks, imgs, device, batch)
        write_split(dump, split, feats, labels, am, mp, imgs)
        for k, v in feats.items():
            dims[k] = int(v.shape[1])
        splits.append(split)
        acc = float((am == labels).mean()) if labels is not None else None
        if not quiet:
            print(f"[extract] {split:34s} n={len(imgs):6d} acc={acc if acc is None else round(acc, 4)}  ({time.time() - t0:.0f}s)")
        return acc

    d = cfg["data"]
    # --- reference: fixed seeded subset of clean train ---
    tr_img, tr_lab = load_split_arrays(volume, d["reference"]["dataset"], cfg, download)
    rng = np.random.default_rng(int(d["reference"]["seed"]))
    ref_idx = np.sort(rng.choice(len(tr_img), size=min(int(d["reference"]["n"]), len(tr_img)), replace=False))
    ref_acc = do("ref", tr_img[ref_idx], tr_lab[ref_idx])
    # --- panel: fixed seeded subset of clean train, disjoint from ref ---
    prng = np.random.default_rng(int(d["panel"]["seed"]))
    pool = np.setdiff1d(np.arange(len(tr_img)), ref_idx)
    panel_idx = np.sort(prng.choice(pool, size=int(d["panel"]["n"]), replace=False))
    do("panel", tr_img[panel_idx], tr_lab[panel_idx])
    # --- clean test: indices 0..n-1 in order (pairing contract) ---
    te_img, te_lab = load_split_arrays(volume, d["clean_test"]["dataset"], cfg, download)
    n_test = min(int(d["clean_test"]["n"]), len(te_img))
    test_acc = do("test", te_img[:n_test], te_lab[:n_test])
    # --- CIFAR-10-C paired sets ---
    c10c = d.get("cifar10c")
    corr_accs = {}
    if c10c:
        from extract.data_loaders import CIFAR10C, CIFAR10C_CORRUPTIONS
        root = _resolve(volume, "cifar10c")
        corrs = CIFAR10C_CORRUPTIONS if c10c["corruptions"] in ("all", None) else list(c10c["corruptions"])
        n_c = min(int(c10c["n_per_set"]), n_test)
        for c in corrs:
            for s in c10c["severities"]:
                ds = CIFAR10C(root, c, int(s))
                imgs, labs = ds.data[:n_c], np.asarray(ds.labels[:n_c])
                if not np.array_equal(labs, te_lab[:n_c]):
                    raise RuntimeError(f"pairing broken: CIFAR-10-C {c} s{s} labels != clean test labels")
                corr_accs[corrupt_split(c, s)] = do(corrupt_split(c, s), imgs, labs)
    # --- OOD sets (unpaired) ---
    for spec in d.get("ood", []) or []:
        name = spec["dataset"]
        if name in ("cifar100", "svhn"):
            im, lab = load_split_arrays(volume, name, cfg, download)
            n = min(int(spec.get("n", 2000)), len(im))
            do(f"ood__{name}", im[:n], lab[:n])
        elif name == "isun":
            from extract.data_loaders import ImageFolderFlat
            from PIL import Image
            ds = ImageFolderFlat(_resolve(volume, "isun"), transform=None, max_images=int(spec.get("n", 2000)))
            im = np.stack([np.asarray(Image.open(p).convert("RGB").resize((32, 32))) for p in ds.paths])
            do("ood__isun", im, np.zeros(len(im), dtype=np.int64))
        else:
            print(f"[extract] ood dataset {name} not wired; skipped")

    meta = {
        "source": "real",
        "arch": arch, "weights": src, "seed_tag": cfg["backbone"].get("seed_tag"),
        "layers": hooks.layer_names, "dims": dims, "splits": splits,
        "n_classes": int(cfg.get("n_classes", 10)),
        "dtype": cfg["extract"]["dtype"], "pooling": cfg["hooks"]["pooling"],
        "pairing": "corrupt splits are paired with test[:n] by row index",
        "ref_indices": ref_idx.tolist(), "panel_indices": panel_idx.tolist(), "n_test": n_test,
        "accuracy": {"ref": ref_acc, "test": test_acc, **corr_accs},
        "git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device, "exp_id": cfg.get("exp_id"),
    }
    if extra_meta:
        meta.update(extra_meta)
    json.dump(meta, open(os.path.join(dump, META_NAME), "w"), indent=2)
    hooks.close()
    if not quiet:
        print(f"[extract] dump complete at {dump} ({time.time() - t0:.0f}s)")
    return dump


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--volume", required=True, help="mount root (e.g. /workspace)")
    ap.add_argument("--weights", help="override backbone weights: 'hub' or a .pt path")
    ap.add_argument("--device")
    ap.add_argument("--download", action="store_true", help="let torchvision download CIFAR if missing")
    ap.add_argument("--dump", help="override dump dir (default <outputs.root>/dump)")
    args = ap.parse_args()
    cfg = load_manifest(args.manifest)
    os.makedirs(cfg["outputs"]["root"], exist_ok=True)
    dump_manifest(cfg, os.path.join(cfg["outputs"]["root"], "manifest_used.yaml"))
    extract(cfg, args.volume, args.weights, args.device, args.download, args.dump)


if __name__ == "__main__":
    main()
