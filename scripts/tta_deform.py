"""
tta_deform.py -- deformation ladder, rung 1 (TENT). GPU. Adapts the backbone on a
CIFAR-10-C stream and re-dumps the whole atlas input at checkpoints, so Stage B can measure
how the map deforms as a function of adaptation dose.

What is recorded at every checkpoint step k (results/<exp>/dump_step<k>/):
  the full dump (ref, panel, test, the stream corruption + a held-out corruption, paired)
  measured in `dump_mode`:
    affine    eval mode, ORIGINAL running stats, ADAPTED BN affine params  (isolates the weight
              deformation; the map-moves regime)                                     [default]
    deployed  train-mode BN on batches of the stream size (what TENT actually serves;
              mixes in input-dependent batch statistics)
  ground truth (labels used offline, never by the adapter), in meta["ground_truth"]:
    acc_stream_heldout   deployed accuracy on unseen batches of the stream corruption
    acc_clean            deployed accuracy on clean test batches
    acc_heldout_corr     deployed accuracy on the held-out corruption (transfer / harm)
    acc_affine_*         the same three in affine mode (what the dump measures)
    pred_entropy         entropy of the predicted-class histogram on stream batches
                         (the trivial collapse monitor the atlas must beat: bar = "atlas
                         deformation moves BEFORE pred_entropy collapses")

Then:  python -m atlas.ladder --exp results/<exp>      (builds each checkpoint atlas, compares
       every step to step 0 with --same-space, writes LADDER.md + ladder.json + plot)

Usage:
  python scripts/tta_deform.py --manifest experiments/queue/tta_tent_resnet20_fog3.yaml --volume /workspace
  python scripts/tta_deform.py --manifest experiments/queue/tta_tent_resnet20_collapse.yaml --volume /workspace

TENT (Wang et al. 2021): entropy minimization, only BN affine (gamma, beta) trainable, BN
uses test-batch statistics. Here running stats are frozen (momentum 0) so the affine view
can be dumped against the original normalization.
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from atlas.config import load_manifest, dump_manifest              # noqa: E402
from atlas.extract_acts import extract, load_model, cifar_transform, load_split_arrays, _resolve  # noqa: E402

TTA_DEFAULTS = {
    "method": "tent", "corruption": "fog", "severity": 3, "order": "block",
    "steps": 300, "batch": 128, "lr": 1e-3, "optimizer": "adam",
    "checkpoints": [0, 10, 25, 50, 100, 200, 300],
    "heldout_corruption": "contrast", "dump_mode": "affine",
    "stream_seed": 0, "n_eval_batches": 16,
}


# ---------------------------------------------------------------------------
# TENT
# ---------------------------------------------------------------------------
def configure_tent(model):
    """Freeze everything but BN affine; BN in train mode with batch stats; running stats frozen."""
    import torch.nn as nn
    model.train()
    params = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.0                # running stats stay at their original values
            m.weight.requires_grad_(True)
            m.bias.requires_grad_(True)
            params += [m.weight, m.bias]
        else:
            for p in m.parameters(recurse=False):
                p.requires_grad_(False)
    return params


def entropy_loss(logits):
    import torch
    p = logits.softmax(1)
    return -(p * logits.log_softmax(1)).sum(1).mean()


def set_bn_mode(model, train):
    import torch.nn as nn
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train(train)


# ---------------------------------------------------------------------------
# streams and evaluation
# ---------------------------------------------------------------------------
def to_tensor(imgs, device):
    import torch
    from PIL import Image
    tf = cifar_transform()
    return torch.stack([tf(Image.fromarray(im)) for im in imgs]).to(device)


def batches(imgs, labels, batch, rng, n=None):
    """Random batches (shuffle order) of (x_uint8, y). With n, reshuffles and cycles over the pool
    until n batches exist (a pool smaller than n*batch is revisited); n=None is one pass."""
    if len(imgs) < batch:
        raise SystemExit(f"[tta] pool of {len(imgs)} < batch {batch}")
    if n is not None and n <= 0:
        return []
    out = []
    while True:
        idx = rng.permutation(len(imgs))
        for i in range(0, len(idx) - batch + 1, batch):
            b = idx[i:i + batch]
            out.append((imgs[b], labels[b]))
            if n is not None and len(out) >= n:
                return out
        if n is None:
            return out


def evaluate(model, sets, device, batch, deployed):
    """sets: {name: (imgs, labels)} -> {name: acc}; deployed=True uses train-mode BN on batches."""
    import torch
    set_bn_mode(model, deployed)
    out = {}
    with torch.no_grad():
        for name, (imgs, labs) in sets.items():
            correct = n = 0
            hist = np.zeros(10)
            for i in range(0, len(imgs) - batch + 1, batch):
                x = to_tensor(imgs[i:i + batch], device)
                p = model(x).argmax(1).cpu().numpy()
                correct += (p == labs[i:i + batch]).sum()
                n += batch
                hist += np.bincount(p, minlength=10)
            out[name] = float(correct / max(1, n))
            if name == "stream_heldout":
                q = hist / max(1, hist.sum())
                out["pred_entropy"] = float(-(q[q > 0] * np.log(q[q > 0])).sum() / np.log(10))
    set_bn_mode(model, True)
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--volume", default="/workspace")
    ap.add_argument("--weights")
    ap.add_argument("--device")
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()
    import torch
    from extract.data_loaders import CIFAR10C

    cfg = load_manifest(args.manifest)
    tta = {**TTA_DEFAULTS, **(cfg.get("tta") or {})}
    root = cfg["outputs"]["root"]
    os.makedirs(root, exist_ok=True)
    dump_manifest(cfg, os.path.join(root, "manifest_used.yaml"))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(int(tta["stream_seed"]))

    # --- data: stream = CIFAR-10-C rows AFTER the paired region (which the dump uses) ---
    c10c_root = _resolve(args.volume, "cifar10c")
    n_pair = int(cfg["data"]["cifar10c"]["n_per_set"])
    ds = CIFAR10C(c10c_root, tta["corruption"], int(tta["severity"]))
    stream_imgs, stream_labs = ds.data[n_pair:], np.asarray(ds.labels[n_pair:])
    n_held = int(tta["n_eval_batches"]) * int(tta["batch"])
    held_imgs, held_labs = stream_imgs[:n_held], stream_labs[:n_held]        # never adapted on
    adapt_imgs, adapt_labs = stream_imgs[n_held:], stream_labs[n_held:]
    te_img, te_lab = load_split_arrays(args.volume, "cifar10", cfg, args.download)
    clean_eval = (te_img[n_pair:n_pair + n_held], te_lab[n_pair:n_pair + n_held])
    hds = CIFAR10C(c10c_root, tta["heldout_corruption"], int(tta["severity"]))
    heldout_corr = (hds.data[n_pair:n_pair + n_held], np.asarray(hds.labels[n_pair:n_pair + n_held]))
    eval_sets = {"stream_heldout": (held_imgs, held_labs), "clean": clean_eval, "heldout_corr": heldout_corr}

    # make sure the dump covers the stream corruption and the held-out one (paired region)
    cfg["data"]["cifar10c"]["corruptions"] = sorted({tta["corruption"], tta["heldout_corruption"]})
    cfg["data"]["cifar10c"]["severities"] = [int(tta["severity"])]

    # --- model ---
    model, src = load_model(cfg, args.weights, device)
    params = configure_tent(model)
    opt = (torch.optim.Adam(params, lr=float(tta["lr"]), betas=(0.9, 0.999)) if tta["optimizer"] == "adam"
           else torch.optim.SGD(params, lr=float(tta["lr"]), momentum=0.9))
    print(f"[tta] {tta['method']} on {tta['corruption']} s{tta['severity']} order={tta['order']} "
          f"steps={tta['steps']} batch={tta['batch']} lr={tta['lr']} dump_mode={tta['dump_mode']}")

    # stream plan: 'block' = corruption only; 'shuffle' = corrupt/clean batches mixed 50/50
    if int(tta["steps"]) < 0:
        raise SystemExit(f"[tta] steps must be >= 0, got {tta['steps']}")
    plan = batches(adapt_imgs, adapt_labs, int(tta["batch"]), rng, n=int(tta["steps"]))
    if tta["order"] == "shuffle":
        cl = batches(te_img[n_pair + n_held:], te_lab[n_pair + n_held:], int(tta["batch"]), rng, n=int(tta["steps"]))
        plan = [plan[i] if i % 2 == 0 else cl[i] for i in range(min(len(plan), len(cl)))]
    checkpoints = sorted(set(int(c) for c in tta["checkpoints"] if int(c) <= len(plan)))
    want = sorted(set(int(c) for c in tta["checkpoints"] if int(c) <= int(tta["steps"])))
    if len(plan) < int(tta["steps"]) or checkpoints != want:
        raise SystemExit(f"[tta] plan has {len(plan)} steps < {tta['steps']}; checkpoints {want} -> {checkpoints}")
    tta["adapt_pool"] = int(len(adapt_imgs))
    n_corrupt = len(plan) if tta["order"] != "shuffle" else (len(plan) + 1) // 2
    tta["adapt_epochs"] = round(n_corrupt * int(tta["batch"]) / len(adapt_imgs), 2)   # > 1: pool revisited
    print(f"[tta] adapt pool {tta['adapt_pool']} images, {tta['adapt_epochs']} passes over it")
    log = []
    t0 = time.time()

    def checkpoint(step):
        gt_dep = evaluate(model, eval_sets, device, int(tta["batch"]), deployed=True)
        gt_aff = evaluate(model, eval_sets, device, int(tta["batch"]), deployed=False)
        gt = {"acc_stream_heldout": gt_dep["stream_heldout"], "acc_clean": gt_dep["clean"],
              "acc_heldout_corr": gt_dep["heldout_corr"], "pred_entropy": gt_dep["pred_entropy"],
              "acc_affine_stream_heldout": gt_aff["stream_heldout"], "acc_affine_clean": gt_aff["clean"],
              "acc_affine_heldout_corr": gt_aff["heldout_corr"]}
        # dump in the requested view
        set_bn_mode(model, tta["dump_mode"] == "deployed")
        if tta["dump_mode"] == "deployed":
            cfg["extract"]["batch"] = int(tta["batch"])
        d = os.path.join(root, f"dump_step{step}")
        extract(cfg, args.volume, None, device, args.download, d, model=model,
                model_src=f"{src} + {tta['method']} step {step}",
                extra_meta={"tta": tta, "step": step, "ground_truth": gt, "dump_mode": tta["dump_mode"]},
                quiet=True)
        set_bn_mode(model, True)
        entry = {"step": step, **gt, "wall_s": round(time.time() - t0, 1)}
        log.append(entry)
        print(f"[tta] step {step:4d}  stream {gt['acc_stream_heldout']:.3f}  clean {gt['acc_clean']:.3f}  "
              f"heldout {gt['acc_heldout_corr']:.3f}  pred_entropy {gt['pred_entropy']:.3f}  ({entry['wall_s']}s)")
        json.dump(log, open(os.path.join(root, "ladder_log.json"), "w"), indent=1)

    if 0 in checkpoints:
        checkpoint(0)
    for step, (xb, _) in enumerate(plan, start=1):
        x = to_tensor(xb, device)
        loss = entropy_loss(model(x))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step in checkpoints:
            checkpoint(step)
    print(f"[tta] done; {len(log)} checkpoints under {root}/dump_step*  ->  python -m atlas.ladder --exp {root}")


if __name__ == "__main__":
    main()
