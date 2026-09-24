#!/usr/bin/env python3
"""
b4_weights.py -- batch-4 weights: full sha256, hub byte-size check, README accuracy gate and head export (docs/plans/
B4_INTEGRATION.md D2, D9 step 5, D18). Owner: T1 (weights and head). scripts/b4_extract.py imports load_model,
last_linear, export_head and to_input from here, so the extractor and this gate load every network the same way.

  python scripts/b4_weights.py --registry experiments/b4/models.json --volume /workspace \
      --out results/b4_weights/weights.json                     (S1, before any extraction; GPU if present)
  python scripts/b4_weights.py --registry ... --volume ... --trained --out results/b4_weights/weights_trained.json
                                                                (S1, after the trainings: sha256 of the new checkpoints)
Per unit (registry order; --units narrows):
  hub   torch.hub at the PINNED ref; the cached file's byte size must equal the GitHub release asset (registry
        hub_asset.bytes; fatal -> FAIL); full sha256 recorded; its first 8 hex must equal the file-name tag (else
        NOT_EVALUABLE, T2 review #4 / T3's finding); full-10k CIFAR-10 test accuracy vs the README top-1: |delta| <= 0.003
        (fatal -> FAIL; the README number is public, so measuring it reveals nothing).
  file  full sha256 of the volume checkpoint; its recorded norm must equal the registry norm (fatal); for an old net the
        16-hex prefix its committed atlas recorded (meta.weights 'sha256:<16>') must match (else FAIL: not the anchor's
        file). A checkpoint that is not there yet (F / F20 / K before training) is ABSENT (the --trained pass records it).
Every loaded unit's final nn.Linear is exported to results/b4_weights/heads/<id>/head.npz (W, b float32); heads of width
<= 64 also go into the JSON (9 significant digits), so the CIFAR ResNet heads survive the volume.
Exit code: 1 when an ANCHOR unit is not PASS (hard, D15); other units fail per unit (the extractor refuses a FAIL unit).

INTERFACE (imported by scripts/b4_extract.py)
  HUB_REPO, HUB_REF;  sha256_file(path) -> hex;  to_input(u8 (N, 32, 32, 3), norm) -> float32 CPU tensor (bitwise the
  PIL ToTensor + Normalize path);  hub_load(arch, pretrained) -> module;  load_model(spec, device='cpu') -> (model in
  eval mode on device, weights record with 'status');  last_linear(model) -> (module name, W float64 (K, d), b (K,));
  export_head(model, out_dir, json_max_dim=64) -> record (writes out_dir/head.npz);  test_accuracy(model, volume, norm,
  device) -> float (all 10,000 CIFAR-10 test rows)
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

SCHEMA = "b4_weights/1"
HUB_REPO = "chenyaofo/pytorch-cifar-models"
HUB_REF = "786c16252c0fc58ee9adac063f8337cc4a7a497a"     # master on 2026-09-23 (GitHub API); pinned for every batch-4 load
README_TOL = 0.003                                        # D2 / D15: |full-10k accuracy - README top-1| <= 0.003


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def to_input(u8, norm):
    """uint8 (N, H, W, 3) -> normalised float32 (N, 3, H, W) on the CPU: x / 255 then (x - mean) / std, the same float32
    operations as torchvision ToTensor + Normalize on PIL images (tests/test_b4_extract.py checks bitwise equality)."""
    import torch
    from atlas.extract_acts import NORMS
    mean, std = NORMS[norm]
    x = torch.from_numpy(np.ascontiguousarray(u8)).permute(0, 3, 1, 2).contiguous().to(torch.float32).div(255)
    m = torch.as_tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    s = torch.as_tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    return (x - m) / s


def hub_load(arch, pretrained):
    import torch
    # HUB_REF was read from the canonical repo, so the fork validation (one unauthenticated GitHub API call) is skipped
    return torch.hub.load(f"{HUB_REPO}:{HUB_REF}", arch, pretrained=pretrained, trust_repo=True,
                          skip_validation=True).eval()


def hub_checkpoint(asset):
    import torch
    return os.path.join(torch.hub.get_dir(), "checkpoints", asset)


def tiny_resnet(seed=0):
    """A 4/8-channel CIFAR-shaped ResNet with the chenyaofo module names (conv1, bn1, relu, layer1-3, avgpool, fc) for the
    extractor's synthetic self-test and tests: BlockHooks taps stem, layer1.0, layer2.0, layer3.0, layer3.1, penult;
    layer3.1's GAP is the penult (a duplicate), so the D2 pre tap is layer3.0. Deterministic in seed; eval mode."""
    import torch
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self, cin, cout, stride):
            super().__init__()
            self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(cout)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(cout)
            self.downsample = None if (stride == 1 and cin == cout) else nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

        def forward(self, x):
            o = self.relu(self.bn1(self.conv1(x)))
            o = self.bn2(self.conv2(o))
            return self.relu(o + (x if self.downsample is None else self.downsample(x)))

    class TinyResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 4, 3, 1, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(4)
            self.relu = nn.ReLU(inplace=True)
            self.layer1 = nn.Sequential(Block(4, 4, 1))
            self.layer2 = nn.Sequential(Block(4, 8, 2))
            self.layer3 = nn.Sequential(Block(8, 8, 2), Block(8, 8, 1))
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(8, 10)

        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.layer3(self.layer2(self.layer1(x)))
            return self.fc(torch.flatten(self.avgpool(x), 1))

    g = torch.random.get_rng_state()
    torch.manual_seed(int(seed))
    m = TinyResNet()
    torch.random.set_rng_state(g)
    return m.eval()


def _committed_prefix(anchor):
    """The 16-hex weight prefix a committed atlas recorded ('file:<path> sha256:<16>'), or None (hub anchors)."""
    if not anchor:
        return None
    p = os.path.join(REPO_ROOT, anchor, "atlas.json")
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        w = str((json.load(f).get("meta") or {}).get("weights") or "")
    return w.split("sha256:", 1)[1].strip()[:16] if "sha256:" in w else None


def load_model(spec, device="cpu"):
    """(model in eval mode on device, weights record). Raises SystemExit on a fatal mismatch (hub byte size, checkpoint
    norm). The record's 'status' is PASS, NOT_EVALUABLE (hub tag is not the sha256 prefix) or FAIL (committed prefix)."""
    import torch
    src = spec["source"]
    if src == "hub":
        model = hub_load(spec["arch"], True)
        a = spec["hub_asset"]
        path = hub_checkpoint(a["file"])
        if not os.path.isfile(path):
            raise SystemExit(f"[b4_weights] {spec['id']}: {path} missing after the hub load")
        size = os.path.getsize(path)
        rec = {"source": "hub", "hub_ref": HUB_REF, "file": path, "asset": a["file"], "bytes": size,
               "bytes_expected": a["bytes"], "tag": a["tag"]}
        if size != a["bytes"]:
            raise SystemExit(f"[b4_weights] {spec['id']}: {path} is {size} bytes, release asset {a['bytes']} (FAIL)")
        rec["sha256"] = sha256_file(path)
        rec["tag_prefix_ok"] = rec["sha256"].startswith(a["tag"])
        rec["status"] = "PASS" if rec["tag_prefix_ok"] else "NOT_EVALUABLE"
    elif src == "file":
        path = spec["weights"]
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        model = hub_load(spec["arch"], False)
        ck = torch.load(path, map_location="cpu")
        ck_norm = ck.get("norm") if isinstance(ck, dict) else None
        if ck_norm != spec["norm"]:
            raise SystemExit(f"[b4_weights] {path}: checkpoint norm {ck_norm} != registry norm {spec['norm']} (FAIL)")
        model.load_state_dict(ck.get("state_dict", ck) if isinstance(ck, dict) else ck)
        rec = {"source": "file", "file": path, "bytes": os.path.getsize(path), "sha256": sha256_file(path),
               "norm": ck_norm, "epochs": ck.get("epochs"), "seed": ck.get("seed"), "arch": ck.get("arch")}
        want = _committed_prefix(spec.get("anchor"))
        rec["committed_prefix"] = want
        rec["committed_prefix_ok"] = None if want is None else rec["sha256"].startswith(want)
        rec["status"] = "FAIL" if rec["committed_prefix_ok"] is False else "PASS"
    elif src == "synthetic":                                          # the extractor's self-test network
        model = tiny_resnet(int(spec.get("seed", 0)))
        rec = {"source": "synthetic", "file": None, "sha256": None, "status": "PASS"}
    else:
        raise SystemExit(f"[b4_weights] unknown source {src}")
    model = model.eval().to(device)
    rec["n_params"] = int(sum(p.numel() for p in model.parameters()))
    return model, rec


def last_linear(model):
    """The final nn.Linear in named_modules order (the module whose INPUT is the penult tap)."""
    import torch.nn as nn
    name, fc = None, None
    for n, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            name, fc = n, mod
    if fc is None:
        raise SystemExit("[b4_weights] no nn.Linear head")
    b = fc.bias.detach().double().cpu().numpy() if fc.bias is not None else np.zeros(fc.out_features)
    return name, fc.weight.detach().double().cpu().numpy(), b


def _arr_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def export_head(model, out_dir, json_max_dim=64):
    """Write out_dir/head.npz (W (K, d), b (K,) float32) and return the head record (hashes, SVD, rank)."""
    name, W, b = last_linear(model)
    os.makedirs(out_dir, exist_ok=True)
    W32, b32 = W.astype(np.float32), b.astype(np.float32)
    np.savez(os.path.join(out_dir, "head.npz"), W=W32, b=b32)
    s = np.linalg.svd(W, compute_uv=False)
    rank = int((s > s[0] * 1e-8).sum()) if len(s) and s[0] > 0 else 0
    rec = {"module": name, "shape": list(W.shape), "W_sha256_f32": _arr_sha(W32), "b_sha256_f32": _arr_sha(b32),
           "svd": {"singular_values": [float(v) for v in s], "rank": rank, "null_dim": int(W.shape[1] - rank)}}
    if W.shape[1] <= json_max_dim:
        rec["W"] = [[float(f"{v:.9g}") for v in row] for row in W32.tolist()]
        rec["b"] = [float(f"{v:.9g}") for v in b32.tolist()]
    return rec


def test_accuracy(model, volume, norm, device, batch=500):
    """Accuracy on all 10,000 CIFAR-10 test rows (aggregate only; the README gate)."""
    import torch
    from atlas.extract_acts import load_split_arrays
    imgs, labs = load_split_arrays(volume, "cifar10", None)
    correct = 0
    with torch.no_grad():
        for i in range(0, len(imgs), batch):
            z = model(to_input(imgs[i:i + batch], norm).to(device))
            correct += int((z.argmax(1).cpu().numpy() == np.asarray(labs[i:i + batch])).sum())
    return correct / len(imgs)


def gate_unit(spec, volume, device, heads_root):
    t0 = time.time()
    try:
        model, rec = load_model(spec, device)
    except FileNotFoundError:
        if spec["source"] == "file":
            return {"status": "ABSENT", "why": f"{spec.get('weights')} not on the volume (trained later in S1)"}
        raise
    except SystemExit as e:
        return {"status": "FAIL", "why": str(e)}
    except Exception as e:                                      # a hub download error fails this unit, not the session
        return {"status": "FAIL", "why": f"{type(e).__name__}: {e}"}
    rec["head"] = export_head(model, os.path.join(heads_root, spec["id"]))
    if spec["source"] == "hub" and spec.get("readme_top1") is not None:
        acc = test_accuracy(model, volume, spec["norm"], device)
        want = float(spec["readme_top1"]) / 100.0
        rec["readme"] = {"acc_10k": acc, "readme_top1": want, "delta": acc - want, "tol": README_TOL,
                         "status": "PASS" if abs(acc - want) <= README_TOL else "FAIL"}
        if rec["readme"]["status"] == "FAIL":
            rec["status"] = "FAIL"
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


def trained_unit(spec):
    path = spec.get("weights")
    if not path or not os.path.isfile(path):
        return {"status": "ABSENT", "file": path}
    rec = {"status": "PASS", "file": path, "bytes": os.path.getsize(path), "sha256": sha256_file(path)}
    tj = os.path.join(REPO_ROOT, "results", f"train_{spec['id']}", "train.json")
    if os.path.isfile(tj):
        with open(tj) as f:
            t = json.load(f)
        rec["train_json"] = {"path": os.path.relpath(tj, REPO_ROOT).replace(os.sep, "/"), "wall_s": t.get("wall_s"),
                             "recipe": t.get("recipe"), "device": t.get("device"), "torch": t.get("torch")}
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--registry", required=True)
    ap.add_argument("--volume", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--units", default="", help="comma list (default: every hub / file unit of the registry)")
    ap.add_argument("--trained", action="store_true", help="sha256 of the checkpoints trained in this session only")
    ap.add_argument("--device")
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        print(f"[b4_weights] {a.out} exists: never overwritten (relaunch with ATLAS_B4_TAG)", file=sys.stderr)
        return 2
    with open(a.registry) as f:
        reg = json.load(f)
    want = {u for u in a.units.split(",") if u}
    units = [m for m in reg["models"] if (not want or m["id"] in want) and m["source"] in ("hub", "file")]
    out = {"schema": SCHEMA, "hub_ref": HUB_REF, "registry": a.registry, "trained_only": a.trained,
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "units": {}}
    if a.trained:
        for m in units:
            if m.get("train"):
                out["units"][m["id"]] = trained_unit(m)
                print(f"[b4_weights] {m['id']}: {out['units'][m['id']]['status']}", flush=True)
    else:
        import torch
        torch.set_grad_enabled(False)
        device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
        out.update({"torch": torch.__version__, "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"})
        heads = os.path.join(os.path.dirname(os.path.abspath(a.out)), "heads")
        for m in units:
            rec = gate_unit(m, a.volume, device, heads)
            out["units"][m["id"]] = rec
            print(f"[b4_weights] {m['id']:22s} {rec['status']}"
                  + (f"  (readme gate {rec['readme']['status']})" if "readme" in rec else ""), flush=True)
    out["counts"] = {s: sum(1 for r in out["units"].values() if r["status"] == s)
                     for s in ("PASS", "NOT_EVALUABLE", "FAIL", "ABSENT")}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1, allow_nan=False)
    os.replace(tmp, a.out)
    bad_anchor = [m["id"] for m in units if "ANCHOR" in m["roles"] and out["units"].get(m["id"], {}).get("status") != "PASS"
                  and not a.trained]
    print(f"[b4_weights] {out['counts']} -> {a.out}" + (f"; ANCHOR FAIL: {bad_anchor}" if bad_anchor else ""))
    return 1 if bad_anchor else 0


if __name__ == "__main__":
    sys.exit(main())
