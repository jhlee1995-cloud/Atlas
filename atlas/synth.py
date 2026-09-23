"""
synth.py -- a SYNTHETIC dump with the real dump layout, for CPU development and tests.

Latents per image: class c (K), luminance b, high-frequency content h. Images are built from
those latents (so the pixel factors are real computations on real arrays), and each layer's
activation is a fixed random nonlinear readout of [onehot(c) * s_l, b * a_l, h * g_l] where
s_l grows with depth (class commits late) and a_l, g_l shrink (low-level factors wash out).
Corrupt splits modify the image AND the latents consistently, so paired displacement exists.

meta.source = "synthetic". The critic will never promote findings from this dump; that is
the point (the synthetic-ceiling trap is a recorded methodology rule).

Usage: python -m atlas.synth --out /tmp/synth_dump [--n-ref 3000]
"""
import argparse
import json
import os

import numpy as np

from .context import META_NAME, corrupt_split
from .factors import compute_pixel_factors

LAYERS = [("stem", 16), ("layer1.1", 16), ("layer2.1", 32), ("layer3.1", 64), ("penult", 64)]
CORRUPTIONS = {"brightness": (1, 3, 5), "defocus_blur": (1, 3, 5), "gaussian_noise": (1, 3, 5)}


def _images(latent_b, latent_h, cls, rng, size=32):
    n = len(cls)
    base = rng.random((n, size, size, 3)).astype(np.float32)
    # class-specific low-frequency pattern
    yy, xx = np.mgrid[0:size, 0:size] / size
    pat = np.stack([np.sin(2 * np.pi * (c + 1) * xx / 3 + c) for c in range(10)])[cls]   # (n, H, W)
    img = 0.35 * base + 0.35 * pat[..., None] + latent_b[:, None, None, None]
    # high-frequency content scaled by h: mix in a smoothed vs raw noise field
    hf = rng.random((n, size, size, 1)).astype(np.float32) - 0.5
    img = img + latent_h[:, None, None, None] * hf
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def _corrupt(img, kind, sev, rng):
    x = img.astype(np.float32) / 255.0
    if kind == "brightness":
        x = x + 0.08 * sev
    elif kind == "defocus_blur":
        k = sev  # box blur radius
        pad = np.pad(x, ((0, 0), (k, k), (k, k), (0, 0)), mode="edge")
        acc = np.zeros_like(x)
        for dy in range(-k, k + 1):
            for dx in range(-k, k + 1):
                acc += pad[:, k + dy:k + dy + x.shape[1], k + dx:k + dx + x.shape[2]]
        x = acc / (2 * k + 1) ** 2
    elif kind == "gaussian_noise":
        x = x + rng.normal(0, 0.04 * sev, size=x.shape)
    return np.clip(x * 255, 0, 255).astype(np.uint8)


class Readout:
    """Fixed random nonlinear readout per layer."""

    def __init__(self, rng, K=10):
        self.K = K
        self.W = {}
        for i, (name, d) in enumerate(LAYERS):
            depth = i / (len(LAYERS) - 1)
            s = 0.3 + 2.5 * depth ** 2          # class signal grows late
            a = 2.0 * (1 - depth) + 0.2         # luminance readable early
            g = 1.5 * (1 - depth) + 0.1         # hf readable early
            A = rng.normal(0, 1, size=(d, K + 2)) / np.sqrt(K + 2)
            B = rng.normal(0, 1, size=(d, d)) / np.sqrt(d)
            self.W[name] = (A, B, s, a, g)

    def __call__(self, cls, b, h, rng, noise=None):
        """noise: optional {layer: (n, d)} so paired splits share per-sample noise."""
        out = {}
        oh = np.eye(self.K)[cls]
        noise = noise if noise is not None else {}
        for name, d in LAYERS:
            A, B, s, a, g = self.W[name]
            lat = np.concatenate([oh * s, (b * a)[:, None], (h * g)[:, None]], axis=1)
            eps = noise.get(name)
            if eps is None:
                eps = rng.normal(0, 1, size=(len(cls), d))
                noise[name] = eps
            z = np.tanh(lat @ A.T) + 0.15 * eps[: len(cls)]
            out[name] = np.tanh(z @ B.T) * 3.0
        self.last_noise = noise
        return out


def make_synth_dump(out, n_ref=3000, n_test=1500, n_panel=64, n_corr=600, seed=0):
    rng = np.random.default_rng(seed)
    ro = Readout(rng)
    os.makedirs(out, exist_ok=True)
    for sub in ("labels", "preds", "factors"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    for name, _ in LAYERS:
        os.makedirs(os.path.join(out, "acts", name), exist_ok=True)

    def make_split(split, n, cls=None, img=None, b=None, h=None, noise=None):
        if cls is None:
            cls = rng.integers(0, 10, size=n)
            b = rng.uniform(0.15, 0.45, size=n).astype(np.float32)
            h = rng.uniform(0.0, 0.5, size=n).astype(np.float32)
            img = _images(b, h, cls, rng)
        acts = ro(cls, b, h, rng, noise=noise)
        for name, _ in LAYERS:
            np.save(os.path.join(out, "acts", name, f"{split}.npy"), acts[name].astype(np.float16))
        np.save(os.path.join(out, "labels", f"{split}.npy"), cls.astype(np.int64))
        # a fake "model prediction": nearest class in penult readout + noise
        pred = cls.copy()
        flip = rng.random(n) < 0.08 + 0.3 * (h > 0.4)
        pred[flip] = rng.integers(0, 10, size=flip.sum())
        np.savez(os.path.join(out, "preds", f"{split}.npz"), argmax=pred,
                 maxprob=np.clip(rng.beta(5, 2, size=n), 0, 1).astype(np.float32))
        fac = compute_pixel_factors(img)
        np.savez(os.path.join(out, "factors", f"{split}.npz"), **fac)
        return cls, img, b, h

    make_split("ref", n_ref)
    cls_t, img_t, b_t, h_t = make_split("test", n_test)
    test_noise = ro.last_noise
    make_split("panel", n_panel)
    splits = ["ref", "test", "panel"]
    for kind, sevs in CORRUPTIONS.items():
        for sev in sevs:
            s = corrupt_split(kind, sev)
            imgc = _corrupt(img_t[:n_corr], kind, sev, rng)
            # latent update consistent with the corruption
            b2 = b_t[:n_corr] + (0.08 * sev if kind == "brightness" else 0)
            h2 = h_t[:n_corr] * (0.5 ** sev if kind == "defocus_blur" else 1) + (0.06 * sev if kind == "gaussian_noise" else 0)
            make_split(s, n_corr, cls=cls_t[:n_corr], img=imgc, b=b2.astype(np.float32),
                       h=h2.astype(np.float32), noise=dict(test_noise))
            splits.append(s)
    meta = {
        "source": "synthetic",
        "arch": "synthetic_readout",
        "weights": "none",
        "layers": [n for n, _ in LAYERS],
        "dims": {n: d for n, d in LAYERS},
        "splits": splits,
        "n_classes": 10,
        "dtype": "float16",
        "pooling": "gap",
        "pairing": "corrupt splits are paired with test[:n] by row index",
        "seed": seed,
    }
    json.dump(meta, open(os.path.join(out, META_NAME), "w"), indent=2)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-ref", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print("synthetic dump at", make_synth_dump(a.out, n_ref=a.n_ref, seed=a.seed))
