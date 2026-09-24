#!/usr/bin/env python3
"""
b4_extract.py -- THE batch-4 CIFAR Stage-A instrument (docs/plans/B4_INTEGRATION.md D2, D4, D7, D11). Owner: T2
(extractor); T1 owns scripts/b4_weights.py (load_model, last_linear, export_head, to_input), imported here. GPU only for
real units (D17: never extract on a CPU pod); the synthetic self-test runs on the CPU.

  python scripts/b4_extract.py --registry experiments/b4/models.json --unit <id> --layout fit|eval|maps --volume /workspace
         [--sealed] [--record results/b4_extract/<id>_<layout><tag>.json] [--weights-record results/b4_weights/weights.json]
  python scripts/b4_extract.py --data-report --volume /workspace --out <json>
  python scripts/b4_extract.py --anchor-gate --registry R --units <id>... --out <json>        (hard for ANCHOR units)
  python scripts/b4_extract.py --seal-manifest --registry R --out <json>                       (end of S1: sealed.json)
  python scripts/b4_extract.py --verify-seals --registry R --manifest <sealed.json> [--deep] [--out <json>]   (S2 guard)
  python scripts/b4_extract.py --selftest [--selftest-out <json>]                              (synthetic, CPU)
  python scripts/b4_extract.py --list --registry R

One instrument for all 43 units (D2): the 18 old nets are re-extracted here (the committed atlas dumps are opened ONLY by
--anchor-gate), so discovery and confirmation numbers come from the same code, rows, dtypes, device and session.
Taps: ResNets through atlas/extract_acts.BlockHooks UNCHANGED (block stride 1/2/3/5 for depth 20/32/44/56: the batch-1..3
taps); VGG-bn / MobileNetV2 / ShuffleNetV2 / RepVGG through ModuleHooks (T2's tap specs at the pinned hub commit). Every tap
is GAP-pooled; penult = the input of the last nn.Linear. Preprocessing: b4_weights.to_input (bitwise the PIL path of every
earlier dump), on the CPU, then moved to the device. Batch 256 (the batch-1..3 extraction batch).

Layouts and rows (global CIFAR-10 test indices; corrupt and fault splits pair by row; D4):
  fit   ref = train rows sorted(default_rng(0).choice(50000, 10000)) (= every atlas dump); test = rows 0-4999;
        corrupt__<disc>__s1/3/5 = CIFAR-10-C rows 0-1999; corrupt__<holdout>__s1/3/5 rows 0-1999 (X4 taps);
        corrupt__<extra>__s3/5 rows 0-1999 (X4 taps; units with layouts.fit.extras: C, F, F20, K);
        ood__cifar100 / ood__svhn rows 0-1999; fault__{deadpix, occlusion_disc, exposure_global}__l1/2/3 = test rows
        3500-4999 through atlas/faults.t1_fault(salt 0)
  eval  (F, F20, R2; never-read rows) test = rows 5000-9999; corrupt__<disc>__s1/3/5 rows 5000-9999; holdout s3/s5 and
        (F, F20) extras s3/s5 rows 5000-9999 (X4 taps); ood rows 2000-6999; T1 faults on test rows 8500-9999 (salt 1).
        No ref: an eval-layout analysis calibrates on the same unit's fit dump.
  maps  (lane S, ResNets) ref = every 2nd row of the atlas reference draw (5000); cal = test rows 7500-8499; eval = rows
        8500-9499; fault__<cond> = atlas/faults.s_conditions on rows 8500-9499 (soiling only on Sconf units);
        global__<corr>__s3 = CIFAR-10-C rows 8500-9499. Stored: penult, headmap_std (spatial std of layer3), raw float16
        maps (maps/headmap = layer3 8x8x64; maps/stage2map = layer2 16x16x32 where the registry lists it), the uint8
        pixels (pixels/<split>.npy) and the fault masks (masks/<split>.npy).
Storage: ref, test, OOD, faults and the 10 discovery corruptions: every GAP tap; holdout corruptions and extras: the X4
set only (first tap, pre, penult). Penult float32 on ref / test / cal / eval / OOD, float16 elsewhere; other taps float16;
logits float32 (the canonical head path, D2); labels, rows int64; preds (argmax, maxprob); pixel factors (fit / eval,
atlas.factors unchanged; cached by image hash in $ATLAS_B4_FACTOR_CACHE, default /root/b4_factor_cache); head/head.npz.
A finite check runs after every cast. Tap metadata in meta.b4.taps: spatial, dup_of_penult (float32 relative RMS < 1e-4),
depth_frac, pre (the last spatial tap that is not a copy of the penult, every track), functional {stem, s1end, s2end, pre,
penult}, x4 [first, pre, penult].
Gates (fatal for the unit; meta.json is then never written): weights record FAIL, hub byte size, checkpoint norm; head
identity max |penult32 W^T + b - logits| / max(1, max |logits|) <= 1e-4; canonical head path per split (stored penult W^T + b vs stored logits:
argmax agreement >= 0.99 and every disagreement a near-tie, stored top-2 gap <= 0.05); finiteness.
Append-only: meta.json is written LAST (tmp + rename); a complete dump is skipped; a partial one is refused (move it
aside with --move-partial, which renames it to <dump>_step_partial_<stamp>: the pull's --exclude="dump_step*" and the
.gitignore pattern results/**/dump_step* already cover that name). A sealed layout gets SEALED.json FIRST and the
extractor prints only PASS / FAIL for it (split names and counts, never an accuracy or a target); its --record copy
carries no accuracy. In an open (discovery-readable) dump the accuracy of a confirmation-only split (D4: holdout
corruptions, extras, exposure_global, soiling) is neither printed nor stored: meta.accuracy holds "CONFIRMATION_ONLY".

INTERFACE for the tracks: read the dumps through atlas/b4_core.open_dump / open_unit only. Split names:
  ref, test, corrupt__<c>__s<k>, ood__cifar100, ood__svhn, fault__<kind>__l<level> (fit / eval);
  ref, cal, eval, fault__<cond> (cond from atlas/faults.S_CONDITIONS), global__<c>__s3 (maps).
"""
import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
import zlib

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from atlas import faults as FAULTS                     # noqa: E402  numpy only
from atlas.b4_core import is_confirmation_only          # noqa: E402  numpy only (D7 split list)
import b4_weights as BW                                 # noqa: E402  torch imported lazily inside its functions

SCHEMA = "b4_dump/1"
DISC = ("gaussian_noise", "shot_noise", "defocus_blur", "motion_blur", "snow", "fog", "brightness", "contrast",
        "pixelate", "jpeg_compression")
HOLDOUT = ("impulse_noise", "glass_blur", "zoom_blur", "frost", "elastic_transform")
EXTRA = ("speckle_noise", "gaussian_blur", "spatter", "saturate")
SEVS = (1, 3, 5)
EVAL_HOLDOUT_SEVS = (3, 5)
EXTRA_SEVS = (3, 5)
T1_FAULT_LEVELS = (1, 2, 3)
RESNET_STRIDE = {20: 1, 32: 2, 44: 3, 56: 5}
SYNTH_ARCH = "synthetic_resnet"
REF_SEED = 0
BATCH = 256
ROWS = {   # D4; experiments/b4/models.json 'rows' repeats these (tests/test_b4_extract.py pins the agreement)
    "fit": {"n_ref": 10000, "test": (0, 5000), "c10c": (0, 2000), "ood": (0, 2000), "faults": (3500, 5000),
            "fault_salt": 0},
    "eval": {"test": (5000, 10000), "c10c": (5000, 10000), "ood": (2000, 7000), "faults": (8500, 10000),
             "fault_salt": 1},
    "maps": {"n_ref": 10000, "ref_stride": 2, "cal": (7500, 8500), "eval": (8500, 9500), "faults": (8500, 9500),
             "c10c": (8500, 9500)},
}
GATE = {"head_identity_rel_max": 1e-4, "argmax_agree_min": 0.99, "near_tie_gap_max": 0.05, "dup_rel_rms": 1e-4}
ANCHOR = {"tap_rel_max": 4e-3, "argmax_agree_min": 0.999, "acc_abs_max": 0.0006}     # D15
MAPS_MODULES = {"headmap": "layer3", "stage2map": "layer2"}                            # ResNet lane-S maps


def say(msg):
    print(f"[b4_extract] {msg}", flush=True)


# =====================================================================================================================
# taps
# =====================================================================================================================
def family(arch):
    if arch == SYNTH_ARCH or re.fullmatch(r"cifar10_resnet\d+", arch):
        return "resnet"
    for f in ("vgg", "mobilenetv2", "shufflenetv2", "repvgg"):
        if arch.startswith(f"cifar10_{f}"):
            return f
    raise ValueError(f"unknown arch {arch}")


class ModuleHooks:
    """GAP of named modules (a clone for 2-D outputs) + penult = input of the last nn.Linear; the forward / close
    contract of atlas/extract_acts.BlockHooks (T2's t2_extract.ModuleHooks)."""

    def __init__(self, model, names):
        import torch.nn as nn
        mods = dict(model.named_modules())
        missing = [n for n in names if n not in mods]
        if missing:
            raise SystemExit(f"[b4_extract] modules not found: {missing}")
        self.model, self.acts, self.handles, self.layer_names = model, {}, [], []
        for n in names:
            self.handles.append(mods[n].register_forward_hook(self._hook(n)))
            self.layer_names.append(n)
        final = [m for m in model.modules() if isinstance(m, nn.Linear)][-1]

        def _penult(_, inp, __):
            self.acts["penult"] = inp[0].detach().clone()
        self.handles.append(final.register_forward_hook(_penult))
        self.layer_names.append("penult")

    def _hook(self, name):
        def hook(_, __, out):
            self.acts[name] = out.mean(dim=(2, 3)).detach() if out.dim() == 4 else out.detach().clone()
        return hook

    def forward(self, x):
        import torch
        self.acts = {}
        with torch.no_grad():
            logits = self.model(x)
        return {k: v.float() for k, v in self.acts.items()}, logits.float()

    def close(self):
        for h in self.handles:
            h.remove()


class MapsHooks:
    """Lane S: penult, raw maps of named modules (cloned) and the spatial std of the head map (unbiased, per channel)."""

    def __init__(self, model, maps):
        import torch.nn as nn
        mods = dict(model.named_modules())
        self.model, self.acts, self.handles = model, {}, []
        self.maps = dict(maps)
        for alias, mod in self.maps.items():
            if mod not in mods:
                raise SystemExit(f"[b4_extract] map module {mod} not found")
            self.handles.append(mods[mod].register_forward_hook(self._hook(alias)))
        final = [m for m in model.modules() if isinstance(m, nn.Linear)][-1]

        def _penult(_, inp, __):
            self.acts["penult"] = inp[0].detach().clone()
        self.handles.append(final.register_forward_hook(_penult))
        self.layer_names = ["headmap_std", "penult"]

    def _hook(self, alias):
        def hook(_, __, out):
            o = out.detach()
            self.acts[f"map:{alias}"] = o.clone()
            if alias == "headmap":
                self.acts["headmap_std"] = o.std(dim=(2, 3))
        return hook

    def forward(self, x):
        import torch
        self.acts = {}
        with torch.no_grad():
            logits = self.model(x)
        return {k: v.float() for k, v in self.acts.items()}, logits.float()

    def close(self):
        for h in self.handles:
            h.remove()


def tap_names(arch, model):
    """Module names tapped for each chenyaofo family at the pinned commit (penult is added by the hook class)."""
    import torch.nn as nn
    f = family(arch)
    if f == "vgg":
        feats = model.features
        relu0 = next(i for i, m in enumerate(feats) if isinstance(m, nn.ReLU))
        pools = [i for i, m in enumerate(feats) if isinstance(m, nn.MaxPool2d)]
        if len(pools) != 5:
            raise SystemExit(f"[b4_extract] {arch}: {len(pools)} MaxPool layers")
        return [f"features.{relu0}"] + [f"features.{i}" for i in pools] + ["classifier.1"]
    if f == "mobilenetv2":
        if len(model.features) != 19:
            raise SystemExit(f"[b4_extract] {arch}: {len(model.features)} feature blocks")
        return ["features.0", "features.1", "features.3", "features.6", "features.10", "features.13", "features.16",
                "features.17"]
    if f == "shufflenetv2":
        if len(model.stage3) != 8:
            raise SystemExit(f"[b4_extract] {arch}: stage3 has {len(model.stage3)} units")
        return ["conv1", "stage2", "stage3.3", "stage3", "stage4"]
    if f == "repvgg":
        if not (len(model.stage3) == 14 and len(model.stage4) == 1):
            raise SystemExit(f"[b4_extract] {arch}: unexpected RepVGG stages")
        return ["stage0", "stage1", "stage2", "stage3.6", "stage3"]
    raise SystemExit(f"[b4_extract] no ModuleHooks tap spec for {arch}")


def make_hooks(arch, model):
    from atlas.extract_acts import BlockHooks                       # unchanged since 2b561a9 (D6)
    m = re.fullmatch(r"cifar10_resnet(\d+)", arch)
    if m:
        return BlockHooks(model, "blocks", RESNET_STRIDE[int(m.group(1))], "gap")
    if arch == SYNTH_ARCH:
        return BlockHooks(model, "blocks", 1, "gap")
    return ModuleHooks(model, tap_names(arch, model))


def _tap_module(t, names):
    return ("relu" if "relu" in names else "bn1") if t == "stem" else t   # BlockHooks' stem (extract_acts.py:55)


def depth_fractions(model, taps):
    """(# Conv2d / Linear modules up to the tap's last descendant, pre-order) / (total); penult := 1.0."""
    import torch.nn as nn
    names = [n for n, _ in model.named_modules()]
    is_w = [isinstance(m, (nn.Conv2d, nn.Linear)) for _, m in model.named_modules()]
    total = sum(is_w)
    out = {}
    for t in taps:
        if t == "penult":
            out[t] = 1.0
            continue
        mod = _tap_module(t, names)
        last = max(i for i, n in enumerate(names) if n == mod or n.startswith(mod + "."))
        out[t] = sum(is_w[:last + 1]) / total
    return out


def tap_spatial(model, taps, device):
    """{tap: output is 4-D} from one zero-input forward in eval mode (T2 review #6)."""
    import torch
    mods, seen, hs = dict(model.named_modules()), {}, []
    for t in taps:
        m = _tap_module(t, mods)
        if t != "penult" and m in mods:
            hs.append(mods[m].register_forward_hook(lambda _, __, o, t=t: seen.__setitem__(t, o.dim() == 4)))
    with torch.no_grad():
        model(torch.zeros(1, 3, 32, 32, device=device))
    for h in hs:
        h.remove()
    return {t: bool(seen.get(t, False)) for t in taps if t != "penult"}


def rel_rms(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.sqrt(((a - b) ** 2).mean() / ((b ** 2).mean() + 1e-30)))


def pre_tap_rule(taps, spatial, dups):
    """D2 (every track): the last spatial tap that is not a copy of the penult. = atlas/b4_collapse.pre_tap (tested)."""
    cand = [t for t in taps if t != "penult" and t not in set(dups) and bool(spatial.get(t, False))]
    return cand[-1] if cand else None


def functional_taps(arch, taps, pre):
    """{stem, s1end, s2end, pre, penult}: stem = the first tap; stage ends per family (T1's map, on T2's tap names)."""
    f = family(arch)
    if f == "resnet":
        s1 = [t for t in taps if t.startswith("layer1.")][-1]
        s2 = [t for t in taps if t.startswith("layer2.")][-1]
    elif f == "vgg":
        s1, s2 = taps[2], taps[3]                                    # pool2, pool3 (taps: relu0, pool1..pool5, fc1)
    elif f == "mobilenetv2":
        s1, s2 = "features.3", "features.6"
    elif f == "shufflenetv2":
        s1, s2 = "stage2", "stage3"
    else:
        s1, s2 = "stage1", "stage2"
    return {"stem": taps[0], "s1end": s1, "s2end": s2, "pre": pre, "penult": "penult"}


# =====================================================================================================================
# data
# =====================================================================================================================
class _Data:
    """Split loaders shared by the real and the synthetic source. Each returns {imgs uint8, labels int64, rows int64,
    masks bool | None} or None when a CIFAR-10-C file is absent (recorded in meta.b4.missing)."""

    def ref(self, n, stride=1):
        imgs, labs = self.train()
        idx = np.sort(np.random.default_rng(REF_SEED).choice(len(imgs), size=min(int(n), len(imgs)), replace=False))
        idx = idx[::int(stride)]
        return {"imgs": imgs[idx], "labels": labs[idx], "rows": idx.astype(np.int64), "masks": None}

    def test_rows(self, lo, hi):
        imgs, labs = self.test()
        return {"imgs": imgs[lo:hi], "labels": labs[lo:hi], "rows": np.arange(lo, hi, dtype=np.int64), "masks": None}

    def c10c(self, c, s, lo, hi):
        raw = self.c10c_raw(c, s, lo, hi)
        if raw is None:
            return None
        imgs, labs = raw
        if not np.array_equal(labs, self.test()[1][lo:hi]):
            raise SystemExit(f"[b4_extract] pairing broken: CIFAR-10-C {c} s{s} rows {lo}-{hi}")
        return {"imgs": imgs, "labels": labs, "rows": np.arange(lo, hi, dtype=np.int64), "masks": None}

    def ood_rows(self, name, lo, hi):
        imgs, labs = self.ood(name)
        return {"imgs": imgs[lo:hi], "labels": labs[lo:hi], "rows": np.arange(lo, hi, dtype=np.int64), "masks": None}

    def t1_fault_rows(self, kind, level, lo, hi, salt):
        d = self.test_rows(lo, hi)
        d["imgs"] = FAULTS.t1_fault(d["imgs"], kind, level, salt)
        return d

    def s_fault_rows(self, cond, lo, hi):
        kind, corr, area = FAULTS.parse_condition(cond)
        d = self.test_rows(lo, hi)
        paste = None
        if kind == "paste":
            p = self.c10c(corr, 5, lo, hi)
            if p is None:
                return None
            paste = p["imgs"]
        d["imgs"], d["masks"] = FAULTS.s_fault_batch(kind, area, d["imgs"], d["rows"], FAULTS.SEED, paste)
        return d


class CifarData(_Data):
    """The volume's datasets (atlas/extract_acts.load_split_arrays and _resolve, unchanged)."""

    def __init__(self, volume):
        self.volume, self._c = volume, {}

    def _arr(self, name):
        if name not in self._c:
            from atlas.extract_acts import load_split_arrays
            imgs, labs = load_split_arrays(self.volume, name, None)
            self._c[name] = (np.ascontiguousarray(imgs), np.asarray(labs).astype(np.int64))
        return self._c[name]

    def train(self):
        return self._arr("cifar10_train")

    def test(self):
        return self._arr("cifar10")

    def ood(self, name):
        return self._arr(name)

    def c10c_root(self):
        from atlas.extract_acts import _resolve
        return _resolve(self.volume, "cifar10c")

    def c10c_raw(self, c, s, lo, hi):
        root = self.c10c_root()
        p = os.path.join(root, f"{c}.npy")
        if not os.path.isfile(p):
            return None
        a = np.load(p, mmap_mode="r")
        lab = np.load(os.path.join(root, "labels.npy"), mmap_mode="r")
        if a.shape != (50000, 32, 32, 3) or a.dtype != np.uint8:
            raise SystemExit(f"[b4_extract] {p}: shape {a.shape} dtype {a.dtype}")
        base = (int(s) - 1) * 10000
        imgs = np.asarray(a[base + lo:base + hi])
        labs = np.asarray(lab[base + lo:base + hi] if len(lab) == 50000 else lab[lo:hi]).astype(np.int64)
        return imgs, labs


class SynthData(_Data):
    """Deterministic random uint8 images for the self-test and tests (no volume). CIFAR-10-C corruptions are seeded
    additive noise on the paired test rows; names in `missing` are absent."""

    def __init__(self, n_train=300, n_test=200, n_ood=120, seed=0, missing=()):
        r = np.random.default_rng(seed)
        self._train = (r.integers(0, 256, (n_train, 32, 32, 3)).astype(np.uint8), r.integers(0, 10, n_train))
        self._test = (r.integers(0, 256, (n_test, 32, 32, 3)).astype(np.uint8), r.integers(0, 10, n_test))
        self._ood = {k: (r.integers(0, 256, (n_ood, 32, 32, 3)).astype(np.uint8), r.integers(0, 10, n_ood))
                     for k in ("cifar100", "svhn")}
        self.missing = set(missing)

    def train(self):
        return self._train

    def test(self):
        return self._test

    def ood(self, name):
        return self._ood[name]

    def c10c_raw(self, c, s, lo, hi):
        if c in self.missing:
            return None
        imgs, labs = self._test
        r = np.random.default_rng([zlib.crc32(c.encode()), int(s)])
        noise = r.integers(-20 * int(s), 20 * int(s) + 1, imgs[lo:hi].shape)
        return np.clip(imgs[lo:hi].astype(np.int64) + noise, 0, 255).astype(np.uint8), labs[lo:hi].copy()


def plan_splits(spec, layout, data, rows=ROWS):
    """[(split, loader, keep, penult_float32)] in extraction order; keep: 'all' | 'x4' | 'maps'."""
    L, R = spec["layouts"][layout], rows[layout]
    P = []
    if layout in ("fit", "eval"):
        if layout == "fit":
            P.append(("ref", lambda: data.ref(R["n_ref"], 1), "all", True))
        lo, hi = R["test"]
        P.append(("test", lambda: data.test_rows(lo, hi), "all", True))
        clo, chi = R["c10c"]
        groups = [(DISC, SEVS, "all"), (HOLDOUT, SEVS if layout == "fit" else EVAL_HOLDOUT_SEVS, "x4")]
        if L.get("extras"):
            groups.append((EXTRA, EXTRA_SEVS, "x4"))
        for names, sevs, keep in groups:
            for c in names:
                for s in sevs:
                    P.append((f"corrupt__{c}__s{s}", lambda c=c, s=s: data.c10c(c, s, clo, chi), keep, False))
        olo, ohi = R["ood"]
        for name in ("cifar100", "svhn"):
            P.append((f"ood__{name}", lambda name=name: data.ood_rows(name, olo, ohi), "all", True))
        flo, fhi = R["faults"]
        for kind in FAULTS.T1_KINDS:
            for lvl in T1_FAULT_LEVELS:
                P.append((f"fault__{kind}__l{lvl}",
                          lambda kind=kind, lvl=lvl: data.t1_fault_rows(kind, lvl, flo, fhi, R["fault_salt"]),
                          "all", False))
        return P
    if layout != "maps":
        raise SystemExit(f"[b4_extract] unknown layout {layout}")
    P.append(("ref", lambda: data.ref(R["n_ref"], R["ref_stride"]), "maps", True))
    for nm in ("cal", "eval"):
        lo, hi = R[nm]
        P.append((nm, lambda lo=lo, hi=hi: data.test_rows(lo, hi), "maps", True))
    flo, fhi = R["faults"]
    for cond in FAULTS.s_conditions(bool(L.get("holdout_faults"))):
        P.append((f"fault__{cond}", lambda cond=cond: data.s_fault_rows(cond, flo, fhi), "maps", False))
    glo, ghi = R["c10c"]
    for c in FAULTS.S_GLOBALS:
        sv = FAULTS.S_GLOBAL_SEVERITY
        P.append((f"global__{c}__s{sv}", lambda c=c, sv=sv: data.c10c(c, sv, glo, ghi), "maps", False))
    return P


def pixel_factors(imgs, cache_dir=None):
    """atlas.factors.compute_pixel_factors, cached by the sha256 of the image bytes (factors depend on pixels only)."""
    from atlas.factors import compute_pixel_factors
    key = hashlib.sha256(np.ascontiguousarray(imgs).tobytes()).hexdigest()[:32]
    p = os.path.join(cache_dir, f"{key}.npz") if cache_dir else None
    if p and os.path.isfile(p):
        try:
            with np.load(p) as z:
                return {k: np.asarray(z[k]) for k in z.files}
        except Exception:                                          # a torn cache file is recomputed
            pass
    f = compute_pixel_factors(imgs)
    if p:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            tmp = f"{p}.{os.getpid()}.tmp"
            with open(tmp, "wb") as fh:
                np.savez(fh, **f)
            os.replace(tmp, p)
        except OSError:
            pass
    return f


# =====================================================================================================================
# writing
# =====================================================================================================================
class Writer:
    """Every file goes through here: written from memory, its sha256 and size recorded for the seal manifest."""

    def __init__(self, dump):
        self.dump, self.files = dump, {}

    def _put(self, rel, data):
        path = os.path.join(self.dump, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        self.files[rel] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

    def npy(self, rel, arr):
        buf = io.BytesIO()
        np.save(buf, arr)
        self._put(rel, buf.getvalue())

    def npz(self, rel, **kw):
        buf = io.BytesIO()
        np.savez(buf, **kw)
        self._put(rel, buf.getvalue())

    def json(self, rel, obj):
        self._put(rel, (json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + "\n").encode())


def cast_checked(arr, dtype, what):
    out = np.asarray(arr).astype(dtype)
    if np.issubdtype(out.dtype, np.floating) and not np.isfinite(out).all():
        raise SystemExit(f"[b4_extract] {what}: non-finite values after the {np.dtype(dtype).name} cast (FAIL)")
    return out


def head_path_check(pen_stored, W, b, L):
    """Canonical head path (D2): stored penult W^T + b against the stored float32 logits."""
    Z = np.asarray(pen_stored, dtype=np.float64) @ W.T + b
    Ls = np.asarray(L, dtype=np.float64)
    dis = Z.argmax(1) != Ls.argmax(1)
    top = -np.sort(-Ls, axis=1)
    gap = top[:, 0] - top[:, 1]
    rec = {"n": int(len(Ls)), "argmax_agree": float(1.0 - dis.mean()), "n_disagree": int(dis.sum()),
           "disagree_gap_max": float(gap[dis].max()) if dis.any() else 0.0,
           "recomputed_max_abs": float(np.abs(Z - Ls).max())}
    # agreement >= 0.99 means at most floor(0.01 n) near-tie flips; at least one is allowed so a tiny split (the
    # self-test's 40 rows) is not failed by a single exact tie. Identical to ">= 0.99" for every n >= 100.
    rec["n_disagree_allowed"] = max(1, int(np.floor((1.0 - GATE["argmax_agree_min"]) * len(Ls) + 1e-9)))
    rec["status"] = "PASS" if (rec["n_disagree"] <= rec["n_disagree_allowed"]
                               and rec["disagree_gap_max"] <= GATE["near_tie_gap_max"]) else "FAIL"
    return rec


def run_split(hooks, imgs, norm, device, batch, keep):
    """-> ({key: float32 array}, logits float32); keep = keys of hooks.forward's dict to retain."""
    feats, logits = {k: [] for k in keep}, []
    for i in range(0, len(imgs), batch):
        f, lg = hooks.forward(BW.to_input(imgs[i:i + batch], norm).to(device))
        logits.append(lg.detach().float().cpu().numpy())
        for k in keep:
            feats[k].append(f[k].detach().float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in feats.items()}, np.concatenate(logits).astype(np.float32)


def dump_state(dump):
    if os.path.isfile(os.path.join(dump, "meta.json")):
        return "complete"
    if os.path.isdir(dump) and os.listdir(dump):
        return "partial"
    return "new"


def _full_commit():
    try:
        return subprocess.check_output(["git", "-C", REPO_ROOT, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return None


def _code_sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)


def weights_gate(spec, weights_record):
    """Refuse a unit b4_weights.py failed; a hub unit needs a record (the README gate runs before any extraction)."""
    if spec["source"] == "synthetic":
        return None
    rec = None
    if weights_record and os.path.isfile(_abs(weights_record)):
        with open(_abs(weights_record)) as f:
            rec = (json.load(f).get("units") or {}).get(spec["id"])
    if rec is not None and rec.get("status") == "FAIL":
        raise SystemExit(f"[b4_extract] {spec['id']}: weights gate FAIL in {weights_record} ({rec.get('why', '')})")
    if spec["source"] == "hub" and (rec is None or rec.get("status") not in ("PASS", "NOT_EVALUABLE")):
        raise SystemExit(f"[b4_extract] {spec['id']}: no PASS weights record in {weights_record} "
                         "(scripts/b4_weights.py runs before any extraction, D9 step 5)")
    return rec


def extract(spec, layout, data, dump, device, sealed, rows=ROWS, batch=BATCH, weights_record=None,
            factor_cache=None, move_partial=False):
    """Write one dump. Returns 'written' | 'skipped' (complete). Raises SystemExit on any fatal gate."""
    import torch
    t0 = time.time()
    uid = spec["id"]
    if layout not in spec["layouts"]:
        raise SystemExit(f"[b4_extract] {uid} has no {layout} layout in the registry")
    if bool(spec["layouts"][layout].get("sealed")) != bool(sealed):
        raise SystemExit(f"[b4_extract] {uid} {layout}: --sealed={bool(sealed)} but the registry says "
                         f"sealed={bool(spec['layouts'][layout].get('sealed'))}")
    if layout == "maps" and family(spec["arch"]) != "resnet":
        raise SystemExit(f"[b4_extract] {uid}: the maps layout is for ResNets only")
    st = dump_state(dump)
    if st == "complete":
        say(f"{uid} {layout}: {dump}/meta.json exists: complete dump skipped (append-only)")
        return "skipped"
    if st == "partial":
        if not move_partial:
            raise SystemExit(f"[b4_extract] {uid} {layout}: {dump} is a PARTIAL dump (no meta.json): refused. Move it "
                             "aside (--move-partial renames it to <dump>_step_partial_<stamp>) and relaunch")
        aside = f"{dump}_step_partial_{time.strftime('%Y%m%d_%H%M%S')}"   # matches dump_step* (pull exclude, .gitignore)
        os.replace(dump, aside)
        say(f"{uid} {layout}: partial dump moved aside to {aside}")
    wgate = weights_gate(spec, weights_record)
    os.makedirs(dump, exist_ok=True)
    wr = Writer(dump)
    if sealed:                                                     # first file: the dump is sealed even while partial
        wr.json("SEALED.json", {"unit": uid, "layout": layout, "rule": "docs/plans/B4_INTEGRATION.md D7: opened only "
                                "by atlas/b4_core.open_dump in the confirmation phase after P2"})
    model, wrec = BW.load_model(spec, device)
    if wrec.get("status") == "FAIL":
        raise SystemExit(f"[b4_extract] {uid}: weights record FAIL ({wrec})")
    _, W, b = BW.last_linear(model)
    head_rec = BW.export_head(model, os.path.join(dump, "head"))
    with open(os.path.join(dump, "head", "head.npz"), "rb") as f:
        hb = f.read()
    wr.files["head/head.npz"] = {"sha256": hashlib.sha256(hb).hexdigest(), "bytes": len(hb)}
    maps_mode = layout == "maps"
    if maps_mode:
        mlist = list(spec["layouts"]["maps"].get("maps") or ["headmap"])
        hooks = MapsHooks(model, {a: MAPS_MODULES[a] for a in mlist})
        taps, spatial, depth = hooks.layer_names, {}, {}
    else:
        hooks = make_hooks(spec["arch"], model)
        taps = list(hooks.layer_names)
        spatial = tap_spatial(model, taps, device)
        depth = depth_fractions(model, taps)
    norm = spec["norm"]
    plan = plan_splits(spec, layout, data, rows)
    mode = "sealed" if sealed else "open"
    say(f"{uid} {layout} ({mode}): {spec['arch']} {wrec['source']} {wrec.get('n_params')} params; "
        f"{len(taps)} taps; {len(plan)} splits -> {dump}")
    splits, dims, dtypes, absmax, acc, missing, hcheck, img_sha, timing, rows_meta = [], {}, {}, {}, {}, [], {}, {}, {}, {}
    dups, pre, functional, x4, ident, ref_idx, n_test = None, None, None, None, 0.0, None, None
    for name, loader, keep, pen32 in plan:
        ts = time.time()
        got = loader()
        if got is None:
            missing.append(name)
            continue
        imgs, labels, grows = got["imgs"], np.asarray(got["labels"], dtype=np.int64), got["rows"]
        if maps_mode:
            want = ["penult", "headmap_std"] + [f"map:{a}" for a in hooks.maps]
        elif keep == "all":
            want = list(taps)
        else:
            if x4 is None:
                raise SystemExit("[b4_extract] internal: an X4 split before the tap metadata")
            want = list(x4)
        feats, L = run_split(hooks, imgs, norm, device, batch, want)
        if not np.isfinite(L).all():
            raise SystemExit(f"[b4_extract] {uid} {name}: non-finite logits (FAIL)")
        pen = feats["penult"]
        ident = max(ident, float(np.abs(pen.astype(np.float64) @ W.T + b - L).max()) / max(1.0, float(np.abs(L).max())))
        if dups is None and not maps_mode:                             # first full split: tap metadata (float32)
            dups = [t for t in taps if t != "penult" and feats[t].shape == pen.shape
                    and rel_rms(feats[t], pen) < GATE["dup_rel_rms"]]
            pre = pre_tap_rule(taps, spatial, dups)
            functional = functional_taps(spec["arch"], taps, pre)
            x4 = [t for t in (taps[0], pre, "penult") if t is not None]
            x4 = list(dict.fromkeys(x4))
        for k, v in feats.items():
            if k == "penult":
                dt = np.float32 if pen32 else np.float16
            else:
                dt = np.float16
            arr = cast_checked(v, dt, f"{uid} {k}/{name}")
            if k.startswith("map:"):
                a = k[4:]
                wr.npy(f"maps/{a}/{name}.npy", arr)
                dims[f"map:{a}"] = list(arr.shape[1:])
                dtypes[f"map:{a}/{name}"] = arr.dtype.name
            else:
                wr.npy(f"acts/{k}/{name}.npy", arr)
                dims[k] = int(arr.shape[1])
                dtypes[f"{k}/{name}"] = arr.dtype.name
            absmax[k] = max(absmax.get(k, 0.0), float(np.abs(arr.astype(np.float32)).max()) if arr.size else 0.0)
            if k == "penult":
                hcheck[name] = head_path_check(arr, W, b, L)
        wr.npy(f"logits/{name}.npy", L)
        wr.npy(f"labels/{name}.npy", labels)
        wr.npy(f"rows/{name}.npy", np.asarray(grows, dtype=np.int64))
        P = np.exp(L.astype(np.float64) - L.astype(np.float64).max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        wr.npz(f"preds/{name}.npz", argmax=L.argmax(1).astype(np.int64), maxprob=P.max(1).astype(np.float32))
        if maps_mode:
            if spec["layouts"]["maps"].get("pixels", True):          # T3S pixel baselines read the dump, not the volume
                wr.npy(f"pixels/{name}.npy", np.ascontiguousarray(imgs, dtype=np.uint8))
            if got.get("masks") is not None:
                wr.npy(f"masks/{name}.npy", np.asarray(got["masks"], dtype=bool))
        else:
            wr.npz(f"factors/{name}.npz", **pixel_factors(imgs, factor_cache))
        img_sha[name] = hashlib.sha256(np.ascontiguousarray(imgs[:16]).tobytes()).hexdigest()[:16]
        if name.startswith("ood__"):
            acc[name] = None
        elif not sealed and is_confirmation_only(name):                # D4: never printed, stored or committed open
            acc[name] = "CONFIRMATION_ONLY"
        else:
            acc[name] = float((L.argmax(1) == labels).mean())
        if name == "ref":
            ref_idx = [int(v) for v in grows]
            rows_meta[name] = {"train_rows": "sorted(default_rng(0).choice(50000, 10000))"
                               + ("[::2]" if maps_mode else ""), "n": int(len(grows))}
        else:
            rows_meta[name] = [int(grows.min()), int(grows.max()) + 1] if len(grows) else None
        if name == "test":
            n_test = int(len(grows))
        splits.append(name)
        timing[f"split:{name}"] = round(time.time() - ts, 2)
        tail = f" acc={acc[name]:.4f}" if (not sealed and isinstance(acc[name], float)) else ""
        say(f"{uid} {layout} {name:40s} n={len(imgs):6d}{tail} ({time.time() - t0:.0f}s)")
    hooks.close()
    bad = [s for s, r in hcheck.items() if r["status"] != "PASS"]
    if ident > GATE["head_identity_rel_max"] or bad:
        raise SystemExit(f"[b4_extract] {uid} {layout}: head gate FAIL (identity max {ident:.3g}; canonical-path "
                         f"failures {bad}); meta.json not written")
    from atlas.extract_acts import NORMS, git_commit
    timing["total"] = round(time.time() - t0, 2)
    b4 = {"schema": SCHEMA, "unit": uid, "layout": layout, "sealed": bool(sealed), "roles": list(spec.get("roles", [])),
          "arch": spec["arch"], "family": family(spec["arch"]), "weights": wrec, "weights_gate": wgate, "head": head_rec,
          "taps": {"all": taps, "spatial": spatial, "depth_frac": depth, "dup_of_penult": dups or [], "pre": pre,
                   "functional": functional, "x4": x4 or []},
          "maps": {a: MAPS_MODULES[a] for a in hooks.maps} if maps_mode else None,
          "rows": rows_meta, "rows_spec": {k: (list(v) if isinstance(v, tuple) else v) for k, v in rows[layout].items()},
          "fault_salt": rows[layout].get("fault_salt"), "missing": missing,
          "extras_present": all(f"corrupt__{c}__s3" in splits for c in EXTRA) if spec["layouts"][layout].get("extras")
          else None,
          "dtypes": dtypes, "absmax": absmax, "image_sha256_first16": img_sha, "files": wr.files,
          "head_check": {"identity_rel_max": ident, "per_split": hcheck, "status": "PASS", "bounds": GATE},
          "finite": "PASS", "batch": int(batch), "preprocess": "b4_weights.to_input (CPU, bitwise the PIL path)",
          "code": {p: _code_sha(os.path.join(REPO_ROOT, p)) for p in
                   ("scripts/b4_extract.py", "scripts/b4_weights.py", "atlas/faults.py", "atlas/extract_acts.py")},
          "repo_commit": _full_commit(), "torch": torch.__version__, "numpy": np.__version__,
          "tf32": {"cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                   "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32)},
          "device_name": torch.cuda.get_device_name(0) if str(device).startswith("cuda") else "cpu",
          "timing_s": timing, "max_rss_mb": _rss_mb()}
    meta = {"source": "synthetic" if spec["source"] == "synthetic" else "real", "arch": spec["arch"],
            "weights": f"{wrec['source']}:{wrec.get('file')} sha256:{(wrec.get('sha256') or '')[:16]}",
            "seed_tag": uid, "layers": taps, "dims": dims, "splits": splits, "n_classes": 10,
            "dtype": "mixed (see b4.dtypes)", "pooling": "gap", "norm": norm, "norm_values": [list(v) for v in NORMS[norm]],
            "pairing": "corrupt / fault / global splits pair with the clean test rows of the same index (rows/<split>.npy)",
            "ref_indices": ref_idx or [], "panel_indices": [], "n_test": n_test, "accuracy": acc,
            "git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M:%S"), "device": str(device),
            "exp_id": f"b4_{uid}_{layout}", "b4": b4}
    tmp = os.path.join(dump, "meta.json.tmp")
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=1, allow_nan=False)
    os.replace(tmp, os.path.join(dump, "meta.json"))                 # written LAST: marks a complete dump
    say(f"{uid} {layout} PASS ({timing['total']:.0f}s)")
    return "written"


def _rss_mb():
    try:
        import resource
    except ImportError:
        return None
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def record_of(dump):
    """The committed copy of a dump's meta (results/b4_extract/<id>_<layout>.json): no reference index list, and for a
    sealed dump no accuracy."""
    p = os.path.join(dump, "meta.json")
    with open(p, "rb") as f:
        raw = f.read()
    meta = json.loads(raw)
    meta.pop("ref_indices", None)
    meta.pop("panel_indices", None)
    b4 = meta.get("b4") or {}
    files = b4.pop("files", {}) or {}
    b4["files_summary"] = {"n": len(files), "bytes": sum(v["bytes"] for v in files.values())}
    b4.pop("dtypes", None)                                     # kept in the dump meta; the record stays small
    for k in ("W", "b"):
        (b4.get("head") or {}).pop(k, None)                    # the ResNet heads are in results/b4_weights/weights.json
    if b4.get("sealed"):
        meta.pop("accuracy", None)
        meta["accuracy"] = "SEALED"
    meta["meta_sha256"] = hashlib.sha256(raw).hexdigest()
    meta["dump"] = os.path.relpath(os.path.abspath(dump), REPO_ROOT).replace(os.sep, "/")
    return meta


def write_record(dump, path):
    if not path:
        return
    if os.path.exists(path):
        say(f"record {path} exists: kept (append-only)")
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(record_of(dump), f, indent=1, sort_keys=True, allow_nan=False)
    os.replace(tmp, path)


# =====================================================================================================================
# gates and seals
# =====================================================================================================================
def _load_json(p):
    with open(p) as f:
        return json.load(f)


def anchor_gate(reg, units, out):
    """D15: per unit, the re-extraction against the committed atlas (test rows 0-4999): per tap max|new - old| /
    max|old| <= 4e-3 (old dump on the volume), argmax agreement >= 0.999, |acc - committed meta.accuracy.test| <= 0.0006,
    and the committed 16-hex weight prefix. Returns the list of units that did not PASS."""
    from atlas.b4_core import open_dump
    res, bad = {"bounds": ANCHOR, "units": {}}, []
    for uid in units:
        spec = next(m for m in reg["models"] if m["id"] == uid)
        r = {"anchor": spec.get("anchor")}
        try:
            new = open_dump(spec["layouts"]["fit"]["dump"], "discovery")
            atlas = _load_json(os.path.join(REPO_ROOT, spec["anchor"], "atlas.json"))
            cm = atlas.get("meta") or {}
            yl, L = new.labels("test"), new.logits("test")
            r["acc_new"] = float((L.argmax(1) == yl).mean())
            r["acc_committed"] = float(cm["accuracy"]["test"])
            r["acc_abs_diff"] = abs(r["acc_new"] - r["acc_committed"])
            w = str(cm.get("weights") or "")
            want = w.split("sha256:", 1)[1].strip()[:16] if "sha256:" in w else None
            got = (new.b4.get("weights") or {}).get("sha256") or ""
            r["weights_prefix"] = {"committed": want, "new": got[:16], "ok": None if want is None else got.startswith(want)}
            od = os.path.join(REPO_ROOT, spec["anchor"], "dump")
            if not os.path.isfile(os.path.join(od, "meta.json")):
                r["old_dump"] = "MISSING"
                r["status"] = "NOT_EVALUABLE"
            else:
                old = open_dump(od, "discovery")
                n = min(len(yl), int(old.meta.get("n_test", len(yl))))
                r["argmax_agree"] = float((L[:n].argmax(1) == old.preds("test")["argmax"][:n]).mean())
                r["taps"] = {}
                for t in [t for t in new.taps if t in old.taps]:
                    a, o = new.acts(t, "test", np.float64)[:n], old.acts(t, "test", np.float64)[:n]
                    r["taps"][t] = float(np.abs(a - o).max() / max(np.abs(o).max(), 1e-12))
                r["tap_rel_max"] = max(r["taps"].values()) if r["taps"] else None
                ok = (r["tap_rel_max"] is not None and r["tap_rel_max"] <= ANCHOR["tap_rel_max"]
                      and r["argmax_agree"] >= ANCHOR["argmax_agree_min"]
                      and r["acc_abs_diff"] <= ANCHOR["acc_abs_max"] and r["weights_prefix"]["ok"] is not False)
                r["status"] = "PASS" if ok else "FAIL"
        except SystemExit as e:
            r.update(status="FAIL", why=str(e))
        except (OSError, KeyError, ValueError) as e:
            r.update(status="FAIL", why=f"{type(e).__name__}: {e}")
        res["units"][uid] = r
        say(f"anchor gate {uid}: {r['status']} (tap rel max {r.get('tap_rel_max')}, argmax agree "
            f"{r.get('argmax_agree')}, acc diff {r.get('acc_abs_diff')})")
        if r["status"] != "PASS":
            bad.append(uid)
    res["status"] = "PASS" if not bad else "FAIL"
    res["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_new(out, res)
    return bad


def _write_new(path, obj):
    if os.path.exists(path):
        raise SystemExit(f"[b4_extract] {path} exists: never overwritten")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True, allow_nan=False)
    os.replace(tmp, path)


def _sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def seal_manifest(reg, out):
    """sealed.json: sha256 of every sealed meta.json (committed in R1; D7)."""
    ent = []
    for m in reg["models"]:
        for lay, L in sorted(m["layouts"].items()):
            if not L.get("sealed"):
                continue
            d = _abs(L["dump"])
            e = {"unit": m["id"], "layout": lay, "dump": L["dump"]}
            if os.path.isfile(os.path.join(d, "meta.json")):
                meta = _load_json(os.path.join(d, "meta.json"))
                e.update(status="SEALED", meta_sha256=_sha_file(os.path.join(d, "meta.json")),
                         n_files=len(meta["b4"]["files"]), bytes=sum(v["bytes"] for v in meta["b4"]["files"].values()))
            else:
                e["status"] = "ABSENT"
            ent.append(e)
    doc = {"schema": "b4_sealed/1", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "repo_commit": _full_commit(), "entries": ent}
    _write_new(out, doc)
    say(f"seal manifest: {sum(e['status'] == 'SEALED' for e in ent)} sealed dumps, "
        f"{sum(e['status'] == 'ABSENT' for e in ent)} absent -> {out}")
    return doc


def verify_seals(manifest, deep=False):
    """-> (ok, report). Every SEALED entry: meta.json sha256 unchanged, every recorded file present with its size (and,
    with deep, its sha256), no extra file."""
    doc = _load_json(manifest)
    rep, ok = [], True
    for e in doc["entries"]:
        if e["status"] != "SEALED":
            continue
        d, why = _abs(e["dump"]), []
        mp = os.path.join(d, "meta.json")
        if not os.path.isfile(mp) or _sha_file(mp) != e["meta_sha256"]:
            why.append("meta.json changed or missing")
        else:
            files = _load_json(mp)["b4"]["files"]
            for rel, v in files.items():
                p = os.path.join(d, rel)
                if not os.path.isfile(p) or os.path.getsize(p) != v["bytes"]:
                    why.append(f"{rel}: missing or size changed")
                elif deep and _sha_file(p) != v["sha256"]:
                    why.append(f"{rel}: sha256 changed")
            have = set()
            for root, _, fs in os.walk(d):
                for fn in fs:
                    have.add(os.path.relpath(os.path.join(root, fn), d).replace(os.sep, "/"))
            extra = sorted(have - set(files) - {"meta.json"})
            if extra:
                why.append(f"extra files {extra[:5]}")
        rep.append({"unit": e["unit"], "layout": e["layout"], "status": "PASS" if not why else "FAIL", "why": why[:10]})
        ok &= not why
    return ok, {"manifest": manifest, "deep": bool(deep), "status": "PASS" if ok else "FAIL", "entries": rep}


def data_report(volume, out):
    """Which datasets and CIFAR-10-C files the volume holds (the 4 extras decide the extras' clauses)."""
    from atlas.extract_acts import _resolve, load_split_arrays
    rep = {"volume": volume, "cifar10c": {}, "datasets": {}}
    root = _resolve(volume, "cifar10c")
    for c in DISC + HOLDOUT + EXTRA:
        p = os.path.join(root, f"{c}.npy")
        if os.path.isfile(p):
            a = np.load(p, mmap_mode="r")
            rep["cifar10c"][c] = {"shape": list(a.shape), "dtype": str(a.dtype)}
        else:
            rep["cifar10c"][c] = None
    lp = os.path.join(root, "labels.npy")
    rep["cifar10c_labels"] = list(np.load(lp, mmap_mode="r").shape) if os.path.isfile(lp) else None
    for name in ("cifar10", "cifar10_train", "cifar100", "svhn"):
        try:
            im, lab = load_split_arrays(volume, name, None)
            rep["datasets"][name] = {"n": int(len(im)), "shape": list(np.asarray(im).shape[1:])}
        except Exception as e:
            rep["datasets"][name] = {"error": f"{type(e).__name__}: {e}"}
    rep["extras_present"] = all(rep["cifar10c"][c] for c in EXTRA)
    rep["main15_present"] = all(rep["cifar10c"][c] for c in DISC + HOLDOUT)
    _write_new(out, rep)
    say(f"data: 15 main corruptions {'present' if rep['main15_present'] else 'INCOMPLETE'}; "
        f"4 extras {'present' if rep['extras_present'] else 'ABSENT (their clauses NOT_EVALUABLE)'}")
    return rep


# =====================================================================================================================
# self-test (synthetic, CPU; no volume, no hub)
# =====================================================================================================================
SMALL_ROWS = {"fit": {"n_ref": 64, "test": (0, 100), "c10c": (0, 40), "ood": (0, 40), "faults": (70, 100),
                      "fault_salt": 0},
              "eval": {"test": (100, 200), "c10c": (100, 160), "ood": (40, 100), "faults": (170, 200), "fault_salt": 1},
              "maps": {"n_ref": 64, "ref_stride": 2, "cal": (100, 140), "eval": (140, 180), "faults": (140, 180),
                       "c10c": (140, 180)}}


def synth_spec(root, uid="synth_u", sealed_eval=True, maps_conf=True):
    """A registry entry for the synthetic tiny ResNet with fit (open), eval (sealed) and maps (sealed) layouts."""
    return {"id": uid, "arch": SYNTH_ARCH, "source": "synthetic", "seed": 0, "norm": "chenyaofo",
            "roles": ["D", "Sconf"] if maps_conf else ["D", "Sdisc"],
            "layouts": {"fit": {"dump": os.path.join(root, f"b4d_{uid}", "dump"), "sealed": False, "extras": True},
                        "eval": {"dump": os.path.join(root, f"b4c_{uid}_eval", "dump"), "sealed": sealed_eval},
                        "maps": {"dump": os.path.join(root, f"b4c_{uid}_maps", "dump"), "sealed": True,
                                 "maps": ["headmap", "stage2map"], "holdout_faults": bool(maps_conf), "pixels": True}}}


def selftest(workdir=None):
    import shutil
    import tempfile
    checks = []

    def ck(name, ok, detail=""):
        checks.append({"name": name, "pass": bool(ok), "detail": str(detail)[:300]})
        say(f"[selftest] {'PASS' if ok else 'FAIL'} {name} {detail if not ok else ''}")
    tmp = workdir or tempfile.mkdtemp(prefix="b4_extract_selftest_")
    try:
        import torch
        from PIL import Image
        from atlas.extract_acts import cifar_transform
        u8 = np.random.default_rng(0).integers(0, 256, (16, 32, 32, 3)).astype(np.uint8)
        a = torch.stack([cifar_transform("chenyaofo")(Image.fromarray(im)) for im in u8])
        ck("vectorised preprocessing is bitwise the PIL path", torch.equal(a, BW.to_input(u8, "chenyaofo")))
        f1 = FAULTS.t1_fault(u8, "deadpix", 2)
        ck("T1 faults deterministic, salt changes the defect", np.array_equal(f1, FAULTS.t1_fault(u8, "deadpix", 2))
           and not np.array_equal(f1, FAULTS.t1_fault(u8, "deadpix", 2, salt=1)))
        data = SynthData(missing=("saturate",))
        spec = synth_spec(tmp)
        fit = spec["layouts"]["fit"]["dump"]
        r = extract(spec, "fit", data, fit, "cpu", False, SMALL_ROWS, batch=32)
        meta = _load_json(os.path.join(fit, "meta.json"))
        tp = meta["b4"]["taps"]
        ck("fit layout written", r == "written" and "ref" in meta["splits"], meta["splits"][:3])
        ck("dup_of_penult = [layer3.1]; pre = layer3.0 (D2)", tp["dup_of_penult"] == ["layer3.1"] and tp["pre"] == "layer3.0",
           tp)
        ck("functional taps", tp["functional"] == {"stem": "stem", "s1end": "layer1.0", "s2end": "layer2.0",
                                                   "pre": "layer3.0", "penult": "penult"}, tp["functional"])
        ck("holdout split holds the X4 taps only", sorted(os.listdir(os.path.join(fit, "acts")))
           and not os.path.isfile(os.path.join(fit, "acts", "layer1.0", "corrupt__frost__s3.npy"))
           and os.path.isfile(os.path.join(fit, "acts", "layer3.0", "corrupt__frost__s3.npy")), tp["x4"])
        ck("penult float32 on ref/test/ood, float16 elsewhere",
           meta["b4"]["dtypes"]["penult/test"] == "float32" and meta["b4"]["dtypes"]["penult/corrupt__fog__s3"] == "float16"
           and meta["b4"]["dtypes"]["layer1.0/test"] == "float16")
        ck("a missing extra is recorded", "corrupt__saturate__s3" in meta["b4"]["missing"]
           and meta["b4"]["extras_present"] is False, meta["b4"]["missing"])
        ck("open dump: no accuracy of a confirmation-only split (D4)",
           meta["accuracy"]["corrupt__frost__s3"] == "CONFIRMATION_ONLY"
           and meta["accuracy"]["fault__exposure_global__l1"] == "CONFIRMATION_ONLY"
           and isinstance(meta["accuracy"]["corrupt__fog__s3"], float) and isinstance(meta["accuracy"]["test"], float),
           {k: meta["accuracy"].get(k) for k in ("corrupt__frost__s3", "fault__exposure_global__l1", "test")})
        ck("head gate PASS on every split", meta["b4"]["head_check"]["status"] == "PASS"
           and all(v["status"] == "PASS" for v in meta["b4"]["head_check"]["per_split"].values()))
        mt = os.path.getmtime(os.path.join(fit, "meta.json"))
        ck("meta.json written last", all(os.path.getmtime(os.path.join(fit, k)) <= mt for k in meta["b4"]["files"]))
        ck("a complete dump is skipped", extract(spec, "fit", data, fit, "cpu", False, SMALL_ROWS, batch=32) == "skipped")
        part = os.path.join(tmp, "b4d_partial", "dump")
        os.makedirs(part)
        open(os.path.join(part, "x.npy"), "wb").close()
        try:
            extract(spec, "fit", data, part, "cpu", False, SMALL_ROWS, batch=32)
            ck("a partial dump is refused", False)
        except SystemExit:
            ck("a partial dump is refused", True)
        r2 = extract(spec, "fit", data, part, "cpu", False, SMALL_ROWS, batch=32, move_partial=True)
        aside = sorted(x for x in os.listdir(os.path.dirname(part)) if x != "dump")
        ck("--move-partial renames a partial dump to dump_step_partial_<stamp> (covered by the pull's dump_step* exclude "
           "and the .gitignore) and extracts afresh", r2 == "written" and len(aside) == 1
           and aside[0].startswith("dump_step_partial_")
           and os.path.isfile(os.path.join(os.path.dirname(part), aside[0], "x.npy"))
           and os.path.isfile(os.path.join(part, "meta.json")), aside)
        from atlas.b4_core import open_dump, ReadRefused
        d = open_dump(fit, "discovery")
        ck("discovery refuses a confirmation-only split", not d.has("penult", "corrupt__frost__s3")
           and d.has("penult", "corrupt__fog__s3"))
        try:
            d.acts("penult", "fault__exposure_global__l1")
            ck("discovery read of exposure_global raises", False)
        except ReadRefused:
            ck("discovery read of exposure_global raises", True)
        ev = spec["layouts"]["eval"]["dump"]
        extract(spec, "eval", data, ev, "cpu", True, SMALL_ROWS, batch=32)
        ck("sealed eval dump carries SEALED.json", os.path.isfile(os.path.join(ev, "SEALED.json")))
        try:
            open_dump(ev, "discovery")
            ck("a sealed dump never opens in discovery", False)
        except ReadRefused:
            ck("a sealed dump never opens in discovery", True)
        rec = record_of(ev)
        ck("the committed record of a sealed dump has no accuracy", rec["accuracy"] == "SEALED")
        mp = spec["layouts"]["maps"]["dump"]
        extract(spec, "maps", data, mp, "cpu", True, SMALL_ROWS, batch=32)
        mm = _load_json(os.path.join(mp, "meta.json"))
        hm = np.load(os.path.join(mp, "maps", "headmap", "fault__soiling__a12.npy"))
        ck("maps layout: soiling on a confirmation unit, maps (n, 8, 8, 8)", hm.shape == (40, 8, 8, 8)
           and "fault__soiling__a12" in mm["splits"], hm.shape)
        ck("maps layout: masks and pixels stored", np.load(os.path.join(mp, "masks", "fault__glare__a12.npy")).shape
           == (40, 32, 32) and np.load(os.path.join(mp, "pixels", "cal.npy")).dtype == np.uint8)
        reg = {"models": [spec]}
        man = os.path.join(tmp, "sealed.json")
        seal_manifest(reg, man)
        ok, _ = verify_seals(man, deep=True)
        ck("seal manifest verifies", ok)
        with open(os.path.join(ev, "logits", "test.npy"), "ab") as f:
            f.write(b"\0")
        ok2, _ = verify_seals(man)
        ck("a changed sealed file fails verification", not ok2)
    except Exception as e:                                                     # a crash is a failed self-test
        import traceback
        ck("self-test ran without an exception", False, f"{type(e).__name__}: {e} {traceback.format_exc()[-400:]}")
    finally:
        if workdir is None:
            shutil.rmtree(tmp, ignore_errors=True)
    bad = [c["name"] for c in checks if not c["pass"]]
    return {"status": "PASS" if not bad else "FAIL", "failed": bad, "checks": checks,
            "code_sha256": _code_sha(os.path.abspath(__file__)),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


# =====================================================================================================================
# CLI
# =====================================================================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description="batch-4 CIFAR Stage A (docs/plans/B4_INTEGRATION.md D2)")
    ap.add_argument("--registry")
    ap.add_argument("--unit")
    ap.add_argument("--layout", choices=("fit", "eval", "maps"))
    ap.add_argument("--volume")
    ap.add_argument("--sealed", action="store_true")
    ap.add_argument("--record")
    ap.add_argument("--weights-record", default="results/b4_weights/weights.json")
    ap.add_argument("--device")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--allow-cpu", action="store_true", help="tests only: real units are GPU-only (D17)")
    ap.add_argument("--move-partial", action="store_true")
    ap.add_argument("--data-report", action="store_true")
    ap.add_argument("--anchor-gate", action="store_true")
    ap.add_argument("--units", nargs="*", default=[])
    ap.add_argument("--seal-manifest", action="store_true")
    ap.add_argument("--verify-seals", action="store_true")
    ap.add_argument("--manifest")
    ap.add_argument("--deep", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--selftest-out")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.selftest:
        if a.selftest_out and os.path.exists(a.selftest_out):
            print(f"[b4_extract] {a.selftest_out} exists: not overwritten", file=sys.stderr)
            return 2
        rep = selftest()
        if a.selftest_out:
            _write_new(a.selftest_out, rep)
        say(f"selftest {rep['status']}")
        return 0 if rep["status"] == "PASS" else 1
    if a.data_report:
        data_report(a.volume, a.out)
        return 0
    reg = _load_json(_abs(a.registry)) if a.registry else None
    if a.list:
        for m in reg["models"]:
            print(m["id"], ",".join(m["roles"]), m["arch"], m["source"],
                  " ".join(f"{k}:{'sealed' if v.get('sealed') else 'open'}" for k, v in sorted(m["layouts"].items())))
        return 0
    if a.anchor_gate:
        bad = anchor_gate(reg, a.units, a.out)
        anchors = {m["id"] for m in reg["models"] if "ANCHOR" in m["roles"]}
        return 1 if set(bad) & anchors else 0
    if a.seal_manifest:
        seal_manifest(reg, a.out)
        return 0
    if a.verify_seals:
        ok, rep = verify_seals(a.manifest, a.deep)
        if a.out:
            _write_new(a.out, rep)
        say(f"verify seals: {rep['status']} ({len(rep['entries'])} sealed dumps)")
        return 0 if ok else 1
    if not (reg and a.unit and a.layout and a.volume):
        ap.error("--registry, --unit, --layout and --volume are required for an extraction")
    spec = next((m for m in reg["models"] if m["id"] == a.unit), None)
    if spec is None:
        raise SystemExit(f"[b4_extract] {a.unit} is not in {a.registry}")
    import torch
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu" and spec["source"] != "synthetic" and not a.allow_cpu:
        raise SystemExit("[b4_extract] extraction is GPU-only (D17, T2 review #1: one instrument); no CUDA here")
    dump = _abs(spec["layouts"][a.layout]["dump"])
    cache = os.environ.get("ATLAS_B4_FACTOR_CACHE", "/root/b4_factor_cache")
    extract(spec, a.layout, CifarData(a.volume), dump, device, a.sealed, ROWS, a.batch, a.weights_record,
            cache if cache else None, a.move_partial)
    write_record(dump, a.record)
    return 0


if __name__ == "__main__":
    sys.exit(main())
