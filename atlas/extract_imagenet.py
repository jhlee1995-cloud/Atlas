"""
extract_imagenet.py -- STAGE A for B1 (GPU): ImageNet-1k validation through torchvision ResNet50 / ViT-B/16 and timm
DeiT-B (docs/plans/B1_VIT_MARGIN.md; pre-registration experiments/queue/margin_b1_vitb16.yaml notes).

Called by atlas/run_imagenet.py only. atlas/extract_acts.py and atlas/run.py stay byte-identical (A4b instrument
files, integration D3); BlockHooks, file_sha256, git_commit and _resolve are imported from extract_acts, unchanged.
The module level imports numpy only (extract_acts is torch-free at import): torch, torchvision, timm, pyarrow,
huggingface_hub, safetensors and PIL are imported inside functions, so validate_cfg and the split / legacy-row helpers
run in CPU tests without them.

Data: the two validation parquet files of the pinned HF mirror (scripts/b1_data.py provisions and verifies them), rows
numbered in the manifest's file order (the sorted-file order of the legacy glob; row i = ILSVRC2012_val_{i+1}).
Splits (pre-registered):
  part A / B    rng = numpy default_rng(split_seed); for class c = 0..K-1 in order, the rows of class c ascending,
                p = rng.permutation(len(rows)); A = rows[p[:per_class]], B = rows[p[per_class:]]; each part ascending.
                reference = one part, clean_test = the other (25,000 each).
  legacy_first  the rows Upgraded-Mod session_experiments/imagenet_extract.py:40-60 read with --n n_target: files in
                order, pq.ParquetFile(f).iter_batches(batch_size=64) over every column, one flush per 64 decoded rows,
                stop after the parquet batch in which 64 x flushes >= n_target (the rest of that batch is kept);
                reference = clean_test = those rows (in-sample centers, as the legacy).
Dump (atlas/context.py layout): acts/<tap>/<split>.npy (extract.dtype, float32 in every B1 manifest), labels/,
preds/<split>.npz (argmax and maxprob exactly as extract_acts.run_backbone; logit_gap = top-1 minus top-2 logit,
logit_top1, logit_top2, float32; real_ok int8 when the pinned ReaL real.json verifies: 1 argmax in the ReaL set,
0 not, -1 empty set), logits/<split>.npy (float16; Stage B never reads them), meta.json (written last) with every key
scripts/b1_gate.py and scripts/b1_verdicts.js read.
Taps: ResNet50 through BlockHooks (blocks: stem, layer1.0 .. layer4.2, penult; stages: stem, layer1 .. layer4,
penult); ViTs through ViTHooks: block.0 .. block.11 (class token of each encoder block output, before the final norm),
penult_mean (mean of the patch tokens after the final LayerNorm), penult (the input of the classifier Linear, i.e.
the final-norm class token), penult last. Forward in float32 with the PyTorch TF32 defaults unchanged (recorded in
meta.tf32). A decode failure is counted (meta.decode_failures, decode_failed_rows) and its row dropped from the split;
P0 refuses any failure outside the legacy block.
"""
import json
import os
import platform
import time

import numpy as np

from .context import META_NAME
from .extract_acts import BlockHooks, _resolve, file_sha256, git_commit

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARCHS = {   # manifest backbone.arch -> library, constructor, torchvision weights enum, tap family, classifier Linear
    "tv_resnet50": {"lib": "torchvision", "ctor": "resnet50", "enum": "ResNet50_Weights", "kind": "cnn",
                    "head": "fc", "weights": ("IMAGENET1K_V2",), "acc1": 80.858},
    "tv_vit_b_16": {"lib": "torchvision", "ctor": "vit_b_16", "enum": "ViT_B_16_Weights", "kind": "vit",
                    "head": "heads.head", "weights": ("IMAGENET1K_V1",), "acc1": 81.072},
    "timm_deit_base_patch16_224": {"lib": "timm", "ctor": "deit_base_patch16_224", "enum": None, "kind": "vit",
                                   "head": "head", "weights": ("fb_in1k",), "acc1": 81.98},
}
IMAGENET_NORM = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))   # not added to extract_acts.NORMS (A4b file)
_B1D = None


def _b1_data():
    """scripts/b1_data.py (pinned repo, revision, files and ReaL constants), loaded by path (scripts/ is no package)."""
    global _B1D
    if _B1D is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("b1_data", os.path.join(REPO_ROOT, "scripts", "b1_data.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _B1D = mod
    return _B1D


def _hex(v, n):
    return isinstance(v, str) and len(v) == n and all(c in "0123456789abcdef" for c in v)


# ---------------------------------------------------------------------------
# manifest contract (pure Python; before any download or data read)
# ---------------------------------------------------------------------------
def validate_cfg(cfg):
    """Every B1 manifest rule Stage A and the decision code rely on. Raises SystemExit listing every problem."""
    from . import invariants as _inv  # noqa: F401  (registers the cross-layer invariants)
    from .registry import CROSS_LAYER, select
    bad = []

    def need(ok, msg):
        if not ok:
            bad.append(msg)
    bd = _b1_data()
    bb, d = cfg.get("backbone") or {}, cfg.get("data") or {}
    hk, ex = cfg.get("hooks") or {}, cfg.get("extract") or {}
    arch, tr = bb.get("arch"), bb.get("transform")
    spec = ARCHS.get(arch)
    need(cfg.get("n_classes") == 1000, f"n_classes {cfg.get('n_classes')!r} (want 1000)")
    need(spec is not None, f"backbone.arch {arch!r} not in {sorted(ARCHS)}")
    need(spec is None or bb.get("weights") in spec["weights"], f"backbone.weights {bb.get('weights')!r} for {arch}")
    need(bb.get("norm") == "imagenet", f"backbone.norm {bb.get('norm')!r} (want imagenet)")
    need(tr == "official" or (tr == "legacy_imagenet_extract" and arch == "tv_resnet50"),
         f"backbone.transform {tr!r}: official | legacy_imagenet_extract (tv_resnet50 only)")
    if arch == "timm_deit_base_patch16_224":
        need(_hex(bb.get("hf_revision"), 40), "backbone.hf_revision: 40 hex characters")
        need(_hex(bb.get("weights_sha256"), 64), "backbone.weights_sha256: 64 hex characters")
    iv = d.get("imagenet_val") or {}
    need(iv.get("repo") == bd.REPO and iv.get("revision") == bd.REVISION,
         f"data.imagenet_val repo/revision {iv.get('repo')}@{iv.get('revision')} != scripts/b1_data.py")
    files = [(f.get("path"), f.get("sha256")) for f in (iv.get("files") or []) if isinstance(f, dict)]
    need(files == [(k, v[1]) for k, v in bd.FILES.items()], "data.imagenet_val.files != scripts/b1_data.py FILES")
    ref, ct = d.get("reference") or {}, d.get("clean_test") or {}
    need(ref.get("dataset") == "imagenet_val" and ct.get("dataset") == "imagenet_val",
         "reference and clean_test must be dataset imagenet_val")
    parts = (ref.get("part"), ct.get("part"))
    if parts in (("A", "B"), ("B", "A")):
        for k in ("split_seed", "per_class", "n"):
            need(ref.get(k) == ct.get(k), f"reference.{k} {ref.get(k)!r} != clean_test.{k} {ct.get(k)!r}")
        need(isinstance(ref.get("split_seed"), int), "reference.split_seed: an int")
        need(ref.get("per_class") == 25 and ref.get("n") == 25000, "per_class 25 and n 25000 (both splits)")
    elif parts == ("legacy_first", "legacy_first"):
        for k in ("n_target", "loop_batch"):
            need(ref.get(k) == ct.get(k), f"reference.{k} != clean_test.{k}")
        need(ref.get("n_target") == 10000 and ref.get("loop_batch") == 64, "legacy_first: n_target 10000, loop_batch 64")
        need("n" in ref and ref["n"] is None and "n" in ct and ct["n"] is None,
             "legacy_first: reference.n and clean_test.n must be null (config DEFAULTS would fill 10000 / 5000)")
        need(arch == "tv_resnet50" and tr == "legacy_imagenet_extract", "legacy_first: tv_resnet50, legacy transform")
    else:
        bad.append(f"reference.part / clean_test.part {parts}: A/B, B/A or legacy_first/legacy_first")
    need(d.get("panel") is None and d.get("cifar10c") is None and d.get("ood") == [],
         "data.panel and data.cifar10c must be null and data.ood []")
    if spec and spec["kind"] == "vit":
        need(hk.get("layers") == "vit_blocks" and hk.get("pooling") == "cls" and hk.get("block_stride") == 1
             and hk.get("extra") == ["penult_mean"],
             f"hooks {hk}: ViTs need {{layers: vit_blocks, block_stride: 1, pooling: cls, extra: [penult_mean]}}")
    elif spec:
        need(hk.get("layers") in ("blocks", "stages") and hk.get("pooling") == "gap" and hk.get("block_stride") == 1,
             f"hooks {hk}: ResNet50 needs layers blocks|stages, block_stride 1, pooling gap")
    need(ex.get("dtype") == "float32", f"extract.dtype {ex.get('dtype')!r} (want float32, integration D8)")
    need(ex.get("store_logits") is True, "extract.store_logits: true")
    need(isinstance(ex.get("batch"), int) and ex.get("batch") > 0, "extract.batch: a positive int")
    need(cfg.get("invariants") == ["margin_typeb"], f"invariants {cfg.get('invariants')!r} (want [margin_typeb])")
    try:
        sel = sorted(select(CROSS_LAYER, cfg.get("cross_layer", "all")))
    except KeyError as e:
        sel = [f"error {e}"]
    need(sel == [], f"cross_layer selects {sel} (want none)")
    mt = (cfg.get("invariant_cfg") or {}).get("margin_typeb") or {}
    need(mt.get("b1") is True, "invariant_cfg.margin_typeb.b1: true")
    need(((mt.get("legacy_imagenet") or {}).get("layers")) == ["penult"],
         "invariant_cfg.margin_typeb.legacy_imagenet.layers: [penult]")
    need((cfg.get("outputs") or {}).get("root") == f"results/{cfg.get('exp_id')}", "outputs.root must be results/<exp_id>")
    if bad:
        raise SystemExit(f"[extract_imagenet] {cfg.get('exp_id')}: manifest refused:\n  " + "\n  ".join(bad))


# ---------------------------------------------------------------------------
# splits (pure numpy)
# ---------------------------------------------------------------------------
def split_parts(labels, split_seed, per_class, n_classes):
    """The pre-registered split: (part A rows, part B rows), each ascending (see the module docstring)."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(int(split_seed))
    A, B = [], []
    for c in range(int(n_classes)):
        rows = np.flatnonzero(labels == c)
        p = rng.permutation(len(rows))
        A.append(rows[p[:per_class]])
        B.append(rows[p[per_class:]])
    return np.sort(np.concatenate(A)), np.sort(np.concatenate(B))


def legacy_rows_from_batches(batch_rows, n_target, loop_batch=64):
    """Rows the legacy loop (imagenet_extract.py:49-60) read, from the parquet batch sizes in read order, every row
    decoding: after each parquet batch the loop has flushed loop_batch x floor(read / loop_batch) rows and stops once
    that is >= n_target; the rest of that batch is kept (final flush). Returns the number of leading rows."""
    R = 0
    for b in batch_rows:
        R += int(b)
        if (R // loop_batch) * loop_batch >= n_target:
            return R
    return R


def legacy_first_rows(volume, files, n_target, loop_batch=64):
    """legacy_rows_from_batches over the pinned files read exactly as the legacy did (pq.ParquetFile(f)
    .iter_batches(batch_size=64), every column). Returns (rows, parquet batch sizes read)."""
    import pyarrow.parquet as pq
    root, sizes = _resolve(volume, "imagenet_val"), []
    for f in files:
        for b in pq.ParquetFile(os.path.join(root, f["path"])).iter_batches(batch_size=loop_batch):
            sizes.append(int(b.num_rows))
            R = sum(sizes)
            if (R // loop_batch) * loop_batch >= n_target:
                return R, sizes
    return sum(sizes), sizes


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_imagenet_val(volume, files):
    """(blob uint8 of every encoded image, offsets (N + 1,), labels int64 (N,), file records) of the validation parquet
    files, in the given order. One contiguous blob and no per-image Python objects afterwards, so forked DataLoader
    workers share it."""
    import pyarrow.parquet as pq
    root = _resolve(volume, "imagenet_val")
    blobs, lens, labels, recs = [], [], [], []
    for f in files:
        p = os.path.join(root, f["path"])
        t = pq.read_table(p, columns=["image", "label"])
        parts = []
        for ch in t.column("image").chunks:                          # struct<bytes: binary, path: string>
            parts += [b if b is not None else b"" for b in ch.field("bytes").to_pylist()]
        labels.append(t.column("label").to_numpy().astype(np.int64))
        lens.append(np.fromiter((len(b) for b in parts), dtype=np.int64, count=len(parts)))
        blobs.append(np.frombuffer(b"".join(parts), dtype=np.uint8))
        recs.append({"path": f["path"], "size": os.path.getsize(p), "sha256": file_sha256(p, 64),
                     "sha256_expected": f.get("sha256"), "rows": int(t.num_rows)})
        del t, parts
    offsets = np.concatenate([[0], np.cumsum(np.concatenate(lens))]).astype(np.int64)
    blob = blobs[0] if len(blobs) == 1 else np.concatenate(blobs)
    return blob, offsets, np.concatenate(labels), recs


def _real_spec():
    """(size, sha256) of the pinned ReaL real.json (scripts/b1_data.py REAL)."""
    return _b1_data().REAL


def _real_sets(volume, n_rows):
    """The ReaL label sets (E11) when <data root>/real.json has the pinned size and sha256 and one list per row;
    else None. Never fatal."""
    p = os.path.join(_resolve(volume, "imagenet_val"), "real.json")
    size, sha = _real_spec()
    rec = {"path": p, "sha256": sha, "used": False}
    if os.path.isfile(p) and os.path.getsize(p) == size and file_sha256(p, 64) == sha:
        with open(p) as f:
            sets = json.load(f)
        if len(sets) == n_rows:
            rec["used"] = True
            return [set(s) for s in sets], rec
        rec["note"] = f"{len(sets)} lists for {n_rows} rows"
    else:
        rec["note"] = "missing or not the pinned file"
    return None, rec


def legacy_transform():
    """Upgraded-Mod imagenet_extract.py:26-27: Resize(256), CenterCrop(224), ToTensor, ImageNet mean/std."""
    import torchvision.transforms as T
    return T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(*IMAGENET_NORM)])


class _EncodedImages:
    """Map-style dataset over rows of the encoded-image blob -> (transformed tensor, row, decoded ok)."""

    def __init__(self, blob, offsets, rows, transform):
        self.blob, self.offsets, self.rows, self.tf = blob, offsets, np.asarray(rows, dtype=np.int64), transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import io
        from PIL import Image
        r = int(self.rows[i])
        ok = 1
        try:
            img = Image.open(io.BytesIO(self.blob[self.offsets[r]:self.offsets[r + 1]].tobytes())).convert("RGB")
        except Exception:                                              # counted; the row is dropped from its split
            img, ok = Image.new("RGB", (256, 256)), 0
        return self.tf(img), r, ok


# ---------------------------------------------------------------------------
# models and hooks
# ---------------------------------------------------------------------------
def load_imagenet_model(cfg, device):
    """Released ImageNet-1k weights -> (model in eval mode on device, source string, meta dict, official eval
    transform). torchvision: the weights enum (load_state_dict_from_url with check_hash=True); the file's full sha256
    must start with the hash in its URL. timm: model.safetensors at backbone.hf_revision through huggingface_hub; its
    sha256 must equal backbone.weights_sha256; timm loads that file (pretrained_cfg_overlay file=, which takes priority
    over the hub source, timm _builder._resolve_pretrained_source) and the classifier weights must equal the file's."""
    import torch
    bb = cfg["backbone"]
    spec = ARCHS[bb["arch"]]
    if spec["lib"] == "torchvision":
        import re
        import torchvision
        w = getattr(torchvision.models, spec["enum"])[bb["weights"]]
        model = getattr(torchvision.models, spec["ctor"])(weights=w)
        path = os.path.join(torch.hub.get_dir(), "checkpoints", os.path.basename(w.url))
        sha = file_sha256(path, 64)
        prefix = re.search(r"-([0-9a-f]+)\.pth$", w.url).group(1)
        if not sha.startswith(prefix):
            raise SystemExit(f"[extract_imagenet] {path}: sha256 {sha} does not start with the URL hash {prefix}")
        tf = w.transforms()
        info = {"weights_url": w.url, "weights_enum": f"{spec['enum']}.{bb['weights']}",
                "weights_acc1_published": float(w.meta["_metrics"]["ImageNet-1K"]["acc@1"])}
    else:
        import timm
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from timm.data import create_transform, resolve_model_data_config
        name = f"{spec['ctor']}.{bb['weights']}"
        path = hf_hub_download(repo_id=f"timm/{name}", filename="model.safetensors", revision=bb["hf_revision"])
        sha = file_sha256(path, 64)
        if sha != bb["weights_sha256"]:
            raise SystemExit(f"[extract_imagenet] {path}: sha256 {sha} != backbone.weights_sha256 {bb['weights_sha256']}")
        model = timm.create_model(name, pretrained=True, pretrained_cfg_overlay=dict(file=path))
        if not torch.equal(load_file(path)["head.weight"], model.head.weight.detach().cpu()):
            raise SystemExit(f"[extract_imagenet] {name}: head weights do not come from {path}")
        dc = resolve_model_data_config(model)
        tf = create_transform(**dc, is_training=False)
        info = {"weights_url": f"https://huggingface.co/timm/{name}/resolve/{bb['hf_revision']}/model.safetensors",
                "hf_repo": f"timm/{name}", "hf_revision": bb["hf_revision"], "weights_acc1_published": spec["acc1"],
                "timm_data_config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in dc.items()}}
    model.eval().to(device)
    return model, f"{spec['lib']} {bb['arch']} {bb['weights']} file:{path} sha256:{sha}", dict(info, weights_sha256=sha), tf


def vit_spec(lib, model):
    """Module names of the ViT taps. torchvision VisionTransformer (encoder.layers.encoder_layer_i, encoder.ln,
    heads.head; forward = heads(ln(...)[:, 0])) or timm VisionTransformer (blocks.i, norm, head; token pool)."""
    if lib == "torchvision":
        return {"blocks": [f"encoder.layers.encoder_layer_{i}" for i in range(len(model.encoder.layers))],
                "final_norm": "encoder.ln", "head": "heads.head", "n_prefix": 1}
    if lib == "timm":
        return {"blocks": [f"blocks.{i}" for i in range(len(model.blocks))], "final_norm": "norm", "head": "head",
                "n_prefix": int(model.num_prefix_tokens)}
    raise ValueError(f"vit_spec: {lib!r}")


class ViTHooks:
    """ViT taps in forward order: block.<i> = class token of encoder block i's output (every block_stride-th block and
    the last), penult_mean = mean of the patch tokens after the final LayerNorm (when listed in hooks.extra), penult =
    the input of the classifier Linear. final_cls keeps the final-norm class token of the last forward (not a tap:
    the penult assertion). forward / close as extract_acts.BlockHooks."""

    def __init__(self, model, lib, block_stride=1, extra=("penult_mean",)):
        spec = vit_spec(lib, model)
        names = dict(model.named_modules())
        missing = [n for n in [*spec["blocks"], spec["final_norm"], spec["head"]] if n not in names]
        if missing:
            raise ValueError(f"ViTHooks: modules not found: {missing}")
        self.model, self.acts, self.handles, self.layer_names, self.final_cls = model, {}, [], [], None
        npfx, L, mean_tap = int(spec["n_prefix"]), len(spec["blocks"]), "penult_mean" in (extra or ())
        for i, b in enumerate(spec["blocks"]):
            if i % max(1, int(block_stride)) == 0 or i == L - 1:
                self._reg(names[b], f"block.{i}")

        def _final(_, __, out):
            self.final_cls = out[:, 0].detach()
            if mean_tap:
                self.acts["penult_mean"] = out[:, npfx:].mean(1).detach()
        self.handles.append(names[spec["final_norm"]].register_forward_hook(_final))
        if mean_tap:
            self.layer_names.append("penult_mean")

        def _penult(_, inp, __):
            self.acts["penult"] = inp[0].detach()
        self.handles.append(names[spec["head"]].register_forward_hook(_penult))
        self.layer_names.append("penult")

    def _reg(self, mod, name):
        def hook(_, __, out):
            self.acts[name] = out[:, 0].detach()
        self.handles.append(mod.register_forward_hook(hook))
        self.layer_names.append(name)

    def forward(self, x):
        import torch
        self.acts, self.final_cls = {}, None
        with torch.no_grad():
            logits = self.model(x)
        return {k: v.float() for k, v in self.acts.items()}, logits.float()

    close = BlockHooks.close


def make_hooks(cfg, model):
    spec, hk = ARCHS[cfg["backbone"]["arch"]], cfg["hooks"]
    if spec["kind"] == "vit":
        return ViTHooks(model, spec["lib"], int(hk.get("block_stride", 1)), tuple(hk.get("extra") or ()))
    return BlockHooks(model, hk["layers"], int(hk.get("block_stride", 1)), hk["pooling"])


def _tf32():
    import torch
    rec = {}
    for k, get in (("matmul_allow_tf32", lambda: torch.backends.cuda.matmul.allow_tf32),
                   ("cudnn_allow_tf32", lambda: torch.backends.cudnn.allow_tf32),
                   ("float32_matmul_precision", torch.get_float32_matmul_precision)):
        try:
            rec[k] = get()
        except Exception as e:                                         # recorded, never fatal
            rec[k] = f"unavailable: {e!r}"[:120]
    return rec


def _versions():
    v = {"python": platform.python_version(), "numpy": np.__version__}
    for mod in ("torch", "torchvision", "timm", "pyarrow", "PIL", "huggingface_hub", "safetensors"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:
            v[mod] = None
    return v


# ---------------------------------------------------------------------------
# self-test on random inputs (no ImageNet image is read)
# ---------------------------------------------------------------------------
def selftest_random(cfg, device=None, model_override=None, size=224, n=4, seed=0):
    """Loads (downloads, hash-checks) the manifest's weights and runs torch.randn inputs through the hooks: the tap
    list (ViT: block.0 .. block.<L-1>, penult_mean, penult; 14 taps for the ViT-B/16 manifests), penult last, penult ==
    final-norm class token (max |d| < 1e-5), head check (max |penult W^T + b - logits| <= 1e-3), logits == model(x)
    (max |d| <= 1e-4), finite taps. model_override (tests only): (model, transform or None, info dict)."""
    import torch
    import torch.nn.functional as F
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    spec = ARCHS[cfg["backbone"]["arch"]]
    if model_override is None:
        model, src, winfo, _ = load_imagenet_model(cfg, device)
    else:
        model, _, winfo = model_override
        model, src = model.eval().to(device), "model_override (tests only)"
    hooks = make_hooks(cfg, model)
    head = dict(model.named_modules())[spec["head"]]
    x = torch.randn(n, 3, size, size, generator=torch.Generator().manual_seed(seed)).to(device)
    f, logits = hooks.forward(x)
    cls = getattr(hooks, "final_cls", None)                      # the final-norm class token of this forward
    cls_dev = float((cls.float() - f["penult"]).abs().max()) if cls is not None else None
    with torch.no_grad():
        again = model(x).float()
        hl = F.linear(f["penult"], head.weight.float(), None if head.bias is None else head.bias.float())
    names = list(hooks.layer_names)
    val = {"head_check_max_abs": float((hl - logits).abs().max()),
           "logits_vs_model_max_abs": float((logits - again).abs().max())}
    checks = {"taps_fired_in_order": list(f) == names, "penult_last": names[-1:] == ["penult"],
              "finite": all(bool(torch.isfinite(v).all()) for v in f.values()),
              "head_check": val["head_check_max_abs"] <= 1e-3, "logits_equal_model": val["logits_vs_model_max_abs"] <= 1e-4}
    if spec["kind"] == "vit":
        L = len(vit_spec(spec["lib"], model)["blocks"])
        want = [f"block.{i}" for i in range(L)] + ["penult_mean", "penult"]
        val["penult_vs_final_norm_cls_max_abs"] = cls_dev
        checks["tap_names"] = names == want
        checks["penult_is_final_norm_cls"] = (val["penult_vs_final_norm_cls_max_abs"] is not None
                                              and val["penult_vs_final_norm_cls_max_abs"] < 1e-5)
        if model_override is None:
            checks["n_taps_14"] = len(names) == 14
    hooks.close()
    ok = all(checks.values())
    return {"status": "PASS" if ok else "FAIL", "exp_id": cfg.get("exp_id"), "arch": cfg["backbone"]["arch"],
            "weights": src, **winfo, "input": f"torch.randn({n}, 3, {size}, {size}), generator seed {seed}; no ImageNet image",
            "taps": names, "dims": {k: int(v.shape[1]) for k, v in f.items()}, "checks": checks, "values": val,
            "device": str(device), "tf32": _tf32(), "versions": _versions(), "git_commit": git_commit(),
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}


# ---------------------------------------------------------------------------
# Stage A
# ---------------------------------------------------------------------------
def _write_split(dump, split, feats, pos, labels, preds, logits, dtype):
    for layer, arr in feats.items():
        os.makedirs(os.path.join(dump, "acts", layer), exist_ok=True)
        np.save(os.path.join(dump, "acts", layer, f"{split}.npy"), arr[pos].astype(dtype, copy=False))
    for sub in ("labels", "preds") + (("logits",) if logits is not None else ()):
        os.makedirs(os.path.join(dump, sub), exist_ok=True)
    np.save(os.path.join(dump, "labels", f"{split}.npy"), np.asarray(labels).astype(np.int64))
    np.savez(os.path.join(dump, "preds", f"{split}.npz"), **{k: v[pos] for k, v in preds.items()})
    if logits is not None:
        np.save(os.path.join(dump, "logits", f"{split}.npy"), logits[pos])


def _head_center_cos(Z, y, W):
    """mean over the classes present of cos(mu_k - mean_k mu_k, w_k - mean_k w_k): centers vs classifier rows
    (rule 8 H1, NC3 self-duality); INFO."""
    cls = np.unique(y)
    mu = np.stack([Z[y == c].mean(0) for c in cls]).astype(np.float64)
    Wc = np.asarray(W, dtype=np.float64)[cls]
    a, b = mu - mu.mean(0), Wc - Wc.mean(0)
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    return float(cos.mean())


def _counts(y, K):
    c = np.bincount(np.asarray(y, dtype=np.int64), minlength=K)
    return [int(c.min()), int(c.max())]


def extract_imagenet(cfg, volume, device=None, dump=None, model_override=None, quiet=False):
    """Stage A for one B1 manifest: one forward pass over every row that reference or clean_test needs (ascending
    row order); the splits are slices of it. No panel, corrupt, OOD or pixel-factor split. model_override (tests
    only): (model, transform or None, weights-info dict) replaces load_imagenet_model."""
    import torch
    import torch.nn.functional as F
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    bb, d, hk, ex = cfg["backbone"], cfg["data"], cfg["hooks"], cfg["extract"]
    K = int(cfg.get("n_classes", 1000))
    spec = ARCHS[bb["arch"]]
    if bb.get("norm") != "imagenet":
        raise SystemExit(f"[extract_imagenet] backbone.norm must be 'imagenet' (got {bb.get('norm')!r})")
    tname = bb.get("transform", "official")
    if tname not in ("official", "legacy_imagenet_extract"):
        raise SystemExit(f"[extract_imagenet] backbone.transform {tname!r}")
    if model_override is None:
        model, src, winfo, tf = load_imagenet_model(cfg, device)      # tf = the model's official eval transform
        if tname == "legacy_imagenet_extract":
            tf = legacy_transform()
    else:
        model, tf, winfo = model_override
        model, src = model.eval().to(device), "model_override (tests only)"
        if tf is None:
            tf = legacy_transform()
    hooks = make_hooks(cfg, model)
    head = dict(model.named_modules())[spec["head"]]
    if not quiet:
        print(f"[extract_imagenet] {bb['arch']} from {src}; {len(hooks.layer_names)} taps {hooks.layer_names}; "
              f"transform {tname}: {tf!r}")

    # ---- data and splits ----
    iv, ref_c, ct_c = d["imagenet_val"], d["reference"], d["clean_test"]
    blob, offsets, labels, frecs = load_imagenet_val(volume, iv["files"])
    if len(labels) == 0 or labels.min() < 0 or labels.max() >= K:
        raise SystemExit(f"[extract_imagenet] labels {labels.min()}..{labels.max()} outside 0..{K - 1}")
    legacy_loop = None
    if ref_c["part"] == "legacy_first":
        L, sizes = legacy_first_rows(volume, iv["files"], int(ref_c["n_target"]), int(ref_c["loop_batch"]))
        ref_idx = test_idx = np.arange(L, dtype=np.int64)
        split = {"rule": "legacy_first (imagenet_extract.py:40-60 replay; reference = clean_test, in-sample)",
                 "n_target": int(ref_c["n_target"]), "loop_batch": int(ref_c["loop_batch"])}
        legacy_loop = {"n_target": int(ref_c["n_target"]), "loop_batch": int(ref_c["loop_batch"]), "n_rows": int(L),
                       "flushes": int(L // int(ref_c["loop_batch"])), "n_parquet_batches": len(sizes),
                       "parquet_batch_rows": sorted(set(sizes)), "assumes_every_row_decodes": True}
    else:
        pc = int(ref_c["per_class"])
        cnt = np.bincount(labels, minlength=K)
        if (cnt != 2 * pc).any():
            raise SystemExit(f"[extract_imagenet] integrity: {len(labels)} rows, {cnt.min()}..{cnt.max()} per class "
                             f"(want {2 * pc} in each of {K})")
        A, B = split_parts(labels, int(ref_c["split_seed"]), pc, K)
        part = {"A": A, "B": B}
        ref_idx, test_idx = part[ref_c["part"]], part[ct_c["part"]]
        for nm, rows, c in (("reference", ref_idx, ref_c), ("clean_test", test_idx, ct_c)):
            if c.get("n") is not None and int(c["n"]) != len(rows):
                raise SystemExit(f"[extract_imagenet] data.{nm}.n {c['n']} != {len(rows)} rows of part {c['part']}")
        split = {"rule": "per class c = 0..K-1: rows ascending, default_rng(split_seed).permutation, first per_class = A",
                 "split_seed": int(ref_c["split_seed"]), "per_class": pc, "ref_part": ref_c["part"],
                 "test_part": ct_c["part"]}
    need = np.union1d(ref_idx, test_idx)
    real_sets, real_rec = _real_sets(volume, len(labels))

    # ---- forward ----
    workers = int(os.environ.get("ATLAS_EXTRACT_WORKERS", ex.get("num_workers", 8)))
    npdt = {"float16": np.float16, "float32": np.float32}[ex.get("dtype", "float32")]
    dl = torch.utils.data.DataLoader(_EncodedImages(blob, offsets, need, tf), batch_size=int(ex["batch"]),
                                     shuffle=False, num_workers=workers, pin_memory=str(device).startswith("cuda"))
    N = len(need)
    feats, absmax, logits_all = {}, {}, None
    preds = {"argmax": np.empty(N, np.int64), "maxprob": np.empty(N, np.float32), "logit_gap": np.empty(N, np.float32),
             "logit_top1": np.empty(N, np.float32), "logit_top2": np.empty(N, np.float32)}
    seen, okm = np.empty(N, np.int64), np.empty(N, bool)
    head_dev, cls_dev, i = 0.0, 0.0, 0
    for x, r, ok in dl:
        f, logits = hooks.forward(x.to(device, non_blocking=True))
        n = len(r)
        with torch.no_grad():
            p = torch.softmax(logits, 1)
            top = logits.topk(2, dim=1).values
            hl = F.linear(f["penult"], head.weight.float(), None if head.bias is None else head.bias.float())
        head_dev = max(head_dev, float((hl - logits).abs().max()))
        if getattr(hooks, "final_cls", None) is not None:
            cls_dev = max(cls_dev, float((hooks.final_cls.float() - f["penult"]).abs().max()))
        preds["argmax"][i:i + n] = p.argmax(1).cpu().numpy()
        preds["maxprob"][i:i + n] = p.max(1).values.cpu().numpy()
        preds["logit_top1"][i:i + n] = top[:, 0].cpu().numpy()
        preds["logit_top2"][i:i + n] = top[:, 1].cpu().numpy()
        preds["logit_gap"][i:i + n] = (top[:, 0] - top[:, 1]).cpu().numpy()
        if ex.get("store_logits"):
            if logits_all is None:
                logits_all = np.empty((N, logits.shape[1]), np.float16)
            logits_all[i:i + n] = logits.cpu().numpy().astype(np.float16)
        for k, v in f.items():
            a = v.cpu().numpy()
            if k not in feats:
                feats[k] = np.empty((N, a.shape[1]), dtype=npdt)
            feats[k][i:i + n] = a
            absmax[k] = max(absmax.get(k, 0.0), float(np.abs(a).max()))
        seen[i:i + n], okm[i:i + n] = r.numpy(), ok.numpy().astype(bool)
        i += n
    if i != N or not np.array_equal(seen, need):
        raise RuntimeError("[extract_imagenet] the DataLoader changed or dropped rows")
    if list(feats) != hooks.layer_names:
        raise RuntimeError(f"[extract_imagenet] taps fired {list(feats)}, registered {hooks.layer_names}")
    for k, v in feats.items():
        if not np.isfinite(v).all():
            raise RuntimeError(f"[extract_imagenet] tap {k}: non-finite values (max |x| {absmax[k]:.4g})")
    if real_sets is not None:
        preds["real_ok"] = np.array([-1 if not real_sets[r] else int(int(a) in real_sets[r])
                                     for r, a in zip(need, preds["argmax"])], dtype=np.int8)
    failed = need[~okm]
    if not quiet:
        print(f"[extract_imagenet] forward {N} rows, {len(failed)} decode failures, {workers} loader workers "
              f"({time.time() - t0:.0f}s)")

    # ---- dump ----
    dump = dump or os.path.join(cfg["outputs"]["root"], "dump")
    os.makedirs(dump, exist_ok=True)
    acc, rows_of = {}, {}
    for split_name, rows in (("ref", ref_idx), ("test", test_idx)):
        rows = rows[~np.isin(rows, failed)]
        pos = np.searchsorted(need, rows)
        _write_split(dump, split_name, feats, pos, labels[rows], preds, logits_all, npdt)
        acc[split_name] = float((preds["argmax"][pos] == labels[rows]).mean()) if len(rows) else None
        rows_of[split_name] = rows
        if not quiet:
            print(f"[extract_imagenet] {split_name:5s} n={len(rows):6d} acc={acc[split_name]}  ({time.time() - t0:.0f}s)")
    ref_rows, test_rows = rows_of["ref"], rows_of["test"]
    W = head.weight.detach().float().cpu().numpy()
    meta = {
        "source": "real", "exp_id": cfg.get("exp_id"), "arch": bb["arch"], "weights": src, "seed_tag": bb.get("seed_tag"),
        "layers": list(hooks.layer_names), "dims": {k: int(v.shape[1]) for k, v in feats.items()},
        "splits": ["ref", "test"], "n_classes": K, "dtype": ex.get("dtype", "float32"), "pooling": hk.get("pooling"),
        "norm": "imagenet",
        "norm_values": [list((winfo.get("timm_data_config") or {}).get("mean", IMAGENET_NORM[0])),
                        list((winfo.get("timm_data_config") or {}).get("std", IMAGENET_NORM[1]))],
        "transform": tname, "transform_repr": repr(tf), "loader_workers": workers, "batch": int(ex["batch"]),
        "pairing": "none (no corrupt splits)", "split": split, "legacy_loop": legacy_loop,
        "ref_indices": ref_rows.tolist(), "test_indices": test_rows.tolist(), "panel_indices": [],
        "n_ref": int(len(ref_rows)), "n_test": int(len(test_rows)),
        "per_class_counts": {"ref": _counts(labels[ref_rows], K), "test": _counts(labels[test_rows], K)},
        "ref_test_disjoint": bool(np.intersect1d(ref_rows, test_rows).size == 0),
        "decode_failures": int(len(failed)), "decode_failed_rows": failed.tolist(),
        "parquet": {"repo": iv.get("repo"), "revision": iv.get("revision"), "files": frecs},
        "parquet_sha256_ok": bool(frecs) and all(r["sha256"] == r["sha256_expected"] for r in frecs),
        "n_rows_dataset": int(len(labels)),
        "head_check_max_abs": head_dev, "head_check_rows": "every extracted row (ref and test), float32, before storage",
        "penult_vs_final_norm_cls_max_abs": cls_dev if spec["kind"] == "vit" else None,
        "head_center_cos": _head_center_cos(feats["penult"][np.searchsorted(need, ref_rows)], labels[ref_rows], W)
        if len(ref_rows) else None,
        "accuracy": acc, "act_absmax": absmax, "extra_preds": sorted(k for k in preds if k not in ("argmax", "maxprob")),
        "logits_stored": logits_all is not None, "real_labels": real_rec, "tf32": _tf32(), "versions": _versions(),
        **winfo, "device": str(device), "git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(dump, META_NAME), "w") as fh:                   # written last: marks a complete dump
        json.dump(meta, fh, indent=2)
    hooks.close()
    if not quiet:
        print(f"[extract_imagenet] dump complete at {dump} ({time.time() - t0:.0f}s)")
    return dump
