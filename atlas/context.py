"""
context.py -- the DUMP contract between Stage A and Stage B, and the LayerContext
every invariant receives.

Dump layout (written by extract_acts.py or synth.py):

  <dump>/meta.json                       arch, weights, layers, dims, splits, dtype, seeds
  <dump>/labels/<split>.npy              int labels  (split in: ref, test, panel, corrupt__X__sN)
  <dump>/preds/<split>.npz               argmax, maxprob  (from the backbone's logits)
  <dump>/acts/<layer>/<split>.npy        (N, D) pooled activations (float16 on disk)
  <dump>/factors/<split>.npz             input-side factors {name: (N,)} from raw pixels

Splits:
  ref      clean TRAIN subset      = the reference distribution the atlas is OF
  test     clean TEST subset       = held-out clean, paired by index with corrupt sets
  panel    fixed tiny clean set    = "self-antigen panel" for atlas-vs-atlas deformation
  corrupt__<corruption>__s<sev>    = CIFAR-10-C set, rows paired with test[:N]
  ood__<name>                      = far/near-OOD sets (no pairing, labels dummy)

Stage B never knows whether a dump is real or synthetic (meta.source says which);
the critic refuses to promote anything measured on a synthetic dump.
"""
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

META_NAME = "meta.json"


def corrupt_split(corruption: str, severity: int) -> str:
    return f"corrupt__{corruption}__s{int(severity)}"


def parse_corrupt_split(split: str):
    """'corrupt__fog__s3' -> ('fog', 3); None if not a corrupt split."""
    if not split.startswith("corrupt__"):
        return None
    _, c, s = split.split("__")
    return c, int(s[1:])


class AtlasDump:
    """Lazy reader over a dump directory."""

    def __init__(self, root: str):
        self.root = root
        mp = os.path.join(root, META_NAME)
        if not os.path.exists(mp):
            raise FileNotFoundError(f"no {META_NAME} under {root}")
        self.meta = json.load(open(mp))
        self.layers: List[str] = self.meta["layers"]
        self.splits: List[str] = self.meta["splits"]
        self.n_classes: int = int(self.meta.get("n_classes", 10))
        self.source: str = self.meta.get("source", "UNKNOWN")
        self._acts = {}
        self._labels = {}
        self._factors = {}
        self._preds = {}

    # ---- accessors -------------------------------------------------------
    def acts(self, layer: str, split: str) -> np.ndarray:
        key = (layer, split)
        if key not in self._acts:
            p = os.path.join(self.root, "acts", layer, f"{split}.npy")
            self._acts[key] = np.load(p).astype(np.float32)
        return self._acts[key]

    def has(self, layer: str, split: str) -> bool:
        return os.path.exists(os.path.join(self.root, "acts", layer, f"{split}.npy"))

    def labels(self, split: str) -> np.ndarray:
        if split not in self._labels:
            self._labels[split] = np.load(os.path.join(self.root, "labels", f"{split}.npy"))
        return self._labels[split]

    def preds(self, split: str) -> Optional[dict]:
        p = os.path.join(self.root, "preds", f"{split}.npz")
        if not os.path.exists(p):
            return None
        if split not in self._preds:
            z = np.load(p)
            self._preds[split] = {k: z[k] for k in z.files}
        return self._preds[split]

    def factors(self, split: str) -> Dict[str, np.ndarray]:
        p = os.path.join(self.root, "factors", f"{split}.npz")
        if not os.path.exists(p):
            return {}
        if split not in self._factors:
            z = np.load(p)
            self._factors[split] = {k: z[k] for k in z.files}
        return self._factors[split]

    def corrupt_splits(self) -> List[str]:
        return [s for s in self.splits if s.startswith("corrupt__")]

    def ood_splits(self) -> List[str]:
        return [s for s in self.splits if s.startswith("ood__")]

    def drop_cache(self):
        self._acts.clear()


@dataclass
class LayerContext:
    """Everything an invariant may read for ONE layer. Missing parts are None/{}."""
    layer: str
    n_classes: int
    ref: np.ndarray                      # (N_ref, D)
    ref_labels: np.ndarray
    test: Optional[np.ndarray] = None    # (N_test, D)
    test_labels: Optional[np.ndarray] = None
    test_preds: Optional[dict] = None    # {"argmax":..., "maxprob":...}
    panel: Optional[np.ndarray] = None   # (N_panel, D)
    corrupt: Dict[str, np.ndarray] = field(default_factory=dict)   # split -> (N_c, D)
    corrupt_labels: Dict[str, np.ndarray] = field(default_factory=dict)
    ood: Dict[str, np.ndarray] = field(default_factory=dict)
    factors: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)  # split -> {name: (N,)}
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    source: str = "UNKNOWN"

    @property
    def dim(self) -> int:
        return int(self.ref.shape[1])

    def available(self) -> set:
        parts = {"ref", "labels"}
        if self.test is not None:
            parts.add("test")
        if self.panel is not None:
            parts.add("panel")
        if self.corrupt:
            parts.add("corrupt")
        if self.ood:
            parts.add("ood")
        if self.factors:
            parts.add("factors")
        return parts


def build_context(dump: AtlasDump, layer: str, seed: int = 0) -> LayerContext:
    ctx = LayerContext(
        layer=layer,
        n_classes=dump.n_classes,
        ref=dump.acts(layer, "ref"),
        ref_labels=dump.labels("ref"),
        rng=np.random.default_rng(seed),
        source=dump.source,
    )
    if dump.has(layer, "test"):
        ctx.test = dump.acts(layer, "test")
        ctx.test_labels = dump.labels("test")
        ctx.test_preds = dump.preds("test")
    if dump.has(layer, "panel"):
        ctx.panel = dump.acts(layer, "panel")
    for s in dump.corrupt_splits():
        if dump.has(layer, s):
            ctx.corrupt[s] = dump.acts(layer, s)
            ctx.corrupt_labels[s] = dump.labels(s)
    for s in dump.ood_splits():
        if dump.has(layer, s):
            ctx.ood[s] = dump.acts(layer, s)
    for s in ["ref", "test"] + dump.corrupt_splits():
        f = dump.factors(s)
        if f:
            ctx.factors[s] = f
    return ctx


# ---- JSON helpers -----------------------------------------------------------
def to_jsonable(obj):
    """Recursively convert numpy types so results serialize; arrays -> lists."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj
