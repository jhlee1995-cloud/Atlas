"""
registry.py -- three small registries behind decorators.

    @invariant(name, needs=("labels",), cost="cheap")
    def my_invariant(ctx, cfg) -> dict

    @cross_layer(name)
    def my_cross(ctxs, per_layer, cfg) -> dict

    @factor(name, kind="continuous"|"categorical", pool="mixed"|"clean"|"corrupt")
    def my_factor(images_uint8) -> np.ndarray (N,)

`needs` declares which parts of the LayerContext an invariant reads, so the builder can
skip it cleanly (and say why) when a dump lacks that part (e.g. no CIFAR-10-C sets).
`cost` is a hint the agent loop uses to order cheap-before-expensive (kill-switch sequencing).

Registries are plain dicts; the manifest picks by name or "all". Import atlas.invariants /
atlas.factors to populate them (they register on import).
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple

INVARIANTS: Dict[str, "Spec"] = {}
CROSS_LAYER: Dict[str, "Spec"] = {}
FACTORS: Dict[str, "FactorSpec"] = {}


@dataclass
class Spec:
    name: str
    fn: Callable
    needs: Tuple[str, ...] = ()
    cost: str = "cheap"          # cheap | medium | expensive
    doc: str = ""


@dataclass
class FactorSpec:
    name: str
    fn: Callable                 # pixels: fn(images_uint8) -> (N,);  meta: fn(split, labels) -> (N,)
    kind: str = "continuous"     # continuous (ridge R^2) | categorical (logistic acc)
    pool: str = "mixed"          # which split the probe trains on: mixed | clean | corrupt
    doc: str = ""
    n_classes: int = 0           # categorical only (0 = infer)
    source: str = "pixels"       # pixels (computed in Stage A) | meta (derived from split/labels in Stage B)


def invariant(name, needs=(), cost="cheap"):
    def deco(fn):
        if name in INVARIANTS:
            raise KeyError(f"invariant '{name}' registered twice")
        INVARIANTS[name] = Spec(name, fn, tuple(needs), cost, (fn.__doc__ or "").strip())
        return fn
    return deco


def cross_layer(name, needs=(), cost="cheap"):
    def deco(fn):
        if name in CROSS_LAYER:
            raise KeyError(f"cross-layer invariant '{name}' registered twice")
        CROSS_LAYER[name] = Spec(name, fn, tuple(needs), cost, (fn.__doc__ or "").strip())
        return fn
    return deco


def factor(name, kind="continuous", pool="mixed", n_classes=0, source="pixels"):
    def deco(fn):
        if name in FACTORS:
            raise KeyError(f"factor '{name}' registered twice")
        FACTORS[name] = FactorSpec(name, fn, kind, pool, (fn.__doc__ or "").strip(),
                                   n_classes, source)
        return fn
    return deco


def select(registry, wanted):
    """wanted: "all" | list of names | list with "-name" exclusions."""
    if wanted in (None, "all", ["all"]):
        return dict(registry)
    out = {}
    excl = {w[1:] for w in wanted if isinstance(w, str) and w.startswith("-")}
    incl = [w for w in wanted if not (isinstance(w, str) and w.startswith("-"))]
    if not incl:
        incl = list(registry)
    for w in incl:
        if w not in registry:
            raise KeyError(f"'{w}' not registered; known: {sorted(registry)}")
        if w not in excl:
            out[w] = registry[w]
    return out


def describe():
    lines = ["per-layer invariants:"]
    for s in INVARIANTS.values():
        lines.append(f"  {s.name:24s} cost={s.cost:9s} needs={list(s.needs)}")
    lines.append("cross-layer invariants:")
    for s in CROSS_LAYER.values():
        lines.append(f"  {s.name:24s} cost={s.cost:9s} needs={list(s.needs)}")
    lines.append("factors:")
    for f in FACTORS.values():
        lines.append(f"  {f.name:24s} kind={f.kind:11s} pool={f.pool}")
    return "\n".join(lines)
