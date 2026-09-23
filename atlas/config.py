"""
config.py -- manifest (YAML) -> dict with defaults filled in.

A manifest is the unit of work the agent loop moves through experiments/queue/. Keep it
declarative: what to extract, which invariants to run, which holdouts to keep. Anything
computed lands under outputs.root; the manifest itself is copied there as manifest_used.yaml.
"""
import copy
import os

import yaml

DEFAULTS = {
    "exp_id": None,
    "stage": "atlas",
    "backbone": {"arch": "cifar10_resnet20", "weights": "hub", "seed_tag": "s0"},
    "data": {
        "reference": {"dataset": "cifar10_train", "n": 10000, "seed": 0},
        "clean_test": {"dataset": "cifar10", "n": 5000},
        "panel": {"dataset": "cifar10_train", "n": 64, "seed": 123},
        "cifar10c": {"corruptions": "all", "severities": [1, 3, 5], "n_per_set": 2000},
        "ood": [],
    },
    "hooks": {"layers": "blocks", "block_stride": 1, "pooling": "gap"},
    "extract": {"batch": 256, "dtype": "float16", "num_workers": 4},
    "invariants": "all",
    "cross_layer": "all",
    "invariant_cfg": {},          # per-invariant overrides, e.g. {"linear_probes": {"n_train": 4000}}
    "probes": {"n_train": 4000, "cv_folds": 5, "factors": "all"},
    "holdout": {"discovery_corruptions": None, "confirmation_corruptions": None,
                "discovery_seeds": None, "confirmation_seeds": None},
    "outputs": {"root": None, "plots": True},
    "notes": "",
}


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_manifest(path):
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    cfg = _merge(DEFAULTS, raw)
    if not cfg["exp_id"]:
        cfg["exp_id"] = os.path.splitext(os.path.basename(path))[0]
    if not cfg["outputs"]["root"]:
        cfg["outputs"]["root"] = os.path.join("results", cfg["exp_id"])
    # probes config is the default invariant_cfg for linear_probes unless overridden
    ic = cfg["invariant_cfg"]
    ic.setdefault("linear_probes", {})
    for k, v in cfg["probes"].items():
        ic["linear_probes"].setdefault(k, v)
    cfg["_manifest_path"] = os.path.abspath(path)
    return cfg


def dump_manifest(cfg, path):
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with open(path, "w") as f:
        yaml.safe_dump(clean, f, sort_keys=False)
