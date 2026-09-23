"""
tests/test_atlas_smoke.py -- CPU-only checks (no torch).

  1. registry populated, every invariant is import + callable
  2. synthetic dump -> build -> atlas.json with zero errors and zero skips
  3. known-answer checks: TwoNN on a 3-d Gaussian in 20-d ~ 3; CKA(X, X) = 1;
     paired displacement is monotone in severity; class probe commits late in the synthetic
  4. compare + critic run and the critic refuses to promote synthetic runs

Run: python -m pytest tests/test_atlas_smoke.py -v   (or: python tests/test_atlas_smoke.py)
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas import invariants as _inv  # noqa: F401,E402
from atlas import factors as _fac      # noqa: F401,E402
from atlas.registry import INVARIANTS, CROSS_LAYER, FACTORS  # noqa: E402
from atlas.invariants.dimension import twonn  # noqa: E402
from atlas.invariants._util import linear_cka  # noqa: E402
from atlas.synth import make_synth_dump  # noqa: E402
from atlas.build import build_atlas  # noqa: E402
from atlas.compare import compare  # noqa: E402
from atlas.critic import run_critic  # noqa: E402

_TMP = tempfile.mkdtemp(prefix="atlas_test_")


_RUNS = {}


def _run(seed):
    """Synthetic dump + atlas, built once per seed (tests only read the result or deep-copy it)."""
    if seed not in _RUNS:
        root = os.path.join(_TMP, f"r{seed}")
        make_synth_dump(os.path.join(root, "dump"), n_ref=2000, n_test=800, n_corr=400, seed=seed)
        _RUNS[seed] = (root, build_atlas(os.path.join(root, "dump"), root, {}, verbose=False))
    return _RUNS[seed]


def test_registry():
    assert len(INVARIANTS) >= 9 and len(CROSS_LAYER) >= 2 and len(FACTORS) >= 15
    for s in list(INVARIANTS.values()) + list(CROSS_LAYER.values()):
        assert callable(s.fn), s.name


def test_twonn_known_answer():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(4000, 3))
    A = rng.normal(size=(3, 20))
    X = Z @ A
    d = twonn(X)
    assert 2.4 < d < 3.6, d


def test_cka_identity():
    X = np.random.default_rng(1).normal(size=(500, 8))
    assert abs(linear_cka(X, X) - 1.0) < 1e-6
    R = np.linalg.qr(np.random.default_rng(2).normal(size=(8, 8)))[0]
    assert abs(linear_cka(X, X @ R) - 1.0) < 1e-6      # rotation invariant


def test_pipeline_no_errors():
    root, atlas = _run(0)
    errs = [(l, k) for l, p in atlas["per_layer"].items() for k, v in p.items()
            if isinstance(v, dict) and "error" in v]
    assert not errs, errs
    assert not atlas["skipped"], atlas["skipped"]
    assert os.path.exists(os.path.join(root, "atlas.json"))


def test_synthetic_structure_recovered():
    root, atlas = _run(0)
    layers = atlas["layers"]
    # class commits late: probe excess at stem < at penult
    lp = lambda l, f: atlas["per_layer"][l]["linear_probes"]["factors"][f]
    assert lp(layers[0], "class")["excess"] < lp(layers[-1], "class")["excess"] - 0.2
    # luminance washes out: R^2 at penult < at layer1
    assert lp(layers[-1], "luminance_mean")["score"] < lp(layers[1], "luminance_mean")["score"]
    # displacement monotone in severity for brightness at stem
    d = atlas["per_layer"][layers[0]]["corruption_displacement"]["splits"]
    m = [d[f"corrupt__brightness__s{s}"]["magnitude"] for s in (1, 3, 5)]
    assert m[0] < m[1] < m[2], m
    # commit layer computed for class
    assert atlas["cross_layer"]["commit_layer"]["per_factor"]["class"]["commit_layer"] in layers


def test_compare_and_critic():
    r0, _ = _run(0)
    r1, _ = _run(1)
    res = compare(r0, r1)
    assert "penult" in res["per_layer"] and "panel_cka" in res["per_layer"]["penult"]
    rep = run_critic([r0, r1])
    assert rep["verdict"] == "PIPELINE_CHECK_ONLY"      # synthetic: never promotable
    assert rep["counts"]["PASS"] == 0


def test_compare_cross_model_known_answers():
    r0, _ = _run(0)
    res = compare(r0, r0)["per_layer"]["penult"]         # a model against itself
    assert abs(res["cka_test"] - 1.0) < 1e-6
    assert res["relrep_argmax_agree_test"] == 1.0
    assert res["relrep_offmax_corr_test"] > 0.999
    assert res["error_consistency_test"] is None or abs(res["error_consistency_test"] - 1.0) < 1e-9
    assert res["adjacency_spearman"] > 0.999


def test_critic_alias_and_id_profile():
    import copy
    from atlas.critic import DEFAULT_TOL, _aliases, id_profile_stability
    runs = []
    for seed in (0, 1):
        d, a = _run(seed)
        a = copy.deepcopy(a)
        a["per_layer"]["layer3.1"] = copy.deepcopy(a["per_layer"]["penult"])   # the gap-pooling alias
        runs.append({"dir": d, "atlas": a, "manifest": {}})
    kept, alias = _aliases(runs)
    assert alias == {"layer3.1": "penult"} and "penult" in kept and "layer3.1" not in kept

    def mk(ids):
        names = [f"l{i}" for i in range(len(ids))]
        return {"dir": "x", "manifest": {},
                "atlas": {"layers": names, "per_layer": {n: {"twonn_id": {"id": v}} for n, v in zip(names, ids)}}}
    prof = [3.0, 5.0, 8.0, 12.0, 9.0, 6.0]
    assert id_profile_stability([mk(prof), mk(prof)], DEFAULT_TOL)[0]["status"] == "PASS"
    assert id_profile_stability([mk(prof), mk(prof[::-1])], DEFAULT_TOL)[0]["status"] == "FAIL"



def _deformed_copy(src, dst, eps, seed=0):
    """Copy a dump and apply a small linear deformation X -> X(I + eps R) + eps*s to every split."""
    import glob
    shutil.copytree(src, dst)
    rng = np.random.default_rng(seed)
    meta = json.load(open(os.path.join(dst, "meta.json")))
    for layer in meta["layers"]:
        files = glob.glob(os.path.join(dst, "acts", layer, "*.npy"))
        d = np.load(files[0]).shape[1]
        R = rng.normal(size=(d, d)) / np.sqrt(d)
        s = rng.normal(size=d)
        for fpath in files:
            X = np.load(fpath).astype(np.float32)
            np.save(fpath, (X @ (np.eye(d) + eps * R) + eps * s).astype(np.float16))
    meta["ground_truth"] = {"acc_stream_heldout": 0.7 - eps, "acc_clean": 0.9 - 2 * eps,
                            "acc_heldout_corr": 0.6 - eps, "pred_entropy": 1.0 - 3 * eps}
    json.dump(meta, open(os.path.join(dst, "meta.json"), "w"))


def test_ladder_monotone():
    from atlas.ladder import run_ladder
    exp = os.path.join(_TMP, "ladder")
    os.makedirs(exp, exist_ok=True)
    make_synth_dump(os.path.join(exp, "dump_step0"), n_ref=1500, n_test=600, n_corr=300, seed=3)
    meta0 = json.load(open(os.path.join(exp, "dump_step0", "meta.json")))
    meta0["ground_truth"] = {"acc_stream_heldout": 0.7, "acc_clean": 0.9, "acc_heldout_corr": 0.6, "pred_entropy": 1.0}
    json.dump(meta0, open(os.path.join(exp, "dump_step0", "meta.json"), "w"))
    for step, eps in ((10, 0.05), (50, 0.2)):
        _deformed_copy(os.path.join(exp, "dump_step0"), os.path.join(exp, f"dump_step{step}"), eps, seed=step)
    res = run_ladder(exp, layer="penult", tol=0.02)
    rows = {r["step"]: r for r in res["rows"]}
    assert rows[10]["panel_shift"] < rows[50]["panel_shift"]          # dose-response
    assert rows[10]["panel_cka"] > rows[50]["panel_cka"]
    assert res["onset_step"]["panel_shift"] is not None
    assert os.path.exists(os.path.join(exp, "LADDER.md"))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    shutil.rmtree(_TMP, ignore_errors=True)
