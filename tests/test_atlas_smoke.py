"""
tests/test_atlas_smoke.py -- CPU-only checks (no torch).

  1. registry populated, every invariant is import + callable
  2. synthetic dump -> build -> atlas.json with zero errors and zero skips
  3. known-answer checks: TwoNN on a 3-d Gaussian in 20-d ~ 3; CKA(X, X) = 1;
     paired displacement is monotone in severity; class probe commits late in the synthetic
  4. compare + critic run and the critic refuses to promote synthetic runs
  5. A3 margin_typeb: ridge vs deep-valley known answers, the strict cut, empty groups (None, no error), no
     ctx.rng draws, DeLong = Mann-Whitney; critic per-key absolute tolerance
  6. A2b scripts/check_rebuild.py leaf comparison; A4 compare cross-depth guard and critic --align position

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


# ---- A3: margin_typeb (atlas/invariants/margin.py) ----------------------------------------------------------
def _margin_ctx(kind, seed=0):
    """Known geometry: 3 classes on a right triangle in 8-d (c0-c1 and c0-c2 6 apart). Correct test points sit
    near their own center. 100 type-b points (label 0, maxprob 0.75) sit on the c0-c1 ridge (kind 'ridge') or
    deep in the wrong valley c2 (kind 'deep'); 50 low-confidence wrong points (label 1, maxprob 0.5, not type-b
    at cut 0.7) sit on the ridge. maxprob is stored as float32, as in a dump (0.75 and 0.5 are exact)."""
    from atlas.context import LayerContext
    rng = np.random.default_rng(seed)
    C = np.zeros((3, 8))
    C[1, 0] = C[2, 1] = 6.0
    ref_y = np.repeat(np.arange(3), 300)
    ref = C[ref_y] + 0.5 * rng.normal(size=(900, 8))
    y_ok = np.repeat(np.arange(3), 200)
    X_ok = C[y_ok] + 0.5 * rng.normal(size=(600, 8))
    ridge = 0.5 * (C[0] + C[1])
    if kind == "ridge":
        X_b, am_b = ridge + 0.3 * rng.normal(size=(100, 8)), np.ones(100, dtype=np.int64)
    else:
        X_b, am_b = C[2] + 0.5 * rng.normal(size=(100, 8)), np.full(100, 2, dtype=np.int64)
    X_l = ridge + 0.3 * rng.normal(size=(50, 8))
    X = np.vstack([X_ok, X_b, X_l]).astype(np.float32)
    y = np.concatenate([y_ok, np.zeros(100, dtype=np.int64), np.ones(50, dtype=np.int64)])
    am = np.concatenate([y_ok, am_b, np.zeros(50, dtype=np.int64)])
    mp = np.concatenate([np.full(600, 0.99), np.full(100, 0.75), np.full(50, 0.5)]).astype(np.float32)
    return LayerContext(layer="penult", n_classes=3, ref=ref.astype(np.float32), ref_labels=ref_y,
                        test=X, test_labels=y, test_preds={"argmax": am, "maxprob": mp})


def test_margin_typeb_known_answer():
    from atlas.invariants.margin import center_dists, margin_typeb, top2_margin
    m, d = top2_margin(center_dists(np.array([[3.0, 0.0], [0.0, 0.0]]),
                                    np.array([[0.0, 0.0], [6.0, 0.0], [0.0, 6.0]])))
    assert np.allclose(m, [0.0, 6.0]) and np.allclose(d, [3.0, 0.0])
    r = margin_typeb(_margin_ctx("ridge"), {})
    assert (r["n_test"], r["n_wrong"], r["n_typeb"]) == (750, 150, 100)      # strict maxprob > 0.7
    assert [s["n_typeb"] for s in r["sweep"]] == [150, 100, 100, 100, 0, 0, 0, 0]
    assert r["sweep"][4]["auc_margin_typeb"] is None                        # cut 0.8: no type-b left
    assert r["auc_margin_typeb"] > 0.99 and r["median_margin_ratio_typeb"] < 0.2   # ridge = small margin
    assert r["legacy"]["dir_auc_margin"] > 0.95 and r["legacy"]["raw_auc_margin"] < 0.05
    dp = margin_typeb(_margin_ctx("deep"), {})
    assert 0.35 < dp["auc_margin_typeb"] < 0.65                             # deep wrong valley: margin is blind
    assert 0.35 < dp["auc_dist_typeb"] < 0.65


def test_margin_typeb_empty_groups():
    """No confident mistakes (a random-init null may have none) or no mistakes at all: None, never an error."""
    from atlas.invariants.margin import margin_typeb
    ctx = _margin_ctx("ridge")
    ctx.test_preds = {"argmax": ctx.test_preds["argmax"], "maxprob": np.full(750, 0.5, dtype=np.float32)}
    r = margin_typeb(ctx, {})
    assert (r["n_wrong"], r["n_typeb"]) == (150, 0)
    assert r["auc_margin_typeb"] is None and r["margin_minus_dist_typeb"] is None
    assert r["median_margin_typeb"] is None and r["median_margin_ratio_typeb"] is None
    assert r["auc_margin_wrong"] is not None and r["legacy"]["dir_auc_margin"] is None
    ctx.test_preds = {"argmax": ctx.test_labels.copy(), "maxprob": np.full(750, 0.99, dtype=np.float32)}
    r = margin_typeb(ctx, {})
    assert r["n_wrong"] == 0 and r["typeb_frac_of_wrong"] is None and r["auc_margin_wrong"] is None
    assert all(s["auc_margin_typeb"] is None for s in r["sweep"])


def test_margin_typeb_rng_neutral():
    """margin_typeb draws nothing from ctx.rng and is registered last (cost expensive), so adding it cannot move
    any other invariant's estimator draws (docs/plans/STAGE1.md amendment 2)."""
    from atlas.invariants.margin import margin_typeb
    ctx = _margin_ctx("ridge")
    before = ctx.rng.bit_generator.state
    margin_typeb(ctx, {})
    assert ctx.rng.bit_generator.state == before
    assert list(INVARIANTS)[-1] == "margin_typeb" and INVARIANTS["margin_typeb"].cost == "expensive"


def test_delong_matches_mann_whitney():
    from sklearn.metrics import roc_auc_score
    from atlas.invariants.margin import delong
    rng = np.random.default_rng(3)
    neg, pos = rng.normal(size=300), rng.normal(0.7, 1.0, size=120)
    pos[:10] = neg[:10]                                                     # exact ties count 1/2
    auc, cov = delong(pos[None], neg[None])
    assert abs(auc[0] - roc_auc_score(np.r_[np.zeros(300), np.ones(120)], np.r_[neg, pos])) < 1e-9
    assert 0 < cov[0, 0] < 0.01
    auc, cov = delong(np.array([[2.0, 3.0]]), np.array([[0.0, 1.0]]))
    assert auc[0] == 1.0 and cov[0, 0] == 0.0


def test_margin_typeb_on_synth():
    _, atlas = _run(0)
    for l in atlas["layers"]:
        m = atlas["per_layer"][l]["margin_typeb"]
        assert "error" not in m, m
        assert m["n_test"] == 800 and len(m["sweep"]) == 8 and m["n_typeb"] <= m["n_wrong"]
        for key in ("auc_margin_typeb", "auc_dist_typeb", "auc_maxprob_typeb", "auc_margin_wrong",
                    "median_margin_ratio_typeb", "margin_minus_dist_typeb_p"):
            assert key in m, key                                          # compare.SCALARS keys exist
        assert "dir_auc_margin" in m["legacy"]


def test_critic_scalar_abs_tol():
    from atlas.critic import DEFAULT_TOL, scalar_stability

    def run(v):
        return {"dir": "x", "manifest": {}, "atlas": {"layers": ["penult"], "per_layer": {"penult": {
            "margin_typeb": {"auc_margin_typeb": v}, "twonn_id": {"id": 10 * v}}}}}
    tol = {**DEFAULT_TOL, "scalar_abs_tol": {"margin_typeb.auc_margin_typeb": 0.05}}
    items = lambda a, b: {i["name"]: i for i in scalar_stability([run(a), run(b)], tol)}
    assert items(0.90, 0.84)["penult/margin_typeb.auc_margin_typeb"]["status"] == "FAIL"   # rel rule alone: PASS
    assert items(0.90, 0.86)["penult/margin_typeb.auc_margin_typeb"]["status"] == "PASS"
    d = items(0.90, 0.84)["penult/twonn_id.id"]                            # unlisted scalar: rule and text unchanged
    assert d["status"] == "PASS" and d["detail"].endswith("rel_spread=0.069") and "abs_tol" not in d["detail"]


# ---- A2b: scripts/check_rebuild.py ------------------------------------------------------------------------
def _scripts_on_path():
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
    if p not in sys.path:
        sys.path.insert(0, p)


def test_check_rebuild_identity():
    """A rebuild of the same dump passes; a moved float and a new key are caught."""
    import copy
    _scripts_on_path()
    from check_rebuild import compare_atlases
    root, atlas = _run(0)
    again = build_atlas(os.path.join(root, "dump"), os.path.join(_TMP, "r0_again"), {}, verbose=False)
    rep = compare_atlases(atlas, again)
    assert rep["status"] == "PASS" and rep["n_mismatch"] == 0, rep["mismatch_first50"]
    bad = copy.deepcopy(again)
    cc = bad["per_layer"]["penult"]["class_centers"]
    cc["sep_ratio"] = 2.0 * cc["sep_ratio"] + 1.0                          # far outside 1e-3 + 1e-3*|x|
    bad["per_layer"]["penult"]["new_invariant"] = {"x": 1.0}
    rep = compare_atlases(atlas, bad)
    assert rep["status"] == "FAIL" and rep["n_mismatch"] == 2, rep["mismatch_first50"]


# ---- A4: scale transfer (compare cross-depth guard, critic --align position) -------------------------------
R20_TAPS = ["stem"] + [f"layer{s}.{i}" for s in (1, 2, 3) for i in range(3)] + ["penult"]
R56_B5_TAPS = ["stem"] + [f"layer{s}.{i}" for s in (1, 2, 3) for i in (0, 5, 8)] + ["penult"]   # block_stride 5
R56_B1_TAPS = ["stem"] + [f"layer{s}.{i}" for s in (1, 2, 3) for i in range(9)] + ["penult"]    # block_stride 1


def _fake_atlas(path, arch, layers):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "atlas.json"), "w") as f:
        json.dump({"layers": layers, "per_layer": {l: {} for l in layers}, "source": "synthetic",
                   "meta": {"arch": arch}}, f)
    return path


def test_compare_scale_transfer_layers():
    """resnet56 at block_stride 5 pairs with resnet20 by position; a name match across depths is refused; the
    same arch at two strides is still compared by name."""
    import pytest
    from atlas.compare import match_layers
    assert match_layers(R20_TAPS, R56_B5_TAPS) == list(zip(R20_TAPS, R56_B5_TAPS))
    d = tempfile.mkdtemp(dir=_TMP)
    a = _fake_atlas(os.path.join(d, "r20"), "cifar10_resnet20", R20_TAPS)
    b5 = _fake_atlas(os.path.join(d, "r56_b5"), "cifar10_resnet56", R56_B5_TAPS)
    b1 = _fake_atlas(os.path.join(d, "r56_b1"), "cifar10_resnet56", R56_B1_TAPS)
    assert compare(a, b5)["layers"] == [list(p) for p in zip(R20_TAPS, R56_B5_TAPS)]
    with pytest.raises(SystemExit):
        compare(a, b1)
    assert len(compare(b5, b1)["layers"]) == 11


def test_critic_align_position():
    """--align position relabels the deeper net's taps (commit layers too) onto resnet20 names without touching
    the loaded atlas, refuses unequal tap counts, and the default mode changes nothing."""
    import copy
    import pytest
    from atlas.critic import DEFAULT_TOL, align_runs, commit_agreement

    def run(name, layers, commit):
        return {"dir": name, "manifest": {}, "atlas": {
            "layers": list(layers), "source": "real", "per_layer": {l: {"dim": 8} for l in layers},
            "cross_layer": {"commit_layer": {"per_factor": {"class": {"commit_layer": commit, "peak_layer": "penult"}}}}}}
    r20, r56 = run("r20", R20_TAPS, "layer3.1"), run("r56", R56_B5_TAPS, "layer3.5")
    loaded = r56["atlas"]
    runs = [r20, r56]
    assert align_runs(runs, "name") == {} and r56["atlas"]["layers"] == R56_B5_TAPS
    assert align_runs(runs, "position") == {"r56": [list(p) for p in zip(R20_TAPS, R56_B5_TAPS)]}
    assert r56["atlas"]["layers"] == R20_TAPS and list(r56["atlas"]["per_layer"]) == R20_TAPS
    assert loaded["layers"] == R56_B5_TAPS                                   # the loaded atlas is not mutated
    assert loaded["cross_layer"]["commit_layer"]["per_factor"]["class"]["commit_layer"] == "layer3.5"
    items = {c["name"]: c for c in commit_agreement(runs, DEFAULT_TOL)}
    assert items["commit/class"]["status"] == "PASS", items                # layer3.5 -> layer3.1, resnet20's commit
    with pytest.raises(SystemExit):
        align_runs([copy.deepcopy(r20), run("r56_b1", R56_B1_TAPS, "layer3.5")], "position")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    shutil.rmtree(_TMP, ignore_errors=True)
