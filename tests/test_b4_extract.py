"""Known-answer tests for scripts/b4_extract.py and scripts/b4_weights.py (docs/plans/B4_INTEGRATION.md D2, D7, D15).
The synthetic tests run a 4/8-channel CIFAR-shaped ResNet (b4_weights.tiny_resnet) through the full write path on the
CPU: layouts, the X4 rule, the storage dtypes, meta.json last, the head gates, sealing, the seal manifest and the anchor
gate. The hub tests (skipped without torch or network) pin every chenyaofo tap spec at the pinned commit: dims, spatial
flags, duplicates, the D2 pre tap, the functional taps and the head identity.
  python -m pytest -q tests/test_b4_extract.py
"""
import importlib.util
import json
import os
import shutil
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def ex():
    return _load("b4_extract")


@pytest.fixture(scope="module")
def reg():
    return json.load(open(os.path.join(ROOT, "experiments", "b4", "models.json")))


# ---------------------------------------------------------------------------------------------------------------------
# contracts that need no torch
# ---------------------------------------------------------------------------------------------------------------------
def test_constants_agree_with_core_and_registry(ex, reg):
    from atlas import b4_core as C
    assert (ex.DISC, ex.HOLDOUT, ex.EXTRA, ex.SEVS) == (C.DISC, C.HOLDOUT, C.EXTRA, C.SEVS)
    for lay in ("fit", "eval", "maps"):
        want = {k: (list(v) if isinstance(v, tuple) else v) for k, v in ex.ROWS[lay].items()}
        assert reg["rows"][lay] == want, lay
    assert tuple(reg["confirmation_only"]) == C.CONFIRMATION_ONLY
    assert reg["hub"]["ref"] == _load("b4_weights").HUB_REF


def test_pre_tap_rule_is_the_collapse_rule(ex):
    from atlas import b4_collapse as NC
    cases = [(["stem", "layer1.0", "layer3.0", "layer3.1", "penult"], {"stem": 1, "layer1.0": 1, "layer3.0": 1,
                                                                       "layer3.1": 1}, ["layer3.1"]),
             (["a", "b", "c", "penult"], {"a": 1, "b": 1, "c": 0}, []), (["a", "penult"], {"a": 0}, [])]
    for taps, sp, dups in cases:
        assert ex.pre_tap_rule(taps, sp, dups) == NC.pre_tap(taps, sp, dups)


def test_plan_splits_fit_eval_maps(ex, reg):
    spec = {m["id"]: m for m in reg["models"]}
    data = ex.SynthData()
    fit = [p[0] for p in ex.plan_splits(spec["resnet44"], "fit", data)]
    assert fit[:2] == ["ref", "test"] and "corrupt__saturate__s5" in fit and "corrupt__frost__s1" in fit
    assert "fault__exposure_global__l3" in fit and len(fit) == 2 + 30 + 15 + 8 + 2 + 9
    assert "corrupt__saturate__s3" not in [p[0] for p in ex.plan_splits(spec["resnet20_s1"], "fit", data)]   # no extras
    ev = ex.plan_splits(spec["resnet20_s1"], "eval", data)
    names = [p[0] for p in ev]
    assert names[0] == "test" and "ref" not in names and "corrupt__frost__s1" not in names and "corrupt__frost__s3" in names
    assert "corrupt__saturate__s3" not in names                                     # R2: no extras (D4)
    assert "corrupt__saturate__s3" in [p[0] for p in ex.plan_splits(spec["resnet56_s31"], "eval", data)]
    keep = {p[0]: p[2] for p in ex.plan_splits(spec["resnet44"], "fit", data)}
    assert keep["corrupt__fog__s3"] == "all" and keep["corrupt__frost__s3"] == "x4" and keep["fault__deadpix__l1"] == "all"
    conf = [p[0] for p in ex.plan_splits(spec["resnet20_s3"], "maps", data)]
    disc = [p[0] for p in ex.plan_splits(spec["resnet20_hub"], "maps", data)]
    assert "fault__soiling__a12" in conf and "fault__soiling__a12" not in disc and len(conf) == 3 + 21 + 7
    assert spec["resnet20_hub"]["layouts"]["maps"]["maps"] == ["headmap", "stage2map"]


def test_weights_gate_rules(ex, tmp_path):
    rec = tmp_path / "w.json"
    rec.write_text(json.dumps({"units": {"h": {"status": "PASS"}, "bad": {"status": "FAIL", "why": "x"},
                                         "ne": {"status": "NOT_EVALUABLE"}}}))
    assert ex.weights_gate({"id": "h", "source": "hub"}, str(rec))["status"] == "PASS"
    assert ex.weights_gate({"id": "ne", "source": "hub"}, str(rec))["status"] == "NOT_EVALUABLE"
    with pytest.raises(SystemExit):
        ex.weights_gate({"id": "bad", "source": "file"}, str(rec))
    with pytest.raises(SystemExit):
        ex.weights_gate({"id": "missing", "source": "hub"}, str(rec))                # the README gate runs first
    assert ex.weights_gate({"id": "missing", "source": "file"}, str(rec)) is None
    assert ex.weights_gate({"id": "x", "source": "synthetic"}, None) is None


def test_head_path_check(ex):
    rng = np.random.default_rng(0)
    W, b = rng.standard_normal((10, 16)), rng.standard_normal(10)
    X = rng.standard_normal((500, 16))
    L = (X @ W.T + b).astype(np.float32)
    assert ex.head_path_check(X.astype(np.float16), W, b, L)["status"] == "PASS"
    assert ex.head_path_check(-X, W, b, L)["status"] == "FAIL"


# ---------------------------------------------------------------------------------------------------------------------
# synthetic end-to-end (torch, CPU)
# ---------------------------------------------------------------------------------------------------------------------
def test_selftest_passes(ex, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    rep = ex.selftest(workdir=str(tmp_path))
    assert rep["status"] == "PASS", [c for c in rep["checks"] if not c["pass"]]
    assert len(rep["checks"]) >= 20


def test_vectorised_normalisation_is_the_pil_path():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from PIL import Image
    from atlas.extract_acts import cifar_transform
    bw = _load("b4_weights")
    u8 = np.random.default_rng(0).integers(0, 256, (8, 32, 32, 3)).astype(np.uint8)
    for norm in ("chenyaofo", "cifar_true"):
        a = torch.stack([cifar_transform(norm)(Image.fromarray(im)) for im in u8])
        assert torch.equal(a, bw.to_input(u8, norm))


def test_sealed_flag_must_match_the_registry(ex, tmp_path):
    pytest.importorskip("torch")
    spec = ex.synth_spec(str(tmp_path))
    with pytest.raises(SystemExit):
        ex.extract(spec, "eval", ex.SynthData(), spec["layouts"]["eval"]["dump"], "cpu", False, ex.SMALL_ROWS, 32)
    with pytest.raises(SystemExit):
        ex.extract(spec, "fit", ex.SynthData(), spec["layouts"]["fit"]["dump"], "cpu", True, ex.SMALL_ROWS, 32)
    assert not os.path.exists(spec["layouts"]["eval"]["dump"])                     # refused before writing


def test_real_units_are_gpu_only(ex):
    pytest.importorskip("torch")
    with pytest.raises(SystemExit):
        ex.main(["--registry", "experiments/b4/models.json", "--unit", "resnet44", "--layout", "fit",
                 "--volume", "/nonexistent", "--sealed", "--device", "cpu"])


def test_anchor_gate_on_a_synthetic_anchor(ex, tmp_path):
    pytest.importorskip("torch")
    spec = ex.synth_spec(str(tmp_path), uid="anc")
    fit = spec["layouts"]["fit"]["dump"]
    ex.extract(spec, "fit", ex.SynthData(), fit, "cpu", False, ex.SMALL_ROWS, 32)
    meta = json.load(open(os.path.join(fit, "meta.json")))
    old = tmp_path / "committed_anchor"
    shutil.copytree(fit, str(old / "dump"))
    (old / "atlas.json").write_text(json.dumps({"meta": {"accuracy": {"test": meta["accuracy"]["test"]},
                                                         "weights": "synthetic"}}))
    spec["anchor"] = str(old)
    reg = {"models": [spec]}
    assert ex.anchor_gate(reg, ["anc"], str(tmp_path / "gate1.json")) == []
    g = json.load(open(tmp_path / "gate1.json"))
    assert g["units"]["anc"]["status"] == "PASS" and g["units"]["anc"]["tap_rel_max"] == 0.0
    p = old / "dump" / "acts" / "layer2.0" / "test.npy"
    a = np.load(p)
    np.save(p, (a.astype(np.float32) * 1.05).astype(np.float16))
    assert ex.anchor_gate(reg, ["anc"], str(tmp_path / "gate2.json")) == ["anc"]
    assert json.load(open(tmp_path / "gate2.json"))["units"]["anc"]["taps"]["layer2.0"] > 4e-3


def test_record_of_a_sealed_dump_is_small_and_blind(ex, tmp_path):
    pytest.importorskip("torch")
    spec = ex.synth_spec(str(tmp_path), uid="rec")
    ev = spec["layouts"]["eval"]["dump"]
    ex.extract(spec, "eval", ex.SynthData(), ev, "cpu", True, ex.SMALL_ROWS, 32)
    out = tmp_path / "records" / "rec_eval.json"
    ex.write_record(ev, str(out))
    r = json.load(open(out))
    assert r["accuracy"] == "SEALED" and "files" not in r["b4"] and r["b4"]["files_summary"]["n"] > 50
    assert "ref_indices" not in r and len(r["meta_sha256"]) == 64


# ---------------------------------------------------------------------------------------------------------------------
# every chenyaofo architecture at the pinned commit (pretrained=False: code only, no weights)
# ---------------------------------------------------------------------------------------------------------------------
EXPECT = {   # arch -> (penult, sum of other tap dims, pre tap, dup taps)
    "cifar10_resnet20": (64, 352, "layer3.1", ["layer3.2"]), "cifar10_resnet32": (64, 352, "layer3.2", ["layer3.4"]),
    "cifar10_resnet44": (64, 352, "layer3.3", ["layer3.6"]), "cifar10_resnet56": (64, 352, "layer3.5", ["layer3.8"]),
    "cifar10_vgg11_bn": (512, 2048, "features.28", []), "cifar10_vgg13_bn": (512, 2048, "features.34", []),
    "cifar10_vgg16_bn": (512, 2048, "features.43", []), "cifar10_vgg19_bn": (512, 2048, "features.52", []),
    "cifar10_mobilenetv2_x0_5": (1280, 376, "features.17", []), "cifar10_mobilenetv2_x0_75": (1280, 568, "features.17", []),
    "cifar10_mobilenetv2_x1_0": (1280, 744, "features.17", []), "cifar10_mobilenetv2_x1_4": (1792, 1048, "features.17", []),
    "cifar10_shufflenetv2_x0_5": (1024, 456, "stage4", []), "cifar10_shufflenetv2_x1_0": (1024, 1068, "stage4", []),
    "cifar10_shufflenetv2_x1_5": (1024, 1608, "stage4", []), "cifar10_shufflenetv2_x2_0": (2048, 2220, "stage4", []),
    "cifar10_repvgg_a0": (1280, 576, "stage3", []), "cifar10_repvgg_a1": (1280, 768, "stage3", []),
    "cifar10_repvgg_a2": (1408, 1120, "stage3", []),
}


@pytest.mark.parametrize("arch", sorted(EXPECT))
def test_hub_tap_specs(ex, reg, arch):
    torch = pytest.importorskip("torch")
    bw = _load("b4_weights")
    try:
        model = bw.hub_load(arch, False)
    except Exception as e:                                                         # no network on this machine
        pytest.skip(f"hub unavailable: {e!r}"[:120])
    torch.manual_seed(0)
    hooks = ex.make_hooks(arch, model)
    taps = list(hooks.layer_names)
    x = torch.randn(64, 3, 32, 32)
    f, logits = hooks.forward(x)
    hooks.close()
    pen, other, pre, dups = EXPECT[arch]
    assert taps[-1] == "penult" and list(f) == taps and f["penult"].shape == (64, pen)
    assert sum(v.shape[1] for k, v in f.items() if k != "penult") == other
    _, W, b = bw.last_linear(model)
    lg = logits.double().numpy()
    assert np.abs(f["penult"].double().numpy() @ W.T + b - lg).max() <= 1e-4 * max(1.0, np.abs(lg).max())
    spatial = ex.tap_spatial(model, taps, "cpu")
    got = [t for t in taps if t != "penult" and f[t].shape == f["penult"].shape
           and ex.rel_rms(f[t].numpy(), f["penult"].numpy()) < ex.GATE["dup_rel_rms"]]
    assert got == dups
    assert ex.pre_tap_rule(taps, spatial, got) == pre
    fun = ex.functional_taps(arch, taps, pre)
    assert set(fun) == {"stem", "s1end", "s2end", "pre", "penult"} and fun["stem"] == taps[0]
    assert all(fun[k] in taps for k in fun)
    depth = ex.depth_fractions(model, taps)
    vals = [depth[t] for t in taps]
    assert all(0 < v <= 1 for v in vals) and vals[-1] == 1.0 and all(a <= b for a, b in zip(vals, vals[1:]))
    for m in reg["models"]:
        if m["arch"] == arch:
            assert (m["penult"], m["f16dims"]) == (pen, other), m["id"]
