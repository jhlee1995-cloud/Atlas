"""Known-answer tests for atlas/b4_core.py (docs/plans/B4_INTEGRATION.md D5, D7, D14): the numpy helpers, the HEAD-ADDITIVE
rule (power and size at realistic size, T1 review C2), the conformal p-value, the D7 seal, the output guard and the probe
record. numpy + pytest only (the import test runs a fresh interpreter).
  python -m pytest -q tests/test_b4_core.py
"""
import argparse
import ast
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from atlas import b4_core as C          # noqa: E402


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------------------------------------------------
def test_import_does_not_import_torch_scipy_sklearn():
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import atlas.b4_core, atlas.b4_collapse, atlas.faults; "
            "bad = [m for m in ('torch', 'scipy', 'sklearn', 'torchvision') if m in sys.modules]; print(bad); "
            "sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code, ROOT], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.parametrize("rel", ["atlas/b4_core.py", "atlas/b4_collapse.py", "atlas/faults.py"])
def test_imports_are_stdlib_and_numpy_only(rel):
    tree = ast.parse(open(os.path.join(ROOT, rel), encoding="utf-8").read())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add((node.module or "").split(".")[0] if node.level == 0 else ".")
    assert mods <= {"argparse", "contextlib", "hashlib", "json", "math", "os", "platform", "re", "resource",
                    "subprocess", "sys", "time", "zlib", "numpy", "."}, mods


def test_confirmation_only_list_is_the_d7_list():
    from atlas import faults
    assert C.CONFIRMATION_ONLY == C.HOLDOUT + C.EXTRA + ("exposure_global", "soiling")
    assert set(faults.CONFIRMATION_ONLY_FAULTS) == {"exposure_global", "soiling"}
    assert not any(C.is_confirmation_only(C.csplit(c, s)) for c in C.DISC for s in C.SEVS)
    assert all(C.is_confirmation_only(C.csplit(c, 3)) for c in C.HOLDOUT + C.EXTRA)
    assert C.is_confirmation_only("fault__exposure_global__l1") and C.is_confirmation_only("fault__soiling__a12")
    assert not C.is_confirmation_only("fault__deadpix__l3") and not C.is_confirmation_only("fault__paste__fog__a12")


# ---------------------------------------------------------------------------------------------------------------------
# ranks, AUC, DeLong, correlations
# ---------------------------------------------------------------------------------------------------------------------
def test_auc_and_delong():
    assert C.auc([2, 3], [0, 1]) == 1.0 and C.auc([0, 1], [2, 3]) == 0.0 and C.auc([1, 1], [1, 1]) == 0.5
    assert C.auc([1, 2], [1, 0]) == 0.875 and C.auc([], [1]) is None and C.auc([np.nan], [1]) is None
    rng = np.random.default_rng(0)
    y = rng.random(800) < 0.3
    s = rng.standard_normal(800) + y
    d, se, p = C.delong_diff(y, s, s)
    assert d == 0.0 and p == 1.0
    a = C.auc_y(y, s)
    d, se, p = C.delong_diff(y, s, -s)
    assert abs(d - (2 * a - 1)) < 1e-12 and se > 0 and p < 1e-6
    a2, se2 = C.delong(y, s)
    assert abs(a2 - a) < 1e-12 and 0 < se2 < 0.05


def test_aucboot_equals_replicated_sample():
    rng = np.random.default_rng(1)
    s = rng.integers(0, 20, 300).astype(float)                # many ties
    y = rng.random(300) < 0.3
    w = rng.integers(0, 3, 300)
    idx = np.repeat(np.arange(300), w)
    assert abs(C.AucBoot(s, y).auc(w) - C.auc_y(y[idx], s[idx])) < 1e-12


def test_spearman_partial_holm():
    assert abs(C.spearman([1, 2, 2, 3], [1, 2, 3, 4]) - 0.9486832980505138) < 1e-12
    assert C.spearman([1, 2], [1, 2]) is None
    rng = np.random.default_rng(2)
    z = rng.standard_normal(400)
    x, y = z + 0.3 * rng.standard_normal(400), z + 0.3 * rng.standard_normal(400)
    assert C.spearman(x, y) > 0.8 and abs(C.partial_spearman(x, y, z)) < 0.15
    adj = C.holm([0.01, 0.04, 0.03, 0.005, None])            # = scripts/b4_stats.js holm() self-test
    assert [None if v is None else round(v, 6) for v in adj] == [0.03, 0.06, 0.06, 0.02, None]
    lo, hi = C.wilson(5, 100)
    assert lo < 0.05 < hi


# ---------------------------------------------------------------------------------------------------------------------
# head statistics, temperature, ECE
# ---------------------------------------------------------------------------------------------------------------------
def test_head_stats_known_answers():
    Z = np.array([[2.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    h = C.head_stats(Z)
    p0 = math.exp(2) / (math.exp(2) + 2)
    assert abs(h["msp"][0] - p0) < 1e-12 and abs(h["gap"][0] - 2.0) < 1e-12 and h["gap"][1] == 0.0
    assert abs(h["energy"][1] - (1.0 + math.log(3))) < 1e-12 and abs(h["entropy"][1] - math.log(3)) < 1e-12
    assert abs(h["smargin"][1]) < 1e-12 and list(h["argmax"]) == [0, 0]
    assert set(C.HEAD_STATS) <= set(h) and set(C.ERR_SIGN) == set(C.HEAD_STATS)


def test_fit_temperature_recovers_the_generating_temperature():
    rng = np.random.default_rng(3)
    Z = rng.standard_normal((6000, 10)) * 3.0
    T0 = 2.0
    P = C.softmax(Z, T0)
    y = np.array([rng.choice(10, p=p) for p in P])
    assert abs(C.fit_temperature(Z, y) - T0) / T0 < 0.1


def test_ece_calibrated_and_overconfident():
    rng = np.random.default_rng(4)
    conf = rng.uniform(0.5, 1.0, 20000)
    assert C.ece(conf, rng.random(20000) < conf) < 0.02
    assert C.ece(np.full(1000, 0.99), np.zeros(1000)) > 0.98


# ---------------------------------------------------------------------------------------------------------------------
# joint model and the HEAD-ADDITIVE rule
# ---------------------------------------------------------------------------------------------------------------------
def test_crossfit_uses_an_informative_bundle():
    rng = np.random.default_rng(5)
    n = 3000
    h, g = rng.standard_normal(n), rng.standard_normal(n)
    y = (rng.random(n) < sigmoid(-1.0 + 1.0 * h + 1.5 * g)).astype(float)
    f = C.folds_for(np.arange(n))
    assert set(np.unique(f)) == set(range(C.N_FOLDS))
    ph = C.crossfit([("h", h, "spline")], y, f)
    pj = C.crossfit([("h", h, "spline"), ("g", g, "spline")], y, f)
    assert np.isfinite(ph).all() and C.auc_y(y, pj) - C.auc_y(y, ph) > 0.1


def test_head_additive_call_three_way():
    assert C.head_additive_call(0.03, [0.01, 0.05], [0.002, 0.02]) == "ADDS"
    assert C.head_additive_call(0.003, [-0.004, 0.012], [-0.01, 0.01]) == "BOUNDED"
    assert C.head_additive_call(0.012, [-0.01, 0.035], [-0.001, 0.02]) == "INCONCLUSIVE"
    assert C.head_additive_call(0.03, [0.01, 0.05], [-0.001, 0.02]) == "INCONCLUSIVE"      # CI of dI touches 0
    assert C.head_additive_call(None, None, None) == "INCONCLUSIVE"


def test_head_additive_rule_power_and_size():
    """T1 review C2: at n = 5000, ~7 % positives and 19 bundle signals, a true dAUC of ~+0.028 is called ADDS in >= 8 of
    10 datasets and an uninformative bundle in <= 1 of 10. (A JS port of this procedure gave 38/40 and 0/40.)"""
    pos = neg = 0
    for s in range(10):
        r = np.random.default_rng(100 + s)
        n = 5000
        h, g, z = r.standard_normal(n), r.standard_normal(n), r.standard_normal((n, 18))
        head = [("h", h, "spline"), ("h2", h + 0.1 * r.standard_normal(n), "spline")]
        bundle = [("g", g, "spline")] + [(f"z{j}", z[:, j], "spline") for j in range(18)]
        yp = (r.random(n) < sigmoid(-3.6 + 1.6 * h + 0.7 * g)).astype(float)
        yn = (r.random(n) < sigmoid(-3.6 + 1.6 * h)).astype(float)
        pos += C.head_increment(head, bundle, yp, np.arange(n), 200, f"p{s}")["call"] == "ADDS"
        neg += C.head_increment(head, bundle, yn, np.arange(n), 200, f"n{s}")["call"] == "ADDS"
    assert pos >= 8 and neg <= 1, (pos, neg)


def test_risk_coverage_and_rates():
    y = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float)
    conf = -np.arange(10.0)                                  # the first rows are the most confident
    aurc, r = C.risk_coverage(y, conf)
    assert abs(r["0.80"] - 2 / 8) < 1e-12 and 0 < aurc < 1
    yy = np.r_[np.ones(100), np.zeros(100)].astype(bool)
    s = np.r_[np.arange(100) + 100.0, np.arange(100.0)]
    assert C.tpr_at_fpr(yy, s) == 1.0 and C.fpr_at_tpr(yy, s) == 0.0


# ---------------------------------------------------------------------------------------------------------------------
# conformal p-values (D14)
# ---------------------------------------------------------------------------------------------------------------------
def test_conformal_p_known_answers_and_validity():
    cal = np.arange(1, 100, dtype=float)                     # 99 calibration scores 1..99
    p = C.conformal_p(cal, [0.5, 50.0, 99.0, 1000.0])
    assert np.allclose(p, [100 / 100, 51 / 100, 2 / 100, 1 / 100])
    assert C.cal_threshold(cal, 0.05) == 95.0 and C.cal_threshold(cal[:5], 0.05) == float("inf")
    rng = np.random.default_rng(6)
    fpr = []
    for _ in range(200):
        c, t = rng.random(750), rng.random(1500)
        fpr.append((C.conformal_p(c, t) <= 0.05).mean())
    assert abs(np.mean(fpr) - math.floor(0.05 * 751) / 751) < 0.004        # E[FPR] = l / (n + 1), exchangeable
    P = np.array([[0.5, 0.5], [0.001, 0.5]])
    s = C.cauchy_stat(P)
    assert abs(s[0]) < 1e-12 and s[1] > 100


# ---------------------------------------------------------------------------------------------------------------------
# kNN and helpers
# ---------------------------------------------------------------------------------------------------------------------
def test_knn_matches_brute_force():
    rng = np.random.default_rng(7)
    F, Q = rng.standard_normal((300, 5)), rng.standard_normal((40, 5))
    D, I = C.knn(F, Q, 3, chunk=16)
    B = np.sqrt(((Q[:, None] - F[None]) ** 2).sum(2))
    assert np.allclose(D, np.sort(B, 1)[:, :3]) and (I[:, 0] == B.argmin(1)).all()
    D2, I2 = C.knn(F, F, 2, exclude_self=True, chunk=64)
    assert (I2[:, 0] != np.arange(300)).all() and np.allclose(C.knn_kth(F, F, 2, exclude_self=True), D2[:, 1])
    assert np.allclose(np.linalg.norm(C.l2n(Q), axis=1), 1.0)
    with pytest.raises(ValueError):
        C.knn(F[:3], Q, 3, exclude_self=True)


def test_parse_split_and_names():
    assert C.parse_split("corrupt__fog__s3") == ("corrupt", "fog", 3)
    assert C.parse_split("fault__deadpix__l2") == ("fault", "deadpix", 2)
    assert C.parse_split("fault__paste__gaussian_noise__a12") == ("fault", "paste__gaussian_noise", "a12")
    assert C.parse_split("global__fog__s3") == ("global", "fog", 3)
    assert C.parse_split("ood__svhn") == ("ood", "svhn", None) and C.parse_split("test") == ("test", None, None)
    assert C.fsplit("occlusion_disc", 2) == "fault__occlusion_disc__l2"


def test_clean_json():
    o = C._clean({"a": np.float32(1.5), "b": [np.nan, np.inf, np.int64(3)], 3: np.bool_(True), "c": np.arange(2)})
    assert o == {"a": 1.5, "b": [None, None, 3], "3": True, "c": [0, 1]}
    json.dumps(o, allow_nan=False)


# ---------------------------------------------------------------------------------------------------------------------
# the seal (D7) and dump access
# ---------------------------------------------------------------------------------------------------------------------
def make_dump(d, splits=("ref", "test", "corrupt__fog__s3", "corrupt__frost__s3", "fault__exposure_global__l1"),
              sealed=False):
    rng = np.random.default_rng(8)
    for sub in ("acts/penult", "logits", "labels", "rows", "preds", "head"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    for s in splits:
        np.save(os.path.join(d, "acts", "penult", f"{s}.npy"), rng.standard_normal((20, 4)).astype(np.float16))
        np.save(os.path.join(d, "logits", f"{s}.npy"), rng.standard_normal((20, 10)).astype(np.float32))
        np.save(os.path.join(d, "labels", f"{s}.npy"), rng.integers(0, 10, 20))
        np.save(os.path.join(d, "rows", f"{s}.npy"), np.arange(20))
        np.savez(os.path.join(d, "preds", f"{s}.npz"), argmax=np.zeros(20, int), maxprob=np.ones(20, np.float32))
    np.savez(os.path.join(d, "head", "head.npz"), W=np.ones((10, 4), np.float32), b=np.zeros(10, np.float32))
    if sealed:
        json.dump({"unit": "u"}, open(os.path.join(d, "SEALED.json"), "w"))
    meta = {"layers": ["penult"], "splits": list(splits), "n_classes": 10,
            "b4": {"unit": "u", "layout": "fit", "taps": {"functional": {"stem": "penult", "pre": None},
                                                             "x4": ["penult"]}}}
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"))
    return d


def test_discovery_refuses_confirmation_only_splits(tmp_path):
    d = C.open_dump(make_dump(str(tmp_path / "b4d_u" / "dump")), "discovery")
    assert d.has("penult", "corrupt__fog__s3") and not d.has("penult", "corrupt__frost__s3")
    assert d.list_splits("corrupt") == ["corrupt__fog__s3"]
    assert d.acts("penult", "test").dtype == np.float64 and d.logits("test").shape == (20, 10)
    for s in ("corrupt__frost__s3", "fault__exposure_global__l1"):
        with pytest.raises(SystemExit):
            d.acts("penult", s)
        with pytest.raises(C.ReadRefused):
            d.labels(s)
    assert d.splits_read == ["test"] and d.x4_taps() == ["penult"] and d.head()[0].shape == (10, 4)
    c = C.open_dump(str(tmp_path / "b4d_u" / "dump"), "confirmation")
    assert c.acts("penult", "corrupt__frost__s3").shape == (20, 4)


def test_partial_dump_is_refused(tmp_path):
    d = str(tmp_path / "partial" / "dump")
    os.makedirs(d)
    with pytest.raises(C.ReadRefused):
        C.open_dump(d, "discovery")
    with pytest.raises(C.ReadRefused):
        C.open_dump(make_dump(str(tmp_path / "x" / "dump")), "exploration")


def test_sealed_dump_needs_confirmation_and_a_p2_freeze(tmp_path, monkeypatch):
    d = make_dump(str(tmp_path / "sealed_u" / "dump"), sealed=True)
    monkeypatch.delenv("ATLAS_B4_UNSEAL", raising=False)
    with pytest.raises(C.ReadRefused):
        C.open_dump(d, "discovery")
    with pytest.raises(C.ReadRefused):
        C.open_dump(d, "confirmation")                        # no ATLAS_B4_UNSEAL
    monkeypatch.setenv("ATLAS_B4_UNSEAL", "p2sha")
    monkeypatch.setenv("ATLAS_B4_CHECK_DIR", str(tmp_path / "check"))
    with pytest.raises(C.ReadRefused):
        C.open_dump(d, "discovery")                           # the env var never opens a seal in discovery

    class R:
        def __init__(self, rc, out=""):
            self.returncode, self.stdout = rc, out
    monkeypatch.setattr(C, "_git", lambda *a: R(1))          # P2 is not an ancestor / lacks freeze_P2.json
    with pytest.raises(C.ReadRefused):
        C.open_dump(d, "confirmation")
    monkeypatch.setattr(C, "_git", lambda *a: R(0, "headsha\n"))
    dd = C.open_dump(d, "confirmation")
    assert dd.sealed and dd.acts("penult", "fault__exposure_global__l1").shape == (20, 4)
    log = [json.loads(x) for x in open(tmp_path / "check" / "unseal_log.jsonl")]
    assert log[-1]["p2"] == "p2sha" and log[-1]["head"] == "headsha"


def test_b4c_path_prefix_seals(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "REPO_ROOT", str(tmp_path))
    d = make_dump(str(tmp_path / "results" / "b4c_zoo" / "dump"))
    assert C.is_sealed_path(d) and C.is_sealed_path("results/b4c_zoo/dump")
    assert not C.is_sealed_path(make_dump(str(tmp_path / "results" / "b4d_zoo" / "dump")))
    monkeypatch.delenv("ATLAS_B4_UNSEAL", raising=False)
    with pytest.raises(C.ReadRefused):
        C.open_dump("results/b4c_zoo/dump", "discovery")


def test_registry_helpers():
    reg = C.load_registry("experiments/b4/models.json")
    assert C.layout_dump(reg, "resnet44", "fit") == "results/b4c_resnet44/dump"
    with pytest.raises(SystemExit):
        C.layout_dump(reg, "resnet44", "eval")
    with pytest.raises(SystemExit):
        C.unit_spec(reg, "no_such_unit")


# ---------------------------------------------------------------------------------------------------------------------
# output guard and the probe record (D5 CLI contract, touched-once)
# ---------------------------------------------------------------------------------------------------------------------
def test_guard_output_append_only_and_touched_once(tmp_path):
    root = tmp_path / "results" / "b4_t2"
    (root / "resnet44").mkdir(parents=True)
    (root / "resnet44" / "probe.json").write_text("{}")
    with pytest.raises(C.ReadRefused):
        C.guard_output(str(root / "resnet44"), "discovery", "probe.json")            # not empty
    with pytest.raises(C.ReadRefused):
        C.guard_output(str(root / "resnet44_r2"), "confirmation", "probe.json")      # probed once under any tag
    C.guard_output(str(root / "resnet44_r2"), "discovery", "probe.json")              # discovery re-runs are allowed
    C.guard_output(str(root / "resnet4"), "confirmation", "probe.json")              # another unit
    (root / "resnet56_s1").mkdir()
    (root / "resnet56_s1" / "probe.json").write_text("{}")
    C.guard_output(str(root / "resnet56_s12m"), "confirmation", "probe.json")        # prefix is not a sibling
    assert C.strip_tag("resnet56_s12m_fit_r2") == "resnet56_s12m_fit" and C.strip_tag("x_s2replay") == "x"


def test_probe_argparser_contract():
    ap = C.probe_argparser("t1", layout=True)
    a = C.parse_probe_args(ap, ["--registry", "r", "--unit", "u", "--phase", "discovery", "--out", "o", "--layout", "eval"])
    assert (a.registry, a.unit, a.phase, a.out, a.layout, a.selftest) == ("r", "u", "discovery", "o", "eval", False)
    assert C.parse_probe_args(ap, ["--selftest"]).selftest
    with pytest.raises(SystemExit):
        C.parse_probe_args(ap, ["--unit", "u"])
    s = C.probe_argparser("t1s", frames=True)
    assert C.parse_probe_args(s, ["--registry", "r", "--unit", "u", "--phase", "confirmation", "--out", "o",
                                  "--frames", "/root/b4_frames/u"]).frames == "/root/b4_frames/u"


def test_probe_run_record(tmp_path):
    d = C.open_dump(make_dump(str(tmp_path / "b4d_u" / "dump")), "discovery")
    args = argparse.Namespace(out=str(tmp_path / "out" / "u"), phase="discovery", unit="u", layout="fit")
    run = C.ProbeRun("t2", os.path.join(ROOT, "atlas", "b4_core.py"), args)
    with run.timed("tap:penult"):
        d.acts("penult", "test")
    p = run.finish({"value": float("nan"), "x": np.float32(2.0)}, dumps=[d])
    rec = json.load(open(p))
    assert os.path.basename(p) == "probe.json" and rec["value"] is None and rec["x"] == 2.0
    assert rec["timing_s"]["total"] >= rec["timing_s"]["tap:penult"] >= 0 and "max_rss_mb" in rec
    assert len(rec["code"]["sha256"]) == 64 and "repo_commit" in rec["code"] and rec["nboot"] == 200
    assert rec["dumps"][0]["splits_read"] == ["test"] and rec["program"] == "t2" and rec["tag"] == ""
    with pytest.raises(C.ReadRefused):
        C.ProbeRun("t2", os.path.join(ROOT, "atlas", "b4_core.py"), args)          # append-only
