"""Known-answer tests for scripts/anomaly_probe.py (docs/plans/ANOMALY_H1.md, AX-1..AX-4).

numpy + pytest only: no torch, no sklearn, no real dump. The synthetic dump is built in a temp dir. The schema test pins
every key scripts/anomaly_eval.js reads, so the probe and the evaluator cannot drift apart silently.
  python -m pytest -q tests/test_anomaly_probe.py
"""
import ast
import importlib.util
import json
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(os.path.dirname(HERE), "scripts", "anomaly_probe.py")


@pytest.fixture(scope="module")
def ap():
    spec = importlib.util.spec_from_file_location("anomaly_probe", PROBE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def selftest_run(ap, tmp_path_factory):
    d = str(tmp_path_factory.mktemp("anomaly_selftest"))
    rep = ap.selftest(n_ref=3000, workdir=d)
    return rep, d


def test_imports_are_stdlib_and_numpy_only():
    tree = ast.parse(open(PROBE, encoding="utf-8").read())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add((node.module or "").split(".")[0])
    assert mods <= {"argparse", "hashlib", "json", "os", "platform", "re", "shutil", "sys", "tempfile", "time", "zlib",
                    "numpy"}, mods


def test_auc_and_ranks_known_answers(ap):
    assert ap.auc([2, 3], [0, 1]) == 1.0
    assert ap.auc([0, 1], [2, 3]) == 0.0
    assert ap.auc([1, 1], [1, 1]) == 0.5
    assert ap.auc([1, 2], [1, 0]) == 0.875
    assert ap.auc([], [1]) is None and ap.auc([np.nan], [1]) is None
    assert list(ap.rank_avg([3, 1, 3, 2])) == [3.5, 1.0, 3.5, 2.0]
    rng = np.random.default_rng(1)
    p, n = rng.integers(0, 5, 40), rng.integers(0, 5, 30)
    brute = np.mean([(a > b) + 0.5 * (a == b) for a in p for b in n])
    assert abs(ap.auc(p, n) - brute) < 1e-12


def test_spearman(ap):
    x = np.arange(20.0)
    assert abs(ap.spearman(x, x ** 3) - 1.0) < 1e-12
    assert abs(ap.spearman(x, -x) + 1.0) < 1e-12
    assert ap.spearman(x, np.ones(20)) is None


def test_knn_kth_matches_brute_force(ap):
    line = np.arange(10, dtype=np.float64)[:, None]
    assert abs(ap.knn_kth(line, [[0.4]], 1)[0] - 0.4) < 1e-12
    assert abs(ap.knn_kth(line, [[0.4]], 3)[0] - 1.6) < 1e-12
    assert np.allclose(ap.knn_kth(line, line, 2, exclude_self=True)[[0, 5, 9]], [2.0, 1.0, 2.0])
    rng = np.random.default_rng(2)
    F, Q = rng.standard_normal((300, 7)), rng.standard_normal((50, 7))
    D = np.sqrt(((Q[:, None, :] - F[None, :, :]) ** 2).sum(-1))
    assert np.allclose(ap.knn_kth(F, Q, 10, chunk=16), np.sort(D, axis=1)[:, 9], atol=1e-10)
    Ds = np.sqrt(((F[:, None, :] - F[None, :, :]) ** 2).sum(-1))
    assert np.allclose(ap.knn_kth(F, F, 10, exclude_self=True), np.sort(Ds, axis=1)[:, 10], atol=1e-6)


def test_class_span_and_e_perp(ap):
    C = np.zeros((10, 16))
    for k in range(9):
        C[k, k] = 3.0
    Q = ap.class_span(C)
    assert Q.shape == (16, 9) and np.allclose(Q.T @ Q, np.eye(9), atol=1e-12)
    x = np.zeros((3, 16))
    x[0, 0], x[0, 12] = 6.0, 4.0                    # 3 e0 in-span + 4 e12 off-span from c_0: 16 / 25
    x[1, 0] = 3.0 + 2.0                             # in-span only: 0
    x[2, 0], x[2, 13] = 3.0, 5.0                    # off-span only: 1
    v = ap.e_perp(x, C, Q, np.array([0, 0, 0]))
    assert np.allclose(v, [16.0 / 25.0, 0.0, 1.0], atol=1e-12)


def test_batch_idx_deterministic_without_replacement(ap):
    a, b = ap.batch_idx("k|s|8", 50, 8, 20), ap.batch_idx("k|s|8", 50, 8, 20)
    assert np.array_equal(a, b) and a.shape == (20, 8) and all(len(set(r)) == 8 for r in a)
    assert not np.array_equal(a, ap.batch_idx("k|other|8", 50, 8, 20))
    assert ap.batch_idx("k|s|8", 50, 8, 20, offset=100).min() >= 100
    assert ap.batch_idx("k|s|8", 5, 8, 20) is None


def test_t2_and_whitening(ap):
    rng = np.random.default_rng(3)
    X = rng.standard_normal((4000, 6)) * np.array([3.0, 2.0, 1.0, 1.0, 0.5, 0.1])
    mu, w, V = ap.pca_frame(X, 4)
    assert len(w) == 4 and np.all(np.diff(w) <= 0)
    Z = ap.whiten(X, mu, w, V)
    assert np.allclose(np.cov(Z.T), np.eye(4), atol=1e-10)
    Zc = np.tile(np.array([[1.0, 0.0, 2.0, 0.0]]), (64, 1))
    assert np.allclose(ap.t2(Zc, np.arange(64)[None, :]), [64 * 5.0])


def test_refusals_before_reading(ap, tmp_path):
    name, rec = ap.process_dump(str(tmp_path / "margin_b1_vitb16" / "dump"))
    assert rec["status"] == "REFUSED" and "never read" in rec["error"]
    d = tmp_path / "atlas_v1_x" / "dump"
    (d / "logits").mkdir(parents=True)
    assert ap.process_dump(str(d))[1]["status"] == "REFUSED"
    d2 = tmp_path / "atlas_v1_y" / "dump"
    d2.mkdir(parents=True)
    (d2 / "meta.json").write_text(json.dumps({"arch": "vit_b_16", "n_classes": 1000, "source": "real", "n_test": 25000}))
    r2 = ap.process_dump(str(d2))[1]
    assert r2["status"] == "REFUSED" and "not a CIFAR ResNet" in r2["error"]
    (d2 / "meta.json").write_text(json.dumps({"arch": "cifar10_resnet20", "n_classes": 10, "source": "real", "n_test": 5000,
                                              "layers": list(ap.SYN_LAYERS)}))
    r3 = ap.process_dump(str(d2), role_override="confirmation")
    assert r3[1]["status"] == "REFUSED" and "synthetic dumps only" in r3[1]["error"]


def test_role_enforcement(ap, tmp_path):
    rd = ap.Reader(str(tmp_path), "discovery")
    for s in ("corrupt__zoom_blur__s3", "panel", "ood__svhn"):
        with pytest.raises(PermissionError):
            rd.acts("penult", s)
    assert ap.Reader(str(tmp_path), "confirmation").allowed("corrupt__zoom_blur__s3")
    assert not ap.Reader(str(tmp_path), "confirmation").allowed("ood__svhn")
    assert ap.process_dump.__defaults__ == (None, False)                      # no override, no synthetic by default


def test_write_refuses_overwrite(ap, tmp_path):
    p = ap.write_probe(str(tmp_path), "t1", {"schema": ap.SCHEMA, "x": np.float64(1.5), "y": float("nan")})
    assert json.load(open(p)) == {"schema": ap.SCHEMA, "x": 1.5, "y": None}
    with pytest.raises(FileExistsError):
        ap.write_probe(str(tmp_path), "t1", {})
    with pytest.raises(ValueError):
        ap.write_probe(str(tmp_path), "../escape", {})


def test_synthetic_known_answers(selftest_run):
    rep, _ = selftest_run
    assert rep["status"] == "PASS", [c for c in rep["checks"] if not c["pass"]]


def test_probe_json_schema_matches_the_evaluator(ap, selftest_run):
    _, d = selftest_run
    j = json.load(open(os.path.join(d, "out", "anomaly_probe_selftest", "probe.json")))
    assert j["schema"] == ap.SCHEMA and j["code"]["sha256"] and "created_utc" in j
    (name, rec), = j["dumps"].items()
    assert rec["status"] == "OK" and rec["role"] == "confirmation"
    a1, a2, a3, a4 = rec["AX1"], rec["AX2"], rec["AX3"], rec["AX4"]
    for k in ("auc_perp_pre", "auc_perp_penult", "auc_cov_pre", "auc_full_penult", "auc_par_pre"):
        assert "corrupt__motion_blur__s1" in a1[k]
    for k in ("perp_pre", "full_penult", "cov_pre", "perp_penult", "par_pre"):
        assert "fpr_mean" in a1["single_class"][k] and len(a1["single_class"][k]["fpr"]) == 10
    assert {"par_pre", "perp_pre"} <= set(a1["single_vs_mixed_auc"])
    assert set(a2["auc"]["corrupt__brightness__s1"]) == set(ap.TAPKEYS)
    assert set(a2["router"]["accuracy_by_family"]) == {"N", "B", "L", "P"}
    assert set(a2["router"]["holdout"]["corrupt__impulse_noise__s3"]["assigned"]) == {"N", "B", "L", "P"}
    s = a3["splits"]["corrupt__snow__s5"]
    assert all(len(s[k]) == ap.AX3["n_batches"] for k in ("h", "loss", "e", "e2"))
    assert len(a3["ood"]["ood__cifar100"]["h"]) == ap.AX3["n_batches"] and "h" in a3["clean_B"]
    assert {"e_perp", "d1", "d1_over_r", "dens2", "n_pos", "n_neg"} <= set(a4["auroc"]["corrupt__jpeg_compression__s3"])
    assert "corrupt__zoom_blur__s3" in a4["auroc"]
