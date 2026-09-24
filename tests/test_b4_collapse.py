"""Known-answer tests for atlas/b4_collapse.py (docs/plans/B4_INTEGRATION.md D5, D2, D12): nc1 equals the committed
atlas formula (atlas/invariants/landmarks.py:71) exactly, the Gram-form centre distances equal the broadcast form (T3
review B2), the cross-validated spectral gap removes the width dependence of the raw gap, and the D2 pre-tap rule.
  python -m pytest -q tests/test_b4_collapse.py
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from atlas import b4_collapse as NC      # noqa: E402


def blobs(n=3000, d=16, K=10, sep=4.0, sigma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    C = rng.standard_normal((K, d)) * sep
    y = rng.integers(0, K, n)
    return C[y] + sigma * rng.standard_normal((n, d)), y, C


def test_nc1_is_the_committed_atlas_formula():
    landmarks = pytest.importorskip("atlas.invariants.landmarks")     # imports scipy / sklearn: the pod has them
    X, y, _ = blobs()
    ctx = SimpleNamespace(n_classes=10, ref=X.astype(np.float32), ref_labels=y)
    want = landmarks.neural_collapse(ctx, {})
    rec, C, Sb = NC.nc_block(X.astype(np.float32), y, 10)
    assert abs(rec["nc1"] - want["nc1"]) <= 1e-12 * max(1.0, abs(want["nc1"]))
    assert abs(rec["etf_dev"] - want["etf_deviation"]) < 1e-12 and abs(rec["norm_cv"] - want["class_mean_norm_cv"]) < 1e-12


def test_nc1_limits_and_head_alignment():
    X, y, _ = blobs(sigma=1e-6)
    assert NC.nc_block(X, y)[0]["nc1"] < 1e-9                             # fully collapsed
    Xw, yw, _ = blobs(sigma=3.0)
    assert NC.nc_block(Xw, yw)[0]["nc1"] > NC.nc_block(*blobs(sigma=1.0)[:2])[0]["nc1"]
    rec, C, _ = NC.nc_block(X, y, 10, W=None)
    W = C - C.mean(0)                                                      # the head = centred class means (NC3)
    rec3 = NC.nc_block(X, y, 10, W=W)[0]
    assert rec3["head_center_cos"] > 1 - 1e-9 and rec3["nc3_dist"] < 1e-12
    assert rec["rank_sb"] == 9 and rec["k_present"] == 10
    assert NC.nc_block(X[:5], np.zeros(5, int))[0] is None                 # one class: undefined
    assert NC.nc1_pseudo(X, np.zeros(len(X), int)) is None


def test_simplex_etf_and_center_distances():
    K, d = 10, 12
    E = np.eye(K)[:, :K] - 1.0 / K                                        # simplex ETF in the first K coordinates
    M = np.zeros((K, d))
    M[:, :K] = E / np.linalg.norm(E, axis=1, keepdims=True) * 5.0
    rng = np.random.default_rng(1)
    y = np.repeat(np.arange(K), 200)
    X = M[y] + 0.01 * rng.standard_normal((len(y), d))
    rec = NC.nc_block(X, y)[0]
    assert rec["etf_dev"] < 1e-3 and rec["norm_cv"] < 1e-2 and rec["center_dist_cv"] < 1e-2
    C = rng.standard_normal((K, 64))
    brute = np.linalg.norm(C[:, None] - C[None], axis=2)
    assert np.abs(NC.center_pair_dists(C) - brute).max() < 1e-9              # T3 review B2
    Q = rng.standard_normal((500, 64))
    assert np.abs(NC.center_dists(Q, C, rows=64) - np.linalg.norm(Q[:, None] - C[None], axis=2)).max() < 1e-9


def test_nc1_around_train_means():
    X, y, _ = blobs(seed=2)
    rec, C, Sb = NC.nc_block(X[:1500], y[:1500])
    same = NC.nc1_around(X[:1500], y[:1500], C, Sb)
    assert abs(same - rec["nc1"]) < 1e-9                                  # around its own means: nc1 itself
    held = NC.nc1_around(X[1500:], y[1500:], C, Sb)
    assert held >= 0.8 * rec["nc1"]


def spiked(n, d, seed, K=10, spike=10.0):
    """K-1 strong directions (variance spike + 1) in isotropic unit noise: the population gap lambda9/lambda10 is 11."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    X[:, :K - 1] *= np.sqrt(spike + 1.0)
    return X


def test_cross_validated_gap_removes_the_width_dependence():
    s64 = NC.spectrum_block(spiked(2000, 64, 3), 10, *[spiked(2000, 64, 3)[h] for h in NC.split_halves(2000)])
    X = spiked(2000, 512, 4)
    h1, h2 = NC.split_halves(2000)
    s512 = NC.spectrum_block(X, 10, X[h1], X[h2])
    assert abs(s64["g_cv"] - 11.0) / 11.0 < 0.2 and abs(s512["g_cv"] - 11.0) / 11.0 < 0.25
    assert s512["g_raw"] < 0.7 * s512["g_cv"]                           # the bulk edge inflates lambda_10 at d/n 0.26
    assert abs(s512["g_raw"] - s64["g_raw"]) / s64["g_raw"] > 0.25        # raw gaps of the two widths disagree
    assert s64["topk_frac"] > 0.5 and s64["khat"] == 9 and len(s64["lam_cv_top"]) == 10


def test_label_free_coords_and_ncc():
    X, y, C = blobs(seed=5)
    lf = NC.label_free_coords(X, y)                                       # a perfect head: pseudo-labels = labels
    assert abs(lf["plnc1"] - NC.nc_block(X, y)[0]["nc1"]) < 1e-12 and lf["log10_plnc1"] < 0
    assert set(lf) == {"plnc1", "log10_plnc1", "g_cv", "log10_g_cv", "topk_frac", "g_raw"}
    rec, Ctr, _ = NC.nc_block(X, y)
    agree, ncc = NC.ncc_agree(X, Ctr, y)
    assert agree > 0.95 and ncc.shape == (len(X),)
    h1, h2 = NC.split_halves(7)
    assert list(h1) == [0, 2, 4, 6] and list(h2) == [1, 3, 5]


def test_pre_tap_rule_and_duplicates():
    rng = np.random.default_rng(6)
    pen = rng.standard_normal((100, 64))
    assert NC.dup_of_penult(pen.astype(np.float16), pen) and not NC.dup_of_penult(pen + 0.1, pen)
    assert not NC.dup_of_penult(pen[:, :32], pen)
    taps = ["stem", "layer1.0", "layer2.0", "layer3.0", "layer3.1", "penult"]
    spatial = {t: True for t in taps[:-1]}
    assert NC.pre_tap(taps, spatial, ["layer3.1"]) == "layer3.0"
    vgg = ["features.2", "features.3", "features.7", "features.14", "features.21", "features.28", "classifier.1", "penult"]
    sp = {t: t.startswith("features") for t in vgg[:-1]}
    assert NC.pre_tap(vgg, sp, []) == "features.28"                        # pool5, never the FC tap (D2)
    assert NC.pre_tap(["penult"], {}, []) is None
