"""Known-answer tests for scripts/t1_scoreboard.py (docs/plans/T1_SCOREBOARD.md; EXTRACTION_PROPOSALS X1, X2, X4, X6, X7,
X8; the O1 / O5 model outcomes of experiments/b4/model_outcomes.schema.json). numpy + pytest only: no torch, no scipy, no
sklearn, no real dump. The shared numerics (AUC, DeLong, the joint model, the HEAD-ADDITIVE rule and its power / size at
realistic size, conformal p) are tested in tests/test_b4_core.py.
  python -m pytest -q tests/test_t1_scoreboard.py
The end-to-end self-test (synthetic b4 dumps, fast counts, fit layout) is the last test; it takes a few CPU-minutes.
"""
import ast
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SB_PATH = os.path.join(ROOT, "scripts", "t1_scoreboard.py")
EVAL_JS = os.path.join(ROOT, "scripts", "t1_eval.js")
SCHEMA = os.path.join(ROOT, "experiments", "b4", "model_outcomes.schema.json")
sys.path.insert(0, ROOT)

from atlas import b4_core as C          # noqa: E402


@pytest.fixture(scope="module")
def sb():
    spec = importlib.util.spec_from_file_location("t1_scoreboard", SB_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def stub_unit(sb, tables, uid="u"):
    """A Unit without dumps: the table-level functions (gather, joint_eval, o5_record) only need .tables / .caltab."""
    u = sb.Unit.__new__(sb.Unit)
    u.tables, u.uid, u.caltab = tables, uid, tables["test"]
    return u


# ---------------------------------------------------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------------------------------------------------
def test_imports_are_stdlib_numpy_and_atlas_only():
    tree = ast.parse(open(SB_PATH, encoding="utf-8").read())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add((node.module or "").split(".")[0])
    assert mods <= {"argparse", "json", "math", "os", "shutil", "sys", "tempfile", "time", "traceback", "numpy",
                    "atlas"}, mods


def test_import_pulls_no_torch_scipy_sklearn():
    code = ("import sys, importlib.util; sys.path.insert(0, sys.argv[1]); "
            "s = importlib.util.spec_from_file_location('t1', sys.argv[2]); m = importlib.util.module_from_spec(s); "
            "s.loader.exec_module(m); bad = [x for x in ('torch', 'scipy', 'sklearn') if x in sys.modules]; "
            "print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code, ROOT, SB_PATH], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_o1_bundle_is_the_schema_constant(sb):
    schema = json.load(open(SCHEMA))
    assert list(sb.O1_BUNDLE) == schema["$defs"]["O1"]["properties"]["bundle_members"]["const"]
    assert list(sb.BUNDLES["penult"]) == list(sb.O1_BUNDLE)
    assert list(sb.O5_HEADS) == schema["$defs"]["O5"]["properties"]["best_head"]["enum"]
    assert list(C.DISC) == schema["$defs"]["O5"]["properties"]["corruptions"]["const"]
    assert list(sb.LAYOUT["fit"]["test"]) == schema["$defs"]["O1"]["properties"]["rows"]["const"]
    assert list(sb.LAYOUT["fit"]["B"]) == schema["$defs"]["O5"]["properties"]["negatives"]["const"]


def test_every_bundle_signal_has_an_orientation(sb):
    for names in sb.BUNDLES.values():
        for n in names:
            assert sb.base(n) in sb.SIGN, n
    for h in C.HEAD_STATS:
        assert sb.SIGN[h] == C.ERR_SIGN[h]


def test_layouts_follow_the_row_ledger(sb):
    f, e = sb.LAYOUT["fit"], sb.LAYOUT["eval"]
    assert f["CAL"] == (2000, 3500) and f["X4_1"] == (2000, 2750) and f["X4_2"] == (2750, 3500) and f["B"] == (3500, 5000)
    assert e["test"] == (5000, 10000) and e["B"] == (8500, 10000) and e["POS"] == (5000, 8500)
    assert e["CAL"] == (0, 5000) and e["X4_1"] == (0, 2500) and e["X4_2"] == (2500, 5000)   # T1 review A2
    assert f["HEAD_SEL"] == e["HEAD_SEL"] == (2000, 3500)


# ---------------------------------------------------------------------------------------------------------------------
# T1-specific numerics
# ---------------------------------------------------------------------------------------------------------------------
def test_auc_inside_gap_deciles(sb):
    rng = np.random.default_rng(0)
    n = 20000
    gap = rng.standard_normal(n)
    y = rng.random(n) < np.where(gap > 0, 0.7, 0.1)            # errors depend on the gap only (a step at the median)
    a, used = sb.auc_in_deciles(y, gap, gap)
    assert used == 10 and abs(a - 0.5) < 0.03 and C.auc_y(y, gap) > 0.75
    s = y + rng.standard_normal(n)                            # a signal independent of the gap
    a2, _ = sb.auc_in_deciles(y, s, gap)
    assert abs(a2 - C.auc_y(y, s)) < 0.03


def test_ks_identities_and_vectorised_pvalue(sb):
    assert sb.ks_stat([1, 2, 3], [1, 2, 3]) == 0.0 and sb.ks_stat([0, 0], [1, 1]) == 1.0
    assert float(sb.ks_pvalue(0.0, 100, 100)) == 1.0 and float(sb.ks_pvalue(0.5, 200, 200)) < 1e-6
    d = np.array([[0.05, 0.2], [0.5, 0.0]])
    v = sb.ks_pvalue(d, 64, 1500)
    for i in range(2):
        for j in range(2):
            assert abs(v[i, j] - float(sb.ks_pvalue(d[i, j], 64, 1500))) < 1e-15
    a = np.random.default_rng(1).random(64)
    b = np.random.default_rng(2).random(1500)
    assert sb.ks_stat(a, b) == sb.ks_stat_sorted(a, np.sort(b))


def test_simplex_bbse_isotonic(sb):
    assert np.allclose(sb.project_simplex([2.0, 0.0]), [1.0, 0.0])
    p = sb.project_simplex([-1.0, 0.5, 0.2])
    assert abs(p.sum() - 1) < 1e-12 and (p >= 0).all()
    rng = np.random.default_rng(11)
    Cm = np.eye(10) * 0.8 + 0.02                                              # P(yhat = i | y = j)
    Cm /= Cm.sum(0, keepdims=True)
    pi = rng.dirichlet(np.ones(10))
    assert np.allclose(sb.project_simplex(np.linalg.pinv(Cm) @ (Cm @ pi)), pi, atol=1e-9)   # BBSE under pure label shift
    x = np.linspace(0, 1, 200)
    f = sb.isotonic_fit(x, -x + 0.1 * rng.standard_normal(200), increasing=False)
    assert (np.diff(f(x)) <= 1e-12).all()
    g = sb.isotonic_fit(x, 2 * x)
    assert np.allclose(g(x), 2 * x)


def test_batch_idx_is_deterministic_and_distinct(sb):
    a, b = sb.batch_idx("k", 500, 64, 30), sb.batch_idx("k", 500, 64, 30)
    assert np.array_equal(a, b) and a.shape == (30, 64)
    assert all(len(set(r.tolist())) == 64 for r in a)
    assert not np.array_equal(a, sb.batch_idx("other", 500, 64, 30)) and sb.batch_idx("k", 10, 64, 3) is None


def test_call_pix_three_way(sb):
    assert sb.call_pix(0.01, [0.004, 0.02], [0.001, 0.01]) == "ADDS"
    assert sb.call_pix(0.001, [-0.01, 0.015], [-0.01, 0.01]) == "BOUNDED"
    assert sb.call_pix(0.004, [-0.01, 0.03], [-0.01, 0.01]) == "INCONCLUSIVE"
    assert sb.call_pix(None, None, None) == "INCONCLUSIVE"


def test_schema_validator_on_the_outcomes_schema(sb):
    schema = json.load(open(SCHEMA))
    o1 = {"value": 0.001, "ci95": [-0.004, 0.006], "dI_bits": 0.0, "dI_ci95": [-0.001, 0.001], "call": "BOUNDED",
          "bundle": "penult", "bundle_members": list(sb.O1_BUNDLE), "target": "err_clean", "rows": [0, 5000],
          "n_pos": 350, "n_neg": 4650, "nboot": 200}
    o5 = {"value": 0.1, "x4_auroc_mean": 0.7, "best_head": "gap", "best_head_auroc_mean": 0.6, "severity": 3,
          "corruptions": list(C.DISC), "negatives": [3500, 5000],
          "per_corruption": {c: {"x4_auroc": 0.7, "head_auroc": {h: 0.6 for h in sb.O5_HEADS}} for c in C.DISC}}
    rec = {"schema": "b4_model_outcomes/1", "unit": "u", "layout": "fit", "phase": "discovery", "O1": o1, "O5": o5}
    assert sb.validate_schema(rec, schema) == []
    for bad in ({**rec, "layout": "eval"}, {**rec, "O1": {**o1, "bundle_members": ["margin_penult"]}},
                {**rec, "O1": {**o1, "call": "MAYBE"}}, {**rec, "O1": {**o1, "n_pos": 3.5}},
                {k: v for k, v in rec.items() if k != "O5"}, {**rec, "O5": {**o5, "value": "x"}}):
        assert sb.validate_schema(bad, schema), bad
    rec_null = {**rec, "O1": {**o1, "value": None, "ci95": [None, None]}}
    assert sb.validate_schema(rec_null, schema) == []                   # non-finite numbers are written as null


# ---------------------------------------------------------------------------------------------------------------------
# per-tap signals (train-reference structures)
# ---------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ref16():
    rng = np.random.default_rng(3)
    D = 16
    Cm = np.zeros((10, D))
    for k in range(10):
        Cm[k, k] = 6.0
    y = rng.integers(0, 10, 3000)
    X = Cm[y] + rng.standard_normal((3000, D))
    return Cm, X, y


def test_centre_margin_and_distance_match_brute_force(sb, ref16):
    Cm, X, y = ref16
    tr = sb.TapRef(X, y, level="light")
    rng = np.random.default_rng(4)
    Q = Cm[rng.integers(0, 10, 50)] + rng.standard_normal((50, 16))
    am = rng.integers(0, 10, 50)
    s = sb.tap_signals(tr, Q, am, "stem")
    D = np.sqrt(((Q[:, None] - tr.C[None]) ** 2).sum(2))
    Ds = np.sort(D, 1)
    assert np.allclose(s["d1_stem"], Ds[:, 0]) and np.allclose(s["margin_stem"], Ds[:, 1] - Ds[:, 0])
    assert np.allclose(s["ncchead_stem"], D[np.arange(50), am] - Ds[:, 0])
    assert np.array_equal(s["nccdis_stem"], (D.argmin(1) != am).astype(float))
    dn = np.sort(np.sqrt(((C.l2n(Q)[:, None] - C.l2n(X)[None]) ** 2).sum(2)), 1)[:, 9]
    assert np.allclose(s["knnL2_stem"], dn)


def test_full_level_known_answers(sb, ref16):
    Cm, X, y = ref16
    W = Cm.copy()
    b = -0.5 * (W * W).sum(1)
    tr = sb.TapRef(X, y, W, b, level="full")
    k = np.arange(10)
    Q = Cm[k]                                                       # the class centres themselves
    s = sb.tap_signals(tr, Q, k, "penult", Q @ W.T + b)
    assert (s["purity50_penult"] == 1.0).all() and (s["trust_penult"] > 2.0).all()
    assert (s["nc3p_penult"] > 0.99).all()                          # NC3+: x - mu parallel to w_yhat - mean(w)
    Qr = tr.mu + np.array([3.0 * (W[0] - W.mean(0))])             # a query in the head's row space
    sr = sb.tap_signals(tr, Qr, np.array([0]), "penult", Qr @ W.T + b)
    assert sr["snull_penult"][0] < 1e-8 * max(sr["srow_penult"][0], 1.0)
    wrong = sb.tap_signals(tr, Cm[[1]], np.array([0]), "penult", Cm[[1]] @ W.T + b)   # at centre 1, predicted 0
    assert wrong["trust_penult"][0] < 0.5 and wrong["purity10_penult"][0] == 0.0
    assert np.isfinite(s["lid_penult"]).all() and (s["lid_penult"] > 0).all()
    m, rel = tr.maha(tr.mu[None])
    assert abs(rel[0] - m[0]) < 1e-9                                # at the global mean the background term is 0


def test_mahalanobis_drops_dead_directions(sb):
    rng = np.random.default_rng(5)
    y = rng.integers(0, 10, 4000)
    X = np.zeros((4000, 12))
    X[:, :8] = np.eye(10, 8)[y] * 4 + rng.standard_normal((4000, 8))     # 4 dead coordinates (ReLU zeros)
    tr = sb.TapRef(X, y, level="full")
    assert tr.mw[0].shape[1] == 8                                   # pinv semantics: no 1/eps weight on dead axes
    Q = X[:5].copy()
    Q[:, 10] = 1e-3                                                 # a tiny move along a dead axis changes nothing
    assert np.allclose(tr.maha(Q)[0], tr.maha(X[:5])[0])


def test_trajectory_prediction_depth(sb):
    rng = np.random.default_rng(6)
    Cm = np.eye(10, 8) * 5
    y = rng.integers(0, 10, 2000)
    refs = {t: sb.TapRef(Cm[y] + 0.5 * rng.standard_normal((2000, 8)), y, level="centres") for t in ("a", "b", "c", "d")}
    am = np.array([3, 3])
    acts = {"a": Cm[[5, 3]], "b": Cm[[5, 3]], "c": Cm[[3, 3]], "d": Cm[[3, 3]]}   # row 0 settles at tap c
    tj = sb.trajectory(refs, ["a", "b", "c", "d"], lambda t: acts[t], am)
    assert np.allclose(tj["pd"], [2 / 4, 0.0]) and np.allclose(tj["agree"], [0.5, 1.0])
    assert np.allclose(tj["lastdis"], [2 / 4, 0.0])


# ---------------------------------------------------------------------------------------------------------------------
# the joint model and the model outcomes
# ---------------------------------------------------------------------------------------------------------------------
def _table(rng, n, rows):
    Z = rng.standard_normal((n, 10)) * 2
    hs = C.head_stats(Z)
    tb = {"rows": rows, "logits": Z, **{k: v for k, v in hs.items() if k != "argmax"}}
    for nm in ("margin_penult", "d1_penult", "knnL2_penult", "trust_penult", "relmaha_penult", "purity10_penult",
               "purity50_penult", "lid_penult"):
        tb[nm] = rng.standard_normal(n)
    return tb


def test_joint_eval_is_head_increment_exactly(sb):
    """O1 = persample.err_clean.joint.penult is by construction atlas/b4_core.head_increment (the PRIMARY reads it)."""
    rng = np.random.default_rng(7)
    n = 3000
    tb = _table(rng, n, np.arange(n))
    y = (rng.random(n) < sigmoid(-2.0 - 0.5 * tb["gap"] + 0.8 * tb["margin_penult"])).astype(float)
    tb["wrong"] = y
    u = stub_unit(sb, {"test": tb})
    t = {"parts": [("test", np.arange(n))], "y": y, "groups": np.arange(n), "kind": "error"}
    r = sb.joint_eval(u, "err_clean", t, 60, ["penult"])["joint"]["penult"]
    hi = C.head_increment(sb.head_blocks(u, t["parts"]), sb.sig_blocks(u, t["parts"], sb.O1_BUNDLE)[0], y,
                          np.arange(n), 60, "u|err_clean|penult")
    assert r["dauc"] == hi["dauc"] and r["dauc_ci"] == hi["dauc_ci"] and r["dI_ci"] == hi["dI_ci"]
    assert r["dI_bits"] == hi["dI_bits"] and r["call"] == hi["call"] and r["delong_p"] == hi["delong_p"]
    assert r["call"] == "ADDS" and r["dauc"] > 0.01                    # the planted margin effect is found


def test_transfer_target_trains_on_train_ok_only(sb):
    rng = np.random.default_rng(8)
    n = 4000
    tb = _table(rng, n, np.arange(n))
    grp = np.where(np.arange(n) < 2000, "seen", "unseen")
    y = (rng.random(n) < 0.3).astype(float)
    tb["margin_penult"] = np.where(grp == "seen", y * 3.0, -y * 3.0) + rng.standard_normal(n)   # sign flips on unseen
    u = stub_unit(sb, {"test": tb})
    train_ok = grp == "seen"
    t = {"parts": [("test", np.arange(n))], "y": y, "groups": np.arange(n), "kind": "detect", "train_ok": train_ok,
         "eval_mask": ~train_ok}
    r = sb.joint_eval(u, "tr", t, 30, ["margin_pen"])
    assert r["n"] == 2000 and r["joint"]["margin_pen"]["dauc"] < -0.2  # a rule learnt on 'seen' is wrong on 'unseen'


def test_o5_record_known_answer(sb):
    n1 = 200
    tables = {"test": {"rows": np.arange(0, 5000), "x4": np.zeros(5000),
                       **{h: np.zeros(5000) for h in sb.O5_HEADS}}}
    for i, c in enumerate(C.DISC):
        tb = {"rows": np.arange(0, n1), "x4": np.ones(n1), **{h: np.zeros(n1) for h in sb.O5_HEADS}}
        if i < 5:
            tb["gap"] = -np.ones(n1)                          # oriented gap (ERR_SIGN -1) is +1: AUROC 1 on 5 corruptions
        tables[C.csplit(c, 3)] = tb
    u = stub_unit(sb, tables)
    o5 = sb.o5_record(u)
    assert o5["best_head"] == "gap" and abs(o5["best_head_auroc_mean"] - 0.75) < 1e-12
    assert abs(o5["x4_auroc_mean"] - 1.0) < 1e-12 and abs(o5["value"] - 0.25) < 1e-12
    del tables[C.csplit("fog", 3)]
    o5m = sb.o5_record(stub_unit(sb, tables))
    assert o5m["value"] is None and o5m["missing"] == [C.csplit("fog", 3)]      # all 10 corruptions or no value


# ---------------------------------------------------------------------------------------------------------------------
# dump access through the D7 seal
# ---------------------------------------------------------------------------------------------------------------------
def test_discovery_plan_and_the_seal(sb, tmp_path, monkeypatch):
    rp, uid = sb.make_synth(str(tmp_path / "synth"))
    reg = C.load_registry(rp)
    u = sb.Unit(reg, uid, "fit", "discovery", quiet=True)
    pl = u.plan()
    assert "corrupt__fog__s3" in pl and "fault__deadpix__l1" in pl
    assert not any(C.is_confirmation_only(s) for s in pl)                  # holdout, extras, exposure_global
    uc = sb.Unit(reg, uid, "fit", "confirmation", quiet=True)
    assert any("exposure_global" in s for s in uc.plan()) and any("impulse_noise" in s for s in uc.plan())
    assert u.ft["pre"] == "layer3.0" and u.x4keys == ["stem", "pre", "penult"] and "layer3.1" not in u.taps_all
    ev = reg["models"][0]["layouts"]["eval"]["dump"]
    open(os.path.join(ev, "SEALED.json"), "w").write("{}")
    monkeypatch.delenv("ATLAS_B4_UNSEAL", raising=False)
    with pytest.raises(C.ReadRefused):
        sb.Unit(reg, uid, "eval", "discovery", quiet=True)                  # a sealed dump never opens in discovery
    with pytest.raises(C.ReadRefused):
        sb.Unit(reg, uid, "eval", "confirmation", quiet=True)               # ... nor without ATLAS_B4_UNSEAL = P2


# ---------------------------------------------------------------------------------------------------------------------
# the end-to-end self-test and the keys the evaluator reads
# ---------------------------------------------------------------------------------------------------------------------
def test_selftest_end_to_end(sb, tmp_path):
    rep = sb.selftest(workdir=str(tmp_path), fast=True, layouts=("fit",))
    assert rep["status"] == "PASS", rep["failed"]
    board = json.load(open(os.path.join(str(tmp_path), "out", "t1_synth_fit", "scoreboard.json")))
    for k in ("unit", "phase", "layout", "tag", "nboot", "code", "env", "dumps", "timing_s", "max_rss_mb", "head",
              "persample", "calibration", "x4", "do3", "batch", "cost", "model_outcomes", "unit_info"):
        assert k in board, k
    ts = board["timing_s"]
    assert "total" in ts and any(k.startswith("tap:") for k in ts) and any(k.startswith("target:") for k in ts)
    assert len(board["code"]["sha256"]) == 64 and len(board["code"]["core_sha256"]) == 64
    j = board["persample"]["err_clean"]["joint"]["margin_pen"]
    for k in ("dauc", "dauc_ci", "dI_bits", "dI_ci", "call", "dtpr5", "daurc", "auc"):
        assert k in j, k
    assert "call_pix" in board["persample"]["err_shift_s3"]["joint"]["early"]
    for k in ("fpr_B", "n_cal_fusion", "n_B", "tpr"):
        assert k in board["x4"], k
    assert "sparse_frac_test" in board["do3"] and "nc1_train" in board["do3"]
    assert "tpr" in board["calibration"]["gap"] and "best_stat" in board["head"]
    b64 = board["batch"]["sizes"]["64"]
    for k in ("estimators", "partial_H_given_ATC_mean", "skew", "typing", "by_split"):
        assert k in b64, k
    assert "ms_per_1000_queries" in board["cost"]["timing_s"] and board["cost"]["ref_bytes"]["knn_bank_penult"] > 0
    schema = json.load(open(SCHEMA))
    assert sb.validate_schema(board["model_outcomes"], schema) == [], board["model_outcomes"]
    ev = open(EVAL_JS, encoding="utf-8").read()
    for key in ("x4band", "n_cal_fusion", "sparse_frac_test", "partial_H_given_ATC_mean", "mae_pp_loso", "dauc_pix_ci",
                "dI_pix_ci", "famid_", "corrupt_holdout_s3_transfer", "corrupt_extra_s3_transfer", "best_stat",
                "ms_per_1000_queries", "delay_cens", "p_det_within_gated", "lead_median", "arl0_heldout"):
        assert key in ev, key
