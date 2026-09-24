"""Known-answer tests for scripts/collapse_probe.py (docs/plans/T2_COLLAPSE.md section 4; B4_INTEGRATION.md D5, D7):
the synthetic self-test (closed-form nc1 and spectral gap, exchangeable false-alarm rates, harm and overconfidence
recomputed directly, the duplicate-tap exclusion, the D7 seal, the role guard, append-only and replay determinism), the
helper known answers, the import restriction, and a schema pin: every probe.json path scripts/collapse_laws.js reads
(its PROBE_KEYS block) exists in a real probe output.
  python -m pytest -q tests/test_collapse_probe.py
"""
import ast
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import collapse_probe as CP              # noqa: E402
from atlas import b4_collapse as NC      # noqa: E402
from atlas import b4_core as C           # noqa: E402

PROBE = os.path.join(ROOT, "scripts", "collapse_probe.py")
LAWS = os.path.join(ROOT, "scripts", "collapse_laws.js")


# ---------------------------------------------------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------------------------------------------------
def test_imports_are_stdlib_numpy_and_atlas_only():
    tree = ast.parse(open(PROBE, encoding="utf-8").read())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add((node.module or "").split(".")[0])
    assert mods <= {"json", "math", "os", "shutil", "sys", "tempfile", "time", "numpy", "atlas"}, mods


def test_import_does_not_import_torch_scipy_sklearn():
    code = ("import sys; sys.path[:0] = [sys.argv[1], sys.argv[1] + '/scripts']; import collapse_probe; "
            "bad = [m for m in ('torch', 'scipy', 'sklearn', 'torchvision') if m in sys.modules]; print(bad); "
            "sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code, ROOT], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


# ---------------------------------------------------------------------------------------------------------------------
# the synthetic self-test (every check must pass; it is also run by the pod blocks with --selftest-out)
# ---------------------------------------------------------------------------------------------------------------------
def test_selftest_passes(tmp_path):
    rep = CP.selftest(str(tmp_path / "st"))
    failed = [(c["check"], c["value"]) for c in rep["checks"] if c["status"] != "PASS"]
    assert rep["status"] == "PASS" and rep["n_pass"] >= 25, failed


@pytest.fixture(scope="module")
def synth_probe(tmp_path_factory):
    wd = tmp_path_factory.mktemp("t2probe")
    dump = str(wd / "b4d_synth" / "dump")
    CP.make_synth_dump(dump, np.random.default_rng(7))
    reg = {"schema": "b4_models/1", "models": [
        {"id": "unit_d", "roles": ["D"], "family": "synthetic", "depth_family": "synthetic", "run_group": "g",
         "source": "synthetic", "full_recipe": True, "trained": True, "penult": 64, "knob": None,
         "layouts": {"fit": {"dump": dump, "sealed": False}}}]}
    regp = str(wd / "models.json")
    with open(regp, "w") as f:
        json.dump(reg, f)
    out = str(wd / "out" / "unit_d")
    CP._QUIET[0] = True
    try:
        assert CP.main(["--registry", regp, "--unit", "unit_d", "--phase", "discovery", "--out", out]) == 0
    finally:
        CP._QUIET[0] = False
    with open(os.path.join(out, "probe.json")) as f:
        return json.load(f)


def _path(obj, dotted):
    cur = obj
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return False, None
        cur = cur[k]
    return True, cur


def test_laws_read_only_paths_the_probe_writes(synth_probe):
    txt = open(LAWS, encoding="utf-8").read()
    m = re.search(r"PROBE_KEYS-BEGIN(.*?)PROBE_KEYS-END", txt, re.S)
    assert m, "scripts/collapse_laws.js has no PROBE_KEYS block"
    keys = re.findall(r'"([A-Za-z0-9_.]+)"', m.group(1))
    assert len(keys) >= 30
    missing = [k for k in keys if not _path(synth_probe, k)[0]]
    assert not missing, missing


def test_outcomes_and_contract_fields(synth_probe):
    rec = synth_probe
    assert rec["schema"] == CP.SCHEMA and rec["program"] == "t2" and rec["phase"] == "discovery"
    assert rec["outcomes"]["O2"] == rec["targets"]["harm"]["H10"]
    assert rec["outcomes"]["O3"] == rec["targets"]["ood"]["cifar100"]["knn_l2_minus_besthead"]
    assert rec["outcomes"]["O4"] == rec["targets"]["overconf"]["shift_mean"]
    assert rec["timing_s"]["total"] >= max(v for k, v in rec["timing_s"].items() if k != "total")
    assert any(k.startswith("tap:") for k in rec["timing_s"]) and any(k.startswith("target:") for k in rec["timing_s"])
    assert rec["nboot"] == C.NBOOT["discovery"] and "max_rss_mb" in rec
    assert set(rec["targets_se"]) >= {"fpr_trainref", "lead_md", "O2", "O3", "O4", "p6b", "harm_slope"}


# ---------------------------------------------------------------------------------------------------------------------
# helper known answers
# ---------------------------------------------------------------------------------------------------------------------
def test_harm_fit_and_hold_band():
    H = [0.0, 0.1, 0.2, 0.3, 0.4]
    hf = CP.harm_fit(H, [2.0 + 20.0 * h for h in H])
    assert abs(hf["slope"] - 20) < 1e-12 and abs(hf["intercept"] - 2) < 1e-12 and abs(hf["H10"] - 0.4) < 1e-12
    assert hf["n"] == 5 and abs(hf["spearman"] - 1.0) < 1e-12
    assert CP.harm_fit(H, [5.0 - h for h in H])["H10"] is None                 # loss falls with H: no edge
    assert CP.harm_fit([0.1, None], [1.0, 2.0])["slope"] is None                # too few points
    assert CP.hold_band([0.1, 0.2, 0.3, 0.25], [9.0, 3.0, 50.0, 8.0]) == {"hold_n": 3, "hold_viol": 1,
                                                                            "max_cost_in_band": 9.0}
    assert CP.hold_band([0.5], [30.0]) == {"hold_n": 0, "hold_viol": 0, "max_cost_in_band": None}


def test_boot_auc_diff_brackets_and_is_keyed():
    rng = np.random.default_rng(3)
    y = rng.random(1500) < 0.08
    a = rng.standard_normal(1500) + 1.2 * y
    b = rng.standard_normal(1500) + 0.4 * y
    d = C.auc(a[y], a[~y]) - C.auc(b[y], b[~y])
    ci = CP.boot_auc_diff(a, b, y, 400, ("k1",))
    assert ci[0] < d < ci[1] and ci[0] > 0                                      # a is clearly better
    assert CP.boot_auc_diff(a, b, y, 400, ("k1",)) == ci                       # deterministic in the key
    assert CP.boot_auc_diff(a, b, y, 400, ("k2",)) != ci
    assert CP.boot_auc_diff(a, b, np.zeros(1500, bool), 50, ("k",)) == [None, None]


def test_trimmed_nc1_equals_verbatim_when_the_pinv_is_stable():
    rng = np.random.default_rng(4)
    Cm = rng.standard_normal((10, 32)) * 3
    y = rng.integers(0, 10, 4000)
    X = Cm[y] + rng.standard_normal((4000, 32))
    rec, Ctr, Sb = NC.nc_block(X[:2000], y[:2000])
    assert abs(rec["nc1_trim"] / rec["nc1"] - 1) < 1e-8
    held = CP.nc1_trim_around(X[2000:], y[2000:], Ctr, Sb)
    assert abs(held / NC.nc1_around(X[2000:], y[2000:], Ctr, Sb) - 1) < 1e-8


def test_tap_coords_uses_the_core_label_free_definitions():
    rng = np.random.default_rng(5)
    Cm = rng.standard_normal((10, 24)) * 3
    ytr, yte = rng.integers(0, 10, 3000), rng.integers(0, 10, 2000)
    Xtr, Xte = Cm[ytr] + rng.standard_normal((3000, 24)), Cm[yte] + rng.standard_normal((2000, 24))
    am = NC.center_dists(Xte, Cm).argmin(1)
    c, Ctr, _ = CP.tap_coords(Xtr, ytr, Xte, yte, am)
    lf = NC.label_free_coords(Xte, am)
    assert c["g_cv_te"] == lf["g_cv"] and c["topk_frac_te"] == lf["topk_frac"] and c["plnc1_te_verbatim"] == lf["plnc1"]
    assert abs(c["plnc1_te"] / lf["plnc1"] - 1) < 1e-8 and c["dim"] == 24 and 0.9 < c["ncc_agree_te"] <= 1.0
    assert CP.tap_coords(Xtr, np.zeros(3000, int), Xte, yte, am)[0] is None     # one class: undefined


def test_ood_block_orientation_and_best_head():
    pos = {"knn_l2": np.array([3.0, 4.0]), "knn": np.array([3.0, 4.0]),
           **{h: np.array([0.0, 0.1]) for h in CP.HEADS}}
    neg = {"knn_l2": np.array([1.0, 2.0]), "knn": np.array([5.0, 6.0]),
           **{h: np.array([0.0, 0.1]) for h in CP.HEADS}}
    pos["energy"] = np.array([1.0, 1.0])
    o = CP.ood_block(pos, neg)
    assert o["auc_knn_l2"] == 1.0 and o["auc_knn"] == 0.0 and o["best_head"] == "energy" and o["auc_energy"] == 1.0
    assert o["knn_l2_minus_besthead"] == 0.0 and o["knn_l2_minus_msp"] == 0.5


def test_best_by_is_deterministic():
    assert CP._best_by({"a": 0.7, "b": 0.7, "c": None}, ("b", "a", "c")) == "b"
    assert CP._best_by({"a": None}, ("a",)) is None


# ---------------------------------------------------------------------------------------------------------------------
# CLI contract and guards
# ---------------------------------------------------------------------------------------------------------------------
def test_cli_requires_the_contract_arguments():
    with pytest.raises(SystemExit) as e:
        CP.main(["--unit", "x"])
    assert e.value.code == 2


def test_selftest_out_is_never_overwritten(tmp_path):
    p = tmp_path / "selftest.json"
    p.write_text("{}")
    with pytest.raises(SystemExit):
        CP.main(["--selftest", "--selftest-out", str(p)])
    assert p.read_text() == "{}"


def test_role_guard():
    CP.check_role({"id": "u", "roles": ["C"]}, "confirmation")
    CP.check_role({"id": "u", "roles": ["D", "ANCHOR"]}, "discovery")
    with pytest.raises(SystemExit):
        CP.check_role({"id": "u", "roles": ["C"]}, "discovery")
    with pytest.raises(SystemExit):
        CP.check_role({"id": "u", "roles": ["R2"]}, "confirmation")
