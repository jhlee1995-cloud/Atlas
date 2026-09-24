"""Known-answer tests for scripts/t3s_spatial_probe.py, the lane-S part of batch 4 (docs/plans/T3S_SPATIAL.md section 4;
docs/plans/B4_INTEGRATION.md D5, D7). numpy + pytest for the planted synthetic dumps; the extractor round trip needs torch
(skipped without it); the evaluator schema check needs node (skipped without it).
  python -m pytest -q tests/test_t3s_spatial.py
"""
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from atlas import b4_core as C           # noqa: E402
from atlas import faults as FAULTS       # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def P():
    return _load("t3s_spatial_probe")


class _GitOK:
    returncode, stdout = 0, "synthetic_head\n"


def _run(P, regp, unit, phase, out):
    P.main(["--registry", regp, "--unit", unit, "--phase", phase, "--out", out])
    with open(os.path.join(out, "probe.json")) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def planted(P, tmp_path_factory):
    """One open dump that nevertheless holds soiling (never readable in discovery), one sealed confirmation dump, and
    the discovery record of the open one."""
    root = tmp_path_factory.mktemp("t3s_planted")
    dd = P.synth_maps_dump(str(root / "b4d_pd_maps" / "dump"), "pd", sealed=False, holdout=True, roles=["Sdisc"])
    cd = P.synth_maps_dump(str(root / "b4c_pc_maps" / "dump"), "pc", sealed=True, holdout=True, seed=1)
    regp = P.synth_registry(str(root / "models.json"), [("pd", dd, False, False, True), ("pc", cd, True, True, True)])
    rec = _run(P, regp, "pd", "discovery", str(root / "out" / "pd"))
    return {"root": root, "reg": regp, "rec": rec}


# ---------------------------------------------------------------------------------------------------------------------
# frozen sets, registry contract
# ---------------------------------------------------------------------------------------------------------------------
def test_frozen_condition_sets(P):
    assert set(P.F5) | set(P.LOCAL_A12) | set(P.SOIL) <= set(FAULTS.S_CONDITIONS)
    assert P.PASTE4 == tuple(f"paste__{c}__a12" for c in FAULTS.S_PASTE)
    assert P.GLOBAL == tuple(f"global__{c}__s3" for c in FAULTS.S_GLOBALS) and FAULTS.S_GLOBAL_SEVERITY == 3
    assert all(C.is_confirmation_only("fault__" + s) for s in P.SOIL)                    # held out (D7)
    assert not any(C.is_confirmation_only("fault__" + s) for s in P.LOCAL_A12)
    assert not any(C.is_confirmation_only(g) for g in P.GLOBAL)
    assert [n for n, *_ in P.INCREMENTS][:4] == ["SP2_map_over_pooled", "SP3_map_over_pixels", "SP7_std_over_pooled",
                                                 "SP8_typing"]                          # = scripts/t3s_eval.js RULES.inc
    assert P.POOLED_KEYS == ("d1", "knn_l2", "gap_maha", "msp", "maxlogit", "gap", "energy", "entropy")
    assert set(P.HEAD_KEYS) <= set(C.HEAD_STATS) and set(P.RAW_HEAD) <= set(C.HEAD_STATS)


def test_registry_lane_s_units():
    reg = C.load_registry("experiments/b4/models.json")
    sd = [m for m in reg["models"] if "Sdisc" in m["roles"]]
    sc = [m for m in reg["models"] if "Sconf" in m["roles"]]
    assert sorted(m["id"] for m in sd) == ["resnet20_hub", "resnet20_rand", "resnet56_hub", "resnet56_rand"]
    assert sorted(m["id"] for m in sc) == ["resnet20_s3", "resnet20_s4", "resnet56_s12m", "resnet56_s13m"]
    for m in sd:
        L = m["layouts"]["maps"]
        assert not L["sealed"] and not L["holdout_faults"] and L["pixels"] and L["dump"].startswith("results/b4d_")
    for m in sc:
        L = m["layouts"]["maps"]
        assert L["sealed"] and L["holdout_faults"] and L["pixels"] and L["dump"].startswith("results/b4c_")
        assert L["maps"] == ["headmap"]
    assert all(m["layouts"]["maps"]["maps"] == ["headmap", "stage2map"] for m in sd if m["trained"])
    assert reg["rows"]["maps"]["cal"] == [7500, 8500] and reg["rows"]["maps"]["eval"] == [8500, 9500]
    assert reg["rows"]["maps"]["faults"] == [8500, 9500] and reg["rows"]["maps"]["c10c"] == [8500, 9500]


# ---------------------------------------------------------------------------------------------------------------------
# helpers: known answers
# ---------------------------------------------------------------------------------------------------------------------
def test_gauss_known_answers(P):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20000, 5, 3))
    g = P.Gauss(X, shrink=0.0)
    s = g.score(X)
    assert s.shape == (20000, 3) and np.allclose(s.mean(0), 5 * (20000 - 1) / 20000, rtol=1e-9)   # exact identity
    Y = X[:10].copy()
    Y[:, :, 1] += 10.0                                                                  # only position 1 moves
    t = g.score(Y)
    assert np.allclose(t[:, 0], s[:10, 0]) and np.allclose(t[:, 2], s[:10, 2]) and (t[:, 1] > s[:10, 1] + 50).all()
    iso = P.Gauss(X, shrink=1.0)                                                        # Sigma = tr(S)/C I
    Z = X[:, :, 0] - X[:, :, 0].mean(0)
    v = np.trace(Z.T @ Z / (len(Z) - 1)) / 5
    assert np.allclose(iso.score(X)[:, 0], (Z ** 2).sum(1) / v)
    assert g.nbytes32() == 3 * (5 + 25) * 4


def test_map_stats_and_gini(P):
    assert abs(P.gini_rows(np.ones((1, 64)))[0]) < 1e-12
    assert abs(P.gini_rows(np.eye(64)[:1])[0] - 63 / 64) < 1e-12
    assert P.gini_rows(np.zeros((1, 64)))[0] == 0.0
    M = np.arange(64, dtype=float)[None, :]
    st = P.map_stats(M, np.full(64, 60.5), "map")
    assert st["map_max"][0] == 63 and st["map_top"][0] == 62 and abs(st["map_area"][0] - 3 / 64) < 1e-12
    assert P.n_top(64) == 3 and P.n_top(256) == 13
    assert set(P.map_stats(M, None, "s2_map")) == {"s2_map_max", "s2_map_top"}


def test_cells_dilation_pointing(P):
    m = np.zeros((2, 32, 32), bool)
    m[0, 0:11, 0:11] = True                                     # corner a12 square: 3 x 3 cells
    m[1, 13:24, 13:24] = True                                   # rows / cols 13-23 -> cells 3..5: 3 x 3 cells
    cm = P.cell_masks(m, 8, 8)
    assert cm.sum(1).tolist() == [9, 9] and cm[0, 0] and cm[1, 3 * 8 + 3] and not cm[1, 2 * 8 + 2]
    cd = P.dilate_cells(cm, 8, 8)
    assert cd.sum(1).tolist() == [16, 25]                       # corner: no wrap-around
    hit, ch = P.pointing([0, 0], cm)
    assert hit == 0.5 and ch == 9 / 64
    assert P.cell_masks(m, 16, 16)[0].sum() == 36               # 2-px cells: 6 x 6


def test_pixel_cells_known(P):
    flat = P.pixel_cells(np.full((1, 32, 32, 3), 77, np.uint8))
    assert flat.shape == (1, 7, 64) and np.allclose(flat[0, 0:3], 77 / 255) and np.allclose(flat[0, 3:6], 0)
    assert np.allclose(flat[0, 6], math.log(P.LAP_EPS))
    im = np.zeros((1, 32, 32, 3), np.uint8)
    im[0, 8:12, 20:24] = 255                                    # cell (row 2, col 5)
    pc = P.pixel_cells(im)
    assert int(np.argmax(pc[0, 0])) == 2 * 8 + 5 and pc[0, 0, 2 * 8 + 5] == 1.0
    assert P.sat_frac(im)[0] == 1.0                             # every pixel is black or white
    assert P.lap_logvar(np.full((1, 32, 32, 3), 9, np.uint8))[0] == pytest.approx(math.log(P.LAP_EPS))


def test_identity_check(P):
    rng = np.random.default_rng(3)
    m = np.abs(rng.normal(size=(20, 8, 8, 8))).astype(np.float16)
    pen = m.astype(np.float64).mean((2, 3))
    std = m.astype(np.float64).reshape(20, 8, -1).std(2, ddof=1)
    assert P.identity_check(m, pen.astype(np.float32), std.astype(np.float16))["pass"]
    assert not P.identity_check(m, pen * 1.05, std)["pass"]                           # another tap / scale
    assert not P.identity_check(m, pen, std * np.sqrt(63 / 64) * 0.9)["pass"]


# ---------------------------------------------------------------------------------------------------------------------
# the probe on planted synthetic dumps
# ---------------------------------------------------------------------------------------------------------------------
def test_planted_known_answers(P, planted):
    r = planted["rec"]
    pc, ka = r["per_condition"], r["known_answers"]
    assert r["program"] == "t3s" and r["schema"] == P.SCHEMA and r["phase"] == "discovery" and r["nboot"] == 200
    assert not any("soiling" in k for k in pc) and not any("soiling" in s for s in r["dumps"][0]["splits_read"])
    assert len(pc) == 19 + 7 and all(v["valid"] for v in pc.values())
    assert all(ka[k]["pass"] for k in ("map_gap_identity", "pairing", "mask_area", "pixel_hash"))
    assert all(v["exact"] for v in ka["mask_area"]["per_condition"].values() if v["expected_px"])
    for c in P.F5:
        assert pc[c]["auroc"]["map_max"] >= 0.99 and pc[c]["auroc"]["std_half"] >= 0.9
        assert all(pc[c]["auroc"][k] == 0.5 for k in P.POOLED_KEYS)                     # the pooled head is blind
        assert pc[c]["pointing"]["hit"] >= 0.95 and pc[c]["pointing"]["chance"] < 0.35
        assert pc[c]["still_right"]["n"] >= 50 and pc[c]["still_right"]["auroc"]["msp"] == 0.5
    assert all(pc["glare__a12"]["auroc"][k] == 0.5 for k in P.MAP_KEYS + P.STD_KEYS + P.POOLED_KEYS)   # invisible
    assert all(pc[g]["auroc"]["d1"] >= 0.99 for g in P.GLOBAL)                           # the pooled head sees globals
    inc = r["increments"]
    assert inc["SP2_map_over_pooled"]["call"] == "ADDS" and inc["SP7_std_over_pooled"]["call"] == "ADDS"
    assert inc["SP2_map_over_pooled"]["conds_pos"] == list(P.F5) and inc["SP2_map_over_pooled"]["conds_neg"] == ["eval"]
    assert inc["SP2_map_over_pooled"]["n_pos"] == 5 * r["n"]["eval"] and not inc["SP3_map_over_pixels"]["skipped"]
    assert inc["SP8_typing"]["conds_neg"] == list(P.GLOBAL) and inc["INFO_soiling_map_over_pooled"]["skipped"]
    assert not inc["INFO_s2_over_map"]["skipped"]
    assert r["maps_shape"] == {"headmap": [8, 8, 8], "stage2map": [4, 16, 16]}
    assert 0.0 <= r["conformal"]["fpr_eval"]["map_max"] <= 0.15 and len(r["conformal"]["curve_map_max"]) == 4
    assert r["functional"]["reference_bytes_float32"]["map_padim"] == 64 * (8 + 64) * 4
    assert "map_max_ms_per_1000" in r["functional"]["timing_s"]
    assert {"tap:headmap", "tap:penult", "tap:pixels", "target:SP2_map_over_pooled", "total"} <= set(r["timing_s"])
    assert "max_rss_mb" in r and len(r["code"]["sha256"]) == 64 and "repo_commit" in r["code"]
    assert r["collapse"]["nc1_ref"] is not None and r["clean"]["acc_eval"] > 0.9


def test_replay_determinism(P, planted):
    rec2 = _run(P, planted["reg"], "pd", "discovery", str(planted["root"] / "out" / "pd_s2replay"))
    bad = []
    _load("b4_reg").same(planted["rec"], rec2, "", bad)          # the S2 replay compare (rel 1e-6, abs 1e-9)
    assert bad == []


def test_seal_touched_once_and_soiling(P, planted, monkeypatch):
    root, regp = planted["root"], planted["reg"]
    monkeypatch.delenv("ATLAS_B4_UNSEAL", raising=False)
    with pytest.raises(C.ReadRefused):
        _run(P, regp, "pc", "discovery", str(root / "out" / "pc_disc"))
    with pytest.raises(C.ReadRefused):
        _run(P, regp, "pc", "confirmation", str(root / "out" / "pc_nounseal"))            # no P2 given
    monkeypatch.setenv("ATLAS_B4_UNSEAL", "p2_synthetic")
    monkeypatch.setenv("ATLAS_B4_CHECK_DIR", str(root / "check"))
    monkeypatch.setattr(C, "_git", lambda *a: _GitOK())
    r = _run(P, regp, "pc", "confirmation", str(root / "out" / "pc"))
    assert r["nboot"] == 1000 and r["env"]["unseal"] == "p2_synthetic" and r["dumps"][0]["sealed"]
    assert r["per_condition"]["soiling__a12"]["auroc"]["map_max"] >= 0.99
    assert not r["increments"]["INFO_soiling_map_over_pooled"]["skipped"]
    with pytest.raises(C.ReadRefused):
        _run(P, regp, "pc", "confirmation", str(root / "out" / "pc_r2"))                  # touched once
    log = [json.loads(x) for x in open(root / "check" / "unseal_log.jsonl")]
    assert log and log[-1]["p2"] == "p2_synthetic"


def test_no_pixels_skips_the_pixel_claim(P, tmp_path):
    d = P.synth_maps_dump(str(tmp_path / "b4d_np_maps" / "dump"), "np", s2=False, n_ref=120, n_cal=60, n_eval=60)
    shutil.rmtree(os.path.join(d, "pixels"))
    regp = P.synth_registry(str(tmp_path / "models.json"), [("np", d, False, False, True)])
    r = _run(P, regp, "np", "discovery", str(tmp_path / "out" / "np"))
    assert r["increments"]["SP3_map_over_pixels"]["skipped"] and r["increments"]["INFO_s2_over_map"]["skipped"]
    assert r["known_answers"]["pixel_hash"]["pass"] is False and not r["keys"]["pix"]
    assert not r["increments"]["SP2_map_over_pooled"]["skipped"]


def test_output_guard_append_only(P, planted):
    with pytest.raises(C.ReadRefused):
        _run(P, planted["reg"], "pd", "discovery", str(planted["root"] / "out" / "pd"))   # exists: append-only


def test_eval_js_reads_the_record(P, planted):
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed (the Windows evaluation reads the same fields)")
    f = str(planted["root"] / "out" / "pd" / "probe.json")
    r = subprocess.run([node, os.path.join(ROOT, "scripts", "t3s_eval.js"), "--check-record", f], capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_selftest_passes(P, tmp_path):
    rep = P.selftest(workdir=str(tmp_path / "st"))
    assert rep["status"] == "PASS", [c for c in rep["checks"] if not c["pass"]]
    assert len(rep["checks"]) >= 20


# ---------------------------------------------------------------------------------------------------------------------
# the real extractor's maps layout (torch, CPU): the probe's conventions hold on what b4_extract.py writes
# ---------------------------------------------------------------------------------------------------------------------
def test_extractor_maps_dump_round_trip(P, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    ex = _load("b4_extract")
    spec = ex.synth_spec(str(tmp_path), uid="xs", maps_conf=False)
    spec["layouts"]["maps"].update(sealed=False, dump=str(tmp_path / "b4d_xs_maps" / "dump"))
    ex.extract(spec, "maps", ex.SynthData(), spec["layouts"]["maps"]["dump"], "cpu", False, ex.SMALL_ROWS, batch=32)
    conf = ex.synth_spec(str(tmp_path), uid="xc", maps_conf=True)
    ex.extract(conf, "maps", ex.SynthData(), conf["layouts"]["maps"]["dump"], "cpu", True, ex.SMALL_ROWS, batch=32)
    regp = str(tmp_path / "models.json")
    with open(regp, "w") as f:
        json.dump({"models": [spec, conf]}, f)
    r = _run(P, regp, "xs", "discovery", str(tmp_path / "out" / "xs"))
    ka = r["known_answers"]
    assert ka["map_gap_identity"]["pass"], ka["map_gap_identity"]["max_mean_rel"]          # layer3 map -> GAP = penult
    assert ka["map_gap_identity"]["max_std_rel"] <= P.IDENT_TOL                            # unbiased spatial std
    assert ka["pairing"]["pass"] and ka["mask_area"]["pass"] and ka["pixel_hash"]["pass"]
    assert r["maps_shape"]["headmap"] == [8, 8, 8] and r["maps_shape"]["stage2map"] == [8, 16, 16]
    assert len(r["per_condition"]) == 19 + 7 and "soiling__a12" not in r["per_condition"]
    assert r["rows"] == {"cal": [100, 140], "eval": [140, 180]}
    assert all(v["valid"] for k, v in r["per_condition"].items() if not k.startswith("global__"))
    monkeypatch.setenv("ATLAS_B4_UNSEAL", "p2_synthetic")
    monkeypatch.setenv("ATLAS_B4_CHECK_DIR", str(tmp_path / "check"))
    monkeypatch.setattr(C, "_git", lambda *a: _GitOK())
    rc = _run(P, regp, "xc", "confirmation", str(tmp_path / "out" / "xc"))
    assert rc["known_answers"]["map_gap_identity"]["pass"] and "soiling__a12" in rc["per_condition"]
    assert rc["per_condition"]["soiling__a12"]["valid"]
