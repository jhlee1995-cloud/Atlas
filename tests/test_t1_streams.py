"""Known-answer tests for scripts/t1_streams.py (EXTRACTION_PROPOSALS X3; docs/plans/T1_SCOREBOARD.md). numpy + pytest
only. A stream-builder bug once flipped a legacy conclusion (docs/history/SEQUENCE_AXIS_REALDATA_RESULTS.md:35-52), so the
builder, the detectors, the ARL calibration and the scenario bookkeeping each get a known answer here.
  python -m pytest -q tests/test_t1_streams.py
"""
import ast
import importlib.util
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ST_PATH = os.path.join(ROOT, "scripts", "t1_streams.py")


@pytest.fixture(scope="module")
def st():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    spec = importlib.util.spec_from_file_location("t1_streams", ST_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_imports_are_stdlib_numpy_and_the_scoreboard_only():
    tree = ast.parse(open(ST_PATH, encoding="utf-8").read())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add((node.module or "").split(".")[0])
    assert mods <= {"argparse", "math", "os", "shutil", "sys", "tempfile", "time", "traceback", "numpy", "atlas",
                    "t1_scoreboard"}, mods


def test_mewma_in_control_is_chi2_and_a_frozen_frame_freezes_it(st):
    rng = np.random.default_rng(0)
    m = st.mewma(rng.standard_normal((200, 3000, 10)), 0.05)
    assert 8.5 < m[:, 1000:].mean() < 11.5                        # asymptotic covariance lam / (2 - lam) I: ~ chi2_10
    x = rng.standard_normal(10)
    f = st.mewma(np.broadcast_to(x, (4, 400, 10)).copy(), 0.05)    # T1 review A3
    assert f[:, -1].min() > 0.9 * (x @ x) / (0.05 / 1.95)


def test_conformal_cusum_has_no_drift_in_control(st):
    p = np.random.default_rng(1).random((200, 3000))
    c = st.conformal_cusum(p, 0.1)
    assert np.median(c[:, -1]) < 10                                 # E[log(eps p^(eps-1))] = log 0.1 + 0.9 < 0


def test_first_alarm_and_calibration(st):
    stat = np.zeros((3, 10))
    stat[0, 4] = 5
    stat[1, 9] = 5
    assert list(st.first_alarm(stat, 1.0)) == [4, 9, 10]
    assert list(st.first_alarm(stat, 1.0, start=5)) == [10, 9, 10]
    rng = np.random.default_rng(2)
    s = st.cusum(rng.standard_normal((300, 20000)), 0.5)
    h, arl, cens = st.calibrate_h(s, 2000)
    assert abs(arl - 2000) < 60 and cens < 0.05
    assert abs(st.siegmund_arl(0.0, 0.5, h) - arl) / arl <= 0.15


def test_repeat_check_and_builders(st):
    s = st.stuck(np.array([[1, 2, 2, 2, 3, 3]]))
    assert s[0].tolist() == [0, 0, 1, 2, 0, 1] and (s[0] > st.SCEN["stuck_h"]).sum() == 1   # 3 identical frames
    so, ro, ids = st.build([(100, lambda r, n, k: (np.zeros(n, int), r.integers(0, 50, n))),
                            (100, lambda r, n, k: (np.ones(n, int), r.integers(0, 50, n)))], 5, "t")
    assert (so[:, :100] == 0).all() and (so[:, 100:] == 1).all() and ids.shape == (5, 200)
    assert np.array_equal(ro, st.build([(100, lambda r, n, k: (np.zeros(n, int), r.integers(0, 50, n))),
                                        (100, lambda r, n, k: (np.ones(n, int), r.integers(0, 50, n)))], 5, "t")[1])
    a, b = st.stuck_seg(0, np.arange(10))(np.random.default_rng(1), 20, 0)
    assert len(set(b.tolist())) == 1
    labels = np.repeat(np.arange(10), 30)
    _, r = st.single_class(0, np.arange(300), labels)(np.random.default_rng(2), 50, 7)
    assert (labels[r] == 7).all()


def test_window_statistics(st):
    rng = np.random.default_rng(3)
    am = rng.integers(0, 10, (3, 300))
    assert np.array_equal(st.window_counts(am, 10, 64)[1, 299], np.bincount(am[1, 236:300], minlength=10))
    hist = np.full(10, 0.1)
    assert np.allclose(st.bbsdh(am, hist, 64, chunk=1), st.bbsdh(am, hist, 64, chunk=100))   # chunking is exact
    ref = np.sort(rng.random(1500))
    msp = rng.random((7, 400))
    assert np.allclose(st.ks_window(msp, ref, 64, chunk=2), st.ks_window(msp, ref, 64, chunk=50))
    same = st.ks_window(rng.random((50, 400)), ref, 64)[:, 63:].mean()
    shifted = st.ks_window(rng.random((50, 400)) ** 3, ref, 64)[:, 63:].mean()
    assert same < 0.2 < shifted


def test_information_rate(st):
    rng = np.random.default_rng(4)
    r = st.info_rate(rng.standard_normal((20000, 3)) + np.array([1.0, 0, 0]), nboot=20)
    assert abs(r["I_nats"] - 0.5) <= 0.03 and r["ci"][0] < r["I_nats"] < r["ci"][1]
    r0 = st.info_rate(rng.standard_normal((20000, 3)), nboot=20)
    assert r0["I_nats"] < 0.01
    X = rng.standard_normal((20000, 2)) @ np.array([[1.0, 0.0], [0.0, 2.0]])     # variance 4 on one axis
    assert abs(st.info_rate(X, nboot=5)["I_nats"] - 0.5 * (4 - np.log(4) - 1)) < 0.05


def test_end_to_end_on_synthetic_frames(st):
    fr = st.make_synth_frames()
    res = st.analyse(fr, quiet=True, fast=True, nboot=30)
    step = res["scenarios"]["step|corrupt__motion_blur__s1"]
    assert step["mewma_stem"]["ced"] is not None and step["mewma_stem"]["ced"] < 50
    assert step["mewma_stem"]["p_det_within"] >= 0.95 and step["cusum_msp"]["p_det_within"] <= 0.5
    assert step["mewma_stem"]["delay_cens"] is not None
    stuck = res["scenarios"]["stuck|frame"]
    assert stuck["stuck"]["p_det_within"] == 1.0 and stuck["mewma_penult"]["p_det_within"] >= 0.9
    # a segment shorter than the 500-frame window: a stream with no alarm is NOT a detection (the draft counted it)
    assert stuck["cusum_msp"]["p_det_within"] <= 0.7
    arl = res["arl"]["cusum_msp"]
    assert 1000 <= arl["arl0_heldout"] <= 4000 and abs(arl["fa_per_1000_heldout"] - 1000 / arl["arl0_heldout"]) < 1e-9
    rate = res["rate"]["corrupt__motion_blur__s1"]
    assert rate["stem"]["ci"][0] > rate["penult"]["ci"][1]
    assert abs(res["rate"]["test"]["head"]["I_nats"]) < 0.02          # CAL-whitened head scalars (T1 review E3)
    assert res["h"]["stuck"] == 1.0
    # X3-7 judges the skew burst against the in-control null run with the same timing (scripts/t1_eval.js reads both)
    assert {"skew|single_class", "null|clean"} <= set(res["scenarios"])
    assert "p_det_within" in res["scenarios"]["null|clean"]["x4"]
    ev = open(os.path.join(ROOT, "scripts", "t1_eval.js"), encoding="utf-8").read()
    assert '"null|clean"' in ev and '"skew|single_class"' in ev


def test_synthetic_heldout_pool_matches_the_cal_pool(st):
    # the held-out ARL0 known answer carries Monte Carlo noise only: test rows 3500-4999 have the cal pool's exact
    # per-scalar mean and SD (an unmatched 1500-row pool offset fails [1000, 4000] at a fixed seed ~11% of the time)
    fr = st.make_synth_frames()
    rows = np.asarray(fr.get("test", "rows"))
    held = (rows >= 3500) & (rows < 5000)
    for nm in st.SCALARS:
        h, c = np.asarray(fr.get("test", nm))[held], np.asarray(fr.get("cal", nm))
        assert abs(h.mean() - c.mean()) < 1e-9 and abs(h.std() - c.std()) < 1e-9, nm


def test_selftest_passes_including_the_dump_path(st, tmp_path):
    rep = st.selftest(workdir=str(tmp_path), dumps=True)
    assert rep["status"] == "PASS", [c for c in rep["checks"] if not c["pass"]]
    import json
    rec = json.load(open(os.path.join(str(tmp_path), "out", "t1_synth", "streams.json")))
    for k in ("program", "unit", "phase", "code", "timing_s", "max_rss_mb", "arl", "rate", "scenarios", "dumps"):
        assert k in rec, k
    assert rec["program"] == "t1s" and rec["stream_layout"] == "fit" and rec["timing_s"]["total"] > 0
    assert any(k.startswith("tap:") for k in rec["timing_s"]) and any(k.startswith("target:") for k in rec["timing_s"])
