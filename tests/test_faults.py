"""Known-answer tests for atlas/faults.py (docs/plans/B4_INTEGRATION.md D5): every fault kind is deterministic in its key,
touches exactly its region, and has the registered strength; the held-out families are the D7 confirmation-only faults.
numpy + pytest only.
  python -m pytest -q tests/test_faults.py
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from atlas import faults as F            # noqa: E402
from atlas import b4_core as C           # noqa: E402


@pytest.fixture(scope="module")
def imgs():
    return np.random.default_rng(0).integers(0, 256, (32, 32, 32, 3)).astype(np.uint8)


def test_t1_faults_deterministic_and_salted(imgs):
    for kind, lv in F.T1_LEVELS.items():
        for level in lv:
            a = F.t1_fault(imgs, kind, level)
            assert a.dtype == np.uint8 and a.shape == imgs.shape
            assert np.array_equal(a, F.t1_fault(imgs, kind, level))
            if kind != "exposure_global":
                assert not np.array_equal(a, F.t1_fault(imgs, kind, level, salt=1))    # eval layout: unseen geometry
    with pytest.raises(ValueError):
        F.t1_fault(imgs, "occlusion", 1)                                               # renamed kinds only (D5)


def test_deadpix_fraction_values_and_fixed_position(imgs):
    for level, frac in F.T1_LEVELS["deadpix"].items():
        d = F.t1_fault(imgs, "deadpix", level)
        m = F.t1_fault_mask("deadpix", level)
        assert m.sum() == round(frac * 1024)
        assert set(np.unique(d[:, m, :])) <= {0, 255}
        assert np.array_equal(d[:, ~m, :], imgs[:, ~m, :])                            # nothing else touched
        other = F.t1_fault(np.full((2, 32, 32, 3), 128, np.uint8), "deadpix", level)
        assert np.array_equal((other != 128).any(axis=(0, 3)), m & (other != 128).any(axis=(0, 3)))


def test_occlusion_disc_area_and_exposure():
    for level, area in F.T1_LEVELS["occlusion_disc"].items():
        o = F.t1_fault(np.full((1, 32, 32, 3), 200, np.uint8), "occlusion_disc", level)
        assert abs(float((o[0, :, :, 0] == F.OCCLUSION_VALUE).mean()) - area) <= 0.02
        assert np.array_equal(o[0, :, :, 0] == F.OCCLUSION_VALUE, F.t1_fault_mask("occlusion_disc", level))
    e = F.t1_fault(np.full((2, 32, 32, 3), 200, np.uint8), "exposure_global", 1)
    assert int(e.min()) == 255
    e2 = F.t1_fault(np.full((1, 32, 32, 3), 100, np.uint8), "exposure_global", 1)
    assert int(e2.max()) == 150 and F.t1_fault_mask("exposure_global", 1) is None


def test_lane_s_regions_and_strengths(imgs):
    clean, src = imgs[0], imgs[1]
    for area, side in F.S_SIDES.items():
        p, m = F.s_fault("paste", clean, area, 17, paste_src=src)
        assert m.sum() == side * side and np.array_equal(p[m], src[m]) and np.array_equal(p[~m], clean[~m])
        o, mo = F.s_fault("occluder_sq", clean, area, 17)
        assert mo.sum() == side * side and len(np.unique(o[mo].reshape(-1, 3), axis=0)) == 1  # one flat colour
        g, mg = F.s_fault("glare", clean, area, 17)
        assert np.array_equal(g[~mg], clean[~mg]) and (g[mg].astype(int) >= clean[mg].astype(int)).all()
        dd, md = F.s_fault("dead_sq", clean, area, 17)
        z = (dd[md] == 0).all(axis=1).mean()
        assert 0.3 < z <= 1.0 and np.array_equal(dd[~md], clean[~md])
    s, ms = F.s_fault("soiling", np.full((32, 32, 3), 200, np.uint8), "a12", 3)
    assert s[ms].mean() < 190 and (s[~ms] >= 190).all() and s[~ms].mean() > 199      # dark blob, soft fringe
    with pytest.raises(ValueError):
        F.s_fault("paste", clean, "a12", 0)
    with pytest.raises(ValueError):
        F.s_fault("occluder", clean, "a12", 0)                                         # renamed: occluder_sq


def test_lane_s_keyed_by_row(imgs):
    rows = np.arange(100, 132)
    a, ma = F.s_fault_batch("occluder_sq", "a12", imgs, rows)
    b, mb = F.s_fault_batch("occluder_sq", "a12", imgs, rows)
    assert np.array_equal(a, b) and np.array_equal(ma, mb)
    assert len({tuple(np.argwhere(m)[0]) for m in ma}) > 5                              # positions vary with the row
    c, mc = F.s_fault_batch("occluder_sq", "a12", imgs[::-1], rows[::-1])
    assert np.array_equal(mc[::-1], ma)                                                 # the row, not the order, keys it


def test_conditions_and_holdout():
    assert len(F.S_CONDITIONS) == 21 and len(F.S_GLOBALS) == 7
    assert F.parse_condition("paste__gaussian_noise__a12") == ("paste", "gaussian_noise", "a12")
    assert F.parse_condition("occluder_sq__a06") == ("occluder_sq", None, "a06")
    assert F.parse_condition("global__fog__s3") == ("global", "fog", 3)
    with pytest.raises(ValueError):
        F.parse_condition("occluder__a12")
    conf, disc = F.s_conditions(True), F.s_conditions(False)
    assert set(conf) - set(disc) == {"soiling__a12", "soiling__a25"} and len(disc) == 19
    assert all(C.is_confirmation_only(f"fault__{c}") for c in set(conf) - set(disc))
    assert not any(C.is_confirmation_only(f"fault__{c}") for c in disc)
    assert all(p in C.DISC for p in F.S_PASTE) and all(g in C.DISC for g in F.S_GLOBALS)   # discovery material only
    assert set(F.CONFIRMATION_ONLY_FAULTS) <= set(C.CONFIRMATION_ONLY)
