"""Known-answer tests for scripts/b4_reg.py, the batch-4 registry and job list (docs/plans/B4_INTEGRATION.md D3, D5, D9,
D10). Standard library + pytest.
  python -m pytest -q tests/test_b4_reg.py
"""
import importlib.util
import json
import os
import re

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REG = os.path.join(ROOT, "experiments", "b4", "models.json")


@pytest.fixture(scope="module")
def br():
    spec = importlib.util.spec_from_file_location("b4_reg", os.path.join(ROOT, "scripts", "b4_reg.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    # hermetic: S2 runs pytest from the repo root, where experiments/b4/timeouts.json (P2) exists and would override
    # the defaults these tests assert; only test_timeouts_and_tags points TIMEOUTS at its own tmp file
    m.TIMEOUTS = os.path.join(HERE, "__no_such_timeouts__.json")
    return m


def load(br, cut=""):
    return br.load(REG, cut)


def test_registry_is_valid_and_ascii(br):
    assert br.validate(load(br)) == []
    raw = open(REG, "rb").read()
    assert all(b < 128 for b in raw)


def test_roles_d3(br):
    r = load(br)
    assert br.ids(r, ["ANCHOR"]) == ["resnet20_hub", "resnet56_hub"]                    # rule 6: resnet20 first
    c = br.ids(r, ["C"])
    assert len(c) == 12 and not {"resnet20", "resnet56"} & {m.split("_")[0] for m in c}
    assert br.ids(r, ["Dnew"]) == ["resnet32", "vgg11_bn", "mobilenetv2_x0_5", "shufflenetv2_x0_5", "repvgg_a0"]
    for m in r["models"]:
        if "C" in m["roles"] or "F" in m["roles"] or "K" in m["roles"] or "F20" in m["roles"]:
            assert m["layouts"]["fit"]["sealed"] and m["layouts"]["fit"]["dump"].startswith("results/b4c_")
        if "D" in m["roles"] or "N" in m["roles"] or "Dnew" in m["roles"]:
            assert not m["layouts"]["fit"]["sealed"] and m["layouts"]["fit"]["dump"].startswith("results/b4d_")
        if m["source"] == "hub":
            assert re.fullmatch(r"cifar10_\w+-[0-9a-f]{8}\.pt", m["hub_asset"]["file"]) and m["hub_asset"]["bytes"] > 1e6


def test_cuts(br):
    assert "resnet56_s31_wd5e5" not in br.ids(load(br, "Kwd"), ["K"])
    r2 = load(br, "R2")
    assert br.ids(r2, ["R2"]) == [] and "resnet20_s1" in br.ids(r2, ["D"])            # R2 drops a layout, not the unit
    f20 = load(br, "F20")                                                           # a kind: the unit goes entirely
    assert br.ids(f20, ["F20"]) == [] and "resnet20_s31" not in br.ids(f20, ["STconf"]) and len(br.ids(f20, ["STconf"])) == 8
    with pytest.raises(SystemExit):
        load(br, "C")                                                                   # never cut (section 2)


def test_discovery_jobs(br):
    r = load(br)
    J = br.jobs(r, br.parse_limits(br.DEFAULT_LIMITS), "", "discovery", "")
    assert J[0].startswith("timeout 5400 python scripts/t1_scoreboard.py") and "--unit resnet20_hub" in J[0]
    assert "--unit resnet20_hub" in J[1] and "collapse_probe.py" in J[1] and "t3s_spatial_probe.py" in J[2]
    units = {re.search(r"--unit (\S+)", j).group(1) for j in J}
    sealed = set(br.ids(r, ["C", "F", "F20", "K"]))
    assert not units & sealed and set(br.ids(r, ["D", "N", "Dnew"])) <= units
    for j in J:
        assert "--registry" in j and "--phase discovery" in j and "--out results/b4_" in j
        if "t1_scoreboard" in j:
            assert "--layout fit" in j
        if "t1_streams" in j:
            assert "--frames /root/b4_frames/" in j
    t2 = [j for j in J if "collapse_probe" in j]
    assert len(t2) == 16 + 2 + 5
    assert any("timeout 7200" in j and "vgg11_bn" in j for j in J if "t1_scoreboard" in j)   # wide unit: t1_wide
    assert any("timeout 3600" in j and "vgg11_bn" in j for j in t2)                           # wide unit: t2_wide


def test_replay_reprobe_and_confirmation_jobs(br):
    r = load(br)
    lim = br.parse_limits(br.DEFAULT_LIMITS)
    rep = br.jobs(r, lim, "", "replay", "")
    assert len(rep) == 6 and all("_s2replay" in j and "--phase discovery" in j for j in rep)
    rp = br.jobs(r, lim, "", "reprobe", "_r2")
    assert all(re.search(r"--out \S+_r2_p2", j) for j in rp)
    C = br.jobs(r, lim, "/logs", "confirmation", "")
    assert all("--phase confirmation" in j and "> /logs/b4_" in j for j in C)
    assert "collapse_probe" in C[0] and "--unit resnet44" in C[0]                  # the owner's hypothesis first
    t1eval = [j for j in C if "t1_scoreboard" in j and "--layout eval" in j]
    assert {re.search(r"--unit (\S+)", j).group(1) for j in t1eval} == set(br.ids(r, ["F", "F20", "R2"]))
    assert all("timeout 7200" in j for j in t1eval)
    streams = {re.search(r"--unit (\S+)", j).group(1) for j in C if "t1_streams" in j}
    assert streams == set(br.ids(r, ["STconf"])) and len(streams) == 10
    assert {re.search(r"--unit (\S+)", j).group(1) for j in C if "t3s_spatial" in j} == set(br.ids(r, ["Sconf"]))
    assert not any("--unit resnet56_e40" in j for j in C)                           # discovery-only unit


def test_timeouts_and_tags(br, tmp_path, monkeypatch):
    r = load(br)
    lim = br.parse_limits("t1=100 t1_wide=200 t2=10 t2_wide=20 t1s=30 t3s=40")
    assert br.timeout(lim, "t1", "resnet44", r) == 100 and br.timeout(lim, "t1", "resnet44", r, "eval") == 200
    assert br.timeout(lim, "t2", "repvgg_a2", r) == 20 and br.timeout(lim, "t3s", "resnet20_s3", r) == 40
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(br, "TIMEOUTS", "experiments/b4/timeouts.json")
    os.makedirs("experiments/b4")
    json.dump({"t1": {"resnet44": 4321, "resnet56_s31_eval": 9999}, "peak_gb": {"t1": 3.5}},
              open("experiments/b4/timeouts.json", "w"))
    assert br.timeout(lim, "t1", "resnet44", r) == 4321 and br.timeout(lim, "t1", "resnet56_s31", r, "eval") == 9999
    assert br.timeout(lim, "t2", "resnet44", r) == 10
    with pytest.raises(SystemExit):
        br.main(["--registry", REG, "ids", "C", "--tag", "_bad"])


def test_gb_groups(br, capsys):
    tot = 0
    for g in ("disc", "C", "F", "K", "R2", "Sconf"):
        br.main(["--registry", REG, "gb", "--group", g])
        v = int(capsys.readouterr().out.strip())
        assert v >= 1
        tot += v
    assert 15 <= tot <= 26                                                            # D11: ~18-19 GB new (+15 %)


def test_replay_compare_tolerance(br, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = load(br)
    base = {"x": 1.0, "y": [1, 2, {"z": 0.5}], "timing_s": {"total": 5}, "code": {"sha256": "a"}}
    for i in br.ids(r, ["ANCHOR"]):
        for prog in ("t1", "t2", "t3s"):
            _, pat, fn = br.OUT[prog]
            for tag, extra in (("", {}), ("_s2replay", {"timing_s": {"total": 9}, "code": {"sha256": "b"},
                                                        "x": 1.0 + 1e-9})):
                d = pat.format(id=i, layout="fit", tag=tag)
                os.makedirs(d)
                json.dump({**base, **extra}, open(os.path.join(d, fn), "w"))
    rep = br.replay_compare(r, "replay.json", "", "")
    assert rep["status"] == "PASS" and rep["tags"] == {"s1": "", "s2": ""}
    p = os.path.join(br.OUT["t2"][1].format(id="resnet20_hub", layout="fit", tag="_s2replay"), "probe.json")
    json.dump({**base, "x": 1.01}, open(p, "w"))
    rep = br.replay_compare(r, "replay2.json", "", "")
    assert rep["status"] == "FAIL" and rep["units"]["t2:resnet20_hub"]["n_diff"] == 1
    # a D10 relaunch of S2 (ATLAS_B4_TAG=_r2) still compares against the S1 outputs (tag ''), not <id>_r2
    for i in br.ids(r, ["ANCHOR"]):
        for prog in ("t1", "t2", "t3s"):
            _, pat, fn = br.OUT[prog]
            d = pat.format(id=i, layout="fit", tag="_r2_s2replay")
            os.makedirs(d)
            json.dump({**base, "timing_s": {"total": 7}}, open(os.path.join(d, fn), "w"))
    rep = br.replay_compare(r, "replay3.json", "_r2", "")
    assert rep["status"] == "PASS" and rep["tags"] == {"s1": "", "s2": "_r2"}
    assert br.replay_compare(r, "replay4.json", "_r2", "_r2")["status"] == "FAIL"    # no S1 output under _r2
    with pytest.raises(SystemExit):
        br.main(["--registry", REG, "replay-compare", "--out", "x.json", "--s1-tag", "_bad"])


def test_timing_collects_probe_records(br, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = load(br)
    os.makedirs("results/b4_t1/vgg11_bn_fit")
    json.dump({"unit": "vgg11_bn", "layout": "fit", "timing_s": {"total": 12.5}, "max_rss_mb": 900.0},
              open("results/b4_t1/vgg11_bn_fit/scoreboard.json", "w"))
    os.makedirs("results/b4_t2/resnet20_hub_r2")
    json.dump({"timing_s": {"total": 3.0}}, open("results/b4_t2/resnet20_hub_r2/probe.json", "w"))
    br.timing(r, "timing.json")
    t = json.load(open("timing.json"))
    assert t["t1"]["vgg11_bn"]["wall_s"] == 12.5 and t["t2"]["resnet20_hub"]["wall_s"] == 3.0
