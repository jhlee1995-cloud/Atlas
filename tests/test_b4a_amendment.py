"""Known-answer tests for scripts/b4_weights_amend.py (B4a, docs/plans/B4A_AMENDMENT.md): the one-sided README rule
(below, within and above the tolerance, the exact edges, the three S1 cases), its lower side equal to the frozen two-sided
rule of scripts/b4_weights.py bit for bit on every 10k count, the S1-record precondition, the heads directory, the
refusals (all before torch loads), a record the extractor's weights gate accepts, and the structure of the amendment
(module-level imports; gate_unit equal to the frozen one outside the marked B4a CHANGE). numpy + pytest only (the
import test runs a fresh interpreter).
  python -m pytest -q tests/test_b4a_amendment.py
"""
import ast
import importlib.util
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

AMEND = os.path.join(ROOT, "scripts", "b4_weights_amend.py")
FROZEN = os.path.join(ROOT, "scripts", "b4_weights.py")
REGISTRY = os.path.join(ROOT, "experiments", "b4", "models.json")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def A():
    return _load("b4_weights_amend")


@pytest.fixture(scope="module")
def reg():
    with open(REGISTRY) as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------------------------------------
# the rule
# ---------------------------------------------------------------------------------------------------------------------
def test_tolerance_and_units(A, reg):
    assert A.BW.README_TOL == 0.003 and A.RULE["tol"] == 0.003
    assert A.UNITS == ("mobilenetv2_x0_75", "shufflenetv2_x0_5", "shufflenetv2_x1_0")
    by = {m["id"]: m for m in reg["models"]}
    assert [by[u]["roles"] for u in A.UNITS] == [["C"], ["Dnew"], ["C"]]
    assert all(by[u]["source"] == "hub" and by[u]["readme_top1"] is not None for u in A.UNITS)
    assert [by[u]["layouts"]["fit"]["sealed"] for u in A.UNITS] == [True, False, True]


def test_readme_status_exact_edges(A):
    # binary-exact values (tol = 2^-3): the edges are exact, so they test the comparison, not the rounding
    want, tol, e = 0.75, 0.125, 2.0 ** -20
    assert A.readme_status(0.5, want, tol) == ("FAIL", None)                         # below
    assert A.readme_status(0.625 - e, want, tol) == ("FAIL", None)                   # just below the lower edge
    assert A.readme_status(0.625, want, tol) == ("PASS", None)                       # the lower edge passes
    assert A.readme_status(0.75, want, tol) == ("PASS", None)                        # equal
    assert A.readme_status(0.875, want, tol) == ("PASS", None)                       # the upper edge: no flag
    assert A.readme_status(0.875 + e, want, tol) == ("PASS", "README-EXCEEDED")      # just above: PASS + INFO
    assert A.readme_status(1.0, want, tol) == ("PASS", "README-EXCEEDED")            # far above
    assert A.readme_status(float("nan"), want, tol) == ("FAIL", None)                # as abs(nan) <= tol: FAIL
    assert A.readme_status(0.75, want) == ("PASS", None)                             # default tol = README_TOL


def test_readme_status_s1_cases(A):
    # the three S1 README FAILs as reported (README top-1 from the registry; 10k counts give the reported deltas)
    for readme, k in ((93.72, 9408), (90.13, 9065), (92.98, 9330)):                  # +0.0036, +0.0052, +0.0032
        acc, want = k / 10000, readme / 100.0
        assert not abs(acc - want) <= 0.003                                          # B4 (frozen): FAIL
        assert A.readme_status(acc, want, 0.003) == ("PASS", "README-EXCEEDED")      # B4a: PASS + INFO
        b = A.readme_block(acc, want, 0.003)
        assert (b["status"], b["status_b4"], b["info"]) == ("PASS", "FAIL", "README-EXCEEDED")
    assert A.readme_status(0.8977, 0.9013, 0.003) == ("FAIL", None)                  # -0.0036: still FAIL
    assert A.readme_status(0.9012, 0.9013, 0.003) == ("PASS", None)                  # -0.0001: within


def test_lower_side_is_the_frozen_rule_on_every_10k_count(A, reg):
    # acc = correct / 10000 (b4_weights.test_accuracy); want = readme_top1 / 100 (b4_weights.gate_unit)
    tol = A.BW.README_TOL
    for m in reg["models"]:
        if m["source"] != "hub":
            continue
        want = float(m["readme_top1"]) / 100.0
        c = int(round(want * 10000))
        for k in range(c - 80, c + 81):
            acc = k / 10000
            frozen_fail = not abs(acc - want) <= tol
            st, info = A.readme_status(acc, want, tol)
            assert (st == "FAIL") == (frozen_fail and acc - want < 0), (m["id"], k)
            assert (info == "README-EXCEEDED") == (frozen_fail and acc - want > 0), (m["id"], k)
            assert A.readme_block(acc, want, tol)["status_b4"] == ("FAIL" if frozen_fail else "PASS")


def test_readme_block_keeps_the_frozen_keys(A):
    b = A.readme_block(0.9408, 0.9372, 0.003)
    assert {"acc_10k", "readme_top1", "delta", "tol", "status"} <= set(b)
    assert b["acc_10k"] == 0.9408 and b["readme_top1"] == 0.9372 and b["delta"] == 0.9408 - 0.9372 and b["tol"] == 0.003
    assert b["rule"].startswith("B4a one-sided")


# ---------------------------------------------------------------------------------------------------------------------
# preconditions and refusals (no torch)
# ---------------------------------------------------------------------------------------------------------------------
def _s1(delta=0.0036, status="FAIL", readme="FAIL", **kw):
    r = {"status": status, "sha256": "ab" * 32, "tag_prefix_ok": True, "bytes": 9024, "bytes_expected": 9024,
         "readme": {"acc_10k": 0.9408, "readme_top1": 0.9372, "delta": delta, "tol": 0.003, "status": readme}}
    r.update(kw)
    return r


def test_s1_precondition(A):
    assert A.s1_readme_only_above(_s1()) is None                                     # the B4a case
    assert A.s1_readme_only_above(None) is not None
    assert A.s1_readme_only_above(_s1(delta=-0.0036)) is not None                    # a FAIL below: never amended
    assert A.s1_readme_only_above(_s1(delta=0.003)) is not None                      # not above the tolerance
    assert A.s1_readme_only_above(_s1(status="PASS", readme="PASS", delta=0.001)) is not None
    assert A.s1_readme_only_above({"status": "FAIL", "why": "byte size"}) is not None     # failed before the README
    assert A.s1_readme_only_above(_s1(why="x")) is not None
    assert A.s1_readme_only_above(_s1(sha256=None)) is not None
    assert A.s1_readme_only_above(_s1(tag_prefix_ok=False)) is not None              # the 8-hex tag check failed too
    no_tag = _s1()
    del no_tag["tag_prefix_ok"]
    assert A.s1_readme_only_above(no_tag) is not None                                # the tag check not recorded
    assert A.s1_readme_only_above(_s1(bytes_expected=9025)) is not None              # byte size differs
    no_bytes = _s1()
    del no_bytes["bytes"]
    assert A.s1_readme_only_above(no_bytes) is not None                              # byte size not recorded


def test_heads_dir(A):
    assert A.heads_dir(os.path.join("results", "b4_weights", "weights_r2.json")) == \
        os.path.join(os.path.abspath(os.path.join("results", "b4_weights")), "heads_r2")
    assert A.heads_dir("weights_r2_r3.json").endswith("heads_r2_r3")
    assert A.heads_dir("weights.json") is None and A.heads_dir("weights_p2.json") is None
    assert A.heads_dir("weights_trained_r2.json") is None


def test_counts(A):
    assert A.counts({"a": {"status": "PASS"}, "b": {"status": "FAIL"}, "c": {"status": "PASS"}}) == \
        {"PASS": 2, "NOT_EVALUABLE": 0, "FAIL": 1, "ABSENT": 0}


def _write(p, obj):
    p.write_text(json.dumps(obj))
    return str(p)


def test_refusals_before_torch(A, tmp_path):
    s1 = _write(tmp_path / "weights.json", {"units": {u: _s1() for u in A.UNITS}})
    out = str(tmp_path / "weights_r2.json")
    base = ["--registry", REGISTRY, "--volume", str(tmp_path), "--s1-record", s1]
    run = lambda *extra: A.main(base + list(extra))                                  # noqa: E731
    assert run("--out", out, "--units", "resnet20_hub") == 2                         # another unit, no --allow-other
    assert run("--out", out, "--units", "resnet20_s1", "--allow-other") == 2         # a file unit has no README gate
    assert run("--out", out, "--units", "no_such_unit") == 2
    assert run("--out", out, "--units", ",") == 2                                    # empty
    assert run("--out", str(tmp_path / "weights.json")) == 2                         # exists (and no relaunch tag)
    assert run("--out", str(tmp_path / "weights_x.json")) == 2                       # no relaunch tag -> no heads dir
    assert run("--out", out, "--heads", str(tmp_path / "heads")) == 2                # the S1 heads dir
    bad = _write(tmp_path / "w_bad.json", {"units": {**{u: _s1() for u in A.UNITS},
                                                     "shufflenetv2_x1_0": _s1(delta=-0.0036)}})
    assert A.main(["--registry", REGISTRY, "--volume", str(tmp_path), "--s1-record", bad, "--out", out]) == 2
    assert A.main(["--registry", REGISTRY, "--volume", str(tmp_path), "--s1-record", str(tmp_path / "none.json"),
                   "--out", out]) == 2                                               # S1 record missing
    (tmp_path / "heads_r2" / "shufflenetv2_x0_5").mkdir(parents=True)
    assert run("--out", out) == 2                                                    # a head export exists
    assert not os.path.exists(out)


def test_extractor_accepts_the_amended_record(A, tmp_path):
    ex = _load("b4_extract")
    s1 = _write(tmp_path / "weights.json", {"units": {u: _s1() for u in A.UNITS}})
    for u in A.UNITS:                                                                # the S1 record: refused
        with pytest.raises(SystemExit):
            ex.weights_gate({"id": u, "source": "hub"}, s1)
    units = {}
    for u, (acc, want) in zip(A.UNITS, ((0.9408, 0.9372), (0.9065, 0.9013), (0.9330, 0.9298))):
        rec = {"status": "PASS", "sha256": "ab" * 32, "tag_prefix_ok": True}         # load_model's record (tag ok)
        rec["readme"] = A.readme_block(acc, want)                                     # as gate_unit sets it
        if rec["readme"]["status"] == "FAIL":
            rec["status"] = "FAIL"
        rec["amendment"], rec["info"] = A.AMENDMENT, [rec["readme"]["info"]]
        units[u] = rec
    p = _write(tmp_path / "weights_r2.json", {"schema": A.BW.SCHEMA, "amendment": "B4a", "units": units,
                                              "counts": A.counts(units)})
    for u in A.UNITS:
        g = ex.weights_gate({"id": u, "source": "hub"}, p)
        assert g["status"] == "PASS" and g["amendment"] == "B4a" and g["info"] == ["README-EXCEEDED"]
    with pytest.raises(SystemExit):                                                  # the record covers these 3 only
        ex.weights_gate({"id": "resnet44", "source": "hub"}, p)


# ---------------------------------------------------------------------------------------------------------------------
# structure: imports, ASCII, the marked change only
# ---------------------------------------------------------------------------------------------------------------------
def test_module_level_imports_are_stdlib_and_b4_weights_only():
    src = open(AMEND, encoding="utf-8").read()
    assert all(ord(ch) < 128 for ch in src)
    tree = ast.parse(src)
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            top.add((node.module or "").split(".")[0] if node.level == 0 else ".")
    assert top <= {"argparse", "json", "math", "os", "re", "sys", "time", "numpy", "b4_weights"}, top
    assert "b4_weights" in top and "torch" not in top
    lazy = set()
    for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef)]:
        for node in ast.walk(fn):
            if isinstance(node, ast.Import):
                lazy |= {a.name.split(".")[0] for a in node.names}
    assert "torch" in lazy                                                           # torch only inside main
    assert "B4a CHANGE" in src


def test_import_does_not_import_torch():
    code = ("import sys; sys.path[:0] = [sys.argv[1], sys.argv[1] + '/scripts']; import b4_weights_amend; "
            "bad = [m for m in ('torch', 'torchvision', 'scipy', 'sklearn') if m in sys.modules]; print(bad); "
            "sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code, ROOT], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def _func_lines(path, name):
    """The lines of a top-level function: from `def name(` to the next non-blank line at column 0."""
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    i = next(k for k, s in enumerate(lines) if s.startswith(f"def {name}("))
    j = next((k for k in range(i + 1, len(lines)) if lines[k] and not lines[k][0].isspace()), len(lines))
    body = lines[i:j]
    while body and not body[-1].strip():
        body.pop()
    return body


def test_gate_unit_is_the_frozen_one_outside_the_marked_change():
    frozen = _func_lines(FROZEN, "gate_unit")
    k = next(i for i, s in enumerate(frozen) if s.strip().startswith('rec["readme"] = {'))
    assert frozen[k + 1].strip() == '"status": "PASS" if abs(acc - want) <= README_TOL else "FAIL"}'   # the B4 rule
    frozen_rest = frozen[:k] + frozen[k + 2:]
    amended, skip, changed = [], False, []
    for s in _func_lines(AMEND, "gate_unit"):
        t = s.strip()
        if t.startswith("# B4a CHANGE begin"):
            skip = True
            continue
        if t.startswith("# B4a CHANGE end"):
            skip = False
            continue
        if skip:
            changed.append(t)
            continue
        amended.append(s.replace("BW.", ""))
    assert changed == ['rec["readme"] = readme_block(acc, want, BW.README_TOL)']
    assert amended == frozen_rest
