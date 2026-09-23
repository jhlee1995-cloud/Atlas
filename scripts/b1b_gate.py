"""
b1b_gate.py -- gate G of B1b (docs/plans/B1B_AMENDMENT.md): scripts/b1_gate.py with ONE change, the G0 anchor of the
legacy valley separation. B1's own record (results/b1_gate/gate.json, CLOSED) is never rewritten; B1b writes
<out>/gate.json once (default results/b1b_gate) and a relaunch reuses it exactly as b1_gate.py does.

  G0 (B1b) lg.valley_sep_legacy in [1.2481, 1.2881] = 1.2681 +- 0.02 (the B1 half-width). 1.2681 is the Upgraded-Mod
     formula (session_experiments/imagenet_extract.py:66-72, min 3 per class) applied to the legacy cache
     <volume>/cache/imagenet/penult_10000.npz, the 10,048-row sample the legacy margin 0.800 came from. B1's band
     [1.12, 1.16] was anchored on 1.14, which is the same formula on the 30,056-row cache penult_30000.npz
     (imagenet_valley_check.py, ~30 per class: 1.1423): a different sample size.
  Everything else (G0's other clauses, G1, G2, G3, thresholds, sources) is b1_gate.py unchanged.

  python scripts/b1b_gate.py --out results/b1b_gate --check-dir results/instrument_check_b1_r2 [--volume /workspace]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b1_gate  # noqa: E402

B1B_LEGACY_SEP = (1.2481, 1.2881)
AMENDMENT = {"doc": "docs/plans/B1B_AMENDMENT.md", "changed": "G0 LEGACY_SEP",
             "b1": list(b1_gate.LEGACY_SEP), "b1b": list(B1B_LEGACY_SEP),
             "anchor": "1.2681 = imagenet_extract.py:66-72 on cache/imagenet/penult_10000.npz (10,048 rows, min 3/class)"}

_evaluate = b1_gate.evaluate


def evaluate(*a, **k):
    rep = _evaluate(*a, **k)
    rep["amendment"] = AMENDMENT
    return rep


if __name__ == "__main__":
    b1_gate.LEGACY_SEP = B1B_LEGACY_SEP      # read at call time by b1_gate.evaluate
    b1_gate.evaluate = evaluate              # b1_gate.main resolves `evaluate` in its module namespace
    b1_gate.main()
