# S0: batch-4 smoke test (D8 step 1)

The first execution of the batch-4 Python code. It ran on 2026-09-24 on a RunPod CPU pod: 2 vCPU, template
`runpod-torch-v280`, no network volume. Versions were Python 3.12.3, numpy 2.1.2, scipy 1.18.1, scikit-learn 1.9.1 and
torch 2.8.0+cu128, all installed from `requirements.txt`.

The tested tree was the uncommitted P1 candidate on top of 2b561a9. `tree_sha256.txt` lists the sha256 of each of its
40 changed or new files. P1 commits those same bytes with one exception: after S0, section 6 of
`docs/plans/B4_INTEGRATION.md` was rewritten to record this outcome. No code or test changed.

## Final run (all green)

| check | result |
|---|---|
| pytest, 9 batch-4 files (`pytest_b4.xml`) | 137 passed, 1 skipped (`test_eval_js_reads_the_record`: no node on the pod; the Windows evaluation reads the same fields) |
| `b4_extract.py --selftest` | PASS |
| `t1_scoreboard.py --selftest` | PASS (23 checks, fit and eval layouts) |
| `t1_streams.py --selftest` | PASS (19 checks) |
| `collapse_probe.py --selftest` | PASS (27 checks) |
| `t3s_spatial_probe.py --selftest` | PASS (23 checks) |
| `b4_reg.py validate` | registry OK |
| `b4_reg.py jobs` per phase | discovery 53, replay 6, reprobe 53, confirmation 66 jobs |
| `bash -n pod_atlas.sh scripts/pod_b4.sh` | ok |

`s0.log` is the console of the final run.

## What the earlier S0 runs found and how it was fixed

The first two runs failed. All three causes were in the synthetic self-test data or in the handling of a tiny split.
No instrument decision on real data changed.

1. **`b4_extract.py` head-path gate.** A 40-row synthetic maps split had one exact near-tie flip (top-2 gap 8.5e-6),
   which gave argmax agreement 0.975, below the 0.99 bar.
   - The gate now allows max(1, floor(0.01 n)) near-tie flips.
   - For n ≥ 100 this is identical to "agreement ≥ 0.99". Every real split has at least 1000 rows.
2. **`collapse_probe.py` synthetic test labels.** The labels were `arange % K`. The even/odd split halves (D4) then
   held 5 classes each, and the cross-validated spectral gap came out at 1.245 against the closed form 2.78.
   - The labels are now shuffled.
   - Real CIFAR test rows are not class-periodic.
3. **`t1_scoreboard.py` synthetic stem.** The first stem was e15 + N(0, 0.3²) in 16 dimensions, so its noise norm (1.2)
   exceeded its mean (1). An L2-normalised kNN could not see a 1.5 shift along the all-ones direction: X4 AUROC was
   0.53 and TPR 0.02.
   - The stem is now 3·e15 + N(0, 0.3²).
   - The noise-family shift is now 3.0 along the all-ones direction orthogonal to e15.
   - X4 AUROC on the planted shift is now 0.999, and the clean false-alarm rate at α 0.05 is 0.047.
