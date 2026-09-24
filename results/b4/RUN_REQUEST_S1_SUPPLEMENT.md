# RUN_REQUEST: batch 4, S1 supplement (B4a, same RTX 4090 pod)

This supplement runs the B4a amendment (`docs/plans/B4A_AMENDMENT.md`, approved by the owner on 2026-09-24) on the S1 pod, after `block_b4s1` has ended. It does four things:
- runs the weights gate with the one-sided README rule;
- extracts exactly the three units the S1 README gate refused: mobilenetv2_x0_75 and shufflenetv2_x1_0 (C, sealed fit) and shufflenetv2_x0_5 (Dnew, open fit);
- runs the S1 discovery probes of shufflenetv2_x0_5;
- writes a complete seal manifest and the timing.

**Commits and names:**
- **P_A** is the commit that adds this file and the amendment.
- **P1** = `ad97d0a659dd9f6f60b24e0b8cc1681f1bd8a166`.
- **Tag and check dir:** tag `_r2`, check dir `results/instrument_check_b4s1_r2`.

**Nothing else runs on the pod.** Never `git pull` or `reset` while `pod_atlas.sh` or the supplement runs.

**Keep the pod running from the end of `block_b4s1` until the supplement ends.** Never stop, restart or delete it in between.
- A stop wipes the container disk. That removes the pip-installed scipy, sklearn and pytest, and `/root/b4_factor_cache` (a cache, rebuilt if lost).
- `RUN_REQUEST_S1.md`'s "Delete the pod" step therefore waits until after the supplement (step 3 of "After the run" below).
- If the pod was restarted anyway, the supplement's stack guard refuses before it writes anything. Follow "Recovery after a pod restart" below.

## 1. S1 has ended

Use its own ssh call:

```bash
pgrep -f "[b]ash /workspace/Atlas/pod_atlas.sh" || echo S1-ENDED
tail -n 40 /workspace/logs/b4s1.log            # "=== block block_b4s1 OK" (FAILED: read the reason first)
ls -l /workspace/Atlas/results/instrument_check_b4s1/sealed.json /workspace/Atlas/results/b4_weights/weights.json
```

If `sealed.json` is missing, `block_b4s1` stopped before GPU-D ended. Do not run the supplement; report instead (the supplement refuses anyway).

## 2. Code to P_A (only now)

```bash
git -C /workspace/Atlas fetch origin && git -C /workspace/Atlas reset --hard <P_A>     # dumps and results/ untouched
cd /workspace/Atlas && git log -1 --format='%H %s'                                      # must be P_A
git -C /workspace/Atlas diff --stat ad97d0a659dd9f6f60b24e0b8cc1681f1bd8a166 HEAD        # only the B4a files
```

The last command must list only these files:
- `docs/plans/B4A_AMENDMENT.md`
- `scripts/b4_weights_amend.py`
- `scripts/b4a_supplement.sh`
- `scripts/b4a_p2.js`
- `tests/test_b4a_amendment.py`
- `results/b4/RUN_REQUEST_S1_SUPPLEMENT.md`

## 3. Volume, then launch (detached)

Projected usage when S1 ends is about 45.8 GB. The supplement's group estimate is 2 GB, so the result would sit at the
48 GB ceiling of `pod_b4.sh`. The owner authorised enlarging the network volume (2026-09-24). From Windows, after S1
has ended:

```bash
runpodctl network-volume update kxfir1tryb --size 75
```

Then launch with `ATLAS_B4A_QUOTA_GB=73`, the new size minus 2 GB. `b4a_supplement.sh` overrides only the ceiling of
`b4_quota`. Without the variable the ceiling stays at 48 GB.

```bash
export ATLAS_B4_P1=ad97d0a659dd9f6f60b24e0b8cc1681f1bd8a166 ATLAS_B4A_QUOTA_GB=73
setsid -f bash /workspace/Atlas/scripts/b4a_supplement.sh /workspace < /dev/null > /workspace/logs/b4a.log 2>&1
```

`ATLAS_B4_TAG` defaults to `_r2`. Do not set `ATLAS_B4_S1_TAG` (S1 ran under `''`).

Launch in its own ssh call, with no `pgrep` or `pod_atlas.sh` in the same command line. The supplement's guard runs `pgrep -f pod_atlas.sh`, and it would match the ssh wrapper shell's command line.

### Recovery after a pod restart

Use this only if the guard refused with "not importable" or "the software stack differs from S1's". Reinstall S1's exact scipy and scikit-learn, and pytest, in its own ssh call:

```bash
cd /workspace/Atlas && V=results/instrument_check_b4s1/versions.json
python -m pip install -q "scipy==$(python -c "import json; print(json.load(open('$V'))['scipy'])")" \
  "scikit-learn==$(python -c "import json; print(json.load(open('$V'))['sklearn'])")" "pytest>=7"
python -c "import numpy, scipy, sklearn, torch, torchvision; print(numpy.__version__, scipy.__version__, sklearn.__version__, torch.__version__, torchvision.__version__)"
cat "$V"
```

- The printed versions must equal `versions.json` (python, numpy, torch and torchvision come with the image).
- If they do, launch again with the same tag: the refused guard wrote nothing.
- If python, numpy, torch or torchvision differ (another image), stop and report.

## 4. Poll

Use its own ssh call. The brackets keep pgrep from matching itself:

```bash
pgrep -f "[b]4a_supplement.sh" || echo B4A-ENDED
tail -n 30 /workspace/logs/b4a.log
```

**Expected duration:** about 20 minutes, about $0.25 at $0.74/h.
- The GPU part takes about 10 minutes.
- The two discovery probes of shufflenetv2_x0_5 are the variable part. Their bound is the D10 limits: `t1_wide` 7200 s and `t2_wide` 3600 s, run in parallel.

## What the supplement does

`scripts/b4a_supplement.sh` runs these steps in order:

1. Guards (hard), including the software stack equal to S1's, before anything is written.
2. The versions record in the check dir (hard).
3. `pytest tests/test_b4a_amendment.py` (hard).
4. `--verify-seals` on S1's manifest (hard).
5. `b4_weights_amend.py` -> `results/b4_weights/weights_r2.json`.
6. The quota guard.
7. The three fit extractions (soft per unit).
8. The discovery probes of shufflenetv2_x0_5 (T2 and T1 fit, tag `_r2`).
9. `--seal-manifest` into the `_r2` check dir, `seal_diff.json` against S1's manifest (hard), then `--verify-seals`.
10. `timing.json` and `supplement.json`.

The log ends in one of three ways:
- **`=== B4a supplement STOPPED at a hard step`:** a guard or hard step refused.
- **a soft-failure list with exit 1:** a unit failed.
- **exit 0.**

## After the run

1. **Pull without dumps.** This is R1: S1 and the supplement together.

   ```bash
   ssh ... 'tar -C /workspace/Atlas -czf - --exclude=dump --exclude="dump_step*" results' | tar -xzf - -C .
   ```

   Besides S1's files, this brings:
   - `results/instrument_check_b4s1_r2/` (sealed.json, seal_diff.json, verify_seals*.json, versions*.json, timing.json, supplement.json, pytest_b4a.xml, jobs_discovery.txt, head.txt);
   - `results/b4_weights/weights_r2.json` and `results/b4_weights/heads_r2/`;
   - `results/b4_extract/*_fit_r2.json`;
   - `results/b4_t2/shufflenetv2_x0_5_r2/` and `results/b4_t1/shufflenetv2_x0_5_fit_r2/`.
2. **Copy the checkpoints** `/workspace/models/*.pt` to a Windows folder outside the repo (D18). This step is unchanged from S1.
3. **Delete the pod** (only now, after the supplement) and confirm with `runpodctl pod list`.
4. **Report.**
   - Sealed units report PASS/FAIL only, never an accuracy. `supplement.json` holds none.
   - Name the `README-EXCEEDED` flags and the `seal_diff.json` status.
   - Give the discovery paths from `supplement.json` (any `_r<k>` tag).

## If something fails

- **A guard refuses.** Nothing was written except the logs. Fix the cause and launch again with the same tag (`_r2` on the first launch), because the check dir was not created.
  - A stack refusal ("not importable", "differs from S1's") means the pod was restarted: see "Recovery after a pod restart".
- **A hard step fails after the check dir exists** (versions, pytest, the S1 seal check, the weights script, the quota, or the seal manifest and diff). Stop and report.
  - A failed `seal_diff.json` means an S1 sealed dump changed, or an unexpected seal appeared. No S2 runs until the owner decides.
- **A unit fails softly** (an extraction, or the discovery lane: no job list, no job line for the unit, or a missing output). Relaunch with the **next unused** tag, plus `ATLAS_B4_MOVE_PARTIAL=1` if a partial dump is left; it is renamed aside, never deleted.
  - The next unused tag is the smallest `_r<k>` with no `results/instrument_check_b4s1_r<k>` and no `results/b4_weights/weights_r<k>.json`. Relaunches of S1 take tags from the same sequence. Normally it is `_r3`.
  - Complete B4a dumps are skipped.
  - Discovery jobs already done under an earlier `_r<k>` are skipped. `supplement.json` still lists their outputs, with the tag that holds them.
  - The new check dir gets a complete manifest and a `timing.json` again. S2 and P2 then use that newest check dir.

## Notes for P2, S2 and step E (from `docs/plans/B4A_AMENDMENT.md`)

Below, `<I>` is the newest check dir with a complete `sealed.json`: `results/instrument_check_b4s1_r2`, unless a relaunch wrote a newer one.

- **At P2.**
  - The ShuffleNet runs were timed with two jobs on an otherwise idle pod, and every other family in the loaded S1 lane. So floor them first, then compute the limits:

    ```bash
    node scripts/b4a_p2.js timing --timing <I>/timing.json --out experiments/b4/timing_b4a.json
    node scripts/b4_timeouts.js --registry experiments/b4/models.json --timing experiments/b4/timing_b4a.json --out experiments/b4/timeouts.json
    ```

  - `freeze_P2.json` records B4a under its own key (`"b4a": {p_a, doc, check_dir, tag, s2_check_dir, timing}`), not under `amendments`.
- **S2.** First check, in its own ssh call, that the manifest lists both B4a dumps SEALED. The Python check is in `docs/plans/B4A_AMENDMENT.md`, "How S2 must be launched"; do not launch if it fails. Then launch:

  ```bash
  ATLAS_B4_P1=<P1> ATLAS_B4_P2=<P2> ATLAS_B4S1_CHECK_DIR=<I> \
    bash /workspace/Atlas/pod_atlas.sh /workspace --b4s2 --b4cpu
  ```

  `b4_guard_p2` then verifies every sealed dump against the complete manifest, S1's plus the two B4a dumps. Leave `ATLAS_B4_S1_TAG` unset. Without `ATLAS_B4S1_CHECK_DIR`, the frozen default falls back to S1's manifest, which skips the two B4a dumps silently.
- **Step E (required).** After R2, before any result of the two B4a C units is reported, run `node scripts/b4a_p2.js seals --json results/b4/b4a_seals.json`. It must PASS.
  - A failing unit is NOT_EVALUABLE for every claim, PRIMARY included, until the check passes.
  - The repair (a later `--verify-seals` against the B4a manifest) is in the amendment doc.
- **Evaluation.** `--p-run` includes P_A:
  - discovery: `--p-run <P1>,<P_A>`;
  - confirmation: `--p-run <P1>,<P_A>,<S2 HEAD>`.
