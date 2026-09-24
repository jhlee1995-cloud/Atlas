# RUN_REQUEST: batch 4, session 1 (S1, RTX 4090)

The GPU session of batch 4 (docs/plans/B4_INTEGRATION.md D7-D11, D17). S1 extracts every dump, trains the eight new
checkpoints and runs the discovery probes. It seals the confirmation dumps, which nobody reads before the P2 freeze.
S2 (numpy only) has its own request, written at P2.

P1 is the commit that adds this file (`atlas: batch 4 pre-registration (P1)`). S1 runs at P1, or at P1 plus changes to
the time limits in `scripts/pod_b4.sh` only (D8). `block_b4s1` refuses any other difference from P1 in
`experiments/b4/frozen_files.txt`.

## Before the launch (Windows, done at P1)

- S0: the whole batch-4 pytest, the five synthetic self-tests, `b4_reg.py validate` and `jobs` for every phase, and
  `bash -n` ran green on a CPU pod without the volume. The record is in `results/instrument_check_b4s0/`.
- Node: `b4_stats.js --selftest`, `b4_timeouts.js --selftest`, `b4_make_registry.js --check`,
  `tests/t1_eval_fixture.js`, `collapse_laws.js --selftest`, `tests/collapse_laws_fixture.js`, `t3s_eval.js --selftest`.
- `git diff --stat 2b561a9 -- <D6 list>` is empty; `pod_atlas.sh` differs from 2b561a9 only by the `--b4s1`, `--b4s2`
  and `--b4cpu` hooks.
- Hub weight tags: the 8-hex tag in each chenyaofo file name is the start of its sha256. This was checked on 2026-09-23
  for resnet20, resnet32, mobilenetv2_x0_5 and shufflenetv2_x0_5. `b4_weights.py` checks all 19 files again on the pod.

## Pod

- RTX 4090 in EU-RO-1, template `runpod-torch-v280` (torch 2.8.0+cu128), network volume `kxfir1tryb` at /workspace,
  container disk 30 GB (the stream frames go to `/root/b4_frames`). Fallback: any Ampere-or-newer GPU in EU-RO-1; record
  it, and the anchor gate decides.
- The volume must hold the existing dumps and checkpoints the anchor gate and the discovery units read
  (`results/atlas_v1_*/dump`, `/workspace/models/*.pt`). S1 adds about 19.8 GB (D11), so the volume ends near 46.5 GB.
  `b4_quota` skips a sealed group that would cross 48 GB, lowest priority first.
- Nothing else runs on the pod. Never `git pull` or `reset` while `pod_atlas.sh` runs.

```bash
git -C /workspace/Atlas fetch && git -C /workspace/Atlas reset --hard origin/main       # dumps are untracked; kept
cd /workspace/Atlas && git log -1 --format='%H %s'                                       # must be P1 (or P1 + limits)
export ATLAS_B4_P1=<P1 sha>
mkdir -p /workspace/logs
setsid -f bash /workspace/Atlas/pod_atlas.sh /workspace --b4s1 < /dev/null > /workspace/logs/b4s1.log 2>&1
```

Poll with `pgrep -f "[b]ash /workspace/Atlas/pod_atlas.sh"` in its own ssh call; the brackets keep pgrep from matching
itself.

Expected duration:

| stage | time |
|---|---|
| setup (pip, data report, pytest, self-tests, weights) | 25-35 min |
| GPU chain: extraction | 55-80 min |
| GPU chain: training pairs | 60-70 min |
| GPU chain: total | 2.0-2.4 h |
| CPU tail | 0.3-1 h |

At $0.74/h that is about $2.0-2.8.

## What S1 does (`block_b4s1`, `scripts/pod_b4.sh`)

1. **Guards.**
   - P1 ≤ HEAD, and the frozen files are equal to P1.
   - The committed instruments are unchanged since 2b561a9.
   - The check directory `results/instrument_check_b4s1` is fresh, and no GPU job is running.
2. **Checks.**
   - Versions and the data report.
   - pytest and the five self-tests (hard).
   - The rebuild check of `atlas_v1_resnet20_s0hub_st3`, recorded, in the background.
   - `b4_weights.py`: sha256, hub byte size, README accuracy gate and head export.
3. **GPU-A:** the two rule-6 anchors (the resnet20 and resnet56 hubs) and the anchor gate (hard).
4. **GPU-B:** the discovery dumps (D, N, Dnew fit), the anchor gate on every old net, and the lane S discovery maps.
   The discovery CPU lane then starts in the background.
5. **GPU-C:** the training pairs: r56 s31/s32, r20 s31/s32, r56 ls10 ×2 and r56 wd5e5 ×2.
6. **GPU-D:** the sealed dumps, in priority order: C12 fit; F/F20 fit and eval; K fit; R2 eval; Sconf maps. Then the
   seal manifest `sealed.json`.
7. **Wrap-up:** wait for the CPU lane, write `timing.json`, and remove the frames.

## After the run (Windows)

1. **Pull** without dumps (R1):
   `ssh ... 'tar -C /workspace/Atlas -czf - --exclude=dump --exclude="dump_step*" results' | tar -xzf - -C .`
   This brings `sealed.json`, `timing.json`, `anchor_gate*.json`, `weights*.json` and the discovery outputs.
2. **Copy the checkpoints** `/workspace/models/*.pt` (about 75 MB) to a Windows folder outside the repo (D18). Their
   sha256 list is in `results/b4_weights/weights.json` and `weights_trained.json`.
3. **Delete the pod** and confirm with `runpodctl pod list`.
4. **Report.**
   - Summarise each part.
   - Anchors that pass or fail are named.
   - Sealed units report PASS/FAIL only, never an accuracy.

## Relaunch rules (D10)

A per-unit timeout is soft: a killed unit writes nothing. Relaunch with unchanged code:

```bash
ATLAS_B4S1_CHECK_DIR=results/instrument_check_b4s1_r2 ATLAS_B4_TAG=_r2
```

`ATLAS_B4_MOVE_PARTIAL=1` renames a partial dump aside; nothing is deleted. If a time limit must change, commit only
the change to `scripts/pod_b4.sh` limits and record it in the relaunch note, as in
`results/margin_b1_vitb16/RELAUNCH_r2.md`.
