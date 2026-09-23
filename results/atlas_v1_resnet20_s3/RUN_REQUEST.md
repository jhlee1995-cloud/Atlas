# RUN_REQUEST: one pod session for Stage 1b (seeds 3, 4), Stage 2 (A4, resnet56) and A3 (margin / type-b)

This is the single command for all three experiments; `results/atlas_v1_resnet56_s0hub/RUN_REQUEST.md` and
`results/margin_v1_resnet20_s1/RUN_REQUEST.md` point here. Push the one commit that holds every edit, manifest and
prediction first (it is the pre-registration commit P of all three):
- Stage 1b: `experiments/queue/atlas_v1_resnet20_s{3,4}.yaml` (predictions in the s3 notes), `docs/plans/STAGE1.md`
  amendment 2, `scripts/check_rebuild.py`, `scripts/d1_distance_only.js`;
- Stage 2: `experiments/queue/atlas_v1_resnet56_*.yaml`, `atlas_v1_resnet20_*_st2.yaml` (predictions R56-* in the
  `atlas_v1_resnet56_s0hub.yaml` notes), `docs/plans/STAGE2.md`;
- A3: `atlas/invariants/margin.py`, `experiments/queue/margin_*.yaml` (predictions M1-M5, N, E1-E5 and the row-9
  table in the `margin_v1_resnet20_s1.yaml` notes), `docs/plans/A3_MARGIN.md`.

Pod: **RTX 4090 required** (s1 and s2 were trained on an NVIDIA GeForce RTX 4090 with torch 2.8.0+cu128,
`results/train_resnet20_s1/train.json`; the Stage 1b block refuses any other GPU or torch), EU-RO-1, network volume
kxfir1tryb at /workspace, template runpod-torch-v280. The volume must hold the Stage 0/1 dumps
(`results/atlas_v0_resnet20_cifar10/dump`, `results/atlas_v1_resnet20_{s0hub,s1,s2,s1_ref1,rand}/dump`) and
`/workspace/models/resnet20_{s1,s2}_chenyaofo.pt`. Never add `--stage1` (it is committed; the script refuses it).

```bash
git -C /workspace/Atlas fetch && git -C /workspace/Atlas reset --hard origin/main   # dumps are untracked; kept
cd /workspace/Atlas && PIP_BREAK_SYSTEM_PACKAGES=1 python -m pip install -q -r requirements.txt \
  && python -m pytest -q tests/test_atlas_smoke.py          # a broken test costs one fix-push here, not the session
mkdir -p /workspace/logs && setsid bash /workspace/Atlas/pod_atlas.sh /workspace --stage1b --stage2 --a3 < /dev/null > /workspace/logs/launch.log 2>&1 &
tail -F /workspace/logs/launch.log
grep -h "epoch   10" /workspace/logs/train_resnet20_s[34].log   # x20 = projected Stage 1b training time
```

Blocks run in this order, each isolated (a failing block is logged, the next still runs; the script exits 1 at
the end if a block or a soft step failed, listed in `/workspace/logs/soft_failures_<timestamp>.log`):

| block | what | estimate |
|---|---|---|
| prelude | install, data check, GPU preflight, smoke tests, Stage 0 (skipped: committed) | ~3 min |
| `--stage1b` | I0 (Stage A inputs unchanged since 6242a17, torch/GPU = Stage 1, s3/s4 manifests agree), I1 rebuild of s1 (~1.5 min) and compare/critic replay (~1 min), 1-epoch preflight, s3 + s4 trained concurrently (~8 min), 2 atlases (~3 min), 7 compares + 3 critics (~2-3 min), I2 | ~16-18 min |
| `--stage2` | resnet56 norm check, ladder e40/e20/e10 + null one at a time (~3-6 min), 9 atlases (3 resnet20 `_st2` re-runs, 5 resnet56 at stride 5, 1 at stride 1; ~15 min), 15 compares + 8 critics (~3-4 min) | ~21-25 min |
| `--a3` | margin_typeb rebuilds of v0, v1 hub, rand, resnet56 hub; M1 gate on v0; if it passes, s1, s2, s1_ref1, s3, s4 and 3-4 critics | ~3-5 min |
| **total** | trainings never overlap (s3/s4 concurrently first, then the resnet56 rungs one by one) | **~45-50 min, about $0.55-0.65 at $0.74/h** |

Early stops by design:
- Stage 1b stops before anything about seeds 3 and 4 exists if an I0 or I1 check fails (read
  `results/instrument_check_stage1b/{check,replay}.json`); Stage 2 and A3 still run, and A3 skips the s3/s4 rebuilds.
- A3 runs the confirmation-seed rebuilds only if the M1 margin clause passes on v0 (`[A3] M1 ...` line in the log).
  A missing v0/v1-hub/rand/s1/s2/s1_ref1 dump is an error (soft failure), and without a v0 margin atlas the gate
  prints `[A3] M1 NOT RUN` and fails the block: record M1 as unevaluated, not FAIL.

Relaunch (append-only): a relaunch skips atlases, compares and critics whose outputs exist. The Stage 1b block
refuses an existing `results/instrument_check_stage1b`; relaunch it with
`ATLAS_STAGE1B_CHECK_DIR=results/instrument_check_stage1b_r2` so the instrument is re-verified into a new directory.

Then pull results/ without dumps (the command is printed at the end), commit on Windows, and evaluate in this
order: `results/atlas_v1_resnet20_s3/SESSION.md` (I/R0/S/T2/D1/C/F verdicts and the Stage 1b decision,
`docs/plans/STAGE1.md` amendment 2 item 9; D1 via `node scripts/d1_distance_only.js`), then
`results/atlas_v1_resnet56_s0hub/SESSION.md` (R56-*, `docs/plans/STAGE2.md`; only after the Stage 1b decision is
committed), and `results/margin_v1_resnet20_s1/SESSION.md` (M1-M5, N, E, row 9).
