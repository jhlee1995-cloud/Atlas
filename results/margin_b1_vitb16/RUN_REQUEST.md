# RUN_REQUEST: B1 (the ViT margin test), in the joint session with A4b

The launch sequence is the one in `results/atlas_v1_resnet56_s1/RUN_REQUEST.md` (integration D21):
`setsid bash /workspace/Atlas/pod_atlas.sh /workspace --a4b --b1`. This file lists what B1 adds to it. Design and
pre-registration: `docs/plans/B1_VIT_MARGIN.md`, `experiments/queue/margin_b1_vitb16.yaml` notes.

## Before the launch (Windows)

1. Locally, besides A4b's checks: `node --check scripts/b1_verdicts.js`, `node tests/fixtures/b1/check_fixtures.js`
   (21 scenarios: every decision branch, the D14 band, ADDS-OVER-MAXPROB-ONLY, SPLIT-UNTESTED / SPLIT-FRAGILE, `_r2`,
   `pod_record_agrees`, the D13 joint reading; all PASS), `bash -n pod_atlas.sh`, the ASCII check of every
   `experiments/queue/margin_b1_*.yaml`, and `git diff --stat` empty for `atlas/extract_acts.py atlas/run.py
   atlas/config.py atlas/context.py atlas/invariants/_util.py atlas/invariants/landmarks.py atlas/invariants/adjacency.py
   scripts/check_data.py scripts/check_rebuild.py` (B1's only instrument edit is `atlas/invariants/margin.py`).
2. B1's part of commit P:
   - `docs/plans/B1_VIT_MARGIN.md`;
   - `experiments/queue/margin_b1_{resnet50_legacy10k,resnet50,resnet50_swap,vitb16,vitb16_swap,deitb,deitb_swap}.yaml`
     (the pre-registration is the `margin_b1_vitb16` notes) and the four E9 manifests
     `margin_b1_{resnet20_s0hub_st3,resnet56_s0hub_st3,resnet56_s1,resnet56_s2}.yaml`;
   - `atlas/invariants/margin.py` (opt-in `b1` / `legacy_imagenet` keys, `GRAM_MIN`), `atlas/extract_imagenet.py`,
     `atlas/run_imagenet.py`;
   - `scripts/b1_data.py`, `scripts/b1_gate.py`, `scripts/b1_verdicts.js`;
   - `tests/test_b1_imagenet.py`, the B1 tests in `tests/test_atlas_smoke.py`, `tests/fixtures/b1/`;
   - the four pins in `requirements.txt`; `block_b1` in `pod_atlas.sh`; ATLAS_STATUS row 10 (⬜); this file.
3. Owner decisions (recorded): the ImageNet terms of access of the mirror and the public repo are approved; committed B1
   results carry validation row indices and statistics only (dumps with activations, predictions and logits stay on the
   volume). The volume question is decided by the pre-launch `du` (≤ 28 GB).

## What B1 needs on the pod

- Everything A4b needs, and in particular the Stage 1 dumps `results/atlas_v1_resnet{20,56}_s0hub/dump` (the margin
  preflight, D6; hard) — already in A4b's `ls` line.
- The E9 runs read A4b's same-session dumps `results/atlas_v1_{resnet20_s0hub_st3,resnet56_s0hub_st3,resnet56_s1,
  resnet56_s2}/dump` (soft: a missing one is recorded as NOT_EVALUABLE, never blocks B1).
- ImageNet val (703 MB) and `real.json` are provisioned by the script (`scripts/b1_data.py provision`, pinned revision
  and sha256; `--data-only --b1` does it without a GPU); the weights (0.77 GB) are downloaded by the self-tests into
  `TORCH_HOME` / `HF_HOME` on the volume.
- About 17.4 GB of new volume for B1 (float32 dumps, float16 logits, weights); `block_b1` refuses when the volume in use
  plus what is still to be written exceeds 48 GB.
- `/dev/shm` sets the DataLoader workers (`df -h /dev/shm` is in the launch sequence; a small one gives 0 workers,
  about +2.5 min per run).

## What `block_b1` does (after `block_a4b`; `pod_atlas.sh`)

refusals (existing check dir, `ATLAS_REBUILD=1`, a running training) → B1 imports → `b1_data.py verify` (hard) → volume
check → `pytest tests/test_b1_imagenet.py` (hard) → ViT self-tests on random inputs (hard, `timeout 1200` each; no ImageNet image read) →
E9 rebuilds + exact A3-key identity (soft) → `margin_b1_resnet50_legacy10k`, `margin_b1_resnet50`,
`margin_b1_resnet50_swap` (hard, `timeout 1800` each) → `scripts/b1_gate.py` → only if the gate is open: the four ViT
runs (soft each) and `critic_b1_vit_pair`, `critic_b1_{vitb16,deitb}_swap` → `critic_b1_resnet50_swap`.
Estimate: 21-40 min (integration §3).

## Relaunch and touch rules (D18)

No `git` operation on the pod while `pod_atlas.sh` runs; `ATLAS_REBUILD` unset. A relaunch sets
`ATLAS_B1_CHECK_DIR=results/instrument_check_b1_r2` (and A4b's `ATLAS_A4B_CHECK_DIR`); built B1 atlases are skipped and
never rebuilt; the gate record `results/b1_gate/gate.json` is reused while both resnet50 atlases are unchanged. A ViT with
a complete dump (meta.json) but no atlas.json gets Stage B only; an interrupted Stage A is extracted again once (log it in
SESSION.md). A post-gate P0 failure of a ViT may be re-run once as `margin_b1_<m>_r2` under a committed amendment
(plumbing faults only; an accuracy-band failure is final); `block_b1` runs an `_r2` manifest when it exists.

## After the pull (Windows)

1. Commit the pulled results first (append-only). Stop the pod: billing continues until then.
2. A4b's evaluation first (`results/atlas_v1_resnet56_s1/a4b_eval.json`), then
   `node scripts/b1_verdicts.js --p <P> --p-run <P_run> --a4b results/atlas_v1_resnet56_s1/a4b_eval.json --json results/margin_b1_vitb16/verdicts.json`.
3. Write `results/margin_b1_vitb16/SESSION.md`: P_run, the preflight record, the gate record and its agreement, c*, every
   V item per run with the split states, the outcome and the two margin_vs labels, V4, AC, the joint reading (D13), E1-E11,
   B1-acc and E9, each prediction as `P<n>: predicted X, observed Y, verdict`. Then ATLAS_STATUS row 10 (and the header
   sentence "B1 (row 10) ended <label>"). A4b and B1 are evaluated independently; neither decision waits for the other.
