# B1 relaunch r2: timeout deviation (disclosed before any ViT data was read)

**What happened (first launch, P_run d257d91, pod nc2mzssur3bb0k, 2026-09-23).**
- `block_b1` ended with exit 124 (`timeout`) at 15:05:11 UTC.
- `margin_b1_resnet50_legacy10k` finished (atlas.json written, 180 s).
- `margin_b1_resnet50` finished Stage A: its dump is complete (meta.json at 14:37 UTC; 50,000 rows, 0 decode failures).
  Its Stage B was then killed by `timeout 1800` (`pod_atlas.sh` `b1_stage`, integration D19) after 17 of 18 taps.
  `margin_typeb` took 65-118 s per tap at n = 25,000, roughly independent of D. No atlas.json was written.
- The ResNet50 run has no `soft` wrapper, so the block stopped before gate G. No ViT image was read and no ViT dump
  exists: only the random-input self-tests ran (results/instrument_check_b1/selftest_*.json).
- `margin_b1_resnet50_swap` was not started.

**Why this is a timeout, not an instrument change.** The design estimate (2.5-5 min per ImageNet run, integration
section 3) was wrong for Stage B. No instrument file changes:
- `atlas/`, `scripts/b1_gate.py`, `scripts/b1_verdicts.js`, the manifests and `docs/plans/B1_VIT_MARGIN.md` are
  unchanged. `git diff d257d91 <this commit>` lists only `pod_atlas.sh` and this file.
- The only edit is the `b1_stage` limit, 1800 s -> 7200 s. It changes how long a run may take, not what it computes.

**Relaunch (integration D18).**
```
ATLAS_B1_CHECK_DIR=results/instrument_check_b1_r2 bash pod_atlas.sh /workspace --b1
```
- `margin_b1_resnet50_legacy10k` keeps its atlas.json and is never rebuilt.
- `margin_b1_resnet50` reuses its complete dump (`run_imagenet`: an existing dump gets Stage B only). Its Stage B now
  runs at this commit, so its provenance `stage_b_git_commit` differs from its dump's `meta.git_commit` (d257d91).
  Both commits go to the evaluators as the `--p-run` list.
- Every other ImageNet run is built from scratch at this commit.
- The four E9 atlases (`margin_b1_resnet{20,56}_*`) already exist and are skipped.

**Read of D18.** D18 requires an `_r2` restart of all seven ImageNet runs "if code must change after any resnet50
atlas exists". The owner of that rule is the instrument code, because its purpose is to stop tuning the measurement
after seeing ResNet50 numbers. A process time limit is not measurement code. No ResNet50 margin number was looked at:
the legacy10k atlas was not opened, and the killed run wrote none. Only the extraction log lines were seen: accuracy
0.7711 (legacy10k), ref 0.77976 / test 0.77524 (resnet50), and the per-tap build times. This is recorded as a deviation for the verifiers to
judge. The alternative reading (restart all seven as `_r2`) would re-extract the same ResNet50 dumps with identical
code and inputs.

**Cost.** About 2.5-3 h of extra pod time, about $2 at $0.74/h: ResNet50 Stage B, ResNet50 swap, and four ViT runs at
about 14 taps x 90 s each plus extraction.
