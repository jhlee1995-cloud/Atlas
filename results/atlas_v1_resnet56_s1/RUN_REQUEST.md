# RUN_REQUEST: A4b (depth-56 replication, matched rung) and B1 in one pod session

This is the one launch sequence of the joint session (integration D21). B1's own request,
`results/margin_b1_vitb16/RUN_REQUEST.md`, points here and lists the dumps B1 needs.

## Before the launch (Windows)

1. Locally: `node --check scripts/a4b_eval.js`, `node scripts/a4b_eval.js --dry-run` (reproduces the Stage 2
   numbers), `bash scripts/a4b_eval_fixture.sh` (all scenarios PASS), `bash -n pod_atlas.sh`, and the ASCII check of
   every manifest (B1 adds its own checks, D17 item 1).
2. **One commit P** holds every A4b and B1 edit, manifest, prediction and decision script, then push. A4b's part:
   - `docs/plans/STAGE2B.md` and the STAGE2.md "Amendment 1" pointer;
   - `experiments/queue/atlas_v1_resnet20_{s0hub,s1,s2,s3,s4}_st3.yaml`;
   - `experiments/queue/atlas_v1_resnet56_{s0hub_st3,s0hub_ref1,e40_st3,e50,e60,e70,e90,s12m,s13m,s1,s2}.yaml`
     (predictions in the `_s1` notes);
   - `experiments/queue/margin_v1_resnet20_{s0hub,s1,s2,s3,s4}_st3.yaml` and
     `margin_v1_resnet56_{s0hub_st3,s1,s2,e50,e60,e70,e90,s12m,s13m}.yaml` (M56 predictions in the
     `margin_v1_resnet56_s1` notes);
   - `scripts/matched_rung.py`, `scripts/maxprob_ties.py`, `scripts/a4b_eval.js`, `scripts/a4b_eval_fixture.sh`;
   - four tests in `tests/test_atlas_smoke.py` (matched-rung rule, its atomic record, tie mass, manifest freeze);
   - the `--a4b` block, the margin preflight and the `--b1` hooks in `pod_atlas.sh`;
   - ATLAS_STATUS row 11 (⬜; row 10 is B1's) and this file.
3. Pre-launch fixes produce P_run (a descendant of P) and may touch code only. Unchanged from P: every A4b and B1
   manifest, `docs/plans/{STAGE2B,STAGE2,B1_VIT_MARGIN}.md`, `scripts/{a4b_eval.js,b1_verdicts.js,b1_gate.py}`,
   `experiments/tolerances_default.yaml`, `ATLAS_STATUS.md` (G0e checks this after the pull).
4. Owner, before the launch: the ImageNet terms and the public repo are approved (committed results carry row indices
   and statistics only); the volume question is decided by the pre-launch `du` below.

## Pod

**RTX 4090 with torch 2.8.0+cu128 required** (A4b G0d compares with `results/train_resnet56_e40/train.json` and stops
the block before any training otherwise), EU-RO-1, network volume kxfir1tryb at /workspace, template runpod-torch-v280.
The volume must hold:
- the Stage 1 dumps `results/atlas_v1_resnet{20,56}_s0hub/dump` (the margin preflight, D6);
- the st2 dumps `results/atlas_v1_resnet20_{s0hub,s1,s2}_st2/dump` (G0c, hard in `block_a4b`);
- the Stage 1b dumps `results/atlas_v1_resnet20_{s3,s4}/dump` and the Stage 2 dumps
  `results/atlas_v1_resnet56_{s0hub,e40}/dump` (G0c same-space compares; soft);
- `/workspace/models/resnet20_s{1,2,3,4}_chenyaofo.pt` (hard) and `/workspace/models/resnet56_s11_e40_chenyaofo.pt`
  (`atlas_v1_resnet56_e40_st3`; soft).

Never add `--stage1`, `--stage1b`, `--stage2` or `--a3`: they ran and are committed.

```bash
# nothing else runs on this pod; never git pull/reset while pod_atlas.sh runs
git -C /workspace/Atlas fetch && git -C /workspace/Atlas reset --hard origin/main      # dumps are untracked; kept
cd /workspace/Atlas && git log -1 --format='%H %cI'                                     # = P_run; record in both SESSION.md
export PIP_BREAK_SYSTEM_PACKAGES=1                                                      # PEP 668 (pod_atlas.sh:61 sets it only inside the script)
python -m pip install -q -r requirements.txt
python - <<'EOF'
import json, platform, numpy, scipy, sklearn, torch, torchvision, timm, pyarrow, huggingface_hub
v = {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__}
ref = json.load(open("results/instrument_check_stage1b/check.json"))["versions"]
t = json.load(open("results/train_resnet56_e40/train.json"))
print(v, torch.__version__, torchvision.__version__, torch.cuda.get_device_name(0), timm.__version__, pyarrow.__version__, huggingface_hub.__version__)
assert v == ref, ("library versions differ from the committed sessions: stop and ask the owner", v, ref)
assert torch.__version__ == t["torch"] and torch.cuda.get_device_name(0) == t["device"], "A4b G0d would refuse"
EOF
python -m pytest -q tests/                        # smoke (hard in-script at pod_atlas.sh after the install) + tests/test_b1_imagenet.py
bash -n pod_atlas.sh
for x in resnet20 resnet56; do                    # the D6 preflight, into scratch
  python scripts/check_rebuild.py rebuild --dump results/atlas_v1_${x}_s0hub/dump --committed results/margin_v1_${x}_s0hub/atlas.json \
    --manifest experiments/queue/margin_v1_${x}_s0hub.yaml --work /workspace/scratch/prelaunch_$x --out /workspace/scratch/prelaunch_$x
done
ls results/atlas_v1_resnet20_{s0hub_st2,s1_st2,s2_st2,s3,s4,s0hub}/dump/meta.json results/atlas_v1_resnet56_{s0hub,e40}/dump/meta.json \
   /workspace/models/resnet20_s{1,2,3,4}_chenyaofo.pt /workspace/models/resnet56_s11_e40_chenyaofo.pt
du -sb /workspace | awk '{ printf "%.1f GB of 50 used (limit 28)\n", $1 / 1e9; exit ($1 > 28e9) }'
df -h /dev/shm
env | grep -E '^ATLAS_' || true                  # first launch: must print nothing
mkdir -p /workspace/logs && setsid bash /workspace/Atlas/pod_atlas.sh /workspace --a4b --b1 < /dev/null > /workspace/logs/launch.log 2>&1 &
```

Watch:

```bash
tail -F /workspace/logs/launch.log
tail -F /workspace/logs/a4b_train_lane.log                     # the GPU lane (e50, e60, e70, [e90], s12m, s13m, s1, s2)
cat /workspace/Atlas/results/instrument_check_a4b/ladder.json  # the matched-rung decision, once e70 (or e90) is done
```

If the pre-launch `du` shows more than 28 GB, stop and ask the owner (integration §4 question 2). `pod_atlas.sh`
refuses `--b1` at a commit without B1 (`scripts/b1_data.py` and `block_b1`); an A4b-only launch drops `--b1` and the
B1 imports from the check above.

| phase | what | estimate |
|---|---|---|
| pre-launch (interactive) | install, version check, pytest, two margin rebuilds, `du` | ~4-8 min |
| prelude (script) | install, CIFAR data check, B1 data, GPU preflight, smoke tests, margin preflight (D6), Stage 0 (skipped) | ~4.5-7 min |
| `block_a4b` | G0d + code-diff gate; GPU lane (e50/e60/e70 ~10.5 min, s12m + s13m ~6-8 min, s1 + s2 ~23.5 min, one at a time) beside the CPU lane (G0c, 15 atlases as checkpoints appear); then up to ~69 compares, 7 critics, dump-meta, 14 margin rebuilds, tie check | ~44-52 min |
| `block_b1` | see B1's RUN_REQUEST | ~21-40 min |
| pull, stop | | ~5 min |

## Relaunch and touch rules (integration D18)

- No `git` operation on the pod while `pod_atlas.sh` runs. `ATLAS_REBUILD` must be unset (`block_a4b` refuses it).
- A relaunch sets `ATLAS_A4B_CHECK_DIR=results/instrument_check_a4b_r2` and
  `ATLAS_B1_CHECK_DIR=results/instrument_check_b1_r2`; the margin preflight diverts its own record to scratch.
  Trainings with a checkpoint and a train.json, built atlases, and existing compares and critics are skipped.
- The evaluation then uses `--check-dir instrument_check_a4b_r2`.

## After the pull (Windows)

1. Commit the pulled results first (append-only; the pull command is printed at the end of the log). Stop the pod:
   billing continues until then.
2. Evaluate: `node scripts/a4b_eval.js --p <P> --p-run <P_run> --json results/atlas_v1_resnet56_s1/a4b_eval.json`.
3. Write `results/atlas_v1_resnet56_s1/SESSION.md`: P_run and the preflight record, gates G0a-G0e and R0, ladder and
   M11 / Mc, D-ID / D-COLL (row 11), K / C / T56, P, X and M56 (margin in the same SESSION.md), each prediction as
   `P<n>: predicted X, observed Y, verdict`. Then ATLAS_STATUS (row 11, the d56 tags of rows 1, 3, 4, 5, 6a, 7 and 9,
   S9 as INFO in row 8). A4b and B1 are evaluated independently; neither decision waits for the other.
