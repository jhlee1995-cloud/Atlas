# RUN_REQUEST: Stage 2 (A4, resnet20 -> resnet56)

Stage 2 runs in the combined pod session with Stage 1b and A3: the exact command, pod requirements and timeline are
in `results/atlas_v1_resnet20_s3/RUN_REQUEST.md` (`pod_atlas.sh /workspace --stage1b --stage2 --a3`; the `--stage2`
block trains the resnet56 ladder one rung at a time, after the Stage 1b trainings).

Design and rules: `docs/plans/STAGE2.md`; predictions R56-0..9 in `experiments/queue/atlas_v1_resnet56_s0hub.yaml`
(notes). Evaluate in `results/atlas_v1_resnet56_s0hub/SESSION.md`, only after the Stage 1b decision is committed
(`docs/plans/STAGE1.md` amendment 2 item 7).
