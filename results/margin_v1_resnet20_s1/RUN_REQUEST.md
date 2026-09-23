# RUN_REQUEST: A3 margin_typeb (row 9)

A3 runs in the combined pod session with Stage 1b and Stage 2: the exact command, pod requirements and timeline are
in `results/atlas_v1_resnet20_s3/RUN_REQUEST.md` (`pod_atlas.sh /workspace --stage1b --stage2 --a3`; the `--a3`
block runs last because it reads the s3, s4 and resnet56 dumps).

Pre-registration: `experiments/queue/margin_v1_resnet20_s1.yaml` (notes), committed with every margin_* manifest,
`atlas/invariants/margin.py` and `docs/plans/A3_MARGIN.md` before the run. The confirmation rebuilds (s1, s2,
s1_ref1, s3, s4) run only if the M1 margin clause passes on v0. Evaluate in `results/margin_v1_resnet20_s1/SESSION.md`.
