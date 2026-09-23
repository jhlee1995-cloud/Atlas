# RUN_REQUEST: Stage 1 (v1)

Commit with every edit, manifest and prediction pushed first (docs/plans/STAGE1.md).
Pod: RTX 4090 (or any Ampere+ GPU), EU-RO-1, network volume kxfir1tryb at /workspace, template runpod-torch-v280.

```bash
git -C /workspace/Atlas fetch && git -C /workspace/Atlas reset --hard origin/main   # dumps are untracked; kept
mkdir -p /workspace/logs && setsid bash /workspace/Atlas/pod_atlas.sh /workspace --stage1 < /dev/null > /workspace/logs/launch.log 2>&1 &
tail -F /workspace/logs/launch.log
grep -h "epoch   10" /workspace/logs/train_resnet20_s*.log   # x20 = projected training time
```

Expected 35-60 min. Then pull results/ without dumps (command printed at the end), commit on Windows,
and write results/atlas_v1_resnet20_s1/SESSION.md with the N/R0/S/C/T verdicts and the STAGE1 decision.
