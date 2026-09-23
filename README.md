# Atlas

Map-first successor of [Upgraded-Mod](https://github.com/jhlee1995-cloud/Upgraded-Mod): builds an
atlas of a backbone's internal space (per-layer invariants of the activation geometry, and how it
deforms under corruption and test-time adaptation).

- [ATLAS_README.md](ATLAS_README.md): what is measured, how to run it, where data lives
- [AGENT_LOOP.md](AGENT_LOOP.md): the Planner / Proposer / Runner / Evaluator / Critic / Recorder loop
- [ATLAS_STATUS.md](ATLAS_STATUS.md): promoted map entries
- [MASTER_SUMMARY.md](MASTER_SUMMARY.md): history and methodology of the earlier project; details in `docs/history/`

Quick start on a RunPod pod with network volume `kxfir1tryb` (EU-RO-1) mounted at `/workspace`:

```bash
cd /workspace && git clone https://github.com/jhlee1995-cloud/Atlas.git
mkdir -p /workspace/logs && setsid bash /workspace/Atlas/pod_atlas.sh /workspace < /dev/null > /workspace/logs/launch.log 2>&1 &
```

Runtime code carried over from Upgraded-Mod@d9683cd: `extract/backbone.py`, `extract/data_loaders.py`,
`extract/populate_data.py`. Everything else from that repo is referenced, not copied.
