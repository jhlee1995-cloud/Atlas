# CLAUDE.md — Atlas loop (agent brief, automation level L1)

Claude Code runs as the CPU-side agent of the loop described in AGENT_LOOP.md. Discussion
with the human is in Korean; code, commits, and result files are in English.

## Where data lives

- Datasets, dumps, models, caches and logs live on the RunPod network volume `kxfir1tryb`
  (EU-RO-1), mounted at `/workspace` on pods. The repo is cloned to `/workspace/Atlas`, so
  `results/<exp>/` (including the gitignored `dump/`, `dump_step*/`) is on the volume.
- The pod only pulls code. `results/` (no dumps) is pulled to Windows over SSH and committed
  there. Anything that needs a dump (real-data `build`, `compare`, `ladder`) runs on the pod.
- Commit and push manifests (with predictions) before any pod run.
- Legacy code (axes, frame, analysis) stays in `jhlee1995-cloud/Upgraded-Mod@d9683cd`; only
  `extract/{backbone,data_loaders,populate_data}.py` were carried over. History docs are in
  `docs/history/`.

## What this repo is

A label-free runtime perception trust system (see MASTER_SUMMARY.md) that now builds an
**atlas** of the backbone's internal space first (atlas/, ATLAS_README.md). The atlas is the
shared substrate for three later applications (immune-style deformation detection, central
knowledge merging, novelty-proportional compute). You do not work on those applications;
you build and validate the map.

## Your roles in this session

Planner, Proposer, Evaluator, Recorder. The Runner (GPU) is the human; the Critic is
`python -m atlas.critic` plus the methodology rules below. Every action ends in a file
under `results/`, `experiments/queue/`, `atlas/`, or `ATLAS_STATUS.md`, and a commit.

## Commands you may run

```bash
python -m pytest tests/ -v                                   # always before proposing code
python -m atlas.build --list                                 # registry
python -m atlas.build --dump results/<exp>/dump --out results/<exp> [--manifest ...]
python -m atlas.critic --results results/<A> results/<B> --tol experiments/tolerances_default.yaml --out results/critic_<pair>
python -m atlas.synth --out /tmp/synth && python -m atlas.build --dump /tmp/synth --out /tmp/synth_out
```

`atlas.compare` needs both `dump/` dirs, which stay on the volume, so it runs on the pod. Locally,
read the pulled `results/<B>/compare_vs_<A>/DEFORMATION.md`; do not re-run it (without dumps it
drops the panel CKA / relrep metrics, and it refuses to overwrite a file that has them).
On Windows set `PYTHONUTF8=1` before running any of the above.

You may not run Stage A (`atlas.extract_acts`, `atlas.run` without `--skip-extract`) or
training; write the exact command into `results/<exp>/RUN_REQUEST.md` and stop.

## Methodology rules (from MASTER_SUMMARY, enforced here)

1. Never conclude redundancy from synthetic data. Synthetic runs are pipeline checks only.
2. Per-sample metrics are the honest metric; never report batch-averaged probe scores.
3. Pre-register predictions in the manifest `notes:` before the run. Evaluate each one in
   `results/<exp>/SESSION.md` as `P<n>: predicted X, observed Y, verdict`.
4. Validate the instrument before interpreting: a new invariant needs a known-answer test in
   `tests/` and a synthetic run with zero errors before any real-data run.
5. Sweep, do not pin: when a threshold or tolerance matters, report the curve.
6. Scale-confirmation-bias guard: any claim about a bigger backbone needs the same
   measurement on the small backbone in the same session.
7. Discovery vs confirmation: search only on `holdout.discovery_*`; touch confirmation sets
   once, after the definition is frozen in a commit.
8. Skepticism is a feature: when a number looks too good (probe R^2 > 0.95, CKA = 1.0
   across seeds), write the artifact hypothesis first and test it.

## Promotion to ATLAS_STATUS.md

✅ requires: real source, replicated across ≥2 seeds within tolerance, holdout declared
before the run, prediction written before the run. 🟡 one real measurement. ⬜ proposed.
Never edit `experiments/tolerances_default.yaml` to flip one entry; if a tolerance is wrong,
change it with a SESSION.md line explaining why, and re-run the critic on everything.

## What to do in a session (in order)

1. Read ATLAS_STATUS.md, the latest CRITIC.md, and the newest `results/*/SESSION.md`.
2. If a result dir has atlas.json and provenance.json but no SESSION.md: evaluate predictions,
   read the pod's `compare_vs_*/DEFORMATION.md` and run critic against its twin if one exists
   (no compare_vs_* yet: write the compare command into RUN_REQUEST.md), write SESSION.md,
   update ATLAS_STATUS.md, commit.
3. Else pick the largest gap (⬜ or 🟡 entries, or an open row-6 question) and either write
   a manifest with predictions (Planner) or a new invariant with tests (Proposer). Commit
   with a message starting `atlas:`.
4. Stop after one loop iteration. Summarize in Korean what changed and what the human needs
   to run on the pod.

## Never

- Never fabricate a number; every value in ATLAS_STATUS.md links to a results dir.
- Never delete or rewrite `results/`; append.
- Never import `datasets` (HF) into the torch environment (known crash); use `huggingface_hub`.
- Never prune an invariant because it looked redundant on one run.
