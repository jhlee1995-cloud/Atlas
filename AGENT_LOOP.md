# AGENT_LOOP — turning atlas-building into a surveying loop

The loop has six roles. Four are already code in this package; two are judgment and stay
with a model (Claude) reading and writing repo files. Every role communicates through files
in the repo, never through chat state, so any session (or any agent) can pick up mid-loop.

```
            ATLAS_STATUS.md  (promoted entries, ✅ 🟡 ⬜)          <- Recorder
                 ^                          |
                 |                          v
   CRITIC.md <- Critic (atlas.critic)     Planner  -> picks the next cell / question
                 ^                          |
                 |                          v
   atlas.json, DEFORMATION.md          Proposer -> new invariant module + manifest w/ predictions
                 ^                          |
                 |                          v
   Evaluator (atlas.compare + session log)  <-  Runner (atlas.run on the pod, commits results/)
```

## Roles and their artifacts

| role | who | reads | writes | tool |
|---|---|---|---|---|
| Planner | Claude | ATLAS_STATUS.md, CRITIC.md, open predictions in session logs | `experiments/queue/<next>.yaml` (chosen) | reasoning |
| Proposer | Claude | registry (`python -m atlas.build --list`), ATLAS.md gaps | `atlas/invariants/<new>.py`, manifest `notes:` with P1..Pn pre-registered | code |
| Runner | pod (human now; Actions later) | manifest | `results/<exp>/` (atlas.json, ATLAS.md, provenance.json) | `atlas.run` |
| Evaluator | Claude | atlas.json, DEFORMATION.md, manifest notes | `results/<exp>/SESSION.md`: prediction vs outcome, one line each | `atlas.compare` |
| Critic | code + Claude | ≥2 results dirs, manifest holdout, methodology rules | `CRITIC.md` verdict + methodology flags | `atlas.critic` |
| Recorder | Claude | CRITIC.md, SESSION.md | `ATLAS_STATUS.md` (promote / demote entries) | edit |

The unit of work is one manifest. The unit of knowledge is one ATLAS_STATUS entry:
`(layer, invariant or factor, claim, status, evidence dirs)`.

## Promotion rule (what ✅ means)

An entry becomes ✅ only when all four hold:
1. source = real (critic `synthetic_refusal` PASS)
2. replicates across ≥2 independent seeds within `experiments/tolerances_default.yaml`
3. the manifest declared discovery/confirmation splits BEFORE the run (critic `holdout_hygiene`)
4. the prediction it answers was written in `notes:` before the run (Evaluator quotes it)

🟡 = measured once on real data, not yet replicated. ⬜ = proposed, not measured.
A tolerance may be changed only with a SESSION.md line saying why, and never to flip a
specific entry from FAIL to PASS.

## Automation levels (raise one level at a time)

- **L0 (now):** human runs `pod_atlas.sh` on the pod (repo cloned onto the network volume),
  pulls `results/` (no dumps) back to Windows over SSH and commits there; the pod never pushes.
  Claude does Planner / Proposer / Evaluator / Recorder in chat, reading `raw.githubusercontent.com`.
- **L1:** Claude Code session in the repo runs everything CPU-side itself (`critic`,
  report edits, reading the pod-computed `compare_vs_*/DEFORMATION.md`, new invariant modules with tests) and opens a PR; human
  still runs GPU jobs. `CLAUDE.md` is the session brief for this level.
- **L2:** GitHub Actions on push to `experiments/queue/` launches the RunPod job
  (`pod_atlas.sh`), which commits `results/<exp>/` (no `dump/`); the workflow must ignore
  commits to `results/` to avoid re-triggering. Claude Code reviews the result PR.
- **L3:** Proposer generates manifests without a human, but only from an allowlist
  (existing invariants x existing backbones x declared holdouts); anything new in
  `atlas/invariants/` still goes through PR review. A human approves promotion to ✅.

Do not skip a level. The Critic must be **backtested** before L2: replay it on the two
historical failures already in MASTER_SUMMARY (the stream-builder bug that produced
alternating instead of sustained blocks; the batch-averaging ceiling that made every type-b
AUC 1.00) and confirm it would have flagged both. If it would not, add the check first.

## Guardrails encoded in the tools

- **Synthetic refusal:** `critic` never lets a synthetic-source run PASS anything.
- **Multiple comparisons:** manifests declare `holdout.discovery_*` / `confirmation_*`;
  the critic WARNs when absent. Rule of use: search (new invariants, tolerances, layer
  choices) on discovery seeds/corruptions only; run the confirmation set once, after the
  definition is frozen in a commit.
- **Cheap before expensive:** `build` orders invariants by declared `cost`; Planner should
  order manifests the same way (Stage 0 -> seed -> scale -> ViT -> CLIP).
- **Per-sample honesty:** probes are per-sample CV scores; `probe_hygiene` fails if
  `cv_folds < 3`. Batch-averaged numbers do not enter the atlas.
- **Instrument validation:** `tests/test_atlas_smoke.py` has known-answer checks (TwoNN on
  a 3-d manifold, CKA identity, monotone displacement). Add one for every new invariant
  before it runs on real data.
- **Provenance:** every result dir carries `manifest_used.yaml` and `provenance.json`
  (git commit, weights source, dump meta). No number without a path.

## The first three loops (already planned)

1. **Stage 0** `atlas_v0_resnet20_cifar10` — predictions P1..P5 are in the manifest notes.
2. **Stage 1** `atlas_v1_resnet20_{s0hub,s1,s2}` — the kill switch (docs/plans/STAGE1.md).
   Two locally trained seeds with the hub recipe and the hub's training normalization, plus a
   reference-resample twin and a random-init null. The Evaluator applies the pre-registered
   KILL / PASS / PARTIAL rule in STAGE1.md; the critic's count-based verdict word is advisory.
   (`atlas_v0_resnet20_seed1` is superseded: it confounded seed with recipe and normalization.)
3. **Stage 2** resnet56 (hub, `norm: chenyaofo`, `block_stride: 5`) — scale transfer (docs/plans/STAGE2.md):
   `compare` pairs the 11 taps by position (and refuses a cross-depth name match), the critic runs with
   `--align position`, resnet20 is re-measured in the same session (`atlas_v1_resnet20_*_st2`), and an accuracy
   ladder decides whether a difference is attributable to depth. It shares one pod session with Stage 1b (seeds 3,
   4; STAGE1.md amendment 2) and A3 (margin/type-b, row 9; docs/plans/A3_MARGIN.md):
   `results/atlas_v1_resnet20_s3/RUN_REQUEST.md`.
   Its follow-up, **Stage 2b (A4b)**, is docs/plans/STAGE2B.md (STAGE2.md amendment 1): resnet56 seeds 1-2, seed-11
   matched rungs with seed-12/13 replicates, the `_ref1` twin and a same-session `_st3` resnet20 band; D-ID / D-COLL
   become ATLAS_STATUS row 11, and the evaluation is frozen as `node scripts/a4b_eval.js`. It shares one pod session
   with B1 (row 10): `results/atlas_v1_resnet56_s1/RUN_REQUEST.md`.
   **B1 (the ViT margin test, MASTER P1)** is docs/plans/B1_VIT_MARGIN.md (pre-registration in the
   `experiments/queue/margin_b1_vitb16.yaml` notes): margin_typeb on ImageNet val for torchvision ViT-B/16 and DeiT-B,
   run once and only after a same-session ResNet50 gate passes (legacy 0.800 reproduction G0-G2, positive control G3;
   `scripts/b1_gate.py`). Its Stage A is the separate entry point `python -m atlas.run_imagenet`; the decision is frozen
   as `node scripts/b1_verdicts.js` (`--a4b` adds the joint reading beside A4b's depth-56 margin tag). Request:
   `results/margin_b1_vitb16/RUN_REQUEST.md`.

After these, the deformation ladder starts: `scripts/tta_deform.py` (rung 1, TENT, two
manifests: standard dose and collapse positive control) writes `dump_step<k>/` checkpoints,
`python -m atlas.ladder` turns them into LADDER.md with the onset-vs-histogram-collapse bar.
Rungs 2+ (held-out EATA/CoTTA, partial and full fine-tuning) add adapter loops only;
`compare --same-space` and `ladder` are unchanged.

## What the loop is NOT

It does not decide what the map is for. Planner picks cells by gap size in ATLAS_STATUS,
not by application. The three applications (immune / merge / compute) read the finished
entries; they do not steer which invariants get measured. That separation is what keeps
the atlas reusable.
