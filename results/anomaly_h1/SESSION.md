# SESSION: ANOMALY_H1 — mechanism predictions (AH) and missing-axis probes (AX), decided before B1

This is the Evaluator pass on the pre-registered ANOMALY_H1 items. AH-4 depends on B1, so it is **DEFERRED** to an append after B1's own verdicts exist. Every other item is decided here.

- **Pre-registration.**
  - `docs/plans/ANOMALY_H1.md` holds the predictions, labels and promotion rules. The decision rule is frozen as `scripts/anomaly_eval.js`. The CPU probe is `scripts/anomaly_probe.py`, and its known-answer tests are in `tests/test_anomaly_probe.py`.
  - All of these were committed together as **P_A = `d257d91`** (`d257d9126322d458b7beb8718d19f42d8bfc2606`) at 2026-09-23T06:32:32-07:00, which is 13:32:32 UTC. P_A is a child of the batch-3 commit P = `f1c3c43`.
- **What the pod ran.**
  - **P_run = `d257d91` = P_A.** Pod `nc2mzssur3bb0k` ran `pod_atlas.sh --a4b --b1 --anomaly`.
  - `block_anomaly` finished OK at 2026-09-23 15:30:55 UTC.
  - Its results were pulled and committed as `d67aa14` before this evaluation: the 11 `results/anomaly_probe_*/probe.json` files and `results/instrument_check_anomaly/`.
- **Inputs decided elsewhere.**
  - A4b was decided in `b76f6d6` (`results/atlas_v1_resnet56_s1/SESSION.md`, `a4b_eval.json`).
  - B1 is being relaunched (`results/margin_b1_vitb16/RELAUNCH_r2.md`, timeout deviation `bf4d531`).
- **Everything here is discovery-derived.** The 71 hypotheses behind these items came from the batch-2 anomaly analysis. The only evidence is the frozen confirmation clauses, read on fresh instances:
  - resnet56 s1 and s2 (primary);
  - s12m and s13m (independent seeds at the matched length);
  - the seed-11 rungs e50, e60 and e70. These count as one training run.

  The twin clauses (AH-1(d), AH-8(e)) re-measure discovery weights and are noise clauses only. The discovery values quoted below are discovery, never support.
- **Amendments:** none. No bug fix to the plan, evaluator or probe was made after P_A.

## Outcome

| item | label | failed clause(s) / note |
|---|---|---|
| AH-1 | **SUPPORTED** | no exceptions |
| AH-2 | **SUPPORTED** | no exceptions; reading (d): the energy HOLD rule fails again, as predicted |
| AH-3 | **REFUTED** | (d) the penult SE ratio in s1 is 1.109, against ≥ 1.25 |
| AH-4 | **DEFERRED** | not evaluated; it waits for B1's `verdicts.json` (see "AH-4") |
| AH-5 | **REFUTED** | (c) in s1 and s2 (penult enrichment 0.789 / 0.777, against ≥ 0.80); (d) in s1 (penult excess 0.102, against ≤ 0.10); (a) in secondaries e50 and e60 (stem PC1 0.47-0.57, against ≤ 0.35) |
| AH-6 | **REFUTED** | (b) in s1 (1.156, against ≥ 1.25); (d.c4) in s1 (0.1125, against ≥ 0.15); (d.order) (0.1125 is not > 0.1355, s13m) |
| AH-7 | **REFUTED** | (b) Spearman(E, c) = 0.667, against ≥ 0.8; (c) e70 CV_all = 0.0524, against ≥ 0.056. (a) reads PREDICTED |
| AH-8 | **SUPPORTED** | no exceptions |
| AH-8d | **SUPPORTED** | — |
| AX-1 | gate **DROPPED** → **ruled out** | the r20 hub fails c3 (motion s1 +0.068) and the single-class FPR (0.104). The s1/s2 values are INFO. The double holdout is INFO |
| AX-2a | gate **CLOSED** → **stays a proposal** | the r56 hub fails (a): brightness s1 argmax set {stem, l10, s1end}, with s1end inside the 0.005 tie band. s1/s2 are spent for this axis |
| AX-2b | gate **OPEN** → **SUPPORTED**; double holdout **SUPPORTED** | every family routes at 1.000 on s1 and s2; impulse → N 1.000, zoom → B 1.000. See rule 8 (near-trivial) |
| AX-3 | gate **OPEN** → **SUPPORTED** | ρ 0.973 / 0.972; 1/1551 and 1/1549 in-band batches lose > 10 pt. See rule 8 (split-level) |
| AX-4 | gate **DROPPED** → **ruled out** | all four gains are −0.16 to −0.41. The s1/s2 values are INFO. The double holdout is INFO |

**Tally of scored items:** 8 decided AH items and 2 scored AX axes.
- 4 AH items are SUPPORTED: AH-1, AH-2, AH-8 and AH-8d.
- 4 AH items are REFUTED: AH-3, AH-5, AH-6 and AH-7.
- AX-2b and AX-3 are SUPPORTED.
- AX-1 and AX-4 are ruled out, and AX-2a is CLOSED.

**Nothing is promoted.** No SUPPORTED label changes ATLAS_STATUS (ANOMALY_H1 section 7). See "What ANOMALY_H1 does not license".

Path shorthands:

| shorthand | path |
|---|---|
| `s1`, `s2`, `s12m`, `s13m`, `e50`, `e60`, `e70` | `results/atlas_v1_resnet56_<x>/atlas.json` |
| `M<x>` | `results/margin_v1_resnet56_<x>/atlas.json` |
| `h56`, `tw`, `e40s` | `results/atlas_v1_resnet56_{s0hub_st3,s0hub_ref1,e40_st3}/atlas.json` (replays: INFO, or the twin) |
| `B` | `results/atlas_v1_resnet20_{s0hub,s1,s2,s3,s4}_st3/atlas.json` (R20 band) |
| `ev` | `results/atlas_v1_resnet56_s1/a4b_eval.json` |
| `pr:<tag>` | `results/anomaly_probe_<tag>/probe.json` `.dumps.atlas_v1_<tag>` |
| `st` | `results/instrument_check_anomaly/selftest.json` |
| `pl.X` | `.per_layer.X` |

## Run facts

- **Atlases read by the AH items.** All were built by A4b at P_run:
  - the primaries: s1, s2 and their `margin_v1_` rebuilds;
  - the new instances: s12m, s13m, e50, e60 and e70 (with margin rebuilds). e90 was not built, correctly: `ev` `.ladder.status` is "matched";
  - the replays: h56, tw and e40s;
  - the five R20 band atlases.
- **Ladder.** `ev` `.ladder`: `status` "matched", `M11` = 60 (e60), `Mc` = M12 (s12m) and M13 (s13m), both evaluable.
- **Training fit.** `meta.accuracy.ref` is ≥ 0.9981 for every rung and for s12m and s13m, and 1.0 for s1 and s2, so no AH-7 misfit caveat applies.
- **Probe records** (`pr:<tag>` `.role`, `.splits_read`, `code`, `created_utc`), in probe order:

| order | tag | role | splits read | holdout splits | created_utc |
|---|---|---|---|---|---|
| 1 | resnet20_s0hub_st3 | discovery (gate) | 33 | 0 | 15:12:36Z |
| 2 | resnet56_s0hub_st3 | discovery (gate) | 33 | 0 | 15:14:16Z |
| 3 | resnet56_e40_st3 | discovery (AX-3 gate) | 33 | 0 | 15:15:58Z |
| 4 | resnet56_s1 | confirmation | 43 | 10 | 15:17:36Z |
| 5 | resnet56_s2 | confirmation | 43 | 10 | 15:19:17Z |
| 6-9 | resnet20_s{1,2,3,4}_st3 | discovery (context, INFO) | 33 | 0 | 15:20:58Z-15:25:55Z |
| 10 | resnet20_rand | null (INFO) | 33 | 0 | 15:27:36Z |
| 11 | resnet56_rand | null (INFO) | 33 | 0 | 15:29:15Z |

Every record has `status` "OK" and `error` null.

- **How it was evaluated.**
  - A preliminary run of the frozen evaluator was made on this machine:
    - `node scripts/anomaly_eval.js --p d257d91 --p-run d257d91 --a4b results/atlas_v1_resnet56_s1/a4b_eval.json`;
    - `--b1` pointed at a non-existent scratch file, so AH-4 printed NOT_EVALUABLE. It is DEFERRED here, not labelled;
    - `--cache` and `--json` were written to scratch, and the output is not committed.
  - Two evaluator families then recomputed every clause from the raw files with their own node scripts. The AH family covered AH-1..AH-8d, including an ID_gauss port written from the section-2 text. The AX family covered AX-1..AX-4 and probe provenance.
  - A decider pass re-checked the contested points from the raw files:
    - AH-3 (all clauses, and e70's position against the sep subsets);
    - AH-5 (a)-(d);
    - AH-6 (a)-(d);
    - AH-7 (a)-(c);
    - AH-8 (a)-(c) against the cache;
    - AH-1 (a)-(c);
    - the AX-1, AX-2a, AX-2b and AX-4 gates and confirmation values;
    - AX-3 (a)-(c) with its rule-8 controls;
    - probe provenance.
  - **No numeric or label discrepancy was found.**
    - The AH port of ID_gauss is bit-identical to the evaluator on all 105 cache keys (max |diff| 0).
    - The replays reproduce the section 8.1 discovery values: h56 ρ 1.021 with ID_gauss 10.45, and r20 hub 0.930 / 10.66.
    - The AX family matched all 32 numeric fields it compared.
  - **The official record** (`results/anomaly_h1/eval.json`, written by the section 9 command with `--b1 results/margin_b1_vitb16/verdicts.json`) is not written yet. It waits for B1.
    - Every non-AH-4 value is a deterministic function of committed files, and the ID_gauss cache is exact.
    - So the official run must reproduce every label in this file. Any difference is a discrepancy to be appended here.
- **Nothing in the repository was modified** except this file. The scratch scripts are not committed.

## Provenance and guards (section 1)

| check | observed | result |
|---|---|---|
| P_A and P_run resolve; P_run is P_A or a descendant | both `d257d9126322…` | PASS |
| `docs/plans/ANOMALY_H1.md` and `scripts/anomaly_eval.js` are unchanged from P_A to P_run | same commit. Also `git diff d257d91 HEAD` is empty for all four frozen files; at this evaluation HEAD = `2ad4e9c` | PASS |
| `scripts/anomaly_probe.py` and `tests/test_anomaly_probe.py` are unchanged (AX) | same | PASS |
| Only code change after P_A | `bf4d531`: `pod_atlas.sh` (the B1 stage time limit). It touches no frozen file | no effect |
| Each atlas read has `meta.git_commit` = P_run and `meta.created` after P_A | every atlas read by an AH item: `d257d91`, created 13:37:51 (r20 s0hub_st3) to 14:25:08 (s2); s1 14:22:24. That covers the 7 fresh resnet56 atlases and their 7 `margin_v1_` rebuilds, h56, tw, e40s and the 5 R20-band atlases. Each margin rebuild carries its atlas's `meta.created` (the dump time) | PASS |
| Self-test (AX) | `st` `.status` "PASS", `.failed` [], 32/32 checks, `.code_sha256` `5054e8b61375…c00fa5e7` = sha256(P_A:`scripts/anomaly_probe.py`), created 15:12:36Z (python 3.12.3, numpy 2.1.2). No `_r2` record exists | PASS |
| Every probe.json read (AX) | `code.repo_commit` is the full `d257d91` in all 11; `code.sha256` is the same as the self-test; `created_utc` 15:12:36Z to 15:29:15Z, after P_A | PASS |
| Confirmation dumps read once | one record per dump, no `_r<k>` directory, no `instrument_check_anomaly_r2`; the evaluator's `probe_duplicates` is [] | PASS |
| A4b guard (AH items other than AH-4) | `ev` `.guard` [], `.R0_56` true, `.instrument_gate.ok` true | no guard |
| Primary-atlas guard (s1, s2, and for AH-3 `Ms1`, `Ms2`) | `source` "real", `skipped` {}, and no `error` key at any depth | no guard |
| AX confirmation / double-holdout guard | the A4b guard, plus `ev` `.G0b.i2_status` PASS | no guard |

Notes:
- `selftest.json` and the first probe record share the same second (15:12:36Z), so their timestamps alone cannot show that the self-test ran first. `block_anomaly` runs the self-test as a hard step under `set -e` before any probe, and the block ended OK.
- No pytest record is committed. A pass is inferred only from the block's errexit, and the text requires only `selftest.json`.
- `meta.created` has no time zone. It is read as UTC, which is consistent with A4b's UTC block times.

## Per item: predicted → observed → verdict

### AH-1. The collapse level sets the train-reference density baseline — **SUPPORTED**

Runs:
- primary: s1, s2;
- secondary-refuting: s12m, s13m, e50, e60, e70;
- INFO: h56, e40s, tw;
- twin: h56 → tw.

Fields:
- `pl.penult.knn_density.splits.test.{sparse_frac, median_log_radius_shift}`;
- `pl.penult.neural_collapse.nc1`;
- `pl.<L>.knn_density.splits.test.sparse_frac`.

| clause | predicted | discovery (not evidence) | observed | result |
|---|---|---|---|---|
| (a) curve | \|sparse − (0.0224 − 0.1080·log10 nc1)\| ≤ 0.035 | residuals −0.013 to +0.009 | s1 +0.003 (nc1 0.0529, sparse 0.163); s2 +0.021 (0.0502, 0.184); s12m +0.002, s13m +0.013, e50 +0.014, e60 +0.012, e70 +0.013 (nc1 0.117-0.145). INFO: h56 +0.003, e40s −0.002, tw +0.006 | PASS (7/7) |
| (b) step | maximum pre-penult test sparse_frac (layer3.8 excluded) ≤ 0.070 | 0.064 | s1 0.061, s2 0.059; new instances 0.057-0.060 | PASS (7/7) |
| (c) tail | penult median shift ≤ 0.05 while sparse ≥ 0.12 | h56 0.027 at 0.167 | s1 0.0396 / 0.163; s2 0.0309 / 0.184 | PASS |
| (d) twin | if \|Δq95\| ≥ 0.01, sign(mean Δsparse) = −sign(Δq95); \|mean Δmed\| ≤ 0.02; SD0(Δmed) ≤ 0.015 | r20 twin Δq95 −0.044, Δsparse +0.049 | Δq95 −0.0188, so the sign condition is active; mean Δsparse +0.0096 (opposite sign); Δmed +0.0073 ± 0.0060; n = 31 | PASS |

Verdict: **SUPPORTED**, with no exceptions. The informative evidence is (a) at the five new instances. They fill the nc1 range 0.10-0.15, which was unmeasured in discovery, and two of them (s12m, s13m) are independent seeds. At s1 and s2, (a) sits near the hub's nc1 (about 0.05). Rule 8 notes on (b) and (d) are below.

### AH-2. A normalized harm grade H: HOLD vs ADAPT; the energy rule fails — **SUPPORTED**

Runs: primary s1, s2; secondary s12m, s13m, e50, e60, e70. H and cost follow section 2.

Fields:
- `pl.penult.knn_density.{splits.<S>.median_log_radius_shift, ref_log_radius_quantiles}`;
- `meta.accuracy`;
- `pl.penult.corruption_displacement.splits.<S>.norm_ratio`.

| clause | predicted | discovery | observed | result |
|---|---|---|---|---|
| (a) | Spearman(H, cost) ≥ 0.93 | 0.948-0.992 | s1 0.989, s2 0.987; secondaries 0.980-0.992 (s13m 0.992) | PASS |
| (b) HOLD band | every split with H ≤ 0.25 costs ≤ 8.0 pt | h56 4.4 over 8 | s1 maximum 5.28 over 8 splits (brightness s5); s2 2.84 over 7; secondaries 4.17-5.64 over 8-9 | PASS |
| (c) | brightness s1, s3 and s5 all have H ≤ 0.30 | maximum 0.239 | s1 0.024 / 0.062 / 0.208; s2 0.020 / 0.104 / 0.264; secondaries maximum 0.242 | PASS |
| (d) reading | ≥ 1 split with norm_ratio ≥ 0.95 and cost > 10 pt | h56 3 | s1 has 4 (gaussian s1, motion s3, snow s3, snow s5); s2 has 9 (gaussian s1/s3/s5, shot s3/s5, snow s3/s5, pixelate s3/s5) | energy rule fails again, as predicted |

Verdict: **SUPPORTED**, with no secondary exceptions. s13m's 0.992 is above the section-7 "too good" line of 0.99. Its artifact reading is in the rule 8 section.

### AH-3. After the final collapse, margin equals distance — **REFUTED**

Runs: s1 and s2 with their margin rebuilds (a-d); M56new, all 7 built (e).

Fields: `M<x>` `pl.{layer3.5,penult}.margin_typeb.*`; `<x>` `pl.{layer3.5,penult}.class_centers.{sep_ratio, nearest_center_agrees_with_model}`.

| clause | predicted (s1 and s2) | discovery h56 / r20 | observed s1; s2 | result |
|---|---|---|---|---|
| (a) | margin_minus_dist_wrong ≥ +0.15 at layer3.5, ≤ +0.02 at penult | 0.207 / −0.000 | 0.209 / 0.0069; 0.168 / 0.0005 | PASS |
| (b) | Δauc_dist_wrong ≥ 0.25 and ≥ 1.8 × Δauc_margin_wrong | 0.337 vs 0.130 | 0.361 vs 0.158; 0.307 vs 0.139 | PASS |
| (c) | spearman_margin_maxprob ≥ 0.85 at penult, ≤ 0.60 pre; nearest_center_agrees ≥ 0.985 at penult, ≤ 0.90 pre; spearman_margin_dist ≤ −0.80 at penult | 0.886 / 0.474; 0.997 / 0.852; −0.853 | 0.886 / 0.479, 0.997 / 0.842, −0.857; 0.878 / 0.479, 0.997 / 0.856, −0.841 | PASS |
| **(d)** | se_maxprob_typeb / se_margin_typeb ≥ 1.25 at penult, ≤ 1.10 at layer3.5 | 1.539 / 0.980 | **s1 1.109** / 0.603; s2 1.253 / 0.844 | **FAIL (s1)** |
| (e) graded | Spearman(sep, lead) ≤ −0.6; sep ≤ 3.5 → lead ≥ +0.02; sep ≥ 4.5 → \|lead\| ≤ 0.02 | stand-in −0.83 | ρ −0.857 (n = 7). Low sep: e50 3.140 / +0.035, s13m 3.314 / +0.022, s12m 3.367 / +0.031, e60 3.368 / +0.031. High sep: s1 5.114 / +0.008, s2 5.213 / +0.0005. e70 (3.506 / +0.022) is in neither subset, and it would pass the low-sep bound anyway | PASS |

Verdict: **REFUTED**. The failed clause is (d), the "saturation signature" sub-claim (A-H09): s1 is 0.141 short, and s2 clears the bound by only 0.003.

`ev` `.M56.c_confidence.tie_call` is [NO-LEAD, TIES-EXCLUDED]. No seed is TIE-ARTIFACT-POSSIBLE, so no float32-saturation reading attaches to (d).

### AH-4. The collapse regime decides margin vs distance in every architecture — **DEFERRED**

Not evaluated. It is recorded as DEFERRED to the evaluation after B1; see the "AH-4" section.

### AH-5. Brightness takes a learned, class-neutral path and is absorbed — **REFUTED**

Runs: primary s1, s2; (a) is also secondary-refuting in s12m, s13m, e50, e60 and e70.

| clause | predicted | discovery | observed | result |
|---|---|---|---|---|
| **(a) stem** | brightness s1 and s5: \|dir[0]\| ≤ 0.35 and √(dir[1]²+dir[2]²) ≥ 0.85 (`pl.stem.corruption_displacement.splits.corrupt__brightness__s{1,5}.direction_pca20`) | trained ≥ 20 epochs: 0.008-0.299 / 0.906-0.995 | s1 0.192, 0.196 / 0.978, 0.968; s2 0.162, 0.184 / 0.971, 0.936; s12m 0.174, 0.169; s13m 0.002, 0.013; e70 0.202, 0.175 (all PASS). **e50 0.561, 0.571 / 0.805, 0.772; e60 0.496, 0.472 / 0.848, 0.830** | **FAIL (e50, e60)** |
| (b) layer3.5 | brightness class_sub_frac ≤ 0.33 at s1 and s3, and ≤ 0.70 × B/T; lowest at s1 with gap ≥ 0.08 | h56 0.195 / 0.232, gap 0.148 | s1 0.130 / 0.152 (bound 0.267, B/T 0.381), lowest, gap 0.176 to snow; s2 0.180 / 0.231 (bound 0.285, B/T 0.407), lowest, gap 0.170 to snow | PASS |
| **(c) penult** | B/T ≥ 0.88 and class_sub_frac / (B/T) in [0.80, 1.20] for all 30 splits | h56 0.934, 0.816-1.027 | s1 B/T 0.930, range **0.789 (snow s1)** to 1.050; s2 0.931, **0.777 (snow s3)** to 1.020. One split of 30 is below 0.80 in each seed | **FAIL (s1, s2)** |
| **(d) absorbed** | cost(brightness s5) ≤ 6.5 pt; penult excess (split − test sparse_frac) ≤ 0.10 | 3.8-6.0 pt; 0.04-0.08 | s1 5.28 pt, **0.102**; s2 4.39 pt, 0.078 | **FAIL (s1)** |
| (d.stem) INFO | stem excess ≥ 0.25 | nulls 0.32-0.36 | s1 0.376, s2 0.369 | INFO |

Verdict: **REFUTED**, on (a) in e50/e60, (c) in s1/s2 and (d) in s1. How serious each failure is:
- (a) is a genuine failure by a clear margin: two trained nets put about half of the brightness shift on stem PC1. In every run, brightness still lies inside the top-3 stem PCs (norm ≥ 0.95).
- (c) is a threshold-tightness failure: the 0.80 bound sat 0.016 below the single discovery instance.
- (d) is at noise level; see rule 8.

All three count under the text.

### AH-6. Motion blur is directional, noise keeps moving, and C4 tracks collapse — **REFUTED**

Runs: primary s1, s2; (d.order) against the built Mc (s12m, s13m); rungs INFO.

Fields: `pl.{layer3.5,penult}.corruption_displacement.splits.<S>.{direction_pca20, magnitude, coherence}`, `pl.penult.knn_density.splits.<S>.median_log_radius_shift`, `pl.penult.class_centers.sep_ratio`.

| clause | predicted | discovery | observed s1; s2 | result |
|---|---|---|---|---|
| (a) | cos15(motion) ≥ 0.90 at penult and layer3.5; inc ≥ 0.85; cos15 gap to snow ≥ 0.25 | gap h56 0.388 | 0.942, 0.928, 0.945, gap 0.409; 0.953, 0.945, 0.933, gap 0.387 | PASS |
| **(b)** | [mag(motion s5)/mag(gauss s5)] at layer3.5 ≥ 1.25 × the same at penult | h56 1.659, e40 1.348 | **1.156**; 1.327 | **FAIL (s1)** |
| (c) | g53 ≥ 1.10, s53 ≥ 1.20, shot Δmed ≥ +0.03, gauss Δmed ≥ −0.02 | h56 1.179 / 1.321 / +0.122 / +0.021 | 1.129 / 1.262 / +0.110 / +0.027; 1.157 / 1.302 / +0.087 / +0.067 | PASS (nulls pass more strongly) |
| **(d.c4)** | C4 min margin ≥ 0.15 | h56 0.192 | **0.1125** (shot s5 − defocus s5); 0.187 | **FAIL (s1)** |
| **(d.order)** | premise: min sep(s1, s2) > max sep(Mc). Then min C4(s1, s2) > max C4(Mc) | — | premise 5.114 > 3.367 holds; 0.1125 is not > 0.1355 (s13m); s12m 0.063 | **FAIL** |
| INFO | Spearman(sep, C4) over the seeds; over the rungs | — | +0.4 (n = 4); −1.0 (e50, e60, e70) | INFO |

Verdict: **REFUTED**, on (b) in s1, (d.c4) in s1 and (d.order).

INFO path ratio at the other fresh nets: s12m 1.385, s13m 1.132, e50 1.239, e60 1.378, e70 1.022. Only 3 of the 7 fresh nets reach 1.25; the discovery hub (1.659) was the extreme.

(a) and (c) held in both seeds, but a partly held item is not SUPPORTED.

### AH-7. How deep the collapse goes depends on training length — **REFUTED**

Fields: `pl.{layer2.8,penult}.neural_collapse.nc1` (c = log10 ratio), `pl.penult.class_centers.centers` (CV), `meta.accuracy.ref`.

| clause | predicted | discovery | observed | result |
|---|---|---|---|---|
| (a) reading | c ≥ 1.90 in both seeds → PREDICTED | h56 2.11 | s1 2.138, s2 2.032 | PREDICTED |
| (b) rungs | c ≤ 1.85 at e50, e60, e70, s12m, s13m | e40 1.69 | 1.734, 1.747, 1.718, 1.838, 1.790 | PASS |
| **(b) Spearman** | Spearman(E, c) ≥ 0.8 over e50, e60, e70, s1 (200), s2 (200) | ladder rose e10 < e20 < e40 < hub | **0.667** (the c ranks of e50, e60, e70 are 2, 3, 1) | **FAIL** |
| (c) seeds | CV_all < 0.056 and CV_nn ≤ 0.035 in s1 and s2 | h56 0.037 / 0.014 | s1 0.0428 / 0.020; s2 0.0366 / 0.018 | PASS |
| **(c) rungs** | CV_all ≥ 0.056 at e50, e60, e70 | e40 0.082 | e50 0.0632, e60 0.0614, **e70 0.0524** (live B20+ edge 0.0562: same outcome) | **FAIL (e70)** |

Verdict: **REFUTED**, on (b) Spearman and (c) at e70. Every rung has `acc_ref` ≥ 0.9981, so no misfit caveat applies.

What the data do show:
- c does not rise measurably from 50 to 70 epochs. The rung spread is 0.029, well inside the seed-to-seed spread at equal E (0.09 at E = 60, 0.11 at E = 200).
- There is a step between 70 and 200 epochs.
- At 70 epochs the class centers are already inside the equidistance edge. s13m (60 epochs, CV_all 0.0535) is also inside it, though it is not scored.

### AH-8. TwoNN ID follows the spectrum, except at a fully trained penult — **SUPPORTED**; AH-8d — **SUPPORTED**

Fields: `pl.<L>.twonn_id.id` and `pl.<L>.pca_spectrum.{dim, eig_top20, top_eig_frac}` → ID_gauss (5 draws, seeds 100-104). R20-band maximum penult ID_gauss is 10.698.

| clause | predicted | discovery | observed | result |
|---|---|---|---|---|
| (a) | Pearson(id, ID_gauss) over the 10 taps ≥ 0.95 | 0.985-0.997 | s1 0.9875, s2 0.9878; s12m 0.984, s13m 0.991, e50 0.985, e60 0.987, e70 0.993; INFO replays 0.982-0.995 | PASS |
| (b) | ρ(layer3.5) ≤ 0.80 | h56 0.727 | s1 0.721, s2 0.743 | PASS |
| (c) | ρ(penult) ≥ 0.98, and ID_gauss(penult) ≤ 10.698 + 0.05 | h56 1.021 / 10.45 | s1 0.9915 / 10.471; s2 0.9954 / 10.508 | PASS |
| (e) twin | \|Δρ(penult)\| ≤ 0.03 (h56 → tw) | r20 0.0002 | 0.0076 | PASS |
| **AH-8d** | penult ID_gauss at M11 and at each evaluable Mc ≥ 10.698 + 0.30 = 10.998; ladder "matched" | e40 12.09 | M11 = e60 11.532; M12 = s12m 11.492; M13 = s13m 11.703 | PASS |

Verdicts: **AH-8 SUPPORTED**; **AH-8d SUPPORTED** (`ev` `.ladder.status` "matched").

INFO: ρ(penult) at the 50-70-epoch nets is 0.917-0.946. So ρ approaches 1 only at the fully trained (200-epoch) penult, and there it is 0.99-1.00, not above 1.

### AX-1. Class-orthogonal drift at the pre-collapse tap — gate **DROPPED** (ruled out)

Gate (`pr:resnet20_s0hub_st3` and `pr:resnet56_s0hub_st3` `.AX1`):

| gate clause | predicted | r20 hub | r56 hub |
|---|---|---|---|
| AUC(T_perp) for motion s1, defocus s3, fog s3, contrast s3 ≥ 0.90; brightness s5 ≥ 0.75 (`auc_perp_pre`) | as stated | all 1.000 | all 1.000 |
| single-class FPR, T_perp ≤ 0.10 (`single_class.perp_pre.fpr_mean`) | ≤ 0.10 | **0.104** | 0.074 |
| single-class FPR, T_full at penult ≥ 0.50 | ≥ 0.50 | 1.0 | 1.0 |
| c3: AUC(pre) − AUC(penult) ≥ +0.10 for motion s1 and brightness s3 | ≥ +0.10 | **+0.068**, +0.303 | **+0.009**, +0.217 |

- **Gate: DROPPED**, because the r20 hub fails a gate clause. c3 motion alone decides it, whatever the marginal 0.104.
- **Recorded as ruled out.**
- INFO values on s1 and s2 are not evidence:
  - c1 PASS on both;
  - c2 s1 0.048, s2 **0.126**;
  - c3 s1 +0.020 / +0.204, s2 +0.001 / +0.199;
  - c4 0.440 / 0.547;
  - the double holdout (zoom and glass s3) 1.0 / 1.0 is INFO.
- Had the gate opened, AX-1 would have been REFUTED on c3 in both seeds.

### AX-2a. Early-tap drift profile — gate **CLOSED** (stays a proposal)

Gate (`.AX2.auc`, batch 16; argmax set = taps within 0.005 of the maximum):
- **r20 hub:** passes.
  - (a) brightness s1 is {l10}.
  - (b) defocus s1 is {s1end}; motion s1 is {s1end, s2end}.
  - (c) penult is in no argmax set (8 of 8 corruptions scored).
- **r56 hub:** fails (a).
  - `pr:resnet56_s0hub_st3` `.AX2.auc["corrupt__brightness__s1"]` gives stem 0.6719, l10 0.6690, s1end 0.6686.
  - s1end is inside the tie band, so the set is not contained in {stem, l10}.
  - (b) and (c) pass.
- **Gate: CLOSED** (the r20 hub passes, the r56 hub fails). AX-2a stays a proposal.
- The INFO s1/s2 values all pass: (a) {l10} on s1 and {stem} on s2. They are not evidence. Under section 6, s1/s2 are now spent for this axis, so a revised version needs new seeds.

### AX-2b. Direction-template family router — gate **OPEN** → **SUPPORTED**; double holdout **SUPPORTED**

| clause | predicted | observed (`.AX2.router`) | result |
|---|---|---|---|
| gate (both hubs) | discovery s3/s5 routing ≥ 0.90 per family | N = B = L = P = 1.000 on both | OPEN |
| (d) confirmation, s1 and s2 | ≥ 0.85 per family | N = B = L = P = 1.000 on both (`.accuracy_by_family`; all 16 split-by-family entries of `.by_split` are 1.000) | PASS |
| (e) double holdout, s1 and s2 | impulse s3/s5 → N ≥ 0.90; zoom s3/s5 → B ≥ 0.80 | impulse → N 1.000 / 1.000; zoom → B 1.000 / 1.000 (`.holdout[...].assigned`) | PASS |
| INFO holdouts | glass, frost, elastic | glass s3 → N 0.998 (s1) / 1.0 (s2), glass s5 → N 1.0; frost → N 1.0; elastic → B 1.0 | INFO |

Verdict: **SUPPORTED**; double holdout **SUPPORTED**. It is near-trivial; see rule 8.

### AX-3. Per-batch harm grade (batch 256) — gate **OPEN** → **SUPPORTED**

| gate instance | pooled Spearman(h, loss) ≥ 0.85 | in-band (h ≤ 0.25) batches losing > 10 pt ≤ 5% |
|---|---|---|
| r20 hub | 0.9644 | 59/1906 = 0.031 |
| r56 hub | 0.9711 | 3/1606 = 0.0019 |
| e40 | 0.9691 | 31/1922 = 0.0161 |

| clause | predicted | s1 | s2 | result |
|---|---|---|---|---|
| (a) | pooled Spearman ≥ 0.85 | 0.9732 | 0.9720 | PASS |
| (b) | in-band batches losing > 10 pt ≤ 5% | 1/1551 | 1/1549 | PASS |
| (c) | ≥ 90% of brightness batches (s1, s3, s5 pooled) with h ≤ 0.30 | 0.965 | 0.918 | PASS |
| (d) reading | ≥ 10% of batches with e ≥ 0.95 lose > 10 pt | 711/3354 = 0.212 | 1266/3809 = 0.332 | the energy grade fails, as predicted |
| INFO | ≥ 90% of ood__cifar100 batches with h > 0.25 | 1.0 | 1.0 | INFO |
| INFO batch 64 | — | in-band 0.031, ρ 0.926 | 0.016, 0.928 | INFO |

Paths: `pr:resnet56_s{1,2}` `.AX3.splits.<S>.{h, loss, e}`, `.AX3.ood`, `.AX3.info_b64`.

Verdict: **SUPPORTED**. See rule 8: (a) is a between-split quantity, and (c) depends on pooling.

### AX-4. Per-sample off-simplex residual e_perp — gate **DROPPED** (ruled out)

Gains are AUROC(e_perp) − AUROC(d1) (`.AX4.auroc`):
- r56 hub: defocus s3 −0.180, motion s3 −0.413.
- r20 hub (INFO): −0.162 and −0.339.

All four are below +0.02, so the gate is **DROPPED** and AX-4 is ruled out.

INFO values on s1 and s2:
- gains −0.182 / −0.428 (s1) and −0.196 / −0.441 (s2);
- e_perp − dens2 from −0.20 to −0.52;
- zoom s3 e_perp 0.32 against d1 0.72.

## What was refuted, and why

| item | failed clause | kind | reading |
|---|---|---|---|
| AH-3 | (d) SE ratio at penult, s1 1.109 against ≥ 1.25 | **genuine** (premise) | The ratio is graded across fresh nets: rungs and Mc 0.99-1.08, s1 1.11, s2 1.25. The 1.25 bound was calibrated on one discovery instance (h56 1.539) against r20 0.989. A4b found float32 saturation nearly absent in s1, s2 and the hub (`top_share_correct` 0.0002). So the "saturation signature" premise was false even for h56. The mechanism clauses (a), (b), (c) and (e) held in both seeds, but they do not rescue the item |
| AH-5 | (a) stem PC1 at e50 / e60 | **genuine** | In two trained nets about half of the brightness shift loads on stem PC1. Across fresh nets its PC1 share runs from 0.00 to 0.57, although it always stays within the top-3 stem PCs. "Brightness avoids stem PC1 in every trained net" is false |
| AH-5 | (c) enrichment ≥ 0.80, s1 0.789 / s2 0.777 (snow, 1 of 30 splits) | threshold tightness | The bound sat 0.016 below the single discovery value (h56 0.816) |
| AH-5 | (d) penult excess, s1 0.102 against ≤ 0.10 | noise-level | Unpaired difference with SE ≈ 0.012 (see rule 8). It still counts |
| AH-6 | (b) path ratio, s1 1.156 | **genuine** | Across 7 fresh nets the ratio runs 1.02-1.39, and only 3 reach 1.25. The discovery hub (1.659) was the extreme |
| AH-6 | (d.c4) s1 0.1125; (d.order) 0.1125 < 0.1355 (s13m) | **genuine** | C4 does not follow collapse across independent seeds: the seed Spearman is +0.4, and across the rungs it is −1.0. This is the verifier's H-T1 correction, and the independent-seed test refutes it |
| AH-7 | (b) Spearman(E, c) 0.667 | genuine for 50→70 | The rung differences (0.029) are far below the seed spread at equal E. A random rung order would pass only about half the time. c has no measurable 50-70-epoch trend, only a step up to 200 epochs |
| AH-7 | (c) e70 CV_all 0.0524 against ≥ 0.056 | threshold transfer | The resnet20-band edge does not separate 50-70-epoch from 200-epoch resnet56: e70 and s13m (0.0535, unscored) fall inside it |

Every refuted item is recorded as **tested, not replicating**, with the failed clauses above (section 7.6). None may be re-thresholded after the fact: a bound moved to 1.10, 0.77 or 0.052 would be discovery for a new pre-registration on new seeds.

## Rule 8 (artifact hypotheses before interpreting)

**Too-good or trivially true numbers on SUPPORTED items.** These change no label, but they bound what each label means.

- **AH-1(b) is passed by the random nulls:** the maximum pre-penult sparse fraction is 0.055 (r56 rand) and 0.064 (r20 rand).
  - The held-out sparse fraction at non-collapsed taps is about 0.05 by construction (`pl.<L>.knn_density.ref_sparse_frac_self` = 0.05).
  - So (b) is near-arithmetic; the content of AH-1 is (a) and (c).
- **AH-1(d) and AH-8(e)** are noise clauses on re-measured hub weights (h56 → tw), not fresh evidence.
- **AH-2(a) is at or above 0.99** (s13m 0.992).
  - Artifact hypothesis: the alignment of H with harm is built in, because the penult is the classifier input (H-D9).
  - Test: other penult readings track cost as well or better. Per-sample displacement magnitude gives 0.9996 / 0.995, and penult sparse_frac 0.994 / 0.992. H at the stem gives 0.19 / 0.26, severity alone 0.72 / 0.70, and the random-init null −0.40.
  - Conclusion: (a) shows that H is a harm-aligned penult reading, not that H is special. The discriminating content is (b), the HOLD band, and (c), brightness.
- **AH-8(a) is passed by the random nulls** (Pearson 0.998 / 0.997), because the range of ID across taps carries the correlation.
- **AH-8(c) in s1 is thin:** 0.9915 against 0.98.
  - The Monte-Carlo SE of ρ is about 0.004, and the TwoNN bootstrap adds about 0.007, so the margin is about 1.4 SE.
  - With alternative sampler seeds 200-204, ρ(penult) is 0.9848 (s1) and 0.9896 (s2): still passing.
- **AX-2b is at 1.000 on every trained net** (9 of 9 runs, discovery and confirmation), so the clause has no dynamic range. Three artifact hypotheses were tested:
  1. Template/test overlap: rejected. Templates use corrupt rows 0-999 at s3, routed batches use rows 1000-1999 at s3 and s5, and whitening uses half A.
  2. The task is easy for any network: largely confirmed. The random-init nulls route N at 0.99-1.00, B at 0.987-0.996 and L at 0.993-1.00; only P separates trained nets (1.0) from random ones (0.50 / 0.60). Templates are also refit on each network, so the confirmation tests the procedure, not a transferred router.
  3. The families are semantic: partly refuted. The INFO holdouts route glass blur and frost to N (0.998-1.0), so "N" behaves as a high-frequency-change template, not a noise template. The nulls never read holdouts, so the holdout pass has no null comparison.
- **AX-3(a) is a between-split quantity.**
  - Replacing each batch's h by its split mean gives 0.9775 / 0.9750. The mean within-split Spearman is only 0.23 / 0.21: the 200 batches per split are drawn from the same 2000 rows.
  - The random nulls fail (a) (−0.18 / −0.14), so a null does not pass it.
  - (b) is diluted by near-zero-cost splits but holds without them: restricted to splits with mean loss ≥ 5 pt it is 1/221 (s1) and 1/84 (s2). The same restriction would have failed on discovery: r20 hub 59/516 = 11.4%, e40 31/379 = 8.2%.
  - (c) passes only pooled. At brightness s5 alone, 0.895 (s1) and 0.755 (s2) of batches have h ≤ 0.30; s1 and s3 are 1.000.
  - No leakage was found. Corrupt rows are paired with clean rows 0-1999, the probe raises on a label mismatch, and clean half-B batches have h ≤ 0.17.

**Measurement notes on REFUTED items and on the text.**
- `atlas/invariants/density.py` (`n_query` 3000) scores test `sparse_frac` and the median shift on a random 3000-of-5000 test subsample. The corrupt splits use their 2000 paired rows.
  - So AH-2's H and AH-5(d)'s excess are unpaired differences, with SE about 0.012 for the excess.
  - AH-5(d)'s 0.002 overshoot in s1 is about 0.17 SE. It counts under the text.
- The random nulls pass AH-6(c) more strongly than any trained net. It tests only "no saturation of the noise translation at depth 56", as the text says.
- AX-1:
  - c1 is at ceiling; the r20 null reaches 1.0 on fog, contrast and brightness, and the r56 null 0.86 / 0.98 / 1.0.
  - Class 4's single-class FPR (`.AX1.single_class.perp_pre.fpr[4]`) is 0.26-0.36 in the gate and confirmation runs (r20 hub 0.32, r56 hub 0.26, s1 0.26, s2 0.36); across all 11 records it ranges 0.00-0.44 (e40 0.04). This is the test-mean leak the text warned about. s2's c2 of 0.126 is driven by class 4 (0.36) and class 8 (0.30).
  - c3 fails by ceiling: at batch 64 the penult already sees motion s1 (AUC 0.93-0.999).
- AX-2a's gate turned on 0.0017 AUC around the frozen tie rule. Top-two brightness s1 gaps across the 11 runs are 0.003-0.024 at AUC 0.66-0.69, which is within sampling noise. The text wins, so the gate stays CLOSED.
- AX-4's frozen orientation ("larger = more corrupt-like") is reversed in practice. e_perp AUROC is below 0.5 on 30 of 30 splits for the r56 hub, s1 and s2, so a smaller share of a corrupted sample's offset from its predicted center lies off the class span (less off-simplex in relative, ratio terms), although corrupted samples are farther from the nearest center (d1 AUROC 0.61-0.74). The text forbids direction-free AUROCs, so the result stands. DROPPED is robust anyway: with the orientation flipped, the gains are still −0.03 to −0.10 on every trained run.

## Frozen evaluator vs frozen text

The text wins. **No gap changes any outcome.**

1. **AH-8d ladder check.** `m11Runs()` (`scripts/anomaly_eval.js:502-508`) adds every evaluable Mc member even when `ladder.status` is not "matched". Section 4 makes AH-8d NOT_EVALUABLE in that case. No effect: the status is "matched".
2. **AH-3 tie note.** The code reads A4b's per-seed `tie_call` [NO-LEAD, TIES-EXCLUDED]. A4b's SESSION.md says the text's call is joint (NO-LEAD). Neither version is TIE-ARTIFACT-POSSIBLE, so there is no effect.
3. **Sampling.** The density subsampling (see rule 8) is not stated in ANOMALY_H1. The AH-2 "check first" note mentions only the 5000- vs 2000-row accuracy offset. There is no outcome effect, but AH-5(d) sits inside the measurement noise.
4. **AX labels under a non-OPEN gate.**
   - The code prints the item as NOT_EVALUABLE ("gate DROPPED" / "gate CLOSED") and the double holdout as NOT_EVALUABLE.
   - The text (sections 3, 5 and 7.6) records the s1/s2 values as INFO, a DROPPED axis as ruled out, a CLOSED axis as a proposal, and the AX-4 holdout as INFO.
   - This file uses the text's labels.
5. **INFO the evaluator never prints.** The following are in probe.json but not printed:
   - the r20 s1-s4 context runs and the random-init nulls (section 5.0);
   - the AX-2b glass, frost and elastic routings;
   - the AX-3 batch-64 summary and the e² energy variant.

   They are quoted here from the probe records where they matter.
6. **AX-1 c4** is one-sided in the code (AUC(single > mixed) ≤ 0.65). The text's "does not separate" can be read two-sided. The values 0.440 and 0.547 satisfy both readings.
7. **Guard source.** `a4bGuard` scans only `ev` `.guard` for "R0 FAIL" and "instrument changed", which is how `scripts/a4b_eval.js:335-339` builds it. The guard is [], and R0 and the instrument gate hold independently.
8. **AH-7(a)** is called a "reading" in the text but decides REFUTED or NOT_EVALUABLE. The code models it as a primary clause with pass true / false / null, which is equivalent.
9. **Cosmetic.**
   - Section 9 estimates about 160 ID_gauss values. The evaluator computes 105: 10 resnet56 atlases × 10 taps, plus 5 R20-band penults.
   - `rho_mc.js` (the discovery sampler the port copies) is not in the repository. The ID_gauss definition is instead checked by an independent port from the text, which is bit-identical, and by the section 8.1 values, which it reproduces.
10. **Command deviation.** The preliminary run did not use the section 9 command: `--b1` was a non-existent scratch file, and `--cache` / `--json` went to scratch. The official `eval.json` is still to be written (see "Next actions").

## Holdout statement

- **Definitions and thresholds** were written on discovery material only (17427c6 atlases and the batch-2 anomaly analysis) and frozen at P_A, before any batch-3 file existed.
- **Confirmation seeds.**
  - resnet56 s1 and s2 were declared at 664bd25, trained at P_run and read once for claims: their atlases and margin rebuilds by the A4b evaluator and by this one, and their dumps by the probe once each (one record per dump, no `_r2`).
  - s12m and s13m are A4b's matched-leg replicates.
  - The seed-11 rungs e50-e70 are one training run.
- **Confirmation corruptions** (impulse_noise, glass_blur, zoom_blur, frost, elastic_transform):
  - No AH item reads them; the evaluator's AH code reads test and the 30 discovery splits only.
  - The probe read them only on s1 and s2 (43 splits = 33 + 10), at s3/s5. They were used only in the AX-1, AX-2b and AX-4 double-holdout clauses and the AX-2b INFO routings.
  - Every discovery and null record read 33 splits and 0 holdout splits.
  - They were never used for a template, threshold or definition.
- **Spent material.**
  - s1 and s2 are now spent for every AX axis, whatever its gate (section 6). No revised AX-1, AX-2a or AX-4 (another threshold, tap or statistic) can be confirmed on them.
  - The same holds for any revision of the refuted AH items.
  - resnet20 s1-s4, the hub, e10, e20 and e40 remain discovery only.
- **B1.** No ViT or ImageNet dump, and no `margin_b1_*`, `b1_gate` or `instrument_check_b1*` file, was read by this evaluation.
  - HEAD at evaluation time is `2ad4e9c`, a B1 results commit. Its contents were not read.
  - An untracked `results/margin_b1_vitb16/verdicts.json` in the working tree was not opened.

## AH-4

**DEFERRED to after B1.** AH-4 reads only B1's `b1_verdicts.js` output (`results/margin_b1_vitb16/verdicts.json`) and the runs named in its `info.runs_used`, after B1's own decision.
- B1 is being relaunched (`RELAUNCH_r2.md`), so AH-4 is not evaluated here.
- The preliminary evaluator line "AH-4: NOT_EVALUABLE (B1 verdicts unreadable)" is a consequence of the scratch `--b1` path. It is **not** AH-4's label.
- Its label (SUPPORTED / REFUTED / NOT_EVALUABLE, with the per-ViT readings (d)) will be appended to this file from the section 9 run once B1 is decided.
- AH-4's two CIFAR points (the E9 rebuilds of s1 and s2) are guarded by the A4b guard and i2. Both hold now.

## What ANOMALY_H1 does not license

- **No ATLAS_STATUS change.** A SUPPORTED label means that the frozen prediction held on fresh material, and nothing more. This file promotes nothing, and no row or tag in ATLAS_STATUS changes because of it (section 7).
- **No scope beyond depth-56 CIFAR-10 ResNets.** The one exception is AH-4's regime clause, which B1 tests.
- **No mechanism claim from the parts of a refuted item.** For example, AH-3 (a)-(c) and (e), AH-6 (a) and (c), and AH-7 (a) held, but a partly held item is not SUPPORTED. They may seed a new pre-registration on new seeds, nothing more.
- **No Gate-1 component.** AX-2b and AX-3 are SUPPORTED at depth 56, and their holdout is not REFUTED. Controller use additionally needs a stream (ramp) test and a transfer test on another backbone (section 7.5). Neither is part of batch 3.
- **No evidence from INFO values.** The AX-1, AX-2a and AX-4 s1/s2 values are INFO, including the fact that AX-2a's would all have passed under a strict argmax. The twin clauses and the replays are noise or INFO.
- **No post-hoc thresholds** for any refuted clause.

**What a later promotion would need** (section 7). It would be a separate, owner-reviewed commit, and the candidates are AH-1, AH-2, AH-8, AH-8d, AX-2b and AX-3:

1. **The official verdict.** SUPPORTED from the section 9 command, run with `--b1` after B1, with item provenance PASS and no guard; for an AX item, the gate OPEN. There are no secondary exceptions to explain.
2. **This SESSION.md.** It must list every clause as predicted / observed / verdict, state the rule-8 hypotheses before any too-good number, and quote discovery values as discovery. That is done above, and the official `eval.json` must reproduce it.
3. **A4b's gates for the runs read.** All hold now:
   - G0b PASS for every atlas; `ev` `.G0c` IDENTICAL; `.G0e.status` PASS;
   - `.T56.fail` false, which is needed for AH-1 and AH-8 (twin clauses);
   - `.critic_pair` `synthetic_refusal` true, `holdout_hygiene` true, `input_norm` PASS.
4. **Scope wording.** A new row numbered 12 or higher, scoped "d56, CIFAR-10", with ✅ only at that scope. The row text must carry the rule-8 limits:
   - AH-2(a) is by construction;
   - AH-1(b) and AH-8(a) are null-passable;
   - AX-2b's N/B/L routing is not learned-specific, and "N" is a high-frequency template;
   - AX-3 is split-level, and its brightness HOLD is weaker at s5.
5. **For AX-2b and AX-3 as controller parts:** the stream (ramp) test and a transfer test on another backbone.

## Next actions

1. **After B1 is decided** (relaunch r2 and `scripts/b1_verdicts.js`), run the section 9 command once:
   `node scripts/anomaly_eval.js --p d257d91 --p-run d257d91 --a4b results/atlas_v1_resnet56_s1/a4b_eval.json --b1 results/margin_b1_vitb16/verdicts.json --cache results/anomaly_h1/idgauss_cache.json --json results/anomaly_h1/eval.json`
   - Check that AH-1..AH-8d and AX-1..AX-4 reproduce this file exactly.
   - Append AH-4 (clauses (a)-(c), the (d) readings per ViT, and the gate-G / c\* branch) to this file.
   - The ID_gauss cache can be recomputed (about 105 values), or seeded from the scratch cache that the independent port matched bit for bit.
2. **Commit** `results/anomaly_h1/` (this file, `eval.json`, the cache).
3. **Owner review.** Any ATLAS_STATUS row for AH-1, AH-2, AH-8, AH-8d, AX-2b or AX-3 is a separate commit, under the conditions above.
4. **Re-registration candidates** (discovery now; new confirmation seeds needed, never s1/s2):
   - AH-3 without the SE-ratio clause, or with a non-saturation mechanism for it;
   - AH-5 with a stem clause that allows a varying PC1 share, a paired density excess, and an enrichment band set from more than one instance;
   - AH-6 without (b) as a ratio bound, and C4 dropped as a collapse tracker;
   - AH-7 with a length ladder that spans 70-200 epochs;
   - AX-2a with a tie rule based on sampling SE;
   - AX-1 at a severity below the penult ceiling, with the test-class leak addressed.
5. **Before any controller use of AX-2b or AX-3:** a stream (ramp) test, and a transfer test on another backbone. For AX-3, also a within-split test (batch-level resolution inside one corruption type). For AX-2b, a null-contrasted holdout.
