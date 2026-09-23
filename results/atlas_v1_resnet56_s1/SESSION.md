# SESSION: Stage 2b (A4b) — depth-56 replication and the matched rung

This is the Evaluator pass on the pulled A4b results. The dumps stay on the RunPod volume.

- **Pre-registration.**
  - `docs/plans/STAGE2B.md`: design, bands, outcome tables and promotion rules. It is `docs/plans/STAGE2.md` "Amendment 1".
  - `experiments/queue/atlas_v1_resnet56_s1.yaml` notes: gates G0a-G0e and R0, ladder L1-L3, D-ID / D-COLL / D-ACC, K0-K1, C1-C7, T56, P1-P4, X and rule 8.
  - `experiments/queue/margin_v1_resnet56_s1.yaml` notes: M56-a, M56-b, M56-c.
  - The evaluation is frozen as `scripts/a4b_eval.js`.
  - Committed as **P = `f1c3c43`** at 2026-09-23T06:02:21-07:00 (13:02:21 UTC), together with B1.
- **What the pod ran.**
  - **P_run = `d257d91`** (P_A, 13:32:32 UTC), a child of P. It adds the ANOMALY_H1 pre-registration: `docs/plans/ANOMALY_H1.md`, `scripts/anomaly_{eval.js,probe.py}`, `tests/test_anomaly_probe.py`, `--anomaly` in `pod_atlas.sh` and an operational edit to `RUN_REQUEST.md`.
  - Pod `nc2mzssur3bb0k` (RTX 4090) ran `pod_atlas.sh --a4b --b1 --anomaly`. `block_a4b` finished OK at 2026-09-23 14:30:44 UTC.
  - The A4b results were pulled and committed as `81f77ac` before this evaluation (STAGE2B / integration D17 item 5).
- **Not evaluated here.** B1 (row 10, `docs/plans/B1_VIT_MARGIN.md`) and ANOMALY_H1 (`docs/plans/ANOMALY_H1.md`) are evaluated separately, in their own SESSION.md files.
  - Their blocks were still running on the pod during this evaluation.
  - No `margin_b1_*`, `b1_gate`, `instrument_check_b1*` or `anomaly_*` file was read. None was present locally, so `a4b_eval.json` `.M56.c_confidence.b1_e9_info` is `[null, null]`.
  - A4b and B1 are decided independently (STAGE2B "Local evaluation"); neither decision waits for the other.
- **How it was checked.**
  - The frozen evaluator was run as `node scripts/a4b_eval.js --p f1c3c43 --p-run d257d91 --json results/atlas_v1_resnet56_s1/a4b_eval.json`.
    - A second run reproduces the JSON byte for byte.
    - `bash scripts/a4b_eval_fixture.sh` passes all 8 scenarios.
    - `scripts/a4b_eval.js` is unchanged from P to HEAD.
  - Three evaluator families recomputed every item from the raw files with their own node scripts: (1) gates, ladder, twin, KILL-56 and the core items; (2) depth labels; (3) claims, tags and margin. `matched_rung.py` was re-implemented in node, since this machine has no Python.
  - A decider pass then recomputed every number a decision rests on, again from the raw files. That covered:
    - the provenance of all 28 A4b atlases;
    - R0, the ladder and Mc;
    - the B20+ bands and the four D fields and three [ACC] fields in all 10 resnet56 atlases;
    - C1-C5a in all 15 atlases;
    - the critic statuses of `c12`, `cT` and `cM`;
    - D1, adjacency, aligned layer3.1 and cka over P20, X, XM, W56, the hub pairs and the twin;
    - the CV band;
    - M56-a/b/c and the tie masses.
  - **No numeric or label discrepancy was found anywhere.**
  - The frozen code falls short of the frozen text in several places. None of them changes an outcome, because the raw files pass the text's version of every check (see "Frozen evaluator vs frozen text").
  - Nothing in the repository was modified by the evaluation. The scratch scripts are not committed.

**Outcome: VALID.**
- **Gates.** Every gate passes:
  - G0a-G0e;
  - R0;
  - the instrument gate;
  - the D6 margin preflight.
  - G0c is IDENTICAL, so Stage 1/1b/2 values may be quoted beside A4b values.
- **No-verdict guards.** KILL-56 is not triggered, T56 passes 6/6 core items and nothing blocks promotion.
- **Row 11.**
  - **D-ID is CONFIRMED, so ✅.**
  - **D-COLL is FIT-SENSITIVE, so 🟡:** nc1 is CONFIRMED; sep_ratio and the bridge ratio are FIT-SENSITIVE.
  - Both outcomes are as predicted.
- **Depth-56 tags.**
  - Rows 1, 3, 4, 5 and 6a: **d56 ✅ (s1, s2)**.
  - Row 7: **d56 🟡** (one pair).
  - Row 9: **d56 🟡 (M2, M3, M4)**, with M2 failing in s2 and the failure NOT-DEPTH.
  - Rows 2 and 8 stay 🟡; S9 is INFO in row 8.
- **SHAPE.** Rows 1, 3, 4, 5, 6a and 8 and S9 are SCALE-ROBUST. Row 7 is COLLAPSE-ARTIFACT, a diagnosis with no scale-robust statement.
- **Pair levels.**
  - D1: DEPTH-56-UNSTABLE (COLLAPSE), NOT-DEPTH at matched accuracy.
  - Adjacency: DEPTH-56-UNSTABLE.
  - cka_test: DEPTH-GAP.
  - Aligned layer3.1: DEPTH-GAP.
- **Rule 8.** The cka_test flag is raised. The artifact hypothesis is collapse-driven agreement (see Rule 8).
- **Tally.** 31 scored forecasts: 26 hit, 1 partial (P3), 4 missed (the T56 decod MAD, P4, the M56-b tag, the M56-c lead).
  - What was scored: L1-L3, D-ID, D-COLL, D-ACC, K0, K1, C1-C5a, C7, the T56 core items, the six T56 forecasts, P1-P4, X (counted once), M56-a, the M56-b tag, the M56-b attribution, M56-c E1 and the M56-c lead.
  - The gates and the rule-8 reference maxima are recorded but not scored.
- **Fragility.** Two outcomes sit within estimator noise; the rules apply as written:
  - the D-ID ✅: s1 is only 0.068 above the band edge, against a twin |Δ| of 0.113;
  - the D-COLL FIT-SENSITIVE trigger: e50 is inside the band edge by less than the twin |Δ|.

Path shorthands:

| shorthand | path |
|---|---|
| `s1`, `s2` | `results/atlas_v1_resnet56_s{1,2}/atlas.json` (seeds 1, 2; 200 epochs, last epoch) |
| `e50`, `e60`, `e70` | `results/atlas_v1_resnet56_e{50,60,70}/atlas.json` (seed 11) |
| `m12`, `m13` | `results/atlas_v1_resnet56_s1{2,3}m/atlas.json` (seeds 12, 13; 60 epochs) |
| `h56`, `tw` | `results/atlas_v1_resnet56_s0hub_st3/atlas.json`, `results/atlas_v1_resnet56_s0hub_ref1/atlas.json` |
| `e40s` | `results/atlas_v1_resnet56_e40_st3/atlas.json` |
| `B` | `results/atlas_v1_resnet20_{s0hub,s1,s2,s3,s4}_st3/atlas.json` (band B20+; values in that order) |
| `t<x>` | `results/train_resnet56_<x>/train.json` |
| `ic` | `results/instrument_check_a4b/` |
| `ev` | `results/atlas_v1_resnet56_s1/a4b_eval.json` |
| `c12` | `results/critic_v1_resnet56_s1_s2/critic.json` (decision pair) |
| `c012` | `results/critic_v1_resnet56_s0_s1_s2/critic.json` ([0,1] = hub-s1, [0,2] = hub-s2) |
| `cT` | `results/critic_v1_resnet56_hub_noise/critic.json` (twin) |
| `cM` | `results/critic_margin_v1_resnet56_s1_s2/critic.json` |
| `cB`, `cX`, `cXM` | `results/critic_v1_resnet20_st3_band`, `critic_v1_scale3_r20_r56seeds`, `critic_v1_scale3_r20_r56matched` (`critic.json`) |
| `mg<x>` | `results/margin_v1_resnet56_<x>/atlas.json` `.per_layer.penult.margin_typeb` |
| `d(a→b)` | `results/<b>/compare_vs_<a>/deformation.json` |

Per-layer fields are `.per_layer.<L>.<invariant>.<key>`. X = the 10 pairs resnet20 `_st3` × {s1, s2}. P20 = the 10 pairs of B20+, and e20(m) = min over P20. W56 = s1-s2. XM = resnet20 `_st3` × {e60, s12m, s13m} (15 pairs).

## Run facts

**Runs.** 15 atlases were built:
- the five resnet20 `_st3` band atlases;
- h56 and tw;
- e40s;
- e50, e60 and e70;
- m12 and m13;
- s1 and s2.

13 margin rebuilds were made: `margin_v1_resnet20_*_st3` ×5 and `margin_v1_resnet56_{s0hub_st3, s1, s2, e50, e60, e70, s12m, s13m}`.

e90 was not trained, correctly: `ic/ladder_first.json` `.status` is "matched", not "below".

**Accuracies** (`t<x>` `.final_test_acc_10k`; these are the matching keys; target `results/norm_check_resnet20/norm_check.json` `.chenyaofo.acc_10k` = 0.9259):

| run | seed, epochs | acc_10k | \|acc − 0.9259\| | `.meta.accuracy.test` (5k) | `.meta.accuracy.ref` (fit) |
|---|---|---|---|---|---|
| e40 (Stage 2 checkpoint) | 11, 40 | 0.9192 | 0.0067 (out) | 0.9160 | 0.9936 |
| e50 | 11, 50 | 0.9220 | 0.0039 | 0.9194 | 0.9981 |
| **e60 (M11)** | 11, 60 | **0.9256** | **0.0003** | 0.9244 | 0.9986 |
| e70 | 11, 70 | 0.9253 | 0.0006 | 0.9248 | 0.9991 |
| s12m (M12) | 12, 60 | 0.9249 | 0.0010 | 0.9204 | 0.9983 |
| s13m (M13) | 13, 60 | 0.9277 | 0.0018 | 0.9242 | 0.9989 |
| s1 | 1, 200 | 0.9442 | — | 0.9438 | 1.0 |
| s2 | 2, 200 | 0.9404 | — | 0.9404 | 1.0 |
| hub (h56) | 0, best epoch | 0.9438 (`norm_check_resnet56`) | — | 0.9446 | 1.0 |
| B20+ | — | — | — | 0.9256 / 0.9226 / 0.9250 / 0.9222 / 0.9286 | 0.9996 / 0.9994 / 0.9997 / 0.9995 / 0.9996 |

**Reading hazard.** s12m's 5k `.meta.accuracy.test` (0.9204) would lie outside the window. The pre-registered key is the 10k `final_test_acc_10k` (0.9249, inside), and that is what is used.

**Training.**
- `.recipe` is cosine with T_max = E, the last epoch is kept, and there are 8 workers.
- `.torch` is 2.8.0+cu128 and `.device` is NVIDIA GeForce RTX 4090 in all 7 new train.json files.
- `.wall_s`:

  | run | wall_s | s per epoch |
  |---|---|---|
  | e50 | 192.4 | 3.85 |
  | e60 | 227.9 | 3.80 |
  | e70 | 274.0 | 3.91 |
  | s12m | 228.1 | 3.80 |
  | s13m | 228.1 | 3.80 |
  | s1 | 777.0 | 3.88 |
  | s2 | 755.2 | 3.78 |

- The seven trainings total 44.7 min, at 3.78-3.91 s per epoch against STAGE2B's estimate of 3.45-3.52 s (INFO). The likely cause is that the CPU lane's extractions shared the GPU.

**Weights** (`.meta.weights` sha256 prefixes):

| run | sha256 |
|---|---|
| s1 | 73c9458c60b66949 |
| s2 | 14a651f000ca2d55 |
| s12m | 71db3bad0a2c41c2 |
| s13m | 0df3f708ebfa3292 |
| e60 | d8f65a2fe8d7a833 |

All are distinct from each other and from every rung (`ic/i2.json`).

**Critic counts** (`.counts`; the verdict word is PARTIAL in all seven and is advisory):

| critic | PASS | FAIL | WARN | INFO |
|---|---|---|---|---|
| c12 | 119 | 25 | 1 | 1 |
| c012 | 197 | 29 | 1 | 3 |
| cT | 140 | 4 | 1 | 1 |
| cB | 483 | 27 | 1 | 10 |
| cX | 870 | 84 | 1 | 21 |
| cXM | 2136 | 174 | 13 | 43 |
| cM | 52 | 2 | 6 | 0 |

- cT's 4 FAILs are all non-core: stem, layer1.0 and layer1.5 nc1, and commit/blockiness.
- cM's 2 FAILs are non-penult: layer3.0 and layer3.5 auc_dist_typeb.

## Gates

**G0a — PASS.** The evaluator does not check it (see "Frozen evaluator vs frozen text"); it was recomputed.
- All 10 resnet56 atlases have `.layers` = `.meta.layers` = [stem, layer1.0, layer1.5, layer1.8, layer2.0, layer2.5, layer2.8, layer3.0, layer3.5, layer3.8, penult], and their `.per_layer` keys match.
- All five `_st3` atlases have the 11 resnet20 taps.
- All 40 resnet20 → resnet56 deformation.json files have `.layers[i]` = [a.layers[i], b.layers[i]].

**G0b — PASS.**
- **Atlases.** All 28 A4b atlases (15 atlas, 13 margin) have `.skipped` = {} and `.source` "real", and a recursive search finds no `error` or `errors` key.
- **i2.** `ic/i2.json` `.status` is PASS. Recomputed: s1, s2, s12m and s13m have `.meta` layers, dims, splits, n_test, ref/panel indices, norm, arch, dtype and pooling equal to h56's, and new sha256 values.
- **Critics.** input_norm (`.checks.norm_consistency[0]`) is PASS in all 7 A4b critics (`ev` `.G0b.critic_input_norm`).
- **cka_test.** `.per_layer.penult.cka_test` is numeric in all 64 A4b deformation.json files.

**G0c (recorded) — IDENTICAL.**
- **Leaf replay.** `ev` `.G0c.replay`: 13 of 13 are IDENTICAL, with 0 differing leaves.
  - Atlases: 38050-38075 leaves each (`_st3` ×5, h56, e40s).
  - Margins: 12368 leaves each (`margin_v1_*_st3` ×6).
  - These are real re-extractions: their `.meta.created` is 13:37-14:01 UTC, against originals from 02:10-05:59 at 664bd25 or 6242a17.
- **Same-space compares.** Penult cka_test is 1.0 in all 7 (`ev` `.G0c.same_space_cka`).
- **Stage B rebuild.** `ic/check.json` `.status` is PASS with `.bitwise_identical` true: 38061 of 38061 leaves exact, `.git_commit` d257d91.
- **Code diff.** `ic/code_diff.txt` lists only `atlas/invariants/margin.py` and `requirements.txt`.
- Stage 1/1b/2 values may therefore be quoted beside A4b values. The hub's Stage 2 numbers (drop 0.450; C2 0.757 / 0.150 / 0.076; C4 margin 0.192; C5a margin 0.085) are h56's.

**Instrument gate — OK.**
- `ic/check.json` is PASS under the leaf rule.
- C1-C5a hold in all five `_st3` atlases:
  - peak layer3.1, drop 0.488-0.503;
  - C2 holds;
  - class commit at layer3.1;
  - C4 6/6 (min margin 0.105);
  - C5a 8/8 (min 0.0395).

**G0d and D7 — PASS.**
- **Versions.** `ic/versions.json` has python 3.12.3, numpy 2.1.2, scipy 1.18.1 and sklearn 1.9.1, equal to `results/instrument_check_stage1b/check.json`, and torch 2.8.0+cu128 on an RTX 4090, equal to `t e40`.
- **Code.** No Stage A/B file other than `margin.py` changed since 664bd25 (`git diff --stat 664bd25 d257d91`). `requirements.txt` adds only B1's four pins.

**D6 margin preflight — PASS (bitwise).** `results/instrument_check_a4b_b1/margin_resnet{20,56}_s0hub/check.json`:
- `.status` PASS and `.bitwise_identical` true;
- 12371 of 12371 leaves exact;
- `.git_commit` d257d91.

No environment drift needs recording.

**G0e — PASS** (`ev` `.G0e.status`, `.problems` []).
- `git merge-base --is-ancestor f1c3c43 d257d91` holds.
- `git diff --quiet f1c3c43 d257d91` holds over:
  - the frozen list: all 41 manifests added in P (30 A4b, 11 B1), STAGE2B/STAGE2/B1_VIT_MARGIN.md, `a4b_eval.js`, `b1_verdicts.js`, `b1_gate.py`, `tolerances_default.yaml` and `ATLAS_STATUS.md`;
  - all of `experiments/queue`.
- The same list, plus `atlas/` and `matched_rung.py`, is unchanged from d257d91 to `81f77ac`.
- In all 28 A4b atlases, `.meta.git_commit` and `provenance.json` `.stage_b_git_commit` are d257d91.
- `.meta.created` runs from 13:37:51 to 14:25:08 UTC. That is after P (13:02:21Z) and also after P_run's own commit (13:32:32Z), so the pod clock is sane.
- `block_a4b` in `pod_atlas.sh` is byte-identical from P to P_run. P_run adds only `--anomaly` and `block_anomaly`.
- All 15 `manifest_used.yaml` files carry the frozen instrument lists, the weights, reference seed 0 (1 for the twin), stride, the 18 factors and the holdout seeds.
- **Deviation from the text (no consequence).** STAGE2B says "Pre-launch fixes after P may touch code only". P_run also added `docs/plans/ANOMALY_H1.md` (a separate pre-registration) and edited `RUN_REQUEST.md` (launch flags and ANOMALY_H1 steps). Neither file is on the frozen list, and `block_a4b` did not change, so G0e passes as defined and no A4b input changed.

**R0 — PASS.** `t s1` / `t s2` `.final_test_acc_10k` = 0.9442 / 0.9404. Both are ≥ 0.935, and |Δ| = 0.0038 ≤ 0.010.

## Ladder and the matched value

**L1 — PASS.**
- e50, e60 and e70 are all within 0.005 of 0.9259; e40 (0.0067) is not.
- So step 1 applies, with no interpolation.

**L2 (forecast) — HIT.**
- Forecast ranges: e50 [0.920, 0.927], e60 [0.922, 0.930], e70 [0.924, 0.933]. Observed: 0.9220, 0.9256, 0.9253, all inside.
- Forecast M11 = e60. Observed: e60, the nearest in-window rung with an atlas (0.0003 vs 0.0006), with no tie.
- `ic/ladder.json`: `.status` matched, `.e_star` 60, `.in_window` [50, 60, 70]. `ladder_first.json` is identical. So M11 = E*, and the Mc are evaluable.
- INFO:
  - e50 and e70 fell 0.0002 and 0.0003 below STAGE2B's curve-model ranges (0.9222-0.9243 and 0.9256-0.9309).
  - e70 is slightly below e60.

**L3 — PASS.**
- s12m (0.9249) and s13m (0.9277) are both in the window: `.recipe.seed` 12 and 13, `.recipe.epochs` 60, atlases present.
- `ev` `.ladder.Mc[*].evaluable` is true for both, so CONFIRMED is reachable.

**W** = {e60, e70, e50}. No in-window rung lacks an atlas.

## Predictions (predicted → observed → verdict)

### D-ID and D-COLL (row 11; STAGE2B outcome table, first match wins)

**Band B20+** (`B` `.per_layer.penult`; min − w .. max + w):

| field (side) | B20+ values | band |
|---|---|---|
| `twonn_id.id` (HIGH) | 9.9150 / 9.7566 / 10.0355 / 9.8362 / 9.9013 | [9.4777, 10.3144] |
| `class_centers.sep_ratio` (HIGH) | 2.9962 / 3.0666 / 3.0212 / 2.9700 / 3.0242 | [2.8734, 3.1633] |
| bridge ratio (HIGH) | 3.3916 / 3.4243 / 3.3880 / 3.3577 / 3.3952 | [3.2912, 3.4908] |
| `neural_collapse.nc1` (LOW) | 0.1699 / 0.1630 / 0.1696 / 0.1720 / 0.1674 | [0.1541, 0.1809] |

The bridge ratio is the mean of the 45 distances between `.class_centers.centers` divided by the mean `.class_centers.radius`. Its B20+ values reproduce Stage 1.

**Field outcomes** (`.per_layer.penult`; H = HIGH, L = LOW, IN = inside):

| field | M11 e60 | M12 | M13 | e50 | e70 | s1 | s2 | outcome |
|---|---|---|---|---|---|---|---|---|
| id | 10.7663 H | 10.7213 H | 10.7269 H | 11.0292 H | 10.8658 H | 10.3820 H | 10.4593 H | **CONFIRMED** |
| sep_ratio | 3.3678 H | 3.3671 H | 3.3138 H | **3.1397 IN** | 3.5057 H | 5.1141 H | 5.2125 H | **FIT-SENSITIVE** |
| bridge | 3.6443 H | 3.6431 H | 3.5888 H | **3.4336 IN** | 3.7594 H | 5.4272 H | 5.4997 H | **FIT-SENSITIVE** |
| nc1 | 0.1286 L | 0.1288 L | 0.1296 L | 0.1449 L | 0.1174 L | 0.0529 L | 0.0502 L | **CONFIRMED** |

**D-ID**
- Predicted (notes, verbatim): "D-ID twonn_id.id HIGH. Predicted CONFIRMED".
- Observed: s1 and s2 are on the side, so it is not NOT-REPLICATED. M11 and both Mc are on the side, so it is neither NOT-DEPTH nor SEED-SENSITIVE. Every W rung is on the side, so it is not FIT-SENSITIVE. Two Mc are evaluable, so it is CONFIRMED.
- Verdict: **CONFIRMED (HIT)**.

**D-COLL**
- Predicted (notes, verbatim): "D-COLL class_centers.sep_ratio HIGH, bridge ratio HIGH, neural_collapse.nc1 LOW; claim = the weakest field. Predicted FIT-SENSITIVE".
- Observed: sep_ratio and bridge fall to FIT-SENSITIVE through the W rung e50; nc1 is CONFIRMED. The weakest field gives FIT-SENSITIVE.
- Verdict: **FIT-SENSITIVE (HIT)**.
- Every sub-forecast hit:
  - e50 is inside the band for sep_ratio (edge 3.16) and for bridge (3.49);
  - e60 and e70 are outside;
  - nc1 is CONFIRMED, with e50 0.1449 against the edge 0.1541 (forecast "~0.15 vs edge 0.154");
  - s1 and s2 are far outside: forecast hub 5.33 / 5.59 / 0.049, observed 5.11-5.21 / 5.43-5.50 / 0.050-0.053.
- `ev` `.D_ID` is CONFIRMED and `.D_COLL` is FIT-SENSITIVE. `.D[*].not_evaluable_because` is [] for all four fields.

**Reported beside the outcome** (INFO; STAGE2B "D-ID and D-COLL"):
- **Stage 2 style label** (M11 as m*, per full-recipe seed): DEPTH / DEPTH for all four fields. The evaluator does not print it; it was recomputed.
- **B20** (three members, INFO):
  - m*, s1 and s2 are on the side for all four fields (`ev` `.D[*].b20_3`).
  - For the window rung e50, which the evaluator does not report: sep_ratio 3.1397 is above the B20 edge 3.1370, so that field alone would be CONFIRMED. D-COLL stays FIT-SENSITIVE through bridge (3.4336 < 3.4606).
- **Fit covariate** (`.meta.accuracy.ref`): M11 0.9986; Mc 0.9983 / 0.9989; W 0.9981-0.9991; s1 / s2 1.0; B20+ 0.9994-0.9997.
  - The matched leg is less fit than resnet20, which biases L1 against D-COLL. nc1 is CONFIRMED regardless.
  - The full-recipe leg is more fit and more accurate, which biases L2 against D-ID. D-ID is CONFIRMED regardless.
- **T56 |Δ| next to (m* − band edge)** (`ev` `.D[*].info`):

  | field | T56 \|Δ\| | m* − edge | ratio |
  |---|---|---|---|
  | id | 0.1127 | 0.4519 | 4.0× |
  | sep_ratio | 0.0508 | 0.2045 | 4.0× |
  | bridge | 0.0662 | 0.1535 | 2.3× |
  | nc1 | 0.0014 | 0.0255 | 18× |

- **Two margins sit within estimator noise:**
  - **s1's ID** is 10.3820 against the edge 10.3144, a margin of 0.068. That is below the twin |Δ| (0.113) and below s1's own bootstrap `.twonn_id.id_std` (0.126). s2's margin is 0.145.
    - The D-ID ✅ therefore rests, in the L2 leg, on a sub-noise margin in one seed. The depth-56 twin's ID noise is also 6.6× the resnet20 twin's (0.113 vs 0.017).
    - The rule is applied as written. No threshold or band was changed.
  - **The e50 values that trigger FIT-SENSITIVE** are inside their edges by 0.024 (sep_ratio) and 0.057 (bridge). Both are below the twin |Δ| (0.051, 0.066).

### D-ACC (Stage 2 rule with M11; INFO)

- Predicted: NOT-DEPTH in s1 and s2 for all three fields.

| field | band | m* e60 | M12 / M13 | s1 / s2 | label |
|---|---|---|---|---|---|
| `.per_layer.penult.linear_probes.factors.class.excess` | [0.8078, 0.8272] | 0.8167 IN | 0.8145 / 0.8150 IN | 0.8390 H / 0.8330 H | NOT-DEPTH / NOT-DEPTH |
| `.per_layer.penult.class_centers.nearest_center_acc_test` | [0.9158, 0.9338] | 0.9256 IN | 0.9228 / 0.9238 IN | 0.9446 H / 0.9398 H | NOT-DEPTH / NOT-DEPTH |
| `.meta.accuracy.test` | [0.9158, 0.9350] | 0.9244 IN | 0.9204 / 0.9242 IN | 0.9438 H / 0.9404 H | NOT-DEPTH / NOT-DEPTH |

- Observed: as predicted.
- Verdict: **HIT**.

### K: depth-56 seed stability

**K1 — core items on s1-s2** (`c12`; the critic statuses decide, and `a4b_eval.js` recomputes them as a cross-check).
- Predicted: 6/6 PASS.

| item | `c12` check | status and value | recomputed |
|---|---|---|---|
| S1 | `.checks.id_profile_stability` `id_profile[0,1]` | PASS, spearman 0.988, peak layer3.5 / layer3.5 | 0.9879, shift 0 |
| S3 | `.checks.scalar_stability` `penult/class_centers.sep_ratio` | PASS, 5.114 / 5.213, rel 0.019 | same |
| S5 | `.checks.adjacency_stability` `penult/adjacency_spearman[0,1]` | PASS, rho 0.900 | 0.8997 (45 pairs) |
| S7 | max `mean_abs_delta` of `decod/<f>[0,1]` over the 10 sharp factors | 0.082 (saturation_mean) ≤ 0.10, PASS | 0.0817 |
| S8 | `.checks.commit_agreement` `commit/class` | PASS, layer3.5 / layer3.5 | shift 0 |
| S9 | `.checks.panel_agreement` `penult/cka_test[0,1]` | PASS, 0.922 | 0.9218 |

- Observed: 6/6 PASS. `ev` `.core_mismatch` is [] and `.core_source` is "critic critic_v1_resnet56_s1_s2".
- Verdict: **HIT**.
- **Promotion prerequisites.**
  - `c12` `.checks.synthetic_refusal[0]` is PASS ("all runs real").
  - `.checks.holdout_hygiene[0,1]` are PASS ("declared", s1 and s2).
  - input_norm is PASS.
  - `ev` `.promotion_blocked_by` is [].
- **Non-core FAILs** (25) are INFO. Two of them are the s1-s2 decod items for sharp factors, which fail on profile rho: saturation_mean (MAD 0.082, rho 0.53) and colorfulness (0.071, 0.61).
  - S7 as pre-registered is MAD-only: "S7 from its decod/<factor>[0,1] details"; notes K1 and STAGE1.md amendment 2.
  - So S7 PASSES. The two factor profile FAILs are recorded as non-core FAILs (INFO) in row 3; STAGE2B has no depth-56 at-risk lists.

**K0 — KILL-56.**
- Predicted: not triggered.
- Observed: no Stage 1 KILL condition holds in any of the three pairs:

| condition | s1-s2 | hub-s1 (`c012` [0,1]) | hub-s2 ([0,2]) |
|---|---|---|---|
| penult adjacency (KILL < 0.50) | 0.900 | 0.886 | 0.905 |
| ID spearman (< 0.80) / peak shift (≥ 2) | 0.988 / 0 | 1.000 / 0 | 0.988 / 0 |
| sharp factors with MAD > 0.10 (≥ 2) | 0 (max 0.082) | 0 (max 0.058) | 0 (max 0.057) |
| class commit shift (≥ 2) | 0 | 0 | 0 |
| penult cka_test (< 0.60) | 0.922 | 0.926 | 0.926 |

- Verdict: **HIT** (`ev` `.KILL56` false).

### T56: estimator twin at depth 56 (h56 vs tw, `cT`)

**Twin construction.** Same hub weights with an independent reference draw: `.meta.ref_indices` overlap is 2001 of 10000.

**Core items — PASS, 6/6.**
- S1: spearman 1.000, peak layer3.5 / layer3.5.
- S3: 5.332 / 5.383, rel 0.009.
- S5: 0.993.
- S7: max MAD 0.026.
- S8: layer3.5 / layer3.5.
- S9: 1.000.
- `ev` `.T56.fail` is false and `.recomputed_mismatch` is [].

**Forecasts** (`ev` `.T56.forecast`; r20 twin values in brackets):

| forecast | observed | result |
|---|---|---|
| penult adjacency ≥ 0.95 [0.994] | 0.9931 | hit |
| id_profile spearman ≥ 0.95 [1.000] | 1.000 | hit |
| decod MAD ≤ 0.03 for every factor [0.0295] | 0.0346, highfreq_ratio (`cT` decod/highfreq_ratio[0,1] mean_abs_delta 0.035) | **MISS** |
| cka_test ≥ 0.9995 [1.000] | 1.000 | hit, by construction |
| D1 ≥ 0.95 [0.993] | 0.9958 | hit |
| \|Δ penult ID\| ≤ 0.14 [0.017] | 0.1127 | hit |

- The missed forecast is not a gate, and highfreq_ratio is not a sharp factor; the sharp-factor maximum is 0.026.
- The evaluator prints the value without a verdict.
- cka_test is exactly 1 by construction: same weights, same test inputs; only the reference draw changes. So T56's S9 cannot fail, and the twin carries no information about cka noise.

### C: the resnet20 claims at depth 56 (resnet56 tap names; must hold in s1 AND s2)

Paths:
- C1: `.per_layer.<L>.twonn_id.id`.
- C2: `.cross_layer.commit_layer.per_factor.{luminance_mean,highfreq_ratio,spectral_anisotropy}.washout`.
- C3: `.cross_layer.commit_layer.per_factor.class.commit_layer`.
- C4: `.per_layer.penult.corruption_displacement.splits.corrupt__<c>__s<k>.coherence`.
- C5a: `.per_layer.penult.knn_density.splits.corrupt__<c>__s{1,5}.sparse_frac`.

The stored washout and commit values equal their recomputation from `.profile` at tau 0.9.

Quoted prediction (experiments/queue/atlas_v1_resnet56_s1.yaml notes): "Predicted: C1, C2, C3, C4, C5a hold in s1 and s2; C7 fails."

| claim (predicted: holds) | s1 | s2 | verdict |
|---|---|---|---|
| **C1** peak at layer3.0 or layer3.5, penult ≥ 30% below | peak layer3.5 (19.632), penult 10.382, drop 0.471 | layer3.5 (19.471), 10.459, 0.463 | **HIT** |
| **C2** luminance > 0.30, highfreq < 0.20, anisotropy < 0.20 | 0.608 / 0.173 / 0.101 | 0.669 / 0.153 / 0.091 | **HIT** |
| **C3** class commit in {layer3.0, layer3.5, layer3.8} | layer3.5 | layer3.5 | **HIT** |
| **C4** 6/6 | 6/6, min margin 0.1125 (defocus s5 vs shot) | 6/6, 0.1868 | **HIT** |
| **C5a** 8/8 | 8/8, smallest brightness 0.0945 | 8/8, brightness 0.0750 | **HIT** |

The other resnet56 atlases (INFO):

| atlas | C1 drop | C2 | C3 | C4 min margin | C5a smallest |
|---|---|---|---|---|---|
| h56 | 0.450 | 0.757 / 0.150 / 0.076 | layer3.5 | 0.192 | 0.085 |
| tw | 0.451 | 0.760 / 0.154 / 0.084 | layer3.5 | 0.192 | 0.0875 |
| e40s | 0.442 | 0.541 / 0.129 / 0.057 | layer3.5 | 0.125 | 0.053 |
| e50 | 0.420 | 0.498 / 0.144 / 0.048 | layer3.5 | 0.099 | 0.0665 |
| e60 | 0.446 | 0.693 / 0.093 / 0.067 | layer3.5 | 0.084 | 0.0645 |
| e70 | 0.430 | 0.653 / 0.127 / 0.074 | layer3.5 | 0.072 | 0.0785 |
| m12 | 0.433 | 0.707 / 0.173 / 0.050 | layer3.5 | 0.063 | 0.055 |
| m13 | 0.453 | 0.557 / 0.129 / 0.080 | layer3.5 | 0.136 | 0.067 |

- C1-C5a hold in all 10 resnet56 atlases.
- The h56 and e40s values equal the notes' expected values.
- `ev` `.claims` omits tw and e40s; they were recomputed here.

**C7 — penult adjacency seed-stable at depth 56.**
- Predicted (notes): "Predicted FAIL: D1(s1, s2) in [0.60, 0.80)".
- Observed: S5 is PASS (0.900), but D1(s1, s2) = 0.6922, which is < 0.80.
- Verdict: C7 fails, as predicted, and D1 is in the predicted range: **HIT**.

### P: pair-metric levels (P20 band; `ev` `.LEVEL`)

Formulas:
- D1 = spearman over the 45 penult center distances (`scripts/d1_distance_only.js` formula).
- adj = the critic's spearman over the positive upper-triangle entries of `.class_adjacency.sep_matrix`.
- l31 = the same, at resnet20 layer3.1 against resnet56 layer3.5.
- cka = `d(a→b)` `.per_layer.penult.cka_test`.

| m | e20 (pair) | P20 max | median X [range] | W56 | label | median XM (n 15) | matched reading |
|---|---|---|---|---|---|---|---|
| D1 | 0.8671 (s1-s3) | 0.9639 | 0.7866 [0.7072, 0.8623] | 0.6922 | **DEPTH-56-UNSTABLE (COLLAPSE)** | 0.8872 | **NOT-DEPTH at matched accuracy** |
| adj | 0.9331 (s0hub-s4) | 0.9746 | 0.9041 [0.8194, 0.9250] | 0.8997 | **DEPTH-56-UNSTABLE** | 0.9231 | not NOT-DEPTH (XM median < e20) |
| cka | 0.8832 (s1-s3) | 0.8919 (s2-s3) | 0.8700 [0.8645, 0.8717] | 0.9218 | **DEPTH-GAP** | 0.8766 | not NOT-DEPTH |
| l31 | 0.8750 (s1-s3) | 0.9725 | 0.8103 [0.7925, 0.9145] | 0.9553 | **DEPTH-GAP** | 0.8449 | not NOT-DEPTH |

- **CV.** The penult center-distance CV (population std / mean of the 45 distances) is 0.0733 / 0.0685 / 0.0808 / 0.0791 / 0.0742 over B20+, reproducing the committed values. The low edge is 0.0562. s1 / s2 are 0.0428 / 0.0366, both below it, so "(COLLAPSE)" applies.
- **Matched reading wording.** The evaluator labels the "not NOT-DEPTH" case "gap persists at matched accuracy". That wording is the code's; the text defines only the NOT-DEPTH case.

Forecasts:
- **P1** (D1)
  - Predicted: DEPTH-56-UNSTABLE (COLLAPSE), e20 ~0.867, CV edge ~0.056, matched reading NOT-DEPTH.
  - Observed: all as predicted.
  - Verdict: **HIT**.
- **P2** (cka)
  - Predicted: DEPTH-GAP with W56 in [0.883, 0.93].
  - Observed: 0.9218.
  - Verdict: **HIT**.
- **P3** (adj)
  - Predicted: DEPTH-56-UNSTABLE, e20 ~0.933, matched reading NOT-DEPTH.
  - Observed: the label and e20 (0.9331) hit, but the XM median 0.9231 is below e20.
  - Verdict: **PARTIAL (matched reading missed)**.
- **P4** (aligned layer3.1)
  - Predicted: median X in [0.84, 0.90] against e20 ~0.875; no side was predicted.
  - Observed: 0.8103; label DEPTH-GAP.
  - Verdict: **MISS** (range).
- **Rule-8 reference maxima**
  - Predicted: ~0.975 / 0.892 / 0.964.
  - Observed: 0.9746 / 0.8919 / 0.9639.
  - Verdict: hit.

INFO:
- **Seed stability depends on fit.** At matched accuracy the depth-56 seed pairs are as seed-stable as resnet20:
  - D1: e60-s12m 0.926, e60-s13m 0.923, s12m-s13m 0.938.
  - Adjacency: 0.975 / 0.964 / 0.967.
  - The instability appears with full fit: hub-s1 / hub-s2 D1 0.840 / 0.760, s1-s2 0.692.
- **The twin's D1 is 0.996,** so the drop is seed-level geometry, not estimator noise.
- **D1's e20 is set by a single pair** (s1-s3 0.867; the next lowest is 0.911). P1's matched NOT-DEPTH (XM 0.887) sits between the two.

### X: SHAPE labels (the 10 X pairs; `ev` `.SHAPE`)

| row | rule (STAGE2B table) | observed | predicted → observed |
|---|---|---|---|
| 1 | C1 in s1 and s2; median spearman ≥ 0.90; max shift ≤ 1 | C1 in both; median 0.9879; max shift 0 | SCALE-ROBUST → **SCALE-ROBUST** |
| 3 | C2 in s1 and s2; ≥ 9/10 sharp factors with median MAD ≤ 0.10 and rho ≥ 0.70 | C2 in both; 10/10 (worst saturation_mean, MAD 0.056 and rho 0.824) | SCALE-ROBUST → **SCALE-ROBUST** |
| 4 | C3 in s1 and s2; max commit shift ≤ 1 | max shift 0 | SCALE-ROBUST → **SCALE-ROBUST** |
| 5 | C4 6/6 in s1 and s2 | 6/6, 6/6 | SCALE-ROBUST → **SCALE-ROBUST** |
| 6a | C5a 8/8 in s1 and s2 | 8/8, 8/8 | SCALE-ROBUST → **SCALE-ROBUST** |
| 7 | ROBUST: median rho ≥ 0.70 and median D1 ≥ 0.80. COLLAPSE-ARTIFACT: median D1 < 0.80, median layer3.1 ≥ 0.80, both r56 CVs < CV edge | rho 0.9041; D1 0.7866; layer3.1 0.8103 (9 of 10 pairs ≥ 0.80); CVs 0.0428 / 0.0366 < 0.0562 | COLLAPSE-ARTIFACT → **COLLAPSE-ARTIFACT** |
| 8 | median relrep argmax agreement / chance ≥ 3 | 4.72 (4.57-4.82) | SCALE-ROBUST → **SCALE-ROBUST** |
| S9 (INFO in row 8) | median cka_test ≥ 0.80 | 0.8700 | SCALE-ROBUST → **SCALE-ROBUST** |

- **Row 7.**
  - Not SCALE-ROBUST: D1 < 0.80.
  - Not SCALE-FRAGILE: that needs D1 < 0.70 and layer3.1 < 0.70.
  - All three COLLAPSE-ARTIFACT clauses hold. The layer3.1 clause holds by only 0.010.
  - The ARTIFACT and FRAGILE conditions are mutually exclusive, so the order in which they are checked does not matter.
  - COLLAPSE-ARTIFACT is a diagnosis: row 7 gets no scale-robust statement, and the d56 tag is set separately.
- **Row 3 X-medians per sharp factor** (MAD / rho):

  | factor | MAD | rho |
  |---|---|---|
  | luminance_mean | 0.037 | 1.000 |
  | spectral_slope | 0.033 | 0.903 |
  | saturation_mean | 0.056 | 0.824 |
  | hue_sin | 0.028 | 0.939 |
  | colorfulness | 0.048 | 0.879 |
  | orientation_entropy | 0.033 | 0.915 |
  | class | 0.022 | 1.000 |
  | corruption_family | 0.024 | 0.952 |
  | corruption_type | 0.031 | 0.952 |
  | severity | 0.017 | 0.939 |

- **Verdict: 8/8 HIT.**

### M56: margin at depth 56 (`margin_v1_resnet56_s1.yaml` notes)

Band = the five `margin_v1_resnet20_*_st3` values, min − w .. max + w.

**M56-a — auc_margin_typeb.**
- Predicted: ≥ 0.80 everywhere; |mean(s1, s2) − median resnet20| ≤ 0.05; "improves with scale" NOT-DEPTH.
- Observed:
  - s1 / s2: 0.9076 / 0.9039.
  - M11 / M12 / M13: 0.8866 / 0.8870 / 0.8953.
  - resnet20 `_st3`: 0.8848 / 0.8825 / 0.8843 / 0.8897 / 0.8995 (median 0.8848; band [0.8655, 0.9165]).
  - |Δ| = 0.0210, so the item is **SCALE-ROBUST**.
  - No matched run is above 0.9165, so "improves with scale" is **NOT-DEPTH**.
  - `ev` `.M56.a_auc`: `.all_ge_080` true, `.scale_robust` true, `.improves_with_scale_at_matched_acc` false.
- Verdict: **HIT**.

**M56-b — row 9's clauses at depth 56.**
- Predicted: d56 REJECT (hub margin − dist −0.0008, p 0.83).

| clause | s1 | s2 |
|---|---|---|
| M2: `margin_minus_dist_typeb` > 0 and `_p` < 0.05 | +0.00814, p 0.0043: holds | +0.00055, p 0.899: **fails** |
| M3: type-b median < correct median; `median_margin_ratio_typeb` ≤ 0.5; `median_margin_ratio_wrong` ≤ 0.5 | 0.362, 0.282: holds | 0.348, 0.234: holds |
| M4: \|auc s1 − s2\| ≤ 0.05 | `cM` `penult/margin_typeb.auc_margin_typeb` PASS (0.908 / 0.904, 0.004); recomputed 0.0036 | |

- Grammar: M2 fails in s2, so it is not PROMOTE. M2 holds in s1, so it is not REJECT. The tag is **d56 YELLOW (M2, M3, M4)**, matching `ev` `.M56.b_row9_d56.tag`.
- Gates: R0 PASS and the instrument gate OK, so the tag stands. KILL-56 and T56 would only void a PROMOTE.
- Verdict: **MISS** (REJECT predicted).
- **M2-failure attribution: NOT-DEPTH** (forecast NOT-DEPTH, **HIT**). M2 holds at M11 and at every Mc:
  - e60: +0.0312, p 2.4e-15;
  - s12m: +0.0312, p 6.2e-12;
  - s13m: +0.0222, p 7.1e-11.
  - `ev` `.m2_failure_attribution` is "NOT-DEPTH" and `.matched_M2` is [true, true, true].
- INFO:
  - The matched gaps (0.022-0.031) lie just above the resnet20 band's low edge (0.0206) and below every resnet20 seed (0.046-0.071). NOT-DEPTH covers the binary M2 clause only.
  - The seed-11 gap falls with training: e50 / e60 / e70 +0.035 / +0.031 / +0.022; s1 / s2 +0.008 / +0.0005; hub −0.0008.
  - `auc_dist_typeb` rises at the same time (e60 0.855, s1 0.899, s2 0.903, hub 0.907). This is the pre-registered mechanism: the distance AUC catches up as the penult collapses.

**M56-c — exploratory, never promoted.**
- **E1.**
  - Predicted: holds.
  - Observed: `spearman_margin_maxprob` is 0.886 / 0.878 (≥ 0.7). The confidence-matched gap (`auc_margin_confmatched − auc_maxprob_confmatched`) is +0.0060 / +0.0121, within ±0.05.
  - Verdict: **HIT**.
- **Margin beyond confidence.**
  - Predicted: the gap replicates (≥ +0.010 in both seeds) and is TIE-ARTIFACT-POSSIBLE.
  - Observed: s1's gap is +0.0060 < 0.010, so the joint call is **NO-LEAD**.
  - Verdict: **MISS**, on both counts.
  - The evaluator emits the call per seed: `ev` `.M56.c_confidence.tie_call` = [NO-LEAD, TIES-EXCLUDED]. Under the text, the call is joint and s2's TIES-EXCLUDED is not the depth-56 call.
- **Ties** (`ic/maxprob_ties.json`):
  - All 13 records' `n`, `n_correct` and `n_typeb` equal the margin atlases' (e.g. s1: 4719 correct, 214 type-b).
  - `half_tie_confmatched` is s1 1.10e-5, s2 2.02e-5 and hub 3.05e-5, orders of magnitude below 0.5 × any gap.
  - Float32 saturation is nearly absent in the full-fit resnet56 runs: `top_share_correct` is 0.0002 in s1, s2 and the hub, against 0.075-0.091 in resnet20 and 0.005-0.013 in the matched rungs. The forecast's premise does not hold.
  - The tie masses themselves need the dumps and were not recomputed.
- B1's E9 paired test (`margin_b1_resnet56_s{1,2}`) is INFO for B1's SESSION.md and is not read here.

## Frozen evaluator vs frozen text

The TEXT wins; the notes do not make the code authoritative.

**No outcome changes.** The raw files pass the text's version of every check below, so the outcomes under the frozen code and under the text are identical. That is the "both versions" report STAGE2B asks for. No amendment is needed for this decision; a fix before any reuse of `a4b_eval.js` needs a committed amendment.

**Code narrower than the text:**
1. **G0a is not implemented** (the header lists G0b-G0e). Recomputed: PASS.
2. **G0b is narrower than the text.** All four gaps are latent; no run failed G0b.
   - The error-key search covers only `per_layer.<l>.<inv>.error` and `cross_layer.<k>.error` (lines 227-232).
   - "input_norm PASS in every A4b critic" is recorded (`CRIT_NORM`) but not enforced.
   - "penult cka_test in every A4b deformation.json" is not checked.
   - `G0B_BAD` voids only the D fields (line 349). The text voids "the items that read the failed run", which also covers the d56 tags, KILL-56, T56, LEVEL, SHAPE and M56.
3. **G0c's status ignores `code_diff.txt`** (lines 511-512); it is only recorded. git confirms only `margin.py` and `requirements.txt` changed.

**Reporting gaps:**

4. The Stage-2-style label for the four D fields is not produced, although the text says it "is also reported for every field". Recomputed: DEPTH / DEPTH ×4.
5. `accLabel` prints "IN-BAND" where STAGE2.md's rule says UNRESOLVED. Latent.
6. "gap persists at matched accuracy" is a code-only wording (see P).
7. The B20 INFO reading is not given for the W rungs (see D-COLL; no outcome change).
8. The M56-c tie call is per seed; the text's call is joint: NO-LEAD.
9. The M2-failure attribution would silently drop a missing matched margin atlas (`.filter(Boolean)`), where the text says "at M11 and at every Mc". Latent: all three exist.
10. `ev` `.claims` omits tw and e40s. Recomputed: C1-C5a hold.
11. Row 7's tag string does not say which condition fired. In fact S5 PASSES and D1(s1, s2) is 0.692.
12. `T56.core_source` names the critic even when input_norm would force the recomputation (line 315). A label bug only.
13. Four of the six T56 forecasts (decod MAD, cka, D1, |Δ ID|) are printed as values without a verdict.

**Rule readings:**

14. **KILL-56.** The code combines "any condition in s1-s2 AND any condition in a hub pair". The text leaves open whether the same condition must hold in both. Moot: no condition holds in any pair.
15. **PROMO_BLOCK** also blocks on the pair critic's input_norm, which the text lists under G0b. No effect.
16. **`a4b_eval.json` depends on when it is run.** It reads `margin_b1_resnet56_s{1,2}` if present, so a re-run after the B1 pull changes `.M56.c_confidence.b1_e9_info` only. The committed `ev` is the pre-B1 run.
17. **Rule 8's "check T56"** is uninformative for cka (see T56).

## Decision (STAGE2B, applied in order)

**1. Validity: VALID.** The following all pass:
- G0a-G0e, R0, the instrument gate and the D6 preflight;
- G0d stopped nothing;
- ladder status "matched", with two evaluable Mc.

**2. No-verdict guards: clear.**
- R0 passes.
- KILL-56 is not triggered.
- T56 passes its core items and the twin exists.
- The instrument gate is OK.
- `ev` `.guard` is [].

**3. Promotion blocks: clear.** STAGE2B row 11 and AGENT_LOOP.md:

| promotion criterion | evidence | status |
|---|---|---|
| source = real | `c12` `.checks.synthetic_refusal` PASS (margin: `cM` PASS) | met |
| ≥ 2 independent seeds within tolerance | s1 and s2 (seeds 1, 2; distinct sha256; declared confirmation seeds since 664bd25, never trained before); core items 6/6 PASS at `experiments/tolerances_default.yaml` tolerances | met |
| holdout declared before the run | `c12` `.checks.holdout_hygiene` PASS for s1 and s2; `confirmation_seeds: [1, 2, 12, 13]` in every A4b resnet56 manifest | met |
| prediction in `notes:` before the run | quoted above from the P notes; G0e PASS (P before every dump) | met |
| no unexplained critic vs evaluator mismatch | `ev` `.core_mismatch` [] | met |

**4. Row 11.**
- **D-ID CONFIRMED → ✅.**
- **D-COLL FIT-SENSITIVE → 🟡.** Stronger collapse at matched accuracy holds for nc1 in every leg. For sep_ratio and the bridge ratio it depends on training length inside the accuracy window.
- The replacement text for NOT-DEPTH ("not attributable to depth at matched accuracy") does not apply.
- Rows 1 and 2 still get the A4b values in their depth-56 evidence, which supersede Stage 2's e40-hub interpolation. This follows from SCALE-VARIANT being retired and the matched rung now existing; it is not a demotion.

**5. d56 tags** (STAGE2B "Depth-56 scope").

| row | rule | observed | tag |
|---|---|---|---|
| 1 | C1 + S1 | C1 holds in s1 and s2; S1 PASS | **d56 ✅ (s1, s2)** |
| 3 | C2 + S7 | C2 holds in both; S7 PASS (max MAD 0.082) | **d56 ✅ (s1, s2)** |
| 4 | C3 + S8 | C3 holds in both; S8 PASS | **d56 ✅ (s1, s2)** |
| 5 | C4 | 6/6, 6/6 | **d56 ✅ (s1, s2)** |
| 6a | C5a | 8/8, 8/8 | **d56 ✅ (s1, s2)** |
| 7 | S5 and D1(s1, s2) ≥ 0.80; otherwise 🟡 if a hub pair has D1 ≥ 0.80 | S5 PASS (0.900); D1 s1-s2 0.692; hub-s1 0.840, hub-s2 0.760 | **d56 🟡** (one pair) |
| 9 | M56-b | M2 holds in s1, fails in s2; M3 both; M4 PASS | **d56 🟡 (M2, M3, M4)**; M2 failure NOT-DEPTH; M56-c beside it: E1 holds, NO-LEAD |
| 2, 8 | stay 🟡 | S9 PASS (0.922), INFO in row 8 | unchanged |

`ev` `.PROMO` agrees for rows 1-7.

**The depth-56 ✅ scope may be shown, in this form only:**
- as "d56 ✅ (s1, s2)" after the resnet20 status in the status column;
- without widening the claim text;
- with the resnet20 status untouched.

It refers to the claim in resnet56 tap names as pre-registered (C1: peak layer3.0 or layer3.5; C3: {layer3.0, layer3.5, layer3.8}). The owner's commit of the ATLAS_STATUS change is the human approval AGENT_LOOP.md asks for.

**6. SHAPE and LEVEL.** SCALE-VARIANT is retired. The status column shows SHAPE, and the evidence names the LEVEL:

| row | SHAPE | LEVEL |
|---|---|---|
| 1 | SCALE-ROBUST | D-ID CONFIRMED |
| 2 | — | D-COLL sep_ratio FIT-SENSITIVE |
| 3 | SCALE-ROBUST | class probe excess NOT-DEPTH |
| 4, 5, 6a | SCALE-ROBUST | no level field |
| 7 | COLLAPSE-ARTIFACT | pair levels D1, adj, l31 |
| 8 | SCALE-ROBUST | S9 cka DEPTH-GAP |
| 9 | SCALE-ROBUST (M56-a) | "improves with scale" NOT-DEPTH |

**7. Pair-metric labels.**
- D1: DEPTH-56-UNSTABLE (COLLAPSE), NOT-DEPTH at matched accuracy.
- Adjacency: DEPTH-56-UNSTABLE; its matched reading is not NOT-DEPTH.
- cka_test: DEPTH-GAP.
- Aligned layer3.1: DEPTH-GAP.

## Rule 8 (too-good, by-construction or edge numbers)

### The cka flag (`ev` `.rule8.cka_gt_max` true)

- W56 cka_test is 0.92185 (`d(s1→s2)`, `results/atlas_v1_resnet56_s2/compare_vs_atlas_v1_resnet56_s1/deformation.json` `.per_layer.penult.cka_test`).
- That is above the P20 maximum of 0.89191 (`results/atlas_v1_resnet20_s3_st3/compare_vs_atlas_v1_resnet20_s2_st3/deformation.json`, same key).
- The adjacency and D1 flags are false (0.8997 < 0.9746; 0.6922 < 0.9639).

**Artifact hypothesis 1 — collapse-driven agreement.** Status: not refuted, and the likely main driver. Linear CKA on test[:2000] is dominated by between-class variance when the penult is collapsed. s1 and s2 are strongly collapsed:
- nc1 0.053 / 0.050, against 0.163-0.172 in resnet20;
- `.neural_collapse.etf_deviation` 0.057 / 0.062, against 0.097-0.109;
- CV 0.043 / 0.037.

The evidence:
- **Ordering.** Same-depth cka_test rises with collapse:

  | pairs | mean nc1 | cka_test |
  |---|---|---|
  | resnet20 P20 | ~0.168 | 0.883-0.892 |
  | matched depth-56 pairs (e60, s12m, s13m) | ~0.129 | 0.895-0.899 |
  | full-fit depth-56 pairs (s1-s2, hub-s1, hub-s2) | 0.050-0.052 | 0.922-0.926 |

- **Linear fit.** Over the 15 other same-depth pairs, cka = 0.9405 − 0.3154 · mean nc1 (rmse 0.0031). It predicts s1-s2 at 0.9242; observed 0.9218.
- **Other agreement statistics.**
  - Not higher than resnet20: s1-s2 `.per_layer.penult.error_consistency_test` 0.566 (inside the P20 range [0.507, 0.581]), D1 0.692 (the lowest of any same-depth pair), and the class-center-only CKA 0.9928 (= the P20 minimum).
  - Higher, as collapse and higher accuracy predict: relrep_argmax_agree_test 0.947 (P20 max 0.938), relrep_argmax_agree_ood_c100 0.606 = 4.91x chance (0.589 = 4.70x), panel_cka 0.946 (0.920).
  - Lower on CIFAR-100 inputs and in the off-max structure: cka_ood_c100 0.676 (P20 min 0.680); relrep_offmax_corr test 0.749 (P20 min 0.818), ood 0.738 (0.759).
  - So there is no excess agreement where class collapse does not apply, which further supports hypothesis A.
- **Caveat.** Depth and collapse are confounded: the trend runs across three clusters, and there is no resnet20 net at nc1 ~0.05. The decisive test is a within-class-residual CKA (class means removed on test[:2000]) on the dumps on the volume. It was not run here.

**Artifact hypothesis 2 — identical data order.** Status: excluded.
- s1 and s2 have `.recipe.seed` 1 and 2, and `torch.manual_seed` runs before the shuffled loader (`scripts/train_second_seed.py:73,85`).
- Their weights differ (sha256 73c9… vs 14a6…).
- The hub, trained in another codebase, pairs just as high: 0.926 / 0.926.

**The two checks the notes ask for:**
- **T56:** uninformative for cka. cka_test is 1.0000 by construction.
- **X pairs:** all 10 are 0.8645-0.8717, below e20 (0.8832). The excess is within depth 56, not agreement with resnet20.

**Reading.**
- The DEPTH-GAP direction survives: W56 ≥ e20 also holds for the less-collapsed matched pairs, 0.895-0.899.
- The part of W56 above the P20 maximum is likely collapse-driven (hypothesis A, not refuted). Depth and collapse are confounded: the less-collapsed matched depth-56 pairs (0.895-0.899) also exceed the resnet20 maximum 0.892, and the within-class-residual CKA was not run.
- There is no promotion consequence: ATLAS_STATUS has no cka row, and S9 is INFO in row 8. Row 8's evidence carries this reading.

### Other edge and by-construction numbers

- **D-ID's s1 margin (0.068)** and **the e50 FIT-SENSITIVE trigger** are within twin noise (see D-ID and D-COLL).
- **Row 7's COLLAPSE-ARTIFACT** holds by 0.010 on the layer3.1 clause (median 0.8103).
- **P1's matched NOT-DEPTH** rests on a single-pair e20 (s1-s3 0.867).
- **G0c, `check.json` and the D6 preflight are bitwise identical by construction:** same code, dumps and libraries (`ic/versions.json`). They show that Stage B is unchanged and deterministic, and say nothing about how reproducible training is.
- **T56's S9 and the cka forecast** are 1.000 by construction.
- **S7 at depth 56** passes on MAD (0.082, the largest across the three depth-56 pairs), while two sharp factors fail the critic's profile spearman in s1-s2. The item is MAD-only by pre-registration; the two factor FAILs are non-core (INFO) in row 3.
- **The C2 highfreq clause** is the only C2 clause that separates learned from random at depth 56 (Stage 2 null luminance 0.473). s1 and s12m sit at 0.173 against < 0.20.

## Holdout

- **Probe pools.** The decodability pools include the five confirmation corruptions (`atlas/invariants/decodability.py:30`). C2, S7 and S8 are clean confirmations only on the seed and depth axes, which are the axes used here.
- **Confirmation corruptions.** No confirmation-corruption value seeds a claim. `.meta.accuracy.corrupt__*` is not used.
- **Seeds.**
  - resnet56 seeds 1, 2, 12 and 13 are now spent for the atlas instrument. Seeds 1 and 2 also had their one margin touch: M56 here, plus B1's E9 in the same session, which B1 evaluates.
  - Seed 11 stays discovery material: every seed-11 rung shares its initial weights and data order with e10/e20/e40.
  - A further depth-56 confirmation needs new seeds.
- **Margin.** No margin-like quantity is computed from the main atlases. M56 reads only `margin_v1_*`.
- **ANOMALY_H1.** It uses these dumps (AX probes) and reads `ev` (AH items). Its predictions were committed at P_A before the dumps, and it is evaluated in `results/anomaly_h1/SESSION.md`.

## Prediction misses

1. **T56 decod MAD ≤ 0.03 for every factor.** highfreq_ratio is 0.0346. It is not a core item, and the sharp-factor maximum is 0.026.
2. **P3's matched reading.** Predicted NOT-DEPTH at matched accuracy; the XM median is 0.923, below e20 0.933. Depth-56 adjacency stays below resnet20's seed spread even at matched accuracy, although the matched depth-56 seed pairs themselves reach 0.964-0.975.
3. **P4.** The median X aligned layer3.1 was forecast in [0.84, 0.90]; observed 0.810.
4. **M56-b.** Predicted REJECT; observed YELLOW, because M2 holds in s1 (+0.0081, p 0.004). The margin > distance advantage shrinks with fit rather than vanishing uniformly.
5. **M56-c.** Predicted: a replicated lead that is TIE-ARTIFACT-POSSIBLE. Observed: NO-LEAD (s1 +0.0060), and float32 ties are nearly absent in the full-fit resnet56.
6. **Not forecasts (INFO):**
   - e50 and e70 fell just below STAGE2B's curve-model ranges;
   - per-epoch time was 3.78-3.91 s, against 3.45-3.52 s.
7. **Hits (26):** L1-L3; D-ID; D-COLL and its sub-forecasts; D-ACC; K0; K1; C1-C5a; C7 (range [0.60, 0.80)); T56 core and 5 of its 6 forecasts; P1; P2; X (all 8 labels, counted once); M56-a; the M56-b attribution; M56-c E1.
   - The rule-8 reference maxima (recorded, not scored) also matched: 0.9746 / 0.8919 / 0.9639.

## ATLAS_STATUS changes

Apply these in the same commit as this SESSION.md and `a4b_eval.json`. The owner's commit is the human approval AGENT_LOOP.md asks for.

**Summary.**
- **Rows 1, 3, 4, 5, 6a.** The status gains "d56 ✅ (s1, s2) · SCALE-ROBUST". The evidence replaces the Stage 2 depth-56 entry and its SCALE-VARIANT label with the A4b entry, the SHAPE evidence and the LEVEL.
- **Row 2.** The status is unchanged. In the evidence, the A4b values and the LEVEL (sep_ratio FIT-SENSITIVE) replace Stage 2's interpolated DEPTH.
- **Row 7.** The status gains "d56 🟡 (one pair) · COLLAPSE-ARTIFACT". In the evidence, the depth-56 values and the pair LEVELs replace Stage 2's hub-reference reading (D1 0.802; UNDECIDED against three references).
- **Row 8.** The status becomes "🟡 · SCALE-ROBUST". The evidence gains the depth-56 relrep and S9 as INFO, with its LEVEL and the rule-8 reading.
- **Row 9.** The status gains "d56 🟡 (M2, M3, M4) · SCALE-ROBUST (M56-a)". The evidence gains the M56 entry.
- **Row 11.** The status becomes "D-ID ✅ · D-COLL 🟡 (FIT-SENSITIVE)".
- **Rows 6b and 10.** Unchanged; row 10 belongs to B1.
- **Header.** The Stage 2 and A4b sentences are replaced.
- **No resnet20 status symbol changes, and no claim text changes.**

**Header paragraph.** Replace the text from "Stage 2 (A4, resnet56 hub; docs/plans/STAGE2.md) is valid and promotes nothing" through "(S9 as INFO in row 8)." with the text below. Keep the B1 sentence that follows it unchanged; B1's SESSION.md updates it.

> Stage 2 (A4, resnet56 hub; docs/plans/STAGE2.md) is valid and promotes nothing (results/atlas_v1_resnet56_s0hub/SESSION.md); A4b supersedes its interpolated penult-level DEPTH readings. A4b (docs/plans/STAGE2B.md: resnet56 seeds 1-2, the seed-11 matched rung e60 with seed-12/13 replicates, the `_ref1` twin, a same-session `_st3` resnet20 band) is valid (results/atlas_v1_resnet56_s1/SESSION.md). Rows 1, 3, 4, 5 and 6a hold at depth 56 on two last-epoch seeds: d56 ✅ (s1, s2), SCALE-ROBUST. Row 7 is d56 🟡 (D1 s1-s2 0.692; hub-s1 0.840), and its cross-depth drop is diagnosed as COLLAPSE-ARTIFACT. Row 9 is d56 🟡: margin > distance holds in s1 and fails in s2, which is not attributable to depth. Row 11: at matched accuracy the resnet56 penult ID is higher (D-ID ✅); the stronger collapse holds for nc1 but depends on training length for sep_ratio and the bridge ratio (D-COLL 🟡, FIT-SENSITIVE). A d56 tag scopes a row's claim to depth 56 in resnet56 tap names (experiments/queue/atlas_v1_resnet56_s1.yaml notes) and does not widen the claim text; the status before it is resnet20's. The status column shows the SHAPE label and the evidence names the LEVEL (SCALE-VARIANT is retired).

**Rows.** Rows 6b and 10 are unchanged. Every other row below is given in full; its claim cell is unchanged. Paste the rows without the header line.

| # | layer(s) | entry | claim (current) | status | evidence |
|---|---|---|---|---|---|
| 1 | all | twonn_id profile | ID peaks at layer3.0/3.1 and penult is ≥ 30% below the peak (C1); only the last-block drop is learned (random init also peaks at layer3.1 but drops 0.02) | ✅ s1, s2, s3, s4 · d56 ✅ (s1, s2) · SCALE-ROBUST | v1 drop s1/s2/s3/s4 0.498 / 0.488 / 0.503 / 0.492, peak layer3.1 in all · v0 19.46 → 9.87 · Stage 1b CORE-PASS (results/atlas_v1_resnet20_s3/SESSION.md) · depth 56 (A4b, results/atlas_v1_resnet56_s1/SESSION.md): C1 in resnet56 taps (peak layer3.0 or layer3.5, penult ≥ 30% below) holds in s1 and s2 (peak layer3.5, drop 0.471 / 0.463; hub 0.450; matched rungs e60 / s12m / s13m 0.446 / 0.433 / 0.453); S1 s1-s2 spearman 0.988, shift 0 → d56 ✅ (s1, s2); depth-56 null drop 0.001 (Stage 2) · SHAPE: median spearman over the 10 resnet20 × {s1, s2} pairs 0.988, max peak shift 0 · LEVEL: penult ID higher than resnet20 at matched accuracy (row 11 D-ID CONFIRMED; Stage 2 R56-1c had predicted lower, FAIL) |
| 2 | penult | class_centers.sep_ratio | nearest-other / RMS on the train reference ≈ 3.0 under v1; not comparable to the historical 2.87 (different formula and split) | 🟡 seed-stable | v1 hub 2.996, s1 3.067, s2 3.021, s3 2.970, s4 3.024 · v0 2.941 · resnet56 (A4b, results/atlas_v1_resnet56_s1/SESSION.md): hub 5.332, s1 5.114, s2 5.213; matched rung e60 3.368, replicates s12m / s13m 3.367 / 3.314, e70 3.506 (all above the B20+ edge 3.163); window rung e50 3.140, inside the band · LEVEL: sep_ratio field FIT-SENSITIVE (row 11 D-COLL); this replaces Stage 2's interpolated DEPTH (m* 3.528 from the e40-hub interpolation) |
| 3 | penult | linear_probes washout | luminance washout > 0.30; highfreq and anisotropy < 0.20 (C2); cite penult luminance, not the stem R² (architecture artifact); the anisotropy clause also holds in random init (0.085) | ✅ s1, s2, s3, s4 · d56 ✅ (s1, s2) · SCALE-ROBUST | v1 s1 0.736/0.173/0.062, s2 0.682/0.119/0.050, s3 0.549/0.133/0.086, s4 0.644/0.161/0.069 · seed-axis confirmation only (probe pool includes the confirmation corruptions) · depth 56 (A4b): C2 holds in s1 0.608 / 0.173 / 0.101 and s2 0.669 / 0.153 / 0.091 (hub 0.757 / 0.150 / 0.076); S7 s1-s2 max sharp-factor MAD 0.082 (saturation_mean) → d56 ✅ (s1, s2); non-core FAIL (INFO): the s1-s2 decod profile spearman fails for saturation_mean (0.53) and colorfulness (0.61); S7 is MAD-only by pre-registration; the depth-56 random null clears the luminance clause (0.473, Stage 2), so at depth 56 only highfreq (null 0.278; s1 0.173) separates learned from random · SHAPE: 10/10 sharp factors aligned over the 10 resnet20 × {s1, s2} pairs (worst median MAD 0.056, rho 0.824) · LEVEL: penult class probe excess NOT-DEPTH (s1 / s2 0.839 / 0.833 above the band; matched rung e60 0.817 inside) |
| 4 | layer3.x | commit_layer class | commits in {layer3.0, layer3.1, layer3.2 (= penult)} at tau 0.9 (C3) | ✅ s1, s2, s3, s4 · d56 ✅ (s1, s2) · SCALE-ROBUST | v1 layer3.1 in hub, s1, s2, s3, s4 (random init: stem) · depth 56 (A4b): C3 in resnet56 taps ({layer3.0, layer3.5, layer3.8}) holds in s1 and s2 (layer3.5; hub, e40 and every matched rung also layer3.5); S8 s1-s2 shift 0 → d56 ✅ (s1, s2); hub stride 1 layer3.4 (23/27), stage 3 at tau 0.85 / 0.90 / 0.95; depth-56 null commits at stem (Stage 2) · SHAPE: max aligned commit shift 0 over the 10 resnet20 × {s1, s2} pairs |
| 5 | penult | corruption_displacement coherence | defocus s5, motion s3/s5 have a smaller mean shift per unit per-sample shift than gaussian and shot noise at the same severity (C4) | ✅ s1, s2, s3, s4 · d56 ✅ (s1, s2) · SCALE-ROBUST | v1 6/6 in all four seeds, min margin 0.132 (s1), 0.115 (s2), 0.105 (s3), 0.117 (s4); hub 0.113; mean-shift reading; query sets fixed · depth 56 (A4b): 6/6 in s1 and s2, min margin 0.113 / 0.187 (hub 0.192; matched rungs e60 / s12m / s13m 6/6 at 0.084 / 0.063 / 0.136; depth-56 null 4/6, Stage 2) → d56 ✅ (s1, s2) · SHAPE: 6/6 in both seeds |
| 6a | penult | knn_density sparse_frac | s5 > s1 for the 8 non-noise discovery corruptions (C5a) | ✅ s3, s4 (pre-registered; s1, s2 post-hoc support) · d56 ✅ (s1, s2) · SCALE-ROBUST | v1 8/8 in s1, s2, s3, s4; smallest margin brightness 0.048 / 0.0525 / 0.0515 / 0.0395; twin reference-draw shift of that margin 0.0165 · depth 56 (A4b): 8/8 in s1 and s2, smallest brightness 0.0945 / 0.075 (hub 0.085; matched rungs 8/8, 0.055-0.067; depth-56 null 3/8, Stage 2) → d56 ✅ (s1, s2) · SHAPE: 8/8 in both seeds |
| 7 | penult | class adjacency (merge order non-core) | penult adjacency is seed-stable; the distance-only rho is the headline number (train-reference caveat); merge order non-core | ✅ penult adjacency (S5 PASS and D1 ≥ 0.80 on s3-s4); merge order 🟡 · d56 🟡 (one pair) · COLLAPSE-ARTIFACT (diagnosis; no scale-robust statement) | distance-only rho (D1) s3-s4 0.919, s1-s2 0.950, all six seed pairs 0.867-0.964, hub-s3/s4 0.923/0.958, null 0.589-0.644 · critic rho s3-s4 0.965, hub-s3/s4 0.973/0.933 (partly per-class radius ranking) · merge tau s1-s2 0.838, s3-s4 0.871 · depth 56 (A4b, results/atlas_v1_resnet56_s1/SESSION.md): s1-s2 critic rho 0.900 (S5 PASS) but D1 0.692 < 0.80; D1 hub-s1 0.840, hub-s2 0.760 → d56 🟡 (one hub pair ≥ 0.80); at matched accuracy the depth-56 seed pairs (e60, s12m, s13m) reach D1 0.923-0.938 and rho 0.964-0.975; twin D1 0.996; r56 penult center-distance CV s1 / s2 0.043 / 0.037 (resnet20 0.069-0.081) · SHAPE COLLAPSE-ARTIFACT over the 10 resnet20 × {s1, s2} pairs: median D1 0.787 < 0.80, aligned layer3.1 rho 0.810 ≥ 0.80, both CVs below the edge 0.056 · LEVEL (against the 10 resnet20 seed pairs): D1 DEPTH-56-UNSTABLE (COLLAPSE), NOT-DEPTH at matched accuracy (median 0.887 ≥ e20 0.867); adjacency DEPTH-56-UNSTABLE (median 0.904 and s1-s2 0.900 < e20 0.933; matched median 0.923, still below); aligned layer3.1 DEPTH-GAP (0.810 < e20 0.875 ≤ s1-s2 0.955) |
| 8 | penult | relrep agreement | exploratory: CIFAR-100 relrep argmax agreement across seeds ≥ 3× chance; panel argmax is trivial (train images) | 🟡 · SCALE-ROBUST | v1 s1-s2 0.589 vs chance 0.126 (4.67×; null 1.36×); s3-s4 0.581 vs 0.124 (4.70×) · depth 56 (A4b): median over the 10 resnet20 × {s1, s2} pairs 4.72× chance (4.57-4.82×; hub 4.62×, Stage 2) · SHAPE SCALE-ROBUST · S9 (INFO; no cka row): penult cka_test s1-s2 0.922 PASS; resnet20 × {s1, s2} median 0.870 (SCALE-ROBUST); LEVEL DEPTH-GAP (0.870 < e20 0.883 ≤ s1-s2 0.922); s1-s2 lies above the resnet20 seed-pair maximum 0.892 (rule 8). The likely driver is penult collapse (rule-8 hypothesis A, not refuted; depth and collapse are confounded and the within-class-residual CKA was not run), not higher seed stability: cka rises with collapse (resnet20 0.883-0.892, matched resnet56 0.895-0.899, full fit 0.922-0.926) |
| 9 | penult | margin_typeb (type-b = wrong and maxprob > 0.7) | margin ~ maxprob (not an independent detector): at cut 0.7 the top-2 margin (d2 − d1) to train-reference class centers separates type-b from correct test samples better than nearest-center distance (M2), and the type-b median margin is ≤ 0.5 × the correct median (M3); the margin AUC equals the maxprob AUC (confidence-matched gap ≤ 0.002, Spearman 0.92-0.93, E1); the ratio clause is cut-dependent (> 0.5 at cut 0.99 in every run) | ✅ s1, s2, s3, s4 · d56 🟡 (M2, M3, M4) · SCALE-ROBUST (M56-a) | v1 s1/s2/s3/s4: auc_margin_typeb 0.882 / 0.884 / 0.890 / 0.899 (range 0.017; twin 0.0005; hub 0.885), margin − dist 0.046 / 0.063 / 0.063 / 0.055 (p ≤ 1.6e-19), ratio 0.280 / 0.284 / 0.285 / 0.266 · M1 gate v0 legacy block 0.893, raw 0.107 (legacy 0.895, docs/history/SESSION_RESULTS_VALLEYS_TYPEB_SCALE.md:55-59) · E1 confmatched margin − maxprob +0.0017 / +0.0003 (s1/s2) · null auc_margin_wrong 0.542, no type-b · resnet56 hub (E5, exploratory) 0.907 but margin − dist −0.0008 (p 0.83) · results/margin_v1_resnet20_s1/SESSION.md; results/margin_*/atlas.json per_layer.penult.margin_typeb · depth 56 (A4b M56, results/atlas_v1_resnet56_s1/SESSION.md): auc_margin_typeb s1 / s2 0.908 / 0.904 vs resnet20 _st3 median 0.885 (abs Δ 0.021: SCALE-ROBUST; hub 0.907); matched e60 / s12m / s13m 0.887 / 0.887 / 0.895 are inside the resnet20 band, so "improves with scale" is NOT-DEPTH · M2 margin − dist s1 +0.0081 (p 0.004), s2 +0.0005 (p 0.90); M3 holds (type-b ratio 0.362 / 0.348); M4 abs Δ auc 0.004 PASS → d56 🟡 (M2, M3, M4; M2 fails in s2); the M2 failure is NOT-DEPTH (M2 holds at e60, s12m, s13m: +0.031 / +0.031 / +0.022, p ≤ 7.2e-11; the distance AUC catches up as the penult collapses) · M56-c (exploratory): E1 holds (Spearman 0.886 / 0.878; confidence-matched margin − maxprob +0.006 / +0.012); margin beyond confidence NO-LEAD (s1 +0.006 < +0.010) |
| 11 | penult | twonn_id.id; class_centers.sep_ratio, bridge ratio, neural_collapse.nc1 (resnet56 vs resnet20) | at accuracy matched to resnet20 (within 0.005 of 0.9259), the resnet56 penult ID is higher than the resnet20 band B20+ (D-ID) and the penult collapse is stronger (sep_ratio and bridge ratio higher, nc1 lower; D-COLL), in the matched leg (seed-11 rung M11 and the seed-12/13 replicates) and in the full-recipe leg (seeds 1, 2) | D-ID ✅ · D-COLL 🟡 (FIT-SENSITIVE) | A4b (results/atlas_v1_resnet56_s1/SESSION.md; docs/plans/STAGE2B.md outcome table; node scripts/a4b_eval.js): B20+ = resnet20 hub, s1-s4 (_st3); matched rung M11 = e60 (acc_10k 0.9256; window rungs e50 / e70 0.9220 / 0.9253), replicates s12m 0.9249 and s13m 0.9277 (seeds 12, 13, 60 epochs); full-recipe seeds s1 0.9442, s2 0.9404 · D-ID CONFIRMED (predicted CONFIRMED): penult ID e60 10.77, s12m 10.72, s13m 10.73, e50 / e70 11.03 / 10.87, s1 10.38, s2 10.46, all above the edge 10.31; s1 clears it by only 0.068, less than the depth-56 twin abs Δ 0.113 (rule applied as written) · D-COLL FIT-SENSITIVE (predicted FIT-SENSITIVE): nc1 CONFIRMED (e60 / s12m / s13m 0.129 / 0.129 / 0.130, e50 0.145, s1 / s2 0.053 / 0.050; edge 0.154); sep_ratio and bridge ratio FIT-SENSITIVE (e50 3.140 and 3.434 inside the band, edges 3.163 and 3.491; e60, e70, s12m, s13m, s1 and s2 above) · [ACC] class probe excess, nearest-center accuracy, accuracy: NOT-DEPTH in s1 and s2 · the matched rungs are less fit than resnet20 (train-reference accuracy 0.998-0.999 vs 0.9994-0.9997) · Stage 2 readings R56-1c, R56-9c (results/atlas_v1_resnet56_s0hub/SESSION.md) |

## Next actions

1. **Commit.** Commit `results/atlas_v1_resnet56_s1/a4b_eval.json` (currently untracked), this SESSION.md and the ATLAS_STATUS changes after `81f77ac`, then push.
   - Commit order: P `f1c3c43` → P_A `d257d91` → results `81f77ac` → A4b decision.
   - Do not re-run `a4b_eval.js` over the committed JSON after the B1 pull. If it is re-run, only `.M56.c_confidence.b1_e9_info` may change.
2. **B1.** The first `block_b1` ended with exit 124 (the 1800 s `b1_stage` limit killed the resnet50 Stage B at its 18th tap; no ViT data was read). It is relaunched with a 7200 s limit and `ATLAS_B1_CHECK_DIR=results/instrument_check_b1_r2` (results/margin_b1_vitb16/RELAUNCH_r2.md). When it finishes, pull and commit the B1 results, then evaluate them separately: `scripts/b1_verdicts.js --a4b results/atlas_v1_resnet56_s1/a4b_eval.json`, which reads `.M56.b_row9_d56.tag` "d56 YELLOW" and `.m2_failure_attribution` "NOT-DEPTH".
   - The joint reading follows STAGE2B "M56" (D13) and is written in B1's SESSION.md.
   - The E9 paired margin − maxprob test at depth 56 is read there as INFO beside M56-c.
3. **ANOMALY_H1.** Evaluate after A4b and B1 (RUN_REQUEST): `node scripts/anomaly_eval.js --p d257d91 --p-run d257d91 --a4b results/atlas_v1_resnet56_s1/a4b_eval.json --b1 results/margin_b1_vitb16/verdicts.json …` → `results/anomaly_h1/SESSION.md`.
4. **Delete the pod** after the B1 relaunch, the anomaly block and the final pull; billing continues until then.
5. **Evaluator gaps.** Before `a4b_eval.js` or a derivative is reused, fix discrepancies 1-13 by a committed amendment:
   - G0a;
   - the G0b clauses and their voiding scope;
   - G0c `code_diff.txt`;
   - Stage-2-style D labels;
   - the joint M56-c call;
   - UNRESOLVED instead of IN-BAND;
   - the Mc completeness check;
   - T56 forecast verdicts;
   - the row 7 tag detail.
6. **Rule 8 follow-up (new pre-registration).** A within-class-residual CKA (class means removed on test[:2000]) on the s1/s2, matched and resnet20 dumps. It would separate collapse-driven cka agreement from depth.
7. **Harden the edge outcomes (optional, new pre-registration).**
   - D-ID's L2 leg rests on s1's sub-noise margin; new depth-56 seeds, not 1, 2, 12 or 13, would harden it.
   - Under the current outcome table any in-window rung off the side gives FIT-SENSITIVE; a rung matched in both accuracy and fit (STAGE2B "Not covered") is one possible design, probably unreachable with this recipe.
   - Row 7 at depth 56 needs a collapse-controlled distance statistic before a d56 ✅ is possible, because D1 is fit-dependent (matched pairs 0.92-0.94, full fit 0.69).
