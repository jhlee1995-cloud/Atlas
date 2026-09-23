# SESSION: Stage 2 (A4) — scale transfer resnet20 → resnet56

This is the Evaluator pass on the pulled Stage 2 results. The dumps stay on the RunPod volume.

- **Pre-registration.**
  - `experiments/queue/atlas_v1_resnet56_s0hub.yaml` notes: R56-0..R56-9 and rule 8.
  - `experiments/queue/atlas_v1_resnet56_s0hub_b1.yaml` notes: R56-5c and F1-F4.
  - `docs/plans/STAGE2.md`: design, band, attribution rule and per-row scale labels.
  - Committed as **P = `664bd25`** at 2026-09-22T22:30:47-07:00 (2026-09-23 05:30:47 UTC). `origin/main` is at P.
- **How it was checked.**
  - Three evaluator families (gates/normalization/ladder; structure; flow/commit/null) recomputed every prediction from the result files with node.
  - This pass re-ran every number a verdict or label rests on: accuracies and t, every [ACC] band and m*, the shape claims in the hub and all rungs, the critic items, D1 with the committed script, rule 8, the `_b1` items and R56-0c.
  - No discrepancy was found in any verdict or label, and nothing in the repository was modified by the evaluation.
  - Two independent verifiers recomputed every verdict and [ACC] label from the raw files and reproduced all of them. Their text corrections are applied here: repository state, the ordering deviation, R56-0c wording, row 7 readings, d1/d2 cka_test digits, and three scale statements.
- **Standing** (`docs/plans/STAGE1.md` amendment 2 item 7).
  - The Stage 1b decision is committed as `e86c815` (CORE-PASS, 2026-09-22 23:10:59 -0700), followed by the A3 decision `a30729f` (23:27:38). Both were local, two commits ahead of `origin/main` (= P), when this SESSION.md was written; they are pushed together with it.
  - The verdicts below stand if that CORE-PASS stands. If Stage 1b ends KILL or INVALID instead, every item here is INFO, nothing from it is used, and Stage 2 is re-run after re-confirmation.
  - **Deviation from amendment 2 item 7** ("evaluated only after the Stage 1b decision is committed"):
    - The Stage 2 results were in the local checkout from about 23:04 -0700 (`compare_vs_*` directory mtimes).
    - Four Stage 2 scratch scripts that read resnet56 files predate `e86c815` (23:10:59): st2.js 23:08:32, r567.js 23:09:14, attr.js 23:10:38 (the [ACC] attribution) and ladder.js 23:10:48. The overlap is 23:08:32-23:10:48.
    - `results/atlas_v1_resnet20_s3/SESSION.md` was last written at 23:10:30. Its line 55 ("contain only RUN_REQUEST.md in this checkout") was probably already out of date when it was committed.
    - Mitigation: the Stage 1b decision reads only resnet20 files and quotes no Stage 2 value. This SESSION.md was written after `a30729f`.

**Outcome: VALID; nothing promoted.**
- **Gates.** R56-0a and R56-0b pass. R56-0c (recorded) reproduces every measured Stage 1 value exactly.
- **Tally.** 31 entries: 28 PASS, 2 FAIL (R56-1c, R56-9c), 0 NOT_EVALUABLE, and 1 recorded without a verdict (F3). Rule 8 is not triggered.
  - R56-8 was marked NOT_EVALUABLE in the evaluator pass, because the task said A3 was still running. A3's M1 passed in `a30729f`, so R56-8 is scored here from the committed A3 E5 lines (see R56-8 below).
- **Shapes transfer; penult levels do not.**
  - What holds at depth 56: C1-C5a, the class commit (layer3.5 at stride 5; stride-1 layer3.4 = 23/27 = 0.852, inside resnet20's resolution interval (0.778, 0.889]), cross-model agreement (cka_test 0.863), and the margin AUC (0.907).
  - Penult ID is higher at matched accuracy: DEPTH. R56-1c predicted lower, so it FAILs.
  - Penult collapse is stronger at matched accuracy (sep_ratio, bridge ratio, nc1): DEPTH. R56-9c predicted NOT-DEPTH, so it FAILs.
  - Class probe excess, nearest-center accuracy and accuracy itself are NOT-DEPTH.
- **The collapse labels rest on the interpolation.** They depend on the pre-registered linear interpolation between e40 and the test-selected hub, and e40 by itself lies on the opposite side of the band. The ID label does not depend on it.

Path shorthands:

| shorthand | path |
|---|---|
| `h56` | `results/atlas_v1_resnet56_s0hub/atlas.json` |
| `e10`, `e20`, `e40` | `results/atlas_v1_resnet56_e{10,20,40}/atlas.json` |
| `n56` | `results/atlas_v1_resnet56_rand/atlas.json` |
| `b1` | `results/atlas_v1_resnet56_s0hub_b1/atlas.json` |
| `B20` | `results/atlas_v1_resnet20_{s0hub,s1,s2}_st2/atlas.json` (values listed in that order) |
| `d0`, `d1`, `d2` | `results/atlas_v1_resnet56_s0hub/compare_vs_atlas_v1_resnet20_{s0hub,s1,s2}_st2/deformation.json` |
| `dE<x>` | `results/atlas_v1_resnet56_<x>/compare_vs_atlas_v1_resnet20_s0hub_st2/deformation.json`, x = e10, e20, e40, rand |
| `dst2` | st2 pair deformations: `results/atlas_v1_resnet20_s1_st2/compare_vs_atlas_v1_resnet20_s0hub_st2`, `results/atlas_v1_resnet20_s2_st2/compare_vs_atlas_v1_resnet20_{s0hub,s1}_st2` |
| `c0`, `c1`, `c2` | `results/critic_v1_scale_r20{hub,s1,s2}_r56hub/critic.json` (`--align position`) |
| `cE<x>` | `results/critic_v1_scale_r20hub_r56<x>/critic.json`, x = e10, e20, e40 |
| `cb` | `results/critic_v1_resnet20_st2_band/critic.json` |
| `cn` | `results/critic_v1_resnet56_null/critic.json` |
| `nc20`, `nc56` | `results/norm_check_resnet{20,56}/norm_check.json` |
| `t<x>` | `results/train_resnet56_<x>/train.json`, x = e10, e20, e40, rand99 |

Per-layer fields are `.per_layer.<L>.<invariant>.<key>`. "Aligned" names are the resnet20 names that the resnet56 taps are paired with.

## Run facts

**Provenance.**
- `.meta.git_commit` is 664bd25 in all nine Stage 2 atlases: the five stride-5 atlases, `b1`, and the three `_st2` atlases.
- `.meta.created` runs from 2026-09-23 05:51:48 (`atlas_v1_resnet20_s0hub_st2`) to 06:02:57 (`b1`), on the pod clock (UTC). All are after P.
- `git diff 664bd25 HEAD -- docs/plans/STAGE2.md experiments/queue atlas scripts` is empty, and those paths are clean in the worktree.

**Accuracies** (10k; these are the matching keys):
- Target: `nc20` `.chenyaofo.acc_10k` = 0.9259.
- Hub: `nc56` `.chenyaofo.acc_10k` = 0.9438 (`.cifar_true.acc_10k` = 0.9425).
- Rungs: `te10` / `te20` / `te40` `.final_test_acc_10k` = 0.8399 / 0.8985 / 0.9192.
- Null: `trand99` = 0.1002.
- All of these match the pod log.

**Rungs.**
- `.recipe`: seed 11, norm chenyaofo, epochs 10 / 20 / 40, batch 256, lr 0.1, wd 5e-4.
- Cosine T_max = E, and the last epoch is kept (`scripts/train_second_seed.py:107,124`).
- `.torch` 2.8.0+cu128 and `.device` RTX 4090, the same as Stage 1b.

**Covariates.**

| run | `.meta.accuracy.test` (5k) | `.meta.accuracy.ref` (train-reference fit) |
|---|---|---|
| hub | 0.9446 | 1.0 |
| e10 | 0.8384 | 0.8753 |
| e20 | 0.8966 | 0.9506 |
| e40 | 0.9160 | 0.9936 |
| null | 0.101 | 0.099 |
| B20 | 0.9256 / 0.9226 / 0.9250 | 0.9996 / 0.9994 / 0.9997 |

**Critic counts** (`.counts`; advisory):

| critic | PASS | FAIL |
|---|---|---|
| c0 | 120 | 24 |
| c1 | 125 | 19 |
| c2 | 116 | 28 |
| cb | 207 | 19 |
| cEe10 | 90 | 54 |
| cEe20 | 94 | 50 |
| cEe40 | 113 | 31 |
| cn | 40 | 104 |

- cn's `.verdict` is DOES_NOT_REPLICATE, which is the expected result for the null.
- Each critic also has 1 WARN (the alias check) and 1-3 INFO.

## Gates (any failure = INVALID)

**R56-0a — PASS.**
- `.layers` of h56, e10, e20, e40 and n56 = [stem, layer1.0, layer1.5, layer1.8, layer2.0, layer2.5, layer2.8, layer3.0, layer3.5, layer3.8, penult].
- All seven r20→r56 deformation.json files (d0, d1, d2, dEe10, dEe20, dEe40, dErand) pair resnet20's 11 taps index by index in `.layers`: stem, 1.0, 1.1→1.5, 1.2→1.8, 2.0, 2.1→2.5, 2.2→2.8, 3.0, 3.1→3.5, 3.2→3.8, penult.
- This is the position branch of `atlas/compare.py:67-71`: 5 shared names is below 0.6 × 11 = 6.6. The critics use the same map (`.alignment`).
- b1 has 29 taps, as expected.

**R56-0b — PASS.**
- `.skipped` = {} and there is no `error` key in any of the nine Stage 2 atlases. Every field that R56-1..7 read is present.
- `.checks.norm_consistency[0]` (input_norm) is PASS in all eight critics.
- `.per_layer.penult.cka_test` is present in all 15 Stage 2 deformation.json files.
- Alias checks are WARN, as STAGE2.md expects: layer3.2 in the aligned critics and layer3.8 in cn are identical to penult and counted once.

**R56-0c (recorded) — PASS.** In `results/atlas_v1_resnet20_s0hub_st2/compare_vs_atlas_v1_resnet20_s0hub/deformation.json`:
- `.same_space` is true and `.per_layer.penult.cka_test` is 1.
- `.meta.accuracy.test` is 0.9256 in both atlases.
- Δ penult sep_ratio = 0 (2.99624); Δ twonn_id.id = 0 at all 11 layers; Δ penult k_occurrence_skew = 0 (0.76316).
- `.per_layer`, `.cross_layer` and `.meta.accuracy` of the two atlas.json files are identical: 0 mismatches over 26,965 numeric leaves. Only `.timing_s` (39 values) and the run metadata strings (exp_id, built, dump, meta.git_commit, meta.created) differ.
- So Stage 1 values may be quoted beside Stage 2 values.

## Accuracy control

**R56-N — PASS (evidence).**
- `nc56` `.chenyaofo.acc_10k` = 0.9438. It lies in [0.9414, 0.9444] and inside the best-epoch window [0.9430, 0.9444].
- So the hub is the epoch-198 checkpoint (log: best 94.37, last 94.21), which means it was selected on the test set.
- `.favours` = "neither (not significant)" (`.mcnemar` 95 vs 82, p = 0.367). That is not cifar_true.
- [ACC] clause: `h56` `.meta.accuracy.test` = 0.9446, above the B20 high edge 0.9286, so it holds.
- Label **NOT-DEPTH**: m* = 0.9238 falls inside [0.9196, 0.9286], by construction.
- Unlike resnet20 (`nc20` `.favours` = chenyaofo), the depth-56 check does not separate the two normalizations. `norm: chenyaofo` rests on the training log (STAGE2.md, "Facts").

**R56-9a — PASS.** 0.8399 < 0.8985 < 0.9192 < 0.9438.

**R56-9b — PASS; m* is evaluable.**
- Step 1 fails: |Δacc| = 0.0860 / 0.0274 / 0.0067 for e10 / e20 / e40, all above 0.005.
- Step 2: the only bracketing adjacent pair is (e40, hub).
  - t = (0.9259 − 0.9192) / (0.9438 − 0.9192) = **0.272358**.
  - m* = m(e40) + t · (m(hub) − m(e40)).
- Boolean clauses use the nearest rung, e40 (Δacc −0.0067).
- Sensitivity of t (reported, not used). The range is 0.260-0.336:

| variant | t |
|---|---|
| hub at its last epoch (0.9421) | 0.2926 |
| target at its last epoch (0.9256) | 0.2602 |
| both at the last epoch | 0.2795 |
| 5k accuracies | 0.3287 / 0.3357 |

- Both hub points are test-selected best-epoch checkpoints; the rungs are last-epoch.

## Attribution ([ACC] fields; STAGE2.md rule, band B20, t = 0.272358)

| field (`.per_layer.penult.` …) | B20 | band | r56 hub | e10 / e20 / e40 | m* | label | flip t* |
|---|---|---|---|---|---|---|---|
| twonn_id.id (R56-1c) | 9.9150 / 9.7566 / 10.0355 | [9.4777, 10.3144] | 10.6695 HIGH | 11.9835 / 11.7399 / 10.8849 (all HIGH) | 10.8262 HIGH | **DEPTH** | none (DEPTH for every t) |
| class_centers.sep_ratio (R56-2) | 2.9962 / 3.0666 / 3.0212 | [2.9258, 3.1370] | 5.3323 HIGH | 1.4530 / 2.0395 / 2.8531 (e40 LOW) | 3.5283 HIGH | **DEPTH** | 0.1145 |
| bridge ratio, derived (R56-2) | 3.3916 / 3.4243 / 3.3880 | [3.3516, 3.4606] | 5.5874 HIGH | 2.0676 / 2.5921 / 3.1921 (e40 LOW) | 3.8445 HIGH | **DEPTH** | 0.1121 |
| neural_collapse.nc1 (R56-2) | 0.16994 / 0.16302 / 0.16962 | [0.15609, 0.17687] | 0.04876 LOW | 0.68593 / 0.32016 / 0.17744 (e40 HIGH) | 0.14239 LOW | **DEPTH** | 0.1659 |
| linear_probes.factors.class.excess (R56-4c) | 0.81600 / 0.81425 / 0.81725 | [0.81125, 0.82025] | 0.83675 HIGH | 0.72275 / 0.78025 / 0.80600 (e40 LOW) | 0.81437 in | **NOT-DEPTH** | 0.4634 |
| class_centers.nearest_center_acc_test (R56-4c) | 0.9232 / 0.9218 / 0.9248 | [0.9188, 0.9278] | 0.9444 HIGH | 0.8322 / 0.8922 / 0.9158 (e40 LOW) | 0.92359 in | **NOT-DEPTH** | 0.4196 |
| `.meta.accuracy.test` (R56-N) | 0.9256 / 0.9226 / 0.9250 | [0.9196, 0.9286] | 0.9446 HIGH | 0.8384 / 0.8966 / 0.9160 | 0.92379 in | **NOT-DEPTH** | 0.4406 |

**Definitions.**
- Band sides are over B20: min, max and w = max − min. Outside means above max + w or below min − w.
- flip t* is the t at which m* reaches the band edge on the hub's side. DEPTH holds iff t > t*.
- Bridge ratio = mean of the 45 off-diagonal distances between `.class_centers.centers` / mean of `.class_centers.radius`. The B20 values reproduce Stage 1 (3.392 / 3.424 / 3.388).

**Every label is stable across the t range 0.260-0.336.**

**Caveat carried with the three collapse labels** (sep_ratio, bridge ratio, nc1).
- The matched rung e40 on its own sits on the *opposite* side of B20: it is less collapsed than resnet20, with `.meta.accuracy.ref` 0.9936 against about 0.9995. For nc1 this is marginal: e40 0.17744 against the edge 0.17687.
- DEPTH therefore comes entirely from the 27% hub weight. That weight draws a straight line across the gap from a 40-epoch last-epoch checkpoint to a 200-epoch best-epoch one.
- If collapse rises convexly near convergence, a straight line overstates m*.
- The nearest-rung-only reading would give NOT-DEPTH for all three; the rule reserves that reading for Boolean clauses.
- STAGE2.md calls DEPTH "the conservative finding" because short schedules push the rungs toward less collapse. Here that push is exactly what separates e40 from the hub, so the collapse labels are weaker than the ID label.

**The ID label does not depend on the interpolation.**
- The hub and every rung lie above B20.
- Penult ID decreases across the depth-56 models ordered by accuracy (11.98 → 11.74 → 10.88 → 10.67: three seed-11 rungs, then the chenyaofo hub; not one training trajectory) and never reaches resnet20's level.

## Predictions (predicted → observed → verdict)

### R56-1: ID profile (`.per_layer.<L>.twonn_id.id`)

**1a — PASS.** C1 holds at depth 56.
- h56 peaks at layer3.5 with 19.3939 (layer3.0 is 15.5720).
- Penult is 10.6695, a drop of 0.4499 ≥ 0.30.
- Nearest rung e40: peak layer3.5 19.5098, penult 10.8849, drop 0.4421.

**1b — PASS.**
- `c0` `.checks.id_profile_stability[0]`: spearman 0.964, peak layer3.1/layer3.1 (aligned; this is r56 layer3.5), shift 0.
- INFO: c1 1.000, c2 0.988; cb 0.964 / 0.988 / 0.988.

**1c [ACC] — FAIL.**
- Predicted: outside B20 on the LOW side.
- Observed: 10.6695, outside on the HIGH side (edge 10.3144).
- Label DEPTH.

**1d — PASS.**
- n56 peaks at layer3.0 with 28.3167; penult is 28.2796, 0.13% below the peak (the limit is 10%).
- The last-block drop is learned at depth 56 too.

### R56-2 [ACC] — PASS

- sep_ratio is HIGH, the bridge ratio is HIGH and nc1 is LOW: all on the predicted side (see the attribution table).
- Labels: DEPTH ×3.
- INFO: the r56 / r20 sep_ratio ratio is 5.3323 / 2.9962 = 1.780 (1.761 against the B20 mean). The legacy ratio is 4.12 / 2.87 = 1.436, a different quantity and not a target.

### R56-3: adjacency (r20 s0hub_st2 → r56 hub)

**3a — PASS.**
- `c0` penult/adjacency_spearman[0,1] rho = 0.858 (`d0` `.per_layer.penult.adjacency_spearman` = 0.85837).
- This is inside the expected 0.80-0.95.

**3b — PASS, by 0.0016.**
- `node scripts/d1_distance_only.js atlas_v1_resnet20_s0hub_st2:atlas_v1_resnet56_s0hub` gives 0.802 (0.80158 unrounded), which is ≥ 0.80.
- INFO, pairs not pre-registered:
  - s1_st2 0.803, and s2_st2 **0.771**.
  - st2 pairs 0.955 / 0.963 / 0.950, which reproduces Stage 1.
  - Nulls: r20→n56 0.326, h56→n56 0.236.
  - h56 → e40, at the same depth: 0.732.

**3c — PASS.**
- `c0` layer3.1/adjacency_spearman[0,1] = 0.857.
- `d0` `.per_layer.layer3.1.adjacency_spearman` = 0.85731, with `.layer_b` = layer3.5.

**3d — PASS.** `c1` 0.909 and `c2` 0.912. |Δ| from 3a is 0.050 and 0.054, both ≤ 0.10.

**3e (non-core) — PASS.**
- `c0` penult/merge_kendall[0,1] = 0.781 ≥ 0.60.
- The only merge FAIL in c0 is at layer2.2 (0.506).

### R56-4: decodability

**4a — PASS.**
- `h56` `.cross_layer.commit_layer.per_factor.{luminance_mean, highfreq_ratio, spectral_anisotropy}.washout` = 0.7567 / 0.1495 / 0.0759. These satisfy > 0.30, < 0.20 and < 0.20.
- e40: 0.5407 / 0.1291 / 0.0566.

**4b — PASS.**
- `c0` `.checks.decodability_stability`: all 10 sharp non-at-risk factors PASS, max MAD 0.056 (saturation_mean).
- The only FAILs are contrast_rms and noise_sigma, both at-risk.
- c1, c2 and cEe40 are also 10/10 on the sharp factors. c1's one FAIL, spectral_anisotropy, is not a sharp factor.

**4c [ACC] — PASS.**
- Class probe excess 0.83675 and nearest_center_acc_test 0.9444 are both outside on the HIGH side.
- Labels: NOT-DEPTH ×2.

### R56-5: class commit

**5a — PASS.**
- `h56` `.cross_layer.commit_layer.per_factor.class.commit_layer` = layer3.5, which is in {layer3.0, layer3.5, layer3.8}.
- The profile crosses the threshold at layer3.5: layer3.0 = 0.6650 < 0.9 × 0.83675 = 0.7531 ≤ layer3.5 = 0.7823.
- e40 also commits at layer3.5.

**5b — PASS.**
- The commit is at layer3.5, i.e. 24/27 = 8/9 of the blocks.
- `c0` commit/class PASS ['layer3.1', 'layer3.1'], shift 0. The same holds in c1 and c2.

**5c — PASS.**
- `b1` `.cross_layer.layer_cka.biggest_reorganization` = layer3.5→layer3.6 (drop 0.14134), which is in the predicted set.
- The next lowest consecutive CKAs are layer3.6→3.7 at 0.8826 and layer3.7→3.8 at 0.9141.
- INFO, stride 5: layer3.5→layer3.8 (0.34514).

### R56-6: cross-model agreement — PASS (all four clauses)

From `d0` `.per_layer.penult`:
- cka_test = 0.86343 ≥ 0.80 (`c0` panel_agreement PASS).
- Null floor: `dErand` cka_test = 0.02906 ≤ 0.50.
- error_consistency_test = 0.52623 > 0.30.
- relrep_argmax_agree_ood_c100 = 0.571 ≥ 3 × 0.123611 = 0.3708 (4.62× chance).
- For comparison, d1 / d2 cka_test = 0.86603 / 0.86443.

### R56-7: corruption claims (h56 penult)

**C4 — PASS, 6/6.**
- `.corruption_displacement.splits.corrupt__<c>__s<k>.coherence` for the three blur cases: defocus s5 0.3942, motion s3 0.3535, motion s5 0.3661.
- The noise cases they must stay below: gaussian 0.5731 (s3) / 0.6004 (s5), and shot 0.5485 / 0.5861.
- Minimum margin 0.1919 (B20: 0.113-0.132).
- e40 is 6/6 (margin 0.1248); n56 is 4/6.

**C5a — PASS, 8/8.**
- `.knn_density.splits.corrupt__<c>__s{1,5}.sparse_frac`; the smallest s5 − s1 difference is brightness at 0.0850.
- e40 is 8/8 (0.0530); e10 is 7/8; n56 is 3/8.

### R56-8: margin — PASS (scored after the evaluator pass; see Outcome)

- R56-8 is carried by A3 item E5, and A3's M1 margin clause passed (`a30729f`, `results/margin_v1_resnet20_s1/SESSION.md`, E5 lines).
- `results/margin_v1_resnet56_s0hub/atlas.json` `.per_layer.penult.margin_typeb.auc_margin_typeb` = 0.906687 ≥ 0.80; resnet20 hub 0.884793; |Δ| 0.0219 ≤ 0.05.
- A3 row: **SCALE-ROBUST** (STAGE2.md table: ≥ 0.80 in both, |Δ| ≤ 0.05).
- INFO: there is no matched-rung margin, and margin − dist is −0.0008 (p 0.83) at depth 56, so the M2 advantage does not transfer.
- The Stage 2 atlases exclude margin_typeb (`manifest_used.yaml` invariants ['-margin_typeb']).

### R56-9c — FAIL

**Predicted:** R56-1c and R56-2 are labelled NOT-DEPTH, and the shape claims hold in the matched rung.

**Attribution clause: fails.** All four fields are DEPTH.

**Shape clause: holds.** Evaluated in the nearest rung e40 (Δacc −0.0067):
- R56-1a: drop 0.442.
- R56-3a: `cEe40` penult rho 0.919.
- R56-3b: D1 r20 → e40 0.922.
- R56-4a: holds.
- R56-5a: commit at layer3.5.

**Readings:**
- The literal conjunction is FAIL; scored per clause it is PARTIAL.
- A strict reading that counts only a rung within 0.005 as "matched" makes the shape clause NOT_EVALUABLE.
- A nearest-rung-only m* keeps R56-1c DEPTH.
- So R56-9c is FAIL under every reading.

### b1 forecasts

**F1 — PASS.** The maximum |Δ| in twonn_id.id and in sep_ratio between `b1` and `h56` at the 11 shared taps is 0 (bit-identical).

**F2 — PASS.** Pearson(log10 nc1, block index) over the 28 `b1` taps without layer3.8 is −0.9726 ≤ −0.95 (Stage 1: −0.982 / −0.977 / −0.976).

**F3 — recorded.**
- (ID layer3.7 13.1756 − ID layer3.8 10.6695) / (peak layer3.5 19.3939 − penult 10.6695) = 0.2873.
- The layer3.6→3.7 step carries 60% of the drop.

**F4 — PASS.**
- From `b1` `.cross_layer.commit_layer.per_factor.class.profile` (best 0.83675), the class commit block by tau is:
  - tau 0.85: layer3.3;
  - tau 0.90: layer3.4;
  - tau 0.95: layer3.6;
  - tau 0.80 (recorded only): layer3.1.
- All of the scored ones are in stage 3.
- Relative depth is 0.815 / 0.852 / 0.926, against 0.889 / 0.889 / 1.000 for the r20 hub. That is 0.04-0.07 earlier, which is less than one resnet20 block.

### Rule 8 — not triggered

- The st2-pair maxima in `cb` are 0.973 for penult adjacency and 0.890 for cka_test.
- r20→r56 hub pairs (c0, c1, c2): adjacency 0.858 / 0.909 / 0.912; cka 0.863 / 0.866 / 0.864.
- Rung pairs (cE): adjacency 0.946 / 0.966 / 0.919; cka 0.766 / 0.853 / 0.881.
- None of these exceeds either maximum.
- Context: the r56 penult center-distance CV is 0.0370, against 0.0733 / 0.0685 / 0.0808 for B20 (derived from `.class_centers.centers`).
  - The hub is more collapsed than resnet20, yet it agrees with resnet20 less than the resnet20 seeds agree with each other.
  - So the collapse-makes-nets-agree artifact is not visible.

## Per-row scale labels (STAGE2.md table)

STAGE2.md defines ROBUST and FRAGILE in the table columns, and SCALE-VARIANT in the text ("the claim holds but a value leaves the band"). Where both apply, the pre-registration does not say which wins or which values count. Three readings are therefore given:

- **C:** columns only.
- **T:** SCALE-VARIANT when the claim holds and one of that row's pre-registered [ACC] fields leaves the band. The text says to attribute the value "with the rule above", and that rule is the [ACC] rule.
- **T+:** any atlas.json value of the row's quantity counts. This is supplementary, since it applies the rule to non-[ACC] fields.

| row | observed | C | T | T+ |
|---|---|---|---|---|
| 1 ID profile | C1 holds (hub, e40); spearman 0.964, shift 0; penult ID HIGH | SCALE-ROBUST | **SCALE-VARIANT (DEPTH)** | SCALE-VARIANT (DEPTH) |
| 2 sep_ratio | 5.332 | value entry: **DEPTH** | same | same |
| 3 washout, decod | C2 holds; 0/10 FAIL, max MAD 0.056; class excess HIGH | SCALE-ROBUST | **SCALE-VARIANT (NOT-DEPTH)** | SCALE-VARIANT (NOT-DEPTH; also anisotropy washout 0.0759 > edge 0.0748, m* 0.0618 inside) |
| 4 commit/class | layer3.5, stage 3, shift 0 | SCALE-ROBUST | SCALE-ROBUST | SCALE-ROBUST |
| 5 C4 | 6/6; all 7 coherence values inside B20 | SCALE-ROBUST | SCALE-ROBUST | SCALE-ROBUST |
| 6a C5a | 8/8 | SCALE-ROBUST | SCALE-ROBUST | SCALE-VARIANT (14/16 sparse_frac HIGH: 10 NOT-DEPTH, 4 DEPTH; clean test sparse_frac NOT-DEPTH) |
| 7 adjacency, hub reference (R56-3b as written) | rho 0.858, D1 0.8016, layer3.1 0.857 | SCALE-ROBUST (marginal) | SCALE-ROBUST | SCALE-VARIANT (penult CV 0.0370 LOW; m* 0.0697 inside: NOT-DEPTH) |
| 7 adjacency, all three B20 references | D1 0.8016 / 0.8026 / 0.7709; layer3.1 0.857 / 0.886 (s1) | **UNDECIDED** (s2 fails ROBUST, not FRAGILE) | — | — |
| 8 relrep c100 | 4.62× chance | SCALE-ROBUST | SCALE-ROBUST | SCALE-ROBUST |
| – cka_test | 0.863 | SCALE-ROBUST | SCALE-ROBUST | SCALE-ROBUST |
| A3 margin (E5) | 0.907 vs r20 hub 0.885, |Δ| 0.022 | SCALE-ROBUST | — | — |

- No row is SCALE-FRAGILE under any reading.
- **Row 7 depends on the reference.** Row 7's ROBUST column ("distance-only >= 0.80") names no reference run, and STAGE2.md "Local evaluation" lists the D1 computation for `_s1_st2` and `_s2_st2` as well.
  - Hub reference only (R56-3b as written): SCALE-ROBUST, marginal (D1 0.8016).
  - All three B20 references: D1 is 0.8016 / 0.8026 / 0.7709. s2 fails ROBUST but is not FRAGILE (0.771 ≥ 0.70, layer3.1 ≥ 0.70), so the row is UNDECIDED.
  - COLLAPSE-ARTIFACT applies when only penult drops (layer3.1 does not) and the r56 penult CV is < 0.068. If "drops" is read as "below the ROBUST threshold 0.80", the s2 reference gives COLLAPSE-ARTIFACT (CV 0.0370, layer3.1 ≥ 0.70). If it is read as reaching the FRAGILE threshold 0.70, it does not apply.

## What transfers and what is depth-specific

**Transfers** (relational and shape claims):
- **ID profile shape.** The peak is at 8/9 relative depth and penult sits about 45% below it (r20: 0.49-0.50). Aligned spearman is 0.96-1.00. The null is flat (0.13%).
- **Class commit.** At stride 5 it is at layer3.5 (8/9; the stride-5 tap spacing) with shift 0. At stride 1 it is one resnet56 block earlier (23/27 = 0.852), still inside resnet20's resolution interval (0.778, 0.889]. It is in stage 3 at every tau.
- **Biggest reorganization.** It comes right after the ID peak at both depths (r56 layer3.5→3.6; r20 layer3.1→3.2).
- **nc1.** It decays log-linearly with depth (−0.973).
- **Decodability and corruption claims.**
  - C2 washout and the decod profiles hold (10/10 sharp factors).
  - C4 holds at 6/6. The hub's min margin 0.192 is above B20's 0.113-0.132, but that is not depth: matched rung e40 0.125, m* 0.143 inside the band [0.094, 0.151] (NOT-DEPTH, supplementary).
  - C5a holds at 8/8.
- **Cross-model agreement.**
  - Class adjacency is ≥ 0.70, and D1 is ≥ 0.80 against the hub reference, though only just (s2 reference 0.771).
  - cka_test 0.863, error consistency 0.526, relrep 4.6× chance.

**Depth-specific at matched accuracy (DEPTH):**
- **Penult ID level is higher:** 10.67 against 9.76-10.04, and every rung is higher too. This is robust to t.
- **Penult collapse is stronger:** sep_ratio, bridge ratio and nc1. This rests on the e40-hub interpolation (see the caveat above).

**Not attributable to depth (NOT-DEPTH):**
- Class probe excess, nearest-center test accuracy, and test accuracy (the last by construction).
- Supplementary, non-[ACC]:
  - most of the C5a sparse_frac level shift;
  - the anisotropy washout excess;
  - the penult CV.

**Weaker agreement across depth than across seeds** (supplementary; pair metrics have no pre-registered band, so the st2 pair values are used as a band):
- r20→r56 penult adjacency 0.858 against a low edge of 0.905. m* = 0.903, which is DEPTH by 0.0026.
- Layer3.1 adjacency 0.857 against an edge of 0.924. m* = 0.865, DEPTH.
- D1 0.802 against an edge of 0.936. m* = 0.889, DEPTH.
- cka 0.863 against an edge of 0.8765. m* = 0.87657, inside by 0.00006, so NOT-DEPTH.
- Error consistency and merge tau: the hub is inside the pair band, which is UNRESOLVED read literally.

## Null at depth 56 (INFO)

**cn** is DOES_NOT_REPLICATE (40 PASS / 104 FAIL):
- id_profile 0.842 FAIL (peaks at layer3.5 vs layer3.0);
- commit/class FAIL (layer3.5 vs stem);
- penult adjacency 0.344 and cka 0.030;
- one decod PASS: luminance_mean, MAD 0.070.

**New at depth 56: the null clears C2's luminance clause.**
- `n56` luminance_mean washout is 0.473, above 0.30 (the r20 null was 0.207).
- C2 still fails in the null, but only through highfreq (0.278 > 0.20). At depth 56, C2's luminance clause alone does not separate learned from random.
- The null's stem luminance probe value is 0.999, so row 3 again has to cite penult luminance, not the stem R².

**Other claims.** n56 reaches C4 4/6 and C5a 3/8 (r20 null: 3/6 and 4/8).

## Holdout

- **Only the depth axis is a clean holdout for R56-4.** The decod pools include the five confirmation corruptions (`atlas/invariants/decodability.py:30`).
- **Confirmation-corruption accuracies** (`.meta.accuracy.corrupt__{impulse_noise,…}`) were displayed during evaluation. They are not used and seed no claims.
- **Seeds.** Depth-56 seeds 1 and 2 remain unused.
- **Margin.** No margin-like quantity is reported from the main atlases.

## Prediction misses

1. **R56-1c (sign).** Predicted: penult ID below B20. Observed: above B20 at the hub (10.67) and at every rung. The relative drop (C1) transfers; the level does not decrease with depth.
2. **R56-9c.** Predicted: the depth-56 differences in collapse and ID are accuracy or fit effects (NOT-DEPTH). All four came out DEPTH under the pre-registered interpolation.
3. **Ladder design.**
   - E ∈ {10, 20, 40} was meant to bracket 0.9259, but no rung landed within 0.005 (e40 is at −0.0067).
   - The upper bracket therefore had to be the test-selected 200-epoch hub. That is the source of the collapse labels' weakness.
4. **Edge numbers not forecast.**
   - R56-3b passes by 0.0016, and the non-registered s2_st2 pair (0.771) would fail.
   - The r56 penult centers are near-equidistant (CV 0.037), which makes their distance ranks fragile.
5. **Null.** The depth-56 null clears C2's luminance clause (0.473).
6. **Normalization.** R56-N passes, but at depth 56 the accuracy test does not separate chenyaofo from cifar_true (p = 0.367).
7. **Expected values** (context, not thresholds). Cross-depth agreement falls below the Stage 1 seed-pair values:
   - cka 0.863 vs 0.883-0.890;
   - layer3.1 adjacency 0.857 vs 0.945-0.966;
   - D1 0.802 vs 0.950-0.963.
8. **Hits:**
   - R56-2 sides; R56-1a, 1b, 1d; R56-3a-e (3a within its expected range); R56-4a-c;
   - R56-5a-c (5c layer3.5→3.6); R56-6; R56-7; R56-8 (via A3 E5);
   - F1, F2, F4.

## ATLAS_STATUS changes (to apply in the same commit as this SESSION.md, after `a30729f`; the owner's commit is the human approval)

**What Stage 2 does not do.**
- Stage 2 promotes nothing. STAGE2.md: one depth-56 instance; nothing is promoted to ✅ from Stage 2.
- It also has no demotion rule. Status symbols are unchanged:
  - rows 1, 3, 4, 5, 6a and 7 (penult adjacency) keep the ✅ that Stage 1b gave them for resnet20;
  - rows 2 and 8 stay 🟡;
  - row 6b stays ✗, and row 9 keeps the ✅ (s1-s4) that A3 gave it in `a30729f`.
- Claim texts are not widened to other depths. The depth-56 scope of every row is 🟡 (measured once).

**What it licenses:** a depth-56 evidence entry with a scale label in each row, and one header sentence.

**Row edits (evidence column):**
- **Row 1:**
  - C1 holds at depth 56: peak layer3.5 (8/9), drop 0.450 (e40 0.442).
  - Aligned spearman 0.964, shift 0.
  - Penult ID 10.67 is above resnet20 at matched accuracy (m* 10.83 against an edge of 10.31): DEPTH. R56-1c FAIL.
  - Null drop 0.001.
  - Scale label: SCALE-VARIANT (DEPTH); SCALE-ROBUST by the table columns.
- **Row 2:**
  - r56 hub 5.332, with m* 3.528 above the edge 3.137: DEPTH.
  - e40 alone is 2.853, below the band; DEPTH needs t > 0.115, and t is 0.272.
- **Row 3:**
  - Depth 56: C2 holds (0.757 / 0.150 / 0.076; e40 0.541 / 0.129 / 0.057), and all 10 sharp factors PASS aligned (max MAD 0.056).
  - Penult class probe excess 0.837 is above the band: NOT-DEPTH.
  - The depth-56 null has luminance washout 0.473, so at depth 56 only the highfreq clause separates learned from random.
  - Scale label: SCALE-VARIANT (NOT-DEPTH); SCALE-ROBUST by the table columns.
- **Row 4:**
  - Depth 56: layer3.5 (8/9), aligned shift 0 against hub, s1 and s2.
  - Stride 1: layer3.4 (23/27); stage 3 for tau 0.85-0.95.
  - Scale label: SCALE-ROBUST.
- **Row 5:**
  - Depth 56: 6/6, minimum margin 0.192 (e40 0.125; m* 0.143 inside the resnet20 band: the larger hub margin is NOT-DEPTH; null 4/6).
  - Scale label: SCALE-ROBUST.
- **Row 6a:**
  - Depth 56: 8/8, smallest margin brightness 0.085 (e40 8/8; null 3/8).
  - Scale label: SCALE-ROBUST. The values sit above B20, which is SCALE-VARIANT, mostly NOT-DEPTH, under the any-value reading.
- **Row 7:**
  - Depth 56 (r20 hub → r56 hub): rho 0.858, D1 0.802 (s1_st2 / s2_st2 references 0.803 / 0.771, computed per STAGE2.md, no threshold), aligned layer3.1 0.857.
  - r56 penult CV 0.037.
  - Scale label: SCALE-ROBUST, marginal, against the hub reference (R56-3b as written); UNDECIDED against all three resnet20 references (s2 D1 0.771).
- **Row 8:**
  - Depth 56: 0.571 vs chance 0.124, i.e. 4.62× (e40 4.42×; null 1.30×).
  - Scale label: SCALE-ROBUST.
- **Row 9:** symbol unchanged (✅ s1-s4). Depth 56 (R56-8 / A3 E5): 0.907 vs r20 hub 0.885, SCALE-ROBUST; margin − dist −0.0008 (p 0.83), no matched-rung margin. This is already in the row's evidence from `a30729f`; add the scale label.

**Header sentence:** "Stage 2 (A4, resnet56 hub; docs/plans/STAGE2.md) is valid and promotes nothing (results/atlas_v1_resnet56_s0hub/SESSION.md). C1-C5a, the class commit (stage 3; stride-1 layer3.4 = 23/27), cross-model agreement (penult cka_test 0.863) and the margin AUC (0.907) transfer to depth 56. At matched accuracy, the penult ID is higher and the penult collapse (sep_ratio, bridge ratio, nc1) is stronger (DEPTH; R56-1c and R56-9c FAIL). The collapse labels rest on interpolating between the 40-epoch rung and the test-selected hub. Row 7 is SCALE-ROBUST against the hub reference only (D1 0.802; s2 reference 0.771)."

**If Stage 1b ends KILL or INVALID,** these entries are marked INFO and carry no label.

## Next actions

1. **Commit and push.** Commit the pulled Stage 2 results, this SESSION.md and the ATLAS_STATUS evidence entries after `a30729f`, then push all three decisions. P → Stage 1b decision → A3 → Stage 2 is the commit order.
2. **Firm up the collapse labels.** Each of these needs a new pre-registration before any run:
   - a seed-11 rung that lands within 0.005 of 0.9259 (step 1). The E value is a guess, for example 60 or 80; the rule does not depend on it;
   - the full-recipe depth-56 seeds 1 and 2 (200 epochs, last epoch; STAGE2.md "Not covered"), which also give a last-epoch counterpart to the hub;
   - the estimator twin `atlas_v1_resnet56_s0hub_ref1` (about 1.6 min).

   The two DEPTH findings (penult ID level, penult collapse) are candidates for such a pre-registered claim. They are not claims yet.
3. **Future pre-registrations** should state:
   - whether SCALE-VARIANT overrides the ROBUST column, and which values count;
   - a band for pair metrics (adjacency, cka, D1), if those are to be attributed.
