# SESSION: Stage 1b (v1) — confirmation seeds 3, 4

This is the Evaluator pass on the pulled Stage 1b results; the dumps stay on the RunPod volume.

- **Pre-registration:** `experiments/queue/atlas_v1_resnet20_s3.yaml` notes. They cover I0-I2, R0, S1-S11, T2, D1, C1-C5a, F1-F4 and the frozen coverage tables. They were committed and pushed as **P = `664bd25`** before the run.
- **Decision procedure:** `docs/plans/STAGE1.md` amendment 1 (the CORE-PASS row) and amendment 2 item 9. It is applied in order: INVALID → KILL → PARTIAL → U → PASS / CORE-PASS → promotion.
- **How it was checked:** three evaluator families recomputed every prediction from the result files with node. This pass then re-checked every number the decision uses:
  - all 80 scalar items of the s3-s4 critic, recomputed from the atlases with the `critic.py:145-164` rule (0 status mismatches);
  - the frozen twin table, recomputed from s1 / s1_ref1 with the amendment-1 thresholds (identical: 31 items, 18 of them abs-only);
  - penult adjacency and the ID profiles;
  - D1, re-run with the committed script.

**Decision: CORE-PASS.**
- Not INVALID, and no KILL.
- 6/6 core items pass.
- U = {stem/pca_spectrum.participation_ratio, layer3.1/neural_collapse.nc1}.
- Under amendment 2 item 5, rows 1, 3, 4, 5, 6a and 7 (penult adjacency) are promoted to ✅.

Path shorthands:

| shorthand | path |
|---|---|
| `c34` | `results/critic_v1_resnet20_s3_s4/critic.json` ([0,1] = s3, s4) |
| `c034` | `results/critic_v1_resnet20_s0_s3_s4/critic.json` ([0,1] = hub-s3, [0,2] = hub-s4) |
| `c1234` | `results/critic_v1_resnet20_s1_s2_s3_s4/critic.json` (INFO) |
| `a3`, `a4` | `results/atlas_v1_resnet20_s{3,4}/atlas.json` |
| `d34` | `results/atlas_v1_resnet20_s4/compare_vs_atlas_v1_resnet20_s3/deformation.json` |
| `dh3`, `dh4` | `results/atlas_v1_resnet20_s{3,4}/compare_vs_atlas_v1_resnet20_s0hub/deformation.json` |
| `ic` | `results/instrument_check_stage1b/` |
| `t3`, `t4` | `results/train_resnet20_s{3,4}/train.json` |

## Run facts

**Runs.** The runs are `atlas_v1_resnet20_s3` and `_s4`.
- Source is `real` in `a3`/`a4` `.source`, and norm is `chenyaofo`.
- `.meta.git_commit` is 664bd25 (= P) for both.
- `.meta.created` is 2026-09-23 05:44:19 for s3 and 05:45:41 for s4 (UTC).
- `.meta.weights`: `resnet20_s3_chenyaofo.pt sha256:9b06806321d39155` and `resnet20_s4_chenyaofo.pt sha256:a67852e591b4618e`.

**Training.**
- `t3`/`t4` `.final_test_acc_10k`: 0.926 / 0.9301.
- `.torch` is 2.8.0+cu128 and `.device` is NVIDIA GeForce RTX 4090 for both, the same as `results/train_resnet20_s1/train.json`.
- `.meta.accuracy.test` (test[:5000]): 0.9222 / 0.9286.

**Critic counts** (`.counts`):

| critic | PASS | FAIL | WARN | INFO |
|---|---|---|---|---|
| s3-s4 (146 items) | 133 | 11 | 1 | 1 |
| hub-s3-s4 | 209 | 17 | 1 | 3 |
| s1..s4 (INFO) | 325 | 23 | 1 | 6 |

The critic's own verdict word (PARTIAL) is advisory and is not the decision.

**Not evaluated here.** Stage 2 (`results/atlas_v1_resnet56_s0hub`) and A3 (`results/margin_v1_resnet20_s1`) contain only `RUN_REQUEST.md` in this checkout. Amendment 2 item 7 puts them after this decision is committed.

## Predictions (predicted → observed → verdict)

**I — instrument (any failure makes the run INVALID)**

- **I0**
  - Predicted: torch and GPU equal s1's; the Stage A inputs are unchanged since 6242a17; the s3 and s4 manifests agree in every instrument key.
  - Observed:
    - `.torch` is 2.8.0+cu128 and `.device` is RTX 4090 in the train.json of s1, s3 and s4.
    - `git diff --stat 6242a17 664bd25` over the seven Stage A paths is empty, and so is the diff of P against the worktree.
    - Outside notes and comments, the queue manifests differ only in `exp_id`, `backbone.weights`, `backbone.seed_tag` and `outputs.root`.
  - Verdict: **PASS**.
  - I0 writes no JSON: `check_rebuild.py manifests` only prints. The pod log reads "I0 PASS", and the three facts above corroborate it independently.

- **I1**
  - Predicted: `check.json` and `replay.json` both PASS.
  - Observed in `ic/check.json`:
    - `.status` PASS, `.bitwise_identical` true.
    - `.n_leaves` = `.n_exact` = 38078, `.n_mismatch` 0.
    - `.errors` [], `.skipped` {}, `.git_commit` 664bd25.
  - Observed in `ic/replay.json`:
    - `.status` PASS.
    - `.compare.n_leaves` 1218, `.compare.n_mismatch` 0.
    - `.n_critic_items` 146, `.n_critic_mismatch` 0, `.critic_new_items_INFO` [].
  - Verdict: **PASS**.
  - Reading note:
    - The re-run critic carries one tolerance key that the committed s1_s2 critic lacks: `scalar_abs_tol`. It applies only to margin_typeb AUCs and was added in P itself (`experiments/tolerances_default.yaml:18`).
    - Every committed tolerance is reproduced; `check_rebuild.py:130-131` compares the committed keys. No Stage 1 item uses the new key.
    - The item-4 gate is replay.json `.status`, which is PASS under both the "every committed tolerance" reading and the "identical tolerance dict" reading.

- **I2**
  - Predicted: `i2.json` status PASS.
  - Observed:
    - `ic/i2.json` `.status` PASS, with `.runs.*.problems` [] for both runs.
    - All 11 meta keys equal s1's (recomputed independently).
    - The sha256 values 9b068063… (s3) and a67852e5… (s4) differ from s1 (d5442d0e…) and s2 (e7e5386f…).
  - Verdict: **PASS**.

**R0**
- Predicted: s3 and s4 both in [0.915, 0.930], and |s3 − s4| ≤ 0.005.
- Observed: `t3` 0.926; `t4` 0.9301, which is 9301/10000, 0.0001 above the band (one test image). |d| = 0.0041.
- Verdict: **PARTIAL**. The range clause fails for s4 and the gap clause holds; as one conjunctive claim it is a FAIL.
- A reading that rounds to the band's 3 dp would PASS; it was rejected in favour of the literal value.
- This is not an INVALID condition, which needs < 0.90 or |d| > 0.010. The gap is 0.0041 at 10k and 0.0064 at 5k (`.meta.accuracy.test`).

**S — s3 vs s4** (`c34`, `d34`)

- **S1 (core)** ID profile.
  - Observed: `.checks.id_profile_stability[0]` spearman 0.988 (recomputed 0.9879); peak layer3.1 in both, shift 0. `a3`/`a4` `.per_layer.layer3.1.twonn_id.id` = 19.790 / 19.503.
  - Verdict: **PASS**.
  - Stage 1 caveat: the null reaches 0.867, so this item barely discriminates.
- **S2** penult dim95 and the location of dim95/PR FAILs.
  - Observed: penult dim95 is 9/9 (`.checks.scalar_stability[74]`). There are two dim95/PR FAILs:
    - stem PR, rel 0.156 at [1], where FAILs are allowed;
    - layer3.1 PR, 12.610 / 9.453, rel 0.286 at [65], where they are not.
  - Verdict: **PARTIAL**. Clause 2 fails; as one claim it is a FAIL.
- **S3 (core)** penult sep_ratio and nearest_center_acc_test.
  - Observed: penult sep_ratio 2.970 / 3.024, rel 0.018 at [75]. nearest_center_acc_test passes at 10/10 layers; the largest rel is 0.033 at layer3.1 (0.796 / 0.822, [71]).
  - Verdict: **PASS**.
- **S4** scalar FAIL budget and S4 cells.
  - Observed: 8 of 80 scalar items FAIL, against a budget of ≤ 8, so it is met with zero margin. The S4 cells:
    - stem nc1 FAIL, rel 0.668 [4];
    - layer1.0 nc1 FAIL, rel 0.269 [12];
    - stem skew PASS, 0.103 [6];
    - layer1.0 skew PASS, 0.141 [14].
  - Verdict: **PASS**.
- **S5 (core)** penult adjacency.
  - Predicted (s3 notes, verbatim): penult/adjacency_spearman[0,1] PASS (rho >= 0.70).
  - Observed: `.checks.adjacency_stability[18]` rho 0.965. Recomputed as 0.96509 over the 45 upper-triangle entries of `a3`/`a4` `.per_layer.penult.class_adjacency.sep_matrix`.
  - Verdict: **PASS**.
- **S6** merge order.
  - Observed: `.checks.adjacency_stability[19]` tau 0.871.
  - Verdict: **PASS**. first_merge_same is not used as evidence.
- **S7** decodability.
  - Observed: `.checks.decodability_stability[0..17]`: 18/18 PASS. The largest MAD is 0.038 (luminance_mean and contrast_rms). There are 0 profile FAILs; the lowest rho is 0.81 (hue_cos).
  - Core part, MAD over the 10 sharp factors (max 0.038):

    | factor | MAD |
    |---|---|
    | luminance_mean [0] | 0.038 |
    | saturation_mean | 0.036 |
    | hue_sin | 0.032 |
    | orientation_entropy | 0.027 |
    | spectral_slope | 0.021 |
    | colorfulness | 0.021 |
    | severity | 0.020 |
    | class | 0.012 |
    | corruption_type | 0.011 |
    | corruption_family | 0.009 |

  - Verdict: **PASS**.
- **S8** commit agreement.
  - Observed: `.checks.commit_agreement[13]` commit/class is layer3.1 / layer3.1 (PASS). There are 3 commit FAILs, all in the S8 set:
    - highfreq_ratio [2]: layer1.0 / layer2.0;
    - orientation_entropy [11]: layer3.0 / layer2.0;
    - severity [17]: layer3.0 / layer2.0.
  - Verdict: **PASS**.
- **S9 (core)** penult cka_test.
  - Observed: `.checks.panel_agreement[0]` 0.889. `d34` `.per_layer.penult.cka_test` is 0.88926, which is > 0.691.
  - Verdict: **PASS**.
- **S10 (INFO)** CIFAR-100 relrep argmax agreement.
  - Observed: `d34` `.per_layer.penult.relrep_argmax_agree_ood_c100` is 0.5805, against 3 × `.relrep_argmax_chance_ood_c100` (0.1236) = 0.3708, i.e. 4.70× chance. `.error_consistency_test` is 0.581 and `.relrep_offmax_corr_test` is 0.831.
  - Verdict: **PASS** (exploratory).
- **S11** input norm and aliasing.
  - Observed: `.checks.norm_consistency[0]` PASS. `.checks.alias_layers` holds one item, alias:layer3.2 WARN.
  - Verdict: **PASS**.

**T2**
- Observed: `c034` `.checks.adjacency_stability[54]` hub-s3 0.973 and [56] hub-s4 0.933, against `c34` 0.965. |Δ| = 0.008 and 0.032.
- Verdict: **PASS**.

**D1** (NEW; gates row 7)
  - Predicted (s3 notes, verbatim): distance-only penult adjacency (Spearman over the 45 upper-triangle center distances, node scripts/d1_distance_only.js): s3-s4 >= 0.80.
- Observed with `node scripts/d1_distance_only.js`, which reads `.per_layer.penult.class_centers.centers`:

  | pair | D1 |
  |---|---|
  | **s3-s4** | **0.919** |
  | hub-s3 | 0.923 |
  | hub-s4 | 0.958 |
  | s1-s2 | 0.950 |
  | s1-s3 | 0.867 |
  | s1-s4 | 0.958 |
  | s2-s3 | 0.911 |
  | s2-s4 | 0.964 |

- Verdict: **PASS** (≥ 0.80).
- The same script reproduces the Stage 1 anchors: hub-s1 0.955, hub-s2 0.963, twin 0.993, null s1-rand 0.644. INFO: s3-rand 0.632, s4-rand 0.589.

**C — each must hold in s3 AND s4** (`a3`, `a4`)

- **C1** [row 1]
  - Predicted (s3 notes, verbatim): TwoNN ID peaks at layer3.0 or layer3.1 and penult ID is >= 30% below the peak.
  - Observed in `.per_layer.<L>.twonn_id.id`: the peak is layer3.1 in both (19.790 / 19.503) and penult is 9.836 / 9.901, so the drop is 0.503 / 0.492.
  - Verdict: **PASS**.
- **C2** [row 3]
  - Predicted (s3 notes, verbatim): washout at tau 0.9 (cross_layer.commit_layer.per_factor.<f>.washout): luminance_mean > 0.30; highfreq_ratio < 0.20; spectral_anisotropy < 0.20.
  - Observed in `.cross_layer.commit_layer.per_factor.<f>.washout` at tau 0.9:

    | seed | luminance | highfreq | anisotropy |
    |---|---|---|---|
    | s3 | 0.549 | 0.133 | 0.086 |
    | s4 | 0.644 | 0.161 | 0.069 |

  - Verdict: **PASS**.
- **C3** [row 4]
  - Predicted (s3 notes, verbatim): cross_layer.commit_layer.per_factor.class.commit_layer in {layer3.0, layer3.1, layer3.2} at tau 0.9 (layer3.2 == penult).
  - Observed: `.cross_layer.commit_layer.per_factor.class.commit_layer` is layer3.1 / layer3.1.
  - Verdict: **PASS**.
- **C4** [row 5]
  - Predicted (s3 notes, verbatim): penult corruption_displacement coherence, same severity: defocus_blur s5, motion_blur s3 and motion_blur s5 each below both gaussian_noise and shot_noise at that severity.
  - Observed in `.per_layer.penult.corruption_displacement.splits.corrupt__<c>__s<k>.coherence`: 6/6 comparisons hold in both. The minimum margin is 0.105 in s3 (defocus s5 0.444 vs shot s5 0.549) and 0.117 in s4.
  - Verdict: **PASS**.
- **C5a** [row 6a]
  - Predicted (s3 notes, verbatim): penult knn_density sparse_frac s5 > s1 for each of defocus_blur, motion_blur, snow, fog, brightness, contrast, pixelate, jpeg_compression.
  - Observed in `.per_layer.penult.knn_density.splits.corrupt__<c>__s{1,5}.sparse_frac`: 8/8 in both. The smallest s5 − s1 margin is brightness in both seeds: 0.0515 in s3 and 0.0395 in s4.
  - Verdict: **PASS**.
  - INFO for row 6b, which is not re-tested (s1 / s3 / s5):

    | seed | gaussian | shot |
    |---|---|---|
    | s3 | 0.324 / 0.376 / 0.318 | 0.231 / 0.397 / 0.387 |
    | s4 | 0.338 / 0.557 / 0.634 | 0.236 / 0.505 / 0.614 |

**F — forecasts** (scored; they never change the outcome)

- **F1**
  - Predicted: the outcome is CORE-PASS, and U contains at least one of the 49 scalar items that are neither twin-covered nor S4 cells.
  - Observed: the outcome is CORE-PASS, and both U items are among those 49.
  - Verdict: **PASS**.
- **F2**
  - Predicted: S4 exceeds its budget again, and S2's clause fails again.
  - Observed: S4 is at 8 of ≤ 8, so the budget is not exceeded. S2's clause does fail (layer3.1 PR).
  - Verdict: **PARTIAL**; as one claim, FAIL.
  - Rejected reading: `c034` has 12 scalar FAILs, but S4 is defined on s3-s4.
- **F3**
  - Predicted: at least 2 of the 4 uncovered Stage 1 FAILs recur.
  - Observed: all four PASS in `c34` `.checks.scalar_stability`:
    - layer1.0 PR, rel 0.086 [9];
    - layer1.0 dim95, 7 / 7 [10];
    - layer1.1 skew, rel 0.019 [22];
    - layer3.0 PR, rel 0.100 [57].

    They also all PASS in `c034`, so 0 of 4 recur.
  - Verdict: **FAIL**.
- **F4**
  - Observed: no C claim fails in s3 or s4.
  - Verdict: **NOT_EVALUABLE**. The antecedent is false; read as a material conditional, F4 is vacuously true.
  - The margins it named stayed the tightest: C2 highfreq is 0.067 below 0.20 in s3 and 0.039 in s4, and C5a brightness is 0.0515 and 0.0395.

**Tally:** 21 PASS, 3 PARTIAL (R0, S2, F2), 1 FAIL (F3), 1 NOT_EVALUABLE (F4). Scored conjunctively (one failed clause fails the prediction): 21 PASS, 4 FAIL (R0, S2, F2, F3), 1 NOT_EVALUABLE (F4). No decision consequence.

## Decision (amendment 2 item 9, in order)

**1. INVALID: no.**
- Item-4 conditions:
  - I0, I1 and I2 are PASS.
  - No check in `c34` or `c034` reports "check crashed": "crash" has 0 matches, and the one WARN is alias:layer3.2.
  - All 146 item names of `results/critic_v1_resnet20_s1_s2/critic.json` are present in `c34` after mapping s1 → s3 and s2 → s4, with 0 missing and 0 extra.
- Item-9 conditions:
  - `a3`/`a4` `.skipped` is {}, and no error-like key exists anywhere in either atlas.
  - Accuracy is ≥ 0.90 in both runs, and |d| = 0.0041 ≤ 0.010.
  - In `c34`, input_norm (`.checks.norm_consistency[0]`) and `.checks.synthetic_refusal[0]` are PASS.
  - `.per_layer.penult.cka_test` is present in `d34` (0.8893), `dh3` (0.8872) and `dh4` (0.8878).

**2. KILL: no.** No condition holds even in s3-s4, so the hub conjunct is moot. The hub pairs clear every threshold as well.

| condition | s3-s4 | hub-s3 | hub-s4 |
|---|---|---|---|
| penult adjacency (KILL < 0.50) | 0.965 | 0.973 (`c034` adjacency_stability[54]) | 0.933 ([56]) |
| ID spearman (< 0.80) / peak shift (≥ 2) | 0.988 / 0 | 0.988 / 0 (id_profile_stability[0]) | 1.000 / 0 ([1]) |
| sharp factors with MAD > 0.10 (≥ 2) | 0 (max 0.038) | 0 (max 0.040, colorfulness) | 0 (max 0.052, colorfulness) |
| class commit shift (≥ 2) | 0 | 0 (`dh3` `.commit_layer.class` layer3.1 → layer3.1) | 0 (`dh4`, same) |
| penult cka_test (< 0.60) | 0.889 | 0.887 (panel_agreement[0]) | 0.888 ([2]) |

The Stage 1 twin passed every core item and is not re-run.

**3. PARTIAL: no.** Core items are 6/6 PASS:
- S1 id_profile[0,1]
- S3 penult sep_ratio
- S5 penult adjacency (0.965)
- S7-core, max MAD 0.038 ≤ 0.10
- S8 commit/class
- S9 penult cka_test (0.889)

**4. U.** `c34` has 11 FAILs, all non-core: 8 scalar, 3 commit and 0 decod. The twin columns come from s1 vs s1_ref1, `.per_layer.<L>.<inv>.<key>`, with the amendment-1 arms rel ≥ 0.075 or abs ≥ 0.025. Amendment 2 item 3 requires naming the arm for every covered FAIL.

| `c34` item | s3 / s4 (rel) | coverage |
|---|---|---|
| stem nc1 `.checks.scalar_stability[4]` | 318.8 / 159.2 (0.668) | twin, rel and abs arms (twin rel 0.308), and an S4 cell |
| layer1.0 nc1 [12] | 123.1 / 93.9 (0.269) | twin, rel and abs arms (rel 0.182), and an S4 cell |
| layer1.1 nc1 [20] | 69.8 / 86.8 (0.218) | twin, rel and abs arms (rel 0.240) |
| layer1.2 hubness skew [30] | 0.767 / 0.922 (0.183) | twin, rel and abs arms (rel 0.094, abs 0.071) |
| layer2.0 nc1 [36] | 24.57 / 20.88 (0.163) | twin, **abs arm only** (rel 0.029, abs 0.673) |
| layer3.1 PR [65] | 12.61 / 9.45 (0.286) | twin, **abs arm only** (rel 0.009, abs 0.106) |
| commit/highfreq_ratio [2], orientation_entropy [11], severity [17] | shift 3 each | S8 set |
| **stem PR [1]** | 2.644 / 3.090 (0.1555, abs 0.446) | **none**: twin 2.993 / 2.991, rel 0.0006 |
| **layer3.1 nc1 [68]** | 0.923 / 0.792 (0.1530, abs 0.131) | **none**: twin 0.774 / 0.768, rel 0.008 |

**U = {stem/pca_spectrum.participation_ratio, layer3.1/neural_collapse.nc1}.** Neither item failed in s1-s2: `results/critic_v1_resnet20_s1_s2/critic.json` gives rel 0.006 and 0.104. Both fail in `c034`, with rel 0.154 and 0.199.

**5. Outcome: CORE-PASS**, because U is non-empty. The U families, pca_spectrum (PR at stem) and neural_collapse (nc1 at layer3.1), stay 🟡 and get a discovery sweep.

Reporting only: under a relative-arm-only reading of twin coverage, layer2.0 nc1 and layer3.1 PR would also be uncovered (U = 4); the frozen table is applied as written, so the outcome and promotions are unchanged. layer3.1 PR joins the discovery sweep (it failed in s1-s2 and s3-s4).

## Promotion (amendment 2 item 5)

**Preconditions (all hold):**
- `c34` `.checks.synthetic_refusal[0]` PASS, and `.checks.holdout_hygiene[0]` and `[1]` PASS ("declared").
- `.meta.git_commit` is 664bd25 = P for both runs. `git diff --quiet P <git_commit>` over the four pre-registration files exits 0 trivially, and the check of P against the worktree also exits 0.
- `.meta.created` is 05:44:19 and 05:45:41 UTC. Both are later than P's commit time, 2026-09-22T22:30:47-07:00 = 2026-09-23 05:30:47 UTC.
- The predictions are quoted from the s3 notes (lines 108-118).

| row | item | s1, s2 (Stage 1) | s3, s4 | status |
|---|---|---|---|---|
| 1 | C1 | held | held | ✅ |
| 3 | C2 | held | held | ✅ |
| 4 | C3 | held | held | ✅ |
| 5 | C4 | held | held | ✅ |
| 6a | C5a | 8/8 (post-hoc support) | held (confirmation) | ✅ |
| 7 | S5 and D1, s3-s4 | — | 0.965 PASS; 0.919 ≥ 0.80 | ✅ penult adjacency; merge order stays 🟡 |
| 2, 8 | — | — | — | stay 🟡 |
| 6b | noise non-monotone | ✗ | not re-tested | stays ✗ |
| 9 | A3 margin | — | not part of this decision | stays ⬜ |

No C claim failed in one or both seeds. So there is no 🟡 row with a seed count, and no ✗ is proposed.

## Rule 8 (too-good, by-construction or edge numbers)

**I1 identity is by construction.** I1 is bitwise identical: 38078 of 38078 leaves, and 0 inside the tolerance band. That follows from running the same code on the same dumps with the same libraries (`ic/check.json` `.versions`: python 3.12.3, numpy 2.1.2, scipy 1.18.1, sklearn 1.9.1). It proves Stage B is unchanged and deterministic; it says nothing about how reproducible training is.

**The outcome sits at an edge.**
- Both U items clear the 0.15 tolerance by less than 0.006 (0.1555 and 0.1530). They were applied literally, and no tolerance was edited.
- S4's budget is met with zero margin (8 of ≤ 8).
- One item moving either way would change the PASS vs CORE-PASS word, but not the promotions, which item 5 grants under both outcomes.

**s3 is the deviant seed** (`results/atlas_v1_resnet20_{s0hub,s1,s2,s3,s4}/atlas.json` `.per_layer.<L>.<inv>.<key>`):

| quantity | hub | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|
| stem PR | 2.979 | 2.993 | 2.975 | **2.644** | 3.090 |
| layer3.1 nc1 | 0.759 | 0.774 | 0.859 | **0.923** | 0.792 |
| layer3.1 PR | 10.468 | 11.259 | 9.641 | **12.610** | 9.453 |

The four lowest D1 values also all involve s3: s1-s3 0.867, s2-s3 0.911, s3-s4 0.919, hub-s3 0.923. s3 still passes every core item and every C claim.

**S5 is partly a per-class radius ranking** (Stage 1 finding), so D1 (0.919) is the honest row-7 number. S5 and D1 also rank the hub pairs in opposite order:
- S5: hub-s3 0.973 > hub-s4 0.933.
- D1: hub-s3 0.923 < hub-s4 0.958.

**ID profile spearman** is 1.000 for hub-s4 and 0.988 for s3-s4. The item barely discriminates: the Stage 1 null reached 0.867.

**Null** (`results/atlas_v1_resnet20_rand/atlas.json`, same paths):
- C1: the null also peaks at layer3.1 (24.161), but its drop is only 0.020.
- C2: the anisotropy clause holds in the null (0.085), while luminance (0.207) and highfreq (0.271) fail.
- C3: the null commits at stem.
- C4: the null reaches 3/6.
- C5a: the null reaches 4/8.

The row text keeps the "only the learned part" caveats for rows 1 and 3.

**C5a margins.**
- The s4 brightness margin (0.0395) is the smallest in any trained run, the twin included (s1 0.048, s2 0.0525, s3 0.0515, hub 0.0585, s1_ref1 0.0645); the random-init null fails C5a (4/8).
- The twin's reference-draw shift in that margin is 0.0165 (s1 0.0480 vs s1_ref1 0.0645), so the margin is 2.4× that shift.
- The query sets for C4 and C5a are fixed, so query-sampling noise remains unmeasured.

**R0.** s4 sits one test image above the band; this is not a too-good value (hub 0.9259).

**layer3.1 cka_test** for s3-s4 is 0.795, below 0.80 (`d34` `.per_layer.layer3.1.cka_test`). It is not judged; only penult is.

## Holdout and amendment 2 item 8

- **Probe pools.** The decodability probe pools include the five confirmation corruptions (`decodability.py:30`). C2, S7 and S8 are therefore clean confirmations only on the seed axis.
- **Confirmation accuracies seen.** `.meta.accuracy.corrupt__{impulse_noise,glass_blur,zoom_blur,frost,elastic_transform}__*` was displayed during evaluation. It is not used here and must not seed new claims.
- **Seeds.** Seeds 1-4 are now spent; any re-confirmation needs seeds 5 and 6.
- **Item 8.** This SESSION.md reports no margin-like quantity (per-sample nearest vs second-nearest centre distance) from the main atlases. margin_typeb is measured only in `results/margin_v1_resnet20_s{3,4}` and is not evaluated here.

## Four-seed picture (INFO; `c1234` and `results/atlas_v1_resnet20_s{1,2,3,4}/atlas.json`)

- **C claims:** C1-C4 and C5a hold in all four seeds and in the hub.
- **Penult adjacency** (critic, six seed pairs, `.checks.adjacency_stability[108..118]`): 0.965-0.975.
- **D1** (six seed pairs): 0.867-0.964.
- **Penult cka_test:** 0.883-0.892.
- **Stage 1's four uncovered FAILs** fail in `c1234` because the s1-s2 pair itself fails (results/critic_v1_resnet20_s1_s2/critic.json: rel 0.164, 0.333, 0.178, 0.161). Pairwise, layer1.0 PR, layer1.0 dim95 and layer1.1 skew also fail s1-s3, s1-s4 and hub-s1, so s1 is the deviant seed there; layer3.0 PR fails s1-s2, hub-s1 and hub-s4. All four PASS in `c34` and `c034`.

## ATLAS_STATUS changes (to apply in the same commit as this SESSION.md; the owner's commit is the human approval AGENT_LOOP.md asks for)

- **Rows 1, 3, 4, 5, 6a:** 🟡 → ✅ (s1-s4; 6a confirmed on s3, s4). Each row keeps its caveats.
- **Row 7:** penult adjacency → ✅, citing D1 and the train-reference caveat. Merge order stays 🟡.
- **Rows 2 and 8:** stay 🟡; the s3/s4 values are added.
- **Rows 6b and 9:** unchanged.
- **Header paragraph:** updated to record Stage 1b CORE-PASS.

## Next actions

1. **Commit the Stage 1b decision.** Commit the pulled Stage 1b results together with this SESSION.md, the ATLAS_STATUS changes, and a one-line "Outcome (seeds 3, 4): CORE-PASS" note in `docs/plans/STAGE1.md`. The note makes no rule change.
2. **Evaluate Stage 2 and A3, in RUN_REQUEST order,** but only after that commit (amendment 2 item 7):
   - `results/atlas_v1_resnet56_s0hub/SESSION.md` (R56-*, `docs/plans/STAGE2.md`);
   - `results/margin_v1_resnet20_s1/SESSION.md` (M1-M5, N, E, row 9).

   Their outputs were pulled with this one and are evaluated separately (results/atlas_v1_resnet56_s0hub/SESSION.md, results/margin_v1_resnet20_s1/SESSION.md).
3. **Run the discovery sweeps for the U families,** on s0 material only:
   - participation-ratio sensitivity at stem;
   - nc1 at layer3.1, merged with Stage 1's open sweep of nc1 against n_ref and pinv conditioning.

   Stage 1's other open sweeps continue as well:
   - PR seed variance at layer1.x / layer3.x;
   - the knn threshold quantile and n_query for C5a (brightness margin 0.0395);
   - sep_matrix on a test-set reference (the row 7 caveat);
   - a clean-only probe pool.
4. **Future pre-registrations:** state the precision of accuracy bands. R0 missed by one image. This does not change this decision.
