# SESSION: Stage 1 (v1) — seed-stability kill switch

Evaluator pass on the pulled results (`dump/` stays on the RunPod volume). Predictions:
`experiments/queue/atlas_v1_resnet20_s1.yaml` notes, committed and pushed before the run (`a33f53e`).
Decision rule: `docs/plans/STAGE1.md`. Five evaluators recomputed every prediction from the result
files with node; a decider applied the rule; two independent verifiers recomputed the decision. Both
agreed on the outcome; one rejected the decider's promotion proposal, and this record follows the verifier.

**Decision: PARTIAL (by closing a gap in the rule). No ✅ promotions from this stage.**

## Run facts
- Runs (all source `real`, norm `chenyaofo`, `meta.git_commit` 6242a17 = a33f53e + a `pod_atlas.sh`
  CPU-quota fix only; dumps created 2026-09-23 02:10:23-02:15:58 UTC, after the pre-registration commit):
  `atlas_v1_resnet20_s0hub` (hub), `_s1` (sha256 d5442d0e…), `_s2` (e7e5386f…), `_s1_ref1` (s1 weights,
  reference seed 1), `_rand` (random init + BN recal).
- `meta.accuracy.test` (test[:5000]): hub 0.9256, s1 0.9226, s2 0.9250, rand 0.1208.
- Training (`results/train_resnet20_s{1,2}/train.json`): 200 ep, batch 256, lr 0.1, wd 5e-4, last
  epoch; final 10k accuracy s1 0.927, s2 0.9264; ~470 s each on one RTX 4090, run concurrently.
- Integrity: `skipped` = {} and no error keys in all five atlases; `input_norm` PASS in all critics;
  `cka_test` present at every layer of all six deformation files. The critic counts 10 distinct layers
  (layer3.2 is an alias of penult).
- Critic counts: s1-s2 131 PASS / 13 FAIL; hub-s1-s2 207 / 19; twin 140 / 4; null 49 / 95.

## Predictions (predicted → observed → verdict)

**N and R0**
- **N1** hub 10k accuracy: chenyaofo in [0.9250, 0.9265], cifar_true in [0.917, 0.924], delta ≥ +0.3 pt,
  McNemar p < 0.05 → 0.9259, 0.921, +0.49 pt, p = 0.00168 (142 vs 93 discordant) → **PASS**.
  (`norm_check.json`; cifar_true test[:5000] = 0.9184 = v0 exactly.)
- **N2** v1 hub test > 0.9184 → 0.9256 → **PASS**.
- **N3** same-space v0 → v1 hub: penult adjacency ≥ 0.90, |Δ sep_ratio| ≤ 0.29 → 0.983, +0.055
  (2.941 → 2.996) → **PASS**. Correct normalization helps blur (+5.3 pt mean), fog/contrast, barely noise.
- **R0** s1, s2 in [0.915, 0.930], gap ≤ 0.005 → 0.927, 0.9264, 0.0006 → **PASS**.

**S — s1 vs s2** (`critic_v1_resnet20_s1_s2/critic.json`; `atlas_v1_resnet20_s2/compare_vs_atlas_v1_resnet20_s1`)
- **S1 (core)** id_profile spearman ≥ 0.90, peak shift ≤ 1, peak layer3.0/3.1 → 0.988, layer3.1 in both
  (19.446 / 19.584) → **PASS**. Caveat: the null reaches 0.867 with the same peak, so this item barely
  discriminates; only the last-block drop is learned (rand drops 0.02).
- **S2** penult dim95 = 9 in both; dim95/PR FAILs only at stem/layer1.0 → 9/9, but PR also fails at
  layer1.1 (0.218), layer3.0 (0.161), layer3.1 (0.155) → **PARTIAL**.
- **S3 (core)** penult sep_ratio PASS; nearest_center_acc_test PASS at all layers → 3.067 / 3.021
  (rel 0.015); 10/10 → **PASS**.
- **S4** ≤ 8 scalar FAILs of 80; ≥ 1 FAIL in {stem, layer1.0} × {nc1, skew} → 12 FAILs; 3 of 4 cells
  fail → **PARTIAL** (budget exceeded).
- **S5 (core)** penult adjacency ≥ 0.70 (expected 0.70-0.90) → 0.973 → **PASS** (above the expected band).
- **S6 (non-core)** penult merge tau ≥ 0.60 → 0.838 → **PASS**. `first_merge_same` is not evidence: the
  random-init null also merges cat-dog [3,5] first.
- **S7** decod MAD ≤ 0.10 for all 18; ≤ 4 profile FAILs in the at-risk set → 18/18 PASS, max MAD 0.049
  (hue_cos) → **PASS**. Core part (10 sharp non-at-risk factors): max MAD 0.037 (saturation) → **PASS**.
- **S8 (core: commit/class)** class PASS; ≤ 5 commit FAILs in the set → layer3.1 in both; one FAIL,
  highfreq_ratio (layer1.1 vs layer2.0), on the list → **PASS**.
- **S9 (core)** penult cka_test ≥ 0.80 and > 0.691 → 0.888 → **PASS**.
- **S10 (exploratory)** CIFAR-100 relrep argmax ≥ 3× chance → 0.589 vs 0.378 (4.67×; null 1.36×) →
  **PASS** (INFO). error_consistency_test 0.535, relrep_offmax_corr_test 0.831.
- **S11** input_norm PASS, alias WARN for layer3.2 only → as predicted → **PASS**.

**C — restated s0 claims, must hold in s1 AND s2** (`atlas_v1_resnet20_s{1,2}/atlas.json`)
- **C1 [row 1]** ID peak at layer3.0/3.1, penult ≥ 30% below → peak layer3.1 in both; drop 0.498 / 0.488 → **PASS**.
- **C2 [row 3]** washout luminance > 0.30, highfreq < 0.20, anisotropy < 0.20 → s1 0.736 / 0.173 / 0.062;
  s2 0.682 / 0.119 / 0.050 → **PASS** (s1 highfreq margin 0.027; the anisotropy clause also holds in the
  null, 0.085, so it is not learned structure).
- **C3 [row 4]** class commit in {layer3.0, layer3.1, layer3.2} → layer3.1 in both (holds for tau > 0.70) → **PASS**.
- **C4 [row 5]** defocus s5, motion s3/s5 coherence below gaussian and shot at the same severity → 6/6 in
  both, min margin 0.115 → **PASS**. Restated for the record: coherence tracks the mean-shift /
  per-sample-shift ratio (r = 0.9957 over 45 penult splits in s1), so this is "smaller mean shift", not
  a separate "non-directional" property.
- **C5 [row 6]** gaussian and shot have s5 < s3; other 8 discovery corruptions s5 > s1 → noise clause
  fails (s1 gaussian 0.381 → 0.383; s2 gaussian 0.485 → 0.506; s2 shot 0.475 → 0.556), other clause 8/8
  in both → **PARTIAL** (FAIL as one claim). Absolute sparse_frac is sensitive to the reference draw
  (s1 gaussian s3 0.381 vs twin 0.455); only s2 shot's +0.082 clearly exceeds twin noise.

**T — controls**
- **T1** twin s1 vs s1_ref1: adjacency ≥ 0.95, ID spearman ≥ 0.95, per-factor decod MAD ≤ 0.03,
  cka_test = 1.000 → 0.994, 1.000, max 0.0295 (highfreq; margin 0.0005), 1.000 → **PASS**. Caveats:
  the two reference draws overlap by ~20%; the query sets (test, corrupt) are fixed, so C4/C5 query noise
  is unmeasured; s1 and s2 share the reference, so "twin-covered" means estimator instability under a
  new reference, not seed noise.
- **T2** hub-s1, hub-s2 penult adjacency within 0.10 of s1-s2 → 0.939, 0.955 vs 0.973 → **PASS**.
- **T3** null penult cka_test ≤ 0.50; rand stem luminance R² ≥ 0.90 → 0.073; 0.9992 → **PASS**
  (the stem R² is an architecture + GAP artifact, as pre-registered).

Tally: 20 PASS, 3 PARTIAL (S2, S4, C5), 0 FAIL.

## Decision
- **INVALID: no.** **KILL: no** — the twin passes every core item; no kill condition holds in s1-s2
  (adjacency 0.973, ID 0.988 / shift 0, sharp MAD max 0.037, class shift 0, cka_test 0.888) or in the hub
  pairs (adjacency 0.939 / 0.955, cka_test 0.883 / 0.890).
- **Core items: 6/6 PASS.**
- **PASS: not met.** Of the 13 s1-s2 FAILs, 5 are covered (commit/highfreq by S8; stem nc1, layer1.0
  nc1, layer1.0 skew by S4; layer2.1 nc1 by the twin). Uncovered count by reading of "twin delta ≥ 50%
  of tolerance": 8 (relative spread only), 6 (if S2's list counts), 4 (relative OR absolute), 2 (both).
  Under every reading, layer1.1 hubness skew (twin rel 0.015) and layer3.0 participation ratio (twin
  rel 0.002) remain: real seed differences the pre-registration did not anticipate.
- **Rule gap.** The PARTIAL row requires a failing core item, and none fails; no row applies literally.
  Closed conservatively as PARTIAL. The STAGE1 promotion table is defined only for PASS, so **no row is
  promoted to ✅ from this stage**; widening the at-risk lists after seeing the data is not allowed.

## Rule 8 (too-good or by-construction numbers)
- S5's 0.973 is partly a per-class tightness ranking: at penult the 45 center distances are nearly equal
  (CV 0.068 / 0.081, neural-collapse regime on the train reference, ref accuracy 0.9994-0.9997), and the
  radius-only term alone gives s1-s2 rho 0.942. The distance-only adjacency is the honest number:
  s1-s2 0.950, hub-s1 / hub-s2 0.955 / 0.963, twin 0.993, null 0.644.
- The null's penult adjacency (0.478) sits just under the KILL threshold 0.50; the threshold is weak.
- luminance R² > 0.95 beyond stem (s1 layer1.0-layer2.0, s2 to layer1.2); the null explains it only to
  layer1.1. noise_sigma R² > 0.95 (s1 layer2.0, hub layer1.0) is not explained by the null (max 0.760).
  Pool = clean + all corruptions (decodability.py:30); a clean-only pool is still untested.
- hub-s1 stem cka_test 0.991 (null-s1 0.513): early layers agree across models largely by construction.

## Holdout
Decodability pools include the five confirmation corruptions (decodability.py:30), so the
corruption-axis holdout is not clean for C2, S7, S8; only the seed axis is a clean confirmation.
Values seen for glass_blur and impulse_noise during evaluation must not seed new claims. Seeds 1 and 2
are now spent; seeds 3 and 4 are reserved.

## ATLAS_STATUS changes (applied)
- Rows 1, 3, 4, 5: stay 🟡, now "replicated on s1 and s2 (v1); Stage 1 did not reach PASS" with the
  caveats above (row 1: only the last-block drop is learned; row 3: cite penult luminance, not stem;
  row 5: mean-shift reading).
- Row 2: stays 🟡; v1 value ~3.0 (hub 2.996, s1 3.067, s2 3.021).
- Row 6: split. Noise non-monotonicity → ✗ (failed in s1 and s2). s5 > s1 for the other discovery
  corruptions → 🟡 (8/8 in both seeds).
- Row 7: ⬜ → 🟡 for penult adjacency, citing the distance-only rho (0.950 vs null 0.644) and the
  train-reference caveat; merge order ⬜ → 🟡 (tau 0.838), without first-merge as evidence.
- Row 8: ⬜ → 🟡 (OOD relrep 4.67× chance, exploratory).

## Next actions
1. Commit the missing decision-table row ("all core items pass, some non-core FAILs uncovered") to
   STAGE1.md **before** seeds 3 and 4 are touched; ✅ for rows 1, 3, 4, 5, 7 is then decided on seeds 3 and 4.
2. Discovery-only sweeps (s0 material): nc1 stability vs n_ref and pinv conditioning; PR seed variance at
   layer1.x / layer3.x; knn threshold quantile and n_query for C5; sep_matrix on a test-set reference
   (the adjacency caveat); a clean-only probe pool (rule-8 flags).
3. Proceed to Stage 2 (resnet56, norm chenyaofo) carrying the passing families, and to the margin/type-b
   invariant (roadmap A3).
