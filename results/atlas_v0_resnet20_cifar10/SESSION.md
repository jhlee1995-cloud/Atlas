# SESSION: atlas_v0_resnet20_cifar10 (Stage 0)

Evaluator pass on the pulled results; `dump/` stays on the RunPod volume. Every number comes from
`results/atlas_v0_resnet20_cifar10/atlas.json` (JSON path given) or a cited file/URL; "derived" marks
values computed with node from atlas.json arrays. Each verdict was recomputed independently by a second
agent; corrections from that pass are folded in.

**Read this first: v0 was measured under a mismatched input normalization** (O3). The verdicts below are
the discovery record for s0; the claims were restated and frozen for Stage 1 (v1) in
`experiments/queue/atlas_v1_resnet20_s1.yaml` before any v1 run.

## Run facts
- source `real`, arch `cifar10_resnet20`, hub weights `chenyaofo cifar10_resnet20`, seed_tag `s0`,
  float16, pooling `gap`, code/manifest commit `aa07258` (`meta.git_commit`); P1-P5 are in the manifest
  at that commit. 50 splits, 11 taps of which 10 are distinct (O1), 0 invariant errors, no skips,
  `provenance.json` wall_s 151.8.
- Accuracy: test[0:5000] 0.9184 (`meta.accuracy.test`), reference (train) 0.9988 (`meta.accuracy.ref`).
- One seed, no twin: nothing here can be ✅; every FAIL is a ✗-candidate.

Verdict rule: PASS = every literal clause holds; PARTIAL = some hold, some fail; FAIL = the universal or
headline clause fails; NOT_EVALUABLE = the measured quantity is not the one the prediction names.

## Predictions

**P1: predicted** a hunchback TwoNN ID profile peaking in layer2 and falling to penult.
**Observed** (`per_layer.<L>.twonn_id.id`):

| stem | layer1.0 | layer1.1 | layer1.2 | layer2.0 | layer2.1 | layer2.2 | layer3.0 | layer3.1 | layer3.2 = penult |
|---|---|---|---|---|---|---|---|---|---|
| 6.62 | 8.70 | 9.85 | 10.06 | 12.03 | 13.07 | 13.98 | 18.01 | **19.46** | 9.87 |

**Verdict: PARTIAL.** Rise-then-fall holds; the location clause is refuted (peak layer3.1, 5.48 above
the layer2 maximum, about 19x the largest `id_std`). `id_per_class_mean` also peaks at layer3.1 (20.25 vs
12.92 in layer2). The fall is a single-block cliff that coincides with the last-block collapse
(`neural_collapse.nc1` 0.779 -> 0.177, `pca_spectrum.dim95` 41 -> 9), not a gradual decline.
Caveats: `id_std` is the spread of 3 overlapping half-subsamples (dimension.py:61-70), not a bootstrap;
ID rises within stages 1 and 2 and from layer3.0 to layer3.1; width (GAP, D = 16/32/64) explains part
of the rise but stem->layer1.0 (+2.08) has no width change. The hubness peak at layer3.1 (skew 1.91) is
expected co-variation with ID; a second estimator would test any bias.

**P2: predicted** class commits at layer2.x or later; luminance/highfreq/anisotropy commit at
stem/layer1 and wash out at penult (washout > 0.3); "row-6 gap is tap placement".
**Observed** (`cross_layer.commit_layer.per_factor.<f>`, tau 0.9, washout = best - penult):

| factor | commit | peak | best | penult | washout | clauses |
|---|---|---|---|---|---|---|
| class (excess acc.) | layer3.1 | layer3.2 | 0.815 | 0.815 | 0.000 | PASS |
| luminance_mean (R2) | stem | stem | 0.992 | 0.398 | 0.594 | PASS / PASS |
| highfreq_ratio | layer1.0 | layer2.0 | 0.619 | 0.487 | 0.132 | PASS / FAIL |
| spectral_anisotropy | layer2.0 | layer3.0 | 0.327 | 0.274 | 0.053 | undetermined / FAIL |

**Verdict: PARTIAL** (4 of 7 atomic clauses pass). The anisotropy commit misses by 0.0038 (layer1.2
0.2903 vs 0.9 x 0.3268), inside single-CV-estimate noise, so it is undetermined, not a clean FAIL;
the commit also moves into stem/layer1 for tau <= 0.888. highfreq's commit stays in layer1 only for
tau <= 0.95; its washout fails at every tau. Class commits at layer2.x+ for any tau > 0.429 (weak test:
the class profile rises monotonically). **Rule 8 flag:** luminance R2 0.992 at stem and 0.964-0.968 at
layer1.x is near-tautological (GAP of an almost linear conv carries channel means; the mixed pool with
brightness/contrast/fog inflates the luminance range), so "luminance commits at stem" is expected by
construction. The pixel-factor pool is clean test plus all 45 corrupt sets, subsampled to 4000
(decodability.py:30), which includes the confirmation corruptions. The row-6 clause is not testable
by probes, but the data lean against tap placement for 2 of 3 targets: defocus/highfreq is still
decodable at penult (0.487) and motion/anisotropy is never decodable above 0.327.

**P3: predicted** at penult, at matched magnitude, brightness/defocus/motion have lower coherence than
noise/contrast, and lower class_sub_frac. "Matched magnitude" was undefined; rules applied post hoc:
(i) interpolate each comparison curve at the A split's `per_sample_magnitude`, (ii) same severity,
(iii) interpolate on `magnitude`.
**Observed** (`per_layer.penult.corruption_displacement.splits.*`):
- vs gaussian/shot noise: lower in 6/6 (i), 18/18 (ii), 5/5 (iii). The (i) evidence rests on three A
  splits only (defocus s5, motion s3, motion s5); brightness never reaches a noise curve (max
  per-sample magnitude 0.736 < shot s1 1.061), so brightness-vs-noise is untestable here.
- vs contrast: lower in 5/7 (i), 6/9 (ii), 4/6 (iii); defocus s3/s5 reverse.
- vs impulse_noise (confirmation): 0/3 (i) and 0/3 (iii), but 7/9 (ii).
- class_sub_frac spans 0.837-0.984 over 45 splits; never lower than contrast; vs noise the margin is
  0.0005-0.025 and flips with the rule.

**Verdict: PARTIAL, close to FAIL.** The surviving clause is nearly a restatement of mean-shift size:
magnitude / per_sample_magnitude is the norm-weighted mean cosine and coherence the unweighted one
(sensitivity.py:38-43, 62-66), so their r = 0.9936 holds by construction. **Code bug found:**
class_sub_frac took a QR of the rank-9 centered class means, whose 10th column is arbitrary; fixed
(SVD, rank-trimmed) before v1. Hypothesis to test with a null: class_sub_frac is saturated at penult
(dim95 = 9). The impulse touch was spent with a post-hoc matching rule, so it is not clean confirmation.

**P4: predicted** penult sep_ratio reproduces "~2.9" within 15%. **Observed**
`per_layer.penult.class_centers.sep_ratio` = 2.9414 (+1.4% vs 2.9, +2.5% vs the historical 2.87).
**Verdict: literal PASS, non-diagnostic (NOT_EVALUABLE as a reproduction).** The definitions differ:

| figure | formula | split |
|---|---|---|
| historical 2.87 | mean off-diagonal center distance / mean L2 within-class distance | CIFAR-10 test (docs/history/SESSION_RESULTS_VALLEYS_TYPEB_SCALE.md:29; Upgraded-Mod@d9683cd extract/extract.py) |
| v0 2.94 | per-class mean of nearest-other-center / RMS radius | 10k train reference |

Bridge (derived): all-pairs numerator over RMS on train ref = 7.4195 / 2.2268 = 3.332 (+14.9% vs 2.9,
+16.1% vs 2.87). The definition mismatch was known before the run (INTEGRATION_PLAN) but not written into
P4; recorded here explicitly rather than reinterpreted silently.

**P5: predicted** knn_density sparse_frac rises monotonically with severity for every corruption.
**Observed** (`per_layer.penult.knn_density.splits.corrupt__*.sparse_frac`): 12/15 monotone.

| corruption | s1 | s3 | s5 | set |
|---|---|---|---|---|
| gaussian_noise | 0.2850 | 0.2925 | 0.2320 | discovery |
| shot_noise | 0.2435 | 0.3450 | 0.2935 | discovery |
| glass_blur | 0.4480 | 0.4085 | 0.4495 | confirmation |

**Verdict: FAIL** (also 2/10 on discovery corruptions alone). With the SE of a difference (~0.015) the
drops are z ~ 4.1, 3.2, 2.5; glass_blur is the weakest, and its accuracy is non-monotone the same way
(0.4335/0.484/0.389), consistent with a property of the corruption (make_cifar_c.py glass blur uses an
unseeded per-image pixel shuffle and near-zero smoothing at s1). **Code bias found:** the reference
radius counted each self-query as its own neighbour (density.py:25, exclude_self=False), so absolute
sparse fractions are inflated (clean test 0.1087, not ~0.05); fixed before v1. Orderings within a
corruption, and so this verdict, are unaffected. Untested hypothesis: s5 noise collapses into a dense
confident-wrong region (s5 accuracy 0.212, highest penult coherence 0.555); check the predicted-class
histogram on the dump.

## Other observations
- **O1 duplicate tap.** layer3.2 == penult in every invariant; `layer_cka` layer3.2->penult = 1.000 by
  construction. It double-weights penult in aggregates; the critic now counts it once.
- **O2 too-good numbers (rule 8).** luminance R2 (P2); noise_sigma R2 0.958-0.968 at layer1.x-layer2.0
  (likely between-split variance in the mixed pool). Both need a clean-only pool. Penult ETF cosine
  -0.1107 vs ideal -0.1111 and NC1 0.177 are on train data, expected.
- **O3 normalization mismatch (verified).** The hub's training log
  (https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar10/resnet20/default.log)
  shows std (0.2023, 0.1994, 0.201), 200 epochs, batch 256; v0 used (0.247, 0.243, 0.261), a contrast
  scale of 0.82/0.82/0.77 on every split. Test 0.9184 vs the README's 92.60 fits but does not prove it.
  The historical 0.921 and 2.87 used the same 0.247 constants. Stage 1 as first queued would have
  confounded seed with recipe and normalization; superseded by v1 (docs/plans/STAGE1.md).
- **O4** the reference set is training data: clean-test sparse_frac is 0.048-0.064 at stem..layer3.1 but
  0.1087 at penult (partly the O-P5 bias).
- **O5** coherence collapses in stage 3 for every corruption (0.8-0.99 at layer1.0-layer2.2, 0.15-0.55 at penult).
- **O6** one reorganization block: layer3.1 -> layer3.2 has the minimum consecutive CKA 0.691, with NC1
  0.779 -> 0.177, dim95 41 -> 9, nearest-center test accuracy 0.801 -> 0.917.
- **O7 OOD** penult sparse_frac: CIFAR-100 0.4745, SVHN 0.6865; contrast s5 (0.757) is sparser than SVHN.
- **O8 holdout.** This run is the single confirmation touch for impulse_noise, glass_blur, zoom_blur,
  frost, elastic_transform; the probe pools mix both sets.

## ATLAS_STATUS changes (applied)
Rows 1-6 move to 🟡 with the claims restated as observed on s0 (C1-C5 in the v1 s1 manifest); rows 7-8
unchanged (no twin yet). See ATLAS_STATUS.md.
