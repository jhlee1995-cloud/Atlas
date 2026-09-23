# A3 plan: the margin / type-b invariant (ATLAS_STATUS row 9)

Committed before the run. The frozen definition is `atlas/invariants/margin.py` (`margin_typeb`) at this commit;
the predictions M1-M5, N, E1-E5 and the row-9 decision table live in `experiments/queue/margin_v1_resnet20_s1.yaml`
(notes). This file holds the archaeology of the legacy numbers and the design choices. Legacy code:
`jhlee1995-cloud/Upgraded-Mod@d9683cd`.

## 1. Where 0.895 / 0.763 / 0.757 come from

One script, Upgraded-Mod `session_experiments/sample_and_scale.py` (`analyze()`, arch `cifar10_resnet20`),
reported at `docs/history/SESSION_RESULTS_VALLEYS_TYPEB_SCALE.md:55-59`:

- per sample, correct vs confident-wrong (`wrong & (maxprob > 0.7)`, strict, `:52`), on the full CIFAR-10 test set
  (10,000, in order, `:31-32`), transform std (0.2470, 0.2435, 0.2616) (`:26`);
- margin = second-nearest minus nearest class-center distance in the full 64-d penult, centers = class means of
  the same test set (true labels, in-sample, misclassified samples included; `:46-48`, `:55-58`), no radius
  normalization;
- "cluster distance" = `AxisRef.per_subnet_nearest(X).mean(1)` (`:54`; `extract/axis_registry.py:33-57`): 4 channel
  groups, per group the nearest class-center distance divided by its median over the fit set (the same test set);
- "energy" = the mean squared penult activation `(F**2).mean(1)` (`:60`), not a logit energy;
- AUC direction-free, `max(a, 1 - a)` (`:21-24`), positives all conf-wrong, negatives all correct (unbalanced,
  ≈526 = 786 × 528/789 inferred vs ≈9,214 = 10,000 − 786 inferred; the recorded counts are 789 wrong and 528
  conf-wrong, `SESSION_RESULTS…:43-44`).

Corrections to how these numbers were quoted:
- 789 is all misclassified samples, 528 is misclassified with maxprob > 0.7; MASTER_SUMMARY's "type-b count 789 |
  2300" row puts all-wrong counts under the label type-b. The AUCs used the 528-type set.
- The ImageNet 0.800 / 0.636 (`session_experiments/imagenet_extract.py`) uses cut 0.5, a random size-matched
  subsample of correct samples, and a raw full-space nearest-center distance: the "0.763 -> 0.636" and
  "0.895 -> 0.800" comparisons mix definitions. The ImageNet scan covers ≈10,000-10,111 images.
- The 2-D projection margins 0.764 / 0.535 / 0.547 (`analysis/structure_geometry.py:98-145` on the
  `extract.py --cluster-struct` dump) are means in a 2-D PCA projection over a shuffled `drop_last` 9,984-image
  sample with constants (0.247, 0.243, 0.261); "type-b 0.535" there is all-wrong. They are not reproduced.
- The 0.7 cut has no recorded rationale (a CLI default of the legacy `extract.py`, introduced in 06fe3b5).
- Which run printed 789/528 is an inference: `extract.py --typeb` (9,984-image scan) prints the same counts in a
  different line format; `mahalanobis_typeb.py` with the sample_and_scale transform records 786 wrong. The exact
  10k reproduction (M1x) is not run this session.
- `AxisRef.valley_margin` (`axis_registry.py:88-96`) is a different, subnet-normalized margin that fed only the
  batch-averaged diagnostic caches (`extract.py:175-181`, `:209-219`, `:242-244`), which produced the false AUC 1.00
  ceiling.

## 2. The atlas definition (`margin_typeb`, frozen)

| choice | frozen value | why |
|---|---|---|
| type-b | `argmax != label AND maxprob > 0.7` (strict) | legacy `sample_and_scale.py:52` |
| cut | 0.7; swept over {0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99} (rule 5) | legacy value, kept so M1 is a reproduction; changing it is discovery-only (v0) and needs a new pre-registration |
| negatives | all correct test samples; secondary confidence-matched (correct with maxprob > 0.7) | legacy used all correct; the matched version removes maxprob's advantage from how type-b is selected |
| space | full GAP-pooled layer, Euclidean, unnormalized | legacy `:55-58` |
| centers | class means of the 10k train reference, true labels | atlas convention (`landmarks.py:25`, `adjacency.py:37`, `sensitivity.py:29`); avoids in-sample leakage. Test-split centers only in the `legacy` block (M1) |
| layer | penult (layer3.2 is its alias) | legacy measured penult only |
| split | clean test[:5000] | dump contract; corrupt splits are not scored this session (no corrupt predictions reach invariants) |
| AUC | oriented (errors positive; -margin, +dist, -maxprob, -energy), DeLong SE; paired DeLong for margin - dist | direction-free AUC only in `legacy` |
| RNG | none drawn from `ctx.rng`; cost "expensive"; registered last | no Stage 1 estimator draw can move (Stage 1b I1 verifies) |
| empty groups | None for every statistic that needs the group; never an exception | a random-init null may have no confident mistakes |

`legacy` block (M1 only): test-split in-sample centers, correct vs conf-wrong, direction-free AUC, 4-group subnet
cluster distance (omitted when the width is not divisible by 4), mean squared activation.

Critic: `compare.SCALARS` gains `margin_typeb.{auc_margin_typeb, auc_dist_typeb, auc_maxprob_typeb,
auc_margin_wrong, median_margin_ratio_typeb}`; the four AUCs are judged by absolute spread
(`experiments/tolerances_default.yaml` `scalar_abs_tol`, 0.05), because the relative rule would pass a 0.13 spread
at AUC 0.9. `margin_typeb.median_margin_correct` joins the critic's alias signature, so margin-only atlases collapse
layer3.2 into penult; atlases without margin keep their aliases unchanged.

## 3. Band derivations (Hanley-McNeil)

SE² = [A(1−A) + (n₊−1)(Q₁−A²) + (n₋−1)(Q₂−A²)]/(n₊n₋), Q₁ = A/(2−A), Q₂ = 2A²/(1+A).
- Legacy full 10k (n₊ ≈ 526, n₋ ≈ 9,214, both inferred; A = 0.895): SE ≈ 0.0092.
- v0 test[:5000]: n_wrong = 408 (5000 × (1 − 0.9184)); expected n₊ ≈ 408 × 528/789 ≈ 273, n₋ = 4,592, SE ≈ 0.0128.
  test[:5000] is half of the legacy 10k, so the half-minus-full difference has SD ≈ 0.009. M1 band = 2 × 0.009 +
  0.012 systematic (in-sample centers from ~500 instead of ~1000 per class; constants 0.243/0.261 vs 0.2435/0.2616
  plus float16 storage; first- vs second-half error rate 408 vs 382) = ±0.03; ±0.04 for AUC ≈ 0.76.
- v1 seeds: 372-387 wrong on test[:5000] (hub 0.9256, s1 0.9226, s2 0.9250), n₊ ≈ 255, SE ≈ 0.013 at A = 0.9; a
  two-seed difference has SD ≈ 0.018, so 0.05 is 2.7 SD for a pair and 3.85 SE for the range of four seeds.
  The shared test images correlate the AUCs positively, so the bound errs toward PASS.

## 4. Runs (Stage B only; `pod_atlas.sh --a3`)

Each `margin_*` run symlinks `results/margin_<x>/dump -> ../atlas_<x>/dump` and runs `atlas.run --skip-extract` into
a new result dir (`manifest_used.yaml` and `provenance.json` with `stage_b_git_commit` and `dump_realpath`); the
data / hooks / extract blocks are copied from the source manifest.

| run | role | when |
|---|---|---|
| `margin_v0_resnet20_cifar10` | M1 (gate), M2, M3, M5 | first |
| `margin_v1_resnet20_s0hub` | discovery: M2, M3, M4, M5, E2, E4; E5 resnet20 side | first |
| `margin_v1_resnet20_rand` | null N | first |
| `margin_v1_resnet56_s0hub` | E5 (exploratory; dump from `--stage2` in the same session) | first (skipped without the dump) |
| `margin_v1_resnet20_{s1,s2,s1_ref1}` | confirmation M2-M4, E1; twin | only if the M1 margin clause passes |
| `margin_v1_resnet20_{s3,s4}` | confirmation (dumps from `--stage1b` in the same session) | only if M1 passes; skipped without a dump |

Critics (margin-only dirs, only penult judged): `critic_margin_v1_s1_s2`, `_s0_s1_s2`, `_s1_noise`, `_s1_s4` (if s3,
s4 ran). Never mix v0 and v1 in one critic (`input_norm` FAIL); M5 and E5 are read from the atlas.json files. About
3-5 min of pod time (each rebuild 10-30 s, mostly reading ~50 splits × 11 layers).

Not measured this session: corrupt-split margins (the old E3; it needs corrupt predictions in `LayerContext`, a
Stage A contract file that Stage 1b's guard keeps frozen) and the exact 10k reproduction M1x (it needs a new
input normalization key in `atlas/extract_acts.py`, also frozen for Stage 1b).
