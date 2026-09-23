# Stage 1 plan: seed stability (the kill switch)

Committed before the run. Predictions N/R0/S/C/T live in `experiments/queue/atlas_v1_resnet20_s1.yaml`
(notes); this file holds the design and the decision rule the Evaluator applies.

## Why v1

Stage 0 (`atlas_v0_resnet20_cifar10`) fed the hub checkpoint inputs normalized with the true CIFAR-10
std (0.247, 0.243, 0.261). The checkpoint's own training log
(https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar10/resnet20/default.log)
shows std (0.2023, 0.1994, 0.2010), 200 epochs, batch 256. v0 therefore mapped the hub under a
mild global contrast shift (x0.82/0.82/0.77), and the queued seed-1 twin (160 ep, batch 128) would
have confounded seed with recipe and normalization. v1 fixes both:

| run | weights | role |
|---|---|---|
| `atlas_v1_resnet20_s0hub` | hub | Stage 0 under the training normalization; reference for later stages |
| `atlas_v1_resnet20_s1`, `_s2` | local seeds 1, 2, hub recipe exactly, last epoch | **kill-switch pair** |
| hub vs s1, hub vs s2 | | corroboration: a KILL must also show here (one pair cannot tell a bad instrument from one outlier seed) |
| `atlas_v1_resnet20_s1_ref1` | s1 weights, reference draw seed 1 | sampling-noise floor for every reference-based invariant |
| `atlas_v1_resnet20_rand` | random init + BN recal | null: what agreement and decodability come free from architecture + GAP |
| v0 -> v1 hub, `--same-space` | | dose-0 deformation: the same weights under a global contrast rescale |

`scripts/check_norm.py` records hub accuracy under both normalizations (N1). It is evidence, not a
gate: s1 vs s2 isolates seed variance under any normalization as long as both share it.

Instrument fixes made before this confirmation touch (discovery = s0 only):
- `class_sub_frac` used a QR basis with one arbitrary extra column (rank K-1 means); now SVD, rank-trimmed.
- `knn_density` self-queries counted each point as its own neighbour, biasing the reference radius low;
  now `exclude_self=True`. Orderings within a corruption are unchanged (same threshold for all splits).
- The critic counts `layer3.2` once, as `penult` (identical under gap pooling), and adds
  `norm_consistency`, `id_profile_stability` and `panel_agreement` (reads the pod-computed
  `compare_vs_*/deformation.json`; only `cka_test` is judged, the rest is INFO).
- The build seed (subsamples, probe rows, CV folds) now follows `data.reference.seed` unless `seed` is
  set, so the reference-resample twin also resamples the probes; every seed-0 manifest is unchanged.
- `compare` adds `cka_test`, relrep on test and CIFAR-100 (argmax agreement with its chance level,
  off-argmax row correlation) and error consistency on test.

## Decision rule (Evaluator, not the critic's verdict word)

The critic's verdict word is count-based over ~150 items and will almost surely read PARTIAL; it is
not the kill switch.

Core items for s1 vs s2: S1 (ID profile), S3 (penult sep_ratio), S5 (penult adjacency), S7 MAD on
the sharp non-at-risk factors {luminance_mean, spectral_slope, saturation_mean, hue_sin, colorfulness,
orientation_entropy, class, corruption_family, corruption_type, severity}, S8 commit/class, S9 cka_test.

| outcome | condition | action |
|---|---|---|
| INVALID | any invariant error or skip; R0 accuracy < 0.90 or seeds differ by > 1.0 pt; `input_norm` FAIL; deformation.json lacks `cka_test` | fix the run and re-run; no verdict |
| KILL | T1 twin fails a core item (the instrument cannot reproduce itself under resampling), **or** one of the following holds in s1-s2 **and** in at least one hub-seed pair: penult adjacency rho < 0.50; ID-profile spearman < 0.80 or peak shift >= 2; >= 2 sharp non-at-risk factors with decod MAD > 0.10; commit/class shift >= 2; penult cka_test < 0.60 | stop before Stage 2; instrument work on discovery material only; re-confirm on seeds 3, 4; no tolerance edits to flip items |
| PASS | every core item passes; every other FAIL is on a pre-registered at-risk list (S4, S7, S8) or has a twin delta >= 50% of its tolerance | promote per the table below; proceed to Stage 2 |
| PARTIAL | no KILL, but >= 1 core item fails | proceed carrying only the passing families; failing families stay 🟡 with a SESSION.md line and a sweep on discovery material |

Promotion on PASS (ATLAS_STATUS rows): a restated claim C<n> becomes ✅ only if it holds in s1
**and** s2; row 7 becomes ✅ for penult adjacency if S5 passes (merge order stays 🟡, non-core);
row 8 stays ⬜/🟡 (S10 is exploratory).

## Pod

`bash pod_atlas.sh /workspace --stage1` (see RUN_REQUEST in `results/atlas_v1_resnet20_s1/`).
Estimated 35-60 min on one RTX 4090 (two 200-epoch trainings run concurrently); actual: 17 min.

## Outcome (seeds 1, 2) and amendment for seeds 3, 4

Outcome: **PARTIAL by gap closure** (`results/atlas_v1_resnet20_s1/SESSION.md`). All six core items
passed and no KILL condition held, but non-core scalar spreads the pre-registration did not anticipate
remained (participation ratio at layer1.1/3.0/3.1, hubness skew at layer1.1, nc1 at layer2.0/2.2), so
PASS clause 2 failed, and the PARTIAL row (which needs a failing core item) did not apply literally
either. No row was promoted to ✅.

Amendment, committed after seeds 1-2 and before seeds 3-4 are touched; it applies only to seeds 3, 4:

| outcome | condition | action |
|---|---|---|
| CORE-PASS | no KILL; every core item passes; ≥ 1 non-core FAIL is on no at-risk list and not twin-covered | proceed; a claim C<n> that held in s1, s2 **and** holds in s3, s4 becomes ✅; uncovered non-core families stay 🟡 and get a discovery sweep |

"Twin-covered" is fixed as: the twin's relative spread ≥ 0.5 × `scalar_rel_spread` **or** its absolute
spread ≥ 0.5 × `scalar_abs_floor` (scalars); twin MAD ≥ 0.05 or profile rho ≤ 0.85 (decod); twin shift
≥ 1 (commit). The at-risk lists are not widened.
