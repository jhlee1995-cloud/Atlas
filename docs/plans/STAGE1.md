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

## Amendment 2 (seeds 3, 4): clarifications committed with their manifests, before they exist

Stage 1 had to close a rule gap after its data was seen. The points below settle, in advance, the calls that
seeds 3 and 4 would otherwise leave to judgment: row 7 under CORE-PASS; row 6a after C5 failed as one claim;
which part of S4 and S7 is "the at-risk list" (S4 cells vs the S4 budget; S7 profile FAIL vs MAD FAIL); which
hub pairs apply. No tolerance or threshold changes and no at-risk list is widened; the decision table and
amendment 1 apply as written. What this amendment does change is stated in item 5: it widens CORE-PASS
promotion (amendment 1 promotes only "a claim C<n> that held in s1, s2") to rows 6a and 7, and makes row 7
stricter (D1 >= 0.80 as well). Predictions: `experiments/queue/atlas_v1_resnet20_s3.yaml` (notes).

1. **Pairs.** The decision pair is s3-s4 (`results/critic_v1_resnet20_s3_s4`). The KILL corroboration and T2 pairs are
   hub-s3 and hub-s4: adjacency, ID profile, decod and `penult/cka_test` from `results/critic_v1_resnet20_s0_s3_s4`
   items [0,1] and [0,2]; the class commit shift from
   `results/atlas_v1_resnet20_s{3,4}/compare_vs_atlas_v1_resnet20_s0hub/deformation.json` (`commit_layer.class`
   holds raw `{a, b}` layer names): shift = |pos(a) - pos(b)| in the critic's 10-layer list (stem ... layer3.1,
   penult) with layer3.2 mapped to penult, because the three-way critic's commit check is a range over all three
   runs. The KILL clause "T1 twin fails a core item" refers to the Stage 1 twin, which passed; no twin or null is
   re-run. `results/critic_v1_resnet20_s1_s2_s3_s4` is INFO.
2. **Instrument.** s3 and s4 are measured with the Stage 1 instrument only: their manifests name the invariants,
   cross-layer invariants and probe factors that `all` resolved to at 6242a17, in registry order. The decision
   items are the items of those checks (the item names in `results/critic_v1_resnet20_s1_s2/critic.json`); anything
   added to the code later is INFO for this decision. The A3 invariant `margin_typeb` (committed with this
   amendment) draws nothing from `ctx.rng`, has cost "expensive" and is registered last, so it cannot move a
   Stage 1 estimator draw even where it runs; it is measured on the s3 and s4 dumps only through the separate
   Stage-B rebuilds `results/margin_v1_resnet20_s{3,4}` and is not part of this decision.
3. **Coverage.** "At-risk list" (PASS clause 2, CORE-PASS) means the S4 cells (scalars), the S7 set (decod FAILs by
   profile rho whose mean_abs_delta is <= 0.10) and the S8 set (commit), as the PASS row names them. S2's list
   covers no FAIL, and a decod FAIL by mean_abs_delta > 0.10 is covered by no list. "Twin-covered" is evaluated
   once, now, on the Stage 1 twin atlases with the amendment-1 formulas. The resulting sets are listed in the s3
   manifest notes and are final: 31 of 80 scalar items, no decod factor, and commit spectral_anisotropy. 18 of the
   31 scalar items are covered only by the absolute arm (twin rel < 0.075), so SESSION.md names the arm for every
   covered FAIL; this is reporting only and does not change the outcome.
4. **INVALID** also when (instrument checks, `scripts/check_rebuild.py`; `pod_atlas.sh --stage1b` stops before
   training on I0 and I1; I2 is checked after the s3/s4 atlases, compares and critics are built, and a failure
   there fails the block (exit 1) but does not remove those artefacts):
   - I0: torch or GPU differs from `results/train_resnet20_s1/train.json`; a Stage A input (atlas/extract_acts.py,
     atlas/factors, atlas/config.py, atlas/context.py, extract/, scripts/train_second_seed.py, requirements.txt)
     changed since 6242a17 without a line in this amendment; or the s3 and s4 manifests differ in an instrument key;
   - I1: `results/instrument_check_stage1b/check.json` (Stage B rebuilt from the s1 dump with the s3 manifest vs
     the committed s1 atlas) or `replay.json` (compare and critic s1-s2 re-run with this commit's code vs the
     committed deformation.json and every critic item and tolerance) is not PASS;
   - I2: `results/instrument_check_stage1b/i2.json` is not PASS (s3/s4 dump meta vs s1's in layers, dims, splits,
     n_test, ref_indices, panel_indices, norm, norm_values, arch, dtype, pooling; own weights with a new sha256;
     no error, no skip);
   - any check of the s3_s4 or s0_s3_s4 critic reports `check crashed` (a crashed check becomes one WARN and its
     items vanish), or an item name of `results/critic_v1_resnet20_s1_s2/critic.json` (146 items; per-run names
     mapped s1 -> s3, s2 -> s4) is missing from `results/critic_v1_resnet20_s3_s4/critic.json`.
5. **Promotion under PASS or CORE-PASS.** C1-C4 (rows 1, 3, 4, 5): ✅ iff the claim holds in s3 and in s4 (all held
   in s1 and s2). Row 6a: C5a is a claim pre-registered here (the second conjunct of Stage 1 C5, verbatim); it
   becomes ✅ iff it holds in s3 and in s4 (CLAUDE.md: >= 2 seeds, prediction written before the run). The s1/s2
   8/8 result is post-hoc support, not confirmation. Row 7 (penult adjacency; merge order excluded): ✅ iff S5
   passes in s3-s4 **and** D1 (distance-only rho, `scripts/d1_distance_only.js`) >= 0.80 in s3-s4; for seeds 3 and 4
   this replaces "if S5 passes" in the PASS promotion table. Rows 2 and 8 stay 🟡. A C claim that fails in exactly
   one of s3, s4 stays 🟡 with its seed count; one that fails in both is proposed ✗ in SESSION.md. Under KILL or
   PARTIAL nothing is promoted, and seeds 3 and 4 are spent; a re-confirmation would need seeds 5 and 6.
6. **Forecasts** (F items in the s3 notes) are scored but never cover a FAIL or change the outcome.
7. **Stage 2 in the same pod session.** KILL says "stop before Stage 2", but Stage 2 (A4, resnet56;
   `docs/plans/STAGE2.md`) runs in the Stage 1b pod session. Stage 2 results produced there are evaluated only
   after the Stage 1b decision is committed. If Stage 1b is KILL or INVALID they are recorded as INFO, nothing
   from them is promoted, and Stage 2 is re-run after re-confirmation.
8. **Dumps and later invariants.** The s1-s4 dumps are first-use confirmation material for the A3 `margin_typeb`
   invariant, frozen in this same commit (`experiments/queue/margin_v1_resnet20_s1.yaml`); its rebuilds run only
   after the M1 gate on v0 passes. The same holds for an invariant frozen in a later commit, provided no
   margin-like quantity (per-sample nearest vs second-nearest centre distance) was computed on those dumps before
   that commit. The Stage 1b SESSION.md reports no such quantity from the main atlases.
9. **Procedure** (Evaluator; `results/atlas_v1_resnet20_s3/SESSION.md`), in this order:
   - INVALID: any item-4 condition; any error key or a non-empty `skipped` in s3 or s4; R0 accuracy < 0.90 or
     |s3 - s4| > 0.010; `input_norm` or `synthetic_refusal` FAIL in the s3_s4 critic; any of the three
     deformation.json files (s3 -> s4, hub -> s3, hub -> s4) lacks penult `cka_test`.
   - KILL: one of these holds in s3-s4 **and** in hub-s3 or hub-s4 (read as in item 1): penult adjacency rho < 0.50;
     id_profile spearman < 0.80 or peak shift >= 2; >= 2 sharp non-at-risk factors with decod MAD > 0.10;
     commit/class shift >= 2; penult cka_test < 0.60.
   - PARTIAL: a core item fails in the s3_s4 critic: S1 `id_profile[0,1]`; S3 `penult/class_centers.sep_ratio`;
     S5 `penult/adjacency_spearman[0,1]`; S7-core max MAD over the 10 sharp factors <= 0.10; S8 `commit/class`;
     S9 `penult/cka_test[0,1]`.
   - U = the non-core FAILs of the s3_s4 critic, within the Stage 1 check families, that are not a scalar in the
     31-item twin table or an S4 cell, not a decod profile FAIL (MAD <= 0.10) of an S7 factor, and not a commit
     FAIL of an S8 factor or of spectral_anisotropy. U empty: **PASS**; otherwise **CORE-PASS** (its U families
     stay 🟡 and get a discovery sweep).
   - Promotion (PASS or CORE-PASS only): item 5. Every ✅ also needs the AGENT_LOOP.md promotion rule
     (`synthetic_refusal` and `holdout_hygiene` PASS) and predictions committed before the dumps: `meta.git_commit`
     of s3 and s4 is the pre-registration commit P or a descendant of it, and `git diff --quiet P <git_commit> --
     experiments/queue/atlas_v1_resnet20_s3.yaml experiments/queue/atlas_v1_resnet20_s4.yaml docs/plans/STAGE1.md
     experiments/tolerances_default.yaml` holds; `meta.created` (pod clock, UTC, no zone) is later than P's commit
     time converted to UTC.

Pod: `pod_atlas.sh --stage1b`, in one session with `--stage2 --a3` (`results/atlas_v1_resnet20_s3/RUN_REQUEST.md`).
