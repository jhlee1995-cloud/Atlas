# ANOMALY_H1: mechanism predictions and missing-axis probes from the batch-2 anomaly analysis

**Status: pre-registration.** The batch-3 pre-registration commit P (f1c3c43: A4b `docs/plans/STAGE2B.md` and B1
`docs/plans/B1_VIT_MARGIN.md`) was made without this file. This file is committed in its own pre-registration commit
**P_A**, a child of P, and pushed before the pod launch, so before any batch-3 file exists (integration amendment D23).
- P_A also holds the `pod_atlas.sh` edits (`--anomaly`) and the amended launch lines of
  `results/atlas_v1_resnet56_s1/RUN_REQUEST.md`.
- P_A changes no file on the A4b or B1 frozen lists (`scripts/a4b_eval.js`, `scripts/b1_verdicts.js`), so their
  provenance (P to P_run) is unaffected. It changes no file under `atlas/`, so A4b's code-diff gate is unaffected. It does
  not touch ATLAS_STATUS.
- The pod runs at P_run = P_A or a descendant of P_A. If the pod is launched at P without P_A, ANOMALY_H1 is void as a
  pre-registration: every scored atlas and probe record would fail its provenance (section 1) by construction.

**Frozen with it** (P_A to P_run, scoped as in section 1):
- `docs/plans/ANOMALY_H1.md` and `scripts/anomaly_eval.js`, the decision rule, node only: needed by every item;
- `scripts/anomaly_probe.py`, the CPU probe, numpy only, and `tests/test_anomaly_probe.py`, its known-answer tests:
  needed by the AX items.

**Governance.**
- A bug found after the run is fixed only by a committed amendment listed in `results/anomaly_h1/SESSION.md`. The
  outcomes are then reported under both versions. The amended evaluator writes a new file (`eval_v2.json`); the evaluator
  refuses to overwrite an existing `--json`.
- **Nothing in this file, and no verdict it produces, promotes anything.** Section 7 says what a later promotion needs.

**Where the material comes from.**
- Every number quoted here as discovery comes from committed atlases at 17427c6, or from the anomaly analysis of the
  batch-2 results: 55 anomalies, 4 themes, 71 hypotheses, each verified adversarially.
- All of it is discovery material. A claim can only come from a prediction written below before its data exist.

---

## 0. One screen

**AH-1..AH-8 (and AH-8d): mechanism predictions.**
- They read only fields that batch 3 already writes:
  - the A4b atlases: `pca_spectrum`, `twonn_id`, `class_centers`, `neural_collapse`, `knn_density` and
    `corruption_displacement`;
  - the `margin_v1_*` rebuilds (M56);
  - the B1 `margin_typeb` keys, through B1's own verdict file.
- No pod code is needed. The evaluator derives every number on Windows after the pull.
- They add mechanisms to A4b's items (D-ID, D-COLL, K1, C1-C7, T56, P1-P4, X, M56-a/b/c) and to B1's (G0-G3, V1-V4,
  E1-E11, D13). None of those items is restated here.
- AH-8(d) has its own verdict, AH-8d, because it alone needs A4b's matched ladder.

**AX-1, AX-2a, AX-2b, AX-3, AX-4: five missing-axis probes** (AX-2 is scored as two axes: the tap profile and the router).
- One new CPU script, numpy only, run in a soft `block_anomaly`, the **last** block of the session, after `block_b1`.
- It reads CIFAR dumps only. It never reads a ViT or ImageNet dump and never reads logits.
- A discovery gate runs on the hub dumps. The confirmation predictions are fixed now and scored on resnet56 s1 and s2
  only if the gate opens.

**Labels.**
- Items: SUPPORTED / REFUTED / NOT_EVALUABLE; INFO when an item has no scored clause.
- AX gates: OPEN / CLOSED / DROPPED / NOT_EVALUABLE.
- Dry-run lines are INFO only (section 3).

**Controller link (MASTER_SUMMARY Gates 1-3, row-6 gap).**

| item | Gate-1 question it answers |
|---|---|
| AH-1 | calibrate density thresholds on held-out clean data; the clean false-alarm rate follows one collapse number |
| AH-2, AX-3 | the adapt-vs-hold magnitude (harm grade) |
| AH-3, AH-4 | margin vs distance is a property of the backbone's collapse, not of its architecture label |
| AH-5, AH-6, AX-1, AX-2a, AX-2b, AX-4 | the row-6 drifts: brightness is held; motion and blur are detected before the collapse; which drift, at which tap |
| AH-7, AH-8, AH-8d | how training length sets the collapse, and what TwoNN adds (little) |

**Deviations.** Freezing the thresholds on discovery data corrected four points of the selection (AH-8(c), AX-3, AX-4
and the AH-1(d) split count; section 8.1). The pre-commit review then changed the placement of the block, the
provenance scope, the AH-4/B1 interface, several clauses that random nulls pass or that sat at chance, and the evaluator's
robustness (section 8.2).

---

## 1. Methodology (binding)

**Discovery versus confirmation.**

| material | role here |
|---|---|
| resnet20 hub (v1, `_st2`, `_st3`), resnet20 s1-s4 (spent), resnet20 twin, resnet56 hub, e10/e20/e40 (seed 11), both random nulls, the anomaly analysis | **discovery** only. Used to write definitions and thresholds, never as evidence |
| `atlas_v1_resnet56_s1`, `_s2` (and their `margin_v1_` / `margin_b1_` rebuilds) | **primary confirmation**. "Confirmed" means the clause holds in s1 AND s2 |
| `_s12m`, `_s13m` (seeds 12, 13 at the matched length) | secondary confirmation, where an item names them |
| seed-11 rungs `_e50`, `_e60`, `_e70` (`_e90` if built) | secondary, where named. They share initialisation and data order with e10/e20/e40 (`scripts/train_second_seed.py:73-74,85,90`), so together they are one training run, not replicates |
| `_s0hub_st3`, `_e40_st3`, `_s0hub_ref1` | re-measures of discovery weights and the estimator twin. INFO or noise clauses only |
| B1 `margin_b1_{resnet50,vitb16,deitb}` and the E9 CIFAR `margin_b1_*` | AH-4 only. Only the keys B1 already writes, only the runs B1's `info.runs_used` names, read after B1's own verdicts exist. Never a dump |

**Evidence rule.** A primary clause must hold in s1 AND in s2. A secondary run that is not built is ignored. A
secondary run that is built counts as the item's scope says (section 3).

**How the methodology rules are met** (CLAUDE.md, rule numbers in brackets).
- [2] Per-sample metrics.
  - AX-4 is per-sample.
  - AX-1, AX-2a, AX-2b and AX-3 are batch statistics by design, because the controller decides per batch. They are not
    batch-averaged probe scores.
  - AX-3's loss is the paired accuracy difference of the same images, per batch.
- [3] Predictions come before data. Every clause below is in this file at P_A, and `results/anomaly_h1/SESSION.md`
  evaluates each as "predicted X, observed Y, verdict".
- [4] The instrument is validated before any real data.
  - `tests/test_anomaly_probe.py` and `anomaly_probe.py --selftest` (known answers on a synthetic dump) run first in
    `block_anomaly`. A failure stops the block before any real dump is read.
  - `node scripts/anomaly_eval.js --dry-run` reproduces the discovery numbers quoted here, and runs AH-4 on B1's
    committed fixtures (section 9).
- [6] Same session. The resnet20 hub is probed in this session first, on its `_st3` dump. Every AH item that
  compares with resnet20 uses the five `_st3` atlases.
- [7] The holdout. Section 6.
- [8] Skepticism. Each item lists its artifact hypotheses under "check first".

**Provenance** (scoped by what an item reads, as in the A4b and B1 evaluators). An item whose provenance fails is
NOT_EVALUABLE, with the failing check named.
- Every item: `--p P_A --p-run P_run` resolve, P_run is P_A or a descendant of P_A, and `docs/plans/ANOMALY_H1.md` and
  `scripts/anomaly_eval.js` are unchanged between them.
- Every item: each atlas the item reads (A4b atlases, `margin_v1_` rebuilds, B1 `margin_b1_` atlases) has
  `meta.git_commit` equal to P_run and `meta.created` after P_A.
- AX items, in addition:
  - `scripts/anomaly_probe.py` and `tests/test_anomaly_probe.py` are unchanged between P_A and P_run;
  - at least one self-test record `results/instrument_check_anomaly{,_r2}/selftest.json` exists, and every one that
    exists has `status == "PASS"` and `code_sha256` equal to sha256(P_A:`scripts/anomaly_probe.py`);
  - every `probe.json` the axis reads has `code.repo_commit == P_run`, the same `code.sha256`, and `created_utc` after
    P_A.
- Consequence: a code fix to the probe after P_A (a pre-launch fix, or a fix after a failed self-test) voids AX-1..AX-4
  and leaves AH-1..AH-8d untouched, because no AH item reads the probe.

**Guards.** Any of the following makes an item NOT_EVALUABLE:
- AH items other than AH-4: A4b's evaluator (`--a4b`) is missing, or reports `R0 FAIL` or `instrument changed`;
- AH items that read s1/s2 (all but AH-4 and AH-8d): a primary atlas (`atlas_v1_resnet56_{s1,s2}`, and for AH-3 also
  `margin_v1_resnet56_{s1,s2}`) has an `error` key in any invariant, a non-empty `skipped`, or a non-real `source`;
- AX confirmation and double-holdout verdicts: the A4b guard above, or A4b's G0b i2 check not PASS
  (`G0b.i2_status`: the s1/s2 dumps were built from s1/s2's own weights). The same guard applies to AH-4's two CIFAR
  points (section 4, AH-4).

Every other run is read value by value: a value that cannot be read makes its clause NULL. No threshold is ever passed or
failed on a missing value. Each item is evaluated in isolation, so an evaluator error makes that item NOT_EVALUABLE (the
error is recorded) and never touches another item.

**Append-only.** Only new directories and files are written:
- `results/anomaly_probe_<tag>/` (the probe refuses to write into an existing non-empty directory);
- `results/instrument_check_anomaly/` (a relaunch uses `_r2`);
- `results/anomaly_h1/` (the evaluation and SESSION.md, written on Windows). The evaluator refuses an existing
  `--json`. Its ID_gauss cache only gains keys: each value is a deterministic function of its key.

---

## 2. Runs, taps and derived quantities

**Run sets** (names under `results/`):

| set | contents |
|---|---|
| R56 primary | `atlas_v1_resnet56_{s1,s2}` |
| R56 new instances | `atlas_v1_resnet56_{s12m,s13m,e50,e60,e70,e90}`. e90 is built only if every first rung is below the accuracy window; s12m/s13m only if the ladder matched |
| Mc | `atlas_v1_resnet56_{s12m,s13m}` (built ones): the independent-seed comparison of AH-6(d) |
| R56 replays | `atlas_v1_resnet56_{s0hub_st3,e40_st3,s0hub_ref1}` |
| R20 band | `atlas_v1_resnet20_{s0hub,s1,s2,s3,s4}_st3` |
| M56 seeds | `margin_v1_resnet56_{s1,s2}` |
| M56new | `margin_v1_resnet56_{s1,s2,s12m,s13m,e50,e60,e70,e90}` (built ones) |
| B1 fresh primaries | `margin_b1_{resnet50,vitb16,deitb}` as named by B1's `info.runs_used` (an `_r2` replacement included), and the E9 rebuilds `margin_b1_resnet56_{s1,s2}` |
| B1 INFO | the swaps (`margin_b1_*_swap`, replicates) and the E9 hub replays `margin_b1_resnet{20,56}_s0hub_st3` |
| M11 / Mc (ladder) | A4b's matched rung and the evaluable replicates, read from `a4b_eval.json` `ladder.M11` and `ladder.Mc` |

**Taps.** resnet56 atlases are stride 5, with 11 taps. `layer3.5` is the pre-collapse tap. `layer3.8` equals `penult`
under GAP pooling, so it is dropped wherever "distinct taps" is used. The end of stage 2 is `layer2.8`. For resnet20
the corresponding taps are `layer3.1`, `layer3.2` (the alias) and `layer2.2`.

**Derived quantities** (the evaluator computes them; `pl.X` is short for `.per_layer.X`).
- **H(S)**, the harm grade: `[pl.penult.knn_density.splits.<S>.median_log_radius_shift - (same).splits.test.median_log_radius_shift] / (q95 - q50)`
  of `pl.penult.knn_density.ref_log_radius_quantiles`.
- **cost(S)** = `100 * (meta.accuracy.test - meta.accuracy.<S>)`.
- **B/T(L)** = `mean_k ||c_k - c_bar||^2 / total`, where:
  - `c_k` = `pl.L.class_centers.centers[k]`;
  - `c_bar` is their unweighted mean;
  - total = `pl.L.pca_spectrum.eig_top20[0] / pl.L.pca_spectrum.top_eig_frac`.
- **CV_all** = SD0 / mean of the 45 pairwise distances between `pl.penult.class_centers.centers`.
- **CV_nn** = SD0 / mean of the 10 nearest-other-center distances.
- **c**, the stage-3 compression: `log10(pl.<s2end>.neural_collapse.nc1 / pl.penult.neural_collapse.nc1)`.
- **Increments**: `m_s = magnitude_s * direction_pca20_s`, from `pl.L.corruption_displacement.splits.<S>`.
  - cos15(c) = cos(direction s1, direction s5);
  - inc(c) = cos(m3 - m1, m5 - m3).
- **C4 min margin** = the minimum, over (defocus s5, motion s3, motion s5) × (gaussian, shot) at the same severity, of
  `coherence(noise) - coherence(blur)` at penult. This is A4b's C4 computation.
- **ID_gauss(L)** = the mean over 5 draws (seeds 100-104) of TwoNN (n = 5000; discard the top 10% of mu; fit through
  the origin, as `atlas/invariants/dimension.py`) on N(0, diag λ).
  - λ = `pca_spectrum.eig_top20` plus a geometric tail whose sum is `eig_top20[0]/top_eig_frac - sum(eig_top20)`.
  - The sampler is mulberry32 with a Marsaglia polar Gaussian. It is ported exactly from the discovery script
    `rho_mc.js`.
- **ρ(L)** = `pl.L.twonn_id.id / ID_gauss(L)`.

---

## 3. Decision labels

**Per clause:** PASS, FAIL or NULL (a value is missing, or a premise or a subset the clause needs is empty).

**Scope of a clause:**
- *primary* (s1/s2): required;
- *required*: a twin clause, or a clause graded across runs;
- *secondary-refuting*: a named new instance that is built and fails refutes; an unbuilt one, or a NULL, is ignored;
- *secondary*: a failure is recorded as an exception, not a refutation;
- *reading*: its own label, never the verdict;
- *info*.

**Item verdict:**
- **REFUTED** if any primary, required or secondary-refuting clause FAILs.
- Otherwise **NOT_EVALUABLE** if a primary or required clause is NULL, a guard holds, or provenance fails.
- Otherwise **SUPPORTED**, with the list of secondary exceptions.
- **INFO** if the item (or a double holdout) has no primary or required clause: nothing is scored, so nothing is
  SUPPORTED.
- A refutation takes precedence over missing values. A guard or provenance failure takes precedence over everything.

**AX gates** (computed on the discovery instances; each axis has its own gate and its own verdict):
- **OPEN**: every gate clause holds in every gate instance. **AX-4 exception**: OPEN needs the resnet56 hub only, because
  confirmation is at depth 56; the resnet20 hub is INFO.
- **DROPPED**: the resnet20 hub fails a gate clause; for AX-4, all four gains are < +0.02.
- **CLOSED**: otherwise.
- **NOT_EVALUABLE**: a gate record is missing or cannot be read (for AX-4: the resnet56 hub's).
- Only OPEN scores the confirmation clauses. Under CLOSED or DROPPED, the s1/s2 values are recorded as INFO and never
  called evidence (section 6: they are spent all the same).
- The double-holdout clauses get their own verdict and never enter the main one.

**Dry run.** `--dry-run` substitutes discovery stand-ins:
- s1 := hub, s2 := e40;
- rungs and Mc := e10/e20;
- twin := resnet20 s1/s1_ref1;
- M11 := e40;
- M56new := the committed A3 margin atlases;
- AH-4 := B1's committed fixtures (`tests/fixtures/b1/<case>/results` with the `b1_verdicts.js` output on them), with
  the E9 s1/s2 points taken from the committed A3 margin atlases of the two hubs;
- synthetic probe records.

Every dry-run verdict is printed as `INFO (dry run) ...`. Its "REFUTED" lines only show that a stand-in, for example
the 40-epoch e40, is not a fully trained resnet56. They are never evidence.

---

## 4. The eight predictions

### AH-1. The collapse level sets the train-reference density baseline (Gate-1 calibration)

**Hypotheses:**
- A-H04, corrected: one collapse scalar and a tail effect, not nc1 specifically;
- H-D8, corrected: train-reference provenance, with collapse and fit not separated;
- H-E1, corrected: the twin clause is conditional on Δq95;
- A-W2: the self-query bias is at most about 0.012, by construction.

**Anomalies explained:** S0-O4, S2-sparse-level, S1-C5-twin-draw, the self-query part of S0-instrument-bugs, and
S1b-C5a (the margin compared with twin noise).

**Fields:**
- `pl.penult.neural_collapse.nc1`;
- `pl.<L>.knn_density.splits.test.{sparse_frac, median_log_radius_shift}` for every tap L;
- `pl.penult.knn_density.ref_log_radius_quantiles.{q50,q95}`;
- `pl.penult.knn_density.splits.<S>.{sparse_frac, median_log_radius_shift}` for S = test plus the 30 discovery splits.

**Runs:**
- primary: s1, s2;
- secondary-refuting: the R56 new instances;
- INFO: the R56 replays;
- twin: `s0hub_st3` → `s0hub_ref1`.

| clause | prediction | discovery (never a threshold) |
|---|---|---|
| (a) curve | `|sparse_frac(test, penult) - (0.0224 - 0.1080 * log10 nc1)| <= 0.035` | the fit on the five distinct weight sets (r20 hub, e10, e20, e40, h56) reproduces 0.0224 / -0.1080. Residuals 0.003 / 0.009 / -0.013 / -0.002 / 0.003; spent seeds and the r20 twin -0.015 to +0.027. Expected for s1/s2 at nc1 ≈ 0.05: 0.163, band [0.128, 0.198] |
| (b) step at the cliff | the maximum over every tap except penult and layer3.8 of test `sparse_frac` is <= 0.070 | 0.064 over the 12 committed atlases, both nulls included |
| (c) tail effect | in s1 and s2, penult test `median_log_radius_shift` <= 0.05 while test `sparse_frac` >= 0.12 | median shift <= 0.034 in every trained atlas (h56 0.027 at sparse 0.167). The sparse part presupposes nc1 <= ~0.10, which AH-7 also predicts |
| (d) twin | Δ = ref1 − st3 over test plus the 30 discovery splits: if `|Δq95| >= 0.01` then sign(mean Δsparse_frac) = −sign(Δq95); always `|mean Δmedian| <= 0.02` and `SD0(Δmedian) <= 0.015` | r20 twin: Δq95 −0.044, Δsparse +0.049 ± 0.016, Δmedian −0.009 ± 0.007 (31 splits) |

**Refuted if:**
- (a) or (b) fails in s1, s2 or any built new instance;
- (c) fails;
- (d) fails. This would make the median move with the draw, meaning a global dilation rather than a tail.

**Check first (rule 8):**
- After the cliff, nc1 and sep_ratio are one scalar (sep × √nc1 ≈ 1.10-1.24). The claim is "one collapse scalar".
- The seed-11 rungs fill the unmeasured nc1 range 0.10-0.15, but they count as one training run.

**Controller.**
- A train-reference density threshold gives 2.2-3.3× the nominal clean false-alarm rate, and that rate is predictable
  offline from one collapse number of the deployed checkpoint.
- Rule: calibrate on held-out, deployment-clean activations, and threshold the median radius, not the q95 tail.

### AH-2. A normalized label-free harm grade H: HOLD vs ADAPT, and the energy rule fails

**Hypotheses:**
- H-M2, reformulated after the verifier refuted "norm_ratio >= 0.95 → HOLD";
- H-D9: the harm alignment is built in (the penult is the classifier input), which is acceptable for a controller;
- H-E2: the distance scale grows with collapse, which is why H is normalized.

**Anomalies explained:** S1b-C5a-brightness-margin, S0-O7, the scale part of S2-sparse-level, and the row-6 reading
"brightness is harmless".

**Fields:** H and cost (section 2); `pl.penult.corruption_displacement.splits.<S>.norm_ratio`.

**Runs:** primary s1, s2; secondary s12m, s13m, e50, e60, e70 (a failure is an exception).

**Clauses** over the 30 discovery splits:

| clause | prediction | discovery |
|---|---|---|
| (a) | Spearman(H, cost) >= 0.93 | 0.948-0.992 over the nine discovery atlases (e20 0.948, e10 0.951) |
| (b) HOLD band | every split with H <= 0.25 costs <= 8.0 pt. No split in the band is NULL, not a pass | nets trained >= 40 epochs: <= 6.5 (h56 4.4 over 8 splits, e40 3.9 over 8). e20 8.3 and e10 27.0 fail, so under-trained rungs are outside the claim |
| (c) | brightness s1, s3 and s5 all have H <= 0.30 | maximum 0.239 (h56 s5) |
| (d) reading | the energy HOLD rule fails again: in s1 or s2, at least one discovery split has norm_ratio >= 0.95 and cost > 10 pt | h56 3 (snow s3, snow s5, pixelate s3), e40 2, r20 s1/s4 1 each, r20 hub 0 |

**Refuted if** (a), (b) or (c) fails in s1 or s2. If (d) fails in both seeds, the reading records "energy is not
refuted at depth 56". That is not a refutation of AH-2.

**Check first:** `meta.accuracy.test` uses 5000 rows while the splits use test[:2000], an offset of about 0.5 pt that
the thresholds absorb.

**Controller.** H is Gate 1's missing magnitude coordinate. It is label-free and normalized per backbone. Brightness
grades as HOLD. Energy alone would put 10-19 pt snow, pixelate and contrast shifts into HOLD. AX-3 is the per-batch
version.

### AH-3. After the final collapse, margin equals distance: mechanism profile and graded curve

**Hypotheses:**
- A-H08, corrected: the effective N is 2 weight sets, and the nearest-other CV is the relevant one;
- A-H10, corrected: the identity also needs equal uncentered center norms or bias compensation;
- A-H09, corrected: the saturation signature is the SE ratio, not the share above 0.99;
- A3 E5 and E1.

**Fields:**
- `margin_v1_*`: `pl.{layer3.5,penult}.margin_typeb.{margin_minus_dist_wrong, margin_minus_dist_typeb, auc_dist_wrong, auc_margin_wrong, spearman_margin_maxprob, spearman_margin_dist, se_maxprob_typeb, se_margin_typeb}`;
- `atlas_v1_*`: `pl.{layer3.5,penult}.class_centers.{sep_ratio, nearest_center_agrees_with_model}`.

**Runs:** M56 seeds with R56 primary (clauses a-d; both the `atlas_v1_` and the `margin_v1_` atlases of s1/s2 are
guarded); M56new, needing at least 4 built (clause e, required).

**Discovery values** are given as h56 / r20 hub (the hub at layer3.1):

| clause | prediction in s1 and s2 | discovery |
|---|---|---|
| (a) | `margin_minus_dist_wrong` >= +0.15 at layer3.5 and <= +0.02 at penult | 0.207 / 0.213 pre-collapse; −0.000 / 0.053 at penult |
| (b) | from layer3.5 to penult, Δ`auc_dist_wrong` >= 0.25 and >= 1.8 × Δ`auc_margin_wrong` | 0.337 vs 0.130; 0.323 vs 0.163 (ratio 1.98) |
| (c) | `spearman_margin_maxprob` >= 0.85 at penult and <= 0.60 at layer3.5; `nearest_center_agrees_with_model` >= 0.985 at penult and <= 0.90 at layer3.5; `spearman_margin_dist` <= −0.80 at penult | 0.886 / 0.474; 0.997 / 0.852; −0.853 (r20 penult −0.683) |
| (d) | `se_maxprob_typeb / se_margin_typeb` >= 1.25 at penult and <= 1.10 at layer3.5 | 1.539 / 0.980 (r20 penult 0.989) |

**(e) graded, across the built M56new** (sep from the matching `atlas_v1_*`):
- Spearman(penult `sep_ratio`, penult `margin_minus_dist_typeb`) <= −0.6;
- every build with sep <= 3.5 has lead >= +0.02;
- every build with sep >= 4.5 has |lead| <= 0.02.
- An empty subset (no build with sep <= 3.5, or none with sep >= 4.5) is NULL, reported as "no build in range", not a
  vacuous pass. Fewer than 4 builds makes all three NULL.
- Dry-run stand-in: the five r20 margin atlases (sep 2.97-3.07, lead +0.046 to +0.071) plus h56 (5.33, −0.001) give
  Spearman −0.83.

**Refuted if** any clause fails. The failures that matter most are:
- a collapsed build (sep >= 4.5) keeping a lead >= +0.03;
- no pre-collapse lead;
- margin − distance not falling as sep rises.

**Relation to A4b.**
- M56-b still gives row 9 its d56 tag and the M11 attribution. AH-3 adds the mechanism and the graded curve.
- If A4b's M56-c calls a seed TIE-ARTIFACT-POSSIBLE, that seed's (d) is read as float32 saturation, not geometry. The
  evaluator records this beside the clause.

**Controller.** On a strongly collapsed backbone, nearest-center distance is the type-b score. Margin matters only at
weakly collapsed taps. The collapse level decides which score Gate 1 uses.

### AH-4. The collapse regime decides margin vs distance in every architecture (a symmetric reading of B1)

**Hypothesis:** A-H18, corrected. The regime is one scalar, the "ViT cannot be equidistant" argument is dropped, and
the escape clause is made symmetric. This is the only item that is not limited to one architecture.

**Fields:**
- `pl.penult.margin_typeb.centers_geometry.sep_ratio_ref` (written by `margin.py` whenever `b1: true`, on CIFAR and on
  ImageNet; on CIFAR it is by definition the atlas `sep_ratio`);
- `pl.penult.margin_typeb.margin_minus_dist_typeb`: the top level (cut 0.7) for CIFAR, and the `sweep` row with
  `cut == c*` for ImageNet.

**Data rules** (what B1 decided is read, never recomputed):
- AH-4 is read only from the `b1_verdicts.js` output (`--b1 results/margin_b1_vitb16/verdicts.json`). If it cannot be
  read, AH-4 is NOT_EVALUABLE.
- c\* is B1's `cstar`. It is never recomputed here.
- A B1 outcome starting with `INVALID-PLUMBING` makes AH-4 NOT_EVALUABLE.
- The ImageNet primaries are exactly the atlases B1's `info.runs_used` names (an `_r2` replacement included). A primary
  that B1 dropped for plumbing is missing from it, and its clauses are NOT_EVALUABLE. The swaps are INFO replicates.
- The two CIFAR primaries (the E9 rebuilds of s1 and s2) are read only if the A4b guard and A4b's i2 check pass
  (section 1). Otherwise they are NULL.
- No dump is read.

**Runs:** 5 fresh primaries: `resnet50`, `vitb16`, `deitb` (as B1's `runs_used`) and `margin_b1_resnet56_{s1,s2}`.
The E9 hub replays `margin_b1_resnet{20,56}_s0hub_st3` (discovery weights: sep 3.0 / lead +0.07 and 5.33 / −0.001) are
INFO and never enter (c).

| clause | prediction |
|---|---|
| (a) premise | every ImageNet primary has penult `sep_ratio_ref` <= 2.0. With 1000 classes and a 25-per-class reference this is near-certain: a premise check, not evidence |
| (b) | every ImageNet primary has `margin_minus_dist_typeb(c*)` >= +0.03 |
| (c) | across the 5 fresh primaries, Spearman(`sep_ratio_ref`, `margin_minus_dist_typeb`) <= −0.6. All 5 are required |
| (d) reading per ViT, entered beside the D13 joint table | see the table below |

(c) is not implied by (a) and (b): with both holding and s1/s2 leading lowest, a reversed order inside the ImageNet
regime still gives −0.5, so (c) tests the grading within the regime as well as across it.

| B1 V1 label (B1's `V1_own` is recorded beside it) | `sep_ratio_ref` | reading |
|---|---|---|
| NULL or NULL(REVERSED) | <= 2.0 | "evidence against margin being representation-general; no regime excuse" |
| NULL or NULL(REVERSED) | >= 4.5 | REGIME-CONFOUNDED: no statement about architecture |
| NULL or NULL(REVERSED) | otherwise | UNDECIDED-REGIME |
| UNRESOLVED, SPLIT-FRAGILE or SPLIT-UNTESTED | any | UNDECIDED-SPLIT: B1 could not resolve V1, or its swap twin disagrees |
| PASS | any | n/a |
| NOT_EVALUABLE, no label, or a run B1 dropped | any | NOT_EVALUABLE |

**If gate G is closed, or c\* is null:**
- the ViT clauses are NOT_EVALUABLE;
- the resnet50 clauses are read at c\* if B1 has one, otherwise at resnet50's own first cut with n_typeb >= 300;
- (c) is NOT_EVALUABLE, with the 3-point Spearman (resnet50, s1, s2) as INFO.

**Refuted if:**
- an ImageNet primary fails (a), meaning the regime premise is false;
- an ImageNet primary has sep <= 2 and lead < +0.03;
- or (c) fails.

**Check first (rule 8):**
- 25 images per class make noisy centers; the swap twin measures that noise.
- (c) mixes two estimators of the same quantity: `sep_ratio_ref` uses 1000 reference rows per class on CIFAR and 25 on
  ImageNet. The ImageNet values are the noisier ones.
- LayerNorm (B1 H4) can raise ViT margin − distance for a reason unrelated to the regime. Read B1 E8 beside it.

**Discovery context:** the legacy ImageNet ResNet50 valley separation is 1.14, but it uses a different formula.

**Controller.** Choosing margin or distance becomes a measurable property of any backbone, including a robot's. Both
escape routes of MASTER P1 are closed.

### AH-5. Brightness takes a learned, class-neutral path and is absorbed (HOLD branch)

**Hypotheses:**
- HB-5, corrected: brightness lies on PC2/PC3 in trained stems and on PC1 at random initialisation;
- H-G4, corrected thresholds;
- P-C1 and H-A1, corrected: B/T is the lower bound of the class-span covariance share;
- the brightness clause of H-D9.

**Anomalies explained:** S1b-C5a-brightness-margin, S0-P3, S0-lum-stem-by-construction, S1b-U-stem-PR,
S1-lum-beyond-stem, and in part S0-O5.

**Fields:**
- `pl.stem.corruption_displacement.splits.corrupt__brightness__s{1,5}.direction_pca20[0..2]`;
- `pl.layer3.5.corruption_displacement.splits.corrupt__<c>__s{1,3}.class_sub_frac`;
- `pl.penult.corruption_displacement.splits.<S>.class_sub_frac`;
- B/T at layer3.5 and at penult;
- `pl.{stem,penult}.knn_density.splits.{test,corrupt__brightness__s5}.sparse_frac`;
- `meta.accuracy`.

**Runs:** primary s1, s2. Clause (a) is also secondary-refuting in s12m, s13m, e50, e60 and e70, because every trained
net must satisfy it.

| clause | prediction | discovery |
|---|---|---|
| (a) stem | brightness s1 and s5: `|dir[0]| <= 0.35` and `sqrt(dir[1]^2 + dir[2]^2) >= 0.85` | trained nets with >= 20 epochs: |[0]| 0.008-0.299 and 0.906-0.995. e10 0.416-0.436 would fail. Nulls 0.938-0.945 / 0.32-0.34 |
| (b) layer3.5 | brightness `class_sub_frac` <= 0.33 at s1 and at s3, and <= 0.70 × B/T(layer3.5); brightness is the lowest of the 10 discovery corruptions at s1, with a gap >= 0.08 | h56 0.195 / 0.232, B/T 0.391, gap 0.148; e40 0.213 / 0.257, 0.406, 0.234. **Depth-56 specific:** it fails in the spent r20 s1 and s4 (snow is lowest), in r20 s2 (gap 0.032), in e10 (gap 0.022), and on the csf bound in the r20 hub (0.338) |
| (c) penult | B/T >= 0.88 and `class_sub_frac / (B/T)` in [0.80, 1.20] for all 30 discovery splits | h56 0.934 and 0.816-1.027. e40 0.820 would fail, so this is a fully-trained prediction |
| (d) absorbed | cost(brightness s5) <= 6.5 pt, and penult excess (split minus test `sparse_frac`) <= 0.10 | 3.8-6.0 pt; trained 0.04-0.08 (nulls 0.15-0.40) |
| (d.stem) INFO | stem excess >= 0.25 | trained 0.31-0.39, but the random nulls give 0.32-0.36: an input-statistics fact, never a clause |

**Refuted if** any of (a)-(d) fails in s1 or s2, or (a) fails in a built secondary. In words: brightness on stem PC1 in
a trained net, not depleted from the class span at layer3.5, or not suppressed at penult.

**Controller (row 6).**
- Brightness can be seen early along a fixed stem direction and grades as harmless at penult, so it maps to HOLD.
- Raw `class_sub_frac` carries no information at penult and is dropped as a feature. Class-orthogonal readings belong
  at the pre-collapse tap (AX-1).

### AH-6. Motion blur is directional, noise keeps moving, and C4 tracks collapse

**Hypotheses:**
- H-C1, corrected: motion's whole mean-shift path is short, not only its late steps;
- H-C2, corrected ranges;
- H-T1, restated as the verifier corrected it: tested on independent seeds (s1/s2 against s12m/s13m), never on the
  seed-11 rungs.

**Anomalies explained:** S0-P5, S1-C5-noise-clause, S2-C4-margin, the blur-below-noise part of S0-P3, and DRIFT_COH's
blindness to motion (row 6).

**Fields:**
- `pl.{layer3.5,penult}.corruption_displacement.splits.<S>.{direction_pca20, magnitude, coherence}`;
- `pl.penult.knn_density.splits.<S>.median_log_radius_shift`;
- `pl.penult.class_centers.sep_ratio`.

**Runs:** primary s1, s2; (d.order) uses the built members of Mc (s12m, s13m); the seed-11 rungs are INFO.

| clause | prediction in s1 and s2 | discovery |
|---|---|---|
| (a) motion keeps its direction | cos15(motion) >= 0.90 at penult and at layer3.5; inc(motion) >= 0.85 at penult; cos15(motion) − cos15(snow) >= 0.25 at penult | penult 0.909-0.969 and pre 0.940-0.961; inc 0.873-0.945; snow 0.386-0.721, so the gap is h56 0.388, e40 0.396, e10 0.218 |
| (b) larger before the collapse | [mag(motion s5)/mag(gaussian s5)] at layer3.5 >= 1.25 × the same ratio at penult | h56 1.659, e40 1.348, r20 hub 1.510; e20 1.129, e10 1.066 |
| (c) noise keeps translating | penult magnitude s5/s3 >= 1.10 for gaussian and >= 1.20 for shot; shot `median_log_radius_shift` s5 − s3 >= +0.03; gaussian s5 − s3 >= −0.02 | 1.179 / 1.321 (h56), 1.141 / 1.259 (e40); shot +0.122 / +0.062 (e20 +0.004, e10 +0.025); gaussian +0.021 / +0.028 (e20 −0.017) |
| (d.c4) | C4 min margin >= 0.15 in s1 and s2 | h56 0.192; e40 0.125; r20 hub 0.113, seeds 0.105-0.132 |
| (d.order) secondary-refuting | if s12m and/or s13m are built and min sep(s1, s2) > max sep(built Mc), then min C4(s1, s2) > max C4(built Mc). A failed premise is NULL; no Mc built means the clause is ignored | discovery ordering over the hub and the seed-11 rungs (not independent seeds): h56 0.192 at sep 5.33 > e40 0.125 at 2.85 > e20 0.044 at 2.04 > e10 0.021 at 1.45 |
| INFO | Spearman(sep, C4) over s1, s2, s12m, s13m (built); the same over the rungs e50-e90 (one training run) | |

**Consequence:** row 6b (noise non-monotone) is predicted absent at depth 56.

**Refuted if** (a), (b), (c) or (d.c4) fails in s1 or s2, or (d.order) fails: motion rotating the way snow does, the
noise translation stalling, or C4 not following collapse across independent seeds.

**Check first (rule 8):**
- The shot-median part of (c) rests on two discovery instances (h56, e40).
- The random nulls pass (c) more strongly than any trained net (r20 rand g53 1.54, s53 1.88, shot Δmedian +0.050; r56
  rand 1.66 / 2.06 / +0.067), while the spent r20 seeds s1/s3/s4 fail g53 (1.073-1.086). (c) therefore tests "no
  saturation of the noise translation at depth 56". It says nothing about learned structure.
- The verifier found the original C4-collapse correlation carried by e10/e20 and by h56 alone; hence (d.order) on
  independent seeds, and the rungs as INFO.

**Controller.**
- Motion is a first-order directional signal. DRIFT_COH misses it because of poor SNR at penult and a short path, so its
  detector belongs at the pre-collapse tap (AX-1).
- Grade noise severity by displacement magnitude, not density.

### AH-7. How deep the collapse goes depends on training length

**Hypotheses:**
- A-H12, corrected: the sep-ratio "iff" is dropped as arithmetic, and the undecided gap is stated explicitly;
- A-H13 (the misfit check);
- the theme-A equidistance point.

**Anomalies explained:** S2-R56-9c, S2-ladder-miss, S2-CV-equidistant, and in part S2-b1-three-block-collapse.

**Fields:** `pl.{layer2.8,penult}.neural_collapse.nc1`; `pl.penult.class_centers.centers`; `meta.accuracy.ref`.

| clause | prediction | discovery |
|---|---|---|
| (a) reading on s1 and s2 | c >= 1.90 in both: PREDICTED ("depth × long training"). c <= 1.75 in both: HUB-SPECIFIC, which refutes. Otherwise UNDECIDED, which is NOT_EVALUABLE unless another clause fails | h56 2.11; e40 1.69; e20 1.66; e10 1.00; r20 hub 1.62; spent seeds s1-s4 1.64-1.75 |
| (b) training length | e50, e60, e70, s12m, s13m (built): c <= 1.85; Spearman(E, c) >= 0.8 over {e50: 50, e60: 60, e70: 70, e90: 90 if built, s1: 200, s2: 200}, at least 4 built. s1 and s2 tie at E = 200 (average ranks). The spent e40_st3 (known c = 1.69) is not a point | the ladder rises: e10 1.00 < e20 1.66 < e40 1.69 < hub 2.11. With the ties, a strictly rising ladder gives 0.97 (5 points) or 0.95 (4); one inverted rung pair still gives 0.87 |
| (c) equidistance | CV_all < 0.056 (the B20+ edge) in s1 and s2, and >= 0.056 in e50, e60, e70; CV_nn <= 0.035 in s1 and s2 | CV_all h56 0.037, r20 0.068-0.081, e40 0.082. The edge 0.056 is reproduced from the Stage 2 band (min 0.068, max 0.081); the live B20+ edge is printed beside it as INFO. CV_nn h56 0.014, r20 0.056-0.075, e40 0.070 |

**Check first (rule 8):** `meta.accuracy.ref` is reported beside every rung. A rung below 0.995 carries the misfit
caveat of A-H13 (e40 0.9936).

**Refuted if** (a) is HUB-SPECIFIC, or (b) or (c) fails.

**Controller.** Collapse, and with it every threshold in AH-1 and AH-3, is a property of the checkpoint's training
length. A retrained or fine-tuned robot model must be measured again.

### AH-8. TwoNN ID follows the spectrum, except at a fully trained penult (and AH-8d)

**Hypotheses:**
- H-D1, corrected;
- H-D2, restricted to the pre-collapse tap, with its controller use refuted.
It answers R56-1c after A-H14 was refuted.

**Fields:** `pl.<L>.pca_spectrum.{dim, eig_top20, top_eig_frac}` and `pl.<L>.twonn_id.id` for the 10 distinct taps.

**Runs:**
- primary s1, s2;
- (a) is secondary-refuting in the R56 new instances and INFO in the replays;
- (c) and (d) use the R20 band;
- (d) uses M11 and every evaluable Mc of A4b's ladder;
- (e) uses the twin.

**Discovery values** come from the pre-registered 5-draw port (section 8.1):

| clause | prediction | discovery |
|---|---|---|
| (a) | Pearson over the 10 taps of (id, ID_gauss) >= 0.95 in each instance | h56 0.985, e40 0.995, e10 0.997, e20 0.994 |
| (b) | ρ(layer3.5) <= 0.80 in s1 and s2 | h56 0.727, e40 0.754; r56 null 1.007; r20 hub at layer3.1 0.759 |
| (c) | ρ(penult) >= 0.98 in s1 and s2, and ID_gauss(penult) <= the maximum over the R20 band of penult ID_gauss + 0.05 | h56 ρ 1.021, ID_gauss 10.45; r20 hub 0.930 / 10.66; r20 s1 0.919 / 10.62; e40 0.901 / 12.09 |
| (d) = **AH-8d**, its own verdict | penult ID_gauss at M11 and at each evaluable Mc >= R20-band maximum + 0.30 | e40 12.09, which is +1.4 above r20 |
| (e) | the twin gives |Δρ(penult)| <= 0.03 | r20 twin 0.0002 |

**Verdicts.** AH-8 is (a), (b), (c) and (e). **AH-8d** is (d) alone: it needs A4b's ladder to be `matched`, which the
other clauses do not, so an interpolated or not-evaluable ladder makes AH-8d NOT_EVALUABLE and leaves AH-8 untouched.

**Consequence (for A4b's row-11 text).** Under (c) and (d), A4b's D-ID legs measure different things:
- L1 (accuracy-matched) sees a spectral excess;
- L2 (full recipe) sees a non-spectral one.
The row-11 text says so whichever outcome D-ID gets.

**Refuted if** any clause of the item fails.

**Check first:** the Monte-Carlo SD is 0.14-0.20 per draw, so the 5-draw SE is about 0.07-0.09 in ID_gauss (about
0.008 in ρ at penult). The observed `id_std` is 0.06-0.15.

**Controller: low.** Do not use TwoNN as a Gate-1 feature, because the Gaussian twin of the PCA spectrum carries it.
The only non-spectral signal is ρ > 1 at a fully collapsed penult.

---

## 5. The missing-axis probes (AX-1, AX-2a, AX-2b, AX-3, AX-4)

### 5.0 Common set-up

**Code.**
- `scripts/anomaly_probe.py` runs in float64, imports numpy and the standard library only, and imports nothing from
  `atlas/`, so no existing behaviour changes.
- It reads `acts/<layer>/<split>.npy`, `labels/<split>.npy` and `preds/<split>.npz`. From preds it reads `argmax`
  only.
- It never reads logits, factors, the panel or `ood__svhn`.
- It refuses before opening any file:
  - any path matching `margin_b1_`, `imagenet`, `vitb16`, `vit_b_16`, `deitb` or `deit_`;
  - any dump holding a `logits/` directory;
  - any meta that is not a 10-class CIFAR ResNet with `n_test >= 5000` and `source: real`.
- It writes one new `results/anomaly_probe_<tag>/probe.json`.

**Roles** are hard-coded by dump name, not chosen by the caller.
- `atlas_v1_resnet56_{s1,s2}` are confirmation.
- `atlas_v1_resnet{20,56}_rand` are null (INFO).
- Every other dump is discovery.
- A discovery or null dump can read only `ref`, `test`, `ood__cifar100` and the 30 discovery splits. The reader raises
  on anything else, and the tests check this.
- The confirmation corruptions (at s3/s5) are read only on the two confirmation dumps, and only for the double-holdout
  clauses.

**Rows.** The CIFAR-10 test rows are 0-4999 in order, and the corrupt splits are paired with test rows 0-1999.

| rows | use |
|---|---|
| test 2000-3499 | half A: the held-out clean mean, covariance and radius median |
| test 3500-4999 | half B: clean batches, single-class stress batches and the AX-4 clean samples |
| corrupt rows 0-1999 | corrupt evaluation (images disjoint from A and B) |

**Batches.**
- Every batch set is drawn with `numpy.random.default_rng([20260923, crc32("<purpose>|<split>|<batch>")])`. It depends
  only on its key, so every dump sees the same image rows.
- The standard is 500 batches per condition.
- **Single-class stress batches**: 50 per class, from half B by true label. The label builds the stream only, never the
  statistic.

**Discovery gate instances** (roles as above):
- the resnet20 hub: `atlas_v1_resnet20_s0hub_st3`, falling back to `_st2`;
- the resnet56 hub: `atlas_v1_resnet56_s0hub_st3`, falling back to the Stage 2 hub dump;
- AX-3 also uses e40: `atlas_v1_resnet56_e40_st3`, falling back to `_e40`.

Discovery context runs are the resnet20 seeds s1-s4 (`_st3`, falling back to the Stage 1/1b dumps) and both random
nulls. They are INFO.

**Confirmation:** `atlas_v1_resnet56_s1` and `_s2`, touched once (a relaunch never probes a dump that already has a
record; section 9).

**Why run in the same session.** Everything above is frozen in P_A before the s1/s2 dumps exist. The probe computes the
confirmation values in the same block as the gate, but the evaluator scores them only if the gate is OPEN. This spends
s1/s2 for every axis, whatever its gate (section 6).

### AX-1. Class-orthogonal first-order drift at the pre-collapse tap, plus an in-span class-prior control

**Source:** H-M1, corrected (held-out calibration arm; confirm on r56 s1/s2 at layer3.5). H-W1 is the competing arm.

**Definition.** The tap is the pre-collapse tap (r20 `layer3.1`, r56 `layer3.5`); the same statistic is also computed
at penult.
- c_k = the train-reference class means; c_bar = their unweighted mean.
- Q = the SVD basis of span{c_k − c_bar}, keeping singular values > 1e-8 s_max, which gives rank 9. P_perp = I − QQ^T.
- **T_perp(B)** = n · zbar^T Λ^-1 zbar, where zbar is the batch mean of U_r^T (x − m_A), and (U_r, Λ) are the top r <= 20
  eigenpairs (> 1e-8 λ_1) of the half-A covariance of P_perp (x − m_A).
- **T_par**: the same statistic on the 9 in-span coordinates, whitened by their half-A covariance.
- **T_full** (penult): T^2 in the top 20 half-A principal components, with no projection.
- **INFO arm (H-W1)**: the Stein divergence tr(S) − log det(S) − r of the batch covariance of the half-A-whitened top-20
  coordinates, at the pre-collapse tap.
- Batch size 64.
- tau95 = the 95th percentile of the statistic over 500 clean half-B batches.

**Gate** (both hubs, each at its own pre-collapse tap):
- AUC(T_perp, corrupt vs clean batches) >= 0.90 for motion_blur s1, defocus_blur s3, fog s3 and contrast s3, and >= 0.75
  for brightness s5;
- the mean single-class false-positive rate at tau95 is <= 0.10 for T_perp, and >= 0.50 for the penult T_full;
- c3 below: AUC(T_perp at the pre tap) − AUC(T_perp at penult) >= +0.10 for motion_blur s1 and brightness s3.

**Confirmation** (s1 and s2, layer3.5):
- (c1) the gate AUC thresholds;
- (c2) single-class FPR of T_perp <= 0.10;
- (c3) AUC(T_perp at layer3.5) − AUC(T_perp at penult) >= +0.10 for motion_blur s1 and brightness s3;
- (c4) T_perp does not separate single-class from mixed clean batches: AUC <= 0.65.
- INFO positive control: T_par separates them with AUC >= 0.95. This holds by construction (a single-class batch mean
  is a class-center offset in whitened in-span coordinates, so T_par ≫ χ²(9) wherever B/T ≳ 0.4), so it is never a
  clause.

**Double holdout** (its own verdict): AUC(T_perp) >= 0.90 for zoom_blur s3 and glass_blur s3.

**INFO (H-W1):** AUC(Stein) <= AUC(T_perp) for defocus s3 and motion s3, and the single-class FPR of Stein >= 0.30.

**Check first (rule 8):**
- Test-class means differ from the train-reference means and may leak into the complement. That threatens (c2), which
  is why (c2) is a falsifiable clause and not an assumption.
- The raw single-class offset at the pre-collapse tap (0.79-0.90 radii) is as large as the motion s3 signal. Only the
  projection removes it.

**Controller.**
- A Gate-1 ADAPT trigger for the row-6 blur, fog and contrast drifts. The idealized batch sizes are 5-40 against 68-449
  at penult, and it is immune to class skew (MASTER 87-99).
- T_par is a label-free class-prior flag: "do not adapt, hand to Gate 3".

### AX-2a. Early-tap drift profile

**Source:** HB-12, corrected: a centred Hotelling statistic with a clean held-out mean. AX-2a and AX-2b replace H-M3's
refuted energy-sign router. They were one axis in the selection; they are scored separately so that a failing profile
cannot hide a working router, or the reverse.

**Definition.**
- Taps: stem, layer1.0, the end of stage 1 (r20 `layer1.2`, r56 `layer1.8`), the end of stage 2 (`layer2.2` /
  `layer2.8`), the pre-collapse tap, and penult.
- At each tap: Hotelling T^2 of the batch mean against m_A, in the top r <= 10 half-A principal components (eigenvalues
  <= 1e-8 λ_1 dropped, because stems have dead channels).
- The profile is AUC_L at batch size 16, chosen to avoid the ceiling.
- The argmax-tap set is every tap with AUC >= max − 0.005, so ties count against a clause.

**Clauses** (the same three are the gate on both hubs and the confirmation on s1 and s2; none was ever computed on
discovery data, so the gate is their calibration):
- (a) the argmax set for brightness s1 is contained in {stem, layer1.0};
- (b) the argmax sets for defocus s1 and motion s1 are contained in {end of stage 1, end of stage 2, pre-collapse};
- (c) penult is in no argmax set of the 8 non-noise discovery corruptions at s1, scored only for corruptions whose
  max_L AUC >= 0.65 on that run. The others are recorded as "no signal" (for them the argmax is noise); if none
  qualifies, (c) is NULL.

**Check first (rule 8):** the committed displacement magnitudes already point against parts of the profile: jpeg s1 is
largest at penult (h56 0.41 against 0.40 / 0.36 at layer2.8 / layer3.5), pixelate s1 is flat from the end of stage 1 to
penult (0.26 → 0.23), and brightness s1 is nearly tied across stem / layer1.0 / end of stage 1 (0.17 / 0.16 / 0.12). The
gate may well close; that is its purpose.

**Controller.** It says what changed and at which tap.

### AX-2b. Direction-template family router

**Definition** (at the pre-collapse tap):
- The whitened (top 20 of half A) unit mean shifts of the 8 family corruptions at s3, taken from corrupt rows 0-999, are
  the templates.
- A batch of 64 from rows 1000-1999 goes to the family of the template with the highest cosine.
- Families: N = {gaussian, shot}, B = {defocus, motion, fog, contrast}, L = {brightness}, P = {pixelate}.

**Gate** (both hubs): the routing accuracy of discovery s3/s5 batches is >= 0.90 for every family.

**Confirmation** (s1 and s2): (d) routing accuracy >= 0.85 for every family.

**Double holdout** (its own verdict): impulse_noise s3/s5 go to N in >= 90% of batches, pooled over the two severities;
zoom_blur s3/s5 go to B in >= 80%. glass_blur, frost and elastic are INFO.

**Controller.** It gives MASTER P5 its type-to-response routing (BN statistics, gain, denoise).

### AX-3. A per-batch harm grade (the batch version of AH-2)

**Source:** H-M2, reformulated.

**Definition** (penult):
- r10(x) is the distance to the 10th nearest train-reference row. The whole reference is used; for the reference itself,
  self is excluded.
- **h(B)** = [median_B log r10 − median_A log r10] / (q95 − q50 of the reference self log r10).
- **loss(B)** = 100 × (clean accuracy − corrupt accuracy) of the same paired images (argmax against label).
- **e(B)** = mean_B ||x|| / mean_A ||x||. This is the batch version of the `norm_ratio` the refuted rule used. The ||x||^2
  version is INFO.
- Scored at **batch 256** with 200 batches per split. Batch 64 is summarized as INFO.

**Gate** (the resnet20 hub, the resnet56 hub and e40; 30 discovery splits × 200 batches):
- pooled Spearman(h, loss) >= 0.85;
- among batches with h <= 0.25, at most 5% lose > 10 pt.

**Confirmation** (s1 and s2):
- (a) and (b): the two gate clauses;
- (c) >= 90% of brightness batches (s1, s3 and s5 pooled) have h <= 0.30.

**Reading (d).** Among corrupt batches with e >= 0.95, >= 10% lose > 10 pt, meaning the energy grade fails (discovery,
split level: h56 3/17 in-band splits cost > 10 pt, e40 2/15, r20 0-1/11). As in AH-2(d), this is a reading, not a
refutation of h.

**INFO:** >= 90% of `ood__cifar100` batches have h > 0.25, so type-a novelty is never held (split-level H 0.88-1.61 in
nets trained >= 40 epochs; e20 0.70).

**Controller.** This is the adapt-vs-hold magnitude on the timescale at which Gate 1 decides. Its alignment with harm is
by construction, because the accuracy is a linear readout of the penult; that is acceptable for a controller.

### AX-4. A per-sample off-simplex residual e_perp

**Source:** A-H19, corrected: benchmarked against an early-tap and a two-sided stage-2 density score; discovery
corruptions only.

**Definition** (penult):
- e_perp(x) = ||P_perp (x − c_a)||^2 / ||x − c_a||^2, where a = argmax. It is label-free, and P_perp is AX-1's
  projection at penult.
- Baselines:
  - d1, the nearest-center distance;
  - d1 / r_a, with r_a the RMS within-class radius of the predicted class;
  - **dens2** = |log r10(x) − median_A log r10| at the end of stage 2, which is two-sided.
- Per-sample AUROC compares the corrupt split (positive, all 2000 rows) with clean half B.
- Every score is oriented so that larger means more corrupt-like. No AUROC is direction-free.

**Gate** (the section-3 exception: confirmation is at depth 56, so the resnet56 hub decides):
- OPEN if the resnet56 hub has AUROC(e_perp) − AUROC(d1) >= +0.05 for both defocus_blur s3 and motion_blur s3. The
  resnet20 hub's gains are INFO;
- DROPPED if all four gains (both hubs) are < +0.02;
- CLOSED otherwise; NOT_EVALUABLE if the resnet56 hub record is missing.

**Confirmation** (s1 and s2):
- the same +0.05 gain for defocus s3 and motion s3;
- AUROC(e_perp) >= AUROC(dens2) − 0.02 for both.

**Double holdout:** zoom_blur s3 is INFO (the holdout verdict is therefore INFO).

**Controller.** The only per-sample sensor for the non-directional part of row-6 drift. If it fails, the off-simplex
residual is ruled out as the carrier of that drift.

---

## 6. Holdout statement

**Definitions and thresholds.** Every definition and threshold in sections 4-5 was written on discovery material only:
the committed atlases at 17427c6 and the anomaly analysis. The dry run reproduces those discovery numbers.

**Confirmation seeds.**
- resnet56 seeds 1 and 2 were declared confirmation seeds in every Stage 2 resnet56 manifest at 664bd25.
- They are trained in this session with the full recipe and are read once: their atlases and margin rebuilds by the
  A4b evaluator and by this one, and their dumps by the probe once each.
- Seeds 12 and 13 are A4b's confirmation replicates of the matched leg.
- The seed-11 rungs are secondary and not independent.

**Probing s1/s2 spends them for every axis.** The probe computes the s1/s2 values of every axis in the same block as
the gates, and they are committed in `probe.json` whatever the gate says. Under CLOSED or DROPPED they are INFO, but
they have been seen: no revised version of that axis (another threshold, tap or statistic) can later be confirmed on
s1/s2. It needs new seeds.

**Confirmation corruptions** (impulse_noise, glass_blur, zoom_blur, frost, elastic_transform; the `holdout` block of
`experiments/queue/atlas_v1_resnet20_s1.yaml`):
- No AH item reads them. The AH items read test and the 30 discovery splits only.
- The probe reads them only on `atlas_v1_resnet56_{s1,s2}`, only at s3/s5, and only for the double-holdout clauses. The
  role is hard-coded, enforced in the reader and tested.
- They are never used to build a template, a threshold or a definition.

**B1.**
- No ViT or ImageNet dump is read by anything here.
- AH-4 reads only keys B1 writes (`margin_typeb` in the B1 atlases, per D2), only the runs B1 used, after gate G, and
  only after B1's own verdicts exist.
- The E9 margin rebuilds of s1/s2 are part of B1's one margin touch (D9).

**Spent material.** resnet20 seeds 1-4 and the resnet56 hub, e10, e20 and e40 are spent for claims. They appear here
only as discovery values, stand-ins and INFO context.

---

## 7. What a verdict means, and what a later promotion needs

**What a SUPPORTED verdict means.** The frozen prediction held on fresh material. It **does not promote anything**, and
nothing in ATLAS_STATUS changes because of it.

**A later promotion** is a separate, owner-reviewed commit. It needs all of the following:

1. **The verdict and its provenance.**
   - SUPPORTED from `node scripts/anomaly_eval.js --p P_A --p-run P_run --a4b ... --b1 ...`, with the item's provenance
     PASS and no guard. For an AX item, the gate must be OPEN.
   - Every secondary exception must be explained in SESSION.md.
2. **`results/anomaly_h1/SESSION.md`.**
   - It lists every clause as "predicted X, observed Y, verdict".
   - Before interpreting any too-good number (an AUC of 1.0, a Spearman >= 0.99), it states the artifact hypothesis and
     how it was tested (rule 8).
   - It quotes the dry-run discovery values as discovery, not as support.
3. **A4b's gates for the runs read.**
   - G0b (dump meta, no error or skip, i2), G0c (instrument unchanged) and G0e (provenance) must PASS.
   - T56 must pass for any item with a twin clause (AH-1, AH-8).
   - The A4b pair critic's `synthetic_refusal`, `holdout_hygiene` and `input_norm` must PASS (the AGENT_LOOP promotion
     rule).
4. **Scope wording.**
   - A SUPPORTED AH item replicates on two seeds at depth 56 on CIFAR-10 with a declared holdout. It can enter
     ATLAS_STATUS as a new row, 12 or higher (row 10 is B1, row 11 is A4b), scoped "d56, CIFAR-10".
   - It may carry the "replicated" status (✅) only at that scope, after steps 1-3.
   - Any statement beyond CIFAR ResNets needs its own pre-registration: an ImageNet atlas with `pca_spectrum`,
     `class_centers`, `knn_density` and `corruption_displacement`, all deferred by D2. The one exception is AH-4, whose
     regime clause B1 tests directly.
5. **Controller use of an AX axis.**
   - It must be SUPPORTED at depth 56, and its double holdout must not be REFUTED.
   - Before it becomes a Gate-1 component, it additionally needs a stream (ramp) test and a transfer test on another
     backbone. Neither is part of batch 3.
6. **A REFUTED item** is recorded as tested and not replicating (the "not replicating" status), with the failed clause
   named. A DROPPED axis is recorded as ruled out. A CLOSED axis stays a proposal; a revised version of it needs new
   confirmation seeds (section 6).

---

## 8. Deviations and amendments before P_A

### 8.1 Calibration against the discovery atlases (deviations from the selection)

The selection's thresholds were checked again against the committed discovery atlases while the evaluator was frozen.
Four points changed. Each one is a calibration fix made on discovery data, before any batch-3 data exist.

**1. AH-8 penult values (clause c).**
- *What happened.* The selection quoted penult ρ and ID_gauss (h56 1.05 / 10.20; r20 0.95-0.96 / 10.36-10.43; e40 0.93)
  from a single draw of another script, `gjob.js`, which used a different Gaussian sampler. Its values lie outside the
  spread of the pre-registered port.
- *Under the definition this file registers* (`rho_mc.js`: 5 draws, seeds 100-104), the discovery values are:
  - h56: ρ 1.021, ID_gauss 10.45;
  - r20 hub: 0.930 / 10.66;
  - r20 s1: 0.919 / 10.62;
  - e40: 0.901 / 12.09;
  - per-draw SD 0.14-0.20.
- *The fix.* Clause (c) moves from ρ >= 1.00 to ρ >= 0.98, midway between r20 (<= 0.93) and h56 (1.02). Clauses (b),
  (d) and (e) and the pre-collapse values (h56 0.727, null 1.007) are unchanged.

**2. AX-3 batch size, brightness bound and energy definition.**
- *Batch size.* At batch 64, sampling noise alone puts the resnet20 hub at about 10% harmful in-band batches, which would
  close the gate for an arithmetic reason. The estimate is a normal-approximation simulation from split-level H and cost
  with binomial batch noise: 8-12% at 64, about 6% at 128, about 2% at 256. The scored batch is therefore 256, and 64
  is INFO.
- *Brightness bound.* At 0.25, h56's brightness s5 batches (split H 0.239) would fall in the band only about 60% of the
  time. The bound is therefore 0.30, the same as AH-2(c).
- *Energy.* e is the norm ratio, so that the clause tests exactly the refuted rule. The squared version is INFO.
- *Reading.* The energy clause is a reading, as in AH-2(d).

**3. AX-4 clean side.** The clean side is half B only (1500 rows), because half A calibrates dens2's median.

**4. AH-1(d) split count.** The twin clause uses test plus the 30 discovery splits (31). The selection's ± values came
from 33 splits, including the two OOD sets. The 31-split values are Δsparse +0.049 ± 0.016 and Δmedian −0.009 ± 0.007.

**Two smaller corrections** (discovery ranges, thresholds unchanged):
- the resnet20 stage-3 compression c is 1.62-1.75 over the hub and s1-s4 (the selection quoted 1.59-1.62, hub and v0);
- the resnet20 v1 CV_nn is 0.056-0.075.

### 8.2 Changes made by the pre-commit review (before P_A)

A review of the drafts against the invariant code at 17427c6 and the batch-3 files changed the following. Every change
is made on discovery material, before any batch-3 file exists.

**Placement and packaging.**
- `block_anomaly` runs last, after `block_b1` (it was between A4b and B1). Its worst case (about 80 min) can no longer
  delay B1's GPU work, and it is equally safe after B1 because the probe refuses every B1 and ViT path.
- The batch-3 commit P was made without this file, so ANOMALY_H1 is its own pre-registration commit P_A (header).
- A relaunch never probes a dump that already has a record under any tag (`<base>` or `<base>_r<k>`), so the
  confirmation dumps are read once.

**Provenance and guards (section 1).**
- Provenance is scoped: a probe change voids the AX items only; the probe self-test record is now checked.
- AX confirmation, AX double holdouts and AH-4's CIFAR points also need the A4b guard and A4b's i2 check. AH-3 guards the
  `margin_v1_` rebuilds of s1/s2 as well as their atlases.

**AH-4 and B1.**
- Read only from B1's verdict file: c\* is never recomputed, only B1's `runs_used` are read, an INVALID-PLUMBING outcome
  voids AH-4, and every V1 label (including UNRESOLVED and the SPLIT-\* labels B1 writes over `V1_own`) has a reading.
- (c) is computed over the 5 fresh primaries. The two E9 hub replays (known discovery values) are INFO.
- (a) is labelled a premise check; the rule-8 note on the two sep estimators is added.
- The dry run reads B1's committed fixtures, so AH-4 is exercised on B1's real schema.

**Clauses that nulls pass, that sat at chance, or that were vacuous.**
- AH-5(d): the stem-excess part is INFO (random nulls give 0.32-0.36); the clause is cost <= 6.5 and penult excess <= 0.10.
- AH-6(c): a rule-8 note that random nulls pass it more strongly (r20 rand 1.54 / 1.88 / +0.050; r56 rand 1.66 / 2.06 /
  +0.067), so it tests "no saturation at depth 56" only.
- AH-6(d): restated on independent seeds as the verifier's H-T1 correction asks: (d.c4) on s1/s2 and (d.order) s1/s2
  against s12m/s13m. The seed-11 rungs are INFO. The earlier Spearman over 7 runs, 3 of them rungs, is dropped.
- AH-7(b): the spent e40_st3 anchor is dropped; the Spearman runs over e50, e60, e70 (e90 if built), s1 and s2, at least
  4 built.
- AH-3(e), and by the same rule AH-2(b): an empty subset is NULL, not a vacuous pass.
- AH-8(d) is AH-8d, with its own verdict.
- AX-1: c3 enters the gate; in c4 only T_perp <= 0.65 is a clause, and T_par becomes an INFO positive control.
- AX-2 is split into AX-2a (profile) and AX-2b (router). The profile clauses (a)-(c) enter the AX-2a gate, and (c) is
  scored only where some tap sees the corruption (max AUC >= 0.65).
- AX-4: the gate needs the resnet56 hub (section 3 exception). An item or holdout with no scored clause is INFO.

**Discovery columns.** AH-5(b) is stated as depth-56 specific; AH-4(a) as a premise check.

**Evaluator robustness.** Each item is evaluated in isolation; every threshold goes through a null-aware comparison; the
output directory is created; the report is printed before any file is written; an existing `--json` is refused.

**Probe self-test.** The two synthetic complement-shift checks (AX-1 motion s1 and holdout glass s3) are lowered from
AUC >= 0.99 to >= 0.97. A node port of the synthetic dump over 12 seeds (n_ref 3000 and 10000) gave 0.9935-0.9998 for
both, too close to 0.99 for a check that gates a frozen file. The other checks keep their margins in the same
simulation (AX-4 jpeg s3 0.93-0.95 against >= 0.90; identity 0.47-0.52 against 0.5 ± 0.06; single-class FPR 0.02-0.09
against <= 0.30; router about 0.999 against >= 0.99).

**Dry-run record.** `node scripts/anomaly_eval.js --dry-run` on the committed files at 17427c6:
- reproduces the AH-1 fit exactly (0.0224 / −0.1080), the pre-penult maximum 0.064, and the AH-2, AH-3, AH-5, AH-6 and
  AH-7 values above, including the random-null values quoted for AH-6(c);
- gives the AH-8 values of section 8.1;
- runs AH-4 on B1's committed fixtures (cases A, B, BREV, MIXED, FRAG, UNTESTED, NULLWIDE, CTRL, REPRO, NOPOW, ACCONLY,
  GATEREC, R2, POW: every V1 reading, gate closed, c\* null, INVALID-PLUMBING and an `_r2` replacement);
- runs the AX code paths on synthetic records.
Its output is INFO.

---

## 9. Execution

**Commit order.**
1. P (f1c3c43, A4b + B1) exists.
2. P_A, a child of P: this file, `scripts/anomaly_probe.py`, `scripts/anomaly_eval.js`, `tests/test_anomaly_probe.py`,
   the `pod_atlas.sh` edits and the amended launch lines of `results/atlas_v1_resnet56_s1/RUN_REQUEST.md` (integration
   D23). Pushed before the launch.
3. Pre-launch fixes produce P_run. A fix to the plan or the evaluator voids every item; a fix to the probe or its tests
   voids the AX items (section 1). Files on the A4b and B1 frozen lists stay as in P.

**Pod.** `block_anomaly` in `pod_atlas.sh` runs under the `--anomaly` flag, as the last block, after `block_b1`.
- It is CPU only and never runs git.
- It runs `pytest tests/test_anomaly_probe.py` and `anomaly_probe.py --selftest`, which are hard for the block (a
  failure stops the block before any real dump is read, never the session).
- It then probes each dump once, as a `soft` step under `timeout 900`, in this order:
  - the gate instances (the resnet20 hub first);
  - the confirmation dumps;
  - the context and null dumps.
- A block budget (`ATLAS_ANOM_BUDGET_S`, default 2700 s) is checked before each dump.
- BLAS threads equal the cgroup quota for this script only.
- A relaunch sets `ATLAS_ANOM_CHECK_DIR=results/instrument_check_anomaly_r2 ATLAS_ANOM_TAG_SUFFIX=_r2`. A dump that
  already has a `probe.json` under its own tag or any `_r<k>` tag is skipped, so only dumps without a record are probed.
  The first complete record of a dump counts.
- The estimate is about 0.5-1.5 min per dump and about 8-20 min for the block (11 dumps plus the tests), after B1; the
  hard caps bound it at about 80 min (details in the block's header).

**Windows, after the pull and the results commit** (A4b and B1 evaluators first):

```
node scripts/anomaly_eval.js --p <P_A> --p-run <P_run> --a4b results/atlas_v1_resnet56_s1/a4b_eval.json \
     --b1 results/margin_b1_vitb16/verdicts.json --cache results/anomaly_h1/idgauss_cache.json \
     --json results/anomaly_h1/eval.json
```

The first evaluation computes about 160 new ID_gauss values (16 atlases × 10 taps, about 6 s each on the drafting
machine), so it takes about 15-20 min; the cache keeps them for any later version.

**Before commit P_A:**
- `node --check scripts/anomaly_eval.js`;
- `node scripts/b1_verdicts.js --root tests/fixtures/b1/<case>/results --quiet --json <scratch>/<case>.json` for the
  fixture cases listed in section 8.2, then for each
  `node scripts/anomaly_eval.js --dry-run --b1 <scratch>/<case>.json --b1-root tests/fixtures/b1/<case>/results --no-idgauss`;
- the case-A dry run once more without `--no-idgauss` (about 4.5 min without a cache);
- `bash -n pod_atlas.sh`;
- a mock of the block in Git Bash, relaunches included;
- if a Python 3.12 / numpy 2.1 environment is available: `python -m pytest -q tests/test_anomaly_probe.py` and
  `python scripts/anomaly_probe.py --selftest`. The drafting machine has no Python, so the probe has never been
  executed; this is why the launch line leaves its tests to the block.

**On the pod, before launch** (integration D21 as amended by D23):
`python -m pytest -q tests/ --ignore=tests/test_anomaly_probe.py`, then the launch with `--a4b --b1 --anomaly`.

## 10. Not pre-registered now

Section 3 of the anomaly selection lists everything not pre-registered now, with reasons: refuted, unfalsifiable,
near-arithmetic, low power, needing new extraction or new code, fields not written by B1, and deferred Stage-B discovery
work. The main groups:
- A-H14, H-D6, H-D7/A-W1, H-D14, H-M3, the H-M2 energy rule and the H-D2 controller use: refuted;
- H-A3 and H-D10: unfalsifiable;
- every ImageNet atlas clause: B1 is margin-only (D2);
- the pod mechanism tests on existing dumps (A-H10 logit duality, A-H16 whitened CKA, HB-3, HB-7/10, H-E3, HB-15,
  H-D12, A-H13 fit-only nc1): discovery work scheduled after batch 3;
- the verifier's optional INFO extensions (H-A1 enrichment against the top-9 spectral fraction; H-W1 with a
  within-predicted-class covariance): left for that discovery work.
