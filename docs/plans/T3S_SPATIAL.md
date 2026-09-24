# T3S: lane S of batch 4 -- what the pooled head cannot see (X5 "beyond GAP")

**Status.** Pre-registration text for P1 (with `docs/plans/B4_INTEGRATION.md`, which binds where the two differ). The
binding rules are the code: `scripts/t3s_eval.js` (labels) and `scripts/t3s_spatial_probe.py` (numbers). Nothing here
has run on data. Owner: T3S (D5), which also owns `atlas/faults.py` from P1 on.

Files: `scripts/t3s_spatial_probe.py`, `scripts/t3s_eval.js`, `tests/test_t3s_spatial.py`, this plan. Shared layer used
(not owned): `atlas/b4_core.py` (dump access with the D7 seal, the probe contract, the HEAD-ADDITIVE rule, conformal
thresholds), `atlas/b4_collapse.py` (class-centre distances, NC1), `atlas/faults.py` (every fault generator),
`scripts/b4_extract.py` (the maps layout), `scripts/b4_stats.js` (beta-binomial band, Holm, Spearman, provenance).

## 0. One screen

| | |
|---|---|
| **Question** (primary question, controller function F1-b) | A local sensor fault (dirt, glare, dead pixels, a corrupted patch) changes a few positions of the head-input map. Global average pooling dilutes it by its area before the head reads it. What can a controller read from the map, and from its spatial std, that the pooled head cannot see, how reliably, and does it beat a plain pixel check? |
| **What the controller gets** | a per-frame local-fault flag with a calibrated false-alarm rate; *where* the fault is (8 x 8 cell); the hold case "sensor changed, output still right"; local-vs-global typing (hold / clean the sensor vs adapt) |
| **Units** | discovery (Sdisc): resnet20 and resnet56 hubs + their two random-init nulls; confirmation (Sconf, sealed until P2): resnet20 s3, s4 and resnet56 s12m, s13m |
| **Rows** | CIFAR-10 test 7500-8499 (cal, clean) and 8500-9499 (eval: clean twins, faults, globals), never read by any earlier Atlas instrument; train reference: every 2nd row of the atlas draw (5000) |
| **Faults** (`atlas/faults.py`) | 21 local conditions: CIFAR-10-C s5 pixels pasted into a square of 6 / 12 / 25% area (noise, blur, pixelate, jpeg), occluder, glare, dead-pixel cluster; **soiling held out** (extracted on Sconf only, never readable in discovery); 7 whole-image CIFAR-10-C s3 corruptions |
| **Primary statistic (frozen)** | `map_max`: max over positions of a per-position Gaussian (PaDiM) on the 8 x 8 x 64 head-input map |
| **Claims** | SP-1 ... SP-8 (section 5): X5's prediction (map sees, pooled blind), the map's increment over head + pooled, over pixels, pointing, held-out soiling, the hold case, the spatial std, local-vs-global typing |
| **Cost** | CPU only: about 1-3 min per unit single-threaded (8 units, about 20 CPU-min in total, D10 limit 3600 s per unit); maps dumps 4.4 GB (registry), within D11 |

## 1. Why this lane, and what earlier work it adopts

- **Every Atlas shift so far is whole-image**, and pooling was never varied (EXTRACTION_PROPOSALS X5 "Why now"). MASTER's
  "partial corruption caught down to 10% area" is batch-level with no localisation. Inventory row F1-b ("per-sample
  sensor-fault flag") lists "localised faults; a calibrated flag with a held-out false-alarm rate" as missing
  (docs/knowledge/README.md, controller-function table).
- **Structural argument (X5).** At the head's own input layer the within-image statistics are invisible to the head: a
  fault of area a moves the pooled vector by about a times the local shift, while a per-position maximum is not diluted.
  X5's prediction: at 6-12% area, map-maximum AUROC >= 0.90 while GAP d1 and MSP / energy stay <= 0.70.
- **Adopted and refined** (the owner's rule: prior art is the starting point, refined toward controller-usable,
  head-incremental information):

  | starting point | how lane S uses and refines it |
  |---|---|
  | PaDiM, per-position Gaussians on CNN maps (Defard et al., ICPR workshops 2021, arXiv 2011.08785) | fitted on the head-input map itself (the layer GAP feeds the head), so the comparison with the pooled head is exact; shrinkage 0.1; image scores max / top-5% / area / Gini |
  | PatchCore coreset patch kNN (Roth et al., CVPR 2022, arXiv 2106.08265) | not needed at 8 x 8; kept for the ImageNet-resolution arm (batch 5) |
  | gap_std / second-order pooling (atlas/extract_acts.py:44, :88-93; Gram-matrix OOD, Sastry & Oore 2020) | the spatial std of the head map as a 64-d vector with one Gaussian: the cheapest "beyond GAP" signal (X5 arm a, claim SP-7) |
  | MSP, energy, L2-normalised kNN (Hendrycks & Gimpel 2017; Liu et al. 2020; Sun et al., ICML 2022) | the pooled comparators, all from the same dump; blindness is tested two-sided (T3 review M10) |
  | the pointing game (Zhang et al., ECCV 2016) | the argmax cell of the per-position scores against the generator's fault mask |
  | split-conformal calibration (Vovk et al. 2005; Bates et al. 2023) | every flag is thresholded on clean cal rows; its clean false-alarm rate on eval is a known answer (beta-binomial band, D14) |
  | X1's increment rule (EXTRACTION_PROPOSALS X1; T1 plan) | the frozen HEAD-ADDITIVE rule of `atlas/b4_core` with T1's full head block, so "adds beyond the head" means the same in every track |
  | pixel outlier checks (local Laplacian variance, saturation; X5 "Metrics") | a pixel PaDiM on 4 x 4 cells plus two whole-image checks: PIXELS-SUFFICE is a legitimate, useful outcome |

## 2. Units, rows and material (D3, D4, D7; `experiments/b4/models.json`)

| role | units | dump | read in |
|---|---|---|---|
| Sdisc, trained | resnet20_hub, resnet56_hub (both ANCHOR) | `results/b4d_<id>_maps/dump`, open; head map and stage-2 map | S1 discovery; S2 replay (rule 6) |
| Sdisc, null | resnet20_rand, resnet56_rand | `results/b4d_<id>_maps/dump`, open; head map | S1 discovery (INFO) |
| Sconf | resnet20_s3, resnet20_s4, resnet56_s12m, resnet56_s13m | `results/b4c_<id>_maps/dump`, **sealed** (D7); head map only | S2 confirmation, once, after P2 |

- Splits of the maps layout (`scripts/b4_extract.py`): `ref` (5000 train rows) fits every per-image model; `cal` (test
  7500-8499) calibrates every threshold (the per-position q99 of `map_area`, the Laplacian median, the 5% operating
  points); `eval` (8500-9499) holds the clean twins; `fault__<cond>` and `global__<c>__s3` pair with eval by row.
- Stored per split: `acts/penult` (= GAP of the head map), `acts/headmap_std`, `maps/headmap` (N, 64, 8, 8),
  `maps/stage2map` (N, 32, 16, 16; hubs only), logits (the canonical head path, D2), labels, rows, uint8 pixels (the
  registry switch `layouts.maps.pixels` stays on: the pixel baseline and the pairing checks read them) and fault masks.
- **Seal and holdout.** Soiling is confirmation-only (D7 `CONFIRMATION_ONLY`): it is extracted on Sconf only and
  `atlas.b4_core.open_dump` refuses it in discovery on any unit. The Sconf dumps open only with `ATLAS_B4_UNSEAL=<P2>`; a
  confirmation unit is probed once (the touched-once guard).
- **Sharing (D4).** The Sconf units are also R2 units whose sealed eval layout covers test 5000-9999 for T1. Both reads are
  confirmation reads frozen before the read, which D4 allows. The hubs' rows 7500-9499 are read in discovery only here.
- **Only two taps have maps.** The core stores the head map on every unit and the stage-2 map on the two hubs (volume,
  D11). Every claim therefore rests on the head-input map; stage 2 is INFO (SP-11). X5's stage-1 maps and ImageNet
  resolution are batch 5.

## 3. The instrument (`scripts/t3s_spatial_probe.py`)

CLI per the D5 contract: `--registry --unit --phase --out`, `--selftest --selftest-out`; dumps opened only through
`atlas.b4_core.open_unit` (the D7 seal); `timing_s` per tap (`tap:penult`, `tap:head`, `tap:headmap_std`, `tap:headmap`,
`tap:stage2map`, `tap:pixels`) and per target (every condition, every increment) plus `total`; `max_rss_mb`;
`code.sha256`, `code.core_sha256`, `repo_commit` (all via `ProbeRun`). Output `results/b4_t3s/<id><tag>/probe.json`.
Bootstrap draws: 200 in discovery and replay, 1000 in confirmation (`run.nboot`). Numbers only: labels are the
evaluator's.

**Per-image scores** (larger = more anomalous):

| family | scores | fitted on |
|---|---|---|
| map (the bundle) | `map_max` (PRIMARY), `map_top` (mean of the top 5% positions), `map_area` (fraction of positions above that position's cal q99), `map_gini` | per-position Gaussian on ref: Sigma_p = 0.9 S_p + 0.1 tr(S_p)/C I; squared Mahalanobis per position |
| stage 2 (hubs, INFO) | `s2_map_max`, `s2_map_top` | the same on the 16 x 16 map |
| spatial std (X5 arm a) | `std_half` | one Gaussian (same shrinkage) on the 64-d spatial std of the head map |
| pooled geometry | `d1` (nearest ref class centre), `knn_l2` (10th-NN of the L2-normalised penult), `gap_maha` (Gaussian on the penult) | ref |
| head (stored logits) | `msp`, `maxlogit`, `gap`, `energy` (logsumexp), `entropy`, oriented by `ERR_SIGN` | - |
| pixels (no network) | `pix_padim` (per 4 x 4 cell: mean RGB, std RGB, log Laplacian variance; Gaussian per cell; max), `pix_sat` (fraction of black / saturated pixels), `pix_lap_dev` (\|log Laplacian variance - its cal median\|) | ref pixels; cal median |

**Per condition** (fault or global vs the clean twins, same rows): AUROC of every score; TPR at the cal-calibrated 5%
threshold; pointing (the argmax position of the per-position map scores lies on a cell whose 4 x 4 pixel block holds a
fault pixel; chance = the mean fraction of such cells, about 0.19 at a12; also the one-cell dilation, the pixel baseline
and the stage-2 map); the "output still right" subset (fault argmax = clean argmax = label) with the AUROCs recomputed on
it; validity (rows and labels equal to the twin, pixels outside the fault square equal to the twin, >= 95% of images
changed, non-empty mask).

**Increments** (the frozen HEAD-ADDITIVE rule of `atlas/b4_core.head_increment`: ADDS when dAUC_joint >= +0.01 and the
95% group-bootstrap CI of dI (bits) lies above 0; BOUNDED when the upper 95% bound of dAUC_joint is below +0.02; else
INCONCLUSIVE). The head block is T1's (sorted logits linear + splines of gap, max-logit, energy, entropy); folds and
bootstrap group by image row, so a fault and its clean twin never straddle a fold.

| name | positives vs negatives | base (with the head block) | bundle |
|---|---|---|---|
| `SP2_map_over_pooled` | F5 faults vs clean twins | d1, knn_l2, gap_maha | map_max, map_top, map_area, map_gini |
| `SP3_map_over_pixels` | the 4 paste families at a12 vs clean | + pix_padim, pix_sat, pix_lap_dev | the map bundle |
| `SP7_std_over_pooled` | F5 vs clean | d1, knn_l2, gap_maha | std_half |
| `SP8_typing` | the 7 local a12 faults vs the 7 global s3 corruptions | d1, knn_l2, gap_maha | the map bundle |
| INFO | soiling a12 + a25 vs clean (Sconf); map + std over pooled; stage 2 over the head map (hubs) | | |

F5 = paste x {gaussian_noise, defocus_blur, pixelate, jpeg_compression} + occluder_sq, all at a12 (an 11 x 11 square,
11.8% of the image): X5's 6-12% band, the discovery corruptions only (D4).

**Also written:** clean accuracy on cal and eval; NC1 of the ref penult and the pseudo-label NC1 of cal
(`atlas/b4_collapse`, SP-10 INFO); functional units (D14): reference bytes in float32 (head-map PaDiM 64 x (64 + 64^2) x 4
= 1.06 MB; kNN reference 1.28 MB; std Gaussian 16.6 kB; pixel PaDiM 14 kB), false alarms per 1000 clean frames at the
calibrated 5%, CPU ms per 1000 queries (under `timing_s`, so the S2 replay ignores it).

## 4. Known answers (CLAUDE.md rule 4)

**Before any data (S0 / S1 / S2, hard):** `tests/test_t3s_spatial.py` and `--selftest`. The self-test writes numpy
"maps" dumps with **planted answers** in the extractor's layout: local faults move activation of one head-map channel
onto the fault cells while keeping every channel's spatial mean, and the stored penult and logits are the clean twin's.
So every pooled and head AUROC must be **exactly 0.5**, `map_max` >= 0.99 and pointing >= 0.95; glare is planted as
invisible (every network AUROC exactly 0.5); global shifts move the penult (pooled d1 >= 0.99). It also checks: soiling
is never read in discovery; a sealed dump is refused in discovery and without P2; confirmation after a (fake) P2 reads
soiling once and a second confirmation run is refused; a second discovery run is identical under the S2 replay
comparator (`scripts/b4_reg.py same`, rel 1e-6); SP-2 and SP-7 call ADDS. The pytest file adds helper known answers
(Gaussian: mean Mahalanobis = C (n - 1)/n exactly; Gini; cell masks and dilation; pixel cells), the lane-S registry
contract, a dump without pixels (SP-3 skipped, KA-3 false), and **the real extractor's maps layout** on the synthetic
tiny ResNet (torch): the probe's map-GAP identity, pairing, mask and hash checks must pass on what `b4_extract.py`
actually writes, for an open Sdisc and a sealed Sconf dump.

**In every run (the evaluator decides; a failing unit is not evaluable):**

| id | check | rule |
|---|---|---|
| KA-1 | map-GAP identity: the mean over positions of the stored float16 map = the stored penult, and its unbiased spatial std = the stored `headmap_std` | max error / max \|map\| <= 2e-3 (float16 half-ulp 4.9e-4), every split |
| KA-2 | pairing: rows and labels of every fault / global split = eval; pixels outside the square = the clean twin (paste, occluder, glare, dead) | exact |
| KA-3 | pixel hashes: sha256 of the first 16 stored images = `meta.b4.image_sha256_first16`; and **equal across all 8 units** (same rows, same generator) | exact; a cross-unit mismatch makes the lane NOT_EVALUABLE |
| KA-4 | calibration: the clean false-alarm rate of the cal-calibrated `map_max` flag (alpha 0.05) on eval lies in the exact central 99% beta-binomial band (n_cal = n_test = 1000: [0.027, 0.077]; `b4_stats.betaBinomBand`); rule-5 curve for alpha 0.02 / 0.03 / 0.04 (bands [0.007, 0.039], [0.013, 0.052], [0.020, 0.065]) | more than one Sconf failure: lane NOT_EVALUABLE (by chance about 6e-4) |
| KA-5 | masks: every square mask has exactly side^2 pixels (64 / 121 / 256) | exact |
| KA-6 | plumbing: clean eval accuracy of a trained unit >= 0.85 (committed test accuracies 0.920-0.945) | - |
| - | the probe's ADDS / BOUNDED / INCONCLUSIVE call equals the frozen rule recomputed from its dAUC and CIs | otherwise not counted |

## 5. Pre-registered claims (confirmation units = Sconf; `scripts/t3s_eval.js`)

Every claim needs >= 3 evaluable confirmation units, else NOT_EVALUABLE. "Two-sided" AUROC = max(a, 1 - a): an AUROC
of 0.2 is a usable flipped detector, so it counts as 0.8 against blindness. Pooled scores = d1, knn_l2, gap_maha, msp,
maxlogit, gap, energy, entropy; head scores = the last five.

| claim | prediction | rule per unit | label |
|---|---|---|---|
| **SP-1** (X5's prediction) | the map sees local faults the pooled head cannot | F5-mean AUROC(map_max) >= 0.90 **and** every pooled score's F5-mean two-sided AUROC <= 0.70 | SUPPORTED >= 3 units; REFUTED (MAP-BLIND) if map_max <= 0.80 on >= 2; REFUTED (POOLED-SEES) if a pooled score >= 0.80 on >= 2; PARTIAL 2; else INCONCLUSIVE |
| **SP-2** | the map adds functionally beyond head + pooled geometry | `SP2_map_over_pooled` call | SUPPORTED if ADDS on >= 3; REFUTED if BOUNDED on >= 3; PARTIAL if ADDS on 2; else INCONCLUSIVE |
| **SP-3** (X1's two-sided rule vs pixels) | geometry adds beyond a pixel check | `SP3_map_over_pixels` call | GEOMETRY-ADDS-OVER-PIXELS (ADDS >= 3); PIXELS-SUFFICE (BOUNDED >= 3); else MIXED |
| **SP-4** | the map says where | F5-mean pointing hit >= 0.60 **and** hit - chance >= 0.30 (chance about 0.19) | SUPPORTED >= 3; PARTIAL 2; else REFUTED |
| **SP-5** (held-out family) | the frozen statistic catches soiling, never seen in discovery | AUROC(map_max, soiling a12) >= 0.85 | SUPPORTED >= 3; PARTIAL 2; else REFUTED |
| **SP-6** (the hold case) | when the output is still right, the map still flags the sensor while the head does not move | on rows with fault argmax = clean argmax = label, over the F5 conditions with >= 50 such rows (>= 3 needed): mean AUROC(map_max) >= 0.85 **and** every head score's mean two-sided AUROC <= 0.65 | SUPPORTED >= 3; PARTIAL 2; else REFUTED |
| **SP-7** (X5 arm a) | the spatial std alone adds beyond head + pooled | `SP7_std_over_pooled` call | as SP-2 |
| **SP-8** (typing) | the map tells local from global where head + pooled cannot | `SP8_typing` call | as SP-2 |

**INFO (no label, or an INFO label):**
- **SP-9** learned vs random-init: F5 blur and jpeg pastes, map_max AUROC of each hub minus its null; LEARNED if >= 0.10
  at both depths, else ARCHITECTURAL-OR-MIXED (conv + GAP maps of random nets are frequency detectors too; the knowledge
  base's "learned-minus-random as the headline").
- **SP-10** collapse link: Spearman(-log10 NC1 of the ref penult, map_max - best pooled) over the 6 trained units
  (n = 6: a pointer for T2, not a test).
- **SP-11** stage-2 map on the hubs: F5-mean s2_map_max and 16 x 16 pointing beside the head map.
- Area curves (a06 / a12 / a25 per kind: map_max, std_half, best pooled, pixel), the functional table (false alarms per
  1000 clean frames, TPR at the calibrated 5%, reference bytes, CPU ms per 1000), Holm over the increment DeLong p-values,
  and every per-unit predicate on the two discovery hubs (the rules were fixed before those numbers existed).

**Multiplicity.** Eight claims with unit-count rules over four units each; the increment claims use the frozen
HEAD-ADDITIVE rule (a bootstrap CI and an effect-size floor, not a p-value), so no family-wise correction changes a label;
Holm over the 16 DeLong p-values is reported beside them.

**What a label licenses.** SUPPORTED SP-1 / SP-2 / SP-4 license a *per-frame local-fault flag with localisation* read from
the head-input map of these CIFAR ResNets (inventory row PC-7), calibrated on clean rows; PIXELS-SUFFICE (SP-3) says a
pixel check is enough for these synthetic faults and the map is not needed for them; SP-6 licenses the "hold" rule (sensor
flag without distrusting the output); SP-8 licenses map-based local-vs-global typing. Nothing transfers to ImageNet
resolution, other architectures or real sensor faults without re-measurement (X5 risks: CIFAR maps are tiny; the faults
are synthetic; localisation is not harm).

## 6. Discovery -> freeze -> confirmation (D7, D8)

- **P1** freezes this plan, the probe, the evaluator and the tests. Lane S emits **no P2 rules**: every statistic,
  threshold and condition set is fixed here.
- **S1** probes the four Sdisc units (discovery; `node scripts/t3s_eval.js --phase discovery --p1 <P1> --p-run <S1 HEAD>`
  prints the predicates on the hubs, INFO).
- **P2 amendments** allowed for lane S (read from `experiments/b4/freeze_P2.json` `amendments` entries with
  `"track": "T3S"`): (b) withdraw a claim to INFO with a reason; (c) a **stricter** threshold of SP-1 (map up, pooled
  down), SP-4 (hit, over-chance up), SP-5 (map up), SP-6 (map up, head down). The evaluator refuses anything else, and the
  claim is then NOT_EVALUABLE. The HEAD-ADDITIVE bounds are the core's and are not amendable here.
- **Probe bug fix at P2** (D7 failure path, with a new known-answer test): S2 re-probes the Sdisc units (`_p2`), and the
  evaluator requires the `_p2` records; the confirmation records must carry the P2 probe sha.
- **S2** probes the Sconf units once (`ATLAS_B4_UNSEAL=<P2>`) and replays the two anchors (`_s2replay`).
- **E**: `node scripts/t3s_eval.js --phase confirmation --p1 <P1> --p2 <P2> --p-run <S1 HEAD>,<S2 HEAD> --json
  results/b4/eval_t3s.json`. Provenance: the probe and core sha of every record equal their P1 (discovery) or P2
  (confirmation, `_p2`) versions; `repo_commit` is in `--p-run`; confirmation records were unsealed with P2; the maps dump
  meta sha equals `sealed.json`; `atlas/faults.py`, `scripts/b4_extract.py` and the registry are unchanged P1 -> P2; the
  evaluator equals its frozen version. A global failure makes the lane NOT_EVALUABLE; a record failure makes its unit
  not evaluable.
- **Replay (rule 6, D15).** The evaluator compares the claim statistics of the S1 and `_s2replay` records of the anchors
  (rel 1e-6, abs 1e-9). Any drift tags every label REPLAY-DRIFT; a claim whose margin to its thresholds is below the
  drift is NOT_EVALUABLE. The lead's `replay.json` status is reported beside it.
- **Relaunch** of a killed unit: `ATLAS_B4_TAG=_r2` with unchanged code (a process limit is not an instrument change); a
  confirmation unit that already has a record under any tag is never re-probed.

## 7. Budget and cuts

- **CPU:** per unit, loading about 30 splits of 1000 rows (8 MB of float16 maps and 3 MB of pixels each), PaDiM scoring
  (64 matmuls of 1000 x 64 x 64 per split), 10-NN against 5000 rows, seven cross-fitted increments on up to 14,000 rows
  (5 folds, ridge-logistic Newton) and 1000 bootstrap draws: an estimated 1-3 min single-threaded per unit, a few more on
  the hubs (the 16 x 16 stage-2 map); peak memory about 1-1.5 GB (the hubs' stage-2 ref maps). The D10 default of 3600 s
  per unit is generous; S2 limits come from S1 timing (`b4_timeouts.js`).
- **Volume:** the maps dumps are the registry's 4.4 GB (hubs 0.94 GB each with stage 2 and pixels; nulls 0.40; Sconf
  0.42 each), of which the stored pixels are about 0.83 GB (33,000-35,000 uint8 images per unit; the integration lead's
  estimate was +0.87 GB) and the masks about 0.16 GB. Kept: the pixel baseline, the pairing and the hash checks need
  them, and the probe then reads dumps only.
- **Cuts** (D10 / section 2 of the integration design): lane S is cut fourth (`ATLAS_B4_CUT="Sdisc Sconf"`), saving about
  $0.4 and 4.4 GB; plan C drops it. Cutting Sconf alone leaves discovery INFO only.

## 8. Risks

1. **Python never ran here** (no local interpreter); the first execution is S0 (pytest, `--selftest`). The evaluator and
   its fixtures ran locally (node). The planted-dump parameters were checked with a node Monte Carlo of the per-position
   Gaussian before freezing (AUROC 1.0 and pointing 1.0 at the chosen separation; a weaker separation gave 0.95).
2. **Synthetic faults may be pixel-easy** (noise and pixelate pastes, occluder). SP-3 records it; PIXELS-SUFFICE is a
   useful answer, not a failure.
3. **8 x 8 maps are coarse**: pointing is scored at the 4 x 4-pixel cell level; the one-cell dilation is INFO.
4. **Spent seeds as confirmation units**: r20 s3/s4 and r56 s12m/s13m were spent for earlier claims; the claims here are
   new, and their rows (7500-9499 and the paired CIFAR-10-C rows) were never extracted before batch 4 (D3, D4).
5. **Upstream test-set checkpoint selection** of the hubs (best epoch on the full test set): the hubs are discovery only.
6. **Replay across CPUs**: rank statistics can move by one tie at the last bit on another BLAS kernel; the replay rule
   turns that into REPLAY-DRIFT or NOT_EVALUABLE, never into a label change.

## 9. Local checks at implementation (Windows, no Python)

- `node --check scripts/t3s_eval.js`; `node scripts/t3s_eval.js --selftest`: every claim flipped to each of its labels,
  NOT_EVALUABLE runs (missing units, KA-1 / KA-6 failures, the KA-4 band, cross-unit hashes, touched-once, another
  condition set, invalid soiling, a call that disagrees with the frozen rule, the seal manifest), replay drift, P2
  amendments (withdraw, stricter, looser refused, core bounds refused), the discovery phase, and CLI runs with the real
  git provenance (NOT_EVALUABLE on fixture records; the output JSON written once; a dry run).
- The Python files passed the structural checker and the undefined-name scan of the core implementation; D6 files are
  byte-identical to `2b561a9`.

## 10. Inventory row PC-7 (draft for the lead, D19; UNTESTED at P1)

| id | information | geometric source | evidence & scope | status | beyond the output head? | already widely known/used? | how prior work uses it | how Atlas refines it | controller readiness |
|---|---|---|---|---|---|---|---|---|---|
| PC-7 (lane S part) | a local sensor fault in this frame, and where | per-position Gaussian (PaDiM) on the head-input map (resnet layer3, 8 x 8); spatial std of that map | batch 4 lane S (SP-1 ... SP-8), CIFAR ResNets, synthetic local faults, soiling held out | UNTESTED | tested directly: map bundle over the full head + pooled geometry (SP-2), over pixels (SP-3), in the hold case (SP-6) | **known-in-research**: industrial anomaly localisation (PaDiM, PatchCore, MVTec AD) | defect localisation on normal-only training | read at the layer the head pools, against the pooled head and a pixel check, with calibrated false alarms and a held-out fault family | **candidate** at best until confirmed; then a per-frame sensor flag (F1-b) with a hold rule |
