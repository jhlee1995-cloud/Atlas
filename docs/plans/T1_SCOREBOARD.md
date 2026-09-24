# T1_SCOREBOARD: what intermediate geometry adds beyond the full output head (batch 4, Track 1)

**Status: pre-registration, committed at P1** (`atlas: batch 4 pre-registration (P1)`). Written against design HEAD
`2b561a9` and the batch-4 shared layer. `docs/plans/B4_INTEGRATION.md` (D1-D19) is binding: where this document and it
disagree, B4_INTEGRATION wins. Nothing here is evidence or a promotion; every discovery number is INFO.

**Owner goals (binding).** The primary question is which controller-usable information backbone geometry carries, how
reliably, and what it adds **beyond the output head** (docs/knowledge/README.md:14-17). End functionality counts, not
novelty; prior art is a starting point to adopt and refine (README.md:21-26). The standard detectors (conformal kNN
flags, Trust Score, relative Mahalanobis, kNN purity, LID, ViM, NECO, ATC, DoC, BBSE / BBSD, CUSUM, MEWMA, conformal
martingales) are therefore adopted as components and as baselines, and every geometric score is judged as an
**increment over the full head**, never over maxprob alone.

---

## 0. One screen

| what | how | answers |
|---|---|---|
| X1 head-conditional scoreboard | per unit and target: cross-fitted head-only vs head + bundle (ridge logistic on restricted cubic splines, 5 group folds by image row); dAUC with a group-bootstrap CI, dI in bits with its CI, DeLong, dTPR at 5% FPR, dAURC; three-way call ADDS / BOUNDED / INCONCLUSIVE | the "beyond the head" column of the inventory (CM-*, DO-*, ST-*, PC-*) |
| X4 calibrated multi-tap flag | L2-normalised 10-NN distance at the X4 taps (first, pre, penult) -> conformal p per tap -> Cauchy fusion -> conformal p again | a false-alarm rate that transports to unseen rows and unseen architectures (DO-6, new) |
| X8, X7, X2 | Trust Score, kNN purity, LID, relative Mahalanobis; prediction depth / cross-tap agreement; the whitened head-null residual | cheap per-sample tests inside the same joint machinery |
| X6 batch arm | batch 16 / 64 / 256: AC, DoC, ATC vs the harm grade H; BBSE / BBSD label-skew tests; the BBSE-explained covariate residual R; covariate / prior / novelty typing | harm and skew against free head estimators (LH-*, ST-3, ST-5) |
| X3 streams | ARL0 = 2000 CUSUM / MEWMA / conformal CUSUM / BBSDh / KS on seeded frame streams; step, ramp, benign ramp, skew, OOD, fault and stuck-frame scenarios; change-information rate per tap | detection delay and false alarms per 1000 frames (ST-9, PC-7, new) |
| per-model outcomes for T2 | O1 (penult bundle dAUC_joint on clean errors) and O5 (X4 AUROC minus the best head AUROC at s3), `experiments/b4/model_outcomes.schema.json` | the inputs of T2's PRIMARY (the owner's collapse hypothesis, D12) |

Files: `scripts/t1_scoreboard.py`, `scripts/t1_streams.py` (numpy only; probes), `scripts/t1_eval.js` (the frozen
evaluator), `scripts/b4_stats.js` (shared, owner T1), `tests/test_t1_scoreboard.py`, `tests/test_t1_streams.py`,
`tests/t1_eval_fixture.js`. The shared numerics (AUC, DeLong, the joint model, the HEAD-ADDITIVE rule, conformal p,
`open_dump`) are `atlas/b4_core.py` (owner T1); NC1 comes from `atlas/b4_collapse.py` (owner T2).

## 1. Scope after the integration (D1, D16)

- **Moved out of T1:** the model-level collapse arm (MP-1 ... MP-9, `collapse_coords`, the zoo table) is T2's; MP-9 is
  the PRIMARY and X4-2M is T2's mechanism item M-DO3. The ImageNet arm (L11) is deferred; the knob lane (L12) is T2's.
  T1's own extraction, head export and pod block are replaced by `scripts/b4_extract.py`, `scripts/b4_weights.py` and
  `scripts/pod_b4.sh`. The never-read eval layout of the spent seeds is reduced to R2 (8 nets).
- **Kept, with every T1-review fix (except the D16 overrides):** X1, X4, X8, X7, X2, X6, X3, the three-way calls (C1),
  the X4 calibration on the whole fit split in the eval layout (A2), the restated stuck-frame claim (A3), the replay gate
  with a tolerance (A4, now D9/D15), X1-1's MSP rule (A5), fresh seeds as the primary per-sample units (B2, as F / F20 in
  D3), the anchor gates against committed Atlas numbers (B6), functional magnitudes (C3), Mahalanobis with pseudo-inverse
  semantics (E1), the cheap residual forms (E2), whitened head / pixel information rates (E3), censored delays (E4), a
  clause on the CIFAR-10-C extras (E5), salt 1 for the eval-layout faults (E6, in `atlas/faults.py`), and the fixture
  requirements (G).
- **Overridden by D16:** T1 B2's copied training script (D13: the frozen `train_second_seed.py` trains the fresh seeds;
  its 10k-accuracy print is an aggregate read, disclosed). T1 B3 is dissolved by D2 (one instrument for every unit).

## 2. Units and rows (D3, D4)

| scope | units | layout | rows read by T1 | role in T1 |
|---|---|---|---|---|
| R (primary, confirmation) | F: resnet56_s31, s32; F20: resnet20_s31, s32 (fresh seeds, frozen recipe) | eval (sealed) | test 5000-9999; CIFAR-10-C 5000-9999; OOD 2000-6999; faults of 8500-9999 (salt 1); calibration from the same unit's fit dump | the per-sample claims |
| R2 (secondary, confirmation) | resnet20 s1-s4, resnet56 s1, s2, s12m, s13m (spent seeds, never-read rows) | eval (sealed) | as R, without the extras | a second label for the per-sample claims |
| M (confirmation) | the 12 C architectures (resnet44, vgg13/16/19_bn, mobilenetv2 x0_75/x1_0/x1_4, shufflenetv2 x1_0/x1_5/x2_0, repvgg a1/a2) | fit (sealed) | test 0-4999, CIFAR-10-C 0-1999 | X4 transport (X4-1M) and breadth labels |
| S (confirmation) | STconf: F, F20 and resnet20 s1, s3, s4, resnet56 s1, s12m, s13m | eval (sealed) | as R | the stream claims |
| discovery (INFO) | D (16: r20 hub, s1-s4; r56 hub, e10-e70, s1, s2, s12m, s13m), Dnew (5), N (2, INFO), STdisc (r20 hub, r56 hub, r56 e40) | fit (open) | test 0-4999, CIFAR-10-C 0-1999, the discovery corruptions and faults only | every claim evaluated as INFO (R := D, M := Dnew, S := STdisc) |
| outcomes for T2 | D, N, Dnew (discovery); C, K, F, F20 (confirmation) | fit | as above | O1, O5 |

The hub resnet20 / resnet56 rows 5000-9999 were read per sample by legacy work and every hub checkpoint was selected on
the full test set (T1 review B1), so no hub unit is an R unit. **Rows (global CIFAR-10 test indices; corrupt and fault
splits pair by row with the clean row):**

| layout | clean test | detection positives (POS) | clean negatives (B) | CAL (fit dump rows) | X4 per-tap / fusion calibration (fit dump rows) |
|---|---|---|---|---|---|
| fit | 0-4999 | corrupt rows 0-1999 | 3500-4999 | A = 2000-3499 | 2000-2749 / 2750-3499 (n 750) |
| eval | 5000-9999 | corrupt rows 5000-8499 | 8500-9999 | 0-4999 (whole fit split, T1 review A2) | 0-2499 / 2500-4999 (n 2500) |

CAL sets the temperature, the BBSE confusion matrix, the ATC thresholds, every univariate threshold, the batch-arm
frames and the stream calibration. The **best head statistic** is chosen on fit rows 2000-3499 (minimum AURC on clean
errors, ties by the list order) in both layouts, before any other decision; for that choice mspT uses a temperature
fitted on the same rows 2000-3499 (`head.temperature_headsel`), not the CAL temperature, which differs between the
layouts, so the choice is a function of rows 2000-3499 only and an eval run reproduces its fit run's choice.
Per-sample geometric signals are referenced to the TRAIN reference rows of the fit dump only (class means, kNN banks,
whitening, Mahalanobis), so the error targets use every clean row without held-out leakage.

## 3. Instruments

### 3.1 Dump access, the seal and the head path
Every dump opens through `atlas/b4_core.open_unit` (D7). A sealed dump (`results/b4c_*`) opens only in the confirmation
phase with `ATLAS_B4_UNSEAL=<P2>`; in the discovery phase every confirmation-only split (the 5 holdout corruptions, the 4
CIFAR-10-C extras, `exposure_global`, `soiling`) is invisible and refused if named (`ReadRefused`, a SystemExit that a
per-target `except Exception` cannot swallow). The head path is the **stored float32 model logits** (D2).

### 3.2 Signals
- **Head (the bar):** MSP, max logit, logit gap, energy (logsumexp), entropy, Gini, softmax margin, p-norm max logit,
  MSP at the CAL temperature (`atlas/b4_core.head_stats`). The **full head summary** of every joint model is the 10 sorted
  logits (linear) plus splines of gap, max logit, energy and entropy.
- **Geometry at the five functional taps** (stem, s1end, s2end, pre, penult: `meta.b4.taps.functional`; pre = the last
  spatial tap that is not a copy of the penult, D2): centre distance d1, centre margin d2 - d1, relative d1, nearest-centre
  vs head disagreement, the head-class centre excess, whitened top-10 norm, L2-normalised 10-NN distance (Sun et al.
  2022). At pre and penult also kNN purity (k 5/10/25/50), LID (MLE, k 20), Trust Score (alpha 0.1), class-conditional and
  relative Mahalanobis (pseudo-inverse semantics: eigen-directions below 1e-6 lambda_1 are dropped, not clamped; T1
  review E1), NECO. At the penult also the raw 10-NN distance (DO-3 and H), the whitened head-null and head-row residual
  energies (X2), the NC3+ alignment cos(x - mu, w_yhat - mean w) and ViM.
- **X4:** conformal p of the L2-kNN distance at each X4 tap (`meta.b4.taps.x4` = first tap, pre, penult) calibrated on
  the X4_1 rows, Cauchy-combined (Liu & Xie 2020), conformal p of the fused statistic calibrated on the X4_2 rows; the
  flag is p <= alpha = 0.05; its score is -log p.
- **X7:** nearest-train-centre label at every non-duplicate tap in forward order: prediction depth, agreement fraction,
  last disagreeing tap, late-half margin area (splits with every tap only; the holdout / extra splits carry the X4 taps
  only, D2).
- **Pixels:** the dumped pixel factors (`atlas.factors`), the comparator of every "beyond head AND pixels" test.

### 3.3 Targets (per sample; fit rows, eval analogues in the eval layout)
`err_clean` (wrong vs right, every clean row); `err_shift_s1/s3/s5` (the 10 discovery corruptions pooled); `flip_s3/s5`
(corrupt-wrong among clean-correct pairs); `conf_err_clean`, `conf_err_shift_s3` (the more confident half by the best
head statistic); `corrupt_{N,B,W,D}_s1/s3` (family POS rows vs B); `corrupt_any_s3_lofo` (leave one family out);
`famid_{F}_s3` (family F vs the other families among corrupt rows); `sev_5v1`; `ood_cifar100`, `ood_svhn` (vs B);
`fault_{deadpix,occlusion_disc[,exposure_global]}` (three levels vs the same clean rows B); confirmation only:
`err_shift_holdout_s3`, `err_shift_extra_s3`, `corrupt_holdout_s3_transfer`, `corrupt_extra_s3_transfer` (trained on
the discovery families and the clean negatives, scored on the unseen families against the clean negatives).

### 3.4 The joint model and the three-way rule (frozen; `atlas/b4_core`)
Ridge-logistic regression (lambda 1e-3 on standardised columns) on restricted cubic splines (knots at the 5/35/65/95%
training-fold quantiles), 5 folds by crc32 of the image row (the same row across clean, corrupt and fault versions stays
in one fold). For target x bundle: dAUC_joint with a paired DeLong test and a 95% group-bootstrap CI (B = 200 in
discovery and the S2 replay, 1000 in confirmation), dI = CE(head) - CE(head + bundle) in bits with its bootstrap CI,
dAURC, risk at coverage, FPR at 95% TPR, dTPR at 5% FPR. For the early bundles also the pixel twin: dAUC_pix and dI_pix
= CE(head + pixels) - CE(head + pixels + bundle).

| call | HEAD-ADDITIVE (`head_additive_call`) | beyond head AND pixels (`call_pix`) |
|---|---|---|
| ADDS | dAUC_joint >= +0.01 and the 95% CI of dI above 0 | dAUC_pix >= +0.005 and the 95% CI of dI_pix above 0 |
| BOUNDED (the functional "no") | the upper 95% bound of dAUC_joint < +0.02 | the upper 95% bound of dAUC_pix < +0.02 |
| INCONCLUSIVE (power) | otherwise | otherwise |

Power and size at realistic size are a known-answer test (`tests/test_b4_core.py`: 5000 rows, about 7% positives, 19
signals; a true dAUC of about +0.028 fires in >= 8 of 10 seeds, none in <= 1 of 10).

**Bundles:** `margin_pen`, `d1_pen`, `knnL2_pen`, `penult` (O1: margin, d1, L2-kNN, Trust Score, relative Mahalanobis,
purity 10 and 50, LID at the penult), `pre` (margin, relative d1, NCC disagreement, L2-kNN, purity 10 at the pre tap),
`s2`, `early` (L2-kNN and whitened norm at stem, s1end, s2end), `x4`, `trust_pen`, `relmaha_pen`, `local_pen`, `nc3p_pen`,
`null_pen`, `ood_std` (NECO, ViM, Mahalanobis), `traj`, `all_geom` (19 signals: bounds the total geometric increment).
Pixel twins for `early`, `s2`, `x4`, `pre`, `all_geom`.

### 3.5 Calibrated false alarms, X4 and DO-3
Every univariate detector: threshold = the conformal-rank (1 - alpha) quantile of the CAL rows (alpha 0.05), FPR on B
with a Wilson CI, TPR on every other split. The X4 record: FPR on B at alpha 0.05 (with n_cal_fusion and n_B), the
FPR curve at alpha 0.02-0.05 (rule 5), TPR per split. The DO-3 record (the AH-1 replication): the fraction of the
layout's clean rows whose raw penult 10-NN distance exceeds the q95 of the reference's self-excluded 10-NN distances,
with the reference nc1 from `atlas/b4_collapse.nc_block` (landmarks.py:71 formula).

### 3.6 Batch arm (X6)
200 seeded batches per condition at n = 16 / 64 / 256; calibration batches from CAL rows, clean evaluation batches from
B. Accuracy estimators AC, DoC, ATC-MC, ATC-NE; harm-like statistics H (median log r10 recentred and scaled on CAL),
sparse fraction, displacement, mean gap / energy, T2 at pre and s2end; every statistic mapped to accuracy by isotonic
regression fitted leave-one-split-out (one footing for the MAE); within-split Spearman and partial Spearman(H, loss |
ATC). Detectors at the clean tau95: T2 (penult, pre, s2end, pixels), the BBSE-explained covariate residual R = n ||(xbar -
sum_k pihat_k m_k) V / s||^2 with pihat = proj_simplex(C^-1 qhat), ||pihat - pi_CAL||_1, BBSDh chi-square, BBSDs (minimum
KS p over softmax dimensions, Bonferroni). Label skew: single class, Dirichlet 0.1 / 1 / 10, Markov stay 0.9 / 0.99.
Typing at n = 64: a geometry rule (T2_s2end -> penult per-tap conformal fraction -> BBSE) against a head rule (BBSDs ->
energy fraction -> BBSDh).

### 3.7 Streams (X3, `scripts/t1_streams.py`)
The probe builds its own per-row features from the dumps (level `streams` of the scoreboard's Unit) and writes them to
`--frames` (container disk). Calibration: every detector's threshold makes the in-control mean run length (200 streams,
censored at 12000) equal ARL0 = 2000 on CAL rows; the realised ARL0 and the false alarms per 1000 frames are measured on
held-out clean rows (B, or the eval clean rows). Scenarios (100 seeds): step at s1 and s3 (500 clean -> 1500 corrupt),
ramps s1 -> s3 -> s5, the benign brightness ramp, a single-class skew burst, its in-control null (the same timing,
clean throughout: every detector's own alarm rate, the X3-7 reference), CIFAR-100 / SVHN bursts, every fault split, the
stuck frame. Detectors: CUSUM (k 0.5 SD) on head scalars and on L2-kNN at the five taps and penult d1; the conformal
CUSUM on X4's p (eps 0.1); MEWMA (lambda 0.05) on the CAL-whitened 10-d frame of each tap and on the whitened pixel
factors; window-64 BBSDh and binned KS on MSP; the frame repeat check (h = 1: three identical frames); the harm gate
(alarm AND window H > 0.25). Metrics: P(false alarm before onset), P(detect within 500 frames) counting only streams that
alarm, the conditional and the censored delay, the gated rate, the ramp lead over the first frame at which the rolling
loss reaches 5 pt, and the change-information rate per tap and for the CAL-whitened head scalars and pixels with a
bootstrap CI. Frames are i.i.d. within a segment: rates are overstated relative to video (declared).

### 3.8 Outputs, timing and cost
`results/b4_t1/<id>_<fit|eval><tag>/scoreboard.json` and `results/b4_t1s/<id><tag>/streams.json` through
`atlas/b4_core.ProbeRun`: program, unit, phase, layout, tag, `code.sha256`, `code.core_sha256`, `repo_commit`, env,
nboot, the dumps with their meta sha256 and the splits read, `timing_s` per tap (`tap:<t>`), per target (`target:<t>`),
per block (`block:<b>`) and total, `max_rss_mb`. The `cost` block holds the reference bytes a deployed float32 detector
needs and the CPU ms per 1000 queries (under `cost.timing_s`, which the S2 replay compare ignores). A failing target or
block is recorded under `errors`, never silently dropped; a refused read stops the probe.

## 4. Per-model outcomes for T2 (fixed at P1; `experiments/b4/model_outcomes.schema.json`)
Every fit-layout scoreboard carries `model_outcomes`, validated against the schema before it is written:
- **O1** = dAUC_joint of the `penult` bundle over the full head summary on `err_clean` (fit rows 0-4999), cross-fitted by
  exactly the `atlas/b4_core.head_increment` arithmetic (the same crossfit calls and bootstrap key; a self-test and a
  pytest check the identity), with its 95% CI, dI and its CI, the three-way call, n_pos / n_neg and nboot.
- **O5** = mean over the 10 discovery corruptions at s3 of AUROC(X4, scored -p) minus the mean AUROC of the best head
  statistic among MSP, max logit, gap, energy, entropy (oriented by ERR_SIGN; the one with the largest mean AUROC over
  the same corruptions, a choice that favours the head); positives the corrupt rows 0-1999, negatives B = 3500-4999; all
  10 corruptions or no value.
T2's `collapse_laws.js` reads O1 and O5 from there and nowhere else; T1 makes no claim on them.

## 5. Known-answer tests (CLAUDE.md rule 4; run at S0, S1 and S2, hard)
- `tests/test_t1_scoreboard.py`: stdlib + numpy + atlas imports only; O1's bundle, O5's heads, corruptions, rows and
  negatives equal the schema constants; every bundle signal is oriented; the row ledger; AUC inside gap deciles removes
  gap information; KS identities and the vectorised p-value; BBSE recovers pi under pure label shift; isotonic fits;
  deterministic distinct batches; the pixel call; the schema validator accepts a valid record and rejects six invalid
  ones; centre margin, NCC disagreement and L2-kNN equal brute force; purity, Trust Score, NC3+, a zero head-null residual
  in the row space, relative Mahalanobis at the mean; Mahalanobis ignores dead ReLU axes; prediction depth on a
  constructed trajectory; the joint model equals `head_increment` exactly and finds a planted effect; a transfer target
  trains only on train_ok; O5 on constructed scores (value 0.25, best head gap; no value without all 10 corruptions);
  the discovery plan holds no confirmation-only split and a sealed dump never opens in discovery or without P2; the
  end-to-end self-test plus every key the evaluator reads.
- `scripts/t1_scoreboard.py --selftest` (synthetic b4 dumps, fit and eval layouts): a Bayes head leaves nothing to add
  (margin |dAUC| <= 0.015; O1 does not ADD); O1 = `head_increment`; a stem-only shift is seen by the early bundle (>= +0.3)
  and not by the penult margin, and beyond uninformative pixels; a head-null shift is invisible to the head and seen by
  the null residual; X4 transfers to an unseen family; X4 clean FPR in [0.02, 0.09] with n_cal 750 (fit) and 2500
  (eval); O5 > 0 with X4 AUROC >= 0.9 on the stem shift; the schema validates; the batch arm runs; append-only and
  touched-once refusals; the eval run reproduces the fit run's best head statistic and writes no outcomes.
- `tests/test_t1_streams.py` and `scripts/t1_streams.py --selftest`: CUSUM ARL0 and 1-SD delay against Siegmund's
  approximation (15%); MEWMA ~ chi2_10 in control and a frozen frame freezes it; the conformal CUSUM has no drift;
  contiguous deterministic segments; the repeat check fires at three identical frames; window counts equal brute force
  and chunking is exact; binned KS; the information rate of a unit shift is 0.5 nats and the whitened head rate is ~0
  where a merely standardised one exceeds 1 nat; end to end on synthetic frames and from synthetic dumps (a stem-only
  step caught by the stem MEWMA within 50 frames, not by the head CUSUM; a stream that never alarms is not a detection;
  the held-out ARL0 of cusum_msp in [1000, 4000], with the synthetic held-out pool rescaled to the cal pool's exact
  per-scalar mean and SD so that the check carries Monte Carlo noise only).
- `tests/t1_eval_fixture.js` (node, Windows): all 31 claims reproduce their constructed labels in the design and the
  flipped direction, give NOT_EVALUABLE with their fields removed and INCONCLUSIVE with inconclusive calls; a non-dry
  in-process run with an injected git passes every gate; a wrong code hash, a missing self-test, a replay FAIL, a
  withdrawn claim, a looser fraction, a rules file other than P2's and a perturbed committed anchor each have their
  frozen effect; a replay whose drift cannot be bounded (a killed t1 unit, a non-numeric diff, more diffs than listed,
  no `replay.json`) makes every label `NOT_EVALUABLE ... [REPLAY-DRIFT]`; a committed DO-3 sparse fraction 0.03 away
  passes the anchor gate and 0.04 away fails it; the CLI refuses to
  overwrite and fails provenance on a repository without the P1 / P2 history.

## 6. Pre-registered claims (frozen evaluator `scripts/t1_eval.js`)
Scopes and fractions: R = F + F20 (4 planned), R2 = the 8 spent seeds, M = the 12 C units, S = the 10 STconf stream
units. A scope needs at least 75% of its planned units evaluable, otherwise NOT_EVALUABLE; the first scope is the
claim's label, the others are reported as secondary labels. With 4 R units, 87.5% means 4 of 4, 75% 3 of 4, 50% 2 of 4.
Type `bool`: SUPPORTED when the fraction of true units reaches the threshold, else REFUTED. Type `bounded` (the claim
asserts BOUNDED): SUPPORTED when the BOUNDED fraction reaches the threshold; REFUTED when the ADDS fraction exceeds 1 -
threshold; otherwise INCONCLUSIVE. Type `adds`: the same with ADDS and BOUNDED exchanged.

| id | inventory | type | claim (frozen statistic and threshold) | scopes | predicted |
|---|---|---|---|---|---|
| X1-1 | CM-4 | bool | the best head statistic (chosen on fit rows 2000-3499) keeps AUROC >= MSP - 0.002 on never-read clean errors; a unit whose choice is MSP is not evaluable; NOT_EVALUABLE if > 25% of units choose MSP | R 0.90, R2 0.90 | SUPPORTED |
| X1-2 | CM-1, CM-2 | bounded | the penult centre margin is BOUNDED over the full head on clean errors | R 0.875, R2 0.875, M 0.75 | SUPPORTED |
| X1-3 | CM-1 ... CM-11 | bounded | all_geom is BOUNDED on clean errors | R 0.75, R2 0.75, M 0.75 | SUPPORTED |
| X1-4 | CM-11 | adds | the pre bundle ADDS on corrupt-split errors (s3) | R 0.5, R2 0.5, M 0.5 | SUPPORTED |
| X1-5 | CM-9 | adds (pixel) | the early bundle ADDS on corrupt-split errors (s3) beyond the head AND the pixel statistics | R 0.5, R2 0.5 | REFUTED |
| PC-2a | PC-2, ST-1 | adds | per-sample family identification (s3): the early bundle ADDS for >= 3 of 4 families (per unit: BOUNDED if >= 2 families BOUNDED) | R 0.75, R2 0.75 | SUPPORTED |
| PC-2b | PC-2 | adds (pixel) | ... and beyond the head AND the pixel statistics | R 0.75, R2 0.75 | REFUTED |
| X8-1 | CM-11 | bounded | Trust Score, purity 10/50, LID at the penult are BOUNDED on clean errors | R 0.875, R2 0.875, M 0.75 | SUPPORTED |
| X7-1 | CM-9 | adds | prediction depth / agreement ADDS on corrupt-split errors (s3) with dAUC >= 2 x its clean dAUC | R 0.5, R2 0.5 | REFUTED |
| X2-1 | ST-4 | adds | the head-null residual ADDS at +0.03 for corruption presence (s3) in >= 2 of 4 families (per unit: BOUNDED if >= 3 BOUNDED) | R 0.5, R2 0.5 | REFUTED |
| X4-1R | DO-6, DO-3 | bool | X4's clean FPR on never-read rows lies in the exact central 99% beta-binomial band at alpha 0.05 for n_cal 2500, n_test 1500: [0.0327, 0.0693] | R 0.875, R2 0.875 | SUPPORTED |
| X4-1M | DO-6 | bool | the same code, zero retuning, 12 new architectures: n_cal 750, n_test 1500: [0.0273, 0.0767] | M 0.8 | SUPPORTED |
| X4-2 | DO-3 | bool | REPLICATION (AH-1a, never-read rows): \|sparse fraction - (0.0224 - 0.108 log10 nc1)\| <= 0.035; NOT_EVALUABLE if the DO-3 anchor gate fails | R 0.75, R2 0.75 | SUPPORTED |
| X4-3 | DO-1 | bool | X4's TPR (flag p <= 0.05) beats the best head statistic's TPR (CAL 5% threshold) by >= 0.05 on >= 60% of the noise / blur / pixelate splits at s3 and s5 (needs >= 8 of 10) | R 0.75, R2 0.75, M 0.75 | SUPPORTED |
| X4-4 | DO-1 | adds | X4 ADDS for corruption presence of the noise AND the blur family (s3) | R 0.75, R2 0.75, M 0.75 | SUPPORTED |
| X4-5 | DO-1 | bool | trained on the discovery families, head + X4 beats the head by >= 0.02 AUROC on the 5 holdout families (s3) | R 0.75, R2 0.75, M 0.75 | SUPPORTED |
| X4-5x | DO-1 | bool | ... on the 4 CIFAR-10-C extras (NOT_EVALUABLE when absent from the volume) | R 0.75, M 0.75 | SUPPORTED |
| X4-6 | DO-4 | adds | X4 ADDS for far-OOD (SVHN vs clean rows) | R 0.75, R2 0.75, M 0.75 | SUPPORTED |
| X6-1 | LH-1, LH-2 | bool | ATC-MC's leave-one-split-out isotonic MAE <= H's (batch 64) | R 0.75, R2 0.75 | SUPPORTED |
| X6-2 | LH-1 | bool | mean within-split partial Spearman(H, loss \| ATC) >= 0.3 (batch 64) | R 0.5, R2 0.5 | REFUTED |
| X6-3 | ST-3 | bool | label skew: R_pre single-class FPR max <= 0.15 and mean <= 0.10, H mean <= 0.10, while penult T2 (single class, mean >= 0.5) and BBSDh (Dirichlet 0.1, >= 0.5) fire | R 0.75, R2 0.75 | SUPPORTED |
| X6-4 | ST-1, ST-5 | bool | typing at batch 64: geometry macro accuracy >= 0.85 and >= the head rule + 0.05 | R 0.75, R2 0.75 | SUPPORTED |
| X6-5 | LH-2 | bool | R_pre flags >= 80% of the motion / pixelate / snow s1 batches (64) at its clean 5% threshold | R 0.75, R2 0.75 | SUPPORTED |
| X3-8 | PC-7 | bool | every fault kind (deadpix, occlusion disc, holdout exposure): the best early-tap per-frame AUROC >= the best head AUROC + 0.10 | R 0.75, R2 0.75, M 0.75 | SUPPORTED |
| X3-1 | ST-9 | bool | I(s1end) > I(penult) with disjoint 95% CIs on >= 8 of 10 discovery corruptions at s1 | S 2/3 | SUPPORTED |
| X3-2 | ST-9 | bool | step at s1 (motion, pixelate, snow): best early-tap censored delay <= 0.25 x best head-only censored delay | S 2/3 | SUPPORTED |
| X3-3 | ST-9 | bool | ramps (motion, defocus): the best early-tap detector alarms >= 100 frames (median) before the rolling loss reaches 5 pt | S 2/3 | SUPPORTED |
| X3-4 | ST-9 | bool | realised held-out ARL0 in [1000, 4000] for >= 90% of the calibrated detectors | S 5/6 | SUPPORTED |
| X3-5 | PC-7 | bool | a stuck frame freezes the EWMA: penult and stem MEWMA detect within 500 frames in >= 90% of streams without pixel access (pixel repeat check >= 0.99 is a sanity clause; T1 review A3) | S 5/6 | SUPPORTED |
| X3-6 | LH-4 | bool | benign brightness ramp: harm-gated early alarms <= 0.2 while the ungated early detector >= 0.8 | S 2/3 | SUPPORTED |
| X3-7 | ST-5 | bool | skew burst: BBSDh and penult MEWMA >= 0.9, while the per-frame conformal flag alarms no more than in its own in-control null run with the same timing (scenario `null\|clean`: 500 + 300 + 500 clean frames): skew minus null p_det_within <= 0.10. A fixed bound would sit at the flag's own alarm rate (at ARL0 2000, P(alarm within 500 frames) in control is ~0.20), so a skew-blind flag would pass <= 0.2 in only ~50% of units | S 2/3 | SUPPORTED |

Why these predictions (discovery material is INFO): a collapsed penult is the nearest-class-centre rule (NC4), so
geometry at the penult restates the head on clean errors (X1-2, X1-3, X8-1); the head's confidence is miscalibrated
across corruption types while the pre-collapse NCC disagrees with it on 14-16% of samples (results/anomaly_h1/SESSION.md
:185; X1-4); early-tap error information is largely pixel-level (X1-5, PC-2b); a kNN distance is family-agnostic and
per-frame p-values stay uniform under pure prior shift (X4, X3-7); within-split rho(H, loss) was only 0.21-0.23 (X6-2);
exchangeability gives the X4 band (X4-1R/M: P(SUPPORTED | valid) about 0.96 for 4 of 4, 0.997 for 7 of 8, 0.9998 for 10
of 12).

## 7. Evaluation rules (`scripts/t1_eval.js`)
- **Labels:** SUPPORTED, REFUTED, INCONCLUSIVE, NOT_EVALUABLE; `INFO (discovery) <label>` in the discovery phase;
  `INFO (dry run) <label>` in a dry run.
- **Hard gates** (every claim `NOT_EVALUABLE (gate: ...)`): the self-tests (a check dir holding PASS records of
  `selftest_t1_scoreboard.json` and `selftest_t1_streams.json`; only `selftest_*.json` is parsed, T1 review A1); provenance
  (confirmation: P1 <= P2 <= HEAD, `freeze_P2.json` and `t1_rules.json` at P2, every T1 file unchanged from P2, every
  output's `code.sha256` and `code.core_sha256` equal to the scripts at P2, `repo_commit` in `--p-run`, nboot 1000, sealed
  dumps opened with `ATLAS_B4_UNSEAL` = P2; discovery: outputs from the P1 scripts, P2 for `_p2` re-probes, nboot 200);
  the read guard (no discovery output opened a sealed dump or read a confirmation-only split; no confirmation unit and
  layout scored twice); the rules (confirmation: `experiments/b4/t1_rules.json` present, valid and byte-identical to the
  file committed at P2; when this gate fails nothing from the file is applied, not even a withdrawal).
- **Replay (D15):** in the confirmation phase a `replay.json` (S2 anchors, tolerance rel 1e-6 / abs 1e-9) that does not
  PASS appends `[REPLAY-DRIFT]` to every label, and every per-unit decision whose deciding number lies within the largest
  listed drift of its threshold becomes not evaluable. The drift is **unbounded** (every threshold decision becomes not
  evaluable) when the record cannot bound it: a t1 anchor unit NOT_EVALUABLE (e.g. its replay was killed by its time
  limit), a FAIL unit with more diffs than `first20` lists, a diff line without a numeric pair (a missing key, a length, a
  string), no t1 unit in `replay.json`, or no `replay.json` at all (`gates.replay.drift_unbounded`).
- **Anchor gates (T1 review B6, D15):** on the discovery boards of the D units with a committed Atlas anchor: the T1
  margin AUROC on clean errors within 0.002 of `margin_typeb.auc_margin_wrong` (13 margin runs); the DO-3 sparse fraction
  within 0.035 of `knn_density.splits.test.sparse_frac` and nc1 within 1% of `neural_collapse.nc1`. The 0.035 covers the
  committed estimator's subsampling (3000 of the 10k reference radii for the q95 threshold, 3000 of the 5000 test rows;
  atlas/invariants/density.py:23-41; T1 uses all rows): per-unit SD 0.004-0.0097, so 0.02 would fail at least one of 16
  units in ~16% of runs with a correct instrument. A margin FAIL makes X1-2
  NOT_EVALUABLE; a DO-3 FAIL makes X4-2 NOT_EVALUABLE.
- **Schema gate (INFO for T1):** every fit board's `model_outcomes` must validate; failures are listed per unit for T2.
- **Functional units (D14, never a verdict):** medians and ranges over units of dAUC, dTPR at 5% FPR and dAURC per target
  x bundle; CPU ms per 1000 queries and reference bytes per signal family; X4's FPR curve at alpha 0.02-0.05 beside the
  exact band curve; false alarms per 1000 frames per stream detector.

## 8. Discovery -> P2 -> confirmation (D7, D8)
1. **S1** runs the discovery probes (`b4_reg.py jobs --phase discovery`: T1 fit on the anchors first, then D, N, Dnew;
   streams on STdisc) and extracts every confirmation dump sealed.
2. **Before P2** (Windows): `node scripts/t1_eval.js --phase discovery --p1 <P1> --p-run <S1 HEAD> --json
   results/b4/eval_discovery.json --emit-rules experiments/b4/t1_rules.json [--amend <file>]`. The rules file lists every
   claim ACTIVE with its P1 fractions and the discovery labels (INFO). The only permitted P1 -> P2 changes (D8): a claim
   withdrawn to INFO with a reason (b), a stricter fraction (c; `--amend` refuses a looser one and the evaluator makes a
   looser one NOT_EVALUABLE), a probe bug fix with a new known-answer test (d; triggers the S2 re-probe), an evaluator bug
   fix with the fixtures updated, never changing a rule (e).
3. **S2** replays the anchors, then scores the confirmation units once (`ATLAS_B4_UNSEAL=<P2>`): T1 fit and eval on F and
   F20, fit on C and K, streams on STconf, eval on R2.
4. **Evaluation:** `node scripts/t1_eval.js --p1 <P1> --p2 <P2> --p-run <S1 HEAD>,<S2 HEAD> --json
   results/b4/eval_t1.json`. A bug found after the run is fixed only by a committed amendment; the evaluator then writes a
   new file (`eval_t1_v2.json`) and both are reported (the A4b / ANOMALY_H1 pattern).
Registered cuts (D10 section 2; `--cut` or `freeze_P2.json`): R2 (its claims' secondary labels become NOT_PLANNED), F20
(R = the two resnet56 fresh seeds), STconf; X6 / X7 / X2 can be switched off in S2 only by withdrawing those claims at P2.

## 9. What runs where
Every T1 program is CPU-only numpy. S1 (4090 pod, niced beside the GPU chain): 21 + 2 fit scoreboards and 3 stream units.
S2 (CPU pod or 4090): 4 fit + 4 eval (F, F20), 12 C fit, 4 K fit, 8 R2 eval scoreboards, 10 stream units. Time limits
come from `scripts/pod_b4.sh` in S1 (t1 5400 s for 64-d, 7200 s for wide units and every eval layout; t1s 3600 s) and
from `b4_timeouts.js` in S2 (per-tap and per-target timing is recorded for that purpose). The node evaluator runs on
Windows.

## 10. What the inventory's "beyond the output head?" column becomes
Mapping rule: a SUPPORTED `bounded` claim writes NO (functional no, with the unit count); a SUPPORTED `adds` claim writes
YES (scope CIFAR-10 / CIFAR-10-C, the named units); REFUTED writes the opposite; INCONCLUSIVE writes "INCONCLUSIVE
(power)" and keeps the row UNTESTED; NOT_EVALUABLE leaves it UNTESTED. Every cell cites `results/b4/eval_t1.json` and the
claim id.

| row | claims |
|---|---|
| CM-1, CM-2 | X1-2 (margin vs the full head), X1-3 |
| CM-4 | X1-1 (which head statistic to use; MSP's share is reported) |
| CM-9 | X1-5, X7-1 |
| CM-11 | X1-3, X1-4, X8-1 |
| DO-1 | X4-3, X4-4, X4-5, X4-5x |
| DO-3 | X4-2 (replication on never-read rows); the across-architecture law is T2's M-DO3 |
| DO-4 | X4-6 |
| DO-6 (new, D19) | X4-1R, X4-1M, X4-3 |
| LH-1, LH-2 | X6-1, X6-2, X6-5 |
| LH-4 | X3-6 |
| ST-1, ST-5 | PC-2a, X6-4, X3-7 |
| ST-3 (successor R) | X6-3 |
| ST-4 (successor X2) | X2-1 |
| ST-9 (new, D19) | X3-1 ... X3-4 |
| PC-2 | PC-2a, PC-2b |
| PC-7 (new, D19; with lane S) | X3-5, X3-8 |

## 11. Risks, confounds, rule 8
- **Power on clean errors:** about 280-390 errors per 5000 rows; the three-way call keeps an absence of evidence from
  being written as NO. With 4 primary R units a single INCONCLUSIVE unit can move a label to INCONCLUSIVE; R2 and M are
  reported beside it.
- **Too-good numbers:** pixel and early-tap AUROC near 1.0 is expected for noise at s5 and for injected faults by
  construction; X1-5, PC-2b and X3-8 compare against pixel statistics or the head, and the random-init nulls are reported.
- **Selection:** every hub checkpoint (the C units) is best-on-test upstream; the F / F20 fresh seeds are last-epoch with
  a disclosed aggregate accuracy print (D13).
- **Transfer designs** can learn corruption identity: folds are grouped by image row and the families are held out.
- **i.i.d. streams** overstate rates (declared); real video is a later step.
- **CPU replay drift:** S2 may run on another host; the replay gate measures it and near-threshold decisions are
  withheld.
- **Python never executed at P1 authoring time:** the first execution is S0 (pytest and the self-tests); S1 stops on any
  failure (hard gates).

## 12. T1 review items (T1-scoreboard.review.md) and their disposition
A1 applied (only `selftest_*.json` parsed; fixture has a `.log`). A2 applied (whole-fit-split calibration in the eval
layout; exact bands). A3 applied (X3-5 restated; repeat check h = 1; frozen-frame test). A4 applied through D9 / D15
(tolerance replay in `b4_reg.py`, read by the evaluator; replay outputs never enter a unit set). A5 applied. B1 applied
(no hub unit in R; streams on STconf). B2 overridden by D13 (fresh seeds F / F20 from the frozen script). B3 dissolved by
D2. B4 superseded by D2 / D7 (sealed extraction, partial dumps refused). B5: X4-2 labelled REPLICATION; X4-2M moved to
T2 (M-DO3). B6 applied (anchor gates). C1 applied (three-way calls, BOUNDED at +0.02). C2 applied (in
`tests/test_b4_core.py`). C3 applied (functional block, cost block). D1-D6: moved to T2 with the collapse arm (D1). D7
superseded by D2's pre-tap rule. E1-E4 applied. E5 applied (X4-5x). E6 applied (salt 1, `atlas/faults.py`). E7, E8 moot
(the head export is `scripts/b4_weights.py`). F1-F5 superseded by D9-D11. G applied (`tests/t1_eval_fixture.js`).
