# T2_COLLAPSE: the collapse arm of batch 4 (model-level collapse laws and the owner's hypothesis)

**Status.** Pre-registration text for P1 (`atlas: batch 4 pre-registration (P1)`). Written against design HEAD `2b561a9`
and the binding decisions in `docs/plans/B4_INTEGRATION.md` (D1-D19), which win wherever this file and they differ.
Nothing here is evidence. The node evaluator and its fixture ran locally (section 10); the Python probe has never
run (no local interpreter), and its first execution is the S0 pytest plus `scripts/collapse_probe.py --selftest`.

**Owner's request (binding).** "Collapse도 포함시켜줘. 모델간에 차이를 예측해내는데 큰 역할을 할거같은데" (include collapse; it
should play a big role in predicting how models differ). T2 tests that hypothesis once, as the single PRIMARY claim of
batch 4 (D12). It asks the controller's question: does a collapse coordinate that can be measured **without labels**, on
unlabelled clean frames of a new backbone, predict what that backbone's detectors will do, **better than** its accuracy,
its head's own confidence summary, and the cheapest guess (a constant)?

---

## 0. One screen

| | |
|---|---|
| **PRIMARY (D12)** | On the 12 sealed confirmation architectures (C), for each of five outcomes O1-O5, the frozen law "accuracy + best label-free collapse coordinate" must have MAE <= 0.8 x the best of three rivals (constant, accuracy, accuracy + best head coordinate), Spearman(pred, obs) >= 0.6, and a family-demeaned Spearman > 0. SUPPORTED if >= 3 outcomes WIN; REFUTED if <= 1 WIN and >= 3 LOSE (ratio > 0.9); otherwise MIXED; NOT_EVALUABLE if < 4 outcomes have >= 10 units. |
| **Outcomes** | O1 penult-bundle dAUC_joint over the full head on clean errors (T1); O2 H10, the harm grade at 10 pt of loss (T2); O3 L2-normalised 10-NN AUROC - best head AUROC, CIFAR-100 (T2); O4 mean MSP - accuracy under the 10 discovery corruptions (T2); O5 X4 flag AUROC - best head AUROC at s3 (T1). Each needs labels or shifted data, so a law is the only way to know it for a new backbone. |
| **Laws** | Fitted by frozen code on the 21 discovery nets (D16 + Dnew5), frozen at P2 (`experiments/b4/laws_frozen.json`), applied once to C in S2; never refitted (except the registered D7 re-probe path). |
| **Secondaries** | 25 items (section 5): the labelled twin (S-PL), 3 mechanism items (M-DO3, M-LF, M-NC4), 1 instrument check (IC-P2), 6 correlation items (T1 MP-1, MP-3, MP-5, MP-6, MP-7 and T2 P2(b); MP-2 is M-LF, MP-4 is INFO, MP-8 is P8), 3 family fixed-effect twins, 6 single-coordinate law items, P3b, P5c, P7 (blind, registered in `experiments/b4/prereg_p7.json`), P8 and MP-K (the knob lane). Holm over the 22 items with a permutation or binomial p-value. |
| **Predicted** | PRIMARY **MIXED** (O2, O3 WIN; O1 LOSS; O4, O5 neither), stated before any data (section 7). |
| **Code** | `scripts/collapse_probe.py` (numpy; one unit per call), `scripts/collapse_laws.js` (node; fit, evaluate, dry run, self-test; contains the PRIMARY rule), `experiments/b4/prereg_p7.json`, `tests/test_collapse_probe.py`, `tests/collapse_laws_fixture.js`. |
| **Pod** | S1 (discovery, niced CPU lane beside the GPU chain): 23 units, about 1 CPU-h. S2 (confirmation, first in the lane): 20 units (C12, K4, F2, F20 2), about 2 CPU-h. No GPU work of its own. |

**What a controller gets if the PRIMARY holds.** From a few thousand unlabelled clean frames of a new backbone: where its
HOLD band's edge sits (O2), whether its penult density detector beats its head on novelty (O3), how much to discount its
confidence under shift (O4), and whether its geometry or its multi-tap flag adds anything beyond its head (O1, O5), with
errors stated in each outcome's own units. Today each of these needs labels, shifted data or a per-backbone sweep.

---

## 1. Units, rows and phases

| role (registry) | units | T2 phase | use |
|---|---|---|---|
| ANCHOR | resnet20_hub, resnet56_hub | discovery; replayed in S2 | rule 6; the replay drift (D15) |
| D (16) | r20 hub, s1-s4; r56 hub, s1, s2, e10...e70, s12m, s13m | discovery | the fit (11 run groups; the e-ladder is one group) |
| Dnew (5) | resnet32, vgg11_bn, mobilenetv2_x0_5, shufflenetv2_x0_5, repvgg_a0 | discovery | the fit (5 more run groups; one per family) |
| N (2) | resnet20_rand, resnet56_rand | discovery | INFO (coordinate range) |
| C (12) | resnet44, vgg13/16/19_bn, mbv2 x0_75/x1_0/x1_4, shufflenet x1_0/x1_5/x2_0, repvgg a1/a2 | confirmation (sealed) | the PRIMARY and every secondary |
| F (2), F20 (2) | resnet56_s31/s32, resnet20_s31/s32 | confirmation (sealed) | MP-K baselines (F); fresh-seed predictions (INFO) |
| K (4) | resnet56_s31/s32 x {ls10, wd5e5} | confirmation (sealed) | MP-K |

- **Dump and rows (D4).** The probe opens only the unit's **fit** layout through `atlas.b4_core.open_unit` (the D7 seal).
  Test rows 0-4999 carry every held-out coordinate; split halves are the even and odd rows; rows 5000-9999 are never
  read by T2. H = rows 0-1999 are the clean sources of the CIFAR-10-C rows 0-1999; A = 2000-3499 is used only for
  selection and calibration (the P3b head choice, the IC-P2 conformal calibration); B = 3500-4999 are the clean
  negatives (OOD, the per-tap shift AUROC, IC-P2). OOD rows 0-1999. The reference is the atlas train reference (10,000
  train rows).
- **Discovery reads the 10 discovery corruptions only** (the D7 split filter makes the 5 holdout corruptions, the 4
  extras and the holdout faults unreadable in the discovery phase). In the confirmation phase the probe also reads the
  holdout corruptions and the extras, only for the INFO clause of P5c.
- **Role guard.** The probe refuses a unit whose roles do not belong to the phase (discovery: ANCHOR, D, N, Dnew;
  confirmation: C, K, F, F20).
- **Selection caveat (recorded, not fixable here).** Every hub checkpoint (both D hubs, Dnew and C) was selected
  upstream on the full CIFAR-10 test set; Atlas's own seeds are last-epoch. This touches accuracy, which is why accuracy
  is a rival and never a coordinate of the collapse law.

---

## 2. Prior art: starting points adopted and refined

| method | adopted as | Atlas refinement | source |
|---|---|---|---|
| NC1-NC4 (Papyan, Han & Donoho 2020) | the definitions | every tap; train and held-out rows; the train-test gap as a coordinate | https://arxiv.org/abs/2008.08186 |
| Collapse holds on train, not on test (Hui, Belkin & Nakkiran 2022) | mechanism of DO-3 | P2(b): the train-test gap orders the over-alarm | https://arxiv.org/abs/2202.08384 |
| NC on target data ranks pre-trained models (NCTI; Wang et al., ICCV 2023) | precedent for "collapse predicts model differences" | label-free coordinates; outcomes are detector behaviours; rivals are accuracy, the head and a constant | ICCV 2023 open access |
| RankMe (Garrido et al. 2023); alpha-ReQ (Agrawal et al. 2022) | label-free spectral coordinates | the class-aware gap is **cross-validated** (width-robust); erank and alpha are INFO | https://arxiv.org/abs/2210.02885 |
| Stronger NC improves OOD detection (Harun, Gallardo & Kanan 2025) | the direction of O3 / P4 / MP-6 | a per-model law with a head baseline (best of MSP, max logit, logit gap, energy, entropy) | https://arxiv.org/abs/2502.10691 |
| Deep kNN with L2 normalisation (Sun et al. 2022) | the density detector of O3 / P4 | the head bar picked on the same rows (a choice that favours the head) | https://arxiv.org/abs/2204.06507 |
| VCI (Xu & Liu 2023, Def. 5.3) | stable collapse coordinate (INFO) | reported per tap | https://arxiv.org/abs/2306.03440 |
| Law of equi-separation (He & Su 2023) | reading of the per-tap nc1 profile | slope of log10 nc1 on depth (INFO) | https://arxiv.org/abs/2210.17020 |
| Split conformal calibration | IC-P2 and the band | the exact beta-binomial band of `scripts/b4_stats.js` (D14) | knowledge base |

The references above were opened by the T2 designer on 2026-09-23 (T2 design, section 2); they are cited here without
re-opening. Prior art never lowers a rank here: it is the starting point, and the refinement is what is tested.

---

## 3. The instrument: `scripts/collapse_probe.py`

One fit-layout dump per call; numpy + `atlas.b4_core` + `atlas.b4_collapse`; the D5 CLI contract (`--registry --unit
--phase --out`, `--selftest --selftest-out`); `timing_s` per tap (`tap:<t>`) and per target (`target:<name>`,
`target:taps:<t>`), `timing_s.total`, `max_rss_mb`, `code.sha256`, `repo_commit` (via `ProbeRun`). Deterministic: the
only random draws (the P3b bootstrap) are keyed by `rng_for`, so the S2 replay compares equal.

### 3.1 Coordinates (every tap; float64; K = 10)

| key | definition | labels |
|---|---|---|
| `nc1_tr` | tr(Sigma_W Sigma_B^+) / K on the train reference with Sigma_B inverted on its top K-1 eigenpairs above 1e-10 lambda_1 (`nc_block.nc1_trim`) | train |
| `nc1_tr_verbatim` | the atlas/invariants/landmarks.py:71 value (numpy-default pinv); `nc1_pinv_rel_dev` = their relative difference | train |
| `nc1_te` | the same (trimmed) on test rows 0-4999 with their own class means | test |
| `nc1_tetr` | test scatter around the TRAIN class means over the train Sigma_B (trimmed) | both |
| `plnc1_te` | trimmed nc1 on test rows 0-4999 with the head's argmax (stored logits) as labels | **none** |
| `g_cv_te` | lambda_{K-1}/lambda_K of cross-validated eigenvalues (eigenvectors from the even rows, variances on the odd rows, and the reverse) | **none** |
| `topk_frac_te` | share of test-row variance in the top K-1 principal components | **none** |
| `ncc_agree_te` | NC4: nearest train class centre equals the head's argmax | none |
| INFO | vci, bt, etf_dev, norm_cv, centre-distance CV, g_raw, erank, pr, khat, alpha, head_center_cos, nc3_dist, the equi-separation slope | - |

Penult only: head-only coordinates `sat999_te` (share of rows with MSP >= 0.999), `gap_mean_te` (mean top-1 minus top-2
logit), `msp_def_te` (1 - mean MSP); `acc_te` / `err_te` (labels); `ece_te` (15 bins); `tt_gap_log10` =
log10(nc1_tetr / nc1_tr); `te_tr_gap_log10` = log10(nc1_te / nc1_tr); `cdepth_pl` = log10 plnc1(pre) - log10
plnc1(penult) with pre = `meta.b4.taps.pre` (the D2 rule, recomputed and checked).

**Instrument decision (rule 8, stated before data).** The verbatim formula uses numpy's pinv with a 1e-15 relative cutoff
on a rank-9 Sigma_B. At a 512-2048-d penult a rounding singular value just above that cutoff would inflate the verbatim
nc1 by orders of magnitude. The laws therefore use the trimmed inverse, which equals the verbatim value whenever the
verbatim pinv resolves exactly the 9 class directions; both are recorded, and the D15 gate (committed atlas nc1
reproduced within 1%, run by `--fit` on the 16 old nets) reads the verbatim one. The same applies to `plnc1_te` and
`nc1_tetr`. The core's `label_free_coords` (verbatim plnc1) is reproduced exactly by the probe and checked in the
self-test.

### 3.2 Targets and outcomes (penult unless stated; scores oriented so that larger = more error-, shift- or OOD-like)

| block | definition | feeds |
|---|---|---|
| `do3` | train-referenced 10-NN radius false-alarm rate on test rows 0-4999 at the reference's self-excluded q95 of log r10 (AH-1's quantity on all rows) | M-DO3, M-LF, P2(b), MP1, FE1 |
| `ic_p2` | conformal p = (1 + #{cal >= s}) / (n + 1), cal = rows A, flag p <= 0.05 on rows B; the rule-5 curve at alpha 0.02-0.05 | IC-P2 |
| `margin` | clean errors on rows 0-4999: AUROC of margin (d2 - d1 to the train centres), d1 and the five head statistics; `lead_md` = margin - d1; `gap_minus_msp` | P3a, MP3, MP5, FE3 |
| `p3b` | head statistic picked on rows A (best AUROC of the five); d = AUROC(margin) - AUROC(pick) on rows H + B; 99% row-bootstrap CI (`run.nboot` draws: 200 discovery, 1000 confirmation) | P3b |
| `ood` | CIFAR-100 / SVHN rows 0-1999 vs clean rows B: L2-normalised 10-NN, raw 10-NN, the five head statistics; `knn_l2_minus_besthead` | **O3** (CIFAR-100), P4 and MP6 (SVHN) |
| `harm` | per discovery corruption x severity (30 splits): H = (median log r10(split) - median log r10(its 2000 clean source rows)) / (q95 - q50 of the reference self log r10); cost = paired accuracy loss in pt; OLS cost ~ H; H10 = (10 - a) / b when b > 0 | **O2** (H10), P5a (slope), P5b |
| `harm_committed` | AH-2(b) as committed (clean median and accuracy over rows 0-4999): HOLD band H <= 0.25 -> cost <= 8 pt; violations over the 30 splits; holdout and extras in the confirmation phase only (INFO) | P5c |
| `overconf` | mean over the 30 discovery splits of (mean MSP - accuracy), stored logits | **O4**, S7 |
| `taps` | per non-duplicate tap: AUROC of the 10-NN radius to the reference at that tap, corrupt rows 0-1999 (10 corruptions at s1 and s3) vs clean rows B; the tap and the head statistic are picked on the even rows, scored on the odd rows; `p6b` = best tap - best head | P6b |
| `targets_se` | split-half noise, abs(half1 - half2) / 2, of every law target (even / odd rows) | the `noise_limited` flag |

O1 and O5 are read from T1's `model_outcomes` (`experiments/b4/model_outcomes.schema.json`, fixed at P1) in
`results/b4_t1/<id>_fit<tag>/scoreboard.json`, and nowhere else.

### 3.3 Self-test and tests (the instrument is validated before any real read; rule 4)
`--selftest` writes a synthetic fit-layout dump (simplex-ETF classes, a float16 duplicate of the penult, a noisy pre tap,
30 corrupt splits, one confirmation-only split, two OOD sets) and runs the real CLI on it. Its 27 checks: nc1 =
sigma^2 (K-1)^2 / (K m^2) within 5%; trimmed = verbatim nc1; g_cv = (a + sigma^2) / sigma^2 within 10%; khat = K - 1;
the label-free candidates equal `atlas.b4_collapse.label_free_coords`; plnc1 <= nc1_te; NC4 agreement; DO-3 and IC-P2
false alarms ~0.05 under exchangeability; the harm fit (30 splits, slope > 0); O4 against a direct recomputation (1e-12);
OOD ordering; the P3b CI; the duplicate tap excluded; the D2 pre tap; the confirmation-only split never read in
discovery and read in confirmation; the timing and provenance contract; append-only; replay determinism; the D7 seal; the
role guard; harm and HOLD-band known answers. `tests/test_collapse_probe.py` adds the import restriction, helper known
answers, the CLI contract, and a **schema pin**: every probe.json path `collapse_laws.js` reads (its `PROBE_KEYS` block)
must exist in a real probe output.

---

## 4. The PRIMARY claim (D12; `ownerPrimary` in `scripts/collapse_laws.js`)

**Laws per outcome**, fitted by `collapse_laws.js --fit` on the D21 nets (trained, evaluable, all candidate coordinates
finite), frozen at P2:

| law | form | selection on D21 |
|---|---|---|
| const | mean of the outcome over the full-recipe discovery nets (13: the r20 hub and s1-s4, the r56 hub, s1, s2, Dnew5) | - |
| acc | o ~ a + b t(err_te) | t = identity or log10, the lower leave-one-run-group-out (LORO) MAE |
| acc_head | o ~ a + b1 t(err_te) + b2 h | h among sat999_te, log10 gap_mean_te, log10 msp_def_te (LORO) |
| acc_coll | o ~ a + b1 t(err_te) + b2 z | z among log10 plnc1_te, log10 g_cv_te, topk_frac_te (LORO) |
| acc_nc1 | o ~ a + b1 t(err_te) + b2 log10 nc1_tr | (the labelled twin S-PL) |

OLS on standardised covariates; LORO over the 16 run groups (the e10-e70 ladder is one group; at least 4 groups); an
outcome with fewer than 12 usable discovery nets is not frozen. The accuracy transform is chosen once per outcome and
shared by the three accuracy-based laws, which gives the rivals the same flexibility as the collapse law.

**Rule (verbatim from `INTEGRATION/b4_primary.js`).** Units: C with `evaluable` (a PASS weights record, i.e. not
NOT_EVALUABLE for the 8-hex tag, and a PASS head check), a finite outcome and all four predictions. Per outcome:
- WIN = MAE(acc_coll) <= 0.8 x min(MAE const, acc, acc_head) AND Spearman(pred, obs) >= 0.6 AND the family-demeaned
  Spearman > 0 (`b4_stats.famDemeanedSpearman`, families of >= 2 members, >= 6 units: resnet44 drops out, n = 11);
- LOSS = ratio > 0.9; NOT_EVALUABLE with fewer than 10 units.
- Verdict: NOT_EVALUABLE if fewer than 4 outcomes are evaluable; SUPPORTED if >= 3 WIN; REFUTED if <= 1 WIN and >= 3
  LOSS; otherwise MIXED.

**Annotations (never change the verdict).** Per outcome: MAE of each law in outcome units (the functional reading, section
9), the number of EXTRAP units (a coordinate outside the D21 range widened by 25%; if more than half, the outcome reads
"extrapolation holds", not "the law transfers"), and the rule-8 block: pooled rho, family-demeaned rho, the partial
Spearman on log10 penult width, the partial on accuracy, and RULE8-CHECK when |rho| >= 0.95.

**Replay drift (D15).** If the S2 replay compare of the anchors FAILs, every label is tagged REPLAY-DRIFT. The drift
itself counts per program: probe fields drift only when a `t2:` unit of `replay.json` did not PASS, the T1 outcomes O1 /
O5 only when a `t1:` unit did not PASS (a T3S-only failure tags and moves nothing). The evaluator recomputes the drift of
each outcome from the anchors' S1 records (the ones the laws were fitted on) and their S2 replay records (|dy| + the
largest change of the four predictions). An outcome whose drift is at least its WIN margin |MAE(acc_coll) - 0.8 min
rival MAE| is NOT_EVALUABLE (REPLAY-DRIFT). A drift the replay cannot bound (a replay output missing) is infinite.

**Replay drift of the secondaries (verifier T2-3; replaces the blanket "relative drift > 1e-3" rule).** Each input an
item reads gets its drift d = the largest absolute S1 - S2 difference over the two anchors, in the input's own
(transformed) units, under the program trigger above; a law's prediction moves by at most sum_j |beta_j / sd_j| d_j.
Each decision is NOT_EVALUABLE only when the drift of its own statistic reaches its margin |stat - bar|:
- Spearman items (MP-1 ... M-NC4, the rho clauses of the law items, M-LF, P8): |delta rho| <= 12 k / (n (n - 1)), k =
  the number of unit pairs whose x or y values lie within 2 d (0 when no order can change);
- family-demeaned Spearman (FE1, FE3, FE9, the rho_fam clause of the law items) and Kendall (P7): exact when no order
  can change, otherwise unbounded;
- MAE clauses: MAE(law) moves by <= dy + d_pred, MAE(const) by <= dy (triangle inequality);
- counts against a band or bound (M-DO3, IC-P2, P5c, P8 within 0.1 dex, P7 families, FE9 outcomes, MP-K cells): the
  count can cross its need only through units within the drift of the band edge;
- P3b: a unit whose plnc1 lies within the drift of theta, or whose ci_lo / d lies within the drift of its bound;
- S-PL: the PRIMARY's per-outcome rule with the acc_nc1 predictions.
A conjunction (e.g. rho >= 0.6 AND rho_fam > 0 AND the MAE clause) is NOT_EVALUABLE only if a clause within its drift
could change its value. The relative measure is kept as INFO with an absolute floor, |x - y| / max(|x|, 1e-3).

**Output selection (verifier T2-1).** The evaluator scans every tag of a unit (`''`, `_r<k>`, `_p2`; never
`*_s2replay`) under `results/b4_t2/<id><t>/probe.json` and `results/b4_t1/<id>_fit<t>/scoreboard.json`, keeping records
whose `unit` (T1: `model_outcomes.unit`) is the unit. A confirmation unit uses its single confirmation-phase record per
program; two confirmation records are a touched-once violation and make that unit not evaluable; a record in another
phase trips the guard. A discovery unit uses its highest relaunch (`_r<k>`), and under `--refit-tag _p2` its `_p2`
re-probe. The T2 probe and the T1 scoreboard are chosen independently (after a D10 relaunch they may carry different
tags); the chosen directories are recorded in `info.units[].sources`. `--tag` / `--disc-tag` only filter to one tag.

**Provenance guard (verifier T2-2).** Every probe records `code.sha256` (collapse_probe.py), `code.core_sha256`
(atlas/b4_core.py) and `code.collapse_sha256` (atlas/b4_collapse.py). `laws_frozen.json` stores them per fitted record
with the T1 scoreboard's `code.sha256` / `code.core_sha256` behind O1 / O5, the record paths and file hashes, and the set
of each (`code_sets`); `--fit --p1` refuses unless each set has one value equal to the file at P1. `--evaluate` requires
each hash to take one value across the fitted discovery records (or the refit) and the confirmation records, equal to the
file at P2; a P1 -> P2 difference without `--refit-tag _p2` is `NOT_EVALUABLE: probe code changed P1->P2 (<file>): rerun
with --refit-tag _p2`. Without a refit, every fitted probe and scoreboard must still hash to the value stored at the fit
(T2 review #17), and the evaluation must read the same files. `--evaluate` without `--p1`, `--p2` and `--p-run` emits no
verdict (guard) and prints `[UNOFFICIAL]`; `--p-run` matches commit prefixes (verifier T2-4).

---

## 5. Secondaries (Holm within T2)

Every item is scored on the C units (evaluable, n >= 10) unless stated. `p` is one-sided in the registered direction:
a permutation p = (1 + #{perm >= obs}) / (B + 1), B = 20,000 (seeded by the item id), unless stated. **Holm** (alpha
0.05, `b4_stats.holm`) runs over the 22 items that carry a p-value; a SUPPORTED rule becomes the final SUPPORTED only if
Holm rejects, else "RULE-MET, HOLM-NS". Three items are bound rules or an instrument check with no p-value (IC-P2,
P3b, P5c); they keep their rule verdict and are never promoted alone.

| id | source | rule (SUPPORTED if ...) | p-value |
|---|---|---|---|
| S-PL | T1 MP-9a | the PRIMARY rule with acc_nc1 (labelled log10 nc1_train) in place of acc_coll | permute nc1_train across C; statistic -mean MAE ratio |
| M-DO3 | T1 X4-2M, T3 CZ-1, T2 P1 (mechanism: directly measurable, T2 #21) | the committed AH-1 law, abs(FPR - (0.0224 - 0.1080 log10 nc1_tr)) <= 0.035, in >= 75% of C (9 of 12); rule-5 curve over bands 0.02-0.05; CZ-7c INFO | permute nc1_train; count inside |
| M-LF | T3 CZ-2 / CZ-7a, T1 MP-2 (mechanism) | the label-free coordinate with the largest abs Spearman with the DO-3 FPR on D21 (sign frozen) ranks C: sign x rho >= 0.7 AND abs(rho) > the abs rho of every head coordinate and of err_te | permute the coordinate |
| MP1 | T1 MP-1 | rho(log10 nc1_tr, DO-3 FPR) <= -0.6 | permutation |
| MP3 | T1 MP-3 (on nc1_train, T1 review D1) | rho(log10 nc1_tr, lead_md) >= +0.6 | permutation |
| MP5 | T1 MP-5 | rho(log10 nc1_tr, AUROC(gap) - AUROC(MSP), clean errors) <= -0.5 | permutation |
| MP6 | T1 MP-6 | rho(log10 nc1_tr, SVHN L2 10-NN AUROC) <= -0.5 | permutation |
| MP7 | T1 MP-7 | rho(log10 nc1_te / nc1_tr, clean ECE) >= +0.5 | permutation |
| P2b | T2 P2(b), Hui et al. | rho(log10 nc1_tetr / nc1_tr, DO-3 FPR) >= +0.6 | permutation |
| M-NC4 | T3 CZ-3 / CZ-7b, T1 review D5 (mechanism) | rho(1 - held-out NC4 agreement, O1) >= +0.5 | permutation |
| FE1, FE3 | T1 review D4 | family-demeaned twins of MP1 and MP3: registered sign x rho_fam >= 0.6 (families >= 2, n = 11) | within-family permutation |
| FE9 | T1 review D4 | rho_fam(pred acc_coll, obs) >= 0.6 in >= 3 of the evaluable outcomes (>= 4 evaluable) | within-family permutation, count |
| P3a | T2 P3a | single-coordinate label-free law (see below) for lead_md; direction: falls with collapse | permutation of rho(pred, obs) |
| P4 | T2 P4 | law for SVHN L2 10-NN AUROC - best head AUROC; rises with collapse | same |
| P5a | T2 P5a | law for the harm slope; falls with collapse | same |
| P5b | T2 P5b | law for H10 (O2); rises with collapse | same |
| P6b | T2 review #22 | law for best non-duplicate tap - best head (candidates add cdepth_pl); rises with collapse | same |
| S7 | T2 review #24, T3 CZ-5 | law for O4 (overconfidence under shift); rises with collapse | same |
| P3b | T2 P3b (review #7) | at C units as collapsed as a full-fit resnet56 (plnc1_te <= theta = max plnc1 of the D16 full-recipe r56 nets, frozen at P2), every d <= +0.005; REFUTED if any has its 99% CI lower bound > 0; PARTIAL otherwise; NOT_EVALUABLE with < 3 such units. The same rule on D21 is recorded at `--fit` ("P3b on D") | none (a bound rule) |
| P5c | T2 P5c (review #16) | the committed HOLD band has no violation over the 30 discovery splits in >= n - 2 C units; holdout + extras INFO | none (a bound rule) |
| IC-P2 | T2 P2(a) relabelled (review #8), D14 | the held-out-calibrated conformal FPR inside the exact 99% beta-binomial band (`betaBinomBand(1500, 1500, 0.05)` = [0.0313, 0.0720]) in >= n - 1 C units | none (an instrument check) |
| P7 | T2 P7 (review #3) | registered in `experiments/b4/prereg_p7.json` at P1: Kendall tau(order, nc1_tr) <= -0.66 in >= 4 families with >= 3 members (resnet: r20 hub, r32, r44, r56 hub; vgg 11/13/16/19; mbv2 x0.5-x1.4; shufflenet x0.5-x2.0; repvgg a0-a2). INFO points: resnet32 nc1_tr 0.112 [0.082, 0.153], resnet44 0.074 [0.054, 0.101] | within-family permutation, count |
| P8 | T1 MP-8 in the T1 review D5 form | abs(log10 plnc1 / nc1_te) <= 0.1 in >= 10 of 12 AND rho(log10 g_cv, log10 nc1_te) <= -0.8 | max of the two permutation p (intersection-union) |
| MP-K | D12, T1 review D6 | knob pairs (ls10 and wd5e5 against F, same seed) x 5 outcomes: abs(dobs - dpred(acc_coll)) < abs(dobs - dpred(acc)) in >= 75% of the cells (15 of 20); a pair with abs(dacc) > 0.5 pt is INFO; NOT_EVALUABLE below 10 cells | binomial tail at 1/2 per cell |

**Single-coordinate law items** (P3a, P4, P5a, P5b, P6b, S7): y = a + b z, z the label-free coordinate with the lowest
LORO MAE on D21, frozen at P2 with its **direction flag** (sign(b) x the collapse sign of z equals the registered
direction; collapse signs: plnc1 -1, g_cv +1, topk_frac +1, cdepth_pl +1). SUPPORTED iff Spearman(pred, obs) >= 0.6 AND
the family-demeaned Spearman > 0 AND MAE <= 0.8 x MAE(constant) AND the frozen direction is as registered. Reported
beside it (INFO): MAE ratios to the accuracy-only and head-only single-coordinate laws, EXTRAP units, and the split-half
noise (`noise_limited` when the between-unit SD is below twice the median split-half SE). A direction contradicted on D21
is on record at P2, before any C read; D8 (b) then allows withdrawing that item to INFO with a reason.

**Rule 8 (applied to the correlation and law items; annotated on the PRIMARY).** FAMILY-DRIVEN: the family-demeaned rho
has the other sign or abs < 0.2 -> a SUPPORTED rule becomes "PARTIAL (FAMILY-DRIVEN)". DIMENSION: the partial Spearman
on log10 penult width falls below the item's bar -> "PARTIAL (DIMENSION)". RULE8-CHECK: abs(rho) >= 0.95 is flagged
with the artifact hypotheses of section 8 and the partial on accuracy.

**INFO (never a verdict).** T1 MP-4 (rho(log10 nc1_tr, O1); noise-limited, T1 review D5); CZ-7c; the frozen laws'
predictions for the fresh seeds F and F20; the null units' coordinates; the per-unit table (coordinates, outcomes,
predictions of every law).

**Power (known before data).** At n = 12 the one-sided permutation p of Spearman is 0.021 at rho 0.6, 0.0064 at 0.7,
0.0013 at 0.8 (node Monte Carlo, 200,000 permutations). Holm's first step over 22 items needs p <= 0.0023, so only
secondaries with rho of about 0.8 or more survive unless others are rejected first. A family-demeaned rho of 1.0 on the
C families (3, 3, 3, 2 members) has p = 0.0022: the FE twins can pass Holm only in a step-down after other rejections.

---

## 6. MP-K, the knob lane (the only within-architecture intervention)

Label smoothing 0.1 (`scripts/b4_train_knob.py`) and weight decay 5e-5 (the frozen `train_second_seed.py`) move collapse
at a fixed architecture, seed and (to within 0.5 pt) accuracy. MP-K asks whether the frozen acc + collapse law, fitted
across architectures, also predicts the **change** of each outcome under such a knob better than the accuracy-only law.
Pairs: (resnet56_s31, resnet56_s31_ls10), (resnet56_s32, resnet56_s32_ls10), (resnet56_s31, resnet56_s31_wd5e5),
(resnet56_s32, resnet56_s32_wd5e5); baselines F are matched by architecture and seed from the registry. If Kwd is cut
(plan B), MP-K runs on 10 cells.

---

## 7. Registered predictions (rule 3; before any data)

| item | predicted | reason |
|---|---|---|
| **PRIMARY** | **MIXED**: O2 and O3 WIN, O1 LOSS, O4 and O5 NEITHER | O1 has almost no between-model spread (T1 predicts geometry is BOUNDED on clean errors), so no law beats the constant; O2 and O3 had the strongest collapse relations in discovery (lambda9/lambda10 r +0.956 and +0.979 on the train-reference preview) and depend on geometry that accuracy does not fix; O4 is driven by both collapse and accuracy under shift; O5 depends on T1's multi-tap flag, whose link to collapse is untested |
| S-PL | MIXED | the labelled coordinate adds nothing over the label-free one |
| M-DO3 | REFUTED (< 9/12) | the law's intercept should move with penult width and VGG's post-FC penult (low confidence; the T2 design predicted transfer) |
| M-LF | SUPPORTED | the held-out label-free coordinates are width-robust by construction |
| MP1 / MP3 / MP5 | SUPPORTED | discovery rho -0.942 / +0.945 / -0.841 (T0, other coordinates, T1 review D1) |
| MP6 | REFUTED | SVHN AUROCs near the ceiling leave no rank signal |
| MP7 | REFUTED | T1 registered it with low confidence |
| P2b | SUPPORTED | Hui et al.'s mechanism |
| M-NC4 | REFUTED | O1 is noise-limited |
| FE1 / FE3 (rule) | SUPPORTED; Holm likely NS | within-family width and depth series move collapse; power is limited (section 5) |
| FE9 | REFUTED | three outcomes with rho_fam >= 0.6 is demanding at 11 units |
| P3a, P5a, P5b | SUPPORTED | discovery r +0.893 (lead), 0.814 (slope), 0.956 (H10) |
| P4 | REFUTED | the SVHN ceiling |
| P6b | REFUTED | no discovery evidence for a collapse ordering of the tap-over-head gain |
| S7 | SUPPORTED | collapsed nets are more overconfident under shift (T3 CZ-5) |
| P3b | SUPPORTED | at a collapsed penult the margin equals the non-saturating head statistic |
| P5c | SUPPORTED | the band is conservative on collapsed nets (H10 0.40-0.45 at full-fit resnet56) |
| IC-P2 | SUPPORTED | exchangeability; a failure means a bug |
| P7 | SUPPORTED; resnet32 and resnet44 inside their intervals | deeper or wider nets collapse more at a fixed recipe |
| P8 | SUPPORTED | at 90-95% accuracy the argmax labels are nearly the labels |
| MP-K | REFUTED | label smoothing collapses the penult but also lowers confidence; the cross-architecture law ties collapse to overconfidence (O4) the other way |

---

## 8. Artifact hypotheses (rule 8, stated before data)

1. **Family offsets.** Five families, resnet44 a singleton, VGG's penult after two FC layers: a pooled rho can come from
   family means alone. Guarded by the family-demeaned clause of every WIN and every law item, and FE1 / FE3 / FE9.
2. **Penult width (64 to 2,048).** kNN radii and the raw spectral gap depend on width. The candidates are the
   cross-validated gap, the trimmed pseudo-label nc1 and the variance share; every item reports the partial on log10
   width (DIMENSION).
3. **Accuracy in disguise.** Accuracy is a rival of every PRIMARY outcome and of every law item; the partial on accuracy
   is reported. Test-selected hub weights bias accuracy slightly, never geometry.
4. **The head's saturation in disguise.** sat999, the logit gap and 1 - mean MSP are the head rivals; if the head law
   is as good, the answer is "collapse matters, but the head already tells you" (a finding).
5. **Numerics at wide penults.** Trimmed and verbatim nc1 are both recorded (section 3.1).
6. **float16 storage** of corrupt-split taps (harm, per-tap shift); the clean coordinates are float32 penults.
7. **Shared rows.** Every unit is measured on the same rows, so row noise is shared; the split-half SE of every target
   is reported and a between-unit spread below twice that SE is flagged `noise_limited`.
8. **Too good.** abs(rho) >= 0.95 is flagged RULE8-CHECK with 1-4 as the hypotheses to check first.
9. **Extrapolation.** C coordinates outside the D21 range: counted per outcome; an EXTRAP-dominated WIN reads
   "extrapolation holds".

---

## 9. Functional reading of a verdict

| outcome | unit of the MAE | what the controller does with it |
|---|---|---|
| O1 | AUROC points of the head-conditional increment | whether to compute penult geometry beside the head on this backbone |
| O2 | H units (x the backbone's slope = pt of undetected loss at a mis-set edge) | where to put the HOLD band's edge without labelled shift data |
| O3 | AUROC points (novelty, CIFAR-100) | whether to trust the density detector or the head for escalation |
| O4 | probability points of overconfidence | how much to discount MSP under shift |
| O5 | AUROC points (per-sample corruption, s3) | whether the calibrated multi-tap flag is worth running |

M-DO3 is also reported as false alarms per 1000 clean frames. A SUPPORTED PRIMARY means: for CIFAR-10-scale,
single-recipe backbones, the label-free coordinate configures these functions with the stated errors. MIXED means it
does so for the WIN outcomes only. Nothing from T2 alone is promoted to ATLAS_STATUS (one recipe, one dataset);
inventory row CD-12 (D19) is updated with the verdicts.

---

## 10. Flow, commands and what was verified locally

| step | command | notes |
|---|---|---|
| S0 (CPU pod) | `python -m pytest -q tests/test_collapse_probe.py`; `python scripts/collapse_probe.py --selftest --selftest-out results/instrument_check_b4s0/selftest_collapse_probe.json` | first execution of the Python |
| P1 (Windows) | `node scripts/collapse_laws.js --selftest`; `node tests/collapse_laws_fixture.js`; `node scripts/collapse_laws.js --dry-run` | must all exit 0 |
| S1 | `scripts/b4_reg.py jobs --phase discovery`: `python scripts/collapse_probe.py --registry experiments/b4/models.json --unit <id> --phase discovery --out results/b4_t2/<id>` for the anchors, D, N, Dnew | niced CPU lane; limits t2 1800 s (64-d) / 3600 s (wide) |
| P2 (Windows, after R1 with T1's discovery scoreboards) | `node scripts/collapse_laws.js --fit --p1 <P1> --out experiments/b4/laws_frozen.json` | refuses if a discovery record did not run collapse_probe.py, b4_core.py, b4_collapse.py or (O1 / O5) t1_scoreboard.py as of P1, if phases are wrong, or if < 12 D21 nets are usable; records the D15 nc1 gate, the code sets and every fitted file's hash |
| S2 | `--phase confirmation` on C, K, F, F20 (first in the lane, D9) | the sealed dumps open once |
| E | `node scripts/collapse_laws.js --evaluate --p1 <P1> --p2 <P2> --p-run <S1 HEAD>,<S2 HEAD> --replay results/instrument_check_b4s2/replay.json --json results/b4/t2_eval.json` | guards: --p1 / --p2 / --p-run given (else no verdict, UNOFFICIAL); laws and evaluator as at P2, P7 read at P1; one instrument (collapse_probe.py, b4_core.py, b4_collapse.py; t1_scoreboard.py behind O1 / O5) for the fitted records and C, equal to P2; fitted files unchanged since the fit; probes run at --p-run (prefixes); no C unit in the fit; confirmation-phase records, each unit touched once. Outputs are found under every tag (a D10 relaunch split is merged) |
| D7 path | add `--refit-tag _p2` | the frozen procedure refitted on the re-probed discovery units; the P2 laws become INFO |

**Verified locally (node; Windows).**
- `node scripts/collapse_laws.js --selftest`: 20/20. It covers least squares, Kendall, the partial Spearman, the binomial
  tail, the permutation p, the replay-drift bounds (the Spearman bound against a brute-force perturbation; the count and
  conjunction flip rules), the four `b4_primary.js` fixtures (good law SUPPORTED, noise REFUTED, family-offset-only law
  not SUPPORTED, 9 units NOT_EVALUABLE), and an end-to-end synthetic fit and evaluation.
- `node tests/collapse_laws_fixture.js`: 71/71. Every one of the 25 secondaries flips (SUPPORTED with lawful synthetic
  confirmation units; not SUPPORTED, and still evaluable, with the confirmation flipped). The PRIMARY gives SUPPORTED,
  REFUTED and MIXED on the matching scenarios. The NOT_EVALUABLE paths are covered (too few C units; T1 outcomes
  missing; too few discovery nets). So are the output selection across tags (a C set split between '' and _r2 gives
  the same PRIMARY; a unit probed twice is not evaluable), the guards (other probe code, a probe outside the confirmation
  phase, a leak, append-only, the CLI without --p1 / --p2 / --p-run, a b4_core.py-only or T1-scoreboard change P1 -> P2
  without --refit-tag, a fitted file changed after the fit), the D7 re-probe, and the D15 replay drift per decision (a
  FAIL with zero drift only tags; a t2 drift in O2 / H10 nulls only the decisions that read them; a t1-only FAIL moves no
  t2 decision; a missing replay output nulls every decision; a 1e-9 wobble at a near-zero leaf moves nothing). A non-dry
  run in a temporary git repository covers --fit --p1 (every instrument hash), P2, the official --evaluate, --p-run
  prefixes and an evaluator edit after P2. The self-test and the dry run close the
  fixture.
- `node scripts/collapse_laws.js --dry-run` on the committed atlas.json files reproduces three known answers: the AH-1
  law (a 0.0224, b -0.1080), X9's Spearman (+0.937, and -0.942 for nc1) and T1 review D2's partial Spearmans (-0.891 for
  all 16 nets, -0.247 within resnet56). It also writes the P7 registration; the two resnet points equal the T2 design's.
- The Python was checked by the structural and undefined-name checkers; a node Monte Carlo of the self-test's synthetic
  design confirmed its statistical thresholds (FPR 0.043 for +-0.03 around 0.05; near-OOD AUROC 0.72 for > 0.6; mid-tap
  shift AUROC 0.67 / 0.91 at s1 / s3).

**Not verified.** No Python has run. The first run of `collapse_probe.py` is S0. Real-data timing (estimated at 1-3 min
per 64-d unit and 5-10 min per wide unit on one BLAS thread) is recorded per tap and target in S1, and
`b4_timeouts.js` sets the S2 limits from it.

---

## 11. Files

| file | owner | role |
|---|---|---|
| `scripts/collapse_probe.py` | T2 | the probe (section 3) |
| `scripts/collapse_laws.js` | T2 | fit, evaluation, PRIMARY rule, secondaries, dry run, self-test |
| `experiments/b4/prereg_p7.json` | T2 | the blind P7 registration (emitted by `--dry-run --emit-p7` from committed data) |
| `tests/test_collapse_probe.py` | T2 | pytest known answers and the schema pin |
| `tests/collapse_laws_fixture.js` | T2 | the node fixture (flips, NOT_EVALUABLE, guards, git non-dry run) |
| `experiments/b4/laws_frozen.json` | emitted at P2 by frozen code | the frozen laws |
| `results/b4_t2/<id><tag>/probe.json` | emitted in S1 / S2 | one per unit |
| `results/b4/t2_eval.json` | emitted at E | the verdicts |
