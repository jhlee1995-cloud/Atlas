# Atlas knowledge base

Last reviewed: 2026-09-23; part of the Atlas knowledge base (docs/knowledge/README.md). This file is its index.

Written against HEAD `343f651`. Companion documents:
- `docs/reviews/EVAL_2026-09-23.md`: the programme evaluation that uses this inventory.
- `docs/reviews/EXTRACTION_PROPOSALS_2026-09-23.md`: ranked proposals X1-X16 for reading more controller-usable
  information out of the activation space.
- [controller-functions-feasibility.md](controller-functions-feasibility.md): feasibility of the controller functions
  F1 (pre-emptive flagging) and F2 (information-gated feeding), with verified citations.

## 1. Purpose

The owner's primary question for Atlas is: **which kinds of information that a controller could use can be read out of
a backbone's internal geometry, how reliably, and does the geometry add anything beyond what the output head (softmax /
logits) already gives?** The label-free adapt / hold / escalate controller, and then a robot, are the final goal. The
map-first work is judged first on the primary question.

Two principles set by the owner govern this knowledge base:
1. **Prior art is a starting point, never a reason to drop or down-rank an item.** When the literature already has a
   method for a kind of information, Atlas adopts it (as a component or as a baseline) and refines it toward its own
   use. The "already widely known/used?" column in section 5 says where to start and what to cite; it is not a penalty.
2. **What matters is end functionality, not novelty.** Every item is judged by whether the controller can act on it
   (adapt / hold / escalate; F1 suspicious-input flagging; F2 information-gated feeding), how reliably (AUROC, false
   alarms, detection delay, cost), and whether it beats or complements the cheapest existing baseline (logit gap,
   max-softmax, a simple density). No novelty grade is given anywhere in this knowledge base.

This knowledge base is a durable asset for future pre-registrations. It holds:
- **Prior art per kind of information** (six topic files). For each family of signals Atlas probes: what the literature
  measures, which works matter (every citation with its URL and a verification mark), how the methods are used in
  research and in practice, their known failure modes, the baselines to adopt, and what they imply for the controller.
- **The information inventory** (section 5): one row per kind of information Atlas has probed or proposed (45 rows),
  with its geometric source, evidence and scope, status, whether it adds beyond the output head, the prior art that is
  its starting point, how prior work uses it, how Atlas refines it for controller use, and its functional readiness.
- **Functional readiness by controller function** (section 6).
- **The adoption and refinement plan** (section 8): which published methods to adopt as components or as baselines,
  and how to refine them toward controller use, including the incremental-over-head test.

**How it was built.** Six researchers each covered one topic, checked every citation against its source (each topic
file states its verification marks) and classified the inventory items their literature bears on. This README merges
their classifications and resolves the conflicts (section 7). The web-search budget ran out in every topic before some
targeted searches could be made; those gaps are listed in the topic files. "Not found in prior art" therefore means "not
found in a limited search", never "shown to be absent from the literature".

## 2. How to use it in a pre-registration

1. **Find the row.** Locate the inventory row (or rows) the plan touches. If the plan probes a new kind of information,
   add a row first, with status UNTESTED.
2. **Name the starting point.** Cite the row's prior-art classification and key citation in the plan's rationale, and
   say which published method is adopted as a component, which as a baseline, and what Atlas refines. The topic file's
   section 5 gives the relation (rediscovered / extends / contradicts / untested).
3. **Fix the head comparator before confirmation.** Take the head baselines for that information family from section 8
   (and the topic file's section 6). On discovery data, choose the best head-only statistic and pre-register it as the
   bar. "Beats maxprob" is not "beats the head" (H2b, ATLAS_STATUS.md:20).
4. **Pre-register an incremental-value test.** A joint (nested) model, head-only against head plus geometry,
   cross-fitted, with an effect-size bar (the HEAD-ADDITIVE rule of X1 in `docs/reviews/EXTRACTION_PROPOSALS_2026-09-23.md`).
   A conditional AUROC win alone is not enough.
5. **State the functional target.** Name the controller action the result would enable (section 6) and the reliability
   numbers it needs: false-alarm rate at a stated budget, detection delay, AURC or risk at coverage, latency and memory.
6. **Design out the known failure modes** listed in the topic file's section 3 (for example: calibrate thresholds on
   held-out clean data; L2-normalise before kNN; score on all errors or risk-coverage, not only on confidence-selected
   positives; group probe folds by source image; keep random-init nulls).
7. **Use fresh seeds for confirmation.** resnet56 s1/s2 are spent for the ANOMALY_H1 axes (results/anomaly_h1/SESSION.md:37,
   :305); any re-analysis on them is discovery-grade.
8. **After the verdict**, update the row's status, evidence, "beyond the head?" and readiness cells, and bump "Last
   reviewed".

## 3. Index

| file | inventory items (primary; also covered) | the question it answers | one-line headline |
|---|---|---|---|
| [confidence-margin.md](confidence-margin.md) | CM-1 to CM-11; also DO-1, DO-4, CD-11, LH-1, LH-2 | Does a class-centre geometric score flag the model's own errors better than the head's softmax or logits? | The Atlas margin is the trust score's centroid form and reproduces its CIFAR result; the one open positive is a matched-confidence lead over the logit gap on two DeiT-lineage ViTs (CM-5), not yet separated from normalisation. |
| [density-ood.md](density-ood.md) | DO-1 to DO-5; also CD-4, CD-9, LH-1 to LH-3, CM-1, CM-8, CM-11, ST-3, ST-4, ST-7 | What does feature-space density/familiarity tell a controller, and does it beat head-based OOD scores? | `knn_density` is deep kNN without L2 normalisation and with a train-referenced threshold; its false-alarm law (DO-3) was not found in prior art; no head baseline was ever run. |
| [collapse-dimension-similarity.md](collapse-dimension-similarity.md) | CD-1 to CD-11; also CM-2, CM-5, CM-6, CM-8, CM-11, DO-3, PC-3, PC-6, ST-3, ST-6, LH-1 | What do neural collapse, intrinsic dimension and representation similarity tell us, per checkpoint? | Mostly published results (NC, hunchback ID, equi-separation, CKA); useful as calibration and change detection, not per-input sensing; no CD item has a head-side comparison other than accuracy. |
| [label-free-accuracy-harm.md](label-free-accuracy-harm.md) | LH-1 to LH-4; also DO-1 to DO-4, ST-3, ST-5, ST-6, ST-8, CM-1, CM-11 | Can geometry estimate the accuracy loss (harm) of a shift without labels, better than confidence-based estimators? | H is a robust variant of the known feature-distance estimators; ρ ≈ 0.99 across CIFAR-10-C splits is the level any shift statistic reaches; AC/DoC/ATC were never compared, although maxprob is stored. |
| [shift-type-tta-monitoring.md](shift-type-tta-monitoring.md) | ST-1 to ST-8; also LH-1 to LH-4, DO-4 | Can geometry identify shift type, gate adaptation and detect TTA failure better than output statistics? | Output statistics are the default baselines for detection, harm and collapse; feature geometry helps mainly for routing and one collapse sensor (PeTTA); the TTA ladder never ran and its planned bar is lenient. |
| [probing-corruption-geometry.md](probing-corruption-geometry.md) | PC-1 to PC-6; also ST-1 to ST-4, ST-8, LH-1, LH-2, LH-4, CD-1, CD-7 | Which input factors (luminance, frequency, corruption type) are readable at which depth, and does that exceed the head? | Much of it is conv + GAP by construction and present in random nets; shift type at pre-collapse taps is the most plausible head-invisible information, but no head or pixel comparator was run. |
| [controller-functions-feasibility.md](controller-functions-feasibility.md) | (controller functions F1, F2) | Are pre-emptive flagging (F1) and information-gated feeding (F2) feasible? | Feasible in weak forms; the strong forms ("catch the error before the model runs", "a rarely-wrong per-sample alarm") are not supported by current evidence. Its sub-variants (F1-a … F2-d) are used in section 6. |

Related, outside this folder:
- `docs/reviews/EVAL_2026-09-23.md`: the programme evaluation, reframed around the primary question.
- `docs/reviews/EXTRACTION_PROPOSALS_2026-09-23.md`: 16 proposals (X1-X16), ranked by functional value per dollar. X1 (a
  head-conditional scoreboard) is the operational form of the incremental-over-head test in section 8; X4 (a calibrated
  multi-tap conformal flag) and X8 (Trust Score, relative Mahalanobis, LID, kNN purity) adopt and refine standard
  detectors.

## 4. Vocabulary

**Status** (the evidence Atlas itself has):

| status | meaning |
|---|---|
| ESTABLISHED | GREEN (✅) in ATLAS_STATUS.md, or confirmed by a pre-registered rule on at least two independent units. Scope is always the stated one. |
| SUPPORTED-NOT-PROMOTED | An ANOMALY_H1 SUPPORTED label: depth-56 CIFAR only, and nothing was promoted (results/anomaly_h1/SESSION.md:48). |
| CANDIDATE | YELLOW (🟡), discovery, or a pre-registered positive that was not predicted. |
| REFUTED | Pre-registered and failed. |
| RULED-OUT | An ANOMALY_H1 AX gate was DROPPED. |
| UNTESTED | Proposed but never run, or a gate closed before confirmation. |
| MEASURED-NO-CLAIM | Numbers exist but carry no scored claim. |

**Beyond the output head?**

| value | meaning |
|---|---|
| NO | A head statistic was measured and gives the same information. |
| YES / PARTIAL | Measured, and the geometry adds something (PARTIAL: only against a weak head statistic such as maxprob). |
| CANDIDATE | A pre-registered positive at narrow scope. |
| UNTESTED | The head comparison was never run. |
| N/A | A checkpoint-level descriptor or a calibration law with no per-input head analogue. |

Two senses of "beyond the head" should be kept apart:
- **Information the head discards.** Possible only where the head map loses information. The CIFAR ResNet head maps a
  64-d penult to 10 logits, so it discards at least 54 dimensions; pre-collapse taps carry more again. The ImageNet
  ViT-B/16 head maps 768 dimensions to 1000 logits and is injective on its input if its weight matrix has full column
  rank, so at that tap geometry can at most be a better *statistic* of what the logits hold
  (`docs/reviews/EXTRACTION_PROPOSALS_2026-09-23.md`, X2).
- **A better statistic than the standard head scores.** A score computed from geometry that ranks or decides better than
  MSP, max logit, logit gap, energy, entropy or their normalised variants. All Atlas beyond-head evidence so far (CM-2,
  CM-4, CM-5) is of this second kind. For a controller both kinds count: a better statistic is directly usable.

**Already widely known / used?** This is the starting point for adoption, not a grade.

| value | meaning |
|---|---|
| standard-practice | Used routinely in deployed systems, or the default baseline everyone reports. Adopt it as the baseline or component as published. |
| known-in-research | Published and benchmarked in research; Atlas's finding restates it. Adopt the published method and its protocol, then refine. |
| partly-known | The idea or mechanism is published, but Atlas's specific form, condition or number was not found. Adopt the published part; Atlas's part needs its own test. |
| not-found-in-prior-art | No precedent found in a search that was limited by budget. Cite the nearest starting points; the claim is not literature-complete. |

**Functional readiness for the controller:**

| value | meaning |
|---|---|
| usable now | Can drive a named controller action today, within the stated scope; reliability numbers are given. |
| candidate | Evidence exists, but named tests (a head baseline, calibration, a stream test) are missing before it may drive an action. |
| calibration input | Configures thresholds, the score choice or the tap choice offline; not a runtime signal. |
| not built | Never run, or never scored as a detector. |
| do not use | The evidence says it should not drive an action. |
| no controller role | A descriptor or an instrument check. |

**Controller functions** (from controller-functions-feasibility.md §0 and EVAL §3): F1-a per-sample "likely wrong"
flag; F1-b per-sample sensor-fault or degradation flag; F1-c pre-model flag; F1-d per-stream harmful-shift flag
(Gate 1); F2 information-gated feeding (F2-a frames, F2-b tokens, F2-c early exit, F2-d adaptation compute);
HOLD-T "hold, output trusted" (harmless shift); HOLD-D "hold, output distrusted" (untrustworthy input); ADAPT;
ROLLBACK / reset of a diverging adaptation; ESCALATE.

## 5. The information inventory

45 rows: every ATLAS_STATUS row (1-11, including 6a/6b and the depth-56 tags), every scored item in the eight SESSION
files (Stage 0, 1, 1b, A3, Stage 2, A4b, ANOMALY_H1 AH/AX, B1/B1b), and the MASTER_SUMMARY and CLAUDE.md items that
Atlas proposed but never tested. Families: CM = per-sample confidence and margin; DO = density and OOD; LH = label-free
harm; ST = shift type and TTA; CD = collapse, dimension and similarity; PC = probing and corruption geometry.

Scope shorthand used in the table: "CIFAR" = CIFAR-10 (test[:5000]) and CIFAR-10-C, ResNet20/56 trained with one
recipe; "d56" = the depth-56 seeds s1/s2 (spent for ANOMALY_H1) plus the hub; "ViTs" = torchvision vit_b_16
IMAGENET1K_V1 and DeiT-B fb_in1k (both DeiT lineage) on a degraded ImageNet mirror (2-3 pt below published). The last
cell of the "already known" column names, in italics, the topic file with the details. "X*n*" refers to
`docs/reviews/EXTRACTION_PROPOSALS_2026-09-23.md`.

| id | information | geometric source (layer, quantity) | evidence & scope | status | beyond the output head? | already widely known/used? (starting point) | how prior work uses it | how Atlas refines it for our use | functional readiness for the controller |
|---|---|---|---|---|---|---|---|---|---|
| CM-1 | Confident-mistake flag per sample, CIFAR resnet20 (row 9) | penult (64-d GAP); top-2 train-reference class-centre margin d2 − d1 (atlas/invariants/margin.py:79, :118-122) | ATLAS_STATUS.md:19; results/margin_v1_resnet20_s1/SESSION.md:167-173, :229-245. s1-s4 + hub, CIFAR, clean. All-errors AUC margin 0.912-0.925 vs maxprob 0.912-0.926; margin − distance +0.046 to +0.063 at cut 0.7 (p ≤ 1.6e-19) | ESTABLISHED (✅ s1-s4) | **NO**: = maxprob (E1 confidence-matched gap ≤ 0.002; E9 hub −0.0013, p 0.45). Logit gap never measured on CIFAR (recomputable) | **standard-practice**: MSP (Hendrycks & Gimpel 2017); the Trust Score reported the same CIFAR result (Jiang et al. 2018). *confidence-margin* | MSP thresholding is the standard reject option; the trust score is the research alternative | Adopt selective classification with a post-hoc choice of head statistic (Geifman & El-Yaniv 2017; Cattelan & Silva 2024). Keep margin only as an input to a cross-fitted joint detector (head vs head + margin) on all errors, scored by AUROC, AURC and risk at 80/90/95% coverage, with the CIFAR logit gap recomputed from penult·W + b (X1) | **usable now** for F1-a as a graded risk score, through the head: all-errors AUC 0.912-0.926 (CIFAR r20, clean). The margin adds nothing. Missing: AURC and risk-at-coverage numbers; shift. At AUC ≈ 0.92 a standalone alarm has many false alarms at useful recall (feasibility F1-a) |
| CM-2 | Same flag in fully fit resnet56 (collapse regime) | resnet56 penult margin; nearest-centre distance d1 | ATLAS_STATUS.md:19 (d56 🟡); results/atlas_v1_resnet56_s1/SESSION.md:518-566; E9 at results/margin_b1_vitb16/SESSION.md:432. M2 fails in s2 (+0.0005, p 0.90) and hub (−0.0008). d56 only | CANDIDATE (d56 🟡) | **PARTIAL** (vs maxprob only): +0.006 / +0.012 / +0.020 (p 0.027 / 0.0017 / 0.0004); not a tie artifact (top_share_correct 0.0002; A4b SESSION:561-564). Distance ≈ margin. Logit gap not measured | **partly-known**: NC2/NC4 predict distance ≈ margin (Papyan et al. 2020); MSP is often not the best head statistic (Cattelan & Silva 2024). *confidence-margin* | Head statistic chosen post hoc per checkpoint; the convergence is theory, not a tool | Recompute the CIFAR logit gap and repeat E9 against it (the literature predicts the lead vanishes); if it survives, enter margin in the joint detector. Use collapse level, or X9's label-free spectral gap, as a prior for which scores to include. Discovery on spent seeds; confirm on fresh seeds (X1) | **candidate** only: +0.006 to +0.020 over maxprob, never tested against the logit gap; d56 CIFAR; drives nothing yet |
| CM-3 | Confident-mistake flag on ImageNet ViT-B/16 (row 10, V1/V2) | final-norm CLS token (head input); margin to reference-half class centres | ATLAS_STATUS.md:20; results/margin_b1_vitb16/SESSION.md:304-329, :511-517. ViTs + swaps; c* 0.5 (fallback forced by ResNet50); amended G0 anchor. V1 margin − distance +0.114 to +0.119. At c* 0.5, type-b is 50% of ViT errors; 42-46% of it is ReaL-correct | ESTABLISHED at narrow scope (✅ B1b) | **mostly NO** for V1: at c* the logit gap scores 0.769 / 0.773 vs margin 0.792 / 0.807 vs distance 0.674 / 0.688; on all errors margin ≈ logit gap | **known-in-research**: top-2 scores are the published form of error flagging, in feature space (trust score, Jiang et al. 2018; DkNN, Papernot & McDaniel 2018) and at the head (LogitsMargin on 84 ImageNet classifiers, Cattelan & Silva 2024); ImageNet label noise (Northcutt et al. 2021). Neither feature-space paper compares with a single distance, so V1's "margin beats distance" is Atlas's own measurement. *confidence-margin* | Top-2 scores (logit margin, trust score) for selective prediction and error flagging | Adopt the head's logit gap as the default ViT trust score. Evaluate on ReaL labels as a co-primary and on all errors with AURC; margin enters only through CM-5's joint test | **usable now** for F1-a through the logit gap (≈ margin on all errors; clean, DeiT lineage, degraded mirror). ESCALATE on confident errors would escalate many acceptable answers: 42-46% of ViT type-b are ReaL-correct |
| CM-4 | Margin vs softmax maxprob at matched confidence | penult / CLS margin vs maxprob | V3 ADDS in 4/4 ViT runs, +0.034 / +0.054, p ≤ 4e-15 (results/margin_b1_vitb16/SESSION.md:331-335); M56-c r56 +0.006 / +0.012 (results/atlas_v1_resnet56_s1/SESSION.md:551-566); r20 E1 ≤ 0.002 | ESTABLISHED on ViTs; CANDIDATE r56; REFUTED r20 (as an addition) | **PARTIAL**: YES vs maxprob on ViTs and full-fit r56, NO on r20. Not beyond the head: maxprob is not the best head statistic (H2b rule, ATLAS_STATUS.md:20) | **partly-known**: LogitsMargin and p-norm max logit beat MSP; soft-label recipes break MSP (Cattelan & Silva 2024; Xia et al. 2025). *confidence-margin* | Post-hoc replacement of MSP by a better head statistic | Adopt Cattelan & Silva's head-statistic set (MSP, max logit, logit gap, softmax margin, entropy, Gini, p-norm max logit) and freeze the best on discovery data as the bar; H2b already encodes this | **calibration input**: do not use maxprob as the trust score where it is saturated (CIFAR: 68-79% of samples ≥ 0.999) or broken by a soft-label recipe (ResNet50) |
| CM-5 | Residual confident-mistake information beyond the head's logit gap | CLS (ViT) margin vs head top-2 logit gap, confidence-matched | V3b (pre-registered, predicted EQUIVALENT): margin ahead in all 4 ViT runs at every cut 0.5-0.8, +0.0076 to +0.0368, p ≤ 0.047 (vitb16 SPLIT-FRAGILE at 0.5). All errors ≈ (vitb16 +0.0017 / −0.0009; deitb +0.0038 / +0.0046). ResNet50 BELOW: −0.046 / −0.047 matched, −0.049 / −0.053 all errors (results/margin_b1_vitb16/SESSION.md:337-345, :497-518). Clean data | CANDIDATE (row 10: V3b MIXED/UNRESOLVED at c*) | **CANDIDATE YES**, conditional on confidence, 2 DeiT-lineage ViTs only; normalisation not ruled out; joint detector untested; untested on CIFAR; head_center_cos does not order the effect | **not-found-in-prior-art** (limited search). Starting points: ViM (Wang et al. 2022), fDBD (Liu & Qin 2024); competing head-only explanation: logit normalisation (Wei et al. 2022; Cattelan & Silva 2024). *confidence-margin* | Joint feature + logit scores exist for OOD detection only | Adopt ViM's joint-score design, fDBD's head-boundary distance and logit normalisation as the head-only control. Run the normalisation ablation (centre vs head numerator × normaliser) on the existing ViT and ResNet50 dumps; a cross-fitted joint detector (logit gap + margin) scored by AURC; confirm on B1c (non-DeiT augreg ViT, cut 0.7) and a non-label-smoothed CNN (X1; confidence-margin.md §6.2) | **candidate** second F1-a axis: +0.008 to +0.037 AUROC inside confident strata, 2 DeiT-lineage ViTs; BELOW on ResNet50; no whole-population gain shown. Usable only after a joint-detector AURC gain and replication |
| CM-6 | Which per-sample score a backbone should use (collapse picks margin vs distance) | sep_ratio / nc1 vs the margin − distance lead | AH-3 REFUTED on (d) (results/anomaly_h1/SESSION.md:175-191); its graded clause (e) passed, ρ −0.857 (n 7); AH-4 NOT_EVALUABLE, INFO runs backwards across ImageNet (:451-473) | REFUTED (AH-3); UNTESTED (AH-4) | N/A | **partly-known**: NC4 (Papyan et al. 2020); NC degree changes what detectors see (Harun et al. 2025). *confidence-margin, collapse-dimension-similarity* | Scores are chosen empirically per checkpoint | Replace the collapse rule by empirical per-checkpoint score selection on held-out labelled data (standard practice); keep nc1, or X9's label-free spectral gap, as a prior for when penult geometry duplicates the head | **calibration input** at most: the registered rule is refuted; the graded clause holds only within resnet56 |
| CM-7 | "Confident mistakes sit on ridges" | median margin ratio (M3, V2) | results/margin_v1_resnet20_s1/SESSION.md:251-262; results/margin_b1_vitb16/SESSION.md:317-329 (V2 fails at cut 0.8 in all 4 ViT runs; ratio > 0.5 at cut 0.99 on CIFAR) | REFUTED as geometry (restates the cut) | **NO** | **known-in-research**: confident errors arise far from the data (Hein et al. 2019) or are label errors (Northcutt et al. 2021). *confidence-margin* | Not used as a detector | None as a signal. Adopt label-noise-aware evaluation (ReaL, multi-label ground truth) for any confident-error claim; drop the "ridges" wording (MASTER_SUMMARY.md:218; commit 343f651 message) | **do not use** |
| CM-8 | Activation energy as an error signal | penult mean squared activation (margin.py:13; not the logit energy score) | CIFAR errors have lower energy, oriented AUC 0.750-0.795 over 7 nets (results/margin_v1_resnet20_s1/SESSION.md:183-185); ImageNet errors higher, 0.23-0.30 in 3 models (results/margin_b1_vitb16/SESSION.md:431) | REFUTED as transferable (the sign flips) | **NO** (the feature norm acts as a hidden max logit, Park et al. 2023) | **partly-known**: feature-norm OOD separation is block- and model-dependent (Yu et al. 2023; Müller & Hein 2025); the sign flip itself was not found. *confidence-margin, density-ood* | Feature-norm OOD scores with per-model selection | Adopt the logit energy score (Liu et al. 2020) as the head baseline and rename Atlas's field (it is a feature-norm score). If the norm is used at all, enter it as one channel of a joint model, sign calibrated per backbone | **do not use** as a transferable signal |
| CM-9 | Depth / token at which confident-mistake information appears | margin AUC per tap; CLS vs mean token | Penult ≥ early maximum + 0.10 in every run (results/margin_v1_resnet20_s1/SESSION.md:175-181; results/margin_b1_vitb16/SESSION.md:424-425) | ESTABLISHED (exploratory, all runs) | **NO** early warning: information concentrates at the head input | **known-in-research**: prediction depth (Baldock et al. 2021). *confidence-margin* | Multi-layer agreement checks (DkNN, SelfChecker), not early sensors | Adopt prediction depth and multi-layer agreement as per-sample features in the last 2-3 blocks only, tested for increment over the head in logit-gap deciles (X7) | **do not use** as a pre-model error sensor (F1-c): all-errors margin AUC is 0.48-0.54 over the first half of the blocks in every model read and passes 0.75 only in the last 1-3 blocks (feasibility §2.4) |
| CM-10 | Normalised ridge position | margin_norm = (d2 − d1)/‖c1 − c2‖ (margin.py:259) | E6 false as written; ResNet50 +0.012 (p 0.36) (results/margin_b1_vitb16/SESSION.md:429) | REFUTED as written | not compared (UNTESTED) | **known-in-research**: fDBD boundary distance (Liu & Qin 2024); normalised hidden-layer margins (Elsayed et al. 2018; Jiang et al. 2019). *confidence-margin* | OOD detection; generalisation studies | Compute the literature's quantities instead, inside CM-5's ablation: the nearest-centre boundary distance (d2² − d1²)/(2‖c1 − c2‖) and the head boundary distance (l1 − l2)/‖w1 − w2‖ | **do not use** margin_norm; the boundary distances are **not built** |
| CM-11 | Nearest-centre vs head disagreement | nearest_center_agrees_with_model (atlas/invariants/landmarks.py:54; margin.py:276) | ImageNet agreement 0.89 / 0.88 / 0.80 (results/margin_b1_vitb16/SESSION.md:501-506); r56 penult 0.997, layer3.5 0.84-0.86 (results/anomaly_h1/SESSION.md:185). Measured only as a covariate | UNTESTED (never scored as a detector) | **UNTESTED**; a direct geometry-vs-head test | **known-in-research**: Trust Score (Jiang et al. 2018); DkNN (Papernot & McDaniel 2018); SelfChecker (Xiao et al. 2021). *confidence-margin* | Per-sample error / credibility alarm; NCTTA uses feature-classifier alignment to weight TTA pseudo-labels | Adopt the published disagreement detectors as starting points: score the nearest-centre ≠ argmax flag and the continuous centre-vs-logit gap (AUROC, AURC, precision at the flag rate) on the ImageNet dumps and at CIFAR pre-collapse taps, with the Trust Score as baseline and the increment in the joint model (X1, X7, X8) | **not built**; a candidate HOLD-D trigger. Capacity depends on the backbone: 11-20% of ImageNet samples disagree, 0.3% at the collapsed resnet56 penult. Reference point from another setting: SelfChecker flags 60.56% of errors at 2.04% false alarms |
| DO-1 | Severity-graded familiarity, 8 non-noise corruptions (row 6a) | penult kNN sparse_frac (k 10, un-normalised, train-q95 threshold; atlas/invariants/density.py:18-53) | ATLAS_STATUS.md:15; 8/8 s5 > s1 in 7 trained nets (smallest margin 0.0395 vs twin 0.0165), nulls 3/8-4/8 (results/atlas_v1_resnet20_s3/SESSION.md:209-218). CIFAR | ESTABLISHED (✅ s3, s4; d56 s1, s2) | **UNTESTED** (no per-split head statistic in atlas.json; maxprob is in the dumps) | **known-in-research**: deep kNN (Sun et al. 2022; Atlas omits L2 normalisation); covariate response of OOD scores (Yang et al. 2022; OpenOOD v1.5). *density-ood* | Full-spectrum OOD treats it as a nuisance; severity sweeps test calibration (Postels et al. 2022) | Adopt deep kNN as published (L2-normalised features, k sweep) with split-conformal thresholds on held-out clean rows (Bates et al. 2023); keep the norm as a separate channel; compare with per-split mean maxprob and entropy, which are free (X4, X1) | **candidate** familiarity channel (F1-b, ESCALATE). Split-level only. Related per-sample scores on the r56 hub (INFO; results/anomaly_probe_resnet56_s0hub_st3/probe.json `.AX4.auroc`): layer-2.8 density and penult d1 give 0.50-0.51 on brightness, contrast, defocus and fog at s1; layer-2.8 density gives 0.95-0.996 on gaussian and shot noise at s3/s5. No head baseline; no calibrated FPR |
| DO-2 | Noise-severity (non-)monotonicity (row 6b) | same | ATLAS_STATUS.md:16; gaussian s1/s3/s5 plateau, drop or rise by seed (resnet20 s1 0.308 / 0.381 / 0.383; s3 0.324 / 0.376 / 0.318; s4 0.338 / 0.556 / 0.633) | REFUTED both ways (seed-dependent) | **UNTESTED** | **partly-known**: feature collapse (van Amersfoort et al. 2021); feature norms shrink with shift (Kang et al. 2024). *density-ood* | L2 normalisation with the norm as a separate channel; spectral normalisation | Adopt L2 normalisation with the norm as a separate channel (Sun et al. 2022; Müller & Hein 2025); inspect the nearest-neighbour class histogram at noise s5 | **do not use** density as a severity grade for high-frequency noise |
| DO-3 | Clean false-alarm rate of a train-referenced density threshold (AH-1) | test sparse_frac vs log10 nc1 of the checkpoint | \|sparse − (0.0224 − 0.108·log10 nc1)\| ≤ 0.035 at 7/7 d56 instances (results/anomaly_h1/SESSION.md:135-155). Clean-test sparse 0.091-0.115 in resnet20 (1.8-2.3× nominal) and 0.163 / 0.184 in resnet56 (3.3-3.7×); random nulls ≈ 0.05 | SUPPORTED-NOT-PROMOTED | N/A (calibration law) | **partly-known**: calibrate on held-out ID data (OpenOOD v1.5; DkNN, Papernot & McDaniel 2018; Bates et al. 2023); train/test collapse gap (Hui et al. 2022). The law itself was not found in prior art. *density-ood* | Held-out or split-conformal threshold calibration | Adopt split-conformal calibration on held-out, deployment-clean rows and report the realised FPR with a CI; keep the law to predict the over-alarm where held-out data are missing; test X9's label-free spectral gap as its label-free form (Spearman +0.937 with the clean over-alarm over 16 CIFAR nets, discovery) | **calibration input**, usable now: predicts a train-referenced threshold's clean false-alarm rate within ±0.035 (7/7 d56). Action: never deploy a train-referenced threshold; calibrate on held-out clean data and re-calibrate after any weight change, including TTA |
| DO-4 | Far-OOD familiarity (CIFAR-100, SVHN) | penult sparse_frac; h on OOD batches | Flag rates r20 0.45-0.49 / 0.60-0.72, r56 0.78-0.79 / 0.85-0.92, nulls 0.10 / 0.03-0.12 (results/atlas_v0_resnet20_cifar10/SESSION.md:133; results/anomaly_h1/SESSION.md:332 INFO). Operating point confounded by DO-3 | MEASURED-NO-CLAIM | **UNTESTED** (no MSP, energy or Mahalanobis baseline) | **known-in-research** (a benchmark task whose standard baseline is MSP): Sun et al. 2022; ViM; OpenOOD v1.5. *density-ood* | Post-hoc OOD detection scored by AUROC / FPR95 with held-out ID calibration | Adopt the OpenOOD v1.5 protocol (near/far groups, held-out ID validation) with MSP, max logit and energy as the bar and normalised kNN, Mahalanobis / relative Mahalanobis, ViM and NECO as peers; X4's calibrated multi-tap profile is the Atlas form | **candidate** ESCALATE / do-not-adapt trigger: flag rates at a miscalibrated threshold only (clean 0.09-0.12 at the same threshold in r20); no AUROC or FPR95. Expected kNN gain over MSP at CIFAR-10 scale: about 2-3 AUROC points (OpenOOD v1.5) |
| DO-5 | Novelty-proportional compute | density + margin | Proposal only (atlas/invariants/density.py:6-9) | UNTESTED | **UNTESTED** | **partly-known**: early exits gated by internal-classifier confidence (BranchyNet; Shallow-Deep Networks); density gating not found. *density-ood* | Adaptive compute via confidence-gated exits | Adopt early-exit / cascade evaluation (accuracy against FLOPs) with confidence-gated exits as the baseline; the Atlas refinement to test is a harm-aware gate (familiarity plus H) (feasibility E-F2a; X10) | **not built**. F2-c is low on these backbones: a training-free exit at 8/9 of resnet20 agrees with the model on about 84% of inputs and saves about 11% of compute (feasibility §0) |
| LH-1 | Split-level harm grade H (AH-2) | penult median kNN log-radius shift, normalised by the reference q95-q50 spread (docs/plans/ANOMALY_H1.md:164) | Spearman(H, cost) 0.989 / 0.987; HOLD band (H ≤ 0.25 → ≤ 8 pt) holds; displacement magnitude 0.9996 / 0.995 and sparse_frac 0.994 / 0.992 match it; stem H 0.19 / 0.26 (results/anomaly_h1/SESSION.md:157-173, :375-378). About 99% of loss variance is between splits. d56 CIFAR-10-C | SUPPORTED-NOT-PROMOTED | **UNTESTED** vs AC, DoC, ATC (maxprob is stored but the probe reads only argmax: scripts/anomaly_probe.py:302) | **partly-known**: AutoEval Fréchet distance (Deng & Zheng 2021); feature distances are weaker than DoC / ATC across shift types (Guillory et al. 2021; Garg et al. 2022); the statistic plus HOLD band was not found. *label-free-accuracy-harm* | Label-free accuracy estimation; monitoring tools use confidence (CBPE) | Adopt AC, DoC and ATC-MC (free from the stored maxprob), GDE / agreement-on-the-line across existing seeds and the Dispersion score as baselines. Fit every estimator on discovery splits, freeze, and report MAE and Spearman on held-out corruption families, natural shift and label shift, with a nested test of whether H adds to ATC (X6) | **candidate** Gate-1 magnitude (HOLD-T / ADAPT): between-split ρ 0.99 is the level any monotone shift statistic reaches here; the HOLD band holds; d56 CIFAR-10-C only. Default to ATC / DoC until the nested test shows H adds |
| LH-2 | Batch harm grade h with a HOLD band (AX-3) | same, per batch of 256 | h > 0.25 flags batches losing > 10 pt 3350/3351 and 3253/3254; ≤ 2 pt 13/1050 and 26/1161; 5-10 pt 84-91%; all ≤ 10 pt 1099/2649 and 1198/2746; clean 0/200. Within-split Spearman 0.21-0.23; about 30 splits per seed; batches overlap (results/anomaly_h1/SESSION.md:318-337, :387-392) | SUPPORTED-NOT-PROMOTED (between splits) | **UNTESTED** | **partly-known**: NMD (Dong et al. 2022); BBSD (Rabanser et al. 2019); sequential harm monitors (Podkopaev & Ramdas 2022; Amoukou et al. 2024). *label-free-accuracy-harm* | Batch and sequential shift / harm tests with false-alarm control | Same baselines per batch (batch-mean maxprob, ATC, BBSDs p-value); wrap the best scores in a sequential monitor with a false-alarm budget (confidence sequences, CUSUM); ramps, skewed and mixed batches, batch sizes 16-256 (X3, X6) | **usable now** at narrow scope for HOLD-T (h ≤ 0.25: do not adapt, output trusted), d56 CIFAR-10-C, batch 256, with the flag counts in the evidence cell; 41% (s1) and 44% (s2) of all non-harmful (≤ 10 pt) batches are still flagged. No stream, delay or sequential false-alarm numbers; predicts harm, not adaptation benefit; no head comparison |
| LH-3 | Energy / DEVIATION: "no norm change means safe to hold" | penult norm_ratio; e(B) | 21-33% of batches with e ≥ 0.95 lose > 10 pt (results/anomaly_h1/SESSION.md:171, :331) | REFUTED | N/A | **known-in-research**: shift magnitude is not harm (Rabanser et al. 2019; Xie et al. 2023); norm behaves like confidence (Park et al. 2023). *label-free-accuracy-harm* | Norm as an OOD / confidence score, never a harm-calibrated hold rule | None; keep the norm only as a channel in a joint model | **do not use** in a hold rule (contradicts MASTER_SUMMARY.md:38) |
| LH-4 | Brightness is harmless → HOLD; stem-absorption mechanism (AH-5) | stem direction; class_sub_frac; H | Brightness H ≤ 0.264, cost ≤ 5.3 pt (AH-2(c)); mechanism clauses (a), (c), (d) fail (results/anomaly_h1/SESSION.md:169, :197-214) | Harm grading SUPPORTED-NOT-PROMOTED; mechanism REFUTED | **UNTESTED** | **partly-known**: benign-shift framing (Rabanser et al. 2019; Podkopaev & Ramdas 2022); luminance invariance with depth (Achille & Soatto 2018). *label-free-accuracy-harm, probing-corruption-geometry* | Avoiding needless alarms and retraining on benign shift | Adopt the benign-shift framing: compare the brightness grade with output-based harm estimates and with BN-statistics adaptation; a pixel or stem channel-mean check is the cheaper detector | **usable now** at narrow scope: HOLD-T for brightness at d56 CIFAR. No head comparison |
| ST-1 | Drift-family router (AX-2b) | pre-collapse tap; whitened unit mean-shift templates | 1.000 on 9/9 trained runs; random-init nulls route N/B/L at 0.99-1.00 (P 0.50 / 0.60); glass and frost route to N (results/anomaly_h1/SESSION.md:307-316, :383-386) | SUPPORTED-NOT-PROMOTED (near-trivial) | **UNTESTED** (the probe refuses logits; docs/plans/ANOMALY_H1.md:585-588) | **known-in-research**: DPCore (Zhang et al. 2025); distortion identification (DIIVINE, Moorthy & Bovik 2011); Fourier view of corruptions (Yin et al. 2019). *shift-type-tta-monitoring, probing-corruption-geometry* | Batch-statistic routing to prompts or experts, judged by downstream error | Adopt DPCore-style batch mean/std signatures with a new-domain (reject) rule and DIIVINE's identify-then-respond design; compare with a predicted-histogram or logit router and a pixel-statistic router; score by the downstream benefit of the routed response, not routing accuracy (X6, X14) | **candidate** at best: random nets route 3 of 4 families; closed set; no routed response shown to help |
| ST-2 | Tap where each drift first appears (AX-2a) | per-tap Hotelling T² | Gate CLOSED by 0.0017 AUC (results/anomaly_h1/SESSION.md:293-305) | UNTESTED (gate closed) | N/A (a geometry-only kind) | **partly-known**: corruptions are best adapted in early layers (Lee et al. 2023); multi-layer detectors (Lee et al. 2018). *shift-type-tta-monitoring* | Choosing layers to adapt; multi-layer OOD scores | Use surgical fine-tuning's result as the downstream test (does the tap profile predict which layers to adapt?); re-register with X3's per-tap change-information rate instead of an argmax-over-taps gate | **not built** |
| ST-3 | Class-orthogonal early drift and class-prior flag (AX-1) | T² off and in the class-mean span, pre-collapse tap | c3 failed by ceiling; single-class FPR failed through a class-4 test-mean leak (results/anomaly_h1/SESSION.md:272-291) | RULED-OUT | N/A; T_par was never compared with BBSE / BBSDh | **partly-known**: BBSE (Lipton et al. 2018); prevalence-adjusted re-test (Roschewitz et al. 2024); class-span decomposition (NECO); the batch drift form was not found. *shift-type-tta-monitoring* | Label-shift detection from the predicted-class distribution | Adopt BBSE / BBSDh on the head as the label-shift detector; X6's BBSE-explained class-mixture residual with held-out class means is the geometric refinement that removes the documented leak | **do not use** as registered. Label-shift gating of ADAPT is available from the head (BBSE / BBSDh), never run in Atlas |
| ST-4 | Per-sample off-simplex residual e_perp (AX-4) | penult residual off the class span | Loses to d1 by 0.16-0.44 AUROC (−0.03 to −0.10 with orientation flipped) (results/anomaly_h1/SESSION.md:339-350) | RULED-OUT | N/A | **partly-known**: residual-subspace scores for semantic OOD (ViM; Kamoi & Kobayashi 2020); the class-span covariate form was not found. *shift-type-tta-monitoring, density-ood* | Semantic-OOD residual scores combined with logits | Adopt ViM and NECO for semantic OOD (DO-4); X2's absolute, whitened head-null-space norm is the only kept successor | **do not use** |
| ST-5 | Abrupt class-mix change / label shift (Gate 3) | not geometry: second difference of the predicted-class histogram (legacy) | MASTER_SUMMARY.md:81-146. Legacy only: mean velocity compared with peak acceleration, spike pools defined by the model's predictions, no false-alarm rate or delay | UNTESTED in Atlas (legacy evidence, flawed) | It **is** a head signal | **known-in-research**: BBSDh (Rabanser et al. 2019); BBSE (Lipton et al. 2018); CUSUM (Page 1954); sequential tests (Podkopaev & Ramdas 2022). *shift-type-tta-monitoring* | Two-sample or sequential tests with false-alarm control, reported as FPR against delay | Adopt BBSDh, BBSE and CUSUM / GLR with ARL-calibrated thresholds; soft nearest-centre weights and X6's covariate veto are the geometric refinements to test (X15 inside X3) | **not built** as a validated ESCALATE trigger: no false-alarm rate or delay on record |
| ST-6 | Map deformation under TTA; adaptation benefit | same-space atlas compare at TENT checkpoints (merge_tau, adjacency_rho, d_nc1) | scripts/tta_deform.py:1-31; atlas/ladder.py:10-12; experiments/queue/tta_tent_resnet20_{fog3,collapse}.yaml. Never run (no results/tta_*); the planned bar is lenient (shift-type-tta-monitoring.md, K5) | UNTESTED | **UNTESTED** (designed bar: pred_entropy) | **partly-known**: PeTTA feature-mean divergence (Hoang et al. 2024); output triggers (SAR, ASR, AETTA); periodic reset (RDumb). Topology metrics as leading indicators were not found. *shift-type-tta-monitoring* | Collapse / reset triggers; label-free accuracy of the adapted model (AETTA, agreement-on-the-line) | Adopt the output triggers (SAR, ASR, AETTA, TTA-loss gradient norm), PeTTA's divergence and periodic reset as baselines; fix the ladder's bar (same false-onset rate on the benign run, same model state); per-step panel monitoring with panel-output baselines and an adaptation-benefit estimate (X11) | **not built**. The only planned source of the ADAPT / ROLLBACK decision variable |
| ST-7 | Legacy Gate-1 roster (DEVIATION, CONSENSUS, SUBNET, DRIFT_COH, PERSIST, Mahalanobis(CLD)) | legacy subnet, vote and trajectory axes (Upgraded-Mod) | MASTER_SUMMARY.md:36-47; never re-measured in Atlas | UNTESTED in Atlas | **UNTESTED** | **known-in-research** (the components): Mahalanobis (Lee et al. 2018); Gram matrices (Sastry & Oore 2019); sequential tests. *density-ood, shift-type-tta-monitoring* | Standard OOD / shift baselines with normalised features and held-out calibration | Re-enter only as normalised feature scores inside the baseline set (Mahalanobis++, Gram matrices), with head baselines (X1, X8) | **not built** in Atlas |
| ST-8 | Input-normalisation mismatch (v0 contrast rescale ≈ 0.82) | v0 → v1 same-space deformation (relrep, CKA) | Found from the training log, not from geometry (results/atlas_v0_resnet20_cifar10/SESSION.md:122-127). Penult relrep argmax agreement 0.9745 (results/atlas_v1_resnet20_s0hub/compare_vs_atlas_v0_resnet20_cifar10/deformation.json) vs head argmax agreement 0.9724 (results/norm_check_resnet20/norm_check.json). One instance | MEASURED-NO-CLAIM | **NO** sign of extra information | **known-in-research**: two-sample shift tests (Rabanser et al. 2019); BN-statistics discrepancy (Dong et al. 2022). *shift-type-tta-monitoring, probing-corruption-geometry* | Covariate-shift monitoring; BN adaptation absorbs it | Adopt an activation-statistics self-test at deployment start (NMD on BN running means, or a two-sample test against a clean reference) | **candidate** pipeline check (F1-b); one instance |
| CD-1 | ID hunchback and penult drop (row 1) | TwoNN per tap (atlas/invariants/dimension.py:40-70) | ATLAS_STATUS.md:10; drop 0.488-0.503 (r20), 0.471 / 0.463 (r56); nulls drop 0.020 / 0.001; the peak location is architectural (random init also peaks) | ESTABLISHED | N/A (checkpoint descriptor) | **known-in-research**: Ansuini et al. 2019 (cited in dimension.py:7-9); Atlas partly contradicts it (random init peaks too). *collapse-dimension-similarity* | Generalisation proxy; unsupervised layer selection (Valeriani et al. 2023) | Keep as an instrument check; possibly unsupervised tap selection on a new backbone (Valeriani et al. 2023) | **no controller role** (instrument check) |
| CD-2 | TwoNN ID reproduces the PCA spectrum (AH-8, AH-8d) | TwoNN vs Gaussian-spectrum twin (ID_gauss) | Penult ρ 0.99-1.00; at the ID peak TwoNN is 26-28% below its twin (ρ 0.721 / 0.743); random nulls pass (a) (results/anomaly_h1/SESSION.md:256-270) | SUPPORTED-NOT-PROMOTED | N/A (a negative for TwoNN) | **partly-known**: the Gaussian-twin control (TwoNN on a Gaussian with the same second-order moments) is Ansuini et al. 2019's own (Fig. 5B), where the twin's ID at the VGG-16 last hidden layer was two orders of magnitude above the data's; Atlas's penult result is the opposite. *collapse-dimension-similarity* | TwoNN vs PCA contrasts | Keep ID_gauss as the standard control next to any TwoNN | **do not use** TwoNN as a controller feature (docs/plans/ANOMALY_H1.md:571-572) |
| CD-3 | Deeper net has higher penult ID at matched accuracy (row 11, D-ID) | penult TwoNN; ID_gauss | ATLAS_STATUS.md:21; results/atlas_v1_resnet56_s1/SESSION.md:246-305; results/anomaly_h1/SESSION.md:264-266. L1 (matched): TwoNN 10.72-10.77 vs edge 10.31, spectral excess (ID_gauss 11.49-11.70). L2 (full recipe): excess real (s1 0.347 above the r20 maximum, about 3× twin \|Δ\|) but non-spectral (ID_gauss 10.471 / 10.508 ≤ 10.698). L1 fit confound | ESTABLISHED by rule (D-ID ✅); needs the L1/L2 qualifier | Beyond head accuracy: yes. Against head calibration: UNTESTED | **not-found-in-prior-art**; in tension with the tunnel / compression literature (Masarczyk et al. 2023). *collapse-dimension-similarity* | None found | Add the L1/L2 qualifier; test against head calibration if it is ever used | **no controller role** |
| CD-4 | Collapse level / training fit of a checkpoint (D-COLL, F2, AH-7) | nc1, sep_ratio, bridge ratio, centre-distance CV, stage-3 compression | nc1 0.129-0.130 vs edge 0.154, 18× twin noise (results/atlas_v1_resnet56_s1/SESSION.md:246-305); Pearson(log10 nc1, block) −0.973 (results/atlas_v1_resnet56_s0hub/SESSION.md:317); D-COLL 🟡 FIT-SENSITIVE; AH-7 REFUTED (results/anomaly_h1/SESSION.md:237-254) | nc1 ESTABLISHED; sep / bridge CANDIDATE; training-length law REFUTED | Beyond head accuracy: yes; drives DO-3. Against head calibration: UNTESTED | **known-in-research**: neural collapse (Papyan et al. 2020); law of equi-separation (He & Su 2023); NC is train-only (Hui et al. 2022). *collapse-dimension-similarity* | Training diagnostic; transferability scores (NCTI, VCI); OOD detector design | Adopt the full NC suite (NC1-NC4, the invariant VCI, equi-separation ρ) and measure it on held-out test data as well as the train reference; X9's label-free spectral gap (Spearman −0.932 with nc1 over 16 nets) is the robot-side form | **calibration input**: sets the DO-3 false-alarm rate and says when penult geometry duplicates the head; re-measure after any weight change |
| CD-5 | Class-valley depth (row 2; ImageNet valleys) | sep_ratio; legacy valley ratio | ATLAS_STATUS.md:11 (≈ 3.0 on 5 r20 seeds; r56 5.1-5.3); ImageNet sep_ratio_ref 0.75-0.93 (results/margin_b1_vitb16/SESSION.md:426); the legacy ratio depends on per-class n (results/margin_b1_vitb16/DIAGNOSIS_G0.md:36-60) | CANDIDATE (🟡; understated, 5 seeds agree) | N/A | **known-in-research**: NC1/NC2 separability (Papyan et al. 2020); K > d prevents an ETF (Jiang et al. 2024). *collapse-dimension-similarity* | NC proxy; transfer predictor | Use a per-class-n-robust ratio and generalised-NC expectations when K > d | **no controller role** (context for DO-3 and CM-6) |
| CD-6 | Class adjacency / merge order (row 7) | centre-distance ranks (D1); single linkage (atlas/invariants/adjacency.py:23-33) | ATLAS_STATUS.md:17. d20: s3-s4 D1 0.919 (class-jackknife SE 0.058), nulls 0.589-0.644. d56: C7 failed (D1 0.692); 🟡 via one hub pair (0.840) (results/atlas_v1_resnet56_s1/SESSION.md:425-496) | ESTABLISHED d20; CANDIDATE d56 (C7 failed); merge order CANDIDATE | **UNTESTED** vs the head confusion matrix / weight cosines | **partly-known**: convergent learning (Li et al. 2016); NC2 predicts the loss at collapse. *collapse-dimension-similarity* | Class structure is usually read from the head (confusion) | Compare with the head's confusion matrix and classifier-weight cosines (NC3 predicts equality); at full collapse look for class structure in within-class residuals (Yang et al. 2023) | **no controller role** until it beats the head's confusion structure |
| CD-7 | Cross-model agreement (CKA, relrep; row 8, S9) | penult linear CKA; relative representations on CIFAR-100 (atlas/compare.py:15-20, :166, :185) | ATLAS_STATUS.md:18; relrep 4.67-4.72× chance; cka_test 0.883-0.922, fit 0.9405 − 0.3154·nc1 (results/atlas_v1_resnet56_s1/SESSION.md:675-715); within-class-residual CKA not run | CANDIDATE (🟡) | **UNTESTED** vs head disagreement | **known-in-research**: CKA (Kornblith et al. 2019); relative representations (Moschella et al. 2023); head disagreement estimates error (Jiang et al. 2022; Baek et al. 2022). *collapse-dimension-similarity* | Model comparison and stitching; label-free accuracy via head agreement | Adopt two-seed head disagreement (GDE) and agreement-on-the-line as the label-free accuracy tools; add residual CKA and Procrustes (Ding et al. 2021); per-sample representation disagreement is X16 | **no controller role** as measured; head disagreement across Atlas's existing seeds is a ready LH baseline |
| CD-8 | Depth-invariant shape vs depth- and fit-dependent level | SHAPE / LEVEL labels across taps | results/atlas_v1_resnet56_s1/SESSION.md:478-512; ATLAS_STATUS.md:10-18. r20 → r56 on CIFAR only | ESTABLISHED (r20 → r56 CIFAR only) | N/A | **partly-known**: cross-depth similarity (Nguyen et al. 2021); equi-separation across architectures (He & Su 2023). *collapse-dimension-similarity* | An analysis result | Treat as a transfer rule; test the shapes on a second dataset or recipe | **calibration input** (rule): re-calibrate levels per checkpoint; only shapes may carry over |
| CD-9 | Collapse as a hidden coordinate (synthesis) | nc1 across items | Would explain the D1 drop, CKA excess, AH-1 law and margin ≈ distance; AH-3 and AH-7 refuted, AH-4 not evaluable, residual CKA unrun (results/atlas_v1_resnet56_s1/SESSION.md:701) | CANDIDATE (hypothesis) | N/A | **partly-known**: NC degree trades OOD detection against transfer (Harun et al. 2025). *collapse-dimension-similarity* | NC as a control knob; transferability estimate | Test via residual CKA and the DO-3 law under L2 normalisation; X9 as the label-free coordinate | **calibration input** (hypothesis): X9's label-free proxy tracks the clean over-alarm (Spearman +0.937, 16 CIFAR nets, discovery) |
| CD-10 | Non-core scalars (participation ratio, early nc1, hubness, dim95) | pca_spectrum, neural_collapse, hubness | Not seed-stable (stem PR 0.1555, layer3.1 nc1 0.1530 vs a 0.15 tolerance; results/atlas_v1_resnet20_s3/SESSION.md:283-299) | MEASURED-NO-CLAIM | N/A | **known-in-research**: PCA-based linear dimension next to TwoNN (Ansuini et al. 2019); NC in intermediate layers (Parker et al. 2023); hubness (Radovanović et al. 2010). *collapse-dimension-similarity* | Routine diagnostics | None | **no controller role** |
| CD-11 | Class readout from geometry vs head accuracy (D-ACC) | nearest-centre accuracy; probe excess | 0.9446 / 0.9398 vs head 0.9438 / 0.9404 (results/atlas_v1_resnet56_s1/SESSION.md:307-318) | ESTABLISHED (as a negative) | **NO**: equals head accuracy | **known-in-research**: NC4 (Papyan et al. 2020). *collapse-dimension-similarity* | Collapse diagnostic | Use the label-free per-sample variant instead (CM-11) | **no controller role** |
| PC-1 | Nuisance-factor washout (row 3) | linear probes per tap (atlas/invariants/decodability.py:57-79); washout (atlas/invariants/flow.py:84) | ATLAS_STATUS.md:12; luminance washout 0.549-0.736 (r20), 0.608 / 0.669 (r56); highfreq and anisotropy < 0.20; the d56 null passes luminance (0.473) and anisotropy (0.134); the probe pool mixes clean and corrupt splits and includes the confirmation corruptions | ESTABLISHED; weak at d56 | N/A as registered; a logit-probe comparison is possible (probing-corruption-geometry.md §6, B1) | **partly-known**: layer-wise probing (Alain & Bengio 2016); invariance with depth (Achille & Soatto 2018); random filters are frequency-selective (Saxe et al. 2011). *probing-corruption-geometry* | Locating factors; choosing taps | Adopt control tasks (Hewitt & Liang 2019) and learned-minus-random as the headline; clean-only and within-split pools; folds grouped by source image; a logit-probe comparison | **calibration input** (tap choice): read nuisance and sensor information before the penult |
| PC-2 | Corruption type / family / severity decodability | linear probes, 18 factors | Seed-stable profiles, MAD 0.009-0.020 (results/atlas_v1_resnet20_s3/SESSION.md:129-146); the corruption-axis holdout leaked | MEASURED-NO-CLAIM | **UNTESTED** | **known-in-research**: distortion identification (DIIVINE, Moorthy & Bovik 2011). *probing-corruption-geometry* | First stage of no-reference image-quality assessment; type-specific response | Adopt DIIVINE's two-stage design (identify the distortion, then respond); add control tasks, a pixel-statistic router, head baselines and leave-one-corruption-out folds | **candidate** at best (holdout leaked; no baselines) |
| PC-3 | Class commit layer; reorganisation block (row 4) | commit_layer (atlas/invariants/flow.py:45-84); layer CKA | ATLAS_STATUS.md:13; layer3.1 (r20) / layer3.5 (r56); the biggest reorganisation comes right after the ID peak (results/atlas_v1_resnet56_s0hub/SESSION.md:261, :375) | ESTABLISHED (low information) | N/A | **known-in-research**: Alain & Bengio 2016; tunnel effect (Masarczyk et al. 2023); intermediate NC (Rangamani et al. 2023). *probing-corruption-geometry, collapse-dimension-similarity* | Picking layers for probing and transfer | Use it to locate the pre-collapse tap on a new backbone | **calibration input** (which tap to read) |
| PC-4 | Displacement coherence (row 5, C4) | mean cosine of paired displacements (atlas/invariants/sensitivity.py:41-46) | ATLAS_STATUS.md:14; 6/6 in 6 trained nets, minimum margin 0.105-0.192; nulls 3/6-4/6; r 0.994-0.996 with the mean-shift ratio (its content is "smaller mean shift") | ESTABLISHED | **UNTESTED**; needs the clean twin (offline only) | **partly-known**: corruption as a shift of feature statistics (Schneider et al. 2020; NMD, Dong et al. 2022); the coherence statistic itself was not found. *probing-corruption-geometry* | BN re-estimation; activation-mean OOD detection | Test the proposed Fourier mechanism with frequency-controlled perturbations (Yin et al. 2019); the runtime analogue is an unpaired batch mean shift (NMD) | **no controller role** at runtime (needs the clean twin) |
| PC-5 | Motion directional / noise path ratio / C4 tracks collapse (AH-6) | displacement directions and magnitudes across taps | results/anomaly_h1/SESSION.md:216-235 | REFUTED | N/A | **partly-known**: corruptions are best seen and adapted early (Lee et al. 2023). *probing-corruption-geometry* | — | None | **do not use** |
| PC-6 | Semantic (class-subspace) fraction of a shift | class_sub_frac (atlas/invariants/sensitivity.py:67) | Tracks the between-class variance share B/T 0.930 / 0.931 at the collapsed penult (results/anomaly_h1/SESSION.md:205; docs/plans/ANOMALY_H1.md:459) | REFUTED as a feature | N/A | **partly-known**: explained by neural collapse (Papyan et al. 2020); NECO uses a per-sample version for OOD. *probing-corruption-geometry, collapse-dimension-similarity* | Residual-subspace OOD scores read the complement instead | Read complement or residual directions at pre-collapse taps instead (ViM, NECO; X2) | **do not use** at the penult |

### 5.1 Tallies

Each row is counted once, by its primary status (mixed rows are counted by the first status listed).

| status | rows | count |
|---|---|---|
| ESTABLISHED (fully or in part) | CM-1, CM-3, CM-4, CM-9, DO-1, CD-1, CD-3, CD-4, CD-6, CD-8, CD-11, PC-1, PC-3, PC-4 | 14 |
| SUPPORTED-NOT-PROMOTED | DO-3, LH-1, LH-2, LH-4, ST-1, CD-2 | 6 |
| CANDIDATE | CM-2, CM-5, CD-5, CD-7, CD-9 | 5 |
| REFUTED | CM-6, CM-7, CM-8, CM-10, DO-2, LH-3, PC-5, PC-6 | 8 |
| RULED-OUT | ST-3, ST-4 | 2 |
| UNTESTED | CM-11, DO-5, ST-2, ST-5, ST-6, ST-7 | 6 |
| MEASURED-NO-CLAIM | DO-4, ST-8, CD-10, PC-2 | 4 |

Secondary parts: CM-4 is CANDIDATE at r56 and REFUTED at r20; CD-4's sep/bridge part is CANDIDATE and its training-length
law (AH-7) REFUTED; CD-6 at depth 56 and its merge order are CANDIDATE; LH-4's mechanism (AH-5) is REFUTED; CM-6's AH-4
part is UNTESTED.

| beyond the output head? | rows | count |
|---|---|---|
| NO | CM-1, CM-3 (mostly), CM-7, CM-8, CM-9, CD-11, ST-8 | 7 |
| PARTIAL (only against maxprob) | CM-2, CM-4 | 2 |
| CANDIDATE | CM-5 | 1 |
| beyond head accuracy only; calibration untested | CD-3, CD-4 | 2 |
| UNTESTED | CM-10, CM-11, DO-1, DO-2, DO-4, DO-5, LH-1, LH-2, LH-4, ST-1, ST-6, ST-7, CD-6, CD-7, PC-2, PC-4 | 16 |
| N/A | CM-6, DO-3, LH-3, ST-2, ST-3, ST-4, ST-5, CD-1, CD-2, CD-5, CD-8, CD-9, CD-10, PC-1, PC-3, PC-5, PC-6 | 17 |

| already known? (starting point) | count | rows |
|---|---|---|
| standard-practice | 1 | CM-1 |
| known-in-research | 20 | CM-3, CM-7, CM-9, CM-10, CM-11, DO-1, DO-4, LH-3, ST-1, ST-5, ST-7, ST-8, CD-1, CD-4, CD-5, CD-7, CD-10, CD-11, PC-2, PC-3 |
| partly-known | 22 | CM-2, CM-4, CM-6, CM-8, DO-2, DO-3, DO-5, LH-1, LH-2, LH-4, ST-2, ST-3, ST-4, ST-6, CD-2, CD-6, CD-8, CD-9, PC-1, PC-4, PC-5, PC-6 |
| not-found-in-prior-art | 2 | CM-5, CD-3 |

| functional readiness (primary label per row) | count | rows |
|---|---|---|
| usable now (narrow scope) | 4 | CM-1 and CM-3 (through the head's own statistic, not the margin), LH-2, LH-4 |
| candidate | 8 | CM-2, CM-5, DO-1, DO-4, LH-1, ST-1, ST-8, PC-2 |
| calibration input | 8 | CM-4, CM-6, DO-3 (usable now as a calibration rule), CD-4, CD-8, CD-9, PC-1, PC-3 |
| not built | 6 | CM-11, DO-5, ST-2, ST-5, ST-6, ST-7 |
| do not use | 11 | CM-7, CM-8, CM-9 (as an early sensor), CM-10 (margin_norm), DO-2, LH-3, ST-3 (as registered), ST-4, CD-2 (TwoNN), PC-5, PC-6 |
| no controller role | 8 | CD-1, CD-3, CD-5, CD-6, CD-7, CD-10, CD-11, PC-4 |

**Reading.**
- **What a controller can act on today** is narrow and mostly head-side. Per sample, the head's own best statistic is
  the trust score (CM-1, CM-3). Per batch, h with its HOLD band can drive "hold, output trusted" at depth-56 CIFAR-10-C
  (LH-2, LH-4). DO-3 tells every density threshold how to be calibrated.
- **Where geometry could add most, the comparisons were never run.** 16 rows have never been compared with the head.
  The cheapest to settle are CM-11 (a published detector type, on dumps that exist), LH-1/LH-2 against AC/DoC/ATC
  (maxprob is stored), DO-1/DO-4 against MSP and normalised kNN, and CM-5's joint detector.
- **Most established rows are checkpoint descriptors** (CD, PC) that act as calibration or configuration inputs, not
  runtime signals.

## 6. Functional readiness by controller function

What the controller can drive today, how reliably, and what is missing. Sub-variant labels follow
[controller-functions-feasibility.md](controller-functions-feasibility.md) §0; "X*n*" are the ranked proposals in
`docs/reviews/EXTRACTION_PROPOSALS_2026-09-23.md`.

| controller function | what can drive it today (scope) | reliability on record | cheapest baseline it must beat or complement | what is missing | rows | proposals |
|---|---|---|---|---|---|---|
| F1-a per-sample "likely wrong" flag (trust score) | The head's best statistic, chosen offline per checkpoint (maxprob or a logit score on CIFAR; logit gap on the ViTs and ResNet50) | All-errors AUC 0.91-0.93 on CIFAR and 0.80-0.87 on ImageNet (feasibility §0); geometry adds +0.008 to +0.037 over the logit gap only inside confident strata on 2 ViTs | logit gap; p-norm max logit | AURC and risk at coverage; a joint detector; any shift condition; CM-11 scored | CM-1, CM-3, CM-5, CM-11 | X1, X7, X8 |
| F1-b per-sample sensor-fault / degradation flag | Nothing validated | Per sample, strong only for noise at s3/s5 (layer-2.8 density 0.95-0.996, hub INFO); mild s1 shifts at chance (0.50-0.51) | max-softmax; simple pixel checks; normalised kNN | A calibrated per-sample flag with a held-out false-alarm rate; localised faults; real sensor faults | DO-1, DO-2, ST-4, ST-8 | X4, X5, X14 |
| F1-c pre-model flag | Nothing; the evidence is against a label-free form | Error information near chance over the first half of the network | — | (supervised probes were never tested) | CM-9 | — |
| F1-d / Gate 1: stream or batch harmful-shift flag; HOLD-T | h with its HOLD band, d56 CIFAR-10-C, batch 256 | > 10 pt batches flagged 3350/3351 and 3253/3254; ≤ 2 pt 13/1050 and 26/1161; clean 0/200; 41-44% of all ≤ 10 pt batches flagged; within-split ρ 0.21-0.23 | batch-mean maxprob (AC), DoC, ATC | Head baselines on the same batches; streams, ramps and delay; sequential false-alarm control; other shift types | LH-1, LH-2, LH-4 | X3, X6 |
| HOLD-D (hold, output distrusted) | Per sample: the F1-a head statistic. Label shift: nothing run (head BBSE / BBSDh available) | — | logit gap; BBSE | CM-11 as a detector; a label-shift test on skewed batches | CM-11, ST-3, ST-5 | X1, X6 |
| ADAPT / ROLLBACK (adaptation benefit; collapse of an adaptation) | Nothing | — | pred_entropy; SAR / ASR triggers; periodic reset | The TTA ladder; a benefit estimate (AETTA, agreement-on-the-line) | ST-6 | X11 |
| ESCALATE (semantic novelty; abrupt class-mix change) | Nothing validated; density flags CIFAR-100 / SVHN at a miscalibrated threshold | Flag rates only (r20 0.45-0.49 / 0.60-0.72 against clean 0.09-0.12) | MSP / energy for OOD; CUSUM or BBSDh for class-mix change | OOD AUROC and FPR95 with held-out calibration; a sequential Gate-3 test | DO-4, ST-5 | X4, X3 (X15) |
| F2 information-gated feeding (F2-a/b/c) | Nothing | Early exit has a low prior here (training-free exit at 8/9 of resnet20: about 84% agreement, about 11% saving) | pixel-difference gate; confidence-gated exits | Video with temporal redundancy; a feature-rate curve | DO-5, CM-9 | X10, X5 |
| F2-d gating adaptation compute | The HOLD band (as HOLD-T) | As F1-d | entropy-based sample gating (EATA, SAR) | Any closed-loop run | LH-2 | X11, X6 |
| Configuration and calibration (offline) | DO-3 law; collapse level; tap choice; per-checkpoint score choice | DO-3 within ±0.035 at 7/7 d56 instances | held-out split-conformal calibration | Held-out calibration protocol in every probe; X9 as a label-free proxy | DO-3, CM-4, CM-6, CD-4, CD-8, CD-9, PC-1, PC-3 | X9, X4 |

## 7. Classification conflicts and how they were resolved

The topic files classify some items differently, because each reads the item through its own literature. The merged
label in section 5 follows these rules: the topic that owns the item (its prefix) decides, unless another topic cites a
closer precedent; "partly-known" is used when the idea is published but Atlas's specific form or number was not found.

| item | classifications in the topic files | merged | reason; where the details are |
|---|---|---|---|
| CM-11 | known-in-research (confidence-margin, density-ood); partly-known (collapse, label-free) | known-in-research | Direct precedents exist for the same use: Trust Score, DkNN and SelfChecker all flag errors by geometry-vs-classifier disagreement. The collapse topic's "not found" refers to its own literature. *confidence-margin* |
| DO-4 | standard-practice (confidence-margin, label-free); known-in-research (density-ood, shift) | known-in-research | Far-OOD detection is a standard benchmark task and MSP is the standard baseline, but the feature-density detectors Atlas uses (kNN, Mahalanobis) are research methods, not deployed defaults. *density-ood* |
| LH-1 | partly-known (confidence-margin, label-free, shift); known-in-research (density-ood, collapse, probing) | partly-known | The estimator class (feature-distribution distance → accuracy, AutoEval 2021) is known; the median kNN-radius statistic with per-backbone normalisation and an explicit HOLD band was not found. *label-free-accuracy-harm* |
| LH-2 | partly-known (most topics); known-in-research (probing) | partly-known | Batch shift and harm tests are known; the harm-calibrated HOLD band on this statistic was not found. *label-free-accuracy-harm* |
| LH-3 | known-in-research (label-free); partly-known (density-ood, shift) | known-in-research | "A shift statistic that does not move does not certify no harm" is a direct published point (Rabanser et al. 2019; Xie et al. 2023). *label-free-accuracy-harm* |
| ST-4 | not-found-in-prior-art (shift); known-in-research (density-ood); partly-known (probing) | partly-known | Residual-subspace scores (ViM, Kamoi & Kobayashi) are known for semantic OOD; the class-span residual as a per-sample covariate-drift sensor was not found (and was ruled out). *shift-type-tta-monitoring* |
| ST-7 | known-in-research (density-ood); partly-known (shift) | known-in-research | Every component has a standard research baseline (Mahalanobis, Gram matrices, sequential tests). *density-ood* |
| ST-8 | known-in-research (label-free, shift); partly-known (probing) | known-in-research | A global normalisation mismatch is an ordinary covariate shift that two-sample tests and BN-statistics checks target. *shift-type-tta-monitoring* |
| CD-4 | known-in-research (collapse); partly-known (density-ood) | known-in-research | NC1 and layer-wise separation (equi-separation) are published; the density-ood reading refers to the Atlas-specific use of nc1 to predict a false-alarm rate, which is recorded under DO-3. *collapse-dimension-similarity* |
| CM-4, CM-2, CM-8, DO-3 | consistent (partly-known) across topics | partly-known | — |

Sub-components that stay "not found in prior art" inside partly-known rows: the DO-3 log-nc1 law, the CD-2 result
that penult TwoNN tracks its Gaussian twin (the control itself is Ansuini et al. 2019's), the CM-8 dataset sign flip, the ST-4 class-span covariate form, and the ST-6 topology
metrics as TTA leading indicators (untested). Section 9 lists them with their scope.

## 8. Adoption and refinement plan

### 8.1 Principles

1. **Adopt as a component** where a published method already does the controller's job, and refine it with what Atlas
   knows (which tap carries which information, collapse-aware calibration, harm grading). Examples: the conformal kNN
   flag (X4), BBSE for label shift, CUSUM or confidence sequences for streams, BN-statistics adaptation with periodic
   reset, ATC as the default harm magnitude.
2. **Adopt as a baseline** (the bar) the cheapest head statistic for each function, and the strongest standard feature
   detector. A geometric score earns a place in the controller if it beats the bar **or complements it** (a positive
   increment in a joint model).
3. **The incremental-over-head test** is the common currency: a cross-fitted joint model, head-only against head plus
   geometry (ΔAUC_joint, ΔI in bits, AUC within logit-gap deciles, AURC), with folds grouped by image row and, for shift
   targets, by corruption family (X1's HEAD-ADDITIVE rule). For early taps, also condition on pixel statistics.
4. **Report functional numbers**, not only AUROC: the false-alarm rate at a stated budget (clean and benign-shift),
   detection delay at a fixed ARL0, risk at coverage, and latency and memory.
5. **Calibrate on held-out, deployment-clean data**, and re-calibrate after any weight change, including TTA (DO-3).

### 8.2 By information family

"Free" means computable from artifacts that already exist while the RunPod volume survives: CIFAR `preds/<split>.npz`
hold argmax and maxprob for every split (atlas/extract_acts.py:188); CIFAR logits are recomputable as penult·W + b from
the checkpoints (float16 penult dumps; re-extraction in float32 is cheaper and cleaner); ImageNet `preds` carry
logit_top1, logit_top2 and logit_gap (atlas/extract_imagenet.py:555-573).

| family | adopt as components (starting points) | adopt as baselines (the bar) | Atlas refinement toward controller use | protocol requirements | cost | details |
|---|---|---|---|---|---|---|
| Per-sample error (CM) | Selective classification with post-hoc head-statistic choice (Geifman & El-Yaniv 2017; Cattelan & Silva 2024); disagreement detectors (Trust Score, DkNN, SelfChecker); ViM's joint-score design | MSP; MSP at fitted temperature; max logit; **logit gap**; softmax margin; entropy; DOCTOR; energy score; **p-norm-normalised max logit** | Centre margin and nearest-centre disagreement as inputs to a joint detector; the normalisation ablation for CM-5; late-block trajectories (X7); local neighbourhood scores (X8) | Primary endpoint on all errors; AURC, FPR@95, risk at 80/90/95% coverage; freeze the best head statistic on discovery; cross-fitted joint detector; ReaL labels on ImageNet; rerun under shift | Free (CIFAR logits need the checkpoint) | confidence-margin.md §6; X1, X7, X8 |
| Novelty / OOD / per-sample shift (DO) | Deep kNN on L2-normalised features (Sun et al. 2022); split-conformal p-values (Bates et al. 2023); layer-wise p-value fusion (Raghuram et al. 2021; Cauchy combination) | MSP; max logit; energy; entropy; peers Mahalanobis / relative Mahalanobis / Mahalanobis++, ViM, NECO, a same-recipe deep ensemble from existing checkpoints | X4: held-out calibration, multi-tap fusion using tap complementarity (layer-2.8 density for noise, penult d1 for jpeg), the same code on every backbone; pairing with H so that harmless novelty does not trigger action | OpenOOD v1.5 CIFAR-10 near/far groups; held-out ID validation; realised FPR with a CI; AUROC / FPR95; keep random-init nulls | Free for existing splits; new OOD sets need extraction | density-ood.md §6; X4 |
| Harm / label-free accuracy (LH) | ATC / DoC as the default Gate-1 magnitude; confidence sequences or CUSUM as the monitor (Podkopaev & Ramdas 2022; Amoukou et al. 2024); agreement-on-the-line across existing seeds | AC (batch-mean maxprob); DoC; ATC-MC / ATC-NE; entropy; predicted-histogram shift (BBSD); mean logit gap; nuclear norm, MaNo, COT; GDE | H as a candidate second axis where the head is known to fail (noise, overconfident shift, headless encoders); the HOLD band at a matched flag rate; a skew-robust version (X6) | Calibrate on discovery splits and freeze; MAE and Spearman on held-out corruption families; nested test (does H add to ATC/DoC?); batch sizes 16-256, ramps, mixed and label-skewed batches; sequential false-alarm budget; fresh seeds | Tier 0 free (maxprob); tier 1 needs logits | label-free-accuracy-harm.md §6; X3, X6 |
| Shift type, label shift, Gate 3 (ST) | BBSE / BBSDh for label shift; CUSUM / GLR for Gate 3; DPCore-style routing with a new-domain rule; MMD two-sample tests | BBSDs (softmax KS); BBSDh (predicted-label chi-squared); BBSE prior estimate; CUSUM on predicted-class frequencies; peak-velocity baseline | X6's BBSE-explained class-mixture residual (skew-robust covariate test); routers scored by the downstream benefit of the routed response; soft nearest-centre weights for Gate 3 | Detection power vs batch size at fixed α; FPR (ARL0) vs delay on seeded streams; pure-skew, pure-covariate and mixed batches | Needs logits in the probe (or recomputation) | shift-type-tta-monitoring.md §6; X6, X15 |
| TTA monitoring (ST-6) | BN-statistics adaptation with periodic reset (RDumb) as the policy floor; EATA / SAR sample gating; AETTA's benefit estimate | pred_entropy; SAR entropy moving average; ASR prediction concentration; TTA-loss gradient norm; AETTA estimated accuracy; PeTTA's class-mean divergence | Geometric monitors (panel drift, residual CKA, NC3+ alignment, participation ratio) must lead PeTTA and ASR at a matched false-onset rate; the adaptation-benefit estimate recorded with them (X11) | Same false-onset rate on the benign run; same model state for all monitors; correlated, recurring and small-batch streams | TTA manifests exist, never run | shift-type-tta-monitoring.md §6; X11 |
| Collapse / similarity (CD) | The NC suite (NC1-NC4), VCI, equi-separation ρ; GDE / agreement-on-the-line | Test confusion matrix; classifier-weight cosines; two-seed head disagreement; logit-vector CKA | Collapse level as an offline calibration coordinate, label-free via X9's spectral gap; within-class-residual CKA; test-side NC | Input-bootstrap CIs and permutation nulls; held-out data as well as the train reference | Free | collapse-dimension-similarity.md §6; X9, X16 |
| Probing / corruption geometry (PC) | Control tasks (Hewitt & Liang 2019); DIIVINE's identify-then-respond design; NMD; Gram matrices; surgical fine-tuning's layer choice | Linear probes on the 10-d logits; probes on scalar head statistics (maxprob, entropy, logit gap); pixel-statistic router | Read type and "what changed" at pre-collapse taps, harm and trust at the penult; learned-minus-random as the headline; beyond-GAP pooling (X5) | Clean-only and within-split pools; leave-one-corruption-out; folds grouped by source image; confirmation corruptions removed from every pool | Free (X5 needs re-extraction) | probing-corruption-geometry.md §6; X5 |
| Controller functions F1 / F2 | Sequential detectors with calibrated false alarms; change-triggered frame or token reuse with forced refresh; early exits; event-triggered sensing | Pixel-difference gates; confidence-gated exits; output-only monitors | The coupling Atlas can supply: which tap carries which information, and a harm grade that keeps a change gate from treating harmless novelty as a trigger | Streams with injected faults and drift; accuracy against compute; false alarms against delay | Varies | controller-functions-feasibility.md §3, §6; X3, X10 |

Already adopted as instruments (cited in code): TwoNN (Facco et al. 2017), the ID profile (Ansuini et al. 2019),
neural-collapse metrics (Papyan et al. 2020), relative representations (Moschella et al. 2023), error consistency
(Geirhos et al. 2020) (atlas/invariants/dimension.py:5-9; landmarks.py:6-7; compare.py:18, :205). Atlas refined them with reference-draw
twins, random-init nulls and the Gaussian spectral twin applied at every tap (CD-2); the last two extend controls that
Ansuini et al. 2019 ran on untrained networks and at one layer.

### 8.3 Cross-cutting rules

- **Beat or complement the best head statistic, not maxprob.** Maxprob is saturated on CIFAR (68-79% of samples ≥ 0.999)
  and broken by soft-label recipes on ImageNet.
- **Report incremental value** with a nested model cross-fitted on held-out folds, not only a conditional AUROC.
- **Between-split rank correlation on synthetic corruptions cannot separate geometry from the head** (about 99% of loss
  variance is between splits); use held-out shift families, natural shift and label shift.
- **Calibrate on held-out, deployment-clean data**, and re-calibrate after any change of weights, including TTA.

## 9. What Atlas has found that prior art does not cover

Factual, at the narrowest scope the evidence supports; every "not found" rests on a search that ran out of budget. The
functional value of each item is in its readiness cell in section 5.

1. **CM-5.** A matched-confidence lead of the class-centre margin over the head's logit gap: V3b, pre-registered and
   predicted EQUIVALENT, ahead in all four runs at every cut 0.5-0.8 (+0.008 to +0.037, p ≤ 0.047) on two DeiT-lineage
   ViT-B/16. No lead on all errors; BELOW on ResNet50; not separated from normalisation; no joint-detector test.
2. **DO-3.** A law for the clean false-alarm rate of a train-referenced density threshold: sparse ≈ 0.0224 −
   0.108·log10 nc1, within ±0.035 at 7/7 depth-56 instances. The mechanism and the remedy (held-out calibration) are
   published.
3. **CM-2.** Within the resnet56 family, the margin − distance lead falls as collapse deepens (Spearman(sep, lead)
   −0.857, n = 7); the direction is predicted by NC theory, and across ImageNet models the relation runs backwards.
4. **CM-8.** The error-vs-correct energy sign differs between CIFAR (0.750-0.795) and ImageNet (0.23-0.30).
5. **CD-2.** With the Gaussian-twin control of Ansuini et al. 2019 applied at every tap, penult TwoNN is effectively
   spectral (ρ 0.99-1.00), the opposite of Ansuini's VGG-16 result; the ID peak is 26-28% below its twin.
6. **CD-3.** A higher penult TwoNN in deeper nets at matched accuracy; non-spectral in the full-recipe leg, with a fit
   confound in the matched leg.
7. **Atlas-specific statistics of known kinds.** LH-1/LH-2 (median kNN log-radius shift with a HOLD band) work between
   splits but were never compared with the head; ST-3/ST-4 (class-span decompositions for drift) were ruled out; ST-6
   (topology metrics as TTA monitors) never ran.

## 10. Maintenance

- Update a row when a verdict lands, and bump "Last reviewed" in the files touched.
- Add rows for new information types before the plan that probes them is pre-registered.
- When a topic file's search gaps (listed in its final sections) are filled, replace "not found in prior art" with the
  finding and its citation, and add the new starting point to section 8.
- Keep the readiness cells and section 6 in step with `docs/reviews/EXTRACTION_PROPOSALS_2026-09-23.md` and
  [controller-functions-feasibility.md](controller-functions-feasibility.md).
