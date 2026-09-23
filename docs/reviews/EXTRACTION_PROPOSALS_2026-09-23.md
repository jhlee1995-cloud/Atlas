# Extraction proposals: how to read more controller-usable information out of the activation space

Draft 2026-09-23, revised twice the same day: after a skeptical review, and then to apply the owner's principles on prior art and functional value (section 8 lists every change). It is a companion to the evaluation of HEAD 343f651, `docs/reviews/EVAL_2026-09-23.md`, and to the knowledge base, `docs/knowledge/README.md` (the information inventory, functional readiness by controller function, and the adoption and refinement plan).

Status: proposals only. Nothing here is pre-registered, evidence or a promotion. The document is written against HEAD 343f651.
- **Owner's principles (binding).** (a) Prior art is never a reason to drop or down-rank a proposal; it is the starting point to adopt and refine toward Atlas's use. (b) Proposals are ranked by end functionality: whether the controller can act on the result (adapt / hold / escalate, F1, F2), how reliably (AUROC, false alarms, detection delay, cost), and whether it beats or complements the cheapest existing baseline (logit gap, max-softmax, a simple density). Items built on standard methods are marked **adopt and refine**.
- **Sources.** Three lenses produced 24 proposals: geometry-topology (GT1-GT8), signal-stream (SIG-1..SIG-8) and information-head (IH-1..IH-8). They were merged into 16 items, X1-X16; the merge map is in section 7. **The ids are stable, but after review the rank no longer follows the id.** The rank is in section 3 and at the top of each item.
- **Evaluation files.** The committed evaluation is `docs/reviews/EVAL_2026-09-23.md`. This file was drafted before it existed, so it cites the evaluation's working files, `synthesis_report.md` and `challenge.md`, which are uncommitted and live in the session scratchpad (`C:/Users/admin/AppData/Local/Temp/claude/C--Users-admin-Desktop-Atlas/88e61e46-9502-47b6-a0b6-463e4e388a17/scratchpad/evalprog/`). They are cited by file name and line; where EVAL and the working files differ, EVAL governs.
- **Numbers.** Numbers from resnet56 s1/s2 are INFO: those seeds are spent for every AX axis (results/anomaly_h1/SESSION.md:443-445). Discovery numbers are quoted from the hubs where possible.
- **Knowledge base.** `docs/knowledge/controller-functions-feasibility.md` is cited as "KB §n". The index, inventory and prior-art topic files are under `docs/knowledge/` (README.md). Prior art is listed only where the paper can be named confidently; doubtful citations from the lens outputs were dropped.
- **Recomputations.** X9's spectral-gap numbers were recomputed from committed atlas.json files and re-checked in review, together with GT8's margin-lead correlation. The probe.json values quoted in section 1 were re-read in review (one correction). Every other number is quoted from a cited file.

## 1. The question and where the evidence stands

The owner's question is which controller-usable information can be read out of a backbone's internal activation geometry, how reliably, and what that geometry adds beyond the output head. The target controller is label-free and decides adapt / hold / escalate. It should also do two things:
- **F1:** flag suspicious sensor input pre-emptively;
- **F2:** feed only the input that is needed (information-gated feeding).

Four facts from the record shape every proposal below.

1. **At a collapsed penultimate layer, geometry is mostly the head.**
   - In the full-fit resnet56 nets, between-class variance is 93% of the penult's variance (B/T 0.930 / 0.931; results/anomaly_h1/SESSION.md:205).
   - Margin ≈ maxprob on resnet20 (ATLAS_STATUS.md:19). On resnet56, margin leads maxprob at matched confidence by +0.006 to +0.020 (E9, p ≤ 0.027; results/margin_b1_vitb16/SESSION.md:432). On CIFAR the logit gap, the stronger head baseline, was never computed.
   - Margin ≈ the logit gap on the ViTs over all errors (+0.0017 / +0.0038; synthesis_report.md:80-89).
   - ViT-B/16's head maps a 768-d penult into 1000 logits. If W has full column rank, which the X2 SVD checks, the head is injective on its input. At that tap geometry can then only be a better *statistic* of what the logits already hold. It cannot hold extra *information*. The ImageNet dumps store all 1000 logits in float16 (atlas/extract_imagenet.py:21-24, :465-466).
2. **The one signal beyond the head is a matched-confidence effect on the ViTs.**
   - V3b has margin ahead of the logit gap by +0.008 to +0.037, with p ≤ 0.047, in all four ViT runs (challenge.md:30-41).
   - E7 gives margin 0.68-0.71 against the logit gap's 0.60-0.64 within maxprob strata (results/margin_b1_vitb16/SESSION.md:430).
   - Nothing has been conditioned on the *full* head output, and "a joint (logit gap + margin) detector is untested" (challenge.md:153).
   - By fact 1 this is, at the ViT penult, a better *statistic* of the logits, not new information. It is still controller-relevant, because a detector uses statistics.
   - On CIFAR no geometric score was ever compared with max-logit, logsumexp energy or the logit gap. The dumps store only argmax and maxprob (atlas/extract_acts.py:188), although the logits can be recomputed (challenge.md:56-59).
3. **Information the head lacks does exist, but only at early taps, at batch level, and partly as pixel statistics.**
   - At batch 16, resnet56-hub stage-1-end AUCs at severity 1 are 1.00 for motion, 0.98 for pixelate, 1.00 for snow and 0.78 for defocus. The penult gives 0.49-0.54 on motion, pixelate and defocus, and 0.77 on snow (results/anomaly_probe_resnet56_s0hub_st3/probe.json `.AX2.auc`; corrected in review, where the draft said "0.49-0.54 on all four").
   - The random-init resnet56 reaches 0.94 on contrast and 0.84 on fog at the same tap, but only 0.50-0.75 on noise, blur, pixelate, jpeg and snow (results/anomaly_probe_resnet56_rand/probe.json).
   - The earliest-tap brightness signal is architectural. At s1 the random-init net gives 0.65-0.67 at stem, layer1.0 and stage-1 end, the same as the trained hub (0.67).
   - Per sample, the same mild shifts sit at chance: d1 and layer-2.8 density give 0.50-0.51 for brightness, contrast, defocus and fog at s1 (hub `.AX4.auroc`). The information exists only when many frames are integrated.
4. **Several instruments are biased or were never varied.**
   - The train-referenced density over-alarms on clean data at 3.3-3.7× the nominal rate at full fit and 2.4-2.7× in the 50-70-epoch nets, following a log-nc1 law (AH-1; results/anomaly_h1/SESSION.md:150).
   - Class-span statistics leak under class skew (single-class FPR 0.26-0.36 for class 4; :401).
   - **Naming hazard.** Atlas's `energy` score is the mean squared feature, (X**2).mean(1) (atlas/invariants/margin.py:118-122), not the logsumexp energy score of Liu et al. Every `auc_energy_*` number, including CIFAR's 0.750-0.795, is a feature-norm AUC.
   - GAP pooling was never varied (experiments/queue/atlas_v0_resnet20_cifar10.yaml:27).
   - No stream exists (synthesis_report.md:329).
   - Nothing measures whether adapting helps (synthesis_report.md:328).

## 2. Rationale: six principles for extracting more, and more useful, information

1. **Report the increment over the full output head, never an unconditional AUC.**
   - Every new score is judged by what it adds to the complete head summary: MSP, max-logit, logit gap, energy, entropy, DOCTOR, and the sorted logit vector.
   - Three measures: a cross-fitted joint ΔAUC, a conditional usable-information ΔI in bits, and the AUC within logit-gap deciles.
   - For early taps, also condition on the stored pixel factors. A signal that raw image statistics carry is not geometry (see the random-init nulls above; results/anomaly_h1/SESSION.md:385).
   - Standard feature-space detectors are adopted twice: as **baselines** that every Atlas score is reported beside, and, where they serve a controller function directly, as **components to refine** (X4, the calibrated multi-tap conformal flag for F1; X8, Trust Score, relative Mahalanobis, kNN purity and LID). The set: Trust Score, Mahalanobis and relative Mahalanobis, L2-normalised kNN, ViM, and conformal kNN p-values. A new Atlas score earns a controller role if it beats or complements the head *and* these.
   - This is X1. It is ranked first because without it no other item can answer the owner's question.
2. **Use held-out references and calibrated units.**
   - Calibrate on held-out clean rows, not on the training reference. The training reference is the source of the AH-1 over-alarm.
   - State thresholds as conformal ranks or p-values, which carry a false-alarm guarantee and port across backbones.
   - Centre class-conditional statistics on *held-out* class means. AX-1's test-mean leak came from global centring (results/anomaly_h1/SESSION.md:401).
   - A robot must recalibrate on its own clean logs.
3. **Use collapse-aware coordinates.**
   - Collapse is the programme's best synthesis: a hidden coordinate behind several level effects (synthesis_report.md:256-262).
   - Read nuisance and sensor information *before* the collapse, at the pre-collapse tap and earlier. Read semantics at the penult.
   - Split the penult by the head itself (row space vs null space), not by the class-mean span.
   - Estimate the collapse level label-free, from the reference covariance spectrum (X9), so the controller can configure itself on a new backbone.
4. **Go beyond GAP.**
   - The head reads a spatial average. Any within-image spatial statistic at the head's own input layer is invisible to it by construction.
   - Localised faults, which are the typical robot sensor fault, are diluted by their area before any Atlas invariant sees them.
   - Spatial std, patch-level density, second-order pooling and ViT patch tokens are the unexplored channels (X5).
5. **Add time.**
   - Mild shifts are invisible per sample and saturated at batch level (AUC 1.0 has no dynamic range; MASTER_SUMMARY.md:156-159).
   - The non-saturating currency is the change-information rate (nats per frame) and the detection delay at a fixed false-alarm budget (ARL0).
   - Streams also expose what still images cannot:
     - class-prior skew in correlated streams;
     - ramps;
     - frozen or duplicated frames;
     - whether adaptation is currently helping.
   - These are X3, X6, X11 and X14.
6. **Keep the discipline that made the record trustworthy.**
   - Contrast every early-tap claim with a random-init null and a pixel baseline.
   - Run discovery on hubs and discovery seeds, and confirm on fresh seeds and never-read rows:
     - CIFAR test rows 5000-9999 and CIFAR-10-C rows 2000-9999. Neither was ever extracted, because extraction takes `test[:n]` (atlas/extract_acts.py:271-274) and `n_per_set` 2000. CIFAR-10-C rows 2000-4999 pair with clean rows that were read as halves A and B. So fully unread *pairs* exist only at rows 5000-9999.
     - The four CIFAR-10-C extra corruptions (speckle_noise, gaussian_blur, spatter, saturate). They are not wired into the loader (extract/data_loaders.py:23-27); check whether the copy on the volume ships them.
   - Treat s1/s2 as spent for anything AX-like.
   - Give every stream builder its own known-answer tests: a legacy builder bug once flipped a conclusion (docs/history/SEQUENCE_AXIS_REALDATA_RESULTS.md:35-52).

## 3. Ranked table

Cost tiers. $ figures use the observed rate of $0.74 per pod-hour (RTX 4090; docs/plans/STAGE2B.md:331). Tiers T1-T3 assume the RunPod volume is still alive (synthesis_report.md:356-362) and that a pod is attached to it, at the GPU rate unless that datacenter offers a CPU pod.
- **T0:** $0; committed JSON, run locally.
- **T1:** CPU on the existing dumps on the volume; minutes to about 1 pod-hour (≤ about $0.75).
- **T2:** cheap re-extraction of existing checkpoints, data generated in code, or short GPU runs; about 0.5-1.5 GPU-h (about $0.4-1.1).
- **T3:** new models, downloads, generated benchmarks or fresh training; 1-3 GPU-h (about $0.75-2.2), plus download and licence time.

**Ranking criteria (reworded in revision 2 to follow the owner's principles).** Rank is by expected **functional value to the controller per dollar**:
1. what the result lets the controller do (adapt / hold / escalate, F1, F2), or what it lets Atlas measure about that (X1);
2. how reliably, in functional units (AUROC or AURC, false-alarm rate at a stated budget, detection delay, latency and memory);
3. whether it can beat or complement the cheapest existing baseline for that function;
4. its value as a component or baseline that other items depend on;
5. cost and risk.

Whether a method is already published never lowers a rank: a published method is the starting point, marked **adopt and refine**. Evidence already in the record (for example AX-4's reversed ratio, or the head's injectivity at the ViT penult) does count, because it changes the expected result. Ids keep their original numbers.

| rank | id | title | information targeted | beyond output head? | cost | first experiment |
|---|---|---|---|---|---|---|
| 1 | X1 | Head-conditional scoreboard, plus the CIFAR head export | how much each geometric score adds, in AUC and bits, over the full logits, over standard detectors and over pixel statistics | it *is* the test; stops future arguments over unconditional AUCs | T1 | export 18 CIFAR fc heads (16 on the volume + 2 hub); known-answer tests; ΔAUC_joint / ΔI of margin, d1, log r10, T², H on r20/r56 hubs and ImageNet dumps, beside the X4/X8 detectors |
| 2 | X4 | Held-out, multi-tap conformal p-value profile (**adopt and refine**) | calibrated per-sample suspicious-input flag (F1), with a false-alarm rate that holds on a new backbone without retuning | a direct F1 component; beyond-head value is plausible for noise and texture shifts at early taps; for mild shifts the per-sample floor is physical (0.49-0.51) | T1 | discovery on the r20/r56 hubs; clean FPR at α = 0.05 as a known-answer check; power against the best head score at matched FPR; increment over the head (X1); p-values feed X3's martingale |
| 3 | X3 | Streams: per-tap change-information rate and ARL-calibrated detection delay | onset time and lead time of covariate change, at a fixed false-alarm budget | **yes** for learned-sensitivity corruptions (early taps ≫ penult at s1, with random-init near chance there); the head's rate is bounded by the penult's | T1 | seeded stream harness; MEWMA/T² per tap vs output-only *and pixel-factor* detectors at ARL0 2000; ramps s1→s3→s5; X15 as a scenario |
| 4 | X5 | Beyond GAP: spatial std, patch/position maps, second-order pooling, ViT tokens | localised faults (occlusion, soiling, glare, dead pixels) and *where* they are; region gating (F2) | **yes, structurally**: within-image statistics at the head's own input layer are invisible to a pooled head | T2 | re-extract with gap_std; store maps; paste paired CIFAR-10-C pixels into 6/12/25% squares; patch-max vs GAP d1 vs head vs pixel patch-outlier |
| 5 | X6 | Batch decision variables robust to label skew: covariate residual, shift typing, harm vs output-only estimators | covariate / class-prior / novelty typing; harm beyond ATC/DoC; "sensor changed, no harm" | yes for typing and skew robustness (output tests confound prior and covariate change); open for harm | T1 | BBSE-explained class-mixture residual per tap under single-class, Dirichlet and Markov skews; H vs ATC/DoC/AC |
| 6 | X11 | Monitors during adaptation, and adaptation benefit | whether adaptation is helping or hurting (adapt → stop/rollback), the one decision variable nothing in Atlas measures | vs stream pred_entropy / maxprob *and* the adapted model's outputs on the stored panel | T2 | wire both Tent manifests; per-step panel drift, participation ratio, frozen-copy disagreement; lead time before a 5-pt drop |
| 7 | X7 | Per-sample trajectory across depth (prediction depth, cross-tap agreement, margin path; X8's block-8 local scores) | when the decision formed and whether it was contested; early-exit data (F2) | plausible only in the last 2-3 blocks (error information is late, KB §2.4); on ViTs the only per-sample place where new information can exist without new extraction | T1 | PD / agreement from kNN and nearest-centre labels on ViT CLS block.0-11 and CIFAR taps; ΔAUC over head in gap deciles |
| 8 | X8 | Local neighbourhood geometry (Trust Score, relative Mahalanobis, kNN purity, LID, anti-hub) (**adopt and refine**) | confident errors in shallow valleys; label noise; geometry-vs-head disagreement as a "hold, output distrusted" trigger | at the head-injective ViT penult only a better statistic of the logits; possibly new information at block.8 (run in X7) and at CIFAR pre-collapse taps | T1 | published detectors on the ViT penult and CIFAR layer3.5 / penult, on held-out references; joint(logit gap + local score) vs joint(logit gap + centre margin); also scored on X1's shift and OOD targets |
| 9 | X2 | Head-null-space residual (successor to the ruled-out AX-4) | covariate and sensor state that the linear head cannot see | **invisible to the head by construction** (W·P_null = 0); whether it is *informative* is open, and AX-4's reversed ratio suggests little at the CIFAR penult; empty at the ViT-B/16 penult | T1 | SVD of the four head types; whitened s_null vs head on the 30 discovery splits; collapse law over the unprobed matched nets |
| 10 | X10 | Information-gated feeding: will more input change the decision? | minimum bits the controller needs; value of the next frame, view or region | vs a free pixel gate and cheap-pass confidence | T1 → T2 | feature-rate curve first; then cheap-view and pseudo-video arms; severity pairs only as a weak proxy |
| 11 | X9 | Label-free collapse coordinate (spectral gap λ_{K−1}/λ_K) | where to read, and whether penult geometry is just the head, on a new unlabelled backbone | no (configures the readout) | T0 → T1 | re-checked: Spearman +0.937 with clean over-alarm over 16 CIFAR nets, −0.923 with the margin − distance lead over 13; next, point predictions for ResNet50 and new-recipe nets |
| 12 | X12 | Headless and self-supervised encoders (DINOv2, CLIP) with stand-in heads | which controller signals survive without a class head (the robot case) | geometry is primary; test against the probe's or zero-shot head's own confidence | T3 | DINOv2 + CLIP + same-session ResNet50 on the val halves, an ImageNet-C subset and ImageNet-V2; H vs probe harm |
| 13 | X13 | Local perturbation response (TTA consistency, feature instability, finite differences) | locally unstable inputs, independent of any stored reference | open; decided at matched confidence; K extra passes per frame | T2 | K+m extra forwards per image; S_pre in gap deciles on 30 discovery splits |
| 14 | X14 | Sensor-fault typing ("sensor vs world"); its fault generator is shared with X5(c) and X3 | "the sensor changed, not the world" (hold, output distrusted); fault family | the head cannot see fault type, but pixel checks may suffice; frozen frames need no geometry | T2-T3 | fault-vs-world typing, leave-one-family-out, random-init null *and* pixel-check baseline |
| 15 | X16 | Representation disagreement across independently trained nets | hidden disagreement beyond output disagreement | conditional on the ensemble's outputs; N-fold compute on a robot | T1 | stitching residual across the 5 resnet20 nets; ΔAUC over the ensemble output summary |
| — | X15 | Gate-3 abruptness re-validated at a matched false-alarm rate | abrupt class-mix change (escalate trigger), with a false-alarm rate | output-level; geometry only via soft assignment | T1 | folded into X3 as a harness scenario; not ranked separately |

Ranking logic:
- **The top six cover the measuring stick, the one direct F1 component, and the four places where the head's information demonstrably runs out.**
  - X1 is the measuring stick.
  - X4 is the calibrated per-sample flag (F1). Its components are published (normalised kNN, conformal p-values, layer-wise p-value fusion), which is why it can be deployed and trusted: the false-alarm guarantee holds by construction under exchangeability, and the same code ports to any backbone. It fixes the one instrument defect the record documents (the train-referenced density over-alarms 3.3-3.7× at full fit; results/anomaly_h1/SESSION.md:150), uses the tap complementarity already seen on the hub, and supplies the p-values X3's martingale needs. It is also the bar every other per-sample score must beat.
  - X3 is time: mild shifts are at chance per sample and saturated per batch.
  - X5 is space: pooling was never varied, and within-image statistics are invisible to a pooled head.
  - X6 is batch composition: label skew vs covariate change, and the first comparison of H with free output-based harm estimators.
  - X11 is adaptation dynamics: the adapt/hold decision variable that nothing in Atlas measures (synthesis_report.md:328).
- **X7, X8 and X2 are cheap per-sample tests on data in hand.** Their expected gains over the head are modest, for evidence-based reasons:
  - error information is late (KB §2.4);
  - at the ViT penult the head is injective, so X8's detectors can only be better statistics of the logits there; the Trust Score gave little over confidence on CIFAR in its own paper (docs/knowledge/confidence-margin.md §4);
  - AX-4's reversed ratio says corrupted penult offsets move *along* the class span more than off it (results/anomaly_h1/SESSION.md:404).

  X8 ranks above X2 because its detectors are directly usable controller components (a geometry-vs-head disagreement trigger for "hold, output distrusted"; relative Mahalanobis and normalised kNN for novelty), while X2's channel may turn out to be empty.
- **X10** matters for F2, but the existing still images give only proxies for inter-frame change.
- **X9** is verified and nearly free, but it configures the readout rather than adding information. Held-out calibration (principle 2) removes half of its use.
- **X12** is the robot-relevant transfer and the main scope gap of the primary question. It ranks 12th on cost (T3: new models and data) and because it needs its own pre-registration; the committed evaluation places it in its breadth step, before the closed loop (docs/reviews/EVAL_2026-09-23.md §9, step 5), so it is the first T3 item to run once batch A's scoreboard exists.
- **X13, X14 and X16** have low expected gain per dollar or a high runtime cost. X15 lives inside X3.

## 4. Proposals by id (the rank after review is stated at the top of each item)

### X1. Head-conditional scoreboard, plus the CIFAR head export
**Rank after review: 1 (unchanged).** Sources: IH-1, and the scoreboard part of GT1. The evaluation's "sensor audit" (synthesis_report.md:524-533) is the same test; X1 is its per-sample and information-theoretic core.

**What.**
- **Head export.** Export the final Linear (W, b) of every CIFAR net into the repo (650 floats each), so every later analysis is head-aware without the volume:
  - the 16 checkpoints on the volume (14 trained plus 2 random-init; challenge.md:114);
  - the two chenyaofo hub heads. These are the main discovery units, but they are not volume checkpoints: they are fetched through torch.hub with no pinned ref (extract/backbone.py:23), so pin the ref and hash on export.
- **Logits.** Recompute logits = penult·Wᵀ + b.
- **Instrument check.** Argmax agreement with the stored preds ≥ 0.999, and |maxprob − stored| ≤ 1e-3. The penult is stored as float16 (atlas/extract_acts.py:176).
- **Head summary h.**
  - MSP, max-logit, top-1 minus top-2 logit gap, energy (logsumexp; not Atlas's `energy` field, which is the squared feature norm, atlas/invariants/margin.py:118-122), entropy, and DOCTOR 1 − Σp²;
  - the sorted logit vector: all 10 on CIFAR, and the top 10 on ImageNet. The ImageNet dumps store all 1000 logits in float16 (atlas/extract_imagenet.py:21-24, :465-466).
- **Standard detectors, from X4 and X8 (both also ranked in their own right as adopt-and-refine items).** Trust Score; class-conditional and relative Mahalanobis; L2-normalised kNN on a *held-out* reference; ViM; the conformal multi-tap p-value (X4). Each Atlas score is reported beside them, and X1's increment is also computed over h plus the best of them. X1 also reports X4's and X8's own increments over the head, since they are candidate controller components.
- **Measures, for every candidate score g and target Y:**
  1. ΔAUC_joint = AUC(p̂(Y | h, g)) − AUC(p̂(Y | h)): out-of-fold, cross-fitted with folds grouped by image row and, for shift targets, by corruption family, with a paired-bootstrap CI.
  2. ΔI_head = CE(Y | h) − CE(Y | h, g) in bits, under one predictive family frozen in advance (logistic regression on natural-spline bases).
  3. The AUC of g inside logit-gap deciles. This generalises the maxprob-only `_strat` at atlas/invariants/margin.py:223-248.
  4. AURC.
- **Pixel twin.** For early-tap scores, also report ΔI_pix, conditioned on h plus the 13 dumped pixel factors (factors/<split>.npz; atlas/extract_acts.py:189-190).
- **Targets:**
  - clean error;
  - corrupt-split error;
  - corrupted vs clean;
  - per-sample harm flip (clean-correct → corrupt-wrong on the paired row);
  - ood__cifar100 / ood__svhn vs clean;
  - batch loss > 10 pt.

**Why now.** Atlas reports three incompatible beyond-head quantities:
- the unconditional all-errors difference (synthesis_report.md:80-89);
- V3b (challenge.md:30-41);
- E7 within *maxprob* strata (results/margin_b1_vitb16/SESSION.md:430).

None of them conditions on the logit gap it is compared against, let alone on the full head. Other gaps:
- CIFAR has no logit-based baseline at all (atlas/extract_acts.py:188).
- The harm grade H was never compared with any output statistic (synthesis_report.md:316). Any penult shift statistic tracks cost as well as H does (results/anomaly_h1/SESSION.md:376-378).
- Early-tap batch signals are partly architectural: the random-init nets route N/B/L at 0.99-1.00 (:385).

**First experiment.**
1. **Known-answer tests first:**
   - a monotone transform of the logit gap gives ΔI ≤ 0.002 bits;
   - pure noise gives ΔI ≤ 0;
   - Y + Gaussian noise recovers the analytic ΔI.
2. **Discovery units:**
   - r20 hub and s1-s4 (_st3), r56 hub, e40;
   - ResNet50 (3 runs);
   - vitb16/deitb, which are discovery for new statistics (spent only for margin_typeb; results/margin_b1_vitb16/SESSION.md:533).
3. **Scores:** margin, d1, log r10, per-tap T² and H.
4. **Freeze** a HEAD-ADDITIVE rule, for example "ΔAUC_joint ≥ +0.01 and the 95% CI of ΔI_head excludes 0", together with per-unit predictions.
5. **Confirm** on:
   - the unspent augreg_in1k ViT;
   - two fresh resnet56 seeds on never-read rows.

**Risks.**
- **Power on CIFAR clean errors.** There are about 280-390 errors per 5,000 rows, and B1's paired SE was 0.0146 at n₊ = 356 (results/margin_b1_vitb16/SESSION.md:447). The corrupt splits and ImageNet's 25k rows must carry the weight.
- **Estimator dependence.** V-information depends on the predictive family, so freeze it.
- **By-construction zero.** A zero at the collapsed penult is expected by construction; it is a finding, not a failure.
- **Pixel side.** The decodability pools include the confirmation corruptions (atlas/invariants/decodability.py:30), so the pixel side must use discovery corruptions only.
- **Labels.** Labels are used offline only.

**Prior art.**
- Baselines: MSP (Hendrycks & Gimpel, ICLR 2017); MaxLogit (Hendrycks et al., ICML 2022); energy score (Liu et al., NeurIPS 2020); DOCTOR (Granese et al., NeurIPS 2021); Mahalanobis (Lee, Lee, Lee & Shin, NeurIPS 2018); ViM (Wang, Li, Feng & Zhang, CVPR 2022); kNN (Sun, Ming, Zhu & Li, ICML 2022).
- Evaluation: AURC and failure-detection evaluation (Jaeger et al., ICLR 2023); DeLong et al. 1988.
- Information beyond a baseline: usable V-information (Xu et al., ICLR 2020); conditional probing (Hewitt, Ethayarajh, Liang & Manning, EMNLP 2021).
- Trust Score (Jiang, Kim, Guan & Gupta, NeurIPS 2018), the uncited precedent for the centre margin (challenge.md:124).

### X2. The head-invisible channel: a head-null-space residual (successor to AX-4)
**Rank after review: 9 (draft 2; 7 after the first review; 9 after X4 and X8 were restored above it).** Sources: GT1 and IH-2.

**Review note: why it was demoted, and why it is kept.** The demotion rests on Atlas's own evidence, not on prior art.
- **Demoted.**
  - At the CIFAR penult this is a close cousin of AX-4, which was ruled out (results/anomaly_h1/SESSION.md:339-350). Under the NC3 alignment expected at full fit, span{w_k − w̄} ≈ span{c_k − c̄}, so the head's row space and the class-mean span nearly coincide.
  - AX-4's failure has a direction that argues against this item. e_perp's AUROC was below 0.5 on 30 of 30 splits while d1 rose (0.61-0.74; :404). So a corrupted sample's extra offset lies *more* along the class span, which the head sees, than off it. Even with the orientation flipped, the gains over d1 were −0.03 to −0.10.
  - "Provably beyond the head" was overstated. Invisibility is provable; informativeness is not.
- **Starting point (not a reason for the rank).** AX-4's ratio has the same form as NECO's score (an in-subspace norm over the full norm), and Mahalanobis-type scores already exploit directions the classifier does not use. They are the published starting points and baselines for this item.
  - At the ViT-B/16 penult the null space is empty. On ResNet50, where it is large (≥ 1048-d), Atlas has only clean ImageNet data, so only error detection can be tested there, and there MSP-type scores are hard to beat.
- **Kept.** Three things are new relative to AX-4:
  - an *absolute, whitened* norm around the held-out clean mean, instead of a ratio around the predicted-class centre. Mahalanobis-type scores draw their strength from directions the classifier does not use (Kamoi & Kobayashi 2020), and the legacy Mahalanobis variant helped exactly on fog and contrast, where d1 is weakest;
  - the head's own row space, which differs from the class-mean span wherever NC3 is loose;
  - the collapse law over the matched nets, whose dumps no probe has read.
- **Confirmation needs fresh seeds.** As an AX-4 revision it cannot be confirmed on s1/s2 (results/anomaly_h1/SESSION.md:443-445).

**What.**
- **The split.** Write the penult offset as z − μ_A = P_row(z − μ_A) + P_null(z − μ_A), where P_null projects onto null(W) and μ_A is the held-out clean mean (CIFAR test half A, rows 2000-3499).
- **Sizes of null(W):**
  - CIFAR (10×64): 54-d if W has rank 10. The softmax-invisible complement of span{w_k − w̄} is 55-d, because MSP also ignores the all-logits direction that energy and max-logit read.
  - ResNet50 (1000×2048): at least 1048-d.
  - ViT-B/16 (1000×768): empty if W has full column rank.
- **Per sample:** whitened norms s_null (around μ_A), s_null|ŷ (around the half-A mean of the predicted class, label-free) and s_row. Whitening uses the top r ≤ 20 half-A eigenpairs with the 1e-8·λ₁ floor of scripts/anomaly_probe.py.
- **Per batch:** the Hotelling T² of each part.
- **Information:** the usable information about corruption family, severity and the pixel factors (luminance, highfreq_ratio, anisotropy, noise_sigma), decoded from the logits alone, from z_null and from z_row.
- **Harm:** H recomputed inside each part.
- **Head vs class means:** the principal angles between span{c_k − c̄} and span{w_k − w̄}, the CIFAR analogue of head_center_cos.
- **Collapse law:** across nets with penult nc1 0.05-0.69, the rank correlation between collapse and ΔI(family; z_null | logits).
- **ViT-B/16:** report the singular spectrum of W·Σ_A^{1/2} and a graded "head-quiet" subspace. Its size r is frozen before any use.

**Why now.**
- **Every "class-orthogonal" statistic used the class-mean span, never the head:**
  - AX-1's T_perp and AX-4's e_perp (scripts/anomaly_probe.py:21-24, :42);
  - class_sub_frac (atlas/invariants/sensitivity.py:67).

  The two spans coincide only under NC3. On ImageNet, head_center_cos is only 0.56-0.78 (results/margin_b1_vitb16/SESSION.md:503), and on CIFAR it was never measured.
- **AX-4 tested a ratio only.** Its orientation reversed on 30 of 30 splits (results/anomaly_h1/SESSION.md:404). The absolute, whitened head-null norm was never measured.
- **Row 3.** highfreq and anisotropy survive to the penult (ATLAS_STATUS.md:12); whether they survive into the logits is unknown.
- **Legacy Mahalanobis.** The legacy Mahalanobis variant gained +0.11-0.14, but only on fog and contrast (MASTER_SUMMARY.md:47). That is where the hub's per-sample d1 and layer-2.8 density are weakest: fog s3 gives 0.550 and 0.476 (`.AX4.auroc`).

**Beyond the head.**
- **Invisible by construction.** Nothing in null(W) reaches the logits. So a positive ΔAUC_joint, or an AUROC above 0.5 inside logit-gap deciles, is information the head cannot supply. Whether any such information exists is the open, empirical question, and AX-4 makes the prior low at the CIFAR penult.
- **Structural corollary for ViT-B/16.** If W has full column rank, the head is injective on its 768-d input. So V3b-type gains there are better statistics of information the logits already hold. Genuinely new information must come from earlier blocks, the patch tokens or time (X3, X5, X7). This corollary is the most useful output of X2, and it costs one SVD.

**First experiment.**
1. SVD of W for resnet20, resnet56, ResNet50 and ViT-B/16: record the rank and null dimension, and verify that the ViT null space is 0.
2. **Discovery** on:
   - the r56 and r20 hubs, with AX-4's row design: calibration half A; clean half B, rows 3500-4999; corrupt rows 0-1999;
   - for the collapse law, the matched nets e50/e60/e70/s12m/s13m, whose dumps no AX probe has read (results/anomaly_h1/SESSION.md:74-83).
3. **Measures:**
   - the per-split AUROC of s_null, s_null|ŷ, s_row and d1, next to the head scores;
   - ΔAUC_joint (X1) with leave-one-family-out fusion.
4. **Freeze** a claim of the form "s_null adds ≥ +0.03 ΔAUC_joint on ≥ k of the 10 discovery corruptions at s3".
5. **Confirm** on:
   - two fresh resnet56 seeds;
   - never-read rows;
   - the four unwired extra corruptions as the family holdout.
6. **ResNet50:** clean errors and type-b only. Atlas has no ImageNet shift data, so the ResNet50 arm cannot test covariate information.

**Risks.**
- **Overlap with AX-4.** With 7% of the penult's variance off the class span and the corruption shift split roughly in proportion (AH-5(c); results/anomaly_h1/SESSION.md:205), s_null may add nothing over d1. AX-4's reversal makes this the expected outcome. That negative is still decisive: "at a collapsed penult the head-invisible channel is empty; read earlier."
- **float16 quantisation** in the small-variance directions. Keep the eigen floor, and cross-check on a small fp32 re-extraction.
- **Fusion can learn corruption identity.** Grouped CV and an unseen-family holdout are mandatory.
- **ResNet50's label-smoothed head** distorts MSP. Use the full-logit features.
- **Real but useless information** (for example JPEG level). Its value must be priced downstream (X6 routing, X11).

**Prior art.**
- ViM (Wang, Li, Feng & Zhang, CVPR 2022).
- Kamoi & Kobayashi, arXiv 2020 ("Why is the Mahalanobis distance effective for anomaly detection?": its strength comes from information not useful for classification).
- Lee, Lee, Lee & Shin, NeurIPS 2018.
- Relative Mahalanobis (Ren et al., 2021).
- NECO (Ammar et al., ICLR 2024). Its score is a subspace-norm ratio, the same form as AX-4's e_perp.
- Papyan, Han & Donoho, PNAS 2020 (NC3).
- INLP (Ravfogel et al., ACL 2020).

### X3. Streams: per-tap change-information rate and ARL-calibrated detection delay
**Rank after review: 3 (draft 3; 2 after the first review; 3 after X4 was restored above it, since X4 supplies its per-frame p-values).** Sources: SIG-1, GT6's conformal martingale, IH-3's ramp clause, X15 (a scenario), and X14's temporal faults. It is the CIFAR stage of KB §6.1's E-F1 stream plan, reduced to data already on the volume.

**What.**
- **Stream harness.** A seeded harness with its own known-answer tests, built from paired rows. Scenarios:
  - in-control: clean half-B rows in random order;
  - an abrupt switch to split S at τ = 500;
  - severity ramps s1 → s3 → s5 on the same rows;
  - an abrupt block;
  - a CIFAR-100 burst;
  - a label-skew burst (a single class, clean);
  - X15's class-mix spikes (f ∈ {2, 5, 10, 20, 40}%);
  - X14's temporal faults: frozen, duplicated and dropped frames. These are recorded as a known limit, since pixel hashes catch them (KB §4.1, hard limit 7);
  - recovery.
- **Taps:** stem, layer1.0, stage-1 end, stage-2 end, the pre-collapse tap (layer3.5 / 3.1) and the penult. Each is whitened by half A (top r = 10 and 20).
- **Measurements:**
  1. **Change-information rate** I_L(S) = ½‖μ_S‖² + ½[tr Σ_S − ln det Σ_S − r], in nats per frame. The covariance term is the Stein divergence already coded at scripts/anomaly_probe.py:234. Add a kNN-divergence check and bootstrap CIs. Compute the same rate for the head's outputs: the predicted histogram, maxprob and the logit gap from X1.
  2. **Detectors:** MEWMA (λ = 0.05), a sliding-window T² (w = 16, 32), and a conformal test martingale on X4's per-frame p-values, all with thresholds at ARL0 = 2000 clean frames.
  3. **Baselines, run through the same detectors:**
     - output-only: MSP / logit-gap CUSUM, a BBSD-style test on softmax vectors, and predicted-histogram GLR;
     - pixel-only: MEWMA on the 13 dumped pixel factors (factors/<split>.npz);
     - the random-init net's taps.

     An early-tap advantage counts only if it beats all three.
  4. **Delays:** the conditional expected delay per tap × split. On ramps, the lead time over the first frame at which the rolling 64-frame accuracy loss reaches 5 pt (labels offline only).
- **Pairing.** Every change alarm is paired with the harm grade H, because the decision is two-dimensional: change × harm.

**Why now.**
- **No stream exists** (synthesis_report.md:329). Controller use of AX-2b and AX-3 explicitly requires a ramp test (results/anomaly_h1/SESSION.md:480, :515).
- **Batch AUCs are saturated.** AX-2a's gate died on a 0.0017 tie (:403).
- **Per-sample detection of mild shifts is at chance** (section 1, fact 3).
- **The legacy streams are unusable.** They were unseeded, and one session was invalidated by a builder bug (docs/history/SEQUENCE_AXIS_REALDATA_RESULTS.md:35-52).

**Beyond the head.**
- The logits are a function of the penult, so the head's change-information rate cannot exceed the penult's (data-processing inequality). Early taps are not bounded that way, so their advantage must be shown empirically.
- On the hub it is likely large for the learned-sensitivity corruptions. At s1 (batch 16), for noise, blur, pixelate, snow and jpeg, stage-1 end gives 0.78-1.00 against the penult's 0.49-0.91, and the random-init net's stage-1 end gives 0.50-0.75.
- It is architectural, not learned, for contrast, fog and brightness: the random-init stage-1-end AUCs are 0.94, 0.84 and 0.65.
- So scope the claims to the learned corruptions, report the null's and the pixel-factor detector's delays alongside, and apply X1's pixel twin.

**First experiment.**
1. **Discovery** on the r20 hub, r20 s1-s4, the r56 hub and e40. Freeze the taps, r, λ, windows, ARL0 and 100 stream seeds per condition.
2. **Predictions:**
   - P1: I(stage-1 end) > I(penult) on ≥ 8 of the 10 discovery corruptions at s1, with the bootstrap CI excluding 0.
   - P2: the delay ratio (best early tap / best output-only detector) is ≤ 0.25 for motion, pixelate and snow at s1.
   - P3: the median ramp lead is ≥ 100 frames for motion and defocus.
   - P4 (instrument check): the predicted delay ln(ARL0)/I is within 2× of the simulated delay.
3. **Known-answer test:** Gaussian streams with known KL reproduce Lorden's delay.
4. **Confirm** on s12m/s13m (never read by an AX probe) or on fresh seeds.

**Risks.**
- **i.i.d. rows overstate the rate for correlated video.** The in-control model must be re-estimated on real clean video.
- **A change is not harm.** Brightness s1 fires at the stem. Hence the 2-D decision.
- **The Gaussian KL misstates heavy-tailed ReLU features.** The kNN check covers this.
- **Windows and ARL0 must be swept.**
- **s1/s2 numbers stay INFO.**

**Prior art.**
- Page 1954 (CUSUM); Lorden 1971 (delay of order log ARL0 / KL).
- Crosier 1988 (multivariate CUSUM); Lowry, Woodall, Champ & Rigdon 1992 (MEWMA); Basseville & Nikiforov 1993.
- Vovk, Nouretdinov & Gammerman, ICML 2003 (exchangeability martingales); Volkhonskiy et al. 2017 (inductive conformal martingales for change-point detection).
- Rabanser, Günnemann & Lipton, NeurIPS 2019 (Failing Loudly).
- Harmful-shift monitoring: Podkopaev & Ramdas, ICLR 2022; Amoukou et al., NeurIPS 2024 (sequential, without labels). Both are in KB §8.
- Wang, Kulkarni & Verdú 2009 (kNN divergence estimation).

### X4. Held-out, multi-tap conformal p-value profile
**Rank: 2. Adopt and refine.** (Draft rank 4; the first review moved it to "baseline instrument, not ranked" because its components are standard; revision 2 restored it by functional value, per the owner's principles.) Source: GT3.

**Review note (revision 2).**
- **Why it ranks second: it is a direct F1 component.** It delivers what the controller's F1 flag needs and nothing in Atlas has yet: a per-sample suspicious-input flag with a stated false-alarm rate that holds on a new backbone without retuning. The train-referenced density Atlas has now over-alarms 3.3-3.7× at full fit (results/anomaly_h1/SESSION.md:150).
- **Adopt.** Every component is published and is adopted as published: kNN distance on L2-normalised features (Sun et al.'s detector), conformal p-values (a clean false-alarm rate by construction under exchangeability), and layer-wise p-value fusion (Raghuram et al.'s framework). So "clean FPR 0.05 ± 0.012" below is a known-answer check of the implementation, not a finding.
- **Refine toward Atlas's use.**
  - Held-out calibration on clean rows never used for anything else, which removes the DO-3 over-alarm by design.
  - Tap choice and fusion informed by the atlas: the hub shows complementary taps per sample (layer-2.8 density 0.984 / 0.996 on gaussian s3/s5 but 0.453 / 0.465 on jpeg; penult d1 0.704 / 0.749 on jpeg; `.AX4.auroc`).
  - Pairing with the harm grade (X3, X6), so that a flag on a harmless shift (brightness) does not trigger action.
  - The per-sample argmin tap as a candidate router, whose value X6 and X14 would have to show.
- **Its functional tests** (not the FPR check): detection power against the best head score at a matched false-alarm rate, split by corruption family and severity; its increment over the head in X1; its delay when its p-values drive X3's conformal martingale; and FPR stability on a second backbone (ResNet50) with zero retuning.
- **Known limits of reach.** Mild shifts are at chance per sample (section 1, fact 3), so for them the flag's value comes through X3's integration over frames. Where it can win per sample (noise at s3/s5), the layer-2.8 density already scores 0.95-0.996 (`.AX4.auroc`); X4 adds the calibrated threshold and the fusion.
- **It is also the bar** every other Atlas per-sample score must beat inside X1.

**What.**
- **Score:** at each tap, s_l(x) is the distance to the 10th nearest neighbour among L2-normalised train-reference features.
- **Calibration:** convert it to a conformal p-value against clean rows A1 (2000-2749). Fuse the taps by Cauchy combination, then recalibrate the fused statistic on A2 (2750-3499) into one valid p-value.
- **Evaluation:** clean half B vs corrupt rows 0-1999 of every split, and ood__cifar100 / svhn.
- **Report:**
  - per-split AUROC;
  - the clean FPR at α = 0.05, which should be 0.05 ± 0.012;
  - the per-sample argmin tap, as a router;
  - the X1 increment over the head;
  - portability: the same code on every backbone, with zero retuning.

**Why now.**
- **The existing density over-alarms.** Train-referenced sparse_frac on clean test is 0.163 / 0.184 at full fit, 3.3-3.7× nominal, and follows the log-nc1 law (results/anomaly_h1/SESSION.md:150).
- **knn_density is Sun et al.'s detector without L2 normalisation, on a train reference** (atlas/invariants/density.py:18-43; synthesis_report.md:286).
- **Taps are complementary per sample.** On the hub:
  - layer-2.8 density gives 0.984 / 0.996 on gaussian s3/s5 but only 0.453 / 0.465 on jpeg;
  - penult d1 gives 0.704 / 0.749 on jpeg (`.AX4.auroc`).
- **A calibrated fusion is needed.** AX-2a's argmax-over-taps gate failed on a tie (results/anomaly_h1/SESSION.md:403), and fusion replaces that fragile argmax.

**Beyond the head.**
- Early-tap p-values read noise and texture statistics that the penult washes out (row 3, ATLAS_STATUS.md:12).
- Under noise the head is confidently wrong, so a win is plausible there.
- The mildest non-noise shifts (0.49-0.51 for every score at s1) are a physical per-sample floor, which X3 addresses with time.

**First experiment.**
1. **Discovery** on the r56 and r20 hubs, with AX-4's row design.
2. **Freeze** k, the taps and the combination rule.
3. **Pre-register** a claim such as: "fused beats the best head score by ≥ 0.02 on ≥ 60% of the noise, blur and pixelate splits at s3/s5, with clean FPR in [0.038, 0.062]".
4. **Confirm** on:
   - fresh seeds;
   - never-read rows;
   - the four extra corruptions;
   - ResNet50, where the FPR must stay nominal without retuning.

**Risks.**
- **Validity is marginal, and it assumes exchangeability.** A robot must calibrate on its own clean logs, and correlated frames break per-frame exchangeability (X3).
- **Dependent taps.** Use the Cauchy combination, not Fisher's method.
- **kNN latency.** Use a coreset or approximate nearest neighbours.
- **Selection risk.** Do not tune the taps on s1/s2.

**Prior art.**
- kNN OOD on normalised features (Sun, Ming, Zhu & Li, ICML 2022).
- Multi-layer Mahalanobis ensemble (Lee et al., NeurIPS 2018).
- Layer-wise statistics combined as p-values (Raghuram et al., ICML 2021).
- Conformal p-values for outlier testing (Bates, Candès, Lei, Romano & Sesia, Annals of Statistics 2023).
- Vovk, Gammerman & Shafer 2005.
- Cauchy combination test (Liu & Xie, JASA 2020).

### X5. Beyond GAP: spatial std, position maps, second-order pooling and ViT tokens
**Rank after review: 4 (draft 5; 3 after the first review; 4 after X4 was restored).** Sources: GT4, SIG-6, IH-4, and X14's photometric and optical fault generator.

**What.** Three arms, cheapest first.
- **(a) Config only.** Re-extract the r20 and r56 hubs with the already-implemented `gap_std` pooling (atlas/extract_acts.py:44, :88-93) into a new exp_id. Score the mean half, the std half and both through X1, with X4's held-out calibration.
- **(b) Spatial maps.**
  - Store the maps at r56 layer1.8, 2.8 and 3.5, and at r20 layer2.2 and 3.2.
  - **Size (corrected in review; the draft said "small").** In fp16, r56 maps take 32, 16 and 8 KB per image at layer1.8, 2.8 and 3.5, so all ~79k dumped images come to about 4.4 GB per net. Store maps only for a 5k reference subset, the 5k test rows and the arm-(c) fault splits (1,500 half-B images × about 16 conditions). That is about 2 GB per net, well within the volume.
  - Build a per-position Gaussian reference with shrinkage (PaDiM-style) and a coreset patch-kNN memory bank (PatchCore-style).
  - Image scores: the maximum, the top-5% mean, the anomalous area and the spatial Gini.
  - Add the deviation of the channel Gram (second-order) matrix.
- **(c) Localised faults, generated in code.**
  - For each half-B image, paste the paired CIFAR-10-C s5 pixels of the same row into a seeded square covering 6%, 12% or 25% of the image (noise, blur, pixelate, jpeg).
  - Add an occluder, glare or saturation, a dead-pixel cluster and a soiling blob. Hold one family out.
- **Metrics:**
  - AUROC as a function of area;
  - pointing game and IoU against the fault mask;
  - local-vs-global typing against fog, brightness and contrast;
  - the subset where the fault does not flip the argmax. This is the "sensor changed, output still right" case that the controller must hold on without distrusting the output;
  - X1 increments over the head *and* over a pixel patch-outlier baseline (local Laplacian variance, saturated-pixel fraction).
- **ImageNet resolution, early:** ResNet50 layers 1-3, and ViT patch tokens at blocks 3/6/9/11. Raw tokens take about 300 KB per image per block in fp16. So use a PCA-128 projection (about 50 KB) or 5k images per block.
- **Cost:** T2. The CIFAR arms need about 1-1.5 GPU-h, including fault generation and scoring. The ImageNet arm is another 1 h (batch C).

**Why now.**
- **Pooling was never varied.** Every manifest uses gap, and v0 marks gap_std "try later" (experiments/queue/atlas_v0_resnet20_cifar10.yaml:27).
- **Every Atlas shift is whole-image** (atlas/extract_acts.py:276-289). MASTER's "partial corruption caught down to 10% area" is batch-level, with no localisation (MASTER_SUMMARY.md:86).
- **Pooling matters at first order on ViTs.**
  - Type-b AUC is 0.792 with the CLS token against 0.580 with the patch mean on vitb16 (results/margin_b1_vitb16/SESSION.md:425).
  - The patch mean, which no head reads, depends on the training recipe: 0.56 on vitb16 against 0.74 on deitb (:400).
  - The ViT dumps keep only one CLS token per block, plus the patch mean (atlas/extract_imagenet.py:26-29).
- **AX-2b's "N" family behaves as a high-frequency template** (results/anomaly_h1/SESSION.md:386), and spatial std measures high-frequency energy directly.

**Beyond the head.**
- Structural: at the head's own input layer (r20 layer3.2 = penult under GAP), within-image statistics are invisible to the head.
- A fault of area a moves the pooled vector by about a times the local shift, while a patch maximum is not diluted.
- Prediction: at 6-12% area, the map-maximum AUROC is ≥ 0.90, while GAP d1 and MSP/energy stay ≤ 0.70.

**First experiment.**
1. Arms (a)-(c) on the r20 and r56 hubs, with the fault generator and seeds frozen.
2. **Confirm** on:
   - fresh seeds, with the held-out fault family and 25% area;
   - then ResNet50 and vitb16 on val half B.

**Risks.**
- **CIFAR maps are tiny** (8×8 at stage 3). Rest localisation claims on ImageNet.
- **Pixel statistics may detect synthetic faults easily.** X1's two-sided rule decides.
- **Circularity**, because we design the faults. Hence the held-out family.
- **Memory and latency on a robot.** Use a coreset or per-position Gaussians.
- **ViT high-norm artifact tokens** can dominate max-over-token scores.
- **Separate instrument.** This must be a separate instrument, never a rebuild of a frozen atlas.
- **Localisation is not harm.**

**Prior art.**
- PaDiM (Defard et al., 2021); PatchCore (Roth et al., CVPR 2022); MVTec AD (Bergmann et al., CVPR 2019).
- Gram-matrix OOD detection (Sastry & Oore, ICML 2020); bilinear pooling (Lin, RoyChowdhury & Maji, ICCV 2015).
- ViT registers and artifact tokens (Darcet, Oquab, Mairal & Bojanowski, ICLR 2024); DynamicViT (Rao et al., NeurIPS 2021).

### X6. Batch decision variables robust to label skew: covariate residual, shift typing, and harm against output-only estimators
**Rank after review: 5 (draft 6; 4 after the first review; 5 after X4 was restored).** Sources: SIG-2, GT6's measure/landscape split, and IH-3's harm and skew clauses. Its harm arm is the evaluation's next step 4 (synthesis_report.md:524-533).

**Review note: not a repeat of AX-1.**
- **What AX-1 did.** It projected out the class-mean span and whitened around the half-A mean. It was ruled out on two counts:
  - c3: the early-tap gain over the penult for motion s1 was +0.068;
  - the r20 hub's single-class FPR was 0.104, with the class-4 leak at 0.26-0.36 (results/anomaly_h1/SESSION.md:272-291, :401).
- **What R_L does differently.** It subtracts a BBSE-weighted mixture of *held-out* class means. Under pure label shift with unchanged class-conditionals, its expectation is zero by construction, which is the documented leak mechanism removed rather than re-thresholded.
- **What X6 does not claim.** It makes no early-tap-advantage claim (AX-1's c3).
- **Seeds.** It needs fresh confirmation seeds, never s1/s2.
- **Why it ranks high.** The H-vs-output-estimator arm is the cheapest decisive test in the plan.

**What.**
- **Batches:** sizes 16, 64 and 256, under these regimes:
  - i.i.d.;
  - single-class;
  - Dirichlet priors (α = 0.1, 1, 10);
  - Markov class runs (stay probability 0.9 and 0.99);
  - mixed corruptions.

  Cross them with the discovery corruptions at s1/s3 and the OOD sets.
- **(i) Covariate residual at tap L.**
  - Estimate the class prior: π̂ = C⁻¹q̂ (BBSE), projected onto the simplex, where C is the head's confusion matrix on half A (labels offline) and q̂ is the predicted-class histogram.
  - Residual: R_L = n·(x̄ − Σ_k π̂_k m_{L,k})ᵀ W_L⁻¹ (x̄ − Σ_k π̂_k m_{L,k}), with held-out class means m_{L,k} and the pooled within-class covariance W_L (top r).
  - Companions: a class-reweighted MMD, and GT6's label-free residual x − c^A_ŷ(x).
- **(ii) Label-shift magnitude:** ‖π̂ − π_A‖₁.
- **(iii) Harm.** H (AX-3; scripts/anomaly_probe.py:434-481), recalibrated on held-out clean data, against the output-only estimators:
  - average confidence (AC), DoC and ATC (threshold fit on half A);
  - mean logsumexp energy from X1's recomputed logits (not Atlas's feature-norm `energy`, whose HOLD rule already failed; AH-2(d));
  - BBSD.
- **Outputs:**
  - the FPR at the clean τ95 under skew alone;
  - the covariate AUC under each skew;
  - 3-way typing accuracy (covariate / class prior / novelty);
  - partial Spearman(H, loss | ATC, AC), and the within-split Spearman;
  - the share of "sensor changed, loss ≤ 2 pt" batches that the mid-tap T² flags and ATC does not.

**Why now.**
- **Every deep-tap batch statistic trips on class skew.** T_full at the penult has single-class FPR 1.0 (results/anomaly_h1/SESSION.md:280). AX-1's fix leaked class 4 (0.26-0.36, :401) and was ruled out.
- **MASTER calls class imbalance the structural blind spot** of point-wise axes (MASTER_SUMMARY.md:87-95). On a robot, nearly every window of a correlated stream is label-skewed.
- **AX-3 has not been tested where the controller needs it.**
  - It works only between splits (within-split Spearman 0.21-0.23; results/anomaly_h1/SESSION.md:388).
  - It was never compared with maxprob or ATC (synthesis_report.md:316).
- **HOLD has two triggers that must stay apart** (challenge.md:140).

**Beyond the head.**
- The head supplies q̂. Output two-sample tests confound a prior change with a covariate change, because covariate shift also moves the histogram. R_L reads the remainder from geometry.
- For harm, ATC or DoC may win outright. Geometry's role then narrows to three things:
  - sensor change without harm (mid-tap AUC ≈ 1 where the penult is ≈ 0.5);
  - robustness to skew;
  - delay (X3).

  Either outcome is decisive.

**First experiment.**
1. **Discovery** on r20 hub + s1-s4, the r56 hub and e40, using AX-3's seeded rows.
2. **Predictions:**
   - R_L's single-class FPR: maximum ≤ 0.15, mean ≤ 0.10;
   - FPR ≤ 0.10 at Dirichlet α = 0.1;
   - AUC ≥ 0.90 for motion, pixelate and snow at s1, batch 64, under α = 1;
   - typing accuracy ≥ 0.85;
   - control: a plain T² at the same tap gives single-class FPR ≥ 0.5;
   - partial Spearman(H, loss | ATC) ≥ 0.3;
   - on skewed clean batches, FPR ≤ 0.10 for H and ≥ 0.5 for BBSD / histogram entropy.
3. **Confirm** on fresh seeds.

**Risks.**
- **π̂ absorbs part of a covariate shift.** Compare it with a non-negative least-squares fit of x̄ on the class means at an earlier tap.
- **The label-shift assumption fails for within-class appearance shifts.** These then read as covariate, which is arguably correct.
- **BBSE is noisy at 1000 classes.**
- **A collapsed penult amplifies BBSE error.** The within-class covariance W_L there is tiny (B/T 0.93), so small errors in π̂ inflate R_L. Use the pre-collapse and stage-2 taps, with shrinkage.
- **Overlapping batches.** Half B has about 150 rows per class, so single-class batches of 64 overlap heavily. Report the effective n.
- **H may leak under single-class batches**, as AX-1 did.

**Prior art.**
- Prior adjustment (Saerens, Latinne & Decaestecker, 2002); BBSE (Lipton, Wang & Smola, ICML 2018); Failing Loudly (Rabanser et al., 2019).
- Kernel two-sample test (Gretton et al., JMLR 2012).
- ATC (Garg et al., ICLR 2022); DoC (Guillory et al., ICCV 2021); AutoEval (Deng & Zheng, CVPR 2021).
- Harmful-shift detection: Podkopaev & Ramdas, ICLR 2022; Amoukou et al., NeurIPS 2024 (KB §8).
- Test-time adaptation under temporally correlated streams: NOTE (Gong et al., NeurIPS 2022); LAME (Boudiaf et al., CVPR 2022).

### X7. Per-sample trajectory across depth
**Rank after review: 7 (draft 7; 6 after the first review; 7 after X4 was restored).** Sources: GT2, IH-5, and X8's block-8 arm (moved here in review).

**Review note.**
- **Error information is late** (KB §2.4). Class-centre margins predict the final errors at AUC 0.48-0.54 over roughly the first half of the blocks, in every model read. They pass 0.75 only in the last one to three blocks.
- **So X7's content is the last few taps:** ViT block.9-11, CIFAR layer3.x, ResNet50 layer4.x.
- **Why that matters on the ViTs.** The ViT penult is head-equivalent (section 1, fact 1). So these taps are the only per-sample place in the existing dumps where information beyond the head could exist.

**What.**
- **At every tap:**
  - the kNN label (k = 10, cosine, L2-normalised) and the nearest-centre label, with half-A centres as a check;
  - the normalised top-2 margin;
  - the conformal rank of the distance to the predicted-class centre.
- **Per-sample features:**
  - prediction depth PD: the first tap after which the label equals the head's argmax at every later tap;
  - the agreement count from the first class-structured tap;
  - the last disagreeing tap;
  - the signed area under the margin path;
  - a late-flip flag at the commit tap;
  - from X8: kNN label purity against the predicted class at k = 5-50, and the Trust Score, at ViT block.8-11 and CIFAR layer3.5. The test is joint(logit gap + local score) against joint(logit gap + centre margin).
- **Taps:**
  - CIFAR: 11 taps (r56 stride 5, r20 stride 1);
  - ResNet50: its block taps;
  - ViT: CLS at block.0-11 plus the penult.

**Why now.**
- **commit_layer exists only at dataset level** (atlas/invariants/flow.py:45-87; ATLAS_STATUS.md:13).
- **The pre-collapse tap is only loosely tied to the head.**
  - At layer3.5, Spearman(margin, maxprob) is 0.479 against 0.886 at the penult, and the nearest centre agrees with the head on only 0.842-0.856 of samples (results/anomaly_h1/SESSION.md:185).
  - Yet the layer3.5 margin still detects clean errors at AUC 0.796 (penult 0.926, maxprob 0.911; results/margin_v1_resnet56_s0hub/atlas.json `.per_layer.layer3.5.margin_typeb.auc_margin_wrong`).
- **ViT dumps already hold everything needed.** Class structure appears from block.7 (E1, results/margin_b1_vitb16/SESSION.md:424), and the dumps hold all 12 CLS taps plus the logits.

**Beyond the head.**
- Plausible: the head sees only the endpoint, and prediction depth tracks example difficulty, which is related to margin but not the same.
- Test in logit-gap deciles and by ΔAUC_joint.
- Also report early-exit precision against a confidence-thresholded early exit at equal compute (feeds X10).

**First experiment.**
1. **Discovery** on vitb16, deitb and ResNet50 (new statistics) and on the CIFAR hubs.
2. **Targets:**
   - all errors;
   - type-b at c* 0.5;
   - ReaL-wrong errors (real_ok is stored);
   - CIFAR harm flips.
3. **Freeze**, then **confirm** on augreg_in1k and fresh CIFAR seeds: "PD is HEAD-ADDITIVE on corrupt-split errors, and its ΔAUC_joint there is ≥ 2× its value on clean errors."

**Risks.**
- **PD may simply re-express confidence.**
- **Early taps are near chance** (CIFAR ≤ layer2.8 margin AUC ≤ 0.57; ViT ≤ block.6 about 0.51-0.53), so PD effectively varies over only 4-5 taps.
- **ImageNet kNN labels are noisy** with 25 reference images per class; check hubness.
- **Pre-final-norm CLS scales differ by block.**

**Prior art.**
- Prediction depth (Baldock, Maennel & Neyshabur, NeurIPS 2021).
- Deep k-Nearest Neighbors (Papernot & McDaniel, 2018); linear probes (Alain & Bengio, 2016).
- Shallow-Deep Networks (Kaya, Hong & Dumitras, ICML 2019); BranchyNet (Teerapittayanon, McDanel & Kung, 2016).
- Nearest class-centre separability in intermediate layers (Ben-Shaul & Dekel, 2022).

### X8. Local neighbourhood geometry instead of class centres
**Rank: 8. Adopt and refine.** (Draft rank 8; the first review moved it to "baseline, not ranked" because its detectors are standard; revision 2 restored it by functional value, per the owner's principles.) Source: GT5.

**Review note (revision 2).**
- **Adopt.** Trust Score, relative Mahalanobis, LID and kNN purity are established misclassification and OOD detectors, adopted as published. The Trust Score is the direct precedent for Atlas's centre margin (challenge.md:124) and the published form of the geometry-vs-head disagreement idea behind inventory row CM-11 (docs/knowledge/README.md §5).
- **Functional value.** These are directly usable controller components: a disagreement score between the classifier and a neighbourhood rule is a per-sample "hold, output distrusted" trigger, and relative Mahalanobis / normalised kNN are novelty flags for "escalate, do not adapt". They also serve as baselines inside X1.
- **Refine toward Atlas's use.** Held-out references and L2 normalisation (the DO-3 and Sun et al. lessons); the tap where the head and the geometry still disagree (ViT block.8-11, which runs inside X7; CIFAR layer3.5, where the nearest centre agrees with the head on only 0.84-0.86 of samples; results/anomaly_h1/SESSION.md:185); and a joint test against the logit gap, not an unconditional AUC.
- **Why it ranks 8th (evidence, not prior art).** At the ViT penult the head is injective, so these scores can only be better statistics of the logits there. On CIFAR-10 the Trust Score fell below the softmax response in ConfidNet's comparison (KB §3.1), and in its own paper it gave little or no improvement over confidence on CIFAR (docs/knowledge/confidence-margin.md §4). Its block-8 arm, the most likely place for new information, runs inside X7's kNN pass. Its "not recommended" list stays below.

**What.**
- **Where:** the ViT penult and block.8 (cosine; the labelled 25k reference half), and CIFAR layer3.5 and the penult.
- **Scores:**
  - kNN label purity with respect to the predicted class at k = 5, 10, 25 and 50 (a purity curve over scale);
  - Trust Score;
  - relative Mahalanobis (shared shrinkage covariance, minus the class-agnostic term);
  - per-sample local ID (MLE, k = 20);
  - reverse-kNN count (anti-hub score).

**Why now.**
- **Every Atlas error statistic is first-order and global:** distance to class centres (atlas/invariants/margin.py:118-122).
- **ImageNet centres are poor summaries.** Valleys are shallow (sep_ratio_ref 0.87-0.93, E3; results/margin_b1_vitb16/SESSION.md:426), and each centre is the mean of 25 images. Even so, the centre margin is the only beyond-head signal Atlas has (V3b, E7).
- **Hubness and ID exist only as dataset scalars** (atlas/invariants/landmarks.py:89; dimension.py:61), and global TwoNN is the spectrum's twin (AH-8; results/anomaly_h1/SESSION.md:256-270).
- **Label noise.** 42-46% of ViT type-b are ReaL-correct (E11, :434), so purity may flag label noise.

**Beyond the head.**
- At the ViT penult, only as a better statistic, because the head is injective there (X2).
- At block.8, possibly as new information.
- Test: joint(logit gap + local score) against joint(logit gap + centre margin).

**First experiment.**
1. **Discovery** on vitb16/deitb, with X7's targets.
2. **Freeze** one score and its k.
3. **Confirm** on augreg_in1k.
4. **CIFAR side:** does purity drop under corruption before confidence does?

**Risks.**
- **Known lineage.** MSP is hard to beat on in-distribution errors (Jaeger et al., ICLR 2023), so the prior for a large gain is modest.
- **Hubness.** Correct with CSLS or reverse kNN.
- **Local ID is noisy.**
- **No labelled reference on a robot.** Purity would need pseudo-labels (X12).
- **Not recommended here:** curvature and persistent homology at these sample sizes. H1 was unstable (atlas/invariants/adjacency.py:12-13), and Takens-PH was dropped (MASTER_SUMMARY.md:46).

**Prior art.**
- Trust Score (Jiang et al., NeurIPS 2018); Deep kNN (Papernot & McDaniel, 2018).
- Relative Mahalanobis (Ren et al., 2021).
- LID for adversarial detection (Ma et al., ICLR 2018); MLE intrinsic dimension (Levina & Bickel, NeurIPS 2004).
- Hubness (Radovanović, Nanopoulos & Ivanović, JMLR 2010).
- kNN OOD (Sun et al., ICML 2022).

### X9. A label-free collapse coordinate
**Rank after review: 11 (9 in the draft and after the first review; 11 after X4 and X8 were restored).** Source: GT8. The numbers below were verified in review. The rank stays low because the coordinate configures the readout rather than adding information.

**What.**
- **Per tap, from the held-out reference covariance spectrum:**
  - g = λ_{K−1}/λ_K;
  - for unknown K, K̂ = argmax_{i ≤ 50} λ_i/λ_{i+1};
  - where a head exists, the energy fraction in the head's row space.
- **Uses:**
  - predict the clean over-alarm of any train-referenced density;
  - predict the margin-vs-distance regime (AH-3 / AH-4);
  - choose the read-out tap;
  - express thresholds in held-out rank units.

**Why now.**
- **Collapse is the programme's best synthesis** (synthesis_report.md:256-262), but every law is written in label-based quantities (nc1, sep_ratio; results/anomaly_h1/SESSION.md:150, :187), which a robot cannot compute.
- **Recomputed for this review** from committed `.per_layer.penult.pca_spectrum.eig_top20`, with no labels:
  - over 16 CIFAR nets (r20 hub and s1-s4 _st3; r56 hub_st3, s1, s2, e10-e70, s12m, s13m), Spearman(λ9/λ10, penult clean sparse_frac) = +0.937, against −0.942 for label-based nc1;
  - within the 11 resnet56 nets it is +0.961.
- **Range of the gap:**
  - resnet20: 7.9-9.5;
  - resnet56 e10-e70: 1.8-14.9;
  - full fit: 21-24.
- **GT8's second number, re-checked in review.** Over the 13 nets with margin atlases (r20 _st3 ×5; r56 hub_st3, s1, s2, e50-e70, s12m, s13m), Spearman(λ9/λ10, penult margin − distance lead) is −0.923 for the all-errors lead and −0.929 for the type-b lead. Label-based nc1 gives +0.962.
- **Scope of the recomputation.**
  - The gap is essentially an nc1 proxy: Spearman(λ9/λ10, nc1) = −0.932 over the 16 nets.
  - `pca_spectrum` is computed on the *train* reference (atlas/invariants/dimension.py:18-22). So these are train-reference spectra, and the held-out-spectrum version proposed above is untested.
  - Once densities are calibrated on held-out clean data (principle 2), the first use (predicting the train-reference over-alarm) disappears. What remains is tap choice and the "penult = head" regime call.

**Beyond the head.** It is not a detector. It tells the controller when penult geometry is just the head (large gap: read confidence) and when an earlier tap carries different information. It also sets calibration without labels.

**First experiment.**
1. **Done ($0):** the exploratory check above. It is discovery only, because s1/s2 were read.
2. **Next (T1):** pre-register point predictions of clean sparse_frac and of the margin − distance lead for:
   - two resnet56 nets with a new recipe (label smoothing 0.1, or 100 epochs);
   - ResNet50 (λ999/λ1000 of its 2048-d reference covariance).
3. **Test K̂** on every CIFAR tap.

**Risks.**
- **One family, and a thin spread.** The resnet20 seeds span too small a range, and e40-e70 share one initialisation.
- **The gap proxies NC1/NC2.** It adds label-freeness, not a mechanism.
- **ViT-B/16 (768 < 1000) has no λ999/λ1000.** It needs a separately validated index, such as effective rank or spectral slope.

**Prior art.**
- Neural collapse (Papyan, Han & Donoho, PNAS 2020).
- Law of equi-separation (He & Su, PNAS 2023).
- Intrinsic-dimension profiles (Ansuini et al., NeurIPS 2019).
- Effective rank (Roy & Vetterli, 2007).

### X10. Information-gated feeding: will more input change the decision?
**Rank after review: 10 (draft 10; 8 after the first review; 10 after X4 and X8 were restored).** Sources: SIG-4, IH-6, and GT2's early-exit clause.

**Review note: narrowed.**
- **Arm (a) is only a weak proxy.** Clean → corrupt and severity pairs are not inter-frame changes, so its skip curves say little about a robot stream. It is kept as a cheap sanity arm.
- **Arm (b) is the Atlas-specific question.** How many bits of which tap does the controller need? It runs first.
- **Arm (d) is the first frame-like test.** Pseudo-video is the nearest thing to it.
- **Early exit (GT2's clause) has a low prior** on these backbones: a training-free exit at 8/9 of resnet20 agrees with the model on about 84% of inputs and saves about 11% (KB §4.3, F2-c).
- **The real test needs video** with natural temporal continuity (KB §6.3). None is on the volume.

**What.** Four arms, labelled as drafted; run order (b), (a), (d), (c).
- **(a) CPU, existing dumps.**
  - Per-tap paired deltas Δ_L = ‖W_L(z_L(x_now) − z_L(x_prev))‖, for clean row i → corrupt row i and for severity pairs s1 → s3 → s5 (paired by row; atlas/extract_acts.py:271-289).
  - Controls: identical pairs, and two different clean images.
  - Targets: argmax flip, and the change in maxprob.
  - Baselines: pixel L2 and the stored pixel-factor deltas.
  - Output: skip curves at ≤ 1% and ≤ 5% flip-miss, and the FLOPs needed to reach each tap.
- **(b) CPU, feature rate.** Truncate the penult and one mid tap to r = 2-64 PCs, or quantise them to b bits per dimension. Plot against bits per sample:
  - the H-loss Spearman;
  - the error ΔI;
  - the family-routing accuracy.
- **(c) GPU minutes, cheap views.**
  - Views: 2× and 4× down-up sampling (CIFAR), 112 / 64 px (ImageNet), 25-50% of ViT tokens.
  - Predict the full view's flip from the cheap view, using head_q against head_q + geometry_q (margin, log r10, s_null, PD).
  - Compare accuracy against the fraction of full views used, for random, confidence, geometry and oracle gating.
- **(d) Pseudo-video:** ±1-4 px translations and small crops or rotations.

**Why now.**
- **F2 is a named function with no Atlas measurement at all.**
  - density.py was designed as "the familiarity field the compute-reduction lever reads" (atlas/invariants/density.py:8-10).
  - Novelty-proportional compute is a named application (CLAUDE.md:22-23).
- **Paired deltas exist only as split means** (atlas/invariants/sensitivity.py:26-71), although row 5 shows that corruptions differ in coherence (ATLAS_STATUS.md:14).
- **The earliest tap fires on decision-irrelevant change** (brightness s1: 0.67 at stage-1 end, 0.50 at the penult), so the best gate is probably not the stem.
- **The penult is low-dimensional** (AH-8), which predicts that the feature-rate curve saturates near r ≈ K.

**Beyond the head.** The head exists only after the full pass. So the comparison is an early gate against a free pixel gate and against the confidence of a cheap pass.

**Predictions:**
- at 5% flip-miss, the best tap skips ≥ 1.5× as many frames as pixel L2;
- the best tap is stage-2 end or the pre-collapse tap;
- geometry gating needs ≥ 10% fewer full views than confidence gating at matched accuracy;
- the H-loss Spearman is kept at ≥ 0.95 of its full value at 10 PCs × 4 bits.

**First experiment.**
1. Arms (a) and (b) on the r20 hub and s1-s4, in batch A.
2. Arms (c) and (d) in batch B.
3. **Confirm** on s12m/s13m or fresh seeds.

**Risks.**
- **Severity pairs are not inter-frame changes.** Hence pseudo-video, then real video.
- **Flips are rare.** Report n and CIs.
- **A flip is not harm.**
- **Savings depend on the backbone's FLOP profile.**
- **Cached-output reuse needs a slow-drift policy** (X3).
- **"Nothing changed since the last frame"** may be the strongest gate, and still images cannot test it.

**Prior art.**
- BranchyNet (2016); Clockwork convnets (Shelhamer, Rakelly, Hoffman & Darrell, 2016).
- Skip-convolutions (Habibian et al., CVPR 2021); DeltaCNN (Parger et al., CVPR 2022).
- MSDNet (Huang et al., ICLR 2018); Glance and Focus (Wang et al., NeurIPS 2020).
- DynamicViT; Token Merging (Bolya et al., ICLR 2023).
- Dynamic-network survey (Han et al., TPAMI 2021).

### X11. Monitors during adaptation, and adaptation benefit
**Rank after review: 6 (draft 11; 5 after the first review; 6 after X4 was restored).** Sources: SIG-8, and IH-3's stage 2.

**Review note: why it was promoted into the top six.**
- **It is the "adapt" decision variable.** Adaptation benefit decides adapt against hold, and it is the one controller variable that no Atlas measurement touches (synthesis_report.md:328). It is also the evaluation's first real test of the end goal (synthesis_report.md:535-540).
- **It reads the space while the space moves.** The question is whether geometry (panel-anchor drift, participation ratio) warns of adaptation harm before outputs do. That is the atlas's original deformation use.
- **The cost is small.** The manifests are pre-registered and resnet20 Tent steps are cheap.
- **The one real risk is new code.** scripts/tta_deform.py has never been executed (synthesis_report.md:326), so budget one failed round trip.

**What.**
- **Wiring.** Wire experiments/queue/tta_tent_resnet20_{fog3,collapse}.yaml into pod_atlas.sh.
- **Per-step logging** (not per-checkpoint atlases):
  - panel-anchor drift of the 64 fixed panel images (atlas/extract_acts.py:266-270), at the pre-collapse tap and the penult, in within-class-radius units, plus argmax agreement with step 0;
  - the participation ratio of the incoming batch's penult covariance;
  - predicted-histogram entropy (scripts/tta_deform.py:136-140) and mean maxprob;
  - disagreement between the adapted model and a frozen copy;
  - H on a clean held-out probe (forgetting).
- **Outcomes** (offline): accuracy on held-out stream rows, on clean data and on the held-out corruption.
- **Targets:**
  - lead time before a 5-pt drop, at a false-alarm rate matched on benign runs;
  - the AUROC for the sign of the accuracy change since step 0;
  - the one-step Tent benefit, as a target for label-free predictors.

**Why now.**
- **Nothing measures whether adapting helps** (synthesis_report.md:328). The TTA manifests never ran and are not wired (:326).
- **The right question is already pre-registered.** The collapse manifest asks it, with an output baseline built in (experiments/queue/tta_tent_resnet20_collapse.yaml:34-37; scripts/tta_deform.py:15-20).
- **The existing ladder is too coarse for lead time.** It rebuilds full atlases at only 8-9 checkpoints (atlas/ladder.py:54-72).

**Beyond the head.**
- The head-side *stream* statistics cannot see clean-data forgetting, because the stream contains no clean data.
- But the controller can run the stored panel through the adapted model and read its *outputs*. So the output baseline must include:
  - panel argmax agreement with step 0;
  - panel mean maxprob and logit gap;
  - an AETTA-style dropout-disagreement estimate;
  - frozen-copy disagreement.

  A geometric alarm counts only if it leads these, not just pred_entropy.

**First experiment.**
1. **Discovery** on the r20 hub: 3 stream seeds × both manifests.
2. **Predictions:**
   - in the collapse run, panel drift at the pre-collapse tap alarms ≥ 10 steps before pred_entropy falls below 0.8× its step-0 value, in ≥ 4 of 5 runs;
   - in the standard run, the geometric alarm fires no later than pred_entropy at a matched false-alarm rate;
   - AUROC ≥ 0.8 for "accuracy below step 0";
   - the participation ratio falls before pred_entropy does.
3. **Confirm** on r20 s1-s4 or r56 s12m/s13m.

**Risks.**
- **The collapse inducer is a synthetic ceiling.**
- **The panel images are memorised train images**, so their drift may not mirror test forgetting.
- **The results are Tent-specific.**
- **A frozen copy doubles cost.** Run it on every k-th batch.
- **The standard dose must carry the claim**, because lead time is trivially positive at extreme learning rates.

**Prior art.**
- Tent (Wang, Shelhamer, Liu, Olshausen & Darrell, ICLR 2021).
- EATA (Niu et al., ICML 2022); SAR (Niu et al., ICLR 2023).
- RDumb (Press et al., NeurIPS 2023); AETTA (Lee et al., CVPR 2024).

### X12. Headless and self-supervised encoders
**Rank after review: 12 (draft 12; 10 after the first review, while X4 and X8 were out of the ranking; 12 after they were restored).**
- **Sources.** GT7 and IH-7.
- **Why it matters.** It is the robot-relevant transfer test and addresses the main scope gap of the primary question: no self-supervised or headless encoder has been read (docs/reviews/EVAL_2026-09-23.md §2.4).
- **Why it ranks 12th.**
  - It needs new weights, ImageNet-V2 and a generated ImageNet-C subset (T3).
  - It needs its own pre-registration (B2's lessons, results/margin_b1_vitb16/SESSION.md:597-602).
- **Sequencing.** The committed evaluation places self-supervised encoders in its breadth step, before the closed loop (docs/reviews/EVAL_2026-09-23.md §9, step 5); the synthesis had sequenced transfer after a closed-loop v0 (synthesis_report.md:554-557). X12 is therefore the first T3 item to run once batch A's scoreboard exists (batch C below).

**What.**
- **Encoders:** DINOv2 ViT-B/14 and CLIP ViT-B/16, next to vitb16 and a same-session ResNet50.
- **Data:**
  - the B1 val halves;
  - an ImageNet-C-style subset: 5 corruptions × s1/s3/s5 × 5k images, generated with pinned code;
  - ImageNet-V2.
- **Stand-in heads:** a linear probe fitted on the reference half, the CLIP zero-shot head, and a kNN classifier.
- **Label-free geometry:**
  - K-means pseudo-centres (K ∈ {100, 1000}), with ARI stability across seeds;
  - the cluster margin;
  - X4's conformal density;
  - H with held-out calibration;
  - X5's patch statistics;
  - X9's index;
  - X2's split, with W = the probe weights or the text embeddings.
- **Key readout:** does one geometric score predict harm to *all* stand-in heads at once?

**Why now.**
- **All evidence comes from supervised, collapsing classifiers.** Type-b is undefined without a head, while H needs only reference features (synthesis_report.md:336).
- **B2 is queued** with the lesson that the comparator must be defined per paradigm (results/margin_b1_vitb16/SESSION.md:597-602).
- **The INFO AH-4 reading** says the less collapsed the penult, the larger margin's lead over distance (Spearman(sep, lead) −0.70; results/anomaly_h1/SESSION.md:465-472). Self-supervised features do not neurally collapse.

**Beyond the head.** There is no native head, so geometry is primary. Geometry wins outright if one score predicts harm to every stand-in head at least as well as each head's own confidence.

**First experiment.**
1. **Discovery** on DINOv2 + ResNet50.
2. **Confirm** on CLIP and a second self-supervised encoder.
3. **Predictions:**
   - Spearman of batch-64 H with probe harm ≥ 0.9;
   - partial Spearman(H, harm | probe ATC) ≥ 0.2;
   - DINOv2's ΔI(log r10; probe error | probe head) ≥ 2× ResNet50's.

**Risks.**
- **Circularity:** the probe and H share the reference.
- **CLIP's temperature** changes every confidence baseline.
- **The mirror's JPEG re-encode is itself a shift** (results/margin_b1_vitb16/SESSION.md:451-463).
- **Artifact tokens.**
- **ImageNet is not egocentric robot data.**
- **A new pre-registration is needed**, with power sized from B1's SEs (:447).

**Prior art.**
- DINOv2 (Oquab et al., 2023); DINO (Caron et al., ICCV 2021); CLIP (Radford et al., ICML 2021).
- SSD (Sehwag, Chiang & Mittal, ICLR 2021); MCM (Ming et al., NeurIPS 2022).
- ImageNet-C (Hendrycks & Dietterich, ICLR 2019); ImageNet-V2 (Recht et al., ICML 2019).

### X13. Local perturbation response
**Rank after review: 13 (draft 13; 11 after the first review; 13 after X4 and X8 were restored).**
- **Source.** SIG-5.
- **Adopt and refine.** Its output-level half, argmax stability and KL under augmentation, is test-time-augmentation uncertainty (Ayhan & Berens 2018). It is adopted as published and doubles as the output-level baseline; the refinement is the feature-instability S_L at matched confidence.
- **Why it ranks low (function and cost, not prior art).**
  - It costs K extra passes per frame on a robot, which a controller can afford only on frames already gated by X3 or X10.
  - The expected gain over the logit gap at matched confidence is modest: static per-sample geometry has so far added little over the head (section 1, facts 1-2).

**What.**
- **Perturbations:** K = 8 label-preserving perturbations outside every CIFAR-10-C family (a horizontal flip, and ±1-2 px translations with reflect padding), plus m = 4 random pixel directions of norm ε.
- **Per-sample scores:**
  - argmax stability A(x);
  - mean KL J(x);
  - feature instability S_L(x), in within-class-radius units, at stage-1 end, the pre-collapse tap and the penult;
  - finite-difference sensitivity G_L(x).
- **Evaluation:**
  - error AUROC and AURC, within logit-gap deciles;
  - at batch level, the batch mean of A(x) as an accuracy estimate, against ATC and DoC.

**Why now.**
- **Static per-sample position looks exhausted.**
  - AX-4 was ruled out (results/anomaly_h1/SESSION.md:339-350).
  - Mild shifts are at chance per sample.
  - Margin ≈ maxprob on CIFAR.
- **No invariant reads local response.** sensitivity.py measures only split-level displacement.
- **The matched-confidence design already exists** (atlas/invariants/margin.py:223-248).

**Beyond the head.** Open. A(x) is output-level; the claim rests on S_L and G_L at matched confidence.

**First experiment.**
1. **Discovery** on r20 hub + s1-s4.
2. **Predictions:**
   - matched-confidence ΔAUROC(S_pre vs logit gap) ≥ +0.02 on ≥ 20 of 30 splits;
   - no loss on clean data;
   - the batch mean of A(x) estimates accuracy with MAE ≤ ATC's.
3. **Known-answer test:** a linear model gives G_L = ‖W‖ exactly.
4. **Confirm** on fresh seeds and on vitb16/deitb.

**Risks.**
- **Redundancy with confidence.**
- **Flip + crop training** may make flip and translation responses uninformative. Hence the random-direction arm.
- **Cost on a robot.** K extra passes per frame, so apply it only to frames gated by X3 or X10.

**Prior art.**
- ODIN (Liang, Li & Srikant, ICLR 2018).
- Test-time augmentation uncertainty (Ayhan & Berens, 2018).
- Sensitivity and generalization (Novak et al., ICLR 2018).
- Trust Score; DoC; ATC.

### X14. Sensor-fault signature library, including temporal faults
**Rank after review: 14 (draft 14; 12 after the first review; 14 after X4 and X8 were restored); partly merged.**
- **Source.** SIG-7.
- **What was merged.**
  - The photometric, optical and pixel fault generator is shared with X5(c).
  - Frozen, dropped and duplicated frames are scenarios in X3's harness.
- **What remains here: fault-vs-world typing and fault family.**
  - AX-2b's family routing was near-trivial: random-init nets route 3 of 4 families at 0.99-1.00 (results/anomaly_h1/SESSION.md:385).
  - Many of these faults are caught by simple pixel checks (KB §4.1, hard limit 7).
  - So the remainder must beat a pixel-check baseline as well as the random-init null.

**What.**
- **Faults generated in code:**
  - photometric: exposure step, gain ramp, white balance, channel dropout, clipping;
  - optical: soiling;
  - pixel: dead or hot pixels, banding;
  - transport: JPEG burst, frozen, dropped or duplicated frames, a torn frame;
  - flicker.
- **Signature of each fault:**
  - its onset profile I_L over taps (X3);
  - the whitened shift direction routed to templates, as in AX-2b;
  - temporal features: delta variance, spectral peak, exact repeats;
  - X5's spatial concentration.
- **Targets:** fault vs world change, and fault family. Evaluation is leave-one-family-out, against a random-init null.

**Why now.**
- **F1 needs "the sensor changed, not the world".** In the controller that means hold with the output distrusted, the second HOLD trigger (challenge.md:140).
- **CIFAR-10-C mixes world-like and sensor-like corruptions.**
- **AX-2b's routing is near-trivial**, with the nulls at 0.99-1.00 (results/anomaly_h1/SESSION.md:385).

**Beyond the head.** The head does not encode fault type. An honest caveat: frozen and duplicated frames are caught by pixel hashes, so geometry adds nothing there. The geometric value is in photometric and optical faults, and in the fault-vs-world typing.

**First experiment.**
1. **Discovery** on the r20 hub.
2. **Confirm** on s12m/s13m and ResNet50.
3. **Predictions:**
   - typing AUROC ≥ 0.90 at batch 16, leave-one-family-out;
   - trained nets beat the null by ≥ 0.10 on non-photometric faults.

**Risks.**
- **Circularity:** templates can learn the generator.
- **Real faults co-occur with world change.**
- **Stem responses differ across checkpoints.** Brightness's PC1 share runs 0.00-0.57 (results/anomaly_h1/SESSION.md:357), so refit the templates per net.

**Prior art.**
- Fault detection and isolation (Isermann, 2006).
- Basseville & Nikiforov 1993.
- The CIFAR-10-C / ImageNet-C taxonomy (Hendrycks & Dietterich, 2019).

### X15. Gate-3 abruptness re-validated at a matched false-alarm rate
**Rank after review: not ranked separately (was 15); folded into X3 as a harness scenario.**
- **Source.** SIG-3.
- **Why it is kept.** It is an output-level audit of a legacy claim, and it is worth running because MASTER still marks the claim ✅.
- **Why it is not ranked.** It is not an extraction proposal. Geometry enters only through soft weights and X6's covariate veto.

**What.**
- **Batches:** predicted-class histograms per batch of 64, from the CIFAR preds and the three logit-storing ImageNet dumps.
- **Scenarios:**
  - stable;
  - gradual prior drift;
  - a spike to f ∈ {2, 5, 10, 20, 40}%;
  - a spike together with a covariate onset.
- **Statistics:**
  - the legacy peak 2nd and 1st differences;
  - a multinomial GLR/CUSUM against a sliding reference (W = 20 batches);
  - a fast-minus-slow EWMA;
  - the same tests on soft nearest-centre weights.
- **Thresholds** at ARL0 = 1000 batches. Run it as a scenario inside X3's harness.

**Why now.**
- **MASTER marks "acceleration necessary" as verified** (MASTER_SUMMARY.md:106-118, :222-224), but it compared mean velocity with peak acceleration and reported no false-alarm rate (synthesis_report.md:331-335).
- **The rationale is backwards.** For i.i.d. batch noise the 2nd difference has variance 6σ², against 2σ² for the 1st. So "acceleration cancels noise" (MASTER_SUMMARY.md:115-118) holds only if it is shown at matched ARL0.

**Beyond the head.** This gate is output-level. Geometry could add two things:
- lower-variance soft class weights;
- X6's residual, as a veto on histogram changes caused by covariate shift.

If neither helps, Gate 3 needs no geometry. That is worth knowing.

**First experiment.**
1. **Discovery** on ResNet50 and the r20 hub.
2. **Predictions:**
   - CUSUM/GLR delay ≤ the peak-2nd-difference delay for f ≥ 10%;
   - the 2nd difference does not beat the 1st at a matched false-alarm rate;
   - the soft-weight delay is ≤ 0.8× the argmax delay at f = 5%.
3. **Declare the new use of vitb16/deitb.**

**Risks.**
- **Relevance depends on a policy map that does not exist yet.**
- **The ImageNet mirror is degraded.**

**Prior art.**
- Page 1954; Lorden 1971; Basseville & Nikiforov 1993.
- Lipton, Wang & Smola 2018; Rabanser et al. 2019.

### X16. Representation disagreement across independently trained nets
**Rank after review: 15 (draft 16; 13 after the first review; 15 after X4 and X8 were restored); optional.**
- **Source.** IH-8.
- **Adopt and refine.** Ensemble disagreement (deep ensembles; two-seed disagreement, GDE; agreement-on-the-line) is the starting point and the output-level baseline, adopted as published. The refinement is representation-level disagreement conditioned on the ensemble's outputs.
- **Why it is optional (function and cost).** A robot would pay N-fold compute for an ensemble, so the result is mainly offline knowledge about Atlas's nets rather than a deployable component.
- **What it would tell Atlas.** Whether its ten-plus same-row nets hide disagreement that the outputs do not show.

**What.**
- **Output-level disagreement,** per sample across the resnet20 hub and s1-s4 (rows paired by contract, atlas/extract_acts.py:271-274):
  - argmax disagreement;
  - BALD-form mutual information;
  - JS divergence.
- **Representation-level disagreement:**
  - the standardised ridge stitching residual between penult spaces (fitted on the train reference, on the top-r PCs), averaged over partner nets;
  - the per-sample agreement of relative representations. atlas/compare.py:188-206 computes this only as a dataset average.
- **Test:** ΔAUC_joint over the ensemble's output summary.

**Why now.**
- **Atlas owns more than 10 nets on identical rows but uses them only for dataset-level replication** (row 8, ATLAS_STATUS.md:18).
- **The CKA excess is likely collapse-driven** (results/atlas_v1_resnet56_s1/SESSION.md:699-715).

**Beyond the head.** An ensemble beating one head is known. The Atlas question is whether representations disagree beyond what the outputs show.

**First experiment.**
1. **Discovery** on the 5 resnet20 nets, with the hub's errors on the 30 discovery splits as the target.
2. **Predictor ladder:**
   - the hub's head;
   - plus the other nets' output disagreement;
   - plus the stitching residual.
3. **Confirm** on fresh resnet56 seeds.

**Risks.**
- **N-fold compute on a robot.** This is the main reason it ranks last among the ranked items.
- **The residual may be distance-to-reference in disguise.** Condition on log r10.
- **float16 noise in low-variance directions.**

**Prior art.**
- Deep ensembles (Lakshminarayanan, Pritzel & Blundell, NeurIPS 2017).
- Uncertainty under dataset shift (Ovadia et al., NeurIPS 2019).
- Disagreement predicts test error (Jiang et al., ICLR 2022); agreement-on-the-line (Baek et al., NeurIPS 2022).
- Model stitching (Bansal, Nakkiran & Barak, NeurIPS 2021; Lenc & Vedaldi, CVPR 2015).
- Relative representations (Moschella et al., ICLR 2023).
- CKA (Kornblith et al., ICML 2019).

## 5. Suggested sequencing into pod batches

**Cost basis (corrected in review).**
- **Rate.** Batch 3 took about 4.1 pod-hours for about $3.0 at $0.74 per hour on an RTX 4090 (challenge.md:113; docs/plans/STAGE2B.md:331). The draft's "$3.0-3.4, $0.74-0.83/h" carried the evaluation's arithmetic slip.
- **Training time.** resnet56 trains at about 3.8 s per epoch, so a 200-epoch seed takes about 13 min (results/atlas_v1_resnet56_s1/SESSION.md:138).
- **The binding costs are not compute.** They are pre-registration and review overhead, and the absence of a local Python lane: every harness bug costs a GPU round trip (synthesis_report.md:345-347).
- **Buffer.** B1's runtime estimate was far too low and ended in an exit-124 timeout (results/margin_b1_vitb16/RELAUNCH_r2.md). So the totals below carry a buffer.

**Batch 0: before any pod, $0.**
- Archive the volume first. The dumps and checkpoints exist only on kxfir1tryb (synthesis_report.md:356-362; challenge.md:214).
- Stand up the minimal CPU Python lane (challenge.md:215). Write the known-answer tests for:
  - the scoreboard (X1), the calibrated conformal flag (X4; its clean FPR at α = 0.05 is the known answer) and the local-neighbourhood detectors (X8);
  - the stream builder (X3);
  - the BBSE residual (X6): zero expected residual under pure label shift;
  - the head-null projector (X2).
- X9's recompute and its review re-check are done (section 4).
- Write one discovery-scope note that declares:
  - which dumps each batch-A item reads;
  - that the vitb16/deitb reads are new uses for new statistics;
  - that X2 and X6 are successors of AX-4 and AX-1, so s1/s2 cannot confirm them.

**Batch A: CPU on existing dumps, discovery plus frozen rules. About 5-6 pod-hours, about $4-4.5.** Use a CPU pod if the volume's datacenter offers one; the estimate assumes the GPU-pod rate.
- **X1 with X4 and X8** (about 1.5 h together, as in the first review's estimate): head export, scoreboard, the calibrated multi-tap conformal flag, and the local-neighbourhood detectors on CIFAR. X4 and X8 are scored both as baselines and as candidate controller components.
- **X3** (about 1 h), with X15 and X14's temporal faults as harness scenarios.
- **X6** (about 1 h).
- **X7** (about 0.5 h; 25k × 25k cosine kNN per ViT tap), including X8's block-8 scores.
- **X2** (about 0.5 h).
- **X9**: ResNet50 spectra (minutes).
- **X10** arms (b), then (a) (about 0.3 h).
- Optionally **X16** (about 0.3 h).

The deliverable is a commit of frozen rules and numeric predictions for every item, *before* batch B.

**Decision gates after A.**
- **If no geometric score is HEAD-ADDITIVE in X1** over the head summary plus the best standard detector, and X7/X2 add nothing in gap deciles, then per-sample penult geometry is the head. Put the remaining weight on time (X3), space (X5), batch composition (X6) and adaptation (X11).
- **If X4's fused flag holds its false-alarm rate on a second backbone and beats the best head score on noise, blur or pixelate at matched FPR,** it becomes the F1 component that X3 and closed-loop v0 build on. If it does not beat the head anywhere, keep it only as the calibrated bar and as X3's p-value source.
- **If X6 finds H no better than ATC/DoC/AC on harm,** drop H as a harm sensor. Keep geometry only for typing, skew robustness and delay.
- **If X2 finds information in the null channel,** add a pre-collapse graded split to batch B.

**Batch B: cheap extraction, adaptation runs, and confirmation. About 5.5-6.5 pod-hours, about $4-5.**
- **Fresh material:**
  - train two fresh resnet56 hub-recipe seeds (about 30 min);
  - extract never-read rows: CIFAR test 5000-9999 with their CIFAR-10-C pairs, and the four extra corruptions if present;
  - a small fp32 penult re-extraction for X2's quantisation check.
- **Reserve ViT:** extract augreg_in1k with all 13 taps and the logits (the B1c extraction; challenge.md:222). It confirms X1 and X7. B1b's four ViT dump-and-build runs took about 35 min in total.
- **New discovery:**
  - X5 arms (a)-(c) at CIFAR (about 1-1.5 h);
  - X11: wire Tent, add per-step logging, run 3 stream seeds × 2 manifests (about 1 h, plus one expected failed round trip);
  - X10 arms (d) and (c);
  - X13.
- **Confirmation:** run batch A's frozen rules for X1-X3 and X6-X7 on the fresh seeds and rows.

**Batch C: new data and encoders, plus confirmations from B. About 4.5-5.5 pod-hours, about $3.5-4.**
- **X12:** DINOv2, CLIP and a same-session ResNet50; a pinned ImageNet-C subset; ImageNet-V2. About 2-3 h, under its own pre-registration carrying B1's lessons.
- **X5 at ImageNet resolution:** ResNet50 maps and ViT patch tokens (about 1 h).
- **X14 remainder:** fault-vs-world typing (about 0.5 h).
- **Confirmation of X5, X11 and X13** on fresh seeds.

**Totals.**
- **Nominal.** About 15-18 pod-hours, about $11-13.5 at $0.74/h, over three batches, excluding review time and volume rent.
- **Budget.** Plan for about 1.5×, roughly $17-20, given the exit-124 precedent.
- **After C.** Real temporally correlated video is the next step. It is the only way to calibrate the in-control models of X3, X4 and X10 for a robot.

## 6. What this plan would give the controller, and what it deliberately leaves out

**Coverage of the controller's functions:**
- **F1 (flag suspicious input):**
  - X4 (rank 2, adopt and refine): the calibrated per-sample flag, with a false-alarm rate that ports across backbones; also the bar every other per-sample score must beat;
  - X3 (rank 3): sustained mild shifts, with delay and lead time;
  - X5 (rank 4): localised faults and where they are;
  - X6 (rank 5): skew-robust batch typing;
  - X7 (rank 7): per-sample error information late in depth;
  - X8 (rank 8, adopt and refine): geometry-vs-head disagreement and local novelty scores;
  - X2 (rank 9): head-invisible covariate state;
  - X14 (rank 14): sensor vs world.
- **F2 (information-gated feeding):**
  - X10 (rank 10): the bit budget and the value of the next input;
  - X5: region maps;
  - X7: early-exit data (a low prior on these backbones; KB §4.3).
- **Adapt / hold / escalate:**
  - X6: covariate vs prior vs novelty typing, and harm against ATC/DoC;
  - X11 (rank 6): whether adaptation helps, and when to stop;
  - X3: onset;
  - X8: "hold, output distrusted" from geometry-vs-head disagreement; novelty for "escalate, do not adapt";
  - X15 (inside X3): abrupt class-mix changes.
- **Portability:** X9 (rank 11, label-free configuration) and X12 (rank 12, headless encoders).

**Not proposed:**
- **Persistent homology or curvature at these sample sizes** (atlas/invariants/adjacency.py:12-13; MASTER_SUMMARY.md:46).
- **TwoNN as a runtime signal.** It is the spectrum's twin at the penult (AH-8).
- **The energy / DEVIATION hold rule.** It failed again (AH-2(d); AX-3(d)).
- **Re-thresholded AX-1 / AX-4 statistics on s1/s2** (spent; results/anomaly_h1/SESSION.md:443-445).
- **Any further off-span *ratio* statistic as a covariate-drift sensor at the CIFAR penult** (AX-4's form). AX-4's orientation reversed on 30 of 30 splits; this exclusion rests on that evidence. X2's absolute, head-null version is the only successor kept, and it ranks 9th. NECO itself stays in X1's detector set for semantic OOD.
- **Any rebuild of a frozen atlas with new pooling.** X5 is a separate instrument.

## 7. Merge map

| item | merged from | after review (revision 2) |
|---|---|---|
| X1 | IH-1; GT1 (scoreboard and targets) | rank 1; runs X4's and X8's detectors as baselines and reports their own increments |
| X2 | GT1 (head null space); IH-2 | rank 9 (demoted on AX-4's evidence); successor to AX-4 |
| X3 | SIG-1; GT6 (c) conformal martingale; IH-3 (ramp / CUSUM clause) | rank 3; hosts X15 and X14's temporal faults; uses X4's p-values |
| X4 | GT3 | rank 2 (restored); adopt and refine; the calibrated F1 flag and the bar inside X1; feeds X3 |
| X5 | GT4; SIG-6; IH-4 | rank 4; shares X14's spatial and photometric fault generator |
| X6 | SIG-2; GT6 (a, b) measure-vs-landscape residuals; IH-3 (harm vs ATC/DoC; skew false alarms) | rank 5 |
| X7 | GT2; IH-5 | rank 7; takes X8's block-8 arm |
| X8 | GT5 | rank 8 (restored); adopt and refine; also a baseline inside X1; block-8 arm runs in X7 |
| X9 | GT8 | rank 11 |
| X10 | SIG-4; IH-6; GT2 (early-exit clause) | rank 10; narrowed |
| X11 | SIG-8; IH-3 (stage 2, adaptation benefit) | rank 6 (promoted in the first review) |
| X12 | GT7; IH-7 | rank 12 |
| X13 | SIG-5 | rank 13 |
| X14 | SIG-7 | rank 14; partly merged into X5(c) and X3 |
| X15 | SIG-3 | scenario inside X3 |
| X16 | IH-8 | rank 15, optional |

## 8. Review log (2026-09-23)

### 8.1 Revision 2: the owner's principles

The owner set two binding principles after the first review (the header of this file; docs/knowledge/README.md §1): prior art is a starting point to adopt and refine, never a reason to drop or down-rank; and items are judged by end functionality (what the controller can act on, how reliably, and against the cheapest baseline), not by novelty. This revision applied them. Everything else from the first review (8.2) was kept, because it rests on evidence.

- **Restored into the ranking by functional value:**
  - X4 → rank 2, marked **adopt and refine**. It is a direct F1 component: a calibrated per-sample flag whose false-alarm rate holds by construction and ports across backbones, which is a functional strength, not a reason to exclude it. Its functional tests (power at matched FPR against the best head score; delay inside X3; FPR stability on ResNet50) were written into the item.
  - X8 → rank 8, marked **adopt and refine**. Trust Score and relative Mahalanobis are usable "hold, output distrusted" and novelty components. Its rank below X7 rests on evidence (the injective ViT head; the Trust Score's CIFAR results), not on its being published.
- **Ranking criteria reworded** (section 3): rank by functional value per dollar; a published method never lowers a rank; recorded evidence does count.
- **Wording that down-ranked for being standard or known was replaced** in principle 1 (section 2), X2 (the NECO similarity is now a starting point, not a demotion reason), X13 and X16 (now ranked on cost and expected gain, with their published halves adopted as baselines).
- **Ranks shifted by insertion only:** X3 2 → 3, X5 3 → 4, X6 4 → 5, X11 5 → 6, X7 6 → 7, X2 7 → 9, X10 8 → 10, X9 9 → 11, X12 10 → 12, X13 11 → 13, X14 12 → 14, X16 13 → 15. The first review's relative order among these items is unchanged.
- **X12's sequencing note** now follows the committed evaluation, which puts self-supervised encoders in its breadth step before the closed loop (docs/reviews/EVAL_2026-09-23.md §9, step 5).
- **Links.** The committed evaluation (`docs/reviews/EVAL_2026-09-23.md`) and the knowledge base index (`docs/knowledge/README.md`) are now cited in the header. The scratchpad file citations are kept, with EVAL governing where they differ.
- **Sequencing (section 5):** X4 and X8 run with X1 in batch A inside the same 1.5 h estimate; a decision gate for X4 was added.

### 8.2 First review

A skeptical review re-checked every repo citation in this file against HEAD 343f651 and the working tree, re-read the quoted probe and atlas values, and checked the paper citations. It changed the following.

**Ranking.**
- **Promoted:** X6 (6 → 4), X11 (11 → 5) and X5 (5 → 3).
- **Demoted:** X2 (2 → 7). It is AX-4's successor at the CIFAR penult, and AX-4's reversed ratio argues against it. "Provably beyond the head" was reworded to "invisible by construction".
- **Moved out of the ranking:** X4 and X8 became baselines, because their components are standard, X4's false-alarm guarantee holds by construction, and X8 is head-equivalent at the ViT penult. *Superseded by revision 2 (8.1): both were restored by functional value.*
- **Folded:** X15 into X3. X14 was partly merged into X5(c) and X3.
- **The top five** became X1, X3, X5, X6 and X11. *After revision 2 the top six are X1, X4, X3, X5, X6 and X11.*

**Factual corrections.**
- **Section 1, fact 3.** The penult's batch-16 AUC at s1 is 0.77 for snow, not within "0.49-0.54 on all four".
- **Section 1, fact 1.**
  - "Margin ≈ maxprob on CIFAR" is true for resnet20 only. resnet56 has E9's +0.006 to +0.020.
  - ViT injectivity is conditional on W's rank.
  - The ImageNet dumps store all 1000 logits, not a top 10.
- **Naming hazard.** Atlas's `energy` is the squared feature norm (atlas/invariants/margin.py:118-122), not the logsumexp energy score.
- **X1's head export.** It named "16 CIFAR checkpoints", which omits the two hub heads, the main discovery units. Those are torch.hub downloads (extract/backbone.py:23), not volume checkpoints.
- **X9.**
  - GT8's −0.923 was re-checked (−0.929 for the type-b lead; nc1 gives +0.962).
  - The spectra are train-reference spectra (atlas/invariants/dimension.py:18-22), not held-out ones.
- **X5.** The map storage "small" became about 32/16/8 KB per image per tap, and about 2 GB per net for the needed subset.
- **Costs.** $0.74/h, not $0.74-0.83. Batch 3 cost about $3.0. Totals were revised, with a failure buffer.
- **Never-read rows.** Fully unread clean/corrupt *pairs* exist only at rows 5000-9999.
- **Citations.**
  - scripts/anomaly_probe.py:23 → :21-24.
  - The dangling `docs/reviews/EVAL_2026-09-23.md` reference was replaced by the scratchpad paths. *Revision 2: EVAL is now in the tree and is cited in the header.*
  - The knowledge base is now cited.

**Added.**
- **X3:** pixel-factor and output-only detector baselines, plus label-skew and class-mix scenarios.
- **X6:** why it is not a repeat of AX-1, and a collapsed-penult risk.
- **X11:** a panel-output baseline, so geometry must lead the adapted model's own outputs on the stored panel.
- **Prior art:** Podkopaev & Ramdas (ICLR 2022) and Amoukou et al. (NeurIPS 2024), both from the KB. Kamoi & Kobayashi's finding was restated more precisely, and NECO's link to AX-4 was noted.

**Verified and kept.**
- Every other repo citation resolves to the claimed content.
- The paper citations were kept; none was judged doubtful. Raghuram et al. (ICML 2021), Kamoi & Kobayashi (arXiv 2020), Volkhonskiy et al. (COPA 2017), Ben-Shaul & Dekel (2022) and AETTA (CVPR 2024) were confirmed by search.
