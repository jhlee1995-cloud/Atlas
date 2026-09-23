# Controller functions F1 and F2: feasibility

Last reviewed: 2026-09-23; part of the Atlas knowledge base (docs/knowledge/README.md)

This note answers the owner's question about two possible functions of the label-free adapt / hold / escalate
controller: are they feasible?

- **F1, pre-emptive flagging.** When a sensor sends an input, flag it if it looks strongly suspicious, meaning likely
  to cause a model error, before the model's output is used.
- **F2, information-gated feeding.** Balance information volume against function, and feed the model only the
  information it needs (for example, changes), filtering out the rest.

Scope and conventions:
- Evidence is read at HEAD `343f651`. Paths are relative to the repo root.
- "Discovery-grade" means the number was not pre-registered, or it comes from spent seeds or a closed gate. It can
  motivate a test but it is not evidence.
- Numbers marked **(illustration)** are arithmetic under a stated assumption. They are not measurements.
- Every literature number below was taken from the linked paper, its abstract or its publisher record (section 8),
  and re-checked against that source on 2026-09-23.

---

## 0. Verdicts at a glance

| id | sub-variant | feasibility | main reason |
|---|---|---|---|
| F1-a | Per-sample "likely wrong" flag, after the forward pass and before action | **High** as a risk score or reject option. **Low** as a standalone alarm with high recall and few false alarms | The head's own confidence is the bar. Atlas matches it (CIFAR all-errors AUC 0.91-0.93, ImageNet 0.80-0.87) and adds only +0.008 to +0.037 over the logit gap at matched confidence, on two DeiT-lineage ViTs. An AUC of 0.8-0.9 means many false alarms at useful recall |
| F1-b | Per-sample sensor-fault or degradation flag | **Medium.** Gross signal faults: **high**, with simple input checks | Batch-level corruption detection is at ceiling in Atlas. Per-sample detection at the penult is weak (AUROC 0.61-0.74). Early taps hold the information, but the formal early-tap tests failed or stayed closed. No real sensor fault was ever tested |
| F1-c | Per-sample flag before the model runs (input only), predicting model error | **Low** for a label-free controller | In Atlas, label-free per-sample error information is near chance over the first half of the network and passes AUC 0.75 only in the last 1-3 blocks. Prior art that does this is supervised (failure labels or a reference sensor) |
| F1-d | Per-stream regime or harmful-shift flag | **High** for the method class. **Medium** for Atlas's own statistic H | Sequential detectors with calibrated false-alarm rates are mature, and one label-free harmful-shift detector exists. Atlas's H tracks harm between corruption types (Spearman 0.97), but only weakly within one (0.21-0.23). It has no stream test, no delay or false-alarm numbers, and no output-based baseline |
| F1-e | Failure prediction seconds ahead | **Low** now | Shown only with temporal data and supervision (simulators, logged disengagements). Atlas has no temporal data |
| F2-a | Frame gating: skip or reuse whole frames when little changed | **High** as a technique; **unknown** gain for the robot | Mature in video analytics and event-triggered control. The gain depends on the temporal redundancy of the robot's stream, which Atlas has never measured. It needs forced refresh and stuck-sensor checks |
| F2-b | Feature or token gating: recompute only the changed regions or tokens | **Medium** | Token reuse ran on a real robot arm (up to 1.7x CUDA latency over simulation and real-robot tests). CNN delta computation needs special kernels to turn FLOP savings into wall-clock savings |
| F2-c | Early exit (per-sample depth) | **Low** on Atlas's CIFAR ResNets and the ImageNet backbones read so far. **Medium** with trained side heads | Class information commits at block 8 of 9 (resnet20) and 24 of 27 (resnet56) (row 4). A training-free exit at 8/9 of the depth agrees with the model on only about 84% of CIFAR inputs, and saves about 11% of compute |
| F2-d | Gating adaptation compute (adapt only when needed), not inference | **Medium-high**; the most Atlas-native form | H has a working HOLD band, and EATA/SAR-style sample gating exists. But no closed-loop run exists in Atlas |

**In one paragraph.**
- Both ideas are feasible in some form, and both have substantial prior art, which is the starting point to adopt
  and refine (docs/knowledge/README.md §8).
- The strong forms are not supported by current evidence:
  - For F1, "catch the error before the model runs" and "a per-sample alarm that is rarely wrong".
  - For F2, "large savings from early exit on these backbones".
- The realistic forms are:
  - F1 as a graded risk score plus a stream-level harmful-shift detector, acting before actuation;
  - separate, cheap signal-integrity checks for sensor faults;
  - F2 as change-triggered frame or token reuse with forced refresh, and as gating of adaptation compute.
- Atlas's contribution would be the coupling: which tap carries which information, and a harm grade that keeps a
  change gate from treating harmless novelty as a trigger. It would not be the gating itself.

---

## 1. The two functions, stated precisely

### 1.1 F1: three different flags

"Suspicious" covers three different questions. Each needs different information, which lives in different places in
the network.

| flag | question it answers | natural unit |
|---|---|---|
| **error flag** | "this output is probably wrong" | per sample |
| **input-quality flag** | "this input is degraded or faulty (sensor fault, occlusion, exposure, noise)" | per sample or per window |
| **regime flag** | "the input distribution has changed, and the change is harmful" | per stream (window) |

"Pre-emptive" has three possible timings:

| timing | what runs before the flag | cost |
|---|---|---|
| **before the model** | only the raw input, or a separate small predictor | a separate predictor |
| **mid-network** | part of the backbone | partial forward pass |
| **before action** | the whole forward pass; the flag gates the use of the output | 2-5 ms monitors in the cited systems (section 3.1) |

A fourth, predictive meaning, "seconds before the failure", needs a stream.

Sub-variants used in this note:
- **F1-a** error flag, per sample, before action.
- **F1-b** input-quality flag, per sample, before action or mid-network.
- **F1-c** error flag before the model runs.
- **F1-d** regime flag, per stream.
- **F1-e** predictive flag, seconds ahead.

### 1.2 F2: what is gated

A pretrained frame-based backbone cannot take "only the changes" as its input. A difference image is out of
distribution for it. For a frozen backbone, "feed only what changed" therefore means one of two things:
- skip or reuse computation where the input did not change enough; or
- use a sensor and model built for change data (event cameras with models trained for them).

"Necessary" is always relative to a consumer. The task head needs class information; a sensor-fault monitor needs
nuisance information. Section 2.3 shows that the backbone keeps the first and discards much of the second.

| id | what is gated | typical signal |
|---|---|---|
| **F2-a** frame gating | whether a frame is processed at all, or its outputs are reused | pixel or feature difference, a key-frame schedule, a learned policy |
| **F2-b** feature or token gating | which regions, tokens or neurons are recomputed | per-region or per-token change, attention, thresholded deltas |
| **F2-c** early exit | how deep each input goes | confidence at a side head |
| **F2-d** adaptation gating | whether a test-time adaptation step runs on this batch | entropy, a harm grade, a change detector |

F2-d is not "feeding the model less". It is included because it is the form closest to the controller's
adapt / hold decision, and to the "novelty-proportional compute" application that CLAUDE.md:22-23 names.

---

## 2. What Atlas already has

### 2.1 Signals relevant to F1

"All errors" means every wrong test sample against every correct one. "Type-b" means wrong with maxprob above a cut.
All per-sample numbers are on clean test data unless stated otherwise.

| signal | where | what was measured | result | scope and status |
|---|---|---|---|---|
| margin (d2 − d1 to class centres) | penult | all-errors AUC (`.per_layer.penult.margin_typeb.auc_margin_wrong`) | resnet20 s1 / s2 0.912 / 0.919; resnet56 s1 / s2 0.928 / 0.925; vitb16 0.861; deitb 0.866; ResNet50 0.801 | row 9 ✅ (CIFAR resnet20, "margin ~ maxprob"; resnet56 d56 🟡); row 10 ✅ vitb16, deitb (B1b, c* 0.5) |
| margin, type-b | penult | AUC at cut 0.7 (CIFAR) or c* 0.5 (ViT) | resnet20 s1-s4 0.882-0.899 (resnet56 s1 / s2 0.908 / 0.904); ViT 0.792 / 0.807 | ATLAS_STATUS.md:19-20 |
| maxprob (head) | head | all-errors AUC | resnet20 0.912 / 0.920; vitb16 0.856; deitb 0.853; ResNet50 0.800 | same files |
| logit gap (head) | head | all-errors AUC | vitb16 0.859; deitb 0.862; ResNet50 0.849. **Not measured on CIFAR**; recomputable from penult · W + b | same files; challenge.md ERRORS 5 |
| energy (head) | head | all-errors AUC, not reoriented | CIFAR 0.750-0.795; ImageNet 0.23-0.30 | the orientation flips between datasets (challenge.md item 13) |
| margin at earlier taps | every tap | all-errors AUC of that tap's margin | near chance until the last blocks (section 2.4) | committed atlas.json files |
| nearest-centre distance d1, corrupt vs clean | penult | per-sample AUROC | 0.61-0.74 | INFO (results/anomaly_h1/SESSION.md:404) |
| off-simplex residual e_perp (AX-4) | penult | per-sample gain over d1 | −0.16 to −0.44 | ruled out (SESSION.md:339-350) |
| kNN density, train reference (AH-1) | penult | clean false-alarm rate at a 5% design rate | 0.163 / 0.184 (3.3x / 3.7x nominal); follows a log(nc1) law | SUPPORTED at depth 56; rule: calibrate on held-out, deployment-clean data (docs/plans/ANOMALY_H1.md:274-277) |
| harm grade H, per split (AH-2) | penult | Spearman(H, accuracy cost) over 30 corruption splits | 0.989 / 0.987; every split with H ≤ 0.25 costs ≤ 5.3 pt in s1 and s2 (≤ 5.6 pt in the secondary nets); brightness H 0.02-0.26 | SUPPORTED at depth 56; other penult shift readings do as well or better: displacement magnitude 0.9996 / 0.995, sparse_frac 0.994 / 0.992 (SESSION.md:375-378) |
| harm grade h, per batch of 256 (AX-3) | penult | pooled Spearman(h, loss); flag rule h > 0.25 | 0.973 / 0.972; batches losing > 10 pt flagged 3350/3351 and 3253/3254; non-harmful (≤ 10 pt) flagged 1099/2649 and 1198/2746; within-split Spearman 0.23 / 0.21 | SUPPORTED at depth 56, between splits only; batch 64 is INFO (ρ 0.93); recounted from `results/anomaly_probe_resnet56_s{1,2}/probe.json` `.AX3` |
| energy-based HOLD grade | penult norm ratio | batches with e ≥ 0.95 that lose > 10 pt | 21% / 33% | the energy rule fails (SESSION.md:331) |
| class-orthogonal drift T_perp (AX-1) | pre-collapse tap, batch 64 | AUC on 5 corruptions; false-positive rate on single-class clean batches | AUC 1.000 on both hubs. Single-class FPR 0.074-0.104 (one class reaches 0.26-0.36, a test-mean leak), against 1.0 for an unprojected penult statistic | ruled out on c3 and on the r20 FPR. Discovery-grade hint: an unprojected batch statistic fires on class skew |
| early-tap drift profile (AX-2a) | 6 taps, batch 16 | which tap is most drift-sensitive | on both hubs the penult was the best tap for none of the 8 corruptions; brightness is best seen at the earliest taps (stem, layer1.0; on the r56 hub the end of stage 1 is inside the tie band) | gate CLOSED on a 0.005 tie; discovery-grade (SESSION.md:293-305) |
| family router (AX-2b) | pre-collapse tap, batch 64 | routing corruption families | 1.000 | SUPPORTED, but random-init nets route 3 of 4 families at 0.99-1.00, so it is near-trivial (SESSION.md:383-386) |
| density acceleration (Gate 3, legacy) | predicted-class histogram | peak second difference | sudden stream 3.4x (CIFAR), class spikes 5-6x (ImageNet) | legacy, not re-run. No false-alarm rate or delay; mean velocity was compared against peak acceleration (MASTER_SUMMARY.md:106-146; synthesis_report.md §6) |

### 2.2 The output-head comparison

This is the controller-relevant core of rows 9 and 10.
- **CIFAR ResNets.**
  - resnet20: margin equals maxprob. Confidence-matched gap ≤ 0.002, Spearman 0.92-0.93 (row 9, E1).
  - resnet56: margin beats maxprob by +0.006 to +0.020 (E9, p ≤ 0.027) (ATLAS_STATUS.md:19; challenge.md ERRORS 3).
  - The logit gap, the stronger head baseline, was never computed on CIFAR.
- **ImageNet ViTs.**
  - On all errors, margin equals the logit gap: vitb16 +0.0017 (p 0.41), swap −0.0009; deitb +0.0038 (p 0.08),
    swap +0.0046 (p 0.038).
  - At matched confidence (V3b), margin leads the logit gap in all four ViT runs at cuts 0.5-0.8: vitb16 +0.008 to
    +0.017 (p ≤ 0.047); deitb +0.016 to +0.037 (p ≤ 2.9e-2) (`sweep[cut].margin_minus_logitgap_confmatched`).
  - A joint score (logit gap + margin) was never tested.
- **ImageNet ResNet50.** The logit gap beats margin: −0.0485 on all errors, −0.046 at c* 0.5.
- **Harm grade.** H was never compared with batch-mean maxprob, entropy or the logit gap.
  - Its pooled harm alignment is partly built in, since the penult is the classifier's input (SESSION.md:376-378).
  - The energy grade, an output-side statistic, fails as a HOLD rule.

**Reading.**
- For per-sample error flags, geometry has not been shown to beat the head, except for a small confidence-matched lead
  on two DeiT-lineage ViTs.
- For batch harm, Atlas has a working statistic but no evidence that it beats free output-based statistics.

### 2.3 Where information lives, by depth (relevant to F1-b, F1-c and F2)

**Linear-probe decodability** (row 3; `results/atlas_v1_resnet20_s1/atlas.json`, `.cross_layer.commit_layer.per_factor`):

| factor | peak tap | value at peak | value at penult |
|---|---|---|---|
| luminance | stem | 0.997 | 0.26 |
| noise level (noise_sigma) | layer2.0 | 0.955 | 0.67 |
| corruption type (excess) | layer2.0 | 0.556 | 0.272 |
| class (excess) | penult | 0.814 | 0.814 (commits at layer3.1) |

resnet56 s1 shows the same pattern: corruption type peaks at layer2.5 (0.561) and falls to 0.275; class commits at
layer3.5.

**Caveat:** the probe pools include the confirmation corruptions (synthesis_report.md §4 item 4), so these readings
support the seed axis only.

**Nearest-class-centre readout at each tap**, a training-free classifier using train-reference centres.
Accuracy / agreement with the model's own prediction (`.per_layer.<tap>.class_centers`):

| net | midpoint tap | late taps | last block and penult |
|---|---|---|---|
| resnet20 s1 | layer2.0: 0.42 / 0.43 | layer3.0 (7/9): 0.65 / 0.66; layer3.1 (8/9): 0.81 / 0.84 | 0.92 / 0.99 |
| resnet56 s1 | layer2.5: 0.47 / 0.48 | layer3.0 (19/27): 0.60 / 0.61; layer3.5 (24/27): 0.82 / 0.84 | 0.95 / 1.00 |

**Reading.** Sensor-quality information (luminance, noise, corruption type) is strongest early and is partly discarded
before the head. Class information, and the information that decides errors, forms late.

### 2.4 Error information is late (the hard limit for F1-c and for early exit)

All-errors AUC of the margin computed at each tap, predicting the final model's errors
(`.per_layer.<tap>.margin_typeb.auc_margin_wrong`, committed atlas.json files):

| model | about the first half of the blocks | about two-thirds | one to three blocks before the end | penult |
|---|---|---|---|---|
| resnet20 s1 (9 blocks) | 0.48-0.53 (stem to layer2.0, 4/9) | 0.57 (layer2.2, 6/9) | 0.65 (layer3.0, 7/9), 0.77 (layer3.1, 8/9) | 0.91 |
| resnet56 s1 (27 blocks) | 0.49-0.54 (stem to layer2.5, 15/27) | 0.56 (layer2.8, 18/27) | 0.61 (layer3.0, 19/27), 0.77 (layer3.5, 24/27) | 0.93 |
| ViT-B/16 (12 blocks; class token after block.k) | 0.50-0.53 (block.0-5, 6/12) | 0.61 (block.7, 8/12) | 0.74 (block.9, 10/12), 0.82 (block.10, 11/12) | 0.86 |
| DeiT-B (12 blocks) | 0.51-0.52 (block.0-5) | 0.57 (block.7) | 0.70 (block.9), 0.82 (block.10) | 0.87 |
| ResNet50 (16 blocks) | 0.50-0.53 (stem to layer3.0, 8/16) | 0.54 (layer3.3, 11/16) | 0.66 (layer4.0, 14/16), 0.74 (layer4.1, 15/16) | 0.80 |

The nearest-centre agreement with the model on the ViTs follows the same curve
(`.per_layer.<tap>.margin_typeb.nearest_center_agrees_with_model` in the margin_b1 files). vitb16 is at 0.41 at
block.7, 0.70 at block.9, 0.81 at block.10 and 0.89 at the penult. ResNet50 is at 0.31 at layer3.5 and 0.56 at layer4.0.

**Caveat.** These are label-free class-centre readouts. A probe trained with error labels at an early tap could do
better; Yatbaz et al. report up to 0.933 AUROC (middle-layer activations, NuScenes) among early, middle and
backbone-output activation inputs of a LiDAR 3D detector, a different task, trained with error labels.
No such supervised probe was tested in Atlas.

### 2.5 What Atlas does not have

- **No stream data.** Every Atlas input is an i.i.d. image: CIFAR-10 / CIFAR-10-C test rows or ImageNet val.
  - So there are no detection delays, false-alarm rates, temporal redundancy numbers or ramps.
  - AX-2b and AX-3 both require a stream (ramp) test before any controller use (SESSION.md:480, :515).
- **No real sensor faults.** Frozen frames, dead pixels, exposure clipping, lens occlusion and compression were never
  injected. The corruptions are CIFAR-10-C families.
- **No per-sample error test under shift.** A3's E3 (margins on corrupt splits) was never run
  (docs/plans/A3_MARGIN.md:97).
- **No CIFAR logits stored.** They are recomputable from the dumped penult activations and the fc weights on the pod
  (challenge.md ERRORS 5).
- **No latency or memory measurements** for any Atlas statistic. The kNN-to-reference cost is unknown.
- **No closed-loop run.** The TTA manifests `experiments/queue/tta_tent_resnet20_{fog3,collapse}.yaml` exist, but they
  were never run.
- **Narrow scope.** One CIFAR recipe with resnet20/56, and two DeiT-lineage ViT-B/16s plus ResNet50 on a re-encoded
  ImageNet mirror.
- **Spent seeds.** resnet56 s1 and s2 are spent for every AX axis and every revised AH item (SESSION.md:443-446).
  New confirmations need fresh seeds.

---

## 3. What prior work shows

### 3.1 F1: usage patterns

| pattern | examples | maturity | reported performance (as read) | known failure modes |
|---|---|---|---|---|
| Head confidence and reject option | MSP (Hendrycks & Gimpel 2017); selective classification (Chow 1970; Geifman & El-Yaniv 2017) | Standard baseline | Jaeger et al. 2023: the simple softmax-response baseline was the best method overall across failure sources. Geifman & El-Yaniv: 2% top-5 risk guaranteed at about 60% coverage on ImageNet, under i.i.d. data | Confident errors far from the data (Hein et al. 2019). Calibration degrades under shift (Ovadia et al. 2019). Popular calibration methods often worsen failure prediction (Zhu et al. 2022). Abstention can widen group gaps (Jones et al. 2021) |
| Feature-space error or OOD scores | Trust Score (Jiang et al. 2018), the precedent for Atlas's margin; Mahalanobis (Lee et al. 2018); energy (Liu et al. 2020); kNN (Sun et al. 2022) | Research; widely benchmarked | ConfidNet's Table 1 (VGG16): Trust Score 88.47 vs MCP 91.53 AUROC on CIFAR-10. OpenOOD v1.5: no single method wins; in the full-spectrum setting the near-OOD AUROC of most methods drops by > 10% on ImageNet-1K | Distances lose meaning in high dimensions; kNN costs runtime and memory; scores fire on harmless covariate shift |
| Learned failure predictors | ConfidNet (Corbière et al. 2019); DOCTOR (Granese et al. 2021); Yatbaz et al. 2024 (LiDAR detector monitor) | Research | ConfidNet: +0.6 AUROC points over MCP on CIFAR-10, and it needs labels. Yatbaz: AUROC up to 0.933 (middle-layer activations, NuScenes); the combined monitor runs in 1.95 ms on GPU | Need error labels; transfer to new shifts unclear |
| Activation-pattern runtime monitors | Cheng et al. 2019; Henzinger et al. 2020 | Research; one highway-pilot case study | Cheng: on GTSRB, 4.58% of inputs flagged, 54.5% of them misclassified. Ferreira et al. 2021: 3 monitors over 79 benchmark datasets were no better than a random monitor | Hand-tuned coarseness; over-alarm; judged by OOD-ness rather than by harm (Guerin et al. 2023) |
| Introspective perception (robotics) | Daftry et al. 2016 (MAV); IVOA (Rabiee & Biswas 2019); Richter & Roy 2017 | Real-robot demonstrations | Daftry: score > 0.5 triggers 3 s of emergency manoeuvres; about 1000 m on average without a crash with introspection | Labels from a stereo pipeline or a reference sensor; hand-designed fallback; novelty ≠ harm |
| Sensor-health detectors | SoilingNet (Uřičář et al. 2019); camera-failure taxonomy and injection (Secci & Ceccarelli 2020); ImageNet-C (Hendrycks & Dietterich 2019) | Supervised research detectors on automotive camera data | Six detectors and a driving agent misbehave on injected camera failures | A separate supervised head; covers only the fault types it was trained on |
| Conformal and temporal monitors | Cai & Koutsoukos 2020; CODiT (Kaur et al. 2023) | Research, tested in driving and braking simulations | Calibrated false-alarm rate by construction | Needs exchangeable calibration data; novelty, not harm |
| Stream and harmful-shift detection | CUSUM (Page 1954); conformal martingales (Volkhonskiy et al. 2017); BBSD (Rabanser et al. 2019); Podkopaev & Ramdas 2022; Amoukou et al. 2024; ATC (Garg et al. 2022); DoC (Guillory et al. 2021) | Mature theory; label-free harmful-shift detection is recent | Rabanser: large shifts detected better than chance with about 20 samples; medium and small shifts need orders of magnitude more. Amoukou: sequential harmful-shift detection without labels, via an error estimator | CUSUM is optimal only for known pre- and post-change models; label-free accuracy estimation is hard in general (Garg et al.) |
| Seconds-ahead prediction | SelfOracle (Stocco et al. 2020); Kuhn et al. 2020 | Simulator and logged driving data | 77% of misbehaviours predicted up to 6 s ahead (simulator); > 80% accuracy up to 7 s ahead (logged BMW disengagements) | Needs temporal failure data; the lead time is scenario-specific |
| Cross-module consistency | Antonante et al. 2021 | Simulator with a full driving stack | < 5 ms on one CPU core; guarantees on how many faults can be identified | Needs redundant modules |

**Lessons for F1:**
1. **Grade by harm, not by novelty.** OpenOOD's full-spectrum drop, Guerin et al.'s out-of-model-scope argument and
   Atlas's brightness result (AH-2(c)) say the same thing.
2. **Per sample, the head's confidence is the bar.** Learned or feature-space scores add little on average.
3. **Calibrate thresholds on held-out, deployment-like clean data, and re-check them under shift** (conformal
   monitors; AH-1; Ovadia et al.).
4. **Evaluate at system level.** Use safety gain, residual hazard and availability cost (Guerin et al. 2022); AURC
   and FPR at fixed recall per sample; delay and false-alarm rate per stream.
5. **"Before the model runs" has worked only with supervision.** For a label-free controller the realistic target is
   "before actuation".

### 3.2 F2: usage patterns

| what is gated | examples | maturity | reported trade-off (as read) | known failure modes |
|---|---|---|---|---|
| Depth (early exit, skipping) | BranchyNet; MSDNet; Shallow-Deep Networks; SkipNet; confidence cascades (Wang et al. 2022); CALM; risk-controlled exits (Jazbec et al. 2024) | Research | BranchyNet ResNet-110/CIFAR-10: 1.9x at 79.17 vs 80.70%. SDN: > 50% average cost reduction. Cascades: 5.4x over EfficientNet-B7 at equal accuracy | Early exits are overconfident on hard inputs (Meronen et al. 2024). Adversarial slowdown: efficacy −90-100%, latency 1.5-5x (Hong et al. 2021). Needs trained side heads |
| Tokens and space | DynamicViT; EViT; A-ViT; ToMe (training-free); VLA-Cache; VLA-Pruner | ToMe merged into a widely used Stable Diffusion UI; VLA-Cache run on a real robot | ToMe: 2x throughput at −0.2-0.3%. VLA-Cache: up to 1.7x CUDA latency, +15% control frequency, negligible success-rate loss | Semantic-only pruning can remove action-critical tokens (VLA-Pruner). Sparse compute often does not speed up GPUs (Han et al. survey; LASNet) |
| Time (reuse unchanged computation) | Delta networks and the EdgeDRNN accelerator; sigma-delta networks; Skip-Convolutions; DeltaCNN; Deep Feature Flow; clockwork convnets | Hardware exists for delta RNNs; the rest is research with code | DeltaCNN: up to 7x GPU speedup without error accumulation. Skip-Convolutions: 3-4x lower compute cost. DFF: 3.7x on segmentation (71.1 → 69.2 mIoU) and 5.0x on detection (73.9 → 73.1 mAP). Delta RNNs: 5.7-100x lower cost | Error accumulation unless designed out; needs periodic key frames; custom kernels |
| Which frames to process | AdaFrame; FrameExit; NoScope; Reducto (on-camera filtering) | Systems research | Reducto: 51-97% of frames filtered while meeting the accuracy target. NoScope: 265-15,500x real time, within 1-5% of the reference model accuracy | Tuned per scene or query; a filter misses what its difference detector cannot see |
| Change-driven sensing | Event cameras (Gallego et al. survey); asynchronous sparse CNNs (Messikommer et al.); Sony event sensors | Shipping hardware; real-robot obstacle dodging (Falanga et al. 2020) | Sony IMX636 "output the changed data only"; 3.5 ms total latency in the dodging demonstration | Needs new models and processing; silence is informative only if the sensor is known to be alive |
| Information budget | Information bottleneck (Tishby et al.); VIB; task-oriented edge inference (Shao et al. 2022); event-triggered and self-triggered control (Heemels et al. 2012); send-on-delta (Miskowicz 2006) | IB: vocabulary, not a validated design rule (Saxe et al. 2018). Event-triggered control: mature theory | Control theory gives stability and rate bounds. Khojasteh et al.: trigger timing carries state information, but it is cancelled once the delay reaches the inverse of the plant's entropy rate | IB compression is not general under ReLU (Saxe et al.) |
| Adaptation compute | EATA; SAR; event-triggered learning (Solowjow & Trimpe 2020) | Research | EATA adapts only on reliable, non-redundant samples, cutting backward passes | Entropy gates trust the model's own confidence |

**Lessons for F2:**
1. **FLOPs are not latency, and mean latency is not worst-case latency.**
   - A gate lowers average compute. The worst case stays that of the full model plus the gate, unless compute is
     capped, as Mixture-of-Depths caps it with a fixed per-block budget.
   - A real-time loop is sized for the worst case. So F2 mainly saves energy and throughput unless a cap is enforced.
2. **Most gates trust the signal that F1 distrusts** (model confidence, or raw change magnitude).
3. **What is "necessary" depends on the consumer.** Task-relevance gating removes the evidence a monitor needs.
4. **Every input-dependent gate is an attack surface for slowdown** (Hong et al. 2021).
5. **Streaming perception** (Li et al. 2020): latency itself costs accuracy, so saved compute has real value in a
   control loop.

---

## 4. Feasibility verdicts, with reasons and hard limits

### 4.1 Hard limits that apply across sub-variants

1. **Error-predictive information lives in late layers.**
   - In every model read, class-centre geometry predicts the final model's errors at AUC 0.48-0.54 over about the
     first half of the blocks, and 0.54-0.61 at two-thirds of the depth (section 2.4). It passes 0.75 only in the
     last one to three blocks.
   - Consequence: a label-free, pre-model or early-tap error flag has little to work with.
2. **An AUC of 0.8-0.9 means many false alarms at useful recall.**
   (illustration: equal-variance binormal scores, computed from the measured AUC and error rate)

   | case | AUC | error rate | recall 0.5 | recall 0.8 | recall 0.9 |
   |---|---|---|---|---|---|
   | CIFAR resnet20, maxprob | 0.912 | 7.7% | FPR 0.03, precision 0.60 | FPR 0.14, precision 0.32 | FPR 0.26, precision 0.22 |
   | ViT-B/16, margin | 0.861 | 21.2% | FPR 0.06, precision 0.68 | FPR 0.24, precision 0.47 | FPR 0.40, precision 0.38 |
   | ResNet50, logit gap | 0.849 | 22.5% | FPR 0.07, precision 0.67 | FPR 0.27, precision 0.46 | FPR 0.43, precision 0.38 |

   - At camera rates (for example, 30 fps is 108,000 frames an hour), a per-frame rule with FPR 0.14 fires constantly.
   - A per-sample flag is therefore useful as a graded input to a policy (slow down, re-look, fuse, defer). It is not
     useful as a hard stop.
   - Real score distributions are not binormal. The actual FPR at fixed recall must be measured (section 6.1).
3. **There is no stream or sensor-fault evidence yet.** Every detection result in Atlas is on i.i.d. images and
   CIFAR-10-C-style corruptions, and every harm result is between corruption types at batch 256.
4. **Class information commits late**, at layer3.1 of 9 blocks and layer3.5 of 27 (row 4). So early-exit savings are
   small on CIFAR ResNets (section 4.3, F2-c).
5. **Degraded is not harmful, and the backbone discards degradation evidence.**
   - Brightness is detected but costs little (AH-2(c); AX-3(c)).
   - Luminance and corruption-type information decays toward the penult (row 3; section 2.3).
6. **Thresholds do not transfer.**
   - The energy orientation flips between CIFAR and ImageNet.
   - The clean false-alarm rate of a density threshold is specific to each checkpoint (AH-1).
   - Any flag needs per-backbone calibration on deployment-clean data.
7. **Some faults are invisible to any per-frame model-internal signal.**
   - A frozen camera repeats a perfectly normal image.
   - A dead or disconnected sensor may produce a plausible frame or none.
   - These need temporal integrity checks: frame differencing, timestamps, heartbeats.
8. **For an encoder without a class head** (self-supervised or a robot policy backbone), "likely wrong" needs a
   task-specific error definition: detection miss, depth error, action error. Type-b and margin are undefined there;
   H needs only reference features.

### 4.2 F1 sub-variants

**F1-a. Per-sample error flag, before action.**
- **Verdict: High** as a graded risk score or reject option. **Low** as a standalone alarm with high recall and few
  false alarms.
- **Why feasible.**
  - The head's confidence gives AUC 0.91-0.92 on CIFAR and 0.80-0.86 on ImageNet (clean data).
  - It is free, and it costs nothing beyond the forward pass.
  - Selective prediction gives a principled coverage-risk trade-off.
- **Limits.**
  - Hard limit 2.
  - Confident errors are structurally hard (Hein et al. 2019).
  - Untested under shift in Atlas (E3 was never run), and the literature says calibration degrades there.
  - On CIFAR the geometric margin is a confidence proxy. On the two ViTs it adds +0.008 to +0.037 over the logit gap
    at matched confidence; that is untested as a joint score and untested on a non-DeiT ViT.
- **What would change the verdict.** A per-sample score that beats the logit gap in AURC under shift, on fresh seeds.

**F1-b. Per-sample input-quality (sensor-fault) flag.**
- **Verdict: Medium.** Gross signal faults: **High**, with classical checks outside the network.
- **Why feasible.**
  - Batch-level detection is at ceiling in Atlas: AX-1 AUC 1.000 at batch 64; the AX-2b router 1.000.
  - Early taps carry the relevant factors: luminance at stem 0.997; noise level peaks at layer2.0 (section 2.3).
  - Supervised sensor-health detectors exist for automotive cameras (SoilingNet).
- **Limits.**
  - Per sample at the penult, detection is weak (d1 AUROC 0.61-0.74), and the off-simplex residual failed.
  - The early-tap axes are ruled out or CLOSED:
    - AX-1 c3: the early-tap gain for motion was only +0.009 to +0.068;
    - AX-2a: the gate closed on a tie.
  - Easy family routing is not learned-specific (random nets route too).
  - Hard limits 5 and 7.
  - No real sensor fault was tested.
- **What would change the verdict.** Per-frame fault AUROC ≥ 0.95 for each injected fault type from some label-free
  source (raw input, early tap or penult), at a stated false-alarm rate, on fresh seeds (section 6.1).

**F1-c. Error flag before the model runs.**
- **Verdict: Low** for a label-free controller.
- **Why.** Hard limit 1. The prior art that does this (Daftry et al.; IVOA) trains an input-only failure predictor
  with labels from a reference pipeline or sensor.
- **Note.** A supervised predictor is possible if the robot has a reference sensor or logged failures. That is a
  different project with different assumptions.

**F1-d. Per-stream harmful-shift (regime) flag.**
- **Verdict: High** for the method class. **Medium** for Atlas's H as the statistic.
- **Why feasible.**
  - Sequential detectors with calibrated false-alarm rates are mature (CUSUM, conformal martingales).
  - Two-sample tests on softmax outputs detect large shifts with about 20 samples (Rabanser et al.).
  - Label-free harmful-shift detection exists (Amoukou et al.).
  - H tracks harm across corruption types (ρ 0.97). It holds brightness: h ≤ 0.30 in 92-97% of brightness batches
    pooled over severities, and in 76-90% at severity 5. It flags every CIFAR-100 batch (INFO).
- **Limits.**
  - H resolves harm between splits, not within one (within-split Spearman 0.21-0.23).
  - It flags 41-44% of batches that lose ≤ 10 pt; most of those lose 2-10 pt.
  - Other penult shift readings (displacement magnitude, sparse_frac) track harm as well or better, and H was never
    compared with output-based statistics.
  - It was measured only at batch 256 (batch 64 as INFO), with no ramp, no delay and no false-alarm rate.
  - An unprojected penult batch statistic also fires on class-skewed but correct batches (AX-1: single-class FPR 1.0).
    The label-skew case must be separated from sensor-driven shift.
  - The legacy Gate 3 density acceleration has no false-alarm rate or delay, and needs re-validation as a sequential
    test.
- **What would change the verdict.** H, or a projected variant, gives a shorter detection delay than output-based
  statistics at a matched false-alarm rate, on ramps and abrupt onsets, without alarming on benign brightness or on
  label skew.

**F1-e. Seconds-ahead failure prediction.**
- **Verdict: Low** now.
- **Why.** It needs temporal data containing failures, and the demonstrations use simulators or logged disengagements
  with learned models. Atlas has no temporal data.
- **Note.** It becomes testable only after the robot or driving replay step (synthesis_report.md next step 9).

### 4.3 F2 sub-variants

**F2-a. Frame gating.**
- **Verdict: High** as a technique. The gain for the robot is **unknown**.
- **Why feasible.** On-camera filtering, difference detectors, key-frame schedules and event-triggered sensing are
  mature, and some ship in products.
- **Limits.**
  - The gain equals the temporal redundancy of the robot's stream, which is unmeasured.
  - A raw change threshold fires on sampling noise. The legacy Gate 3 velocity was about 0.30 for stable, gradual and
    sudden streams alike (MASTER_SUMMARY.md:109-118).
  - A skipped frame is a frame no monitor saw, unless the monitor runs on every frame.
  - Stuck sensors produce zero change (hard limit 7), so gating needs a heartbeat and forced refresh.
- **Safe form.** Gate the task path only. Keep a cheap per-frame integrity check. Force a full recompute every N
  frames. Bypass the gate whenever F1 fires.

**F2-b. Feature or token gating.**
- **Verdict: Medium.**
- **Why feasible.** Training-free token merging and reuse give 1.7-2x (ToMe; VLA-Cache, which also ran on a real
  robot arm). On video, Skip-Convolutions cut compute 3-4x, and DeltaCNN reaches up to 7x GPU speedup with its own
  kernels.
- **Limits.**
  - FLOP savings from sparse updates often do not become GPU latency savings without custom kernels.
  - Errors can accumulate without key frames.
  - Relevance-based pruning can drop action-critical or fault-relevant regions (VLA-Pruner; row 3 washout).
  - Nothing is measured in Atlas.

**F2-c. Early exit.**
- **Verdict: Low** on Atlas's CIFAR ResNets and on the ImageNet backbones read so far. **Medium** with trained side
  heads on other architectures.
- **Why low here.**
  - CIFAR ResNet blocks cost about the same, because channels × spatial size is constant across stages. An exit after
    block k of n therefore saves about (n − k)/n of block compute.
  - A training-free nearest-centre exit agrees with the model on:
    - 0.84 of inputs at 8 of 9 blocks, saving about 11%;
    - 0.66 at 7 of 9, saving about 22% (resnet20 s1);
    - resnet56 is similar: 0.84 at 24 of 27, 0.61 at 19 of 27.
  - On ViT-B/16, agreement is 0.70 at block 10 of 12 and 0.81 at block 11 (section 2.4).
- **What could raise it.**
  - Per-sample confidence-gated exits with linear side heads trained on frozen features. The backbone stays unchanged,
    so the atlas stays valid. They could route easy inputs out early.
  - The cited multi-exit results are 1.9x (BranchyNet, ResNet-110) and a > 50% cut in average cost (SDN), both with
    exits trained for the purpose.
  - A cascade (a small net first, a large net only on low confidence) is a strong, simple alternative.
- **Limits.**
  - Early exits are overconfident on hard inputs (Meronen et al.), which is exactly where F1 wants the full model.
  - Exits are vulnerable to slowdown attacks.
  - The worst case is unchanged.

**F2-d. Adaptation gating.**
- **Verdict: Medium-high.** It is the most Atlas-native form.
- **Why feasible.**
  - H has a pre-registered HOLD band: every split with H ≤ 0.25 costs ≤ 5.3 pt in s1 and s2; brightness is held.
  - The energy alternative fails.
  - Sample-level gating of adaptation (EATA, SAR) and event-triggered re-learning exist.
- **Limits.**
  - Nothing measures whether adapting on a batch helps. That is the actual decision variable.
  - No closed-loop run exists.
  - H has limited within-type resolution.

---

## 5. How F1 and F2 fit the primary question and the controller gates

### 5.1 Information items F1 and F2 need

Atlas's primary question is which controller-usable information can be read from a backbone's internal geometry. F1
and F2 turn into these map entries:

| id | information item | needed by | where it lives (Atlas evidence) | status |
|---|---|---|---|---|
| I1 | per-sample error likelihood | F1-a, F2-c exit rule | head and penult; near chance before the last blocks (section 2.4) | measured on clean data (rows 9, 10); open: under shift, and beyond the logit gap |
| I2 | per-sample input quality or fault | F1-b | early taps and raw input (row 3; AX-2a hint) | not established per sample; the formal axes failed or closed |
| I3 | fault or corruption type | F1-b routing, MASTER P5 routing | early to mid taps; near-trivial (AX-2b) | batch-level only; not learned-specific |
| I4 | shift magnitude graded by harm | F1-d, F2-d | penult density (AH-2, AX-3) | between-type only; no stream |
| I5 | class-prior change (label skew) versus sensor shift | F1-d, Gate 3 | in-span versus orthogonal coordinates (AX-1 T_par / T_perp) | discovery-grade hint only |
| I6 | temporal redundancy of features per tap | F2-a, F2-b | unknown | unmeasured (no temporal data) |
| I7 | per-sample decision depth | F2-c | row 4 (population-level only) | per-sample distribution computable from existing dumps |
| I8 | adaptation benefit | F2-d, Gate 1 | none | unmeasured |

I2, I5, I6 and I7 are new map cells that the current atlas lacks. I1 and I4 exist but need the output-head baseline
and a shift or stream test before a controller can rely on them.

### 5.2 Placement in the gates (MASTER_SUMMARY.md:14-18)

- **F1-a → Gate 2 (act vs fall back, per decision).**
  - A per-sample risk score chooses between acting, acting cautiously and deferring.
  - Keep two separate HOLD meanings: "hold, output trusted" (harmless shift, do not adapt) and "hold, output
    distrusted" (challenge.md MISSING 9).
- **F1-b → a veto input to Gate 1**, plus Gate 3 when persistent.
  - Never adapt on faulted input.
  - A persistent fault escalates to fault handling or a human, not to adaptation.
- **F1-d → Gate 1 magnitude (adapt or hold) and Gate 3 (abrupt change → escalate).**
  - The class-skew case goes to Gate 3's policy layer, not to adaptation (MASTER_SUMMARY.md:97-104).
- **F2 → a compute and sampling layer under Gates 1-2.** This is CLAUDE.md's "novelty-proportional compute"
  application. It follows four composition rules:
  1. The monitor path is never gated. F1's integrity checks see every frame.
  2. An F1 alarm bypasses every F2 gate and forces a full forward pass.
  3. Forced refresh sets a maximum interval between full computations.
  4. Compute is capped for the worst case, and gate decisions are logged as a monitored signal.

---

## 6. Minimal pre-registrable experiment plans

**Common protocol.**
- **Seeds.** Fresh seeds only. s1/s2 are spent (challenge.md MISSING 6).
- **Order.** Definitions, thresholds and predictions are committed before any dump exists (CLAUDE.md rules 3, 7).
- **Logits.** Stored, or recomputed from penult · W + b.
- **Thresholds.** Set on a held-out clean calibration split. Conformal quantiles at nominal rates of 1% and 5%.
- **Baselines.** Every score is compared with MSP, max-logit, logit gap and energy.

### 6.1 E-F1: flagging on a stream with injected sensor faults and gradual drift

**Material.**
- Two fresh resnet20 seeds and, optionally, two fresh resnet56 seeds, with the existing CIFAR recipe.
- CIFAR-10 test rows split into calibration and stream rows; CIFAR-10-C for corruptions.
- An optional second stage: ViT-B/16 and ResNet50 on the ImageNet val mirror, with the same fault transforms applied
  on the pod.

**Stream.** 20 episodes of about 14,000 frames. The segment order is shuffled per episode with fixed seeds, and frames
are drawn i.i.d., with replacement, from the stream rows within a segment.

| segment | content | length (frames) | purpose |
|---|---|---|---|
| clean | calibration-disjoint test rows | 2000 + 1000 + 2000 | ARL₀ and false alarms |
| benign drift | brightness ramp, severity 1 → 5, then hold | 1000 + 500 | alarm must not fire (harm grade) |
| gradual harmful drift | fog ramp, severity 1 → 5 | 2000 | detection delay on slow drift (MASTER P3's weak case) |
| abrupt harmful shift | gaussian noise s3 | 500 | delay on an abrupt onset |
| label-skew burst | a single class, clean images | 500 | must be separated from sensor shift |
| injected sensor faults | frozen frame (one image repeated); dead or stuck pixels (a fixed 2% mask); exposure clipping (gain ×3, clip); row dropout (25% of rows zeroed); JPEG quality 10; lens occlusion (a fixed opaque blob, 20% of the area) | 300 each, separated by 500 clean | per-fault detection |

Ramps step through the CIFAR-10-C severities, mixing severity k and k + 1 with a linearly rising probability.

**Scores.**

| group | scores |
|---|---|
| head | MSP, max-logit, logit gap, entropy, energy, DOCTOR |
| penult | margin, d1, class-conditional Mahalanobis, L2-normalised kNN (Sun et al.), a joint logit-gap + margin score fitted on the calibration split |
| early taps (stem, end of stage 1, end of stage 2) | per-frame Hotelling T² and kNN density |
| raw input | difference from the previous frame, saturated-pixel fraction, Laplacian variance, per-pixel persistence |
| windowed | H, AX-1-style T_perp / T_par, BBSD-KS on softmax outputs, mean MSP, ATC accuracy estimate |

**Sequential wrappers.** CUSUM on each score, standardised on calibration streams. The CUSUM threshold is set for a
nominal ARL₀ (for example, 10⁴ frames). A conformal-martingale variant is optional.

**Metrics.**

| level | metric |
|---|---|
| per frame, error flag | AUROC (error vs correct), AURC, FPR at 95% recall, precision at flag rates of 1%, 5% and 10%; on clean and on each corruption |
| per frame, fault flag | AUROC (fault vs clean) per fault type and per source (raw, early tap, penult, head) |
| per stream | detection delay (frames from onset to alarm) at matched ARL₀ |
| per stream | false alarms per hour at a declared 30 fps: alarm onsets after debouncing, per 108,000 clean or benign frames |
| per stream | alarm rate on benign brightness and on the label-skew burst |
| system | missed harmful segments; availability cost, the fraction of frames deferred (Guerin et al. 2022) |

**Candidate predictions** (the owner fixes the thresholds at pre-registration):
- **P1.** On clean and corrupted frames, no geometric per-sample score beats the better of MSP and the logit gap in
  AURC by more than a pre-set margin, in both seeds. This follows from rows 9 and 10.
- **P2.** The frozen-frame fault is invisible to every head and penult score (per-frame AUROC 0.45-0.55), and trivial
  for raw frame differencing. This is recorded to document hard limit 7.
- **P3.** For stuck pixels, occlusion and exposure clipping, an early-tap or raw-input score beats the penult and head
  scores in per-frame AUROC.
- **P4.** At matched ARL₀, CUSUM on H detects the fog ramp and the noise onset no later than CUSUM on mean MSP. This
  is the first head-to-head test of H against an output baseline. The direction is a guess, so the test is two-sided.
- **P5.** H-CUSUM does not alarm on the brightness segment; kNN-density CUSUM does. This tests novelty against harm.
- **P6.** Unprojected penult batch statistics and BBSD alarm on the label-skew burst; T_perp does not. This is the
  first confirmation test of the AX-1 hint.

**Cost** (basis in section 6.5). Training two fresh resnet20 seeds takes about 17 min, since two 200-epoch runs go
concurrently (docs/plans/STAGE1.md:62). Dumps with logits for the stream rows take minutes. Scoring is CPU work.
Estimate 1-1.5 pod-hours, about $0.75-1.1. Budget 2-3 h, about $1.5-2.2. The optional ImageNet stage adds 1-2 h,
about $0.75-1.5. Four ViT dump-and-build runs took about 35 min in B1b.

### 6.2 E-F2a: early exit and cascade curves on frozen backbones

**What.** Per-sample exit rules at every tap, using two exit heads:
- a training-free nearest-centre readout;
- linear side heads fitted on train-reference activations. The backbone is frozen, so the atlas is unchanged.

**Exit rule and baselines.**
- Exit rule: side-head max-softmax ≥ θ, swept over θ.
- Baselines:
  - the full model;
  - resnet20 alone;
  - a fixed exit at the commit tap;
  - a resnet20 → resnet56 cascade on low confidence;
  - on ViT, the block.9 and block.10 readouts that already exist (discovery-grade).

**Metrics.**
- Accuracy against mean block compute, and against measured batch-1 latency on GPU and CPU (mean and p99).
- Error enrichment among early exiters, which tests the overconfidence risk.
- The same curves on the corrupted and faulted segments from 6.1.

**Candidate prediction.** On CIFAR ResNets, no exit rule saves more than 25% of mean block compute at ≤ 0.5 pt accuracy
loss, and early exiters are enriched for errors under corruption.

**Cost.** Discovery runs on existing dumps are CPU only. Confirmation on the 6.1 seeds adds about 20-30 pod-minutes,
including latency timing (about $0.25-0.4).

### 6.3 E-F2b: frame and feature gating on real video, with fault injection

**What.** A public driving or robot video set with natural temporal continuity. Choose it and pin its version and
licence at pre-registration; none is on the volume today. A frozen ImageNet backbone (ResNet50, ViT-B/16), or the
robot's encoder once chosen.

**Labels.** None needed. The reference is the full model's output on every frame, as NoScope and Reducto evaluate
against a reference model.

**Measurements.**
- Temporal redundancy per tap: the fraction of frame-to-frame feature deltas below a threshold, per tap and per
  threshold. This bounds the F2 gain (item I6).
- Gating policies:
  - fixed frame skip, k ∈ {2, 3, 5};
  - a pixel-difference trigger;
  - a feature-difference trigger at an early tap;
  - ViT token reuse emulated by FLOP accounting;
  - all with forced refresh every N frames.
- Inject the 6.1 faults into short windows.

**Metrics.**
- Agreement with the full model against the fraction of compute.
- p99 latency.
- Delay of F1 fault alarms with each gate against ungated.
- Frames on which a gate skipped a faulted input.

**Candidate prediction.** Without an integrity check, a pixel-difference gate skips most frozen-frame frames. With
forced refresh and the integrity check, fault-alarm delay grows by no more than N frames.

**Cost.** Download and extraction plus scoring: 2-4 pod-hours, about $1.5-3. Dataset choice and licence review are
owner time. This is the most uncertain estimate.

### 6.4 E-F2c: adaptation gating (the closed-loop v0 in synthesis_report.md next step 5)

**What.** TENT, rung 1 of the deformation ladder (the existing manifests, AGENT_LOOP.md:108-110), with five policies: never adapt, always adapt,
entropy-gated (EATA-style), H-gated, and H plus output. It runs on the 6.1 stream.

**Metrics.**
- Online accuracy of the worst segment.
- Accuracy after recovery.
- Backward passes saved.
- Adaptation on label-skew or faulted segments, which should be zero.

**Cost.** About $1-3 (synthesis_report.md next step 5).

### 6.5 Cost basis (observed in batch 3)

| item | observed | source |
|---|---|---|
| GPU and rate | RTX 4090 at $0.74/h | docs/plans/STAGE2B.md:331 |
| whole batch-3 session | about 4.1 pod-hours (13:32 → 17:39 UTC), about $3.0 | results/margin_b1_vitb16/SESSION.md timeline; synthesis_report.md §6 as corrected in challenge.md item 16 |
| resnet20 training | two 200-epoch seeds concurrently in 17 min | docs/plans/STAGE1.md:62 |
| resnet56 training | 3.45-3.52 s per epoch (observed); two 200-epoch seeds estimated at about 23 min | docs/plans/STAGE2B.md:24, :324-326 |
| CIFAR atlas build | 80-84 s per dump; margin rebuild 9.3-9.5 s | docs/plans/STAGE2B.md:30, :258 |
| CPU probe | 11 probe records created between 15:12 and 15:29 UTC | results/anomaly_h1/SESSION.md:74-83 |
| ImageNet runs | four ViT dump-and-build runs in about 35 min; Stage B 15-118 s per tap at n = 25,000 | results/margin_b1_vitb16/SESSION.md:79-102 |
| estimate risk | B1's runtime estimate was far too low and caused an exit-124 timeout | results/margin_b1_vitb16/RELAUNCH_r2.md |

**Totals.** E-F1 (CIFAR), E-F2a and E-F2c together come to about 4-7 pod-hours, about $3-5.5 with margin. Adding the
ImageNet stage and E-F2b brings the total to about $5-11. GPU rent is not the constraint. Pre-registration, review and
the missing local Python lane are (synthesis_report.md §6).

---

## 7. Open questions

1. **Does any geometric per-sample score beat the logit gap under shift?** E3 was never run. On CIFAR, the logit gap
   was never computed.
2. **Is the ViT confidence-matched lead (V3b) real outside the DeiT lineage?** B1c on the unspent augreg reserve ViT
   would test it, with a joint logit-gap + margin detector pre-registered.
3. **Which source carries per-sample sensor-fault information?** Raw input, an early tap, or neither? At what false
   alarm rate?
4. **Can a harm grade resolve harm within one corruption type,** and at small batch sizes (16-64) and on ramps?
5. **Does projecting out the class span (T_perp) separate label skew from sensor shift** on fresh seeds? AX-1 was
   ruled out on other clauses.
6. **What is the temporal redundancy of the robot's own feature stream, per tap?** This decides whether F2-a and F2-b
   are worth building.
7. **Which encoder will the robot use, and what counts as an error for it?** Self-supervised encoders have no class
   head, so F1-a needs a task-specific error definition.
8. **What does a false alarm cost** compared with a miss, per action? CONTROLLER.md (synthesis_report.md next step 3)
   should fix the availability cost before any threshold is chosen.
9. **What are the frame rate, the latency budget and the worst-case bound?** Without them, F2's value cannot be priced
   in a control loop.
10. **Can an adversary exploit the gates** (slowdown, or suppressing a flag), and how should gate decisions themselves
    be monitored?

---

## 8. References

### Atlas files read

- ATLAS_STATUS.md (rows 1-11)
- MASTER_SUMMARY.md (:14-18, :81-150, :233-274)
- CLAUDE.md (:20-24)
- AGENT_LOOP.md (:95-119)
- docs/plans/A3_MARGIN.md (:92-97)
- docs/plans/ANOMALY_H1.md (:56-66, :142-186, :265-310, :576-786)
- docs/plans/B1_VIT_MARGIN.md (:125-132, :236-276)
- docs/plans/STAGE1.md (:62)
- docs/plans/STAGE2B.md (:24-30, :258, :315-331)
- docs/history/VISION.md (:30-70)
- results/margin_v1_resnet20_s1/SESSION.md
- results/margin_b1_vitb16/SESSION.md (timeline, :79-102) and RELAUNCH_r2.md
- results/anomaly_h1/SESSION.md
- `.per_layer.*.margin_typeb` and `.per_layer.*.class_centers` in:
  - results/margin_v1_resnet20_s{1,2}/atlas.json
  - results/margin_v1_resnet56_{s1,s2}/atlas.json
  - results/margin_b1_{vitb16,deitb,resnet50}{,_swap}/atlas.json
  - results/atlas_v1_resnet{20,56}_s{1,2}/atlas.json
- `.cross_layer.commit_layer` in results/atlas_v1_resnet{20,56}_s1/atlas.json
- `.dumps.*.AX3` in results/anomaly_probe_resnet56_s{1,2}/probe.json
- The external evaluation: synthesis_report.md and challenge.md in the session scratchpad (`.../scratchpad/evalprog/`),
  not committed.

### Literature, F1 (URLs opened)

- Hendrycks, Gimpel. A Baseline for Detecting Misclassified and Out-of-Distribution Examples. ICLR 2017. https://arxiv.org/abs/1610.02136
- Jiang, Kim, Guan, Gupta. To Trust Or Not To Trust A Classifier. NeurIPS 2018. https://arxiv.org/abs/1805.11783
- Corbière et al. Addressing Failure Prediction by Learning Model Confidence (ConfidNet). NeurIPS 2019. https://arxiv.org/abs/1910.04851
- Granese et al. DOCTOR: A Simple Method for Detecting Misclassification Errors. NeurIPS 2021. https://arxiv.org/abs/2106.02395
- Zhu, Cheng, Zhang, Liu. Rethinking Confidence Calibration for Failure Prediction. ECCV 2022. https://arxiv.org/abs/2303.02970
- Jaeger, Lüth, Klein, Bungert. A Call to Reflect on Evaluation Practices for Failure Detection in Image Classification. ICLR 2023. https://arxiv.org/abs/2211.15259
- Hein, Andriushchenko, Bitterwolf. Why ReLU networks yield high-confidence predictions far away from the training data and how to mitigate the problem. CVPR 2019. https://arxiv.org/abs/1812.05720
- Ovadia et al. Can You Trust Your Model's Uncertainty? NeurIPS 2019. https://arxiv.org/abs/1906.02530
- Lee, Lee, Lee, Shin. A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks. NeurIPS 2018. https://papers.nips.cc/paper/7947-a-simple-unified-framework-for-detecting-out-of-distribution-samples-and-adversarial-attacks
- Liu, Wang, Owens, Li. Energy-based Out-of-distribution Detection. NeurIPS 2020. https://arxiv.org/abs/2010.03759
- Sun, Ming, Zhu, Li. Out-of-Distribution Detection with Deep Nearest Neighbors. ICML 2022. https://arxiv.org/abs/2204.06507
- Zhang, Yang et al. OpenOOD v1.5. https://arxiv.org/abs/2306.09301
- Cheng, Nührenberg, Yasuoka. Runtime Monitoring Neuron Activation Patterns. DATE 2019. https://arxiv.org/abs/1809.06573
- Henzinger, Lukina, Schilling. Outside the Box: Abstraction-Based Monitoring of Neural Networks. ECAI 2020. https://arxiv.org/abs/1911.09032
- Yatbaz, Dianati, Koufos, Woodman. Run-time Monitoring of 3D Object Detection in Automated Driving Systems Using Early Layer Neural Activation Patterns. CVPR 2024 Workshop (SAIAD). https://arxiv.org/html/2404.07685v1
- Ferreira, Arlat, Guiochet, Waeselynck. Benchmarking Safety Monitors for Image Classifiers with Machine Learning. PRDC 2021. https://arxiv.org/abs/2110.01232
- Guerin, Delmas, Ferreira, Guiochet. Out-Of-Distribution Detection Is Not All You Need. AAAI 2023. https://arxiv.org/abs/2211.16158v2
- Guerin, Ferreira, Delmas, Guiochet. Unifying Evaluation of Machine Learning Safety Monitors. ISSRE 2022. https://arxiv.org/abs/2208.14660
- Daftry, Zeng, Bagnell, Hebert. Introspective Perception: Learning to Predict Failures in Vision Systems. IROS 2016. https://arxiv.org/abs/1607.08665
- Rabiee, Biswas. IVOA: Introspective Vision for Obstacle Avoidance. IROS 2019. https://arxiv.org/abs/1903.01028
- Richter, Roy. Safe Visual Navigation via Deep Learning and Novelty Detection. RSS 2017. https://dspace.mit.edu/entities/publication/cf623545-bd1d-4190-87ce-a096ccc61e0f
- Stocco, Weiss, Calzana, Tonella. Misbehaviour Prediction for Autonomous Driving Systems. ICSE 2020. https://2020.icse-conferences.org/details/icse-2020-papers/129/Misbehaviour-Prediction-for-Autonomous-Driving-Systems
- Kuhn, Hofbauer, Petrovic, Steinbach. Introspective Black Box Failure Prediction for Autonomous Driving. IV 2020. https://ieeexplore.ieee.org/document/9304844/ (DOI 10.1109/IV47402.2020.9304844; numbers from the abstract as served by the Semantic Scholar API)
- Antonante, Spivak, Carlone. Monitoring and Diagnosability of Perception Systems. IROS 2021. https://arxiv.org/abs/2011.07010
- Rahman, Corke, Dayoub. Run-Time Monitoring of Machine Learning for Robotic Perception: A Survey. IEEE Access 2021. https://arxiv.org/abs/2101.01364
- Sinha et al. A System-Level View on Out-of-Distribution Data in Robotics. arXiv 2022. https://arxiv.org/abs/2212.14020
- Cai, Koutsoukos. Real-time Out-of-distribution Detection in Learning-Enabled Cyber-Physical Systems. ICCPS 2020. https://arxiv.org/abs/2001.10494
- Kaur et al. CODiT: Conformal Out-of-Distribution Detection in Time-Series Data (ICCPS title adds "for Cyber-Physical Systems"). ICCPS 2023. https://arxiv.org/abs/2207.11769
- Hendrycks, Dietterich. Benchmarking Neural Network Robustness to Common Corruptions and Perturbations. ICLR 2019. https://arxiv.org/pdf/1903.12261
- Secci, Ceccarelli. On failures of RGB cameras and their effects in autonomous driving applications. ISSRE 2020; extended as Ceccarelli, Secci, RGB Cameras Failures and Their Effects in Autonomous Driving Applications, IEEE TDSC 20(4), 2023. https://arxiv.org/abs/2008.05938 (the current arXiv version, whose abstract gives the six detectors and the driving agent, carries the TDSC title)
- Uřičář, Křížek, Sistu, Yogamani. SoilingNet: Soiling Detection on Automotive Surround-View Cameras. ITSC 2019. https://arxiv.org/abs/1905.01492
- Chow. On Optimum Recognition Error and Reject Tradeoff. IEEE T-IT 1970. https://ieeexplore.ieee.org/document/1054406/
- Geifman, El-Yaniv. Selective Classification for Deep Neural Networks. NeurIPS 2017. https://papers.neurips.cc/paper/7073-selective-classification-for-deep-neural-networks
- Jones, Sagawa, Koh, Kumar, Liang. Selective Classification Can Magnify Disparities Across Groups. ICLR 2021. https://iclr.cc/virtual/2021/poster/3060
- Page. Continuous Inspection Schemes. Biometrika 41(1-2):100-115, 1954. https://doi.org/10.1093/biomet/41.1-2.100
- Volkhonskiy, Nouretdinov, Gammerman, Vovk, Burnaev. Inductive Conformal Martingales for Change-Point Detection. COPA 2017 (PMLR 60). https://arxiv.org/abs/1706.03415
- Rabanser, Günnemann, Lipton. Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift. NeurIPS 2019. https://arxiv.org/abs/1810.11953
- Podkopaev, Ramdas. Tracking the Risk of a Deployed Model and Detecting Harmful Distribution Shifts. ICLR 2022. https://arxiv.org/abs/2110.06177
- Amoukou et al. Sequential Harmful Shift Detection Without Labels. NeurIPS 2024. https://arxiv.org/abs/2412.12910
- Garg, Balakrishnan, Lipton, Neyshabur, Sedghi. Leveraging Unlabeled Data to Predict Out-of-Distribution Performance (ATC). ICLR 2022. https://arxiv.org/abs/2201.04234
- Guillory, Shankar, Ebrahimi, Darrell, Schmidt. Predicting with Confidence on Unseen Distributions (DoC). ICCV 2021. https://arxiv.org/abs/2107.03315

### Literature, F2 (URLs opened unless marked)

- Teerapittayanon, McDanel, Kung. BranchyNet. https://arxiv.org/abs/1709.01686
- Huang et al. Multi-Scale Dense Networks (MSDNet). https://arxiv.org/abs/1703.09844
- Kaya, Hong, Dumitras. Shallow-Deep Networks. ICML 2019. https://arxiv.org/abs/1810.07052
- Wang et al. SkipNet. ECCV 2018. https://arxiv.org/abs/1711.09485
- Wang et al. Wisdom of Committees: An Overlooked Approach to Faster and More Accurate Models. ICLR 2022. https://arxiv.org/abs/2012.01988
- Raposo et al. Mixture-of-Depths. 2024. https://arxiv.org/abs/2404.02258
- Schuster et al. Confident Adaptive Language Modeling (CALM). NeurIPS 2022. https://arxiv.org/abs/2207.07061
- Jazbec et al. Fast Yet Safe: Early-Exiting with Risk Control. NeurIPS 2024. https://arxiv.org/abs/2405.20915
- Meronen et al. Fixing Overconfidence in Dynamic Neural Networks. WACV 2024. https://arxiv.org/abs/2302.06359
- Han et al. Dynamic Neural Networks: A Survey. https://arxiv.org/abs/2102.04906
- Han et al. Latency-aware Spatial-wise Dynamic Networks (LASNet). NeurIPS 2022. https://arxiv.org/abs/2210.06223
- Hong et al. A Panda? No, It's a Sloth: Slowdown Attacks on Adaptive Multi-Exit Neural Network Inference. ICLR 2021. https://arxiv.org/abs/2010.02432
- Rao et al. DynamicViT. NeurIPS 2021. https://arxiv.org/abs/2106.02034
- Liang et al. Not All Patches are What You Need: Expediting Vision Transformers via Token Reorganizations (EViT). ICLR 2022. https://arxiv.org/abs/2202.07800
- Yin et al. A-ViT (arXiv title: AdaViT: Adaptive Tokens for Efficient Vision Transformer). CVPR 2022. https://arxiv.org/abs/2112.07658
- Bolya et al. Token Merging (ToMe). ICLR 2023. https://arxiv.org/abs/2210.09461 ; merge of ToMe for Stable Diffusion into the AUTOMATIC1111 web UI: https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9256
- Xu et al. VLA-Cache: Efficient Vision-Language-Action Manipulation via Adaptive Token Caching. NeurIPS 2025. https://arxiv.org/abs/2502.02175
- Liu et al. Bridging the Semantic-Action Gap in Visual Token Pruning for Efficient VLA Inference (VLA-Pruner). 2025. https://arxiv.org/abs/2511.16449
- Neil, Lee, Delbruck, Liu. Delta Networks for Optimized Recurrent Network Computation. ICML 2017. https://arxiv.org/abs/1612.05571 ; EdgeDRNN (Gao et al., AICAS 2020): https://arxiv.org/abs/1912.12193
- O'Connor, Welling. Sigma-Delta Quantized Networks. https://arxiv.org/abs/1611.02024
- Habibian et al. Skip-Convolutions for Efficient Video Processing. CVPR 2021. https://arxiv.org/abs/2104.11487
- Parger et al. DeltaCNN. CVPR 2022. https://arxiv.org/abs/2203.03996
- Zhu et al. Deep Feature Flow for Video Recognition. CVPR 2017. https://arxiv.org/abs/1611.07715 ; https://github.com/msracver/Deep-Feature-Flow
- Shelhamer et al. Clockwork Convnets for Video Semantic Segmentation. https://arxiv.org/abs/1608.03609
- Wu et al. AdaFrame. CVPR 2019. https://arxiv.org/abs/1811.12432
- Ghodrati et al. FrameExit. CVPR 2021. https://arxiv.org/abs/2104.13400
- Kang et al. NoScope. PVLDB 2017. https://arxiv.org/abs/1703.02529
- Li, Padmanabhan, Zhao, Wang, Xu, Netravali. Reducto: On-Camera Filtering for Resource-Efficient Real-Time Video Analytics. SIGCOMM 2020. https://collaborate.princeton.edu/en/publications/reducto-on-camera-filtering-for-resource-efficient-real-time-vide/
- Li, Wang, Ramanan. Towards Streaming Perception. ECCV 2020. https://arxiv.org/abs/2005.10420
- Gallego et al. Event-based Vision: A Survey. IEEE TPAMI 2020. https://arxiv.org/abs/1904.08405
- Messikommer et al. Event-based Asynchronous Sparse Convolutional Networks. ECCV 2020. https://arxiv.org/abs/2003.09148
- Sony Semiconductor Solutions, IMX636/IMX637 event-based vision sensors announcement, 2021-09-09. https://www.sony-semicon.com/en/news/2021/2021090901.html
- Falanga, Kleber, Scaramuzza. Dynamic obstacle avoidance for quadrotors with event cameras. Science Robotics 5(40), 2020. https://doi.org/10.1126/scirobotics.aaz9712 (title from Crossref; the 3.5 ms latency from the abstract as served by the Semantic Scholar API)
- Tishby, Pereira, Bialek. The Information Bottleneck Method. https://arxiv.org/abs/physics/0004057
- Saxe et al. On the Information Bottleneck Theory of Deep Learning. ICLR 2018. https://research.ibm.com/publications/on-the-information-bottleneck-theory-of-deep-learning--1
- Alemi et al. Deep Variational Information Bottleneck. ICLR 2017. https://arxiv.org/abs/1612.00410
- Shao, Mao, Zhang. Learning Task-Oriented Communication for Edge Inference. IEEE JSAC 2022. https://arxiv.org/abs/2102.04170
- Heemels, Johansson, Tabuada. An Introduction to Event-triggered and Self-triggered Control. CDC 2012. https://heemels.tue.nl/content/papers/HeeJoh_CDC12a.pdf
- Miskowicz. Send-On-Delta Concept: An Event-Based Data Reporting Strategy. Sensors 6(1):49-63, 2006. https://pmc.ncbi.nlm.nih.gov/articles/PMC3865911/
- Khojasteh, Tallapragada, Cortés, Franceschetti. Time-triggering versus event-triggering control over communication channels. CDC 2017. https://arxiv.org/abs/1703.10744
- Niu et al. Efficient Test-Time Model Adaptation without Forgetting (EATA). ICML 2022. https://arxiv.org/abs/2204.02610
- Niu et al. Towards Stable Test-Time Adaptation in Dynamic Wild World (SAR). ICLR 2023. https://arxiv.org/abs/2302.12400
- Solowjow, Trimpe. Event-triggered Learning. Automatica 2020. https://arxiv.org/abs/1904.03042
