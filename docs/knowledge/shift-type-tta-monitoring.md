# Shift-type identification and test-time-adaptation monitoring: prior art for Atlas

Last reviewed: 2026-09-23; part of the Atlas knowledge base (docs/knowledge/README.md)

Prior-art labels in this file (rediscovered, extends, known-in-research, not found in prior art, and so on) mark
starting points to adopt and refine for Atlas's use; they never down-rank an item (docs/knowledge/README.md §1).

- **Written against:** Atlas HEAD 343f651.
- **Scope:** the inventory items ST-1 to ST-8, plus LH-1 to LH-4 and DO-4 where this literature bears on them. Atlas
  paths are relative to the repo root.

**How citations were checked.** Each citation carries one of three marks:
- **PDF**: the full text was read.
- **abs**: the arXiv abstract, HTML page or publisher page was opened.
- **search**: the title, authors and venue were seen in search results, but the paper was not opened.

Nothing is cited from memory alone. Venues that could not be confirmed are marked "venue unconfirmed".

---

## 1. What this family of methods measures

This family answers five questions. They are listed in order of how directly they feed an adapt/hold/escalate
controller.

1. **Has the input distribution changed? (shift detection)**
   - The core tool is a two-sample test between a reference window and a deployment window, usually run after a
     dimensionality reduction (DR).
   - The reduced representation can be:
     - the network's own softmax outputs (BBSDs);
     - its hard predictions (BBSDh);
     - a feature layer;
     - an untrained or trained autoencoder.
   - The test is either multivariate (MMD) or univariate per dimension (KS with a Bonferroni correction) [R1].
   - Online versions add control of the false-alarm rate under continuous monitoring [R5, R6, R9, M7]. CUSUM is the
     classical change-point statistic [R10].
2. **What kind of shift is it? (identification and characterisation)** The literature splits this into four
   sub-questions:
   - covariate vs label (prevalence) vs mixed shift [R2, R3, R4];
   - which domain or corruption family, as read by routers in continual TTA [T14, T15];
   - which samples typify the shift, found through domain-classifier exemplars [R1];
   - which layers carry it, and therefore which layers should respond [R13, R14, R15, T16].
3. **Is it harmful? (malignancy, harmful-shift detection, label-free accuracy)**
   - A shift is not the same as harm.
   - The labelled versions are Rabanser's malignancy check [R1] and sequential risk tracking [R5].
   - The label-free versions replace the loss with a proxy: a learned error estimator [R6], model disagreement [R7, R8,
     M3] or confidence thresholds [R16].
4. **How is the model changed at test time? (the "adapt" action)** The main families are:
   - replacing the normalisation statistics [T1-T3];
   - entropy minimisation on the BN affine parameters [T4];
   - entropy minimisation with sample filtering and a guard against forgetting [T5];
   - sharpness-aware updates with reliable-sample filtering and a reset rule [T9];
   - teacher-student training with stochastic restore to the source weights [T6];
   - robust BN plus a class-balanced memory for correlated streams [T8, T10];
   - adjusting only the outputs, with no parameter update [T7];
   - prompts with a per-domain coreset [T14].
5. **Is the adaptation going wrong, and should it continue? (TTA monitoring)**
   - Collapse and reset triggers [T9, M1, M2, M3, M5, M6].
   - Label-free estimates of the adapted model's accuracy, that is, of the benefit of adapting [M3, M4].
   - Statistical risk monitoring while the model is being updated [M7].

**How this maps onto Atlas's primary question.** The question is which of these quantities internal geometry can read,
and whether geometry adds anything beyond the head (softmax, predicted-class histogram, entropy).
- In this literature, output statistics are the default, strong baseline for questions 1, 3 and 5.
- Feature geometry is used mainly for question 2 (identification and routing) and in a few collapse sensors [M2, T14].

---

## 2. Key works

### 2.1 Shift detection, identification and harmful-shift monitoring

| id | citation | URL | checked | content relevant to Atlas |
|---|---|---|---|---|
| R1 | Rabanser, S., Günnemann, S., Lipton, Z. C. (2019). *Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift.* NeurIPS 32. | https://arxiv.org/abs/1810.11953 | PDF | Compares DR methods (NoRed, PCA, SRP, UAE, TAE, BBSDs, BBSDh, domain classifier) against tests (MMD, KS + Bonferroni, chi-squared, binomial). BBSDs is the best DR method overall, and UAE the best for multivariate tests. The domain classifier is weak below about 100 samples. Large shifts are detected better than chance with about 20 samples. Small shifts, and batches with only 10% of samples perturbed, are hard. Domain-classifier exemplars characterise the shift; malignancy is judged by labelling the top exemplars. BBSDs works "even when some of its underlying assumptions do not hold". |
| R2 | Lipton, Z. C., Wang, Y.-X., Smola, A. (2018). *Detecting and Correcting for Label Shift with Black Box Predictors.* ICML. | https://arxiv.org/abs/1802.03916 | search | BBSE estimates the test label marginal from a black-box predictor's confusion matrix; the matrix only needs to be invertible. This is the origin of BBSD. |
| R3 | Roschewitz, M., Mehta, R., Jones, C., Glocker, B. (2024; v3 2025). *Automatic dataset shift identification to support safe deployment of medical imaging AI* (v3 title; v1 and v2 end "…root cause analysis of AI performance drift"). MICCAI 2025 per the arXiv v3 comment; v3 is an extended version. | https://arxiv.org/abs/2411.07940 | PDF (v3) | Unsupervised identification of prevalence vs covariate vs mixed shift, in four steps: (1) a BBSD + MMD "Duo" detector; (2) a prevalence estimate; (3) resampling the reference to that prevalence; (4) a re-test of features and outputs. Features from a self-supervised (SSL) encoder catch subtle covariate shifts that task-model outputs miss. Reported average identification accuracy: 95% for prevalence shift (1,000 test images), 89% for acquisition shift (500 images; 250 exams on EMBED) and 85% for mixed shifts (1,000 images). |
| R4 | Maia Polo, F., Izbicki, R., Lacerda Jr, E. G., Ibieta-Jimenez, J. P., Vicente, R. (2023). *A unified framework for dataset shift diagnostics* (DetectShift). Information Sciences. | https://arxiv.org/abs/2205.08340 | abs | Test statistics of the same form for shifts in P(X,Y), P(X), P(Y), P(X given Y) and P(Y given X). Some of these need a few target labels. |
| R5 | Podkopaev, A., Ramdas, A. (2022). *Tracking the risk of a deployed model and detecting harmful distribution shifts.* ICLR. | https://arxiv.org/abs/2110.06177 | search | A sequential test on a user-chosen risk. It separates harmful from benign shifts and controls false alarms under continuous monitoring. Needs labels. |
| R6 | Amoukou, S. I., Bewley, T., Mishra, S., Lecue, F., Magazzeni, D., Veloso, M. (2024). *Sequential Harmful Shift Detection Without Labels.* NeurIPS. | https://arxiv.org/abs/2412.12910 | abs | Replaces the true error in R5 with a trained error-estimator proxy. Controls false alarms on covariate, label, temporal and geographic shifts. |
| R7 | Ginsberg, T., Liang, Z., Krishnan, R. G. (2023). *A Learning Based Hypothesis Test for Harmful Covariate Shift* (Detectron). ICLR. | https://arxiv.org/abs/2212.02742 | search | An ensemble trained to agree on training data and disagree on test data. Its disagreement rate and entropy detect harmful covariate shift. |
| R8 | Nguyen, V., Shui, C., Giri, V., Arya, S., Verma, A., Razak, F., Krishnan, R. G. (2025). *Reliably Detecting Model Failures in Deployment Without Labels* (D3M). NeurIPS. | https://arxiv.org/abs/2506.05047 | abs | A monitor built on model disagreement: few false positives on benign shifts, with detection bounds when performance really drops. |
| R9 | Luo, R., Sinha, R., Sun, Y., Hindy, A., Zhao, S., Savarese, S., Schmerling, E., Pavone, M. (2022; revised 2024). *Online Distribution Shift Detection via Recency Prediction.* arXiv cs.RO; venue unconfirmed. | https://arxiv.org/abs/2211.09916 | abs | A streaming detector for high-dimensional robot inputs, with a guaranteed bound on the false-positive rate. Tested on simulated and hardware visual servoing. |
| R10 | Page, E. S. (1954). *Continuous Inspection Schemes.* Biometrika 41(1-2):100-115. | https://doi.org/10.1093/biomet/41.1-2.100 | abs (publisher page) | CUSUM, the classical comparator for any "abrupt change" trigger. |
| R11 | Wu, R., Guo, C., Su, Y., Weinberger, K. Q. (2021). *Online Adaptation to Label Distribution Shift.* arXiv 2107.04520; venue unconfirmed. | https://arxiv.org/abs/2107.04520 | abs | Adapts to label shift online without labels (Follow The Leader and online gradient descent, with regret bounds). |
| R12 | Hendrycks, D., Dietterich, T. (2019). *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations.* ICLR. | https://arxiv.org/abs/1903.12261 | abs | The ImageNet-C benchmark (the CIFAR-10-C corruptions Atlas uses follow the same design). |
| R13 | Lee, K., Lee, K., Lee, H., Shin, J. (2018). *A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks.* NeurIPS. | https://arxiv.org/abs/1807.03888 | abs | A Mahalanobis score under class-conditional Gaussians, ensembled across low- and high-level layers. Detects both out-of-distribution (OOD) inputs and adversarial attacks. |
| R14 | Sastry, C. S., Oore, S. (2019). *Detecting Out-of-Distribution Examples with In-distribution Examples and Gram Matrices.* NeurIPS 2019 Workshop on Safety and Robustness in Decision Making; published as *Detecting Out-of-Distribution Examples with Gram Matrices*, ICML 2020 (PMLR 119). | https://arxiv.org/abs/1912.12510 | abs | Layer-wise Gram-matrix deviations from their training ranges, used as an OOD score. |
| R15 | Lee, Y., Chen, A. S., Tajwar, F., Kumar, A., Yao, H., Liang, P., Finn, C. (2023). *Surgical Fine-Tuning Improves Adaptation to Distribution Shifts.* ICLR. | https://arxiv.org/abs/2210.11466 | abs | Which layers should be tuned depends on the shift type. Image corruptions are best handled by tuning only the early layers. |
| R16 | Garg, S., Balakrishnan, S., Lipton, Z. C., Neyshabur, B., Sedghi, H. (2022). *Leveraging Unlabeled Data to Predict Out-of-Distribution Performance* (ATC). ICLR. | https://arxiv.org/abs/2201.04234 | abs | Estimates accuracy from a learned confidence threshold, 2-4× more accurate than DoC and GDE. It also shows that identifying accuracy under shift is as hard as identifying the optimal predictor. |

### 2.2 Test-time adaptation (the "adapt" action)

| id | citation | URL | checked | content relevant to Atlas |
|---|---|---|---|---|
| T1 | Li, Y., Wang, N., Shi, J., Liu, J., Hou, X. (2016). *Revisiting Batch Normalization For Practical Domain Adaptation* (AdaBN). arXiv 1603.04779; later venue unconfirmed. | https://arxiv.org/abs/1603.04779 | abs | Replaces the BN statistics with target-domain statistics; no parameters are learned. |
| T2 | Schneider, S., Rusak, E., Eck, L., Bringmann, O., Brendel, W., Bethge, M. (2020). *Improving robustness against common corruptions by covariate shift adaptation.* NeurIPS 33. | https://proceedings.neurips.cc/paper/2020/hash/85690f81aadc1749175c187784afc9ee-Abstract.html | abs | Replaces the BN statistics with those of the corrupted data. For ResNet-50, 32 samples already beat the prior state of the art, and ImageNet-C mCE falls from 76.7 to 62.2. Even a single sample helps ResNet-50 and AugMix models. |
| T3 | Nado, Z., Padhy, S., Sculley, D., D'Amour, A., Lakshminarayanan, B., Snoek, J. (2020). *Evaluating Prediction-Time Batch Normalization for Robustness under Covariate Shift.* arXiv 2006.10963. | https://arxiv.org/abs/2006.10963 | abs | Prediction-time BN reaches mCE 60.28 on ImageNet-C and improves calibration. Results are mixed with pre-training and weaker under natural shifts. |
| T4 | Wang, D., Shelhamer, E., Liu, S., Olshausen, B., Darrell, T. (2021). *Tent: Fully Test-time Adaptation by Entropy Minimization.* ICLR. | https://arxiv.org/abs/2006.10726 | search | Minimises prediction entropy over the BN affine parameters, using test-batch statistics. This is Atlas's rung-1 adapter (scripts/tta_deform.py:29-31). |
| T5 | Niu, S. et al. (2022). *Efficient Test-Time Model Adaptation without Forgetting* (EATA). ICML, PMLR 162:16888-16905. | https://arxiv.org/abs/2204.02610 | search; threshold confirmed in the SAR PDF | Excludes unreliable (high-entropy) and redundant samples from the update, and adds a Fisher regulariser against forgetting. SAR reuses its entropy threshold of 0.4·ln C. |
| T6 | Wang, Q., Fink, O., Van Gool, L., Dai, D. (2022). *Continual Test-Time Domain Adaptation* (CoTTA). CVPR. | https://arxiv.org/abs/2203.13591 | search | A weight-averaged teacher, pseudo-labels averaged over augmentations, and stochastic restore of a small fraction of weights to the source values. |
| T7 | Boudiaf, M., Mueller, R., Ben Ayed, I., Bertinetto, L. (2022). *Parameter-free Online Test-time Adaptation* (LAME). CVPR (oral). | https://arxiv.org/abs/2201.05718 | abs | Adjusts only the outputs. Shows that prior methods fail catastrophically when their hyperparameters were not tuned for the scenario being tested. |
| T8 | Gong, T., Jeong, J., Kim, T., Kim, Y., Shin, J., Lee, S.-J. (2022). *NOTE: Robust Continual Test-time Adaptation Against Temporal Correlation.* NeurIPS. | https://arxiv.org/abs/2208.05117 | abs | Instance-aware BN plus prediction-balanced reservoir sampling. Most TTA methods degrade on temporally correlated (non-i.i.d.) streams. |
| T9 | Niu, S., Wu, J., Zhang, Y., Wen, Z., Chen, Y., Zhao, P., Tan, M. (2023). *Towards Stable Test-Time Adaptation in Dynamic Wild World* (SAR). ICLR. | https://arxiv.org/abs/2302.12400 | PDF | See the SAR notes below this table. |
| T10 | Yuan, L., Xie, B., Li, S. (2023). *Robust Test-Time Adaptation in Dynamic Scenarios* (RoTTA). CVPR. | https://arxiv.org/abs/2303.13899 | search | "Practical TTA" on gradually changing, correlated streams: robust BN, a class-balanced memory weighted by recency and uncertainty, and time-aware reweighting. |
| T11 | Zhao, B., Chen, C., Xia, S.-T. (2023). *DELTA: Degradation-Free Fully Test-Time Adaptation.* arXiv 2301.13018; venue unconfirmed. | https://arxiv.org/abs/2301.13018 | abs | Test-time batch renormalisation plus dynamic online reweighting, against class-imbalanced and dependent streams. |
| T12 | Su, Z., Guo, J., Yao, K., Yang, X., Wang, Q., Huang, K. (2024). *Unraveling Batch Normalization for Realistic Test-Time Adaptation* (TEMA). AAAI 38. | https://arxiv.org/abs/2312.09486 | arXiv listing; AAAI record in search | BN TTA degrades when the mini-batches have less class diversity than the training batches. |
| T13 | Park, S., Yang, S., Choo, J., Yun, S. (2023). *Label Shift Adapter for Test-Time Adaptation under Covariate and Label Shifts.* ICCV. | https://arxiv.org/abs/2308.08810 | abs | Estimates the target label distribution and adapts to joint covariate and label shift. |
| T14 | Zhang, Y., Mehra, A., Niu, S., Hamm, J. (2025). *DPCore: Dynamic Prompt Coreset for Continual Test-Time Adaptation.* ICML. | https://arxiv.org/abs/2406.10737 | abs + HTML | A label-free shift-type router. See the DPCore notes below this table. |
| T15 | Lee, D., Yoon, J., Hwang, S. J. (2024). *BECoTTA: Input-dependent Online Blending of Experts for Continual Test-time Adaptation.* arXiv 2402.08712; venue unconfirmed. | https://arxiv.org/abs/2402.08712 | arXiv listing | Routes inputs to a mixture of experts through domain-adaptive routers. |
| T16 | Sahoo, R. et al. (2025). *A Layer Selection Approach to Test Time Adaptation* (GALA). AAAI. | https://arxiv.org/abs/2404.03784 | abs | Chooses per batch which layers to adapt, by gradient alignment. |
| T17 | Rusak, E., Schneider, S., Pachitariu, G., Eck, L., Gehler, P., Bringmann, O., Brendel, W., Bethge, M. (2021; revised 2023). *If your data distribution shifts, use self-learning.* arXiv 2104.12928; venue unconfirmed. | https://arxiv.org/abs/2104.12928 | abs | Entropy minimisation and pseudo-labelling improve robustness across architectures. |
| T18 | Yu, Y., Sheng, L., He, R., Liang, J. (2024). *STAMP: Outlier-Aware Test-Time Adaptation with Stable Memory Replay.* ECCV. | https://arxiv.org/abs/2407.15773 | abs | Adapts only on a memory of low-entropy, label-consistent, class-balanced samples, so OOD samples in the stream do not drive the update. |

**SAR (T9) in detail**, from the full text:
- TTA fails under mixed shifts, batch size 1 and online label-imbalanced shift, and BN is a main cause.
- Models with GroupNorm or LayerNorm still collapse to predicting one class.
- The gradient norm spikes as collapse begins and then falls to about 0.
- SAR's remedies:
  - drop high-entropy, large-gradient samples;
  - use a sharpness-aware update;
  - reset the weights when the moving average (factor 0.9) of the entropy loss falls below e0 = 0.2.

**DPCore (T14) in detail**, from the HTML:
- The coreset stores pairs of (prompt, CLS-feature mean and standard deviation).
- The distance between the batch and a stored domain is the L2 distance between the means plus the L2 distance between
  the standard deviations.
- Prompts are weighted by a softmax over these distances.
- If adding the weighted prompt reduces that distance by too little (a ratio above ρ = 0.8), DPCore learns a new domain
  prompt.

### 2.3 TTA monitoring, collapse, reset and evaluation

| id | citation | URL | checked | content relevant to Atlas |
|---|---|---|---|---|
| M1 | Press, O., Schneider, S., Kümmerer, M., Bethge, M. (2023). *RDumb: A simple approach that questions our progress in continual test-time adaptation.* NeurIPS 36. | https://arxiv.org/abs/2306.05401 | abs | Introduces the Continually Changing Corruptions (CCC) benchmark. Eventually all but one state-of-the-art method collapse and do worse than a non-adapting model. A periodic reset to the pretrained weights matches or beats the state of the art. |
| M2 | Hoang, T.-H., Vo, D. M., Do, M. N. (2024). *Persistent Test-time Adaptation in Recurring Testing Scenarios* (PeTTA). NeurIPS. | https://arxiv.org/abs/2311.18193 | PDF | A collapse sensor built from feature geometry. See the PeTTA notes below this table. |
| M3 | Lee, T., Chottananurak, S., Gong, T., Lee, S.-J. (2024). *AETTA: Label-Free Accuracy Estimation for Test-Time Adaptation.* CVPR. | https://arxiv.org/abs/2404.01351 | abs | Estimates accuracy from the disagreement between the adapted model and its own dropout inferences, corrected for overconfident failures. It is 19.8 percentage points more accurate than the baselines. A model-recovery rule resets the model on consecutive or sudden drops in estimated accuracy. |
| M4 | Kim, E., Sun, M., Baek, C., Raghunathan, A., Kolter, J. Z. (2024). *Test-Time Adaptation Induces Stronger Accuracy and Agreement-on-the-Line.* NeurIPS. | https://arxiv.org/abs/2310.04941 | abs | After TTA, complex shifts reduce approximately to a single scaling variable in feature space, and the ID-vs-OOD accuracy and agreement lines strengthen. This supports label-free OOD accuracy estimates and the label-free choice of TTA hyperparameters and adaptation strategy. |
| M5 | Lim, T., Hwang, J.-W., Lee, K. (2026). *When and Where to Reset Matters for Long-Term Test-Time Adaptation* (ASR). ICLR. | https://arxiv.org/abs/2603.03796 | abs + HTML | The reset trigger is "prediction concentration": the entropy of the softmax of batch-averaged logits, compared with its own moving average. The share of weights reset grows with the excess, deepest layers first. A regulariser recovers knowledge lost in the reset. |
| M6 | Singh, V., Ganguly, D., Chen, W., et al. (2026). *Reliability-Gated Source Anchoring for Continual Test-Time Adaptation* (RMemSafe). arXiv preprint. | https://arxiv.org/abs/2605.14063 | abs | Switches the source-anchoring terms on or off using the frozen source model's normalised predictive entropy. |
| M7 | Schirmer, M., Jazbec, M., Naesseth, C. A., Nalisnick, E. (2025). *Monitoring Risks in Test-Time Adaptation.* arXiv preprint. | https://arxiv.org/abs/2507.08721 | abs | Extends confidence-sequence risk monitoring to models updated at test time, with uncertainty proxies in place of labels. It flags the point where adaptation can no longer hold performance. |
| M8 | Chen, X., Du, Z., Huang, J., Jiang, X., Lu, L., Jiang, J., Wang, Z. (2026). *Neural Collapse in Test-Time Adaptation* (NCTTA). CVPR. | https://arxiv.org/abs/2512.10421 | abs | Defines a sample-level NC3: each feature aligns with its classifier weight. This alignment degrades as the shift grows. The pseudo-label target mixes geometric proximity with confidence. Reported +14.52 points over Tent on ImageNet-C. |
| M9 | Zhao, H., Liu, Y., Alahi, A., Lin, T. (2023). *On Pitfalls of Test-Time Adaptation* (TTAB). ICML. | https://arxiv.org/abs/2306.03536 | search | Three pitfalls: hyperparameter and model selection is very hard under online batch dependency; how well TTA works depends on the model being adapted; and no method handles every shift type. |
| M10 | Cygert, S., Sójka, D., Trzciński, T., Twardowski, B. (2024). *Realistic Evaluation of Test-Time Adaptation Algorithms: Unsupervised Hyperparameter Selection.* arXiv 2407.14231; venue unconfirmed. | https://arxiv.org/abs/2407.14231 | abs | When hyperparameters are chosen by label-free surrogates, some recent state-of-the-art methods fall below older ones. Reliable selection needs some supervision. |
| M11 | Alfarra, M., Itani, H., Pardo, A., et al. (2024). *Evaluation of Test-Time Adaptation Under Computational Time Constraints.* ICML. | https://arxiv.org/abs/2304.04795 | abs | The stream runs at constant speed, so slower methods see fewer samples. Under this protocol SHOT (2020) beats SAR (2023). |
| M12 | Liang, J., He, R., Tan, T. (2024). *A Comprehensive Survey on Test-Time Adaptation under Distribution Shifts.* IJCV. | https://arxiv.org/abs/2303.15361 | abs | Taxonomy of TTA settings. |
| M13 | Maharana, et al. (2026). *Continual Test-Time Adaptation in Computer Vision: Methods, Benchmarks, and Future Directions.* TMLR. | https://arxiv.org/abs/2607.08164 | abs | Survey of continual TTA: forgetting, parameter restoration, error accumulation over long horizons. |

**PeTTA (M2) in detail**, from the full text:
- For each class y in the batch, it keeps a running mean of the penultimate features of samples pseudo-labelled y.
- The divergence is γ = 1 − exp(−(μ̂_t − μ_0)ᵀ Σ_0⁻¹ (μ̂_t − μ_0)), where μ_0 and a diagonal Σ_0 are the source statistics
  for that class, and it is averaged over the classes present in the batch.
- The averaged divergence scales both the regularisation strength and the rate of the moving-average update.
- The authors inspect feature space rather than outputs because the head is fixed and features keep more information.

**Also relevant, verified outside this table.** AutoEval (Deng & Zheng 2021) and DoC (Guillory et al. 2021) are cited and
checked in `label-free-accuracy-harm.md`. Sequential classifier two-sample tests: Jang, S., Park, S., Lee, I., Bastani,
O. (2022). *Sequential Covariate Shift Detection Using Classifier Two-Sample Tests.* ICML 2022 (PMLR 162).
https://proceedings.mlr.press/v162/jang22a.html (landing page).

---

## 3. How it is used

### 3.1 Usage patterns

| pattern | typical signal | standard vs research | typical metric | known failure modes |
|---|---|---|---|---|
| Offline robustness benchmarking | accuracy on CIFAR-10-C / ImageNet-C [R12] | standard research practice | mCE; per-corruption error | synthetic corruptions are not natural shifts [T3] |
| Window-level shift detection (monitoring) | two-sample test on softmax (BBSDs), on predicted labels (BBSDh), or MMD on features [R1] | an established research baseline, reusable on any existing classifier [R1]; production adoption not assessed here | detection accuracy at α = 0.05 against the number of target samples | small shifts and 10% contamination are hard [R1]; a detected shift need not be harmful; repeated testing inflates false alarms [R5] |
| Sequential detection with guarantees | a risk or error proxy with confidence sequences [R5, R6, M7]; recency prediction [R9]; CUSUM [R10] | research | false-alarm rate or ARL₀ (the average run length before a false alarm) against detection delay | needs labels [R5] or a trained error estimator [R6]; R1 lists correlated online data as open |
| Shift identification / root cause | BBSD + MMD + prevalence-adjusted resampling [R3]; DetectShift [R4]; domain-classifier exemplars [R1] | research | identification accuracy | errors in the prevalence estimate; needs a good encoder (an SSL encoder beats the task model on subtle shifts [R3]) |
| Label-shift estimation and correction | inverting the confusion matrix of predicted labels [R2]; online versions [R11]; inside TTA [T13] | known research method, widely reused | error of the prior estimate; accuracy after correction | assumes p(x given y) is fixed, so it breaks under mixed shift, which is why R3 resamples |
| Harmful-shift detection / label-free accuracy | disagreement [R7, R8, M3]; confidence threshold [R16]; error-estimator proxy [R6]; agreement-on-the-line [M4] | research | MAE of the accuracy estimate; true-positive rate at a fixed false-positive rate on benign shifts | overconfident failures (M3 corrects for them); every estimator assumes something about the shift [R16] |
| Adapt: normalisation statistics | replace the BN statistics [T1-T3] | the most widely used TTA baseline in research | online error | small, class-imbalanced or correlated batches, and mixed shifts [T8, T9, T11, T12]; weaker under natural shift and with pre-training [T3] |
| Adapt: entropy minimisation | TENT, EATA, SAR, CoTTA, RoTTA [T4-T6, T9, T10] | research; the standard baselines | online error; long-horizon error on CCC | collapse to a few classes [T9, M1, M2]; hyperparameter sensitivity [T7, M9, M10]; compute cost [M11] |
| Per-sample adapt gating | an entropy threshold of 0.4·ln C [T5, T9]; a redundancy filter [T5]; a large-gradient filter [T9]; OOD filtering by low entropy and label consistency [T18] | standard inside TTA methods | error; share of samples used | output entropy is itself wrong on confident mistakes [M3] |
| Collapse detection and reset | moving average of the entropy loss below e0 [T9]; prediction concentration above its moving average [M5]; drops in estimated accuracy [M3]; feature-mean divergence [M2]; periodic reset [M1]; a gate on the source model's entropy [M6] | research | long-run accuracy against no adaptation; number of resets | periodic reset is a strong baseline that most triggers do not clearly beat [M1]; every trigger has a threshold that needs tuning |
| Shift-type routing | distance from the batch's CLS mean and standard deviation to stored domains [T14]; expert routers [T15]; choosing layers by shift type [R15, T16] | research | accuracy on recurring domains; judged by downstream error, not routing accuracy | the new-vs-seen threshold (ρ) is set per benchmark |
| Label-free model and hyperparameter selection | entropy, consistency, agreement-on-the-line [M4, M9, M10] | research, unresolved | regret against oracle selection | the surrogates often rank methods wrongly; some supervision is needed [M10] |

### 3.2 Where this literature uses geometry, and where it uses the head

Geometry is used in four roles:
- as the space for two-sample tests: MMD on features [R1], and MMD on an SSL encoder, which beats task-model outputs on
  subtle covariate shifts [R3];
- as class-conditional source statistics that adapted features are compared against: PeTTA's divergence of
  pseudo-labelled class means [M2], and the Mahalanobis OOD score [R13];
- as batch mean and standard-deviation signatures of domain identity: DPCore [T14], and the BN statistics themselves
  [T1-T3];
- as a pseudo-label target aligned with the head: NCTTA [M8].

Everything else uses output statistics: collapse triggers [T9, M5, M3, M6], harm and accuracy estimates [R6-R8, R16, M3,
M4], and label shift [R2, R11, T13]. These are entropy, the predicted-class distribution, disagreement and confidence.

---

## 4. Known results relevant to Atlas

**K1. Output-based detection is the baseline to beat.**
- BBSDs was the best DR method overall in R1, even where its label-shift assumptions fail.
- Large shifts were detected better than chance with about 20 samples.
- Atlas's AX-1 clause c3 failed partly by ceiling: at batch 64 the penult already sees motion_blur s1 (AUC 0.93-0.999;
  results/anomaly_h1/SESSION.md:402). A representation that close to the head (penult or outputs) would be expected to
  share that ceiling.

**K2. Covariate vs label vs mixed shift is identified with output and feature tests plus prevalence-adjusted resampling
[R3, R4].**
- The published methods do not project features onto the class-mean subspace, as Atlas's AX-1 did.
- R3 finds that SSL-encoder features catch subtle covariate shifts that task outputs miss. This is the closest published
  parallel to Atlas's "read before the collapse" rationale (docs/plans/ANOMALY_H1.md:673-676).

**K3. Class skew and label shift break BN-based TTA [T8, T9, T11, T12].** A skew or label-shift flag ("do not adapt on
this batch") is therefore a real controller input. The standard way to obtain it is from the head: BBSE [R2] or a test
on the predicted histogram (BBSDh [R1]).

**K4. Entropy-minimisation TTA collapses on long, continual or recurring streams, and periodic reset is a hard baseline
[M1, T9, M2].**
- The recent triggers read the head: SAR's entropy moving average, ASR's prediction concentration, AETTA's disagreement.
- PeTTA's class-mean divergence is the one feature-geometric exception.
- SAR also records a gradient-norm spike at the onset of collapse, a free label-free early warning.

**K5. Atlas's planned collapse bar is lenient in favour of geometry.** Three asymmetries (a sketch of a fair setup
follows this list):
- **Unequal onset rules.** `atlas.ladder` counts a geometric onset when a metric leaves its step-0 value by more than tol
  = 0.05. It counts collapse only when pred_entropy falls below 0.8 of its step-0 value (atlas/ladder.py:10-12, :72,
  :104-112).
- **Different model states.** pred_entropy is measured in deployed mode, with train-mode BN
  (scripts/tta_deform.py:136-140, :212-216). The geometry is dumped in `affine` mode, with the original running
  statistics (the manifests' `dump_mode: affine`; ATLAS_README.md:152-160).
- **A weak comparator.** The literature's triggers react to trends: ASR compares against a moving average, SAR uses an
  entropy moving average. Waiting for a 20% entropy collapse is much slower.

  A fair test calibrates every monitor to the same false-onset rate on the benign run and compares them in the same
  model state.

**K6. The benefit of adapting, the variable an adapt/hold decision needs, can already be estimated without labels in
research.** AETTA estimates the adapted model's accuracy [M3]. Agreement-on-the-line after TTA supports accuracy
estimates and the choice of hyperparameters [M4]. Atlas has no such signal (ST-6).

**K7. Harmful vs benign is the right frame for a HOLD band [R1, R5-R8].**
- The label-free versions use disagreement or learned error proxies.
- Atlas's H (LH-1, LH-2) is a feature-distance proxy for harm. Its between-split alignment with cost is matched by any
  penult shift statistic (results/anomaly_h1/SESSION.md:374-378).
- That fits M4's finding that shifts reduce to about one scaling variable in feature space. M4's finding is
  post-adaptation; applying it to unadapted features is an inference, not a result.

**K8. Routing by batch feature statistics is known [T14, T15].**
- DPCore matches the batch's CLS mean and standard deviation to stored domain signatures and has a new-domain threshold.
  Atlas's AX-2b whitened mean-shift templates belong to the same family.
- R15 shows that the right response (which layers to tune) depends on shift type. That downstream benefit, not routing
  accuracy, is what a router has to deliver.

**K9. Where a shift lives in depth.**
- Input-level corruptions are best handled in early layers [R15].
- Feature ensembles across layers and Gram matrices catch low-level anomalies [R13, R14].
- This is the known background for AX-2a's per-tap drift profile.

**K10. Label-free hyperparameter and model selection is unreliable [M9, M10, T7], and compute matters [M11].** Any
closed-loop claim from Atlas must be tested under a realistic protocol, not tuned on the test stream.

**K11. Sequential monitors with false-alarm guarantees exist** for labelled risk [R5], for label-free risk [R6, M7] and
for robot streams [R9]. The legacy Gate 3 "acceleration" statistic has none of these properties, and no FPR or
detection-delay figures are on record (MASTER_SUMMARY.md:81-146).

---

## 5. Relation to Atlas items

| item | relation | literature | Atlas evidence | note |
|---|---|---|---|---|
| ST-1 (AX-2b family router) | rediscovered (known in research) | T14, T15, R3, R1 | results/anomaly_h1/SESSION.md:307-316, :383-386; docs/plans/ANOMALY_H1.md:707-722 | Routing is 1.000, but random-init nets route N, B and L as well, and the templates are refit per network. DPCore uses the same kind of batch-statistic template plus a new-domain rule. No head baseline could be run: the probe refuses any dump that holds logits and reads only argmax (docs/plans/ANOMALY_H1.md:585-588; scripts/anomaly_probe.py:302). |
| ST-2 (AX-2a tap profile) | partly known; untested here (gate CLOSED) | R15, R13, R14, T16 | results/anomaly_h1/SESSION.md:293-305, :403 | "Low-level shifts show early" is known. A per-tap onset profile used as a readout of shift type was not found in the literature. The gate turned on 0.0017 AUC, inside sampling noise. |
| ST-3 (AX-1 T_perp / T_par) | goal known; the mechanism was not found in prior art, and was ruled out | R2, R3, R4, T9, T11, T12 | results/anomaly_h1/SESSION.md:272-291, :399-402 | The known route from covariate vs prior shift runs through the head: BBSE plus a re-test after resampling to the estimated prevalence [R3]. T_perp failed c3 by ceiling, and its single-class FPR through a class-4 test-mean leak. T_par was never compared with BBSE or BBSDh. |
| ST-4 (AX-4 e_perp) | not found in prior art; ruled out | nearest analogue R13 (per-sample feature distance) | results/anomaly_h1/SESSION.md:339-350, :404 | No published method uses the residual off the class span. d1 (AUROC 0.61-0.74) was the better per-sample covariate score. (For semantic OOD, residual-subspace scores such as ViM and NECO are known; see `density-ood.md`.) |
| ST-5 (Gate 3, density acceleration) | building blocks known in research; the legacy evidence is below the literature's standard | R1 (BBSDh), R2, R11, R10, R9, R5 | MASTER_SUMMARY.md:81-146 | A second difference of the predicted-class histogram is an ad-hoc change detector. The standard is a two-sample or sequential test on the predicted-label distribution, with CUSUM or confidence-sequence alarm control, reported as FPR against detection delay. It is a head signal, not geometry. |
| ST-6 (TTA deformation monitor; adaptation benefit) | partly known; untested here | M2, M8, T9, M5, M3, M1, M4, M7 | scripts/tta_deform.py:1-31, :136-140, :212-216; atlas/ladder.py:10-12; experiments/queue/tta_tent_resnet20_{fog3,collapse}.yaml; ATLAS_README.md:152-175; AGENT_LOOP.md:108-111; no results/tta_* directory | PeTTA already senses collapse through the divergence of class-conditional feature means from the source means, so a geometric collapse sensor is not new. Atlas's topology metrics (merge_tau, adjacency_rho, d_nc1) as leading indicators were not found in the literature. The benefit of adapting is estimated label-free by M3 and M4. |
| ST-7 (legacy Gate-1 roster) | partly known | R13 (adversarial and OOD via Mahalanobis), R1 (adversarial shifts detectable; 10% contamination hard), R14 | MASTER_SUMMARY.md:36-47, :85-86 | Never re-measured in Atlas. Each part has an output or Mahalanobis baseline in the literature. |
| ST-8 (pipeline mis-normalisation) | known in research as a covariate shift; Atlas's geometry is no better than the head | R1, T1-T3 | results/atlas_v0_resnet20_cifar10/SESSION.md:122-127; results/atlas_v1_resnet20_s1/SESSION.md:27-32 | A global contrast rescale is a covariate shift that BBSD and MMD tests target, and BN-statistics adaptation largely absorbs it. Atlas found this one from the training log, not from geometry. |
| LH-1 (split-level H) | the task is known; the statistic is Atlas's own; head comparison missing | R1 (malignancy), R5-R8, R16, M3, M4 | results/anomaly_h1/SESSION.md:157-173, :374-378 | Any penult shift statistic matches H. It was never compared with ATC, batch confidence or disagreement. |
| LH-2 (batch-level h) | the task is known; no controller-grade test yet | R5, R6, R7, R8, R16, M7, R9 | results/anomaly_h1/SESSION.md:318-337, :387-392, :480, :515 | Missing: FPR on benign shifts, delay on ramps, resolution within a split, and the head baselines (batch-mean maxprob, BBSDs p-value, ATC). |
| LH-3 (energy hold rule) | consistent with the literature | R1, R8 | results/anomaly_h1/SESSION.md:171, :331 | Refuted in Atlas. The literature separates shift magnitude from harm, and nothing in it supports a hold rule based on norms alone. |
| LH-4 (brightness → HOLD) | the harm grading fits the benign-shift frame; the mechanism was refuted | R1, R8, T2 | results/anomaly_h1/SESSION.md:197-214 | Not compared with BN-statistics adaptation on brightness, or with output-based harm estimates. |
| DO-4 (far-OOD flag → do not adapt) | known in research as TTA practice | T18, T5, T9 | results/anomaly_h1/SESSION.md:332 (AX-3 INFO) | TTA methods already keep high-entropy and OOD samples out of the update. The open question is whether h beats entropy filtering. |

---

## 6. Baselines Atlas should include next time

**A. Shift detection and type (ST-1, ST-2, ST-3, ST-8)**
1. **BBSDs and BBSDh.** Run BBSDs as per-class KS tests on the softmax outputs with a Bonferroni correction, and BBSDh as
   a chi-squared test on the predicted-label histogram [R1].
   - This needs logits. Either drop the probe's refusal of logits for this audit (docs/plans/ANOMALY_H1.md:585-588), or
     recompute logits as penult·W + b from the existing dumps.
2. **MMD with a permutation test** at the penult and at the pre-collapse tap [R1]. Also run it on an untrained-network
   embedding; the random-init nulls Atlas already has can serve as R1's UAE-style control.
3. **Two type routers on the same batches:**
   - a router built on the predicted histogram or logit shift;
   - a DPCore-style router on the batch mean and standard deviation [T14].

   Score each by the downstream benefit of the response it routes to, not by routing accuracy.
4. **Covariate vs label (T_par's job).** Compare with BBSE's prior estimate [R2] and with R3's test after
   prevalence-adjusted resampling. Report identification accuracy on pure-skew, pure-covariate and mixed batches.
5. **Detection statistics.** Report detection power against batch size (8, 16, 32, 64, 128) at a fixed α. Add a
   null-contrasted holdout and cross-seed replication.

**B. Gate 3 (ST-5)**

6. **Change detectors to compare** (the fourth is the legacy Gate 3 rule):
   - CUSUM on predicted-class frequencies [R10];
   - a BBSDh chi-squared test per window [R1];
   - a change in BBSE's prior estimate [R2];
   - the legacy acceleration statistic;
   - a peak-velocity (first difference) baseline.

   Report the false-alarm rate (or ARL₀) against detection delay over seeded replicate streams [R5, R9].

**C. TTA monitoring (ST-6)**

7. **Collapse triggers to compare:**
   - pred_entropy, the current bar;
   - SAR's moving average of the entropy loss (factor 0.9, e0 = 0.2) [T9];
   - ASR's prediction concentration against its moving average [M5];
   - the gradient norm of the TTA loss [T9];
   - AETTA's estimated accuracy [M3];
   - PeTTA's class-mean Mahalanobis divergence [M2].

   The geometric claim to test becomes: merge_tau, adjacency_rho or d_nc1 lead PeTTA's γ and ASR's trigger at a matched
   false-onset rate on the benign run.
8. **Policies to compare:**
   - never adapt;
   - BN statistics only [T2];
   - TENT;
   - EATA;
   - SAR with recovery;
   - RDumb periodic reset [M1];
   - reset triggered by AETTA;
   - output-gated;
   - geometry-gated.
9. **Protocols:**
   - a continual stream in the style of CCC [M1];
   - recurring domains [M2];
   - temporally correlated and class-imbalanced streams [T8, T10, T9];
   - batch sizes from 1 to 64 [T9];
   - a compute-constrained stream [M11];
   - label-free hyperparameter selection [M9, M10].
10. **Onset statistics.**
    - Before comparing lead times on the collapse run, calibrate every monitor to the same false-onset rate on the
      benign run (the fog3 manifest).
    - Measure all monitors in the same model state (deployed or affine), or report both.
    - Report the lead in steps, with bootstrap confidence intervals over stream seeds.

**D. Harm and adapt/hold (LH-1, LH-2, DO-4)**

11. **Harm baselines:**
    - batch-mean maxprob;
    - batch entropy;
    - ATC [R16];
    - the BBSDs p-value [R1];
    - disagreement, as in R8 or M3's dropout disagreement;
    - a learned error proxy [R6].

    Metrics: the true-positive rate at a fixed false-positive rate on benign batches (loss ≤ 2 pt), detection delay on
    ramps, and Spearman within a split.
12. **For DO-4.** Use an EATA-style entropy filter (0.4·ln C) and STAMP-style filtering [T18] as the "do not adapt on
    novelty" baselines.

---

## 7. Implications for the controller

1. **The adapt action is well studied and cheap.** The controller's value lies in deciding when to adapt, when to reset
   and when to hold. Start from BN-statistics adaptation [T1-T3] with a reset policy, and treat TENT, EATA and SAR as
   heavier adapters. Periodic reset [M1] is the floor any learned trigger must beat.
2. **ST-6 is this topic's cleanest test of the primary question.** Every published "stop or undo adaptation" trigger
   except PeTTA reads the head.
   - The test is the TENT ladder with the output triggers (pred_entropy, SAR, ASR, AETTA, gradient norm) and PeTTA's
     divergence as baselines, run under the fairness fixes in K5.
   - If Atlas's topology metrics do not lead PeTTA and ASR at a matched false-onset rate, geometry adds nothing new for
     collapse monitoring.
3. **Per-sample HOLD inside an adaptation step is standard practice** [T5, T9, T18]. A geometric per-sample filter would
   have to beat entropy filtering, and AX-4 already lost to plain d1.
4. **The literature distinguishes three "do not keep adapting" responses:**
   - (a) do not update: the shift is harmless, or the batch is unreliable (skewed, OOD, too small);
   - (b) reset or roll back: the adaptation is diverging [T9, M1, M3, M5];
   - (c) escalate or take offline: a risk bound has been violated [R5, M7].

   (a) is HOLD proper, which the evaluation treats as one action with two triggers, "output trusted" and "output
   distrusted" (`docs/reviews/EVAL_2026-09-23.md` §6, correction 14). (b) and (c) are distinct actions that the
   controller specification must name separately.
5. **Label shift and class skew must gate adaptation** [T9, T11, T12, T13]. The standard detector reads the head (BBSE or
   BBSDh). T_par (ST-3) could matter only if it beats BBSE at small batch sizes, and it was never compared.
6. **The benefit of adapting is the missing decision variable.** AETTA [M3] and agreement-on-the-line [M4] give
   label-free estimates of the adapted model's accuracy, so a controller can compare estimated accuracy before and after
   adapting. H predicts the harm a shift does, not the benefit of adapting to it.
7. **Build shift-type routing only if it pays in closed loop.** Type-specific responses must beat a single BN-plus-reset
   policy. R15 shows that the right response depends on shift type; T14 shows that a routed coreset helps on recurring
   domains. AX-2b's 1.000 routing, which random nets also reach, is not evidence of that.
8. **Escalation (Gate 3) should be a sequential test with a stated false-alarm rate** [R5, R6, M7, R9, R10]. Re-validate
   the legacy acceleration statistic against CUSUM and BBSDh on seeded streams.
9. **Robot relevance.** TTA fails on temporally correlated streams, recurring environments and small batches [T8, T10,
   M2, T9]. R9 provides an online detector for robot inputs with a false-positive guarantee. Closed-loop v0 should
   include correlated frames and recurrence.
10. **Beyond-head summary for this topic:**
    - shift detection: outputs are the baseline (BBSDs);
    - identification: features help for subtle covariate shifts [R3] and for domain signatures [T14];
    - harm and accuracy: outputs and disagreement;
    - collapse: outputs, with one feature-geometric method (PeTTA).

    Atlas has not yet tested any of its shift-type or TTA items against a head baseline. The ANOMALY_H1 probe excluded
    logits by design (docs/plans/ANOMALY_H1.md:585-588), and the TTA ladder has never run.

---

## 8. Item classifications (this topic's reading)

The merged, conflict-resolved classification is in `docs/knowledge/README.md`. This table keeps this topic's reading
and citations.

| item | known? | key citations | how prior work uses it |
|---|---|---|---|
| ST-1 | known-in-research | Zhang, Mehra, Niu, Hamm 2025, DPCore, ICML (arXiv 2406.10737); Lee, Yoon, Hwang 2024, BECoTTA (arXiv 2402.08712, venue unconfirmed); Roschewitz et al. 2024 (arXiv 2411.07940); Rabanser, Günnemann, Lipton 2019, NeurIPS (arXiv 1810.11953) | In continual TTA, the batch's feature statistics (the mean and standard deviation of the CLS feature in DPCore) are matched to stored domain signatures. The match routes the batch to a prompt, expert or domain, and a distance-ratio threshold of 0.8 creates a new domain. Routers are judged by downstream error, not routing accuracy. Atlas's AX-2b (whitened mean-shift templates at the pre-collapse tap) is the same idea. It reached 1.000, but random-init nets route 3 of the 4 families, and no head-based router was compared, because the probe refuses logits. |
| ST-2 | partly-known | Lee Y. et al. 2023, Surgical Fine-Tuning, ICLR (arXiv 2210.11466); Lee K. et al. 2018, NeurIPS, multi-layer Mahalanobis (arXiv 1807.03888); Sastry & Oore 2019, Gram matrices (arXiv 1912.12510); Sahoo et al. 2025, GALA, AAAI (arXiv 2404.03784) | The literature uses layer location to choose which layers to adapt (corruptions are best handled by tuning early layers) and ensembles low-level layers in detectors. A per-tap drift-onset profile as a readout of "what changed where" was not found. Atlas's AX-2a gate is CLOSED, decided by 0.0017 AUC, inside sampling noise, so it stays an unconfirmed proposal. |
| ST-3 | partly-known | Lipton, Wang, Smola 2018, BBSE, ICML (arXiv 1802.03916); Roschewitz et al. 2024 (arXiv 2411.07940); Maia Polo et al. 2023, DetectShift, Information Sciences (arXiv 2205.08340); Niu et al. 2023, SAR, ICLR (arXiv 2302.12400); Zhao, Chen, Xia 2023, DELTA (arXiv 2301.13018); Su et al. 2024, TEMA, AAAI (arXiv 2312.09486) | The goal of separating covariate from label shift, and making detection immune to class skew, is known and needed, because class-imbalanced batches break BN-based TTA. The standard tools read the head: BBSE prior estimation, then R3's prevalence-adjusted re-test. Projecting onto the complement of the class-mean span was not found in prior art, and it was ruled out in Atlas: c3 failed by ceiling, and the single-class FPR failed through a class-4 test-mean leak. T_par was never compared with BBSE or BBSDh. |
| ST-4 | not-found-in-prior-art | Nearest analogue: Lee K. et al. 2018, NeurIPS (arXiv 1807.03888), class-conditional Mahalanobis per-sample score | No published method uses the share of a sample's offset from its predicted centre that lies off the class span. Atlas ruled it out: gains over d1 are −0.16 to −0.44, and still −0.03 to −0.10 with the orientation flipped. Plain distance-type per-sample scores remain the reference. |
| ST-5 | known-in-research | Rabanser et al. 2019, NeurIPS (BBSDh, chi-squared test on predicted labels; arXiv 1810.11953); Lipton et al. 2018, ICML (arXiv 1802.03916); Wu, Guo, Su, Weinberger 2021 (arXiv 2107.04520); Page 1954, Biometrika 41:100-115 (CUSUM; doi 10.1093/biomet/41.1-2.100); Luo et al. 2022-2024, recency prediction (arXiv 2211.09916); Podkopaev & Ramdas 2022, ICLR (arXiv 2110.06177) | The predicted-label distribution is monitored with two-sample or sequential tests that control false alarms, and label-shift estimates drive correction or alerts. The legacy second-difference "acceleration" is an ad-hoc change detector with no FPR or delay figures, and it is a head signal, not geometry. It should be re-validated against CUSUM, BBSDh and BBSE on seeded streams. |
| ST-6 | partly-known | Hoang, Vo, Do 2024, PeTTA, NeurIPS (arXiv 2311.18193); Niu et al. 2023, SAR, ICLR (arXiv 2302.12400); Lim, Hwang, Lee 2026, ASR, ICLR (arXiv 2603.03796); Lee, Chottananurak, Gong, Lee 2024, AETTA, CVPR (arXiv 2404.01351); Press et al. 2023, RDumb, NeurIPS (arXiv 2306.05401); Chen et al. 2026, NCTTA, CVPR (arXiv 2512.10421); Kim et al. 2024, NeurIPS (arXiv 2310.04941); Schirmer et al. 2025 (arXiv 2507.08721) | The literature senses and undoes TTA collapse mostly from outputs: SAR resets when the moving average of the entropy loss falls below 0.2; ASR resets on prediction concentration above its moving average; AETTA resets on drops in its dropout-disagreement accuracy estimate; RDumb resets periodically. PeTTA uses a feature-geometric sensor, the Mahalanobis divergence of pseudo-label class means from the source means, to modulate regularisation. AETTA and agreement-on-the-line estimate the benefit of adapting without labels. Atlas's topology metrics (merge_tau, adjacency_rho, d_nc1) as leading indicators are not found in the literature. They are untested: the ladder was never run, and its bar is lenient (unequal onset rules; geometry in affine mode against pred_entropy in deployed mode). |
| ST-7 | partly-known | Lee K. et al. 2018, NeurIPS (arXiv 1807.03888); Rabanser et al. 2019, NeurIPS (arXiv 1810.11953); Sastry & Oore 2019 (arXiv 1912.12510) | Detecting adversarial inputs, OOD inputs and partial corruption are standard detection tasks, served by Mahalanobis and Gram baselines. R1 reports adversarial shifts as detectable and 10%-contaminated batches as hard. Persistence-type axes correspond to sequential tests. The legacy roster was never re-measured in Atlas against these baselines or head statistics. |
| ST-8 | known-in-research | Rabanser et al. 2019, NeurIPS (arXiv 1810.11953); Schneider et al. 2020, NeurIPS; Li et al. 2016, AdaBN (arXiv 1603.04779); Nado et al. 2020 (arXiv 2006.10963) | A global contrast or normalisation mismatch is an ordinary covariate shift. BBSD and MMD tests target it, and BN-statistics adaptation largely absorbs it. Atlas found its own mismatch from the training log. The geometric change it showed (penult relrep argmax agreement 0.9745, results/atlas_v1_resnet20_s0hub/compare_vs_atlas_v0_resnet20_cifar10/deformation.json) was about the size of the head's argmax change (agreement 0.9724, results/norm_check_resnet20/norm_check.json), so there is no sign that geometry adds anything. |
| LH-1 | partly-known | Rabanser et al. 2019 (malignancy; arXiv 1810.11953); Podkopaev & Ramdas 2022, ICLR (arXiv 2110.06177); Amoukou et al. 2024, NeurIPS (arXiv 2412.12910); Ginsberg et al. 2023, Detectron, ICLR (arXiv 2212.02742); Nguyen et al. 2025, D3M, NeurIPS (arXiv 2506.05047); Garg et al. 2022, ATC, ICLR (arXiv 2201.04234); Kim et al. 2024, NeurIPS (arXiv 2310.04941) | Separating harmful from benign shift, and estimating accuracy without labels, is an established research task. It uses disagreement, confidence thresholds or learned error proxies. Atlas's H (normalised shift in kNN radius) is its own feature-distance proxy for harm. Any penult shift statistic matches it between splits, and it was never compared with ATC, batch confidence or disagreement. |
| LH-2 | partly-known | Podkopaev & Ramdas 2022, ICLR (arXiv 2110.06177); Amoukou et al. 2024, NeurIPS (arXiv 2412.12910); Schirmer et al. 2025 (arXiv 2507.08721); Luo et al. (arXiv 2211.09916); Nguyen et al. 2025, D3M (arXiv 2506.05047); Garg et al. 2022 (arXiv 2201.04234) | Batch-level harm monitoring is done with sequential tests that control false alarms and report delay. The per-batch h has a working HOLD band between splits, but its within-split Spearman is only 0.21-0.23. It has no stream or ramp test, no FPR-versus-delay curve and no head baseline (batch-mean maxprob, BBSDs p-value, ATC). |
| LH-3 | partly-known | Rabanser et al. 2019 (arXiv 1810.11953); Nguyen et al. 2025, D3M (arXiv 2506.05047); Amoukou et al. 2024 (arXiv 2412.12910) | The literature separates shift magnitude from harm and monitors harm-aligned proxies. Nothing in it supports a hold rule based on norms alone. Atlas's refutation (21-33% of batches with e ≥ 0.95 lose more than 10 pt) is consistent with that. |
| LH-4 | partly-known | Rabanser et al. 2019 (benign vs malignant shifts; arXiv 1810.11953); Schneider et al. 2020, NeurIPS (BN adaptation on corruptions); Nguyen et al. 2025, D3M (arXiv 2506.05047) | "Benign shift, so do not act" is the standard malignancy framing. Atlas's graded result for brightness fits it at depth-56 CIFAR. The proposed mechanism (a class-neutral stem direction) is Atlas's own and was refuted. The grading was not compared with output-based harm estimates or with BN-statistics adaptation. |
| DO-4 | known-in-research | Yu et al. 2024, STAMP, ECCV (arXiv 2407.15773); Niu et al. 2022, EATA, ICML (arXiv 2204.02610); Niu et al. 2023, SAR, ICLR (arXiv 2302.12400) | TTA methods keep OOD, high-entropy and unreliable samples out of the adaptation update: an entropy threshold of 0.4 ln C, low-entropy label-consistent memories, and large-gradient filters. "Do not adapt on semantic novelty" is therefore standard practice. Atlas's h flags every CIFAR-100 batch (INFO only), but it was never compared with entropy filtering. |
