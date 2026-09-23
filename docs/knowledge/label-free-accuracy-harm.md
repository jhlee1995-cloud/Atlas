# Label-free accuracy estimation and harmful-shift detection: prior art for Atlas

Last reviewed: 2026-09-23; part of the Atlas knowledge base (docs/knowledge/README.md)

Prior-art labels in this file (rediscovered, extends, known-in-research, not found in prior art, and so on) mark
starting points to adopt and refine for Atlas's use; they never down-rank an item (docs/knowledge/README.md §1).

- **Written against:** Atlas HEAD 343f651.
- **Topic items:** LH-1 to LH-4. Cross-references to DO, ST and CM items.

**How the sources were checked.** Every work in section 2 was opened (arXiv abstract page, ar5iv/HTML full text,
proceedings or documentation page) or appeared in search results with matching bibliographic data. Each entry says
which.

**Limits of the search.** The web-search budget ran out during the pass, so two leads were never searched:
- kNN- or Mahalanobis-based *accuracy* estimators, as opposed to kNN/Mahalanobis OOD detectors;
- recent benchmark surveys of unsupervised accuracy estimation.

"Not found" below therefore means "not found in the works opened". It is not an exhaustive negative.

## Summary for the primary question

- **Atlas's harm grade H is a known kind of estimator.** It is a robust (median kNN-radius) variant of the
  feature-distribution-distance estimators that the literature proposed first (AutoEval, Fréchet distance of
  penultimate features). The literature then reported them as weaker than confidence-based estimators outside synthetic
  corruptions.
- **The ρ ≈ 0.99 across CIFAR-10-C splits is the expected level for this literature.** Output-based and feature-based
  scores both reach it. It cannot show that geometry adds anything beyond the head.
  - In Atlas's own AX-3 data, 99% of the per-batch loss variance lies between splits: the within-split SD is about 2.1
    pt and the between-split SD about 20.5 pt, recomputed below. Any monotone shift statistic therefore scores near 1 on
    this test.
- **The baselines that could answer the question are nearly free.** Atlas already stores per-sample max-softmax for
  every split (atlas/extract_acts.py:171-188, :250-251), which is all that AC, DoC and ATC-MC need. The harm probe never
  reads it (scripts/anomaly_probe.py:302).
- **Status:** "H adds beyond the head" is UNTESTED. The literature gives no reason to expect it on synthetic corruptions.
  The regimes where it might add are:
  - shifts where the head is overconfident (noise, natural shift);
  - backbones without a class head.

## 1. What this family of methods measures

**Setting.** There are four inputs:
- labelled source data (train plus held-out in-distribution validation);
- a trained model;
- an unlabelled target sample, which may be a dataset, a chunk or a stream.

The methods aim at three different targets:

| target | question | output | usual name |
|---|---|---|---|
| accuracy estimate | what is the model's accuracy on this unlabelled target? | a number (or the drop from ID) | unsupervised accuracy estimation, AutoEval, OOD performance prediction |
| harm test | has accuracy dropped by more than ε? | an alarm, ideally with false-alarm control | harmful-shift detection, risk monitoring |
| shift test | has p(x) (or p(y)) changed at all? | a p-value | dataset-shift / drift detection |

Two facts organise the field:
- **A shift test is not a harm test.**
  - Rabanser et al. (2019) show a statistically significant shift that leaves accuracy unchanged.
  - Podkopaev & Ramdas (2022) define monitoring as alarming on harmful shift while ignoring benign shift.
- **Accuracy is unidentifiable without assumptions.** Garg et al. (2022) prove that, unless the shift is restricted,
  estimating target accuracy is as hard as finding the optimal target predictor. Every method therefore rests on an
  assumption, and its failure modes are the places where that assumption breaks.

**Signal families, with the assumption each rests on:**
1. **Output / confidence.** AC (mean max-softmax), DoC, ATC, entropy, nuclear norm, MaNo, COT, MDE, SoftmaxCorr; the
   production version is CBPE.
   - Assumption: confidence stays calibrated, or its threshold stays transferable, under the shift.
2. **Feature-distribution distance / feature geometry.**
   - Methods:
     - the Fréchet distance of penultimate features (AutoEval);
     - MMD or KS two-sample tests on features or softmax (BBSD);
     - kNN distance, as an OOD score;
     - the inter-class dispersion of pseudo-labelled features (Dispersion).
   - Assumption: a distance-to-accuracy map learned on some shifts transfers to other shifts.
3. **Agreement / disagreement.**
   - Methods:
     - two-run disagreement (GDE);
     - agreement-on-the-line (ALine);
     - self-training ensembles;
     - constrained-disagreement ensembles (Detectron);
     - D3M;
     - dropout disagreement (AETTA).
   - Assumption: ensembles are calibrated, or ID-to-OOD agreement trends are linear.
4. **Auxiliary task or retraining.** Rotation prediction, Projection Norm, GdScore (gradients). These are heavier and
   mostly incompatible with a gradient-free runtime (MASTER_SUMMARY.md:2-5).
5. **Sequential monitoring wrappers.** Confidence sequences over a risk proxy:
   - Podkopaev & Ramdas, with delayed labels;
   - Amoukou et al. 2024, label-free, with an error-estimator proxy;
   - Schirmer et al. 2025, for TTA;
   - PPRM, with a few labels.

**Where Atlas sits.**
- **H is a family-2 statistic.** It is the median shift of the penult 10-NN log-radius, normalised by the reference's
  q95-q50 spread (docs/plans/ANOMALY_H1.md:164, :724-753), and it is used as a harm grade with a HOLD band.
- **The per-sample confidence line (CM items) is the family-1 statistic.**
- **ATC connects the two levels.** It is exactly the per-sample max-softmax score, thresholded and averaged. So in the
  output family, the per-sample trust score and the batch harm grade are one statistic at two aggregation levels. The
  geometric counterpart, an "ATC on the class-mean margin", has not been built (section 6).

## 2. Key works

**Verification legend:**
- **[A]** abstract or landing page opened.
- **[F]** full text or HTML opened via fetch. Numbers are as read by the fetch tool and were not re-checked against the
  PDF.
- **[S]** bibliographic data from search results only.

### 2.1 Confidence / output-based estimators

- **Hendrycks, D. & Gimpel, K. (2017).** *A Baseline for Detecting Misclassified and Out-of-Distribution Examples in
  Neural Networks.* ICLR 2017. https://arxiv.org/abs/1610.02136 **[A]**
  - Introduces MSP as the per-sample error and OOD score.
- **Guillory, D., Shankar, V., Ebrahimi, S., Darrell, T. & Schmidt, L. (2021).** *Predicting with Confidence on Unseen
  Distributions.* ICCV 2021, pp. 1134-1144. https://arxiv.org/abs/2107.03315 **[A]**
  - Proposes DoC, the difference of confidences.
  - Reports that Fréchet distance and MMD do not give reliable accuracy estimates.
  - DoC cuts prediction error by about 46% on ImageNet-Vid-Robust and ImageNet-R.
- **Garg, S., Balakrishnan, S., Lipton, Z. C., Neyshabur, B. & Sedghi, H. (2022).** *Leveraging Unlabeled Data to
  Predict Out-of-Distribution Performance.* ICLR 2022. https://arxiv.org/abs/2201.04234 **[F]**
  - **ATC:**
    - fit a threshold t on held-out source validation data, so that the fraction of scores below t equals the source
      error;
    - predicted target accuracy is the fraction of target scores above t;
    - the score is the max softmax (ATC-MC) or the negative entropy (ATC-NE).
  - Baselines: AC, DoC, IM, GDE.
  - Reported results:
    - 2-4× lower error than prior methods;
    - CIFAR-10 MAE ≈ 2.9 (natural shifts) / 3.9 (synthetic) for ATC-NE after temperature scaling;
    - larger error on BREEDS novel subpopulations: ATC-NE after temperature scaling, MAE 6.6-18.3 across the four
      BREEDS tasks (18.3 on NONLIVING-26), against 4.2-10.3 on the same subpopulations.
  - Theory:
    - accuracy is not identifiable without assumptions on the shift;
    - no single method handles both covariate and label shift.
- **Ovadia, Y., Fertig, E., Ren, J., Nado, Z., Sculley, D., Nowozin, S., Dillon, J. V., Lakshminarayanan, B. & Snoek,
  J. (2019).** *Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift.* NeurIPS
  2019. https://arxiv.org/abs/1906.02530 **[A]**
  - Calibration degrades as shift grows, and post-hoc calibration falls short.
  - This is the mechanism by which AC overestimates accuracy under shift.
- **Deng, W., Suh, Y., Gould, S. & Zheng, L. (2023).** *Confidence and Dispersity Speak: Characterising Prediction
  Matrix for Unsupervised Accuracy Estimation.* ICML 2023, pp. 7658-7674. https://arxiv.org/abs/2302.01094 **[S]**
  - Uses the nuclear norm of the softmax prediction matrix, which captures both confidence and dispersity across
    classes.
- **Lu, Y. et al. (2023).** *Characterizing Out-of-Distribution Error via Optimal Transport.* NeurIPS 2023.
  https://arxiv.org/abs/2305.15640 **[S]**
  - Proposes COT.
  - Names pseudo-label shift as the reason confidence methods underestimate error.
- **Xie, R., Odonnat, A., Feofanov, V., Deng, W., Zhang, J. & An, B. (2024).** *MaNo: Exploiting Matrix Norm for
  Unsupervised Accuracy Estimation Under Distribution Shifts.* https://arxiv.org/abs/2405.18979 **[A]**
  - Venue not confirmed.
  - Uses a norm of the normalised-logit matrix, aimed at the overconfidence bias under natural shift.
- **Peng, R., Zou, H., Wang, H., Zeng, Y., Huang, Z. & Zhao, J. (2024).** *Energy-based Automated Model Evaluation.*
  ICLR 2024. https://arxiv.org/abs/2401.12689 **[A]**
  - Proposes MDE.
- **Tu, W., Deng, W., Zheng, L. & Gedeon, T. (2024).** *What Does Softmax Probability Tell Us about Classifiers Ranking
  Across Diverse Test Conditions?* TMLR 2024. https://arxiv.org/abs/2406.09908 **[A]**
  - Proposes SoftmaxCorr, for ranking models on unlabelled OOD sets.
- **NannyML.** *Confidence-based Performance Estimation (CBPE)*, documentation.
  https://nannyml.readthedocs.io/en/stable/how_it_works/performance_estimation.html **[F]**
  - A production tool. Its stated assumptions:
    - calibrated probabilities;
    - no concept drift;
    - no covariate shift into unseen regions;
    - enough samples per chunk.

### 2.2 Feature-distribution and feature-geometry estimators

- **Deng, W. & Zheng, L. (2021).** *Are Labels Always Necessary for Classifier Accuracy Evaluation?* CVPR 2021, pp.
  15069-15078. https://arxiv.org/abs/2007.02915 **[F, via ar5iv]**
  - AutoEval:
    - builds a meta-set of synthetically transformed datasets;
    - the Fréchet distance between the train-set and test-set penultimate features correlates with accuracy (Spearman
      ≈ −0.91 on the meta-set);
    - fits a linear regression on FD, and a network regression on the feature mean, the covariance and FD.
  - Stated limitation: test conditions outside the meta-set's coverage.
- **Xie, R., Wei, H., Feng, L., Cao, Y. & An, B. (2023).** *On the Importance of Feature Separability in Predicting
  Out-Of-Distribution Error.* NeurIPS 2023. https://arxiv.org/abs/2303.15488 **[F, via ar5iv]**
  - **Dispersion score:** the log of the size-weighted spread of the pseudo-label class centroids of penultimate
    features around their global mean.
  - **A large domain gap need not mean low accuracy.** Two CIFAR-10-C sets with FD ≈ 223-224 have accuracies of about
    61% and 47%.
  - **CIFAR-10-C results:**
    - Dispersion: R² 0.972, Spearman 0.990;
    - ProjNorm: R² 0.947, Spearman 0.987.
  - Inter-class dispersion tracks OOD accuracy; intra-class compactness does not.
- **Sun, Y., Ming, Y., Zhu, X. & Li, Y. (2022).** *Out-of-Distribution Detection with Deep Nearest Neighbors.* ICML
  2022. https://arxiv.org/abs/2204.06507 **[A]**
  - Uses the k-th-NN distance on L2-normalised penultimate features, with the threshold set on ID data so that 95% of it
    is retained.
  - This is per-sample OOD detection, not accuracy estimation.
  - Atlas's knn_density is this detector without the normalisation.
- **Rabanser, S., Günnemann, S. & Lipton, Z. C. (2019).** *Failing Loudly: An Empirical Study of Methods for Detecting
  Dataset Shift.* NeurIPS 2019. https://arxiv.org/abs/1810.11953 **[F, via ar5iv]**
  - Compares dimensionality reduction plus two-sample tests. BBSD on softmax outputs with univariate KS tests works best.
  - Large shifts are detectable from about 20 samples.
  - **Detection ≠ harm:** a COIL-100 rotation is a significant shift with the same accuracy.
  - "Malignancy" is assessed with a domain classifier plus labels on the most-shifted samples.
- **Lipton, Z. C., Wang, Y.-X. & Smola, A. J. (2018).** *Detecting and Correcting for Label Shift with Black Box
  Predictors.* ICML 2018, pp. 3122-3130. https://arxiv.org/abs/1802.03916 **[S]**
  - BBSE: estimates label shift from a black-box predictor's confusion matrix.
- **Chen, L., Zaharia, M. & Zou, J. (2022).** *Estimating and Explaining Model Performance When Both Covariates and
  Labels Shift.* NeurIPS 2022. https://arxiv.org/abs/2209.08436 **[S]**
  - Proposes SEES, for sparse joint shift.

### 2.3 Agreement / disagreement estimators

- **Miller, J. P., Taori, R., Raghunathan, A., Sagawa, S., Koh, P. W., Shankar, V., Liang, P., Carmon, Y. & Schmidt,
  L. (2021).** *Accuracy on the Line: on the Strong Correlation Between Out-of-Distribution and In-Distribution
  Generalization.* ICML 2021, PMLR 139:7721-7735. https://proceedings.mlr.press/v139/miller21b.html **[S]**
- **Jiang, Y., Nagarajan, V., Baek, C. & Kolter, J. Z. (2022).** *Assessing Generalization of SGD via Disagreement.*
  ICLR 2022 (spotlight). https://arxiv.org/abs/2106.13799 **[A]**
  - The disagreement of two SGD runs on unlabelled data approximates the test error (GDE).
  - The explanation is class-aggregated calibration of SGD ensembles.
- **Kirsch, A. & Gal, Y. (2022).** *A Note on "Assessing Generalization of SGD via Disagreement".*
  https://arxiv.org/abs/2202.01851 **[A]**
  - Venue not confirmed.
  - Ensemble calibration can deteriorate as disagreement grows.
- **Baek, C., Jiang, Y., Raghunathan, A. & Kolter, J. Z. (2022).** *Agreement-on-the-Line: Predicting the Performance
  of Neural Networks under Distribution Shift.* NeurIPS 2022, 35:19274-19289. https://arxiv.org/abs/2206.13089 **[F,
  via ar5iv]**
  - ALine-S and ALine-D need a set of models (ALine-D at least three), ID labels and unlabelled OOD data. They use
    probit-scaled linear fits.
  - They lose accuracy (MAE about 5%) where accuracy-on-the-line fails (e.g., Camelyon17-WILDS, iWildCam-WILDS), as do
    the other methods compared. Whether agreement is on the line can be checked from unlabelled data.
- **Chen, J., Liu, F., Avci, B., Wu, X., Liang, Y. & Jha, S. (2021).** *Detecting Errors and Estimating Accuracy on
  Unlabeled Data with Self-training Ensembles.* NeurIPS 2021. https://arxiv.org/abs/2106.15728 **[S]**
  - Joint error detection and accuracy estimation.
  - Reports more than 40% lower estimation error than Proxy Risk on Digits and CIFAR-10-C.
- **Rosenfeld, E. & Garg, S. (2023).** *(Almost) Provable Error Bounds Under Distribution Shift via Disagreement
  Discrepancy.* NeurIPS 2023. https://arxiv.org/abs/2306.00312 **[S]**
  - Gives non-vacuous upper bounds on error.
- **Ginsberg, T., Liang, Z. & Krishnan, R. G. (2023).** *A Learning Based Hypothesis Test for Harmful Covariate Shift.*
  ICLR 2023. https://arxiv.org/abs/2212.02742 **[A]**
  - The venue comes from the search listing (ICLR 2023 slides).
  - Proposes the Detectron, built on constrained disagreement classifiers. It is strongest when the target sample is
    small.
- **Nguyen, V., Shui, C., Giri, V., Arya, S., Verma, A., Razak, F. & Krishnan, R. G. (2025).** *Reliably Detecting
  Model Failures in Deployment Without Labels.* NeurIPS 2025 (per arXiv listing). https://arxiv.org/abs/2506.05047
  **[A]**
  - D3M: a disagreement-based monitor for post-deployment deterioration.
  - It keeps false positives low under non-deteriorating shifts.

### 2.4 Auxiliary-task and retraining estimators

- **Deng, W., Gould, S. & Zheng, L. (2021).** *What Does Rotation Prediction Tell Us about Classifier Accuracy under
  Varying Testing Environments?* ICML 2021, pp. 2579-2589. https://arxiv.org/abs/2106.05961 **[S]**
- **Yu, Y., Yang, Z., Wei, A., Ma, Y. & Steinhardt, J. (2022).** *Predicting Out-of-Distribution Error with the
  Projection Norm.* ICML 2022, PMLR 162:25721-25746. https://arxiv.org/abs/2202.05834 **[S]**
- **Xie, R., Odonnat, A., Feofanov, V., Redko, I., Zhang, J. & An, B. (2024).** *Leveraging Gradients for Unsupervised
  Accuracy Estimation under Distribution Shift.* https://arxiv.org/abs/2401.08909 **[A]**
  - Venue not confirmed.
  - GdScore uses the norm of the classification-layer gradient. It needs gradients, so it falls outside a gradient-free
    controller.

### 2.5 Sequential harm monitoring

- **Podkopaev, A. & Ramdas, A. (2022).** *Tracking the risk of a deployed model and detecting harmful distribution
  shifts.* ICLR 2022. https://arxiv.org/abs/2110.06177 **[A]**
  - Uses time-uniform confidence sequences on the risk.
  - Needs labels, which may arrive delayed.
  - Alarms on harmful shift with a controlled false-alarm rate.
- **Amoukou, S. I., Bewley, T., Mishra, S., Lecue, F., Magazzeni, D. & Veloso, M. (2024).** *Sequential Harmful Shift
  Detection Without Labels.* NeurIPS 2024 (per arXiv listing). https://arxiv.org/abs/2412.12910 **[A]**
  - Replaces labels with a trained error-estimator proxy.
- **Zhang, G., Cai, Y., Yu, G. & Simeone, O. (2026).** *Prediction-Powered Risk Monitoring of Deployed Models for
  Detecting Harmful Distribution Shifts.* ICML 2026 (per arXiv listing). https://arxiv.org/abs/2602.02229 **[A]**
  - Combines a few true labels with synthetic labels to give anytime-valid bounds.

### 2.6 Test-time adaptation: estimation, gating and recovery

- **Wang, D., Shelhamer, E., Liu, S., Olshausen, B. & Darrell, T. (2021).** *Tent: Fully Test-time Adaptation by
  Entropy Minimization.* ICLR 2021. https://arxiv.org/abs/2006.10726 **[A]**
- **Niu, S., Wu, J., Zhang, Y., Wen, Z., Chen, Y., Zhao, P. & Tan, M. (2023).** *Towards Stable Test-Time Adaptation in
  Dynamic Wild World.* ICLR 2023. https://arxiv.org/abs/2302.12400 **[S]**
  - Proposes SAR.
  - TTA fails under mixed shifts, small batches and online imbalanced label shift; batch norm is a key source of
    instability.
  - Its model-recovery rule was not re-read here (see `shift-type-tta-monitoring.md`, which read the full text).
- **Zhao, H., Liu, Y., Alahi, A. & Lin, T. (2023).** *On Pitfalls of Test-Time Adaptation.* ICML 2023.
  https://arxiv.org/abs/2306.03536 **[S]**
  - Introduces the TTAB benchmark. Three pitfalls:
    - hyper-parameter and model selection is hard, because online batches are dependent;
    - the benefit of TTA depends on the base model;
    - no method handles all shift types.
- **Lee, T., Chottananurak, S., Gong, T. & Lee, S.-J. (2024).** *AETTA: Label-Free Accuracy Estimation for Test-Time
  Adaptation.* CVPR 2024, pp. 28643-28652. https://arxiv.org/abs/2404.01351 **[A]**
  - Estimates accuracy from the disagreement between the prediction and dropout inferences, re-weighted by batch softmax
    entropy to handle adaptation failures.
  - Estimates are 19.8 percentage points more accurate on average than the baselines.
  - Includes a model-recovery case study.
- **Kim, E., Sun, M., Baek, C., Raghunathan, A. & Kolter, J. Z. (2024).** *Test-Time Adaptation Induces Stronger
  Accuracy and Agreement-on-the-Line.* NeurIPS 2024. https://arxiv.org/abs/2310.04941 **[A]**
  - TTA strengthens ACL and AGL, including on shifts where they were weak (CIFAR-10-C Gaussian noise is named).
  - Uses this to estimate post-adaptation accuracy and to select TTA hyper-parameters without labels.
- **Schirmer, M., Jazbec, M., Naesseth, C. A. & Nalisnick, E. (2025).** *Monitoring Risks in Test-Time Adaptation.*
  https://arxiv.org/abs/2507.08721 **[A]**
  - Venue not confirmed.
  - Label-free sequential risk monitoring during TTA.
- **Wang, L., Li, J., Sun, X., Hu, X., Gu, Z., Liu, J., Nelakuditi, S. & Tong, Y. (2026).** *Should This Case Be
  Adapted? Prediction Fragmentation Controls Test-Time Adaptation.* https://arxiv.org/abs/2609.20700 **[A]**
  - Very recent, venue unknown; medical-segmentation setting.
  - A case-level, label-free router that decides whether to adapt.
  - Reports that many cases worsen under TTA even when the mean gain looks neutral.

### 2.7 Benchmark and a per-sample cross-reference

- **Hendrycks, D. & Dietterich, T. (2019).** *Benchmarking Neural Network Robustness to Common Corruptions and
  Perturbations.* ICLR 2019. https://arxiv.org/abs/1903.12261 **[A]**
  - The source of the corruption families and five severities that CIFAR-10-C and ImageNet-C use.
- **Jiang, H., Kim, B., Guan, M. Y. & Gupta, M. (2018).** *To Trust Or Not To Trust A Classifier.* NeurIPS 2018.
  https://arxiv.org/abs/1805.11783 **[A]**
  - Trust Score: the agreement between the classifier and a modified nearest-neighbour classifier.

## 3. How it is used

### 3.1 Usage patterns

| pattern | what is estimated | representative methods | maturity |
|---|---|---|---|
| production model monitoring | accuracy per data chunk, without labels | CBPE (AC with calibration); drift tests (BBSD, MMD, KS) | standard for confidence-based estimation (CBPE verified); drift tests are common (tooling not verified here) |
| harmful-shift alarm / retrain trigger | "accuracy fell by more than ε", with false-alarm control | Podkopaev & Ramdas (labels), Amoukou et al. 2024, D3M, Detectron, PPRM | research |
| dataset-level accuracy estimation (AutoEval) | a model's accuracy on a new unlabelled set | ATC, DoC, AC, FD/AutoEval, ALine, Dispersion, nuclear norm, MaNo, COT, ProjNorm | research; ATC/DoC/AC are the de-facto baselines |
| model selection / ranking on an unlabelled target | which model or checkpoint is best on the target | ACL/ALine, SoftmaxCorr, AutoEval scores | research |
| TTA gating, hyper-parameter selection, reset | the adapted model's accuracy; when to stop, reset or not adapt | AETTA, ALine after TTA (Kim et al.), Schirmer et al., SAR, case-level routers | research |
| label-shift detection / correction | change in p(y) | BBSE / BBSD on the predicted-class distribution | research, well established |
| per-sample selective prediction | whether one prediction is wrong | MSP, Trust Score (ATC is the dataset-level aggregate of MSP thresholding) | standard (MSP) |

### 3.2 Metrics

- **Accuracy estimation:**
  - mean absolute error (MAE) between estimated and true accuracy across target sets;
  - R² and Spearman ρ of score against accuracy across sets (AutoEval style);
  - results reported per shift type: synthetic corruption, dataset reproduction, natural or subpopulation shift.
- **Harm detection:**
  - detection power at a fixed false-alarm rate;
  - false alarms under benign shift;
  - detection delay in a stream;
  - sample complexity (how many target samples are needed).
- **TTA:**
  - estimation error for the adapted model's accuracy;
  - the downstream gain when the estimate drives recovery or selection.

### 3.3 Known failure modes

1. **Overconfidence under shift.** AC and CBPE overestimate accuracy once calibration degrades (Ovadia 2019; MaNo). COT
   names pseudo-label shift as the cause of the systematic under-estimation of error.
2. **Unidentifiability.** Every method fails when the shift violates its assumption (Garg 2022). ATC's error rises
   on novel subpopulations (BREEDS: MAE 6.6-18.3 against 4.2-10.3 on the same subpopulations).
3. **Covariate versus label shift.** Methods tuned for one fail on the other (Garg 2022). CBPE explicitly excludes
   concept drift and shift into unseen regions.
4. **Distance ≠ accuracy.**
   - FD and MMD give unreliable accuracy estimates across shift types (Guillory 2021).
   - Equal FD can come with very different accuracy (Xie 2023).
   - A detected shift can be harmless (Rabanser 2019).
   - Distance regressions fit on one synthetic meta-set do not transfer to unlike shifts (Deng & Zheng 2021, stated
     limitation).
5. **Linear-trend methods fail on some shifts.** ACL does not hold on Camelyon17 and iWildCam, and ALine's error rises
   there (MAE about 5%). Kim et al. 2024 name CIFAR-10-C noise as weak before TTA. The failure can be detected from
   unlabelled data.
6. **Disagreement methods depend on ensemble calibration,** which can itself degrade as disagreement grows (Kirsch & Gal
   2022).
7. **Small samples.** Every dataset-level estimator loses accuracy at small n (CBPE docs; Detectron targets this
   regime).
8. **TTA.** Dependent online batches break model selection (Zhao 2023). Mixed shifts, small batches and label imbalance
   break TTA itself (Niu 2023).

### 3.4 Standard practice versus research only

- **Standard:**
  - MSP as a per-sample trust score;
  - average confidence (CBPE) as a label-free performance estimate in monitoring;
  - two-sample drift tests;
  - calibrating thresholds on held-out ID data.
- **Research only:**
  - ATC, DoC, ALine;
  - AutoEval regressions, Dispersion;
  - nuclear norm, MaNo, COT;
  - sequential label-free harm monitors;
  - TTA accuracy estimation (AETTA) and TTA gating.
- **Convention on baselines:** AC, DoC and ATC are the baselines a new accuracy estimator must beat. Feature-distance
  estimators (FD, MMD) are reported as weaker than confidence-based ones on natural shifts.

## 4. Known results relevant to Atlas

1. **A between-split Spearman of about 0.99 on CIFAR-10-C is typical, for geometric and output-based scores alike.**
   - Literature:
     - Xie et al. 2023: 0.990 (Dispersion) and 0.987 (ProjNorm) on CIFAR-10-C;
     - AutoEval: |ρ| ≈ 0.91 for FD on its meta-set.
   - Atlas:
     - H reaches 0.989 / 0.987 (results/anomaly_h1/SESSION.md:167);
     - Atlas's own rule-8 check shows other penult shift statistics do as well: displacement magnitude 0.9996 / 0.995,
       sparse_frac 0.994 / 0.992 (:375-378).
   - Why: AX-3 was recomputed from results/anomaly_probe_resnet56_s{1,2}/probe.json
     `.dumps.atlas_v1_resnet56_s{1,2}.AX3.splits`. The mean within-split SD of the per-batch loss is 2.09 / 2.12 pt
     (range 0.55-3.34). The between-split SD of the split-mean loss is 20.5 pt. About 99% of the loss variance is
     therefore between splits, so any statistic that is monotone in shift size scores near 1.
   - Consequence: between-split rank correlation on synthetic corruptions cannot separate geometry from the head. The
     literature's discriminating tests are:
     - absolute error after a frozen calibration;
     - held-out shift families;
     - natural shifts;
     - label shift.
2. **Feature-distribution distance is a known predictor of accuracy, and a known weak one.**
   - The closest prior art to H is AutoEval: the distance between the train and test penultimate-feature
     distributions, regressed onto accuracy. DoC and Dispersion were proposed partly because FD and MMD fail across
     shift types.
   - H is a robust variant of the same idea (median kNN radius instead of Gaussian moments).
   - No work opened uses the median k-NN log-radius shift, normalised by the reference radius spread, as a harm grade
     with an explicit HOLD band.
   - Verdict: the idea is known and the specific statistic was not found in prior art. The literature's verdict on this
     family is unfavourable next to confidence-based estimators outside synthetic corruptions.
3. **Confidence-based estimators are the baseline to beat, and Atlas already stores their input.**
   - ATC-MC, AC and DoC need only the per-sample max-softmax.
   - Every Atlas CIFAR dump stores it for every split: preds/<split>.npz holds argmax and maxprob
     (atlas/extract_acts.py:171-188, :250-251).
   - The harm probe reads only argmax (scripts/anomaly_probe.py:302).
4. **Detection is not harm, in either direction.**
   - Rabanser's harmless but detectable shift is the literature's version of the HOLD-band problem.
   - Over-flagging: H (h > 0.25) flags the following batches (recomputed as above):

     | batch loss | s1 | s2 |
     |---|---|---|
     | 5-10 pt | 912/1091 | 931/1026 |
     | 2-10 pt | 1086/1599 | 1172/1585 |
     | ≤ 10 pt (the natural non-harmful split) | 1099/2649 | 1198/2746 |
     | ≤ 2 pt | 13/1050 | 26/1161 |

   - Under-flagging: the energy rule that H replaced (LH-3) is the other side.
5. **Noise corruptions are a known outlier family for label-free estimators.** Kim et al. 2024 name CIFAR-10-C Gaussian
   noise as a shift where ACL/AGL were weak. In Atlas:
   - density familiarity is non-monotone in severity for noise (DO-2);
   - most splits where the energy rule misses a > 10-pt cost are noise splits (s2: gaussian s1/s3/s5 and shot s3/s5;
     results/anomaly_h1/SESSION.md:170).
6. **Calibrating thresholds on held-out ID data is standard.**
   - ATC fits t on source validation data; deep-kNN sets its threshold on ID data.
   - AH-1 found that a train-reference density threshold over-alarms on clean test data, at 1.8-2.3× the nominal rate in
     resnet20 and 3.3-3.7× in resnet56 (recomputed; `density-ood.md` §5 DO-3; the plan quotes 2.2-3.3× from discovery,
     docs/plans/ANOMALY_H1.md:274-277). That is the expected consequence of calibrating on training data.
   - The log-nc1 law for the size of the over-alarm was not found in the works opened.
7. **Within-split batch resolution has a low ceiling for any estimator.**
   - Within one split, per-batch loss varies by about 2 pt, mostly from sampling. The 200 batches of 256 are drawn from
     2,000 rows, so they also overlap.
   - The literature's small-n caveats (CBPE, Detectron) say the same.
   - So Atlas's within-split Spearman of 0.21-0.23 (results/anomaly_h1/SESSION.md:387-392) should be read against head
     baselines computed on the same batches, not in absolute terms.
   - Ramps (graded severity inside a stream) are the informative within-stream test.
8. **Harm is not adaptation benefit.**
   - The TTA literature estimates the adapted model's accuracy (AETTA; ALine after TTA).
   - It shows that the benefit depends on the model and the shift, and can be negative for many cases (Zhao 2023; Wang
     2026).
   - A harm grade on the frozen model is therefore only half of the adapt/hold decision.
9. **Harm monitoring normally uses sequential false-alarm control:** confidence sequences in Podkopaev & Ramdas 2022,
   Amoukou et al. 2024 and Schirmer et al. 2025. AX-3 has only a static clean false-alarm count (0/200) and no stream
   test (results/anomaly_h1/SESSION.md:475-515).
10. **Label shift needs its own test.** BBSE/BBSD use the predicted-class distribution. H and ATC both assume covariate
    shift, and a class-skewed clean batch can move either of them.

## 5. Relation to Atlas items

| id | relation | literature anchor | Atlas evidence | note |
|---|---|---|---|---|
| LH-1 | **Rediscovered:** feature distance tracks accuracy across synthetic splits. **Extends:** a median kNN-radius statistic, per-backbone normalisation and a HOLD band. **Untested** against the head. | Deng & Zheng 2021; Xie et al. 2023; Guillory et al. 2021; Garg et al. 2022 | results/anomaly_h1/SESSION.md:157-173, :375-378; docs/plans/ANOMALY_H1.md:164, :279-310 | ρ 0.99 is the expected level (section 4.1). The only discriminating clauses are the HOLD band (b) and brightness (c), as Atlas's rule 8 already says. |
| LH-2 | **Extends** the grade to batch 256. **Untested** against chunk-level AC/ATC (CBPE-style). Has no sequential false-alarm control. | Garg 2022; NannyML CBPE; Podkopaev & Ramdas 2022; Amoukou 2024; Nguyen 2025 (D3M) | results/anomaly_h1/SESSION.md:318-337, :387-392; scripts/anomaly_probe.py:434-482 | The within-split 0.21-0.23 sits against a sampling-noise floor (section 4.7). It needs head baselines on the same batches. |
| LH-3 | **Agrees with the literature:** a shift statistic that does not move is not evidence of no harm. **Contradicts** the legacy MASTER_SUMMARY.md:38 ("keep"). | Rabanser 2019; Xie 2023 | results/anomaly_h1/SESSION.md:170, :331 | Keep as a recorded negative. |
| LH-4 | Grading brightness as benign fits the benign/harmful framing. The mechanism is refuted; no prior art was searched for the stem-direction mechanism. | Rabanser 2019; Podkopaev & Ramdas 2022 | results/anomaly_h1/SESSION.md:169, :197-214 | Brightness as a mild corruption in CIFAR-10-C / ImageNet-C results is commonly reported but was not re-verified here. |
| DO-1 | **Rediscovered:** kNN distance to training features grows with shift. **Untested** against the head. | Sun et al. 2022; Deng & Zheng 2021 | ATLAS_STATUS.md:15; results/atlas_v1_resnet20_s3/SESSION.md:209-212 | Per-split AC/ATC were never stored. |
| DO-2 | **Consistent with** noise being a known hard family for label-free estimators. | Kim et al. 2024 | ATLAS_STATUS.md:16; results/atlas_v0_resnet20_cifar10/SESSION.md:97-114 | — |
| DO-3 | The practice is **known** (calibrate on held-out ID data). The log-nc1 law was **not found in prior art**. | Garg 2022; Sun 2022 | results/anomaly_h1/SESSION.md:135-155; docs/plans/ANOMALY_H1.md:274-277 | — |
| DO-4 | **Standard** OOD detection. **Untested** against MSP or normalised kNN. | Hendrycks & Gimpel 2017; Sun 2022 | results/anomaly_h1/SESSION.md:332 | — |
| ST-3 | T_par is a geometric label-shift monitor; the head's analogue is BBSD/BBSE. | Lipton 2018; Rabanser 2019 | results/anomaly_h1/SESSION.md:272-291 | Never compared with the head. |
| ST-5 | The Gate-3 signal is a head statistic; the known comparators are BBSE/BBSD plus sequential tests. | Lipton 2018; Rabanser 2019; Podkopaev & Ramdas 2022 | MASTER_SUMMARY.md:11-17 | — |
| ST-6 | **Untested here.** The literature already has label-free accuracy estimation for adapted models and TTA gating/recovery. Atlas's bar (the pred_entropy onset) is weaker. | AETTA 2024; Kim 2024; Schirmer 2025; Zhao 2023; Niu 2023; Wang 2026 | scripts/tta_deform.py:15-20; ATLAS_README.md:152-173 | — |
| ST-8 | The standard way to flag it is a two-sample shift test. **Untested.** | Rabanser 2019 | results/atlas_v0_resnet20_cifar10/SESSION.md:122-127 | — |
| CM-1 | **Standard** MSP misclassification detection; ATC is its dataset-level aggregate. | Hendrycks & Gimpel 2017; Garg 2022 | results/margin_v1_resnet20_s1/SESSION.md:167-173 | — |
| CM-11 | Classifier-versus-neighbour disagreement is Trust Score's idea; error detection from ensemble disagreement is **known**. | Jiang et al. 2018; Chen et al. 2021; Jiang et al. 2022 | results/margin_b1_vitb16/SESSION.md:503-506 | **Untested** as a detector. |

## 6. Baselines Atlas should include next time

**Principle.** For every harm-grade row:
- compute output-based peers on the same splits and the same batches;
- fit them on the same discovery splits;
- score them on held-out corruption families.

**Tier 0: computable from the stored dumps without re-extraction.** This runs on the pod, because the dumps live on the
volume.
- **AC:** mean max-softmax per split or batch (preds/<split>.npz `maxprob`).
- **DoC:** AC(clean) − AC(S), as the predicted drop.
- **ATC-MC:**
  - fit t on clean half A, rows 2000-3499. This is the same clean anchor H uses for its median
    (scripts/anomaly_probe.py:81), so both estimators see identical calibration data;
  - predicted accuracy is the fraction of maxprob above t;
  - false alarms are measured on half B (rows 3500-4999).
- **Predicted-class histogram:** its shift from clean (BBSD style: total variation or chi-square), and its entropy,
  which is already the TTA ladder's bar.
- **GDE / ALine:** argmax agreement between seeds on the same paired split rows (resnet20 s1-s4 plus the hub; resnet56
  s1, s2, s12m, s13m and the hub). With three or more models this gives ALine-S and ALine-D.

**Tier 1: needs logits.** Recompute penult·W + b from the checkpoint on the pod, or add logits to preds at extraction.
- ATC-NE (entropy);
- mean energy (logsumexp);
- mean top-2 logit gap;
- nuclear norm (Deng 2023);
- MaNo;
- COT.

**Tier 2: feature-based peers of H** (same penult, same reference).
- FD / AutoEval, as a linear regression on penult features.
- The MMD two-sample statistic, or BBSD KS.
- The L2-normalised kNN median radius (Sun 2022 normalisation). This separates out the question of normalisation.
- The Dispersion score (Xie 2023): the spread of pseudo-label class centres of the target penult features. It can reuse
  the `class_centers` machinery.
- **A "geometric ATC":** ATC with the class-mean margin (the CM line) as the score instead of max-softmax. This is the
  harm-level analogue of the CM-4/CM-5 beyond-head test.

**Tier 3: reference only.** These need retraining or gradients, which conflicts with a gradient-free runtime: ProjNorm,
self-training ensembles, Detectron, GdScore, rotation prediction.

**Controls already in the record** (results/anomaly_h1/SESSION.md:375-378):
- severity alone: ρ 0.72 / 0.70;
- stem H: 0.19 / 0.26;
- random-init null: −0.40;
- paired displacement magnitude: 0.9996 / 0.995. It needs the clean twin of each image, so it is an oracle, not a
  runtime signal.

**Protocol (pre-register):**
1. **Split-level calibration.** Fit each score's map to accuracy (linear, or probit for ALine) on the 30 discovery
   splits, then freeze it. Report MAE and Spearman on the 15 confirmation-corruption splits: impulse, glass, zoom, frost,
   elastic (experiments/queue/atlas_v1_resnet56_s1.yaml:40-42).
2. **HOLD band at matched flag rate.**
   - For each score, use the threshold that flags the same fraction of batches as h > 0.25.
   - Compare misses (in-band batches losing more than 10 pt) and false flags (batches losing ≤ 2 pt that are flagged).
   - Also report the AUROC of harmful (> 10 pt) against benign (≤ 2 pt).
3. **Incremental value (the primary question).**
   - Use a nested model: does H add to ATC, or to DoC, in predicting cost? Test by partial Spearman or a paired
     bootstrap on MAE over splits, with an effect-size bar fixed in advance.
   - With about 30-45 splits per seed the power is limited. Use fresh seeds, because s1/s2 are spent for the AX axes
     (results/anomaly_h1/SESSION.md:505-515).
4. **Batch level.** Use the same batches for every score, and include:
   - ramps (severity changing within a stream);
   - mixed-corruption batches;
   - class-skewed clean batches (label shift);
   - batch sizes 16, 32, 64 and 256.
5. **Sequential.** Wrap the best two scores in a confidence-sequence or CUSUM monitor. Report false alarms on clean
   streams and the detection delay on ramps.
6. **Transfer.** Add one more backbone and one natural shift.
   - For example, an ImageNet model with ImageNet-C.
   - Or CIFAR-10.1, which would need extraction; no CIFAR-10.1 split exists in the current manifests.

## 7. Implications for the controller

1. **Use an output-based estimate as the default Gate-1 magnitude until geometry is shown to add.**
   - ATC or DoC on the head costs nothing, and it is label-free and gradient-free.
   - It fits the modulator's stated signal set (MASTER_SUMMARY.md:2-5).
   - H stays only as a candidate second axis, kept if the nested test in section 6 shows incremental value.
2. **Geometry could add only where the head is known to fail:**
   - overconfident shifts (noise, natural shift);
   - shifts that move the decision without moving confidence;
   - backbones with no class head.

   H needs only reference features, which matters for head-less encoders. But FD and MMD share that property, and the
   literature finds them unreliable on natural shift. That is the comparison to run.
3. **Harm is half the decision.**
   - The adapt/hold variable is the predicted benefit: the adapted model's estimated accuracy minus the frozen model's.
   - The known label-free way to get it is to run an accuracy estimator (ATC, ALine or AETTA) on both models.
   - The TTA ladder (ST-6) should record this next to its deformation metrics, keeping pred_entropy as the trivial
     floor.
4. **Use a sequential monitor with a false-alarm budget,** not a per-batch threshold. The h > 0.25 threshold has no
   time-uniform guarantee.
5. **Handle label shift separately,** with BBSE/BBSD on the predicted-class distribution. Do not read a class-skewed
   batch's H or ATC as covariate harm.
6. **Calibrate every threshold on held-out, deployment-clean data,** and re-calibrate after any adaptation. AH-1 and ATC
   practice agree on this, and TTA changes both the head's calibration and the feature geometry.
7. **Split HOLD into two triggers:**
   - "hold, output trusted": small estimated harm, e.g. brightness;
   - "hold, output distrusted": large harm or label shift, where adaptation is not expected to help.

   The accuracy estimators in this document address the first. The second also needs the benefit estimate from point
   3. HOLD is one action ("do not adapt") with two triggers (`docs/reviews/EVAL_2026-09-23.md` §6, correction 14).

## 8. Item classifications (this topic's reading)

The merged, conflict-resolved classification is in `docs/knowledge/README.md`. This table keeps this topic's reading
and citations.

| item | known? | key citations | how prior work uses it |
|---|---|---|---|
| LH-1 | partly-known | Deng & Zheng 2021, CVPR, arXiv 2007.02915 (FD of penultimate features vs accuracy, Spearman ~ −0.91 on a synthetic meta-set); Xie et al. 2023, NeurIPS, arXiv 2303.15488 (Dispersion ρ 0.990 on CIFAR-10-C; equal FD with 61% vs 47% accuracy); Guillory et al. 2021, ICCV, arXiv 2107.03315 (FD/MMD unreliable; DoC); Garg et al. 2022, ICLR, arXiv 2201.04234 (ATC); Rabanser et al. 2019, NeurIPS, arXiv 1810.11953 (detection is not harm) | In the literature: dataset-level label-free accuracy estimation (AutoEval) and model monitoring. Feature-distance versions are research only and weaker than confidence-based estimators on natural shift. Confidence versions ship in monitoring tools (CBPE). Atlas's median kNN-radius statistic with per-backbone normalisation and a HOLD band was not found as such. Its ρ 0.99 on CIFAR-10-C is the typical level for any shift statistic (99% of loss variance is between splits). It was never compared with AC, DoC or ATC, although maxprob is stored for every split. |
| LH-2 | partly-known | Garg et al. 2022 (ATC applies to any unlabelled set); NannyML CBPE docs (chunk-level confidence-based performance estimation); Podkopaev & Ramdas 2022, ICLR, arXiv 2110.06177 (harmful vs benign, confidence sequences); Amoukou et al. 2024, arXiv 2412.12910 (label-free sequential harmful-shift detection); Nguyen et al. 2025, arXiv 2506.05047 (D3M); Ginsberg et al. 2023, ICLR, arXiv 2212.02742 (Detectron, small samples) | In the literature: chunk or batch accuracy estimation in monitoring, and harmful-shift alarms with false-alarm control (sequential tests). In Atlas the per-batch grade has only a static clean false-alarm count and no stream test. Its within-split resolution (0.21-0.23) sits against a sampling-noise floor of about 2 pt, so it needs head baselines on the same batches. It predicts harm, not adaptation benefit. |
| LH-3 | known-in-research | Rabanser et al. 2019 (shift detection and harm are distinct); Xie et al. 2023 (equal feature distance, very different accuracy) | The literature treats any single shift-magnitude statistic as insufficient to certify no harm. Atlas's refutation of the legacy energy hold rule (21-33% of e ≥ 0.95 batches lose more than 10 pt) is consistent with that, and it contradicts MASTER_SUMMARY.md:38. Drop energy from any hold rule. |
| LH-4 | partly-known | Rabanser et al. 2019 and Podkopaev & Ramdas 2022 (benign vs harmful shift framing); Hendrycks & Dietterich 2019, ICLR, arXiv 1903.12261 (corruption benchmark; brightness mildness not re-verified here) | The literature uses benign-shift identification to avoid needless alarms and retraining. Atlas's brightness → HOLD grading fits that framing (at depth-56 CIFAR). The proposed stem-direction mechanism is refuted, and no prior art was searched for it. |
| DO-1 | known-in-research | Sun et al. 2022, ICML, arXiv 2204.06507 (deep kNN OOD on normalised penultimate features); Deng & Zheng 2021 (feature-distribution distance grows as accuracy falls on synthetic shifts) | kNN distance to training features is a standard OOD score, and feature-distribution distance is an AutoEval predictor. Atlas's severity monotonicity for non-noise corruptions is the expected behaviour. The head analogue (per-split AC/ATC) was never stored, so whether density adds is untested. |
| DO-2 | partly-known | Kim et al. 2024, NeurIPS, arXiv 2310.04941 (CIFAR-10-C Gaussian noise named as a shift where accuracy/agreement-on-the-line were weak) | Noise corruptions are a known hard family for label-free accuracy estimators. The seed-dependent noise non-monotonicity in Atlas is consistent with that. Density should not be used as a severity grade for high-frequency noise. |
| DO-3 | partly-known | Garg et al. 2022 (ATC threshold fit on held-out source validation); Sun et al. 2022 (kNN threshold set on ID data, 95% retained) | Calibrating detector thresholds on held-out ID data is standard practice. Atlas's over-alarm of train-reference thresholds is the expected consequence of calibrating on training data. The log-nc1 law that predicts the over-alarm size was not found in the works opened. |
| DO-4 | standard-practice | Hendrycks & Gimpel 2017, ICLR, arXiv 1610.02136 (MSP); Sun et al. 2022 (deep kNN) | OOD / novelty detection with MSP, energy, Mahalanobis or kNN scores is standard. Atlas measured it as INFO only, with no MSP or normalised-kNN baseline. It is a candidate escalate / do-not-adapt trigger. |
| ST-3 | partly-known | Lipton et al. 2018, ICML, arXiv 1802.03916 (BBSE label-shift estimation); Rabanser et al. 2019 (BBSD two-sample tests on softmax outputs) | Label shift is detected and estimated from the predicted-class distribution (a head statistic) in the literature. Atlas's T_par (class-span T²) is a geometric counterpart, never compared with BBSD/BBSE. T_perp (class-orthogonal covariate drift) was ruled out as registered. |
| ST-5 | known-in-research | Lipton et al. 2018 (BBSE); Rabanser et al. 2019 (BBSD with univariate/chi-square tests); Podkopaev & Ramdas 2022 (sequential monitoring with false-alarm control) | Detecting class-mix change from predicted labels is established research practice, with two-sample or sequential tests that report false-alarm rates and delays. The legacy Gate-3 density-acceleration evidence lacks those comparators and metrics. |
| ST-6 | partly-known | Lee et al. 2024, CVPR, arXiv 2404.01351 (AETTA: label-free accuracy of adapted model, model recovery); Kim et al. 2024, NeurIPS, arXiv 2310.04941 (ALine after TTA for accuracy estimation and hyper-parameter selection); Schirmer et al. 2025, arXiv 2507.08721 (risk monitoring in TTA); Zhao et al. 2023, ICML, arXiv 2306.03536 (TTA pitfalls); Niu et al. 2023, ICLR, arXiv 2302.12400 (SAR failure regimes); Wang et al. 2026, arXiv 2609.20700 (case-level adapt/not-adapt router) | The literature estimates the adapted model's accuracy label-free, and uses it to reset, select or gate TTA. Detecting adaptation failure through geometric deformation of the atlas was not found in prior art, and is untested. Its planned bar (onset before pred_entropy collapse) is weaker than the literature's accuracy-estimate comparators. |
| ST-8 | known-in-research | Rabanser et al. 2019 (two-sample shift tests on softmax or features) | The standard way to flag a pipeline mis-normalisation is a two-sample shift test against a clean reference. Atlas measured the deformation once and never tested label-free detection. |
| CM-1 | standard-practice | Hendrycks & Gimpel 2017 (MSP baseline); Garg et al. 2022 (ATC = thresholded max-softmax averaged over a set) | Per-sample misclassification detection by max-softmax is the standard baseline, and ATC aggregates the same score into a dataset-level accuracy estimate. Atlas's margin reproduces it on resnet20. |
| CM-11 | partly-known | Jiang et al. 2018, NeurIPS, arXiv 1805.11783 (Trust Score: classifier vs modified nearest-neighbour agreement); Chen et al. 2021, NeurIPS, arXiv 2106.15728 (ensemble disagreement for error detection and accuracy estimation); Jiang et al. 2022, ICLR, arXiv 2106.13799 (disagreement tracks error) | Disagreement between the classifier and a second predictor (nearest-neighbour, ensemble) is a known per-sample error signal, and in aggregate an accuracy estimate. Atlas's nearest-centre-vs-head disagreement was never scored as a detector. The ImageNet dumps (11-20% disagreement) allow a cheap test. |
