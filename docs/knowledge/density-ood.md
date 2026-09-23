# Density-based familiarity and OOD detection: prior art for Atlas

Last reviewed: 2026-09-23; part of the Atlas knowledge base (docs/knowledge/README.md)

Prior-art labels in this file (rediscovered, extends, known-in-research, not found in prior art, and so on) mark
starting points to adopt and refine for Atlas's use; they never down-rank an item (docs/knowledge/README.md §1).

- **Written against:** repo HEAD 343f651. It applies the programme evaluation and its corrections
  (`docs/reviews/EVAL_2026-09-23.md`).
- **Scope:** inventory items DO-1 to DO-5, plus the items this literature also covers: CD-4, CD-9, LH-1 to LH-3, CM-1,
  CM-8, CM-11, ST-3, ST-4 and ST-7.

**How each source was checked.** Every work below was found in a search result or fetched from arXiv. The "check" field
says how far:
- **full text:** the PDF text was extracted and read.
- **abstract:** the abstract was read, from the arXiv API or the landing page.
- **metadata:** only the title, authors and venue were checked, from the arXiv API or a search result. No finding is
  attributed to these works beyond what the title states.
- **venue unconfirmed:** the arXiv record exists, but the venue is from memory. Check it before citing in a paper.

The web-search budget ran out before three questions could be searched. They are listed in section 8, and nothing is
claimed about them.

---

## 1. What this family of methods measures

**What is measured.** How typical a feature vector z = f(x) is under a reference feature distribution P_ref. The
reference is usually the training set's features at the penultimate layer. Low typicality means "unfamiliar". This
describes where the input sits relative to the training data. It says nothing about whether the prediction is correct.

**Estimators.**
- **Non-parametric kNN.** The score is the distance r_k(z) to the k-th nearest reference feature (Sun et al. 2022). On
  data of local dimension m, the kNN density estimate is roughly p(z) ∝ k / (n · r_k(z)^m). So a shift in log r_k equals
  a log-density ratio divided by m. At the Atlas penult, TwoNN ID is about 10 (CD-1), so a median log-radius shift of
  0.04 is a density ratio of about e^0.4 ≈ 1.5. Sun et al. derive this on L2-normalised features on the unit sphere.
- **Parametric (Gaussian).**
  - Class-conditional Gaussians with a tied covariance, scored by the Mahalanobis distance to the nearest class mean
    (Lee et al. 2018).
  - Relative Mahalanobis, which subtracts a class-agnostic background Gaussian (Ren et al. 2021).
  - Normalisation variants such as Mahalanobis++ (Müller & Hein 2025).
  - A GMM/GDA density fitted after training (DDU, Mukhoti et al. 2023).
  - Flows and other density models on features (metadata only: Zisselman & Tamar 2020; Peng et al. 2024, ConjNorm).
- **Subspace and residual scores.**
  - ViM: the energy of z outside the principal subspace of the ID features, turned into a virtual logit (Wang et al.
    2022).
  - Kamoi & Kobayashi (2020) show that Mahalanobis works through directions that are "not useful for classification".
  - Neural-collapse (NC) scores: the relative norm of z inside the simplex equiangular tight frame (ETF) subspace
    (NECO), or proximity to the class weight vectors plus a norm filter (Liu & Qin 2025).
- **Feature norm.** ||z|| as a novelty score (Dhamija et al. 2018). Park et al. (2023) show the feature norm acts as the
  maximum logit of a classifier hidden in the network layer. It therefore separates OOD inputs the same way head
  confidence does.

**Granularity.**
- Per sample: classic OOD detection.
- Per batch: two-sample shift tests on representations or outputs (Rabanser et al. 2019), or batch activation-mean
  discrepancy (NMD, Dong et al. 2022).

**Two kinds of shift.**
- Semantic shift: new classes, which is "OOD" in the narrow sense.
- Covariate shift: corruption and style.

Most feature detectors respond to both. "Full-spectrum" OOD detection (Yang et al. 2022) asks a detector to tolerate
covariate shift and still detect semantic shift. Atlas's CIFAR-10-C `sparse_frac` measures the covariate-shift response.

**What density does not measure: correctness.** A familiar input can be misclassified, as in Atlas's clean-data
confident mistakes (CM-1 to CM-5), and an unfamiliar one can be classified correctly. Postels et al. (2022), Guérin et
al. (2023) and Jaeger et al. (2023) show this empirically (section 4).

**Thresholds.** A density score becomes a decision only once it has a threshold. The threshold's false-alarm rate
depends on the data it was calibrated on (DO-3).

**Atlas's instrument.** `knn_density` (atlas/invariants/density.py:18-53) works as follows:
- It takes the Euclidean distance from each query to its k-th nearest neighbour (k = 10) in a 10,000-row subsample of
  the training reference.
- It uses raw, un-normalised penult features (atlas/invariants/_util.py:43-47, sklearn NearestNeighbors).
- The threshold is the q95 of the reference's own leave-one-out radii (density.py:25-30).
- `sparse_frac` is the share of queries above that threshold (:41).

This is Sun et al.'s detector without L2 normalisation, with the threshold taken from training features rather than
held-out ID data. Sun et al. scale k with the fraction of the bank sampled (stated for ImageNet; k = 50 on the full
CIFAR-10 train set). Under that rule, Atlas's k = 10 on a 20% subsample is the same operating point as their CIFAR-10
k = 50.

---

## 2. Key works

### A. kNN and non-parametric density

| id | citation | URL | check | contribution relevant to Atlas |
|---|---|---|---|---|
| K1 | Y. Sun, Y. Ming, X. Zhu, Y. Li (2022). *Out-of-Distribution Detection with Deep Nearest Neighbors.* ICML 2022 (PMLR). | https://arxiv.org/abs/2204.06507 | full text | See the list below the table. |
| K2 | N. Papernot, P. McDaniel (2018). *Deep k-Nearest Neighbors: Towards Confident, Interpretable and Robust Deep Learning.* arXiv. | https://arxiv.org/abs/1803.04765 | full text; venue unconfirmed | kNN on every layer's representation, with a nonconformity score from neighbour labels. The calibration set "is not used to train the model", so it is a held-out conformal calibration. |
| K3 | H. Jiang, B. Kim, M. Y. Guan, M. Gupta (2018). *To Trust Or Not To Trust A Classifier.* NeurIPS 2018. | https://arxiv.org/abs/1805.11783 | full text | Each class's training set is first filtered to an α-high-density set using kNN radii. The trust score is the ratio of the distance to the nearest other class over the distance to the predicted class. In their experiments it beat the classifier's confidence at flagging errors. It is the direct precedent for Atlas's d2 − d1 margin (CM-1) and for the geometry-vs-head agreement flag (CM-11). |

What K1 (Sun et al. 2022) establishes:
- The score is the k-th NN distance on the L2-normalised penult.
- The threshold is chosen "so that a high fraction of ID data (e.g., 95%) is correctly classified", and "does not
  depend on OOD data".
- **Normalisation is critical.** It improved FPR95 by 61.05% on their ImageNet SupCon model, and ID features have larger
  norms than OOD features.
- A 1% sample of the ImageNet training bank, with k scaled accordingly, performs about as well as the full bank.
- On ViT-B/16 (ImageNet), kNN beats Mahalanobis on FPR95:

  | OOD set | kNN | Mahalanobis |
  |---|---|---|
  | iNaturalist | 7.30 | 17.56 |
  | SUN | 48.40 | 80.51 |
  | Places | 56.46 | 84.12 |
  | Textures | 39.91 | 70.51 |

- On a CIFAR-10 cross-entropy model, kNN has an average FPR95 of 29.15 against 37.94 for Mahalanobis.
- The penultimate layer is better than the projection head.
- **It contains no corruption or severity experiment** (full-text grep).

### B. Gaussian / Mahalanobis family

| id | citation | URL | check | contribution |
|---|---|---|---|---|
| M1 | K. Lee, K. Lee, H. Lee, J. Shin (2018). *A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks.* NeurIPS 2018. | https://arxiv.org/abs/1807.03888 | abstract | Class-conditional Gaussians with tied covariance; Mahalanobis distance to the closest class mean; a feature ensemble across layers. OpenOOD v1.5 lists this line among works that tuned hyperparameters on test OOD data. |
| M2 | J. Ren, S. Fort, J. Liu, A. G. Roy, S. Padhy, B. Lakshminarayanan (2021). *A Simple Fix to Mahalanobis Distance for Improving Near-OOD Detection.* arXiv. | https://arxiv.org/abs/2106.09022 | abstract (search); venue unconfirmed | Relative Mahalanobis distance (RMD). It analyses why Mahalanobis fails on near-OOD, and RMD is more robust to hyperparameter choice. |
| M3 | S. Fort, J. Ren, B. Lakshminarayanan (2021). *Exploring the Limits of Out-of-Distribution Detection.* NeurIPS 2021. | https://arxiv.org/abs/2106.03004 | abstract | With ViTs pre-trained on ImageNet-21k, near-OOD AUROC on CIFAR-100 vs CIFAR-10 rises from 85% (the previous state of the art) to over 96%; Mahalanobis and RMD are among the scores used. |
| M4 | M. Müller, M. Hein (2025). *Mahalanobis++: Improving OOD Detection via Feature Normalization.* ICML 2025 (PMLR 267). | https://arxiv.org/abs/2505.18032 | abstract (search + PMLR page) | Mahalanobis performance varies strongly across models because of variation in feature norms, which violates the Gaussian assumption. ℓ2 normalisation fixes this consistently across 44 models. |
| M5 | R. Kamoi, K. Kobayashi (2020). *Why is the Mahalanobis Distance Effective for Anomaly Detection?* arXiv. | https://arxiv.org/abs/2003.00402 | abstract; venue unconfirmed | Mahalanobis succeeds through information not useful for classification. Combining it with the confidence-based ODIN improves detection. |
| M6 | J. Mukhoti, A. Kirsch, J. van Amersfoort, P. H. S. Torr, Y. Gal (2023). *Deep Deterministic Uncertainty: A New Simple Baseline.* CVPR 2023. | https://arxiv.org/abs/2102.11582 | abstract | A GDA feature density fitted after training carries epistemic uncertainty, and softmax entropy carries aleatoric. It needs a well-regularised feature space (residual connections plus spectral normalisation). |
| M7 | E. Zisselman, A. Tamar (2020), *Deep Residual Flow for OOD Detection* (arXiv 2001.05419); B. Peng et al. (2024), *ConjNorm: Tractable Density Estimation for OOD Detection*, ICLR 2024 (arXiv 2402.17888). | https://arxiv.org/abs/2001.05419 ; https://arxiv.org/abs/2402.17888 | metadata only | Other feature-space density estimators; findings not read. |

### C. Residual, norm and neural-collapse-based scores

| id | citation | URL | check | contribution |
|---|---|---|---|---|
| R1 | H. Wang, Z. Li, L. Feng, W. Zhang (2022). *ViM: Out-Of-Distribution with Virtual-logit Matching.* CVPR 2022. | https://arxiv.org/abs/2203.10807 | abstract (search + CVF page) | The residual of the feature against the principal subspace becomes a virtual OOD logit, matched to the real logits. It is the canonical "feature plus head" joint score. |
| R2 | M. Ben Ammar, N. Belkhir, S. Popescu, A. Manzanera, G. Franchi (2024). *NECO: NEural Collapse Based Out-of-distribution detection.* ICLR 2024. | https://arxiv.org/abs/2310.06823 | abstract | Hypothesis: neural collapse also shapes OOD features. The score uses NC geometry and principal-component subspaces, reported as the relative norm of the feature within the ETF subspace. |
| R3 | J. Haas, W. Yolland, B. T. Rabus (2023). *Linking Neural Collapse and L2 Normalization with Improved Out-of-Distribution Detection in Deep Neural Networks.* TMLR (accepted 2022 per arXiv). | https://arxiv.org/abs/2209.08378 | abstract | Feature L2 normalisation induces early NC and better OOD detection on the DDU benchmark, including better worst-case OOD performance across random seeds. |
| R4 | L. Liu, Y. Qin (2025). *Detecting Out-of-Distribution Through the Lens of Neural Collapse.* CVPR 2025. | https://arxiv.org/abs/2311.01479 | abstract | See the list below the table. |
| R5 | Y. Wu, R. Yu, X. Cheng, Z. He, X. Huang (2025). *Pursuing Feature Separation based on Neural Collapse for Out-of-Distribution Detection.* ICLR 2025. | https://arxiv.org/abs/2405.17816 | abstract | A training loss, using auxiliary OOD data, that binds OOD features to the subspace orthogonal to the NC principal subspace. |
| R6 | A. R. Dhamija, M. Günther, T. E. Boult (2018). *Reducing Network Agnostophobia.* NeurIPS 2018. | https://arxiv.org/abs/1811.04110 | abstract | Softmax thresholding and a background class are insufficient for unseen classes. The Objectosphere loss separates known from unknown inputs by feature magnitude. |
| R7 | J. Park, J. C. L. Chai, J. Yoon, A. B. J. Teoh (2023). *Understanding the Feature Norm for Out-of-Distribution Detection.* ICCV 2023. | https://arxiv.org/abs/2310.05316 | abstract | The feature norm is the maximum logit of a classifier hidden in the layer, so it is a confidence-like signal. It is class-agnostic. Proposes the negative-aware norm (NAN). |
| R8 | K. Kang, A. Setlur, C. Tomlin, S. Levine (2024). *Deep Neural Networks Tend To Extrapolate Predictably.* ICLR 2024. | https://arxiv.org/abs/2310.00873 | full text | See the list below the table. |
| R9 | J. van Amersfoort, L. Smith, A. Jesson, O. Key, Y. Gal (2021). *On Feature Collapse and Deep Kernel Learning for Single Forward Pass Uncertainty.* arXiv. | https://arxiv.org/abs/2102.11409 | abstract; venue unconfirmed | "Feature collapse": without constraints, "far-away" inputs are mapped to the same features as training points. A bi-Lipschitz constraint is the proposed fix. |
| R10 | J. Zhang et al. (2024). *EPA: Neural Collapse Inspired Robust Out-of-Distribution Detector.* ICASSP 2024. | https://arxiv.org/abs/2401.01710 | metadata only | Another NC-based detector. |

What R4 (Liu & Qin 2025) reports:
- Centred ID clusters align with the class weight vectors.
- ID features expand into a simplex ETF, which explains why ID features lie farther from the origin than OOD features.
- The detector combines proximity to the weight vectors with a norm filter.

What R8 (Kang et al. 2024) reports:
- As inputs become more OOD, predictions revert to the "optimal constant solution".
- The datasets include CIFAR10-C, and the mechanism study uses ResNet20 on CIFAR-10.
- Late-layer feature norms decrease as the shift increases.
- There are exceptions, for example impulse noise on UTKFace.

### D. Output-head baselines

| id | citation | URL | check | contribution |
|---|---|---|---|---|
| H1 | D. Hendrycks, K. Gimpel (2017). *A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks.* ICLR 2017. | https://arxiv.org/abs/1610.02136 | metadata | The MSP (maximum softmax probability) baseline for both errors and OOD. |
| H2 | W. Liu, X. Wang, J. D. Owens, Y. Li (2020). *Energy-based Out-of-distribution Detection.* NeurIPS 2020. | https://arxiv.org/abs/2010.03759 | metadata | The energy (logsumexp) score. |
| H3 | D. Hendrycks, S. Basart, M. Mazeika, A. Zou, J. Kwon, M. Mostajabi, et al. (2022). *Scaling Out-of-Distribution Detection for Real-World Settings.* ICML 2022. | https://arxiv.org/abs/1911.11132 | abstract | The maximum logit beats prior methods in large-scale multi-class, multi-label and segmentation settings. |
| H4 | G. Xia, C.-S. Bouganis (2022). *Augmenting Softmax Information for Selective Classification with Out-of-Distribution Data.* ACCV 2022. | https://arxiv.org/abs/2207.07506 | abstract | For selective classification with OOD data (SCOD), OOD detectors behave differently than on OOD detection alone. SIRC adds secondary information to softmax confidence. OOD detection improves without losing the separation between correct and incorrect ID predictions. |

### E. Ensembles and uncertainty under shift

| id | citation | URL | check | contribution |
|---|---|---|---|---|
| U1 | B. Lakshminarayanan, A. Pritzel, C. Blundell (2017). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NIPS 2017. | https://arxiv.org/abs/1612.01474 | abstract | Ensembles of independently trained nets give well-calibrated uncertainty that is higher on OOD inputs. |
| U2 | Y. Ovadia, E. Fertig, J. Ren, Z. Nado, D. Sculley, S. Nowozin, J. V. Dillon, B. Lakshminarayanan, J. Snoek (2019). *Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift.* NeurIPS 2019. | https://arxiv.org/abs/1906.02530 | abstract | A large-scale study of accuracy and calibration under shift. Post-hoc calibration falls short; methods that marginalise over models are strongest. |
| U3 | J. Postels, M. Segù, T. Sun, L. D. Sieber, L. Van Gool, F. Yu, F. Tombari (2022). *On the Practicality of Deterministic Epistemic Uncertainty.* ICML 2022 (PMLR 162). | https://arxiv.org/abs/2107.00649 | full text | See the list below the table. |
| U4 | J. van Amersfoort, L. Smith, Y. W. Teh, Y. Gal (2020), *Uncertainty Estimation Using a Single Deep Deterministic Neural Network* (DUQ), ICML 2020; J. Z. Liu et al. (2020), *Simple and Principled Uncertainty Estimation with Deterministic Deep Learning via Distance Awareness* (SNGP), NeurIPS 2020. | https://arxiv.org/abs/2003.02037 ; https://arxiv.org/abs/2006.10108 | metadata | Distance-aware single-model uncertainty. |

What U3 (Postels et al. 2022) finds:
- The data are CIFAR-10-C and CIFAR-100-C across 5 severities, on a ResNet-50 backbone.
- "Calibration" here means the AUROC and rAULC of separating correct from incorrect predictions.
- Ensembles and MC dropout are best.
- SNGP is the only deterministic uncertainty method (DUM) that consistently beats softmax entropy.
- Methods that rely on the distribution of hidden representations (DDU, MIR, DCU) are worse calibrated, although DUMs
  remain good at OOD detection.
- The authors advise against methods that rely purely on feature-space distances or likelihoods when calibrated
  uncertainty is needed.

### F. Thresholds, calibration and the train-vs-test feature gap

| id | citation | URL | check | contribution |
|---|---|---|---|---|
| C1 | S. Bates, E. Candès, L. Lei, Y. Romano, M. Sesia (2023). *Testing for Outliers with Conformal p-values.* Annals of Statistics 51(1):149-178. | https://arxiv.org/abs/2104.08279 | full text | See the list below the table. |
| C2 | L. Hui, M. Belkin, P. Nakkiran (2022). *Limitations of Neural Collapse for Understanding Generalization in Deep Learning.* arXiv. | https://arxiv.org/abs/2202.08384 | abstract; venue unconfirmed | Neural collapse "often occurs on the train set" but "does not occur on the test set". Training longer can worsen last-layer features for transfer. |
| C3 | J. Zhang, J. Yang, et al. (2023). *OpenOOD v1.5: Enhanced Benchmark for Out-of-Distribution Detection.* arXiv. | https://arxiv.org/abs/2306.09301 | full text | See the list below the table. |
| C4 | J. Yang, P. Wang, D. Zou, et al. (2022). *OpenOOD: Benchmarking Generalized Out-of-Distribution Detection.* NeurIPS 2022 Datasets and Benchmarks. | https://arxiv.org/abs/2210.07242 | metadata | The v1 benchmark. |
| C5 | J. Yang, K. Zhou, Z. Liu (2022). *Full-Spectrum Out-of-Distribution Detection.* arXiv. | https://arxiv.org/abs/2204.05306 | abstract; venue unconfirmed | The literature has no consensus on how to treat covariate shift. Their benchmarks separate training ID, covariate-shifted ID, near-OOD and far-OOD. SEM cancels the non-semantic part of the score. |

What C1 (Bates et al. 2023) specifies:
- A one-class score is trained on D_train and evaluated on a held-out D_cal.
- The conformal p-values are valid when the score is independent of the calibration and test data.
- Standard p-values are only marginally valid. The paper introduces calibration-conditional p-values, proves BH FDR
  control, and gives a uniform confidence bound on the FPR as a function of the threshold.

What C3 (OpenOOD v1.5) reports:
- **Held-out validation.** It uses held-out ID validation data (1,000 CIFAR test images; 5,000 of the 50,000 ImageNet
  val images) and held-out OOD validation data for tuning.
- **No single winner.** No method wins on every benchmark. KNN is strong on small datasets but shows no clear advantage
  on large ones.
- **Full spectrum.** When covariate-shifted ID must be accepted, the near-OOD AUROC of most methods drops by more than
  10% on ImageNet-1K.
- **Cost.** KNN and MDS store ID features, which costs memory and raises a privacy risk.
- **CIFAR-10 ResNet-18 near/far AUROC:**

  | method | near-OOD | far-OOD |
  |---|---|---|
  | MSP | 88.03 | 90.73 |
  | EBO | 87.58 | 91.21 |
  | MDS | 84.20 | 89.72 |
  | RMDS | 89.80 | 92.20 |
  | ViM | 88.68 | 93.48 |
  | KNN | 90.64 | 92.96 |

- **Benchmark groups.** CIFAR-10 near-OOD is CIFAR-100 and TinyImageNet; far-OOD is MNIST, SVHN, Textures and
  Places365.

### G. Shift monitoring and label-free accuracy

| id | citation | URL | check | contribution |
|---|---|---|---|---|
| S1 | S. Rabanser, S. Günnemann, Z. C. Lipton (2019). *Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift.* NeurIPS 2019. | https://arxiv.org/abs/1810.11953 | abstract | Across the shifts studied, two-sample tests that use a pre-trained classifier for dimensionality reduction perform best; this is BBSD (black-box shift detection). Domain classifiers help characterise shifts and judge whether they are harmful. |
| S2 | Z. C. Lipton, Y.-X. Wang, A. Smola (2018). *Detecting and Correcting for Label Shift with Black Box Predictors.* ICML 2018. | https://arxiv.org/abs/1802.03916 | metadata | Label-shift detection from classifier outputs. |
| S3 | X. Dong, J. Guo, A. Li, W.-T. Ting, C. Liu, H. T. Kung (2022). *Neural Mean Discrepancy for Efficient Out-of-Distribution Detection.* CVPR 2022. | https://arxiv.org/abs/2104.11408 | abstract | Activation means of OOD mini-batches deviate more from the training means. The training means are available for free from the batch-norm layers. |
| S4 | D. Guillory, V. Shankar, S. Ebrahimi, T. Darrell, L. Schmidt (2021). *Predicting with Confidence on Unseen Distributions.* ICCV 2021. | https://arxiv.org/abs/2107.03315 | abstract | Distributional distances such as Fréchet distance and MMD "fail to induce reliable estimates of performance under distribution shift". The difference of confidences (DoC) works. |
| S5 | S. Garg, S. Balakrishnan, Z. C. Lipton, B. Neyshabur, H. Sedghi (2022). *Leveraging Unlabeled Data to Predict Out-of-Distribution Performance.* ICLR 2022. | https://arxiv.org/abs/2201.04234 | abstract | ATC (average thresholded confidence) estimates target accuracy 2-4× more accurately than prior methods. In general, identifying the accuracy is as hard as identifying the optimal predictor. |
| S6 | W. Deng, L. Zheng (2021). *Are Labels Always Necessary for Classifier Accuracy Evaluation?* CVPR 2021. | https://arxiv.org/abs/2007.02915 | abstract | AutoEval regresses accuracy on dataset-level feature statistics, using a synthetic meta-dataset. |
| S7 | D. Hendrycks, T. Dietterich (2019). *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations.* ICLR 2019. | https://arxiv.org/abs/1903.12261 | metadata | CIFAR-10-C and ImageNet-C. |

### H. Framing: failure detection versus OOD detection

| id | citation | URL | check | contribution |
|---|---|---|---|---|
| F1 | J. Guérin, K. Delmas, R. S. Ferreira, J. Guiochet (2023). *Out-Of-Distribution Detection Is Not All You Need.* AAAI 2023. | https://arxiv.org/abs/2211.16158 | abstract | Runtime monitors should be judged on how well they discard incorrect predictions ("out-of-model-scope" detection). Good OOD results can give a false impression of safety. |
| F2 | P. F. Jaeger, C. T. Lüth, L. Klein, T. J. Bungert (2023). *A Call to Reflect on Evaluation Practices for Failure Detection in Image Classification.* ICLR 2023. | https://arxiv.org/abs/2211.15259 | abstract | A unified failure-detection benchmark (FD-Shifts). A plain softmax-response baseline is the best method overall. |
| F3 | S. Vaze, K. Han, A. Vedaldi, A. Zisserman (2022), *Open-Set Recognition: a Good Closed-Set Classifier is All You Need?*, ICLR 2022; F. Tajwar et al. (2021), *No True State-of-the-Art? OOD Detection Methods are Inconsistent across Datasets*, ICML UDL workshop; J. Bitterwolf, M. Müller, M. Hein (2023), *In or Out? Fixing ImageNet OOD Detection Evaluation*, ICML 2023; I. Galil, M. Dabbah, R. El-Yaniv (2023), *A framework for benchmarking class-out-of-distribution detection and its application to ImageNet*, ICLR 2023. | https://arxiv.org/abs/2110.06207 ; https://arxiv.org/abs/2109.05554 ; https://arxiv.org/abs/2306.00826 ; https://arxiv.org/abs/2302.11893 | metadata | Evaluation hygiene. The accuracy-OOD correlation is also cited in OpenOOD v1.5. |

### I. Input-space density, unsupervised features, compute and TTA gating

| id | citation | URL | check | contribution |
|---|---|---|---|---|
| I1 | P. Kirichenko, P. Izmailov, A. G. Wilson (2020). *Why Normalizing Flows Fail to Detect Out-of-Distribution Data.* NeurIPS 2020. | https://arxiv.org/abs/2006.08545 | abstract | Input-space likelihood fails: "a flow trained on pictures of clothing assigns higher likelihood to handwritten digits". Flows learn local pixel correlations. Related: E. Nalisnick et al. (2019), *Do Deep Generative Models Know What They Don't Know?*, ICLR 2019, https://arxiv.org/abs/1810.09136 (metadata). |
| I2 | V. Sehwag, M. Chiang, P. Mittal (2021). *SSD: A Unified Framework for Self-Supervised Outlier Detection.* ICLR 2021. | https://arxiv.org/abs/2103.12051 | abstract | Mahalanobis distance on self-supervised features, trained on unlabeled ID data only, performs on par with supervised detectors. It is the route to encoders without a class head. Related (metadata only): CIDER (Ming et al., ICLR 2023, arXiv 2203.04450) and CSI (Tack et al., NeurIPS 2020, arXiv 2007.08176). |
| I3 | S. Niu, J. Wu, Y. Zhang, Y. Chen, S. Zheng, P. Zhao, et al. (2022). *Efficient Test-Time Model Adaptation without Forgetting* (EATA). ICML 2022. | https://arxiv.org/abs/2204.02610 | abstract | TTA updates only on "reliable and non-redundant samples". High-entropy samples give noisy gradients. The gate is the head's entropy. |
| I4 | S. Teerapittayanon, B. McDanel, H. T. Kung, *BranchyNet* (arXiv 1709.01686); Y. Kaya, S. Hong, T. Dumitras, *Shallow-Deep Networks*, ICML 2019 (arXiv 1810.07052); G. Huang et al., *Multi-Scale Dense Networks* (arXiv 1703.09844). | https://arxiv.org/abs/1709.01686 ; https://arxiv.org/abs/1810.07052 ; https://arxiv.org/abs/1703.09844 | metadata; the exit mechanism is from memory | Adaptive-compute networks. Their exits are gated by the confidence of internal classifiers (not re-read here). |

Also cited in the repo: Papyan, Han & Donoho (2020) on neural collapse (atlas/invariants/landmarks.py:6-7); its entry
and verification are in `collapse-dimension-similarity.md` §2.1.

---

## 3. How it is used

Each use below lists its status (standard or research-only), typical metrics and known failure modes.

**1. Post-hoc OOD / novelty flag per input.**
- **Standard vs research:**
  - In research benchmarks this is standard practice (OpenOOD).
  - The deployed default is a head score (MSP, max logit, energy), because it costs nothing.
  - Feature scores (kNN, Mahalanobis, RMD, ViM, NECO) need a stored ID bank or ID statistics.
- **Metrics:** AUROC, FPR@95%TPR and AUPR. The threshold is set at a target TPR on ID data (K1); OpenOOD v1.5 takes
  that ID data from a held-out validation split (C3).
- **Failure modes:**
  - no single winner, and near-OOD is hard (C3);
  - a feature-norm confound without L2 normalisation (K1, M4);
  - the class-Gaussian assumption (K1, M4);
  - hyperparameters tuned on test OOD data (C3 on M1);
  - OOD test sets contaminated with ID objects (F3, Bitterwolf);
  - memory, latency and privacy of the bank (C3).

**2. Covariate-shift tolerance or sensitivity (full-spectrum OOD).**
- **Standard vs research:** research only. Whether a corrupted input should be flagged depends on the task (C5).
- **Metrics:** AUROC in which covariate-shifted ID counts as ID.
- **Failure modes:** most detectors fire on covariate shift; near-OOD AUROC drops more than 10% on ImageNet-1K (C3).

**3. Selective prediction / misclassification ("failure") detection.**
- **Standard vs research:** softmax response or MSP is standard and hard to beat (F2, H1). kNN trust scores (K3, K2) and
  combined scores (H4 SIRC; M5 Mahalanobis + ODIN) are research.
- **Metrics:** AURC / E-AURC; AUROC of correct vs wrong; risk at a fixed coverage.
- **Failure modes:** feature-density scores rank correctness worse than softmax entropy under corruption (U3).
  OOD-optimised scores can hurt failure detection (F1, H4).

**4. Epistemic uncertainty from one deterministic model (DDU, DUQ, SNGP).**
- **Standard vs research:** research.
- **Metrics:** OOD AUROC; correctness AUROC under shift (U3).
- **Failure modes:** feature collapse without spectral normalisation or a bi-Lipschitz constraint (R9, M6); poor
  calibration under continuous shift (U3).

**5. Conformity / trust from kNN (DkNN, Trust Score).**
- **Standard vs research:** research. DkNN uses a held-out calibration set (K2).
- **Metrics:** error-flag precision; credibility calibration.
- **Failure modes:** bank cost; the choice of layer.

**6. Batch shift monitoring (model monitoring).**
- **Standard vs research:** practice uses two-sample tests on model outputs (BBSD) or on reduced representations (S1).
  NMD on batch-norm means is research (S3).
- **Metrics:** detection power against batch size at a fixed false-alarm rate; detection delay in streams; FDR across
  repeated tests (C1).
- **Failure modes:** detecting a shift says nothing about whether it is harmful (S1). Repeated tests over a stream need
  multiplicity control (C1).

**7. Label-free accuracy estimation.**
- **Standard vs research:** confidence-based estimators (average confidence, DoC, ATC) are the standard baselines (S4,
  S5). Feature-statistic regression (AutoEval, S6) is research.
- **Metrics:** MAE of the predicted accuracy; rank correlation across shifts.
- **Failure modes:** distributional distances do not transfer reliably across shift types (S4). The problem is
  unidentifiable without assumptions (S5).

**8. TTA gating.**
- **Standard vs research:** the standard gate is the head's entropy (EATA sample selection, I3). Feature-density gating
  was **not found** in the searches that could be run (section 8).
- **Metrics:** online accuracy; forgetting on clean data.
- **Failure modes:** entropy gates trust confident mistakes.

**9. Adaptive ("novelty-proportional") compute.**
- **Standard vs research:** early exits gated by internal-classifier confidence (I4). Density-gated compute was **not
  found**.
- **Metrics:** accuracy against FLOPs.
- **Failure modes:** confident mistakes exit early.

**10. Model / checkpoint selection.**
- **Standard vs research:** research. OOD performance tracks closed-set accuracy (F3, as cited in C3), and augmentation
  helps OOD detection (C3).
- **Metrics:** OOD AUROC against ID accuracy.
- **Failure modes:** benchmark inconsistency (F3).

**11. Encoders without a class head (self-supervised, robot).**
- **Standard vs research:** research. Mahalanobis or kNN on self-supervised features (I2; KNN+ on SupCon features in K1).
- **Metrics:** OOD AUROC.
- **Failure modes:** no correctness signal is available at all.

---

## 4. Known results relevant to Atlas

1. **Normalise before measuring distance.**
   - Un-normalised kNN distances are confounded by feature norm, because ID features have larger norms than OOD
     features. Normalisation improved kNN FPR95 by 61% on ImageNet SupCon (K1).
   - Mahalanobis fails across models for the same reason, and ℓ2 normalisation fixes it (M4).
   - Atlas's `knn_density` is un-normalised (_util.py:43-47).
2. **Feature norms shrink as shift grows, and the norm behaves like confidence.**
   - Kang et al. show late-layer feature norms decrease as distribution shift increases, in a ResNet20 on CIFAR-10
     among other settings; predictions revert toward the optimal constant solution (R8).
   - Park et al. show the feature norm acts as a hidden max logit (R7).
   - Atlas's own `norm_ratio` falls with severity for most corruptions. In resnet20 s1, gaussian noise goes s1 / s3 /
     s5 = 0.909 / 0.811 / 0.762 (recomputed from `results/atlas_v1_resnet20_s1/atlas.json`
     `.per_layer.penult.corruption_displacement.splits.*.norm_ratio`). That is consistent with R8.
3. **Feature collapse.** Standard networks can map far-away inputs onto training features (R9). Density scores are
   reliable only with regularised features (M6). Nothing guarantees that density rises monotonically with severity.
4. **Thresholds must be calibrated on data the feature map never trained on.**
   - Neural collapse holds on the training set but not on the test set (C2), so the training features' leave-one-out
     kNN radii are tighter than those of test features.
   - The standard remedies:
     - set the threshold on ID data at a target TPR (K1);
     - hold out ID validation data (C3);
     - use a calibration set not used for training (K2);
     - use split-conformal p-values with a finite-sample bound on the FPR (C1).
5. **kNN against Mahalanobis depends on the setting.**

   | setting | outcome | source |
   |---|---|---|
   | ViT-B/16 on ImageNet | kNN beats Mahalanobis on FPR95 | K1 |
   | CIFAR-10 cross-entropy model | kNN FPR95 29.15 vs Mahalanobis 37.94 | K1 |
   | ImageNet-21k-pretrained ViT, CIFAR-100 vs CIFAR-10 near-OOD | Mahalanobis family reaches AUROC > 96% | M3 |
   | Mahalanobis on pre-logit features, after ℓ2 normalisation | outperforms other recent methods across 44 models | M4 |
   | across benchmarks | no single winner; KNN strong on small data but no clear advantage on large | C3 |

6. **The expected size of a feature-over-head gain on CIFAR-10 is small.**
   - OpenOOD v1.5 (ResNet-18) near/far AUROC: KNN 90.64 / 92.96 against MSP 88.03 / 90.73, a gain of about 2-3 points.
   - Mahalanobis (MDS) is *below* MSP: 84.20 / 89.72.
   - RMD recovers to 89.80 / 92.20 (C3).
7. **For per-sample correctness under corruption, feature density loses to the head.**
   - On CIFAR-10-C across severities, methods based on hidden-representation distributions (DDU, MIR, DCU) rank correct
     against incorrect predictions worse than ensembles and MC dropout, and only SNGP consistently beats softmax
     entropy (U3).
   - Softmax response is the best failure detector overall (F2).
8. **Ensembles are the strongest uncertainty baseline under shift** (U1, U2). Atlas already has 4-5 same-recipe
   resnet20 nets and 2-3 resnet56 nets, so an ensemble baseline costs almost nothing.
9. **Batch-level shift detection.**
   - Two-sample tests on classifier outputs (BBSD) were best in Failing Loudly (S1).
   - Batch activation-mean discrepancy is a cheap feature analogue (S3).
   - For label-free accuracy, confidence-based DoC and ATC are the standard baselines. Distributional distances were
     unreliable across shift types (S4, S5).
10. **Neural collapse and OOD.**
    - L2 normalisation plus early neural collapse improves OOD detection (R3).
    - NC-derived scores exploit the ETF geometry: NECO (R2), NC-based detection (R4), and a separation loss (R5).
    - None of these papers quantifies how collapse sets the *false-alarm rate* of a train-referenced threshold
      (section 8).
11. **Input-space density is the wrong place** (I1). Atlas's penult placement is consistent with the field.
12. **A self-supervised feature density can work without labels** (I2). It is the only member of this family that needs
    neither a class head nor labels, which matters for the robot goal.

**Corrections to the evaluation's prior-art notes.** The evaluation's science-and-novelty audit lens (summarised in
`docs/reviews/EVAL_2026-09-23.md` §2.5) made two statements this research does not support:
- **"kNN sparsity rises with shift severity (Sun et al. 2022)"** is not supported by that paper: it has no corruption or
  severity experiment (full-text grep for "corrupt" and "severity"). The relevant prior art is:
  - the covariate-shift sensitivity of OOD detectors (C5; C3 §6.2);
  - the CIFAR-10-C severity sweeps of feature-density methods in Postels et al. (U3).
- **"Calibrating kNN thresholds on held-out ID data is standard practice (Sun et al. 2022)"** is overstated. Sun sets the
  threshold on ID data at 95% TPR but does not say the data are held out. The held-out requirement is explicit in
  OpenOOD v1.5 (C3), DkNN (K2) and split-conformal testing (C1).

---

## 5. Relation to Atlas items

Relation labels: **rediscovered**, **extends**, **contradicts** or **untested here**.

### DO-1: density rises with severity for the 8 non-noise corruptions (row 6a)

- **Relation:** rediscovered, and extends.
- **Literature:**
  - Feature detectors respond to covariate shift (C5; C3 §6.2).
  - The detector is K1's without normalisation.
  - The head also degrades with severity (R8, U2).
  - U3 predicts that density will not beat softmax entropy at ranking correctness across severities.
- **Atlas evidence:**
  - 8/8 corruptions, s5 > s1, in 7 trained nets; nulls 3/8-4/8.
  - The smallest margin is brightness, 0.0395 in s4, against a twin reference-draw shift of 0.0165.
  - Paths: ATLAS_STATUS.md:15; results/atlas_v1_resnet20_s3/SESSION.md:209-218, :357-360.
  - The extension over the literature is the per-corruption monotonicity replicated across seeds and depth.
- **Beyond the head:** UNTESTED. No per-split head statistic is stored.
- **Next check:** on the pod, compute per-split mean maxprob, entropy and predicted-histogram shift from `preds`
  (atlas/extract_acts.py:171-188). Add L2-normalised kNN.

### DO-2: noise non-monotone (row 6b); v0 "monotone for every corruption"

- **Relation:** the v0 prediction **contradicts** known behaviour; the observed non-monotonicity is **rediscovered**.
- **Literature:** feature collapse (R9); norm shrinkage with shift (R8); kNN's sensitivity to feature norm (K1).
- **Atlas evidence**, recomputed from committed `atlas.json` (penult `sparse_frac` for gaussian s1 / s3 / s5):

  | net | gaussian sparse_frac s1 / s3 / s5 | shape |
  |---|---|---|
  | resnet20 s1 | 0.308 / 0.381 / 0.383 | plateau |
  | resnet20 s3 | 0.324 / 0.376 / 0.318 | drop at s5 |
  | resnet20 s4 | 0.338 / 0.556 / 0.633 | rising |
  | resnet56 s1 | 0.427 / 0.621 / 0.650 | rising |

  - Meanwhile the resnet20 `norm_ratio` keeps falling (s1: 0.909 → 0.811 → 0.762).
  - v0 is at results/atlas_v0_resnet20_cifar10/SESSION.md:97-114; row 6b is ATLAS_STATUS.md:16.
- **Beyond the head:** UNTESTED.
- **Next check:**
  - L2-normalised kNN, with ||z|| as a separate channel.
  - The nearest-neighbour class histogram at noise s5, which tests the v0 hypothesis that s5 noise collapses into a
    dense confident-wrong region.

### DO-3: train-reference threshold over-alarms, with a log-nc1 law (AH-1)

- **Relation:** the principle is **rediscovered**; the quantitative law **extends** it.
- **Literature:**
  - Train-set collapse exceeds test-set collapse (C2).
  - Calibrate on held-out ID data (K1, K2, C1, C3).
  - Neural collapse is linked to OOD geometry (R2-R4).
  - No published law found linking nc1 to the false-alarm rate (section 8).
- **Atlas evidence:**
  - Clean-test `sparse_frac` at the train-q95 threshold (recomputed from the committed atlas.json files):

    | nets | clean-test sparse_frac | × nominal (0.05) |
    |---|---|---|
    | resnet20 s1-s4 | 0.106 / 0.115 / 0.105 / 0.091 | 1.8-2.3× |
    | resnet56 s1 / s2 | 0.163 / 0.184 | 3.3-3.7× |
    | random-init nulls | 0.049 / 0.052 | at nominal |

    The null row shows that the excess comes from fitting the training data. (The plan's controller paragraph quotes
    2.2-3.3× from discovery instances; docs/plans/ANOMALY_H1.md:274-277.)
  - AH-1(a) holds within ±0.035 at 7/7 fresh instances (results/anomaly_h1/SESSION.md:135-155). Rule:
    docs/plans/ANOMALY_H1.md:274-277.
- **Beyond the head:** N/A.
- **Next check:**
  - A split-conformal threshold on held-out clean rows, reporting the realised FPR with a confidence interval (C1).
  - The same test for an MSP threshold.

### DO-4: far-OOD novelty flag (CIFAR-100, SVHN)

- **Relation:** rediscovered.
- **Literature:** K1, M1-M4, R1, C3.
- **Atlas evidence:**
  - Flag rates at the train-q95 threshold (recomputed):

    | nets | CIFAR-100 | SVHN |
    |---|---|---|
    | resnet20 s1-s4 | 0.45-0.49 | 0.60-0.72 |
    | resnet56 s1 / s2 | 0.78 / 0.79 | 0.85 / 0.92 |
    | random-init nulls | 0.10 | 0.03-0.12 |

  - The deeper nets flag both more OOD *and* more clean test data, so the operating point is confounded.
  - v0: results/atlas_v0_resnet20_cifar10/SESSION.md:133. AX-3 INFO: results/anomaly_h1/SESSION.md:332.
- **Beyond the head:** UNTESTED. The expected gain of kNN over MSP is about 2-3 AUROC points at CIFAR-10 scale (C3).
- **Next check:** AUROC and FPR95 against MSP, max logit, energy, Mahalanobis/RMD, ViM and normalised kNN. The MSP values
  are available from the stored `preds` (maxprob).

### DO-5: familiarity for cheap inference (novelty-proportional compute)

- **Relation:** untested here.
- **Literature:** adaptive compute gates on head confidence (I4). A 1% bank suffices for kNN (K1). Bank memory and
  latency are real costs (C3).
- **Atlas evidence:** a proposal only (density.py:6-9).
- **Beyond the head:** UNTESTED.
- **Next check:** accuracy against FLOPs for exits gated on density vs on confidence.

### CD-4: nc1 as the checkpoint covariate that sets DO-3

- **Relation:** extends.
- **Literature:** C2 explains the train-vs-test gap; R2-R4 show collapse shapes OOD separability.
- **Atlas evidence:** results/atlas_v1_resnet56_s1/SESSION.md:259-305; results/anomaly_h1/SESSION.md:135-155.
- **Beyond the head:** N/A.
- **Next check:** whether the law holds for normalised kNN and for Mahalanobis.

### CD-9: one collapse scalar explains several map quantities

- **Relation:** extends; a candidate hypothesis.
- **Literature:** consistent with NC-OOD work (R2-R4). No work tests a single-scalar account across density, CKA and
  adjacency (section 8).
- **Atlas evidence:** the evaluation's synthesis (`docs/reviews/EVAL_2026-09-23.md` §2.2); AH-3 and AH-7 refuted.
- **Beyond the head:** N/A.
- **Next check:** the within-class-residual CKA, and the DO-3 law under normalisation.

### LH-1: split-level harm grade H from the kNN log-radius shift

- **Relation:** rediscovered as a class of estimator; untested against the head.
- **Literature:**
  - AutoEval (S6) regresses accuracy on feature statistics.
  - Distributional distances did not reliably estimate performance across shift types, while DoC did (S4). ATC is the
    standard (S5). BBSD was best for detection (S1).
- **Atlas evidence:**
  - Spearman(H, cost) 0.98-0.99.
  - Any penult shift statistic does as well; stem H gives only 0.19 / 0.26.
  - Paths: results/anomaly_h1/SESSION.md:157-173, :375-378; docs/plans/ANOMALY_H1.md:158-160, :279-310.
- **Beyond the head:** UNTESTED. H is built on the classifier's own input, so its alignment with harm is structural.
- **Next check:** H against batch-mean MSP, entropy, DoC, ATC and BBSD, across shift *types* (S4's warning), not only
  across CIFAR-10-C severities.

### LH-2: batch harm grade h(B) with a HOLD band (AX-3)

- **Relation:** extends.
- **Literature:** NMD (S3), two-sample batch tests (S1), multiplicity control for repeated tests (C1).
- **Atlas evidence:** results/anomaly_h1/SESSION.md:318-337, :387-392; ANOMALY_H1.md:724-753. h(B) is already centred on
  a held-out clean median, rows 2000-3499 (scripts/anomaly_probe.py:442-451).
- **Beyond the head:** UNTESTED.
- **Next check:** the same comparison as LH-1 at batch sizes 16-256, with streaming false-alarm and detection-delay
  curves.

### LH-3: "no norm change means safe to hold" refuted

- **Relation:** consistent with the literature. Norm tracks shift and confidence (R7, R8) but is not calibrated to harm.
- **Atlas evidence:** results/anomaly_h1/SESSION.md:171, :331.
- **Beyond the head:** N/A.
- **Next check:** none. Keep norm only as a channel in a joint model.

### CM-8: activation energy flips sign between CIFAR and ImageNet

- **Relation:** partly rediscovered.
- **Literature:**
  - ID or confident inputs have larger norms (K1, R4, R6, R7).
  - Feature norms vary strongly across models (M4).
  - Nothing found explains the ImageNet reversal.
- **Atlas evidence:** results/margin_b1_vitb16/SESSION.md:431 (E8); CIFAR errors have lower energy (oriented AUC
  0.750-0.795 over seven nets), ImageNet errors higher (0.23-0.30) (`docs/reviews/EVAL_2026-09-23.md` §6, correction 10).
- **Beyond the head:** NO. Norm ≈ hidden max logit (R7).
- **Next check:** per-backbone sign calibration if norm is used at all.

### CM-11: nearest-centre vs head disagreement

- **Relation:** rediscovered as an idea; untested here.
- **Literature:** Trust Score (K3); DkNN (K2).
- **Atlas evidence:** atlas/invariants/landmarks.py:54; results/anomaly_h1/SESSION.md:185.
- **Beyond the head:** UNTESTED. It is a direct beyond-head test.
- **Next check:** score it as a detector on the ImageNet dumps, where 11-20% of samples disagree, with a Trust Score
  baseline.

### CM-1: margin as a confident-mistake flag

- **Relation:** rediscovered.
- **Literature:** MSP (H1); Trust Score (K3); the softmax-response result (F2).
- **Atlas evidence:** results/margin_v1_resnet20_s1/SESSION.md:107-173.
- **Beyond the head:** NO on CIFAR resnet20.
- **Next check:** add a Trust Score baseline.

### ST-3: batch Hotelling test in and off the class-mean span

- **Relation:** partly rediscovered.
- **Literature:** BBSD and two-sample tests (S1); label shift (S2); NMD (S3).
- **Atlas evidence:** results/anomaly_h1/SESSION.md:272-291.
- **Beyond the head:** T_par was never compared with BBSD.
- **Next check:** re-register against BBSD.

### ST-4: per-sample off-class-span residual e_perp

- **Relation:** a relative of known scores; the negative applies to a different shift type.
- **Literature:** ViM residual (R1); low-variance directions (M5); NECO (R2). These target *semantic* OOD.
- **Atlas evidence:** results/anomaly_h1/SESSION.md:339-350.
- **Beyond the head:** N/A.
- **Next check:** test ViM and NECO on semantic OOD (DO-4) before drawing conclusions from the covariate negative.

### ST-7: legacy Mahalanobis(CLD) and CLUSTER_DISTANCE axes

- **Relation:** rediscovered; untested in Atlas.
- **Literature:** M1-M4.
- **Atlas evidence:** MASTER_SUMMARY.md:36-47.
- **Beyond the head:** UNTESTED.
- **Next check:** re-enter only with normalisation and the head baselines.

---

## 6. Baselines Atlas should include next time

Most of these run on the pod from existing dumps without re-extraction:
- CIFAR logits = penult · W + b from the checkpoint's fc layer.
- `preds` already hold maxprob and argmax per split.
- The ImageNet (B1/B1b) `preds` hold the top-2 logits and the logit gap (atlas/extract_imagenet.py:555-573).

**Per-sample: novelty and correctness.**
1. MSP, max logit (H3), energy (H2), logit gap and softmax entropy. These are the free head baselines.
2. L2-normalised kNN (K1), with a sweep over k ∈ {1, 10, 50, 200} and over the bank sampling ratio. Keep the
   un-normalised Atlas variant for continuity, and report ||z|| as its own score (R7).
3. Mahalanobis at the penult only with tied covariance (M1); relative Mahalanobis (M2); Mahalanobis++ with
   L2-normalised features (M4).
4. ViM (R1) and NECO (R2). They are the standard joint feature-plus-head scores and the natural bridge to the primary
   question.
5. A deep ensemble built from checkpoints Atlas already has (U1, U2):
   - resnet20 s1-s4 plus the hub, 5 members;
   - resnet56 s1, s2 plus the hub.

   Score it by mean MSP, predictive entropy and member disagreement. Label it "same-recipe ensemble".
6. Trust Score (K3), for CM-1 and CM-11.

**Batch and split level: harm and familiarity.**

7. Batch-mean MSP and entropy; DoC (S4); ATC fitted on held-out clean data (S5); the predicted-class histogram and BBSD
   KS tests (S1, S2); NMD on batch-norm means (S3).

**Calibration.**

8. A split-conformal threshold (C1):
   - fit the bank on training data;
   - calibrate q95 on held-out clean rows that are used for nothing else;
   - report the realised clean FPR with a binomial or conformal CI;
   - repeat the same procedure for the MSP threshold.

**Protocol, so that "beyond the head" gets a direct answer.**

9. Report:
   - AUROC and FPR95 for novelty;
   - AUROC(correct vs wrong) and AURC for failure detection (F2);
   - Spearman and MAE against accuracy loss for harm.
10. Run a **nested-model test** on held-out folds: logistic(head scores) against logistic(head scores + density score),
    with a paired DeLong or likelihood-ratio test. A geometric signal "adds" only if the nested model wins on fresh
    seeds.
11. Use OpenOOD v1.5's CIFAR-10 groups: near-OOD is CIFAR-100 and TinyImageNet; far-OOD is MNIST, SVHN, Textures and
    Places365. Take held-out ID validation from the test set (C3), and avoid tuning on test OOD.
12. Keep the random-init nulls. In this family they are the cleanest proof that an effect is learned, as in DO-3 and
    DO-4.

---

## 7. Implications for the controller

1. **Use density for novelty, not for correctness.**
   - The literature and Atlas agree that feature density is a familiarity channel. It fires on semantic novelty and on
     covariate shift.
   - It is not a better per-sample trust score than the head (U3, F2; Atlas CM-1).
   - In the controller, density belongs to "escalate / do not adapt on semantic novelty" (DO-4; AX-3 INFO: CIFAR-100
     batches are never held). The per-sample trust score should come from the head, or from a joint head-plus-feature
     score such as ViM, NECO or SIRC, once the nested test in section 6 shows a gain.
2. **Normalise features, and calibrate thresholds on held-out, deployment-clean data.**
   - A train-referenced threshold over-alarms by a factor that grows with collapse: 1.8-2.3× in resnet20 and 3.3-3.7×
     in resnet56 (DO-3).
   - Use split-conformal calibration (C1). Re-calibrate after any fine-tuning or test-time adaptation, since collapse,
     and with it the radius scale, changes with training (C2; ANOMALY_H1.md:529).
3. **Treat a Gate-1 magnitude (H or h) as unproven until it beats confidence-based estimators.**
   - The published warning (S4) is that feature-distribution distances look good on synthetic corruption sweeps but do
     not transfer reliably across shift types. Confidence-based DoC and ATC are the baselines to beat.
   - Atlas's evidence is exactly the favourable case: between-split correlation 0.99 on CIFAR-10-C, and any penult shift
     statistic matches it.
   - Test H or h against batch-mean MSP, entropy, DoC, ATC and BBSD, across shift types and in streams (section 6, items
     7-10).
4. **Do not use density as a severity grade for high-frequency noise.**
   - Density plateaus or reverses at s3-s5 while the feature norm keeps falling (DO-2; R8).
   - If severity matters (for routing), read displacement magnitude or ||z||.
5. **TTA gating: the standard gate is head entropy (I3). Density is a candidate *complement*.**
   - It could keep semantically novel samples out of the adaptation batch.
   - No published feature-density TTA gate was found (section 8), so treat this as a research item for the closed-loop
     v0.
6. **The strongest case for density is on encoders with no class head.**
   - On a robot's self-supervised encoder, the head baselines do not exist. Feature density is then the only
     familiarity signal left, and it is known to work there (I2).
   - This argues for keeping density in the controller even if it adds little over the head on supervised CIFAR and
     ImageNet classifiers.
7. **Engineering.** kNN needs a stored bank. A 1% subsample with k scaled to match performs about as well as the full
   bank (K1), which makes on-robot latency and memory feasible. Privacy and memory costs of storing features are real
   (C3).
8. **HOLD semantics.** Density-based "unfamiliar" should map to "hold, output distrusted" or escalate. "Familiar with
   small h" maps to "hold, output trusted". These are the two triggers the evaluation asks CONTROLLER.md to separate
   (`docs/reviews/EVAL_2026-09-23.md` §6, correction 14).

---

## 8. Open literature gaps (not searched: the web-search budget was exhausted)

- A published study of kNN or Mahalanobis familiarity as a *monotone severity grade* on CIFAR-10-C or ImageNet-C. U3
  studies the correctness calibration of feature-density methods across severities, not monotonicity.
- Feature-density gating for test-time adaptation, and density-gated adaptive compute.
- A quantitative law linking a checkpoint's collapse level (NC1) to the false-alarm rate of a train-referenced density
  threshold. The DO-3 law is therefore "not found in prior art", on a limited search.
- Whether any work reports the ImageNet-vs-CIFAR sign flip of error-vs-correct feature energy (CM-8).

---

## 9. Item classifications (this topic's reading)

The merged, conflict-resolved classification is in `docs/knowledge/README.md`. This table keeps this topic's reading
and citations.

| item | known? | key citations | how prior work uses it |
|---|---|---|---|
| DO-1 | known-in-research | Sun, Ming, Zhu & Li 2022 (ICML, arXiv 2204.06507; kNN detector, Atlas omits L2 normalisation); Yang, Zhou & Liu 2022 Full-Spectrum OOD (arXiv 2204.05306); Zhang, Yang et al. 2023 OpenOOD v1.5 §6.2 (arXiv 2306.09301; near-OOD AUROC drops > 10% when covariate-shifted ID must be accepted); Postels et al. 2022 (ICML, arXiv 2107.00649; feature-density methods across CIFAR-10-C severities); Kang et al. 2024 (ICLR, arXiv 2310.00873; head predictions revert to the constant solution as shift grows); Ovadia et al. 2019 (NeurIPS, arXiv 1906.02530) | The literature treats the response of feature-space OOD scores to covariate shift as a known property. Full-spectrum work treats it as a nuisance to cancel; Postels uses severity sweeps to test calibration. Nobody uses kNN density as a graded severity axis in practice, and the head analogue (confidence and accuracy falling with severity) is the standard comparator. Atlas extends this with per-corruption s5 > s1 monotonicity replicated over 7 trained nets against random nulls; beyond-head is untested. |
| DO-2 | partly-known | van Amersfoort et al. 2021 feature collapse (arXiv 2102.11409); Kang et al. 2024 (ICLR, arXiv 2310.00873; late-layer feature norms shrink as shift grows, ResNet20/CIFAR-10); Sun et al. 2022 (kNN is sensitive to feature norm; normalisation improved FPR95 by 61%); Mukhoti et al. 2023 DDU (CVPR; density needs a regularised feature space) | The literature documents why feature density need not rise monotonically: OOD inputs can be mapped onto ID features (feature collapse), and feature norms shrink with shift. Fixes are spectral normalisation or bi-Lipschitz constraints, or L2 normalisation plus norm as a separate channel. The specific CIFAR-10-C noise plateau or reversal at s3-s5, seed-dependent while the resnet20 norm_ratio keeps falling, is an Atlas-local observation. |
| DO-3 | partly-known | Bates, Candès, Lei, Romano & Sesia 2023 (Annals of Statistics 51(1); arXiv 2104.08279; calibration data must be independent of score training); Papernot & McDaniel 2018 DkNN (arXiv 1803.04765; held-out calibration set); Zhang, Yang et al. 2023 OpenOOD v1.5 (held-out ID validation); Sun et al. 2022 (threshold at 95% ID TPR); Hui, Belkin & Nakkiran 2022 (arXiv 2202.08384; neural collapse on the train set but not the test set) | Standard practice calibrates OOD thresholds on held-out ID data at a target TPR, or with split-conformal p-values that carry finite-sample FPR guarantees, so the train-feature over-alarm is an expected consequence of violating that practice. The mechanism (train collapse stronger than test collapse) is published. The quantitative law sparse = 0.0224 − 0.108·log10 nc1 across depth-56 checkpoints was not found in prior art (limited search). |
| DO-4 | known-in-research | Sun et al. 2022 (ICML, arXiv 2204.06507); Lee et al. 2018 Mahalanobis (NeurIPS, arXiv 1807.03888); Ren et al. 2021 RMD (arXiv 2106.09022); Wang et al. 2022 ViM (CVPR, arXiv 2203.10807); Hendrycks & Gimpel 2017 MSP (ICLR, arXiv 1610.02136); OpenOOD v1.5 2023 (CIFAR-10 ResNet-18 near/far AUROC: KNN 90.64/92.96 vs MSP 88.03/90.73, MDS 84.20/89.72) | Post-hoc far/near-OOD detection with kNN or Mahalanobis on penult features is a benchmarked research method. It is scored by AUROC/FPR95 against the MSP, energy and max-logit baselines, with thresholds calibrated on held-out ID data. The expected gain of kNN over MSP at CIFAR-10 scale is about 2-3 AUROC points. Atlas reports only flag rates at a miscalibrated threshold, so the comparison is untested. |
| DO-5 | partly-known | Teerapittayanon, McDanel & Kung BranchyNet (arXiv 1709.01686); Kaya, Hong & Dumitras 2019 Shallow-Deep Networks (ICML, arXiv 1810.07052); Huang et al. MSDNet (arXiv 1703.09844) (metadata only; exit mechanism from memory); Sun et al. 2022 (1% kNN bank suffices); OpenOOD v1.5 (memory cost of feature banks) | Adaptive or early-exit compute is established, but gates on internal-classifier confidence. A density-gated cheap path was not found in the searches that could be run. It is untested in Atlas and costs bank memory and latency. |
| CD-4 | partly-known | Hui, Belkin & Nakkiran 2022 (arXiv 2202.08384); Haas, Yolland & Rabus 2023 (TMLR, arXiv 2209.08378); Ammar et al. 2024 NECO (ICLR, arXiv 2310.06823); Liu & Qin 2025 (CVPR, arXiv 2311.01479); Papyan, Han & Donoho 2020 (PNAS, arXiv 2008.08186; cited in landmarks.py:6-7) | The literature uses the neural-collapse level to explain and design OOD scores: NC plus L2 normalisation improves OOD, and NECO and NC-alignment detectors use the ETF geometry. It also notes that train-set collapse is stronger than test-set collapse. Using nc1 as a checkpoint covariate that predicts a density threshold's clean false-alarm rate is Atlas's extension. |
| CD-9 | partly-known | Ammar et al. 2024 NECO; Liu & Qin 2025; Haas et al. 2023; Wu et al. 2025 (ICLR, arXiv 2405.17816); Hui et al. 2022 | NC-OOD papers treat collapse as shaping the ID/OOD feature geometry and build detectors or training losses on it. A single collapse scalar accounting jointly for density false alarms, CKA, adjacency loss and margin-vs-distance was not found and remains an Atlas hypothesis. |
| LH-1 | known-in-research | Deng & Zheng 2021 AutoEval (CVPR, arXiv 2007.02915); Guillory et al. 2021 DoC (ICCV, arXiv 2107.03315; distributional distances such as Fréchet and MMD fail to reliably estimate performance under shift); Garg et al. 2022 ATC (ICLR, arXiv 2201.04234); Rabanser, Günnemann & Lipton 2019 Failing Loudly (NeurIPS, arXiv 1810.11953) | Label-free accuracy estimation from feature-distribution shift (AutoEval) is a research method. The standard baselines are confidence-based: average confidence, DoC, ATC. Published evidence warns that feature distances do not transfer reliably across shift types. Atlas's H is a feature-shift magnitude scored on CIFAR-10-C severities, the favourable setting, and was never compared with DoC, ATC or BBSD. |
| LH-2 | partly-known | Dong et al. 2022 Neural Mean Discrepancy (CVPR, arXiv 2104.11408); Rabanser et al. 2019 (two-sample batch tests; BBSD best; detection is not harm); Bates et al. 2023 (multiplicity/FDR for repeated conformal tests); Guillory et al. 2021; Garg et al. 2022 | Batch-level shift detection from activation statistics (NMD) or classifier outputs (BBSD) is known. It is scored by detection power against batch size at a fixed false-alarm rate. A harm-calibrated HOLD band on a normalised kNN log-radius shift is Atlas-specific, with no head comparator yet. |
| LH-3 | partly-known | Kang et al. 2024 (feature norms shrink with shift); Park et al. 2023 (ICCV, arXiv 2310.05316; feature norm acts as a hidden max logit); Dhamija, Günther & Boult 2018 (NeurIPS, arXiv 1811.04110); Sun et al. 2022 (ID features have larger norms than OOD) | The feature norm is used as a novelty or confidence-like OOD score, never as a harm-calibrated hold rule. Atlas's refutation (norm_ratio ≥ 0.95 splits and batches still losing > 10 pt) agrees with the literature: norm tracks shift and confidence, not harm. |
| CM-8 | partly-known | Park et al. 2023 (norm ~ hidden max logit); Liu & Qin 2025 (ID features farther from the origin); Sun et al. 2022 Fig. 4; Müller & Hein 2025 Mahalanobis++ (ICML, arXiv 2505.18032; feature norms vary strongly across models) | Feature norm and energy are used as OOD and confidence-like scores. Lower energy for errors on CIFAR matches norm ~ confidence. The ImageNet sign reversal is not explained by any work found, and cross-model norm variation (Mahalanobis++) warns against transferring a sign. |
| CM-11 | known-in-research | Jiang, Kim, Guan & Gupta 2018 Trust Score (NeurIPS, arXiv 1805.11783; agreement of the classifier with a density-filtered nearest-neighbour classifier); Papernot & McDaniel 2018 Deep kNN (arXiv 1803.04765; kNN label nonconformity with a held-out calibration set) | The literature uses the disagreement between the classifier and a nearest-neighbour or class-distance rule in feature space as a per-sample trust or credibility score for flagging errors. The Trust Score beat classifier confidence in its own experiments. Atlas has measured agreement only as a covariate; its use as a detector is untested. |
| CM-1 | standard-practice | Hendrycks & Gimpel 2017 MSP (ICLR, arXiv 1610.02136); Jaeger et al. 2023 (ICLR, arXiv 2211.15259; softmax response is the best failure detector overall); Jiang et al. 2018 Trust Score (class-distance ratio precedent) | Misclassification detection with head confidence is the standard baseline. kNN or class-distance trust scores are the research alternative. Atlas's margin reproduces MSP-level detection on CIFAR resnet20 and needs a Trust Score baseline. |
| ST-3 | partly-known | Rabanser et al. 2019 (NeurIPS; two-sample tests on representations and outputs, BBSD best); Lipton, Wang & Smola 2018 (ICML, arXiv 1802.03916; label-shift detection from black-box outputs); Dong et al. 2022 NMD | Batch shift detection by two-sample tests on classifier outputs or activation means is known. Label shift is detected from predicted-class distributions (BBSD). Splitting a batch-mean Hotelling test into class-span and complement components is an Atlas variant; it was never compared with BBSD. |
| ST-4 | known-in-research | Wang et al. 2022 ViM (residual outside the principal subspace); Kamoi & Kobayashi 2020 (arXiv 2003.00402; Mahalanobis works via directions not useful for classification); Ammar et al. 2024 NECO | Residual energy outside the principal or ETF subspace is a published semantic-OOD score, usually combined with logits. Atlas tested a class-span version as a per-sample covariate-drift sensor and ruled it out. That is a different shift type, so the negative does not contradict ViM or NECO; ViM and NECO on DO-4 remain untested. |
| ST-7 | known-in-research | Lee et al. 2018 Mahalanobis (NeurIPS); Ren et al. 2021 RMD; Fort, Ren & Lakshminarayanan 2021 (NeurIPS, arXiv 2106.03004); Müller & Hein 2025 Mahalanobis++ | Mahalanobis and cluster-distance axes are standard feature-distance OOD scores, benchmarked against MSP and energy with normalised features and held-out calibration. The legacy Atlas axes have never been re-measured under that protocol. |
