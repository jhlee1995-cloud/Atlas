# Confidence, top-2 margin and geometric error scores: prior art for Atlas

Last reviewed: 2026-09-23; part of the Atlas knowledge base (docs/knowledge/README.md)

Prior-art labels in this file (rediscovered, extends, known-in-research, not found in prior art, and so on) mark
starting points to adopt and refine for Atlas's use; they never down-rank an item (docs/knowledge/README.md §1).

- **Written against:** HEAD `343f651`.
- **Topic:** `confidence-margin`, inventory items CM-1 to CM-11. Where this literature also bears on DO-1, DO-4,
  CD-11, LH-1 and LH-2, those items are covered too.
- **Related notes:** `docs/knowledge/controller-functions-feasibility.md`; the inventory and the resolved
  classifications are in `docs/knowledge/README.md`; the programme evaluation is `docs/reviews/EVAL_2026-09-23.md`.

**Owner's question this note serves.** Which kinds of information a controller could use can be read out of a
backbone's internal geometry, how reliably, and does the geometry add anything beyond the output head? For this topic,
the question narrows to: does a class-centre geometric score flag the model's own errors, confident ones in
particular, better than the head's softmax or logits?

**How the citations were checked.** Every work cited below was either:
- **opened**: the arXiv, ar5iv, proceedings or CVF page was fetched and read; or
- **seen in a search result**: the result showed the authors, title and venue.

The "checked" column in section 2 says which. A venue marked "(venue from memory)" was not shown on the page that was
opened. A work cited only because another audit cites it is marked "(not re-verified)". Nothing else is cited.

The web-search budget ran out before a targeted search for prior work showing a *feature-space top-2 margin that beats
the head's logit margin inside confident strata*. The prior-art status of CM-5 is therefore "not found in prior art" on a
limited search. It is not a literature-complete claim.

---

## 1. What this family of methods measures

**The task.** Take a fixed, trained classifier with head input z (the penult; for a ViT, the final-norm CLS token) and
logits l = Wz + b. A *misclassification* or *failure* detector is a per-sample score s(x) that should rank correct
predictions above wrong ones. It is judged in one of two ways:
- as a ranking: AUROC with errors as positives, AUPR-Err, FPR at 95% TPR;
- as a *selective classifier* that abstains below a threshold: the risk-coverage curve and AURC.

This is not calibration. A score can rank well and be badly calibrated, or the other way round (Guo et al. 2017;
Zhu et al. 2022, who show that most calibration methods do not help, or actively hurt, failure prediction).

**Three families of scores:**

| family | scores | inputs needed |
|---|---|---|
| head output | MSP (max softmax probability, "softmax response"); max logit; **logit gap** (top-1 minus top-2 logit, "LogitsMargin"); softmax margin; entropy; Gini / DOCTOR; energy score (−T·logsumexp(l/T)); temperature-scaled and p-norm-normalised variants | logits only |
| feature geometry | nearest-class-mean distance; Mahalanobis to class means with tied covariance; relative Mahalanobis; kNN distance; **trust score** (distance to the nearest other class over distance to the predicted class); DkNN nonconformity; feature distance to the head's decision boundary (fDBD); feature or activation norm | reference features; labels for the class means |
| learned | ConfidNet (regresses the true-class probability); SelectiveNet; REL-U | extra training |

**The algebra that links Atlas's margin to the head.** This extends results/margin_b1_vitb16/SESSION.md:497-507. Let
d_k = |z − c_k| for class centres c_k, and g_k = c_k·z − |c_k|²/2.
- −d_k²/2 = g_k − |z|²/2. So the nearest-centre classifier is a *linear* classifier with weights c_k and biases
  −|c_k|²/2.
- The Atlas margin (atlas/invariants/margin.py:79, :118-122) is d2 − d1 = (d2² − d1²)/(d1 + d2) = 2(g1 − g2)/(d1 + d2).
  It is a linear-score gap divided by the local scale (d1 + d2)/2.
- The Euclidean distance from z to the nearest-centre boundary between c1 and c2 is (g1 − g2)/|c1 − c2| =
  (d2² − d1²)/(2|c1 − c2|). Atlas's `margin_norm` = (d2 − d1)/|c1 − c2| (margin.py:259) is this distance divided by
  (d1 + d2)/2. It is not the boundary distance itself.
- The head's own feature-space distance to its boundary between the top-2 classes is (l1 − l2)/|w1 − w2|. This is the
  per-class term of fDBD (Liu & Qin 2024).
- Under neural-collapse self-duality (NC3: the classifier rows align with the centred class means; Papyan et al. 2020),
  g1 − g2 and l1 − l2 agree up to scale and bias.
- **"Margin versus logit gap" therefore splits into two separable questions:**
  1. the numerator: centres or head weights;
  2. the denominator: normalising by (d1 + d2)/2, or not normalising.

  Atlas has not yet separated them (section 6).

**Epistemic versus aleatoric.** The deterministic-uncertainty literature assigns feature-space density or distance to
*epistemic* uncertainty (novelty) and softmax entropy to *aleatoric* uncertainty (ambiguity inside the training
distribution). It notes that density alone "confounds unambiguous and ambiguous iD samples" (Mukhoti et al. 2023, DDU).
Most in-distribution errors are aleatoric, so head scores are expected to win at in-distribution error detection.
Feature scores are expected to add mainly off-distribution.

**Atlas terms and their standard names:**

| Atlas term | standard name |
|---|---|
| type-b (wrong with maxprob > cut) | high-confidence error or overconfident failure. Standard evaluations do **not** pick the positives by the baseline's own score. |
| maxprob | MSP or softmax response |
| logit gap | LogitsMargin (Cattelan & Silva 2024) |
| margin (d2 − d1 to class means) | a centroid form of the trust-score idea (Jiang et al. 2018 use a ratio of distances to kNN-filtered class sets) |
| confidence-matched AUROC | conditional AUROC among predictions above a confidence threshold |
| Atlas "energy" (mean squared penult activation, margin.py:13) | an activation-norm score. **Not** the energy score of Liu et al. 2020. |

---

## 2. Key works

| # | citation | URL | checked | why it matters for Atlas |
|---|---|---|---|---|
| 1 | Hendrycks, D., Gimpel, K. (2017). A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks. ICLR 2017. | https://arxiv.org/abs/1610.02136 | opened (ar5iv) | The MSP baseline. Reports AUROC and AUPR-Succ/Err; MSP misclassification AUROC on CIFAR-10 is about 93%. |
| 2 | Hendrycks, D., Basart, S., Mazeika, M., et al. (2022). Scaling Out-of-Distribution Detection for Real-World Settings. ICML 2022. | https://arxiv.org/abs/1911.11132 | search | Max logit as a head score. |
| 3 | Liu, W., Wang, X., Owens, J., Li, Y. (2020). Energy-based Out-of-distribution Detection. NeurIPS 2020. | https://arxiv.org/abs/2010.03759 | search | Logit energy score. Atlas's "energy" is a different quantity. |
| 4 | Granese, F., Romanelli, M., Gorla, D., Palamidessi, C., Piantanida, P. (2021). DOCTOR: A Simple Method for Detecting Misclassification Errors. NeurIPS 2021. | https://arxiv.org/abs/2106.02395 | search | A head-only misclassification detector that works from soft predictions alone ("totally black box"). |
| 5 | Dadalto, E., Romanelli, M., Pichler, G., Piantanida, P. (2024). A Data-Driven Measure of Relative Uncertainty for Misclassification Detection. ICLR 2024. | https://arxiv.org/abs/2306.01710 | search | REL-U, learned on soft predictions. |
| 6 | Jiang, H., Kim, B., Guan, M. Y., Gupta, M. R. (2018). To Trust Or Not To Trust A Classifier. NeurIPS 2018. | https://arxiv.org/abs/1805.11783 | opened (ar5iv) | **The direct precedent for the Atlas margin.** Score = distance to the α-high-density set of the nearest other class over the distance to the predicted class's set. It was applied to the logit layer and the penultimate layer of MNIST, SVHN and CIFAR nets. The authors report "little or no improvement" over model confidence on high-dimensional CIFAR-10/100, and call CIFAR-100 "essentially a negative result". |
| 7 | Papernot, N., McDaniel, P. (2018). Deep k-Nearest Neighbors: Towards Confident, Interpretable and Robust Deep Learning. arXiv. | https://arxiv.org/abs/1803.04765 | opened (ar5iv) | Nonconformity is the count of kNN labels across layers that disagree with the candidate label. Confidence uses the runner-up p-value. Evaluated on OOD and adversarial inputs, not on misclassification alone. |
| 8 | Mandelbaum, A., Weinshall, D. (2017). Distance-based Confidence Score for Neural Network Classifiers. arXiv. | https://arxiv.org/abs/1709.09844 | search | Penultimate-embedding distance as a confidence score for predicting classification errors. Needs a distance-based loss or adversarial training to work well. |
| 9 | Lee, K., Lee, K., Lee, H., Shin, J. (2018). A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks. NeurIPS 2018. | https://papers.nips.cc/paper/7947-a-simple-unified-framework-for-detecting-out-of-distribution-samples-and-adversarial-attacks | search | Mahalanobis distance to class-conditional Gaussians: the whitened version of Atlas's class-centre geometry. |
| 10 | Ren, J., Fort, S., Liu, J., Roy, A. G., Padhy, S., Lakshminarayanan, B. (2021). A Simple Fix to Mahalanobis Distance for Improving Near-OOD Detection. arXiv (UDL workshop 2021). | https://arxiv.org/abs/2106.09022 | search | Relative Mahalanobis (RMD). |
| 11 | Sun, Y., Ming, Y., Zhu, X., Li, Y. (2022). Out-of-Distribution Detection with Deep Nearest Neighbors. ICML 2022. | https://arxiv.org/abs/2204.06507 | search | L2-normalised kNN distance. Atlas's `knn_density` is this without the normalisation. |
| 12 | Wang, H., Li, Z., Feng, L., Zhang, W. (2022). ViM: Out-Of-Distribution with Virtual-logit Matching. CVPR 2022. | https://openaccess.thecvf.com/content/CVPR2022/html/Wang_ViM_Out-of-Distribution_With_Virtual-Logit_Matching_CVPR_2022_paper.html | search | A **joint** feature-residual plus logit score: the design pattern for a joint detector. |
| 13 | Ammar, M. B., Belkhir, N., Popescu, S., Manzanera, A., Franchi, G. (2024). NECO: NEural Collapse Based Out-of-distribution detection. ICLR 2024. | https://arxiv.org/abs/2310.06823 | search | Neural-collapse geometry used as an OOD score, evaluated on ViT-B/16 among others. |
| 14 | Liu, L., Qin, Y. (2024). Fast Decision Boundary based Out-of-Distribution Detector. ICML 2024. | https://arxiv.org/abs/2312.11536 | opened | Closed-form feature distance to the linear head's boundaries, \|(w_pred − w_c)ᵀz + (b_pred − b_c)\| / \|w_pred − w_c\|, averaged over c and divided by \|z − μ_train\|. A head-weight geometric margin. Used for OOD, not for misclassification. |
| 15 | Mukhoti, J., Kirsch, A., van Amersfoort, J., Torr, P., Gal, Y. (2023). Deep Deterministic Uncertainty: A New Simple Baseline. CVPR 2023. | https://openaccess.thecvf.com/content/CVPR2023/html/Mukhoti_Deep_Deterministic_Uncertainty_A_New_Simple_Baseline_CVPR_2023_paper.html (arXiv 2102.11582) | opened (ar5iv) | GDA feature density for epistemic uncertainty; softmax entropy for aleatoric. Feature density confounds ambiguous in-distribution samples. |
| 16 | van Amersfoort, J., Smith, L., Teh, Y. W., Gal, Y. (2020). Uncertainty Estimation Using a Single Deep Deterministic Neural Network. ICML 2020. | https://arxiv.org/abs/2003.02037 | opened | DUQ: RBF distance to learned class centroids. Distance-to-centre confidence, obtained by training rather than post hoc. |
| 17 | Geifman, Y., El-Yaniv, R. (2017). Selective Classification for Deep Neural Networks. NeurIPS 2017. | https://arxiv.org/abs/1705.08500 | search | Reject option with a guaranteed risk at a chosen coverage, built on a confidence score such as softmax response. |
| 18 | Geifman, Y., Uziel, G., El-Yaniv, R. (2019). Bias-Reduced Uncertainty Estimation for Deep Neural Classifiers. ICLR 2019. | https://arxiv.org/abs/1805.08206 | opened (PDF) | In the final model, uncertainty estimates for *highly confident* points are "jittered"; earlier snapshots rank them better. The paper also proposes AURC, the area under the risk-coverage curve, and E-AURC, its excess over the optimal AURC, as the performance measure. |
| 19 | Corbière, C., Thome, N., Bar-Hen, A., Cord, M., Pérez, P. (2019). Addressing Failure Prediction by Learning Model Confidence. NeurIPS 2019. | https://arxiv.org/abs/1910.04851 | search | ConfidNet / TCP, a learned failure predictor. |
| 20 | Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. ICML 2017. | https://arxiv.org/abs/1706.04599 | search | Temperature scaling. Calibration is not the same as ranking. |
| 21 | Zhu, F., Cheng, Z., Zhang, X.-Y., Liu, C.-L. (2022). Rethinking Confidence Calibration for Failure Prediction. ECCV 2022. DOI 10.1007/978-3-031-19806-9_30. | https://arxiv.org/abs/2303.02970 | search | Most calibration methods are "useless or harmful" for failure prediction. |
| 22 | Zhu, F., Cheng, Z., Zhang, X.-Y., Liu, C.-L. (2023). OpenMix: Exploring Outlier Samples for Misclassification Detection. CVPR 2023. | https://arxiv.org/abs/2303.17093 | search | Outlier Exposure, strong for OOD, gives no gain in misclassification detection. OOD and misclassification detection are different problems. |
| 23 | Jaeger, P. F., Lüth, C. T., Klein, L., Bungert, T. J. (2023). A Call to Reflect on Evaluation Practices for Failure Detection in Image Classification. ICLR 2023. | https://arxiv.org/abs/2211.15259 | opened (abstract) | Across uncertainty, learned and OOD-style scorers, a simple softmax-response baseline was "overall best performing". Introduces the fd-shifts benchmark (https://github.com/IML-DKFZ/fd-shifts). |
| 24 | Cattelan, L. F. P., Silva, D. (2024). How to Fix a Broken Confidence Estimator: Evaluating Post-hoc Methods for Selective Classification with Deep Neural Networks. UAI 2024. | https://arxiv.org/abs/2305.15508 | opened | **Key for the ResNet50 contrast.** Compares MSP, MaxLogit, LogitsMargin, SoftmaxMargin, NegativeEntropy and NegativeGini on 84 ImageNet classifiers. Many classifiers have a "broken" MSP; p-norm-normalised max logit fixes it; underconfidence from soft-label recipes (label smoothing, mixup) is the suspected cause. |
| 25 | Xia, G., Laurent, O., Franchi, G., Bouganis, C.-S. (2025). Towards Understanding Why Label Smoothing Degrades Selective Classification and How to Fix It. ICLR 2025. | https://arxiv.org/abs/2403.14715 | search | Label smoothing regularises the max logit more when a prediction is likely correct, which degrades the ranking. A post-hoc logit-normalisation fix restores it. |
| 26 | Wei, H., Xie, R., Cheng, H., Feng, L., An, B., Li, Y. (2022). Mitigating Neural Network Overconfidence with Logit Normalization. ICML 2022. | https://proceedings.mlr.press/v162/wei22d.html | search | Logit norm grows during training and drives overconfidence. **Normalisation is a known lever on confidence quality.** |
| 27 | Galil, I., Dabbah, M., El-Yaniv, R. (2023). What Can We Learn From The Selective Prediction And Uncertainty Estimation Performance Of 523 ImageNet Classifiers? ICLR 2023. | https://arxiv.org/abs/2302.11874 | opened (abstract) | The training regime drives uncertainty quality: distillation helps, and a subset of ViTs is best. Reports AUROC, ECE, AURC and coverage at a target selective accuracy. |
| 28 | Papyan, V., Han, X. Y., Donoho, D. L. (2020). Prevalence of neural collapse during the terminal phase of deep learning training. PNAS 117(40). | https://arxiv.org/abs/2008.08186 | search | NC1-NC4. NC3 self-duality links class means to head weights; NC4 says the head's decision equals the nearest-class-centre decision. |
| 29 | Baldock, R., Maennel, H., Neyshabur, B. (2021). Deep Learning Through the Lens of Example Difficulty. NeurIPS 2021. | https://arxiv.org/abs/2106.09647 | search | *Prediction depth*, from layer-wise kNN probes, relates to an example's uncertainty, confidence and accuracy. |
| 30 | Xiao, Y., Beschastnikh, I., Rosenblum, D. S., Sun, C., Elbaum, S., Lin, Y., Dong, J. S. (2021). Self-Checking Deep Neural Networks in Deployment (SelfChecker). ICSE 2021. | https://arxiv.org/abs/2103.02371 | search | Layer-wise class densities (KDE) are compared with the final prediction, and an alarm fires on disagreement: 60.56% of wrong predictions flagged, at 2.04% false alarms on correct ones. **The direct precedent for CM-11.** |
| 31 | Hein, M., Andriushchenko, M., Bitterwolf, J. (2019). Why ReLU Networks Yield High-Confidence Predictions Far Away From the Training Data and How to Mitigate the Problem. CVPR 2019. | https://openaccess.thecvf.com/content_CVPR_2019/html/Hein_Why_ReLU_Networks_Yield_High-Confidence_Predictions_Far_Away_From_the_CVPR_2019_paper.html | search | High confidence arises far from the data, not only near boundaries. |
| 32 | Mickisch, D., Assion, F., Greßner, F., Günther, W., Motta, M. (2020). Understanding the Decision Boundary of Deep Neural Networks: An Empirical Study. arXiv. | https://arxiv.org/abs/2002.01810 | opened (abstract) | Decision boundaries move closer to natural images over training, measured in input space. |
| 33 | Elsayed, G. F., Krishnan, D., Mobahi, H., Regan, K., Bengio, S. (2018). Large Margin Deep Networks for Classification. NeurIPS 2018. | https://arxiv.org/abs/1803.05598 | search | Margins imposed at hidden layers: the geometric margin in feature space. |
| 34 | Jiang, Y., Krishnan, D., Mobahi, H., Bengio, S. (2019). Predicting the Generalization Gap in Deep Networks with Margin Distributions. ICLR 2019. | https://arxiv.org/abs/1810.00113 | search | Normalised margin distributions at several layers. |
| 35 | Beyer, L., Hénaff, O. J., Kolesnikov, A., Zhai, X., van den Oord, A. (2020). Are we done with ImageNet? arXiv. | https://arxiv.org/abs/2006.07159 | search | The ReaL labels that Atlas uses in E11. |
| 36 | Northcutt, C. G., Athalye, A., Mueller, J. (2021). Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks. NeurIPS 2021 Datasets and Benchmarks. | https://arxiv.org/abs/2103.14749 | opened | At least 6% of ImageNet validation labels are wrong. |
| 37 | Vasudevan, V., Caine, B., Gontijo-Lopes, R., Fridovich-Keil, S., Roelofs, R. (2022). When does dough become a bagel? Analyzing the remaining mistakes on ImageNet. NeurIPS 2022. | https://arxiv.org/abs/2205.04596 | opened (abstract) | "Nearly half" of top models' remaining mistakes are not mistakes. |
| 38 | Garg, S., Balakrishnan, S., Lipton, Z. C., Neyshabur, B., Sedghi, H. (2022). Leveraging Unlabeled Data to Predict Out-of-Distribution Performance. ICLR 2022. | https://arxiv.org/abs/2201.04234 | opened | ATC: label-free accuracy estimation from confidence. |
| 39 | Guillory, D., Shankar, V., Ebrahimi, S., Darrell, T., Schmidt, L. (2021). Predicting with Confidence on Unseen Distributions. ICCV 2021. | https://arxiv.org/abs/2107.03315 | opened | DoC (difference of confidences) for accuracy under shift. |
| 40 | Liang, S., Li, Y., Srikant, R. (2018). Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks. ICLR 2018. | https://arxiv.org/abs/1706.02690 | opened | ODIN: temperature scaling plus input perturbation. |
| 41 | Sun, Y., Guo, C., Li, Y. (2021). ReAct: Out-of-distribution Detection With Rectified Activations. NeurIPS 2021. | https://arxiv.org/abs/2111.12797 | opened | OOD inputs show distinctive penultimate activation patterns. |
| 42 | Yu, Y., Shin, S., Lee, S., Jun, C., Lee, K. (2023). Block Selection Method for Using Feature Norm in Out-of-distribution Detection. CVPR 2023. | https://arxiv.org/abs/2212.02295 | opened | Whether a feature norm separates OOD depends on the block. A block other than the last can be better. |

Also mentioned, not re-verified here: SAR (Niu et al. 2023, arXiv 2302.12400), an entropy-filtered, reset-guarded
test-time adaptation method; it was cited by the evaluation's science audit and is covered in
`shift-type-tta-monitoring.md`.

---

## 3. How these methods are used

### 3.1 Usage patterns

| use | typical score | status | notes |
|---|---|---|---|
| **Selective prediction / reject option** | MSP threshold; a risk guarantee at a chosen coverage (Geifman & El-Yaniv 2017) | **standard practice** | Choosing the head statistic post hoc (MSP vs max logit vs LogitsMargin vs p-norm max logit) on held-out labelled data is established (Cattelan & Silva 2024), and cheap. |
| **Misclassification / failure detection** | MSP; DOCTOR; ConfidNet; REL-U; OpenMix | MSP is the standard baseline; the rest are research | MSP remains hard to beat (Jaeger et al. 2023). OOD-oriented methods do not transfer to misclassification (Zhu et al. 2023; Mukhoti et al. 2023). |
| **OOD detection** | MSP, max logit, energy, ODIN, Mahalanobis, RMD, kNN, ViM, NECO, fDBD | the baselines are standard in research | This is where feature geometry adds most over the head (ViM, Mahalanobis, kNN). |
| **Failure detection under shift** | the same scores evaluated on shifted data (fd-shifts) | research | Rankings of scorers change under shift. Atlas has no shifted-data margin test: A3's E3 was never run. |
| **Label-free accuracy estimation** | average confidence, DoC, ATC | research, widely used as baselines | These are the head baselines for Atlas's LH-1 / LH-2 harm grade. None was compared. |
| **Model selection or comparison** | AURC or AUROC across checkpoints | research practice (Galil et al. 2023; Cattelan & Silva 2024) | Training recipe changes the ranking quality of MSP. |
| **TTA gating / monitoring** | entropy or confidence filters on which samples to adapt on; batch-mean confidence as a monitor | research (SAR, not re-verified here) | Atlas's own TENT ladder (ST-6) has no results yet. |
| **Training for better confidence** | ConfidNet; LogitNorm; large-margin losses; distance-aware models (DUQ, DDU) | research | These change the backbone, so they are out of Atlas's measure-only scope. |
| **Label auditing** | confident errors as candidate label errors | research and practice (Northcutt et al. 2021) | On ImageNet, a large share of "confident errors" are label problems (section 4). |

### 3.2 Metrics

- **AUROC with errors as positives.** This is the Atlas choice: oriented, with DeLong SEs and paired DeLong tests
  (margin.py:17-19).
- **AUPR-Err and AUPR-Succ** (Hendrycks & Gimpel 2017).
- **FPR at 95% TPR.**
- **AURC and E-AURC**, from the risk-coverage curve, plus normalised variants (NAURC in Cattelan & Silva 2024).
- **Selective risk at a fixed coverage**, or **coverage at a target risk**. These are the controller-relevant units:
  "at 90% coverage, what is the error rate?"
- **ECE**, for calibration only. It does not measure ranking.

Atlas reports oriented AUROC only. It has **no AURC, no FPR@95 and no risk at coverage**.

### 3.3 Known failure modes (with the Atlas echo)

1. **MSP resolution and saturation.** For overfit nets the confidence of highly confident points is poorly ordered
   (Geifman et al. 2019). Atlas echo: CIFAR maxprob is saturated, with 68-79% of samples ≥ 0.999 and 57-59% unique
   values (results/margin_b1_vitb16/SESSION.md:432). resnet20 also shows float ties (top_share_correct 0.075-0.091;
   results/atlas_v1_resnet56_s1/SESSION.md:561-564). Logit-based scores do not saturate. In the full-fit resnet56 nets
   ties are nearly absent (top_share_correct 0.0002, same lines), so saturation does not explain the E9 lead there.
2. **Soft-label training recipes break MSP ranking.** Label smoothing and mixup do this (Zhu et al. 2022; Cattelan &
   Silva 2024; Xia et al. 2025). Atlas echo: the ResNet50 IMAGENET1K_V2 contrast is strongly under-confident. At c* 0.5,
   auc_maxprob_typeb is 0.16 (results/margin_b1_vitb16/SESSION.md:473-487). On all errors the logit gap scores 0.849
   against maxprob 0.800 (results/margin_b1_resnet50/atlas.json `.per_layer.penult.margin_typeb.auc_{logitgap,maxprob}_wrong`).
3. **Calibration is not ranking.** Temperature scaling fixes ECE but is not a failure detector (Guo et al. 2017; Zhu et
   al. 2022).
4. **Feature-space OOD scores are weak at in-distribution misclassification.** See Jaeger et al. 2023, Zhu et al. 2023
   and Mukhoti et al. 2023. The trust score gives "little or no improvement" over confidence on CIFAR-10/100 (Jiang et
   al. 2018). Atlas echo: nearest-centre distance is below margin and maxprob everywhere on CIFAR
   (results/margin_v1_resnet20_s1/SESSION.md:229-245).
5. **Label noise caps apparent error detection on ImageNet.** At least 6% of labels are wrong (Northcutt et al. 2021),
   and about half of top models' "mistakes" are acceptable answers (Vasudevan et al. 2022). Atlas echo: 42-46% of ViT
   type-b are ReaL-correct. Restricting to ReaL-wrong type-b raises the margin AUC to 0.84-0.85
   (results/margin_b1_vitb16/SESSION.md:434).
6. **Selecting positives by confidence biases the comparison.** Scoring "type-b vs all correct" with type-b selected on
   maxprob builds maxprob's weakness into the test. Standard protocols use all errors, or the risk-coverage curve. Atlas
   found this itself: V1's ResNet50 reversal is a selection artifact, and on all errors margin − distance is flat,
   +0.087 to +0.091 across models (results/margin_b1_vitb16/SESSION.md:489-495).
7. **High confidence far from the data.** ReLU nets can be confident far from the training data (Hein et al. 2019). A
   top-2 score, whether head or geometric, is blind to this. It is a novelty problem (DO items), not a margin problem.
8. **Collapse removes distance information.** Under NC2 the centres become equidistant, and under NC4 the nearest centre
   equals the head's argmax (Papyan et al. 2020). Distance-type and softmax-type scores then converge. Atlas echo: CM-2
   and AH-3(a)-(c) (results/anomaly_h1/SESSION.md:175-191).

---

## 4. Known results relevant to Atlas

1. **MSP is a strong baseline for misclassification.** It gives about 93% AUROC on CIFAR-10 (Hendrycks & Gimpel 2017)
   and was the best overall in a large cross-field comparison (Jaeger et al. 2023). Atlas's resnet20 all-errors maxprob
   AUROC is 0.912-0.926. That is the expected level, and margin matches it (0.912-0.925).
2. **The trust score is the published precedent for Atlas's margin, and it reported the same negative on CIFAR.**
   "Little or no improvement" over model confidence on CIFAR-10/100 with penultimate or logit features (Jiang et al.
   2018). Atlas's A3 result (margin ~ maxprob, confidence-matched gap within ±0.002, Spearman 0.92-0.93;
   results/margin_v1_resnet20_s1/SESSION.md:167-173) **rediscovers this without citing it**. At HEAD 343f651 no tracked
   file cites misclassification-detection literature: a grep for Hendrycks, trust score, DOCTOR, Geifman and AURC finds
   hits only in the knowledge-base and review files written on 2026-09-23.
3. **MSP is often not the best head statistic.** LogitsMargin and p-norm-normalised max logit beat it on many ImageNet
   models, especially under soft-label recipes (Cattelan & Silva 2024; Xia et al. 2025). So Atlas findings of the form
   "margin beats maxprob" (CM-4: ViTs, resnet56) are **not** beyond-head findings until the best head statistic is also
   beaten. Atlas's own H2b rule encodes exactly this (ATLAS_STATUS.md:20; results/margin_b1_vitb16/SESSION.md:411-418).
4. **Top-2 structure is the published design for geometric error scores.** The trust score is a ratio of the top-2
   class distances, and DkNN's confidence uses the runner-up p-value. Neither paper compares its top-2 score with a
   single-distance score, so "a top-2 gap beats nearest-centre distance" is Atlas's own measurement (V1: margin −
   distance +0.114 to +0.119 on ViTs). The head's own logit gap reaches 0.769 / 0.773 against distance 0.674 / 0.688
   (results/margin_b1_vitb16/SESSION.md:511-517).
5. **Self-duality predicts margin ≈ logit gap.** NC3 (Papyan et al. 2020) predicts it, and Atlas measured it on all
   errors for the ViTs: margin − logit gap −0.001 to +0.005 (results/margin_b1_vitb16/SESSION.md:509). The premise holds
   only loosely: head_center_cos is 0.78 / 0.66 / 0.56, and the nearest centre agrees with the head for 0.89 / 0.88 /
   0.80 of samples (:501-506). head_center_cos does not order the matched-confidence effect: deitb (0.66) leads most.
6. **Ranking inside the confident region is where head confidence is weakest.** Geifman et al. 2019 report "jittered"
   estimates for highly confident points. This is exactly the stratum where Atlas's V3b finds margin ahead of the logit
   gap on the ViTs. The finding therefore sits in the one region where the literature expects head scores to be
   weakest. That makes it plausible, but it also means **normalisation** is a known alternative explanation (Wei et al.
   2022; Cattelan & Silva 2024) that must be ruled out (section 6).
7. **Joint feature-plus-logit scores exist for OOD, not for misclassification.** ViM combines a feature residual with
   logits; NECO and fDBD use feature geometry relative to the head. No published test was found that a *class-centre
   top-2 margin adds beyond the logit margin for in-distribution confident errors* (limited search). This is the space
   CM-5 occupies.
8. **ViT uncertainty quality depends on the training regime** (Galil et al. 2023). Both Atlas ViTs are DeiT-lineage
   recipes (torchvision vit_b_16 IMAGENET1K_V1 and DeiT-B fb_in1k; docs/plans/B1_VIT_MARGIN.md:21). Any "ViT" statement
   is therefore confounded with the recipe, and the non-DeiT reserve (B1c) is the right next unit.
9. **Intermediate layers resolve hard examples late** (prediction depth; Baldock et al. 2021). Atlas's CM-9 matches: no
   early confident-mistake signal, and penult ≥ early maximum + 0.10 in every run.
10. **Layer-wise geometry-vs-head disagreement is a published error detector.** SelfChecker flags 60.56% of errors at
    2.04% false alarms; DkNN and the trust score are disagreement scores too. Atlas measures the ingredient
    (`nearest_center_agrees_with_model`, landmarks.py:54, margin.py:276) but never scores it (CM-11).
11. **Activation-norm scores are not transferable by sign.** Whether a feature norm separates OOD depends on the block
    (Yu et al. 2023), and ReAct shows OOD inputs have distinctive penultimate activation patterns. Atlas's energy sign
    flip between CIFAR (errors lower; oriented AUC 0.750-0.795 over seven nets) and ImageNet (errors higher; 0.23-0.30)
    (CM-8) is consistent with this literature.

---

## 5. Relation to Atlas items

Relation codes: **R** = rediscovered, **E** = extends, **C** = contradicts, **U** = untested here.

| item | relation | literature | Atlas evidence | reading |
|---|---|---|---|---|
| CM-1 | **R** | MSP baseline (Hendrycks & Gimpel 2017); trust-score CIFAR negative (Jiang et al. 2018); NC3 (Papyan et al. 2020) | ATLAS_STATUS.md:19; results/margin_v1_resnet20_s1/SESSION.md:107-122, :167-173, :229-249 | A textbook result, reproduced carefully. The geometric margin is a restatement of head confidence on a CIFAR ResNet. |
| CM-2 | **E** (partly R) | NC2 / NC4 (Papyan et al. 2020); high-confidence jitter (Geifman et al. 2019); MSP not the best head statistic (Cattelan & Silva 2024) | results/atlas_v1_resnet56_s1/SESSION.md:531-564; results/margin_b1_vitb16/SESSION.md:432 | "Distance catches up with margin at full collapse" is what NC predicts for error ranking. The +0.006 / +0.012 / +0.020 lead over maxprob is **expected to vanish against the logit gap**. That is a falsifiable prediction, testable from dumped penult·W + b. |
| CM-3 | **R / E** | Top-2 structure of the trust score and DkNN; label noise on ImageNet (Beyer et al. 2020; Northcutt et al. 2021; Vasudevan et al. 2022); ViT uncertainty (Galil et al. 2023) | results/margin_b1_vitb16/SESSION.md:304-329, :434, :477-495, :511-517 | V1 measures "top-2 beats distance", which the trust-score and DkNN designs assume but neither paper tests against a single distance. The ReaL restriction (E11) extends it in line with the label-noise literature. |
| CM-4 | **R** (as mechanism) | Cattelan & Silva 2024; Xia et al. 2025; Zhu et al. 2022; Geifman et al. 2019 | results/margin_b1_vitb16/SESSION.md:331-335, :432; results/atlas_v1_resnet56_s1/SESSION.md:551-564 | "Adds over maxprob" is a known consequence of maxprob being a weak head statistic. H2b correctly refuses it as a beyond-head claim. |
| CM-5 | **E / U** (not found in prior art) | Nearest neighbours in design: ViM; fDBD; Mahalanobis (Lee et al. 2018); DDU. Head-only alternatives: LogitNorm (Wei et al. 2022); p-norm max logit (Cattelan & Silva 2024) | results/margin_b1_vitb16/SESSION.md:337-345, :360-372, :430, :497-518; ATLAS_STATUS.md:20 | The only pre-registered positive beyond-logit result (V3b, +0.008 to +0.037 AUROC in confident strata over cuts 0.5-0.8, 2 DeiT-lineage ViTs; predicted EQUIVALENT). Not yet separated from a pure normalisation effect, which a head-only normalised score could reproduce (section 6). Absent on ResNet50 (BELOW, −0.046 / −0.047). |
| CM-6 | **C** (partly) | NC theory predicts distance ≈ margin under collapse; NECO uses NC for OOD | results/anomaly_h1/SESSION.md:175-191, :461-473; results/margin_b1_vitb16/SESSION.md:489-495 | The NC direction holds within CIFAR ResNets. Across ImageNet models it runs backwards, and the all-errors lead is flat. The ViT-vs-CNN gap is calibration and selection, as the literature on recipe-driven confidence would predict. |
| CM-7 | **R** (restatement) | Confident errors have large head margins by construction; Hein et al. 2019; ImageNet label-noise papers | results/margin_v1_resnet20_s1/SESSION.md:251-262; results/margin_b1_vitb16/SESSION.md:317-329; MASTER_SUMMARY.md:53, :218 | The "ridges" reading has no support distinct from the confidence floor. Many ImageNet "deep in the wrong valley" cases are label problems. |
| CM-8 | **R** (consistent) | Block-dependent feature-norm scores (Yu et al. 2023); ReAct (Sun et al. 2021) | results/margin_v1_resnet20_s1/SESSION.md:183-185; results/margin_b1_vitb16/SESSION.md:431 | The sign flip is what the literature leads one to expect. Not a transferable per-sample signal. |
| CM-9 | **R** | Prediction depth (Baldock et al. 2021); DkNN and SelfChecker use multiple layers | results/margin_v1_resnet20_s1/SESSION.md:175-181; results/margin_b1_vitb16/SESSION.md:424-425 | Error information concentrates at the head input. The CLS token is the head's input by design. |
| CM-10 | **R / C** | fDBD boundary distance (Liu & Qin 2024); hidden-layer margins (Elsayed et al. 2018; Jiang et al. 2019) | results/margin_b1_vitb16/SESSION.md:429; atlas/invariants/margin.py:259 | `margin_norm` is not the nearest-centre boundary distance, (d2² − d1²)/(2\|c1 − c2\|). The refuted item does not test the literature's quantity. |
| CM-11 | **U** | Trust score (classifier vs modified NN agreement); DkNN; SelfChecker | atlas/invariants/landmarks.py:54; margin.py:276; results/margin_b1_vitb16/SESSION.md:501-506; results/anomaly_h1/SESSION.md:185 | A known-in-research detector that Atlas measured but never scored. It is the cheapest direct beyond-head test on the ImageNet dumps, where 11-20% of samples disagree. |
| DO-1 | **R** | Deep kNN (Sun et al. 2022); Mahalanobis (Lee et al. 2018) | ATLAS_STATUS.md:15 | Sun et al.'s detector without L2 normalisation. The head comparator (batch-mean MSP or energy) is untested. |
| DO-4 | **U** vs standard baselines | MSP, max logit, energy, Mahalanobis, kNN, ViM, NECO | results/atlas_v0_resnet20_cifar10/SESSION.md:133 | No standard OOD baseline was run. |
| CD-11 | **R** | NC4 (Papyan et al. 2020) | results/atlas_v1_resnet56_s1/SESSION.md:307-318 | Nearest-centre accuracy equals head accuracy: the NC4 prediction. |
| LH-1 / LH-2 | **U** vs head baselines | ATC (Garg et al. 2022); DoC (Guillory et al. 2021); average confidence | results/anomaly_h1/SESSION.md:157-173, :318-337 | Label-free accuracy estimation has standard head baselines. None was compared, so "beyond the head" is unknown. |

---

## 6. Baselines Atlas should include next time

**Everything below can be computed from existing artifacts** (while the RunPod volume survives; see
`docs/reviews/EVAL_2026-09-23.md` §7):
- **CIFAR:** the dumps hold float16 penult activations (atlas/extract_acts.py:176-186) and `preds` hold only argmax and
  maxprob (:188). The fc head is linear, so logits = penult·W + b from the checkpoint on the volume. Re-extracting 5,000
  images in float32 is cheaper and avoids float16 rounding.
- **ImageNet:** `preds` already carry logit_top1, logit_top2 and logit_gap (atlas/extract_imagenet.py:555-573). Full
  logits need the head weights from the torchvision/timm checkpoints.

### 6.1 Score set, grouped by what it reads

| group | score | formula / source |
|---|---|---|
| head only | MSP | Hendrycks & Gimpel 2017 |
| | MSP at a fitted temperature | Guo et al. 2017. The ranking can change with T, because MSP depends on the whole logit vector. |
| | max logit | Hendrycks et al. 2022 |
| | **logit gap** (LogitsMargin), softmax margin | Cattelan & Silva 2024 |
| | entropy; DOCTOR (Gini) | Granese et al. 2021 |
| | energy score | Liu et al. 2020 |
| | **p-norm-normalised max logit** | Cattelan & Silva 2024 |
| head weights plus a feature norm | head boundary distance (l1 − l2)/\|w1 − w2\| | fDBD per-class term (Liu & Qin 2024) |
| | logit gap / \|z\| | a normalisation control |
| class-centre geometry | d1 (distance) | current |
| | margin d2 − d1 | current |
| | unnormalised centre gap g1 − g2 = (d2² − d1²)/2 | section 1 |
| | nearest-centre boundary distance (d2² − d1²)/(2\|c1 − c2\|) | section 1 |
| | centroid trust score d2/d1 | Jiang et al. 2018, with class means instead of α-filtered sets |
| | Mahalanobis top-1 and top-2 difference | Lee et al. 2018, tied covariance. The ViT reference half has 25,000 samples against 768 dimensions, so it is estimable. |
| | relative Mahalanobis | Ren et al. 2021 |
| other feature geometry | L2-normalised kNN distance | Sun et al. 2022 |
| | DkNN nonconformity | Papernot & McDaniel 2018 |
| | **nearest-centre ≠ argmax flag** (CM-11) | SelfChecker-style |
| | ViM, NECO | OOD references for the joint-score design |

### 6.2 Protocol changes, each tied to a literature lesson

1. **Primary endpoint on all errors, not on type-b vs all correct** (failure mode 6). Report AUROC, **AURC**, FPR@95
   and risk at 80/90/95% coverage. Keep the confidence-matched and stratified comparisons as pre-registered
   secondaries.
2. **Freeze the head comparator before confirmation.** On discovery runs, pick the best head-only statistic from the
   head-only group above, and pre-register it as the bar. H2b already demands the logit gap. The literature says the
   p-norm max logit belongs in that set, especially for soft-label recipes.
3. **Test incremental value with a joint detector** (the ViM pattern). Fit logistic regressions on the reference split
   and score them on the test split with swap twins: head-only features against head plus geometry. Compare
   cross-fitted AUROC and AURC with a paired bootstrap, and pre-register the margin. This is the test the evaluation
   asks for (`docs/reviews/EVAL_2026-09-23.md` §9, the B1c step). A win in a conditional AUROC does not by itself show a gain
   over the whole population.
4. **A normalisation ablation for CM-5**, run on the existing ViT and ResNet50 dumps. Compare, in the confident strata:

   | score | varies |
   |---|---|
   | g1 − g2 | centre numerator, no normaliser |
   | margin = (g1 − g2)/((d1 + d2)/2) | centre numerator, centre normaliser |
   | l1 − l2 | head numerator, no normaliser |
   | (l1 − l2)/((d1 + d2)/2) | head numerator, centre normaliser |
   | (l1 − l2)/\|z\| | head numerator, feature-norm normaliser |
   | (l1 − l2)/\|w1 − w2\| | head boundary distance |
   | p-norm max logit | head-only normalised score |

   How to read it:
   - If the head numerator with the centre normaliser matches margin, **the lead is normalisation, not centre
     geometry**.
   - If the p-norm max logit also matches, the lead is available **from the head alone**, and CM-5 is not a beyond-head
     result.
   - If only centre numerators carry it, the centres hold information the head weights lack. That is the
     primary-question positive.

5. **Separate the recipe from the architecture.** Add a non-label-smoothed CNN, such as torchvision ResNet50
   IMAGENET1K_V1 (an older recipe; check it before use), next to the V2 weights, and a non-DeiT ViT (the augreg reserve,
   B1c). The literature ties MSP breakage to soft-label recipes, not to architectures.
6. **Label-noise sensitivity as standard.** Report ImageNet results on ReaL-wrong positives next to the original labels
   (E11 already does this; make it a co-primary).
7. **Shift.** Rerun the margin, logit gap and joint detector on the corruption splits (A3's E3, never run), fd-shifts
   style. Confidence-score rankings are known to move under shift.
8. **CM-11 as a detector.** On the ImageNet dumps, compute the AUROC and AURC of the disagreement flag, and its
   precision at the flag rate. Also test its value in the joint model of step 3. Read it against the SelfChecker
   operating point (60.56% of errors at 2.04% false alarms, on different models and data, so as context only).

---

## 7. Implications for the controller

1. **Default per-sample trust score (Gate 1, "hold, output distrusted"): the best head statistic for that checkpoint,
   chosen once offline on held-out labelled data.** This is standard practice (selective classification with post-hoc
   estimator choice). Atlas's ResNet50 shows why the choice matters: logit gap 0.849 against MSP 0.800 on all errors.
   On overfit CIFAR heads, prefer a logit-based score to MSP, because of saturation. Use a coverage-risk threshold
   (Geifman & El-Yaniv 2017) rather than a fixed maxprob cut such as 0.7.
2. **Geometry's per-sample role inside the training distribution is, at best, a small second axis.**
   - CM-5 is worth +0.008 to +0.037 AUROC inside confident strata on two DeiT-lineage ViTs, and it is negative on
     ResNet50.
   - It becomes controller-relevant only if three things hold: it survives the normalisation ablation (6.2 step 4), it
     improves a joint detector's AURC (step 3), and it replicates on a non-DeiT ViT at cut 0.7 (B1c).
   - Until then, the controller should not spend compute or complexity on the class-centre margin for
     in-distribution errors.
3. **Where geometry is more likely to earn its place is novelty, not error.** The literature consistently assigns
   feature-space density and distance to epistemic uncertainty (DDU, Mahalanobis, kNN, ViM) and softmax to aleatoric
   uncertainty. Head confidence is structurally blind to confident predictions made far from the data (Hein et al.
   2019). The geometry-beyond-head question is therefore better posed on the DO and LH items, with head baselines (MSP,
   energy, ATC, DoC), than on CM.
4. **The cheapest direct test of "geometry beyond the head" is CM-11.** A geometry-vs-head disagreement flag is a
   published detector type. It needs no new data on ImageNet, and it maps directly onto a "hold, output distrusted"
   trigger. On collapsed CIFAR heads it has almost no capacity (0.3% disagree), so its value depends on the backbone.
5. **Label noise sets a ceiling on what any trust score can show on ImageNet.** Nearly half of ViT type-b are
   ReaL-correct. A controller tuned to distrust them would escalate many acceptable answers. Controller-facing
   evaluations should use ReaL or multi-label ground truth.
6. **Per-checkpoint calibration is unavoidable.** The best head statistic depends on the training recipe (Cattelan &
   Silva 2024; Xia et al. 2025). Whether margin, distance or MSP ranks best depends on the collapse level (CM-2, CM-6).
   Any retraining or fine-tuning, including test-time adaptation, invalidates the choice. Re-select after every change.
7. **Headless encoders (the robot case, DINO/CLIP in B2).** Where there is no class head, class-centre or kNN geometry
   is the only per-sample score available. The right comparator there is the logit gap of a linear probe fitted on the
   same reference features, not "no baseline". CM work transfers to that setting as "geometry stands in for a missing
   head", not as "geometry beats the head".
8. **Clean-data only.** Every CM number is on clean data. The controller's adapt/hold/escalate decisions happen under
   shift, where the literature (fd-shifts) shows confidence scorers re-rank. Nothing in CM licenses a shifted-data claim
   yet.

---

## 8. Item classifications (this topic's reading)

The merged, conflict-resolved classification is in `docs/knowledge/README.md`. This table keeps this topic's reading
and citations.

| item | known? | key citations | how prior work uses it |
|---|---|---|---|
| CM-1 | standard-practice | Hendrycks & Gimpel 2017 (ICLR), arXiv 1610.02136: MSP baseline, CIFAR-10 misclassification AUROC about 93%. Jiang, Kim, Guan & Gupta 2018 (NeurIPS), arXiv 1805.11783: trust score, a ratio of top-2 class distances in penult or logit space, gives "little or no improvement" over confidence on CIFAR-10/100. Jaeger et al. 2023 (ICLR), arXiv 2211.15259: softmax response is the best overall. Papyan, Han & Donoho 2020 (PNAS), arXiv 2008.08186: NC3 self-duality. | Thresholding MSP is the standard reject option and selective-prediction trigger, and MSP is the standard misclassification baseline. The Atlas centroid margin rediscovers the trust score's own CIFAR negative: on a CIFAR ResNet it is a geometric restatement of head confidence. |
| CM-2 | partly-known | Papyan et al. 2020, arXiv 2008.08186 (NC2 equidistant centres; NC4 nearest-centre equals the head). Geifman, Uziel & El-Yaniv 2019 (ICLR), arXiv 1805.08206: confidence estimates of highly confident points are "jittered" in the final model. Cattelan & Silva 2024 (UAI), arXiv 2305.15508: MSP is often not the best head statistic. | That distance-type and softmax-type scores converge under collapse follows from neural-collapse theory. Nobody uses it as a per-sample tool. The resnet56 margin-over-maxprob lead is expected, from the literature, to vanish against a logit-based head score. Standard practice is to pick the head statistic post hoc on held-out data. |
| CM-3 | known-in-research | Jiang et al. 2018, arXiv 1805.11783 (top-2 distance ratio). Papernot & McDaniel 2018, arXiv 1803.04765 (DkNN, runner-up p-value confidence). Galil, Dabbah & El-Yaniv 2023 (ICLR), arXiv 2302.11874 (ViT uncertainty depends on training regime). Beyer et al. 2020, arXiv 2006.07159 (ReaL). Northcutt, Athalye & Mueller 2021, arXiv 2103.14749. Vasudevan et al. 2022, arXiv 2205.04596. | Top-2 scores (logit margin, trust score) are routinely used for selective prediction and error flagging; the logit margin (LogitsMargin) is benchmarked on 84 ImageNet classifiers (Cattelan & Silva 2024, arXiv 2305.15508). Neither the trust-score nor the DkNN paper compares its top-2 score with a single distance, so V1's "top-2 beats distance" is Atlas's own measurement. Label-noise papers explain why ReaL-restricted confident errors give higher AUC. The ViT result is confounded with the DeiT training recipe. |
| CM-4 | partly-known | Cattelan & Silva 2024 (UAI), arXiv 2305.15508: LogitsMargin and p-norm max logit beat MSP on many ImageNet classifiers; soft-label recipes (label smoothing, mixup) are their suspected cause of a broken MSP. Xia, Laurent, Franchi & Bouganis 2025 (ICLR), arXiv 2403.14715: label smoothing degrades selective classification. Zhu, Cheng, Zhang & Liu 2022 (ECCV), arXiv 2303.02970. Geifman et al. 2019, arXiv 1805.08206. | Post-hoc replacement of MSP with a better head statistic is an established selective-classification practice. "Margin beats maxprob" is therefore expected, and it is not a beyond-head result unless the best head statistic (logit gap, p-norm max logit) is also beaten. This is Atlas's H2b rule. |
| CM-5 | not-found-in-prior-art | Closest prior art. Joint feature-plus-logit OOD score: ViM (Wang, Li, Feng & Zhang, CVPR 2022). Feature-space distance to the head's boundaries: fDBD (Liu & Qin, ICML 2024), arXiv 2312.11536. Class-mean Mahalanobis: Lee et al., NeurIPS 2018. Density vs softmax complementarity: DDU (Mukhoti et al., CVPR 2023). Competing head-only explanation: logit normalisation (Wei et al., ICML 2022; Cattelan & Silva 2024). The search was limited, because the web-search budget ran out before a targeted query. | No published test found of a class-centre top-2 margin adding beyond the head's logit margin for in-distribution confident errors. Related joint scores are used for OOD detection only. It must be separated from a pure normalisation effect and tested as a joint detector (AURC) on a non-DeiT ViT before any controller use. |
| CM-6 | partly-known | Papyan et al. 2020, arXiv 2008.08186 (NC predicts distance ~ margin under collapse). NECO (Ammar et al., ICLR 2024), arXiv 2310.06823 (NC geometry used for OOD). Cattelan & Silva 2024 and Xia et al. 2025 (the training recipe, not the architecture, drives confidence quality). | The collapse-to-score-choice mapping follows from NC theory within one family. It is not used as a score-selection rule in practice. Atlas's cross-ImageNet data run against it, consistent with recipe-driven calibration effects. Standard practice picks the score empirically per checkpoint. |
| CM-7 | known-in-research | Hein, Andriushchenko & Bitterwolf 2019 (CVPR): high confidence far from the data. Top-2 margin equals confidence by construction (NC3, Papyan et al. 2020). ImageNet confident errors are often label problems (Beyer et al. 2020; Northcutt et al. 2021; Vasudevan et al. 2022). | Not used as a detector. The "ridge" ratio is a restatement of the confidence cut, and the literature explains confident errors by far-from-data overconfidence or label noise, not by ridge location. |
| CM-8 | partly-known | Yu, Shin, Lee, Jun & Lee 2023 (CVPR), arXiv 2212.02295: feature-norm OOD separation depends on the block. Sun, Guo & Li 2021 (NeurIPS), arXiv 2111.12797 (ReAct): distinctive penultimate activation patterns on OOD inputs. No source found for the specific dataset sign flip on misclassification. | Activation and feature-norm scores are used for OOD detection with per-model or per-block selection. Their non-transferable sign is consistent with the literature. As a per-sample misclassification signal it is not used in practice. |
| CM-9 | known-in-research | Baldock, Maennel & Neyshabur 2021 (NeurIPS), arXiv 2106.09647: prediction depth, where hard examples are resolved late. Papernot & McDaniel 2018 (DkNN) and SelfChecker (Xiao et al., ICSE 2021, arXiv 2103.02371): multi-layer checks. | Intermediate layers are used in research through layer-wise kNN or density agreement with the final prediction, not as standalone early confident-error sensors. Atlas's finding that error information concentrates at the head input agrees with prediction-depth results. |
| CM-10 | known-in-research | Liu & Qin 2024 (ICML), arXiv 2312.11536: fDBD, the normalised feature distance to the linear head's decision boundaries. Elsayed et al. 2018 (NeurIPS), arXiv 1803.05598. Jiang, Krishnan, Mobahi & Bengio 2019 (ICLR), arXiv 1810.00113: normalised hidden-layer margins. | Normalised feature-space boundary distances are used for OOD detection (fDBD) and for studying generalisation. Atlas's margin_norm is not the nearest-centre boundary distance (d2² − d1²)/(2\|c1 − c2\|), so the refuted item did not test the literature's quantity. |
| CM-11 | known-in-research | Jiang et al. 2018, arXiv 1805.11783: trust score as agreement between the classifier and a modified nearest-neighbour classifier. Papernot & McDaniel 2018, arXiv 1803.04765: DkNN label disagreement. Xiao et al. 2021 (ICSE), arXiv 2103.02371: SelfChecker, layer-wise density vs final prediction, 60.56% of errors flagged at 2.04% false alarms. | Geometry-vs-head disagreement is a published misclassification and deployment-monitoring alarm, with an alternative-prediction "advice" mode in SelfChecker. Atlas measured the ingredient but never scored it. It is the cheapest direct beyond-head test on the existing ImageNet dumps. |
| DO-1 | known-in-research | Sun, Ming, Zhu & Li 2022 (ICML), arXiv 2204.06507: deep kNN OOD detection with L2 normalisation. Lee et al. 2018 (NeurIPS): class-conditional Mahalanobis. | kNN and Mahalanobis feature distances are standard OOD baselines. Atlas's sparse_frac is Sun et al.'s detector without normalisation. It was never compared against a head baseline such as batch-mean MSP or energy. |
| DO-4 | standard-practice | Hendrycks & Gimpel 2017 (MSP); Hendrycks et al. 2022 (max logit, arXiv 1911.11132); Liu et al. 2020 (energy, arXiv 2010.03759); Lee et al. 2018 (Mahalanobis); Sun et al. 2022 (kNN); ViM (CVPR 2022); NECO (ICLR 2024). | Far-OOD novelty detection with these scores is standard in the OOD literature. Atlas ran none of the standard baselines, so whether the geometry adds beyond the head is unknown. |
| CD-11 | known-in-research | Papyan, Han & Donoho 2020 (PNAS), arXiv 2008.08186: NC4, where the classifier's decision collapses to the nearest class centre. | That the nearest-centre accuracy equals the head's accuracy is the NC4 prediction. It is used as a collapse diagnostic, not as a controller signal. |
| LH-1 | partly-known | Garg et al. 2022 (ICLR), arXiv 2201.04234: ATC. Guillory et al. 2021 (ICCV), arXiv 2107.03315: DoC. Average confidence is the usual baseline. | Label-free accuracy estimation under shift from head confidence is an established research task with standard head baselines. Atlas's geometric harm grade H was never compared with ATC, DoC or average confidence, so its beyond-head value is untested. |
| LH-2 | partly-known | Garg et al. 2022, arXiv 2201.04234 (ATC); Guillory et al. 2021, arXiv 2107.03315 (DoC). | Batch-level accuracy estimation from confidence is the head-side counterpart of the per-batch harm grade h(B). Batch-mean MSP, ATC and DoC are the missing comparators. |
