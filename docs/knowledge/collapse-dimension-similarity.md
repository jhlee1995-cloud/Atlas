# Collapse, dimension and representation similarity: prior art for Atlas

Last reviewed: 2026-09-23; part of the Atlas knowledge base (docs/knowledge/README.md)

Prior-art labels in this file (rediscovered, extends, known-in-research, not found in prior art, and so on) mark
starting points to adopt and refine for Atlas's use; they never down-rank an item (docs/knowledge/README.md §1).

- **Written against:** Atlas HEAD 343f651.
- **Scope:** neural collapse and layer-wise separation, intrinsic dimension, and representation similarity. It covers
  the Atlas inventory items built on them (CD-1 to CD-11) and the items this literature bears on (CM-2, CM-5, CM-6, CM-8,
  CM-11, DO-3, PC-3, PC-6, ST-3, ST-6, LH-1).

**Verification legend** (used for every citation):
- **[V]**: the abstract or landing page was opened.
- **[S]**: title, authors, venue and URL were confirmed only from search-result metadata. The abstract was not opened.
- **[U]**: not confirmed. Check it before citing.

Nothing here is cited from memory without a tag.

**Headline for the primary question.** This family of methods reads **checkpoint-level** (population) information from
the backbone: how collapsed the classes are, how many directions the data uses, and how similar two networks are. It
reads per-input information only in a few derived forms (per-sample LID, nearest-centre versus head disagreement,
sample-wise feature/classifier alignment).

Atlas measured the checkpoint-level quantities reliably; nc1 and the ID profile are seed-stable. Most of them restate
published results. The one place where the geometry clearly carries information that head *accuracy* does not is the
collapse level at matched accuracy (CD-4), which also predicts the density detector's false-alarm rate (DO-3). **No item
in this family has been compared with a head-side counterpart other than accuracy.** Those counterparts are head
disagreement across seeds, confusion or weight similarity, and calibration. So "does geometry add beyond the head?" is
still **open** for every CD item. The cheapest decisive tests use dumps that already exist (section 6).

---

## 1. What this family of methods measures

Three sub-families. All are computed from layer activations, usually on a labelled reference set.

**(a) Class-conditional geometry: neural collapse (NC) and layer-wise separation.** Papyan, Han & Donoho (2020) define
four properties at the last hidden layer in the terminal phase of training (TPT: training continued after the training
error reaches zero):

| NC property | What it states | Atlas field |
|---|---|---|
| NC1 | within-class variability collapses; measured as Tr(Σ_W Σ_B⁺)/K | `neural_collapse.nc1` (atlas/invariants/landmarks.py:59-82) |
| NC2 | the centred class means form a simplex equiangular tight frame (ETF): equal norms, equal pairwise angles of −1/(K−1) | `etf_deviation`, `class_mean_norm_cv` (landmarks.py:71-82); the centre-distance CV |
| NC3 | the classifier rows align with the class means up to scale (self-duality) | `meta.head_center_cos` in B1 (results/margin_b1_vitb16/SESSION.md:503) |
| NC4 | the head's decision equals the nearest-class-centre (NCC) decision | `nearest_center_agrees_with_model` (landmarks.py:54); `nearest_center_acc_test` |

Layer-wise extensions measure the same ratio at every layer:
- He & Su's separation fuzziness D = Tr(S_w S_b⁺), which is Atlas's nc1 times K;
- intermediate-layer NC;
- NCC accuracy per layer.

Atlas's `sep_ratio` (nearest-other centre over RMS within-class radius), the bridge ratio, `commit_layer` and the
class-adjacency matrix (atlas/invariants/adjacency.py:23-33) are further class-conditional summaries in this family.

**(b) Dimension.**
- **Intrinsic dimension (ID)** is the number of parameters needed to describe the data manifold locally. Atlas uses
  TwoNN, which is based on the ratio of each point's second to first nearest-neighbour distance
  (atlas/invariants/dimension.py:40-70).
- **Linear effective dimension** comes from the PCA spectrum: participation ratio and dim95 (dimension.py:18-38).
- **Per-sample local ID (LID)** is the dimension of the neighbourhood around a single input.
- Atlas's control, **ID_gauss**, is TwoNN on a Gaussian with the same PCA spectrum (AH-8,
  results/anomaly_h1/SESSION.md:256-270). It separates "nonlinear" ID from what the spectrum already implies. The
  control is published: Ansuini et al. (2019, Fig. 5B) ran TwoNN on a synthetic Gaussian with the same second-order
  moments as the last hidden layer of VGG-16. Atlas applies it at every tap.

**(c) Representation similarity.** These methods compare two representations of the same inputs:
- **CKA** (centred kernel alignment; Atlas uses the linear version, `linear_cka`, in atlas/compare.py:166, :185 and
  atlas/invariants/flow.py:21-39);
- CCA variants (SVCCA, PWCCA) and orthogonal Procrustes;
- **relative representations** (each point written as cosines to shared anchors; Atlas uses class centres as anchors,
  compare.py:15-20);
- unit matching (convergent learning);
- **functional** similarity: error consistency (compare.py:198-206) and prediction disagreement.

**What they are not.** Except for LID, NC3/NC4 disagreement and sample-wise alignment, these are **population
statistics**: one number per checkpoint, layer or batch. Most need class labels on the reference set, and NC is by
definition a **training-set** phenomenon. None is a per-input trust score as defined.

---

## 2. Key works

### 2.1 Neural collapse: core and theory

| Citation | URL | Tag | One-line content |
|---|---|---|---|
| V. Papyan, X.Y. Han, D.L. Donoho (2020). *Prevalence of neural collapse during the terminal phase of deep learning training.* PNAS 117(40):24652-24663 | https://arxiv.org/abs/2008.08186 · https://doi.org/10.1073/pnas.2015509117 | [V] | Defines NC1-NC4 across 3 architectures and 7 datasets; claims better generalization, robustness and interpretability in the TPT. Cited in atlas/invariants/landmarks.py:6-8. |
| X.Y. Han, V. Papyan, D.L. Donoho (2022). *Neural collapse under MSE loss: proximity to and dynamics on the central path.* ICLR 2022 | https://arxiv.org/abs/2106.02073 | [S] | NC under MSE loss; the "central path" dynamics. |
| Z. Zhu, T. Ding, J. Zhou, X. Li, C. You, J. Sulam, Q. Qu (2021). *A geometric analysis of neural collapse with unconstrained features.* NeurIPS 2021 | https://arxiv.org/abs/2105.02375 | [S] | Under the unconstrained-features model with cross-entropy and weight decay, the only global minimizers are simplex ETFs; all other critical points are strict saddles. |
| V. Kothapalli, E. Rasromani, V. Awatramani (2023). *Neural collapse: a review on modelling principles and generalization.* TMLR | https://arxiv.org/abs/2206.04041 | [S] | Survey of NC models and of what NC implies for generalization and transfer. |
| J. Zhou, C. You, X. Li, K. Liu, S. Liu, Q. Qu, Z. Zhu (2022). *Are all losses created equal: a neural collapse perspective.* NeurIPS 2022 | https://arxiv.org/abs/2210.02192 | [V] | Cross-entropy, label smoothing, focal loss and MSE all reach NC when the network is large and converged, with largely identical test performance. |
| J. Jiang, J. Zhou, P. Wang, Q. Qu, D.G. Mixon, C. You, Z. Zhu (2024). *Generalized neural collapse for a large number of classes.* ICML 2024 (PMLR 235) | https://arxiv.org/abs/2310.05351 | [S] | When the number of classes exceeds the feature dimension, a simplex ETF is impossible; features instead maximize the minimum one-vs-rest margin. |
| C. Fang, H. He, Q. Long, W.J. Su (2021). *Exploring deep neural networks via layer-peeled model: minority collapse in imbalanced training.* PNAS 118(43):e2103091118 | https://arxiv.org/abs/2101.12699 | [S] | Under class imbalance, minority-class classifiers collapse together. |
| T. Galanti, A. György, M. Hutter (2022). *On the role of neural collapse in transfer learning.* ICLR 2022 | https://arxiv.org/abs/2112.15121 | [S] | NC on many source classes explains why a plain classifier's features transfer to few-shot tasks on new classes. |

### 2.2 Layer-wise and intermediate collapse, training length, limits

| Citation | URL | Tag | One-line content |
|---|---|---|---|
| H. He, W.J. Su (2023). *A law of data separation in deep learning.* PNAS 120(36) | https://arxiv.org/abs/2210.17020 · https://doi.org/10.1073/pnas.2221704120 | [V] (PMC full text) | **Law of equi-separation**: D_l ≈ ρ^l D_0, with D = Tr(S_w S_b⁺) on **training** data. Holds for feedforward, VGG/AlexNet, ResNet and DenseNet (per block) on MNIST, Fashion-MNIST and CIFAR-10. Emerges in the TPT, earlier than NC. When the law holds, test accuracy is higher and separation is robust to weight perturbation. |
| A. Rangamani, M. Lindegaard, T. Galanti, T. Poggio (2023). *Feature learning in deep classifiers through intermediate neural collapse.* ICML 2023 (PMLR 202:28729-28745) | https://proceedings.mlr.press/v202/rangamani23a.html | [V] | Within-class covariance falls relative to between-class covariance with depth. In the top layers, the class-mean subspace aligns with the top singular vectors of each layer's weights. |
| L. Parker, E. Onal, A. Stengel, J. Intrater (2023). *Neural collapse in the intermediate hidden layers of classification neural networks.* arXiv | https://arxiv.org/abs/2308.02760 | [V] | Some NC appears in most hidden layers and strengthens with depth. Within-class variance falls mostly in shallow layers; simple datasets use only the shallow layers. |
| I. Ben-Shaul, S. Dekel (2022). *Nearest class-center simplification through intermediate layers.* arXiv | https://arxiv.org/abs/2201.08924 | [V] | NCC mismatch inside the network; a loss that improves intermediate NC geometry improves generalization. |
| P. Wang, X. Li, C. Yaras, Z. Zhu, L. Balzano, W. Hu, Q. Qu. *Understanding deep representation learning via layerwise feature compression and discrimination.* JMLR (accepted, per arXiv) | https://arxiv.org/abs/2311.02960 | [V] | In deep linear networks, within-class features compress at a geometric rate and between-class features separate at a linear rate with depth; similar patterns in nonlinear networks. |
| W. Masarczyk, M. Ostaszewski, E. Imani, R. Pascanu, P. Miłoś, T. Trzciński (2023). *The tunnel effect: building data representations in deep neural networks.* NeurIPS 2023 | https://arxiv.org/abs/2305.19753 | [S] | In deep enough classifiers, the early layers build linearly separable representations and the later "tunnel" compresses them. The tunnel forms early in training and hurts OOD generalization and continual learning. |
| L. Hui, M. Belkin, P. Nakkiran (2022). *Limitations of neural collapse for understanding generalization in deep learning.* arXiv | https://arxiv.org/abs/2202.08384 | [V] | NC appears on the training set but **not on the test set**, so it is mainly an optimization phenomenon. Training longer can make last-layer features **transfer worse**. Also gives preliminary evidence of "cascading collapse" into earlier layers. |
| Y. Yang, J. Steinhardt, W. Hu (2023). *Are neurons actually collapsed? On the fine-grained structure in neural representations.* ICML 2023 | https://arxiv.org/abs/2306.17105 | [V] | The residual within-class variation after apparent collapse still encodes input structure. A network trained on 5 CIFAR-10 superclasses lets unsupervised clustering recover the 10 original labels at about 93%. |
| J. Xu, H. Liu (2023). *Quantifying the variability collapse of neural networks.* arXiv | https://arxiv.org/abs/2306.03440 | [V] | Variability Collapse Index (VCI): invariant to invertible linear maps and numerically stable (unlike raw NC1). It indicates the transferability of pretrained networks. |
| S. Kornblith, T. Chen, H. Lee, M. Norouzi (2021). *Why do better loss functions lead to less transferable features?* NeurIPS 2021 | https://arxiv.org/abs/2010.16402 | [V] | Higher class separation in the last layers raises source accuracy but lowers linear transfer. CKA shows that loss-function differences appear only in the last few layers. |

### 2.3 NC used downstream (OOD, transfer, TTA)

| Citation | URL | Tag | One-line content |
|---|---|---|---|
| M. Ben Ammar, N. Belkhir, S. Popescu, A. Manzanera, G. Franchi (2024). *NECO: NEural Collapse based Out-of-distribution detection.* ICLR 2024 | https://arxiv.org/abs/2310.06823 | [S] | Post-hoc OOD score: a sample's relative norm inside the ETF/principal subspace. Reported ahead of ViM on FPR95. |
| L. Liu, Y. Qin. *Detecting out-of-distribution through the lens of neural collapse.* CVPR 2025 (per arXiv) | https://arxiv.org/abs/2311.01479 | [V] | After centring, in-distribution features sit close to the classifier weight vectors and farther from the origin than OOD features. The detector combines weight proximity and feature norm. |
| J. Haas, W. Yolland, B.T. Rabus (2023). *Linking neural collapse and L2 normalization with improved out-of-distribution detection in deep neural networks.* TMLR | https://arxiv.org/abs/2209.08378 | [S] | L2-normalized features induce early NC and better OOD detection on the DDU benchmark. |
| M.Y. Harun, J. Gallardo, C. Kanan (2025). *Controlling neural collapse enhances out-of-distribution detection and transfer learning.* ICML 2025 | https://arxiv.org/abs/2502.10691 | [V] | **Stronger NC improves OOD detection but hurts generalization and transfer**, and weaker NC does the reverse. NC is controlled per layer (entropy regularization; a fixed ETF projector). |
| Z. Wang, Y. Luo, L. Zheng, Z. Huang, M. Baktashmotlagh (2023). *How far pre-trained models are from neural collapse on the target dataset informs their transferability.* ICCV 2023, pp. 5549-5558 | https://openaccess.thecvf.com/content/ICCV2023/html/Wang_How_Far_Pre-trained_Models_Are_from_Neural_Collapse_on_the_ICCV_2023_paper.html | [S] | NCTI: ranks pre-trained models by how close their target-set features are to NC (NC1, the simplex-encoded label geometry, NCC optimality); about 10x faster than the previous state of the art. |
| X. Chen, Z. Du, J. Huang, X. Jiang, L. Lu, J. Jiang, Z. Wang. *Neural collapse in test-time adaptation.* CVPR 2026 (per arXiv) | https://arxiv.org/abs/2512.10421 | [V] | Defines **sample-wise alignment collapse (NC3+)**. Feature/classifier-weight misalignment grows with shift severity and makes pseudo-labels unreliable. NCTTA blends geometric proximity with confidence; +14.52 pt over TENT on ImageNet-C. |
| K. Kang, A. Setlur, C. Tomlin, S. Levine. *Deep neural networks tend to extrapolate predictably.* ICLR 2024 | https://arxiv.org/abs/2310.00873 | [V] | As inputs move further OOD, predictions move toward the optimal constant solution and **feature norms shrink** (CIFAR10-C, ImageNet-R/S; CNNs and transformers). |

### 2.4 Intrinsic dimension

| Citation | URL | Tag | One-line content |
|---|---|---|---|
| E. Facco, M. d'Errico, A. Rodriguez, A. Laio (2017). *Estimating the intrinsic dimension of datasets by a minimal neighborhood information.* Scientific Reports 7:12140 | https://doi.org/10.1038/s41598-017-11873-y | [S] | The TwoNN estimator. Cited in atlas/invariants/dimension.py:5. |
| A. Ansuini, A. Laio, J.H. Macke, D. Zoccolan (2019). *Intrinsic dimension of data representations in deep neural networks.* NeurIPS 2019 | https://arxiv.org/abs/1905.12784 | [V] | **Hunchback** profile: ID rises, then falls in the final layers. **Last-hidden-layer ID predicts test accuracy.** Linear (PCA) estimates cannot find these results, and the manifolds are curved. The profile is **not found in untrained networks or with random labels**. Cited in dimension.py:7-9. |
| S. Recanatesi, M. Farrell, M. Advani, T. Moore, G. Lajoie, E. Shea-Brown (2019). *Dimensionality compression and expansion in deep neural networks.* arXiv | https://arxiv.org/abs/1906.00443 | [V] | Early layers expand dimension (feature generation) and later layers compress it (task-relevant selection). SGD noise regularizes dimension, and compression goes with better generalization. |
| P. Pope, C. Zhu, A. Abdelkader, M. Goldblum, T. Goldstein (2021). *The intrinsic dimension of images and its impact on learning.* ICLR 2021 | https://arxiv.org/abs/2104.08894 | [S] | Natural-image datasets have low ID; lower-ID datasets are easier to learn and generalize better. |
| L. Valeriani, D. Doimo, F. Cuturello, A. Laio, A. Ansuini, A. Cazzaniga (2023). *The geometry of hidden representations of large transformer models.* NeurIPS 2023 | https://arxiv.org/abs/2302.00294 | [V] | ID expands, then contracts in intermediate layers. **Layers at a relative ID minimum are best for downstream tasks**, which gives unsupervised layer selection. |
| X. Ma, B. Li, Y. Wang, S.M. Erfani, S. Wijewickrema, G. Schoenebeck, D. Song, M.E. Houle, J. Bailey (2018). *Characterizing adversarial subspaces using local intrinsic dimensionality.* ICLR 2018 | https://arxiv.org/abs/1801.02613 | [V] | **Per-sample LID** of activations detects adversarial examples (five attacks, three datasets). |

### 2.5 Representation similarity and functional agreement

| Citation | URL | Tag | One-line content |
|---|---|---|---|
| S. Kornblith, M. Norouzi, H. Lee, G. Hinton (2019). *Similarity of neural network representations revisited.* ICML 2019 | https://arxiv.org/abs/1905.00414 | [V] | Introduces CKA; statistics invariant to invertible linear maps fail when the dimension exceeds the number of points; **CKA identifies corresponding layers across networks trained from different initializations**. Cross-depth layer correspondence is in the full paper, not verified here [U]. |
| T. Nguyen, M. Raghu, S. Kornblith (2021). *Do wide and deep networks learn the same things?* ICLR 2021 | https://arxiv.org/abs/2010.15327 | [V] | Wide or deep networks develop a **block structure** in which layers carry the **dominant principal component**. Representations outside the block are similar across widths and depths; the block is model-specific. Wide and deep models make different per-class errors. |
| M. Raghu, J. Gilmer, J. Yosinski, J. Sohl-Dickstein (2017). *SVCCA: singular vector canonical correlation analysis for deep learning dynamics and interpretability.* NeurIPS 2017 | https://arxiv.org/abs/1706.05806 | [S] | SVCCA; used to measure layer dimensionality, learning dynamics and **where class-specific information forms**. |
| A.S. Morcos, M. Raghu, S. Bengio (2018). *Insights on representational similarity in neural networks with canonical correlation.* NeurIPS 2018 | https://arxiv.org/abs/1806.05759 | [S] | Projection-weighted CCA. Networks that generalize converge to more similar representations than networks that memorize; wider networks converge more. |
| Y. Li, J. Yosinski, J. Clune, H. Lipson, J. Hopcroft (2016). *Convergent learning: do different neural networks learn the same representations?* ICLR 2016 | https://arxiv.org/abs/1511.07543 | [S] | Unit-level matching across seeds. |
| L. Moschella, V. Maiorca, M. Fumero, A. Norelli, F. Locatello, E. Rodolà (2023). *Relative representations enable zero-shot latent space communication.* ICLR 2023 | https://arxiv.org/abs/2209.15430 | [S] | Represents each sample by its similarity to anchors, which gives zero-shot model stitching. Cited in atlas/compare.py:18. |
| F. Ding, J.-S. Denain, J. Steinhardt (2021). *Grounding representation similarity with statistical testing.* NeurIPS 2021 | https://arxiv.org/abs/2108.01661 | [V] | Similarity measures disagree even on whether different seeds learn similar representations. **CKA misses the removal of low-variance principal components that change function**; orthogonal Procrustes does better. |
| M. Davari, S. Horoi, A. Natik, G. Lajoie, G. Wolf, E. Belilovsky (2023). *Reliability of CKA as a similarity measure in deep learning.* ICLR 2023 | https://arxiv.org/abs/2210.16156 | [S] | CKA is sensitive to outliers and can be manipulated without changing function. |
| M. Klabunde, T. Schumacher, M. Strohmaier, F. Lemmerich (2025). *Similarity of neural network models: a survey of functional and representational measures.* ACM Computing Surveys 57(9):242 | https://arxiv.org/abs/2305.06329 | [S] | Survey of representational and functional similarity measures. |
| M. Huh, B. Cheung, T. Wang, P. Isola (2024). *The platonic representation hypothesis.* ICML 2024 | https://arxiv.org/abs/2405.07987 | [V] | Representations of different models and modalities converge in their distance structure as models scale. |
| R. Geirhos, K. Meding, F.A. Wichmann (2020). *Beyond accuracy: quantifying trial-by-trial behaviour of CNNs and humans by measuring error consistency.* NeurIPS 2020 | https://arxiv.org/abs/2006.16736 | [V] | Error consistency, a kappa over the chance overlap of errors. **CNNs are highly consistent with each other regardless of architecture.** Cited in atlas/compare.py:205. |
| Y. Jiang, V. Nagarajan, C. Baek, J.Z. Kolter (2022). *Assessing generalization of SGD via disagreement.* ICLR 2022 | https://arxiv.org/abs/2106.13799 | [V] | **The disagreement rate of two seeds on unlabeled test data estimates test error.** Explained by ensemble calibration. |
| C. Baek, Y. Jiang, A. Raghunathan, J.Z. Kolter (2022). *Agreement-on-the-line: predicting the performance of neural networks under distribution shift.* NeurIPS 2022 | https://arxiv.org/abs/2206.13089 | [V] | Pairwise OOD agreement is linear in ID agreement, which gives OOD accuracy from unlabeled data. The effect holds for neural networks only. |
| R. Xie, H. Wei, L. Feng, Y. Cao, B. An (2023). *On the importance of feature separability in predicting out-of-distribution error.* NeurIPS 2023 | https://arxiv.org/abs/2303.15488 | [V] | **Dispersion score**: inter-class feature dispersion on the shifted set predicts OOD accuracy without test labels; intra-class compactness does not. Target features are grouped into classes by the model's predicted label (pseudo-labels). |
| V.V. Ramasesh, E. Dyer, M. Raghu. *Anatomy of catastrophic forgetting: hidden representations and task semantics.* ICLR 2021 | https://arxiv.org/abs/2007.07400 | [V] | Representational similarity localizes forgetting to **deeper layers**; mitigations work by stabilizing them. |

Cited in Atlas code (landmarks.py:9-11): M. Radovanović, A. Nanopoulos, M. Ivanović (2010). *Hubs in space: popular
nearest neighbors in high-dimensional data.* JMLR 11:2487-2531. https://jmlr.org/papers/v11/radovanovic10a.html [S].
Hubness is the skewness of the k-occurrence distribution.

---

## 3. How it is used

Nothing in this family is **standard production practice** for per-input trust or drift gating. In deployed systems
those roles go to head confidence and input/output statistics (the confidence-margin and shift-monitoring topics). What
*is* standard is the **analysis toolkit**: NC1-NC4 are the fixed metrics in NC papers, linear CKA is the default
similarity measure, and NCC and linear probes are routine. Everything below is research practice unless marked.

| Usage pattern | Methods | Status | Typical metrics | Known failure modes |
|---|---|---|---|---|
| Training diagnostics and last-layer theory | NC1-NC4 tracking (Papyan 2020; Han 2022; Zhu 2021; review: Kothapalli 2023) | Research convention | NC1 = Tr(Σ_W Σ_B⁺)/K; ETF angle and norm deviation; NC3 weight/mean alignment; NCC mismatch | Train-set only; NC does not transfer to test (Hui 2022). Imbalance causes minority collapse (Fang 2021). No ETF when K > d+1 (Jiang 2024). Raw NC1 is not invariant to linear maps (Xu & Liu 2023). |
| Layer-wise separation profile; depth and architecture design | Law of equi-separation (He & Su 2023); intermediate NC (Rangamani 2023; Parker 2023; Ben-Shaul & Dekel 2022); compression/discrimination law (Wang et al.); tunnel effect (Masarczyk 2023) | Research | Pearson(log D, layer), decay ratio ρ; per-layer NCC accuracy | Needs the TPT; measured on training data; DenseNet needs per-block granularity; the late-layer tunnel hurts OOD and transfer |
| Transferability estimation and model selection | NCTI (Wang et al. ICCV 2023); VCI (Xu & Liu 2023); NC and few-shot transfer (Galanti 2022) | Research, benchmarked across source models | Rank correlation between the score and fine-tuned accuracy | Strong collapse lowers transfer (Hui 2022; Kornblith 2021; Harun 2025) |
| OOD detection | NECO (ETF-subspace norm ratio); Liu & Qin (weight proximity plus norm); Haas (L2 normalization induces NC); Harun (fixed ETF projector) | Research; post-hoc detectors | AUROC, FPR95 | Need a well-collapsed model; the degree of NC trades OOD detection against transfer (Harun 2025) |
| Test-time adaptation | NCTTA: sample-wise feature/classifier alignment (NC3+) weights pseudo-labels (Chen et al., CVPR 2026) | Research | Accuracy under shift (ImageNet-C) | Alignment degrades with shift severity, which is the signal it exploits |
| Label-free accuracy estimation under shift | Dispersion score (Xie 2023); disagreement between seeds (Jiang 2022); agreement-on-the-line (Baek 2022) | Research, active benchmark area | Absolute error of the estimated accuracy; R² or rank correlation across shifts | Dispersion: compactness does not predict. Agreement needs two or more models, and agreement-on-the-line holds only for neural networks. |
| ID as a generalization proxy and for layer selection | Last-layer ID predicts accuracy (Ansuini 2019); layer at an ID minimum (Valeriani 2023); dataset ID and sample complexity (Pope 2021); expansion/compression (Recanatesi 2019) | Research | Correlation of ID with test accuracy | Estimator and scale dependence. In Atlas the penult TwoNN is mostly spectral (AH-8), and random-init ResNets also peak (CD-1). |
| Per-sample adversarial or anomaly detection | LID (Ma 2018) | Research | AUROC | Robustness to adaptive attacks not checked here [U] |
| Localizing what changed between two models or checkpoints | CKA (Kornblith 2019); SVCCA/PWCCA (Raghu 2017; Morcos 2018); block structure (Nguyen 2021); loss effects only in the last layers (Kornblith 2021); forgetting in deep layers (Ramasesh) | Standard **analysis** practice; not a deployed monitor | CKA heatmaps (0-1); Procrustes distance | CKA is dominated by the top principal components (Nguyen 2021) and misses functionally relevant low-variance directions (Ding 2021). It can be manipulated and is outlier-sensitive (Davari 2023). It needs input-resampling nulls. |
| Cross-model communication and merging | Relative representations (Moschella 2023); convergent learning (Li 2016); platonic convergence (Huh 2024) | Research | Stitching accuracy; anchor-space agreement | Depends on the anchor choice; the class-centre anchors Atlas uses bake in class structure |
| Functional similarity | Error consistency (Geirhos 2020); disagreement (Jiang 2022) | Research, common in robustness studies | Cohen's kappa; disagreement rate | CNNs are consistent regardless of architecture, so high values carry little architecture information |

---

## 4. Known results relevant to Atlas

1. **Layer-wise nc1 decay is the law of equi-separation.** Stage 2 F2 found Pearson(log10 nc1, block index) = −0.9726
   over 28 resnet56 taps (results/atlas_v1_resnet56_s0hub/SESSION.md:317). He & Su (2023) report log D linear in
   layer, with D = Tr(S_w S_b⁺) on **training** data. For balanced classes Atlas's nc1 is exactly D/K, also on the train
   reference (landmarks.py:59-72). He & Su also tie the law to higher test accuracy and robustness to weight
   perturbation; Atlas never tested either.
2. **NC is a training-set phenomenon** (Hui, Belkin & Nakkiran 2022). This is the known mechanism behind two Atlas
   findings:
   - DO-3/AH-1: a train-reference density threshold over-alarms on clean test data, at 1.8-2.3× the nominal rate in
     resnet20 and 3.3-3.7× in resnet56 (recomputed; `density-ood.md` §5);
   - the gap between the collapsed train reference and less-collapsed test features.
   Hui et al. also show that longer training can make last-layer features transfer worse.
3. **Collapse keeps deepening through the terminal phase** (Papyan 2020; He & Su 2023). AH-7 found no 50 → 70-epoch
   trend in stage-3 compression but a step up to 200 epochs (results/anomaly_h1/SESSION.md:237-254). This is consistent
   with TPT-driven collapse. Whether the step coincides with the end of the learning-rate schedule was not checked here.
4. **NC4 means the NCC and the head agree once collapse is complete.** Three Atlas findings follow:
   - nearest-centre accuracy equals head accuracy (CD-11: 0.9446/0.9398 vs 0.9438/0.9404;
     results/atlas_v1_resnet56_s1/SESSION.md:307-318);
   - penult agreement is 0.997 at resnet56 (results/anomaly_h1/SESSION.md:185);
   - plain distance catches up with the top-2 margin at full collapse (CM-2).
   The disagreement that remains before full collapse, "NCC mismatch" (Ben-Shaul & Dekel 2022), is the natural
   per-sample signal (CM-11).
5. **NC2 equidistance removes class-distance ranking information.** At resnet56 the centre-distance CV falls to
   0.037-0.043, and the seed-pair distance-only adjacency D1 drops to 0.692 (row 7 COLLAPSE-ARTIFACT,
   results/atlas_v1_resnet56_s1/SESSION.md:425-496). This is what NC2 predicts. Yang, Steinhardt & Hu (2023) show that
   the fine structure survives in the **within-class residual**, so the residual, not the centre geometry, is where
   cross-seed class structure should be looked for at full collapse.
6. **Linear CKA is dominated by the top principal components** (Nguyen 2021; Ding 2021). At a collapsed penult the top
   K−1 components are the class-mean simplex. So the Atlas fit cka_test = 0.9405 − 0.3154·nc1 (rmse 0.0031;
   results/atlas_v1_resnet56_s1/SESSION.md:675-715) is an expected consequence, not a depth effect. The
   within-class-residual CKA that Atlas left unrun (:701) is the test this literature recommends. Ding et al. prefer
   Procrustes for functional sensitivity.
7. **Seed-stable similarity is published.** Kornblith 2019 shows CKA finds corresponding layers across initializations;
   Li 2016 shows convergent learning; Morcos 2018 shows that generalizing networks converge more. Moschella 2023 shows
   relative representations transfer. Geirhos 2020 shows CNNs are highly error-consistent. Atlas's relrep at
   4.67-4.72x chance on CIFAR-100, cka_test 0.883-0.892, and error consistency 0.52-0.57 are in line with all of this.
8. **The head's own agreement is the literature's label-free accuracy tool.** Two-seed disagreement estimates test error
   (Jiang 2022), and agreement-on-the-line extends it to OOD (Baek 2022). Atlas computed geometric agreement (relrep)
   but never the head-disagreement baseline on the same CIFAR-100 inputs. The head version is the one with a known use.
9. **Hunchback ID and last-layer compression** (Ansuini 2019; Recanatesi 2019). Atlas rows 1 and 4 rediscover them; the
   code cites Ansuini (dimension.py:7-9). **Two Atlas results disagree with Ansuini:**
   - the random-init ResNet also peaks at layer3.1 and drops only 0.020 (r20) / 0.001 (r56) (ATLAS_STATUS.md:10),
     whereas Ansuini report no hunchback in untrained networks;
   - TwoNN follows its Gaussian spectral twin across taps (Pearson ≥ 0.984, random nulls 0.997-0.998), whereas Ansuini
     say linear estimates cannot reproduce the ID results. Ansuini ran the same control at the VGG-16 last hidden layer
     and found the twin's ID two orders of magnitude above the data's (their Fig. 5B).
   These reconcile if the Atlas peak location is set by the architecture (16/32/64 channels, GAP) and only the
   last-block drop is learned. At the peak TwoNN is 26-28% **below** its spectral twin (ρ 0.721/0.743), which fits
   Ansuini's "curved manifold" reading there.
10. **Deeper, over-parameterized classifiers compress more in their late layers.** This comes from the tunnel effect
    (Masarczyk 2023), equi-separation (more layers mean more factors of ρ), and intermediate NC. It matches D-COLL: nc1
    is lower at matched accuracy, 0.129-0.130 vs the edge 0.154. No work found predicts D-ID, a **higher** penult TwoNN
    in deeper nets at matched accuracy. In the full-recipe leg that excess is non-spectral (ID_gauss 10.471/10.508 ≤
    10.698; results/anomaly_h1/SESSION.md:264-266).
11. **K > d rules out an ETF** (Jiang et al. 2024, generalized NC). 1000 ImageNet classes in 768-d cannot form a simplex
    ETF. The ViT "shallow valleys" (sep_ratio_ref 0.747-0.930, results/margin_b1_vitb16/SESSION.md:426) are therefore
    partly forced by dimension and are not comparable with CIFAR's 3.0-5.3.
12. **Collapse level trades OOD detection against transfer and fine structure** (Harun 2025; Hui 2022; Kornblith 2021;
    Yang 2023). This is the closest published relative of Atlas's "collapse is a hidden coordinate" synthesis (CD-9).
    The degree of NC changes what a feature-space detector can see.
13. **NC-based OOD detectors use feature norm and the class-span decomposition.**
    - Liu & Qin find in-distribution features farther from the origin than OOD features, and Kang et al. find OOD
      feature norms shrink.
    - This matches the CIFAR direction of CM-8: errors have lower penult energy, oriented AUROC 0.750-0.795. It does
      **not** predict the ImageNet sign flip (errors have higher energy, 0.23-0.30).
    - NECO's in-subspace norm ratio is the per-sample relative of Atlas's class_sub_frac (PC-6) and of T_par/T_perp
      (ST-3).
14. **Sample-wise feature/classifier misalignment (NC3+) grows with shift** and is used to weight TTA pseudo-labels
    (Chen et al., CVPR 2026). Atlas's head_center_cos (0.78/0.66/0.56 on vitb16/deitb/resnet50) is the checkpoint-level
    NC3 statistic. `nearest_center_agrees_with_model` is its per-sample, argmax-level analogue. Neither was scored as a
    signal in Atlas.
15. **Label smoothing still produces NC** (Zhou et al. 2022). ResNet50's poor softmax ranking is therefore a calibration
    effect, not an absence of collapse. That result belongs to the confidence-margin topic.

---

## 5. Relation to Atlas items

Legend: **rediscovered** = already published; **extends** = adds a new condition, control or number to a published
result; **contradicts** = at odds with a published result; **untested here** = the literature has a comparison or use
that Atlas did not run.

| Item | Relation | Prior art | Atlas evidence | Beyond the head? |
|---|---|---|---|---|
| CD-1 ID peak and last-block drop | Rediscovered (hunchback, last-layer drop). **Partly contradicts** Ansuini on untrained nets: the Atlas random init also peaks. | Ansuini 2019; Recanatesi 2019; Facco 2017 | ATLAS_STATUS.md:10; dimension.py:5-9; drop 0.488-0.503 (r20), 0.471/0.463 (r56) | N/A (checkpoint level) |
| CD-2 TwoNN vs spectrum | **Partly contradicts** Ansuini's "linear estimates insufficient". The Gaussian-twin control is Ansuini's own (Fig. 5B, VGG-16 last hidden layer, where the twin's ID was two orders of magnitude higher); in Atlas the twin reproduces TwoNN except at the peak. **Extends** the control to every tap | Ansuini 2019; Facco 2017 | results/anomaly_h1/SESSION.md:256-270; docs/plans/ANOMALY_H1.md:570-572 | N/A |
| CD-3 deeper net, higher penult ID at matched accuracy | Not found in the literature. In tension with the compression/tunnel results, which predict lower late-layer dimension in deeper nets. The L2 excess is non-spectral, and L1 has a fit confound. | Ansuini 2019; Masarczyk 2023; He & Su 2023 | ATLAS_STATUS.md:21; results/anomaly_h1/SESSION.md:266; results/atlas_v1_resnet56_s1/SESSION.md:259-305; qualifier conflict docs/plans/ANOMALY_H1.md:561-564 vs :822-825 | Beyond head accuracy: yes. Calibration: untested. |
| CD-4 collapse at matched accuracy (nc1); training length | nc1 lower in deeper nets: **rediscovered** (equi-separation, tunnel, intermediate NC). F2 log-linear decay: **rediscovered** (He & Su). AH-7 refuted: consistent with TPT. | Papyan 2020; He & Su 2023; Rangamani 2023; Parker 2023; Masarczyk 2023; Hui 2022 | results/atlas_v1_resnet56_s1/SESSION.md:259-305 (nc1 CONFIRMED, 18x twin noise); results/atlas_v1_resnet56_s0hub/SESSION.md:317; results/anomaly_h1/SESSION.md:237-254 | Beyond accuracy: yes. **Untested** against head calibration or the confidence distribution, which also changes in the TPT. |
| CD-5 sep_ratio; ImageNet valleys | Rediscovered (an NC1/NC2-type separability ratio). ImageNet levels are explained by generalized NC (K > d). | Papyan 2020; Kornblith 2021; Jiang 2024 | ATLAS_STATUS.md:11; results/margin_b1_vitb16/SESSION.md:195-210, :426 | N/A |
| CD-6 class adjacency / merge order | Seed stability partly known (convergent learning, CKA). The d56 loss follows from NC2. **Untested here**: a head-side adjacency (test confusion matrix; classifier-weight cosines, which NC3 predicts equal the centre cosines). | Li 2016; Kornblith 2019; Papyan 2020; Yang 2023 | ATLAS_STATUS.md:17; results/atlas_v1_resnet56_s1/SESSION.md:425-496; adjacency.py:23-33; scripts/d1_distance_only.js | **Untested** |
| CD-7 cross-seed similarity (CKA, relrep, error consistency) | Rediscovered. The CKA-collapse relation is expected (top-PC dominance). **Untested here**: residual CKA, Procrustes, head disagreement on the same inputs. | Kornblith 2019; Nguyen 2021; Ding 2021; Davari 2023; Moschella 2023; Geirhos 2020; Jiang 2022; Baek 2022 | results/atlas_v1_resnet56_s1/SESSION.md:675-715; ATLAS_STATUS.md:18; compare.py:15-20, :198-206 | **Untested**; the head-based version (disagreement) is the one with a published use |
| CD-8 shape transfers, level does not | Partly known: representations outside the block structure are similar across width and depth (Nguyen 2021); equi-separation holds across architectures | Nguyen 2021; Kornblith 2019 [U detail]; He & Su 2023 | results/atlas_v1_resnet56_s1/SESSION.md:478-512 | N/A |
| CD-9 collapse as a single hidden coordinate | **Extends**: NC theory predicts each component, and Harun 2025 shows NC degree trades OOD detection against transfer. The "one offline calibration number" framing was not found. | Papyan 2020; Hui 2022; Harun 2025; Nguyen 2021; Ding 2021 | Evaluation synthesis (docs/reviews/EVAL_2026-09-23.md §2.2); AH-3 and AH-7 refuted, AH-4 not evaluable (results/anomaly_h1/SESSION.md:175-191, :237-254) | N/A |
| CD-10 per-layer PR, dim95, hubness | Standard measurements (rediscovered); not seed-stable in Atlas | Ansuini 2019 (PCA-based linear dimension next to TwoNN); Parker 2023 (NC in intermediate layers); Radovanović 2010 (hubness) | results/atlas_v1_resnet20_s3/SESSION.md:283-299 | N/A |
| CD-11 geometric class readout = head accuracy | Rediscovered (NC4) | Papyan 2020; Ben-Shaul & Dekel 2022; Galanti 2022 | results/atlas_v1_resnet56_s1/SESSION.md:307-318; results/atlas_v0_resnet20_cifar10/SESSION.md:131-132 | **No** (by NC4) |
| CM-2 margin ≈ distance at full collapse | The mechanism is known (NC2+NC4); the AUROC observation extends it | Papyan 2020 | ATLAS_STATUS.md:19; results/atlas_v1_resnet56_s1/SESSION.md:518-549 | Partial vs maxprob (confidence-margin topic) |
| CM-5 margin residual beyond the logit gap | Not found. NC3 predicts margin = logit gap once centres ∝ head weights, so the residual should shrink as head_center_cos → 1. That was **not** observed: deitb (0.66) leads most. | Papyan 2020 (NC3); Chen et al. 2026 (geometry + confidence blend) | results/margin_b1_vitb16/SESSION.md:337-345, :497-507 | Candidate yes (confidence-margin topic) |
| CM-6 collapse picks the per-sample score | Mechanism partly known (NC4); refuted as registered | Papyan 2020; Harun 2025 | results/anomaly_h1/SESSION.md:175-191 | N/A |
| CM-8 energy sign | CIFAR direction consistent with the NC-based OOD norm findings; ImageNet flip unexplained | Liu & Qin 2025; Kang et al. 2023/24; Haas 2023 | results/margin_b1_vitb16/SESSION.md:431 | No |
| CM-11 NCC vs argmax disagreement | Partly known (NCC mismatch; NC3+ misalignment for TTA). **Untested here** as a detector. | Ben-Shaul & Dekel 2022; Chen et al. 2026 | landmarks.py:54; results/anomaly_h1/SESSION.md:185; results/margin_b1_vitb16/SESSION.md:503-506 | **Untested**; a direct geometry-vs-head test |
| DO-3 log-nc1 false-alarm law | Mechanism known (train/test collapse gap); the quantitative law was not found | Hui 2022; Papyan 2020 | results/anomaly_h1/SESSION.md:135-155 | N/A |
| PC-3 commit layer / reorganization | Rediscovered (intermediate NC; SVCCA locates class information; tunnel) | Rangamani 2023; Parker 2023; Ben-Shaul & Dekel 2022; Raghu 2017; Masarczyk 2023 | ATLAS_STATUS.md:13; results/atlas_v1_resnet56_s0hub/SESSION.md:260-263 | N/A |
| PC-6 class_sub_frac | Partly known: equals the between-class variance share; NECO uses a per-sample version for OOD | Ammar et al. 2024 | results/anomaly_h1/SESSION.md:205 | N/A |
| ST-3 T_perp / T_par | Partly known: class-span decomposition as in NECO; the batch-mean drift version was not found | Ammar et al. 2024; Liu & Qin 2025 | results/anomaly_h1/SESSION.md:272-291 | T_par vs the head's label-shift monitor: untested |
| ST-6 TTA deformation monitor | Partly known: CKA localizes change and forgetting (Ramasesh; Kornblith 2021); NC under TTA (Chen et al.). An online early-warning version was not found. **Untested here.** | Ramasesh 2020/21; Kornblith 2021; Chen et al. 2026 | scripts/tta_deform.py:1-20; no results/tta_* | **Untested** (the bar is pred_entropy) |
| LH-1 label-free harm grade | The task is published; this topic supplies baselines (dispersion, disagreement, agreement-on-the-line). **Untested here.** | Xie 2023; Jiang 2022; Baek 2022 | results/anomaly_h1/SESSION.md:157-173 | **Untested** |

Evaluation correction applied (`docs/reviews/EVAL_2026-09-23.md` §6, correction 16): rows 1, 2 and 8 were built as
known-answer measurements, and the code cites the literature (dimension.py:5-9 Facco/Ansuini; landmarks.py:6-8 Papyan;
compare.py:18 Moschella; compare.py:205 Geirhos). What is missing is citation in ATLAS_STATUS.md, the plans and the
SESSION files. He & Su (the F2 law), Kornblith/CKA, Hui and the NC applications are cited nowhere in the tracked repo.

---

## 6. Baselines Atlas should include next time

Ordered by cost. Items 1-5 are recomputable from existing dumps (`acts/<layer>/<split>.npy`, `preds/<split>.npz` with
argmax and maxprob; atlas/extract_acts.py:181-188) if the pod volume is kept. The fc weights are needed for logits.

1. **Head-side twin for every class-geometry readout.**
   - CD-6: Spearman between seeds of (a) the test confusion matrix (off-diagonal, symmetrized) and (b) classifier-weight
     cosines, next to D1.
   - CD-7: two-seed **head argmax disagreement** on test and on CIFAR-100 next to relrep argmax agreement (Jiang 2022).
     Also the logit-vector CKA next to the penult CKA.
2. **Within-class-residual CKA and class-mean-only CKA**, plus orthogonal Procrustes (Ding 2021), with an input-bootstrap
   CI and a permutation null. This decides whether the CD-7 excess is anything beyond NC2.
3. **CM-11 as a detector.** Score `nearest_center != argmax` (and the continuous gap between the NCC distance and the
   head logit) as a misclassification detector on the ImageNet dumps, where 11-20% of samples disagree. Compare it
   against the logit gap and maxprob in a joint model. This is the NC4/NC3+ analogue of NCTTA's alignment signal.
4. **Test-side NC.** Report nc1, etf_deviation and NCC accuracy on test[:5000] next to the train reference (Hui 2022).
   Test the train-test nc1 gap, rather than train nc1, as the DO-3 predictor.
5. **The complete NC suite per checkpoint.**
   - NC1 plus an invariant variant (VCI, Xu & Liu 2023).
   - NC2 (norm CV and angle deviation).
   - NC3 (head_center_cos, now also on CIFAR).
   - NC4 (agreement).
   - He & Su's decay ratio ρ, and the goodness of the log-linear fit.
6. **ID controls.** Random-label and random-init nulls (Ansuini's two negative controls); PR, dim95 and ID_gauss next to
   every TwoNN; a multi-scale check (TwoNN on subsamples of different size) for scale dependence.
7. **Label-free accuracy/harm baselines for LH-1 and LH-2** from this topic: the dispersion score (Xie 2023) and
   agreement-on-the-line (Baek 2022; needs at least two seeds, which Atlas has). They sit beside average confidence and
   ATC from the harm topic.
8. **NC-based OOD detectors** for DO-4: NECO and Liu & Qin's weight-proximity-plus-norm, beside MSP, energy, Mahalanobis,
   ViM and normalized kNN.
9. **For the TTA ladder (ST-6):** layer-wise CKA and residual CKA before and after each TENT step, the NC3+ alignment
   per sample (Chen et al.), head pred_entropy, and head disagreement with the frozen source model. Pre-register which
   fires first.
10. **For model or checkpoint selection, if the controller ever chooses a backbone:** NCTI (Wang et al. 2023) and VCI.

---

## 7. Implications for the controller

1. **Treat this family as calibration and change detection, not as per-input sensing.** Collapse, dimension and
   similarity are one number per checkpoint, layer or pair. Their controller role is:
   - setting thresholds (the density false-alarm rate follows log nc1, DO-3);
   - choosing which per-sample score to trust (the NC4 logic behind CM-2 and CM-6);
   - detecting that the backbone itself changed (after fine-tuning, retraining or TTA: ST-6).
   Re-measure after any weight change. Measure on held-out data as well as the train reference, because NC is a
   training-set phenomenon.
2. **At full collapse, the geometry at the head input largely duplicates the head** (NC3/NC4):
   - distance ≈ margin;
   - NCC = argmax (0.997 agreement);
   - nearest-centre accuracy = head accuracy.
   Any beyond-head per-sample information must come from:
   - (a) the **within-class residual** (Yang et al. 2023);
   - (b) **pre-collapse taps**, where the NCC and the head still disagree (0.84-0.86 agreement at layer3.5);
   - (c) models that are **not** fully collapsed, such as the ImageNet ViTs, where CM-5 lives.
   The controller should expect geometry to add most on under-collapsed or K > d backbones and least on fully fit
   small-K classifiers.
3. **The most useful per-sample idea in this literature is NC3/NC4 disagreement.** "The geometry and the head disagree
   about this input" is used by NCTTA to decide which pseudo-labels to trust during adaptation, which is exactly the
   adapt-versus-hold question. In Atlas it exists as a field (CM-11) but was never scored. It is the cheapest direct
   test of the primary question and a natural "hold, output distrusted" trigger.
4. **The degree of collapse is a trade-off knob, not a quality score.** Stronger collapse helps OOD detection and hurts
   transfer and fine structure (Harun 2025; Hui 2022; Kornblith 2021). A controller that adapts the backbone (TTA)
   changes the collapse level and therefore the behaviour of every geometric detector calibrated on it. The ST-6
   deformation monitor must track nc1 and residual structure, not only CKA, and must be compared with head entropy and
   disagreement.
5. **Drop TwoNN as a controller feature** (AH-8; consistent with its literature role as a generalization proxy). Keep ID
   only as an instrument check (CD-1) and possibly for unsupervised layer selection (Valeriani 2023), should the
   controller need to pick a tap on a new backbone.
6. **For cross-model or cross-seed questions, prefer the head's disagreement.** Two-seed disagreement and
   agreement-on-the-line are published label-free accuracy estimators. Geometric agreement (relrep, CKA) has not been
   shown to add to them. Until item 1 of section 6 is run, the controller should use the head version.
7. **Scope.** Every Atlas result in this family comes from CIFAR-10 ResNet-20/56 trained with one recipe, plus two
   DeiT-lineage ViTs and one ResNet50 on a degraded ImageNet mirror. The literature shows that NC geometry changes
   qualitatively when K > d, under imbalance and under label smoothing. None of the CIFAR levels (nc1, sep_ratio, CKA)
   should be carried to another backbone or dataset without re-measuring.

---

## 8. Item classifications (this topic's reading)

The merged, conflict-resolved classification is in `docs/knowledge/README.md`. This table keeps this topic's reading
and citations.

| item | known? | key citations | how prior work uses it |
|---|---|---|---|
| CD-1 | known-in-research | Ansuini, Laio, Macke & Zoccolan 2019, NeurIPS (arXiv 1905.12784) [V]; Recanatesi et al. 2019 (arXiv 1906.00443) [V]; Facco et al. 2017, Sci Rep (doi 10.1038/s41598-017-11873-y) [S]; Valeriani et al. 2023 (arXiv 2302.00294) [V] | The hunchback ID profile is used as a generalization proxy (last-layer ID predicts test accuracy) and for unsupervised layer selection (the layer at an ID minimum). Atlas rediscovers the shape and cites Ansuini in dimension.py:7-9. It partly contradicts Ansuini: the Atlas random-init ResNet also peaks at layer3.1, whereas Ansuini found no hunchback in untrained nets. So only the last-block drop is learned. No controller use. |
| CD-2 | partly-known | Ansuini et al. 2019 (arXiv 1905.12784) [V]: linear PCA estimates cannot reproduce the ID results, and the manifolds are curved; Facco et al. 2017 [S] | Published work contrasts TwoNN with PCA thresholds, and Ansuini et al. also ran TwoNN on a Gaussian with the same second-order moments (Fig. 5B): at the VGG-16 last hidden layer the twin's ID was two orders of magnitude above the data's. Atlas applies the same control (ID_gauss) at every tap and finds the opposite at the penult: TwoNN is spectral (ρ 0.99-1.00), and 26-28% below the twin at the ID peak. Use: drop TwoNN as a controller feature; read penult "ID" as effective spectral dimension. |
| CD-3 | not-found-in-prior-art | No direct precedent found. Related: Ansuini 2019 (last-layer ID vs accuracy) [V]; Masarczyk et al. 2023, tunnel effect, NeurIPS (arXiv 2305.19753) [S], which predicts stronger late compression in deeper over-capacity nets; He & Su 2023, PNAS [V] | No published usage. The narrow claim (resnet56 penult TwoNN above resnet20 at matched accuracy) is in tension with the compression/tunnel literature. The full-recipe excess is non-spectral, and the matched leg has a fit confound. No controller use. |
| CD-4 | known-in-research | Papyan, Han & Donoho 2020, PNAS (arXiv 2008.08186) [V]; He & Su 2023, "A law of data separation", PNAS (arXiv 2210.17020) [V]; Rangamani et al. 2023, ICML [V]; Parker et al. 2023 (arXiv 2308.02760) [V]; Hui, Belkin & Nakkiran 2022 (arXiv 2202.08384) [V]; Masarczyk et al. 2023 [S]; Xu & Liu 2023 VCI (arXiv 2306.03440) [V]; Wang et al. 2023 NCTI, ICCV [S] | NC1 and layer-wise separation are used as training diagnostics, as transferability and model-selection scores (NCTI, VCI), and in OOD-detector design. Atlas's F2 log-linear nc1 decay is He & Su's law of equi-separation (same Tr(Sw Sb+) on training data). The AH-7 refutation is consistent with TPT dynamics. New Atlas use: nc1 predicts the train-reference density false-alarm rate (DO-3). Comparison with head calibration is untested. |
| CD-5 | known-in-research | Papyan et al. 2020 [V]; Kornblith, Chen, Lee & Norouzi 2021, NeurIPS (arXiv 2010.16402) [V]: class separation raises accuracy and lowers transfer; Jiang et al. 2024, Generalized neural collapse, ICML (arXiv 2310.05351) [S]: no simplex ETF when K > d | Class-separation ratios serve as NC1/NC2 proxies and as predictors of transfer and OOD accuracy (the dispersion score uses inter-class dispersion). ImageNet's low sep_ratio is partly forced by K = 1000 > d = 768 (generalized NC). No controller use beyond context for DO-3 and CM-6. |
| CD-6 | partly-known | Li et al. 2016, Convergent learning, ICLR (arXiv 1511.07543) [S]; Kornblith et al. 2019, ICML [V]; Papyan et al. 2020 (NC2 equidistance) [V]; Yang, Steinhardt & Hu 2023, ICML (arXiv 2306.17105) [V]: residual within-class structure | Seed-stable class structure is known from convergent learning and CKA. The loss of distance ranking at full collapse is what NC2 predicts. Class-similarity structure is usually read from the head (confusion matrix, weight cosines), which Atlas never compared. The fine structure survives in within-class residuals, not in the centres. |
| CD-7 | known-in-research | Kornblith et al. 2019, CKA, ICML [V]; Nguyen, Raghu & Kornblith 2021, ICLR [V]; Morcos, Raghu & Bengio 2018, NeurIPS [S]; Raghu et al. 2017, SVCCA, NeurIPS [S]; Moschella et al. 2023, ICLR [S]; Ding, Denain & Steinhardt 2021, NeurIPS [V]; Davari et al. 2023, ICLR [S]; Geirhos, Meding & Wichmann 2020, NeurIPS [V]; Jiang et al. 2022 (arXiv 2106.13799) [V]; Baek et al. 2022, NeurIPS [V] | CKA, SVCCA and relative representations are standard analysis tools for model comparison and zero-shot stitching or merging. CKA is dominated by the top PCs, so its rise with collapse is expected. The published label-free use is head-based: two-seed disagreement estimates test error, and agreement-on-the-line estimates OOD accuracy. Atlas did not run that baseline, the within-class-residual CKA, or Procrustes. |
| CD-8 | partly-known | Nguyen, Raghu & Kornblith 2021, ICLR [V]: representations outside the block structure are similar across width and depth; Kornblith et al. 2019 [V] (cross-depth detail [U]); He & Su 2023 [V]: the law holds across architectures | Cross-architecture layer correspondence is an analysis result, not an established deployment pattern. Atlas's SHAPE-versus-LEVEL split (re-calibrate levels per checkpoint) is consistent with it but was tested only from resnet20 to resnet56 on CIFAR-10. |
| CD-9 | partly-known | Papyan et al. 2020 [V]; Hui et al. 2022 [V]; Harun, Gallardo & Kanan 2025, ICML (arXiv 2502.10691) [V]: NC degree trades OOD detection against transfer; Nguyen et al. 2021 and Ding et al. 2021 [V] on CKA top-PC dominance; Kothapalli et al. 2023, TMLR review [S] | NC degree is used as a control knob (Harun: per-layer ETF projector or entropy regularization) and as a transferability estimate (NCTI). The components Atlas unifies (D1 loss, CKA excess, false-alarm law, margin ~ distance) each follow from NC theory. The "single offline calibration coordinate for a controller" framing was not found. |
| CD-10 | known-in-research | Ansuini et al. 2019 [V] (a PCA-based linear dimension, PC-ID, next to TwoNN); Parker et al. 2023 [V] (NC in intermediate layers); Radovanović, Nanopoulos & Ivanović 2010, JMLR, hubness (cited at landmarks.py:9-11) [S] | Participation ratio, dim95 and hubness skew are routine diagnostics; hubness matters for kNN-based methods. They were not seed-stable in Atlas. No controller use. |
| CD-11 | known-in-research | Papyan et al. 2020 (NC4) [V]; Ben-Shaul & Dekel 2022 (arXiv 2201.08924) [V]; Galanti, György & Hutter 2022, ICLR [S] | By NC4, NCC accuracy equals head accuracy at collapse. The NCC classifier is used for few-shot and transfer evaluation and as a convergence diagnostic. It carries no beyond-head accuracy information by construction. The label-free variant is CM-11. |
| CM-2 | partly-known | Papyan et al. 2020 (NC2 equidistant centres, NC4 NCC = head) [V]; Hui et al. 2022 [V] | NC theory predicts that plain nearest-centre distance and the top-2 margin converge once the centres are equidistant and the head is NCC. The AUROC observation on resnet56 extends it. No published misclassification-detection use of this equivalence was found in this topic. |
| CM-5 | not-found-in-prior-art | Papyan et al. 2020 (NC3 self-duality predicts margin = logit gap when centres are proportional to head weights) [V]; Chen et al. 2025/2026, NCTTA (arXiv 2512.10421) [V]: blends geometric proximity with confidence; Liu & Qin 2025 (arXiv 2311.01479) [V]: feature/weight proximity for OOD | NC-based methods combine feature/classifier alignment with confidence for OOD detection and TTA pseudo-label weighting, not for confident-error detection. NC3 explains when the residual should vanish, but head_center_cos does not order the Atlas effect (deitb leads most). The ViT residual beyond the logit gap was not found in this topic's literature. |
| CM-6 | partly-known | Papyan et al. 2020 (NC4) [V]; Harun et al. 2025, ICML [V]; Hui et al. 2022 [V] | That the NC degree changes what feature-space scores can see is published (OOD vs transfer trade-off). Using a measured collapse level to pick margin or distance is not an established practice and was refuted as registered (AH-3). |
| CM-8 | partly-known | Liu & Qin 2025, CVPR (arXiv 2311.01479) [V]: ID features farther from the origin than OOD; Kang, Setlur, Tomlin & Levine 2023/24 (arXiv 2310.00873) [V]: OOD feature norms shrink; Haas, Yolland & Rabus 2023, TMLR [S] | Feature norm is a component of NC-based OOD detectors, which is consistent with the CIFAR direction (errors have lower energy). The ImageNet sign flip for clean misclassifications is not explained by this literature. Any use needs per-backbone sign calibration. |
| CM-11 | partly-known | Papyan et al. 2020 (NC4) [V]; Ben-Shaul & Dekel 2022, NCC mismatch (arXiv 2201.08924) [V]; Chen et al., Neural collapse in test-time adaptation, CVPR 2026 (arXiv 2512.10421) [V]: sample-wise alignment collapse NC3+ degrades with shift and marks unreliable pseudo-labels | Geometry-versus-classifier misalignment is used in NCTTA to weight TTA targets. NCC mismatch is used as a training diagnostic. As a label-free misclassification or distrust detector it is untested in Atlas and was not found in this topic's literature (DOCTOR-type agreement methods belong to the confidence-margin topic). |
| DO-3 | partly-known | Hui, Belkin & Nakkiran 2022 (arXiv 2202.08384) [V]: NC occurs on train but not on test; Papyan et al. 2020 [V] | The mechanism (a train-reference geometry more collapsed than test) is known. The standard remedy is to calibrate detector thresholds on held-out in-distribution data. The quantitative log10-nc1 law for the false-alarm rate was not found. |
| PC-3 | known-in-research | Rangamani et al. 2023, ICML [V]; Parker et al. 2023 [V]; Ben-Shaul & Dekel 2022 [V]; Raghu et al. 2017, SVCCA, NeurIPS [S] (locates where class-specific information forms); Masarczyk et al. 2023, NeurIPS [S]; Wang et al. (arXiv 2311.02960) [V] | Intermediate-layer collapse and linear-probe commit depth are used to pick layers for probing and transfer and to explain tunnel or compression behaviour. In Atlas the commit layer is low-information but marks the pre-collapse tap. |
| PC-6 | partly-known | Ammar et al. 2024, NECO, ICLR (arXiv 2310.06823) [S]; Papyan et al. 2020 [V] | NECO scores each sample by its relative norm inside the ETF/principal subspace for OOD detection. Atlas's batch-mean class_sub_frac reduces to the between-class variance share and was dropped. |
| ST-3 | partly-known | Ammar et al. 2024, NECO [S]; Liu & Qin 2025 [V] | Decomposing features into the class-mean span and its complement is used per sample for OOD detection. Atlas's batch-mean Hotelling T-squared drift version (T_perp/T_par) was not found and was ruled out as registered. |
| ST-6 | partly-known | Ramasesh, Dyer & Raghu 2020/21 (arXiv 2007.07400) [V]: similarity analysis localizes forgetting to deeper layers; Kornblith et al. 2021, NeurIPS [V]: CKA localizes loss-function effects to the last layers; Chen et al. 2026, NCTTA [V]; Kornblith et al. 2019 [V] | CKA and NC statistics are used offline to localize what changed between checkpoints, and NC3+ alignment is used inside a TTA method. An online monitor that fires on backbone deformation before the output entropy collapses was not found. It is untested in Atlas (no results/tta_*). |
| LH-1 | known-in-research | Xie et al. 2023, dispersion score (arXiv 2303.15488) [V]; Jiang, Nagarajan, Baek & Kolter 2022 (arXiv 2106.13799) [V]; Baek et al. 2022, Agreement-on-the-line, NeurIPS (arXiv 2206.13089) [V] | Label-free accuracy estimation under shift is an established research task. Feature-separability (dispersion) and multi-model agreement estimators are benchmarked there. Atlas's H has not been compared with them (nor with confidence-based estimators from the harm topic). |
