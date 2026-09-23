# Probing and corruption geometry: prior art for Atlas

Last reviewed: 2026-09-23; part of the Atlas knowledge base (docs/knowledge/README.md)

Prior-art labels in this file (rediscovered, extends, known-in-research, not found in prior art, and so on) mark
starting points to adopt and refine for Atlas's use; they never down-rank an item (docs/knowledge/README.md §1).

- **Topic:** probing-corruption-geometry, inventory items PC-1 to PC-6.
- **Also classified here** where this literature bears on them: ST-1 to ST-4, ST-8, LH-1, LH-2, LH-4, CD-1 and CD-7.
- **Written against:** HEAD 343f651.

**How citations were checked:**
- [opened]: the abstract or venue page was opened.
- [search]: authors, title and venue were confirmed from search-result listings only.
- [unconfirmed: ...]: a detail that could not be confirmed.

No citation was written from memory alone. The web-search budget ran out part-way through, so a few relevant works are
named in section 2.9 but not cited.

**Owner's framing applied:** the question is which kinds of input information can be read out of the backbone's
internal geometry, how reliably, and whether that adds anything to what the output head already exposes. The controller
comes later.

---

## 1. What this family of methods measures

### 1.1 Layer-wise linear probes
- **What a probe is.** A linear classifier or regressor fitted on frozen activations at one tap, predicting one factor:
  class, a pixel statistic, or a corruption label. The score profile over depth shows where that factor is linearly
  accessible.
- **Atlas implementation.**
  - `atlas/invariants/decodability.py:57-79` standardises the features, then fits logistic regression (C = 1) or RidgeCV
    with 5-fold shuffled CV. n_train is 4000 (`experiments/queue/atlas_v1_resnet20_s1.yaml:26`).
  - `atlas/invariants/flow.py:45-84` derives two quantities. The commit layer is the first tap whose score reaches tau ×
    the best score. Washout is the best score minus the penult score (`flow.py:84`).
- **What a probe score does not tell you:**
  1. whether the network uses the information: decodable is not the same as used;
  2. whether the score reflects the representation, or instead the probe's capacity and the make-up of the probe pool.

  The probing literature documents both limits (section 2.1).

### 1.2 Invariance and nuisance removal across depth
- **The question:** which nuisance factors (luminance, contrast, spatial-frequency content, orientation) are discarded
  along the way, and which survive to the head input.
- **Theory:** framed as information minimality.
  - The information bottleneck view.
  - Achille & Soatto: invariance to nuisances is equivalent to minimality, and depth biases networks toward it.
- **Empirical tools:**
  - invariance tests on units;
  - equivariance maps;
  - probe washout;
  - at the extreme, neural collapse: within-class variability at the penult goes to zero on train data.

### 1.3 Corruption-displacement geometry
- **The object:** for a clean input and its corrupted copy, the displacement delta_i = f(x_i^c) − f(x_i) at a tap.
- **Statistics Atlas computes** (`atlas/invariants/sensitivity.py:25-73`):
  - size of the mean displacement (`:64-65`);
  - its direction in a PCA frame;
  - its share inside the class-mean span (`:67`, after the SVD fix at `:36-37`);
  - coherence, the mean cosine between each sample's displacement and the mean displacement (`:46`).
- **The same object in the literature:**
  - as a shift in feature statistics (re-estimating BN statistics; activation-mean discrepancy);
  - as the input to two-sample shift tests;
  - in the Fourier view of which corruption moves which frequencies.

### 1.4 The pooling layer decides what is probed
Every Atlas conv tap is global-average-pooled to channel means (`atlas/extract_acts.py:88-95`), giving 16-, 32- or 64-d
vectors (`results/atlas_v0_resnet20_cifar10/SESSION.md:37`). A mean-plus-spatial-std option (`gap_std`,
`extract_acts.py:44`) was deferred ("try later", `experiments/queue/atlas_v0_resnet20_cifar10.yaml:27`) and never run.
This has four consequences:
- **(a) Luminance at the stem is readable by construction.** Mean luminance is almost a linear function of the stem
  channel means, so a stem luminance R² near 1 is guaranteed. Atlas flags this itself
  (`results/atlas_v0_resnet20_cifar10/SESSION.md:56-59`).
- **(b) Frequency and orientation are readable only through filter-response energy.** Random filters already provide
  that energy.
- **(c) Width is confounded with depth.** Tap width changes by stage, so a probe profile over depth mixes depth with
  dimensionality.
- **(d) Second-moment statistics are missing from the dumps.** Texture and style statistics (Gram matrices) carry much of
  the corruption signal in the OOD literature.

### 1.5 Offline map descriptors versus runtime sensors
- **Probes and displacement statistics are offline tools.** Probes need factor values. Displacement statistics need the
  clean twin of each corrupted input.
- **Candidate controller sensors are the rest.** Only statistics computable from an unlabeled, unpaired input stream
  qualify: batch means, densities, and templates fitted offline and applied online.

---

## 2. Key works

### 2.1 Probing methodology
| Work | Link | Check | Why it matters here |
|---|---|---|---|
| Alain, G. & Bengio, Y. (2016). *Understanding intermediate layers using linear classifier probes.* arXiv:1610.01644; ICLR 2017 Workshop Track | https://arxiv.org/abs/1610.01644 | [search] | Probes at every layer of Inception v3 and ResNet-50. Linear separability of the class rises monotonically with depth |
| Hewitt, J. & Liang, P. (2019). *Designing and Interpreting Probes with Control Tasks.* EMNLP-IJCNLP 2019, pp. 2733-2743 | https://aclanthology.org/D19-1275/ | [search] | Control tasks and "selectivity": a probe should score high on the real task and low on a matched random-label task |
| Belinkov, Y. (2022). *Probing Classifiers: Promises, Shortcomings, and Advances.* Computational Linguistics 48(1):207-219 | https://aclanthology.org/2022.cl-1.7/ | [search] | Review of probing's limits: probe capacity, and correlation versus use |
| Hermann, K. L. & Lampinen, A. K. (2020). *What shapes feature representations? Exploring datasets, architectures, and training.* NeurIPS 2020 | https://arxiv.org/abs/2006.12433 | [search] | Task-irrelevant features are partly suppressed, not removed. Which of two redundant features a network adopts reflects what was most linearly decodable in the untrained network |
| Kirichenko, P., Izmailov, P. & Wilson, A. G. (2023). *Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations.* ICLR 2023 | https://arxiv.org/abs/2204.02937 | [opened] | The penult keeps features the head does not use; re-training only the last layer recovers them |
| Goyal, P., Mahajan, D., Gupta, A. & Misra, I. (2019). *Scaling and Benchmarking Self-Supervised Visual Representation Learning.* ICCV 2019 | https://openaccess.thecvf.com/content_ICCV_2019/html/Goyal_Scaling_and_Benchmarking_Self-Supervised_Visual_Representation_Learning_ICCV_2019_paper.html | [search; the layer-wise pooling protocol was not confirmed (PDF unreadable)] | Standard benchmark that uses linear probes on frozen features |

### 2.2 Invariance, information and compression across depth
| Work | Link | Check | Why it matters here |
|---|---|---|---|
| Tishby, N. & Zaslavsky, N. (2015). *Deep Learning and the Information Bottleneck Principle.* IEEE ITW 2015 | https://arxiv.org/abs/1503.02406 | [opened] | The information-bottleneck framing of layer-wise compression |
| Shwartz-Ziv, R. & Tishby, N. (2017). *Opening the Black Box of Deep Neural Networks via Information.* arXiv:1703.00810 | https://arxiv.org/abs/1703.00810 | [search] | Claims that a compression phase dominates training |
| Saxe, A. M., Bansal, Y., Dapello, J., Advani, M., Kolchinsky, A., Tracey, B. D. & Cox, D. D. (2018). *On the information bottleneck theory of deep learning.* ICLR 2018; J. Stat. Mech. 2019, 124020 | https://iopscience.iop.org/article/10.1088/1742-5468/ab3985 | [search] | Counter-evidence: compression is not universal and has no evident causal link to generalisation. So washout should not be narrated as a "compression phase" |
| Achille, A. & Soatto, S. (2018). *Emergence of Invariance and Disentanglement in Deep Representations.* JMLR 19:1-34 | https://jmlr.org/papers/v19/17-646.html | [opened] | Invariance to nuisances is equivalent to information minimality, and "stacking layers and injecting noise during training naturally bias the network towards learning invariant representations" |
| Goodfellow, I., Lee, H., Le, Q. V., Saxe, A. & Ng, A. Y. (2009). *Measuring Invariances in Deep Networks.* NIPS 2009, pp. 646-654 | https://papers.nips.cc/paper_files/paper/2009/file/428fca9bc1921c25c5121f9da7815cde-Paper.pdf | [search] | Direct invariance tests of features against input transformations |
| Lenc, K. & Vedaldi, A. (2015). *Understanding image representations by measuring their equivariance and equivalence.* CVPR 2015 (arXiv:1411.5908) | https://arxiv.org/abs/1411.5908 | [opened] | Equivariance, invariance and equivalence of representations under input transformations |
| Masarczyk, W. et al. (2023). *The Tunnel Effect: Building Data Representations in Deep Neural Networks.* NeurIPS 2023 | https://arxiv.org/abs/2305.19753 | [search] | Early layers build linearly separable features; later layers (the "tunnel") compress and cut rank. Probes placed before the tunnel transfer better out of distribution |
| Papyan, V., Han, X. Y. & Donoho, D. L. (2020). *Prevalence of neural collapse during the terminal phase of deep learning training.* PNAS | https://arxiv.org/abs/2008.08186 | [opened] | Within-class variability at the penult collapses: the limiting case of nuisance removal (on train data) |
| Ansuini, A., Laio, A., Macke, J. H. & Zoccolan, D. (2019). *Intrinsic dimension of data representations in deep neural networks.* NeurIPS 2019 | https://arxiv.org/abs/1905.12784 | [opened] | The intrinsic-dimension profile rises then falls with depth (a "hunchback"); last-layer ID predicts test accuracy |
| Kornblith, S., Norouzi, M., Lee, H. & Hinton, G. (2019). *Similarity of Neural Network Representations Revisited.* ICML 2019 | https://arxiv.org/abs/1905.00414 | [opened] | CKA; layer correspondence across networks trained from different initialisations |
| Nguyen, T., Raghu, M. & Kornblith, S. (2021). *Do Wide and Deep Networks Learn the Same Things?* ICLR 2021 | https://arxiv.org/abs/2010.15327 | [opened] | "Block structure": runs of layers that preserve a dominant principal component |
| Ben-Shaul, I. & Dekel, S. (2022). *Nearest Class-Center Simplification through Intermediate Layers.* arXiv:2201.08924 | https://arxiv.org/abs/2201.08924 | [opened; venue unconfirmed] | Nearest-class-centre structure in intermediate layers |

### 2.3 Random-weight baselines
| Work | Link | Check | Why it matters here |
|---|---|---|---|
| Saxe, A. M., Koh, P. W., Chen, Z., Bhand, M., Suresh, B. & Ng, A. Y. (2011). *On Random Weights and Unsupervised Feature Learning.* ICML 2011 | https://icml.cc/2011/papers/551_icmlpaper.pdf | [search] | Convolutional pooling architectures are frequency-selective and translation-invariant even with random weights |
| Ustyuzhaninov, I., Brendel, W., Gatys, L. A. & Bethge, M. (2017). *What does it take to generate natural textures?* ICLR 2017. Earlier arXiv version: *Texture Synthesis Using Shallow Convolutional Networks with Random Filters*, arXiv:1606.00021 | https://openreview.net/forum?id=BJhZeLsxx ; https://arxiv.org/abs/1606.00021 | [search] | One layer of random filters is enough for a good texture model |

### 2.4 Pooling
| Work | Link | Check | Why it matters here |
|---|---|---|---|
| Lin, M., Chen, Q. & Yan, S. (2014). *Network In Network.* ICLR 2014 | https://arxiv.org/abs/1312.4400 | [search] | Introduces global average pooling (GAP) |
| Islam, M. A., Jia, S. & Bruce, N. D. B. (2020). *How Much Position Information Do Convolutional Neural Networks Encode?* ICLR 2020 | https://arxiv.org/abs/2001.08248 | [search] | Zero padding lets CNNs encode absolute position |
| Islam, M. A., Kowal, M., Jia, S., Derpanis, K. G. & Bruce, N. D. B. (2021). *Global Pooling, More than Meets the Eye: Position Information is Encoded Channel-Wise in CNNs.* ICCV 2021 | https://arxiv.org/abs/2108.07884 | [opened] | GAP does not remove all spatial information: position is encoded in the channel ordering |

### 2.5 Corruption benchmarks, frequency and degradation
| Work | Link | Check | Why it matters here |
|---|---|---|---|
| Hendrycks, D. & Dietterich, T. (2019). *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations.* ICLR 2019 | https://arxiv.org/abs/1903.12261 ; https://github.com/hendrycks/robustness | [search] | ImageNet-C/-P; the CIFAR-10-C sets Atlas uses come from the same release |
| Yin, D., Gontijo Lopes, R., Shlens, J., Cubuk, E. D. & Gilmer, J. (2019). *A Fourier Perspective on Model Robustness in Computer Vision.* NeurIPS 2019 | https://arxiv.org/abs/1906.08988 | [opened] | Gaussian augmentation and adversarial training raise robustness to high-frequency corruptions and lower it to low-frequency ones (fog, contrast). Corruptions are characterised by where their energy sits in the Fourier spectrum |
| Wang, H., Wu, X., Huang, Z. & Xing, E. P. (2020). *High-frequency Component Helps Explain the Generalization of Convolutional Neural Networks.* CVPR 2020 | https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_High-Frequency_Component_Helps_Explain_the_Generalization_of_Convolutional_Neural_Networks_CVPR_2020_paper.html | [search] | CNNs pick up high-frequency components that humans barely perceive |
| Jo, J. & Bengio, Y. (2017). *Measuring the tendency of CNNs to Learn Surface Statistical Regularities.* arXiv:1711.11561 | https://arxiv.org/abs/1711.11561 | [opened] | Fourier-filtered test sets reveal a reliance on surface statistics |
| Geirhos, Temme, Rauber, Schütt, Bethge & Wichmann (2018). *Generalisation in humans and deep neural networks.* NeurIPS 2018 | https://proceedings.neurips.cc/paper/2018/hash/0937fb5864ed06ffb59ae5f9b5ed67a9-Abstract.html | [search] | Twelve degradations; DNNs are less robust than humans to nearly all of them |
| Geirhos, R., Rubisch, P., Michaelis, C., Bethge, M., Wichmann, F. A. & Brendel, W. (2019). *ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness.* ICLR 2019 | https://arxiv.org/abs/1811.12231 | [opened] | Texture bias; shape-biased training improves robustness to distortions |
| Dodge, S. & Karam, L. (2016). *Understanding how image quality affects deep neural networks.* QoMEX 2016. doi:10.1109/QoMEX.2016.7498955 | https://arxiv.org/abs/1604.04004 | [search] | Classifiers are especially sensitive to blur and noise |
| Ford, N., Gilmer, J., Carlini, N. & Cubuk, E. D. (2019). *Adversarial Examples Are a Natural Consequence of Test Error in Noise.* arXiv:1901.10513 | https://arxiv.org/abs/1901.10513 | [opened; ICML 2019 venue unconfirmed] | Noise corruption and adversarial vulnerability are two faces of one phenomenon |
| Mintun, E., Kirillov, A. & Xie, S. (2021). *On Interaction Between Augmentations and Corruptions in Natural Corruption Robustness.* NeurIPS 2021 | https://arxiv.org/abs/2102.11273 | [opened] | Similarity between augmentations and test corruptions predicts measured robustness; augmentations "may not generalize well beyond the existing benchmark" |

### 2.6 Corruptions as a shift in feature statistics; adaptation
| Work | Link | Check | Why it matters here |
|---|---|---|---|
| Schneider, S., Rusak, E., Eck, L., Bringmann, O., Brendel, W. & Bethge, M. (2020). *Improving robustness against common corruptions by covariate shift adaptation.* NeurIPS 2020 | https://proceedings.neurips.cc/paper/2020/hash/85690f81aadc1749175c187784afc9ee-Abstract.html | [search] | Replacing the BN statistics with statistics of the corrupted images improves robustness across 25 models |
| Nado, Z., Padhy, S., Sculley, D., D'Amour, A., Lakshminarayanan, B. & Snoek, J. (2020). *Evaluating Prediction-Time Batch Normalization for Robustness under Covariate Shift.* arXiv:2006.10963 | https://arxiv.org/abs/2006.10963 | [opened] | Strong on synthetic corruptions; weaker on natural shift and with pre-training |
| Wang, D., Shelhamer, E., Liu, S., Olshausen, B. & Darrell, T. (2021). *Tent: Fully Test-time Adaptation by Entropy Minimization.* ICLR 2021 | https://arxiv.org/abs/2006.10726 | [opened] | Updates the normalisation statistics and channel-wise affine parameters |
| Lee, Y., Chen, A. S., Tajwar, F., Kumar, A., Yao, H., Liang, P. & Finn, C. (2023). *Surgical Fine-Tuning Improves Adaptation to Distribution Shifts.* ICLR 2023 | https://arxiv.org/abs/2210.11466 | [opened] | "for image corruptions, fine-tuning only the first few layers works best" |

### 2.7 Detecting shift and OOD from intermediate features
| Work | Link | Check | Why it matters here |
|---|---|---|---|
| Lee, K., Lee, K., Lee, H. & Shin, J. (2018). *A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks.* NeurIPS 2018 | https://arxiv.org/abs/1807.03888 | [opened] | Class-conditional Gaussians on low- and upper-level features (a multi-layer feature ensemble) |
| Sastry, C. S. & Oore, S. *Detecting Out-of-Distribution Examples with In-distribution Examples and Gram Matrices.* arXiv:1912.12510 (NeurIPS 2019 workshop); published as *Detecting Out-of-Distribution Examples with Gram Matrices*, ICML 2020 (PMLR 119) | https://arxiv.org/abs/1912.12510 | [opened] | Gram-matrix (second-moment) anomalies; needs no OOD data for tuning |
| Dong, X., Guo, J., Li, A., Ting, W.-T., Liu, C. & Kung, H. T. (2022). *Neural Mean Discrepancy for Efficient Out-of-Distribution Detection.* CVPR 2022 | https://arxiv.org/abs/2104.11408 | [search] | Activation means of OOD mini-batches deviate from the training means, which BN layers store for free |
| Rabanser, S., Günnemann, S. & Lipton, Z. C. (2019). *Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift.* NeurIPS 2019 | https://arxiv.org/abs/1810.11953 | [search] | Two-sample tests after dimensionality reduction by a pretrained classifier perform best |
| Kamoi, R. & Kobayashi, K. (2020). *Why is the Mahalanobis Distance Effective for Anomaly Detection?* arXiv:2003.00402 | https://arxiv.org/abs/2003.00402 | [search; venue unconfirmed] | The anomaly signal sits in low-variance principal directions that classification ignores |
| Wang, H., Li, Z., Feng, L. & Zhang, W. (2022). *ViM: Out-Of-Distribution with Virtual-logit Matching.* CVPR 2022 | https://arxiv.org/abs/2203.10807 | [opened] | Residual off the principal subspace, combined with the logits |
| Ginsberg, T., Liang, Z. & Krishnan, R. G. *A Learning Based Hypothesis Test for Harmful Covariate Shift.* arXiv:2212.02742 | https://arxiv.org/abs/2212.02742 | [opened; venue unconfirmed] | Separates harmful shift from benign shift |

### 2.8 Distortion identification, label-free accuracy and head baselines
| Work | Link | Check | Why it matters here |
|---|---|---|---|
| Moorthy, A. K. & Bovik, A. C. (2011). *Blind Image Quality Assessment: From Natural Scene Statistics to Perceptual Quality.* IEEE TIP. doi:10.1109/TIP.2011.2147325 | https://pubmed.ncbi.nlm.nih.gov/21521667/ | [search] | DIIVINE, a two-stage design: identify the distortion from scene statistics, then apply a distortion-specific quality model. It is the template for type-to-response routing |
| Deng, W. & Zheng, L. (2021). *Are Labels Always Necessary for Classifier Accuracy Evaluation?* CVPR 2021 | https://arxiv.org/abs/2007.02915 | [opened; that it uses the Fréchet distance comes from the evaluation's science audit, not the abstract; confirmed in `label-free-accuracy-harm.md`, which read the full text] | AutoEval: regresses accuracy from feature statistics of transformed meta-sets |
| Garg, S., Balakrishnan, S., Lipton, Z. C., Neyshabur, B. & Sedghi, H. (2022). *Leveraging Unlabeled Data to Predict Out-of-Distribution Performance.* ICLR 2022 | https://arxiv.org/abs/2201.04234 | [opened] | ATC, a confidence-threshold accuracy estimate |
| Hendrycks, D. & Gimpel, K. (2017). *A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks.* ICLR 2017 | https://arxiv.org/abs/1610.02136 | [opened] | The MSP baseline |
| Park, N. & Kim, S. (2022). *How Do Vision Transformers Work?* ICLR 2022 | https://arxiv.org/abs/2202.06709 | [search] | Fourier analysis of feature maps over depth: attention layers act as low-pass filters, convolutions as high-pass. Relevant once probing moves to ViTs |

### 2.9 Named but not cited (search budget exhausted)
- amnesic and MDL probing;
- AugMix-style augmentation baselines;
- neural collapse in intermediate layers beyond Ben-Shaul & Dekel (covered in `collapse-dimension-similarity.md`);
- SAR and AETTA test-time-adaptation monitors (covered in `shift-type-tta-monitoring.md`);
- Hui, Belkin & Nakkiran 2022 on collapse being weaker on test data (covered in `collapse-dimension-similarity.md`).

---

## 3. How it is used

| Usage pattern | Standard practice or research | Typical metrics | Known failure modes |
|---|---|---|---|
| Evaluating a frozen representation with a linear probe (final layer), and locating information by depth (layer-wise probes) | Final-layer linear evaluation is standard practice in self-supervised learning. Layer-wise probing is standard *research* methodology (Alain & Bengio; Belinkov review) | Probe accuracy or R²; selectivity over a control task (Hewitt & Liang) | Probe capacity; pool structure; decodable is not used (Hermann & Lampinen; Kirichenko); random-weight features already decode frequency and texture (Saxe et al. 2011; Ustyuzhaninov et al. 2017) |
| Choosing where to read or adapt | Research, but BN re-estimation and Tent are the default test-time-adaptation baselines | Post-adaptation accuracy or mCE on CIFAR-10-C / ImageNet-C | Gains on synthetic corruptions carry less to natural shift and to pre-trained models (Nado et al.). Needs batches. Corruptions are best adapted in the first layers (Lee et al. 2023) |
| Shift and OOD detection from intermediate features (multi-layer Mahalanobis, Gram matrices, NMD, residual subspace, two-sample tests on representations) | Research. "Pretrained classifier + two-sample test" (Rabanser et al.) is the closest to production drift monitoring | Per sample: AUROC, FPR at 95% TPR. Per batch: detection power against batch size | Flags benign shift as well as harmful shift (Ginsberg et al.); rankings depend on the shift type and the sample size |
| Robustness benchmarking and diagnosis (CIFAR-10-C / ImageNet-C, Fourier heat maps, degradation studies) | Standard practice for robustness evaluation | mCE; accuracy per corruption and severity; Fourier sensitivity maps | Benchmark overfitting (Mintun et al.); frequency trade-offs make a single robustness number misleading (Yin et al.) |
| Label-free accuracy estimation from features or confidence (AutoEval, ATC) | Research | MAE of the predicted accuracy across shifted sets | Needs a meta-set of synthetic shifts similar to deployment |
| Identifying the distortion before a type-specific response (NR-IQA, DIIVINE) | Established in image-quality assessment; research for DNN test-time adaptation | Distortion-classification accuracy; downstream quality correlation | Closed label set: an unseen distortion is forced into a known class |

**Failure modes of probing and displacement analysis, as they apply to Atlas:**
- **F1. Pool-driven scores.**
  - All pixel factors are probed on a pool mixing clean test with the 45 corrupt splits
    (`atlas/factors/lowlevel.py:54-183`, pool = "mixed"; `atlas/invariants/decodability.py:30`).
  - A factor that varies mostly *between* splits is therefore decodable whenever the split is identifiable: luminance
    under brightness, contrast and fog; high-frequency energy under noise and blur.
  - A clean-only pool was promised and never run (`results/atlas_v1_resnet20_s1/SESSION.md:106, :131`;
    `results/atlas_v1_resnet20_s3/SESSION.md:405`).
- **F2. Paired-image leakage.**
  - Corrupt rows are paired with test rows by index (`atlas/context.py:15-17`; `atlas/extract_acts.py:313`), and the
    probes use shuffled KFold (`decodability.py:70, :77`).
  - So versions of the same source image land on both sides of a CV split.
  - For a 64-d linear probe the inflation is probably small, but it was never measured. The standard fix is to group
    folds by source image.
- **F3. Holdout leak on the corruption axis.**
  - The probe pools contain the five confirmation corruptions (`results/atlas_v1_resnet20_s3/SESSION.md:367`), so no
    corruption-generalisation claim has clean evidence.
  - This is the Atlas analogue of Mintun et al.'s benchmark-overfitting point.
- **F4. Random weights already decode input statistics.**
  - Atlas's own nulls show it:
    - row 3's anisotropy clause holds in random init (0.085);
    - the depth-56 null clears the luminance clause (0.473) and fails only on highfreq (0.278) (`ATLAS_STATUS.md:12`;
      `results/atlas_v1_resnet56_s0hub/SESSION.md:411-414`).
  - Saxe et al. 2011 and Achille & Soatto both predict this.
- **F5. Width and pooling confound.**
  - GAP width runs 16 → 32 → 64 over the stages, so scores across taps compare different dimensionalities.
  - GAP keeps only first moments. Islam et al. 2021 caution that pooled channels can still carry more (for example,
    position).
- **F6. Decodable is not used, and used is not useful.**
  - For a controller, the question is not whether the penult encodes a factor.
  - It is whether the factor predicts harm, or predicts which response helps.
- **F7. Compression narratives.**
  - Report washout as a measured drop in linear decodability.
  - Do not call it an information-bottleneck compression phase: Saxe et al. 2018 dispute that such a phase is
    universal.

---

## 4. Known results relevant to Atlas

- **K1. Class separability rises monotonically with depth** (Alain & Bengio).
  - A monotone profile with tau 0.9 puts the commit late.
  - Atlas already notes that the class commits at layer2.x or later for any tau > 0.429
    (`results/atlas_v0_resnet20_cifar10/SESSION.md:55-56`).
- **K2. Depth biases networks toward invariance** (Achille & Soatto).
  - A deep random-init network that washes out luminance (depth-56 null 0.473 > 0.30) is therefore expected, not
    anomalous.
- **K3. Random conv-plus-pooling stacks are frequency-selective** (Saxe et al. 2011), **and random filters capture
  texture statistics** (Ustyuzhaninov et al. 2017).
  - Expect random nets to decode highfreq and anisotropy, and to route the N/B/L families.
  - They do: N at 0.99-1.00, B at 0.987-0.996, L at 0.993-1.00 (`results/anomaly_h1/SESSION.md:385`).
- **K4. What the untrained network can decode shapes what is learned** (Hermann & Lampinen).
  - The random-init profile is the right baseline.
  - Atlas measures it; it should report learned-minus-random as the headline.
- **K5. Task-irrelevant features are suppressed, not deleted, and the penult keeps unused features** (Hermann &
  Lampinen; Kirichenko et al.).
  - So highfreq and anisotropy surviving to the penult (washout < 0.20) is the expected pattern.
  - The specific observation is that luminance is the factor lost: washout 0.549-0.736 at resnet20 and 0.608 / 0.669 at
    resnet56 (`ATLAS_STATUS.md:12`).
- **K6. CNNs use high-frequency content, and their robustness trade-offs follow where a corruption sits in the Fourier
  spectrum** (Yin et al.; Wang et al. 2020; Jo & Bengio).
  - Atlas's "noise" template also routes glass blur and frost. The SESSION's own reading is that N is a
    "high-frequency-change template" (`results/anomaly_h1/SESSION.md:386`).
  - That is the Fourier view restated. The family labels are Fourier-energy classes, not semantic families.
- **K7. Corruptions act largely through feature statistics.**
  - Re-estimating BN statistics recovers much of the lost accuracy (Schneider et al.; Nado et al.).
  - OOD mini-batches have shifted activation means (NMD).
  - Atlas's batch-mean T² statistics at pre-collapse taps (AX-1, AX-2a) belong to this family, and its baselines apply.
- **K8. Corruptions are best adapted in the first layers** (Lee et al. 2023).
  - This fits the AX-2a INFO argmax for brightness at l10 / stem (`results/anomaly_h1/SESSION.md:293-305`; gate CLOSED).
  - But Stage 0 found that the data leaned *against* "the row-6 gap is tap placement" for 2 of 3 targets. Defocus/highfreq
    is still decodable at the penult (0.487), and motion/anisotropy is never above 0.327
    (`results/atlas_v0_resnet20_cifar10/SESSION.md:61-62`).
- **K9. Late layers compress, and probes placed before the compression transfer better out of distribution** (tunnel
  effect; ID hunchback; block structure).
  - Atlas's biggest CKA reorganisation comes right after the ID peak at both depths
    (`results/atlas_v1_resnet56_s0hub/SESSION.md:375`), matching this literature.
  - Pre-collapse taps are therefore the natural place for shift-type readouts.
- **K10. Anomaly signal lives in low-variance directions** (Kamoi & Kobayashi; ViM's residual).
  - This is the literature motivation for T_perp and e_perp.
  - But at a strongly collapsed penult (B/T 0.930 / 0.931, `results/anomaly_h1/SESSION.md:205`), the class span holds
    most of the variance.
  - A mean displacement's class-span share then tracks B/T and carries no information (PC-6;
    `docs/plans/ANOMALY_H1.md:459`).
- **K11. Two-sample tests after reduction by the pretrained classifier are strong batch shift detectors** (Rabanser et
  al.).
  - That makes a head-based batch detector the baseline any geometric batch statistic must beat.
- **K12. Benchmark overfitting** (Mintun et al.).
  - Corruption-axis claims need a held-out corruption set that is dissimilar to the probe pool.

---

## 5. Relation to Atlas items

| id | Atlas finding (short) | Relation | Prior art | Atlas evidence |
|---|---|---|---|---|
| PC-1 | Luminance washes out at the penult (> 0.30); highfreq and anisotropy are kept (< 0.20). Seed-stable; weak learned-versus-random separation at depth 56 | **Rediscovered in kind, extends in specifics.** Nuisance suppression with depth, partial retention of unused features, and a random-weight frequency code are all known. The exact factor triple on CIFAR ResNets at two depths with nulls was not found in the literature. **Untested:** clean-only pool, grouped CV, head comparison | Alain & Bengio 2016; Achille & Soatto 2018; Hermann & Lampinen 2020; Kirichenko et al. 2023; Saxe et al. 2011; Masarczyk et al. 2023 | `ATLAS_STATUS.md:12`; `results/atlas_v1_resnet20_s1/SESSION.md:59-62`; `results/atlas_v1_resnet56_s0hub/SESSION.md:237, :411-414`; `results/atlas_v0_resnet20_cifar10/SESSION.md:41-62` |
| PC-2 | Corruption type, family and severity probe profiles are seed-stable (MAD 0.009-0.020 in s3-s4) | **Rediscovered.** Distortion identification from image statistics is the first stage of NR-IQA. Seed-stable profiles follow from seed-stable representations. **Untested:** whether this beats pixel statistics or the head | Moorthy & Bovik 2011; Kornblith et al. 2019; Hewitt & Liang 2019 | `results/atlas_v1_resnet20_s3/SESSION.md:129-146`; `results/atlas_v1_resnet20_s1/SESSION.md:104-106` |
| PC-3 | The class commits at layer3.1 (r20) / layer3.5 (r56); the biggest reorganisation comes right after the ID peak | **Rediscovered.** Monotone separability; ID hunchback; compression tunnel; block structure | Alain & Bengio 2016; Ansuini et al. 2019; Masarczyk et al. 2023; Nguyen et al. 2021; Ben-Shaul & Dekel 2022 | `ATLAS_STATUS.md:13`; `results/atlas_v1_resnet56_s0hub/SESSION.md:261, :375` |
| PC-4 | Defocus and motion blur are less coherent than Gaussian/shot noise at equal severity, 6/6 in every trained net. Coherence tracks mean shift / per-sample shift (r 0.994-0.996) | **Partly known.** Corruption as a mean shift of feature statistics underlies BN adaptation, NMD and two-sample tests; the coherence statistic itself was not found. A mechanism consistent with Yin et al. (a hypothesis, untested): additive noise injects roughly image-independent high-frequency energy, which pushes every sample the same way; blur removes image-dependent high-frequency energy, which scatters the samples. **Untested** against the head, and offline only (it needs clean pairs) | Schneider et al. 2020; Dong et al. 2022; Rabanser et al. 2019; Yin et al. 2019 | `ATLAS_STATUS.md:14`; `results/atlas_v1_resnet20_s1/SESSION.md:63-66`; `atlas/invariants/sensitivity.py:41-46, :64-65` |
| PC-5 | AH-6: motion blur is directional, noise keeps moving, C4 tracks collapse | **Extends, then refuted.** The premise that corruptions are best seen and handled early is known (surgical fine-tuning). The specific path-ratio and C4-tracks-collapse clauses failed | Lee et al. 2023; Yin et al. 2019; Masarczyk et al. 2023 | `results/anomaly_h1/SESSION.md:216-235, :360-361, :511` |
| PC-6 | Penult class_sub_frac equals the between-class variance share (dropped) | **Explained by known theory.** Under neural collapse, penult variance is dominated by the class span, so any displacement lies mostly inside it. The class-orthogonal readings move to the pre-collapse tap, in the spirit of residual-subspace detectors | Papyan et al. 2020; Kamoi & Kobayashi 2020; Wang et al. 2022 (ViM) | `results/anomaly_h1/SESSION.md:205`; `docs/plans/ANOMALY_H1.md:459`; `results/atlas_v0_resnet20_cifar10/SESSION.md:74-81` |
| ST-1 | Family router: N/B/L/P = 1.000; random nets route N/B/L; glass and frost go to N | **Rediscovered.** Distortion identification (DIIVINE), Fourier characterisation of corruptions, random-filter frequency selectivity. **Untested:** logit or predicted-histogram router; pixel-statistic router | Moorthy & Bovik 2011; Yin et al. 2019; Saxe et al. 2011; Ustyuzhaninov et al. 2017 | `results/anomaly_h1/SESSION.md:307-316, :383-386` |
| ST-2 | Which tap first sees each drift (gate CLOSED) | **Partly known.** Multi-layer detectors and the corruptions-live-early result exist; a per-corruption argmax-tap profile was not found. Untested at confirmation | Lee et al. 2018; Sastry & Oore; Dong et al. 2022; Lee et al. 2023 | `results/anomaly_h1/SESSION.md:293-305` |
| ST-3 | T_perp / T_par batch tests at the pre-collapse tap (ruled out as registered) | **Partly known.** Batch mean-shift tests on representations (NMD, Rabanser et al.) and residual-subspace scores. **Untested:** the head-based comparator for label shift | Rabanser et al. 2019; Dong et al. 2022; Kamoi & Kobayashi 2020; Wang et al. 2022 | `results/anomaly_h1/SESSION.md:272-291` |
| ST-4 | Per-sample e_perp loses to d1 by 0.16-0.44 AUROC | **Does not carry over.** At a collapsed CIFAR penult, the residual-subspace idea did not transfer to per-sample row-6 drift (the literature targets semantic OOD, not corruptions) | Kamoi & Kobayashi 2020; Wang et al. 2022 | `results/anomaly_h1/SESSION.md:339-350` |
| ST-8 | A v0 pipeline contrast rescale (about 0.82) was found from the training log, not from geometry | **Untested here.** Activation-mean or BN-statistics discrepancy is the standard detector for a global input rescale | Dong et al. 2022; Schneider et al. 2020; Nado et al. 2020 | `results/atlas_v0_resnet20_cifar10/SESSION.md:122-127` |
| LH-4 | Brightness costs ≤ 5.3 pt and grades H ≤ 0.30; the stem-direction mechanism is refuted | **Partly known.** Brightness is a low-frequency, near-global change that depth tends to wash out (consistent with row 3's luminance washout). The mechanism clauses are Atlas-specific and failed | Yin et al. 2019; Achille & Soatto 2018 | `results/anomaly_h1/SESSION.md:197-214, :357-359` |
| LH-1, LH-2 | Harm grade H tracks the accuracy cost between splits (Spearman about 0.97-0.99) | **Known-in-research family:** label-free accuracy prediction from feature statistics or confidence (primary classification belongs to the label-free-accuracy topic). **Untested** against ATC or average confidence | Deng & Zheng 2021; Garg et al. 2022; Ginsberg et al. | `results/anomaly_h1/SESSION.md:157-173, :318-337` |
| CD-1 | ID peaks at layer3.1; last-block drop 0.46-0.50 | **Known** (the code cites Ansuini as a sanity check) | Ansuini et al. 2019 | `ATLAS_STATUS.md:10`; `atlas/invariants/dimension.py:5-9` |
| CD-7 | Seed-pair CKA 0.88-0.92; relrep agreement 4.7× chance | **Known** | Kornblith et al. 2019; Nguyen et al. 2021 | `ATLAS_STATUS.md:18` |

No code or plan file in this topic cites the literature above:
- `decodability.py`, `sensitivity.py` and `flow.py` cite nothing;
- a grep of `docs/plans/*.md` for Alain, Kornblith, Hendrycks, Yin, Achille and Soatto finds no hits.

This differs from the dimension and landmark code, which does cite prior art (`docs/reviews/EVAL_2026-09-23.md` §6,
correction 16).

---

## 6. Baselines Atlas should include next time

**B1. Head-side readouts of the same factors.** This answers the primary question directly.
- Recompute logits = penult · W + b from the existing dumps. This is possible because the fc head is linear
  (`docs/reviews/EVAL_2026-09-23.md` §6, correction 5).
- A linear probe on the 10-d logits can never beat a linear probe on the penult, since the logits are a rank-10 linear
  image of the penult.
- So report per factor:
  - penult R² minus logit R² ("information the head discards");
  - probes on scalar head statistics (maxprob, entropy, logit gap) for severity.
- For ST-1, fit the same whitened-template router on:
  - the 10-d logit mean shift;
  - the predicted-class histogram.

  If these route the families as well as the pre-collapse templates, the geometry adds nothing for shift type.

**B2. Control tasks** (Hewitt & Liang). Shuffle factor values within each split and report selectivity.

**B3. Pool hygiene** (F1-F3):
- a clean-only pool and within-split probes;
- leave-one-corruption-out folds;
- folds grouped by source image;
- the confirmation corruptions removed from every pool.

**B4. Learned-minus-random as the headline.** Atlas already runs the random-init null at every tap. Report the
difference against it, not raw washout.

**B5. A pixel-statistic baseline.**
- Route family and severity from the raw factor values in `atlas/factors/lowlevel.py` (highfreq_ratio, noise_sigma,
  luminance_mean, spectral_anisotropy).
- This is the "no network" baseline for shift type.

**B6. Feature-statistics shift detectors:**
- per-tap activation-mean discrepancy (NMD; BN running means cost nothing);
- second moments (`gap_std`; Gram matrices);
- multi-layer Mahalanobis;
- batch two-sample tests (MMD or KS) on the penult versus on the softmax (Rabanser et al.).

**B7. Label-free accuracy comparators** for the severity and harm readouts: average confidence, ATC, and an
AutoEval-style feature-statistics regression.

**B8. Frequency-controlled perturbations.**
- Use Fourier-basis noise at matched L2 norm (Yin et al.), split into low- and high-frequency corruptions.
- Use it to test the coherence mechanism proposed for PC-4 and the N/B routing.
- Check whether coherence correlates with each image's clean high-frequency content.

**B9. An adaptation reference.** Use BN re-estimation and Tent as the responses whose benefit a type or harm readout
should predict. Surgical fine-tuning's "adapt the first layers" is the reference for choosing which layer to adapt.

---

## 7. Implications for the controller

**Answer to the primary question for this topic:**
- **(i) Low-level input statistics (luminance, spatial-frequency energy, orientation).**
  - Readable at early taps with high seed reliability: S7 MAD ≤ 0.04 at resnet20 and ≤ 0.08 at resnet56
    (`ATLAS_STATUS.md:12`; `results/atlas_v1_resnet20_s3/SESSION.md:129-146`).
  - Much of this is by construction of conv + GAP, and it is present in random networks.
  - At the penult, luminance is mostly lost and frequency content is partly kept.
- **(ii) Corruption type or family.**
  - Readable almost perfectly at pre-collapse taps from batch mean shifts.
  - Random networks do almost as well, so this is input statistics seen through a filter bank, not learned knowledge.
- **(iii) Severity.** Decodable, but never compared with head confidence, which also falls with severity.
- **(iv) Directionality (coherence).** Reduces to mean-shift size, and needs clean pairs, so it is offline only.

**Beyond the head:** shift *type* and *where in the network it appears* are the most plausible kinds of information in
this topic that the head does not expose; the head has only 10 class logits. But no head-side or pixel-side comparator
was run (B1, B5), so "geometry adds" is **untested**, not established.

**Consequences for the design:**
- **C1. Read type early, harm and trust late.**
  - Read type and family, and "what changed", at pre-collapse taps.
  - Read harm and per-sample trust at the penult or the head.
  - This agrees with surgical fine-tuning, with BN adaptation and with the tunnel effect.
- **C2. Treat a type router as a closed-set input-statistics classifier** (DIIVINE-style).
  - Unseen corruption types get forced into the nearest Fourier class, as glass and frost already are forced into N.
  - Add a reject ("unknown type") option before any type-to-response routing.
- **C3. Keep these out of runtime inputs:**
  - penult class_sub_frac (no information);
  - coherence (needs clean pairs);
  - energy-based hold rules (LH-3 refuted).
- **C4. Brightness.**
  - Luminance washout at the penult plus a small brightness cost support HOLD for brightness at depth-56 CIFAR (LH-4).
  - The stem-direction mechanism is refuted. If brightness must be detected, a pixel statistic or a stem channel-mean
    check is enough and cheaper.
- **C5. The measurement that matters is adaptation benefit (ST-6).**
  - BN-statistics adaptation (Schneider et al.; Nado et al.; Tent) is the natural response for the N and B families.
  - A type readout earns its place only if it predicts which response helps.
- **C6. Scope.**
  - Everything here comes from:
    - one dataset (CIFAR-10-C) with synthetic corruptions;
    - one architecture family (ResNet-20 and ResNet-56);
    - GAP-pooled taps.
  - Nado et al. report that BN adaptation is weaker under natural shift, so transfer to natural shift is an open risk.

---

## 8. Item classifications (this topic's reading)

The merged, conflict-resolved classification is in `docs/knowledge/README.md`. This table keeps this topic's reading
and citations.

| item | known? | key citations | how prior work uses it |
|---|---|---|---|
| PC-1 | partly-known | Alain & Bengio 2016 (arXiv:1610.01644); Achille & Soatto 2018, JMLR 19; Hermann & Lampinen 2020, NeurIPS (arXiv:2006.12433); Kirichenko, Izmailov & Wilson 2023, ICLR (arXiv:2204.02937); Saxe et al. 2011, ICML; Masarczyk et al. 2023, NeurIPS (arXiv:2305.19753) | Layer-wise linear probing is standard research practice for finding where factors are linearly accessible. Losing nuisance information with depth while keeping unused features at the penult is known. Atlas's specific factor triple (luminance lost; highfreq and anisotropy kept) on CIFAR ResNets is a local measurement that separates learned from random nets only weakly (depth-56 null passes 2 of 3 clauses). It is inflated by a mixed clean-plus-corrupt pool and by GAP-by-construction effects. Use: choosing the tap to read. Untested against a head-side (logit) probe. |
| PC-2 | known-in-research | Moorthy & Bovik 2011, IEEE TIP (DIIVINE, doi:10.1109/TIP.2011.2147325); Kornblith et al. 2019, ICML (arXiv:1905.00414); Hewitt & Liang 2019, EMNLP (control tasks) | Identifying the distortion from image or feature statistics is the first stage of no-reference image-quality assessment, feeding a distortion-specific response. Seed-stable probe profiles follow from seed-stable representations. Atlas claims only seed stability. It needs control-task, pixel-statistic and head baselines before any readout claim. |
| PC-3 | known-in-research | Alain & Bengio 2016 (arXiv:1610.01644); Ansuini et al. 2019, NeurIPS (arXiv:1905.12784); Masarczyk et al. 2023, NeurIPS; Nguyen, Raghu & Kornblith 2021, ICLR (arXiv:2010.15327); Ben-Shaul & Dekel 2022 (arXiv:2201.08924) | Monotone class separability with depth, the rise-then-fall intrinsic-dimension profile, and compression in late layers (the "tunnel" and "block structure" results) are published. They are used to pick feature-extraction or adaptation layers. In Atlas the commit layer is nearly guaranteed for trained nets, and its role is to locate the pre-collapse tap. |
| PC-4 | partly-known | Schneider et al. 2020, NeurIPS (BN re-estimation); Dong et al. 2022, CVPR (Neural Mean Discrepancy, arXiv:2104.11408); Rabanser, Günnemann & Lipton 2019, NeurIPS (arXiv:1810.11953); Yin et al. 2019, NeurIPS (arXiv:1906.08988) | Treating a corruption as a mean shift of feature statistics underlies BN-statistics adaptation, activation-mean OOD detection and two-sample shift tests. The coherence statistic itself was not found in the literature. By construction it is the ratio of mean shift to per-sample shift. The noise-versus-blur ordering fits a Fourier mechanism (proposed here, untested). It is an offline descriptor that needs clean pairs, never compared with the head. |
| PC-5 | partly-known | Lee et al. 2023, ICLR (Surgical Fine-Tuning, arXiv:2210.11466); Yin et al. 2019, NeurIPS; Masarczyk et al. 2023, NeurIPS | The premise is known: corruptions are best seen and adapted in early layers (surgical fine-tuning: "for image corruptions, fine-tuning only the first few layers works best"). Atlas's specific clauses were refuted: the motion/noise path ratio, and C4 tracking collapse. No controller use. |
| PC-6 | partly-known | Papyan, Han & Donoho 2020, PNAS (arXiv:2008.08186); Kamoi & Kobayashi 2020 (arXiv:2003.00402); Wang et al. 2022, CVPR (ViM, arXiv:2203.10807) | Neural collapse implies that penult variance is dominated by the class span, so a displacement's class-span share tracks the between-class variance share (what Atlas found). The literature reads anomaly and shift signal from low-variance or residual directions instead (Mahalanobis, ViM). Atlas correctly dropped the penult feature. |
| ST-1 | known-in-research | Moorthy & Bovik 2011, IEEE TIP (DIIVINE); Yin et al. 2019, NeurIPS; Saxe et al. 2011, ICML (random weights are frequency-selective); Ustyuzhaninov et al. 2017, ICLR (random filters model textures) | Classifying the distortion before a type-specific response is the DIIVINE design in image-quality assessment. Characterising corruptions by where their energy sits in the Fourier spectrum is Yin et al. Random conv stacks already separate frequency classes, which explains Atlas's random-init nulls routing N/B/L. It is a closed-set classifier (glass and frost route to N). Untested against logit, predicted-histogram or raw-pixel routers. |
| ST-2 | partly-known | Lee, Lee, Lee & Shin 2018, NeurIPS (multi-layer Mahalanobis, arXiv:1807.03888); Sastry & Oore (Gram matrices, arXiv:1912.12510); Dong et al. 2022, CVPR (NMD); Lee et al. 2023, ICLR (surgical fine-tuning) | Multi-layer feature ensembles and per-layer statistics are research OOD and shift detectors, and early layers are known to carry corruption shift. A per-corruption "which tap sees it first" profile was not found in the literature. In Atlas it is unconfirmed (gate closed). |
| ST-3 | partly-known | Rabanser, Günnemann & Lipton 2019, NeurIPS; Dong et al. 2022, CVPR (NMD); Kamoi & Kobayashi 2020; Wang et al. 2022, CVPR (ViM) | Batch mean-shift tests on representations and residual-subspace scores are research shift detectors. Rabanser et al. found that two-sample tests after reduction by a pretrained classifier work best, which makes a head-based detector the comparator. Atlas's T_perp was ruled out as registered. |
| ST-4 | partly-known | Kamoi & Kobayashi 2020 (arXiv:2003.00402); Wang et al. 2022, CVPR (ViM) | Anomaly signal in low-variance or residual directions is known for semantic OOD. Atlas's per-sample e_perp at a collapsed CIFAR penult did not carry that idea over to corruption drift (it loses to d1 in every run). |
| ST-8 | partly-known | Dong et al. 2022, CVPR (NMD); Schneider et al. 2020, NeurIPS; Nado et al. 2020 (arXiv:2006.10963) | Detecting an input-pipeline or global-intensity shift through activation-mean or BN-statistics discrepancy against training statistics is the standard feature-statistics approach. Atlas found the mis-normalisation from the training log, and never tested a label-free geometric detector for it. |
| LH-4 | partly-known | Yin et al. 2019, NeurIPS; Achille & Soatto 2018, JMLR | Brightness is a low-frequency, near-global nuisance that depth tends to make the representation invariant to; this is consistent with Atlas's penult luminance washout and the small brightness cost. Atlas's specific stem-direction mechanism was refuted. HOLD for brightness is supported only at depth-56 CIFAR. |
| LH-1 | known-in-research | Deng & Zheng 2021, CVPR (AutoEval, arXiv:2007.02915); Garg et al. 2022, ICLR (ATC, arXiv:2201.04234); Ginsberg, Liang & Krishnan (harmful covariate shift, arXiv:2212.02742) | Predicting accuracy on unlabeled shifted sets from feature statistics or confidence is an established research line, with average confidence and ATC as head baselines. Atlas's H was never compared with them. Primary classification belongs to the label-free-accuracy topic. |
| LH-2 | known-in-research | Deng & Zheng 2021, CVPR; Garg et al. 2022, ICLR; Rabanser et al. 2019, NeurIPS; Ginsberg, Liang & Krishnan (arXiv:2212.02742) | Batch-level shift and harm tests on representations or confidences are research practice. Separating harmful from benign shift is its own line of work. Head-based comparators are missing in Atlas. Primary classification belongs to the label-free-accuracy topic. |
| CD-1 | known-in-research | Ansuini, Laio, Macke & Zoccolan 2019, NeurIPS (arXiv:1905.12784) | The rise-then-fall intrinsic-dimension profile over depth, and the last-layer ID drop, are published. Atlas uses them as a known-answer instrument check (the code cites Ansuini). |
| CD-7 | known-in-research | Kornblith et al. 2019, ICML (arXiv:1905.00414); Nguyen, Raghu & Kornblith 2021, ICLR (arXiv:2010.15327) | CKA-based layer correspondence across seeds and depths, and seed-dependent block structure, are published analysis tools. Atlas's seed-pair CKA replicates them. No controller use. |
