# ATLAS — margin_b1_deitb_swap
built 2026-09-23 17:31:21 · source **real** · arch `timm_deit_base_patch16_224` · weights `timm timm_deit_base_patch16_224 fb_in1k file:/workspace/.cache/huggingface/hub/models--timm--deit_base_patch16_224.fb_in1k/snapshots/b78cc5532a69df6bcad9c3a8d76653fd20b31ac6/model.safetensors sha256:cd2da27b74ed7f68b599f16c77af3e1e80f01c75f9ad96029d22ce747a247e8e` · layers 14

## 1. Per-layer invariants
| layer | D | PR | eff.rank | dim95 | TwoNN ID | sep ratio | NC1 | ETF dev | hub skew | test sparse | NC acc(test) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| block.0 | 768 | · | · | · | · | · | · | · | · | · | · |
| block.1 | 768 | · | · | · | · | · | · | · | · | · | · |
| block.2 | 768 | · | · | · | · | · | · | · | · | · | · |
| block.3 | 768 | · | · | · | · | · | · | · | · | · | · |
| block.4 | 768 | · | · | · | · | · | · | · | · | · | · |
| block.5 | 768 | · | · | · | · | · | · | · | · | · | · |
| block.6 | 768 | · | · | · | · | · | · | · | · | · | · |
| block.7 | 768 | · | · | · | · | · | · | · | · | · | · |
| block.8 | 768 | · | · | · | · | · | · | · | · | · | · |
| block.9 | 768 | · | · | · | · | · | · | · | · | · | · |
| block.10 | 768 | · | · | · | · | · | · | · | · | · | · |
| block.11 | 768 | · | · | · | · | · | · | · | · | · | · |
| penult_mean | 768 | · | · | · | · | · | · | · | · | · | · |
| penult | 768 | · | · | · | · | · | · | · | · | · | · |

## 2. Decodability profile (which factor lives where)
_linear_probes not run_

## 3. Layer flow (CKA)
_layer_cka not run_

## 4. Sensitivity field (paired corruption displacement)
_corruption_displacement not run_

## 5. Density per split
_knn_density not run_

## 6. Adjacency
layer `penult`


## 7. Type-b margin (margin_typeb)
| layer | n_typeb | AUC margin | AUC dist | AUC maxprob | AUC energy | AUC margin (all wrong) | AUC margin (conf-matched) | median ratio type-b/correct |
|---|---|---|---|---|---|---|---|---|
| block.0 | 1303 | 0.500 | 0.508 | 0.593 | 0.526 | 0.507 | 0.501 | 0.975 |
| block.1 | 1303 | 0.501 | 0.515 | 0.593 | 0.507 | 0.507 | 0.502 | 0.976 |
| block.2 | 1303 | 0.499 | 0.516 | 0.593 | 0.493 | 0.508 | 0.504 | 1.022 |
| block.3 | 1303 | 0.499 | 0.515 | 0.593 | 0.496 | 0.512 | 0.503 | 1.036 |
| block.4 | 1303 | 0.506 | 0.517 | 0.593 | 0.513 | 0.518 | 0.510 | 0.970 |
| block.5 | 1303 | 0.510 | 0.521 | 0.593 | 0.512 | 0.523 | 0.517 | 0.971 |
| block.6 | 1303 | 0.528 | 0.526 | 0.593 | 0.513 | 0.539 | 0.536 | 0.914 |
| block.7 | 1303 | 0.548 | 0.532 | 0.593 | 0.527 | 0.567 | 0.561 | 0.842 |
| block.8 | 1303 | 0.580 | 0.538 | 0.593 | 0.544 | 0.606 | 0.601 | 0.727 |
| block.9 | 1303 | 0.637 | 0.539 | 0.593 | 0.583 | 0.704 | 0.680 | 0.503 |
| block.10 | 1303 | 0.713 | 0.525 | 0.593 | 0.643 | 0.820 | 0.791 | 0.329 |
| block.11 | 1303 | 0.713 | 0.562 | 0.593 | 0.653 | 0.818 | 0.787 | 0.348 |
| penult_mean | 1303 | 0.733 | 0.647 | 0.593 | 0.440 | 0.841 | 0.811 | 0.373 |
| penult | 1303 | 0.739 | 0.645 | 0.593 | 0.372 | 0.861 | 0.829 | 0.404 |

cut sweep at `penult` (type-b = wrong and maxprob > cut; oriented AUCs vs all correct)

| cut | n_typeb | margin | dist | maxprob | energy | margin (conf-matched) |
|---|---|---|---|---|---|---|
| 0.00 | 5030 | 0.861 | 0.772 | 0.848 | 0.249 | 0.861 |
| 0.50 | 2626 | 0.807 | 0.689 | 0.735 | 0.333 | 0.846 |
| 0.60 | 1915 | 0.779 | 0.668 | 0.674 | 0.354 | 0.840 |
| 0.70 | 1303 | 0.739 | 0.645 | 0.593 | 0.372 | 0.829 |
| 0.80 | 681 | 0.674 | 0.600 | 0.440 | 0.412 | 0.813 |
| 0.90 | 74 | 0.651 | 0.673 | 0.048 | 0.320 | 0.768 |
| 0.95 | 3 | · | · | · | · | · |
| 0.99 | 0 | · | · | · | · | · |

legacy (test-split centers, direction-free AUC, conf-wrong n=1303): margin 0.816, cluster_subnet 0.666, energy 0.628
