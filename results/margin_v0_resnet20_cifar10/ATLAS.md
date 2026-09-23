# ATLAS — margin_v0_resnet20_cifar10
built 2026-09-23 06:05:00 · source **real** · arch `cifar10_resnet20` · weights `chenyaofo cifar10_resnet20` · layers 11

## 1. Per-layer invariants
| layer | D | PR | eff.rank | dim95 | TwoNN ID | sep ratio | NC1 | ETF dev | hub skew | test sparse | NC acc(test) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | 16 | · | · | · | · | · | · | · | · | · | · |
| layer1.0 | 16 | · | · | · | · | · | · | · | · | · | · |
| layer1.1 | 16 | · | · | · | · | · | · | · | · | · | · |
| layer1.2 | 16 | · | · | · | · | · | · | · | · | · | · |
| layer2.0 | 32 | · | · | · | · | · | · | · | · | · | · |
| layer2.1 | 32 | · | · | · | · | · | · | · | · | · | · |
| layer2.2 | 32 | · | · | · | · | · | · | · | · | · | · |
| layer3.0 | 64 | · | · | · | · | · | · | · | · | · | · |
| layer3.1 | 64 | · | · | · | · | · | · | · | · | · | · |
| layer3.2 | 64 | · | · | · | · | · | · | · | · | · | · |
| penult | 64 | · | · | · | · | · | · | · | · | · | · |

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
| stem | 260 | 0.471 | 0.502 | 0.888 | 0.540 | 0.470 | 0.471 | 1.179 |
| layer1.0 | 260 | 0.516 | 0.511 | 0.888 | 0.548 | 0.524 | 0.516 | 0.937 |
| layer1.1 | 260 | 0.495 | 0.507 | 0.888 | 0.531 | 0.508 | 0.496 | 1.039 |
| layer1.2 | 260 | 0.497 | 0.504 | 0.888 | 0.520 | 0.511 | 0.498 | 0.991 |
| layer2.0 | 260 | 0.524 | 0.519 | 0.888 | 0.493 | 0.541 | 0.526 | 0.933 |
| layer2.1 | 260 | 0.562 | 0.521 | 0.888 | 0.483 | 0.565 | 0.563 | 0.771 |
| layer2.2 | 260 | 0.583 | 0.521 | 0.888 | 0.503 | 0.595 | 0.585 | 0.731 |
| layer3.0 | 260 | 0.634 | 0.528 | 0.888 | 0.600 | 0.644 | 0.638 | 0.542 |
| layer3.1 | 260 | 0.768 | 0.523 | 0.888 | 0.685 | 0.769 | 0.776 | 0.370 |
| layer3.2 | 260 | 0.894 | 0.828 | 0.888 | 0.760 | 0.927 | 0.916 | 0.255 |
| penult | 260 | 0.894 | 0.828 | 0.888 | 0.760 | 0.927 | 0.916 | 0.255 |

cut sweep at `penult` (type-b = wrong and maxprob > cut; oriented AUCs vs all correct)

| cut | n_typeb | margin | dist | maxprob | energy | margin (conf-matched) |
|---|---|---|---|---|---|---|
| 0.00 | 408 | 0.927 | 0.878 | 0.924 | 0.792 | 0.927 |
| 0.50 | 367 | 0.920 | 0.867 | 0.916 | 0.778 | 0.924 |
| 0.60 | 321 | 0.910 | 0.853 | 0.905 | 0.774 | 0.923 |
| 0.70 | 260 | 0.894 | 0.828 | 0.888 | 0.760 | 0.916 |
| 0.80 | 214 | 0.879 | 0.805 | 0.870 | 0.747 | 0.911 |
| 0.90 | 141 | 0.841 | 0.746 | 0.829 | 0.701 | 0.897 |
| 0.95 | 106 | 0.813 | 0.700 | 0.799 | 0.684 | 0.892 |
| 0.99 | 53 | 0.736 | 0.590 | 0.720 | 0.629 | 0.868 |

legacy (test-split centers, direction-free AUC, conf-wrong n=260): margin 0.893, cluster_subnet 0.756, energy 0.760
