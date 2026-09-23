# ATLAS — margin_v1_resnet20_s4_st3
built 2026-09-23 14:29:05 · source **real** · arch `cifar10_resnet20` · weights `file:/workspace/models/resnet20_s4_chenyaofo.pt sha256:a67852e591b4618e` · layers 11

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
| stem | 254 | 0.474 | 0.477 | 0.899 | 0.556 | 0.487 | 0.475 | 1.084 |
| layer1.0 | 254 | 0.515 | 0.487 | 0.899 | 0.568 | 0.514 | 0.516 | 0.962 |
| layer1.1 | 254 | 0.547 | 0.494 | 0.899 | 0.551 | 0.549 | 0.549 | 0.873 |
| layer1.2 | 254 | 0.527 | 0.502 | 0.899 | 0.545 | 0.532 | 0.528 | 0.844 |
| layer2.0 | 254 | 0.523 | 0.514 | 0.899 | 0.505 | 0.533 | 0.524 | 0.899 |
| layer2.1 | 254 | 0.536 | 0.515 | 0.899 | 0.518 | 0.550 | 0.537 | 0.854 |
| layer2.2 | 254 | 0.552 | 0.521 | 0.899 | 0.533 | 0.565 | 0.555 | 0.783 |
| layer3.0 | 254 | 0.615 | 0.528 | 0.899 | 0.586 | 0.627 | 0.619 | 0.594 |
| layer3.1 | 254 | 0.763 | 0.534 | 0.899 | 0.674 | 0.774 | 0.770 | 0.358 |
| layer3.2 | 254 | 0.899 | 0.845 | 0.899 | 0.771 | 0.925 | 0.918 | 0.266 |
| penult | 254 | 0.899 | 0.845 | 0.899 | 0.771 | 0.925 | 0.918 | 0.266 |

cut sweep at `penult` (type-b = wrong and maxprob > cut; oriented AUCs vs all correct)

| cut | n_typeb | margin | dist | maxprob | energy | margin (conf-matched) |
|---|---|---|---|---|---|---|
| 0.00 | 357 | 0.925 | 0.880 | 0.926 | 0.791 | 0.925 |
| 0.50 | 325 | 0.918 | 0.869 | 0.919 | 0.780 | 0.920 |
| 0.60 | 289 | 0.910 | 0.859 | 0.909 | 0.777 | 0.918 |
| 0.70 | 254 | 0.899 | 0.845 | 0.899 | 0.771 | 0.918 |
| 0.80 | 213 | 0.886 | 0.825 | 0.885 | 0.764 | 0.914 |
| 0.90 | 146 | 0.853 | 0.780 | 0.851 | 0.729 | 0.901 |
| 0.95 | 105 | 0.821 | 0.735 | 0.819 | 0.705 | 0.889 |
| 0.99 | 50 | 0.755 | 0.670 | 0.741 | 0.620 | 0.877 |

legacy (test-split centers, direction-free AUC, conf-wrong n=254): margin 0.899, cluster_subnet 0.779, energy 0.771
