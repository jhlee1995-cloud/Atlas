# ATLAS — margin_v1_resnet56_s12m
built 2026-09-23 14:30:23 · source **real** · arch `cifar10_resnet56` · weights `file:/workspace/models/resnet56_s12m_chenyaofo.pt sha256:71db3bad0a2c41c2` · layers 11

## 1. Per-layer invariants
| layer | D | PR | eff.rank | dim95 | TwoNN ID | sep ratio | NC1 | ETF dev | hub skew | test sparse | NC acc(test) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | 16 | · | · | · | · | · | · | · | · | · | · |
| layer1.0 | 16 | · | · | · | · | · | · | · | · | · | · |
| layer1.5 | 16 | · | · | · | · | · | · | · | · | · | · |
| layer1.8 | 16 | · | · | · | · | · | · | · | · | · | · |
| layer2.0 | 32 | · | · | · | · | · | · | · | · | · | · |
| layer2.5 | 32 | · | · | · | · | · | · | · | · | · | · |
| layer2.8 | 32 | · | · | · | · | · | · | · | · | · | · |
| layer3.0 | 64 | · | · | · | · | · | · | · | · | · | · |
| layer3.5 | 64 | · | · | · | · | · | · | · | · | · | · |
| layer3.8 | 64 | · | · | · | · | · | · | · | · | · | · |
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
| stem | 267 | 0.501 | 0.497 | 0.886 | 0.543 | 0.515 | 0.503 | 1.009 |
| layer1.0 | 267 | 0.473 | 0.494 | 0.886 | 0.547 | 0.499 | 0.475 | 1.078 |
| layer1.5 | 267 | 0.504 | 0.500 | 0.886 | 0.540 | 0.504 | 0.504 | 0.969 |
| layer1.8 | 267 | 0.519 | 0.512 | 0.886 | 0.540 | 0.507 | 0.519 | 0.902 |
| layer2.0 | 267 | 0.517 | 0.516 | 0.886 | 0.531 | 0.521 | 0.518 | 0.916 |
| layer2.5 | 267 | 0.551 | 0.524 | 0.886 | 0.516 | 0.565 | 0.552 | 0.803 |
| layer2.8 | 267 | 0.576 | 0.525 | 0.886 | 0.540 | 0.583 | 0.577 | 0.698 |
| layer3.0 | 267 | 0.601 | 0.531 | 0.886 | 0.560 | 0.610 | 0.604 | 0.706 |
| layer3.5 | 267 | 0.732 | 0.553 | 0.886 | 0.610 | 0.743 | 0.738 | 0.365 |
| layer3.8 | 267 | 0.887 | 0.856 | 0.886 | 0.717 | 0.920 | 0.904 | 0.321 |
| penult | 267 | 0.887 | 0.856 | 0.886 | 0.717 | 0.920 | 0.904 | 0.321 |

cut sweep at `penult` (type-b = wrong and maxprob > cut; oriented AUCs vs all correct)

| cut | n_typeb | margin | dist | maxprob | energy | margin (conf-matched) |
|---|---|---|---|---|---|---|
| 0.00 | 398 | 0.920 | 0.896 | 0.920 | 0.750 | 0.920 |
| 0.50 | 369 | 0.914 | 0.889 | 0.914 | 0.737 | 0.916 |
| 0.60 | 321 | 0.903 | 0.875 | 0.902 | 0.729 | 0.912 |
| 0.70 | 267 | 0.887 | 0.856 | 0.886 | 0.717 | 0.904 |
| 0.80 | 228 | 0.872 | 0.837 | 0.870 | 0.701 | 0.899 |
| 0.90 | 178 | 0.849 | 0.805 | 0.847 | 0.697 | 0.899 |
| 0.95 | 130 | 0.819 | 0.762 | 0.815 | 0.671 | 0.886 |
| 0.99 | 68 | 0.742 | 0.655 | 0.733 | 0.614 | 0.851 |

legacy (test-split centers, direction-free AUC, conf-wrong n=267): margin 0.886, cluster_subnet 0.817, energy 0.717
