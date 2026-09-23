# ATLAS — margin_v1_resnet56_e50
built 2026-09-23 14:29:48 · source **real** · arch `cifar10_resnet56` · weights `file:/workspace/models/resnet56_s11_e50_chenyaofo.pt sha256:b135fdac0dbe43be` · layers 11

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
| stem | 271 | 0.497 | 0.469 | 0.890 | 0.552 | 0.505 | 0.498 | 1.013 |
| layer1.0 | 271 | 0.513 | 0.479 | 0.890 | 0.554 | 0.509 | 0.514 | 0.939 |
| layer1.5 | 271 | 0.536 | 0.482 | 0.890 | 0.556 | 0.517 | 0.537 | 0.902 |
| layer1.8 | 271 | 0.503 | 0.494 | 0.890 | 0.540 | 0.498 | 0.504 | 0.953 |
| layer2.0 | 271 | 0.508 | 0.514 | 0.890 | 0.517 | 0.513 | 0.510 | 1.007 |
| layer2.5 | 271 | 0.546 | 0.518 | 0.890 | 0.529 | 0.552 | 0.547 | 0.760 |
| layer2.8 | 271 | 0.560 | 0.522 | 0.890 | 0.531 | 0.568 | 0.561 | 0.760 |
| layer3.0 | 271 | 0.585 | 0.531 | 0.890 | 0.539 | 0.596 | 0.587 | 0.742 |
| layer3.5 | 271 | 0.733 | 0.539 | 0.890 | 0.682 | 0.767 | 0.740 | 0.386 |
| layer3.8 | 271 | 0.889 | 0.854 | 0.890 | 0.751 | 0.921 | 0.908 | 0.321 |
| penult | 271 | 0.889 | 0.854 | 0.890 | 0.751 | 0.921 | 0.908 | 0.321 |

cut sweep at `penult` (type-b = wrong and maxprob > cut; oriented AUCs vs all correct)

| cut | n_typeb | margin | dist | maxprob | energy | margin (conf-matched) |
|---|---|---|---|---|---|---|
| 0.00 | 403 | 0.921 | 0.895 | 0.923 | 0.786 | 0.921 |
| 0.50 | 374 | 0.916 | 0.887 | 0.917 | 0.778 | 0.919 |
| 0.60 | 325 | 0.904 | 0.873 | 0.906 | 0.764 | 0.914 |
| 0.70 | 271 | 0.889 | 0.854 | 0.890 | 0.751 | 0.908 |
| 0.80 | 230 | 0.874 | 0.836 | 0.876 | 0.735 | 0.904 |
| 0.90 | 163 | 0.840 | 0.791 | 0.843 | 0.724 | 0.888 |
| 0.95 | 126 | 0.813 | 0.756 | 0.817 | 0.703 | 0.882 |
| 0.99 | 58 | 0.724 | 0.632 | 0.728 | 0.627 | 0.846 |

legacy (test-split centers, direction-free AUC, conf-wrong n=271): margin 0.887, cluster_subnet 0.811, energy 0.751
