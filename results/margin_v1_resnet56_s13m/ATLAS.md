# ATLAS — margin_v1_resnet56_s13m
built 2026-09-23 14:30:33 · source **real** · arch `cifar10_resnet56` · weights `file:/workspace/models/resnet56_s13m_chenyaofo.pt sha256:0df3f708ebfa3292` · layers 11

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
| stem | 269 | 0.505 | 0.533 | 0.891 | 0.521 | 0.513 | 0.506 | 1.033 |
| layer1.0 | 269 | 0.501 | 0.534 | 0.891 | 0.522 | 0.507 | 0.502 | 1.066 |
| layer1.5 | 269 | 0.518 | 0.520 | 0.891 | 0.545 | 0.513 | 0.518 | 1.002 |
| layer1.8 | 269 | 0.510 | 0.504 | 0.891 | 0.531 | 0.514 | 0.511 | 0.932 |
| layer2.0 | 269 | 0.514 | 0.514 | 0.891 | 0.542 | 0.516 | 0.515 | 0.958 |
| layer2.5 | 269 | 0.539 | 0.515 | 0.891 | 0.572 | 0.544 | 0.541 | 0.834 |
| layer2.8 | 269 | 0.543 | 0.524 | 0.891 | 0.570 | 0.550 | 0.546 | 0.777 |
| layer3.0 | 269 | 0.584 | 0.530 | 0.891 | 0.530 | 0.591 | 0.588 | 0.696 |
| layer3.5 | 269 | 0.725 | 0.577 | 0.891 | 0.568 | 0.740 | 0.732 | 0.426 |
| layer3.8 | 269 | 0.895 | 0.873 | 0.891 | 0.710 | 0.922 | 0.916 | 0.303 |
| penult | 269 | 0.895 | 0.873 | 0.891 | 0.710 | 0.922 | 0.916 | 0.303 |

cut sweep at `penult` (type-b = wrong and maxprob > cut; oriented AUCs vs all correct)

| cut | n_typeb | margin | dist | maxprob | energy | margin (conf-matched) |
|---|---|---|---|---|---|---|
| 0.00 | 379 | 0.922 | 0.905 | 0.920 | 0.748 | 0.922 |
| 0.50 | 351 | 0.917 | 0.898 | 0.914 | 0.733 | 0.919 |
| 0.60 | 307 | 0.906 | 0.886 | 0.903 | 0.724 | 0.916 |
| 0.70 | 269 | 0.895 | 0.873 | 0.891 | 0.710 | 0.916 |
| 0.80 | 233 | 0.883 | 0.860 | 0.879 | 0.693 | 0.915 |
| 0.90 | 180 | 0.862 | 0.833 | 0.856 | 0.689 | 0.910 |
| 0.95 | 132 | 0.835 | 0.799 | 0.827 | 0.669 | 0.901 |
| 0.99 | 67 | 0.773 | 0.733 | 0.753 | 0.583 | 0.883 |

legacy (test-split centers, direction-free AUC, conf-wrong n=269): margin 0.895, cluster_subnet 0.837, energy 0.710
