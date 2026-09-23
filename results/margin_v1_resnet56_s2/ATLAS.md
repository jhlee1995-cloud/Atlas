# ATLAS — margin_v1_resnet56_s2
built 2026-09-23 14:29:36 · source **real** · arch `cifar10_resnet56` · weights `file:/workspace/models/resnet56_s2_chenyaofo.pt sha256:14a651f000ca2d55` · layers 11

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
| stem | 229 | 0.505 | 0.459 | 0.892 | 0.553 | 0.510 | 0.505 | 1.077 |
| layer1.0 | 229 | 0.521 | 0.460 | 0.892 | 0.549 | 0.520 | 0.521 | 0.925 |
| layer1.5 | 229 | 0.513 | 0.508 | 0.892 | 0.520 | 0.527 | 0.513 | 0.982 |
| layer1.8 | 229 | 0.479 | 0.513 | 0.892 | 0.510 | 0.502 | 0.479 | 1.064 |
| layer2.0 | 229 | 0.502 | 0.527 | 0.892 | 0.472 | 0.514 | 0.503 | 1.014 |
| layer2.5 | 229 | 0.538 | 0.531 | 0.892 | 0.478 | 0.548 | 0.539 | 0.818 |
| layer2.8 | 229 | 0.555 | 0.537 | 0.892 | 0.497 | 0.569 | 0.557 | 0.743 |
| layer3.0 | 229 | 0.591 | 0.549 | 0.892 | 0.514 | 0.605 | 0.592 | 0.706 |
| layer3.5 | 229 | 0.774 | 0.602 | 0.892 | 0.682 | 0.786 | 0.778 | 0.327 |
| layer3.8 | 229 | 0.904 | 0.903 | 0.892 | 0.716 | 0.925 | 0.914 | 0.348 |
| penult | 229 | 0.904 | 0.903 | 0.892 | 0.716 | 0.925 | 0.914 | 0.348 |

cut sweep at `penult` (type-b = wrong and maxprob > cut; oriented AUCs vs all correct)

| cut | n_typeb | margin | dist | maxprob | energy | margin (conf-matched) |
|---|---|---|---|---|---|---|
| 0.00 | 298 | 0.925 | 0.924 | 0.916 | 0.750 | 0.925 |
| 0.50 | 283 | 0.921 | 0.920 | 0.911 | 0.739 | 0.922 |
| 0.60 | 254 | 0.912 | 0.912 | 0.902 | 0.726 | 0.918 |
| 0.70 | 229 | 0.904 | 0.903 | 0.892 | 0.716 | 0.914 |
| 0.80 | 202 | 0.893 | 0.893 | 0.879 | 0.694 | 0.909 |
| 0.90 | 160 | 0.871 | 0.871 | 0.854 | 0.672 | 0.898 |
| 0.95 | 132 | 0.851 | 0.851 | 0.831 | 0.657 | 0.888 |
| 0.99 | 87 | 0.806 | 0.807 | 0.775 | 0.570 | 0.876 |

legacy (test-split centers, direction-free AUC, conf-wrong n=229): margin 0.905, cluster_subnet 0.883, energy 0.716
