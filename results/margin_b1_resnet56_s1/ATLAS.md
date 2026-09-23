# ATLAS — margin_b1_resnet56_s1
built 2026-09-23 14:31:42 · source **real** · arch `cifar10_resnet56` · weights `file:/workspace/models/resnet56_s1_chenyaofo.pt sha256:73c9458c60b66949` · layers 11

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
| stem | 214 | 0.507 | 0.458 | 0.901 | 0.553 | 0.501 | 0.507 | 0.968 |
| layer1.0 | 214 | 0.502 | 0.457 | 0.901 | 0.554 | 0.500 | 0.502 | 1.043 |
| layer1.5 | 214 | 0.478 | 0.464 | 0.901 | 0.528 | 0.489 | 0.477 | 1.078 |
| layer1.8 | 214 | 0.479 | 0.477 | 0.901 | 0.516 | 0.492 | 0.479 | 1.114 |
| layer2.0 | 214 | 0.511 | 0.480 | 0.901 | 0.547 | 0.518 | 0.512 | 0.961 |
| layer2.5 | 214 | 0.524 | 0.491 | 0.901 | 0.541 | 0.535 | 0.525 | 0.852 |
| layer2.8 | 214 | 0.547 | 0.490 | 0.901 | 0.559 | 0.561 | 0.547 | 0.814 |
| layer3.0 | 214 | 0.605 | 0.489 | 0.901 | 0.594 | 0.609 | 0.606 | 0.646 |
| layer3.5 | 214 | 0.757 | 0.541 | 0.901 | 0.686 | 0.770 | 0.761 | 0.359 |
| layer3.8 | 214 | 0.908 | 0.899 | 0.901 | 0.734 | 0.928 | 0.918 | 0.362 |
| penult | 214 | 0.908 | 0.899 | 0.901 | 0.734 | 0.928 | 0.918 | 0.362 |

cut sweep at `penult` (type-b = wrong and maxprob > cut; oriented AUCs vs all correct)

| cut | n_typeb | margin | dist | maxprob | energy | margin (conf-matched) |
|---|---|---|---|---|---|---|
| 0.00 | 281 | 0.928 | 0.921 | 0.924 | 0.767 | 0.928 |
| 0.50 | 269 | 0.925 | 0.918 | 0.920 | 0.758 | 0.927 |
| 0.60 | 233 | 0.914 | 0.906 | 0.909 | 0.746 | 0.920 |
| 0.70 | 214 | 0.908 | 0.899 | 0.901 | 0.734 | 0.918 |
| 0.80 | 191 | 0.898 | 0.889 | 0.891 | 0.726 | 0.914 |
| 0.90 | 160 | 0.884 | 0.874 | 0.875 | 0.708 | 0.912 |
| 0.95 | 131 | 0.867 | 0.855 | 0.856 | 0.675 | 0.909 |
| 0.99 | 73 | 0.814 | 0.793 | 0.794 | 0.585 | 0.889 |

legacy (test-split centers, direction-free AUC, conf-wrong n=214): margin 0.905, cluster_subnet 0.881, energy 0.734
