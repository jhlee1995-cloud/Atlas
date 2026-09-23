# ATLAS — margin_v1_resnet20_s3_st3
built 2026-09-23 14:28:55 · source **real** · arch `cifar10_resnet20` · weights `file:/workspace/models/resnet20_s3_chenyaofo.pt sha256:9b06806321d39155` · layers 11

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
| stem | 254 | 0.512 | 0.501 | 0.890 | 0.559 | 0.493 | 0.513 | 0.942 |
| layer1.0 | 254 | 0.478 | 0.489 | 0.890 | 0.552 | 0.494 | 0.479 | 1.022 |
| layer1.1 | 254 | 0.492 | 0.495 | 0.890 | 0.558 | 0.512 | 0.493 | 1.024 |
| layer1.2 | 254 | 0.513 | 0.500 | 0.890 | 0.571 | 0.520 | 0.512 | 0.965 |
| layer2.0 | 254 | 0.512 | 0.518 | 0.890 | 0.516 | 0.527 | 0.512 | 0.973 |
| layer2.1 | 254 | 0.534 | 0.509 | 0.890 | 0.538 | 0.545 | 0.534 | 0.873 |
| layer2.2 | 254 | 0.563 | 0.512 | 0.890 | 0.543 | 0.573 | 0.564 | 0.718 |
| layer3.0 | 254 | 0.641 | 0.513 | 0.890 | 0.599 | 0.653 | 0.643 | 0.581 |
| layer3.1 | 254 | 0.751 | 0.527 | 0.890 | 0.686 | 0.764 | 0.755 | 0.359 |
| layer3.2 | 254 | 0.890 | 0.827 | 0.890 | 0.744 | 0.923 | 0.908 | 0.285 |
| penult | 254 | 0.890 | 0.827 | 0.890 | 0.744 | 0.923 | 0.908 | 0.285 |

cut sweep at `penult` (type-b = wrong and maxprob > cut; oriented AUCs vs all correct)

| cut | n_typeb | margin | dist | maxprob | energy | margin (conf-matched) |
|---|---|---|---|---|---|---|
| 0.00 | 389 | 0.923 | 0.874 | 0.924 | 0.780 | 0.923 |
| 0.50 | 358 | 0.917 | 0.865 | 0.918 | 0.768 | 0.921 |
| 0.60 | 315 | 0.908 | 0.852 | 0.908 | 0.761 | 0.917 |
| 0.70 | 254 | 0.890 | 0.827 | 0.890 | 0.744 | 0.908 |
| 0.80 | 206 | 0.871 | 0.800 | 0.871 | 0.739 | 0.901 |
| 0.90 | 152 | 0.842 | 0.760 | 0.842 | 0.703 | 0.891 |
| 0.95 | 118 | 0.817 | 0.728 | 0.816 | 0.688 | 0.888 |
| 0.99 | 62 | 0.753 | 0.654 | 0.748 | 0.634 | 0.884 |

legacy (test-split centers, direction-free AUC, conf-wrong n=254): margin 0.890, cluster_subnet 0.763, energy 0.744
