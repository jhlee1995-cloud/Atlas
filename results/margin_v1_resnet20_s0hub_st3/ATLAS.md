# ATLAS — margin_v1_resnet20_s0hub_st3
built 2026-09-23 14:28:25 · source **real** · arch `cifar10_resnet20` · weights `chenyaofo cifar10_resnet20` · layers 11

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
| stem | 251 | 0.494 | 0.488 | 0.884 | 0.545 | 0.476 | 0.493 | 1.073 |
| layer1.0 | 251 | 0.537 | 0.501 | 0.884 | 0.545 | 0.524 | 0.537 | 0.806 |
| layer1.1 | 251 | 0.508 | 0.504 | 0.884 | 0.529 | 0.511 | 0.509 | 0.917 |
| layer1.2 | 251 | 0.498 | 0.512 | 0.884 | 0.517 | 0.517 | 0.499 | 0.950 |
| layer2.0 | 251 | 0.533 | 0.531 | 0.884 | 0.492 | 0.541 | 0.534 | 0.869 |
| layer2.1 | 251 | 0.566 | 0.530 | 0.884 | 0.485 | 0.565 | 0.568 | 0.752 |
| layer2.2 | 251 | 0.606 | 0.530 | 0.884 | 0.505 | 0.599 | 0.608 | 0.635 |
| layer3.0 | 251 | 0.659 | 0.537 | 0.884 | 0.597 | 0.650 | 0.663 | 0.495 |
| layer3.1 | 251 | 0.749 | 0.527 | 0.884 | 0.682 | 0.754 | 0.757 | 0.380 |
| layer3.2 | 251 | 0.885 | 0.814 | 0.884 | 0.766 | 0.917 | 0.908 | 0.282 |
| penult | 251 | 0.885 | 0.814 | 0.884 | 0.766 | 0.917 | 0.908 | 0.282 |

cut sweep at `penult` (type-b = wrong and maxprob > cut; oriented AUCs vs all correct)

| cut | n_typeb | margin | dist | maxprob | energy | margin (conf-matched) |
|---|---|---|---|---|---|---|
| 0.00 | 372 | 0.917 | 0.864 | 0.918 | 0.795 | 0.917 |
| 0.50 | 338 | 0.910 | 0.852 | 0.910 | 0.787 | 0.914 |
| 0.60 | 296 | 0.899 | 0.836 | 0.899 | 0.781 | 0.913 |
| 0.70 | 251 | 0.885 | 0.814 | 0.884 | 0.766 | 0.908 |
| 0.80 | 199 | 0.864 | 0.782 | 0.863 | 0.754 | 0.900 |
| 0.90 | 150 | 0.837 | 0.742 | 0.836 | 0.734 | 0.895 |
| 0.95 | 105 | 0.802 | 0.684 | 0.800 | 0.716 | 0.879 |
| 0.99 | 61 | 0.745 | 0.599 | 0.742 | 0.669 | 0.878 |

legacy (test-split centers, direction-free AUC, conf-wrong n=251): margin 0.884, cluster_subnet 0.741, energy 0.766
