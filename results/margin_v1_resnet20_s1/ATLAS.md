# ATLAS — margin_v1_resnet20_s1
built 2026-09-23 06:05:38 · source **real** · arch `cifar10_resnet20` · weights `file:/workspace/models/resnet20_s1_chenyaofo.pt sha256:d5442d0eadd72592` · layers 11

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
| stem | 273 | 0.476 | 0.490 | 0.880 | 0.561 | 0.478 | 0.476 | 1.072 |
| layer1.0 | 273 | 0.497 | 0.485 | 0.880 | 0.549 | 0.496 | 0.498 | 1.065 |
| layer1.1 | 273 | 0.514 | 0.486 | 0.880 | 0.532 | 0.501 | 0.515 | 0.926 |
| layer1.2 | 273 | 0.502 | 0.501 | 0.880 | 0.504 | 0.519 | 0.503 | 0.946 |
| layer2.0 | 273 | 0.514 | 0.540 | 0.880 | 0.520 | 0.528 | 0.515 | 0.935 |
| layer2.1 | 273 | 0.536 | 0.541 | 0.880 | 0.526 | 0.542 | 0.538 | 0.849 |
| layer2.2 | 273 | 0.556 | 0.544 | 0.880 | 0.540 | 0.568 | 0.559 | 0.842 |
| layer3.0 | 273 | 0.634 | 0.551 | 0.880 | 0.580 | 0.649 | 0.639 | 0.638 |
| layer3.1 | 273 | 0.752 | 0.528 | 0.880 | 0.734 | 0.766 | 0.759 | 0.391 |
| layer3.2 | 273 | 0.882 | 0.837 | 0.880 | 0.729 | 0.912 | 0.906 | 0.280 |
| penult | 273 | 0.882 | 0.837 | 0.880 | 0.729 | 0.912 | 0.906 | 0.280 |

cut sweep at `penult` (type-b = wrong and maxprob > cut; oriented AUCs vs all correct)

| cut | n_typeb | margin | dist | maxprob | energy | margin (conf-matched) |
|---|---|---|---|---|---|---|
| 0.00 | 387 | 0.912 | 0.875 | 0.912 | 0.756 | 0.912 |
| 0.50 | 358 | 0.906 | 0.867 | 0.905 | 0.742 | 0.910 |
| 0.60 | 317 | 0.895 | 0.854 | 0.894 | 0.739 | 0.910 |
| 0.70 | 273 | 0.882 | 0.837 | 0.880 | 0.729 | 0.906 |
| 0.80 | 217 | 0.861 | 0.807 | 0.858 | 0.709 | 0.895 |
| 0.90 | 155 | 0.825 | 0.762 | 0.821 | 0.689 | 0.878 |
| 0.95 | 128 | 0.803 | 0.732 | 0.798 | 0.671 | 0.874 |
| 0.99 | 67 | 0.722 | 0.630 | 0.713 | 0.567 | 0.841 |

legacy (test-split centers, direction-free AUC, conf-wrong n=273): margin 0.883, cluster_subnet 0.780, energy 0.729
