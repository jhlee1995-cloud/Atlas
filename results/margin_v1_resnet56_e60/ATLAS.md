# ATLAS — margin_v1_resnet56_e60
built 2026-09-23 14:30:00 · source **real** · arch `cifar10_resnet56` · weights `file:/workspace/models/resnet56_s11_e60_chenyaofo.pt sha256:d8f65a2fe8d7a833` · layers 11

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
| stem | 267 | 0.488 | 0.456 | 0.885 | 0.543 | 0.505 | 0.490 | 1.068 |
| layer1.0 | 267 | 0.505 | 0.467 | 0.885 | 0.556 | 0.508 | 0.506 | 1.021 |
| layer1.5 | 267 | 0.499 | 0.482 | 0.885 | 0.551 | 0.507 | 0.500 | 0.991 |
| layer1.8 | 267 | 0.518 | 0.495 | 0.885 | 0.521 | 0.530 | 0.519 | 0.895 |
| layer2.0 | 267 | 0.519 | 0.518 | 0.885 | 0.520 | 0.540 | 0.520 | 0.921 |
| layer2.5 | 267 | 0.562 | 0.529 | 0.885 | 0.511 | 0.577 | 0.564 | 0.789 |
| layer2.8 | 267 | 0.580 | 0.540 | 0.885 | 0.481 | 0.591 | 0.582 | 0.756 |
| layer3.0 | 267 | 0.590 | 0.538 | 0.885 | 0.520 | 0.608 | 0.592 | 0.690 |
| layer3.5 | 267 | 0.736 | 0.570 | 0.885 | 0.596 | 0.750 | 0.742 | 0.397 |
| layer3.8 | 267 | 0.887 | 0.855 | 0.885 | 0.692 | 0.917 | 0.901 | 0.294 |
| penult | 267 | 0.887 | 0.855 | 0.885 | 0.692 | 0.917 | 0.901 | 0.294 |

cut sweep at `penult` (type-b = wrong and maxprob > cut; oriented AUCs vs all correct)

| cut | n_typeb | margin | dist | maxprob | energy | margin (conf-matched) |
|---|---|---|---|---|---|---|
| 0.00 | 378 | 0.917 | 0.893 | 0.917 | 0.717 | 0.917 |
| 0.50 | 352 | 0.911 | 0.885 | 0.910 | 0.707 | 0.913 |
| 0.60 | 306 | 0.899 | 0.871 | 0.898 | 0.702 | 0.906 |
| 0.70 | 267 | 0.887 | 0.855 | 0.885 | 0.692 | 0.901 |
| 0.80 | 217 | 0.866 | 0.833 | 0.863 | 0.670 | 0.891 |
| 0.90 | 167 | 0.838 | 0.797 | 0.834 | 0.653 | 0.881 |
| 0.95 | 123 | 0.801 | 0.754 | 0.796 | 0.628 | 0.863 |
| 0.99 | 70 | 0.732 | 0.662 | 0.720 | 0.582 | 0.843 |

legacy (test-split centers, direction-free AUC, conf-wrong n=267): margin 0.886, cluster_subnet 0.819, energy 0.692
