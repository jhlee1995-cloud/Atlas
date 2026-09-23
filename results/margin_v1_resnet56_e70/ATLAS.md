# ATLAS — margin_v1_resnet56_e70
built 2026-09-23 14:30:11 · source **real** · arch `cifar10_resnet56` · weights `file:/workspace/models/resnet56_s11_e70_chenyaofo.pt sha256:32e0876cd7a22a27` · layers 11

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
| stem | 280 | 0.514 | 0.494 | 0.887 | 0.533 | 0.504 | 0.515 | 0.967 |
| layer1.0 | 280 | 0.483 | 0.493 | 0.887 | 0.545 | 0.485 | 0.482 | 1.001 |
| layer1.5 | 280 | 0.506 | 0.520 | 0.887 | 0.523 | 0.507 | 0.507 | 0.942 |
| layer1.8 | 280 | 0.547 | 0.534 | 0.887 | 0.516 | 0.530 | 0.548 | 0.811 |
| layer2.0 | 280 | 0.544 | 0.531 | 0.887 | 0.538 | 0.528 | 0.545 | 0.799 |
| layer2.5 | 280 | 0.571 | 0.526 | 0.887 | 0.582 | 0.568 | 0.572 | 0.727 |
| layer2.8 | 280 | 0.601 | 0.540 | 0.887 | 0.594 | 0.591 | 0.602 | 0.630 |
| layer3.0 | 280 | 0.653 | 0.549 | 0.887 | 0.593 | 0.644 | 0.655 | 0.490 |
| layer3.5 | 280 | 0.832 | 0.608 | 0.887 | 0.738 | 0.836 | 0.838 | 0.284 |
| layer3.8 | 280 | 0.891 | 0.869 | 0.887 | 0.724 | 0.916 | 0.904 | 0.310 |
| penult | 280 | 0.891 | 0.869 | 0.887 | 0.724 | 0.916 | 0.904 | 0.310 |

cut sweep at `penult` (type-b = wrong and maxprob > cut; oriented AUCs vs all correct)

| cut | n_typeb | margin | dist | maxprob | energy | margin (conf-matched) |
|---|---|---|---|---|---|---|
| 0.00 | 376 | 0.916 | 0.898 | 0.915 | 0.754 | 0.916 |
| 0.50 | 351 | 0.911 | 0.892 | 0.909 | 0.743 | 0.914 |
| 0.60 | 312 | 0.901 | 0.881 | 0.898 | 0.737 | 0.908 |
| 0.70 | 280 | 0.891 | 0.869 | 0.887 | 0.724 | 0.904 |
| 0.80 | 240 | 0.876 | 0.853 | 0.872 | 0.712 | 0.899 |
| 0.90 | 184 | 0.849 | 0.820 | 0.844 | 0.679 | 0.888 |
| 0.95 | 146 | 0.824 | 0.791 | 0.817 | 0.648 | 0.879 |
| 0.99 | 82 | 0.755 | 0.718 | 0.742 | 0.569 | 0.850 |

legacy (test-split centers, direction-free AUC, conf-wrong n=280): margin 0.892, cluster_subnet 0.833, energy 0.724
