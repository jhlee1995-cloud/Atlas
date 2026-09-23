# ATLAS — margin_b1_resnet56_s0hub_st3
built 2026-09-23 14:31:29 · source **real** · arch `cifar10_resnet56` · weights `chenyaofo cifar10_resnet56` · layers 11

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
| stem | 215 | 0.499 | 0.469 | 0.887 | 0.563 | 0.502 | 0.499 | 1.052 |
| layer1.0 | 215 | 0.465 | 0.469 | 0.887 | 0.560 | 0.480 | 0.466 | 1.133 |
| layer1.5 | 215 | 0.497 | 0.487 | 0.887 | 0.560 | 0.499 | 0.498 | 1.034 |
| layer1.8 | 215 | 0.509 | 0.486 | 0.887 | 0.549 | 0.508 | 0.510 | 0.995 |
| layer2.0 | 215 | 0.513 | 0.501 | 0.887 | 0.499 | 0.509 | 0.515 | 0.899 |
| layer2.5 | 215 | 0.549 | 0.489 | 0.887 | 0.510 | 0.543 | 0.551 | 0.753 |
| layer2.8 | 215 | 0.579 | 0.496 | 0.887 | 0.522 | 0.573 | 0.582 | 0.712 |
| layer3.0 | 215 | 0.611 | 0.498 | 0.887 | 0.559 | 0.608 | 0.614 | 0.604 |
| layer3.5 | 215 | 0.788 | 0.565 | 0.887 | 0.685 | 0.796 | 0.793 | 0.332 |
| layer3.8 | 215 | 0.907 | 0.907 | 0.887 | 0.726 | 0.926 | 0.919 | 0.364 |
| penult | 215 | 0.907 | 0.907 | 0.887 | 0.726 | 0.926 | 0.919 | 0.364 |

cut sweep at `penult` (type-b = wrong and maxprob > cut; oriented AUCs vs all correct)

| cut | n_typeb | margin | dist | maxprob | energy | margin (conf-matched) |
|---|---|---|---|---|---|---|
| 0.00 | 277 | 0.926 | 0.926 | 0.911 | 0.762 | 0.926 |
| 0.50 | 260 | 0.922 | 0.922 | 0.906 | 0.751 | 0.923 |
| 0.60 | 236 | 0.914 | 0.915 | 0.896 | 0.738 | 0.920 |
| 0.70 | 215 | 0.907 | 0.907 | 0.887 | 0.726 | 0.919 |
| 0.80 | 195 | 0.899 | 0.900 | 0.877 | 0.708 | 0.916 |
| 0.90 | 145 | 0.872 | 0.874 | 0.843 | 0.680 | 0.899 |
| 0.95 | 127 | 0.859 | 0.861 | 0.826 | 0.671 | 0.896 |
| 0.99 | 80 | 0.811 | 0.816 | 0.757 | 0.597 | 0.876 |

legacy (test-split centers, direction-free AUC, conf-wrong n=215): margin 0.909, cluster_subnet 0.895, energy 0.726
