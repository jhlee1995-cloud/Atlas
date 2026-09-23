# ATLAS — margin_v1_resnet20_s1_ref1
built 2026-09-23 06:05:57 · source **real** · arch `cifar10_resnet20` · weights `file:/workspace/models/resnet20_s1_chenyaofo.pt sha256:d5442d0eadd72592` · layers 11

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
| stem | 273 | 0.490 | 0.489 | 0.880 | 0.561 | 0.499 | 0.492 | 1.058 |
| layer1.0 | 273 | 0.507 | 0.484 | 0.880 | 0.549 | 0.507 | 0.509 | 0.958 |
| layer1.1 | 273 | 0.515 | 0.485 | 0.880 | 0.532 | 0.506 | 0.517 | 0.899 |
| layer1.2 | 273 | 0.501 | 0.501 | 0.880 | 0.504 | 0.521 | 0.502 | 0.947 |
| layer2.0 | 273 | 0.515 | 0.539 | 0.880 | 0.520 | 0.531 | 0.516 | 0.875 |
| layer2.1 | 273 | 0.536 | 0.540 | 0.880 | 0.526 | 0.543 | 0.538 | 0.878 |
| layer2.2 | 273 | 0.550 | 0.542 | 0.880 | 0.540 | 0.564 | 0.553 | 0.830 |
| layer3.0 | 273 | 0.627 | 0.548 | 0.880 | 0.580 | 0.643 | 0.631 | 0.639 |
| layer3.1 | 273 | 0.748 | 0.527 | 0.880 | 0.734 | 0.763 | 0.756 | 0.411 |
| layer3.2 | 273 | 0.882 | 0.836 | 0.880 | 0.729 | 0.912 | 0.905 | 0.280 |
| penult | 273 | 0.882 | 0.836 | 0.880 | 0.729 | 0.912 | 0.905 | 0.280 |

cut sweep at `penult` (type-b = wrong and maxprob > cut; oriented AUCs vs all correct)

| cut | n_typeb | margin | dist | maxprob | energy | margin (conf-matched) |
|---|---|---|---|---|---|---|
| 0.00 | 387 | 0.912 | 0.874 | 0.912 | 0.756 | 0.912 |
| 0.50 | 358 | 0.906 | 0.866 | 0.905 | 0.742 | 0.910 |
| 0.60 | 317 | 0.895 | 0.853 | 0.894 | 0.739 | 0.909 |
| 0.70 | 273 | 0.882 | 0.836 | 0.880 | 0.729 | 0.905 |
| 0.80 | 217 | 0.861 | 0.806 | 0.858 | 0.709 | 0.895 |
| 0.90 | 155 | 0.825 | 0.760 | 0.821 | 0.689 | 0.878 |
| 0.95 | 128 | 0.803 | 0.730 | 0.798 | 0.671 | 0.873 |
| 0.99 | 67 | 0.720 | 0.628 | 0.713 | 0.567 | 0.839 |

legacy (test-split centers, direction-free AUC, conf-wrong n=273): margin 0.883, cluster_subnet 0.780, energy 0.729
