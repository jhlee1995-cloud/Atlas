# ATLAS — margin_v1_resnet20_s2
built 2026-09-23 06:05:48 · source **real** · arch `cifar10_resnet20` · weights `file:/workspace/models/resnet20_s2_chenyaofo.pt sha256:e7e5386fe7d26961` · layers 11

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
| stem | 247 | 0.505 | 0.491 | 0.883 | 0.558 | 0.487 | 0.507 | 0.951 |
| layer1.0 | 247 | 0.498 | 0.497 | 0.883 | 0.571 | 0.496 | 0.499 | 0.963 |
| layer1.1 | 247 | 0.538 | 0.502 | 0.883 | 0.533 | 0.543 | 0.538 | 0.853 |
| layer1.2 | 247 | 0.547 | 0.504 | 0.883 | 0.562 | 0.534 | 0.548 | 0.730 |
| layer2.0 | 247 | 0.525 | 0.516 | 0.883 | 0.522 | 0.527 | 0.526 | 0.867 |
| layer2.1 | 247 | 0.545 | 0.512 | 0.883 | 0.549 | 0.542 | 0.545 | 0.807 |
| layer2.2 | 247 | 0.581 | 0.516 | 0.883 | 0.568 | 0.577 | 0.582 | 0.701 |
| layer3.0 | 247 | 0.684 | 0.543 | 0.883 | 0.603 | 0.669 | 0.689 | 0.452 |
| layer3.1 | 247 | 0.750 | 0.526 | 0.883 | 0.693 | 0.765 | 0.758 | 0.393 |
| layer3.2 | 247 | 0.884 | 0.822 | 0.883 | 0.744 | 0.919 | 0.906 | 0.284 |
| penult | 247 | 0.884 | 0.822 | 0.883 | 0.744 | 0.919 | 0.906 | 0.284 |

cut sweep at `penult` (type-b = wrong and maxprob > cut; oriented AUCs vs all correct)

| cut | n_typeb | margin | dist | maxprob | energy | margin (conf-matched) |
|---|---|---|---|---|---|---|
| 0.00 | 375 | 0.919 | 0.870 | 0.920 | 0.785 | 0.919 |
| 0.50 | 336 | 0.911 | 0.857 | 0.911 | 0.772 | 0.915 |
| 0.60 | 290 | 0.899 | 0.841 | 0.898 | 0.756 | 0.910 |
| 0.70 | 247 | 0.884 | 0.822 | 0.883 | 0.744 | 0.906 |
| 0.80 | 202 | 0.866 | 0.797 | 0.864 | 0.727 | 0.901 |
| 0.90 | 153 | 0.840 | 0.763 | 0.837 | 0.712 | 0.897 |
| 0.95 | 105 | 0.801 | 0.711 | 0.797 | 0.687 | 0.880 |
| 0.99 | 56 | 0.730 | 0.615 | 0.722 | 0.616 | 0.860 |

legacy (test-split centers, direction-free AUC, conf-wrong n=247): margin 0.884, cluster_subnet 0.762, energy 0.744
