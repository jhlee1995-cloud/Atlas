# ATLAS — margin_b1_resnet50
built 2026-09-23 15:34:26 · source **real** · arch `tv_resnet50` · weights `torchvision tv_resnet50 IMAGENET1K_V2 file:/workspace/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth sha256:11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca` · layers 18

## 1. Per-layer invariants
| layer | D | PR | eff.rank | dim95 | TwoNN ID | sep ratio | NC1 | ETF dev | hub skew | test sparse | NC acc(test) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | 64 | · | · | · | · | · | · | · | · | · | · |
| layer1.0 | 256 | · | · | · | · | · | · | · | · | · | · |
| layer1.1 | 256 | · | · | · | · | · | · | · | · | · | · |
| layer1.2 | 256 | · | · | · | · | · | · | · | · | · | · |
| layer2.0 | 512 | · | · | · | · | · | · | · | · | · | · |
| layer2.1 | 512 | · | · | · | · | · | · | · | · | · | · |
| layer2.2 | 512 | · | · | · | · | · | · | · | · | · | · |
| layer2.3 | 512 | · | · | · | · | · | · | · | · | · | · |
| layer3.0 | 1024 | · | · | · | · | · | · | · | · | · | · |
| layer3.1 | 1024 | · | · | · | · | · | · | · | · | · | · |
| layer3.2 | 1024 | · | · | · | · | · | · | · | · | · | · |
| layer3.3 | 1024 | · | · | · | · | · | · | · | · | · | · |
| layer3.4 | 1024 | · | · | · | · | · | · | · | · | · | · |
| layer3.5 | 1024 | · | · | · | · | · | · | · | · | · | · |
| layer4.0 | 2048 | · | · | · | · | · | · | · | · | · | · |
| layer4.1 | 2048 | · | · | · | · | · | · | · | · | · | · |
| layer4.2 | 2048 | · | · | · | · | · | · | · | · | · | · |
| penult | 2048 | · | · | · | · | · | · | · | · | · | · |

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
| stem | 38 | 0.587 | 0.501 | 0.026 | 0.498 | 0.503 | 0.588 | 0.459 |
| layer1.0 | 38 | 0.534 | 0.459 | 0.026 | 0.555 | 0.505 | 0.525 | 0.926 |
| layer1.1 | 38 | 0.542 | 0.472 | 0.026 | 0.546 | 0.511 | 0.532 | 0.805 |
| layer1.2 | 38 | 0.545 | 0.488 | 0.026 | 0.502 | 0.513 | 0.550 | 0.907 |
| layer2.0 | 38 | 0.565 | 0.528 | 0.026 | 0.462 | 0.511 | 0.558 | 0.896 |
| layer2.1 | 38 | 0.491 | 0.549 | 0.026 | 0.401 | 0.515 | 0.500 | 1.155 |
| layer2.2 | 38 | 0.463 | 0.548 | 0.026 | 0.393 | 0.510 | 0.475 | 1.373 |
| layer2.3 | 38 | 0.496 | 0.550 | 0.026 | 0.412 | 0.525 | 0.501 | 1.071 |
| layer3.0 | 38 | 0.514 | 0.569 | 0.026 | 0.387 | 0.528 | 0.515 | 1.265 |
| layer3.1 | 38 | 0.626 | 0.583 | 0.026 | 0.416 | 0.540 | 0.624 | 0.659 |
| layer3.2 | 38 | 0.588 | 0.595 | 0.026 | 0.405 | 0.541 | 0.584 | 0.475 |
| layer3.3 | 38 | 0.518 | 0.603 | 0.026 | 0.367 | 0.542 | 0.523 | 1.009 |
| layer3.4 | 38 | 0.471 | 0.637 | 0.026 | 0.325 | 0.547 | 0.482 | 1.221 |
| layer3.5 | 38 | 0.529 | 0.644 | 0.026 | 0.373 | 0.560 | 0.551 | 1.007 |
| layer4.0 | 38 | 0.501 | 0.737 | 0.026 | 0.262 | 0.656 | 0.537 | 1.131 |
| layer4.1 | 38 | 0.569 | 0.795 | 0.026 | 0.234 | 0.739 | 0.610 | 0.665 |
| layer4.2 | 38 | 0.640 | 0.814 | 0.026 | 0.175 | 0.801 | 0.671 | 0.625 |
| penult | 38 | 0.640 | 0.814 | 0.026 | 0.175 | 0.801 | 0.671 | 0.625 |

cut sweep at `penult` (type-b = wrong and maxprob > cut; oriented AUCs vs all correct)

| cut | n_typeb | margin | dist | maxprob | energy | margin (conf-matched) |
|---|---|---|---|---|---|---|
| 0.00 | 5619 | 0.801 | 0.713 | 0.800 | 0.300 | 0.801 |
| 0.50 | 356 | 0.676 | 0.647 | 0.161 | 0.338 | 0.773 |
| 0.60 | 119 | 0.658 | 0.719 | 0.066 | 0.263 | 0.729 |
| 0.70 | 38 | 0.640 | 0.814 | 0.026 | 0.175 | 0.671 |
| 0.80 | 12 | 0.711 | 0.863 | 0.010 | 0.129 | 0.711 |
| 0.90 | 1 | · | · | · | · | · |
| 0.95 | 0 | · | · | · | · | · |
| 0.99 | 0 | · | · | · | · | · |

legacy (test-split centers, direction-free AUC, conf-wrong n=38): margin 0.691, cluster_subnet 0.815, energy 0.825
