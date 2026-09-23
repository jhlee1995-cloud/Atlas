# ATLAS — margin_b1_resnet50_swap
built 2026-09-23 15:42:17 · source **real** · arch `tv_resnet50` · weights `torchvision tv_resnet50 IMAGENET1K_V2 file:/workspace/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth sha256:11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca` · layers 18

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
| stem | 35 | 0.400 | 0.571 | 0.030 | 0.439 | 0.500 | 0.384 | 1.314 |
| layer1.0 | 35 | 0.553 | 0.636 | 0.030 | 0.456 | 0.513 | 0.556 | 0.787 |
| layer1.1 | 35 | 0.577 | 0.636 | 0.030 | 0.445 | 0.512 | 0.579 | 0.513 |
| layer1.2 | 35 | 0.530 | 0.635 | 0.030 | 0.454 | 0.508 | 0.545 | 0.773 |
| layer2.0 | 35 | 0.454 | 0.617 | 0.030 | 0.456 | 0.513 | 0.471 | 1.120 |
| layer2.1 | 35 | 0.532 | 0.631 | 0.030 | 0.437 | 0.512 | 0.540 | 0.728 |
| layer2.2 | 35 | 0.510 | 0.636 | 0.030 | 0.424 | 0.513 | 0.534 | 0.885 |
| layer2.3 | 35 | 0.509 | 0.639 | 0.030 | 0.420 | 0.516 | 0.530 | 0.938 |
| layer3.0 | 35 | 0.503 | 0.672 | 0.030 | 0.404 | 0.528 | 0.519 | 1.317 |
| layer3.1 | 35 | 0.539 | 0.664 | 0.030 | 0.450 | 0.533 | 0.560 | 0.877 |
| layer3.2 | 35 | 0.547 | 0.674 | 0.030 | 0.415 | 0.538 | 0.569 | 0.820 |
| layer3.3 | 35 | 0.574 | 0.679 | 0.030 | 0.375 | 0.538 | 0.585 | 0.670 |
| layer3.4 | 35 | 0.558 | 0.682 | 0.030 | 0.393 | 0.554 | 0.575 | 0.802 |
| layer3.5 | 35 | 0.491 | 0.693 | 0.030 | 0.313 | 0.572 | 0.517 | 1.047 |
| layer4.0 | 35 | 0.536 | 0.756 | 0.030 | 0.304 | 0.654 | 0.565 | 0.890 |
| layer4.1 | 35 | 0.571 | 0.745 | 0.030 | 0.344 | 0.740 | 0.593 | 0.867 |
| layer4.2 | 35 | 0.596 | 0.735 | 0.030 | 0.264 | 0.801 | 0.616 | 0.769 |
| penult | 35 | 0.596 | 0.735 | 0.030 | 0.264 | 0.801 | 0.616 | 0.769 |

cut sweep at `penult` (type-b = wrong and maxprob > cut; oriented AUCs vs all correct)

| cut | n_typeb | margin | dist | maxprob | energy | margin (conf-matched) |
|---|---|---|---|---|---|---|
| 0.00 | 5506 | 0.801 | 0.710 | 0.803 | 0.303 | 0.801 |
| 0.50 | 303 | 0.649 | 0.604 | 0.168 | 0.383 | 0.746 |
| 0.60 | 98 | 0.623 | 0.681 | 0.067 | 0.303 | 0.691 |
| 0.70 | 35 | 0.596 | 0.735 | 0.030 | 0.264 | 0.616 |
| 0.80 | 4 | · | · | · | · | · |
| 0.90 | 1 | · | · | · | · | · |
| 0.95 | 0 | · | · | · | · | · |
| 0.99 | 0 | · | · | · | · | · |

legacy (test-split centers, direction-free AUC, conf-wrong n=35): margin 0.756, cluster_subnet 0.743, energy 0.736
