# ATLAS — margin_b1_vitb16
built 2026-09-23 17:06:58 · source **real** · arch `tv_vit_b_16` · weights `torchvision tv_vit_b_16 IMAGENET1K_V1 file:/workspace/.cache/torch/hub/checkpoints/vit_b_16-c867db91.pth sha256:c867db91d3e12c6cbadabb610d73c24a546bf82d8c03a9fea34f43a712ddb0e9` · layers 14

## 1. Per-layer invariants
| layer | D | PR | eff.rank | dim95 | TwoNN ID | sep ratio | NC1 | ETF dev | hub skew | test sparse | NC acc(test) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| block.0 | 768 | · | · | · | · | · | · | · | · | · | · |
| block.1 | 768 | · | · | · | · | · | · | · | · | · | · |
| block.2 | 768 | · | · | · | · | · | · | · | · | · | · |
| block.3 | 768 | · | · | · | · | · | · | · | · | · | · |
| block.4 | 768 | · | · | · | · | · | · | · | · | · | · |
| block.5 | 768 | · | · | · | · | · | · | · | · | · | · |
| block.6 | 768 | · | · | · | · | · | · | · | · | · | · |
| block.7 | 768 | · | · | · | · | · | · | · | · | · | · |
| block.8 | 768 | · | · | · | · | · | · | · | · | · | · |
| block.9 | 768 | · | · | · | · | · | · | · | · | · | · |
| block.10 | 768 | · | · | · | · | · | · | · | · | · | · |
| block.11 | 768 | · | · | · | · | · | · | · | · | · | · |
| penult_mean | 768 | · | · | · | · | · | · | · | · | · | · |
| penult | 768 | · | · | · | · | · | · | · | · | · | · |

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
| block.0 | 1340 | 0.494 | 0.517 | 0.599 | 0.509 | 0.501 | 0.497 | 1.047 |
| block.1 | 1340 | 0.501 | 0.511 | 0.599 | 0.509 | 0.514 | 0.503 | 0.950 |
| block.2 | 1340 | 0.491 | 0.513 | 0.599 | 0.511 | 0.506 | 0.494 | 1.046 |
| block.3 | 1340 | 0.498 | 0.519 | 0.599 | 0.508 | 0.515 | 0.501 | 1.008 |
| block.4 | 1340 | 0.500 | 0.518 | 0.599 | 0.505 | 0.523 | 0.504 | 1.012 |
| block.5 | 1340 | 0.499 | 0.517 | 0.599 | 0.507 | 0.527 | 0.507 | 1.028 |
| block.6 | 1340 | 0.526 | 0.527 | 0.599 | 0.499 | 0.553 | 0.538 | 0.907 |
| block.7 | 1340 | 0.586 | 0.536 | 0.599 | 0.494 | 0.614 | 0.607 | 0.696 |
| block.8 | 1340 | 0.614 | 0.534 | 0.599 | 0.501 | 0.657 | 0.644 | 0.586 |
| block.9 | 1340 | 0.644 | 0.540 | 0.599 | 0.535 | 0.739 | 0.696 | 0.470 |
| block.10 | 1340 | 0.685 | 0.558 | 0.599 | 0.600 | 0.819 | 0.762 | 0.435 |
| block.11 | 1340 | 0.701 | 0.544 | 0.599 | 0.643 | 0.828 | 0.781 | 0.431 |
| penult_mean | 1340 | 0.561 | 0.505 | 0.599 | 0.553 | 0.615 | 0.585 | 0.785 |
| penult | 1340 | 0.712 | 0.624 | 0.599 | 0.372 | 0.861 | 0.808 | 0.504 |

cut sweep at `penult` (type-b = wrong and maxprob > cut; oriented AUCs vs all correct)

| cut | n_typeb | margin | dist | maxprob | energy | margin (conf-matched) |
|---|---|---|---|---|---|---|
| 0.00 | 5296 | 0.861 | 0.773 | 0.856 | 0.231 | 0.861 |
| 0.50 | 2673 | 0.792 | 0.674 | 0.740 | 0.324 | 0.831 |
| 0.60 | 1975 | 0.758 | 0.653 | 0.681 | 0.344 | 0.821 |
| 0.70 | 1340 | 0.712 | 0.624 | 0.599 | 0.372 | 0.808 |
| 0.80 | 722 | 0.642 | 0.591 | 0.453 | 0.402 | 0.787 |
| 0.90 | 97 | 0.650 | 0.710 | 0.077 | 0.277 | 0.798 |
| 0.95 | 9 | · | · | · | · | · |
| 0.99 | 0 | · | · | · | · | · |

legacy (test-split centers, direction-free AUC, conf-wrong n=1340): margin 0.786, cluster_subnet 0.647, energy 0.628
