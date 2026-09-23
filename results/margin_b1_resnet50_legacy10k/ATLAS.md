# ATLAS — margin_b1_resnet50_legacy10k
built 2026-09-23 14:32:28 · source **real** · arch `tv_resnet50` · weights `torchvision tv_resnet50 IMAGENET1K_V2 file:/workspace/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth sha256:11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca` · layers 6

## 1. Per-layer invariants
| layer | D | PR | eff.rank | dim95 | TwoNN ID | sep ratio | NC1 | ETF dev | hub skew | test sparse | NC acc(test) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | 64 | · | · | · | · | · | · | · | · | · | · |
| layer1 | 256 | · | · | · | · | · | · | · | · | · | · |
| layer2 | 512 | · | · | · | · | · | · | · | · | · | · |
| layer3 | 1024 | · | · | · | · | · | · | · | · | · | · |
| layer4 | 2048 | · | · | · | · | · | · | · | · | · | · |
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
| stem | 17 | 0.484 | 0.433 | 0.024 | 0.544 | 0.505 | 0.496 | 1.080 |
| layer1 | 17 | 0.567 | 0.488 | 0.024 | 0.615 | 0.523 | 0.586 | 0.715 |
| layer2 | 17 | 0.524 | 0.521 | 0.024 | 0.526 | 0.545 | 0.521 | 1.160 |
| layer3 | 17 | 0.654 | 0.618 | 0.024 | 0.433 | 0.637 | 0.695 | 0.370 |
| layer4 | 17 | 0.864 | 0.821 | 0.024 | 0.194 | 0.870 | 0.899 | 0.294 |
| penult | 17 | 0.864 | 0.821 | 0.024 | 0.194 | 0.870 | 0.899 | 0.294 |

cut sweep at `penult` (type-b = wrong and maxprob > cut; oriented AUCs vs all correct)

| cut | n_typeb | margin | dist | maxprob | energy | margin (conf-matched) |
|---|---|---|---|---|---|---|
| 0.00 | 2300 | 0.870 | 0.743 | 0.809 | 0.296 | 0.870 |
| 0.50 | 122 | 0.790 | 0.644 | 0.167 | 0.367 | 0.853 |
| 0.60 | 36 | 0.856 | 0.751 | 0.057 | 0.278 | 0.902 |
| 0.70 | 17 | 0.864 | 0.821 | 0.024 | 0.194 | 0.899 |
| 0.80 | 5 | · | · | · | · | · |
| 0.90 | 2 | · | · | · | · | · |
| 0.95 | 1 | · | · | · | · | · |
| 0.99 | 0 | · | · | · | · | · |

legacy (test-split centers, direction-free AUC, conf-wrong n=17): margin 0.864, cluster_subnet 0.810, energy 0.806
