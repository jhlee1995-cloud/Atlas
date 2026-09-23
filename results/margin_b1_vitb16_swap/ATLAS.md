# ATLAS — margin_b1_vitb16_swap
built 2026-09-23 17:16:24 · source **real** · arch `tv_vit_b_16` · weights `torchvision tv_vit_b_16 IMAGENET1K_V1 file:/workspace/.cache/torch/hub/checkpoints/vit_b_16-c867db91.pth sha256:c867db91d3e12c6cbadabb610d73c24a546bf82d8c03a9fea34f43a712ddb0e9` · layers 14

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
| block.0 | 1266 | 0.497 | 0.514 | 0.587 | 0.512 | 0.504 | 0.498 | 1.037 |
| block.1 | 1266 | 0.507 | 0.518 | 0.587 | 0.510 | 0.505 | 0.508 | 0.938 |
| block.2 | 1266 | 0.505 | 0.521 | 0.587 | 0.502 | 0.508 | 0.506 | 1.023 |
| block.3 | 1266 | 0.517 | 0.518 | 0.587 | 0.504 | 0.515 | 0.519 | 0.926 |
| block.4 | 1266 | 0.520 | 0.515 | 0.587 | 0.499 | 0.526 | 0.523 | 0.917 |
| block.5 | 1266 | 0.505 | 0.519 | 0.587 | 0.493 | 0.527 | 0.511 | 0.980 |
| block.6 | 1266 | 0.527 | 0.537 | 0.587 | 0.483 | 0.555 | 0.540 | 0.937 |
| block.7 | 1266 | 0.569 | 0.543 | 0.587 | 0.481 | 0.601 | 0.588 | 0.779 |
| block.8 | 1266 | 0.620 | 0.547 | 0.587 | 0.495 | 0.661 | 0.650 | 0.583 |
| block.9 | 1266 | 0.643 | 0.541 | 0.587 | 0.544 | 0.741 | 0.696 | 0.483 |
| block.10 | 1266 | 0.684 | 0.546 | 0.587 | 0.618 | 0.820 | 0.759 | 0.424 |
| block.11 | 1266 | 0.695 | 0.538 | 0.587 | 0.659 | 0.827 | 0.775 | 0.432 |
| penult_mean | 1266 | 0.578 | 0.514 | 0.587 | 0.562 | 0.611 | 0.601 | 0.695 |
| penult | 1266 | 0.707 | 0.634 | 0.587 | 0.362 | 0.858 | 0.802 | 0.500 |

cut sweep at `penult` (type-b = wrong and maxprob > cut; oriented AUCs vs all correct)

| cut | n_typeb | margin | dist | maxprob | energy | margin (conf-matched) |
|---|---|---|---|---|---|---|
| 0.00 | 5136 | 0.858 | 0.771 | 0.853 | 0.234 | 0.858 |
| 0.50 | 2627 | 0.792 | 0.678 | 0.739 | 0.322 | 0.830 |
| 0.60 | 1890 | 0.755 | 0.656 | 0.674 | 0.341 | 0.818 |
| 0.70 | 1266 | 0.707 | 0.634 | 0.587 | 0.362 | 0.802 |
| 0.80 | 700 | 0.634 | 0.591 | 0.442 | 0.404 | 0.778 |
| 0.90 | 111 | 0.644 | 0.717 | 0.073 | 0.264 | 0.791 |
| 0.95 | 5 | · | · | · | · | · |
| 0.99 | 0 | · | · | · | · | · |

legacy (test-split centers, direction-free AUC, conf-wrong n=1266): margin 0.782, cluster_subnet 0.659, energy 0.638
