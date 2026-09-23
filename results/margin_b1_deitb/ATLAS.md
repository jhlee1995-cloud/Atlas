# ATLAS — margin_b1_deitb
built 2026-09-23 17:24:21 · source **real** · arch `timm_deit_base_patch16_224` · weights `timm timm_deit_base_patch16_224 fb_in1k file:/workspace/.cache/huggingface/hub/models--timm--deit_base_patch16_224.fb_in1k/snapshots/b78cc5532a69df6bcad9c3a8d76653fd20b31ac6/model.safetensors sha256:cd2da27b74ed7f68b599f16c77af3e1e80f01c75f9ad96029d22ce747a247e8e` · layers 14

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
| block.0 | 1346 | 0.500 | 0.501 | 0.586 | 0.529 | 0.509 | 0.502 | 0.989 |
| block.1 | 1346 | 0.500 | 0.507 | 0.586 | 0.508 | 0.509 | 0.502 | 0.969 |
| block.2 | 1346 | 0.512 | 0.511 | 0.586 | 0.492 | 0.520 | 0.515 | 0.948 |
| block.3 | 1346 | 0.499 | 0.510 | 0.586 | 0.498 | 0.519 | 0.502 | 0.992 |
| block.4 | 1346 | 0.499 | 0.512 | 0.586 | 0.518 | 0.515 | 0.503 | 1.019 |
| block.5 | 1346 | 0.511 | 0.520 | 0.586 | 0.514 | 0.521 | 0.515 | 0.973 |
| block.6 | 1346 | 0.531 | 0.528 | 0.586 | 0.514 | 0.539 | 0.536 | 0.896 |
| block.7 | 1346 | 0.563 | 0.540 | 0.586 | 0.514 | 0.569 | 0.573 | 0.792 |
| block.8 | 1346 | 0.585 | 0.543 | 0.586 | 0.527 | 0.610 | 0.604 | 0.700 |
| block.9 | 1346 | 0.634 | 0.540 | 0.586 | 0.566 | 0.702 | 0.673 | 0.558 |
| block.10 | 1346 | 0.710 | 0.542 | 0.586 | 0.634 | 0.823 | 0.785 | 0.360 |
| block.11 | 1346 | 0.711 | 0.571 | 0.586 | 0.646 | 0.819 | 0.784 | 0.371 |
| penult_mean | 1346 | 0.736 | 0.650 | 0.586 | 0.430 | 0.848 | 0.812 | 0.395 |
| penult | 1346 | 0.739 | 0.645 | 0.586 | 0.367 | 0.866 | 0.827 | 0.432 |

cut sweep at `penult` (type-b = wrong and maxprob > cut; oriented AUCs vs all correct)

| cut | n_typeb | margin | dist | maxprob | energy | margin (conf-matched) |
|---|---|---|---|---|---|---|
| 0.00 | 5319 | 0.866 | 0.775 | 0.853 | 0.243 | 0.866 |
| 0.50 | 2678 | 0.807 | 0.688 | 0.733 | 0.330 | 0.843 |
| 0.60 | 1989 | 0.779 | 0.668 | 0.673 | 0.349 | 0.837 |
| 0.70 | 1346 | 0.739 | 0.645 | 0.586 | 0.367 | 0.827 |
| 0.80 | 722 | 0.677 | 0.611 | 0.432 | 0.394 | 0.814 |
| 0.90 | 89 | 0.694 | 0.721 | 0.048 | 0.268 | 0.826 |
| 0.95 | 5 | · | · | · | · | · |
| 0.99 | 0 | · | · | · | · | · |

legacy (test-split centers, direction-free AUC, conf-wrong n=1346): margin 0.812, cluster_subnet 0.664, energy 0.633
