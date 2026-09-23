# ATLAS — margin_v1_resnet20_rand
built 2026-09-23 06:05:20 · source **real** · arch `cifar10_resnet20` · weights `file:/workspace/models/resnet20_rand99_chenyaofo.pt sha256:c761ef798fb28398` · layers 11

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
| stem | 0 | · | · | · | · | 0.520 | · | · |
| layer1.0 | 0 | · | · | · | · | 0.536 | · | · |
| layer1.1 | 0 | · | · | · | · | 0.555 | · | · |
| layer1.2 | 0 | · | · | · | · | 0.558 | · | · |
| layer2.0 | 0 | · | · | · | · | 0.530 | · | · |
| layer2.1 | 0 | · | · | · | · | 0.555 | · | · |
| layer2.2 | 0 | · | · | · | · | 0.554 | · | · |
| layer3.0 | 0 | · | · | · | · | 0.560 | · | · |
| layer3.1 | 0 | · | · | · | · | 0.540 | · | · |
| layer3.2 | 0 | · | · | · | · | 0.542 | · | · |
| penult | 0 | · | · | · | · | 0.542 | · | · |

cut sweep at `penult` (type-b = wrong and maxprob > cut; oriented AUCs vs all correct)

| cut | n_typeb | margin | dist | maxprob | energy | margin (conf-matched) |
|---|---|---|---|---|---|---|
| 0.00 | 4396 | 0.542 | 0.474 | 0.564 | 0.566 | 0.542 |
| 0.50 | 1 | · | · | · | · | · |
| 0.60 | 0 | · | · | · | · | · |
| 0.70 | 0 | · | · | · | · | · |
| 0.80 | 0 | · | · | · | · | · |
| 0.90 | 0 | · | · | · | · | · |
| 0.95 | 0 | · | · | · | · | · |
| 0.99 | 0 | · | · | · | · | · |

legacy (test-split centers, direction-free AUC, conf-wrong n=0): margin ·, cluster_subnet ·, energy ·
