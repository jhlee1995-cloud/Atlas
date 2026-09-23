# DEFORMATION  A=`results/atlas_v1_resnet20_s1_st2`  B=`results/atlas_v1_resnet20_s1_st3`  same_space=True
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.000 | 1.000 | 1.000 | True | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| layer1.0 | layer1.0 | 0.000 | 1.000 | 1.000 | True | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| layer1.1 | layer1.1 | 0.000 | 1.000 | 1.000 | True | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| layer1.2 | layer1.2 | 0.000 | 1.000 | 1.000 | True | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| layer2.0 | layer2.0 | 0.000 | 1.000 | 1.000 | True | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| layer2.1 | layer2.1 | 0.000 | 1.000 | 1.000 | True | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| layer2.2 | layer2.2 | 0.000 | 1.000 | 1.000 | True | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| layer3.0 | layer3.0 | 0.000 | 1.000 | 1.000 | True | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| layer3.1 | layer3.1 | 0.000 | 1.000 | 1.000 | True | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| layer3.2 | layer3.2 | 0.000 | 1.000 | 1.000 | True | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| penult | penult | 0.000 | 1.000 | 1.000 | True | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 1.000 | 1.000 | 0.120 | 1.000 | 1.000 | 1.000 | 1.000 | 0.122 | 1.000 |
| layer1.0 | 1.000 | 1.000 | 0.118 | 1.000 | 1.000 | 1.000 | 1.000 | 0.119 | 1.000 |
| layer1.1 | 1.000 | 1.000 | 0.115 | 1.000 | 1.000 | 1.000 | 1.000 | 0.120 | 1.000 |
| layer1.2 | 1.000 | 1.000 | 0.112 | 1.000 | 1.000 | 1.000 | 1.000 | 0.120 | 1.000 |
| layer2.0 | 1.000 | 1.000 | 0.106 | 1.000 | 1.000 | 1.000 | 1.000 | 0.113 | 1.000 |
| layer2.1 | 1.000 | 1.000 | 0.106 | 1.000 | 1.000 | 1.000 | 1.000 | 0.113 | 1.000 |
| layer2.2 | 1.000 | 1.000 | 0.105 | 1.000 | 1.000 | 1.000 | 1.000 | 0.113 | 1.000 |
| layer3.0 | 1.000 | 1.000 | 0.103 | 1.000 | 1.000 | 1.000 | 1.000 | 0.113 | 1.000 |
| layer3.1 | 1.000 | 1.000 | 0.102 | 1.000 | 1.000 | 1.000 | 1.000 | 0.144 | 1.000 |
| layer3.2 | 1.000 | 1.000 | 0.100 | 1.000 | 1.000 | 1.000 | 1.000 | 0.127 | 1.000 |
| penult | 1.000 | 1.000 | 0.100 | 1.000 | 1.000 | 1.000 | 1.000 | 0.127 | 1.000 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | · | · | · | · | · |
| layer1.0 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | · | · | · | · | · |
| layer1.1 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | · | · | · | · | · |
| layer1.2 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | · | · | · | · | · |
| layer2.0 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | · | · | · | · | · |
| layer2.1 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | · | · | · | · | · |
| layer2.2 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | · | · | · | · | · |
| layer3.0 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | · | · | · | · | · |
| layer3.1 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | · | · | · | · | · |
| layer3.2 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | · | · | · | · | · |
| penult | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer1.1 | layer1.1 |
| spectral_slope | layer2.0 | layer2.0 |
| spectral_anisotropy | layer1.2 | layer1.2 |
| noise_sigma | layer1.0 | layer1.0 |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | stem | stem |
| edge_density | layer1.0 | layer1.0 |
| orientation_entropy | layer2.0 | layer2.0 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.1 |
| coarse_animal_vehicle | layer3.0 | layer3.0 |
| corruption_family | layer2.0 | layer2.0 |
| corruption_type | layer2.0 | layer2.0 |
| severity | layer3.0 | layer3.0 |