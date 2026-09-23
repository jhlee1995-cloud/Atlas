# DEFORMATION  A=`results/atlas_v1_resnet56_e40`  B=`results/atlas_v1_resnet56_e40_st3`  same_space=True
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.000 | 1.000 | 1.000 | True | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| layer1.0 | layer1.0 | 0.000 | 1.000 | 1.000 | True | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| layer1.5 | layer1.5 | 0.000 | 1.000 | 1.000 | True | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| layer1.8 | layer1.8 | 0.000 | 1.000 | 1.000 | True | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| layer2.0 | layer2.0 | 0.000 | 1.000 | 1.000 | True | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| layer2.5 | layer2.5 | 0.000 | 1.000 | 1.000 | True | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| layer2.8 | layer2.8 | 0.000 | 1.000 | 1.000 | True | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| layer3.0 | layer3.0 | 0.000 | 1.000 | 1.000 | True | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| layer3.5 | layer3.5 | 0.000 | 1.000 | 1.000 | True | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| layer3.8 | layer3.8 | 0.000 | 1.000 | 1.000 | True | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| penult | penult | 0.000 | 1.000 | 1.000 | True | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 1.000 | 1.000 | 0.119 | 1.000 | 1.000 | 1.000 | 1.000 | 0.126 | 1.000 |
| layer1.0 | 1.000 | 1.000 | 0.121 | 1.000 | 1.000 | 1.000 | 1.000 | 0.126 | 1.000 |
| layer1.5 | 1.000 | 1.000 | 0.115 | 1.000 | 1.000 | 1.000 | 1.000 | 0.120 | 1.000 |
| layer1.8 | 1.000 | 1.000 | 0.110 | 1.000 | 1.000 | 1.000 | 1.000 | 0.116 | 1.000 |
| layer2.0 | 1.000 | 1.000 | 0.108 | 1.000 | 1.000 | 1.000 | 1.000 | 0.115 | 1.000 |
| layer2.5 | 1.000 | 1.000 | 0.106 | 1.000 | 1.000 | 1.000 | 1.000 | 0.115 | 1.000 |
| layer2.8 | 1.000 | 1.000 | 0.104 | 1.000 | 1.000 | 1.000 | 1.000 | 0.115 | 1.000 |
| layer3.0 | 1.000 | 1.000 | 0.103 | 1.000 | 1.000 | 1.000 | 1.000 | 0.116 | 1.000 |
| layer3.5 | 1.000 | 1.000 | 0.101 | 1.000 | 1.000 | 1.000 | 1.000 | 0.129 | 1.000 |
| layer3.8 | 1.000 | 1.000 | 0.100 | 1.000 | 1.000 | 1.000 | 1.000 | 0.125 | 1.000 |
| penult | 1.000 | 1.000 | 0.100 | 1.000 | 1.000 | 1.000 | 1.000 | 0.125 | 1.000 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | · | · | · | · | · |
| layer1.0 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | · | · | · | · | · |
| layer1.5 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | · | · | · | · | · |
| layer1.8 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | · | · | · | · | · |
| layer2.0 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | · | · | · | · | · |
| layer2.5 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | · | · | · | · | · |
| layer2.8 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | · | · | · | · | · |
| layer3.0 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | · | · | · | · | · |
| layer3.5 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | · | · | · | · | · |
| layer3.8 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | · | · | · | · | · |
| penult | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer1.5 | layer1.5 |
| spectral_slope | layer1.5 | layer1.5 |
| spectral_anisotropy | layer2.0 | layer2.0 |
| noise_sigma | layer1.0 | layer1.0 |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | layer1.0 | layer1.0 |
| edge_density | stem | stem |
| orientation_entropy | layer2.0 | layer2.0 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.5 | layer3.5 |
| coarse_animal_vehicle | layer3.0 | layer3.0 |
| corruption_family | layer1.5 | layer1.5 |
| corruption_type | layer1.8 | layer1.8 |
| severity | layer3.0 | layer3.0 |