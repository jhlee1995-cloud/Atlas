# DEFORMATION  A=`results/atlas_v1_resnet20_s1`  B=`results/atlas_v1_resnet20_rand`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.089 | 0.904 | 0.831 | True | 0.926 | 0.054 | 0.531 | 0.689 | 0.344 |
| layer1.0 | layer1.0 | 0.060 | 0.900 | 0.910 | True | 0.870 | 0.093 | 0.416 | 0.565 | 0.312 |
| layer1.1 | layer1.1 | 0.091 | 0.895 | 0.936 | True | 0.928 | 0.122 | 0.359 | 0.609 | 0.266 |
| layer1.2 | layer1.2 | 0.282 | 0.589 | 0.062 | True | 0.930 | 0.147 | 0.213 | 0.463 | 0.156 |
| layer2.0 | layer2.0 | 0.204 | 0.671 | 0.642 | True | 0.928 | 0.188 | 0.219 | 0.457 | 0.188 |
| layer2.1 | layer2.1 | 0.257 | 0.742 | 0.781 | True | 0.928 | 0.204 | 0.210 | 0.460 | 0.172 |
| layer2.2 | layer2.2 | 0.281 | 0.702 | 0.470 | True | 0.880 | 0.229 | 0.209 | 0.472 | 0.203 |
| layer3.0 | layer3.0 | 0.322 | 0.747 | 0.433 | False | 0.833 | 0.238 | 0.218 | 0.423 | 0.203 |
| layer3.1 | layer3.1 | 0.495 | 0.600 | 0.227 | False | 0.682 | 0.224 | 0.130 | 0.313 | 0.188 |
| layer3.2 | layer3.2 | 0.758 | 0.478 | 0.279 | True | 0.492 | 0.217 | 0.114 | 0.205 | 0.172 |
| penult | penult | 0.758 | 0.478 | 0.279 | True | 0.492 | 0.217 | 0.114 | 0.205 | 0.172 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.513 | 0.385 | 0.100 | 0.683 | -0.005 | 0.475 | 0.336 | 0.098 | 0.648 |
| layer1.0 | 0.469 | 0.379 | 0.109 | 0.679 | -0.005 | 0.460 | 0.338 | 0.110 | 0.663 |
| layer1.1 | 0.396 | 0.353 | 0.114 | 0.656 | -0.005 | 0.407 | 0.326 | 0.116 | 0.634 |
| layer1.2 | 0.232 | 0.288 | 0.113 | 0.470 | -0.005 | 0.265 | 0.279 | 0.117 | 0.441 |
| layer2.0 | 0.239 | 0.263 | 0.107 | 0.413 | -0.005 | 0.273 | 0.251 | 0.105 | 0.409 |
| layer2.1 | 0.225 | 0.255 | 0.109 | 0.404 | -0.005 | 0.263 | 0.233 | 0.112 | 0.388 |
| layer2.2 | 0.214 | 0.237 | 0.107 | 0.389 | -0.005 | 0.249 | 0.216 | 0.110 | 0.365 |
| layer3.0 | 0.228 | 0.244 | 0.106 | 0.394 | -0.005 | 0.260 | 0.241 | 0.116 | 0.373 |
| layer3.1 | 0.119 | 0.212 | 0.105 | 0.250 | -0.005 | 0.165 | 0.197 | 0.117 | 0.229 |
| layer3.2 | 0.073 | 0.198 | 0.100 | 0.242 | -0.005 | 0.053 | 0.145 | 0.107 | 0.205 |
| penult | 0.073 | 0.198 | 0.100 | 0.242 | -0.005 | 0.053 | 0.145 | 0.107 | 0.205 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers |
|---|---|---|---|---|---|---|---|---|
| stem | -0.210 | -0.898 | +0.000 | +0.017 | +91.382 | +0.057 | +0.082 | -0.022 |
| layer1.0 | -0.797 | -1.220 | +0.000 | -0.028 | +139.655 | -0.016 | -0.060 | -0.064 |
| layer1.1 | -0.906 | -1.603 | -1.000 | -0.055 | +190.844 | +0.025 | -0.061 | -0.103 |
| layer1.2 | -1.179 | -2.776 | -3.000 | -0.155 | +132.778 | +0.046 | -0.147 | -0.145 |
| layer2.0 | +0.162 | -3.976 | -3.000 | -0.193 | +70.836 | +0.007 | +0.102 | -0.216 |
| layer2.1 | +0.080 | -5.641 | -5.000 | -0.244 | +85.494 | +0.270 | -0.097 | -0.277 |
| layer2.2 | -0.283 | -6.145 | -7.000 | -0.266 | +111.221 | +0.339 | -0.199 | -0.327 |
| layer3.0 | +5.565 | -6.706 | -4.000 | -0.422 | +68.241 | +0.365 | +1.231 | -0.444 |
| layer3.1 | +4.715 | -8.089 | -8.000 | -0.738 | +73.420 | +0.471 | +0.535 | -0.618 |
| layer3.2 | +13.914 | -5.906 | +21.000 | -2.936 | +72.308 | +0.680 | +1.757 | -0.730 |
| penult | +13.914 | -5.906 | +21.000 | -2.936 | +72.308 | +0.680 | +1.757 | -0.730 |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer1.1 | stem |
| spectral_slope | layer2.0 | stem |
| spectral_anisotropy | layer1.2 | stem |
| noise_sigma | layer1.0 | stem |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | stem | stem |
| edge_density | layer1.0 | stem |
| orientation_entropy | layer2.0 | stem |
| blockiness | layer3.0 | stem |
| class | layer3.1 | stem |
| coarse_animal_vehicle | layer3.0 | stem |
| corruption_family | layer2.0 | stem |
| corruption_type | layer2.0 | stem |
| severity | layer3.0 | stem |