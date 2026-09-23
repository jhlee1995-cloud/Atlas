# DEFORMATION  A=`results/atlas_v1_resnet56_s0hub`  B=`results/atlas_v1_resnet56_rand`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.095 | 0.900 | 0.810 | True | 0.936 | 0.036 | 0.572 | 0.731 | 0.469 |
| layer1.0 | layer1.0 | 0.067 | 0.913 | 0.847 | True | 0.886 | 0.080 | 0.435 | 0.590 | 0.312 |
| layer1.5 | layer1.5 | 0.132 | 0.894 | 0.904 | True | 0.920 | 0.144 | 0.353 | 0.612 | 0.266 |
| layer1.8 | layer1.8 | 0.258 | 0.775 | 0.859 | False | 0.926 | 0.188 | 0.237 | 0.551 | 0.156 |
| layer2.0 | layer2.0 | 0.186 | 0.811 | 0.716 | True | 0.905 | 0.230 | 0.333 | 0.618 | 0.203 |
| layer2.5 | layer2.5 | 0.422 | 0.550 | 0.088 | True | 0.868 | 0.283 | 0.235 | 0.461 | 0.156 |
| layer2.8 | layer2.8 | 0.429 | 0.558 | 0.088 | True | 0.849 | 0.298 | 0.234 | 0.471 | 0.109 |
| layer3.0 | layer3.0 | 0.423 | 0.573 | 0.046 | True | 0.845 | 0.360 | 0.226 | 0.407 | 0.172 |
| layer3.5 | layer3.5 | 0.651 | 0.428 | -0.061 | False | 0.756 | 0.359 | 0.141 | 0.328 | 0.172 |
| layer3.8 | layer3.8 | 0.968 | 0.344 | -0.073 | False | 0.366 | 0.273 | 0.107 | 0.167 | 0.141 |
| penult | penult | 0.968 | 0.344 | -0.073 | False | 0.366 | 0.273 | 0.107 | 0.167 | 0.141 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.591 | 0.398 | 0.104 | 0.725 | 0.002 | 0.531 | 0.385 | 0.107 | 0.686 |
| layer1.0 | 0.437 | 0.365 | 0.113 | 0.664 | 0.002 | 0.426 | 0.334 | 0.110 | 0.640 |
| layer1.5 | 0.396 | 0.339 | 0.117 | 0.605 | 0.002 | 0.382 | 0.335 | 0.125 | 0.580 |
| layer1.8 | 0.254 | 0.280 | 0.114 | 0.496 | 0.002 | 0.241 | 0.253 | 0.117 | 0.446 |
| layer2.0 | 0.303 | 0.289 | 0.109 | 0.548 | 0.002 | 0.267 | 0.250 | 0.109 | 0.515 |
| layer2.5 | 0.186 | 0.234 | 0.102 | 0.336 | 0.002 | 0.213 | 0.194 | 0.095 | 0.317 |
| layer2.8 | 0.191 | 0.242 | 0.104 | 0.344 | 0.002 | 0.228 | 0.212 | 0.099 | 0.330 |
| layer3.0 | 0.172 | 0.202 | 0.106 | 0.284 | 0.002 | 0.179 | 0.171 | 0.101 | 0.272 |
| layer3.5 | 0.098 | 0.175 | 0.104 | 0.230 | 0.002 | 0.115 | 0.152 | 0.101 | 0.223 |
| layer3.8 | 0.030 | 0.157 | 0.101 | 0.129 | 0.002 | 0.025 | 0.110 | 0.085 | 0.158 |
| penult | 0.030 | 0.157 | 0.101 | 0.129 | 0.002 | 0.025 | 0.110 | 0.085 | 0.158 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | -0.126 | -0.985 | +0.000 | +0.012 | -83.797 | +0.081 | +0.140 | -0.032 | · | · | · | · | · |
| layer1.0 | -0.216 | -1.284 | +0.000 | +0.012 | +123.584 | +0.013 | +0.088 | -0.031 | · | · | · | · | · |
| layer1.5 | -0.644 | -3.505 | -3.000 | -0.069 | +276.744 | +0.184 | -0.084 | -0.115 | · | · | · | · | · |
| layer1.8 | -1.417 | -3.922 | -5.000 | -0.160 | +302.918 | +0.195 | -0.187 | -0.173 | · | · | · | · | · |
| layer2.0 | +2.483 | -3.399 | +1.000 | -0.175 | +89.818 | +0.092 | +0.482 | -0.176 | · | · | · | · | · |
| layer2.5 | +1.856 | -7.219 | -8.000 | -0.259 | +102.568 | +0.211 | +0.427 | -0.318 | · | · | · | · | · |
| layer2.8 | +1.795 | -8.366 | -9.000 | -0.297 | +109.099 | +0.245 | +0.280 | -0.374 | · | · | · | · | · |
| layer3.0 | +12.745 | -7.143 | +16.000 | -0.418 | +84.906 | +0.270 | +2.752 | -0.439 | · | · | · | · | · |
| layer3.5 | +8.826 | -13.852 | -11.000 | -0.801 | +127.152 | +0.496 | +1.181 | -0.676 | · | · | · | · | · |
| layer3.8 | +17.610 | -7.327 | +19.000 | -5.249 | +151.325 | +0.740 | +2.203 | -0.788 | · | · | · | · | · |
| penult | +17.610 | -7.327 | +19.000 | -5.249 | +151.325 | +0.740 | +2.203 | -0.788 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer2.0 | stem |
| spectral_slope | layer2.0 | stem |
| spectral_anisotropy | layer3.0 | stem |
| noise_sigma | layer1.0 | layer1.5 |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | layer1.0 | stem |
| edge_density | layer1.0 | stem |
| orientation_entropy | layer2.0 | stem |
| blockiness | layer3.0 | layer1.5 |
| class | layer3.5 | stem |
| coarse_animal_vehicle | layer3.5 | stem |
| corruption_family | layer2.0 | layer1.5 |
| corruption_type | layer2.0 | layer1.0 |
| severity | layer3.0 | stem |