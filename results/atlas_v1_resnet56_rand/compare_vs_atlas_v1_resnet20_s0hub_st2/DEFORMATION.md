# DEFORMATION  A=`results/atlas_v1_resnet20_s0hub_st2`  B=`results/atlas_v1_resnet56_rand`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.126 | 0.884 | 0.825 | True | 0.924 | 0.036 | 0.525 | 0.720 | 0.422 |
| layer1.0 | layer1.0 | 0.087 | 0.912 | 0.847 | True | 0.891 | 0.140 | 0.449 | 0.710 | 0.328 |
| layer1.1 | layer1.5 | 0.142 | 0.868 | 0.904 | True | 0.946 | 0.159 | 0.314 | 0.652 | 0.328 |
| layer1.2 | layer1.8 | 0.271 | 0.736 | 0.850 | False | 0.926 | 0.204 | 0.220 | 0.552 | 0.141 |
| layer2.0 | layer2.0 | 0.236 | 0.740 | 0.710 | True | 0.913 | 0.225 | 0.265 | 0.564 | 0.172 |
| layer2.1 | layer2.5 | 0.471 | 0.447 | 0.110 | True | 0.884 | 0.283 | 0.183 | 0.397 | 0.156 |
| layer2.2 | layer2.8 | 0.398 | 0.548 | 0.341 | True | 0.866 | 0.300 | 0.221 | 0.462 | 0.156 |
| layer3.0 | layer3.0 | 0.479 | 0.474 | 0.054 | True | 0.812 | 0.331 | 0.203 | 0.402 | 0.156 |
| layer3.1 | layer3.5 | 0.672 | 0.319 | -0.040 | False | 0.725 | 0.328 | 0.131 | 0.340 | 0.156 |
| layer3.2 | layer3.8 | 0.901 | 0.288 | -0.107 | False | 0.517 | 0.253 | 0.100 | 0.198 | 0.141 |
| penult | penult | 0.901 | 0.288 | -0.107 | False | 0.517 | 0.253 | 0.100 | 0.198 | 0.141 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.477 | 0.390 | 0.099 | 0.695 | 0.007 | 0.446 | 0.366 | 0.101 | 0.656 |
| layer1.0 | 0.467 | 0.425 | 0.114 | 0.744 | 0.007 | 0.445 | 0.395 | 0.117 | 0.736 |
| layer1.1 | 0.378 | 0.326 | 0.117 | 0.626 | 0.007 | 0.310 | 0.316 | 0.119 | 0.594 |
| layer1.2 | 0.248 | 0.278 | 0.114 | 0.513 | 0.007 | 0.221 | 0.247 | 0.116 | 0.460 |
| layer2.0 | 0.259 | 0.271 | 0.103 | 0.467 | 0.007 | 0.224 | 0.211 | 0.099 | 0.414 |
| layer2.1 | 0.150 | 0.214 | 0.095 | 0.278 | 0.007 | 0.189 | 0.150 | 0.081 | 0.246 |
| layer2.2 | 0.200 | 0.224 | 0.095 | 0.329 | 0.007 | 0.242 | 0.173 | 0.087 | 0.317 |
| layer3.0 | 0.132 | 0.186 | 0.103 | 0.246 | 0.007 | 0.154 | 0.148 | 0.093 | 0.231 |
| layer3.1 | 0.061 | 0.169 | 0.103 | 0.218 | 0.007 | 0.102 | 0.127 | 0.081 | 0.197 |
| layer3.2 | 0.029 | 0.160 | 0.101 | 0.157 | 0.007 | 0.025 | 0.108 | 0.084 | 0.157 |
| penult | 0.029 | 0.160 | 0.101 | 0.157 | 0.007 | 0.025 | 0.108 | 0.084 | 0.157 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | -0.405 | -1.022 | +0.000 | +0.012 | +150.666 | +0.103 | +0.074 | -0.027 | · | · | · | · | · |
| layer1.0 | -1.259 | -2.141 | -1.000 | -0.006 | +157.195 | +0.076 | -0.162 | -0.072 | · | · | · | · | · |
| layer1.1 | -1.136 | -3.558 | -3.000 | -0.044 | +272.379 | +0.200 | -0.119 | -0.122 | · | · | · | · | · |
| layer1.2 | -1.507 | -3.949 | -5.000 | -0.129 | +306.263 | +0.178 | -0.202 | -0.161 | · | · | · | · | · |
| layer2.0 | +1.614 | -3.974 | -1.000 | -0.170 | +92.037 | +0.090 | +0.430 | -0.198 | · | · | · | · | · |
| layer2.1 | +1.979 | -6.315 | -7.000 | -0.251 | +101.068 | +0.161 | +0.474 | -0.309 | · | · | · | · | · |
| layer2.2 | +1.869 | -7.499 | -9.000 | -0.289 | +108.259 | +0.192 | +0.518 | -0.355 | · | · | · | · | · |
| layer3.0 | +10.215 | -9.246 | +4.000 | -0.447 | +85.992 | +0.289 | +2.327 | -0.466 | · | · | · | · | · |
| layer3.1 | +8.866 | -8.492 | -8.000 | -0.805 | +127.064 | +0.431 | +1.153 | -0.642 | · | · | · | · | · |
| layer3.2 | +18.365 | -6.772 | +19.000 | -2.913 | +151.204 | +0.687 | +2.380 | -0.767 | · | · | · | · | · |
| penult | +18.365 | -6.772 | +19.000 | -2.913 | +151.204 | +0.687 | +2.380 | -0.767 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer1.0 | stem |
| spectral_slope | layer2.0 | stem |
| spectral_anisotropy | layer2.0 | stem |
| noise_sigma | layer1.0 | layer1.5 |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | layer1.0 | stem |
| edge_density | layer1.0 | stem |
| orientation_entropy | layer2.0 | stem |
| blockiness | layer3.0 | layer1.5 |
| class | layer3.1 | stem |
| coarse_animal_vehicle | layer3.0 | stem |
| corruption_family | layer2.0 | layer1.5 |
| corruption_type | layer1.2 | layer1.0 |
| severity | layer3.0 | stem |