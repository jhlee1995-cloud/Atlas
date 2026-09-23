# DEFORMATION  A=`results/atlas_v1_resnet20_s0hub_st3`  B=`results/atlas_v1_resnet56_s1`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.011 | 0.993 | 0.992 | True | 0.953 | 0.031 | 0.951 | 0.973 | 0.844 |
| layer1.0 | layer1.0 | 0.054 | 0.956 | 0.984 | True | 0.897 | 0.091 | 0.819 | 0.885 | 0.625 |
| layer1.1 | layer1.5 | 0.018 | 0.977 | 1.000 | True | 0.979 | 0.024 | 0.862 | 0.903 | 0.672 |
| layer1.2 | layer1.8 | 0.008 | 0.986 | 1.000 | True | 0.996 | 0.013 | 0.954 | 0.920 | 0.703 |
| layer2.0 | layer2.0 | 0.022 | 0.965 | 0.933 | True | 0.981 | 0.025 | 0.939 | 0.922 | 0.750 |
| layer2.1 | layer2.5 | 0.011 | 0.981 | 0.992 | True | 0.988 | 0.020 | 0.943 | 0.924 | 0.844 |
| layer2.2 | layer2.8 | 0.010 | 0.988 | 0.463 | True | 0.990 | 0.017 | 0.937 | 0.945 | 0.781 |
| layer3.0 | layer3.0 | 0.022 | 0.963 | 0.978 | True | 0.955 | 0.044 | 0.886 | 0.941 | 0.812 |
| layer3.1 | layer3.5 | 0.070 | 0.810 | 0.766 | True | 0.990 | 0.045 | 0.797 | 0.848 | 0.750 |
| layer3.2 | layer3.8 | 0.010 | 0.819 | 0.721 | True | 0.977 | 0.020 | 0.885 | 0.954 | 1.000 |
| penult | penult | 0.010 | 0.819 | 0.721 | True | 0.977 | 0.020 | 0.885 | 0.954 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.954 | 0.812 | 0.119 | 0.963 | 0.499 | 0.935 | 0.822 | 0.124 | 0.955 |
| layer1.0 | 0.876 | 0.690 | 0.118 | 0.898 | 0.499 | 0.801 | 0.679 | 0.119 | 0.876 |
| layer1.1 | 0.893 | 0.657 | 0.113 | 0.903 | 0.499 | 0.811 | 0.617 | 0.117 | 0.875 |
| layer1.2 | 0.962 | 0.766 | 0.111 | 0.918 | 0.499 | 0.948 | 0.737 | 0.121 | 0.910 |
| layer2.0 | 0.935 | 0.721 | 0.104 | 0.907 | 0.499 | 0.927 | 0.689 | 0.112 | 0.892 |
| layer2.1 | 0.936 | 0.750 | 0.104 | 0.926 | 0.499 | 0.925 | 0.690 | 0.110 | 0.909 |
| layer2.2 | 0.931 | 0.777 | 0.103 | 0.938 | 0.499 | 0.926 | 0.711 | 0.109 | 0.926 |
| layer3.0 | 0.896 | 0.789 | 0.102 | 0.925 | 0.499 | 0.883 | 0.690 | 0.110 | 0.915 |
| layer3.1 | 0.727 | 0.843 | 0.101 | 0.701 | 0.499 | 0.723 | 0.600 | 0.138 | 0.671 |
| layer3.2 | 0.868 | 0.930 | 0.100 | 0.759 | 0.499 | 0.656 | 0.574 | 0.126 | 0.739 |
| penult | 0.868 | 0.930 | 0.100 | 0.759 | 0.499 | 0.656 | 0.574 | 0.126 | 0.739 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | -0.614 | -0.336 | -1.000 | -0.001 | +239.319 | +0.035 | -0.115 | -0.008 | · | · | · | · | · |
| layer1.0 | -1.407 | -1.419 | -2.000 | -0.014 | +126.195 | +0.078 | -0.360 | -0.033 | · | · | · | · | · |
| layer1.1 | -0.388 | -0.418 | -1.000 | +0.025 | +10.514 | +0.016 | -0.099 | -0.007 | · | · | · | · | · |
| layer1.2 | -0.226 | -0.251 | -1.000 | +0.023 | -0.260 | -0.005 | -0.070 | +0.007 | · | · | · | · | · |
| layer2.0 | -1.177 | -0.962 | -3.000 | -0.014 | +9.090 | +0.039 | -0.127 | -0.027 | · | · | · | · | · |
| layer2.1 | -0.282 | -0.183 | +0.000 | -0.010 | +0.540 | -0.020 | -0.083 | -0.021 | · | · | · | · | · |
| layer2.2 | -0.130 | -0.243 | +0.000 | -0.011 | +0.201 | -0.008 | +0.399 | -0.001 | · | · | · | · | · |
| layer3.0 | -2.621 | -2.645 | -12.000 | -0.050 | +1.073 | +0.049 | -0.502 | -0.048 | · | · | · | · | · |
| layer3.1 | +0.278 | +5.193 | +3.000 | -0.038 | -0.083 | -0.059 | +0.135 | +0.021 | · | · | · | · | · |
| layer3.2 | +0.467 | +0.481 | +0.000 | +2.118 | -0.117 | -0.049 | +0.128 | +0.021 | · | · | · | · | · |
| penult | +0.467 | +0.481 | +0.000 | +2.118 | -0.117 | -0.049 | +0.128 | +0.021 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | layer1.5 |
| highfreq_ratio | layer1.0 | layer2.0 |
| spectral_slope | layer2.0 | layer2.0 |
| spectral_anisotropy | layer2.0 | layer2.0 |
| noise_sigma | layer1.0 | layer1.5 |
| saturation_mean | stem | layer1.0 |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | layer1.0 | layer1.0 |
| edge_density | layer1.0 | layer1.5 |
| orientation_entropy | layer2.0 | layer2.5 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.5 |
| coarse_animal_vehicle | layer3.0 | layer3.5 |
| corruption_family | layer2.0 | layer2.0 |
| corruption_type | layer1.2 | layer2.0 |
| severity | layer3.0 | layer2.0 |