# DEFORMATION  A=`results/atlas_v1_resnet20_s2_st3`  B=`results/atlas_v1_resnet20_s3_st3`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.026 | 0.978 | 0.987 | True | 0.977 | 0.024 | 0.956 | 0.885 | 0.688 |
| layer1.0 | layer1.0 | 0.015 | 0.984 | 1.000 | True | 0.984 | 0.033 | 0.901 | 0.938 | 0.656 |
| layer1.1 | layer1.1 | 0.025 | 0.949 | 0.695 | True | 0.990 | 0.024 | 0.914 | 0.924 | 0.750 |
| layer1.2 | layer1.2 | 0.010 | 0.974 | 0.989 | True | 0.992 | 0.021 | 0.948 | 0.952 | 0.734 |
| layer2.0 | layer2.0 | 0.006 | 0.987 | 0.953 | True | 0.996 | 0.017 | 0.968 | 0.981 | 0.781 |
| layer2.1 | layer2.1 | 0.009 | 0.979 | 0.979 | True | 0.998 | 0.015 | 0.949 | 0.971 | 0.844 |
| layer2.2 | layer2.2 | 0.011 | 0.977 | 0.997 | True | 0.996 | 0.022 | 0.943 | 0.969 | 0.859 |
| layer3.0 | layer3.0 | 0.009 | 0.984 | 0.974 | True | 0.969 | 0.017 | 0.919 | 0.957 | 0.781 |
| layer3.1 | layer3.1 | 0.049 | 0.880 | 0.901 | True | 0.981 | 0.016 | 0.811 | 0.887 | 0.875 |
| layer3.2 | layer3.2 | 0.003 | 0.971 | 0.939 | True | 0.950 | 0.025 | 0.916 | 0.967 | 1.000 |
| penult | penult | 0.003 | 0.971 | 0.939 | True | 0.950 | 0.025 | 0.916 | 0.967 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.877 | 0.687 | 0.111 | 0.861 | 0.527 | 0.862 | 0.682 | 0.117 | 0.841 |
| layer1.0 | 0.913 | 0.659 | 0.124 | 0.907 | 0.527 | 0.882 | 0.663 | 0.128 | 0.894 |
| layer1.1 | 0.885 | 0.726 | 0.118 | 0.896 | 0.527 | 0.850 | 0.672 | 0.117 | 0.867 |
| layer1.2 | 0.937 | 0.735 | 0.113 | 0.931 | 0.527 | 0.915 | 0.715 | 0.125 | 0.913 |
| layer2.0 | 0.968 | 0.837 | 0.107 | 0.978 | 0.527 | 0.957 | 0.822 | 0.116 | 0.971 |
| layer2.1 | 0.951 | 0.801 | 0.105 | 0.960 | 0.527 | 0.942 | 0.776 | 0.112 | 0.952 |
| layer2.2 | 0.937 | 0.802 | 0.103 | 0.947 | 0.527 | 0.924 | 0.750 | 0.109 | 0.934 |
| layer3.0 | 0.895 | 0.802 | 0.102 | 0.939 | 0.527 | 0.860 | 0.663 | 0.111 | 0.917 |
| layer3.1 | 0.807 | 0.856 | 0.101 | 0.826 | 0.527 | 0.756 | 0.622 | 0.128 | 0.808 |
| layer3.2 | 0.892 | 0.929 | 0.100 | 0.839 | 0.527 | 0.700 | 0.575 | 0.125 | 0.778 |
| penult | 0.892 | 0.929 | 0.100 | 0.839 | 0.527 | 0.700 | 0.575 | 0.125 | 0.778 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | -0.204 | -0.330 | +0.000 | -0.010 | +165.869 | -0.010 | +0.015 | -0.000 | · | · | · | · | · |
| layer1.0 | -0.370 | +0.428 | +0.000 | +0.009 | -13.077 | -0.052 | -0.099 | +0.002 | · | · | · | · | · |
| layer1.1 | -0.250 | -0.123 | +0.000 | +0.018 | +1.295 | -0.043 | +0.045 | +0.004 | · | · | · | · | · |
| layer1.2 | -0.099 | +0.787 | +0.000 | -0.005 | +1.143 | +0.013 | -0.054 | +0.001 | · | · | · | · | · |
| layer2.0 | +0.275 | +0.253 | +1.000 | -0.008 | +5.159 | +0.020 | +0.063 | -0.006 | · | · | · | · | · |
| layer2.1 | +0.072 | -0.130 | +1.000 | -0.012 | +4.875 | +0.047 | +0.056 | -0.012 | · | · | · | · | · |
| layer2.2 | +0.221 | +0.610 | +0.000 | -0.006 | +1.463 | -0.002 | -0.023 | -0.005 | · | · | · | · | · |
| layer3.0 | -0.303 | -0.558 | -1.000 | -0.027 | +0.208 | +0.021 | +0.169 | -0.026 | · | · | · | · | · |
| layer3.1 | +0.207 | +2.968 | +2.000 | +0.010 | +0.064 | -0.013 | -0.051 | -0.014 | · | · | · | · | · |
| layer3.2 | -0.199 | -0.056 | +0.000 | -0.051 | +0.002 | +0.002 | +0.006 | -0.001 | · | · | · | · | · |
| penult | -0.199 | -0.056 | +0.000 | -0.051 | +0.002 | +0.002 | +0.006 | -0.001 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer2.0 | layer1.0 |
| spectral_slope | layer2.0 | layer2.0 |
| spectral_anisotropy | layer2.0 | layer2.0 |
| noise_sigma | layer1.0 | layer1.0 |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | layer1.0 | layer1.0 |
| edge_density | layer1.0 | layer1.0 |
| orientation_entropy | layer2.0 | layer3.0 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.1 |
| coarse_animal_vehicle | layer3.0 | layer3.0 |
| corruption_family | layer1.2 | layer2.0 |
| corruption_type | layer2.0 | layer2.0 |
| severity | layer2.2 | layer3.0 |