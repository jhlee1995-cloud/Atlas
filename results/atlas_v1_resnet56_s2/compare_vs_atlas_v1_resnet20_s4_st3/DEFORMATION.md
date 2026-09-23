# DEFORMATION  A=`results/atlas_v1_resnet20_s4_st3`  B=`results/atlas_v1_resnet56_s2`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.011 | 0.980 | 0.990 | True | 0.955 | 0.032 | 0.937 | 0.978 | 0.750 |
| layer1.0 | layer1.0 | 0.024 | 0.979 | 1.000 | True | 0.953 | 0.038 | 0.888 | 0.920 | 0.594 |
| layer1.1 | layer1.5 | 0.041 | 0.958 | 0.937 | True | 1.000 | 0.032 | 0.857 | 0.904 | 0.688 |
| layer1.2 | layer1.8 | 0.016 | 0.969 | 0.960 | True | 0.988 | 0.029 | 0.950 | 0.951 | 0.734 |
| layer2.0 | layer2.0 | 0.005 | 0.986 | 0.969 | True | 0.994 | 0.020 | 0.971 | 0.979 | 0.891 |
| layer2.1 | layer2.5 | 0.012 | 0.984 | 0.494 | True | 0.996 | 0.017 | 0.954 | 0.959 | 0.797 |
| layer2.2 | layer2.8 | 0.011 | 0.985 | 0.540 | True | 0.981 | 0.021 | 0.941 | 0.966 | 0.797 |
| layer3.0 | layer3.0 | 0.019 | 0.975 | 0.929 | True | 0.959 | 0.034 | 0.895 | 0.940 | 0.750 |
| layer3.1 | layer3.5 | 0.089 | 0.792 | 0.719 | True | 0.990 | 0.028 | 0.762 | 0.822 | 0.844 |
| layer3.2 | layer3.8 | 0.009 | 0.906 | 0.786 | False | 0.973 | 0.024 | 0.912 | 0.951 | 1.000 |
| penult | penult | 0.009 | 0.906 | 0.786 | False | 0.973 | 0.024 | 0.912 | 0.951 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.956 | 0.804 | 0.114 | 0.971 | 0.550 | 0.926 | 0.805 | 0.117 | 0.964 |
| layer1.0 | 0.893 | 0.649 | 0.119 | 0.927 | 0.550 | 0.867 | 0.643 | 0.123 | 0.917 |
| layer1.1 | 0.856 | 0.681 | 0.114 | 0.881 | 0.550 | 0.795 | 0.628 | 0.115 | 0.844 |
| layer1.2 | 0.951 | 0.759 | 0.110 | 0.946 | 0.550 | 0.931 | 0.754 | 0.121 | 0.940 |
| layer2.0 | 0.966 | 0.841 | 0.106 | 0.974 | 0.550 | 0.959 | 0.808 | 0.115 | 0.967 |
| layer2.1 | 0.944 | 0.775 | 0.105 | 0.947 | 0.550 | 0.935 | 0.734 | 0.112 | 0.939 |
| layer2.2 | 0.934 | 0.786 | 0.104 | 0.954 | 0.550 | 0.925 | 0.720 | 0.113 | 0.947 |
| layer3.0 | 0.897 | 0.807 | 0.102 | 0.927 | 0.550 | 0.882 | 0.700 | 0.111 | 0.919 |
| layer3.1 | 0.710 | 0.854 | 0.101 | 0.704 | 0.550 | 0.711 | 0.560 | 0.131 | 0.689 |
| layer3.2 | 0.870 | 0.933 | 0.100 | 0.770 | 0.550 | 0.650 | 0.586 | 0.123 | 0.731 |
| penult | 0.870 | 0.933 | 0.100 | 0.770 | 0.550 | 0.650 | 0.586 | 0.123 | 0.731 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | +0.034 | -0.350 | +0.000 | -0.009 | +54.775 | -0.004 | -0.037 | +0.000 | · | · | · | · | · |
| layer1.0 | -1.093 | -0.373 | -2.000 | -0.022 | +39.250 | +0.082 | -0.351 | -0.030 | · | · | · | · | · |
| layer1.1 | +0.073 | +0.778 | +1.000 | +0.082 | -0.174 | +0.014 | -0.023 | +0.043 | · | · | · | · | · |
| layer1.2 | +0.365 | +0.379 | +1.000 | +0.044 | +17.715 | -0.009 | -0.019 | +0.029 | · | · | · | · | · |
| layer2.0 | -0.528 | -0.161 | -1.000 | +0.015 | -4.063 | -0.002 | -0.016 | +0.002 | · | · | · | · | · |
| layer2.1 | -0.152 | +0.107 | +0.000 | +0.021 | -5.601 | +0.007 | -0.045 | +0.037 | · | · | · | · | · |
| layer2.2 | -0.096 | +0.201 | -1.000 | +0.028 | -3.418 | -0.012 | -0.004 | +0.030 | · | · | · | · | · |
| layer3.0 | -2.773 | -1.106 | -13.000 | -0.046 | +0.939 | +0.027 | -0.398 | -0.033 | · | · | · | · | · |
| layer3.1 | -0.032 | +5.277 | +1.000 | +0.101 | -0.182 | -0.109 | -0.114 | +0.015 | · | · | · | · | · |
| layer3.2 | +0.558 | +0.440 | +0.000 | +2.188 | -0.117 | -0.036 | +0.051 | +0.012 | · | · | · | · | · |
| penult | +0.558 | +0.440 | +0.000 | +2.188 | -0.117 | -0.036 | +0.051 | +0.012 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer2.0 | layer2.0 |
| spectral_slope | layer2.0 | layer2.0 |
| spectral_anisotropy | layer2.0 | layer2.0 |
| noise_sigma | layer1.0 | layer1.0 |
| saturation_mean | stem | stem |
| hue_cos | stem | layer1.0 |
| hue_sin | stem | stem |
| colorfulness | stem | layer1.0 |
| edge_density | layer1.0 | stem |
| orientation_entropy | layer2.0 | layer2.0 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.5 |
| coarse_animal_vehicle | layer3.0 | layer3.0 |
| corruption_family | layer2.0 | layer2.0 |
| corruption_type | layer2.0 | layer1.8 |
| severity | layer2.0 | layer2.0 |