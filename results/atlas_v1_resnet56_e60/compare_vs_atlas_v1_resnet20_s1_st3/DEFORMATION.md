# DEFORMATION  A=`results/atlas_v1_resnet20_s1_st3`  B=`results/atlas_v1_resnet56_e60`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.029 | 0.977 | 0.979 | True | 0.963 | 0.036 | 0.924 | 0.803 | 0.578 |
| layer1.0 | layer1.0 | 0.014 | 0.981 | 0.997 | True | 0.915 | 0.050 | 0.844 | 0.857 | 0.609 |
| layer1.1 | layer1.5 | 0.020 | 0.985 | 0.951 | True | 0.983 | 0.034 | 0.850 | 0.880 | 0.656 |
| layer1.2 | layer1.8 | 0.022 | 0.965 | 0.978 | True | 0.994 | 0.032 | 0.923 | 0.918 | 0.719 |
| layer2.0 | layer2.0 | 0.018 | 0.969 | 0.895 | True | 0.992 | 0.023 | 0.915 | 0.947 | 0.672 |
| layer2.1 | layer2.5 | 0.008 | 0.984 | 0.960 | True | 0.986 | 0.021 | 0.936 | 0.952 | 0.703 |
| layer2.2 | layer2.8 | 0.013 | 0.984 | 0.505 | True | 0.996 | 0.019 | 0.929 | 0.955 | 0.750 |
| layer3.0 | layer3.0 | 0.014 | 0.974 | 0.977 | True | 0.990 | 0.028 | 0.858 | 0.921 | 0.797 |
| layer3.1 | layer3.5 | 0.049 | 0.901 | 0.911 | True | 0.990 | 0.037 | 0.800 | 0.887 | 0.875 |
| layer3.2 | layer3.8 | 0.003 | 0.941 | 0.802 | True | 0.967 | 0.032 | 0.910 | 0.958 | 1.000 |
| penult | penult | 0.003 | 0.941 | 0.802 | True | 0.967 | 0.032 | 0.910 | 0.958 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.856 | 0.636 | 0.113 | 0.796 | 0.537 | 0.854 | 0.589 | 0.115 | 0.790 |
| layer1.0 | 0.889 | 0.671 | 0.118 | 0.901 | 0.537 | 0.820 | 0.642 | 0.120 | 0.873 |
| layer1.1 | 0.857 | 0.645 | 0.111 | 0.887 | 0.537 | 0.808 | 0.615 | 0.115 | 0.854 |
| layer1.2 | 0.928 | 0.743 | 0.110 | 0.911 | 0.537 | 0.915 | 0.724 | 0.119 | 0.913 |
| layer2.0 | 0.930 | 0.769 | 0.106 | 0.943 | 0.537 | 0.908 | 0.740 | 0.114 | 0.935 |
| layer2.1 | 0.937 | 0.782 | 0.105 | 0.950 | 0.537 | 0.918 | 0.719 | 0.112 | 0.941 |
| layer2.2 | 0.927 | 0.773 | 0.104 | 0.949 | 0.537 | 0.915 | 0.724 | 0.112 | 0.935 |
| layer3.0 | 0.884 | 0.780 | 0.102 | 0.925 | 0.537 | 0.856 | 0.678 | 0.110 | 0.909 |
| layer3.1 | 0.762 | 0.833 | 0.102 | 0.778 | 0.537 | 0.744 | 0.590 | 0.132 | 0.744 |
| layer3.2 | 0.875 | 0.924 | 0.100 | 0.788 | 0.537 | 0.671 | 0.570 | 0.126 | 0.748 |
| penult | 0.875 | 0.924 | 0.100 | 0.788 | 0.537 | 0.671 | 0.570 | 0.126 | 0.748 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | +0.260 | -0.020 | +0.000 | +0.006 | -115.767 | -0.014 | +0.030 | +0.000 | · | · | · | · | · |
| layer1.0 | -0.426 | -0.194 | +0.000 | -0.009 | -21.971 | -0.004 | -0.097 | -0.016 | · | · | · | · | · |
| layer1.1 | +0.391 | +0.749 | +1.000 | +0.039 | +3.262 | -0.059 | +0.080 | +0.023 | · | · | · | · | · |
| layer1.2 | +0.248 | +0.501 | +1.000 | +0.024 | -4.325 | +0.022 | +0.101 | +0.032 | · | · | · | · | · |
| layer2.0 | -0.818 | -0.534 | -2.000 | +0.025 | +1.987 | -0.068 | -0.123 | +0.003 | · | · | · | · | · |
| layer2.1 | +0.004 | -0.818 | -1.000 | +0.030 | -4.777 | +0.012 | -0.146 | +0.018 | · | · | · | · | · |
| layer2.2 | -0.246 | -1.502 | -2.000 | +0.031 | -1.990 | +0.046 | -0.179 | +0.007 | · | · | · | · | · |
| layer3.0 | -2.922 | -2.275 | -17.000 | -0.032 | +1.482 | +0.028 | -0.463 | -0.062 | · | · | · | · | · |
| layer3.1 | -0.016 | +2.596 | +2.000 | -0.089 | +0.063 | -0.043 | +0.098 | -0.024 | · | · | · | · | · |
| layer3.2 | +1.010 | +0.303 | +0.000 | +0.301 | -0.034 | -0.015 | +0.042 | +0.004 | · | · | · | · | · |
| penult | +1.010 | +0.303 | +0.000 | +0.301 | -0.034 | -0.015 | +0.042 | +0.004 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer1.1 | layer1.5 |
| spectral_slope | layer2.0 | layer1.5 |
| spectral_anisotropy | layer1.2 | layer2.5 |
| noise_sigma | layer1.0 | layer1.5 |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | stem | layer1.0 |
| edge_density | layer1.0 | layer1.0 |
| orientation_entropy | layer2.0 | layer2.5 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.5 |
| coarse_animal_vehicle | layer3.0 | layer3.5 |
| corruption_family | layer2.0 | layer2.0 |
| corruption_type | layer2.0 | layer2.0 |
| severity | layer3.0 | layer3.0 |