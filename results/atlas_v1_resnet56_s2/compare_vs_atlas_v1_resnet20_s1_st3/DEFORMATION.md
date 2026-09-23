# DEFORMATION  A=`results/atlas_v1_resnet20_s1_st3`  B=`results/atlas_v1_resnet56_s2`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.006 | 0.986 | 0.992 | True | 0.950 | 0.033 | 0.956 | 0.984 | 0.875 |
| layer1.0 | layer1.0 | 0.009 | 0.983 | 0.995 | True | 0.948 | 0.033 | 0.927 | 0.965 | 0.734 |
| layer1.1 | layer1.5 | 0.037 | 0.971 | 0.946 | True | 0.998 | 0.028 | 0.830 | 0.866 | 0.578 |
| layer1.2 | layer1.8 | 0.026 | 0.957 | 0.948 | True | 0.998 | 0.021 | 0.929 | 0.915 | 0.719 |
| layer2.0 | layer2.0 | 0.015 | 0.972 | 0.933 | True | 0.998 | 0.019 | 0.957 | 0.958 | 0.734 |
| layer2.1 | layer2.5 | 0.012 | 0.984 | 0.547 | True | 0.988 | 0.018 | 0.952 | 0.948 | 0.703 |
| layer2.2 | layer2.8 | 0.010 | 0.987 | 0.536 | True | 0.979 | 0.020 | 0.938 | 0.944 | 0.703 |
| layer3.0 | layer3.0 | 0.011 | 0.974 | 1.000 | True | 0.990 | 0.029 | 0.889 | 0.951 | 0.766 |
| layer3.1 | layer3.5 | 0.063 | 0.802 | 0.814 | True | 0.977 | 0.035 | 0.793 | 0.858 | 0.859 |
| layer3.2 | layer3.8 | 0.007 | 0.902 | 0.931 | False | 0.969 | 0.024 | 0.900 | 0.952 | 1.000 |
| penult | penult | 0.007 | 0.902 | 0.931 | False | 0.969 | 0.024 | 0.900 | 0.952 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.971 | 0.820 | 0.116 | 0.978 | 0.489 | 0.960 | 0.839 | 0.119 | 0.971 |
| layer1.0 | 0.923 | 0.782 | 0.118 | 0.971 | 0.489 | 0.885 | 0.747 | 0.122 | 0.957 |
| layer1.1 | 0.822 | 0.554 | 0.112 | 0.851 | 0.489 | 0.764 | 0.528 | 0.113 | 0.811 |
| layer1.2 | 0.929 | 0.730 | 0.110 | 0.905 | 0.489 | 0.915 | 0.716 | 0.120 | 0.905 |
| layer2.0 | 0.955 | 0.781 | 0.106 | 0.954 | 0.489 | 0.943 | 0.744 | 0.112 | 0.947 |
| layer2.1 | 0.946 | 0.784 | 0.105 | 0.941 | 0.489 | 0.934 | 0.724 | 0.110 | 0.935 |
| layer2.2 | 0.934 | 0.778 | 0.104 | 0.946 | 0.489 | 0.920 | 0.725 | 0.111 | 0.938 |
| layer3.0 | 0.903 | 0.810 | 0.102 | 0.937 | 0.489 | 0.880 | 0.701 | 0.109 | 0.925 |
| layer3.1 | 0.752 | 0.863 | 0.101 | 0.712 | 0.489 | 0.750 | 0.577 | 0.130 | 0.708 |
| layer3.2 | 0.871 | 0.927 | 0.100 | 0.773 | 0.489 | 0.643 | 0.585 | 0.125 | 0.731 |
| penult | 0.871 | 0.927 | 0.100 | 0.773 | 0.489 | 0.643 | 0.585 | 0.125 | 0.731 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | +0.148 | -0.253 | +0.000 | -0.000 | -47.111 | +0.010 | -0.014 | +0.006 | · | · | · | · | · |
| layer1.0 | -0.420 | +0.289 | +0.000 | -0.009 | -31.497 | -0.034 | -0.147 | -0.015 | · | · | · | · | · |
| layer1.1 | +0.653 | +1.601 | +2.000 | +0.088 | +12.365 | -0.071 | +0.133 | +0.049 | · | · | · | · | · |
| layer1.2 | +1.108 | +0.682 | +2.000 | +0.033 | +16.787 | -0.006 | +0.190 | +0.032 | · | · | · | · | · |
| layer2.0 | -0.059 | -0.083 | -1.000 | +0.012 | -7.012 | -0.062 | +0.068 | -0.001 | · | · | · | · | · |
| layer2.1 | +0.640 | -0.094 | +1.000 | +0.000 | -8.009 | +0.028 | +0.057 | +0.032 | · | · | · | · | · |
| layer2.2 | +0.442 | +0.219 | +0.000 | +0.024 | -3.771 | +0.012 | -0.010 | +0.037 | · | · | · | · | · |
| layer3.0 | -2.477 | -0.958 | -14.000 | -0.039 | +0.945 | +0.036 | -0.215 | -0.036 | · | · | · | · | · |
| layer3.1 | +0.025 | +3.470 | +1.000 | +0.061 | -0.164 | -0.102 | -0.152 | +0.024 | · | · | · | · | · |
| layer3.2 | +0.703 | +0.374 | +0.000 | +2.146 | -0.113 | -0.036 | +0.022 | +0.018 | · | · | · | · | · |
| penult | +0.703 | +0.374 | +0.000 | +2.146 | -0.113 | -0.036 | +0.022 | +0.018 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer1.1 | layer2.0 |
| spectral_slope | layer2.0 | layer2.0 |
| spectral_anisotropy | layer1.2 | layer2.0 |
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
| severity | layer3.0 | layer2.0 |