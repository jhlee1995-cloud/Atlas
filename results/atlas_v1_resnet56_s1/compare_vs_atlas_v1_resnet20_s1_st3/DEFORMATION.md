# DEFORMATION  A=`results/atlas_v1_resnet20_s1_st3`  B=`results/atlas_v1_resnet56_s1`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.009 | 0.992 | 0.982 | True | 0.967 | 0.034 | 0.966 | 0.970 | 0.875 |
| layer1.0 | layer1.0 | 0.019 | 0.985 | 0.984 | True | 0.920 | 0.061 | 0.934 | 0.940 | 0.703 |
| layer1.1 | layer1.5 | 0.019 | 0.973 | 0.951 | True | 0.996 | 0.030 | 0.904 | 0.925 | 0.625 |
| layer1.2 | layer1.8 | 0.014 | 0.973 | 0.978 | True | 0.998 | 0.018 | 0.949 | 0.915 | 0.688 |
| layer2.0 | layer2.0 | 0.008 | 0.984 | 0.946 | True | 0.994 | 0.014 | 0.956 | 0.958 | 0.812 |
| layer2.1 | layer2.5 | 0.008 | 0.988 | 0.919 | True | 0.994 | 0.020 | 0.952 | 0.958 | 0.766 |
| layer2.2 | layer2.8 | 0.010 | 0.982 | 0.501 | True | 0.988 | 0.020 | 0.933 | 0.951 | 0.766 |
| layer3.0 | layer3.0 | 0.010 | 0.988 | 0.974 | True | 0.955 | 0.052 | 0.880 | 0.942 | 0.828 |
| layer3.1 | layer3.5 | 0.061 | 0.811 | 0.763 | True | 0.944 | 0.060 | 0.796 | 0.878 | 0.859 |
| layer3.2 | layer3.8 | 0.008 | 0.917 | 0.934 | True | 0.944 | 0.025 | 0.898 | 0.949 | 1.000 |
| penult | penult | 0.008 | 0.917 | 0.934 | True | 0.944 | 0.025 | 0.898 | 0.949 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.957 | 0.784 | 0.119 | 0.962 | 0.523 | 0.946 | 0.791 | 0.122 | 0.955 |
| layer1.0 | 0.900 | 0.729 | 0.116 | 0.949 | 0.523 | 0.833 | 0.713 | 0.117 | 0.934 |
| layer1.1 | 0.916 | 0.673 | 0.115 | 0.920 | 0.523 | 0.891 | 0.620 | 0.118 | 0.901 |
| layer1.2 | 0.950 | 0.744 | 0.110 | 0.901 | 0.523 | 0.946 | 0.736 | 0.118 | 0.895 |
| layer2.0 | 0.958 | 0.801 | 0.106 | 0.961 | 0.523 | 0.943 | 0.781 | 0.111 | 0.950 |
| layer2.1 | 0.950 | 0.832 | 0.105 | 0.964 | 0.523 | 0.941 | 0.765 | 0.110 | 0.955 |
| layer2.2 | 0.936 | 0.788 | 0.103 | 0.956 | 0.523 | 0.929 | 0.750 | 0.110 | 0.949 |
| layer3.0 | 0.898 | 0.784 | 0.102 | 0.941 | 0.523 | 0.871 | 0.702 | 0.110 | 0.925 |
| layer3.1 | 0.753 | 0.869 | 0.101 | 0.740 | 0.523 | 0.730 | 0.622 | 0.138 | 0.727 |
| layer3.2 | 0.870 | 0.940 | 0.100 | 0.755 | 0.523 | 0.656 | 0.585 | 0.126 | 0.737 |
| penult | 0.870 | 0.940 | 0.100 | 0.755 | 0.523 | 0.656 | 0.585 | 0.126 | 0.737 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | -0.515 | -0.350 | -1.000 | +0.000 | +238.177 | +0.028 | -0.082 | -0.009 | · | · | · | · | · |
| layer1.0 | -1.154 | -0.665 | -1.000 | -0.019 | +99.910 | +0.023 | -0.264 | -0.025 | · | · | · | · | · |
| layer1.1 | +0.210 | +0.683 | +0.000 | +0.026 | -5.818 | -0.034 | +0.055 | +0.011 | · | · | · | · | · |
| layer1.2 | +0.417 | +0.467 | +0.000 | +0.007 | +3.668 | +0.031 | +0.087 | +0.014 | · | · | · | · | · |
| layer2.0 | -0.670 | -0.213 | -2.000 | -0.026 | +7.153 | -0.021 | -0.089 | -0.022 | · | · | · | · | · |
| layer2.1 | +0.365 | -0.098 | +1.000 | -0.018 | -4.092 | +0.019 | -0.045 | -0.006 | · | · | · | · | · |
| layer2.2 | +0.385 | +0.185 | +1.000 | -0.002 | -1.912 | +0.037 | +0.214 | +0.014 | · | · | · | · | · |
| layer3.0 | -2.531 | -0.794 | -13.000 | -0.043 | +1.099 | +0.028 | -0.364 | -0.046 | · | · | · | · | · |
| layer3.1 | +0.186 | +4.401 | +3.000 | -0.012 | -0.098 | -0.072 | +0.145 | +0.011 | · | · | · | · | · |
| layer3.2 | +0.625 | +0.347 | +0.000 | +2.047 | -0.110 | -0.040 | +0.058 | +0.023 | · | · | · | · | · |
| penult | +0.625 | +0.347 | +0.000 | +2.047 | -0.110 | -0.040 | +0.058 | +0.023 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | layer1.5 |
| highfreq_ratio | layer1.1 | layer2.0 |
| spectral_slope | layer2.0 | layer2.0 |
| spectral_anisotropy | layer1.2 | layer2.0 |
| noise_sigma | layer1.0 | layer1.5 |
| saturation_mean | stem | layer1.0 |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | stem | layer1.0 |
| edge_density | layer1.0 | layer1.5 |
| orientation_entropy | layer2.0 | layer2.5 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.5 |
| coarse_animal_vehicle | layer3.0 | layer3.5 |
| corruption_family | layer2.0 | layer2.0 |
| corruption_type | layer2.0 | layer2.0 |
| severity | layer3.0 | layer2.0 |