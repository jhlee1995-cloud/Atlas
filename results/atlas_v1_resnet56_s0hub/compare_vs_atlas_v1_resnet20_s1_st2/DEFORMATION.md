# DEFORMATION  A=`results/atlas_v1_resnet20_s1_st2`  B=`results/atlas_v1_resnet56_s0hub`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.003 | 0.995 | 0.995 | True | 0.971 | 0.025 | 0.973 | 0.987 | 0.875 |
| layer1.0 | layer1.0 | 0.012 | 0.984 | 1.000 | True | 0.975 | 0.049 | 0.907 | 0.966 | 0.688 |
| layer1.1 | layer1.5 | 0.022 | 0.970 | 0.951 | True | 0.983 | 0.026 | 0.921 | 0.908 | 0.641 |
| layer1.2 | layer1.8 | 0.011 | 0.984 | 0.987 | True | 0.998 | 0.020 | 0.957 | 0.960 | 0.797 |
| layer2.0 | layer2.0 | 0.014 | 0.977 | 0.904 | True | 0.992 | 0.022 | 0.951 | 0.949 | 0.781 |
| layer2.1 | layer2.5 | 0.012 | 0.981 | 0.980 | True | 0.988 | 0.022 | 0.943 | 0.948 | 0.766 |
| layer2.2 | layer2.8 | 0.010 | 0.981 | 0.559 | True | 0.990 | 0.019 | 0.935 | 0.955 | 0.828 |
| layer3.0 | layer3.0 | 0.014 | 0.957 | 0.980 | True | 0.957 | 0.041 | 0.889 | 0.941 | 0.812 |
| layer3.1 | layer3.5 | 0.055 | 0.860 | 0.871 | True | 0.946 | 0.052 | 0.776 | 0.873 | 0.828 |
| layer3.2 | layer3.8 | 0.010 | 0.909 | 0.751 | True | 0.957 | 0.027 | 0.892 | 0.944 | 1.000 |
| penult | penult | 0.010 | 0.909 | 0.751 | True | 0.957 | 0.027 | 0.892 | 0.944 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.977 | 0.818 | 0.117 | 0.982 | 0.534 | 0.973 | 0.819 | 0.120 | 0.976 |
| layer1.0 | 0.930 | 0.726 | 0.113 | 0.961 | 0.534 | 0.856 | 0.700 | 0.116 | 0.945 |
| layer1.1 | 0.908 | 0.670 | 0.114 | 0.898 | 0.534 | 0.890 | 0.628 | 0.116 | 0.881 |
| layer1.2 | 0.957 | 0.794 | 0.111 | 0.951 | 0.534 | 0.950 | 0.775 | 0.120 | 0.950 |
| layer2.0 | 0.952 | 0.786 | 0.105 | 0.951 | 0.534 | 0.947 | 0.770 | 0.112 | 0.944 |
| layer2.1 | 0.942 | 0.795 | 0.104 | 0.951 | 0.534 | 0.934 | 0.760 | 0.110 | 0.942 |
| layer2.2 | 0.933 | 0.791 | 0.104 | 0.955 | 0.534 | 0.921 | 0.753 | 0.111 | 0.948 |
| layer3.0 | 0.895 | 0.804 | 0.102 | 0.932 | 0.534 | 0.878 | 0.713 | 0.110 | 0.919 |
| layer3.1 | 0.765 | 0.852 | 0.101 | 0.759 | 0.534 | 0.735 | 0.625 | 0.140 | 0.726 |
| layer3.2 | 0.866 | 0.939 | 0.100 | 0.725 | 0.534 | 0.649 | 0.573 | 0.124 | 0.722 |
| penult | 0.866 | 0.939 | 0.100 | 0.725 | 0.534 | 0.649 | 0.573 | 0.124 | 0.722 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | -0.180 | -0.051 | +0.000 | +0.001 | +233.321 | +0.016 | -0.032 | +0.004 | · | · | · | · | · |
| layer1.0 | -0.791 | -0.103 | +0.000 | -0.023 | +7.327 | +0.008 | -0.154 | -0.032 | · | · | · | · | · |
| layer1.1 | +0.107 | +1.048 | +1.000 | +0.026 | -20.696 | -0.035 | +0.119 | +0.011 | · | · | · | · | · |
| layer1.2 | +0.553 | +0.691 | +1.000 | +0.015 | +7.273 | +0.019 | +0.143 | +0.018 | · | · | · | · | · |
| layer2.0 | -0.363 | +0.174 | -1.000 | -0.006 | +0.282 | -0.062 | -0.014 | -0.017 | · | · | · | · | · |
| layer2.1 | +0.770 | +0.989 | +2.000 | -0.000 | -6.132 | -0.010 | +0.084 | +0.024 | · | · | · | · | · |
| layer2.2 | +0.589 | +1.295 | +1.000 | +0.018 | -2.953 | -0.008 | +0.052 | +0.033 | · | · | · | · | · |
| layer3.0 | -2.440 | -0.253 | -13.000 | -0.022 | +1.111 | -0.003 | -0.287 | -0.026 | · | · | · | · | · |
| layer3.1 | -0.052 | +4.568 | +3.000 | +0.021 | -0.104 | -0.079 | -0.019 | +0.024 | · | · | · | · | · |
| layer3.2 | +0.913 | +0.421 | +0.000 | +2.266 | -0.114 | -0.044 | +0.107 | +0.023 | · | · | · | · | · |
| penult | +0.913 | +0.421 | +0.000 | +2.266 | -0.114 | -0.044 | +0.107 | +0.023 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer1.1 | layer2.0 |
| spectral_slope | layer2.0 | layer2.0 |
| spectral_anisotropy | layer1.2 | layer3.0 |
| noise_sigma | layer1.0 | layer1.0 |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | stem | layer1.0 |
| edge_density | layer1.0 | layer1.0 |
| orientation_entropy | layer2.0 | layer2.0 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.5 |
| coarse_animal_vehicle | layer3.0 | layer3.5 |
| corruption_family | layer2.0 | layer2.0 |
| corruption_type | layer2.0 | layer2.0 |
| severity | layer3.0 | layer3.0 |