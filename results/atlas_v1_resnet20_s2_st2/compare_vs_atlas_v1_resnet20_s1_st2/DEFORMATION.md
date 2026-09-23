# DEFORMATION  A=`results/atlas_v1_resnet20_s1_st2`  B=`results/atlas_v1_resnet20_s2_st2`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.016 | 0.989 | 0.984 | True | 0.963 | 0.024 | 0.975 | 0.944 | 0.703 |
| layer1.0 | layer1.0 | 0.016 | 0.987 | 0.995 | True | 0.990 | 0.027 | 0.911 | 0.911 | 0.688 |
| layer1.1 | layer1.1 | 0.014 | 0.977 | 0.734 | True | 0.992 | 0.028 | 0.892 | 0.863 | 0.656 |
| layer1.2 | layer1.2 | 0.011 | 0.980 | 0.987 | True | 0.998 | 0.019 | 0.945 | 0.942 | 0.703 |
| layer2.0 | layer2.0 | 0.013 | 0.974 | 0.915 | True | 0.996 | 0.019 | 0.954 | 0.963 | 0.734 |
| layer2.1 | layer2.1 | 0.007 | 0.986 | 0.959 | True | 0.996 | 0.015 | 0.945 | 0.959 | 0.812 |
| layer2.2 | layer2.2 | 0.010 | 0.980 | 0.997 | True | 0.994 | 0.017 | 0.935 | 0.953 | 0.766 |
| layer3.0 | layer3.0 | 0.014 | 0.967 | 0.987 | True | 0.963 | 0.024 | 0.897 | 0.953 | 0.766 |
| layer3.1 | layer3.1 | 0.028 | 0.945 | 0.930 | True | 0.961 | 0.031 | 0.867 | 0.906 | 0.875 |
| layer3.2 | layer3.2 | 0.002 | 0.973 | 0.838 | True | 0.971 | 0.025 | 0.910 | 0.963 | 1.000 |
| penult | penult | 0.002 | 0.973 | 0.838 | True | 0.971 | 0.025 | 0.910 | 0.963 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.933 | 0.735 | 0.111 | 0.934 | 0.535 | 0.925 | 0.749 | 0.117 | 0.929 |
| layer1.0 | 0.899 | 0.717 | 0.118 | 0.916 | 0.535 | 0.860 | 0.699 | 0.121 | 0.890 |
| layer1.1 | 0.878 | 0.655 | 0.115 | 0.871 | 0.535 | 0.847 | 0.619 | 0.117 | 0.856 |
| layer1.2 | 0.939 | 0.724 | 0.113 | 0.932 | 0.535 | 0.928 | 0.718 | 0.123 | 0.929 |
| layer2.0 | 0.951 | 0.791 | 0.106 | 0.958 | 0.535 | 0.938 | 0.772 | 0.114 | 0.952 |
| layer2.1 | 0.945 | 0.793 | 0.105 | 0.955 | 0.535 | 0.932 | 0.747 | 0.111 | 0.947 |
| layer2.2 | 0.933 | 0.784 | 0.103 | 0.948 | 0.535 | 0.924 | 0.743 | 0.110 | 0.941 |
| layer3.0 | 0.869 | 0.813 | 0.102 | 0.920 | 0.535 | 0.850 | 0.669 | 0.110 | 0.907 |
| layer3.1 | 0.857 | 0.855 | 0.101 | 0.847 | 0.535 | 0.762 | 0.615 | 0.137 | 0.849 |
| layer3.2 | 0.888 | 0.926 | 0.100 | 0.831 | 0.535 | 0.694 | 0.589 | 0.126 | 0.779 |
| penult | 0.888 | 0.926 | 0.100 | 0.831 | 0.535 | 0.694 | 0.589 | 0.126 | 0.779 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | +0.251 | -0.018 | +0.000 | +0.006 | -108.144 | +0.024 | +0.030 | +0.002 | · | · | · | · | · |
| layer1.0 | +0.623 | +0.587 | +2.000 | +0.002 | -28.459 | -0.022 | +0.206 | +0.008 | · | · | · | · | · |
| layer1.1 | +0.498 | +0.997 | +1.000 | +0.000 | -5.845 | +0.031 | +0.126 | +0.007 | · | · | · | · | · |
| layer1.2 | +0.535 | -0.300 | +1.000 | -0.005 | -1.476 | +0.024 | +0.108 | -0.003 | · | · | · | · | · |
| layer2.0 | -0.158 | -0.469 | -1.000 | -0.002 | -4.413 | -0.068 | -0.012 | -0.008 | · | · | · | · | · |
| layer2.1 | +0.306 | -0.581 | +0.000 | -0.014 | -5.939 | -0.001 | -0.048 | -0.008 | · | · | · | · | · |
| layer2.2 | +0.266 | -0.629 | +0.000 | -0.002 | -1.820 | +0.031 | +0.015 | -0.002 | · | · | · | · | · |
| layer3.0 | +0.272 | +1.805 | +0.000 | +0.009 | +0.143 | +0.005 | +0.010 | +0.018 | · | · | · | · | · |
| layer3.1 | +0.138 | -1.618 | -1.000 | -0.063 | +0.085 | +0.001 | -0.092 | -0.003 | · | · | · | · | · |
| layer3.2 | +0.279 | -0.137 | +0.000 | -0.045 | +0.007 | +0.009 | -0.066 | +0.003 | · | · | · | · | · |
| penult | +0.279 | -0.137 | +0.000 | -0.045 | +0.007 | +0.009 | -0.066 | +0.003 | · | · | · | · | · |

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
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | stem | layer1.0 |
| edge_density | layer1.0 | layer1.0 |
| orientation_entropy | layer2.0 | layer2.0 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.1 |
| coarse_animal_vehicle | layer3.0 | layer3.0 |
| corruption_family | layer2.0 | layer1.2 |
| corruption_type | layer2.0 | layer2.0 |
| severity | layer3.0 | layer2.2 |