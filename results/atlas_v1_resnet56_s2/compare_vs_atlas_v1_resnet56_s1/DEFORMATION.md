# DEFORMATION  A=`results/atlas_v1_resnet56_s1`  B=`results/atlas_v1_resnet56_s2`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.019 | 0.975 | 0.990 | True | 0.975 | 0.036 | 0.885 | 0.948 | 0.797 |
| layer1.0 | layer1.0 | 0.018 | 0.988 | 0.984 | True | 0.926 | 0.067 | 0.896 | 0.950 | 0.609 |
| layer1.5 | layer1.5 | 0.019 | 0.983 | 0.995 | True | 0.994 | 0.022 | 0.912 | 0.939 | 0.656 |
| layer1.8 | layer1.8 | 0.008 | 0.991 | 0.969 | True | 0.996 | 0.016 | 0.966 | 0.973 | 0.766 |
| layer2.0 | layer2.0 | 0.010 | 0.978 | 0.977 | True | 0.992 | 0.024 | 0.948 | 0.965 | 0.797 |
| layer2.5 | layer2.5 | 0.007 | 0.991 | 0.508 | True | 0.986 | 0.030 | 0.956 | 0.974 | 0.781 |
| layer2.8 | layer2.8 | 0.007 | 0.989 | 0.922 | True | 0.965 | 0.032 | 0.942 | 0.972 | 0.781 |
| layer3.0 | layer3.0 | 0.009 | 0.979 | 0.974 | True | 0.959 | 0.028 | 0.941 | 0.966 | 0.875 |
| layer3.5 | layer3.5 | 0.015 | 0.955 | 0.721 | True | 0.961 | 0.035 | 0.853 | 0.937 | 0.844 |
| layer3.8 | layer3.8 | 0.004 | 0.900 | 0.931 | False | 0.971 | 0.021 | 0.946 | 0.974 | 1.000 |
| penult | penult | 0.004 | 0.900 | 0.931 | False | 0.971 | 0.021 | 0.946 | 0.974 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.914 | 0.722 | 0.117 | 0.925 | 0.566 | 0.884 | 0.739 | 0.120 | 0.907 |
| layer1.0 | 0.918 | 0.725 | 0.118 | 0.958 | 0.566 | 0.864 | 0.709 | 0.119 | 0.940 |
| layer1.5 | 0.914 | 0.680 | 0.113 | 0.920 | 0.566 | 0.877 | 0.648 | 0.116 | 0.902 |
| layer1.8 | 0.969 | 0.790 | 0.109 | 0.960 | 0.566 | 0.955 | 0.776 | 0.120 | 0.949 |
| layer2.0 | 0.951 | 0.795 | 0.105 | 0.960 | 0.566 | 0.925 | 0.750 | 0.110 | 0.946 |
| layer2.5 | 0.956 | 0.809 | 0.104 | 0.961 | 0.566 | 0.947 | 0.770 | 0.110 | 0.956 |
| layer2.8 | 0.940 | 0.803 | 0.103 | 0.960 | 0.566 | 0.932 | 0.749 | 0.110 | 0.954 |
| layer3.0 | 0.944 | 0.834 | 0.103 | 0.957 | 0.566 | 0.932 | 0.754 | 0.108 | 0.954 |
| layer3.5 | 0.857 | 0.873 | 0.101 | 0.878 | 0.566 | 0.801 | 0.635 | 0.129 | 0.854 |
| layer3.8 | 0.922 | 0.947 | 0.100 | 0.749 | 0.566 | 0.676 | 0.606 | 0.123 | 0.738 |
| penult | 0.922 | 0.947 | 0.100 | 0.749 | 0.566 | 0.676 | 0.606 | 0.123 | 0.738 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | +0.663 | +0.097 | +1.000 | -0.000 | -285.288 | -0.018 | +0.068 | +0.015 | · | · | · | · | · |
| layer1.0 | +0.734 | +0.954 | +1.000 | +0.011 | -131.407 | -0.058 | +0.118 | +0.010 | · | · | · | · | · |
| layer1.5 | +0.443 | +0.917 | +2.000 | +0.062 | +18.182 | -0.037 | +0.078 | +0.038 | · | · | · | · | · |
| layer1.8 | +0.690 | +0.214 | +2.000 | +0.026 | +13.119 | -0.036 | +0.102 | +0.018 | · | · | · | · | · |
| layer2.0 | +0.612 | +0.130 | +1.000 | +0.038 | -14.166 | -0.041 | +0.157 | +0.021 | · | · | · | · | · |
| layer2.5 | +0.275 | +0.004 | +0.000 | +0.018 | -3.916 | +0.009 | +0.102 | +0.039 | · | · | · | · | · |
| layer2.8 | +0.057 | +0.034 | -1.000 | +0.026 | -1.859 | -0.024 | -0.224 | +0.023 | · | · | · | · | · |
| layer3.0 | +0.054 | -0.164 | -1.000 | +0.004 | -0.154 | +0.008 | +0.149 | +0.011 | · | · | · | · | · |
| layer3.5 | -0.161 | -0.931 | -2.000 | +0.073 | -0.066 | -0.029 | -0.297 | +0.013 | · | · | · | · | · |
| layer3.8 | +0.077 | +0.027 | +0.000 | +0.098 | -0.003 | +0.004 | -0.036 | -0.005 | · | · | · | · | · |
| penult | +0.077 | +0.027 | +0.000 | +0.098 | -0.003 | +0.004 | -0.036 | -0.005 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | layer1.5 | stem |
| highfreq_ratio | layer2.0 | layer2.0 |
| spectral_slope | layer2.0 | layer2.0 |
| spectral_anisotropy | layer2.0 | layer2.0 |
| noise_sigma | layer1.5 | layer1.0 |
| saturation_mean | layer1.0 | stem |
| hue_cos | stem | layer1.0 |
| hue_sin | stem | stem |
| colorfulness | layer1.0 | layer1.0 |
| edge_density | layer1.5 | stem |
| orientation_entropy | layer2.5 | layer2.0 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.5 | layer3.5 |
| coarse_animal_vehicle | layer3.5 | layer3.0 |
| corruption_family | layer2.0 | layer2.0 |
| corruption_type | layer2.0 | layer1.8 |
| severity | layer2.0 | layer2.0 |