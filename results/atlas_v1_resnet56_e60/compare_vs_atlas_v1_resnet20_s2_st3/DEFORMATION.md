# DEFORMATION  A=`results/atlas_v1_resnet20_s2_st3`  B=`results/atlas_v1_resnet56_e60`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.010 | 0.987 | 0.995 | True | 0.994 | 0.021 | 0.915 | 0.886 | 0.656 |
| layer1.0 | layer1.0 | 0.025 | 0.974 | 0.997 | True | 0.934 | 0.042 | 0.858 | 0.877 | 0.531 |
| layer1.1 | layer1.5 | 0.023 | 0.962 | 0.684 | True | 0.979 | 0.036 | 0.846 | 0.888 | 0.719 |
| layer1.2 | layer1.8 | 0.020 | 0.962 | 0.991 | True | 0.992 | 0.041 | 0.925 | 0.954 | 0.734 |
| layer2.0 | layer2.0 | 0.008 | 0.977 | 0.925 | True | 0.996 | 0.017 | 0.939 | 0.967 | 0.781 |
| layer2.1 | layer2.5 | 0.009 | 0.982 | 0.994 | True | 0.992 | 0.013 | 0.947 | 0.973 | 0.828 |
| layer2.2 | layer2.8 | 0.014 | 0.986 | 0.508 | True | 0.994 | 0.014 | 0.938 | 0.962 | 0.766 |
| layer3.0 | layer3.0 | 0.027 | 0.957 | 0.964 | True | 0.973 | 0.021 | 0.876 | 0.932 | 0.812 |
| layer3.1 | layer3.5 | 0.078 | 0.883 | 0.875 | True | 0.967 | 0.021 | 0.759 | 0.854 | 0.922 |
| layer3.2 | layer3.8 | 0.005 | 0.931 | 0.723 | True | 0.983 | 0.020 | 0.923 | 0.962 | 1.000 |
| penult | penult | 0.005 | 0.931 | 0.723 | True | 0.983 | 0.020 | 0.923 | 0.962 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.918 | 0.675 | 0.115 | 0.870 | 0.566 | 0.921 | 0.631 | 0.122 | 0.854 |
| layer1.0 | 0.875 | 0.642 | 0.121 | 0.883 | 0.566 | 0.832 | 0.634 | 0.124 | 0.872 |
| layer1.1 | 0.864 | 0.655 | 0.111 | 0.879 | 0.566 | 0.831 | 0.625 | 0.114 | 0.840 |
| layer1.2 | 0.929 | 0.744 | 0.111 | 0.929 | 0.566 | 0.919 | 0.696 | 0.123 | 0.917 |
| layer2.0 | 0.951 | 0.797 | 0.106 | 0.955 | 0.566 | 0.935 | 0.743 | 0.115 | 0.939 |
| layer2.1 | 0.950 | 0.777 | 0.104 | 0.959 | 0.566 | 0.941 | 0.732 | 0.111 | 0.949 |
| layer2.2 | 0.938 | 0.781 | 0.103 | 0.950 | 0.566 | 0.926 | 0.716 | 0.109 | 0.942 |
| layer3.0 | 0.879 | 0.778 | 0.102 | 0.911 | 0.566 | 0.866 | 0.669 | 0.110 | 0.904 |
| layer3.1 | 0.733 | 0.825 | 0.102 | 0.743 | 0.566 | 0.708 | 0.586 | 0.128 | 0.687 |
| layer3.2 | 0.879 | 0.928 | 0.100 | 0.797 | 0.566 | 0.672 | 0.576 | 0.125 | 0.758 |
| penult | 0.879 | 0.928 | 0.100 | 0.797 | 0.566 | 0.672 | 0.576 | 0.125 | 0.758 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | +0.009 | -0.003 | +0.000 | -0.000 | -7.623 | -0.038 | -0.000 | -0.001 | · | · | · | · | · |
| layer1.0 | -1.049 | -0.781 | -2.000 | -0.011 | +6.487 | +0.018 | -0.303 | -0.024 | · | · | · | · | · |
| layer1.1 | -0.107 | -0.248 | +0.000 | +0.039 | +9.107 | -0.090 | -0.046 | +0.015 | · | · | · | · | · |
| layer1.2 | -0.288 | +0.801 | +0.000 | +0.029 | -2.849 | -0.003 | -0.007 | +0.035 | · | · | · | · | · |
| layer2.0 | -0.660 | -0.066 | -1.000 | +0.027 | +6.400 | +0.000 | -0.111 | +0.011 | · | · | · | · | · |
| layer2.1 | -0.303 | -0.237 | -1.000 | +0.044 | +1.162 | +0.012 | -0.098 | +0.026 | · | · | · | · | · |
| layer2.2 | -0.511 | -0.873 | -2.000 | +0.033 | -0.170 | +0.014 | -0.194 | +0.009 | · | · | · | · | · |
| layer3.0 | -3.194 | -4.081 | -17.000 | -0.042 | +1.340 | +0.023 | -0.473 | -0.080 | · | · | · | · | · |
| layer3.1 | -0.154 | +4.214 | +3.000 | -0.026 | -0.022 | -0.044 | +0.190 | -0.020 | · | · | · | · | · |
| layer3.2 | +0.731 | +0.439 | +0.000 | +0.347 | -0.041 | -0.025 | +0.108 | +0.001 | · | · | · | · | · |
| penult | +0.731 | +0.439 | +0.000 | +0.347 | -0.041 | -0.025 | +0.108 | +0.001 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer2.0 | layer1.5 |
| spectral_slope | layer2.0 | layer1.5 |
| spectral_anisotropy | layer2.0 | layer2.5 |
| noise_sigma | layer1.0 | layer1.5 |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | layer1.0 | layer1.0 |
| edge_density | layer1.0 | layer1.0 |
| orientation_entropy | layer2.0 | layer2.5 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.5 |
| coarse_animal_vehicle | layer3.0 | layer3.5 |
| corruption_family | layer1.2 | layer2.0 |
| corruption_type | layer2.0 | layer2.0 |
| severity | layer2.2 | layer3.0 |