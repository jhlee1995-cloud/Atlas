# DEFORMATION  A=`results/atlas_v1_resnet20_s3_st3`  B=`results/atlas_v1_resnet56_s1`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.016 | 0.982 | 0.984 | True | 0.981 | 0.025 | 0.892 | 0.931 | 0.781 |
| layer1.0 | layer1.0 | 0.048 | 0.973 | 0.990 | True | 0.915 | 0.070 | 0.802 | 0.904 | 0.625 |
| layer1.1 | layer1.5 | 0.012 | 0.984 | 0.942 | True | 0.983 | 0.025 | 0.915 | 0.933 | 0.594 |
| layer1.2 | layer1.8 | 0.006 | 0.988 | 0.997 | True | 0.992 | 0.022 | 0.958 | 0.927 | 0.750 |
| layer2.0 | layer2.0 | 0.005 | 0.991 | 0.992 | True | 0.998 | 0.018 | 0.948 | 0.962 | 0.812 |
| layer2.1 | layer2.5 | 0.009 | 0.984 | 0.939 | True | 0.996 | 0.020 | 0.946 | 0.960 | 0.797 |
| layer2.2 | layer2.8 | 0.014 | 0.969 | 0.501 | True | 0.996 | 0.025 | 0.937 | 0.965 | 0.766 |
| layer3.0 | layer3.0 | 0.014 | 0.979 | 0.987 | True | 0.953 | 0.046 | 0.889 | 0.948 | 0.719 |
| layer3.1 | layer3.5 | 0.034 | 0.914 | 0.733 | True | 0.975 | 0.045 | 0.803 | 0.897 | 0.859 |
| layer3.2 | layer3.8 | 0.010 | 0.878 | 0.721 | True | 0.973 | 0.027 | 0.888 | 0.954 | 1.000 |
| penult | penult | 0.010 | 0.878 | 0.721 | True | 0.973 | 0.027 | 0.888 | 0.954 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.893 | 0.688 | 0.118 | 0.897 | 0.530 | 0.868 | 0.696 | 0.122 | 0.870 |
| layer1.0 | 0.833 | 0.672 | 0.121 | 0.905 | 0.530 | 0.716 | 0.632 | 0.121 | 0.875 |
| layer1.1 | 0.930 | 0.709 | 0.119 | 0.938 | 0.530 | 0.884 | 0.656 | 0.120 | 0.918 |
| layer1.2 | 0.964 | 0.759 | 0.111 | 0.918 | 0.530 | 0.948 | 0.731 | 0.120 | 0.904 |
| layer2.0 | 0.949 | 0.812 | 0.106 | 0.962 | 0.530 | 0.927 | 0.772 | 0.113 | 0.953 |
| layer2.1 | 0.949 | 0.811 | 0.104 | 0.961 | 0.530 | 0.939 | 0.774 | 0.111 | 0.952 |
| layer2.2 | 0.942 | 0.796 | 0.103 | 0.956 | 0.530 | 0.932 | 0.747 | 0.108 | 0.946 |
| layer3.0 | 0.901 | 0.794 | 0.103 | 0.940 | 0.530 | 0.867 | 0.702 | 0.110 | 0.922 |
| layer3.1 | 0.799 | 0.856 | 0.101 | 0.819 | 0.530 | 0.742 | 0.609 | 0.127 | 0.764 |
| layer3.2 | 0.868 | 0.937 | 0.100 | 0.781 | 0.530 | 0.659 | 0.584 | 0.125 | 0.740 |
| penult | 0.868 | 0.937 | 0.100 | 0.781 | 0.530 | 0.659 | 0.584 | 0.125 | 0.740 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | -0.562 | -0.002 | -1.000 | +0.004 | +180.452 | +0.015 | -0.127 | -0.011 | · | · | · | · | · |
| layer1.0 | -1.407 | -1.680 | -3.000 | -0.030 | +141.445 | +0.097 | -0.372 | -0.034 | · | · | · | · | · |
| layer1.1 | -0.039 | -0.191 | -1.000 | +0.008 | -1.268 | -0.022 | -0.116 | -0.001 | · | · | · | · | · |
| layer1.2 | -0.019 | -0.019 | -1.000 | +0.017 | +4.002 | -0.006 | +0.033 | +0.016 | · | · | · | · | · |
| layer2.0 | -0.787 | +0.002 | -2.000 | -0.017 | +6.408 | +0.027 | -0.140 | -0.008 | · | · | · | · | · |
| layer2.1 | -0.014 | +0.613 | +0.000 | +0.009 | -3.029 | -0.027 | -0.053 | +0.013 | · | · | · | · | · |
| layer2.2 | -0.102 | +0.204 | +1.000 | +0.007 | -1.555 | +0.008 | +0.222 | +0.021 | · | · | · | · | · |
| layer3.0 | -2.500 | -2.041 | -12.000 | -0.025 | +0.748 | +0.002 | -0.543 | -0.038 | · | · | · | · | · |
| layer3.1 | -0.158 | +3.051 | +2.000 | +0.041 | -0.247 | -0.061 | +0.288 | +0.028 | · | · | · | · | · |
| layer3.2 | +0.546 | +0.539 | +0.000 | +2.144 | -0.119 | -0.052 | +0.119 | +0.021 | · | · | · | · | · |
| penult | +0.546 | +0.539 | +0.000 | +2.144 | -0.119 | -0.052 | +0.119 | +0.021 | · | · | · | · | · |

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
| orientation_entropy | layer3.0 | layer2.5 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.5 |
| coarse_animal_vehicle | layer3.0 | layer3.5 |
| corruption_family | layer2.0 | layer2.0 |
| corruption_type | layer2.0 | layer2.0 |
| severity | layer3.0 | layer2.0 |