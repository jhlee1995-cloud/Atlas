# DEFORMATION  A=`results/atlas_v1_resnet56_s0hub`  B=`results/atlas_v1_resnet56_e10`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.033 | 0.966 | 0.979 | True | 0.965 | 0.021 | 0.891 | 0.913 | 0.625 |
| layer1.0 | layer1.0 | 0.010 | 0.990 | 0.992 | True | 0.858 | 0.056 | 0.897 | 0.941 | 0.625 |
| layer1.5 | layer1.5 | 0.024 | 0.973 | 1.000 | True | 0.971 | 0.043 | 0.865 | 0.942 | 0.641 |
| layer1.8 | layer1.8 | 0.021 | 0.972 | 0.962 | True | 0.963 | 0.046 | 0.878 | 0.920 | 0.719 |
| layer2.0 | layer2.0 | 0.031 | 0.939 | 0.933 | True | 0.965 | 0.067 | 0.852 | 0.940 | 0.781 |
| layer2.5 | layer2.5 | 0.030 | 0.935 | 0.537 | True | 0.891 | 0.089 | 0.770 | 0.891 | 0.594 |
| layer2.8 | layer2.8 | 0.040 | 0.882 | 0.676 | True | 0.899 | 0.088 | 0.756 | 0.887 | 0.641 |
| layer3.0 | layer3.0 | 0.041 | 0.911 | 0.753 | True | 0.833 | 0.104 | 0.756 | 0.884 | 0.672 |
| layer3.5 | layer3.5 | 0.053 | 0.893 | 0.873 | True | 0.843 | 0.100 | 0.752 | 0.856 | 0.766 |
| layer3.8 | layer3.8 | 0.144 | 0.818 | 0.738 | True | 0.932 | 0.055 | 0.736 | 0.763 | 0.906 |
| penult | penult | 0.144 | 0.818 | 0.738 | True | 0.932 | 0.055 | 0.736 | 0.763 | 0.906 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.896 | 0.697 | 0.114 | 0.914 | 0.307 | 0.867 | 0.712 | 0.121 | 0.904 |
| layer1.0 | 0.923 | 0.664 | 0.112 | 0.945 | 0.307 | 0.870 | 0.649 | 0.116 | 0.931 |
| layer1.5 | 0.864 | 0.595 | 0.108 | 0.921 | 0.307 | 0.801 | 0.557 | 0.112 | 0.903 |
| layer1.8 | 0.885 | 0.675 | 0.107 | 0.908 | 0.307 | 0.842 | 0.650 | 0.115 | 0.887 |
| layer2.0 | 0.866 | 0.701 | 0.104 | 0.906 | 0.307 | 0.831 | 0.645 | 0.111 | 0.898 |
| layer2.5 | 0.799 | 0.693 | 0.103 | 0.879 | 0.307 | 0.758 | 0.593 | 0.109 | 0.855 |
| layer2.8 | 0.779 | 0.691 | 0.103 | 0.869 | 0.307 | 0.735 | 0.587 | 0.110 | 0.845 |
| layer3.0 | 0.798 | 0.743 | 0.102 | 0.871 | 0.307 | 0.759 | 0.583 | 0.113 | 0.842 |
| layer3.5 | 0.763 | 0.815 | 0.101 | 0.785 | 0.307 | 0.650 | 0.532 | 0.123 | 0.755 |
| layer3.8 | 0.698 | 0.853 | 0.100 | 0.655 | 0.307 | 0.532 | 0.480 | 0.120 | 0.646 |
| penult | 0.698 | 0.853 | 0.100 | 0.655 | 0.307 | 0.532 | 0.480 | 0.120 | 0.646 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | -0.037 | -0.459 | +0.000 | -0.000 | -37.654 | +0.051 | +0.104 | +0.007 | · | · | · | · | · |
| layer1.0 | -0.416 | -0.166 | +0.000 | +0.013 | +290.895 | -0.003 | +0.029 | +0.019 | · | · | · | · | · |
| layer1.5 | -0.045 | -1.051 | +0.000 | +0.023 | +52.801 | -0.039 | -0.054 | +0.029 | · | · | · | · | · |
| layer1.8 | -0.901 | -0.548 | -1.000 | +0.059 | +14.420 | -0.004 | -0.157 | +0.034 | · | · | · | · | · |
| layer2.0 | -0.332 | -1.550 | -2.000 | +0.075 | -0.787 | -0.056 | -0.075 | +0.038 | · | · | · | · | · |
| layer2.5 | -0.929 | -2.772 | -4.000 | +0.140 | -0.701 | -0.005 | -0.197 | +0.041 | · | · | · | · | · |
| layer2.8 | -1.796 | -3.785 | -6.000 | +0.159 | +0.646 | -0.014 | -0.424 | +0.026 | · | · | · | · | · |
| layer3.0 | -1.436 | -3.977 | -10.000 | +0.151 | -0.015 | +0.059 | -0.260 | +0.025 | · | · | · | · | · |
| layer3.5 | -3.913 | -7.890 | -23.000 | +0.086 | +0.580 | +0.105 | -0.401 | -0.053 | · | · | · | · | · |
| layer3.8 | +1.314 | -2.594 | +1.000 | -3.879 | +0.637 | +0.225 | -0.063 | -0.112 | · | · | · | · | · |
| penult | +1.314 | -2.594 | +1.000 | -3.879 | +0.637 | +0.225 | -0.063 | -0.112 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer2.0 | layer1.5 |
| spectral_slope | layer2.0 | layer1.5 |
| spectral_anisotropy | layer3.0 | layer1.5 |
| noise_sigma | layer1.0 | layer1.5 |
| saturation_mean | stem | layer1.0 |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | layer1.0 | stem |
| edge_density | layer1.0 | layer1.0 |
| orientation_entropy | layer2.0 | layer2.0 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.5 | layer3.0 |
| coarse_animal_vehicle | layer3.5 | layer2.5 |
| corruption_family | layer2.0 | layer2.0 |
| corruption_type | layer2.0 | layer1.8 |
| severity | layer3.0 | layer2.0 |