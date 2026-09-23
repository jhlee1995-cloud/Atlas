# DEFORMATION  A=`results/atlas_v1_resnet20_s0hub_st3`  B=`results/atlas_v1_resnet56_e40_st3`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.020 | 0.988 | 0.995 | True | 0.953 | 0.038 | 0.950 | 0.921 | 0.688 |
| layer1.0 | layer1.0 | 0.023 | 0.982 | 0.997 | True | 0.961 | 0.048 | 0.900 | 0.925 | 0.625 |
| layer1.1 | layer1.5 | 0.039 | 0.961 | 0.948 | True | 0.981 | 0.031 | 0.787 | 0.884 | 0.594 |
| layer1.2 | layer1.8 | 0.035 | 0.927 | 0.929 | True | 0.983 | 0.030 | 0.888 | 0.884 | 0.688 |
| layer2.0 | layer2.0 | 0.015 | 0.974 | 0.951 | True | 0.992 | 0.025 | 0.920 | 0.940 | 0.859 |
| layer2.1 | layer2.5 | 0.015 | 0.985 | 0.937 | True | 0.983 | 0.030 | 0.910 | 0.914 | 0.797 |
| layer2.2 | layer2.8 | 0.013 | 0.981 | 0.544 | True | 0.986 | 0.030 | 0.913 | 0.923 | 0.719 |
| layer3.0 | layer3.0 | 0.023 | 0.960 | 0.909 | True | 0.959 | 0.022 | 0.880 | 0.941 | 0.859 |
| layer3.1 | layer3.5 | 0.051 | 0.868 | 0.950 | True | 0.979 | 0.023 | 0.802 | 0.848 | 0.781 |
| layer3.2 | layer3.8 | 0.004 | 0.919 | 0.939 | True | 0.967 | 0.023 | 0.908 | 0.960 | 1.000 |
| penult | penult | 0.004 | 0.919 | 0.939 | True | 0.967 | 0.023 | 0.908 | 0.960 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.923 | 0.721 | 0.114 | 0.918 | 0.502 | 0.914 | 0.726 | 0.119 | 0.910 |
| layer1.0 | 0.915 | 0.701 | 0.120 | 0.932 | 0.502 | 0.879 | 0.706 | 0.125 | 0.917 |
| layer1.1 | 0.815 | 0.610 | 0.113 | 0.865 | 0.502 | 0.729 | 0.579 | 0.116 | 0.830 |
| layer1.2 | 0.898 | 0.733 | 0.111 | 0.894 | 0.502 | 0.881 | 0.712 | 0.119 | 0.892 |
| layer2.0 | 0.934 | 0.742 | 0.107 | 0.927 | 0.502 | 0.908 | 0.702 | 0.116 | 0.906 |
| layer2.1 | 0.907 | 0.728 | 0.105 | 0.907 | 0.502 | 0.885 | 0.657 | 0.111 | 0.880 |
| layer2.2 | 0.906 | 0.753 | 0.103 | 0.915 | 0.502 | 0.883 | 0.681 | 0.110 | 0.895 |
| layer3.0 | 0.874 | 0.774 | 0.102 | 0.916 | 0.502 | 0.865 | 0.681 | 0.112 | 0.903 |
| layer3.1 | 0.752 | 0.848 | 0.101 | 0.723 | 0.502 | 0.702 | 0.585 | 0.135 | 0.694 |
| layer3.2 | 0.881 | 0.914 | 0.100 | 0.804 | 0.502 | 0.667 | 0.555 | 0.125 | 0.752 |
| penult | 0.881 | 0.914 | 0.100 | 0.804 | 0.502 | 0.667 | 0.555 | 0.125 | 0.752 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | +0.335 | -0.025 | +1.000 | +0.010 | -111.976 | +0.025 | +0.052 | +0.009 | · | · | · | · | · |
| layer1.0 | -0.424 | -0.395 | +0.000 | +0.009 | -38.516 | +0.023 | -0.177 | +0.001 | · | · | · | · | · |
| layer1.1 | -0.456 | -0.916 | +0.000 | +0.074 | +45.318 | +0.030 | -0.121 | +0.014 | · | · | · | · | · |
| layer1.2 | -0.450 | -0.474 | -1.000 | +0.023 | +19.626 | +0.021 | -0.138 | +0.025 | · | · | · | · | · |
| layer2.0 | -1.559 | -1.498 | -3.000 | +0.030 | +7.224 | +0.008 | -0.134 | -0.004 | · | · | · | · | · |
| layer2.1 | -1.068 | -1.700 | -2.000 | +0.048 | -0.649 | -0.012 | -0.144 | +0.006 | · | · | · | · | · |
| layer2.2 | -1.488 | -1.078 | -3.000 | +0.037 | +1.623 | -0.045 | +0.000 | +0.000 | · | · | · | · | · |
| layer3.0 | -3.313 | -3.717 | -15.000 | -0.038 | +1.471 | +0.017 | -0.431 | -0.058 | · | · | · | · | · |
| layer3.1 | +0.156 | +3.284 | +0.000 | -0.010 | -0.009 | -0.055 | -0.094 | +0.019 | · | · | · | · | · |
| layer3.2 | +0.970 | +0.203 | +0.000 | -0.143 | +0.007 | -0.000 | +0.195 | -0.007 | · | · | · | · | · |
| penult | +0.970 | +0.203 | +0.000 | -0.143 | +0.007 | -0.000 | +0.195 | -0.007 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer1.0 | layer1.5 |
| spectral_slope | layer2.0 | layer1.5 |
| spectral_anisotropy | layer2.0 | layer2.0 |
| noise_sigma | layer1.0 | layer1.0 |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | layer1.0 | layer1.0 |
| edge_density | layer1.0 | stem |
| orientation_entropy | layer2.0 | layer2.0 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.5 |
| coarse_animal_vehicle | layer3.0 | layer3.0 |
| corruption_family | layer2.0 | layer1.5 |
| corruption_type | layer1.2 | layer1.8 |
| severity | layer3.0 | layer3.0 |