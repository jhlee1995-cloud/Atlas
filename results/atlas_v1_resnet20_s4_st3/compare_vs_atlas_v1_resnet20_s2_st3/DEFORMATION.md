# DEFORMATION  A=`results/atlas_v1_resnet20_s2_st3`  B=`results/atlas_v1_resnet20_s4_st3`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.007 | 0.997 | 0.997 | True | 0.973 | 0.016 | 0.971 | 0.945 | 0.766 |
| layer1.0 | layer1.0 | 0.017 | 0.974 | 0.995 | True | 0.983 | 0.036 | 0.918 | 0.902 | 0.656 |
| layer1.1 | layer1.1 | 0.037 | 0.920 | 0.695 | True | 0.990 | 0.030 | 0.875 | 0.896 | 0.703 |
| layer1.2 | layer1.2 | 0.005 | 0.989 | 1.000 | True | 0.992 | 0.021 | 0.950 | 0.967 | 0.766 |
| layer2.0 | layer2.0 | 0.007 | 0.981 | 0.925 | True | 0.996 | 0.018 | 0.967 | 0.981 | 0.828 |
| layer2.1 | layer2.1 | 0.007 | 0.990 | 0.947 | True | 0.994 | 0.017 | 0.950 | 0.975 | 0.703 |
| layer2.2 | layer2.2 | 0.009 | 0.979 | 0.994 | True | 0.998 | 0.016 | 0.933 | 0.968 | 0.828 |
| layer3.0 | layer3.0 | 0.011 | 0.986 | 0.941 | True | 0.950 | 0.021 | 0.907 | 0.961 | 0.781 |
| layer3.1 | layer3.1 | 0.016 | 0.972 | 0.713 | True | 0.979 | 0.017 | 0.905 | 0.934 | 0.906 |
| layer3.2 | layer3.2 | 0.002 | 0.967 | 0.934 | True | 0.967 | 0.025 | 0.912 | 0.962 | 1.000 |
| penult | penult | 0.002 | 0.967 | 0.934 | True | 0.967 | 0.025 | 0.912 | 0.962 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.953 | 0.756 | 0.112 | 0.941 | 0.539 | 0.944 | 0.753 | 0.118 | 0.932 |
| layer1.0 | 0.906 | 0.673 | 0.121 | 0.888 | 0.539 | 0.886 | 0.657 | 0.125 | 0.872 |
| layer1.1 | 0.883 | 0.685 | 0.115 | 0.885 | 0.539 | 0.849 | 0.676 | 0.118 | 0.866 |
| layer1.2 | 0.941 | 0.736 | 0.112 | 0.941 | 0.539 | 0.922 | 0.717 | 0.125 | 0.928 |
| layer2.0 | 0.960 | 0.810 | 0.106 | 0.977 | 0.539 | 0.949 | 0.771 | 0.116 | 0.970 |
| layer2.1 | 0.949 | 0.789 | 0.105 | 0.968 | 0.539 | 0.936 | 0.737 | 0.113 | 0.959 |
| layer2.2 | 0.936 | 0.796 | 0.103 | 0.957 | 0.539 | 0.919 | 0.736 | 0.111 | 0.950 |
| layer3.0 | 0.883 | 0.812 | 0.102 | 0.928 | 0.539 | 0.862 | 0.671 | 0.113 | 0.914 |
| layer3.1 | 0.878 | 0.849 | 0.101 | 0.881 | 0.539 | 0.773 | 0.611 | 0.138 | 0.857 |
| layer3.2 | 0.892 | 0.933 | 0.100 | 0.833 | 0.539 | 0.699 | 0.572 | 0.124 | 0.773 |
| penult | 0.892 | 0.933 | 0.100 | 0.833 | 0.539 | 0.699 | 0.572 | 0.124 | 0.773 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | -0.137 | +0.115 | +0.000 | +0.002 | +6.257 | -0.009 | -0.007 | +0.004 | · | · | · | · | · |
| layer1.0 | +0.050 | +0.075 | +0.000 | +0.011 | -42.288 | -0.094 | -0.001 | +0.007 | · | · | · | · | · |
| layer1.1 | +0.082 | -0.174 | +0.000 | +0.006 | +18.384 | -0.116 | +0.030 | -0.001 | · | · | · | · | · |
| layer1.2 | +0.207 | +0.603 | +0.000 | -0.006 | +0.549 | -0.020 | +0.100 | +0.006 | · | · | · | · | · |
| layer2.0 | +0.627 | +0.547 | +1.000 | -0.001 | +1.464 | +0.008 | +0.095 | +0.005 | · | · | · | · | · |
| layer2.1 | +0.485 | +0.380 | +1.000 | -0.007 | +3.531 | +0.022 | +0.150 | +0.003 | · | · | · | · | · |
| layer2.2 | +0.273 | +0.647 | +1.000 | -0.002 | +1.468 | -0.007 | -0.021 | +0.008 | · | · | · | · | · |
| layer3.0 | +0.024 | -1.657 | -1.000 | -0.003 | -0.137 | +0.004 | +0.173 | -0.021 | · | · | · | · | · |
| layer3.1 | -0.081 | -0.189 | +1.000 | +0.023 | -0.067 | +0.006 | +0.054 | +0.012 | · | · | · | · | · |
| layer3.2 | -0.134 | +0.071 | +0.000 | +0.003 | -0.002 | -0.009 | +0.037 | +0.003 | · | · | · | · | · |
| penult | -0.134 | +0.071 | +0.000 | +0.003 | -0.002 | -0.009 | +0.037 | +0.003 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer2.0 | layer2.0 |
| spectral_slope | layer2.0 | layer2.0 |
| spectral_anisotropy | layer2.0 | layer2.0 |
| noise_sigma | layer1.0 | layer1.0 |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | layer1.0 | stem |
| edge_density | layer1.0 | layer1.0 |
| orientation_entropy | layer2.0 | layer2.0 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.1 |
| coarse_animal_vehicle | layer3.0 | layer3.0 |
| corruption_family | layer1.2 | layer2.0 |
| corruption_type | layer2.0 | layer2.0 |
| severity | layer2.2 | layer2.0 |