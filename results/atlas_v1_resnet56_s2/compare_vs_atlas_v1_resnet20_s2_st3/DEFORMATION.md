# DEFORMATION  A=`results/atlas_v1_resnet20_s2_st3`  B=`results/atlas_v1_resnet56_s2`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.028 | 0.970 | 0.987 | True | 0.975 | 0.024 | 0.962 | 0.911 | 0.656 |
| layer1.0 | layer1.0 | 0.017 | 0.972 | 0.995 | True | 0.965 | 0.033 | 0.871 | 0.922 | 0.594 |
| layer1.1 | layer1.5 | 0.041 | 0.947 | 0.679 | True | 0.990 | 0.024 | 0.878 | 0.900 | 0.719 |
| layer1.2 | layer1.8 | 0.019 | 0.965 | 0.960 | True | 0.996 | 0.024 | 0.948 | 0.964 | 0.750 |
| layer2.0 | layer2.0 | 0.008 | 0.982 | 0.938 | True | 0.998 | 0.017 | 0.969 | 0.978 | 0.906 |
| layer2.1 | layer2.5 | 0.012 | 0.981 | 0.545 | True | 0.988 | 0.018 | 0.946 | 0.963 | 0.797 |
| layer2.2 | layer2.8 | 0.010 | 0.987 | 0.539 | True | 0.977 | 0.026 | 0.929 | 0.956 | 0.766 |
| layer3.0 | layer3.0 | 0.017 | 0.973 | 0.987 | True | 0.965 | 0.027 | 0.901 | 0.946 | 0.781 |
| layer3.1 | layer3.5 | 0.088 | 0.823 | 0.779 | True | 0.975 | 0.023 | 0.752 | 0.817 | 0.859 |
| layer3.2 | layer3.8 | 0.011 | 0.909 | 0.852 | False | 0.973 | 0.017 | 0.919 | 0.955 | 1.000 |
| penult | penult | 0.011 | 0.909 | 0.852 | False | 0.973 | 0.017 | 0.919 | 0.955 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.909 | 0.713 | 0.111 | 0.891 | 0.552 | 0.906 | 0.727 | 0.115 | 0.878 |
| layer1.0 | 0.865 | 0.679 | 0.118 | 0.916 | 0.552 | 0.831 | 0.660 | 0.123 | 0.895 |
| layer1.1 | 0.864 | 0.642 | 0.112 | 0.886 | 0.552 | 0.841 | 0.612 | 0.113 | 0.842 |
| layer1.2 | 0.943 | 0.761 | 0.111 | 0.944 | 0.552 | 0.935 | 0.727 | 0.125 | 0.937 |
| layer2.0 | 0.967 | 0.806 | 0.106 | 0.972 | 0.552 | 0.957 | 0.771 | 0.114 | 0.965 |
| layer2.1 | 0.939 | 0.776 | 0.104 | 0.946 | 0.552 | 0.932 | 0.732 | 0.110 | 0.940 |
| layer2.2 | 0.926 | 0.789 | 0.103 | 0.947 | 0.552 | 0.913 | 0.724 | 0.110 | 0.935 |
| layer3.0 | 0.892 | 0.795 | 0.102 | 0.924 | 0.552 | 0.881 | 0.678 | 0.109 | 0.909 |
| layer3.1 | 0.732 | 0.845 | 0.101 | 0.695 | 0.552 | 0.710 | 0.570 | 0.125 | 0.657 |
| layer3.2 | 0.870 | 0.932 | 0.100 | 0.774 | 0.552 | 0.646 | 0.597 | 0.124 | 0.731 |
| penult | 0.870 | 0.932 | 0.100 | 0.774 | 0.552 | 0.646 | 0.597 | 0.124 | 0.731 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | -0.103 | -0.235 | +0.000 | -0.007 | +61.033 | -0.014 | -0.044 | +0.004 | · | · | · | · | · |
| layer1.0 | -1.043 | -0.298 | -2.000 | -0.011 | -3.038 | -0.013 | -0.353 | -0.022 | · | · | · | · | · |
| layer1.1 | +0.155 | +0.604 | +1.000 | +0.088 | +18.209 | -0.102 | +0.007 | +0.042 | · | · | · | · | · |
| layer1.2 | +0.572 | +0.982 | +1.000 | +0.038 | +18.264 | -0.030 | +0.082 | +0.035 | · | · | · | · | · |
| layer2.0 | +0.100 | +0.386 | +0.000 | +0.014 | -2.599 | +0.006 | +0.079 | +0.007 | · | · | · | · | · |
| layer2.1 | +0.333 | +0.487 | +1.000 | +0.015 | -2.070 | +0.029 | +0.105 | +0.040 | · | · | · | · | · |
| layer2.2 | +0.177 | +0.848 | +0.000 | +0.026 | -1.951 | -0.019 | -0.025 | +0.038 | · | · | · | · | · |
| layer3.0 | -2.749 | -2.763 | -14.000 | -0.048 | +0.802 | +0.031 | -0.226 | -0.054 | · | · | · | · | · |
| layer3.1 | -0.113 | +5.088 | +2.000 | +0.124 | -0.249 | -0.103 | -0.060 | +0.027 | · | · | · | · | · |
| layer3.2 | +0.424 | +0.511 | +0.000 | +2.191 | -0.119 | -0.045 | +0.088 | +0.015 | · | · | · | · | · |
| penult | +0.424 | +0.511 | +0.000 | +2.191 | -0.119 | -0.045 | +0.088 | +0.015 | · | · | · | · | · |

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
| hue_cos | stem | layer1.0 |
| hue_sin | stem | stem |
| colorfulness | layer1.0 | layer1.0 |
| edge_density | layer1.0 | stem |
| orientation_entropy | layer2.0 | layer2.0 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.5 |
| coarse_animal_vehicle | layer3.0 | layer3.0 |
| corruption_family | layer1.2 | layer2.0 |
| corruption_type | layer2.0 | layer1.8 |
| severity | layer2.2 | layer2.0 |