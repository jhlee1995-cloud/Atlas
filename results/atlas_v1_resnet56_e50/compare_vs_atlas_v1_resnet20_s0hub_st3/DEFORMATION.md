# DEFORMATION  A=`results/atlas_v1_resnet20_s0hub_st3`  B=`results/atlas_v1_resnet56_e50`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.023 | 0.981 | 0.948 | True | 0.913 | 0.041 | 0.951 | 0.894 | 0.719 |
| layer1.0 | layer1.0 | 0.045 | 0.982 | 0.951 | True | 0.870 | 0.133 | 0.927 | 0.931 | 0.609 |
| layer1.1 | layer1.5 | 0.018 | 0.982 | 0.995 | True | 0.990 | 0.033 | 0.843 | 0.889 | 0.734 |
| layer1.2 | layer1.8 | 0.012 | 0.986 | 0.991 | True | 0.986 | 0.031 | 0.941 | 0.904 | 0.641 |
| layer2.0 | layer2.0 | 0.017 | 0.980 | 0.978 | True | 0.992 | 0.020 | 0.941 | 0.946 | 0.781 |
| layer2.1 | layer2.5 | 0.010 | 0.983 | 0.934 | True | 0.996 | 0.022 | 0.929 | 0.929 | 0.750 |
| layer2.2 | layer2.8 | 0.016 | 0.988 | 0.973 | True | 0.992 | 0.027 | 0.909 | 0.914 | 0.750 |
| layer3.0 | layer3.0 | 0.020 | 0.957 | 0.966 | True | 0.965 | 0.028 | 0.877 | 0.936 | 0.812 |
| layer3.1 | layer3.5 | 0.061 | 0.851 | 0.911 | True | 0.981 | 0.031 | 0.789 | 0.824 | 0.797 |
| layer3.2 | layer3.8 | 0.005 | 0.904 | 0.611 | True | 0.967 | 0.030 | 0.907 | 0.952 | 1.000 |
| penult | penult | 0.005 | 0.904 | 0.611 | True | 0.967 | 0.030 | 0.907 | 0.952 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.890 | 0.662 | 0.111 | 0.889 | 0.490 | 0.882 | 0.667 | 0.115 | 0.881 |
| layer1.0 | 0.899 | 0.656 | 0.119 | 0.923 | 0.490 | 0.875 | 0.643 | 0.126 | 0.909 |
| layer1.1 | 0.871 | 0.660 | 0.114 | 0.897 | 0.490 | 0.799 | 0.627 | 0.116 | 0.876 |
| layer1.2 | 0.931 | 0.712 | 0.113 | 0.913 | 0.490 | 0.910 | 0.689 | 0.122 | 0.904 |
| layer2.0 | 0.945 | 0.731 | 0.107 | 0.926 | 0.490 | 0.929 | 0.691 | 0.117 | 0.913 |
| layer2.1 | 0.927 | 0.719 | 0.105 | 0.924 | 0.490 | 0.903 | 0.661 | 0.111 | 0.899 |
| layer2.2 | 0.903 | 0.722 | 0.103 | 0.907 | 0.490 | 0.879 | 0.637 | 0.110 | 0.879 |
| layer3.0 | 0.886 | 0.789 | 0.102 | 0.925 | 0.490 | 0.861 | 0.700 | 0.111 | 0.911 |
| layer3.1 | 0.741 | 0.833 | 0.101 | 0.719 | 0.490 | 0.729 | 0.571 | 0.133 | 0.684 |
| layer3.2 | 0.877 | 0.911 | 0.100 | 0.765 | 0.490 | 0.667 | 0.582 | 0.124 | 0.741 |
| penult | 0.877 | 0.911 | 0.100 | 0.765 | 0.490 | 0.667 | 0.582 | 0.124 | 0.741 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | -0.221 | -0.068 | +0.000 | +0.004 | +42.397 | +0.031 | +0.036 | +0.007 | · | · | · | · | · |
| layer1.0 | -1.604 | -1.145 | -1.000 | -0.001 | +369.454 | +0.082 | -0.266 | -0.034 | · | · | · | · | · |
| layer1.1 | -1.295 | -1.304 | -1.000 | +0.001 | +56.268 | +0.080 | -0.163 | -0.024 | · | · | · | · | · |
| layer1.2 | -0.615 | -0.556 | -1.000 | -0.000 | +27.844 | +0.020 | -0.207 | -0.013 | · | · | · | · | · |
| layer2.0 | -1.352 | -1.468 | -4.000 | -0.005 | +9.761 | +0.012 | -0.241 | -0.029 | · | · | · | · | · |
| layer2.1 | -0.421 | -1.365 | -2.000 | +0.039 | -0.612 | -0.016 | -0.196 | -0.013 | · | · | · | · | · |
| layer2.2 | -0.603 | -1.803 | -3.000 | +0.033 | +0.784 | +0.003 | -0.044 | -0.019 | · | · | · | · | · |
| layer3.0 | -2.959 | -3.288 | -15.000 | -0.059 | +1.606 | +0.071 | -0.325 | -0.059 | · | · | · | · | · |
| layer3.1 | -0.328 | +3.629 | -1.000 | -0.048 | +0.025 | -0.035 | -0.182 | +0.019 | · | · | · | · | · |
| layer3.2 | +1.114 | +0.454 | +0.000 | +0.143 | -0.025 | -0.015 | +0.262 | -0.002 | · | · | · | · | · |
| penult | +1.114 | +0.454 | +0.000 | +0.143 | -0.025 | -0.015 | +0.262 | -0.002 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | layer2.0 |
| highfreq_ratio | layer1.0 | layer2.0 |
| spectral_slope | layer2.0 | layer2.0 |
| spectral_anisotropy | layer2.0 | layer2.0 |
| noise_sigma | layer1.0 | layer1.5 |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | layer1.0 | layer1.5 |
| edge_density | layer1.0 | layer1.5 |
| orientation_entropy | layer2.0 | layer3.0 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.5 |
| coarse_animal_vehicle | layer3.0 | layer3.5 |
| corruption_family | layer2.0 | layer2.0 |
| corruption_type | layer1.2 | layer2.0 |
| severity | layer3.0 | layer3.0 |