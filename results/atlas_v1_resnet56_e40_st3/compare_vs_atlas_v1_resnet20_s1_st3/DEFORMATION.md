# DEFORMATION  A=`results/atlas_v1_resnet20_s1_st3`  B=`results/atlas_v1_resnet56_e40_st3`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.018 | 0.986 | 0.984 | True | 0.926 | 0.040 | 0.957 | 0.931 | 0.672 |
| layer1.0 | layer1.0 | 0.012 | 0.992 | 0.997 | True | 0.955 | 0.030 | 0.951 | 0.962 | 0.688 |
| layer1.1 | layer1.5 | 0.044 | 0.943 | 0.953 | True | 0.981 | 0.034 | 0.854 | 0.889 | 0.531 |
| layer1.2 | layer1.8 | 0.034 | 0.920 | 0.925 | True | 0.986 | 0.017 | 0.911 | 0.899 | 0.656 |
| layer2.0 | layer2.0 | 0.017 | 0.973 | 0.964 | True | 0.996 | 0.030 | 0.928 | 0.960 | 0.734 |
| layer2.1 | layer2.5 | 0.012 | 0.987 | 0.980 | True | 0.990 | 0.034 | 0.926 | 0.938 | 0.719 |
| layer2.2 | layer2.8 | 0.010 | 0.982 | 0.541 | True | 0.990 | 0.031 | 0.924 | 0.945 | 0.750 |
| layer3.0 | layer3.0 | 0.018 | 0.963 | 0.931 | True | 0.975 | 0.025 | 0.886 | 0.937 | 0.812 |
| layer3.1 | layer3.5 | 0.046 | 0.874 | 0.941 | True | 0.967 | 0.030 | 0.835 | 0.892 | 0.844 |
| layer3.2 | layer3.8 | 0.004 | 0.960 | 0.838 | True | 0.936 | 0.038 | 0.919 | 0.962 | 1.000 |
| penult | penult | 0.004 | 0.960 | 0.838 | True | 0.936 | 0.038 | 0.919 | 0.962 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.932 | 0.745 | 0.115 | 0.924 | 0.523 | 0.924 | 0.736 | 0.118 | 0.919 |
| layer1.0 | 0.916 | 0.756 | 0.119 | 0.957 | 0.523 | 0.877 | 0.745 | 0.122 | 0.945 |
| layer1.1 | 0.824 | 0.541 | 0.112 | 0.853 | 0.523 | 0.784 | 0.520 | 0.113 | 0.826 |
| layer1.2 | 0.920 | 0.698 | 0.109 | 0.887 | 0.523 | 0.907 | 0.690 | 0.117 | 0.892 |
| layer2.0 | 0.933 | 0.773 | 0.106 | 0.950 | 0.523 | 0.912 | 0.736 | 0.113 | 0.939 |
| layer2.1 | 0.924 | 0.762 | 0.105 | 0.926 | 0.523 | 0.906 | 0.708 | 0.112 | 0.911 |
| layer2.2 | 0.927 | 0.797 | 0.104 | 0.945 | 0.523 | 0.912 | 0.712 | 0.113 | 0.933 |
| layer3.0 | 0.893 | 0.778 | 0.102 | 0.928 | 0.523 | 0.881 | 0.676 | 0.113 | 0.913 |
| layer3.1 | 0.784 | 0.870 | 0.101 | 0.782 | 0.523 | 0.745 | 0.603 | 0.134 | 0.752 |
| layer3.2 | 0.879 | 0.924 | 0.100 | 0.813 | 0.523 | 0.674 | 0.568 | 0.126 | 0.759 |
| penult | 0.879 | 0.924 | 0.100 | 0.813 | 0.523 | 0.674 | 0.568 | 0.126 | 0.759 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | +0.434 | -0.039 | +1.000 | +0.011 | -113.118 | +0.019 | +0.085 | +0.008 | · | · | · | · | · |
| layer1.0 | -0.171 | +0.359 | +1.000 | +0.003 | -64.801 | -0.031 | -0.081 | +0.009 | · | · | · | · | · |
| layer1.1 | +0.142 | +0.185 | +1.000 | +0.075 | +28.986 | -0.020 | +0.033 | +0.032 | · | · | · | · | · |
| layer1.2 | +0.193 | +0.245 | +0.000 | +0.007 | +23.554 | +0.057 | +0.020 | +0.032 | · | · | · | · | · |
| layer2.0 | -1.053 | -0.749 | -2.000 | +0.018 | +5.287 | -0.053 | -0.096 | +0.001 | · | · | · | · | · |
| layer2.1 | -0.421 | -1.615 | -1.000 | +0.040 | -5.281 | +0.027 | -0.106 | +0.021 | · | · | · | · | · |
| layer2.2 | -0.973 | -0.651 | -2.000 | +0.046 | -0.489 | -0.000 | -0.185 | +0.015 | · | · | · | · | · |
| layer3.0 | -3.223 | -1.866 | -16.000 | -0.031 | +1.497 | -0.004 | -0.293 | -0.056 | · | · | · | · | · |
| layer3.1 | +0.064 | +2.492 | +0.000 | +0.016 | -0.025 | -0.068 | -0.085 | +0.009 | · | · | · | · | · |
| layer3.2 | +1.128 | +0.068 | +0.000 | -0.214 | +0.014 | +0.009 | +0.125 | -0.006 | · | · | · | · | · |
| penult | +1.128 | +0.068 | +0.000 | -0.214 | +0.014 | +0.009 | +0.125 | -0.006 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer1.1 | layer1.5 |
| spectral_slope | layer2.0 | layer1.5 |
| spectral_anisotropy | layer1.2 | layer2.0 |
| noise_sigma | layer1.0 | layer1.0 |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | stem | layer1.0 |
| edge_density | layer1.0 | stem |
| orientation_entropy | layer2.0 | layer2.0 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.5 |
| coarse_animal_vehicle | layer3.0 | layer3.0 |
| corruption_family | layer2.0 | layer1.5 |
| corruption_type | layer2.0 | layer1.8 |
| severity | layer3.0 | layer3.0 |