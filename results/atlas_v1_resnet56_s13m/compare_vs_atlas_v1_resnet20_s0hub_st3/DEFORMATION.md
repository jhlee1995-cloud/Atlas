# DEFORMATION  A=`results/atlas_v1_resnet20_s0hub_st3`  B=`results/atlas_v1_resnet56_s13m`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.007 | 0.994 | 0.997 | True | 0.930 | 0.042 | 0.918 | 0.969 | 0.828 |
| layer1.0 | layer1.0 | 0.039 | 0.973 | 0.997 | True | 0.895 | 0.093 | 0.884 | 0.947 | 0.562 |
| layer1.1 | layer1.5 | 0.017 | 0.983 | 0.995 | True | 0.981 | 0.044 | 0.840 | 0.879 | 0.578 |
| layer1.2 | layer1.8 | 0.008 | 0.984 | 0.997 | True | 0.983 | 0.027 | 0.937 | 0.920 | 0.688 |
| layer2.0 | layer2.0 | 0.013 | 0.979 | 0.929 | True | 0.986 | 0.023 | 0.945 | 0.942 | 0.719 |
| layer2.1 | layer2.5 | 0.020 | 0.970 | 0.495 | True | 0.990 | 0.029 | 0.921 | 0.908 | 0.750 |
| layer2.2 | layer2.8 | 0.022 | 0.981 | 0.536 | True | 0.990 | 0.027 | 0.907 | 0.907 | 0.734 |
| layer3.0 | layer3.0 | 0.018 | 0.968 | 0.918 | True | 0.965 | 0.029 | 0.882 | 0.941 | 0.844 |
| layer3.1 | layer3.5 | 0.073 | 0.823 | 0.740 | True | 0.983 | 0.033 | 0.763 | 0.826 | 0.797 |
| layer3.2 | layer3.8 | 0.005 | 0.917 | 0.525 | True | 0.950 | 0.026 | 0.909 | 0.956 | 1.000 |
| penult | penult | 0.005 | 0.917 | 0.525 | True | 0.950 | 0.026 | 0.909 | 0.956 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.933 | 0.761 | 0.115 | 0.955 | 0.540 | 0.891 | 0.755 | 0.117 | 0.946 |
| layer1.0 | 0.923 | 0.677 | 0.118 | 0.945 | 0.540 | 0.902 | 0.671 | 0.122 | 0.934 |
| layer1.1 | 0.854 | 0.641 | 0.115 | 0.885 | 0.540 | 0.773 | 0.622 | 0.117 | 0.855 |
| layer1.2 | 0.943 | 0.732 | 0.113 | 0.931 | 0.540 | 0.917 | 0.706 | 0.124 | 0.921 |
| layer2.0 | 0.951 | 0.769 | 0.106 | 0.937 | 0.540 | 0.931 | 0.732 | 0.117 | 0.927 |
| layer2.1 | 0.917 | 0.725 | 0.104 | 0.913 | 0.540 | 0.899 | 0.657 | 0.110 | 0.896 |
| layer2.2 | 0.896 | 0.734 | 0.103 | 0.911 | 0.540 | 0.876 | 0.672 | 0.110 | 0.894 |
| layer3.0 | 0.893 | 0.796 | 0.102 | 0.925 | 0.540 | 0.875 | 0.701 | 0.111 | 0.908 |
| layer3.1 | 0.699 | 0.810 | 0.101 | 0.705 | 0.540 | 0.713 | 0.577 | 0.131 | 0.677 |
| layer3.2 | 0.877 | 0.923 | 0.100 | 0.781 | 0.540 | 0.662 | 0.567 | 0.125 | 0.738 |
| penult | 0.877 | 0.923 | 0.100 | 0.781 | 0.540 | 0.662 | 0.567 | 0.125 | 0.738 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | -1.107 | -0.026 | -1.000 | +0.012 | +16.902 | +0.011 | -0.207 | -0.007 | · | · | · | · | · |
| layer1.0 | -2.326 | -0.899 | -2.000 | -0.011 | +71.228 | +0.044 | -0.512 | -0.043 | · | · | · | · | · |
| layer1.1 | -1.084 | -1.046 | -1.000 | +0.025 | +73.868 | +0.078 | -0.194 | -0.016 | · | · | · | · | · |
| layer1.2 | -0.320 | -0.358 | +0.000 | +0.019 | +6.589 | +0.005 | -0.152 | +0.013 | · | · | · | · | · |
| layer2.0 | -1.425 | -1.065 | -3.000 | +0.009 | +7.159 | -0.006 | -0.198 | -0.010 | · | · | · | · | · |
| layer2.1 | -0.569 | -0.736 | -1.000 | +0.041 | -2.557 | -0.056 | +0.021 | +0.017 | · | · | · | · | · |
| layer2.2 | -0.978 | -1.872 | -3.000 | +0.008 | +0.922 | -0.041 | +0.064 | -0.016 | · | · | · | · | · |
| layer3.0 | -3.419 | -3.284 | -17.000 | -0.044 | +1.506 | +0.043 | -0.567 | -0.050 | · | · | · | · | · |
| layer3.1 | +0.274 | +4.284 | +2.000 | -0.142 | +0.252 | -0.032 | +0.121 | -0.040 | · | · | · | · | · |
| layer3.2 | +0.812 | +0.553 | +0.000 | +0.318 | -0.040 | -0.027 | +0.145 | +0.001 | · | · | · | · | · |
| penult | +0.812 | +0.553 | +0.000 | +0.318 | -0.040 | -0.027 | +0.145 | +0.001 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | layer1.0 |
| highfreq_ratio | layer1.0 | layer2.0 |
| spectral_slope | layer2.0 | layer2.0 |
| spectral_anisotropy | layer2.0 | layer3.0 |
| noise_sigma | layer1.0 | layer1.5 |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | layer1.0 | layer1.0 |
| edge_density | layer1.0 | layer1.0 |
| orientation_entropy | layer2.0 | layer3.0 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.5 |
| coarse_animal_vehicle | layer3.0 | layer3.5 |
| corruption_family | layer2.0 | layer1.8 |
| corruption_type | layer1.2 | layer1.8 |
| severity | layer3.0 | layer3.0 |