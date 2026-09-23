# DEFORMATION  A=`results/atlas_v1_resnet20_s2_st3`  B=`results/atlas_v1_resnet56_e40_st3`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.002 | 0.997 | 1.000 | True | 0.942 | 0.038 | 0.968 | 0.969 | 0.812 |
| layer1.0 | layer1.0 | 0.013 | 0.987 | 0.997 | True | 0.977 | 0.024 | 0.899 | 0.929 | 0.641 |
| layer1.1 | layer1.5 | 0.069 | 0.896 | 0.685 | True | 0.986 | 0.021 | 0.807 | 0.847 | 0.594 |
| layer1.2 | layer1.8 | 0.034 | 0.925 | 0.938 | True | 0.990 | 0.025 | 0.880 | 0.920 | 0.766 |
| layer2.0 | layer2.0 | 0.008 | 0.981 | 0.925 | True | 0.996 | 0.018 | 0.947 | 0.970 | 0.844 |
| layer2.1 | layer2.5 | 0.008 | 0.991 | 0.979 | True | 0.992 | 0.022 | 0.942 | 0.967 | 0.812 |
| layer2.2 | layer2.8 | 0.012 | 0.980 | 0.538 | True | 0.992 | 0.026 | 0.937 | 0.961 | 0.812 |
| layer3.0 | layer3.0 | 0.030 | 0.956 | 0.918 | True | 0.938 | 0.025 | 0.897 | 0.932 | 0.812 |
| layer3.1 | layer3.5 | 0.075 | 0.876 | 0.905 | True | 0.986 | 0.023 | 0.780 | 0.828 | 0.875 |
| layer3.2 | layer3.8 | 0.004 | 0.952 | 1.000 | True | 0.957 | 0.023 | 0.923 | 0.962 | 1.000 |
| penult | penult | 0.004 | 0.952 | 1.000 | True | 0.957 | 0.023 | 0.923 | 0.962 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.979 | 0.784 | 0.114 | 0.961 | 0.543 | 0.967 | 0.753 | 0.121 | 0.951 |
| layer1.0 | 0.895 | 0.725 | 0.120 | 0.935 | 0.543 | 0.837 | 0.712 | 0.124 | 0.912 |
| layer1.1 | 0.805 | 0.561 | 0.111 | 0.826 | 0.543 | 0.784 | 0.546 | 0.112 | 0.794 |
| layer1.2 | 0.879 | 0.719 | 0.111 | 0.896 | 0.543 | 0.872 | 0.670 | 0.121 | 0.887 |
| layer2.0 | 0.956 | 0.804 | 0.106 | 0.961 | 0.543 | 0.946 | 0.760 | 0.114 | 0.949 |
| layer2.1 | 0.942 | 0.784 | 0.105 | 0.949 | 0.543 | 0.932 | 0.733 | 0.113 | 0.941 |
| layer2.2 | 0.939 | 0.780 | 0.103 | 0.947 | 0.543 | 0.928 | 0.730 | 0.111 | 0.943 |
| layer3.0 | 0.861 | 0.768 | 0.102 | 0.902 | 0.543 | 0.865 | 0.659 | 0.110 | 0.891 |
| layer3.1 | 0.748 | 0.844 | 0.101 | 0.719 | 0.543 | 0.702 | 0.622 | 0.130 | 0.686 |
| layer3.2 | 0.884 | 0.925 | 0.100 | 0.814 | 0.543 | 0.685 | 0.563 | 0.125 | 0.760 |
| penult | 0.884 | 0.925 | 0.100 | 0.814 | 0.543 | 0.685 | 0.563 | 0.125 | 0.760 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | +0.184 | -0.021 | +1.000 | +0.005 | -4.974 | -0.005 | +0.055 | +0.006 | · | · | · | · | · |
| layer1.0 | -0.794 | -0.228 | -1.000 | +0.001 | -36.342 | -0.009 | -0.288 | +0.001 | · | · | · | · | · |
| layer1.1 | -0.356 | -0.812 | +0.000 | +0.075 | +34.831 | -0.051 | -0.093 | +0.025 | · | · | · | · | · |
| layer1.2 | -0.342 | +0.545 | -1.000 | +0.012 | +25.031 | +0.033 | -0.088 | +0.035 | · | · | · | · | · |
| layer2.0 | -0.895 | -0.280 | -1.000 | +0.019 | +9.700 | +0.015 | -0.084 | +0.009 | · | · | · | · | · |
| layer2.1 | -0.728 | -1.033 | -1.000 | +0.054 | +0.657 | +0.028 | -0.058 | +0.028 | · | · | · | · | · |
| layer2.2 | -1.239 | -0.022 | -2.000 | +0.048 | +1.331 | -0.032 | -0.200 | +0.017 | · | · | · | · | · |
| layer3.0 | -3.496 | -3.672 | -16.000 | -0.041 | +1.354 | -0.009 | -0.303 | -0.075 | · | · | · | · | · |
| layer3.1 | -0.074 | +4.110 | +1.000 | +0.079 | -0.110 | -0.069 | +0.007 | +0.012 | · | · | · | · | · |
| layer3.2 | +0.849 | +0.205 | +0.000 | -0.168 | +0.008 | -0.000 | +0.191 | -0.009 | · | · | · | · | · |
| penult | +0.849 | +0.205 | +0.000 | -0.168 | +0.008 | -0.000 | +0.191 | -0.009 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer2.0 | layer1.5 |
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
| corruption_family | layer1.2 | layer1.5 |
| corruption_type | layer2.0 | layer1.8 |
| severity | layer2.2 | layer3.0 |