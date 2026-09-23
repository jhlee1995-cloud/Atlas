# DEFORMATION  A=`results/atlas_v1_resnet20_s0hub_st2`  B=`results/atlas_v1_resnet56_s0hub`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.006 | 0.992 | 0.984 | True | 0.944 | 0.021 | 0.966 | 0.985 | 0.875 |
| layer1.0 | layer1.0 | 0.028 | 0.979 | 1.000 | True | 0.975 | 0.070 | 0.926 | 0.938 | 0.656 |
| layer1.1 | layer1.5 | 0.011 | 0.981 | 1.000 | True | 0.992 | 0.027 | 0.873 | 0.930 | 0.703 |
| layer1.2 | layer1.8 | 0.006 | 0.992 | 0.991 | True | 0.996 | 0.025 | 0.957 | 0.966 | 0.766 |
| layer2.0 | layer2.0 | 0.019 | 0.966 | 0.945 | True | 0.983 | 0.021 | 0.948 | 0.919 | 0.797 |
| layer2.1 | layer2.5 | 0.017 | 0.969 | 0.937 | True | 0.988 | 0.017 | 0.932 | 0.913 | 0.859 |
| layer2.2 | layer2.8 | 0.011 | 0.980 | 0.506 | True | 0.988 | 0.014 | 0.919 | 0.927 | 0.750 |
| layer3.0 | layer3.0 | 0.026 | 0.938 | 0.968 | True | 0.961 | 0.033 | 0.888 | 0.935 | 0.844 |
| layer3.1 | layer3.5 | 0.064 | 0.857 | 0.879 | True | 0.981 | 0.037 | 0.775 | 0.857 | 0.797 |
| layer3.2 | layer3.8 | 0.014 | 0.858 | 0.781 | True | 0.905 | 0.027 | 0.900 | 0.948 | 1.000 |
| penult | penult | 0.014 | 0.858 | 0.781 | True | 0.905 | 0.027 | 0.900 | 0.948 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.968 | 0.848 | 0.116 | 0.981 | 0.526 | 0.961 | 0.855 | 0.121 | 0.976 |
| layer1.0 | 0.920 | 0.684 | 0.114 | 0.933 | 0.526 | 0.901 | 0.665 | 0.120 | 0.918 |
| layer1.1 | 0.909 | 0.705 | 0.113 | 0.930 | 0.526 | 0.830 | 0.679 | 0.116 | 0.906 |
| layer1.2 | 0.960 | 0.790 | 0.112 | 0.964 | 0.526 | 0.940 | 0.780 | 0.123 | 0.962 |
| layer2.0 | 0.936 | 0.730 | 0.105 | 0.908 | 0.526 | 0.930 | 0.691 | 0.113 | 0.895 |
| layer2.1 | 0.919 | 0.725 | 0.104 | 0.911 | 0.526 | 0.910 | 0.670 | 0.110 | 0.884 |
| layer2.2 | 0.905 | 0.737 | 0.103 | 0.915 | 0.526 | 0.901 | 0.662 | 0.108 | 0.897 |
| layer3.0 | 0.880 | 0.786 | 0.102 | 0.905 | 0.526 | 0.872 | 0.683 | 0.110 | 0.890 |
| layer3.1 | 0.737 | 0.837 | 0.101 | 0.725 | 0.526 | 0.714 | 0.601 | 0.141 | 0.695 |
| layer3.2 | 0.863 | 0.927 | 0.100 | 0.717 | 0.526 | 0.644 | 0.571 | 0.124 | 0.716 |
| penult | 0.863 | 0.927 | 0.100 | 0.717 | 0.526 | 0.644 | 0.571 | 0.124 | 0.716 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | -0.279 | -0.037 | +0.000 | -0.001 | +234.463 | +0.022 | -0.065 | +0.005 | · | · | · | · | · |
| layer1.0 | -1.044 | -0.857 | -1.000 | -0.018 | +33.611 | +0.063 | -0.249 | -0.041 | · | · | · | · | · |
| layer1.1 | -0.491 | -0.053 | +0.000 | +0.025 | -4.364 | +0.015 | -0.035 | -0.007 | · | · | · | · | · |
| layer1.2 | -0.090 | -0.027 | +0.000 | +0.031 | +3.345 | -0.017 | -0.015 | +0.011 | · | · | · | · | · |
| layer2.0 | -0.869 | -0.575 | -2.000 | +0.006 | +2.219 | -0.002 | -0.051 | -0.022 | · | · | · | · | · |
| layer2.1 | +0.124 | +0.904 | +1.000 | +0.008 | -1.500 | -0.050 | +0.047 | +0.009 | · | · | · | · | · |
| layer2.2 | +0.074 | +0.867 | +0.000 | +0.008 | -0.841 | -0.053 | +0.237 | +0.019 | · | · | · | · | · |
| layer3.0 | -2.530 | -2.103 | -12.000 | -0.029 | +1.086 | +0.018 | -0.425 | -0.027 | · | · | · | · | · |
| layer3.1 | +0.040 | +5.359 | +3.000 | -0.005 | -0.088 | -0.066 | -0.028 | +0.034 | · | · | · | · | · |
| layer3.2 | +0.755 | +0.555 | +0.000 | +2.336 | -0.121 | -0.053 | +0.177 | +0.021 | · | · | · | · | · |
| penult | +0.755 | +0.555 | +0.000 | +2.336 | -0.121 | -0.053 | +0.177 | +0.021 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer1.0 | layer2.0 |
| spectral_slope | layer2.0 | layer2.0 |
| spectral_anisotropy | layer2.0 | layer3.0 |
| noise_sigma | layer1.0 | layer1.0 |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | layer1.0 | layer1.0 |
| edge_density | layer1.0 | layer1.0 |
| orientation_entropy | layer2.0 | layer2.0 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.5 |
| coarse_animal_vehicle | layer3.0 | layer3.5 |
| corruption_family | layer2.0 | layer2.0 |
| corruption_type | layer1.2 | layer2.0 |
| severity | layer3.0 | layer3.0 |