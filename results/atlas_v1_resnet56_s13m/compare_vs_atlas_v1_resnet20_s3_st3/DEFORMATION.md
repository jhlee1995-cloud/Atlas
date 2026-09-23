# DEFORMATION  A=`results/atlas_v1_resnet20_s3_st3`  B=`results/atlas_v1_resnet56_s13m`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.011 | 0.989 | 0.984 | True | 0.977 | 0.044 | 0.870 | 0.955 | 0.766 |
| layer1.0 | layer1.0 | 0.037 | 0.982 | 0.997 | True | 0.917 | 0.070 | 0.850 | 0.949 | 0.625 |
| layer1.1 | layer1.5 | 0.007 | 0.991 | 0.937 | True | 0.983 | 0.027 | 0.921 | 0.958 | 0.719 |
| layer1.2 | layer1.8 | 0.006 | 0.985 | 1.000 | True | 0.988 | 0.017 | 0.951 | 0.944 | 0.719 |
| layer2.0 | layer2.0 | 0.006 | 0.990 | 0.934 | True | 0.996 | 0.019 | 0.955 | 0.979 | 0.750 |
| layer2.1 | layer2.5 | 0.022 | 0.966 | 0.549 | True | 0.975 | 0.030 | 0.938 | 0.947 | 0.703 |
| layer2.2 | layer2.8 | 0.017 | 0.972 | 0.539 | True | 0.969 | 0.028 | 0.921 | 0.936 | 0.766 |
| layer3.0 | layer3.0 | 0.012 | 0.964 | 0.930 | True | 0.975 | 0.025 | 0.888 | 0.938 | 0.828 |
| layer3.1 | layer3.5 | 0.034 | 0.925 | 0.789 | True | 0.990 | 0.028 | 0.797 | 0.884 | 0.797 |
| layer3.2 | layer3.8 | 0.007 | 0.923 | 0.525 | True | 0.988 | 0.025 | 0.897 | 0.955 | 1.000 |
| penult | penult | 0.007 | 0.923 | 0.525 | True | 0.988 | 0.025 | 0.897 | 0.955 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.911 | 0.705 | 0.114 | 0.951 | 0.518 | 0.858 | 0.716 | 0.116 | 0.938 |
| layer1.0 | 0.867 | 0.634 | 0.120 | 0.932 | 0.518 | 0.829 | 0.631 | 0.123 | 0.921 |
| layer1.1 | 0.908 | 0.731 | 0.120 | 0.948 | 0.518 | 0.844 | 0.711 | 0.121 | 0.929 |
| layer1.2 | 0.952 | 0.778 | 0.112 | 0.944 | 0.518 | 0.923 | 0.738 | 0.123 | 0.927 |
| layer2.0 | 0.960 | 0.838 | 0.107 | 0.972 | 0.518 | 0.941 | 0.803 | 0.117 | 0.963 |
| layer2.1 | 0.936 | 0.781 | 0.105 | 0.942 | 0.518 | 0.927 | 0.729 | 0.110 | 0.934 |
| layer2.2 | 0.916 | 0.783 | 0.104 | 0.937 | 0.518 | 0.900 | 0.719 | 0.110 | 0.927 |
| layer3.0 | 0.892 | 0.789 | 0.103 | 0.934 | 0.518 | 0.858 | 0.699 | 0.111 | 0.915 |
| layer3.1 | 0.777 | 0.818 | 0.101 | 0.821 | 0.518 | 0.740 | 0.608 | 0.124 | 0.765 |
| layer3.2 | 0.873 | 0.925 | 0.100 | 0.767 | 0.518 | 0.680 | 0.560 | 0.124 | 0.744 |
| penult | 0.873 | 0.925 | 0.100 | 0.767 | 0.518 | 0.680 | 0.560 | 0.124 | 0.744 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | -1.055 | +0.308 | -1.000 | +0.017 | -41.965 | -0.010 | -0.218 | -0.009 | · | · | · | · | · |
| layer1.0 | -2.326 | -1.160 | -3.000 | -0.027 | +86.479 | +0.063 | -0.524 | -0.045 | · | · | · | · | · |
| layer1.1 | -0.734 | -0.819 | -1.000 | +0.008 | +62.085 | +0.040 | -0.212 | -0.009 | · | · | · | · | · |
| layer1.2 | -0.113 | -0.126 | +0.000 | +0.014 | +10.851 | +0.004 | -0.048 | +0.022 | · | · | · | · | · |
| layer2.0 | -1.035 | -0.101 | -2.000 | +0.007 | +4.476 | -0.019 | -0.212 | +0.009 | · | · | · | · | · |
| layer2.1 | -0.301 | +0.060 | -1.000 | +0.059 | -6.125 | -0.063 | +0.051 | +0.051 | · | · | · | · | · |
| layer2.2 | -0.949 | -1.426 | -2.000 | +0.026 | -0.834 | -0.025 | -0.114 | +0.006 | · | · | · | · | · |
| layer3.0 | -3.298 | -2.680 | -17.000 | -0.020 | +1.181 | -0.004 | -0.608 | -0.041 | · | · | · | · | · |
| layer3.1 | -0.162 | +2.142 | +1.000 | -0.063 | +0.088 | -0.033 | +0.273 | -0.033 | · | · | · | · | · |
| layer3.2 | +0.891 | +0.611 | +0.000 | +0.344 | -0.042 | -0.029 | +0.136 | +0.000 | · | · | · | · | · |
| penult | +0.891 | +0.611 | +0.000 | +0.344 | -0.042 | -0.029 | +0.136 | +0.000 | · | · | · | · | · |

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
| orientation_entropy | layer3.0 | layer3.0 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.5 |
| coarse_animal_vehicle | layer3.0 | layer3.5 |
| corruption_family | layer2.0 | layer1.8 |
| corruption_type | layer2.0 | layer1.8 |
| severity | layer3.0 | layer3.0 |