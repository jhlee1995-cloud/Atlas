# DEFORMATION  A=`results/atlas_v1_resnet20_s3_st3`  B=`results/atlas_v1_resnet56_e50`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.028 | 0.972 | 0.940 | True | 0.986 | 0.024 | 0.973 | 0.847 | 0.672 |
| layer1.0 | layer1.0 | 0.048 | 0.985 | 0.946 | True | 0.892 | 0.108 | 0.892 | 0.933 | 0.578 |
| layer1.1 | layer1.5 | 0.011 | 0.992 | 0.937 | True | 0.981 | 0.036 | 0.916 | 0.959 | 0.562 |
| layer1.2 | layer1.8 | 0.007 | 0.990 | 0.989 | True | 0.981 | 0.028 | 0.961 | 0.947 | 0.688 |
| layer2.0 | layer2.0 | 0.011 | 0.985 | 0.937 | True | 0.996 | 0.013 | 0.934 | 0.955 | 0.750 |
| layer2.1 | layer2.5 | 0.009 | 0.991 | 0.997 | True | 0.992 | 0.020 | 0.943 | 0.966 | 0.750 |
| layer2.2 | layer2.8 | 0.014 | 0.988 | 0.972 | True | 0.979 | 0.018 | 0.911 | 0.939 | 0.766 |
| layer3.0 | layer3.0 | 0.011 | 0.978 | 0.974 | True | 0.979 | 0.023 | 0.905 | 0.938 | 0.797 |
| layer3.1 | layer3.5 | 0.031 | 0.936 | 0.964 | True | 0.992 | 0.025 | 0.836 | 0.886 | 0.812 |
| layer3.2 | layer3.8 | 0.008 | 0.916 | 0.611 | True | 0.992 | 0.028 | 0.886 | 0.953 | 1.000 |
| penult | penult | 0.008 | 0.916 | 0.611 | True | 0.992 | 0.028 | 0.886 | 0.953 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.878 | 0.663 | 0.112 | 0.850 | 0.503 | 0.862 | 0.664 | 0.117 | 0.838 |
| layer1.0 | 0.893 | 0.637 | 0.122 | 0.925 | 0.503 | 0.862 | 0.613 | 0.127 | 0.915 |
| layer1.1 | 0.914 | 0.701 | 0.118 | 0.944 | 0.503 | 0.876 | 0.691 | 0.119 | 0.929 |
| layer1.2 | 0.951 | 0.762 | 0.113 | 0.938 | 0.503 | 0.935 | 0.731 | 0.121 | 0.927 |
| layer2.0 | 0.945 | 0.780 | 0.107 | 0.949 | 0.503 | 0.925 | 0.763 | 0.117 | 0.940 |
| layer2.1 | 0.949 | 0.789 | 0.105 | 0.953 | 0.503 | 0.933 | 0.734 | 0.111 | 0.939 |
| layer2.2 | 0.924 | 0.762 | 0.103 | 0.932 | 0.503 | 0.902 | 0.702 | 0.110 | 0.912 |
| layer3.0 | 0.898 | 0.780 | 0.103 | 0.936 | 0.503 | 0.866 | 0.699 | 0.111 | 0.915 |
| layer3.1 | 0.800 | 0.842 | 0.101 | 0.819 | 0.503 | 0.763 | 0.603 | 0.124 | 0.760 |
| layer3.2 | 0.876 | 0.921 | 0.100 | 0.752 | 0.503 | 0.675 | 0.563 | 0.123 | 0.737 |
| penult | 0.876 | 0.921 | 0.100 | 0.752 | 0.503 | 0.675 | 0.563 | 0.123 | 0.737 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | -0.169 | +0.266 | +0.000 | +0.009 | -16.470 | +0.011 | +0.024 | +0.004 | · | · | · | · | · |
| layer1.0 | -1.604 | -1.406 | -2.000 | -0.017 | +384.704 | +0.102 | -0.278 | -0.035 | · | · | · | · | · |
| layer1.1 | -0.945 | -1.077 | -1.000 | -0.017 | +44.485 | +0.042 | -0.181 | -0.018 | · | · | · | · | · |
| layer1.2 | -0.408 | -0.324 | -1.000 | -0.006 | +32.106 | +0.020 | -0.103 | -0.004 | · | · | · | · | · |
| layer2.0 | -0.963 | -0.504 | -3.000 | -0.008 | +7.078 | -0.001 | -0.255 | -0.010 | · | · | · | · | · |
| layer2.1 | -0.152 | -0.568 | -2.000 | +0.058 | -4.180 | -0.023 | -0.166 | +0.021 | · | · | · | · | · |
| layer2.2 | -0.574 | -1.356 | -2.000 | +0.050 | -0.972 | +0.019 | -0.222 | +0.002 | · | · | · | · | · |
| layer3.0 | -2.838 | -2.684 | -15.000 | -0.034 | +1.281 | +0.024 | -0.366 | -0.050 | · | · | · | · | · |
| layer3.1 | -0.765 | +1.487 | -2.000 | +0.031 | -0.139 | -0.036 | -0.029 | +0.026 | · | · | · | · | · |
| layer3.2 | +1.193 | +0.512 | +0.000 | +0.170 | -0.027 | -0.017 | +0.253 | -0.002 | · | · | · | · | · |
| penult | +1.193 | +0.512 | +0.000 | +0.170 | -0.027 | -0.017 | +0.253 | -0.002 | · | · | · | · | · |

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
| orientation_entropy | layer3.0 | layer3.0 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.5 |
| coarse_animal_vehicle | layer3.0 | layer3.5 |
| corruption_family | layer2.0 | layer2.0 |
| corruption_type | layer2.0 | layer2.0 |
| severity | layer3.0 | layer3.0 |