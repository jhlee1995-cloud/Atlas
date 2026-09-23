# DEFORMATION  A=`results/atlas_v1_resnet20_s3_st3`  B=`results/atlas_v1_resnet56_e60`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.043 | 0.967 | 0.982 | True | 0.975 | 0.026 | 0.908 | 0.740 | 0.578 |
| layer1.0 | layer1.0 | 0.024 | 0.992 | 0.997 | True | 0.927 | 0.057 | 0.913 | 0.930 | 0.656 |
| layer1.1 | layer1.5 | 0.024 | 0.975 | 0.942 | True | 0.981 | 0.032 | 0.881 | 0.923 | 0.609 |
| layer1.2 | layer1.8 | 0.018 | 0.984 | 0.997 | True | 0.981 | 0.039 | 0.933 | 0.938 | 0.766 |
| layer2.0 | layer2.0 | 0.008 | 0.984 | 0.910 | True | 0.992 | 0.014 | 0.928 | 0.962 | 0.766 |
| layer2.1 | layer2.5 | 0.012 | 0.980 | 0.973 | True | 0.994 | 0.017 | 0.948 | 0.977 | 0.812 |
| layer2.2 | layer2.8 | 0.011 | 0.984 | 0.505 | True | 0.994 | 0.019 | 0.939 | 0.970 | 0.766 |
| layer3.0 | layer3.0 | 0.017 | 0.969 | 0.985 | True | 0.992 | 0.016 | 0.866 | 0.931 | 0.734 |
| layer3.1 | layer3.5 | 0.040 | 0.914 | 0.973 | True | 0.979 | 0.023 | 0.779 | 0.873 | 0.859 |
| layer3.2 | layer3.8 | 0.007 | 0.916 | 0.676 | True | 0.955 | 0.034 | 0.908 | 0.960 | 1.000 |
| penult | penult | 0.007 | 0.916 | 0.676 | True | 0.955 | 0.034 | 0.908 | 0.960 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.828 | 0.631 | 0.114 | 0.746 | 0.544 | 0.813 | 0.618 | 0.118 | 0.734 |
| layer1.0 | 0.910 | 0.690 | 0.125 | 0.941 | 0.544 | 0.874 | 0.656 | 0.126 | 0.934 |
| layer1.1 | 0.877 | 0.685 | 0.113 | 0.914 | 0.544 | 0.808 | 0.627 | 0.117 | 0.871 |
| layer1.2 | 0.938 | 0.739 | 0.110 | 0.934 | 0.544 | 0.913 | 0.712 | 0.121 | 0.920 |
| layer2.0 | 0.947 | 0.798 | 0.106 | 0.958 | 0.544 | 0.922 | 0.747 | 0.116 | 0.946 |
| layer2.1 | 0.948 | 0.782 | 0.105 | 0.960 | 0.544 | 0.936 | 0.737 | 0.111 | 0.953 |
| layer2.2 | 0.933 | 0.770 | 0.103 | 0.954 | 0.544 | 0.919 | 0.723 | 0.110 | 0.941 |
| layer3.0 | 0.887 | 0.789 | 0.102 | 0.925 | 0.544 | 0.854 | 0.689 | 0.111 | 0.913 |
| layer3.1 | 0.782 | 0.820 | 0.101 | 0.808 | 0.544 | 0.750 | 0.591 | 0.123 | 0.757 |
| layer3.2 | 0.871 | 0.926 | 0.100 | 0.787 | 0.544 | 0.671 | 0.557 | 0.125 | 0.752 |
| penult | 0.871 | 0.926 | 0.100 | 0.787 | 0.544 | 0.671 | 0.557 | 0.125 | 0.752 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | +0.214 | +0.328 | +0.000 | +0.010 | -173.491 | -0.028 | -0.015 | -0.001 | · | · | · | · | · |
| layer1.0 | -0.679 | -1.209 | -2.000 | -0.019 | +19.564 | +0.070 | -0.204 | -0.026 | · | · | · | · | · |
| layer1.1 | +0.142 | -0.125 | +0.000 | +0.021 | +7.811 | -0.047 | -0.091 | +0.011 | · | · | · | · | · |
| layer1.2 | -0.188 | +0.014 | +0.000 | +0.034 | -3.992 | -0.015 | +0.047 | +0.034 | · | · | · | · | · |
| layer2.0 | -0.935 | -0.319 | -2.000 | +0.034 | +1.242 | -0.020 | -0.174 | +0.017 | · | · | · | · | · |
| layer2.1 | -0.375 | -0.107 | -2.000 | +0.056 | -3.713 | -0.034 | -0.154 | +0.038 | · | · | · | · | · |
| layer2.2 | -0.732 | -1.483 | -2.000 | +0.039 | -1.633 | +0.017 | -0.171 | +0.014 | · | · | · | · | · |
| layer3.0 | -2.891 | -3.522 | -16.000 | -0.015 | +1.132 | +0.001 | -0.642 | -0.054 | · | · | · | · | · |
| layer3.1 | -0.360 | +1.245 | +1.000 | -0.036 | -0.086 | -0.031 | +0.241 | -0.006 | · | · | · | · | · |
| layer3.2 | +0.930 | +0.495 | +0.000 | +0.398 | -0.043 | -0.027 | +0.102 | +0.002 | · | · | · | · | · |
| penult | +0.930 | +0.495 | +0.000 | +0.398 | -0.043 | -0.027 | +0.102 | +0.002 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer1.0 | layer1.5 |
| spectral_slope | layer2.0 | layer1.5 |
| spectral_anisotropy | layer2.0 | layer2.5 |
| noise_sigma | layer1.0 | layer1.5 |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | layer1.0 | layer1.0 |
| edge_density | layer1.0 | layer1.0 |
| orientation_entropy | layer3.0 | layer2.5 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.5 |
| coarse_animal_vehicle | layer3.0 | layer3.5 |
| corruption_family | layer2.0 | layer2.0 |
| corruption_type | layer2.0 | layer2.0 |
| severity | layer3.0 | layer3.0 |