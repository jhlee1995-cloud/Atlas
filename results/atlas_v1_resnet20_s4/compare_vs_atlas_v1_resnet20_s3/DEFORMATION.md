# DEFORMATION  A=`results/atlas_v1_resnet20_s3`  B=`results/atlas_v1_resnet20_s4`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.009 | 0.987 | 0.984 | True | 0.948 | 0.027 | 0.948 | 0.968 | 0.766 |
| layer1.0 | layer1.0 | 0.012 | 0.988 | 0.995 | True | 0.996 | 0.021 | 0.920 | 0.933 | 0.688 |
| layer1.1 | layer1.1 | 0.025 | 0.968 | 1.000 | True | 0.992 | 0.019 | 0.886 | 0.936 | 0.688 |
| layer1.2 | layer1.2 | 0.006 | 0.988 | 0.989 | True | 0.981 | 0.020 | 0.952 | 0.948 | 0.703 |
| layer2.0 | layer2.0 | 0.004 | 0.986 | 0.964 | True | 0.992 | 0.018 | 0.961 | 0.977 | 0.875 |
| layer2.1 | layer2.1 | 0.008 | 0.987 | 0.926 | True | 0.996 | 0.020 | 0.944 | 0.974 | 0.797 |
| layer2.2 | layer2.2 | 0.005 | 0.990 | 0.991 | True | 0.994 | 0.018 | 0.940 | 0.969 | 0.750 |
| layer3.0 | layer3.0 | 0.008 | 0.985 | 0.941 | True | 0.983 | 0.019 | 0.917 | 0.954 | 0.812 |
| layer3.1 | layer3.1 | 0.047 | 0.886 | 0.808 | True | 0.992 | 0.018 | 0.823 | 0.887 | 0.844 |
| layer3.2 | layer3.2 | 0.003 | 0.965 | 0.871 | True | 0.950 | 0.033 | 0.895 | 0.961 | 1.000 |
| penult | penult | 0.003 | 0.965 | 0.871 | True | 0.950 | 0.033 | 0.895 | 0.961 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.946 | 0.798 | 0.117 | 0.955 | 0.581 | 0.924 | 0.791 | 0.119 | 0.943 |
| layer1.0 | 0.933 | 0.730 | 0.124 | 0.948 | 0.581 | 0.889 | 0.712 | 0.127 | 0.939 |
| layer1.1 | 0.930 | 0.738 | 0.118 | 0.926 | 0.581 | 0.892 | 0.716 | 0.119 | 0.907 |
| layer1.2 | 0.969 | 0.784 | 0.112 | 0.943 | 0.581 | 0.947 | 0.767 | 0.122 | 0.934 |
| layer2.0 | 0.965 | 0.840 | 0.107 | 0.979 | 0.581 | 0.956 | 0.791 | 0.117 | 0.971 |
| layer2.1 | 0.947 | 0.785 | 0.105 | 0.961 | 0.581 | 0.932 | 0.744 | 0.113 | 0.952 |
| layer2.2 | 0.942 | 0.787 | 0.103 | 0.955 | 0.581 | 0.933 | 0.739 | 0.111 | 0.943 |
| layer3.0 | 0.916 | 0.835 | 0.102 | 0.948 | 0.581 | 0.893 | 0.707 | 0.114 | 0.929 |
| layer3.1 | 0.795 | 0.836 | 0.101 | 0.835 | 0.581 | 0.754 | 0.591 | 0.132 | 0.820 |
| layer3.2 | 0.889 | 0.938 | 0.100 | 0.831 | 0.581 | 0.708 | 0.581 | 0.124 | 0.781 |
| penult | 0.889 | 0.938 | 0.100 | 0.831 | 0.581 | 0.708 | 0.581 | 0.124 | 0.781 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | +0.067 | +0.446 | +0.000 | +0.012 | -159.612 | +0.001 | -0.022 | +0.004 | · | · | · | · | · |
| layer1.0 | +0.420 | -0.353 | +0.000 | +0.002 | -29.211 | -0.042 | +0.097 | +0.005 | · | · | · | · | · |
| layer1.1 | +0.331 | -0.051 | +0.000 | -0.012 | +17.088 | -0.073 | -0.015 | -0.006 | · | · | · | · | · |
| layer1.2 | +0.307 | -0.184 | +0.000 | -0.000 | -0.594 | -0.033 | +0.154 | +0.005 | · | · | · | · | · |
| layer2.0 | +0.352 | +0.294 | +0.000 | +0.007 | -3.695 | -0.013 | +0.033 | +0.011 | · | · | · | · | · |
| layer2.1 | +0.413 | +0.510 | +0.000 | +0.005 | -1.344 | -0.025 | +0.094 | +0.015 | · | · | · | · | · |
| layer2.2 | +0.052 | +0.037 | +1.000 | +0.004 | +0.005 | -0.005 | +0.002 | +0.013 | · | · | · | · | · |
| layer3.0 | +0.328 | -1.098 | +0.000 | +0.024 | -0.345 | -0.018 | +0.004 | +0.006 | · | · | · | · | · |
| layer3.1 | -0.288 | -3.157 | -1.000 | +0.013 | -0.131 | +0.019 | +0.105 | +0.027 | · | · | · | · | · |
| layer3.2 | +0.065 | +0.126 | +0.000 | +0.054 | -0.005 | -0.011 | +0.031 | +0.004 | · | · | · | · | · |
| penult | +0.065 | +0.126 | +0.000 | +0.054 | -0.005 | -0.011 | +0.031 | +0.004 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer1.0 | layer2.0 |
| spectral_slope | layer2.0 | layer2.0 |
| spectral_anisotropy | layer2.0 | layer2.0 |
| noise_sigma | layer1.0 | layer1.0 |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | layer1.0 | stem |
| edge_density | layer1.0 | layer1.0 |
| orientation_entropy | layer3.0 | layer2.0 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.1 |
| coarse_animal_vehicle | layer3.0 | layer3.0 |
| corruption_family | layer2.0 | layer2.0 |
| corruption_type | layer2.0 | layer2.0 |
| severity | layer3.0 | layer2.0 |