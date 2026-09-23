# DEFORMATION  A=`results/atlas_v1_resnet20_s1`  B=`results/atlas_v1_resnet20_s1_ref1`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.006 | 0.993 | 0.995 | True | 0.992 | 0.010 | 0.039 | 0.243 | 0.094 |
| layer1.0 | layer1.0 | 0.004 | 0.990 | 0.953 | True | 0.992 | 0.008 | 0.047 | 0.218 | 0.094 |
| layer1.1 | layer1.1 | 0.004 | 0.993 | 0.953 | True | 1.000 | 0.011 | 0.054 | 0.256 | 0.125 |
| layer1.2 | layer1.2 | 0.003 | 0.994 | 0.991 | True | 0.998 | 0.009 | 0.071 | 0.346 | 0.141 |
| layer2.0 | layer2.0 | 0.002 | 0.990 | 0.991 | True | 1.000 | 0.009 | 0.061 | 0.201 | 0.094 |
| layer2.1 | layer2.1 | 0.002 | 0.991 | 0.954 | True | 0.994 | 0.011 | 0.084 | 0.164 | 0.047 |
| layer2.2 | layer2.2 | 0.002 | 0.994 | 0.969 | True | 1.000 | 0.010 | 0.100 | 0.120 | 0.078 |
| layer3.0 | layer3.0 | 0.001 | 0.994 | 0.982 | True | 0.998 | 0.012 | 0.123 | 0.110 | 0.109 |
| layer3.1 | layer3.1 | 0.000 | 0.997 | 0.977 | True | 0.996 | 0.014 | 0.127 | 0.084 | 0.062 |
| layer3.2 | layer3.2 | 0.000 | 0.994 | 0.805 | True | 0.971 | 0.013 | 0.100 | -0.026 | 0.047 |
| penult | penult | 0.000 | 0.994 | 0.805 | True | 0.971 | 0.013 | 0.100 | -0.026 | 0.047 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 1.000 | 0.881 | 0.117 | 0.992 | 1.000 | 1.000 | 0.886 | 0.118 | 0.992 |
| layer1.0 | 1.000 | 0.911 | 0.118 | 0.994 | 1.000 | 1.000 | 0.894 | 0.119 | 0.994 |
| layer1.1 | 1.000 | 0.918 | 0.116 | 0.995 | 1.000 | 1.000 | 0.904 | 0.121 | 0.995 |
| layer1.2 | 1.000 | 0.933 | 0.113 | 0.997 | 1.000 | 1.000 | 0.917 | 0.120 | 0.997 |
| layer2.0 | 1.000 | 0.942 | 0.106 | 0.997 | 1.000 | 1.000 | 0.935 | 0.113 | 0.997 |
| layer2.1 | 1.000 | 0.942 | 0.106 | 0.997 | 1.000 | 1.000 | 0.940 | 0.112 | 0.997 |
| layer2.2 | 1.000 | 0.945 | 0.104 | 0.998 | 1.000 | 1.000 | 0.935 | 0.113 | 0.997 |
| layer3.0 | 1.000 | 0.967 | 0.103 | 0.998 | 1.000 | 1.000 | 0.950 | 0.113 | 0.998 |
| layer3.1 | 1.000 | 0.990 | 0.102 | 0.999 | 1.000 | 1.000 | 0.973 | 0.144 | 0.999 |
| layer3.2 | 1.000 | 0.999 | 0.100 | 0.999 | 1.000 | 1.000 | 0.995 | 0.127 | 1.000 |
| penult | 1.000 | 0.999 | 0.100 | 0.999 | 1.000 | 1.000 | 0.995 | 0.127 | 1.000 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers |
|---|---|---|---|---|---|---|---|---|
| stem | -0.234 | -0.002 | +0.000 | +0.030 | -69.672 | +0.002 | -0.026 | +0.003 |
| layer1.0 | -0.147 | +0.018 | +0.000 | +0.028 | -27.455 | +0.004 | -0.047 | +0.001 |
| layer1.1 | -0.019 | +0.041 | +0.000 | +0.018 | -15.910 | -0.009 | -0.010 | +0.000 |
| layer1.2 | +0.011 | -0.024 | +0.000 | +0.012 | +16.582 | -0.008 | +0.071 | +0.001 |
| layer2.0 | -0.240 | +0.039 | +0.000 | +0.007 | -0.673 | -0.014 | +0.029 | -0.002 |
| layer2.1 | +0.290 | +0.024 | +0.000 | +0.011 | -1.704 | -0.015 | +0.015 | -0.005 |
| layer2.2 | +0.504 | +0.030 | +0.000 | +0.010 | -0.553 | -0.012 | -0.058 | -0.000 |
| layer3.0 | +0.158 | +0.018 | +0.000 | +0.006 | -0.101 | -0.008 | -0.122 | -0.002 |
| layer3.1 | +0.089 | -0.106 | +0.000 | +0.005 | -0.006 | -0.001 | -0.169 | -0.003 |
| layer3.2 | -0.017 | -0.039 | +0.000 | +0.011 | -0.002 | +0.002 | -0.087 | +0.000 |
| penult | -0.017 | -0.039 | +0.000 | +0.011 | -0.002 | +0.002 | -0.087 | +0.000 |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer1.1 | layer1.1 |
| spectral_slope | layer2.0 | layer2.0 |
| spectral_anisotropy | layer1.2 | layer2.0 |
| noise_sigma | layer1.0 | layer1.0 |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | stem | stem |
| edge_density | layer1.0 | layer1.0 |
| orientation_entropy | layer2.0 | layer2.0 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.1 |
| coarse_animal_vehicle | layer3.0 | layer3.0 |
| corruption_family | layer2.0 | layer2.0 |
| corruption_type | layer2.0 | layer2.0 |
| severity | layer3.0 | layer3.0 |