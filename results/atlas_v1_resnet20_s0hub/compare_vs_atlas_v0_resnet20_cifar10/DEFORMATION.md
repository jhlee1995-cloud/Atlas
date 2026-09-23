# DEFORMATION  A=`results/atlas_v0_resnet20_cifar10`  B=`results/atlas_v1_resnet20_s0hub`  same_space=True
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.001 | 0.999 | 1.000 | True | 0.971 | 0.026 | 0.991 | 0.990 | 0.891 |
| layer1.0 | layer1.0 | 0.001 | 0.998 | 0.962 | True | 0.996 | 0.006 | 0.989 | 0.991 | 0.844 |
| layer1.1 | layer1.1 | 0.003 | 0.991 | 0.942 | True | 0.996 | 0.007 | 0.990 | 0.988 | 0.812 |
| layer1.2 | layer1.2 | 0.003 | 0.991 | 0.854 | True | 0.998 | 0.005 | 0.991 | 0.989 | 0.922 |
| layer2.0 | layer2.0 | 0.002 | 0.994 | 0.969 | True | 1.000 | 0.007 | 0.987 | 0.989 | 0.844 |
| layer2.1 | layer2.1 | 0.002 | 0.995 | 0.960 | True | 0.998 | 0.007 | 0.988 | 0.991 | 0.875 |
| layer2.2 | layer2.2 | 0.002 | 0.992 | 0.985 | True | 0.996 | 0.007 | 0.988 | 0.992 | 0.938 |
| layer3.0 | layer3.0 | 0.001 | 0.991 | 1.000 | True | 0.998 | 0.010 | 0.990 | 0.993 | 0.969 |
| layer3.1 | layer3.1 | 0.000 | 0.999 | 0.991 | True | 0.996 | 0.012 | 0.991 | 0.996 | 0.969 |
| layer3.2 | layer3.2 | 0.000 | 0.983 | 0.952 | True | 0.986 | 0.013 | 0.991 | 0.997 | 1.000 |
| penult | penult | 0.000 | 0.983 | 0.952 | True | 0.986 | 0.013 | 0.991 | 0.997 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.989 | 0.888 | 0.119 | 0.983 | 0.853 | 0.990 | 0.876 | 0.123 | 0.978 |
| layer1.0 | 0.987 | 0.900 | 0.122 | 0.988 | 0.853 | 0.984 | 0.891 | 0.131 | 0.986 |
| layer1.1 | 0.987 | 0.897 | 0.112 | 0.985 | 0.853 | 0.987 | 0.867 | 0.115 | 0.982 |
| layer1.2 | 0.991 | 0.901 | 0.115 | 0.983 | 0.853 | 0.990 | 0.880 | 0.126 | 0.983 |
| layer2.0 | 0.988 | 0.901 | 0.110 | 0.985 | 0.853 | 0.987 | 0.888 | 0.123 | 0.983 |
| layer2.1 | 0.989 | 0.904 | 0.107 | 0.988 | 0.853 | 0.986 | 0.890 | 0.117 | 0.984 |
| layer2.2 | 0.988 | 0.916 | 0.105 | 0.988 | 0.853 | 0.986 | 0.895 | 0.115 | 0.983 |
| layer3.0 | 0.988 | 0.938 | 0.102 | 0.990 | 0.853 | 0.985 | 0.877 | 0.113 | 0.986 |
| layer3.1 | 0.989 | 0.961 | 0.102 | 0.993 | 0.853 | 0.977 | 0.898 | 0.147 | 0.990 |
| layer3.2 | 0.989 | 0.975 | 0.100 | 0.984 | 0.853 | 0.962 | 0.866 | 0.127 | 0.969 |
| penult | 0.989 | 0.975 | 0.100 | 0.984 | 0.853 | 0.962 | 0.866 | 0.127 | 0.969 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers |
|---|---|---|---|---|---|---|---|---|
| stem | +0.206 | +0.090 | +0.000 | +0.002 | -50.560 | +0.006 | +0.027 | +0.012 |
| layer1.0 | +0.132 | +0.139 | +0.000 | +0.005 | -34.175 | +0.020 | +0.058 | +0.001 |
| layer1.1 | +0.091 | +0.006 | +0.000 | +0.003 | -0.231 | +0.025 | -0.003 | +0.001 |
| layer1.2 | +0.100 | +0.370 | +0.000 | -0.005 | -3.768 | +0.002 | +0.002 | +0.005 |
| layer2.0 | +0.073 | +0.283 | +1.000 | -0.004 | -0.034 | +0.017 | -0.022 | +0.002 |
| layer2.1 | +0.172 | +0.233 | +0.000 | -0.005 | +0.267 | +0.014 | +0.044 | -0.004 |
| layer2.2 | +0.097 | +0.245 | +0.000 | -0.011 | +0.129 | +0.007 | -0.035 | -0.002 |
| layer3.0 | +0.091 | +0.325 | +0.000 | -0.000 | -0.062 | +0.001 | +0.012 | +0.009 |
| layer3.1 | -0.104 | -0.291 | +0.000 | +0.013 | -0.020 | +0.005 | -0.003 | +0.002 |
| layer3.2 | +0.046 | +0.020 | +0.000 | +0.055 | -0.008 | -0.002 | +0.024 | +0.006 |
| penult | +0.046 | +0.020 | +0.000 | +0.055 | -0.008 | -0.002 | +0.024 | +0.006 |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer1.0 | layer1.0 |
| spectral_slope | layer2.0 | layer2.0 |
| spectral_anisotropy | layer2.0 | layer2.0 |
| noise_sigma | layer1.0 | layer1.0 |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | layer1.0 | layer1.0 |
| edge_density | layer1.0 | layer1.0 |
| orientation_entropy | layer2.0 | layer2.0 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.1 |
| coarse_animal_vehicle | layer3.0 | layer3.0 |
| corruption_family | layer2.0 | layer2.0 |
| corruption_type | layer2.0 | layer1.2 |
| severity | layer3.0 | layer3.0 |