# DEFORMATION  A=`results/atlas_v1_resnet20_s1_st3`  B=`results/atlas_v1_resnet56_s12m`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.022 | 0.983 | 0.938 | True | 0.934 | 0.040 | 0.970 | 0.916 | 0.781 |
| layer1.0 | layer1.0 | 0.008 | 0.995 | 1.000 | True | 0.948 | 0.047 | 0.941 | 0.957 | 0.719 |
| layer1.1 | layer1.5 | 0.028 | 0.962 | 0.962 | True | 0.979 | 0.032 | 0.903 | 0.907 | 0.656 |
| layer1.2 | layer1.8 | 0.012 | 0.988 | 0.987 | True | 0.981 | 0.020 | 0.945 | 0.952 | 0.781 |
| layer2.0 | layer2.0 | 0.044 | 0.938 | 0.955 | True | 0.994 | 0.021 | 0.885 | 0.908 | 0.672 |
| layer2.1 | layer2.5 | 0.012 | 0.979 | 0.960 | True | 0.979 | 0.035 | 0.933 | 0.958 | 0.797 |
| layer2.2 | layer2.8 | 0.011 | 0.984 | 0.992 | True | 0.983 | 0.032 | 0.929 | 0.961 | 0.828 |
| layer3.0 | layer3.0 | 0.009 | 0.975 | 0.977 | True | 0.975 | 0.028 | 0.895 | 0.948 | 0.781 |
| layer3.1 | layer3.5 | 0.056 | 0.805 | 0.941 | True | 0.990 | 0.025 | 0.791 | 0.868 | 0.859 |
| layer3.2 | layer3.8 | 0.004 | 0.933 | 0.959 | True | 0.963 | 0.027 | 0.896 | 0.954 | 1.000 |
| penult | penult | 0.004 | 0.933 | 0.959 | True | 0.963 | 0.027 | 0.896 | 0.954 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.917 | 0.748 | 0.113 | 0.903 | 0.512 | 0.901 | 0.735 | 0.114 | 0.896 |
| layer1.0 | 0.954 | 0.724 | 0.115 | 0.933 | 0.512 | 0.898 | 0.695 | 0.119 | 0.924 |
| layer1.1 | 0.879 | 0.679 | 0.113 | 0.892 | 0.512 | 0.849 | 0.656 | 0.115 | 0.880 |
| layer1.2 | 0.943 | 0.747 | 0.111 | 0.944 | 0.512 | 0.928 | 0.735 | 0.120 | 0.934 |
| layer2.0 | 0.869 | 0.744 | 0.106 | 0.884 | 0.512 | 0.862 | 0.709 | 0.113 | 0.871 |
| layer2.1 | 0.932 | 0.808 | 0.104 | 0.951 | 0.512 | 0.913 | 0.759 | 0.111 | 0.941 |
| layer2.2 | 0.929 | 0.788 | 0.104 | 0.952 | 0.512 | 0.911 | 0.732 | 0.112 | 0.944 |
| layer3.0 | 0.898 | 0.798 | 0.102 | 0.937 | 0.512 | 0.871 | 0.682 | 0.110 | 0.919 |
| layer3.1 | 0.755 | 0.847 | 0.101 | 0.739 | 0.512 | 0.748 | 0.618 | 0.130 | 0.725 |
| layer3.2 | 0.876 | 0.927 | 0.100 | 0.777 | 0.512 | 0.663 | 0.588 | 0.123 | 0.740 |
| penult | 0.876 | 0.927 | 0.100 | 0.777 | 0.512 | 0.663 | 0.588 | 0.123 | 0.740 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | -0.400 | -0.119 | +0.000 | +0.010 | +4.280 | +0.027 | -0.104 | -0.001 | · | · | · | · | · |
| layer1.0 | -0.762 | -0.406 | +0.000 | -0.015 | -45.808 | +0.047 | -0.140 | -0.022 | · | · | · | · | · |
| layer1.1 | +0.193 | +1.056 | +1.000 | +0.011 | +1.554 | -0.069 | +0.075 | +0.017 | · | · | · | · | · |
| layer1.2 | +0.246 | +0.203 | +1.000 | -0.001 | +17.692 | +0.002 | +0.118 | +0.002 | · | · | · | · | · |
| layer2.0 | -0.726 | -0.506 | -3.000 | +0.005 | +5.942 | -0.065 | -0.126 | -0.021 | · | · | · | · | · |
| layer2.1 | +0.652 | -0.411 | +0.000 | +0.038 | -4.840 | +0.001 | +0.013 | +0.030 | · | · | · | · | · |
| layer2.2 | +0.231 | -0.836 | -1.000 | +0.054 | -0.308 | +0.036 | -0.167 | +0.025 | · | · | · | · | · |
| layer3.0 | -2.393 | -1.582 | -14.000 | -0.007 | +1.104 | +0.022 | -0.261 | -0.024 | · | · | · | · | · |
| layer3.1 | -0.540 | +1.566 | +1.000 | -0.074 | +0.149 | -0.013 | -0.026 | -0.017 | · | · | · | · | · |
| layer3.2 | +0.965 | +0.298 | +0.000 | +0.300 | -0.034 | -0.013 | +0.045 | +0.001 | · | · | · | · | · |
| penult | +0.965 | +0.298 | +0.000 | +0.300 | -0.034 | -0.013 | +0.045 | +0.001 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer1.1 | layer2.0 |
| spectral_slope | layer2.0 | layer2.0 |
| spectral_anisotropy | layer1.2 | layer2.5 |
| noise_sigma | layer1.0 | layer1.0 |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | stem | layer1.0 |
| edge_density | layer1.0 | layer2.0 |
| orientation_entropy | layer2.0 | layer2.5 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.5 |
| coarse_animal_vehicle | layer3.0 | layer3.0 |
| corruption_family | layer2.0 | layer2.0 |
| corruption_type | layer2.0 | layer2.0 |
| severity | layer3.0 | layer2.0 |