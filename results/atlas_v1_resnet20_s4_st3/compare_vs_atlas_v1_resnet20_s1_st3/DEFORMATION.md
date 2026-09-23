# DEFORMATION  A=`results/atlas_v1_resnet20_s1_st3`  B=`results/atlas_v1_resnet20_s4_st3`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.004 | 0.995 | 0.982 | True | 0.942 | 0.026 | 0.980 | 0.988 | 0.797 |
| layer1.0 | layer1.0 | 0.030 | 0.965 | 0.995 | True | 0.983 | 0.019 | 0.882 | 0.924 | 0.688 |
| layer1.1 | layer1.1 | 0.031 | 0.950 | 0.962 | True | 0.998 | 0.021 | 0.873 | 0.890 | 0.703 |
| layer1.2 | layer1.2 | 0.009 | 0.990 | 0.987 | True | 0.990 | 0.024 | 0.951 | 0.955 | 0.734 |
| layer2.0 | layer2.0 | 0.010 | 0.977 | 0.964 | True | 0.992 | 0.020 | 0.959 | 0.970 | 0.781 |
| layer2.1 | layer2.1 | 0.007 | 0.987 | 0.914 | True | 0.992 | 0.016 | 0.944 | 0.951 | 0.734 |
| layer2.2 | layer2.2 | 0.006 | 0.994 | 0.991 | True | 0.996 | 0.016 | 0.933 | 0.961 | 0.812 |
| layer3.0 | layer3.0 | 0.011 | 0.971 | 0.929 | True | 0.981 | 0.022 | 0.921 | 0.959 | 0.844 |
| layer3.1 | layer3.1 | 0.023 | 0.958 | 0.767 | True | 0.971 | 0.027 | 0.871 | 0.915 | 0.891 |
| layer3.2 | layer3.2 | 0.001 | 0.975 | 0.855 | True | 0.955 | 0.029 | 0.905 | 0.961 | 1.000 |
| penult | penult | 0.001 | 0.975 | 0.855 | True | 0.955 | 0.029 | 0.905 | 0.961 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.981 | 0.831 | 0.118 | 0.983 | 0.526 | 0.969 | 0.826 | 0.118 | 0.980 |
| layer1.0 | 0.878 | 0.701 | 0.120 | 0.933 | 0.526 | 0.832 | 0.678 | 0.121 | 0.916 |
| layer1.1 | 0.877 | 0.668 | 0.115 | 0.896 | 0.526 | 0.848 | 0.657 | 0.119 | 0.872 |
| layer1.2 | 0.955 | 0.775 | 0.111 | 0.951 | 0.526 | 0.944 | 0.762 | 0.121 | 0.946 |
| layer2.0 | 0.954 | 0.793 | 0.107 | 0.964 | 0.526 | 0.947 | 0.762 | 0.115 | 0.963 |
| layer2.1 | 0.943 | 0.783 | 0.105 | 0.945 | 0.526 | 0.929 | 0.720 | 0.113 | 0.937 |
| layer2.2 | 0.935 | 0.781 | 0.104 | 0.950 | 0.526 | 0.924 | 0.731 | 0.114 | 0.939 |
| layer3.0 | 0.914 | 0.808 | 0.103 | 0.937 | 0.526 | 0.906 | 0.715 | 0.113 | 0.929 |
| layer3.1 | 0.864 | 0.862 | 0.102 | 0.858 | 0.526 | 0.776 | 0.616 | 0.144 | 0.851 |
| layer3.2 | 0.886 | 0.934 | 0.100 | 0.826 | 0.526 | 0.695 | 0.571 | 0.125 | 0.776 |
| penult | 0.886 | 0.934 | 0.100 | 0.826 | 0.526 | 0.695 | 0.571 | 0.125 | 0.776 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | +0.114 | +0.098 | +0.000 | +0.008 | -101.887 | +0.015 | +0.023 | +0.006 | · | · | · | · | · |
| layer1.0 | +0.673 | +0.662 | +2.000 | +0.013 | -70.747 | -0.116 | +0.205 | +0.015 | · | · | · | · | · |
| layer1.1 | +0.580 | +0.823 | +1.000 | +0.006 | +12.539 | -0.084 | +0.156 | +0.006 | · | · | · | · | · |
| layer1.2 | +0.743 | +0.303 | +1.000 | -0.011 | -0.928 | +0.004 | +0.208 | +0.003 | · | · | · | · | · |
| layer2.0 | +0.469 | +0.078 | +0.000 | -0.003 | -2.949 | -0.060 | +0.084 | -0.003 | · | · | · | · | · |
| layer2.1 | +0.791 | -0.201 | +1.000 | -0.021 | -2.407 | +0.021 | +0.102 | -0.005 | · | · | · | · | · |
| layer2.2 | +0.538 | +0.018 | +1.000 | -0.004 | -0.352 | +0.024 | -0.006 | +0.006 | · | · | · | · | · |
| layer3.0 | +0.296 | +0.149 | -1.000 | +0.007 | +0.005 | +0.009 | +0.183 | -0.002 | · | · | · | · | · |
| layer3.1 | +0.057 | -1.807 | +0.000 | -0.040 | +0.018 | +0.007 | -0.038 | +0.009 | · | · | · | · | · |
| layer3.2 | +0.145 | -0.066 | +0.000 | -0.042 | +0.004 | -0.000 | -0.029 | +0.006 | · | · | · | · | · |
| penult | +0.145 | -0.066 | +0.000 | -0.042 | +0.004 | -0.000 | -0.029 | +0.006 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer1.1 | layer2.0 |
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
| severity | layer3.0 | layer2.0 |