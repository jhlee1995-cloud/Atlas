# DEFORMATION  A=`results/atlas_v1_resnet20_s0hub_st3`  B=`results/atlas_v1_resnet56_e70`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.015 | 0.990 | 0.995 | True | 0.893 | 0.047 | 0.934 | 0.961 | 0.828 |
| layer1.0 | layer1.0 | 0.029 | 0.991 | 0.997 | True | 0.988 | 0.052 | 0.908 | 0.933 | 0.594 |
| layer1.1 | layer1.5 | 0.021 | 0.980 | 1.000 | True | 0.992 | 0.024 | 0.823 | 0.894 | 0.516 |
| layer1.2 | layer1.8 | 0.014 | 0.985 | 1.000 | True | 0.996 | 0.019 | 0.950 | 0.935 | 0.656 |
| layer2.0 | layer2.0 | 0.017 | 0.969 | 0.908 | True | 0.986 | 0.017 | 0.927 | 0.921 | 0.750 |
| layer2.1 | layer2.5 | 0.019 | 0.971 | 0.506 | True | 0.981 | 0.025 | 0.916 | 0.896 | 0.766 |
| layer2.2 | layer2.8 | 0.020 | 0.981 | 0.556 | True | 0.975 | 0.023 | 0.898 | 0.903 | 0.719 |
| layer3.0 | layer3.0 | 0.024 | 0.956 | 0.922 | True | 0.998 | 0.014 | 0.850 | 0.929 | 0.828 |
| layer3.1 | layer3.5 | 0.070 | 0.886 | 0.680 | True | 0.973 | 0.020 | 0.826 | 0.849 | 0.844 |
| layer3.2 | layer3.8 | 0.009 | 0.881 | 0.831 | True | 0.940 | 0.030 | 0.894 | 0.946 | 1.000 |
| penult | penult | 0.009 | 0.881 | 0.831 | True | 0.940 | 0.030 | 0.894 | 0.946 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.942 | 0.776 | 0.119 | 0.952 | 0.516 | 0.919 | 0.775 | 0.123 | 0.942 |
| layer1.0 | 0.914 | 0.684 | 0.120 | 0.933 | 0.516 | 0.881 | 0.669 | 0.125 | 0.916 |
| layer1.1 | 0.840 | 0.636 | 0.109 | 0.879 | 0.516 | 0.746 | 0.581 | 0.112 | 0.843 |
| layer1.2 | 0.942 | 0.750 | 0.112 | 0.923 | 0.516 | 0.924 | 0.708 | 0.121 | 0.919 |
| layer2.0 | 0.932 | 0.726 | 0.106 | 0.919 | 0.516 | 0.918 | 0.705 | 0.114 | 0.902 |
| layer2.1 | 0.913 | 0.729 | 0.104 | 0.895 | 0.516 | 0.890 | 0.640 | 0.112 | 0.877 |
| layer2.2 | 0.895 | 0.720 | 0.103 | 0.881 | 0.516 | 0.871 | 0.620 | 0.109 | 0.856 |
| layer3.0 | 0.866 | 0.791 | 0.102 | 0.906 | 0.516 | 0.848 | 0.675 | 0.114 | 0.891 |
| layer3.1 | 0.765 | 0.834 | 0.101 | 0.706 | 0.516 | 0.748 | 0.559 | 0.134 | 0.696 |
| layer3.2 | 0.868 | 0.918 | 0.100 | 0.705 | 0.516 | 0.669 | 0.555 | 0.127 | 0.745 |
| penult | 0.868 | 0.918 | 0.100 | 0.705 | 0.516 | 0.669 | 0.555 | 0.127 | 0.745 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | -0.048 | -0.271 | +0.000 | -0.003 | +103.914 | +0.030 | -0.044 | +0.000 | · | · | · | · | · |
| layer1.0 | -1.044 | -1.450 | -1.000 | -0.021 | +174.809 | +0.083 | -0.205 | -0.039 | · | · | · | · | · |
| layer1.1 | -0.057 | +0.248 | +0.000 | +0.061 | +2.862 | -0.046 | +0.011 | +0.019 | · | · | · | · | · |
| layer1.2 | -0.222 | -0.153 | +0.000 | +0.044 | -5.387 | -0.051 | -0.056 | +0.010 | · | · | · | · | · |
| layer2.0 | -0.877 | -1.520 | -3.000 | +0.034 | -0.133 | -0.006 | -0.017 | -0.005 | · | · | · | · | · |
| layer2.1 | +0.152 | -1.361 | +0.000 | +0.040 | -3.298 | -0.054 | +0.029 | +0.015 | · | · | · | · | · |
| layer2.2 | -0.309 | -1.827 | -2.000 | +0.064 | -0.929 | -0.039 | +0.150 | +0.010 | · | · | · | · | · |
| layer3.0 | -2.118 | -3.488 | -12.000 | -0.006 | +0.564 | +0.030 | -0.433 | -0.015 | · | · | · | · | · |
| layer3.1 | -0.303 | +3.942 | -1.000 | +0.184 | -0.238 | -0.091 | -0.094 | +0.057 | · | · | · | · | · |
| layer3.2 | +0.951 | +0.607 | +0.000 | +0.509 | -0.053 | -0.036 | +0.135 | +0.002 | · | · | · | · | · |
| penult | +0.951 | +0.607 | +0.000 | +0.509 | -0.053 | -0.036 | +0.135 | +0.002 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer1.0 | layer1.5 |
| spectral_slope | layer2.0 | layer2.0 |
| spectral_anisotropy | layer2.0 | layer3.0 |
| noise_sigma | layer1.0 | layer1.0 |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | layer1.0 | layer1.0 |
| edge_density | layer1.0 | stem |
| orientation_entropy | layer2.0 | layer2.8 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.5 |
| coarse_animal_vehicle | layer3.0 | layer3.0 |
| corruption_family | layer2.0 | layer1.8 |
| corruption_type | layer1.2 | layer2.0 |
| severity | layer3.0 | layer3.0 |