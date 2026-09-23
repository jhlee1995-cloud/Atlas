# DEFORMATION  A=`results/atlas_v1_resnet20_s0hub_st3`  B=`results/atlas_v1_resnet56_e60`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.027 | 0.979 | 0.990 | True | 0.957 | 0.027 | 0.918 | 0.809 | 0.609 |
| layer1.0 | layer1.0 | 0.028 | 0.981 | 0.997 | True | 0.909 | 0.077 | 0.929 | 0.906 | 0.688 |
| layer1.1 | layer1.5 | 0.031 | 0.956 | 1.000 | True | 0.983 | 0.025 | 0.816 | 0.861 | 0.766 |
| layer1.2 | layer1.8 | 0.014 | 0.982 | 1.000 | True | 0.990 | 0.036 | 0.932 | 0.922 | 0.719 |
| layer2.0 | layer2.0 | 0.011 | 0.990 | 0.954 | True | 0.988 | 0.020 | 0.936 | 0.946 | 0.828 |
| layer2.1 | layer2.5 | 0.013 | 0.985 | 0.910 | True | 0.986 | 0.017 | 0.935 | 0.935 | 0.703 |
| layer2.2 | layer2.8 | 0.017 | 0.985 | 0.467 | True | 0.992 | 0.013 | 0.914 | 0.942 | 0.688 |
| layer3.0 | layer3.0 | 0.023 | 0.967 | 0.955 | True | 0.996 | 0.020 | 0.856 | 0.932 | 0.828 |
| layer3.1 | layer3.5 | 0.062 | 0.875 | 0.920 | True | 0.981 | 0.029 | 0.795 | 0.867 | 0.844 |
| layer3.2 | layer3.8 | 0.005 | 0.883 | 0.676 | True | 0.955 | 0.034 | 0.907 | 0.963 | 1.000 |
| penult | penult | 0.005 | 0.883 | 0.676 | True | 0.955 | 0.034 | 0.907 | 0.963 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.854 | 0.639 | 0.113 | 0.811 | 0.509 | 0.841 | 0.605 | 0.118 | 0.802 |
| layer1.0 | 0.929 | 0.710 | 0.122 | 0.927 | 0.509 | 0.923 | 0.705 | 0.125 | 0.917 |
| layer1.1 | 0.835 | 0.669 | 0.110 | 0.870 | 0.509 | 0.738 | 0.632 | 0.114 | 0.832 |
| layer1.2 | 0.938 | 0.744 | 0.110 | 0.932 | 0.509 | 0.912 | 0.715 | 0.122 | 0.929 |
| layer2.0 | 0.939 | 0.743 | 0.106 | 0.937 | 0.509 | 0.912 | 0.718 | 0.116 | 0.920 |
| layer2.1 | 0.930 | 0.728 | 0.104 | 0.935 | 0.509 | 0.907 | 0.666 | 0.110 | 0.915 |
| layer2.2 | 0.904 | 0.724 | 0.102 | 0.924 | 0.509 | 0.880 | 0.643 | 0.107 | 0.904 |
| layer3.0 | 0.890 | 0.800 | 0.102 | 0.924 | 0.509 | 0.865 | 0.698 | 0.111 | 0.915 |
| layer3.1 | 0.729 | 0.819 | 0.101 | 0.744 | 0.509 | 0.717 | 0.567 | 0.129 | 0.699 |
| layer3.2 | 0.876 | 0.920 | 0.100 | 0.782 | 0.509 | 0.662 | 0.549 | 0.125 | 0.751 |
| penult | 0.876 | 0.920 | 0.100 | 0.782 | 0.509 | 0.662 | 0.549 | 0.125 | 0.751 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | +0.161 | -0.006 | +0.000 | +0.005 | -114.624 | -0.008 | -0.003 | +0.001 | · | · | · | · | · |
| layer1.0 | -0.679 | -0.948 | -1.000 | -0.003 | +4.313 | +0.051 | -0.193 | -0.024 | · | · | · | · | · |
| layer1.1 | -0.207 | -0.352 | +0.000 | +0.038 | +19.594 | -0.009 | -0.074 | +0.005 | · | · | · | · | · |
| layer1.2 | -0.395 | -0.218 | +0.000 | +0.040 | -8.253 | -0.014 | -0.057 | +0.025 | · | · | · | · | · |
| layer2.0 | -1.324 | -1.284 | -3.000 | +0.037 | +3.924 | -0.008 | -0.161 | -0.002 | · | · | · | · | · |
| layer2.1 | -0.643 | -0.903 | -2.000 | +0.037 | -0.145 | -0.027 | -0.183 | +0.003 | · | · | · | · | · |
| layer2.2 | -0.760 | -1.930 | -3.000 | +0.022 | +0.123 | +0.001 | +0.007 | -0.008 | · | · | · | · | · |
| layer3.0 | -3.011 | -4.126 | -16.000 | -0.039 | +1.457 | +0.049 | -0.600 | -0.063 | · | · | · | · | · |
| layer3.1 | +0.076 | +3.387 | +2.000 | -0.115 | +0.078 | -0.030 | +0.089 | -0.013 | · | · | · | · | · |
| layer3.2 | +0.851 | +0.437 | +0.000 | +0.372 | -0.041 | -0.025 | +0.111 | +0.002 | · | · | · | · | · |
| penult | +0.851 | +0.437 | +0.000 | +0.372 | -0.041 | -0.025 | +0.111 | +0.002 | · | · | · | · | · |

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
| orientation_entropy | layer2.0 | layer2.5 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.5 |
| coarse_animal_vehicle | layer3.0 | layer3.5 |
| corruption_family | layer2.0 | layer2.0 |
| corruption_type | layer1.2 | layer2.0 |
| severity | layer3.0 | layer3.0 |