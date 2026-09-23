# DEFORMATION  A=`results/atlas_v1_resnet20_s3_st3`  B=`results/atlas_v1_resnet56_s2`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.003 | 0.994 | 0.995 | True | 0.957 | 0.028 | 0.983 | 0.988 | 0.844 |
| layer1.0 | layer1.0 | 0.019 | 0.982 | 0.995 | True | 0.960 | 0.038 | 0.885 | 0.949 | 0.688 |
| layer1.1 | layer1.5 | 0.022 | 0.982 | 0.937 | True | 0.992 | 0.035 | 0.885 | 0.936 | 0.672 |
| layer1.2 | layer1.8 | 0.015 | 0.977 | 0.967 | True | 0.988 | 0.032 | 0.956 | 0.944 | 0.750 |
| layer2.0 | layer2.0 | 0.010 | 0.975 | 0.979 | True | 0.994 | 0.022 | 0.969 | 0.976 | 0.797 |
| layer2.1 | layer2.5 | 0.012 | 0.979 | 0.566 | True | 0.990 | 0.020 | 0.953 | 0.959 | 0.781 |
| layer2.2 | layer2.8 | 0.014 | 0.977 | 0.536 | True | 0.975 | 0.022 | 0.925 | 0.950 | 0.734 |
| layer3.0 | layer3.0 | 0.010 | 0.988 | 0.962 | True | 0.988 | 0.023 | 0.895 | 0.948 | 0.797 |
| layer3.1 | layer3.5 | 0.035 | 0.914 | 0.767 | True | 0.981 | 0.019 | 0.819 | 0.883 | 0.828 |
| layer3.2 | layer3.8 | 0.012 | 0.918 | 0.785 | False | 0.961 | 0.026 | 0.895 | 0.954 | 1.000 |
| penult | penult | 0.012 | 0.918 | 0.785 | False | 0.961 | 0.026 | 0.895 | 0.954 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.983 | 0.821 | 0.117 | 0.987 | 0.521 | 0.972 | 0.831 | 0.121 | 0.981 |
| layer1.0 | 0.896 | 0.671 | 0.122 | 0.938 | 0.521 | 0.865 | 0.658 | 0.127 | 0.925 |
| layer1.1 | 0.879 | 0.689 | 0.116 | 0.920 | 0.521 | 0.819 | 0.635 | 0.116 | 0.881 |
| layer1.2 | 0.953 | 0.752 | 0.111 | 0.942 | 0.521 | 0.933 | 0.722 | 0.121 | 0.926 |
| layer2.0 | 0.967 | 0.832 | 0.106 | 0.966 | 0.521 | 0.960 | 0.792 | 0.115 | 0.959 |
| layer2.1 | 0.945 | 0.786 | 0.105 | 0.949 | 0.521 | 0.939 | 0.752 | 0.110 | 0.938 |
| layer2.2 | 0.918 | 0.779 | 0.104 | 0.941 | 0.521 | 0.912 | 0.723 | 0.110 | 0.933 |
| layer3.0 | 0.904 | 0.800 | 0.102 | 0.944 | 0.521 | 0.871 | 0.696 | 0.109 | 0.927 |
| layer3.1 | 0.788 | 0.848 | 0.101 | 0.800 | 0.521 | 0.742 | 0.599 | 0.124 | 0.755 |
| layer3.2 | 0.864 | 0.935 | 0.100 | 0.772 | 0.521 | 0.650 | 0.589 | 0.124 | 0.732 |
| penult | 0.864 | 0.935 | 0.100 | 0.772 | 0.521 | 0.650 | 0.589 | 0.124 | 0.732 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | +0.101 | +0.096 | +0.000 | +0.003 | -104.836 | -0.003 | -0.059 | +0.004 | · | · | · | · | · |
| layer1.0 | -0.673 | -0.726 | -2.000 | -0.020 | +10.039 | +0.040 | -0.254 | -0.024 | · | · | · | · | · |
| layer1.1 | +0.405 | +0.727 | +1.000 | +0.070 | +16.914 | -0.059 | -0.038 | +0.038 | · | · | · | · | · |
| layer1.2 | +0.672 | +0.195 | +1.000 | +0.044 | +17.121 | -0.042 | +0.135 | +0.034 | · | · | · | · | · |
| layer2.0 | -0.176 | +0.133 | -1.000 | +0.021 | -7.758 | -0.015 | +0.016 | +0.014 | · | · | · | · | · |
| layer2.1 | +0.261 | +0.617 | +0.000 | +0.027 | -6.945 | -0.018 | +0.049 | +0.052 | · | · | · | · | · |
| layer2.2 | -0.044 | +0.238 | +0.000 | +0.032 | -3.414 | -0.017 | -0.003 | +0.043 | · | · | · | · | · |
| layer3.0 | -2.446 | -2.205 | -13.000 | -0.021 | +0.594 | +0.009 | -0.395 | -0.028 | · | · | · | · | · |
| layer3.1 | -0.319 | +2.120 | +0.000 | +0.114 | -0.313 | -0.090 | -0.009 | +0.041 | · | · | · | · | · |
| layer3.2 | +0.623 | +0.566 | +0.000 | +2.243 | -0.122 | -0.047 | +0.082 | +0.016 | · | · | · | · | · |
| penult | +0.623 | +0.566 | +0.000 | +2.243 | -0.122 | -0.047 | +0.082 | +0.016 | · | · | · | · | · |

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
| hue_cos | stem | layer1.0 |
| hue_sin | stem | stem |
| colorfulness | layer1.0 | layer1.0 |
| edge_density | layer1.0 | stem |
| orientation_entropy | layer3.0 | layer2.0 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.5 |
| coarse_animal_vehicle | layer3.0 | layer3.0 |
| corruption_family | layer2.0 | layer2.0 |
| corruption_type | layer2.0 | layer1.8 |
| severity | layer3.0 | layer2.0 |