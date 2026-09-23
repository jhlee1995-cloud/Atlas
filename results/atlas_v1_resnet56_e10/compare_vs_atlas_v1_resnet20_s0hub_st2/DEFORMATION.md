# DEFORMATION  A=`results/atlas_v1_resnet20_s0hub_st2`  B=`results/atlas_v1_resnet56_e10`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.044 | 0.966 | 0.984 | True | 0.963 | 0.019 | 0.907 | 0.912 | 0.625 |
| layer1.0 | layer1.0 | 0.027 | 0.991 | 0.992 | True | 0.831 | 0.118 | 0.888 | 0.936 | 0.625 |
| layer1.1 | layer1.5 | 0.019 | 0.983 | 1.000 | True | 0.979 | 0.033 | 0.752 | 0.902 | 0.578 |
| layer1.2 | layer1.8 | 0.033 | 0.958 | 0.953 | True | 0.955 | 0.053 | 0.831 | 0.879 | 0.656 |
| layer2.0 | layer2.0 | 0.022 | 0.969 | 0.971 | True | 0.963 | 0.062 | 0.863 | 0.936 | 0.812 |
| layer2.1 | layer2.5 | 0.045 | 0.920 | 0.483 | True | 0.905 | 0.091 | 0.781 | 0.874 | 0.656 |
| layer2.2 | layer2.8 | 0.049 | 0.863 | 0.263 | True | 0.917 | 0.091 | 0.754 | 0.865 | 0.641 |
| layer3.0 | layer3.0 | 0.037 | 0.869 | 0.775 | True | 0.897 | 0.075 | 0.787 | 0.904 | 0.766 |
| layer3.1 | layer3.5 | 0.032 | 0.944 | 0.977 | True | 0.868 | 0.077 | 0.812 | 0.876 | 0.766 |
| layer3.2 | layer3.8 | 0.084 | 0.946 | 0.932 | True | 0.926 | 0.061 | 0.741 | 0.851 | 0.906 |
| penult | penult | 0.084 | 0.946 | 0.932 | True | 0.926 | 0.061 | 0.741 | 0.851 | 0.906 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.874 | 0.671 | 0.114 | 0.908 | 0.379 | 0.870 | 0.686 | 0.120 | 0.895 |
| layer1.0 | 0.893 | 0.651 | 0.116 | 0.944 | 0.379 | 0.837 | 0.650 | 0.123 | 0.934 |
| layer1.1 | 0.807 | 0.545 | 0.108 | 0.890 | 0.379 | 0.674 | 0.506 | 0.111 | 0.848 |
| layer1.2 | 0.850 | 0.644 | 0.109 | 0.881 | 0.379 | 0.787 | 0.605 | 0.117 | 0.862 |
| layer2.0 | 0.886 | 0.661 | 0.106 | 0.897 | 0.379 | 0.831 | 0.609 | 0.114 | 0.875 |
| layer2.1 | 0.795 | 0.642 | 0.103 | 0.850 | 0.379 | 0.744 | 0.521 | 0.109 | 0.812 |
| layer2.2 | 0.770 | 0.658 | 0.102 | 0.840 | 0.379 | 0.718 | 0.533 | 0.108 | 0.802 |
| layer3.0 | 0.816 | 0.750 | 0.101 | 0.879 | 0.379 | 0.768 | 0.606 | 0.115 | 0.865 |
| layer3.1 | 0.799 | 0.808 | 0.101 | 0.808 | 0.379 | 0.663 | 0.539 | 0.129 | 0.738 |
| layer3.2 | 0.766 | 0.853 | 0.100 | 0.807 | 0.379 | 0.587 | 0.501 | 0.122 | 0.714 |
| penult | 0.766 | 0.853 | 0.100 | 0.807 | 0.379 | 0.587 | 0.501 | 0.122 | 0.714 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | -0.316 | -0.496 | +0.000 | -0.001 | +196.809 | +0.073 | +0.039 | +0.012 | · | · | · | · | · |
| layer1.0 | -1.460 | -1.023 | -1.000 | -0.005 | +324.506 | +0.059 | -0.220 | -0.021 | · | · | · | · | · |
| layer1.1 | -0.536 | -1.104 | +0.000 | +0.048 | +48.437 | -0.024 | -0.089 | +0.022 | · | · | · | · | · |
| layer1.2 | -0.991 | -0.576 | -1.000 | +0.090 | +17.766 | -0.021 | -0.172 | +0.046 | · | · | · | · | · |
| layer2.0 | -1.202 | -2.125 | -4.000 | +0.081 | +1.432 | -0.058 | -0.127 | +0.016 | · | · | · | · | · |
| layer2.1 | -0.806 | -1.868 | -3.000 | +0.148 | -2.201 | -0.054 | -0.150 | +0.050 | · | · | · | · | · |
| layer2.2 | -1.721 | -2.918 | -6.000 | +0.167 | -0.194 | -0.067 | -0.187 | +0.044 | · | · | · | · | · |
| layer3.0 | -3.965 | -6.080 | -22.000 | +0.122 | +1.071 | +0.077 | -0.685 | -0.003 | · | · | · | · | · |
| layer3.1 | -3.873 | -2.531 | -20.000 | +0.081 | +0.492 | +0.039 | -0.429 | -0.019 | · | · | · | · | · |
| layer3.2 | +2.069 | -2.039 | +1.000 | -1.543 | +0.516 | +0.172 | +0.114 | -0.091 | · | · | · | · | · |
| penult | +2.069 | -2.039 | +1.000 | -1.543 | +0.516 | +0.172 | +0.114 | -0.091 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer1.0 | layer1.5 |
| spectral_slope | layer2.0 | layer1.5 |
| spectral_anisotropy | layer2.0 | layer1.5 |
| noise_sigma | layer1.0 | layer1.5 |
| saturation_mean | stem | layer1.0 |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | layer1.0 | stem |
| edge_density | layer1.0 | layer1.0 |
| orientation_entropy | layer2.0 | layer2.0 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.0 |
| coarse_animal_vehicle | layer3.0 | layer2.5 |
| corruption_family | layer2.0 | layer2.0 |
| corruption_type | layer1.2 | layer1.8 |
| severity | layer3.0 | layer2.0 |