# DEFORMATION  A=`results/atlas_v1_resnet20_s0hub_st3`  B=`results/atlas_v1_resnet56_s2`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.011 | 0.981 | 0.982 | True | 0.940 | 0.035 | 0.945 | 0.979 | 0.844 |
| layer1.0 | layer1.0 | 0.021 | 0.969 | 0.995 | True | 0.959 | 0.054 | 0.901 | 0.943 | 0.719 |
| layer1.1 | layer1.5 | 0.028 | 0.980 | 0.995 | True | 0.992 | 0.023 | 0.808 | 0.865 | 0.656 |
| layer1.2 | layer1.8 | 0.018 | 0.972 | 0.969 | True | 0.996 | 0.015 | 0.943 | 0.922 | 0.672 |
| layer2.0 | layer2.0 | 0.015 | 0.961 | 0.920 | True | 0.994 | 0.018 | 0.946 | 0.921 | 0.875 |
| layer2.1 | layer2.5 | 0.013 | 0.981 | 0.501 | True | 0.988 | 0.020 | 0.937 | 0.919 | 0.781 |
| layer2.2 | layer2.8 | 0.011 | 0.990 | 0.529 | True | 0.971 | 0.023 | 0.928 | 0.935 | 0.719 |
| layer3.0 | layer3.0 | 0.025 | 0.948 | 0.978 | True | 0.988 | 0.022 | 0.872 | 0.930 | 0.812 |
| layer3.1 | layer3.5 | 0.072 | 0.809 | 0.822 | True | 0.979 | 0.021 | 0.785 | 0.828 | 0.828 |
| layer3.2 | layer3.8 | 0.010 | 0.895 | 0.785 | False | 0.940 | 0.023 | 0.908 | 0.955 | 1.000 |
| penult | penult | 0.010 | 0.895 | 0.785 | False | 0.940 | 0.023 | 0.908 | 0.955 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.962 | 0.806 | 0.116 | 0.971 | 0.569 | 0.952 | 0.812 | 0.120 | 0.964 |
| layer1.0 | 0.915 | 0.702 | 0.120 | 0.943 | 0.569 | 0.890 | 0.714 | 0.127 | 0.934 |
| layer1.1 | 0.829 | 0.617 | 0.112 | 0.860 | 0.569 | 0.736 | 0.572 | 0.114 | 0.819 |
| layer1.2 | 0.944 | 0.746 | 0.112 | 0.922 | 0.569 | 0.923 | 0.707 | 0.123 | 0.918 |
| layer2.0 | 0.949 | 0.742 | 0.106 | 0.915 | 0.569 | 0.936 | 0.701 | 0.115 | 0.898 |
| layer2.1 | 0.931 | 0.730 | 0.103 | 0.913 | 0.569 | 0.917 | 0.661 | 0.108 | 0.894 |
| layer2.2 | 0.917 | 0.736 | 0.102 | 0.925 | 0.569 | 0.898 | 0.651 | 0.107 | 0.907 |
| layer3.0 | 0.883 | 0.796 | 0.102 | 0.907 | 0.569 | 0.871 | 0.679 | 0.110 | 0.899 |
| layer3.1 | 0.726 | 0.845 | 0.101 | 0.675 | 0.569 | 0.722 | 0.587 | 0.134 | 0.664 |
| layer3.2 | 0.870 | 0.922 | 0.100 | 0.772 | 0.569 | 0.641 | 0.581 | 0.124 | 0.722 |
| penult | 0.870 | 0.922 | 0.100 | 0.772 | 0.569 | 0.641 | 0.581 | 0.124 | 0.722 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | +0.049 | -0.239 | +0.000 | -0.002 | -45.969 | +0.017 | -0.047 | +0.007 | · | · | · | · | · |
| layer1.0 | -0.673 | -0.465 | -1.000 | -0.003 | -5.212 | +0.020 | -0.242 | -0.023 | · | · | · | · | · |
| layer1.1 | +0.055 | +0.499 | +1.000 | +0.087 | +28.696 | -0.021 | -0.021 | +0.031 | · | · | · | · | · |
| layer1.2 | +0.465 | -0.037 | +1.000 | +0.049 | +12.859 | -0.041 | +0.032 | +0.025 | · | · | · | · | · |
| layer2.0 | -0.565 | -0.832 | -2.000 | +0.024 | -5.075 | -0.002 | +0.030 | -0.006 | · | · | · | · | · |
| layer2.1 | -0.007 | -0.179 | +0.000 | +0.008 | -3.377 | -0.011 | +0.019 | +0.017 | · | · | · | · | · |
| layer2.2 | -0.073 | -0.209 | -1.000 | +0.014 | -1.658 | -0.032 | +0.175 | +0.022 | · | · | · | · | · |
| layer3.0 | -2.567 | -2.808 | -13.000 | -0.046 | +0.919 | +0.057 | -0.353 | -0.037 | · | · | · | · | · |
| layer3.1 | +0.117 | +4.262 | +1.000 | +0.035 | -0.149 | -0.088 | -0.162 | +0.035 | · | · | · | · | · |
| layer3.2 | +0.544 | +0.509 | +0.000 | +2.216 | -0.120 | -0.045 | +0.092 | +0.017 | · | · | · | · | · |
| penult | +0.544 | +0.509 | +0.000 | +2.216 | -0.120 | -0.045 | +0.092 | +0.017 | · | · | · | · | · |

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
| orientation_entropy | layer2.0 | layer2.0 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.5 |
| coarse_animal_vehicle | layer3.0 | layer3.0 |
| corruption_family | layer2.0 | layer2.0 |
| corruption_type | layer1.2 | layer1.8 |
| severity | layer3.0 | layer2.0 |