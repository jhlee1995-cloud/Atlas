# DEFORMATION  A=`results/atlas_v1_resnet56_s0hub_st3`  B=`results/atlas_v1_resnet56_s2`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.006 | 0.991 | 0.997 | True | 0.979 | 0.024 | 0.984 | 0.990 | 0.891 |
| layer1.0 | layer1.0 | 0.012 | 0.992 | 0.995 | True | 0.983 | 0.034 | 0.927 | 0.969 | 0.641 |
| layer1.5 | layer1.5 | 0.017 | 0.991 | 0.995 | True | 0.988 | 0.031 | 0.926 | 0.949 | 0.734 |
| layer1.8 | layer1.8 | 0.010 | 0.983 | 0.960 | True | 0.996 | 0.023 | 0.957 | 0.954 | 0.719 |
| layer2.0 | layer2.0 | 0.004 | 0.987 | 0.904 | True | 0.994 | 0.024 | 0.968 | 0.976 | 0.844 |
| layer2.5 | layer2.5 | 0.009 | 0.982 | 0.566 | True | 0.990 | 0.022 | 0.945 | 0.962 | 0.781 |
| layer2.8 | layer2.8 | 0.007 | 0.986 | 0.972 | True | 0.973 | 0.023 | 0.931 | 0.955 | 0.703 |
| layer3.0 | layer3.0 | 0.007 | 0.984 | 0.980 | True | 0.965 | 0.017 | 0.950 | 0.972 | 0.891 |
| layer3.5 | layer3.5 | 0.017 | 0.930 | 0.831 | True | 0.971 | 0.023 | 0.868 | 0.928 | 0.859 |
| layer3.8 | layer3.8 | 0.004 | 0.905 | 0.747 | False | 0.969 | 0.021 | 0.961 | 0.977 | 1.000 |
| penult | penult | 0.004 | 0.905 | 0.747 | False | 0.969 | 0.021 | 0.961 | 0.977 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.980 | 0.836 | 0.115 | 0.988 | 0.569 | 0.979 | 0.853 | 0.119 | 0.988 |
| layer1.0 | 0.936 | 0.712 | 0.116 | 0.962 | 0.569 | 0.909 | 0.700 | 0.121 | 0.951 |
| layer1.5 | 0.921 | 0.724 | 0.113 | 0.934 | 0.569 | 0.892 | 0.680 | 0.117 | 0.915 |
| layer1.8 | 0.958 | 0.788 | 0.110 | 0.953 | 0.569 | 0.947 | 0.767 | 0.121 | 0.944 |
| layer2.0 | 0.967 | 0.831 | 0.105 | 0.973 | 0.569 | 0.959 | 0.798 | 0.112 | 0.964 |
| layer2.5 | 0.939 | 0.804 | 0.104 | 0.948 | 0.569 | 0.935 | 0.776 | 0.109 | 0.945 |
| layer2.8 | 0.925 | 0.788 | 0.104 | 0.943 | 0.569 | 0.923 | 0.749 | 0.109 | 0.938 |
| layer3.0 | 0.946 | 0.822 | 0.102 | 0.962 | 0.569 | 0.943 | 0.759 | 0.109 | 0.955 |
| layer3.5 | 0.848 | 0.872 | 0.101 | 0.861 | 0.569 | 0.815 | 0.653 | 0.133 | 0.847 |
| layer3.8 | 0.926 | 0.947 | 0.100 | 0.719 | 0.569 | 0.681 | 0.632 | 0.123 | 0.737 |
| penult | 0.926 | 0.947 | 0.100 | 0.719 | 0.569 | 0.681 | 0.632 | 0.123 | 0.737 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | +0.328 | -0.202 | +0.000 | -0.001 | -280.432 | -0.005 | +0.018 | +0.002 | · | · | · | · | · |
| layer1.0 | +0.371 | +0.392 | +0.000 | +0.015 | -38.824 | -0.043 | +0.007 | +0.018 | · | · | · | · | · |
| layer1.5 | +0.546 | +0.552 | +1.000 | +0.062 | +33.061 | -0.036 | +0.014 | +0.038 | · | · | · | · | · |
| layer1.8 | +0.555 | -0.010 | +1.000 | +0.019 | +9.514 | -0.025 | +0.047 | +0.014 | · | · | · | · | · |
| layer2.0 | +0.304 | -0.257 | +0.000 | +0.018 | -7.295 | -0.000 | +0.081 | +0.016 | · | · | · | · | · |
| layer2.5 | -0.131 | -1.083 | -1.000 | +0.001 | -1.877 | +0.039 | -0.027 | +0.008 | · | · | · | · | · |
| layer2.8 | -0.147 | -1.076 | -1.000 | +0.006 | -0.818 | +0.021 | -0.062 | +0.003 | · | · | · | · | · |
| layer3.0 | -0.037 | -0.705 | -1.000 | -0.017 | -0.167 | +0.039 | +0.072 | -0.010 | · | · | · | · | · |
| layer3.5 | +0.077 | -1.098 | -2.000 | +0.040 | -0.060 | -0.023 | -0.134 | +0.000 | · | · | · | · | · |
| layer3.8 | -0.210 | -0.047 | +0.000 | -0.120 | +0.001 | +0.008 | -0.085 | -0.005 | · | · | · | · | · |
| penult | -0.210 | -0.047 | +0.000 | -0.120 | +0.001 | +0.008 | -0.085 | -0.005 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer2.0 | layer2.0 |
| spectral_slope | layer2.0 | layer2.0 |
| spectral_anisotropy | layer3.0 | layer2.0 |
| noise_sigma | layer1.0 | layer1.0 |
| saturation_mean | stem | stem |
| hue_cos | stem | layer1.0 |
| hue_sin | stem | stem |
| colorfulness | layer1.0 | layer1.0 |
| edge_density | layer1.0 | stem |
| orientation_entropy | layer2.0 | layer2.0 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.5 | layer3.5 |
| coarse_animal_vehicle | layer3.5 | layer3.0 |
| corruption_family | layer2.0 | layer2.0 |
| corruption_type | layer2.0 | layer1.8 |
| severity | layer3.0 | layer2.0 |