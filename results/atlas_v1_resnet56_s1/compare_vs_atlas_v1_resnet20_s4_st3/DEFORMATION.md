# DEFORMATION  A=`results/atlas_v1_resnet20_s4_st3`  B=`results/atlas_v1_resnet56_s1`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.003 | 0.998 | 1.000 | True | 0.969 | 0.039 | 0.974 | 0.977 | 0.828 |
| layer1.0 | layer1.0 | 0.067 | 0.955 | 0.984 | True | 0.903 | 0.074 | 0.795 | 0.861 | 0.531 |
| layer1.1 | layer1.5 | 0.040 | 0.936 | 0.942 | True | 0.994 | 0.026 | 0.873 | 0.909 | 0.609 |
| layer1.2 | layer1.8 | 0.008 | 0.980 | 0.991 | True | 0.992 | 0.026 | 0.961 | 0.949 | 0.734 |
| layer2.0 | layer2.0 | 0.008 | 0.978 | 0.956 | True | 0.994 | 0.024 | 0.949 | 0.965 | 0.859 |
| layer2.1 | layer2.5 | 0.011 | 0.988 | 0.938 | True | 0.992 | 0.026 | 0.944 | 0.958 | 0.812 |
| layer2.2 | layer2.8 | 0.011 | 0.980 | 0.510 | True | 0.988 | 0.026 | 0.935 | 0.967 | 0.812 |
| layer3.0 | layer3.0 | 0.016 | 0.971 | 0.934 | True | 0.926 | 0.055 | 0.889 | 0.941 | 0.766 |
| layer3.1 | layer3.5 | 0.085 | 0.803 | 0.607 | True | 0.967 | 0.052 | 0.788 | 0.850 | 0.828 |
| layer3.2 | layer3.8 | 0.009 | 0.925 | 0.838 | True | 0.969 | 0.023 | 0.896 | 0.952 | 1.000 |
| penult | penult | 0.009 | 0.925 | 0.838 | True | 0.969 | 0.023 | 0.896 | 0.952 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.973 | 0.776 | 0.118 | 0.961 | 0.489 | 0.973 | 0.783 | 0.121 | 0.950 |
| layer1.0 | 0.841 | 0.611 | 0.117 | 0.880 | 0.489 | 0.792 | 0.610 | 0.118 | 0.853 |
| layer1.1 | 0.896 | 0.681 | 0.115 | 0.904 | 0.489 | 0.862 | 0.629 | 0.118 | 0.882 |
| layer1.2 | 0.962 | 0.787 | 0.109 | 0.942 | 0.489 | 0.951 | 0.775 | 0.120 | 0.937 |
| layer2.0 | 0.948 | 0.807 | 0.106 | 0.964 | 0.489 | 0.931 | 0.757 | 0.113 | 0.958 |
| layer2.1 | 0.942 | 0.793 | 0.104 | 0.953 | 0.489 | 0.926 | 0.744 | 0.112 | 0.946 |
| layer2.2 | 0.939 | 0.800 | 0.103 | 0.958 | 0.489 | 0.926 | 0.727 | 0.111 | 0.950 |
| layer3.0 | 0.892 | 0.786 | 0.102 | 0.936 | 0.489 | 0.866 | 0.683 | 0.111 | 0.920 |
| layer3.1 | 0.715 | 0.846 | 0.101 | 0.719 | 0.489 | 0.704 | 0.587 | 0.139 | 0.674 |
| layer3.2 | 0.872 | 0.942 | 0.100 | 0.778 | 0.489 | 0.659 | 0.593 | 0.124 | 0.742 |
| penult | 0.872 | 0.942 | 0.100 | 0.778 | 0.489 | 0.659 | 0.593 | 0.124 | 0.742 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | -0.629 | -0.448 | -1.000 | -0.008 | +340.064 | +0.014 | -0.105 | -0.015 | · | · | · | · | · |
| layer1.0 | -1.827 | -1.327 | -3.000 | -0.032 | +170.657 | +0.140 | -0.469 | -0.040 | · | · | · | · | · |
| layer1.1 | -0.370 | -0.140 | -1.000 | +0.020 | -18.357 | +0.051 | -0.101 | +0.005 | · | · | · | · | · |
| layer1.2 | -0.325 | +0.165 | -1.000 | +0.017 | +4.596 | +0.027 | -0.121 | +0.011 | · | · | · | · | · |
| layer2.0 | -1.140 | -0.291 | -2.000 | -0.023 | +10.103 | +0.039 | -0.173 | -0.019 | · | · | · | · | · |
| layer2.1 | -0.427 | +0.103 | +0.000 | +0.003 | -1.685 | -0.002 | -0.147 | -0.002 | · | · | · | · | · |
| layer2.2 | -0.154 | +0.167 | +0.000 | +0.002 | -1.560 | +0.013 | +0.220 | +0.008 | · | · | · | · | · |
| layer3.0 | -2.827 | -0.943 | -12.000 | -0.049 | +1.094 | +0.019 | -0.547 | -0.044 | · | · | · | · | · |
| layer3.1 | +0.129 | +6.208 | +3.000 | +0.028 | -0.116 | -0.080 | +0.183 | +0.002 | · | · | · | · | · |
| layer3.2 | +0.481 | +0.413 | +0.000 | +2.090 | -0.115 | -0.040 | +0.088 | +0.017 | · | · | · | · | · |
| penult | +0.481 | +0.413 | +0.000 | +2.090 | -0.115 | -0.040 | +0.088 | +0.017 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | layer1.5 |
| highfreq_ratio | layer2.0 | layer2.0 |
| spectral_slope | layer2.0 | layer2.0 |
| spectral_anisotropy | layer2.0 | layer2.0 |
| noise_sigma | layer1.0 | layer1.5 |
| saturation_mean | stem | layer1.0 |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | stem | layer1.0 |
| edge_density | layer1.0 | layer1.5 |
| orientation_entropy | layer2.0 | layer2.5 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.5 |
| coarse_animal_vehicle | layer3.0 | layer3.5 |
| corruption_family | layer2.0 | layer2.0 |
| corruption_type | layer2.0 | layer2.0 |
| severity | layer2.0 | layer2.0 |