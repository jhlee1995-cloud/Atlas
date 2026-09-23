# DEFORMATION  A=`results/atlas_v1_resnet20_s1_st3`  B=`results/atlas_v1_resnet56_s13m`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.006 | 0.994 | 0.987 | True | 0.967 | 0.052 | 0.920 | 0.962 | 0.766 |
| layer1.0 | layer1.0 | 0.028 | 0.964 | 0.997 | True | 0.905 | 0.061 | 0.752 | 0.934 | 0.594 |
| layer1.1 | layer1.5 | 0.019 | 0.976 | 0.946 | True | 0.981 | 0.032 | 0.865 | 0.888 | 0.594 |
| layer1.2 | layer1.8 | 0.019 | 0.962 | 0.981 | True | 0.986 | 0.023 | 0.923 | 0.917 | 0.641 |
| layer2.0 | layer2.0 | 0.017 | 0.970 | 0.889 | True | 0.992 | 0.021 | 0.931 | 0.949 | 0.766 |
| layer2.1 | layer2.5 | 0.013 | 0.979 | 0.571 | True | 0.988 | 0.031 | 0.932 | 0.945 | 0.750 |
| layer2.2 | layer2.8 | 0.015 | 0.980 | 0.539 | True | 0.988 | 0.026 | 0.925 | 0.928 | 0.750 |
| layer3.0 | layer3.0 | 0.015 | 0.961 | 0.922 | True | 0.969 | 0.036 | 0.883 | 0.941 | 0.781 |
| layer3.1 | layer3.5 | 0.057 | 0.845 | 0.731 | True | 0.959 | 0.052 | 0.791 | 0.865 | 0.797 |
| layer3.2 | layer3.8 | 0.003 | 0.953 | 0.739 | True | 0.950 | 0.043 | 0.890 | 0.953 | 1.000 |
| penult | penult | 0.003 | 0.953 | 0.739 | True | 0.950 | 0.043 | 0.890 | 0.953 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.938 | 0.719 | 0.115 | 0.950 | 0.518 | 0.898 | 0.710 | 0.116 | 0.941 |
| layer1.0 | 0.869 | 0.661 | 0.114 | 0.937 | 0.518 | 0.760 | 0.675 | 0.118 | 0.916 |
| layer1.1 | 0.852 | 0.630 | 0.115 | 0.887 | 0.518 | 0.815 | 0.595 | 0.118 | 0.856 |
| layer1.2 | 0.930 | 0.743 | 0.111 | 0.910 | 0.518 | 0.919 | 0.731 | 0.121 | 0.906 |
| layer2.0 | 0.939 | 0.791 | 0.106 | 0.953 | 0.518 | 0.921 | 0.780 | 0.114 | 0.948 |
| layer2.1 | 0.934 | 0.798 | 0.105 | 0.952 | 0.518 | 0.917 | 0.743 | 0.110 | 0.946 |
| layer2.2 | 0.931 | 0.799 | 0.105 | 0.948 | 0.518 | 0.915 | 0.741 | 0.112 | 0.939 |
| layer3.0 | 0.887 | 0.782 | 0.102 | 0.931 | 0.518 | 0.858 | 0.681 | 0.110 | 0.912 |
| layer3.1 | 0.737 | 0.825 | 0.101 | 0.764 | 0.518 | 0.738 | 0.603 | 0.133 | 0.742 |
| layer3.2 | 0.880 | 0.926 | 0.100 | 0.777 | 0.518 | 0.679 | 0.569 | 0.125 | 0.750 |
| penult | 0.880 | 0.926 | 0.100 | 0.777 | 0.518 | 0.679 | 0.569 | 0.125 | 0.750 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | -1.009 | -0.040 | -1.000 | +0.014 | +15.760 | +0.004 | -0.174 | -0.008 | · | · | · | · | · |
| layer1.0 | -2.073 | -0.144 | -1.000 | -0.016 | +44.943 | -0.011 | -0.416 | -0.035 | · | · | · | · | · |
| layer1.1 | -0.485 | +0.055 | +0.000 | +0.027 | +57.536 | +0.028 | -0.041 | +0.002 | · | · | · | · | · |
| layer1.2 | +0.323 | +0.360 | +1.000 | +0.003 | +10.517 | +0.041 | +0.006 | +0.020 | · | · | · | · | · |
| layer2.0 | -0.918 | -0.316 | -2.000 | -0.003 | +5.222 | -0.067 | -0.160 | -0.005 | · | · | · | · | · |
| layer2.1 | +0.078 | -0.651 | +0.000 | +0.033 | -7.189 | -0.017 | +0.058 | +0.032 | · | · | · | · | · |
| layer2.2 | -0.463 | -1.445 | -2.000 | +0.018 | -1.191 | +0.004 | -0.121 | -0.001 | · | · | · | · | · |
| layer3.0 | -3.329 | -1.433 | -18.000 | -0.038 | +1.531 | +0.023 | -0.429 | -0.049 | · | · | · | · | · |
| layer3.1 | +0.182 | +3.492 | +2.000 | -0.116 | +0.237 | -0.045 | +0.130 | -0.051 | · | · | · | · | · |
| layer3.2 | +0.970 | +0.418 | +0.000 | +0.247 | -0.033 | -0.018 | +0.075 | +0.002 | · | · | · | · | · |
| penult | +0.970 | +0.418 | +0.000 | +0.247 | -0.033 | -0.018 | +0.075 | +0.002 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | layer1.0 |
| highfreq_ratio | layer1.1 | layer2.0 |
| spectral_slope | layer2.0 | layer2.0 |
| spectral_anisotropy | layer1.2 | layer3.0 |
| noise_sigma | layer1.0 | layer1.5 |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | stem | layer1.0 |
| edge_density | layer1.0 | layer1.0 |
| orientation_entropy | layer2.0 | layer3.0 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.5 |
| coarse_animal_vehicle | layer3.0 | layer3.5 |
| corruption_family | layer2.0 | layer1.8 |
| corruption_type | layer2.0 | layer1.8 |
| severity | layer3.0 | layer3.0 |