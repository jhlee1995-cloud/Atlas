# DEFORMATION  A=`results/atlas_v1_resnet20_s4_st3`  B=`results/atlas_v1_resnet56_e70`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.005 | 0.997 | 0.997 | True | 0.903 | 0.043 | 0.970 | 0.982 | 0.859 |
| layer1.0 | layer1.0 | 0.049 | 0.982 | 0.997 | True | 0.983 | 0.044 | 0.824 | 0.863 | 0.500 |
| layer1.1 | layer1.5 | 0.034 | 0.956 | 0.942 | True | 0.996 | 0.035 | 0.836 | 0.881 | 0.578 |
| layer1.2 | layer1.8 | 0.014 | 0.983 | 0.991 | True | 0.992 | 0.030 | 0.944 | 0.944 | 0.719 |
| layer2.0 | layer2.0 | 0.004 | 0.992 | 0.916 | True | 0.994 | 0.024 | 0.951 | 0.958 | 0.797 |
| layer2.1 | layer2.5 | 0.015 | 0.977 | 0.488 | True | 0.990 | 0.030 | 0.936 | 0.942 | 0.766 |
| layer2.2 | layer2.8 | 0.015 | 0.984 | 0.526 | True | 0.988 | 0.023 | 0.916 | 0.946 | 0.750 |
| layer3.0 | layer3.0 | 0.021 | 0.972 | 0.990 | True | 0.959 | 0.022 | 0.887 | 0.939 | 0.797 |
| layer3.1 | layer3.5 | 0.082 | 0.894 | 0.731 | True | 0.973 | 0.023 | 0.813 | 0.855 | 0.906 |
| layer3.2 | layer3.8 | 0.009 | 0.906 | 0.830 | True | 0.950 | 0.038 | 0.900 | 0.951 | 1.000 |
| penult | penult | 0.009 | 0.906 | 0.830 | True | 0.950 | 0.038 | 0.900 | 0.951 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.967 | 0.825 | 0.119 | 0.975 | 0.519 | 0.961 | 0.814 | 0.122 | 0.970 |
| layer1.0 | 0.813 | 0.519 | 0.115 | 0.865 | 0.519 | 0.767 | 0.513 | 0.118 | 0.840 |
| layer1.1 | 0.843 | 0.621 | 0.109 | 0.860 | 0.519 | 0.783 | 0.570 | 0.113 | 0.828 |
| layer1.2 | 0.944 | 0.740 | 0.110 | 0.934 | 0.519 | 0.926 | 0.726 | 0.121 | 0.930 |
| layer2.0 | 0.953 | 0.788 | 0.108 | 0.963 | 0.519 | 0.940 | 0.758 | 0.117 | 0.957 |
| layer2.1 | 0.932 | 0.773 | 0.105 | 0.941 | 0.519 | 0.915 | 0.711 | 0.117 | 0.936 |
| layer2.2 | 0.921 | 0.781 | 0.105 | 0.934 | 0.519 | 0.899 | 0.699 | 0.117 | 0.916 |
| layer3.0 | 0.889 | 0.805 | 0.103 | 0.925 | 0.519 | 0.869 | 0.682 | 0.116 | 0.907 |
| layer3.1 | 0.766 | 0.850 | 0.101 | 0.752 | 0.519 | 0.740 | 0.570 | 0.141 | 0.709 |
| layer3.2 | 0.866 | 0.928 | 0.100 | 0.711 | 0.519 | 0.664 | 0.571 | 0.126 | 0.742 |
| penult | 0.866 | 0.928 | 0.100 | 0.711 | 0.519 | 0.664 | 0.571 | 0.126 | 0.742 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | -0.063 | -0.383 | +0.000 | -0.010 | +204.659 | +0.008 | -0.034 | -0.007 | · | · | · | · | · |
| layer1.0 | -1.465 | -1.358 | -2.000 | -0.040 | +219.271 | +0.145 | -0.314 | -0.046 | · | · | · | · | · |
| layer1.1 | -0.039 | +0.527 | +0.000 | +0.056 | -26.009 | -0.011 | +0.008 | +0.031 | · | · | · | · | · |
| layer1.2 | -0.321 | +0.263 | +0.000 | +0.038 | -0.532 | -0.019 | -0.107 | +0.014 | · | · | · | · | · |
| layer2.0 | -0.839 | -0.849 | -2.000 | +0.024 | +0.879 | -0.006 | -0.063 | +0.004 | · | · | · | · | · |
| layer2.1 | +0.007 | -1.075 | +0.000 | +0.053 | -5.523 | -0.036 | -0.035 | +0.034 | · | · | · | · | · |
| layer2.2 | -0.332 | -1.418 | -2.000 | +0.077 | -2.689 | -0.018 | -0.030 | +0.019 | · | · | · | · | · |
| layer3.0 | -2.325 | -1.786 | -12.000 | -0.006 | +0.584 | +0.001 | -0.478 | -0.011 | · | · | · | · | · |
| layer3.1 | -0.452 | +4.957 | -1.000 | +0.250 | -0.271 | -0.112 | -0.047 | +0.037 | · | · | · | · | · |
| layer3.2 | +0.964 | +0.538 | +0.000 | +0.481 | -0.050 | -0.027 | +0.094 | -0.003 | · | · | · | · | · |
| penult | +0.964 | +0.538 | +0.000 | +0.481 | -0.050 | -0.027 | +0.094 | -0.003 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer2.0 | layer1.5 |
| spectral_slope | layer2.0 | layer2.0 |
| spectral_anisotropy | layer2.0 | layer3.0 |
| noise_sigma | layer1.0 | layer1.0 |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | stem | layer1.0 |
| edge_density | layer1.0 | stem |
| orientation_entropy | layer2.0 | layer2.8 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.5 |
| coarse_animal_vehicle | layer3.0 | layer3.0 |
| corruption_family | layer2.0 | layer1.8 |
| corruption_type | layer2.0 | layer2.0 |
| severity | layer2.0 | layer3.0 |