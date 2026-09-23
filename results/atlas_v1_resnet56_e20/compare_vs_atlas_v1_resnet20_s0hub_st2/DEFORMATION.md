# DEFORMATION  A=`results/atlas_v1_resnet20_s0hub_st2`  B=`results/atlas_v1_resnet56_e20`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.014 | 0.990 | 0.997 | True | 0.899 | 0.048 | 0.916 | 0.871 | 0.516 |
| layer1.0 | layer1.0 | 0.030 | 0.976 | 0.992 | True | 0.878 | 0.122 | 0.853 | 0.952 | 0.594 |
| layer1.1 | layer1.5 | 0.032 | 0.935 | 0.670 | True | 0.969 | 0.040 | 0.801 | 0.852 | 0.750 |
| layer1.2 | layer1.8 | 0.021 | 0.957 | 0.976 | True | 0.992 | 0.033 | 0.907 | 0.909 | 0.625 |
| layer2.0 | layer2.0 | 0.028 | 0.977 | 0.991 | True | 0.990 | 0.040 | 0.907 | 0.916 | 0.750 |
| layer2.1 | layer2.5 | 0.018 | 0.907 | 0.939 | True | 0.983 | 0.034 | 0.891 | 0.905 | 0.750 |
| layer2.2 | layer2.8 | 0.019 | 0.964 | 0.715 | True | 0.986 | 0.038 | 0.886 | 0.927 | 0.766 |
| layer3.0 | layer3.0 | 0.035 | 0.955 | 0.902 | True | 0.915 | 0.041 | 0.846 | 0.914 | 0.812 |
| layer3.1 | layer3.5 | 0.062 | 0.808 | 0.895 | True | 0.965 | 0.040 | 0.760 | 0.829 | 0.812 |
| layer3.2 | layer3.8 | 0.024 | 0.966 | 0.848 | True | 0.955 | 0.055 | 0.840 | 0.927 | 0.953 |
| penult | penult | 0.024 | 0.966 | 0.848 | True | 0.955 | 0.055 | 0.840 | 0.927 | 0.953 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.887 | 0.595 | 0.108 | 0.883 | 0.465 | 0.832 | 0.601 | 0.113 | 0.867 |
| layer1.0 | 0.912 | 0.630 | 0.110 | 0.937 | 0.465 | 0.860 | 0.618 | 0.115 | 0.927 |
| layer1.1 | 0.842 | 0.670 | 0.111 | 0.854 | 0.465 | 0.732 | 0.631 | 0.114 | 0.818 |
| layer1.2 | 0.902 | 0.677 | 0.111 | 0.887 | 0.465 | 0.861 | 0.657 | 0.118 | 0.874 |
| layer2.0 | 0.908 | 0.662 | 0.105 | 0.887 | 0.465 | 0.872 | 0.624 | 0.109 | 0.873 |
| layer2.1 | 0.827 | 0.711 | 0.104 | 0.896 | 0.465 | 0.872 | 0.651 | 0.109 | 0.877 |
| layer2.2 | 0.877 | 0.741 | 0.103 | 0.910 | 0.465 | 0.850 | 0.668 | 0.108 | 0.889 |
| layer3.0 | 0.856 | 0.756 | 0.102 | 0.896 | 0.465 | 0.843 | 0.647 | 0.109 | 0.880 |
| layer3.1 | 0.722 | 0.819 | 0.101 | 0.710 | 0.465 | 0.684 | 0.544 | 0.126 | 0.640 |
| layer3.2 | 0.853 | 0.894 | 0.100 | 0.825 | 0.465 | 0.650 | 0.547 | 0.123 | 0.751 |
| penult | 0.853 | 0.894 | 0.100 | 0.825 | 0.465 | 0.650 | 0.547 | 0.123 | 0.751 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | -0.865 | +0.159 | +0.000 | +0.013 | +251.508 | +0.016 | -0.131 | -0.007 | · | · | · | · | · |
| layer1.0 | -2.155 | -0.665 | -2.000 | -0.003 | +66.041 | +0.025 | -0.452 | -0.031 | · | · | · | · | · |
| layer1.1 | -1.001 | -0.973 | +0.000 | +0.004 | +39.585 | +0.095 | -0.188 | -0.013 | · | · | · | · | · |
| layer1.2 | -0.974 | -0.929 | -2.000 | +0.012 | +7.121 | +0.003 | -0.221 | -0.013 | · | · | · | · | · |
| layer2.0 | -2.366 | -1.913 | -5.000 | -0.010 | +24.582 | +0.017 | -0.287 | -0.039 | · | · | · | · | · |
| layer2.1 | -1.960 | -1.017 | -4.000 | +0.013 | +13.702 | -0.032 | -0.316 | -0.018 | · | · | · | · | · |
| layer2.2 | -2.038 | -1.667 | -5.000 | +0.004 | +7.562 | -0.003 | -0.195 | -0.033 | · | · | · | · | · |
| layer3.0 | -3.565 | -4.737 | -18.000 | -0.082 | +2.753 | +0.076 | -0.613 | -0.088 | · | · | · | · | · |
| layer3.1 | -1.332 | +0.444 | -8.000 | -0.031 | +0.227 | -0.021 | -0.237 | -0.003 | · | · | · | · | · |
| layer3.2 | +1.825 | -0.723 | +1.000 | -0.957 | +0.150 | +0.079 | +0.137 | -0.031 | · | · | · | · | · |
| penult | +1.825 | -0.723 | +1.000 | -0.957 | +0.150 | +0.079 | +0.137 | -0.031 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer1.0 | layer2.0 |
| spectral_slope | layer2.0 | layer2.0 |
| spectral_anisotropy | layer2.0 | layer3.0 |
| noise_sigma | layer1.0 | layer1.5 |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | layer1.0 | layer1.5 |
| edge_density | layer1.0 | stem |
| orientation_entropy | layer2.0 | layer2.5 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.5 |
| coarse_animal_vehicle | layer3.0 | layer3.0 |
| corruption_family | layer2.0 | layer2.0 |
| corruption_type | layer1.2 | layer2.0 |
| severity | layer3.0 | layer3.0 |