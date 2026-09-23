# DEFORMATION  A=`results/atlas_v1_resnet20_s3_st3`  B=`results/atlas_v1_resnet56_s12m`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.033 | 0.970 | 0.940 | True | 0.953 | 0.038 | 0.934 | 0.852 | 0.734 |
| layer1.0 | layer1.0 | 0.025 | 0.985 | 0.995 | True | 0.933 | 0.055 | 0.807 | 0.920 | 0.641 |
| layer1.1 | layer1.5 | 0.016 | 0.985 | 1.000 | True | 0.981 | 0.027 | 0.943 | 0.954 | 0.672 |
| layer1.2 | layer1.8 | 0.013 | 0.988 | 0.989 | True | 0.988 | 0.019 | 0.956 | 0.952 | 0.781 |
| layer2.0 | layer2.0 | 0.026 | 0.967 | 0.945 | True | 0.990 | 0.018 | 0.900 | 0.932 | 0.750 |
| layer2.1 | layer2.5 | 0.021 | 0.969 | 0.973 | True | 0.977 | 0.033 | 0.925 | 0.949 | 0.750 |
| layer2.2 | layer2.8 | 0.014 | 0.981 | 0.992 | True | 0.961 | 0.031 | 0.922 | 0.956 | 0.797 |
| layer3.0 | layer3.0 | 0.011 | 0.991 | 0.985 | True | 0.977 | 0.023 | 0.915 | 0.955 | 0.734 |
| layer3.1 | layer3.5 | 0.027 | 0.913 | 0.948 | True | 0.975 | 0.030 | 0.814 | 0.889 | 0.891 |
| layer3.2 | layer3.8 | 0.007 | 0.919 | 0.814 | True | 0.963 | 0.030 | 0.887 | 0.956 | 1.000 |
| penult | penult | 0.007 | 0.919 | 0.814 | True | 0.963 | 0.030 | 0.887 | 0.956 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.865 | 0.682 | 0.112 | 0.827 | 0.571 | 0.833 | 0.681 | 0.115 | 0.804 |
| layer1.0 | 0.854 | 0.641 | 0.119 | 0.903 | 0.571 | 0.769 | 0.589 | 0.124 | 0.879 |
| layer1.1 | 0.904 | 0.724 | 0.117 | 0.940 | 0.571 | 0.866 | 0.698 | 0.117 | 0.920 |
| layer1.2 | 0.947 | 0.750 | 0.112 | 0.942 | 0.571 | 0.930 | 0.733 | 0.122 | 0.923 |
| layer2.0 | 0.909 | 0.777 | 0.107 | 0.915 | 0.571 | 0.894 | 0.737 | 0.115 | 0.894 |
| layer2.1 | 0.928 | 0.778 | 0.104 | 0.941 | 0.571 | 0.916 | 0.718 | 0.111 | 0.925 |
| layer2.2 | 0.924 | 0.769 | 0.103 | 0.947 | 0.571 | 0.905 | 0.726 | 0.110 | 0.935 |
| layer3.0 | 0.902 | 0.801 | 0.102 | 0.940 | 0.571 | 0.867 | 0.678 | 0.110 | 0.926 |
| layer3.1 | 0.785 | 0.837 | 0.101 | 0.810 | 0.571 | 0.741 | 0.598 | 0.122 | 0.749 |
| layer3.2 | 0.874 | 0.927 | 0.100 | 0.771 | 0.571 | 0.661 | 0.581 | 0.122 | 0.730 |
| penult | 0.874 | 0.927 | 0.100 | 0.771 | 0.571 | 0.661 | 0.581 | 0.122 | 0.730 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | -0.447 | +0.229 | +0.000 | +0.013 | -53.444 | +0.013 | -0.148 | -0.002 | · | · | · | · | · |
| layer1.0 | -1.015 | -1.421 | -2.000 | -0.026 | -4.273 | +0.121 | -0.248 | -0.032 | · | · | · | · | · |
| layer1.1 | -0.055 | +0.183 | +0.000 | -0.007 | +6.104 | -0.057 | -0.096 | +0.006 | · | · | · | · | · |
| layer1.2 | -0.190 | -0.284 | +0.000 | +0.010 | +18.026 | -0.035 | +0.063 | +0.004 | · | · | · | · | · |
| layer2.0 | -0.843 | -0.290 | -3.000 | +0.014 | +5.197 | -0.018 | -0.177 | -0.006 | · | · | · | · | · |
| layer2.1 | +0.274 | +0.300 | -1.000 | +0.064 | -3.777 | -0.045 | +0.005 | +0.049 | · | · | · | · | · |
| layer2.2 | -0.256 | -0.817 | -1.000 | +0.062 | +0.049 | +0.007 | -0.160 | +0.031 | · | · | · | · | · |
| layer3.0 | -2.362 | -2.829 | -13.000 | +0.011 | +0.753 | -0.005 | -0.440 | -0.016 | · | · | · | · | · |
| layer3.1 | -0.884 | +0.215 | +0.000 | -0.021 | +0.001 | -0.001 | +0.117 | +0.001 | · | · | · | · | · |
| layer3.2 | +0.885 | +0.491 | +0.000 | +0.397 | -0.043 | -0.024 | +0.106 | -0.001 | · | · | · | · | · |
| penult | +0.885 | +0.491 | +0.000 | +0.397 | -0.043 | -0.024 | +0.106 | -0.001 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer1.0 | layer2.0 |
| spectral_slope | layer2.0 | layer2.0 |
| spectral_anisotropy | layer2.0 | layer2.5 |
| noise_sigma | layer1.0 | layer1.0 |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | layer1.0 | layer1.0 |
| edge_density | layer1.0 | layer2.0 |
| orientation_entropy | layer3.0 | layer2.5 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.5 |
| coarse_animal_vehicle | layer3.0 | layer3.0 |
| corruption_family | layer2.0 | layer2.0 |
| corruption_type | layer2.0 | layer2.0 |
| severity | layer3.0 | layer2.0 |