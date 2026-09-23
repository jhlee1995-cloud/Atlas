# DEFORMATION  A=`results/atlas_v1_resnet20_s1_st3`  B=`results/atlas_v1_resnet56_e70`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.012 | 0.991 | 0.984 | True | 0.930 | 0.034 | 0.958 | 0.965 | 0.797 |
| layer1.0 | layer1.0 | 0.020 | 0.976 | 0.997 | True | 0.975 | 0.036 | 0.829 | 0.903 | 0.641 |
| layer1.1 | layer1.5 | 0.046 | 0.950 | 0.951 | True | 0.994 | 0.036 | 0.847 | 0.872 | 0.578 |
| layer1.2 | layer1.8 | 0.019 | 0.976 | 0.978 | True | 0.998 | 0.023 | 0.950 | 0.927 | 0.688 |
| layer2.0 | layer2.0 | 0.013 | 0.975 | 0.880 | True | 0.990 | 0.025 | 0.935 | 0.951 | 0.734 |
| layer2.1 | layer2.5 | 0.013 | 0.982 | 0.552 | True | 0.977 | 0.031 | 0.933 | 0.951 | 0.766 |
| layer2.2 | layer2.8 | 0.015 | 0.984 | 0.530 | True | 0.983 | 0.028 | 0.926 | 0.937 | 0.672 |
| layer3.0 | layer3.0 | 0.013 | 0.979 | 0.918 | True | 0.981 | 0.024 | 0.870 | 0.935 | 0.812 |
| layer3.1 | layer3.5 | 0.060 | 0.894 | 0.654 | True | 0.994 | 0.022 | 0.814 | 0.872 | 0.875 |
| layer3.2 | layer3.8 | 0.007 | 0.911 | 0.733 | True | 0.979 | 0.028 | 0.895 | 0.946 | 1.000 |
| penult | penult | 0.007 | 0.911 | 0.733 | True | 0.979 | 0.028 | 0.895 | 0.946 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.950 | 0.770 | 0.120 | 0.954 | 0.523 | 0.936 | 0.764 | 0.121 | 0.947 |
| layer1.0 | 0.909 | 0.684 | 0.116 | 0.917 | 0.523 | 0.821 | 0.661 | 0.119 | 0.890 |
| layer1.1 | 0.818 | 0.591 | 0.109 | 0.847 | 0.523 | 0.793 | 0.556 | 0.113 | 0.816 |
| layer1.2 | 0.939 | 0.761 | 0.111 | 0.920 | 0.523 | 0.931 | 0.761 | 0.120 | 0.922 |
| layer2.0 | 0.939 | 0.775 | 0.107 | 0.957 | 0.523 | 0.926 | 0.760 | 0.114 | 0.951 |
| layer2.1 | 0.933 | 0.791 | 0.106 | 0.949 | 0.523 | 0.911 | 0.710 | 0.115 | 0.935 |
| layer2.2 | 0.923 | 0.779 | 0.105 | 0.924 | 0.523 | 0.906 | 0.683 | 0.115 | 0.898 |
| layer3.0 | 0.901 | 0.813 | 0.103 | 0.937 | 0.523 | 0.878 | 0.692 | 0.115 | 0.919 |
| layer3.1 | 0.788 | 0.856 | 0.101 | 0.759 | 0.523 | 0.754 | 0.591 | 0.139 | 0.745 |
| layer3.2 | 0.872 | 0.925 | 0.100 | 0.727 | 0.523 | 0.665 | 0.578 | 0.127 | 0.739 |
| penult | 0.872 | 0.925 | 0.100 | 0.727 | 0.523 | 0.665 | 0.578 | 0.127 | 0.739 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | +0.051 | -0.285 | +0.000 | -0.002 | +102.772 | +0.023 | -0.011 | -0.001 | · | · | · | · | · |
| layer1.0 | -0.792 | -0.696 | +0.000 | -0.027 | +148.524 | +0.029 | -0.109 | -0.031 | · | · | · | · | · |
| layer1.1 | +0.541 | +1.350 | +1.000 | +0.062 | -13.470 | -0.096 | +0.164 | +0.037 | · | · | · | · | · |
| layer1.2 | +0.421 | +0.566 | +1.000 | +0.028 | -1.459 | -0.015 | +0.101 | +0.017 | · | · | · | · | · |
| layer2.0 | -0.370 | -0.771 | -2.000 | +0.022 | -2.070 | -0.067 | +0.021 | +0.001 | · | · | · | · | · |
| layer2.1 | +0.799 | -1.276 | +1.000 | +0.032 | -7.930 | -0.014 | +0.067 | +0.030 | · | · | · | · | · |
| layer2.2 | +0.206 | -1.400 | -1.000 | +0.073 | -3.041 | +0.006 | -0.036 | +0.025 | · | · | · | · | · |
| layer3.0 | -2.029 | -1.637 | -13.000 | +0.001 | +0.590 | +0.010 | -0.295 | -0.014 | · | · | · | · | · |
| layer3.1 | -0.396 | +3.150 | -1.000 | +0.210 | -0.253 | -0.104 | -0.085 | +0.046 | · | · | · | · | · |
| layer3.2 | +1.109 | +0.472 | +0.000 | +0.439 | -0.046 | -0.027 | +0.065 | +0.003 | · | · | · | · | · |
| penult | +1.109 | +0.472 | +0.000 | +0.439 | -0.046 | -0.027 | +0.065 | +0.003 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer1.1 | layer1.5 |
| spectral_slope | layer2.0 | layer2.0 |
| spectral_anisotropy | layer1.2 | layer3.0 |
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
| severity | layer3.0 | layer3.0 |