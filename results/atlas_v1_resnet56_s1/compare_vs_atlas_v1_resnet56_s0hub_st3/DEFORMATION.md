# DEFORMATION  A=`results/atlas_v1_resnet56_s0hub_st3`  B=`results/atlas_v1_resnet56_s1`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.006 | 0.992 | 0.987 | True | 0.996 | 0.028 | 0.923 | 0.961 | 0.828 |
| layer1.0 | layer1.0 | 0.013 | 0.990 | 0.984 | True | 0.911 | 0.038 | 0.877 | 0.947 | 0.719 |
| layer1.5 | layer1.5 | 0.005 | 0.991 | 1.000 | True | 0.975 | 0.037 | 0.957 | 0.961 | 0.797 |
| layer1.8 | layer1.8 | 0.005 | 0.992 | 0.991 | True | 0.994 | 0.025 | 0.965 | 0.954 | 0.688 |
| layer2.0 | layer2.0 | 0.006 | 0.990 | 0.927 | True | 0.994 | 0.018 | 0.967 | 0.981 | 0.922 |
| layer2.5 | layer2.5 | 0.009 | 0.986 | 0.939 | True | 0.990 | 0.019 | 0.956 | 0.973 | 0.875 |
| layer2.8 | layer2.8 | 0.010 | 0.979 | 0.935 | True | 0.992 | 0.018 | 0.945 | 0.965 | 0.828 |
| layer3.0 | layer3.0 | 0.011 | 0.962 | 0.964 | True | 0.986 | 0.018 | 0.935 | 0.958 | 0.797 |
| layer3.5 | layer3.5 | 0.017 | 0.944 | 0.887 | True | 0.986 | 0.017 | 0.873 | 0.929 | 0.781 |
| layer3.8 | layer3.8 | 0.003 | 0.886 | 0.816 | True | 0.930 | 0.028 | 0.951 | 0.973 | 1.000 |
| penult | penult | 0.003 | 0.886 | 0.816 | True | 0.930 | 0.028 | 0.951 | 0.973 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.942 | 0.780 | 0.118 | 0.944 | 0.543 | 0.927 | 0.789 | 0.123 | 0.925 |
| layer1.0 | 0.931 | 0.738 | 0.114 | 0.954 | 0.543 | 0.858 | 0.710 | 0.116 | 0.940 |
| layer1.5 | 0.958 | 0.750 | 0.116 | 0.949 | 0.543 | 0.936 | 0.706 | 0.120 | 0.944 |
| layer1.8 | 0.970 | 0.800 | 0.109 | 0.948 | 0.543 | 0.960 | 0.783 | 0.119 | 0.937 |
| layer2.0 | 0.967 | 0.833 | 0.105 | 0.975 | 0.543 | 0.952 | 0.824 | 0.111 | 0.971 |
| layer2.5 | 0.951 | 0.822 | 0.103 | 0.960 | 0.543 | 0.944 | 0.785 | 0.110 | 0.954 |
| layer2.8 | 0.947 | 0.826 | 0.103 | 0.958 | 0.543 | 0.942 | 0.774 | 0.108 | 0.952 |
| layer3.0 | 0.936 | 0.803 | 0.102 | 0.951 | 0.543 | 0.923 | 0.729 | 0.108 | 0.942 |
| layer3.5 | 0.844 | 0.870 | 0.101 | 0.869 | 0.543 | 0.790 | 0.657 | 0.137 | 0.827 |
| layer3.8 | 0.926 | 0.953 | 0.100 | 0.737 | 0.543 | 0.683 | 0.620 | 0.123 | 0.736 |
| penult | 0.926 | 0.953 | 0.100 | 0.737 | 0.543 | 0.683 | 0.620 | 0.123 | 0.736 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | -0.335 | -0.299 | -1.000 | -0.001 | +4.856 | +0.013 | -0.050 | -0.014 | · | · | · | · | · |
| layer1.0 | -0.363 | -0.562 | -1.000 | +0.004 | +92.583 | +0.015 | -0.111 | +0.008 | · | · | · | · | · |
| layer1.5 | +0.103 | -0.365 | -1.000 | +0.000 | +14.878 | +0.001 | -0.064 | -0.000 | · | · | · | · | · |
| layer1.8 | -0.136 | -0.224 | -1.000 | -0.008 | -3.605 | +0.012 | -0.056 | -0.005 | · | · | · | · | · |
| layer2.0 | -0.307 | -0.387 | -1.000 | -0.020 | +6.871 | +0.041 | -0.075 | -0.005 | · | · | · | · | · |
| layer2.5 | -0.406 | -1.087 | -1.000 | -0.018 | +2.040 | +0.030 | -0.130 | -0.030 | · | · | · | · | · |
| layer2.8 | -0.204 | -1.110 | +0.000 | -0.019 | +1.041 | +0.045 | +0.162 | -0.020 | · | · | · | · | · |
| layer3.0 | -0.091 | -0.541 | +0.000 | -0.021 | -0.012 | +0.031 | -0.077 | -0.020 | · | · | · | · | · |
| layer3.5 | +0.238 | -0.166 | +0.000 | -0.033 | +0.005 | +0.006 | +0.164 | -0.013 | · | · | · | · | · |
| layer3.8 | -0.287 | -0.074 | +0.000 | -0.218 | +0.004 | +0.004 | -0.049 | +0.000 | · | · | · | · | · |
| penult | -0.287 | -0.074 | +0.000 | -0.218 | +0.004 | +0.004 | -0.049 | +0.000 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | layer1.5 |
| highfreq_ratio | layer2.0 | layer2.0 |
| spectral_slope | layer2.0 | layer2.0 |
| spectral_anisotropy | layer3.0 | layer2.0 |
| noise_sigma | layer1.0 | layer1.5 |
| saturation_mean | stem | layer1.0 |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | layer1.0 | layer1.0 |
| edge_density | layer1.0 | layer1.5 |
| orientation_entropy | layer2.0 | layer2.5 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.5 | layer3.5 |
| coarse_animal_vehicle | layer3.5 | layer3.5 |
| corruption_family | layer2.0 | layer2.0 |
| corruption_type | layer2.0 | layer2.0 |
| severity | layer3.0 | layer2.0 |