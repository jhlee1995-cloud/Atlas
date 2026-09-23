# DEFORMATION  A=`results/atlas_v1_resnet20_s1`  B=`results/atlas_v1_resnet20_s3`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.006 | 0.993 | 0.997 | True | 0.994 | 0.015 | 0.954 | 0.978 | 0.906 |
| layer1.0 | layer1.0 | 0.015 | 0.984 | 0.995 | True | 0.977 | 0.017 | 0.888 | 0.937 | 0.703 |
| layer1.1 | layer1.1 | 0.013 | 0.983 | 0.962 | True | 0.994 | 0.022 | 0.922 | 0.911 | 0.656 |
| layer1.2 | layer1.2 | 0.011 | 0.984 | 0.981 | True | 0.990 | 0.017 | 0.941 | 0.899 | 0.703 |
| layer2.0 | layer2.0 | 0.009 | 0.980 | 0.954 | True | 0.996 | 0.013 | 0.958 | 0.958 | 0.734 |
| layer2.1 | layer2.1 | 0.009 | 0.983 | 0.980 | True | 0.992 | 0.014 | 0.944 | 0.955 | 0.781 |
| layer2.2 | layer2.2 | 0.009 | 0.984 | 1.000 | True | 0.992 | 0.016 | 0.935 | 0.957 | 0.828 |
| layer3.0 | layer3.0 | 0.011 | 0.978 | 0.962 | True | 0.994 | 0.018 | 0.905 | 0.952 | 0.844 |
| layer3.1 | layer3.1 | 0.034 | 0.875 | 0.920 | True | 0.971 | 0.029 | 0.843 | 0.900 | 0.891 |
| layer3.2 | layer3.2 | 0.004 | 0.965 | 0.771 | True | 0.946 | 0.026 | 0.899 | 0.962 | 1.000 |
| penult | penult | 0.004 | 0.965 | 0.771 | True | 0.946 | 0.026 | 0.899 | 0.962 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.963 | 0.808 | 0.119 | 0.963 | 0.537 | 0.956 | 0.804 | 0.121 | 0.951 |
| layer1.0 | 0.893 | 0.718 | 0.120 | 0.940 | 0.537 | 0.826 | 0.686 | 0.123 | 0.921 |
| layer1.1 | 0.898 | 0.671 | 0.118 | 0.917 | 0.537 | 0.853 | 0.638 | 0.119 | 0.895 |
| layer1.2 | 0.942 | 0.742 | 0.112 | 0.905 | 0.537 | 0.921 | 0.731 | 0.120 | 0.902 |
| layer2.0 | 0.957 | 0.811 | 0.106 | 0.961 | 0.537 | 0.943 | 0.790 | 0.114 | 0.958 |
| layer2.1 | 0.948 | 0.811 | 0.105 | 0.953 | 0.537 | 0.937 | 0.772 | 0.112 | 0.946 |
| layer2.2 | 0.935 | 0.798 | 0.104 | 0.955 | 0.537 | 0.922 | 0.760 | 0.111 | 0.947 |
| layer3.0 | 0.900 | 0.808 | 0.102 | 0.934 | 0.537 | 0.881 | 0.691 | 0.112 | 0.919 |
| layer3.1 | 0.810 | 0.854 | 0.101 | 0.811 | 0.537 | 0.760 | 0.604 | 0.131 | 0.794 |
| layer3.2 | 0.883 | 0.929 | 0.100 | 0.820 | 0.537 | 0.693 | 0.577 | 0.126 | 0.772 |
| penult | 0.883 | 0.929 | 0.100 | 0.820 | 0.537 | 0.693 | 0.577 | 0.126 | 0.772 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | +0.046 | -0.348 | +0.000 | -0.004 | +57.725 | +0.014 | +0.045 | +0.002 | · | · | · | · | · |
| layer1.0 | +0.253 | +1.015 | +2.000 | +0.011 | -41.535 | -0.074 | +0.107 | +0.010 | · | · | · | · | · |
| layer1.1 | +0.249 | +0.874 | +1.000 | +0.018 | -4.549 | -0.012 | +0.171 | +0.011 | · | · | · | · | · |
| layer1.2 | +0.436 | +0.487 | +1.000 | -0.010 | -0.334 | +0.037 | +0.054 | -0.002 | · | · | · | · | · |
| layer2.0 | +0.117 | -0.215 | +0.000 | -0.009 | +0.746 | -0.048 | +0.051 | -0.014 | · | · | · | · | · |
| layer2.1 | +0.378 | -0.711 | +1.000 | -0.026 | -1.064 | +0.046 | +0.008 | -0.020 | · | · | · | · | · |
| layer2.2 | +0.486 | -0.019 | +0.000 | -0.009 | -0.357 | +0.029 | -0.008 | -0.007 | · | · | · | · | · |
| layer3.0 | -0.031 | +1.247 | -1.000 | -0.018 | +0.350 | +0.027 | +0.179 | -0.008 | · | · | · | · | · |
| layer3.1 | +0.344 | +1.350 | +1.000 | -0.053 | +0.149 | -0.012 | -0.143 | -0.018 | · | · | · | · | · |
| layer3.2 | +0.080 | -0.193 | +0.000 | -0.097 | +0.009 | +0.011 | -0.061 | +0.002 | · | · | · | · | · |
| penult | +0.080 | -0.193 | +0.000 | -0.097 | +0.009 | +0.011 | -0.061 | +0.002 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer1.1 | layer1.0 |
| spectral_slope | layer2.0 | layer2.0 |
| spectral_anisotropy | layer1.2 | layer2.0 |
| noise_sigma | layer1.0 | layer1.0 |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | stem | layer1.0 |
| edge_density | layer1.0 | layer1.0 |
| orientation_entropy | layer2.0 | layer3.0 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.1 |
| coarse_animal_vehicle | layer3.0 | layer3.0 |
| corruption_family | layer2.0 | layer2.0 |
| corruption_type | layer2.0 | layer2.0 |
| severity | layer3.0 | layer3.0 |