# DEFORMATION  A=`results/atlas_v1_resnet20_s0hub_st3`  B=`results/atlas_v1_resnet56_s12m`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.024 | 0.985 | 0.948 | True | 0.953 | 0.031 | 0.950 | 0.914 | 0.797 |
| layer1.0 | layer1.0 | 0.025 | 0.983 | 1.000 | True | 0.936 | 0.079 | 0.838 | 0.912 | 0.609 |
| layer1.1 | layer1.5 | 0.008 | 0.988 | 0.942 | True | 0.990 | 0.033 | 0.879 | 0.927 | 0.719 |
| layer1.2 | layer1.8 | 0.012 | 0.990 | 0.991 | True | 0.979 | 0.031 | 0.947 | 0.947 | 0.766 |
| layer2.0 | layer2.0 | 0.014 | 0.981 | 0.986 | True | 0.992 | 0.015 | 0.936 | 0.951 | 0.828 |
| layer2.1 | layer2.5 | 0.017 | 0.972 | 0.910 | True | 0.983 | 0.029 | 0.919 | 0.928 | 0.797 |
| layer2.2 | layer2.8 | 0.012 | 0.982 | 0.942 | True | 0.979 | 0.029 | 0.913 | 0.940 | 0.812 |
| layer3.0 | layer3.0 | 0.018 | 0.965 | 0.955 | True | 0.961 | 0.024 | 0.902 | 0.953 | 0.844 |
| layer3.1 | layer3.5 | 0.074 | 0.772 | 0.920 | True | 0.965 | 0.033 | 0.784 | 0.831 | 0.797 |
| layer3.2 | layer3.8 | 0.005 | 0.908 | 0.814 | True | 0.913 | 0.043 | 0.905 | 0.955 | 1.000 |
| penult | penult | 0.005 | 0.908 | 0.814 | True | 0.913 | 0.043 | 0.905 | 0.955 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.905 | 0.737 | 0.112 | 0.905 | 0.463 | 0.878 | 0.735 | 0.116 | 0.897 |
| layer1.0 | 0.889 | 0.646 | 0.117 | 0.901 | 0.463 | 0.831 | 0.626 | 0.123 | 0.882 |
| layer1.1 | 0.914 | 0.715 | 0.112 | 0.928 | 0.463 | 0.823 | 0.672 | 0.114 | 0.910 |
| layer1.2 | 0.943 | 0.724 | 0.112 | 0.943 | 0.463 | 0.916 | 0.712 | 0.122 | 0.930 |
| layer2.0 | 0.934 | 0.769 | 0.108 | 0.934 | 0.463 | 0.917 | 0.752 | 0.118 | 0.921 |
| layer2.1 | 0.920 | 0.742 | 0.104 | 0.925 | 0.463 | 0.897 | 0.683 | 0.112 | 0.901 |
| layer2.2 | 0.908 | 0.750 | 0.103 | 0.933 | 0.463 | 0.881 | 0.685 | 0.110 | 0.912 |
| layer3.0 | 0.892 | 0.808 | 0.102 | 0.925 | 0.463 | 0.870 | 0.697 | 0.111 | 0.912 |
| layer3.1 | 0.707 | 0.826 | 0.101 | 0.679 | 0.463 | 0.719 | 0.580 | 0.129 | 0.675 |
| layer3.2 | 0.875 | 0.908 | 0.100 | 0.765 | 0.463 | 0.656 | 0.583 | 0.123 | 0.732 |
| penult | 0.875 | 0.908 | 0.100 | 0.765 | 0.463 | 0.656 | 0.583 | 0.123 | 0.732 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | -0.499 | -0.105 | +0.000 | +0.008 | +5.423 | +0.033 | -0.137 | +0.000 | · | · | · | · | · |
| layer1.0 | -1.015 | -1.160 | -1.000 | -0.009 | -19.524 | +0.102 | -0.236 | -0.030 | · | · | · | · | · |
| layer1.1 | -0.405 | -0.045 | +0.000 | +0.010 | +17.886 | -0.019 | -0.079 | -0.001 | · | · | · | · | · |
| layer1.2 | -0.397 | -0.516 | +0.000 | +0.015 | +13.764 | -0.034 | -0.040 | -0.005 | · | · | · | · | · |
| layer2.0 | -1.232 | -1.255 | -4.000 | +0.017 | +7.879 | -0.005 | -0.164 | -0.026 | · | · | · | · | · |
| layer2.1 | +0.005 | -0.497 | -1.000 | +0.045 | -0.209 | -0.038 | -0.024 | +0.015 | · | · | · | · | · |
| layer2.2 | -0.284 | -1.264 | -2.000 | +0.044 | +1.804 | -0.009 | +0.018 | +0.010 | · | · | · | · | · |
| layer3.0 | -2.483 | -3.432 | -13.000 | -0.013 | +1.078 | +0.042 | -0.399 | -0.025 | · | · | · | · | · |
| layer3.1 | -0.448 | +2.357 | +1.000 | -0.100 | +0.165 | +0.000 | -0.035 | -0.006 | · | · | · | · | · |
| layer3.2 | +0.806 | +0.433 | +0.000 | +0.371 | -0.041 | -0.022 | +0.115 | -0.000 | · | · | · | · | · |
| penult | +0.806 | +0.433 | +0.000 | +0.371 | -0.041 | -0.022 | +0.115 | -0.000 | · | · | · | · | · |

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
| orientation_entropy | layer2.0 | layer2.5 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.5 |
| coarse_animal_vehicle | layer3.0 | layer3.0 |
| corruption_family | layer2.0 | layer2.0 |
| corruption_type | layer1.2 | layer2.0 |
| severity | layer3.0 | layer2.0 |