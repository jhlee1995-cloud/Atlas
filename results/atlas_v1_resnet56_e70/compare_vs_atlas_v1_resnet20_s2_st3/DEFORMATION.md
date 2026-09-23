# DEFORMATION  A=`results/atlas_v1_resnet20_s2_st3`  B=`results/atlas_v1_resnet56_e70`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.007 | 0.997 | 1.000 | True | 0.934 | 0.043 | 0.934 | 0.947 | 0.734 |
| layer1.0 | layer1.0 | 0.038 | 0.973 | 0.992 | True | 0.988 | 0.027 | 0.851 | 0.906 | 0.578 |
| layer1.1 | layer1.5 | 0.046 | 0.924 | 0.684 | True | 0.992 | 0.034 | 0.842 | 0.838 | 0.531 |
| layer1.2 | layer1.8 | 0.014 | 0.981 | 0.991 | True | 0.996 | 0.028 | 0.949 | 0.946 | 0.734 |
| layer2.0 | layer2.0 | 0.007 | 0.980 | 0.915 | True | 0.994 | 0.016 | 0.953 | 0.960 | 0.734 |
| layer2.1 | layer2.5 | 0.010 | 0.984 | 0.540 | True | 0.983 | 0.023 | 0.942 | 0.953 | 0.734 |
| layer2.2 | layer2.8 | 0.012 | 0.984 | 0.533 | True | 0.990 | 0.023 | 0.921 | 0.931 | 0.781 |
| layer3.0 | layer3.0 | 0.022 | 0.972 | 0.931 | True | 0.973 | 0.015 | 0.873 | 0.940 | 0.766 |
| layer3.1 | layer3.5 | 0.094 | 0.921 | 0.724 | True | 0.963 | 0.021 | 0.779 | 0.828 | 0.938 |
| layer3.2 | layer3.8 | 0.009 | 0.928 | 0.895 | True | 0.975 | 0.020 | 0.912 | 0.949 | 1.000 |
| penult | penult | 0.009 | 0.928 | 0.895 | True | 0.975 | 0.020 | 0.912 | 0.949 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.945 | 0.704 | 0.115 | 0.930 | 0.566 | 0.927 | 0.684 | 0.121 | 0.917 |
| layer1.0 | 0.882 | 0.626 | 0.118 | 0.899 | 0.566 | 0.838 | 0.606 | 0.121 | 0.883 |
| layer1.1 | 0.833 | 0.587 | 0.109 | 0.818 | 0.566 | 0.813 | 0.562 | 0.111 | 0.779 |
| layer1.2 | 0.948 | 0.750 | 0.113 | 0.943 | 0.566 | 0.932 | 0.735 | 0.124 | 0.936 |
| layer2.0 | 0.959 | 0.794 | 0.107 | 0.959 | 0.566 | 0.952 | 0.756 | 0.115 | 0.954 |
| layer2.1 | 0.942 | 0.764 | 0.105 | 0.948 | 0.566 | 0.932 | 0.708 | 0.115 | 0.940 |
| layer2.2 | 0.926 | 0.766 | 0.104 | 0.915 | 0.566 | 0.908 | 0.682 | 0.112 | 0.896 |
| layer3.0 | 0.869 | 0.800 | 0.102 | 0.917 | 0.566 | 0.850 | 0.654 | 0.111 | 0.897 |
| layer3.1 | 0.760 | 0.843 | 0.101 | 0.706 | 0.566 | 0.721 | 0.574 | 0.133 | 0.688 |
| layer3.2 | 0.871 | 0.928 | 0.100 | 0.715 | 0.566 | 0.672 | 0.582 | 0.126 | 0.744 |
| penult | 0.871 | 0.928 | 0.100 | 0.715 | 0.566 | 0.672 | 0.582 | 0.126 | 0.744 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | -0.200 | -0.267 | +0.000 | -0.008 | +210.916 | -0.001 | -0.041 | -0.003 | · | · | · | · | · |
| layer1.0 | -1.415 | -1.283 | -2.000 | -0.029 | +176.983 | +0.051 | -0.315 | -0.039 | · | · | · | · | · |
| layer1.1 | +0.043 | +0.353 | +0.000 | +0.062 | -7.625 | -0.127 | +0.038 | +0.029 | · | · | · | · | · |
| layer1.2 | -0.114 | +0.866 | +0.000 | +0.033 | +0.017 | -0.040 | -0.007 | +0.020 | · | · | · | · | · |
| layer2.0 | -0.212 | -0.302 | -1.000 | +0.023 | +2.343 | +0.001 | +0.032 | +0.008 | · | · | · | · | · |
| layer2.1 | +0.492 | -0.695 | +1.000 | +0.046 | -1.991 | -0.014 | +0.115 | +0.037 | · | · | · | · | · |
| layer2.2 | -0.059 | -0.771 | -1.000 | +0.075 | -1.222 | -0.025 | -0.051 | +0.027 | · | · | · | · | · |
| layer3.0 | -2.301 | -3.442 | -13.000 | -0.009 | +0.447 | +0.004 | -0.306 | -0.032 | · | · | · | · | · |
| layer3.1 | -0.533 | +4.768 | +0.000 | +0.273 | -0.338 | -0.106 | +0.008 | +0.050 | · | · | · | · | · |
| layer3.2 | +0.830 | +0.609 | +0.000 | +0.484 | -0.052 | -0.036 | +0.131 | +0.000 | · | · | · | · | · |
| penult | +0.830 | +0.609 | +0.000 | +0.484 | -0.052 | -0.036 | +0.131 | +0.000 | · | · | · | · | · |

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
| colorfulness | layer1.0 | layer1.0 |
| edge_density | layer1.0 | stem |
| orientation_entropy | layer2.0 | layer2.8 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.5 |
| coarse_animal_vehicle | layer3.0 | layer3.0 |
| corruption_family | layer1.2 | layer1.8 |
| corruption_type | layer2.0 | layer2.0 |
| severity | layer2.2 | layer3.0 |