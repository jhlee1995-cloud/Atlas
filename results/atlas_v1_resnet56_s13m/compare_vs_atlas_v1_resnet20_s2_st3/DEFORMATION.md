# DEFORMATION  A=`results/atlas_v1_resnet20_s2_st3`  B=`results/atlas_v1_resnet56_s13m`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.015 | 0.991 | 0.992 | True | 0.975 | 0.037 | 0.879 | 0.900 | 0.688 |
| layer1.0 | layer1.0 | 0.035 | 0.959 | 0.997 | True | 0.913 | 0.059 | 0.729 | 0.916 | 0.516 |
| layer1.1 | layer1.5 | 0.031 | 0.944 | 0.679 | True | 0.990 | 0.035 | 0.833 | 0.899 | 0.672 |
| layer1.2 | layer1.8 | 0.015 | 0.960 | 0.989 | True | 0.990 | 0.015 | 0.930 | 0.953 | 0.719 |
| layer2.0 | layer2.0 | 0.004 | 0.989 | 0.950 | True | 0.996 | 0.015 | 0.959 | 0.980 | 0.812 |
| layer2.1 | layer2.5 | 0.007 | 0.990 | 0.529 | True | 0.977 | 0.026 | 0.948 | 0.966 | 0.812 |
| layer2.2 | layer2.8 | 0.012 | 0.984 | 0.541 | True | 0.973 | 0.027 | 0.935 | 0.947 | 0.781 |
| layer3.0 | layer3.0 | 0.020 | 0.963 | 0.909 | True | 0.950 | 0.030 | 0.897 | 0.941 | 0.797 |
| layer3.1 | layer3.5 | 0.092 | 0.820 | 0.785 | True | 0.983 | 0.032 | 0.743 | 0.821 | 0.828 |
| layer3.2 | layer3.8 | 0.005 | 0.945 | 0.577 | True | 0.959 | 0.027 | 0.915 | 0.959 | 1.000 |
| penult | penult | 0.005 | 0.945 | 0.577 | True | 0.959 | 0.027 | 0.915 | 0.959 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.884 | 0.654 | 0.112 | 0.887 | 0.554 | 0.868 | 0.679 | 0.118 | 0.873 |
| layer1.0 | 0.831 | 0.615 | 0.117 | 0.904 | 0.554 | 0.749 | 0.611 | 0.120 | 0.887 |
| layer1.1 | 0.833 | 0.673 | 0.117 | 0.877 | 0.554 | 0.795 | 0.636 | 0.116 | 0.843 |
| layer1.2 | 0.929 | 0.733 | 0.113 | 0.926 | 0.554 | 0.918 | 0.715 | 0.126 | 0.917 |
| layer2.0 | 0.961 | 0.839 | 0.106 | 0.973 | 0.554 | 0.947 | 0.802 | 0.116 | 0.963 |
| layer2.1 | 0.948 | 0.786 | 0.105 | 0.960 | 0.554 | 0.936 | 0.743 | 0.110 | 0.953 |
| layer2.2 | 0.932 | 0.786 | 0.104 | 0.947 | 0.554 | 0.914 | 0.730 | 0.110 | 0.939 |
| layer3.0 | 0.886 | 0.780 | 0.102 | 0.919 | 0.554 | 0.876 | 0.678 | 0.110 | 0.897 |
| layer3.1 | 0.698 | 0.816 | 0.101 | 0.708 | 0.554 | 0.693 | 0.600 | 0.128 | 0.682 |
| layer3.2 | 0.879 | 0.929 | 0.100 | 0.780 | 0.554 | 0.681 | 0.575 | 0.124 | 0.751 |
| penult | 0.879 | 0.929 | 0.100 | 0.780 | 0.554 | 0.681 | 0.575 | 0.124 | 0.751 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | -1.259 | -0.022 | -1.000 | +0.007 | +123.904 | -0.020 | -0.203 | -0.009 | · | · | · | · | · |
| layer1.0 | -2.697 | -0.732 | -3.000 | -0.018 | +73.402 | +0.011 | -0.622 | -0.043 | · | · | · | · | · |
| layer1.1 | -0.984 | -0.942 | -1.000 | +0.026 | +63.380 | -0.003 | -0.167 | -0.005 | · | · | · | · | · |
| layer1.2 | -0.212 | +0.661 | +0.000 | +0.008 | +11.994 | +0.017 | -0.102 | +0.024 | · | · | · | · | · |
| layer2.0 | -0.760 | +0.153 | -1.000 | -0.001 | +9.635 | +0.001 | -0.149 | +0.003 | · | · | · | · | · |
| layer2.1 | -0.229 | -0.070 | +0.000 | +0.047 | -1.251 | -0.016 | +0.106 | +0.039 | · | · | · | · | · |
| layer2.2 | -0.728 | -0.816 | -2.000 | +0.020 | +0.629 | -0.028 | -0.136 | +0.001 | · | · | · | · | · |
| layer3.0 | -3.602 | -3.239 | -18.000 | -0.047 | +1.389 | +0.018 | -0.439 | -0.067 | · | · | · | · | · |
| layer3.1 | +0.044 | +5.110 | +3.000 | -0.053 | +0.152 | -0.047 | +0.222 | -0.047 | · | · | · | · | · |
| layer3.2 | +0.691 | +0.555 | +0.000 | +0.293 | -0.040 | -0.027 | +0.141 | -0.001 | · | · | · | · | · |
| penult | +0.691 | +0.555 | +0.000 | +0.293 | -0.040 | -0.027 | +0.141 | -0.001 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | layer1.0 |
| highfreq_ratio | layer2.0 | layer2.0 |
| spectral_slope | layer2.0 | layer2.0 |
| spectral_anisotropy | layer2.0 | layer3.0 |
| noise_sigma | layer1.0 | layer1.5 |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | layer1.0 | layer1.0 |
| edge_density | layer1.0 | layer1.0 |
| orientation_entropy | layer2.0 | layer3.0 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.5 |
| coarse_animal_vehicle | layer3.0 | layer3.5 |
| corruption_family | layer1.2 | layer1.8 |
| corruption_type | layer2.0 | layer1.8 |
| severity | layer2.2 | layer3.0 |