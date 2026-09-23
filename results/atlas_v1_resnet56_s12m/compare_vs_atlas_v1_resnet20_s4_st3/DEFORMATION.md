# DEFORMATION  A=`results/atlas_v1_resnet20_s4_st3`  B=`results/atlas_v1_resnet56_s12m`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.011 | 0.994 | 0.951 | True | 0.975 | 0.029 | 0.984 | 0.936 | 0.797 |
| layer1.0 | layer1.0 | 0.046 | 0.965 | 0.995 | True | 0.932 | 0.060 | 0.791 | 0.868 | 0.547 |
| layer1.1 | layer1.5 | 0.009 | 0.985 | 1.000 | True | 0.983 | 0.032 | 0.907 | 0.940 | 0.703 |
| layer1.2 | layer1.8 | 0.006 | 0.994 | 1.000 | True | 0.975 | 0.023 | 0.958 | 0.967 | 0.766 |
| layer2.0 | layer2.0 | 0.023 | 0.963 | 0.955 | True | 0.990 | 0.014 | 0.920 | 0.940 | 0.781 |
| layer2.1 | layer2.5 | 0.013 | 0.982 | 0.948 | True | 0.979 | 0.027 | 0.937 | 0.950 | 0.750 |
| layer2.2 | layer2.8 | 0.010 | 0.989 | 0.983 | True | 0.979 | 0.025 | 0.926 | 0.968 | 0.922 |
| layer3.0 | layer3.0 | 0.016 | 0.976 | 0.931 | True | 0.973 | 0.027 | 0.912 | 0.945 | 0.781 |
| layer3.1 | layer3.5 | 0.074 | 0.796 | 0.817 | True | 0.969 | 0.033 | 0.790 | 0.845 | 0.875 |
| layer3.2 | layer3.8 | 0.006 | 0.915 | 0.814 | True | 0.936 | 0.045 | 0.891 | 0.955 | 1.000 |
| penult | penult | 0.006 | 0.915 | 0.814 | True | 0.936 | 0.045 | 0.891 | 0.955 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.951 | 0.798 | 0.115 | 0.929 | 0.492 | 0.943 | 0.787 | 0.118 | 0.922 |
| layer1.0 | 0.818 | 0.606 | 0.116 | 0.851 | 0.492 | 0.729 | 0.556 | 0.120 | 0.826 |
| layer1.1 | 0.918 | 0.716 | 0.114 | 0.926 | 0.492 | 0.873 | 0.697 | 0.115 | 0.910 |
| layer1.2 | 0.956 | 0.770 | 0.111 | 0.959 | 0.492 | 0.937 | 0.748 | 0.122 | 0.948 |
| layer2.0 | 0.914 | 0.773 | 0.107 | 0.922 | 0.492 | 0.891 | 0.730 | 0.116 | 0.899 |
| layer2.1 | 0.936 | 0.781 | 0.104 | 0.947 | 0.492 | 0.916 | 0.716 | 0.112 | 0.931 |
| layer2.2 | 0.930 | 0.794 | 0.103 | 0.958 | 0.492 | 0.903 | 0.735 | 0.113 | 0.943 |
| layer3.0 | 0.896 | 0.802 | 0.102 | 0.929 | 0.492 | 0.872 | 0.694 | 0.112 | 0.917 |
| layer3.1 | 0.717 | 0.829 | 0.101 | 0.739 | 0.492 | 0.707 | 0.561 | 0.128 | 0.726 |
| layer3.2 | 0.877 | 0.926 | 0.100 | 0.765 | 0.492 | 0.664 | 0.575 | 0.121 | 0.734 |
| penult | 0.877 | 0.926 | 0.100 | 0.765 | 0.492 | 0.664 | 0.575 | 0.121 | 0.734 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | -0.514 | -0.217 | +0.000 | +0.002 | +106.167 | +0.012 | -0.127 | -0.007 | · | · | · | · | · |
| layer1.0 | -1.436 | -1.068 | -2.000 | -0.028 | +24.938 | +0.164 | -0.345 | -0.037 | · | · | · | · | · |
| layer1.1 | -0.387 | +0.233 | +0.000 | +0.005 | -10.985 | +0.015 | -0.081 | +0.011 | · | · | · | · | · |
| layer1.2 | -0.497 | -0.100 | +0.000 | +0.010 | +18.620 | -0.002 | -0.091 | -0.001 | · | · | · | · | · |
| layer2.0 | -1.195 | -0.584 | -3.000 | +0.007 | +8.892 | -0.005 | -0.210 | -0.018 | · | · | · | · | · |
| layer2.1 | -0.139 | -0.210 | -1.000 | +0.059 | -2.433 | -0.020 | -0.089 | +0.034 | · | · | · | · | · |
| layer2.2 | -0.307 | -0.855 | -2.000 | +0.058 | +0.044 | +0.012 | -0.162 | +0.018 | · | · | · | · | · |
| layer3.0 | -2.689 | -1.730 | -13.000 | -0.013 | +1.098 | +0.013 | -0.444 | -0.022 | · | · | · | · | · |
| layer3.1 | -0.597 | +3.372 | +1.000 | -0.034 | +0.132 | -0.021 | +0.012 | -0.026 | · | · | · | · | · |
| layer3.2 | +0.820 | +0.364 | +0.000 | +0.343 | -0.039 | -0.012 | +0.075 | -0.005 | · | · | · | · | · |
| penult | +0.820 | +0.364 | +0.000 | +0.343 | -0.039 | -0.012 | +0.075 | -0.005 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer2.0 | layer2.0 |
| spectral_slope | layer2.0 | layer2.0 |
| spectral_anisotropy | layer2.0 | layer2.5 |
| noise_sigma | layer1.0 | layer1.0 |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | stem | layer1.0 |
| edge_density | layer1.0 | layer2.0 |
| orientation_entropy | layer2.0 | layer2.5 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.5 |
| coarse_animal_vehicle | layer3.0 | layer3.0 |
| corruption_family | layer2.0 | layer2.0 |
| corruption_type | layer2.0 | layer2.0 |
| severity | layer2.0 | layer2.0 |