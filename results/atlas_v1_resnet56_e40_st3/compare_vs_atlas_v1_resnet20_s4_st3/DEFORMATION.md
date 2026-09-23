# DEFORMATION  A=`results/atlas_v1_resnet20_s4_st3`  B=`results/atlas_v1_resnet56_e40_st3`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.008 | 0.995 | 0.997 | True | 0.965 | 0.027 | 0.982 | 0.952 | 0.797 |
| layer1.0 | layer1.0 | 0.024 | 0.981 | 0.992 | True | 0.963 | 0.041 | 0.897 | 0.938 | 0.766 |
| layer1.1 | layer1.5 | 0.065 | 0.920 | 0.929 | True | 0.983 | 0.036 | 0.808 | 0.873 | 0.625 |
| layer1.2 | layer1.8 | 0.028 | 0.934 | 0.938 | True | 0.977 | 0.027 | 0.912 | 0.920 | 0.719 |
| layer2.0 | layer2.0 | 0.006 | 0.989 | 1.000 | True | 0.996 | 0.029 | 0.945 | 0.982 | 0.922 |
| layer2.1 | layer2.5 | 0.012 | 0.985 | 0.926 | True | 0.992 | 0.032 | 0.930 | 0.960 | 0.812 |
| layer2.2 | layer2.8 | 0.011 | 0.978 | 0.532 | True | 0.994 | 0.026 | 0.924 | 0.956 | 0.734 |
| layer3.0 | layer3.0 | 0.022 | 0.965 | 0.977 | True | 0.986 | 0.022 | 0.890 | 0.940 | 0.766 |
| layer3.1 | layer3.5 | 0.072 | 0.861 | 0.787 | True | 0.988 | 0.027 | 0.798 | 0.853 | 0.797 |
| layer3.2 | layer3.8 | 0.004 | 0.943 | 0.934 | True | 0.953 | 0.038 | 0.909 | 0.959 | 1.000 |
| penult | penult | 0.004 | 0.943 | 0.934 | True | 0.953 | 0.038 | 0.909 | 0.959 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.968 | 0.819 | 0.116 | 0.958 | 0.505 | 0.970 | 0.810 | 0.120 | 0.957 |
| layer1.0 | 0.925 | 0.733 | 0.122 | 0.934 | 0.505 | 0.889 | 0.706 | 0.125 | 0.922 |
| layer1.1 | 0.820 | 0.626 | 0.114 | 0.848 | 0.505 | 0.771 | 0.576 | 0.115 | 0.820 |
| layer1.2 | 0.927 | 0.740 | 0.108 | 0.925 | 0.505 | 0.922 | 0.728 | 0.118 | 0.926 |
| layer2.0 | 0.953 | 0.791 | 0.107 | 0.972 | 0.505 | 0.936 | 0.747 | 0.116 | 0.963 |
| layer2.1 | 0.931 | 0.776 | 0.105 | 0.942 | 0.505 | 0.916 | 0.729 | 0.115 | 0.939 |
| layer2.2 | 0.930 | 0.790 | 0.104 | 0.944 | 0.505 | 0.915 | 0.734 | 0.115 | 0.939 |
| layer3.0 | 0.886 | 0.783 | 0.102 | 0.930 | 0.505 | 0.881 | 0.700 | 0.114 | 0.916 |
| layer3.1 | 0.737 | 0.845 | 0.101 | 0.751 | 0.505 | 0.700 | 0.583 | 0.133 | 0.727 |
| layer3.2 | 0.879 | 0.925 | 0.100 | 0.810 | 0.505 | 0.671 | 0.559 | 0.123 | 0.757 |
| penult | 0.879 | 0.925 | 0.100 | 0.810 | 0.505 | 0.671 | 0.559 | 0.123 | 0.757 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | +0.320 | -0.137 | +1.000 | +0.003 | -11.231 | +0.004 | +0.062 | +0.002 | · | · | · | · | · |
| layer1.0 | -0.844 | -0.303 | -1.000 | -0.010 | +5.946 | +0.085 | -0.286 | -0.006 | · | · | · | · | · |
| layer1.1 | -0.438 | -0.638 | +0.000 | +0.069 | +16.447 | +0.065 | -0.123 | +0.026 | · | · | · | · | · |
| layer1.2 | -0.549 | -0.058 | -1.000 | +0.017 | +24.482 | +0.053 | -0.188 | +0.029 | · | · | · | · | · |
| layer2.0 | -1.522 | -0.827 | -2.000 | +0.020 | +8.237 | +0.007 | -0.180 | +0.004 | · | · | · | · | · |
| layer2.1 | -1.213 | -1.413 | -2.000 | +0.061 | -2.874 | +0.006 | -0.208 | +0.025 | · | · | · | · | · |
| layer2.2 | -1.511 | -0.669 | -3.000 | +0.050 | -0.137 | -0.024 | -0.179 | +0.009 | · | · | · | · | · |
| layer3.0 | -3.520 | -2.015 | -15.000 | -0.038 | +1.492 | -0.013 | -0.476 | -0.054 | · | · | · | · | · |
| layer3.1 | +0.007 | +4.299 | +0.000 | +0.056 | -0.042 | -0.075 | -0.047 | -0.000 | · | · | · | · | · |
| layer3.2 | +0.984 | +0.134 | +0.000 | -0.171 | +0.010 | +0.009 | +0.154 | -0.012 | · | · | · | · | · |
| penult | +0.984 | +0.134 | +0.000 | -0.171 | +0.010 | +0.009 | +0.154 | -0.012 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer2.0 | layer1.5 |
| spectral_slope | layer2.0 | layer1.5 |
| spectral_anisotropy | layer2.0 | layer2.0 |
| noise_sigma | layer1.0 | layer1.0 |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | stem | layer1.0 |
| edge_density | layer1.0 | stem |
| orientation_entropy | layer2.0 | layer2.0 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.5 |
| coarse_animal_vehicle | layer3.0 | layer3.0 |
| corruption_family | layer2.0 | layer1.5 |
| corruption_type | layer2.0 | layer1.8 |
| severity | layer2.0 | layer3.0 |