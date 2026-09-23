# DEFORMATION  A=`results/atlas_v1_resnet20_s4_st3`  B=`results/atlas_v1_resnet56_e50`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.011 | 0.992 | 0.951 | True | 0.930 | 0.040 | 0.956 | 0.917 | 0.812 |
| layer1.0 | layer1.0 | 0.062 | 0.975 | 0.951 | True | 0.878 | 0.108 | 0.883 | 0.880 | 0.531 |
| layer1.1 | layer1.5 | 0.031 | 0.960 | 0.937 | True | 0.983 | 0.038 | 0.848 | 0.905 | 0.625 |
| layer1.2 | layer1.8 | 0.009 | 0.980 | 1.000 | True | 0.969 | 0.035 | 0.943 | 0.943 | 0.656 |
| layer2.0 | layer2.0 | 0.012 | 0.975 | 0.948 | True | 0.996 | 0.021 | 0.943 | 0.972 | 0.812 |
| layer2.1 | layer2.5 | 0.008 | 0.988 | 0.924 | True | 0.992 | 0.024 | 0.942 | 0.969 | 0.766 |
| layer2.2 | layer2.8 | 0.016 | 0.986 | 0.963 | True | 0.992 | 0.024 | 0.913 | 0.951 | 0.797 |
| layer3.0 | layer3.0 | 0.014 | 0.977 | 0.921 | True | 0.950 | 0.029 | 0.897 | 0.944 | 0.797 |
| layer3.1 | layer3.5 | 0.068 | 0.871 | 0.773 | True | 0.988 | 0.036 | 0.803 | 0.848 | 0.859 |
| layer3.2 | layer3.8 | 0.005 | 0.915 | 0.594 | True | 0.950 | 0.046 | 0.903 | 0.951 | 1.000 |
| penult | penult | 0.005 | 0.915 | 0.594 | True | 0.950 | 0.046 | 0.903 | 0.951 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.925 | 0.724 | 0.114 | 0.916 | 0.484 | 0.911 | 0.724 | 0.119 | 0.910 |
| layer1.0 | 0.842 | 0.597 | 0.119 | 0.871 | 0.484 | 0.795 | 0.570 | 0.123 | 0.849 |
| layer1.1 | 0.879 | 0.640 | 0.115 | 0.893 | 0.484 | 0.847 | 0.635 | 0.117 | 0.878 |
| layer1.2 | 0.949 | 0.770 | 0.111 | 0.949 | 0.484 | 0.931 | 0.753 | 0.121 | 0.941 |
| layer2.0 | 0.945 | 0.786 | 0.108 | 0.960 | 0.484 | 0.924 | 0.753 | 0.117 | 0.951 |
| layer2.1 | 0.943 | 0.773 | 0.105 | 0.958 | 0.484 | 0.925 | 0.719 | 0.113 | 0.948 |
| layer2.2 | 0.919 | 0.746 | 0.104 | 0.940 | 0.484 | 0.896 | 0.700 | 0.114 | 0.923 |
| layer3.0 | 0.887 | 0.778 | 0.102 | 0.936 | 0.484 | 0.864 | 0.688 | 0.113 | 0.917 |
| layer3.1 | 0.744 | 0.837 | 0.101 | 0.769 | 0.484 | 0.714 | 0.567 | 0.133 | 0.734 |
| layer3.2 | 0.880 | 0.929 | 0.100 | 0.761 | 0.484 | 0.670 | 0.576 | 0.122 | 0.742 |
| penult | 0.880 | 0.929 | 0.100 | 0.761 | 0.484 | 0.670 | 0.576 | 0.122 | 0.742 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | -0.236 | -0.179 | +0.000 | -0.003 | +143.141 | +0.010 | +0.046 | -0.000 | · | · | · | · | · |
| layer1.0 | -2.024 | -1.053 | -2.000 | -0.019 | +413.916 | +0.144 | -0.375 | -0.041 | · | · | · | · | · |
| layer1.1 | -1.276 | -1.026 | -1.000 | -0.004 | +27.397 | +0.115 | -0.166 | -0.012 | · | · | · | · | · |
| layer1.2 | -0.715 | -0.140 | -1.000 | -0.006 | +32.700 | +0.052 | -0.258 | -0.009 | · | · | · | · | · |
| layer2.0 | -1.315 | -0.797 | -3.000 | -0.015 | +10.773 | +0.012 | -0.287 | -0.021 | · | · | · | · | · |
| layer2.1 | -0.565 | -1.078 | -2.000 | +0.052 | -2.837 | +0.002 | -0.261 | +0.006 | · | · | · | · | · |
| layer2.2 | -0.626 | -1.393 | -3.000 | +0.046 | -0.977 | +0.024 | -0.224 | -0.011 | · | · | · | · | · |
| layer3.0 | -3.165 | -1.586 | -15.000 | -0.059 | +1.626 | +0.041 | -0.370 | -0.055 | · | · | · | · | · |
| layer3.1 | -0.477 | +4.644 | -1.000 | +0.018 | -0.008 | -0.056 | -0.134 | -0.000 | · | · | · | · | · |
| layer3.2 | +1.128 | +0.385 | +0.000 | +0.115 | -0.023 | -0.005 | +0.222 | -0.007 | · | · | · | · | · |
| penult | +1.128 | +0.385 | +0.000 | +0.115 | -0.023 | -0.005 | +0.222 | -0.007 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | layer2.0 |
| highfreq_ratio | layer2.0 | layer2.0 |
| spectral_slope | layer2.0 | layer2.0 |
| spectral_anisotropy | layer2.0 | layer2.0 |
| noise_sigma | layer1.0 | layer1.5 |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | stem | layer1.5 |
| edge_density | layer1.0 | layer1.5 |
| orientation_entropy | layer2.0 | layer3.0 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.5 |
| coarse_animal_vehicle | layer3.0 | layer3.5 |
| corruption_family | layer2.0 | layer2.0 |
| corruption_type | layer2.0 | layer2.0 |
| severity | layer2.0 | layer3.0 |