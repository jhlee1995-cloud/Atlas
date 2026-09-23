# DEFORMATION  A=`results/atlas_v1_resnet56_s0hub_st3`  B=`results/atlas_v1_resnet56_s0hub_ref1`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.006 | 0.987 | 0.962 | True | 0.977 | 0.011 | 0.032 | 0.220 | 0.094 |
| layer1.0 | layer1.0 | 0.005 | 0.991 | 1.000 | True | 0.998 | 0.011 | 0.041 | 0.224 | 0.109 |
| layer1.5 | layer1.5 | 0.003 | 0.989 | 0.997 | True | 0.988 | 0.009 | 0.065 | 0.221 | 0.156 |
| layer1.8 | layer1.8 | 0.003 | 0.989 | 0.960 | True | 0.998 | 0.010 | 0.072 | 0.254 | 0.094 |
| layer2.0 | layer2.0 | 0.003 | 0.987 | 0.651 | True | 0.996 | 0.009 | 0.073 | 0.166 | 0.094 |
| layer2.5 | layer2.5 | 0.002 | 0.992 | 0.971 | True | 0.998 | 0.008 | 0.097 | 0.090 | 0.094 |
| layer2.8 | layer2.8 | 0.002 | 0.994 | 0.991 | True | 0.994 | 0.010 | 0.119 | 0.062 | 0.094 |
| layer3.0 | layer3.0 | 0.001 | 0.991 | 0.964 | True | 0.992 | 0.010 | 0.116 | 0.080 | 0.125 |
| layer3.5 | layer3.5 | 0.000 | 0.994 | 0.904 | True | 0.983 | 0.011 | 0.193 | 0.047 | 0.078 |
| layer3.8 | layer3.8 | 0.000 | 0.993 | 1.000 | True | 0.977 | 0.011 | 0.108 | -0.057 | 0.047 |
| penult | penult | 0.000 | 0.993 | 1.000 | True | 0.977 | 0.011 | 0.108 | -0.057 | 0.047 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 1.000 | 0.856 | 0.113 | 0.992 | 1.000 | 1.000 | 0.859 | 0.117 | 0.992 |
| layer1.0 | 1.000 | 0.873 | 0.113 | 0.993 | 1.000 | 1.000 | 0.870 | 0.117 | 0.993 |
| layer1.5 | 1.000 | 0.929 | 0.117 | 0.996 | 1.000 | 1.000 | 0.914 | 0.121 | 0.996 |
| layer1.8 | 1.000 | 0.940 | 0.110 | 0.997 | 1.000 | 1.000 | 0.940 | 0.121 | 0.997 |
| layer2.0 | 1.000 | 0.942 | 0.105 | 0.997 | 1.000 | 1.000 | 0.928 | 0.113 | 0.997 |
| layer2.5 | 1.000 | 0.940 | 0.103 | 0.997 | 1.000 | 1.000 | 0.941 | 0.110 | 0.997 |
| layer2.8 | 1.000 | 0.949 | 0.104 | 0.997 | 1.000 | 1.000 | 0.937 | 0.109 | 0.997 |
| layer3.0 | 1.000 | 0.959 | 0.103 | 0.998 | 1.000 | 1.000 | 0.949 | 0.110 | 0.998 |
| layer3.5 | 1.000 | 0.985 | 0.101 | 0.998 | 1.000 | 1.000 | 0.964 | 0.140 | 0.998 |
| layer3.8 | 1.000 | 1.000 | 0.100 | 0.999 | 1.000 | 1.000 | 0.996 | 0.122 | 1.000 |
| penult | 1.000 | 1.000 | 0.100 | 0.999 | 1.000 | 1.000 | 0.996 | 0.122 | 1.000 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | -0.135 | +0.003 | +0.000 | +0.028 | -250.426 | -0.001 | +0.016 | -0.002 | · | · | · | · | · |
| layer1.0 | -0.348 | +0.012 | +0.000 | +0.024 | +263.746 | +0.004 | -0.047 | +0.000 | · | · | · | · | · |
| layer1.5 | +0.159 | +0.028 | +0.000 | +0.011 | +21.261 | -0.011 | +0.041 | +0.000 | · | · | · | · | · |
| layer1.8 | +0.010 | -0.047 | +0.000 | +0.013 | +4.018 | -0.010 | +0.031 | -0.001 | · | · | · | · | · |
| layer2.0 | -0.050 | +0.010 | +0.000 | +0.012 | -1.354 | -0.017 | -0.022 | -0.001 | · | · | · | · | · |
| layer2.5 | -0.033 | -0.009 | +0.000 | +0.010 | -0.801 | -0.016 | -0.003 | -0.003 | · | · | · | · | · |
| layer2.8 | +0.059 | -0.006 | +0.000 | +0.007 | -0.387 | -0.013 | +0.006 | -0.002 | · | · | · | · | · |
| layer3.0 | -0.032 | -0.004 | +0.000 | +0.007 | -0.167 | -0.014 | +0.005 | -0.003 | · | · | · | · | · |
| layer3.5 | -0.151 | -0.040 | -1.000 | +0.005 | -0.023 | -0.004 | +0.049 | -0.001 | · | · | · | · | · |
| layer3.8 | -0.113 | -0.013 | +0.000 | +0.051 | -0.001 | +0.000 | +0.027 | +0.000 | · | · | · | · | · |
| penult | -0.113 | -0.013 | +0.000 | +0.051 | -0.001 | +0.000 | +0.027 | +0.000 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer2.0 | layer2.0 |
| spectral_slope | layer2.0 | layer2.0 |
| spectral_anisotropy | layer3.0 | layer3.0 |
| noise_sigma | layer1.0 | layer1.0 |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | layer1.0 | layer1.0 |
| edge_density | layer1.0 | layer1.0 |
| orientation_entropy | layer2.0 | layer2.0 |
| blockiness | layer3.0 | layer2.0 |
| class | layer3.5 | layer3.5 |
| coarse_animal_vehicle | layer3.5 | layer3.0 |
| corruption_family | layer2.0 | layer2.0 |
| corruption_type | layer2.0 | layer2.0 |
| severity | layer3.0 | layer3.0 |