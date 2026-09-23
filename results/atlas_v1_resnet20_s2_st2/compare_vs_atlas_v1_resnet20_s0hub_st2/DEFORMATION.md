# DEFORMATION  A=`results/atlas_v1_resnet20_s0hub_st2`  B=`results/atlas_v1_resnet20_s2_st2`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.018 | 0.990 | 0.995 | True | 0.957 | 0.023 | 0.963 | 0.935 | 0.703 |
| layer1.0 | layer1.0 | 0.011 | 0.987 | 0.995 | True | 0.988 | 0.037 | 0.909 | 0.930 | 0.688 |
| layer1.1 | layer1.1 | 0.040 | 0.917 | 0.684 | True | 0.983 | 0.033 | 0.839 | 0.829 | 0.656 |
| layer1.2 | layer1.2 | 0.011 | 0.979 | 0.991 | True | 0.996 | 0.024 | 0.941 | 0.943 | 0.766 |
| layer2.0 | layer2.0 | 0.020 | 0.963 | 0.929 | True | 0.992 | 0.015 | 0.936 | 0.916 | 0.812 |
| layer2.1 | layer2.1 | 0.015 | 0.972 | 0.916 | True | 0.990 | 0.015 | 0.931 | 0.925 | 0.734 |
| layer2.2 | layer2.2 | 0.009 | 0.992 | 0.948 | True | 0.990 | 0.013 | 0.925 | 0.940 | 0.719 |
| layer3.0 | layer3.0 | 0.011 | 0.978 | 0.991 | True | 0.977 | 0.017 | 0.909 | 0.964 | 0.906 |
| layer3.1 | layer3.1 | 0.022 | 0.964 | 0.956 | True | 0.975 | 0.022 | 0.885 | 0.944 | 0.875 |
| layer3.2 | layer3.2 | 0.001 | 0.955 | 0.939 | True | 0.955 | 0.020 | 0.919 | 0.968 | 1.000 |
| penult | penult | 0.001 | 0.955 | 0.939 | True | 0.955 | 0.020 | 0.919 | 0.968 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.920 | 0.720 | 0.111 | 0.931 | 0.528 | 0.909 | 0.745 | 0.118 | 0.925 |
| layer1.0 | 0.894 | 0.664 | 0.120 | 0.914 | 0.528 | 0.832 | 0.663 | 0.125 | 0.900 |
| layer1.1 | 0.857 | 0.655 | 0.113 | 0.837 | 0.528 | 0.794 | 0.623 | 0.113 | 0.820 |
| layer1.2 | 0.942 | 0.736 | 0.114 | 0.927 | 0.528 | 0.921 | 0.700 | 0.126 | 0.917 |
| layer2.0 | 0.933 | 0.725 | 0.106 | 0.907 | 0.528 | 0.919 | 0.686 | 0.115 | 0.886 |
| layer2.1 | 0.927 | 0.741 | 0.104 | 0.922 | 0.528 | 0.910 | 0.679 | 0.111 | 0.906 |
| layer2.2 | 0.917 | 0.745 | 0.103 | 0.926 | 0.528 | 0.893 | 0.665 | 0.107 | 0.910 |
| layer3.0 | 0.897 | 0.820 | 0.102 | 0.935 | 0.528 | 0.874 | 0.709 | 0.112 | 0.925 |
| layer3.1 | 0.867 | 0.844 | 0.101 | 0.899 | 0.528 | 0.773 | 0.619 | 0.137 | 0.889 |
| layer3.2 | 0.890 | 0.920 | 0.100 | 0.834 | 0.528 | 0.686 | 0.585 | 0.125 | 0.772 |
| penult | 0.890 | 0.920 | 0.100 | 0.834 | 0.528 | 0.686 | 0.585 | 0.125 | 0.772 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | +0.152 | -0.004 | +0.000 | +0.005 | -107.002 | +0.031 | -0.003 | +0.003 | · | · | · | · | · |
| layer1.0 | +0.371 | -0.167 | +1.000 | +0.007 | -2.174 | +0.032 | +0.110 | -0.000 | · | · | · | · | · |
| layer1.1 | -0.100 | -0.104 | +0.000 | -0.001 | +10.487 | +0.081 | -0.028 | -0.011 | · | · | · | · | · |
| layer1.2 | -0.108 | -1.019 | +0.000 | +0.011 | -5.404 | -0.012 | -0.050 | -0.010 | · | · | · | · | · |
| layer2.0 | -0.664 | -1.218 | -2.000 | +0.010 | -2.476 | -0.008 | -0.049 | -0.013 | · | · | · | · | · |
| layer2.1 | -0.340 | -0.667 | -1.000 | -0.006 | -1.307 | -0.040 | -0.086 | -0.023 | · | · | · | · | · |
| layer2.2 | -0.249 | -1.056 | -1.000 | -0.012 | +0.292 | -0.013 | +0.200 | -0.016 | · | · | · | · | · |
| layer3.0 | +0.183 | -0.045 | +1.000 | +0.003 | +0.117 | +0.026 | -0.128 | +0.017 | · | · | · | · | · |
| layer3.1 | +0.230 | -0.826 | -1.000 | -0.089 | +0.100 | +0.015 | -0.102 | +0.007 | · | · | · | · | · |
| layer3.2 | +0.121 | -0.002 | +0.000 | +0.025 | -0.000 | +0.000 | +0.003 | +0.002 | · | · | · | · | · |
| penult | +0.121 | -0.002 | +0.000 | +0.025 | -0.000 | +0.000 | +0.003 | +0.002 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer1.0 | layer2.0 |
| spectral_slope | layer2.0 | layer2.0 |
| spectral_anisotropy | layer2.0 | layer2.0 |
| noise_sigma | layer1.0 | layer1.0 |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | layer1.0 | layer1.0 |
| edge_density | layer1.0 | layer1.0 |
| orientation_entropy | layer2.0 | layer2.0 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.1 |
| coarse_animal_vehicle | layer3.0 | layer3.0 |
| corruption_family | layer2.0 | layer1.2 |
| corruption_type | layer1.2 | layer2.0 |
| severity | layer3.0 | layer2.2 |