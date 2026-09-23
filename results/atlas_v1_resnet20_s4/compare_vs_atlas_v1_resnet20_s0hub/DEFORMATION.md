# DEFORMATION  A=`results/atlas_v1_resnet20_s0hub`  B=`results/atlas_v1_resnet20_s4`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.006 | 0.995 | 0.992 | True | 0.986 | 0.019 | 0.973 | 0.986 | 0.828 |
| layer1.0 | layer1.0 | 0.015 | 0.986 | 0.995 | True | 0.996 | 0.037 | 0.910 | 0.932 | 0.672 |
| layer1.1 | layer1.1 | 0.019 | 0.969 | 0.942 | True | 0.992 | 0.026 | 0.907 | 0.923 | 0.703 |
| layer1.2 | layer1.2 | 0.009 | 0.987 | 0.991 | True | 0.988 | 0.027 | 0.956 | 0.970 | 0.797 |
| layer2.0 | layer2.0 | 0.010 | 0.979 | 0.951 | True | 0.983 | 0.016 | 0.945 | 0.936 | 0.828 |
| layer2.1 | layer2.1 | 0.013 | 0.978 | 0.930 | True | 0.992 | 0.014 | 0.938 | 0.939 | 0.781 |
| layer2.2 | layer2.2 | 0.011 | 0.984 | 0.946 | True | 0.992 | 0.015 | 0.928 | 0.947 | 0.766 |
| layer3.0 | layer3.0 | 0.007 | 0.983 | 0.932 | True | 0.963 | 0.020 | 0.909 | 0.958 | 0.844 |
| layer3.1 | layer3.1 | 0.023 | 0.969 | 0.758 | True | 0.981 | 0.025 | 0.891 | 0.938 | 0.812 |
| layer3.2 | layer3.2 | 0.002 | 0.933 | 0.871 | True | 0.963 | 0.023 | 0.920 | 0.965 | 1.000 |
| penult | penult | 0.002 | 0.933 | 0.871 | True | 0.963 | 0.023 | 0.920 | 0.965 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.971 | 0.837 | 0.117 | 0.980 | 0.565 | 0.958 | 0.837 | 0.120 | 0.976 |
| layer1.0 | 0.899 | 0.690 | 0.121 | 0.940 | 0.565 | 0.865 | 0.678 | 0.125 | 0.928 |
| layer1.1 | 0.889 | 0.693 | 0.114 | 0.914 | 0.565 | 0.841 | 0.667 | 0.116 | 0.902 |
| layer1.2 | 0.955 | 0.785 | 0.112 | 0.961 | 0.565 | 0.935 | 0.767 | 0.123 | 0.956 |
| layer2.0 | 0.950 | 0.766 | 0.107 | 0.935 | 0.565 | 0.932 | 0.722 | 0.117 | 0.918 |
| layer2.1 | 0.936 | 0.748 | 0.105 | 0.935 | 0.565 | 0.918 | 0.686 | 0.112 | 0.920 |
| layer2.2 | 0.925 | 0.753 | 0.102 | 0.939 | 0.565 | 0.905 | 0.677 | 0.110 | 0.923 |
| layer3.0 | 0.895 | 0.810 | 0.102 | 0.939 | 0.565 | 0.866 | 0.715 | 0.115 | 0.927 |
| layer3.1 | 0.857 | 0.852 | 0.101 | 0.877 | 0.565 | 0.766 | 0.622 | 0.146 | 0.861 |
| layer3.2 | 0.888 | 0.932 | 0.100 | 0.826 | 0.565 | 0.687 | 0.578 | 0.125 | 0.770 |
| penult | 0.888 | 0.932 | 0.100 | 0.826 | 0.565 | 0.687 | 0.578 | 0.125 | 0.770 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | +0.015 | +0.112 | +0.000 | +0.007 | -100.745 | +0.021 | -0.010 | +0.007 | · | · | · | · | · |
| layer1.0 | +0.421 | -0.092 | +1.000 | +0.019 | -44.462 | -0.062 | +0.109 | +0.007 | · | · | · | · | · |
| layer1.1 | -0.018 | -0.278 | +0.000 | +0.005 | +28.871 | -0.035 | +0.003 | -0.012 | · | · | · | · | · |
| layer1.2 | +0.100 | -0.416 | +0.000 | +0.006 | -4.856 | -0.032 | +0.051 | -0.004 | · | · | · | · | · |
| layer2.0 | -0.037 | -0.671 | -1.000 | +0.009 | -1.012 | +0.000 | +0.046 | -0.008 | · | · | · | · | · |
| layer2.1 | +0.145 | -0.287 | +0.000 | -0.013 | +2.225 | -0.018 | +0.065 | -0.020 | · | · | · | · | · |
| layer2.2 | +0.023 | -0.410 | +0.000 | -0.014 | +1.760 | -0.021 | +0.180 | -0.008 | · | · | · | · | · |
| layer3.0 | +0.207 | -1.702 | +0.000 | -0.000 | -0.020 | +0.030 | +0.045 | -0.004 | · | · | · | · | · |
| layer3.1 | +0.149 | -1.015 | +0.000 | -0.066 | +0.033 | +0.021 | -0.047 | +0.020 | · | · | · | · | · |
| layer3.2 | -0.014 | +0.069 | +0.000 | +0.028 | -0.002 | -0.009 | +0.040 | +0.005 | · | · | · | · | · |
| penult | -0.014 | +0.069 | +0.000 | +0.028 | -0.002 | -0.009 | +0.040 | +0.005 | · | · | · | · | · |

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
| colorfulness | layer1.0 | stem |
| edge_density | layer1.0 | layer1.0 |
| orientation_entropy | layer2.0 | layer2.0 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.1 |
| coarse_animal_vehicle | layer3.0 | layer3.0 |
| corruption_family | layer2.0 | layer2.0 |
| corruption_type | layer1.2 | layer2.0 |
| severity | layer3.0 | layer2.0 |