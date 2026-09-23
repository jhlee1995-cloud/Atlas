# DEFORMATION  A=`results/atlas_v1_resnet20_s2_st2`  B=`results/atlas_v1_resnet56_s0hub`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.011 | 0.989 | 0.990 | True | 0.992 | 0.011 | 0.971 | 0.925 | 0.703 |
| layer1.0 | layer1.0 | 0.027 | 0.975 | 0.995 | True | 0.975 | 0.040 | 0.848 | 0.901 | 0.562 |
| layer1.1 | layer1.5 | 0.033 | 0.940 | 0.684 | True | 0.983 | 0.018 | 0.890 | 0.885 | 0.750 |
| layer1.2 | layer1.8 | 0.012 | 0.980 | 1.000 | True | 0.994 | 0.025 | 0.940 | 0.958 | 0.719 |
| layer2.0 | layer2.0 | 0.004 | 0.993 | 0.934 | True | 0.996 | 0.017 | 0.962 | 0.972 | 0.812 |
| layer2.1 | layer2.5 | 0.011 | 0.979 | 0.979 | True | 0.994 | 0.018 | 0.940 | 0.950 | 0.797 |
| layer2.2 | layer2.8 | 0.012 | 0.975 | 0.557 | True | 0.994 | 0.018 | 0.927 | 0.954 | 0.812 |
| layer3.0 | layer3.0 | 0.018 | 0.969 | 0.977 | True | 0.953 | 0.041 | 0.913 | 0.949 | 0.766 |
| layer3.1 | layer3.5 | 0.080 | 0.886 | 0.924 | True | 0.961 | 0.038 | 0.732 | 0.839 | 0.844 |
| layer3.2 | layer3.8 | 0.014 | 0.912 | 0.847 | True | 0.973 | 0.015 | 0.921 | 0.950 | 1.000 |
| penult | penult | 0.014 | 0.912 | 0.847 | True | 0.973 | 0.015 | 0.921 | 0.950 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.946 | 0.733 | 0.111 | 0.914 | 0.504 | 0.946 | 0.755 | 0.118 | 0.903 |
| layer1.0 | 0.857 | 0.636 | 0.114 | 0.901 | 0.504 | 0.795 | 0.599 | 0.117 | 0.876 |
| layer1.1 | 0.888 | 0.701 | 0.114 | 0.867 | 0.504 | 0.878 | 0.689 | 0.115 | 0.851 |
| layer1.2 | 0.940 | 0.740 | 0.112 | 0.941 | 0.504 | 0.930 | 0.731 | 0.124 | 0.930 |
| layer2.0 | 0.962 | 0.809 | 0.106 | 0.969 | 0.504 | 0.954 | 0.783 | 0.114 | 0.963 |
| layer2.1 | 0.939 | 0.788 | 0.104 | 0.943 | 0.504 | 0.934 | 0.756 | 0.110 | 0.936 |
| layer2.2 | 0.925 | 0.789 | 0.103 | 0.940 | 0.504 | 0.912 | 0.730 | 0.108 | 0.927 |
| layer3.0 | 0.888 | 0.801 | 0.102 | 0.921 | 0.504 | 0.878 | 0.684 | 0.110 | 0.904 |
| layer3.1 | 0.735 | 0.838 | 0.101 | 0.726 | 0.504 | 0.704 | 0.599 | 0.133 | 0.674 |
| layer3.2 | 0.864 | 0.934 | 0.100 | 0.722 | 0.504 | 0.651 | 0.586 | 0.123 | 0.723 |
| penult | 0.864 | 0.934 | 0.100 | 0.722 | 0.504 | 0.651 | 0.586 | 0.123 | 0.723 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | -0.431 | -0.033 | +0.000 | -0.005 | +341.465 | -0.008 | -0.062 | +0.002 | · | · | · | · | · |
| layer1.0 | -1.414 | -0.690 | -2.000 | -0.025 | +35.786 | +0.030 | -0.360 | -0.040 | · | · | · | · | · |
| layer1.1 | -0.391 | +0.051 | +0.000 | +0.026 | -14.851 | -0.066 | -0.007 | +0.004 | · | · | · | · | · |
| layer1.2 | +0.018 | +0.992 | +0.000 | +0.020 | +8.750 | -0.005 | +0.035 | +0.022 | · | · | · | · | · |
| layer2.0 | -0.205 | +0.643 | +0.000 | -0.005 | +4.696 | +0.006 | -0.002 | -0.009 | · | · | · | · | · |
| layer2.1 | +0.464 | +1.570 | +2.000 | +0.014 | -0.193 | -0.010 | +0.132 | +0.032 | · | · | · | · | · |
| layer2.2 | +0.324 | +1.923 | +1.000 | +0.020 | -1.133 | -0.040 | +0.037 | +0.035 | · | · | · | · | · |
| layer3.0 | -2.712 | -2.058 | -13.000 | -0.032 | +0.969 | -0.008 | -0.297 | -0.044 | · | · | · | · | · |
| layer3.1 | -0.190 | +6.186 | +4.000 | +0.084 | -0.189 | -0.080 | +0.073 | +0.027 | · | · | · | · | · |
| layer3.2 | +0.634 | +0.557 | +0.000 | +2.311 | -0.121 | -0.053 | +0.173 | +0.020 | · | · | · | · | · |
| penult | +0.634 | +0.557 | +0.000 | +2.311 | -0.121 | -0.053 | +0.173 | +0.020 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer2.0 | layer2.0 |
| spectral_slope | layer2.0 | layer2.0 |
| spectral_anisotropy | layer2.0 | layer3.0 |
| noise_sigma | layer1.0 | layer1.0 |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | layer1.0 | layer1.0 |
| edge_density | layer1.0 | layer1.0 |
| orientation_entropy | layer2.0 | layer2.0 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.5 |
| coarse_animal_vehicle | layer3.0 | layer3.5 |
| corruption_family | layer1.2 | layer2.0 |
| corruption_type | layer2.0 | layer2.0 |
| severity | layer2.2 | layer3.0 |