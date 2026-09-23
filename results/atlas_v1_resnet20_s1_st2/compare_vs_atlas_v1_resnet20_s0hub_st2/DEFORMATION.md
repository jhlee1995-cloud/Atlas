# DEFORMATION  A=`results/atlas_v1_resnet20_s0hub_st2`  B=`results/atlas_v1_resnet20_s1_st2`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.002 | 0.998 | 0.990 | True | 0.953 | 0.024 | 0.984 | 0.995 | 0.906 |
| layer1.0 | layer1.0 | 0.015 | 0.979 | 1.000 | True | 0.986 | 0.035 | 0.900 | 0.946 | 0.797 |
| layer1.1 | layer1.1 | 0.029 | 0.960 | 0.951 | True | 0.986 | 0.024 | 0.895 | 0.928 | 0.766 |
| layer1.2 | layer1.2 | 0.014 | 0.985 | 0.978 | True | 0.998 | 0.019 | 0.956 | 0.970 | 0.797 |
| layer2.0 | layer2.0 | 0.021 | 0.964 | 0.941 | True | 0.992 | 0.017 | 0.929 | 0.917 | 0.656 |
| layer2.1 | layer2.1 | 0.010 | 0.982 | 0.927 | True | 0.996 | 0.012 | 0.939 | 0.932 | 0.734 |
| layer2.2 | layer2.2 | 0.009 | 0.986 | 0.945 | True | 0.998 | 0.011 | 0.927 | 0.943 | 0.734 |
| layer3.0 | layer3.0 | 0.014 | 0.967 | 0.978 | True | 0.986 | 0.016 | 0.894 | 0.944 | 0.828 |
| layer3.1 | layer3.1 | 0.012 | 0.966 | 0.974 | True | 0.969 | 0.025 | 0.877 | 0.924 | 0.812 |
| layer3.2 | layer3.2 | 0.002 | 0.939 | 0.771 | True | 0.920 | 0.034 | 0.903 | 0.963 | 1.000 |
| penult | penult | 0.002 | 0.939 | 0.771 | True | 0.920 | 0.034 | 0.903 | 0.963 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.991 | 0.896 | 0.120 | 0.994 | 0.552 | 0.985 | 0.896 | 0.122 | 0.994 |
| layer1.0 | 0.916 | 0.764 | 0.118 | 0.950 | 0.552 | 0.859 | 0.735 | 0.122 | 0.937 |
| layer1.1 | 0.891 | 0.729 | 0.113 | 0.927 | 0.552 | 0.851 | 0.698 | 0.116 | 0.916 |
| layer1.2 | 0.948 | 0.784 | 0.112 | 0.957 | 0.552 | 0.937 | 0.759 | 0.121 | 0.955 |
| layer2.0 | 0.933 | 0.739 | 0.105 | 0.915 | 0.552 | 0.929 | 0.701 | 0.114 | 0.906 |
| layer2.1 | 0.940 | 0.759 | 0.104 | 0.933 | 0.552 | 0.930 | 0.701 | 0.110 | 0.919 |
| layer2.2 | 0.921 | 0.752 | 0.103 | 0.932 | 0.552 | 0.907 | 0.687 | 0.109 | 0.914 |
| layer3.0 | 0.875 | 0.798 | 0.102 | 0.912 | 0.552 | 0.859 | 0.685 | 0.112 | 0.901 |
| layer3.1 | 0.864 | 0.859 | 0.101 | 0.876 | 0.552 | 0.765 | 0.631 | 0.143 | 0.845 |
| layer3.2 | 0.883 | 0.924 | 0.100 | 0.818 | 0.552 | 0.680 | 0.583 | 0.126 | 0.766 |
| penult | 0.883 | 0.924 | 0.100 | 0.818 | 0.552 | 0.680 | 0.583 | 0.126 | 0.766 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | -0.099 | +0.014 | +0.000 | -0.001 | +1.142 | +0.007 | -0.033 | +0.001 | · | · | · | · | · |
| layer1.0 | -0.253 | -0.754 | -1.000 | +0.005 | +26.285 | +0.054 | -0.096 | -0.008 | · | · | · | · | · |
| layer1.1 | -0.598 | -1.101 | -1.000 | -0.001 | +16.332 | +0.050 | -0.153 | -0.018 | · | · | · | · | · |
| layer1.2 | -0.643 | -0.719 | -1.000 | +0.016 | -3.928 | -0.036 | -0.158 | -0.007 | · | · | · | · | · |
| layer2.0 | -0.506 | -0.749 | -1.000 | +0.012 | +1.937 | +0.061 | -0.038 | -0.005 | · | · | · | · | · |
| layer2.1 | -0.647 | -0.086 | -1.000 | +0.008 | +4.632 | -0.039 | -0.038 | -0.015 | · | · | · | · | · |
| layer2.2 | -0.515 | -0.428 | -1.000 | -0.009 | +2.112 | -0.045 | +0.186 | -0.015 | · | · | · | · | · |
| layer3.0 | -0.090 | -1.851 | +1.000 | -0.007 | -0.026 | +0.021 | -0.138 | -0.001 | · | · | · | · | · |
| layer3.1 | +0.092 | +0.792 | +0.000 | -0.026 | +0.015 | +0.013 | -0.009 | +0.011 | · | · | · | · | · |
| layer3.2 | -0.158 | +0.135 | +0.000 | +0.070 | -0.007 | -0.009 | +0.070 | -0.001 | · | · | · | · | · |
| penult | -0.158 | +0.135 | +0.000 | +0.070 | -0.007 | -0.009 | +0.070 | -0.001 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer1.0 | layer1.1 |
| spectral_slope | layer2.0 | layer2.0 |
| spectral_anisotropy | layer2.0 | layer1.2 |
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
| severity | layer3.0 | layer3.0 |