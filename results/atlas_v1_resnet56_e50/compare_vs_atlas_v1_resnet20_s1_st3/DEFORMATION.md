# DEFORMATION  A=`results/atlas_v1_resnet20_s1_st3`  B=`results/atlas_v1_resnet56_e50`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.021 | 0.981 | 0.938 | True | 0.971 | 0.031 | 0.947 | 0.889 | 0.688 |
| layer1.0 | layer1.0 | 0.039 | 0.963 | 0.951 | True | 0.897 | 0.103 | 0.872 | 0.883 | 0.609 |
| layer1.1 | layer1.5 | 0.018 | 0.982 | 0.946 | True | 0.979 | 0.031 | 0.889 | 0.909 | 0.641 |
| layer1.2 | layer1.8 | 0.021 | 0.969 | 0.987 | True | 0.988 | 0.020 | 0.941 | 0.911 | 0.625 |
| layer2.0 | layer2.0 | 0.019 | 0.970 | 0.963 | True | 0.996 | 0.019 | 0.931 | 0.954 | 0.734 |
| layer2.1 | layer2.5 | 0.008 | 0.986 | 0.977 | True | 0.994 | 0.025 | 0.931 | 0.943 | 0.734 |
| layer2.2 | layer2.8 | 0.013 | 0.982 | 0.972 | True | 0.994 | 0.025 | 0.922 | 0.934 | 0.750 |
| layer3.0 | layer3.0 | 0.012 | 0.966 | 0.987 | True | 0.969 | 0.035 | 0.883 | 0.939 | 0.812 |
| layer3.1 | layer3.5 | 0.043 | 0.882 | 0.903 | True | 0.969 | 0.042 | 0.815 | 0.869 | 0.797 |
| layer3.2 | layer3.8 | 0.004 | 0.948 | 0.739 | True | 0.940 | 0.043 | 0.881 | 0.949 | 1.000 |
| penult | penult | 0.004 | 0.948 | 0.739 | True | 0.940 | 0.043 | 0.881 | 0.949 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.894 | 0.654 | 0.111 | 0.879 | 0.511 | 0.883 | 0.640 | 0.113 | 0.872 |
| layer1.0 | 0.900 | 0.673 | 0.116 | 0.891 | 0.511 | 0.830 | 0.645 | 0.120 | 0.865 |
| layer1.1 | 0.887 | 0.666 | 0.114 | 0.906 | 0.511 | 0.858 | 0.632 | 0.116 | 0.883 |
| layer1.2 | 0.935 | 0.720 | 0.112 | 0.903 | 0.511 | 0.928 | 0.707 | 0.120 | 0.898 |
| layer2.0 | 0.934 | 0.761 | 0.107 | 0.939 | 0.511 | 0.917 | 0.751 | 0.114 | 0.936 |
| layer2.1 | 0.939 | 0.785 | 0.105 | 0.945 | 0.511 | 0.915 | 0.724 | 0.111 | 0.929 |
| layer2.2 | 0.929 | 0.770 | 0.104 | 0.938 | 0.511 | 0.909 | 0.703 | 0.112 | 0.917 |
| layer3.0 | 0.890 | 0.779 | 0.102 | 0.932 | 0.511 | 0.866 | 0.678 | 0.111 | 0.912 |
| layer3.1 | 0.779 | 0.841 | 0.101 | 0.776 | 0.511 | 0.751 | 0.599 | 0.131 | 0.732 |
| layer3.2 | 0.881 | 0.920 | 0.100 | 0.772 | 0.511 | 0.676 | 0.585 | 0.124 | 0.742 |
| penult | 0.881 | 0.920 | 0.100 | 0.772 | 0.511 | 0.676 | 0.585 | 0.124 | 0.742 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | -0.123 | -0.082 | +0.000 | +0.005 | +41.255 | +0.024 | +0.069 | +0.006 | · | · | · | · | · |
| layer1.0 | -1.351 | -0.391 | +0.000 | -0.006 | +343.169 | +0.028 | -0.171 | -0.025 | · | · | · | · | · |
| layer1.1 | -0.696 | -0.203 | +0.000 | +0.002 | +39.936 | +0.030 | -0.010 | -0.007 | · | · | · | · | · |
| layer1.2 | +0.028 | +0.162 | +0.000 | -0.016 | +31.772 | +0.056 | -0.049 | -0.006 | · | · | · | · | · |
| layer2.0 | -0.846 | -0.719 | -3.000 | -0.017 | +7.824 | -0.049 | -0.204 | -0.024 | · | · | · | · | · |
| layer2.1 | +0.226 | -1.279 | -1.000 | +0.031 | -5.244 | +0.023 | -0.158 | +0.002 | · | · | · | · | · |
| layer2.2 | -0.088 | -1.375 | -2.000 | +0.042 | -1.329 | +0.048 | -0.230 | -0.004 | · | · | · | · | · |
| layer3.0 | -2.869 | -1.437 | -16.000 | -0.052 | +1.631 | +0.050 | -0.187 | -0.058 | · | · | · | · | · |
| layer3.1 | -0.420 | +2.838 | -1.000 | -0.022 | +0.010 | -0.048 | -0.172 | +0.009 | · | · | · | · | · |
| layer3.2 | +1.273 | +0.319 | +0.000 | +0.073 | -0.018 | -0.005 | +0.193 | -0.001 | · | · | · | · | · |
| penult | +1.273 | +0.319 | +0.000 | +0.073 | -0.018 | -0.005 | +0.193 | -0.001 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | layer2.0 |
| highfreq_ratio | layer1.1 | layer2.0 |
| spectral_slope | layer2.0 | layer2.0 |
| spectral_anisotropy | layer1.2 | layer2.0 |
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
| severity | layer3.0 | layer3.0 |