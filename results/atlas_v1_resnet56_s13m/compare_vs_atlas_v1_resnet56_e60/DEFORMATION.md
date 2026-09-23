# DEFORMATION  A=`results/atlas_v1_resnet56_e60`  B=`results/atlas_v1_resnet56_s13m`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.026 | 0.985 | 0.987 | True | 0.981 | 0.021 | 0.947 | 0.825 | 0.688 |
| layer1.0 | layer1.0 | 0.012 | 0.995 | 1.000 | True | 0.950 | 0.024 | 0.924 | 0.937 | 0.672 |
| layer1.5 | layer1.5 | 0.026 | 0.975 | 0.995 | True | 0.977 | 0.049 | 0.901 | 0.924 | 0.656 |
| layer1.8 | layer1.8 | 0.010 | 0.989 | 0.997 | True | 0.990 | 0.036 | 0.947 | 0.973 | 0.750 |
| layer2.0 | layer2.0 | 0.004 | 0.989 | 0.976 | True | 0.990 | 0.020 | 0.961 | 0.980 | 0.797 |
| layer2.5 | layer2.5 | 0.012 | 0.982 | 0.530 | True | 0.971 | 0.031 | 0.950 | 0.959 | 0.766 |
| layer2.8 | layer2.8 | 0.013 | 0.986 | 0.921 | True | 0.975 | 0.028 | 0.945 | 0.942 | 0.781 |
| layer3.0 | layer3.0 | 0.006 | 0.984 | 0.945 | True | 0.979 | 0.020 | 0.936 | 0.961 | 0.859 |
| layer3.5 | layer3.5 | 0.015 | 0.940 | 0.801 | True | 0.975 | 0.021 | 0.865 | 0.916 | 0.875 |
| layer3.8 | layer3.8 | 0.003 | 0.964 | 0.854 | True | 0.959 | 0.020 | 0.922 | 0.960 | 1.000 |
| penult | penult | 0.003 | 0.964 | 0.854 | True | 0.959 | 0.020 | 0.922 | 0.960 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.869 | 0.663 | 0.116 | 0.835 | 0.568 | 0.870 | 0.662 | 0.124 | 0.841 |
| layer1.0 | 0.925 | 0.745 | 0.118 | 0.948 | 0.568 | 0.896 | 0.749 | 0.121 | 0.943 |
| layer1.5 | 0.882 | 0.674 | 0.114 | 0.909 | 0.568 | 0.837 | 0.632 | 0.118 | 0.875 |
| layer1.8 | 0.950 | 0.784 | 0.110 | 0.965 | 0.568 | 0.933 | 0.761 | 0.122 | 0.957 |
| layer2.0 | 0.966 | 0.834 | 0.106 | 0.978 | 0.568 | 0.948 | 0.780 | 0.116 | 0.970 |
| layer2.5 | 0.950 | 0.792 | 0.104 | 0.957 | 0.568 | 0.937 | 0.737 | 0.111 | 0.946 |
| layer2.8 | 0.944 | 0.788 | 0.104 | 0.954 | 0.568 | 0.930 | 0.719 | 0.110 | 0.941 |
| layer3.0 | 0.944 | 0.828 | 0.103 | 0.955 | 0.568 | 0.934 | 0.743 | 0.111 | 0.945 |
| layer3.5 | 0.860 | 0.843 | 0.101 | 0.876 | 0.568 | 0.830 | 0.657 | 0.124 | 0.846 |
| layer3.8 | 0.895 | 0.931 | 0.100 | 0.768 | 0.568 | 0.694 | 0.576 | 0.124 | 0.770 |
| penult | 0.895 | 0.931 | 0.100 | 0.768 | 0.568 | 0.694 | 0.576 | 0.124 | 0.770 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | -1.269 | -0.020 | -1.000 | +0.007 | +131.527 | +0.019 | -0.203 | -0.008 | · | · | · | · | · |
| layer1.0 | -1.647 | +0.050 | -1.000 | -0.008 | +66.915 | -0.007 | -0.320 | -0.019 | · | · | · | · | · |
| layer1.5 | -0.876 | -0.694 | -1.000 | -0.013 | +54.274 | +0.087 | -0.121 | -0.020 | · | · | · | · | · |
| layer1.8 | +0.076 | -0.140 | +0.000 | -0.020 | +14.843 | +0.019 | -0.095 | -0.012 | · | · | · | · | · |
| layer2.0 | -0.100 | +0.218 | +0.000 | -0.028 | +3.235 | +0.001 | -0.037 | -0.008 | · | · | · | · | · |
| layer2.5 | +0.074 | +0.167 | +1.000 | +0.004 | -2.412 | -0.028 | +0.204 | +0.014 | · | · | · | · | · |
| layer2.8 | -0.217 | +0.057 | +0.000 | -0.013 | +0.799 | -0.042 | +0.058 | -0.008 | · | · | · | · | · |
| layer3.0 | -0.408 | +0.842 | -1.000 | -0.005 | +0.049 | -0.005 | +0.033 | +0.013 | · | · | · | · | · |
| layer3.5 | +0.198 | +0.896 | +0.000 | -0.027 | +0.174 | -0.002 | +0.032 | -0.027 | · | · | · | · | · |
| layer3.8 | -0.039 | +0.116 | +0.000 | -0.054 | +0.001 | -0.003 | +0.033 | -0.002 | · | · | · | · | · |
| penult | -0.039 | +0.116 | +0.000 | -0.054 | +0.001 | -0.003 | +0.033 | -0.002 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | layer1.0 |
| highfreq_ratio | layer1.5 | layer2.0 |
| spectral_slope | layer1.5 | layer2.0 |
| spectral_anisotropy | layer2.5 | layer3.0 |
| noise_sigma | layer1.5 | layer1.5 |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | layer1.0 | layer1.0 |
| edge_density | layer1.0 | layer1.0 |
| orientation_entropy | layer2.5 | layer3.0 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.5 | layer3.5 |
| coarse_animal_vehicle | layer3.5 | layer3.5 |
| corruption_family | layer2.0 | layer1.8 |
| corruption_type | layer2.0 | layer1.8 |
| severity | layer3.0 | layer3.0 |