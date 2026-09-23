# DEFORMATION  A=`results/atlas_v1_resnet56_s12m`  B=`results/atlas_v1_resnet56_s13m`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.019 | 0.989 | 0.946 | True | 0.955 | 0.036 | 0.942 | 0.912 | 0.797 |
| layer1.0 | layer1.0 | 0.025 | 0.966 | 0.997 | True | 0.965 | 0.026 | 0.726 | 0.904 | 0.594 |
| layer1.5 | layer1.5 | 0.021 | 0.977 | 0.937 | True | 0.992 | 0.026 | 0.904 | 0.933 | 0.672 |
| layer1.8 | layer1.8 | 0.017 | 0.975 | 0.989 | True | 0.994 | 0.022 | 0.947 | 0.956 | 0.734 |
| layer2.0 | layer2.0 | 0.015 | 0.979 | 0.933 | True | 0.981 | 0.020 | 0.949 | 0.951 | 0.688 |
| layer2.5 | layer2.5 | 0.006 | 0.995 | 0.530 | True | 0.992 | 0.020 | 0.949 | 0.958 | 0.781 |
| layer2.8 | layer2.8 | 0.018 | 0.971 | 0.531 | True | 0.986 | 0.023 | 0.929 | 0.927 | 0.812 |
| layer3.0 | layer3.0 | 0.009 | 0.967 | 0.945 | True | 0.953 | 0.030 | 0.928 | 0.954 | 0.859 |
| layer3.5 | layer3.5 | 0.019 | 0.926 | 0.755 | True | 0.969 | 0.037 | 0.848 | 0.922 | 0.812 |
| layer3.8 | layer3.8 | 0.003 | 0.967 | 0.698 | True | 0.963 | 0.025 | 0.919 | 0.960 | 1.000 |
| penult | penult | 0.003 | 0.967 | 0.698 | True | 0.963 | 0.025 | 0.919 | 0.960 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.903 | 0.702 | 0.115 | 0.893 | 0.537 | 0.896 | 0.713 | 0.120 | 0.885 |
| layer1.0 | 0.858 | 0.640 | 0.114 | 0.897 | 0.537 | 0.784 | 0.635 | 0.119 | 0.882 |
| layer1.5 | 0.885 | 0.693 | 0.118 | 0.914 | 0.537 | 0.842 | 0.654 | 0.118 | 0.887 |
| layer1.8 | 0.942 | 0.756 | 0.111 | 0.947 | 0.537 | 0.929 | 0.736 | 0.122 | 0.935 |
| layer2.0 | 0.942 | 0.802 | 0.107 | 0.950 | 0.537 | 0.937 | 0.775 | 0.116 | 0.941 |
| layer2.5 | 0.953 | 0.816 | 0.104 | 0.964 | 0.537 | 0.941 | 0.756 | 0.110 | 0.953 |
| layer2.8 | 0.936 | 0.790 | 0.104 | 0.945 | 0.537 | 0.922 | 0.731 | 0.111 | 0.929 |
| layer3.0 | 0.929 | 0.805 | 0.103 | 0.947 | 0.537 | 0.917 | 0.719 | 0.110 | 0.935 |
| layer3.5 | 0.848 | 0.832 | 0.101 | 0.865 | 0.537 | 0.823 | 0.670 | 0.125 | 0.836 |
| layer3.8 | 0.899 | 0.925 | 0.100 | 0.790 | 0.537 | 0.686 | 0.589 | 0.122 | 0.763 |
| penult | 0.899 | 0.925 | 0.100 | 0.790 | 0.537 | 0.686 | 0.589 | 0.122 | 0.763 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | -0.608 | +0.079 | -1.000 | +0.004 | +11.480 | -0.023 | -0.070 | -0.007 | · | · | · | · | · |
| layer1.0 | -1.311 | +0.261 | -1.000 | -0.002 | +90.751 | -0.058 | -0.276 | -0.013 | · | · | · | · | · |
| layer1.5 | -0.679 | -1.001 | -1.000 | +0.015 | +55.982 | +0.097 | -0.115 | -0.015 | · | · | · | · | · |
| layer1.8 | +0.077 | +0.157 | +0.000 | +0.004 | -7.175 | +0.039 | -0.112 | +0.018 | · | · | · | · | · |
| layer2.0 | -0.192 | +0.189 | +1.000 | -0.007 | -0.720 | -0.001 | -0.035 | +0.015 | · | · | · | · | · |
| layer2.5 | -0.574 | -0.239 | +0.000 | -0.004 | -2.349 | -0.017 | +0.045 | +0.002 | · | · | · | · | · |
| layer2.8 | -0.694 | -0.608 | -1.000 | -0.036 | -0.883 | -0.033 | +0.046 | -0.026 | · | · | · | · | · |
| layer3.0 | -0.936 | +0.149 | -4.000 | -0.031 | +0.428 | +0.001 | -0.168 | -0.025 | · | · | · | · | · |
| layer3.5 | +0.722 | +1.926 | +1.000 | -0.042 | +0.087 | -0.032 | +0.156 | -0.034 | · | · | · | · | · |
| layer3.8 | +0.006 | +0.120 | +0.000 | -0.053 | +0.001 | -0.006 | +0.030 | +0.001 | · | · | · | · | · |
| penult | +0.006 | +0.120 | +0.000 | -0.053 | +0.001 | -0.006 | +0.030 | +0.001 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | layer1.0 |
| highfreq_ratio | layer2.0 | layer2.0 |
| spectral_slope | layer2.0 | layer2.0 |
| spectral_anisotropy | layer2.5 | layer3.0 |
| noise_sigma | layer1.0 | layer1.5 |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | layer1.0 | layer1.0 |
| edge_density | layer2.0 | layer1.0 |
| orientation_entropy | layer2.5 | layer3.0 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.5 | layer3.5 |
| coarse_animal_vehicle | layer3.0 | layer3.5 |
| corruption_family | layer2.0 | layer1.8 |
| corruption_type | layer2.0 | layer1.8 |
| severity | layer2.0 | layer3.0 |