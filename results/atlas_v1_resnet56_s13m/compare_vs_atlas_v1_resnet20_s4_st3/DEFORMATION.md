# DEFORMATION  A=`results/atlas_v1_resnet20_s4_st3`  B=`results/atlas_v1_resnet56_s13m`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.005 | 0.996 | 0.995 | True | 0.950 | 0.044 | 0.955 | 0.982 | 0.844 |
| layer1.0 | layer1.0 | 0.041 | 0.983 | 0.992 | True | 0.905 | 0.069 | 0.770 | 0.888 | 0.469 |
| layer1.1 | layer1.5 | 0.037 | 0.958 | 0.937 | True | 0.983 | 0.029 | 0.839 | 0.893 | 0.531 |
| layer1.2 | layer1.8 | 0.013 | 0.972 | 0.989 | True | 0.983 | 0.020 | 0.938 | 0.941 | 0.734 |
| layer2.0 | layer2.0 | 0.004 | 0.988 | 0.899 | True | 0.990 | 0.021 | 0.956 | 0.980 | 0.812 |
| layer2.1 | layer2.5 | 0.014 | 0.981 | 0.480 | True | 0.977 | 0.031 | 0.941 | 0.953 | 0.797 |
| layer2.2 | layer2.8 | 0.014 | 0.978 | 0.548 | True | 0.977 | 0.027 | 0.927 | 0.947 | 0.812 |
| layer3.0 | layer3.0 | 0.016 | 0.968 | 0.968 | True | 0.946 | 0.032 | 0.885 | 0.939 | 0.750 |
| layer3.1 | layer3.5 | 0.090 | 0.811 | 0.646 | True | 0.986 | 0.039 | 0.764 | 0.828 | 0.797 |
| layer3.2 | layer3.8 | 0.005 | 0.936 | 0.594 | True | 0.957 | 0.042 | 0.914 | 0.956 | 1.000 |
| penult | penult | 0.005 | 0.936 | 0.594 | True | 0.957 | 0.042 | 0.914 | 0.956 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.965 | 0.785 | 0.116 | 0.973 | 0.514 | 0.956 | 0.799 | 0.119 | 0.969 |
| layer1.0 | 0.830 | 0.548 | 0.116 | 0.898 | 0.514 | 0.747 | 0.548 | 0.119 | 0.881 |
| layer1.1 | 0.866 | 0.645 | 0.115 | 0.878 | 0.514 | 0.816 | 0.636 | 0.118 | 0.855 |
| layer1.2 | 0.947 | 0.762 | 0.111 | 0.944 | 0.514 | 0.926 | 0.749 | 0.123 | 0.935 |
| layer2.0 | 0.960 | 0.840 | 0.107 | 0.978 | 0.514 | 0.940 | 0.793 | 0.117 | 0.971 |
| layer2.1 | 0.936 | 0.756 | 0.105 | 0.946 | 0.514 | 0.918 | 0.710 | 0.111 | 0.938 |
| layer2.2 | 0.926 | 0.787 | 0.104 | 0.949 | 0.514 | 0.902 | 0.721 | 0.113 | 0.942 |
| layer3.0 | 0.886 | 0.795 | 0.102 | 0.932 | 0.514 | 0.864 | 0.712 | 0.112 | 0.917 |
| layer3.1 | 0.687 | 0.810 | 0.101 | 0.721 | 0.514 | 0.688 | 0.563 | 0.133 | 0.683 |
| layer3.2 | 0.877 | 0.930 | 0.100 | 0.767 | 0.514 | 0.671 | 0.561 | 0.123 | 0.745 |
| penult | 0.877 | 0.930 | 0.100 | 0.767 | 0.514 | 0.671 | 0.561 | 0.123 | 0.745 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | -1.122 | -0.138 | -1.000 | +0.005 | +117.647 | -0.011 | -0.196 | -0.014 | · | · | · | · | · |
| layer1.0 | -2.747 | -0.806 | -3.000 | -0.029 | +115.690 | +0.106 | -0.621 | -0.050 | · | · | · | · | · |
| layer1.1 | -1.065 | -0.768 | -1.000 | +0.020 | +44.997 | +0.113 | -0.197 | -0.004 | · | · | · | · | · |
| layer1.2 | -0.419 | +0.058 | +0.000 | +0.014 | +11.445 | +0.037 | -0.203 | +0.017 | · | · | · | · | · |
| layer2.0 | -1.387 | -0.394 | -2.000 | +0.000 | +8.171 | -0.006 | -0.244 | -0.002 | · | · | · | · | · |
| layer2.1 | -0.714 | -0.449 | -1.000 | +0.054 | -4.782 | -0.038 | -0.044 | +0.036 | · | · | · | · | · |
| layer2.2 | -1.001 | -1.463 | -3.000 | +0.022 | -0.838 | -0.021 | -0.115 | -0.007 | · | · | · | · | · |
| layer3.0 | -3.626 | -1.582 | -17.000 | -0.044 | +1.526 | +0.014 | -0.612 | -0.047 | · | · | · | · | · |
| layer3.1 | +0.125 | +5.299 | +2.000 | -0.076 | +0.219 | -0.053 | +0.168 | -0.060 | · | · | · | · | · |
| layer3.2 | +0.826 | +0.484 | +0.000 | +0.290 | -0.038 | -0.018 | +0.104 | -0.004 | · | · | · | · | · |
| penult | +0.826 | +0.484 | +0.000 | +0.290 | -0.038 | -0.018 | +0.104 | -0.004 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | layer1.0 |
| highfreq_ratio | layer2.0 | layer2.0 |
| spectral_slope | layer2.0 | layer2.0 |
| spectral_anisotropy | layer2.0 | layer3.0 |
| noise_sigma | layer1.0 | layer1.5 |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | stem | layer1.0 |
| edge_density | layer1.0 | layer1.0 |
| orientation_entropy | layer2.0 | layer3.0 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.5 |
| coarse_animal_vehicle | layer3.0 | layer3.5 |
| corruption_family | layer2.0 | layer1.8 |
| corruption_type | layer2.0 | layer1.8 |
| severity | layer2.0 | layer3.0 |