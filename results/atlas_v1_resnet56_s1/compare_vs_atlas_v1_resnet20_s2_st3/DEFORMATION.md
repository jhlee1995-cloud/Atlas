# DEFORMATION  A=`results/atlas_v1_resnet20_s2_st3`  B=`results/atlas_v1_resnet56_s1`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.006 | 0.997 | 0.997 | True | 0.996 | 0.028 | 0.937 | 0.950 | 0.781 |
| layer1.0 | layer1.0 | 0.044 | 0.966 | 0.990 | True | 0.928 | 0.064 | 0.834 | 0.854 | 0.562 |
| layer1.1 | layer1.5 | 0.037 | 0.935 | 0.684 | True | 0.988 | 0.036 | 0.866 | 0.912 | 0.688 |
| layer1.2 | layer1.8 | 0.010 | 0.972 | 0.991 | True | 1.000 | 0.022 | 0.952 | 0.954 | 0.750 |
| layer2.0 | layer2.0 | 0.005 | 0.993 | 0.961 | True | 0.994 | 0.027 | 0.955 | 0.977 | 0.781 |
| layer2.1 | layer2.5 | 0.007 | 0.990 | 0.919 | True | 0.998 | 0.022 | 0.953 | 0.969 | 0.828 |
| layer2.2 | layer2.8 | 0.008 | 0.990 | 0.504 | True | 0.990 | 0.024 | 0.943 | 0.970 | 0.797 |
| layer3.0 | layer3.0 | 0.024 | 0.958 | 0.987 | True | 0.922 | 0.051 | 0.893 | 0.944 | 0.766 |
| layer3.1 | layer3.5 | 0.087 | 0.833 | 0.811 | True | 0.957 | 0.049 | 0.759 | 0.835 | 0.859 |
| layer3.2 | layer3.8 | 0.012 | 0.888 | 0.790 | True | 0.965 | 0.024 | 0.911 | 0.952 | 1.000 |
| penult | penult | 0.012 | 0.888 | 0.790 | True | 0.965 | 0.024 | 0.911 | 0.952 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.948 | 0.721 | 0.114 | 0.947 | 0.528 | 0.936 | 0.737 | 0.120 | 0.940 |
| layer1.0 | 0.799 | 0.638 | 0.118 | 0.872 | 0.528 | 0.729 | 0.608 | 0.119 | 0.836 |
| layer1.1 | 0.867 | 0.674 | 0.116 | 0.903 | 0.528 | 0.840 | 0.647 | 0.115 | 0.885 |
| layer1.2 | 0.947 | 0.748 | 0.111 | 0.916 | 0.528 | 0.941 | 0.715 | 0.122 | 0.900 |
| layer2.0 | 0.952 | 0.796 | 0.106 | 0.972 | 0.528 | 0.926 | 0.766 | 0.112 | 0.962 |
| layer2.1 | 0.949 | 0.796 | 0.104 | 0.965 | 0.528 | 0.934 | 0.769 | 0.111 | 0.960 |
| layer2.2 | 0.941 | 0.802 | 0.103 | 0.959 | 0.528 | 0.924 | 0.734 | 0.108 | 0.950 |
| layer3.0 | 0.886 | 0.785 | 0.102 | 0.922 | 0.528 | 0.876 | 0.688 | 0.108 | 0.910 |
| layer3.1 | 0.732 | 0.850 | 0.101 | 0.712 | 0.528 | 0.716 | 0.604 | 0.131 | 0.670 |
| layer3.2 | 0.869 | 0.936 | 0.100 | 0.757 | 0.528 | 0.659 | 0.594 | 0.125 | 0.739 |
| penult | 0.869 | 0.936 | 0.100 | 0.757 | 0.528 | 0.659 | 0.594 | 0.125 | 0.739 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | -0.766 | -0.332 | -1.000 | -0.006 | +346.321 | +0.004 | -0.112 | -0.011 | · | · | · | · | · |
| layer1.0 | -1.777 | -1.252 | -3.000 | -0.021 | +128.369 | +0.045 | -0.470 | -0.032 | · | · | · | · | · |
| layer1.1 | -0.288 | -0.313 | -1.000 | +0.026 | +0.027 | -0.065 | -0.071 | +0.004 | · | · | · | · | · |
| layer1.2 | -0.118 | +0.768 | -1.000 | +0.012 | +5.145 | +0.006 | -0.021 | +0.017 | · | · | · | · | · |
| layer2.0 | -0.512 | +0.255 | -1.000 | -0.024 | +11.566 | +0.047 | -0.077 | -0.014 | · | · | · | · | · |
| layer2.1 | +0.058 | +0.483 | +1.000 | -0.003 | +1.846 | +0.020 | +0.003 | +0.001 | · | · | · | · | · |
| layer2.2 | +0.119 | +0.814 | +1.000 | +0.000 | -0.092 | +0.005 | +0.199 | +0.016 | · | · | · | · | · |
| layer3.0 | -2.803 | -2.599 | -13.000 | -0.052 | +0.956 | +0.023 | -0.374 | -0.065 | · | · | · | · | · |
| layer3.1 | +0.048 | +6.019 | +4.000 | +0.051 | -0.183 | -0.074 | +0.237 | +0.014 | · | · | · | · | · |
| layer3.2 | +0.346 | +0.483 | +0.000 | +2.093 | -0.117 | -0.050 | +0.125 | +0.020 | · | · | · | · | · |
| penult | +0.346 | +0.483 | +0.000 | +2.093 | -0.117 | -0.050 | +0.125 | +0.020 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | layer1.5 |
| highfreq_ratio | layer2.0 | layer2.0 |
| spectral_slope | layer2.0 | layer2.0 |
| spectral_anisotropy | layer2.0 | layer2.0 |
| noise_sigma | layer1.0 | layer1.5 |
| saturation_mean | stem | layer1.0 |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | layer1.0 | layer1.0 |
| edge_density | layer1.0 | layer1.5 |
| orientation_entropy | layer2.0 | layer2.5 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.5 |
| coarse_animal_vehicle | layer3.0 | layer3.5 |
| corruption_family | layer1.2 | layer2.0 |
| corruption_type | layer2.0 | layer2.0 |
| severity | layer2.2 | layer2.0 |