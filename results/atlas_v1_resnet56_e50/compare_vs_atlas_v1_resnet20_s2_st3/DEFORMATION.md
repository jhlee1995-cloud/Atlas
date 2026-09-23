# DEFORMATION  A=`results/atlas_v1_resnet20_s2_st3`  B=`results/atlas_v1_resnet56_e50`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.003 | 0.996 | 0.953 | True | 0.967 | 0.032 | 0.969 | 0.935 | 0.688 |
| layer1.0 | layer1.0 | 0.051 | 0.960 | 0.946 | True | 0.899 | 0.108 | 0.859 | 0.906 | 0.578 |
| layer1.1 | layer1.5 | 0.030 | 0.948 | 0.679 | True | 0.973 | 0.033 | 0.815 | 0.921 | 0.641 |
| layer1.2 | layer1.8 | 0.013 | 0.969 | 1.000 | True | 0.981 | 0.032 | 0.926 | 0.953 | 0.766 |
| layer2.0 | layer2.0 | 0.009 | 0.985 | 0.925 | True | 1.000 | 0.015 | 0.954 | 0.971 | 0.812 |
| layer2.1 | layer2.5 | 0.008 | 0.984 | 0.977 | True | 0.990 | 0.019 | 0.943 | 0.970 | 0.734 |
| layer2.2 | layer2.8 | 0.013 | 0.985 | 0.969 | True | 0.988 | 0.028 | 0.925 | 0.947 | 0.750 |
| layer3.0 | layer3.0 | 0.020 | 0.982 | 0.974 | True | 0.955 | 0.029 | 0.894 | 0.942 | 0.828 |
| layer3.1 | layer3.5 | 0.080 | 0.863 | 0.866 | True | 0.986 | 0.029 | 0.774 | 0.815 | 0.859 |
| layer3.2 | layer3.8 | 0.006 | 0.934 | 0.660 | True | 0.953 | 0.032 | 0.907 | 0.951 | 1.000 |
| penult | penult | 0.006 | 0.934 | 0.660 | True | 0.953 | 0.032 | 0.907 | 0.951 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.964 | 0.721 | 0.115 | 0.923 | 0.508 | 0.964 | 0.690 | 0.122 | 0.912 |
| layer1.0 | 0.897 | 0.667 | 0.119 | 0.896 | 0.508 | 0.851 | 0.651 | 0.125 | 0.888 |
| layer1.1 | 0.824 | 0.638 | 0.115 | 0.894 | 0.508 | 0.786 | 0.613 | 0.115 | 0.876 |
| layer1.2 | 0.917 | 0.722 | 0.113 | 0.924 | 0.508 | 0.903 | 0.717 | 0.125 | 0.916 |
| layer2.0 | 0.956 | 0.794 | 0.107 | 0.959 | 0.508 | 0.943 | 0.775 | 0.116 | 0.950 |
| layer2.1 | 0.953 | 0.797 | 0.105 | 0.958 | 0.508 | 0.943 | 0.760 | 0.111 | 0.951 |
| layer2.2 | 0.936 | 0.763 | 0.103 | 0.936 | 0.508 | 0.920 | 0.714 | 0.111 | 0.925 |
| layer3.0 | 0.880 | 0.770 | 0.102 | 0.922 | 0.508 | 0.857 | 0.670 | 0.110 | 0.904 |
| layer3.1 | 0.741 | 0.843 | 0.101 | 0.720 | 0.508 | 0.709 | 0.579 | 0.129 | 0.671 |
| layer3.2 | 0.880 | 0.924 | 0.100 | 0.764 | 0.508 | 0.677 | 0.589 | 0.123 | 0.741 |
| penult | 0.880 | 0.924 | 0.100 | 0.764 | 0.508 | 0.677 | 0.589 | 0.123 | 0.741 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | -0.373 | -0.064 | +0.000 | -0.001 | +149.399 | +0.000 | +0.039 | +0.004 | · | · | · | · | · |
| layer1.0 | -1.974 | -0.978 | -2.000 | -0.008 | +371.628 | +0.050 | -0.377 | -0.033 | · | · | · | · | · |
| layer1.1 | -1.195 | -1.200 | -1.000 | +0.002 | +45.780 | -0.001 | -0.136 | -0.014 | · | · | · | · | · |
| layer1.2 | -0.507 | +0.462 | -1.000 | -0.011 | +33.249 | +0.032 | -0.157 | -0.003 | · | · | · | · | · |
| layer2.0 | -0.687 | -0.250 | -2.000 | -0.016 | +12.237 | +0.019 | -0.192 | -0.016 | · | · | · | · | · |
| layer2.1 | -0.081 | -0.698 | -1.000 | +0.046 | +0.695 | +0.024 | -0.110 | +0.009 | · | · | · | · | · |
| layer2.2 | -0.353 | -0.746 | -2.000 | +0.044 | +0.491 | +0.016 | -0.245 | -0.003 | · | · | · | · | · |
| layer3.0 | -3.141 | -3.242 | -16.000 | -0.061 | +1.489 | +0.045 | -0.197 | -0.076 | · | · | · | · | · |
| layer3.1 | -0.558 | +4.456 | +0.000 | +0.040 | -0.075 | -0.049 | -0.080 | +0.012 | · | · | · | · | · |
| layer3.2 | +0.994 | +0.456 | +0.000 | +0.118 | -0.025 | -0.015 | +0.259 | -0.004 | · | · | · | · | · |
| penult | +0.994 | +0.456 | +0.000 | +0.118 | -0.025 | -0.015 | +0.259 | -0.004 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | layer2.0 |
| highfreq_ratio | layer2.0 | layer2.0 |
| spectral_slope | layer2.0 | layer2.0 |
| spectral_anisotropy | layer2.0 | layer2.0 |
| noise_sigma | layer1.0 | layer1.5 |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | layer1.0 | layer1.5 |
| edge_density | layer1.0 | layer1.5 |
| orientation_entropy | layer2.0 | layer3.0 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.5 |
| coarse_animal_vehicle | layer3.0 | layer3.5 |
| corruption_family | layer1.2 | layer2.0 |
| corruption_type | layer2.0 | layer2.0 |
| severity | layer2.2 | layer3.0 |