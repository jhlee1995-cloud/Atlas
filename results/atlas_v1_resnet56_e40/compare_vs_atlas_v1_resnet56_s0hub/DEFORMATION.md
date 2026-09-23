# DEFORMATION  A=`results/atlas_v1_resnet56_s0hub`  B=`results/atlas_v1_resnet56_e40`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.014 | 0.986 | 0.990 | True | 0.942 | 0.035 | 0.942 | 0.916 | 0.703 |
| layer1.0 | layer1.0 | 0.019 | 0.992 | 0.997 | True | 0.944 | 0.042 | 0.892 | 0.925 | 0.562 |
| layer1.5 | layer1.5 | 0.029 | 0.970 | 0.948 | True | 0.977 | 0.029 | 0.918 | 0.922 | 0.703 |
| layer1.8 | layer1.8 | 0.025 | 0.951 | 0.938 | True | 0.979 | 0.026 | 0.924 | 0.913 | 0.734 |
| layer2.0 | layer2.0 | 0.007 | 0.985 | 0.915 | True | 0.996 | 0.028 | 0.932 | 0.959 | 0.828 |
| layer2.5 | layer2.5 | 0.012 | 0.986 | 1.000 | True | 0.988 | 0.033 | 0.915 | 0.946 | 0.797 |
| layer2.8 | layer2.8 | 0.008 | 0.981 | 0.963 | True | 0.986 | 0.032 | 0.925 | 0.948 | 0.719 |
| layer3.0 | layer3.0 | 0.021 | 0.933 | 0.910 | True | 0.915 | 0.037 | 0.931 | 0.950 | 0.828 |
| layer3.5 | layer3.5 | 0.023 | 0.898 | 0.845 | True | 0.961 | 0.047 | 0.846 | 0.916 | 0.797 |
| layer3.8 | layer3.8 | 0.016 | 0.900 | 0.847 | True | 0.922 | 0.028 | 0.906 | 0.944 | 1.000 |
| penult | penult | 0.016 | 0.900 | 0.847 | True | 0.922 | 0.028 | 0.906 | 0.944 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.949 | 0.724 | 0.113 | 0.919 | 0.472 | 0.935 | 0.730 | 0.118 | 0.910 |
| layer1.0 | 0.901 | 0.621 | 0.113 | 0.924 | 0.472 | 0.868 | 0.606 | 0.118 | 0.903 |
| layer1.5 | 0.898 | 0.643 | 0.113 | 0.895 | 0.472 | 0.876 | 0.626 | 0.118 | 0.880 |
| layer1.8 | 0.924 | 0.740 | 0.109 | 0.908 | 0.472 | 0.913 | 0.717 | 0.118 | 0.906 |
| layer2.0 | 0.936 | 0.759 | 0.105 | 0.947 | 0.472 | 0.921 | 0.723 | 0.113 | 0.936 |
| layer2.5 | 0.906 | 0.749 | 0.104 | 0.923 | 0.472 | 0.896 | 0.716 | 0.112 | 0.910 |
| layer2.8 | 0.919 | 0.773 | 0.104 | 0.937 | 0.472 | 0.905 | 0.711 | 0.111 | 0.925 |
| layer3.0 | 0.910 | 0.791 | 0.102 | 0.927 | 0.472 | 0.901 | 0.715 | 0.111 | 0.916 |
| layer3.5 | 0.820 | 0.864 | 0.101 | 0.829 | 0.472 | 0.771 | 0.623 | 0.131 | 0.806 |
| layer3.8 | 0.866 | 0.925 | 0.100 | 0.718 | 0.472 | 0.635 | 0.577 | 0.123 | 0.720 |
| penult | 0.866 | 0.925 | 0.100 | 0.718 | 0.472 | 0.635 | 0.577 | 0.123 | 0.720 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | +0.615 | +0.012 | +1.000 | +0.010 | -346.438 | +0.003 | +0.117 | +0.004 | · | · | · | · | · |
| layer1.0 | +0.620 | +0.462 | +1.000 | +0.027 | -72.127 | -0.039 | +0.072 | +0.041 | · | · | · | · | · |
| layer1.5 | +0.035 | -0.863 | +0.000 | +0.049 | +49.682 | +0.015 | -0.086 | +0.021 | · | · | · | · | · |
| layer1.8 | -0.360 | -0.447 | -1.000 | -0.008 | +16.281 | +0.038 | -0.123 | +0.013 | · | · | · | · | · |
| layer2.0 | -0.690 | -0.923 | -1.000 | +0.024 | +5.005 | +0.009 | -0.082 | +0.018 | · | · | · | · | · |
| layer2.5 | -1.192 | -2.604 | -3.000 | +0.040 | +0.851 | +0.038 | -0.190 | -0.003 | · | · | · | · | · |
| layer2.8 | -1.562 | -1.945 | -3.000 | +0.029 | +2.463 | +0.008 | -0.237 | -0.019 | · | · | · | · | · |
| layer3.0 | -0.784 | -1.614 | -3.000 | -0.009 | +0.386 | -0.001 | -0.006 | -0.030 | · | · | · | · | · |
| layer3.5 | +0.116 | -2.075 | -3.000 | -0.005 | +0.079 | +0.011 | -0.066 | -0.015 | · | · | · | · | · |
| layer3.8 | +0.215 | -0.353 | +0.000 | -2.479 | +0.129 | +0.053 | +0.018 | -0.029 | · | · | · | · | · |
| penult | +0.215 | -0.353 | +0.000 | -2.479 | +0.129 | +0.053 | +0.018 | -0.029 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer2.0 | layer1.5 |
| spectral_slope | layer2.0 | layer1.5 |
| spectral_anisotropy | layer3.0 | layer2.0 |
| noise_sigma | layer1.0 | layer1.0 |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | layer1.0 | layer1.0 |
| edge_density | layer1.0 | stem |
| orientation_entropy | layer2.0 | layer2.0 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.5 | layer3.5 |
| coarse_animal_vehicle | layer3.5 | layer3.0 |
| corruption_family | layer2.0 | layer1.5 |
| corruption_type | layer2.0 | layer1.8 |
| severity | layer3.0 | layer3.0 |