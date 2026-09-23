# DEFORMATION  A=`results/atlas_v1_resnet20_s3_st3`  B=`results/atlas_v1_resnet56_e40_st3`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.027 | 0.974 | 0.987 | True | 0.932 | 0.040 | 0.937 | 0.886 | 0.641 |
| layer1.0 | layer1.0 | 0.020 | 0.988 | 0.997 | True | 0.971 | 0.034 | 0.895 | 0.933 | 0.672 |
| layer1.1 | layer1.5 | 0.029 | 0.965 | 0.929 | True | 0.983 | 0.030 | 0.864 | 0.888 | 0.547 |
| layer1.2 | layer1.8 | 0.031 | 0.943 | 0.926 | True | 0.983 | 0.027 | 0.917 | 0.897 | 0.734 |
| layer2.0 | layer2.0 | 0.009 | 0.984 | 0.964 | True | 0.992 | 0.027 | 0.951 | 0.974 | 0.828 |
| layer2.1 | layer2.5 | 0.012 | 0.981 | 1.000 | True | 0.994 | 0.031 | 0.939 | 0.951 | 0.859 |
| layer2.2 | layer2.8 | 0.013 | 0.980 | 0.541 | True | 0.988 | 0.027 | 0.924 | 0.949 | 0.797 |
| layer3.0 | layer3.0 | 0.020 | 0.957 | 0.939 | True | 0.986 | 0.017 | 0.898 | 0.943 | 0.828 |
| layer3.1 | layer3.5 | 0.027 | 0.935 | 0.978 | True | 0.992 | 0.025 | 0.814 | 0.885 | 0.859 |
| layer3.2 | layer3.8 | 0.005 | 0.952 | 0.939 | True | 0.990 | 0.022 | 0.897 | 0.960 | 1.000 |
| penult | penult | 0.005 | 0.952 | 0.939 | True | 0.990 | 0.022 | 0.897 | 0.960 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.897 | 0.718 | 0.115 | 0.882 | 0.515 | 0.876 | 0.731 | 0.121 | 0.870 |
| layer1.0 | 0.911 | 0.702 | 0.124 | 0.933 | 0.515 | 0.858 | 0.677 | 0.127 | 0.913 |
| layer1.1 | 0.849 | 0.601 | 0.115 | 0.866 | 0.515 | 0.768 | 0.566 | 0.117 | 0.829 |
| layer1.2 | 0.911 | 0.698 | 0.109 | 0.884 | 0.515 | 0.890 | 0.679 | 0.118 | 0.870 |
| layer2.0 | 0.955 | 0.796 | 0.107 | 0.965 | 0.515 | 0.938 | 0.753 | 0.116 | 0.953 |
| layer2.1 | 0.939 | 0.759 | 0.105 | 0.936 | 0.515 | 0.929 | 0.716 | 0.113 | 0.926 |
| layer2.2 | 0.927 | 0.784 | 0.104 | 0.943 | 0.515 | 0.914 | 0.713 | 0.110 | 0.933 |
| layer3.0 | 0.886 | 0.792 | 0.102 | 0.924 | 0.515 | 0.869 | 0.692 | 0.112 | 0.912 |
| layer3.1 | 0.801 | 0.847 | 0.101 | 0.806 | 0.515 | 0.734 | 0.614 | 0.126 | 0.762 |
| layer3.2 | 0.879 | 0.919 | 0.100 | 0.804 | 0.515 | 0.684 | 0.560 | 0.125 | 0.760 |
| penult | 0.879 | 0.919 | 0.100 | 0.804 | 0.515 | 0.684 | 0.560 | 0.125 | 0.760 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | +0.388 | +0.309 | +1.000 | +0.015 | -170.843 | +0.005 | +0.041 | +0.006 | · | · | · | · | · |
| layer1.0 | -0.424 | -0.656 | -1.000 | -0.008 | -23.265 | +0.043 | -0.189 | -0.001 | · | · | · | · | · |
| layer1.1 | -0.106 | -0.689 | +0.000 | +0.057 | +33.535 | -0.008 | -0.138 | +0.021 | · | · | · | · | · |
| layer1.2 | -0.243 | -0.242 | -1.000 | +0.017 | +23.888 | +0.020 | -0.034 | +0.034 | · | · | · | · | · |
| layer2.0 | -1.170 | -0.534 | -2.000 | +0.027 | +4.542 | -0.005 | -0.147 | +0.015 | · | · | · | · | · |
| layer2.1 | -0.799 | -0.903 | -2.000 | +0.066 | -4.217 | -0.019 | -0.114 | +0.040 | · | · | · | · | · |
| layer2.2 | -1.460 | -0.632 | -2.000 | +0.055 | -0.133 | -0.029 | -0.178 | +0.022 | · | · | · | · | · |
| layer3.0 | -3.192 | -3.113 | -15.000 | -0.014 | +1.147 | -0.030 | -0.472 | -0.048 | · | · | · | · | · |
| layer3.1 | -0.280 | +1.142 | -1.000 | +0.069 | -0.173 | -0.056 | +0.058 | +0.026 | · | · | · | · | · |
| layer3.2 | +1.049 | +0.260 | +0.000 | -0.117 | +0.005 | -0.003 | +0.185 | -0.008 | · | · | · | · | · |
| penult | +1.049 | +0.260 | +0.000 | -0.117 | +0.005 | -0.003 | +0.185 | -0.008 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer1.0 | layer1.5 |
| spectral_slope | layer2.0 | layer1.5 |
| spectral_anisotropy | layer2.0 | layer2.0 |
| noise_sigma | layer1.0 | layer1.0 |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | layer1.0 | layer1.0 |
| edge_density | layer1.0 | stem |
| orientation_entropy | layer3.0 | layer2.0 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.5 |
| coarse_animal_vehicle | layer3.0 | layer3.0 |
| corruption_family | layer2.0 | layer1.5 |
| corruption_type | layer2.0 | layer1.8 |
| severity | layer3.0 | layer3.0 |