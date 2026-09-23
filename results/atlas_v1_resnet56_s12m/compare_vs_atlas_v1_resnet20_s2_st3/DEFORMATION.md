# DEFORMATION  A=`results/atlas_v1_resnet20_s2_st3`  B=`results/atlas_v1_resnet56_s12m`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.002 | 0.997 | 0.953 | True | 0.988 | 0.019 | 0.973 | 0.976 | 0.781 |
| layer1.0 | layer1.0 | 0.025 | 0.985 | 0.995 | True | 0.961 | 0.046 | 0.813 | 0.908 | 0.625 |
| layer1.1 | layer1.5 | 0.030 | 0.930 | 0.695 | True | 0.992 | 0.022 | 0.917 | 0.926 | 0.734 |
| layer1.2 | layer1.8 | 0.009 | 0.987 | 1.000 | True | 0.988 | 0.020 | 0.962 | 0.975 | 0.750 |
| layer2.0 | layer2.0 | 0.026 | 0.955 | 0.933 | True | 0.990 | 0.017 | 0.915 | 0.931 | 0.719 |
| layer2.1 | layer2.5 | 0.010 | 0.987 | 0.994 | True | 0.973 | 0.026 | 0.932 | 0.958 | 0.797 |
| layer2.2 | layer2.8 | 0.009 | 0.980 | 0.990 | True | 0.973 | 0.028 | 0.934 | 0.959 | 0.875 |
| layer3.0 | layer3.0 | 0.018 | 0.979 | 0.964 | True | 0.934 | 0.028 | 0.916 | 0.954 | 0.828 |
| layer3.1 | layer3.5 | 0.082 | 0.812 | 0.875 | True | 0.961 | 0.026 | 0.759 | 0.833 | 0.922 |
| layer3.2 | layer3.8 | 0.006 | 0.917 | 0.879 | True | 0.959 | 0.027 | 0.909 | 0.954 | 1.000 |
| penult | penult | 0.006 | 0.917 | 0.879 | True | 0.959 | 0.027 | 0.909 | 0.954 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.980 | 0.811 | 0.114 | 0.973 | 0.525 | 0.972 | 0.778 | 0.120 | 0.967 |
| layer1.0 | 0.842 | 0.690 | 0.117 | 0.903 | 0.525 | 0.728 | 0.648 | 0.122 | 0.876 |
| layer1.1 | 0.893 | 0.727 | 0.115 | 0.910 | 0.525 | 0.880 | 0.704 | 0.114 | 0.902 |
| layer1.2 | 0.953 | 0.736 | 0.111 | 0.955 | 0.525 | 0.942 | 0.737 | 0.124 | 0.944 |
| layer2.0 | 0.909 | 0.764 | 0.107 | 0.913 | 0.525 | 0.901 | 0.728 | 0.114 | 0.891 |
| layer2.1 | 0.945 | 0.786 | 0.104 | 0.956 | 0.525 | 0.928 | 0.735 | 0.110 | 0.944 |
| layer2.2 | 0.940 | 0.790 | 0.103 | 0.951 | 0.525 | 0.923 | 0.728 | 0.110 | 0.938 |
| layer3.0 | 0.894 | 0.803 | 0.102 | 0.926 | 0.525 | 0.883 | 0.703 | 0.110 | 0.915 |
| layer3.1 | 0.711 | 0.830 | 0.101 | 0.693 | 0.525 | 0.698 | 0.596 | 0.127 | 0.667 |
| layer3.2 | 0.880 | 0.922 | 0.100 | 0.772 | 0.525 | 0.672 | 0.593 | 0.123 | 0.743 |
| penult | 0.880 | 0.922 | 0.100 | 0.772 | 0.525 | 0.672 | 0.593 | 0.123 | 0.743 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | -0.651 | -0.101 | +0.000 | +0.004 | +112.425 | +0.003 | -0.134 | -0.002 | · | · | · | · | · |
| layer1.0 | -1.386 | -0.993 | -2.000 | -0.017 | -17.349 | +0.069 | -0.346 | -0.030 | · | · | · | · | · |
| layer1.1 | -0.305 | +0.060 | +0.000 | +0.011 | +7.399 | -0.100 | -0.051 | +0.010 | · | · | · | · | · |
| layer1.2 | -0.289 | +0.503 | +0.000 | +0.004 | +19.168 | -0.022 | +0.009 | +0.006 | · | · | · | · | · |
| layer2.0 | -0.568 | -0.037 | -2.000 | +0.006 | +10.355 | +0.003 | -0.114 | -0.013 | · | · | · | · | · |
| layer2.1 | +0.345 | +0.170 | +0.000 | +0.052 | +1.098 | +0.001 | +0.061 | +0.037 | · | · | · | · | · |
| layer2.2 | -0.035 | -0.208 | -1.000 | +0.056 | +1.512 | +0.005 | -0.182 | +0.026 | · | · | · | · | · |
| layer3.0 | -2.665 | -3.387 | -14.000 | -0.016 | +0.961 | +0.017 | -0.271 | -0.042 | · | · | · | · | · |
| layer3.1 | -0.678 | +3.184 | +2.000 | -0.011 | +0.064 | -0.014 | +0.066 | -0.014 | · | · | · | · | · |
| layer3.2 | +0.686 | +0.435 | +0.000 | +0.346 | -0.041 | -0.022 | +0.112 | -0.002 | · | · | · | · | · |
| penult | +0.686 | +0.435 | +0.000 | +0.346 | -0.041 | -0.022 | +0.112 | -0.002 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer2.0 | layer2.0 |
| spectral_slope | layer2.0 | layer2.0 |
| spectral_anisotropy | layer2.0 | layer2.5 |
| noise_sigma | layer1.0 | layer1.0 |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | layer1.0 | layer1.0 |
| edge_density | layer1.0 | layer2.0 |
| orientation_entropy | layer2.0 | layer2.5 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.5 |
| coarse_animal_vehicle | layer3.0 | layer3.0 |
| corruption_family | layer1.2 | layer2.0 |
| corruption_type | layer2.0 | layer2.0 |
| severity | layer2.2 | layer2.0 |