# DEFORMATION  A=`results/atlas_v1_resnet20_s3_st3`  B=`results/atlas_v1_resnet56_e70`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.016 | 0.982 | 0.987 | True | 0.940 | 0.038 | 0.903 | 0.942 | 0.766 |
| layer1.0 | layer1.0 | 0.031 | 0.994 | 0.992 | True | 0.984 | 0.041 | 0.874 | 0.946 | 0.641 |
| layer1.1 | layer1.5 | 0.029 | 0.975 | 0.942 | True | 0.990 | 0.038 | 0.873 | 0.905 | 0.531 |
| layer1.2 | layer1.8 | 0.021 | 0.981 | 0.997 | True | 0.988 | 0.031 | 0.946 | 0.941 | 0.750 |
| layer2.0 | layer2.0 | 0.009 | 0.980 | 0.926 | True | 0.988 | 0.023 | 0.953 | 0.958 | 0.719 |
| layer2.1 | layer2.5 | 0.021 | 0.964 | 0.560 | True | 0.990 | 0.028 | 0.937 | 0.940 | 0.750 |
| layer2.2 | layer2.8 | 0.019 | 0.976 | 0.530 | True | 0.990 | 0.028 | 0.917 | 0.940 | 0.734 |
| layer3.0 | layer3.0 | 0.015 | 0.985 | 0.952 | True | 0.983 | 0.015 | 0.873 | 0.946 | 0.781 |
| layer3.1 | layer3.5 | 0.052 | 0.913 | 0.683 | True | 0.971 | 0.022 | 0.833 | 0.875 | 0.828 |
| layer3.2 | layer3.8 | 0.010 | 0.910 | 0.831 | True | 0.973 | 0.029 | 0.882 | 0.949 | 1.000 |
| penult | penult | 0.010 | 0.910 | 0.831 | True | 0.973 | 0.029 | 0.882 | 0.949 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.932 | 0.770 | 0.119 | 0.938 | 0.529 | 0.897 | 0.781 | 0.123 | 0.927 |
| layer1.0 | 0.866 | 0.628 | 0.122 | 0.925 | 0.529 | 0.835 | 0.596 | 0.124 | 0.913 |
| layer1.1 | 0.843 | 0.628 | 0.111 | 0.894 | 0.529 | 0.780 | 0.581 | 0.115 | 0.871 |
| layer1.2 | 0.939 | 0.748 | 0.112 | 0.932 | 0.529 | 0.917 | 0.732 | 0.121 | 0.925 |
| layer2.0 | 0.956 | 0.795 | 0.108 | 0.961 | 0.529 | 0.944 | 0.769 | 0.116 | 0.957 |
| layer2.1 | 0.942 | 0.781 | 0.105 | 0.947 | 0.529 | 0.929 | 0.726 | 0.115 | 0.938 |
| layer2.2 | 0.918 | 0.778 | 0.105 | 0.919 | 0.529 | 0.899 | 0.677 | 0.112 | 0.900 |
| layer3.0 | 0.896 | 0.813 | 0.103 | 0.938 | 0.529 | 0.862 | 0.673 | 0.114 | 0.923 |
| layer3.1 | 0.792 | 0.825 | 0.101 | 0.772 | 0.529 | 0.755 | 0.557 | 0.126 | 0.733 |
| layer3.2 | 0.867 | 0.925 | 0.100 | 0.711 | 0.529 | 0.672 | 0.568 | 0.127 | 0.741 |
| penult | 0.867 | 0.925 | 0.100 | 0.711 | 0.529 | 0.672 | 0.568 | 0.127 | 0.741 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | +0.004 | +0.063 | +0.000 | +0.002 | +45.047 | +0.009 | -0.056 | -0.003 | · | · | · | · | · |
| layer1.0 | -1.045 | -1.711 | -2.000 | -0.038 | +190.059 | +0.103 | -0.217 | -0.041 | · | · | · | · | · |
| layer1.1 | +0.292 | +0.476 | +0.000 | +0.044 | -8.921 | -0.084 | -0.007 | +0.025 | · | · | · | · | · |
| layer1.2 | -0.015 | +0.079 | +0.000 | +0.038 | -1.126 | -0.052 | +0.047 | +0.019 | · | · | · | · | · |
| layer2.0 | -0.487 | -0.555 | -2.000 | +0.031 | -2.816 | -0.019 | -0.030 | +0.015 | · | · | · | · | · |
| layer2.1 | +0.421 | -0.565 | +0.000 | +0.058 | -6.866 | -0.061 | +0.059 | +0.049 | · | · | · | · | · |
| layer2.2 | -0.280 | -1.381 | -1.000 | +0.081 | -2.685 | -0.023 | -0.028 | +0.032 | · | · | · | · | · |
| layer3.0 | -1.997 | -2.884 | -12.000 | +0.018 | +0.239 | -0.017 | -0.475 | -0.006 | · | · | · | · | · |
| layer3.1 | -0.740 | +1.800 | -2.000 | +0.263 | -0.402 | -0.092 | +0.058 | +0.064 | · | · | · | · | · |
| layer3.2 | +1.030 | +0.665 | +0.000 | +0.536 | -0.055 | -0.038 | +0.126 | +0.002 | · | · | · | · | · |
| penult | +1.030 | +0.665 | +0.000 | +0.536 | -0.055 | -0.038 | +0.126 | +0.002 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer1.0 | layer1.5 |
| spectral_slope | layer2.0 | layer2.0 |
| spectral_anisotropy | layer2.0 | layer3.0 |
| noise_sigma | layer1.0 | layer1.0 |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | layer1.0 | layer1.0 |
| edge_density | layer1.0 | stem |
| orientation_entropy | layer3.0 | layer2.8 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.5 |
| coarse_animal_vehicle | layer3.0 | layer3.0 |
| corruption_family | layer2.0 | layer1.8 |
| corruption_type | layer2.0 | layer2.0 |
| severity | layer3.0 | layer3.0 |