# DEFORMATION  A=`results/atlas_v1_resnet56_s0hub`  B=`results/atlas_v1_resnet56_e20`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.007 | 0.992 | 0.982 | True | 0.977 | 0.036 | 0.925 | 0.883 | 0.578 |
| layer1.0 | layer1.0 | 0.014 | 0.987 | 0.992 | True | 0.901 | 0.058 | 0.892 | 0.923 | 0.688 |
| layer1.5 | layer1.5 | 0.027 | 0.959 | 0.670 | True | 0.963 | 0.031 | 0.894 | 0.905 | 0.719 |
| layer1.8 | layer1.8 | 0.019 | 0.968 | 0.985 | True | 0.990 | 0.021 | 0.921 | 0.924 | 0.672 |
| layer2.0 | layer2.0 | 0.017 | 0.978 | 0.954 | True | 0.990 | 0.031 | 0.926 | 0.954 | 0.828 |
| layer2.5 | layer2.5 | 0.023 | 0.905 | 0.992 | True | 0.986 | 0.037 | 0.879 | 0.918 | 0.766 |
| layer2.8 | layer2.8 | 0.019 | 0.953 | 0.697 | True | 0.975 | 0.040 | 0.880 | 0.936 | 0.766 |
| layer3.0 | layer3.0 | 0.025 | 0.945 | 0.903 | True | 0.911 | 0.043 | 0.893 | 0.928 | 0.844 |
| layer3.5 | layer3.5 | 0.036 | 0.859 | 0.790 | True | 0.944 | 0.056 | 0.820 | 0.884 | 0.828 |
| layer3.8 | layer3.8 | 0.059 | 0.871 | 0.830 | True | 0.866 | 0.064 | 0.824 | 0.868 | 0.953 |
| penult | penult | 0.059 | 0.871 | 0.830 | True | 0.866 | 0.064 | 0.824 | 0.868 | 0.953 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.928 | 0.638 | 0.108 | 0.909 | 0.458 | 0.886 | 0.659 | 0.112 | 0.898 |
| layer1.0 | 0.939 | 0.677 | 0.107 | 0.927 | 0.458 | 0.890 | 0.662 | 0.111 | 0.917 |
| layer1.5 | 0.889 | 0.665 | 0.113 | 0.885 | 0.458 | 0.840 | 0.628 | 0.117 | 0.861 |
| layer1.8 | 0.922 | 0.681 | 0.110 | 0.902 | 0.458 | 0.896 | 0.684 | 0.117 | 0.893 |
| layer2.0 | 0.934 | 0.743 | 0.106 | 0.948 | 0.458 | 0.895 | 0.717 | 0.109 | 0.940 |
| layer2.5 | 0.824 | 0.743 | 0.103 | 0.917 | 0.458 | 0.877 | 0.693 | 0.109 | 0.901 |
| layer2.8 | 0.887 | 0.730 | 0.103 | 0.930 | 0.458 | 0.864 | 0.684 | 0.109 | 0.909 |
| layer3.0 | 0.898 | 0.761 | 0.103 | 0.923 | 0.458 | 0.878 | 0.671 | 0.108 | 0.906 |
| layer3.5 | 0.799 | 0.831 | 0.101 | 0.801 | 0.458 | 0.743 | 0.568 | 0.123 | 0.746 |
| layer3.8 | 0.802 | 0.906 | 0.100 | 0.710 | 0.458 | 0.605 | 0.554 | 0.121 | 0.705 |
| penult | 0.802 | 0.906 | 0.100 | 0.710 | 0.458 | 0.605 | 0.554 | 0.121 | 0.705 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | -0.586 | +0.196 | +0.000 | +0.013 | +17.046 | -0.006 | -0.066 | -0.012 | · | · | · | · | · |
| layer1.0 | -1.111 | +0.192 | -1.000 | +0.015 | +32.429 | -0.037 | -0.202 | +0.009 | · | · | · | · | · |
| layer1.5 | -0.509 | -0.920 | +0.000 | -0.020 | +43.949 | +0.080 | -0.153 | -0.006 | · | · | · | · | · |
| layer1.8 | -0.884 | -0.902 | -2.000 | -0.019 | +3.776 | +0.020 | -0.206 | -0.025 | · | · | · | · | · |
| layer2.0 | -1.497 | -1.338 | -3.000 | -0.016 | +22.362 | +0.018 | -0.236 | -0.017 | · | · | · | · | · |
| layer2.5 | -2.084 | -1.921 | -5.000 | +0.005 | +15.202 | +0.018 | -0.362 | -0.027 | · | · | · | · | · |
| layer2.8 | -2.112 | -2.534 | -5.000 | -0.005 | +8.403 | +0.050 | -0.432 | -0.052 | · | · | · | · | · |
| layer3.0 | -1.035 | -2.634 | -6.000 | -0.053 | +1.668 | +0.058 | -0.188 | -0.061 | · | · | · | · | · |
| layer3.5 | -1.373 | -4.915 | -11.000 | -0.027 | +0.316 | +0.044 | -0.209 | -0.037 | · | · | · | · | · |
| layer3.8 | +1.070 | -1.278 | +1.000 | -3.293 | +0.271 | +0.132 | -0.040 | -0.052 | · | · | · | · | · |
| penult | +1.070 | -1.278 | +1.000 | -3.293 | +0.271 | +0.132 | -0.040 | -0.052 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer2.0 | layer2.0 |
| spectral_slope | layer2.0 | layer2.0 |
| spectral_anisotropy | layer3.0 | layer3.0 |
| noise_sigma | layer1.0 | layer1.5 |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | layer1.0 | layer1.5 |
| edge_density | layer1.0 | stem |
| orientation_entropy | layer2.0 | layer2.5 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.5 | layer3.5 |
| coarse_animal_vehicle | layer3.5 | layer3.0 |
| corruption_family | layer2.0 | layer2.0 |
| corruption_type | layer2.0 | layer2.0 |
| severity | layer3.0 | layer3.0 |