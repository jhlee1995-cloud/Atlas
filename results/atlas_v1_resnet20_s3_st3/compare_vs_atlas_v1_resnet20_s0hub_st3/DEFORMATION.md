# DEFORMATION  A=`results/atlas_v1_resnet20_s0hub_st3`  B=`results/atlas_v1_resnet20_s3_st3`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.009 | 0.989 | 0.987 | True | 0.942 | 0.028 | 0.953 | 0.975 | 0.906 |
| layer1.0 | layer1.0 | 0.007 | 0.992 | 0.995 | True | 0.991 | 0.029 | 0.949 | 0.968 | 0.750 |
| layer1.1 | layer1.1 | 0.015 | 0.989 | 0.942 | True | 0.983 | 0.033 | 0.867 | 0.902 | 0.578 |
| layer1.2 | layer1.2 | 0.007 | 0.993 | 0.997 | True | 0.988 | 0.028 | 0.956 | 0.916 | 0.688 |
| layer2.0 | layer2.0 | 0.012 | 0.980 | 0.941 | True | 0.988 | 0.019 | 0.939 | 0.932 | 0.750 |
| layer2.1 | layer2.1 | 0.014 | 0.975 | 0.937 | True | 0.988 | 0.017 | 0.930 | 0.932 | 0.750 |
| layer2.2 | layer2.2 | 0.013 | 0.981 | 0.945 | True | 0.990 | 0.019 | 0.922 | 0.947 | 0.734 |
| layer3.0 | layer3.0 | 0.012 | 0.968 | 0.966 | True | 0.988 | 0.016 | 0.907 | 0.949 | 0.828 |
| layer3.1 | layer3.1 | 0.041 | 0.890 | 0.946 | True | 0.990 | 0.014 | 0.837 | 0.892 | 0.750 |
| layer3.2 | layer3.2 | 0.003 | 0.973 | 1.000 | True | 0.965 | 0.022 | 0.895 | 0.964 | 1.000 |
| penult | penult | 0.003 | 0.973 | 1.000 | True | 0.965 | 0.022 | 0.895 | 0.964 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.957 | 0.812 | 0.119 | 0.959 | 0.507 | 0.954 | 0.806 | 0.122 | 0.947 |
| layer1.0 | 0.924 | 0.774 | 0.125 | 0.967 | 0.507 | 0.889 | 0.749 | 0.130 | 0.960 |
| layer1.1 | 0.886 | 0.698 | 0.115 | 0.918 | 0.507 | 0.794 | 0.671 | 0.117 | 0.901 |
| layer1.2 | 0.953 | 0.735 | 0.113 | 0.922 | 0.507 | 0.927 | 0.713 | 0.123 | 0.917 |
| layer2.0 | 0.948 | 0.760 | 0.106 | 0.928 | 0.507 | 0.929 | 0.715 | 0.117 | 0.911 |
| layer2.1 | 0.933 | 0.744 | 0.105 | 0.926 | 0.507 | 0.922 | 0.683 | 0.110 | 0.915 |
| layer2.2 | 0.922 | 0.741 | 0.103 | 0.931 | 0.507 | 0.900 | 0.671 | 0.106 | 0.913 |
| layer3.0 | 0.894 | 0.808 | 0.102 | 0.928 | 0.507 | 0.862 | 0.709 | 0.113 | 0.915 |
| layer3.1 | 0.793 | 0.830 | 0.101 | 0.810 | 0.507 | 0.752 | 0.602 | 0.134 | 0.806 |
| layer3.2 | 0.887 | 0.923 | 0.100 | 0.827 | 0.507 | 0.683 | 0.588 | 0.126 | 0.759 |
| penult | 0.887 | 0.923 | 0.100 | 0.827 | 0.507 | 0.683 | 0.588 | 0.126 | 0.759 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | -0.052 | -0.334 | +0.000 | -0.005 | +58.867 | +0.020 | +0.012 | +0.003 | · | · | · | · | · |
| layer1.0 | +0.000 | +0.261 | +1.000 | +0.016 | -15.251 | -0.020 | +0.012 | +0.002 | · | · | · | · | · |
| layer1.1 | -0.350 | -0.227 | +0.000 | +0.017 | +11.783 | +0.038 | +0.017 | -0.006 | · | · | · | · | · |
| layer1.2 | -0.207 | -0.232 | +0.000 | +0.006 | -4.262 | +0.001 | -0.103 | -0.009 | · | · | · | · | · |
| layer2.0 | -0.389 | -0.965 | -1.000 | +0.003 | +2.683 | +0.013 | +0.013 | -0.019 | · | · | · | · | · |
| layer2.1 | -0.268 | -0.797 | +0.000 | -0.018 | +3.568 | +0.007 | -0.030 | -0.035 | · | · | · | · | · |
| layer2.2 | -0.029 | -0.447 | -1.000 | -0.018 | +1.755 | -0.016 | +0.178 | -0.021 | · | · | · | · | · |
| layer3.0 | -0.121 | -0.604 | +0.000 | -0.024 | +0.325 | +0.047 | +0.041 | -0.009 | · | · | · | · | · |
| layer3.1 | +0.437 | +2.142 | +1.000 | -0.079 | +0.164 | +0.001 | -0.152 | -0.007 | · | · | · | · | · |
| layer3.2 | -0.079 | -0.058 | +0.000 | -0.026 | +0.002 | +0.002 | +0.009 | +0.000 | · | · | · | · | · |
| penult | -0.079 | -0.058 | +0.000 | -0.026 | +0.002 | +0.002 | +0.009 | +0.000 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer1.0 | layer1.0 |
| spectral_slope | layer2.0 | layer2.0 |
| spectral_anisotropy | layer2.0 | layer2.0 |
| noise_sigma | layer1.0 | layer1.0 |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | layer1.0 | layer1.0 |
| edge_density | layer1.0 | layer1.0 |
| orientation_entropy | layer2.0 | layer3.0 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.1 |
| coarse_animal_vehicle | layer3.0 | layer3.0 |
| corruption_family | layer2.0 | layer2.0 |
| corruption_type | layer1.2 | layer2.0 |
| severity | layer3.0 | layer3.0 |