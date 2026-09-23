# DEFORMATION  A=`results/atlas_v1_resnet20_s4_st3`  B=`results/atlas_v1_resnet56_e60`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.020 | 0.986 | 0.992 | True | 0.971 | 0.032 | 0.938 | 0.832 | 0.719 |
| layer1.0 | layer1.0 | 0.035 | 0.984 | 0.992 | True | 0.920 | 0.058 | 0.882 | 0.861 | 0.547 |
| layer1.1 | layer1.5 | 0.031 | 0.941 | 0.942 | True | 0.988 | 0.033 | 0.853 | 0.900 | 0.719 |
| layer1.2 | layer1.8 | 0.017 | 0.975 | 0.991 | True | 0.988 | 0.033 | 0.936 | 0.942 | 0.688 |
| layer2.0 | layer2.0 | 0.005 | 0.987 | 0.905 | True | 0.996 | 0.014 | 0.932 | 0.968 | 0.828 |
| layer2.1 | layer2.5 | 0.004 | 0.991 | 0.953 | True | 0.994 | 0.018 | 0.947 | 0.974 | 0.672 |
| layer2.2 | layer2.8 | 0.008 | 0.986 | 0.506 | True | 0.996 | 0.012 | 0.929 | 0.969 | 0.781 |
| layer3.0 | layer3.0 | 0.018 | 0.968 | 0.931 | True | 0.965 | 0.025 | 0.868 | 0.931 | 0.750 |
| layer3.1 | layer3.5 | 0.065 | 0.885 | 0.782 | True | 0.979 | 0.026 | 0.784 | 0.882 | 0.875 |
| layer3.2 | layer3.8 | 0.004 | 0.928 | 0.657 | True | 0.981 | 0.040 | 0.903 | 0.959 | 1.000 |
| penult | penult | 0.004 | 0.928 | 0.657 | True | 0.981 | 0.040 | 0.903 | 0.959 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.885 | 0.703 | 0.116 | 0.835 | 0.534 | 0.891 | 0.659 | 0.121 | 0.829 |
| layer1.0 | 0.903 | 0.613 | 0.120 | 0.903 | 0.534 | 0.878 | 0.583 | 0.123 | 0.884 |
| layer1.1 | 0.879 | 0.703 | 0.112 | 0.899 | 0.534 | 0.821 | 0.662 | 0.117 | 0.866 |
| layer1.2 | 0.943 | 0.767 | 0.109 | 0.945 | 0.534 | 0.928 | 0.743 | 0.121 | 0.944 |
| layer2.0 | 0.946 | 0.794 | 0.106 | 0.961 | 0.534 | 0.922 | 0.741 | 0.116 | 0.948 |
| layer2.1 | 0.948 | 0.777 | 0.104 | 0.962 | 0.534 | 0.930 | 0.726 | 0.113 | 0.951 |
| layer2.2 | 0.936 | 0.778 | 0.103 | 0.962 | 0.534 | 0.918 | 0.725 | 0.113 | 0.953 |
| layer3.0 | 0.882 | 0.780 | 0.102 | 0.928 | 0.534 | 0.856 | 0.678 | 0.112 | 0.914 |
| layer3.1 | 0.728 | 0.817 | 0.101 | 0.788 | 0.534 | 0.696 | 0.556 | 0.128 | 0.737 |
| layer3.2 | 0.879 | 0.931 | 0.100 | 0.788 | 0.534 | 0.666 | 0.555 | 0.123 | 0.751 |
| penult | 0.879 | 0.931 | 0.100 | 0.788 | 0.534 | 0.666 | 0.555 | 0.123 | 0.751 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | +0.146 | -0.118 | +0.000 | -0.002 | -13.880 | -0.029 | +0.007 | -0.006 | · | · | · | · | · |
| layer1.0 | -1.099 | -0.856 | -2.000 | -0.022 | +48.775 | +0.113 | -0.302 | -0.031 | · | · | · | · | · |
| layer1.1 | -0.189 | -0.074 | +0.000 | +0.033 | -9.277 | +0.025 | -0.076 | +0.017 | · | · | · | · | · |
| layer1.2 | -0.495 | +0.198 | +0.000 | +0.034 | -3.398 | +0.018 | -0.107 | +0.029 | · | · | · | · | · |
| layer2.0 | -1.287 | -0.612 | -2.000 | +0.028 | +4.937 | -0.008 | -0.207 | +0.006 | · | · | · | · | · |
| layer2.1 | -0.788 | -0.616 | -2.000 | +0.051 | -2.370 | -0.009 | -0.248 | +0.023 | · | · | · | · | · |
| layer2.2 | -0.784 | -1.520 | -3.000 | +0.035 | -1.637 | +0.022 | -0.173 | +0.001 | · | · | · | · | · |
| layer3.0 | -3.218 | -2.424 | -16.000 | -0.039 | +1.477 | +0.019 | -0.645 | -0.060 | · | · | · | · | · |
| layer3.1 | -0.073 | +4.402 | +2.000 | -0.049 | +0.045 | -0.050 | +0.136 | -0.033 | · | · | · | · | · |
| layer3.2 | +0.865 | +0.369 | +0.000 | +0.344 | -0.039 | -0.015 | +0.071 | -0.002 | · | · | · | · | · |
| penult | +0.865 | +0.369 | +0.000 | +0.344 | -0.039 | -0.015 | +0.071 | -0.002 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer2.0 | layer1.5 |
| spectral_slope | layer2.0 | layer1.5 |
| spectral_anisotropy | layer2.0 | layer2.5 |
| noise_sigma | layer1.0 | layer1.5 |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | stem | layer1.0 |
| edge_density | layer1.0 | layer1.0 |
| orientation_entropy | layer2.0 | layer2.5 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.1 | layer3.5 |
| coarse_animal_vehicle | layer3.0 | layer3.5 |
| corruption_family | layer2.0 | layer2.0 |
| corruption_type | layer2.0 | layer2.0 |
| severity | layer2.0 | layer3.0 |