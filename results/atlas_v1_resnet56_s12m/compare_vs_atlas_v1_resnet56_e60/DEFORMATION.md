# DEFORMATION  A=`results/atlas_v1_resnet56_e60`  B=`results/atlas_v1_resnet56_s12m`  same_space=False
sources: real vs real

| layer A | layer B | Procrustes disp | adjacency ρ | merge τ | first merge same | decod ρ | decod |Δ| | panel CKA | relrep corr | relrep argmax |
|---|---|---|---|---|---|---|---|---|---|---|
| stem | stem | 0.006 | 0.992 | 0.948 | True | 0.983 | 0.022 | 0.963 | 0.929 | 0.734 |
| layer1.0 | layer1.0 | 0.010 | 0.983 | 0.997 | True | 0.953 | 0.029 | 0.792 | 0.838 | 0.562 |
| layer1.5 | layer1.5 | 0.023 | 0.962 | 0.942 | True | 0.975 | 0.039 | 0.918 | 0.916 | 0.812 |
| layer1.8 | layer1.8 | 0.011 | 0.980 | 0.991 | True | 0.979 | 0.039 | 0.950 | 0.961 | 0.703 |
| layer2.0 | layer2.0 | 0.013 | 0.983 | 0.940 | True | 0.992 | 0.018 | 0.958 | 0.959 | 0.797 |
| layer2.5 | layer2.5 | 0.007 | 0.986 | 0.995 | True | 0.975 | 0.021 | 0.944 | 0.959 | 0.781 |
| layer2.8 | layer2.8 | 0.013 | 0.984 | 0.503 | True | 0.979 | 0.021 | 0.938 | 0.956 | 0.781 |
| layer3.0 | layer3.0 | 0.009 | 0.965 | 1.000 | True | 0.961 | 0.024 | 0.922 | 0.953 | 0.891 |
| layer3.5 | layer3.5 | 0.017 | 0.927 | 0.939 | True | 0.988 | 0.025 | 0.852 | 0.930 | 0.891 |
| layer3.8 | layer3.8 | 0.003 | 0.975 | 0.844 | True | 0.942 | 0.020 | 0.925 | 0.965 | 1.000 |
| penult | penult | 0.003 | 0.975 | 0.844 | True | 0.942 | 0.020 | 0.925 | 0.965 | 1.000 |

## cross-model agreement on identical inputs (test[:2000], CIFAR-100 test[:2000])
| layer | cka_test | relrep_argmax_agree_test | relrep_argmax_chance_test | relrep_offmax_corr_test | error_consistency_test | cka_ood_c100 | relrep_argmax_agree_ood_c100 | relrep_argmax_chance_ood_c100 | relrep_offmax_corr_ood_c100 |
|---|---|---|---|---|---|---|---|---|---|
| stem | 0.956 | 0.737 | 0.119 | 0.921 | 0.549 | 0.964 | 0.699 | 0.126 | 0.914 |
| layer1.0 | 0.863 | 0.633 | 0.117 | 0.863 | 0.549 | 0.787 | 0.603 | 0.121 | 0.836 |
| layer1.5 | 0.909 | 0.730 | 0.111 | 0.909 | 0.549 | 0.874 | 0.689 | 0.114 | 0.886 |
| layer1.8 | 0.951 | 0.757 | 0.109 | 0.951 | 0.549 | 0.937 | 0.730 | 0.121 | 0.941 |
| layer2.0 | 0.938 | 0.783 | 0.106 | 0.952 | 0.549 | 0.929 | 0.751 | 0.115 | 0.940 |
| layer2.5 | 0.947 | 0.786 | 0.104 | 0.959 | 0.549 | 0.934 | 0.742 | 0.111 | 0.947 |
| layer2.8 | 0.941 | 0.797 | 0.103 | 0.955 | 0.549 | 0.926 | 0.734 | 0.111 | 0.945 |
| layer3.0 | 0.937 | 0.821 | 0.102 | 0.949 | 0.549 | 0.929 | 0.739 | 0.110 | 0.946 |
| layer3.5 | 0.859 | 0.844 | 0.101 | 0.880 | 0.549 | 0.837 | 0.661 | 0.124 | 0.864 |
| layer3.8 | 0.898 | 0.926 | 0.100 | 0.786 | 0.549 | 0.684 | 0.584 | 0.122 | 0.768 |
| penult | 0.898 | 0.926 | 0.100 | 0.786 | 0.549 | 0.684 | 0.584 | 0.122 | 0.768 |

## scalar deltas (B − A)
| layer | twonn_id | pca_spectrum | pca_spectrum | class_centers | neural_collapse | neural_collapse | hubness | class_centers | margin_typeb | margin_typeb | margin_typeb | margin_typeb | margin_typeb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | -0.660 | -0.099 | +0.000 | +0.004 | +120.047 | +0.041 | -0.134 | -0.001 | · | · | · | · | · |
| layer1.0 | -0.336 | -0.212 | +0.000 | -0.006 | -23.837 | +0.051 | -0.043 | -0.006 | · | · | · | · | · |
| layer1.5 | -0.198 | +0.307 | +0.000 | -0.028 | -1.708 | -0.010 | -0.005 | -0.006 | · | · | · | · | · |
| layer1.8 | -0.001 | -0.298 | +0.000 | -0.024 | +22.017 | -0.020 | +0.017 | -0.030 | · | · | · | · | · |
| layer2.0 | +0.092 | +0.029 | -1.000 | -0.021 | +3.955 | +0.003 | -0.003 | -0.024 | · | · | · | · | · |
| layer2.5 | +0.648 | +0.406 | +1.000 | +0.008 | -0.063 | -0.011 | +0.159 | +0.012 | · | · | · | · | · |
| layer2.8 | +0.476 | +0.666 | +1.000 | +0.023 | +1.682 | -0.010 | +0.012 | +0.018 | · | · | · | · | · |
| layer3.0 | +0.529 | +0.693 | +3.000 | +0.026 | -0.379 | -0.006 | +0.201 | +0.038 | · | · | · | · | · |
| layer3.5 | -0.524 | -1.030 | -1.000 | +0.015 | +0.087 | +0.030 | -0.124 | +0.007 | · | · | · | · | · |
| layer3.8 | -0.045 | -0.005 | +0.000 | -0.001 | +0.000 | +0.003 | +0.004 | -0.003 | · | · | · | · | · |
| penult | -0.045 | -0.005 | +0.000 | -0.001 | +0.000 | +0.003 | +0.004 | -0.003 | · | · | · | · | · |

## commit layer A → B
| factor | A | B |
|---|---|---|
| luminance_mean | stem | stem |
| contrast_rms | stem | stem |
| highfreq_ratio | layer1.5 | layer2.0 |
| spectral_slope | layer1.5 | layer2.0 |
| spectral_anisotropy | layer2.5 | layer2.5 |
| noise_sigma | layer1.5 | layer1.0 |
| saturation_mean | stem | stem |
| hue_cos | stem | stem |
| hue_sin | stem | stem |
| colorfulness | layer1.0 | layer1.0 |
| edge_density | layer1.0 | layer2.0 |
| orientation_entropy | layer2.5 | layer2.5 |
| blockiness | layer3.0 | layer3.0 |
| class | layer3.5 | layer3.5 |
| coarse_animal_vehicle | layer3.5 | layer3.0 |
| corruption_family | layer2.0 | layer2.0 |
| corruption_type | layer2.0 | layer2.0 |
| severity | layer3.0 | layer2.0 |