# ATLAS — atlas_v1_resnet56_e10
built 2026-09-23 05:57:15 · source **real** · arch `cifar10_resnet56` · weights `file:/workspace/models/resnet56_s11_e10_chenyaofo.pt sha256:3fc878f9a9565fc7` · layers 11

## 1. Per-layer invariants
| layer | D | PR | eff.rank | dim95 | TwoNN ID | sep ratio | NC1 | ETF dev | hub skew | test sparse | NC acc(test) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | 16 | 2.5 | 3.3 | 4 | 6.5 | 0.16 | 456.74 | 0.700 | 0.25 | 0.060 | 0.255 |
| layer1.0 | 16 | 3.0 | 4.1 | 5 | 7.4 | 0.17 | 462.85 | 0.640 | 0.41 | 0.059 | 0.267 |
| layer1.5 | 16 | 4.1 | 5.8 | 8 | 9.4 | 0.26 | 106.41 | 0.470 | 0.71 | 0.045 | 0.357 |
| layer1.8 | 16 | 4.9 | 6.5 | 8 | 9.2 | 0.38 | 70.94 | 0.443 | 0.70 | 0.051 | 0.405 |
| layer2.0 | 32 | 4.7 | 7.0 | 9 | 10.9 | 0.41 | 23.32 | 0.401 | 0.83 | 0.052 | 0.442 |
| layer2.5 | 32 | 5.9 | 8.7 | 12 | 12.4 | 0.53 | 10.21 | 0.386 | 1.09 | 0.048 | 0.541 |
| layer2.8 | 32 | 6.1 | 9.0 | 12 | 12.4 | 0.60 | 6.87 | 0.338 | 1.05 | 0.043 | 0.584 |
| layer3.0 | 64 | 6.1 | 9.9 | 15 | 14.1 | 0.70 | 3.49 | 0.385 | 1.14 | 0.048 | 0.648 |
| layer3.5 | 64 | 7.9 | 13.3 | 21 | 15.5 | 0.98 | 1.25 | 0.314 | 1.48 | 0.048 | 0.784 |
| layer3.8 | 64 | 6.4 | 9.6 | 10 | 12.0 | 1.45 | 0.69 | 0.278 | 0.88 | 0.049 | 0.832 |
| penult | 64 | 6.4 | 9.6 | 10 | 12.0 | 1.45 | 0.69 | 0.278 | 0.88 | 0.049 | 0.832 |

## 2. Decodability profile (which factor lives where)
categorical = accuracy minus majority-class rate; continuous = CV R²

| factor | kind | stem | layer1.0 | layer1.5 | layer1.8 | layer2.0 | layer2.5 | layer2.8 | layer3.0 | layer3.5 | layer3.8 | penult | commit | peak | washout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| luminance_mean | continuous | 1.00 | 0.99 | 0.93 | 0.91 | 0.88 | 0.73 | 0.72 | 0.60 | 0.44 | 0.33 | 0.33 | stem | stem | 0.67 |
| contrast_rms | continuous | 0.65 | 0.62 | 0.61 | 0.57 | 0.57 | 0.55 | 0.55 | 0.53 | 0.48 | 0.46 | 0.46 | stem | stem | 0.19 |
| highfreq_ratio | continuous | 0.23 | 0.21 | 0.58 | 0.56 | 0.57 | 0.51 | 0.51 | 0.54 | 0.49 | 0.46 | 0.46 | layer1.5 | layer1.5 | 0.12 |
| spectral_slope | continuous | 0.12 | 0.13 | 0.55 | 0.54 | 0.59 | 0.58 | 0.57 | 0.58 | 0.54 | 0.51 | 0.51 | layer1.5 | layer2.0 | 0.09 |
| spectral_anisotropy | continuous | 0.20 | 0.17 | 0.31 | 0.28 | 0.31 | 0.33 | 0.33 | 0.34 | 0.33 | 0.33 | 0.33 | layer1.5 | layer3.0 | 0.02 |
| noise_sigma | continuous | 0.66 | 0.74 | 0.94 | 0.93 | 0.93 | 0.90 | 0.88 | 0.87 | 0.83 | 0.79 | 0.79 | layer1.5 | layer1.5 | 0.14 |
| saturation_mean | continuous | 0.77 | 0.86 | 0.63 | 0.55 | 0.60 | 0.52 | 0.51 | 0.45 | 0.32 | 0.28 | 0.28 | layer1.0 | layer1.0 | 0.58 |
| hue_cos | continuous | 0.77 | 0.83 | 0.81 | 0.79 | 0.74 | 0.66 | 0.63 | 0.62 | 0.55 | 0.49 | 0.49 | stem | layer1.0 | 0.34 |
| hue_sin | continuous | 0.76 | 0.81 | 0.78 | 0.75 | 0.73 | 0.66 | 0.65 | 0.63 | 0.59 | 0.55 | 0.55 | stem | layer1.0 | 0.26 |
| colorfulness | continuous | 0.81 | 0.80 | 0.70 | 0.61 | 0.57 | 0.50 | 0.47 | 0.40 | 0.29 | 0.23 | 0.23 | stem | stem | 0.58 |
| edge_density | continuous | 0.75 | 0.79 | 0.86 | 0.85 | 0.87 | 0.84 | 0.82 | 0.81 | 0.78 | 0.73 | 0.73 | layer1.0 | layer2.0 | 0.14 |
| orientation_entropy | continuous | 0.19 | 0.15 | 0.40 | 0.39 | 0.47 | 0.43 | 0.44 | 0.50 | 0.49 | 0.48 | 0.48 | layer2.0 | layer3.0 | 0.02 |
| blockiness | continuous | 0.01 | 0.02 | 0.02 | 0.03 | 0.04 | 0.08 | 0.08 | 0.10 | 0.06 | 0.05 | 0.05 | layer3.0 | layer3.0 | 0.06 |
| class | categorical | 0.23 | 0.26 | 0.36 | 0.40 | 0.48 | 0.58 | 0.60 | 0.65 | 0.71 | 0.72 | 0.72 | layer3.0 | layer3.8 | 0.00 |
| coarse_animal_vehicle | categorical | 0.22 | 0.24 | 0.27 | 0.29 | 0.31 | 0.34 | 0.35 | 0.36 | 0.37 | 0.38 | 0.38 | layer2.5 | layer3.8 | 0.00 |
| corruption_family | categorical | 0.10 | 0.12 | 0.25 | 0.25 | 0.29 | 0.28 | 0.27 | 0.26 | 0.23 | 0.20 | 0.20 | layer2.0 | layer2.0 | 0.09 |
| corruption_type | categorical | 0.14 | 0.16 | 0.34 | 0.35 | 0.38 | 0.36 | 0.35 | 0.34 | 0.29 | 0.24 | 0.24 | layer1.8 | layer2.0 | 0.14 |
| severity | continuous | 0.04 | 0.05 | 0.10 | 0.11 | 0.23 | 0.23 | 0.23 | 0.22 | 0.21 | 0.19 | 0.19 | layer2.0 | layer2.5 | 0.04 |

## 3. Layer flow (CKA)
| pair | CKA |
|---|---|
| stem->layer1.0 | 0.966 |
| layer1.0->layer1.5 | 0.770 |
| layer1.5->layer1.8 | 0.913 |
| layer1.8->layer2.0 | 0.938 |
| layer2.0->layer2.5 | 0.843 |
| layer2.5->layer2.8 | 0.970 |
| layer2.8->layer3.0 | 0.929 |
| layer3.0->layer3.5 | 0.886 |
| layer3.5->layer3.8 | 0.908 |
| layer3.8->penult | 1.000 |

biggest reorganization: `layer1.0->layer1.5` (drop 0.230)

## 4. Sensitivity field (paired corruption displacement)
layer `penult`; magnitude in within-class-radius units

| corruption | sev | magnitude | per-sample | top5 frac | class-sub frac | coherence | class consistency | norm ratio |
|---|---|---|---|---|---|---|---|---|
| brightness | 1 | 0.04 | 0.11 | 0.85 | 0.88 | 0.34 | 0.74 | 1.002 |
| brightness | 3 | 0.11 | 0.31 | 0.87 | 0.93 | 0.35 | 0.66 | 0.996 |
| brightness | 5 | 0.22 | 0.59 | 0.85 | 0.95 | 0.36 | 0.47 | 0.966 |
| contrast | 1 | 0.17 | 0.30 | 0.72 | 0.85 | 0.54 | 0.78 | 0.980 |
| contrast | 3 | 0.56 | 0.98 | 0.63 | 0.77 | 0.55 | 0.57 | 0.877 |
| contrast | 5 | 1.08 | 1.83 | 0.45 | 0.62 | 0.57 | 0.42 | 0.719 |
| defocus_blur | 1 | 0.09 | 0.15 | 0.62 | 0.87 | 0.56 | 0.80 | 0.997 |
| defocus_blur | 3 | 0.50 | 0.76 | 0.61 | 0.91 | 0.63 | 0.71 | 0.943 |
| defocus_blur | 5 | 0.91 | 1.43 | 0.53 | 0.89 | 0.61 | 0.58 | 0.853 |
| elastic_transform | 1 | 0.27 | 0.67 | 0.64 | 0.88 | 0.38 | 0.52 | 0.935 |
| elastic_transform | 3 | 0.46 | 0.84 | 0.62 | 0.91 | 0.52 | 0.61 | 0.916 |
| elastic_transform | 5 | 0.47 | 1.02 | 0.86 | 0.94 | 0.40 | 0.34 | 0.883 |
| fog | 1 | 0.10 | 0.21 | 0.73 | 0.85 | 0.48 | 0.78 | 0.990 |
| fog | 3 | 0.33 | 0.62 | 0.68 | 0.83 | 0.52 | 0.64 | 0.931 |
| fog | 5 | 0.63 | 1.14 | 0.68 | 0.82 | 0.53 | 0.49 | 0.842 |
| frost | 1 | 0.25 | 0.53 | 0.60 | 0.92 | 0.43 | 0.61 | 0.963 |
| frost | 3 | 0.53 | 1.03 | 0.66 | 0.90 | 0.49 | 0.49 | 0.900 |
| frost | 5 | 0.71 | 1.26 | 0.67 | 0.87 | 0.54 | 0.51 | 0.866 |
| gaussian_noise | 1 | 0.47 | 0.81 | 0.51 | 0.83 | 0.56 | 0.59 | 0.917 |
| gaussian_noise | 3 | 0.91 | 1.41 | 0.60 | 0.80 | 0.63 | 0.58 | 0.844 |
| gaussian_noise | 5 | 1.04 | 1.60 | 0.61 | 0.80 | 0.64 | 0.57 | 0.831 |
| glass_blur | 1 | 0.84 | 1.43 | 0.72 | 0.94 | 0.55 | 0.52 | 0.887 |
| glass_blur | 3 | 0.73 | 1.32 | 0.71 | 0.95 | 0.52 | 0.49 | 0.889 |
| glass_blur | 5 | 0.86 | 1.52 | 0.72 | 0.95 | 0.54 | 0.47 | 0.883 |
| impulse_noise | 1 | 0.27 | 0.60 | 0.47 | 0.79 | 0.43 | 0.47 | 0.954 |
| impulse_noise | 3 | 0.62 | 1.14 | 0.50 | 0.78 | 0.53 | 0.47 | 0.885 |
| impulse_noise | 5 | 0.98 | 1.64 | 0.56 | 0.75 | 0.59 | 0.47 | 0.820 |
| jpeg_compression | 1 | 0.21 | 0.54 | 0.49 | 0.93 | 0.36 | 0.50 | 0.969 |
| jpeg_compression | 3 | 0.24 | 0.72 | 0.63 | 0.91 | 0.31 | 0.31 | 0.952 |
| jpeg_compression | 5 | 0.28 | 0.83 | 0.73 | 0.90 | 0.32 | 0.26 | 0.937 |
| motion_blur | 1 | 0.41 | 0.71 | 0.63 | 0.92 | 0.55 | 0.65 | 0.933 |
| motion_blur | 3 | 0.68 | 1.18 | 0.57 | 0.88 | 0.55 | 0.56 | 0.849 |
| motion_blur | 5 | 0.74 | 1.31 | 0.55 | 0.86 | 0.54 | 0.52 | 0.821 |
| pixelate | 1 | 0.23 | 0.46 | 0.66 | 0.92 | 0.47 | 0.70 | 0.984 |
| pixelate | 3 | 0.60 | 0.94 | 0.71 | 0.92 | 0.60 | 0.67 | 0.964 |
| pixelate | 5 | 1.11 | 1.66 | 0.60 | 0.83 | 0.65 | 0.58 | 0.922 |
| shot_noise | 1 | 0.35 | 0.61 | 0.44 | 0.86 | 0.55 | 0.64 | 0.942 |
| shot_noise | 3 | 0.82 | 1.26 | 0.57 | 0.78 | 0.63 | 0.61 | 0.848 |
| shot_noise | 5 | 1.02 | 1.54 | 0.59 | 0.78 | 0.65 | 0.59 | 0.823 |
| snow | 1 | 0.21 | 0.60 | 0.34 | 0.81 | 0.33 | 0.38 | 0.964 |
| snow | 3 | 0.33 | 0.92 | 0.33 | 0.74 | 0.35 | 0.31 | 0.919 |
| snow | 5 | 0.44 | 1.12 | 0.49 | 0.82 | 0.38 | 0.34 | 0.899 |
| zoom_blur | 1 | 0.51 | 0.83 | 0.68 | 0.91 | 0.58 | 0.65 | 0.914 |
| zoom_blur | 3 | 0.72 | 1.12 | 0.62 | 0.92 | 0.61 | 0.64 | 0.889 |
| zoom_blur | 5 | 0.90 | 1.40 | 0.56 | 0.90 | 0.61 | 0.60 | 0.853 |

## 5. Density per split
layer `penult`

| split | sparse frac (ref q95) | median log-radius shift |
|---|---|---|
| corrupt__brightness__s1 | 0.047 | -0.004 |
| corrupt__brightness__s3 | 0.048 | 0.003 |
| corrupt__brightness__s5 | 0.058 | 0.020 |
| corrupt__contrast__s1 | 0.041 | -0.009 |
| corrupt__contrast__s3 | 0.050 | 0.041 |
| corrupt__contrast__s5 | 0.232 | 0.192 |
| corrupt__defocus_blur__s1 | 0.043 | -0.006 |
| corrupt__defocus_blur__s3 | 0.064 | 0.035 |
| corrupt__defocus_blur__s5 | 0.097 | 0.079 |
| corrupt__elastic_transform__s1 | 0.039 | 0.008 |
| corrupt__elastic_transform__s3 | 0.056 | 0.026 |
| corrupt__elastic_transform__s5 | 0.069 | 0.033 |
| corrupt__fog__s1 | 0.038 | -0.009 |
| corrupt__fog__s3 | 0.037 | 0.002 |
| corrupt__fog__s5 | 0.055 | 0.043 |
| corrupt__frost__s1 | 0.058 | 0.016 |
| corrupt__frost__s3 | 0.121 | 0.064 |
| corrupt__frost__s5 | 0.167 | 0.090 |
| corrupt__gaussian_noise__s1 | 0.108 | 0.041 |
| corrupt__gaussian_noise__s3 | 0.244 | 0.113 |
| corrupt__gaussian_noise__s5 | 0.287 | 0.121 |
| corrupt__glass_blur__s1 | 0.253 | 0.108 |
| corrupt__glass_blur__s3 | 0.242 | 0.097 |
| corrupt__glass_blur__s5 | 0.271 | 0.122 |
| corrupt__impulse_noise__s1 | 0.068 | 0.033 |
| corrupt__impulse_noise__s3 | 0.169 | 0.108 |
| corrupt__impulse_noise__s5 | 0.352 | 0.182 |
| corrupt__jpeg_compression__s1 | 0.075 | 0.020 |
| corrupt__jpeg_compression__s3 | 0.085 | 0.041 |
| corrupt__jpeg_compression__s5 | 0.091 | 0.049 |
| corrupt__motion_blur__s1 | 0.057 | 0.023 |
| corrupt__motion_blur__s3 | 0.059 | 0.058 |
| corrupt__motion_blur__s5 | 0.052 | 0.067 |
| corrupt__pixelate__s1 | 0.069 | 0.026 |
| corrupt__pixelate__s3 | 0.184 | 0.108 |
| corrupt__pixelate__s5 | 0.527 | 0.248 |
| corrupt__shot_noise__s1 | 0.073 | 0.015 |
| corrupt__shot_noise__s3 | 0.202 | 0.089 |
| corrupt__shot_noise__s5 | 0.256 | 0.114 |
| corrupt__snow__s1 | 0.087 | 0.033 |
| corrupt__snow__s3 | 0.134 | 0.071 |
| corrupt__snow__s5 | 0.188 | 0.101 |
| corrupt__zoom_blur__s1 | 0.069 | 0.029 |
| corrupt__zoom_blur__s3 | 0.085 | 0.062 |
| corrupt__zoom_blur__s5 | 0.084 | 0.081 |
| ood__cifar100 | 0.175 | 0.104 |
| ood__svhn | 0.172 | 0.166 |
| panel | 0.078 | 0.007 |
| test | 0.049 | 0.002 |

## 6. Adjacency
layer `penult`


closest class pairs (sep = center distance / RMS radius): (3,5) 0.81, (2,3) 1.31, (2,4) 1.37, (3,4) 1.39, (2,5) 1.44, (0,2) 1.46

valley ratio mean 1.35, min 0.99, pairs with a valley 0.98

nearest-center confusions (true → nearest): 5→3 0.18, 3→5 0.09, 2→0 0.08, 6→3 0.07, 8→0 0.07, 2→3 0.06

single-linkage merge order (first 5): [3, 5]@0.81 ; [2, 3, 5]@1.31 ; [2, 3, 4, 5]@1.37 ; [0, 2, 3, 4, 5]@1.46 ; [0, 2, 3, 4, 5, 6]@1.49

## 7. Type-b margin (margin_typeb)
_margin_typeb not run_
