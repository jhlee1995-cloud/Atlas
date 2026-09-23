# ATLAS — atlas_v1_resnet56_rand
built 2026-09-23 06:01:19 · source **real** · arch `cifar10_resnet56` · weights `file:/workspace/models/resnet56_rand99_chenyaofo.pt sha256:8e5b8ede1c75a62d` · layers 11

## 1. Per-layer invariants
| layer | D | PR | eff.rank | dim95 | TwoNN ID | sep ratio | NC1 | ETF dev | hub skew | test sparse | NC acc(test) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | 16 | 2.0 | 2.8 | 4 | 6.4 | 0.17 | 410.59 | 0.730 | 0.29 | 0.052 | 0.216 |
| layer1.0 | 16 | 1.9 | 3.0 | 5 | 7.6 | 0.17 | 295.54 | 0.657 | 0.47 | 0.054 | 0.217 |
| layer1.5 | 16 | 1.6 | 2.5 | 5 | 8.8 | 0.17 | 330.35 | 0.694 | 0.68 | 0.055 | 0.213 |
| layer1.8 | 16 | 1.6 | 2.3 | 4 | 8.7 | 0.16 | 359.44 | 0.642 | 0.67 | 0.053 | 0.198 |
| layer2.0 | 32 | 2.8 | 5.4 | 12 | 13.7 | 0.16 | 113.93 | 0.548 | 1.39 | 0.051 | 0.227 |
| layer2.5 | 32 | 1.5 | 2.5 | 8 | 15.2 | 0.14 | 113.48 | 0.601 | 1.71 | 0.050 | 0.182 |
| layer2.8 | 32 | 1.5 | 2.7 | 9 | 15.9 | 0.14 | 115.32 | 0.597 | 1.76 | 0.051 | 0.184 |
| layer3.0 | 64 | 2.9 | 9.2 | 41 | 28.3 | 0.13 | 88.41 | 0.597 | 4.16 | 0.051 | 0.184 |
| layer3.5 | 64 | 2.0 | 5.3 | 33 | 28.2 | 0.09 | 127.82 | 0.706 | 3.06 | 0.048 | 0.160 |
| layer3.8 | 64 | 1.7 | 4.1 | 28 | 28.3 | 0.08 | 151.37 | 0.793 | 3.14 | 0.052 | 0.157 |
| penult | 64 | 1.7 | 4.1 | 28 | 28.3 | 0.08 | 151.37 | 0.793 | 3.14 | 0.052 | 0.157 |

## 2. Decodability profile (which factor lives where)
categorical = accuracy minus majority-class rate; continuous = CV R²

| factor | kind | stem | layer1.0 | layer1.5 | layer1.8 | layer2.0 | layer2.5 | layer2.8 | layer3.0 | layer3.5 | layer3.8 | penult | commit | peak | washout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| luminance_mean | continuous | 1.00 | 0.98 | 0.94 | 0.90 | 0.90 | 0.83 | 0.78 | 0.68 | 0.57 | 0.53 | 0.53 | stem | stem | 0.47 |
| contrast_rms | continuous | 0.66 | 0.56 | 0.56 | 0.54 | 0.59 | 0.51 | 0.50 | 0.45 | 0.42 | 0.41 | 0.41 | stem | stem | 0.25 |
| highfreq_ratio | continuous | 0.33 | 0.32 | 0.34 | 0.26 | 0.31 | 0.19 | 0.15 | 0.12 | 0.07 | 0.07 | 0.07 | stem | layer1.5 | 0.28 |
| spectral_slope | continuous | 0.21 | 0.21 | 0.22 | 0.17 | 0.19 | 0.11 | 0.08 | 0.04 | 0.03 | 0.03 | 0.03 | stem | layer1.5 | 0.19 |
| spectral_anisotropy | continuous | 0.20 | 0.12 | 0.14 | 0.11 | 0.17 | 0.12 | 0.13 | 0.10 | 0.08 | 0.07 | 0.07 | stem | stem | 0.13 |
| noise_sigma | continuous | 0.64 | 0.64 | 0.73 | 0.72 | 0.75 | 0.66 | 0.61 | 0.54 | 0.45 | 0.43 | 0.43 | layer1.5 | layer2.0 | 0.32 |
| saturation_mean | continuous | 0.84 | 0.82 | 0.61 | 0.48 | 0.53 | 0.40 | 0.36 | 0.28 | 0.20 | 0.19 | 0.19 | stem | stem | 0.65 |
| hue_cos | continuous | 0.73 | 0.80 | 0.66 | 0.64 | 0.59 | 0.50 | 0.47 | 0.34 | 0.22 | 0.17 | 0.17 | stem | layer1.0 | 0.64 |
| hue_sin | continuous | 0.75 | 0.68 | 0.49 | 0.40 | 0.40 | 0.31 | 0.31 | 0.23 | 0.15 | 0.13 | 0.13 | stem | stem | 0.62 |
| colorfulness | continuous | 0.76 | 0.71 | 0.58 | 0.51 | 0.51 | 0.42 | 0.38 | 0.32 | 0.23 | 0.21 | 0.21 | stem | stem | 0.55 |
| edge_density | continuous | 0.75 | 0.63 | 0.74 | 0.75 | 0.77 | 0.69 | 0.65 | 0.57 | 0.50 | 0.47 | 0.47 | stem | layer2.0 | 0.30 |
| orientation_entropy | continuous | 0.19 | 0.15 | 0.16 | 0.12 | 0.14 | 0.11 | 0.09 | 0.09 | 0.06 | 0.04 | 0.04 | stem | stem | 0.14 |
| blockiness | continuous | 0.01 | 0.00 | 0.01 | 0.00 | 0.00 | 0.00 | -0.00 | -0.00 | -0.00 | -0.00 | -0.00 | layer1.5 | layer1.5 | 0.01 |
| class | categorical | 0.22 | 0.22 | 0.20 | 0.19 | 0.21 | 0.17 | 0.17 | 0.11 | 0.09 | 0.09 | 0.09 | stem | layer1.0 | 0.14 |
| coarse_animal_vehicle | categorical | 0.20 | 0.19 | 0.19 | 0.16 | 0.18 | 0.12 | 0.12 | 0.09 | 0.08 | 0.06 | 0.06 | stem | stem | 0.15 |
| corruption_family | categorical | 0.11 | 0.14 | 0.15 | 0.12 | 0.13 | 0.11 | 0.08 | 0.05 | 0.02 | 0.03 | 0.03 | layer1.5 | layer1.5 | 0.13 |
| corruption_type | categorical | 0.14 | 0.16 | 0.16 | 0.15 | 0.16 | 0.12 | 0.11 | 0.09 | 0.07 | 0.07 | 0.07 | layer1.0 | layer1.0 | 0.09 |
| severity | continuous | 0.08 | 0.02 | 0.06 | 0.05 | 0.07 | 0.05 | 0.05 | 0.04 | 0.02 | 0.02 | 0.02 | stem | stem | 0.05 |

## 3. Layer flow (CKA)
| pair | CKA |
|---|---|
| stem->layer1.0 | 0.924 |
| layer1.0->layer1.5 | 0.928 |
| layer1.5->layer1.8 | 0.975 |
| layer1.8->layer2.0 | 0.931 |
| layer2.0->layer2.5 | 0.939 |
| layer2.5->layer2.8 | 0.995 |
| layer2.8->layer3.0 | 0.977 |
| layer3.0->layer3.5 | 0.922 |
| layer3.5->layer3.8 | 0.987 |
| layer3.8->penult | 1.000 |

biggest reorganization: `layer3.0->layer3.5` (drop 0.078)

## 4. Sensitivity field (paired corruption displacement)
layer `penult`; magnitude in within-class-radius units

| corruption | sev | magnitude | per-sample | top5 frac | class-sub frac | coherence | class consistency | norm ratio |
|---|---|---|---|---|---|---|---|---|
| brightness | 1 | 0.14 | 0.63 | 0.99 | 0.99 | 0.19 | 0.87 | 1.021 |
| brightness | 3 | 0.49 | 0.94 | 1.00 | 0.99 | 0.43 | 0.97 | 1.072 |
| brightness | 5 | 0.96 | 1.35 | 1.00 | 0.99 | 0.62 | 0.99 | 1.146 |
| contrast | 1 | 0.50 | 0.82 | 1.00 | 1.00 | 0.49 | 0.99 | 0.932 |
| contrast | 3 | 0.72 | 1.04 | 0.99 | 0.99 | 0.59 | 0.98 | 0.908 |
| contrast | 5 | 0.80 | 1.17 | 0.96 | 0.94 | 0.61 | 0.97 | 0.905 |
| defocus_blur | 1 | 0.10 | 0.41 | 0.99 | 0.99 | 0.19 | 0.89 | 0.986 |
| defocus_blur | 3 | 0.29 | 0.66 | 0.99 | 0.99 | 0.35 | 0.97 | 0.959 |
| defocus_blur | 5 | 0.42 | 0.78 | 0.99 | 1.00 | 0.43 | 0.97 | 0.942 |
| elastic_transform | 1 | 0.19 | 0.69 | 0.99 | 0.99 | 0.22 | 0.92 | 0.972 |
| elastic_transform | 3 | 0.27 | 0.71 | 0.99 | 1.00 | 0.30 | 0.95 | 0.961 |
| elastic_transform | 5 | 0.27 | 0.71 | 0.99 | 0.99 | 0.30 | 0.95 | 0.961 |
| fog | 1 | 0.47 | 0.80 | 1.00 | 1.00 | 0.47 | 0.99 | 0.936 |
| fog | 3 | 0.69 | 1.00 | 0.99 | 1.00 | 0.57 | 0.99 | 0.910 |
| fog | 5 | 0.73 | 1.05 | 0.99 | 0.99 | 0.58 | 0.98 | 0.906 |
| frost | 1 | 0.36 | 0.90 | 0.99 | 0.99 | 0.34 | 0.77 | 1.052 |
| frost | 3 | 0.50 | 1.04 | 0.99 | 0.99 | 0.42 | 0.97 | 1.067 |
| frost | 5 | 0.11 | 0.86 | 0.87 | 0.89 | 0.15 | 0.43 | 1.001 |
| gaussian_noise | 1 | 0.08 | 0.64 | 0.95 | 0.97 | 0.11 | 0.74 | 1.011 |
| gaussian_noise | 3 | 0.34 | 0.79 | 0.99 | 0.99 | 0.38 | 0.96 | 1.047 |
| gaussian_noise | 5 | 0.56 | 0.93 | 0.99 | 1.00 | 0.54 | 0.98 | 1.081 |
| glass_blur | 1 | 0.03 | 0.67 | 0.72 | 0.77 | 0.04 | 0.16 | 1.003 |
| glass_blur | 3 | 0.17 | 0.68 | 0.99 | 0.99 | 0.19 | 0.92 | 0.975 |
| glass_blur | 5 | 0.16 | 0.69 | 0.98 | 0.99 | 0.17 | 0.90 | 0.977 |
| impulse_noise | 1 | 0.22 | 0.70 | 0.99 | 1.00 | 0.25 | 0.94 | 1.034 |
| impulse_noise | 3 | 0.71 | 1.05 | 1.00 | 0.99 | 0.56 | 0.99 | 1.113 |
| impulse_noise | 5 | 1.84 | 2.05 | 1.00 | 0.99 | 0.84 | 1.00 | 1.312 |
| jpeg_compression | 1 | 0.03 | 0.59 | 0.86 | 0.83 | 0.04 | 0.36 | 0.995 |
| jpeg_compression | 3 | 0.05 | 0.62 | 0.94 | 0.94 | 0.06 | 0.55 | 0.992 |
| jpeg_compression | 5 | 0.08 | 0.64 | 0.97 | 0.97 | 0.09 | 0.66 | 0.988 |
| motion_blur | 1 | 0.23 | 0.66 | 0.99 | 0.99 | 0.28 | 0.95 | 0.967 |
| motion_blur | 3 | 0.37 | 0.76 | 0.99 | 1.00 | 0.39 | 0.97 | 0.948 |
| motion_blur | 5 | 0.43 | 0.80 | 0.99 | 1.00 | 0.43 | 0.97 | 0.940 |
| pixelate | 1 | 0.04 | 0.53 | 0.92 | 0.93 | 0.05 | 0.42 | 0.995 |
| pixelate | 3 | 0.08 | 0.60 | 0.97 | 0.97 | 0.10 | 0.69 | 0.987 |
| pixelate | 5 | 0.15 | 0.65 | 0.99 | 0.99 | 0.18 | 0.89 | 0.978 |
| shot_noise | 1 | 0.05 | 0.61 | 0.91 | 0.92 | 0.07 | 0.52 | 1.007 |
| shot_noise | 3 | 0.28 | 0.75 | 0.99 | 1.00 | 0.32 | 0.96 | 1.041 |
| shot_noise | 5 | 0.58 | 0.96 | 0.99 | 1.00 | 0.53 | 0.99 | 1.087 |
| snow | 1 | 0.16 | 0.69 | 0.99 | 0.99 | 0.21 | 0.62 | 1.024 |
| snow | 3 | 0.49 | 0.94 | 1.00 | 1.00 | 0.46 | 0.97 | 1.072 |
| snow | 5 | 1.04 | 1.41 | 1.00 | 0.99 | 0.66 | 0.99 | 1.157 |
| zoom_blur | 1 | 0.27 | 0.68 | 0.99 | 0.99 | 0.31 | 0.96 | 0.961 |
| zoom_blur | 3 | 0.34 | 0.73 | 0.99 | 1.00 | 0.37 | 0.97 | 0.952 |
| zoom_blur | 5 | 0.40 | 0.78 | 0.99 | 1.00 | 0.41 | 0.97 | 0.943 |

## 5. Density per split
layer `penult`

| split | sparse frac (ref q95) | median log-radius shift |
|---|---|---|
| corrupt__brightness__s1 | 0.066 | 0.024 |
| corrupt__brightness__s3 | 0.117 | 0.114 |
| corrupt__brightness__s5 | 0.203 | 0.214 |
| corrupt__contrast__s1 | 0.007 | -0.089 |
| corrupt__contrast__s3 | 0.004 | -0.083 |
| corrupt__contrast__s5 | 0.039 | 0.075 |
| corrupt__defocus_blur__s1 | 0.032 | -0.013 |
| corrupt__defocus_blur__s3 | 0.016 | -0.048 |
| corrupt__defocus_blur__s5 | 0.011 | -0.062 |
| corrupt__elastic_transform__s1 | 0.028 | -0.026 |
| corrupt__elastic_transform__s3 | 0.017 | -0.046 |
| corrupt__elastic_transform__s5 | 0.019 | -0.044 |
| corrupt__fog__s1 | 0.003 | -0.078 |
| corrupt__fog__s3 | 0.000 | -0.113 |
| corrupt__fog__s5 | 0.000 | -0.118 |
| corrupt__frost__s1 | 0.095 | 0.086 |
| corrupt__frost__s3 | 0.107 | 0.130 |
| corrupt__frost__s5 | 0.035 | 0.033 |
| corrupt__gaussian_noise__s1 | 0.055 | 0.007 |
| corrupt__gaussian_noise__s3 | 0.073 | 0.067 |
| corrupt__gaussian_noise__s5 | 0.111 | 0.114 |
| corrupt__glass_blur__s1 | 0.051 | 0.004 |
| corrupt__glass_blur__s3 | 0.026 | -0.026 |
| corrupt__glass_blur__s5 | 0.028 | -0.024 |
| corrupt__impulse_noise__s1 | 0.069 | 0.027 |
| corrupt__impulse_noise__s3 | 0.160 | 0.120 |
| corrupt__impulse_noise__s5 | 0.367 | 0.295 |
| corrupt__jpeg_compression__s1 | 0.039 | -0.002 |
| corrupt__jpeg_compression__s3 | 0.036 | -0.007 |
| corrupt__jpeg_compression__s5 | 0.033 | -0.013 |
| corrupt__motion_blur__s1 | 0.025 | -0.037 |
| corrupt__motion_blur__s3 | 0.011 | -0.051 |
| corrupt__motion_blur__s5 | 0.010 | -0.062 |
| corrupt__pixelate__s1 | 0.041 | -0.001 |
| corrupt__pixelate__s3 | 0.034 | -0.008 |
| corrupt__pixelate__s5 | 0.029 | -0.020 |
| corrupt__shot_noise__s1 | 0.049 | 0.009 |
| corrupt__shot_noise__s3 | 0.083 | 0.052 |
| corrupt__shot_noise__s5 | 0.121 | 0.119 |
| corrupt__snow__s1 | 0.063 | 0.030 |
| corrupt__snow__s3 | 0.108 | 0.106 |
| corrupt__snow__s5 | 0.234 | 0.234 |
| corrupt__zoom_blur__s1 | 0.018 | -0.044 |
| corrupt__zoom_blur__s3 | 0.011 | -0.052 |
| corrupt__zoom_blur__s5 | 0.009 | -0.060 |
| ood__cifar100 | 0.102 | 0.019 |
| ood__svhn | 0.030 | -0.041 |
| panel | 0.047 | -0.003 |
| test | 0.052 | -0.000 |

## 6. Adjacency
layer `penult`


closest class pairs (sep = center distance / RMS radius): (5,7) 0.06, (2,6) 0.06, (1,9) 0.08, (0,1) 0.09, (3,5) 0.10, (0,7) 0.11

valley ratio mean 1.00, min 0.98, pairs with a valley 0.33

nearest-center confusions (true → nearest): 2→4 0.66, 6→4 0.64, 3→4 0.49, 8→4 0.47, 5→4 0.43, 0→4 0.42

single-linkage merge order (first 5): [5, 7]@0.06 ; [2, 6]@0.06 ; [1, 9]@0.08 ; [0, 1, 9]@0.09 ; [3, 5, 7]@0.10

## 7. Type-b margin (margin_typeb)
_margin_typeb not run_
