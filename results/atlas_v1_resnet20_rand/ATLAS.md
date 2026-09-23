# ATLAS — atlas_v1_resnet20_rand
built 2026-09-23 02:15:58 · source **real** · arch `cifar10_resnet20` · weights `file:/workspace/models/resnet20_rand99_chenyaofo.pt sha256:c761ef798fb28398` · layers 11

## 1. Per-layer invariants
| layer | D | PR | eff.rank | dim95 | TwoNN ID | sep ratio | NC1 | ETF dev | hub skew | test sparse | NC acc(test) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | 16 | 2.1 | 3.0 | 4 | 6.5 | 0.17 | 352.45 | 0.691 | 0.26 | 0.056 | 0.222 |
| layer1.0 | 16 | 2.1 | 3.1 | 5 | 7.8 | 0.16 | 304.28 | 0.619 | 0.48 | 0.053 | 0.217 |
| layer1.1 | 16 | 2.5 | 4.0 | 6 | 8.4 | 0.15 | 265.15 | 0.569 | 0.58 | 0.057 | 0.215 |
| layer1.2 | 16 | 2.0 | 3.3 | 5 | 8.3 | 0.15 | 182.02 | 0.474 | 0.57 | 0.064 | 0.207 |
| layer2.0 | 32 | 2.1 | 4.0 | 9 | 11.8 | 0.15 | 94.67 | 0.526 | 1.03 | 0.059 | 0.204 |
| layer2.1 | 32 | 2.1 | 4.0 | 9 | 12.7 | 0.15 | 102.54 | 0.670 | 1.11 | 0.056 | 0.199 |
| layer2.2 | 32 | 2.4 | 4.6 | 10 | 13.3 | 0.15 | 120.40 | 0.699 | 1.23 | 0.057 | 0.198 |
| layer3.0 | 64 | 3.6 | 9.0 | 34 | 23.6 | 0.14 | 70.63 | 0.694 | 2.92 | 0.052 | 0.205 |
| layer3.1 | 64 | 3.2 | 8.0 | 33 | 24.2 | 0.13 | 74.19 | 0.760 | 2.43 | 0.053 | 0.195 |
| layer3.2 | 64 | 2.7 | 6.7 | 30 | 23.7 | 0.13 | 72.47 | 0.778 | 2.59 | 0.049 | 0.192 |
| penult | 64 | 2.7 | 6.7 | 30 | 23.7 | 0.13 | 72.47 | 0.778 | 2.59 | 0.049 | 0.192 |

## 2. Decodability profile (which factor lives where)
categorical = accuracy minus majority-class rate; continuous = CV R²

| factor | kind | stem | layer1.0 | layer1.1 | layer1.2 | layer2.0 | layer2.1 | layer2.2 | layer3.0 | layer3.1 | layer3.2 | penult | commit | peak | washout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| luminance_mean | continuous | 1.00 | 0.99 | 0.95 | 0.93 | 0.91 | 0.90 | 0.88 | 0.84 | 0.82 | 0.79 | 0.79 | stem | stem | 0.21 |
| contrast_rms | continuous | 0.71 | 0.58 | 0.54 | 0.55 | 0.63 | 0.60 | 0.58 | 0.58 | 0.58 | 0.56 | 0.56 | stem | stem | 0.15 |
| highfreq_ratio | continuous | 0.47 | 0.40 | 0.40 | 0.34 | 0.35 | 0.28 | 0.24 | 0.22 | 0.21 | 0.20 | 0.20 | stem | stem | 0.27 |
| spectral_slope | continuous | 0.35 | 0.29 | 0.30 | 0.24 | 0.26 | 0.19 | 0.15 | 0.15 | 0.13 | 0.12 | 0.12 | stem | stem | 0.24 |
| spectral_anisotropy | continuous | 0.24 | 0.13 | 0.13 | 0.14 | 0.19 | 0.18 | 0.15 | 0.13 | 0.14 | 0.15 | 0.15 | stem | stem | 0.08 |
| noise_sigma | continuous | 0.76 | 0.67 | 0.72 | 0.71 | 0.76 | 0.72 | 0.68 | 0.68 | 0.64 | 0.60 | 0.60 | stem | stem | 0.16 |
| saturation_mean | continuous | 0.85 | 0.78 | 0.69 | 0.66 | 0.67 | 0.62 | 0.56 | 0.50 | 0.46 | 0.41 | 0.41 | stem | stem | 0.44 |
| hue_cos | continuous | 0.73 | 0.75 | 0.74 | 0.72 | 0.68 | 0.67 | 0.61 | 0.50 | 0.48 | 0.43 | 0.43 | stem | layer1.0 | 0.33 |
| hue_sin | continuous | 0.74 | 0.70 | 0.59 | 0.60 | 0.61 | 0.56 | 0.51 | 0.44 | 0.38 | 0.34 | 0.34 | stem | stem | 0.40 |
| colorfulness | continuous | 0.81 | 0.67 | 0.65 | 0.64 | 0.62 | 0.54 | 0.50 | 0.46 | 0.45 | 0.43 | 0.43 | stem | stem | 0.38 |
| edge_density | continuous | 0.80 | 0.70 | 0.70 | 0.70 | 0.77 | 0.74 | 0.71 | 0.70 | 0.67 | 0.63 | 0.63 | stem | stem | 0.17 |
| orientation_entropy | continuous | 0.26 | 0.13 | 0.12 | 0.14 | 0.17 | 0.14 | 0.12 | 0.13 | 0.13 | 0.11 | 0.11 | stem | stem | 0.15 |
| blockiness | continuous | 0.01 | 0.00 | -0.00 | 0.00 | 0.01 | 0.00 | 0.00 | 0.01 | 0.01 | 0.01 | 0.01 | stem | stem | 0.00 |
| class | categorical | 0.25 | 0.21 | 0.20 | 0.20 | 0.22 | 0.22 | 0.20 | 0.17 | 0.17 | 0.15 | 0.15 | stem | stem | 0.10 |
| coarse_animal_vehicle | categorical | 0.23 | 0.20 | 0.19 | 0.18 | 0.18 | 0.18 | 0.18 | 0.16 | 0.15 | 0.14 | 0.14 | stem | stem | 0.09 |
| corruption_family | categorical | 0.17 | 0.17 | 0.16 | 0.16 | 0.15 | 0.12 | 0.12 | 0.10 | 0.08 | 0.08 | 0.08 | stem | stem | 0.10 |
| corruption_type | categorical | 0.17 | 0.17 | 0.16 | 0.15 | 0.16 | 0.15 | 0.14 | 0.11 | 0.10 | 0.10 | 0.10 | stem | stem | 0.07 |
| severity | continuous | 0.09 | 0.05 | 0.04 | 0.04 | 0.06 | 0.05 | 0.05 | 0.03 | 0.04 | 0.03 | 0.03 | stem | stem | 0.05 |

## 3. Layer flow (CKA)
| pair | CKA |
|---|---|
| stem->layer1.0 | 0.908 |
| layer1.0->layer1.1 | 0.939 |
| layer1.1->layer1.2 | 0.977 |
| layer1.2->layer2.0 | 0.980 |
| layer2.0->layer2.1 | 0.973 |
| layer2.1->layer2.2 | 0.983 |
| layer2.2->layer3.0 | 0.965 |
| layer3.0->layer3.1 | 0.971 |
| layer3.1->layer3.2 | 0.985 |
| layer3.2->penult | 1.000 |

biggest reorganization: `stem->layer1.0` (drop 0.092)

## 4. Sensitivity field (paired corruption displacement)
layer `penult`; magnitude in within-class-radius units

| corruption | sev | magnitude | per-sample | top5 frac | class-sub frac | coherence | class consistency | norm ratio |
|---|---|---|---|---|---|---|---|---|
| brightness | 1 | 0.21 | 0.44 | 0.99 | 0.98 | 0.44 | 0.93 | 1.040 |
| brightness | 3 | 0.68 | 1.03 | 0.99 | 0.98 | 0.61 | 0.97 | 1.140 |
| brightness | 5 | 1.36 | 1.73 | 0.98 | 0.98 | 0.75 | 0.98 | 1.301 |
| contrast | 1 | 0.47 | 0.68 | 0.99 | 0.99 | 0.62 | 0.98 | 0.914 |
| contrast | 3 | 0.86 | 1.26 | 0.97 | 0.97 | 0.64 | 0.98 | 0.892 |
| contrast | 5 | 1.03 | 1.69 | 0.92 | 0.94 | 0.59 | 0.96 | 0.932 |
| defocus_blur | 1 | 0.08 | 0.16 | 0.97 | 0.97 | 0.42 | 0.96 | 0.985 |
| defocus_blur | 3 | 0.23 | 0.44 | 0.95 | 0.97 | 0.47 | 0.96 | 0.957 |
| defocus_blur | 5 | 0.36 | 0.64 | 0.95 | 0.97 | 0.51 | 0.96 | 0.938 |
| elastic_transform | 1 | 0.14 | 0.60 | 0.95 | 0.97 | 0.21 | 0.89 | 0.974 |
| elastic_transform | 3 | 0.22 | 0.60 | 0.95 | 0.97 | 0.32 | 0.94 | 0.960 |
| elastic_transform | 5 | 0.21 | 0.58 | 0.96 | 0.97 | 0.32 | 0.94 | 0.961 |
| fog | 1 | 0.41 | 0.63 | 0.99 | 0.98 | 0.60 | 0.96 | 0.922 |
| fog | 3 | 0.81 | 1.15 | 0.98 | 0.97 | 0.67 | 0.95 | 0.887 |
| fog | 5 | 0.96 | 1.33 | 0.96 | 0.95 | 0.70 | 0.94 | 0.900 |
| frost | 1 | 0.59 | 0.94 | 0.99 | 0.99 | 0.56 | 0.97 | 1.126 |
| frost | 3 | 0.86 | 1.26 | 0.98 | 0.99 | 0.60 | 0.97 | 1.182 |
| frost | 5 | 0.38 | 0.97 | 0.93 | 0.96 | 0.31 | 0.90 | 1.056 |
| gaussian_noise | 1 | 0.07 | 0.45 | 0.83 | 0.89 | 0.14 | 0.75 | 1.006 |
| gaussian_noise | 3 | 0.27 | 0.68 | 0.93 | 0.95 | 0.39 | 0.93 | 1.040 |
| gaussian_noise | 5 | 0.42 | 0.80 | 0.95 | 0.96 | 0.51 | 0.96 | 1.069 |
| glass_blur | 1 | 0.05 | 0.54 | 0.86 | 0.84 | 0.09 | 0.60 | 1.009 |
| glass_blur | 3 | 0.12 | 0.54 | 0.95 | 0.96 | 0.20 | 0.87 | 0.976 |
| glass_blur | 5 | 0.11 | 0.59 | 0.92 | 0.93 | 0.17 | 0.81 | 0.979 |
| impulse_noise | 1 | 0.16 | 0.51 | 0.92 | 0.94 | 0.28 | 0.92 | 1.027 |
| impulse_noise | 3 | 0.45 | 0.81 | 0.95 | 0.96 | 0.53 | 0.96 | 1.083 |
| impulse_noise | 5 | 1.01 | 1.30 | 0.96 | 0.96 | 0.76 | 0.98 | 1.206 |
| jpeg_compression | 1 | 0.02 | 0.36 | 0.85 | 0.84 | 0.06 | 0.41 | 0.996 |
| jpeg_compression | 3 | 0.04 | 0.41 | 0.90 | 0.90 | 0.09 | 0.67 | 0.992 |
| jpeg_compression | 5 | 0.06 | 0.44 | 0.93 | 0.92 | 0.11 | 0.73 | 0.989 |
| motion_blur | 1 | 0.15 | 0.44 | 0.95 | 0.97 | 0.29 | 0.94 | 0.974 |
| motion_blur | 3 | 0.28 | 0.60 | 0.95 | 0.97 | 0.41 | 0.95 | 0.952 |
| motion_blur | 5 | 0.33 | 0.65 | 0.95 | 0.97 | 0.45 | 0.96 | 0.945 |
| pixelate | 1 | 0.03 | 0.29 | 0.77 | 0.81 | 0.08 | 0.56 | 0.995 |
| pixelate | 3 | 0.06 | 0.39 | 0.90 | 0.93 | 0.12 | 0.73 | 0.989 |
| pixelate | 5 | 0.15 | 0.50 | 0.94 | 0.97 | 0.26 | 0.92 | 0.972 |
| shot_noise | 1 | 0.05 | 0.40 | 0.77 | 0.86 | 0.10 | 0.57 | 1.006 |
| shot_noise | 3 | 0.23 | 0.63 | 0.91 | 0.94 | 0.34 | 0.92 | 1.039 |
| shot_noise | 5 | 0.43 | 0.81 | 0.94 | 0.95 | 0.50 | 0.96 | 1.078 |
| snow | 1 | 0.26 | 0.52 | 0.99 | 0.98 | 0.44 | 0.96 | 1.051 |
| snow | 3 | 0.65 | 0.95 | 0.99 | 0.99 | 0.63 | 0.98 | 1.136 |
| snow | 5 | 1.32 | 1.63 | 0.99 | 0.99 | 0.75 | 0.99 | 1.297 |
| zoom_blur | 1 | 0.21 | 0.50 | 0.95 | 0.97 | 0.37 | 0.95 | 0.962 |
| zoom_blur | 3 | 0.28 | 0.59 | 0.96 | 0.98 | 0.42 | 0.96 | 0.949 |
| zoom_blur | 5 | 0.35 | 0.67 | 0.96 | 0.98 | 0.46 | 0.96 | 0.938 |

## 5. Density per split
layer `penult`

| split | sparse frac (ref q95) | median log-radius shift |
|---|---|---|
| corrupt__brightness__s1 | 0.082 | 0.062 |
| corrupt__brightness__s3 | 0.193 | 0.191 |
| corrupt__brightness__s5 | 0.451 | 0.386 |
| corrupt__contrast__s1 | 0.018 | -0.073 |
| corrupt__contrast__s3 | 0.048 | -0.004 |
| corrupt__contrast__s5 | 0.465 | 0.390 |
| corrupt__defocus_blur__s1 | 0.042 | -0.001 |
| corrupt__defocus_blur__s3 | 0.027 | -0.019 |
| corrupt__defocus_blur__s5 | 0.025 | -0.027 |
| corrupt__elastic_transform__s1 | 0.028 | -0.011 |
| corrupt__elastic_transform__s3 | 0.025 | -0.026 |
| corrupt__elastic_transform__s5 | 0.029 | -0.027 |
| corrupt__fog__s1 | 0.007 | -0.069 |
| corrupt__fog__s3 | 0.004 | -0.115 |
| corrupt__fog__s5 | 0.008 | -0.109 |
| corrupt__frost__s1 | 0.170 | 0.159 |
| corrupt__frost__s3 | 0.233 | 0.224 |
| corrupt__frost__s5 | 0.108 | 0.092 |
| corrupt__gaussian_noise__s1 | 0.044 | -0.007 |
| corrupt__gaussian_noise__s3 | 0.053 | 0.017 |
| corrupt__gaussian_noise__s5 | 0.072 | 0.064 |
| corrupt__glass_blur__s1 | 0.050 | 0.008 |
| corrupt__glass_blur__s3 | 0.030 | -0.013 |
| corrupt__glass_blur__s5 | 0.031 | -0.017 |
| corrupt__impulse_noise__s1 | 0.058 | 0.006 |
| corrupt__impulse_noise__s3 | 0.105 | 0.051 |
| corrupt__impulse_noise__s5 | 0.193 | 0.179 |
| corrupt__jpeg_compression__s1 | 0.043 | 0.004 |
| corrupt__jpeg_compression__s3 | 0.042 | -0.002 |
| corrupt__jpeg_compression__s5 | 0.038 | -0.003 |
| corrupt__motion_blur__s1 | 0.035 | -0.006 |
| corrupt__motion_blur__s3 | 0.026 | -0.022 |
| corrupt__motion_blur__s5 | 0.029 | -0.022 |
| corrupt__pixelate__s1 | 0.047 | 0.008 |
| corrupt__pixelate__s3 | 0.040 | 0.008 |
| corrupt__pixelate__s5 | 0.036 | -0.016 |
| corrupt__shot_noise__s1 | 0.042 | -0.001 |
| corrupt__shot_noise__s3 | 0.056 | 0.019 |
| corrupt__shot_noise__s5 | 0.086 | 0.069 |
| corrupt__snow__s1 | 0.084 | 0.052 |
| corrupt__snow__s3 | 0.151 | 0.144 |
| corrupt__snow__s5 | 0.331 | 0.309 |
| corrupt__zoom_blur__s1 | 0.028 | -0.018 |
| corrupt__zoom_blur__s3 | 0.022 | -0.025 |
| corrupt__zoom_blur__s5 | 0.022 | -0.028 |
| ood__cifar100 | 0.101 | 0.020 |
| ood__svhn | 0.116 | 0.102 |
| panel | 0.047 | -0.078 |
| test | 0.049 | 0.007 |

## 6. Adjacency
layer `penult`


closest class pairs (sep = center distance / RMS radius): (3,5) 0.08, (4,6) 0.08, (2,4) 0.12, (2,6) 0.12, (0,8) 0.15, (1,8) 0.18

valley ratio mean 1.00, min 0.96, pairs with a valley 0.47

nearest-center confusions (true → nearest): 2→4 0.42, 6→4 0.39, 3→4 0.34, 1→9 0.29, 8→4 0.27, 8→9 0.24

single-linkage merge order (first 5): [3, 5]@0.08 ; [4, 6]@0.08 ; [2, 4, 6]@0.12 ; [0, 8]@0.15 ; [0, 1, 8]@0.18
