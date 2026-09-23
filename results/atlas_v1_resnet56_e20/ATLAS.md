# ATLAS — atlas_v1_resnet56_e20
built 2026-09-23 05:58:38 · source **real** · arch `cifar10_resnet56` · weights `file:/workspace/models/resnet56_s11_e20_chenyaofo.pt sha256:76c8b3374498f0f1` · layers 11

## 1. Per-layer invariants
| layer | D | PR | eff.rank | dim95 | TwoNN ID | sep ratio | NC1 | ETF dev | hub skew | test sparse | NC acc(test) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | 16 | 3.1 | 3.6 | 4 | 6.0 | 0.17 | 511.43 | 0.643 | 0.08 | 0.059 | 0.236 |
| layer1.0 | 16 | 3.4 | 4.1 | 4 | 6.7 | 0.18 | 204.38 | 0.606 | 0.18 | 0.057 | 0.257 |
| layer1.5 | 16 | 4.2 | 5.8 | 8 | 8.9 | 0.21 | 97.56 | 0.589 | 0.61 | 0.060 | 0.322 |
| layer1.8 | 16 | 4.6 | 6.2 | 7 | 9.2 | 0.30 | 60.30 | 0.468 | 0.65 | 0.058 | 0.346 |
| layer2.0 | 32 | 4.9 | 6.9 | 8 | 9.7 | 0.32 | 46.47 | 0.475 | 0.67 | 0.054 | 0.386 |
| layer2.5 | 32 | 6.8 | 9.4 | 11 | 11.3 | 0.40 | 26.11 | 0.408 | 0.92 | 0.049 | 0.473 |
| layer2.8 | 32 | 7.3 | 10.7 | 13 | 12.0 | 0.43 | 14.63 | 0.402 | 1.05 | 0.050 | 0.506 |
| layer3.0 | 64 | 7.4 | 12.4 | 19 | 14.5 | 0.49 | 5.17 | 0.384 | 1.22 | 0.049 | 0.562 |
| layer3.5 | 64 | 10.9 | 19.4 | 33 | 18.0 | 0.87 | 0.99 | 0.254 | 1.67 | 0.047 | 0.800 |
| layer3.8 | 64 | 7.7 | 10.3 | 10 | 11.7 | 2.04 | 0.32 | 0.185 | 0.90 | 0.063 | 0.892 |
| penult | 64 | 7.7 | 10.3 | 10 | 11.7 | 2.04 | 0.32 | 0.185 | 0.90 | 0.063 | 0.892 |

## 2. Decodability profile (which factor lives where)
categorical = accuracy minus majority-class rate; continuous = CV R²

| factor | kind | stem | layer1.0 | layer1.5 | layer1.8 | layer2.0 | layer2.5 | layer2.8 | layer3.0 | layer3.5 | layer3.8 | penult | commit | peak | washout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| luminance_mean | continuous | 1.00 | 0.99 | 0.97 | 0.96 | 0.96 | 0.93 | 0.90 | 0.88 | 0.72 | 0.55 | 0.55 | stem | stem | 0.45 |
| contrast_rms | continuous | 0.63 | 0.62 | 0.58 | 0.57 | 0.63 | 0.60 | 0.59 | 0.64 | 0.54 | 0.46 | 0.46 | stem | layer3.0 | 0.17 |
| highfreq_ratio | continuous | 0.22 | 0.29 | 0.54 | 0.53 | 0.60 | 0.59 | 0.58 | 0.58 | 0.54 | 0.53 | 0.53 | layer2.0 | layer2.0 | 0.07 |
| spectral_slope | continuous | 0.11 | 0.19 | 0.48 | 0.52 | 0.60 | 0.62 | 0.62 | 0.63 | 0.57 | 0.54 | 0.54 | layer2.0 | layer3.0 | 0.09 |
| spectral_anisotropy | continuous | 0.14 | 0.13 | 0.24 | 0.29 | 0.33 | 0.31 | 0.33 | 0.36 | 0.37 | 0.34 | 0.34 | layer3.0 | layer3.5 | 0.03 |
| noise_sigma | continuous | 0.75 | 0.73 | 0.93 | 0.92 | 0.95 | 0.93 | 0.92 | 0.92 | 0.87 | 0.80 | 0.80 | layer1.5 | layer2.0 | 0.15 |
| saturation_mean | continuous | 0.76 | 0.84 | 0.73 | 0.63 | 0.77 | 0.64 | 0.59 | 0.60 | 0.40 | 0.32 | 0.32 | stem | layer1.0 | 0.52 |
| hue_cos | continuous | 0.76 | 0.83 | 0.81 | 0.80 | 0.80 | 0.78 | 0.76 | 0.72 | 0.64 | 0.55 | 0.55 | stem | layer1.0 | 0.28 |
| hue_sin | continuous | 0.79 | 0.82 | 0.82 | 0.79 | 0.78 | 0.76 | 0.75 | 0.73 | 0.65 | 0.58 | 0.58 | stem | layer1.0 | 0.25 |
| colorfulness | continuous | 0.63 | 0.70 | 0.79 | 0.79 | 0.78 | 0.69 | 0.63 | 0.58 | 0.39 | 0.31 | 0.31 | layer1.5 | layer1.8 | 0.48 |
| edge_density | continuous | 0.81 | 0.80 | 0.84 | 0.81 | 0.88 | 0.87 | 0.86 | 0.86 | 0.80 | 0.72 | 0.72 | stem | layer2.0 | 0.16 |
| orientation_entropy | continuous | 0.12 | 0.13 | 0.33 | 0.40 | 0.44 | 0.47 | 0.48 | 0.51 | 0.48 | 0.48 | 0.48 | layer2.5 | layer3.0 | 0.03 |
| blockiness | continuous | 0.01 | 0.01 | 0.01 | 0.02 | 0.05 | 0.07 | 0.09 | 0.11 | 0.10 | 0.07 | 0.07 | layer3.0 | layer3.0 | 0.04 |
| class | categorical | 0.22 | 0.22 | 0.31 | 0.36 | 0.44 | 0.50 | 0.54 | 0.61 | 0.74 | 0.78 | 0.78 | layer3.5 | layer3.8 | 0.00 |
| coarse_animal_vehicle | categorical | 0.21 | 0.21 | 0.27 | 0.28 | 0.31 | 0.32 | 0.33 | 0.35 | 0.37 | 0.38 | 0.38 | layer3.0 | layer3.8 | 0.00 |
| corruption_family | categorical | 0.09 | 0.12 | 0.21 | 0.29 | 0.33 | 0.33 | 0.32 | 0.33 | 0.27 | 0.25 | 0.25 | layer2.0 | layer3.0 | 0.09 |
| corruption_type | categorical | 0.13 | 0.15 | 0.26 | 0.38 | 0.46 | 0.44 | 0.44 | 0.42 | 0.35 | 0.28 | 0.28 | layer2.0 | layer2.0 | 0.18 |
| severity | continuous | 0.03 | 0.04 | 0.07 | 0.10 | 0.22 | 0.22 | 0.23 | 0.27 | 0.25 | 0.23 | 0.23 | layer3.0 | layer3.0 | 0.04 |

## 3. Layer flow (CKA)
| pair | CKA |
|---|---|
| stem->layer1.0 | 0.961 |
| layer1.0->layer1.5 | 0.807 |
| layer1.5->layer1.8 | 0.802 |
| layer1.8->layer2.0 | 0.955 |
| layer2.0->layer2.5 | 0.853 |
| layer2.5->layer2.8 | 0.920 |
| layer2.8->layer3.0 | 0.941 |
| layer3.0->layer3.5 | 0.789 |
| layer3.5->layer3.8 | 0.777 |
| layer3.8->penult | 1.000 |

biggest reorganization: `layer3.5->layer3.8` (drop 0.223)

## 4. Sensitivity field (paired corruption displacement)
layer `penult`; magnitude in within-class-radius units

| corruption | sev | magnitude | per-sample | top5 frac | class-sub frac | coherence | class consistency | norm ratio |
|---|---|---|---|---|---|---|---|---|
| brightness | 1 | 0.03 | 0.12 | 0.54 | 0.61 | 0.28 | 0.59 | 1.000 |
| brightness | 3 | 0.10 | 0.35 | 0.49 | 0.67 | 0.29 | 0.47 | 0.988 |
| brightness | 5 | 0.21 | 0.66 | 0.38 | 0.73 | 0.30 | 0.30 | 0.948 |
| contrast | 1 | 0.16 | 0.32 | 0.72 | 0.87 | 0.49 | 0.71 | 0.995 |
| contrast | 3 | 0.52 | 1.03 | 0.64 | 0.83 | 0.50 | 0.47 | 0.907 |
| contrast | 5 | 1.23 | 2.10 | 0.56 | 0.80 | 0.58 | 0.40 | 0.753 |
| defocus_blur | 1 | 0.06 | 0.14 | 0.69 | 0.82 | 0.44 | 0.72 | 0.999 |
| defocus_blur | 3 | 0.41 | 0.74 | 0.81 | 0.88 | 0.54 | 0.60 | 0.946 |
| defocus_blur | 5 | 0.87 | 1.51 | 0.81 | 0.87 | 0.56 | 0.45 | 0.837 |
| elastic_transform | 1 | 0.22 | 0.70 | 0.73 | 0.82 | 0.30 | 0.31 | 0.933 |
| elastic_transform | 3 | 0.39 | 0.87 | 0.80 | 0.88 | 0.43 | 0.45 | 0.916 |
| elastic_transform | 5 | 0.42 | 1.10 | 0.80 | 0.93 | 0.34 | 0.20 | 0.885 |
| fog | 1 | 0.09 | 0.22 | 0.74 | 0.85 | 0.44 | 0.73 | 0.999 |
| fog | 3 | 0.31 | 0.65 | 0.72 | 0.86 | 0.48 | 0.57 | 0.956 |
| fog | 5 | 0.65 | 1.26 | 0.75 | 0.86 | 0.50 | 0.40 | 0.869 |
| frost | 1 | 0.24 | 0.59 | 0.72 | 0.77 | 0.39 | 0.45 | 0.949 |
| frost | 3 | 0.58 | 1.19 | 0.71 | 0.79 | 0.46 | 0.36 | 0.867 |
| frost | 5 | 0.81 | 1.49 | 0.77 | 0.82 | 0.52 | 0.40 | 0.840 |
| gaussian_noise | 1 | 0.53 | 0.94 | 0.83 | 0.83 | 0.55 | 0.55 | 0.903 |
| gaussian_noise | 3 | 1.00 | 1.64 | 0.81 | 0.84 | 0.59 | 0.49 | 0.822 |
| gaussian_noise | 5 | 1.16 | 1.86 | 0.79 | 0.85 | 0.60 | 0.48 | 0.807 |
| glass_blur | 1 | 0.87 | 1.59 | 0.70 | 0.92 | 0.52 | 0.42 | 0.848 |
| glass_blur | 3 | 0.76 | 1.47 | 0.76 | 0.93 | 0.48 | 0.40 | 0.862 |
| glass_blur | 5 | 0.90 | 1.69 | 0.72 | 0.93 | 0.50 | 0.38 | 0.845 |
| impulse_noise | 1 | 0.22 | 0.68 | 0.51 | 0.70 | 0.31 | 0.25 | 0.942 |
| impulse_noise | 3 | 0.57 | 1.26 | 0.61 | 0.70 | 0.44 | 0.29 | 0.839 |
| impulse_noise | 5 | 1.10 | 1.93 | 0.68 | 0.78 | 0.55 | 0.37 | 0.768 |
| jpeg_compression | 1 | 0.23 | 0.66 | 0.71 | 0.82 | 0.34 | 0.33 | 0.937 |
| jpeg_compression | 3 | 0.27 | 0.88 | 0.58 | 0.79 | 0.29 | 0.15 | 0.904 |
| jpeg_compression | 5 | 0.31 | 1.00 | 0.60 | 0.83 | 0.29 | 0.12 | 0.893 |
| motion_blur | 1 | 0.33 | 0.67 | 0.83 | 0.88 | 0.47 | 0.53 | 0.943 |
| motion_blur | 3 | 0.63 | 1.23 | 0.76 | 0.84 | 0.49 | 0.40 | 0.851 |
| motion_blur | 5 | 0.71 | 1.39 | 0.72 | 0.82 | 0.49 | 0.37 | 0.818 |
| pixelate | 1 | 0.20 | 0.46 | 0.80 | 0.92 | 0.39 | 0.62 | 0.977 |
| pixelate | 3 | 0.52 | 0.93 | 0.73 | 0.89 | 0.51 | 0.56 | 0.925 |
| pixelate | 5 | 1.02 | 1.75 | 0.55 | 0.83 | 0.55 | 0.41 | 0.841 |
| shot_noise | 1 | 0.41 | 0.72 | 0.82 | 0.84 | 0.54 | 0.60 | 0.932 |
| shot_noise | 3 | 0.88 | 1.44 | 0.82 | 0.84 | 0.59 | 0.51 | 0.840 |
| shot_noise | 5 | 1.12 | 1.79 | 0.79 | 0.84 | 0.60 | 0.48 | 0.811 |
| snow | 1 | 0.19 | 0.65 | 0.42 | 0.73 | 0.28 | 0.23 | 0.956 |
| snow | 3 | 0.32 | 1.01 | 0.47 | 0.69 | 0.31 | 0.17 | 0.899 |
| snow | 5 | 0.47 | 1.23 | 0.51 | 0.73 | 0.37 | 0.22 | 0.866 |
| zoom_blur | 1 | 0.42 | 0.81 | 0.86 | 0.88 | 0.50 | 0.51 | 0.919 |
| zoom_blur | 3 | 0.63 | 1.12 | 0.82 | 0.89 | 0.53 | 0.50 | 0.892 |
| zoom_blur | 5 | 0.83 | 1.47 | 0.78 | 0.88 | 0.54 | 0.45 | 0.856 |

## 5. Density per split
layer `penult`

| split | sparse frac (ref q95) | median log-radius shift |
|---|---|---|
| corrupt__brightness__s1 | 0.066 | 0.004 |
| corrupt__brightness__s3 | 0.087 | 0.020 |
| corrupt__brightness__s5 | 0.104 | 0.061 |
| corrupt__contrast__s1 | 0.063 | 0.006 |
| corrupt__contrast__s3 | 0.155 | 0.120 |
| corrupt__contrast__s5 | 0.623 | 0.322 |
| corrupt__defocus_blur__s1 | 0.060 | -0.002 |
| corrupt__defocus_blur__s3 | 0.098 | 0.066 |
| corrupt__defocus_blur__s5 | 0.166 | 0.160 |
| corrupt__elastic_transform__s1 | 0.080 | 0.038 |
| corrupt__elastic_transform__s3 | 0.095 | 0.065 |
| corrupt__elastic_transform__s5 | 0.155 | 0.119 |
| corrupt__fog__s1 | 0.058 | -0.001 |
| corrupt__fog__s3 | 0.089 | 0.044 |
| corrupt__fog__s5 | 0.189 | 0.148 |
| corrupt__frost__s1 | 0.099 | 0.046 |
| corrupt__frost__s3 | 0.208 | 0.137 |
| corrupt__frost__s5 | 0.262 | 0.173 |
| corrupt__gaussian_noise__s1 | 0.140 | 0.092 |
| corrupt__gaussian_noise__s3 | 0.245 | 0.146 |
| corrupt__gaussian_noise__s5 | 0.236 | 0.129 |
| corrupt__glass_blur__s1 | 0.281 | 0.171 |
| corrupt__glass_blur__s3 | 0.256 | 0.160 |
| corrupt__glass_blur__s5 | 0.297 | 0.184 |
| corrupt__impulse_noise__s1 | 0.098 | 0.058 |
| corrupt__impulse_noise__s3 | 0.204 | 0.146 |
| corrupt__impulse_noise__s5 | 0.308 | 0.186 |
| corrupt__jpeg_compression__s1 | 0.090 | 0.042 |
| corrupt__jpeg_compression__s3 | 0.098 | 0.073 |
| corrupt__jpeg_compression__s5 | 0.127 | 0.094 |
| corrupt__motion_blur__s1 | 0.087 | 0.057 |
| corrupt__motion_blur__s3 | 0.127 | 0.128 |
| corrupt__motion_blur__s5 | 0.140 | 0.146 |
| corrupt__pixelate__s1 | 0.096 | 0.030 |
| corrupt__pixelate__s3 | 0.208 | 0.114 |
| corrupt__pixelate__s5 | 0.507 | 0.284 |
| corrupt__shot_noise__s1 | 0.114 | 0.059 |
| corrupt__shot_noise__s3 | 0.234 | 0.142 |
| corrupt__shot_noise__s5 | 0.239 | 0.146 |
| corrupt__snow__s1 | 0.124 | 0.069 |
| corrupt__snow__s3 | 0.172 | 0.120 |
| corrupt__snow__s5 | 0.258 | 0.170 |
| corrupt__zoom_blur__s1 | 0.105 | 0.071 |
| corrupt__zoom_blur__s3 | 0.147 | 0.124 |
| corrupt__zoom_blur__s5 | 0.170 | 0.167 |
| ood__cifar100 | 0.269 | 0.192 |
| ood__svhn | 0.343 | 0.220 |
| panel | 0.047 | -0.014 |
| test | 0.063 | -0.003 |

## 6. Adjacency
layer `penult`


closest class pairs (sep = center distance / RMS radius): (3,5) 1.26, (2,3) 1.85, (0,2) 1.97, (3,4) 2.02, (3,6) 2.08, (2,5) 2.08

valley ratio mean 1.63, min 1.09, pairs with a valley 1.00

nearest-center confusions (true → nearest): 5→3 0.14, 3→5 0.08, 2→0 0.05, 2→3 0.04, 7→3 0.04, 8→0 0.03

single-linkage merge order (first 5): [3, 5]@1.26 ; [2, 3, 5]@1.85 ; [0, 2, 3, 5]@1.97 ; [0, 2, 3, 4, 5]@2.02 ; [0, 2, 3, 4, 5, 6]@2.08

## 7. Type-b margin (margin_typeb)
_margin_typeb not run_
