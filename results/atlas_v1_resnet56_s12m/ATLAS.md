# ATLAS — atlas_v1_resnet56_s12m
built 2026-09-23 14:16:24 · source **real** · arch `cifar10_resnet56` · weights `file:/workspace/models/resnet56_s12m_chenyaofo.pt sha256:71db3bad0a2c41c2` · layers 11

## 1. Per-layer invariants
| layer | D | PR | eff.rank | dim95 | TwoNN ID | sep ratio | NC1 | ETF dev | hub skew | test sparse | NC acc(test) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | 16 | 2.9 | 3.5 | 4 | 6.3 | 0.17 | 265.35 | 0.660 | 0.07 | 0.058 | 0.243 |
| layer1.0 | 16 | 2.9 | 4.0 | 5 | 7.8 | 0.17 | 118.82 | 0.682 | 0.39 | 0.060 | 0.258 |
| layer1.5 | 16 | 5.1 | 6.9 | 8 | 9.5 | 0.22 | 75.86 | 0.475 | 0.72 | 0.057 | 0.334 |
| layer1.8 | 16 | 5.0 | 7.1 | 9 | 9.8 | 0.30 | 66.94 | 0.430 | 0.83 | 0.058 | 0.355 |
| layer2.0 | 32 | 5.6 | 8.1 | 9 | 10.9 | 0.35 | 29.77 | 0.453 | 0.80 | 0.049 | 0.400 |
| layer2.5 | 32 | 7.3 | 10.8 | 14 | 13.2 | 0.43 | 12.20 | 0.401 | 1.22 | 0.046 | 0.506 |
| layer2.8 | 32 | 7.7 | 11.7 | 16 | 13.8 | 0.47 | 8.87 | 0.396 | 1.26 | 0.046 | 0.549 |
| layer3.0 | 64 | 8.7 | 14.8 | 24 | 15.6 | 0.56 | 3.49 | 0.350 | 1.43 | 0.048 | 0.625 |
| layer3.5 | 64 | 12.8 | 24.1 | 42 | 18.9 | 0.80 | 0.92 | 0.275 | 1.87 | 0.049 | 0.796 |
| layer3.8 | 64 | 8.9 | 10.3 | 9 | 10.7 | 3.37 | 0.13 | 0.085 | 0.88 | 0.121 | 0.923 |
| penult | 64 | 8.9 | 10.3 | 9 | 10.7 | 3.37 | 0.13 | 0.085 | 0.88 | 0.121 | 0.923 |

## 2. Decodability profile (which factor lives where)
categorical = accuracy minus majority-class rate; continuous = CV R²

| factor | kind | stem | layer1.0 | layer1.5 | layer1.8 | layer2.0 | layer2.5 | layer2.8 | layer3.0 | layer3.5 | layer3.8 | penult | commit | peak | washout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| luminance_mean | continuous | 1.00 | 0.98 | 0.95 | 0.95 | 0.92 | 0.86 | 0.83 | 0.75 | 0.58 | 0.29 | 0.29 | stem | stem | 0.71 |
| contrast_rms | continuous | 0.67 | 0.59 | 0.56 | 0.53 | 0.68 | 0.60 | 0.60 | 0.59 | 0.53 | 0.43 | 0.43 | stem | layer2.0 | 0.25 |
| highfreq_ratio | continuous | 0.19 | 0.41 | 0.54 | 0.53 | 0.64 | 0.61 | 0.60 | 0.61 | 0.58 | 0.47 | 0.47 | layer2.0 | layer2.0 | 0.17 |
| spectral_slope | continuous | 0.08 | 0.28 | 0.52 | 0.53 | 0.69 | 0.67 | 0.67 | 0.66 | 0.61 | 0.51 | 0.51 | layer2.0 | layer2.0 | 0.18 |
| spectral_anisotropy | continuous | 0.21 | 0.23 | 0.30 | 0.29 | 0.31 | 0.34 | 0.34 | 0.36 | 0.33 | 0.30 | 0.30 | layer2.5 | layer3.0 | 0.05 |
| noise_sigma | continuous | 0.64 | 0.86 | 0.88 | 0.90 | 0.95 | 0.93 | 0.93 | 0.90 | 0.85 | 0.75 | 0.75 | layer1.0 | layer2.0 | 0.21 |
| saturation_mean | continuous | 0.78 | 0.87 | 0.70 | 0.59 | 0.69 | 0.57 | 0.53 | 0.47 | 0.34 | 0.24 | 0.24 | stem | layer1.0 | 0.63 |
| hue_cos | continuous | 0.85 | 0.84 | 0.83 | 0.82 | 0.80 | 0.78 | 0.76 | 0.72 | 0.66 | 0.52 | 0.52 | stem | stem | 0.33 |
| hue_sin | continuous | 0.82 | 0.82 | 0.82 | 0.78 | 0.75 | 0.72 | 0.71 | 0.69 | 0.63 | 0.52 | 0.52 | stem | layer1.0 | 0.30 |
| colorfulness | continuous | 0.66 | 0.84 | 0.84 | 0.76 | 0.78 | 0.70 | 0.65 | 0.57 | 0.44 | 0.27 | 0.27 | layer1.0 | layer1.5 | 0.56 |
| edge_density | continuous | 0.76 | 0.79 | 0.76 | 0.78 | 0.89 | 0.86 | 0.85 | 0.84 | 0.78 | 0.66 | 0.66 | layer2.0 | layer2.0 | 0.24 |
| orientation_entropy | continuous | 0.22 | 0.27 | 0.36 | 0.40 | 0.47 | 0.48 | 0.49 | 0.52 | 0.49 | 0.44 | 0.44 | layer2.5 | layer3.0 | 0.08 |
| blockiness | continuous | 0.01 | 0.01 | 0.01 | 0.02 | 0.06 | 0.06 | 0.07 | 0.10 | 0.08 | 0.05 | 0.05 | layer3.0 | layer3.0 | 0.04 |
| class | categorical | 0.24 | 0.27 | 0.32 | 0.36 | 0.45 | 0.54 | 0.56 | 0.66 | 0.77 | 0.81 | 0.81 | layer3.5 | layer3.8 | 0.00 |
| coarse_animal_vehicle | categorical | 0.22 | 0.25 | 0.27 | 0.28 | 0.31 | 0.33 | 0.32 | 0.36 | 0.37 | 0.39 | 0.39 | layer3.0 | layer3.8 | 0.00 |
| corruption_family | categorical | 0.09 | 0.16 | 0.28 | 0.34 | 0.38 | 0.35 | 0.35 | 0.36 | 0.31 | 0.24 | 0.24 | layer2.0 | layer2.0 | 0.14 |
| corruption_type | categorical | 0.14 | 0.18 | 0.35 | 0.48 | 0.53 | 0.53 | 0.52 | 0.50 | 0.43 | 0.29 | 0.29 | layer2.0 | layer2.0 | 0.24 |
| severity | continuous | 0.05 | 0.03 | 0.09 | 0.15 | 0.29 | 0.28 | 0.27 | 0.31 | 0.29 | 0.25 | 0.25 | layer2.0 | layer3.0 | 0.06 |

## 3. Layer flow (CKA)
| pair | CKA |
|---|---|
| stem->layer1.0 | 0.918 |
| layer1.0->layer1.5 | 0.757 |
| layer1.5->layer1.8 | 0.840 |
| layer1.8->layer2.0 | 0.955 |
| layer2.0->layer2.5 | 0.912 |
| layer2.5->layer2.8 | 0.986 |
| layer2.8->layer3.0 | 0.943 |
| layer3.0->layer3.5 | 0.859 |
| layer3.5->layer3.8 | 0.667 |
| layer3.8->penult | 1.000 |

biggest reorganization: `layer3.5->layer3.8` (drop 0.333)

## 4. Sensitivity field (paired corruption displacement)
layer `penult`; magnitude in within-class-radius units

| corruption | sev | magnitude | per-sample | top5 frac | class-sub frac | coherence | class consistency | norm ratio |
|---|---|---|---|---|---|---|---|---|
| brightness | 1 | 0.03 | 0.13 | 0.80 | 0.88 | 0.22 | 0.66 | 1.002 |
| brightness | 3 | 0.09 | 0.38 | 0.82 | 0.88 | 0.23 | 0.63 | 1.000 |
| brightness | 5 | 0.20 | 0.72 | 0.77 | 0.89 | 0.26 | 0.45 | 0.985 |
| contrast | 1 | 0.13 | 0.34 | 0.70 | 0.90 | 0.38 | 0.72 | 1.008 |
| contrast | 3 | 0.48 | 1.09 | 0.53 | 0.87 | 0.43 | 0.57 | 0.983 |
| contrast | 5 | 1.26 | 2.49 | 0.46 | 0.86 | 0.50 | 0.30 | 0.862 |
| defocus_blur | 1 | 0.06 | 0.17 | 0.60 | 0.89 | 0.32 | 0.62 | 1.004 |
| defocus_blur | 3 | 0.50 | 0.98 | 0.72 | 0.93 | 0.47 | 0.64 | 0.986 |
| defocus_blur | 5 | 1.15 | 2.06 | 0.75 | 0.92 | 0.53 | 0.44 | 0.914 |
| elastic_transform | 1 | 0.23 | 0.89 | 0.66 | 0.87 | 0.24 | 0.34 | 0.977 |
| elastic_transform | 3 | 0.47 | 1.15 | 0.72 | 0.92 | 0.37 | 0.48 | 0.966 |
| elastic_transform | 5 | 0.52 | 1.53 | 0.69 | 0.93 | 0.33 | 0.21 | 0.935 |
| fog | 1 | 0.08 | 0.24 | 0.71 | 0.88 | 0.35 | 0.71 | 1.006 |
| fog | 3 | 0.30 | 0.72 | 0.56 | 0.88 | 0.41 | 0.67 | 1.000 |
| fog | 5 | 0.89 | 1.65 | 0.49 | 0.90 | 0.53 | 0.52 | 0.949 |
| frost | 1 | 0.32 | 0.76 | 0.67 | 0.91 | 0.37 | 0.59 | 0.983 |
| frost | 3 | 0.80 | 1.56 | 0.72 | 0.91 | 0.48 | 0.48 | 0.946 |
| frost | 5 | 1.15 | 2.00 | 0.72 | 0.90 | 0.55 | 0.50 | 0.942 |
| gaussian_noise | 1 | 0.73 | 1.36 | 0.72 | 0.91 | 0.52 | 0.57 | 0.945 |
| gaussian_noise | 3 | 1.49 | 2.43 | 0.80 | 0.92 | 0.59 | 0.47 | 0.905 |
| gaussian_noise | 5 | 1.69 | 2.71 | 0.82 | 0.92 | 0.59 | 0.44 | 0.896 |
| glass_blur | 1 | 0.99 | 2.15 | 0.70 | 0.92 | 0.42 | 0.24 | 0.877 |
| glass_blur | 3 | 0.92 | 2.02 | 0.76 | 0.92 | 0.42 | 0.26 | 0.894 |
| glass_blur | 5 | 1.08 | 2.31 | 0.76 | 0.93 | 0.43 | 0.24 | 0.886 |
| impulse_noise | 1 | 0.40 | 1.10 | 0.46 | 0.82 | 0.32 | 0.36 | 0.984 |
| impulse_noise | 3 | 0.63 | 1.83 | 0.52 | 0.83 | 0.33 | 0.12 | 0.887 |
| impulse_noise | 5 | 1.15 | 2.57 | 0.51 | 0.86 | 0.43 | 0.17 | 0.795 |
| jpeg_compression | 1 | 0.34 | 0.97 | 0.42 | 0.93 | 0.33 | 0.43 | 0.974 |
| jpeg_compression | 3 | 0.42 | 1.29 | 0.30 | 0.93 | 0.31 | 0.22 | 0.949 |
| jpeg_compression | 5 | 0.49 | 1.49 | 0.35 | 0.92 | 0.31 | 0.17 | 0.934 |
| motion_blur | 1 | 0.34 | 0.85 | 0.84 | 0.92 | 0.37 | 0.54 | 0.981 |
| motion_blur | 3 | 0.73 | 1.59 | 0.85 | 0.90 | 0.43 | 0.39 | 0.917 |
| motion_blur | 5 | 0.87 | 1.85 | 0.82 | 0.89 | 0.44 | 0.34 | 0.890 |
| pixelate | 1 | 0.22 | 0.63 | 0.68 | 0.94 | 0.33 | 0.60 | 0.990 |
| pixelate | 3 | 0.64 | 1.26 | 0.59 | 0.96 | 0.46 | 0.55 | 0.959 |
| pixelate | 5 | 1.46 | 2.42 | 0.39 | 0.94 | 0.57 | 0.46 | 0.958 |
| shot_noise | 1 | 0.54 | 1.03 | 0.67 | 0.91 | 0.50 | 0.63 | 0.963 |
| shot_noise | 3 | 1.25 | 2.08 | 0.77 | 0.92 | 0.57 | 0.50 | 0.906 |
| shot_noise | 5 | 1.59 | 2.58 | 0.79 | 0.92 | 0.59 | 0.44 | 0.892 |
| snow | 1 | 0.22 | 0.83 | 0.46 | 0.86 | 0.25 | 0.27 | 0.987 |
| snow | 3 | 0.33 | 1.24 | 0.48 | 0.85 | 0.26 | 0.16 | 0.956 |
| snow | 5 | 0.61 | 1.54 | 0.51 | 0.91 | 0.37 | 0.31 | 0.935 |
| zoom_blur | 1 | 0.46 | 1.01 | 0.86 | 0.92 | 0.42 | 0.54 | 0.962 |
| zoom_blur | 3 | 0.77 | 1.48 | 0.79 | 0.93 | 0.49 | 0.52 | 0.937 |
| zoom_blur | 5 | 1.07 | 1.98 | 0.76 | 0.92 | 0.51 | 0.45 | 0.910 |

## 5. Density per split
layer `penult`

| split | sparse frac (ref q95) | median log-radius shift |
|---|---|---|
| corrupt__brightness__s1 | 0.119 | 0.015 |
| corrupt__brightness__s3 | 0.137 | 0.030 |
| corrupt__brightness__s5 | 0.174 | 0.082 |
| corrupt__contrast__s1 | 0.135 | 0.024 |
| corrupt__contrast__s3 | 0.260 | 0.206 |
| corrupt__contrast__s5 | 0.719 | 0.561 |
| corrupt__defocus_blur__s1 | 0.118 | 0.023 |
| corrupt__defocus_blur__s3 | 0.230 | 0.187 |
| corrupt__defocus_blur__s5 | 0.483 | 0.446 |
| corrupt__elastic_transform__s1 | 0.201 | 0.147 |
| corrupt__elastic_transform__s3 | 0.265 | 0.251 |
| corrupt__elastic_transform__s5 | 0.428 | 0.392 |
| corrupt__fog__s1 | 0.130 | 0.018 |
| corrupt__fog__s3 | 0.193 | 0.098 |
| corrupt__fog__s5 | 0.439 | 0.411 |
| corrupt__frost__s1 | 0.201 | 0.113 |
| corrupt__frost__s3 | 0.419 | 0.372 |
| corrupt__frost__s5 | 0.540 | 0.486 |
| corrupt__gaussian_noise__s1 | 0.348 | 0.315 |
| corrupt__gaussian_noise__s3 | 0.586 | 0.512 |
| corrupt__gaussian_noise__s5 | 0.555 | 0.502 |
| corrupt__glass_blur__s1 | 0.617 | 0.531 |
| corrupt__glass_blur__s3 | 0.564 | 0.502 |
| corrupt__glass_blur__s5 | 0.639 | 0.559 |
| corrupt__impulse_noise__s1 | 0.302 | 0.247 |
| corrupt__impulse_noise__s3 | 0.560 | 0.503 |
| corrupt__impulse_noise__s5 | 0.851 | 0.665 |
| corrupt__jpeg_compression__s1 | 0.226 | 0.169 |
| corrupt__jpeg_compression__s3 | 0.329 | 0.298 |
| corrupt__jpeg_compression__s5 | 0.404 | 0.365 |
| corrupt__motion_blur__s1 | 0.197 | 0.152 |
| corrupt__motion_blur__s3 | 0.392 | 0.374 |
| corrupt__motion_blur__s5 | 0.451 | 0.427 |
| corrupt__pixelate__s1 | 0.189 | 0.095 |
| corrupt__pixelate__s3 | 0.362 | 0.290 |
| corrupt__pixelate__s5 | 0.644 | 0.603 |
| corrupt__shot_noise__s1 | 0.262 | 0.186 |
| corrupt__shot_noise__s3 | 0.552 | 0.494 |
| corrupt__shot_noise__s5 | 0.593 | 0.515 |
| corrupt__snow__s1 | 0.204 | 0.146 |
| corrupt__snow__s3 | 0.311 | 0.293 |
| corrupt__snow__s5 | 0.410 | 0.378 |
| corrupt__zoom_blur__s1 | 0.255 | 0.203 |
| corrupt__zoom_blur__s3 | 0.383 | 0.352 |
| corrupt__zoom_blur__s5 | 0.496 | 0.453 |
| ood__cifar100 | 0.628 | 0.534 |
| ood__svhn | 0.775 | 0.611 |
| panel | 0.047 | -0.047 |
| test | 0.121 | 0.025 |

## 6. Adjacency
layer `penult`


closest class pairs (sep = center distance / RMS radius): (3,5) 2.76, (2,3) 2.88, (3,4) 3.11, (0,2) 3.14, (2,5) 3.15, (2,4) 3.18

valley ratio mean 2.56, min 1.62, pairs with a valley 1.00

nearest-center confusions (true → nearest): 3→5 0.07, 5→3 0.07, 9→1 0.04, 0→8 0.03, 2→0 0.03, 2→4 0.03

single-linkage merge order (first 5): [3, 5]@2.76 ; [2, 3, 5]@2.88 ; [2, 3, 4, 5]@3.11 ; [0, 2, 3, 4, 5]@3.14 ; [0, 2, 3, 4, 5, 7]@3.21

## 7. Type-b margin (margin_typeb)
_margin_typeb not run_
