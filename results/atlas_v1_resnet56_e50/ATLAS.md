# ATLAS — atlas_v1_resnet56_e50
built 2026-09-23 14:05:13 · source **real** · arch `cifar10_resnet56` · weights `file:/workspace/models/resnet56_s11_e50_chenyaofo.pt sha256:b135fdac0dbe43be` · layers 11

## 1. Per-layer invariants
| layer | D | PR | eff.rank | dim95 | TwoNN ID | sep ratio | NC1 | ETF dev | hub skew | test sparse | NC acc(test) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | 16 | 2.9 | 3.6 | 4 | 6.6 | 0.16 | 302.32 | 0.658 | 0.25 | 0.058 | 0.250 |
| layer1.0 | 16 | 2.9 | 4.0 | 5 | 7.2 | 0.18 | 507.80 | 0.663 | 0.36 | 0.058 | 0.255 |
| layer1.5 | 16 | 3.9 | 5.5 | 7 | 8.7 | 0.21 | 114.24 | 0.574 | 0.63 | 0.058 | 0.311 |
| layer1.8 | 16 | 5.0 | 6.8 | 8 | 9.5 | 0.29 | 81.02 | 0.485 | 0.66 | 0.056 | 0.346 |
| layer2.0 | 32 | 5.3 | 7.7 | 9 | 10.7 | 0.33 | 31.65 | 0.470 | 0.72 | 0.055 | 0.397 |
| layer2.5 | 32 | 6.4 | 9.7 | 13 | 12.8 | 0.43 | 11.80 | 0.424 | 1.04 | 0.053 | 0.478 |
| layer2.8 | 32 | 7.2 | 11.0 | 15 | 13.5 | 0.46 | 7.85 | 0.408 | 1.20 | 0.049 | 0.520 |
| layer3.0 | 64 | 8.8 | 14.5 | 22 | 15.1 | 0.51 | 4.02 | 0.379 | 1.50 | 0.053 | 0.591 |
| layer3.5 | 64 | 14.1 | 24.6 | 40 | 19.0 | 0.85 | 0.78 | 0.240 | 1.73 | 0.044 | 0.822 |
| layer3.8 | 64 | 8.9 | 10.5 | 9 | 11.0 | 3.14 | 0.14 | 0.092 | 1.03 | 0.127 | 0.921 |
| penult | 64 | 8.9 | 10.5 | 9 | 11.0 | 3.14 | 0.14 | 0.092 | 1.03 | 0.127 | 0.921 |

## 2. Decodability profile (which factor lives where)
categorical = accuracy minus majority-class rate; continuous = CV R²

| factor | kind | stem | layer1.0 | layer1.5 | layer1.8 | layer2.0 | layer2.5 | layer2.8 | layer3.0 | layer3.5 | layer3.8 | penult | commit | peak | washout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| luminance_mean | continuous | 1.00 | 0.99 | 0.97 | 0.94 | 0.95 | 0.91 | 0.89 | 0.84 | 0.75 | 0.50 | 0.50 | stem | stem | 0.50 |
| contrast_rms | continuous | 0.57 | 0.52 | 0.51 | 0.51 | 0.67 | 0.62 | 0.60 | 0.65 | 0.58 | 0.44 | 0.44 | layer2.0 | layer2.0 | 0.23 |
| highfreq_ratio | continuous | 0.29 | 0.28 | 0.52 | 0.52 | 0.60 | 0.60 | 0.60 | 0.64 | 0.59 | 0.50 | 0.50 | layer2.0 | layer3.0 | 0.14 |
| spectral_slope | continuous | 0.16 | 0.17 | 0.45 | 0.52 | 0.65 | 0.65 | 0.64 | 0.66 | 0.61 | 0.50 | 0.50 | layer2.0 | layer3.0 | 0.16 |
| spectral_anisotropy | continuous | 0.17 | 0.18 | 0.32 | 0.30 | 0.33 | 0.33 | 0.33 | 0.36 | 0.34 | 0.31 | 0.31 | layer2.0 | layer3.0 | 0.05 |
| noise_sigma | continuous | 0.69 | 0.67 | 0.93 | 0.92 | 0.96 | 0.94 | 0.93 | 0.92 | 0.87 | 0.74 | 0.74 | layer1.5 | layer2.0 | 0.22 |
| saturation_mean | continuous | 0.76 | 0.75 | 0.75 | 0.70 | 0.70 | 0.61 | 0.57 | 0.57 | 0.43 | 0.31 | 0.31 | stem | stem | 0.46 |
| hue_cos | continuous | 0.79 | 0.83 | 0.82 | 0.80 | 0.79 | 0.76 | 0.76 | 0.71 | 0.63 | 0.53 | 0.53 | stem | layer1.0 | 0.30 |
| hue_sin | continuous | 0.78 | 0.80 | 0.83 | 0.81 | 0.75 | 0.73 | 0.73 | 0.72 | 0.68 | 0.56 | 0.56 | stem | layer1.5 | 0.26 |
| colorfulness | continuous | 0.68 | 0.72 | 0.84 | 0.84 | 0.82 | 0.75 | 0.70 | 0.67 | 0.47 | 0.31 | 0.31 | layer1.5 | layer1.5 | 0.53 |
| edge_density | continuous | 0.65 | 0.64 | 0.86 | 0.81 | 0.91 | 0.88 | 0.87 | 0.85 | 0.77 | 0.63 | 0.63 | layer1.5 | layer2.0 | 0.28 |
| orientation_entropy | continuous | 0.14 | 0.14 | 0.35 | 0.39 | 0.44 | 0.43 | 0.44 | 0.53 | 0.48 | 0.42 | 0.42 | layer3.0 | layer3.0 | 0.11 |
| blockiness | continuous | 0.01 | 0.00 | 0.03 | 0.05 | 0.07 | 0.08 | 0.08 | 0.09 | 0.10 | 0.05 | 0.05 | layer3.0 | layer3.5 | 0.05 |
| class | categorical | 0.24 | 0.23 | 0.28 | 0.35 | 0.46 | 0.52 | 0.55 | 0.65 | 0.76 | 0.81 | 0.81 | layer3.5 | layer3.8 | 0.00 |
| coarse_animal_vehicle | categorical | 0.23 | 0.23 | 0.27 | 0.28 | 0.31 | 0.32 | 0.33 | 0.35 | 0.37 | 0.39 | 0.39 | layer3.5 | layer3.8 | 0.00 |
| corruption_family | categorical | 0.11 | 0.13 | 0.25 | 0.34 | 0.38 | 0.35 | 0.35 | 0.35 | 0.31 | 0.25 | 0.25 | layer2.0 | layer2.0 | 0.13 |
| corruption_type | categorical | 0.14 | 0.14 | 0.34 | 0.46 | 0.52 | 0.53 | 0.52 | 0.49 | 0.42 | 0.29 | 0.29 | layer2.0 | layer2.5 | 0.24 |
| severity | continuous | 0.03 | 0.02 | 0.15 | 0.13 | 0.20 | 0.21 | 0.22 | 0.29 | 0.29 | 0.25 | 0.25 | layer3.0 | layer3.5 | 0.05 |

## 3. Layer flow (CKA)
| pair | CKA |
|---|---|
| stem->layer1.0 | 0.959 |
| layer1.0->layer1.5 | 0.900 |
| layer1.5->layer1.8 | 0.795 |
| layer1.8->layer2.0 | 0.955 |
| layer2.0->layer2.5 | 0.953 |
| layer2.5->layer2.8 | 0.984 |
| layer2.8->layer3.0 | 0.937 |
| layer3.0->layer3.5 | 0.804 |
| layer3.5->layer3.8 | 0.694 |
| layer3.8->penult | 1.000 |

biggest reorganization: `layer3.5->layer3.8` (drop 0.306)

## 4. Sensitivity field (paired corruption displacement)
layer `penult`; magnitude in within-class-radius units

| corruption | sev | magnitude | per-sample | top5 frac | class-sub frac | coherence | class consistency | norm ratio |
|---|---|---|---|---|---|---|---|---|
| brightness | 1 | 0.04 | 0.14 | 0.67 | 0.78 | 0.27 | 0.66 | 1.004 |
| brightness | 3 | 0.12 | 0.40 | 0.68 | 0.85 | 0.29 | 0.62 | 1.004 |
| brightness | 5 | 0.26 | 0.77 | 0.66 | 0.90 | 0.32 | 0.51 | 0.992 |
| contrast | 1 | 0.13 | 0.35 | 0.76 | 0.90 | 0.38 | 0.62 | 1.015 |
| contrast | 3 | 0.47 | 1.12 | 0.68 | 0.86 | 0.43 | 0.54 | 0.989 |
| contrast | 5 | 1.20 | 2.48 | 0.59 | 0.78 | 0.47 | 0.25 | 0.829 |
| defocus_blur | 1 | 0.06 | 0.17 | 0.77 | 0.86 | 0.33 | 0.58 | 1.006 |
| defocus_blur | 3 | 0.43 | 0.90 | 0.83 | 0.89 | 0.43 | 0.61 | 0.997 |
| defocus_blur | 5 | 0.96 | 1.88 | 0.85 | 0.89 | 0.48 | 0.40 | 0.930 |
| elastic_transform | 1 | 0.22 | 0.86 | 0.75 | 0.83 | 0.24 | 0.32 | 0.975 |
| elastic_transform | 3 | 0.41 | 1.06 | 0.81 | 0.88 | 0.35 | 0.46 | 0.977 |
| elastic_transform | 5 | 0.46 | 1.41 | 0.64 | 0.90 | 0.32 | 0.20 | 0.940 |
| fog | 1 | 0.08 | 0.24 | 0.84 | 0.88 | 0.33 | 0.61 | 1.010 |
| fog | 3 | 0.28 | 0.72 | 0.78 | 0.88 | 0.40 | 0.63 | 1.006 |
| fog | 5 | 0.68 | 1.50 | 0.82 | 0.91 | 0.45 | 0.42 | 0.936 |
| frost | 1 | 0.32 | 0.77 | 0.85 | 0.90 | 0.39 | 0.50 | 0.976 |
| frost | 3 | 0.78 | 1.55 | 0.82 | 0.91 | 0.47 | 0.43 | 0.934 |
| frost | 5 | 1.04 | 1.92 | 0.83 | 0.91 | 0.52 | 0.42 | 0.904 |
| gaussian_noise | 1 | 0.69 | 1.30 | 0.89 | 0.89 | 0.51 | 0.50 | 0.913 |
| gaussian_noise | 3 | 1.33 | 2.23 | 0.87 | 0.88 | 0.57 | 0.43 | 0.831 |
| gaussian_noise | 5 | 1.51 | 2.50 | 0.84 | 0.86 | 0.58 | 0.40 | 0.799 |
| glass_blur | 1 | 0.91 | 1.99 | 0.83 | 0.92 | 0.41 | 0.24 | 0.862 |
| glass_blur | 3 | 0.83 | 1.87 | 0.85 | 0.93 | 0.41 | 0.23 | 0.875 |
| glass_blur | 5 | 0.97 | 2.12 | 0.82 | 0.93 | 0.42 | 0.22 | 0.860 |
| impulse_noise | 1 | 0.35 | 0.98 | 0.54 | 0.84 | 0.32 | 0.30 | 0.961 |
| impulse_noise | 3 | 0.65 | 1.68 | 0.67 | 0.81 | 0.37 | 0.19 | 0.885 |
| impulse_noise | 5 | 1.29 | 2.46 | 0.80 | 0.83 | 0.50 | 0.27 | 0.776 |
| jpeg_compression | 1 | 0.34 | 0.95 | 0.80 | 0.91 | 0.34 | 0.36 | 0.958 |
| jpeg_compression | 3 | 0.42 | 1.27 | 0.75 | 0.90 | 0.32 | 0.21 | 0.936 |
| jpeg_compression | 5 | 0.48 | 1.43 | 0.74 | 0.90 | 0.32 | 0.18 | 0.931 |
| motion_blur | 1 | 0.29 | 0.79 | 0.75 | 0.89 | 0.34 | 0.52 | 0.988 |
| motion_blur | 3 | 0.62 | 1.46 | 0.82 | 0.90 | 0.40 | 0.37 | 0.940 |
| motion_blur | 5 | 0.73 | 1.71 | 0.82 | 0.89 | 0.40 | 0.31 | 0.918 |
| pixelate | 1 | 0.25 | 0.61 | 0.82 | 0.92 | 0.37 | 0.55 | 0.973 |
| pixelate | 3 | 0.65 | 1.21 | 0.75 | 0.93 | 0.49 | 0.55 | 0.940 |
| pixelate | 5 | 1.25 | 2.15 | 0.64 | 0.90 | 0.55 | 0.42 | 0.901 |
| shot_noise | 1 | 0.53 | 1.00 | 0.89 | 0.91 | 0.50 | 0.56 | 0.939 |
| shot_noise | 3 | 1.14 | 1.94 | 0.88 | 0.88 | 0.56 | 0.46 | 0.853 |
| shot_noise | 5 | 1.43 | 2.37 | 0.84 | 0.85 | 0.58 | 0.41 | 0.804 |
| snow | 1 | 0.22 | 0.80 | 0.29 | 0.82 | 0.26 | 0.24 | 0.979 |
| snow | 3 | 0.34 | 1.19 | 0.41 | 0.81 | 0.29 | 0.18 | 0.952 |
| snow | 5 | 0.60 | 1.50 | 0.65 | 0.89 | 0.39 | 0.30 | 0.942 |
| zoom_blur | 1 | 0.38 | 0.93 | 0.86 | 0.88 | 0.37 | 0.52 | 0.977 |
| zoom_blur | 3 | 0.63 | 1.32 | 0.83 | 0.89 | 0.44 | 0.49 | 0.957 |
| zoom_blur | 5 | 0.90 | 1.79 | 0.81 | 0.90 | 0.47 | 0.42 | 0.932 |

## 5. Density per split
layer `penult`

| split | sparse frac (ref q95) | median log-radius shift |
|---|---|---|
| corrupt__brightness__s1 | 0.140 | 0.016 |
| corrupt__brightness__s3 | 0.157 | 0.044 |
| corrupt__brightness__s5 | 0.206 | 0.115 |
| corrupt__contrast__s1 | 0.137 | 0.026 |
| corrupt__contrast__s3 | 0.263 | 0.207 |
| corrupt__contrast__s5 | 0.799 | 0.567 |
| corrupt__defocus_blur__s1 | 0.132 | 0.022 |
| corrupt__defocus_blur__s3 | 0.222 | 0.145 |
| corrupt__defocus_blur__s5 | 0.474 | 0.390 |
| corrupt__elastic_transform__s1 | 0.200 | 0.132 |
| corrupt__elastic_transform__s3 | 0.244 | 0.199 |
| corrupt__elastic_transform__s5 | 0.401 | 0.334 |
| corrupt__fog__s1 | 0.131 | 0.021 |
| corrupt__fog__s3 | 0.176 | 0.088 |
| corrupt__fog__s5 | 0.375 | 0.307 |
| corrupt__frost__s1 | 0.210 | 0.112 |
| corrupt__frost__s3 | 0.428 | 0.349 |
| corrupt__frost__s5 | 0.535 | 0.434 |
| corrupt__gaussian_noise__s1 | 0.336 | 0.278 |
| corrupt__gaussian_noise__s3 | 0.551 | 0.448 |
| corrupt__gaussian_noise__s5 | 0.538 | 0.431 |
| corrupt__glass_blur__s1 | 0.546 | 0.438 |
| corrupt__glass_blur__s3 | 0.518 | 0.421 |
| corrupt__glass_blur__s5 | 0.592 | 0.462 |
| corrupt__impulse_noise__s1 | 0.285 | 0.201 |
| corrupt__impulse_noise__s3 | 0.523 | 0.427 |
| corrupt__impulse_noise__s5 | 0.677 | 0.533 |
| corrupt__jpeg_compression__s1 | 0.242 | 0.164 |
| corrupt__jpeg_compression__s3 | 0.336 | 0.277 |
| corrupt__jpeg_compression__s5 | 0.383 | 0.317 |
| corrupt__motion_blur__s1 | 0.187 | 0.118 |
| corrupt__motion_blur__s3 | 0.338 | 0.293 |
| corrupt__motion_blur__s5 | 0.407 | 0.348 |
| corrupt__pixelate__s1 | 0.204 | 0.092 |
| corrupt__pixelate__s3 | 0.365 | 0.257 |
| corrupt__pixelate__s5 | 0.613 | 0.503 |
| corrupt__shot_noise__s1 | 0.253 | 0.182 |
| corrupt__shot_noise__s3 | 0.530 | 0.427 |
| corrupt__shot_noise__s5 | 0.560 | 0.440 |
| corrupt__snow__s1 | 0.221 | 0.161 |
| corrupt__snow__s3 | 0.344 | 0.278 |
| corrupt__snow__s5 | 0.441 | 0.363 |
| corrupt__zoom_blur__s1 | 0.239 | 0.170 |
| corrupt__zoom_blur__s3 | 0.347 | 0.290 |
| corrupt__zoom_blur__s5 | 0.485 | 0.399 |
| ood__cifar100 | 0.614 | 0.467 |
| ood__svhn | 0.813 | 0.583 |
| panel | 0.047 | 0.011 |
| test | 0.127 | 0.021 |

## 6. Adjacency
layer `penult`


closest class pairs (sep = center distance / RMS radius): (3,5) 2.46, (2,3) 2.75, (2,4) 2.89, (3,4) 2.92, (0,2) 2.92, (3,6) 2.96

valley ratio mean 2.33, min 1.54, pairs with a valley 1.00

nearest-center confusions (true → nearest): 3→5 0.07, 5→3 0.07, 3→4 0.03, 9→1 0.03, 0→8 0.03, 8→0 0.03

single-linkage merge order (first 5): [3, 5]@2.46 ; [2, 3, 5]@2.75 ; [2, 3, 4, 5]@2.89 ; [0, 2, 3, 4, 5]@2.92 ; [0, 2, 3, 4, 5, 6]@2.96

## 7. Type-b margin (margin_typeb)
_margin_typeb not run_
