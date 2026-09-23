# ATLAS — atlas_v1_resnet20_s2
built 2026-09-23 02:13:14 · source **real** · arch `cifar10_resnet20` · weights `file:/workspace/models/resnet20_s2_chenyaofo.pt sha256:e7e5386fe7d26961` · layers 11

## 1. Per-layer invariants
| layer | D | PR | eff.rank | dim95 | TwoNN ID | sep ratio | NC1 | ETF dev | hub skew | test sparse | NC acc(test) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | 16 | 3.0 | 3.7 | 4 | 7.0 | 0.16 | 152.92 | 0.658 | 0.21 | 0.052 | 0.246 |
| layer1.0 | 16 | 3.9 | 5.4 | 7 | 9.2 | 0.19 | 136.17 | 0.613 | 0.74 | 0.054 | 0.288 |
| layer1.1 | 16 | 5.1 | 6.9 | 8 | 9.8 | 0.21 | 68.46 | 0.575 | 0.77 | 0.058 | 0.324 |
| layer1.2 | 16 | 4.5 | 6.7 | 9 | 10.1 | 0.30 | 47.77 | 0.452 | 0.82 | 0.059 | 0.349 |
| layer2.0 | 32 | 5.6 | 8.5 | 11 | 11.4 | 0.34 | 19.42 | 0.451 | 0.91 | 0.049 | 0.413 |
| layer2.1 | 32 | 7.1 | 10.8 | 14 | 12.9 | 0.38 | 11.10 | 0.400 | 1.15 | 0.044 | 0.469 |
| layer2.2 | 32 | 7.9 | 12.3 | 17 | 13.8 | 0.42 | 7.36 | 0.391 | 1.44 | 0.046 | 0.523 |
| layer3.0 | 64 | 12.1 | 21.8 | 38 | 18.3 | 0.58 | 2.53 | 0.334 | 1.70 | 0.045 | 0.667 |
| layer3.1 | 64 | 9.6 | 20.2 | 40 | 19.6 | 0.81 | 0.86 | 0.290 | 1.81 | 0.055 | 0.810 |
| layer3.2 | 64 | 8.5 | 9.7 | 9 | 10.0 | 3.02 | 0.17 | 0.107 | 0.77 | 0.115 | 0.925 |
| penult | 64 | 8.5 | 9.7 | 9 | 10.0 | 3.02 | 0.17 | 0.107 | 0.77 | 0.115 | 0.925 |

## 2. Decodability profile (which factor lives where)
categorical = accuracy minus majority-class rate; continuous = CV R²

| factor | kind | stem | layer1.0 | layer1.1 | layer1.2 | layer2.0 | layer2.1 | layer2.2 | layer3.0 | layer3.1 | layer3.2 | penult | commit | peak | washout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| luminance_mean | continuous | 0.99 | 0.98 | 0.97 | 0.95 | 0.92 | 0.90 | 0.87 | 0.75 | 0.61 | 0.31 | 0.31 | stem | stem | 0.68 |
| contrast_rms | continuous | 0.67 | 0.63 | 0.55 | 0.54 | 0.67 | 0.65 | 0.64 | 0.61 | 0.53 | 0.38 | 0.38 | stem | stem | 0.30 |
| highfreq_ratio | continuous | 0.23 | 0.49 | 0.53 | 0.52 | 0.62 | 0.60 | 0.61 | 0.62 | 0.58 | 0.50 | 0.50 | layer2.0 | layer3.0 | 0.12 |
| spectral_slope | continuous | 0.13 | 0.36 | 0.47 | 0.53 | 0.66 | 0.65 | 0.65 | 0.64 | 0.59 | 0.48 | 0.48 | layer2.0 | layer2.0 | 0.18 |
| spectral_anisotropy | continuous | 0.21 | 0.28 | 0.29 | 0.29 | 0.34 | 0.34 | 0.34 | 0.34 | 0.33 | 0.29 | 0.29 | layer2.0 | layer2.1 | 0.05 |
| noise_sigma | continuous | 0.66 | 0.89 | 0.91 | 0.94 | 0.94 | 0.93 | 0.91 | 0.89 | 0.85 | 0.69 | 0.69 | layer1.0 | layer1.2 | 0.25 |
| saturation_mean | continuous | 0.78 | 0.82 | 0.72 | 0.56 | 0.69 | 0.67 | 0.65 | 0.56 | 0.45 | 0.28 | 0.28 | stem | layer1.0 | 0.54 |
| hue_cos | continuous | 0.81 | 0.83 | 0.82 | 0.81 | 0.78 | 0.76 | 0.74 | 0.64 | 0.57 | 0.48 | 0.48 | stem | layer1.0 | 0.34 |
| hue_sin | continuous | 0.81 | 0.82 | 0.76 | 0.74 | 0.74 | 0.73 | 0.72 | 0.64 | 0.58 | 0.47 | 0.47 | stem | layer1.0 | 0.35 |
| colorfulness | continuous | 0.70 | 0.88 | 0.84 | 0.78 | 0.81 | 0.79 | 0.77 | 0.68 | 0.51 | 0.32 | 0.32 | layer1.0 | layer1.0 | 0.56 |
| edge_density | continuous | 0.74 | 0.88 | 0.80 | 0.86 | 0.88 | 0.87 | 0.85 | 0.82 | 0.76 | 0.61 | 0.61 | layer1.0 | layer2.0 | 0.28 |
| orientation_entropy | continuous | 0.21 | 0.32 | 0.40 | 0.41 | 0.47 | 0.48 | 0.51 | 0.52 | 0.46 | 0.41 | 0.41 | layer2.0 | layer3.0 | 0.11 |
| blockiness | continuous | 0.01 | 0.02 | 0.03 | 0.04 | 0.08 | 0.09 | 0.10 | 0.12 | 0.11 | 0.06 | 0.06 | layer3.0 | layer3.0 | 0.05 |
| class | categorical | 0.24 | 0.28 | 0.31 | 0.37 | 0.49 | 0.53 | 0.56 | 0.68 | 0.76 | 0.82 | 0.82 | layer3.1 | layer3.2 | 0.00 |
| coarse_animal_vehicle | categorical | 0.24 | 0.26 | 0.27 | 0.28 | 0.31 | 0.32 | 0.34 | 0.36 | 0.38 | 0.39 | 0.39 | layer3.0 | layer3.2 | 0.00 |
| corruption_family | categorical | 0.13 | 0.25 | 0.32 | 0.36 | 0.39 | 0.38 | 0.37 | 0.36 | 0.30 | 0.24 | 0.24 | layer1.2 | layer2.0 | 0.15 |
| corruption_type | categorical | 0.15 | 0.27 | 0.40 | 0.48 | 0.54 | 0.55 | 0.55 | 0.48 | 0.43 | 0.29 | 0.29 | layer2.0 | layer2.1 | 0.26 |
| severity | continuous | 0.06 | 0.12 | 0.08 | 0.18 | 0.25 | 0.27 | 0.28 | 0.30 | 0.29 | 0.24 | 0.24 | layer2.2 | layer3.0 | 0.07 |

## 3. Layer flow (CKA)
| pair | CKA |
|---|---|
| stem->layer1.0 | 0.868 |
| layer1.0->layer1.1 | 0.833 |
| layer1.1->layer1.2 | 0.716 |
| layer1.2->layer2.0 | 0.896 |
| layer2.0->layer2.1 | 0.968 |
| layer2.1->layer2.2 | 0.971 |
| layer2.2->layer3.0 | 0.844 |
| layer3.0->layer3.1 | 0.732 |
| layer3.1->layer3.2 | 0.676 |
| layer3.2->penult | 1.000 |

biggest reorganization: `layer3.1->layer3.2` (drop 0.324)

## 4. Sensitivity field (paired corruption displacement)
layer `penult`; magnitude in within-class-radius units

| corruption | sev | magnitude | per-sample | top5 frac | class-sub frac | coherence | class consistency | norm ratio |
|---|---|---|---|---|---|---|---|---|
| brightness | 1 | 0.03 | 0.14 | 0.87 | 0.93 | 0.20 | 0.61 | 1.001 |
| brightness | 3 | 0.10 | 0.41 | 0.82 | 0.95 | 0.22 | 0.49 | 0.993 |
| brightness | 5 | 0.21 | 0.77 | 0.72 | 0.96 | 0.25 | 0.30 | 0.962 |
| contrast | 1 | 0.10 | 0.35 | 0.59 | 0.97 | 0.27 | 0.59 | 1.000 |
| contrast | 3 | 0.38 | 1.11 | 0.46 | 0.93 | 0.33 | 0.29 | 0.937 |
| contrast | 5 | 0.97 | 2.38 | 0.42 | 0.78 | 0.40 | 0.13 | 0.724 |
| defocus_blur | 1 | 0.05 | 0.18 | 0.73 | 0.95 | 0.25 | 0.62 | 1.005 |
| defocus_blur | 3 | 0.38 | 0.93 | 0.87 | 0.96 | 0.38 | 0.50 | 0.969 |
| defocus_blur | 5 | 0.83 | 1.88 | 0.79 | 0.94 | 0.42 | 0.24 | 0.868 |
| elastic_transform | 1 | 0.19 | 0.88 | 0.86 | 0.96 | 0.20 | 0.17 | 0.957 |
| elastic_transform | 3 | 0.38 | 1.10 | 0.86 | 0.97 | 0.31 | 0.32 | 0.947 |
| elastic_transform | 5 | 0.46 | 1.44 | 0.75 | 0.98 | 0.30 | 0.11 | 0.910 |
| fog | 1 | 0.07 | 0.25 | 0.60 | 0.97 | 0.25 | 0.61 | 1.002 |
| fog | 3 | 0.24 | 0.74 | 0.55 | 0.95 | 0.31 | 0.45 | 0.982 |
| fog | 5 | 0.68 | 1.56 | 0.55 | 0.96 | 0.42 | 0.33 | 0.923 |
| frost | 1 | 0.26 | 0.76 | 0.61 | 0.94 | 0.31 | 0.31 | 0.955 |
| frost | 3 | 0.61 | 1.46 | 0.67 | 0.93 | 0.39 | 0.25 | 0.882 |
| frost | 5 | 0.84 | 1.81 | 0.71 | 0.92 | 0.44 | 0.25 | 0.848 |
| gaussian_noise | 1 | 0.66 | 1.34 | 0.84 | 0.94 | 0.47 | 0.41 | 0.899 |
| gaussian_noise | 3 | 1.24 | 2.27 | 0.87 | 0.93 | 0.52 | 0.31 | 0.833 |
| gaussian_noise | 5 | 1.43 | 2.52 | 0.85 | 0.93 | 0.55 | 0.32 | 0.837 |
| glass_blur | 1 | 0.75 | 1.91 | 0.69 | 0.93 | 0.36 | 0.12 | 0.819 |
| glass_blur | 3 | 0.67 | 1.79 | 0.70 | 0.93 | 0.34 | 0.11 | 0.829 |
| glass_blur | 5 | 0.76 | 2.03 | 0.70 | 0.93 | 0.34 | 0.10 | 0.804 |
| impulse_noise | 1 | 0.41 | 1.09 | 0.68 | 0.96 | 0.32 | 0.28 | 0.938 |
| impulse_noise | 3 | 0.67 | 1.81 | 0.74 | 0.94 | 0.35 | 0.13 | 0.866 |
| impulse_noise | 5 | 1.29 | 2.57 | 0.60 | 0.94 | 0.49 | 0.24 | 0.863 |
| jpeg_compression | 1 | 0.27 | 0.97 | 0.57 | 0.96 | 0.27 | 0.19 | 0.940 |
| jpeg_compression | 3 | 0.33 | 1.26 | 0.54 | 0.96 | 0.24 | 0.06 | 0.907 |
| jpeg_compression | 5 | 0.36 | 1.44 | 0.56 | 0.96 | 0.23 | 0.03 | 0.892 |
| motion_blur | 1 | 0.29 | 0.85 | 0.82 | 0.96 | 0.32 | 0.44 | 0.969 |
| motion_blur | 3 | 0.60 | 1.52 | 0.79 | 0.95 | 0.36 | 0.22 | 0.895 |
| motion_blur | 5 | 0.70 | 1.75 | 0.76 | 0.95 | 0.37 | 0.18 | 0.868 |
| pixelate | 1 | 0.25 | 0.65 | 0.79 | 0.97 | 0.34 | 0.51 | 0.958 |
| pixelate | 3 | 0.67 | 1.29 | 0.68 | 0.97 | 0.47 | 0.48 | 0.897 |
| pixelate | 5 | 1.28 | 2.24 | 0.45 | 0.93 | 0.54 | 0.36 | 0.836 |
| shot_noise | 1 | 0.50 | 1.02 | 0.81 | 0.94 | 0.46 | 0.52 | 0.932 |
| shot_noise | 3 | 1.05 | 1.97 | 0.86 | 0.94 | 0.51 | 0.33 | 0.854 |
| shot_noise | 5 | 1.34 | 2.42 | 0.82 | 0.93 | 0.53 | 0.31 | 0.843 |
| snow | 1 | 0.20 | 0.79 | 0.52 | 0.95 | 0.22 | 0.15 | 0.957 |
| snow | 3 | 0.26 | 1.14 | 0.38 | 0.93 | 0.22 | 0.06 | 0.930 |
| snow | 5 | 0.45 | 1.44 | 0.36 | 0.96 | 0.30 | 0.13 | 0.901 |
| zoom_blur | 1 | 0.40 | 1.03 | 0.96 | 0.97 | 0.36 | 0.42 | 0.961 |
| zoom_blur | 3 | 0.61 | 1.41 | 0.92 | 0.96 | 0.40 | 0.34 | 0.920 |
| zoom_blur | 5 | 0.82 | 1.82 | 0.86 | 0.96 | 0.42 | 0.28 | 0.883 |

## 5. Density per split
layer `penult`

| split | sparse frac (ref q95) | median log-radius shift |
|---|---|---|
| corrupt__brightness__s1 | 0.119 | 0.034 |
| corrupt__brightness__s3 | 0.132 | 0.056 |
| corrupt__brightness__s5 | 0.172 | 0.117 |
| corrupt__contrast__s1 | 0.112 | 0.039 |
| corrupt__contrast__s3 | 0.242 | 0.196 |
| corrupt__contrast__s5 | 0.773 | 0.528 |
| corrupt__defocus_blur__s1 | 0.115 | 0.040 |
| corrupt__defocus_blur__s3 | 0.202 | 0.172 |
| corrupt__defocus_blur__s5 | 0.408 | 0.368 |
| corrupt__elastic_transform__s1 | 0.186 | 0.132 |
| corrupt__elastic_transform__s3 | 0.222 | 0.201 |
| corrupt__elastic_transform__s5 | 0.353 | 0.314 |
| corrupt__fog__s1 | 0.117 | 0.036 |
| corrupt__fog__s3 | 0.170 | 0.124 |
| corrupt__fog__s5 | 0.394 | 0.345 |
| corrupt__frost__s1 | 0.177 | 0.120 |
| corrupt__frost__s3 | 0.346 | 0.310 |
| corrupt__frost__s5 | 0.441 | 0.381 |
| corrupt__gaussian_noise__s1 | 0.286 | 0.266 |
| corrupt__gaussian_noise__s3 | 0.484 | 0.408 |
| corrupt__gaussian_noise__s5 | 0.505 | 0.417 |
| corrupt__glass_blur__s1 | 0.443 | 0.383 |
| corrupt__glass_blur__s3 | 0.419 | 0.364 |
| corrupt__glass_blur__s5 | 0.459 | 0.395 |
| corrupt__impulse_noise__s1 | 0.266 | 0.229 |
| corrupt__impulse_noise__s3 | 0.579 | 0.461 |
| corrupt__impulse_noise__s5 | 0.727 | 0.528 |
| corrupt__jpeg_compression__s1 | 0.203 | 0.163 |
| corrupt__jpeg_compression__s3 | 0.277 | 0.249 |
| corrupt__jpeg_compression__s5 | 0.346 | 0.313 |
| corrupt__motion_blur__s1 | 0.188 | 0.141 |
| corrupt__motion_blur__s3 | 0.328 | 0.316 |
| corrupt__motion_blur__s5 | 0.375 | 0.346 |
| corrupt__pixelate__s1 | 0.170 | 0.080 |
| corrupt__pixelate__s3 | 0.328 | 0.244 |
| corrupt__pixelate__s5 | 0.545 | 0.454 |
| corrupt__shot_noise__s1 | 0.213 | 0.173 |
| corrupt__shot_noise__s3 | 0.474 | 0.399 |
| corrupt__shot_noise__s5 | 0.556 | 0.436 |
| corrupt__snow__s1 | 0.177 | 0.139 |
| corrupt__snow__s3 | 0.278 | 0.245 |
| corrupt__snow__s5 | 0.353 | 0.309 |
| corrupt__zoom_blur__s1 | 0.251 | 0.220 |
| corrupt__zoom_blur__s3 | 0.313 | 0.295 |
| corrupt__zoom_blur__s5 | 0.398 | 0.359 |
| ood__cifar100 | 0.494 | 0.411 |
| ood__svhn | 0.718 | 0.532 |
| panel | 0.000 | -0.070 |
| test | 0.115 | 0.033 |

## 6. Adjacency
layer `penult`


closest class pairs (sep = center distance / RMS radius): (3,5) 2.41, (2,3) 2.68, (2,4) 2.73, (0,2) 2.83, (3,4) 2.83, (2,5) 2.84

valley ratio mean 2.31, min 1.48, pairs with a valley 1.00

nearest-center confusions (true → nearest): 5→3 0.09, 3→5 0.05, 6→3 0.03, 9→1 0.03, 2→6 0.03, 0→8 0.02

single-linkage merge order (first 5): [3, 5]@2.41 ; [2, 3, 5]@2.68 ; [2, 3, 4, 5]@2.73 ; [0, 2, 3, 4, 5]@2.83 ; [0, 2, 3, 4, 5, 7]@2.86
