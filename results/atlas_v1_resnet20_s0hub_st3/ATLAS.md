# ATLAS — atlas_v1_resnet20_s0hub_st3
built 2026-09-23 13:37:52 · source **real** · arch `cifar10_resnet20` · weights `chenyaofo cifar10_resnet20` · layers 11

## 1. Per-layer invariants
| layer | D | PR | eff.rank | dim95 | TwoNN ID | sep ratio | NC1 | ETF dev | hub skew | test sparse | NC acc(test) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | 16 | 3.0 | 3.6 | 4 | 6.8 | 0.16 | 259.93 | 0.627 | 0.21 | 0.059 | 0.243 |
| layer1.0 | 16 | 4.0 | 5.4 | 6 | 8.8 | 0.18 | 138.34 | 0.581 | 0.63 | 0.055 | 0.289 |
| layer1.1 | 16 | 5.2 | 7.0 | 8 | 9.9 | 0.21 | 57.98 | 0.494 | 0.80 | 0.055 | 0.335 |
| layer1.2 | 16 | 5.5 | 7.9 | 9 | 10.2 | 0.29 | 53.17 | 0.464 | 0.87 | 0.053 | 0.360 |
| layer2.0 | 32 | 6.8 | 10.2 | 13 | 12.1 | 0.33 | 21.89 | 0.458 | 0.96 | 0.047 | 0.426 |
| layer2.1 | 32 | 7.8 | 11.8 | 15 | 13.2 | 0.39 | 12.41 | 0.440 | 1.24 | 0.049 | 0.491 |
| layer2.2 | 32 | 9.0 | 13.6 | 18 | 14.1 | 0.43 | 7.06 | 0.405 | 1.24 | 0.047 | 0.539 |
| layer3.0 | 64 | 12.1 | 21.6 | 37 | 18.1 | 0.57 | 2.42 | 0.308 | 1.83 | 0.044 | 0.650 |
| layer3.1 | 64 | 10.5 | 20.9 | 41 | 19.4 | 0.90 | 0.76 | 0.275 | 1.91 | 0.054 | 0.802 |
| layer3.2 | 64 | 8.5 | 9.6 | 9 | 9.9 | 3.00 | 0.17 | 0.107 | 0.76 | 0.108 | 0.923 |
| penult | 64 | 8.5 | 9.6 | 9 | 9.9 | 3.00 | 0.17 | 0.107 | 0.76 | 0.108 | 0.923 |

## 2. Decodability profile (which factor lives where)
categorical = accuracy minus majority-class rate; continuous = CV R²

| factor | kind | stem | layer1.0 | layer1.1 | layer1.2 | layer2.0 | layer2.1 | layer2.2 | layer3.0 | layer3.1 | layer3.2 | penult | commit | peak | washout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| luminance_mean | continuous | 0.99 | 0.96 | 0.96 | 0.96 | 0.92 | 0.91 | 0.87 | 0.77 | 0.68 | 0.46 | 0.46 | stem | stem | 0.53 |
| contrast_rms | continuous | 0.66 | 0.62 | 0.63 | 0.62 | 0.69 | 0.68 | 0.65 | 0.59 | 0.51 | 0.37 | 0.37 | stem | layer2.0 | 0.32 |
| highfreq_ratio | continuous | 0.23 | 0.58 | 0.59 | 0.54 | 0.63 | 0.62 | 0.61 | 0.60 | 0.54 | 0.48 | 0.48 | layer1.0 | layer2.0 | 0.15 |
| spectral_slope | continuous | 0.12 | 0.53 | 0.56 | 0.56 | 0.66 | 0.65 | 0.65 | 0.64 | 0.58 | 0.48 | 0.48 | layer2.0 | layer2.0 | 0.19 |
| spectral_anisotropy | continuous | 0.24 | 0.27 | 0.28 | 0.29 | 0.32 | 0.31 | 0.32 | 0.33 | 0.31 | 0.28 | 0.28 | layer2.0 | layer3.0 | 0.05 |
| noise_sigma | continuous | 0.63 | 0.96 | 0.93 | 0.95 | 0.96 | 0.94 | 0.92 | 0.89 | 0.83 | 0.70 | 0.70 | layer1.0 | layer1.0 | 0.27 |
| saturation_mean | continuous | 0.78 | 0.76 | 0.73 | 0.64 | 0.66 | 0.64 | 0.64 | 0.57 | 0.45 | 0.31 | 0.31 | stem | stem | 0.48 |
| hue_cos | continuous | 0.78 | 0.83 | 0.82 | 0.81 | 0.81 | 0.80 | 0.80 | 0.74 | 0.66 | 0.49 | 0.49 | stem | layer1.0 | 0.34 |
| hue_sin | continuous | 0.76 | 0.78 | 0.78 | 0.78 | 0.74 | 0.73 | 0.71 | 0.67 | 0.61 | 0.44 | 0.44 | stem | layer1.1 | 0.34 |
| colorfulness | continuous | 0.80 | 0.89 | 0.88 | 0.84 | 0.80 | 0.80 | 0.79 | 0.69 | 0.54 | 0.34 | 0.34 | layer1.0 | layer1.0 | 0.55 |
| edge_density | continuous | 0.71 | 0.90 | 0.86 | 0.89 | 0.90 | 0.88 | 0.85 | 0.82 | 0.73 | 0.58 | 0.58 | layer1.0 | layer2.0 | 0.32 |
| orientation_entropy | continuous | 0.25 | 0.34 | 0.35 | 0.41 | 0.46 | 0.45 | 0.49 | 0.50 | 0.46 | 0.41 | 0.41 | layer2.0 | layer3.0 | 0.08 |
| blockiness | continuous | 0.01 | 0.01 | 0.02 | 0.04 | 0.09 | 0.07 | 0.07 | 0.10 | 0.10 | 0.06 | 0.06 | layer3.0 | layer3.1 | 0.04 |
| class | categorical | 0.24 | 0.30 | 0.34 | 0.35 | 0.47 | 0.53 | 0.57 | 0.68 | 0.77 | 0.82 | 0.82 | layer3.1 | layer3.2 | 0.00 |
| coarse_animal_vehicle | categorical | 0.24 | 0.26 | 0.27 | 0.27 | 0.32 | 0.32 | 0.33 | 0.36 | 0.37 | 0.39 | 0.39 | layer3.0 | layer3.2 | 0.00 |
| corruption_family | categorical | 0.10 | 0.24 | 0.27 | 0.35 | 0.40 | 0.38 | 0.38 | 0.36 | 0.32 | 0.26 | 0.26 | layer2.0 | layer2.0 | 0.14 |
| corruption_type | categorical | 0.15 | 0.34 | 0.40 | 0.50 | 0.55 | 0.55 | 0.55 | 0.49 | 0.42 | 0.28 | 0.28 | layer1.2 | layer2.0 | 0.28 |
| severity | continuous | 0.05 | 0.15 | 0.15 | 0.17 | 0.27 | 0.27 | 0.26 | 0.31 | 0.29 | 0.22 | 0.22 | layer3.0 | layer3.0 | 0.09 |

## 3. Layer flow (CKA)
| pair | CKA |
|---|---|
| stem->layer1.0 | 0.920 |
| layer1.0->layer1.1 | 0.919 |
| layer1.1->layer1.2 | 0.791 |
| layer1.2->layer2.0 | 0.926 |
| layer2.0->layer2.1 | 0.965 |
| layer2.1->layer2.2 | 0.953 |
| layer2.2->layer3.0 | 0.868 |
| layer3.0->layer3.1 | 0.701 |
| layer3.1->layer3.2 | 0.693 |
| layer3.2->penult | 1.000 |

biggest reorganization: `layer3.1->layer3.2` (drop 0.307)

## 4. Sensitivity field (paired corruption displacement)
layer `penult`; magnitude in within-class-radius units

| corruption | sev | magnitude | per-sample | top5 frac | class-sub frac | coherence | class consistency | norm ratio |
|---|---|---|---|---|---|---|---|---|
| brightness | 1 | 0.03 | 0.15 | 0.90 | 0.94 | 0.18 | 0.62 | 1.000 |
| brightness | 3 | 0.09 | 0.42 | 0.79 | 0.94 | 0.21 | 0.44 | 0.989 |
| brightness | 5 | 0.21 | 0.79 | 0.65 | 0.95 | 0.25 | 0.26 | 0.957 |
| contrast | 1 | 0.09 | 0.34 | 0.50 | 0.90 | 0.26 | 0.55 | 0.992 |
| contrast | 3 | 0.37 | 1.10 | 0.46 | 0.86 | 0.32 | 0.25 | 0.907 |
| contrast | 5 | 1.04 | 2.44 | 0.52 | 0.82 | 0.41 | 0.13 | 0.716 |
| defocus_blur | 1 | 0.05 | 0.18 | 0.79 | 0.96 | 0.24 | 0.64 | 1.001 |
| defocus_blur | 3 | 0.35 | 0.92 | 0.84 | 0.95 | 0.36 | 0.44 | 0.959 |
| defocus_blur | 5 | 0.80 | 1.88 | 0.78 | 0.93 | 0.41 | 0.20 | 0.853 |
| elastic_transform | 1 | 0.18 | 0.89 | 0.85 | 0.95 | 0.18 | 0.11 | 0.951 |
| elastic_transform | 3 | 0.35 | 1.09 | 0.87 | 0.96 | 0.29 | 0.26 | 0.937 |
| elastic_transform | 5 | 0.47 | 1.46 | 0.84 | 0.98 | 0.29 | 0.12 | 0.914 |
| fog | 1 | 0.06 | 0.25 | 0.64 | 0.94 | 0.23 | 0.58 | 0.996 |
| fog | 3 | 0.22 | 0.73 | 0.63 | 0.93 | 0.29 | 0.38 | 0.960 |
| fog | 5 | 0.66 | 1.56 | 0.64 | 0.97 | 0.41 | 0.29 | 0.891 |
| frost | 1 | 0.24 | 0.76 | 0.49 | 0.94 | 0.28 | 0.28 | 0.959 |
| frost | 3 | 0.54 | 1.43 | 0.58 | 0.93 | 0.34 | 0.19 | 0.878 |
| frost | 5 | 0.75 | 1.76 | 0.68 | 0.92 | 0.39 | 0.20 | 0.838 |
| gaussian_noise | 1 | 0.62 | 1.35 | 0.85 | 0.95 | 0.42 | 0.34 | 0.890 |
| gaussian_noise | 3 | 1.25 | 2.32 | 0.83 | 0.87 | 0.51 | 0.29 | 0.771 |
| gaussian_noise | 5 | 1.41 | 2.55 | 0.79 | 0.83 | 0.53 | 0.29 | 0.741 |
| glass_blur | 1 | 0.72 | 1.96 | 0.73 | 0.94 | 0.33 | 0.09 | 0.825 |
| glass_blur | 3 | 0.64 | 1.82 | 0.76 | 0.95 | 0.31 | 0.08 | 0.838 |
| glass_blur | 5 | 0.74 | 2.05 | 0.71 | 0.95 | 0.32 | 0.08 | 0.816 |
| impulse_noise | 1 | 0.42 | 1.12 | 0.65 | 0.95 | 0.32 | 0.33 | 0.955 |
| impulse_noise | 3 | 0.57 | 1.78 | 0.81 | 0.95 | 0.29 | 0.07 | 0.866 |
| impulse_noise | 5 | 1.07 | 2.43 | 0.73 | 0.81 | 0.42 | 0.15 | 0.716 |
| jpeg_compression | 1 | 0.27 | 0.97 | 0.44 | 0.94 | 0.26 | 0.18 | 0.933 |
| jpeg_compression | 3 | 0.34 | 1.25 | 0.39 | 0.96 | 0.25 | 0.08 | 0.905 |
| jpeg_compression | 5 | 0.38 | 1.45 | 0.46 | 0.97 | 0.25 | 0.05 | 0.895 |
| motion_blur | 1 | 0.25 | 0.79 | 0.83 | 0.97 | 0.29 | 0.36 | 0.962 |
| motion_blur | 3 | 0.53 | 1.43 | 0.81 | 0.96 | 0.34 | 0.19 | 0.895 |
| motion_blur | 5 | 0.62 | 1.65 | 0.78 | 0.94 | 0.34 | 0.15 | 0.864 |
| pixelate | 1 | 0.25 | 0.65 | 0.75 | 0.98 | 0.35 | 0.57 | 0.965 |
| pixelate | 3 | 0.65 | 1.25 | 0.63 | 0.96 | 0.47 | 0.48 | 0.894 |
| pixelate | 5 | 1.29 | 2.23 | 0.46 | 0.93 | 0.54 | 0.38 | 0.845 |
| shot_noise | 1 | 0.48 | 1.05 | 0.83 | 0.96 | 0.41 | 0.47 | 0.933 |
| shot_noise | 3 | 1.07 | 2.03 | 0.84 | 0.91 | 0.49 | 0.31 | 0.810 |
| shot_noise | 5 | 1.33 | 2.44 | 0.79 | 0.85 | 0.52 | 0.29 | 0.759 |
| snow | 1 | 0.21 | 0.80 | 0.56 | 0.96 | 0.23 | 0.22 | 0.969 |
| snow | 3 | 0.28 | 1.18 | 0.30 | 0.96 | 0.23 | 0.07 | 0.928 |
| snow | 5 | 0.46 | 1.46 | 0.25 | 0.95 | 0.30 | 0.14 | 0.894 |
| zoom_blur | 1 | 0.33 | 0.97 | 0.93 | 0.96 | 0.30 | 0.27 | 0.939 |
| zoom_blur | 3 | 0.52 | 1.35 | 0.87 | 0.95 | 0.36 | 0.25 | 0.904 |
| zoom_blur | 5 | 0.76 | 1.77 | 0.81 | 0.94 | 0.40 | 0.23 | 0.866 |

## 5. Density per split
layer `penult`

| split | sparse frac (ref q95) | median log-radius shift |
|---|---|---|
| corrupt__brightness__s1 | 0.114 | 0.028 |
| corrupt__brightness__s3 | 0.128 | 0.050 |
| corrupt__brightness__s5 | 0.172 | 0.115 |
| corrupt__contrast__s1 | 0.109 | 0.033 |
| corrupt__contrast__s3 | 0.201 | 0.169 |
| corrupt__contrast__s5 | 0.682 | 0.487 |
| corrupt__defocus_blur__s1 | 0.108 | 0.032 |
| corrupt__defocus_blur__s3 | 0.194 | 0.157 |
| corrupt__defocus_blur__s5 | 0.445 | 0.380 |
| corrupt__elastic_transform__s1 | 0.178 | 0.132 |
| corrupt__elastic_transform__s3 | 0.222 | 0.195 |
| corrupt__elastic_transform__s5 | 0.381 | 0.324 |
| corrupt__fog__s1 | 0.107 | 0.027 |
| corrupt__fog__s3 | 0.139 | 0.088 |
| corrupt__fog__s5 | 0.343 | 0.306 |
| corrupt__frost__s1 | 0.172 | 0.111 |
| corrupt__frost__s3 | 0.316 | 0.288 |
| corrupt__frost__s5 | 0.385 | 0.341 |
| corrupt__gaussian_noise__s1 | 0.301 | 0.256 |
| corrupt__gaussian_noise__s3 | 0.397 | 0.351 |
| corrupt__gaussian_noise__s5 | 0.396 | 0.366 |
| corrupt__glass_blur__s1 | 0.466 | 0.391 |
| corrupt__glass_blur__s3 | 0.428 | 0.367 |
| corrupt__glass_blur__s5 | 0.482 | 0.399 |
| corrupt__impulse_noise__s1 | 0.284 | 0.234 |
| corrupt__impulse_noise__s3 | 0.526 | 0.421 |
| corrupt__impulse_noise__s5 | 0.625 | 0.464 |
| corrupt__jpeg_compression__s1 | 0.199 | 0.150 |
| corrupt__jpeg_compression__s3 | 0.285 | 0.241 |
| corrupt__jpeg_compression__s5 | 0.332 | 0.298 |
| corrupt__motion_blur__s1 | 0.171 | 0.125 |
| corrupt__motion_blur__s3 | 0.289 | 0.258 |
| corrupt__motion_blur__s5 | 0.326 | 0.312 |
| corrupt__pixelate__s1 | 0.170 | 0.091 |
| corrupt__pixelate__s3 | 0.331 | 0.227 |
| corrupt__pixelate__s5 | 0.582 | 0.471 |
| corrupt__shot_noise__s1 | 0.258 | 0.187 |
| corrupt__shot_noise__s3 | 0.422 | 0.364 |
| corrupt__shot_noise__s5 | 0.419 | 0.375 |
| corrupt__snow__s1 | 0.197 | 0.142 |
| corrupt__snow__s3 | 0.280 | 0.240 |
| corrupt__snow__s5 | 0.350 | 0.309 |
| corrupt__zoom_blur__s1 | 0.207 | 0.179 |
| corrupt__zoom_blur__s3 | 0.304 | 0.267 |
| corrupt__zoom_blur__s5 | 0.398 | 0.349 |
| ood__cifar100 | 0.514 | 0.415 |
| ood__svhn | 0.629 | 0.471 |
| panel | 0.047 | -0.020 |
| test | 0.108 | 0.027 |

## 6. Adjacency
layer `penult`


closest class pairs (sep = center distance / RMS radius): (3,5) 2.57, (2,3) 2.77, (2,4) 2.85, (0,2) 2.86, (4,7) 2.86, (3,4) 2.90

valley ratio mean 2.20, min 1.48, pairs with a valley 1.00

nearest-center confusions (true → nearest): 5→3 0.08, 3→5 0.06, 2→6 0.03, 9→1 0.03, 3→2 0.03, 2→3 0.03

single-linkage merge order (first 5): [3, 5]@2.57 ; [2, 3, 5]@2.77 ; [2, 3, 4, 5]@2.85 ; [0, 2, 3, 4, 5]@2.86 ; [0, 2, 3, 4, 5, 7]@2.86

## 7. Type-b margin (margin_typeb)
_margin_typeb not run_
