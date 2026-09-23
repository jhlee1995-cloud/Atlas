# ATLAS — atlas_v0_resnet20_cifar10
built 2026-09-23 00:50:30 · source **real** · arch `cifar10_resnet20` · weights `chenyaofo cifar10_resnet20` · layers 11

## 1. Per-layer invariants
| layer | D | PR | eff.rank | dim95 | TwoNN ID | sep ratio | NC1 | ETF dev | hub skew | test sparse | NC acc(test) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | 16 | 2.9 | 3.4 | 4 | 6.6 | 0.16 | 310.49 | 0.621 | 0.19 | 0.064 | 0.231 |
| layer1.0 | 16 | 3.9 | 5.2 | 6 | 8.7 | 0.17 | 172.52 | 0.560 | 0.57 | 0.060 | 0.288 |
| layer1.1 | 16 | 5.2 | 6.9 | 8 | 9.9 | 0.21 | 58.21 | 0.470 | 0.80 | 0.060 | 0.334 |
| layer1.2 | 16 | 5.1 | 7.5 | 9 | 10.1 | 0.29 | 56.94 | 0.462 | 0.87 | 0.060 | 0.355 |
| layer2.0 | 32 | 6.5 | 9.8 | 12 | 12.0 | 0.34 | 21.93 | 0.441 | 0.98 | 0.059 | 0.424 |
| layer2.1 | 32 | 7.5 | 11.3 | 15 | 13.1 | 0.39 | 12.14 | 0.426 | 1.20 | 0.053 | 0.495 |
| layer2.2 | 32 | 8.8 | 13.1 | 18 | 14.0 | 0.44 | 6.94 | 0.397 | 1.28 | 0.051 | 0.541 |
| layer3.0 | 64 | 11.8 | 21.0 | 37 | 18.0 | 0.57 | 2.48 | 0.307 | 1.82 | 0.048 | 0.642 |
| layer3.1 | 64 | 10.8 | 21.1 | 41 | 19.5 | 0.89 | 0.78 | 0.270 | 1.91 | 0.056 | 0.801 |
| layer3.2 | 64 | 8.4 | 9.6 | 9 | 9.9 | 2.94 | 0.18 | 0.109 | 0.74 | 0.109 | 0.917 |
| penult | 64 | 8.4 | 9.6 | 9 | 9.9 | 2.94 | 0.18 | 0.109 | 0.74 | 0.109 | 0.917 |

## 2. Decodability profile (which factor lives where)
categorical = accuracy minus majority-class rate; continuous = CV R²

| factor | kind | stem | layer1.0 | layer1.1 | layer1.2 | layer2.0 | layer2.1 | layer2.2 | layer3.0 | layer3.1 | layer3.2 | penult | commit | peak | washout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| luminance_mean | continuous | 0.99 | 0.97 | 0.97 | 0.96 | 0.92 | 0.90 | 0.86 | 0.73 | 0.61 | 0.40 | 0.40 | stem | stem | 0.59 |
| contrast_rms | continuous | 0.62 | 0.61 | 0.63 | 0.62 | 0.69 | 0.67 | 0.65 | 0.59 | 0.51 | 0.38 | 0.38 | stem | layer2.0 | 0.31 |
| highfreq_ratio | continuous | 0.18 | 0.58 | 0.59 | 0.54 | 0.62 | 0.62 | 0.61 | 0.60 | 0.55 | 0.49 | 0.49 | layer1.0 | layer2.0 | 0.13 |
| spectral_slope | continuous | 0.09 | 0.53 | 0.58 | 0.56 | 0.66 | 0.65 | 0.64 | 0.64 | 0.58 | 0.48 | 0.48 | layer2.0 | layer2.0 | 0.17 |
| spectral_anisotropy | continuous | 0.22 | 0.27 | 0.29 | 0.29 | 0.32 | 0.31 | 0.32 | 0.33 | 0.31 | 0.27 | 0.27 | layer2.0 | layer3.0 | 0.05 |
| noise_sigma | continuous | 0.54 | 0.97 | 0.93 | 0.96 | 0.96 | 0.95 | 0.93 | 0.90 | 0.84 | 0.72 | 0.72 | layer1.0 | layer1.0 | 0.25 |
| saturation_mean | continuous | 0.74 | 0.73 | 0.69 | 0.63 | 0.63 | 0.62 | 0.62 | 0.55 | 0.43 | 0.30 | 0.30 | stem | stem | 0.44 |
| hue_cos | continuous | 0.78 | 0.82 | 0.80 | 0.79 | 0.80 | 0.79 | 0.78 | 0.72 | 0.64 | 0.47 | 0.47 | stem | layer1.0 | 0.35 |
| hue_sin | continuous | 0.76 | 0.78 | 0.78 | 0.77 | 0.73 | 0.71 | 0.70 | 0.65 | 0.59 | 0.42 | 0.42 | stem | layer1.1 | 0.36 |
| colorfulness | continuous | 0.75 | 0.89 | 0.87 | 0.84 | 0.79 | 0.79 | 0.78 | 0.68 | 0.53 | 0.34 | 0.34 | layer1.0 | layer1.0 | 0.55 |
| edge_density | continuous | 0.62 | 0.88 | 0.87 | 0.90 | 0.91 | 0.89 | 0.87 | 0.83 | 0.76 | 0.61 | 0.61 | layer1.0 | layer2.0 | 0.30 |
| orientation_entropy | continuous | 0.23 | 0.34 | 0.35 | 0.41 | 0.46 | 0.45 | 0.48 | 0.50 | 0.46 | 0.42 | 0.42 | layer2.0 | layer3.0 | 0.08 |
| blockiness | continuous | 0.01 | 0.01 | 0.02 | 0.04 | 0.09 | 0.07 | 0.07 | 0.10 | 0.11 | 0.06 | 0.06 | layer3.0 | layer3.1 | 0.05 |
| class | categorical | 0.24 | 0.30 | 0.33 | 0.35 | 0.48 | 0.53 | 0.57 | 0.68 | 0.77 | 0.81 | 0.81 | layer3.1 | layer3.2 | 0.00 |
| coarse_animal_vehicle | categorical | 0.24 | 0.26 | 0.26 | 0.26 | 0.32 | 0.32 | 0.33 | 0.36 | 0.37 | 0.39 | 0.39 | layer3.0 | layer3.2 | 0.00 |
| corruption_family | categorical | 0.09 | 0.26 | 0.28 | 0.35 | 0.40 | 0.38 | 0.38 | 0.37 | 0.32 | 0.27 | 0.27 | layer2.0 | layer2.0 | 0.13 |
| corruption_type | categorical | 0.14 | 0.34 | 0.40 | 0.50 | 0.56 | 0.55 | 0.55 | 0.49 | 0.43 | 0.29 | 0.29 | layer2.0 | layer2.0 | 0.27 |
| severity | continuous | 0.04 | 0.15 | 0.16 | 0.17 | 0.28 | 0.27 | 0.27 | 0.32 | 0.30 | 0.23 | 0.23 | layer3.0 | layer3.0 | 0.09 |

## 3. Layer flow (CKA)
| pair | CKA |
|---|---|
| stem->layer1.0 | 0.917 |
| layer1.0->layer1.1 | 0.903 |
| layer1.1->layer1.2 | 0.752 |
| layer1.2->layer2.0 | 0.923 |
| layer2.0->layer2.1 | 0.959 |
| layer2.1->layer2.2 | 0.949 |
| layer2.2->layer3.0 | 0.871 |
| layer3.0->layer3.1 | 0.714 |
| layer3.1->layer3.2 | 0.691 |
| layer3.2->penult | 1.000 |

biggest reorganization: `layer3.1->layer3.2` (drop 0.309)

## 4. Sensitivity field (paired corruption displacement)
layer `penult`; magnitude in within-class-radius units

| corruption | sev | magnitude | per-sample | top5 frac | class-sub frac | coherence | class consistency | norm ratio |
|---|---|---|---|---|---|---|---|---|
| brightness | 1 | 0.02 | 0.13 | 0.75 | 0.90 | 0.15 | 0.49 | 0.999 |
| brightness | 3 | 0.07 | 0.37 | 0.68 | 0.93 | 0.18 | 0.37 | 0.991 |
| brightness | 5 | 0.18 | 0.74 | 0.59 | 0.94 | 0.22 | 0.21 | 0.962 |
| contrast | 1 | 0.10 | 0.36 | 0.42 | 0.86 | 0.28 | 0.36 | 0.979 |
| contrast | 3 | 0.46 | 1.29 | 0.51 | 0.87 | 0.35 | 0.17 | 0.868 |
| contrast | 5 | 1.16 | 2.54 | 0.52 | 0.84 | 0.44 | 0.15 | 0.701 |
| defocus_blur | 1 | 0.05 | 0.19 | 0.75 | 0.95 | 0.26 | 0.57 | 0.998 |
| defocus_blur | 3 | 0.40 | 1.02 | 0.81 | 0.94 | 0.37 | 0.34 | 0.936 |
| defocus_blur | 5 | 0.87 | 2.03 | 0.76 | 0.91 | 0.41 | 0.16 | 0.827 |
| elastic_transform | 1 | 0.20 | 0.92 | 0.82 | 0.94 | 0.20 | 0.10 | 0.941 |
| elastic_transform | 3 | 0.39 | 1.16 | 0.84 | 0.95 | 0.31 | 0.21 | 0.920 |
| elastic_transform | 5 | 0.49 | 1.51 | 0.82 | 0.98 | 0.30 | 0.11 | 0.909 |
| fog | 1 | 0.07 | 0.26 | 0.60 | 0.89 | 0.26 | 0.37 | 0.987 |
| fog | 3 | 0.27 | 0.81 | 0.62 | 0.91 | 0.32 | 0.25 | 0.933 |
| fog | 5 | 0.69 | 1.67 | 0.75 | 0.95 | 0.40 | 0.19 | 0.860 |
| frost | 1 | 0.25 | 0.74 | 0.52 | 0.94 | 0.30 | 0.30 | 0.958 |
| frost | 3 | 0.57 | 1.43 | 0.65 | 0.93 | 0.35 | 0.20 | 0.880 |
| frost | 5 | 0.78 | 1.78 | 0.72 | 0.93 | 0.40 | 0.20 | 0.841 |
| gaussian_noise | 1 | 0.65 | 1.36 | 0.85 | 0.95 | 0.45 | 0.38 | 0.896 |
| gaussian_noise | 3 | 1.29 | 2.31 | 0.86 | 0.91 | 0.53 | 0.33 | 0.789 |
| gaussian_noise | 5 | 1.48 | 2.56 | 0.83 | 0.88 | 0.55 | 0.33 | 0.765 |
| glass_blur | 1 | 0.77 | 1.98 | 0.79 | 0.95 | 0.35 | 0.11 | 0.834 |
| glass_blur | 3 | 0.70 | 1.87 | 0.82 | 0.96 | 0.33 | 0.10 | 0.841 |
| glass_blur | 5 | 0.79 | 2.08 | 0.77 | 0.95 | 0.34 | 0.09 | 0.823 |
| impulse_noise | 1 | 0.42 | 1.11 | 0.63 | 0.95 | 0.33 | 0.38 | 0.966 |
| impulse_noise | 3 | 0.57 | 1.73 | 0.79 | 0.95 | 0.30 | 0.10 | 0.888 |
| impulse_noise | 5 | 1.15 | 2.43 | 0.82 | 0.89 | 0.45 | 0.20 | 0.762 |
| jpeg_compression | 1 | 0.28 | 0.97 | 0.46 | 0.95 | 0.28 | 0.18 | 0.932 |
| jpeg_compression | 3 | 0.36 | 1.26 | 0.41 | 0.96 | 0.27 | 0.08 | 0.904 |
| jpeg_compression | 5 | 0.41 | 1.45 | 0.47 | 0.97 | 0.26 | 0.06 | 0.892 |
| motion_blur | 1 | 0.28 | 0.84 | 0.86 | 0.96 | 0.31 | 0.31 | 0.949 |
| motion_blur | 3 | 0.59 | 1.53 | 0.83 | 0.94 | 0.36 | 0.16 | 0.872 |
| motion_blur | 5 | 0.69 | 1.78 | 0.79 | 0.93 | 0.36 | 0.12 | 0.841 |
| pixelate | 1 | 0.25 | 0.65 | 0.75 | 0.98 | 0.35 | 0.57 | 0.966 |
| pixelate | 3 | 0.65 | 1.25 | 0.63 | 0.96 | 0.47 | 0.46 | 0.894 |
| pixelate | 5 | 1.28 | 2.21 | 0.48 | 0.94 | 0.54 | 0.37 | 0.849 |
| shot_noise | 1 | 0.51 | 1.06 | 0.81 | 0.95 | 0.44 | 0.49 | 0.936 |
| shot_noise | 3 | 1.11 | 2.02 | 0.87 | 0.93 | 0.52 | 0.35 | 0.824 |
| shot_noise | 5 | 1.40 | 2.44 | 0.84 | 0.89 | 0.54 | 0.33 | 0.778 |
| snow | 1 | 0.22 | 0.79 | 0.59 | 0.96 | 0.24 | 0.24 | 0.973 |
| snow | 3 | 0.29 | 1.17 | 0.35 | 0.96 | 0.24 | 0.09 | 0.931 |
| snow | 5 | 0.47 | 1.44 | 0.35 | 0.95 | 0.30 | 0.15 | 0.898 |
| zoom_blur | 1 | 0.36 | 1.05 | 0.91 | 0.95 | 0.31 | 0.20 | 0.919 |
| zoom_blur | 3 | 0.57 | 1.47 | 0.84 | 0.93 | 0.36 | 0.19 | 0.878 |
| zoom_blur | 5 | 0.81 | 1.90 | 0.78 | 0.92 | 0.40 | 0.17 | 0.841 |

## 5. Density per split
layer `penult`

| split | sparse frac (ref q95) | median log-radius shift |
|---|---|---|
| corrupt__brightness__s1 | 0.111 | 0.041 |
| corrupt__brightness__s3 | 0.117 | 0.058 |
| corrupt__brightness__s5 | 0.162 | 0.115 |
| corrupt__contrast__s1 | 0.114 | 0.051 |
| corrupt__contrast__s3 | 0.252 | 0.252 |
| corrupt__contrast__s5 | 0.757 | 0.517 |
| corrupt__defocus_blur__s1 | 0.106 | 0.052 |
| corrupt__defocus_blur__s3 | 0.214 | 0.206 |
| corrupt__defocus_blur__s5 | 0.475 | 0.411 |
| corrupt__elastic_transform__s1 | 0.181 | 0.157 |
| corrupt__elastic_transform__s3 | 0.234 | 0.224 |
| corrupt__elastic_transform__s5 | 0.381 | 0.344 |
| corrupt__fog__s1 | 0.111 | 0.053 |
| corrupt__fog__s3 | 0.165 | 0.134 |
| corrupt__fog__s5 | 0.337 | 0.329 |
| corrupt__frost__s1 | 0.169 | 0.126 |
| corrupt__frost__s3 | 0.302 | 0.294 |
| corrupt__frost__s5 | 0.345 | 0.341 |
| corrupt__gaussian_noise__s1 | 0.285 | 0.271 |
| corrupt__gaussian_noise__s3 | 0.292 | 0.283 |
| corrupt__gaussian_noise__s5 | 0.232 | 0.276 |
| corrupt__glass_blur__s1 | 0.448 | 0.392 |
| corrupt__glass_blur__s3 | 0.408 | 0.365 |
| corrupt__glass_blur__s5 | 0.450 | 0.391 |
| corrupt__impulse_noise__s1 | 0.273 | 0.234 |
| corrupt__impulse_noise__s3 | 0.507 | 0.423 |
| corrupt__impulse_noise__s5 | 0.515 | 0.426 |
| corrupt__jpeg_compression__s1 | 0.198 | 0.169 |
| corrupt__jpeg_compression__s3 | 0.270 | 0.250 |
| corrupt__jpeg_compression__s5 | 0.333 | 0.311 |
| corrupt__motion_blur__s1 | 0.179 | 0.160 |
| corrupt__motion_blur__s3 | 0.294 | 0.286 |
| corrupt__motion_blur__s5 | 0.350 | 0.338 |
| corrupt__pixelate__s1 | 0.171 | 0.102 |
| corrupt__pixelate__s3 | 0.324 | 0.245 |
| corrupt__pixelate__s5 | 0.562 | 0.466 |
| corrupt__shot_noise__s1 | 0.243 | 0.200 |
| corrupt__shot_noise__s3 | 0.345 | 0.323 |
| corrupt__shot_noise__s5 | 0.293 | 0.305 |
| corrupt__snow__s1 | 0.187 | 0.157 |
| corrupt__snow__s3 | 0.269 | 0.257 |
| corrupt__snow__s5 | 0.332 | 0.306 |
| corrupt__zoom_blur__s1 | 0.224 | 0.211 |
| corrupt__zoom_blur__s3 | 0.330 | 0.309 |
| corrupt__zoom_blur__s5 | 0.429 | 0.385 |
| ood__cifar100 | 0.474 | 0.404 |
| ood__svhn | 0.686 | 0.516 |
| panel | 0.062 | -0.008 |
| test | 0.109 | 0.035 |

## 6. Adjacency
layer `penult`


closest class pairs (sep = center distance / RMS radius): (3,5) 2.51, (2,3) 2.72, (0,2) 2.82, (4,7) 2.83, (2,4) 2.89, (3,4) 2.90

valley ratio mean 2.18, min 1.36, pairs with a valley 1.00

nearest-center confusions (true → nearest): 5→3 0.08, 3→5 0.06, 2→6 0.04, 3→2 0.03, 6→3 0.03, 9→1 0.03

single-linkage merge order (first 5): [3, 5]@2.51 ; [2, 3, 5]@2.72 ; [0, 2, 3, 5]@2.82 ; [4, 7]@2.83 ; [0, 2, 3, 4, 5, 7]@2.89
