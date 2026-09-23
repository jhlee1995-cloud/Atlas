# ATLAS — atlas_v1_resnet56_e70
built 2026-09-23 14:12:29 · source **real** · arch `cifar10_resnet56` · weights `file:/workspace/models/resnet56_s11_e70_chenyaofo.pt sha256:32e0876cd7a22a27` · layers 11

## 1. Per-layer invariants
| layer | D | PR | eff.rank | dim95 | TwoNN ID | sep ratio | NC1 | ETF dev | hub skew | test sparse | NC acc(test) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | 16 | 2.7 | 3.3 | 4 | 6.8 | 0.15 | 363.84 | 0.657 | 0.17 | 0.055 | 0.243 |
| layer1.0 | 16 | 2.6 | 3.6 | 5 | 7.8 | 0.16 | 313.15 | 0.664 | 0.43 | 0.057 | 0.249 |
| layer1.5 | 16 | 5.4 | 7.2 | 8 | 9.9 | 0.27 | 60.84 | 0.448 | 0.81 | 0.052 | 0.354 |
| layer1.8 | 16 | 5.4 | 7.5 | 9 | 9.9 | 0.33 | 47.79 | 0.413 | 0.81 | 0.056 | 0.369 |
| layer2.0 | 32 | 5.3 | 8.0 | 10 | 11.2 | 0.37 | 21.76 | 0.452 | 0.94 | 0.054 | 0.421 |
| layer2.5 | 32 | 6.4 | 10.3 | 15 | 13.4 | 0.43 | 9.11 | 0.386 | 1.27 | 0.054 | 0.506 |
| layer2.8 | 32 | 7.2 | 11.7 | 16 | 13.8 | 0.49 | 6.14 | 0.366 | 1.39 | 0.051 | 0.550 |
| layer3.0 | 64 | 8.6 | 15.3 | 25 | 16.0 | 0.57 | 2.98 | 0.338 | 1.40 | 0.048 | 0.635 |
| layer3.5 | 64 | 14.4 | 24.2 | 40 | 19.1 | 1.08 | 0.52 | 0.184 | 1.81 | 0.048 | 0.859 |
| layer3.8 | 64 | 9.1 | 10.4 | 9 | 10.9 | 3.51 | 0.12 | 0.070 | 0.90 | 0.136 | 0.925 |
| penult | 64 | 9.1 | 10.4 | 9 | 10.9 | 3.51 | 0.12 | 0.070 | 0.90 | 0.136 | 0.925 |

## 2. Decodability profile (which factor lives where)
categorical = accuracy minus majority-class rate; continuous = CV R²

| factor | kind | stem | layer1.0 | layer1.5 | layer1.8 | layer2.0 | layer2.5 | layer2.8 | layer3.0 | layer3.5 | layer3.8 | penult | commit | peak | washout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| luminance_mean | continuous | 1.00 | 0.99 | 0.96 | 0.95 | 0.87 | 0.83 | 0.80 | 0.73 | 0.56 | 0.34 | 0.34 | stem | stem | 0.65 |
| contrast_rms | continuous | 0.72 | 0.67 | 0.65 | 0.61 | 0.63 | 0.60 | 0.58 | 0.60 | 0.53 | 0.39 | 0.39 | stem | stem | 0.33 |
| highfreq_ratio | continuous | 0.33 | 0.47 | 0.56 | 0.55 | 0.60 | 0.60 | 0.60 | 0.61 | 0.56 | 0.49 | 0.49 | layer1.5 | layer3.0 | 0.13 |
| spectral_slope | continuous | 0.20 | 0.35 | 0.58 | 0.56 | 0.64 | 0.66 | 0.66 | 0.65 | 0.58 | 0.49 | 0.49 | layer2.0 | layer2.8 | 0.17 |
| spectral_anisotropy | continuous | 0.24 | 0.22 | 0.32 | 0.29 | 0.31 | 0.32 | 0.32 | 0.36 | 0.33 | 0.29 | 0.29 | layer3.0 | layer3.0 | 0.07 |
| noise_sigma | continuous | 0.80 | 0.90 | 0.92 | 0.92 | 0.95 | 0.92 | 0.91 | 0.91 | 0.85 | 0.72 | 0.72 | layer1.0 | layer2.0 | 0.23 |
| saturation_mean | continuous | 0.74 | 0.78 | 0.73 | 0.71 | 0.65 | 0.62 | 0.60 | 0.56 | 0.42 | 0.26 | 0.26 | stem | layer1.0 | 0.52 |
| hue_cos | continuous | 0.79 | 0.80 | 0.79 | 0.77 | 0.78 | 0.75 | 0.74 | 0.72 | 0.65 | 0.54 | 0.54 | stem | layer1.0 | 0.26 |
| hue_sin | continuous | 0.78 | 0.79 | 0.78 | 0.75 | 0.73 | 0.68 | 0.68 | 0.67 | 0.62 | 0.51 | 0.51 | stem | layer1.0 | 0.28 |
| colorfulness | continuous | 0.73 | 0.88 | 0.87 | 0.84 | 0.82 | 0.80 | 0.77 | 0.69 | 0.54 | 0.35 | 0.35 | layer1.0 | layer1.0 | 0.52 |
| edge_density | continuous | 0.90 | 0.90 | 0.85 | 0.85 | 0.89 | 0.85 | 0.84 | 0.83 | 0.77 | 0.65 | 0.65 | stem | layer1.0 | 0.25 |
| orientation_entropy | continuous | 0.25 | 0.26 | 0.41 | 0.42 | 0.47 | 0.47 | 0.48 | 0.52 | 0.48 | 0.43 | 0.43 | layer2.8 | layer3.0 | 0.09 |
| blockiness | continuous | 0.02 | 0.01 | 0.03 | 0.04 | 0.10 | 0.09 | 0.08 | 0.11 | 0.09 | 0.06 | 0.06 | layer3.0 | layer3.0 | 0.05 |
| class | categorical | 0.25 | 0.26 | 0.37 | 0.38 | 0.48 | 0.56 | 0.59 | 0.68 | 0.78 | 0.81 | 0.81 | layer3.5 | layer3.8 | 0.00 |
| coarse_animal_vehicle | categorical | 0.24 | 0.24 | 0.28 | 0.29 | 0.32 | 0.33 | 0.33 | 0.36 | 0.38 | 0.39 | 0.39 | layer3.0 | layer3.8 | 0.00 |
| corruption_family | categorical | 0.13 | 0.19 | 0.33 | 0.36 | 0.39 | 0.38 | 0.37 | 0.38 | 0.31 | 0.24 | 0.24 | layer1.8 | layer2.0 | 0.14 |
| corruption_type | categorical | 0.17 | 0.23 | 0.45 | 0.48 | 0.54 | 0.53 | 0.51 | 0.50 | 0.41 | 0.29 | 0.29 | layer2.0 | layer2.0 | 0.24 |
| severity | continuous | 0.08 | 0.10 | 0.11 | 0.17 | 0.26 | 0.26 | 0.24 | 0.31 | 0.30 | 0.27 | 0.27 | layer3.0 | layer3.0 | 0.04 |

## 3. Layer flow (CKA)
| pair | CKA |
|---|---|
| stem->layer1.0 | 0.956 |
| layer1.0->layer1.5 | 0.670 |
| layer1.5->layer1.8 | 0.934 |
| layer1.8->layer2.0 | 0.938 |
| layer2.0->layer2.5 | 0.954 |
| layer2.5->layer2.8 | 0.975 |
| layer2.8->layer3.0 | 0.945 |
| layer3.0->layer3.5 | 0.738 |
| layer3.5->layer3.8 | 0.764 |
| layer3.8->penult | 1.000 |

biggest reorganization: `layer1.0->layer1.5` (drop 0.330)

## 4. Sensitivity field (paired corruption displacement)
layer `penult`; magnitude in within-class-radius units

| corruption | sev | magnitude | per-sample | top5 frac | class-sub frac | coherence | class consistency | norm ratio |
|---|---|---|---|---|---|---|---|---|
| brightness | 1 | 0.03 | 0.14 | 0.72 | 0.84 | 0.23 | 0.66 | 1.002 |
| brightness | 3 | 0.10 | 0.39 | 0.71 | 0.83 | 0.25 | 0.56 | 0.999 |
| brightness | 5 | 0.20 | 0.76 | 0.59 | 0.82 | 0.27 | 0.36 | 0.981 |
| contrast | 1 | 0.12 | 0.32 | 0.69 | 0.88 | 0.37 | 0.70 | 1.004 |
| contrast | 3 | 0.46 | 1.08 | 0.61 | 0.87 | 0.43 | 0.56 | 0.977 |
| contrast | 5 | 1.32 | 2.62 | 0.67 | 0.83 | 0.49 | 0.28 | 0.833 |
| defocus_blur | 1 | 0.06 | 0.18 | 0.83 | 0.86 | 0.32 | 0.66 | 1.000 |
| defocus_blur | 3 | 0.48 | 0.99 | 0.86 | 0.91 | 0.46 | 0.61 | 0.980 |
| defocus_blur | 5 | 1.19 | 2.21 | 0.83 | 0.91 | 0.52 | 0.42 | 0.938 |
| elastic_transform | 1 | 0.24 | 0.91 | 0.76 | 0.86 | 0.26 | 0.30 | 0.968 |
| elastic_transform | 3 | 0.48 | 1.18 | 0.82 | 0.91 | 0.38 | 0.49 | 0.968 |
| elastic_transform | 5 | 0.56 | 1.58 | 0.75 | 0.93 | 0.34 | 0.25 | 0.945 |
| fog | 1 | 0.08 | 0.23 | 0.74 | 0.87 | 0.34 | 0.71 | 1.004 |
| fog | 3 | 0.28 | 0.70 | 0.73 | 0.87 | 0.41 | 0.66 | 0.994 |
| fog | 5 | 0.82 | 1.63 | 0.83 | 0.92 | 0.50 | 0.51 | 0.959 |
| frost | 1 | 0.30 | 0.76 | 0.67 | 0.88 | 0.38 | 0.59 | 0.990 |
| frost | 3 | 0.75 | 1.52 | 0.72 | 0.90 | 0.47 | 0.48 | 0.956 |
| frost | 5 | 1.03 | 1.94 | 0.78 | 0.91 | 0.51 | 0.45 | 0.939 |
| gaussian_noise | 1 | 0.77 | 1.41 | 0.89 | 0.92 | 0.53 | 0.60 | 0.952 |
| gaussian_noise | 3 | 1.59 | 2.56 | 0.86 | 0.92 | 0.59 | 0.49 | 0.906 |
| gaussian_noise | 5 | 1.79 | 2.85 | 0.84 | 0.92 | 0.60 | 0.46 | 0.893 |
| glass_blur | 1 | 1.07 | 2.20 | 0.75 | 0.94 | 0.44 | 0.29 | 0.905 |
| glass_blur | 3 | 0.96 | 2.04 | 0.78 | 0.93 | 0.43 | 0.30 | 0.911 |
| glass_blur | 5 | 1.13 | 2.35 | 0.75 | 0.93 | 0.44 | 0.27 | 0.901 |
| impulse_noise | 1 | 0.40 | 1.09 | 0.50 | 0.83 | 0.34 | 0.35 | 0.979 |
| impulse_noise | 3 | 0.81 | 1.91 | 0.73 | 0.84 | 0.40 | 0.24 | 0.902 |
| impulse_noise | 5 | 1.53 | 2.79 | 0.73 | 0.88 | 0.52 | 0.31 | 0.837 |
| jpeg_compression | 1 | 0.34 | 1.02 | 0.56 | 0.89 | 0.33 | 0.36 | 0.959 |
| jpeg_compression | 3 | 0.42 | 1.37 | 0.48 | 0.86 | 0.30 | 0.18 | 0.936 |
| jpeg_compression | 5 | 0.50 | 1.58 | 0.54 | 0.87 | 0.31 | 0.16 | 0.928 |
| motion_blur | 1 | 0.36 | 0.86 | 0.80 | 0.92 | 0.40 | 0.58 | 0.980 |
| motion_blur | 3 | 0.79 | 1.65 | 0.84 | 0.93 | 0.46 | 0.45 | 0.954 |
| motion_blur | 5 | 0.96 | 1.95 | 0.84 | 0.93 | 0.47 | 0.41 | 0.946 |
| pixelate | 1 | 0.25 | 0.64 | 0.83 | 0.92 | 0.35 | 0.63 | 0.986 |
| pixelate | 3 | 0.68 | 1.31 | 0.83 | 0.94 | 0.47 | 0.53 | 0.939 |
| pixelate | 5 | 1.42 | 2.46 | 0.68 | 0.91 | 0.54 | 0.39 | 0.881 |
| shot_noise | 1 | 0.57 | 1.07 | 0.83 | 0.92 | 0.51 | 0.67 | 0.972 |
| shot_noise | 3 | 1.34 | 2.21 | 0.88 | 0.92 | 0.58 | 0.52 | 0.917 |
| shot_noise | 5 | 1.67 | 2.70 | 0.85 | 0.91 | 0.59 | 0.45 | 0.888 |
| snow | 1 | 0.24 | 0.83 | 0.36 | 0.76 | 0.27 | 0.31 | 0.989 |
| snow | 3 | 0.37 | 1.24 | 0.40 | 0.76 | 0.30 | 0.22 | 0.960 |
| snow | 5 | 0.59 | 1.56 | 0.47 | 0.88 | 0.37 | 0.30 | 0.945 |
| zoom_blur | 1 | 0.46 | 1.06 | 0.85 | 0.91 | 0.42 | 0.51 | 0.965 |
| zoom_blur | 3 | 0.78 | 1.55 | 0.82 | 0.92 | 0.48 | 0.51 | 0.958 |
| zoom_blur | 5 | 1.14 | 2.11 | 0.77 | 0.92 | 0.52 | 0.46 | 0.951 |

## 5. Density per split
layer `penult`

| split | sparse frac (ref q95) | median log-radius shift |
|---|---|---|
| corrupt__brightness__s1 | 0.135 | 0.035 |
| corrupt__brightness__s3 | 0.154 | 0.057 |
| corrupt__brightness__s5 | 0.213 | 0.130 |
| corrupt__contrast__s1 | 0.137 | 0.042 |
| corrupt__contrast__s3 | 0.271 | 0.191 |
| corrupt__contrast__s5 | 0.823 | 0.652 |
| corrupt__defocus_blur__s1 | 0.138 | 0.033 |
| corrupt__defocus_blur__s3 | 0.249 | 0.179 |
| corrupt__defocus_blur__s5 | 0.599 | 0.516 |
| corrupt__elastic_transform__s1 | 0.240 | 0.159 |
| corrupt__elastic_transform__s3 | 0.299 | 0.236 |
| corrupt__elastic_transform__s5 | 0.469 | 0.405 |
| corrupt__fog__s1 | 0.139 | 0.032 |
| corrupt__fog__s3 | 0.193 | 0.104 |
| corrupt__fog__s5 | 0.449 | 0.390 |
| corrupt__frost__s1 | 0.214 | 0.146 |
| corrupt__frost__s3 | 0.436 | 0.371 |
| corrupt__frost__s5 | 0.560 | 0.501 |
| corrupt__gaussian_noise__s1 | 0.404 | 0.338 |
| corrupt__gaussian_noise__s3 | 0.638 | 0.558 |
| corrupt__gaussian_noise__s5 | 0.632 | 0.553 |
| corrupt__glass_blur__s1 | 0.649 | 0.573 |
| corrupt__glass_blur__s3 | 0.610 | 0.541 |
| corrupt__glass_blur__s5 | 0.696 | 0.600 |
| corrupt__impulse_noise__s1 | 0.326 | 0.256 |
| corrupt__impulse_noise__s3 | 0.642 | 0.555 |
| corrupt__impulse_noise__s5 | 0.818 | 0.662 |
| corrupt__jpeg_compression__s1 | 0.265 | 0.189 |
| corrupt__jpeg_compression__s3 | 0.370 | 0.310 |
| corrupt__jpeg_compression__s5 | 0.431 | 0.389 |
| corrupt__motion_blur__s1 | 0.222 | 0.162 |
| corrupt__motion_blur__s3 | 0.440 | 0.393 |
| corrupt__motion_blur__s5 | 0.536 | 0.472 |
| corrupt__pixelate__s1 | 0.213 | 0.104 |
| corrupt__pixelate__s3 | 0.372 | 0.277 |
| corrupt__pixelate__s5 | 0.612 | 0.558 |
| corrupt__shot_noise__s1 | 0.311 | 0.218 |
| corrupt__shot_noise__s3 | 0.602 | 0.538 |
| corrupt__shot_noise__s5 | 0.668 | 0.579 |
| corrupt__snow__s1 | 0.252 | 0.181 |
| corrupt__snow__s3 | 0.373 | 0.316 |
| corrupt__snow__s5 | 0.456 | 0.401 |
| corrupt__zoom_blur__s1 | 0.294 | 0.221 |
| corrupt__zoom_blur__s3 | 0.429 | 0.376 |
| corrupt__zoom_blur__s5 | 0.581 | 0.510 |
| ood__cifar100 | 0.684 | 0.579 |
| ood__svhn | 0.835 | 0.667 |
| panel | 0.031 | 0.009 |
| test | 0.136 | 0.023 |

## 6. Adjacency
layer `penult`


closest class pairs (sep = center distance / RMS radius): (3,5) 2.86, (2,3) 3.19, (2,4) 3.25, (3,6) 3.26, (0,2) 3.28, (3,4) 3.30

valley ratio mean 2.49, min 1.72, pairs with a valley 1.00

nearest-center confusions (true → nearest): 5→3 0.07, 3→5 0.07, 8→0 0.03, 9→1 0.03, 2→3 0.02, 0→8 0.02

single-linkage merge order (first 5): [3, 5]@2.86 ; [2, 3, 5]@3.19 ; [2, 3, 4, 5]@3.25 ; [2, 3, 4, 5, 6]@3.26 ; [0, 2, 3, 4, 5, 6]@3.28

## 7. Type-b margin (margin_typeb)
_margin_typeb not run_
