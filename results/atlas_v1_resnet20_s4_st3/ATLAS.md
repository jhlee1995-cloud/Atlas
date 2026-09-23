# ATLAS — atlas_v1_resnet20_s4_st3
built 2026-09-23 13:51:08 · source **real** · arch `cifar10_resnet20` · weights `file:/workspace/models/resnet20_s4_chenyaofo.pt sha256:a67852e591b4618e` · layers 11

## 1. Per-layer invariants
| layer | D | PR | eff.rank | dim95 | TwoNN ID | sep ratio | NC1 | ETF dev | hub skew | test sparse | NC acc(test) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | 16 | 3.1 | 3.9 | 4 | 6.8 | 0.16 | 159.18 | 0.648 | 0.20 | 0.054 | 0.250 |
| layer1.0 | 16 | 3.9 | 5.5 | 7 | 9.3 | 0.20 | 93.88 | 0.519 | 0.74 | 0.058 | 0.296 |
| layer1.1 | 16 | 4.9 | 6.7 | 8 | 9.9 | 0.21 | 86.85 | 0.460 | 0.80 | 0.056 | 0.323 |
| layer1.2 | 16 | 5.1 | 7.3 | 9 | 10.3 | 0.29 | 48.32 | 0.432 | 0.92 | 0.049 | 0.356 |
| layer2.0 | 32 | 6.1 | 9.4 | 12 | 12.1 | 0.34 | 20.88 | 0.459 | 1.01 | 0.050 | 0.417 |
| layer2.1 | 32 | 7.5 | 11.5 | 15 | 13.4 | 0.37 | 14.64 | 0.422 | 1.30 | 0.054 | 0.472 |
| layer2.2 | 32 | 8.6 | 13.1 | 18 | 14.1 | 0.42 | 8.83 | 0.384 | 1.42 | 0.049 | 0.531 |
| layer3.0 | 64 | 10.4 | 19.6 | 37 | 18.3 | 0.57 | 2.40 | 0.338 | 1.87 | 0.043 | 0.647 |
| layer3.1 | 64 | 9.5 | 20.2 | 41 | 19.5 | 0.83 | 0.79 | 0.296 | 1.86 | 0.052 | 0.822 |
| layer3.2 | 64 | 8.5 | 9.7 | 9 | 9.9 | 3.02 | 0.17 | 0.097 | 0.80 | 0.091 | 0.928 |
| penult | 64 | 8.5 | 9.7 | 9 | 9.9 | 3.02 | 0.17 | 0.097 | 0.80 | 0.091 | 0.928 |

## 2. Decodability profile (which factor lives where)
categorical = accuracy minus majority-class rate; continuous = CV R²

| factor | kind | stem | layer1.0 | layer1.1 | layer1.2 | layer2.0 | layer2.1 | layer2.2 | layer3.0 | layer3.1 | layer3.2 | penult | commit | peak | washout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| luminance_mean | continuous | 1.00 | 0.98 | 0.95 | 0.95 | 0.92 | 0.89 | 0.86 | 0.71 | 0.59 | 0.35 | 0.35 | stem | stem | 0.64 |
| contrast_rms | continuous | 0.66 | 0.62 | 0.64 | 0.63 | 0.67 | 0.66 | 0.63 | 0.64 | 0.57 | 0.37 | 0.37 | stem | layer2.0 | 0.30 |
| highfreq_ratio | continuous | 0.24 | 0.53 | 0.55 | 0.55 | 0.64 | 0.62 | 0.60 | 0.58 | 0.52 | 0.48 | 0.48 | layer2.0 | layer2.0 | 0.16 |
| spectral_slope | continuous | 0.14 | 0.44 | 0.50 | 0.55 | 0.67 | 0.66 | 0.66 | 0.64 | 0.58 | 0.48 | 0.48 | layer2.0 | layer2.0 | 0.19 |
| spectral_anisotropy | continuous | 0.24 | 0.27 | 0.27 | 0.27 | 0.32 | 0.32 | 0.33 | 0.34 | 0.32 | 0.27 | 0.27 | layer2.0 | layer3.0 | 0.07 |
| noise_sigma | continuous | 0.65 | 0.92 | 0.92 | 0.93 | 0.95 | 0.93 | 0.93 | 0.90 | 0.83 | 0.66 | 0.66 | layer1.0 | layer2.0 | 0.29 |
| saturation_mean | continuous | 0.84 | 0.69 | 0.66 | 0.55 | 0.73 | 0.67 | 0.63 | 0.56 | 0.47 | 0.31 | 0.31 | stem | stem | 0.53 |
| hue_cos | continuous | 0.81 | 0.79 | 0.79 | 0.80 | 0.80 | 0.79 | 0.77 | 0.69 | 0.59 | 0.44 | 0.44 | stem | stem | 0.37 |
| hue_sin | continuous | 0.82 | 0.79 | 0.76 | 0.74 | 0.72 | 0.72 | 0.71 | 0.68 | 0.60 | 0.43 | 0.43 | stem | stem | 0.39 |
| colorfulness | continuous | 0.75 | 0.80 | 0.80 | 0.74 | 0.80 | 0.77 | 0.76 | 0.62 | 0.50 | 0.30 | 0.30 | stem | layer2.0 | 0.51 |
| edge_density | continuous | 0.73 | 0.82 | 0.84 | 0.82 | 0.89 | 0.87 | 0.85 | 0.83 | 0.76 | 0.55 | 0.55 | layer1.0 | layer2.0 | 0.34 |
| orientation_entropy | continuous | 0.26 | 0.31 | 0.35 | 0.38 | 0.43 | 0.43 | 0.44 | 0.47 | 0.44 | 0.36 | 0.36 | layer2.0 | layer3.0 | 0.11 |
| blockiness | continuous | 0.01 | 0.01 | 0.02 | 0.05 | 0.06 | 0.05 | 0.05 | 0.09 | 0.06 | 0.05 | 0.05 | layer3.0 | layer3.0 | 0.05 |
| class | categorical | 0.25 | 0.29 | 0.30 | 0.35 | 0.47 | 0.52 | 0.58 | 0.69 | 0.78 | 0.82 | 0.82 | layer3.1 | layer3.2 | 0.00 |
| coarse_animal_vehicle | categorical | 0.25 | 0.25 | 0.26 | 0.28 | 0.31 | 0.32 | 0.33 | 0.36 | 0.38 | 0.39 | 0.39 | layer3.0 | layer3.2 | 0.00 |
| corruption_family | categorical | 0.12 | 0.24 | 0.28 | 0.32 | 0.39 | 0.39 | 0.37 | 0.36 | 0.31 | 0.22 | 0.22 | layer2.0 | layer2.0 | 0.17 |
| corruption_type | categorical | 0.16 | 0.31 | 0.39 | 0.48 | 0.56 | 0.55 | 0.54 | 0.48 | 0.41 | 0.27 | 0.27 | layer2.0 | layer2.0 | 0.29 |
| severity | continuous | 0.06 | 0.07 | 0.13 | 0.16 | 0.31 | 0.31 | 0.29 | 0.29 | 0.28 | 0.23 | 0.23 | layer2.0 | layer2.1 | 0.08 |

## 3. Layer flow (CKA)
| pair | CKA |
|---|---|
| stem->layer1.0 | 0.863 |
| layer1.0->layer1.1 | 0.934 |
| layer1.1->layer1.2 | 0.818 |
| layer1.2->layer2.0 | 0.916 |
| layer2.0->layer2.1 | 0.967 |
| layer2.1->layer2.2 | 0.965 |
| layer2.2->layer3.0 | 0.877 |
| layer3.0->layer3.1 | 0.709 |
| layer3.1->layer3.2 | 0.669 |
| layer3.2->penult | 1.000 |

biggest reorganization: `layer3.1->layer3.2` (drop 0.331)

## 4. Sensitivity field (paired corruption displacement)
layer `penult`; magnitude in within-class-radius units

| corruption | sev | magnitude | per-sample | top5 frac | class-sub frac | coherence | class consistency | norm ratio |
|---|---|---|---|---|---|---|---|---|
| brightness | 1 | 0.03 | 0.15 | 0.89 | 0.95 | 0.20 | 0.65 | 0.999 |
| brightness | 3 | 0.10 | 0.42 | 0.83 | 0.94 | 0.22 | 0.45 | 0.986 |
| brightness | 5 | 0.21 | 0.77 | 0.66 | 0.91 | 0.25 | 0.28 | 0.948 |
| contrast | 1 | 0.11 | 0.35 | 0.69 | 0.96 | 0.31 | 0.67 | 0.997 |
| contrast | 3 | 0.40 | 1.11 | 0.68 | 0.93 | 0.35 | 0.33 | 0.938 |
| contrast | 5 | 1.15 | 2.44 | 0.66 | 0.81 | 0.46 | 0.21 | 0.754 |
| defocus_blur | 1 | 0.05 | 0.18 | 0.76 | 0.94 | 0.27 | 0.68 | 1.001 |
| defocus_blur | 3 | 0.40 | 0.93 | 0.85 | 0.95 | 0.40 | 0.54 | 0.965 |
| defocus_blur | 5 | 0.88 | 1.87 | 0.87 | 0.95 | 0.44 | 0.29 | 0.881 |
| elastic_transform | 1 | 0.20 | 0.87 | 0.84 | 0.96 | 0.21 | 0.26 | 0.963 |
| elastic_transform | 3 | 0.40 | 1.09 | 0.85 | 0.97 | 0.33 | 0.38 | 0.951 |
| elastic_transform | 5 | 0.50 | 1.46 | 0.75 | 0.98 | 0.32 | 0.19 | 0.928 |
| fog | 1 | 0.08 | 0.25 | 0.75 | 0.96 | 0.30 | 0.70 | 1.003 |
| fog | 3 | 0.28 | 0.75 | 0.76 | 0.97 | 0.36 | 0.55 | 0.986 |
| fog | 5 | 0.84 | 1.65 | 0.73 | 0.98 | 0.49 | 0.44 | 0.949 |
| frost | 1 | 0.29 | 0.77 | 0.69 | 0.95 | 0.34 | 0.44 | 0.975 |
| frost | 3 | 0.70 | 1.49 | 0.72 | 0.95 | 0.44 | 0.37 | 0.924 |
| frost | 5 | 1.00 | 1.89 | 0.74 | 0.95 | 0.50 | 0.39 | 0.915 |
| gaussian_noise | 1 | 0.72 | 1.39 | 0.85 | 0.97 | 0.49 | 0.48 | 0.926 |
| gaussian_noise | 3 | 1.42 | 2.39 | 0.84 | 0.91 | 0.57 | 0.37 | 0.852 |
| gaussian_noise | 5 | 1.54 | 2.61 | 0.80 | 0.88 | 0.57 | 0.35 | 0.823 |
| glass_blur | 1 | 0.89 | 2.04 | 0.85 | 0.97 | 0.40 | 0.18 | 0.860 |
| glass_blur | 3 | 0.78 | 1.89 | 0.86 | 0.97 | 0.37 | 0.16 | 0.861 |
| glass_blur | 5 | 0.89 | 2.13 | 0.85 | 0.97 | 0.37 | 0.15 | 0.845 |
| impulse_noise | 1 | 0.46 | 1.12 | 0.51 | 0.96 | 0.36 | 0.36 | 0.946 |
| impulse_noise | 3 | 0.66 | 1.77 | 0.73 | 0.96 | 0.34 | 0.13 | 0.871 |
| impulse_noise | 5 | 1.22 | 2.47 | 0.79 | 0.87 | 0.48 | 0.22 | 0.789 |
| jpeg_compression | 1 | 0.28 | 0.98 | 0.49 | 0.95 | 0.27 | 0.18 | 0.934 |
| jpeg_compression | 3 | 0.32 | 1.25 | 0.42 | 0.94 | 0.24 | 0.06 | 0.898 |
| jpeg_compression | 5 | 0.37 | 1.42 | 0.47 | 0.94 | 0.24 | 0.05 | 0.881 |
| motion_blur | 1 | 0.31 | 0.83 | 0.80 | 0.98 | 0.34 | 0.52 | 0.976 |
| motion_blur | 3 | 0.62 | 1.48 | 0.84 | 0.97 | 0.39 | 0.32 | 0.914 |
| motion_blur | 5 | 0.73 | 1.71 | 0.84 | 0.97 | 0.39 | 0.26 | 0.892 |
| pixelate | 1 | 0.23 | 0.62 | 0.76 | 0.98 | 0.33 | 0.50 | 0.961 |
| pixelate | 3 | 0.58 | 1.17 | 0.68 | 0.97 | 0.44 | 0.44 | 0.897 |
| pixelate | 5 | 1.18 | 2.10 | 0.62 | 0.95 | 0.51 | 0.34 | 0.821 |
| shot_noise | 1 | 0.52 | 1.06 | 0.85 | 0.97 | 0.45 | 0.56 | 0.957 |
| shot_noise | 3 | 1.23 | 2.09 | 0.87 | 0.94 | 0.56 | 0.42 | 0.880 |
| shot_noise | 5 | 1.43 | 2.47 | 0.81 | 0.88 | 0.56 | 0.34 | 0.822 |
| snow | 1 | 0.21 | 0.82 | 0.39 | 0.97 | 0.21 | 0.15 | 0.955 |
| snow | 3 | 0.23 | 1.17 | 0.30 | 0.93 | 0.19 | 0.01 | 0.924 |
| snow | 5 | 0.45 | 1.44 | 0.50 | 0.96 | 0.30 | 0.15 | 0.908 |
| zoom_blur | 1 | 0.37 | 0.98 | 0.93 | 0.97 | 0.34 | 0.42 | 0.955 |
| zoom_blur | 3 | 0.60 | 1.36 | 0.89 | 0.96 | 0.40 | 0.38 | 0.921 |
| zoom_blur | 5 | 0.87 | 1.79 | 0.85 | 0.96 | 0.45 | 0.34 | 0.893 |

## 5. Density per split
layer `penult`

| split | sparse frac (ref q95) | median log-radius shift |
|---|---|---|
| corrupt__brightness__s1 | 0.096 | 0.029 |
| corrupt__brightness__s3 | 0.106 | 0.043 |
| corrupt__brightness__s5 | 0.135 | 0.084 |
| corrupt__contrast__s1 | 0.098 | 0.035 |
| corrupt__contrast__s3 | 0.199 | 0.193 |
| corrupt__contrast__s5 | 0.736 | 0.543 |
| corrupt__defocus_blur__s1 | 0.098 | 0.032 |
| corrupt__defocus_blur__s3 | 0.181 | 0.161 |
| corrupt__defocus_blur__s5 | 0.391 | 0.381 |
| corrupt__elastic_transform__s1 | 0.159 | 0.133 |
| corrupt__elastic_transform__s3 | 0.209 | 0.203 |
| corrupt__elastic_transform__s5 | 0.352 | 0.347 |
| corrupt__fog__s1 | 0.098 | 0.036 |
| corrupt__fog__s3 | 0.148 | 0.124 |
| corrupt__fog__s5 | 0.377 | 0.373 |
| corrupt__frost__s1 | 0.165 | 0.130 |
| corrupt__frost__s3 | 0.345 | 0.326 |
| corrupt__frost__s5 | 0.458 | 0.410 |
| corrupt__gaussian_noise__s1 | 0.338 | 0.304 |
| corrupt__gaussian_noise__s3 | 0.556 | 0.469 |
| corrupt__gaussian_noise__s5 | 0.633 | 0.495 |
| corrupt__glass_blur__s1 | 0.416 | 0.400 |
| corrupt__glass_blur__s3 | 0.379 | 0.379 |
| corrupt__glass_blur__s5 | 0.424 | 0.404 |
| corrupt__impulse_noise__s1 | 0.251 | 0.217 |
| corrupt__impulse_noise__s3 | 0.461 | 0.420 |
| corrupt__impulse_noise__s5 | 0.698 | 0.514 |
| corrupt__jpeg_compression__s1 | 0.168 | 0.163 |
| corrupt__jpeg_compression__s3 | 0.221 | 0.229 |
| corrupt__jpeg_compression__s5 | 0.273 | 0.278 |
| corrupt__motion_blur__s1 | 0.163 | 0.139 |
| corrupt__motion_blur__s3 | 0.285 | 0.306 |
| corrupt__motion_blur__s5 | 0.336 | 0.341 |
| corrupt__pixelate__s1 | 0.140 | 0.077 |
| corrupt__pixelate__s3 | 0.262 | 0.204 |
| corrupt__pixelate__s5 | 0.371 | 0.344 |
| corrupt__shot_noise__s1 | 0.235 | 0.218 |
| corrupt__shot_noise__s3 | 0.505 | 0.444 |
| corrupt__shot_noise__s5 | 0.614 | 0.487 |
| corrupt__snow__s1 | 0.159 | 0.141 |
| corrupt__snow__s3 | 0.240 | 0.240 |
| corrupt__snow__s5 | 0.304 | 0.310 |
| corrupt__zoom_blur__s1 | 0.210 | 0.200 |
| corrupt__zoom_blur__s3 | 0.287 | 0.286 |
| corrupt__zoom_blur__s5 | 0.386 | 0.375 |
| ood__cifar100 | 0.452 | 0.416 |
| ood__svhn | 0.703 | 0.531 |
| panel | 0.031 | 0.008 |
| test | 0.091 | 0.026 |

## 6. Adjacency
layer `penult`


closest class pairs (sep = center distance / RMS radius): (3,5) 2.40, (2,3) 2.80, (5,7) 2.87, (3,4) 2.89, (2,4) 2.89, (2,5) 2.90

valley ratio mean 2.31, min 1.61, pairs with a valley 1.00

nearest-center confusions (true → nearest): 5→3 0.06, 3→5 0.06, 9→1 0.04, 8→0 0.03, 2→3 0.03, 2→4 0.02

single-linkage merge order (first 5): [3, 5]@2.40 ; [2, 3, 5]@2.80 ; [2, 3, 5, 7]@2.87 ; [2, 3, 4, 5, 7]@2.89 ; [0, 2, 3, 4, 5, 7]@3.00

## 7. Type-b margin (margin_typeb)
_margin_typeb not run_
