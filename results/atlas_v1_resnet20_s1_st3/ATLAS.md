# ATLAS — atlas_v1_resnet20_s1_st3
built 2026-09-23 13:41:01 · source **real** · arch `cifar10_resnet20` · weights `file:/workspace/models/resnet20_s1_chenyaofo.pt sha256:d5442d0eadd72592` · layers 11

## 1. Per-layer invariants
| layer | D | PR | eff.rank | dim95 | TwoNN ID | sep ratio | NC1 | ETF dev | hub skew | test sparse | NC acc(test) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | 16 | 3.0 | 3.7 | 4 | 6.7 | 0.16 | 261.07 | 0.634 | 0.18 | 0.054 | 0.244 |
| layer1.0 | 16 | 3.3 | 4.6 | 5 | 8.6 | 0.18 | 164.63 | 0.635 | 0.54 | 0.053 | 0.280 |
| layer1.1 | 16 | 4.1 | 5.7 | 7 | 9.3 | 0.21 | 74.31 | 0.544 | 0.64 | 0.051 | 0.317 |
| layer1.2 | 16 | 4.8 | 6.8 | 8 | 9.5 | 0.30 | 49.25 | 0.428 | 0.71 | 0.055 | 0.353 |
| layer2.0 | 32 | 6.1 | 9.3 | 12 | 11.6 | 0.34 | 23.83 | 0.519 | 0.92 | 0.051 | 0.420 |
| layer2.1 | 32 | 7.7 | 11.4 | 14 | 12.6 | 0.39 | 17.04 | 0.401 | 1.20 | 0.050 | 0.476 |
| layer2.2 | 32 | 8.6 | 12.9 | 17 | 13.6 | 0.42 | 9.18 | 0.360 | 1.43 | 0.051 | 0.525 |
| layer3.0 | 64 | 10.3 | 20.0 | 38 | 18.0 | 0.57 | 2.39 | 0.329 | 1.69 | 0.045 | 0.649 |
| layer3.1 | 64 | 11.3 | 22.0 | 41 | 19.4 | 0.87 | 0.77 | 0.289 | 1.90 | 0.051 | 0.813 |
| layer3.2 | 64 | 8.6 | 9.7 | 9 | 9.8 | 3.07 | 0.16 | 0.098 | 0.83 | 0.106 | 0.922 |
| penult | 64 | 8.6 | 9.7 | 9 | 9.8 | 3.07 | 0.16 | 0.098 | 0.83 | 0.106 | 0.922 |

## 2. Decodability profile (which factor lives where)
categorical = accuracy minus majority-class rate; continuous = CV R²

| factor | kind | stem | layer1.0 | layer1.1 | layer1.2 | layer2.0 | layer2.1 | layer2.2 | layer3.0 | layer3.1 | layer3.2 | penult | commit | peak | washout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| luminance_mean | continuous | 1.00 | 0.99 | 0.97 | 0.97 | 0.95 | 0.93 | 0.91 | 0.76 | 0.53 | 0.26 | 0.26 | stem | stem | 0.74 |
| contrast_rms | continuous | 0.67 | 0.59 | 0.59 | 0.55 | 0.67 | 0.67 | 0.65 | 0.61 | 0.52 | 0.40 | 0.40 | stem | layer2.0 | 0.27 |
| highfreq_ratio | continuous | 0.29 | 0.54 | 0.57 | 0.53 | 0.62 | 0.62 | 0.62 | 0.59 | 0.56 | 0.44 | 0.44 | layer1.1 | layer2.1 | 0.17 |
| spectral_slope | continuous | 0.19 | 0.44 | 0.51 | 0.55 | 0.64 | 0.64 | 0.66 | 0.63 | 0.59 | 0.49 | 0.49 | layer2.0 | layer2.2 | 0.16 |
| spectral_anisotropy | continuous | 0.21 | 0.28 | 0.26 | 0.31 | 0.33 | 0.32 | 0.32 | 0.31 | 0.31 | 0.27 | 0.27 | layer1.2 | layer2.0 | 0.06 |
| noise_sigma | continuous | 0.67 | 0.92 | 0.93 | 0.91 | 0.96 | 0.94 | 0.92 | 0.87 | 0.80 | 0.67 | 0.67 | layer1.0 | layer2.0 | 0.29 |
| saturation_mean | continuous | 0.73 | 0.78 | 0.71 | 0.63 | 0.72 | 0.66 | 0.62 | 0.53 | 0.39 | 0.24 | 0.24 | stem | layer1.0 | 0.54 |
| hue_cos | continuous | 0.77 | 0.84 | 0.81 | 0.81 | 0.83 | 0.81 | 0.80 | 0.74 | 0.67 | 0.56 | 0.56 | stem | layer1.0 | 0.28 |
| hue_sin | continuous | 0.78 | 0.79 | 0.76 | 0.75 | 0.77 | 0.75 | 0.74 | 0.70 | 0.61 | 0.49 | 0.49 | stem | layer1.0 | 0.30 |
| colorfulness | continuous | 0.79 | 0.86 | 0.86 | 0.81 | 0.81 | 0.79 | 0.76 | 0.66 | 0.48 | 0.30 | 0.30 | stem | layer1.0 | 0.57 |
| edge_density | continuous | 0.70 | 0.81 | 0.86 | 0.84 | 0.89 | 0.87 | 0.85 | 0.81 | 0.72 | 0.57 | 0.57 | layer1.0 | layer2.0 | 0.32 |
| orientation_entropy | continuous | 0.21 | 0.31 | 0.32 | 0.40 | 0.49 | 0.45 | 0.46 | 0.46 | 0.43 | 0.39 | 0.39 | layer2.0 | layer2.0 | 0.10 |
| blockiness | continuous | 0.01 | 0.01 | 0.03 | 0.04 | 0.09 | 0.08 | 0.08 | 0.12 | 0.07 | 0.06 | 0.06 | layer3.0 | layer3.0 | 0.06 |
| class | categorical | 0.25 | 0.28 | 0.28 | 0.35 | 0.46 | 0.51 | 0.57 | 0.68 | 0.77 | 0.81 | 0.81 | layer3.1 | layer3.2 | 0.00 |
| coarse_animal_vehicle | categorical | 0.25 | 0.25 | 0.24 | 0.27 | 0.30 | 0.30 | 0.33 | 0.36 | 0.37 | 0.39 | 0.39 | layer3.0 | layer3.2 | 0.00 |
| corruption_family | categorical | 0.13 | 0.22 | 0.29 | 0.35 | 0.41 | 0.39 | 0.38 | 0.37 | 0.32 | 0.24 | 0.24 | layer2.0 | layer2.0 | 0.17 |
| corruption_type | categorical | 0.16 | 0.28 | 0.38 | 0.47 | 0.56 | 0.55 | 0.55 | 0.49 | 0.41 | 0.27 | 0.27 | layer2.0 | layer2.0 | 0.28 |
| severity | continuous | 0.07 | 0.08 | 0.12 | 0.14 | 0.28 | 0.27 | 0.27 | 0.32 | 0.28 | 0.23 | 0.23 | layer3.0 | layer3.0 | 0.09 |

## 3. Layer flow (CKA)
| pair | CKA |
|---|---|
| stem->layer1.0 | 0.899 |
| layer1.0->layer1.1 | 0.965 |
| layer1.1->layer1.2 | 0.773 |
| layer1.2->layer2.0 | 0.878 |
| layer2.0->layer2.1 | 0.955 |
| layer2.1->layer2.2 | 0.972 |
| layer2.2->layer3.0 | 0.889 |
| layer3.0->layer3.1 | 0.708 |
| layer3.1->layer3.2 | 0.691 |
| layer3.2->penult | 1.000 |

biggest reorganization: `layer3.1->layer3.2` (drop 0.309)

## 4. Sensitivity field (paired corruption displacement)
layer `penult`; magnitude in within-class-radius units

| corruption | sev | magnitude | per-sample | top5 frac | class-sub frac | coherence | class consistency | norm ratio |
|---|---|---|---|---|---|---|---|---|
| brightness | 1 | 0.03 | 0.15 | 0.77 | 0.95 | 0.15 | 0.58 | 1.000 |
| brightness | 3 | 0.08 | 0.43 | 0.69 | 0.93 | 0.17 | 0.39 | 0.985 |
| brightness | 5 | 0.20 | 0.81 | 0.60 | 0.93 | 0.22 | 0.19 | 0.947 |
| contrast | 1 | 0.12 | 0.34 | 0.69 | 0.96 | 0.32 | 0.68 | 0.999 |
| contrast | 3 | 0.44 | 1.09 | 0.66 | 0.94 | 0.39 | 0.44 | 0.955 |
| contrast | 5 | 1.25 | 2.46 | 0.61 | 0.88 | 0.50 | 0.27 | 0.839 |
| defocus_blur | 1 | 0.05 | 0.18 | 0.84 | 0.94 | 0.24 | 0.69 | 1.001 |
| defocus_blur | 3 | 0.39 | 0.94 | 0.95 | 0.96 | 0.38 | 0.49 | 0.969 |
| defocus_blur | 5 | 0.84 | 1.87 | 0.93 | 0.96 | 0.43 | 0.24 | 0.893 |
| elastic_transform | 1 | 0.20 | 0.89 | 0.94 | 0.96 | 0.20 | 0.19 | 0.959 |
| elastic_transform | 3 | 0.40 | 1.11 | 0.95 | 0.97 | 0.33 | 0.34 | 0.949 |
| elastic_transform | 5 | 0.46 | 1.43 | 0.73 | 0.98 | 0.30 | 0.14 | 0.912 |
| fog | 1 | 0.08 | 0.25 | 0.76 | 0.97 | 0.30 | 0.69 | 1.002 |
| fog | 3 | 0.28 | 0.73 | 0.76 | 0.97 | 0.36 | 0.58 | 0.983 |
| fog | 5 | 0.69 | 1.52 | 0.74 | 0.97 | 0.44 | 0.37 | 0.916 |
| frost | 1 | 0.27 | 0.76 | 0.67 | 0.96 | 0.31 | 0.36 | 0.956 |
| frost | 3 | 0.66 | 1.50 | 0.77 | 0.95 | 0.41 | 0.29 | 0.896 |
| frost | 5 | 0.95 | 1.89 | 0.85 | 0.95 | 0.47 | 0.32 | 0.879 |
| gaussian_noise | 1 | 0.79 | 1.45 | 0.92 | 0.97 | 0.52 | 0.51 | 0.909 |
| gaussian_noise | 3 | 1.51 | 2.50 | 0.91 | 0.92 | 0.58 | 0.37 | 0.811 |
| gaussian_noise | 5 | 1.62 | 2.69 | 0.86 | 0.87 | 0.58 | 0.34 | 0.762 |
| glass_blur | 1 | 0.84 | 1.97 | 0.88 | 0.96 | 0.38 | 0.17 | 0.855 |
| glass_blur | 3 | 0.75 | 1.84 | 0.87 | 0.97 | 0.36 | 0.16 | 0.864 |
| glass_blur | 5 | 0.86 | 2.09 | 0.88 | 0.96 | 0.37 | 0.14 | 0.842 |
| impulse_noise | 1 | 0.47 | 1.15 | 0.69 | 0.98 | 0.36 | 0.35 | 0.935 |
| impulse_noise | 3 | 0.58 | 1.78 | 0.66 | 0.96 | 0.30 | 0.07 | 0.850 |
| impulse_noise | 5 | 0.94 | 2.40 | 0.68 | 0.72 | 0.38 | 0.10 | 0.661 |
| jpeg_compression | 1 | 0.28 | 0.99 | 0.42 | 0.97 | 0.27 | 0.19 | 0.933 |
| jpeg_compression | 3 | 0.34 | 1.28 | 0.27 | 0.96 | 0.25 | 0.07 | 0.900 |
| jpeg_compression | 5 | 0.41 | 1.46 | 0.38 | 0.97 | 0.26 | 0.06 | 0.891 |
| motion_blur | 1 | 0.29 | 0.83 | 0.84 | 0.98 | 0.32 | 0.47 | 0.974 |
| motion_blur | 3 | 0.57 | 1.46 | 0.89 | 0.97 | 0.36 | 0.25 | 0.911 |
| motion_blur | 5 | 0.66 | 1.67 | 0.88 | 0.97 | 0.37 | 0.19 | 0.891 |
| pixelate | 1 | 0.25 | 0.63 | 0.67 | 0.99 | 0.36 | 0.56 | 0.964 |
| pixelate | 3 | 0.66 | 1.25 | 0.59 | 0.98 | 0.48 | 0.50 | 0.898 |
| pixelate | 5 | 1.27 | 2.23 | 0.47 | 0.96 | 0.53 | 0.36 | 0.830 |
| shot_noise | 1 | 0.58 | 1.10 | 0.90 | 0.97 | 0.49 | 0.58 | 0.938 |
| shot_noise | 3 | 1.27 | 2.17 | 0.92 | 0.94 | 0.56 | 0.39 | 0.837 |
| shot_noise | 5 | 1.50 | 2.55 | 0.87 | 0.88 | 0.56 | 0.33 | 0.767 |
| snow | 1 | 0.19 | 0.78 | 0.45 | 0.97 | 0.20 | 0.15 | 0.959 |
| snow | 3 | 0.25 | 1.14 | 0.22 | 0.96 | 0.22 | 0.05 | 0.926 |
| snow | 5 | 0.47 | 1.42 | 0.44 | 0.96 | 0.32 | 0.16 | 0.903 |
| zoom_blur | 1 | 0.39 | 1.00 | 0.96 | 0.97 | 0.36 | 0.37 | 0.947 |
| zoom_blur | 3 | 0.61 | 1.41 | 0.95 | 0.97 | 0.41 | 0.32 | 0.923 |
| zoom_blur | 5 | 0.85 | 1.84 | 0.92 | 0.97 | 0.43 | 0.26 | 0.902 |

## 5. Density per split
layer `penult`

| split | sparse frac (ref q95) | median log-radius shift |
|---|---|---|
| corrupt__brightness__s1 | 0.108 | 0.033 |
| corrupt__brightness__s3 | 0.118 | 0.046 |
| corrupt__brightness__s5 | 0.156 | 0.110 |
| corrupt__contrast__s1 | 0.110 | 0.037 |
| corrupt__contrast__s3 | 0.228 | 0.195 |
| corrupt__contrast__s5 | 0.725 | 0.573 |
| corrupt__defocus_blur__s1 | 0.109 | 0.038 |
| corrupt__defocus_blur__s3 | 0.206 | 0.187 |
| corrupt__defocus_blur__s5 | 0.446 | 0.414 |
| corrupt__elastic_transform__s1 | 0.172 | 0.141 |
| corrupt__elastic_transform__s3 | 0.225 | 0.216 |
| corrupt__elastic_transform__s5 | 0.330 | 0.331 |
| corrupt__fog__s1 | 0.111 | 0.037 |
| corrupt__fog__s3 | 0.152 | 0.106 |
| corrupt__fog__s5 | 0.339 | 0.330 |
| corrupt__frost__s1 | 0.160 | 0.127 |
| corrupt__frost__s3 | 0.329 | 0.331 |
| corrupt__frost__s5 | 0.418 | 0.395 |
| corrupt__gaussian_noise__s1 | 0.308 | 0.296 |
| corrupt__gaussian_noise__s3 | 0.381 | 0.383 |
| corrupt__gaussian_noise__s5 | 0.383 | 0.392 |
| corrupt__glass_blur__s1 | 0.449 | 0.418 |
| corrupt__glass_blur__s3 | 0.424 | 0.395 |
| corrupt__glass_blur__s5 | 0.465 | 0.426 |
| corrupt__impulse_noise__s1 | 0.264 | 0.241 |
| corrupt__impulse_noise__s3 | 0.451 | 0.413 |
| corrupt__impulse_noise__s5 | 0.714 | 0.524 |
| corrupt__jpeg_compression__s1 | 0.199 | 0.158 |
| corrupt__jpeg_compression__s3 | 0.263 | 0.262 |
| corrupt__jpeg_compression__s5 | 0.292 | 0.302 |
| corrupt__motion_blur__s1 | 0.166 | 0.154 |
| corrupt__motion_blur__s3 | 0.293 | 0.304 |
| corrupt__motion_blur__s5 | 0.338 | 0.343 |
| corrupt__pixelate__s1 | 0.155 | 0.084 |
| corrupt__pixelate__s3 | 0.314 | 0.240 |
| corrupt__pixelate__s5 | 0.471 | 0.418 |
| corrupt__shot_noise__s1 | 0.244 | 0.202 |
| corrupt__shot_noise__s3 | 0.430 | 0.406 |
| corrupt__shot_noise__s5 | 0.422 | 0.414 |
| corrupt__snow__s1 | 0.177 | 0.147 |
| corrupt__snow__s3 | 0.251 | 0.245 |
| corrupt__snow__s5 | 0.318 | 0.304 |
| corrupt__zoom_blur__s1 | 0.220 | 0.205 |
| corrupt__zoom_blur__s3 | 0.321 | 0.313 |
| corrupt__zoom_blur__s5 | 0.436 | 0.407 |
| ood__cifar100 | 0.489 | 0.439 |
| ood__svhn | 0.708 | 0.554 |
| panel | 0.016 | 0.016 |
| test | 0.106 | 0.033 |

## 6. Adjacency
layer `penult`


closest class pairs (sep = center distance / RMS radius): (3,5) 2.47, (2,3) 2.79, (2,4) 2.86, (5,7) 2.91, (0,2) 2.94, (2,5) 2.95

valley ratio mean 2.27, min 1.50, pairs with a valley 1.00

nearest-center confusions (true → nearest): 5→3 0.07, 3→5 0.07, 9→1 0.03, 0→8 0.03, 6→2 0.03, 8→0 0.03

single-linkage merge order (first 5): [3, 5]@2.47 ; [2, 3, 5]@2.79 ; [2, 3, 4, 5]@2.86 ; [2, 3, 4, 5, 7]@2.91 ; [0, 2, 3, 4, 5, 7]@2.94

## 7. Type-b margin (margin_typeb)
_margin_typeb not run_
