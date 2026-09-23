# ATLAS — atlas_v1_resnet56_s0hub_st3
built 2026-09-23 13:54:25 · source **real** · arch `cifar10_resnet56` · weights `chenyaofo cifar10_resnet56` · layers 11

## 1. Per-layer invariants
| layer | D | PR | eff.rank | dim95 | TwoNN ID | sep ratio | NC1 | ETF dev | hub skew | test sparse | NC acc(test) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | 16 | 2.9 | 3.6 | 4 | 6.5 | 0.16 | 494.39 | 0.649 | 0.15 | 0.062 | 0.248 |
| layer1.0 | 16 | 3.2 | 4.1 | 5 | 7.8 | 0.16 | 171.95 | 0.643 | 0.38 | 0.062 | 0.248 |
| layer1.5 | 16 | 5.1 | 7.0 | 8 | 9.5 | 0.23 | 53.61 | 0.509 | 0.76 | 0.055 | 0.328 |
| layer1.8 | 16 | 5.5 | 7.8 | 9 | 10.1 | 0.32 | 56.52 | 0.447 | 0.86 | 0.057 | 0.371 |
| layer2.0 | 32 | 6.2 | 9.1 | 11 | 11.2 | 0.34 | 24.11 | 0.456 | 0.91 | 0.058 | 0.404 |
| layer2.5 | 32 | 8.7 | 12.5 | 16 | 13.4 | 0.39 | 10.91 | 0.390 | 1.29 | 0.050 | 0.500 |
| layer2.8 | 32 | 9.9 | 14.4 | 18 | 14.2 | 0.44 | 6.22 | 0.352 | 1.48 | 0.048 | 0.558 |
| layer3.0 | 64 | 10.0 | 16.6 | 25 | 15.6 | 0.54 | 3.50 | 0.326 | 1.40 | 0.045 | 0.623 |
| layer3.5 | 64 | 15.8 | 27.3 | 44 | 19.4 | 0.89 | 0.67 | 0.210 | 1.88 | 0.050 | 0.837 |
| layer3.8 | 64 | 9.0 | 9.9 | 9 | 10.7 | 5.33 | 0.05 | 0.054 | 0.94 | 0.167 | 0.944 |
| penult | 64 | 9.0 | 9.9 | 9 | 10.7 | 5.33 | 0.05 | 0.054 | 0.94 | 0.167 | 0.944 |

## 2. Decodability profile (which factor lives where)
categorical = accuracy minus majority-class rate; continuous = CV R²

| factor | kind | stem | layer1.0 | layer1.5 | layer1.8 | layer2.0 | layer2.5 | layer2.8 | layer3.0 | layer3.5 | layer3.8 | penult | commit | peak | washout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| luminance_mean | continuous | 1.00 | 0.99 | 0.96 | 0.95 | 0.93 | 0.86 | 0.83 | 0.81 | 0.65 | 0.24 | 0.24 | stem | stem | 0.76 |
| contrast_rms | continuous | 0.66 | 0.64 | 0.58 | 0.55 | 0.66 | 0.65 | 0.63 | 0.65 | 0.57 | 0.39 | 0.39 | stem | layer2.0 | 0.27 |
| highfreq_ratio | continuous | 0.23 | 0.39 | 0.53 | 0.51 | 0.62 | 0.62 | 0.61 | 0.64 | 0.58 | 0.49 | 0.49 | layer2.0 | layer3.0 | 0.15 |
| spectral_slope | continuous | 0.13 | 0.27 | 0.49 | 0.52 | 0.68 | 0.67 | 0.65 | 0.66 | 0.62 | 0.48 | 0.48 | layer2.0 | layer2.0 | 0.20 |
| spectral_anisotropy | continuous | 0.23 | 0.26 | 0.29 | 0.29 | 0.32 | 0.30 | 0.32 | 0.36 | 0.34 | 0.29 | 0.29 | layer3.0 | layer3.0 | 0.08 |
| noise_sigma | continuous | 0.69 | 0.89 | 0.92 | 0.93 | 0.95 | 0.93 | 0.92 | 0.92 | 0.86 | 0.72 | 0.72 | layer1.0 | layer2.0 | 0.22 |
| saturation_mean | continuous | 0.77 | 0.78 | 0.77 | 0.67 | 0.79 | 0.71 | 0.66 | 0.67 | 0.57 | 0.29 | 0.29 | stem | layer2.0 | 0.50 |
| hue_cos | continuous | 0.79 | 0.81 | 0.80 | 0.78 | 0.80 | 0.78 | 0.77 | 0.72 | 0.64 | 0.48 | 0.48 | stem | layer1.0 | 0.33 |
| hue_sin | continuous | 0.81 | 0.84 | 0.80 | 0.78 | 0.75 | 0.74 | 0.74 | 0.73 | 0.68 | 0.48 | 0.48 | stem | layer1.0 | 0.36 |
| colorfulness | continuous | 0.71 | 0.81 | 0.87 | 0.82 | 0.85 | 0.81 | 0.79 | 0.76 | 0.65 | 0.37 | 0.37 | layer1.0 | layer1.5 | 0.50 |
| edge_density | continuous | 0.74 | 0.88 | 0.80 | 0.83 | 0.88 | 0.87 | 0.85 | 0.85 | 0.77 | 0.61 | 0.61 | layer1.0 | layer2.0 | 0.27 |
| orientation_entropy | continuous | 0.23 | 0.26 | 0.37 | 0.42 | 0.46 | 0.46 | 0.48 | 0.50 | 0.47 | 0.41 | 0.41 | layer2.0 | layer3.0 | 0.09 |
| blockiness | continuous | 0.01 | 0.01 | 0.03 | 0.04 | 0.07 | 0.06 | 0.07 | 0.10 | 0.08 | 0.06 | 0.06 | layer3.0 | layer3.0 | 0.04 |
| class | categorical | 0.25 | 0.25 | 0.32 | 0.37 | 0.48 | 0.55 | 0.58 | 0.67 | 0.78 | 0.84 | 0.84 | layer3.5 | layer3.8 | 0.00 |
| coarse_animal_vehicle | categorical | 0.24 | 0.24 | 0.27 | 0.28 | 0.31 | 0.32 | 0.33 | 0.36 | 0.38 | 0.40 | 0.40 | layer3.5 | layer3.8 | 0.00 |
| corruption_family | categorical | 0.11 | 0.17 | 0.29 | 0.33 | 0.40 | 0.37 | 0.36 | 0.39 | 0.34 | 0.23 | 0.23 | layer2.0 | layer2.0 | 0.17 |
| corruption_type | categorical | 0.14 | 0.18 | 0.38 | 0.50 | 0.56 | 0.56 | 0.56 | 0.53 | 0.45 | 0.28 | 0.28 | layer2.0 | layer2.8 | 0.28 |
| severity | continuous | 0.05 | 0.05 | 0.10 | 0.10 | 0.24 | 0.24 | 0.25 | 0.32 | 0.29 | 0.23 | 0.23 | layer3.0 | layer3.0 | 0.09 |

## 3. Layer flow (CKA)
| pair | CKA |
|---|---|
| stem->layer1.0 | 0.979 |
| layer1.0->layer1.5 | 0.774 |
| layer1.5->layer1.8 | 0.864 |
| layer1.8->layer2.0 | 0.940 |
| layer2.0->layer2.5 | 0.944 |
| layer2.5->layer2.8 | 0.981 |
| layer2.8->layer3.0 | 0.941 |
| layer3.0->layer3.5 | 0.803 |
| layer3.5->layer3.8 | 0.655 |
| layer3.8->penult | 1.000 |

biggest reorganization: `layer3.5->layer3.8` (drop 0.345)

## 4. Sensitivity field (paired corruption displacement)
layer `penult`; magnitude in within-class-radius units

| corruption | sev | magnitude | per-sample | top5 frac | class-sub frac | coherence | class consistency | norm ratio |
|---|---|---|---|---|---|---|---|---|
| brightness | 1 | 0.03 | 0.17 | 0.74 | 0.85 | 0.15 | 0.39 | 0.999 |
| brightness | 3 | 0.09 | 0.48 | 0.74 | 0.83 | 0.18 | 0.28 | 0.991 |
| brightness | 5 | 0.22 | 0.94 | 0.70 | 0.84 | 0.22 | 0.18 | 0.972 |
| contrast | 1 | 0.13 | 0.40 | 0.67 | 0.89 | 0.32 | 0.59 | 0.994 |
| contrast | 3 | 0.50 | 1.32 | 0.62 | 0.88 | 0.37 | 0.40 | 0.958 |
| contrast | 5 | 1.41 | 3.41 | 0.78 | 0.90 | 0.39 | 0.12 | 0.800 |
| defocus_blur | 1 | 0.06 | 0.20 | 0.56 | 0.81 | 0.28 | 0.59 | 0.997 |
| defocus_blur | 3 | 0.41 | 1.09 | 0.79 | 0.88 | 0.36 | 0.41 | 0.964 |
| defocus_blur | 5 | 1.04 | 2.51 | 0.86 | 0.93 | 0.39 | 0.20 | 0.882 |
| elastic_transform | 1 | 0.23 | 1.08 | 0.67 | 0.82 | 0.22 | 0.20 | 0.976 |
| elastic_transform | 3 | 0.44 | 1.38 | 0.77 | 0.90 | 0.31 | 0.27 | 0.959 |
| elastic_transform | 5 | 0.70 | 2.07 | 0.65 | 0.93 | 0.33 | 0.19 | 0.939 |
| fog | 1 | 0.08 | 0.28 | 0.70 | 0.92 | 0.29 | 0.62 | 0.999 |
| fog | 3 | 0.30 | 0.87 | 0.67 | 0.90 | 0.34 | 0.50 | 0.981 |
| fog | 5 | 1.01 | 2.13 | 0.65 | 0.94 | 0.45 | 0.38 | 0.930 |
| frost | 1 | 0.40 | 0.98 | 0.51 | 0.89 | 0.36 | 0.50 | 0.983 |
| frost | 3 | 0.96 | 1.99 | 0.62 | 0.89 | 0.45 | 0.40 | 0.953 |
| frost | 5 | 1.39 | 2.61 | 0.66 | 0.90 | 0.51 | 0.40 | 0.938 |
| gaussian_noise | 1 | 0.97 | 1.88 | 0.76 | 0.94 | 0.48 | 0.48 | 0.949 |
| gaussian_noise | 3 | 2.20 | 3.59 | 0.62 | 0.96 | 0.57 | 0.43 | 0.921 |
| gaussian_noise | 5 | 2.60 | 4.08 | 0.56 | 0.96 | 0.60 | 0.43 | 0.932 |
| glass_blur | 1 | 1.33 | 2.95 | 0.70 | 0.90 | 0.41 | 0.23 | 0.921 |
| glass_blur | 3 | 1.19 | 2.73 | 0.75 | 0.91 | 0.40 | 0.23 | 0.918 |
| glass_blur | 5 | 1.46 | 3.22 | 0.74 | 0.92 | 0.42 | 0.21 | 0.905 |
| impulse_noise | 1 | 0.59 | 1.52 | 0.41 | 0.81 | 0.35 | 0.44 | 0.995 |
| impulse_noise | 3 | 1.01 | 2.66 | 0.47 | 0.82 | 0.37 | 0.18 | 0.933 |
| impulse_noise | 5 | 2.33 | 4.12 | 0.37 | 0.93 | 0.54 | 0.31 | 0.905 |
| jpeg_compression | 1 | 0.41 | 1.30 | 0.45 | 0.90 | 0.30 | 0.29 | 0.965 |
| jpeg_compression | 3 | 0.56 | 1.77 | 0.19 | 0.89 | 0.30 | 0.17 | 0.944 |
| jpeg_compression | 5 | 0.66 | 2.08 | 0.13 | 0.90 | 0.31 | 0.13 | 0.933 |
| motion_blur | 1 | 0.32 | 0.97 | 0.78 | 0.91 | 0.31 | 0.40 | 0.971 |
| motion_blur | 3 | 0.75 | 1.95 | 0.83 | 0.94 | 0.35 | 0.24 | 0.920 |
| motion_blur | 5 | 0.94 | 2.35 | 0.85 | 0.95 | 0.37 | 0.20 | 0.900 |
| pixelate | 1 | 0.23 | 0.73 | 0.45 | 0.91 | 0.28 | 0.51 | 0.989 |
| pixelate | 3 | 0.77 | 1.56 | 0.49 | 0.94 | 0.43 | 0.51 | 0.962 |
| pixelate | 5 | 1.84 | 3.22 | 0.41 | 0.95 | 0.52 | 0.40 | 0.941 |
| shot_noise | 1 | 0.72 | 1.41 | 0.74 | 0.93 | 0.46 | 0.56 | 0.968 |
| shot_noise | 3 | 1.79 | 3.02 | 0.66 | 0.95 | 0.55 | 0.45 | 0.922 |
| shot_noise | 5 | 2.37 | 3.82 | 0.56 | 0.95 | 0.59 | 0.42 | 0.920 |
| snow | 1 | 0.36 | 1.09 | 0.54 | 0.82 | 0.30 | 0.42 | 0.995 |
| snow | 3 | 0.49 | 1.61 | 0.43 | 0.76 | 0.31 | 0.24 | 0.979 |
| snow | 5 | 0.75 | 1.97 | 0.49 | 0.87 | 0.37 | 0.29 | 0.957 |
| zoom_blur | 1 | 0.41 | 1.18 | 0.84 | 0.90 | 0.33 | 0.34 | 0.960 |
| zoom_blur | 3 | 0.69 | 1.73 | 0.84 | 0.93 | 0.38 | 0.31 | 0.934 |
| zoom_blur | 5 | 1.09 | 2.48 | 0.86 | 0.95 | 0.42 | 0.24 | 0.902 |

## 5. Density per split
layer `penult`

| split | sparse frac (ref q95) | median log-radius shift |
|---|---|---|
| corrupt__brightness__s1 | 0.167 | 0.042 |
| corrupt__brightness__s3 | 0.189 | 0.067 |
| corrupt__brightness__s5 | 0.252 | 0.165 |
| corrupt__contrast__s1 | 0.171 | 0.037 |
| corrupt__contrast__s3 | 0.296 | 0.215 |
| corrupt__contrast__s5 | 0.849 | 1.041 |
| corrupt__defocus_blur__s1 | 0.164 | 0.041 |
| corrupt__defocus_blur__s3 | 0.261 | 0.209 |
| corrupt__defocus_blur__s5 | 0.623 | 0.757 |
| corrupt__elastic_transform__s1 | 0.284 | 0.204 |
| corrupt__elastic_transform__s3 | 0.337 | 0.302 |
| corrupt__elastic_transform__s5 | 0.524 | 0.619 |
| corrupt__fog__s1 | 0.160 | 0.050 |
| corrupt__fog__s3 | 0.223 | 0.121 |
| corrupt__fog__s5 | 0.495 | 0.563 |
| corrupt__frost__s1 | 0.272 | 0.180 |
| corrupt__frost__s3 | 0.485 | 0.551 |
| corrupt__frost__s5 | 0.609 | 0.749 |
| corrupt__gaussian_noise__s1 | 0.449 | 0.483 |
| corrupt__gaussian_noise__s3 | 0.629 | 0.792 |
| corrupt__gaussian_noise__s5 | 0.666 | 0.813 |
| corrupt__glass_blur__s1 | 0.686 | 0.883 |
| corrupt__glass_blur__s3 | 0.651 | 0.838 |
| corrupt__glass_blur__s5 | 0.730 | 0.948 |
| corrupt__impulse_noise__s1 | 0.406 | 0.427 |
| corrupt__impulse_noise__s3 | 0.694 | 0.885 |
| corrupt__impulse_noise__s5 | 0.899 | 1.054 |
| corrupt__jpeg_compression__s1 | 0.330 | 0.279 |
| corrupt__jpeg_compression__s3 | 0.450 | 0.477 |
| corrupt__jpeg_compression__s5 | 0.528 | 0.616 |
| corrupt__motion_blur__s1 | 0.263 | 0.175 |
| corrupt__motion_blur__s3 | 0.475 | 0.533 |
| corrupt__motion_blur__s5 | 0.558 | 0.664 |
| corrupt__pixelate__s1 | 0.245 | 0.134 |
| corrupt__pixelate__s3 | 0.404 | 0.373 |
| corrupt__pixelate__s5 | 0.637 | 0.862 |
| corrupt__shot_noise__s1 | 0.353 | 0.314 |
| corrupt__shot_noise__s3 | 0.593 | 0.743 |
| corrupt__shot_noise__s5 | 0.698 | 0.865 |
| corrupt__snow__s1 | 0.305 | 0.263 |
| corrupt__snow__s3 | 0.436 | 0.471 |
| corrupt__snow__s5 | 0.497 | 0.571 |
| corrupt__zoom_blur__s1 | 0.304 | 0.263 |
| corrupt__zoom_blur__s3 | 0.432 | 0.470 |
| corrupt__zoom_blur__s5 | 0.603 | 0.742 |
| ood__cifar100 | 0.782 | 0.952 |
| ood__svhn | 0.899 | 1.036 |
| panel | 0.047 | -0.011 |
| test | 0.167 | 0.027 |

## 6. Adjacency
layer `penult`


closest class pairs (sep = center distance / RMS radius): (3,5) 4.53, (2,3) 4.73, (2,5) 4.91, (5,7) 4.94, (0,3) 4.96, (0,2) 4.99

valley ratio mean 4.44, min 3.18, pairs with a valley 1.00

nearest-center confusions (true → nearest): 5→3 0.06, 3→5 0.05, 0→8 0.02, 8→0 0.02, 9→1 0.02, 6→3 0.02

single-linkage merge order (first 5): [3, 5]@4.53 ; [2, 3, 5]@4.73 ; [2, 3, 5, 7]@4.94 ; [0, 2, 3, 5, 7]@4.96 ; [0, 2, 3, 5, 6, 7]@5.10

## 7. Type-b margin (margin_typeb)
_margin_typeb not run_
