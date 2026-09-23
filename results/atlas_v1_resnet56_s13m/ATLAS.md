# ATLAS — atlas_v1_resnet56_s13m
built 2026-09-23 14:19:39 · source **real** · arch `cifar10_resnet56` · weights `file:/workspace/models/resnet56_s13m_chenyaofo.pt sha256:0df3f708ebfa3292` · layers 11

## 1. Per-layer invariants
| layer | D | PR | eff.rank | dim95 | TwoNN ID | sep ratio | NC1 | ETF dev | hub skew | test sparse | NC acc(test) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | 16 | 3.0 | 3.4 | 3 | 5.7 | 0.17 | 276.83 | 0.638 | 0.01 | 0.054 | 0.236 |
| layer1.0 | 16 | 3.1 | 3.7 | 4 | 6.5 | 0.17 | 209.57 | 0.624 | 0.12 | 0.053 | 0.246 |
| layer1.5 | 16 | 4.1 | 5.6 | 7 | 8.9 | 0.24 | 131.84 | 0.572 | 0.60 | 0.054 | 0.319 |
| layer1.8 | 16 | 5.2 | 7.2 | 9 | 9.8 | 0.31 | 59.76 | 0.469 | 0.72 | 0.056 | 0.373 |
| layer2.0 | 32 | 5.8 | 8.3 | 10 | 10.7 | 0.34 | 29.05 | 0.452 | 0.76 | 0.059 | 0.415 |
| layer2.5 | 32 | 7.0 | 10.7 | 14 | 12.7 | 0.43 | 9.85 | 0.384 | 1.26 | 0.048 | 0.508 |
| layer2.8 | 32 | 7.1 | 11.0 | 15 | 13.1 | 0.44 | 7.99 | 0.364 | 1.31 | 0.050 | 0.523 |
| layer3.0 | 64 | 8.9 | 14.1 | 20 | 14.7 | 0.53 | 3.92 | 0.351 | 1.26 | 0.046 | 0.600 |
| layer3.5 | 64 | 14.8 | 25.9 | 43 | 19.6 | 0.76 | 1.01 | 0.243 | 2.03 | 0.041 | 0.763 |
| layer3.8 | 64 | 9.0 | 10.5 | 9 | 10.7 | 3.31 | 0.13 | 0.079 | 0.91 | 0.131 | 0.924 |
| penult | 64 | 9.0 | 10.5 | 9 | 10.7 | 3.31 | 0.13 | 0.079 | 0.91 | 0.131 | 0.924 |

## 2. Decodability profile (which factor lives where)
categorical = accuracy minus majority-class rate; continuous = CV R²

| factor | kind | stem | layer1.0 | layer1.5 | layer1.8 | layer2.0 | layer2.5 | layer2.8 | layer3.0 | layer3.5 | layer3.8 | penult | commit | peak | washout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| luminance_mean | continuous | 1.00 | 1.00 | 0.97 | 0.96 | 0.96 | 0.93 | 0.92 | 0.88 | 0.77 | 0.44 | 0.44 | stem | stem | 0.56 |
| contrast_rms | continuous | 0.56 | 0.61 | 0.50 | 0.54 | 0.67 | 0.65 | 0.64 | 0.65 | 0.58 | 0.45 | 0.45 | layer1.0 | layer2.0 | 0.22 |
| highfreq_ratio | continuous | 0.23 | 0.40 | 0.55 | 0.54 | 0.62 | 0.61 | 0.60 | 0.62 | 0.58 | 0.49 | 0.49 | layer2.0 | layer2.0 | 0.13 |
| spectral_slope | continuous | 0.11 | 0.29 | 0.49 | 0.55 | 0.61 | 0.63 | 0.62 | 0.66 | 0.61 | 0.51 | 0.51 | layer2.0 | layer3.0 | 0.15 |
| spectral_anisotropy | continuous | 0.13 | 0.18 | 0.25 | 0.26 | 0.33 | 0.33 | 0.33 | 0.37 | 0.35 | 0.29 | 0.29 | layer3.0 | layer3.0 | 0.08 |
| noise_sigma | continuous | 0.62 | 0.82 | 0.86 | 0.94 | 0.94 | 0.91 | 0.91 | 0.92 | 0.88 | 0.76 | 0.76 | layer1.5 | layer2.0 | 0.18 |
| saturation_mean | continuous | 0.81 | 0.85 | 0.66 | 0.58 | 0.72 | 0.57 | 0.56 | 0.58 | 0.47 | 0.29 | 0.29 | stem | layer1.0 | 0.56 |
| hue_cos | continuous | 0.79 | 0.84 | 0.81 | 0.77 | 0.77 | 0.74 | 0.73 | 0.69 | 0.62 | 0.47 | 0.47 | stem | layer1.0 | 0.37 |
| hue_sin | continuous | 0.80 | 0.83 | 0.78 | 0.76 | 0.74 | 0.72 | 0.71 | 0.70 | 0.65 | 0.52 | 0.52 | stem | layer1.0 | 0.31 |
| colorfulness | continuous | 0.64 | 0.80 | 0.80 | 0.74 | 0.78 | 0.69 | 0.67 | 0.66 | 0.56 | 0.34 | 0.34 | layer1.0 | layer1.5 | 0.46 |
| edge_density | continuous | 0.69 | 0.80 | 0.74 | 0.85 | 0.88 | 0.85 | 0.85 | 0.85 | 0.79 | 0.65 | 0.65 | layer1.0 | layer2.0 | 0.23 |
| orientation_entropy | continuous | 0.13 | 0.17 | 0.33 | 0.39 | 0.46 | 0.45 | 0.46 | 0.53 | 0.49 | 0.42 | 0.42 | layer3.0 | layer3.0 | 0.11 |
| blockiness | continuous | 0.00 | 0.01 | 0.02 | 0.04 | 0.05 | 0.08 | 0.07 | 0.10 | 0.11 | 0.04 | 0.04 | layer3.0 | layer3.5 | 0.06 |
| class | categorical | 0.21 | 0.23 | 0.31 | 0.38 | 0.48 | 0.56 | 0.57 | 0.63 | 0.75 | 0.81 | 0.81 | layer3.5 | layer3.8 | 0.00 |
| coarse_animal_vehicle | categorical | 0.19 | 0.20 | 0.25 | 0.28 | 0.31 | 0.33 | 0.34 | 0.35 | 0.37 | 0.39 | 0.39 | layer3.5 | layer3.8 | 0.00 |
| corruption_family | categorical | 0.09 | 0.16 | 0.27 | 0.36 | 0.38 | 0.37 | 0.36 | 0.37 | 0.33 | 0.26 | 0.26 | layer1.8 | layer2.0 | 0.12 |
| corruption_type | categorical | 0.13 | 0.18 | 0.34 | 0.49 | 0.53 | 0.53 | 0.54 | 0.50 | 0.43 | 0.29 | 0.29 | layer1.8 | layer2.8 | 0.24 |
| severity | continuous | 0.04 | 0.07 | 0.12 | 0.17 | 0.25 | 0.24 | 0.22 | 0.32 | 0.30 | 0.24 | 0.24 | layer3.0 | layer3.0 | 0.08 |

## 3. Layer flow (CKA)
| pair | CKA |
|---|---|
| stem->layer1.0 | 0.977 |
| layer1.0->layer1.5 | 0.756 |
| layer1.5->layer1.8 | 0.770 |
| layer1.8->layer2.0 | 0.949 |
| layer2.0->layer2.5 | 0.941 |
| layer2.5->layer2.8 | 0.986 |
| layer2.8->layer3.0 | 0.931 |
| layer3.0->layer3.5 | 0.846 |
| layer3.5->layer3.8 | 0.632 |
| layer3.8->penult | 1.000 |

biggest reorganization: `layer3.5->layer3.8` (drop 0.368)

## 4. Sensitivity field (paired corruption displacement)
layer `penult`; magnitude in within-class-radius units

| corruption | sev | magnitude | per-sample | top5 frac | class-sub frac | coherence | class consistency | norm ratio |
|---|---|---|---|---|---|---|---|---|
| brightness | 1 | 0.03 | 0.14 | 0.60 | 0.79 | 0.24 | 0.63 | 1.001 |
| brightness | 3 | 0.11 | 0.41 | 0.52 | 0.80 | 0.26 | 0.53 | 0.996 |
| brightness | 5 | 0.23 | 0.79 | 0.46 | 0.81 | 0.28 | 0.39 | 0.979 |
| contrast | 1 | 0.13 | 0.34 | 0.75 | 0.90 | 0.38 | 0.71 | 1.005 |
| contrast | 3 | 0.51 | 1.12 | 0.71 | 0.87 | 0.45 | 0.54 | 0.990 |
| contrast | 5 | 1.62 | 2.78 | 0.63 | 0.88 | 0.58 | 0.41 | 0.966 |
| defocus_blur | 1 | 0.06 | 0.17 | 0.74 | 0.84 | 0.33 | 0.69 | 1.001 |
| defocus_blur | 3 | 0.43 | 0.91 | 0.85 | 0.87 | 0.45 | 0.62 | 0.982 |
| defocus_blur | 5 | 0.99 | 1.96 | 0.86 | 0.88 | 0.49 | 0.40 | 0.937 |
| elastic_transform | 1 | 0.20 | 0.87 | 0.79 | 0.79 | 0.23 | 0.26 | 0.973 |
| elastic_transform | 3 | 0.40 | 1.09 | 0.85 | 0.86 | 0.35 | 0.42 | 0.966 |
| elastic_transform | 5 | 0.54 | 1.52 | 0.80 | 0.93 | 0.34 | 0.26 | 0.951 |
| fog | 1 | 0.08 | 0.24 | 0.79 | 0.89 | 0.36 | 0.72 | 1.004 |
| fog | 3 | 0.30 | 0.71 | 0.77 | 0.89 | 0.41 | 0.62 | 0.997 |
| fog | 5 | 0.89 | 1.65 | 0.82 | 0.92 | 0.53 | 0.53 | 0.984 |
| frost | 1 | 0.28 | 0.77 | 0.75 | 0.89 | 0.33 | 0.47 | 0.983 |
| frost | 3 | 0.74 | 1.53 | 0.79 | 0.92 | 0.45 | 0.43 | 0.959 |
| frost | 5 | 1.11 | 1.98 | 0.85 | 0.92 | 0.52 | 0.48 | 0.962 |
| gaussian_noise | 1 | 0.78 | 1.39 | 0.88 | 0.92 | 0.54 | 0.60 | 0.956 |
| gaussian_noise | 3 | 1.64 | 2.52 | 0.87 | 0.92 | 0.63 | 0.55 | 0.946 |
| gaussian_noise | 5 | 1.87 | 2.81 | 0.85 | 0.91 | 0.64 | 0.52 | 0.954 |
| glass_blur | 1 | 0.93 | 2.03 | 0.79 | 0.93 | 0.42 | 0.27 | 0.914 |
| glass_blur | 3 | 0.85 | 1.92 | 0.82 | 0.93 | 0.40 | 0.27 | 0.923 |
| glass_blur | 5 | 0.98 | 2.19 | 0.79 | 0.93 | 0.41 | 0.24 | 0.911 |
| impulse_noise | 1 | 0.40 | 1.06 | 0.61 | 0.82 | 0.34 | 0.43 | 0.989 |
| impulse_noise | 3 | 0.78 | 1.81 | 0.78 | 0.84 | 0.41 | 0.30 | 0.938 |
| impulse_noise | 5 | 1.54 | 2.69 | 0.81 | 0.87 | 0.55 | 0.36 | 0.907 |
| jpeg_compression | 1 | 0.31 | 0.98 | 0.42 | 0.91 | 0.30 | 0.33 | 0.965 |
| jpeg_compression | 3 | 0.39 | 1.31 | 0.27 | 0.90 | 0.28 | 0.17 | 0.942 |
| jpeg_compression | 5 | 0.46 | 1.51 | 0.35 | 0.90 | 0.29 | 0.13 | 0.935 |
| motion_blur | 1 | 0.31 | 0.83 | 0.82 | 0.89 | 0.35 | 0.52 | 0.981 |
| motion_blur | 3 | 0.69 | 1.53 | 0.86 | 0.90 | 0.43 | 0.43 | 0.953 |
| motion_blur | 5 | 0.84 | 1.79 | 0.86 | 0.91 | 0.45 | 0.40 | 0.946 |
| pixelate | 1 | 0.22 | 0.60 | 0.63 | 0.93 | 0.33 | 0.60 | 0.985 |
| pixelate | 3 | 0.62 | 1.20 | 0.55 | 0.95 | 0.47 | 0.55 | 0.955 |
| pixelate | 5 | 1.37 | 2.33 | 0.35 | 0.95 | 0.56 | 0.45 | 0.955 |
| shot_noise | 1 | 0.57 | 1.06 | 0.88 | 0.91 | 0.51 | 0.64 | 0.969 |
| shot_noise | 3 | 1.40 | 2.18 | 0.88 | 0.92 | 0.61 | 0.58 | 0.943 |
| shot_noise | 5 | 1.74 | 2.66 | 0.86 | 0.91 | 0.63 | 0.52 | 0.936 |
| snow | 1 | 0.22 | 0.82 | 0.31 | 0.77 | 0.24 | 0.27 | 0.992 |
| snow | 3 | 0.34 | 1.24 | 0.37 | 0.78 | 0.27 | 0.18 | 0.969 |
| snow | 5 | 0.60 | 1.53 | 0.51 | 0.90 | 0.37 | 0.32 | 0.960 |
| zoom_blur | 1 | 0.45 | 1.01 | 0.87 | 0.88 | 0.42 | 0.55 | 0.974 |
| zoom_blur | 3 | 0.70 | 1.44 | 0.88 | 0.88 | 0.47 | 0.49 | 0.951 |
| zoom_blur | 5 | 1.00 | 1.94 | 0.87 | 0.88 | 0.50 | 0.42 | 0.930 |

## 5. Density per split
layer `penult`

| split | sparse frac (ref q95) | median log-radius shift |
|---|---|---|
| corrupt__brightness__s1 | 0.138 | 0.014 |
| corrupt__brightness__s3 | 0.157 | 0.040 |
| corrupt__brightness__s5 | 0.204 | 0.110 |
| corrupt__contrast__s1 | 0.146 | 0.027 |
| corrupt__contrast__s3 | 0.286 | 0.220 |
| corrupt__contrast__s5 | 0.927 | 0.741 |
| corrupt__defocus_blur__s1 | 0.139 | 0.018 |
| corrupt__defocus_blur__s3 | 0.241 | 0.164 |
| corrupt__defocus_blur__s5 | 0.581 | 0.484 |
| corrupt__elastic_transform__s1 | 0.214 | 0.139 |
| corrupt__elastic_transform__s3 | 0.273 | 0.219 |
| corrupt__elastic_transform__s5 | 0.444 | 0.379 |
| corrupt__fog__s1 | 0.145 | 0.015 |
| corrupt__fog__s3 | 0.206 | 0.105 |
| corrupt__fog__s5 | 0.497 | 0.428 |
| corrupt__frost__s1 | 0.218 | 0.130 |
| corrupt__frost__s3 | 0.434 | 0.371 |
| corrupt__frost__s5 | 0.567 | 0.480 |
| corrupt__gaussian_noise__s1 | 0.384 | 0.317 |
| corrupt__gaussian_noise__s3 | 0.630 | 0.524 |
| corrupt__gaussian_noise__s5 | 0.655 | 0.533 |
| corrupt__glass_blur__s1 | 0.598 | 0.494 |
| corrupt__glass_blur__s3 | 0.549 | 0.470 |
| corrupt__glass_blur__s5 | 0.631 | 0.522 |
| corrupt__impulse_noise__s1 | 0.287 | 0.217 |
| corrupt__impulse_noise__s3 | 0.580 | 0.492 |
| corrupt__impulse_noise__s5 | 0.858 | 0.650 |
| corrupt__jpeg_compression__s1 | 0.252 | 0.168 |
| corrupt__jpeg_compression__s3 | 0.340 | 0.298 |
| corrupt__jpeg_compression__s5 | 0.406 | 0.347 |
| corrupt__motion_blur__s1 | 0.221 | 0.150 |
| corrupt__motion_blur__s3 | 0.439 | 0.374 |
| corrupt__motion_blur__s5 | 0.511 | 0.439 |
| corrupt__pixelate__s1 | 0.195 | 0.077 |
| corrupt__pixelate__s3 | 0.340 | 0.259 |
| corrupt__pixelate__s5 | 0.629 | 0.561 |
| corrupt__shot_noise__s1 | 0.296 | 0.211 |
| corrupt__shot_noise__s3 | 0.583 | 0.493 |
| corrupt__shot_noise__s5 | 0.659 | 0.542 |
| corrupt__snow__s1 | 0.226 | 0.161 |
| corrupt__snow__s3 | 0.351 | 0.287 |
| corrupt__snow__s5 | 0.454 | 0.390 |
| corrupt__zoom_blur__s1 | 0.280 | 0.217 |
| corrupt__zoom_blur__s3 | 0.427 | 0.366 |
| corrupt__zoom_blur__s5 | 0.574 | 0.485 |
| ood__cifar100 | 0.690 | 0.549 |
| ood__svhn | 0.859 | 0.674 |
| panel | 0.078 | 0.003 |
| test | 0.131 | 0.008 |

## 6. Adjacency
layer `penult`


closest class pairs (sep = center distance / RMS radius): (3,5) 2.74, (2,3) 2.91, (3,4) 3.06, (2,4) 3.08, (3,6) 3.12, (5,7) 3.18

valley ratio mean 2.45, min 1.63, pairs with a valley 1.00

nearest-center confusions (true → nearest): 5→3 0.07, 3→5 0.06, 9→1 0.03, 8→0 0.03, 0→8 0.03, 6→3 0.03

single-linkage merge order (first 5): [3, 5]@2.74 ; [2, 3, 5]@2.91 ; [2, 3, 4, 5]@3.06 ; [2, 3, 4, 5, 6]@3.12 ; [2, 3, 4, 5, 6, 7]@3.18

## 7. Type-b margin (margin_typeb)
_margin_typeb not run_
