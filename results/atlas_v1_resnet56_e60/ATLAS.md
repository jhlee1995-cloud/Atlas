# ATLAS — atlas_v1_resnet56_e60
built 2026-09-23 14:08:30 · source **real** · arch `cifar10_resnet56` · weights `file:/workspace/models/resnet56_s11_e60_chenyaofo.pt sha256:d8f65a2fe8d7a833` · layers 11

## 1. Per-layer invariants
| layer | D | PR | eff.rank | dim95 | TwoNN ID | sep ratio | NC1 | ETF dev | hub skew | test sparse | NC acc(test) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | 16 | 3.0 | 3.7 | 4 | 7.0 | 0.16 | 145.30 | 0.619 | 0.21 | 0.058 | 0.244 |
| layer1.0 | 16 | 3.1 | 4.3 | 5 | 8.2 | 0.17 | 142.66 | 0.631 | 0.44 | 0.060 | 0.265 |
| layer1.5 | 16 | 4.8 | 6.7 | 8 | 9.7 | 0.25 | 77.57 | 0.485 | 0.72 | 0.052 | 0.340 |
| layer1.8 | 16 | 5.3 | 7.3 | 9 | 9.8 | 0.33 | 44.92 | 0.450 | 0.81 | 0.052 | 0.385 |
| layer2.0 | 32 | 5.5 | 8.0 | 10 | 10.8 | 0.37 | 25.82 | 0.451 | 0.80 | 0.054 | 0.423 |
| layer2.5 | 32 | 6.9 | 10.3 | 13 | 12.6 | 0.42 | 12.27 | 0.413 | 1.06 | 0.052 | 0.494 |
| layer2.8 | 32 | 7.1 | 11.1 | 15 | 13.3 | 0.45 | 7.19 | 0.406 | 1.25 | 0.053 | 0.532 |
| layer3.0 | 64 | 8.0 | 13.5 | 21 | 15.1 | 0.53 | 3.87 | 0.357 | 1.23 | 0.052 | 0.587 |
| layer3.5 | 64 | 13.9 | 25.3 | 43 | 19.4 | 0.78 | 0.84 | 0.246 | 2.00 | 0.049 | 0.790 |
| layer3.8 | 64 | 8.9 | 10.3 | 9 | 10.8 | 3.37 | 0.13 | 0.082 | 0.87 | 0.131 | 0.926 |
| penult | 64 | 8.9 | 10.3 | 9 | 10.8 | 3.37 | 0.13 | 0.082 | 0.87 | 0.131 | 0.926 |

## 2. Decodability profile (which factor lives where)
categorical = accuracy minus majority-class rate; continuous = CV R²

| factor | kind | stem | layer1.0 | layer1.5 | layer1.8 | layer2.0 | layer2.5 | layer2.8 | layer3.0 | layer3.5 | layer3.8 | penult | commit | peak | washout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| luminance_mean | continuous | 1.00 | 0.98 | 0.93 | 0.93 | 0.91 | 0.85 | 0.83 | 0.73 | 0.59 | 0.30 | 0.30 | stem | stem | 0.69 |
| contrast_rms | continuous | 0.63 | 0.65 | 0.61 | 0.61 | 0.65 | 0.64 | 0.64 | 0.62 | 0.55 | 0.41 | 0.41 | stem | layer2.0 | 0.24 |
| highfreq_ratio | continuous | 0.22 | 0.44 | 0.60 | 0.58 | 0.61 | 0.60 | 0.60 | 0.61 | 0.58 | 0.52 | 0.52 | layer1.5 | layer3.0 | 0.09 |
| spectral_slope | continuous | 0.12 | 0.32 | 0.62 | 0.61 | 0.67 | 0.65 | 0.65 | 0.64 | 0.60 | 0.50 | 0.50 | layer1.5 | layer2.0 | 0.16 |
| spectral_anisotropy | continuous | 0.15 | 0.19 | 0.26 | 0.27 | 0.29 | 0.32 | 0.33 | 0.35 | 0.35 | 0.28 | 0.28 | layer2.5 | layer3.5 | 0.07 |
| noise_sigma | continuous | 0.65 | 0.80 | 0.91 | 0.91 | 0.95 | 0.93 | 0.93 | 0.92 | 0.86 | 0.77 | 0.77 | layer1.5 | layer2.0 | 0.17 |
| saturation_mean | continuous | 0.78 | 0.84 | 0.76 | 0.74 | 0.73 | 0.68 | 0.64 | 0.59 | 0.46 | 0.29 | 0.29 | stem | layer1.0 | 0.55 |
| hue_cos | continuous | 0.79 | 0.81 | 0.79 | 0.75 | 0.79 | 0.78 | 0.77 | 0.71 | 0.62 | 0.51 | 0.51 | stem | layer1.0 | 0.29 |
| hue_sin | continuous | 0.77 | 0.82 | 0.78 | 0.74 | 0.74 | 0.71 | 0.71 | 0.68 | 0.60 | 0.50 | 0.50 | stem | layer1.0 | 0.32 |
| colorfulness | continuous | 0.65 | 0.86 | 0.84 | 0.81 | 0.83 | 0.79 | 0.76 | 0.69 | 0.57 | 0.36 | 0.36 | layer1.0 | layer1.0 | 0.51 |
| edge_density | continuous | 0.73 | 0.82 | 0.82 | 0.82 | 0.89 | 0.88 | 0.87 | 0.86 | 0.78 | 0.65 | 0.65 | layer1.0 | layer2.0 | 0.24 |
| orientation_entropy | continuous | 0.17 | 0.22 | 0.34 | 0.39 | 0.44 | 0.49 | 0.50 | 0.52 | 0.48 | 0.41 | 0.41 | layer2.5 | layer3.0 | 0.12 |
| blockiness | continuous | 0.01 | 0.01 | 0.02 | 0.03 | 0.07 | 0.07 | 0.07 | 0.10 | 0.09 | 0.05 | 0.05 | layer3.0 | layer3.0 | 0.06 |
| class | categorical | 0.24 | 0.25 | 0.36 | 0.39 | 0.48 | 0.53 | 0.57 | 0.65 | 0.75 | 0.82 | 0.82 | layer3.5 | layer3.8 | 0.00 |
| coarse_animal_vehicle | categorical | 0.22 | 0.24 | 0.28 | 0.29 | 0.31 | 0.33 | 0.34 | 0.35 | 0.37 | 0.39 | 0.39 | layer3.5 | layer3.8 | 0.00 |
| corruption_family | categorical | 0.09 | 0.17 | 0.33 | 0.34 | 0.39 | 0.37 | 0.37 | 0.37 | 0.32 | 0.25 | 0.25 | layer2.0 | layer2.0 | 0.14 |
| corruption_type | categorical | 0.14 | 0.19 | 0.41 | 0.45 | 0.52 | 0.53 | 0.52 | 0.51 | 0.43 | 0.29 | 0.29 | layer2.0 | layer2.5 | 0.24 |
| severity | continuous | 0.05 | 0.09 | 0.13 | 0.15 | 0.28 | 0.28 | 0.28 | 0.33 | 0.32 | 0.25 | 0.25 | layer3.0 | layer3.0 | 0.08 |

## 3. Layer flow (CKA)
| pair | CKA |
|---|---|
| stem->layer1.0 | 0.886 |
| layer1.0->layer1.5 | 0.789 |
| layer1.5->layer1.8 | 0.851 |
| layer1.8->layer2.0 | 0.961 |
| layer2.0->layer2.5 | 0.960 |
| layer2.5->layer2.8 | 0.982 |
| layer2.8->layer3.0 | 0.940 |
| layer3.0->layer3.5 | 0.836 |
| layer3.5->layer3.8 | 0.650 |
| layer3.8->penult | 1.000 |

biggest reorganization: `layer3.5->layer3.8` (drop 0.350)

## 4. Sensitivity field (paired corruption displacement)
layer `penult`; magnitude in within-class-radius units

| corruption | sev | magnitude | per-sample | top5 frac | class-sub frac | coherence | class consistency | norm ratio |
|---|---|---|---|---|---|---|---|---|
| brightness | 1 | 0.03 | 0.14 | 0.84 | 0.90 | 0.22 | 0.64 | 1.001 |
| brightness | 3 | 0.10 | 0.40 | 0.85 | 0.91 | 0.25 | 0.58 | 0.998 |
| brightness | 5 | 0.23 | 0.78 | 0.83 | 0.91 | 0.28 | 0.44 | 0.984 |
| contrast | 1 | 0.13 | 0.36 | 0.59 | 0.89 | 0.36 | 0.68 | 1.011 |
| contrast | 3 | 0.51 | 1.17 | 0.51 | 0.86 | 0.44 | 0.57 | 1.005 |
| contrast | 5 | 1.60 | 2.73 | 0.66 | 0.88 | 0.57 | 0.43 | 0.988 |
| defocus_blur | 1 | 0.06 | 0.18 | 0.80 | 0.89 | 0.32 | 0.70 | 1.004 |
| defocus_blur | 3 | 0.47 | 0.96 | 0.90 | 0.91 | 0.45 | 0.65 | 0.996 |
| defocus_blur | 5 | 1.16 | 2.12 | 0.91 | 0.92 | 0.52 | 0.45 | 0.963 |
| elastic_transform | 1 | 0.22 | 0.91 | 0.83 | 0.85 | 0.23 | 0.35 | 0.982 |
| elastic_transform | 3 | 0.45 | 1.15 | 0.88 | 0.90 | 0.36 | 0.49 | 0.982 |
| elastic_transform | 5 | 0.56 | 1.58 | 0.70 | 0.92 | 0.34 | 0.25 | 0.952 |
| fog | 1 | 0.08 | 0.25 | 0.67 | 0.86 | 0.33 | 0.72 | 1.008 |
| fog | 3 | 0.30 | 0.75 | 0.63 | 0.87 | 0.40 | 0.67 | 1.009 |
| fog | 5 | 0.86 | 1.70 | 0.72 | 0.92 | 0.49 | 0.49 | 0.969 |
| frost | 1 | 0.29 | 0.76 | 0.54 | 0.90 | 0.34 | 0.49 | 0.983 |
| frost | 3 | 0.71 | 1.52 | 0.67 | 0.92 | 0.43 | 0.42 | 0.948 |
| frost | 5 | 1.03 | 1.94 | 0.74 | 0.94 | 0.50 | 0.45 | 0.943 |
| gaussian_noise | 1 | 0.70 | 1.34 | 0.77 | 0.93 | 0.50 | 0.54 | 0.949 |
| gaussian_noise | 3 | 1.55 | 2.48 | 0.87 | 0.93 | 0.60 | 0.50 | 0.911 |
| gaussian_noise | 5 | 1.78 | 2.78 | 0.84 | 0.90 | 0.61 | 0.46 | 0.889 |
| glass_blur | 1 | 0.97 | 2.11 | 0.82 | 0.94 | 0.42 | 0.27 | 0.915 |
| glass_blur | 3 | 0.86 | 1.98 | 0.82 | 0.95 | 0.40 | 0.26 | 0.924 |
| glass_blur | 5 | 1.01 | 2.24 | 0.85 | 0.95 | 0.41 | 0.23 | 0.908 |
| impulse_noise | 1 | 0.45 | 1.13 | 0.61 | 0.81 | 0.35 | 0.48 | 1.005 |
| impulse_noise | 3 | 0.80 | 1.91 | 0.74 | 0.80 | 0.41 | 0.27 | 0.940 |
| impulse_noise | 5 | 1.73 | 2.86 | 0.63 | 0.86 | 0.58 | 0.38 | 0.858 |
| jpeg_compression | 1 | 0.33 | 0.99 | 0.42 | 0.93 | 0.33 | 0.41 | 0.974 |
| jpeg_compression | 3 | 0.43 | 1.33 | 0.39 | 0.91 | 0.32 | 0.23 | 0.955 |
| jpeg_compression | 5 | 0.51 | 1.52 | 0.46 | 0.91 | 0.32 | 0.20 | 0.946 |
| motion_blur | 1 | 0.34 | 0.88 | 0.86 | 0.91 | 0.36 | 0.56 | 0.987 |
| motion_blur | 3 | 0.71 | 1.61 | 0.89 | 0.91 | 0.42 | 0.41 | 0.960 |
| motion_blur | 5 | 0.84 | 1.86 | 0.89 | 0.91 | 0.43 | 0.37 | 0.947 |
| pixelate | 1 | 0.24 | 0.64 | 0.61 | 0.94 | 0.34 | 0.62 | 0.988 |
| pixelate | 3 | 0.66 | 1.26 | 0.55 | 0.97 | 0.47 | 0.56 | 0.950 |
| pixelate | 5 | 1.34 | 2.39 | 0.37 | 0.94 | 0.52 | 0.38 | 0.918 |
| shot_noise | 1 | 0.53 | 1.04 | 0.69 | 0.92 | 0.47 | 0.59 | 0.963 |
| shot_noise | 3 | 1.29 | 2.11 | 0.84 | 0.93 | 0.58 | 0.54 | 0.927 |
| shot_noise | 5 | 1.67 | 2.65 | 0.81 | 0.90 | 0.61 | 0.47 | 0.894 |
| snow | 1 | 0.26 | 0.87 | 0.39 | 0.75 | 0.29 | 0.42 | 1.004 |
| snow | 3 | 0.35 | 1.27 | 0.28 | 0.78 | 0.28 | 0.21 | 0.975 |
| snow | 5 | 0.52 | 1.53 | 0.38 | 0.91 | 0.34 | 0.24 | 0.954 |
| zoom_blur | 1 | 0.43 | 1.03 | 0.88 | 0.90 | 0.39 | 0.54 | 0.984 |
| zoom_blur | 3 | 0.71 | 1.49 | 0.91 | 0.91 | 0.45 | 0.49 | 0.963 |
| zoom_blur | 5 | 1.03 | 2.01 | 0.90 | 0.91 | 0.48 | 0.41 | 0.943 |

## 5. Density per split
layer `penult`

| split | sparse frac (ref q95) | median log-radius shift |
|---|---|---|
| corrupt__brightness__s1 | 0.130 | 0.040 |
| corrupt__brightness__s3 | 0.144 | 0.061 |
| corrupt__brightness__s5 | 0.195 | 0.134 |
| corrupt__contrast__s1 | 0.136 | 0.051 |
| corrupt__contrast__s3 | 0.302 | 0.275 |
| corrupt__contrast__s5 | 0.875 | 0.728 |
| corrupt__defocus_blur__s1 | 0.130 | 0.045 |
| corrupt__defocus_blur__s3 | 0.247 | 0.210 |
| corrupt__defocus_blur__s5 | 0.561 | 0.510 |
| corrupt__elastic_transform__s1 | 0.213 | 0.167 |
| corrupt__elastic_transform__s3 | 0.280 | 0.254 |
| corrupt__elastic_transform__s5 | 0.456 | 0.431 |
| corrupt__fog__s1 | 0.135 | 0.047 |
| corrupt__fog__s3 | 0.197 | 0.115 |
| corrupt__fog__s5 | 0.460 | 0.433 |
| corrupt__frost__s1 | 0.197 | 0.145 |
| corrupt__frost__s3 | 0.396 | 0.370 |
| corrupt__frost__s5 | 0.496 | 0.468 |
| corrupt__gaussian_noise__s1 | 0.320 | 0.312 |
| corrupt__gaussian_noise__s3 | 0.538 | 0.502 |
| corrupt__gaussian_noise__s5 | 0.580 | 0.522 |
| corrupt__glass_blur__s1 | 0.573 | 0.525 |
| corrupt__glass_blur__s3 | 0.538 | 0.498 |
| corrupt__glass_blur__s5 | 0.591 | 0.532 |
| corrupt__impulse_noise__s1 | 0.325 | 0.277 |
| corrupt__impulse_noise__s3 | 0.596 | 0.547 |
| corrupt__impulse_noise__s5 | 0.658 | 0.543 |
| corrupt__jpeg_compression__s1 | 0.239 | 0.192 |
| corrupt__jpeg_compression__s3 | 0.351 | 0.338 |
| corrupt__jpeg_compression__s5 | 0.399 | 0.395 |
| corrupt__motion_blur__s1 | 0.216 | 0.170 |
| corrupt__motion_blur__s3 | 0.432 | 0.411 |
| corrupt__motion_blur__s5 | 0.481 | 0.459 |
| corrupt__pixelate__s1 | 0.195 | 0.098 |
| corrupt__pixelate__s3 | 0.346 | 0.275 |
| corrupt__pixelate__s5 | 0.595 | 0.567 |
| corrupt__shot_noise__s1 | 0.260 | 0.217 |
| corrupt__shot_noise__s3 | 0.500 | 0.471 |
| corrupt__shot_noise__s5 | 0.586 | 0.527 |
| corrupt__snow__s1 | 0.239 | 0.194 |
| corrupt__snow__s3 | 0.349 | 0.338 |
| corrupt__snow__s5 | 0.423 | 0.405 |
| corrupt__zoom_blur__s1 | 0.277 | 0.255 |
| corrupt__zoom_blur__s3 | 0.415 | 0.395 |
| corrupt__zoom_blur__s5 | 0.545 | 0.500 |
| ood__cifar100 | 0.643 | 0.559 |
| ood__svhn | 0.834 | 0.663 |
| panel | 0.031 | -0.020 |
| test | 0.131 | 0.028 |

## 6. Adjacency
layer `penult`


closest class pairs (sep = center distance / RMS radius): (3,5) 2.55, (2,3) 2.87, (2,5) 3.05, (3,4) 3.11, (0,2) 3.18, (2,4) 3.18

valley ratio mean 2.49, min 1.58, pairs with a valley 1.00

nearest-center confusions (true → nearest): 5→3 0.08, 3→5 0.06, 0→8 0.03, 9→1 0.03, 3→4 0.03, 3→6 0.02

single-linkage merge order (first 5): [3, 5]@2.55 ; [2, 3, 5]@2.87 ; [2, 3, 4, 5]@3.11 ; [0, 2, 3, 4, 5]@3.18 ; [0, 2, 3, 4, 5, 7]@3.19

## 7. Type-b margin (margin_typeb)
_margin_typeb not run_
