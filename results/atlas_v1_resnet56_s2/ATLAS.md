# ATLAS — atlas_v1_resnet56_s2
built 2026-09-23 14:25:09 · source **real** · arch `cifar10_resnet56` · weights `file:/workspace/models/resnet56_s2_chenyaofo.pt sha256:14a651f000ca2d55` · layers 11

## 1. Per-layer invariants
| layer | D | PR | eff.rank | dim95 | TwoNN ID | sep ratio | NC1 | ETF dev | hub skew | test sparse | NC acc(test) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | 16 | 2.7 | 3.5 | 4 | 6.9 | 0.16 | 213.96 | 0.644 | 0.16 | 0.058 | 0.250 |
| layer1.0 | 16 | 3.6 | 4.8 | 5 | 8.2 | 0.17 | 133.13 | 0.600 | 0.39 | 0.054 | 0.266 |
| layer1.5 | 16 | 5.7 | 7.6 | 9 | 10.0 | 0.30 | 86.67 | 0.473 | 0.78 | 0.058 | 0.366 |
| layer1.8 | 16 | 5.5 | 7.8 | 10 | 10.6 | 0.34 | 66.03 | 0.423 | 0.90 | 0.059 | 0.385 |
| layer2.0 | 32 | 6.0 | 8.9 | 11 | 11.5 | 0.36 | 16.82 | 0.456 | 0.99 | 0.049 | 0.420 |
| layer2.5 | 32 | 7.6 | 11.6 | 15 | 13.2 | 0.39 | 9.03 | 0.429 | 1.26 | 0.044 | 0.509 |
| layer2.8 | 32 | 8.8 | 13.3 | 17 | 14.0 | 0.44 | 5.41 | 0.372 | 1.42 | 0.045 | 0.561 |
| layer3.0 | 64 | 9.3 | 15.9 | 24 | 15.5 | 0.53 | 3.34 | 0.365 | 1.48 | 0.048 | 0.613 |
| layer3.5 | 64 | 14.7 | 26.0 | 42 | 19.5 | 0.93 | 0.61 | 0.187 | 1.75 | 0.049 | 0.837 |
| layer3.8 | 64 | 9.0 | 9.9 | 9 | 10.5 | 5.21 | 0.05 | 0.062 | 0.85 | 0.184 | 0.940 |
| penult | 64 | 9.0 | 9.9 | 9 | 10.5 | 5.21 | 0.05 | 0.062 | 0.85 | 0.184 | 0.940 |

## 2. Decodability profile (which factor lives where)
categorical = accuracy minus majority-class rate; continuous = CV R²

| factor | kind | stem | layer1.0 | layer1.5 | layer1.8 | layer2.0 | layer2.5 | layer2.8 | layer3.0 | layer3.5 | layer3.8 | penult | commit | peak | washout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| luminance_mean | continuous | 1.00 | 0.99 | 0.97 | 0.94 | 0.93 | 0.89 | 0.86 | 0.81 | 0.66 | 0.33 | 0.33 | stem | stem | 0.67 |
| contrast_rms | continuous | 0.72 | 0.72 | 0.61 | 0.60 | 0.64 | 0.61 | 0.58 | 0.62 | 0.54 | 0.36 | 0.36 | stem | layer1.0 | 0.35 |
| highfreq_ratio | continuous | 0.23 | 0.52 | 0.56 | 0.57 | 0.60 | 0.60 | 0.59 | 0.63 | 0.57 | 0.48 | 0.48 | layer2.0 | layer3.0 | 0.15 |
| spectral_slope | continuous | 0.13 | 0.43 | 0.55 | 0.57 | 0.64 | 0.65 | 0.65 | 0.67 | 0.62 | 0.50 | 0.50 | layer2.0 | layer3.0 | 0.17 |
| spectral_anisotropy | continuous | 0.19 | 0.24 | 0.29 | 0.29 | 0.35 | 0.32 | 0.31 | 0.35 | 0.33 | 0.26 | 0.26 | layer2.0 | layer3.0 | 0.09 |
| noise_sigma | continuous | 0.71 | 0.89 | 0.94 | 0.94 | 0.95 | 0.93 | 0.93 | 0.92 | 0.85 | 0.70 | 0.70 | layer1.0 | layer2.0 | 0.25 |
| saturation_mean | continuous | 0.80 | 0.82 | 0.73 | 0.67 | 0.69 | 0.66 | 0.59 | 0.58 | 0.45 | 0.27 | 0.27 | stem | layer1.0 | 0.55 |
| hue_cos | continuous | 0.76 | 0.86 | 0.82 | 0.81 | 0.80 | 0.79 | 0.77 | 0.72 | 0.64 | 0.45 | 0.45 | layer1.0 | layer1.0 | 0.40 |
| hue_sin | continuous | 0.79 | 0.84 | 0.76 | 0.75 | 0.75 | 0.75 | 0.74 | 0.71 | 0.66 | 0.52 | 0.52 | stem | layer1.0 | 0.33 |
| colorfulness | continuous | 0.72 | 0.81 | 0.84 | 0.82 | 0.77 | 0.75 | 0.69 | 0.70 | 0.54 | 0.35 | 0.35 | layer1.0 | layer1.5 | 0.49 |
| edge_density | continuous | 0.84 | 0.89 | 0.86 | 0.87 | 0.90 | 0.88 | 0.86 | 0.85 | 0.76 | 0.58 | 0.58 | stem | layer2.0 | 0.32 |
| orientation_entropy | continuous | 0.16 | 0.26 | 0.40 | 0.41 | 0.47 | 0.46 | 0.46 | 0.51 | 0.44 | 0.39 | 0.39 | layer2.0 | layer3.0 | 0.11 |
| blockiness | continuous | 0.01 | 0.01 | 0.03 | 0.05 | 0.08 | 0.08 | 0.08 | 0.12 | 0.12 | 0.06 | 0.06 | layer3.0 | layer3.5 | 0.06 |
| class | categorical | 0.25 | 0.27 | 0.36 | 0.38 | 0.50 | 0.57 | 0.60 | 0.67 | 0.78 | 0.83 | 0.83 | layer3.5 | layer3.8 | 0.00 |
| coarse_animal_vehicle | categorical | 0.24 | 0.24 | 0.28 | 0.28 | 0.31 | 0.33 | 0.33 | 0.36 | 0.37 | 0.39 | 0.39 | layer3.0 | layer3.8 | 0.00 |
| corruption_family | categorical | 0.11 | 0.22 | 0.33 | 0.36 | 0.41 | 0.40 | 0.39 | 0.39 | 0.34 | 0.25 | 0.25 | layer2.0 | layer2.0 | 0.17 |
| corruption_type | categorical | 0.16 | 0.21 | 0.46 | 0.51 | 0.55 | 0.55 | 0.55 | 0.54 | 0.46 | 0.28 | 0.28 | layer1.8 | layer2.8 | 0.27 |
| severity | continuous | 0.07 | 0.07 | 0.11 | 0.15 | 0.30 | 0.29 | 0.27 | 0.31 | 0.29 | 0.23 | 0.23 | layer2.0 | layer3.0 | 0.08 |

## 3. Layer flow (CKA)
| pair | CKA |
|---|---|
| stem->layer1.0 | 0.948 |
| layer1.0->layer1.5 | 0.701 |
| layer1.5->layer1.8 | 0.904 |
| layer1.8->layer2.0 | 0.950 |
| layer2.0->layer2.5 | 0.951 |
| layer2.5->layer2.8 | 0.981 |
| layer2.8->layer3.0 | 0.929 |
| layer3.0->layer3.5 | 0.789 |
| layer3.5->layer3.8 | 0.665 |
| layer3.8->penult | 1.000 |

biggest reorganization: `layer3.5->layer3.8` (drop 0.335)

## 4. Sensitivity field (paired corruption displacement)
layer `penult`; magnitude in within-class-radius units

| corruption | sev | magnitude | per-sample | top5 frac | class-sub frac | coherence | class consistency | norm ratio |
|---|---|---|---|---|---|---|---|---|
| brightness | 1 | 0.03 | 0.17 | 0.63 | 0.85 | 0.18 | 0.60 | 0.999 |
| brightness | 3 | 0.11 | 0.50 | 0.67 | 0.84 | 0.22 | 0.50 | 0.992 |
| brightness | 5 | 0.28 | 0.98 | 0.67 | 0.87 | 0.28 | 0.37 | 0.972 |
| contrast | 1 | 0.12 | 0.38 | 0.74 | 0.85 | 0.32 | 0.63 | 0.992 |
| contrast | 3 | 0.46 | 1.28 | 0.62 | 0.82 | 0.37 | 0.39 | 0.944 |
| contrast | 5 | 1.56 | 3.36 | 0.59 | 0.89 | 0.44 | 0.19 | 0.781 |
| defocus_blur | 1 | 0.06 | 0.20 | 0.69 | 0.81 | 0.25 | 0.59 | 0.997 |
| defocus_blur | 3 | 0.47 | 1.15 | 0.80 | 0.89 | 0.37 | 0.48 | 0.966 |
| defocus_blur | 5 | 1.15 | 2.61 | 0.85 | 0.93 | 0.41 | 0.24 | 0.894 |
| elastic_transform | 1 | 0.23 | 1.10 | 0.67 | 0.77 | 0.21 | 0.22 | 0.974 |
| elastic_transform | 3 | 0.46 | 1.39 | 0.77 | 0.88 | 0.31 | 0.34 | 0.959 |
| elastic_transform | 5 | 0.63 | 2.03 | 0.69 | 0.92 | 0.30 | 0.14 | 0.940 |
| fog | 1 | 0.08 | 0.28 | 0.84 | 0.89 | 0.29 | 0.63 | 0.997 |
| fog | 3 | 0.29 | 0.84 | 0.77 | 0.86 | 0.34 | 0.50 | 0.976 |
| fog | 5 | 0.92 | 2.04 | 0.78 | 0.93 | 0.43 | 0.37 | 0.921 |
| frost | 1 | 0.36 | 0.99 | 0.76 | 0.90 | 0.34 | 0.45 | 0.977 |
| frost | 3 | 0.91 | 2.02 | 0.87 | 0.92 | 0.42 | 0.36 | 0.940 |
| frost | 5 | 1.34 | 2.64 | 0.89 | 0.93 | 0.47 | 0.38 | 0.929 |
| gaussian_noise | 1 | 0.97 | 1.86 | 0.89 | 0.94 | 0.48 | 0.54 | 0.957 |
| gaussian_noise | 3 | 2.20 | 3.53 | 0.88 | 0.95 | 0.58 | 0.47 | 0.952 |
| gaussian_noise | 5 | 2.54 | 4.00 | 0.87 | 0.94 | 0.61 | 0.46 | 0.966 |
| glass_blur | 1 | 1.37 | 2.99 | 0.75 | 0.93 | 0.41 | 0.25 | 0.930 |
| glass_blur | 3 | 1.21 | 2.77 | 0.80 | 0.94 | 0.38 | 0.23 | 0.926 |
| glass_blur | 5 | 1.47 | 3.22 | 0.76 | 0.94 | 0.40 | 0.22 | 0.918 |
| impulse_noise | 1 | 0.67 | 1.58 | 0.62 | 0.82 | 0.39 | 0.52 | 1.007 |
| impulse_noise | 3 | 1.14 | 2.74 | 0.74 | 0.82 | 0.39 | 0.26 | 0.965 |
| impulse_noise | 5 | 2.18 | 3.98 | 0.72 | 0.91 | 0.52 | 0.30 | 0.942 |
| jpeg_compression | 1 | 0.41 | 1.29 | 0.60 | 0.89 | 0.31 | 0.34 | 0.965 |
| jpeg_compression | 3 | 0.50 | 1.72 | 0.48 | 0.87 | 0.29 | 0.16 | 0.947 |
| jpeg_compression | 5 | 0.59 | 2.00 | 0.48 | 0.87 | 0.28 | 0.11 | 0.935 |
| motion_blur | 1 | 0.35 | 1.03 | 0.78 | 0.89 | 0.31 | 0.42 | 0.970 |
| motion_blur | 3 | 0.82 | 2.01 | 0.77 | 0.93 | 0.37 | 0.29 | 0.918 |
| motion_blur | 5 | 1.01 | 2.39 | 0.73 | 0.94 | 0.38 | 0.24 | 0.896 |
| pixelate | 1 | 0.25 | 0.75 | 0.56 | 0.92 | 0.30 | 0.56 | 0.991 |
| pixelate | 3 | 0.76 | 1.58 | 0.47 | 0.95 | 0.44 | 0.53 | 0.967 |
| pixelate | 5 | 1.81 | 3.20 | 0.39 | 0.95 | 0.53 | 0.43 | 0.954 |
| shot_noise | 1 | 0.68 | 1.37 | 0.84 | 0.93 | 0.46 | 0.58 | 0.970 |
| shot_noise | 3 | 1.80 | 2.99 | 0.89 | 0.94 | 0.56 | 0.50 | 0.952 |
| shot_noise | 5 | 2.34 | 3.75 | 0.86 | 0.94 | 0.59 | 0.46 | 0.965 |
| snow | 1 | 0.36 | 1.09 | 0.58 | 0.81 | 0.29 | 0.39 | 0.999 |
| snow | 3 | 0.45 | 1.61 | 0.49 | 0.72 | 0.28 | 0.19 | 0.978 |
| snow | 5 | 0.72 | 2.02 | 0.63 | 0.87 | 0.35 | 0.25 | 0.955 |
| zoom_blur | 1 | 0.48 | 1.27 | 0.86 | 0.91 | 0.34 | 0.41 | 0.965 |
| zoom_blur | 3 | 0.79 | 1.84 | 0.85 | 0.93 | 0.40 | 0.37 | 0.938 |
| zoom_blur | 5 | 1.18 | 2.54 | 0.83 | 0.94 | 0.43 | 0.31 | 0.914 |

## 5. Density per split
layer `penult`

| split | sparse frac (ref q95) | median log-radius shift |
|---|---|---|
| corrupt__brightness__s1 | 0.186 | 0.042 |
| corrupt__brightness__s3 | 0.202 | 0.087 |
| corrupt__brightness__s5 | 0.262 | 0.172 |
| corrupt__contrast__s1 | 0.197 | 0.047 |
| corrupt__contrast__s3 | 0.319 | 0.231 |
| corrupt__contrast__s5 | 0.864 | 0.989 |
| corrupt__defocus_blur__s1 | 0.195 | 0.039 |
| corrupt__defocus_blur__s3 | 0.314 | 0.235 |
| corrupt__defocus_blur__s5 | 0.667 | 0.782 |
| corrupt__elastic_transform__s1 | 0.299 | 0.216 |
| corrupt__elastic_transform__s3 | 0.365 | 0.334 |
| corrupt__elastic_transform__s5 | 0.551 | 0.630 |
| corrupt__fog__s1 | 0.192 | 0.040 |
| corrupt__fog__s3 | 0.243 | 0.110 |
| corrupt__fog__s5 | 0.513 | 0.551 |
| corrupt__frost__s1 | 0.285 | 0.187 |
| corrupt__frost__s3 | 0.506 | 0.551 |
| corrupt__frost__s5 | 0.601 | 0.695 |
| corrupt__gaussian_noise__s1 | 0.454 | 0.456 |
| corrupt__gaussian_noise__s3 | 0.670 | 0.784 |
| corrupt__gaussian_noise__s5 | 0.727 | 0.851 |
| corrupt__glass_blur__s1 | 0.708 | 0.859 |
| corrupt__glass_blur__s3 | 0.674 | 0.804 |
| corrupt__glass_blur__s5 | 0.734 | 0.884 |
| corrupt__impulse_noise__s1 | 0.448 | 0.441 |
| corrupt__impulse_noise__s3 | 0.702 | 0.853 |
| corrupt__impulse_noise__s5 | 0.892 | 0.997 |
| corrupt__jpeg_compression__s1 | 0.333 | 0.271 |
| corrupt__jpeg_compression__s3 | 0.448 | 0.442 |
| corrupt__jpeg_compression__s5 | 0.530 | 0.580 |
| corrupt__motion_blur__s1 | 0.287 | 0.190 |
| corrupt__motion_blur__s3 | 0.520 | 0.574 |
| corrupt__motion_blur__s5 | 0.596 | 0.676 |
| corrupt__pixelate__s1 | 0.258 | 0.134 |
| corrupt__pixelate__s3 | 0.418 | 0.366 |
| corrupt__pixelate__s5 | 0.673 | 0.870 |
| corrupt__shot_noise__s1 | 0.362 | 0.309 |
| corrupt__shot_noise__s3 | 0.648 | 0.770 |
| corrupt__shot_noise__s5 | 0.732 | 0.857 |
| corrupt__snow__s1 | 0.324 | 0.256 |
| corrupt__snow__s3 | 0.443 | 0.451 |
| corrupt__snow__s5 | 0.519 | 0.564 |
| corrupt__zoom_blur__s1 | 0.347 | 0.302 |
| corrupt__zoom_blur__s3 | 0.479 | 0.497 |
| corrupt__zoom_blur__s5 | 0.632 | 0.725 |
| ood__cifar100 | 0.794 | 0.918 |
| ood__svhn | 0.924 | 1.052 |
| panel | 0.016 | -0.001 |
| test | 0.184 | 0.031 |

## 6. Adjacency
layer `penult`


closest class pairs (sep = center distance / RMS radius): (2,3) 4.38, (3,5) 4.47, (2,5) 4.79, (0,2) 4.84, (3,4) 4.92, (0,3) 4.98

valley ratio mean 4.38, min 3.01, pairs with a valley 1.00

nearest-center confusions (true → nearest): 3→5 0.06, 5→3 0.06, 9→1 0.03, 0→8 0.02, 3→6 0.02, 2→6 0.02

single-linkage merge order (first 5): [2, 3]@4.38 ; [2, 3, 5]@4.47 ; [0, 2, 3, 5]@4.84 ; [0, 2, 3, 4, 5]@4.92 ; [0, 2, 3, 4, 5, 7]@5.02

## 7. Type-b margin (margin_typeb)
_margin_typeb not run_
