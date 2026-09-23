# ATLAS — atlas_v1_resnet56_s1
built 2026-09-23 14:22:25 · source **real** · arch `cifar10_resnet56` · weights `file:/workspace/models/resnet56_s1_chenyaofo.pt sha256:73c9458c60b66949` · layers 11

## 1. Per-layer invariants
| layer | D | PR | eff.rank | dim95 | TwoNN ID | sep ratio | NC1 | ETF dev | hub skew | test sparse | NC acc(test) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | 16 | 2.6 | 3.2 | 3 | 6.2 | 0.16 | 499.25 | 0.662 | 0.10 | 0.056 | 0.235 |
| layer1.0 | 16 | 2.6 | 3.5 | 4 | 7.4 | 0.16 | 264.54 | 0.658 | 0.27 | 0.056 | 0.256 |
| layer1.5 | 16 | 4.8 | 6.4 | 7 | 9.6 | 0.23 | 68.49 | 0.510 | 0.70 | 0.056 | 0.328 |
| layer1.8 | 16 | 5.3 | 7.4 | 8 | 9.9 | 0.31 | 52.91 | 0.459 | 0.80 | 0.056 | 0.366 |
| layer2.0 | 32 | 5.9 | 8.5 | 10 | 10.9 | 0.32 | 30.98 | 0.498 | 0.84 | 0.061 | 0.398 |
| layer2.5 | 32 | 7.6 | 11.5 | 15 | 13.0 | 0.38 | 12.95 | 0.420 | 1.16 | 0.052 | 0.470 |
| layer2.8 | 32 | 8.8 | 13.4 | 18 | 13.9 | 0.42 | 7.27 | 0.397 | 1.64 | 0.047 | 0.538 |
| layer3.0 | 64 | 9.5 | 16.1 | 25 | 15.5 | 0.52 | 3.49 | 0.357 | 1.33 | 0.050 | 0.603 |
| layer3.5 | 64 | 15.7 | 27.7 | 44 | 19.6 | 0.86 | 0.68 | 0.216 | 2.04 | 0.044 | 0.824 |
| layer3.8 | 64 | 8.9 | 9.8 | 9 | 10.4 | 5.11 | 0.05 | 0.057 | 0.89 | 0.163 | 0.945 |
| penult | 64 | 8.9 | 9.8 | 9 | 10.4 | 5.11 | 0.05 | 0.057 | 0.89 | 0.163 | 0.945 |

## 2. Decodability profile (which factor lives where)
categorical = accuracy minus majority-class rate; continuous = CV R²

| factor | kind | stem | layer1.0 | layer1.5 | layer1.8 | layer2.0 | layer2.5 | layer2.8 | layer3.0 | layer3.5 | layer3.8 | penult | commit | peak | washout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| luminance_mean | continuous | 1.00 | 0.99 | 0.98 | 0.98 | 0.97 | 0.95 | 0.93 | 0.87 | 0.75 | 0.39 | 0.39 | stem | stem | 0.61 |
| contrast_rms | continuous | 0.60 | 0.59 | 0.61 | 0.59 | 0.68 | 0.66 | 0.65 | 0.67 | 0.57 | 0.36 | 0.36 | layer1.5 | layer2.0 | 0.32 |
| highfreq_ratio | continuous | 0.22 | 0.32 | 0.56 | 0.56 | 0.61 | 0.63 | 0.62 | 0.63 | 0.59 | 0.46 | 0.46 | layer2.0 | layer3.0 | 0.17 |
| spectral_slope | continuous | 0.12 | 0.22 | 0.57 | 0.56 | 0.65 | 0.66 | 0.64 | 0.66 | 0.61 | 0.45 | 0.45 | layer2.0 | layer2.5 | 0.20 |
| spectral_anisotropy | continuous | 0.17 | 0.21 | 0.28 | 0.29 | 0.33 | 0.33 | 0.33 | 0.37 | 0.35 | 0.27 | 0.27 | layer2.0 | layer3.0 | 0.10 |
| noise_sigma | continuous | 0.66 | 0.75 | 0.94 | 0.95 | 0.95 | 0.94 | 0.93 | 0.93 | 0.86 | 0.68 | 0.68 | layer1.5 | layer1.8 | 0.27 |
| saturation_mean | continuous | 0.69 | 0.77 | 0.67 | 0.63 | 0.79 | 0.74 | 0.71 | 0.71 | 0.57 | 0.30 | 0.30 | layer1.0 | layer2.0 | 0.49 |
| hue_cos | continuous | 0.78 | 0.82 | 0.81 | 0.80 | 0.82 | 0.80 | 0.79 | 0.79 | 0.71 | 0.52 | 0.52 | stem | layer1.0 | 0.30 |
| hue_sin | continuous | 0.75 | 0.80 | 0.79 | 0.78 | 0.78 | 0.75 | 0.74 | 0.71 | 0.66 | 0.48 | 0.48 | stem | layer1.0 | 0.32 |
| colorfulness | continuous | 0.68 | 0.87 | 0.80 | 0.78 | 0.86 | 0.83 | 0.82 | 0.78 | 0.67 | 0.33 | 0.33 | layer1.0 | layer1.0 | 0.54 |
| edge_density | continuous | 0.69 | 0.73 | 0.90 | 0.88 | 0.91 | 0.89 | 0.87 | 0.86 | 0.78 | 0.56 | 0.56 | layer1.5 | layer2.0 | 0.34 |
| orientation_entropy | continuous | 0.18 | 0.21 | 0.36 | 0.42 | 0.48 | 0.50 | 0.50 | 0.54 | 0.51 | 0.38 | 0.38 | layer2.5 | layer3.0 | 0.16 |
| blockiness | continuous | 0.01 | 0.01 | 0.02 | 0.04 | 0.06 | 0.07 | 0.07 | 0.10 | 0.08 | 0.04 | 0.04 | layer3.0 | layer3.0 | 0.06 |
| class | categorical | 0.23 | 0.25 | 0.32 | 0.36 | 0.46 | 0.52 | 0.58 | 0.66 | 0.78 | 0.84 | 0.84 | layer3.5 | layer3.8 | 0.00 |
| coarse_animal_vehicle | categorical | 0.23 | 0.24 | 0.26 | 0.27 | 0.31 | 0.32 | 0.33 | 0.35 | 0.37 | 0.40 | 0.40 | layer3.5 | layer3.8 | 0.00 |
| corruption_family | categorical | 0.10 | 0.16 | 0.35 | 0.35 | 0.41 | 0.36 | 0.37 | 0.38 | 0.33 | 0.25 | 0.25 | layer2.0 | layer2.0 | 0.16 |
| corruption_type | categorical | 0.15 | 0.19 | 0.44 | 0.48 | 0.55 | 0.56 | 0.55 | 0.54 | 0.45 | 0.27 | 0.27 | layer2.0 | layer2.5 | 0.29 |
| severity | continuous | 0.04 | 0.06 | 0.15 | 0.14 | 0.29 | 0.26 | 0.26 | 0.31 | 0.29 | 0.23 | 0.23 | layer2.0 | layer3.0 | 0.08 |

## 3. Layer flow (CKA)
| pair | CKA |
|---|---|
| stem->layer1.0 | 0.906 |
| layer1.0->layer1.5 | 0.775 |
| layer1.5->layer1.8 | 0.846 |
| layer1.8->layer2.0 | 0.919 |
| layer2.0->layer2.5 | 0.944 |
| layer2.5->layer2.8 | 0.977 |
| layer2.8->layer3.0 | 0.936 |
| layer3.0->layer3.5 | 0.787 |
| layer3.5->layer3.8 | 0.646 |
| layer3.8->penult | 1.000 |

biggest reorganization: `layer3.5->layer3.8` (drop 0.354)

## 4. Sensitivity field (paired corruption displacement)
layer `penult`; magnitude in within-class-radius units

| corruption | sev | magnitude | per-sample | top5 frac | class-sub frac | coherence | class consistency | norm ratio |
|---|---|---|---|---|---|---|---|---|
| brightness | 1 | 0.03 | 0.17 | 0.79 | 0.87 | 0.17 | 0.56 | 1.000 |
| brightness | 3 | 0.10 | 0.48 | 0.71 | 0.86 | 0.19 | 0.45 | 0.995 |
| brightness | 5 | 0.24 | 0.96 | 0.61 | 0.86 | 0.23 | 0.28 | 0.977 |
| contrast | 1 | 0.13 | 0.40 | 0.81 | 0.94 | 0.31 | 0.66 | 0.999 |
| contrast | 3 | 0.51 | 1.36 | 0.81 | 0.91 | 0.36 | 0.41 | 0.965 |
| contrast | 5 | 1.49 | 3.39 | 0.70 | 0.91 | 0.41 | 0.16 | 0.819 |
| defocus_blur | 1 | 0.06 | 0.20 | 0.59 | 0.88 | 0.24 | 0.61 | 0.999 |
| defocus_blur | 3 | 0.46 | 1.12 | 0.82 | 0.92 | 0.37 | 0.52 | 0.973 |
| defocus_blur | 5 | 1.25 | 2.58 | 0.86 | 0.95 | 0.44 | 0.33 | 0.915 |
| elastic_transform | 1 | 0.25 | 1.08 | 0.74 | 0.84 | 0.21 | 0.28 | 0.978 |
| elastic_transform | 3 | 0.50 | 1.40 | 0.81 | 0.92 | 0.32 | 0.40 | 0.969 |
| elastic_transform | 5 | 0.61 | 1.94 | 0.67 | 0.90 | 0.30 | 0.15 | 0.946 |
| fog | 1 | 0.09 | 0.29 | 0.84 | 0.94 | 0.30 | 0.68 | 1.001 |
| fog | 3 | 0.32 | 0.87 | 0.84 | 0.92 | 0.35 | 0.56 | 0.986 |
| fog | 5 | 0.96 | 2.08 | 0.79 | 0.95 | 0.44 | 0.37 | 0.944 |
| frost | 1 | 0.36 | 0.97 | 0.71 | 0.92 | 0.34 | 0.49 | 0.986 |
| frost | 3 | 0.94 | 2.01 | 0.80 | 0.94 | 0.43 | 0.38 | 0.954 |
| frost | 5 | 1.38 | 2.64 | 0.82 | 0.95 | 0.49 | 0.41 | 0.948 |
| gaussian_noise | 1 | 0.93 | 1.79 | 0.86 | 0.97 | 0.47 | 0.53 | 0.951 |
| gaussian_noise | 3 | 2.07 | 3.38 | 0.77 | 0.98 | 0.57 | 0.44 | 0.898 |
| gaussian_noise | 5 | 2.34 | 3.79 | 0.75 | 0.97 | 0.57 | 0.39 | 0.883 |
| glass_blur | 1 | 1.26 | 2.78 | 0.79 | 0.95 | 0.40 | 0.26 | 0.910 |
| glass_blur | 3 | 1.12 | 2.58 | 0.82 | 0.95 | 0.39 | 0.23 | 0.913 |
| glass_blur | 5 | 1.34 | 3.04 | 0.82 | 0.95 | 0.40 | 0.21 | 0.900 |
| impulse_noise | 1 | 0.57 | 1.46 | 0.56 | 0.81 | 0.36 | 0.46 | 0.989 |
| impulse_noise | 3 | 0.98 | 2.55 | 0.75 | 0.85 | 0.36 | 0.17 | 0.920 |
| impulse_noise | 5 | 1.72 | 3.66 | 0.61 | 0.93 | 0.44 | 0.16 | 0.810 |
| jpeg_compression | 1 | 0.38 | 1.20 | 0.73 | 0.91 | 0.31 | 0.34 | 0.968 |
| jpeg_compression | 3 | 0.49 | 1.68 | 0.57 | 0.88 | 0.29 | 0.14 | 0.946 |
| jpeg_compression | 5 | 0.59 | 1.99 | 0.52 | 0.88 | 0.29 | 0.10 | 0.937 |
| motion_blur | 1 | 0.34 | 0.99 | 0.90 | 0.93 | 0.30 | 0.47 | 0.985 |
| motion_blur | 3 | 0.82 | 1.95 | 0.92 | 0.94 | 0.38 | 0.34 | 0.951 |
| motion_blur | 5 | 1.04 | 2.36 | 0.92 | 0.95 | 0.40 | 0.30 | 0.938 |
| pixelate | 1 | 0.26 | 0.73 | 0.73 | 0.94 | 0.32 | 0.54 | 0.984 |
| pixelate | 3 | 0.79 | 1.53 | 0.64 | 0.96 | 0.45 | 0.53 | 0.948 |
| pixelate | 5 | 1.84 | 3.10 | 0.52 | 0.96 | 0.54 | 0.42 | 0.917 |
| shot_noise | 1 | 0.66 | 1.33 | 0.87 | 0.96 | 0.45 | 0.58 | 0.968 |
| shot_noise | 3 | 1.69 | 2.87 | 0.80 | 0.97 | 0.54 | 0.47 | 0.910 |
| shot_noise | 5 | 2.14 | 3.57 | 0.75 | 0.97 | 0.56 | 0.39 | 0.881 |
| snow | 1 | 0.25 | 0.97 | 0.32 | 0.73 | 0.25 | 0.31 | 0.996 |
| snow | 3 | 0.40 | 1.50 | 0.36 | 0.75 | 0.27 | 0.19 | 0.975 |
| snow | 5 | 0.73 | 1.93 | 0.55 | 0.90 | 0.36 | 0.27 | 0.951 |
| zoom_blur | 1 | 0.49 | 1.25 | 0.88 | 0.93 | 0.36 | 0.45 | 0.969 |
| zoom_blur | 3 | 0.82 | 1.83 | 0.88 | 0.94 | 0.41 | 0.40 | 0.941 |
| zoom_blur | 5 | 1.25 | 2.57 | 0.88 | 0.96 | 0.45 | 0.34 | 0.922 |

## 5. Density per split
layer `penult`

| split | sparse frac (ref q95) | median log-radius shift |
|---|---|---|
| corrupt__brightness__s1 | 0.171 | 0.053 |
| corrupt__brightness__s3 | 0.194 | 0.074 |
| corrupt__brightness__s5 | 0.265 | 0.155 |
| corrupt__contrast__s1 | 0.176 | 0.042 |
| corrupt__contrast__s3 | 0.331 | 0.277 |
| corrupt__contrast__s5 | 0.873 | 1.019 |
| corrupt__defocus_blur__s1 | 0.168 | 0.051 |
| corrupt__defocus_blur__s3 | 0.289 | 0.230 |
| corrupt__defocus_blur__s5 | 0.614 | 0.721 |
| corrupt__elastic_transform__s1 | 0.286 | 0.219 |
| corrupt__elastic_transform__s3 | 0.350 | 0.321 |
| corrupt__elastic_transform__s5 | 0.529 | 0.616 |
| corrupt__fog__s1 | 0.168 | 0.051 |
| corrupt__fog__s3 | 0.232 | 0.157 |
| corrupt__fog__s5 | 0.505 | 0.564 |
| corrupt__frost__s1 | 0.271 | 0.175 |
| corrupt__frost__s3 | 0.478 | 0.518 |
| corrupt__frost__s5 | 0.594 | 0.715 |
| corrupt__gaussian_noise__s1 | 0.427 | 0.441 |
| corrupt__gaussian_noise__s3 | 0.621 | 0.760 |
| corrupt__gaussian_noise__s5 | 0.650 | 0.787 |
| corrupt__glass_blur__s1 | 0.652 | 0.820 |
| corrupt__glass_blur__s3 | 0.632 | 0.783 |
| corrupt__glass_blur__s5 | 0.701 | 0.874 |
| corrupt__impulse_noise__s1 | 0.406 | 0.396 |
| corrupt__impulse_noise__s3 | 0.694 | 0.852 |
| corrupt__impulse_noise__s5 | 0.919 | 1.053 |
| corrupt__jpeg_compression__s1 | 0.312 | 0.263 |
| corrupt__jpeg_compression__s3 | 0.430 | 0.436 |
| corrupt__jpeg_compression__s5 | 0.505 | 0.564 |
| corrupt__motion_blur__s1 | 0.268 | 0.203 |
| corrupt__motion_blur__s3 | 0.500 | 0.556 |
| corrupt__motion_blur__s5 | 0.558 | 0.660 |
| corrupt__pixelate__s1 | 0.233 | 0.126 |
| corrupt__pixelate__s3 | 0.395 | 0.334 |
| corrupt__pixelate__s5 | 0.637 | 0.808 |
| corrupt__shot_noise__s1 | 0.343 | 0.288 |
| corrupt__shot_noise__s3 | 0.596 | 0.701 |
| corrupt__shot_noise__s5 | 0.671 | 0.811 |
| corrupt__snow__s1 | 0.285 | 0.230 |
| corrupt__snow__s3 | 0.398 | 0.399 |
| corrupt__snow__s5 | 0.495 | 0.549 |
| corrupt__zoom_blur__s1 | 0.334 | 0.292 |
| corrupt__zoom_blur__s3 | 0.471 | 0.493 |
| corrupt__zoom_blur__s5 | 0.604 | 0.710 |
| ood__cifar100 | 0.783 | 0.909 |
| ood__svhn | 0.853 | 0.926 |
| panel | 0.062 | 0.052 |
| test | 0.163 | 0.040 |

## 6. Adjacency
layer `penult`


closest class pairs (sep = center distance / RMS radius): (3,5) 4.16, (2,3) 4.51, (2,5) 4.66, (5,7) 4.78, (0,2) 4.83, (3,4) 4.86

valley ratio mean 4.25, min 2.81, pairs with a valley 1.00

nearest-center confusions (true → nearest): 3→5 0.05, 5→3 0.05, 0→8 0.03, 9→1 0.03, 8→0 0.02, 4→3 0.02

single-linkage merge order (first 5): [3, 5]@4.16 ; [2, 3, 5]@4.51 ; [2, 3, 5, 7]@4.78 ; [0, 2, 3, 5, 7]@4.83 ; [0, 2, 3, 4, 5, 7]@4.86

## 7. Type-b margin (margin_typeb)
_margin_typeb not run_
