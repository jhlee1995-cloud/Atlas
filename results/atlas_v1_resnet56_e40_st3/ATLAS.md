# ATLAS — atlas_v1_resnet56_e40_st3
built 2026-09-23 14:01:52 · source **real** · arch `cifar10_resnet56` · weights `file:/workspace/models/resnet56_s11_e40_chenyaofo.pt sha256:a5eceba4d28b9e0d` · layers 11

## 1. Per-layer invariants
| layer | D | PR | eff.rank | dim95 | TwoNN ID | sep ratio | NC1 | ETF dev | hub skew | test sparse | NC acc(test) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | 16 | 3.0 | 3.9 | 5 | 7.2 | 0.17 | 147.95 | 0.652 | 0.26 | 0.053 | 0.252 |
| layer1.0 | 16 | 3.6 | 5.0 | 6 | 8.4 | 0.19 | 99.83 | 0.604 | 0.45 | 0.056 | 0.289 |
| layer1.5 | 16 | 4.3 | 6.3 | 8 | 9.5 | 0.28 | 103.29 | 0.524 | 0.68 | 0.053 | 0.349 |
| layer1.8 | 16 | 5.0 | 7.0 | 8 | 9.7 | 0.31 | 72.80 | 0.485 | 0.73 | 0.051 | 0.384 |
| layer2.0 | 32 | 5.3 | 7.9 | 10 | 10.5 | 0.36 | 29.12 | 0.466 | 0.83 | 0.048 | 0.422 |
| layer2.5 | 32 | 6.1 | 9.6 | 13 | 12.2 | 0.43 | 11.76 | 0.428 | 1.10 | 0.047 | 0.497 |
| layer2.8 | 32 | 7.9 | 11.6 | 15 | 12.6 | 0.47 | 8.69 | 0.360 | 1.24 | 0.043 | 0.539 |
| layer3.0 | 64 | 8.4 | 14.0 | 22 | 14.8 | 0.54 | 3.89 | 0.325 | 1.40 | 0.052 | 0.593 |
| layer3.5 | 64 | 13.8 | 24.2 | 41 | 19.5 | 0.89 | 0.75 | 0.221 | 1.81 | 0.051 | 0.822 |
| layer3.8 | 64 | 8.7 | 10.4 | 9 | 10.9 | 2.85 | 0.18 | 0.106 | 0.96 | 0.102 | 0.916 |
| penult | 64 | 8.7 | 10.4 | 9 | 10.9 | 2.85 | 0.18 | 0.106 | 0.96 | 0.102 | 0.916 |

## 2. Decodability profile (which factor lives where)
categorical = accuracy minus majority-class rate; continuous = CV R²

| factor | kind | stem | layer1.0 | layer1.5 | layer1.8 | layer2.0 | layer2.5 | layer2.8 | layer3.0 | layer3.5 | layer3.8 | penult | commit | peak | washout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| luminance_mean | continuous | 1.00 | 0.99 | 0.95 | 0.92 | 0.93 | 0.90 | 0.87 | 0.81 | 0.67 | 0.46 | 0.46 | stem | stem | 0.54 |
| contrast_rms | continuous | 0.69 | 0.61 | 0.56 | 0.53 | 0.61 | 0.61 | 0.60 | 0.62 | 0.51 | 0.38 | 0.38 | stem | stem | 0.31 |
| highfreq_ratio | continuous | 0.27 | 0.50 | 0.57 | 0.57 | 0.60 | 0.60 | 0.60 | 0.61 | 0.55 | 0.48 | 0.48 | layer1.5 | layer3.0 | 0.13 |
| spectral_slope | continuous | 0.15 | 0.40 | 0.58 | 0.57 | 0.62 | 0.62 | 0.61 | 0.63 | 0.58 | 0.50 | 0.50 | layer1.5 | layer3.0 | 0.14 |
| spectral_anisotropy | continuous | 0.27 | 0.22 | 0.27 | 0.31 | 0.32 | 0.36 | 0.34 | 0.35 | 0.33 | 0.30 | 0.30 | layer2.0 | layer2.5 | 0.06 |
| noise_sigma | continuous | 0.76 | 0.88 | 0.89 | 0.92 | 0.94 | 0.92 | 0.91 | 0.90 | 0.85 | 0.74 | 0.74 | layer1.0 | layer2.0 | 0.20 |
| saturation_mean | continuous | 0.86 | 0.86 | 0.73 | 0.63 | 0.68 | 0.63 | 0.61 | 0.54 | 0.39 | 0.27 | 0.27 | stem | stem | 0.59 |
| hue_cos | continuous | 0.81 | 0.85 | 0.81 | 0.81 | 0.78 | 0.76 | 0.74 | 0.72 | 0.59 | 0.48 | 0.48 | stem | layer1.0 | 0.36 |
| hue_sin | continuous | 0.79 | 0.80 | 0.75 | 0.74 | 0.72 | 0.70 | 0.70 | 0.68 | 0.61 | 0.51 | 0.51 | stem | layer1.0 | 0.29 |
| colorfulness | continuous | 0.77 | 0.89 | 0.83 | 0.80 | 0.78 | 0.73 | 0.71 | 0.60 | 0.43 | 0.32 | 0.32 | layer1.0 | layer1.0 | 0.57 |
| edge_density | continuous | 0.86 | 0.86 | 0.77 | 0.80 | 0.87 | 0.85 | 0.83 | 0.84 | 0.77 | 0.66 | 0.66 | stem | layer2.0 | 0.22 |
| orientation_entropy | continuous | 0.28 | 0.26 | 0.39 | 0.41 | 0.46 | 0.48 | 0.45 | 0.50 | 0.45 | 0.44 | 0.44 | layer2.0 | layer3.0 | 0.06 |
| blockiness | continuous | 0.01 | 0.02 | 0.02 | 0.04 | 0.09 | 0.08 | 0.08 | 0.14 | 0.10 | 0.06 | 0.06 | layer3.0 | layer3.0 | 0.08 |
| class | categorical | 0.26 | 0.30 | 0.37 | 0.39 | 0.49 | 0.55 | 0.57 | 0.65 | 0.77 | 0.81 | 0.81 | layer3.5 | layer3.8 | 0.00 |
| coarse_animal_vehicle | categorical | 0.23 | 0.27 | 0.28 | 0.28 | 0.31 | 0.33 | 0.34 | 0.35 | 0.38 | 0.39 | 0.39 | layer3.0 | layer3.8 | 0.00 |
| corruption_family | categorical | 0.11 | 0.20 | 0.32 | 0.32 | 0.35 | 0.34 | 0.33 | 0.35 | 0.28 | 0.23 | 0.23 | layer1.5 | layer2.0 | 0.12 |
| corruption_type | categorical | 0.16 | 0.24 | 0.42 | 0.47 | 0.51 | 0.50 | 0.49 | 0.48 | 0.39 | 0.29 | 0.29 | layer1.8 | layer2.0 | 0.23 |
| severity | continuous | 0.05 | 0.08 | 0.10 | 0.14 | 0.23 | 0.24 | 0.22 | 0.30 | 0.29 | 0.24 | 0.24 | layer3.0 | layer3.0 | 0.06 |

## 3. Layer flow (CKA)
| pair | CKA |
|---|---|
| stem->layer1.0 | 0.880 |
| layer1.0->layer1.5 | 0.769 |
| layer1.5->layer1.8 | 0.933 |
| layer1.8->layer2.0 | 0.958 |
| layer2.0->layer2.5 | 0.950 |
| layer2.5->layer2.8 | 0.961 |
| layer2.8->layer3.0 | 0.940 |
| layer3.0->layer3.5 | 0.779 |
| layer3.5->layer3.8 | 0.739 |
| layer3.8->penult | 1.000 |

biggest reorganization: `layer3.5->layer3.8` (drop 0.261)

## 4. Sensitivity field (paired corruption displacement)
layer `penult`; magnitude in within-class-radius units

| corruption | sev | magnitude | per-sample | top5 frac | class-sub frac | coherence | class consistency | norm ratio |
|---|---|---|---|---|---|---|---|---|
| brightness | 1 | 0.03 | 0.13 | 0.74 | 0.82 | 0.21 | 0.58 | 1.000 |
| brightness | 3 | 0.09 | 0.37 | 0.75 | 0.83 | 0.23 | 0.47 | 0.994 |
| brightness | 5 | 0.19 | 0.72 | 0.72 | 0.84 | 0.26 | 0.30 | 0.969 |
| contrast | 1 | 0.13 | 0.33 | 0.81 | 0.90 | 0.39 | 0.71 | 1.002 |
| contrast | 3 | 0.49 | 1.10 | 0.73 | 0.88 | 0.44 | 0.46 | 0.962 |
| contrast | 5 | 1.32 | 2.40 | 0.64 | 0.87 | 0.54 | 0.35 | 0.881 |
| defocus_blur | 1 | 0.06 | 0.16 | 0.75 | 0.86 | 0.35 | 0.66 | 1.000 |
| defocus_blur | 3 | 0.40 | 0.82 | 0.84 | 0.89 | 0.44 | 0.56 | 0.971 |
| defocus_blur | 5 | 0.91 | 1.75 | 0.84 | 0.89 | 0.49 | 0.40 | 0.899 |
| elastic_transform | 1 | 0.20 | 0.81 | 0.71 | 0.76 | 0.24 | 0.24 | 0.957 |
| elastic_transform | 3 | 0.39 | 1.01 | 0.82 | 0.87 | 0.36 | 0.38 | 0.949 |
| elastic_transform | 5 | 0.52 | 1.40 | 0.76 | 0.93 | 0.35 | 0.22 | 0.925 |
| fog | 1 | 0.08 | 0.23 | 0.81 | 0.88 | 0.36 | 0.72 | 1.003 |
| fog | 3 | 0.30 | 0.70 | 0.79 | 0.89 | 0.42 | 0.62 | 0.985 |
| fog | 5 | 0.82 | 1.54 | 0.75 | 0.92 | 0.52 | 0.48 | 0.932 |
| frost | 1 | 0.28 | 0.68 | 0.78 | 0.94 | 0.37 | 0.54 | 0.974 |
| frost | 3 | 0.68 | 1.36 | 0.76 | 0.94 | 0.46 | 0.45 | 0.940 |
| frost | 5 | 0.95 | 1.72 | 0.79 | 0.95 | 0.52 | 0.47 | 0.925 |
| gaussian_noise | 1 | 0.66 | 1.19 | 0.86 | 0.94 | 0.53 | 0.57 | 0.935 |
| gaussian_noise | 3 | 1.31 | 2.11 | 0.85 | 0.94 | 0.60 | 0.50 | 0.903 |
| gaussian_noise | 5 | 1.50 | 2.36 | 0.85 | 0.93 | 0.61 | 0.48 | 0.903 |
| glass_blur | 1 | 1.00 | 1.95 | 0.84 | 0.95 | 0.47 | 0.35 | 0.898 |
| glass_blur | 3 | 0.87 | 1.80 | 0.82 | 0.95 | 0.45 | 0.31 | 0.901 |
| glass_blur | 5 | 1.01 | 2.04 | 0.82 | 0.95 | 0.46 | 0.29 | 0.888 |
| impulse_noise | 1 | 0.28 | 0.88 | 0.45 | 0.71 | 0.29 | 0.27 | 0.977 |
| impulse_noise | 3 | 0.65 | 1.56 | 0.56 | 0.78 | 0.41 | 0.25 | 0.910 |
| impulse_noise | 5 | 1.39 | 2.39 | 0.66 | 0.85 | 0.57 | 0.38 | 0.887 |
| jpeg_compression | 1 | 0.29 | 0.88 | 0.65 | 0.90 | 0.31 | 0.32 | 0.957 |
| jpeg_compression | 3 | 0.37 | 1.17 | 0.54 | 0.88 | 0.30 | 0.17 | 0.935 |
| jpeg_compression | 5 | 0.44 | 1.34 | 0.53 | 0.87 | 0.31 | 0.15 | 0.927 |
| motion_blur | 1 | 0.31 | 0.77 | 0.79 | 0.87 | 0.37 | 0.48 | 0.964 |
| motion_blur | 3 | 0.65 | 1.45 | 0.82 | 0.86 | 0.42 | 0.34 | 0.904 |
| motion_blur | 5 | 0.78 | 1.70 | 0.82 | 0.86 | 0.43 | 0.29 | 0.881 |
| pixelate | 1 | 0.26 | 0.61 | 0.69 | 0.94 | 0.40 | 0.62 | 0.979 |
| pixelate | 3 | 0.69 | 1.24 | 0.59 | 0.94 | 0.51 | 0.55 | 0.937 |
| pixelate | 5 | 1.37 | 2.25 | 0.37 | 0.88 | 0.58 | 0.44 | 0.903 |
| shot_noise | 1 | 0.50 | 0.91 | 0.84 | 0.95 | 0.52 | 0.64 | 0.952 |
| shot_noise | 3 | 1.16 | 1.85 | 0.85 | 0.94 | 0.60 | 0.55 | 0.912 |
| shot_noise | 5 | 1.46 | 2.27 | 0.83 | 0.92 | 0.62 | 0.50 | 0.906 |
| snow | 1 | 0.23 | 0.75 | 0.36 | 0.83 | 0.28 | 0.34 | 0.990 |
| snow | 3 | 0.34 | 1.12 | 0.42 | 0.87 | 0.31 | 0.21 | 0.957 |
| snow | 5 | 0.53 | 1.37 | 0.58 | 0.93 | 0.38 | 0.30 | 0.940 |
| zoom_blur | 1 | 0.38 | 0.93 | 0.84 | 0.84 | 0.39 | 0.40 | 0.940 |
| zoom_blur | 3 | 0.61 | 1.29 | 0.84 | 0.87 | 0.44 | 0.40 | 0.915 |
| zoom_blur | 5 | 0.85 | 1.71 | 0.83 | 0.88 | 0.47 | 0.36 | 0.883 |

## 5. Density per split
layer `penult`

| split | sparse frac (ref q95) | median log-radius shift |
|---|---|---|
| corrupt__brightness__s1 | 0.106 | 0.020 |
| corrupt__brightness__s3 | 0.111 | 0.044 |
| corrupt__brightness__s5 | 0.159 | 0.110 |
| corrupt__contrast__s1 | 0.098 | 0.035 |
| corrupt__contrast__s3 | 0.233 | 0.215 |
| corrupt__contrast__s5 | 0.709 | 0.486 |
| corrupt__defocus_blur__s1 | 0.098 | 0.025 |
| corrupt__defocus_blur__s3 | 0.152 | 0.127 |
| corrupt__defocus_blur__s5 | 0.319 | 0.316 |
| corrupt__elastic_transform__s1 | 0.154 | 0.125 |
| corrupt__elastic_transform__s3 | 0.186 | 0.179 |
| corrupt__elastic_transform__s5 | 0.335 | 0.314 |
| corrupt__fog__s1 | 0.103 | 0.031 |
| corrupt__fog__s3 | 0.140 | 0.104 |
| corrupt__fog__s5 | 0.369 | 0.328 |
| corrupt__frost__s1 | 0.150 | 0.105 |
| corrupt__frost__s3 | 0.330 | 0.294 |
| corrupt__frost__s5 | 0.428 | 0.368 |
| corrupt__gaussian_noise__s1 | 0.247 | 0.234 |
| corrupt__gaussian_noise__s3 | 0.518 | 0.412 |
| corrupt__gaussian_noise__s5 | 0.579 | 0.440 |
| corrupt__glass_blur__s1 | 0.470 | 0.387 |
| corrupt__glass_blur__s3 | 0.429 | 0.369 |
| corrupt__glass_blur__s5 | 0.500 | 0.403 |
| corrupt__impulse_noise__s1 | 0.224 | 0.181 |
| corrupt__impulse_noise__s3 | 0.472 | 0.389 |
| corrupt__impulse_noise__s5 | 0.809 | 0.548 |
| corrupt__jpeg_compression__s1 | 0.181 | 0.139 |
| corrupt__jpeg_compression__s3 | 0.237 | 0.232 |
| corrupt__jpeg_compression__s5 | 0.305 | 0.280 |
| corrupt__motion_blur__s1 | 0.156 | 0.135 |
| corrupt__motion_blur__s3 | 0.276 | 0.283 |
| corrupt__motion_blur__s5 | 0.344 | 0.327 |
| corrupt__pixelate__s1 | 0.171 | 0.076 |
| corrupt__pixelate__s3 | 0.359 | 0.262 |
| corrupt__pixelate__s5 | 0.665 | 0.537 |
| corrupt__shot_noise__s1 | 0.196 | 0.150 |
| corrupt__shot_noise__s3 | 0.460 | 0.382 |
| corrupt__shot_noise__s5 | 0.588 | 0.444 |
| corrupt__snow__s1 | 0.191 | 0.147 |
| corrupt__snow__s3 | 0.280 | 0.252 |
| corrupt__snow__s5 | 0.339 | 0.309 |
| corrupt__zoom_blur__s1 | 0.180 | 0.168 |
| corrupt__zoom_blur__s3 | 0.257 | 0.255 |
| corrupt__zoom_blur__s5 | 0.346 | 0.329 |
| ood__cifar100 | 0.508 | 0.408 |
| ood__svhn | 0.740 | 0.534 |
| panel | 0.031 | -0.035 |
| test | 0.102 | 0.020 |

## 6. Adjacency
layer `penult`


closest class pairs (sep = center distance / RMS radius): (3,5) 2.09, (2,3) 2.41, (3,4) 2.60, (0,2) 2.60, (2,5) 2.70, (5,7) 2.71

valley ratio mean 2.06, min 1.30, pairs with a valley 1.00

nearest-center confusions (true → nearest): 5→3 0.08, 3→5 0.05, 3→2 0.04, 8→0 0.04, 9→1 0.04, 3→4 0.03

single-linkage merge order (first 5): [3, 5]@2.09 ; [2, 3, 5]@2.41 ; [2, 3, 4, 5]@2.60 ; [0, 2, 3, 4, 5]@2.60 ; [0, 2, 3, 4, 5, 7]@2.71

## 7. Type-b margin (margin_typeb)
_margin_typeb not run_
