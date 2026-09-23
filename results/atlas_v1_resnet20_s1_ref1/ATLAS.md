# ATLAS — atlas_v1_resnet20_s1_ref1
built 2026-09-23 02:14:36 · source **real** · arch `cifar10_resnet20` · weights `file:/workspace/models/resnet20_s1_chenyaofo.pt sha256:d5442d0eadd72592` · layers 11

## 1. Per-layer invariants
| layer | D | PR | eff.rank | dim95 | TwoNN ID | sep ratio | NC1 | ETF dev | hub skew | test sparse | NC acc(test) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | 16 | 3.0 | 3.7 | 4 | 6.5 | 0.19 | 191.40 | 0.636 | 0.15 | 0.045 | 0.247 |
| layer1.0 | 16 | 3.3 | 4.6 | 5 | 8.4 | 0.21 | 137.17 | 0.639 | 0.49 | 0.043 | 0.281 |
| layer1.1 | 16 | 4.1 | 5.7 | 7 | 9.3 | 0.23 | 58.40 | 0.535 | 0.63 | 0.047 | 0.317 |
| layer1.2 | 16 | 4.8 | 6.8 | 8 | 9.5 | 0.31 | 65.83 | 0.420 | 0.78 | 0.046 | 0.354 |
| layer2.0 | 32 | 6.1 | 9.3 | 12 | 11.4 | 0.35 | 23.16 | 0.504 | 0.95 | 0.046 | 0.419 |
| layer2.1 | 32 | 7.7 | 11.4 | 14 | 12.9 | 0.41 | 15.34 | 0.386 | 1.22 | 0.044 | 0.471 |
| layer2.2 | 32 | 8.6 | 12.9 | 17 | 14.1 | 0.43 | 8.62 | 0.348 | 1.37 | 0.043 | 0.524 |
| layer3.0 | 64 | 10.3 | 20.0 | 38 | 18.2 | 0.57 | 2.29 | 0.321 | 1.57 | 0.055 | 0.647 |
| layer3.1 | 64 | 11.2 | 21.9 | 41 | 19.5 | 0.88 | 0.77 | 0.288 | 1.73 | 0.051 | 0.810 |
| layer3.2 | 64 | 8.6 | 9.7 | 9 | 9.7 | 3.08 | 0.16 | 0.099 | 0.75 | 0.135 | 0.922 |
| penult | 64 | 8.6 | 9.7 | 9 | 9.7 | 3.08 | 0.16 | 0.099 | 0.75 | 0.135 | 0.922 |

## 2. Decodability profile (which factor lives where)
categorical = accuracy minus majority-class rate; continuous = CV R²

| factor | kind | stem | layer1.0 | layer1.1 | layer1.2 | layer2.0 | layer2.1 | layer2.2 | layer3.0 | layer3.1 | layer3.2 | penult | commit | peak | washout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| luminance_mean | continuous | 1.00 | 0.99 | 0.97 | 0.97 | 0.95 | 0.93 | 0.91 | 0.76 | 0.55 | 0.29 | 0.29 | stem | stem | 0.70 |
| contrast_rms | continuous | 0.67 | 0.60 | 0.60 | 0.56 | 0.67 | 0.66 | 0.65 | 0.60 | 0.51 | 0.38 | 0.38 | stem | stem | 0.29 |
| highfreq_ratio | continuous | 0.28 | 0.52 | 0.53 | 0.51 | 0.57 | 0.58 | 0.58 | 0.56 | 0.52 | 0.42 | 0.42 | layer1.1 | layer2.2 | 0.16 |
| spectral_slope | continuous | 0.22 | 0.47 | 0.52 | 0.58 | 0.66 | 0.67 | 0.68 | 0.67 | 0.63 | 0.53 | 0.53 | layer2.0 | layer2.2 | 0.15 |
| spectral_anisotropy | continuous | 0.19 | 0.26 | 0.24 | 0.29 | 0.32 | 0.30 | 0.30 | 0.31 | 0.30 | 0.28 | 0.28 | layer2.0 | layer2.0 | 0.04 |
| noise_sigma | continuous | 0.69 | 0.92 | 0.93 | 0.91 | 0.96 | 0.94 | 0.93 | 0.88 | 0.81 | 0.67 | 0.67 | layer1.0 | layer2.0 | 0.29 |
| saturation_mean | continuous | 0.73 | 0.78 | 0.71 | 0.63 | 0.74 | 0.68 | 0.64 | 0.53 | 0.39 | 0.25 | 0.25 | stem | layer1.0 | 0.53 |
| hue_cos | continuous | 0.77 | 0.83 | 0.81 | 0.81 | 0.82 | 0.81 | 0.80 | 0.72 | 0.65 | 0.54 | 0.54 | stem | layer1.0 | 0.29 |
| hue_sin | continuous | 0.78 | 0.79 | 0.75 | 0.75 | 0.77 | 0.75 | 0.73 | 0.70 | 0.61 | 0.50 | 0.50 | stem | layer1.0 | 0.29 |
| colorfulness | continuous | 0.78 | 0.86 | 0.86 | 0.81 | 0.80 | 0.77 | 0.74 | 0.61 | 0.45 | 0.27 | 0.27 | stem | layer1.0 | 0.59 |
| edge_density | continuous | 0.72 | 0.81 | 0.86 | 0.83 | 0.89 | 0.87 | 0.85 | 0.81 | 0.71 | 0.58 | 0.58 | layer1.0 | layer2.0 | 0.31 |
| orientation_entropy | continuous | 0.22 | 0.32 | 0.34 | 0.41 | 0.48 | 0.46 | 0.47 | 0.46 | 0.43 | 0.39 | 0.39 | layer2.0 | layer2.0 | 0.09 |
| blockiness | continuous | 0.00 | 0.01 | 0.02 | 0.05 | 0.10 | 0.10 | 0.09 | 0.12 | 0.10 | 0.05 | 0.05 | layer3.0 | layer3.0 | 0.06 |
| class | categorical | 0.26 | 0.28 | 0.30 | 0.36 | 0.46 | 0.50 | 0.57 | 0.67 | 0.77 | 0.81 | 0.81 | layer3.1 | layer3.2 | 0.00 |
| coarse_animal_vehicle | categorical | 0.25 | 0.24 | 0.23 | 0.26 | 0.30 | 0.30 | 0.32 | 0.35 | 0.37 | 0.38 | 0.38 | layer3.0 | layer3.2 | 0.00 |
| corruption_family | categorical | 0.12 | 0.22 | 0.31 | 0.35 | 0.41 | 0.39 | 0.38 | 0.37 | 0.33 | 0.25 | 0.25 | layer2.0 | layer2.0 | 0.16 |
| corruption_type | categorical | 0.16 | 0.28 | 0.37 | 0.47 | 0.56 | 0.55 | 0.54 | 0.51 | 0.41 | 0.26 | 0.26 | layer2.0 | layer2.0 | 0.30 |
| severity | continuous | 0.06 | 0.07 | 0.10 | 0.13 | 0.26 | 0.25 | 0.26 | 0.32 | 0.28 | 0.23 | 0.23 | layer3.0 | layer3.0 | 0.09 |

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
| brightness | 1 | 0.03 | 0.15 | 0.76 | 0.95 | 0.15 | 0.58 | 1.000 |
| brightness | 3 | 0.08 | 0.43 | 0.67 | 0.93 | 0.17 | 0.39 | 0.985 |
| brightness | 5 | 0.20 | 0.82 | 0.58 | 0.93 | 0.22 | 0.19 | 0.947 |
| contrast | 1 | 0.12 | 0.34 | 0.66 | 0.96 | 0.32 | 0.68 | 0.999 |
| contrast | 3 | 0.44 | 1.10 | 0.66 | 0.95 | 0.39 | 0.44 | 0.955 |
| contrast | 5 | 1.26 | 2.48 | 0.63 | 0.88 | 0.50 | 0.27 | 0.839 |
| defocus_blur | 1 | 0.05 | 0.18 | 0.82 | 0.94 | 0.24 | 0.69 | 1.001 |
| defocus_blur | 3 | 0.39 | 0.95 | 0.95 | 0.96 | 0.38 | 0.49 | 0.969 |
| defocus_blur | 5 | 0.85 | 1.89 | 0.94 | 0.96 | 0.43 | 0.24 | 0.893 |
| elastic_transform | 1 | 0.20 | 0.89 | 0.94 | 0.96 | 0.20 | 0.19 | 0.959 |
| elastic_transform | 3 | 0.40 | 1.12 | 0.96 | 0.97 | 0.33 | 0.34 | 0.949 |
| elastic_transform | 5 | 0.47 | 1.44 | 0.74 | 0.98 | 0.30 | 0.14 | 0.912 |
| fog | 1 | 0.08 | 0.25 | 0.71 | 0.97 | 0.30 | 0.69 | 1.002 |
| fog | 3 | 0.28 | 0.73 | 0.73 | 0.97 | 0.36 | 0.58 | 0.983 |
| fog | 5 | 0.69 | 1.53 | 0.72 | 0.97 | 0.44 | 0.37 | 0.916 |
| frost | 1 | 0.27 | 0.77 | 0.61 | 0.96 | 0.31 | 0.36 | 0.956 |
| frost | 3 | 0.67 | 1.51 | 0.74 | 0.95 | 0.41 | 0.29 | 0.896 |
| frost | 5 | 0.96 | 1.91 | 0.82 | 0.95 | 0.47 | 0.32 | 0.879 |
| gaussian_noise | 1 | 0.80 | 1.46 | 0.91 | 0.97 | 0.52 | 0.51 | 0.909 |
| gaussian_noise | 3 | 1.52 | 2.52 | 0.90 | 0.92 | 0.58 | 0.37 | 0.811 |
| gaussian_noise | 5 | 1.63 | 2.71 | 0.86 | 0.87 | 0.58 | 0.34 | 0.762 |
| glass_blur | 1 | 0.85 | 1.99 | 0.87 | 0.96 | 0.38 | 0.17 | 0.855 |
| glass_blur | 3 | 0.76 | 1.86 | 0.86 | 0.97 | 0.36 | 0.16 | 0.864 |
| glass_blur | 5 | 0.87 | 2.10 | 0.88 | 0.96 | 0.37 | 0.14 | 0.842 |
| impulse_noise | 1 | 0.48 | 1.16 | 0.63 | 0.98 | 0.36 | 0.35 | 0.935 |
| impulse_noise | 3 | 0.59 | 1.80 | 0.61 | 0.96 | 0.30 | 0.07 | 0.850 |
| impulse_noise | 5 | 0.95 | 2.42 | 0.68 | 0.72 | 0.38 | 0.10 | 0.661 |
| jpeg_compression | 1 | 0.29 | 1.00 | 0.35 | 0.97 | 0.27 | 0.19 | 0.933 |
| jpeg_compression | 3 | 0.34 | 1.29 | 0.20 | 0.96 | 0.25 | 0.07 | 0.900 |
| jpeg_compression | 5 | 0.41 | 1.47 | 0.33 | 0.97 | 0.26 | 0.06 | 0.891 |
| motion_blur | 1 | 0.29 | 0.84 | 0.85 | 0.98 | 0.32 | 0.47 | 0.974 |
| motion_blur | 3 | 0.58 | 1.47 | 0.90 | 0.97 | 0.36 | 0.25 | 0.911 |
| motion_blur | 5 | 0.66 | 1.68 | 0.90 | 0.97 | 0.37 | 0.19 | 0.891 |
| pixelate | 1 | 0.25 | 0.64 | 0.64 | 0.99 | 0.36 | 0.56 | 0.964 |
| pixelate | 3 | 0.67 | 1.26 | 0.58 | 0.98 | 0.48 | 0.50 | 0.898 |
| pixelate | 5 | 1.28 | 2.25 | 0.45 | 0.96 | 0.53 | 0.36 | 0.830 |
| shot_noise | 1 | 0.58 | 1.11 | 0.88 | 0.97 | 0.49 | 0.58 | 0.938 |
| shot_noise | 3 | 1.28 | 2.18 | 0.91 | 0.94 | 0.56 | 0.39 | 0.837 |
| shot_noise | 5 | 1.51 | 2.57 | 0.86 | 0.88 | 0.56 | 0.33 | 0.767 |
| snow | 1 | 0.19 | 0.79 | 0.42 | 0.96 | 0.20 | 0.15 | 0.959 |
| snow | 3 | 0.25 | 1.15 | 0.21 | 0.95 | 0.22 | 0.05 | 0.926 |
| snow | 5 | 0.47 | 1.43 | 0.42 | 0.96 | 0.32 | 0.16 | 0.903 |
| zoom_blur | 1 | 0.39 | 1.01 | 0.96 | 0.97 | 0.36 | 0.37 | 0.947 |
| zoom_blur | 3 | 0.62 | 1.42 | 0.96 | 0.97 | 0.41 | 0.32 | 0.923 |
| zoom_blur | 5 | 0.85 | 1.86 | 0.94 | 0.97 | 0.43 | 0.26 | 0.902 |

## 5. Density per split
layer `penult`

| split | sparse frac (ref q95) | median log-radius shift |
|---|---|---|
| corrupt__brightness__s1 | 0.129 | 0.027 |
| corrupt__brightness__s3 | 0.145 | 0.044 |
| corrupt__brightness__s5 | 0.194 | 0.101 |
| corrupt__contrast__s1 | 0.140 | 0.025 |
| corrupt__contrast__s3 | 0.269 | 0.191 |
| corrupt__contrast__s5 | 0.779 | 0.549 |
| corrupt__defocus_blur__s1 | 0.138 | 0.027 |
| corrupt__defocus_blur__s3 | 0.250 | 0.186 |
| corrupt__defocus_blur__s5 | 0.533 | 0.412 |
| corrupt__elastic_transform__s1 | 0.214 | 0.133 |
| corrupt__elastic_transform__s3 | 0.274 | 0.210 |
| corrupt__elastic_transform__s5 | 0.393 | 0.317 |
| corrupt__fog__s1 | 0.131 | 0.030 |
| corrupt__fog__s3 | 0.185 | 0.096 |
| corrupt__fog__s5 | 0.401 | 0.323 |
| corrupt__frost__s1 | 0.199 | 0.117 |
| corrupt__frost__s3 | 0.400 | 0.322 |
| corrupt__frost__s5 | 0.493 | 0.391 |
| corrupt__gaussian_noise__s1 | 0.363 | 0.285 |
| corrupt__gaussian_noise__s3 | 0.455 | 0.372 |
| corrupt__gaussian_noise__s5 | 0.435 | 0.365 |
| corrupt__glass_blur__s1 | 0.529 | 0.414 |
| corrupt__glass_blur__s3 | 0.502 | 0.395 |
| corrupt__glass_blur__s5 | 0.537 | 0.419 |
| corrupt__impulse_noise__s1 | 0.304 | 0.232 |
| corrupt__impulse_noise__s3 | 0.522 | 0.407 |
| corrupt__impulse_noise__s5 | 0.770 | 0.501 |
| corrupt__jpeg_compression__s1 | 0.234 | 0.152 |
| corrupt__jpeg_compression__s3 | 0.323 | 0.255 |
| corrupt__jpeg_compression__s5 | 0.354 | 0.297 |
| corrupt__motion_blur__s1 | 0.213 | 0.146 |
| corrupt__motion_blur__s3 | 0.353 | 0.296 |
| corrupt__motion_blur__s5 | 0.404 | 0.331 |
| corrupt__pixelate__s1 | 0.193 | 0.081 |
| corrupt__pixelate__s3 | 0.366 | 0.241 |
| corrupt__pixelate__s5 | 0.534 | 0.428 |
| corrupt__shot_noise__s1 | 0.292 | 0.188 |
| corrupt__shot_noise__s3 | 0.502 | 0.395 |
| corrupt__shot_noise__s5 | 0.487 | 0.390 |
| corrupt__snow__s1 | 0.225 | 0.142 |
| corrupt__snow__s3 | 0.307 | 0.231 |
| corrupt__snow__s5 | 0.375 | 0.297 |
| corrupt__zoom_blur__s1 | 0.272 | 0.198 |
| corrupt__zoom_blur__s3 | 0.380 | 0.312 |
| corrupt__zoom_blur__s5 | 0.504 | 0.397 |
| ood__cifar100 | 0.575 | 0.428 |
| ood__svhn | 0.761 | 0.548 |
| panel | 0.031 | 0.029 |
| test | 0.135 | 0.019 |

## 6. Adjacency
layer `penult`


closest class pairs (sep = center distance / RMS radius): (3,5) 2.52, (2,4) 2.83, (2,3) 2.85, (3,4) 2.96, (5,7) 2.97, (0,2) 2.98

valley ratio mean 2.33, min 1.48, pairs with a valley 1.00

nearest-center confusions (true → nearest): 5→3 0.07, 3→5 0.07, 9→1 0.04, 0→8 0.03, 6→2 0.03, 8→0 0.03

single-linkage merge order (first 5): [3, 5]@2.52 ; [2, 4]@2.83 ; [2, 3, 4, 5]@2.85 ; [2, 3, 4, 5, 7]@2.97 ; [0, 2, 3, 4, 5, 7]@2.98
