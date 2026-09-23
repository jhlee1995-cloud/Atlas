# ATLAS — atlas_v1_resnet56_s0hub_ref1
built 2026-09-23 13:58:31 · source **real** · arch `cifar10_resnet56` · weights `chenyaofo cifar10_resnet56` · layers 11

## 1. Per-layer invariants
| layer | D | PR | eff.rank | dim95 | TwoNN ID | sep ratio | NC1 | ETF dev | hub skew | test sparse | NC acc(test) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | 16 | 2.9 | 3.6 | 4 | 6.4 | 0.19 | 243.96 | 0.648 | 0.16 | 0.048 | 0.246 |
| layer1.0 | 16 | 3.2 | 4.1 | 5 | 7.4 | 0.18 | 435.70 | 0.647 | 0.33 | 0.045 | 0.248 |
| layer1.5 | 16 | 5.1 | 7.0 | 8 | 9.6 | 0.25 | 74.87 | 0.499 | 0.80 | 0.048 | 0.328 |
| layer1.8 | 16 | 5.4 | 7.7 | 9 | 10.1 | 0.33 | 60.54 | 0.437 | 0.89 | 0.048 | 0.370 |
| layer2.0 | 32 | 6.3 | 9.1 | 11 | 11.2 | 0.35 | 22.76 | 0.440 | 0.89 | 0.044 | 0.402 |
| layer2.5 | 32 | 8.7 | 12.5 | 16 | 13.3 | 0.40 | 10.11 | 0.375 | 1.28 | 0.044 | 0.497 |
| layer2.8 | 32 | 9.9 | 14.3 | 18 | 14.2 | 0.44 | 5.84 | 0.339 | 1.48 | 0.043 | 0.556 |
| layer3.0 | 64 | 10.0 | 16.6 | 25 | 15.5 | 0.55 | 3.34 | 0.313 | 1.41 | 0.050 | 0.620 |
| layer3.5 | 64 | 15.8 | 27.2 | 43 | 19.2 | 0.90 | 0.65 | 0.206 | 1.93 | 0.054 | 0.835 |
| layer3.8 | 64 | 9.0 | 9.8 | 9 | 10.6 | 5.38 | 0.05 | 0.054 | 0.97 | 0.172 | 0.944 |
| penult | 64 | 9.0 | 9.8 | 9 | 10.6 | 5.38 | 0.05 | 0.054 | 0.97 | 0.172 | 0.944 |

## 2. Decodability profile (which factor lives where)
categorical = accuracy minus majority-class rate; continuous = CV R²

| factor | kind | stem | layer1.0 | layer1.5 | layer1.8 | layer2.0 | layer2.5 | layer2.8 | layer3.0 | layer3.5 | layer3.8 | penult | commit | peak | washout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| luminance_mean | continuous | 1.00 | 0.99 | 0.96 | 0.95 | 0.93 | 0.87 | 0.84 | 0.81 | 0.65 | 0.24 | 0.24 | stem | stem | 0.76 |
| contrast_rms | continuous | 0.67 | 0.66 | 0.59 | 0.56 | 0.66 | 0.64 | 0.63 | 0.64 | 0.56 | 0.39 | 0.39 | stem | stem | 0.28 |
| highfreq_ratio | continuous | 0.22 | 0.35 | 0.49 | 0.47 | 0.59 | 0.59 | 0.58 | 0.60 | 0.54 | 0.44 | 0.44 | layer2.0 | layer3.0 | 0.15 |
| spectral_slope | continuous | 0.16 | 0.30 | 0.52 | 0.55 | 0.70 | 0.69 | 0.68 | 0.69 | 0.65 | 0.50 | 0.50 | layer2.0 | layer2.0 | 0.19 |
| spectral_anisotropy | continuous | 0.24 | 0.25 | 0.28 | 0.28 | 0.31 | 0.30 | 0.31 | 0.36 | 0.34 | 0.27 | 0.27 | layer3.0 | layer3.0 | 0.08 |
| noise_sigma | continuous | 0.72 | 0.90 | 0.92 | 0.92 | 0.95 | 0.93 | 0.93 | 0.92 | 0.86 | 0.72 | 0.72 | layer1.0 | layer2.0 | 0.22 |
| saturation_mean | continuous | 0.78 | 0.78 | 0.77 | 0.68 | 0.81 | 0.72 | 0.68 | 0.68 | 0.57 | 0.26 | 0.26 | stem | layer2.0 | 0.55 |
| hue_cos | continuous | 0.78 | 0.81 | 0.80 | 0.78 | 0.80 | 0.79 | 0.77 | 0.73 | 0.65 | 0.47 | 0.47 | stem | layer1.0 | 0.34 |
| hue_sin | continuous | 0.81 | 0.84 | 0.80 | 0.78 | 0.75 | 0.74 | 0.73 | 0.73 | 0.67 | 0.47 | 0.47 | stem | layer1.0 | 0.36 |
| colorfulness | continuous | 0.67 | 0.80 | 0.86 | 0.82 | 0.84 | 0.80 | 0.77 | 0.76 | 0.63 | 0.34 | 0.34 | layer1.0 | layer1.5 | 0.52 |
| edge_density | continuous | 0.75 | 0.88 | 0.80 | 0.83 | 0.88 | 0.87 | 0.84 | 0.84 | 0.76 | 0.60 | 0.60 | layer1.0 | layer1.0 | 0.28 |
| orientation_entropy | continuous | 0.25 | 0.28 | 0.38 | 0.43 | 0.48 | 0.46 | 0.49 | 0.51 | 0.46 | 0.41 | 0.41 | layer2.0 | layer3.0 | 0.10 |
| blockiness | continuous | 0.00 | 0.01 | 0.02 | 0.04 | 0.08 | 0.07 | 0.07 | 0.08 | 0.08 | 0.07 | 0.07 | layer2.0 | layer3.0 | 0.02 |
| class | categorical | 0.26 | 0.26 | 0.33 | 0.37 | 0.47 | 0.55 | 0.58 | 0.66 | 0.78 | 0.84 | 0.84 | layer3.5 | layer3.8 | 0.00 |
| coarse_animal_vehicle | categorical | 0.23 | 0.23 | 0.26 | 0.28 | 0.30 | 0.32 | 0.32 | 0.35 | 0.37 | 0.39 | 0.39 | layer3.0 | layer3.8 | 0.00 |
| corruption_family | categorical | 0.11 | 0.15 | 0.30 | 0.35 | 0.40 | 0.39 | 0.38 | 0.39 | 0.33 | 0.23 | 0.23 | layer2.0 | layer2.0 | 0.17 |
| corruption_type | categorical | 0.15 | 0.19 | 0.37 | 0.50 | 0.55 | 0.55 | 0.55 | 0.55 | 0.47 | 0.28 | 0.28 | layer2.0 | layer2.5 | 0.27 |
| severity | continuous | 0.04 | 0.05 | 0.10 | 0.09 | 0.22 | 0.24 | 0.25 | 0.32 | 0.28 | 0.22 | 0.22 | layer3.0 | layer3.0 | 0.10 |

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
| brightness | 1 | 0.03 | 0.17 | 0.71 | 0.85 | 0.15 | 0.39 | 0.999 |
| brightness | 3 | 0.09 | 0.48 | 0.73 | 0.84 | 0.18 | 0.28 | 0.991 |
| brightness | 5 | 0.22 | 0.95 | 0.71 | 0.84 | 0.22 | 0.18 | 0.972 |
| contrast | 1 | 0.13 | 0.40 | 0.68 | 0.89 | 0.32 | 0.59 | 0.994 |
| contrast | 3 | 0.51 | 1.33 | 0.66 | 0.88 | 0.37 | 0.40 | 0.958 |
| contrast | 5 | 1.43 | 3.44 | 0.82 | 0.90 | 0.39 | 0.12 | 0.800 |
| defocus_blur | 1 | 0.06 | 0.21 | 0.60 | 0.81 | 0.28 | 0.59 | 0.997 |
| defocus_blur | 3 | 0.42 | 1.10 | 0.83 | 0.88 | 0.36 | 0.41 | 0.964 |
| defocus_blur | 5 | 1.05 | 2.54 | 0.90 | 0.93 | 0.39 | 0.20 | 0.882 |
| elastic_transform | 1 | 0.23 | 1.09 | 0.73 | 0.82 | 0.22 | 0.20 | 0.976 |
| elastic_transform | 3 | 0.45 | 1.39 | 0.82 | 0.90 | 0.31 | 0.27 | 0.959 |
| elastic_transform | 5 | 0.71 | 2.09 | 0.69 | 0.93 | 0.33 | 0.19 | 0.939 |
| fog | 1 | 0.08 | 0.29 | 0.70 | 0.92 | 0.29 | 0.62 | 0.999 |
| fog | 3 | 0.30 | 0.88 | 0.69 | 0.90 | 0.34 | 0.50 | 0.981 |
| fog | 5 | 1.02 | 2.16 | 0.68 | 0.94 | 0.45 | 0.38 | 0.930 |
| frost | 1 | 0.41 | 0.99 | 0.51 | 0.89 | 0.36 | 0.50 | 0.983 |
| frost | 3 | 0.97 | 2.01 | 0.64 | 0.89 | 0.45 | 0.40 | 0.953 |
| frost | 5 | 1.40 | 2.63 | 0.68 | 0.90 | 0.51 | 0.40 | 0.938 |
| gaussian_noise | 1 | 0.98 | 1.90 | 0.81 | 0.94 | 0.48 | 0.48 | 0.949 |
| gaussian_noise | 3 | 2.23 | 3.63 | 0.71 | 0.96 | 0.57 | 0.43 | 0.921 |
| gaussian_noise | 5 | 2.63 | 4.12 | 0.65 | 0.96 | 0.60 | 0.43 | 0.932 |
| glass_blur | 1 | 1.34 | 2.98 | 0.73 | 0.90 | 0.41 | 0.23 | 0.921 |
| glass_blur | 3 | 1.20 | 2.76 | 0.77 | 0.91 | 0.40 | 0.23 | 0.918 |
| glass_blur | 5 | 1.48 | 3.25 | 0.77 | 0.92 | 0.42 | 0.21 | 0.905 |
| impulse_noise | 1 | 0.59 | 1.53 | 0.37 | 0.81 | 0.35 | 0.44 | 0.995 |
| impulse_noise | 3 | 1.03 | 2.69 | 0.45 | 0.82 | 0.37 | 0.18 | 0.933 |
| impulse_noise | 5 | 2.35 | 4.16 | 0.43 | 0.93 | 0.54 | 0.31 | 0.905 |
| jpeg_compression | 1 | 0.42 | 1.32 | 0.39 | 0.90 | 0.30 | 0.29 | 0.965 |
| jpeg_compression | 3 | 0.56 | 1.79 | 0.15 | 0.89 | 0.30 | 0.17 | 0.944 |
| jpeg_compression | 5 | 0.67 | 2.11 | 0.12 | 0.90 | 0.31 | 0.13 | 0.933 |
| motion_blur | 1 | 0.32 | 0.98 | 0.81 | 0.91 | 0.31 | 0.40 | 0.971 |
| motion_blur | 3 | 0.76 | 1.97 | 0.86 | 0.94 | 0.35 | 0.24 | 0.920 |
| motion_blur | 5 | 0.95 | 2.38 | 0.89 | 0.94 | 0.37 | 0.20 | 0.900 |
| pixelate | 1 | 0.24 | 0.74 | 0.44 | 0.91 | 0.28 | 0.51 | 0.989 |
| pixelate | 3 | 0.78 | 1.58 | 0.50 | 0.94 | 0.43 | 0.51 | 0.962 |
| pixelate | 5 | 1.86 | 3.26 | 0.43 | 0.95 | 0.52 | 0.40 | 0.941 |
| shot_noise | 1 | 0.72 | 1.43 | 0.76 | 0.93 | 0.46 | 0.56 | 0.968 |
| shot_noise | 3 | 1.81 | 3.05 | 0.73 | 0.95 | 0.55 | 0.45 | 0.922 |
| shot_noise | 5 | 2.40 | 3.86 | 0.63 | 0.95 | 0.59 | 0.42 | 0.920 |
| snow | 1 | 0.36 | 1.10 | 0.53 | 0.82 | 0.30 | 0.42 | 0.995 |
| snow | 3 | 0.50 | 1.63 | 0.43 | 0.76 | 0.31 | 0.24 | 0.979 |
| snow | 5 | 0.76 | 1.99 | 0.52 | 0.87 | 0.37 | 0.29 | 0.957 |
| zoom_blur | 1 | 0.41 | 1.19 | 0.85 | 0.90 | 0.33 | 0.34 | 0.960 |
| zoom_blur | 3 | 0.70 | 1.74 | 0.87 | 0.93 | 0.38 | 0.31 | 0.934 |
| zoom_blur | 5 | 1.10 | 2.50 | 0.90 | 0.95 | 0.42 | 0.24 | 0.902 |

## 5. Density per split
layer `penult`

| split | sparse frac (ref q95) | median log-radius shift |
|---|---|---|
| corrupt__brightness__s1 | 0.176 | 0.047 |
| corrupt__brightness__s3 | 0.197 | 0.078 |
| corrupt__brightness__s5 | 0.264 | 0.170 |
| corrupt__contrast__s1 | 0.174 | 0.048 |
| corrupt__contrast__s3 | 0.305 | 0.224 |
| corrupt__contrast__s5 | 0.853 | 1.029 |
| corrupt__defocus_blur__s1 | 0.170 | 0.050 |
| corrupt__defocus_blur__s3 | 0.271 | 0.218 |
| corrupt__defocus_blur__s5 | 0.635 | 0.768 |
| corrupt__elastic_transform__s1 | 0.293 | 0.210 |
| corrupt__elastic_transform__s3 | 0.347 | 0.310 |
| corrupt__elastic_transform__s5 | 0.543 | 0.628 |
| corrupt__fog__s1 | 0.170 | 0.052 |
| corrupt__fog__s3 | 0.231 | 0.125 |
| corrupt__fog__s5 | 0.501 | 0.567 |
| corrupt__frost__s1 | 0.280 | 0.191 |
| corrupt__frost__s3 | 0.500 | 0.563 |
| corrupt__frost__s5 | 0.631 | 0.765 |
| corrupt__gaussian_noise__s1 | 0.458 | 0.489 |
| corrupt__gaussian_noise__s3 | 0.639 | 0.792 |
| corrupt__gaussian_noise__s5 | 0.677 | 0.825 |
| corrupt__glass_blur__s1 | 0.701 | 0.904 |
| corrupt__glass_blur__s3 | 0.668 | 0.851 |
| corrupt__glass_blur__s5 | 0.745 | 0.958 |
| corrupt__impulse_noise__s1 | 0.425 | 0.444 |
| corrupt__impulse_noise__s3 | 0.708 | 0.904 |
| corrupt__impulse_noise__s5 | 0.901 | 1.065 |
| corrupt__jpeg_compression__s1 | 0.340 | 0.285 |
| corrupt__jpeg_compression__s3 | 0.460 | 0.482 |
| corrupt__jpeg_compression__s5 | 0.539 | 0.626 |
| corrupt__motion_blur__s1 | 0.270 | 0.185 |
| corrupt__motion_blur__s3 | 0.485 | 0.536 |
| corrupt__motion_blur__s5 | 0.577 | 0.678 |
| corrupt__pixelate__s1 | 0.257 | 0.153 |
| corrupt__pixelate__s3 | 0.418 | 0.379 |
| corrupt__pixelate__s5 | 0.650 | 0.872 |
| corrupt__shot_noise__s1 | 0.354 | 0.310 |
| corrupt__shot_noise__s3 | 0.602 | 0.748 |
| corrupt__shot_noise__s5 | 0.706 | 0.872 |
| corrupt__snow__s1 | 0.316 | 0.274 |
| corrupt__snow__s3 | 0.448 | 0.478 |
| corrupt__snow__s5 | 0.519 | 0.591 |
| corrupt__zoom_blur__s1 | 0.311 | 0.270 |
| corrupt__zoom_blur__s3 | 0.444 | 0.473 |
| corrupt__zoom_blur__s5 | 0.611 | 0.757 |
| ood__cifar100 | 0.786 | 0.958 |
| ood__svhn | 0.903 | 1.062 |
| panel | 0.031 | 0.004 |
| test | 0.172 | 0.035 |

## 6. Adjacency
layer `penult`


closest class pairs (sep = center distance / RMS radius): (3,5) 4.64, (2,3) 4.87, (2,5) 5.02, (5,7) 5.07, (0,3) 5.08, (0,2) 5.09

valley ratio mean 4.47, min 3.31, pairs with a valley 1.00

nearest-center confusions (true → nearest): 5→3 0.06, 3→5 0.05, 0→8 0.02, 8→0 0.02, 9→1 0.02, 6→3 0.02

single-linkage merge order (first 5): [3, 5]@4.64 ; [2, 3, 5]@4.87 ; [2, 3, 5, 7]@5.07 ; [0, 2, 3, 5, 7]@5.08 ; [0, 2, 3, 5, 6, 7]@5.26

## 7. Type-b margin (margin_typeb)
_margin_typeb not run_
