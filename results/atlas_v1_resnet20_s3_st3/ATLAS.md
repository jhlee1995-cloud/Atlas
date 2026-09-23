# ATLAS — atlas_v1_resnet20_s3_st3
built 2026-09-23 13:47:09 · source **real** · arch `cifar10_resnet20` · weights `file:/workspace/models/resnet20_s3_chenyaofo.pt sha256:9b06806321d39155` · layers 11

## 1. Per-layer invariants
| layer | D | PR | eff.rank | dim95 | TwoNN ID | sep ratio | NC1 | ETF dev | hub skew | test sparse | NC acc(test) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | 16 | 2.6 | 3.5 | 4 | 6.8 | 0.15 | 318.79 | 0.647 | 0.22 | 0.057 | 0.246 |
| layer1.0 | 16 | 4.3 | 5.7 | 7 | 8.8 | 0.19 | 123.09 | 0.561 | 0.64 | 0.054 | 0.290 |
| layer1.1 | 16 | 4.9 | 6.6 | 8 | 9.6 | 0.23 | 69.76 | 0.532 | 0.81 | 0.055 | 0.329 |
| layer1.2 | 16 | 5.3 | 7.4 | 9 | 10.0 | 0.29 | 48.91 | 0.465 | 0.77 | 0.056 | 0.351 |
| layer2.0 | 32 | 5.9 | 9.2 | 12 | 11.7 | 0.34 | 24.57 | 0.471 | 0.98 | 0.050 | 0.406 |
| layer2.1 | 32 | 7.0 | 11.0 | 15 | 13.0 | 0.37 | 15.98 | 0.447 | 1.21 | 0.049 | 0.457 |
| layer2.2 | 32 | 8.6 | 12.9 | 17 | 14.1 | 0.41 | 8.82 | 0.389 | 1.42 | 0.051 | 0.518 |
| layer3.0 | 64 | 11.5 | 21.2 | 37 | 18.0 | 0.55 | 2.74 | 0.355 | 1.87 | 0.053 | 0.641 |
| layer3.1 | 64 | 12.6 | 23.6 | 42 | 19.8 | 0.82 | 0.92 | 0.277 | 1.75 | 0.058 | 0.796 |
| layer3.2 | 64 | 8.4 | 9.6 | 9 | 9.8 | 2.97 | 0.17 | 0.109 | 0.77 | 0.105 | 0.923 |
| penult | 64 | 8.4 | 9.6 | 9 | 9.8 | 2.97 | 0.17 | 0.109 | 0.77 | 0.105 | 0.923 |

## 2. Decodability profile (which factor lives where)
categorical = accuracy minus majority-class rate; continuous = CV R²

| factor | kind | stem | layer1.0 | layer1.1 | layer1.2 | layer2.0 | layer2.1 | layer2.2 | layer3.0 | layer3.1 | layer3.2 | penult | commit | peak | washout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| luminance_mean | continuous | 1.00 | 0.99 | 0.98 | 0.97 | 0.95 | 0.92 | 0.90 | 0.77 | 0.65 | 0.45 | 0.45 | stem | stem | 0.55 |
| contrast_rms | continuous | 0.63 | 0.58 | 0.56 | 0.56 | 0.67 | 0.64 | 0.61 | 0.62 | 0.56 | 0.45 | 0.45 | stem | layer2.0 | 0.22 |
| highfreq_ratio | continuous | 0.27 | 0.57 | 0.56 | 0.54 | 0.59 | 0.58 | 0.57 | 0.59 | 0.54 | 0.46 | 0.46 | layer1.0 | layer2.0 | 0.13 |
| spectral_slope | continuous | 0.17 | 0.48 | 0.51 | 0.51 | 0.66 | 0.65 | 0.63 | 0.63 | 0.60 | 0.49 | 0.49 | layer2.0 | layer2.0 | 0.17 |
| spectral_anisotropy | continuous | 0.20 | 0.26 | 0.26 | 0.28 | 0.32 | 0.32 | 0.32 | 0.35 | 0.32 | 0.26 | 0.26 | layer2.0 | layer3.0 | 0.09 |
| noise_sigma | continuous | 0.69 | 0.92 | 0.92 | 0.94 | 0.94 | 0.93 | 0.92 | 0.91 | 0.83 | 0.70 | 0.70 | layer1.0 | layer2.0 | 0.24 |
| saturation_mean | continuous | 0.72 | 0.76 | 0.68 | 0.59 | 0.74 | 0.68 | 0.64 | 0.55 | 0.46 | 0.25 | 0.25 | stem | layer1.0 | 0.51 |
| hue_cos | continuous | 0.77 | 0.81 | 0.76 | 0.77 | 0.80 | 0.78 | 0.77 | 0.71 | 0.63 | 0.50 | 0.50 | stem | layer1.0 | 0.31 |
| hue_sin | continuous | 0.78 | 0.81 | 0.80 | 0.77 | 0.77 | 0.75 | 0.74 | 0.69 | 0.61 | 0.50 | 0.50 | stem | layer1.0 | 0.31 |
| colorfulness | continuous | 0.75 | 0.84 | 0.81 | 0.76 | 0.82 | 0.81 | 0.77 | 0.67 | 0.51 | 0.29 | 0.29 | layer1.0 | layer1.0 | 0.56 |
| edge_density | continuous | 0.72 | 0.84 | 0.80 | 0.84 | 0.89 | 0.88 | 0.87 | 0.83 | 0.74 | 0.60 | 0.60 | layer1.0 | layer2.0 | 0.29 |
| orientation_entropy | continuous | 0.19 | 0.30 | 0.36 | 0.40 | 0.46 | 0.44 | 0.44 | 0.52 | 0.47 | 0.40 | 0.40 | layer3.0 | layer3.0 | 0.12 |
| blockiness | continuous | 0.01 | 0.01 | 0.04 | 0.05 | 0.08 | 0.08 | 0.07 | 0.12 | 0.12 | 0.07 | 0.07 | layer3.0 | layer3.0 | 0.05 |
| class | categorical | 0.25 | 0.30 | 0.31 | 0.35 | 0.45 | 0.50 | 0.55 | 0.68 | 0.76 | 0.82 | 0.82 | layer3.1 | layer3.2 | 0.00 |
| coarse_animal_vehicle | categorical | 0.23 | 0.25 | 0.26 | 0.27 | 0.30 | 0.32 | 0.33 | 0.36 | 0.38 | 0.39 | 0.39 | layer3.0 | layer3.2 | 0.00 |
| corruption_family | categorical | 0.12 | 0.23 | 0.31 | 0.32 | 0.39 | 0.37 | 0.35 | 0.36 | 0.31 | 0.23 | 0.23 | layer2.0 | layer2.0 | 0.16 |
| corruption_type | categorical | 0.16 | 0.30 | 0.39 | 0.47 | 0.53 | 0.53 | 0.54 | 0.49 | 0.42 | 0.27 | 0.27 | layer2.0 | layer2.2 | 0.27 |
| severity | continuous | 0.06 | 0.08 | 0.13 | 0.14 | 0.28 | 0.27 | 0.27 | 0.34 | 0.30 | 0.23 | 0.23 | layer3.0 | layer3.0 | 0.11 |

## 3. Layer flow (CKA)
| pair | CKA |
|---|---|
| stem->layer1.0 | 0.858 |
| layer1.0->layer1.1 | 0.938 |
| layer1.1->layer1.2 | 0.815 |
| layer1.2->layer2.0 | 0.911 |
| layer2.0->layer2.1 | 0.974 |
| layer2.1->layer2.2 | 0.950 |
| layer2.2->layer3.0 | 0.883 |
| layer3.0->layer3.1 | 0.792 |
| layer3.1->layer3.2 | 0.673 |
| layer3.2->penult | 1.000 |

biggest reorganization: `layer3.1->layer3.2` (drop 0.327)

## 4. Sensitivity field (paired corruption displacement)
layer `penult`; magnitude in within-class-radius units

| corruption | sev | magnitude | per-sample | top5 frac | class-sub frac | coherence | class consistency | norm ratio |
|---|---|---|---|---|---|---|---|---|
| brightness | 1 | 0.03 | 0.14 | 0.81 | 0.94 | 0.16 | 0.60 | 1.000 |
| brightness | 3 | 0.08 | 0.41 | 0.72 | 0.93 | 0.18 | 0.37 | 0.987 |
| brightness | 5 | 0.17 | 0.76 | 0.55 | 0.90 | 0.21 | 0.17 | 0.947 |
| contrast | 1 | 0.10 | 0.34 | 0.66 | 0.96 | 0.26 | 0.60 | 1.004 |
| contrast | 3 | 0.39 | 1.10 | 0.66 | 0.92 | 0.33 | 0.31 | 0.934 |
| contrast | 5 | 1.02 | 2.34 | 0.69 | 0.81 | 0.42 | 0.15 | 0.732 |
| defocus_blur | 1 | 0.05 | 0.17 | 0.86 | 0.93 | 0.27 | 0.69 | 1.001 |
| defocus_blur | 3 | 0.36 | 0.82 | 0.90 | 0.95 | 0.39 | 0.54 | 0.963 |
| defocus_blur | 5 | 0.80 | 1.69 | 0.86 | 0.94 | 0.44 | 0.31 | 0.869 |
| elastic_transform | 1 | 0.18 | 0.84 | 0.86 | 0.93 | 0.19 | 0.16 | 0.952 |
| elastic_transform | 3 | 0.35 | 1.01 | 0.90 | 0.95 | 0.31 | 0.33 | 0.936 |
| elastic_transform | 5 | 0.45 | 1.40 | 0.82 | 0.97 | 0.29 | 0.11 | 0.886 |
| fog | 1 | 0.07 | 0.25 | 0.65 | 0.94 | 0.26 | 0.67 | 1.007 |
| fog | 3 | 0.25 | 0.74 | 0.68 | 0.96 | 0.32 | 0.49 | 0.983 |
| fog | 5 | 0.80 | 1.63 | 0.73 | 0.97 | 0.47 | 0.38 | 0.910 |
| frost | 1 | 0.23 | 0.74 | 0.53 | 0.94 | 0.28 | 0.29 | 0.954 |
| frost | 3 | 0.54 | 1.39 | 0.57 | 0.94 | 0.35 | 0.22 | 0.880 |
| frost | 5 | 0.76 | 1.72 | 0.67 | 0.95 | 0.41 | 0.24 | 0.852 |
| gaussian_noise | 1 | 0.72 | 1.38 | 0.83 | 0.97 | 0.49 | 0.49 | 0.901 |
| gaussian_noise | 3 | 1.37 | 2.31 | 0.83 | 0.98 | 0.56 | 0.39 | 0.855 |
| gaussian_noise | 5 | 1.48 | 2.51 | 0.83 | 0.97 | 0.56 | 0.34 | 0.844 |
| glass_blur | 1 | 0.61 | 1.81 | 0.64 | 0.93 | 0.31 | 0.07 | 0.810 |
| glass_blur | 3 | 0.57 | 1.71 | 0.67 | 0.94 | 0.30 | 0.07 | 0.825 |
| glass_blur | 5 | 0.66 | 1.92 | 0.63 | 0.93 | 0.31 | 0.06 | 0.796 |
| impulse_noise | 1 | 0.42 | 1.10 | 0.38 | 0.93 | 0.34 | 0.30 | 0.949 |
| impulse_noise | 3 | 0.60 | 1.75 | 0.54 | 0.92 | 0.32 | 0.10 | 0.868 |
| impulse_noise | 5 | 1.09 | 2.39 | 0.72 | 0.95 | 0.44 | 0.17 | 0.822 |
| jpeg_compression | 1 | 0.26 | 0.97 | 0.25 | 0.91 | 0.25 | 0.15 | 0.924 |
| jpeg_compression | 3 | 0.31 | 1.23 | 0.22 | 0.91 | 0.24 | 0.06 | 0.893 |
| jpeg_compression | 5 | 0.36 | 1.41 | 0.41 | 0.93 | 0.24 | 0.04 | 0.881 |
| motion_blur | 1 | 0.29 | 0.77 | 0.79 | 0.96 | 0.32 | 0.47 | 0.961 |
| motion_blur | 3 | 0.57 | 1.41 | 0.80 | 0.94 | 0.36 | 0.25 | 0.884 |
| motion_blur | 5 | 0.67 | 1.63 | 0.80 | 0.93 | 0.37 | 0.20 | 0.854 |
| pixelate | 1 | 0.22 | 0.61 | 0.70 | 0.99 | 0.33 | 0.54 | 0.966 |
| pixelate | 3 | 0.60 | 1.19 | 0.65 | 0.98 | 0.46 | 0.47 | 0.912 |
| pixelate | 5 | 1.13 | 2.11 | 0.49 | 0.95 | 0.49 | 0.30 | 0.835 |
| shot_noise | 1 | 0.53 | 1.04 | 0.79 | 0.97 | 0.47 | 0.55 | 0.930 |
| shot_noise | 3 | 1.19 | 2.03 | 0.84 | 0.98 | 0.56 | 0.43 | 0.866 |
| shot_noise | 5 | 1.39 | 2.39 | 0.84 | 0.97 | 0.55 | 0.34 | 0.849 |
| snow | 1 | 0.23 | 0.79 | 0.52 | 0.94 | 0.25 | 0.23 | 0.962 |
| snow | 3 | 0.25 | 1.15 | 0.26 | 0.93 | 0.22 | 0.04 | 0.921 |
| snow | 5 | 0.44 | 1.41 | 0.24 | 0.94 | 0.30 | 0.13 | 0.882 |
| zoom_blur | 1 | 0.36 | 0.94 | 0.88 | 0.95 | 0.34 | 0.36 | 0.936 |
| zoom_blur | 3 | 0.54 | 1.24 | 0.89 | 0.94 | 0.40 | 0.35 | 0.902 |
| zoom_blur | 5 | 0.75 | 1.62 | 0.86 | 0.94 | 0.43 | 0.31 | 0.864 |

## 5. Density per split
layer `penult`

| split | sparse frac (ref q95) | median log-radius shift |
|---|---|---|
| corrupt__brightness__s1 | 0.114 | 0.025 |
| corrupt__brightness__s3 | 0.126 | 0.047 |
| corrupt__brightness__s5 | 0.166 | 0.097 |
| corrupt__contrast__s1 | 0.112 | 0.037 |
| corrupt__contrast__s3 | 0.242 | 0.193 |
| corrupt__contrast__s5 | 0.694 | 0.479 |
| corrupt__defocus_blur__s1 | 0.108 | 0.021 |
| corrupt__defocus_blur__s3 | 0.157 | 0.108 |
| corrupt__defocus_blur__s5 | 0.316 | 0.291 |
| corrupt__elastic_transform__s1 | 0.171 | 0.111 |
| corrupt__elastic_transform__s3 | 0.199 | 0.144 |
| corrupt__elastic_transform__s5 | 0.315 | 0.285 |
| corrupt__fog__s1 | 0.114 | 0.035 |
| corrupt__fog__s3 | 0.176 | 0.111 |
| corrupt__fog__s5 | 0.399 | 0.340 |
| corrupt__frost__s1 | 0.172 | 0.108 |
| corrupt__frost__s3 | 0.334 | 0.289 |
| corrupt__frost__s5 | 0.400 | 0.337 |
| corrupt__gaussian_noise__s1 | 0.324 | 0.269 |
| corrupt__gaussian_noise__s3 | 0.376 | 0.297 |
| corrupt__gaussian_noise__s5 | 0.318 | 0.264 |
| corrupt__glass_blur__s1 | 0.442 | 0.364 |
| corrupt__glass_blur__s3 | 0.399 | 0.341 |
| corrupt__glass_blur__s5 | 0.454 | 0.379 |
| corrupt__impulse_noise__s1 | 0.275 | 0.215 |
| corrupt__impulse_noise__s3 | 0.523 | 0.413 |
| corrupt__impulse_noise__s5 | 0.723 | 0.512 |
| corrupt__jpeg_compression__s1 | 0.190 | 0.148 |
| corrupt__jpeg_compression__s3 | 0.263 | 0.222 |
| corrupt__jpeg_compression__s5 | 0.287 | 0.263 |
| corrupt__motion_blur__s1 | 0.170 | 0.108 |
| corrupt__motion_blur__s3 | 0.275 | 0.264 |
| corrupt__motion_blur__s5 | 0.305 | 0.289 |
| corrupt__pixelate__s1 | 0.169 | 0.083 |
| corrupt__pixelate__s3 | 0.329 | 0.230 |
| corrupt__pixelate__s5 | 0.545 | 0.438 |
| corrupt__shot_noise__s1 | 0.231 | 0.185 |
| corrupt__shot_noise__s3 | 0.397 | 0.333 |
| corrupt__shot_noise__s5 | 0.387 | 0.330 |
| corrupt__snow__s1 | 0.185 | 0.137 |
| corrupt__snow__s3 | 0.281 | 0.210 |
| corrupt__snow__s5 | 0.331 | 0.283 |
| corrupt__zoom_blur__s1 | 0.191 | 0.150 |
| corrupt__zoom_blur__s3 | 0.256 | 0.214 |
| corrupt__zoom_blur__s5 | 0.314 | 0.292 |
| ood__cifar100 | 0.484 | 0.391 |
| ood__svhn | 0.601 | 0.457 |
| panel | 0.031 | -0.009 |
| test | 0.105 | 0.021 |

## 6. Adjacency
layer `penult`


closest class pairs (sep = center distance / RMS radius): (3,5) 2.38, (2,3) 2.65, (2,4) 2.81, (0,2) 2.81, (2,5) 2.88, (3,4) 2.89

valley ratio mean 2.21, min 1.37, pairs with a valley 1.00

nearest-center confusions (true → nearest): 5→3 0.07, 3→5 0.06, 9→1 0.04, 2→6 0.03, 6→3 0.03, 8→0 0.03

single-linkage merge order (first 5): [3, 5]@2.38 ; [2, 3, 5]@2.65 ; [2, 3, 4, 5]@2.81 ; [0, 2, 3, 4, 5]@2.81 ; [0, 2, 3, 4, 5, 7]@2.96

## 7. Type-b margin (margin_typeb)
_margin_typeb not run_
