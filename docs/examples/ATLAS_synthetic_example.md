# ATLAS — unnamed
built 2026-09-22 21:44:39 · source **synthetic** · arch `synthetic_readout` · weights `none` · layers 5
> **SYNTHETIC dump.** Nothing here is evidence about a real model. Pipeline check only.

## 1. Per-layer invariants
| layer | D | PR | eff.rank | dim95 | TwoNN ID | sep ratio | NC1 | ETF dev | hub skew | test sparse | NC acc(test) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stem | 16 | 5.2 | 7.7 | 10 | 11.0 | 0.41 | 12.29 | 0.275 | 0.98 | 0.070 | 0.347 |
| layer1.1 | 16 | 6.0 | 7.9 | 9 | 10.6 | 0.76 | 3.57 | 0.325 | 0.99 | 0.055 | 0.661 |
| layer2.1 | 32 | 8.4 | 12.2 | 16 | 16.3 | 1.79 | 0.28 | 0.191 | 1.86 | 0.054 | 1.000 |
| layer3.1 | 64 | 8.8 | 12.5 | 19 | 23.1 | 3.28 | 0.04 | 0.139 | 2.06 | 0.060 | 1.000 |
| penult | 64 | 8.3 | 10.8 | 11 | 23.3 | 4.65 | 0.02 | 0.130 | 2.28 | 0.066 | 1.000 |

## 2. Decodability profile (which factor lives where)
categorical = accuracy minus majority-class rate; continuous = CV R²

| factor | kind | stem | layer1.1 | layer2.1 | layer3.1 | penult | commit | peak | washout |
|---|---|---|---|---|---|---|---|---|---|
| luminance_mean | continuous | 0.51 | 0.62 | 0.66 | 0.63 | 0.28 | layer1.1 | layer2.1 | 0.38 |
| contrast_rms | continuous | 0.28 | 0.25 | 0.40 | 0.45 | 0.32 | layer3.1 | layer3.1 | 0.13 |
| highfreq_ratio | continuous | 0.36 | 0.52 | 0.56 | 0.55 | 0.29 | layer1.1 | layer2.1 | 0.27 |
| spectral_slope | continuous | 0.36 | 0.49 | 0.58 | 0.61 | 0.49 | layer2.1 | layer3.1 | 0.12 |
| spectral_anisotropy | continuous | 0.39 | 0.37 | 0.52 | 0.48 | 0.30 | layer2.1 | layer2.1 | 0.22 |
| noise_sigma | continuous | 0.60 | 0.54 | 0.48 | 0.43 | 0.01 | stem | stem | 0.58 |
| saturation_mean | continuous | 0.43 | 0.37 | 0.35 | 0.38 | 0.05 | stem | stem | 0.38 |
| hue_cos | continuous | -0.00 | -0.00 | -0.00 | -0.00 | -0.00 | · | · | · |
| hue_sin | continuous | -0.00 | -0.00 | -0.00 | 0.00 | 0.00 | penult | penult | 0.00 |
| colorfulness | continuous | 0.41 | 0.34 | 0.30 | 0.28 | -0.00 | stem | stem | 0.41 |
| edge_density | continuous | 0.44 | 0.39 | 0.46 | 0.44 | 0.18 | stem | layer2.1 | 0.28 |
| orientation_entropy | continuous | 0.40 | 0.39 | 0.33 | 0.28 | 0.09 | stem | stem | 0.31 |
| blockiness | continuous | 0.11 | 0.03 | 0.37 | 0.44 | 0.45 | layer3.1 | penult | 0.00 |
| class | categorical | 0.46 | 0.72 | 0.89 | 0.89 | 0.89 | layer2.1 | layer2.1 | 0.00 |
| coarse_animal_vehicle | categorical | 0.14 | 0.32 | 0.40 | 0.42 | 0.42 | layer2.1 | layer3.1 | 0.00 |
| corruption_family | categorical | 0.32 | 0.32 | 0.27 | 0.24 | 0.03 | stem | stem | 0.29 |
| corruption_type | categorical | 0.31 | 0.31 | 0.27 | 0.24 | 0.04 | stem | stem | 0.27 |
| severity | continuous | 0.09 | 0.07 | 0.07 | 0.05 | 0.01 | stem | stem | 0.08 |

## 3. Layer flow (CKA)
| pair | CKA |
|---|---|
| stem->layer1.1 | 0.106 |
| layer1.1->layer2.1 | 0.247 |
| layer2.1->layer3.1 | 0.569 |
| layer3.1->penult | 0.752 |

biggest reorganization: `stem->layer1.1` (drop 0.894)

## 4. Sensitivity field (paired corruption displacement)
layer `penult`; magnitude in within-class-radius units

| corruption | sev | magnitude | per-sample | top5 frac | class-sub frac | coherence | class consistency | norm ratio |
|---|---|---|---|---|---|---|---|---|
| brightness | 1 | 0.06 | 1.41 | 0.07 | 0.22 | 0.04 | 0.01 | 1.001 |
| brightness | 3 | 0.09 | 1.41 | 0.28 | 0.51 | 0.06 | 0.12 | 0.998 |
| brightness | 5 | 0.12 | 1.42 | 0.17 | 0.33 | 0.08 | 0.21 | 1.001 |
| defocus_blur | 1 | 0.04 | 1.40 | 0.04 | 0.12 | 0.03 | -0.04 | 0.999 |
| defocus_blur | 3 | 0.06 | 1.40 | 0.19 | 0.43 | 0.04 | 0.00 | 1.001 |
| defocus_blur | 5 | 0.05 | 1.41 | 0.35 | 0.48 | 0.04 | -0.02 | 1.000 |
| gaussian_noise | 1 | 0.05 | 1.42 | 0.11 | 0.23 | 0.03 | -0.04 | 1.001 |
| gaussian_noise | 3 | 0.05 | 1.42 | 0.22 | 0.40 | 0.03 | -0.03 | 1.000 |
| gaussian_noise | 5 | 0.08 | 1.42 | 0.03 | 0.21 | 0.05 | 0.09 | 1.000 |

## 5. Density per split
layer `penult`

| split | sparse frac (ref q95) | median log-radius shift |
|---|---|---|
| corrupt__brightness__s1 | 0.072 | 0.007 |
| corrupt__brightness__s3 | 0.080 | 0.008 |
| corrupt__brightness__s5 | 0.065 | 0.013 |
| corrupt__defocus_blur__s1 | 0.053 | 0.007 |
| corrupt__defocus_blur__s3 | 0.062 | 0.007 |
| corrupt__defocus_blur__s5 | 0.052 | 0.004 |
| corrupt__gaussian_noise__s1 | 0.058 | 0.015 |
| corrupt__gaussian_noise__s3 | 0.068 | 0.016 |
| corrupt__gaussian_noise__s5 | 0.078 | 0.002 |
| panel | 0.047 | 0.004 |
| test | 0.066 | 0.010 |

## 6. Adjacency
layer `penult`


closest class pairs (sep = center distance / RMS radius): (1,6) 4.16, (1,4) 4.40, (1,2) 4.43, (0,3) 4.44, (0,2) 4.64, (1,7) 4.78

valley ratio mean 3.31, min 2.62, pairs with a valley 1.00

single-linkage merge order (first 5): [1, 6]@4.16 ; [1, 4, 6]@4.40 ; [1, 2, 4, 6]@4.43 ; [0, 3]@4.44 ; [0, 1, 2, 3, 4, 6]@4.64
