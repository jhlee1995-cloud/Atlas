# CRITIC  verdict: **PARTIAL**
runs: ['results/atlas_v1_resnet20_s0hub_st3', 'results/atlas_v1_resnet20_s1_st3', 'results/atlas_v1_resnet20_s2_st3', 'results/atlas_v1_resnet20_s3_st3', 'results/atlas_v1_resnet20_s4_st3', 'results/atlas_v1_resnet56_e40_st3', 'results/atlas_v1_resnet56_e50', 'results/atlas_v1_resnet56_e60', 'results/atlas_v1_resnet56_e70', 'results/atlas_v1_resnet56_s12m', 'results/atlas_v1_resnet56_s13m']
counts: {'PASS': 2136, 'FAIL': 174, 'WARN': 13, 'INFO': 43}
layer alignment: position {"atlas_v1_resnet56_e40_st3": [["stem", "stem"], ["layer1.0", "layer1.0"], ["layer1.1", "layer1.5"], ["layer1.2", "layer1.8"], ["layer2.0", "layer2.0"], ["layer2.1", "layer2.5"], ["layer2.2", "layer2.8"], ["layer3.0", "layer3.0"], ["layer3.1", "layer3.5"], ["layer3.2", "layer3.8"], ["penult", "penult"]], "atlas_v1_resnet56_e50": [["stem", "stem"], ["layer1.0", "layer1.0"], ["layer1.1", "layer1.5"], ["layer1.2", "layer1.8"], ["layer2.0", "layer2.0"], ["layer2.1", "layer2.5"], ["layer2.2", "layer2.8"], ["layer3.0", "layer3.0"], ["layer3.1", "layer3.5"], ["layer3.2", "layer3.8"], ["penult", "penult"]], "atlas_v1_resnet56_e60": [["stem", "stem"], ["layer1.0", "layer1.0"], ["layer1.1", "layer1.5"], ["layer1.2", "layer1.8"], ["layer2.0", "layer2.0"], ["layer2.1", "layer2.5"], ["layer2.2", "layer2.8"], ["layer3.0", "layer3.0"], ["layer3.1", "layer3.5"], ["layer3.2", "layer3.8"], ["penult", "penult"]], "atlas_v1_resnet56_e70": [["stem", "stem"], ["layer1.0", "layer1.0"], ["layer1.1", "layer1.5"], ["layer1.2", "layer1.8"], ["layer2.0", "layer2.0"], ["layer2.1", "layer2.5"], ["layer2.2", "layer2.8"], ["layer3.0", "layer3.0"], ["layer3.1", "layer3.5"], ["layer3.2", "layer3.8"], ["penult", "penult"]], "atlas_v1_resnet56_s12m": [["stem", "stem"], ["layer1.0", "layer1.0"], ["layer1.1", "layer1.5"], ["layer1.2", "layer1.8"], ["layer2.0", "layer2.0"], ["layer2.1", "layer2.5"], ["layer2.2", "layer2.8"], ["layer3.0", "layer3.0"], ["layer3.1", "layer3.5"], ["layer3.2", "layer3.8"], ["penult", "penult"]], "atlas_v1_resnet56_s13m": [["stem", "stem"], ["layer1.0", "layer1.0"], ["layer1.1", "layer1.5"], ["layer1.2", "layer1.8"], ["layer2.0", "layer2.0"], ["layer2.1", "layer2.5"], ["layer2.2", "layer2.8"], ["layer3.0", "layer3.0"], ["layer3.1", "layer3.5"], ["layer3.2", "layer3.8"], ["penult", "penult"]]}

## synthetic_refusal
1 items, 0 FAIL
- PASS: 1 items

## norm_consistency
1 items, 0 FAIL
- PASS: 1 items

## alias_layers
1 items, 0 FAIL
- **WARN** `alias:layer3.2` identical to penult in every run; counted once as penult

## holdout_hygiene
11 items, 0 FAIL
- PASS: 11 items

## probe_hygiene
11 items, 0 FAIL
- PASS: 11 items

## scalar_stability
80 items, 48 FAIL
- **FAIL** `stem/twonn_id.id` values=[6.827, 6.729, 6.979, 6.775, 6.843, 7.163, 6.606, 6.989, 6.78, 6.329, 5.72] rel_spread=0.215
- **FAIL** `stem/pca_spectrum.participation_ratio` values=[2.979, 2.993, 2.975, 2.644, 3.09, 2.954, 2.911, 2.972, 2.708, 2.874, 2.953] rel_spread=0.153
- **FAIL** `stem/pca_spectrum.dim95` values=[4.0, 4.0, 4.0, 4.0, 4.0, 5.0, 4.0, 4.0, 4.0, 4.0, 3.0] rel_spread=0.500
- **FAIL** `stem/neural_collapse.nc1` values=[259.926, 261.068, 152.924, 318.793, 159.182, 147.951, 302.323, 145.302, 363.84, 265.349, 276.828] rel_spread=0.906
- **FAIL** `stem/hubness.k_occurrence_skew` values=[0.212, 0.179, 0.209, 0.223, 0.202, 0.264, 0.248, 0.209, 0.168, 0.075, 0.005] rel_spread=1.428
- **FAIL** `layer1.0/twonn_id.id` values=[8.83, 8.577, 9.201, 8.83, 9.25, 8.406, 7.226, 8.151, 7.786, 7.815, 6.504] rel_spread=0.334
- **FAIL** `layer1.0/pca_spectrum.participation_ratio` values=[4.042, 3.288, 3.875, 4.303, 3.95, 3.647, 2.897, 3.094, 2.592, 2.882, 3.144] rel_spread=0.499
- **FAIL** `layer1.0/pca_spectrum.dim95` values=[6.0, 5.0, 7.0, 7.0, 7.0, 6.0, 5.0, 5.0, 5.0, 5.0, 4.0] rel_spread=0.532
- **FAIL** `layer1.0/neural_collapse.nc1` values=[138.342, 164.627, 136.168, 123.091, 93.88, 99.826, 507.796, 142.655, 313.151, 118.818, 209.57] rel_spread=2.223
- **FAIL** `layer1.0/neural_collapse.etf_deviation` values=[0.581, 0.635, 0.613, 0.561, 0.519, 0.604, 0.663, 0.631, 0.664, 0.682, 0.624] rel_spread=0.266
- **FAIL** `layer1.0/hubness.k_occurrence_skew` values=[0.631, 0.535, 0.741, 0.642, 0.74, 0.454, 0.364, 0.438, 0.426, 0.395, 0.119] rel_spread=1.248
- **FAIL** `layer1.1/pca_spectrum.participation_ratio` values=[5.171, 4.07, 5.067, 4.944, 4.893, 4.255, 3.866, 4.819, 5.419, 5.126, 4.125] rel_spread=0.330
- **FAIL** `layer1.1/class_centers.sep_ratio` values=[0.21, 0.209, 0.209, 0.227, 0.215, 0.284, 0.211, 0.248, 0.271, 0.22, 0.235] rel_spread=0.326
- **FAIL** `layer1.1/neural_collapse.nc1` values=[57.975, 74.307, 68.463, 69.758, 86.846, 103.293, 114.243, 77.569, 60.837, 75.862, 131.843] rel_spread=0.882
- **FAIL** `layer1.1/neural_collapse.etf_deviation` values=[0.494, 0.544, 0.575, 0.532, 0.46, 0.524, 0.574, 0.485, 0.448, 0.475, 0.572] rel_spread=0.246
- **FAIL** `layer1.1/hubness.k_occurrence_skew` values=[0.797, 0.644, 0.77, 0.815, 0.8, 0.677, 0.634, 0.724, 0.808, 0.718, 0.603] rel_spread=0.292
- **FAIL** `layer1.2/pca_spectrum.participation_ratio` values=[5.518, 4.8, 4.499, 5.286, 5.102, 5.044, 4.962, 5.3, 5.366, 5.003, 5.16] rel_spread=0.200
- **FAIL** `layer1.2/neural_collapse.nc1` values=[53.174, 49.246, 47.77, 48.912, 48.318, 72.8, 81.018, 44.92, 47.787, 66.938, 59.763] rel_spread=0.640
- **FAIL** `layer1.2/neural_collapse.etf_deviation` values=[0.464, 0.428, 0.452, 0.465, 0.432, 0.485, 0.485, 0.45, 0.413, 0.43, 0.469] rel_spread=0.160
- **FAIL** `layer1.2/hubness.k_occurrence_skew` values=[0.871, 0.713, 0.821, 0.767, 0.922, 0.733, 0.664, 0.814, 0.815, 0.831, 0.719] rel_spread=0.327
- **FAIL** `layer2.0/pca_spectrum.participation_ratio` values=[6.816, 6.067, 5.598, 5.852, 6.145, 5.318, 5.348, 5.533, 5.296, 5.561, 5.751] rel_spread=0.264
- **FAIL** `layer2.0/pca_spectrum.dim95` values=[13.0, 12.0, 11.0, 12.0, 12.0, 10.0, 9.0, 10.0, 10.0, 9.0, 10.0] rel_spread=0.373
- **FAIL** `layer2.0/neural_collapse.nc1` values=[21.892, 23.829, 19.416, 24.574, 20.88, 29.116, 31.653, 25.816, 21.759, 29.771, 29.051] rel_spread=0.485
- **FAIL** `layer2.0/hubness.k_occurrence_skew` values=[0.962, 0.924, 0.913, 0.975, 1.008, 0.828, 0.721, 0.801, 0.945, 0.798, 0.764] rel_spread=0.328
- **FAIL** `layer2.1/pca_spectrum.participation_ratio` values=[7.778, 7.692, 7.111, 6.981, 7.491, 6.078, 6.413, 6.875, 6.417, 7.281, 7.042] rel_spread=0.242
- **FAIL** `layer2.1/class_centers.sep_ratio` values=[0.386, 0.394, 0.38, 0.368, 0.373, 0.434, 0.425, 0.424, 0.426, 0.432, 0.427] rel_spread=0.163
- **FAIL** `layer2.1/neural_collapse.nc1` values=[12.411, 17.043, 11.104, 15.979, 14.636, 11.762, 11.799, 12.266, 9.113, 12.202, 9.854] rel_spread=0.631
- **FAIL** `layer2.1/neural_collapse.etf_deviation` values=[0.44, 0.401, 0.4, 0.447, 0.422, 0.428, 0.424, 0.413, 0.386, 0.401, 0.384] rel_spread=0.152
- **FAIL** `layer2.1/hubness.k_occurrence_skew` values=[1.239, 1.202, 1.154, 1.21, 1.304, 1.096, 1.043, 1.056, 1.269, 1.215, 1.26] rel_spread=0.220
- **FAIL** `layer2.2/pca_spectrum.participation_ratio` values=[9.005, 8.577, 7.948, 8.558, 8.595, 7.927, 7.202, 7.075, 7.178, 7.741, 7.132] rel_spread=0.244
- **FAIL** `layer2.2/pca_spectrum.dim95` values=[18.0, 17.0, 17.0, 17.0, 18.0, 15.0, 15.0, 15.0, 16.0, 16.0, 15.0] rel_spread=0.184
- **FAIL** `layer2.2/class_centers.sep_ratio` values=[0.429, 0.42, 0.417, 0.411, 0.415, 0.466, 0.462, 0.451, 0.492, 0.473, 0.437] rel_spread=0.184
- **FAIL** `layer2.2/neural_collapse.nc1` values=[7.065, 9.177, 7.357, 8.82, 8.825, 8.688, 7.849, 7.188, 6.136, 8.869, 7.987] rel_spread=0.380
- **FAIL** `layer2.2/hubness.k_occurrence_skew` values=[1.241, 1.427, 1.442, 1.419, 1.421, 1.241, 1.197, 1.248, 1.391, 1.259, 1.305] rel_spread=0.184
- **FAIL** `layer3.0/twonn_id.id` values=[18.102, 18.012, 18.284, 17.981, 18.308, 14.789, 15.143, 15.09, 15.983, 15.619, 14.683] rel_spread=0.219
- **FAIL** `layer3.0/pca_spectrum.participation_ratio` values=[12.136, 10.286, 12.091, 11.533, 10.434, 8.419, 8.849, 8.01, 8.649, 8.704, 8.852] rel_spread=0.420
- **FAIL** `layer3.0/pca_spectrum.dim95` values=[37.0, 38.0, 38.0, 37.0, 37.0, 22.0, 22.0, 21.0, 25.0, 24.0, 20.0] rel_spread=0.617
- **FAIL** `layer3.0/neural_collapse.nc1` values=[2.416, 2.391, 2.533, 2.741, 2.396, 3.888, 4.022, 3.873, 2.98, 3.494, 3.922] rel_spread=0.518
- **FAIL** `layer3.0/neural_collapse.etf_deviation` values=[0.308, 0.329, 0.334, 0.355, 0.338, 0.325, 0.379, 0.357, 0.338, 0.35, 0.351] rel_spread=0.207
- **FAIL** `layer3.0/hubness.k_occurrence_skew` values=[1.829, 1.691, 1.701, 1.87, 1.874, 1.398, 1.504, 1.228, 1.396, 1.43, 1.262] rel_spread=0.413
- **FAIL** `layer3.1/pca_spectrum.participation_ratio` values=[10.468, 11.259, 9.641, 12.61, 9.453, 13.752, 14.097, 13.855, 14.41, 12.825, 14.751] rel_spread=0.425
- **FAIL** `layer3.1/class_centers.sep_ratio` values=[0.899, 0.873, 0.81, 0.82, 0.833, 0.889, 0.851, 0.784, 1.083, 0.799, 0.757] rel_spread=0.382
- **FAIL** `layer3.1/neural_collapse.nc1` values=[0.759, 0.774, 0.859, 0.923, 0.792, 0.75, 0.784, 0.837, 0.521, 0.924, 1.011] rel_spread=0.603
- **FAIL** `layer3.1/neural_collapse.etf_deviation` values=[0.275, 0.289, 0.29, 0.277, 0.296, 0.221, 0.24, 0.246, 0.184, 0.275, 0.243] rel_spread=0.433
- **FAIL** `layer3.1/hubness.k_occurrence_skew` values=[1.907, 1.898, 1.805, 1.755, 1.86, 1.813, 1.725, 1.996, 1.813, 1.872, 2.028] rel_spread=0.163
- **FAIL** `penult/class_centers.sep_ratio` values=[2.996, 3.067, 3.021, 2.97, 3.024, 2.853, 3.14, 3.368, 3.506, 3.367, 3.314] rel_spread=0.207
- **FAIL** `penult/neural_collapse.nc1` values=[0.17, 0.163, 0.17, 0.172, 0.167, 0.177, 0.145, 0.129, 0.117, 0.129, 0.13] rel_spread=0.396
- **FAIL** `penult/hubness.k_occurrence_skew` values=[0.763, 0.833, 0.767, 0.772, 0.804, 0.958, 1.026, 0.875, 0.898, 0.878, 0.908] rel_spread=0.304
- PASS: 32 items

## id_profile_stability
55 items, 0 FAIL
- PASS: 55 items

## adjacency_stability
1100 items, 52 FAIL
- **FAIL** `layer2.1/merge_kendall[0,8]` tau=0.506
- **FAIL** `layer2.1/merge_kendall[0,10]` tau=0.495
- **FAIL** `layer2.1/merge_kendall[1,8]` tau=0.552
- **FAIL** `layer2.1/merge_kendall[1,10]` tau=0.571
- **FAIL** `layer2.1/merge_kendall[2,8]` tau=0.540
- **FAIL** `layer2.1/merge_kendall[2,10]` tau=0.529
- **FAIL** `layer2.1/merge_kendall[3,8]` tau=0.560
- **FAIL** `layer2.1/merge_kendall[3,10]` tau=0.549
- **FAIL** `layer2.1/merge_kendall[4,8]` tau=0.488
- **FAIL** `layer2.1/merge_kendall[4,10]` tau=0.480
- **FAIL** `layer2.1/merge_kendall[5,8]` tau=0.560
- **FAIL** `layer2.1/merge_kendall[5,10]` tau=0.549
- **FAIL** `layer2.1/merge_kendall[6,8]` tau=0.558
- **FAIL** `layer2.1/merge_kendall[6,10]` tau=0.547
- **FAIL** `layer2.1/merge_kendall[7,8]` tau=0.539
- **FAIL** `layer2.1/merge_kendall[7,10]` tau=0.530
- **FAIL** `layer2.1/merge_kendall[8,9]` tau=0.533
- **FAIL** `layer2.1/merge_kendall[9,10]` tau=0.530
- **FAIL** `layer2.2/merge_kendall[0,5]` tau=0.544
- **FAIL** `layer2.2/merge_kendall[0,7]` tau=0.467
- **FAIL** `layer2.2/merge_kendall[0,8]` tau=0.556
- **FAIL** `layer2.2/merge_kendall[0,10]` tau=0.536
- **FAIL** `layer2.2/merge_kendall[1,5]` tau=0.541
- **FAIL** `layer2.2/merge_kendall[1,7]` tau=0.505
- **FAIL** `layer2.2/merge_kendall[1,8]` tau=0.530
- **FAIL** `layer2.2/merge_kendall[1,10]` tau=0.539
- **FAIL** `layer2.2/merge_kendall[2,5]` tau=0.538
- **FAIL** `layer2.2/merge_kendall[2,7]` tau=0.508
- **FAIL** `layer2.2/merge_kendall[2,8]` tau=0.533
- **FAIL** `layer2.2/merge_kendall[2,10]` tau=0.541
- **FAIL** `layer2.2/merge_kendall[3,5]` tau=0.541
- **FAIL** `layer2.2/merge_kendall[3,7]` tau=0.505
- **FAIL** `layer2.2/merge_kendall[3,8]` tau=0.530
- **FAIL** `layer2.2/merge_kendall[3,10]` tau=0.539
- **FAIL** `layer2.2/merge_kendall[4,5]` tau=0.532
- **FAIL** `layer2.2/merge_kendall[4,7]` tau=0.506
- **FAIL** `layer2.2/merge_kendall[4,8]` tau=0.526
- **FAIL** `layer2.2/merge_kendall[4,10]` tau=0.548
- **FAIL** `layer2.2/merge_kendall[5,6]` tau=0.570
- **FAIL** `layer2.2/merge_kendall[5,9]` tau=0.533
- **FAIL** `layer2.2/merge_kendall[6,7]` tau=0.478
- **FAIL** `layer2.2/merge_kendall[6,8]` tau=0.559
- **FAIL** `layer2.2/merge_kendall[6,10]` tau=0.535
- **FAIL** `layer2.2/merge_kendall[7,9]` tau=0.503
- **FAIL** `layer2.2/merge_kendall[8,9]` tau=0.522
- **FAIL** `layer2.2/merge_kendall[9,10]` tau=0.531
- **FAIL** `penult/merge_kendall[0,10]` tau=0.525
- **FAIL** `penult/merge_kendall[2,10]` tau=0.577
- **FAIL** `penult/merge_kendall[3,10]` tau=0.525
- **FAIL** `penult/merge_kendall[4,6]` tau=0.594
- **FAIL** `penult/merge_kendall[4,10]` tau=0.594
- **FAIL** `penult/merge_kendall[5,10]` tau=0.577
- PASS: 1048 items

## decodability_stability
990 items, 65 FAIL
- **FAIL** `decod/contrast_rms[0,5]` profile_spearman=0.55 mean_abs_delta=0.044
- **FAIL** `decod/contrast_rms[0,6]` profile_spearman=0.52 mean_abs_delta=0.073
- **FAIL** `decod/contrast_rms[0,8]` profile_spearman=0.54 mean_abs_delta=0.040
- **FAIL** `decod/contrast_rms[0,10]` profile_spearman=0.55 mean_abs_delta=0.057
- **FAIL** `decod/contrast_rms[1,8]` profile_spearman=0.50 mean_abs_delta=0.045
- **FAIL** `decod/contrast_rms[1,10]` profile_spearman=0.68 mean_abs_delta=0.039
- **FAIL** `decod/contrast_rms[2,6]` profile_spearman=0.61 mean_abs_delta=0.049
- **FAIL** `decod/contrast_rms[2,8]` profile_spearman=0.61 mean_abs_delta=0.043
- **FAIL** `decod/contrast_rms[2,10]` profile_spearman=0.64 mean_abs_delta=0.035
- **FAIL** `decod/contrast_rms[3,8]` profile_spearman=0.35 mean_abs_delta=0.056
- **FAIL** `decod/contrast_rms[4,7]` profile_spearman=0.65 mean_abs_delta=0.023
- **FAIL** `decod/contrast_rms[4,8]` profile_spearman=0.48 mean_abs_delta=0.040
- **FAIL** `decod/contrast_rms[4,10]` profile_spearman=0.66 mean_abs_delta=0.044
- **FAIL** `decod/contrast_rms[5,6]` profile_spearman=0.61 mean_abs_delta=0.050
- **FAIL** `decod/contrast_rms[5,7]` profile_spearman=0.70 mean_abs_delta=0.039
- **FAIL** `decod/contrast_rms[5,8]` profile_spearman=0.64 mean_abs_delta=0.034
- **FAIL** `decod/contrast_rms[5,10]` profile_spearman=0.59 mean_abs_delta=0.049
- **FAIL** `decod/contrast_rms[6,7]` profile_spearman=0.62 mean_abs_delta=0.054
- **FAIL** `decod/contrast_rms[6,8]` profile_spearman=-0.01 mean_abs_delta=0.076
- **FAIL** `decod/contrast_rms[7,8]` profile_spearman=0.52 mean_abs_delta=0.035
- **FAIL** `decod/contrast_rms[8,9]` profile_spearman=0.44 mean_abs_delta=0.042
- **FAIL** `decod/contrast_rms[8,10]` profile_spearman=0.05 mean_abs_delta=0.073
- **FAIL** `decod/spectral_anisotropy[1,6]` profile_spearman=0.62 mean_abs_delta=0.033
- **FAIL** `decod/spectral_anisotropy[1,7]` profile_spearman=0.64 mean_abs_delta=0.034
- **FAIL** `decod/spectral_anisotropy[1,8]` profile_spearman=0.50 mean_abs_delta=0.027
- **FAIL** `decod/spectral_anisotropy[1,9]` profile_spearman=0.70 mean_abs_delta=0.028
- **FAIL** `decod/spectral_anisotropy[1,10]` profile_spearman=0.68 mean_abs_delta=0.041
- **FAIL** `decod/spectral_anisotropy[2,8]` profile_spearman=0.65 mean_abs_delta=0.022
- **FAIL** `decod/noise_sigma[0,5]` profile_spearman=0.68 mean_abs_delta=0.042
- **FAIL** `decod/noise_sigma[0,6]` profile_spearman=0.33 mean_abs_delta=0.052
- **FAIL** `decod/noise_sigma[0,7]` profile_spearman=0.53 mean_abs_delta=0.041
- **FAIL** `decod/noise_sigma[0,8]` profile_spearman=0.70 mean_abs_delta=0.038
- **FAIL** `decod/noise_sigma[0,9]` profile_spearman=0.61 mean_abs_delta=0.032
- **FAIL** `decod/noise_sigma[0,10]` profile_spearman=0.54 mean_abs_delta=0.041
- **FAIL** `decod/noise_sigma[1,10]` profile_spearman=0.60 mean_abs_delta=0.050
- **FAIL** `decod/noise_sigma[2,6]` profile_spearman=0.67 mean_abs_delta=0.045
- **FAIL** `decod/hue_cos[0,4]` profile_spearman=0.61 mean_abs_delta=0.032
- **FAIL** `decod/hue_cos[1,4]` profile_spearman=0.64 mean_abs_delta=0.045
- **FAIL** `decod/hue_cos[1,9]` profile_spearman=0.68 mean_abs_delta=0.028
- **FAIL** `decod/hue_cos[3,9]` profile_spearman=0.68 mean_abs_delta=0.031
- **FAIL** `decod/edge_density[0,4]` profile_spearman=0.68 mean_abs_delta=0.028
- **FAIL** `decod/edge_density[0,5]` profile_spearman=0.60 mean_abs_delta=0.057
- **FAIL** `decod/edge_density[0,6]` profile_spearman=0.53 mean_abs_delta=0.055
- **FAIL** `decod/edge_density[0,7]` profile_spearman=0.66 mean_abs_delta=0.040
- **FAIL** `decod/edge_density[0,8]` profile_spearman=0.59 mean_abs_delta=0.043
- **FAIL** `decod/edge_density[0,9]` profile_spearman=0.64 mean_abs_delta=0.054
- **FAIL** `decod/edge_density[0,10]` profile_spearman=0.67 mean_abs_delta=0.049
- **FAIL** `decod/edge_density[1,5]` profile_spearman=0.39 mean_abs_delta=0.054
- **FAIL** `decod/edge_density[1,8]` profile_spearman=0.31 mean_abs_delta=0.048
- **FAIL** `decod/edge_density[2,5]` profile_spearman=0.70 mean_abs_delta=0.033
- **FAIL** `decod/edge_density[2,6]` profile_spearman=0.58 mean_abs_delta=0.055
- **FAIL** `decod/edge_density[2,8]` profile_spearman=0.52 mean_abs_delta=0.034
- **FAIL** `decod/edge_density[3,5]` profile_spearman=0.64 mean_abs_delta=0.039
- **FAIL** `decod/edge_density[3,8]` profile_spearman=0.38 mean_abs_delta=0.044
- **FAIL** `decod/edge_density[4,5]` profile_spearman=0.44 mean_abs_delta=0.045
- **FAIL** `decod/edge_density[4,8]` profile_spearman=0.25 mean_abs_delta=0.044
- **FAIL** `decod/edge_density[5,6]` profile_spearman=0.35 mean_abs_delta=0.067
- **FAIL** `decod/edge_density[5,7]` profile_spearman=0.48 mean_abs_delta=0.036
- **FAIL** `decod/edge_density[5,9]` profile_spearman=0.59 mean_abs_delta=0.027
- **FAIL** `decod/edge_density[5,10]` profile_spearman=0.56 mean_abs_delta=0.037
- **FAIL** `decod/edge_density[6,8]` profile_spearman=0.14 mean_abs_delta=0.068
- **FAIL** `decod/edge_density[7,8]` profile_spearman=0.20 mean_abs_delta=0.041
- **FAIL** `decod/edge_density[8,9]` profile_spearman=0.18 mean_abs_delta=0.046
- **FAIL** `decod/edge_density[8,10]` profile_spearman=0.14 mean_abs_delta=0.048
- **FAIL** `decod/severity[4,6]` profile_spearman=0.62 mean_abs_delta=0.043
- PASS: 925 items

## commit_agreement
18 items, 9 FAIL
- **FAIL** `commit/contrast_rms` ['stem', 'stem', 'stem', 'stem', 'stem', 'stem', 'layer2.0', 'stem', 'stem', 'stem', 'layer1.0']
- **FAIL** `commit/highfreq_ratio` ['layer1.0', 'layer1.1', 'layer2.0', 'layer1.0', 'layer2.0', 'layer1.1', 'layer2.0', 'layer1.1', 'layer1.1', 'layer2.0', 'layer2.0']
- **FAIL** `commit/spectral_slope` ['layer2.0', 'layer2.0', 'layer2.0', 'layer2.0', 'layer2.0', 'layer1.1', 'layer2.0', 'layer1.1', 'layer2.0', 'layer2.0', 'layer2.0']
- **FAIL** `commit/spectral_anisotropy` ['layer2.0', 'layer1.2', 'layer2.0', 'layer2.0', 'layer2.0', 'layer2.0', 'layer2.0', 'layer2.1', 'layer3.0', 'layer2.1', 'layer3.0']
- **FAIL** `commit/colorfulness` ['layer1.0', 'stem', 'layer1.0', 'layer1.0', 'stem', 'layer1.0', 'layer1.1', 'layer1.0', 'layer1.0', 'layer1.0', 'layer1.0']
- **FAIL** `commit/edge_density` ['layer1.0', 'layer1.0', 'layer1.0', 'layer1.0', 'layer1.0', 'stem', 'layer1.1', 'layer1.0', 'stem', 'layer2.0', 'layer1.0']
- **FAIL** `commit/orientation_entropy` ['layer2.0', 'layer2.0', 'layer2.0', 'layer3.0', 'layer2.0', 'layer2.0', 'layer3.0', 'layer2.1', 'layer2.2', 'layer2.1', 'layer3.0']
- **FAIL** `commit/corruption_family` ['layer2.0', 'layer2.0', 'layer1.2', 'layer2.0', 'layer2.0', 'layer1.1', 'layer2.0', 'layer2.0', 'layer1.2', 'layer2.0', 'layer1.2']
- **FAIL** `commit/severity` ['layer3.0', 'layer3.0', 'layer2.2', 'layer3.0', 'layer2.0', 'layer3.0', 'layer3.0', 'layer3.0', 'layer3.0', 'layer2.0', 'layer3.0']
- PASS: 9 items

## panel_agreement
98 items, 0 FAIL
- **INFO** `penult/agreement_info[0,1]` {"panel_cka": 0.903, "relrep_row_corr_mean": 0.963, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.924, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.818, "error_consistency_test": 0.552, "cka_ood_c100": 0.68, "relrep_argmax_agree_ood_c100": 0.583, "relrep_argmax_chance_ood_c100": 0.126, "relrep_offmax_corr_ood_c100": 0.766, "landmark_procrustes_disparity": 0.002}
- **INFO** `penult/agreement_info[0,2]` {"panel_cka": 0.919, "relrep_row_corr_mean": 0.968, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.92, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.834, "error_consistency_test": 0.528, "cka_ood_c100": 0.686, "relrep_argmax_agree_ood_c100": 0.585, "relrep_argmax_chance_ood_c100": 0.125, "relrep_offmax_corr_ood_c100": 0.772, "landmark_procrustes_disparity": 0.001}
- **INFO** `penult/agreement_info[0,3]` {"panel_cka": 0.895, "relrep_row_corr_mean": 0.964, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.923, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.827, "error_consistency_test": 0.507, "cka_ood_c100": 0.683, "relrep_argmax_agree_ood_c100": 0.588, "relrep_argmax_chance_ood_c100": 0.126, "relrep_offmax_corr_ood_c100": 0.759, "landmark_procrustes_disparity": 0.003}
- **INFO** `penult/agreement_info[0,4]` {"panel_cka": 0.92, "relrep_row_corr_mean": 0.965, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.932, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.826, "error_consistency_test": 0.565, "cka_ood_c100": 0.687, "relrep_argmax_agree_ood_c100": 0.578, "relrep_argmax_chance_ood_c100": 0.125, "relrep_offmax_corr_ood_c100": 0.77, "landmark_procrustes_disparity": 0.002}
- **INFO** `penult/agreement_info[0,5]` {"panel_cka": 0.908, "relrep_row_corr_mean": 0.96, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.914, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.804, "error_consistency_test": 0.502, "cka_ood_c100": 0.667, "relrep_argmax_agree_ood_c100": 0.555, "relrep_argmax_chance_ood_c100": 0.125, "relrep_offmax_corr_ood_c100": 0.752, "landmark_procrustes_disparity": 0.004}
- **INFO** `penult/agreement_info[0,6]` {"panel_cka": 0.907, "relrep_row_corr_mean": 0.952, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.911, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.765, "error_consistency_test": 0.49, "cka_ood_c100": 0.667, "relrep_argmax_agree_ood_c100": 0.582, "relrep_argmax_chance_ood_c100": 0.124, "relrep_offmax_corr_ood_c100": 0.741, "landmark_procrustes_disparity": 0.005}
- **INFO** `penult/agreement_info[0,7]` {"panel_cka": 0.907, "relrep_row_corr_mean": 0.963, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.92, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.782, "error_consistency_test": 0.509, "cka_ood_c100": 0.662, "relrep_argmax_agree_ood_c100": 0.549, "relrep_argmax_chance_ood_c100": 0.125, "relrep_offmax_corr_ood_c100": 0.751, "landmark_procrustes_disparity": 0.005}
- **INFO** `penult/agreement_info[0,8]` {"panel_cka": 0.894, "relrep_row_corr_mean": 0.946, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.918, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.705, "error_consistency_test": 0.516, "cka_ood_c100": 0.669, "relrep_argmax_agree_ood_c100": 0.555, "relrep_argmax_chance_ood_c100": 0.127, "relrep_offmax_corr_ood_c100": 0.745, "landmark_procrustes_disparity": 0.009}
- **INFO** `penult/agreement_info[0,9]` {"panel_cka": 0.905, "relrep_row_corr_mean": 0.955, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.908, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.765, "error_consistency_test": 0.463, "cka_ood_c100": 0.656, "relrep_argmax_agree_ood_c100": 0.583, "relrep_argmax_chance_ood_c100": 0.123, "relrep_offmax_corr_ood_c100": 0.732, "landmark_procrustes_disparity": 0.005}
- **INFO** `penult/agreement_info[0,10]` {"panel_cka": 0.909, "relrep_row_corr_mean": 0.956, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.923, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.781, "error_consistency_test": 0.54, "cka_ood_c100": 0.662, "relrep_argmax_agree_ood_c100": 0.567, "relrep_argmax_chance_ood_c100": 0.125, "relrep_offmax_corr_ood_c100": 0.738, "landmark_procrustes_disparity": 0.005}
- **INFO** `penult/agreement_info[1,2]` {"panel_cka": 0.91, "relrep_row_corr_mean": 0.963, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.926, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.831, "error_consistency_test": 0.535, "cka_ood_c100": 0.694, "relrep_argmax_agree_ood_c100": 0.589, "relrep_argmax_chance_ood_c100": 0.126, "relrep_offmax_corr_ood_c100": 0.779, "landmark_procrustes_disparity": 0.002}
- **INFO** `penult/agreement_info[1,3]` {"panel_cka": 0.899, "relrep_row_corr_mean": 0.962, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.929, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.82, "error_consistency_test": 0.537, "cka_ood_c100": 0.693, "relrep_argmax_agree_ood_c100": 0.577, "relrep_argmax_chance_ood_c100": 0.126, "relrep_offmax_corr_ood_c100": 0.772, "landmark_procrustes_disparity": 0.004}
- **INFO** `penult/agreement_info[1,4]` {"panel_cka": 0.905, "relrep_row_corr_mean": 0.961, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.934, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.826, "error_consistency_test": 0.526, "cka_ood_c100": 0.695, "relrep_argmax_agree_ood_c100": 0.571, "relrep_argmax_chance_ood_c100": 0.125, "relrep_offmax_corr_ood_c100": 0.776, "landmark_procrustes_disparity": 0.001}
- **INFO** `penult/agreement_info[1,5]` {"panel_cka": 0.919, "relrep_row_corr_mean": 0.962, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.924, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.813, "error_consistency_test": 0.523, "cka_ood_c100": 0.674, "relrep_argmax_agree_ood_c100": 0.568, "relrep_argmax_chance_ood_c100": 0.126, "relrep_offmax_corr_ood_c100": 0.759, "landmark_procrustes_disparity": 0.004}
- **INFO** `penult/agreement_info[1,6]` {"panel_cka": 0.881, "relrep_row_corr_mean": 0.949, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.92, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.772, "error_consistency_test": 0.511, "cka_ood_c100": 0.676, "relrep_argmax_agree_ood_c100": 0.585, "relrep_argmax_chance_ood_c100": 0.124, "relrep_offmax_corr_ood_c100": 0.742, "landmark_procrustes_disparity": 0.004}
- **INFO** `penult/agreement_info[1,7]` {"panel_cka": 0.91, "relrep_row_corr_mean": 0.958, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.924, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.788, "error_consistency_test": 0.537, "cka_ood_c100": 0.671, "relrep_argmax_agree_ood_c100": 0.57, "relrep_argmax_chance_ood_c100": 0.126, "relrep_offmax_corr_ood_c100": 0.748, "landmark_procrustes_disparity": 0.003}
- **INFO** `penult/agreement_info[1,8]` {"panel_cka": 0.895, "relrep_row_corr_mean": 0.946, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.925, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.727, "error_consistency_test": 0.523, "cka_ood_c100": 0.665, "relrep_argmax_agree_ood_c100": 0.578, "relrep_argmax_chance_ood_c100": 0.127, "relrep_offmax_corr_ood_c100": 0.739, "landmark_procrustes_disparity": 0.007}
- **INFO** `penult/agreement_info[1,9]` {"panel_cka": 0.896, "relrep_row_corr_mean": 0.954, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.927, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.777, "error_consistency_test": 0.512, "cka_ood_c100": 0.663, "relrep_argmax_agree_ood_c100": 0.588, "relrep_argmax_chance_ood_c100": 0.123, "relrep_offmax_corr_ood_c100": 0.74, "landmark_procrustes_disparity": 0.004}
- **INFO** `penult/agreement_info[1,10]` {"panel_cka": 0.89, "relrep_row_corr_mean": 0.953, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.926, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.777, "error_consistency_test": 0.518, "cka_ood_c100": 0.679, "relrep_argmax_agree_ood_c100": 0.569, "relrep_argmax_chance_ood_c100": 0.125, "relrep_offmax_corr_ood_c100": 0.75, "landmark_procrustes_disparity": 0.003}
- **INFO** `penult/agreement_info[2,3]` {"panel_cka": 0.916, "relrep_row_corr_mean": 0.967, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.929, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.839, "error_consistency_test": 0.527, "cka_ood_c100": 0.7, "relrep_argmax_agree_ood_c100": 0.575, "relrep_argmax_chance_ood_c100": 0.125, "relrep_offmax_corr_ood_c100": 0.778, "landmark_procrustes_disparity": 0.003}
- **INFO** `penult/agreement_info[2,4]` {"panel_cka": 0.912, "relrep_row_corr_mean": 0.962, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.933, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.833, "error_consistency_test": 0.539, "cka_ood_c100": 0.699, "relrep_argmax_agree_ood_c100": 0.572, "relrep_argmax_chance_ood_c100": 0.124, "relrep_offmax_corr_ood_c100": 0.773, "landmark_procrustes_disparity": 0.002}
- **INFO** `penult/agreement_info[2,5]` {"panel_cka": 0.923, "relrep_row_corr_mean": 0.962, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.925, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.814, "error_consistency_test": 0.543, "cka_ood_c100": 0.685, "relrep_argmax_agree_ood_c100": 0.563, "relrep_argmax_chance_ood_c100": 0.125, "relrep_offmax_corr_ood_c100": 0.76, "landmark_procrustes_disparity": 0.004}
- **INFO** `penult/agreement_info[2,6]` {"panel_cka": 0.907, "relrep_row_corr_mean": 0.951, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.924, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.764, "error_consistency_test": 0.508, "cka_ood_c100": 0.677, "relrep_argmax_agree_ood_c100": 0.589, "relrep_argmax_chance_ood_c100": 0.123, "relrep_offmax_corr_ood_c100": 0.741, "landmark_procrustes_disparity": 0.006}
- **INFO** `penult/agreement_info[2,7]` {"panel_cka": 0.923, "relrep_row_corr_mean": 0.962, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.928, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.797, "error_consistency_test": 0.566, "cka_ood_c100": 0.672, "relrep_argmax_agree_ood_c100": 0.576, "relrep_argmax_chance_ood_c100": 0.125, "relrep_offmax_corr_ood_c100": 0.758, "landmark_procrustes_disparity": 0.005}
- **INFO** `penult/agreement_info[2,8]` {"panel_cka": 0.912, "relrep_row_corr_mean": 0.949, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.928, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.715, "error_consistency_test": 0.566, "cka_ood_c100": 0.672, "relrep_argmax_agree_ood_c100": 0.582, "relrep_argmax_chance_ood_c100": 0.126, "relrep_offmax_corr_ood_c100": 0.744, "landmark_procrustes_disparity": 0.009}
- **INFO** `penult/agreement_info[2,9]` {"panel_cka": 0.909, "relrep_row_corr_mean": 0.954, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.922, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.772, "error_consistency_test": 0.525, "cka_ood_c100": 0.672, "relrep_argmax_agree_ood_c100": 0.593, "relrep_argmax_chance_ood_c100": 0.123, "relrep_offmax_corr_ood_c100": 0.743, "landmark_procrustes_disparity": 0.006}
- **INFO** `penult/agreement_info[2,10]` {"panel_cka": 0.915, "relrep_row_corr_mean": 0.959, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.929, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.78, "error_consistency_test": 0.554, "cka_ood_c100": 0.681, "relrep_argmax_agree_ood_c100": 0.575, "relrep_argmax_chance_ood_c100": 0.124, "relrep_offmax_corr_ood_c100": 0.751, "landmark_procrustes_disparity": 0.005}
- **INFO** `penult/agreement_info[3,4]` {"panel_cka": 0.895, "relrep_row_corr_mean": 0.961, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.938, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.831, "error_consistency_test": 0.581, "cka_ood_c100": 0.708, "relrep_argmax_agree_ood_c100": 0.581, "relrep_argmax_chance_ood_c100": 0.124, "relrep_offmax_corr_ood_c100": 0.781, "landmark_procrustes_disparity": 0.003}
- **INFO** `penult/agreement_info[3,5]` {"panel_cka": 0.897, "relrep_row_corr_mean": 0.96, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.919, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.804, "error_consistency_test": 0.515, "cka_ood_c100": 0.684, "relrep_argmax_agree_ood_c100": 0.56, "relrep_argmax_chance_ood_c100": 0.125, "relrep_offmax_corr_ood_c100": 0.76, "landmark_procrustes_disparity": 0.005}
- **INFO** `penult/agreement_info[3,6]` {"panel_cka": 0.886, "relrep_row_corr_mean": 0.953, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.921, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.752, "error_consistency_test": 0.503, "cka_ood_c100": 0.675, "relrep_argmax_agree_ood_c100": 0.563, "relrep_argmax_chance_ood_c100": 0.123, "relrep_offmax_corr_ood_c100": 0.737, "landmark_procrustes_disparity": 0.008}
- **INFO** `penult/agreement_info[3,7]` {"panel_cka": 0.908, "relrep_row_corr_mean": 0.96, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.926, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.787, "error_consistency_test": 0.544, "cka_ood_c100": 0.671, "relrep_argmax_agree_ood_c100": 0.557, "relrep_argmax_chance_ood_c100": 0.125, "relrep_offmax_corr_ood_c100": 0.752, "landmark_procrustes_disparity": 0.007}
- **INFO** `penult/agreement_info[3,8]` {"panel_cka": 0.882, "relrep_row_corr_mean": 0.949, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.925, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.711, "error_consistency_test": 0.529, "cka_ood_c100": 0.672, "relrep_argmax_agree_ood_c100": 0.568, "relrep_argmax_chance_ood_c100": 0.127, "relrep_offmax_corr_ood_c100": 0.741, "landmark_procrustes_disparity": 0.01}
- **INFO** `penult/agreement_info[3,9]` {"panel_cka": 0.887, "relrep_row_corr_mean": 0.956, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.927, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.771, "error_consistency_test": 0.571, "cka_ood_c100": 0.661, "relrep_argmax_agree_ood_c100": 0.581, "relrep_argmax_chance_ood_c100": 0.122, "relrep_offmax_corr_ood_c100": 0.73, "landmark_procrustes_disparity": 0.007}
- **INFO** `penult/agreement_info[3,10]` {"panel_cka": 0.897, "relrep_row_corr_mean": 0.955, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.925, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.767, "error_consistency_test": 0.518, "cka_ood_c100": 0.68, "relrep_argmax_agree_ood_c100": 0.56, "relrep_argmax_chance_ood_c100": 0.124, "relrep_offmax_corr_ood_c100": 0.744, "landmark_procrustes_disparity": 0.007}
- **INFO** `penult/agreement_info[4,5]` {"panel_cka": 0.909, "relrep_row_corr_mean": 0.959, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.925, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.81, "error_consistency_test": 0.505, "cka_ood_c100": 0.671, "relrep_argmax_agree_ood_c100": 0.559, "relrep_argmax_chance_ood_c100": 0.123, "relrep_offmax_corr_ood_c100": 0.757, "landmark_procrustes_disparity": 0.004}
- **INFO** `penult/agreement_info[4,6]` {"panel_cka": 0.903, "relrep_row_corr_mean": 0.951, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.929, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.761, "error_consistency_test": 0.484, "cka_ood_c100": 0.67, "relrep_argmax_agree_ood_c100": 0.576, "relrep_argmax_chance_ood_c100": 0.122, "relrep_offmax_corr_ood_c100": 0.742, "landmark_procrustes_disparity": 0.005}
- **INFO** `penult/agreement_info[4,7]` {"panel_cka": 0.903, "relrep_row_corr_mean": 0.959, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.931, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.788, "error_consistency_test": 0.534, "cka_ood_c100": 0.666, "relrep_argmax_agree_ood_c100": 0.555, "relrep_argmax_chance_ood_c100": 0.123, "relrep_offmax_corr_ood_c100": 0.751, "landmark_procrustes_disparity": 0.004}
- **INFO** `penult/agreement_info[4,8]` {"panel_cka": 0.9, "relrep_row_corr_mean": 0.951, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.928, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.711, "error_consistency_test": 0.519, "cka_ood_c100": 0.664, "relrep_argmax_agree_ood_c100": 0.571, "relrep_argmax_chance_ood_c100": 0.126, "relrep_offmax_corr_ood_c100": 0.742, "landmark_procrustes_disparity": 0.009}
- **INFO** `penult/agreement_info[4,9]` {"panel_cka": 0.891, "relrep_row_corr_mean": 0.955, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.926, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.765, "error_consistency_test": 0.492, "cka_ood_c100": 0.664, "relrep_argmax_agree_ood_c100": 0.575, "relrep_argmax_chance_ood_c100": 0.121, "relrep_offmax_corr_ood_c100": 0.734, "landmark_procrustes_disparity": 0.006}
- **INFO** `penult/agreement_info[4,10]` {"panel_cka": 0.914, "relrep_row_corr_mean": 0.956, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.93, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.767, "error_consistency_test": 0.514, "cka_ood_c100": 0.671, "relrep_argmax_agree_ood_c100": 0.561, "relrep_argmax_chance_ood_c100": 0.123, "relrep_offmax_corr_ood_c100": 0.745, "landmark_procrustes_disparity": 0.005}
- **WARN** `panel[5,6]` no compare_vs_* deformation.json
- **WARN** `panel[5,7]` no compare_vs_* deformation.json
- **WARN** `panel[5,8]` no compare_vs_* deformation.json
- **WARN** `panel[5,9]` no compare_vs_* deformation.json
- **WARN** `panel[5,10]` no compare_vs_* deformation.json
- **WARN** `panel[6,7]` no compare_vs_* deformation.json
- **WARN** `panel[6,8]` no compare_vs_* deformation.json
- **WARN** `panel[6,9]` no compare_vs_* deformation.json
- **WARN** `panel[6,10]` no compare_vs_* deformation.json
- **WARN** `panel[7,8]` no compare_vs_* deformation.json
- **INFO** `penult/agreement_info[7,9]` {"panel_cka": 0.925, "relrep_row_corr_mean": 0.965, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.926, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.786, "error_consistency_test": 0.549, "cka_ood_c100": 0.684, "relrep_argmax_agree_ood_c100": 0.584, "relrep_argmax_chance_ood_c100": 0.122, "relrep_offmax_corr_ood_c100": 0.768, "landmark_procrustes_disparity": 0.003}
- **INFO** `penult/agreement_info[7,10]` {"panel_cka": 0.922, "relrep_row_corr_mean": 0.96, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.931, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.768, "error_consistency_test": 0.568, "cka_ood_c100": 0.694, "relrep_argmax_agree_ood_c100": 0.576, "relrep_argmax_chance_ood_c100": 0.124, "relrep_offmax_corr_ood_c100": 0.77, "landmark_procrustes_disparity": 0.003}
- **WARN** `panel[8,9]` no compare_vs_* deformation.json
- **WARN** `panel[8,10]` no compare_vs_* deformation.json
- **INFO** `penult/agreement_info[9,10]` {"panel_cka": 0.919, "relrep_row_corr_mean": 0.96, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.925, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.79, "error_consistency_test": 0.537, "cka_ood_c100": 0.686, "relrep_argmax_agree_ood_c100": 0.589, "relrep_argmax_chance_ood_c100": 0.122, "relrep_offmax_corr_ood_c100": 0.763, "landmark_procrustes_disparity": 0.003}
- PASS: 43 items
