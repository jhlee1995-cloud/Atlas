# CRITIC  verdict: **PARTIAL**
runs: ['results/atlas_v1_resnet20_s0hub_st3', 'results/atlas_v1_resnet20_s1_st3', 'results/atlas_v1_resnet20_s2_st3', 'results/atlas_v1_resnet20_s3_st3', 'results/atlas_v1_resnet20_s4_st3', 'results/atlas_v1_resnet56_s1', 'results/atlas_v1_resnet56_s2']
counts: {'PASS': 870, 'FAIL': 84, 'WARN': 1, 'INFO': 21}
layer alignment: position {"atlas_v1_resnet56_s1": [["stem", "stem"], ["layer1.0", "layer1.0"], ["layer1.1", "layer1.5"], ["layer1.2", "layer1.8"], ["layer2.0", "layer2.0"], ["layer2.1", "layer2.5"], ["layer2.2", "layer2.8"], ["layer3.0", "layer3.0"], ["layer3.1", "layer3.5"], ["layer3.2", "layer3.8"], ["penult", "penult"]], "atlas_v1_resnet56_s2": [["stem", "stem"], ["layer1.0", "layer1.0"], ["layer1.1", "layer1.5"], ["layer1.2", "layer1.8"], ["layer2.0", "layer2.0"], ["layer2.1", "layer2.5"], ["layer2.2", "layer2.8"], ["layer3.0", "layer3.0"], ["layer3.1", "layer3.5"], ["layer3.2", "layer3.8"], ["penult", "penult"]]}

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
7 items, 0 FAIL
- PASS: 7 items

## probe_hygiene
7 items, 0 FAIL
- PASS: 7 items

## scalar_stability
80 items, 41 FAIL
- **FAIL** `stem/pca_spectrum.participation_ratio` values=[2.979, 2.993, 2.975, 2.644, 3.09, 2.643, 2.74] rel_spread=0.156
- **FAIL** `stem/pca_spectrum.dim95` values=[4.0, 4.0, 4.0, 4.0, 4.0, 3.0, 4.0] rel_spread=0.259
- **FAIL** `stem/neural_collapse.nc1` values=[259.926, 261.068, 152.924, 318.793, 159.182, 499.245, 213.957] rel_spread=1.300
- **FAIL** `stem/hubness.k_occurrence_skew` values=[0.212, 0.179, 0.209, 0.223, 0.202, 0.096, 0.165] rel_spread=0.691
- **FAIL** `layer1.0/twonn_id.id` values=[8.83, 8.577, 9.201, 8.83, 9.25, 7.423, 8.157] rel_spread=0.212
- **FAIL** `layer1.0/pca_spectrum.participation_ratio` values=[4.042, 3.288, 3.875, 4.303, 3.95, 2.623, 3.577] rel_spread=0.458
- **FAIL** `layer1.0/pca_spectrum.dim95` values=[6.0, 5.0, 7.0, 7.0, 7.0, 4.0, 5.0] rel_spread=0.512
- **FAIL** `layer1.0/neural_collapse.nc1` values=[138.342, 164.627, 136.168, 123.091, 93.88, 264.537, 133.13] rel_spread=1.134
- **FAIL** `layer1.0/neural_collapse.etf_deviation` values=[0.581, 0.635, 0.613, 0.561, 0.519, 0.658, 0.6] rel_spread=0.235
- **FAIL** `layer1.0/hubness.k_occurrence_skew` values=[0.631, 0.535, 0.741, 0.642, 0.74, 0.271, 0.388] rel_spread=0.834
- **FAIL** `layer1.1/pca_spectrum.participation_ratio` values=[5.171, 4.07, 5.067, 4.944, 4.893, 4.753, 5.67] rel_spread=0.324
- **FAIL** `layer1.1/pca_spectrum.dim95` values=[8.0, 7.0, 8.0, 8.0, 8.0, 7.0, 9.0] rel_spread=0.255
- **FAIL** `layer1.1/class_centers.sep_ratio` values=[0.21, 0.209, 0.209, 0.227, 0.215, 0.235, 0.297] rel_spread=0.386
- **FAIL** `layer1.1/neural_collapse.nc1` values=[57.975, 74.307, 68.463, 69.758, 86.846, 68.49, 86.672] rel_spread=0.394
- **FAIL** `layer1.1/neural_collapse.etf_deviation` values=[0.494, 0.544, 0.575, 0.532, 0.46, 0.51, 0.473] rel_spread=0.225
- **FAIL** `layer1.1/hubness.k_occurrence_skew` values=[0.797, 0.644, 0.77, 0.815, 0.8, 0.699, 0.777] rel_spread=0.226
- **FAIL** `layer1.2/pca_spectrum.participation_ratio` values=[5.518, 4.8, 4.499, 5.286, 5.102, 5.267, 5.481] rel_spread=0.198
- **FAIL** `layer1.2/pca_spectrum.dim95` values=[9.0, 8.0, 9.0, 9.0, 9.0, 8.0, 10.0] rel_spread=0.226
- **FAIL** `layer1.2/neural_collapse.nc1` values=[53.174, 49.246, 47.77, 48.912, 48.318, 52.914, 66.033] rel_spread=0.349
- **FAIL** `layer1.2/hubness.k_occurrence_skew` values=[0.871, 0.713, 0.821, 0.767, 0.922, 0.801, 0.903] rel_spread=0.252
- **FAIL** `layer2.0/pca_spectrum.participation_ratio` values=[6.816, 6.067, 5.598, 5.852, 6.145, 5.854, 5.984] rel_spread=0.201
- **FAIL** `layer2.0/pca_spectrum.dim95` values=[13.0, 12.0, 11.0, 12.0, 12.0, 10.0, 11.0] rel_spread=0.259
- **FAIL** `layer2.0/neural_collapse.nc1` values=[21.892, 23.829, 19.416, 24.574, 20.88, 30.982, 16.817] rel_spread=0.626
- **FAIL** `layer2.0/hubness.k_occurrence_skew` values=[0.962, 0.924, 0.913, 0.975, 1.008, 0.835, 0.992] rel_spread=0.183
- **FAIL** `layer2.1/neural_collapse.nc1` values=[12.411, 17.043, 11.104, 15.979, 14.636, 12.951, 9.034] rel_spread=0.602
- **FAIL** `layer2.2/neural_collapse.nc1` values=[7.065, 9.177, 7.357, 8.82, 8.825, 7.266, 5.407] rel_spread=0.490
- **FAIL** `layer2.2/hubness.k_occurrence_skew` values=[1.241, 1.427, 1.442, 1.419, 1.421, 1.641, 1.416] rel_spread=0.279
- **FAIL** `layer3.0/twonn_id.id` values=[18.102, 18.012, 18.284, 17.981, 18.308, 15.481, 15.535] rel_spread=0.163
- **FAIL** `layer3.0/pca_spectrum.participation_ratio` values=[12.136, 10.286, 12.091, 11.533, 10.434, 9.492, 9.328] rel_spread=0.261
- **FAIL** `layer3.0/pca_spectrum.dim95` values=[37.0, 38.0, 38.0, 37.0, 37.0, 25.0, 24.0] rel_spread=0.415
- **FAIL** `layer3.0/neural_collapse.nc1` values=[2.416, 2.391, 2.533, 2.741, 2.396, 3.49, 3.335] rel_spread=0.398
- **FAIL** `layer3.0/neural_collapse.etf_deviation` values=[0.308, 0.329, 0.334, 0.355, 0.338, 0.357, 0.365] rel_spread=0.167
- **FAIL** `layer3.0/hubness.k_occurrence_skew` values=[1.829, 1.691, 1.701, 1.87, 1.874, 1.327, 1.476] rel_spread=0.325
- **FAIL** `layer3.1/pca_spectrum.participation_ratio` values=[10.468, 11.259, 9.641, 12.61, 9.453, 15.661, 14.729] rel_spread=0.518
- **FAIL** `layer3.1/neural_collapse.nc1` values=[0.759, 0.774, 0.859, 0.923, 0.792, 0.676, 0.61] rel_spread=0.406
- **FAIL** `layer3.1/neural_collapse.etf_deviation` values=[0.275, 0.289, 0.29, 0.277, 0.296, 0.216, 0.187] rel_spread=0.416
- **FAIL** `layer3.1/hubness.k_occurrence_skew` values=[1.907, 1.898, 1.805, 1.755, 1.86, 2.042, 1.745] rel_spread=0.160
- **FAIL** `penult/class_centers.sep_ratio` values=[2.996, 3.067, 3.021, 2.97, 3.024, 5.114, 5.213] rel_spread=0.618
- **FAIL** `penult/neural_collapse.nc1` values=[0.17, 0.163, 0.17, 0.172, 0.167, 0.053, 0.05] rel_spread=0.901
- **FAIL** `penult/neural_collapse.etf_deviation` values=[0.107, 0.098, 0.107, 0.109, 0.097, 0.057, 0.062] rel_spread=0.570
- **FAIL** `penult/hubness.k_occurrence_skew` values=[0.763, 0.833, 0.767, 0.772, 0.804, 0.891, 0.855] rel_spread=0.158
- PASS: 39 items

## id_profile_stability
21 items, 0 FAIL
- PASS: 21 items

## adjacency_stability
420 items, 16 FAIL
- **FAIL** `layer2.1/merge_kendall[0,6]` tau=0.501
- **FAIL** `layer2.1/merge_kendall[1,6]` tau=0.547
- **FAIL** `layer2.1/merge_kendall[2,6]` tau=0.545
- **FAIL** `layer2.1/merge_kendall[3,6]` tau=0.566
- **FAIL** `layer2.1/merge_kendall[4,6]` tau=0.494
- **FAIL** `layer2.1/merge_kendall[5,6]` tau=0.508
- **FAIL** `layer2.2/merge_kendall[0,5]` tau=0.463
- **FAIL** `layer2.2/merge_kendall[0,6]` tau=0.529
- **FAIL** `layer2.2/merge_kendall[1,5]` tau=0.501
- **FAIL** `layer2.2/merge_kendall[1,6]` tau=0.536
- **FAIL** `layer2.2/merge_kendall[2,5]` tau=0.504
- **FAIL** `layer2.2/merge_kendall[2,6]` tau=0.539
- **FAIL** `layer2.2/merge_kendall[3,5]` tau=0.501
- **FAIL** `layer2.2/merge_kendall[3,6]` tau=0.536
- **FAIL** `layer2.2/merge_kendall[4,5]` tau=0.510
- **FAIL** `layer2.2/merge_kendall[4,6]` tau=0.540
- PASS: 404 items

## decodability_stability
378 items, 22 FAIL
- **FAIL** `decod/contrast_rms[0,6]` profile_spearman=0.45 mean_abs_delta=0.046
- **FAIL** `decod/contrast_rms[1,6]` profile_spearman=0.59 mean_abs_delta=0.046
- **FAIL** `decod/contrast_rms[2,5]` profile_spearman=0.66 mean_abs_delta=0.037
- **FAIL** `decod/contrast_rms[2,6]` profile_spearman=0.68 mean_abs_delta=0.040
- **FAIL** `decod/contrast_rms[3,6]` profile_spearman=0.56 mean_abs_delta=0.053
- **FAIL** `decod/contrast_rms[4,6]` profile_spearman=0.55 mean_abs_delta=0.039
- **FAIL** `decod/contrast_rms[5,6]` profile_spearman=0.41 mean_abs_delta=0.048
- **FAIL** `decod/noise_sigma[0,5]` profile_spearman=0.62 mean_abs_delta=0.038
- **FAIL** `decod/saturation_mean[0,5]` profile_spearman=0.48 mean_abs_delta=0.074
- **FAIL** `decod/saturation_mean[1,5]` profile_spearman=0.66 mean_abs_delta=0.073
- **FAIL** `decod/saturation_mean[2,5]` profile_spearman=0.67 mean_abs_delta=0.076
- **FAIL** `decod/saturation_mean[5,6]` profile_spearman=0.53 mean_abs_delta=0.082
- **FAIL** `decod/hue_cos[0,4]` profile_spearman=0.61 mean_abs_delta=0.032
- **FAIL** `decod/hue_cos[1,4]` profile_spearman=0.64 mean_abs_delta=0.045
- **FAIL** `decod/hue_cos[4,5]` profile_spearman=0.54 mean_abs_delta=0.044
- **FAIL** `decod/hue_cos[4,6]` profile_spearman=0.60 mean_abs_delta=0.027
- **FAIL** `decod/colorfulness[4,6]` profile_spearman=0.70 mean_abs_delta=0.045
- **FAIL** `decod/colorfulness[5,6]` profile_spearman=0.61 mean_abs_delta=0.071
- **FAIL** `decod/edge_density[0,4]` profile_spearman=0.68 mean_abs_delta=0.028
- **FAIL** `decod/edge_density[0,5]` profile_spearman=0.70 mean_abs_delta=0.036
- **FAIL** `decod/edge_density[2,5]` profile_spearman=0.62 mean_abs_delta=0.048
- **FAIL** `decod/edge_density[5,6]` profile_spearman=0.65 mean_abs_delta=0.043
- PASS: 356 items

## commit_agreement
18 items, 5 FAIL
- **FAIL** `commit/contrast_rms` ['stem', 'stem', 'stem', 'stem', 'stem', 'layer1.1', 'stem']
- **FAIL** `commit/highfreq_ratio` ['layer1.0', 'layer1.1', 'layer2.0', 'layer1.0', 'layer2.0', 'layer2.0', 'layer2.0']
- **FAIL** `commit/edge_density` ['layer1.0', 'layer1.0', 'layer1.0', 'layer1.0', 'layer1.0', 'layer1.1', 'stem']
- **FAIL** `commit/orientation_entropy` ['layer2.0', 'layer2.0', 'layer2.0', 'layer3.0', 'layer2.0', 'layer2.1', 'layer2.0']
- **FAIL** `commit/severity` ['layer3.0', 'layer3.0', 'layer2.2', 'layer3.0', 'layer2.0', 'layer2.0', 'layer2.0']
- PASS: 13 items

## panel_agreement
42 items, 0 FAIL
- **INFO** `penult/agreement_info[0,1]` {"panel_cka": 0.903, "relrep_row_corr_mean": 0.963, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.924, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.818, "error_consistency_test": 0.552, "cka_ood_c100": 0.68, "relrep_argmax_agree_ood_c100": 0.583, "relrep_argmax_chance_ood_c100": 0.126, "relrep_offmax_corr_ood_c100": 0.766, "landmark_procrustes_disparity": 0.002}
- **INFO** `penult/agreement_info[0,2]` {"panel_cka": 0.919, "relrep_row_corr_mean": 0.968, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.92, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.834, "error_consistency_test": 0.528, "cka_ood_c100": 0.686, "relrep_argmax_agree_ood_c100": 0.585, "relrep_argmax_chance_ood_c100": 0.125, "relrep_offmax_corr_ood_c100": 0.772, "landmark_procrustes_disparity": 0.001}
- **INFO** `penult/agreement_info[0,3]` {"panel_cka": 0.895, "relrep_row_corr_mean": 0.964, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.923, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.827, "error_consistency_test": 0.507, "cka_ood_c100": 0.683, "relrep_argmax_agree_ood_c100": 0.588, "relrep_argmax_chance_ood_c100": 0.126, "relrep_offmax_corr_ood_c100": 0.759, "landmark_procrustes_disparity": 0.003}
- **INFO** `penult/agreement_info[0,4]` {"panel_cka": 0.92, "relrep_row_corr_mean": 0.965, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.932, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.826, "error_consistency_test": 0.565, "cka_ood_c100": 0.687, "relrep_argmax_agree_ood_c100": 0.578, "relrep_argmax_chance_ood_c100": 0.125, "relrep_offmax_corr_ood_c100": 0.77, "landmark_procrustes_disparity": 0.002}
- **INFO** `penult/agreement_info[0,5]` {"panel_cka": 0.885, "relrep_row_corr_mean": 0.954, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.93, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.759, "error_consistency_test": 0.499, "cka_ood_c100": 0.656, "relrep_argmax_agree_ood_c100": 0.574, "relrep_argmax_chance_ood_c100": 0.126, "relrep_offmax_corr_ood_c100": 0.739, "landmark_procrustes_disparity": 0.01}
- **INFO** `penult/agreement_info[0,6]` {"panel_cka": 0.908, "relrep_row_corr_mean": 0.955, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.922, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.772, "error_consistency_test": 0.569, "cka_ood_c100": 0.641, "relrep_argmax_agree_ood_c100": 0.581, "relrep_argmax_chance_ood_c100": 0.124, "relrep_offmax_corr_ood_c100": 0.722, "landmark_procrustes_disparity": 0.01}
- **INFO** `penult/agreement_info[1,2]` {"panel_cka": 0.91, "relrep_row_corr_mean": 0.963, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.926, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.831, "error_consistency_test": 0.535, "cka_ood_c100": 0.694, "relrep_argmax_agree_ood_c100": 0.589, "relrep_argmax_chance_ood_c100": 0.126, "relrep_offmax_corr_ood_c100": 0.779, "landmark_procrustes_disparity": 0.002}
- **INFO** `penult/agreement_info[1,3]` {"panel_cka": 0.899, "relrep_row_corr_mean": 0.962, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.929, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.82, "error_consistency_test": 0.537, "cka_ood_c100": 0.693, "relrep_argmax_agree_ood_c100": 0.577, "relrep_argmax_chance_ood_c100": 0.126, "relrep_offmax_corr_ood_c100": 0.772, "landmark_procrustes_disparity": 0.004}
- **INFO** `penult/agreement_info[1,4]` {"panel_cka": 0.905, "relrep_row_corr_mean": 0.961, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.934, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.826, "error_consistency_test": 0.526, "cka_ood_c100": 0.695, "relrep_argmax_agree_ood_c100": 0.571, "relrep_argmax_chance_ood_c100": 0.125, "relrep_offmax_corr_ood_c100": 0.776, "landmark_procrustes_disparity": 0.001}
- **INFO** `penult/agreement_info[1,5]` {"panel_cka": 0.898, "relrep_row_corr_mean": 0.949, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.94, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.755, "error_consistency_test": 0.523, "cka_ood_c100": 0.656, "relrep_argmax_agree_ood_c100": 0.585, "relrep_argmax_chance_ood_c100": 0.126, "relrep_offmax_corr_ood_c100": 0.737, "landmark_procrustes_disparity": 0.008}
- **INFO** `penult/agreement_info[1,6]` {"panel_cka": 0.9, "relrep_row_corr_mean": 0.952, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.927, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.773, "error_consistency_test": 0.489, "cka_ood_c100": 0.643, "relrep_argmax_agree_ood_c100": 0.585, "relrep_argmax_chance_ood_c100": 0.125, "relrep_offmax_corr_ood_c100": 0.731, "landmark_procrustes_disparity": 0.007}
- **INFO** `penult/agreement_info[2,3]` {"panel_cka": 0.916, "relrep_row_corr_mean": 0.967, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.929, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.839, "error_consistency_test": 0.527, "cka_ood_c100": 0.7, "relrep_argmax_agree_ood_c100": 0.575, "relrep_argmax_chance_ood_c100": 0.125, "relrep_offmax_corr_ood_c100": 0.778, "landmark_procrustes_disparity": 0.003}
- **INFO** `penult/agreement_info[2,4]` {"panel_cka": 0.912, "relrep_row_corr_mean": 0.962, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.933, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.833, "error_consistency_test": 0.539, "cka_ood_c100": 0.699, "relrep_argmax_agree_ood_c100": 0.572, "relrep_argmax_chance_ood_c100": 0.124, "relrep_offmax_corr_ood_c100": 0.773, "landmark_procrustes_disparity": 0.002}
- **INFO** `penult/agreement_info[2,5]` {"panel_cka": 0.911, "relrep_row_corr_mean": 0.952, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.936, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.757, "error_consistency_test": 0.528, "cka_ood_c100": 0.659, "relrep_argmax_agree_ood_c100": 0.594, "relrep_argmax_chance_ood_c100": 0.125, "relrep_offmax_corr_ood_c100": 0.739, "landmark_procrustes_disparity": 0.012}
- **INFO** `penult/agreement_info[2,6]` {"panel_cka": 0.919, "relrep_row_corr_mean": 0.955, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.932, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.774, "error_consistency_test": 0.552, "cka_ood_c100": 0.646, "relrep_argmax_agree_ood_c100": 0.597, "relrep_argmax_chance_ood_c100": 0.124, "relrep_offmax_corr_ood_c100": 0.731, "landmark_procrustes_disparity": 0.011}
- **INFO** `penult/agreement_info[3,4]` {"panel_cka": 0.895, "relrep_row_corr_mean": 0.961, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.938, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.831, "error_consistency_test": 0.581, "cka_ood_c100": 0.708, "relrep_argmax_agree_ood_c100": 0.581, "relrep_argmax_chance_ood_c100": 0.124, "relrep_offmax_corr_ood_c100": 0.781, "landmark_procrustes_disparity": 0.003}
- **INFO** `penult/agreement_info[3,5]` {"panel_cka": 0.888, "relrep_row_corr_mean": 0.954, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.937, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.781, "error_consistency_test": 0.53, "cka_ood_c100": 0.659, "relrep_argmax_agree_ood_c100": 0.584, "relrep_argmax_chance_ood_c100": 0.125, "relrep_offmax_corr_ood_c100": 0.74, "landmark_procrustes_disparity": 0.01}
- **INFO** `penult/agreement_info[3,6]` {"panel_cka": 0.895, "relrep_row_corr_mean": 0.954, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.935, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.772, "error_consistency_test": 0.521, "cka_ood_c100": 0.65, "relrep_argmax_agree_ood_c100": 0.589, "relrep_argmax_chance_ood_c100": 0.124, "relrep_offmax_corr_ood_c100": 0.732, "landmark_procrustes_disparity": 0.012}
- **INFO** `penult/agreement_info[4,5]` {"panel_cka": 0.896, "relrep_row_corr_mean": 0.952, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.942, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.778, "error_consistency_test": 0.489, "cka_ood_c100": 0.659, "relrep_argmax_agree_ood_c100": 0.593, "relrep_argmax_chance_ood_c100": 0.124, "relrep_offmax_corr_ood_c100": 0.742, "landmark_procrustes_disparity": 0.009}
- **INFO** `penult/agreement_info[4,6]` {"panel_cka": 0.912, "relrep_row_corr_mean": 0.951, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.933, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.77, "error_consistency_test": 0.55, "cka_ood_c100": 0.65, "relrep_argmax_agree_ood_c100": 0.586, "relrep_argmax_chance_ood_c100": 0.123, "relrep_offmax_corr_ood_c100": 0.731, "landmark_procrustes_disparity": 0.009}
- **INFO** `penult/agreement_info[5,6]` {"panel_cka": 0.946, "relrep_row_corr_mean": 0.974, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.947, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.749, "error_consistency_test": 0.566, "cka_ood_c100": 0.676, "relrep_argmax_agree_ood_c100": 0.606, "relrep_argmax_chance_ood_c100": 0.123, "relrep_offmax_corr_ood_c100": 0.738, "landmark_procrustes_disparity": 0.004}
- PASS: 21 items
