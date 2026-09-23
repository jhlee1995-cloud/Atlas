# CRITIC  verdict: **PARTIAL**
runs: ['results/atlas_v1_resnet20_s0hub_st3', 'results/atlas_v1_resnet20_s1_st3', 'results/atlas_v1_resnet20_s2_st3', 'results/atlas_v1_resnet20_s3_st3', 'results/atlas_v1_resnet20_s4_st3']
counts: {'PASS': 483, 'FAIL': 27, 'WARN': 1, 'INFO': 10}

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
5 items, 0 FAIL
- PASS: 5 items

## probe_hygiene
5 items, 0 FAIL
- PASS: 5 items

## scalar_stability
80 items, 21 FAIL
- **FAIL** `stem/pca_spectrum.participation_ratio` values=[2.979, 2.993, 2.975, 2.644, 3.09] rel_spread=0.152
- **FAIL** `stem/neural_collapse.nc1` values=[259.926, 261.068, 152.924, 318.793, 159.182] rel_spread=0.720
- **FAIL** `layer1.0/pca_spectrum.participation_ratio` values=[4.042, 3.288, 3.875, 4.303, 3.95] rel_spread=0.261
- **FAIL** `layer1.0/pca_spectrum.dim95` values=[6.0, 5.0, 7.0, 7.0, 7.0] rel_spread=0.312
- **FAIL** `layer1.0/neural_collapse.nc1` values=[138.342, 164.627, 136.168, 123.091, 93.88] rel_spread=0.539
- **FAIL** `layer1.0/neural_collapse.etf_deviation` values=[0.581, 0.635, 0.613, 0.561, 0.519] rel_spread=0.200
- **FAIL** `layer1.0/hubness.k_occurrence_skew` values=[0.631, 0.535, 0.741, 0.642, 0.74] rel_spread=0.313
- **FAIL** `layer1.1/pca_spectrum.participation_ratio` values=[5.171, 4.07, 5.067, 4.944, 4.893] rel_spread=0.228
- **FAIL** `layer1.1/neural_collapse.nc1` values=[57.975, 74.307, 68.463, 69.758, 86.846] rel_spread=0.404
- **FAIL** `layer1.1/neural_collapse.etf_deviation` values=[0.494, 0.544, 0.575, 0.532, 0.46] rel_spread=0.222
- **FAIL** `layer1.1/hubness.k_occurrence_skew` values=[0.797, 0.644, 0.77, 0.815, 0.8] rel_spread=0.223
- **FAIL** `layer1.2/pca_spectrum.participation_ratio` values=[5.518, 4.8, 4.499, 5.286, 5.102] rel_spread=0.202
- **FAIL** `layer1.2/hubness.k_occurrence_skew` values=[0.871, 0.713, 0.821, 0.767, 0.922] rel_spread=0.255
- **FAIL** `layer2.0/pca_spectrum.participation_ratio` values=[6.816, 6.067, 5.598, 5.852, 6.145] rel_spread=0.200
- **FAIL** `layer2.0/pca_spectrum.dim95` values=[13.0, 12.0, 11.0, 12.0, 12.0] rel_spread=0.167
- **FAIL** `layer2.0/neural_collapse.nc1` values=[21.892, 23.829, 19.416, 24.574, 20.88] rel_spread=0.233
- **FAIL** `layer2.1/neural_collapse.nc1` values=[12.411, 17.043, 11.104, 15.979, 14.636] rel_spread=0.417
- **FAIL** `layer2.2/neural_collapse.nc1` values=[7.065, 9.177, 7.357, 8.82, 8.825] rel_spread=0.256
- **FAIL** `layer3.0/pca_spectrum.participation_ratio` values=[12.136, 10.286, 12.091, 11.533, 10.434] rel_spread=0.164
- **FAIL** `layer3.1/pca_spectrum.participation_ratio` values=[10.468, 11.259, 9.641, 12.61, 9.453] rel_spread=0.295
- **FAIL** `layer3.1/neural_collapse.nc1` values=[0.759, 0.774, 0.859, 0.923, 0.792] rel_spread=0.200
- PASS: 59 items

## id_profile_stability
10 items, 0 FAIL
- PASS: 10 items

## adjacency_stability
200 items, 0 FAIL
- PASS: 200 items

## decodability_stability
180 items, 3 FAIL
- **FAIL** `decod/hue_cos[0,4]` profile_spearman=0.61 mean_abs_delta=0.032
- **FAIL** `decod/hue_cos[1,4]` profile_spearman=0.64 mean_abs_delta=0.045
- **FAIL** `decod/edge_density[0,4]` profile_spearman=0.68 mean_abs_delta=0.028
- PASS: 177 items

## commit_agreement
18 items, 3 FAIL
- **FAIL** `commit/highfreq_ratio` ['layer1.0', 'layer1.1', 'layer2.0', 'layer1.0', 'layer2.0']
- **FAIL** `commit/orientation_entropy` ['layer2.0', 'layer2.0', 'layer2.0', 'layer3.0', 'layer2.0']
- **FAIL** `commit/severity` ['layer3.0', 'layer3.0', 'layer2.2', 'layer3.0', 'layer2.0']
- PASS: 15 items

## panel_agreement
20 items, 0 FAIL
- **INFO** `penult/agreement_info[0,1]` {"panel_cka": 0.903, "relrep_row_corr_mean": 0.963, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.924, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.818, "error_consistency_test": 0.552, "cka_ood_c100": 0.68, "relrep_argmax_agree_ood_c100": 0.583, "relrep_argmax_chance_ood_c100": 0.126, "relrep_offmax_corr_ood_c100": 0.766, "landmark_procrustes_disparity": 0.002}
- **INFO** `penult/agreement_info[0,2]` {"panel_cka": 0.919, "relrep_row_corr_mean": 0.968, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.92, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.834, "error_consistency_test": 0.528, "cka_ood_c100": 0.686, "relrep_argmax_agree_ood_c100": 0.585, "relrep_argmax_chance_ood_c100": 0.125, "relrep_offmax_corr_ood_c100": 0.772, "landmark_procrustes_disparity": 0.001}
- **INFO** `penult/agreement_info[0,3]` {"panel_cka": 0.895, "relrep_row_corr_mean": 0.964, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.923, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.827, "error_consistency_test": 0.507, "cka_ood_c100": 0.683, "relrep_argmax_agree_ood_c100": 0.588, "relrep_argmax_chance_ood_c100": 0.126, "relrep_offmax_corr_ood_c100": 0.759, "landmark_procrustes_disparity": 0.003}
- **INFO** `penult/agreement_info[0,4]` {"panel_cka": 0.92, "relrep_row_corr_mean": 0.965, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.932, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.826, "error_consistency_test": 0.565, "cka_ood_c100": 0.687, "relrep_argmax_agree_ood_c100": 0.578, "relrep_argmax_chance_ood_c100": 0.125, "relrep_offmax_corr_ood_c100": 0.77, "landmark_procrustes_disparity": 0.002}
- **INFO** `penult/agreement_info[1,2]` {"panel_cka": 0.91, "relrep_row_corr_mean": 0.963, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.926, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.831, "error_consistency_test": 0.535, "cka_ood_c100": 0.694, "relrep_argmax_agree_ood_c100": 0.589, "relrep_argmax_chance_ood_c100": 0.126, "relrep_offmax_corr_ood_c100": 0.779, "landmark_procrustes_disparity": 0.002}
- **INFO** `penult/agreement_info[1,3]` {"panel_cka": 0.899, "relrep_row_corr_mean": 0.962, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.929, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.82, "error_consistency_test": 0.537, "cka_ood_c100": 0.693, "relrep_argmax_agree_ood_c100": 0.577, "relrep_argmax_chance_ood_c100": 0.126, "relrep_offmax_corr_ood_c100": 0.772, "landmark_procrustes_disparity": 0.004}
- **INFO** `penult/agreement_info[1,4]` {"panel_cka": 0.905, "relrep_row_corr_mean": 0.961, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.934, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.826, "error_consistency_test": 0.526, "cka_ood_c100": 0.695, "relrep_argmax_agree_ood_c100": 0.571, "relrep_argmax_chance_ood_c100": 0.125, "relrep_offmax_corr_ood_c100": 0.776, "landmark_procrustes_disparity": 0.001}
- **INFO** `penult/agreement_info[2,3]` {"panel_cka": 0.916, "relrep_row_corr_mean": 0.967, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.929, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.839, "error_consistency_test": 0.527, "cka_ood_c100": 0.7, "relrep_argmax_agree_ood_c100": 0.575, "relrep_argmax_chance_ood_c100": 0.125, "relrep_offmax_corr_ood_c100": 0.778, "landmark_procrustes_disparity": 0.003}
- **INFO** `penult/agreement_info[2,4]` {"panel_cka": 0.912, "relrep_row_corr_mean": 0.962, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.933, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.833, "error_consistency_test": 0.539, "cka_ood_c100": 0.699, "relrep_argmax_agree_ood_c100": 0.572, "relrep_argmax_chance_ood_c100": 0.124, "relrep_offmax_corr_ood_c100": 0.773, "landmark_procrustes_disparity": 0.002}
- **INFO** `penult/agreement_info[3,4]` {"panel_cka": 0.895, "relrep_row_corr_mean": 0.961, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.938, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.831, "error_consistency_test": 0.581, "cka_ood_c100": 0.708, "relrep_argmax_agree_ood_c100": 0.581, "relrep_argmax_chance_ood_c100": 0.124, "relrep_offmax_corr_ood_c100": 0.781, "landmark_procrustes_disparity": 0.003}
- PASS: 10 items
