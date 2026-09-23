# CRITIC  verdict: **PARTIAL**
runs: ['results/atlas_v1_resnet20_s1', 'results/atlas_v1_resnet20_s2', 'results/atlas_v1_resnet20_s3', 'results/atlas_v1_resnet20_s4']
counts: {'PASS': 325, 'FAIL': 23, 'WARN': 1, 'INFO': 6}

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
4 items, 0 FAIL
- PASS: 4 items

## probe_hygiene
4 items, 0 FAIL
- PASS: 4 items

## scalar_stability
80 items, 19 FAIL
- **FAIL** `stem/pca_spectrum.participation_ratio` values=[2.993, 2.975, 2.644, 3.09] rel_spread=0.152
- **FAIL** `stem/neural_collapse.nc1` values=[261.068, 152.924, 318.793, 159.182] rel_spread=0.744
- **FAIL** `layer1.0/pca_spectrum.participation_ratio` values=[3.288, 3.875, 4.303, 3.95] rel_spread=0.263
- **FAIL** `layer1.0/pca_spectrum.dim95` values=[5.0, 7.0, 7.0, 7.0] rel_spread=0.308
- **FAIL** `layer1.0/neural_collapse.nc1` values=[164.627, 136.168, 123.091, 93.88] rel_spread=0.547
- **FAIL** `layer1.0/neural_collapse.etf_deviation` values=[0.635, 0.613, 0.561, 0.519] rel_spread=0.200
- **FAIL** `layer1.0/hubness.k_occurrence_skew` values=[0.535, 0.741, 0.642, 0.74] rel_spread=0.310
- **FAIL** `layer1.1/pca_spectrum.participation_ratio` values=[4.07, 5.067, 4.944, 4.893] rel_spread=0.210
- **FAIL** `layer1.1/neural_collapse.nc1` values=[74.307, 68.463, 69.758, 86.846] rel_spread=0.246
- **FAIL** `layer1.1/neural_collapse.etf_deviation` values=[0.544, 0.575, 0.532, 0.46] rel_spread=0.219
- **FAIL** `layer1.1/hubness.k_occurrence_skew` values=[0.644, 0.77, 0.815, 0.8] rel_spread=0.226
- **FAIL** `layer1.2/pca_spectrum.participation_ratio` values=[4.8, 4.499, 5.286, 5.102] rel_spread=0.160
- **FAIL** `layer1.2/hubness.k_occurrence_skew` values=[0.713, 0.821, 0.767, 0.922] rel_spread=0.259
- **FAIL** `layer2.0/neural_collapse.nc1` values=[23.829, 19.416, 24.574, 20.88] rel_spread=0.233
- **FAIL** `layer2.1/neural_collapse.nc1` values=[17.043, 11.104, 15.979, 14.636] rel_spread=0.404
- **FAIL** `layer2.2/neural_collapse.nc1` values=[9.177, 7.357, 8.82, 8.825] rel_spread=0.213
- **FAIL** `layer3.0/pca_spectrum.participation_ratio` values=[10.286, 12.091, 11.533, 10.434] rel_spread=0.163
- **FAIL** `layer3.1/pca_spectrum.participation_ratio` values=[11.259, 9.641, 12.61, 9.453] rel_spread=0.294
- **FAIL** `layer3.1/neural_collapse.nc1` values=[0.774, 0.859, 0.923, 0.792] rel_spread=0.178
- PASS: 61 items

## id_profile_stability
6 items, 0 FAIL
- PASS: 6 items

## adjacency_stability
120 items, 0 FAIL
- PASS: 120 items

## decodability_stability
108 items, 1 FAIL
- **FAIL** `decod/hue_cos[0,3]` profile_spearman=0.64 mean_abs_delta=0.045
- PASS: 107 items

## commit_agreement
18 items, 3 FAIL
- **FAIL** `commit/highfreq_ratio` ['layer1.1', 'layer2.0', 'layer1.0', 'layer2.0']
- **FAIL** `commit/orientation_entropy` ['layer2.0', 'layer2.0', 'layer3.0', 'layer2.0']
- **FAIL** `commit/severity` ['layer3.0', 'layer2.2', 'layer3.0', 'layer2.0']
- PASS: 15 items

## panel_agreement
12 items, 0 FAIL
- **INFO** `penult/agreement_info[0,1]` {"panel_cka": 0.91, "relrep_row_corr_mean": 0.963, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.926, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.831, "error_consistency_test": 0.535, "cka_ood_c100": 0.694, "relrep_argmax_agree_ood_c100": 0.589, "relrep_argmax_chance_ood_c100": 0.126, "relrep_offmax_corr_ood_c100": 0.779, "landmark_procrustes_disparity": 0.002}
- **INFO** `penult/agreement_info[0,2]` {"panel_cka": 0.899, "relrep_row_corr_mean": 0.962, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.929, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.82, "error_consistency_test": 0.537, "cka_ood_c100": 0.693, "relrep_argmax_agree_ood_c100": 0.577, "relrep_argmax_chance_ood_c100": 0.126, "relrep_offmax_corr_ood_c100": 0.772, "landmark_procrustes_disparity": 0.004}
- **INFO** `penult/agreement_info[0,3]` {"panel_cka": 0.905, "relrep_row_corr_mean": 0.961, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.934, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.826, "error_consistency_test": 0.526, "cka_ood_c100": 0.695, "relrep_argmax_agree_ood_c100": 0.571, "relrep_argmax_chance_ood_c100": 0.125, "relrep_offmax_corr_ood_c100": 0.776, "landmark_procrustes_disparity": 0.001}
- **INFO** `penult/agreement_info[1,2]` {"panel_cka": 0.916, "relrep_row_corr_mean": 0.967, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.929, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.839, "error_consistency_test": 0.527, "cka_ood_c100": 0.7, "relrep_argmax_agree_ood_c100": 0.575, "relrep_argmax_chance_ood_c100": 0.125, "relrep_offmax_corr_ood_c100": 0.778, "landmark_procrustes_disparity": 0.003}
- **INFO** `penult/agreement_info[1,3]` {"panel_cka": 0.912, "relrep_row_corr_mean": 0.962, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.933, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.833, "error_consistency_test": 0.539, "cka_ood_c100": 0.699, "relrep_argmax_agree_ood_c100": 0.572, "relrep_argmax_chance_ood_c100": 0.124, "relrep_offmax_corr_ood_c100": 0.773, "landmark_procrustes_disparity": 0.002}
- **INFO** `penult/agreement_info[2,3]` {"panel_cka": 0.895, "relrep_row_corr_mean": 0.961, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.938, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.831, "error_consistency_test": 0.581, "cka_ood_c100": 0.708, "relrep_argmax_agree_ood_c100": 0.581, "relrep_argmax_chance_ood_c100": 0.124, "relrep_offmax_corr_ood_c100": 0.781, "landmark_procrustes_disparity": 0.003}
- PASS: 6 items
