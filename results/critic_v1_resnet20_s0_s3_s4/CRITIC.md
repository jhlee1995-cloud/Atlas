# CRITIC  verdict: **PARTIAL**
runs: ['results/atlas_v1_resnet20_s0hub', 'results/atlas_v1_resnet20_s3', 'results/atlas_v1_resnet20_s4']
counts: {'PASS': 209, 'FAIL': 17, 'WARN': 1, 'INFO': 3}

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
3 items, 0 FAIL
- PASS: 3 items

## probe_hygiene
3 items, 0 FAIL
- PASS: 3 items

## scalar_stability
80 items, 12 FAIL
- **FAIL** `stem/pca_spectrum.participation_ratio` values=[2.979, 2.644, 3.09] rel_spread=0.154
- **FAIL** `stem/neural_collapse.nc1` values=[259.926, 318.793, 159.182] rel_spread=0.649
- **FAIL** `layer1.0/neural_collapse.nc1` values=[138.342, 123.091, 93.88] rel_spread=0.375
- **FAIL** `layer1.0/hubness.k_occurrence_skew` values=[0.631, 0.642, 0.74] rel_spread=0.162
- **FAIL** `layer1.1/neural_collapse.nc1` values=[57.975, 69.758, 86.846] rel_spread=0.404
- **FAIL** `layer1.2/hubness.k_occurrence_skew` values=[0.871, 0.767, 0.922] rel_spread=0.181
- **FAIL** `layer2.0/pca_spectrum.participation_ratio` values=[6.816, 5.852, 6.145] rel_spread=0.154
- **FAIL** `layer2.0/neural_collapse.nc1` values=[21.892, 24.574, 20.88] rel_spread=0.165
- **FAIL** `layer2.1/neural_collapse.nc1` values=[12.411, 15.979, 14.636] rel_spread=0.249
- **FAIL** `layer2.2/neural_collapse.nc1` values=[7.065, 8.82, 8.825] rel_spread=0.214
- **FAIL** `layer3.1/pca_spectrum.participation_ratio` values=[10.468, 12.61, 9.453] rel_spread=0.291
- **FAIL** `layer3.1/neural_collapse.nc1` values=[0.759, 0.923, 0.792] rel_spread=0.199
- PASS: 68 items

## id_profile_stability
3 items, 0 FAIL
- PASS: 3 items

## adjacency_stability
60 items, 0 FAIL
- PASS: 60 items

## decodability_stability
54 items, 2 FAIL
- **FAIL** `decod/hue_cos[0,2]` profile_spearman=0.61 mean_abs_delta=0.032
- **FAIL** `decod/edge_density[0,2]` profile_spearman=0.68 mean_abs_delta=0.028
- PASS: 52 items

## commit_agreement
18 items, 3 FAIL
- **FAIL** `commit/highfreq_ratio` ['layer1.0', 'layer1.0', 'layer2.0']
- **FAIL** `commit/orientation_entropy` ['layer2.0', 'layer3.0', 'layer2.0']
- **FAIL** `commit/severity` ['layer3.0', 'layer3.0', 'layer2.0']
- PASS: 15 items

## panel_agreement
6 items, 0 FAIL
- **INFO** `penult/agreement_info[0,1]` {"panel_cka": 0.895, "relrep_row_corr_mean": 0.964, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.923, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.827, "error_consistency_test": 0.507, "cka_ood_c100": 0.683, "relrep_argmax_agree_ood_c100": 0.588, "relrep_argmax_chance_ood_c100": 0.126, "relrep_offmax_corr_ood_c100": 0.759, "landmark_procrustes_disparity": 0.003}
- **INFO** `penult/agreement_info[0,2]` {"panel_cka": 0.92, "relrep_row_corr_mean": 0.965, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.932, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.826, "error_consistency_test": 0.565, "cka_ood_c100": 0.687, "relrep_argmax_agree_ood_c100": 0.578, "relrep_argmax_chance_ood_c100": 0.125, "relrep_offmax_corr_ood_c100": 0.77, "landmark_procrustes_disparity": 0.002}
- **INFO** `penult/agreement_info[1,2]` {"panel_cka": 0.895, "relrep_row_corr_mean": 0.961, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.938, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.831, "error_consistency_test": 0.581, "cka_ood_c100": 0.708, "relrep_argmax_agree_ood_c100": 0.581, "relrep_argmax_chance_ood_c100": 0.124, "relrep_offmax_corr_ood_c100": 0.781, "landmark_procrustes_disparity": 0.003}
- PASS: 3 items
