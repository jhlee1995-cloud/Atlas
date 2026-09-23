# CRITIC  verdict: **PARTIAL**
runs: ['results/atlas_v1_resnet20_s0hub_st2', 'results/atlas_v1_resnet20_s1_st2', 'results/atlas_v1_resnet20_s2_st2']
counts: {'PASS': 207, 'FAIL': 19, 'WARN': 1, 'INFO': 3}

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
80 items, 18 FAIL
- **FAIL** `stem/neural_collapse.nc1` values=[259.926, 261.068, 152.924] rel_spread=0.481
- **FAIL** `layer1.0/pca_spectrum.participation_ratio` values=[4.042, 3.288, 3.875] rel_spread=0.202
- **FAIL** `layer1.0/pca_spectrum.dim95` values=[6.0, 5.0, 7.0] rel_spread=0.333
- **FAIL** `layer1.0/neural_collapse.nc1` values=[138.342, 164.627, 136.168] rel_spread=0.194
- **FAIL** `layer1.0/hubness.k_occurrence_skew` values=[0.631, 0.535, 0.741] rel_spread=0.324
- **FAIL** `layer1.1/pca_spectrum.participation_ratio` values=[5.171, 4.07, 5.067] rel_spread=0.231
- **FAIL** `layer1.1/neural_collapse.nc1` values=[57.975, 74.307, 68.463] rel_spread=0.244
- **FAIL** `layer1.1/neural_collapse.etf_deviation` values=[0.494, 0.544, 0.575] rel_spread=0.150
- **FAIL** `layer1.1/hubness.k_occurrence_skew` values=[0.797, 0.644, 0.77] rel_spread=0.208
- **FAIL** `layer1.2/pca_spectrum.participation_ratio` values=[5.518, 4.8, 4.499] rel_spread=0.206
- **FAIL** `layer1.2/hubness.k_occurrence_skew` values=[0.871, 0.713, 0.821] rel_spread=0.197
- **FAIL** `layer2.0/pca_spectrum.participation_ratio` values=[6.816, 6.067, 5.598] rel_spread=0.198
- **FAIL** `layer2.0/pca_spectrum.dim95` values=[13.0, 12.0, 11.0] rel_spread=0.167
- **FAIL** `layer2.0/neural_collapse.nc1` values=[21.892, 23.829, 19.416] rel_spread=0.203
- **FAIL** `layer2.1/neural_collapse.nc1` values=[12.411, 17.043, 11.104] rel_spread=0.439
- **FAIL** `layer2.2/neural_collapse.nc1` values=[7.065, 9.177, 7.357] rel_spread=0.269
- **FAIL** `layer3.0/pca_spectrum.participation_ratio` values=[12.136, 10.286, 12.091] rel_spread=0.161
- **FAIL** `layer3.1/pca_spectrum.participation_ratio` values=[10.468, 11.259, 9.641] rel_spread=0.155
- PASS: 62 items

## id_profile_stability
3 items, 0 FAIL
- PASS: 3 items

## adjacency_stability
60 items, 0 FAIL
- PASS: 60 items

## decodability_stability
54 items, 0 FAIL
- PASS: 54 items

## commit_agreement
18 items, 1 FAIL
- **FAIL** `commit/highfreq_ratio` ['layer1.0', 'layer1.1', 'layer2.0']
- PASS: 17 items

## panel_agreement
6 items, 0 FAIL
- **INFO** `penult/agreement_info[0,1]` {"panel_cka": 0.903, "relrep_row_corr_mean": 0.963, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.924, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.818, "error_consistency_test": 0.552, "cka_ood_c100": 0.68, "relrep_argmax_agree_ood_c100": 0.583, "relrep_argmax_chance_ood_c100": 0.126, "relrep_offmax_corr_ood_c100": 0.766, "landmark_procrustes_disparity": 0.002}
- **INFO** `penult/agreement_info[0,2]` {"panel_cka": 0.919, "relrep_row_corr_mean": 0.968, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.92, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.834, "error_consistency_test": 0.528, "cka_ood_c100": 0.686, "relrep_argmax_agree_ood_c100": 0.585, "relrep_argmax_chance_ood_c100": 0.125, "relrep_offmax_corr_ood_c100": 0.772, "landmark_procrustes_disparity": 0.001}
- **INFO** `penult/agreement_info[1,2]` {"panel_cka": 0.91, "relrep_row_corr_mean": 0.963, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.926, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.831, "error_consistency_test": 0.535, "cka_ood_c100": 0.694, "relrep_argmax_agree_ood_c100": 0.589, "relrep_argmax_chance_ood_c100": 0.126, "relrep_offmax_corr_ood_c100": 0.779, "landmark_procrustes_disparity": 0.002}
- PASS: 3 items
