# CRITIC  verdict: **PARTIAL**
runs: ['results/atlas_v1_resnet20_s3', 'results/atlas_v1_resnet20_s4']
counts: {'PASS': 133, 'FAIL': 11, 'WARN': 1, 'INFO': 1}

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
2 items, 0 FAIL
- PASS: 2 items

## probe_hygiene
2 items, 0 FAIL
- PASS: 2 items

## scalar_stability
80 items, 8 FAIL
- **FAIL** `stem/pca_spectrum.participation_ratio` values=[2.644, 3.09] rel_spread=0.156
- **FAIL** `stem/neural_collapse.nc1` values=[318.793, 159.182] rel_spread=0.668
- **FAIL** `layer1.0/neural_collapse.nc1` values=[123.091, 93.88] rel_spread=0.269
- **FAIL** `layer1.1/neural_collapse.nc1` values=[69.758, 86.846] rel_spread=0.218
- **FAIL** `layer1.2/hubness.k_occurrence_skew` values=[0.767, 0.922] rel_spread=0.183
- **FAIL** `layer2.0/neural_collapse.nc1` values=[24.574, 20.88] rel_spread=0.163
- **FAIL** `layer3.1/pca_spectrum.participation_ratio` values=[12.61, 9.453] rel_spread=0.286
- **FAIL** `layer3.1/neural_collapse.nc1` values=[0.923, 0.792] rel_spread=0.153
- PASS: 72 items

## id_profile_stability
1 items, 0 FAIL
- PASS: 1 items

## adjacency_stability
20 items, 0 FAIL
- PASS: 20 items

## decodability_stability
18 items, 0 FAIL
- PASS: 18 items

## commit_agreement
18 items, 3 FAIL
- **FAIL** `commit/highfreq_ratio` ['layer1.0', 'layer2.0']
- **FAIL** `commit/orientation_entropy` ['layer3.0', 'layer2.0']
- **FAIL** `commit/severity` ['layer3.0', 'layer2.0']
- PASS: 15 items

## panel_agreement
2 items, 0 FAIL
- **INFO** `penult/agreement_info[0,1]` {"panel_cka": 0.895, "relrep_row_corr_mean": 0.961, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.938, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.831, "error_consistency_test": 0.581, "cka_ood_c100": 0.708, "relrep_argmax_agree_ood_c100": 0.581, "relrep_argmax_chance_ood_c100": 0.124, "relrep_offmax_corr_ood_c100": 0.781, "landmark_procrustes_disparity": 0.003}
- PASS: 1 items
