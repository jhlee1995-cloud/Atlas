# CRITIC  verdict: **PARTIAL**
runs: ['results/atlas_v1_resnet20_s1', 'results/atlas_v1_resnet20_s2']
counts: {'PASS': 131, 'FAIL': 13, 'WARN': 1, 'INFO': 1}

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
80 items, 12 FAIL
- **FAIL** `stem/neural_collapse.nc1` values=[261.068, 152.924] rel_spread=0.522
- **FAIL** `layer1.0/pca_spectrum.participation_ratio` values=[3.288, 3.875] rel_spread=0.164
- **FAIL** `layer1.0/pca_spectrum.dim95` values=[5.0, 7.0] rel_spread=0.333
- **FAIL** `layer1.0/neural_collapse.nc1` values=[164.627, 136.168] rel_spread=0.189
- **FAIL** `layer1.0/hubness.k_occurrence_skew` values=[0.535, 0.741] rel_spread=0.323
- **FAIL** `layer1.1/pca_spectrum.participation_ratio` values=[4.07, 5.067] rel_spread=0.218
- **FAIL** `layer1.1/hubness.k_occurrence_skew` values=[0.644, 0.77] rel_spread=0.178
- **FAIL** `layer2.0/neural_collapse.nc1` values=[23.829, 19.416] rel_spread=0.204
- **FAIL** `layer2.1/neural_collapse.nc1` values=[17.043, 11.104] rel_spread=0.422
- **FAIL** `layer2.2/neural_collapse.nc1` values=[9.177, 7.357] rel_spread=0.220
- **FAIL** `layer3.0/pca_spectrum.participation_ratio` values=[10.286, 12.091] rel_spread=0.161
- **FAIL** `layer3.1/pca_spectrum.participation_ratio` values=[11.259, 9.641] rel_spread=0.155
- PASS: 68 items

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
18 items, 1 FAIL
- **FAIL** `commit/highfreq_ratio` ['layer1.1', 'layer2.0']
- PASS: 17 items

## panel_agreement
2 items, 0 FAIL
- **INFO** `penult/agreement_info[0,1]` {"panel_cka": 0.91, "relrep_row_corr_mean": 0.963, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.926, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.831, "error_consistency_test": 0.535, "cka_ood_c100": 0.694, "relrep_argmax_agree_ood_c100": 0.589, "relrep_argmax_chance_ood_c100": 0.126, "relrep_offmax_corr_ood_c100": 0.779, "landmark_procrustes_disparity": 0.002}
- PASS: 1 items
