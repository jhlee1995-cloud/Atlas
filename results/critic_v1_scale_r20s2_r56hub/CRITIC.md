# CRITIC  verdict: **PARTIAL**
runs: ['results/atlas_v1_resnet20_s2_st2', 'results/atlas_v1_resnet56_s0hub']
counts: {'PASS': 116, 'FAIL': 28, 'WARN': 1, 'INFO': 1}
layer alignment: position {"atlas_v1_resnet56_s0hub": [["stem", "stem"], ["layer1.0", "layer1.0"], ["layer1.1", "layer1.5"], ["layer1.2", "layer1.8"], ["layer2.0", "layer2.0"], ["layer2.1", "layer2.5"], ["layer2.2", "layer2.8"], ["layer3.0", "layer3.0"], ["layer3.1", "layer3.5"], ["layer3.2", "layer3.8"], ["penult", "penult"]]}

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
80 items, 26 FAIL
- **FAIL** `stem/neural_collapse.nc1` values=[152.924, 494.389] rel_spread=1.055
- **FAIL** `stem/hubness.k_occurrence_skew` values=[0.209, 0.146] rel_spread=0.351
- **FAIL** `layer1.0/twonn_id.id` values=[9.201, 7.786] rel_spread=0.167
- **FAIL** `layer1.0/pca_spectrum.participation_ratio` values=[3.875, 3.185] rel_spread=0.196
- **FAIL** `layer1.0/pca_spectrum.dim95` values=[7.0, 5.0] rel_spread=0.333
- **FAIL** `layer1.0/neural_collapse.nc1` values=[136.168, 171.954] rel_spread=0.232
- **FAIL** `layer1.0/hubness.k_occurrence_skew` values=[0.741, 0.381] rel_spread=0.641
- **FAIL** `layer1.1/neural_collapse.nc1` values=[68.463, 53.611] rel_spread=0.243
- **FAIL** `layer1.2/pca_spectrum.participation_ratio` values=[4.499, 5.491] rel_spread=0.198
- **FAIL** `layer1.2/neural_collapse.nc1` values=[47.77, 56.519] rel_spread=0.168
- **FAIL** `layer2.0/neural_collapse.nc1` values=[19.416, 24.111] rel_spread=0.216
- **FAIL** `layer2.1/pca_spectrum.participation_ratio` values=[7.111, 8.682] rel_spread=0.199
- **FAIL** `layer2.2/pca_spectrum.participation_ratio` values=[7.948, 9.872] rel_spread=0.216
- **FAIL** `layer2.2/neural_collapse.nc1` values=[7.357, 6.224] rel_spread=0.167
- **FAIL** `layer3.0/twonn_id.id` values=[18.284, 15.572] rel_spread=0.160
- **FAIL** `layer3.0/pca_spectrum.participation_ratio` values=[12.091, 10.033] rel_spread=0.186
- **FAIL** `layer3.0/pca_spectrum.dim95` values=[38.0, 25.0] rel_spread=0.413
- **FAIL** `layer3.0/neural_collapse.nc1` values=[2.533, 3.502] rel_spread=0.321
- **FAIL** `layer3.0/hubness.k_occurrence_skew` values=[1.701, 1.404] rel_spread=0.192
- **FAIL** `layer3.1/pca_spectrum.participation_ratio` values=[9.641, 15.827] rel_spread=0.486
- **FAIL** `layer3.1/neural_collapse.nc1` values=[0.859, 0.671] rel_spread=0.247
- **FAIL** `layer3.1/neural_collapse.etf_deviation` values=[0.29, 0.21] rel_spread=0.321
- **FAIL** `penult/class_centers.sep_ratio` values=[3.021, 5.332] rel_spread=0.553
- **FAIL** `penult/neural_collapse.nc1` values=[0.17, 0.049] rel_spread=1.107
- **FAIL** `penult/neural_collapse.etf_deviation` values=[0.107, 0.054] rel_spread=0.663
- **FAIL** `penult/hubness.k_occurrence_skew` values=[0.767, 0.94] rel_spread=0.203
- PASS: 54 items

## id_profile_stability
1 items, 0 FAIL
- PASS: 1 items

## adjacency_stability
20 items, 1 FAIL
- **FAIL** `layer2.2/merge_kendall[0,1]` tau=0.557
- PASS: 19 items

## decodability_stability
18 items, 0 FAIL
- PASS: 18 items

## commit_agreement
18 items, 1 FAIL
- **FAIL** `commit/spectral_anisotropy` ['layer2.0', 'layer3.0']
- PASS: 17 items

## panel_agreement
2 items, 0 FAIL
- **INFO** `penult/agreement_info[0,1]` {"panel_cka": 0.921, "relrep_row_corr_mean": 0.95, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.934, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.722, "error_consistency_test": 0.504, "cka_ood_c100": 0.651, "relrep_argmax_agree_ood_c100": 0.586, "relrep_argmax_chance_ood_c100": 0.123, "relrep_offmax_corr_ood_c100": 0.723, "landmark_procrustes_disparity": 0.014}
- PASS: 1 items
