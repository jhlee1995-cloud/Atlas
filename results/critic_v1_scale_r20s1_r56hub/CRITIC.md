# CRITIC  verdict: **PARTIAL**
runs: ['results/atlas_v1_resnet20_s1_st2', 'results/atlas_v1_resnet56_s0hub']
counts: {'PASS': 125, 'FAIL': 19, 'WARN': 1, 'INFO': 1}
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
80 items, 15 FAIL
- **FAIL** `stem/neural_collapse.nc1` values=[261.068, 494.389] rel_spread=0.618
- **FAIL** `layer1.0/hubness.k_occurrence_skew` values=[0.535, 0.381] rel_spread=0.335
- **FAIL** `layer1.1/pca_spectrum.participation_ratio` values=[4.07, 5.118] rel_spread=0.228
- **FAIL** `layer1.1/neural_collapse.nc1` values=[74.307, 53.611] rel_spread=0.324
- **FAIL** `layer1.1/hubness.k_occurrence_skew` values=[0.644, 0.763] rel_spread=0.169
- **FAIL** `layer1.2/hubness.k_occurrence_skew` values=[0.713, 0.856] rel_spread=0.182
- **FAIL** `layer2.1/neural_collapse.nc1` values=[17.043, 10.911] rel_spread=0.439
- **FAIL** `layer2.2/neural_collapse.nc1` values=[9.177, 6.224] rel_spread=0.383
- **FAIL** `layer3.0/pca_spectrum.dim95` values=[38.0, 25.0] rel_spread=0.413
- **FAIL** `layer3.0/neural_collapse.nc1` values=[2.391, 3.502] rel_spread=0.377
- **FAIL** `layer3.0/hubness.k_occurrence_skew` values=[1.691, 1.404] rel_spread=0.186
- **FAIL** `layer3.1/pca_spectrum.participation_ratio` values=[11.259, 15.827] rel_spread=0.337
- **FAIL** `layer3.1/neural_collapse.etf_deviation` values=[0.289, 0.21] rel_spread=0.317
- **FAIL** `penult/class_centers.sep_ratio` values=[3.067, 5.332] rel_spread=0.540
- **FAIL** `penult/neural_collapse.nc1` values=[0.163, 0.049] rel_spread=1.079
- PASS: 65 items

## id_profile_stability
1 items, 0 FAIL
- PASS: 1 items

## adjacency_stability
20 items, 1 FAIL
- **FAIL** `layer2.2/merge_kendall[0,1]` tau=0.559
- PASS: 19 items

## decodability_stability
18 items, 1 FAIL
- **FAIL** `decod/spectral_anisotropy[0,1]` profile_spearman=0.67 mean_abs_delta=0.021
- PASS: 17 items

## commit_agreement
18 items, 2 FAIL
- **FAIL** `commit/highfreq_ratio` ['layer1.1', 'layer2.0']
- **FAIL** `commit/spectral_anisotropy` ['layer1.2', 'layer3.0']
- PASS: 16 items

## panel_agreement
2 items, 0 FAIL
- **INFO** `penult/agreement_info[0,1]` {"panel_cka": 0.892, "relrep_row_corr_mean": 0.944, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.939, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.725, "error_consistency_test": 0.534, "cka_ood_c100": 0.649, "relrep_argmax_agree_ood_c100": 0.573, "relrep_argmax_chance_ood_c100": 0.124, "relrep_offmax_corr_ood_c100": 0.722, "landmark_procrustes_disparity": 0.01}
- PASS: 1 items
