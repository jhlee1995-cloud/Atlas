# CRITIC  verdict: **PARTIAL**
runs: ['results/atlas_v1_resnet56_s1', 'results/atlas_v1_resnet56_s2']
counts: {'PASS': 119, 'FAIL': 25, 'WARN': 1, 'INFO': 1}

## synthetic_refusal
1 items, 0 FAIL
- PASS: 1 items

## norm_consistency
1 items, 0 FAIL
- PASS: 1 items

## alias_layers
1 items, 0 FAIL
- **WARN** `alias:layer3.8` identical to penult in every run; counted once as penult

## holdout_hygiene
2 items, 0 FAIL
- PASS: 2 items

## probe_hygiene
2 items, 0 FAIL
- PASS: 2 items

## scalar_stability
80 items, 18 FAIL
- **FAIL** `stem/pca_spectrum.dim95` values=[3.0, 4.0] rel_spread=0.286
- **FAIL** `stem/neural_collapse.nc1` values=[499.245, 213.957] rel_spread=0.800
- **FAIL** `stem/hubness.k_occurrence_skew` values=[0.096, 0.165] rel_spread=0.523
- **FAIL** `layer1.0/pca_spectrum.participation_ratio` values=[2.623, 3.577] rel_spread=0.308
- **FAIL** `layer1.0/pca_spectrum.dim95` values=[4.0, 5.0] rel_spread=0.222
- **FAIL** `layer1.0/neural_collapse.nc1` values=[264.537, 133.13] rel_spread=0.661
- **FAIL** `layer1.0/hubness.k_occurrence_skew` values=[0.271, 0.388] rel_spread=0.358
- **FAIL** `layer1.5/pca_spectrum.participation_ratio` values=[4.753, 5.67] rel_spread=0.176
- **FAIL** `layer1.5/pca_spectrum.dim95` values=[7.0, 9.0] rel_spread=0.250
- **FAIL** `layer1.5/class_centers.sep_ratio` values=[0.235, 0.297] rel_spread=0.234
- **FAIL** `layer1.5/neural_collapse.nc1` values=[68.49, 86.672] rel_spread=0.234
- **FAIL** `layer1.8/pca_spectrum.dim95` values=[8.0, 10.0] rel_spread=0.222
- **FAIL** `layer1.8/neural_collapse.nc1` values=[52.914, 66.033] rel_spread=0.221
- **FAIL** `layer2.0/neural_collapse.nc1` values=[30.982, 16.817] rel_spread=0.593
- **FAIL** `layer2.0/hubness.k_occurrence_skew` values=[0.835, 0.992] rel_spread=0.171
- **FAIL** `layer2.5/neural_collapse.nc1` values=[12.951, 9.034] rel_spread=0.356
- **FAIL** `layer2.8/neural_collapse.nc1` values=[7.266, 5.407] rel_spread=0.293
- **FAIL** `layer3.5/hubness.k_occurrence_skew` values=[2.042, 1.745] rel_spread=0.157
- PASS: 62 items

## id_profile_stability
1 items, 0 FAIL
- PASS: 1 items

## adjacency_stability
20 items, 1 FAIL
- **FAIL** `layer2.5/merge_kendall[0,1]` tau=0.508
- PASS: 19 items

## decodability_stability
18 items, 4 FAIL
- **FAIL** `decod/contrast_rms[0,1]` profile_spearman=0.41 mean_abs_delta=0.048
- **FAIL** `decod/saturation_mean[0,1]` profile_spearman=0.53 mean_abs_delta=0.082
- **FAIL** `decod/colorfulness[0,1]` profile_spearman=0.61 mean_abs_delta=0.071
- **FAIL** `decod/edge_density[0,1]` profile_spearman=0.65 mean_abs_delta=0.043
- PASS: 14 items

## commit_agreement
18 items, 2 FAIL
- **FAIL** `commit/contrast_rms` ['layer1.5', 'stem']
- **FAIL** `commit/edge_density` ['layer1.5', 'stem']
- PASS: 16 items

## panel_agreement
2 items, 0 FAIL
- **INFO** `penult/agreement_info[0,1]` {"panel_cka": 0.946, "relrep_row_corr_mean": 0.974, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.947, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.749, "error_consistency_test": 0.566, "cka_ood_c100": 0.676, "relrep_argmax_agree_ood_c100": 0.606, "relrep_argmax_chance_ood_c100": 0.123, "relrep_offmax_corr_ood_c100": 0.738, "landmark_procrustes_disparity": 0.004}
- PASS: 1 items
