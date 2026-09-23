# CRITIC  verdict: **PARTIAL**
runs: ['results/atlas_v1_resnet56_s0hub_st3', 'results/atlas_v1_resnet56_s1', 'results/atlas_v1_resnet56_s2']
counts: {'PASS': 197, 'FAIL': 29, 'WARN': 1, 'INFO': 3}

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
3 items, 0 FAIL
- PASS: 3 items

## probe_hygiene
3 items, 0 FAIL
- PASS: 3 items

## scalar_stability
80 items, 18 FAIL
- **FAIL** `stem/pca_spectrum.dim95` values=[4.0, 3.0, 4.0] rel_spread=0.273
- **FAIL** `stem/neural_collapse.nc1` values=[494.389, 499.245, 213.957] rel_spread=0.709
- **FAIL** `stem/hubness.k_occurrence_skew` values=[0.146, 0.096, 0.165] rel_spread=0.503
- **FAIL** `layer1.0/pca_spectrum.participation_ratio` values=[3.185, 2.623, 3.577] rel_spread=0.305
- **FAIL** `layer1.0/pca_spectrum.dim95` values=[5.0, 4.0, 5.0] rel_spread=0.214
- **FAIL** `layer1.0/neural_collapse.nc1` values=[171.954, 264.537, 133.13] rel_spread=0.692
- **FAIL** `layer1.0/hubness.k_occurrence_skew` values=[0.381, 0.271, 0.388] rel_spread=0.340
- **FAIL** `layer1.5/pca_spectrum.participation_ratio` values=[5.118, 4.753, 5.67] rel_spread=0.177
- **FAIL** `layer1.5/pca_spectrum.dim95` values=[8.0, 7.0, 9.0] rel_spread=0.250
- **FAIL** `layer1.5/class_centers.sep_ratio` values=[0.235, 0.235, 0.297] rel_spread=0.244
- **FAIL** `layer1.5/neural_collapse.nc1` values=[53.611, 68.49, 86.672] rel_spread=0.475
- **FAIL** `layer1.8/pca_spectrum.dim95` values=[9.0, 8.0, 10.0] rel_spread=0.222
- **FAIL** `layer1.8/neural_collapse.nc1` values=[56.519, 52.914, 66.033] rel_spread=0.224
- **FAIL** `layer2.0/neural_collapse.nc1` values=[24.111, 30.982, 16.817] rel_spread=0.591
- **FAIL** `layer2.0/hubness.k_occurrence_skew` values=[0.91, 0.835, 0.992] rel_spread=0.172
- **FAIL** `layer2.5/neural_collapse.nc1` values=[10.911, 12.951, 9.034] rel_spread=0.357
- **FAIL** `layer2.8/neural_collapse.nc1` values=[6.224, 7.266, 5.407] rel_spread=0.295
- **FAIL** `layer3.5/hubness.k_occurrence_skew` values=[1.879, 2.042, 1.745] rel_spread=0.157
- PASS: 62 items

## id_profile_stability
3 items, 0 FAIL
- PASS: 3 items

## adjacency_stability
60 items, 2 FAIL
- **FAIL** `layer2.5/merge_kendall[0,2]` tau=0.566
- **FAIL** `layer2.5/merge_kendall[1,2]` tau=0.508
- PASS: 58 items

## decodability_stability
54 items, 5 FAIL
- **FAIL** `decod/contrast_rms[1,2]` profile_spearman=0.41 mean_abs_delta=0.048
- **FAIL** `decod/saturation_mean[1,2]` profile_spearman=0.53 mean_abs_delta=0.082
- **FAIL** `decod/colorfulness[1,2]` profile_spearman=0.61 mean_abs_delta=0.071
- **FAIL** `decod/edge_density[0,1]` profile_spearman=0.58 mean_abs_delta=0.049
- **FAIL** `decod/edge_density[1,2]` profile_spearman=0.65 mean_abs_delta=0.043
- PASS: 49 items

## commit_agreement
18 items, 4 FAIL
- **FAIL** `commit/contrast_rms` ['stem', 'layer1.5', 'stem']
- **FAIL** `commit/spectral_anisotropy` ['layer3.0', 'layer2.0', 'layer2.0']
- **FAIL** `commit/edge_density` ['layer1.0', 'layer1.5', 'stem']
- **FAIL** `commit/severity` ['layer3.0', 'layer2.0', 'layer2.0']
- PASS: 14 items

## panel_agreement
6 items, 0 FAIL
- **INFO** `penult/agreement_info[0,1]` {"panel_cka": 0.951, "relrep_row_corr_mean": 0.973, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.953, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.737, "error_consistency_test": 0.543, "cka_ood_c100": 0.683, "relrep_argmax_agree_ood_c100": 0.62, "relrep_argmax_chance_ood_c100": 0.123, "relrep_offmax_corr_ood_c100": 0.736, "landmark_procrustes_disparity": 0.003}
- **INFO** `penult/agreement_info[0,2]` {"panel_cka": 0.961, "relrep_row_corr_mean": 0.977, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.947, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.719, "error_consistency_test": 0.569, "cka_ood_c100": 0.681, "relrep_argmax_agree_ood_c100": 0.632, "relrep_argmax_chance_ood_c100": 0.123, "relrep_offmax_corr_ood_c100": 0.737, "landmark_procrustes_disparity": 0.004}
- **INFO** `penult/agreement_info[1,2]` {"panel_cka": 0.946, "relrep_row_corr_mean": 0.974, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.947, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.749, "error_consistency_test": 0.566, "cka_ood_c100": 0.676, "relrep_argmax_agree_ood_c100": 0.606, "relrep_argmax_chance_ood_c100": 0.123, "relrep_offmax_corr_ood_c100": 0.738, "landmark_procrustes_disparity": 0.004}
- PASS: 3 items
