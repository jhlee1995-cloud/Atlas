# CRITIC  verdict: **PARTIAL**
runs: ['results/atlas_v1_resnet20_s0hub_st2', 'results/atlas_v1_resnet56_s0hub']
counts: {'PASS': 120, 'FAIL': 24, 'WARN': 1, 'INFO': 1}
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
80 items, 19 FAIL
- **FAIL** `stem/neural_collapse.nc1` values=[259.926, 494.389] rel_spread=0.622
- **FAIL** `stem/hubness.k_occurrence_skew` values=[0.212, 0.146] rel_spread=0.365
- **FAIL** `layer1.0/pca_spectrum.participation_ratio` values=[4.042, 3.185] rel_spread=0.237
- **FAIL** `layer1.0/pca_spectrum.dim95` values=[6.0, 5.0] rel_spread=0.182
- **FAIL** `layer1.0/neural_collapse.nc1` values=[138.342, 171.954] rel_spread=0.217
- **FAIL** `layer1.0/hubness.k_occurrence_skew` values=[0.631, 0.381] rel_spread=0.493
- **FAIL** `layer2.0/pca_spectrum.dim95` values=[13.0, 11.0] rel_spread=0.167
- **FAIL** `layer2.2/hubness.k_occurrence_skew` values=[1.241, 1.478] rel_spread=0.175
- **FAIL** `layer3.0/twonn_id.id` values=[18.102, 15.572] rel_spread=0.150
- **FAIL** `layer3.0/pca_spectrum.participation_ratio` values=[12.136, 10.033] rel_spread=0.190
- **FAIL** `layer3.0/pca_spectrum.dim95` values=[37.0, 25.0] rel_spread=0.387
- **FAIL** `layer3.0/neural_collapse.nc1` values=[2.416, 3.502] rel_spread=0.367
- **FAIL** `layer3.0/hubness.k_occurrence_skew` values=[1.829, 1.404] rel_spread=0.263
- **FAIL** `layer3.1/pca_spectrum.participation_ratio` values=[10.468, 15.827] rel_spread=0.408
- **FAIL** `layer3.1/neural_collapse.etf_deviation` values=[0.275, 0.21] rel_spread=0.270
- **FAIL** `penult/class_centers.sep_ratio` values=[2.996, 5.332] rel_spread=0.561
- **FAIL** `penult/neural_collapse.nc1` values=[0.17, 0.049] rel_spread=1.108
- **FAIL** `penult/neural_collapse.etf_deviation` values=[0.107, 0.054] rel_spread=0.663
- **FAIL** `penult/hubness.k_occurrence_skew` values=[0.763, 0.94] rel_spread=0.208
- PASS: 61 items

## id_profile_stability
1 items, 0 FAIL
- PASS: 1 items

## adjacency_stability
20 items, 1 FAIL
- **FAIL** `layer2.2/merge_kendall[0,1]` tau=0.506
- PASS: 19 items

## decodability_stability
18 items, 2 FAIL
- **FAIL** `decod/contrast_rms[0,1]` profile_spearman=0.68 mean_abs_delta=0.038
- **FAIL** `decod/noise_sigma[0,1]` profile_spearman=0.70 mean_abs_delta=0.029
- PASS: 16 items

## commit_agreement
18 items, 2 FAIL
- **FAIL** `commit/highfreq_ratio` ['layer1.0', 'layer2.0']
- **FAIL** `commit/spectral_anisotropy` ['layer2.0', 'layer3.0']
- PASS: 16 items

## panel_agreement
2 items, 0 FAIL
- **INFO** `penult/agreement_info[0,1]` {"panel_cka": 0.9, "relrep_row_corr_mean": 0.948, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.927, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.717, "error_consistency_test": 0.526, "cka_ood_c100": 0.644, "relrep_argmax_agree_ood_c100": 0.571, "relrep_argmax_chance_ood_c100": 0.124, "relrep_offmax_corr_ood_c100": 0.716, "landmark_procrustes_disparity": 0.014}
- PASS: 1 items
