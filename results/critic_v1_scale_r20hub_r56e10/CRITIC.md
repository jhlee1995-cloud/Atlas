# CRITIC  verdict: **PARTIAL**
runs: ['results/atlas_v1_resnet20_s0hub_st2', 'results/atlas_v1_resnet56_e10']
counts: {'PASS': 90, 'FAIL': 54, 'WARN': 1, 'INFO': 1}
layer alignment: position {"atlas_v1_resnet56_e10": [["stem", "stem"], ["layer1.0", "layer1.0"], ["layer1.1", "layer1.5"], ["layer1.2", "layer1.8"], ["layer2.0", "layer2.0"], ["layer2.1", "layer2.5"], ["layer2.2", "layer2.8"], ["layer3.0", "layer3.0"], ["layer3.1", "layer3.5"], ["layer3.2", "layer3.8"], ["penult", "penult"]]}

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
80 items, 41 FAIL
- **FAIL** `stem/pca_spectrum.participation_ratio` values=[2.979, 2.483] rel_spread=0.181
- **FAIL** `stem/neural_collapse.nc1` values=[259.926, 456.735] rel_spread=0.549
- **FAIL** `layer1.0/twonn_id.id` values=[8.83, 7.37] rel_spread=0.180
- **FAIL** `layer1.0/pca_spectrum.participation_ratio` values=[4.042, 3.019] rel_spread=0.290
- **FAIL** `layer1.0/pca_spectrum.dim95` values=[6.0, 5.0] rel_spread=0.182
- **FAIL** `layer1.0/neural_collapse.nc1` values=[138.342, 462.848] rel_spread=1.080
- **FAIL** `layer1.0/hubness.k_occurrence_skew` values=[0.631, 0.41] rel_spread=0.423
- **FAIL** `layer1.1/pca_spectrum.participation_ratio` values=[5.171, 4.067] rel_spread=0.239
- **FAIL** `layer1.1/neural_collapse.nc1` values=[57.975, 106.412] rel_spread=0.589
- **FAIL** `layer1.2/class_centers.sep_ratio` values=[0.287, 0.377] rel_spread=0.271
- **FAIL** `layer1.2/neural_collapse.nc1` values=[53.174, 70.94] rel_spread=0.286
- **FAIL** `layer1.2/hubness.k_occurrence_skew` values=[0.871, 0.699] rel_spread=0.219
- **FAIL** `layer2.0/pca_spectrum.participation_ratio` values=[6.816, 4.691] rel_spread=0.369
- **FAIL** `layer2.0/pca_spectrum.dim95` values=[13.0, 9.0] rel_spread=0.364
- **FAIL** `layer2.0/class_centers.sep_ratio` values=[0.333, 0.414] rel_spread=0.217
- **FAIL** `layer2.1/pca_spectrum.participation_ratio` values=[7.778, 5.91] rel_spread=0.273
- **FAIL** `layer2.1/pca_spectrum.dim95` values=[15.0, 12.0] rel_spread=0.222
- **FAIL** `layer2.1/class_centers.sep_ratio` values=[0.386, 0.534] rel_spread=0.322
- **FAIL** `layer2.1/neural_collapse.nc1` values=[12.411, 10.21] rel_spread=0.195
- **FAIL** `layer2.2/pca_spectrum.participation_ratio` values=[9.005, 6.087] rel_spread=0.387
- **FAIL** `layer2.2/pca_spectrum.dim95` values=[18.0, 12.0] rel_spread=0.400
- **FAIL** `layer2.2/class_centers.sep_ratio` values=[0.429, 0.596] rel_spread=0.326
- **FAIL** `layer2.2/neural_collapse.etf_deviation` values=[0.405, 0.338] rel_spread=0.180
- **FAIL** `layer2.2/hubness.k_occurrence_skew` values=[1.241, 1.054] rel_spread=0.163
- **FAIL** `layer3.0/twonn_id.id` values=[18.102, 14.136] rel_spread=0.246
- **FAIL** `layer3.0/pca_spectrum.participation_ratio` values=[12.136, 6.056] rel_spread=0.668
- **FAIL** `layer3.0/pca_spectrum.dim95` values=[37.0, 15.0] rel_spread=0.846
- **FAIL** `layer3.0/class_centers.sep_ratio` values=[0.573, 0.695] rel_spread=0.192
- **FAIL** `layer3.0/neural_collapse.nc1` values=[2.416, 3.488] rel_spread=0.363
- **FAIL** `layer3.0/neural_collapse.etf_deviation` values=[0.308, 0.385] rel_spread=0.222
- **FAIL** `layer3.0/hubness.k_occurrence_skew` values=[1.829, 1.144] rel_spread=0.461
- **FAIL** `layer3.1/twonn_id.id` values=[19.354, 15.481] rel_spread=0.222
- **FAIL** `layer3.1/pca_spectrum.participation_ratio` values=[10.468, 7.937] rel_spread=0.275
- **FAIL** `layer3.1/pca_spectrum.dim95` values=[41.0, 21.0] rel_spread=0.645
- **FAIL** `layer3.1/neural_collapse.nc1` values=[0.759, 1.251] rel_spread=0.489
- **FAIL** `layer3.1/hubness.k_occurrence_skew` values=[1.907, 1.478] rel_spread=0.253
- **FAIL** `penult/twonn_id.id` values=[9.915, 11.984] rel_spread=0.189
- **FAIL** `penult/pca_spectrum.participation_ratio` values=[8.467, 6.428] rel_spread=0.274
- **FAIL** `penult/class_centers.sep_ratio` values=[2.996, 1.453] rel_spread=0.694
- **FAIL** `penult/neural_collapse.nc1` values=[0.17, 0.686] rel_spread=1.206
- **FAIL** `penult/neural_collapse.etf_deviation` values=[0.107, 0.278] rel_spread=0.893
- PASS: 39 items

## id_profile_stability
1 items, 0 FAIL
- PASS: 1 items

## adjacency_stability
20 items, 2 FAIL
- **FAIL** `layer2.1/merge_kendall[0,1]` tau=0.483
- **FAIL** `layer2.2/merge_kendall[0,1]` tau=0.263
- PASS: 18 items

## decodability_stability
18 items, 6 FAIL
- **FAIL** `decod/luminance_mean[0,1]` profile_spearman=0.99 mean_abs_delta=0.104
- **FAIL** `decod/contrast_rms[0,1]` profile_spearman=0.53 mean_abs_delta=0.062
- **FAIL** `decod/highfreq_ratio[0,1]` profile_spearman=0.58 mean_abs_delta=0.080
- **FAIL** `decod/noise_sigma[0,1]` profile_spearman=0.49 mean_abs_delta=0.051
- **FAIL** `decod/colorfulness[0,1]` profile_spearman=0.82 mean_abs_delta=0.202
- **FAIL** `decod/corruption_type[0,1]` profile_spearman=0.98 mean_abs_delta=0.125
- PASS: 12 items

## commit_agreement
18 items, 4 FAIL
- **FAIL** `commit/spectral_slope` ['layer2.0', 'layer1.1']
- **FAIL** `commit/spectral_anisotropy` ['layer2.0', 'layer1.1']
- **FAIL** `commit/coarse_animal_vehicle` ['layer3.0', 'layer2.1']
- **FAIL** `commit/severity` ['layer3.0', 'layer2.0']
- PASS: 14 items

## panel_agreement
2 items, 1 FAIL
- **FAIL** `penult/cka_test[0,1]` cka_test=0.766 min=0.8
- **INFO** `penult/agreement_info[0,1]` {"panel_cka": 0.741, "relrep_row_corr_mean": 0.851, "relrep_argmax_agree": 0.906, "relrep_argmax_agree_test": 0.853, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.807, "error_consistency_test": 0.379, "cka_ood_c100": 0.587, "relrep_argmax_agree_ood_c100": 0.501, "relrep_argmax_chance_ood_c100": 0.122, "relrep_offmax_corr_ood_c100": 0.714, "landmark_procrustes_disparity": 0.084}
