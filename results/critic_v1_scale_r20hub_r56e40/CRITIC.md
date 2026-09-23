# CRITIC  verdict: **PARTIAL**
runs: ['results/atlas_v1_resnet20_s0hub_st2', 'results/atlas_v1_resnet56_e40']
counts: {'PASS': 113, 'FAIL': 31, 'WARN': 1, 'INFO': 1}
layer alignment: position {"atlas_v1_resnet56_e40": [["stem", "stem"], ["layer1.0", "layer1.0"], ["layer1.1", "layer1.5"], ["layer1.2", "layer1.8"], ["layer2.0", "layer2.0"], ["layer2.1", "layer2.5"], ["layer2.2", "layer2.8"], ["layer3.0", "layer3.0"], ["layer3.1", "layer3.5"], ["layer3.2", "layer3.8"], ["penult", "penult"]]}

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
80 items, 25 FAIL
- **FAIL** `stem/pca_spectrum.dim95` values=[4.0, 5.0] rel_spread=0.222
- **FAIL** `stem/neural_collapse.nc1` values=[259.926, 147.951] rel_spread=0.549
- **FAIL** `stem/hubness.k_occurrence_skew` values=[0.212, 0.264] rel_spread=0.219
- **FAIL** `layer1.0/neural_collapse.nc1` values=[138.342, 99.826] rel_spread=0.323
- **FAIL** `layer1.0/hubness.k_occurrence_skew` values=[0.631, 0.454] rel_spread=0.327
- **FAIL** `layer1.1/pca_spectrum.participation_ratio` values=[5.171, 4.255] rel_spread=0.194
- **FAIL** `layer1.1/class_centers.sep_ratio` values=[0.21, 0.284] rel_spread=0.300
- **FAIL** `layer1.1/neural_collapse.nc1` values=[57.975, 103.293] rel_spread=0.562
- **FAIL** `layer1.1/hubness.k_occurrence_skew` values=[0.797, 0.677] rel_spread=0.164
- **FAIL** `layer1.2/neural_collapse.nc1` values=[53.174, 72.8] rel_spread=0.312
- **FAIL** `layer1.2/hubness.k_occurrence_skew` values=[0.871, 0.733] rel_spread=0.172
- **FAIL** `layer2.0/pca_spectrum.participation_ratio` values=[6.816, 5.318] rel_spread=0.247
- **FAIL** `layer2.0/pca_spectrum.dim95` values=[13.0, 10.0] rel_spread=0.261
- **FAIL** `layer2.0/neural_collapse.nc1` values=[21.892, 29.116] rel_spread=0.283
- **FAIL** `layer2.1/pca_spectrum.participation_ratio` values=[7.778, 6.078] rel_spread=0.245
- **FAIL** `layer2.2/pca_spectrum.dim95` values=[18.0, 15.0] rel_spread=0.182
- **FAIL** `layer2.2/neural_collapse.nc1` values=[7.065, 8.688] rel_spread=0.206
- **FAIL** `layer3.0/twonn_id.id` values=[18.102, 14.789] rel_spread=0.201
- **FAIL** `layer3.0/pca_spectrum.participation_ratio` values=[12.136, 8.419] rel_spread=0.362
- **FAIL** `layer3.0/pca_spectrum.dim95` values=[37.0, 22.0] rel_spread=0.508
- **FAIL** `layer3.0/neural_collapse.nc1` values=[2.416, 3.888] rel_spread=0.467
- **FAIL** `layer3.0/hubness.k_occurrence_skew` values=[1.829, 1.398] rel_spread=0.267
- **FAIL** `layer3.1/pca_spectrum.participation_ratio` values=[10.468, 13.752] rel_spread=0.271
- **FAIL** `layer3.1/neural_collapse.etf_deviation` values=[0.275, 0.221] rel_spread=0.220
- **FAIL** `penult/hubness.k_occurrence_skew` values=[0.763, 0.958] rel_spread=0.226
- PASS: 55 items

## id_profile_stability
1 items, 0 FAIL
- PASS: 1 items

## adjacency_stability
20 items, 1 FAIL
- **FAIL** `layer2.2/merge_kendall[0,1]` tau=0.544
- PASS: 19 items

## decodability_stability
18 items, 3 FAIL
- **FAIL** `decod/contrast_rms[0,1]` profile_spearman=0.55 mean_abs_delta=0.044
- **FAIL** `decod/noise_sigma[0,1]` profile_spearman=0.68 mean_abs_delta=0.042
- **FAIL** `decod/edge_density[0,1]` profile_spearman=0.60 mean_abs_delta=0.057
- PASS: 15 items

## commit_agreement
18 items, 2 FAIL
- **FAIL** `commit/spectral_slope` ['layer2.0', 'layer1.1']
- **FAIL** `commit/corruption_family` ['layer2.0', 'layer1.1']
- PASS: 16 items

## panel_agreement
2 items, 0 FAIL
- **INFO** `penult/agreement_info[0,1]` {"panel_cka": 0.908, "relrep_row_corr_mean": 0.96, "relrep_argmax_agree": 1.0, "relrep_argmax_agree_test": 0.914, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.804, "error_consistency_test": 0.502, "cka_ood_c100": 0.667, "relrep_argmax_agree_ood_c100": 0.555, "relrep_argmax_chance_ood_c100": 0.125, "relrep_offmax_corr_ood_c100": 0.752, "landmark_procrustes_disparity": 0.004}
- PASS: 1 items
