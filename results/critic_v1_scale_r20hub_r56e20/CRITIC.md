# CRITIC  verdict: **PARTIAL**
runs: ['results/atlas_v1_resnet20_s0hub_st2', 'results/atlas_v1_resnet56_e20']
counts: {'PASS': 94, 'FAIL': 50, 'WARN': 1, 'INFO': 1}
layer alignment: position {"atlas_v1_resnet56_e20": [["stem", "stem"], ["layer1.0", "layer1.0"], ["layer1.1", "layer1.5"], ["layer1.2", "layer1.8"], ["layer2.0", "layer2.0"], ["layer2.1", "layer2.5"], ["layer2.2", "layer2.8"], ["layer3.0", "layer3.0"], ["layer3.1", "layer3.5"], ["layer3.2", "layer3.8"], ["penult", "penult"]]}

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
80 items, 42 FAIL
- **FAIL** `stem/neural_collapse.nc1` values=[259.926, 511.435] rel_spread=0.652
- **FAIL** `stem/hubness.k_occurrence_skew` values=[0.212, 0.081] rel_spread=0.898
- **FAIL** `layer1.0/twonn_id.id` values=[8.83, 6.675] rel_spread=0.278
- **FAIL** `layer1.0/pca_spectrum.participation_ratio` values=[4.042, 3.378] rel_spread=0.179
- **FAIL** `layer1.0/pca_spectrum.dim95` values=[6.0, 4.0] rel_spread=0.400
- **FAIL** `layer1.0/neural_collapse.nc1` values=[138.342, 204.383] rel_spread=0.385
- **FAIL** `layer1.0/hubness.k_occurrence_skew` values=[0.631, 0.179] rel_spread=1.116
- **FAIL** `layer1.1/pca_spectrum.participation_ratio` values=[5.171, 4.198] rel_spread=0.208
- **FAIL** `layer1.1/neural_collapse.nc1` values=[57.975, 97.56] rel_spread=0.509
- **FAIL** `layer1.1/neural_collapse.etf_deviation` values=[0.494, 0.589] rel_spread=0.175
- **FAIL** `layer1.1/hubness.k_occurrence_skew` values=[0.797, 0.609] rel_spread=0.268
- **FAIL** `layer1.2/pca_spectrum.participation_ratio` values=[5.518, 4.589] rel_spread=0.184
- **FAIL** `layer1.2/pca_spectrum.dim95` values=[9.0, 7.0] rel_spread=0.250
- **FAIL** `layer1.2/hubness.k_occurrence_skew` values=[0.871, 0.65] rel_spread=0.291
- **FAIL** `layer2.0/twonn_id.id` values=[12.102, 9.736] rel_spread=0.217
- **FAIL** `layer2.0/pca_spectrum.participation_ratio` values=[6.816, 4.903] rel_spread=0.327
- **FAIL** `layer2.0/pca_spectrum.dim95` values=[13.0, 8.0] rel_spread=0.476
- **FAIL** `layer2.0/neural_collapse.nc1` values=[21.892, 46.474] rel_spread=0.719
- **FAIL** `layer2.0/hubness.k_occurrence_skew` values=[0.962, 0.674] rel_spread=0.351
- **FAIL** `layer2.1/twonn_id.id` values=[13.24, 11.28] rel_spread=0.160
- **FAIL** `layer2.1/pca_spectrum.dim95` values=[15.0, 11.0] rel_spread=0.308
- **FAIL** `layer2.1/neural_collapse.nc1` values=[12.411, 26.113] rel_spread=0.711
- **FAIL** `layer2.1/hubness.k_occurrence_skew` values=[1.239, 0.924] rel_spread=0.292
- **FAIL** `layer2.2/twonn_id.id` values=[14.079, 12.041] rel_spread=0.156
- **FAIL** `layer2.2/pca_spectrum.participation_ratio` values=[9.005, 7.338] rel_spread=0.204
- **FAIL** `layer2.2/pca_spectrum.dim95` values=[18.0, 13.0] rel_spread=0.323
- **FAIL** `layer2.2/neural_collapse.nc1` values=[7.065, 14.627] rel_spread=0.697
- **FAIL** `layer2.2/hubness.k_occurrence_skew` values=[1.241, 1.046] rel_spread=0.170
- **FAIL** `layer3.0/twonn_id.id` values=[18.102, 14.537] rel_spread=0.218
- **FAIL** `layer3.0/pca_spectrum.participation_ratio` values=[12.136, 7.399] rel_spread=0.485
- **FAIL** `layer3.0/pca_spectrum.dim95` values=[37.0, 19.0] rel_spread=0.643
- **FAIL** `layer3.0/class_centers.sep_ratio` values=[0.573, 0.492] rel_spread=0.154
- **FAIL** `layer3.0/neural_collapse.nc1` values=[2.416, 5.17] rel_spread=0.726
- **FAIL** `layer3.0/neural_collapse.etf_deviation` values=[0.308, 0.384] rel_spread=0.219
- **FAIL** `layer3.0/hubness.k_occurrence_skew` values=[1.829, 1.216] rel_spread=0.402
- **FAIL** `layer3.1/pca_spectrum.dim95` values=[41.0, 33.0] rel_spread=0.216
- **FAIL** `layer3.1/neural_collapse.nc1` values=[0.759, 0.986] rel_spread=0.260
- **FAIL** `penult/twonn_id.id` values=[9.915, 11.74] rel_spread=0.169
- **FAIL** `penult/class_centers.sep_ratio` values=[2.996, 2.04] rel_spread=0.380
- **FAIL** `penult/neural_collapse.nc1` values=[0.17, 0.32] rel_spread=0.613
- **FAIL** `penult/neural_collapse.etf_deviation` values=[0.107, 0.185] rel_spread=0.539
- **FAIL** `penult/hubness.k_occurrence_skew` values=[0.763, 0.9] rel_spread=0.164
- PASS: 38 items

## id_profile_stability
1 items, 1 FAIL
- **FAIL** `id_profile[0,1]` spearman=0.879 peak=layer3.1/layer3.1

## adjacency_stability
20 items, 0 FAIL
- PASS: 20 items

## decodability_stability
18 items, 5 FAIL
- **FAIL** `decod/contrast_rms[0,1]` profile_spearman=0.52 mean_abs_delta=0.050
- **FAIL** `decod/spectral_anisotropy[0,1]` profile_spearman=0.65 mean_abs_delta=0.046
- **FAIL** `decod/noise_sigma[0,1]` profile_spearman=0.32 mean_abs_delta=0.059
- **FAIL** `decod/colorfulness[0,1]` profile_spearman=0.90 mean_abs_delta=0.107
- **FAIL** `decod/edge_density[0,1]` profile_spearman=0.41 mean_abs_delta=0.059
- PASS: 13 items

## commit_agreement
18 items, 2 FAIL
- **FAIL** `commit/highfreq_ratio` ['layer1.0', 'layer2.0']
- **FAIL** `commit/spectral_anisotropy` ['layer2.0', 'layer3.0']
- PASS: 16 items

## panel_agreement
2 items, 0 FAIL
- **INFO** `penult/agreement_info[0,1]` {"panel_cka": 0.84, "relrep_row_corr_mean": 0.927, "relrep_argmax_agree": 0.953, "relrep_argmax_agree_test": 0.894, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.825, "error_consistency_test": 0.465, "cka_ood_c100": 0.65, "relrep_argmax_agree_ood_c100": 0.547, "relrep_argmax_chance_ood_c100": 0.123, "relrep_offmax_corr_ood_c100": 0.751, "landmark_procrustes_disparity": 0.024}
- PASS: 1 items
