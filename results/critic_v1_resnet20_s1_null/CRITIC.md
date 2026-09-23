# CRITIC  verdict: **DOES_NOT_REPLICATE**
runs: ['results/atlas_v1_resnet20_s1', 'results/atlas_v1_resnet20_rand']
counts: {'PASS': 49, 'FAIL': 95, 'WARN': 1, 'INFO': 1}

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
80 items, 58 FAIL
- **FAIL** `stem/pca_spectrum.participation_ratio` values=[2.993, 2.095] rel_spread=0.353
- **FAIL** `stem/neural_collapse.nc1` values=[261.068, 352.45] rel_spread=0.298
- **FAIL** `stem/hubness.k_occurrence_skew` values=[0.179, 0.261] rel_spread=0.374
- **FAIL** `layer1.0/pca_spectrum.participation_ratio` values=[3.288, 2.068] rel_spread=0.456
- **FAIL** `layer1.0/neural_collapse.nc1` values=[164.627, 304.282] rel_spread=0.596
- **FAIL** `layer1.0/class_centers.nearest_center_acc_test` values=[0.28, 0.217] rel_spread=0.256
- **FAIL** `layer1.1/pca_spectrum.participation_ratio` values=[4.07, 2.466] rel_spread=0.491
- **FAIL** `layer1.1/pca_spectrum.dim95` values=[7.0, 6.0] rel_spread=0.154
- **FAIL** `layer1.1/class_centers.sep_ratio` values=[0.209, 0.154] rel_spread=0.304
- **FAIL** `layer1.1/neural_collapse.nc1` values=[74.307, 265.151] rel_spread=1.124
- **FAIL** `layer1.1/class_centers.nearest_center_acc_test` values=[0.317, 0.215] rel_spread=0.386
- **FAIL** `layer1.2/pca_spectrum.participation_ratio` values=[4.8, 2.024] rel_spread=0.814
- **FAIL** `layer1.2/pca_spectrum.dim95` values=[8.0, 5.0] rel_spread=0.462
- **FAIL** `layer1.2/class_centers.sep_ratio` values=[0.303, 0.148] rel_spread=0.686
- **FAIL** `layer1.2/neural_collapse.nc1` values=[49.246, 182.024] rel_spread=1.148
- **FAIL** `layer1.2/hubness.k_occurrence_skew` values=[0.713, 0.566] rel_spread=0.230
- **FAIL** `layer1.2/class_centers.nearest_center_acc_test` values=[0.353, 0.207] rel_spread=0.519
- **FAIL** `layer2.0/pca_spectrum.participation_ratio` values=[6.067, 2.091] rel_spread=0.975
- **FAIL** `layer2.0/pca_spectrum.dim95` values=[12.0, 9.0] rel_spread=0.286
- **FAIL** `layer2.0/class_centers.sep_ratio` values=[0.345, 0.152] rel_spread=0.778
- **FAIL** `layer2.0/neural_collapse.nc1` values=[23.829, 94.665] rel_spread=1.196
- **FAIL** `layer2.0/class_centers.nearest_center_acc_test` values=[0.42, 0.204] rel_spread=0.691
- **FAIL** `layer2.1/pca_spectrum.participation_ratio` values=[7.692, 2.052] rel_spread=1.158
- **FAIL** `layer2.1/pca_spectrum.dim95` values=[14.0, 9.0] rel_spread=0.435
- **FAIL** `layer2.1/class_centers.sep_ratio` values=[0.394, 0.15] rel_spread=0.897
- **FAIL** `layer2.1/neural_collapse.nc1` values=[17.043, 102.537] rel_spread=1.430
- **FAIL** `layer2.1/neural_collapse.etf_deviation` values=[0.401, 0.67] rel_spread=0.503
- **FAIL** `layer2.1/class_centers.nearest_center_acc_test` values=[0.476, 0.199] rel_spread=0.821
- **FAIL** `layer2.2/pca_spectrum.participation_ratio` values=[8.577, 2.432] rel_spread=1.116
- **FAIL** `layer2.2/pca_spectrum.dim95` values=[17.0, 10.0] rel_spread=0.519
- **FAIL** `layer2.2/class_centers.sep_ratio` values=[0.42, 0.153] rel_spread=0.929
- **FAIL** `layer2.2/neural_collapse.nc1` values=[9.177, 120.398] rel_spread=1.717
- **FAIL** `layer2.2/neural_collapse.etf_deviation` values=[0.36, 0.699] rel_spread=0.641
- **FAIL** `layer2.2/hubness.k_occurrence_skew` values=[1.427, 1.227] rel_spread=0.150
- **FAIL** `layer2.2/class_centers.nearest_center_acc_test` values=[0.525, 0.198] rel_spread=0.904
- **FAIL** `layer3.0/twonn_id.id` values=[18.012, 23.577] rel_spread=0.268
- **FAIL** `layer3.0/pca_spectrum.participation_ratio` values=[10.286, 3.579] rel_spread=0.967
- **FAIL** `layer3.0/class_centers.sep_ratio` values=[0.566, 0.144] rel_spread=1.189
- **FAIL** `layer3.0/neural_collapse.nc1` values=[2.391, 70.632] rel_spread=1.869
- **FAIL** `layer3.0/neural_collapse.etf_deviation` values=[0.329, 0.694] rel_spread=0.714
- **FAIL** `layer3.0/hubness.k_occurrence_skew` values=[1.691, 2.922] rel_spread=0.534
- **FAIL** `layer3.0/class_centers.nearest_center_acc_test` values=[0.649, 0.205] rel_spread=1.041
- **FAIL** `layer3.1/twonn_id.id` values=[19.446, 24.161] rel_spread=0.216
- **FAIL** `layer3.1/pca_spectrum.participation_ratio` values=[11.259, 3.17] rel_spread=1.121
- **FAIL** `layer3.1/pca_spectrum.dim95` values=[41.0, 33.0] rel_spread=0.216
- **FAIL** `layer3.1/class_centers.sep_ratio` values=[0.873, 0.135] rel_spread=1.464
- **FAIL** `layer3.1/neural_collapse.nc1` values=[0.774, 74.195] rel_spread=1.959
- **FAIL** `layer3.1/neural_collapse.etf_deviation` values=[0.289, 0.76] rel_spread=0.899
- **FAIL** `layer3.1/hubness.k_occurrence_skew` values=[1.898, 2.432] rel_spread=0.247
- **FAIL** `layer3.1/class_centers.nearest_center_acc_test` values=[0.813, 0.195] rel_spread=1.226
- **FAIL** `penult/twonn_id.id` values=[9.757, 23.671] rel_spread=0.832
- **FAIL** `penult/pca_spectrum.participation_ratio` values=[8.602, 2.696] rel_spread=1.045
- **FAIL** `penult/pca_spectrum.dim95` values=[9.0, 30.0] rel_spread=1.077
- **FAIL** `penult/class_centers.sep_ratio` values=[3.067, 0.13] rel_spread=1.837
- **FAIL** `penult/neural_collapse.nc1` values=[0.163, 72.471] rel_spread=1.991
- **FAIL** `penult/neural_collapse.etf_deviation` values=[0.098, 0.778] rel_spread=1.554
- **FAIL** `penult/hubness.k_occurrence_skew` values=[0.833, 2.59] rel_spread=1.026
- **FAIL** `penult/class_centers.nearest_center_acc_test` values=[0.922, 0.192] rel_spread=1.310
- PASS: 22 items

## id_profile_stability
1 items, 1 FAIL
- **FAIL** `id_profile[0,1]` spearman=0.867 peak=layer3.1/layer3.1

## adjacency_stability
20 items, 9 FAIL
- **FAIL** `layer1.2/adjacency_spearman[0,1]` rho=0.589
- **FAIL** `layer1.2/merge_kendall[0,1]` tau=0.062
- **FAIL** `layer2.0/adjacency_spearman[0,1]` rho=0.671
- **FAIL** `layer2.2/merge_kendall[0,1]` tau=0.470
- **FAIL** `layer3.0/merge_kendall[0,1]` tau=0.433
- **FAIL** `layer3.1/adjacency_spearman[0,1]` rho=0.600
- **FAIL** `layer3.1/merge_kendall[0,1]` tau=0.227
- **FAIL** `penult/adjacency_spearman[0,1]` rho=0.478
- **FAIL** `penult/merge_kendall[0,1]` tau=0.279
- PASS: 11 items

## decodability_stability
18 items, 16 FAIL
- **FAIL** `decod/luminance_mean[0,1]` profile_spearman=1.00 mean_abs_delta=0.108
- **FAIL** `decod/highfreq_ratio[0,1]` profile_spearman=-0.15 mean_abs_delta=0.262
- **FAIL** `decod/spectral_slope[0,1]` profile_spearman=-0.41 mean_abs_delta=0.349
- **FAIL** `decod/spectral_anisotropy[0,1]` profile_spearman=0.08 mean_abs_delta=0.137
- **FAIL** `decod/noise_sigma[0,1]` profile_spearman=0.36 mean_abs_delta=0.183
- **FAIL** `decod/hue_cos[0,1]` profile_spearman=0.81 mean_abs_delta=0.134
- **FAIL** `decod/hue_sin[0,1]` profile_spearman=0.95 mean_abs_delta=0.170
- **FAIL** `decod/colorfulness[0,1]` profile_spearman=0.87 mean_abs_delta=0.166
- **FAIL** `decod/edge_density[0,1]` profile_spearman=0.54 mean_abs_delta=0.109
- **FAIL** `decod/orientation_entropy[0,1]` profile_spearman=0.04 mean_abs_delta=0.259
- **FAIL** `decod/blockiness[0,1]` profile_spearman=0.18 mean_abs_delta=0.056
- **FAIL** `decod/class[0,1]` profile_spearman=-0.82 mean_abs_delta=0.297
- **FAIL** `decod/coarse_animal_vehicle[0,1]` profile_spearman=-0.94 mean_abs_delta=0.128
- **FAIL** `decod/corruption_family[0,1]` profile_spearman=-0.36 mean_abs_delta=0.187
- **FAIL** `decod/corruption_type[0,1]` profile_spearman=-0.10 mean_abs_delta=0.271
- **FAIL** `decod/severity[0,1]` profile_spearman=-0.35 mean_abs_delta=0.163
- PASS: 2 items

## commit_agreement
18 items, 10 FAIL
- **FAIL** `commit/highfreq_ratio` ['layer1.1', 'stem']
- **FAIL** `commit/spectral_slope` ['layer2.0', 'stem']
- **FAIL** `commit/spectral_anisotropy` ['layer1.2', 'stem']
- **FAIL** `commit/orientation_entropy` ['layer2.0', 'stem']
- **FAIL** `commit/blockiness` ['layer3.0', 'stem']
- **FAIL** `commit/class` ['layer3.1', 'stem']
- **FAIL** `commit/coarse_animal_vehicle` ['layer3.0', 'stem']
- **FAIL** `commit/corruption_family` ['layer2.0', 'stem']
- **FAIL** `commit/corruption_type` ['layer2.0', 'stem']
- **FAIL** `commit/severity` ['layer3.0', 'stem']
- PASS: 8 items

## panel_agreement
2 items, 1 FAIL
- **FAIL** `penult/cka_test[0,1]` cka_test=0.073 min=0.8
- **INFO** `penult/agreement_info[0,1]` {"panel_cka": 0.114, "relrep_row_corr_mean": 0.205, "relrep_argmax_agree": 0.172, "relrep_argmax_agree_test": 0.198, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.242, "error_consistency_test": -0.005, "cka_ood_c100": 0.053, "relrep_argmax_agree_ood_c100": 0.145, "relrep_argmax_chance_ood_c100": 0.107, "relrep_offmax_corr_ood_c100": 0.205, "landmark_procrustes_disparity": 0.758}
