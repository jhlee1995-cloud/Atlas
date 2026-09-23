# CRITIC  verdict: **DOES_NOT_REPLICATE**
runs: ['results/atlas_v1_resnet56_s0hub', 'results/atlas_v1_resnet56_rand']
counts: {'PASS': 40, 'FAIL': 104, 'WARN': 1, 'INFO': 1}

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
80 items, 65 FAIL
- **FAIL** `stem/pca_spectrum.participation_ratio` values=[2.942, 1.956] rel_spread=0.402
- **FAIL** `stem/neural_collapse.nc1` values=[494.389, 410.593] rel_spread=0.185
- **FAIL** `stem/hubness.k_occurrence_skew` values=[0.146, 0.286] rel_spread=0.645
- **FAIL** `layer1.0/pca_spectrum.participation_ratio` values=[3.185, 1.901] rel_spread=0.505
- **FAIL** `layer1.0/neural_collapse.nc1` values=[171.954, 295.537] rel_spread=0.529
- **FAIL** `layer1.0/hubness.k_occurrence_skew` values=[0.381, 0.469] rel_spread=0.206
- **FAIL** `layer1.5/pca_spectrum.participation_ratio` values=[5.118, 1.613] rel_spread=1.042
- **FAIL** `layer1.5/pca_spectrum.dim95` values=[8.0, 5.0] rel_spread=0.462
- **FAIL** `layer1.5/class_centers.sep_ratio` values=[0.235, 0.166] rel_spread=0.342
- **FAIL** `layer1.5/neural_collapse.nc1` values=[53.611, 330.355] rel_spread=1.442
- **FAIL** `layer1.5/neural_collapse.etf_deviation` values=[0.509, 0.694] rel_spread=0.306
- **FAIL** `layer1.5/class_centers.nearest_center_acc_test` values=[0.328, 0.213] rel_spread=0.426
- **FAIL** `layer1.8/twonn_id.id` values=[10.073, 8.656] rel_spread=0.151
- **FAIL** `layer1.8/pca_spectrum.participation_ratio` values=[5.491, 1.569] rel_spread=1.111
- **FAIL** `layer1.8/pca_spectrum.dim95` values=[9.0, 4.0] rel_spread=0.769
- **FAIL** `layer1.8/class_centers.sep_ratio` values=[0.318, 0.158] rel_spread=0.671
- **FAIL** `layer1.8/neural_collapse.nc1` values=[56.519, 359.437] rel_spread=1.456
- **FAIL** `layer1.8/neural_collapse.etf_deviation` values=[0.447, 0.642] rel_spread=0.358
- **FAIL** `layer1.8/hubness.k_occurrence_skew` values=[0.856, 0.669] rel_spread=0.245
- **FAIL** `layer1.8/class_centers.nearest_center_acc_test` values=[0.371, 0.198] rel_spread=0.606
- **FAIL** `layer2.0/twonn_id.id` values=[11.232, 13.715] rel_spread=0.199
- **FAIL** `layer2.0/pca_spectrum.participation_ratio` values=[6.241, 2.842] rel_spread=0.748
- **FAIL** `layer2.0/class_centers.sep_ratio` values=[0.339, 0.163] rel_spread=0.699
- **FAIL** `layer2.0/neural_collapse.nc1` values=[24.111, 113.929] rel_spread=1.301
- **FAIL** `layer2.0/neural_collapse.etf_deviation` values=[0.456, 0.548] rel_spread=0.183
- **FAIL** `layer2.0/hubness.k_occurrence_skew` values=[0.91, 1.392] rel_spread=0.418
- **FAIL** `layer2.0/class_centers.nearest_center_acc_test` values=[0.404, 0.227] rel_spread=0.558
- **FAIL** `layer2.5/pca_spectrum.participation_ratio` values=[8.682, 1.463] rel_spread=1.423
- **FAIL** `layer2.5/pca_spectrum.dim95` values=[16.0, 8.0] rel_spread=0.667
- **FAIL** `layer2.5/class_centers.sep_ratio` values=[0.394, 0.135] rel_spread=0.979
- **FAIL** `layer2.5/neural_collapse.nc1` values=[10.911, 113.479] rel_spread=1.649
- **FAIL** `layer2.5/neural_collapse.etf_deviation` values=[0.39, 0.601] rel_spread=0.425
- **FAIL** `layer2.5/hubness.k_occurrence_skew` values=[1.286, 1.713] rel_spread=0.285
- **FAIL** `layer2.5/class_centers.nearest_center_acc_test` values=[0.5, 0.182] rel_spread=0.931
- **FAIL** `layer2.8/pca_spectrum.participation_ratio` values=[9.872, 1.505] rel_spread=1.471
- **FAIL** `layer2.8/pca_spectrum.dim95` values=[18.0, 9.0] rel_spread=0.667
- **FAIL** `layer2.8/class_centers.sep_ratio` values=[0.437, 0.14] rel_spread=1.030
- **FAIL** `layer2.8/neural_collapse.nc1` values=[6.224, 115.324] rel_spread=1.795
- **FAIL** `layer2.8/neural_collapse.etf_deviation` values=[0.352, 0.597] rel_spread=0.517
- **FAIL** `layer2.8/hubness.k_occurrence_skew` values=[1.478, 1.759] rel_spread=0.173
- **FAIL** `layer2.8/class_centers.nearest_center_acc_test` values=[0.558, 0.184] rel_spread=1.009
- **FAIL** `layer3.0/twonn_id.id` values=[15.572, 28.317] rel_spread=0.581
- **FAIL** `layer3.0/pca_spectrum.participation_ratio` values=[10.033, 2.89] rel_spread=1.105
- **FAIL** `layer3.0/pca_spectrum.dim95` values=[25.0, 41.0] rel_spread=0.485
- **FAIL** `layer3.0/class_centers.sep_ratio` values=[0.544, 0.126] rel_spread=1.246
- **FAIL** `layer3.0/neural_collapse.nc1` values=[3.502, 88.409] rel_spread=1.848
- **FAIL** `layer3.0/neural_collapse.etf_deviation` values=[0.326, 0.597] rel_spread=0.586
- **FAIL** `layer3.0/hubness.k_occurrence_skew` values=[1.404, 4.156] rel_spread=0.990
- **FAIL** `layer3.0/class_centers.nearest_center_acc_test` values=[0.623, 0.184] rel_spread=1.087
- **FAIL** `layer3.5/twonn_id.id` values=[19.394, 28.22] rel_spread=0.371
- **FAIL** `layer3.5/pca_spectrum.participation_ratio` values=[15.827, 1.975] rel_spread=1.556
- **FAIL** `layer3.5/pca_spectrum.dim95` values=[44.0, 33.0] rel_spread=0.286
- **FAIL** `layer3.5/class_centers.sep_ratio` values=[0.894, 0.094] rel_spread=1.621
- **FAIL** `layer3.5/neural_collapse.nc1` values=[0.671, 127.823] rel_spread=1.979
- **FAIL** `layer3.5/neural_collapse.etf_deviation` values=[0.21, 0.706] rel_spread=1.084
- **FAIL** `layer3.5/hubness.k_occurrence_skew` values=[1.879, 3.06] rel_spread=0.478
- **FAIL** `layer3.5/class_centers.nearest_center_acc_test` values=[0.837, 0.16] rel_spread=1.357
- **FAIL** `penult/twonn_id.id` values=[10.669, 28.28] rel_spread=0.904
- **FAIL** `penult/pca_spectrum.participation_ratio` values=[9.023, 1.695] rel_spread=1.367
- **FAIL** `penult/pca_spectrum.dim95` values=[9.0, 28.0] rel_spread=1.027
- **FAIL** `penult/class_centers.sep_ratio` values=[5.332, 0.083] rel_spread=1.939
- **FAIL** `penult/neural_collapse.nc1` values=[0.049, 151.374] rel_spread=1.999
- **FAIL** `penult/neural_collapse.etf_deviation` values=[0.054, 0.793] rel_spread=1.747
- **FAIL** `penult/hubness.k_occurrence_skew` values=[0.94, 3.143] rel_spread=1.079
- **FAIL** `penult/class_centers.nearest_center_acc_test` values=[0.944, 0.157] rel_spread=1.431
- PASS: 15 items

## id_profile_stability
1 items, 1 FAIL
- **FAIL** `id_profile[0,1]` spearman=0.842 peak=layer3.5/layer3.0

## adjacency_stability
20 items, 10 FAIL
- **FAIL** `layer2.5/adjacency_spearman[0,1]` rho=0.550
- **FAIL** `layer2.5/merge_kendall[0,1]` tau=0.088
- **FAIL** `layer2.8/adjacency_spearman[0,1]` rho=0.558
- **FAIL** `layer2.8/merge_kendall[0,1]` tau=0.088
- **FAIL** `layer3.0/adjacency_spearman[0,1]` rho=0.573
- **FAIL** `layer3.0/merge_kendall[0,1]` tau=0.046
- **FAIL** `layer3.5/adjacency_spearman[0,1]` rho=0.428
- **FAIL** `layer3.5/merge_kendall[0,1]` tau=-0.061
- **FAIL** `penult/adjacency_spearman[0,1]` rho=0.344
- **FAIL** `penult/merge_kendall[0,1]` tau=-0.073
- PASS: 10 items

## decodability_stability
18 items, 17 FAIL
- **FAIL** `decod/contrast_rms[0,1]` profile_spearman=0.62 mean_abs_delta=0.081
- **FAIL** `decod/highfreq_ratio[0,1]` profile_spearman=-0.36 mean_abs_delta=0.325
- **FAIL** `decod/spectral_slope[0,1]` profile_spearman=-0.30 mean_abs_delta=0.404
- **FAIL** `decod/spectral_anisotropy[0,1]` profile_spearman=-0.22 mean_abs_delta=0.176
- **FAIL** `decod/noise_sigma[0,1]` profile_spearman=0.64 mean_abs_delta=0.254
- **FAIL** `decod/saturation_mean[0,1]` profile_spearman=0.85 mean_abs_delta=0.219
- **FAIL** `decod/hue_cos[0,1]` profile_spearman=0.89 mean_abs_delta=0.226
- **FAIL** `decod/hue_sin[0,1]` profile_spearman=0.98 mean_abs_delta=0.351
- **FAIL** `decod/colorfulness[0,1]` profile_spearman=0.58 mean_abs_delta=0.290
- **FAIL** `decod/edge_density[0,1]` profile_spearman=0.31 mean_abs_delta=0.154
- **FAIL** `decod/orientation_entropy[0,1]` profile_spearman=-0.66 mean_abs_delta=0.292
- **FAIL** `decod/blockiness[0,1]` profile_spearman=-0.71 mean_abs_delta=0.051
- **FAIL** `decod/class[0,1]` profile_spearman=-0.95 mean_abs_delta=0.342
- **FAIL** `decod/coarse_animal_vehicle[0,1]` profile_spearman=-0.99 mean_abs_delta=0.174
- **FAIL** `decod/corruption_family[0,1]` profile_spearman=-0.20 mean_abs_delta=0.203
- **FAIL** `decod/corruption_type[0,1]` profile_spearman=-0.14 mean_abs_delta=0.294
- **FAIL** `decod/severity[0,1]` profile_spearman=-0.21 mean_abs_delta=0.145
- PASS: 1 items

## commit_agreement
18 items, 10 FAIL
- **FAIL** `commit/highfreq_ratio` ['layer2.0', 'stem']
- **FAIL** `commit/spectral_slope` ['layer2.0', 'stem']
- **FAIL** `commit/spectral_anisotropy` ['layer3.0', 'stem']
- **FAIL** `commit/orientation_entropy` ['layer2.0', 'stem']
- **FAIL** `commit/blockiness` ['layer3.0', 'layer1.5']
- **FAIL** `commit/class` ['layer3.5', 'stem']
- **FAIL** `commit/coarse_animal_vehicle` ['layer3.5', 'stem']
- **FAIL** `commit/corruption_family` ['layer2.0', 'layer1.5']
- **FAIL** `commit/corruption_type` ['layer2.0', 'layer1.0']
- **FAIL** `commit/severity` ['layer3.0', 'stem']
- PASS: 8 items

## panel_agreement
2 items, 1 FAIL
- **FAIL** `penult/cka_test[0,1]` cka_test=0.030 min=0.8
- **INFO** `penult/agreement_info[0,1]` {"panel_cka": 0.107, "relrep_row_corr_mean": 0.167, "relrep_argmax_agree": 0.141, "relrep_argmax_agree_test": 0.157, "relrep_argmax_chance_test": 0.101, "relrep_offmax_corr_test": 0.129, "error_consistency_test": 0.002, "cka_ood_c100": 0.025, "relrep_argmax_agree_ood_c100": 0.11, "relrep_argmax_chance_ood_c100": 0.085, "relrep_offmax_corr_ood_c100": 0.158, "landmark_procrustes_disparity": 0.968}
