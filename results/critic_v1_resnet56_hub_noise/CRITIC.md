# CRITIC  verdict: **PARTIAL**
runs: ['results/atlas_v1_resnet56_s0hub_st3', 'results/atlas_v1_resnet56_s0hub_ref1']
counts: {'PASS': 140, 'FAIL': 4, 'WARN': 1, 'INFO': 1}

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
80 items, 3 FAIL
- **FAIL** `stem/neural_collapse.nc1` values=[494.389, 243.963] rel_spread=0.678
- **FAIL** `layer1.0/neural_collapse.nc1` values=[171.954, 435.7] rel_spread=0.868
- **FAIL** `layer1.5/neural_collapse.nc1` values=[53.611, 74.872] rel_spread=0.331
- PASS: 77 items

## id_profile_stability
1 items, 0 FAIL
- PASS: 1 items

## adjacency_stability
20 items, 0 FAIL
- PASS: 20 items

## decodability_stability
18 items, 0 FAIL
- PASS: 18 items

## commit_agreement
18 items, 1 FAIL
- **FAIL** `commit/blockiness` ['layer3.0', 'layer2.0']
- PASS: 17 items

## panel_agreement
2 items, 0 FAIL
- **INFO** `penult/agreement_info[0,1]` {"panel_cka": 0.108, "relrep_row_corr_mean": -0.057, "relrep_argmax_agree": 0.047, "relrep_argmax_agree_test": 1.0, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.999, "error_consistency_test": 1.0, "cka_ood_c100": 1.0, "relrep_argmax_agree_ood_c100": 0.996, "relrep_argmax_chance_ood_c100": 0.122, "relrep_offmax_corr_ood_c100": 1.0, "landmark_procrustes_disparity": 0.0}
- PASS: 1 items
