# CRITIC  verdict: **PARTIAL**
runs: ['results/atlas_v1_resnet20_s1', 'results/atlas_v1_resnet20_s1_ref1']
counts: {'PASS': 140, 'FAIL': 4, 'WARN': 1, 'INFO': 1}

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
80 items, 4 FAIL
- **FAIL** `stem/neural_collapse.nc1` values=[261.068, 191.397] rel_spread=0.308
- **FAIL** `layer1.0/neural_collapse.nc1` values=[164.627, 137.172] rel_spread=0.182
- **FAIL** `layer1.1/neural_collapse.nc1` values=[74.307, 58.398] rel_spread=0.240
- **FAIL** `layer1.2/neural_collapse.nc1` values=[49.246, 65.828] rel_spread=0.288
- PASS: 76 items

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
18 items, 0 FAIL
- PASS: 18 items

## panel_agreement
2 items, 0 FAIL
- **INFO** `penult/agreement_info[0,1]` {"panel_cka": 0.1, "relrep_row_corr_mean": -0.026, "relrep_argmax_agree": 0.047, "relrep_argmax_agree_test": 0.999, "relrep_argmax_chance_test": 0.1, "relrep_offmax_corr_test": 0.999, "error_consistency_test": 1.0, "cka_ood_c100": 1.0, "relrep_argmax_agree_ood_c100": 0.995, "relrep_argmax_chance_ood_c100": 0.127, "relrep_offmax_corr_ood_c100": 1.0, "landmark_procrustes_disparity": 0.0}
- PASS: 1 items
