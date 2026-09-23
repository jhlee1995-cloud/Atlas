# CRITIC  verdict: **PARTIAL**
runs: ['results/margin_b1_vitb16', 'results/margin_b1_deitb']
counts: {'PASS': 66, 'FAIL': 9, 'WARN': 5, 'INFO': 0}

## synthetic_refusal
1 items, 0 FAIL
- PASS: 1 items

## norm_consistency
1 items, 0 FAIL
- PASS: 1 items

## alias_layers
1 items, 0 FAIL
- PASS: 1 items

## holdout_hygiene
2 items, 0 FAIL
- PASS: 2 items

## probe_hygiene
2 items, 0 FAIL
- **WARN** `probes:margin_b1_vitb16` no probes
- **WARN** `probes:margin_b1_deitb` no probes

## scalar_stability
70 items, 9 FAIL
- **FAIL** `block.8/margin_typeb.median_margin_ratio_typeb` values=[0.586, 0.7] rel_spread=0.177
- **FAIL** `block.9/margin_typeb.median_margin_ratio_typeb` values=[0.47, 0.558] rel_spread=0.171
- **FAIL** `block.10/margin_typeb.median_margin_ratio_typeb` values=[0.435, 0.36] rel_spread=0.188
- **FAIL** `block.11/margin_typeb.median_margin_ratio_typeb` values=[0.431, 0.371] rel_spread=0.150
- **FAIL** `penult_mean/margin_typeb.auc_margin_typeb` values=[0.561, 0.736] abs_spread=0.175 abs_tol=0.05
- **FAIL** `penult_mean/margin_typeb.auc_dist_typeb` values=[0.505, 0.65] abs_spread=0.145 abs_tol=0.05
- **FAIL** `penult_mean/margin_typeb.auc_margin_wrong` values=[0.615, 0.848] abs_spread=0.233 abs_tol=0.05
- **FAIL** `penult_mean/margin_typeb.median_margin_ratio_typeb` values=[0.785, 0.395] rel_spread=0.659
- **FAIL** `penult/margin_typeb.median_margin_ratio_typeb` values=[0.504, 0.432] rel_spread=0.155
- PASS: 61 items

## id_profile_stability
0 items, 0 FAIL

## adjacency_stability
0 items, 0 FAIL

## decodability_stability
1 items, 0 FAIL
- **WARN** `decodability` linear_probes not present

## commit_agreement
1 items, 0 FAIL
- **WARN** `commit_layer` commit_layer missing in some run

## panel_agreement
1 items, 0 FAIL
- **WARN** `panel[0,1]` no compare_vs_* deformation.json
