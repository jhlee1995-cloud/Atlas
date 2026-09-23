# CRITIC  verdict: **PARTIAL**
runs: ['results/margin_v1_resnet56_s1', 'results/margin_v1_resnet56_s2']
counts: {'PASS': 52, 'FAIL': 2, 'WARN': 6, 'INFO': 0}

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
- **WARN** `probes:margin_v1_resnet56_s1` no probes
- **WARN** `probes:margin_v1_resnet56_s2` no probes

## scalar_stability
50 items, 2 FAIL
- **FAIL** `layer3.0/margin_typeb.auc_dist_typeb` values=[0.489, 0.549] abs_spread=0.059 abs_tol=0.05
- **FAIL** `layer3.5/margin_typeb.auc_dist_typeb` values=[0.541, 0.602] abs_spread=0.061 abs_tol=0.05
- PASS: 48 items

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
