# CRITIC  verdict: **PARTIAL**
runs: ['results/margin_v1_resnet20_s0hub', 'results/margin_v1_resnet20_s1', 'results/margin_v1_resnet20_s2']
counts: {'PASS': 50, 'FAIL': 5, 'WARN': 9, 'INFO': 0}

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
3 items, 0 FAIL
- PASS: 3 items

## probe_hygiene
3 items, 0 FAIL
- **WARN** `probes:margin_v1_resnet20_s0hub` no probes
- **WARN** `probes:margin_v1_resnet20_s1` no probes
- **WARN** `probes:margin_v1_resnet20_s2` no probes

## scalar_stability
50 items, 5 FAIL
- **FAIL** `layer1.0/margin_typeb.median_margin_ratio_typeb` values=[0.806, 1.065, 0.963] rel_spread=0.274
- **FAIL** `layer1.2/margin_typeb.median_margin_ratio_typeb` values=[0.95, 0.946, 0.73] rel_spread=0.251
- **FAIL** `layer2.2/margin_typeb.median_margin_ratio_typeb` values=[0.635, 0.842, 0.701] rel_spread=0.286
- **FAIL** `layer3.0/margin_typeb.auc_margin_typeb` values=[0.659, 0.634, 0.684] abs_spread=0.050 abs_tol=0.05
- **FAIL** `layer3.0/margin_typeb.median_margin_ratio_typeb` values=[0.495, 0.638, 0.452] rel_spread=0.351
- PASS: 45 items

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
3 items, 0 FAIL
- **WARN** `panel[0,1]` no compare_vs_* deformation.json
- **WARN** `panel[0,2]` no compare_vs_* deformation.json
- **WARN** `panel[1,2]` no compare_vs_* deformation.json
