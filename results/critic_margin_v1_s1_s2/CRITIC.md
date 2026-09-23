# CRITIC  verdict: **PARTIAL**
runs: ['results/margin_v1_resnet20_s1', 'results/margin_v1_resnet20_s2']
counts: {'PASS': 50, 'FAIL': 4, 'WARN': 6, 'INFO': 0}

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
- **WARN** `probes:margin_v1_resnet20_s1` no probes
- **WARN** `probes:margin_v1_resnet20_s2` no probes

## scalar_stability
50 items, 4 FAIL
- **FAIL** `layer1.2/margin_typeb.median_margin_ratio_typeb` values=[0.946, 0.73] rel_spread=0.258
- **FAIL** `layer2.2/margin_typeb.median_margin_ratio_typeb` values=[0.842, 0.701] rel_spread=0.183
- **FAIL** `layer3.0/margin_typeb.auc_margin_typeb` values=[0.634, 0.684] abs_spread=0.050 abs_tol=0.05
- **FAIL** `layer3.0/margin_typeb.median_margin_ratio_typeb` values=[0.638, 0.452] rel_spread=0.340
- PASS: 46 items

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
