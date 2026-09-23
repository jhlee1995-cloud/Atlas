# CRITIC  verdict: **PARTIAL**
runs: ['results/margin_v1_resnet20_s1', 'results/margin_v1_resnet20_s2', 'results/margin_v1_resnet20_s3', 'results/margin_v1_resnet20_s4']
counts: {'PASS': 50, 'FAIL': 6, 'WARN': 13, 'INFO': 0}

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
4 items, 0 FAIL
- PASS: 4 items

## probe_hygiene
4 items, 0 FAIL
- **WARN** `probes:margin_v1_resnet20_s1` no probes
- **WARN** `probes:margin_v1_resnet20_s2` no probes
- **WARN** `probes:margin_v1_resnet20_s3` no probes
- **WARN** `probes:margin_v1_resnet20_s4` no probes

## scalar_stability
50 items, 6 FAIL
- **FAIL** `layer1.1/margin_typeb.auc_margin_typeb` values=[0.514, 0.538, 0.492, 0.547] abs_spread=0.055 abs_tol=0.05
- **FAIL** `layer1.1/margin_typeb.median_margin_ratio_typeb` values=[0.926, 0.853, 1.024, 0.873] rel_spread=0.186
- **FAIL** `layer1.2/margin_typeb.median_margin_ratio_typeb` values=[0.946, 0.73, 0.965, 0.844] rel_spread=0.269
- **FAIL** `layer2.2/margin_typeb.median_margin_ratio_typeb` values=[0.842, 0.701, 0.718, 0.783] rel_spread=0.186
- **FAIL** `layer3.0/margin_typeb.auc_margin_typeb` values=[0.634, 0.684, 0.641, 0.615] abs_spread=0.069 abs_tol=0.05
- **FAIL** `layer3.0/margin_typeb.median_margin_ratio_typeb` values=[0.638, 0.452, 0.581, 0.594] rel_spread=0.327
- PASS: 44 items

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
6 items, 0 FAIL
- **WARN** `panel[0,1]` no compare_vs_* deformation.json
- **WARN** `panel[0,2]` no compare_vs_* deformation.json
- **WARN** `panel[0,3]` no compare_vs_* deformation.json
- **WARN** `panel[1,2]` no compare_vs_* deformation.json
- **WARN** `panel[1,3]` no compare_vs_* deformation.json
- **WARN** `panel[2,3]` no compare_vs_* deformation.json
