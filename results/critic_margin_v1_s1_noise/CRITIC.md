# CRITIC  verdict: **REPLICATES**
runs: ['results/margin_v1_resnet20_s1', 'results/margin_v1_resnet20_s1_ref1']
counts: {'PASS': 54, 'FAIL': 0, 'WARN': 6, 'INFO': 0}

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
- **WARN** `probes:margin_v1_resnet20_s1_ref1` no probes

## scalar_stability
50 items, 0 FAIL
- PASS: 50 items

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
