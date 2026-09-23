# CRITIC  verdict: **REPLICATES**
runs: ['results/margin_b1_vitb16', 'results/margin_b1_vitb16_swap']
counts: {'PASS': 75, 'FAIL': 0, 'WARN': 5, 'INFO': 0}

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
- **WARN** `probes:margin_b1_vitb16_swap` no probes

## scalar_stability
70 items, 0 FAIL
- PASS: 70 items

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
