# CRITIC  verdict: **PARTIAL**
runs: ['results/margin_b1_resnet50', 'results/margin_b1_resnet50_swap']
counts: {'PASS': 56, 'FAIL': 33, 'WARN': 6, 'INFO': 0}

## synthetic_refusal
1 items, 0 FAIL
- PASS: 1 items

## norm_consistency
1 items, 0 FAIL
- PASS: 1 items

## alias_layers
1 items, 0 FAIL
- **WARN** `alias:layer4.2` identical to penult in every run; counted once as penult

## holdout_hygiene
2 items, 0 FAIL
- PASS: 2 items

## probe_hygiene
2 items, 0 FAIL
- **WARN** `probes:margin_b1_resnet50` no probes
- **WARN** `probes:margin_b1_resnet50_swap` no probes

## scalar_stability
85 items, 33 FAIL
- **FAIL** `stem/margin_typeb.auc_margin_typeb` values=[0.587, 0.4] abs_spread=0.187 abs_tol=0.05
- **FAIL** `stem/margin_typeb.auc_dist_typeb` values=[0.501, 0.571] abs_spread=0.070 abs_tol=0.05
- **FAIL** `stem/margin_typeb.median_margin_ratio_typeb` values=[0.459, 1.314] rel_spread=0.964
- **FAIL** `layer1.0/margin_typeb.auc_dist_typeb` values=[0.459, 0.636] abs_spread=0.177 abs_tol=0.05
- **FAIL** `layer1.0/margin_typeb.median_margin_ratio_typeb` values=[0.926, 0.787] rel_spread=0.162
- **FAIL** `layer1.1/margin_typeb.auc_dist_typeb` values=[0.472, 0.636] abs_spread=0.164 abs_tol=0.05
- **FAIL** `layer1.1/margin_typeb.median_margin_ratio_typeb` values=[0.805, 0.513] rel_spread=0.442
- **FAIL** `layer1.2/margin_typeb.auc_dist_typeb` values=[0.488, 0.635] abs_spread=0.147 abs_tol=0.05
- **FAIL** `layer1.2/margin_typeb.median_margin_ratio_typeb` values=[0.907, 0.773] rel_spread=0.160
- **FAIL** `layer2.0/margin_typeb.auc_margin_typeb` values=[0.565, 0.454] abs_spread=0.111 abs_tol=0.05
- **FAIL** `layer2.0/margin_typeb.auc_dist_typeb` values=[0.528, 0.617] abs_spread=0.089 abs_tol=0.05
- **FAIL** `layer2.0/margin_typeb.median_margin_ratio_typeb` values=[0.896, 1.12] rel_spread=0.222
- **FAIL** `layer2.1/margin_typeb.auc_dist_typeb` values=[0.549, 0.631] abs_spread=0.082 abs_tol=0.05
- **FAIL** `layer2.1/margin_typeb.median_margin_ratio_typeb` values=[1.155, 0.728] rel_spread=0.453
- **FAIL** `layer2.2/margin_typeb.auc_dist_typeb` values=[0.548, 0.636] abs_spread=0.089 abs_tol=0.05
- **FAIL** `layer2.2/margin_typeb.median_margin_ratio_typeb` values=[1.373, 0.885] rel_spread=0.432
- **FAIL** `layer2.3/margin_typeb.auc_dist_typeb` values=[0.55, 0.639] abs_spread=0.089 abs_tol=0.05
- **FAIL** `layer3.0/margin_typeb.auc_dist_typeb` values=[0.569, 0.672] abs_spread=0.102 abs_tol=0.05
- **FAIL** `layer3.1/margin_typeb.auc_margin_typeb` values=[0.626, 0.539] abs_spread=0.087 abs_tol=0.05
- **FAIL** `layer3.1/margin_typeb.auc_dist_typeb` values=[0.583, 0.664] abs_spread=0.081 abs_tol=0.05
- **FAIL** `layer3.1/margin_typeb.median_margin_ratio_typeb` values=[0.659, 0.877] rel_spread=0.284
- **FAIL** `layer3.2/margin_typeb.auc_dist_typeb` values=[0.595, 0.674] abs_spread=0.080 abs_tol=0.05
- **FAIL** `layer3.2/margin_typeb.median_margin_ratio_typeb` values=[0.475, 0.82] rel_spread=0.533
- **FAIL** `layer3.3/margin_typeb.auc_margin_typeb` values=[0.518, 0.574] abs_spread=0.056 abs_tol=0.05
- **FAIL** `layer3.3/margin_typeb.auc_dist_typeb` values=[0.603, 0.679] abs_spread=0.076 abs_tol=0.05
- **FAIL** `layer3.3/margin_typeb.median_margin_ratio_typeb` values=[1.009, 0.67] rel_spread=0.403
- **FAIL** `layer3.4/margin_typeb.auc_margin_typeb` values=[0.471, 0.558] abs_spread=0.088 abs_tol=0.05
- **FAIL** `layer3.4/margin_typeb.median_margin_ratio_typeb` values=[1.221, 0.802] rel_spread=0.414
- **FAIL** `layer4.0/margin_typeb.median_margin_ratio_typeb` values=[1.131, 0.89] rel_spread=0.239
- **FAIL** `layer4.1/margin_typeb.auc_dist_typeb` values=[0.795, 0.745] abs_spread=0.051 abs_tol=0.05
- **FAIL** `layer4.1/margin_typeb.median_margin_ratio_typeb` values=[0.665, 0.867] rel_spread=0.264
- **FAIL** `penult/margin_typeb.auc_dist_typeb` values=[0.814, 0.735] abs_spread=0.079 abs_tol=0.05
- **FAIL** `penult/margin_typeb.median_margin_ratio_typeb` values=[0.625, 0.769] rel_spread=0.208
- PASS: 52 items

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
