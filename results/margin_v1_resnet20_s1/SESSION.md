# SESSION: A3 — margin_typeb (ATLAS_STATUS row 9)

This is the Evaluator pass on the pulled A3 results. The dumps stay on the RunPod volume.

- **Pre-registration:** `experiments/queue/margin_v1_resnet20_s1.yaml` notes (M1-M5, N, E1-E5 and the row-9 decision table). Design and archaeology are in `docs/plans/A3_MARGIN.md`, and the frozen definition is `atlas/invariants/margin.py`. All of it was committed and pushed as **P = `664bd25`** (2026-09-22 22:30:47 -0700 = 2026-09-23 05:30:47 UTC) before the run.
- **How it was checked:** two evaluator families recomputed every prediction from the result files with node. This pass then re-read every number the decision uses from the atlas.json and critic.json files.
- **What is judged:** only penult. layer3.2 is its alias, and its values are identical to penult.

**Decision: PROMOTE (replicated). Row 9 goes from ⬜ to ✅ s1, s2, s3, s4.**
- E1 holds, so the row states "margin ~ maxprob (not an independent detector)".
- The ten pre-registered items all PASS.
- The replicated quantity is a confidence proxy. At the frozen cut of 0.7, penult margin separates type-b from correct samples exactly as well as maxprob does.

Path shorthands. `pm` = `.per_layer.penult.margin_typeb`.

| shorthand | path |
|---|---|
| `m0` | `results/margin_v0_resnet20_cifar10/atlas.json` (v0, discovery, norm cifar_true, std 0.247) |
| `mh` | `results/margin_v1_resnet20_s0hub/atlas.json` (v1 hub, discovery) |
| `m1` .. `m4` | `results/margin_v1_resnet20_s{1,2,3,4}/atlas.json` (confirmation) |
| `mr1` | `results/margin_v1_resnet20_s1_ref1/atlas.json` (twin: reference seed 1) |
| `mrand` | `results/margin_v1_resnet20_rand/atlas.json` (random-init null) |
| `m56` | `results/margin_v1_resnet56_s0hub/atlas.json` (E5, exploratory) |
| `c12`, `c012`, `cn`, `c14` | `results/critic_margin_v1_{s1_s2,s0_s1_s2,s1_noise,s1_s4}/critic.json` |

## Run facts

**Provenance.**
- Every margin_* `provenance.json` has `.stage_b_git_commit` 664bd25, and every `manifest_used.yaml` has cut 0.7, the pre-registered cut list and min_n 10.
- The Stage B runs finished in this order (`.finished`, pod clock, UTC):

  | run | finished |
  |---|---|
  | v0 | 06:05:09 |
  | hub | 06:05:19 |
  | rand | 06:05:27 |
  | resnet56 | 06:05:37 |
  | s1 | 06:05:47 |
  | s2 | 06:05:56 |
  | s1_ref1 | 06:06:06 |
  | s3 | 06:06:16 |
  | s4 | 06:06:27 |

  All of these are after P. The confirmation rebuilds ran only after v0, which matches the gate. The pod log reports the M1 gate PASS (`[A3] M1 dir_auc_margin=0.8931519... raw_auc_margin=0.1068480...`, then `=== A3: M1 PASS -> margin_typeb on the confirmation dumps ...`).

**Dumps** (`.dump_meta`):

| dump | git_commit | created (UTC) | norm |
|---|---|---|---|
| v0 | aa07258 | 00:50:29 | cifar_true, std 0.247 |
| hub, s1, s2, s1_ref1, rand | 6242a17 | 02:10-02:16 | chenyaofo, std (0.2023, 0.1994, 0.201) |
| s3, s4, resnet56 | 664bd25 | 05:44:19 / 05:45:41 / 05:55:51 | chenyaofo |

**Code and first touch.**
- HEAD is e86c815 (the Stage 1b result commit, ahead of origin/main by 1). `git diff 664bd25 HEAD` outside `results/` touches ATLAS_STATUS.md (Stage 1b header and rows 1-8; row 9 unchanged) and docs/plans/STAGE1.md (+6); atlas/, experiments/, pod_atlas.sh and docs/plans/A3_MARGIN.md are unchanged.
- `git diff --quiet 664bd25 -- atlas experiments pod_atlas.sh docs/plans/A3_MARGIN.md` holds on the worktree.
- `git grep -i -E "maxprob|margin_typeb|auc_margin" 664bd25 -- results` finds 2 files, results/atlas_v1_resnet20_s3/RUN_REQUEST.md:36 and results/margin_v1_resnet20_s1/RUN_REQUEST.md:1. Both are text mentions only; no committed result held a margin, maxprob or type-b value for s1 or s2, so this was their first margin touch (STAGE1.md amendment 2 item 8).

**Accuracy on test[:5000]** (`pm.acc`):

| v0 | hub | s1 | s2 | s3 | s4 | rand | resnet56 |
|---|---|---|---|---|---|---|---|
| 0.9184 | 0.9256 | 0.9226 | 0.9250 | 0.9222 | 0.9286 | 0.1208 | 0.9446 |

**Critic counts** (`.counts`). The critic's verdict word is advisory (AGENT_LOOP.md:89).

| critic | PASS | FAIL | WARN | verdict word |
|---|---|---|---|---|
| s1_s2 | 50 | 4 | 6 | PARTIAL |
| s0_s1_s2 | 50 | 5 | 9 | PARTIAL |
| s1_noise | 54 | 0 | 6 | REPLICATES |
| s1_s4 | 50 | 6 | 13 | PARTIAL |

- All 15 FAILs are `scalar_stability` items on non-penult layers.
- Every WARN is an expected margin-only WARN: alias layer3.2, no probes, no linear_probes, no commit_layer, no compare_vs_*.
- All five penult margin items (`.checks.scalar_stability[45..49]`: auc_margin_typeb, auc_dist_typeb, auc_maxprob_typeb, auc_margin_wrong, median_margin_ratio_typeb) are PASS in all four critics.

**Internal consistency** (every run):
- `pm.legacy.n_cw` == `pm.n_typeb`.
- `pm.legacy.dir_auc_energy` == `pm.auc_energy_typeb`, because energy does not depend on the centers.
- `pm.legacy` raw + dir = 1.
- `median_margin_ratio_*` = median_* / median_margin_correct.
- The s1 `pm.sweep` row at cut 0.7 equals the headline keys.

## Predictions (predicted → observed → verdict)

**M1 (reproduction, the gate)**
- Scope: `m0` `pm.legacy`.
- Result: all six clauses hold.

  | clause | band (legacy) | observed |
  |---|---|---|
  | dir_auc_margin | [0.865, 0.925] (0.895) | 0.893152 |
  | raw_auc_margin (margin clause, core) | < 0.5 | 0.106848 |
  | dir_auc_cluster_subnet | [0.723, 0.803] (0.763) | 0.756094 |
  | dir_auc_energy | [0.717, 0.797] (0.757) | 0.760058 |
  | n_wrong | == 408 | 408 |
  | n_cw | [245, 300] | 260 (plan expected about 273) |

- Verdict: **PASS**. The gate is open.
- The legacy anchors are `docs/history/SESSION_RESULTS_VALLEYS_TYPEB_SCALE.md:55-59`.
- INFO, not pre-registered: applying the M1 bands to the v1 seeds.
  - Every band holds in hub, s1, s2, s3 and s4 except n_wrong == 408, which is v0's plumbing count by construction (372 / 387 / 375 / 389 / 357).
  - The literal reading is PARTIAL per seed; the intended reading is not applicable. So these are NOT_EVALUABLE.
  - Values nearest a band edge: s2 n_cw 247 and s1 energy 0.729151.

**M2 (primary)**
- Predicted: margin_minus_dist_typeb > 0 and p < 0.05 in each run, plus the v0 legacy clause.
- Observed (`pm.{auc_margin_typeb, auc_dist_typeb, margin_minus_dist_typeb, _se, _p}`):

  | run | auc_margin | auc_dist | margin − dist (se) | p |
  |---|---|---|---|---|
  | v0 | 0.894382 | 0.828187 | 0.066195 (0.005801) | 3.71e-30 |
  | hub | 0.884793 | 0.814116 | 0.070677 (0.006121) | 7.67e-31 |
  | s1 | 0.882476 | 0.836817 | 0.045659 (0.005053) | 1.63e-19 |
  | s2 | 0.884333 | 0.821665 | 0.062668 (0.006128) | 1.51e-24 |
  | s3 | 0.889701 | 0.827108 | 0.062593 (0.005885) | 2.01e-26 |
  | s4 | 0.899490 | 0.844821 | 0.054669 (0.006013) | 9.77e-20 |

- The v0 legacy clause holds: dir_auc_margin 0.893152 > dir_auc_cluster_subnet 0.756094.
- The twin gives 0.046389, p 6.33e-20.
- Verdict: **PASS** in every run.

**M3**
- Predicted: median_margin_typeb < median_margin_correct, ratio_typeb ≤ 0.5 and ratio_wrong ≤ 0.5.
- Observed (`pm.{median_margin_typeb, median_margin_correct, median_margin_ratio_typeb, median_margin_ratio_wrong}`):

  | run | median typeb | median correct | ratio typeb | ratio wrong |
  |---|---|---|---|---|
  | v0 | 1.198798 | 4.695836 | 0.255290 | 0.166015 |
  | hub | 1.336735 | 4.731824 | 0.282499 | 0.181482 |
  | s1 | 1.353659 | 4.838099 | 0.279792 | 0.183109 |
  | s2 | 1.343046 | 4.722663 | 0.284383 | 0.180369 |
  | s3 | 1.348330 | 4.728243 | 0.285165 | 0.170658 |
  | s4 | 1.259590 | 4.739730 | 0.265751 | 0.202123 |

- Verdict: **PASS**; all 18 clause checks hold.
- The ratio depends on the cut; see the cut sweep and Rule 8.

**M4 (penult auc_margin_typeb)**
- |s1 − s2| = 0.001857 ≤ 0.05. `c12` `.checks.scalar_stability[45]` is PASS (abs_spread 0.002, abs_tol 0.05).
- The range over s1..s4 is 0.017014 (0.882476 / 0.884333 / 0.889701 / 0.899490) ≤ 0.05. `c14` [45] is PASS (abs_spread 0.017).
- Twin: |0.8824758 − 0.8819986| = 0.000477 ≤ 0.01. `cn` [45] abs_spread is 0.000, and `cn` has 0 FAILs.
- Hub: 0.884793 − mean(s1, s2) 0.883404 = +0.001389, within 0.05. `c012` [45] is PASS.
- Verdict: **PASS**.
- Reading note:
  - `c12`, `c012` and `c14` report the verdict word PARTIAL, because of non-penult FAILs:
    - 11 × median_margin_ratio_typeb (rel 0.18-0.35, at layers with AUC about 0.5);
    - 4 × auc_margin_typeb (layer1.1 0.055, layer3.0 0.050007 / 0.050007 / 0.069).
  - A reading that requires REPLICATES would fail M4. I rejected it: M4 names the item penult/margin_typeb.auc_margin_typeb, the notes say "Only penult is judged", and AGENT_LOOP.md:89 makes the verdict word advisory.

**M5 (same weights, v1 hub vs v0)**
- Δ auc_margin_typeb = 0.884793 − 0.894382 = −0.009589, within ±0.03.
- Δ median_margin_ratio_typeb = 0.282499 − 0.255290 = +0.027209, within ±0.10.
- Recorded:
  - n_typeb 260 → 251;
  - typeb_frac_of_wrong 0.637255 → 0.674731;
  - n_wrong 408 → 372;
  - maxprob AUC 0.887735 → 0.884426.
- Verdict: **PASS**.

**N (null)**
- Observed: `mrand` `pm.auc_margin_wrong` 0.541790 (se 0.012871; the 95% CI 0.517-0.567 lies inside [0.40, 0.65]).
- n_typeb is 0, so every type-b, confmatched and legacy AUC is null, as designed.
- Verdict: **PASS**.

**E1 (exploratory; rule 8: margin may restate confidence)**
- Predicted: spearman ≥ 0.7 and |confmatched margin − maxprob| ≤ 0.05 in s1 and s2.
- Observed:
  - s1: `pm.spearman_margin_maxprob` 0.928639; `pm.auc_margin_confmatched` 0.905570 vs `pm.auc_maxprob_confmatched` 0.903857, a difference of +0.001712 (se about 0.0083).
  - s2: 0.920383; 0.906066 vs 0.905746, a difference of +0.000320 (se about 0.0076).
  - Context, not part of E1: hub 0.925 / −0.0013, s3 0.919 / −0.0018, s4 0.928 / −0.0005, v0 0.921 / +0.0050.
- Verdict: **PASS**. Row 9 must state "margin ~ maxprob (not an independent detector)".

**E2**
- Predicted: penult ≥ max(stem .. layer2.2) + 0.10.
- Observed:
  - Hub (the A3_MARGIN.md run table assigns E2 to s0hub): 0.884793 vs 0.605508 (layer2.2) + 0.10, a slack of +0.179.
  - Every other resnet20 run also passes, with slack v0 +0.212, s1 +0.226, s2 +0.203, s1_ref1 +0.232, s3 +0.227, s4 +0.247. The early maximum is always at layer2.2.
- Verdict: **PASS** under both readings.
- The signal arrives late: stem to layer1.2 is about 0.47-0.55, layer3.0 0.61-0.68, layer3.1 0.75-0.77, and penult 0.88-0.90.

**E4**
- Observed: hub `pm.auc_energy_typeb` 0.766001. Every run is above 0.5: v0 0.760058, s1 0.729151, s2 0.743754, s3 0.744445, s4 0.771084, resnet56 0.725699.
- Verdict: **PASS**. Type-b has the lower mean squared activation.

**E5 (scale, exploratory)**
- Evaluable, because M1 passed.
- Observed: `m56` `pm.auc_margin_typeb` 0.906687 ≥ 0.80. The resnet20 hub is 0.884793, so |Δ| = 0.021894 ≤ 0.05. penult = layer3.8 in resnet56. Both were built by Stage B at 664bd25 in this session.
- Verdict: **PASS**.
- INFO:
  - legacy 0.895 → 0.905.
  - In resnet56, margin − dist is −0.000765 (p 0.8296), so the M2 advantage is absent at depth 56.
  - Accuracy is 0.9446 vs 0.9256 with no accuracy control. "Improves with scale" is not claimable.

**Not measured, as pre-registered:** E3 (corrupt-split margins) and M1x (the exact 10k legacy reproduction).

**Tally:** 10 pre-registered items (M1, M2, M3, M4, M5, N, E1, E2, E4, E5): 10 PASS, 0 PARTIAL, 0 FAIL, 0 NOT_EVALUABLE.

## Decision (the pre-registered row-9 table, applied in order)

1. **"No verdict" if the M1 margin clause fails:** not taken. dir_auc_margin 0.893152 is in band, and raw_auc_margin 0.106848 is below 0.5.
2. **PROMOTE (replicated):** every condition holds.
   - The M1 margin clause passes.
   - M2 and M3 hold in every confirmation seed that ran (s1, s2, s3, s4); they also hold in v0 and the hub.
   - M4 holds.
3. **REJECT (M2 fails in s1 and s2):** not taken; M2 passes in both.
4. **YELLOW:** not reached.
5. **E1 holds**, so the row states "margin ~ maxprob (not an independent detector)".

Alternative readings:

| reading | result |
|---|---|
| A (literal, adopted): the critic item level decides | PROMOTE |
| B: every critic verdict word must be REPLICATES | YELLOW, failing clause M4; rejected (see M4) |
| C: M2/M3 must also hold in v0 and hub | PROMOTE |

**Promotion rule (AGENT_LOOP.md):**
1. Real source: `.checks.synthetic_refusal[0]` is PASS in `c12`, `c012`, `cn` and `c14`.
2. Replicated within tolerances: penult `.checks.scalar_stability[45..49]` is PASS in all four critics (scalar_abs_tol 0.05 for the AUCs, rel 0.15 for the ratio).
3. Holdout declared: `.checks.holdout_hygiene[*]` is PASS for every run.
4. Prediction written before the run: P precedes every `.finished` time, `.stage_b_git_commit` = P, and the code is unchanged since P (see Run facts).

The owner's commit of this SESSION.md and the row change is the human approval.

## Rule 8 (too-good, by-construction or edge numbers)

**Margin restates confidence (E1 is the artifact hypothesis, and it holds).**
- Spearman(margin, maxprob) is 0.919-0.929 in every resnet20 run.
- Margin minus maxprob at cut 0.7, from `pm.auc_*_typeb` and `pm.auc_*_confmatched`:

  | run | type-b vs correct | confidence-matched |
  |---|---|---|
  | v0 | +0.0066 | +0.0050 |
  | hub | +0.0004 | −0.0013 |
  | s1 | +0.0021 | +0.0017 |
  | s2 | +0.0011 | +0.0003 |
  | s3 | −0.0001 | −0.0018 |
  | s4 | +0.0007 | −0.0005 |

  The per-AUC SE is about 0.008. The sign of the gap flips between seeds. 0.008 is the marginal per-AUC SE, not the SE of the paired difference; the paired margin-vs-maxprob SE is not stored and is likely several times smaller (the paired margin-vs-dist SE is 0.005-0.006 at |rho| about 0.7), so the paired z of the v1 gaps is unknown and v0 +0.0066 may be about 2 paired SE. The sign flips across seeds carry the E1 verdict.
- Across s1..s4, margin AUC moves with maxprob AUC: range 0.017 vs 0.018 (`c14` [45], [47]). The M4 stability is therefore the stability of a confidence proxy.
- M2 reads as "a confidence proxy beats raw nearest-center distance". maxprob beats distance by about as much: 0.044 / 0.062 / 0.063 / 0.054 in s1-s4.
- Untested mechanism hypothesis: penult feeds the linear classifier directly, so the d2 − d1 gap between class means tracks the top-2 logit gap.

**M2 is not created by the 0.7 selection.**
- At cut 0 (all wrong vs correct), `pm.auc_margin_wrong` − `pm.auc_dist_wrong` is 0.037 / 0.050 / 0.049 / 0.045 in s1-s4.
- In the s1 sweep, the gap grows with the cut: 0.037 → 0.092 at cut 0.99.

**M3 depends on the cut (rule 5).** The ratio median_margin_typeb / median_margin_correct by cut (`pm.sweep[*]`):

| cut | v0 | hub | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|---|
| 0.7 | 0.255 | 0.282 | 0.280 | 0.284 | 0.285 | 0.266 |
| 0.95 | 0.465 | **0.513** | 0.472 | 0.488 | 0.491 | 0.499 |
| 0.99 | 0.640 | 0.634 | 0.620 | 0.627 | 0.633 | 0.593 |

- Type-b has a maxprob floor, and margin ~ maxprob, so the ratio mostly measures that floor.
- The row states "at cut 0.7".
- The original wording "type-b sits on ridges" is dropped: it is an interpretation that no measurement separated from confidence.
- Changing the cut is discovery-only and needs a new pre-registration.

**The tolerances are looser than the observed noise.**
- A3_MARGIN.md §3 assumed an SE of about 0.013. The observed DeLong `pm.se_margin_typeb` is 0.0072-0.0084, so 0.05 is about 4.4 SD for an independent pair.
- The observed spreads (0.0019 for the pair, 0.017 for the range) would also pass a 2-SD threshold of about 0.023. The loose tolerance did not decide M4.
- No tolerance was changed.

**The twin measures reference-draw noise only.** maxprob, energy and the legacy block are identical by construction in s1 and s1_ref1 (0.880345, 0.729151, 0.883021).

**The null has no type-b.**
- `mrand` n_typeb is 0, so M2-M4 have no null counterpart.
- N is 3.2 SE above 0.5, and `mrand` `pm.auc_maxprob_wrong` is 0.5637, so random-init features carry a weak signal.

**M1's closeness to legacy is expected.** The difference −0.0018 is about 0.2 of the planned 0.009 half-vs-full SD. raw 0.107 confirms that the direction-free transform hides no flipped sign.

**Borderline non-penult critic FAILs.** The layer3.0 auc_margin_typeb FAILs in `c12` and `c012` are 0.050007 (0.684320 − 0.634313). They are not judged.

**resnet56 is not claimable.**
- In `m56`, margin − maxprob is +0.0195 for type-b and +0.0196 confidence-matched, and it rises to +0.054 / +0.057 at cut 0.99.
- margin ≈ dist there.
- se_maxprob is 0.012163 vs se_margin 0.007902, while in resnet20 the two agree within 0.00031 (s4). That hints at a different maxprob distribution, possibly saturation ties near 1.0; this is untested because the dumps store no logits.
- It is one instance, with no paired test and no accuracy control.

## Cut sweep, s1 (rule 5; INFO; `m1` `pm.sweep[0..7]`)

| cut | n_typeb | ratio | auc margin | auc dist | auc maxprob | confmatched margin / maxprob |
|---|---|---|---|---|---|---|
| 0.0 | 387 | 0.183 | 0.9121 | 0.8753 | 0.9118 | 0.9121 / 0.9118 |
| 0.5 | 358 | 0.196 | 0.9059 | 0.8666 | 0.9049 | 0.9103 / 0.9096 |
| 0.6 | 317 | 0.234 | 0.8954 | 0.8536 | 0.8939 | 0.9097 / 0.9087 |
| 0.7 | 273 | 0.280 | 0.8825 | 0.8368 | 0.8803 | 0.9056 / 0.9039 |
| 0.8 | 217 | 0.343 | 0.8614 | 0.8074 | 0.8581 | 0.8951 / 0.8923 |
| 0.9 | 155 | 0.421 | 0.8253 | 0.7616 | 0.8207 | 0.8785 / 0.8744 |
| 0.95 | 128 | 0.472 | 0.8032 | 0.7321 | 0.7978 | 0.8739 / 0.8694 |
| 0.99 | 67 | 0.620 | 0.7217 | 0.6298 | 0.7130 | 0.8407 / 0.8333 |

## Holdout

- **Discovery:** v0 and the v1 hub (s0), plus resnet56 (exploratory).
- **Confirmation:** s1, s2, s3 and s4, touched once, after the gate.
- Seeds 1-4 are now spent for margin_typeb. A cut change, a margin-beyond-confidence claim or a re-confirmation needs seeds 5 and 6 and a new pre-registration.
- The confirmation-seed margin values seen here must not seed new definitions.

## ATLAS_STATUS changes (to apply in the same commit as this SESSION.md)

**Row 9:** ⬜ → ✅ s1, s2, s3, s4, with the E1 statement. Exact row:

| 9 | penult | margin_typeb (type-b = wrong and maxprob > 0.7) | margin ~ maxprob (not an independent detector): at cut 0.7 the top-2 margin (d2 − d1) to train-reference class centers separates type-b from correct test samples better than nearest-center distance (M2), and the type-b median margin is ≤ 0.5 × the correct median (M3); the margin AUC equals the maxprob AUC (confidence-matched gap ≤ 0.002, Spearman 0.92-0.93, E1); the ratio clause is cut-dependent (> 0.5 at cut 0.99 in every run) | ✅ s1, s2, s3, s4 | v1 s1/s2/s3/s4: auc_margin_typeb 0.882 / 0.884 / 0.890 / 0.899 (range 0.017; twin 0.0005; hub 0.885), margin − dist 0.046 / 0.063 / 0.063 / 0.055 (p ≤ 1.6e-19), ratio 0.280 / 0.284 / 0.285 / 0.266 · M1 gate v0 legacy block 0.893, raw 0.107 (legacy 0.895, docs/history/SESSION_RESULTS_VALLEYS_TYPEB_SCALE.md:55-59) · E1 confmatched margin − maxprob +0.0017 / +0.0003 (s1/s2) · null auc_margin_wrong 0.542, no type-b · resnet56 hub (E5, exploratory) 0.907 but margin − dist −0.0008 (p 0.83) · results/margin_v1_resnet20_s1/SESSION.md; results/margin_*/atlas.json per_layer.penult.margin_typeb |

**Header paragraph:** add "A3 (row 9) ended **PROMOTE** (results/margin_v1_resnet20_s1/SESSION.md): row 9 is ✅ with the pre-registered qualifier margin ~ maxprob (not an independent detector)."

## Next actions

1. **Commit.** Commit the pulled `results/margin_*` and `results/critic_margin_*` dirs with this SESSION.md and the row-9 and header changes.
2. **Stage 2.** Write `results/atlas_v1_resnet56_s0hub/SESSION.md`; it is not present in this checkout. The row-9 decision does not depend on it, and E5 is read from `m56`.
3. **Discovery only (s0 material):**
   - add a paired DeLong test of margin vs maxprob, and store the top-2 logit gap, so "no information beyond maxprob" becomes testable;
   - check the maxprob distribution of resnet56 for saturation ties.
4. **Still unmeasured:** E3 and M1x. Both need Stage A contract changes (A3_MARGIN.md §4).