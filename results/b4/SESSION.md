# SESSION: batch 4 — what geometry adds beyond the output head (T1), model-level collapse laws (T2), the head-input spatial map (T3S)

This is the Evaluator pass on the three batch-4 tracks (RUN_REQUEST_S2.md "Step E"). Every label below is the frozen
evaluator's output, run once at HEAD `ee08b34` into the path the plan names. No script was edited, nothing was cut, and
no evaluator output was overwritten. Each track section was drafted from the evaluator output, checked by two independent
verifiers against the per-unit files, and corrected (see "Review record").

## Header

- **Plans (binding).** `docs/plans/B4_INTEGRATION.md` (D1-D19), `docs/plans/T1_SCOREBOARD.md`, `docs/plans/T2_COLLAPSE.md`,
  `docs/plans/T3S_SPATIAL.md`, `docs/plans/B4A_AMENDMENT.md`, `docs/plans/B4B_AMENDMENT.md`, `results/b4/RUN_REQUEST_S2.md`.
- **Commits.**

  | tag | commit | what |
  |---|---|---|
  | P1 | `ad97d0a` (`ad97d0a659dd9f6f60b24e0b8cc1681f1bd8a166`) | pre-registration (predictions, rules, evaluators); S1 HEAD |
  | P_A | `14caf51` (`14caf51b7ca241a2dd95dacc531cc63a5bda6a6a`) | amendment B4a: one-sided README gate, S1 supplement for mobilenetv2_x0_75, shufflenetv2_x1_0 (C) and shufflenetv2_x0_5 (Dnew) |
  | R1 | `cb617c3` | S1 and supplement results |
  | P2 | `272bc41` (`272bc41b9395145f50f4289e4eb1df02ce6de273`) | freeze: discovery evaluations, `laws_frozen.json`, `t1_rules.json`, `freeze_P2.json`; amendment B4b (every S2 limit 2x the emitted one); T2 P4 withdrawn to INFO; T1 X3-3 kept ACTIVE with a disclosed caveat; S2 HEAD (`results/instrument_check_b4s2/head.txt`) |
  | R2 | `ee08b34` | S2 confirmation outputs |

- **Sessions and hosts.**

  | session | when (UTC, 2026-09-24) | host | outcome |
  |---|---|---|---|
  | S1 | 06:32-08:52 | 1x RTX 4090, EU-RO-1, AMD EPYC-Genoa (40 host CPUs) | anchors and all 18 old nets pass the anchor gate; rebuild PASS; 8 trainings; 51/53 discovery probes; 34 sealed dumps. Soft failures: the 3 README-gate units only |
  | S1 supplement (B4a) | 08:54-09:07 | same pod | 0 soft failures; 2 more sealed dumps; shufflenetv2_x0_5 discovery probes (tag `_r2`) |
  | S2 | 16:55-18:47 | 1x RTX 4090, EU-RO-1, AMD Ryzen 9 7950X (32 host CPUs), CPU quota 6.8, 46 GB, P = 5 | `block_b4s2` OK, 0 soft failures; `--verify-seals` PASS (36); pytest 137 passed, 1 skipped; 5/5 self-tests PASS; anchor replay PASS (rel 1e-6); 66/66 confirmation and 6/6 replay jobs; `unseal_log.jsonl` 74 records |
  | E (Windows) | 18:49-18:51 | local, node | `b4a_p2.js seals` PASS (18:49:38Z), then `t1_eval.js`, `collapse_laws.js --evaluate`, `t3s_eval.js` (18:50:48Z), each exit 0 |

  The S2 host is not the EPYC-Genoa class that RUN_REQUEST_S2 asks for. Genoa hosts were unavailable and the owner decided
  to run on this Zen 4 host with the same stack (see "Deviations").
- **Cost (RunPod).** Estimates: S1 with the B4a supplement about $1.9, S2 about $1.4, test pods about $0.05. Account
  balance 178.09 → 175.13 USD (−2.96). The per-session figures are estimates and sum to about $3.35; the balance change
  is the billed figure.
- **Outputs of step E** (untracked until committed): `results/b4/b4a_seals.json`, `results/b4/eval_t1.json`,
  `results/b4/t2_eval.json`, `results/b4/eval_t3s.json`.
- **Scope of every result here.** CIFAR-10 (test rows), CIFAR-10-C, CIFAR-100 and SVHN; synthetic corruptions and
  synthetic sensor faults; i.i.d. frame streams (which overstate rates relative to video). Confirmation units were never
  read before P2: 4 fresh ResNet20/56 seeds (R), 8 older ResNet20/56 on never-read rows (R2), 12 sealed hub architectures
  of five families (C / M), 10 stream units (STconf), 4 spatial units (Sconf).

## The batch-level answer (primary question)

**Question** (owner): which controller-usable information does a backbone's internal geometry give beyond the output head,
and how reliably?

**Answer, in one line.** On these CIFAR backbones, geometry adds nothing to the head on clean inputs; its controller value
is under shift and at early taps (what changed, which corruption, how soon), where it adds modestly and consistently on
ResNet20/56; much of that information is also available from pixel statistics; and neither the collapse level of a
backbone nor the head-input spatial map adds what was hoped.

**Where geometry adds beyond the head, and how reliably.**
1. **Error flag under corruption shift (F1-a under shift).** On CIFAR-10-C s3 errors, a head + pre-tap joint model adds
   +0.014 to +0.049 AUROC over the full head on all 12 ResNet20/56 units (X1-4, pre-registered), which is +5 to +17 pt of
   TPR at 5% FPR. Early-tap features add beyond the head *and* pixel statistics (+0.005 to +0.020, X1-5), and prediction
   depth adds (X7-1); both were predicted to fail. Limits: the pre-tap gain is ResNet-specific (4/12 other architectures);
   the early-tap gain appears on 11/12 (INFO); on the 5 holdout corruption families (the joint model cross-fitted on
   those families' own rows, not a transfer test) the pre-tap gain is smaller, +0.004 to +0.014 (INFO).
2. **Which corruption it is (ADAPT routing).** Early taps identify the corruption family of each input far better than the
   head: +0.12 to +0.34 AUROC per family on 12/12 ResNet units (PC-2a). Most of this is pixel-level: beyond pixel
   statistics it drops to +0.005 to +0.082 on R (PC-2b; +0.004 to +0.083 over R and R2, two R2 family calls just below
   the +0.005 bar).
3. **When a stream changes (F1-d).** At the same false-alarm budget (ARL0 2000), a stage-1/2 MEWMA detects mild steps in
   0.07-0.16 of the best head-only delay (7.5-30.8 vs 65-252 frames; X3-2, 10/10 units), and stage 1 carries about 10x the
   penult's change information (X3-1). Limits: a pixel MEWMA is comparable (faster on pixelate, slower on snow; INFO); there
   is no ≥ 100-frame warning before harm (X3-3); early-tap thresholds calibrated on one set of clean rows over-protect on
   held-out rows (X3-4).
4. **Novel inputs (ESCALATE).** The calibrated multi-tap flag X4 adds to the head for SVHN on ResNet20/56 (+0.025 to
   +0.085 AUROC, X4-6), on only 5 of the 12 other architectures, and not for CIFAR-100. Whether an unsupervised pixel flag
   would do as well is untested.
5. **Batches.** A pre-tap batch residual flags 99.5-100% of mild covariate-shift batches that head batch tests largely miss
   (X6-5), but a pixel T2 test flags them too. As a batch accuracy estimator, the geometric grade H beats ATC-MC on 12/12
   units but not the head's own mean-gap / mean-energy estimator (X6-1, INFO).
6. **Calibration (configuration).** The DO-3 law (train-referenced density thresholds over-alarm 2-3.7x on clean data)
   replicates on never-read rows of 12 ResNet units (X4-2) and transfers to 10/12 hub architectures (T2 M-DO3). Held-out
   calibration fixes it: X4's false-alarm budget transfers to 12 architectures with no retuning (X4-1M), running slightly
   conservative.

**Where it does not add.**
7. **Clean inputs (F1-a).** The centre margin, 19 geometric signals at five taps, and Trust Score / kNN purity / LID add
   nothing to the full head on clean errors (X1-2, X1-3, X8-1: 4/4 fresh seeds, 8/8 R2, 11-12/12 architectures; highest
   upper 95% bound on R +0.0062). A clean-input "likely wrong" flag is a head function.
8. **Collapse as a model-level predictor (T2, the owner's PRIMARY).** A backbone's label-free collapse level does not
   predict how its detectors behave beyond what its accuracy already says: 0 WIN, 5 LOSS on 12 sealed architectures (new
   members of the five families in the discovery set; CIFAR-10 only). The one exception is the harm slope (P5a, candidate).
9. **Local sensor faults (T3S, and T1 X3-8).** The frozen per-position statistic of the head-input spatial map is at chance
   on 12%-area local faults, does not localise them, misses a held-out fault family and does not flag the "sensor changed,
   output right" case. Per frame, early taps beat the head by ≥ 0.10 only on global exposure (and on dead pixels on some
   architectures); the head wins on occluders. A pixel check sees occluders, glare and soiling (INFO).
10. **Harm warnings, benign-shift gating, label shift.** No ≥ 100-frame warning before harm on ramps (X3-3; the early alarm
    does come first on motion blur, by 16-64 frames), no demonstrated HOLD-T gate (X3-6), and label shift is a head signal
    (BBSDh), not a geometric one (X3-7, X6-3, X6-4).

**How reliably.** Every labelled result above is a pre-registered claim scored on units never read before P2, with
group-bootstrap CIs (B = 1000) and a frozen three-way rule (ADDS / BOUNDED / INCONCLUSIVE); three positives (X1-5, X7-1,
PC-2b) were predicted to fail and stay CANDIDATE, and INFO items are marked as such. The scope limits are one
dataset family (CIFAR), synthetic shifts and faults, i.i.d. frames, and, for the positives in items 1-5, mostly ResNet20/56.
Because several positives are shared with pixel statistics (items 2-5), the next batch must compare with unsupervised
pixel baselines, not only with the head.

## Review record

Each track decision was checked by two verifiers who recomputed every number from the evaluator outputs and the per-unit
files with node, without running an evaluator or editing a file. All labels, gates and counts reproduced. The corrections
concerned secondary numbers, R-only ranges quoted beside R2/M counts, and plain-language readings; every one was checked
again against the data before it was applied. Applied: all, except two counts in the second T1 verifier's report,
corrected against the data:
- X4-3 INFO (X4 against the frozen head.best_stat): the verifier's "at least 6 of 10 splits on 6/8 R2 units" is 5/8
  (splits won: 3, 7, 5, 7, 6, 8, 4, 8), so 7/12 R/R2 units, not 8/12.
- X4-5: the verifier's "positive (+0.010 to +0.049) on every R/R2 unit" starts at +0.0056 (resnet20_s2); the range used
  here is +0.006 to +0.049, with the 95% CI above 0 on every R/R2 unit.

Found while applying them: the X1-2 margin's dI CI is above 0 on resnet20_s31 and on 7 of 12 M units (a verifier
listed three as examples), all at dAUC < +0.005; and the run-wide timing maximum is 0.528 of the emitted limit, not the 0.56 in
the R2 commit message (see "Deviations").

One writer addition, flagged as INFO where it appears: the X1-4 pre-bundle gain on the five holdout corruption families
(`persample.err_shift_holdout_s3`; no claim reads it).

Final review (corrections applied after the two verifiers, each checked against the data with node):
- `err_shift_holdout_s3` is not a transfer target. `scripts/t1_scoreboard.py` `targets()` builds it from the five
  holdout splits only and gives it no `transfer_from`, so the joint model is cross-fitted on those families' own rows
  (only the `corrupt_*_s3_transfer` detection targets train on the discovery families). The writer's "trained on the
  discovery corruptions and scored on the unseen families" reading was wrong. It is now reported as the gain on those
  families, which is smaller; how a fitted joint model transfers to unseen families was not measured for errors.
- The clean increments are not all below +0.01: vgg16_bn's all_geom is +0.0109 (X1-3 INCONCLUSIVE). Some that are
  detectable (dI CI above 0) exceed +0.005: vgg19_bn all_geom +0.0083, repvgg_a2 local_pen +0.0054.
- Smaller fixes: the R2 dTPR maximum is 0.166498 (resnet20_s1), so +0.166, not +0.167; PC-2b's +0.005 to +0.082 is
  the R range; the head rule's covariate typing accuracy is 0.785-0.865 on R/R2; P6b's rank law also beats accuracy and
  the head by MAE (it fails only its constant clause), so P5a is the only *supported* label-free model-level law, not
  the only one ahead of them; the S2 host was not uniformly slower (T1 anchor replays 225-227 s against S1's 258-259 s,
  T2 anchor replays 81 s against 65-70 s).
- The same corrections were made in ATLAS_STATUS.md (rows 12, 13) and docs/knowledge/README.md (CM-1, CM-11, PC-2,
  ST-1, section 6), plus two there only: X4's calibrated clean FPR on R/R2 is 2.5-5.1%, not 2.5-5.3% (0.0527 is a hub
  unit; DO-4, section 6), and X4's blur-presence increment ADDS on 11/12 R/R2 units, not "mostly below the bars" (DO-1).

---

## Track T1: what intermediate geometry adds beyond the full output head (`docs/plans/T1_SCOREBOARD.md`)

**Run.** The evaluator ran once at HEAD `ee08b34` and exited 0 with an empty stderr:
`node scripts/t1_eval.js --p1 ad97d0a659dd9f6f60b24e0b8cc1681f1bd8a166 --p2 272bc41b9395145f50f4289e4eb1df02ce6de273 --p-run ad97d0a…,14caf51…,272bc41… --json results/b4/eval_t1.json`
- It wrote `results/b4/eval_t1.json` (87,590 bytes, created 2026-09-24T18:50:48Z).
- No script was edited and nothing was cut (`cuts: []`). `rules_applied` shows every claim ACTIVE at P2, X3-3 included.

**Units.**

| scope | units | layout and rows |
|---|---|---|
| R (primary) | resnet56_s31, resnet56_s32, resnet20_s31, resnet20_s32 (the fresh seeds) | eval layout; never-read test rows 5000-9999 |
| R2 | resnet20_s1 to s4; resnet56_s1, s2, s12m, s13m | eval layout |
| M | the 12 sealed C architectures (resnet44; vgg13/16/19_bn; mobilenetv2 x0_75/x1_0/x1_4; shufflenetv2 x1_0/x1_5/x2_0; repvgg a1/a2) | fit layout |
| S | the 10 STconf stream units | streams |

**Gate line (verbatim):** `gates: selftests PASS; provenance PASS; read_guard PASS; replay PASS; anchors PASS; model_outcomes PASS; rules PASS`
- **selftests:** 5 PASS records in `results/instrument_check_b4s2/`, including the two required ones
  (`selftest_t1_scoreboard.json`, `selftest_t1_streams.json`).
- **provenance:** PASS against P2 with 0 failures: P1 ≤ P2 ≤ HEAD; all 13 frozen files byte-identical to P2; every
  output's `repo_commit` in `--p-run`, nboot 1000; every sealed dump opened with `ATLAS_B4_UNSEAL` = P2.
- **read_guard:** PASS; nothing was scored twice.
- **replay:** PASS with drift 0 (`t1:resnet20_hub`, `t1:resnet56_hub`; `results/instrument_check_b4s2/replay.json`).
- **anchors:** margin PASS (13 units), DO-3 PASS (16 units).
- **model_outcomes:** 20 of 20 fit-layout units are schema-valid (T2's O1/O5 inputs; no T1 claim reads them).

**Outcome.** All 31 claims got a label: 18 SUPPORTED, 13 REFUTED, 0 INCONCLUSIVE, 0 NOT_EVALUABLE. The label matched
the P1 prediction on 17 claims and differed on 14; 11 of those 14 were already on record as discovery INFO labels at P2.
The labels first seen at confirmation are X4-1R and X4-4 (both flipped from discovery) and X4-5 (no discovery label).

How to read the numbers: the three-way rule is frozen (T1_SCOREBOARD §3.4). **ADDS**: dAUC_joint ≥ +0.01 and the 95% CI
of dI above 0. **BOUNDED** (the functional "no"): the upper 95% bound of dAUC_joint < +0.02. dAUC is over the full head
(the 10 sorted logits plus splines, cross-fitted), with its 95% group-bootstrap CI (B = 1000); dTPR is at 5% FPR.

### T1.1 Headline answers (plain language)

1. **Clean inputs: no geometric addition to the full head.**
   - The penult centre margin (X1-2), all 19 geometric signals at five taps (X1-3), and Trust Score, kNN purity and LID
     (X8-1) are BOUNDED against the full head on every R, R2 and M unit, with one INCONCLUSIVE (M vgg16_bn on X1-3).
   - The increments are below +0.01 AUROC except vgg16_bn's all_geom (+0.0109, INCONCLUSIVE); the highest upper 95%
     bound is +0.0062 on R and +0.0093 for the margin on M. Some are statistically detectable (margin dI CI above 0 on
     resnet20_s31 and on 7 of 12 M units, all at dAUC < +0.005; on M also vgg19_bn all_geom +0.0083 and four local_pen
     increments up to +0.0054), but none is usable.
   - For a clean-input "likely wrong" flag (F1-a), use the head statistic chosen on fit rows (gap, pnorm2 or entropy); it
     stays within 0.002 of MSP or beats it on R (X1-1).
2. **Shifted inputs: a modest, consistent gain on the ResNets.**
   - On CIFAR-10-C s3 errors, the pre-tap bundle adds +0.014 to +0.049 AUROC over the full head on R and R2 (X1-4), which
     is +4.6 to +16.6 pt of TPR at 5% FPR.
   - The early-tap bundle adds beyond the head *and* the pixel statistics (X1-5, +0.005 to +0.020 on R/R2), and prediction
     depth adds (X7-1). Both were predicted REFUTED.
   - Across the 12 C architectures the pre bundle adds on only 4 (M REFUTED); the other 8 are +0.003 to +0.008 with CIs
     above 0, below the +0.01 bar. The early bundle (INFO: no claim has M scope) ADDS on 11/12 C architectures (median
     +0.022), also beyond pixels on 11/12. So the architecture dependence belongs to the pre bundle.
   - INFO (writer addition, corrected at final review): on the 5 holdout corruption families, with the joint model
     cross-fitted on those families' own rows (not trained on the discovery families), the pre-bundle increment is
     +0.007 to +0.014 on R (ADDS 2/4) and +0.004 to +0.013 on R2 (ADDS 1/8) (`persample.err_shift_holdout_s3`). On the
     4 extras, fitted the same way, it is +0.011 to +0.029 on R (ADDS 4/4). The size of the gain depends on the
     corruption family; transfer of a fitted joint model to unseen families was not measured for errors.
3. **Which corruption it is: early taps read it, the head does not.**
   - Early-tap family identification adds +0.12 to +0.34 AUROC over the head (PC-2a).
   - Much of that is pixel-level: beyond the head and pixels it drops to +0.005 to +0.082 (PC-2b); the noise family sits
     just above the +0.005 bar (+0.0052 to +0.0064 on R) and falls below it on one R2 unit.
   - This is an input for ADAPT routing (which fix to apply).
4. **The calibrated multi-tap flag X4: its false-alarm budget transfers; as a corruption flag it does not beat the
   registered head comparator.**
   - It holds its clean false-alarm band with no retuning on all 12 C architectures (X4-1M).
   - On R it runs conservative: one fresh seed (0.0253) fell below the exact band, so X4-1R is REFUTED on the safe side.
     It is conservative on average (pooled 0.036 R, 0.040 R2, 0.043 M), not a strict upper bound (four R2/M units reach
     0.051-0.053).
   - At a matched 5% false-alarm rate it does not beat the per-split best of 9 head statistics on noise, blur or pixelate
     (X4-3). That comparator is an oracle choice, disclosed at P2 as favouring the head. Against the head statistic
     pre-selected for F1-a (head.best_stat) X4 gains ≥ 0.05 TPR on ≥ 6/10 splits on 2/4 R and 5/8 R2 units (INFO).
   - Corruption presence (X4-4) and transfer to unseen families (X4-5) fail their bars, although the increments are
     positive on most ResNet units.
   - It adds for SVHN over the head on the ResNets (X4-6); univariate SVHN AUROC is 0.927-0.962 for X4 against
     0.855-0.930 for the best head statistic (R).
5. **Streams: early taps detect changes about ten times sooner, but give no ≥ 100-frame warning before harm, and their
   held-out calibration runs conservative.**
   - At ARL0 2000, stage-1/2 MEWMA detects s1 steps in 0.07-0.16 of the best head-only delay (X3-2).
   - Stage 1 carries about 10x the penult's change information (X3-1), about as much as the pixel factors.
   - INFO: a model-free pixel MEWMA detects the same steps in 14.1 (motion), 5.0 (pixelate) and 70.9 (snow) frames,
     against 7.5-11.1, 14.2-30.8 and 8.7-14.0 for the best early tap; its held-out false alarms are 0.80 per 1000 frames
     (nominal 0.5).
   - On motion-blur ramps the best early-tap median alarm precedes the 5-pt harm reference on all 10 units, by 16-64
     frames; none reaches 100 (X3-3, with the disclosed harm-reference caveat). On defocus ramps the leads are −74 to
     +347.5.
   - Held-out ARL0 is too long, never too short, for early-tap detectors (X3-4).
6. **Batches: H beats ATC-MC but not the head's best batch estimator.**
   - H estimates batch-64 accuracy better than ATC-MC on 12 of 12 units (X6-1 REFUTED, in geometry's favour).
   - INFO: the head's mean-gap or mean-energy estimator is as good as or better than H on 8 of 12; the penult
     displacement estimator is the best of all estimators on 7 of 12 (all 6 resnet56 units), not on the resnet20 R units.
   - H adds nothing within splits (X6-2). The label-skew-quiet residual fails (X6-3). Window typing fails for both rules
     (X6-4; T1.4 item 7 explains why the frozen cascade is degenerate).
   - R_pre flags mild covariate-shift batches at 0.995-1.00 (X6-5); a pixel T2 test flags them at 0.96-1.00.
7. **Sensor faults per frame: early taps win only under global exposure; the head wins on occluders.**
   - Early taps beat the head by ≥ 0.10 for global exposure. The head beats them on occlusion discs by 0.22-0.30 (R).
   - On dead pixels the frozen early list is at or above the head (−0.003 to +0.067 on R). INFO: the s2end tap (outside
     the frozen list) gives +0.09 to +0.145, and on M dead pixels clear +0.10 on 6/12 units (VGG, ShuffleNetV2). X3-8 still
     fails on every unit because of occlusion discs.
   - A frozen frame is caught by the penult and stem MEWMA (X3-5), and just as well by head window tests (INFO).
8. **The DO-3 law replicates** on never-read rows of 12 units, 8 of them models new to the law (the 4 fresh seeds and
   resnet20 s1-s4; X4-2). Train-referenced thresholds over-alarm at 2-3.7x nominal, as the law predicts.
9. **Cost** (functional block, R medians; measured on the 7950X under the concurrent S2 load and not covered by the
   replay): head statistics 0.45 ms per 1000 queries; centre-based signals 0.11-0.20 ms; L2-kNN per tap 122-142 ms; X4
   411 ms with a 5.84 MB reference. On M, X4 needs up to 1199 ms and 122 MB.

**Controller use (F1/F2), within scope (CIFAR-10/-C, i.i.d. frames):**

| function | recommendation | basis |
|---|---|---|
| F1-a, clean input | head only (the statistic chosen on fit rows) | X1-1, X1-2, X1-3, X8-1 |
| F1-a, under shift | head + early bundle (ResNet20/56; 11/12 C, INFO); the pre bundle is ResNet-specific; the gain is smaller on some corruption families (holdout families, INFO) and its transfer to unseen families is untested | X1-4, X1-5, X7-1; INFO |
| ADAPT routing | per-sample family from early taps (mostly pixel-level) | PC-2a, PC-2b |
| F1-d stream change | early-tap or pixel MEWMA; thresholds conservative on held-out rows; a change detector, not a harm detector | X3-2, X3-4, X3-3; INFO |
| F1-c / ESCALATE far-OOD | X4 over the head (SVHN only, ResNets); an unsupervised pixel pre-model flag is untested | X4-6 and INFO |
| label shift | head BBSDh | X3-7 |
| HOLD-T benign-shift gate | not shown | X3-6 |

### T1.2 Claim table

R values are in the order resnet56_s31, resnet56_s32, resnet20_s31, resnet20_s32. For adds / bounded claims, "need" is
the number of units with the asserted call. Per-unit numbers come from `results/b4_t1/<unit>_eval/scoreboard.json` (R, R2),
`results/b4_t1/<unit>_fit/scoreboard.json` (M) and `results/b4_t1s/<unit>/streams.json` (S); labels and counts from
`results/b4/eval_t1.json`.

| claim (rows) | prediction | label | units passing / needed | key numbers (with CIs) | what it means for the controller |
|---|---|---|---|---|---|
| X1-1 (CM-4) | SUPPORTED | **SUPPORTED** | R 4/4 (need 4); R2 5/7 REFUTED (need 7; resnet56_s13m chose MSP and is not evaluable) | Chosen statistic minus MSP AUROC on clean errors: gap +0.0041, pnorm2 +0.0080, entropy −0.0008, gap −0.0002 (univariate AUC, no CI; 282-355 errors per 5000 rows). R2 failures: resnet20_s2 −0.0033 and resnet20_s4 −0.0025 (both mspT). MSP chosen on 0/4 R, 1/8 R2 | Use the statistic chosen on fit rows as the F1-a score and the bar. On resnet20 it gains nothing over MSP, and mspT can lose 0.003 |
| X1-2 (CM-1, CM-2) | SUPPORTED | **SUPPORTED** | R BOUNDED 4/4 (need 4); R2 8/8; M 12/12 | margin_pen −0.0004 [−0.0014, +0.0004], −0.0000 [−0.0012, +0.0011], +0.0010 [−0.0002, +0.0023], +0.0002 [−0.0007, +0.0010]; highest upper bound on M +0.0093 (repvgg_a2). dI CI above 0 on resnet20_s31 ([0.0002, 0.0026] bits) and on 7 of 12 M units, all at dAUC < +0.005 | On clean inputs the centre margin adds under +0.01 AUROC to the full head (functionally nothing); do not add it to F1-a |
| X1-3 (CM-1, CM-11) | SUPPORTED | **SUPPORTED** | R BOUNDED 4/4 (need 3); R2 8/8; M 11/12 (need 9; vgg16_bn INCONCLUSIVE) | all_geom −0.0014 [−0.0052, +0.0022], +0.0010 [−0.0028, +0.0046], +0.0028 [−0.0004, +0.0062], −0.0021 [−0.0057, +0.0015]; dTPR −0.025 to +0.034. vgg16_bn +0.0109 [+0.0018, +0.0212] with dI CI [−0.0045, +0.0047] | No tap gives a usable clean-input error signal beyond the full head. F1-a on clean data is a head function |
| X1-4 (CM-11) | SUPPORTED | **SUPPORTED** | R ADDS 4/4 (need 2); R2 8/8; M REFUTED ADDS 4 / BOUNDED 8 | pre bundle on s3 errors: +0.0426 [0.0395, 0.0454], +0.0418 [0.0385, 0.0451], +0.0317 [0.0290, 0.0343], +0.0137 [0.0122, 0.0152]; dI CI > 0 on all four; dTPR +0.046 to +0.142. R2 +0.021 to +0.049, dTPR +0.073 to +0.166. On M only resnet44, vgg16_bn, vgg19_bn and repvgg_a2 add; the other 8 are +0.0029 to +0.0083 with CIs above 0. INFO: the early bundle ADDS on 11/12 M units (median +0.0215; shufflenetv2_x1_0 BOUNDED at +0.0096) | Under shift, a head + pre-tap joint model catches 5-17 pt more wrong inputs at 5% FPR on the ResNets. The pre bundle's gain is ResNet-specific; the early bundle's is not (INFO) |
| X1-5 (CM-9) | REFUTED | **SUPPORTED** (not predicted) | R ADDS 4/4 (need 2); R2 8/8 | Beyond head AND pixels: +0.0151 [0.0134, 0.0168], +0.0174 [0.0156, 0.0192], +0.0194 [0.0177, 0.0211], +0.0054 [0.0044, 0.0063]; R2 +0.0085 to +0.0203. Over the head alone +0.010 to +0.040 (R) | Early-tap features add shift-time error information that neither the head nor pixel statistics hold. The gain beyond pixels is small (≤ +0.02) |
| PC-2a (PC-2, ST-1) | SUPPORTED | **SUPPORTED** | R 4/4 (need 3; each unit ADDS on 4 of 4 families); R2 8/8 | Early bundle over the head, one family vs the rest at s3 (R): noise +0.186 to +0.295, blur +0.289 to +0.340, weather +0.122 to +0.169, digital +0.218 to +0.235; smallest CI lower bound +0.1165 (weather, resnet56_s31) | Each input's corruption family is readable from early taps, not from the head. This is the input for ADAPT routing |
| PC-2b (PC-2) | REFUTED | **SUPPORTED** (not predicted) | R 4/4 (need 3); R2 8/8 | Beyond head AND pixels (R): noise +0.0052 to +0.0064 (bar +0.005; CI lower 0.0046-0.0056), blur +0.0134 to +0.0150, weather +0.0077 to +0.0232, digital +0.0762 to +0.0820. On R2 two family calls fall below the bar: resnet20_s4 noise +0.0049, resnet56_s12m weather +0.0043 | Most of the family signal is pixel-level. The features add a little beyond pixels, most for the digital family |
| X8-1 (CM-11) | SUPPORTED | **SUPPORTED** | R BOUNDED 4/4 (need 4); R2 8/8; M 12/12 | local_pen −0.0006 [−0.0023, +0.0013], +0.0010 [−0.0009, +0.0036], +0.0018 [+0.0002, +0.0034], +0.0003 [−0.0018, +0.0024]; highest upper bound on M +0.0146 (vgg16_bn) | Trust Score and kNN purity are baselines, not additions, on clean inputs |
| X7-1 (CM-9) | REFUTED | **SUPPORTED** (not predicted) | R ADDS 3/4 (need 2); R2 8/8 | Trajectory on s3 errors: +0.0326 [0.0301, 0.0354], +0.0341 [0.0308, 0.0371], +0.0257 [0.0232, 0.0282]; resnet20_s32 BOUNDED +0.0062 [0.0049, 0.0074]. R2 ADDS +0.013 to +0.050. Clean-error trajectory dAUC −0.0008 to +0.0015 (R) | The depth at which the nearest-centre label settles is error information that appears only under shift |
| X2-1 (ST-4) | REFUTED | **REFUTED** | R ADDS 0/4, BOUNDED 4/4 (need 2 ADDS at the claim's +0.03 bar); R2 BOUNDED 8/8 | Head-null residual, corruption presence: noise-family increments on R +0.0082 to +0.0356 (ADDS at the general +0.01 rule on 3/4); above +0.03 only resnet20_s31 (+0.0356 [0.0294, 0.0417]) on R, and resnet20_s1 (+0.0364) and resnet56_s13m (+0.0333) on R2; every other R family increment ≤ +0.018 | The whitened off-head residual does not add ≥ +0.03 in two or more families; on noise it adds a little |
| X4-1R (DO-6, DO-3) | SUPPORTED | **REFUTED** | R 3/4 (need 4); R2 7/8 SUPPORTED | Clean FPR on 1500 never-read rows (Wilson 95% CI); band [0.0327, 0.0693]: 0.0427 [0.0336, 0.0541], 0.0360 [0.0277, 0.0467], **0.0253** [0.0185, 0.0346], 0.0400 [0.0312, 0.0511]. The R2 failure is resnet20_s4 at 0.0320 (48/1500; the band starts at 49) | The failure is on the safe side (too few false alarms). On R a 5% budget was realised at 2.5-4.3%; pooled it is conservative (0.036 R, 0.040 R2, 0.043 M) but not a strict upper bound: resnet56_s12m (R2) and resnet44, mobilenetv2_x1_4, shufflenetv2_x2_0 (M) reach 0.0507-0.0527 |
| X4-1M (DO-6) | SUPPORTED | **SUPPORTED** | M 12/12 (need 10) | FPR 0.0300-0.0527 (median 0.045), all inside the band [0.0273, 0.0767] (n_cal 750), with no retuning | The conformal flag's false-alarm budget transfers to 12 unseen architectures |
| X4-2 (DO-3) | SUPPORTED | **SUPPORTED** (replication) | R 4/4 (need 3); R2 8/8 | abs(sparse − (0.0224 − 0.108 log10 nc1)) = 0.0150, 0.0216, 0.0081, 0.0035 (tolerance 0.035); sparse 0.176, 0.184, 0.115, 0.102 at nc1 0.053, 0.050, 0.166, 0.171; largest R2 deviation 0.0159 | A train-referenced threshold over-alarms 2-3.7x on clean data, as predicted. Calibrate on held-out rows, as X4 does |
| X4-3 (DO-1) | SUPPORTED | **REFUTED** | R 1/4 (need 3); R2 3/8; M 1/12 | Splits (of 10) where X4 TPR ≥ the per-split best of 9 head TPRs + 0.05: 3, 4, 1, 6 (need 6); median difference +0.024, −0.002, −0.026, +0.081. On M, 10 of 12 units won 0/10. INFO: against the frozen head.best_stat X4 wins on 7, 4, 4, 8 splits (R) and on ≥ 6 on 5/8 R2 units | At a matched 5% false-alarm rate X4 does not beat an oracle head comparator (the per-split best of 9 statistics, disclosed at P2 as favouring the head). Against the deployable pre-selected statistic it often does (INFO). Its value as a stand-alone corruption flag is not settled; as a feature it helps (X4-4, X4-6) |
| X4-4 (DO-1) | SUPPORTED | **REFUTED** | R ADDS 2/4, BOUNDED 2/4 (need 3 ADDS); R2 ADDS 4, BOUNDED 4; M ADDS 4, BOUNDED 8 | Noise family: resnet56 BOUNDED +0.0079 [0.0045, 0.0112] and +0.0091 [0.0057, 0.0121] (dI CI above 0); resnet20 ADDS +0.0152 [0.0105, 0.0202] and +0.0203 [0.0156, 0.0253]; R2 resnet56_s1 +0.0274 and resnet56_s2 +0.0796 ADDS. Blur family ADDS on all four R (+0.0132 to +0.0444) | X4 helps the head detect blur. For noise, the increment on the fresh resnet56 seeds is below +0.01 (CI above 0); on R2 resnet56 s1/s2 it adds |
| X4-5 (DO-1) | SUPPORTED | **REFUTED** | R 2/4 (need 3); R2 3/8; M 2/12 | Head + X4 over the head on the 5 holdout families: +0.0263 [0.0223, 0.0309], +0.0261 [0.0219, 0.0303], +0.0131 [0.0098, 0.0166], +0.0147 [0.0098, 0.0191] (bar +0.02; head-only AUROC 0.761-0.772). R2 passes on resnet20_s4, resnet56_s1, resnet56_s2 and fails on resnet56_s12m (+0.0178) and s13m (+0.0096). Every R/R2 increment is positive (+0.006 to +0.049, CI above 0). On M, 10 of 12 are below the bar (shufflenetv2_x1_0 −0.0006) | X4 carries a small gain to unseen families on every ResNet unit, above the +0.02 bar on 2/4 R and 3/8 R2 with no clean depth split; not on most other architectures |
| X4-5x (DO-1) | SUPPORTED | **SUPPORTED** | R 3/4 (need 3); M REFUTED 5/12 | On the 4 CIFAR-10-C extras: +0.0333 [0.0287, 0.0377], +0.0331 [0.0283, 0.0379], +0.0158 [0.0126, 0.0192] (fails), +0.0229 [0.0182, 0.0277] | As X4-5, with one more resnet20 unit passing. Not an architecture-general transfer |
| X4-6 (DO-4) | SUPPORTED | **SUPPORTED** | R ADDS 4/4 (need 3); R2 8/8; M REFUTED ADDS 5 / BOUNDED 7 | SVHN vs clean: +0.0311 [0.0268, 0.0357], +0.0307 [0.0263, 0.0349], +0.0607 [0.0528, 0.0685], +0.0853 [0.0766, 0.0948]; dTPR +0.18 to +0.42; R2 +0.025 to +0.069, dTPR +0.14 to +0.41. Univariate SVHN AUROC on R: X4 0.927-0.962, best head statistic 0.855-0.930. INFO: a cross-fitted pixel classifier trained on SVHN-vs-clean labels reaches 0.9968 (head + pixels 0.9985-0.9990; X4 adds ≤ +0.0005 beyond it); no unsupervised pixel flag was scored. CIFAR-100: X4 BOUNDED on R 4/4 (+0.0044 to +0.0099, median +0.0061) | X4 improves far-OOD flagging over the head on the ResNets (SVHN only). Pixel factors carry the information too; whether an unsupervised pixel flag suffices is untested |
| X6-1 (LH-1, LH-2) | SUPPORTED | **REFUTED** | R 0/4 (need 3); R2 0/8 | Batch-64 leave-one-split-out MAE (pp): ATC-MC 8.92, 8.03, 8.09, 7.14 vs H 6.37, 6.51, 6.23, 5.31; H better on all 12 R/R2 units (R2 by 0.84-4.18 pp). INFO: best head estimator (mean gap) 6.55, 5.97, 5.58, 5.44; min(mean gap, mean energy) is as good as or better than H on 8/12; penult displacement is the lowest of all estimators on 7/12 (all 6 resnet56 units, 5.15-5.81 pp) and worse on the resnet20 R units (7.24, 7.85) | H estimates batch accuracy 1.5-2.5 pp better than ATC-MC (R), but beats the head's own mean-gap / mean-energy estimator on only 4 of 12 units. The penult displacement estimator is the geometric one to register (INFO) |
| X6-2 (LH-1) | REFUTED | **REFUTED** | R 0/4 (need 2); R2 0/8 | Mean within-split partial Spearman of H and loss given ATC: 0.136, 0.150, 0.115, 0.094 (need ≥ 0.3); R2 0.104-0.159 | H does not grade harm within a split beyond ATC |
| X6-3 (ST-3) | SUPPORTED | **REFUTED** | R 0/4 (need 3); R2 0/8 | Single-class skew, batch 64: R_pre FPR mean 0.176, 0.148, 0.198, 0.106 (need ≤ 0.10) and max 0.36, 0.50, 0.40, 0.22 (need ≤ 0.15); R2 mean up to 0.244; H mean 0.154-0.282 on R/R2 (need ≤ 0.10). Penult T2 (1.00) and BBSDh at Dirichlet 0.1 (1.00) fire as required | The BBSE-explained residual is not quiet under label skew, so it cannot tell a class-mix change from covariate shift |
| X6-4 (ST-1, ST-5) | SUPPORTED | **REFUTED** | R 0/4 (need 3); R2 0/8 | Macro typing accuracy, geometry vs head rule: 0.340 vs 0.277, 0.342 vs 0.283, 0.347 vs 0.265, 0.340 vs 0.277 (need ≥ 0.85). By type: covariate 1.00 vs 0.785-0.865; prior ≤ 0.015; novelty ≤ 0.05 (R/R2). See T1.4 item 7 | Neither frozen cascade types batch windows; both call almost every window covariate |
| X6-5 (LH-2) | SUPPORTED | **SUPPORTED** | R 4/4 (need 3); R2 8/8 | R_pre flag rate on motion, pixelate and snow s1 batches: 0.995-1.000. INFO: R_pre's clean-batch FPR 0.05-0.105 (R), up to 0.165 (R2 resnet56_s2); pixel T2 0.96-1.00; head BBSDs 0.345-0.86, BBSDh ≤ 0.11 (R/R2) | Mild covariate shift is visible at batch 64 in R_pre, and also in pixel T2. Head batch tests see much less of it |
| X3-8 (PC-7) | SUPPORTED | **REFUTED** | R 0/4 (need 3); R2 0/8; M 0/12 | Best early-tap minus best head per-frame AUROC (frozen early list: knnL2 / wnorm at stem and s1end, x4tap_stem), R: occlusion disc −0.30, −0.25, −0.23, −0.22; dead pixels −0.003, +0.018, +0.067, +0.056; global exposure +0.182, +0.166, +0.146, +0.146 (bar +0.10 for every kind). INFO: at s2end dead pixels +0.145, +0.140, +0.091, +0.100; on M dead pixels ≥ +0.10 on 6/12 (vgg13/16/19_bn, shufflenetv2 x1_0/x1_5/x2_0), occlusion −0.03 to −0.24 on all 12 | Early taps beat the head by ≥ 0.10 for global exposure; the head wins on occluders; on dead pixels early taps are at or above the head. No per-frame early-tap flag works for all three kinds |
| X3-1 (ST-9) | SUPPORTED | **SUPPORTED** | S 10/10 (need 7) | Disjoint CIs on 10 of 10 corruptions in every unit. Over the 100 unit x corruption cells: median I(s1end) 0.587 nats vs I(penult) 0.114 (ratio median 10.6, min 1.28); head 0.185; pixels 0.663 | Stage-1 features carry about 10x the penult's change information per frame, about as much as the pixel factors |
| X3-2 (ST-9) | SUPPORTED | **SUPPORTED** | S 10/10 (need 7) | Best early-tap / best head-only censored delay at ARL0 2000: 0.072-0.161 over 30 cases (need ≤ 0.25). Early 7.5-30.8 frames vs head 65-252 frames. INFO: pixel MEWMA 14.1 / 5.0 / 70.9 frames on motion / pixelate / snow s1 steps | A stage-1/2 MEWMA detects an s1 step 6-14x sooner than any head-only detector at the same false-alarm budget (F1-d); a pixel MEWMA is comparable |
| X3-3 (ST-9) | SUPPORTED | **REFUTED** (ACTIVE; disclosed caveat, T1.5) | S 0/10 (need 7) | Best early-tap median lead before the rolling loss reaches 5 pt: motion blur +16 to +64 frames (positive on 10/10; 0/10 reach 100); defocus −74 to +347.5 (4/10 reach 100). Cap from the disclosed method (stuck lead + 1800): motion 108.5-174.5, defocus 147.5-577 | No ≥ 100-frame warning before the 5-pt harm reference. On motion blur the early alarm does come first, by 16-64 frames. Read with the harm-reference caveat |
| X3-4 (ST-9) | SUPPORTED | **REFUTED** | S 4/10 (need 9) | Detectors with held-out ARL0 in [1000, 4000]: 14-19 of 19 per unit (need 18). All 20 out-of-band unit x detector cases (5 early-tap detector types: cusum_knnL2_s1end 7, cusum_knnL2_stem 6, mewma_stem 5, cusum_knnL2_s2end 1, mewma_s1end 1) are above 4000 (4037-9437); none is below 1000. Held-out false alarms per 1000 frames: stem CUSUM median 0.17, stem MEWMA 0.24 (nominal 0.5) | Early-tap thresholds over-protect on held-out rows: fewer false alarms than budgeted, and delays measured at those thresholds |
| X3-5 (PC-7) | SUPPORTED | **SUPPORTED** | S 10/10 (need 9) | Stuck frame, P(detect within 500): penult MEWMA 1.00, stem MEWMA 0.99-1.00, pixel repeat check 1.00. INFO: the head window tests ks_msp and BBSDh are also 1.00; head CUSUMs 0.07-0.37 | A frozen camera is caught from features alone. Head window tests catch it too, so this is not beyond the head |
| X3-6 (LH-4) | SUPPORTED | **REFUTED** | S 0/10 (need 7) | Benign brightness ramp: gated early-tap alarms 0.00-0.03 (the ≤ 0.2 part is met), but the ungated early detector fires in only 0.30-0.55 of streams (need ≥ 0.8) | The premise failed: early detectors do not reliably see the benign ramp, so the harm gate's HOLD-T value is not shown |
| X3-7 (ST-5) | SUPPORTED | **SUPPORTED** | S 10/10 (need 7) | Skew burst: BBSDh 1.00, penult MEWMA 1.00; X4 flag, skew minus null: −0.10 to +0.09 (need ≤ 0.10) | Label shift is caught by the head (BBSDh) and the penult MEWMA. The per-frame novelty flag does not mistake it for novelty |

### T1.3 Comparison with the discovery phase

Discovery labels are INFO and come from `results/b4/eval_discovery.json`. Discovery units: R := D (16 fit-layout units
including the hubs and e10-e70), M := Dnew (5 units), S := STdisc (3 units).

| outcome | claims |
|---|---|
| Same label in discovery and confirmation (27) | X1-1, X1-2, X1-3, X1-4, X1-5, PC-2a, PC-2b, X8-1, X7-1, X2-1, X4-1M, X4-2, X4-3, X4-6, X6-1, X6-2, X6-3, X6-4, X6-5, X3-8, X3-1, X3-2, X3-3, X3-4, X3-5, X3-6, X3-7 |
| Did not replicate (2) | **X4-1R**: discovery 16/16 SUPPORTED against the wider fit-layout band [0.0273, 0.0767] (n_cal 750); confirmation 3/4 against the narrower eval band [0.0327, 0.0693] (n_cal 2500), with resnet20_s31 at 0.0253. **X4-4**: discovery ADDS 12 / BOUNDED 4 SUPPORTED; confirmation ADDS 2 / BOUNDED 2 REFUTED. The noise-family X4 increment was a median +0.0192 on D and +0.0122 on R |
| No discovery label (2; they read confirmation-only splits) | X4-5 (REFUTED), X4-5x (SUPPORTED) |

- **Secondary patterns that replicated.** The architecture dependence carried over from Dnew to M (X1-4: ADDS 2 /
  BOUNDED 3, then ADDS 4 / BOUNDED 8; X4-4: ADDS 1 / BOUNDED 4, then 4 / 8; X4-6: ADDS 3 / BOUNDED 2, then 5 / 7; X4-3: 0/5,
  then 1/12). Clean BOUNDED results held on every scope (X1-2, X1-3, X8-1). X4 transfer held (X4-1M 5/5, then 12/12).
  Stream claims matched: X3-1, X3-2, X3-5 and X3-7 went 3/3, then 10/10; X3-3 0/3, then 0/10; X3-4 1/3, then 4/10; X3-6
  1/3, then 0/10.
- **Effect sizes** (functional medians, `functional` block; discovery D → confirmation R / R2):

  | quantity | discovery D | confirmation R | confirmation R2 |
  |---|---|---|---|
  | pre bundle, s3 errors | +0.0263 | +0.0367 | +0.0361 |
  | early bundle, s3 errors | +0.0271 | +0.0378 | +0.0347 |
  | all_geom, s3 errors | +0.0428 | +0.0549 | +0.0558 |
  | X4 on SVHN | +0.0402 | +0.0459 | +0.0444 |
  | X4 on blur presence | +0.0263 | +0.0390 | +0.0277 |
  | X4 on noise presence | +0.0192 | +0.0122 | +0.0128 |
  | X4 clean FPR | 0.0397 | 0.038 | 0.040 |

  The clean increments stayed within about ±0.001 (all_geom median −0.0002 on D, −0.0002 on R, −0.0009 on R2).
- **P1 predictions vs discovery.** All three unpredicted positives (X1-5, PC-2b, X7-1) were already SUPPORTED in
  discovery. Eight of the eleven unpredicted REFUTED labels were already REFUTED in discovery (X4-3, X6-1, X6-3, X6-4, X3-8,
  X3-3, X3-4, X3-6). The new negatives at confirmation are X4-1R, X4-4 and X4-5.

### T1.4 Rule 8 notes (numbers that look too good, too regular, or degenerate)

1. **PC-2a's +0.12 to +0.34 is mostly pixel information.** Beyond pixels it drops to +0.005 to +0.082, with the noise
   family just above the +0.005 bar on R (+0.0052 to +0.0064) and below it on one R2 unit (PC-2b).
2. **SVHN is separable from pixel factors by a trained classifier.** `pixel_only` (0.9968) and `head_pix` (0.9985-0.9990)
   are cross-fitted ridge-logistic models trained on the SVHN-vs-clean labels (`scripts/t1_scoreboard.py:887-889`), so
   they show that the information is in the pixels, not that a deployable unsupervised pixel flag works. No unsupervised
   pixel OOD flag was scored. X4 adds ≤ +0.0005 beyond head + pixels on every R and M unit; X4-6's YES is beyond the head,
   not beyond pixel factors. SVHN is also the only far-OOD set; on CIFAR-100 X4 is BOUNDED on R.
3. **X3-5 = 1.00 holds by construction:** a frozen frame freezes every EWMA. Head window tests (ks_msp, BBSDh) are also at
   1.00.
4. **X6-5 at 0.995-1.00 is matched by pixel T2 (0.96-1.00).** R_pre's own clean-batch FPR is 0.05-0.165, not 0.05.
5. **X3-1's 10x ratio sits next to pixels.** The pixel change information (median 0.663 nats) is about equal to stage 1
   (0.587), which fits early taps being close to pixels; a pixel MEWMA is comparable on delay (INFO).
6. **Both calibrated flags run conservative on held-out clean rows.**
   - X4's pooled FPR is 216/6000 = 0.036 (R), 483/12000 = 0.040 (R2), 777/18000 = 0.043 (M) and 992/24000 = 0.041 (D,
     discovery), against a nominal of about 0.05.
   - 20 unit x detector cases (5 early-tap detector types) have held-out ARL0 of 4037-9437 against a target of 2000.
   - Every unit uses the same calibration rows and the same B rows (global test indices), so these are not independent
     draws. Artifact hypothesis to test first: a shared row-sample effect, where the calibration rows are slightly more
     atypical than the B rows at the early taps. Not tested here. The direction is safe for false alarms and costs
     detection delay.
7. **X6-4's macro accuracy of about 1/3 on all 12 units is structural.** Both frozen cascades test covariate first
   (`scripts/t1_scoreboard.py:1237-1240`), and that first stage (T2_s2end for geometry, BBSDs for the head) also fires on
   label-skew and CIFAR-100 windows (Dirichlet 0.1 skew on resnet56_s31: T2_s2end 0.97, BBSDs 1.00), so almost every window
   is typed covariate (covariate 1.00, prior ≤ 0.015, novelty ≤ 0.05). The REFUTED is of the frozen cascade order, not
   evidence that prior shift cannot be seen: BBSDh fires at 1.00 on Dirichlet 0.1.
8. **The shift-time gain is smaller on the holdout corruption families (INFO, writer addition, corrected at final
   review).** Fitted and scored on the 5 holdout families (cross-fitted by image row, as X1-4 is on the 10 discovery
   families), the pre-bundle increment is +0.007 to +0.014 on R and +0.004 to +0.013 on R2, against +0.014 to +0.049 on
   the discovery families (X1-4). So the size of the gain depends on the corruption family. This is not a transfer
   measurement: `err_shift_holdout_s3` has no `transfer_from` in `scripts/t1_scoreboard.py`, and how a joint model
   fitted offline on synthetic corruptions does on a new kind of shift was not measured for errors.

### T1.5 Required caveats (RUN_REQUEST_S2.md "Step E")

- **X3-3, label REFUTED (0/10), still ACTIVE.** Harm-reference caveat from `experiments/b4/freeze_P2.json` "disclosed":
  - t_harm is the first frame at or after onset where the trailing 64-frame error exceeds the CAL clean error by 5 pt. An
    alarm is never before onset, so a unit's median lead is capped by median(t_harm − onset).
  - The cap is mostly sampling noise at an integer 5-pt threshold; the disclosure says the confirmation label "can turn
    more on each unit's clean error and on that integer threshold than on the detector".
  - With the disclosed method (stuck detector lead_median + 1800; the stuck detector raised no alarm on either ramp,
    P(detect) 0 on all 10 units), the confirmation caps are 108.5-174.5 frames on motion_blur and 147.5-577 on
    defocus_blur. So a SUPPORTED was reachable in principle on every unit; on motion_blur it needed a median early-tap
    alarm within 8.5-74.5 frames of onset. The observed best leads were 16-64.
  - The label rests mainly on motion_blur (0/10 reach 100); on defocus 4/10 do.
- **P4** is a T2 item, reported in the T2 section as "INFO (withdrawn at P2)". No T1 claim reads it.
- **SP-3 and SP-9** are T3S items, carried in the T3S section. No T1 claim reads them.
- **Other T1 disclosures in `freeze_P2.json` "disclosed", all before any confirmation read:**
  - X3-8 uses the frozen early list (stem, s1end, x4tap_stem; no s2end), the stricter reading.
  - X4-3's comparator is the per-split maximum TPR over 9 head statistics, which favours the head.
  - do3.nc1_train (verbatim pinv) is invalid at wide units; X4-2 is scoped to the 64-d R/R2 ResNets (nc1_train
    0.050-0.172).
- **B4a.** C = 12; the frozen README gate alone would have left 10 (mobilenetv2_x0_75 and shufflenetv2_x1_0 missing).
  `results/b4/b4a_seals.json` is PASS, and both T1 records show `meta_sha256_equal` true. No ShuffleNet timing value was
  raised. No T1 label depends on the two B4a units: removing them from `per_unit` leaves every M label and X4-1M unchanged
  (for example, X4-6 on M becomes ADDS 3 / BOUNDED 7, still REFUTED; X4-1M becomes 10/10). Eight of the 12 M units are
  README group B (mobilenetv2 x3, shufflenetv2 x3, repvgg x2); no T1 claim reads a README (see "B4a").
- **Host.** S2 ran on an AMD Ryzen 9 7950X (`results/instrument_check_b4s2/versions.json`), not the EPYC-Genoa class of
  RUN_REQUEST_S2's CPU check (owner decision; Genoa hosts were unavailable). Same stack (`versions_match.json` match true
  for python, numpy, scipy and sklearn; torch 2.8.0+cu128 in both S1 and S2 `versions.json`), quota 6.8, P = 5. The T1
  replay of the two 64-d anchors PASSed with drift 0 at rel 1e-6, so no host-related drift was measured on the replayed
  statistics. The cost block's milliseconds are host- and load-specific and are not covered by the replay (the replay
  ignores `timing_s` and `cost.timing_s`).
- **Timing** (`timing_s.total` of each output against `experiments/b4/timeouts.json` (B4b) and
  `experiments/b4/timeouts_b4_emitted.json`, keyed by the output's own directory):
  - 42 T1 confirmation jobs ran (32 scoreboards, 10 stream units), plus 2 T1 anchor replays.
  - Scoreboards used 0.094-0.264 of their B4b limits and 0.189-0.528 of the emitted ones. The largest share of a limit was
    resnet20_s1_eval (955 s: 0.528 of 1809 s emitted, 0.264 of 3618 s B4b). The longest in wall time was resnet56_s31_eval
    (1007 s: 0.474 of 2126 s emitted, 0.237 of 4252 s B4b).
  - Streams took 252-294 s: 0.140-0.164 of the emitted limits and 0.070-0.082 of B4b.
  - No kills and 0 soft failures.
- **Other.** verify_seals PASS (36 dumps); `unseal_log.jsonl` has 74 records. i.i.d. frame streams overstate rates
  relative to video (declared). The C checkpoints are best-on-test upstream; F / F20 are last-epoch, with a disclosed
  aggregate accuracy print (D13).

### T1.6 Inventory mapping (T1_SCOREBOARD.md section 10; written into `docs/knowledge/README.md` section 5)

Rule: a SUPPORTED bounded claim writes NO; a SUPPORTED adds claim writes YES; REFUTED writes the opposite; a
pre-registered positive that was not predicted is CANDIDATE. Statuses are additive: a batch-4 result is added beside a
row's existing status unless it tests the same thing. Every cell cites `results/b4/eval_t1.json`.

| row | T1 claims | "beyond the output head?" | status |
|---|---|---|---|
| CM-1 | X1-2, X1-3 | **NO** against the full head (BOUNDED on R 4/4, R2 8/8, M 11-12/12; increments below +0.01 except vgg16_bn's INCONCLUSIVE all_geom +0.0109) | ESTABLISHED (unchanged); beyond-head NO now pre-registered and confirmed |
| CM-2 | X1-2, X1-3 | **NO** against the full head (fresh resnet56 s31/s32 −0.0004 and −0.0000; resnet56 s1, s2, s12m, s13m BOUNDED). The earlier gain over maxprob is about the size of the best head statistic's own gain over MSP on resnet56 (+0.004 to +0.015, X1-1); that comparison is suggestive, not registered | CANDIDATE as a flag (unchanged); its beyond-head part is closed |
| CM-4 | X1-1 | **PARTIAL** (unchanged for ViTs). CIFAR: the statistic chosen on fit rows is the bar (X1-1 SUPPORTED on R 4/4, REFUTED on R2 5/7; MSP chosen on 0/4 R, 1/8 R2). Cross-reference X1-2: the margin is BOUNDED against the full head on 24/24 units (12 ResNet20/56 on R/R2, 12 C architectures) | CIFAR head-statistic rule ESTABLISHED on R, not on R2 |
| CM-9 | X1-5, X7-1 | **NO** on clean inputs (unchanged); **YES under shift** (CANDIDATE): early bundle beyond head and pixels +0.005 to +0.020 (R/R2); prediction depth +0.013 to +0.050 where it ADDS (R 3/4, R2 8/8) | ESTABLISHED (clean depth profile, unchanged); shift-time increment CANDIDATE (not predicted) |
| CM-11 | X1-3, X1-4, X8-1 | **YES** on corrupt-split errors (pre bundle ADDS on R 4/4, R2 8/8; +0.014 to +0.049), **NO** on clean (X1-3, X8-1). Not architecture-general (M 4/12) | UNTESTED → ESTABLISHED (CIFAR-10-C s3, ResNet20/56) |
| DO-1 | X4-3, X4-4, X4-5, X4-5x | **NO** by the registered rules (X4-3 against an oracle head comparator; X4-4 and X4-5 below their bars, increments positive on most ResNet units); **YES** for the 4 extras on R (X4-5x 3/4; M 5/12) | ESTABLISHED (unchanged); X4 head-increment claims REFUTED except X4-5x |
| DO-3 | X4-2 | N/A (calibration law); replicates on never-read rows of 12 units (8 models new to the law); the cross-architecture law is T2's M-DO3 | SUPPORTED-NOT-PROMOTED → ESTABLISHED (CIFAR-10 ResNet20/56, 64-d penult) |
| DO-4 | X4-6 | **YES** for SVHN on ResNet20/56 (R 4/4, R2 8/8, +0.025 to +0.085); **NO** across architectures (M 5/12) and on CIFAR-100 (R BOUNDED); not beyond a supervised pixel classifier (INFO) | MEASURED-NO-CLAIM → ESTABLISHED (X4 over the head for SVHN, ResNet20/56) |
| DO-6 | X4-1R, X4-1M, X4-3 | **NO** under the registered comparator (X4-3). INFO: against the frozen head.best_stat X4 wins by ≥ 0.05 on ≥ 6/10 splits on 2/4 R, 5/8 R2 | UNTESTED → REFUTED (X4-1R conservative side; X4-3); calibration transfer ESTABLISHED (X4-1M) |
| LH-1 | X6-1, X6-2 | **PARTIAL**: H beats ATC-MC on 12/12; the head's mean-gap / mean-energy estimator is as good or better on 8/12; no within-split gain (X6-2). INFO: penult displacement beats both on every resnet56 unit | SUPPORTED-NOT-PROMOTED (AH-2, unchanged); X6-1's reverse result CANDIDATE |
| LH-2 | X6-1, X6-5 | **YES** for mild covariate shift at batch 64: R_pre flags batches that head BBSDs (0.345-0.86) and BBSDh (≤ 0.11) largely miss, but pixel T2 flags them too (not beyond pixels); h's grading **PARTIAL**, as LH-1 | SUPPORTED-NOT-PROMOTED (AX-3, unchanged); R_pre batch covariate-shift flag ESTABLISHED (X6-5) |
| LH-4 | X3-6 | **UNTESTED** (X3-6's premise failed; no head comparison was made) | stream HOLD-T gate REFUTED (X3-6); harm grading unchanged |
| ST-1 | PC-2a, X6-4 | **YES** per sample (+0.12 to +0.34, PC-2a), mostly pixel-level (PC-2b); **NO** for batch-window typing (X6-4, degenerate cascade) | SUPPORTED-NOT-PROMOTED (AX-2b, unchanged); per-sample family identification ESTABLISHED; window typing REFUTED |
| ST-3 (successor R) | X6-3 | **NO**: R_pre and H are not quiet under single-class skew; label shift is a head signal | RULED-OUT → REFUTED (successor R) |
| ST-4 (successor X2) | X2-1 | **NO** by the registered rule (≥ +0.03 in ≥ 2 families); on noise +0.008 to +0.036 (R), three R/R2 units above +0.03 | RULED-OUT → REFUTED (successor X2) |
| ST-5 | X3-7, X6-4 | **NO** (a head signal): BBSDh 1.00 on the skew burst, the penult MEWMA matches it; X4 stays at its null rate. Prior-shift typing fails for both rules (X6-4) | UNTESTED → ESTABLISHED (skew-burst scenario, 10 STconf units) |
| ST-9 | X3-1 … X3-4 | **YES** for detection delay and change information (X3-1, X3-2), not beyond pixels (INFO); **NO** ≥ 100-frame harm warning (X3-3); ARL0 transport conservative (X3-4) | UNTESTED → ESTABLISHED (X3-1, X3-2); REFUTED (X3-3, X3-4) |
| PC-2 | PC-2a, PC-2b | **YES** beyond the head (PC-2a) and a little beyond pixels (PC-2b) | MEASURED-NO-CLAIM → ESTABLISHED (PC-2a); CANDIDATE (PC-2b, not predicted) |
| PC-7 (T1 part) | X3-5, X3-8 | **NO**: per-frame early-tap fault flags beat the head by ≥ 0.10 only under global exposure (X3-8 REFUTED 0/4, 0/8, 0/12); a stuck frame is caught by features (X3-5) and equally by head window tests | stuck frame ESTABLISHED (X3-5); per-frame fault flag REFUTED (X3-8). Lane S: see T3S |

### T1.7 Sources

| what | where |
|---|---|
| Labels, gates, units, `per_unit` calls, functional medians, cost, false alarms per 1000 frames | `results/b4/eval_t1.json` |
| Per-unit joint-model numbers and CIs; X4 FPR and Wilson CI; DO-3; batch estimators; typing; skew; faults | `results/b4_t1/<unit>_eval/scoreboard.json` (R, R2), `results/b4_t1/<unit>_fit/scoreboard.json` (M) |
| Stream rates, delays, leads, ARL0, scenario detection rates | `results/b4_t1s/<unit>/streams.json` |
| Discovery labels and medians | `results/b4/eval_discovery.json` |
| Disclosures, B4a | `experiments/b4/freeze_P2.json`, `results/b4/b4a_seals.json` |
| Host and timing | `results/instrument_check_b4s2/{versions.json, versions_match.json, replay.json, head.txt}`, `experiments/b4/timeouts.json`, `experiments/b4/timeouts_b4_emitted.json` |

---

## Track T2: the collapse arm (the owner's PRIMARY hypothesis, D12; `docs/plans/T2_COLLAPSE.md`)

**Source.** `results/b4/t2_eval.json` (schema `b4_t2_eval/1`, created 2026-09-24T18:50:48.974Z, sha256
`5385374526a33eabf913a748bae2df1a8d2ab870ca49e664b52d4537b9a6c063`), written by one run of
`node scripts/collapse_laws.js --evaluate --p1 ad97d0a6… --p2 272bc41b… --p-run ad97d0a6…,14caf51b…,272bc41b… --replay results/instrument_check_b4s2/replay.json --json results/b4/t2_eval.json`,
which exited 0.

**Gates.** Every gate passed:
- **Guard.** `guard: null` and `provenance.official: true`.
- **Code hashes.** The probe, core and collapse hashes, and the T1 scoreboard hash behind O1 and O5, each take one value
  across the fitted and confirmation records, and each equals P2.
- **Frozen files.** `laws_read = laws_at_p2` (b959c386…) and `evaluator_now = evaluator_at_p2` (ed19775b…).
  `fitted_files` n = 21 with none bad.
- **Read discipline.** `probes_outside_p_run = []` and `touched_twice = []`. The t2/t1 duplicate and off-phase fields are
  null for all 20 confirmation units.
- **Replay.** PASS: `t2:`, `t1:` and `t3s:` on both anchors have n_diff 0 at rel 1e-6 / abs 1e-9
  (`results/instrument_check_b4s2/replay.json`). No label carries REPLAY-DRIFT.
- **D15 nc1 gate at the fit.** PASS, n = 16, max rel 6.44e-6.
- **Self-test.** `collapse_laws.js --selftest` PASS (20 checks; read-only).
- **Units.** All 20 confirmation units are evaluable: 12 C, 2 F, 2 F20 and 4 K. No outcome and no secondary is
  NOT_EVALUABLE.

**Scope.** The 12 C units are sealed CIFAR-10 hub checkpoints from one upstream recipe. Every one belongs to one of the
five families already in the 21 discovery nets (D21 has one Dnew member per family: resnet32, vgg11_bn, mobilenetv2_x0_5,
shufflenetv2_x0_5, repvgg_a0). The confirmation therefore tests transfer to new members of seen families, not to new
families, on one dataset.

### T2.1 Decision

**PRIMARY: REFUTED (0 WIN, 5 LOSS of 5 evaluable outcomes).** The P1 prediction was MIXED (O2 and O3 WIN, O1 LOSS, O4
and O5 NEITHER). The result is worse than predicted: no outcome wins, and every outcome is a LOSS.

On the 12 sealed architectures, accuracy plus the best label-free collapse coordinate (frozen at P2) predicted each of
the five detector outcomes *worse* than the best of three rivals: a constant; accuracy alone; accuracy plus the best head
coordinate. The labelled twin (S-PL) also lost on 5 of 5, so the failure is not explained by the label-free proxy alone
(P8 also shows that the proxy does not track labelled test NC1).

In plain terms: in this setting, a backbone's collapse level, read from unlabelled clean frames, does not tell a
controller more about how its detectors will behave than the backbone's accuracy already does.

### T2.2 Headline answers

**PRIMARY per outcome** (law `acc_coll`; n = 12 C units in every outcome). Sources: `primary.per_outcome.*` in
`results/b4/t2_eval.json`; the coordinates from `experiments/b4/laws_frozen.json` `primary.*.laws.acc_coll.covs`.

| outcome (MAE unit) | collapse coordinate (frozen on D21) | MAE acc_coll | best rival (MAE) | MAE ratio | rho(pred, obs) | family-demeaned rho | call | rule 8 (partial on log width / on accuracy) | EXTRAP |
|---|---|---|---|---|---|---|---|---|---|
| O1: penult dAUC_joint over the head, clean errors (AUROC) | log10 g_cv | 0.00237 | acc_head (0.00181) | 1.311 | 0.000 | +0.204 | **LOSS** | FAMILY-DRIVEN, DIMENSION (0.049 / −0.259) | 0 |
| O2: H10, the HOLD-band edge (H units) | topk_frac | 0.0578 | acc (0.0366) | 1.580 | 0.385 | +0.397 | **LOSS** | DIMENSION (0.361 / 0.241) | 0 |
| O3: CIFAR-100 L2 10-NN minus best head (AUROC) | log10 plnc1 | 0.00929 | acc (0.00876) | 1.060 | 0.056 | +0.613 | **LOSS** | DIMENSION (0.071 / −0.287) | 1 (vgg16_bn) |
| O4: mean MSP minus accuracy under shift (probability) | log10 plnc1 | 0.0140 | const (0.0107) | 1.307 | 0.119 | −0.257 | **LOSS** | FAMILY-DRIVEN, DIMENSION (0.223 / 0.112) | 1 (vgg16_bn) |
| O5: X4 flag minus best head at s3 (AUROC) | topk_frac | 0.0165 | acc (0.0101) | 1.638 | 0.063 | −0.471 | **LOSS** | FAMILY-DRIVEN, DIMENSION (0.064 / 0.189) | 0 |

- **Verdict rule.** SUPPORTED needs at least 3 WIN; REFUTED means at most 1 WIN and at least 3 LOSS. 0 WIN and 5 LOSS
  gives REFUTED.
- **Rule-8 annotations** never change the PRIMARY verdict. DIMENSION is on all 5 outcomes and FAMILY-DRIVEN on O1, O4 and
  O5. No outcome has |rho| ≥ 0.95 (no RULE8-CHECK), and none is EXTRAP-dominated (at most 1 of 12 units).
- **Where collapse did beat the head law.** Against the accuracy + head law alone, `acc_coll` has lower MAE on O2 (ratio
  0.932), O3 (0.850) and O5 (0.678). On those three outcomes accuracy alone was the stronger rival; the head coordinate
  added error rather than information.
- **What orders the outcomes on C** (a reading computed from `info.units`, not an evaluator field). No frozen law predicts
  the C outcomes much better than the C units' own mean, an in-sample reference no frozen law has (for example O2: accuracy
  law MAE 0.0366 against a mean absolute deviation of 0.0356). Single coordinates still rank O2 (log10 plnc1 rho −0.71,
  cf. P5b) and O5 (err rho −0.67); O1, O3 and O4 are not ordered by any coordinate (|rho| ≤ 0.52). Part of the constant's
  error on O2 and O3 is a level offset between D21 and C (the constant 0.326 against a C mean of 0.403 on O2; 0.0111
  against 0.0237 on O3). C's accuracy range is narrow (err 0.049-0.075) compared with D21's (0.055-0.162).

**Secondaries** (Holm, alpha 0.05, over the m = 22 items with a p-value; 25 items). Counts: **7 SUPPORTED, 3 RULE-MET,
HOLM-NS, 14 REFUTED and 1 INFO** (P4, withdrawn at P2; the evaluator labels it REFUTED and keeps it in Holm, m = 22);
0 NOT_EVALUABLE.
- **Final SUPPORTED (7):** with Holm, M-DO3, MP1, P2b, P5a and P7; bound rules without a p-value, P5c and IC-P2.
- **Holm-significant but rule not met:** M-LF (Holm 0.022) and P6b (Holm 0.0030) are REFUTED on their rules.
- **RULE-MET, HOLM-NS:** MP6, FE3 and P5b.
- **Rule-8 flags on items:** MP3 DIMENSION; MP5, MP7, P4 and S7 FAMILY-DRIVEN, DIMENSION; M-NC4 DIMENSION. No supported or
  rule-met item carries a flag, so none is downgraded to PARTIAL.

**Answer to the owner's question** (which controller-usable information collapse geometry gives beyond the output head,
and how reliably), at model level, on CIFAR-10 hubs:
1. **Not the five detector outcomes.** Label-free collapse does not predict O1-O5 beyond accuracy or the head (PRIMARY,
   S-PL and FE9 all REFUTED).
2. **The harm slope is the one exception, and a candidate.** Pseudo-label nc1 predicts how many points of accuracy a
   backbone loses per unit of the label-free harm grade H: MAE 3.19 pt/H against 12.70 for the constant (P5a SUPPORTED,
   Holm 0.003). Reported beside it as INFO (the registered rule compares only with the constant): 7.67 for accuracy alone
   and 12.67 for the head alone (ratios 0.42 and 0.25; single-coordinate rivals, not the PRIMARY's accuracy + head), and a
   partial on accuracy of 0.893. It is the only supported label-free model-level law in T2 that runs ahead of accuracy
   and the head (P6b's rank law also beats both by MAE but fails its constant clause), and it rests on one batch.
3. **The DO-3 over-alarm law transfers; its label-free form adds nothing to the head.** Labelled collapse predicts a
   train-referenced density threshold's clean over-alarm on 10 of 12 architectures (M-DO3; MP1, P2b). The head's
   saturation (sat999) ranks that over-alarm at least as well as the frozen label-free coordinate (M-LF REFUTED). Whether
   labelled nc1 beats the head was not registered; on the raw numbers it is about a tie (|rho| 0.846 for nc1_train, 0.909
   for the train-test gap, 0.839 for sat999). Held-out conformal calibration removes the over-alarm on 12 of 12 (IC-P2). So
   the controller action stays "calibrate on held-out clean data"; the law is the fallback where no such data exist.
4. **The HOLD band holds; within-family collapse ordering holds, but does not order the over-alarm.** The committed band
   (H ≤ 0.25 → cost ≤ 8 pt) has no violation on all 12 (P5c), with C's H10 at 0.318-0.490, well above the 0.25 edge. Within
   each of 5 families, deeper or wider nets are more collapsed (P7), but within families collapse does not order the
   over-alarm (FE1 REFUTED, rho_fam −0.419).

### T2.3 Claim table

Sources: `secondary.items.<id>` and `primary` in `results/b4/t2_eval.json`; predictions from `docs/plans/T2_COLLAPSE.md`
section 7. The evaluator emits no CI for the Spearman or MAE statistics; the permutation p and the Holm p are its
uncertainty measures. The CIs quoted are the ones the JSON carries (P3b, IC-P2, P7).

| claim | prediction | label | units passing / needed | key numbers (with CIs) | what it means for the controller |
|---|---|---|---|---|---|
| **PRIMARY** (acc + label-free collapse) | MIXED | **REFUTED** | 0 of 5 outcomes WIN (need ≥ 3); 5 LOSS; n = 12 per outcome (need ≥ 10) | MAE ratio to the best rival O1-O5: 1.311 / 1.580 / 1.060 / 1.307 / 1.638; rho 0.000 / 0.385 / 0.056 / 0.119 / 0.063 | Do not set the O1-O5 quantities (HOLD edge, novelty-detector choice, MSP discount, whether to run penult geometry or X4) from a label-free collapse reading. Accuracy (with the head coordinate on O1) or a constant (O4) is the better prior |
| S-PL (labelled nc1_train twin) | MIXED | REFUTED (p 0.025, Holm 0.325) | 0 of 5 WIN, 5 LOSS | ratio 1.072 / 1.320 / 1.061 / 1.451 / 2.712 | The labelled coordinate fails too, so the gap is not in the proxy alone |
| M-DO3 (AH-1 law transfers) | REFUTED (< 9/12) | **SUPPORTED** (p 2.7e-3, Holm 0.043) | 10/12 inside ±0.035 (need 9) | vgg16_bn and vgg19_bn outside (obs − law −0.0525 and −0.0508). Rule-5 curve at bands 0.020-0.050: 7/8/9/10/10/10/10. MAE 20.6 false alarms per 1000 frames. Clean train-referenced FPR on C 0.146-0.210 (`results/b4_t2/<id>/probe.json` `targets.do3.fpr_trainref`) | Predicts the over-alarm of a train-referenced threshold on new backbones, except deep VGG. The action stays: never deploy a train-referenced threshold (F1) |
| M-LF (label-free coordinate ranks the over-alarm beyond head and accuracy) | SUPPORTED | REFUTED (p 1.3e-3, Holm 0.022) | n 12; \|rho\| ≥ 0.7 met; beats every head coordinate: not met | log10 plnc1 rho −0.797. Rivals: sat999 +0.839, log10 msp_def −0.804, log10 gap_mean +0.629, err −0.309 | For ranking backbones by over-alarm, the head's saturation is at least as informative: **no gain beyond the head** for the label-free form |
| MP1 | SUPPORTED | **SUPPORTED** (p 2.5e-4, Holm 4.5e-3) | n 12; rho ≤ −0.6 | rho −0.846; rho_fam −0.419; partial on width −0.865, on accuracy −0.905 | More collapsed backbones over-alarm more (labelled coordinate, beyond accuracy), partly through family offsets (FE1) |
| MP3 | SUPPORTED | REFUTED [DIMENSION] (p 0.082, Holm 0.740) | rho ≥ +0.6 | rho +0.427; rho_fam +0.734; partial on width 0.425 | Collapse does not rank, across families, when the train-centre margin beats plain distance |
| MP5 | SUPPORTED | REFUTED [FAMILY-DRIVEN, DIMENSION] (p 0.592, Holm 1) | rho ≤ −0.5 | rho +0.070; rho_fam −0.496; partial on accuracy +0.617 | Collapse does not say when the logit gap beats MSP; choose the head statistic empirically |
| MP6 | REFUTED | RULE-MET, HOLM-NS (p 0.0186, Holm 0.260) | rho ≤ −0.5 | rho −0.622; rho_fam −0.594; partial on width −0.669, on accuracy −0.398 | Suggestive, in Harun et al.'s direction: more collapse goes with a better SVHN kNN AUROC. Not established; the weaker partial on accuracy leaves accuracy as a possible driver (artifact hypothesis 3) |
| MP7 | REFUTED | REFUTED [FAMILY-DRIVEN, DIMENSION] (p 0.254, Holm 1) | rho ≥ +0.5 | rho +0.217; rho_fam −0.179 | The train-test collapse gap does not predict clean ECE |
| P2b (Hui et al. mechanism) | SUPPORTED | **SUPPORTED** (p 5.0e-5, Holm 1.1e-3) | rho ≥ +0.6 | rho +0.909; rho_fam +0.577; partial on width +0.906, on accuracy +0.945 | Explains the over-alarm: the train-test collapse gap orders it. Needs labels |
| M-NC4 | REFUTED | REFUTED [DIMENSION] (p 0.747, Holm 1) | rho ≥ +0.5 | rho −0.207 | NC4 disagreement does not predict geometry's clean-error gain (O1) |
| FE1 (within-family MP1) | SUPPORTED (Holm likely NS) | REFUTED (p 0.175, Holm 1) | rho_fam ≥ 0.6 | rho_fam −0.419 | MP1's ordering is partly a family offset; within a family, collapse does not order the over-alarm |
| FE3 (within-family MP3) | SUPPORTED (Holm likely NS) | RULE-MET, HOLM-NS (p 0.0353, Holm 0.424) | rho_fam ≥ 0.6 | rho_fam +0.734 | Within a family, collapse may rank the margin-over-distance lead. Not established |
| FE9 (within-family PRIMARY) | REFUTED | REFUTED (p 0.318, Holm 1) | 1 of 5 outcomes (need 3) | rho_fam O1-O5: 0.204 / 0.397 / 0.613 / −0.257 / −0.471 | No within-family law for O1-O5 either |
| P3a | SUPPORTED | REFUTED (p 0.057, Holm 0.601) | rho ≥ 0.6 and MAE ≤ 0.8 × const: not met | log10 g_cv. rho 0.490; rho_fam 0.524. MAE 0.0306 vs const 0.0218 (1.40), acc 0.0090 (3.40), head 0.0172 (1.78) | Accuracy predicts the margin lead far better |
| P4 | REFUTED | **INFO (withdrawn at P2)**; evaluator label REFUTED [FAMILY-DRIVEN, DIMENSION] (p 0.0546, Holm 0.601) | n/a (direction not as registered at P2) | log10 plnc1. rho 0.490; rho_fam −0.104. MAE ratio: const 0.821, acc 0.859, head 1.003 | No usable law for the SVHN kNN-minus-head gain |
| P5a (harm slope) | SUPPORTED | **SUPPORTED** (p 1.5e-4, Holm 3.0e-3) | all four clauses met (rho ≥ 0.6, rho_fam > 0, MAE ≤ 0.8 × const, direction) | log10 plnc1. rho 0.902; rho_fam 0.785. MAE 3.19 pt/H vs const 12.70. INFO (not part of the rule): acc 7.67 (ratio 0.416), head 12.67 (0.252), partial on width 0.914 and on accuracy 0.893; EXTRAP 1; not noise-limited (SD 7.19 vs split-half SE 0.33). Slope on C 25.5-50.9 pt/H (`probe.json` `targets.harm.slope`) | **The one supported label-free model-level law that runs ahead of accuracy and the head** (its lead over them is INFO; P6b also leads them by MAE but is REFUTED on its constant clause). From unlabelled clean frames, estimate a backbone's accuracy loss per unit of H, the exchange rate from the H flag to expected loss (F2). A candidate: one batch |
| P5b (H10) | SUPPORTED | RULE-MET, HOLM-NS (p 5.55e-3, Holm 0.083) | rule met | log10 plnc1. rho 0.706; rho_fam 0.673. MAE 0.0421 vs const 0.0778 (0.541), acc 0.0366 (1.151), head 0.0422 (0.997) | The HOLD edge is not predicted beyond accuracy |
| P6b (best non-duplicate tap minus head, per-sample corruption) | REFUTED | REFUTED (p 1.5e-4, Holm 3.0e-3; fails the MAE clause only) | rho, rho_fam and direction met; MAE 0.830 × const (bar 0.8) | cdepth_pl. rho 0.895; rho_fam 0.483. MAE 0.0185 vs const 0.0223, acc 0.0287 (0.647), head 0.0298 (0.621). Best tap is intermediate on 10/12 (layer1.6, features.6, features.34/52, stage2), the penult on repvgg_a1 and classifier.1 on vgg16_bn | A strong rank signal for how much a feature tap beats the head on shift (F1), but not a law. A candidate for a later batch |
| S7 (O4 overconfidence) | SUPPORTED | REFUTED [FAMILY-DRIVEN, DIMENSION] (p 0.654, Holm 1) | not met | topk_frac. rho −0.126. MAE 0.0162, 1.51 × const | Collapse does not say how much to discount MSP under shift |
| P3b (margin vs head at a collapsed penult) | SUPPORTED | REFUTED (bound rule) | 3/7 collapsed units within d ≤ +0.005 (need all 7); 3 have a 99% CI lower bound > 0 | theta 0.10945. mobilenetv2_x1_4 d +0.0064 [0.00001, 0.0152] (borderline: plnc1 0.10938, just under theta); repvgg_a1 +0.0059 [0.0009, 0.0127]; repvgg_a2 +0.0215 [0.0070, 0.0393] | At a collapsed penult the univariate train-centre margin beats the head statistic picked on rows A by up to about 2 AUROC points on 3 units. Against the full head (T1 X1-2) the margin is BOUNDED on all 12 C units, so this is a better single statistic, not information the head lacks |
| P5c (committed HOLD band) | SUPPORTED | **SUPPORTED** (bound rule) | 12/12 without violation over the 30 discovery splits (need 10) | Holdout corruptions and extras: 0 violations on all 12 (INFO). C's H10 0.318-0.490 | F2: H ≤ 0.25 → cost ≤ 8 pt is safe on all 12 new architectures, with a wide margin |
| IC-P2 (held-out conformal FPR) | SUPPORTED | **SUPPORTED** (instrument check) | 12/12 inside (need 11) | Realised FPR at alpha 0.05: mean 0.052, range 0.034-0.0593, inside the exact 99% band [0.0313, 0.0720]. The rule-5 curve's means (0.0200 / 0.0300 / 0.0400 / 0.0500 at alpha 0.02-0.05) are the bands' expected values, not observed rates | Instrument check passed: held-out conformal calibration on clean rows of the same split gives the nominal rate on all 12 backbones. Expected under exchangeability ("a failure means a bug", T2_COLLAPSE section 7); not a test under deployment shift |
| P7 (blind within-family ordering) | SUPPORTED; resnet32 and resnet44 inside | **SUPPORTED** (p 1.0e-4, Holm 2.1e-3) | 5/5 families (need 4) | tau −1 in each (n 4/4/4/4/3). resnet32 obs 0.0871, in [0.0819, 0.1533]; resnet44 obs 0.0601, in [0.0540, 0.1011] | Within a family, deeper or wider nets are more collapsed (labelled nc1_train). Whether that orders a sibling's over-alarm was tested by FE1 and not supported (within VGG the train-referenced FPR is flat: 0.210 / 0.199 / 0.205) |
| P8 (label-free tracks labelled) | SUPPORTED | REFUTED (p 1, Holm 1) | 0/12 within 0.1 dex (need 10) | rho(log10 g_cv, log10 nc1_te) −0.168; p_a 1, p_b 0.305 | Label-free coordinates are not a stand-in for labelled test NC1 |
| MP-K (knob pairs) | REFUTED | REFUTED (p 0.377, Holm 1) | 6/10 cells (need 8) | Both ls10 pairs win on O3, O4 and O5 and lose on O1 and O2. The wd5e5 pairs are INFO (Δacc 2.02 and 2.42 pt) | The cross-architecture law does not predict a training knob's effect well enough |

**INFO.** MP4: rho(log10 nc1_train, O1) −0.287, n 12. Fresh-seed predictions F and F20: `info.fresh_seeds_F_F20`.
Random-init nulls (`info.nulls_N`): resnet20_rand plnc1 1.769, g_cv 1.104, nc1_tr 72.5; resnet56_rand plnc1 0.0120, g_cv
1.054, nc1_tr 151.4. Reading, not a registered claim: resnet56_rand's pseudo-label nc1 (0.012) is lower than any trained C
unit's (0.0445 at vgg16_bn), so plnc1 alone cannot tell a random-init net from a collapsed one.

### T2.4 Discovery (D21, frozen at P2) against confirmation (C12)

Law selections were made on the same 21 nets (in-sample), and the D21 LORO MAEs are cross-validated. The MP and P2b
previews use fixed, registered coordinates. Discovery correlation rows are INFO previews with no verdict; only P3b had a
discovery rule. Sources: `experiments/b4/laws_frozen.json`, `experiments/b4/freeze_P2.json` "disclosed".

| item | discovery | confirmation | replicated? |
|---|---|---|---|
| PRIMARY | The collapse law had the lowest LORO MAE on O2-O5 (ratios 0.77 / 0.90 / 0.85 / 0.89; O1 1.04), and only O2 cleared the 0.8 WIN bar (topk_frac law LORO MAE 0.0233 vs 0.0304 head, 0.0414 accuracy, 0.0655 constant) | O2 ratio 1.580 (accuracy best); all 5 LOSS | **No.** The one discovery win reversed |
| M-LF (log10 plnc1 vs DO-3 FPR) | rho −0.920; g_cv 0.620; topk 0.420 | rho −0.797, beaten by sat999 (+0.839) | Sign yes; beyond the head no |
| MP1 | −0.968 | −0.846 | **Yes** |
| P2b | +0.973 | +0.909 | **Yes** |
| P5a | rho_D 0.918 | 0.902 (rule met; INFO lead over accuracy and head) | **Yes** |
| P5b | 0.955 | 0.706; not beyond accuracy | Weakened |
| P6b | 0.713 | 0.895, but MAE 0.83 × const | Rank yes; law no |
| MP3 | +0.895 | +0.427 | No |
| MP5 | −0.945 | +0.070 | No |
| P3a | 0.853 | 0.490 | No |
| S7 | 0.445 | −0.126 | No |
| MP6 | +0.013 (against the registered direction) | −0.622 (Holm-NS) | New on C only |
| MP7 | +0.156 (below the bar on the preview, INFO) | +0.217, REFUTED | Below the bar both times |
| M-NC4 | −0.498 (against the registered direction; INFO) | −0.207, REFUTED | Below the bar both times |
| P4 | 0.448, direction flipped (withdrawn) | 0.490, rho_fam −0.104 | n/a (INFO) |
| P3b | REFUTED on D: 3 of 5 have CI lo > 0 (resnet56_hub, resnet56_s2, repvgg_a0) | REFUTED: 3 of 7 (mobilenetv2_x1_4, repvgg_a1, repvgg_a2) | **Yes** |
| P7 INFO points | registered at P1 | both inside their intervals | **Yes** |

### T2.5 Required caveats (RUN_REQUEST_S2.md "Step E")

- **P4 is INFO (withdrawn at P2).** Reason, from `freeze_P2.json` "amendments" (D8 (b), T2_COLLAPSE section 5): the frozen
  law chose log10_plnc1 on D21 with slope 0.0489 and rho_D 0.448, so the gain *falls* as collapse deepens and
  `direction_as_registered = false`; P4 could not be SUPPORTED at E. P1 had predicted REFUTED (the SVHN ceiling). The
  evaluator's label beside it is REFUTED [FAMILY-DRIVEN, DIMENSION] (p 0.0546, Holm 0.601); `collapse_laws.js` still
  counts P4 in Holm over 22 items, which is conservative for every other item.
- **X3-3 and SP-3 / SP-9 are not T2 claims**; no T2 input reads them. They are carried in the T1 and T3S sections. T2's
  harm target (median log r10 against paired accuracy loss) does not use the stream t_harm reference.
- **B4a.** All 12 C units are evaluable in every PRIMARY outcome and in every secondary that reads C (n = 12; P3b screens
  all 12 and finds 7 collapsed; MP-K reads the F / K knob pairs, no C unit; P7's family series mix D and C members).
  D21 includes shufflenetv2_x0_5 via B4a (P_A, tag `_r2`). Under the frozen gate alone C would have held 10
  (mobilenetv2_x0_75 and shufflenetv2_x1_0 missing; `freeze_P2.json` `b4a.c_count_under_frozen_gate_only`).
  `results/b4/b4a_seals.json` is PASS: seal_diff PASS; both units PASS, each covered by
  `results/instrument_check_b4s2/verify_seals.json` with `s2_guard: true`; both T2 probe records are confirmation-phase with
  `meta_sha256_equal: true`. No ShuffleNet timing value was raised (`experiments/b4/timing_b4a.json` `b4a.adjusted.t2`:
  measured 71.527 s against a floor of 65.268 s, `raised: false`; `freeze_P2.json` `b4a.timing_raised = []`).
  `verify_seals.json`: PASS, 36 entries, 0 not PASS.
- **Accuracy provenance.** The README-gate INFO (group-B README accuracy = the maximum of 200 augmented evaluations; see
  "B4a") does not enter T2. The accuracy rival is the probe's own err_te on test rows 0-4999. Hub checkpoints were selected
  upstream on the test set, which is why accuracy is a rival and never a collapse coordinate (T2_COLLAPSE section 1).
- **Host.** S2 ran on 1x RTX 4090 with an AMD Ryzen 9 7950X (`results/instrument_check_b4s2/versions.json` `cpu_model`),
  not S1's EPYC-Genoa, by owner decision; CPU quota 6.8, P = 5. T2 is CPU numpy on one BLAS thread (`probe.json`
  `env.threads`). On the two 64-d anchors the t2 replays agree with S1 within rel 1e-6 / abs 1e-9 (n_diff 0). No
  wide-penult unit (512-2048-d) was replayed across hosts: the laws were fitted on S1's Genoa host (including the wide Dnew
  units), and the wide C units ran only on the Ryzen.
- **Timing.** The 20 T2 confirmation jobs took 83.4 to 206.3 s (`results/b4_t2/<id>/probe.json` `timing_s.total`), with
  max RSS 301 to 894 MB. The longest, shufflenetv2_x2_0 at 206.3 s, used 0.057 of its B4b limit (3600 s,
  `experiments/b4/timeouts.json`) and 0.115 of the emitted limit (1800 s, `experiments/b4/timeouts_b4_emitted.json`). The
  anchor replays took 81.3 and 81.8 s. No T2 job was killed or relaunched: the chosen directories in `info.units[].sources`
  carry no `_r<k>` suffix, and `probe.json` `tag` is '' for all 20.

### T2.6 Inventory mapping (`docs/knowledge/README.md`; D19; T2_COLLAPSE section 9: nothing from T2 alone is promoted to ATLAS_STATUS)

| row | status now → proposed | new "beyond the output head?" cell | cites |
|---|---|---|---|
| **CD-12** (model-level collapse laws) | UNTESTED → **REFUTED** (batch 4 PRIMARY; CIFAR-10, 12 sealed hub architectures of families seen in D21, one upstream recipe; not promoted); P5a SUPPORTED (candidate) | **NO.** Accuracy plus the best label-free collapse coordinate lost to the best of constant, accuracy and accuracy + head on all 5 outcomes (0 WIN, 5 LOSS, MAE ratio 1.06-1.64, rho 0.00-0.39). The labelled twin is also 0 of 5, within-family 1 of 5, knob pairs 6 of 10. Exception: pseudo-label nc1 predicts the harm slope (P5a SUPPORTED on one batch; its lead over accuracy (0.42) and the head (0.25) is INFO). H10 is not predicted beyond accuracy (1.15, Holm-NS). Readiness: "not built" → "do not use as a configuration input for O1-O5; harm slope a candidate" | PRIMARY, S-PL, FE9, MP-K, P5a, P5b |
| **CD-4** (collapse level) | Unchanged: nc1 ESTABLISHED; sep / bridge CANDIDATE; training-length law REFUTED. Add: cross-architecture DO-3 driver SUPPORTED (MP1, P2b; M-DO3 10/12), within-family ordering SUPPORTED (P7) | Beyond head accuracy: yes (MP1 partial on accuracy −0.905; P2b +0.909). Against the head: the label-free form is not beyond it (M-LF REFUTED: plnc1 −0.797 vs sat999 +0.839); labelled nc1_train (\|rho\| 0.846) and the gap (0.909) against sat999 (0.839) were not compared by any registered item: **UNTESTED** (about a tie). The collapse gap does not predict clean ECE (MP7). Collapse adds nothing over the head for O1-O5 (PRIMARY, S-PL) | M-DO3, MP1, P2b, P7, FE1, M-LF, MP7, PRIMARY, S-PL |
| **CD-9** (collapse as a hidden coordinate) | CANDIDATE (hypothesis) → **CANDIDATE, narrowed**: REFUTED as a predictor of model differences in detector behaviour; kept as the DO-3 mechanism | **NO** as a hidden coordinate for detector behaviour beyond accuracy or the head (PRIMARY 0/5, FE9 1/5, M-LF). Label-free proxies do not track labelled test NC1 (P8: 0/12 within 0.1 dex; rho(g_cv, nc1_te) −0.168). Only the harm slope is predicted (P5a). The readiness cell's X9 +0.937 (discovery) gains a confirmation line: on C, the D21-chosen label-free coordinate (pseudo-label nc1, not X9's gap) ranks the over-alarm at −0.797, no better than the head's sat999 (+0.839) | PRIMARY, FE9, M-LF, P8, P5a, M-DO3, P2b |
| **DO-3** (train-referenced over-alarm law) | T1 X4-2 makes the ResNet20/56 form ESTABLISHED (see T1.6); the cross-architecture form is CANDIDATE (M-DO3 10/12, predicted REFUTED; not promoted by T2 alone) | N/A (calibration law). Its label-free form is not beyond the head (M-LF REFUTED). Held-out conformal calibration gives 0.034-0.059 at alpha 0.05 on 12/12 (IC-P2, an instrument check). Action: calibrate on held-out clean data; the law is the fallback where none exist | M-DO3, IC-P2, M-LF, MP1, P2b |

Other rows touched, INFO only (owned by T1):
- **DO-4.** On CIFAR-100, L2 10-NN beats the best head statistic on 12 of 12 C units by 0.0068 to 0.0320 AUROC; the head
  bar was picked on the same rows, which favours the head (`info.units[].y.O3`; no CI emitted).
- **CM-2.** P3b gives +0.0059 to +0.0215 over the picked head statistic on 3 of 7 collapsed units; against the full head
  T1's X1-2 is BOUNDED on the same units.

**Notes.** MP-K's item text still says "15 of 20"; the evaluated rule used 10 cells and needed 8, because both wd5e5
pairs are INFO (Δacc above 0.5 pt). No script was edited, and nothing was written apart from the evaluator's own
`results/b4/t2_eval.json`.

---

## Track T3S: lane S, what the pooled head cannot see (X5 "beyond GAP"; `docs/plans/T3S_SPATIAL.md`)

**Evaluator run (step E).** `node scripts/t3s_eval.js --phase confirmation --p1 ad97d0a… --p2 272bc41… --p-run ad97d0a…,14caf51…,272bc41… --sealed results/instrument_check_b4s1_r2/sealed.json --replay results/instrument_check_b4s2/replay.json --json results/b4/eval_t3s.json`
- It ran once at HEAD ee08b34 and exited 0. It wrote `results/b4/eval_t3s.json` (schema `t3s_eval/1`, phase
  confirmation, created_utc 2026-09-24T18:50:48.046Z). It reads the probe records `results/b4_t3s/<unit>/probe.json`.
- **Units.** Sconf (confirmation): resnet20_s3, resnet20_s4, resnet56_s12m, resnet56_s13m; B = 1000. Sdisc (discovery):
  resnet20_hub, resnet56_hub and the two random-init nulls; B = 200.
- **Rows.** Calibration uses CIFAR-10 test 7500-8499; evaluation uses 8500-9499, the clean twins of the faulted rows.
- **Faults.** Synthetic local faults. F5 = the four paste families (noise, blur, pixelate, jpeg) plus the occluder, all at
  12% area (a12). Soiling is held out.
- **Primary statistic.** `map_max`, the frozen per-position PaDiM statistic on the 8 x 8 x 64 head-input map.

### T3S.1 Gates (all pass)

- **Lane and units.** `lane.status` OK, `why` []. All 8 units evaluable with an empty `why`; no NOT_EVALUABLE claim.
- **Provenance.** `problems` = [], `seal_problems` = []. P1 and P2 are ancestors of HEAD. Every record's `repo_commit` is
  in `--p-run`. Eight tracked files are unchanged since P1 (worktree sha = sha at P1 = sha at P2): t3s_spatial_probe.py,
  t3s_eval.js (f66733cd…), b4_core.py, b4_collapse.py, faults.py, b4_extract.py, models.json, T3S_SPATIAL.md.
- **Amendments.** Applied, refused and withdrawn are all empty; `freeze_P2.json` has no T3S amendment.
- **Seals.** The meta sha of each of the 4 Sconf maps dumps equals its entry in
  `results/instrument_check_b4s1_r2/sealed.json`; `verify_seals.json` PASS; each Sconf maps dump was opened exactly once,
  with P2 (`results/instrument_check_b4s2/unseal_log.jsonl` lines 71-74); the touched-once check held.
- **Replay (rule 6).** PASS, drift 0: resnet20_hub and resnet56_hub vs `_s2replay`, n_diff 0, max_abs 0 (a full leaf
  comparison with timing, env and code ignored); `replay.json` has `t3s:resnet20_hub` and `t3s:resnet56_hub` PASS.
- **Known answers.** KA-1 (map-GAP identity) max relative error 1.6e-4 to 1.9e-4 on the Sconf units (bound 2e-3); KA-2
  (pairing) and KA-5 (masks) pass; KA-3 (pixel hashes) 31 splits, 0 differ across the 8 units; KA-4 (calibration band
  [0.027, 0.077] at alpha 0.05): clean false-alarm rates 0.066 / 0.054 / 0.058 / 0.042, and every alpha 0.02-0.05 of the
  rule-5 curve is inside its band on every unit (closest: s3 at alpha 0.04, 0.064 against an upper edge of 0.065); KA-6
  (clean eval accuracy) 0.931 / 0.931 / 0.928 / 0.929.
- **Self-tests.** `node scripts/t3s_eval.js --selftest` 49/49 PASS (writes only to the OS temp folder). S2 probe self-test
  `results/instrument_check_b4s2/selftest_t3s_spatial_probe.json` PASS 23/23; on the planted dump map_max AUROC and
  pointing are 1.0.

### T3S.2 Decision table (Sconf units, in the order s3 / s4 / s12m / s13m)

CIs are 95% group-bootstrap intervals (B = 1000, grouped by image row) from `results/b4/eval_t3s.json` `.claims`. The base
and joint AUROCs are `auc_head` / `auc_joint` in `results/b4_t3s/<unit>/probe.json` `.increments`. The frozen instrument
gives no CI for the AUROC and pointing claims (SP-1, SP-4, SP-5, SP-6); their evaluator margins to the nearest threshold
are 0.045-0.389.

| claim | prediction (P1, T3S_SPATIAL.md §5) | label | units passing / needed | key numbers | what it means for the controller |
|---|---|---|---|---|---|
| SP-1 | X5: on F5 at a12, map_max F5-mean AUROC ≥ 0.90, while every pooled or head score stays ≤ 0.70 two-sided | **REFUTED (MAP-BLIND)** | SUPPORTED rule: 0/4 (needs ≥ 3). MAP-BLIND (map ≤ 0.80): 4/4 (needs ≥ 2). POOLED-SEES: 0/4 | map_max 0.488 / 0.475 / 0.503 / 0.490. Best pooled, two-sided: 0.597 energy / 0.599 energy / 0.586 knn_l2 / 0.586 energy. std_half 0.537 / 0.540 / 0.512 / 0.521. Margin 0.101 | X5's pooled half holds on the F5 mean (the head barely sees these faults); its map half fails. The frozen per-position statistic gives no per-frame local-fault flag |
| SP-2 | The map bundle adds to the full T1 head block plus pooled geometry (d1, knn_l2, gap_maha) on F5 | **REFUTED** | BOUNDED 4/4 (≥ 3 gives REFUTED); ADDS 0/4 | dAUC +0.00001 [−0.0017, +0.0019] / −0.0005 [−0.0019, +0.0009] / +0.0012 [−0.0006, +0.0030] / +0.0004 [−0.0008, +0.0017]. Every dI CI contains 0 (hull [−0.00068, +0.00043] bits). Base 0.601 / 0.596 / 0.595 / 0.597, joint 0.601 / 0.596 / 0.596 / 0.598. TPR at 5% FPR, head to joint: 0.116 to 0.107 / 0.109 to 0.114 / 0.096 to 0.100 / 0.113 to 0.119. DeLong p 0.53-1.00, Holm 1 | **Functional NO.** Adding the map to a head + pooled fault detector gains nothing. That detector is itself weak (AUROC about 0.60) |
| SP-3 | Geometry adds beyond a pixel check (the map over head + pooled + pixel PaDiM, saturation and Laplacian) on the 4 paste families at a12 | **PIXELS-SUFFICE** | BOUNDED 4/4 (needs ≥ 3) | dAUC +0.0017 [−0.0010, +0.0044] / +0.0004 [−0.0021, +0.0027] / +0.0015 [−0.0009, +0.0040] / +0.0015 [−0.0008, +0.0036]. Base 0.573 / 0.569 / 0.570 / 0.572, joint 0.575 / 0.569 / 0.571 / 0.573. p 0.50-0.88, Holm 1. **Caveat:** the pixel check is blind on these families too (T3S.5) | The map adds nothing on top of head + pooled + pixels. On the paste families nothing tested sees the fault well, so this label is **not** evidence that a pixel check is a working detector for them |
| SP-4 | The map says where: F5-mean pointing hit ≥ 0.60 and hit − chance ≥ 0.30 | **REFUTED** | 0/4 (needs ≥ 3) | Hit 0.091 / 0.087 / 0.096 / 0.097 against chance 0.186, i.e. **below chance**. One-cell dilation 0.266 / 0.262 / 0.283 / 0.290 against a dilated chance of 0.404, also below chance. Pixel-PaDiM pointing 0.431 (INFO). Random-init nulls point near chance (0.163 / 0.172; INFO). Margin 0.389 | No localisation. The map's argmax cell lands on the fault less often than a random cell would |
| SP-5 | The frozen statistic catches held-out soiling: AUROC(map_max, soiling a12) ≥ 0.85 | **REFUTED** | 0/4 (needs ≥ 3) | map_max 0.491 / 0.523 / 0.510 / 0.509 (a25: 0.492 / 0.515 / 0.534 / 0.521). Best pooled two-sided 0.641 / 0.649 / 0.628 / 0.629. Pixel PaDiM 0.701 at a12 and 0.853 at a25 (INFO, `.info.area_curve`). INFO increment of the map over pooled on soiling: BOUNDED 4/4, dAUC +0.0012 to +0.0023, upper bound ≤ +0.0049 | No transfer to a held-out fault family. For soiling the pixel check is the strongest score measured |
| SP-6 | Hold case. On still-right rows (fault argmax = clean argmax = label), mean map_max ≥ 0.85 and every head score ≤ 0.65 two-sided | **REFUTED** | 0/4 (needs ≥ 3). Each unit has 5 F5 conditions with 784-903 rows each | map 0.486 / 0.474 / 0.503 / 0.489. Head (max two-sided) on the F5 mean 0.604 / 0.605 / 0.582 / 0.592. Margin 0.045. INFO: on the occluder alone, head / pooled scores on still-right rows reach 0.67-0.73 (energy, msp, knn_l2) | No network-based hold rule meets the registered bar. On still-right rows the head stays within its predicted bound on the F5 mean and the map does not move, but the head is not silent: it registers the occluder |
| SP-7 | X5 arm a: the spatial std (std_half) alone adds to head + pooled on F5 | **REFUTED** | BOUNDED 4/4 (≥ 3 gives REFUTED); ADDS 0/4 | dAUC +0.0004 [−0.0010, +0.0018] / +0.0011 [−0.0008, +0.0029] / +0.0019 [−0.0003, +0.0043] / +0.0043 [+0.0014, +0.0075]. dI CI contains 0 on 3 units; on s13m it is [+0.0002, +0.0017] bits. p 0.12-0.76, Holm 1 | The cheapest beyond-GAP signal (16.6 kB reference, 0.3-0.6 CPU-ms per 1000 frames) adds nothing usable. On s13m the gain is detectable but below the +0.01 floor and the +0.02 bound: functionally no |
| SP-8 | The map tells the 7 local a12 faults from the 7 global s3 corruptions better than head + pooled | **SUPPORTED** | ADDS 4/4 (needs ≥ 3) | dAUC +0.084 [+0.076, +0.093] / +0.080 [+0.072, +0.088] / +0.063 [+0.055, +0.071] / +0.073 [+0.066, +0.081]. dI lower 95% bounds 0.046 / 0.055 / 0.041 / 0.052 bits. Base 0.593 / 0.628 / 0.635 / 0.634, joint 0.677 / 0.708 / 0.698 / 0.707. TPR at 5% FPR, head to joint: 0.095 to 0.144 / 0.108 to 0.170 / 0.124 to 0.151 / 0.117 to 0.168. n = 7000 vs 7000. Holm-adjusted p ≤ 1.53e-58 | A real but weak typing signal: a **candidate** input for routing local faults to HOLD or sensor cleaning and global shifts to ADAPT, too weak to route alone (joint AUROC ≤ 0.71). Consistent with the mechanism disclosed at P2: global corruptions lower the map scores (map_max AUROC 0.24-0.49 on the 7 globals; map_area and map_gini also below 0.5), while the map is near chance on local faults. No ablation of the bundle was run |

INFO blocks (`results/b4/eval_t3s.json` `.info`):
- **SP-9, learned vs random-init: ARCHITECTURAL-OR-MIXED.** Blur and jpeg pastes, map_max: trained 0.4887 vs random 0.4859
  at resnet20 (+0.0028); 0.4894 vs 0.4894 at resnet56 (0). Glare at a12 (area curve): the trained hubs 0.441 / 0.512, the
  random-init nulls 0.960 / 0.921; on the Sconf maps 0.439-0.516.
- **SP-10:** Spearman rho(−log10 NC1, map advantage) = 0.5429 over n = 6 trained units. The map advantage (map_max minus
  the best pooled score) is negative on all 6. A pointer for T2 only, not a test.
- **SP-11, stage-2 map on the hubs:** s2_map_max 0.4924 / 0.4844, beside the head map's 0.4795 / 0.4911; 16 x 16 pointing
  0.106 / 0.087 against chance 0.141; INFO increment of stage 2 over the head map BOUNDED on both hubs.
- **Area curve (`.info.area_curve`).** The pooled head gains sensitivity with fault area faster than the map, which gains
  little (at most +0.09 from a12 to a25, on the resnet56 occluder: s12m 0.541 → 0.627, s13m 0.519 → 0.588; pooled +0.13
  to +0.14). On the Sconf units, occluder at a25: map 0.50-0.63, best pooled 0.82-0.83, pixel PaDiM 0.999; glare at a25:
  map 0.45-0.58, pooled 0.84-0.86, pixel PaDiM 0.9996.
- **Functional table (`.info.functional`), Sconf units:** calibrated map_max flag 66 / 54 / 58 / 42 false alarms per 1000
  clean frames; TPR at the calibrated 5% on F5: map_max 0.053 / 0.051 / 0.060 / 0.040, std_half 0.038-0.054, d1
  0.067-0.087, knn_l2 0.080-0.102, msp 0.070-0.090, pix_padim 0.239. Reference size: map PaDiM 1,064,960 B; kNN 1,280,000
  B; pixel PaDiM 14,336 B. CPU-ms per 1000 frames: map_max 17-37; pooled 82-94; pix_padim 106-116.
- **Holm over the 16 increment DeLong p-values (INFO).** Only the 4 SP-8 tests reject at 0.05 (Holm-adjusted p
  1.08e-72 / 1.66e-79 / 1.53e-58 / 1.66e-68). Every SP-2, SP-3 and SP-7 test has Holm-adjusted p = 1. No label depends on
  Holm: the increment claims use the frozen HEAD-ADDITIVE rule (a bootstrap CI plus an effect floor).

### T3S.3 Headline answers (plain language)

1. **Can a controller read a local sensor-fault flag from the head-input map that the head cannot give?** No. On these
   CIFAR ResNets the frozen per-position statistic is at chance on 12%-area local faults (F5-mean AUROC 0.475-0.503), and
   the map adds nothing to head + pooled geometry (SP-2, BOUNDED 4/4).
2. **Does the spatial std, the cheapest beyond-GAP signal, help?** No (SP-7, BOUNDED 4/4).
3. **Does the map add anything over a pixel check (plus head + pooled)?** No (SP-3, PIXELS-SUFFICE). SP-3 is not a
   head-to-head test of geometry against pixels. INFO: a pixel check sees occluder, glare, dead pixels and soiling at a12
   (0.988, 0.990, 0.789, 0.701) where the map does not; on the pastes nothing sees much, and on the noise paste the pooled
   head is ahead of every pixel score (0.60-0.62 vs 0.47 at a12; 0.69-0.73 vs 0.44 at a25).
4. **Where is the fault?** The map cannot say; its pointing is below chance (SP-4).
5. **Held-out family (soiling)?** Missed by the map (SP-5).
6. **Hold case ("sensor changed, output still right")?** No network-based hold rule meets the bar (SP-6). The map does
   not move; the head stays within its bound on the F5 mean but registers the occluder (0.67-0.73).
7. **Local vs global typing?** Yes, narrowly (SP-8). The map adds +0.063 to +0.084 AUROC over head + pooled on all 4 units
   (dI lower bounds ≥ 0.041 bits). The typer is weak (joint AUROC 0.68-0.71, TPR 0.14-0.17 at 5% FPR). Random-init nets
   reach nearly the same joint AUROC (0.675-0.677) through head + pooled alone (base 0.660 / 0.662; map increment +0.017 /
   +0.013), so the typing information is not mainly learned; training moves where it is readable (the map's increment is
   +0.063 to +0.096 in trained nets).
8. **Calibration.** The conformal machinery is sound: KA-4 passes on 4/4 units, and the rule-5 curve is inside its band at
   every alpha. The problem is the missing signal, not the calibration.
9. **Why map_max is blind.** On the F5 conditions the random-init maps' map_max is as blind as the trained ones (F5 mean
   0.490 / 0.498 against 0.475-0.503), so for F5 the blindness is architectural-or-mixed (SP-9). Only on glare and dead
   pixels are the random-init maps sensitive where the trained ones are not (glare 0.92-0.96 vs 0.44-0.52; dead pixels
   0.70 / 0.76 vs 0.48-0.56): training removes that sensitivity. The map as a representation is not blind: its spatial
   mean (the input of every pooled score, KA-1) carries the occluder and glare (pooled 0.69-0.75 at a12, 0.82-0.86 at
   a25). The information is spread over positions rather than concentrated at the fault cells (an interpretation, not
   tested).

**Lane verdict.** X5's "beyond GAP" hypothesis, as registered (the frozen per-position statistic at the head-input
layer), is refuted on these CIFAR ResNets. Six claims are refuted on 4/4 units (SP-1, SP-2, SP-4, SP-5, SP-6, SP-7); SP-3 is
PIXELS-SUFFICE with its caveat; the only confirmed increment is local-vs-global typing (SP-8), modest and consistent with
a one-sided mechanism. Scope (T3S_SPATIAL.md §5): synthetic faults, 8 x 8 maps, CIFAR ResNets. Nothing here transfers to
ImageNet resolution, other architectures or real sensor faults without new measurement.

### T3S.4 Rule 8 notes

- **map_max below 0.5 on trained units looks suspiciously bad.** Artifact hypothesis: a broken scorer. Ruled out: on the
  planted dump the scorer gives AUROC 1.0 and pointing 1.0 (S2 self-test 23/23); KA-1 holds on every split; KA-4 is
  calibrated; the same scorer on the same pixels sees glare in the random-init nets (0.92-0.97 at a12/a25) and dead pixels
  (0.70 / 0.76 at a12). The blindness is a property of the frozen statistic on these maps, not a scorer bug.
- **SP-8's DeLong p of 1e-80 to 1e-59** (raw 1.04e-80 to 1.18e-59; Holm-adjusted ≤ 1.53e-58) comes from n = 7000 vs 7000;
  the effect is moderate. The one-sided mechanism was disclosed at P2, before any confirmation read.
- **Tiny evaluator margins on SP-2, SP-3 and SP-7 (0.00022, 0.000034, 0.0001).** These are the distance of a dI lower
  bound from 0, which does not decide BOUNDED. The decisive quantities are far from their thresholds: every dAUC is
  ≤ 0.0043 against the +0.01 floor, and every upper bound is ≤ 0.0076 against +0.02. Replay drift is 0.
- **SP-7 on s13m.** The bootstrap CIs exclude 0 while DeLong p is 0.12. Detectable but functionally negligible; the frozen
  rule calls it BOUNDED.

### T3S.5 Discovery vs confirmation

Discovery values are the frozen predicates on the hubs (INFO, B = 200), from `.discovery_info.predicates`. That block is
identical to `results/b4/eval_t3s_discovery.json`, which was written after S1 at HEAD R1 (`cb617c3`, created
2026-09-24T09:15Z) and committed in P2, before any confirmation read.

| claim | discovery (resnet20_hub / resnet56_hub) | confirmation (4 Sconf) | replicated? |
|---|---|---|---|
| SP-1 | map 0.479 / 0.491; pooled 0.596 (energy) / 0.595 (knn_l2); MAP-BLIND on both | map 0.475-0.503; pooled 0.586-0.599; MAP-BLIND 4/4 | yes |
| SP-2 | BOUNDED on both: −0.0005 [−0.0021, +0.0011] / +0.0008 [−0.0019, +0.0033] | BOUNDED 4/4 | yes |
| SP-3 | BOUNDED on both: +0.0001 [−0.0028, +0.0024] / +0.0021 [−0.0011, +0.0061] | BOUNDED 4/4 | yes |
| SP-4 | hit 0.089 / 0.111 against chance 0.186 | 0.087-0.097 | yes |
| SP-5 | not readable (soiling is confirmation-only, D7) | 0.491-0.523, REFUTED | new at confirmation |
| SP-6 | map 0.476 / 0.489; head 0.603 / 0.587 | map 0.474-0.503; head 0.582-0.605 | yes |
| SP-7 | BOUNDED on both: −0.00003 [−0.0005, +0.0004] / +0.0013 [−0.0007, +0.0030] | BOUNDED 4/4 | yes |
| SP-8 | ADDS on both: +0.096 [+0.087, +0.105] / +0.092 [+0.083, +0.100]; joint 0.698 / 0.691 | ADDS 4/4: +0.063 to +0.084; joint 0.677-0.708 | yes. Every Sconf dAUC is below both hubs'; the cause was not tested |
| SP-9 / SP-11 | same Sdisc records | identical values | yes (same data) |
| SP-10 | null at n = 2 | 0.5429 at n = 6 | first value |

Everything that discovery could read replicated in direction and, in size, to within about 0.03 AUROC. No label changed
direction.

### T3S.6 Required caveats (RUN_REQUEST_S2.md "Step E")

- **SP-3 caveat** (`experiments/b4/freeze_P2.json` "disclosed"): PIXELS-SUFFICE means only that the map adds nothing over
  head + pooled + pixels; the pixel check does not see the paste families either. Confirmation values of pix_padim at a12:
  noise 0.473, blur 0.606, pixelate 0.489, jpeg 0.486 (identical on all 4 units, because KA-3 shows the same pixels). The
  base AUROC with pixels included is 0.569-0.573.
- **SP-9 reading** (disclosed): ARCHITECTURAL-OR-MIXED because trained and random-init maps are both blind on blur and
  jpeg (about 0.49). The disclosure quotes "learned minus random -0.40 to -0.52" on glare; recomputed from the area curve it
  is −0.41 to −0.52 (0.5118 − 0.9208 and 0.4410 − 0.9597). The Sconf maps agree: glare at a12 is 0.439-0.516.
- **Batch-5 notes** (disclosed; no P2 change was allowed): map_max is one-sided (global corruptions push it below clean,
  which drives SP-8's ADDS); pix_padim's log(Laplacian var + 1e-6) makes flat cells the clean maximum, which bears on the
  pixel baseline's blindness to pastes.
- **Not in this track.** X3-3 (T1, ACTIVE, harm-reference caveat) is in the T1 section; P4 (T2, INFO (withdrawn at P2)) is
  in the T2 section. B4a: C = 12 (the frozen gate alone would have left 10); `results/b4/b4a_seals.json` PASS; no ShuffleNet
  timing value raised; no lane-S unit is a C unit, so B4a does not touch T3S.
- **Host.** S2 ran on an AMD Ryzen 9 7950X 16-Core (`results/instrument_check_b4s2/versions.json`, cpu_count_host 32),
  CPU quota 6.8, P = 5 (R2 commit message), by owner decision (RUN_REQUEST_S2 §2 asks for EPYC-Genoa, unavailable). The
  stack equals S1's: `versions_match.json` match = true for python 3.12.3, numpy 2.1.2, scipy 1.18.1, scikit-learn 1.9.1;
  torch 2.8.0+cu128 in both S1 and S2 `versions.json`. The T3S anchor replays across the two hosts are bit-identical
  (n_diff 0, max_abs 0), so the host moved no replayed T3S statistic. The replay covers the anchors in discovery mode
  (B = 200, no soiling increment); the confirmation-only paths (B = 1000, soiling) share the same deterministic code but
  were not replayed across hosts.
- **Timing** (`results/instrument_check_b4s2/timing.json`). Each T3S confirmation probe took 18.4-19.2 s wall: 0.51-0.53%
  of its B4b limit (3600 s) and 1.02-1.06% of the emitted limit (1800 s, `timeouts_b4_emitted.json`). The replays took
  23.3-23.4 s (0.65% and 1.30%). Peak RSS 438-440 MB, against a peak_gb of 0.5. No kills or relaunch (0 soft failures in
  `block_b4s2`, R2 commit). The largest share of a limit in S2 was 0.528 of the emitted limit, a T1 job (see
  "Deviations").

### T3S.7 Inventory mapping: PC-7, lane-S part (`docs/knowledge/README.md`, D19)

The row is shared with T1 (X3-5, X3-8; T1.6). The lane-S text does not depend on it.
- **"beyond the output head?" (lane-S part):** **NO** for a local-fault flag (4/4 Sconf units; CIFAR-10 test 8500-9499;
  synthetic local faults at 12% area): the map bundle adds nothing to head + pooled geometry (SP-2: dAUC −0.0005 to
  +0.0012, every upper 95% bound ≤ +0.0030); the spatial std adds nothing (SP-7: dAUC +0.0004 to +0.0043, upper bound
  ≤ +0.0076); over head + pooled + a pixel check the map adds nothing (SP-3 PIXELS-SUFFICE; the pixel check is itself blind
  on the pastes); map_max is at chance (SP-1: 0.475-0.503), does not localise (SP-4: 0.087-0.097 against 0.186), misses
  held-out soiling (SP-5: 0.491-0.523) and does not flag the hold case (SP-6: 0.474-0.503). **YES, narrow**, for
  local-vs-global typing: +0.063 to +0.084 over head + pooled (SP-8, ADDS 4/4, dI lower bounds 0.041-0.055 bits; joint
  AUROC 0.677-0.708), consistent with global corruptions lowering the map scores.
- **Status (lane-S part):** REFUTED for the lane-S flag, localisation, held-out family and hold case (SP-1, SP-2, SP-4 …
  SP-7 on 4/4 Sconf units, plus SP-3 PIXELS-SUFFICE); all except SP-5 replicate the discovery hubs (SP-5 is
  confirmation-only, soiling held out, D7). ESTABLISHED at narrow scope for SP-8 typing (4/4 Sconf units and 2/2 discovery
  hubs, pre-registered at P1).
- **Controller readiness (lane-S part):** do not use map_max on the trained head-input map, or its spatial std, as an
  F1-b sensor-fault flag or a HOLD-T trigger on these nets: calibrated (42-66 false alarms per 1000 clean frames at alpha
  0.05), but TPR 0.040-0.060 on F5. For F1-b, a pixel check is the better starting point for occluder, glare, dead pixels
  and soiling (INFO; not tested against the head); on the noise paste the pooled head is ahead of it. SP-8 is a candidate
  ADAPT-vs-HOLD routing feature, too weak alone (TPR 0.144-0.170 at 5% FPR).
- **F1-b line for the controller-function table:** "batch 4 lane S: map_max on the trained head-input map is blind to
  synthetic local faults (SP-1 … SP-7 on 4/4 units; random-init maps see glare); a pixel check sees occluder, glare and
  soiling; `results/b4/eval_t3s.json`."
- **CD-12 (T2).** SP-10 (rho 0.5429, n = 6) is a pointer only and writes nothing into CD-12.

---

## B4a: the one-sided README gate and the C = 12 units

- **What happened.** The frozen weights check (`scripts/b4_weights.py`) refuses a hub unit when |acc_10k − README top-1|
  > 0.003. Three hub units failed it, all by an *excess* over the README, and nothing else: mobilenetv2_x0_75 (C, +0.0036),
  shufflenetv2_x0_5 (Dnew, +0.0052) and shufflenetv2_x1_0 (C, +0.0032) (`docs/plans/B4A_AMENDMENT.md`). Over all 19 hub
  units the deltas run from −0.0001 to +0.0052; none is below its README by more than 0.0001.
- **The amendment** (owner-approved 2026-09-24, P_A `14caf51`, before any sealed dump was read): the gate is one-sided. It
  fails only when acc_10k < README − 0.003; an excess passes with the INFO flag `README-EXCEEDED`. The lower side equals
  the frozen rule bit for bit. The three units were extracted in a supplement on the same S1 pod (0 soft failures), the two
  C units sealed, shufflenetv2_x0_5 read as discovery (tag `_r2`).
- **C = 12.** Under the frozen gate alone C would have held 10 (mobilenetv2_x0_75 and shufflenetv2_x1_0 missing;
  `freeze_P2.json` `b4a.c_count_under_frozen_gate_only`), which would have left the PRIMARY one unusable unit away from
  NOT_EVALUABLE, and discovery without a ShuffleNet.
- **Seals.** `results/b4/b4a_seals.json` PASS (created 18:49:38Z, before any evaluator ran): seal_diff PASS (the 34 S1
  seals unchanged, 2 added; manifest `results/instrument_check_b4s1_r2/sealed.json`, 36 entries); both units PASS, each
  covered by `results/instrument_check_b4s2/verify_seals.json` with `s2_guard: true`; their T1 and T2 records are
  confirmation-phase with `meta_sha256_equal: true`. `verify_seals.json` PASS on all 36 dumps.
- **Timing.** No ShuffleNet timing value was raised (`freeze_P2.json` `b4a.timing_raised = []`; T1 measured 306.8 s
  against a floor of 301.1 s, T2 71.5 s against 65.3 s).
- **Does any label depend on B4a?** No T1 label: removing the two units leaves every M label and X4-1M unchanged. T2 needs
  n ≥ 10 per outcome, so the PRIMARY would have been evaluable at 10 but fragile. No lane-S unit is a C unit.
- **README-gate cause (INFO, found after S1; not a registered check).** For the 11 group-B hub units (mobilenetv2,
  shufflenetv2 and repvgg, trained 2021-04-14/15) the upstream chenyaofo code evaluated the validation set under the
  training augmentation, so their README value is the maximum of 200 augmented evaluations. Group A (resnet, vgg) matches
  the upstream logs within ±1 image. This is consistent with the one-sided excess B4a allowed. The inspection of the
  upstream logs and code is recorded in `docs/reviews/HUB_README_GATE_2026-09-24.md`. B4a does not depend on it, and no
  batch-4 claim reads a README value (T2's accuracy
  rival is the probe's own error on test rows 0-4999). Eight of the 12 C units are group B.

## Deviations and disclosures

1. **S2 host.** RUN_REQUEST_S2 §2 asks for an EPYC-Genoa host; S2 ran on an AMD Ryzen 9 7950X (quota 6.8, P = 5) by owner
   decision, because Genoa hosts were unavailable. The stack is identical (python, numpy, scipy, sklearn, torch). The rule-6
   replay of both 64-d anchors PASSed for t1, t2 and t3s with n_diff 0 at rel 1e-6 / abs 1e-9 (T3S bit-identical). Scope of
   that evidence: the anchors, at discovery settings. No wide-penult unit was replayed across hosts, and the cost block's
   milliseconds are host- and load-specific.
2. **B4b limits.** Every S2 limit was 2x the emitted one (owner-approved amendment B4b, `docs/plans/B4B_AMENDMENT.md`,
   committed in P2). Observed, from each output's `timing_s.total` against its own directory's limits: the largest share
   was **0.528 of the emitted limit** and 0.264 of the B4b limit (T1 resnet20_s1_eval, 955 s of 1809 s). No job would have
   been killed under the emitted limits either. The R2 commit message says 0.56 / 0.28; that figure divides
   resnet56_s31_eval's 1007 s by the fit-layout limit of the same unit (1800 / 3600 s), because
   `results/instrument_check_b4s2/timing.json` keys the eval job under the bare unit id. Against its own eval limits
   (2126 / 4252 s) it used 0.474 / 0.237.
3. **X3-3 (T1)** stays ACTIVE with the harm-reference caveat disclosed at P2 before any confirmation read (T1.5). No rule
   was changed.
4. **P4 (T2)** was withdrawn to INFO at P2 under D8 (b), its direction contradicted on D21 (T2.5). The evaluator still
   labels it and counts it in Holm (m = 22), which is conservative for the other items.
5. **Other P2 disclosures, no rule change:** X3-8 uses the stricter frozen early list; X4-3's comparator is the per-split
   best of 9 head statistics; do3.nc1_train is invalid at wide units (X4-2 scoped to the 64-d ResNets); T2's D21 PRIMARY
   ratios are optimistic (in-sample coordinate choice); SP-3 caveat; SP-9 reading; T3S batch-5 notes.
6. **Nothing else.** No script was edited, no evaluator was re-run, no output was overwritten, and no claim was cut.

## What goes to ATLAS_STATUS

Rows 12-19 are added to ATLAS_STATUS.md (row 9's evidence gains the X1-2 cross-reference). The promotion rule is the
CLAUDE.md one: ✅ needs a pre-registered prediction confirmed on real data across ≥ 2 seeds with the holdout declared
before the run. R has two fresh seeds at each depth (resnet20 s31/s32, resnet56 s31/s32). So:
- ✅ for pre-registered SUPPORTED claims confirmed on R (or on the 10 STconf / 4 Sconf units), stated at their scope;
- 🟡 for pre-registered positives that were not predicted (X1-5, PC-2b, X7-1) and for T2's SUPPORTED items (nothing from T2
  alone is promoted: one recipe, one dataset);
- ✗ for REFUTED claims.

## What batch 5 should do

In order of controller value per dollar, each tied to a batch-4 result:
1. **Make the shift-time error gain deployable (F1-a under shift).** Register the head + early-bundle joint model fitted on
   synthetic corruption families and scored on held-out families and a natural-shift set (a transfer that batch 4 did
   not measure for errors), with the holdout-family figure (+0.004 to +0.014 when fitted on those families, INFO here)
   as a reference size and "beyond pixels" as a co-primary. Carry the early bundle's
   architecture generality (11/12 C, INFO here) into a registered claim on new architectures.
2. **Register unsupervised pixel baselines beside the head everywhere.** Several batch-4 positives are shared with pixel
   statistics (PC-2b, X3-1, X6-5, SVHN). Needed comparators: a pixel MEWMA for streams (F1-d), a pixel PaDiM / pixel OOD
   score for F1-b and F1-c (it was never scored unsupervised), a pixel T2 for batches, a pixel router for routing.
3. **Test routing by its response, not its AUROC.** Does routing by the early-tap family (PC-2a) make a routed ADAPT
   response better than a pixel router or no routing? (ST-1, PC-2.)
4. **Fix the stream harm reference and test the conservative calibration.** Register a harm reference not dominated by
   integer-threshold sampling noise (X3-3 caveat), temporally correlated streams, and a rotation of the calibration rows to
   test the shared-row hypothesis behind the conservative held-out ARL0 and X4 FPR (T1.4 item 6).
5. **Re-register X4 against a deployable comparator.** X4-3 against the pre-selected head statistic (head.best_stat), where
   the INFO here favours X4 on 7/12 R/R2 units; X4 as a feature in the F1-a joint model.
6. **Batch estimators.** Register the penult displacement estimator (best of all on every resnet56 unit, INFO) against the
   head's mean gap / mean energy and ATC; replace the degenerate typing cascade (X6-4) with an order-free or joint rule.
7. **T2.** Retire the O1-O5 model-level laws as a configuration route (PRIMARY REFUTED, 0/5). If kept, register P5a (harm
   slope) and P6b (rank of the tap-over-head gain) on unseen families and a second dataset, and register labelled nc1
   against the head's saturation for the DO-3 ranking (an unregistered near-tie here).
8. **T3S.** From the P2 batch-5 notes and T3S.3 item 9: a two-sided or spread-aware spatial statistic (the fault
   information is in the map's mean, not its per-position maximum), the pix_padim flat-cell fix, a pixel check registered
   as the F1-b baseline against the head, and real sensor faults. The 16 x 16 stage-2 map was also blind (SP-11, INFO), so
   resolution alone is not the lever.
9. **Scope.** Ask the shift-time question (X1-4 / X1-5 analogue) on a second dataset or on the ImageNet ViTs, where the
   head is injective at the CLS tap, with fresh seeds or never-read rows.
10. **Instruments.** Keep B4b-style limits (≤ 0.53 of the emitted limits even on the substitute host). Key `timing.json` by
    output directory, not unit id (the source of the 0.56 figure). Include one wide-penult unit in the cross-host replay.
