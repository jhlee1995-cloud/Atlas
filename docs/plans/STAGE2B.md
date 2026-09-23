# Stage 2b plan (A4b): depth-56 replication and the matched rung

Committed before the run, in one pre-registration commit P together with B1 (docs/plans/B1_VIT_MARGIN.md). Predictions:
`experiments/queue/atlas_v1_resnet56_s1.yaml` notes (gates, ladder, D-ID / D-COLL, depth-56 claims, twin, pair
levels, scale labels) and `experiments/queue/margin_v1_resnet56_s1.yaml` notes (M56). The evaluation is frozen as
code: `node scripts/a4b_eval.js` (its `--dry-run` on the committed Stage 2 files reproduces the Stage 2 numbers, and
`bash scripts/a4b_eval_fixture.sh` runs it on committed atlases under A4b names in the matched, interp and below
scenarios). This file holds the design, the outcome tables and the promotion rules. It amends
`docs/plans/STAGE2.md` for A4b only (STAGE2.md "Amendment 1"); Stage 2's own labels and SESSION.md stay as recorded.
The integration decisions that shaped it are cited as D<n> (A4b + B1 integration, 2026-09-23).

## What Stage 2 left open

- The two DEPTH findings (penult ID higher; penult collapse stronger at matched accuracy) are not claims yet
  (`results/atlas_v1_resnet56_s0hub/SESSION.md` "Next actions" 2). The collapse labels rest on interpolating between
  e40 (last epoch, 40 epochs) and the test-selected hub, and e40 alone sits on the other side of the band.
- No rung landed within 0.005 of 0.9259 (e40 at -0.0067).
- Depth 56 is one instance (the hub); estimator noise at depth 56 is unmeasured (no twin).
- Two rule gaps: whether SCALE-VARIANT overrides the ROBUST column, and no band for pair metrics. Row 7 depends on the
  reference: D1 0.802 / 0.803 / 0.771 against the hub / s1 / s2 resnet20 references.

## Facts that shape the design

- **Per-epoch cost.** resnet56 alone, 8 workers: 3.52 / 3.52 / 3.45 s per epoch (e10 35.2 s, e20 70.3 s, e40 138.1 s,
  `results/train_resnet56_e{10,20,40}/train.json` `wall_s`, including start-up and evaluations). That is about 14.5k
  training images per second. resnet20 with 5 workers ran at 1.26 s per epoch, about 40k images per second
  (`results/train_resnet20_s1/train.json` history, epochs 10 -> 20: 13.2 -> 25.8 s; recipe.workers 5). So the data
  pipeline is not what limits resnet56: its training is GPU-bound, and a second concurrent training would only share
  the GPU. Trainings therefore run one at a time; atlas builds (about 58 s of per-image CPU work and 23 s of Stage B
  each, `provenance.json` `wall_s` 80-84 s) run beside them.
- **Accuracy vs E** (seed 11, cosine T_max = E, last epoch): e10 0.8399, e20 0.8985, e40 0.9192; the hub (200 epochs,
  another codebase run) best 0.9438, last 0.9421 (`docs/plans/STAGE2.md` "Facts"). Three curve models (log-linear
  e40 -> e200; power law through e20, e40, e200; power law through e10, e20, e40) predict:

  | E | predicted acc_10k (range over models) | inside [0.9209, 0.9309]? |
  |---|---|---|
  | 50 | 0.9222-0.9243 | yes, all models |
  | 60 | 0.9244-0.9280 | yes, all models |
  | 70 | 0.9256-0.9309 | yes (upper edge in one model) |
  | 80 | 0.9265-0.9332 | not in the steeper model |

  Seed spread of a full-recipe resnet20 run is 0.0019 (sd of s1-s4 final_test_acc_10k 0.927 / 0.9264 / 0.926 /
  0.9301). {50, 60, 70} puts every model's prediction for every rung inside the window and brackets the target from
  both sides under every model; {60, 80} relies on e60 alone under the steep model. Cost: 180 epochs, 10.4-10.6 min.
- **The ladder trend of the collapse fields** (e10 / e20 / e40 / hub): sep_ratio 1.45 / 2.04 / 2.85 / 5.33, bridge
  2.07 / 2.59 / 3.19 / 5.59, nc1 0.686 / 0.320 / 0.177 / 0.049, train-reference fit 0.875 / 0.951 / 0.994 / 1.0.
  Collapse moves several band widths per doubling of E while accuracy moves a few tenths of a point, so the matched
  value of a collapse field depends on where inside the accuracy window the rung falls. The outcome table tests that.
- **Seed 11 is discovery material.** `torch.manual_seed(args.seed)` runs before the model is built and the loader is
  shuffled (`scripts/train_second_seed.py:73-74,85,90`), so every seed-11 rung shares its initial weights and data
  order with e10/e20/e40, the rungs that produced the DEPTH hypothesis (`atlas_v1_resnet56_e40.yaml` holdout). The
  matched leg therefore gets two independent replicates, seeds 12 and 13.

## Runs and seeds

| run | weights | role |
|---|---|---|
| `atlas_v1_resnet20_{s0hub,s1,s2,s3,s4}_st3` | Stage 1/1b weights | band B20+ and pair band P20, this session at this commit (rule 6) |
| `atlas_v1_resnet56_s0hub_st3` | hub (best epoch, test-selected) | discovery instance re-measured; twin partner; G0c replay of Stage 2 |
| `atlas_v1_resnet56_s0hub_ref1` | hub, reference seed 1 | estimator twin at depth 56 (T56) |
| `atlas_v1_resnet56_e40_st3` | the Stage 2 e40 checkpoint (seed 11, 40 epochs) | same-session re-measure; enters only as an interpolation end |
| `atlas_v1_resnet56_e{50,60,70}` | seed 11, E epochs, cosine T_max = E, last epoch | ladder; M11 = the matched rung |
| `atlas_v1_resnet56_e90` | seed 11, 90 epochs | only if e50, e60 and e70 are all below 0.9209 |
| `atlas_v1_resnet56_s12m`, `_s13m` | seeds 12 and 13, E* epochs (E* = the pod's matched rung) | the two confirmation replicates of the matched leg (M12, M13); only if E* exists |
| `atlas_v1_resnet56_s1`, `_s2` | seeds 1, 2, 200 epochs, hub recipe, last epoch | depth-56 confirmation pair |
| `margin_v1_*` (item M56) | Stage B only on the dumps above | margin at depth 56, full recipe and matched |

Seeds: 0 = chenyaofo hub (discovery). 1 and 2 = depth-56 confirmation seeds, declared as `confirmation_seeds: [1, 2]`
in every Stage 2 resnet56 manifest at 664bd25 and never trained. 11 = the ladder seed (discovery; e10/e20/e40 in
Stage 2). 12 and 13 = the matched leg's confirmation replicates (new). 99 = the random-init null (Stage 2; not
re-run). The hub counts as a third, discovery instance only. Every A4b resnet56 manifest declares
`confirmation_seeds: [1, 2, 12, 13]`.

## Instrument and rule 6

- Every A4b manifest names the Stage 1 instrument (the `atlas_v1_resnet20_s3.yaml` lists: invariants, cross-layer
  invariants and probe factors, in registry order). `select()` keeps the listed order (`atlas/registry.py:76-90`) and
  `build.py:40` sorts stably by cost, so anything registered later can neither enter an A4b atlas nor shift its
  estimator draws. The margin manifests exclude every registered cross-layer invariant by name. B1 registers no
  invariant and no cross-layer invariant (D2), and `tests/test_atlas_smoke.py::test_a4b_manifests_freeze_the_instrument`
  checks all of this before any pod block runs.
- **B1's only instrument edit is `atlas/invariants/margin.py`**, with its keys off unless a manifest sets
  `margin_typeb.b1: true` (no A4b manifest does). A4b's main atlases never call it (explicit lists); the M56 rebuilds
  do. Before any block uses it, the session preflight (D6, `pod_atlas.sh` after the smoke tests, hard) rebuilds the
  committed A3 atlases `margin_v1_resnet{20,56}_s0hub` from their dumps with the committed manifests and requires
  `scripts/check_rebuild.py` PASS (leaf rule); the record is `results/instrument_check_a4b_b1/`, and a PASS that is
  not bitwise is recorded in SESSION.md as environment drift.
- **Code-diff gate (G0d, D7).** `block_a4b` stops before any training if a Stage A or Stage B file other than
  `atlas/invariants/margin.py` changed since 664bd25 (`atlas/extract_acts.py atlas/factors atlas/config.py
  atlas/context.py extract scripts/train_second_seed.py atlas/build.py atlas/registry.py atlas/invariants
  atlas/compare.py atlas/critic.py experiments/tolerances_default.yaml`). `requirements.txt` changes (B1's pins) are
  judged by `versions.json`: python, numpy, scipy and sklearn must equal `results/instrument_check_stage1b/check.json`
  (3.12.3, 2.1.2, 1.18.1, 1.9.1), torch and the GPU must equal `results/train_resnet56_e40/train.json`
  (2.8.0+cu128, NVIDIA GeForce RTX 4090). `ATLAS_A4B_ALLOW_DIFF=1` overrides the file gate only after the change is
  listed here in an amendment.
- **Decision: re-measure, do not reuse.** All five resnet20 band members, the resnet56 hub and e40 are re-extracted
  and rebuilt in this session. Reasons: rule 6 says "same session" (CLAUDE.md:59-60); B1 changes margin.py and the
  installs, which R56-0c (Stage 2, 664bd25) cannot vouch for; the re-measure runs beside the GPU lane, so it costs
  almost no wall time; a replay-only check would re-extract at least one run anyway and yields no band if it fails.
- **G0c (recorded)** records whether the re-measure equals the committed atlases: the Stage A/B file diff since
  664bd25 (`<check dir>/code_diff.txt`), a Stage-B rebuild of the st2 hub dump with the st3 manifest (`check.json`),
  the same-space compares (penult cka_test >= 0.999), and a leaf-by-leaf comparison of every `_st3` atlas.json and
  every `margin_v1_*_st3` with its committed original (`a4b_eval.js`). IDENTICAL lets Stage 1/1b/2 values be quoted
  beside A4b values; DIFFERS forbids it. The labels are unaffected, with one exception, the **instrument gate**
  (review item 4): every d56 tag, row 11 and the M56-b tag need `check.json` PASS (leaf rule) and C1-C5a holding in
  all five `_st3` atlases; otherwise "no verdict (instrument changed)".

`<check dir>` is `results/instrument_check_a4b`; a relaunch uses `ATLAS_A4B_CHECK_DIR=results/instrument_check_a4b_r2`
and `a4b_eval.js --check-dir instrument_check_a4b_r2`.

## Bands

- **B20+** (scalars) = the five `_st3` atlases. Per field: min, max, w = max - min; HIGH above max + w, LOW below
  min - w, IN otherwise. B20+ contains the Stage 2 band B20 (three members), so HIGH/LOW under B20+ implies the same
  under B20. The B20 reading is reported beside it (INFO, continuity). s3 and s4 enter because Stage 1b is decided;
  at the committed values the ID edge is unchanged (10.314) and the collapse edges widen (sep_ratio 3.137 -> 3.163,
  bridge 3.461 -> 3.491, nc1 0.156 -> 0.154).
- **P20** (pair metrics) = the 10 pairs of B20+; e20(m) = min over P20. No w extension: with 10 pairs the range is
  estimated far better than with three members, and the extension would put the D1 edge at 0.770, below every
  cross-depth value observed so far. The three st2 pairs alone (D1 0.950-0.963) understate the seed spread of D1 at
  depth 20 (0.867-0.964 over the 10 committed pairs); that is why s3 and s4 are re-measured too.
- **CV band**: penult center-distance CV (std / mean of the 45 distances) over B20+; committed values
  0.0733 / 0.0685 / 0.0808 / 0.0791 / 0.0742, low edge 0.056. Hub 0.037.

## Matched value (A4b; replaces STAGE2.md's steps for A4b)

1. Rungs = the seed-11 last-epoch rungs {e40, e50, e60, e70, e90 if trained}; accuracy = `train.json`
   `final_test_acc_10k`; target 0.9259 (`results/norm_check_resnet20/norm_check.json` `chenyaofo.acc_10k`). In window:
   |acc - 0.9259|, rounded to 4 decimals, <= 0.005. **M11** = the in-window rung nearest 0.9259 **whose atlas was
   built**; a tie goes to the longer E (closer to resnet20's training fit). **W** = every in-window seed-11 rung with
   a built atlas; in-window rungs without an atlas are reported. The pod applies the same rule without the atlas
   condition (`scripts/matched_rung.py`, `<check dir>/ladder.json`, E*) to decide the replicates' length; if M11 is
   not E*, no Mc is evaluable (CONFIRMED-1 at best).
2. No rung in the window: linear interpolation between the two adjacent seed-11 rungs (sorted by accuracy) that
   bracket 0.9259. Never the hub, never s1/s2/s12m/s13m. e40 enters only as `atlas_v1_resnet56_e40_st3` (this session),
   never through the Stage 2 atlas. W is empty, the replicates are not trained, and the best outcome is CONFIRMED-1.
3. Otherwise NOT_EVALUABLE. The pod adds e90 only when every rung is below the window (`scripts/matched_rung.py`
   status `below`), before choosing.
- **Mc** = M12 (`atlas_v1_resnet56_s12m`) or M13 (`atlas_v1_resnet56_s13m`), both trained at E*. Mc count iff their
  own accuracy is in the window (and they were trained at M11's E and their atlas exists); this inclusion rule is
  pre-registered. Their length is fixed by seed 11.
- s1 and s2 are never matched values: they replace the test-selected hub as the depth-56 side of the label.

## D-ID and D-COLL (penult levels at matched accuracy; ATLAS_STATUS row 11)

Pre-registered sides (the Stage 2 m* sides): twonn_id.id HIGH; class_centers.sep_ratio HIGH; bridge ratio (mean
off-diagonal center distance / mean radius) HIGH; neural_collapse.nc1 LOW. Legs, per field:
- **L1 (accuracy-matched):** M11 (or the interpolated value) on the side, and every Mc on the side.
- **L1-window:** every rung in W on the side.
- **L2 (recipe- and fit-matched, last epoch):** s1 and s2 on the side. They share resnet20's recipe, codebase and
  training fit (train-reference accuracy about 1.0 vs 0.9994-0.9997) and are more accurate.

Each claim has one conservative leg. Along the ladder, penult ID falls and collapse rises with training (11.98 ->
11.74 -> 10.88 -> 10.67; sep_ratio 1.45 -> 5.33). For D-COLL the matched rungs are less trained than resnet20, which
biases L1 against the claim; for D-ID the full-recipe seeds are more trained and more accurate, which biases L2
against it. Both legs are required, so neither confound can produce CONFIRMED on its own.

Outcome per field (first match wins; review items 1, 2, 10):

| outcome | condition | meaning |
|---|---|---|
| NOT_EVALUABLE | no matched value (no M11 and no interpolation, or a missing interpolation atlas); R0 fails; KILL-56; T56 core FAIL or no twin; a G0b failure in any run the field reads; the instrument gate | none |
| NOT-REPLICATED | s1 or s2 not on the side | the hub's level is instance-specific |
| NOT-DEPTH | M11 off the side and no Mc on the side | Stage 2's DEPTH label is withdrawn |
| SEED-SENSITIVE | M11 and the Mc disagree on the side | same length, different seed: seed variation |
| FIT-SENSITIVE | a W rung (with atlas) off the side | depends on training length inside the accuracy window |
| CONFIRMED-1 | all on the side, with at most one Mc (or the interpolation branch) | one matched confirmation instance |
| CONFIRMED | all on the side, M12 and M13 both evaluable | two matched confirmation instances and two full-recipe seeds |

D-ID = the ID field's outcome. D-COLL = the weakest of its three fields (order as in the table). The Stage 2 style
label (DEPTH / NOT-DEPTH with M11 as m*, per full-recipe seed) is also reported for every field, and for the
non-claim [ACC] fields class probe excess, nearest_center_acc_test and meta.accuracy.test (expected NOT-DEPTH).
`meta.accuracy.ref` of M11, Mc and W is reported as the fit covariate; per field, T56's |delta| is reported next to
(m* - band edge), INFO.

**ATLAS_STATUS row 11** ("penult level, resnet56 vs resnet20 at matched accuracy"; D12: row 10 is B1's), with the two
claims D-ID and D-COLL, each:
- ✅ for CONFIRMED, when nothing blocks promotion: G0e PASS, `synthetic_refusal` and `holdout_hygiene` PASS in
  `critic_v1_resnet56_s1_s2` (AGENT_LOOP.md promotion rule), and no unexplained mismatch between the critic and
  `a4b_eval.js`. A blocked CONFIRMED is 🟡 until the block is resolved in SESSION.md.
- 🟡 for CONFIRMED-1, SEED-SENSITIVE or FIT-SENSITIVE.
- ✗ for NOT-DEPTH. Stage 2's DEPTH text in rows 1 and 2 is then replaced by "not attributable to depth at matched
  accuracy".
- 🟡 "hub-specific" for NOT-REPLICATED.
- ⬜ (no verdict) for NOT_EVALUABLE.

## Depth-56 scope of rows 1-9 (the d56 tag)

- **Decision pair:** s1-s2 (`results/critic_v1_resnet56_s1_s2`). **Corroboration pairs:** hub_st3-s1 and hub_st3-s2
  (`results/critic_v1_resnet56_s0_s1_s2` [0,1], [0,2]). **Twin:** `results/critic_v1_resnet56_hub_noise`.
- **No depth-56 verdict when:** R0 fails; or KILL-56, i.e. a Stage 1 KILL condition holds in s1-s2 AND in a hub pair
  (penult adjacency < 0.50; id_profile spearman < 0.80 or shift >= 2; >= 2 sharp factors with MAD > 0.10;
  commit/class shift >= 2; cka_test < 0.60); or T56 fails a core item (or the twin is missing); or the instrument gate.
  For row 9 (the M56-b tag) the M56 notes apply: R0 FAIL and the instrument gate give no verdict; KILL-56 and a T56
  core FAIL or missing twin void only a PROMOTE.
- **Core items on s1-s2** (tolerances unchanged): S1 `id_profile[0,1]`, S3 `penult/class_centers.sep_ratio`, S5
  `penult/adjacency_spearman[0,1]`, S7 max `mean_abs_delta` of `decod/<factor>[0,1]` over the 10 sharp factors
  (<= 0.10), S8 `commit/class`, S9 `penult/cka_test[0,1]`. **The critic.json statuses decide** (review item 6);
  `a4b_eval.js` recomputes each item with the critic's formulas, and any mismatch is reported and blocks promotion
  until it is explained in SESSION.md. Non-core FAILs are INFO. There are no depth-56 at-risk lists, and none are
  invented after the data.
- **Per row:** "d56 ✅ (s1, s2)" iff the row's claim holds in s1 AND s2 and its core item passes, and nothing blocks
  promotion (as for row 11). Row 1: C1 + S1. Row 3: C2 + S7. Row 4: C3 + S8. Row 5: C4. Row 6a: C5a. Row 7: S5 and
  D1(s1, s2) >= 0.80.
  - A claim that holds in one seed: d56 🟡 (1 of 2). One that fails in both: d56 ✗ (does not transfer). Holds in both
    with its core item FAIL: d56 🟡.
  - Row 7 otherwise: d56 🟡 if D1 >= 0.80 in a hub pair, else d56 ✗.
  - A missing claim or core value: NOT_EVALUABLE (missing) for that row.
  - Rows 2 and 8 stay 🟡 (🟡 at depth 20). ATLAS_STATUS has no cka row: S9 at depth 56 is recorded as INFO in row 8's
    evidence (review item 21, D12). Row 9 is covered by M56.
  - The claim text is not widened, and the resnet20 status is untouched.

## Scale labels (the Stage 2 open points)

- **SCALE-VARIANT does not override the ROBUST column.** Each row gets two separate labels:
  - SHAPE, from the table below: SCALE-ROBUST, SCALE-FRAGILE, COLLAPSE-ARTIFACT or UNDECIDED.
  - LEVEL, from the pre-registered level fields only: D-ID / D-COLL / [ACC] outcomes, and the pair levels for row 7
    and S9. An atlas.json value outside the band that is not a pre-registered field is INFO (Stage 2's reading T+).
  - "SCALE-VARIANT" is retired as a word. The status column shows SHAPE, and the evidence names the LEVEL.
- **SHAPE is computed on X** = the 10 pairs (resnet20 {hub, s1, s2, s3, s4}_st3 x resnet56 {s1, s2}): medians for
  continuous pair items, the maximum for commit shifts. The hub-only and single-reference readings are INFO. This
  removes row 7's reference dependence: Stage 2's three references were three draws of the same resnet20 seed
  spread.

| row | SCALE-ROBUST | SCALE-FRAGILE |
|---|---|---|
| 1 | C1 in s1 and s2; median X id_profile spearman >= 0.90; max X peak shift <= 1 | C1 fails in s1, s2 and M11, or median spearman < 0.80, or max shift >= 2 |
| 3 | C2 in s1 and s2; >= 9 of the 10 sharp factors with median X MAD <= 0.10 and median X rho >= 0.70 | C2 fails in s1, s2 and M11, or >= 2 sharp factors with median X MAD > 0.10 |
| 4 | C3 in s1 and s2; max X commit shift <= 1 | C3 fails in s1 and s2, or max X shift >= 2 |
| 5 | C4 6/6 in s1 and s2 | < 6/6 in s1, s2 and M11 |
| 6a | C5a 8/8 in s1 and s2 | < 8/8 in s1, s2 and M11 |
| 7 | median X penult rho >= 0.70 and median X D1 >= 0.80 | median X D1 < 0.70 and median X aligned-layer3.1 rho < 0.70; COLLAPSE-ARTIFACT instead when median X D1 < 0.80, median X layer3.1 >= 0.80 and both r56 CVs < the CV band low edge |
| 8 | median X relrep argmax agreement / chance >= 3 | < 3 |
| S9 (INFO in row 8) | median X cka_test >= 0.80 | < 0.60 |

Anything else is UNDECIDED. **COLLAPSE-ARTIFACT is a diagnosis, not SCALE-ROBUST:** row 7 gets no scale-robust
statement, and it does not change the d56 tag (review item 9).

- **Pair LEVEL** for m in {D1, penult adjacency rho, cka_test, aligned layer3.1 rho}:
  - AGREE: median X >= e20(m).
  - DEPTH-GAP: median X < e20 and W56 (s1-s2) >= e20. The depth-56 seeds agree with each other like resnet20 seeds
    do, but not with resnet20.
  - DEPTH-56-UNSTABLE: median X < e20 and W56 < e20. The depth-56 penult is itself less seed-stable, so the
    cross-depth gap is not a depth mismatch. For D1, "(COLLAPSE)" is added when both r56 CVs are below the CV edge.
  - Matched reading with XM = resnet20 x {M11, Mc} (interpolation branch: per resnet20 reference, lo + t (hi - lo) of
    the bracketing rungs): median XM >= e20 while X is not AGREE means "NOT-DEPTH at matched accuracy".

## M56: margin at depth 56 (A3 E5 follow-up)

It adds three things E5 could not give:
- two last-epoch depth-56 instances; the hub is test-selected and alone;
- the accuracy-matched margin that "improves with scale" needs (MASTER_SUMMARY.md:228, "confounded by model accuracy";
  STAGE2.md A3 row: "no matched-rung margin");
- the status at depth 56 of row 9's M2 clause, "margin beats distance". This is the same criterion as B1's ViT exit
  ("margin > absolute distance", MASTER_SUMMARY.md:240). The hub already fails it (margin - dist -0.0008, p 0.83), so
  B1's reading needs to know whether this is depth, fit or a single instance.

Stage B only, about 10 s per rebuild (`provenance.json` of the A3 rebuilds: 9.3-9.5 s). The rules are in the M56
notes: the A3 grammar (M2, M3, M4) on s1 and s2 gives row 9 its depth-56 tag, which names the clauses it covers (M2,
M3, M4) and carries M56-c beside it (row 9's claim includes "margin ~ maxprob"). M4 is the critic's
`penult/margin_typeb.auc_margin_typeb` status in `critic_margin_v1_resnet56_s1_s2`. The M11 and Mc rebuilds attribute
an M2 failure: NOT-DEPTH if M2 holds in all of them, DEPTH if it fails in all, SPLIT otherwise; in the interpolation
branch the attribution and the M56-a DEPTH reading are NOT_EVALUABLE. The E1 qualifier and "margin beyond confidence"
are exploratory: margin_typeb has no paired test of margin - maxprob, so the lead is descriptive, and float32 maxprob
ties are tested first (rule 8, `scripts/maxprob_ties.py`): TIES-EXCLUDED / TIE-ARTIFACT-POSSIBLE / NO-LEAD. B1's
same-session E9 rebuilds `margin_b1_resnet56_s{1,2}` add the first paired test of that gap at depth 56
(`margin_minus_maxprob_confmatched` and its `_p`), read beside M56-c as INFO (D9); this is part of the one margin
touch of s1 and s2.

**Joint reading of A4b M56-b and B1** (D13; the same text is in the B1 notes; the two labels are computed
independently, and `scripts/b1_verdicts.js --a4b results/atlas_v1_resnet56_s1/a4b_eval.json` prints the reading from
`M56.b_row9_d56.tag` and `.m2_failure_attribution`):

| A4b M56-b | B1 label | joint reading (roadmap consequence) |
|---|---|---|
| any | INVALID-* / UNDECIDED-POWER | no A/B label. With G3 FAIL and M56-b REJECT, it is recorded as "the atlas-definition margin > distance does not hold in fully fit CNNs"; MASTER exit clause 1 goes to D-AB |
| any | A | outcome A as pre-registered; M56-b reported beside it |
| PROMOTE or YELLOW | B | "margin loses its advantage over distance on ViT"; B2 is cancelled and Phase 3 is CNN-scoped only if V1 is NULL(REVERSED) or V2 FAILs on both ViTs (prereg review item 21) |
| REJECT, attribution NOT-DEPTH | B | not evidence for "CNN-specific": the clause also fails in a fully fit CNN. The decision goes to D-AB with V3 and `row(c*).auc_margin_typeb` of both ViTs and resnet50; B2 is not cancelled on this basis |
| REJECT, attribution DEPTH, SPLIT or null | B | recorded as "depth/architecture-dependent"; the B2 rule is as in row 3 |
| any | MIXED / UNDECIDED | as pre-registered; M56-b beside it |

## Provenance (G0e) and relaunch rules

- **G0e** (review item 5; D17; the `docs/plans/STAGE1.md` amendment 2 pattern): every A4b dump's `meta.git_commit`
  and every A4b `provenance.json` `stage_b_git_commit` equal P_run, the commit the pod ran; P_run is P or a
  descendant of it; `git diff --quiet P P_run --` over the frozen list holds; and `meta.created` (pod clock, UTC) is
  later than P's commit time in UTC. The frozen list: every A4b and B1 manifest, `docs/plans/{STAGE2B,STAGE2,
  B1_VIT_MARGIN}.md`, `scripts/{a4b_eval.js,b1_verdicts.js,b1_gate.py}`, `experiments/tolerances_default.yaml` and
  `ATLAS_STATUS.md`. Pre-launch fixes after P may touch code only. `node scripts/a4b_eval.js --p P --p-run P_run`
  checks all of it; a G0e failure blocks every promotion.
- **No `git` operation on the pod while `pod_atlas.sh` runs** (bash reads the script as it goes, and later `python -m`
  calls would import new code). `ATLAS_REBUILD` must be unset (`block_a4b` refuses it). A relaunch sets
  `ATLAS_A4B_CHECK_DIR=results/instrument_check_a4b_r2` (and B1's `ATLAS_B1_CHECK_DIR`); the preflight diverts its own
  record to scratch; trainings with a checkpoint and train.json, built atlases, compares and critics are skipped.
- An evaluator bug found after the run is fixed only by a committed amendment listed in SESSION.md, and the outcomes
  are then reported under both versions.

## Pod: `pod_atlas.sh --a4b`

Session order (D5): prelude (install, CIFAR data, B1 data if `--b1`, GPU preflight, smoke tests), the margin preflight
(D6, hard), Stage 0 (skipped: committed), `block_a4b`, then `block_b1`. A4b's same-session `_st3` dumps are B1's
rule-6 CIFAR anchor (D9), and B1 never shares the GPU with a training.

`block_a4b`, one block with two lanes:
- **Gates first:** a fresh check dir, `ATLAS_REBUILD` unset, the st2 dumps and the resnet20 checkpoints on the volume,
  G0d (torch/GPU and python/numpy/scipy/sklearn, `versions.json`) and the D7 code-diff gate. Any failure stops the
  block before any training.
- **GPU lane (background):** e50, e60, e70 (seed 11). Then the rule, and e90 only if every rung is below the window.
  Then s12m and s13m at E*, and s1 and s2 (200 epochs each). One training at a time, `min(8, CPUs - 2)` workers. An
  `EXIT` trap kills the lane and its training if the block ends early (review item 17), so no training outlives it.
- **CPU lane (foreground):** G0c's rebuild, then the eight atlases that need no new checkpoint (resnet20 band first,
  then hub_st3, the twin, e40_st3), then each new checkpoint's atlas as soon as its `train.json` exists. A failed
  training is skipped as soon as `train_one` logs it. `scripts/matched_rung.py` writes `ladder.json` atomically, also
  on bad input (status `error`), so the lane never reads a partial file or idles on a missing one. Then the compares
  (G0c same-space; P20; resnet20 x every resnet56 rung, replicate and seed, so the interpolation branch has its
  matched pairs; W56; hub pairs; T56), the critics, dump-meta (s1, s2, s12m, s13m against the hub_st3 dump, their
  sha256 also against every rung's), the margin rebuilds and the tie check.

Every step after the gates is `soft`, compares and critics are `*_once`, and atlases are skipped when built.

| step | basis | estimate |
|---|---|---|
| GPU lane: e50 + e60 + e70 | 180 epochs x 3.45-3.52 s | 10.4-10.6 min |
| GPU lane: s12m + s13m at E* in {50, 60, 70} | 100-140 epochs | 5.8-8.2 min |
| GPU lane: s1 + s2 | 400 epochs | 23.0-23.5 min |
| GPU lane total (a few % for the CPU lane beside it) | | about 40-44 min (+5.2 min if e90) |
| CPU lane: 8 + 7 atlases (80-84 s each) | overlapped with the GPU lane, except s2's | +1.4 min after the lane |
| up to ~69 compares, 7 critics, dump-meta | Stage 2's 15 compares + 8 critics took <= 46 s | about 2-3 min |
| 14 margin rebuilds, ties | 9.3-9.5 s each | about 2.5 min |
| **A4b block** | | **about 44-52 min, $0.54-0.64 at $0.74/h** (plus the shared prelude and preflight) |

Volume: 15 new dumps (16 with e90) of about 0.09 GB each, about 1.36-1.45 GB (109,064 rows x 416 float16 dims per
stride-5 or resnet20 dump, plus factors). The session total with B1 is in the RUN_REQUEST (pre-launch `du` <= 28 GB).

## Local evaluation

`node scripts/a4b_eval.js --p <P> --p-run <P_run> --json results/atlas_v1_resnet56_s1/a4b_eval.json` computes everything
above from the pulled files. D1 is `scripts/d1_distance_only.js`'s formula; adjacency, id_profile, decod and commit
shift are the critic's formulas. Before the commit: `node scripts/a4b_eval.js --dry-run` (Stage 2 numbers) and
`bash scripts/a4b_eval_fixture.sh` (all scenarios PASS). SESSION.md: `results/atlas_v1_resnet56_s1/SESSION.md`. A4b
and B1 are evaluated independently; neither decision waits for the other.

## Not covered

- Stride-1 profiles of s1/s2 (no `_b1`).
- A second full ladder seed. Seeds 12 and 13 are trained only at E*.
- A rung that matches resnet20 in both accuracy and training fit (train-reference accuracy 0.9994-0.9997). With this
  recipe it probably does not exist: fit is 0.9936 at E = 40, and the curve models put the accuracy window's upper
  edge near E = 70-80. The two legs above replace it.
- A depth-56 null re-run.
- Corrupt-split margins.
