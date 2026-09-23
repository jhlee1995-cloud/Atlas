# Stage 2 plan (A4): scale transfer resnet20 -> resnet56 (hub, norm chenyaofo)

Committed before the run. Predictions R56-0..9 live in `experiments/queue/atlas_v1_resnet56_s0hub.yaml` (notes;
R56-5c and F1-F4 in `atlas_v1_resnet56_s0hub_b1.yaml`); this file holds the design, the attribution rule and the
per-row scale labels the Evaluator applies. Stage 2 runs in the same pod session as Stage 1b and A3, and its
results are evaluated only after the Stage 1b decision is committed (`docs/plans/STAGE1.md` amendment 2 item 7).

## Facts that shape the design

- **Normalization.** The hub resnet56 training log
  (`https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar10/resnet56/default.log`) uses the
  resnet20 settings: std (0.2023, 0.1994, 0.201) (log line 59), 200 epochs, batch 256, SGD nesterov lr 0.1 wd 5e-4,
  cosine T_max 200 (line 77). So `norm: chenyaofo` is right for resnet56 too.
- **The hub checkpoints are best-epoch, i.e. selected on the test set.** Log line 5287: `best val top1-acc=94.37%
  (epoch=198)`; line 5286: last epoch (200) 94.21%. The val set is the CIFAR-10 test set. The resnet20 log shows
  best 92.60 (epoch 183) and last 92.56; `results/norm_check_resnet20/norm_check.json` `chenyaofo.acc_10k` = 0.9259
  is 1 image from the best epoch and 3 from the last. The hub point on the accuracy line is therefore
  test-selected (at most 0.16 pt above its own last epoch), while the ladder rungs are last-epoch
  (`scripts/train_second_seed.py`, no test-set selection). Accuracy keys: hub = `results/norm_check_resnet56/
  norm_check.json` `chenyaofo.acc_10k` (`scripts/check_norm.py`); the target 0.9259 = the same key in
  `results/norm_check_resnet20/norm_check.json`; rungs = `results/train_resnet56_e{10,20,40}/train.json`
  `final_test_acc_10k`. All are 10k accuracies; `meta.accuracy.test` (5k) is reported, not used for matching.
- **Taps.** `BlockHooks` keeps block i of a stage if `i % stride == 0` or it is the stage's last block
  (`atlas/extract_acts.py:66-73`), plus stem and penult. Penult is the GAP of the last block, so resnet56 `layer3.8`
  aliases penult as resnet20 `layer3.2` does.

| resnet56 stride | taps | names shared with resnet20 (threshold 6.6) | `compare.match_layers` | pairs (resnet20 -> resnet56) |
|---|---|---|---|---|
| 1 | 29 | 11 | name (trap) | layer3.2 -> layer3.2 (block 21/27); 18 taps unused |
| 2 | 17 | 8 | name (trap) | layer1.1/2.1/3.1 dropped; layer3.2 -> layer3.2 |
| 3 | 14 | 5 | position | 1.1 -> 1.6, 2.1 -> 2.3, 3.1 -> 3.3 (inconsistent) |
| 4 | 11 | 5 | position | 1.1 -> 1.4 (relative depth 0.19/0.52/0.85 vs 0.22/0.56/0.89) |
| **5** | **11** | 5 | **position** | **1.1 -> 1.5, 2.1 -> 2.5, 3.1 -> 3.5 (k/9 exact); s.2 -> s.8 (exact); s.0 -> s.0 (downsampling role)** |

  Stride 5 is used. resnet20's ID peak and class commit are both at `layer3.1` (8/9 of the blocks); stride 5 pairs
  it with `layer3.5` (24/27 = 8/9). The queued `atlas_v0_resnet56_cifar10.yaml` is SUPERSEDED (no `norm`, stride 3).
- **Two traps, both closed in code.** (1) At stride 1 or 2 `match_layers` pairs by name across depths;
  `compare()` now refuses a name match between different `meta.arch` with unequal tap counts (same-arch pairs at
  different strides stay allowed; `match_layers` itself is unchanged, and `atlas/ladder.py` compares identical
  layer lists). (2) The critic matches by name; `python -m atlas.critic ... --align position` relabels the deeper
  run onto the first run's names by index and refuses unequal tap counts or cross-stage pairs. resnet20 must be
  the first `--results` dir. After alignment `layer3.8` becomes `layer3.2` and the existing alias collapse counts
  it once as penult, so there are 10 distinct layers, as in Stage 1.

## Runs

| run | weights | role |
|---|---|---|
| `atlas_v1_resnet56_s0hub` | hub, block_stride 5 | B (11 taps, positional match) |
| `atlas_v1_resnet20_{s0hub,s1,s2}_st2` | Stage 1 weights | A and the band B20, re-measured this session at this commit (rule 6) |
| `atlas_v1_resnet56_e{10,20,40}` | local, hub recipe, seed 11, E epochs full cosine, last epoch | accuracy ladder; the matched rung is the control |
| `atlas_v1_resnet56_rand` | random init + BN recal, seed 99 | null at depth 56 |
| `atlas_v1_resnet56_s0hub_b1` | hub, block_stride 1 | within-model fine profile; never compared with resnet20 |
| `margin_v1_resnet56_s0hub` (A3) | hub dump above, Stage B only | margin at depth 56 (A3 item E5, exploratory) |

Every Stage 2 atlas uses `invariants: ["-margin_typeb"]` (everything registered except the A3 margin, in registry
order, i.e. exactly the Stage 1 set); margin at depth 56 comes only from the A3 rebuild.

## Rule 6 and the band

- **B20 = {`atlas_v1_resnet20_s0hub_st2`, `_s1_st2`, `_s2_st2`}, exactly.** s3 and s4 never enter A4 (they are
  Stage 1b's confirmation material). Every resnet20 <-> resnet56 item uses the `_st2` directories (with their own
  `dump/`) as A. Band sides: min, max and w = max - min over B20; "outside on the high side" = above max + w,
  "on the low side" = below min - w, "inside" = within [min - w, max + w]. Stage 1 values quoted in the notes are
  expected values, not thresholds.
- **R56-0c (recorded, not a gate)** compares `_s0hub_st2` with the Stage 1 hub atlas (`--same-space`). If it fails,
  no Stage 1 value is quoted beside a Stage 2 value; verdicts are unaffected.
- **Stage 1b does not need a `_st2` set.** The A3 `margin_typeb` draws nothing from `ctx.rng`, has cost
  "expensive" and is registered last, so no Stage 1 estimator draw moves; Stage 1b's I1 check
  (`scripts/check_rebuild.py`) verifies this on the s1 dump. Stage 1b compares s3/s4 with the committed s1/s2
  atlases; the `_st2` set belongs to A4 only.
- **Holdout.** Decodability pools include the five confirmation corruptions (`atlas/invariants/decodability.py:30`),
  so only the seed/depth axis is a clean holdout for R56-4; confirmation-corruption values seen here seed no new
  claims.

## Accuracy control and attribution rule ([ACC] fields)

- **Ladder.** `train_second_seed.py --arch cifar10_resnet56 --seed 11 --norm chenyaofo --epochs E`, E in {10, 20,
  40}: the hub recipe with cosine T_max = E, last epoch kept. Seed 11 keeps depth-56 seeds 1 and 2 free for a later
  full-recipe replication. The E values are guesses meant to bracket 0.9259; the rule does not depend on them.
- **Matched value m\*** of a field m:
  1. if some rung has |acc_E - 0.9259| <= 0.005 (the R0 gap), m\* is the value of the nearest such rung; on a tie,
     the longer E;
  2. else sort {e10, e20, e40, r56 hub} by acc_10k and interpolate m linearly between the adjacent pair whose
     accuracies bracket 0.9259 (this also covers a ladder whose order R56-9a fails);
  3. else NOT_EVALUABLE.
  Boolean claim clauses use the nearest rung only and report its delta accuracy.
- **Labels.** DEPTH: m(r56 hub) and m\* both outside B20, on the same side. NOT-DEPTH (accuracy or training fit):
  m(r56 hub) outside, m\* not outside on that side. UNRESOLVED: m(r56 hub) inside but m\* outside, or anything else.
- **Caveat.** Short schedules also mean less training fit and less collapse on the train reference, which pushes
  the rungs toward resnet20-like collapse values. NOT-DEPTH therefore means "not attributable to depth", never
  "caused by accuracy"; DEPTH is the conservative finding. `meta.accuracy.ref` of every rung is reported as a
  covariate. The hub point is test-selected (see above).

## Per-row scale labels

One depth-56 instance: rows stay 🟡 and nothing is promoted to ✅ from Stage 2. SCALE-VARIANT = the claim holds but
a value leaves the band (attribute it with the rule above). INVALID = any R56-0a/b gate fails. Any outcome not
covered by the ROBUST or FRAGILE column is UNDECIDED (reported, no label). The critic's count verdict is advisory:
scalar_stability FAILs across depth are expected.

| row | item (field) | SCALE-ROBUST | SCALE-FRAGILE |
|---|---|---|---|
| 1 | ID profile (`twonn_id.id`; critic `--align position` `id_profile[0,1]`) | C1 holds in r56 hub; spearman >= 0.90 and shift <= 1 | C1 fails in r56 hub and in the matched rung, or spearman < 0.80 or shift >= 2 |
| 2 | penult sep_ratio | value entry only: r56 value + DEPTH / NOT-DEPTH / UNRESOLVED | n/a |
| 3 | washout C2; decod profiles of the 10 sharp non-at-risk factors | C2 holds and <= 1 of the 10 FAIL (R56-4b: >= 9 of 10 PASS) | C2 fails in hub and matched rung, OR >= 2 of the 10 have MAD > 0.10 |
| 4 | commit/class | in stage 3 and aligned shift <= 1 | outside stage 3, or aligned shift >= 2 |
| 5 | C4 coherence | 6/6 in r56 hub | a comparison reverses in r56 hub and in the matched rung |
| 6a | C5a (sparse_frac s5 > s1, 8 non-noise corruptions) | 8/8 in r56 hub | < 8/8 in r56 hub and in the matched rung |
| 7 | penult adjacency (+ distance-only, + aligned layer3.1) | rho >= 0.70 and distance-only >= 0.80 | distance-only < 0.70 and aligned layer3.1 < 0.70; if only penult drops and the r56 penult center-distance CV < 0.068, label COLLAPSE-ARTIFACT instead |
| 8 | relrep CIFAR-100 | argmax agreement >= 3x chance | < 3x chance |
| - | penult cka_test | >= 0.80 | < 0.60 |
| A3 | margin AUC (E5, exploratory) | >= 0.80 in both, abs delta <= 0.05 | r56 lower by > 0.05 (INFO: no matched-rung margin) |

## Pod

`pod_atlas.sh --stage2` (block `block_stage2`), in one session with `--stage1b --a3`; the exact command is in
`results/atlas_v1_resnet20_s3/RUN_REQUEST.md`. Order: `check_norm.py --arch cifar10_resnet56` (skipped if its
output exists), the ladder e40, e20, e10 and the null trained one at a time AFTER the Stage 1b trainings (never
overlapping; `(CPUs - 2)` data workers, at most 8), the atlases (resnet20 `_st2` first), 15 compares, 8 critics.
Every compare writes into a new B directory and every critic into a new directory; every step is wrapped in
`soft()`, so one failure is logged and the rest still runs. On a relaunch, compares and critics whose outputs
already exist are skipped (append-only; the dumps stay on the volume); a single command that has to be re-run by
hand uses `--out <new dir>`.

| step | estimate |
|---|---|
| check_norm resnet56 | ~0.5 min |
| ladder e40 + e20 + e10 (70 epochs, one at a time) + null | ~3-6 min |
| resnet20 `_st2` x 3 | ~4.3 min (Stage 1: 82-85 s each) |
| resnet56 hub, e10, e20, e40, rand (11 taps) | ~8 min |
| resnet56 `_b1` (29 taps) | ~2.5 min |
| 15 compares + 8 critics | ~3-4 min |
| **Stage 2 block** | **~21-25 min** |

## Local evaluation (node, from the pulled JSON)

Distance-only adjacency: `node scripts/d1_distance_only.js atlas_v1_resnet20_s0hub_st2:atlas_v1_resnet56_s0hub`
(also `_s1_st2`, `_s2_st2`, and the band pairs). Bridge ratio: mean off-diagonal distance between
`class_centers.centers` / mean `class_centers.radius`. Center-distance CV: std / mean of the 45 penult center
distances. The attribution interpolation as above.

## Not covered

- Estimator noise at depth 56 (no reference-resample twin at resnet56); an `atlas_v1_resnet56_s0hub_ref1` would
  cost ~1.6 min.
- A full-recipe depth-56 replicate (resnet56 s1, 200 epochs, +5-14 min).

## Outcome

Stage 2 is **VALID and promotes nothing** (`results/atlas_v1_resnet56_s0hub/SESSION.md`). Of 31 entries, 28 PASS, 2 FAIL
(R56-1c, R56-9c) and 1 is recorded (F3). R56-8 was scored from A3 E5 after `a30729f`.
- **Transfers to depth 56:** C1-C5a, the class commit (stage 3), cross-model agreement (penult cka_test 0.863) and
  the margin AUC (0.907).
- **Depth-specific at matched accuracy (DEPTH):**
  - Penult ID is higher: the hub and every rung sit above the band.
  - Penult collapse is stronger (sep_ratio, bridge ratio, nc1). This label rests on interpolating between e40 and the
    test-selected hub, because e40 alone lies on the other side of the band.
- **Ladder miss:** no rung landed within 0.005 of 0.9259 (e40 −0.0067).
- **Row 7** is SCALE-ROBUST against the hub reference only. Against the s2 reference, D1 is 0.771, which makes the
  row UNDECIDED.
- **Deviation:** Stage 2 scratch computations overlapped the Stage 1b commit by about 2.5 minutes (amendment 2 item 7).
  This is disclosed in the SESSION.md.
- **Firming up the DEPTH labels** needs a new pre-registration first: a rung within 0.005, the full-recipe depth-56
  seeds 1-2, and the `_ref1` twin.

## Amendment 1 (A4b, committed before its run)

The follow-up that "Outcome" asks for is pre-registered in `docs/plans/STAGE2B.md`, with predictions in
`experiments/queue/atlas_v1_resnet56_s1.yaml` and `margin_v1_resnet56_s1.yaml`, and its evaluation frozen as
`scripts/a4b_eval.js`. For A4b only:
- the band is B20+ (five `_st3` members) and pair metrics get the band P20;
- the matched value never uses the hub: an in-window seed-11 rung with a built atlas, or interpolation between
  adjacent seed-11 rungs (e40 only as the same-session `atlas_v1_resnet56_e40_st3`); seeds 12 and 13 replicate the
  matched leg;
- the depth-56 side of a label comes from the last-epoch seeds s1 and s2;
- SCALE-VARIANT is replaced by separate SHAPE and LEVEL labels;
- scale labels are computed over all resnet20 references.

Stage 2's labels and its SESSION.md stay as recorded. A4b replaces the Stage 2 DEPTH entries in ATLAS_STATUS only
through STAGE2B.md's outcome tables (row 11, D-ID and D-COLL).
