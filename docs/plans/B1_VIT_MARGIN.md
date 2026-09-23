# B1: the ViT margin test (MASTER_SUMMARY.md P1)

Committed before the run, in one pre-registration commit P together with A4b (`docs/plans/STAGE2B.md`). The binding
text of every prediction, band and decision rule is the notes of `experiments/queue/margin_b1_vitb16.yaml`; this file
holds the archaeology, the design choices, the derivations and the plumbing behind them. The decision code is frozen
with P: `scripts/b1_gate.py` (on the pod, before any ViT run) and `scripts/b1_verdicts.js` (on Windows, after the pull).
Integration decisions with A4b are cited as D<n> (A4b + B1 integration, 2026-09-23); the two design reviews as
"prereg review item n" and "data review item n".

**Question.** MASTER exit clause 1 ("margin > absolute distance") and clause 2 ("type-b is low-margin") were measured on
CNNs only (CIFAR resnet20/56, ImageNet resnet50 in the legacy session). Do they hold on supervised ViT-B/16?
Outcome A: margin survives on ViTs. Outcome B: margin loses its advantage over distance on ViTs.

## 0. Summary

| exp id | model / weights | reference / test | transform | role | holdout |
|---|---|---|---|---|---|
| `margin_b1_resnet50_legacy10k` | torchvision resnet50 IMAGENET1K_V2 | legacy block (the first N ∈ [10048, 10111] val rows in the legacy loop order); ref = test (in-sample) | legacy Resize(256)/CenterCrop(224) | gate G0-G2 | discovery |
| `margin_b1_resnet50` | same | val part A / part B (25 + 25 per class) | official (resize 232, bilinear) | gate G3, c*, CNN contrast | discovery |
| `margin_b1_resnet50_swap` | same | B / A | official | split robustness (INFO on P0) | discovery |
| `margin_b1_vitb16` | torchvision vit_b_16 IMAGENET1K_V1 | A / B | official (resize 256 bilinear = the legacy transform) | primary: V1-V3 | confirmation |
| `margin_b1_vitb16_swap` | same | B / A | official | split robustness | confirmation |
| `margin_b1_deitb` | timm deit_base_patch16_224.fb_in1k (rev b78cc553, safetensors sha256 cd2da27b…) | A / B | official (resize 248, bicubic) | replication: V4 | confirmation |
| `margin_b1_deitb_swap` | same | B / A | official | split robustness | confirmation |
| `margin_b1_{resnet20_s0hub_st3, resnet56_s0hub_st3, resnet56_s1, resnet56_s2}` | CIFAR, Stage B only on A4b's same-session dumps | as in A3 | as in A3 | E9: first paired margin-vs-maxprob test on CIFAR (INFO) | discovery / A4b |

Critics: `critic_b1_vit_pair` (promotion input, read at c* through `verdicts.json info.vit_pair_at_cstar`),
`critic_b1_{resnet50,vitb16,deitb}_swap` (INFO).

Decision order: gate record, INVALID-PLUMBING (G0 or a post-gate P0), INVALID-REPRO (G1/G2), INVALID-CONTROL (G3),
UNDECIDED-POWER, then A / B / UNDECIDED / MIXED; separately margin_vs_maxprob (V3) and margin_vs_logitgap (V3b).

Predictions: G PASS; outcome A; margin_vs_maxprob MARGIN~CONFIDENCE and V3b EQUIVALENT on both ViTs (H1); resnet50 V3:
no direction.

Corrections to the premise the design started from:
1. "n ≈ 2,300 type-b" in the legacy ImageNet session is the all-wrong count (10,048 × (1 − 0.771) ≈ 2,301); the cut-0.5
   conf-wrong count was never recorded (`docs/plans/A3_MARGIN.md:26-27`, `MASTER_SUMMARY.md:195`). The G1 band therefore
   carries a measured subsample SD.
2. The legacy ImageNet energy AUC was never recorded; only the order "margin > cluster > energy" is stated
   (`imagenet_extract.py:6-7`). G2 bands margin and cluster and checks the order.
3. The legacy "first 10k" is N ∈ [10048, 10111] rows, not 10,000 (the loop flushes 64 rows at a time and checks the
   stop condition only after each parquet batch). With the mirror's footer (250 row groups of 100 rows per file) and
   Arrow's column reader, `iter_batches(64)` yields 64-row batches and N = 10,048 (data review item 10); the band is
   kept because the legacy pyarrow version is unknown (D16).
4. The A3 definition uses train-reference centers; the mirror's train split is class-sorted over 40 shards (17.6 GB),
   so B1's reference is the other half of val. This is the one definitional change, committed before any data.
5. `margin.py` could not run at ImageNet size: `center_dists` built an (N, K, D) tensor (25,000 × 1,000 × 768 × 8 B =
   154 GB). It now switches to a chunked float64 Gram form above `GRAM_MIN` = 2e8 elements; every CIFAR problem is
   ≤ 3.2e6, so the CIFAR path is the old expression literally.
6. The paired margin-vs-maxprob test did not exist (`margin.py` paired only margin − dist, and dumps stored no logits).

## 1. Facts

### 1.1 Legacy method (Upgraded-Mod@d9683cd `session_experiments/imagenet_extract.py`)

| item | lines |
|---|---|
| resnet50 IMAGENET1K_V2, avgpool hook (= the fc input) | :31-33 |
| transform Resize(256), CenterCrop(224), ImageNet mean/std | :26-27 |
| sorted glob of `datasets/imagenet_val/data/*.parquet`, `iter_batches(batch_size=64)` over every column | :40, :50 |
| PIL decode `.convert('RGB')`, undecodable rows skipped | :53-56 |
| flush per 64 decoded rows; stop after the parquet batch in which 64 × flushes ≥ n; final flush | :49-60 |
| centers = in-sample class means of classes with ≥ 3 rows | :68 |
| valley separation = mean pairwise center distance / mean per-class mean within-distance | :69-72 |
| conf-wrong = wrong & conf > 0.5; "cluster" = raw nearest-center distance; top-2 margin | :75, :77-79 |
| unseeded shuffle subsample of correct rows to the conf-wrong count; direction-free AUC | :80, :21-24 |

Legacy numbers: margin 0.800, cluster 0.636, valley separation 1.14, accuracy 0.771 (`MASTER_SUMMARY.md:193-197`;
`docs/history/session_experiments_README.md:19-20`). The mirror was read with huggingface_hub, never `datasets`.

### 1.2 Live facts (checked 2026-09-22/23)

- Mirror `benjamin-paine/imagenet-1k-256x256` at `1bd0400450249a7fe90c0aece37d0d03e7ea956a`: not gated; the card carries
  the ImageNet terms of access (non-commercial research and education). Validation = two parquet files, 351,766,666 B
  (sha256 399ae54c…) and 351,313,139 B (sha256 3c665d47…), 703.1 MB, 50,000 rows, labels 0..999 × 50 (the torchvision
  index; 300 rows checked against the official synset order, 0 mismatches). `data/` also holds 40 train shards
  (17.6 GB, class-sorted) and test shards with label 1000: only the two validation files are downloaded. Images were
  center-cropped to a square and Lanczos-resized to 256 × 256 by the uploader.
- ResNet50 IMAGENET1K_V2 acc@1 80.858 (resize 232, crop 224); ViT_B_16 IMAGENET1K_V1 81.072 (resize 256 bilinear, crop
  224; trained with "a modified version of DeiT's training recipe"); DeiT-B fb_in1k 81.98 (timm CSV; crop_pct 0.9,
  bicubic, resize floor(224 / 0.9) = 248); `model.safetensors` 346,284,714 B, sha256
  cd2da27b74ed7f68b599f16c77af3e1e80f01c75f9ad96029d22ce747a247e8e at HF revision b78cc553….
- Head input: torchvision ViT `heads.head(ln(encoder(x))[:, 0])`; timm `head(fc_norm(pool(norm(x))))` with `fc_norm`
  Identity and token pooling, i.e. `norm(x)[:, 0]`.
- Training recipes are all label-smoothed (R50 V2 0.1 + mixup/cutmix, 600 epochs; vit_b_16 0.11 + mixup/cutmix; DeiT
  0.1, mixup 0.8, cutmix 1.0).
- ReaL labels: `real.json` at google-research/reassessed-imagenet@bcd006fe (388,478 B, sha256 d83e9bff…), 50,000 lists
  (3,163 empty, 7,443 multi-label); index i = ILSVRC2012_val_{i+1}.
- Pins (D4): timm 1.0.29 (1.0.30 is one day old), huggingface_hub 1.32.0, pyarrow 25.0.1, safetensors 0.8.0; none of
  them requires numpy, scipy or scikit-learn.

## 2. Design

### 2.1 Models, transforms, accuracy control

- Every model is evaluated with its official eval transform, so the accuracy control compares like with like. The
  official accuracies lie within 1.12 points (80.858 / 81.072 / 81.98). vit_b_16's official transform equals the legacy
  one on the mirror (resize 256 is a no-op on 256 × 256 images). resnet50 differs (232 vs 256), so the gate uses its own
  legacy-transform dump and G3 and the contrast use the official one.
- **P0 accuracy bands** catch label or normalization faults: [official − 0.06, official + 0.01] for resnet50
  ([0.7486, 0.8186]) and [official − 0.10, official + 0.01] for the ViTs (vitb16 [0.7107, 0.8207], deitb
  [0.7198, 0.8298]; D14). The legacy resnet50 lost 3.8 points on this mirror (0.771 vs 80.858); torchvision's own
  ablation credits the 256 → 232 resize with only +0.224 points, so about 3.5 points are the mirror (H0). A ViT more
  sensitive to that must not be spent for a non-plumbing reason; label order is tested before the gate by the same
  loader in the resnet50 runs (G0 legacy accuracy, ReaL label check).
- **AC** (|Δacc| ≤ 0.02 against resnet50 on the test half) decides whether cross-architecture level comparisons may be
  read; V1-V3 are paired within one model and do not depend on it.
- **Replication unit:** an independently trained ImageNet-1k supervised ViT-B/16 checkpoint. vitb16 and deitb come from
  different codebases and hyperparameters but share the DeiT recipe lineage; training paradigm is B2's job (DINO, CLIP).

### 2.2 Data and split

- Rows are numbered in the order of the two validation files by name (the legacy glob order).
- Split (pre-registered): `rng = numpy.random.default_rng(0)`; for class c = 0..999 in order, the 50 rows of class c
  ascending, `p = rng.permutation(50)`; part A = `rows[p[:25]]`, part B = `rows[p[25:]]`; each part ascending.
- Primary: reference A, test B. Swap twin: reference B, test A. The swap's test images are disjoint from the primary's,
  so it is an independent-image replication within a model and absorbs the 25-per-class center noise (H6). Each run is
  its own forward pass (the parts are slices of one pass inside a run).
- Legacy block: the legacy loop replayed over the pinned files (`atlas.extract_imagenet.legacy_first_rows`, reading
  every column with `iter_batches(64)` as the legacy did); ref = test = those rows (in-sample centers).
- Manifests write `n: 25000` (split runs) and `n: null` (legacy) so `atlas/config.py` DEFAULTS cannot fill CIFAR sizes
  (prereg review item 11).

### 2.3 Taps (14 per ViT; D1)

- ViT: `block.0 … block.11` = class token of each encoder block output (before the final norm); `penult_mean` = mean of
  the 196 patch tokens after the final LayerNorm; `penult` = the head input (final-norm class token), last. The names are
  the same in vitb16 and deitb, so the critic pairs them by name. The patch-embedding tap is not taken (its class token
  is the same for every image).
- ResNet50: `BlockHooks(model, "blocks", 1, "gap")` (stem, layer1.0 … layer4.2, penult; 18 taps); the legacy dump
  `stages` (6 taps; only penult is read).

### 2.4 Definition reuse and the one change

Unchanged from A3: type-b = wrong & maxprob > cut (strict); margin d2 − d1; dist d1; Euclidean in the full pooled
layer; all-correct negatives plus the confidence-matched secondary; oriented DeLong; the 0 … 0.99 sweep; no rng draws.
The primary cut stays 0.7 with a pre-registered, count-only power fallback to 0.5 (c*). The one change: val reference
half instead of a train subset.

New atlas.json keys, only with `invariant_cfg.margin_typeb.b1: true` or `legacy_imagenet` (every A3 key keeps its
exact value; `tests/test_atlas_smoke.py` checks both):

| key | meaning |
|---|---|
| `margin_minus_maxprob_{typeb,wrong,confmatched}[_se,_p]` | paired DeLong, margin vs maxprob (own two-score call; the four-score A3 call is untouched) |
| `margin_minus_logitgap_*`, `margin_norm_minus_dist_*` | margin vs the head's top-2 logit gap (preds `logit_gap`); normalized margin vs dist |
| `auc_margin_norm_*`, `auc_logitgap_*` (+ `se_`) | AUCs of the added scores |
| `sweep[i]` additions | `margin_minus_dist_{typeb,confmatched}[_se,_p]` and all of the above per cut, `median_margin_correct_confmatched`, `strat_confmatched`; with ReaL, `*_realwrong`, `median_margin_typeb_realwrong`, `n_typeb_real_ok` |
| `strat_confmatched` | confidence-stratified AUCs (10 equal-count maxprob bins, ≥ 5 per side) |
| diagnostics | `nearest_center_agrees_with_model`, `frac_maxprob_ge_0999`, `frac_maxprob_unique`, `d1_plus_d2_cv`, `median_center_spacing_top2`, `median_margin_ratio_typeb_confmatched`, `spearman_margin_logitgap`, `spearman_maxprob_logitgap`, `centers_geometry` |
| `legacy_imagenet` (B1: penult only) | the legacy block on the scored split: `acc`, `n_cw`, `valley_sep_legacy`, `raw_/dir_auc_{margin,cluster,energy}_full` (all correct negatives) and `dir_auc_*_sub_{mean,sd,q025,q975}` over 200 seeded subsamples |

### 2.5 Numerics (D8)

float32 forward with the PyTorch TF32 defaults unchanged (legacy parity; recorded in `meta.tf32`); float32 dumps (a
float16 overflow on an intermediate ViT tap would be a post-gate plumbing failure that spends the ViT; the volume
allows it, §6); logits stored as float16 (`logits/<split>.npy`; Stage B never reads them). Stage B distances above
`GRAM_MIN` use float64 Gram blocks of 4,096 rows; BLAS threads are set to the cgroup quota for the ImageNet runs only
(D19), so E9 and A4b keep the environment of their identity checks.

### 2.6 Holdout (rule 7)

Discovery: resnet50 (all three dumps), the CIFAR E9 rebuilds, the legacy numbers. Confirmation: vitb16 and deitb plus
their swaps. No ViT activation, prediction or statistic on ImageNet val exists before P; before the gate the ViT code
runs on random tensors only (`--selftest-random`, which the gate requires, prereg review item 4). The ViTs run once,
after the gate passes in the same session; afterwards they are spent (a redesign is searched on resnet50 and confirmed
on the reserve `vit_base_patch16_224.augreg_in1k`). The val halves are not a holdout; the model is the confirmation unit.

### 2.7 Session (D5, D10)

Prelude: install (requirements.txt with the B1 pins), CIFAR data, B1 data (`scripts/b1_data.py provision`, soft,
`timeout 1200`), GPU preflight, smoke tests, the margin preflight (D6, hard: `margin.py` with the flags off reproduces
the committed A3 atlases `margin_v1_resnet{20,56}_s0hub`), then `block_a4b`, then `block_b1`:

1. refusals: an existing check dir (relaunch: `ATLAS_B1_CHECK_DIR=results/instrument_check_b1_r2`), `ATLAS_REBUILD=1`,
   a running training (B1 never shares the GPU); the B1 imports;
2. `scripts/b1_data.py verify` (hard) into `<check dir>/data.json`; the volume check (what is still to be written);
3. `pytest tests/test_b1_imagenet.py` (hard); the two ViT self-tests on random inputs (hard, each under `timeout 1200`:
   they download the weights, and torch.hub's download has no socket timeout);
4. E9 (soft): the four CIFAR Stage B rebuilds with the b1 keys and the exact A3-key identity check
   (`<check dir>/e9_identity.json`);
5. the three resnet50 runs (hard: a failure stops the block and the ViTs stay untouched), each under `timeout 1800`;
6. `scripts/b1_gate.py` → `results/b1_gate/gate.json`; only on exit 0 the four ViT runs (soft each) and the critics.

B1 runs are never rebuilt (`b1_stage` skips an existing `atlas.json`; `ATLAS_REBUILD` is refused). A closed gate returns
0 (the A3 M1 precedent).

## 3. Pre-registration (shorthand; the notes are binding)

`pm` = `per_layer.penult.margin_typeb`, `row(c)` = its sweep entry with cut c, `lg` = `pm.legacy_imagenet`.

### 3.1 Gate, bands and power

- **P0** (every run): real source, 1000 classes, parquet sha256 verified, head check ≤ 1e-3, no error; split runs also
  decode failures 0, n_test 25,000, disjoint halves, 25 per class, the accuracy band (§2.1). The legacy block records
  its decode failures (the legacy loop skipped them; prereg review item 17). A post-gate ViT P0 failure may be re-run
  once as `margin_b1_<m>_r2` for a plumbing fault only; an accuracy-band failure alone is final.
- **G0**: legacy n_test ∈ [10048, 10111], lg.acc ∈ [0.766, 0.776], lg.valley_sep_legacy ∈ [1.12, 1.16], P0 of both
  resnet50 runs, both ViT self-tests PASS, data.json PASS and (with ReaL) `label_in_real_frac` ≥ 0.5 (D15).
- **G1**: |lg.dir_auc_margin_full − 0.800| ≤ w, w = max(0.035, 2.58 · lg.dir_auc_margin_sub_sd + 0.005), and
  lg.raw_auc_margin_full < 0.5. **G2**: margin > cluster > energy (direction-free, full) and
  |lg.dir_auc_cluster_full − 0.636| ≤ max(0.04, 2.58 · sd + 0.005).
- **Band derivation** (prereg review item 8, corrected): 0.800 is one draw of the legacy's unseeded subsample of correct
  rows around the all-correct AUC. Subsampling the negatives varies V01, so SD_sub = √((Q1 − A²)/n · (1 − n/N_c)) with
  Q1 = A/(2 − A). With N_c ≈ 7,747 (10,048 × 0.771):

  | A | n_cw = 300 | 500 | 1,000 | 2,300 |
  |---|---|---|---|---|
  | 0.800: 2.58 SD + 0.005 | 0.029 | 0.023 | 0.017 | 0.012 |
  | 0.636: 2.58 SD + 0.005 | 0.041 | 0.033 | 0.024 | 0.016 |

  The floors 0.035 and 0.04 are therefore declared allowances for transform and library drift (about 1.5 × and 1.2 ×
  the width at n_cw = 500), not derived widths; the measured SD widens the band when n_cw is small (D16).
- **G3**: in `margin_b1_resnet50`, at every c ∈ {0.7, 0.5} with row(c).n_typeb ≥ 300, margin − dist ≥ 0.01 and
  p < 0.05; at least one such c. This is the atlas instrument's positive control: without it a ViT NULL cannot mean
  "not like the CNN". A3 E5 (resnet56 hub margin − dist −0.0008) shows it can fail.
- **Gate record**: `b1_verdicts.js` recomputes G0-G3 and requires them to equal `gate.json`; a relaunch reuses
  `gate.json` while both resnet50 `atlas.json` files are unchanged (prereg review item 3).
- **c\***: 0.7 if row(0.7).n_typeb ≥ 300 in resnet50, vitb16 and deitb; else 0.5 if all three have ≥ 300 there; else
  UNDECIDED-POWER. At n₊ = 300 and n₋ ≈ 20,000 the paired SE scaled from A3 is ≈ 0.005-0.006, so SESOI 0.01 is ≈ 2 SE.

### 3.2 Per-model items (at c*)

| item | key | states |
|---|---|---|
| V1 (exit 1) | row(c*).margin_minus_dist_typeb, _se, _p | PASS d ≥ 0.01 & p < 0.05 · NULL d < 0.01 & d + 1.96 se ≤ 0.02 (prereg review item 6) · NULL(REVERSED) d ≤ −0.01 & p < 0.05 · UNRESOLVED |
| V2 (exit 2) | row(c*).median_margin_typeb / pm.median_margin_correct | PASS typeb < correct and ratio ≤ 0.5, else FAIL |
| V3 | row(c*).margin_minus_maxprob_confmatched | ADDS · BELOW · EQUIVALENT (\|d3\| < 0.01 and d3 ± 1.96 se ⊂ [−0.02, 0.02]) · UNRESOLVED |
| V3b | row(c*).margin_minus_logitgap_confmatched | same rule |
| V4 | deitb V1, V2, V3 == vitb16's | PASS / FAIL |
| split | state vs the swap twin's (twins read at c* whatever their count) | SPLIT-FRAGILE if they differ, SPLIT-UNTESTED if the twin cannot give it (both = UNRESOLVED); resnet50_swap's P0 is INFO |
| AC | \|acc(ViT) − acc(resnet50)\| ≤ 0.02 | ACC-MATCHED / ACC-CONFOUNDED |

### 3.3 Decision table (`scripts/b1_verdicts.js`)

0 gate record mismatch → INVALID-PLUMBING (gate record); 1 INVALID-PLUMBING (G0 or a post-gate P0); 2 INVALID-REPRO
(G1/G2); 3 INVALID-CONTROL (G3); 4 UNDECIDED-POWER; 5 **A**: V1 PASS and V2 PASS on both ViTs; 6 **B**: V1 NULL (or
NULL(REVERSED)) on both, sub-labels by V2; 7 UNDECIDED: any V1 UNRESOLVED / SPLIT-* / not evaluable, or V2 SPLIT-*;
8 MIXED. Separately `margin_vs_maxprob` (ADDS becomes ADDS-OVER-MAXPROB-ONLY unless V3b is ADDS too, H2b) and
`margin_vs_logitgap`.

B is "margin loses its advantage over distance on ViT" (prereg review item 21): B2 is cancelled and Phase 3 is scoped
to the CNN deployment model only if V1 is NULL(REVERSED) on both ViTs or V2 FAILs on both; otherwise the roadmap
decision goes to D-AB with row(c*).auc_margin_typeb of both ViTs beside resnet50's.

**Joint reading with A4b M56-b** (D13; the same table is in `STAGE2B.md` "M56"; labels computed independently;
`node scripts/b1_verdicts.js --a4b results/atlas_v1_resnet56_s1/a4b_eval.json`):

| A4b M56-b | B1 label | joint reading (roadmap consequence) |
|---|---|---|
| any | INVALID-* / UNDECIDED-POWER | no A/B label. With G3 FAIL and M56-b REJECT, it is recorded as "the atlas-definition margin > distance does not hold in fully fit CNNs"; MASTER exit clause 1 goes to D-AB |
| any | A | outcome A as pre-registered; M56-b reported beside it |
| PROMOTE or YELLOW | B | "margin loses its advantage over distance on ViT"; B2 is cancelled and Phase 3 is CNN-scoped only if V1 is NULL(REVERSED) or V2 FAILs on both ViTs (prereg review item 21) |
| REJECT, attribution NOT-DEPTH | B | not evidence for "CNN-specific": the clause also fails in a fully fit CNN. The decision goes to D-AB with V3 and `row(c*).auc_margin_typeb` of both ViTs and resnet50; B2 is not cancelled on this basis |
| REJECT, attribution DEPTH, SPLIT or null | B | recorded as "depth/architecture-dependent"; the B2 rule is as in row 3 |
| any | MIXED / UNDECIDED | as pre-registered; M56-b beside it |

"On both ViTs" in row 3 applies to both conditions. An M56-b tag that is neither PROMOTE, YELLOW nor REJECT (not
evaluable, no verdict) leaves B read alone with row 3's rule.

### 3.4 Row 10 and the roadmap

Row 10 is added as ⬜ in P (not a row-9 extension: B1 changes dataset, reference, architecture and possibly the cut, and
B must be recordable as ✗ without touching row 9). After the run: **A** and every `info.vit_pair_at_cstar` item PASS,
`critic_b1_vit_pair` `penult/margin_typeb.auc_margin_wrong` PASS (the critic's other items are INFO at c* = 0.5,
because the critic reads the manifest cut 0.7; prereg review item 5), hygiene and provenance PASS → ✅ vitb16, deitb,
naming c* and the V3 qualifier; A with a pair item FAIL → 🟡; B → ✗; MIXED / UNDECIDED → 🟡 naming the clause;
INVALID / UNDECIDED-POWER → ⬜ with a note. Row 9 does not change. Recommended (owner decides): `MASTER_SUMMARY.md:213`
"robust across architectures" → ✅ on A or ❌ on B; `:211` adds "≈ maxprob" on MARGIN~CONFIDENCE.

A: margin becomes a Gate-1 axis for supervised CNN and ViT; B2 opens (DINO, then CLIP). With MARGIN~CONFIDENCE the
controller can trigger on maxprob or the logit gap and margin stays the geometric explanation; with ADDS margin is an
independent axis. MIXED / UNDECIDED: no branch; a follow-up needs a new pre-registration on the reserve ViT.

### 3.5 Exploratory items and rule-8 hypotheses

E1 per-block profile; E2 CLS vs mean token; E3 valley depth; E4 Spearman with the logit gap, d1 + d2 CV,
nearest-center/head agreement, `meta.head_center_cos`; E5 in-sample vs reference-A centers on the same images
(`1 − lg.raw_auc_margin_full` vs row(0.5).auc_margin_typeb in `margin_b1_resnet50`; prereg review item 10); E6
normalized margin; E7 confidence strata; E8 energy under LayerNorm; E9 CIFAR paired margin − maxprob; E10 the
confidence-matched ratio; E11 the ReaL restriction; B1-acc (INFO: official-232 minus legacy-256 accuracy on the legacy
rows, mirror minus published per model, the legacy `penult_10000.npz` cache against the legacy dump when present).

- **H0** the mirror (square crop, Lanczos, re-encode) depresses every model (B1-acc).
- **H1** margin restates the linear head: d_k² = |z|² − 2g_k with g_k = c_k·z − |c_k|²/2, so margin = 2(g(1) − g(2)) /
  (d1 + d2); if the centers are an affine image of the head weights and the biases align, margin = 2a · Δℓ / (d1 + d2),
  a monotone function of the logit gap when d1 + d2 is nearly constant (LayerNorm pushes it there on the ViT). Tests:
  V3b, E4, E7.
- **H2** type-b is selected on maxprob (V3 is confidence-matched). **H2b** saturated float32 maxprob ties can make
  margin "add" over maxprob but not over the logit gap (ADDS-OVER-MAXPROB-ONLY).
- **H3** in-sample legacy centers (≈ 10 per class) inflate 0.800 (E5). **H4** LayerNorm removes the norm component
  that helps dist on CNNs (E8). **H5** dense fine-grained regions shrink every margin (E6). **H6** 25 reference images
  per class give noisy centers (the swap twin). **H7** label smoothing compresses maxprob (the c* rule). **H8** any
  penult AUC ≥ 0.97 or margin − dist ≥ 0.15: read E11 first, then the P0 checks. **H9** many confident val errors are
  label or multi-object errors (E11).

## 4. Files (what B1 adds; D2, D3)

- `atlas/invariants/margin.py`: the only existing instrument file B1 edits. `center_dists` keeps the A3 broadcast
  expression up to `GRAM_MIN`; everything else is new functions (`_pair`, `_b1_aucs`, `_strat`, `_b1`,
  `_legacy_imagenet`) called only when a manifest sets `b1` or `legacy_imagenet`. The D6 preflight rebuilds the
  committed A3 atlases with the flags off before any block uses the file; A4b's D7 code-diff gate allows exactly this
  file.
- `atlas/extract_imagenet.py` (new): `validate_cfg`, the split and legacy-row helpers, the parquet loader, model loading
  (torchvision weights enums; timm through the pinned safetensors file), `ViTHooks`, `selftest_random`,
  `extract_imagenet`. It imports `BlockHooks`, `_resolve`, `file_sha256` and `git_commit` from `atlas/extract_acts.py`
  and edits nothing there.
- `atlas/run_imagenet.py` (new): the entry point (`python -m atlas.run_imagenet`), as `atlas.run` for CIFAR.
- `scripts/b1_data.py` (new): provisioning and verification of the pinned data and ReaL labels.
- `scripts/b1_gate.py`, `scripts/b1_verdicts.js` (new): the decision code; `tests/fixtures/b1/` holds 21 scenarios
  shared by `tests/test_b1_imagenet.py` (the gate) and `tests/fixtures/b1/check_fixtures.js` (the verdicts).
- `requirements.txt`: the four pins (D4). No `pip install` inside any block.
- Untouched: `atlas/extract_acts.py`, `atlas/run.py`, `atlas/config.py`, `atlas/context.py`, `atlas/invariants/_util.py`,
  `landmarks.py`, `adjacency.py`, `scripts/check_data.py`, `scripts/check_rebuild.py`, `atlas/compare.py`,
  `atlas/critic.py`, `experiments/tolerances_default.yaml` (the ImageNet atlas beyond margin is deferred to its own
  pre-registration; no P1 exit clause reads it).

## 5. Plumbing contract (dump → decision code)

- Layout of `atlas/context.py`; splits `ref`, `test`; `preds/<split>.npz` = argmax, maxprob (as `run_backbone`),
  logit_gap, logit_top1, logit_top2 (float32), real_ok (int8, when ReaL verified); `logits/<split>.npy` float16.
- `meta.json` (written last, so its presence marks a complete dump): `source, exp_id, arch, weights, seed_tag, layers
  (penult last), dims, splits, n_classes, dtype, pooling, norm ("imagenet"), norm_values, transform, transform_repr,
  loader_workers, batch, split, legacy_loop, ref_indices, test_indices, n_ref, n_test, per_class_counts,
  ref_test_disjoint, decode_failures, decode_failed_rows, parquet {repo, revision, files}, parquet_sha256_ok,
  head_check_max_abs (float32 penult · Wᵀ + b vs logits over every extracted row), penult_vs_final_norm_cls_max_abs,
  head_center_cos (INFO), accuracy, act_absmax, extra_preds, logits_stored, real_labels, tf32, versions, weights_sha256,
  weights_url, weights_acc1_published, timm_data_config (DeiT), device, git_commit, created`.
- Committed results carry row indices and statistics only: dumps (activations, predictions, logits) stay on the volume
  (the pull excludes `dump`). The owner approved this use of the mirror under its terms and the public repo.

## 6. Budget (integration §3; estimates)

`block_b1` ≈ 21-40 min (tests, self-tests and 0.77 GB of weights 2-4 min; E9 ≈ 1 min; 7 ImageNet runs at 2.5-5 min
each), plus ≈ 2.5 min per run if `/dev/shm` forces 0 DataLoader workers. New volume ≈ 17.4 GB (resnet50 dumps 3.44 GB
× 2, legacy 0.48 GB, ViT dumps 2.15 GB × 4, float16 logits 0.64 GB, weights 0.77 GB) plus 0.70 GB of data; the
pre-launch `du` must show ≤ 28 GB for the joint session, and `block_b1` refuses when used + still-to-write > 48 GB.

## 7. Risks

1. G3 may fail (A3 E5): B1 then ends INVALID-CONTROL with the ViTs unspent; the joint reading records it with A4b M56-b.
2. Power: all three models are label-smoothed; c* may fall back to 0.5, where V2 is easier. Row 10 names c*.
3. 25 per class is a noisy reference for shallow ImageNet valleys; the swap twin and the resnet50 contrast share it.
4. "Confidence-matched" is a truncation at the cut, not a full match (E7 reports the strata).
5. The two ViTs share DeiT-recipe lineage (replication across codebases and hyperparameters, not paradigms).
6. None of the Python was executed where it was written (no Python on the Windows machine): `pytest -q tests/` on the
   pod before the launch is its first execution; `tests/test_b1_imagenet.py` runs again inside `block_b1`.

## 8. Freeze, provenance and relaunch (D17, D18)

P holds every A4b and B1 file. Pre-launch fixes produce P_run (a descendant) and may touch code only; the manifests,
`docs/plans/{STAGE2B,STAGE2,B1_VIT_MARGIN}.md`, `scripts/{a4b_eval.js,b1_verdicts.js,b1_gate.py}`,
`experiments/tolerances_default.yaml` and `ATLAS_STATUS.md` stay as in P. `node scripts/b1_verdicts.js --p P --p-run
P_run` checks that every B1 dump and provenance names P_run, that the frozen files did not change, and that the dumps
were created after P. No `git` operation on the pod while `pod_atlas.sh` runs; `ATLAS_REBUILD` unset; a relaunch sets
`ATLAS_B1_CHECK_DIR=results/instrument_check_b1_r2`. A ViT with a complete dump but no `atlas.json` gets Stage B only; an
interrupted Stage A is extracted again once (logged); if code must change after a resnet50 atlas exists and before any
ViT atlas, all seven ImageNet runs and the gate restart as `_r2` under a committed amendment; once a ViT atlas exists,
only the P0 re-run rule applies. The launch sequence is `results/atlas_v1_resnet56_s1/RUN_REQUEST.md` (D21);
`results/margin_b1_vitb16/RUN_REQUEST.md` points to it.
