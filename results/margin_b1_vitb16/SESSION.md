# SESSION: B1 and B1b — margin_typeb on ImageNet ViT-B/16 (ATLAS_STATUS row 10)

This is the Evaluator pass on B1 and B1b.
- B1 is the pre-registered ViT margin test (MASTER_SUMMARY P1).
- B1b is the owner-approved amendment that re-anchored the one clause of B1's gate that failed.

The dumps stayed on the RunPod network volume. The pod is deleted, and its logs are archived outside the repo.

**Pre-registration and runs.**
- **B1 pre-registration.**
  - Binding text: the notes of `experiments/queue/margin_b1_vitb16.yaml`. Design: `docs/plans/B1_VIT_MARGIN.md`.
  - Decision code: `scripts/b1_gate.py` and `scripts/b1_verdicts.js`.
  - Committed with A4b as **P = `f1c3c43`** (13:02:21 UTC).
- **What the pod ran for B1.**
  - First launch: P_run `d257d91` (13:32:32 UTC), which adds ANOMALY_H1.
  - Relaunch r2: P_run `68244f7` (15:32:16 UTC). It exists on branch `b1-r2` only and equals d257d91 plus the `b1_stage` time limit (`results/margin_b1_vitb16/RELAUNCH_r2.md`).
  - The same `pod_atlas.sh` change is on main as `bf4d531`.
- **B1 record.** `results/b1_gate/gate.json` (CLOSED) and `results/margin_b1_vitb16/verdicts.json`, committed in `2ad4e9c` and `8fb1ae3`.
- **B1b pre-registration.**
  - Files: `docs/plans/B1B_AMENDMENT.md`, `scripts/b1b_gate.py`, `scripts/b1b_verdicts.js`.
  - Committed as **P_B = `98ef723`** (17:03:12 UTC), one second after the B1 verdict.
  - The pod ran P_B itself (P_B_run = `98ef723`). The results are in `67bd814`.
- **B1b verdict.**
  - Command: `node scripts/b1b_verdicts.js --p f1c3c43 --p-run d257d91,68244f7 --p-b 98ef723 --p-b-run 98ef723 --a4b results/atlas_v1_resnet56_s1/a4b_eval.json --json results/margin_b1_vitb16/verdicts_b1b.json`.
  - The file is untracked at HEAD `adf6c3c`. A fresh re-run into scratch reproduces it byte for byte.

**How it was checked.**
- Two evaluator families recomputed every item from the raw files with their own node scripts.
  - Family 1: gate, provenance and amendment.
  - Family 2: outcome, labels and exploratory items.
- A decider pass then recomputed the following from the raw files:
  - P0;
  - both gates;
  - c\*;
  - every V state at cuts 0.5-0.9, with the swap twins;
  - the labels, AC and the pair items;
  - the critic reads;
  - provenance with git.
- **No numeric or label discrepancy** with `verdicts.json` or `verdicts_b1b.json` was found. The discrepancies are in wording, in evaluator coverage and in the amendment text; they are listed below.
- Nothing in the repository was modified.

**Decisions.**
- **B1: INVALID-PLUMBING (gate G0).** There is no c\* and no A/B label; row 10 stayed ⬜ under B1. The record stands and is not rewritten.
- **B1b: outcome A at c\* = 0.5**, the pre-registered power fallback.
  - Labels:
    - margin_vs_maxprob: **ADDS-OVER-MAXPROB-ONLY**;
    - margin_vs_logitgap: **MIXED/UNRESOLVED**;
    - V4: **PASS**;
    - accuracy control: **ACC-MATCHED**.
  - ResNet50 contrast: V1 PASS, V2 PASS, V3 ADDS, V3b BELOW.
  - Joint reading with A4b: row 2, "outcome A as pre-registered; M56-b reported beside it".
- **Row 10: ⬜ → ✅ vitb16, deitb (B1b; c\* 0.5).**
  - Qualifier: "margin not shown to add over the head's top-2 logit gap (V3b MIXED/UNRESOLVED: margin ahead by +0.008 to +0.019 in all four ViT runs, ADDS in deitb, SPLIT-FRAGILE in vitb16, EQUIVALENT in 1 of 4 runs); "margin adds over maxprob" is not claimed (H2b)".
  - Every condition of the notes' row-10 rule and of AGENT_LOOP's promotion rule holds (see "Promotion").
- **Amendment caveat.** Row 10's ✅ exists only because of B1b.
  - After B1's gate was seen to fail, B1b changed the one clause that failed. The new band is centred on a value equal to the observed one, so that clause can no longer fail.
  - I judge the amendment legitimate, for three reasons:
    1. the anchor was mis-sourced;
    2. the plumbing is shown more strongly by a bitwise identity;
    3. no ViT image had been read.
  - A reader must still discount the G0 valley clause entirely. They should also know that:
    - G1 is a weak test (band ±0.057 at n_cw 122) but passed by 0.010; G2's order gap (0.011) and G3 (p 0.048) are thin;
    - outcome A rests on the cut-0.5 fallback, which ResNet50 forced. At the primary cut 0.7, the same rule gives UNDECIDED.

**Path shorthands.**

| shorthand | path |
|---|---|
| `pm(x)` | `results/margin_b1_<x>/atlas.json` `.per_layer.penult.margin_typeb` |
| `row(x, c)` | `pm(x).sweep[cut = c]` |
| `lg` | `pm(resnet50_legacy10k).legacy_imagenet` (= `g.legacy_imagenet`) |
| `g`, `gb` | `results/b1_gate/gate.json`, `results/b1b_gate/gate.json` |
| `V`, `Vb` | `results/margin_b1_vitb16/verdicts.json` (B1), `results/margin_b1_vitb16/verdicts_b1b.json` (B1b) |
| `cp` | `results/critic_b1_vit_pair/critic.json` |
| `ck` | `results/instrument_check_b1_r2/` |

## Run facts

### Timeline (UTC; commit times from git, pod times from the logs and `meta.created`)

| time | event |
|---|---|
| 13:02:21 | P `f1c3c43` (A4b + B1 pre-registration) |
| 13:32:32 | P_run `d257d91`; first launch `pod_atlas_20260923_133250.log` (`--a4b --b1 --anomaly`) |
| 14:32:28 | `margin_b1_resnet50_legacy10k` dump (10,048 rows, legacy transform); its atlas took 180 s |
| 14:37:57 | `margin_b1_resnet50` dump complete (50,000 rows, 0 decode failures) |
| 15:05:11 | `block_b1` FAILED (exit 124): `timeout 1800` killed resnet50 Stage B after 17 of 18 taps; no atlas.json |
| 15:31:46 / 15:32:16 | `bf4d531` (main) / `68244f7` (`b1-r2`): `b1_stage` limit 1800 → 7200 s, plus RELAUNCH_r2.md |
| 15:32:32 | relaunch r2 `pod_atlas_20260923_153232.log` (`--b1`, check dir `instrument_check_b1_r2`) |
| 16:02:48 | `block_b1` OK: gate CLOSED (G0 false), no ViT run |
| 16:06:12 | B1 results `2ad4e9c` |
| 17:03:11 | B1 verdict `8fb1ae3` (INVALID-PLUMBING) |
| 17:03:12 | P_B `98ef723` (B1b pre-registration) |
| 17:04:12 | B1b launch `pod_atlas_20260923_170412.log` (`--b1b`): gate OPEN, then the ViT runs |
| 17:06:58-17:31:21 | ViT dumps (vitb16, vitb16_swap, deitb, deitb_swap) |
| 17:39:03 | `block_b1b` OK |
| 17:41:25 / 17:52:38 | B1b results `67bd814` / ANOMALY_H1 official evaluation `adf6c3c` |

### First launch: the timeout
- The design estimate was 2.5-5 min per ImageNet run (B1_VIT_MARGIN.md §6). It was wrong for Stage B at n = 25,000.
  - 17 taps at 64.9-117.7 s each, plus 165 s of Stage A, used up the 1800 s per-run limit (integration D19).
  - The same Stage B took 15-28 s per tap in r2 (353.8 s in total). The r2 resnet50_swap Stage B, run right after its own Stage A, also took 49-97 s per tap (1337 s), so the cause of the slow first launch is not established (volume read caching is one candidate).
  - Build time does not enter any computed value.
- The ResNet50 run has no soft wrapper, so the block stopped before gate G.
- No ViT image was read. Only the random-input self-tests had run (`results/instrument_check_b1/selftest_*.json`, input `torch.randn`).

### Relaunch r2: a deviation from D18, accepted
- `git diff d257d91 68244f7` touches only `pod_atlas.sh` (the limit and a comment) and `RELAUNCH_r2.md`.
- **The deviation.** D18 literally required all seven ImageNet runs to restart as `_r2`, because a committed file changed after a ResNet50 atlas (legacy10k) existed. r2 instead:
  - kept the legacy10k atlas;
  - reused the complete resnet50 dump, running Stage B only, at 68244f7.
- The deviation was disclosed before any gate result or ViT data existed.
- **Nothing measured changed, as far as it can be checked:**
  - The legacy10k atlas sha256 `4e3972a7…` is the `sources` entry of both gate records and equals the committed file. The resnet50 atlas sha256 is `61951ccf…`.
  - `pm(resnet50)` has dump `meta.git_commit` d257d91 and `provenance.stage_b_git_commit` 68244f7. `atlas/` is identical at the two commits.
  - The r2 D6 preflight rebuilt the committed A3 atlases bitwise (12,371 leaves, 0 mismatches), and `tests/test_b1_imagenet.py` passed (13 tests).
  - The self-test JSONs, `data.json` and `e9_identity.json` in `instrument_check_b1` and `instrument_check_b1_r2` differ only in `created` and `git_commit`.
- The killed Stage B saved nothing to compare against. The relaunch cannot have changed a computed number.

### The B1b run
- `pod_atlas.sh --b1b` at 98ef723 did the following:
  - read B1's committed ResNet50 atlases and `ck` (self-tests, `data.json`) without changing them;
  - re-verified the data into `results/instrument_check_b1b/data.json`, which equals `ck/data.json`;
  - ran `scripts/b1b_gate.py` (OPEN);
  - then ran the four ViT runs and the critics.
- It did not re-run the D6 preflight, `tests/test_b1_imagenet.py` or the self-tests. This is harmless: no code differs between 68244f7 and 98ef723 (`atlas/`, `tests/`, `experiments/`, `scripts/b1_*` are identical; only `pod_atlas.sh`, the B1b scripts, `B1B_AMENDMENT.md` and `ATLAS_STATUS.md` differ), so the ViTs ran the code that r2 tested.
- The amendment states that `ck` is reused, but it does not mention the skipped preflight and tests.

### Provenance (every ImageNet and E9 atlas)

| run | `meta.git_commit` | `stage_b_git_commit` | `meta.created` | after |
|---|---|---|---|---|
| resnet50_legacy10k | d257d91 | d257d91 | 14:32:28 | P ✓ |
| resnet50 | d257d91 | 68244f7 | 14:37:57 | P ✓ |
| resnet50_swap | 68244f7 | 68244f7 | 15:42:17 | P ✓ |
| vitb16 / vitb16_swap | 98ef723 | 98ef723 | 17:06:58 / 17:16:24 | P_B ✓ |
| deitb / deitb_swap | 98ef723 | 98ef723 | 17:24:21 / 17:31:21 | P_B ✓ |
| E9 × 4 (`margin_b1_resnet{20,56}_*`) | d257d91 | d257d91 | A4b dumps | P ✓ |

- `Vb.provenance.b1_part` and `.b1b_part` are both PASS.
- **The B1b part's frozen-file check is empty.** It diffs the frozen list between P_B and P_B_run, which are the same commit (discrepancy 2).
- **The decider's stricter check passes except for one waived file:**
  - From P to P_B, the only file on B1's frozen list that changed is `ATLAS_STATUS.md` (`b76f6d6`: A4b rows and header; row 10 untouched).
  - From d257d91 to 98ef723, `atlas/`, `tests/`, `experiments/` and the B1 scripts are unchanged.
  - From P_B to HEAD, nothing on the frozen list, the amendment or the two B1b scripts changed, and no commit touched `ATLAS_STATUS.md`.
- `68244f7` exists only on `b1-r2` / `origin/b1-r2`. Both evaluators need that branch to resolve it.

### P0 (every run; decider recomputation)
All seven ImageNet runs PASS.
- **Every run:**
  - `meta.source` real, `n_classes` 1000, `parquet_sha256_ok` true;
  - `head_check_max_abs` 0; for the ViTs, also `penult_vs_final_norm_cls_max_abs` 0;
  - no error.
- **Split runs:**
  - `decode_failures` 0, `n_test` 25,000, `per_class_counts` [25, 25], `ref_test_disjoint` true;
  - my own index check: overlap 0, union 50,000;
  - each swap is the exact reverse of its primary, and all primaries share one test half.
- **Test-half accuracy** `pm(x).acc` against the P0 bands:

| run | acc (primary / swap) | band | published acc@1 |
|---|---|---|---|
| resnet50 | 0.77524 / 0.77976 | [0.7486, 0.8186] | 0.80858 |
| vitb16 | 0.78816 / 0.79456 | [0.7107, 0.8207] | 0.81072 |
| deitb | 0.78724 / 0.79880 | [0.7198, 0.8298] | 0.81980 |

- **Legacy block:** `meta.n_test` 10,048, `lg.acc` 0.771099, 0 decode failures.
- **Self-tests:** `ck/selftest_margin_b1_{vitb16,deitb}.json` status PASS. All 8 checks are true; the input is `torch.randn`.
- **Data:** `ck/data.json` PASS; ReaL `label_in_real_frac` 0.9002.
- **E9 identity:** `ck/e9_identity.json` shows 2156/2156 exact leaves in all four CIFAR rebuilds.

## Gate G

### B1's gate record (`g`)

| clause | band | observed | |
|---|---|---|---|
| G0 n_legacy | [10048, 10111] | 10,048 | ✓ |
| G0 `lg.acc` | [0.766, 0.776] | 0.77110 | ✓ |
| G0 `lg.valley_sep_legacy` | [1.12, 1.16] | **1.26813** | ✗ |
| G0 P0 (resnet50_legacy10k + resnet50; the swap's P0 is INFO), self-tests, data.json, ReaL ≥ 0.5 | | PASS | ✓ |
| G1 `lg.dir_auc_margin_full` | 0.800 ± 0.0573 (from `lg.dir_auc_margin_sub_sd` 0.0203 at n_cw 122) | 0.78999; raw 0.210 < 0.5 | ✓ |
| G2 order and cluster | margin > cluster > energy; cluster 0.636 ± 0.0642 | 0.790 > 0.644 > 0.633 | ✓ |
| G3 `row(resnet50, 0.7)` | n ≥ 300 to be evaluable | n 38: not evaluable | — |
| G3 `row(resnet50, 0.5)` | d ≥ 0.01 and p < 0.05 | d +0.02889, se 0.01461, p 0.0480 (n 356) | ✓ |

- **Gate CLOSED.** `V.gate.pod_record_agrees` is true.
- The decider's own gate code reproduces every flag under both bands.
- The `g.sources` sha256 values equal the committed atlases.

### The G0 diagnosis
**The failing value is correct.**
- `lg.valley_sep_legacy` 1.2681329 is the legacy formula (Upgraded-Mod `imagenet_extract.py:66-72`, implemented in `atlas/invariants/margin.py`).
- It was applied to features that equal the legacy cache bit for bit: `g.b1_acc_info.legacy_cache.max_abs_dF` is 0.0 over 10,048 rows.

**The anchor was wrong for this sample.** The in-sample valley ratio falls as the number of samples per class rises:

| samples per class | valley sep | source |
|---|---|---|
| about 10 | 1.268 | `lg` |
| 25 | 1.169 / 1.175 | `pm(resnet50).legacy_imagenet.valley_sep_legacy` and the swap |
| about 30 | 1.1423 | pod diagnosis on `penult_30000.npz` (30,056 rows; the legacy `imagenet_valley_check.py` sample) |

- So the "1.14" of MASTER_SUMMARY.md:194 is the 30-per-class value.
- The legacy margin 0.800 comes from the 10-per-class sample: 2,300 wrong and accuracy 0.771 match the 10,048-row block, and G1 reproduces the margin there.

**The 30-per-class values are supported but not archived.** 1.1423, and 1.1315 at min 30, appear in no committed file and no archived pod log. Four things support them:
- the legacy README (`docs/history/session_experiments_README.md:20`, "flat 1.13–1.14");
- the row count: 30,056 is what the legacy loop yields at n = 30,000;
- the committed 25-per-class values, which lie between the two;
- the reviewers' noise models. Fitted only to 1.1423, they predict 1.24-1.26 at about 10 per class and 1.15 at 25 (`scratchpad/b1eval/gpa/valley_model.js`, `…/family/`).

### The amendment (B1b)
**What it changed: one threshold.** `LEGACY_SEP` went from [1.12, 1.16] to [1.2481, 1.2881], i.e. 1.2681 ± 0.02.
- `scripts/b1b_gate.py` imports `b1_gate`, replaces `LEGACY_SEP` and wraps `evaluate` at call time.
- `scripts/b1b_verdicts.js` differs from `scripts/b1_verdicts.js` by exactly the three marked changes plus the `--p-b` and `--p-b-run` arguments. The three changes are:
  1. the band;
  2. the gate path `results/b1b_gate`;
  3. a two-part provenance.
- **Differential test** (family 1, `fxdiff.js`): B1's 21 fixtures, with the valley shifted by +0.1281 and the gate directory renamed, give B1's gate, labels, states and row 10 in 21/21.
- `gb` equals `g` except for `gate`, `open` and `amendment`.

**Is it post-hoc loosening?** It is post-hoc, and in effect it removes the clause rather than correcting it.
- It was written after the gate record was seen: the gate closed at 16:02 UTC and P_B is from 17:03 UTC.
- It keeps B1's half-width but re-centres the band on 1.2681. Because the features are bitwise identical, that is numerically the observed value (Δ 3.3e-5).
- So the B1b valley clause cannot fail and has zero power. B1b's gate was certain to open: G1-G3 and every other G0 clause had already passed in `g`.
- It changes no decision rule for the ViTs (see the differential test).
- It was committed before any ViT image was read. There is no ViT extraction in `launch.log` or `launch_r2.log`, and every ViT dump was created after P_B.

**Why I accept it.**
- The clause's purpose was plumbing: "the legacy block is the legacy computation". Two things meet that purpose more strongly than the clause did:
  - the bitwise identity of the features with the legacy cache;
  - the exact legacy accuracy (0.7711) and n (10,048).
- The anchor was mis-sourced in the legacy record; B1 did not mis-measure it.
- The confirmation material (the ViTs) was unspent.
- B1's record stays as it is, and B1b is reported as a separate entry.

**What a reader must discount.**
1. **G0's valley clause, entirely.** It is vacuous in B1b.
2. **The strength of G1-G3**, which the amendment calls "the scientific content of the gate".
   - **G1:** 0.790 against 0.800, inside a band widened to ±0.057 by n_cw = 122. The pre-registered band at n_cw 300-500 would have been the ±0.035 floor; the observed deviation 0.010 passes even that, so G1's weakness is low power, not a near-miss.
   - **G2:** cluster 0.644 > energy 0.633 is a gap of 0.011, within one subsample SD (0.023 / 0.026).
   - **G3:** +0.029 with p 0.048, at the only evaluable cut. ResNet50's margin − dist reverses at higher cuts:

     | cut | resnet50 | swap | n (resnet50 / swap) |
     |---|---|---|---|
     | 0.6 | −0.061 (p 0.010) | −0.058 (p 0.019) | 119 / 98 |
     | 0.7 | −0.174 | −0.140 | 38 / 35 |

     So the CNN positive control holds at cut 0.5 only. Rule 8 shows this is a selection effect, not an instrument failure.
3. **c\* was fixed before B1b was written.**
   - `g.G3_detail."0.7".n_typeb` was 38 < 300, so the ViTs would be read at 0.5 or not at all.
   - At 0.7 the same rule gives UNDECIDED: vitb16's V2 is 0.5042 against 0.4996 in its swap, which is SPLIT-FRAGILE, and V4 would FAIL.
4. **Row 10's ✅ rests on B1b alone.** B1 is INVALID-PLUMBING, and ANOMALY_H1 AH-4, which reads B1, is NOT_EVALUABLE.

**Where `B1B_AMENDMENT.md` overstates or omits.**
- **"A transcription error in the gate" (line 39).** B1 copied the legacy record faithfully. The legacy record itself attributes "valley sep 1.14" to `imagenet_extract.py` (`docs/history/session_experiments_README.md:19`, MASTER_SUMMARY.md:194-195). The error is in the legacy record.
- **"1.14 is the 30k-sample value."** This is an inference from an unarchived pod computation. It is supported, as shown above.
- **"Legacy margin 0.800 reproduced" (line 17).** The observed value is 0.790, inside a widened band.
- **"Computed from legacy material only, not from any B1 number."** True in form. The amendment says the anchor equals B1's value, but not that the clause thereby lost its content.
- **The mechanism is half-stated.** "Centre noise inflates the between-centre distances" leaves out the other half: in-sample within-distances also shrink when there are few samples per class. In family 1's model that shrinkage is about 40% of the rise.
- **It does not flag that its own diagnosis contradicts** MASTER_SUMMARY.md:210 ("sep 1.14, not a sample artifact", ✅) and legacy README:20. The legacy sweep varied the minimum class count (3 → 30) at about 30 per class, not the per-class sample size. The ratio does depend on the per-class sample size.
- **It omits that c\* ≤ 0.5 was already fixed.**
- **Its B1b provenance check is vacuous for frozen files** because P_B = P_B_run (the commit and time checks still apply). B1's frozen list from P to P_B is checked by no evaluator; the decider's manual check passes except `ATLAS_STATUS.md` (b76f6d6: A4b header and rows 1-9 and 11; row 10 unchanged), which is waived.
- **It does not disclose a second D18 deviation:** code changed after the ResNet50 atlases existed (`pod_atlas.sh` block_b1b and the two B1b scripts), and those atlases were reused instead of restarting all seven ImageNet runs as `_r2`. There is no numeric effect: the b1b gate's `sources` sha256 equal the committed atlases, and `atlas/` is identical from d257d91 to 98ef723.
- **It records no owner approval.** The approval appears only in the session record. This SESSION.md is its first committed record, and the commit that applies it is the owner's confirmation.

**Tolerance rule (AGENT_LOOP.md: "A tolerance may be changed only with a SESSION.md line saying why, and never to flip a specific entry from FAIL to PASS").**
- This paragraph is that SESSION.md line. G0's valley band changed because its anchor belonged to a different legacy sample.
- The change did flip gate G0. It did not flip a row-10 result, because no ViT result existed. B1's own entry stays INVALID-PLUMBING.

## B1 decision (the record)
**Applied in order:**
1. The gate record agrees (`V.gate.pod_record_agrees` true).
2. **INVALID-PLUMBING (gate G0)** (`V.labels.outcome`).
3. There is no c\* (`V.cstar` null) and no A/B label.

**Consequences:**
- `V.row10.status` is ⬜ ("row 10 stays proposed").
- `V.provenance.status` is PASS.
- The joint reading is row 1, "no A/B label". G3 passed, so the clause for "G3 FAIL and M56-b REJECT" does not apply.

**Reproducibility.**
- `V` at HEAD is byte-equal to the file committed in `8fb1ae3`.
- It reproduces only from that tree. Re-running `b1_verdicts.js` today gives the same outcome, but provenance FAILs (the ViT atlases carry 98ef723) and `info.*_present_with_gate_closed` becomes true.

**The prediction "G PASS" failed, on G0.**

**Verdict: CONFIRMED. B1 stands as INVALID-PLUMBING (gate G0); row 10 was ⬜ after B1.**

## B1b predictions (predicted → observed → verdict)
Values are at c\* = 0.5 unless a cut is named. "primary / swap" pairs are given with a slash.

**Gate G (`gb`).**
- Predicted: PASS.
- Observed: G0-G3 all true, OPEN; `Vb.gate.pod_record_agrees` true.
- Verdict: PASS, subject to the discount list above.

**c\*.**
- Predicted: 0.7, with the pre-registered power fallback to 0.5.
- Observed: `row(x, 0.7).n_typeb` is 38 for resnet50, 1,340 for vitb16 and 1,346 for deitb. At 0.5 the counts are 356 / 2,673 / 2,678.
- Verdict: **c\* = 0.5** (`Vb.cstar`).
- With the ViTs alone, c\* would be 0.7. The fallback is pre-registered in B1_VIT_MARGIN.md §7, risk 2: "c\* may fall back to 0.5, where V2 is easier. Row 10 names c\*".

**V1 (margin − distance, type-b against all correct).**
- Predicted: PASS in both ViTs.
- Observed:

| run | d | se | p | auc_margin / auc_dist |
|---|---|---|---|---|
| vitb16 | +0.1190 | 0.0047 | 6e-144 | 0.792 / 0.674 |
| vitb16_swap | +0.1144 | 0.0047 | 7e-133 | 0.792 / 0.678 |
| deitb | +0.1194 | 0.0046 | 1e-147 | 0.807 / 0.688 |
| deitb_swap | +0.1183 | 0.0046 | 3e-144 | 0.807 / 0.689 |

- Verdict: **CONFIRMED.** PASS in all four runs, and PASS at every cut from 0.5 to 0.8 (+0.044 to +0.119).

**V2 (type-b median margin ≤ 0.5 × the correct median).**
- Predicted: PASS.
- Observed: the ratio `row(x, 0.5).median_margin_typeb / pm(x).median_margin_correct` is 0.315 / 0.307 for vitb16 and 0.281 / 0.270 for deitb.
- Verdict: **CONFIRMED at c\*.**
- The ratio depends on the cut:

| cut | vitb16 | deitb | state |
|---|---|---|---|
| 0.6 | 0.393 / 0.398 | 0.341 / 0.321 | PASS |
| 0.7 | 0.504 / 0.500 | 0.432 / 0.404 | vitb16 SPLIT-FRAGILE |
| 0.8 | 0.644 / 0.655 | 0.571 / 0.547 | FAIL in all four |

  The same holds in row 9, where the ratio exceeds 0.5 at cut 0.99.

**V3 (margin vs maxprob, confidence-matched).**
- Predicted: EQUIVALENT (MARGIN~CONFIDENCE).
- Observed: +0.0344 / +0.0336 for vitb16 and +0.0536 / +0.0493 for deitb, all with p ≤ 4e-15.
- Verdict: **FAILED: ADDS** in all four runs.
- On all errors (cut 0), margin ≈ maxprob: `pm.margin_minus_maxprob_wrong` is +0.004 to +0.014. So ADDS is a property of the confident population. It grows with the cut: +0.05 to +0.07 at 0.6, and +0.33 to +0.41 at 0.9.

**V3b (margin vs the head's top-2 logit gap, confidence-matched).**
- Predicted: EQUIVALENT.
- Observed:
  - vitb16: +0.0105 (se 0.0032, p 0.001), which is ADDS by 0.0005;
  - vitb16_swap: +0.0076 (p 0.018), which is EQUIVALENT;
  - so vitb16 is SPLIT-FRAGILE;
  - deitb: +0.0187 / +0.0167 (p ≤ 2e-7), ADDS.
- Verdict: **FAILED.** Only one of the four runs is EQUIVALENT.
- The label is a knife-edge: at cuts 0.6, 0.7 and 0.8, V3b is ADDS in all four ViT runs.

**V4 (replication).**
- Predicted: PASS.
- Observed: the V1, V2 and V3 states are equal in vitb16 and deitb.
- Verdict: **PASS** at c\*. It would FAIL at 0.7, because of the vitb16 V2 split.

**Split robustness.**
- Only vitb16's V3b differs from its twin: `Vb.models.vitb16.V3b_own` is ADDS and `_swap` is EQUIVALENT.
- ResNet50's states at 0.5 equal its twin's.

**AC (accuracy control).**
- |Δacc| is 0.0129 for vitb16 and 0.0120 for deitb, both ≤ 0.02.
- Verdict: **ACC-MATCHED.** Accuracy is matched; calibration is not (see Rule 8).

**Labels** (`Vb.labels`, recomputed):

| label | value |
|---|---|
| outcome | **A** |
| margin_vs_maxprob | **ADDS-OVER-MAXPROB-ONLY** (V3 ADDS in both ViTs, V3b not ADDS in both: the H2b rule) |
| margin_vs_logitgap | **MIXED/UNRESOLVED** |
| V4 | **PASS** |
| accuracy control | **ACC-MATCHED** |
| resnet50 contrast V1 | PASS (+0.029 / +0.045) |
| resnet50 contrast V2 | PASS (0.479 / 0.493) |
| resnet50 contrast V3 | ADDS (+0.222 / +0.170) |
| resnet50 contrast V3b | **BELOW** (−0.046 / −0.047) |

- Outcome A was predicted and is CONFIRMED.
- margin_vs_maxprob and V3b FAILED their predictions.
- No direction was predicted for resnet50's V3.

**The same rule at other cuts** (INFO only; not the decision):

| cut | result |
|---|---|
| 0.6 | A, with plain ADDS / ADDS |
| 0.7 | UNDECIDED (vitb16 V2 SPLIT-FRAGILE) |
| 0.8 | MIXED (V2 FAIL in both ViTs) |
| 0.9 | V1 NULL in both ViTs (n 74-111, below the power floor) |

### Promotion (row 10)
**The notes' "Row 10" rule, item by item:**
- Outcome A ✓.
- Every `Vb.info.vit_pair_at_cstar` item PASSes ✓:
  - auc_margin_typeb spread 0.0146;
  - auc_dist_typeb spread 0.0142;
  - auc_maxprob_typeb spread 0.0069;
  - median ratio spread 0.0344.

  All four are ≤ 0.05.
- `cp` `penult/margin_typeb.auc_margin_wrong` PASSes (0.861 / 0.866) ✓.
- The critic's other penult items are INFO at c\* = 0.5, because the critic reads cut 0.7. For the record:
  - `median_margin_ratio_typeb` FAILs there (0.504 / 0.432, relative spread 0.155). This is the same V2 fragility.
  - The other 8 of `cp`'s 9 FAILs are at block.8-11 (the ratio) and at `penult_mean`. `penult_mean` is the mean patch token, which neither head reads: AUC 0.56 for vitb16 against 0.74 for deitb.
- `cp` synthetic_refusal PASS and holdout_hygiene PASS / PASS ✓.
- Provenance PASS ✓. The stricter manual check from P to P_B passes except `ATLAS_STATUS.md` (A4b rows; row 10 unchanged), waived.
- Result: **✅ vitb16, deitb**, naming c\* 0.5 and the V3 qualifier.

**The AGENT_LOOP.md promotion rule:**
1. **Source is real.** `meta.source` is real in every run, and `cp` synthetic_refusal PASSes ✓.
2. **Replicates across ≥ 2 independent units within tolerances.** The notes declare the replication unit as an independently trained ImageNet-1k supervised ViT-B/16 checkpoint: vitb16 and deitb. They share the DeiT-recipe lineage (risk 5). The pair items are within `tolerances_default.yaml` ✓.
3. **Holdout declared before the run.** It is in the P notes and in P_B. The ViTs' first ImageNet touch comes after both ✓.
4. **Prediction written before the run.** Outcome A, V1 and V2 are in the P notes ✓. The V3 and V3b predictions failed and are reported.

**The V3 qualifier.**
- The notes give qualifier text only for MARGIN~CONFIDENCE and for plain ADDS. H2b forbids claiming "margin adds over maxprob" unless V3b is ADDS too.
- For this case the frozen code writes "margin adds over maxprob but not over the logit gap (maxprob resolution, H2b)" (`Vb.row10.why`). That text has three problems:
  1. It makes the forbidden claim.
  2. It names a mechanism the data rule out. In every ViT run `pm.frac_maxprob_ge_0999` is 0 and `frac_maxprob_unique` ≥ 0.9955, so there are no saturated ties.
  3. It overstates "not over the logit gap". V3b is ADDS in 3 of 4 runs at c\*, and in all 4 at 0.6-0.8.
- The frozen text wins over the frozen code, as in A4b's "Frozen evaluator vs frozen text".
- Row 10 therefore states **"margin not shown to add over the head's top-2 logit gap (V3b MIXED/UNRESOLVED: margin ahead by +0.008 to +0.019 in all four ViT runs, ADDS in deitb, SPLIT-FRAGILE in vitb16, EQUIVALENT in 1 of 4 runs); "margin adds over maxprob" is not claimed (H2b)"**, with the V3 and V3b values in the evidence.

### Exploratory items (E1-E11, B1-acc; never promoted)

| item | prediction | observed (`Vb.exploratory`) | verdict |
|---|---|---|---|
| E1 per-block | penult ≥ early max + 0.10 | penult 0.792 / 0.807 / 0.676 vs early max 0.513 / 0.520 / 0.526. First tap with V1 PASS: block.7 (vitb16, vitb16_swap, deitb), block.8 (deitb_swap), layer4.2 (resnet50) | holds |
| E2 CLS vs mean token | CLS ≥ mean | vitb16 0.792 vs 0.580; deitb 0.807 vs 0.793 | holds |
| E3 valley depth | none | `sep_legacy_ref` 1.175 / 1.307 / 1.226; `sep_ratio_ref` 0.747 / 0.930 / 0.871 (resnet50 / vitb16 / deitb) | recorded |
| E4 H1 | Spearman(margin, logit gap) ≥ Spearman(margin, maxprob) in every run; d1+d2 CV lower in both ViTs | Spearman holds in all 6 (0.84-0.85 vs 0.72-0.74 ViT; 0.71 vs 0.52 R50). CV 0.250 / 0.320 vs R50 0.297: not lower in deitb | partly false |
| E5 H3 | in-sample ≥ reference-A | 0.753 ≥ 0.676 | holds: in-sample centres inflate the legacy 0.800 |
| E6 H5 | margin_norm − dist > 0 (p < 0.05) in every V1-PASS run | ViTs +0.099 to +0.106; resnet50 +0.012 (p 0.36), swap +0.020 (p 0.13) | false as written |
| E7 strata | \|margin − maxprob\| < 0.02 | 0.141-0.175 (ViT), 0.229-0.258 (R50). Within maxprob strata, margin 0.68-0.71 vs logit gap 0.60-0.64 on the ViTs; R50 0.78 vs 0.82 | FAILED |
| E8 H4 | \|energy − 0.5\| smaller in both ViTs | 0.176 / 0.170 vs 0.162 | false |
| E9 CIFAR | none | confidence-matched margin − maxprob −0.0013 (p 0.45), +0.0196, +0.0060, +0.0121 (r20 hub, r56 hub, r56 s1, r56 s2). CIFAR maxprob is saturated (68-79% ≥ 0.999; 57-59% unique values), so the H2b regime is possible there, not on ImageNet | recorded (INFO beside A4b M56-c) |
| E10 | none | ratio against confident-correct 0.294 / 0.286, 0.264 / 0.250, R50 0.365 / 0.382 | recorded |
| E11 H9 | none | ReaL-correct type-b: 46 / 44% (vitb16), 45 / 42% (deitb), 54 / 55% (R50). On ReaL-wrong type-b: auc_margin 0.84-0.85 (ViT), margin − dist +0.096 to +0.118, ratio 0.22-0.24 | removing label noise strengthens A |
| B1-acc | official − legacy ≈ +0.002 | +0.0084; mirror − published (test half) −0.0333 / −0.0226 / −0.0326; legacy cache max_abs_dF 0 | H0 supported; the +0.0084 is unexplained (INFO) |

## Frozen evaluator vs frozen text
None of these changes an outcome.

1. **Row-10 qualifier.** The code's text for ADDS-OVER-MAXPROB-ONLY contradicts H2b and the data (see "The V3 qualifier"). The text is used.
2. **Provenance, B1b part.** `b1b_verdicts.js` diffs B1's frozen list plus the B1b files between P_B and P_B_run, which are the same commit. B1's frozen list from P to P_B is never checked. The decider checked it, and it PASSes.
3. **B1's `accuracy_control`.** `V` says ACC-CONFOUNDED although no ViT ran (`b1_verdicts.js:187`). It should read not evaluable.
4. **Split rule.** It compares `base()` states, so NULL and NULL(REVERSED) count as the same state. The notes say that any differing state is SPLIT-FRAGILE. Not triggered here.
5. **Fixtures.** All 21 B1 fixtures use valley 1.139, inside the old band. The clause that closed the gate was never exercised, and B1b adds no fixtures.
6. **B1's verdict file** reproduces only from the `8fb1ae3` tree (see "B1 decision").
7. **`Vb.args`** omits `--p-b` / `--p-b-run`. They appear only in `Vb.provenance.b1b_part.p` / `.p_run`.
8. **Power estimate.** The notes (§3.1) predicted a paired SE of about 0.005-0.006 at n₊ = 300. Observed: 0.0146 for resnet50 (n 356) and 0.0156 for its swap (n 303), about 2.5× larger. This is why G3 passed by 0.002 in p.

## Rule 8 (edge, too-good and by-construction numbers)

### ViT accuracy on the mirror (H0)
- The ViTs score 0.787-0.799 on the mirror against published 0.811 / 0.820.
- Mirror minus published:

| model | test half | ref half | all 50k |
|---|---|---|---|
| vitb16 | −0.0226 | −0.0162 | −0.0194 |
| deitb | −0.0326 | −0.0210 | −0.0268 |
| resnet50 | −0.0333 | −0.0288 | −0.0311 |

- **This is not a fault.** It holds for every model, as H0 predicted.
  - The crop region equals the official pipeline's. The uploader centre-cropped to a square and Lanczos-resized to 256; vitb16's resize to 256 is then a no-op.
  - So the loss comes from the uploader's resampling and JPEG re-encode, plus DeiT's second (bicubic 248) resample.
- **Plumbing is exact:**
  - head check 0;
  - penult equals the final-norm CLS token (0);
  - ReaL agreement 0.90;
  - legacy accuracy 0.7711 reproduced;
  - all runs inside their P0 bands.
- **The D14 worry did not materialise.** The ViTs lose less than ResNet50.
- Part A is easier than part B for all three models (ref − test +0.0045 / +0.0064 / +0.0116). The swap twins absorb this.

### Why c\* fell back to 0.5: ResNet50's confidence (H7)
- The c\* rule needs ≥ 300 type-b in resnet50, vitb16 and deitb.
- ResNet50 IMAGENET1K_V2 is strongly under-confident:

  | model | errors with maxprob > 0.5 | errors > 0.7 | correct > 0.5 | correct > 0.7 |
  |---|---|---|---|---|
  | resnet50 | 6.3% (356 of 5,619) | 0.7% (38) | 29% | 4.6% |
  | ViTs | 50% | 25% | 93% | 82% |

- **H7 predicted this compression for all three label-smoothed models. It holds only for the CNN**, although DeiT also uses label smoothing and mixup/cutmix.
- The cause is not measured here. The candidates are in ResNet50's torchvision V2 training recipe: label smoothing with mixup/cutmix over a long schedule, EMA, and training at a lower resolution than evaluation.
- **Consequences:**
  - The fallback reflects the comparator's calibration, not ViT power.
  - The same cut selects very different errors: at 0.5 the ViT type-b set is half of all errors, while resnet50's is its top 6%.
  - At 0.5, `auc_maxprob_typeb` is 0.16 for resnet50: its confident errors outrank most of its correct samples on maxprob.

### The CNN positive control and the selection effect (H2)
- ResNet50's V1 passes at 0.5 only just (+0.0289 against 1.96 · se = 0.0286) and reverses at 0.6-0.7. The ViTs reverse too at the matched error quantile of about 2%, which is cut 0.9: vitb16 −0.060 / −0.073, deitb −0.027 / −0.022.
- **Against confidence-matched negatives, margin − dist is positive at every evaluable cut in every run.** At 0.5 (`row.margin_minus_dist_confmatched`): resnet50 +0.125 / +0.141, vitb16 +0.129 / +0.124, deitb +0.128 / +0.128.
- **On all errors** (`pm.margin_minus_dist_wrong`) the contrast is +0.088 / +0.091 for resnet50 and +0.087 to +0.090 for the ViTs.
- **So the reversal is a selection artifact.** Type-b is selected on maxprob and scored against all correct samples. It is not an instrument failure.
- The CNN-vs-ViT gap at c\* (+0.03 against +0.12) is a calibration and selection effect, not an architecture effect.
- The pre-registered V1 and G3 definitions carry this bias. Future designs should add a confidence-matched co-primary.

### Margin ≈ logit gap by construction (H1, A-H10)
- **The identity.** margin = 2(g₁ − g₂)/(d₁ + d₂), where g_k = c_k·z − |c_k|²/2 is the nearest-centre linear score. It becomes the head's logit gap only if the centres are an affine image of the head weights, the biases align and d₁ + d₂ is nearly constant.
- **The premises hold only loosely** (vitb16 / deitb / resnet50):

  | quantity | vitb16 | deitb | resnet50 |
  |---|---|---|---|
  | `meta.head_center_cos` | 0.78 | 0.66 | 0.56 |
  | d1+d2 CV | 0.250 | 0.320 | 0.297 |
  | nearest centre agrees with the head | 0.89 | 0.88 | 0.80 |
  | Spearman(margin, logit gap) | 0.85 | 0.84 | 0.71 |

- **On all errors** margin ≈ logit gap for the ViTs (`pm.margin_minus_logitgap_wrong` −0.001 to +0.005). It is below the logit gap for resnet50 (−0.048 / −0.053).
- **At c\*:**

  | type-b AUC | vitb16 | deitb | resnet50 |
  |---|---|---|---|
  | logit gap | 0.769 | 0.773 | 0.555 |
  | distance | 0.674 | 0.688 | 0.647 |

  So on the ViTs, V1's advantage is largely "a top-2 gap beats nearest-centre distance". The head's own gap does almost as well as margin.
- **This is the controller-relevant comparison.** On the ViTs it is unresolved (+0.008 to +0.019 in margin's favour). On resnet50 it is BELOW.
- E7 hints that margin carries information beyond confidence: within maxprob strata, margin reaches 0.68-0.71 against the logit gap's 0.60-0.64. This is exploratory and not a pre-registered comparison.

### Other rule-8 items
- **H8 is not triggered.** No penult AUC reaches 0.97, and the largest margin − dist is 0.119 < 0.15.
- **Label noise (E11).** Nearly half of the ViT "confident errors" are correct under ReaL labels. Restricting to ReaL-wrong type-b raises the margin AUC and keeps A.
- **The legacy anchors were built from in-sample centres (E5, H3).** At the atlas definition (reference-A centres, cut 0.5), resnet50 gives 0.676, not 0.800.
- **The valley diagnosis bears on MASTER_SUMMARY.md:210.** The in-sample valley ratio depends on the per-class sample size. "Shallow at ImageNet" survives at every size (1.14-1.27, out-of-sample `sep_legacy_ref` 1.175 for resnet50, against CIFAR 2.87). "Not a sample artifact" does not survive.

## Holdout
- **Discovery (every value may be read):** resnet50 (all three runs), the four E9 rebuilds and the legacy numbers.
- **Confirmation:** vitb16 and deitb and their swaps.
  - Their first ImageNet touch is in `pod_atlas_20260923_170412.log`, after the B1b gate printed OPEN.
  - Their dumps were created at 17:06:58-17:31:21, after P_B (17:03:12).
  - `launch.log` and `launch_r2.log` contain no ViT extraction. B1 and r2 ran only the random-input self-tests.
- The ResNet50 margin numbers and the fact that c\* ≤ 0.5 were known when B1b was written. B1B_AMENDMENT.md says the former, but not the latter.
- vitb16 and deitb are now spent for margin_typeb. The reserve `vit_base_patch16_224.augreg_in1k` is unspent.
- The val halves are not a holdout. The model is the confirmation unit.

## Joint reading with A4b (D13) and ANOMALY_H1
- **A4b M56-b** (`results/atlas_v1_resnet56_s1/a4b_eval.json`): tag "d56 YELLOW", `m2_failure_attribution` NOT-DEPTH.
- **With B1b outcome A this is row 2** (`Vb.joint_reading`): "outcome A as pre-registered; M56-b reported beside it".
- **Row 9's status and claim do not change.**
- **The reading.** Margin > distance fails in a fully fit depth-56 CIFAR ResNet (s2), where the penult collapses. It holds, by +0.12, in two ImageNet ViTs whose class valleys are shallow (`sep_ratio_ref` 0.87-0.93). On all errors it holds equally in the ImageNet CNN.
- **ANOMALY_H1 AH-4 is NOT_EVALUABLE (official).** It reads B1's verdict. Its INFO read with `Vb` is in `results/anomaly_h1/SESSION.md` "AH-4":
  - resnet50's lead at c\* is 0.029, below the 0.03 bar;
  - that evaluator's provenance fails by construction.
- **Addition from this pass.** The CNN-vs-ViT difference in lead at c\* is a calibration and selection effect (Rule 8). So any re-registration of AH-4 should use the all-errors or confidence-matched lead.

## Prediction misses
- **G PASS.** It failed in B1: the G0 anchor belonged to a different legacy sample.
- **c\* = 0.7.** Not reached. ResNet50 has 38 type-b at 0.7; H7 bit only the CNN.
- **V3 MARGIN~CONFIDENCE.** Observed ADDS in all four ViT runs.
- **V3b EQUIVALENT.** Observed deitb ADDS and vitb16 SPLIT-FRAGILE.
- **Exploratory:**
  - E4's CV clause (deitb 0.320 > R50 0.297);
  - E6 (fails in resnet50);
  - E7 (|Δ| 0.14-0.26, not < 0.02);
  - E8 (false);
  - B1-acc (+0.0084, not +0.002).
- **Design estimates:**
  - the paired SE at n₊ ≈ 300 was about 2.5× too small;
  - the Stage B runtime (2.5-5 min per run) was far too small, which caused the exit-124 timeout.

## ATLAS_STATUS changes
Apply these in the same commit as this SESSION.md and `results/margin_b1_vitb16/verdicts_b1b.json`. The owner's commit is the human approval that AGENT_LOOP.md asks for.

**Header.** Replace the last sentence of the header paragraph, from "B1 (docs/plans/B1_VIT_MARGIN.md: margin_typeb on two ImageNet-1k supervised ViT-B/16 checkpoints" to "row 9's status and claim do not change.", with:

> B1 (docs/plans/B1_VIT_MARGIN.md: margin_typeb on two ImageNet-1k supervised ViT-B/16 checkpoints, gated on a same-session ResNet50 legacy reproduction and positive control) ended INVALID-PLUMBING because its gate's legacy valley anchor (1.14) came from a different legacy sample (per the pod diagnosis archived in results/margin_b1_vitb16/DIAGNOSIS_G0.md); B1b (docs/plans/B1B_AMENDMENT.md), a post-hoc re-anchoring of that one clause committed before any ViT image was read, opened the gate and found outcome A at the power-fallback cut c* = 0.5, so row 10 is ✅ vitb16, deitb with the qualifier: margin not shown to add over the logit gap (results/margin_b1_vitb16/SESSION.md); row 9's status and claim do not change.

**Row 10.** Replace it with the row below. No other row changes.

| # | layer(s) | entry | claim (current) | status | evidence |
|---|---|---|---|---|---|
| 10 | penult (ImageNet; ViT = final-norm CLS, the head input) | margin_typeb on ImageNet ViT-B/16 (type-b = wrong and maxprob > c*; c* = 0.5, the pre-registered power fallback, forced by resnet50's 38 type-b at 0.7; centers = class means of the val reference half) | margin not shown to add over the head's top-2 logit gap (V3b MIXED/UNRESOLVED: margin ahead by +0.008 to +0.019 in all four ViT runs, ADDS in deitb, SPLIT-FRAGILE in vitb16, EQUIVALENT in 1 of 4 runs); "margin adds over maxprob" is not claimed (H2b): in two ImageNet-1k supervised ViT-B/16 (torchvision IMAGENET1K_V1, DeiT-B fb_in1k) at c* = 0.5 the top-2 margin (d2 − d1) to reference-half class centers separates type-b from correct test samples better than nearest-center distance by ≥ 0.01 AUC (paired DeLong p < 0.05; V1) and the type-b median margin is ≤ 0.5 × the correct median (V2); the ratio clause is cut-dependent (vitb16 0.504 at cut 0.7, > 0.5 in both ViTs at 0.8); contrast = same-session ResNet50 IMAGENET1K_V2 (discovery) | ✅ vitb16, deitb (B1b, amended G0 anchor; c* 0.5) | B1b (docs/plans/B1B_AMENDMENT.md, P_B 98ef723; results/margin_b1_vitb16/SESSION.md; node scripts/b1b_verdicts.js → results/margin_b1_vitb16/verdicts_b1b.json): outcome A, V4 PASS, ACC-MATCHED (acc 0.788 / 0.787 vs resnet50 0.775) · V1 margin − dist vitb16 +0.119 (swap +0.114), deitb +0.119 (+0.118), p ≤ 7e-133; auc_margin_typeb 0.792 / 0.807 vs dist 0.674 / 0.688; V1 PASS at every cut 0.5-0.8 · V2 ratio 0.315 / 0.281 (swaps 0.307 / 0.270) · V3 confmatched margin − maxprob +0.034 / +0.054 · V3b margin − logit gap vitb16 +0.0105 vs swap +0.0076 (SPLIT-FRAGILE), deitb +0.019 / +0.017 (ADDS); ADDS in all four ViT runs at cuts 0.6-0.8; no maxprob saturation (frac_maxprob_ge_0999 0), so H2b's mechanism does not apply · pair spreads 0.015 / 0.014 / 0.007, ratio 0.034; critic_b1_vit_pair auc_margin_wrong 0.861 / 0.866 PASS · caveats: B1 (P f1c3c43; results/margin_b1_vitb16/verdicts.json) is INVALID-PLUMBING, its gate closed on the legacy valley band (1.2681 vs [1.12, 1.16], results/b1_gate/gate.json); B1b re-anchored that band post hoc to 1.2681 ± 0.02, which equals the observed value (legacy features bitwise equal to the legacy cache), so that clause is void; G1 0.790 (band ± 0.057, n_cw 122), G2 cluster 0.644 > energy 0.633, G3 resnet50 +0.029 (p 0.048) at cut 0.5 only; c* fell to 0.5 because resnet50 has 38 type-b at 0.7 (ViTs 1340 / 1346), and at 0.7 the same rule gives UNDECIDED (vitb16 V2 0.504 / 0.500) · resnet50 contrast (discovery): V1 PASS at 0.5 only (reversed at 0.6-0.7; positive against confidence-matched negatives and on all errors, +0.088 vs ViT +0.087 / +0.090), V3 ADDS, V3b BELOW (−0.046: the logit gap beats margin) · E11: 42-46% of ViT type-b are ReaL-correct; ReaL-wrong type-b auc_margin 0.84-0.85 · mirror accuracy 0.787-0.799 vs published 0.811 / 0.820 (H0) · joint reading with row 9 M56-b (d56 YELLOW, NOT-DEPTH): outcome A as pre-registered, M56-b reported beside it · reserve ViT augreg_in1k unspent |

## MASTER_SUMMARY (recommended; the owner decides)
- **:213 "Margin robust across architectures" ⏳ → ✅** (notes §3.4 on A), with this scope: supervised ImageNet ViT-B/16 (two checkpoints, DeiT lineage), c\* 0.5, B1b. Add "margin not shown to add over the logit gap".
- **:211 "Margin is the best type-b detector".** Add that it was never compared with the head's logit gap before B1. At matched confidence the logit gap beats margin on ResNet50 (−0.046), and on the ViTs margin is ahead of the logit gap by +0.008 to +0.019 (V3b MIXED/UNRESOLVED).
- **:210 "Valleys shallow at ImageNet scale … not a sample artifact" ✅.** Keep "shallow". Drop "not a sample artifact": the in-sample ratio is 1.268 at about 10 per class, 1.169 at 25 and 1.142 at about 30.
- **:194 (valley separation 1.14).** Attribute it to the 30-per-class valley check.

## Next actions
1. **Commit.** Commit `results/margin_b1_vitb16/verdicts_b1b.json` (currently untracked), this SESSION.md and the ATLAS_STATUS changes after `adf6c3c`, then push.
   - Record the owner's approval of B1b in the commit message.
   - Keep `origin/b1-r2` (68244f7), or tag it, so that both evaluators' provenance checks can resolve it.
   - Do not re-run `b1_verdicts.js` over the committed `verdicts.json`. It would now print provenance FAIL; the `8fb1ae3` file is the record.
2. **MASTER_SUMMARY.** Apply the recommendations above if the owner agrees.
3. **Evaluator fixes** (by a committed amendment, before `b1_verdicts.js` / `b1b_verdicts.js` or a derivative is reused):
   - the P → P_B frozen-list check;
   - the ADDS-OVER-MAXPROB-ONLY qualifier;
   - `accuracy_control` not evaluable without ViTs;
   - the NULL / NULL(REVERSED) split;
   - a fixture outside the G0 valley band;
   - `--p-b` in `args`.
4. **Archive the G0 diagnosis.** If network volume `kxfir1tryb` still exists:
   - rerun the legacy formula on `cache/imagenet/penult_{10000,30000}.npz`;
   - add a per-class-n sweep, as a committed script and output;
   - record where the caches came from.
5. **B2 (DINO, then CLIP), open under outcome A. It needs a new pre-registration** that carries B1's lessons:
   - a cut rule per model, or by error quantile, so that a comparator's calibration cannot force the cut;
   - a confidence-matched margin − dist co-primary (H2);
   - the "head" comparator of V3b defined per paradigm before the run: a linear probe for DINO, the zero-shot gap for CLIP;
   - power sized from B1's observed SEs (0.0146 at n 356; 0.0047 at n about 2,650);
   - anchors computed and archived before the run.
6. **Optional, cheap B1c.** The reserve ViT (augreg_in1k, a non-DeiT recipe) at cut 0.7 with a ViT-only power rule. It would test whether V2 and V3b hold at the primary cut, outside the DeiT lineage.
7. **AH-4.** A label needs a new pre-registration that names `Vb` as its input and uses a calibration-free lead.
8. **Controller follow-up.** A pre-registered margin-vs-logit-gap test at fixed confidence (E7's hint), in B1c or B2.
