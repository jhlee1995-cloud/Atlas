# B1b: the ViT margin test after correcting the B1 gate anchor

This is a new pre-registration, committed before any ViT image is read. It is **not** a re-decision of B1.

**B1's own record stands:**
- Outcome: INVALID-PLUMBING (gate G0), from `results/margin_b1_vitb16/verdicts.json` at P `f1c3c43`, P_run `d257d91,68244f7`.
- `results/b1_gate/gate.json` stays CLOSED and is never rewritten.
- B1b is a separate entry that reuses B1's ResNet50 runs and B1's decision rule, with exactly one threshold changed.

## Why B1's gate closed

Gate G's four parts came out as follows (`results/b1_gate/gate.json`):

| part | result | detail |
|---|---|---|
| G0 | false | only the legacy valley-separation clause failed |
| G1 | true | legacy margin 0.800 reproduced |
| G2 | true | order margin > cluster > energy, cluster ~0.636 |
| G3 | true | atlas-definition margin > distance at cut 0.5: +0.029, p = 0.048 |

The failing clause: `lg.valley_sep_legacy` was 1.2681, outside the pre-registered band [1.12, 1.16] around the legacy "valley sep 1.14" (MASTER_SUMMARY.md). Every other G0 clause held:
- n_test 10048;
- accuracy 0.7711;
- both ViT self-tests PASS;
- data.json PASS;
- ReaL label agreement 0.90.

The plumbing reproduces **exactly**:
- The legacy block's penult features equal the legacy cache `<volume>/cache/imagenet/penult_10000.npz` bit for bit. `b1_acc_info.legacy_cache.max_abs_dF` is 0.0 over 10,048 rows.
- The atlas code computes the same formula as `Upgraded-Mod session_experiments/imagenet_extract.py:66-72`: in-sample class means over classes with at least 3 samples; valley separation = mean pairwise center distance divided by mean per-class mean within-distance.

The reference value was the wrong one. The legacy formula, applied on the pod on 2026-09-23 to the two legacy caches (read only; no ViT data involved):

| legacy cache | rows | per class | classes (min 3) | separation |
|---|---|---|---|---|
| `penult_10000.npz` (the sample of the legacy margin 0.800) | 10,048 | ~10 | 999 | **1.2681** |
| `penult_30000.npz` (`imagenet_valley_check.py`) | 30,056 | ~30 | 1000 | **1.1423** (1.1315 at min 30) |

So "1.14" is the 30k-sample value. It came from the valley-check experiment, which had about 30 images per class. With about 10 per class, the center noise inflates the between-center distances and gives 1.268. B1's pre-registration paired the 10k legacy block with the 30k anchor. That is a transcription error in the gate, not a failure of the reproduction.

## What B1b changes

**One threshold.**
- The G0 valley-separation band becomes [1.2481, 1.2881]: the 10k legacy value 1.2681 ± 0.02, the same half-width as B1's band.
- It is implemented without editing any B1 file:
  - `scripts/b1b_gate.py` imports `scripts/b1_gate.py` and replaces `LEGACY_SEP` only;
  - `scripts/b1b_verdicts.js` is `scripts/b1_verdicts.js` with three marked changes: this band, the gate record `results/b1b_gate/gate.json`, and a two-part provenance.

**Everything else is B1's rule, unchanged:**
- the other G0-G3 clauses and thresholds;
- P0;
- c\* (0.7, with the pre-registered power fallback to 0.5);
- V1-V4 and E1-E11;
- the outcome table A / B / MIXED / UNDECIDED;
- margin vs maxprob, margin vs logit gap;
- the D13 joint reading with A4b;
- the ViT manifests `experiments/queue/margin_b1_{vitb16,vitb16_swap,deitb,deitb_swap}.yaml` and their notes;
- ATLAS_STATUS row 10.

**Runs.**
- `pod_atlas.sh --b1b` reuses B1's committed ResNet50 atlases and B1's check dir `results/instrument_check_b1_r2` (read only).
- It writes `results/b1b_gate` and `results/instrument_check_b1b`.
- It builds the four ViT runs only if the B1b gate is OPEN. Those runs are the ViTs' first ImageNet touch.
- Pull, commit results first, then:
  `node scripts/b1b_verdicts.js --p f1c3c43 --p-run d257d91,68244f7 --p-b <P_B> --p-b-run <P_B_run> --a4b results/atlas_v1_resnet56_s1/a4b_eval.json --json results/margin_b1_vitb16/verdicts_b1b.json`
- Provenance:
  - the ResNet50 and E9 atlases against B1's P and P_run list;
  - the ViT atlases against P_B (this commit) and P_B_run, with B1's frozen list plus this file and the two B1b scripts.
  - `ATLAS_STATUS.md` is not edited between P_B and the ViT run.

## What a reader should discount

- **The gate's anchor was changed after the gate was seen to fail.**
  - Mitigation: the new anchor is computed from legacy material only (the legacy cache and the legacy formula), not from any B1 number. It equals the B1 legacy block's value because the features are bitwise identical.
  - G1-G3, which carry the scientific content of the gate, passed under B1's own thresholds.
- **The ResNet50 margin numbers are no longer unseen.** The gate record shows G3's d and p. They were already fixed by B1's run; B1b adds no ResNet50 computation.
- **The ViT confirmation material is unspent.** No ViT image was read in B1: only the random-input self-tests ran.
- B1b's ViT outcome is reported as B1b, beside B1's INVALID-PLUMBING. It is not a replacement.

## Session facts

- Pod `nc2mzssur3bb0k` (RTX 4090), the same session as A4b, B1 and ANOMALY_H1.
- The B1 relaunch branch `b1-r2` (68244f7) raised only the `b1_stage` time limit. The ViT runs use the 7200 s limit, which is also on main (bf4d531).
