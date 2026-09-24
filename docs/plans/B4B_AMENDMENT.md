# B4b: doubled session-2 time limits

This is an amendment to batch 4. It is part of the P2 commit, before session S2 has run and before any confirmation data has been read. It changes how long an S2 job may run, never what a job computes.

**Approved by the owner on 2026-09-24** (binding decision): every S2 time limit = min(14400, 2 x the limit the frozen `scripts/b4_timeouts.js` emitted), never lower.

## Why

D10 sets the S2 limits with `scripts/b4_timeouts.js` (frozen, `experiments/b4/frozen_files.txt`):

> clamp(ceil(3 t_ref max(1, (d/d_ref)^2) rows/rows_ref), 1800, 14400); t_ref = the slowest S1 run of the same program on the same family

The rule scales S1's fit-layout times by penult width and rows only. It leaves out three things:
- **The bootstrap.** Confirmation runs B = 1000 draws, S1 ran B = 200. The bootstrap share of T1 target time was not measured; an operation count gives about 0.2-0.33.
- **The eval layouts.** No eval layout was timed in S1. The row factor divides by the reference dump's 122,500 rows, but S1's discovery run read 88,000 of them. The rows read grow 3.22x (F/F20 eval) and 2.77x (R2 eval), against the rule's 2.19x and 1.87x.
- **The wide C units' earlier taps.** These are 2.4-3.7x wider than the reference's; the rule scales by penult width only.

Three cost models (the P2 lead's and two reviews') gave these margins (limit / estimate) under the emitted limits:

| jobs | emitted limit | share 0.25 | share 0.5 | share 1.0 |
|---|---|---|---|---|
| F/F20 eval (4 T1 jobs) | 2126 s | 1.55-1.62x | 1.14-1.16x | over the limit from a share of about 0.6-0.7 |
| R2 eval (8 T1 jobs) | 1809 s | 1.56-1.86x | 1.13-1.37x | over the limit from a share of about 0.6-0.85 |
| wide C T1 fits (6) | 1800-1931 s | 1.86-2.14x | 1.48-1.65x | 1.05-1.12x |

The share at which an emitted eval limit is reached differs by model. For F/F20 it is about 0.64 in the P2 lead's model and in the first review's. For R2 it is about 0.62 in the second review's model, 0.70 in the P2 lead's, and 0.74-0.86 across the three units of the first review's.

- **Load.** Within S1 the same probe's per-target time varied 1.3-1.5x with load. S2 runs up to 16 probes at once, so its per-job speed is not established to be better than S1's.
- **The cost of a kill.** The four F/F20 eval jobs share one limit and one cost model, so what kills one is likely to kill the others. T1's R scope needs 3 of the 4 F/F20 units, so the kills would leave it NOT_EVALUABLE. D10's single relaunch might not rescue them, and no limit can be raised after S2 has unsealed.

## What B4b changes

**One rule, applied to all 72 keyed limits:**

| | limit |
|---|---|
| B4 (`scripts/b4_timeouts.js`, frozen) | clamp(ceil(3 t_ref ...), 1800, 14400) |
| B4b (`scripts/b4b_timeouts.js`) | min(14400, 2 x the B4 limit), never lower |

- **The result.** No limit reaches the cap, so every limit is exactly 2x the emitted one. The limits run from 3600 s to 8324 s (shufflenetv2_x2_0 T1 fit, 58% of the cap). The F/F20 eval jobs get 4252 s, R2 eval 3618 s, the wide C fits 3600-3862 s, and every T2, streams and T3S job 3600 s.
- **The margins** in the same three cost models:

  | jobs | B4b limit | share 0.25 | share 0.5 | share 1.0 |
  |---|---|---|---|---|
  | F/F20 eval | 4252 s | about 1310-1370 s, 3.10-3.25x | about 1830-1860 s, 2.29-2.32x | about 2750-2960 s, 1.44-1.55x |
  | R2 eval | 3618 s | 3.12-3.73x | 2.26-2.75x | 1.46-1.80x |
  | wide C T1 fits | 3600-3862 s | 3.72-4.28x | 2.96-3.30x | 2.10-2.24x |

  The share-1.0 values come from a re-run of the P2 lead's model and the first review's. The second review's model is extrapolated linearly from its 0.25 and 0.5 values; bootstrap time is linear in B. At share 1.0 all target time is bootstrap, so no limit is reached at any share in any of the three models.
- **How it is implemented without editing a frozen file:**
  - The frozen emission is kept unchanged as `experiments/b4/timeouts_b4_emitted.json`.
  - `scripts/b4b_timeouts.js` reads it and writes `experiments/b4/timeouts.json`. It copies `peak_gb`, `split` and `unmeasured` unchanged. It rewrites `rule` to state the B4b rule and quote the emitted rule, so the file's stated rule matches its values. It adds an `amendment` key with the rule and the source file's sha256.
  - That sha256 is taken over the CRLF-normalised bytes, the batch-4 convention (`scripts/b4_stats.js` `fileSha256`). A Windows checkout with `core.autocrlf=true` has a CRLF worktree, and it records the same value.
  - It refuses a non-positive limit and never overwrites its output. `--selftest` checks these (10/10):
    - the doubling, the cap and never-lower;
    - the copied keys and the stated rule;
    - the record and the refusal to overwrite;
    - that a CRLF input gives the LF input's sha256 and the same output bytes.

## What B4b does not change

- **What any job computes.** A probe runs as `timeout <s> python ...`. The limit decides only whether the process may finish; a job that finishes writes the same output under any limit.
- **`scripts/b4_timeouts.js` and its emission.** A second emission into a scratch file is byte-identical to `timeouts_b4_emitted.json`.
- **`peak_gb`, and so S2's parallelism P.** `split` and `unmeasured` (both empty) are unchanged too, so there is still no SPLIT, cap or cut.
- **Every file in `experiments/b4/frozen_files.txt`,** plus `pod_atlas.sh` and `scripts/pod_b4.sh`. That includes `B4_LIMITS`, the S1 defaults; S2 does not fall back to them, because every one of its 72 jobs has a key in `timeouts.json`.
- **`scripts/b4_reg.py`.** Its `timeout()` reads the per-job keys of `timeouts.json`, and `peak` reads `peak_gb`; nothing reads `rule` or `amendment`. `tests/test_b4_reg.py` stays hermetic.
- **The probe code,** so there is no re-probe.
- **The evaluators and every claim, threshold and outcome rule.** No evaluator reads `timeouts.json`.
- **D10's relaunch rule.** A job killed by its limit (exit 124) is relaunched once under the next S2 tag, with unchanged code and an unchanged `timeouts.json`. A job killed in that relaunch too is not relaunched again.
- **The S2 host rule** in `results/b4/RUN_REQUEST_S2.md` section 2 (EPYC-Genoa class only).

**D8.** B4b sits outside D8's list of P1 -> P2 changes, like B4a. It is an owner-approved amendment. Unlike B4a, it changes one file that D8 governs: `timeouts.json` is no longer the direct output of `b4_timeouts.js`. So the direct output is committed beside it, and `timeouts.json` is a deterministic function of it. B4b is not a D8 (f) change, because `scripts/pod_b4.sh` is unchanged.

## Why this is not result-driven

- **A limit cannot change a computed value.** It can only turn a finished output into a missing one, and B4b makes that less likely for every job alike.
- **No confirmation data has been read.** S2 has not run, so no probe has opened a sealed dump: S1 and the supplement ran in the discovery phase, where `open_dump` refuses them (`results/instrument_check_b4s1_r2/sealed.json`, 36 SEALED). No confirmation-phase evaluation has run.
- **The inputs are S1's timings and the cost models only.** Those are the same inputs that set the emitted limits.
- **The rule is uniform.** One factor applies to all 72 limits. No unit, program or claim is selected.

**What a reader should discount:**
- **The factor 2 is a judgment, not a derivation.** It was chosen to cover the unmeasured bootstrap share up to 1.0 in the cost models. The emitted limits are kept on record.
- **Longer limits raise the worst-case cost.** A job that hangs runs twice as long before it is killed, and so does a relaunch that reruns a job to its full limit.

## Files

| file | what | sha256 |
|---|---|---|
| `docs/plans/B4B_AMENDMENT.md` | this document | in `freeze_P2.json` "b4b" |
| `scripts/b4b_timeouts.js` | the B4b rule | in `freeze_P2.json` "b4b" |
| `experiments/b4/timeouts_b4_emitted.json` | the frozen emission, D8 (a), renamed unchanged | `9ca331204f42c969c2d3f9809630f8f0fed9aeb9c398b3ee06a3214d4f89fba8` |
| `experiments/b4/timeouts.json` | the S2 limits under B4b | `6695791559fe44a5659dd4338b8fc59df7575e50e7930050cc81801f7be28138` |

The commands (Windows, before the P2 commit; both refuse to overwrite):

```bash
node scripts/b4_timeouts.js --registry experiments/b4/models.json --timing experiments/b4/timing_b4a.json \
  --out experiments/b4/timeouts.json          # emitted at P2, then renamed to timeouts_b4_emitted.json
node scripts/b4b_timeouts.js --in experiments/b4/timeouts_b4_emitted.json --out experiments/b4/timeouts.json
```

A second run of `b4b_timeouts.js` into a scratch file is byte-identical to `timeouts.json`. In a fresh Windows checkout (`core.autocrlf=true`) the input has CRLF line ends (91 CRs). A run there still records source sha256 `9ca33120...` and writes exactly the committed bytes of `timeouts.json`.

## Provenance

- **The commit.** B4b is part of the P2 commit (`atlas: batch 4 freeze (P2)`). It needs no pod run of its own, so it has no separate commit, unlike P_A.
- **The S2 guard.** `timeouts.json` is one of `B4_P2_FILES`. `b4_guard_p2` requires it at P2 and refuses to run if it differs between P2 and HEAD, so S2 runs with exactly these limits.
- **`experiments/b4/freeze_P2.json`** records:
  - B4b under its own key "b4b" and under `d8_changes`;
  - the sha256 of both timeouts files;
  - the margins under "timeouts".

  As with B4a, B4b is not in `amendments`, which holds the D8 (b)/(c) claim amendments that the evaluators apply per track.
- **`results/b4/SESSION.md`** reports:
  - B4b and the emitted limits;
  - the largest wall time / limit ratio against both;
  - every job killed by its limit, if any.
