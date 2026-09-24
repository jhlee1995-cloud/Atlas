# B4a: a one-sided README gate and an S1 supplement for three hub units

This is an amendment to batch 4. It is committed as P_A before any dump of the three units it concerns exists. It is **not** a re-decision of session S1's record.

**Approved by the owner on 2026-09-24** (binding decision): a one-sided README gate in the B1b style, plus a supplementary extraction of exactly these three units on the same S1 pod after `block_b4s1` ends.

**S1's record stands:**
- `results/b4_weights/weights.json` (PASS 32, FAIL 3, ABSENT 8) is never rewritten. Under the frozen rule it still lists the three units as FAIL.
- The B4a record `results/b4_weights/weights_r2.json` is the weights record for these three units only. For the other 40 units, `weights.json` remains the record.

## What happened in S1

`scripts/b4_weights.py` (frozen, `experiments/b4/frozen_files.txt`) runs the README gate before any extraction:

> |acc_10k - README top-1| <= 0.003 (`README_TOL`, D2 and D15; "fatal per unit")

Three hub units failed this gate and nothing else. In all three, the measured full-10k accuracy was **above** the README value:

| unit | role | dump (registry) | README top-1 | delta |
|---|---|---|---|---|
| mobilenetv2_x0_75 | C | `results/b4c_mobilenetv2_x0_75/dump` (sealed fit) | 93.72 | +0.0036 |
| shufflenetv2_x0_5 | Dnew | `results/b4d_shufflenetv2_x0_5/dump` (open fit) | 90.13 | +0.0052 |
| shufflenetv2_x1_0 | C | `results/b4c_shufflenetv2_x1_0/dump` (sealed fit) | 92.98 | +0.0032 |

- **All 19 hub units:** 13 deltas are positive, 3 are zero and 3 are negative. The minimum is -0.0001 and the maximum is +0.0052. No hub unit is below its README value by more than 0.0001.
- **The README values:** the registry's values equal the upstream README at the pinned ref `786c1625` (checked).
- **The other weights checks:** the three files passed the hub byte-size check and the 8-hex tag check. They are README-only failures.
- **The anchors:** the rule-6 anchors (resnet20 and resnet56 hub) passed the anchor gate exactly. That gate checks the loader, the normalisation, the data rows and the taps at the pinned ref.

`scripts/b4_extract.py` `weights_gate()` refuses a unit whose weights record is FAIL. A hub unit needs a PASS or NOT_EVALUABLE record. S1 therefore skipped the three units softly: no dump, record or probe output of them exists.

Without an amendment, two things follow:
- **C drops from 12 units to 10.** The PRIMARY claim needs at least 10 units per outcome in at least 4 outcomes (D12). At 10 units, any single unit an outcome cannot use makes PRIMARY NOT_EVALUABLE.
- **Discovery loses its only ShuffleNet.** Dnew exists to check each family's plumbing on discovery material and to let the laws see family offsets (D3). The ShuffleNetV2 units in C (x1_0, x1_5, x2_0) would then have no discovery unit of their family.

## What the gate is for

The README gate exists to catch a network that was loaded, normalised or built wrongly: the wrong weights, the wrong mean and std, or the wrong architecture at the pinned ref.

- **Such errors lower the accuracy.** They do not raise a released checkpoint above its published accuracy.
- **The two-sided band also refuses excesses.** That is not what the check is for.
- **The cause of the excess is not established.** One possibility is a README figure from a different evaluation or checkpoint than the release asset. This is not verified, and B4a does not depend on it.

## What B4a changes

**One rule:**

| | FAIL when | above README + 0.003 |
|---|---|---|
| B4 (`scripts/b4_weights.py`, frozen) | \|acc_10k - README\| > 0.003 | FAIL |
| B4a (`scripts/b4_weights_amend.py`) | acc_10k < README - 0.003 | PASS, with the INFO flag `README-EXCEEDED` |

**How it is implemented without editing a frozen file** (the B1b pattern):
- **The script:** `scripts/b4_weights_amend.py` imports `scripts/b4_weights.py`.
  - The rule is a pure function `readme_status(acc, want, tol)`.
  - `gate_unit` is the frozen function line for line. The only difference is the one line marked `B4a CHANGE`, which calls `readme_block`, and the calls go through `BW.`.
  - It records what the frozen rule says (`readme.status_b4`), the INFO flag (`readme.info` and the unit's `info`) and `"amendment": "B4a"`.
- **The tests:** `tests/test_b4a_amendment.py` checks three things.
  - The lower side equals the frozen rule bit for bit, at every 10k count within ±0.008 of each of the 19 hub README values.
  - The upper side never fails, and it flags exactly the cases the frozen rule failed.
  - `gate_unit` is identical to the frozen one outside the marked line.
  - `node scripts/b4a_p2.js --selftest` covers the two Windows steps (the P2 timing floor and the step-E seal check) on fixtures. It runs on Windows before P_A is committed; the pod has no node.
- **Scope:** the default `--units` is exactly the three units.
  - Any other unit is refused unless `--allow-other` is given.
  - A unit that is not a hub unit is always refused.
  - Each of the three must appear in the S1 record as a README-only FAIL above the README: status FAIL, readme status FAIL, delta > +0.003, a sha256 recorded, no load error, the hub byte size equal to the release asset (`bytes` = `bytes_expected`) and the 8-hex tag check passed (`tag_prefix_ok` true).
- **The record:** `results/b4_weights/weights_r2.json` uses the schema of `weights.json`, which `weights_gate()` reads through `units[<id>].status`.
  - It adds `amendment`, `amendment_rule`, the S1 record's path and sha256, and a per-unit `s1_crosscheck`. The cross-check (sha256 and 10k count equal to S1's) is recorded; it is not a gate.
  - The heads go to `results/b4_weights/heads_r2/<id>/head.npz`, so the S1 heads are never overwritten. Nothing reads either heads directory; the extractor exports its own head into each dump.

## What B4a does not change

- **Every other weights check:**
  - `torch.hub` at the pinned ref;
  - the hub byte size (fatal);
  - the full sha256;
  - the 8-hex tag (a mismatch gives NOT_EVALUABLE);
  - the full-10k accuracy measurement (`b4_weights.test_accuracy`);
  - the head export.
- **The lower side of the README rule,** bit for bit.
- **Every file in `experiments/b4/frozen_files.txt`,** plus `pod_atlas.sh`, `scripts/pod_b4.sh` and the committed instruments (`B4_COMMITTED`). The supplement refuses to run if any of them differs from P1. The dumps therefore come from the P1 extractor, registry, rows, dtypes and taps.
- **The evaluators and every claim, threshold and outcome rule,** including PRIMARY (D12) and the D7 seal rule. The two C dumps are sealed like the other ten and opened once, in S2.
- **The verdicts of the other 40 units.**
- **D8's list of P1 -> P2 changes.** B4a sits outside that list: it is an owner-approved amendment that only adds files (this document, the three scripts `b4_weights_amend.py`, `b4a_supplement.sh` and `b4a_p2.js`, the test and the run request) and changes none of the files D8 governs.

## Why this is not result-driven

- **No dump, record, probe output or outcome of the three units exists.** The extractor refused them before writing anything, and the supplement's guards check this again on the pod.
  - The only number read for these units is the full-10k accuracy against the README. D2 treats it as public ("measuring it reveals nothing").
- **The rule is decided (2026-09-24) before any extraction of the three units.**
- **The rule follows from the gate's purpose, not from these three units.** The spread over all 19 hub units shows that the measurement sits at or above the README. A one-sided rule is what the purpose implies.
- **The instrument is verified independently.** The anchors passed the anchor gate exactly, and the three files passed the byte-size and tag checks. `b4_weights_amend.py` refuses to run unless S1's record shows both for each unit.
- **The gate selected the three units, not an outcome.** Nobody knows whether restoring them helps or hurts any claim.

**What a reader should discount:**
- **The rule was changed after the gate was seen to fail**, as in B1b.
  - Mitigation: the lower side, which carries the gate's purpose, is unchanged.
  - The change acts before any dump exists.
  - It restores the pre-registered C set (12) and D21 (16 + 5) rather than creating a new one.
- **The two sealed dumps are extracted hours after the other ten,** in the same pod, on the same GPU, with the same software stack (checked, hard) and the same frozen code.
  - Their `meta.b4.repo_commit` is P_A, not P1.
  - Their `meta.b4.weights_gate` carries `"amendment": "B4a"`.

## The supplement (same S1 pod, after `block_b4s1` has ended)

`scripts/b4a_supplement.sh` runs standalone.
- It sets what `scripts/pod_b4.sh` expects from `pod_atlas.sh`'s prelude: `VOLUME`, `SOFT_LOG`, `STAMP`, `soft`, `cpu_quota`, `has_mode` and the caches on the volume. `TORCH_HOME` is S1's hub cache.
- It then sources `scripts/pod_b4.sh` unchanged.
- It uses the D10 relaunch conventions: tag `ATLAS_B4_TAG=_r2` and check dir `results/instrument_check_b4s1_r2`.

The exact pod commands are in `results/b4/RUN_REQUEST_S1_SUPPLEMENT.md`.

1. **Guards (hard).** A failure here leaves nothing behind except the logs, and the tag stays unused.
   - The tag is `_r<k>` and differs from S1's.
   - The check dir is fresh.
   - The `b4_guard_common` checks hold: P1 <= HEAD, the committed instruments are unchanged since 2b561a9, and no GPU job is running.
   - The frozen files, `pod_atlas.sh` and `scripts/pod_b4.sh` equal P1.
   - The tracked files equal HEAD, and HEAD holds this amendment.
   - `pod_atlas.sh` is not running, and no batch-4 GPU job or probe is running (`pgrep` with the bracket trick).
   - S1's `sealed.json`, `versions.json`, `head.txt` and `weights.json` exist, and S1's HEAD is an ancestor of HEAD.
   - The three units are neither cut nor re-roled.
   - No dump of the three units exists. A complete dump is accepted only if an earlier supplement run wrote it, which means its meta carries the B4a gate. A partial dump needs `ATLAS_B4_MOVE_PARTIAL=1`.
   - No extract record exists under `''` or `_r2`.
   - No complete probe output exists under `''`, and no `_r2` output directory is non-empty. An incomplete `''` output left by S1's discovery lane is logged and kept, never deleted; no evaluator reads a directory without its output file.
   - `weights_r2.json` and `heads_r2` are absent.
   - CUDA is present.
   - The software stack is S1's: numpy, scipy, sklearn and pytest import, and the versions equal S1's in python, numpy, scipy, sklearn, torch, torchvision, device and the TF32 flags.
     - This is checked before the check dir exists, into `logs/b4a_<stamp>/versions_guard.json` on the volume.
     - A stopped and restarted pod has lost its pip-installed scipy, sklearn and pytest (container disk). It refuses here and keeps the tag; the run request gives the recovery.
2. **Versions.**
   - `results/instrument_check_b4s1_r2/versions.json` must equal S1's in the same fields (hard; the guard compared the same stack).
   - The `versions-match` record is written beside it.
3. **Known-answer tests.** `pytest tests/test_b4a_amendment.py` (hard).
4. **S1's seals are intact.** `--verify-seals` against S1's `sealed.json` (hard) -> `verify_seals_s1.json`.
5. **The weights gate.** `scripts/b4_weights_amend.py` -> `results/b4_weights/weights_r2.json` and `heads_r2/`.
6. **Quota and extraction.**
   - The quota guard (the three fit dumps, about 1.5 GB).
   - Then `b4_extract <id> fit` in registry order, with `--sealed` for the two C units and the `_r2` weights record. Each unit fails softly.
   - Extract records go to `results/b4_extract/<id>_fit_r2.json`.
7. **The S1 discovery probes of shufflenetv2_x0_5.** These are the two jobs of `b4_reg.py jobs --phase discovery --tag _r2` for that unit: T2 `collapse_probe.py` and T1 `t1_scoreboard.py` fit.
   - They run with `b4_cpu_lane`'s settings: niced, one BLAS thread, no CUDA.
   - The full lane is not used, because it would re-probe every discovery unit under `_r2`.
   - The sealed units get no probe in S1.
   - The step fails softly when `b4_reg.py jobs` fails, when its list holds no line for the unit, or when an output file is missing after the jobs ran. It skips a job only when that job's output exists under an earlier `_r<k>` tag, and it reports "done" only when every job of the unit was skipped that way.
8. **A complete seal manifest.**
   - `b4_extract.py --seal-manifest` writes `results/instrument_check_b4s1_r2/sealed.json`. It lists every sealed layout in the registry, so it covers S1's sealed dumps and the new ones.
   - `seal_diff.json` checks, hard, that every S1 entry is unchanged (meta sha256, file count, bytes) and that the only additions are the two B4a sealed fits.
   - `--verify-seals` then runs on the new manifest (hard).
9. **Timing.** `b4_reg.py timing` writes `results/instrument_check_b4s1_r2/timing.json`. It covers every probe output, S1's and the supplement's.
   - The two ShuffleNet runs are timed with only two jobs on an otherwise idle pod. Every other family was timed in the loaded S1 lane. P2 corrects for this (see `b4a_p2.js timing` below).
10. **Summary.** `supplement.json` holds each unit's weights status and its dump and record presence. It contains no accuracy.
    - Discovery presence is looked up under every `_r<k>` tag, with the path found, because a relaunch skips the jobs an earlier run completed.
    - It also records the timing condition.

A failed unit is soft: the script exits 1 at the end, as `block_b4s1` does. The relaunch uses the next unused `_r<k>` (see "Relaunch tags" below and the run request).

## How S2 must be launched

First, in its own ssh call, check that the chosen manifest lists both B4a dumps SEALED (hard: do not launch otherwise):

```bash
cd /workspace/Atlas && python - results/instrument_check_b4s1_r2 <<'PY'
import json, sys
d = sys.argv[1]
e = {(x["unit"], x["layout"]): x["status"] for x in json.load(open(f"{d}/sealed.json"))["entries"]}
miss = [u for u in ("mobilenetv2_x0_75", "shufflenetv2_x1_0") if e.get((u, "fit")) != "SEALED"]
sys.exit(f"B4a dumps not SEALED in {d}/sealed.json: {miss}" if miss else 0)
PY
```

Then launch S2 with the same directory:

```bash
ATLAS_B4_P1=<P1> ATLAS_B4_P2=<P2> ATLAS_B4S1_CHECK_DIR=results/instrument_check_b4s1_r2 \
  bash /workspace/Atlas/pod_atlas.sh /workspace --b4s2 --b4cpu
```

If "Which check dir" below selects another directory, use it in both commands.

- **The seal manifest.** `b4_guard_p2` runs `--verify-seals` against `$B4_S1_DIR/sealed.json`.
  - S1's own manifest was written at the end of GPU-D, before the supplement, so it lists the two new sealed dumps as ABSENT.
  - `verify_seals` checks only SEALED entries, so S1's manifest would pass without covering them.
  - The `_r2` manifest is S1's manifest plus the two new dumps; `seal_diff.json` proves the S1 entries are unchanged.
  - Only the environment variable connects S2 to this manifest. If it is left out, the frozen default reads S1's manifest and passes silently. So the guarantee does not rest on the variable alone:
    - the pre-launch check above makes sure the chosen directory is complete;
    - the step-E check below catches, after the run, a launch without the variable or with the wrong directory.
- **Which check dir.** `ATLAS_B4S1_CHECK_DIR` is the check dir of the latest run that wrote a complete `sealed.json` after every sealed extraction.
  - Normally this is the newest supplement check dir (`_r2`, or the tag of a supplement relaunch), whose `seal_diff.json` is PASS.
  - If S1 itself was relaunched after the supplement, it is that relaunch's check dir.
  - `freeze_P2.json` records it (see Provenance).
- **The versions record.** `versions-match` reads `$B4_S1_DIR/versions.json`. The `_r2` file comes from the same pod and stack, checked equal to S1's.
- **Leave `ATLAS_B4_S1_TAG` unset.** The anchors' S1 outputs are under `''`; the supplement does not touch the anchors.
- **S2's own tag is its own business.** The default is `''`; a relaunched S2 uses `_r2` and `results/instrument_check_b4s2_r2`. It does not collide with the supplement, which probed only shufflenetv2_x0_5 in the discovery phase.
- **The S2 run request, written at P2,** must carry the pre-launch check, the launch line and the step-E check.

### Relaunch tags

Relaunches of S1 and of the supplement take their tags from one sequence. Each takes the **next unused** `_r<k>`: one with no `results/instrument_check_b4s1_r<k>` and no `results/b4_weights/weights_r<k>.json`.
- Normally `_r2` is the supplement's first run. The next relaunch of either kind is `_r3`, the one after it `_r4`, and so on.
- If S1 was relaunched before the supplement (under `_r2`), the supplement takes the next unused tag and sets `ATLAS_B4_S1_TAG` to S1's tag, so that it reads S1's check dir and weights record.
- Both refuse an existing check dir, and the supplement also refuses an existing `weights_r<k>.json`. A collision therefore stops before anything is written.
- The check dir that S2 reads follows "Which check dir" above.

### The step-E seal check (required)

After R2, before any result of the two B4a C units is reported, run on Windows:

```bash
node scripts/b4a_p2.js seals --json results/b4/b4a_seals.json
```

It reads `--b4a-dir` from `freeze_P2.json` `"b4a".check_dir`. A unit PASSES only when all three conditions hold:
1. The B4a manifest lists its fit dump SEALED, and its `seal_diff.json` is PASS for that manifest.
2. A `results/instrument_check_b4s2[_r<k>]/verify_seals*.json` has a PASS entry for the dump, against a manifest that lists the dump SEALED with the same `meta_sha256`.
3. Every T1 fit and T2 record of the unit, under every tag, lists the dump with that `meta_sha256`.

**If a unit FAILS:**
- It is NOT_EVALUABLE for every claim, PRIMARY included, until the check passes.
- The repair for a missing S2 seal check (condition 2) is a `--verify-seals` run against the B4a manifest. It needs a pod with the volume, while the dumps still exist:

  ```bash
  python scripts/b4_extract.py --verify-seals --registry experiments/b4/models.json \
    --manifest results/instrument_check_b4s1_r2/sealed.json --out results/instrument_check_b4s2<tag>/verify_seals_b4a.json
  ```

  Then run the check again. The record says whether S2's own guard covered the dump or a later run did.
- Until the check passes, `SESSION.md` reports every claim that counts the failing units from an evaluator run without them:
  - `t1_eval.js --cut "<units>"`;
  - `collapse_laws.js --root` on a copy of `results/` without their confirmation outputs (never a deletion).
- The run with them is reported as INFO.

## How the evaluators find the outputs

No evaluator changes. `t1_eval.js`, `collapse_laws.js` and `t3s_eval.js` never read a weights record.

- **Extraction.**
  - With `B4_TAG=_r2`, `pod_b4.sh`'s `b4_extract()` passes `--weights-record results/b4_weights/weights_r2.json`.
  - The gate record is copied into each dump's `meta.b4.weights_gate`.
  - The probes' `instrument.weights_status` is the load record (`meta.b4.weights`), which is unchanged.
- **Dumps.** Dumps carry no tag; their paths are the registry's.
- **Discovery outputs of shufflenetv2_x0_5:** `results/b4_t2/shufflenetv2_x0_5_r2/probe.json` and `results/b4_t1/shufflenetv2_x0_5_fit_r2/scoreboard.json`.
  - `t1_eval.js` reads every `results/b4_t1/*` directory and keys each by the unit and layout in its JSON (tags `''`, `_r<k>`, `_p2`).
  - `collapse_laws.js` `scanRecords` accepts `''`, `_r<k>` and `_p2` (`TAG_OK`) and takes the highest relaunch.
  - `b4_reg.py timing` strips the tag.
  - No `''` output exists for the unit (guarded), so nothing is duplicated.
- **Confirmation.** The two C units are probed in S2 like the other ten, under S2's tag. The touched-once rule holds, because no output of them exists under any tag.
- **Check dirs.** `t1_eval.js`'s discovery check-dir scan (`/^instrument_check_b4s1(_r\d+)?$/`) also lists `instrument_check_b4s1_r2`.
  - It reads only `selftest_*.json` there.
  - The supplement writes none, and S1's directory satisfies the self-test gate.
- **`t3s_eval.js`** reads `results/instrument_check_b4s1/sealed.json` by default, for the Sconf maps only.
  - Those entries are identical in both manifests (`seal_diff.json`), so the default keeps working.
  - `--sealed results/instrument_check_b4s1_r2/sealed.json` gives the same result.
- **`b4_timeouts.js` at P2.** S1's `timing.json` has no ShuffleNet run. The supplement's gives the ShuffleNet `t_ref`, but it was measured with two jobs on an otherwise idle pod. Every other family's `t_ref` was measured in the loaded S1 lane: about P probes at nice 19, beside the training pairs and GPU-D. S2 again runs about P jobs at once.
  - **The rule, fixed here before any ShuffleNet timing exists:** per program (T1, T2), the ShuffleNet `t_ref` is max(measured, the S1-lane `wall_s` of the Dnew units of the nearest width × (1024 / 1280)^2). Those units are mobilenetv2_x0_5 and repvgg_a0; they read the same fit rows without extras. The rule never lowers a value.
  - **The input:** the `timing.json` of the newest check dir (the newest supplement run, or a later S1 relaunch). `b4_reg.py timing` scans every tag, so the newest file covers every output.
  - **The commands:**

    ```bash
    node scripts/b4a_p2.js timing --timing results/instrument_check_b4s1_r2/timing.json --out experiments/b4/timing_b4a.json
    node scripts/b4_timeouts.js --registry experiments/b4/models.json --timing experiments/b4/timing_b4a.json \
      --out experiments/b4/timeouts.json
    ```

  - `timing_b4a.json` is committed in P2 beside `timeouts.json`. Its `b4a` key holds the rule, the source file's sha256 and each value it raised.
  - `b4a_p2.js timing` refuses when the timing holds no ShuffleNet T1 or T2 run. The lead then relaunches the supplement's discovery jobs, or runs `b4_timeouts.js` on the timing directly (ShuffleNet falls back to the widest measured unit) and records that in `freeze_P2.json`.

## Provenance

- **The commits.** S1 ran at P1 = `ad97d0a659dd9f6f60b24e0b8cc1681f1bd8a166`. The supplement runs at P_A, the commit that adds this file.
  - P_A is a descendant of P1 with identical frozen files, `pod_atlas.sh` and `scripts/pod_b4.sh`; the supplement's guard enforces this.
  - The new dumps' `meta.b4.repo_commit` and the discovery probes' `code.repo_commit` are therefore P_A.
  - Their code sha256 values equal P1's.
- **`--p-run` must include P_A** wherever it lists the S1 HEAD:
  - discovery: `--p-run <P1>,<P_A>` (t1_eval.js, t3s_eval.js);
  - confirmation: `--p-run <P1>,<P_A>,<S2 HEAD>` (t1_eval.js, collapse_laws.js `--evaluate`, t3s_eval.js).
  - Without P_A, the discovery records of shufflenetv2_x0_5 fail the `repo_commit` check.
- **`experiments/b4/freeze_P2.json`** records B4a under its own key and lists P_A beside the S1 HEAD. For example:

  ```json
  "b4a": {"p_a": "<P_A>", "doc": "docs/plans/B4A_AMENDMENT.md", "check_dir": "results/instrument_check_b4s1_r2",
          "tag": "_r2", "s2_check_dir": "results/instrument_check_b4s1_r2", "timing": "experiments/b4/timing_b4a.json"}
  ```

  - `check_dir` and `tag` are the newest supplement run's. `b4a_p2.js seals` reads `check_dir`.
  - `s2_check_dir` is the `ATLAS_B4S1_CHECK_DIR` S2 is launched with (see "Which check dir").
  - Do not put B4a in `amendments`. That list holds the D8 (b)/(c) claim amendments the evaluators apply per track.
- **R1** is the pull after the supplement: S1 and the supplement together, dumps excluded.
- **`results/b4/SESSION.md`** reports B4a with a C count of 12. It also states that under the frozen gate alone, C would have held 10 units, and names the two.
  - It reports the step-E seal check (`results/b4/b4a_seals.json`) and whether S2's own guard covered each B4a dump.
  - It names each ShuffleNet value that `timing_b4a.json` raised.

## Cost

- **Expected:** about 20 minutes on the RTX 4090, about $0.25 at $0.74/h.
  - The GPU part (the three gate loads and accuracies, then three extractions of about 1 minute each at `est_gpu_s` 55-57) takes about 10 minutes.
  - The variable part is the two discovery probes of shufflenetv2_x0_5, which run in parallel. S1's `timing.json` for mobilenetv2_x0_5, a Dnew unit of similar width, is the best estimate.
- **Upper bound:** the D10 limits (`t1_wide` 7200 s), about 2.3 hours or $1.7.

## Session facts

- The pod is the S1 pod: RTX 4090, EU-RO-1, volume `kxfir1tryb`, template `runpod-torch-v280`.
- The supplement starts only after `block_b4s1` has ended. Its `pgrep` guard refuses to start while `pod_atlas.sh` or any batch-4 job runs.
- The pod stays running from the end of `block_b4s1` until the supplement ends: it is never stopped or deleted in between. A stop wipes the container disk and the pip-installed scipy, sklearn and pytest on it. The supplement's stack guard would then refuse, and the run request gives the recovery.
- The pod's code moves from P1 to P_A (`git reset --hard <P_A>`) only after S1 has ended. No S1 result is tracked yet, so the reset touches none of them.
