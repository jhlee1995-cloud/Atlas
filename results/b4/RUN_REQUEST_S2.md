# RUN_REQUEST: batch 4, session 2 (S2, numpy only)

S2 is the confirmation session of batch 4 (docs/plans/B4_INTEGRATION.md D7-D10, D17; docs/plans/B4A_AMENDMENT.md,
"How S2 must be launched"; docs/plans/B4B_AMENDMENT.md, the time limits). It runs no GPU work. It replays the two
rule-6 anchors, then runs the confirmation probes under the P2 rules and the P2 time limits. Only these confirmation
jobs open the sealed dumps, and every open is logged. The verdicts are computed afterwards on Windows (step E below).

**Commits:**

| name | commit | what |
|---|---|---|
| P1 | `ad97d0a659dd9f6f60b24e0b8cc1681f1bd8a166` | pre-registration; S1 ran here (S1 HEAD = P1) |
| P_A | `14caf51b7ca241a2dd95dacc531cc63a5bda6a6a` | B4a amendment; the supplement ran here |
| R1 | `cb617c3cf814b5080f4d11e6cbc38453da0831f4` | the S1 + supplement pull |
| P2 | the commit that adds this file (`atlas: batch 4 freeze (P2)`) | `experiments/b4/freeze_P2.json` and the P2 files |

Below, `<P2>` is that commit's full sha. S2 runs at P2 exactly.

**What P2 fixed** (`experiments/b4/freeze_P2.json`):
- No cut: `b4_timeouts.js` flagged no SPLIT, so C stays 12 (10 from S1, 2 from B4a).
- One claim withdrawn to INFO (D8 b): T2 P4 (reporting only). T1 X3-3 stays ACTIVE; a validity concern about its harm
  reference is disclosed in `freeze_P2.json` "disclosed" (see step E).
- `reprobe` is false: no probe code changed P1 -> P2, so S2 re-probes nothing.
- The time limits are `experiments/b4/timeouts.json`. Amendment B4b (owner-approved) doubled every limit that
  `b4_timeouts.js` emitted (`experiments/b4/timeouts_b4_emitted.json`). Every one of the 72 S2 jobs has a key there.

**Nothing else runs on the pod.** Never `git pull` or `reset` while `pod_atlas.sh` runs. Never stop or restart the pod
during S2: a stop wipes the container disk, which holds the pinned scipy / scikit-learn / pytest (step 4) and
`/root/b4_frames`.

## 0. Before the pod (Windows)

1. P2 is committed and pushed. Record `<P2>` from `git log -1 --format=%H`.
2. The P2 checks passed. They are listed in "Checks done at P2" at the end.
3. Balance: `runpodctl user` shows `clientBalance`. The floor is $100 and the worst case below (P = 4 on the 2x pod at a
   bootstrap share of 1.0, with one relaunch) is about $12, so anything above $112 is enough. (The balance was $176.65
   at P2.)

## 1. Pod: 2x RTX 4090 in EU-RO-1 (fallback 1x RTX 4090)

**This departs from D17's order.** D17 ranks first a CPU pod in EU-RO-1 with `kxfir1tryb`, >= 16 vCPU and >= 2 GB per
vCPU, created in the web console, and only then "the 4090 or the EU-RO-1 pod with the most vCPU per dollar". S2 does not
try the CPU pod, for this reason:
- The tightest S2 limits are scaled from `t_ref` values measured on S1's host class: AMD EPYC-Genoa RTX 4090 hosts in
  EU-RO-1 (the `versions.json` of S1 and of the supplement).
- Under B4b their margins are 2.26-2.75x (eval layouts) and 2.96-3.30x (wide C fits) at a bootstrap share of 0.5, and
  1.44-1.80x (eval layouts) at a share of 1.0; see "Time limits". These margins rest on S1's host class, and no limit
  can be raised after S2 has unsealed.
- Neither pod type guarantees a CPU model. But the EU-RO-1 4090 hosts gave EPYC-Genoa in S1 and in the supplement, and
  no CPU pod's model has been observed in this project.

`freeze_P2.json` records the departure under `s2.pod`. It changes no result: S2 is numpy only, one BLAS thread per job,
on S1's pinned stack, and the rule-6 replay bounds any numerical drift.

The 2x pod is the same host class as a 1x4090. The second GPU is not used: it is there for the CPU quota and the memory
that come with it (a larger P, a shorter wall time). Its vCPU per dollar is the same as a 1x4090's.

The parallelism is P = min(CPU quota - 1, free cgroup memory / 2.1 GB) (`b4_P`, with `peak_gb` 2.1 from
`timeouts.json`). S1 recorded no CPU quota: its `versions.json` holds only `cpu_count_host` 40, the host's
`/proc/cpuinfo` count. S1's discovery lane ran 4 jobs at once at the default 4 GB peak (no `timeouts.json` existed at P1).
So S1's 1x4090 had either a CPU quota of 5 or 16-19 GB of free cgroup memory when the lane started. At a peak of
2.1 GB the same host would give P = 4 or 7-9.

```bash
MSYS_NO_PATHCONV=1 runpodctl pod create --name atlas-b4s2 --template-id runpod-torch-v280 \
  --gpu-id "NVIDIA GeForce RTX 4090" --gpu-count 2 --data-center-ids EU-RO-1 \
  --network-volume-id kxfir1tryb --volume-mount-path /workspace --min-cuda-version 12.8 \
  --container-disk-in-gb 30 --wait
```

- Check the flags with `runpodctl pod create --help` first.
- If no 2x pod is available in EU-RO-1, create the same pod with `--gpu-count 1`.
- The volume `kxfir1tryb` (75 GB) holds the sealed dumps, the discovery dumps of the anchors and the repo.
- S2 writes only small JSON files to the volume. The stream frames go to `/root/b4_frames` on the container disk.

## 2. First ssh call: record the host

```bash
{ date -u; nproc; grep -m1 'model name' /proc/cpuinfo
  cat /sys/fs/cgroup/cpu.max 2>/dev/null || cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us /sys/fs/cgroup/cpu/cpu.cfs_period_us
  cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.current 2>/dev/null \
    || cat /sys/fs/cgroup/memory/memory.limit_in_bytes /sys/fs/cgroup/memory/memory.usage_in_bytes
  nvidia-smi -L; df -h /root; } | tee /workspace/logs/b4s2_host.txt
```

Compute the CPU quota (quota / period), the memory limit in GB and the expected P. Report all three.

**Keep or replace the pod:**
- **CPU model (hard).** The limits are scaled from S1's host class (see "Time limits" below). Under B4b a host about 13%
  slower per core no longer crosses any modelled margin (the tightest is about 1.44x, F/F20 eval at a bootstrap share
  of 1.0). The rule is kept anyway: the cost models are estimates, and a replacement pod costs under $0.10.
  - Keep the pod only if the model name is `AMD EPYC-Genoa Processor` (S1's and the supplement's string) or an AMD EPYC
    9004 / 9005 model other than a 97x4 (Bergamo, lower clocks).
  - Anything else: delete the pod and create another. This includes Zen 3 (EPYC 7xx3), older EPYCs, Ryzen or
    Threadripper, and any Intel Xeon. Each try costs under $0.10.
  - After 3 pods without a Genoa-class CPU, stop and report. Running on a slower host is an owner decision.
- **P, not the quota alone.** Compute P = min(quota - 1, floor(free GB / 2.1)) for the 2x pod.
  - If it is 9 or fewer, the 2x pod gives no more parallelism than S1's 1x4090 host implies (4-9). Delete it and create
    a 1x4090 instead, at half the price, with the same CPU check.
  - Memory bounds P as well as the quota does, so judge by P.
- **Memory.** A memory-bound P only lengthens the run: about 2.5 h of lanes at P = 4, or about 4.8 h at a bootstrap
  share of 1.0. It does not change any result, so it is no reason to replace a 1x4090.

## 3. Code to P2

```bash
git -C /workspace/Atlas fetch origin && git -C /workspace/Atlas reset --hard <P2>      # dumps and results/ outputs kept
cd /workspace/Atlas && git log -1 --format='%H %s'                                     # must be <P2>
git -C /workspace/Atlas diff --stat cb617c3cf814b5080f4d11e6cbc38453da0831f4 HEAD        # only the 11 P2 files
ls -d results/instrument_check_b4s2* 2>/dev/null || echo "no S2 check dir yet"          # must print the echo
```

The diff must list exactly the files in "Files in P2" at the end. The pod's `results/` is ignored locally
(`.git/info/exclude`), and the S1 outputs there equal R1, so the reset only rewrites identical files.

## 4. Pin S1's software stack

`pod_atlas.sh` runs `pip install -r requirements.txt`, whose lower bounds would accept a newer scipy or scikit-learn
than S1 used. So install S1's exact versions first; pip then leaves them in place. Use its own ssh call:

```bash
cd /workspace/Atlas && V=results/instrument_check_b4s1/versions.json
PIP_BREAK_SYSTEM_PACKAGES=1 PIP_CACHE_DIR=/workspace/.cache/pip python -m pip install -q \
  "scipy==$(python -c "import json; print(json.load(open('$V'))['scipy'])")" \
  "scikit-learn==$(python -c "import json; print(json.load(open('$V'))['sklearn'])")" "pytest>=7"
python -c "import platform, numpy, scipy, sklearn, torch, torchvision; print(platform.python_version(), numpy.__version__, scipy.__version__, sklearn.__version__, torch.__version__, torchvision.__version__)"
cat "$V"
```

The printed versions must equal S1's: python 3.12.3, numpy 2.1.2, scipy 1.18.1, scikit-learn 1.9.1,
torch 2.8.0+cu128 and torchvision 0.23.0+cu128.
- If python, numpy, torch or torchvision differ, the image differs. Stop and report; do not launch.
- The replay compare (rule 6, tolerance rel 1e-6) would flag a different numerical stack. A replay FAIL tags every
  label `[REPLAY-DRIFT]`, and a drift the record cannot bound makes the decisions NOT_EVALUABLE.

## 5. Pre-launch seal check (B4a; hard)

Use its own ssh call. It checks that the manifest S2 will use lists both B4a dumps SEALED:

```bash
cd /workspace/Atlas && python - results/instrument_check_b4s1_r2 <<'PY'
import json, sys
d = sys.argv[1]
e = {(x["unit"], x["layout"]): x["status"] for x in json.load(open(f"{d}/sealed.json"))["entries"]}
miss = [u for u in ("mobilenetv2_x0_75", "shufflenetv2_x1_0") if e.get((u, "fit")) != "SEALED"]
sys.exit(f"B4a dumps not SEALED in {d}/sealed.json: {miss}" if miss else 0)
PY
echo "exit $?"
```

It must print `exit 0`. On Windows, the committed manifest holds 36 entries, all SEALED, including both B4a dumps.
Its sha256 is `213c03a5...` (see `freeze_P2.json` "b4a"). Otherwise do not launch; report.

## 6. Launch (detached)

Use its own ssh call:

```bash
cd /workspace/Atlas && mkdir -p /workspace/logs && \
export ATLAS_B4_P1=ad97d0a659dd9f6f60b24e0b8cc1681f1bd8a166 ATLAS_B4_P2=<P2> \
       ATLAS_B4S1_CHECK_DIR=results/instrument_check_b4s1_r2 && \
setsid -f bash /workspace/Atlas/pod_atlas.sh /workspace --b4s2 --b4cpu < /dev/null > /workspace/logs/b4s2.log 2>&1
```

**Flags:**
- `--b4s2` selects `block_b4s2`.
- `--b4cpu` only skips the GPU preflight. S2 has no GPU step, and every probe runs with `CUDA_VISIBLE_DEVICES=""`.
  This is B4a's launch line; `--b4s2` alone would also work on this GPU pod.

**Environment:**
- `ATLAS_B4S1_CHECK_DIR` is required. Without it the frozen default reads S1's own manifest, which lists the two
  B4a dumps as ABSENT, and the seal check would skip them silently.

**Leave unset:**
- `ATLAS_B4_S1_TAG`: the anchors' S1 outputs are under `''`.
- `ATLAS_B4_TAG` and `ATLAS_B4S2_CHECK_DIR` on the first launch: the defaults are `''` and
  `results/instrument_check_b4s2`.
- `ATLAS_B4_CUT`: no cut.
- `ATLAS_B4_UNSEAL`: `block_b4s2` sets it to P2 for the confirmation lane only.
- `ATLAS_REBUILD`.

## 7. Poll

Use its own ssh call. The brackets keep pgrep from matching itself:

```bash
pgrep -f "[b]ash /workspace/Atlas/pod_atlas.sh" >/dev/null && echo RUNNING || echo S2-ENDED
tail -n 20 /workspace/logs/b4s2.log
grep -h "lane:" /workspace/logs/b4s2.log                     # "[b4] confirmation lane: 66 jobs, <P> in parallel"
ST=$(grep -m1 -o 'pod_atlas_[0-9_]*' /workspace/logs/b4s2.log | sed 's/pod_atlas_//')
cat "/workspace/logs/soft_failures_$ST.log" 2>/dev/null        # this run's soft failures only
```

Poll every 10-15 minutes. The per-job logs are in `/workspace/logs/b4_$ST/`.

**Expected duration** (confirmation lane from the S1 timings; "central" assumes that a quarter of T1 target time is
bootstrap, which grows 5x at B = 1000; the range is for a bootstrap share of 0.1-0.5; "share 1.0", all T1 target time
bootstrap, is the case B4b's limits were sized for):

| stage | 2x4090, P = 16 | 1x4090, P = 8 (S1's host implies 4-9) | memory-bound, P = 4 |
|---|---|---|---|
| prelude: install, data check, guards, seal check, versions, pytest, 5 self-tests | 15-25 min | 15-25 min | 15-25 min |
| replay lane: 6 jobs; longest ~260 s | ~5 min | ~5 min | ~5 min |
| confirmation lane: 66 jobs; central (range) | 49 (40-64) min | 80 (66-103) min | 152 (125-197) min |
| confirmation lane at share 1.0 | 93 min | 147 min | 288 min |
| total with pod setup and pull | 1.2-1.7 h | 1.7-2.4 h | 2.7-4.0 h |
| total at share 1.0 | 2.0-2.3 h | 3.0-3.2 h | 5.3-5.6 h |
| cost | $1.8-2.5 at ~$1.48/h | $1.3-1.8 at $0.74/h | 2x: $4.0-5.9; 1x: $2.0-3.0 |
| cost at share 1.0 | $3.0-3.4 | $2.2-2.4 | 2x: $7.8-8.2; 1x: $3.9-4.1 |

- These assume S1's per-job speed. S2 runs more jobs at once than S1 did (4), so each may run slower (see "Time
  limits").
- **A lane up to about twice the central time is not a hang.** The bootstrap share is not measured (see "Time
  limits"), and at a share of 1.0 the confirmation lane takes about 1.9x its central time. Judge a job by its limit
  and the soft-failure log, not by this table.
- A relaunch (see "Relaunch rules") adds the prelude and the replay lane again, about 20-30 min, plus the jobs it
  reruns: about $0.5-1.5, or up to about $4 when it reruns a job to its full B4b limit.
- The confirmation work is about 10 CPU-h (range 8.2-13.0), and about 19 CPU-h at a share of 1.0.
- The longest single jobs set the wall time above P of about 16.
  - At the central share: shufflenetv2_x2_0 T1 fit (~27 min), then mobilenetv2_x1_4 T1 fit and the four F/F20 eval
    jobs (~22 min each).
  - From a share of about 0.45 the F/F20 eval jobs are the longest: ~31 min at 0.5 and ~48 min at 1.0. At 1.0 the R2
    eval jobs take ~38 min, and the shufflenetv2_x2_0 and mobilenetv2_x1_4 fits ~36 min each.

## What S2 does (`block_b4s2`, `scripts/pod_b4.sh`)

1. **`b4_guard_common`.**
   - The check dir `results/instrument_check_b4s2` is fresh.
   - Batch 4 runs alone.
   - The committed instruments are unchanged since 2b561a9.
   - P1 is an ancestor of HEAD, and no GPU job is running.
2. **`b4_guard_p2`.**
   - P1 <= P2 <= HEAD.
   - P2 holds the five `B4_P2_FILES`.
   - The frozen files and the P2 files are unchanged from P2 to HEAD.
   - The extraction code and the registry are unchanged from P1 to P2 (otherwise S2 refuses).
   - The probe code is unchanged from P1 to P2, so there is no re-probe.
   - `--verify-seals` runs against `results/instrument_check_b4s1_r2/sealed.json` (36 SEALED) -> `verify_seals.json`
     (hard).
3. **Versions.** `versions.json`, plus `versions_match.json` against S1 (soft).
4. **Tests.** The batch-4 pytest and the five synthetic self-tests (hard).
5. **Replay lane (rule 6).** T1 fit, T2 and T3S on both anchor hubs, at discovery settings (B = 200), tag
   `_s2replay`. Then `replay-compare` -> `replay.json` (soft; the evaluators read it).
6. **Confirmation lane.** `ATLAS_B4_UNSEAL=<P2>`. Only these jobs open the sealed dumps, and every open is logged to
   `unseal_log.jsonl`.
   - Touched-once holds per (program, unit, layout), not per dump. A dump read by several programs is opened by each
     of them, and a relaunch adds opens, so the log holds more records than there are sealed dumps.
   - B = 1000.
   - The jobs, in order:
     - T2 on C12, K4, F2, F20 2 (20);
     - T1 fit and eval on F and F20 (8);
     - T1 fit on C and K (16);
     - streams on STconf (10);
     - T1 eval on R2 (8);
     - T3S on Sconf (4).
7. **Timing.** `timing.json`.

A job that fails or is killed by its limit is soft: it writes nothing, and a `[soft] FAILED (exit N)` line is added. Exit
124 is a time-limit kill; 137 is usually the OOM killer.

## After the run

1. **Check the end of the log.**
   - `=== block block_b4s2 OK`. On FAILED, read the reason first.
   - The soft-failure list, if any.
   - `results/instrument_check_b4s2/replay.json` status.
   - `verify_seals.json` status.
   - The count of confirmation outputs, against 66.
2. **Pull without dumps (R2).**

   ```bash
   ssh ... 'tar -C /workspace/Atlas -czf - --exclude=dump --exclude="dump_step*" results' | tar -xzf - -C .
   ```

   This brings `results/instrument_check_b4s2/`: head, versions, verify_seals, pytest XML, self-tests, job lists,
   replay, unseal log and timing. It also brings the confirmation outputs under `results/b4_t1`, `b4_t1s`, `b4_t2` and
   `b4_t3s`, and the `*_s2replay` outputs.
3. **Delete the pod** and confirm with `runpodctl pod list`. S2 trains nothing, so there are no checkpoints to copy.
4. **Commit R2 on Windows** by explicit paths (`git status --short results/` lists them).
5. **Report** (a Korean summary per part):
   - the host: CPU model, quota, memory, P;
   - the guards and seals;
   - pytest and self-tests;
   - the replay status;
   - jobs done and failed, with names;
   - the largest wall time / limit ratio;
   - the cost.

   Claim statistics come from step E only.

## Step E (Windows, after R2)

`<S2 HEAD>` is `results/instrument_check_b4s2/head.txt`; it should equal `<P2>`. `--p-run` lists P1 and P_A beside
the S2 HEAD, because the discovery records (shufflenetv2_x0_5 under `_r2`) ran at P_A.

```bash
node scripts/b4a_p2.js seals --json results/b4/b4a_seals.json
node scripts/t1_eval.js --p1 ad97d0a659dd9f6f60b24e0b8cc1681f1bd8a166 --p2 <P2> \
  --p-run ad97d0a659dd9f6f60b24e0b8cc1681f1bd8a166,14caf51b7ca241a2dd95dacc531cc63a5bda6a6a,<S2 HEAD> \
  --json results/b4/eval_t1.json
node scripts/collapse_laws.js --evaluate --p1 ad97d0a659dd9f6f60b24e0b8cc1681f1bd8a166 --p2 <P2> \
  --p-run ad97d0a659dd9f6f60b24e0b8cc1681f1bd8a166,14caf51b7ca241a2dd95dacc531cc63a5bda6a6a,<S2 HEAD> \
  --replay results/instrument_check_b4s2/replay.json --json results/b4/t2_eval.json
node scripts/t3s_eval.js --phase confirmation --p1 ad97d0a659dd9f6f60b24e0b8cc1681f1bd8a166 --p2 <P2> \
  --p-run ad97d0a659dd9f6f60b24e0b8cc1681f1bd8a166,14caf51b7ca241a2dd95dacc531cc63a5bda6a6a,<S2 HEAD> \
  --sealed results/instrument_check_b4s1_r2/sealed.json --replay results/instrument_check_b4s2/replay.json \
  --json results/b4/eval_t3s.json
```

**The B4a seal check (`b4a_p2.js seals`) comes first and must PASS** before any result of mobilenetv2_x0_75 or
shufflenetv2_x1_0 is reported.
- It reads `freeze_P2.json` "b4a".check_dir.
- A failing unit is NOT_EVALUABLE for every claim, PRIMARY included.
- The repair is in `docs/plans/B4A_AMENDMENT.md`: a `--verify-seals` against the B4a manifest, which needs the volume.
- Until the check passes, report the claims from `t1_eval.js --cut "<units>"` and from `collapse_laws.js --root` on a
  copy of `results/` without their confirmation outputs; the run with them is INFO.

**The replay record.** `t1_eval.js` uses the newest `results/instrument_check_b4s2[_r<k>]` that holds a `replay.json`.
If S2 was relaunched, pass that same file to `--replay` in the other two.

**`results/b4/SESSION.md`**, one section per track, must also carry:
- P4 as "INFO (withdrawn at P2)", with its reason from `freeze_P2.json` "amendments". P4 keeps the evaluator's label
  beside it; `collapse_laws.js` still counts it in Holm over 22 items, which is conservative.
- X3-3 with its confirmation label (it stays ACTIVE) and, beside it, the harm-reference caveat from `freeze_P2.json`
  "disclosed".
- The SP-3 caveat and the SP-9 reading from `freeze_P2.json` "disclosed".
- B4a:
  - C = 12, where the frozen gate alone would have left 10 (mobilenetv2_x0_75 and shufflenetv2_x1_0 missing);
  - the `b4a_seals.json` result;
  - that no ShuffleNet timing value was raised.
- The host, P and the timing against the B4b limits and against the emitted ones (`timeouts_b4_emitted.json`).

Then ATLAS_STATUS and the inventory (the D19 rows).

## Relaunch rules (D10, B4a)

- **A guard refuses before the check dir exists** (`b4_guard_common`). Nothing was written. Fix the cause and launch
  again with the same command.
- **The check dir exists.** Every relaunch then needs the next unused S2 tag:

  ```bash
  ATLAS_B4S2_CHECK_DIR=results/instrument_check_b4s2_r2 ATLAS_B4_TAG=_r2    # then _r3, ...
  ```

  - Add these to the step 6 exports. `ATLAS_B4_P1`, `ATLAS_B4_P2` and `ATLAS_B4S1_CHECK_DIR` stay exactly the same.
  - `ATLAS_B4_S1_TAG` stays unset.
  - S2's tags are separate from S1's and the supplement's.
- **Stop and report; do not relaunch:**
  - a failed `--verify-seals` (a sealed dump changed or is missing);
  - a failed pytest or self-test;
  - a `b4_guard_p2` refusal about the P2 files or the extraction code.

  These need a decision by the lead or owner. Nothing sealed was opened before these steps.
- **A replay job failed or was killed** (`replay.json` not PASS for a missing unit). Relaunch with the next tag.
  - The replay lane runs again, under `_r2_s2replay`, and `replay-compare` still compares with S1's `''` outputs.
  - Every confirmation job that already wrote its output refuses (touched-once) before opening a dump. Those refusals
    appear as soft failures in the relaunch; they are expected.
  - A missing or killed anchor replay makes the drift unbounded, and then every threshold decision is NOT_EVALUABLE.
    So this relaunch matters.
- **A confirmation job crashed or was OOM-killed** (exit other than 124). Relaunch with the next tag. Only the missing
  jobs run.
- **A confirmation job was killed by its time limit** (exit 124). Relaunch once with the next tag, as D10 provides
  ("a killed unit writes nothing; the relaunch uses ATLAS_B4_TAG=_r2 with unchanged code"). `timeouts.json` stays
  unchanged.
  - Only the missing jobs run. Every job that already wrote its output refuses (touched-once) before opening a dump, so
    the killed job reruns almost alone, without the contention of the first launch.
  - Under B4b a kill is not expected: every modelled margin is at least about 1.44x even if all target time were
    bootstrap (see "Time limits"). A kill points to something the models leave out, such as a much slower host or heavy
    contention; within S1 the same probe's per-target time varied 1.3-1.5x with load.
  - One relaunch covers every job that was killed or crashed in the run before it.
  - A job killed by its limit in the relaunch too is not relaunched again. Report it; its unit is NOT_EVALUABLE for the
    claims that need that output.
  - Raising a limit would change a P2 file after S2 has unsealed, which D8 does not provide for. That needs an owner
    decision. (B4b raised them at P2, before anything sealed was opened.)
  - Cost: the prelude and the replay lane again (about 20-30 min) plus at most the job's limit: about $1-2.5 for an
    eval-layout job (limits 3618-4252 s), and up to about $4 on the 2x pod for the largest limit (8324 s).
- **The pod was stopped or restarted.** Redo step 4 before any relaunch, because the pinned stack was on the container
  disk.
- **Never:**
  - delete or rename an output;
  - change `ATLAS_B4_P2`;
  - set `ATLAS_B4_UNSEAL` by hand;
  - `git pull` / `reset` while `pod_atlas.sh` runs.

## Time limits (`experiments/b4/timeouts.json`)

- **The rule.** B4b (`docs/plans/B4B_AMENDMENT.md`, approved by the owner on 2026-09-24): every limit is
  min(14400, 2 x the limit the frozen `b4_timeouts.js` emitted), never lower.
  - The emission is kept unchanged as `experiments/b4/timeouts_b4_emitted.json`. `scripts/b4b_timeouts.js` wrote
    `timeouts.json` from it; `peak_gb`, and so P, is unchanged.
  - The emitted rule is D10's: 3x the slowest S1 run of the same program and family, scaled by width^2 and rows,
    clamped to [1800, 14400] s.
- **No SPLIT.** `b4_timeouts.js` flagged none: its largest limit was shufflenetv2_x2_0 T1 fit at 4162 s, 29% of the
  14,400 s cap. No B4b limit reaches the cap, so each is exactly 2x the emitted one. The largest is that job at 8324 s
  (58% of the cap). Next is mobilenetv2_x1_4 T1 fit at 6222 s. Every T2, streams and T3S job is at 3600 s.
- **What the emitted rule leaves out.** It scales S1's fit-layout time by penult width and rows only.
  - It leaves out confirmation's B = 1000 bootstrap draws (200 in S1) and the 3-5 confirmation-only targets. No eval
    layout was timed in S1.
  - Its row factor divides by the reference dump's 122,500 rows, but S1's discovery run read only 88,000 of them:
    discovery refuses the 34,500 CONFIRMATION_ONLY rows. The rows read grow 1.57x (C/K/F fits), 2.77x (R2 eval) and
    3.22x (F/F20 eval), against the rule's 1.13x, 1.87x and 2.19x. The estimates below model that growth per component.
- **The bootstrap share is not measured.** An operation count gives about 0.2-0.33 of T1 target time at B = 200: about
  21 `b4_core.boot_ci` calls per target, each draw 25-30 full-length numpy passes. (A first count of 0.13 was wrong.)
  So the central share of 0.25 is a central value, not a conservative one, and 0.5 is plausible.
- **The tightest limits are the 12 T1 eval-layout jobs:** F/F20 at 4252 s and R2 at 3618 s (emitted: 2126 s and
  1809 s). Their `t_ref` is the resnet32 fit, 323 s. Three cost models (the P2 lead's and two reviews') give:

  | jobs | share 0.25 | share 0.5 | share 1.0 |
  |---|---|---|---|
  | F/F20 eval (4) | about 1310-1370 s, margin 3.10-3.25x | about 1830-1860 s, margin 2.29-2.32x | about 2750-2960 s, margin 1.44-1.55x |
  | R2 eval (8) | margin 3.12-3.73x | margin 2.26-2.75x | margin 1.46-1.80x |

  - At a share of 1.0 all target time is bootstrap, so no limit is reached at any share.
  - The share-1.0 column re-runs the P2 lead's model and the first review's. The second review's is extrapolated
    linearly from its 0.25 and 0.5 values.
  - Under the emitted limits the margins were 1.55-1.62x and 1.14-1.16x (F/F20) and 1.56-1.86x and 1.13-1.37x (R2),
    and the limit was reached from a share of about 0.6-0.7 (F/F20) and about 0.6-0.85 (R2), depending on the model.
    That is why B4b was adopted.
- **Next: the six wide C T1 fits** (mobilenetv2_x0_75, mobilenetv2_x1_0, repvgg_a1, repvgg_a2, shufflenetv2_x1_0,
  shufflenetv2_x1_5; limits 3600-3862 s, emitted 1800-1931 s).
  - The width rule scales by penult width only, but their earlier taps are 2.4-3.7x wider than the reference's.
  - Margin 3.72-4.28x at a share of 0.25, 2.96-3.30x at 0.5, and 2.10-2.24x at 1.0 (emitted: 1.86-2.14x, 1.48-1.65x
    and 1.05-1.12x).
- **Every other job** has at least about 4.6x at a share of 0.25 in the P2 lead's model.
- **Load.** S1's reference times come from a lane of 4 probes at once (P = 4 from the default 4 GB peak), niced beside
  the GPU chain and the trainings.
  - Within S1 the same probe's per-target time varied with load. The 18 T1 fits started 07:03-07:26 took a median of
    5.8 s per target. The four started 07:28-07:30 took 7.6-7.7 s; resnet32 was one of them. The supplement's ShuffleNet
    fit took 5.1 s on an idle pod.
  - S2 runs up to 16 probes at once, without trainings. So S2's per-job speed is not established to be better than
    S1's.
  - Under B4b the smallest margin at a share of 0.5, about 2.26x, exceeds that 1.3-1.5x load effect combined with a
    host about 13% slower per core (about 1.7x).
- **What protects the margins:**
  - B4b's doubled limits (above);
  - the CPU check in section 2 (EPYC-Genoa class only), which keeps S2 on the host class the `t_ref` values were
    measured on;
  - one relaunch of a job killed by its limit (see "Relaunch rules"), which then runs almost alone.

## Files in P2

- `experiments/b4/freeze_P2.json`
- `experiments/b4/t1_rules.json` (every claim ACTIVE, X3-3 included)
- `experiments/b4/laws_frozen.json`
- `experiments/b4/timing_b4a.json`
- `experiments/b4/timeouts_b4_emitted.json` (the frozen `b4_timeouts.js` emission)
- `experiments/b4/timeouts.json` (B4b: 2x the emission)
- `results/b4/eval_discovery.json`
- `results/b4/eval_t3s_discovery.json`
- `results/b4/RUN_REQUEST_S2.md` (this file)
- `docs/plans/B4B_AMENDMENT.md`
- `scripts/b4b_timeouts.js`

## Checks done at P2 (Windows)

- **Node self-tests and fixtures:**
  - `b4_stats.js --selftest` 14/14;
  - `b4_timeouts.js --selftest` 10/10;
  - `b4b_timeouts.js --selftest` 10/10;
  - `b4_make_registry.js --check` (43 units);
  - `tests/t1_eval_fixture.js` 176;
  - `collapse_laws.js --selftest` 20/20;
  - `tests/collapse_laws_fixture.js` 71/71;
  - `t3s_eval.js --selftest` 49/49;
  - `b4a_p2.js --selftest` 25/25;
  - `node --check` on every batch-4 script;
  - `bash -n` on `pod_atlas.sh`, `scripts/pod_b4.sh` and `scripts/b4a_supplement.sh`.
- **Reproducibility.** Every emitted file was re-emitted into a scratch directory from the same inputs.
  `timeouts_b4_emitted.json`, and `timeouts.json` from it (B4b), came out byte-identical; the others differed only in
  their timestamps.
  - `b4b_timeouts.js` hashes its input CRLF-normalised, the batch-4 convention.
  - A fresh Windows checkout (`core.autocrlf=true`) has CRLF line ends (91 CRs in `timeouts_b4_emitted.json`). A run
    there still recorded source sha256 `9ca33120...` and wrote exactly the committed bytes of `timeouts.json`.
- **After review.**
  - X3-3 stays ACTIVE (lead decision). `t1_rules.json` and `eval_discovery.json` were re-emitted in one discovery run
    without `--amend` (31 claims ACTIVE). Each equals the earlier unamended emission apart from `created`. The
    harm-reference concern is in `freeze_P2.json` "disclosed".
  - B4b (owner-approved): the emission was renamed to `timeouts_b4_emitted.json`, and `b4b_timeouts.js` wrote
    `timeouts.json` from it.
  - `freeze_P2.json` was rebuilt with the new sha256, and the node checks above were run again with the same counts.
  - Fixes after the review of the P2 set:
    - `b4b_timeouts.js` hashed its input's raw bytes, so a CRLF checkout recorded another source sha256. It now hashes
      them CRLF-normalised.
    - Its `rule` key was the emitted rule, which the doubled values contradict. It now states the B4b rule and quotes
      the emitted one.
    - `timeouts.json` was rewritten from the same emission. Only `rule` changed: every limit, `peak_gb`, `split`,
      `unmeasured` and `amendment` are as before. The 72 job lines were checked again: each maps to a key equal to 2x
      its emitted limit.
    - `freeze_P2.json` lists `timing_b4a.json` under B4a in `d8_changes.amendment`, not under D8 (a).
      `b4a_p2.js`, which writes it, is a P_A file, not frozen code.
    - Corrected figures: the R2 margins (3.73x at 0.25, 2.75x at 0.5), the share at which the emitted R2 limit is
      reached (about 0.6-0.85), and the duration and cost at a bootstrap share of 1.0 in section 7.
    - `freeze_P2.json` was rebuilt again; all 17 sha256 values it records match the files. The node checks above were
      run again: `b4b_timeouts.js --selftest` is now 10/10, and every other count is unchanged.
  - The confirmation-phase rules reader (`t1_eval.js` `rulesFor`) parses `t1_rules.json`: 31 claims, all ACTIVE, no
    fraction looser than P1. A confirmation-phase `evaluate()` on an empty results directory, with the P2 blob stubbed
    to these bytes, passes the rules gate; with any other P2 sha256 the gate fails.
- **Unchanged since P1.** Every file in `experiments/b4/frozen_files.txt`, plus `pod_atlas.sh` and `scripts/pod_b4.sh`,
  is identical to P1. The D6 committed instruments are identical to 2b561a9.
- **The S2 guards, simulated** in a scratch clone with a temporary P2 commit of the 11 files above (re-run on the
  final set, 15/15). `b4_guard_common` and `b4_guard_p2` from the frozen `scripts/pod_b4.sh` were sourced, with
  python stubbed for the seal check. The clone has an LF worktree, as on the pod: in a CRLF worktree,
  `$(cat "$B4_FROZEN")` gives paths that end in CR, and the frozen-file check then passes without comparing anything.
  - Both pass at HEAD = P2, and at a later HEAD.
  - They refuse when a frozen file or `timeouts.json` changes after P2, when the extraction code or the registry changed
    P1 -> P2, and when a P2 file is missing.
  - A probe-code change P1 -> P2 sets the re-probe.
- **Nothing sealed was read.** No confirmation-phase evaluation of any output was run (the rules-reader check above
  ran on an empty results directory).
