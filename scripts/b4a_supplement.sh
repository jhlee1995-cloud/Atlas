#!/usr/bin/env bash
# scripts/b4a_supplement.sh -- the B4a supplement of batch-4 session 1 (docs/plans/B4A_AMENDMENT.md). On the S1 pod, AFTER
# block_b4s1 has ended, at the amendment commit P_A: the weights gate with the one-sided README rule
# (scripts/b4_weights_amend.py) and the extraction of exactly the three units the S1 README gate refused
# (mobilenetv2_x0_75 and shufflenetv2_x1_0: C, sealed fit; shufflenetv2_x0_5: Dnew, open fit), the S1 discovery probes of
# shufflenetv2_x0_5, a complete seal manifest and the timing. Everything under ATLAS_B4_TAG=_r2 (default) with the check
# dir results/instrument_check_b4s1<tag> (the D10 relaunch conventions).
# Standalone: it sets what scripts/pod_b4.sh expects from pod_atlas.sh's prelude (VOLUME, SOFT_LOG, STAMP, soft,
# cpu_quota, has_mode, the caches on the volume) and sources scripts/pod_b4.sh; neither file changes (guarded against P1).
#   git -C /workspace/Atlas fetch origin && git -C /workspace/Atlas reset --hard <P_A>      (only after block_b4s1 ended)
#   export ATLAS_B4_P1=<P1>
#   setsid -f bash /workspace/Atlas/scripts/b4a_supplement.sh /workspace < /dev/null > /workspace/logs/b4a.log 2>&1
# Order: guards (the stack equal to S1's is checked here, before the check dir exists) -> versions record (hard) ->
# pytest (hard) -> S1 seals verified (hard) -> b4_weights_amend.py -> quota -> extraction of the 3 units (soft per unit)
# -> discovery probes of shufflenetv2_x0_5 (soft) -> complete seal manifest + diff against S1's (hard) + verify ->
# timing -> summary.
# Append-only; it never deletes anything: every output is new (the guards refuse otherwise), a failed unit is logged in
# the soft-failure log and the script exits 1 at the end, as block_b4s1 does.
# Relaunch after a soft failure: ATLAS_B4_TAG=<the next unused _r<k>> (no results/instrument_check_b4s1_r<k> and no
# results/b4_weights/weights_r<k>.json exist yet; an S1 relaunch takes tags from the same sequence). A complete dump
# written by an earlier supplement run (its meta.json carries the B4a weights gate) is skipped by the extractor, a
# discovery job whose output exists under an earlier _r<k> tag is skipped, and the new check dir gets a complete seal
# manifest again; S2 then reads the newest check dir with a complete sealed.json (docs/plans/B4A_AMENDMENT.md). A partial
# dump needs ATLAS_B4_MOVE_PARTIAL=1 (renamed aside by the extractor).
# Env: ATLAS_B4_P1 (required); ATLAS_B4_TAG (default _r2); ATLAS_B4_S1_TAG (the tag S1 ran under, default '');
# ATLAS_B4_MOVE_PARTIAL; ATLAS_ALLOW_LOCAL=1 skips the network-volume guard (stubbed dry runs only).
set -euo pipefail
: "${ATLAS_B4_P1:?usage: ATLAS_B4_P1=<P1> bash scripts/b4a_supplement.sh <volume>}"
VOLUME="$(realpath -m "${1:?usage: ATLAS_B4_P1=<P1> bash scripts/b4a_supplement.sh <volume>}")"
shift
if (( $# )); then echo "ERROR: b4a_supplement.sh takes no flags (got: $*)" >&2; exit 1; fi
cd "$(dirname "$(readlink -f "$0")")/.."                       # repo root, whatever the caller's cwd

# --- pod_atlas.sh guards: the volume is a network volume, and the repo lives on it ---
if [[ "${ATLAS_ALLOW_LOCAL:-0}" != 1 ]]; then
  SRC="$(findmnt -no SOURCE --target "$VOLUME" 2>/dev/null || true)"
  [[ "$SRC" == mfs#* ]] || { echo "ERROR: $VOLUME is not a network volume (mount source: '${SRC:-none}')" >&2; exit 1; }
fi
case "$(pwd -P)/" in
  "$(cd "$VOLUME" && pwd -P)"/*) ;;
  *) echo "ERROR: repo is at $(pwd -P), outside $VOLUME; results/ would be lost on pod stop" >&2; exit 1;;
esac

# --- the prelude scripts/pod_b4.sh expects (as pod_atlas.sh sets it) ---
export PIP_BREAK_SYSTEM_PACKAGES=1 PYTHONUNBUFFERED=1
export PIP_CACHE_DIR="$VOLUME/.cache/pip" TORCH_HOME="$VOLUME/.cache/torch"   # the S1 torch.hub cache (pinned ref, weights)
export HF_HOME="$VOLUME/.cache/huggingface" XDG_CACHE_HOME="$VOLUME/.cache"
mkdir -p "$VOLUME/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$VOLUME/logs/b4a_supplement_$STAMP.log"
SOFT_LOG="$VOLUME/logs/soft_failures_b4a_$STAMP.log"             # one line per failed soft() step
exec > >(tee -a "$LOG") 2>&1
B4A_DONE=0                                                        # 1 once every hard step has passed
b4a_on_exit() {
  local rc=$?
  if (( rc && ! B4A_DONE )); then echo "=== B4a supplement STOPPED at a hard step (exit $rc) $(date); log $LOG ==="; fi
}
trap b4a_on_exit EXIT
echo "=== B4a supplement | log $LOG | git $(git rev-parse --short HEAD 2>/dev/null || echo NO_GIT) | $(date) ==="

MODES=" "                                                         # no pod_atlas.sh mode is active in the supplement
has_mode() { [[ "$MODES" == *" $1 "* ]]; }
soft() {                                                          # = pod_atlas.sh soft()
  "$@" || { echo "[soft] FAILED (exit $?): $*" | tee -a "$SOFT_LOG"; }
}
cpu_quota() {                                                     # = pod_atlas.sh cpu_quota()
  local c q p
  c=$(nproc)
  if [[ -r /sys/fs/cgroup/cpu.max ]]; then                       # cgroup v2: "<quota> <period>" or "max <period>"
    read -r q p < /sys/fs/cgroup/cpu.max
    if [[ "$q" != max ]]; then c=$(( q / p )); fi
  elif [[ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]]; then        # cgroup v1 (RunPod 4090 hosts): -1 = unlimited
    q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us); p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
    if (( q > 0 )); then c=$(( q / p )); fi
  fi
  echo "$c"
}

export ATLAS_B4_TAG="${ATLAS_B4_TAG:-_r2}"
# shellcheck source=scripts/pod_b4.sh
source scripts/pod_b4.sh                                          # B4_TAG=$ATLAS_B4_TAG; b4_extract, b4_reg, b4_quota, ...

# Volume ceiling. pod_b4.sh's b4_quota stops at 48 GB (the 50 GB volume). The owner authorised enlarging the network
# volume (2026-09-24); after `runpodctl network-volume update kxfir1tryb --size <N>` the operator sets
# ATLAS_B4A_QUOTA_GB=<N - 2>. Unset, the ceiling stays 48 GB. Only the ceiling differs from pod_b4.sh's b4_quota.
B4A_QUOTA_GB="${ATLAS_B4A_QUOTA_GB:-48}"
if ! [[ "$B4A_QUOTA_GB" =~ ^[0-9]+$ ]] || (( B4A_QUOTA_GB < 48 || B4A_QUOTA_GB > 500 )); then
  echo "ERROR: ATLAS_B4A_QUOTA_GB must be an integer in [48, 500] (got '$B4A_QUOTA_GB')" >&2; exit 1
fi
b4_quota() {     # b4_quota <GB this group still writes>: du against the volume ceiling (B4A_QUOTA_GB)
  local need="$1" used
  used=$(du -sb "$VOLUME" 2>/dev/null | cut -f1 || true)
  [[ "$used" =~ ^[0-9]+$ ]] || used=0
  echo "[b4a] volume holds $(( used / 1000000000 )) GB; this group writes <= ${need} GB; ceiling ${B4A_QUOTA_GB} GB"
  if (( used + need * 1000000000 > B4A_QUOTA_GB * 1000000000 )); then
    echo "[b4a] QUOTA: $(( used / 1000000000 )) + ${need} GB > ${B4A_QUOTA_GB} GB: group skipped" | tee -a "$SOFT_LOG"; return 1
  fi
}

B4A_UNITS="mobilenetv2_x0_75 shufflenetv2_x0_5 shufflenetv2_x1_0"   # = scripts/b4_weights_amend.py UNITS (registry order)
B4A_DISC="shufflenetv2_x0_5"                                        # Dnew: open fit dump; its S1 discovery probes run here
B4A_SEALED="mobilenetv2_x0_75 shufflenetv2_x1_0"                    # C: sealed fit dumps, never probed before P2
S1_TAG="${ATLAS_B4_S1_TAG:-}"
S1D="results/instrument_check_b4s1$S1_TAG"                          # block_b4s1's check dir (read only)
S1W="results/b4_weights/weights$S1_TAG.json"                        # block_b4s1's weights record (read only)
I="results/instrument_check_b4s1$B4_TAG"                            # the supplement's check dir (new)

B4A_JOBS_RE='[b]4_weights|[b]4_extract\.py|[t]rain_second_seed\.py|[b]4_train_knob\.py|[t]1_scoreboard\.py'
B4A_JOBS_RE+='|[t]1_streams\.py|[c]ollapse_probe\.py|[t]3s_spatial_probe\.py|[c]heck_rebuild\.py'   # GPU jobs and probes

nonempty_dir() { [[ -d "$1" ]] && [[ -n "$(ls -A "$1" 2>/dev/null)" ]]; }
b4a_outputs() {  # b4a_outputs <id> <tag>: every probe output file a unit can have under one tag (scripts/b4_reg.py OUT)
  echo "results/b4_t1/${1}_fit$2/scoreboard.json results/b4_t1/${1}_eval$2/scoreboard.json results/b4_t2/$1$2/probe.json"
  echo "results/b4_t1s/$1$2/streams.json results/b4_t3s/$1$2/probe.json"
}

b4a_guards() {
  local id f d s1head
  echo "=== B4a: guards ==="
  if ! [[ "$B4_TAG" =~ ^(_r[0-9]+)+$ ]] || [[ "$B4_TAG" == "$S1_TAG" ]]; then
    echo "ERROR: ATLAS_B4_TAG='$B4_TAG' must be a relaunch tag _r<k> other than the S1 tag '$S1_TAG'" >&2; return 1
  fi
  if [[ -e "$I" ]]; then
    echo "ERROR: $I exists: relaunch the supplement with the next unused tag (ATLAS_B4_TAG=_r<k>: no such check dir" \
         "and no results/b4_weights/weights_r<k>.json)" >&2; return 1
  fi
  # fresh check dir, no ATLAS_REBUILD, P1 <= HEAD, committed instruments unchanged since B4_BASE, no GPU job running
  b4_guard_common "$I" || return 1
  # shellcheck disable=SC2046
  if ! git diff --quiet "$ATLAS_B4_P1" HEAD -- $(cat "$B4_FROZEN") pod_atlas.sh scripts/pod_b4.sh; then
    echo "ERROR: a frozen file, pod_atlas.sh or scripts/pod_b4.sh differs from P1 (B4a changes none of them):" >&2
    # shellcheck disable=SC2046
    git diff --stat "$ATLAS_B4_P1" HEAD -- $(cat "$B4_FROZEN") pod_atlas.sh scripts/pod_b4.sh >&2; return 1
  fi
  if ! git diff --quiet HEAD -- || ! git diff --cached --quiet; then
    echo "ERROR: tracked files differ from HEAD; the supplement runs a commit (git reset --hard <P_A>)" >&2; return 1
  fi
  for f in docs/plans/B4A_AMENDMENT.md scripts/b4_weights_amend.py scripts/b4a_supplement.sh scripts/b4a_p2.js \
           tests/test_b4a_amendment.py; do
    if ! git cat-file -e "HEAD:$f" 2>/dev/null; then
      echo "ERROR: HEAD lacks $f: not the amendment commit P_A" >&2; return 1
    fi
  done
  if ! command -v pgrep >/dev/null; then
    echo "ERROR: pgrep missing: cannot check that block_b4s1 has ended" >&2; return 1
  fi
  if pgrep -f '[p]od_atlas\.sh' >/dev/null; then                  # brackets: pgrep never matches its own pattern
    echo "ERROR: pod_atlas.sh is still running: start the supplement after block_b4s1 has ended" >&2; return 1
  fi
  if pgrep -f "$B4A_JOBS_RE" >/dev/null; then
    echo "ERROR: a batch-4 job (GPU or discovery lane) is still running" >&2; return 1
  fi
  for f in "$S1D/sealed.json" "$S1D/versions.json" "$S1D/head.txt" "$S1W"; do
    if [[ ! -s "$f" ]]; then
      echo "ERROR: $f missing: block_b4s1 did not reach its seal manifest (report; no supplement)" >&2; return 1
    fi
  done
  s1head=$(cat "$S1D/head.txt")
  if ! git merge-base --is-ancestor "$s1head" HEAD; then
    echo "ERROR: S1's HEAD $s1head is not an ancestor of HEAD" >&2; return 1
  fi
  if [[ ! -f "$S1D/timing.json" ]]; then
    echo "[b4a] note: $S1D/timing.json absent (block_b4s1 did not reach its wrap-up)"
  fi
  for id in $B4A_UNITS; do
    if [[ " $(b4_ids C Dnew) " != *" $id "* ]]; then
      echo "ERROR: $id is cut (ATLAS_B4_CUT) or not a C / Dnew unit" >&2; return 1
    fi
    d=$(b4_reg get "$id" layouts.fit.dump) || return 1
    if [[ -f "$d/meta.json" ]]; then
      if ! grep -qF '"amendment": "B4a"' "$d/meta.json"; then
        echo "ERROR: $d is a complete dump without the B4a weights gate: B4a is decided before any dump of its units" >&2
        return 1
      fi
      echo "[b4a] note: $d is complete from an earlier supplement run (B4a gate): the extractor skips it"
    elif nonempty_dir "$d" && [[ "${ATLAS_B4_MOVE_PARTIAL:-0}" != 1 ]]; then
      echo "ERROR: $d is a partial dump: relaunch with ATLAS_B4_MOVE_PARTIAL=1 (renamed aside, never deleted)" >&2; return 1
    fi
    for f in "results/b4_extract/${id}_fit$S1_TAG.json" "results/b4_extract/${id}_fit$B4_TAG.json"; do
      if [[ -e "$f" ]]; then echo "ERROR: $f exists" >&2; return 1; fi
    done
    for f in $(b4a_outputs "$id" "$S1_TAG"); do
      if [[ -f "$f" ]]; then
        echo "ERROR: $f exists: a complete output of $id under the S1 tag '$S1_TAG'" >&2; return 1
      fi
      if nonempty_dir "$(dirname "$f")"; then
        echo "[b4a] note: $(dirname "$f") holds an incomplete S1 output (kept; no evaluator reads it):" \
             "$(ls -A "$(dirname "$f")" | tr '\n' ' ')"
      fi
    done
    for f in $(b4a_outputs "$id" "$B4_TAG"); do
      if nonempty_dir "$(dirname "$f")"; then echo "ERROR: $(dirname "$f") exists and is not empty" >&2; return 1; fi
    done
  done
  for f in "results/b4_weights/weights$B4_TAG.json" "results/b4_weights/heads$B4_TAG"; do
    if [[ -e "$f" ]]; then echo "ERROR: $f exists" >&2; return 1; fi
  done
  python -c "import torch; assert torch.cuda.is_available(), 'the supplement extracts: GPU only (D17)'" || return 1
  # the stack S1 ran on, checked BEFORE the check dir exists, so a refusal keeps the tag: a stopped and restarted pod
  # has lost the pip-installed scipy, sklearn and pytest (container disk; results/b4/RUN_REQUEST_S1_SUPPLEMENT.md)
  if ! python -c "import numpy, scipy, sklearn, pytest"; then
    echo "ERROR: numpy, scipy, sklearn or pytest not importable: pod restarted? (the run request's recovery step)" >&2
    return 1
  fi
  mkdir -p "$VOLUME/logs/b4a_$STAMP"
  b4_versions "$VOLUME/logs/b4a_$STAMP/versions_guard.json" || return 1
  if ! b4a_stack_cmp "$VOLUME/logs/b4a_$STAMP/versions_guard.json"; then
    echo "ERROR: the software stack differs from S1's: pod restarted or packages changed (the run request's" \
         "recovery step)" >&2; return 1
  fi
  echo "[b4a] guards PASS: HEAD $(git rev-parse HEAD); P1 $ATLAS_B4_P1; S1 HEAD $s1head; tag $B4_TAG; check dir $I"
}

b4a_stack_cmp() {   # b4a_stack_cmp <versions.json>: the stack must be S1's (one instrument, D2)
  python - "$S1D/versions.json" "$1" <<'PY'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
keys = ("python", "numpy", "scipy", "sklearn", "torch", "torchvision", "device", "cudnn_allow_tf32", "matmul_allow_tf32")
diff = {k: [a.get(k), b.get(k)] for k in keys if a.get(k) != b.get(k)}
print("[b4a] stack equal to S1's" if not diff else f"[b4a] STACK DIFFERS FROM S1: {diff}")
sys.exit(1 if diff else 0)
PY
}

b4a_versions() {   # the record in the check dir (hard; the guard already compared the stack), the S2-style one beside it
  b4_versions "$I/versions.json"
  b4a_stack_cmp "$I/versions.json"
  soft b4_reg versions-match --a "$S1D/versions.json" --b "$I/versions.json" --out "$I/versions_match_s1.json"
}

b4a_gb() {   # GB the three fit dumps write: registry gb, +15 % and ceil (as b4_reg.py gb)
  local id g=""
  for id in $B4A_UNITS; do g+=" $(b4_reg get "$id" layouts.fit.gb)"; done
  awk -v s="$g" 'BEGIN{n=split(s,a," "); t=0; for(i=1;i<=n;i++)t+=a[i]; t*=1.15; c=int(t); if(c<t)c++; print c}'
}

b4a_lane() {   # b4a_lane <check dir> <id>: b4_cpu_lane's discovery settings for ONE unit (the full lane would re-probe all)
  local I="$1" id="$2" jobs all P n m=0 s=0 line out f done_as want=() miss=()
  jobs="$I/jobs_discovery.txt"; all="$VOLUME/logs/b4a_$STAMP/jobs_discovery_all.txt"
  mkdir -p "$VOLUME/logs/b4a_$STAMP"
  if ! b4_reg --logdir "$VOLUME/logs/b4a_$STAMP" jobs --phase discovery --tag "$B4_TAG" > "$all"; then
    echo "[b4a] ERROR: b4_reg.py jobs --phase discovery failed: no discovery job of $id queued"; return 1
  fi
  : > "$jobs"
  while IFS= read -r line; do                  # this unit's jobs; a job done under an earlier _r<k> tag is skipped
    m=$(( m + 1 ))
    out=$(sed -n 's/.* --out \([^ ]*\).*/\1/p' <<< "$line")
    case "$out" in results/b4_t1/*) f=scoreboard.json;; results/b4_t1s/*) f=streams.json;; *) f=probe.json;; esac
    done_as=$(compgen -G "${out%"$B4_TAG"}_r*/$f" | head -n 1 || true)
    if [[ -n "$done_as" ]]; then
      echo "[b4a] skip: $done_as exists (an earlier supplement run)"; s=$(( s + 1 ))
    else
      echo "$line" >> "$jobs"; want+=("$out/$f")
    fi
  done < <(grep -F -e " --unit $id " "$all" || true)
  if (( m == 0 )); then                        # no job line of the unit: the generator is broken, never "all done"
    echo "[b4a] ERROR: no discovery job line of $id in $all"; return 1
  fi
  n=$(wc -l < "$jobs")
  if (( n == 0 )); then echo "[b4a] all $m discovery jobs of $id are done under an earlier _r<k> tag"; return 0; fi
  P=$(b4_P "$(b4_reg peak --phase discovery)")
  echo "[b4a] discovery lane: $n jobs of $id ($s done earlier), $P in parallel (order: $jobs; logs: $VOLUME/logs/b4a_$STAMP)"
  ATLAS_B4_CHECK_DIR="$I" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    xargs -d '\n' -P "$P" -n 1 nice -n 19 bash -c 'eval "$1" || echo "[soft] FAILED (exit $?): $1" >> "$SOFT_LOG"' _ < "$jobs"
  for f in "${want[@]}"; do [[ -f "$f" ]] || miss+=("$f"); done
  if (( ${#miss[@]} )); then echo "[b4a] discovery output missing after the lane: ${miss[*]}"; return 1; fi
}

b4a_seal_diff() {   # the complete manifest must be S1's plus the new B4a sealed dumps, nothing else (hard)
  # shellcheck disable=SC2086
  python - "$S1D/sealed.json" "$I/sealed.json" "$I/seal_diff.json" $B4A_SEALED <<'PY'
import hashlib, json, os, sys
s1p, newp, outp, allow = sys.argv[1], sys.argv[2], sys.argv[3], set(sys.argv[4:])
key = lambda e: (e["unit"], e["layout"])                                      # noqa: E731
A = {key(e): e for e in json.load(open(s1p))["entries"]}
B = {key(e): e for e in json.load(open(newp))["entries"]}
bad, added, kept = [], [], 0
for k in sorted(set(A) | set(B)):
    a, b = A.get(k), B.get(k)
    if a is None or b is None:
        bad.append(f"{k[0]}:{k[1]} in one manifest only")
    elif a["status"] == "SEALED":
        same = b["status"] == "SEALED" and all(a.get(f) == b.get(f) for f in ("dump", "meta_sha256", "n_files", "bytes"))
        kept += int(same)
        if not same:
            bad.append(f"{k[0]}:{k[1]} S1 seal changed: {a.get('meta_sha256')} -> "
                       f"{b.get('status')} {b.get('meta_sha256')}")
    elif b["status"] == "SEALED":
        if k[0] in allow and k[1] == "fit":
            added.append(f"{k[0]}:{k[1]}")
        else:
            bad.append(f"{k[0]}:{k[1]} sealed after S1 but not a B4a unit")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()              # noqa: E731
rep = {"schema": "b4a_seal_diff/1", "rule": "docs/plans/B4A_AMENDMENT.md: S1's sealed entries unchanged; only the B4a "
       "sealed fit dumps may be added", "s1_manifest": s1p, "s1_sha256": sha(s1p), "manifest": newp, "sha256": sha(newp),
       "s1_sealed_unchanged": kept, "added": added,
       "b4a_absent": sorted(f"{u}:fit" for u in allow if (B.get((u, "fit")) or {}).get("status") != "SEALED"),
       "problems": bad, "status": "PASS" if not bad else "FAIL"}
if os.path.exists(outp):
    sys.exit(f"{outp} exists: never overwritten")
with open(outp, "w") as f:
    json.dump(rep, f, indent=1)
print(f"[b4a] seal diff {rep['status']}: {kept} S1 seals unchanged; added {added}; B4a absent {rep['b4a_absent']}"
      + (f"; PROBLEMS {bad[:5]}" if bad else ""))
sys.exit(1 if bad else 0)
PY
}

b4a_summary() {   # PASS/FAIL and presence only: no accuracy of a sealed unit is printed or stored here
  # shellcheck disable=SC2086
  python - "$I/supplement.json" "$B4_REG" "$B4_TAG" "$S1_TAG" "$ATLAS_B4_P1" "$(cat "$S1D/head.txt")" "$SOFT_LOG" \
    $B4A_UNITS <<'PY'
import glob, json, os, re, subprocess, sys, time
out, regp, tag, s1tag, p1, s1head, softlog = sys.argv[1:8]
units = sys.argv[8:]
reg = {m["id"]: m for m in json.load(open(regp))["models"]}
wp = f"results/b4_weights/weights{tag}.json"
w = json.load(open(wp))["units"] if os.path.isfile(wp) else {}
head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
jf = os.path.join(os.path.dirname(out), "jobs_discovery.txt")
njobs = sum(1 for ln in open(jf) if ln.strip()) if os.path.isfile(jf) else 0
rep = {"schema": "b4a_supplement/1", "amendment": "B4a", "doc": "docs/plans/B4A_AMENDMENT.md", "p1": p1,
       "s1_head": s1head, "head": head, "tag": tag, "s1_tag": s1tag, "weights_record": wp,
       "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "units": {},
       "timing_condition": {"discovery_jobs_run_here": njobs,
                            "condition": "run alone on an otherwise idle pod (no GPU job, no other probe); every other "
                                         "family was timed in the loaded S1 lane",
                            "p2": "scripts/b4a_p2.js timing floors the ShuffleNet t_ref at the S1-lane REF units"}}
for u in units:
    lay = reg[u]["layouts"]["fit"]
    d = lay["dump"]
    st = ("complete" if os.path.isfile(os.path.join(d, "meta.json")) else
          "partial" if os.path.isdir(d) and os.listdir(d) else "absent")
    r = {"roles": reg[u]["roles"], "sealed": bool(lay.get("sealed")), "dump": d, "dump_state": st,
         "weights_status": (w.get(u) or {}).get("status"), "info": (w.get(u) or {}).get("info"),
         "record": f"results/b4_extract/{u}_fit{tag}.json"}
    r["record_written"] = os.path.isfile(r["record"])
    if "Dnew" in reg[u]["roles"]:                 # under any _r<k> tag: a relaunch skips the jobs an earlier run did
        r["discovery"] = {}
        for k, base, pat, fn in (("t2", "results/b4_t2", re.escape(u) + r"(_r\d+)+", "probe.json"),
                                 ("t1_fit", "results/b4_t1", re.escape(u) + r"_fit(_r\d+)+", "scoreboard.json")):
            found = sorted(p for p in glob.glob(f"{base}/*/{fn}")
                           if re.fullmatch(pat, os.path.basename(os.path.dirname(p))))
            r["discovery"][k] = {"present": bool(found), "paths": found}
    rep["units"][u] = r
rep["soft_failures"] = open(softlog).read().splitlines() if os.path.isfile(softlog) else []
if os.path.exists(out):
    sys.exit(f"{out} exists: never overwritten")
with open(out, "w") as f:
    json.dump(rep, f, indent=1)
for u, r in rep["units"].items():
    disc = ""
    if "discovery" in r:
        disc = "; discovery " + ", ".join(k + " " + (" ".join(v["paths"]) or "MISSING") for k, v in r["discovery"].items())
    print(f"[b4a] {u:20s} weights {r['weights_status']} {r['info'] or ''}; dump {r['dump_state']}" + disc)
print(f"[b4a] {len(rep['soft_failures'])} soft failures -> {out}")
PY
}

# =====================================================================================================================
b4a_guards
echo "=== B4a: check dir, versions (hard), known-answer tests (hard), S1 seals intact (hard) ==="
mkdir -p "$I" results/b4_extract results/b4_weights
git rev-parse HEAD > "$I/head.txt"
b4a_versions
timeout "$B4_T_TEST" python -m pytest -q tests/test_b4a_amendment.py --junitxml "$I/pytest_b4a.xml"
python scripts/b4_extract.py --verify-seals --registry "$B4_REG" --manifest "$S1D/sealed.json" \
  --out "$I/verify_seals_s1.json"

echo "=== B4a: the weights gate with the one-sided README rule, before any extraction (record weights$B4_TAG.json) ==="
timeout "$B4_T_WEIGHTS" python scripts/b4_weights_amend.py --registry "$B4_REG" --volume "$VOLUME" \
  --s1-record "$S1W" --units "${B4A_UNITS// /,}" --out "results/b4_weights/weights$B4_TAG.json"

echo "=== B4a GPU: quota, then the three fit extractions (C sealed; the extractor reads weights$B4_TAG.json) ==="
b4_quota "$(b4a_gb)"
for id in $B4A_UNITS; do
  if [[ " $B4A_SEALED " == *" $id "* ]]; then soft b4_extract "$id" fit --sealed; else soft b4_extract "$id" fit; fi
done

echo "=== B4a CPU: the S1 discovery probes of $B4A_DISC (tag $B4_TAG; b4_cpu_lane settings) ==="
if [[ -f "$(b4_reg get "$B4A_DISC" layouts.fit.dump)/meta.json" ]]; then
  soft b4a_lane "$I" "$B4A_DISC"
else
  echo "[soft] FAILED: no complete $B4A_DISC dump: its discovery probes are skipped" | tee -a "$SOFT_LOG"
fi

echo "=== B4a: the complete seal manifest (every sealed dump: S1's and the new ones), diffed against S1's (hard) ==="
python scripts/b4_extract.py --seal-manifest --registry "$B4_REG" --out "$I/sealed.json"
b4a_seal_diff
python scripts/b4_extract.py --verify-seals --registry "$B4_REG" --manifest "$I/sealed.json" --out "$I/verify_seals.json"

echo "=== B4a: timing of every probe output (S1's and the supplement's; b4a_p2.js timing reads it at P2) ==="
soft b4_reg timing --out "$I/timing.json"
soft b4a_summary
B4A_DONE=1
echo "[b4a] done $(date). volume usage: $(du -sh "$VOLUME" 2>/dev/null | cut -f1)"
echo "     Pull (dumps excluded; S1 and the supplement together), copy $VOLUME/models/*.pt (D18), terminate the pod,"
echo "     commit R1 on Windows (results/b4/RUN_REQUEST_S1_SUPPLEMENT.md). At P2: node scripts/b4a_p2.js timing --timing"
echo "     $I/timing.json --out experiments/b4/timing_b4a.json, then b4_timeouts.js --timing experiments/b4/timing_b4a.json;"
echo "     S2 with ATLAS_B4S1_CHECK_DIR=$I (the complete sealed.json and versions.json); --p-run adds P_A;"
echo "     step E: node scripts/b4a_p2.js seals --json results/b4/b4a_seals.json (must PASS)."
if [[ -s "$SOFT_LOG" ]]; then echo "soft failures ($SOFT_LOG):"; cat "$SOFT_LOG"; exit 1; fi
