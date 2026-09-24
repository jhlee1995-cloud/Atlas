# shellcheck shell=bash
# scripts/pod_b4.sh -- batch 4 pod blocks (docs/plans/B4_INTEGRATION.md D7-D11, D17). SOURCED by pod_atlas.sh after its
# prelude (network-volume and repo guards, caches on the volume, pip install, data check, GPU preflight unless --b4cpu,
# smoke tests); uses VOLUME, SOFT_LOG, STAMP, soft, cpu_quota and has_mode from there. Defines block_b4s1, block_b4s2.
#   session 1 (RTX 4090):    ATLAS_B4_P1=<P1> bash /workspace/Atlas/pod_atlas.sh /workspace --b4s1
#   session 2 (numpy only):  ATLAS_B4_P1=<P1> ATLAS_B4_P2=<P2> bash /workspace/Atlas/pod_atlas.sh /workspace --b4s2 [--b4cpu]
# Append-only: every extractor / probe refuses an existing non-empty output; a complete dump is skipped, a partial one
# refused (ATLAS_B4_MOVE_PARTIAL=1 moves it aside first). Relaunch: ATLAS_B4S1_CHECK_DIR=results/instrument_check_b4s1_r2
# ATLAS_B4_TAG=_r2 (same code; a process limit is not an instrument change: results/margin_b1_vitb16/RELAUNCH_r2.md).
# The S2 replay compares the S1 anchor outputs (tag ATLAS_B4_S1_TAG, default '') with <ATLAS_B4_TAG>_s2replay, so an S2
# relaunch under _r2 still finds the S1 outputs; set ATLAS_B4_S1_TAG=_r2 in S2 only if S1 itself ran under _r2.
# Cuts: ATLAS_B4_CUT="<roles or ids>", only from the registered list (R2 Kwd F20 Sdisc Sconf STconf; scripts/b4_reg.py).
# TIME LIMITS live here and only here (D8: S1 may run at P1 plus changes to these limits); S2 reads
# experiments/b4/timeouts.json (P2) through scripts/b4_reg.py and falls back to B4_LIMITS.

B4_REG=experiments/b4/models.json
B4_BASE=2b561a9                    # design HEAD: the committed CIFAR/ImageNet instruments below may not change in batch 4
B4_COMMITTED="atlas/extract_acts.py atlas/extract_imagenet.py atlas/run.py atlas/run_imagenet.py atlas/build.py
 atlas/config.py atlas/context.py atlas/registry.py atlas/compare.py atlas/critic.py atlas/ladder.py atlas/report.py
 atlas/synth.py atlas/factors atlas/invariants extract scripts/train_second_seed.py scripts/anomaly_probe.py
 scripts/check_rebuild.py scripts/check_data.py experiments/tolerances_default.yaml experiments/queue requirements.txt
 tests/test_atlas_smoke.py tests/test_anomaly_probe.py tests/test_b1_imagenet.py"
B4_FROZEN=experiments/b4/frozen_files.txt        # P1: every batch-4 instrument, evaluator and test file, one path per line
B4_EXTRACT_CODE="scripts/b4_extract.py scripts/b4_weights.py scripts/b4_train_knob.py atlas/faults.py experiments/b4/models.json"
B4_PROBE_CODE="atlas/b4_core.py atlas/b4_collapse.py scripts/t1_scoreboard.py scripts/t1_streams.py scripts/collapse_probe.py
 scripts/t3s_spatial_probe.py scripts/b4_reg.py"   # a P1 -> P2 change here re-probes every discovery unit (D7 failure path)
B4_P2_FILES="experiments/b4/freeze_P2.json experiments/b4/t1_rules.json experiments/b4/laws_frozen.json
 experiments/b4/timeouts.json results/b4/eval_discovery.json"
B4_TESTS="tests/test_b4_core.py tests/test_b4_collapse.py tests/test_faults.py tests/test_b4_extract.py tests/test_b4_reg.py
 tests/test_t1_scoreboard.py tests/test_t1_streams.py tests/test_collapse_probe.py tests/test_t3s_spatial.py"
B4_SELFTESTS="b4_extract t1_scoreboard t1_streams collapse_probe t3s_spatial_probe"
# D10 limits (seconds): 3x the upper design estimate, never below 1800
B4_LIMITS="t1=5400 t1_wide=7200 t2=1800 t2_wide=3600 t1s=3600 t3s=3600"
B4_T_EXTRACT=1800; B4_T_TRAIN=5400; B4_T_TEST=1800; B4_T_WEIGHTS=2400; B4_T_REBUILD=1800
B4_S1_DIR="${ATLAS_B4S1_CHECK_DIR:-results/instrument_check_b4s1}"
B4_TAG="${ATLAS_B4_TAG:-}"
B4_REPROBE=0
export B4_REG B4_TAG VOLUME SOFT_LOG

b4_reg() { python scripts/b4_reg.py --registry "$B4_REG" --cut "${ATLAS_B4_CUT:-}" --limits "$B4_LIMITS" "$@"; }
b4_ids() { b4_reg ids "$@"; }                    # b4_ids <role>...: ids with any of the roles (registry order, cuts applied)
b4_has() { b4_ids "$1" | tr ' ' '\n' | grep -qxF "$2"; }

b4_mem_gb() {    # memory the cgroup still allows (GB); /proc/meminfo shows the host
  local lim="" use=0
  if [[ -r /sys/fs/cgroup/memory.max ]]; then lim=$(cat /sys/fs/cgroup/memory.max); use=$(cat /sys/fs/cgroup/memory.current)
  elif [[ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]]; then
    lim=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes); use=$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes)
  fi
  if [[ -z "$lim" || "$lim" == max ]] || (( lim > 1125899906842624 )); then
    awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo
  else
    echo $(( (lim - use) / 1073741824 ))
  fi
}
b4_P() {         # b4_P <peak GB per job>: parallel CPU jobs = min(CPU quota - 1, free memory / peak), at least 1 (D9)
  local peak="${1:-4}" c m p
  c=$(( $(cpu_quota) - 1 )); if (( c < 1 )); then c=1; fi
  m=$(b4_mem_gb); if ! [[ "$m" =~ ^[0-9]+$ ]] || (( m == 0 )); then echo "$c"; return 0; fi   # memory unknown: CPU bound
  p=$(awk -v m="$m" -v k="$peak" 'BEGIN{p=int(m/k); if(p<1)p=1; print p}')
  echo $(( c < p ? c : p ))
}
b4_quota() {     # b4_quota <GB this group still writes>: du against the 50 GB quota (df shows the cluster); B1's 48 GB ceiling
  local need="$1" used
  used=$(du -sb "$VOLUME" 2>/dev/null | cut -f1 || true)   # du exits non-zero if a file vanishes mid-walk (pipefail)
  [[ "$used" =~ ^[0-9]+$ ]] || used=0                      # empty or multi-line: never reach the arithmetic
  echo "[b4] volume holds $(( used / 1000000000 )) GB; this group writes <= ${need} GB"
  if (( used + need * 1000000000 > 48000000000 )); then
    echo "[b4] QUOTA: $(( used / 1000000000 )) + ${need} GB > 48 GB: group skipped" | tee -a "$SOFT_LOG"; return 1
  fi
}
b4_versions() {  # what A4b G0d recorded, plus the CPU model (session 2 may run on another host)
  python - "$1" <<'PY'
import json, platform, sys
import numpy, scipy, sklearn
v = {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__}
try:
    import torch, torchvision
    c = torch.cuda.is_available()
    v.update(torch=torch.__version__, torchvision=torchvision.__version__, device=torch.cuda.get_device_name(0) if c else "cpu",
             cudnn_allow_tf32=torch.backends.cudnn.allow_tf32, matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
except Exception as e:                                             # a CPU pod without torch still runs the numpy probes
    v["torch"] = f"unavailable: {e}"
cpu = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")]
v["cpu_model"], v["cpu_count_host"] = (cpu[0] if cpu else "?"), len(cpu)
json.dump(v, open(sys.argv[1], "w"), indent=1); print("[b4] versions", v)
PY
}

b4_guard_common() {   # <check dir>
  local m
  if [[ -e "$1" ]]; then echo "ERROR: $1 exists (committed record); relaunch with a new check dir and ATLAS_B4_TAG=_r2" >&2; return 1; fi
  if [[ "${ATLAS_REBUILD:-0}" == 1 ]]; then echo "ERROR: ATLAS_REBUILD=1 would rewrite committed results" >&2; return 1; fi
  for m in --stage1 --stage1b --stage2 --a3 --a4b --b1 --b1b --anomaly; do
    if has_mode "$m"; then echo "ERROR: batch 4 runs alone ($m given)" >&2; return 1; fi
  done
  # shellcheck disable=SC2086
  if ! git diff --quiet "$B4_BASE" HEAD -- $B4_COMMITTED; then
    echo "ERROR: a committed instrument changed since $B4_BASE (CIFAR bit-identity, D6):" >&2
    # shellcheck disable=SC2086
    git diff --stat "$B4_BASE" HEAD -- $B4_COMMITTED >&2; return 1
  fi
  if [[ -z "${ATLAS_B4_P1:-}" ]] || ! git merge-base --is-ancestor "$ATLAS_B4_P1" HEAD; then
    echo "ERROR: ATLAS_B4_P1 unset or not an ancestor of HEAD" >&2; return 1
  fi
  if [[ ! -f "$B4_FROZEN" ]]; then echo "ERROR: $B4_FROZEN missing" >&2; return 1; fi
  if command -v pgrep >/dev/null && pgrep -f 'train_second_seed.py|b4_train_knob.py|b4_extract.py' >/dev/null; then
    echo "ERROR: a GPU job is already running (batch 4 never shares the GPU)" >&2; return 1
  fi
  return 0
}
b4_guard_p2() {
  local f
  if [[ -z "${ATLAS_B4_P2:-}" ]] || ! git merge-base --is-ancestor "$ATLAS_B4_P1" "$ATLAS_B4_P2" \
     || ! git merge-base --is-ancestor "$ATLAS_B4_P2" HEAD; then
    echo "ERROR: need P1 <= P2 <= HEAD (ATLAS_B4_P1, ATLAS_B4_P2)" >&2; return 1
  fi
  for f in $B4_P2_FILES; do
    if ! git cat-file -e "$ATLAS_B4_P2:$f" 2>/dev/null; then echo "ERROR: P2 lacks $f" >&2; return 1; fi
  done
  # shellcheck disable=SC2046,SC2086
  if ! git diff --quiet "$ATLAS_B4_P2" HEAD -- $(cat "$B4_FROZEN") $B4_P2_FILES; then
    echo "ERROR: a frozen file changed after P2" >&2; return 1
  fi
  # shellcheck disable=SC2086
  if ! git diff --quiet "$ATLAS_B4_P1" "$ATLAS_B4_P2" -- $B4_EXTRACT_CODE; then
    echo "ERROR: extraction code or registry changed P1 -> P2: the sealed dumps are not from the frozen instrument." >&2
    echo "       Re-extract in a new GPU session under a new P1' (docs/plans/B4_INTEGRATION.md D7)." >&2; return 1
  fi
  # shellcheck disable=SC2086
  if ! git diff --quiet "$ATLAS_B4_P1" "$ATLAS_B4_P2" -- $B4_PROBE_CODE; then
    B4_REPROBE=1; echo "[b4] probe code changed P1 -> P2 (P2 amendment log): every discovery unit is re-probed first"
  fi                                          # evaluator-only fixes (D8 e) run on Windows and need no re-probe
  python scripts/b4_extract.py --verify-seals --registry "$B4_REG" --manifest "$B4_S1_DIR/sealed.json" \
    --out "$1/verify_seals.json"                                                          # hard
}

b4_extract() {   # b4_extract <id> <fit|eval|maps> [--sealed]: one unit and layout; the extractor skips a complete dump
  local id="$1" lay="$2"; shift 2
  local mv=(); if [[ "${ATLAS_B4_MOVE_PARTIAL:-0}" == 1 ]]; then mv=(--move-partial); fi
  timeout "$B4_T_EXTRACT" python scripts/b4_extract.py --registry "$B4_REG" --unit "$id" --layout "$lay" \
    --volume "$VOLUME" --weights-record "results/b4_weights/weights${B4_TAG}.json" \
    --record "results/b4_extract/${id}_${lay}${B4_TAG}.json" ${mv[@]+"${mv[@]}"} "$@"
}
b4_train_one() { # b4_train_one <id>: trains when the checkpoint is absent (append-only); frozen recipe script unless K-ls
  local id="$1" ck script args w
  ck=$(b4_reg get "$id" weights); script=$(b4_reg get "$id" train.script); args=$(b4_reg get "$id" train.args)
  if [[ -f "$ck" ]]; then echo "[skip] $ck exists"; return 0; fi
  w=$(( ($(cpu_quota) - 2) / 2 )); if (( w < 2 )); then w=2; fi; if (( w > 8 )); then w=8; fi
  mkdir -p "$(dirname "$ck")" "results/train_${id}"
  # shellcheck disable=SC2086
  timeout "$B4_T_TRAIN" python "$script" --volume "$VOLUME" $args --workers "$w" --out "$ck" \
    --train-json "results/train_${id}/train.json" > "$VOLUME/logs/train_${id}.log" 2>&1
}
b4_train_pair() { # b4_train_pair <id> <id>: two trainings of one architecture share the GPU (Stage 1 precedent); nothing else does
  local p=() id x
  for id in "$@"; do
    if b4_has K "$id" || b4_has F "$id" || b4_has F20 "$id"; then b4_train_one "$id" & p+=($!); sleep 30; fi
  done
  for x in ${p[@]+"${p[@]}"}; do wait "$x" || echo "[soft] FAILED: a training of $* (see $VOLUME/logs/train_*.log)" | tee -a "$SOFT_LOG"; done
  return 0
}
b4_cpu_lane() {  # b4_cpu_lane <check dir> <discovery|replay|reprobe|confirmation>: numpy jobs, niced, 1 BLAS thread, no CUDA
  local I="$1" ph="$2" jobs P
  jobs="$I/jobs_$ph.txt"
  mkdir -p "$VOLUME/logs/b4_$STAMP"
  b4_reg --logdir "$VOLUME/logs/b4_$STAMP" jobs --phase "$ph" --tag "$B4_TAG" > "$jobs"
  P=$(b4_P "$(b4_reg peak --phase "$ph")")
  echo "[b4] $ph lane: $(wc -l < "$jobs") jobs, $P in parallel (order: $jobs; logs: $VOLUME/logs/b4_$STAMP)"
  ATLAS_B4_CHECK_DIR="$I" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    xargs -d '\n' -P "$P" -n 1 nice -n 19 bash -c 'eval "$1" || echo "[soft] FAILED (exit $?): $1" >> "$SOFT_LOG"' _ < "$jobs"
}

block_b4s1() {
  local I="$B4_S1_DIR" bg=0 rb=0 id g
  b4_guard_common "$I"
  # shellcheck disable=SC2046
  if ! git diff --quiet "$ATLAS_B4_P1" HEAD -- $(cat "$B4_FROZEN"); then
    echo "ERROR: the batch-4 instrument differs from P1 (only scripts/pod_b4.sh limits may differ, D8)" >&2; return 1
  fi
  python -c "import torch; assert torch.cuda.is_available(), 'S1 is GPU-only (D17)'"
  mkdir -p "$I" results/b4_extract results/b4_weights /root/b4_frames
  git rev-parse HEAD > "$I/head.txt"; b4_versions "$I/versions.json"
  python scripts/b4_reg.py --registry "$B4_REG" validate
  echo "=== B4 S1: data (the four CIFAR-10-C extras are recorded; absent -> their clauses NOT_EVALUABLE) ==="
  python scripts/b4_extract.py --data-report --volume "$VOLUME" --out "$I/data.json"
  echo "=== B4 S1: known-answer tests and synthetic self-tests (hard; CLAUDE.md rule 4) ==="
  # shellcheck disable=SC2086
  timeout "$B4_T_TEST" python -m pytest -q $B4_TESTS --junitxml "$I/pytest_b4.xml"
  for g in $B4_SELFTESTS; do timeout "$B4_T_TEST" python "scripts/$g.py" --selftest --selftest-out "$I/selftest_$g.json"; done
  echo "=== B4 S1: committed CIFAR results still reproduce here (recorded, in the background; D6) ==="
  ( soft timeout "$B4_T_REBUILD" nice -n 19 python scripts/check_rebuild.py rebuild \
       --dump results/atlas_v1_resnet20_s0hub_st3/dump --committed results/atlas_v1_resnet20_s0hub_st3/atlas.json \
       --manifest experiments/queue/atlas_v1_resnet20_s0hub_st3.yaml --work "$VOLUME/scratch/b4_rebuild_$STAMP" \
       --out "$I/rebuild_resnet20_s0hub_st3" ) & rb=$!
  echo "=== B4 S1: weights (sha256, hub byte size, README accuracy gate, head export; before any extraction) ==="
  timeout "$B4_T_WEIGHTS" python scripts/b4_weights.py --registry "$B4_REG" --volume "$VOLUME" \
    --out "results/b4_weights/weights${B4_TAG}.json"

  echo "=== B4 S1 GPU-A: rule-6 anchors first (resnet20 hub, resnet56 hub), then the anchor gate (hard) ==="
  b4_quota "$(b4_reg gb --group disc)"
  for id in $(b4_ids ANCHOR); do b4_extract "$id" fit; done
  # shellcheck disable=SC2046
  python scripts/b4_extract.py --anchor-gate --registry "$B4_REG" --units $(b4_ids ANCHOR) --out "$I/anchor_gate.json"
  echo "=== B4 S1 GPU-B: discovery dumps (D, N, Dnew fit; lane S discovery maps) ==="
  for id in $(b4_ids D N Dnew); do if ! b4_has ANCHOR "$id"; then soft b4_extract "$id" fit; fi; done
  # shellcheck disable=SC2046
  soft python scripts/b4_extract.py --anchor-gate --registry "$B4_REG" --units $(b4_ids D N) --out "$I/anchor_gate_all.json"
  for id in $(b4_ids Sdisc); do soft b4_extract "$id" maps; done
  echo "=== B4 S1 CPU: discovery probes start in the background (rule-6 anchors first) ==="
  b4_cpu_lane "$I" discovery & bg=$!

  echo "=== B4 S1 GPU-C: trainings, two of one architecture at a time; never beside an extraction ==="
  b4_train_pair resnet56_s31 resnet56_s32
  b4_train_pair resnet20_s31 resnet20_s32
  b4_train_pair resnet56_s31_ls10 resnet56_s32_ls10
  b4_train_pair resnet56_s31_wd5e5 resnet56_s32_wd5e5
  soft python scripts/b4_weights.py --registry "$B4_REG" --volume "$VOLUME" --trained \
    --out "results/b4_weights/weights_trained${B4_TAG}.json"
  echo "=== B4 S1 GPU-D: SEALED confirmation dumps in priority order (nothing reads them before P2) ==="
  if b4_quota "$(b4_reg gb --group C)"; then for id in $(b4_ids C); do soft b4_extract "$id" fit --sealed; done; fi
  if b4_quota "$(b4_reg gb --group F)"; then
    for id in $(b4_ids F F20); do soft b4_extract "$id" fit --sealed; soft b4_extract "$id" eval --sealed; done
  fi
  if b4_quota "$(b4_reg gb --group K)"; then for id in $(b4_ids K); do soft b4_extract "$id" fit --sealed; done; fi
  if b4_quota "$(b4_reg gb --group R2)"; then for id in $(b4_ids R2); do soft b4_extract "$id" eval --sealed; done; fi
  if b4_quota "$(b4_reg gb --group Sconf)"; then for id in $(b4_ids Sconf); do soft b4_extract "$id" maps --sealed; done; fi
  python scripts/b4_extract.py --seal-manifest --registry "$B4_REG" --out "$I/sealed.json"
  echo "[b4] GPU chain done $(date); waiting for the discovery lane and the rebuild check"
  wait "$bg" || echo "[soft] FAILED: discovery CPU lane" | tee -a "$SOFT_LOG"
  wait "$rb" || true
  b4_reg timing --out "$I/timing.json"
  rm -rf /root/b4_frames
  echo "[b4] volume usage: $(du -sh "$VOLUME" 2>/dev/null | cut -f1). Pull (dumps excluded) AND copy the checkpoints"
  echo "     ($VOLUME/models/*.pt, ~75 MB; sha256 in results/b4_weights/weights*.json) to a Windows folder outside the repo"
  echo "     (D18); terminate the pod; commit R1 on Windows; then:"
  echo "      node scripts/t1_eval.js --phase discovery --p1 <P1> --json results/b4/eval_discovery.json --emit-rules experiments/b4/t1_rules.json"
  echo "      node scripts/collapse_laws.js --fit --p1 <P1> --out experiments/b4/laws_frozen.json"
  echo "      node scripts/b4_timeouts.js --registry $B4_REG --timing $I/timing.json --out experiments/b4/timeouts.json"
}

block_b4s2() {
  local I="${ATLAS_B4S2_CHECK_DIR:-results/instrument_check_b4s2}" g
  b4_guard_common "$I"
  mkdir -p "$I" /root/b4_frames
  b4_guard_p2 "$I"
  git rev-parse HEAD > "$I/head.txt"; b4_versions "$I/versions.json"
  soft b4_reg versions-match --a "$B4_S1_DIR/versions.json" --b "$I/versions.json" --out "$I/versions_match.json"
  echo "=== B4 S2: known-answer tests and self-tests (hard) ==="
  # shellcheck disable=SC2086
  timeout "$B4_T_TEST" python -m pytest -q $B4_TESTS --junitxml "$I/pytest_b4.xml"
  for g in $B4_SELFTESTS; do timeout "$B4_T_TEST" python "scripts/$g.py" --selftest --selftest-out "$I/selftest_$g.json"; done
  echo "=== B4 S2: rule-6 replay of the anchors at P2 (discovery settings; tolerance compare, read by the evaluators) ==="
  b4_cpu_lane "$I" replay
  soft b4_reg replay-compare --out "$I/replay.json" --s1-tag "${ATLAS_B4_S1_TAG:-}" --tag "$B4_TAG"
  if (( B4_REPROBE )); then b4_cpu_lane "$I" reprobe; fi
  echo "=== B4 S2: confirmation: sealed dumps opened once, P2 rules, P2 time limits ==="
  ATLAS_B4_UNSEAL="$ATLAS_B4_P2" b4_cpu_lane "$I" confirmation
  b4_reg timing --out "$I/timing.json"
  rm -rf /root/b4_frames
  echo "[b4] pull, commit R2 on Windows, then the confirmation evaluators (docs/plans/B4_INTEGRATION.md D8 step E)."
}
