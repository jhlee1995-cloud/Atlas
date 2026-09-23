#!/usr/bin/env bash
# RunPod: datasets (idempotent) + Stage 0 (+ optional stage blocks, docs/plans/) of the atlas.
#
# The repo MUST be a git clone ON the network volume (e.g. /workspace/Atlas): results/ is written
# relative to the repo, and only the volume survives a pod stop. Create the pod with
#   --network-volume-id kxfir1tryb --data-center-ids EU-RO-1   (mounted at /workspace)
#
#   bash /workspace/Atlas/pod_atlas.sh /workspace              # data + Stage 0
#   bash /workspace/Atlas/pod_atlas.sh /workspace --stage1     # + norm check, train s1/s2, v1 atlases, compare, critic
#                                                              #   (ran 2026-09-23 and is committed: refused now)
#   bash /workspace/Atlas/pod_atlas.sh /workspace --stage1b    # + instrument checks, train s3/s4, atlases, compare,
#                                                              #   critic (docs/plans/STAGE1.md amendment 2)
#   bash /workspace/Atlas/pod_atlas.sh /workspace --stage2     # + resnet56: norm check, accuracy ladder, null, rule-6
#                                                              #   resnet20 re-runs, atlases, compare, critic (STAGE2.md)
#   bash /workspace/Atlas/pod_atlas.sh /workspace --a3         # + margin_typeb Stage-B rebuilds, M1 gate, critic
#                                                              #   (docs/plans/A3_MARGIN.md)
#   bash /workspace/Atlas/pod_atlas.sh /workspace --data-only  # datasets only (no GPU needed)
# Flags combine (e.g. --stage1b --stage2 --a3); blocks always run in this order: stage1, stage1b, stage2, a3.
# stage1b / stage2 / a3 each run isolated (run_block): a failing block is logged and later blocks still run;
# steps wrapped in soft() log a failure and let their block go on. The script exits 1 at the end if any block
# or soft step failed (soft failures are listed in <volume>/logs/soft_failures_<timestamp>.log).
#
# Run it detached so an SSH drop does not kill it; the full log goes to <volume>/logs/:
#   mkdir -p /workspace/logs && setsid bash /workspace/Atlas/pod_atlas.sh /workspace < /dev/null > /workspace/logs/launch.log 2>&1 &
#   tail -F /workspace/logs/launch.log
# A stage whose results/<exp>/provenance.json (written last by atlas.run) exists is skipped
# (ATLAS_REBUILD=1 forces it).
set -euo pipefail
VOLUME="$(realpath -m "${1:?usage: bash pod_atlas.sh <volume> [--data-only | --stage1 | --stage1b | --stage2 | --a3 ...]}")"
shift
for m in "$@"; do                                              # a typo must not silently skip a block
  case "$m" in --data-only|--stage1|--stage1b|--stage2|--a3) ;; *) echo "ERROR: unknown flag '$m'" >&2; exit 1;; esac
done
MODES=" $* "
has_mode() { [[ "$MODES" == *" $1 "* ]]; }
cd "$(dirname "$(readlink -f "$0")")"                        # repo root, whatever the caller's cwd
if has_mode --stage1 && [[ -f results/critic_v1_resnet20_s1_s2/critic.json && "${ATLAS_ALLOW_STAGE1_RERUN:-0}" != 1 ]]; then
  echo "ERROR: --stage1 already ran and is committed; its compare/critic steps would overwrite committed results." >&2
  echo "       Use --stage1b / --stage2 / --a3 (ATLAS_ALLOW_STAGE1_RERUN=1 overrides)." >&2; exit 1
fi

# --- guards: the volume is a network volume, and the repo lives on it ---
if [[ "${ATLAS_ALLOW_LOCAL:-0}" != 1 ]]; then
  SRC="$(findmnt -no SOURCE --target "$VOLUME" 2>/dev/null || true)"
  [[ "$SRC" == mfs#* ]] || {                                  # network volumes are MooseFS FUSE mounts
    echo "ERROR: $VOLUME is not a network volume (mount source: '${SRC:-none}')." >&2
    echo "       Create the pod with --network-volume-id kxfir1tryb --data-center-ids EU-RO-1." >&2; exit 1; }
fi
case "$(pwd -P)/" in
  "$(cd "$VOLUME" && pwd -P)"/*) ;;
  *) echo "ERROR: repo is at $(pwd -P), outside $VOLUME; results/ would be lost on pod stop. Clone to $VOLUME/Atlas." >&2
     exit 1;;
esac
# ignore results/ while untracked on the pod: `git pull` overwrites ignored files. Once Windows commits a
# result it is tracked, and if this script rewrites it (ATLAS_REBUILD=1, a --stage1 re-run of compare/critic)
# a plain `git pull` aborts. Update the pod only after its results are committed on Windows:
#   git -C /workspace/Atlas fetch && git -C /workspace/Atlas reset --hard origin/main   (dumps are untracked; kept)
grep -qxF '/results/' .git/info/exclude 2>/dev/null || echo '/results/' >> .git/info/exclude

# --- env: PEP 668 + caches on the volume (pod --env reaches PID 1 only, not SSH shells) ---
export PIP_BREAK_SYSTEM_PACKAGES=1 PYTHONUNBUFFERED=1
export PIP_CACHE_DIR="$VOLUME/.cache/pip" TORCH_HOME="$VOLUME/.cache/torch"
export HF_HOME="$VOLUME/.cache/huggingface" XDG_CACHE_HOME="$VOLUME/.cache"

# --- log on the volume ---
mkdir -p "$VOLUME/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$VOLUME/logs/pod_atlas_$STAMP.log"
SOFT_LOG="$VOLUME/logs/soft_failures_$STAMP.log"             # one line per failed soft() step
exec > >(tee -a "$LOG") 2>&1
echo "=== log $LOG | git $(git rev-parse --short HEAD 2>/dev/null || echo NO_GIT) | $(date) | flags:${MODES% } ==="

echo "=== install ==="
python -m pip install -q -r requirements.txt
python -c "import extract.backbone, extract.data_loaders, torchvision, yaml, sklearn, scipy, matplotlib"

echo "=== data (idempotent; $VOLUME/datasets) ==="
if ! python scripts/check_data.py "$VOLUME"; then
  python -m extract.populate_data --volume "$VOLUME"          # cifar10, cifar10_train, cifar100, svhn
  C10C="$VOLUME/datasets/cifar10c"
  if [[ -d "$C10C/CIFAR-10-C" ]]; then                        # nested layout from an earlier manual extract
    find "$C10C/CIFAR-10-C" -maxdepth 1 -name '*.npy' -exec mv -f -t "$C10C/" {} +
    rm -rf "$C10C/CIFAR-10-C"
  fi
  if ! python scripts/check_data.py "$VOLUME" cifar10c; then
    TAR=/root/CIFAR-10-C.tar                                   # container disk: the 2.9 GB tar never hits the volume
    rm -f "$TAR"
    curl -fL --retry 5 --retry-all-errors -o "$TAR" "https://zenodo.org/api/records/2535967/files/CIFAR-10-C.tar/content"
    echo "56bf5dcef84df0e2308c6dcbcbbd8499  $TAR" | md5sum -c -
    rm -rf "$C10C.tmp" && mkdir -p "$C10C.tmp"
    tar -xf "$TAR" -C "$C10C.tmp" --strip-components=1         # tar root is CIFAR-10-C/; the loader needs it flat
    rm -rf "$C10C" && mv "$C10C.tmp" "$C10C" && rm -f "$TAR"
  fi
  python -m extract.populate_data --volume "$VOLUME"          # re-scan so manifest.json records cifar10c
  python scripts/check_data.py "$VOLUME"                      # hard gate
fi
echo "volume usage: $(du -sh "$VOLUME" 2>/dev/null | cut -f1) of the 50 GB quota (df shows the whole cluster, not the quota)"
has_mode --data-only && { echo "=== data only: done ==="; exit 0; }

echo "=== GPU preflight ==="
python - <<'EOF'
import torch, torch.nn.functional as F
assert torch.cuda.is_available(), f"torch {torch.__version__}: CUDA unavailable (image/driver mismatch); refusing to run on CPU"
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability())
F.conv2d(torch.randn(1, 3, 8, 8, device="cuda"), torch.randn(4, 3, 3, 3, device="cuda")); torch.cuda.synchronize()
EOF

echo "=== tests (CPU) ==="
python -m pytest -q tests/test_atlas_smoke.py

# run_stage <exp_id> [atlas.run args...]: skip when the atlas is already built (keeps committed results stable)
run_stage() {
  local exp="$1"; shift
  if [[ -f "results/$exp/provenance.json" && "${ATLAS_REBUILD:-0}" != 1 ]]; then
    echo "[skip] results/$exp/provenance.json exists (ATLAS_REBUILD=1 to rebuild)"
  else
    python -m atlas.run --manifest "experiments/queue/$exp.yaml" --volume "$VOLUME" "$@"
  fi
}

echo "=== Stage 0: atlas v0 (resnet20 hub weights) ==="
run_stage atlas_v0_resnet20_cifar10

if has_mode --stage1; then                                     # docs/plans/STAGE1.md
  M="$VOLUME/models"; mkdir -p "$M" results/norm_check_resnet20
  echo "=== Stage 1: hub accuracy under both normalizations (evidence, not a gate) ==="
  python scripts/check_norm.py --volume "$VOLUME" --out results/norm_check_resnet20/norm_check.json

  echo "=== Stage 1: training preflight (1 epoch, container disk) ==="
  python scripts/train_second_seed.py --volume "$VOLUME" --seed 0 --norm chenyaofo --epochs 1 --out /root/preflight.pt
  rm -f /root/preflight.pt

  echo "=== Stage 1: train s1, s2 concurrently (hub recipe; logs in $VOLUME/logs/train_resnet20_s*.log) ==="
  CPUS=$(nproc)                                                # nproc can report host cores; prefer the cgroup quota
  if [[ -r /sys/fs/cgroup/cpu.max ]]; then                     # cgroup v2: "<quota> <period>" or "max <period>"
    read -r q p < /sys/fs/cgroup/cpu.max
    if [[ "$q" != max ]]; then CPUS=$(( q / p )); fi
  elif [[ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]]; then      # cgroup v1 (RunPod 4090 hosts): -1 = unlimited
    q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us); p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
    if (( q > 0 )); then CPUS=$(( q / p )); fi
  fi
  W=$(( (CPUS - 2) / 2 )); if (( W < 2 )); then W=2; fi; if (( W > 8 )); then W=8; fi
  echo "[stage1] $CPUS CPUs -> $W data workers per training process"
  pids=()
  for s in 1 2; do
    ck="$M/resnet20_s${s}_chenyaofo.pt"
    if [[ ! -f "$ck" ]]; then
      python scripts/train_second_seed.py --volume "$VOLUME" --seed "$s" --norm chenyaofo --workers "$W" \
             --out "$ck" --train-json "results/train_resnet20_s${s}/train.json" \
             > "$VOLUME/logs/train_resnet20_s${s}.log" 2>&1 &
      pids+=($!); sleep 30                                     # stagger torch.hub cache reads
    else
      echo "[skip] $ck exists"
    fi
  done
  for p in "${pids[@]}"; do wait "$p"; done                   # set -e: a failed training stops here
  grep -h "saved" "$VOLUME"/logs/train_resnet20_s[12].log || true
  ck="$M/resnet20_rand99_chenyaofo.pt"                         # null: random init + BN recal (~1 min)
  [[ -f "$ck" ]] || python scripts/train_second_seed.py --volume "$VOLUME" --seed 99 --norm chenyaofo --epochs 0 \
                      --out "$ck" --train-json results/train_resnet20_rand99/train.json

  echo "=== Stage 1: atlases ==="
  for e in s0hub s1 s2 s1_ref1 rand; do run_stage "atlas_v1_resnet20_$e"; done

  echo "=== Stage 1: compare (needs the dumps, so it runs here) ==="
  R=results/atlas_v1_resnet20
  python -m atlas.compare --a ${R}_s1    --b ${R}_s2
  python -m atlas.compare --a ${R}_s0hub --b ${R}_s1
  python -m atlas.compare --a ${R}_s0hub --b ${R}_s2
  python -m atlas.compare --a ${R}_s1    --b ${R}_s1_ref1
  python -m atlas.compare --a ${R}_s1    --b ${R}_rand
  python -m atlas.compare --a results/atlas_v0_resnet20_cifar10 --b ${R}_s0hub --same-space

  echo "=== Stage 1: critic ==="
  T=experiments/tolerances_default.yaml
  python -m atlas.critic --results ${R}_s1 ${R}_s2 --tol $T --out results/critic_v1_resnet20_s1_s2
  python -m atlas.critic --results ${R}_s0hub ${R}_s1 ${R}_s2 --tol $T --out results/critic_v1_resnet20_s0_s1_s2
  python -m atlas.critic --results ${R}_s1 ${R}_s1_ref1 --tol $T --out results/critic_v1_resnet20_s1_noise
  python -m atlas.critic --results ${R}_s1 ${R}_rand --tol $T --out results/critic_v1_resnet20_s1_null
fi

# ======================================================================================================
# Blocks for --stage1b, --stage2, --a3. Every output directory below is new (never a committed Stage 0/1
# result dir): compare always writes into its --b run, critics and checks into their own --out.
# Relaunch (append-only): compares and critics whose output already exists are skipped (once), atlases are
# skipped by run_stage, and the Stage 1b instrument checks refuse an existing check dir (ATLAS_STAGE1B_CHECK_DIR).
# ======================================================================================================

# soft <cmd...>: run a step; a failure is logged (and fails the session at the end) but its block goes on
soft() {
  "$@" || { echo "[soft] FAILED (exit $?): $*" | tee -a "$SOFT_LOG"; }
}

# once <target> <cmd...>: skip <cmd> when <target> exists, so a relaunch after the pull was committed never
# rewrites a committed compare/critic output (compare and critic are pure functions of existing atlases)
once() { local t="$1"; shift; if [[ -e "$t" ]]; then echo "[skip] $t exists (committed; re-run by hand with --out <new dir>)"; return 0; fi; "$@"; }
# compare_once <a> <b> [compare args...]: once <b>/compare_vs_<basename a>/deformation.json python -m atlas.compare
compare_once() {
  local a="$1" b="$2"; shift 2
  once "$b/compare_vs_$(basename "$a")/deformation.json" python -m atlas.compare --a "$a" --b "$b" "$@"
}
# critic_once <out> [critic args...]: once <out>/critic.json python -m atlas.critic ... --out <out>
critic_once() {
  local out="$1"; shift
  once "$out/critic.json" python -m atlas.critic "$@" --out "$out"
}

# cpu_quota: usable CPUs (nproc can report host cores; prefer the cgroup quota), as in the Stage 1 block
cpu_quota() {
  local c q p
  c=$(nproc)
  if [[ -r /sys/fs/cgroup/cpu.max ]]; then                     # cgroup v2: "<quota> <period>" or "max <period>"
    read -r q p < /sys/fs/cgroup/cpu.max
    if [[ "$q" != max ]]; then c=$(( q / p )); fi
  elif [[ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]]; then      # cgroup v1 (RunPod 4090 hosts): -1 = unlimited
    q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us); p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
    if (( q > 0 )); then c=$(( q / p )); fi
  fi
  echo "$c"
}

# log_tail <what> <log>: a failed training's log tail goes into the launch log
log_tail() {
  echo "=== FAILED: $1; last 30 lines of $2 ==="
  tail -n 30 "$2" 2>/dev/null || true
}

# train_seeds <seed>...: resnet20, hub recipe (train_second_seed.py defaults), all seeds concurrently on the one
# GPU. A seed is skipped only when its checkpoint AND its train.json exist. Every training is waited for before
# this returns (no training outlives its block); any failure prints its log tail and returns 1.
train_seeds() {
  local cpus w s ck tj i=0 rc=0 pids=() seeds=()
  cpus=$(cpu_quota)
  w=$(( (cpus - 2) / $# )); if (( w < 2 )); then w=2; fi; if (( w > 8 )); then w=8; fi
  echo "[train] $cpus CPUs -> $w data workers per training process"
  mkdir -p "$VOLUME/models"
  for s in "$@"; do
    ck="$VOLUME/models/resnet20_s${s}_chenyaofo.pt"; tj="results/train_resnet20_s${s}/train.json"
    if [[ -f "$ck" && -f "$tj" ]]; then echo "[skip] $ck and $tj exist"; continue; fi
    python scripts/train_second_seed.py --volume "$VOLUME" --seed "$s" --norm chenyaofo --workers "$w" \
           --out "$ck" --train-json "$tj" > "$VOLUME/logs/train_resnet20_s${s}.log" 2>&1 &
    pids+=($!); seeds+=("$s"); sleep 30                        # stagger torch.hub cache reads
  done
  while (( i < ${#pids[@]} )); do
    if ! wait "${pids[i]}"; then
      rc=1; log_tail "training resnet20 s${seeds[i]}" "$VOLUME/logs/train_resnet20_s${seeds[i]}.log"
    fi
    i=$(( i + 1 ))
  done
  for s in "$@"; do grep -h "saved" "$VOLUME/logs/train_resnet20_s${s}.log" 2>/dev/null || true; done
  return "$rc"
}

# train_one <name> <ckpt> <train.json> <train_second_seed.py args...>: one training in the foreground (log
# <volume>/logs/train_<name>.log), skipped when <ckpt> AND <train.json> exist. A failure prints the log tail and
# is recorded as a soft failure; it returns 0 so the rest of its block still runs.
train_one() {
  local name="$1" ck="$2" tj="$3"; shift 3
  local log="$VOLUME/logs/train_${name}.log"
  if [[ -f "$ck" && -f "$tj" ]]; then echo "[skip] $ck and $tj exist"; return 0; fi
  if python scripts/train_second_seed.py --volume "$VOLUME" --norm chenyaofo --out "$ck" --train-json "$tj" "$@" \
       > "$log" 2>&1; then
    grep -h "saved" "$log" || true
  else
    log_tail "training $name" "$log"
    echo "[soft] FAILED: training $name ($log)" >> "$SOFT_LOG"
  fi
}

# --- Stage 1b: seeds 3, 4 (docs/plans/STAGE1.md amendments 1-2; predictions atlas_v1_resnet20_s3.yaml) -------
block_stage1b() {
  local R=results/atlas_v1_resnet20 T=experiments/tolerances_default.yaml
  local I="${ATLAS_STAGE1B_CHECK_DIR:-results/instrument_check_stage1b}"    # I1/I2 records; never overwritten
  local W="$VOLUME/scratch/stage1b_replay" e a b
  # Stage A + training inputs: I1 cannot see them (it reuses the s1 dump and its stored pixel factors)
  local SA="atlas/extract_acts.py atlas/factors atlas/config.py atlas/context.py extract scripts/train_second_seed.py requirements.txt"
  for e in s0hub s1 s2; do                                     # Stage 1 dumps (volume) + committed atlases (git)
    [[ -f ${R}_$e/dump/meta.json && -f ${R}_$e/atlas.json ]] || { echo "ERROR: ${R}_$e dump or atlas.json missing" >&2; return 1; }
  done
  # a relaunch must re-verify the instrument, never skip it: it writes to a fresh check dir
  [[ ! -e "$I" ]] || { echo "ERROR: $I exists (committed I1/I2 record); relaunch with ATLAS_STAGE1B_CHECK_DIR=${I}_r2" >&2; return 1; }
  echo "=== Stage 1b: I0 Stage A inputs unchanged since the Stage 1 run (6242a17) ==="
  if ! git diff --quiet 6242a17 HEAD -- $SA; then
    git diff --stat 6242a17 HEAD -- $SA || true
    [[ "${ATLAS_STAGE1B_ALLOW_STAGEA_DIFF:-0}" == 1 ]] || {
      echo "ERROR: Stage A / training code changed since 6242a17; review, record it in STAGE1.md amendment 2, then set ATLAS_STAGE1B_ALLOW_STAGEA_DIFF=1" >&2
      return 1; }
  fi
  echo "=== Stage 1b: I0 torch and GPU equal Stage 1 (results/train_resnet20_s1/train.json) ==="
  python - <<'EOF'
import json, torch
r = json.load(open("results/train_resnet20_s1/train.json"))
dev = torch.cuda.get_device_name(0)
print(f"[stage1b] torch {torch.__version__} on {dev}; Stage 1 trained with torch {r['torch']} on {r['device']}")
assert torch.__version__ == r["torch"], ("torch differs from Stage 1", torch.__version__, r["torch"])
assert dev == r["device"], ("GPU differs from Stage 1", dev, r["device"])
EOF
  echo "=== Stage 1b: I0 s3 and s4 manifests define the same instrument ==="
  python scripts/check_rebuild.py manifests experiments/queue/atlas_v1_resnet20_s3.yaml \
         experiments/queue/atlas_v1_resnet20_s4.yaml --seed-tags s3 s4

  echo "=== Stage 1b: I1 rebuild (Stage B from the s1 dump with the s3 manifest vs the committed s1 atlas) ==="
  python scripts/check_rebuild.py rebuild --dump ${R}_s1/dump --committed ${R}_s1/atlas.json \
         --manifest experiments/queue/atlas_v1_resnet20_s3.yaml --work "$VOLUME/scratch/stage1b_rebuild_s1" --out $I
  echo "=== Stage 1b: I1 replay (compare + critic s1-s2 with this commit's code vs the committed Stage 1 files) ==="
  python -m atlas.compare --a ${R}_s1 --b ${R}_s2 --out "$W/compare_s1_s2"
  python -m atlas.critic --results ${R}_s1 ${R}_s2 --tol $T --out "$W/critic_s1_s2"
  python scripts/check_rebuild.py replay \
         --deformation ${R}_s2/compare_vs_atlas_v1_resnet20_s1/deformation.json \
         --deformation-new "$W/compare_s1_s2/deformation.json" \
         --critic results/critic_v1_resnet20_s1_s2/critic.json --critic-new "$W/critic_s1_s2/critic.json" --out $I

  echo "=== Stage 1b: training preflight (1 epoch, container disk) ==="
  python scripts/train_second_seed.py --volume "$VOLUME" --seed 0 --norm chenyaofo --epochs 1 --out /root/preflight.pt
  rm -f /root/preflight.pt
  echo "=== Stage 1b: train s3, s4 concurrently (hub recipe; logs in $VOLUME/logs/train_resnet20_s{3,4}.log) ==="
  train_seeds 3 4

  echo "=== Stage 1b: atlases ==="
  for e in s3 s4; do run_stage "atlas_v1_resnet20_$e"; done

  echo "=== Stage 1b: compare (B is always s3 or s4) ==="
  soft compare_once ${R}_s3    ${R}_s4
  soft compare_once ${R}_s0hub ${R}_s3
  soft compare_once ${R}_s0hub ${R}_s4
  for a in s1 s2; do for b in s3 s4; do                        # cross-generation pairs: four-way critic (INFO)
    soft compare_once ${R}_$a ${R}_$b
  done; done
  echo "=== Stage 1b: critic ==="
  soft critic_once results/critic_v1_resnet20_s3_s4 --results ${R}_s3 ${R}_s4 --tol $T
  soft critic_once results/critic_v1_resnet20_s0_s3_s4 --results ${R}_s0hub ${R}_s3 ${R}_s4 --tol $T
  soft critic_once results/critic_v1_resnet20_s1_s2_s3_s4 --results ${R}_s1 ${R}_s2 ${R}_s3 ${R}_s4 --tol $T
  echo "=== Stage 1b: I2 dump meta (s3, s4 vs s1; weights differ from s1, s2) ==="
  python scripts/check_rebuild.py dump-meta --ref ${R}_s1 --others ${R}_s2 --new ${R}_s3 ${R}_s4 --out $I
  echo "[stage1b] D1 is computed on Windows after the pull: node scripts/d1_distance_only.js"
}

# --- Stage 2 (A4): resnet20 -> resnet56 (docs/plans/STAGE2.md; predictions atlas_v1_resnet56_s0hub.yaml) -------
block_stage2() {
  local M="$VOLUME/models" R20=results/atlas_v1_resnet20 R56=results/atlas_v1_resnet56 T=experiments/tolerances_default.yaml
  local cpus w E e a
  mkdir -p "$M"
  echo "=== Stage 2: resnet56 hub accuracy under both normalizations (evidence, not a gate) ==="
  if [[ -f results/norm_check_resnet56/norm_check.json ]]; then
    echo "[skip] results/norm_check_resnet56/norm_check.json exists"
  else
    soft python scripts/check_norm.py --volume "$VOLUME" --arch cifar10_resnet56 --out results/norm_check_resnet56/norm_check.json
  fi

  echo "=== Stage 2: resnet56 accuracy ladder e40, e20, e10 (hub recipe, seed 11, full cosine; one at a time) + null ==="
  cpus=$(cpu_quota); w=$(( cpus - 2 )); if (( w < 2 )); then w=2; fi; if (( w > 8 )); then w=8; fi
  echo "[stage2] $cpus CPUs -> $w data workers (one training at a time)"
  for E in 40 20 10; do
    train_one "resnet56_e$E" "$M/resnet56_s11_e${E}_chenyaofo.pt" "results/train_resnet56_e${E}/train.json" \
              --arch cifar10_resnet56 --seed 11 --epochs "$E" --workers "$w"
  done
  train_one resnet56_rand99 "$M/resnet56_rand99_chenyaofo.pt" results/train_resnet56_rand99/train.json \
            --arch cifar10_resnet56 --seed 99 --epochs 0 --workers "$w"

  echo "=== Stage 2: atlases (resnet20 same-session re-measure first: rule 6) ==="
  for e in resnet20_s0hub_st2 resnet20_s1_st2 resnet20_s2_st2 resnet56_s0hub resnet56_e10 resnet56_e20 \
           resnet56_e40 resnet56_rand resnet56_s0hub_b1; do
    soft run_stage "atlas_v1_$e"
  done

  echo "=== Stage 2: compare (needs the dumps; B is always a Stage 2 run) ==="
  soft compare_once ${R20}_s0hub ${R20}_s0hub_st2 --same-space                            # R56-0c (recorded)
  for a in s0hub s1 s2; do soft compare_once ${R20}_${a}_st2 ${R56}_s0hub; done
  soft compare_once ${R20}_s1_st2    ${R20}_s2_st2                                        # band B20 pairs
  soft compare_once ${R20}_s0hub_st2 ${R20}_s1_st2
  soft compare_once ${R20}_s0hub_st2 ${R20}_s2_st2
  for E in 10 20 40; do
    soft compare_once ${R20}_s0hub_st2 ${R56}_e${E}                                      # depth at matched accuracy
    soft compare_once ${R56}_s0hub     ${R56}_e${E}                                      # accuracy at fixed depth
  done
  soft compare_once ${R20}_s0hub_st2 ${R56}_rand                                          # cross-depth null floor
  soft compare_once ${R56}_s0hub     ${R56}_rand

  echo "=== Stage 2: critic (resnet20 first: it is the name reference for --align position) ==="
  soft critic_once results/critic_v1_scale_r20hub_r56hub \
       --results ${R20}_s0hub_st2 ${R56}_s0hub --tol $T --align position
  for a in s1 s2; do
    soft critic_once results/critic_v1_scale_r20${a}_r56hub \
         --results ${R20}_${a}_st2 ${R56}_s0hub --tol $T --align position
  done
  soft critic_once results/critic_v1_resnet20_st2_band \
       --results ${R20}_s0hub_st2 ${R20}_s1_st2 ${R20}_s2_st2 --tol $T
  for E in 10 20 40; do
    soft critic_once results/critic_v1_scale_r20hub_r56e${E} \
         --results ${R20}_s0hub_st2 ${R56}_e${E} --tol $T --align position
  done
  soft critic_once results/critic_v1_resnet56_null --results ${R56}_s0hub ${R56}_rand --tol $T
}

# --- A3: margin_typeb (docs/plans/A3_MARGIN.md; pre-registration experiments/queue/margin_v1_resnet20_s1.yaml) --
# margin_rebuild <margin exp>: Stage B only on results/atlas_<same>/dump through a dump symlink (as atlas/ladder.py)
margin_rebuild() {
  local exp="$1" src="atlas_${1#margin_}"
  [[ -f "experiments/queue/$exp.yaml" ]] || { echo "[skip] $exp: no manifest"; return 0; }
  [[ -f "results/$src/dump/meta.json" ]] || {                 # tolerated only for dumps made earlier this session
    case "$exp" in
      margin_v1_resnet20_s3|margin_v1_resnet20_s4|margin_v1_resnet56_s0hub) echo "[skip] $exp: no dump at results/$src/dump"; return 0;;
      *) echo "ERROR: $exp: required dump missing at results/$src/dump" >&2; return 1;;
    esac; }
  mkdir -p "results/$exp" || return 1
  [[ -e "results/$exp/dump" ]] || ln -s "../$src/dump" "results/$exp/dump" || return 1
  run_stage "$exp" --skip-extract
}

block_a3() {
  local e T=experiments/tolerances_default.yaml M=results/margin_v1_resnet20
  echo "=== A3: margin_typeb on the discovery and null dumps (v0, v1 hub, rand; resnet56 hub for E5) ==="
  for e in margin_v0_resnet20_cifar10 margin_v1_resnet20_s0hub margin_v1_resnet20_rand margin_v1_resnet56_s0hub; do
    soft margin_rebuild "$e"
  done
  echo "=== A3: M1 gate (v0 penult legacy block); FAIL leaves the confirmation seeds untouched ==="
  [[ -f results/margin_v0_resnet20_cifar10/atlas.json ]] || { echo "[A3] M1 NOT RUN (no v0 margin atlas): confirmation seeds untouched; M1 is unevaluated, not FAIL"; return 1; }
  if python - results/margin_v0_resnet20_cifar10/atlas.json <<'EOF'
import json, sys
lg = (json.load(open(sys.argv[1]))["per_layer"]["penult"].get("margin_typeb") or {}).get("legacy") or {}
a, r = lg.get("dir_auc_margin"), lg.get("raw_auc_margin")
print(f"[A3] M1 dir_auc_margin={a} raw_auc_margin={r} (pass: 0.865 <= dir <= 0.925 and raw < 0.5)")
sys.exit(0 if a is not None and r is not None and 0.865 <= a <= 0.925 and r < 0.5 else 1)
EOF
  then
    echo "=== A3: M1 PASS -> margin_typeb on the confirmation dumps (s1, s2, twin s1_ref1, s3, s4) ==="
    for e in margin_v1_resnet20_s1 margin_v1_resnet20_s2 margin_v1_resnet20_s1_ref1 margin_v1_resnet20_s3 \
             margin_v1_resnet20_s4; do
      soft margin_rebuild "$e"
    done
    echo "=== A3: critic (margin-only dirs; only penult is judged) ==="
    soft critic_once results/critic_margin_v1_s1_s2 --results ${M}_s1 ${M}_s2 --tol $T
    soft critic_once results/critic_margin_v1_s0_s1_s2 --results ${M}_s0hub ${M}_s1 ${M}_s2 --tol $T
    soft critic_once results/critic_margin_v1_s1_noise --results ${M}_s1 ${M}_s1_ref1 --tol $T
    if [[ -f ${M}_s3/atlas.json && -f ${M}_s4/atlas.json ]]; then
      soft critic_once results/critic_margin_v1_s1_s4 --results ${M}_s1 ${M}_s2 ${M}_s3 ${M}_s4 --tol $T
    fi
  else
    echo "[A3] M1 margin clause FAIL (values above): confirmation seeds untouched; row 9 gets no verdict"
  fi
}

# run_block <function>: isolated in a subshell with errexit; a failure is logged and later blocks still run
FAILED=()
run_block() {
  local rc
  echo "=== block $1: start $(date) ==="
  set +e; ( set -e; "$1" ); rc=$?; set -e
  if (( rc )); then FAILED+=("$1"); echo "=== block $1 FAILED (exit $rc) $(date) ==="; else echo "=== block $1 OK $(date) ==="; fi
}

if has_mode --stage1b; then run_block block_stage1b; fi       # trains s3, s4 (concurrently) first
if has_mode --stage2;  then run_block block_stage2;  fi       # then the resnet56 rungs, one at a time: never overlapping
if has_mode --a3;      then run_block block_a3;      fi       # last: reads the s3, s4 and resnet56 dumps

echo "=== done. results (dumps stay on the volume, gitignored): $(pwd -P)/results ==="
echo "Pull to Windows (Git Bash; IP/port from 'runpodctl ssh info <pod-id>'), then commit + push there:"
echo "  ssh -i ~/.runpod/ssh/runpodctl-ssh-key -p <port> root@<ip> \\"
echo "    'tar -C $(pwd -P) -czf - --exclude=dump --exclude=\"dump_step*\" results' \\"
echo "    | tar -xzf - -C /c/Users/admin/Documents/GitHub/Atlas"
echo "After committing on Windows, update the pod: git -C $(pwd -P) fetch && git -C $(pwd -P) reset --hard origin/main"
if [[ -s "$SOFT_LOG" ]]; then echo "soft failures ($SOFT_LOG):"; cat "$SOFT_LOG"; FAILED+=("soft-steps"); fi
if (( ${#FAILED[@]} )); then echo "FAILED: ${FAILED[*]}" >&2; exit 1; fi
