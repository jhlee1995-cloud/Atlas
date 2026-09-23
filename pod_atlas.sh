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
#   bash /workspace/Atlas/pod_atlas.sh /workspace --a4b        # + depth-56 seeds s1/s2, matched rungs e50/e60/e70 (+e90),
#                                                              #   seed-12/13 replicates, _st3 band, twin, compare, critic,
#                                                              #   M56 margin rebuilds (docs/plans/STAGE2B.md)
#   bash /workspace/Atlas/pod_atlas.sh /workspace --b1         # + ImageNet val (pinned HF revision), ResNet50 gate, ViT-B/16
#                                                              #   + DeiT-B margin test, CIFAR E9 (docs/plans/B1_VIT_MARGIN.md)
#   bash /workspace/Atlas/pod_atlas.sh /workspace --anomaly    # + ANOMALY_H1 CPU probes AX-1..AX-4 on CIFAR dumps (numpy only;
#                                                              #   last, after B1; docs/plans/ANOMALY_H1.md)
#   bash /workspace/Atlas/pod_atlas.sh /workspace --b1b        # B1b: ViT runs after the gate-anchor correction
#                                                              #   (docs/plans/B1B_AMENDMENT.md; reuses B1's ResNet50 runs)
#   bash /workspace/Atlas/pod_atlas.sh /workspace --data-only  # datasets only (no GPU needed)
# Flags combine (e.g. --a4b --b1); blocks always run in this order: stage1, stage1b, stage2, a3, a4b, b1, anomaly; --a4b or
# --b1 adds the margin preflight (after the smoke tests). --stage1b, --stage2 and --a3 ran on 2026-09-23 and are
# committed: never add them again.
# stage1b / stage2 / a3 / a4b / b1 each run isolated (run_block): a failing block is logged and later blocks still run;
# steps wrapped in soft() log a failure and let their block go on. The script exits 1 at the end if any block
# or soft step failed (soft failures are listed in <volume>/logs/soft_failures_<timestamp>.log).
#
# Run it detached so an SSH drop does not kill it; the full log goes to <volume>/logs/:
#   mkdir -p /workspace/logs && setsid bash /workspace/Atlas/pod_atlas.sh /workspace < /dev/null > /workspace/logs/launch.log 2>&1 &
#   tail -F /workspace/logs/launch.log
# A stage whose results/<exp>/provenance.json (written last by atlas.run) exists is skipped
# (ATLAS_REBUILD=1 forces it).
set -euo pipefail
VOLUME="$(realpath -m "${1:?usage: bash pod_atlas.sh <volume> [--data-only | --stage1 | --stage1b | --stage2 | --a3 | --a4b | --b1 | --anomaly | --b1b ...]}")"
shift
for m in "$@"; do                                              # a typo must not silently skip a block
  case "$m" in --data-only|--stage1|--stage1b|--stage2|--a3|--a4b|--b1|--anomaly|--b1b) ;; *) echo "ERROR: unknown flag '$m'" >&2; exit 1;; esac
done
MODES=" $* "
has_mode() { [[ "$MODES" == *" $1 "* ]]; }
cd "$(dirname "$(readlink -f "$0")")"                        # repo root, whatever the caller's cwd
# --b1 needs the B1 code (scripts/b1_data.py and block_b1, docs/plans/B1_VIT_MARGIN.md); at a commit without it the
# flag would reach the dispatch only after every other block. Refuse it here instead.
if has_mode --b1 && ! { [[ -f scripts/b1_data.py ]] && grep -q '^block_b1() *{' pod_atlas.sh; }; then
  echo "ERROR: --b1: B1 is not in this checkout (scripts/b1_data.py or block_b1 missing)" >&2; exit 1
fi
if has_mode --anomaly && ! { [[ -f scripts/anomaly_probe.py ]] && grep -q '^block_anomaly() *{' pod_atlas.sh; }; then
  echo "ERROR: --anomaly: ANOMALY_H1 is not in this checkout (scripts/anomaly_probe.py or block_anomaly missing)" >&2; exit 1
fi
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
# B1 data (integration D11): ImageNet val at the pinned HF revision + ReaL labels. Inline soft form: soft() is
# defined further down. `--data-only --b1` provisions it without a GPU.
if has_mode --b1; then   # scripts/b1_data.py; a failure stops only block_b1, which re-verifies (hard)
  HF_HUB_DOWNLOAD_TIMEOUT=60 timeout 1200 python scripts/b1_data.py provision "$VOLUME" \
    || echo "[soft] FAILED (exit $?): B1 data provisioning" | tee -a "$SOFT_LOG"
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

# --- A4b/B1 preflight (hard; integration D6): B1 changes atlas/invariants/margin.py and A4b's M56 rebuilds run it too;
# with the B1 keys off it must reproduce the committed A3 atlases (check_rebuild leaf rule, scripts/check_rebuild.py:
# 20-22; bitwise recorded) before any block uses it. A relaunch records into scratch (the committed record stays).
if has_mode --a4b || has_mode --b1; then
  P0="${ATLAS_P0_CHECK_DIR:-results/instrument_check_a4b_b1}"
  if [[ -e "$P0" ]]; then P0="$VOLUME/scratch/instrument_check_a4b_b1_$STAMP"; echo "[preflight] relaunch: record in $P0"; fi
  echo "=== preflight: margin_typeb (flags off) reproduces the committed A3 atlases -> $P0 ==="
  for x in resnet20 resnet56; do
    [[ -f results/atlas_v1_${x}_s0hub/dump/meta.json ]] || { echo "ERROR: results/atlas_v1_${x}_s0hub/dump missing" >&2; exit 1; }
    python scripts/check_rebuild.py rebuild --dump results/atlas_v1_${x}_s0hub/dump \
           --committed results/margin_v1_${x}_s0hub/atlas.json --manifest experiments/queue/margin_v1_${x}_s0hub.yaml \
           --work "$VOLUME/scratch/p0_${x}_$STAMP" --out "$P0/margin_${x}_s0hub"
  done
fi

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
# Blocks for --stage1b, --stage2, --a3, --a4b, --b1. Every output directory below is new (never a committed Stage 0/1
# result dir): compare always writes into its --b run, critics and checks into their own --out.
# Relaunch (append-only): compares and critics whose output already exists are skipped (once), atlases are
# skipped by run_stage (B1's ImageNet runs by b1_stage, which never rebuilds), the B1 gate record is reused, and the
# Stage 1b / A4b / B1 instrument checks refuse an existing check dir (ATLAS_STAGE1B_CHECK_DIR, ATLAS_A4B_CHECK_DIR,
# ATLAS_B1_CHECK_DIR).
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

# --- A4b: depth-56 replication + matched rung (docs/plans/STAGE2B.md; predictions atlas_v1_resnet56_s1.yaml) -------
# a4b_wait <file> <pid> [<soft-log text>]: block until <file> exists, the training lane <pid> has exited, or the soft
# log records <text> (train_one's "FAILED: training <name> (" line); 0 iff the file exists
a4b_wait() {
  while [[ ! -f "$1" ]] && kill -0 "$2" 2>/dev/null && ! { [[ -n "${3:-}" ]] && grep -qF "$3" "$SOFT_LOG" 2>/dev/null; }; do
    sleep 15
  done
  [[ -f "$1" ]]
}

# a4b_ladder <out.json>: the pre-registered matched-rung rule on the seed-11 train.json files that exist. matched_rung.py
# writes <out.json> atomically, also on bad input (status "error", exit 2), so the CPU lane never reads a partial file
# and never idles on a missing one.
a4b_ladder() {
  local tj=() E
  for E in 40 50 60 70 90; do
    if [[ -f "results/train_resnet56_e$E/train.json" ]]; then tj+=("results/train_resnet56_e$E/train.json"); fi
  done
  python scripts/matched_rung.py --out "$1" "${tj[@]}" || echo "error none"
}

# a4b_train_lane <workers> <check dir>: the GPU lane, one training at a time. resnet56 at 8 workers consumes about
# 14.5k img/s (3.45-3.52 s per epoch) while 5 workers fed resnet20 at about 40k img/s, so resnet56 training is not
# data-bound; it is GPU-bound, and a second concurrent training would only share the GPU. train_one never fails the
# lane (a failed training is a soft failure), and every step below tolerates a missing run.
a4b_train_lane() {
  local M="$VOLUME/models" w="$1" I="$2" E s res
  for E in 50 60 70; do
    train_one "resnet56_e$E" "$M/resnet56_s11_e${E}_chenyaofo.pt" "results/train_resnet56_e${E}/train.json" \
              --arch cifar10_resnet56 --seed 11 --epochs "$E" --workers "$w"
  done
  res=$(a4b_ladder "$I/ladder_first.json")
  if [[ "$res" == below* ]]; then                 # every rung below 0.9209: one extension rung (STAGE2B.md)
    train_one resnet56_e90 "$M/resnet56_s11_e90_chenyaofo.pt" results/train_resnet56_e90/train.json \
              --arch cifar10_resnet56 --seed 11 --epochs 90 --workers "$w"
  fi
  res=$(a4b_ladder "$I/ladder.json")              # final; the CPU lane waits for this file
  echo "[a4b] matched-rung rule: $res"
  if [[ "$res" == matched* ]]; then               # seed-12 and seed-13 replicates at E* (review item 1): their length
    for s in 12 13; do                            # is fixed by seed 11; each counts iff its own accuracy is in the window
      train_one "resnet56_s${s}m" "$M/resnet56_s${s}m_chenyaofo.pt" "results/train_resnet56_s${s}m/train.json" \
                --arch cifar10_resnet56 --seed "$s" --epochs "${res#matched }" --workers "$w"
    done
  fi
  for s in 1 2; do                                # full recipe, last epoch (train_second_seed.py defaults)
    train_one "resnet56_s$s" "$M/resnet56_s${s}_chenyaofo.pt" "results/train_resnet56_s${s}/train.json" \
              --arch cifar10_resnet56 --seed "$s" --epochs 200 --workers "$w"
  done
}

block_a4b() {
  local M="$VOLUME/models" R20=results/atlas_v1_resnet20 R56=results/atlas_v1_resnet56 T=experiments/tolerances_default.yaml
  local I="${ATLAS_A4B_CHECK_DIR:-results/instrument_check_a4b}" cpus w lane e a b i j res changed EM=""
  local S=(s0hub s1 s2 s3 s4) R20S=() X=() OTH=() MG=()
  # Stage A inputs (training included) and Stage B code; requirements.txt is judged by versions.json (integration D7)
  local SA_CODE="atlas/extract_acts.py atlas/factors atlas/config.py atlas/context.py extract scripts/train_second_seed.py"
  local SB="atlas/build.py atlas/registry.py atlas/invariants atlas/compare.py atlas/critic.py experiments/tolerances_default.yaml"
  [[ ! -e "$I" ]] || { echo "ERROR: $I exists (committed A4b record); relaunch with ATLAS_A4B_CHECK_DIR=${I}_r2" >&2; return 1; }
  [[ "${ATLAS_REBUILD:-0}" != 1 ]] || { echo "ERROR: ATLAS_REBUILD=1 would rewrite committed results; unset it (integration D18)" >&2; return 1; }
  for e in s0hub s1 s2; do
    [[ -f ${R20}_${e}_st2/dump/meta.json ]] || { echo "ERROR: ${R20}_${e}_st2/dump missing on the volume" >&2; return 1; }
  done
  for e in 1 2 3 4; do
    [[ -f "$M/resnet20_s${e}_chenyaofo.pt" ]] || { echo "ERROR: $M/resnet20_s${e}_chenyaofo.pt missing" >&2; return 1; }
  done
  mkdir -p "$M" "$I" "$VOLUME/scratch"

  echo "=== A4b G0d: torch and GPU equal the Stage 2 ladder; python, numpy, scipy, sklearn equal Stage 1b ==="
  python - "$I" <<'EOF'
import json, os, platform, sys
import numpy, scipy, sklearn, torch, torchvision
v = {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
     "sklearn": sklearn.__version__, "torch": torch.__version__, "torchvision": torchvision.__version__,
     "device": torch.cuda.get_device_name(0)}
with open(os.path.join(sys.argv[1], "versions.json"), "w") as f:
    json.dump(v, f, indent=1)
r = json.load(open("results/train_resnet56_e40/train.json"))
ref = json.load(open("results/instrument_check_stage1b/check.json"))["versions"]
print(f"[a4b] {v}; Stage 2 ladder: torch {r['torch']} on {r['device']}; Stage 1b: {ref}")
assert v["torch"] == r["torch"] and v["device"] == r["device"], ("torch/GPU differ from the Stage 2 ladder", v, r["torch"], r["device"])
assert {k: v[k] for k in ref} == ref, ("python/numpy/scipy/sklearn differ from results/instrument_check_stage1b", v, ref)
EOF
  echo "=== A4b G0c/G0d: Stage A/B code diff since the Stage 2 run 664bd25 (allowlist: B1's margin.py, preflight-checked) ==="
  { echo "HEAD $(git rev-parse --short HEAD)"; git diff --stat 664bd25 HEAD -- $SA_CODE $SB requirements.txt; } > "$I/code_diff.txt"
  cat "$I/code_diff.txt"
  changed=$(git diff --name-only 664bd25 HEAD -- $SA_CODE $SB | tr '\n' ' ')
  if [[ -n "${changed// /}" && "$changed" != "atlas/invariants/margin.py " && "${ATLAS_A4B_ALLOW_DIFF:-0}" != 1 ]]; then
    echo "ERROR: Stage A/B files changed since 664bd25 beyond B1's margin.py (preflight-checked): $changed" >&2; return 1
  fi

  cpus=$(cpu_quota); w=$(( cpus - 2 )); if (( w < 2 )); then w=2; fi; if (( w > 8 )); then w=8; fi
  echo "=== A4b: GPU lane in the background ($cpus CPUs -> $w data workers; log $VOLUME/logs/a4b_train_lane.log) ==="
  a4b_train_lane "$w" "$I" > "$VOLUME/logs/a4b_train_lane.log" 2>&1 &
  lane=$!
  # no training outlives this block (review item 17): an early exit of the run_block subshell kills the lane and its
  # training. $lane is expanded now: the trap runs after block_a4b's locals are gone.
  trap "pkill -TERM -P $lane 2>/dev/null; kill $lane 2>/dev/null; true" EXIT

  echo "=== A4b G0c (recorded): Stage B replay of the st2 hub dump with the st3 manifest ==="
  soft python scripts/check_rebuild.py rebuild --dump ${R20}_s0hub_st2/dump --committed ${R20}_s0hub_st2/atlas.json \
       --manifest experiments/queue/atlas_v1_resnet20_s0hub_st3.yaml --work "$VOLUME/scratch/a4b_rebuild_st2_$STAMP" --out "$I"

  echo "=== A4b: atlases that need no new checkpoint, while the lane trains (resnet20 band first: rule 6) ==="
  for e in resnet20_s0hub_st3 resnet20_s1_st3 resnet20_s2_st3 resnet20_s3_st3 resnet20_s4_st3 resnet56_s0hub_st3 \
           resnet56_s0hub_ref1; do
    soft run_stage "atlas_v1_$e"
  done
  if [[ -f "$M/resnet56_s11_e40_chenyaofo.pt" ]]; then    # e40 enters only as this same-session re-measure (review item 7)
    soft run_stage atlas_v1_resnet56_e40_st3
  else
    echo "[soft] FAILED: $M/resnet56_s11_e40_chenyaofo.pt missing: no atlas_v1_resnet56_e40_st3 (no interpolation through e40)" | tee -a "$SOFT_LOG"
  fi
  echo "=== A4b: atlases of the new checkpoints, each as soon as its train.json exists ==="
  for e in e50 e60 e70; do
    if a4b_wait "results/train_resnet56_$e/train.json" "$lane" "FAILED: training resnet56_$e ("; then soft run_stage "atlas_v1_resnet56_$e"; fi
  done
  a4b_wait "$I/ladder.json" "$lane" || true
  res=$(python -c "import json, sys; d = json.load(open(sys.argv[1])); print(d['status'], d['e_star'])" "$I/ladder.json" \
        2>/dev/null || echo "error None")
  echo "[a4b] ladder: $res"
  if [[ "$res" == matched* ]]; then EM="e${res#matched }"; fi
  if [[ -f results/train_resnet56_e90/train.json ]]; then soft run_stage atlas_v1_resnet56_e90; fi
  for e in s12m s13m s1 s2; do
    if [[ ( "$e" == s12m || "$e" == s13m ) && -z "$EM" ]]; then continue; fi
    if a4b_wait "results/train_resnet56_$e/train.json" "$lane" "FAILED: training resnet56_$e ("; then soft run_stage "atlas_v1_resnet56_$e"; fi
  done
  wait "$lane" || true
  trap - EXIT
  grep -h "saved" "$VOLUME"/logs/train_resnet56_{e50,e60,e70,e90,s12m,s13m,s1,s2}.log 2>/dev/null || true

  echo "=== A4b: compare (needs the dumps; B is always an A4b run) ==="
  for e in s0hub s1 s2; do soft compare_once ${R20}_${e}_st2 ${R20}_${e}_st3 --same-space; done       # G0c replay
  for e in s3 s4; do soft compare_once ${R20}_$e ${R20}_${e}_st3 --same-space; done
  soft compare_once ${R56}_s0hub ${R56}_s0hub_st3 --same-space
  if [[ -f ${R56}_e40/dump/meta.json && -f ${R56}_e40_st3/atlas.json ]]; then
    soft compare_once ${R56}_e40 ${R56}_e40_st3 --same-space
  fi
  for i in 0 1 2 3; do for j in 1 2 3 4; do                                                           # W20 -> P20
    if (( j > i )); then soft compare_once ${R20}_${S[i]}_st3 ${R20}_${S[j]}_st3; fi
  done; done
  for a in "${S[@]}"; do                        # X (s1, s2) and XM (every rung and replicate: interpolation too)
    for b in s1 s2 s12m s13m e40_st3 e50 e60 e70 e90; do
      if [[ -f ${R56}_$b/atlas.json ]]; then soft compare_once ${R20}_${a}_st3 ${R56}_$b; fi
    done
  done
  soft compare_once ${R56}_s1 ${R56}_s2                                                                # W56 decision pair
  soft compare_once ${R56}_s0hub_st3 ${R56}_s1                                                         # corroboration
  soft compare_once ${R56}_s0hub_st3 ${R56}_s2
  soft compare_once ${R56}_s0hub_st3 ${R56}_s0hub_ref1                                                 # T56 twin
  for b in s12m s13m; do                                                                               # INFO
    if [[ -n "$EM" && -f ${R56}_$b/atlas.json ]]; then soft compare_once ${R56}_$EM ${R56}_$b; fi
  done
  if [[ -f ${R56}_s12m/atlas.json && -f ${R56}_s13m/atlas.json ]]; then soft compare_once ${R56}_s12m ${R56}_s13m; fi

  echo "=== A4b: critic (resnet20 first in every --align position run: it is the name reference) ==="
  for e in "${S[@]}"; do R20S+=("${R20}_${e}_st3"); done
  soft critic_once results/critic_v1_resnet56_s1_s2 --results ${R56}_s1 ${R56}_s2 --tol $T
  soft critic_once results/critic_v1_resnet56_s0_s1_s2 --results ${R56}_s0hub_st3 ${R56}_s1 ${R56}_s2 --tol $T
  soft critic_once results/critic_v1_resnet56_hub_noise --results ${R56}_s0hub_st3 ${R56}_s0hub_ref1 --tol $T
  soft critic_once results/critic_v1_resnet20_st3_band --results "${R20S[@]}" --tol $T
  soft critic_once results/critic_v1_scale3_r20_r56seeds --results "${R20S[@]}" ${R56}_s1 ${R56}_s2 --tol $T --align position
  X=(); for e in e40_st3 e50 e60 e70 e90 s12m s13m; do if [[ -f ${R56}_$e/atlas.json ]]; then X+=("${R56}_$e"); fi; done
  if (( ${#X[@]} )); then
    soft critic_once results/critic_v1_scale3_r20_r56matched --results "${R20S[@]}" "${X[@]}" --tol $T --align position
  fi
  echo "=== A4b G0b: s1, s2, s12m, s13m dump meta vs the hub_st3 dump; sha256 new, also against every rung ==="
  X=(); for e in s1 s2 s12m s13m; do if [[ -f ${R56}_$e/atlas.json ]]; then X+=("${R56}_$e"); fi; done
  OTH=(); for e in e40_st3 e50 e60 e70 e90; do if [[ -f ${R56}_$e/atlas.json ]]; then OTH+=("${R56}_$e"); fi; done
  if (( ${#X[@]} )); then
    soft python scripts/check_rebuild.py dump-meta --ref ${R56}_s0hub_st3 --others "${OTH[@]}" --new "${X[@]}" --out "$I"
  fi

  echo "=== A4b M56: margin_typeb Stage-B rebuilds (resnet20 band first: rule 6), critic, maxprob ties ==="
  for e in resnet20_s0hub_st3 resnet20_s1_st3 resnet20_s2_st3 resnet20_s3_st3 resnet20_s4_st3 resnet56_s0hub_st3 \
           resnet56_s1 resnet56_s2 resnet56_e50 resnet56_e60 resnet56_e70 resnet56_e90 resnet56_s12m resnet56_s13m; do
    if [[ -f results/atlas_v1_$e/dump/meta.json ]]; then soft margin_rebuild "margin_v1_$e"; MG+=("results/atlas_v1_$e"); fi
  done
  soft critic_once results/critic_margin_v1_resnet56_s1_s2 --results results/margin_v1_resnet56_s1 results/margin_v1_resnet56_s2 --tol $T
  if (( ${#MG[@]} )); then soft python scripts/maxprob_ties.py --out "$I/maxprob_ties.json" "${MG[@]}"; fi
  echo "[a4b] D1, bands, levels and labels are computed on Windows after the pull:"
  echo "      node scripts/a4b_eval.js --p <P> --p-run <P_run> --check-dir $(basename "$I") --json results/atlas_v1_resnet56_s1/a4b_eval.json"
}

# --- B1: the ViT margin test (docs/plans/B1_VIT_MARGIN.md; pre-registration experiments/queue/margin_b1_vitb16.yaml) --
# b1_stage <exp>: ImageNet Stage A + B; never rebuilt (ATLAS_REBUILD ignored): an existing atlas.json is the one touch.
# A killed or failed run leaves no atlas.json (integration D18 decides what a relaunch may do).
b1_stage() {
  local exp="$1" n
  if [[ -f "results/$exp/atlas.json" ]]; then echo "[skip] results/$exp/atlas.json exists (never rebuilt)"; return 0; fi
  n=$(cpu_quota)                                  # BLAS threads = cgroup quota, only for ImageNet runs (D19)
  OMP_NUM_THREADS=$n OPENBLAS_NUM_THREADS=$n MKL_NUM_THREADS=$n \
    timeout 7200 python -m atlas.run_imagenet --manifest "experiments/queue/$exp.yaml" --volume "$VOLUME"   # r2: 1800 s killed the
    # resnet50 Stage B at its 18th tap (margin_typeb ~65-118 s per tap at n = 25,000); results/margin_b1_vitb16/RELAUNCH_r2.md
}
b1_cifar() {   # E9: Stage B only (b1: true) on a dump A4b built in this session, through a dump symlink
  local exp="$1" src="$2"
  [[ -f "results/$src/dump/meta.json" ]] || { echo "[b1] $exp: no dump at results/$src/dump (A4b did not build it)"; return 1; }
  mkdir -p "results/$exp" || return 1
  [[ -e "results/$exp/dump" ]] || ln -s "../$src/dump" "results/$exp/dump" || return 1
  run_stage "$exp" --skip-extract
}
b1_workers() { # DataLoader workers: 2 prefetched 224x224 float32 batches per worker in /dev/shm, x1.5 headroom
  local b="$1" shm w
  shm=$(df -B1 --output=size /dev/shm 2>/dev/null | tail -n 1 | tr -d ' ' || true)
  w=$(( $(cpu_quota) - 2 )); if (( w > 8 )); then w=8; fi; if (( w < 0 )); then w=0; fi
  while (( w > 0 && ${shm:-0} < w * 2 * b * 602112 * 3 / 2 )); do w=$(( w / 2 )); done
  echo "$w"
}
b1_need_bytes() { # volume still to be written: float32 dumps + float16 logits of the runs without a dump (integration
  local e b=774000000                             # section 3), plus the ViT/ResNet weights (0.77 GB, counted anyway)
  for e in margin_b1_resnet50_legacy10k margin_b1_resnet50 margin_b1_resnet50_swap margin_b1_vitb16 \
           margin_b1_vitb16_swap margin_b1_deitb margin_b1_deitb_swap; do
    if [[ -f "results/$e/dump/meta.json" ]]; then continue; fi
    case "$e" in *legacy10k) b=$(( b + 520000000 ));; *resnet50*) b=$(( b + 3540000000 ));; *) b=$(( b + 2250000000 ));; esac
  done
  echo "$b"
}
b1_dir() {     # results dir of a run, or of its P0 re-run when that exists (prereg review item 2)
  if [[ -f "results/${1}_r2/atlas.json" ]]; then echo "results/${1}_r2"; else echo "results/$1"; fi
}
block_b1() {
  local T=experiments/tolerances_default.yaml I="${ATLAS_B1_CHECK_DIR:-results/instrument_check_b1}" e x used need v d c
  [[ ! -e "$I" ]] || { echo "ERROR: $I exists (committed B1 record); relaunch with ATLAS_B1_CHECK_DIR=${I}_r2" >&2; return 1; }
  [[ "${ATLAS_REBUILD:-0}" != 1 ]] || { echo "ERROR: ATLAS_REBUILD=1 would rewrite committed results; unset it (integration D18)" >&2; return 1; }
  mkdir -p "$I"
  export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"   # 330-346 MB weight files; the default is 10 s
  echo "=== B1: imports (requirements.txt pins; integration D4) ==="
  python -c "import pyarrow, huggingface_hub, timm, safetensors; print('[b1]', pyarrow.__version__, huggingface_hub.__version__, timm.__version__, safetensors.__version__)"
  if command -v pgrep >/dev/null && pgrep -f scripts/train_second_seed.py >/dev/null; then
    echo "ERROR: a training is still running; B1 never shares the GPU" >&2; return 1
  fi
  echo "=== B1: data (hard: both parquet files at the pinned sha256, 50,000 rows, 1000 x 50; ReaL recorded) ==="
  python scripts/b1_data.py verify "$VOLUME" --out "$I/data.json"
  used=$(du -sb "$VOLUME" 2>/dev/null | cut -f1 || true); need=$(b1_need_bytes)
  echo "[b1] volume holds $(( ${used:-0} / 1000000 )) MB; B1 still adds ~$(( need / 1000000 )) MB (float32 dumps, float16 logits, weights)"
  if (( ${used:-0} + need > 48000000000 )); then echo "ERROR: not enough of the 50 GB quota left for B1" >&2; return 1; fi
  export ATLAS_EXTRACT_WORKERS; ATLAS_EXTRACT_WORKERS=$(b1_workers 256)
  echo "[b1] /dev/shm -> $ATLAS_EXTRACT_WORKERS DataLoader workers (recorded in every dump meta)"
  echo "=== B1: CPU tests (hooks, parquet, legacy replay, extractor meta contract, gate known answers) ==="
  python -m pytest -q tests/test_b1_imagenet.py
  echo "=== B1: ViT self-tests on random inputs (weights downloaded + hash-checked; no ImageNet image read) ==="
  for e in margin_b1_vitb16 margin_b1_deitb; do   # timeout: torch.hub's weight download has no socket timeout (D19)
    timeout 1200 python -m atlas.run_imagenet --manifest "experiments/queue/$e.yaml" --volume "$VOLUME" --selftest-random --out "$I/selftest_$e.json"
  done
  echo "=== B1 E9 (INFO): b1 keys on this session's A4b dumps; every A3 key must equal A4b's flag-off rebuild ==="
  for x in resnet20_s0hub_st3 resnet56_s0hub_st3 resnet56_s1 resnet56_s2; do soft b1_cifar "margin_b1_$x" "atlas_v1_$x"; done
  soft python - "$I" <<'EOF'
import json, os, sys
sys.path.insert(0, "scripts")
from check_rebuild import _walk
def keep(u, v):                      # v restricted to u's keys: b1 adds keys, every A3 key must be unchanged
    if isinstance(u, dict) and isinstance(v, dict):
        return {k: keep(u[k], v[k]) for k in u if k in v}
    if isinstance(u, list) and isinstance(v, list) and len(u) == len(v):
        return [keep(p, q) for p, q in zip(u, v)]
    return v
rep, bad = {}, []
for x in ("resnet20_s0hub_st3", "resnet56_s0hub_st3", "resnet56_s1", "resnet56_s2"):
    pa, pb = f"results/margin_v1_{x}/atlas.json", f"results/margin_b1_{x}/atlas.json"
    if not (os.path.exists(pa) and os.path.exists(pb)):
        rep[x] = "NOT_EVALUABLE (missing atlas)"; continue
    a, b = json.load(open(pa))["per_layer"], json.load(open(pb))["per_layer"]
    acc = {"n_leaves": 0, "n_exact": 0, "n_within_tol": 0, "max_abs_dev": {}, "mismatch": []}
    _walk(a, keep(a, b), "", acc)
    rep[x] = {"n_leaves": acc["n_leaves"], "n_exact": acc["n_exact"], "mismatch_first20": acc["mismatch"][:20]}
    if acc["mismatch"] or acc["n_exact"] != acc["n_leaves"]:
        bad.append(x)
json.dump(rep, open(os.path.join(sys.argv[1], "e9_identity.json"), "w"), indent=1)
print("[B1] E9 A3-key identity (exact)", "PASS" if not bad else f"FAIL {bad}")
sys.exit(1 if bad else 0)
EOF
  echo "=== B1: ResNet50 (discovery: gate G source and CNN contrast; a failure stops the block, the ViTs stay untouched) ==="
  for e in margin_b1_resnet50_legacy10k margin_b1_resnet50 margin_b1_resnet50_swap; do b1_stage "$e"; done
  echo "volume usage: $(du -sh "$VOLUME" 2>/dev/null | cut -f1) of the 50 GB quota"
  echo "=== B1: gate G (G0-G3 + self-tests); CLOSED leaves ViT-B/16 and DeiT-B untouched ==="
  if python scripts/b1_gate.py --out results/b1_gate --check-dir "$I" --volume "$VOLUME"; then
    for e in margin_b1_vitb16 margin_b1_vitb16_swap margin_b1_deitb margin_b1_deitb_swap; do
      soft b1_stage "$e"
      if [[ -f "experiments/queue/${e}_r2.yaml" ]]; then soft b1_stage "${e}_r2"; fi   # P0 re-run (committed amendment)
    done
    v=$(b1_dir margin_b1_vitb16); d=$(b1_dir margin_b1_deitb); c=results/critic_b1_vit_pair
    if [[ "$v$d" == *_r2* ]]; then c="${c}_r2"; fi
    soft critic_once "$c" --results "$v" "$d" --tol $T
    for e in vitb16 deitb; do
      soft critic_once "results/critic_b1_${e}_swap" --results "results/margin_b1_$e" "results/margin_b1_${e}_swap" --tol $T
    done
  else
    echo "[B1] gate G CLOSED or not evaluable (results/b1_gate/gate.json): no ViT run, no A/B label"
  fi
  soft critic_once results/critic_b1_resnet50_swap --results results/margin_b1_resnet50 results/margin_b1_resnet50_swap --tol $T
  echo "[b1] labels are computed on Windows after the pull (commit the results first):"
  echo "      node scripts/b1_verdicts.js --p <P> --p-run <P_run> --a4b results/atlas_v1_resnet56_s1/a4b_eval.json --json results/margin_b1_vitb16/verdicts.json"
}

# --- B1b: the ViT margin test after the B1 gate-anchor correction (docs/plans/B1B_AMENDMENT.md) -------------------------
# Reuses B1's committed ResNet50 atlases and check dir (read only), recomputes gate G with the corrected G0 anchor into
# results/b1b_gate (scripts/b1b_gate.py), and only if it is OPEN builds the four ViT runs (their first ImageNet touch).
block_b1b() {
  local T=experiments/tolerances_default.yaml I0="${ATLAS_B1B_SRC_CHECK_DIR:-results/instrument_check_b1_r2}" \
        I="${ATLAS_B1B_CHECK_DIR:-results/instrument_check_b1b}" e v d c
  [[ ! -e "$I" ]] || { echo "ERROR: $I exists (committed B1b record); relaunch with ATLAS_B1B_CHECK_DIR=${I}_r2" >&2; return 1; }
  [[ "${ATLAS_REBUILD:-0}" != 1 ]] || { echo "ERROR: ATLAS_REBUILD=1 would rewrite committed results" >&2; return 1; }
  for e in margin_b1_resnet50_legacy10k margin_b1_resnet50 margin_b1_resnet50_swap; do
    [[ -f "results/$e/atlas.json" ]] || { echo "ERROR: results/$e/atlas.json missing (B1b reuses B1's ResNet50 runs)" >&2; return 1; }
  done
  [[ -f "$I0/data.json" && -f "$I0/selftest_margin_b1_vitb16.json" && -f "$I0/selftest_margin_b1_deitb.json" ]] \
    || { echo "ERROR: $I0 lacks data.json or the ViT self-tests" >&2; return 1; }
  if command -v pgrep >/dev/null && pgrep -f scripts/train_second_seed.py >/dev/null; then
    echo "ERROR: a training is still running; B1b never shares the GPU" >&2; return 1
  fi
  mkdir -p "$I"
  export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
  python -c "import pyarrow, huggingface_hub, timm, safetensors; print('[b1b]', pyarrow.__version__, huggingface_hub.__version__, timm.__version__, safetensors.__version__)"
  python scripts/b1_data.py verify "$VOLUME" --out "$I/data.json"
  export ATLAS_EXTRACT_WORKERS; ATLAS_EXTRACT_WORKERS=$(b1_workers 256)
  echo "=== B1b: gate G with the corrected G0 anchor (scripts/b1b_gate.py; B1's results/b1_gate stays CLOSED) ==="
  if python scripts/b1b_gate.py --out results/b1b_gate --check-dir "$I0" --volume "$VOLUME"; then
    for e in margin_b1_vitb16 margin_b1_vitb16_swap margin_b1_deitb margin_b1_deitb_swap; do soft b1_stage "$e"; done
    v=$(b1_dir margin_b1_vitb16); d=$(b1_dir margin_b1_deitb); c=results/critic_b1_vit_pair
    if [[ "$v$d" == *_r2* ]]; then c="${c}_r2"; fi
    soft critic_once "$c" --results "$v" "$d" --tol $T
    for e in vitb16 deitb; do
      soft critic_once "results/critic_b1_${e}_swap" --results "results/margin_b1_$e" "results/margin_b1_${e}_swap" --tol $T
    done
  else
    echo "[B1b] gate G CLOSED or not evaluable (results/b1b_gate/gate.json): no ViT run, no A/B label"
  fi
  echo "[b1b] labels are computed on Windows after the pull (commit the results first):"
  echo "      node scripts/b1b_verdicts.js --p f1c3c43 --p-run d257d91,68244f7 --p-b <P_B> --p-b-run <P_B_run> --a4b results/atlas_v1_resnet56_s1/a4b_eval.json --json results/margin_b1_vitb16/verdicts_b1b.json"
}

# --- ANOMALY_H1: CPU Stage-B probes AX-1..AX-4 (docs/plans/ANOMALY_H1.md; scripts/anomaly_probe.py) ---------------------
# anomaly_probe_one <run> [<fallback run>]: one CIFAR dump -> results/anomaly_probe_<tag>/probe.json (a NEW dir; the probe
# refuses to write into an existing non-empty one). The fallback is the same weights measured in an earlier session (used
# only for discovery instances, and recorded in probe.json by its dump name). A dump that already has a probe.json under
# its own tag or any _r<k> tag is never probed again (review item C21: a relaunch never touches a confirmation dump twice).
# BLAS threads = the cgroup quota, for this new script only (integration D19 keeps A4b's and E9's environment unchanged).
anomaly_probe_one() {
  local run="$1" fb="${2:-}" src="$1" base tag n
  if [[ ! -f "results/$src/dump/meta.json" && -n "$fb" && -f "results/$fb/dump/meta.json" ]]; then
    echo "[anomaly] results/$run/dump missing: using $fb (same weights, earlier session; discovery role, recorded)"
    src="$fb"
  fi
  [[ -f "results/$src/dump/meta.json" ]] || { echo "[anomaly] no dump for $run${fb:+ or $fb}"; return 1; }
  base="${src#atlas_v1_}"
  if compgen -G "results/anomaly_probe_${base}/probe.json" >/dev/null || compgen -G "results/anomaly_probe_${base}_r[0-9]*/probe.json" >/dev/null; then
    echo "[skip] results/anomaly_probe_${base}{,_r<k>}/probe.json exists (append-only: the first complete record counts)"
    return 0
  fi
  tag="${base}${ATLAS_ANOM_TAG_SUFFIX:-}"
  if (( SECONDS - ANOM_T0 > ${ATLAS_ANOM_BUDGET_S:-2700} )); then
    echo "[anomaly] block budget ${ATLAS_ANOM_BUDGET_S:-2700} s spent: $run not probed"; return 1
  fi
  n=$(cpu_quota)
  OMP_NUM_THREADS=$n OPENBLAS_NUM_THREADS=$n MKL_NUM_THREADS=$n \
    timeout 900 python scripts/anomaly_probe.py --tag "$tag" --dumps "results/$src/dump"
}

block_anomaly() {
  local I="${ATLAS_ANOM_CHECK_DIR:-results/instrument_check_anomaly}" x
  ANOM_T0=$SECONDS
  [[ ! -e "$I" ]] || { echo "ERROR: $I exists (committed record); relaunch with ATLAS_ANOM_CHECK_DIR=${I}_r2 ATLAS_ANOM_TAG_SUFFIX=_r2" >&2; return 1; }
  [[ -f scripts/anomaly_probe.py && -f tests/test_anomaly_probe.py ]] || { echo "ERROR: ANOMALY_H1 code missing at this commit" >&2; return 1; }
  mkdir -p "$I"
  if command -v pgrep >/dev/null && pgrep -f scripts/train_second_seed.py >/dev/null; then
    echo "[anomaly] WARNING: a training is still running; the probes are CPU-only and continue"
  fi
  echo "=== ANOMALY_H1: known-answer tests and synthetic self-test (numpy only; a failure stops this block, not the session) ==="
  timeout 600 python -m pytest -q tests/test_anomaly_probe.py
  timeout 600 python scripts/anomaly_probe.py --selftest --selftest-out "$I/selftest.json"
  echo "=== ANOMALY_H1: gate instances (discovery; the resnet20 hub first: rule 6, same session) ==="
  soft anomaly_probe_one atlas_v1_resnet20_s0hub_st3 atlas_v1_resnet20_s0hub_st2
  soft anomaly_probe_one atlas_v1_resnet56_s0hub_st3 atlas_v1_resnet56_s0hub
  soft anomaly_probe_one atlas_v1_resnet56_e40_st3 atlas_v1_resnet56_e40
  echo "=== ANOMALY_H1: confirmation dumps (read once; definitions and thresholds frozen in P_A; holdout corruptions only here) ==="
  for x in s1 s2; do soft anomaly_probe_one "atlas_v1_resnet56_$x"; done
  echo "=== ANOMALY_H1: discovery context (spent resnet20 seeds) and random-init nulls (INFO) ==="
  soft anomaly_probe_one atlas_v1_resnet20_s1_st3 atlas_v1_resnet20_s1_st2
  soft anomaly_probe_one atlas_v1_resnet20_s2_st3 atlas_v1_resnet20_s2_st2
  soft anomaly_probe_one atlas_v1_resnet20_s3_st3 atlas_v1_resnet20_s3
  soft anomaly_probe_one atlas_v1_resnet20_s4_st3 atlas_v1_resnet20_s4
  for x in resnet20_rand resnet56_rand; do soft anomaly_probe_one "atlas_v1_$x"; done
  { ls -1 results/anomaly_probe_*/probe.json 2>/dev/null || true; } > "$I/probes.txt"
  echo "[anomaly] block wall $(( SECONDS - ANOM_T0 )) s; $(wc -l < "$I/probes.txt") probe.json file(s) (listed in $I/probes.txt)"
  echo "[anomaly] verdicts are computed on Windows after the pull, after the A4b and B1 evaluators (commit the results first):"
  echo "      node scripts/anomaly_eval.js --p <P_A> --p-run <P_run> --a4b results/atlas_v1_resnet56_s1/a4b_eval.json \\"
  echo "           --b1 results/margin_b1_vitb16/verdicts.json --cache results/anomaly_h1/idgauss_cache.json --json results/anomaly_h1/eval.json"
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
if has_mode --a3;      then run_block block_a3;      fi       # reads the s3, s4 and resnet56 dumps
if has_mode --a4b;     then run_block block_a4b;     fi       # before B1: its _st3 dumps are B1's same-session CIFAR anchor
if has_mode --b1;      then run_block block_b1;      fi       # last GPU block: never shares the GPU with a training
if has_mode --b1b;     then run_block block_b1b;     fi       # B1b: ViTs after the gate-anchor correction
if has_mode --anomaly; then run_block block_anomaly; fi       # CPU only, after B1; reads A4b's dumps, never a B1 dump

echo "=== done. results (dumps stay on the volume, gitignored): $(pwd -P)/results ==="
echo "Pull to Windows (Git Bash; IP/port from 'runpodctl ssh info <pod-id>'), then commit + push there:"
echo "  ssh -i ~/.runpod/ssh/runpodctl-ssh-key -p <port> root@<ip> \\"
echo "    'tar -C $(pwd -P) -czf - --exclude=dump --exclude=\"dump_step*\" results' \\"
echo "    | tar -xzf - -C /c/Users/admin/Documents/GitHub/Atlas"
echo "After committing on Windows, update the pod: git -C $(pwd -P) fetch && git -C $(pwd -P) reset --hard origin/main"
if [[ -s "$SOFT_LOG" ]]; then echo "soft failures ($SOFT_LOG):"; cat "$SOFT_LOG"; FAILED+=("soft-steps"); fi
if (( ${#FAILED[@]} )); then echo "FAILED: ${FAILED[*]}" >&2; exit 1; fi
