#!/usr/bin/env bash
# RunPod: datasets (idempotent) + Stage 0 (+ optional Stage 1) of the atlas.
#
# The repo MUST be a git clone ON the network volume (e.g. /workspace/Atlas): results/ is written
# relative to the repo, and only the volume survives a pod stop. Create the pod with
#   --network-volume-id kxfir1tryb --data-center-ids EU-RO-1   (mounted at /workspace)
#
#   bash /workspace/Atlas/pod_atlas.sh /workspace              # data + Stage 0
#   bash /workspace/Atlas/pod_atlas.sh /workspace --seed1      # + train seed 1, dump it, compare, critic
#   bash /workspace/Atlas/pod_atlas.sh /workspace --data-only  # datasets only (no GPU needed)
#
# Run it detached so an SSH drop does not kill it; the full log goes to <volume>/logs/:
#   mkdir -p /workspace/logs && setsid bash /workspace/Atlas/pod_atlas.sh /workspace < /dev/null > /workspace/logs/launch.log 2>&1 &
#   tail -F /workspace/logs/launch.log
# A stage whose results/<exp>/provenance.json (written last by atlas.run) exists is skipped
# (ATLAS_REBUILD=1 forces it).
set -euo pipefail
VOLUME="$(realpath -m "${1:?usage: bash pod_atlas.sh <volume> [--seed1|--data-only]}")"
MODE="${2:-}"
cd "$(dirname "$(readlink -f "$0")")"                        # repo root, whatever the caller's cwd

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
# result it is tracked, and if this script rewrites it (ATLAS_REBUILD=1, a --seed1 re-run of compare/critic)
# a plain `git pull` aborts. Update the pod only after its results are committed on Windows:
#   git -C /workspace/Atlas fetch && git -C /workspace/Atlas reset --hard origin/main   (dumps are untracked; kept)
grep -qxF '/results/' .git/info/exclude 2>/dev/null || echo '/results/' >> .git/info/exclude

# --- env: PEP 668 + caches on the volume (pod --env reaches PID 1 only, not SSH shells) ---
export PIP_BREAK_SYSTEM_PACKAGES=1 PYTHONUNBUFFERED=1
export PIP_CACHE_DIR="$VOLUME/.cache/pip" TORCH_HOME="$VOLUME/.cache/torch"
export HF_HOME="$VOLUME/.cache/huggingface" XDG_CACHE_HOME="$VOLUME/.cache"

# --- log on the volume ---
mkdir -p "$VOLUME/logs"
LOG="$VOLUME/logs/pod_atlas_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "=== log $LOG | git $(git rev-parse --short HEAD 2>/dev/null || echo NO_GIT) | $(date) ==="

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
[[ "$MODE" == "--data-only" ]] && { echo "=== data only: done ==="; exit 0; }

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

if [[ "$MODE" == "--seed1" ]]; then
  echo "=== Stage 1: second seed ==="
  mkdir -p "$VOLUME/models"
  if [[ ! -f "$VOLUME/models/resnet20_seed1.pt" ]]; then
    python scripts/train_second_seed.py --volume "$VOLUME" --seed 1 --out "$VOLUME/models/resnet20_seed1.pt"
  fi
  run_stage atlas_v0_resnet20_seed1 --weights "$VOLUME/models/resnet20_seed1.pt"
  python -m atlas.compare --a results/atlas_v0_resnet20_cifar10 --b results/atlas_v0_resnet20_seed1
  python -m atlas.critic  --results results/atlas_v0_resnet20_cifar10 results/atlas_v0_resnet20_seed1 \
         --tol experiments/tolerances_default.yaml --out results/critic_resnet20_s0_s1
fi

echo "=== done. results (dumps stay on the volume, gitignored): $(pwd -P)/results ==="
echo "Pull to Windows (Git Bash; IP/port from 'runpodctl ssh info <pod-id>'), then commit + push there:"
echo "  ssh -i ~/.runpod/ssh/runpodctl-ssh-key -p <port> root@<ip> \\"
echo "    'tar -C $(pwd -P) -czf - --exclude=dump --exclude=\"dump_step*\" results' \\"
echo "    | tar -xzf - -C /c/Users/admin/Documents/GitHub/Atlas"
echo "After committing on Windows, update the pod: git -C $(pwd -P) fetch && git -C $(pwd -P) reset --hard origin/main"
