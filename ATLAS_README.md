# atlas/ — Stage 0 of the map-first plan

Builds a reference **atlas** of a backbone's internal space: per-layer invariants of the
activation geometry on a fixed reference distribution, plus how that geometry deforms under
corruption. Same two-stage split as the rest of the repo: GPU once, CPU forever after.

```
STAGE A (GPU)   python -m atlas.extract_acts   data -> backbone -> dump/  (acts, labels, preds, pixel factors)
STAGE B (CPU)   python -m atlas.build          dump/ -> atlas.json + ATLAS.md + plots/
                python -m atlas.compare        atlas A vs atlas B -> deformation.json / DEFORMATION.md
                python -m atlas.critic         N atlases -> CRITIC.md (replicates? promotable?)
ONE SHOT        python -m atlas.run --manifest experiments/queue/<exp>.yaml --volume /workspace
```

## Install

```bash
python -m pip install -r requirements.txt       # on runpod-torch-v280 set PIP_BREAK_SYSTEM_PACKAGES=1 (PEP 668)
python -m pytest tests/test_atlas_smoke.py -v   # CPU, ~30 s, no torch needed
```

## Where things live

Everything that is data lives on the RunPod network volume `kxfir1tryb` (EU-RO-1), mounted at
`/workspace` on pods (`/runpod-volume` is the serverless path; do not use it on a pod). The repo is
cloned ONTO the volume, because `outputs.root` is relative to the repo and only the volume survives
a pod stop:

```
/workspace/
├── Atlas/                      git clone of this repo; results/<exp>/ (dump/ and dump_step*/ are gitignored)
├── datasets/                   cifar10/ cifar10_train/ cifar100/ svhn/ cifar10c/ (flat .npy) + manifest.json
├── models/                     resnet20_seed1.pt
├── .cache/                     torch hub, pip
└── logs/                       launch.log (latest launch) + pod_atlas_<timestamp>.log (per run)
```

The pod only pulls code. Results come back to Windows over SSH (`tar` without dumps; the command is
printed at the end of `pod_atlas.sh`) and are committed there. Update the code on the pod only after
its results are committed on Windows; a plain `git pull` aborts once the pod has rewritten a committed
result file (ATLAS_REBUILD=1, or a `--seed1` re-run of compare/critic):

```bash
git -C /workspace/Atlas fetch && git -C /workspace/Atlas reset --hard origin/main   # gitignored dumps are kept
```

## Stage 0 on the pod

```bash
cd /workspace && git clone https://github.com/jhlee1995-cloud/Atlas.git
mkdir -p /workspace/logs && setsid bash /workspace/Atlas/pod_atlas.sh /workspace < /dev/null > /workspace/logs/launch.log 2>&1 &
# data (once per volume, incl. CIFAR-10-C from zenodo) -> GPU preflight -> smoke test -> Stage 0
# -> results/atlas_v0_resnet20_cifar10/{atlas.json, ATLAS.md, plots/, manifest_used.yaml, provenance.json, dump/}
tail -F /workspace/logs/launch.log      # everything, including guard errors
```

`bash /workspace/Atlas/pod_atlas.sh /workspace --data-only` populates datasets without a GPU. CIFAR-10-C
must sit flat in `datasets/cifar10c/` (`<corruption>.npy`, `labels.npy`); `scripts/check_data.py` is the gate.

Stage 1 (seed stability, the kill switch): the same `setsid` launch with `--seed1` appended, or by hand
from the repo root (`train_second_seed.py` saves only after the last epoch, so keep it under setsid/tmux):

```bash
cd /workspace/Atlas
python scripts/train_second_seed.py --volume /workspace --seed 1 --out /workspace/models/resnet20_seed1.pt
python -m atlas.run --manifest experiments/queue/atlas_v0_resnet20_seed1.yaml --volume /workspace \
       --weights /workspace/models/resnet20_seed1.pt
python -m atlas.compare --a results/atlas_v0_resnet20_cifar10 --b results/atlas_v0_resnet20_seed1
python -m atlas.critic  --results results/atlas_v0_resnet20_cifar10 results/atlas_v0_resnet20_seed1 \
       --tol experiments/tolerances_default.yaml --out results/critic_resnet20_s0_s1
```

Stage 2 (scale): `experiments/queue/atlas_v0_resnet56_cifar10.yaml`, then `compare --a resnet20 --b resnet56`.

## What is measured (v0)

| module | invariant | reads | records |
|---|---|---|---|
| dimension | `pca_spectrum` | ref | participation ratio, entropy rank, dim95/99 |
| dimension | `twonn_id` | ref | TwoNN intrinsic dimension (+ per-class) |
| landmarks | `class_centers` | ref, test | centers, RMS radii, sep ratio, nearest-center acc |
| landmarks | `neural_collapse` | ref | NC1, simplex-ETF deviation, norm CV |
| landmarks | `hubness` | ref | k-occurrence skewness |
| adjacency | `class_adjacency` | ref, test | pairwise separation, valley ratio, NC confusion |
| adjacency | `merge_order` | ref | single-linkage merge sequence, H0-vs-scale, cophenetic |
| density | `knn_density` | ref, all splits | sparse fraction per split, log-radius shift |
| decodability | `linear_probes` | test, corrupt, factors | CV probe score per factor (18 factors) |
| sensitivity | `corruption_displacement` | test, corrupt | magnitude, coherence, class-subspace frac, PCA direction |
| flow (cross) | `layer_cka` | test | consecutive CKA, biggest reorganization |
| flow (cross) | `commit_layer` | probes | first layer at tau·best, peak, washout per factor |

Factors (`atlas/factors/`): 13 pixel factors (luminance, contrast, high-freq ratio, spectral
slope, spectral anisotropy, noise sigma, saturation, hue cos/sin, colorfulness, edge density,
orientation entropy, blockiness) + 5 meta factors (class, coarse animal/vehicle, corruption
family, corruption type, severity). `spectral_anisotropy`, `luminance_mean`, `highfreq_ratio`
are the row-6 (motion / brightness / defocus) targets.

## Adding a measurement (the whole point of the modular layout)

```python
# atlas/invariants/my_thing.py
from ..registry import invariant

@invariant("my_thing", needs=("ref", "labels"), cost="medium")
def my_thing(ctx, cfg):
    """One line: what it measures and what would falsify it."""
    ...
    return {"scalar": 1.23, "matrix": M.tolist()}
```

then add `from . import my_thing` in `atlas/invariants/__init__.py`. It runs on every layer
of every future manifest (or list it under `invariants:`). Cross-layer measurements use
`@cross_layer`; input-side factors use `@factor(...)` in `atlas/factors/`. Meta factors get
`source="meta"` and a `fn(split, labels)` signature. Per-invariant options go under
`invariant_cfg:` in the manifest and arrive as `cfg`.

To have `compare` / `critic` track a new scalar, append `("my_thing", "scalar")` to
`SCALARS` in `atlas/compare.py`.

`ctx` (`atlas/context.py`) gives: `ref`, `ref_labels`, `test`, `test_labels`, `test_preds`,
`panel`, `corrupt[split]`, `corrupt_labels[split]`, `ood[split]`, `factors[split][name]`,
`n_classes`, `rng`, `source`. Declare what you read in `needs=` so the builder can skip
cleanly on dumps that lack it.

## Outputs

`atlas.json` is the machine-readable atlas; `ATLAS.md` is the human view (numbers only, no
interpretation); `plots/` has the dimension profile, decodability heatmap, sensitivity field
and H0-vs-scale. `compare` writes `deformation.json` + `DEFORMATION.md`; `critic` writes
`critic.json` + `CRITIC.md` with a verdict in {REPLICATES, PARTIAL, DOES_NOT_REPLICATE,
PIPELINE_CHECK_ONLY}.

## Pairing contract (do not break)

Clean test = torchvision CIFAR-10 test indices `0..n_test-1` in order. Each corrupt split =
rows `0..n_per_set-1` of that severity block in the CIFAR-10-C `.npy`. `extract_acts`
asserts label equality. `corruption_displacement` and the corrupt-pool probes rely on it.

## Synthetic dumps

`python -m atlas.synth --out /tmp/synth` produces a dump with the real layout from a random
readout of known latents. Use it to develop invariants on CPU. `meta.source = synthetic`;
the critic marks every result from it non-promotable, by design.

## Deformation ladder (rung 1: TENT)

`scripts/tta_deform.py` adapts the backbone with TENT on a CIFAR-10-C stream and re-dumps the
full atlas input at checkpoints (`results/<exp>/dump_step<k>/`), with offline ground truth in
each dump's `meta.json` (held-out stream accuracy, clean accuracy, held-out corruption
accuracy, predicted-class histogram entropy). `dump_mode: affine` measures the adapted BN
affine parameters against the original normalization (the map-moves regime, isolated);
`deployed` measures what TENT actually serves (batch statistics included).

```bash
python scripts/tta_deform.py --manifest experiments/queue/tta_tent_resnet20_fog3.yaml --volume /workspace
python -m atlas.ladder --exp results/tta_tent_resnet20_fog3           # LADDER.md, ladder.json, plots/ladder.png
python scripts/tta_deform.py --manifest experiments/queue/tta_tent_resnet20_collapse.yaml --volume /workspace
python -m atlas.ladder --exp results/tta_tent_resnet20_collapse --tol 0.05
```

`LADDER.md` has one row per checkpoint (ground truth next to every label-free deformation
metric from `compare --same-space`) and an onset table: the first step each metric leaves its
step-0 value, against the first step the histogram monitor collapses. The bar for the immune
claim is that column reading "yes" for topology metrics (merge_tau, adjacency_rho, d_nc1)
on the collapse run, and staying "no" (no false onset) on the benign run.

Rungs 2+ (EATA/CoTTA held-out, partial and full fine-tuning) only need a different adapter
loop writing the same `dump_step<k>/` layout; `atlas.ladder` is unchanged.
