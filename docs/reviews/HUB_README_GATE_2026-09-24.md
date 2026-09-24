# Why three chenyaofo hub units failed the batch-4 README gate (INFO, 2026-09-24)

**Status.** A post-hoc explanation, recorded as INFO. The owner asked for it during S1. No batch-4 claim depends on it.
The B4a amendment (`docs/plans/B4A_AMENDMENT.md`) was approved before this analysis and does not rely on it.

## What happened

S1's frozen weights gate (`scripts/b4_weights.py`) requires |full-10k accuracy − README top-1| ≤ 0.003. Three hub units
failed, all measuring **above** their README value:

| unit | delta |
|---|---|
| shufflenetv2_x0_5 | +0.0052 |
| mobilenetv2_x0_75 | +0.0036 |
| shufflenetv2_x1_0 | +0.0032 |

Non-sealed units, for comparison:

| unit | delta |
|---|---|
| resnet20 hub | 0 |
| resnet56 hub | +0.0001 |
| resnet32 | 0 |
| vgg11_bn | −0.0001 |
| mobilenetv2_x0_5 | +0.0024 |
| repvgg_a0 | +0.0008 |

## Cause

Sources: the public upstream training logs, `logs/cifar10/<model>/default.log` on the `logs` branch of
chenyaofo/pytorch-cifar-models, and the upstream training code chenyaofo/image-classification-codebase.

1. **README is the best-epoch accuracy.** In all 19 logs the README top-1 and top-5 equal the best-epoch EVAL row
   exactly. The first hypothesis was "README = last epoch, weights = best epoch". It is refuted: it flags none of the 3
   failures and 5 units that passed.
2. **Two groups of runs.**
   - **Group A:** 8 runs, resnet* and vgg*, logged 2021-04-10.
   - **Group B:** 11 runs, every mobilenetv2, shufflenetv2 and repvgg, logged 2021-04-14/15. Their logged config
     carries `is_vit` and `sync_batchnorm` keys, which exist only in the later code.
3. **The upstream bug.**
   - From commit `19f1077e` (2021-04-11), `codebase/data/cifar.py` sets `val_transforms = get_train_transforms(...)`,
     that is RandomCrop(32, padding 4) plus horizontal flip.
   - Commit `81e70a97` (2021-11-30, "Fix the bug with wrong transformation in cifar") fixed it.
   - The earlier commit used by group A evaluates on clean images.
   - Every group-B run therefore scored its test set under training augmentation.
   - Group B's README value is the maximum over 200 noisy augmented evaluations. The released weights are that epoch's
     checkpoint. This is inferred from upload times, which follow each run's end.
4. **Size of the gap.**
   - Atlas measures the clean accuracy of the same checkpoint. So measured − README ≈ (clean − augmented accuracy) −
     (the luck of selecting the maximum epoch).
   - The first term is largest for the least-fit, smallest nets. shufflenetv2_x0_5 is the only one of the 19 below 99.66%
     final train top-1 (98.44%).
   - Estimated clean − augmented gaps are 0.5-0.77 pt for the failing and near-failing mobile units and about 0.26 pt
     for repvgg_a0.
   - Group A's measured accuracies match their logs within ±1 image.
5. **Rivals ruled out:**
   - TF32 numerics, since group A is exact;
   - AMP: upstream `use_amp` was false;
   - sampler padding or `drop_last`: all 10,000 images were evaluated every epoch;
   - EMA: none;
   - normalisation or architecture errors, which would lower accuracy.

## Why ShuffleNet showed up twice

- **Not an architecture effect.** Late-epoch noise and the gap relative to error rate are the same as MobileNetV2's.
- **shufflenetv2_x0_5 is a near-certain fail.** It is the least-fit model, with the largest gap. On a fresh draw of the
  augmentation noise it fails with probability about 0.93-0.96.
- **shufflenetv2_x1_0 missed the bar by 2 images.** That is about a coin flip: mobilenetv2_x0_5, with a similar gap,
  passed.

## Consequences

- **B4a (one-sided gate) is the right fix.** A gate against the log's best epoch gives the same 16 PASS / 3 FAIL as the
  README gate. A gate against the last epoch would fail at least 5 units.
- **The group-B lower-side check is weaker.** A loading error is caught only if it costs more than delta + 0.003. Real
  loading errors (wrong weights, normalisation, architecture) cost well over 1 pt, so they are still caught.
- **Batch 5.** Record `upstream_val` (clean or augmented) and `selected_on` per hub unit in the registry, classified from
  each log's config. Use a two-sided gate for group A and a one-sided gate for group B. Read the CIFAR-100 log configs
  before using any CIFAR-100 hub unit.
- **Test-set selection.** Every hub checkpoint was selected on the full CIFAR-10 test set, which contains every Atlas
  row. For group A the optimism of choosing the best epoch over the last is 4-17 images on the resnets. This was already
  noted in `docs/plans/STAGE2.md:14-17`.

This analysis used only public upstream files, non-sealed unit values, the two FAIL deltas and PASS/FAIL status. It made
no inference about any sealed unit's accuracy.
