# B1 gate G0 diagnosis: which legacy sample does "valley sep 1.14" come from?

This page archives the diagnosis cited in `docs/plans/B1B_AMENDMENT.md`. It was run on pod `nc2mzssur3bb0k` on 2026-09-23 between 16:03 and 16:20 UTC, after B1's gate closed and before B1b was committed. It read only the Upgraded-Mod legacy caches on the network volume; no ViT data and no B1 result were read. The output below is transcribed verbatim from the session terminal. The pod is deleted. The caches stay on the volume (`/workspace/cache/imagenet/`), so the script can be re-run.

## Legacy caches on the volume

```
-rw-rw-rw-  1 root root  82515150 Jun 27 08:06 penult_10000.npz
-rw-rw-rw-  1 root root 246941070 Jun 27 08:09 penult_30000.npz
```

## Script

This is the Upgraded-Mod formula, `session_experiments/imagenet_extract.py:66-72` / `imagenet_valley_check.py:57-59`:
- centers are in-sample class means over the classes with at least `mn` samples;
- the ratio is the mean pairwise center distance divided by the mean per-class mean within-distance.

The two runs differ only in the loop over caches.

```python
import numpy as np, glob
def sep(F, lab, mn=3):
    cls=[c for c in np.unique(lab) if (lab==c).sum()>=mn]
    if len(cls)<2: return None, len(cls), 0
    C=np.array([F[lab==c].mean(0) for c in cls])
    within=np.array([np.linalg.norm(F[lab==c]-C[i],axis=1).mean() for i,c in enumerate(cls)])
    G=(C*C).sum(1); D=np.sqrt(np.maximum(G[:,None]+G[None]-2*C@C.T,0)); np.fill_diagonal(D,np.nan)
    return np.nanmean(D)/(within.mean()+1e-9), len(cls), sum((lab==c).sum() for c in cls)/len(cls)
d=np.load(PATH); F=d["F"].astype(np.float64); lab=d["lab"]
for mn in (3,10,20,30): print(PATH, "min/class", mn, *sep(F, lab, mn))
```

## Output

On `penult_10000.npz`:

```
/workspace/cache/imagenet/penult_10000.npz min/class 3 classes 999 avg n 10.1 sep 1.2681
/workspace/cache/imagenet/penult_10000.npz min/class 10 classes 551 avg n 18.2 sep 1.2230
/workspace/cache/imagenet/penult_10000.npz min/class 20 classes 2 avg n 5024.0 sep 1.3478
```

At min/class 20, the first script version's degenerate-class guard produced a wrong average and then crashed at min 30. That row is meaningless. The corrected function above adds the `len(cls) < 2` guard.

On `penult_30000.npz`, the corrected function:

```
n 30056
30000 cache: min/class 3 classes 1000 avg n 30.1 sep 1.1423
30000 cache: min/class 10 classes 1000 avg n 30.1 sep 1.1423
30000 cache: min/class 20 classes 997 avg n 30.1 sep 1.142
30000 cache: min/class 30 classes 569 avg n 32.4 sep 1.1315
```

## Reading

- **"1.14" is the 30k value.** The legacy "valley sep 1.14", and "flat 1.13-1.14 across 3→30 samples/class" in `imagenet_valley_check.py`, match the 30,056-row cache at about 30 images per class.
- **The 10k sample gives 1.2681.** That is the 10,048-row sample the legacy margin 0.800 came from, and the sample B1's legacy block reproduces bitwise (`results/b1_gate/gate.json` `b1_acc_info.legacy_cache.max_abs_dF` 0.0). Its separation is 1.2681, which equals B1's `legacy_imagenet.valley_sep_legacy`.
- **The difference is expected from sample size.** With about 10 images per class, center noise inflates the between-center distances.
- **This contradicts a line in MASTER_SUMMARY.md.** Its line 210 ("Valleys shallow at ImageNet scale ✅ ... not a sample artifact") holds for the 1.13-1.14 range at about 30 per class. It does not make 1.14 the value of the 10k sample.
