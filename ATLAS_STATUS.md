# ATLAS_STATUS — promoted map entries

Legend: ✅ replicated on real data (≥2 seeds, holdout declared, prediction pre-registered) ·
🟡 measured on real data, not (yet) promoted · ⬜ proposed / synthetic-only · ✗ tested, does not replicate

Rows 1-6 were restated from s0 (atlas_v0, discovery) and frozen as C1-C5 before Stage 1. Stage 1 (v1:
hub under its training normalization + locally trained seeds s1, s2) ended **PARTIAL** (all core items
passed, but some non-core scalar spreads were not anticipated; results/atlas_v1_resnet20_s1/SESSION.md),
so nothing is promoted to ✅ yet: ✅ is decided on seeds 3 and 4 under an amended rule committed first.

| # | layer(s) | entry | claim (current) | status | evidence |
|---|---|---|---|---|---|
| 1 | all | twonn_id profile | ID peaks at layer3.0/3.1 and penult is ≥ 30% below the peak (C1); only the last-block drop is learned (random init also peaks at layer3.1 but drops 0.02) | 🟡 replicated s1, s2 | v1 s1/s2 drop 0.498 / 0.488 · v0 19.46 → 9.87 |
| 2 | penult | class_centers.sep_ratio | nearest-other / RMS on the train reference ≈ 3.0 under v1; not comparable to the historical 2.87 (different formula and split) | 🟡 seed-stable | v1 hub 2.996, s1 3.067, s2 3.021 · v0 2.941 |
| 3 | penult | linear_probes washout | luminance washout > 0.30; highfreq and anisotropy < 0.20 (C2); cite penult luminance, not the stem R² (architecture artifact) | 🟡 replicated s1, s2 | v1 s1 0.736/0.173/0.062, s2 0.682/0.119/0.050 |
| 4 | layer3.x | commit_layer class | commits in {layer3.0, layer3.1, layer3.2 (= penult)} at tau 0.9 (C3) | 🟡 replicated s1, s2 | v1 layer3.1 in hub, s1, s2 |
| 5 | penult | corruption_displacement coherence | defocus s5, motion s3/s5 have a smaller mean shift per unit per-sample shift than gaussian and shot noise at the same severity (C4) | 🟡 replicated s1, s2 | v1 6/6 in both, min margin 0.115 |
| 6a | penult | knn_density sparse_frac | s5 > s1 for the 8 non-noise discovery corruptions (C5) | 🟡 replicated s1, s2 | v1 8/8 in both |
| 6b | penult | knn_density sparse_frac | gaussian and shot noise are non-monotone (s5 < s3) | ✗ | failed in s1 (gaussian) and s2 (both); v0 original "monotone for every corruption" also ✗ |
| 7 | penult | class adjacency (merge order non-core) | penult adjacency is seed-stable: distance-only rho 0.950 s1-s2 (null 0.644); merge tau 0.838 | 🟡 | v1 critic s1-s2 rho 0.973 (partly per-class radius ranking); hub pairs 0.939 / 0.955; train-reference caveat |
| 8 | penult | relrep agreement | exploratory: CIFAR-100 relrep argmax agreement across seeds ≥ 3× chance; panel argmax is trivial (train images) | 🟡 | v1 s1-s2 0.589 vs chance 0.126 (4.67×; null 1.36×) |
