# ATLAS_STATUS — promoted map entries

Legend: ✅ replicated on real data (≥2 seeds, holdout declared, prediction pre-registered) ·
🟡 measured once on real data · ⬜ proposed / synthetic-only · ✗ tested, does not replicate

Rows 1-6 were restated from what s0 showed (atlas_v0, discovery) and frozen as C1-C5 in
`experiments/queue/atlas_v1_resnet20_s1.yaml` before Stage 1; the original pre-registered wording and
its verdict are kept in the last column. v0 was measured under a mismatched input normalization
(SESSION.md O3); v1 re-measures under the hub's training normalization.

| # | layer(s) | entry | claim (current) | status | evidence · original claim → verdict |
|---|---|---|---|---|---|
| 1 | all | twonn_id profile | ID peaks at layer3.0/3.1 and penult is ≥ 30% below the peak (C1) | 🟡 | atlas_v0: 19.46 at layer3.1 → 9.87 · "hunchback, peak in layer2" → PARTIAL |
| 2 | penult | class_centers.sep_ratio | nearest-other / RMS on train ref ≈ 2.94; not comparable to the historical 2.87 (different formula and split) | 🟡 | atlas_v0 · "≈ 2.9" → literal PASS, non-diagnostic |
| 3 | penult | linear_probes washout | luminance washout > 0.30; highfreq and anisotropy < 0.20 (row-6 factors stay decodable) (C2) | 🟡 | atlas_v0: 0.594 / 0.132 / 0.053 · "all three wash out > 0.3" → PARTIAL |
| 4 | layer3.x | commit_layer class | commits in {layer3.0, layer3.1, layer3.2 (= penult)} at tau 0.9 (C3) | 🟡 | atlas_v0: layer3.1 · "layer2.x or later" → PASS |
| 5 | penult | corruption_displacement coherence | defocus s5, motion s3/s5 below gaussian and shot noise at the same severity (C4; weak: ≈ mean-shift size) | 🟡 | atlas_v0 · "brightness/defocus/motion below noise/contrast, lower class_sub_frac" → PARTIAL |
| 6 | penult | knn_density sparse_frac | gaussian and shot noise are non-monotone (s5 < s3); other discovery corruptions s5 > s1 (C5) | 🟡 | atlas_v0 · "monotone for every corruption" → FAIL (✗ for the original claim) |
| 7 | penult | class adjacency (+ merge order, non-core) | penult adjacency rho ≥ 0.70 across seeds; merge tau ≥ 0.60 reported | ⬜ | stage 1 (v1) |
| 8 | penult | relrep agreement | exploratory: OOD (CIFAR-100) relrep argmax agreement ≥ 3x chance across seeds; panel argmax is trivial (train images) | ⬜ | stage 1 (v1), S10 |
