# ATLAS_STATUS — promoted map entries

Legend: ✅ replicated on real data (≥2 seeds, holdout declared, prediction pre-registered) ·
🟡 measured once on real data · ⬜ proposed / synthetic-only · ✗ tested, does not replicate

| # | layer(s) | entry | claim | status | evidence |
|---|---|---|---|---|---|
| 1 | all | twonn_id profile | hunchback shape across depth | ⬜ | (P1, atlas_v0) |
| 2 | penult | class_centers.sep_ratio | ≈ 2.9 (prior session figure) | ⬜ | (P4, atlas_v0) |
| 3 | stem..layer1 | linear_probes luminance/highfreq/anisotropy | decodable early, washout > 0.3 at penult | ⬜ | (P2, atlas_v0) |
| 4 | layer2+ | commit_layer class | commits at layer2.x or later | ⬜ | (P2, atlas_v0) |
| 5 | penult | corruption_displacement | brightness/defocus/motion lower coherence than noise/contrast | ⬜ | (P3, atlas_v0) |
| 6 | penult | knn_density sparse_frac | monotone in severity for every corruption | ⬜ | (P5, atlas_v0) |
| 7 | all | adjacency + merge_order | replicate across seeds (rho ≥ 0.7, tau ≥ 0.6) | ⬜ | (stage 1) |
| 8 | penult | relrep agreement | landmark-relative coordinates agree across seeds (argmax ≥ 0.8) | ⬜ | (stage 1, moon-shot a) |
