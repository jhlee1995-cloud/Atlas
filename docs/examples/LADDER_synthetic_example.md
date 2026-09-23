# LADDER  /tmp/ladder_demo  (layer penult, tol 0.02)

ground truth uses labels offline; every other column is label-free map deformation vs step 0

| step | acc_stream_heldout | acc_clean | acc_heldout_corr | pred_entropy | panel_cka | panel_shift | center_disp | adjacency_rho | merge_tau | decod_abs_delta | d_twonn_id | d_sep_ratio | d_nc1 | d_etf_dev | panel_cka_min_layer | panel_cka_argmin_layer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.700 | 0.900 | 0.600 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | · | · |
| 5 | 0.680 | 0.860 | 0.580 | 0.940 | 1.000 | 0.097 | 0.094 | 0.998 | 1.000 | 0.000 | 0.015 | 0.009 | -0.000 | 0.000 | 1.000 | stem |
| 10 | 0.650 | 0.800 | 0.550 | 0.850 | 1.000 | 0.237 | 0.232 | 0.995 | 0.958 | 0.000 | 0.142 | 0.008 | 0.000 | 0.001 | 0.998 | stem |
| 50 | 0.500 | 0.500 | 0.400 | 0.400 | 0.997 | 0.930 | 0.908 | 0.976 | 0.698 | 0.000 | 0.063 | -0.024 | 0.001 | 0.006 | 0.976 | stem |
| 100 | 0.350 | 0.200 | 0.250 | -0.050 | 0.985 | 1.575 | 1.539 | 0.905 | 0.760 | 0.001 | -0.382 | -0.019 | 0.001 | 0.009 | 0.960 | stem |

## onset (first step the quantity leaves its step-0 value by more than tol)

| quantity | onset step | earlier than histogram collapse? |
|---|---|---|
| collapse(pred_entropy) | 50 |  |
| panel_cka | None | no |
| panel_shift | 5 | yes |
| center_disp | 5 | yes |
| adjacency_rho | 50 | no |
| merge_tau | 10 | yes |
| decod_abs_delta | None | no |
| d_twonn_id | 10 | yes |
| d_sep_ratio | 50 | no |
| d_nc1 | None | no |
| d_etf_dev | None | no |
| panel_cka_min_layer | 50 | no |