# Nonlinear (VPFA-inspired) map vs linear map vs sp_mixed

Source: `runs/nonlinear_map/nonlinear_map.csv`; sp_mixed from `runs/adaptive.csv` (fixed budgets). Mean ± sample std over 3 seeds.
Baseline = 216-epoch (equal-GPU-hour) baseline, frozen. All maps fitted on training images only, no labels.

| Query | GFLOPs | map | mAP | R1 | VAL loss | selected |
| --- | --- | --- | --- | --- | --- | --- |
| full | 2.93 | raw | 85.11 ± 0.15 | 94.08 ± 0.39 | — |  |
| lowres192x96 | 1.62 | raw | 82.06 ± 0.19 | 92.39 ± 0.55 | 0.15 ± 0.01 |  |
| lowres192x96 | 1.62 | ridge_report | 82.80 ± 0.23 | 92.79 ± 0.66 | — |  |
| lowres192x96 | 1.62 | ridge_cv | 82.96 ± 0.22 | 92.82 ± 0.62 | 0.12 ± 0.01 | lambda 0.1/0.1/0.1 |
| lowres192x96 | 1.62 | mlp | 82.77 ± 0.31 | 92.85 ± 0.51 | 0.12 ± 0.01 | h512_wd1e-5/h2048_wd1e-5/h2048_wd1e-5; epoch 10/20/20 |
| lowres192x96 | 1.62 | **sp_mixed (raw)** | 83.60 ± 0.14 | 93.59 ± 0.45 | — | — |
| lowres160x80 | 1.122 | raw | 75.72 ± 0.51 | 88.22 ± 0.29 | 0.34 ± 0.03 |  |
| lowres160x80 | 1.122 | ridge_report | 78.96 ± 0.18 | 90.44 ± 0.35 | — |  |
| lowres160x80 | 1.122 | ridge_cv | 79.16 ± 0.21 | 90.46 ± 0.41 | 0.27 ± 0.02 | lambda 0.1/0.1/0.1 |
| lowres160x80 | 1.122 | mlp | 78.61 ± 0.33 | 89.99 ± 0.32 | 0.26 ± 0.02 | h512_wd1e-5/h2048_wd1e-5/h512_wd1e-5; epoch 10/15/10 |
| lowres160x80 | 1.122 | **sp_mixed (raw)** | 83.01 ± 0.15 | 92.95 ± 0.27 | — | — |
| lowres128x64 | 0.72 | raw | 57.45 ± 2.34 | 70.67 ± 2.71 | 0.71 ± 0.06 |  |
| lowres128x64 | 0.72 | ridge_report | 68.88 ± 0.70 | 82.35 ± 1.13 | — |  |
| lowres128x64 | 0.72 | ridge_cv | 68.84 ± 0.69 | 82.23 ± 1.05 | 0.57 ± 0.03 | lambda 1/1/1 |
| lowres128x64 | 0.72 | mlp | 66.88 ± 0.22 | 80.38 ± 0.36 | 0.57 ± 0.04 | h2048_wd1e-5/h2048_wd1e-5/h2048_wd1e-5; epoch 15/15/10 |
| lowres128x64 | 0.72 | **sp_mixed (raw)** | 81.50 ± 0.22 | 92.09 ± 0.02 | — | — |

## Paired differences (per seed, then mean ± std)

| Query | mlp − ridge_report (mAP) | mlp − ridge_cv (mAP) | sp_mixed − best baseline map (mAP) | sp_mixed − best map (R1) |
| --- | --- | --- | --- | --- |
| lowres192x96 | -0.02 ± 0.09 | -0.19 ± 0.10 | 0.64 ± 0.35 | 0.69 ± 0.66 |
| lowres160x80 | -0.35 ± 0.15 | -0.55 ± 0.12 | 3.85 ± 0.30 | 2.45 ± 0.22 |
| lowres128x64 | -2.00 ± 0.69 | -1.96 ± 0.66 | 12.61 ± 0.71 | 9.74 ± 1.14 |

sp_mixed − best map pairs seed i of sp_mixed with seed i of the baseline (independent runs; the pairing is only a bookkeeping convenience). 'best map' = per-seed maximum over the three maps on the TEST set, an optimistic (oracle-selected) baseline.

## Cost of the map

| Query | backbone GFLOPs | MLP GMACs | MLP params | MLP fit time (s) | MLP GPU b1 (ms) | MLP GPU b256 (ms/img) | MLP CPU b1 (ms) | backbone GPU b1 / b256 / CPU b1 (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lowres192x96 | 1.62 | 0.00 ± 0.00 | 4220672.00 ± 2956624.58 | 18.07 ± 0.92 | 0.22 ± 0.01 | 0.00100 | 0.44 ± 0.27 | 6.16 / 0.058 / 13.02 |
| lowres160x80 | 1.122 | 0.00 ± 0.00 | 2513664.00 ± 2956624.58 | 17.87 ± 0.87 | 0.21 ± 0.00 | 0.00098 | 0.29 ± 0.26 | 6.16 / 0.042 / 10.60 |
| lowres128x64 | 0.72 | 0.01 ± 0.00 | 5927680.00 ± 0.00 | 18.60 ± 0.30 | 0.22 ± 0.00 | 0.00100 | 0.61 ± 0.01 | 6.16 / 0.029 / 8.87 |
