# PROGRESS.md — status summary

Current state at a glance. Detailed chronological record with exact commands and full tables:
`PROGRESS_LOG.md`. Background, decisions and literature: `PROJECT_CONTEXT.md`.
All numbers are Market-1501, ViT-S/16 (ImageNet-21k->1k augreg weights), 256x128, 120 epochs, unless stated.

_Last updated: 2026-11-14 ~23:40 local._

## Completed
- [x] Environment: Windows 11, `.venv` with torch 2.15.0+cu132 on RTX 5080 (`requirements.txt`).
- [x] Kit verified on GPU (synthetic + real data). Fixed: reduced-token paths now use fused SDPA
      (previously slower than full tokens). Throughput timed at batch 256.
- [x] Market-1501 downloaded from the official source (outside repo: `C:/Users/user/data/`).
- [x] Iteration-1 runs, seed 0: baseline, TBSD, 25%-ID baseline, 25%-ID TBSD, each with a token sweep.
- [x] Seeds 1 and 2: baseline and TBSD (+ sweeps). 3-seed aggregate: `aggregate.py` ->
      `runs/iter1_seeds.csv`, `runs/iter1_seeds.png`.
- [x] Git repository with checkpoints of code, metrics, plots and docs.

## Experiment results (verified runs)

### Headline: 3 seeds, mean ± std, token merging, mAP (%)
| Merge keep | GFLOPs | Baseline, full-token gallery | TBSD, full-token gallery | Baseline, reduced gallery | TBSD, reduced gallery |
| --- | --- | --- | --- | --- | --- |
| 1.0 (full) | 2.93 | 84.45 ± 0.49 | 83.95 ± 0.13 | 84.45 ± 0.49 | 83.95 ± 0.13 |
| 0.8 | 2.22 | 83.70 ± 0.49 | 83.89 ± 0.13 | 83.08 ± 0.55 | 83.84 ± 0.15 |
| 0.7 | 1.92 | 82.50 ± 0.55 | 83.70 ± 0.12 | 81.04 ± 0.65 | 83.48 ± 0.12 |
| 0.6 | 1.68 | 79.99 ± 0.75 | 83.29 ± 0.13 | 76.79 ± 0.92 | 82.71 ± 0.06 |
| 0.5 | 1.46 | 68.56 ± 1.14 | **81.98 ± 0.18** | 59.60 ± 1.12 | **80.26 ± 0.13** |

Training: baseline 0.22 GPU-h, TBSD 0.41 GPU-h (mean of 3). TBSD's full-token mAP is ~0.5 lower than
baseline, lower in all 3 seeds (-0.1, -0.6, -0.8): a small, probably real, cost. Its gain at keep 0.5
(+13.4 full-token gallery, +20.7 reduced gallery) is far outside the seed spread.

### Per run
| Run | Seed | Full-token mAP / R1 | Merge keep 0.5, full-token gallery mAP | Train GPU-h |
| --- | --- | --- | --- | --- |
| baseline | 0 | 83.9 / 93.5 | 67.4 | 0.22 |
| TBSD (merge, random keep) | 0 | 83.8 / 93.9 | 81.9 | 0.40 |
| baseline | 1 | 84.7 / 94.2 | see `runs/base_s1/sweep.csv` | 0.22 |
| TBSD | 1 | 84.1 / 93.4 | see `runs/tbsd_s1/sweep.csv` | 0.45 |
| baseline | 2 | 84.8 / 94.2 | 69.7 | 0.23 |
| TBSD | 2 | 84.0 / 93.6 | see `runs/tbsd_s2/sweep.csv` | 0.39 |
| baseline, 25% IDs | 0 | 71.6 / 87.2 | 51.2 | 0.06 |
| TBSD, 25% IDs | 0 | 70.0 / 86.1 | 66.4 | 0.10 |

Hypotheses: H1 reframed (asymmetric > symmetric; the real issue is query drift, which TBSD reduces);
H2 untested; H3 not supported (seed 0); H4 untested. Figures: `runs/iter1_summary.png`, `runs/iter1_seeds.png`.

## In progress
- [ ] `run_iter1b.sh` controls (expected end ~00:50): equal-GPU-hour baseline (216 epochs, started 23:38),
      then TBSD without distillation (`--sd-weight 0`). Re-run `aggregate.py` after; commit.

## Remaining tasks
1. Investigate class-token-attention selection doing worse than random (compare official EViT with fused token).
2. Fix/ablate BNNeck running-stat mixing in TBSD.
3. Fixed-budget training control (`--train-mode merge --train-keep 0.5`).
4. Second dataset (MSMT17/CUHK03 loader, download approval) for H2; low-data at 10/50%, 3 seeds (H3).
5. Latency at batch 1 and CPU (`evaluate.py --cpu`); OSNet and ViT-B references.
6. Literature matrix to 30–40 full reads; full novelty check; one-page proposal.
7. Update the cloud iteration-1 report with results.

## Known issues
- TBSD full-token mAP is ~0.5 below baseline, consistently over 3 seeds (-0.1, -0.6, -0.8). Candidate cause: BNNeck stat mixing.
- TBSD's reduced pass updates BNNeck running stats (mixed statistics at eval).
- TBSD costs ~1.8–2x training time; fairness controls pending.
- `merge`/`cls_attn` are own implementations, not official ToMe/EViT; GFLOPs are analytic estimates.
- At batch 64 a ViT-S on the 5080 is overhead-bound; token reduction gives no wall-clock gain there.
- Only Market-1501-format loaders exist.
