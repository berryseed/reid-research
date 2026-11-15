# PROGRESS.md — status summary

Current state at a glance. Detailed chronological record with exact commands and full tables:
`PROGRESS_LOG.md`. Background, decisions and literature: `PROJECT_CONTEXT.md`.
All numbers are Market-1501, ViT-S/16 (ImageNet-21k->1k augreg weights), 256x128, 120 epochs, unless stated.

_Last updated: 2026-11-15 ~15:15 local (session 2, after power-outage recovery)._

## Current status: iteration 2 running (queue `run_iter2.sh`, log `runs/queue_iter2.txt`)
- **Power outage 2026-11-15 ~12:33** during `tbsd_bnns_s0` (epoch 95 of 120). Recovery at 14:59:
  - Validated: every `last.pth` (16 files) loads with finite weights; every `log.json` parses; every sweep CSV is
    complete; every PNG verifies; `git fsck` clean; local `main` equal to `origin/main`; no `.tmp` leftovers.
  - Latest valid checkpoint: `runs/tbsd_bnns_s0/ckpt.pth`, epoch 94, step 18988, 0.314 h trained, with model,
    optimizer (152 momentum buffers), AMP scaler, LR-schedule step and all RNG states. Resumed from it at 15:02
    (`resumed from ... epoch 94 done`), not restarted.
  - **Lost to the outage:** the unfinished part of epoch 95 of that run (under 12 s of training) and about 2.5 h of
    idle wall-clock. No completed result was lost. Finished-but-uncommitted results (selection sweeps, latency,
    `tbsd_nosd_s1/s2`) were intact on disk and are now pushed.
  - Resume on GPU is statistically equivalent, not bit-exact (see Known issues); `tbsd_bnns_s0` is a resumed run.
- Protection added: `run_iter2.sh` commits and pushes each run's metrics as soon as its sweep finishes;
  `train.py` already writes an atomic resumable `ckpt.pth` every epoch (about 12 s).
- Phase status: S (selection diagnostics, latency) done; A (no-distillation seeds done; BNNeck ablation running);
  B (fixed-budget control, equal-GPU-h seeds), C (low data 10/25/50%), D (ViT-B) pending.
- If interrupted again: `bash run_iter2.sh /c/Users/user/data/Market-1501-v15.09.15 A B C D >> runs/queue_iter2.txt 2>&1`
  (finished runs are skipped, the interrupted one resumes from its `ckpt.pth`).
- Final checkpoints (not in Git) are `runs/<name>/last.pth` on the training PC. Dataset: `C:/Users/user/data/Market-1501-v15.09.15`.

### New results so far in iteration 2 (verified runs; full tables in PROGRESS_LOG.md)
- **Distillation term, 3 seeds:** multi-budget training without distillation 80.56 ± 0.06 vs TBSD 81.98 ± 0.18 mAP at
  keep 0.5 (full-token gallery); 78.31 ± 0.12 vs 80.26 ± 0.13 with a reduced gallery; q_cos 0.902 vs 0.944.
  The distillation term adds about +1.4 / +1.9 mAP at the lowest budget, outside seed noise. At full tokens the
  difference (+0.24) is not outside noise. Multi-budget training itself still accounts for most of the gain
  (+12.0 of the +13.4 over the baseline).
- **Selection diagnostics, 3 seeds, baseline checkpoints, keep 0.7, full-token gallery mAP:** random 77.99 ± 0.61,
  class-attention 72.66 ± 3.29, class-attention + EViT fused token 73.82 ± 2.99, least-attended tokens 37.62 ± 6.26.
  Class attention does carry identity information (inverting it is far worse), the fused token helps by about 1 mAP,
  and class-attention selection is still below random for a full-token gallery, with much larger seed variance.
  With query and gallery both reduced, class attention and random are within noise at keep 0.8 and 0.7.
- **Latency (forward only, random input, ViT-S, `runs/latency.csv`):** merge keep 0.5 vs full tokens:
  GPU batch 256 0.062 vs 0.111 ms/img (1.78x); CPU batch 1 11.9 vs 16.3 ms (1.37x); CPU batch 8 6.2 vs 11.1 ms (1.78x);
  **GPU batch 1 is slower with merging (8.5 vs 6.0 ms)**: the matching step dominates at that size.

## How to resume the project
1. Open the folder in Claude Code; read `CLAUDE.md`, then this file, then the end of `PROGRESS_LOG.md`.
2. Environment: `.venv` (already installed). Recreate if needed: see `requirements.txt`.
3. Re-check finished state (skips everything already done, starts nothing new):
   `bash run_iter1b.sh /c/Users/user/data/Market-1501-v15.09.15`
4. Re-make tables/figure: `PYTHONUTF8=1 .venv/Scripts/python.exe aggregate.py`
5. Pick the next task from "Remaining tasks" below.

Pausing/resuming future runs (new in `train.py`, verified bit-exact on CPU):
- Pause a running run gracefully: `touch runs/<name>/STOP` -> stops after the current epoch, saves
  `runs/<name>/ckpt.pth` (model, optimizer, AMP scaler, epoch, step/LR schedule, RNG states, log, args), exit code 3.
- Resume: rerun the same queue script (it passes `--resume`), or the same `train.py` command plus `--resume`, e.g.
  `.venv/Scripts/python.exe train.py --data /c/Users/user/data/Market-1501-v15.09.15 --out runs/<name> --seed <s> <same flags> --resume`

## Completed
- [x] Environment: Windows 11, `.venv` with torch 2.15.0+cu132 on RTX 5080 (`requirements.txt`).
- [x] Kit verified on GPU (synthetic + real data). Fixed: reduced-token paths now use fused SDPA
      (previously slower than full tokens). Throughput timed at batch 256.
- [x] Market-1501 downloaded from the official source (outside repo: `C:/Users/user/data/`).
- [x] Seed 0: baseline, TBSD, 25%-ID baseline, 25%-ID TBSD, each with a token sweep.
- [x] Seeds 1 and 2: baseline and TBSD (+ sweeps). 3-seed aggregate: `aggregate.py` ->
      `runs/iter1_seeds.csv`, `runs/iter1_seeds.png`.
- [x] Controls (seed 0): equal-GPU-hour baseline (216 epochs); TBSD without the distillation term (`--sd-weight 0`).
- [x] Resumable checkpoints + graceful pause (STOP file) in `train.py`; queue scripts resume automatically.
- [x] Git + private GitHub repo https://github.com/berryseed/reid-research.

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

### Controls (seed 0 only), merge, mAP full-token gallery / reduced gallery
| Run | GPU-h | keep 1.0 | keep 0.7 | keep 0.6 | keep 0.5 |
| --- | --- | --- | --- | --- | --- |
| Baseline, 216 epochs (equal GPU-h) | 0.36 | 85.28 | 83.55 / 82.19 | 81.25 / 78.51 | 69.02 / 59.40 |
| TBSD without distillation (`--sd-weight 0`) | 0.41 | 83.63 | 83.13 / 82.78 | 82.48 / 81.67 | 80.62 / 78.37 |
| TBSD (3-seed mean, for reference) | 0.41 | 83.95 | 83.70 / 83.48 | 83.29 / 82.71 | 81.98 / 80.26 |

Reading (one seed for controls, needs more seeds):
- Longer training gives +1.3 full-token mAP but no budget robustness (69.0 at keep 0.5).
- **Most of TBSD's gain comes from multi-budget training itself** (reduced-token pass with its own ID +
  triplet loss): 80.6 vs 68.6 at keep 0.5. The distillation term adds ~+1.4 (full-token gallery) / +1.9
  (reduced gallery) and raises query cosine to the full-token embedding (0.944 vs 0.903). This matters for the
  novelty claim: the contribution must be framed around budget-robust retrieval embeddings, and the role of
  distillation needs more seeds before claiming it helps.

### Per run
| Run | Seed | Full-token mAP / R1 | Merge keep 0.5, full-token gallery mAP | Train GPU-h |
| --- | --- | --- | --- | --- |
| baseline | 0 | 83.9 / 93.5 | 67.4 | 0.22 |
| TBSD (merge, random keep) | 0 | 83.8 / 93.9 | 81.9 | 0.40 |
| baseline | 1 | 84.7 / 94.2 | see `runs/base_s1/sweep.csv` | 0.22 |
| TBSD | 1 | 84.1 / 93.4 | see `runs/tbsd_s1/sweep.csv` | 0.45 |
| baseline | 2 | 84.8 / 94.2 | 69.7 | 0.23 |
| TBSD | 2 | 84.0 / 93.6 | see `runs/tbsd_s2/sweep.csv` | 0.39 |
| baseline, 216 epochs | 0 | 85.3 / 94.5 | 69.0 | 0.36 |
| TBSD without distillation | 0 | 83.6 / 93.8 | 80.6 | 0.41 |
| baseline, 25% IDs | 0 | 71.6 / 87.2 | 51.2 | 0.06 |
| TBSD, 25% IDs | 0 | 70.0 / 86.1 | 66.4 | 0.10 |

Hypotheses: H1 reframed (asymmetric > symmetric; the real issue is query drift, which TBSD reduces);
H2 untested; H3 not supported (seed 0); H4 untested. Figures: `runs/iter1_summary.png`, `runs/iter1_seeds.png`.

## Remaining tasks (suggested order)
1. Seeds 1–2 for the no-distillation control (decides whether distillation is a real contribution).
2. Fix/ablate BNNeck running-stat mixing in TBSD (candidate cause of the -0.5 full-token cost).
3. Investigate class-token-attention selection doing worse than random (official EViT with fused token).
4. Fixed-budget training control (`--train-mode merge --train-keep 0.5`).
5. Second dataset (MSMT17/CUHK03 loader, needs download approval) for H2; low-data at 10/50%, 3 seeds (H3).
6. Latency at batch 1 and on CPU (`evaluate.py --cpu`); OSNet and ViT-B references.
7. Literature matrix to 30–40 full reads; full novelty check (incl. multi-budget/elastic token training); proposal.
8. Update the cloud iteration-1 report with these results.

## Known issues
- TBSD full-token mAP is ~0.5 below baseline, consistently over 3 seeds (-0.1, -0.6, -0.8). Candidate cause: BNNeck stat mixing.
- TBSD's reduced pass updates BNNeck running stats (mixed statistics at eval).
- TBSD costs ~1.85x training time. Equal-GPU-h baseline (seed 0): +1.3 full-token mAP over TBSD, but 69.0 vs 82.0 at keep 0.5.
- Distillation term's own contribution is small (~+1.4 mAP at keep 0.5, one seed).
- `merge`/`cls_attn` are own implementations, not official ToMe/EViT; GFLOPs are analytic estimates.
- At batch 64 a ViT-S on the 5080 is overhead-bound; token reduction gives no wall-clock gain there.
- Resume is bit-exact on CPU with `--workers 0`; on GPU / with workers it is statistically equivalent only.
- Only Market-1501-format loaders exist. Iteration-1 runs predate resumable checkpoints.
