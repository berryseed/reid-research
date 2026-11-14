# CLAUDE.md — persistent instructions for this project

Low-resource person re-identification research (iteration 1 of a paper project).
Full background: `PROJECT_CONTEXT.md` (objectives, decisions, literature, gaps, method).
Current state and history: `PROGRESS_LOG.md` — **read it first and append to it at the end of every session.**

## Research rules (from the user and their instructor)
- Thesis: accuracy per unit of compute, labels and training time ("no money, no computation, no data").
  Every experiment and claim must be tied to cost (GFLOPs, latency, GPU-hours, peak VRAM, fraction of IDs).
- Novelty is mandatory; do not replicate papers. Proposed method: token-budget self-distillation (TBSD),
  see PROJECT_CONTEXT.md §6.1. "Token pruning + distillation for re-ID" alone is NOT novel.
- Baselines get the same compute/data budget as the proposed method (e.g. equal-GPU-hour baseline for TBSD).
- 3 seeds per configuration for paper numbers; gains inside seed noise are not results.
- Never present unverified numbers as verified. Keep "ran" separate from "wrote". Label estimates.
- Distil embeddings (BNNeck), not class logits.

## Working rules
- Do not delete existing files or recreate completed work.
- Downloading new datasets needs the user's explicit OK.
- Never interrupt, restart or modify a running training experiment (check `runs/queue_*.txt`, `nvidia-smi`).
- Keep `PROGRESS.md` (status summary: completed, results, remaining, known issues) and `PROGRESS_LOG.md`
  (chronological detail: exact commands, full tables) updated.

## Version control (standing instructions from the user, 2026-11-14)
- Git repo in this folder; private GitHub remote `origin`. The user authorised ongoing commits and pushes.
- Create a checkpoint commit (and push) **before** major code or architecture changes.
- Commit and push after every meaningful completed milestone (finished run batch, fix, new script, results).
- Descriptive commit messages: what changed and why. Verify changes work before committing.
- `.gitignore` excludes datasets, `*.pth` checkpoints, `.venv`, caches, secrets. Commit metrics (`log.json`,
  `sweep.csv`, `stdout.txt`), plots, configs and docs. Do not commit runs that are still training.
- Never force-push, rewrite history (amend/rebase/reset of pushed commits) or delete branches without approval.
- After every push, verify it succeeded (`git status -sb` shows no ahead/behind; `git log origin/main -1`).

## Environment (verified 2026-11-14)
- Windows 11 Pro, RTX 5080 16 GB, Ryzen 7 7800X3D, 16 GB RAM (RAM is the bottleneck: workers 4, drop to 2).
- Python 3.13. Project venv: `.venv` (torch 2.15.0+cu132, torchvision 0.29.2+cu132, timm, numpy, pillow, matplotlib).
  Run scripts with `.venv/Scripts/python.exe` (Bash) or `.venv\Scripts\python.exe` (PowerShell).
- Long runs (training) go in the background; log to `runs/<name>/stdout.txt`.

## Pause / resume (added 2026-11-15)
- `train.py` writes resumable `<out>/ckpt.pth` each epoch; `touch <out>/STOP` pauses at the next epoch
  boundary (exit code 3); resume = same command + `--resume`. Queue scripts pass `--resume` and stop on code 3.
  Details in README "Pausing and resuming training". Runs started before this change (all of iteration 1)
  had no mid-run checkpoints.
- To pause the project safely: STOP-file the running run (or let it finish), wait for the process to exit,
  commit/push metrics, confirm no python.exe from `.venv` is running.

## Key decisions and results so far (details: PROGRESS.md)
- Main backbone ViT-S/16 (open question to instructor: ViT-B?). Market-1501 at `C:/Users/user/data/`.
- TBSD (merge, random keep) is the method under test. 3 seeds: 81.98±0.18 vs 68.56±1.14 mAP at keep 0.5
  (full-token gallery); full-token cost ~0.5 mAP (TBSD lower in all seeds); ~1.85x training time.
- Equal-GPU-h baseline (216 ep) gains +1.3 full-token mAP but not budget robustness.
- No-distillation control (`--sd-weight 0`) reaches 80.6 at keep 0.5: most of the gain is multi-budget training;
  distillation adds ~+1.4 (one seed). Do not claim distillation as the source of the gain without more seeds.
- Project paused 2026-11-15 00:45 with all runs complete; resume instructions at the top of PROGRESS.md.
- H1 reframed (asymmetric > symmetric; problem is query drift). H3 not supported (one seed).
- cls_attn selection < random selection: unexplained, verify against official EViT before claiming.
- Throughput must be timed at batch >= 256 on GPU (batch 64 is overhead-bound).

## Code layout
- `common.py` data parsing (Market-1501 names), PK sampler, `ReIDViT` + token reduction
  (`none|merge|cls_attn|random`, reduction after attention of blocks 4/7/10), analytic GFLOPs, triplet, mAP/CMC.
- `train.py` baseline / low-data (`--id-fraction`) / reduced-token training / TBSD (`--sd-mode`).
- `evaluate.py` token-reduction sweep: symmetric vs asymmetric (full gallery, reduced query) mAP, CSV + PNG.
- `make_fake_market.py` synthetic data for install checks only.
- `runs/` experiment outputs (`log.json`, `last.pth`, `sweep.csv/png`).

## Sanity expectations
- Full-token ViT-S GFLOPs at 256x128 ≈ 2.93 (analytic). `keep=1.0` must equal `none`.
- Baseline ViT-S Market-1501 mAP expected (unverified) in the 80s after 120 epochs; far lower = bug.
