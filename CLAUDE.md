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

## Autonomy (standing instructions from the user, 2026-11-15; apply to every session)
On resume: read CLAUDE.md, PROJECT_CONTEXT.md, PROGRESS.md, experiment logs and saved checkpoints first.
- Restore the latest valid training checkpoint and continue; do not repeat completed work.
- Work autonomously until all remaining research objectives and experiments are completed.
- Do not wait for approval between routine tasks, experiments, implementation stages or debugging steps.
- Make reasonable technical decisions independently and document the reasoning (PROGRESS_LOG.md).
- Run the necessary terminal commands, install required project dependencies, modify code, run experiments.
- Monitor training progress, GPU utilisation, VRAM use and estimated completion times.
- Investigate errors, implement fixes and retry failed operations where appropriate.
- Save resumable model checkpoints regularly.
- Evaluate results objectively against the baseline. Do not fabricate results or selectively exclude
  unsuccessful experiments.
- Update PROGRESS.md continuously. Commit and push after every meaningful milestone.
- Preserve existing checkpoints, datasets and successful experimental results.
- Work within the hardware resources of the RTX 5080.
- Do not stop merely to summarise progress or to ask whether to proceed; continue to the next pending task.
- Stop and ask only for a genuine blocker: credentials, payment, destructive operations, elevated system
  permissions, or a major change to the research objective.
- When all tasks are complete, write a comprehensive final report: methodology, experimental results,
  comparisons, limitations, conclusions, potential research contributions.
- Interpretation (Claude's, not stated by the user): the earlier rule "downloading new datasets needs the
  user's explicit OK" still stands, because these instructions do not mention datasets and several re-ID
  datasets need a signed agreement. Dataset-dependent tasks are deferred and listed, not silently skipped.

## Explain everything (standing instructions from the user, 2026-11-15; apply to every session)
The user wants to be able to reproduce and defend this research independently. Never silently run commands,
modify files or run experiments. Explanations go **in the conversation**, not only in Markdown files.
- **Before** each meaningful action: what is about to be done, why it is necessary, how it serves the main
  research objective, what is expected to happen.
- **While** doing it: what the commands do; what the code being written or changed does (files, functions);
  the underlying mathematics, algorithms and concepts; the important parameters and why those values.
- **After**: what happened and what the output means; whether it met the expectation; any errors, their cause
  and the fix; what was learned; what comes next and why.
- For training experiments state: architecture, dataset, batch size, epochs, iterations, loss functions,
  hyperparameters, GPU usage, evaluation metrics, expected duration.
- Never just say something worked: explain how and why.
- Start beginner-friendly, then go deeper into the technical and mathematical detail.
- Do not wait for acknowledgement after explaining; keep working autonomously.
- Group repetitive operations, but never skip the explanation of an important technical decision.
- Maintain `RESEARCH_LOG.md` (explanatory record: concepts, reasoning, interpretation of every experiment).
  `PROGRESS_LOG.md` stays the terse chronological record (commands, tables); `PROGRESS.md` the status summary.

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

## State at the last pause (2026-11-16 01:00, PC shutdown; details and resume commands: PROGRESS.md)
- Nothing was left running. `run_iter3.sh` was paused by STOP file. Done: phase M; `sp_lowres_s0/s1`,
  `sp_merge_r_s0/s1`, `sp_mixed_s0`. Partial: `sp_mixed_s1` (`runs/sp_mixed_s1/ckpt.pth`, epoch 61/120).
  Pending: `sp_lowres_s2`, `sp_merge_r_s2`, `sp_mixed_s2`, phase D (`vitb_base_s0`, `vitb_tbsd_s0`, ViT-B latency).
- Resume: `bash run_iter3.sh /c/Users/user/data/Market-1501-v15.09.15 E D >> runs/queue_iter3.txt 2>&1`
  (skips finished runs, resumes the partial one). Verify first that no `.venv` python.exe is already running.
- **Priority on return (user, 2026-11-16): finish the first research-iteration report** — compare about 5-6
  papers, defensible preliminary gap, our own methodology, evaluation against appropriate benchmarks. It does
  not need to be perfect or to beat existing methods. Checklist in PROGRESS.md; reasoning in RESEARCH_LOG.md Part 5.
- Preliminary (2/2/1 seeds, NOT results yet): gradual-merge second pass loses least at full tokens (84.25) and
  reaches 82.8 at 1.41 GFLOPs; low-resolution second pass reaches 82.4 at 1.12 and 80.9 at 0.72 GFLOPs but
  83.1 at full tokens; both are ahead of the equal-GPU-hour baseline + linear map below ~1.5 GFLOPs.
- Second dataset still blocked on the user's own sign-up (CUHK03 Google Form preferred; Occluded-REID agreement).

## Key decisions and results so far (details: PROGRESS.md)
- Main backbone ViT-S/16 (open question to instructor: ViT-B?). Market-1501 at `C:/Users/user/data/`.
- TBSD (merge, random keep) is the method under test. 3 seeds: 81.98±0.18 vs 68.56±1.14 mAP at keep 0.5
  (full-token gallery); full-token cost ~0.5 mAP (TBSD lower in all seeds); ~1.85x training time.
- Equal-GPU-h baseline (216 ep) gains +1.3 full-token mAP but not budget robustness.
- No-distillation control (`--sd-weight 0`) reaches 80.6 at keep 0.5: most of the gain is multi-budget training;
  distillation adds ~+1.4 (one seed). Do not claim distillation as the source of the gain without more seeds.
- **Direction (user, 2026-11-15 evening): TBSD is an exploratory probe, NOT the contribution.** Priorities:
  full-read literature (verified metadata; never infer a paper lacks a feature from its abstract), gap
  reassessment, mechanism, own justified methodology. Self-distillation and token pruning are options, not
  requirements. Finishing a queue is not finishing the research. See `LITERATURE_MATRIX.md`,
  `METHOD_CANDIDATES.md`, RESEARCH_LOG.md Part 4.
- Gap is narrow: SFSC (CVPR 2023) and D3still (CVPR 2024) already do compatible / asymmetric re-ID embeddings
  (CNN width, separate networks, resolution); our alignment loss equals D3still's feature term. Not found:
  token count of one ViT as the cheap-query knob.
- **Honest size of TBSD's gain:** our 3-stage schedule (reduce at blocks 4/7/10) weakens the baseline.
  At ~1.4-1.5 GFLOPs, asymmetric mAP: baseline 3-stage merge 68.55, gradual ToMe (`merge_r`) 77.45,
  low-resolution queries ~79.6 (interpolated), TBSD 81.99. Always compare against gradual ToMe and
  low-resolution queries, not only against the 3-stage baseline.
- Market-1501 crops are natively 128x64 (upsampled to 256x128): low-resolution results may not transfer.
- H3 (low-data regulariser) not supported at 10/25/50% of identities, 3 seeds.
- Second dataset: user approved in principle; CUHK03 (Google Form), Occluded-REID (user agreement) and MSMT17
  (request) all need the user's own sign-up. Nothing downloaded.
- Downloaded third-party code may be read but not executed (blocked by the permission system); validate
  against transcriptions (`check_reduction.py`) and say so.
- H1 reframed (asymmetric > symmetric; problem is query drift). H3 not supported (one seed).
- cls_attn selection < random selection: holds with the exact EViT fusion (`cls_attn_evit`), 3 seeds; cause
  unexplained ("systematic offset" ruled out).
- Throughput must be timed at batch >= 256 on GPU (batch 64 is overhead-bound).

## Code layout
- `common.py` data parsing (Market-1501 names), PK sampler, `ReIDViT` + token reduction
  (`none|merge|cls_attn|random`, reduction after attention of blocks 4/7/10), analytic GFLOPs, triplet, mAP/CMC.
- `train.py` baseline / low-data (`--id-fraction`) / reduced-token training / TBSD (`--sd-mode`).
- `evaluate.py` token-reduction sweep: symmetric vs asymmetric (full gallery, reduced query) mAP, CSV + PNG.
- `make_fake_market.py` synthetic data for install checks only.
- Added in session 2: modes `cls_attn_fuse|cls_attn_low|cls_attn_evit|merge_r` (`model.tome_r`), optional
  low-resolution input `model(x, mode, keep, hw)`; `train.py --sd-bn`, `--sd-mode lowres|merge_r|mixed`;
  `mechanism.py` (drift diagnostics, training-free corrections, low-res queries) + `aggregate_mech.py`;
  `check_reduction.py` (ours vs transcribed official ToMe/EViT); `bench_latency.py`; `aggregate2.py`;
  queues `run_iter2.sh` (done) and `run_iter3.sh` (phases M, E, D).
- `runs/` experiment outputs (`log.json`, `last.pth`, `sweep.csv/png`).

## Sanity expectations
- Full-token ViT-S GFLOPs at 256x128 ≈ 2.93 (analytic). `keep=1.0` must equal `none`.
- Baseline ViT-S Market-1501 mAP expected (unverified) in the 80s after 120 epochs; far lower = bug.
