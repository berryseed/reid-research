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

## PRIORITIES FROM 2026-11-17 (user; supersede earlier novelty and scope rules where they conflict)
- **Iteration-1 deadline is 2026-11-18.** Goal: a complete, understandable, defensible first iteration,
  not a publication-ready paper. Use existing results as the foundation.
- **Revised novelty requirement:** the methodology does not need to be entirely novel, beat existing methods
  or reach state of the art. A preliminary adaptation or combination of existing techniques is acceptable if
  prior work is acknowledged accurately. Do not keep rejecting or redesigning the method in pursuit of
  absolute novelty. Worse performance than baselines is acceptable if reported fairly and explained.
- Time boxes: per-query budget selection literature <= 45 min; extra experiments <= 2 h, evaluation-only with
  existing checkpoints (no retraining). If the adaptive mechanism is not promising, keep the multi-budget
  method (`sp_mixed`) as iteration 1 and propose adaptive budget selection for iteration 2.
- Second dataset (CUHK03 etc.) is future work, not a blocker.
- Deliverables: `REPORT_ITERATION1.md` (5-6 paper comparison incl. 2025-2026, gap, method, architecture,
  maths, results, fair comparison, negative results, limitations), a beginner study guide
  (`STUDY_GUIDE.md`), supervisor questions with answers (`SUPERVISOR_QA.md`). Save and push regularly.

## State 2026-11-17 evening: ITERATION 1 COMPLETE
- Deliverables written and pushed: `REPORT_ITERATION1.md` (final), `STUDY_GUIDE.md`, `SUPERVISOR_QA.md`.
- The framing decision below is resolved by the user's 2026-11-17 priorities: iteration 1 = `sp_mixed`
  as a credited combination of existing techniques, evaluated against equal-budget baselines.
- Per-query cascade (`adaptive.py`, E17): oracle-quality top-1 signal but dominated by fixed budgets;
  iteration-2 work (needs stage-1 feature reuse). Iteration-2 plan: REPORT_ITERATION1.md Section 10.
- Nothing is running. Code added today: `adaptive.py`, `aggregate_adaptive.py`.

## Session 4 morning (2026-11-17): literature pass (framing decision since resolved) (details: RESEARCH_LOG.md Part 7)
- PrunNet (arXiv 2504.11879), ResFormer (CVPR 2023) and RFD (cross-resolution re-ID) read in full
  (`FULL-S`, LITERATURE_MATRIX.md section F). **Every component of our method is prior work.** Remaining
  gap (not found): input-side compute knobs of one ViT re-ID model compatible with its own full-cost
  gallery, with measured latency and equal-budget accounting = an empirical study, not a method.
- **Do not start new experiments until the user chooses the framing** (REPORT_ITERATION1.md Section 9):
  (a) empirical study, (b) a new mechanism such as per-query budget selection (read DVT and
  dynamic-token ViTs first), (c) native-resolution data (needs the CUHK03 sign-up). This is a change of
  research objective, so it is the user's call under the autonomy rules.
- Do not describe the low-resolution second pass or one-network multi-budget compatibility as new
  (ResFormer; SFSC, PrunNet).

## State at the end of session 3 (2026-11-16 23:50, PC shutdown; details: PROGRESS.md, RESEARCH_LOG.md Part 6)
- **Nothing is running and nothing is partial.** `run_iter3.sh` (phases M, E, D) and `run_iter4.sh` (phase F) are
  finished. No `ckpt.pth`, no STOP file, nothing to resume. All metrics pushed; `*.pth` local only.
- **First research-iteration report written: `REPORT_ITERATION1.md`.** Keep it in step with any new result.
- **On resume, in this order** (full list: PROGRESS.md "What to do first when resuming"):
  1. confirm the machine is idle and git is in sync;
  2. literature first, no GPU: cross-resolution / low-resolution-query re-ID with a compatible gallery; one
     query model for several budgets in asymmetric retrieval (ACD, AML, ROP); ToCom, MatMamba, ThinkingViT.
     This decides whether the remaining gap survives;
  3. raise the open method decision with the user (below);
  4. then the "Remaining experiments" table in PROGRESS.md.
- **Open decision, not settled:** `sp_mixed` (one set of weights; two-pass training, cheap pass drawn from
  gradual merging and low resolution; 83.95 full / 83.60 @1.62 G / 82.38 @1.41 G / 83.01 @1.12 G / 81.50 @0.72 G)
  versus the frozen-gallery student `fg_mixed` (prior-work style, Suma and Tolias; two sets of weights,
  label-free; 84.45 full / 83.68 / 81.50 / 82.78 / 80.48). They tie on low resolution; `sp_mixed` is about
  1 mAP ahead under merging and at 128x64 (suggestive only, 3 seeds). Do not claim `sp_mixed` beats it.
- Both beat the equal-GPU-hour baseline + linear map below about 1.7 GFLOPs (`sp_mixed` +0.8 to +4.1) and are
  below it at full tokens (85.11). Specialists (`sp_lowres`, `sp_merge_r`) do not transfer across knobs.
- Measured latency (`runs/latency_extra.csv`): low-resolution queries beat gradual merging in speed and accuracy
  on Market-1501; gradual merging is 2.3x slower than full tokens at GPU batch 1; nothing reduces single-image
  GPU latency (about 6 ms). Do not present token merging as the better knob.
- Suma and Tolias (WACV 2023) read in full: frozen gallery + same-architecture student = candidate C1. C1 is a
  baseline (now implemented), not a contribution.
- ViT-B (1 seed): baseline 86.38, TBSD 83.68 (deficit 2.7); worse than ViT-S per GFLOP. ViT-S stays main.
  **ViT-B runs need `--workers 2`** (Windows error 1455, system memory, at evaluation with 4 workers).
- To hide the GPU from a script use `CUDA_VISIBLE_DEVICES=-1`; an empty value does NOT hide it.
- Apply long documentation edits from files (Write tool + a small script), not from long inline shell heredocs.
- Budgets are only approximately equal and must be stated: 216-epoch baseline 0.36-0.38 GPU-h; `sp_mixed`
  0.35-0.44; `sp_merge_r` 0.40-0.46; teacher + `fg_mixed` student about 0.46.
- Second dataset still blocked on the user's own sign-up (CUHK03 Google Form preferred; Occluded-REID agreement).
  It now decides whether token merging has any role.

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
- Added in session 3: `train.py --teacher <last.pth>` (frozen-gallery student, label-free);
  `mechanism.py --gallery-ckpt` (gallery from another checkpoint); `bench_latency.py --extra` (low-resolution
  and gradual-merge rows); queue `run_iter4.sh` (phase F, done); `REPORT_ITERATION1.md`; aggregates
  `runs/sp_mechanism*.{csv,md}`, `runs/fg_mechanism*.{csv,md}`, `runs/latency_extra.csv`, `runs/latency_vitb.csv`.
- `runs/` experiment outputs (`log.json`, `last.pth`, `sweep.csv/png`).

## Sanity expectations
- Full-token ViT-S GFLOPs at 256x128 ≈ 2.93 (analytic). `keep=1.0` must equal `none`.
- Baseline ViT-S Market-1501 mAP expected (unverified) in the 80s after 120 epochs; far lower = bug.
