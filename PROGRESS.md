# PROGRESS.md — status summary

Current state at a glance. Detailed chronological record with exact commands and full tables:
`PROGRESS_LOG.md`. Background, decisions and literature: `PROJECT_CONTEXT.md`.
All numbers are Market-1501, ViT-S/16 (ImageNet-21k->1k augreg weights), 256x128, 120 epochs, unless stated.

_Last updated: 2026-11-17 evening (iteration 1 complete; deliverables written)._

## ITERATION 1 COMPLETE (2026-11-17). Nothing is running.
Deliverables: **`REPORT_ITERATION1.md`** (final report), **`STUDY_GUIDE.md`** (beginner explanation),
**`SUPERVISOR_QA.md`** (likely questions with answers). Deadline 2026-11-18.
- **Method of iteration 1: `sp_mixed`** (one ViT-S, two passes, cheap pass drawn from low resolution and gradual
  merging, shared classifier, cosine alignment to the detached full pass). Credited combination of SFSC/BCT,
  D3still / Suma-Tolias, ResFormer, ToMe.
- **Per-query cascade (E17, evaluation only, 3 seeds):** top-1 gallery similarity of the cheap query is an
  oracle-quality escalation signal, but the cascade is dominated by fixed budgets at equal cost (`sp_mixed`
  −0.1 to −0.7 mAP). Moved to iteration 2 (with stage-1 feature reuse). `runs/adaptive_agg.md`, `runs/adaptive.png`.
- **Iteration 2 plan:** REPORT_ITERATION1.md Section 10 (second dataset; cascade with feature reuse; reduce the
  full-cost deficit; re-implement one published multi-budget baseline; ViT-B seeds; validation split).

### History (morning of 2026-11-17)
Session 4 (2026-11-17): literature pass done; WAITING on the user's framing decision. Nothing is running.
- Read in full (`FULL-S`): **PrunNet** (arXiv 2504.11879: one network, pruned subnetworks compatible with the
  dense model, incl. ViT-S and Market-1501/MSMT17), **ResFormer** (CVPR 2023: detached higher-resolution class
  token teaches lower resolution = our low-resolution training mechanism), **RFD** (cross-resolution re-ID:
  low-resolution queries upsampled to full input, no compute saving). LITERATURE_MATRIX.md section F.
- **Consequence:** every component of our method is prior work. Remaining gap (not found): input-side compute
  knobs (resolution, tokens) of one ViT re-ID model, compatible with its own full-cost gallery, measured in
  latency against equal-budget baselines. Supports an empirical study, not a method contribution.
- **Decision needed from the user (changes the research objective):** (a) empirical study; (b) new mechanism,
  e.g. per-query budget selection (literature check first: DVT and dynamic-token ViTs); (c) native-resolution
  data (needs CUHK03 sign-up). REPORT_ITERATION1.md Section 9, RESEARCH_LOG.md Part 7.
- Earlier open decision (`sp_mixed` vs frozen-gallery student) is subordinate to this one.

## Current status: PAUSED CLEANLY (2026-11-16 23:50, PC shutdown at the user's request). Nothing is running.
All queues are finished (`run_iter3.sh` phases M, E, D; `run_iter4.sh` phase F). There is **no partial run, no
`ckpt.pth`, no STOP file**: nothing has to be resumed. All 9 checkpoints produced on 2026-11-16 (`sp_mixed_s1`, `sp_lowres_s2`,
`sp_merge_r_s2`, `sp_mixed_s2`, `vitb_base_s0`, `vitb_tbsd_s0`, `fg_mixed_s0/1/2`) were loaded on the CPU and
are finite. Everything except `*.pth` is pushed to GitHub.

### Where we stopped (end of session 3)
- **First research-iteration report: written**, `REPORT_ITERATION1.md` (paper comparison, gap, method, 3-seed
  evaluation, measured latency, ViT-B check, frozen-gallery baseline, limitations, conclusions).
- **Open decision (do not settle silently):** which scheme is "the method"? `sp_mixed` (one set of weights,
  -1.16 mAP at full tokens) or the frozen-gallery student `fg_mixed` (prior-work style, two sets of weights, no
  full-token loss, label-free second stage). They tie on low-resolution queries; `sp_mixed` is about 1 mAP
  ahead under gradual merging and at 128x64 (suggestive only). Table below.
- **Main risk to novelty:** resolution beats token merging on Market-1501 in accuracy and measured speed, and
  resolution asymmetry is prior work (Suma and Tolias). Cross-resolution / low-resolution-query re-ID
  literature has **not been searched yet**.

### What to do first when resuming (in this order)
1. Read `CLAUDE.md`, this file, `RESEARCH_LOG.md` Part 6, `REPORT_ITERATION1.md`.
2. Confirm the machine is idle: `tasklist //FI "IMAGENAME eq python.exe"` and `nvidia-smi` (Git Bash).
   Confirm sync: `git status -sb` (no ahead/behind), `git log origin/main -1`.
3. **Literature first (no GPU):** search and read in full for (a) cross-resolution / low-resolution-query person
   re-ID with a compatible high-resolution gallery, (b) one query model serving several budgets in asymmetric
   retrieval (ACD CVPR 2026 Findings, AML, ROP), (c) ToCom, MatMamba, ThinkingViT. Update
   `LITERATURE_MATRIX.md` section E. This decides whether the remaining gap survives.
4. Ask the user / instructor about the open method decision above, or recommend one with the evidence.
5. Then the experiments in "Remaining experiments" below. Ask the user about the CUHK03 sign-up.

### Remaining experiments (none started; times are estimates from today's run times, not measurements)
| # | Experiment | Why | Command sketch | Est. time |
| --- | --- | --- | --- | --- |
| 1 | Frozen-gallery student on the 216-epoch teacher, 3 seeds | Keeps 85.11 at full tokens by construction; tests whether the low-cost numbers follow the stronger teacher | as `run_iter4.sh` with `--teacher runs/base_e216_s$s/last.pth` and `--gallery-ckpt` the same file, out `runs/fg216_mixed_s$s` | 3 x (0.24 h + 2 min) = about 0.8 h |
| 2 | Gradient-conflict diagnostic (cosine between full-pass and cheap-pass gradients during `sp_mixed` training) | Decides candidate C3 (protect the full-token objective); needs new code in `train.py` | new flag, 1 short run | about 0.5 h incl. coding |
| 3 | Separate-classifier control for `sp_mixed`, 3 seeds | Tests whether the shared classifier is what makes budgets compatible | new flag in `train.py` | 3 x 0.45 h = about 1.4 h |
| 4 | `sp_mixed` and `fg_mixed` on ViT-B, seed 0 (`--workers 2`) | Second backbone for the retained method (only TBSD was run on ViT-B) | `train.py --model vit_base_patch16_224 --sd-mode mixed --workers 2` | about 0.8 h + 0.5 h |
| 5 | More seeds of the ViT-B baseline and probe | ViT-B numbers are 1 seed | as phase D with `--seed 1/2 --workers 2` | 2 x 1.3 h = about 2.6 h |
| 6 | Second dataset (CUHK03-NP), loader + main table | The single most important missing piece | blocked on the user's sign-up | loader about 1 h; runs about 4-5 h |
| 7 | Spatial-coverage test for class-attention selection | Unexplained negative result | evaluation only | about 0.5 h |

### Frozen-gallery baseline vs our method, FINAL for Market-1501 / ViT-S (3 seeds, asymmetric mAP)
| Query setting | GFLOPs | baseline 216 ep + ridge map | `sp_mixed` | `fg_mixed` |
| --- | --- | --- | --- | --- |
| full tokens | 2.93 | 85.11 | 83.95 ± 0.15 | 84.45 ± 0.49 (= teacher `base_s*`) |
| low-res 224x112 | 2.22 | not tabulated | 83.91 ± 0.17 | 84.22 ± 0.49 |
| gradual ToMe r=7 | 1.94 | 83.75 ± 0.03 | 83.70 ± 0.14 | 83.74 ± 0.53 |
| low-res 192x96 | 1.62 | 82.80 ± 0.23 | 83.60 ± 0.14 | 83.68 ± 0.49 |
| gradual ToMe r=11 | 1.41 | 79.70 ± 0.13 | 82.38 ± 0.27 | 81.50 ± 0.54 |
| low-res 160x80 | 1.12 | 78.96 ± 0.18 | 83.01 ± 0.15 | 82.78 ± 0.55 |
| low-res 128x64 | 0.72 | not measured | 81.50 ± 0.22 | 80.48 ± 0.69 |

`fg_mixed`: `train.py --teacher runs/base_s$s/last.pth --sd-mode mixed --epochs 96` (label-free,
loss `mean(1 - cos(student_cheap, teacher_full))`), evaluated with `mechanism.py --gallery-ckpt`. Student
0.23 / 0.24 / 0.24 GPU-h, peak VRAM 1.83 GB; teacher + student about 0.46 GPU-h (about 20 % above the 216-epoch
baseline). Per-seed teacher full-token mAP 83.89 / 84.71 / 84.75. Aggregates: `runs/fg_mechanism.csv`,
`runs/fg_mechanism_agg.md`.

### Session-3 summary (2026-11-16)
- Phase E complete: 9 second-pass runs. Phase D complete: `vitb_base_s0` 86.38 mAP; `vitb_tbsd_s0` 83.68 (failed
  once on Windows error 1455 = system memory at the epoch-60 evaluation; resumed with `--workers 2`).
  **Use `--workers 2` for ViT-B runs on this machine.**
- Latency measured on the idle machine: `runs/latency_extra.csv` (ViT-S, incl. low-res and gradual merge),
  `runs/latency_vitb.csv`.
- **Main conclusions of iteration 1** (Market-1501, ViT-S, 3 seeds): `sp_mixed` is compatible under both knobs
  (83.0 mAP at 1.12 G, 82.4 at 1.41 G) and beats the equal-GPU-hour baseline + linear map by +0.8 to +4.1 below
  1.7 GFLOPs, but is 1.16 below it at full tokens. Low-resolution queries beat gradual merging in accuracy and in
  measured speed; merging is 2.3x slower than full tokens at GPU batch 1. ViT-B (1 seed): TBSD full-token deficit
  2.7 mAP; ViT-B is worse than ViT-S per GFLOP. ViT-S stays the main backbone. The frozen-gallery baseline ties
  `sp_mixed` on low resolution without a full-token loss.
- Mistakes made today, recorded so they are not repeated: (1) `CUDA_VISIBLE_DEVICES=""` does not hide the GPU
  (use `-1`); a smoke test shared the GPU with `sp_mixed_s1` for about a minute. (2) A long inline shell script
  failed on quoting; long text edits are now applied from files. (3) Student epochs were estimated at 0.11 min
  and took 0.15 min, so the frozen-gallery budget is about 20 % over the intended equal budget.

### History: state of the iteration-3 queue at the pause of 2026-11-16 01:00 (all of it has since finished)
- **Research direction (user, 2026-11-15):** TBSD is an exploratory probe, not the contribution. Priorities:
  full-read literature review, gap reassessment, mechanism, own methodology. See `LITERATURE_MATRIX.md`,
  `METHOD_CANDIDATES.md`, RESEARCH_LOG.md Parts 4 and 5.
- **Priority on return (user, 2026-11-16): complete the first research iteration report** — compare about 5-6
  relevant papers, state a defensible preliminary gap, present our own methodology, evaluate it against
  appropriate benchmarks. It does not need to be perfect or to beat existing methods. Checklist below.

### State of the iteration-3 queue at the pause
| Run | Status | Files |
| --- | --- | --- |
| phase M (diagnostics on `base_e216_s0-2`, `tbsd_nosd_s0-2`) | complete, pushed | `runs/<name>/mech.csv` |
| `sp_lowres_s0`, `sp_lowres_s1` | complete (train + mech + sweep), pushed | `last.pth`, `log.json`, `mech.csv`, `sweep.csv` |
| `sp_merge_r_s0`, `sp_merge_r_s1` | complete, pushed | same |
| `sp_mixed_s0` | complete, pushed | same |
| `sp_mixed_s1` | **partial: paused after epoch 61/120** (step 12,322; 11.4 min trained; mAP 83.2 at epoch 60) | `runs/sp_mixed_s1/ckpt.pth` (verified loadable: model, optimizer, AMP scaler, epoch, step, RNG, log, args) |
| `sp_lowres_s2`, `sp_merge_r_s2`, `sp_mixed_s2` | pending, not started | none |
| `vitb_base_s0`, `vitb_tbsd_s0`, `runs/latency_vitb.csv` (phase D) | pending, not started | none |

Checkpoints (`*.pth`) are local only (git-ignored); metrics are on GitHub.

### Exact resume commands (Git Bash, from the project folder)
- **Everything that is left, in order (recommended):**
  `bash run_iter3.sh /c/Users/user/data/Market-1501-v15.09.15 E D >> runs/queue_iter3.txt 2>&1`
  Finished runs are skipped (`last.pth` present); `sp_mixed_s1` continues from `ckpt.pth` at epoch 62; each
  finished run is evaluated and auto-pushed. Estimated (not measured) remaining time: phase E about 1.6 h,
  phase D about 2.5 h. Run only phase E by passing `E` alone.
- **Single runs by hand** (the queue does exactly this):
  - `.venv/Scripts/python.exe train.py --data C:/Users/user/data/Market-1501-v15.09.15 --out runs/sp_mixed_s1 --resume --seed 1 --sd-mode mixed`
  - `.venv/Scripts/python.exe train.py --data C:/Users/user/data/Market-1501-v15.09.15 --out runs/sp_lowres_s2 --resume --seed 2 --sd-mode lowres`
  - `.venv/Scripts/python.exe train.py --data C:/Users/user/data/Market-1501-v15.09.15 --out runs/sp_merge_r_s2 --resume --seed 2 --sd-mode merge_r`
  - `.venv/Scripts/python.exe train.py --data C:/Users/user/data/Market-1501-v15.09.15 --out runs/sp_mixed_s2 --resume --seed 2 --sd-mode mixed`
  - after each: `.venv/Scripts/python.exe mechanism.py --data C:/Users/user/data/Market-1501-v15.09.15 --ckpts runs/<name>/last.pth --out runs/<name>/mech.csv`
    and `.venv/Scripts/python.exe evaluate.py --data C:/Users/user/data/Market-1501-v15.09.15 --ckpt runs/<name>/last.pth --out runs/<name>/sweep.csv --modes merge`
  - `.venv/Scripts/python.exe train.py --data C:/Users/user/data/Market-1501-v15.09.15 --out runs/vitb_base_s0 --resume --seed 0 --model vit_base_patch16_224`
  - `.venv/Scripts/python.exe train.py --data C:/Users/user/data/Market-1501-v15.09.15 --out runs/vitb_tbsd_s0 --resume --seed 0 --model vit_base_patch16_224 --sd-mode merge --sd-keep random`
  - `.venv/Scripts/python.exe bench_latency.py --model vit_base_patch16_224 --out runs/latency_vitb.csv`

### Second-pass variants, FINAL for Market-1501 / ViT-S (3 seeds each, mean ± std)
Asymmetric mAP, full-token gallery. Brackets: with the closed-form linear (ridge) map. Aggregate:
`runs/sp_mechanism.csv` -> `runs/sp_mechanism_agg.md`; full-token row from `runs/sp_*/sweep.csv`.
| Query setting | GFLOPs | baseline 120 ep | baseline 216 ep | TBSD 3-stage | `sp_lowres` | `sp_merge_r` | `sp_mixed` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full tokens | 2.93 | 84.45 ± 0.49 | 85.11 | 83.95 ± 0.13 | 83.09 ± 0.25 | 84.27 ± 0.07 | 83.95 ± 0.15 |
| low-res 192x96 | 1.62 | 81.38 ± 0.36 | 82.06 (82.80 ± 0.23) | 81.91 ± 0.32 | 82.85 ± 0.17 | 82.68 ± 0.10 (82.94) | 83.60 ± 0.14 |
| gradual ToMe r=11 | 1.41 | 77.45 ± 1.39 | 78.97 (79.70 ± 0.13) | 81.77 ± 0.18 | 79.59 ± 0.37 | 82.90 ± 0.13 | 82.38 ± 0.27 |
| low-res 160x80 | 1.12 | 75.81 ± 0.43 | 75.72 (78.96 ± 0.18) | 78.26 ± 0.38 | 82.36 ± 0.19 | 79.89 ± 0.18 (81.00) | 83.01 ± 0.15 |
| low-res 128x64 | 0.72 | 60.13 ± 2.68 | not measured | 69.26 ± 0.90 | 80.96 ± 0.20 | 73.04 ± 0.37 (76.31) | 81.50 ± 0.22 |

Per-seed full-token mAP: `sp_lowres` 82.83 / 83.33 / 83.10; `sp_merge_r` 84.19 / 84.32 / 84.30; `sp_mixed`
83.77 / 84.03 / 84.04. Train GPU-h: `sp_lowres` 0.36 / 0.31 / 0.37; `sp_merge_r` 0.44 / 0.40 / 0.46; `sp_mixed`
0.35 / 0.41 (unreliable: resumed, GPU shared for about a minute) / 0.44; 216-epoch baseline 0.36-0.38.
Reading:
- Specialists do not transfer: `sp_merge_r` 73.0 at 128x64; `sp_lowres` 79.6 under gradual merging (= baseline
  216 ep + map, 79.7).
- **`sp_mixed` is compatible under both knobs**: best on all low-resolution settings (+0.5 to +0.75 over
  `sp_lowres`, about 3x the seed spread), 0.5 below `sp_merge_r` at r=11 (about 2 std, not clearly separated).
- vs equal-GPU-hour baseline + map: +4.1 (1.12 G), +2.7 (1.41 G), +0.8 (1.62 G). The specialists only tie it at 1.62 G.
- Full-token price: `sp_mixed` -1.16 vs the 216-epoch baseline (same as TBSD); `sp_merge_r` loses least (-0.84).
- Budgets are only approximately equal: `sp_merge_r` used about 15 % more GPU time than the 216-epoch baseline.
- Caveat unchanged: Market-1501 is natively 128x64; resolution results need a second dataset.
- **Method retained: `sp_mixed`.** Open problem: the full-token deficit.

### Earlier status notes
- Iteration 2 (phases S, A, B, C of `run_iter2.sh`) is complete; all 16 low-data runs done and pushed.
- Power outage 2026-11-15 ~12:33: recovered without loss of completed results (details in PROGRESS_LOG.md).

### Findings that changed the picture (3 seeds unless stated; full tables in PROGRESS_LOG.md)
- **Gap is narrower than assumed.** SFSC (CVPR 2023) already trains mutually compatible multi-capacity re-ID
  embeddings (CNN width) on Market-1501/MSMT17; D3still (CVPR 2024) does asymmetric retrieval on MSMT17; our
  alignment loss equals their feature term. Not found: token count of one ViT as the asymmetric knob.
- **Our headline gain was inflated by our own 3-stage schedule.** Baseline at ~1.4-1.5 GFLOPs (asymmetric mAP):
  3-stage merge 68.55 ± 1.11; official-style gradual ToMe 77.45 ± 1.39; low-resolution queries ~79.6
  (interpolated; 81.38 ± 0.36 at 1.62 G, 75.81 ± 0.43 at 1.12 G); TBSD 81.99 ± 0.18. TBSD's margin over the
  best training-free option is about +2.4 mAP for 1.85x training time and ~1 mAP at full tokens.
- **Equal-GPU-hour control (phase M): the 216-epoch baseline + low-resolution queries + closed-form linear map
  matches TBSD below full cost (82.80 vs 81.91-82.34 at 1.62 G; 78.96 vs 78.26-79.90 at 1.12 G) and is 1.16 mAP
  better at full tokens. On Market-1501, TBSD as built is not justified against this baseline.**
- **Low-resolution queries are a strong training-free competitor** (caveat: Market-1501 is natively 128x64).
- **Drift is per-image**, not a shared shift (offset fraction 0.03-0.16); a closed-form linear map recovers
  +1 to +4 mAP. Early reduction is the costly part (block 4 / 7 / 10 only: -3.8 / -1.2 / -0.2 mAP).
- **Implementation check:** our merge and top-k selection equal a transcription of official ToMe/EViT; our
  fused token was an average where EViT uses a sum. With the exact EViT fusion, class-attention selection is
  still below random (72.93 ± 3.28 vs 78.06 ± 0.63, keep 0.7).
- **H3 not supported:** at 10/25/50% of identities TBSD is 2.2/0.9/0.9 mAP below the baseline at full tokens;
  the low-budget gain persists (+12 to +15 with the 3-stage schedule).
- Earlier iteration-2 results stand: distillation term adds +1.4 at keep 0.5; BNNeck-stat ablation negative;
  fixed-budget specialist loses to TBSD; equal-GPU-hour baseline 85.11 vs TBSD 83.95 at full tokens.

### Blocked on the user: second dataset (nothing downloaded)
- **CUHK03 (preferred; higher native resolution, Market-style CUHK03-NP split 767/700 identities, about 3 GB):**
  the official CUHK page gives access through a Google Form that asks for the applicant's personal details and
  acceptance of CUHK's terms. The user must fill it in themselves, then place the archive under
  `C:/Users/user/data/` and say so. Claude will then write the loader; no code depends on it yet.
- **Occluded-REID (test-only, 2,000 images; cross-domain test comparable with SUReID):** zip in
  github.com/tinajia2012/ICME2018_Occluded-Person-Reidentification_datasets with a user agreement (academic
  use only). Needs the user's acceptance.
- **MSMT17:** request to the PKU authors. Slowest route.

### Requirements for the first research-iteration report (checklist) — ALL ADDRESSED on 2026-11-16 in `REPORT_ITERATION1.md`
Status of each item: 1 done (six papers; Suma and Tolias read in full); 2 done (gap stated as "not found",
narrowed further by our own latency and frozen-gallery results); 3 done (`sp_mixed`, with the frozen-gallery
student as the open alternative); 4 done (3 seeds, all listed baselines); 5 done (`runs/latency_extra.csv`);
6 and 7 done. The original wording is kept below as history.
1. **Paper comparison (about 5-6 papers).** Proposed set: SFSC, D3still, SUReID, CSD (all read in full),
   ToMe and EViT (method sections and official code read), plus Suma and Tolias (resolution asymmetry; **still
   to read in full** because it is the closest prior work to the low-resolution result). Source:
   `LITERATURE_MATRIX.md`. Remaining full reads (BCT, AML, MatMamba, ThinkingViT, ACD, ToCom, DiffRate) are
   second-tier for the report.
2. **Preliminary gap statement.** Current wording: compatible multi-budget re-ID embeddings (SFSC) and
   asymmetric re-ID retrieval (D3still) exist with CNN width, separate networks and resolution as the knob;
   not found in the texts read: the token count of a single ViT as the cheap-query knob, nor compatibility
   across more than one knob. Must be stated as "not found", not "does not exist".
3. **Our methodology.** To be fixed after seed 2: budget-compatible training of one ViT with a gradual-merge
   and/or low-resolution second pass (candidate C2), with candidate C1 (frozen gallery model, trained
   cheap-query copy) as the alternative that removes the full-token loss. See `METHOD_CANDIDATES.md`.
4. **Evaluation.** 3 seeds of `sp_*` (3 runs + half a run left); comparison against: 120-epoch baseline,
   equal-GPU-hour baseline, gradual ToMe untrained, low-resolution queries with and without the linear map,
   3-stage TBSD, no-distillation control. All exist except seed 2.
5. **Efficiency measured, not only analytic:** latency/throughput of low-resolution and gradual-merge queries
   (`bench_latency.py`; GPU batch 1 and 256, CPU). Not yet run for these two settings.
6. **Stated limitations:** one dataset; native-resolution caveat; one backbone unless phase D is run;
   full-token deficit; class-attention result unexplained.
7. Report structure: methodology, results, comparisons, limitations, conclusions, potential contributions.

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

## Remaining tasks (suggested order) — SUPERSEDED: iteration-1 list, kept as history
**Use the "Remaining experiments" table and "What to do first when resuming" at the top of this file instead.**
Of the list below: 1, 2, 4, the low-data part of 5 and the latency and ViT-B parts of 6 are done; 3 is still
unexplained; the second dataset is blocked on sign-up; 7 continues (literature); 8 is replaced by
`REPORT_ITERATION1.md`.
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
