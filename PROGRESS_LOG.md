# PROGRESS_LOG.md

Append-only log. Newest session at the bottom. Each entry: what ran (exact command), outputs, numbers, issues, next step.
Numbers from synthetic data (`fake_market`) are install checks only, never results.

---

## Session 1 (Claude Code) — 2026-11-14 ~18:45–19:30 Dhaka

**Starting point:** kit written in the earlier chat and only CPU-tested on synthetic data; no GPU run, no Market-1501
run (PROJECT_CONTEXT.md §8). Left off at §10 A step 1 (environment setup).

### Done
- Confirmed OS: Windows 11 Pro 10.0.26200. Python 3.13.17.
- Created project venv `.venv`: torch 2.15.0+cu132, torchvision 0.29.2+cu132, timm 1.0.31, numpy, pillow, matplotlib.
  `torch.cuda.is_available()` = True on the RTX 5080. (Same torch build as `~/resnet-lab/.venv`.)
- Wrote `CLAUDE.md` (persistent instructions) and this log.
- GPU smoke tests on `fake_market` (synthetic), all ran end to end:
  - `train.py ... --out runs/smoke --epochs 1 --P 4 --K 2 --eval-every 1` (baseline)
  - `... --out runs/smoke_tbsd --epochs 2 --sd-mode merge --sd-keep random` (TBSD; sd loss 0.064 -> 0.001)
  - `... --out runs/smoke_low --id-fraction 0.5 --train-mode cls_attn --train-keep 0.7`
  - `evaluate.py` sweep on both checkpoints; CSV + PNG written; `keep=1.0` rows equal `none` rows (check passed);
    GFLOPs 2.930 / 1.921 / 1.463 at keep 1.0 / 0.7 / 0.5 (as expected).
- `bench_step.py` (new): compute-only step timing on random input, ViT-S, batch 64, AMP:
  baseline 24.8 ms/step (~0.17 h for 120 Market epochs, compute only), peak VRAM 1.80 GB;
  TBSD 47.9 ms/step (~0.32 h), peak 2.90 GB. Real runs will be longer because of JPEG decoding/augmentation
  on 4 workers; time the first real run.

### Bug/issue found and fixed: token reduction was slower than full tokens on GPU
- Reduction layers and all post-merge blocks used the explicit-attention path (`attention_with_maps`), while
  full-token blocks use timm's fused SDPA. Result: merge keep 0.7 ran at 6.7k img/s vs 9.3k full (batch 64).
- Fix (`common.py`): new `attention_fast` uses `F.scaled_dot_product_attention` with `log(size)` as an additive
  bias and computes only the class-token attention row when `cls_attn` needs it. `ReIDViT.fast_attn = True` by
  default; `False` keeps the old reference path. Max difference vs reference: 1.8e-5 (fp32, BNNeck output).
- Second finding: at batch 64 ViT-S on the 5080 is launch-overhead bound, so reduction barely helps wall-clock.
  Throughput with fast path (img/s, AMP, forward only):

  | batch | none | merge 0.7 | merge 0.5 | cls_attn 0.7 | random 0.5 |
  | --- | --- | --- | --- | --- | --- |
  | 64 | 9272 | 7572 | 7578 | 9651 | 10174 |
  | 256 | 8531 | 12275 | 15238 | 12988 | 16460 |

  So `evaluate.py` now times throughput at batch 256 on GPU (`--tp-batch` to change). For the paper: report
  latency at batch 1 and at a large batch, and on CPU; GFLOPs savings do not translate to small-batch GPU speed.
  Merge also has overhead over plain selection (15.2k vs 16.5k at keep 0.5).
- `evaluate.py` now loads checkpoints with `weights_only=True` explicitly.

### Known issues noted (not fixed)
- TBSD's reduced pass runs BNNeck in train mode, so BN running stats mix full- and reduced-token features;
  full-token evaluation uses those mixed stats. Check on real data; option: separate BN for reduced pass, or
  eval-mode BN for the second pass.
- HF Hub warns about symlinks on Windows (harmless; set `HF_HUB_DISABLE_SYMLINKS_WARNING=1`).

### Blocker
- **Market-1501 is not on this machine** (searched home, Downloads, D:). Downloading needs the user's OK /
  the user to fetch it. Once present: `bash run_iter1.sh <Market root>` runs §10 A steps 4–8 in order,
  logging to `runs/<name>/stdout.txt`, skipping finished runs.

### Next
1. Get Market-1501 -> run `run_iter1.sh` -> fill results into the report (interpretation rules: PROJECT_CONTEXT §10 A.9).
2. Check baseline mAP lands in the 80s (sanity); then H1 diagnosis from `runs/base_s0/sweep.csv`.

---

## Session 1, continued — first Market-1501 results (2026-11-14, local clock BST 19:00–)

### Dataset
- User approved the download. Source: official Zheng lab page (zheng-lab-anu.github.io/Project/project_reid.html),
  Google Drive link. `Market-1501-v15.09.15.zip`, 152,733,771 bytes (matches Academic Torrents listing).
  Extracted to `C:/Users/user/data/Market-1501-v15.09.15` (zip kept in `C:/Users/user/data/dl/`).
- Counts: train 12936 imgs / 751 IDs, query 3368 / 750 IDs, gallery 19732 (15913 after dropping pid -1 junk). Standard.
- Cite Zheng et al., ICCV 2015 if used.

### Queue
`bash run_iter1.sh /c/Users/user/data/Market-1501-v15.09.15 > runs/queue_iter1.txt` (background). Order:
base_s0 -> sweep -> tbsd_s0 -> sweep -> base_id25_s0 -> sweep -> tbsd_id25_s0 -> sweep.

### base_s0 (ViT-S/16, ImageNet-21k->1k augreg weights, 120 ep, seed 0) — VERIFIED RUN, single seed
- mAP/R1 by epoch: 20: 81.3/92.5, 40: 82.5/93.5, 60: 83.1/93.4, 80: 83.7/93.2, 100: 83.8/93.4, **120: 83.9/93.5**.
- Train time 0.22 GPU-h (13.4 min compute; ~6.2 s/epoch), peak VRAM 1.81 GB. Sanity check passed (80s).

### base_s0 sweep (`runs/base_s0/sweep.csv`, `.png`), mAP; img/s at batch 256 AMP
| mode | keep | GFLOPs | img/s | sym mAP | asym mAP | q_cos |
| --- | --- | --- | --- | --- | --- | --- |
| none | 1.0 | 2.93 | 8252 | 83.89 | 83.89 | 1.000 |
| merge | 0.8 | 2.22 | 11146 | 82.46 | 83.14 | 0.980 |
| merge | 0.7 | 1.92 | 12896 | 80.29 | 81.87 | 0.952 |
| merge | 0.6 | 1.68 | 14453 | 75.73 | 79.13 | 0.897 |
| merge | 0.5 | 1.46 | 16066 | 58.69 | 67.40 | 0.729 |
| cls_attn | 0.7 | 1.92 | 13652 | 61.34 | 69.65 | 0.778 |
| cls_attn | 0.5 | 1.46 | 15151 | 18.58 | 31.29 | 0.452 |
| random | 0.7 | 1.92 | 13896 | 67.52 | 77.29 | 0.848 |
| random | 0.5 | 1.46 | 17297 | 27.55 | 55.77 | 0.601 |

### Interpretation (single seed, training-free reduction only)
1. **H1 as originally framed is not confirmed.** Asymmetric mAP is *higher* than symmetric at every budget
   (full-token gallery helps). The real cost is the gap between asymmetric and full-token mAP
   (merge 0.6: 79.1 vs 83.9, -4.8; merge 0.5: -16.5), which tracks query-embedding drift (q_cos 0.90, 0.73).
   TBSD's test: shrink that gap (raise q_cos) without losing full-token mAP.
2. **Class-token attention selection (EViT-style, no fused token) is worse than random selection** at every
   keep <= 0.8 (e.g. keep 0.7 sym 61.3 vs random 67.5). Unexpected; candidate gap ("which tokens matter for
   re-ID identity is not what CLS attention says"). Check first: omitted fused token, attention from a single
   layer, possible attention "sink"/register tokens in augreg ViT. Must confirm vs official EViT before claiming.
3. Merge is clearly the best training-free rule; it keeps >= 98% of mAP down to keep 0.8 (1.32x GFLOPs saving,
   1.35x throughput) and degrades sharply below 0.6.

### tbsd_s0 (`--sd-mode merge --sd-keep random`, 120 ep, seed 0) — VERIFIED RUN, single seed
- Full-token mAP/R1 at 120: **83.8 / 93.9** (baseline 83.9 / 93.5: no loss, difference inside expected noise).
- Train time **0.40 GPU-h vs 0.22** for baseline (1.8x, two forward passes per step). sd loss 0.030 -> 0.015.

| mode | keep | GFLOPs | base sym | TBSD sym | base asym | TBSD asym | base q_cos | TBSD q_cos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| merge | 0.8 | 2.22 | 82.46 | 83.66 | 83.14 | 83.74 | 0.980 | 0.996 |
| merge | 0.7 | 1.92 | 80.29 | 83.34 | 81.87 | 83.56 | 0.952 | 0.990 |
| merge | 0.6 | 1.68 | 75.73 | 82.65 | 79.13 | 83.14 | 0.897 | 0.978 |
| merge | 0.5 | 1.46 | 58.69 | **80.13** | 67.40 | **81.90** | 0.729 | 0.944 |
| cls_attn | 0.7 | 1.92 | 61.34 | 68.71 | 69.65 | 74.65 | 0.778 | 0.846 |
| random | 0.7 | 1.92 | 67.52 | 77.61 | 77.29 | 80.96 | 0.848 | 0.932 |
| random | 0.5 | 1.46 | 27.55 | 59.14 | 55.77 | 72.41 | 0.601 | 0.791 |

Interpretation (one seed; needs 3 seeds before any claim):
- **TBSD passes its first test.** With merging, one set of weights keeps 95.5% of full-token mAP at 50% of the
  backbone GFLOPs (asym 81.9 vs 83.8), vs 80% for the baseline. Gain grows as budget shrinks (+0.6 at 0.8,
  +14.5 at 0.5 asym; +21.4 sym). q_cos rises accordingly: the reduced embedding stays in the full-token space.
- Gains transfer partly to reduction rules not seen in training (cls_attn, random), so TBSD is not only
  memorising the merge rule. Ranking of rules is unchanged (merge > random > cls_attn).
- With TBSD, asym-sym gap shrinks to ~1.8 at keep 0.5 (base 8.7).
- Fairness to-do: TBSD used 1.8x GPU-hours. Needed controls: (a) baseline trained ~1.8x longer;
  (b) baseline fine-tuned with merged tokens (single-budget, no distillation) at same cost;
  (c) TBSD without the sd term (only multi-budget CE+triplet) to isolate the distillation's contribution;
  (d) BN running-stat issue (see known issues).

### Low-data pair (`--id-fraction 0.25`, 188 IDs / 3178 imgs, seed 0) — VERIFIED RUNS, single seed
- base_id25_s0: full-token mAP/R1 **71.6 / 87.2**, 0.06 GPU-h. tbsd_id25_s0: **70.0 / 86.1**, 0.10 GPU-h.

| mode | keep | base sym | TBSD sym | base asym | TBSD asym | base q_cos | TBSD q_cos |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none | 1.0 | 71.65 | 70.02 | 71.65 | 70.02 | 1.000 | 1.000 |
| merge | 0.7 | 67.40 | 69.23 | 68.98 | 69.59 | 0.946 | 0.986 |
| merge | 0.5 | 43.92 | 63.83 | 51.24 | 66.36 | 0.711 | 0.929 |
| cls_attn | 0.7 | 56.99 | 53.78 | 62.36 | 57.58 | 0.853 | 0.858 |
| random | 0.5 | 19.96 | 42.11 | 42.82 | 56.18 | 0.610 | 0.795 |

- **H3 (TBSD as low-data regulariser) not supported** in this run: full-token mAP -1.6 with TBSD (one seed; may be
  noise, but no sign of a gain). Budget robustness still holds at low data (merge 0.5 asym +15.1).
- With 25% IDs, TBSD *hurts* cls_attn (-4.8 asym at 0.7) — transfer to unseen rules is not reliable.

### State at end of queue (2026-11-14 20:40 BST local clock)
- All 4 runs + 4 sweeps done; queue log `runs/queue_iter1.txt`. Summary figure: `runs/iter1_summary.png`
  (`plot_results.py`). Total GPU time for the 4 trainings: 0.78 h.
- Headline (seed 0): TBSD keeps full-token mAP (83.8 vs 83.9) and gives 81.9 vs 67.4 mAP at 50% backbone GFLOPs
  (merge keep 0.5, full-token gallery). Cost: 1.8x training time.
- Hypothesis status: H1 reframed (asym > sym; the problem is query drift vs full-token, which TBSD fixes);
  H2 untested (no second dataset); H3 not supported; H4 untested.
- Iteration-1 report (cloud doc) NOT yet updated with these results.

### Next (priority order)
1. Seeds 1, 2 for base and TBSD (full IDs) -> mean/std. ~1.3 GPU-h.
2. Controls for fairness: baseline at ~216 epochs (equal GPU-h); TBSD with `--sd-weight 0` (multi-budget
   training without distillation) to isolate the distillation term; fixed-budget `--train-mode merge --train-keep 0.5`.
3. Investigate cls_attn < random (official EViT with fused token; attention-sink check).
4. Fix/ablate BNNeck running-stat mixing in TBSD.
5. Second dataset for H2 (MSMT17 or CUHK03 loader; needs download approval). Latency at batch 1 and CPU.

---

## Session 1, continued — seeds 1–2 and Git (2026-11-14, 21:34–23:40 local)

- Queue `run_iter1b.sh` (log `runs/queue_iter1b.txt`): base_s1, tbsd_s1, base_s2, tbsd_s2 done with sweeps.
  Full-token mAP/R1: base_s1 84.7/94.2, tbsd_s1 84.1/93.4, base_s2 84.8/94.2, tbsd_s2 84.0/93.6.
  GPU-h: base 0.22/0.23, TBSD 0.45/0.39.
- 3-seed aggregate (`aggregate.py`, `runs/iter1_seeds.csv/.png`), merge, mean ± std mAP:
  full tokens: base 84.45±0.49, TBSD 83.95±0.13. Keep 0.5 full-token gallery: base 68.56±1.14, TBSD 81.98±0.18;
  reduced gallery: 59.60±1.12 vs 80.26±0.13. q_cos at 0.5: 0.733 vs 0.944.
- **Conclusion (3 seeds):** TBSD's budget robustness is a robust effect. It costs ~0.5 full-token mAP (lower in
  all three seeds) and ~1.85x training time. Controls (equal GPU-h baseline, no-distillation) still running.
- Git: repo initialised, private GitHub remote https://github.com/berryseed/reid-research (gh CLI 2.104.0
  installed via winget; user signed in as berryseed). Commits a47b61b (initial), e1da769 (base_s2).

### Control: equal-GPU-hour baseline `base_e216_s0` (216 epochs, seed 0) — VERIFIED, single seed
- Full-token mAP/R1 **85.3 / 94.5**, 0.36 GPU-h (TBSD mean 0.41 h). Merge sweep, full-token gallery mAP:
  0.8: 84.61, 0.7: 83.55, 0.6: 81.25, 0.5: **69.02** (q_cos 0.716). Reduced gallery at 0.5: 59.40.
- vs TBSD 3-seed mean (83.95 full; 83.89/83.70/83.29/81.98 at 0.8/0.7/0.6/0.5):
  at equal training compute the baseline is **+1.3 mAP at full tokens** but loses from keep ~0.7 down
  (-2.0 at 0.6, -13.0 at 0.5). Longer training does not fix budget robustness (q_cos at 0.5 unchanged ~0.72).
- Honest framing: TBSD trades ~1 mAP of full-token accuracy (at matched compute) for large gains at low budgets.

### Control: TBSD without distillation `tbsd_nosd_s0` (`--sd-weight 0`, seed 0) — VERIFIED, single seed
- Full-token 83.6 / 93.8, 0.41 GPU-h. Merge, full-token gallery / reduced gallery mAP:
  0.8: 83.46/83.37, 0.7: 83.13/82.78, 0.6: 82.48/81.67, 0.5: **80.62/78.37**; q_cos at 0.5: 0.903 (TBSD 0.944).
- Most of the budget robustness comes from multi-budget training (reduced pass with its own ID + triplet);
  the cosine distillation term adds ~+1.4 / +1.9 mAP at keep 0.5. Needs seeds 1–2 before any claim.
- Queue `run_iter1b.sh` finished 00:38 (all runs complete).

## Session 1 end — project paused by the user (2026-11-15 ~00:45 local)
- Added to `train.py`: per-epoch resumable `ckpt.pth` (atomic write; model, optimizer, scaler, epoch, step,
  train time, log, args, Python/NumPy/torch/CUDA/sampler RNG states), `--resume`, `--ckpt-every`, `--keep-ckpt`,
  graceful pause via `<out>/STOP` (exit code 3). Queue scripts now pass `--resume`, append to stdout.txt,
  and exit on code 3.
- Verification: synthetic data, CPU, workers 0, TBSD config, 3 epochs: run paused after epoch 1 and resumed
  -> max weight difference vs uninterrupted run = 0.0, identical losses/mAP. Resume refuses mismatched args.
- All 10 final `last.pth` checkpoints load into `ReIDViT` and give finite 384-d embeddings.
- No training was interrupted; nothing to resume. No python processes left; GPU idle.

---

## Session 2 (Claude Code) — 2026-11-15, started 10:25 local

**Resume state:** no python processes, GPU idle, `git status` clean, all 10 iteration-1 runs have `last.pth`,
no `ckpt.pth` anywhere -> nothing to restore; continued from "Remaining tasks" in PROGRESS.md.
User's standing autonomy instructions saved in CLAUDE.md (commit ebf5913).

### Decisions (reasoning)
- Order of GPU work = how much each item changes the paper's claim: (A) seeds for the no-distillation control
  and the BNNeck ablation decide what the method *is*; (B) fixed-budget and equal-GPU-h controls with 3 seeds make
  the comparison fair; (C) low-data at 10/25/50% tests H3 properly; (D) ViT-B answers the open backbone question.
- BNNeck ablation implemented as `--sd-bn nostat` (reduced pass normalises with batch stats, momentum 0 so running
  stats come from the full-token pass only). Chosen over a separate BN for the reduced pass because a second BN
  would make reduced and full embeddings pass through different affine maps, which works against compatibility.
- Selection diagnostics are eval-only on the six existing checkpoints: `cls_attn_fuse` (EViT's fused inattentive
  token, the piece missing from our `cls_attn`) and `cls_attn_low` (keep the LEAST attended tokens; if this is
  not much worse than `cls_attn`, class attention carries little identity information at these layers).
- H2 (cross-dataset) needs a second dataset. Not downloaded: the dataset rule in CLAUDE.md still requires the
  user's OK, and MSMT17 needs a signed release agreement. Deferred, listed as open.
- OSNet reference: not in timm; would need torchreid weights from a third-party Google Drive. Deferred, listed.

### Code (commit after ebf5913; verified on synthetic data before the queue started)
- `train.py --sd-bn {shared,nostat}`; resume check extended to sd_weight/sd_bn/sd_keep/train_keep.
- `common.py`: modes `cls_attn_fuse`, `cls_attn_low`. Checks: keep=1.0 equals `none` for all six modes; fast path
  vs reference path max diff 1.8e-6; `nostat` leaves `running_mean` bit-identical, `shared` changes it.
- `bench_latency.py`, `run_iter2.sh` (phases S A B C D), `aggregate2.py`.
- Queue: `bash run_iter2.sh /c/Users/user/data/Market-1501-v15.09.15 S A >> runs/queue_iter2.txt` started 10:29.

### Novelty check, round 2 (8 web searches, abstracts/summaries only, NOT full reads)
New closest work found (add to the literature matrix; verify by full read before citing):
- **SFSC, "Switchable Representation Learning Framework with Self-Compatibility", CVPR 2023, arXiv 2206.08289.**
  One training run yields sub-models of different capacity (switchable CNN widths) whose features are mutually
  compatible; handles gradient conflict between sub-models. Evaluated on re-ID-style visual search. This is the
  closest conceptual prior: "one network, several budgets, compatible embeddings". Difference: budget = channel
  width of a CNN, not tokens; no token reduction, no cost-of-training accounting. **This lowers novelty of the
  "self-compatible budgets" idea itself; ours is the token-budget instance of it for ViT re-ID.**
- **Asymmetric image retrieval: CSD (CVPR 2022), D3still (CVPR 2024), arXiv 2403.00648 / 2403.00671.** Small query
  network distilled to be compatible with a large gallery network. Two separate networks; landmark retrieval.
- **DaRe, "Resource Aware Person Re-identification across Multiple Resolutions", CVPR 2018, arXiv 1805.08805.**
  Budgeted/anytime re-ID with one model via early exits (depth), not tokens.
- **FPC, "Dynamic Feature Pruning and Consolidation for Occluded Person Re-ID", AAAI 2024, arXiv 2211.14742.**
  Token sparsification of the query, matched against holistic gallery features: a query/gallery asymmetry
  precedent inside re-ID (occlusion-driven, not budget-driven).
- **STM-ReID, "Lighter is Better ... Spatial-Aware Token Merging"** (OpenReview 8Bs3mz49Gp, ICLR 2026 submission,
  withdrawn): training-free merging for ViT re-ID.
- **MergeTok (arXiv 2605.30904):** samples the merge ratio at random during training for robustness to token
  granularity (visual tokenizer, not retrieval). Relevant because our no-distillation control shows random-ratio
  training is where most of the gain comes from.
- **Haurum et al., "Which Tokens to Use?", ICCVW 2023, arXiv 2308.04657:** on classification, Top-K is a strong
  baseline and EViT (Top-K + fused token) beats it. **"When Token Pruning is Worse than Random" (Wang et al.,
  2026, VLMs):** attention-based pruning no better than random in deep layers. Context for cls_attn < random.
- No paper found that trains one ViT for several *token* budgets with cross-budget-compatible *retrieval*
  embeddings and evaluates asymmetric (full gallery, reduced query) re-ID. "Not found in ~20 searches" only.

### Power outage and recovery (2026-11-15; outage ~12:33, recovery 14:59–15:15)
- Before the outage (from `runs/queue_iter2.txt`): phase S finished 11:11 (6 selection sweeps, `runs/latency.csv`);
  `tbsd_nosd_s1` (83.9/94.2, 0.39 h) and `tbsd_nosd_s2` (83.6/93.5, 0.40 h) trained and swept;
  `tbsd_bnns_s0` started 12:12, last stdout line "epoch 94/120 ... 18.8 min", `ckpt.pth` written 12:33.
- Checks run: `git fsck` clean, `main` == `origin/main` (554cc1f); no python processes; GPU idle; no `*.tmp`;
  `torch.load` of all 16 `last.pth` (finite weights) and of `ckpt.pth` (epoch 94, step 18988, 94 log epochs,
  evals 80.1/81.5/82.7/83.4 at 20/40/60/80); all `log.json` parse; all CSVs have complete rows; all PNGs verify.
- Nothing corrupted. Lost: partial epoch 95 of `tbsd_bnns_s0` only.
- Committed the finished results, added `save()` to `run_iter2.sh` (commit + push after every finished run),
  resumed with `bash run_iter2.sh /c/Users/user/data/Market-1501-v15.09.15 A >> runs/queue_iter2.txt` at 15:02;
  stdout confirms `resumed from runs/tbsd_bnns_s0\ckpt.pth: epoch 94 done, step 18988, 0.31 h trained`.

### Phase S results (eval only, 3 seeds each, mean ± std mAP; `runs/*/sweep_sel.csv`)
Keeps 0.8 / 0.7 / 0.5 (GFLOPs 2.22 / 1.92 / 1.46; fused variant 2.24 / 1.95 / 1.49).

| checkpoints | rule | reduced gallery | full-token gallery |
| --- | --- | --- | --- |
| baseline | random | 77.40±0.84 / 68.13±0.53 / 27.19±0.47 | 81.40±0.61 / 77.99±0.61 / 56.37±0.68 |
| baseline | cls_attn | 78.31±2.15 / 66.24±5.11 / 23.62±5.70 | 80.53±1.56 / 72.66±3.29 / 36.32±5.17 |
| baseline | cls_attn_fuse | 79.04±1.77 / 68.74±3.84 / 28.05±5.70 | 80.93±1.43 / 73.82±2.99 / 39.19±5.07 |
| baseline | cls_attn_low | 57.22±2.25 / 39.11±3.84 / 12.75±1.99 | 57.73±4.80 / 37.62±6.26 / 8.50±2.28 |
| TBSD | random | 81.01±0.22 / 77.78±0.22 / 58.98±0.55 | 82.60±0.22 / 81.11±0.15 / 72.40±0.04 |
| TBSD | cls_attn | 78.09±0.67 / 67.74±2.20 / 30.04±4.73 | 80.34±0.59 / 73.33±1.99 / 41.41±6.27 |
| TBSD | cls_attn_fuse | 78.89±0.49 / 69.84±1.78 / 33.90±4.72 | 80.87±0.42 / 74.82±1.65 / 45.22±6.17 |
| TBSD | cls_attn_low | 68.92±1.11 / 55.60±1.90 / 25.56±3.14 | 73.47±1.06 / 62.06±1.71 / 28.32±2.82 |

Reading: (1) "cls_attn < random" holds over 3 seeds for a full-token gallery (keep 0.7: -5.3, keep 0.5: -20), but
with both sides reduced the two are within noise at 0.8/0.7; (2) the missing fused token explains only about 1 mAP;
(3) inverse selection is far worse, so class attention is informative and this is not an inverted-attention bug;
(4) class-attention selection has 3-8x the seed variance of random; (5) TBSD trained with merging transfers to
random selection (+16 at keep 0.5, full-token gallery) but barely to class-attention selection.
Still our own implementation, not official EViT code. Why attention-selected tokens shift the embedding more than
a uniform sample is not explained.

### Latency (`runs/latency.csv`; forward only, random input, ViT-S 256x128; GPU AMP, CPU fp32 8 threads; idle machine)
| rule | keep | GFLOPs | GPU b1 ms | GPU b256 ms/img | CPU b1 ms | CPU b8 ms/img |
| --- | --- | --- | --- | --- | --- | --- |
| none | 1.0 | 2.93 | 6.00 | 0.111 | 16.3 | 11.1 |
| merge | 0.8 | 2.22 | 8.47 | 0.089 | 15.1 | 7.1 |
| merge | 0.7 | 1.92 | 8.47 | 0.077 | 13.6 | 7.4 |
| merge | 0.6 | 1.68 | 8.51 | 0.069 | 12.7 | 5.9 |
| merge | 0.5 | 1.46 | 8.55 | 0.062 | 11.9 | 6.2 |
| random | 0.5 | 1.46 | 6.59 | 0.058 | 11.0 | 5.5 |

Reading: 2.0x fewer GFLOPs gives 1.78x at GPU batch 256 and CPU batch 8, 1.37x at CPU batch 1, and a slowdown at
GPU batch 1 (merge 8.5 ms vs 6.0 ms). Token reduction pays off for throughput and on CPU, not for single-image GPU
inference. CPU batch-8 numbers come from 10 iterations and are not monotonic (noise about 10%).

### No-distillation control, 3 seeds (`tbsd_nosd_s0..2`), merge, mAP full-token gallery / reduced gallery
| | GPU-h | keep 1.0 | 0.8 | 0.7 | 0.6 | 0.5 | q_cos@0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Multi-budget, no distillation | 0.40 | 83.71±0.20 | 83.53±0.19 / 83.37±0.18 | 83.19±0.16 / 82.80±0.14 | 82.51±0.11 / 81.63±0.10 | 80.56±0.06 / 78.31±0.12 | 0.902 |
| TBSD | 0.41 | 83.95±0.13 | 83.89±0.13 / 83.84±0.15 | 83.70±0.12 / 83.48±0.12 | 83.29±0.13 / 82.71±0.06 | 81.98±0.18 / 80.26±0.13 | 0.944 |

Conclusion: the cosine distillation term is a real but modest contribution (+1.42 / +1.95 mAP at keep 0.5, well
outside seed spread; +0.24 at full tokens, not significant) at no measurable extra training cost. Most of the budget
robustness (+12.0 of +13.4 over the baseline at keep 0.5) comes from multi-budget training.

### Phase A finished 16:11 — BNNeck running-stat ablation, 3 seeds (`tbsd_bnns_s0..2`, `--sd-bn nostat`)
`tbsd_bnns_s0` is the run resumed after the outage (epochs 95-120 after resume). mAP, merge, full-token / reduced gallery:

| | GPU-h | keep 1.0 | 0.8 | 0.7 | 0.6 | 0.5 |
| --- | --- | --- | --- | --- | --- | --- |
| TBSD (shared BN stats) | 0.41 | 83.95±0.13 | 83.89±0.13 / 83.84±0.15 | 83.70±0.12 / 83.48±0.12 | 83.29±0.13 / 82.71±0.06 | 81.98±0.18 / 80.26±0.13 |
| TBSD, stats from full pass only | 0.40 | 83.83±0.20 | 83.79±0.18 / 83.72±0.19 | 83.60±0.19 / 83.35±0.19 | 83.22±0.22 / 82.65±0.24 | 81.99±0.19 / 80.33±0.23 |

**Negative result:** no difference at any budget (all within seed spread). BNNeck running-stat mixing is NOT the
cause of TBSD's ~0.5 mAP full-token deficit against the 120-epoch baseline; the hypothesis is rejected and the
default (`shared`) stays. The deficit is shared by the no-distillation variant (83.71), so it comes from
multi-budget training itself (the reduced pass's ID/triplet gradients), not from the distillation term or BN.
Phase B started 16:12: `bash run_iter2.sh ... B`.


### Phase B results (3 seeds; `fix50_s0..2`, `base_e216_s0..2`), merge 3-stage, mAP full-token gallery / reduced gallery
| | GPU-h | VRAM | keep 1.0 | 0.8 | 0.7 | 0.6 | 0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline 216 ep | 0.38 | 1.81 | 85.11±0.15 | 84.48±0.15 / 83.95±0.12 | 83.42±0.12 / 82.12±0.06 | 81.08±0.16 / 78.31±0.18 | 69.76±0.73 / 60.38±0.94 |
| Fixed-budget merge 0.5 | 0.24 | 1.13 | 80.68±0.45 | 80.59±0.44 / 80.56±0.43 | 80.43±0.45 / 80.34±0.43 | 80.04±0.40 / 79.70±0.37 | 79.24±0.39 / 78.45±0.42 |

### Phase C results (low data, 3 seeds; finished 21:47), mAP full-token gallery
| IDs | config | GPU-h | keep 1.0 | keep 0.7 | keep 0.5 | q_cos@0.5 |
| --- | --- | --- | --- | --- | --- | --- |
| 10% | baseline | 0.03 | 55.88±1.83 | 52.62±1.79 | 36.96±0.99 | 0.744 |
| 10% | TBSD | 0.04 | 53.72±0.20 | 53.02±0.08 | 49.35±0.03 | 0.929 |
| 25% | baseline | 0.06 | 71.06±0.63 | 68.39±0.63 | 51.39±0.80 | 0.715 |
| 25% | TBSD | 0.10 | 70.11±0.73 | 69.63±0.79 | 66.87±1.03 | 0.932 |
| 50% | baseline | 0.12 | 78.66±0.47 | 76.11±0.53 | 59.94±1.44 | 0.716 |
| 50% | TBSD | 0.21 | 77.76±0.46 | 77.42±0.45 | 75.30±0.50 | 0.940 |
H3 (regulariser under low data) not supported. Full table: `runs/iter2_summary.csv`, plot `runs/iter2_lowdata.png`.

### Direction review and literature pass (2026-11-15 evening)
- User instruction: TBSD is exploratory; prioritise full reads, gap reassessment, mechanism, own methodology;
  second dataset approved subject to verifying source/licence/access.
- Papers: `pip install pypdf` into `.venv`; PDFs of SFSC, D3still, CSD, SUReID converted to text in the scratchpad
  (not in the repo). Result: `LITERATURE_MATRIX.md`. Corrections: SUReID real title; MergeTok irrelevant (removed);
  "worse than random" paper is about VLLMs.
- Dataset vetting: CUHK03 official page links a Google Form; Occluded-REID repo has a user agreement; MSMT17 by
  request. Nothing downloaded.
- Official code read: facebookresearch/ToMe `tome/merge.py`, `tome/patch/timm.py`; youweiliang/evit `evit.py`
  (saved in the scratchpad, not executed: running downloaded code was blocked by the permission system).
  `check_reduction.py` compares ours with a transcription: merge diff 0, top-k diff 0, fused token same direction,
  different norm.

### Code changes (commit 64ac40e)
- `common.py`: modes `merge_r` (`model.tome_r` tokens merged in every block, cap half) and `cls_attn_evit`;
  `select_tokens_fuse(..., normalize=)`; `forward/backbone(..., hw=None)` low-resolution input with resampled
  position embedding. Checks: same-size `hw` diff 0.0; `merge@0.5` on base_s0 reproduces 67.42 (sweep 67.4).
- `train.py`: `--sd-mode lowres | merge_r | mixed`; `SD_RES`, `SD_RS`. Smoke: `runs/smoke_{lowres,merge_r,mixed}`.
- New: `mechanism.py`, `aggregate_mech.py`, `check_reduction.py`, `run_iter3.sh`, `METHOD_CANDIDATES.md`.

### Mechanism diagnostics (`runs/mechanism.csv`, 3 seeds each; asymmetric mAP, full-token gallery)
Command: `.venv/Scripts/python.exe mechanism.py --data C:/Users/user/data/Market-1501-v15.09.15 --ckpts runs/{base,tbsd}_s{0,1,2}/last.pth --out runs/mechanism.csv`;
table: `.venv/Scripts/python.exe aggregate_mech.py`. Corrections fitted on 6,468 unlabeled training images.

| group | query setting | GFLOPs | q_cos | offset frac | raw | mean fix | ridge map |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base | merge 3-stage 0.7 | 1.92 | 0.953 | 0.044 | 82.50±0.55 | 82.58 | 82.64 |
| base | gradual ToMe r=7 | 1.94 | 0.956 | 0.038 | 82.79±0.71 | 82.83 | 82.84 |
| base | random 0.7 | 1.92 | 0.851 | 0.028 | 78.06±0.63 | 78.21 | 78.21 |
| base | cls_attn 0.7 | 1.92 | 0.815 | 0.044 | 72.66±3.29 | 73.19 | 74.87 |
| base | cls_attn avg-fuse 0.7 | 1.95 | 0.829 | 0.044 | 73.82±2.99 | 74.31 | 75.72 |
| base | cls_attn EViT-fuse 0.7 | 1.95 | 0.819 | 0.050 | 72.93±3.28 | 73.49 | 75.05 |
| base | merge 3-stage 0.5 | 1.46 | 0.733 | 0.066 | 68.55±1.11 | 69.99 | 72.70 |
| base | gradual ToMe r=11 | 1.41 | 0.836 | 0.055 | 77.45±1.39 | 77.86 | 78.65 |
| base | random 0.5 | 1.46 | 0.595 | 0.069 | 56.81±0.87 | 58.57 | 61.19 |
| base | cls_attn EViT-fuse 0.5 | 1.49 | 0.513 | 0.080 | 37.51±4.69 | 40.92 | 46.27 |
| base | merge 0.5 at block 4 only | 1.87 | 0.917 | 0.064 | 80.65±0.88 | 80.90 | 81.14 |
| base | merge 0.5 at block 7 only | 2.24 | 0.969 | 0.043 | 83.22±0.45 | 83.28 | 83.27 |
| base | merge 0.5 at block 10 only | 2.61 | 0.995 | 0.054 | 84.24±0.47 | 84.25 | 84.17 |
| base | low-res 224x112 | 2.22 | 0.943 | 0.060 | 83.67±0.49 | 83.75 | 83.71 |
| base | low-res 192x96 | 1.62 | 0.893 | 0.128 | 81.38±0.36 | 81.76 | 82.13 |
| base | low-res 160x80 | 1.12 | 0.789 | 0.159 | 75.81±0.43 | 77.09 | 78.58 |
| base | low-res 128x64 | 0.72 | 0.609 | 0.175 | 60.13±2.68 | 63.88 | 69.00 |
| tbsd | merge 3-stage 0.5 | 1.46 | 0.944 | 0.042 | 81.99±0.18 | 82.03 | 81.77 |
| tbsd | gradual ToMe r=11 | 1.41 | 0.942 | 0.055 | 81.77±0.18 | 81.89 | 81.87 |
| tbsd | random 0.5 | 1.46 | 0.793 | 0.087 | 72.58±0.33 | 73.30 | 73.57 |
| tbsd | low-res 192x96 | 1.62 | 0.922 | 0.168 | 81.91±0.32 | 82.14 | 82.34 |
| tbsd | low-res 160x80 | 1.12 | 0.844 | 0.209 | 78.26±0.38 | 78.99 | 79.90 |
| tbsd | low-res 128x64 | 0.72 | 0.714 | 0.232 | 69.26±0.90 | 71.69 | 74.21 |

Low-resolution queries here are downsized from the 256x128 network input by bicubic interpolation with
antialiasing; the patch-embedding cost of the removed patches is subtracted from the GFLOPs estimate.

### Iteration-3 queue launched 22:02
`bash run_iter3.sh /c/Users/user/data/Market-1501-v15.09.15 M E D >> runs/queue_iter3.txt 2>&1`
Decision recorded: ViT-B reference moved behind the second-pass variants, because the diagnostics made the
resolution-vs-merge comparison more decisive than validating the 3-stage TBSD on a larger backbone.


### Control diagnostics, phase M of run_iter3.sh (3 seeds; `runs/{base_e216,tbsd_nosd}_s*/mech.csv`), asymmetric mAP
| group | query setting | GFLOPs | q_cos | raw | ridge map |
| --- | --- | --- | --- | --- | --- |
| baseline 216 ep | gradual ToMe r=7 | 1.94 | 0.961 | 83.80±0.04 | 83.75±0.03 |
| baseline 216 ep | merge 3-stage 0.5 | 1.46 | 0.729 | 69.75±0.72 | 73.56±0.42 |
| baseline 216 ep | gradual ToMe r=11 | 1.41 | 0.849 | 78.97±0.12 | 79.70±0.13 |
| baseline 216 ep | low-res 192x96 | 1.62 | 0.890 | 82.06±0.19 | 82.80±0.23 |
| baseline 216 ep | low-res 160x80 | 1.12 | 0.770 | 75.72±0.51 | 78.96±0.18 |
| multi-budget, no distillation | merge 3-stage 0.5 | 1.46 | 0.902 | 80.56±0.07 | 80.56±0.16 |
| multi-budget, no distillation | gradual ToMe r=11 | 1.41 | 0.916 | 80.84±0.26 | 81.02±0.30 |
| multi-budget, no distillation | low-res 192x96 | 1.62 | 0.907 | 81.51±0.21 | 81.99±0.12 |
| multi-budget, no distillation | low-res 160x80 | 1.12 | 0.820 | 77.37±0.46 | 79.17±0.23 |

**Equal-GPU-hour reading (negative result for TBSD on Market-1501):** the 216-epoch baseline with low-resolution
queries plus the closed-form linear map reaches 82.80 at 1.62 GFLOPs and 78.96 at 1.12 GFLOPs; TBSD reaches
81.91-82.34 and 78.26-79.90 at the same costs, and is 1.16 mAP lower at full tokens (83.95 vs 85.11).
Linear interpolation at 1.46 GFLOPs (an estimate): about 80.0 raw / 81.6 with the map, vs TBSD 81.99.
Caveat: Market-1501 crops are natively 128x64; needs a higher-resolution dataset before generalising.


## 2026-11-16 01:00 — pause for PC shutdown (user request)
- Inspection 00:59: one run active, `sp_mixed_s1` (PID 11696/6276, started 00:46:41), epoch 59/120, 5.3 GB VRAM.
  Queue script PID 2872.
- Pause: `touch runs/sp_mixed_s1/STOP` at 00:59:46. `train.py` printed
  `paused after epoch 61 (STOP file)` and exited with code 3 at 00:59:54; `run_iter3.sh` printed
  `PAUSED sp_mixed_s1` and exited (code 3). STOP file removed by `train.py`. Nothing else started.
- Verification: `runs/sp_mixed_s1/ckpt.pth` (175.6 MB) loaded on CPU: keys model, optim, scaler, epoch=61,
  step=12322, train_seconds=681.2, log, args (seed 1, sd_mode mixed, 120 epochs), num_classes=751, rng; 156
  tensors finite; lr 0.004172. The five `runs/sp_*/last.pth` (83 MB each) load, finite, 751 classes.
  No python.exe left; GPU memory 1.3 GB (desktop only).
- Iteration-3 phase E timings from `runs/queue_iter3.txt`: sp_lowres_s0 0.36 h (mAP 82.8 / R1 93.0),
  sp_merge_r_s0 0.44 h (84.2 / 93.8), sp_mixed_s0 0.35 h (83.8 / 93.8), sp_lowres_s1 0.31 h (83.3 / 92.9),
  sp_merge_r_s1 0.40 h (84.3 / 94.7).
- Remaining: `sp_mixed_s1` (from epoch 62), `sp_lowres_s2`, `sp_merge_r_s2`, `sp_mixed_s2`, phase D
  (`vitb_base_s0`, `vitb_tbsd_s0`, `runs/latency_vitb.csv`).
  Resume: `bash run_iter3.sh /c/Users/user/data/Market-1501-v15.09.15 E D >> runs/queue_iter3.txt 2>&1`

### Partial phase E table (asymmetric mAP raw / with ridge map; per-seed values in brackets)
| setting | GFLOPs | sp_lowres (s0, s1) | sp_merge_r (s0, s1) | sp_mixed (s0) |
| --- | --- | --- | --- | --- |
| full tokens | 2.93 | 83.08 [82.83 83.33] | 84.25 [84.19 84.32] | 83.77 |
| merge 3-stage 0.7 | 1.92 | 82.13 / 82.06 | 83.73 / 83.59 | 83.17 / 82.94 |
| gradual ToMe r=7 | 1.94 | 82.22 / 82.09 | 84.06 / 83.86 | 83.55 / 83.30 |
| random 0.7 | 1.92 | 78.66 / 78.46 | 80.39 / 80.31 | 80.02 / 79.71 |
| cls_attn EViT-fuse 0.7 | 1.95 | 72.25 / 73.68 | 73.56 / 75.64 | 74.49 / 76.02 |
| merge 3-stage 0.5 | 1.46 | 74.94 / 76.28 | 79.66 / 79.95 | 79.11 / 79.20 |
| gradual ToMe r=11 | 1.41 | 79.78 [79.59 79.96] / 79.98 | 82.82 [82.82 82.83] / 82.59 | 82.11 / 81.86 |
| random 0.5 | 1.46 | 63.93 / 65.85 | 68.62 / 70.31 | 67.89 / 68.88 |
| low-res 224x112 | 2.22 | 83.06 / 82.88 | 83.90 / 83.84 | 83.72 / 83.52 |
| low-res 192x96 | 1.62 | 82.84 [82.67 83.01] / 82.61 | 82.72 [82.64 82.79] / 82.96 | 83.46 / 83.20 |
| low-res 160x80 | 1.12 | 82.37 [82.18 82.55] / 82.12 | 79.84 [79.68 80.00] / 81.03 | 82.84 / 82.59 |
| low-res 128x64 | 0.72 | 80.91 [80.73 81.08] / 80.59 | 72.83 [72.87 72.79] / 76.22 | 81.25 / 80.95 |
Incomplete (seed 2 missing; `sp_mixed` one seed): not to be quoted as results. Source: `runs/sp_*/mech.csv`,
`runs/sp_*/sweep.csv`.


## 2026-11-16 19:00 — session 3: resume after shutdown
- Pre-checks 19:00: no python.exe; GPU 1 %, 1.2 GB; `runs/sp_mixed_s1/ckpt.pth` 175.6 MB present; no STOP file;
  `git status -sb` in sync with origin/main (only `runs/sp_mixed_s1/` untracked).
- Queue relaunched 19:02:30: `bash run_iter3.sh /c/Users/user/data/Market-1501-v15.09.15 E D >> runs/queue_iter3.txt 2>&1`
  Five finished `sp_*` runs skipped; `sp_mixed_s1` resumed at epoch 62 (loss 2.143 at epoch 61 -> 2.135 at 65).
  The `WARN: commit/push of sp_merge_r_s1 did not complete` line is benign (nothing new to commit).
- Literature: Suma and Tolias (arXiv 2210.05463) read in full via ar5iv HTML (`FULL-S`). Entry A5 added to
  LITERATURE_MATRIX.md. Finding: frozen teacher + same-architecture student initialised from the teacher,
  label-free, low-resolution queries = candidate C1 with another knob. METHOD_CANDIDATES.md C1 corrected.
- `bench_latency.py --extra`: adds rows `merge_r7`, `merge_r11`, `lowres224x112|192x96|160x80|128x64`
  (default behaviour unchanged, so phase D's ViT-B call is unaffected). Smoke test: GFLOPs column = 2.93, 1.463,
  1.941, 1.407, 2.221, 1.62, 1.122, 0.72 (matches mechanism.py).
- **Error made:** the smoke test was meant to be CPU-only, but `CUDA_VISIBLE_DEVICES=""` does not hide the GPU,
  so it ran its GPU section (~1 min) while `sp_mixed_s1` was training (epochs ~67-72). The run was not
  interrupted (epoch 74 at 14.6 min, 4.8 GB VRAM), but (a) the smoke-test timings are contaminated and were
  not saved in the repo, (b) the train time of `sp_mixed_s1` is slightly inflated: do not use it as a timing.
- Pending for the report: seed 2 of `sp_*`; clean latency run (`bench_latency.py --extra --out runs/latency_extra.csv`)
  only after the queue has finished.


## 2026-11-16 21:05 — phase E complete (9 runs, 3 seeds per variant); phase D started
- Queue log: `sp_mixed_s1` done 0.41 h (resumed; timing unreliable); `sp_lowres_s2` 19:23 -> 0.37 h;
  `sp_merge_r_s2` 19:53 -> 0.46 h (mAP 84.3 / R1 93.8); `sp_mixed_s2` 20:30 -> 0.44 h (84.0 / 93.9). Each followed
  by mech + sweep and auto-pushed. Phase D: `train vitb_base_s0` started 21:05:31.
- Aggregate: `head -1 runs/sp_lowres_s0/mech.csv > runs/sp_mechanism.csv; for f in runs/sp_*_s[012]/mech.csv; do tail -n +2 $f >> runs/sp_mechanism.csv; done`
  then `PYTHONUTF8=1 .venv/Scripts/python.exe aggregate_mech.py runs/sp_mechanism.csv > runs/sp_mechanism_agg.md`
  (171 rows; full table with all selection rules and corrections in that file).
- Full-token mAP / R1 per seed (`sweep.csv`, keep 1.0): sp_lowres 82.83/92.99, 83.33/92.93, 83.10/93.20;
  sp_merge_r 84.19/93.82, 84.32/94.69, 84.30/93.76; sp_mixed 83.77/93.79, 84.03/93.56, 84.04/93.88.
  Means ± std: 83.09 ± 0.25, 84.27 ± 0.07, 83.95 ± 0.15.
- A first attempt to write these log entries failed on shell quoting before running anything; redone from files.

### Final phase E table, asymmetric mAP raw (ridge map), 3 seeds
| setting | GFLOPs | sp_lowres | sp_merge_r | sp_mixed |
| --- | --- | --- | --- | --- |
| merge 3-stage 0.7 | 1.92 | 82.11 ± 0.25 | 83.80 ± 0.11 | 83.43 ± 0.23 |
| gradual ToMe r=7 | 1.94 | 82.18 ± 0.25 | 84.07 ± 0.06 | 83.70 ± 0.14 |
| random 0.7 | 1.92 | 78.69 ± 0.31 | 80.39 ± 0.27 | 80.34 ± 0.28 |
| cls_attn EViT-fuse 0.7 | 1.95 | 72.87 ± 1.33 | 73.90 ± 1.11 | 72.22 ± 2.10 |
| merge 3-stage 0.5 | 1.46 | 74.97 ± 0.50 (76.29) | 79.76 ± 0.19 (80.02) | 79.34 ± 0.29 (79.53) |
| gradual ToMe r=11 | 1.41 | 79.59 ± 0.37 (79.87) | 82.90 ± 0.13 (82.69) | 82.38 ± 0.27 (82.16) |
| random 0.5 | 1.46 | 64.15 ± 0.39 | 68.59 ± 0.14 | 68.41 ± 0.56 |
| cls_attn EViT-fuse 0.5 | 1.49 | 39.30 ± 3.96 | 41.52 ± 0.29 | 35.52 ± 4.50 |
| low-res 224x112 | 2.22 | 83.07 ± 0.27 | 83.92 ± 0.08 | 83.91 ± 0.17 |
| low-res 192x96 | 1.62 | 82.85 ± 0.17 (82.62) | 82.68 ± 0.10 (82.94) | 83.60 ± 0.14 (83.39) |
| low-res 160x80 | 1.12 | 82.36 ± 0.19 (82.11) | 79.89 ± 0.18 (81.00) | 83.01 ± 0.15 (82.79) |
| low-res 128x64 | 0.72 | 80.96 ± 0.20 (80.67) | 73.04 ± 0.37 (76.31) | 81.50 ± 0.22 (81.25) |
Query cosine to the full-token embedding (q_cos) at 160x80: 0.962 / 0.865 / 0.953; at r=11: 0.920 / 0.952 / 0.953.


## 2026-11-16 22:12 — phase D, latency, queue finished
- `vitb_base_s0` (21:05-21:40): 0.50 h, mAP 86.4 / R1 94.0; sweep done; auto-pushed.
- `vitb_tbsd_s0` (21:44): **FAILED rc 1 after epoch 59**: `RuntimeError: Couldn't open shared file mapping ...
  error code: <1455>` in DataLoader worker 0 (Windows: paging file too small = system commit limit reached when
  the epoch-60 evaluation started 8 more workers). `ckpt.pth` at epoch 59, step 11918 intact.
- Queue then ran `bench_latency.py --model vit_base_patch16_224 --out runs/latency_vitb.csv` and printed
  `=== all done` 22:11:59.
- 22:12 machine idle (no python.exe, GPU 0 %). Clean latency run:
  `PYTHONUTF8=1 .venv/Scripts/python.exe bench_latency.py --extra --out runs/latency_extra.csv > runs/latency_extra_stdout.txt 2>&1` (rc 0).
- 22:14 resume by hand with 2 workers (rc 0 at 22:42:41, train time 0.79 h total, mAP 83.7 / R1 93.0):
  `.venv/Scripts/python.exe train.py --data C:/Users/user/data/Market-1501-v15.09.15 --out runs/vitb_tbsd_s0 --resume --seed 0 --model vit_base_patch16_224 --sd-mode merge --sd-keep random --workers 2 >> runs/vitb_tbsd_s0/stdout.txt 2>&1`
  then `.venv/Scripts/python.exe evaluate.py --data ... --ckpt runs/vitb_tbsd_s0/last.pth --out runs/vitb_tbsd_s0/sweep.csv --modes merge --workers 2`.

### Latency, ViT-S (`runs/latency_extra.csv`; idle machine; GPU AMP, CPU fp32 8 threads)
| setting | GFLOPs | GPU b1 ms | GPU b256 ms/img | CPU b1 ms | CPU b8 ms/img |
| --- | --- | --- | --- | --- | --- |
| full | 2.93 | 6.04 | 0.1115 | 17.94 | 12.92 |
| merge 3-stage 0.5 | 1.46 | 8.70 | 0.0667 | 13.39 | 7.28 |
| gradual ToMe r=7 | 1.94 | 13.57 | 0.0951 | 18.11 | 9.77 |
| gradual ToMe r=11 | 1.41 | 13.89 | 0.0726 | 15.64 | 7.52 |
| low-res 224x112 | 2.22 | 6.13 | 0.0799 | 15.05 | 10.13 |
| low-res 192x96 | 1.62 | 6.16 | 0.0577 | 13.02 | 7.56 |
| low-res 160x80 | 1.12 | 6.16 | 0.0417 | 10.60 | 5.49 |
| low-res 128x64 | 0.72 | 6.16 | 0.0285 | 8.87 | 3.78 |
GPU rows agree with `runs/latency.csv` (6.00 / 0.1108 full); CPU rows are 10-15 % slower than in that file.

### ViT-B/16, seed 0, merge 3-stage, asymmetric (symmetric) mAP
| keep | GFLOPs | vitb_base_s0 | vitb_tbsd_s0 |
| --- | --- | --- | --- |
| 1.0 | 11.34 | 86.38 | 83.68 |
| 0.9 | 9.91 | 86.20 (86.04) | 83.69 (83.69) |
| 0.8 | 8.59 | 85.61 (84.99) | 83.66 (83.67) |
| 0.7 | 7.44 | 84.52 (82.85) | 83.54 (83.45) |
| 0.6 | 6.50 | 81.94 (78.66) | 83.29 (82.98) |
| 0.5 | 5.65 | 71.51 (62.69) | 82.37 (81.18) |
ViT-B latency (`runs/latency_vitb.csv`): full 6.16 ms GPU b1, 0.328 ms/img GPU b256, 60.5 ms CPU b1, 51.7 ms/img CPU b8.
- Report `REPORT_ITERATION1.md`: sections 6.6 (latency) and 6.7 (ViT-B) written; status line no longer "draft".
- Queue `run_iter3.sh` is finished. Nothing is running.


## 2026-11-16 22:45-23:50 — phase F (frozen-gallery student), shutdown preparation
- Code (commit 2089cba): `train.py --teacher <last.pth>` (student = copy of the checkpoint; frozen eval-mode
  teacher; label-free loss `mean(1 - cos(bn_student_cheap, bn_teacher_full))`; student BNNeck in eval mode;
  `teacher` added to the resume consistency check); `mechanism.py --gallery-ckpt` (gallery and all full-token
  references from the teacher); `run_iter4.sh`.
- Smoke test (scratchpad, 2 epochs, teacher `base_s0`): loss 0.083 -> 0.065; reference mAP 83.89 (= base_s0);
  160x80 queries 80.74 after 2 epochs. Bug fixed afterwards: header line printed GFLOPs from the student
  (0.038); now read from the reference model (rows were always correct).
- Queue: `bash run_iter4.sh /c/Users/user/data/Market-1501-v15.09.15 >> runs/queue_iter4.txt 2>&1`
  22:49:50 fg_mixed_s0 -> 0.23 h; 23:08:47 fg_mixed_s1 -> 0.24 h; 23:28:00 fg_mixed_s2 -> 0.24 h; each with
  mech and auto-push; `=== all done` 23:46:55. Student symmetric full-token mAP at epoch 96 (student gallery,
  not the deployment setting): 84.0 / 84.9 / 84.8. Peak VRAM 1.83 GB.
- User asked at about 23:40 to prepare for shutdown: no new runs; the queue was on its last item (seed 2,
  epoch 92/96) and was left to finish.
- Aggregate: `runs/fg_mechanism.csv` (concatenated `runs/fg_mixed_s[012]/mech.csv`) -> `runs/fg_mechanism_agg.md`.

### fg_mixed, asymmetric mAP vs the teacher's gallery, raw (ridge map), 3 seeds; q_cos = cos(student cheap, teacher full)
| setting | GFLOPs | q_cos | fg_mixed |
| --- | --- | --- | --- |
| full tokens (teacher) | 2.93 | 1 | 84.45 ± 0.49 (83.89 / 84.71 / 84.75) |
| merge 3-stage 0.7 | 1.92 | 0.965 | 83.46 ± 0.49 (83.35) |
| gradual ToMe r=7 | 1.94 | 0.969 | 83.74 ± 0.53 (83.58) |
| merge 3-stage 0.5 | 1.46 | 0.834 | 76.92 ± 0.43 (77.62) |
| random 0.5 | 1.46 | 0.658 | 62.04 ± 0.39 (64.41) |
| gradual ToMe r=11 | 1.41 | 0.913 | 81.50 ± 0.54 (81.44) |
| low-res 224x112 | 2.22 | 0.958 | 84.22 ± 0.49 (84.07) |
| low-res 192x96 | 1.62 | 0.950 | 83.68 ± 0.49 (83.52) |
| low-res 160x80 | 1.12 | 0.931 | 82.78 ± 0.55 (82.60) |
| low-res 128x64 | 0.72 | 0.891 | 80.48 ± 0.69 (80.21) |

### Shutdown checks 23:47-23:55
- `check_ckpts.py` (scratchpad; CPU load, all float tensors finite, metadata present): OK for
  `sp_mixed_s1`, `sp_lowres_s2`, `sp_merge_r_s2`, `sp_mixed_s2` (87.8 MB, 156 tensors), `vitb_base_s0`,
  `vitb_tbsd_s0` (345.4 MB), `fg_mixed_s0/1/2` (87.8 MB). Earlier checkpoints were verified at the previous pause.
- No python.exe; GPU 0 %, 1.5 GB (desktop); no `runs/*/STOP`; no `runs/*/ckpt.pth` (no partial run).
- Nothing to resume. Next-session order of work: PROGRESS.md "What to do first when resuming".


## 2026-11-17 — session 4: resume and literature pass (no GPU used)
- Resume checks: no python.exe; GPU 4 %, 1.6 GB (desktop); no `runs/*/ckpt.pth`, no STOP; `git status -sb`
  in sync, `origin/main` da52f10; `git diff REPORT_ITERATION1.md` empty.
- Searches (WebSearch, extended unless stated): "cross-resolution person re-identification low-resolution
  query high-resolution gallery vision transformer"; "resolution-invariant person re-identification single
  model multiple resolutions without super-resolution MLR-Market-1501"; "asymmetric image retrieval one query
  model multiple computational budgets compatible embeddings elastic"; "FlexiViT ResFormer multi-resolution
  training ..." (standard); "person re-identification efficient query low resolution ... token reduction
  asymmetric"; "token merging token pruning compatible embeddings asymmetric retrieval ...".
- Full reads (WebFetch, summarising tool): arxiv.org/html/2504.11879 (PrunNet, two passes, offset 100000 =
  proof + references); arxiv.org/html/2109.07871 (RFD); ar5iv.labs.arxiv.org/html/2212.00776 (ResFormer; the
  arXiv PDF was unreadable). Abstract pages: 2109.07871, 2212.00776, 2105.12684 (MRJL).
- Edits: LITERATURE_MATRIX.md section F (F1 PrunNet, F2 ResFormer, F3 RFD, F4 unread leads, F5 gap);
  REPORT_ITERATION1.md section 2 (items 6-8), section 3 rewritten ("Preliminary gap (revised 2026-11-17)"),
  section 9 (framing options a/b/c); RESEARCH_LOG.md Part 7; PROGRESS.md and CLAUDE.md state.
- Outcome: all components of the method are prior work; remaining gap = input-side compute knobs for
  compatible re-ID retrieval with measured latency and equal-budget accounting. Framing decision put to the user.


## 2026-11-17 afternoon — deadline mode (user): cascade E17, final iteration-1 documents
- CLAUDE.md: "PRIORITIES FROM 2026-11-17" (deadline 2026-11-18, revised novelty rule, time boxes, deliverables).
- Literature (WebSearch, standard/extended): DVT; query-adaptive retrieval cascades (QuARI found); RANet;
  query performance prediction for image retrieval (iQPP). Abstract level only.
- New: `adaptive.py` (cascade evaluation), `aggregate_adaptive.py` (seed aggregate, gain vs fixed curve at equal
  GFLOPs by linear interpolation, figure).
- Test: `adaptive.py --ckpts runs/sp_mixed_s0/last.pth` (74 s; full-cost 83.77 = known value; 3,368 queries,
  15,913 gallery).
- Runs:
  `.venv/Scripts/python.exe adaptive.py --data C:/Users/user/data/Market-1501-v15.09.15 --ckpts runs/base_s{0,1,2}/last.pth runs/sp_mixed_s{0,1,2}/last.pth --out runs/adaptive.csv`
  `.venv/Scripts/python.exe adaptive.py --data ... --ckpts runs/fg_mixed_s{0,1,2}/last.pth --gallery-ckpts runs/base_s{0,1,2}/last.pth --out runs/adaptive_fg.csv`
  `PYTHONUTF8=1 .venv/Scripts/python.exe aggregate_adaptive.py runs/adaptive.csv runs/adaptive_fg.csv --out runs/adaptive_agg.md --png runs/adaptive.png`

### E17 cascade, asymmetric mAP, 3 seeds (gain = vs fixed curve at equal average GFLOPs)
| model | cheap | p | GFLOPs | top1 | random | oracle | gain top1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sp_mixed | 128x64 | 0.1 | 1.01 | 82.11 ± 0.16 | 81.75 | 82.20 | -0.50 ± 0.01 |
| sp_mixed | 128x64 | 0.3 | 1.60 | 82.91 ± 0.20 | 82.26 | 82.95 | -0.67 ± 0.07 |
| sp_mixed | 160x80 | 0.1 | 1.42 | 83.23 ± 0.17 | 83.13 | 83.27 | -0.13 ± 0.04 |
| sp_mixed | 160x80 | 0.3 | 2.00 | 83.55 ± 0.18 | 83.32 | 83.55 | -0.25 ± 0.03 |
| base | 160x80 | 0.1 | 1.42 | 78.11 ± 0.48 | 76.68 | 78.09 | -0.98 ± 0.19 |
| base | 160x80 | 0.3 | 2.00 | 80.91 ± 0.61 | 78.32 | 80.80 | -1.92 ± 0.33 |
| base | 128x64 | 0.3 | 1.60 | 73.27 ± 1.82 | 67.42 | 73.12 | -7.88 ± 1.76 |
| fg_mixed | 128x64 | 0.3 | 1.60 | 82.67 ± 0.51 | 81.77 | 82.81 | -0.97 ± 0.12 |
| fg_mixed | 160x80 | 0.1 | 1.42 | 83.19 ± 0.50 | 82.93 | 83.25 | -0.12 ± 0.01 |
| fg_mixed | 160x80 | 0.3 | 2.00 | 83.72 ± 0.51 | 83.24 | 83.69 | -0.31 ± 0.03 |
"gap" signal (top-1 minus mean of ranks 2-10) ≈ random everywhere. Full table: `runs/adaptive_agg.md`.
Decision: not promising (dominated by fixed budgets); `sp_mixed` stays the iteration-1 method.
- Documents: REPORT_ITERATION1.md rewritten (final); STUDY_GUIDE.md; SUPERVISOR_QA.md. Training compute of
  the project: 13.4 GPU-h over 57 runs. No GPU training today.

## Session 5 (2026-11-21): 2025-2026 literature review (no GPU, nothing committed)
- User request: focused review of 2025-2026 papers (2026 preferred); no training; no git changes (the user
  commits and pushes). 12 WebSearch queries; full reads (web-fetch, `FULL-S`): ThinkingViT arXiv 2507.10800v3,
  VPFA arXiv 2510.00936v2 (+ PMLR v306 page), OSKT arXiv 2511.06016v1. Browser pane (abstract pages): ACD
  (CVPR 2026 Findings) + its GitHub README, TE-VMamba (CVPR 2026), ThinkingViT CVPR 2026 poster page.
  SRE-Merge (ICML 2026): poster abstract only (OpenReview bot check, not bypassed). CVF PDFs: 403 to the fetch
  tool; in the browser they are downloads (one save dialog was triggered by mistake; nothing downloaded on purpose).
- Venue correction: PrunNet = CVPR 2025 (pp. 15255-15264).
- New file `RECENT_LITERATURE_2025_2026.md`; `LITERATURE_MATRIX.md` section G (+ F1 venue);
  `REPORT_ITERATION1.md` Section 2.3 added, PrunNet venue, gap bullet 4 narrowed.
- Gap verdict: valid but narrowed (per-query budget choice: ThinkingViT). Iteration-2 ranking: D1 cascade with
  cheap-pass reuse; D2 VPFA-style non-linear post-hoc map + ACD-style baselines; D3 native-resolution dataset.

## Session 5b (2026-11-21): VPFA-inspired nonlinear map baseline (eval-time only, nothing committed)
- Script: `nonlinear_map.py` (new), `aggregate_nonlinear.py` (new). Frozen `runs/base_e216_s{0,1,2}/last.pth`; maps fitted on
  bounding_box_train only (601 FIT / 150 VAL identities, flips), no labels; selection by VAL loss (VAL mAP saturated 99.98).
- Command: `.venv/Scripts/python.exe nonlinear_map.py --data C:/Users/user/data/Market-1501-v15.09.15 --ckpts runs/base_e216_s{0,1,2}/last.pth --out runs/nonlinear_map/nonlinear_map.csv`
  (750 s; + 2 smoke tests about 130 s; about 15 GPU-min of the 30-min budget).
- mAP (3 seeds) raw / ridge (report) / ridge (tuned) / MLP / sp_mixed: 192x96 82.06 / 82.80 / 82.96 / 82.77 / 83.60;
  160x80 75.72 / 78.96 / 79.16 / 78.61 / 83.01; 128x64 57.45 / 68.88 / 68.84 / 66.88 / 81.50. ridge_report reproduces the report.
- Conclusion: the nonlinear map does not beat the linear one; sp_mixed stays ahead (+0.64 / +3.85 / +12.6 vs the best map).
  Details: NONLINEAR_MAPPING_BASELINE.md. REPORT_ITERATION1.md not changed (user request).
