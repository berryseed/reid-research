# First research-iteration report: cheap queries against a full-cost gallery in ViT person re-ID

**Status: first-iteration report, 2026-11-16.** Accuracy results are 3-seed results on Market-1501 with
ViT-S/16; latency is measured. Not done: a second dataset, a full second backbone (one ViT-B seed, Section
6.7), re-implemented SFSC / D3still baselines (Section 7). Section 6.8 adds a frozen-gallery baseline in
the style of Suma and Tolias, which matches the proposed method on low-resolution queries.
Sources of every number: `PROGRESS.md`, `PROGRESS_LOG.md`, `runs/*/mech.csv`, `runs/*/sweep.csv`.

## 1. Problem and thesis

Person re-identification (re-ID) matches a *query* image of a person against a *gallery* of images from other
cameras by comparing embedding vectors. The project's thesis is accuracy per unit of resource: compute, labels
and training time.

The setting studied here is **asymmetric retrieval**. The gallery is embedded once, offline, at full cost. The
query is embedded at request time, possibly on a weak device, so the query encoder should be cheap. This only
works if the cheap query embedding is *compatible* with the full-cost gallery embedding, meaning the two can be
compared directly without re-indexing the gallery.

A vision transformer (ViT) offers two ways to make one network cheaper without changing its weights: process
fewer tokens (token merging or pruning), or feed a smaller image (which also means fewer tokens). The question
of this iteration: **how much accuracy does a ViT re-ID model lose when only the query is made cheap, why, and
what is the least expensive way to train one model that stays compatible with its own full-cost gallery?**

## 2. Comparison with prior work

Reading-depth codes: `FULL` whole text read; `FULL-S` whole text read through a summarising tool (equations
not checked line by line); `CODE` official code read. "Not found" means not found in the text read. It does not
mean the paper lacks the feature.

| | SFSC | D3still | Suma and Tolias | SUReID | ToMe | EViT | This work |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Reference | Wu et al., CVPR 2023, arXiv 2206.08289 | Xie et al., CVPR 2024 | WACV 2023, arXiv 2210.05463 | Song and Liu, arXiv 2401.07469 | Bolya et al., ICLR 2023 | Liang et al., ICLR 2022 | — |
| Depth | `FULL-S` | `FULL` | `FULL-S` | `FULL` | `CODE` + abstract | `CODE` + abstract | — |
| Goal | Sub-models of several capacities with mutually compatible features | Small query network ranks a large-network gallery correctly | Small-resolution queries against a large-resolution database | Faster occluded re-ID by token pruning | Faster ViT by merging similar tokens | Faster ViT by keeping attentive tokens | One ViT, cheap queries compatible with its full-token gallery |
| What is made cheap | CNN channel width | Separate small network, plus resolution | Input resolution | Token count (learned pruning) | Token count (merging in every block) | Token count (top-k at 3 layers) | Token count (gradual merging) and input resolution |
| One set of weights for several budgets | Shared weights, separate BatchNorm per sub-model | No (two networks) | No (one student per resolution) | No (one keep ratio per model) | Yes, but accuracy changes; compatibility not studied | No | Yes |
| Cheap-query vs full-gallery evaluation | Yes (all pairs) | Yes | Yes | Not found | Not applicable (classification) | Not applicable | Yes |
| Training signal for compatibility | Shared classifier, evidential loss, gradient projection | Frozen teacher; feature term + ranking-difference terms | Frozen teacher; same-architecture student initialised from it; label-free | Frozen separate teacher; logit and feature distillation | — | — | Shared weights and classifier, two passes, cosine alignment to the full pass |
| Person re-ID data | Market-1501, MSMT17 | MSMT17 | None (CUB200, Cars196, SOP) | Occluded-Duke, Occluded-REID, Market-1501 and others | None | None | Market-1501 |
| Costs reported | Parameter ratios | Inference FLOPs | Query GFLOPs (figures) | Throughput | Throughput, FLOPs | Throughput, FLOPs | GFLOPs, latency (GPU and CPU, batch 1 and large), training GPU-hours, equal-GPU-hour baseline |

Papers added on 2026-11-17 (summarising-tool full reads, `FULL-S`; details in `LITERATURE_MATRIX.md` section
F): PrunNet (arXiv 2504.11879), ResFormer (CVPR 2023), RFD for cross-resolution re-ID (arXiv 2109.07871).
Supporting reads: CSD (CVPR 2022; label-free neighbour-structure distillation, landmarks only) and the papers
in `LITERATURE_MATRIX.md` sections B and C, most at abstract depth and therefore not used for any claim of
absence.

What these papers settle:
1. One training run yielding several mutually compatible re-ID models exists (SFSC), on our dataset.
2. Asymmetric re-ID retrieval with a cheap query encoder exists (SFSC, D3still).
3. Our alignment loss `1 - cos(cheap, stop-gradient(full))` is the standard feature term of asymmetric
   retrieval (D3still's `L_f`, Suma and Tolias's `l_abs`). It is not new.
4. A frozen gallery model with a same-architecture student trained on a cheaper input exists (Suma and
   Tolias). Our candidate "C1" is that method with another knob.
5. Token pruning with distillation for re-ID exists (SUReID), with one budget per model.
6. **(Added 2026-11-17.)** One set of weights with several compatible budgets, **including a ViT-S and
   Market-1501 / MSMT17**, exists: PrunNet (Sun et al., arXiv 2504.11879, 2025) prunes weights (unstructured)
   and handles gradient conflicts. Its authors state that the sparsity is hard to accelerate on hardware.
7. **(Added 2026-11-17.)** Multi-resolution ViT training in which the detached higher-resolution class token
   teaches the lower-resolution one exists: ResFormer (Tian et al., CVPR 2023), for classification and dense
   tasks. This is the training mechanism of our low-resolution second pass.
8. **(Added 2026-11-17.)** Low-resolution queries against a high-resolution re-ID gallery are a known
   sub-field (cross-resolution re-ID, MLR-Market-1501, CAVIAR). In the paper read in full (RFD, Munir et al.,
   arXiv 2109.07871) and the methods it lists, low resolution is a degradation: the query is super-resolved or
   upsampled to the full network input, so it costs at least full compute.

## 3. Preliminary gap (revised 2026-11-17)

Not found in the texts read and in about 35 searches (absence of evidence, not proof of absence):

- **input-side** budgets of one ViT re-ID model (input resolution, token count) used as a **compute** knob
  for the query, with the query embedding compatible with the same model's full-cost gallery. Cross-resolution
  re-ID upsamples the query; PrunNet and SFSC shrink the weights, not the input; ResFormer has no retrieval;
- one set of weights compatible under **two input-side knobs** at once (resolution and token merging);
- **cost accounting** of compatibility: training GPU-hours against an equal-GPU-hour baseline, the accuracy
  lost at full cost, and **measured** latency per regime rather than FLOPs alone;
- measurements of how token reduction moves a re-ID embedding relative to its full-token version.

**Honest assessment.** Every component of our method is prior work: the training mechanism (ResFormer), the
alignment loss (D3still, Suma and Tolias), one-network multi-budget compatibility (SFSC, PrunNet), the frozen
gallery with a same-architecture cheap student (Suma and Tolias). What remains is a *setting* and an
*evaluation*: input-side compute knobs for compatible re-ID retrieval, measured in latency, against
equal-budget baselines. Our own results further show that, on Market-1501, resolution dominates token
merging in accuracy and measured speed (Section 6.6), and that the prior-work-style frozen-gallery student
matches our single-model method on low resolution (Section 6.8). As it stands this supports an empirical
study, not a method contribution. A method contribution would need a mechanism that none of the above has;
candidates are listed in Section 9.

## 4. Methodology

### 4.1 Notation
`f(x; b)` is the L2-normalised BNNeck embedding of image `x` computed by one ViT at budget `b`. Budgets:
`full` (256x128 input, 129 tokens, 2.93 GFLOPs for ViT-S/16); `merge_r` (ToMe-style bipartite merging of `r`
token pairs after the attention of every block); `lowres` (input resized to a smaller size, position embedding
resampled). The gallery is always `f(x; full)`. Asymmetric mAP = queries at budget `b` against that gallery.

### 4.2 Budget-compatible two-pass training
Each training step runs the same batch twice through the same weights:

1. a full pass, with the standard re-ID losses: identity cross-entropy (label smoothing 0.1) on the classifier
   output and batch-hard triplet loss (margin 0.3) on the pre-BNNeck feature;
2. a cheap pass at a budget `b` sampled for that step, with the same two losses through the **same
   classifier**, plus an alignment term.

`L = L_id(full) + L_tri(full) + L_id(b) + L_tri(b) + lambda * mean_i [1 - cos(f(x_i; b), sg f(x_i; full))]`

`sg` is stop-gradient: the full embedding is the target and is not pulled towards the cheap one.
Three choices of the cheap family are compared:

| Variant | Cheap pass | Budgets sampled |
| --- | --- | --- |
| `sp_merge_r` | gradual token merging | r in {3, 5, 7, 9, 11} |
| `sp_lowres` | smaller input | 224x112, 192x96, 160x80, 128x64 |
| `sp_mixed` | one of the two, at random per step | union of both |

The exploratory probe that preceded them, TBSD, used a 3-stage schedule (reduce after blocks 4, 7, 10) with a
random keep ratio. It is kept as a comparison point, not as the method.

### 4.3 What is and is not claimed as new
Not new: two passes with shared weights and classifier (SFSC's "BCT-S" baseline has this structure); the
alignment term (D3still, Suma and Tolias); token merging (ToMe). Candidate contribution, to be confirmed: the
combination applied to the token and resolution knobs of one ViT, the multi-knob variant, the diagnostics of
embedding drift, and the equal-budget cost account.

### 4.4 Training recipe (identical for every run)
ViT-S/16 (ImageNet-21k to 1k augreg weights, timm), Market-1501 (751 training identities, 12,936 images), input
256x128, batch 64 (16 identities x 4 images), 120 epochs x 202 iterations, SGD, learning rate 0.008 with 5
warm-up epochs then cosine decay, mixed precision. Peak VRAM about 4.9 GB on an RTX 5080. Two-pass runs take
0.31-0.46 GPU-hours; the single-pass baseline 0.22.

## 5. Evaluation protocol

- **Dataset:** Market-1501 standard split (3,368 queries, 19,732 gallery images). Metrics: mAP and Rank-1.
- **Seeds:** 3 per configuration, mean ± sample standard deviation. Differences inside the seed spread are not
  counted as results.
- **Baselines, all on the same recipe:**
  1. the ordinary 120-epoch model with cheap queries and no extra training (3-stage merge, gradual ToMe,
     low-resolution queries);
  2. the **equal-GPU-hour baseline**: the ordinary model trained for 216 epochs, which costs what a two-pass
     run costs, with and without a closed-form linear (ridge) map fitted from cheap to full embeddings;
  3. multi-budget training without the alignment term;
  4. the 3-stage TBSD probe.
- **Costs:** analytic GFLOPs of the query encoder; training GPU-hours; measured latency on GPU and CPU.
- **Code check:** our merging and top-k selection were compared numerically with a transcription of the
  official ToMe and EViT functions (`check_reduction.py`; the official code was read, not executed).

## 6. Results

### 6.1 The ordinary model degrades quickly when only the query is made cheap (3 seeds)
Asymmetric mAP, 3-stage merging:

| Keep ratio | GFLOPs | Baseline | TBSD probe |
| --- | --- | --- | --- |
| 1.0 | 2.93 | 84.45 ± 0.49 | 83.95 ± 0.13 |
| 0.8 | 2.22 | 83.70 ± 0.49 | 83.89 ± 0.13 |
| 0.7 | 1.92 | 82.50 ± 0.55 | 83.70 ± 0.12 |
| 0.6 | 1.68 | 79.99 ± 0.75 | 83.29 ± 0.13 |
| 0.5 | 1.46 | 68.56 ± 1.14 | 81.98 ± 0.18 |

Two-pass training removes most of the loss at half cost, costs about 0.5 mAP at full cost (lower in all three
seeds) and takes about 1.85 times the training time.

### 6.2 Most of that apparent gain came from a weak baseline schedule (3 seeds)
At about 1.4-1.5 GFLOPs, asymmetric mAP of the **untrained** baseline depends strongly on how the query is made
cheap:

| Cheap-query setting on the ordinary model | GFLOPs | mAP |
| --- | --- | --- |
| 3-stage merge, keep 0.5 | 1.46 | 68.55 ± 1.11 |
| gradual ToMe, r = 11 | 1.41 | 77.45 ± 1.39 |
| low-resolution query 192x96 | 1.62 | 81.38 ± 0.36 |
| low-resolution query 160x80 | 1.12 | 75.81 ± 0.43 |
| TBSD probe, 3-stage merge | 1.46 | 81.99 ± 0.18 |

Against the best training-free option at equal cost (about 79.6 by linear interpolation between the two
low-resolution points, an estimate), the probe's gain is about +2.4 mAP, not +13.4.

Mechanism findings (3 seeds, `mechanism.py`):
- The movement of the embedding under token reduction is per-image. A shared offset explains only 3-16 % of it,
  and mean or affine corrections recover 0.1-1.5 mAP.
- A closed-form linear map recovers part of it (3-stage merge 68.55 to 72.70).
- Early reduction is the costly part: halving the tokens at block 4, 7 or 10 alone costs 3.8, 1.2 or 0.2 mAP.
- Selecting tokens by class-token attention (EViT rule, exact fusion) is worse than random selection for
  cross-budget matching (72.93 ± 3.28 vs 78.06 ± 0.63 at keep 0.7). Unexplained.

### 6.3 Negative result: the TBSD probe does not beat the equal-GPU-hour baseline (3 seeds)

| Query setting | GFLOPs | Baseline 216 epochs, raw | Baseline 216 epochs + linear map | TBSD probe (raw to mapped) |
| --- | --- | --- | --- | --- |
| full tokens | 2.93 | 85.11 | — | 83.95 |
| low-resolution 192x96 | 1.62 | 82.06 ± 0.19 | 82.80 ± 0.23 | 81.91 to 82.34 |
| low-resolution 160x80 | 1.12 | 75.72 ± 0.51 | 78.96 ± 0.18 | 78.26 to 79.90 |

Training the ordinary model for the same GPU time, then using low-resolution queries with a linear map, matches
the probe at reduced cost and is 1.16 mAP better at full cost. On Market-1501 the probe as built is not
justified.

### 6.4 Proposed variants: gradual-merge, low-resolution and mixed second passes (3 seeds each)
Asymmetric mAP, full-token gallery, mean ± sample std. Source: `runs/sp_mechanism.csv`,
`runs/sp_mechanism_agg.md`, `runs/sp_*/sweep.csv`. Values in brackets: with the closed-form linear map.

| Query setting | GFLOPs | Baseline 120 ep | Baseline 216 ep | TBSD probe | `sp_lowres` | `sp_merge_r` | `sp_mixed` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full tokens | 2.93 | 84.45 ± 0.49 | 85.11 | 83.95 ± 0.13 | 83.09 ± 0.25 | 84.27 ± 0.07 | 83.95 ± 0.15 |
| low-resolution 224x112 | 2.22 | 83.67 ± 0.49 | not tabulated | not tabulated | 83.07 ± 0.27 | 83.92 ± 0.08 | 83.91 ± 0.17 |
| gradual ToMe r = 7 | 1.94 | not tabulated | 83.80 ± 0.04 | not tabulated | 82.18 ± 0.25 | 84.07 ± 0.06 | 83.70 ± 0.14 |
| low-resolution 192x96 | 1.62 | 81.38 ± 0.36 | 82.06 ± 0.19 (82.80 ± 0.23) | 81.91 ± 0.32 | 82.85 ± 0.17 | 82.68 ± 0.10 (82.94) | **83.60 ± 0.14** |
| gradual ToMe r = 11 | 1.41 | 77.45 ± 1.39 | 78.97 ± 0.12 (79.70 ± 0.13) | 81.77 ± 0.18 | 79.59 ± 0.37 | **82.90 ± 0.13** | 82.38 ± 0.27 |
| low-resolution 160x80 | 1.12 | 75.81 ± 0.43 | 75.72 ± 0.51 (78.96 ± 0.18) | 78.26 ± 0.38 | 82.36 ± 0.19 | 79.89 ± 0.18 (81.00) | **83.01 ± 0.15** |
| low-resolution 128x64 | 0.72 | 60.13 ± 2.68 | not measured | 69.26 ± 0.90 | 80.96 ± 0.20 | 73.04 ± 0.37 (76.31) | **81.50 ± 0.22** |

Training time per run (GPU-hours, three seeds): `sp_lowres` 0.36 / 0.31 / 0.37; `sp_merge_r` 0.44 / 0.40 / 0.46;
`sp_mixed` 0.35 / 0.41 / 0.44; 216-epoch baseline 0.36-0.38. The budgets are therefore only approximately
equal: `sp_merge_r` used about 15 % more GPU time than the baseline it is compared with, `sp_lowres` slightly
less. The 0.41 of `sp_mixed` seed 1 is unreliable (resumed run; another process used the GPU for about a
minute). Seed 2 of every variant ran slower than seeds 0-1, so timing noise between sessions is about 0.05 h.

Reading:
1. **Specialists do not transfer well.** A model trained with gradual-merge cheap passes falls to 73.0 mAP with
   128x64 queries; a model trained with low-resolution cheap passes reaches 79.6 with gradual merging, which is
   no better than the equal-GPU-hour baseline with a linear map (79.7).
2. **The mixed model is compatible under both knobs.** It is the best of the three on every low-resolution
   setting, including against the low-resolution specialist (+0.5 to +0.75 mAP, about three times the seed
   spread), and 0.5 below the merge specialist under gradual merging (about two standard deviations with three
   seeds: not clearly separated).
3. **Against the equal-GPU-hour baseline with the linear map:** +4.1 mAP at 1.12 GFLOPs, +2.7 at 1.41, +0.8 at
   1.62. At 1.62 GFLOPs the two specialists only tie that baseline (82.85 and 82.68-82.94 vs 82.80).
4. **Full-cost price.** `sp_mixed` is 1.16 mAP below the 216-epoch baseline at full tokens (83.95 vs 85.11) and
   0.5 below the 120-epoch baseline, the same deficit as the TBSD probe. `sp_merge_r` loses least (84.27).
5. With 128x64 queries (0.72 GFLOPs, a quarter of the full cost) the mixed model keeps 81.5 mAP. Market-1501
   crops are natively 128x64, so this number in particular must not be generalised (Section 7).
6. Class-attention token selection remains far below random selection in all three variants (for example
   `sp_merge_r`, keep 0.5: 41.5 vs 68.6).

**Best of the three two-pass variants: `sp_mixed`** (cheap pass drawn from both families), with its full-cost
deficit as the main open problem. Whether it or the frozen-gallery student of Section 6.8 is carried forward
is an open decision.

### 6.5 Ablations and rejected hypotheses (completed)
- **Alignment term:** multi-budget training without it reaches 80.56 ± 0.07 at 3-stage keep 0.5, against 81.99
  with it. Most of the robustness comes from training at several budgets through a shared classifier; the
  alignment term adds about 1.4 mAP.
- **Per-budget BatchNorm statistics as the cause of the full-cost deficit:** rejected (ablation 83.83 vs 83.95).
- **Fixed-budget specialist** (trained only at keep 0.5): loses to the multi-budget probe.
- **Two-pass training as a low-data regulariser (H3):** not supported. With 10 / 25 / 50 % of the identities
  the probe is 2.2 / 0.9 / 0.9 mAP below the baseline at full cost (3 seeds).

### 6.6 Measured efficiency
Forward pass only, random input, ViT-S, RTX 5080 (mixed precision) and Ryzen 7 7800X3D (8 threads, fp32),
machine otherwise idle, median over repeated runs after warm-up. Source: `runs/latency_extra.csv`
(2026-11-16). Low-resolution rows include the image resize and the resampling of the position embedding.
mAP column: `sp_mixed`, asymmetric, 3 seeds (Section 6.4).

| Query setting | GFLOPs | GPU batch 1, ms | GPU batch 256, ms/image | CPU batch 1, ms | CPU batch 8, ms/image | mAP (`sp_mixed`) |
| --- | --- | --- | --- | --- | --- | --- |
| full tokens | 2.93 | 6.04 | 0.112 | 17.9 | 12.9 | 83.95 |
| 3-stage merge 0.5 | 1.46 | 8.70 | 0.067 | 13.4 | 7.28 | 79.34 |
| gradual ToMe r = 7 | 1.94 | 13.6 | 0.095 | 18.1 | 9.77 | 83.70 |
| gradual ToMe r = 11 | 1.41 | 13.9 | 0.073 | 15.6 | 7.52 | 82.38 |
| low-resolution 224x112 | 2.22 | 6.13 | 0.080 | 15.1 | 10.1 | 83.91 |
| low-resolution 192x96 | 1.62 | 6.16 | 0.058 | 13.0 | 7.56 | 83.60 |
| low-resolution 160x80 | 1.12 | 6.16 | 0.042 | 10.6 | 5.49 | 83.01 |
| low-resolution 128x64 | 0.72 | 6.16 | 0.029 | 8.87 | 3.78 | 81.50 |

Speed-up over full tokens: gradual ToMe r = 11 gives 1.54x at GPU batch 256, 1.15x at CPU batch 1, 1.72x at CPU
batch 8, and is **2.3 times slower at GPU batch 1**. Low resolution 160x80 gives 2.67x, 1.69x and 2.35x, and no
change at GPU batch 1. CPU timings in this file are 10-15 % slower than in the earlier `runs/latency.csv`
(different session), so comparisons should stay within one file.

Reading:
1. **GFLOPs do not predict latency.** A single image on the GPU costs about 6 ms whatever its token count: the
   time is fixed overhead. Savings appear only where arithmetic is the bottleneck (large batches, CPU).
2. **Token merging has its own cost.** The matching step runs in every block, which more than doubles
   single-image GPU latency and cancels most of the saving at CPU batch 1.
3. **On Market-1501, resolution is the better knob on both axes.** At similar analytic cost, a 192x96 query is
   faster than gradual merging with r = 11 in three of the four regimes (equal at CPU batch 8) and more accurate
   (83.60 vs 82.38 mAP). At 160x80 it is faster in all four and still more accurate (83.01).
4. For the thesis (accuracy per unit of compute), the practical operating points of the mixed model are
   low-resolution queries: 83.0 mAP at 2.4-2.7x the throughput, or 81.5 mAP at 3.4-3.9x. Whether merging earns
   a place depends on data where downsizing removes real detail, which this dataset cannot test (Section 7).

### 6.7 Second backbone: ViT-B/16, one seed (indicative only)
Same recipe as ViT-S, seed 0, 3-stage merging, asymmetric mAP against the full-token gallery of the same model.
Source: `runs/vitb_base_s0/sweep.csv`, `runs/vitb_tbsd_s0/sweep.csv`, `runs/latency_vitb.csv`.

| Keep ratio | GFLOPs | ViT-B baseline | ViT-B TBSD probe |
| --- | --- | --- | --- |
| 1.0 | 11.34 | 86.38 | 83.68 |
| 0.9 | 9.91 | 86.20 | 83.69 |
| 0.8 | 8.59 | 85.61 | 83.66 |
| 0.7 | 7.44 | 84.52 | 83.54 |
| 0.6 | 6.50 | 81.94 | 83.29 |
| 0.5 | 5.65 | 71.51 | 82.37 |

Training time 0.50 GPU-hours (baseline) and 0.79 (probe). No equal-GPU-hour ViT-B baseline was run. The probe
run stopped once on a system-memory error after epoch 59 and was resumed from its checkpoint with 2 data-loader
workers instead of 4. Latency of the full ViT-B: 6.16 ms at GPU batch 1 (the same as ViT-S), 0.328 ms/image at
GPU batch 256, 60.5 ms at CPU batch 1.

Reading (one seed; none of this is a result yet):
1. The two qualitative effects carry over: the ordinary model collapses under abrupt token reduction
   (86.4 to 71.5), and two-pass training flattens the curve (83.7 to 82.4).
2. The full-token deficit is much larger than on ViT-S: 2.7 mAP against 0.5. The probe is ahead only at keep 0.6
   and below. Cause not investigated (the recipe was tuned on ViT-S).
3. In accuracy per unit of compute ViT-B is not competitive here: the probe at keep 0.5 gives 82.4 mAP at 5.65
   GFLOPs, whereas an ordinary ViT-S at full tokens gives 84.45 at 2.93. The ViT-B baseline buys +1.9 mAP over
   ViT-S for 3.9 times the GFLOPs. ViT-S stays the main backbone.
4. The mixed variant (Section 6.4) has not been run on ViT-B.

### 6.8 Prior-work-style baseline: frozen-gallery student (3 seeds)
In the style of Suma and Tolias (Section 2): the finished 120-epoch baseline is frozen and encodes the gallery.
A copy of it is trained for 96 epochs **without labels** so that its cheap view (low resolution or gradual
merging, sampled as in `sp_mixed`) reproduces the frozen model's full-token embedding,
`L = mean(1 - cos(S(x; b), T(x; full)))`. Only their absolute term is implemented, not their relational terms
or multi-augmentation scheme. Runs `fg_mixed_s0-2`, teacher `base_s0-2`; source `runs/fg_mechanism.csv`,
`runs/fg_mechanism_agg.md`. Asymmetric mAP against the teacher's gallery, mean ± std.

| Query setting | GFLOPs | Baseline 216 ep + linear map | `sp_mixed` | `fg_mixed` |
| --- | --- | --- | --- | --- |
| full tokens | 2.93 | 85.11 | 83.95 ± 0.15 | 84.45 ± 0.49 (teacher) |
| low-resolution 224x112 | 2.22 | not tabulated | 83.91 ± 0.17 | 84.22 ± 0.49 |
| gradual ToMe r = 7 | 1.94 | 83.75 ± 0.03 | 83.70 ± 0.14 | 83.74 ± 0.53 |
| low-resolution 192x96 | 1.62 | 82.80 ± 0.23 | 83.60 ± 0.14 | 83.68 ± 0.49 |
| gradual ToMe r = 11 | 1.41 | 79.70 ± 0.13 | 82.38 ± 0.27 | 81.50 ± 0.54 |
| low-resolution 160x80 | 1.12 | 78.96 ± 0.18 | 83.01 ± 0.15 | 82.78 ± 0.55 |
| low-resolution 128x64 | 0.72 | not measured | 81.50 ± 0.22 | 80.48 ± 0.69 |

Cost: student 0.23-0.24 GPU-hours and 1.8 GB peak VRAM (two-pass training: 4.9 GB); with the teacher's 0.22
the total is about 0.46 GPU-hours, roughly 20 % more than the 216-epoch baseline (0.36-0.38) and slightly more
than `sp_mixed` (0.35-0.44). Two sets of weights are stored (2 x 88 MB). The larger spread of `fg_mixed`
is inherited from its teachers (84.45 ± 0.49).

Reading:
1. The frozen-gallery student **equals `sp_mixed` on low-resolution queries at 192x96 and 160x80** (differences
   of 0.1-0.2 mAP, inside the spread) and has **no full-cost deficit** (+0.5 over `sp_mixed` at full tokens).
   It needs no labels for the second stage.
2. `sp_mixed` is ahead by 0.9 mAP under gradual merging (r = 11) and 1.0 at 128x64. With spreads of 0.3-0.7 over
   three seeds this is suggestive, not established. Its real distinguishing property is one set of weights.
3. Neither reaches the 216-epoch baseline at full tokens (85.11). A frozen-gallery student on that stronger
   teacher has not been run; it would keep 85.11 at full cost by construction, at a larger total budget.
4. Consequence for the claim: single-model multi-budget training (`sp_mixed`) is **not shown to be better** than
   the closest prior-work scheme. The contribution, if any, lies in the multi-budget, two-knob setting itself
   and its cost account, in which both schemes can be instantiated; which one to carry forward is open.

## 7. Limitations

1. **One dataset.** Market-1501 crops are natively 128x64 and are upsampled to 256x128, so low-resolution
   queries lose little real information. The resolution results may not transfer. A second dataset with higher
   native resolution (CUHK03-NP preferred) is required and needs the author's own sign-up.
2. **One backbone with seeds** (ViT-S/16). ViT-B/16 has one seed, only the baseline and the TBSD probe, and no
   equal-GPU-hour baseline (Section 6.7).
3. **Full-cost deficit.** Every two-pass variant is below the equal-GPU-hour baseline at full tokens (0.8-2.0
   mAP on ViT-S; 2.7 mAP below the 120-epoch baseline on ViT-B, one seed).
4. **Narrow novelty** (Section 3). Several prior-work entries are at abstract depth.
5. **No comparison with a re-implemented SFSC or D3still baseline**, and none with a small CNN at equal query
   cost. The frozen-teacher baseline (Section 6.8) implements only the absolute term of Suma and Tolias.
6. GFLOPs are analytic estimates; the class-attention result is unexplained; gradient conflict between the two
   passes has not been measured.

## 8. Conclusions and potential contributions (one dataset; ViT-S unless stated)

1. In ViT re-ID, the cost of a cheap query depends more on *where and how* tokens are removed than on how many:
   early, abrupt reduction is expensive; gradual merging and lower resolution are far cheaper in accuracy.
2. Reported gains of token-budget training shrink sharply against the right baselines. An equal-GPU-hour
   baseline with low-resolution queries and a closed-form map is a strong, nearly free competitor, and it beat
   our first method.
3. One ViT trained with a cheap pass drawn from both gradual merging and lower resolution stays compatible
   with its own full-cost gallery under either knob: 82.4-83.6 mAP at 1.1-1.6 GFLOPs (38-55 % of full cost),
   which is 0.8 to 4.1 mAP above the equal-GPU-hour baseline with a linear map in that range, at the price of
   1.2 mAP at full cost. Training on one knob alone does not give compatibility on the other (3 seeds,
   Market-1501 only).
4. Measured latency changes the ranking of the knobs. Low-resolution queries give 1.7-3.9x throughput on CPU
   and at large GPU batch with 81.5-83.6 mAP; gradual merging is slower and less accurate at similar analytic
   cost, and 2.3 times slower than full tokens for a single image on the GPU. No setting reduces single-image
   GPU latency.
5. A frozen-gallery student in the style of prior work matches the single-model method on low-resolution
   queries with no loss at full cost (Section 6.8). The single-model method is therefore not shown to be
   better; it differs by needing one set of weights and by about 1 mAP under merging (not established).
6. Negative results worth reporting: the first method lost to an equal-budget baseline; no low-data
   regularisation effect; class-attention token selection worse than random for cross-budget matching;
   merging dominated by resolution on this dataset.

## 9. Next steps

1. Second dataset with higher native resolution (blocked on sign-up). It decides whether merging has a role.
2. **Decide the framing (open, needs the author and instructor).** After the 2026-11-17 literature pass
   (Section 3) there are three honest options, none verified as novel yet:
   - (a) **Empirical study:** input-side compute knobs for compatible re-ID retrieval, measured in latency
     against equal-budget baselines, with the negative results. Defensible now; weak on method novelty.
   - (b) **Per-query budget selection:** spend more compute only on queries whose cheap embedding is likely to
     be unreliable, using a predictor of drift from the full-cost embedding, so the *average* query cost drops
     at fixed accuracy. Not searched yet; dynamic-inference ViTs (for example DVT, NeurIPS 2021, from prior
     knowledge, not re-read) do this for classification, so a literature check comes first.
   - (c) **Native-resolution data:** test whether token merging keeps information that downsizing destroys
     on a dataset whose crops are larger than 128x64. Needs the second dataset.
3. Frozen-gallery student on the 216-epoch teacher; gradient-conflict diagnostic; separate-classifier control;
   more ViT-B seeds and `sp_mixed` on ViT-B. These refine the evaluation; they do not change the novelty.

## Reproduction
Commands and per-run tables: `PROGRESS_LOG.md`. Queues: `run_iter1b.sh`, `run_iter2.sh`, `run_iter3.sh`.
Diagnostics: `mechanism.py`, `aggregate_mech.py`. Latency: `bench_latency.py`.
