# First research iteration: budget-compatible queries for ViT person re-identification

**Iteration-1 report. Written 2026-11-17, audited and revised 2026-11-22 for supervisor review.**
Market-1501, ViT-S/16, 3 seeds per configuration unless stated; latency measured on an RTX 5080 / Ryzen 7
7800X3D. No measurement was changed or re-run in the revision. Interpretations that the evidence did not support
were corrected; `SUPERVISOR_REVIEW_AUDIT.md` lists each one. Short version: `SUPERVISOR_BRIEF.md`.

**How to read the numbers.** Results tables in Section 6 and Table 7.2 contain only **our own runs** (`runs/`).
Numbers **published by other papers** appear only in Section 2 and Table 7.1, and are not compared with ours.
Methods named after a paper are our own adaptations ("-style", "-inspired"), not reproductions.

## Summary

**Problem.** In person re-identification (re-ID) a gallery is embedded once at full cost, while queries
arrive continuously, possibly on weak devices. We want cheaper queries whose embeddings stay *compatible* with
the full-cost gallery, using two input-side knobs of a vision transformer: a smaller input image and fewer
tokens (token merging).

**Method (`sp_mixed`).** One ViT trained with two passes per step through shared weights: a full pass and a
cheap pass whose budget is drawn at random from both knobs (four input resolutions, five merge rates). Both
passes use the standard re-ID losses through a shared classifier, and the cheap embedding is pulled onto the
detached full-cost embedding. It is a credited combination of existing techniques, not a new learning principle.

**Positive findings (asymmetric mAP, 3 seeds).**
- `sp_mixed` scores 83.60 at 1.62 GFLOPs (55 % of full cost), 83.01 at 1.12 (38 %) and 81.50 at 0.72 (25 %).
- Against the similar-GPU-hour baseline with the best of the post-hoc maps we tested (linear or VPFA-inspired
  non-linear, picked on test in the baseline's favour) it is +0.64 at 1.62 (borderline), +3.85 at 1.12 and +12.61
  at 0.72. Under merging at 1.41 GFLOPs it is +2.68 (non-linear maps not tested there).
- Single-knob training carries over only partly to the other knob. The mixed model works on both.
- Lower resolution gives measured speed-ups where compute dominates: 1.7-3.9x at 160x80 and 128x64.

**Negative findings.**
- **`sp_mixed` is not clearly better than a prior-work-style frozen-gallery student.** It ties it at four of
  the six cheap trained budgets evaluated. It is ahead in all 3 seeds at r = 11 (+0.88) and 128x64 (+1.02), but
  both are borderline: whether they pass 2 std depends on how the std is defined (Section 5).
- It is **1.16 mAP worse at full cost** than the similar-GPU-hour baseline (83.95 vs 85.11), borderline worse
  at 2.22 GFLOPs (−0.43), and ties it at 1.94.
- Gradual merging saves GFLOPs but is 2.3x slower than full tokens for a single GPU image. No setting speeds up
  a single GPU image.
- The per-query cascade is below fixed budgets at equal cost, and its idea is prior work (DaRe, CVPR 2018).
- No low-data benefit was found.

**Gap.** Preliminary and narrow: an application-and-evaluation gap. Compatible multi-budget retrieval,
resolution as a compute knob with compatible embeddings, and per-query budgets in re-ID all exist (Section 3).

**Status.** An empirical first iteration. Main threats to validity: one dataset with native 128x64 crops, no
validation split, no single-pass multi-resolution-augmentation or small-model baseline, and merging code that
is ToMe-style but not identical to official ToMe.

---

## 1. Problem and thesis

The project thesis is accuracy per unit of resource: compute, labels and training time. Re-ID compares a query
image of a person with a gallery of images from other cameras by the cosine similarity of their embeddings.

**Asymmetric retrieval.** The gallery is embedded offline at full cost and stored. The query is embedded at
request time and should be cheap. This works only if the cheap query embedding is **compatible** with the
full-cost gallery embedding, i.e. directly comparable without re-indexing the gallery.

**Input-side knobs of a ViT.** A ViT's cost scales with its number of tokens. It can be reduced without changing
the weights in two ways: a smaller input image (fewer patches) or merging tokens inside the network. Both can
be chosen per query at run time.

**Research questions.**
1. How much accuracy does a ViT re-ID model lose when only the query is made cheap, and why?
2. What is the least expensive way to train one model that stays compatible with its own full-cost gallery
   under these knobs?
3. Do GFLOPs savings become real speed-ups?
4. Can compute be spent per query, only where it is needed?

## 2. Related work and comparison

Reading-depth codes: `FULL` = whole text processed; `FULL-S` = whole text processed through a summarising tool
(equations as reported by the tool); `PART` = only the relevant sections seen; `ABS` = abstract or search result
only; `PK` = prior knowledge, not re-read. The reading was done by the AI assistant, so these codes do not mean
that the author has read the paper. "Not found" means not found in what was read, never "the paper lacks it".

### 2.1 Comparison table (six closest papers)

| | SFSC [1] | PrunNet [2] | D3still [3] | Suma and Tolias [4] | ResFormer [5] | VPFA [6] | **This work** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Venue | CVPR 2023 | CVPR 2025 | CVPR 2024 | WACV 2023 | CVPR 2023 | ICML 2026 | — |
| Depth read | `FULL-S` | `FULL-S` | `FULL` | `FULL-S` | `FULL-S` | `FULL-S` | — |
| Goal | Compatible sub-models from one training run | Compatible sub-networks of one network | Small query network ranks like a large gallery network | Small-resolution queries, large-resolution database | One ViT for many resolutions | Low-resolution query features aligned to high resolution | One ViT whose cheap queries match its full-cost gallery |
| What is made cheap | CNN channel width | Weights (pruning) | Separate small network and resolution | Input resolution | Input resolution | Not a compute knob (degradation) | Input resolution and token count |
| One set of weights for several budgets | Yes (separate BN per width) | Yes | No | No (a student per resolution) | Yes | One backbone + MLP | Yes |
| Cheap query vs full gallery evaluated | Yes | Yes | Yes | Yes | No (classification) | Low- vs high-resolution, yes | Yes |
| Training signal | Shared classifier, evidential loss, gradient projection | Shared classifier, gradient-conflict projection | Frozen teacher; feature and ranking-difference losses | Frozen teacher; label-free same-architecture student | Detached high-resolution class token teaches low resolution | Post-hoc label-free gated MLP on a frozen backbone | Shared classifier + cosine alignment to detached full pass |
| Backbone | CNNs | CNNs, ViT-S | CNNs, SwinV2 | ResNet-50/18 | ViT (DeiT) | TransReID | ViT-S/16 |
| Person re-ID data | Market-1501, MSMT17 | Market-1501, MSMT17 (appendix, ResNet-18) | MSMT17 | None found | None found | MLR-Market, MLR-CUHK03, others | Market-1501 |
| Costs reported | Parameter ratios | FLOPs in a figure | Inference FLOPs | Query GFLOPs (figures) | Training time | MLP throughput and parameters | GFLOPs, latency (4 regimes), GPU-hours |

Three further papers **directly narrow the gap**. They were found or re-examined in the 2026-11-22 audit:

| Paper | Depth | What it already establishes | What differs from this work |
| --- | --- | --- | --- |
| DaRe [7], CVPR 2018 | `PART` (Sections 4 and 5.3) | Budgeted person re-ID on Market-1501: each query exits early according to its **nearest-gallery distance**, compared with random routing, with shared computation between stages | CNN depth as the knob; gallery of the same stage, as described in its Section 4.2 |
| MatFormer / MatViT [8], NeurIPS 2024 | `PART` (introduction; results through a summary) | **One ViT** whose smaller sub-models embed queries against a **gallery from the largest model** | MLP width as the knob; ImageNet retrieval; no re-ID found in the part read |
| BTNet [9], BMVC 2023 | `PART` (method and part of evaluation) | **Resolution-dependent compute with embeddings compatible across resolutions** | Face recognition; CNN branches with a shared trunk |

Other work used (details in `LITERATURE_MATRIX.md`):
- **Token reduction:** ToMe [10] (basis of our ToMe-style merging; see the code-check caveat in Section 5) and
  EViT [11] (class-attention top-k).
- **Nested width models for retrieval:** MatMamba (arXiv 2410.06718, retrieval section `FULL-S`).
- **Distillation with token pruning for re-ID, one budget per model:** SUReID (arXiv 2401.07469, `FULL`),
  which reports 87.2 mAP on Market-1501 with a pruned DeiT-S student and a separate teacher.
- **Cross-resolution re-ID** (RFD, arXiv 2109.07871, `FULL-S`): low resolution is a degradation and the query
  is upsampled, so no compute is saved.
- **Cascades for classification:** RANet (CVPR 2020, `ABS`), DVT (NeurIPS 2021, `PK` / `ABS`).
- **Query performance prediction for image retrieval:** iQPP (2023, `ABS`).
- **Abstract level only:** QuARI (NeurIPS 2025). ACD and ThinkingViT are covered in Section 2.3.

### 2.2 What prior work already establishes
1. One training run giving several mutually compatible re-ID models (SFSC; PrunNet).
2. Asymmetric re-ID with a cheap query encoder (SFSC, D3still).
3. One ViT with cheap queries against its own largest-model gallery (MatFormer).
4. Resolution-dependent compute with compatible embeddings (BTNet; Suma and Tolias with two sets of weights).
5. A feature-alignment loss to the full-cost embedding. Ours, `1 - cos`, is in the same family as D3still's
   `L_f` (root of summed squares of `1 - cos`) and Suma and Tolias's `l_abs` (its square).
6. A frozen gallery model with a same-architecture student (Suma and Tolias).
7. Multi-resolution ViT training with a detached higher-resolution teacher (ResFormer).
8. Per-query compute in re-ID chosen by nearest-gallery distance (DaRe), and per-sample compute by confidence
   in classification (RANet, DVT, ThinkingViT).

### 2.3 Update with 2025-2026 work (added 2026-11-21)
Details, equations and reading depth: `RECENT_LITERATURE_2025_2026.md` and `LITERATURE_MATRIX.md` section G.
All numbers in this table are **published by those papers, under their protocols**.

| Paper | Venue | Depth | Relation to this work |
| --- | --- | --- | --- |
| ThinkingViT [12] (Hojjat et al.) | CVPR 2026 | `FULL-S` | One nested ViT (3 then 6 heads), joint loss, per-image entropy exit; the second stage **reuses** the first stage's tokens. Classification only. Template for iteration 2. |
| ACD [13] (Xie et al.) | CVPR 2026 Findings | `ABS` + code README | Asymmetric retrieval with capacity and resolution gaps; freezes the query network and adapts the gallery network. |
| TE-VMamba [14] (Huang, Yoon) | CVPR 2026 | `ABS` | Token pruning and merging for re-ID on VMamba; over 60 % fewer FLOPs on Market-1501 (abstract). |
| SRE-Merge [15] (Li et al.) | ICML 2026 | `ABS` | Training-free re-ID-specific token merging; about 24 % fewer GFLOPs (abstract). |
| VPFA [6] (Liu et al.) | ICML 2026 | `FULL-S` | Cross-resolution re-ID by a label-free residual gated MLP on frozen features. Our adaptation of it is tested in Section 6.10. |
| PrunNet [2] (Sun et al.) | CVPR 2025 | `FULL-S` | Closest in training methodology. |
| OSKT [16] (Li, Qi, Geng) | ICCV 2025 | `FULL-S` | Several re-ID model sizes from one weight chain, fine-tuned separately; cross-model compatibility not evaluated. |

Token reduction for re-ID and resolution-asymmetric retrieval are active topics and are not claimed as new.

### 2.4 References
[1] S. Wu et al. Switchable Representation Learning Framework with Self-compatibility. CVPR 2023, 15943-15953. https://arxiv.org/abs/2206.08289
[2] Y. Sun et al. Learning Compatible Multi-Prize Subnetworks for Asymmetric Retrieval. CVPR 2025, 15255-15264. https://arxiv.org/abs/2504.11879
[3] Y. Xie et al. D3still: Decoupled Differential Distillation for Asymmetric Image Retrieval. CVPR 2024, 17181-17190. https://openaccess.thecvf.com/content/CVPR2024/html/Xie_D3still_Decoupled_Differential_Distillation_for_Asymmetric_Image_Retrieval_CVPR_2024_paper.html
[4] P. Suma, G. Tolias. Large-to-small Image Resolution Asymmetry in Deep Metric Learning. WACV 2023, 1451-1460. https://arxiv.org/abs/2210.05463
[5] R. Tian et al. ResFormer: Scaling ViTs with Multi-Resolution Training. CVPR 2023, 22721-22731. https://arxiv.org/abs/2212.00776
[6] Z. Liu et al. Resolution as a Direction: Vector-Panning Feature Alignment for Cross-Resolution Re-Identification. ICML 2026, PMLR 306:79598-79614. https://proceedings.mlr.press/v306/liu26fr.html
[7] Y. Wang et al. Resource Aware Person Re-identification across Multiple Resolutions. CVPR 2018, 8042-8051. https://arxiv.org/abs/1805.08805
[8] Devvrit et al. MatFormer: Nested Transformer for Elastic Inference. NeurIPS 2024. https://arxiv.org/abs/2310.07707
[9] H. He et al. Learning Unified Representations for Multi-Resolution Face Recognition. BMVC 2023. https://arxiv.org/abs/2310.09563
[10] D. Bolya et al. Token Merging: Your ViT But Faster. ICLR 2023. https://arxiv.org/abs/2210.09461
[11] Y. Liang et al. Not All Patches are What You Need: Expediting Vision Transformers via Token Reorganizations. ICLR 2022. https://arxiv.org/abs/2202.07800
[12] A. Hojjat et al. ThinkingViT: Matryoshka Thinking Vision Transformer for Elastic Inference. CVPR 2026. https://arxiv.org/abs/2507.10800
[13] Y. Xie et al. Asymmetric Collaborative Distillation for Asymmetric Image Retrieval. CVPR 2026 Findings, 6706-6716. https://openaccess.thecvf.com/content/CVPR2026F/html/Xie_Asymmetric_Collaborative_Distillation_for_Asymmetric_Image_Retrieval_CVPRF_2026_paper.html
[14] H. Huang, S. M. Yoon. SSM-Aware Token-Efficient VMamba via Adaptive Patch Pruning and Merging for Person Re-ID. CVPR 2026, 4363-4372. https://openaccess.thecvf.com/content/CVPR2026/html/Huang_SSM-Aware_Token-Efficient_VMamba_via_Adaptive_Patch_Pruning_and_Merging_for_CVPR_2026_paper.html
[15] S. Li et al. Spatially-Regularized Entropy for Discriminative Token Merging in Fine-Grained Re-ID. ICML 2026. https://icml.cc/virtual/2026/poster/64967
[16] L. Li, L. Qi, X. Geng. One-Shot Knowledge Transfer for Scalable Person Re-Identification. ICCV 2025. https://arxiv.org/abs/2511.06016
[17] L. Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015 (Market-1501). https://openaccess.thecvf.com/content_iccv_2015/html/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.html

Bibliographic data of [1]-[9] and the venue of [12] were re-checked online on 2026-11-22. The others are as
recorded in the earlier literature passes.

## 3. Research gap (preliminary)

**Statement.** In the papers read we did not find a study of **one ViT person re-ID model that uses both input
resolution and token merging as query compute budgets, keeps them compatible with its own full-cost gallery,
and accounts for training GPU-hours, full-cost accuracy loss and measured latency.**

The search was not exhaustive, so this is absence of evidence, not proof of absence. It is a gap of
application and evaluation:

| Candidate claim | Can it be claimed? | Reason |
| --- | --- | --- |
| One network with several compatible budgets | No | SFSC, PrunNet |
| One ViT with cheap query and full gallery | No | MatFormer (width, not re-ID) |
| Resolution as a compute knob with compatible embeddings | No, in general | BTNet (faces), Suma and Tolias (two models) |
| Gallery-similarity signal for per-query budgets in re-ID | **No. Withdrawn 2026-11-22** | DaRe, 2018. The earlier version of this report listed it as a gap. |
| Both input knobs in one compatible ViT re-ID model | Preliminary | Not found in what was read |
| The cost account of that setting | Preliminary | Not found in this combination |

**Position.** The contribution of iteration 1 is an adaptation and combination of existing techniques for
this setting, plus a cost-accounted evaluation with negative results. It is not a new learning principle.
This is acceptable for iteration 1 under the revised project rule of 2026-11-17.

## 4. Methodology

### 4.1 Architecture
```
                             shared weights θ (ViT-S/16), shared BNNeck, shared classifier W
                    ┌──────────────────────────────────────────────────────────────────────────────┐
 full pass   x ───► │ resize 256x128 → 128 patches + [CLS] → 12 blocks → [CLS] → BNNeck → e_full  │──► W ─► CE, Tri
                    │                                                                              │
 cheap pass  x ───► │ budget b ~ U{ lowres 224x112 | 192x96 | 160x80 | 128x64   (prob. 1/2)        │──► W ─► CE, Tri
                    │              merge r ∈ {3,5,7,9,11} tokens per block       (prob. 1/2) } → e_b│
                    └──────────────────────────────────────────────────────────────────────────────┘
                                         alignment: 1 − cos(e_b, stopgrad(e_full))
 Test time:  gallery ← e_full (once, offline);   query ← e_b for any budget the device affords
             (optional, Section 6.9: cheap query first, escalate low-confidence queries to full cost)
```
ViT-S/16: width d = 384, 12 blocks, 6 heads, MLP width 1536, 21.6 M backbone parameters, ImageNet-21k→1k pretrained
(timm augreg). Input 256x128 → 16x8 = 128 patch tokens + 1 class token. Retrieval embedding: BNNeck output of
the class token, L2-normalised.

### 4.2 Notation
`x` image, `y` identity label (751 training identities). `f(x; b)` = L2-normalised BNNeck embedding at budget
`b`, `h(x; b)` = pre-BNNeck feature. Budgets: `full`; `lowres(H, W)`; `merge(r)`. Gallery always `f(x; full)`.
`sg` = stop-gradient.

### 4.3 The two cheap-query knobs
**Lower resolution.** The image is resized (bicubic, anti-aliased) to H x W, giving `n = (H/16)(W/16)` patch
tokens (98, 72, 50 or 32 for 224x112, 192x96, 160x80, 128x64). The learned 16x8 position embedding is resampled bicubically
to the new grid. All 12 blocks then run on n + 1 tokens.

**Gradual token merging (ToMe-style, own implementation).** After the attention of every block, the patch tokens are split alternately
into sets A and B; each token of A is matched to its most similar token of B by the cosine of the attention
keys (averaged over heads); the `r` best-matching pairs are merged by a size-weighted average,
`t_new = (s_a t_a + s_b t_b) / (s_a + s_b)`, `s_new = s_a + s_b` (s = number of original patches represented),
and attention uses proportional attention (log-size bias). Each block removes `min(r, n/2)` of its n patch tokens,
so with r = 11 the patch tokens shrink block by block as 128 → 117 → 106 → 95 → 84 → 73 → 62 → 51 → 40 → 29 →
18 → 9 → 5 (plus the class token), and the analytic cost, summed over the actual per-block counts, is 1.41 GFLOPs.
Official ToMe forms A and B over the sequence *including* the class token and re-sorts the unmerged tokens;
ours does neither, so the token counts agree but the merged tokens can differ (Section 5, code check).

### 4.4 Training objective (`sp_mixed`)
For a batch of P x K = 16 x 4 images, with one budget b drawn per step:

`L = L_id(full) + L_tri(full) + L_id(b) + L_tri(b) + λ · L_align(b)`, λ = 1, where

- `L_id(b) = (1/N) Σ_i CE_ε(W f̃(x_i; b), y_i)`: cross-entropy with label smoothing ε = 0.1 on the classifier
  logits (`f̃` = un-normalised BNNeck output). **W is shared by both passes**, so both embeddings must fall on
  the same side of the same class boundaries (the BCT mechanism).
- `L_tri(b) = (1/N) Σ_i max(0, max_{p: y_p = y_i} ‖h_i − h_p‖ − min_{n: y_n ≠ y_i} ‖h_i − h_n‖ + m)`: batch-hard
  triplet loss on the pre-BNNeck feature, margin m = 0.3.
- `L_align(b) = (1/N) Σ_i [1 − cos(f(x_i; b), sg f(x_i; full))]`: pulls the cheap embedding onto the full one;
  stop-gradient blocks the direct pull of the full embedding towards the cheap one. Because the weights are
  shared, the term still changes the full embedding indirectly (Section 6.11).

Variants for the ablation: `sp_lowres` (b always a resolution), `sp_merge_r` (b always a merge rate).

### 4.5 Per-query budget selection (preliminary, evaluation only)
For a query q: compute `f(q; c)` at a cheap budget c (128x64 or 160x80) and its similarities to the gallery.
Confidence `s(q) = max_g cos(f(q; c), f(g; full))` (top-1 similarity). Escalate the fraction p of queries with the
lowest `s(q)` to `f(q; full)`. Average query cost `G(p) = G_c + p · G_full`. Compared at equal `G(p)` with the
fixed-budget curve (all queries at one resolution), with random escalation (does `s` carry information?) and an
drift oracle escalating the largest `1 − cos(f(q; c), f(q; full))` (needs the full pass). The drift oracle is a
reference, **not an upper bound on mAP**: it does not know which queries would gain AP. Whole curve over p
computed, no threshold tuned. The fraction p is ranked over the whole query set, so the rule is transductive.
The routing signal is that of DaRe (CVPR 2018), applied to our knobs and full-cost gallery.

### 4.6 Cost model
Analytic GFLOPs (multiply-accumulates) per block with n tokens: `4nd² + 2n²d + 2ndh` (d = 384, h = 1536), plus
the patch embedding `n_patches · 768 · d`. Full cost: 12 x 0.241 G + 0.038 G = **2.93 GFLOPs**. Merging uses the
actual token count of every block. Latency is measured separately (Section 6.6), because GFLOPs do not predict it.

## 5. Experimental protocol

- **Data:** Market-1501 [17]. Train: 12,936 images of 751 identities. Test: 3,368 queries of 750 identities;
  gallery 19,732 files, of which 3,819 junk images (label −1) are removed, leaving 15,913 (including 2,798
  distractors, label 0000). Counts verified on the files; no identity and no byte-identical image is shared
  between train and test. Same-identity same-camera gallery images are ignored per query.
- **Input:** the crops are natively 128x64. Every input is first upsampled to 256x128; a low-resolution query
  is that tensor downsized again.
- **Metrics:** mAP (main; non-interpolated AP) and Rank-1. **Asymmetric** = cheap queries against the full-cost
  gallery. `q_cos` = mean cosine between an image's cheap and full embeddings (drift).
- **Training recipe:** batch 64 (16 identities x 4 images), 120 epochs x 202 iterations unless stated, SGD
  (momentum 0.9, weight decay 1e-4), lr 0.008, 5 warm-up epochs, cosine decay, mixed precision, RTX 5080.
  Peak PyTorch-allocated VRAM: 1.81 GB baseline, 3.17 GB `sp_mixed`, 1.83 GB frozen-gallery student.
- **Selection:** the last epoch is always used. **There was no validation split for training or design:** test
  mAP was logged every 20 epochs, and variants and budget sets were chosen after seeing test results.
- **Noise rule:** 3 seeds; mean ± sample std. The project rule ("about 2 std") did not say which std, so both
  readings are applied to every comparison of two methods:
  - *combined*: `sqrt(sd_a² + sd_b²)`, which treats the runs as independent;
  - *paired*: the std of the per-seed differences (seed i shares batch order and head initialisation).
  **Clear** = at least 2 under both; **tie** = below 2 under both; **borderline** = they disagree. A tie means
  "not separated", not "equal". This convention was fixed on 2026-11-22, after the results were known. It is a
  heuristic, not a significance test, and no multiple-comparison correction is made.
- **Baselines (our own runs):**
  1. ordinary model, 120 epochs, with cheap queries and no retraining;
  2. **similar-GPU-hour baseline** (`base_e216`, also called equal-GPU-hour in older documents): ordinary
     model, 216 epochs, 0.36-0.40 GPU-h (mean 0.38) vs 0.35-0.44 (mean 0.40) for `sp_mixed`; raw and with
     post-hoc maps fitted on training images (mean, affine, ridge, VPFA-inspired MLP);
  3. multi-budget training without the alignment term (3-stage schedule only);
  4. our earlier 3-stage TBSD probe;
  5. fixed-budget training at keep 0.5;
  6. **frozen-gallery student** in the style of Suma and Tolias (frozen baseline as gallery and teacher, 96
     epochs). Its loss uses no labels, but its batch sampler uses identity labels.
- **Code check:**
  - `evaluate_rank` equals a brute-force implementation on 200 random cases (difference below 1e-14).
  - EViT top-k selection equals a transcription of the official function.
  - **Merging is not identical to official ToMe.** The earlier "difference 0" check (`check_reduction.py`)
    tested the official `class_token=False` path, while official ToMe uses `class_token=True` for these ViTs.
    Against a transcription of that path on random input, one merge step gave the same token set in 56/64
    samples at r = 3, 29/64 at r = 7 and 10/64 at r = 11. All merging results are valid for our variant only.
    The effect on mAP is unknown.
- **Repeatability:** repeated evaluations of the same merge checkpoints differ by up to 0.05 mAP and 0.12
  Rank-1. Three training runs were resumed (`tbsd_bnns_s0`, `sp_mixed_s1`, `vitb_tbsd_s0`) and are not
  bit-identical to uninterrupted runs.
- **Total compute:** 13.38 GPU-hours of training over **50 real runs** (sum of `train_hours`; 7 synthetic
  smoke runs add 0.02). Evaluation, map fitting and latency runs are not included.

## 6. Results

### 6.1 The problem: an ordinary model degrades under cheap queries
Ordinary 120-epoch model, asymmetric mAP: full 84.45 ± 0.49; low resolution 224x112 83.67, 192x96 81.38 ± 0.36,
160x80 75.81 ± 0.43, 128x64 60.13 ± 2.68; gradual merging r = 11 77.45 ± 1.39; 3-stage merging (half the tokens at
blocks 4, 7, 10) 68.56 ± 1.14. Drift is large (`q_cos` 0.61 at 128x64).

### 6.2 How the query is made cheap matters
At nearly equal cost the ordinary model scores 68.55 with 3-stage merging (1.46 GFLOPs) and 77.45 with gradual
merging (1.41). Low resolution, linearly interpolated (an estimate), gives about 79.0 at 1.41 and 79.6 at 1.46.
Its lead over gradual merging at 1.41 is therefore about 1.5, close to one std of the merging result (1.39).

Halving the tokens at a single block costs 3.8 mAP at block 4, 1.2 at block 7 and 0.2 at block 10: early
reduction is the costly part in this comparison (these settings also differ in GFLOPs: 1.87, 2.24, 2.61).

Drift is mostly per-image: a shared offset explains 3-18 % of it. Post-hoc corrections on the 120-epoch model
gain little where drift is small and more where it is large: mean or affine shift 0 to +3.8 mAP, linear map
−0.1 to +8.9. Source: `runs/mechanism.csv`.

### 6.3 Our first method (3-stage TBSD probe): mixed result

| Query | GFLOPs | 216-ep baseline raw | 216-ep baseline + map | TBSD probe raw (map) |
| --- | --- | --- | --- | --- |
| full | 2.93 | 85.11 ± 0.15 | — | 83.95 ± 0.13 |
| 192x96 | 1.62 | 82.06 ± 0.19 | 82.80 ± 0.23 | 81.91 ± 0.32 (82.34) |
| 3-stage merge 0.5 | 1.46 | 69.75 ± 0.72 | 73.56 ± 0.42 | 81.99 ± 0.18 (81.77) |
| merge r = 11 | 1.41 | 78.97 ± 0.12 | 79.70 ± 0.13 | 81.77 ± 0.18 (81.87) |
| 160x80 | 1.12 | 75.72 ± 0.51 | 78.96 ± 0.18 | 78.26 ± 0.38 (79.90) |

Its apparent +13.4 mAP at half the tokens came from comparing against the weak 3-stage schedule. Against the
similar-GPU-hour baseline with a linear map it is better under merging (+2.1 at r = 11), not better at low
resolution, and 1.16 worse at full cost. This motivated the second-pass variants.

### 6.4 Main result: which cheap pass to train with (3 seeds each)
Asymmetric mAP, full-cost gallery. All cells are raw except the column marked "+ ridge map".

| Query | GFLOPs | Baseline 120 ep | Baseline 216 ep | Baseline 216 ep + ridge map | TBSD probe | `sp_lowres` | `sp_merge_r` | **`sp_mixed`** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | 2.93 | 84.45 ± 0.49 | 85.11 ± 0.15 | — | 83.95 ± 0.13 | 83.09 ± 0.25 | 84.27 ± 0.07 | 83.95 ± 0.15 |
| 224x112 | 2.22 | 83.67 ± 0.49 | 84.28 ± 0.12 | 84.30 ± 0.10 | 83.40 ± 0.20 | 83.07 ± 0.27 | 83.92 ± 0.08 | 83.91 ± 0.17 |
| merge r = 7 | 1.94 | 82.79 ± 0.71 | 83.80 ± 0.04 | 83.75 ± 0.03 | 83.33 ± 0.26 | 82.18 ± 0.25 | 84.07 ± 0.06 | 83.70 ± 0.14 |
| 192x96 | 1.62 | 81.38 ± 0.36 | 82.06 ± 0.19 | 82.80 ± 0.23 | 81.91 ± 0.32 | 82.85 ± 0.17 | 82.68 ± 0.10 | 83.60 ± 0.14 |
| 3-stage merge 0.5 | 1.46 | 68.55 ± 1.11 | 69.75 ± 0.72 | 73.56 ± 0.42 | 81.99 ± 0.18 | 74.97 ± 0.50 | 79.76 ± 0.19 | 79.34 ± 0.29 |
| merge r = 11 | 1.41 | 77.45 ± 1.39 | 78.97 ± 0.12 | 79.70 ± 0.13 | 81.77 ± 0.18 | 79.59 ± 0.37 | 82.90 ± 0.13 | 82.38 ± 0.27 |
| 160x80 | 1.12 | 75.81 ± 0.43 | 75.72 ± 0.51 | 78.96 ± 0.18 | 78.26 ± 0.38 | 82.36 ± 0.19 | 79.89 ± 0.18 | 83.01 ± 0.15 |
| 128x64 | 0.72 | 60.13 ± 2.68 | 57.45 ± 2.34 | 68.88 ± 0.70 | 69.26 ± 0.90 | 80.96 ± 0.20 | 73.04 ± 0.37 | 81.50 ± 0.22 |

Evidence: `runs/mechanism.csv`, `runs/base_e216_s*/mech.csv`, `runs/sp_*_s*/mech.csv`, `runs/*/sweep.csv`.

Rank-1 where it was measured (`runs/*/sweep.csv`, `runs/adaptive*.csv`, `runs/nonlinear_map/nonlinear_map.csv`,
`runs/mech_diag/mech_diag.csv`):

| Query | Baseline 120 ep | Baseline 216 ep | `sp_lowres` | `sp_merge_r` | `sp_mixed` | `fg_mixed` |
| --- | --- | --- | --- | --- | --- | --- |
| full | 93.97 ± 0.39 | 94.08 ± 0.39 | 93.04 ± 0.14 | 94.09 ± 0.52 | 93.74 ± 0.17 | 93.97 ± 0.39 (teacher) |
| 192x96 | 92.03 ± 0.19 | 92.39 ± 0.55 | 92.82 ± 0.09 | 93.07 ± 0.43 | 93.59 ± 0.45 | 93.26 ± 0.27 |
| merge r = 11 | not measured | 90.49 ± 0.48 | 90.77 ± 0.42 | 93.04 ± 0.17 | 92.68 ± 0.13 | not measured |
| 160x80 | 87.99 ± 0.32 | 88.22 ± 0.29 | 92.70 ± 0.05 | 91.22 ± 0.36 | 92.95 ± 0.27 | 92.51 ± 0.39 |
| 128x64 | 74.08 ± 2.05 | 70.67 ± 2.71 | 91.60 ± 0.33 | 85.85 ± 0.46 | 92.09 ± 0.02 | 91.39 ± 0.61 |

Training time (GPU-h, three seeds): `sp_lowres` 0.36/0.31/0.37, `sp_merge_r` 0.44/0.40/0.46, `sp_mixed`
0.35/0.41/0.44 (seed 1 resumed and briefly shared the GPU: unreliable), 216-epoch baseline 0.36/0.40/0.38.
Identical configurations vary by up to about 25 % in time, so "equal GPU-hours" is approximate.

Findings:
1. **Specialists carry over only partly.** The merge-trained model reaches 73.04 at 128x64 (above the baseline
   with a map, 68.88, but far below 81.50). The resolution-trained model reaches 79.59 at r = 11 (equal to the
   baseline with a map, 79.70).
2. **`sp_mixed` works under both knobs.** Over the resolution specialist it is +0.54 to +0.84 at low resolution
   (clear at three resolutions; borderline at 128x64). Against the merge specialist it is −0.52 at r = 11 (a tie) and
   −0.37 at r = 7 (clear: the specialist is better).
3. **Against the 216-epoch baseline + ridge map:** +0.81 at 1.62 GFLOPs, +2.68 at 1.41, +4.05 at 1.12, +12.61
   at 0.72. Section 6.10 gives the stricter comparison against the best map.
4. **The advantage disappears at high budgets:** a tie at 1.94; borderline worse at 2.22 (−0.43, best baseline
   correction; the uncorrected baseline is a tie);
   1.16 worse at full cost. Against the 120-epoch baseline the full-cost difference (−0.50) is a tie.
5. 81.50 mAP at a quarter of the full cost is not to be generalised: the crops are natively 128x64.
6. `sp_mixed` is weaker on the abrupt 3-stage schedule it was not trained on (79.34 vs 81.99 for TBSD).

### 6.5 Ablations (completed)
- **Alignment term (3-stage schedule):** 80.56 ± 0.07 without it vs 81.99 ± 0.18 with it at keep 0.5, so it adds
  about 1.4. Most of the gain comes from two-pass training with a shared classifier. **The shared classifier was
  never isolated**, and `sp_mixed` was never run without alignment.
- **Per-budget BatchNorm statistics** as the cause of the full-cost deficit: rejected (83.83 ± 0.20 vs
  83.95 ± 0.13; `runs/tbsd_bnns_s*`).
- **Fixed-budget specialist** (trained only at keep 0.5): 80.68 ± 0.45 at full, 79.24 ± 0.39 at keep 0.5. It
  loses to multi-budget training (83.95, 81.99) but costs 0.24 vs 0.41 GPU-h (`runs/fix50_s*`).
- **Low-data regularisation (H3): not supported** (`runs/{base,tbsd}_id{10,25,50}_s*`).

  | Identities | Baseline, full | TBSD, full | Baseline, keep 0.5 | TBSD, keep 0.5 | GPU-h, baseline / TBSD |
  | --- | --- | --- | --- | --- | --- |
  | 10 % | 55.88 ± 1.83 | 53.72 ± 0.20 | 36.96 ± 0.99 | 49.35 ± 0.03 | 0.03 / 0.04 |
  | 25 % | 71.06 ± 0.63 | 70.11 ± 0.73 | 51.39 ± 0.80 | 66.87 ± 1.03 | 0.06 / 0.10 |
  | 50 % | 78.66 ± 0.47 | 77.76 ± 0.46 | 59.94 ± 1.44 | 75.30 ± 0.50 | 0.12 / 0.21 |

  The full-cost differences are ties at 10 and 25 % and borderline at 50 % (TBSD lower in all seeds).
  `sp_mixed` was not run on low data.
- **Token selection rule (evaluation only; no model was trained with EViT):** class-attention top-k with EViT
  fusion is below random selection for cross-budget matching (72.93 ± 3.28 vs 78.06 ± 0.63 at keep 0.7 on the
  baseline). The mean ordering holds in every model evaluated at keep 0.7 and 0.5, but at keep 0.7 it is a tie
  on both baselines and on the no-alignment model. At keep 0.5 it is clear except on the 216-epoch baseline
  (borderline). Keeping the *least* attended tokens is far worse (`runs/*/sweep_sel.csv`), so attention is not
  uninformative. Cause unexplained.

### 6.6 Measured efficiency
Forward pass, random input, idle machine, median after warm-up (`runs/latency_extra.csv`). GPU: RTX 5080,
mixed precision. CPU: 8 threads, fp32. Low-resolution rows include the resize and position-embedding resampling.

| Query | GFLOPs | GPU b1 (ms) | GPU b256 (ms/img) | CPU b1 (ms) | CPU b8 (ms/img) | mAP `sp_mixed` |
| --- | --- | --- | --- | --- | --- | --- |
| full | 2.93 | 6.04 | 0.1115 | 17.9 | 12.9 | 83.95 |
| 3-stage merge 0.5 | 1.46 | 8.70 | 0.067 | 13.4 | 7.28 | 79.34 |
| merge r = 7 | 1.94 | 13.6 | 0.095 | 18.1 | 9.77 | 83.70 |
| merge r = 11 | 1.41 | 13.9 | 0.073 | 15.6 | 7.52 | 82.38 |
| 224x112 | 2.22 | 6.13 | 0.080 | 15.1 | 10.1 | 83.91 |
| 192x96 | 1.62 | 6.16 | 0.058 | 13.0 | 7.56 | 83.60 |
| 160x80 | 1.12 | 6.16 | 0.042 | 10.6 | 5.49 | 83.01 |
| 128x64 | 0.72 | 6.16 | 0.029 | 8.87 | 3.78 | 81.50 |

1. **GFLOPs do not predict latency.** A single image on the GPU costs about 6 ms at any resolution (overhead
   bound). Savings appear where arithmetic dominates: 160x80 is 2.67x faster at GPU batch 256, 1.69x at CPU
   batch 1, 2.35x at CPU batch 8; 128x64 3.9x, 2.0x and 3.4x. 192x96 gives only 1.4-1.9x.
2. **Merging has its own cost.** Its matching step runs in every block: 2.3x *slower* than full tokens at GPU
   batch 1, and little gain at CPU batch 1.
3. **Resolution has the edge over merging here, with qualifications.** 192x96 (1.62 G) is faster than r = 11
   (1.41 G) in 3 of 4 regimes; the 4th (CPU batch 8) is too noisy to rank. For `sp_mixed`, resolution interpolated to 1.41 G gives
   about 83.4 vs 82.38 for merging; at 1.94 G the two are equal (about 83.8 vs 83.70).
4. **Scope:** timings exclude image decoding and the initial resize, and come from one machine. Between our two
   benchmark sessions (`runs/latency.csv`, `runs/latency_extra.csv`) the same settings differ by up to 7 % on
   the GPU, 9-17 % at CPU batch 1 and 17-46 % at CPU batch 8. Treat CPU batch-8 speed-ups as rough.

### 6.7 Second backbone: ViT-B/16 (one seed, indicative)
3-stage merging, asymmetric mAP. Baseline: 86.38 (full, 11.34 GFLOPs), 84.52 (keep 0.7), 71.51 (keep 0.5). TBSD
probe: 83.68, 83.54, 82.37. Training 0.50 / 0.79 GPU-h. Same qualitative behaviour; the full-cost deficit is
larger (2.7 mAP); per GFLOP ViT-B is worse than ViT-S (82.4 at 5.65 G vs 84.45 at 2.93 G). ViT-S stays the main
backbone. `sp_mixed` was not run on ViT-B. Full-token ViT-B latency: 6.16 ms at GPU batch 1, 60.5 ms at CPU
batch 1 (`runs/latency_vitb.csv`).

### 6.8 Prior-work-style baseline: frozen-gallery student (3 seeds)
Frozen 120-epoch baseline = gallery encoder and teacher; a copy trained 96 epochs on cheap views (same budget
distribution as `sp_mixed`) with a label-free loss, `L = mean(1 − cos(S(x; b), T(x; full)))`.

**This is our adaptation, not Suma and Tolias's method as published:** their absolute term is squared, and they
add relational terms and 8 coupled augmentations; we use a ViT, one student for all budgets, and an
identity-balanced sampler.

| Query | GFLOPs | 216-ep baseline + map | `sp_mixed` | `fg_mixed` (frozen gallery) | `sp_mixed` − `fg_mixed` |
| --- | --- | --- | --- | --- | --- |
| full | 2.93 | 85.11 ± 0.15 (raw) | 83.95 ± 0.15 | 84.45 ± 0.49 (teacher) | −0.50 (tie) |
| 224x112 | 2.22 | 84.30 ± 0.10 | 83.91 ± 0.17 | 84.22 ± 0.49 | −0.31 (tie) |
| merge r = 7 | 1.94 | 83.75 ± 0.03 | 83.70 ± 0.14 | 83.74 ± 0.53 | −0.04 (tie) |
| 192x96 | 1.62 | 82.80 ± 0.23 | 83.60 ± 0.14 | 83.68 ± 0.49 | −0.07 (tie) |
| 3-stage merge 0.5 (untrained schedule) | 1.46 | 73.56 ± 0.42 | 79.34 ± 0.29 | 76.92 ± 0.43 | +2.42 (clear) |
| merge r = 11 | 1.41 | 79.70 ± 0.13 | 82.38 ± 0.27 | 81.50 ± 0.54 | +0.88 (borderline: 1.4 combined, 2.8 paired) |
| 160x80 | 1.12 | 78.96 ± 0.18 | 83.01 ± 0.15 | 82.78 ± 0.55 | +0.23 (tie) |
| 128x64 | 0.72 | 68.88 ± 0.70 | 81.50 ± 0.22 | 80.48 ± 0.69 | +1.02 (borderline: 1.4 combined, 2.1 paired) |

Cost: student 0.23-0.24 GPU-h + teacher 0.22 = about 0.46 (21 % above the 216-epoch baseline); two sets of
weights (2 x 88 MB). Source: `runs/fg_mixed_s*/mech.csv`, `runs/fg_mechanism_agg.md`.

Reading: **`sp_mixed` is not clearly better than this scheme at any budget it was trained for.** It has two
borderline leads (r = 11 and 128x64), positive in all three seeds. More seeds are needed to settle them. Its one
clear lead is on the abrupt 3-stage schedule, which neither model was trained on. The student needs no labels in
its loss; `sp_mixed` needs one set of weights. The student inherits its teacher's full-cost accuracy; that is
0.66 below the 216-epoch baseline on average, which is itself a tie under the noise rule.

### 6.9 Preliminary: per-query budget selection (evaluation only, 3 seeds)
Cascade of Section 4.5. "Gain" = cascade mAP minus the fixed-budget mAP of the same model at the same average
GFLOPs (linear interpolation of the fixed curve). Source: `runs/adaptive*.csv`, `runs/adaptive_agg.md`,
figure `runs/adaptive.png`.

| Model | Cheap → full | p | Avg GFLOPs | top-1 signal | random | oracle | Gain of top-1 vs fixed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `sp_mixed` | 128x64 | 0.1 | 1.01 | 82.11 ± 0.16 | 81.75 | 82.20 | −0.50 ± 0.01 |
| `sp_mixed` | 128x64 | 0.3 | 1.60 | 82.91 ± 0.20 | 82.26 | 82.95 | −0.67 ± 0.07 |
| `sp_mixed` | 160x80 | 0.1 | 1.42 | 83.23 ± 0.17 | 83.13 | 83.27 | −0.13 ± 0.04 |
| `sp_mixed` | 160x80 | 0.3 | 2.00 | 83.55 ± 0.18 | 83.32 | 83.55 | −0.25 ± 0.03 |
| baseline | 160x80 | 0.1 | 1.42 | 78.11 ± 0.48 | 76.68 | 78.09 | −0.98 ± 0.19 |
| baseline | 160x80 | 0.3 | 2.00 | 80.91 ± 0.61 | 78.32 | 80.80 | −1.92 ± 0.33 |
| baseline | 128x64 | 0.3 | 1.60 | 73.27 ± 1.82 | 67.42 | 73.12 | −7.88 ± 1.76 |
| `fg_mixed` | 128x64 | 0.3 | 1.60 | 82.67 ± 0.51 | 81.77 | 82.81 | −0.97 ± 0.12 |
| `fg_mixed` | 160x80 | 0.1 | 1.42 | 83.19 ± 0.50 | 82.93 | 83.25 | −0.12 ± 0.01 |
| `fg_mixed` | 160x80 | 0.3 | 2.00 | 83.72 ± 0.51 | 83.24 | 83.69 | −0.31 ± 0.03 |

![Cascade vs fixed budgets](runs/adaptive.png)

The figure legend calls the drift oracle an "upper bound". That label is inaccurate (Section 4.5).

Findings:
1. **The distance signal selects better than random, as in DaRe.** Over all p strictly between 0 and 1 it is
   within 0.15 mAP of the drift oracle, sometimes above it (the oracle is not an upper bound), and up to +6.13
   over random on the baseline. The alternative signal (top-1 minus mean of ranks 2-10) is mostly **worse**
   than random (up to −1.21), except from 128x64 on the baseline (up to +0.94).
2. **The cascade is below fixed budgets at every p with a fixed counterpart:** −0.07 to −0.68 mAP for
   `sp_mixed`, −0.07 to −1.05 for the student, −0.31 to −8.39 for the baseline. An escalated query pays the
   cheap pass *and* the full pass, and the fixed-budget curve is concave. Escalation to an intermediate budget
   was not tested.
3. **Scope:** for `fg_mixed`, escalated queries use the student's full pass (84.37 ± 0.48), not the teacher's.
   The 216-epoch baseline was not evaluated as a cascade.
4. **Decision (time-boxed, as planned):** not promising in this form on this data. Iteration 1 keeps
   `sp_mixed`; reuse of the cheap pass is iteration-2 work.

### 6.10 Post-hoc non-linear map (VPFA-inspired baseline, 2026-11-21; 3 seeds)
Frozen 216-epoch baseline; maps fitted on training images only, chosen by regression loss on 150 held-out
training identities. **Our VPFA-inspired implementation, not a reproduction of VPFA.** Details:
`NONLINEAR_MAPPING_BASELINE.md`. Evidence: `runs/nonlinear_map/{nonlinear_map.csv,agg.md,stdout.txt}`.

| Query | GFLOPs | Baseline raw | + ridge, λ = 1 | + ridge, λ tuned | + MLP (VPFA-inspired) | `sp_mixed` | `sp_mixed` − best correction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 192x96 | 1.62 | 82.06 ± 0.19 | 82.80 ± 0.23 | 82.96 ± 0.22 | 82.77 ± 0.31 | 83.60 ± 0.14 | +0.64 ± 0.35 (borderline) |
| 160x80 | 1.12 | 75.72 ± 0.51 | 78.96 ± 0.18 | 79.16 ± 0.21 | 78.61 ± 0.33 | 83.01 ± 0.15 | +3.85 ± 0.30 |
| 128x64 | 0.72 | 57.45 ± 2.34 | 68.88 ± 0.70 | 68.84 ± 0.69 | 66.88 ± 0.22 | 81.50 ± 0.22 | +12.61 ± 0.71 |

- The MLP does not beat the linear map, although its held-out regression loss is lower in 8 of 9 cases.
- "Best correction" is picked per seed on test, which favours the baseline.
- +0.64 is 2.5 combined std but 1.8 paired std, hence borderline.
- Caveats: the smoke test printed test mAP before the candidate grid was finalised, so the test split was
  **not** used strictly once (`runs/nonlinear_smoke/smoke.csv`). The tuned λ is at the edge of its grid (0.1)
  for two of three resolutions. The search was small, and merging queries were not mapped.

### 6.11 Mechanism diagnostic (2026-11-21; evaluation only, 18 checkpoints)
Details: `MECHANISM_ANALYSIS.md`, read with the corrections below. Evidence:
`runs/mech_diag/{mech_diag.csv,agg.md,mech_diag.png}`. "Same-budget" = query and gallery both cheap.

| Model, 160x80 | Asymmetric mAP | Same-budget mAP | `q_cos` | Mean margin | Top-1 broken vs full query |
| --- | --- | --- | --- | --- | --- |
| `base_e216` | 75.72 ± 0.51 | 70.92 ± 2.20 | 0.770 | 0.142 | 238 |
| `sp_lowres` | 82.36 ± 0.19 | 81.79 ± 0.14 | 0.962 | 0.184 | 42 |
| `sp_mixed` | 83.01 ± 0.15 | 82.26 ± 0.18 | 0.953 | 0.184 | 59 |

What it supports:
- `sp_mixed` makes cheap embeddings both more discriminative on their own and closer to the full embeddings,
  and breaks about 4x fewer correct top-1 matches than the baseline at 160x80.
- On the 3-stage models, the alignment term causes lower drift (`q_cos` 0.916 → 0.942) and higher mAP
  (80.84 → 81.78) at r = 11.

What it does not support:
- **That information loss, rather than incompatibility, is the dominant failure.** Same-budget mAP is lower
  than asymmetric mAP, but that test also degrades the gallery, so it cannot separate the two. At 128x64 the
  baseline's same-budget Rank-1 is higher (73.15 vs 70.67; borderline), and a linear map lifts asymmetric mAP to 68.88,
  far above the same-budget 48.07. Information loss is a hypothesis.
- **That higher hardest-negative similarity explains the full-cost deficit.** The model without alignment has
  the deficit too (83.71 ± 0.20) with *lower* hardest-negative similarity than the baseline (0.484 vs 0.500).
- That alignment is needed in `sp_mixed`, or that lower drift in general causes higher mAP.

### 6.12 Inventory of all completed experiments

| Experiment | Runs | Section | Evidence |
| --- | --- | --- | --- |
| Baseline, 120 epochs | 3 | 6.1, 6.4 | `runs/base_s*` |
| Baseline, 216 epochs | 3 | 6.3, 6.4 | `runs/base_e216_s*` |
| 3-stage TBSD probe | 3 | 6.3 | `runs/tbsd_s*` |
| TBSD without alignment | 3 | 6.5 | `runs/tbsd_nosd_s*` |
| BatchNorm-statistics ablation | 3 | 6.5 | `runs/tbsd_bnns_s*` |
| Fixed-budget training | 3 | 6.5 | `runs/fix50_s*` |
| Low data (10/25/50 % identities, 2 methods) | 18 | 6.5 | `runs/{base,tbsd}_id*_s*` |
| Second-pass variants (3 methods) | 9 | 6.4 | `runs/sp_*_s*` |
| Frozen-gallery student | 3 | 6.8 | `runs/fg_mixed_s*` |
| ViT-B baseline and probe | 2 | 6.7 | `runs/vitb_*_s0` |
| **Training total** | **50** | | 13.38 GPU-h |
| Drift and correction diagnostics | eval | 6.2 | `runs/mechanism.csv`, `runs/*/mech.csv` |
| Token-selection diagnostics | eval | 6.5 | `runs/*/sweep_sel.csv` |
| Latency, ViT-S and ViT-B | eval | 6.6, 6.7 | `runs/latency*.csv` |
| Per-query cascade | eval | 6.9 | `runs/adaptive*.csv` |
| Non-linear map baseline | map fits | 6.10 | `runs/nonlinear_map/` |
| Mechanism diagnostic | eval | 6.11 | `runs/mech_diag/` |

Synthetic smoke runs (`runs/smoke*`) are install checks and are not results.

## 7. Fair comparison with existing methods and baselines

### 7.1 Numbers published by other papers (NOT comparable with ours or with each other)

| Paper | Its protocol | What it reports |
| --- | --- | --- |
| D3still | MSMT17; ResNet-18 query at 64x64, ResNet-101 gallery at 256x256 | 18.64 mAP (gallery model alone: 59.81) |
| PrunNet | VeRi-776 (vehicles); self-test | 67.82 mAP dense; 64.45 at 25 % capacity |
| Suma and Tolias | CUB200; query 224, database 448 | 40.8 mAP for the trained student |
| VPFA | MLR-Market-1501; TransReID features | 94.1 Rank-1 (its baseline 90.3); no mAP in the main text |
| SUReID | Market-1501; pruned DeiT-S with a separate teacher; symmetric retrieval | 87.2 mAP, 94.5 Rank-1 |
| SFSC, ResFormer | Market-1501 / MSMT17 with CNN sub-models; ImageNet classification | Not quoted: no setting matches ours |

Datasets, backbones, recipes, metrics and retrieval settings differ. No ranking between this table and Table
7.2 is valid.

### 7.2 Our own implementations, on the same data, backbone and recipe (3 seeds)

| Scheme (our implementation) | Related idea (not a reproduction) | 1.12 GFLOPs (160x80) | 1.41 (merge r = 11) | Full cost | Weights | Training GPU-h |
| --- | --- | --- | --- | --- | --- | --- |
| Ordinary model, 120 epochs | — | 75.81 | 77.45 | 84.45 | 1 | 0.22 |
| Longer training + linear map (λ = 1) | training-free correction | 78.96 | 79.70 | 85.11 | 1 + map | 0.38 |
| Longer training + tuned linear map | training-free correction | 79.16 | not run | 85.11 | 1 + map | 0.38 + map fit |
| Longer training + MLP | VPFA-inspired | 78.61 | not run | 85.11 | 1 + MLP | 0.38 + map fit |
| Two-pass, no alignment (3-stage) | shared classifier, as in BCT | 77.37 | 80.84 | 83.71 | 1 | 0.40 |
| 3-stage two-pass probe (TBSD) | token reduction + distillation | 78.26 | 81.77 | 83.95 | 1 | 0.41 |
| Resolution specialist | ResFormer-like | 82.36 | 79.59 | 83.09 | 1 | 0.35 |
| Merge specialist | ToMe-style | 79.89 | 82.90 | 84.27 | 1 | 0.44 |
| Frozen-gallery student | Suma and Tolias style | 82.78 | 81.50 | 84.45 | 2 | 0.46 |
| **`sp_mixed` (proposed)** | SFSC + ResFormer + ToMe ideas | **83.01** | **82.38** | 83.95 | 1 | 0.40 |

The per-query cascade on `sp_mixed` (DaRe-style signal) is below the fixed curve at equal cost and needs no
extra training.

`sp_mixed` clearly beats the ordinary and post-hoc baselines at 1.41, 1.12 and 0.72 GFLOPs. It is **not
clearly separated from the frozen-gallery student** (ties, plus two borderline leads), ties or slightly trails
the merge specialist under gradual merging, and is below the 216-epoch baseline at full cost.

## 8. Negative results and limitations

**Negative results** (reported, not hidden):
1. `sp_mixed` is not clearly better than the prior-work-style frozen-gallery student at the budgets it was
   trained for (two borderline leads).
2. It loses to the similar-GPU-hour baseline at full cost, and borderline at 2.22 GFLOPs.
3. The first method (3-stage TBSD) did not beat that baseline + linear map at low resolution.
4. No low-data regularisation effect at 10/25/50 % of identities.
5. Class-attention token selection is below random for cross-budget matching; cause unknown.
6. Merging has no latency benefit for single GPU images and little on single CPU images.
7. The per-query cascade is below fixed budgets.
8. The VPFA-inspired MLP does not improve on a linear map.
9. ViT-B: larger full-cost deficit (2.7 mAP, one seed).

**Limitations:**
1. **One dataset, natively 128x64.** Crops are upsampled to 256x128 and then downsized, so low-resolution
   queries lose little real information. Resolution results may not transfer.
2. **Test-set reuse.** No validation split; variants and budgets were chosen on test results.
3. **Missing baselines.** No single-pass multi-resolution-augmentation training (the simplest competitor to
   two-pass training), no small model such as ViT-Tiny or OSNet at equal GFLOPs (planned, never run), no
   low-resolution model with a re-indexed gallery, no student of the 216-epoch teacher, and no re-implementation
   of SFSC, PrunNet or D3still.
4. **Missing ablations.** No `sp_mixed` without alignment and no separate-classifier control.
5. **Merging is not official ToMe** (Section 5).
6. **One backbone with seeds** (ViT-S); ViT-B one seed, without `sp_mixed`.
7. **Statistics.** 3 seeds, a heuristic noise rule, many comparisons.
8. **Cost.** Training budgets only approximately equal; GFLOPs analytic and excluding matching; latency on one
   machine.
9. **"No data" part of the thesis not supported.** The method uses all identity labels, and low-data training
   showed no benefit. Cross-dataset behaviour (H2) is untested.
10. **Literature.** Search not exhaustive; several papers seen only in part or at abstract level.

## 9. Conclusions and contributions of iteration 1

1. **Diagnosis.** In this setting the way a query is made cheap matters: abrupt early token reduction is
   expensive, gradual merging and lower resolution much less so.
2. **Method.** One ViT trained with a cheap pass drawn from both input-side knobs stays compatible with its own
   full-cost gallery under either: 82.4-83.6 mAP at 1.1-1.6 GFLOPs. It beats the similar-GPU-hour baseline with
   post-hoc maps at low budgets (borderline at 1.62 GFLOPs), at a price of 1.16 mAP at full cost.
3. **Limit of the method's value.** A prior-work-style frozen-gallery student is not clearly
   separated from it (two borderline leads of about 1 mAP for `sp_mixed`). The practical difference is one set of weights against two.
4. **Cost reality.** Resolution gives real speed-ups where compute dominates; gradual merging often does not;
   no setting speeds up a single image on the GPU.
5. **Contribution.** A credited combination and a cost-accounted empirical comparison, including the results
   that count against it. Per-query selection is not a contribution (DaRe).

## 10. Plan for iteration 2 (proposed, for discussion; none of it has been run)

1. **Make the result trustworthy.** A validation split of training identities; single-pass multi-resolution
   and small-model baselines at equal budget; `sp_mixed` with λ = 0; a student of the 216-epoch teacher; more
   seeds for the close comparisons; merging aligned with official ToMe.
2. **Test generality.** A second dataset with higher native resolution (CUHK03-NP or MSMT17; needs sign-up).
   This decides whether resolution results hold and whether merging has a role.
3. **Then, a mechanism, conditional on 1 and 2.** An escalation that reuses the cheap pass (ThinkingViT-style
   token recycling) while staying compatible with the full-cost gallery. DaRe, MatFormer and HashReID
   (WACV 2024, unread) must be read in full first, because per-query re-ID budgets with reuse already exist.
4. **Re-implement one published multi-budget method** (SFSC- or PrunNet-style) in our setting.

## Reproduction
Exact commands and per-run tables: `PROGRESS_LOG.md`; walk-through that writes to new paths:
`STUDY_GUIDE.md` Section 15. **Commands copied from `PROGRESS_LOG.md` or `PROGRESS.md` write to
the original result paths: `train.py` refuses them, the evaluation scripts overwrite them.**
Training `train.py` (`--sd-mode mixed`; `--teacher` for the frozen-gallery student); diagnostics `mechanism.py`
(+ `aggregate_mech.py`); token sweep `evaluate.py`; cascade `adaptive.py` (+ `aggregate_adaptive.py`); latency
`bench_latency.py --extra`; maps `nonlinear_map.py` (+ `aggregate_nonlinear.py`); mechanism `mech_diag.py`
(+ `aggregate_mech_diag.py`); queues (note: the queue scripts commit and push automatically) `run_iter3.sh`, `run_iter4.sh`. Code, logs and metrics: GitHub (private);
checkpoints local.
