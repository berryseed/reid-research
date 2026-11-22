# Study guide: cheap queries against a full-cost gallery in ViT person re-identification

Written 2026-11-17 for the author of this project; revised 2026-11-22 to match the audited report. It explains, from the beginning, everything needed to
understand, reproduce and defend iteration 1. Read it top to bottom once; afterwards use it as a reference.
Numbers are from our own runs (Market-1501, ViT-S/16, mean of 3 seeds unless stated); sources are given in
`REPORT_ITERATION1.md`; known issues are in `SUPERVISOR_REVIEW_AUDIT.md`. Words in **bold** are defined where they first appear.

Contents
1. The problem in plain words
2. Person re-identification: data, task, metrics
3. The model: a Vision Transformer, step by step
4. How a re-ID model is trained (the standard recipe we use)
5. Making a query cheaper: tokens and resolution
6. Asymmetric retrieval and "drift"
7. What we built: two-pass multi-budget training (`sp_mixed`)
8. The baselines, and why each one is there
9. Results and what they mean
10. Per-query budget selection (the cascade)
11. Cost: GFLOPs, latency, GPU-hours
12. Negative results and limitations
13. Related work in one page
14. Implementation map (which file does what)
15. How to reproduce
16. Glossary

---

## 1. The problem in plain words

A camera network sees a person. We want to find every other image of that person in a large collection taken
by other cameras. A neural network turns each image into a list of numbers (a vector, called an
**embedding**); images of the same person should give vectors that point in similar directions.

The collection (the **gallery**) is processed once, in advance, on a big machine. The image we search with
(the **query**) arrives later and may be processed on a weak device, so we want the query to be cheap to
compute. The catch: a cheap query vector must still be *comparable* with gallery vectors that were computed
the expensive way. If it is not, search quality can collapse.

Our project thesis is "accuracy per unit of compute, labels and training time". Iteration 1 asks:
how much accuracy do we lose when only the query is made cheap, why, and what is the cheapest way to train a
model that stays compatible?

## 2. Person re-identification: data, task, metrics

### 2.1 Market-1501
- 32,668 cropped images of 1,501 people from 6 cameras on a university campus.
- **Training set:** 12,936 images of 751 people. **Test set:** 750 other people (no overlap with training).
- **Query set:** 3,368 images. **Gallery:** 19,732 files, of which 3,819 are "junk" (label −1: bad crops)
  and are removed by every standard toolkit, leaving **15,913** gallery images; 2,798 of those are
  **distractors** (label 0000: people or background not in the query set), kept on purpose to make search harder.
- File names encode identity and camera, e.g. `0002_c1s1_000451_03.jpg` = person 2, camera 1.
- Crops are natively about **128x64 pixels**. We resize them to 256x128 (the usual re-ID input), so the network
  sees an upsampled image. This matters for the low-resolution results (Section 12).

### 2.2 The task, precisely
For each query, sort all gallery images by similarity (we use the cosine of the embeddings; Section 3.6).
Gallery images of the same person **taken by the same camera** are ignored for that query (they would be too
easy). A good model puts the remaining same-person images at the top.

### 2.3 Metrics
- **Rank-1 (R1):** the percentage of queries whose top-ranked gallery image is a correct match.
- **Average precision (AP)** of one query: walk down the ranked list; every time you reach a correct match,
  note the precision so far (correct matches / items seen); average those numbers. If the correct matches are
  at ranks 1, 3 and 6, AP = (1/1 + 2/3 + 3/6) / 3 = 0.72.
- **mAP** = mean of AP over all queries, written as a percentage. mAP rewards finding *all* matches early, not
  only the first, so it is the main metric here. Our implementation: `evaluate_rank` in `common.py`.

## 3. The model: a Vision Transformer, step by step

We use **ViT-S/16** ("small", patches of 16x16 pixels), pretrained on ImageNet, from the `timm` library.

### 3.1 Patches and tokens
The 256x128 image is cut into 16x16 squares: 16 rows x 8 columns = **128 patches**. Each patch (16·16·3 = 768
numbers) is multiplied by a learned matrix to give a vector of width **d = 384**. Each such vector is a
**token**. A learned extra token, the **class token** (`[CLS]`), is put in front: **129 tokens** in total.
A learned **position embedding** (one vector per position) is added so the network knows where each patch was.

### 3.2 A transformer block (repeated 12 times)
Each block updates all tokens together in two steps:
1. **Self-attention.** Every token makes a *query* q, *key* k and *value* v vector (three learned linear maps).
   Token i looks at token j with weight `softmax_j(q_i · k_j / sqrt(d_head))`; its new value is the weighted
   sum of all v_j. This lets every patch gather information from every other patch. ViT-S uses 6 **heads**
   (6 parallel attentions of width 64).
2. **MLP.** Each token separately goes through a small two-layer network (384 → 1536 → 384).
Both steps use a residual connection (`x = x + step(LayerNorm(x))`).

### 3.3 The output embedding
After 12 blocks, the class token has gathered information from the whole image. We take it (after a final
LayerNorm) as the image's **feature** (384 numbers).

### 3.4 The BNNeck
Standard re-ID trick ("bag of tricks", Luo et al. 2019): the feature goes through a **BatchNorm** layer (the
**BNNeck**); its output is used for retrieval and feeds the identity classifier. BatchNorm normalises each of
the 384 dimensions using statistics of the batch during training and stored running averages at test time.

### 3.5 The classifier (training only)
A linear layer from 384 to 751 outputs (one per training identity) gives class scores (**logits**). At test
time the classifier is thrown away: test people are new, so we only use embeddings.

### 3.6 Cosine similarity
`cos(a, b) = a·b / (|a| |b|)`, between −1 and 1. We L2-normalise embeddings (divide by their length), so the
cosine is just the dot product. Higher = more similar.

## 4. How a re-ID model is trained (our standard recipe)

- **Batches by identity (PK sampling):** each batch has P = 16 people x K = 4 images = 64 images, so every
  batch contains positive pairs (same person) and negative pairs.
- **Identity loss:** cross-entropy on the 751 logits with **label smoothing** 0.1 (0.1 of the target is spread
  evenly over all 751 people, so the true person gets 0.9 + 0.1/751; reduces over-confidence).
- **Triplet loss (batch-hard):** for each image (the *anchor*), take the furthest same-person image (hardest
  positive) and the closest other-person image (hardest negative); loss = max(0, d(anchor, pos) −
  d(anchor, neg) + 0.3). It pushes same-person images together and others apart by a margin of 0.3.
- **Optimiser:** SGD with momentum 0.9, weight decay 1e-4, learning rate 0.008, 5 warm-up epochs then cosine
  decay; 120 epochs x 202 steps; mixed precision (AMP) on an RTX 5080. One run takes about 0.22 GPU-hours.
- Our ordinary model ("baseline") reaches **84.45 ± 0.49 mAP** at full cost. Trained for 216 epochs (the
  **similar-GPU-hour baseline**, called "equal-GPU-hour" in older documents) it reaches 85.11.

## 5. Making a query cheaper: tokens and resolution

The cost of a ViT grows with the number of tokens n: the linear layers cost about n·d², attention about n²·d.
Two ways to reduce n without changing the weights:

### 5.1 Token merging (ToMe-style, after Bolya et al., ICLR 2023)
In each block, after attention, the tokens are split alternately into two sets A and B. For each token in A
find its most similar token in B (cosine of the attention keys). Merge the r most similar pairs by averaging
them (weighted by how many original patches each already represents). After 12 blocks with r = 11, the 128
patch tokens shrink gradually and the cost falls to 1.41 GFLOPs (48 % of 2.93). This is **gradual merging**
(`merge_r` in our code). Our earlier **3-stage** schedule removed up to half the tokens at once inside blocks 4,
7 and 10; that turned out to be much worse (Section 9.2).

**Our merging is not official ToMe.** The difference is which tokens may pair up:
- official ToMe alternates over the whole sequence, class token included: `A = {CLS, p1, p3, ...}`,
  `B = {p0, p2, ...}`, and CLS is forbidden to merge;
- our code alternates over the patches only: `A = {p0, p2, ...}`, `B = {p1, p3, ...}`.

Each token of A picks a partner in B, and several may pick the same one, so swapping the sets changes the result.
Official ToMe also re-sorts the surviving tokens, which changes the split in the next block. Token counts and
GFLOPs are the same; the merged tokens are often not (one step at r = 11 on random input: the same in 10 of 64
samples). An earlier check reported "difference 0" because it removed the class token before calling the
reference, which made the two splits equal by construction.

### 5.2 Lower input resolution
Resize the image before the network: 224x112 gives 14x7 = 98 patches, 192x96 gives 12x6 = 72, 160x80 gives 50, 128x64 gives 32. The
position embedding (learned for a 16x8 grid) is resized to the new grid by bicubic interpolation
(`resample_abs_pos_embed` from timm). Costs: 2.22, 1.62, 1.12 and 0.72 GFLOPs.

### 5.3 Other selection rules we tested
- **Random:** keep a random subset of tokens.
- **Class-attention top-k (EViT, ICLR 2022):** keep the tokens the class token attends to most; fuse the rest
  into one token. Surprisingly, this scored *below random* for our purpose (Section 12).

## 6. Asymmetric retrieval and "drift"

- **Symmetric:** query and gallery computed the same way. **Asymmetric:** query cheap, gallery full cost.
  Asymmetric is the realistic setting: the gallery is computed once, at full quality.
- **Drift:** the cheap embedding of an image is not exactly its full-cost embedding. We measure it with
  `q_cos = cos(f(x; cheap), f(x; full))` averaged over queries (1 = no drift).
- An ordinary model drifts a lot under strong reduction: at 128x64, `q_cos` ≈ 0.61 and mAP drops from 84.45 to
  60.13. The model was never trained to give the same answer from fewer tokens.
- The drift is **mostly per image**: a shared offset explains 3-18 % of it. Correcting the average offset gains
  0 to +3.8 mAP. A closed-form linear map (ridge regression from cheap to full embeddings, fitted on training
  images) gains −0.1 to +8.9, most where drift is large.
- **A trap in interpreting this.** For the baseline, cheap-vs-cheap mAP is lower than cheap-vs-full. That does
  *not* prove the cheap embedding lost information rather than moved: cheap-vs-cheap also degrades the gallery.
  At 128x64 a linear map lifts cheap-vs-full to 68.88, far above cheap-vs-cheap (48.07).

## 7. What we built: two-pass multi-budget training (`sp_mixed`)

### 7.1 Idea
Teach one network to give (almost) the same embedding whatever the budget, so that any cheap query can be
compared with the full-cost gallery.

### 7.2 Architecture (one network, two passes per training step)
```
                    same weights θ, same BNNeck, same classifier W
 image x ──► full pass  (256x128, 129 tokens) ──► feat_f ──► BN ──► e_f ──► W ──► logits_f
        └──► cheap pass (budget b sampled per step) ──► feat_c ──► BN ──► e_c ──► W ──► logits_c
             b = a low resolution {224x112, 192x96, 160x80, 128x64}   (prob. 1/2)
               or gradual merging r ∈ {3, 5, 7, 9, 11}               (prob. 1/2)

 loss = CE(logits_f, y) + Tri(feat_f, y)            ← standard re-ID losses, full pass
      + CE(logits_c, y) + Tri(feat_c, y)            ← same losses, cheap pass, SAME classifier
      + λ · mean_i [ 1 − cos(e_c,i , stopgrad(e_f,i)) ]   ← pull cheap onto full (λ = 1)
```
At test time: the gallery uses the full pass; a query uses whichever cheap budget the device can afford.

### 7.3 The mathematics, term by term
Let `f(x; b)` be the L2-normalised BNNeck embedding at budget b, y the identity label.
1. `L_id(b) = CE(W f̃(x; b), y)` with label smoothing, where `f̃` is the BNNeck output *before* normalisation.
   W is **shared** between the two passes, so both embeddings must place a person on the same side of the same
   class boundaries (the idea of BCT, Shen et al. CVPR 2020). We never tested how much this alone contributes.
2. `L_tri(b)`: batch-hard triplet on the pre-BNNeck feature, as in the baseline.
3. `L_align = mean_i [1 − cos(f(x_i; b), sg f(x_i; full))]`. `sg` = stop-gradient: the full embedding is treated
   as a fixed target in this term, so it has no direct gradient. The shared weights still change it indirectly.
   This is the same family as, not identical to, the feature terms of asymmetric retrieval. With `c_i = 1 − cos`
   for image i: ours is `mean(c_i)`, D3still's `L_f` is `sqrt(sum c_i²) / n`, Suma and Tolias's `l_abs` is `c_i²`.
Total: `L = L_id(full) + L_tri(full) + L_id(b) + L_tri(b) + λ L_align`, λ = 1.

### 7.4 Why "mixed"
We tried three versions (3 seeds each): cheap pass always low resolution (`sp_lowres`), always gradual merging
(`sp_merge_r`), or either at random (`sp_mixed`). Each specialist carries over only partly to the other knob;
`sp_mixed` works on both, at or slightly below the merge specialist under merging (Section 9.3). So `sp_mixed` is
our method. It was chosen on test results (Section 12).

### 7.5 What is NOT new (say this clearly)
- **Components:** two passes with a shared classifier (BCT, SFSC), alignment to the full embedding (D3still,
  Suma and Tolias), multi-resolution training with a detached higher-resolution teacher (ResFormer), token
  merging (ToMe).
- **Settings:** one ViT with cheap queries against its full gallery (MatFormer), resolution-dependent compute
  with compatible embeddings (BTNet), per-query budgets in re-ID (DaRe).

What we did not find is the combination of both input knobs in one compatible ViT re-ID model with a cost
account. That is an empirical contribution. The revised project rule (2026-11-17) accepts a combination of
existing techniques if prior work is credited.

## 8. The baselines, and why each one is there

| Baseline | What it tests |
| --- | --- |
| Ordinary model, 120 epochs, cheap queries without retraining | How bad the problem is without any fix |
| **Similar-GPU-hour baseline**: ordinary model trained 216 epochs | Our method trains about 1.8x longer; is the gain just "more training"? |
| Same + **linear map** from cheap to full embeddings | A nearly free training-free correction |
| Same + **VPFA-inspired MLP map** | Does a non-linear correction close the gap? (No) |
| Single-knob specialists (`sp_lowres`, `sp_merge_r`) | Is mixing the knobs needed? |
| 3-stage TBSD probe (our first attempt) | Our own earlier method, kept for honesty |
| Multi-budget training without the alignment term | Is the alignment loss needed? (3-stage schedule only) |
| **Frozen-gallery student** (`fg_mixed`) in the style of Suma and Tolias | Freeze the gallery model, train a copy for cheap queries with a label-free loss (the batch sampler still uses labels) |

Not run, and important: single-pass multi-resolution augmentation, and a small model at equal GFLOPs.

### 8.1 When is a difference a result?
With three seeds per method there are two ways to measure the noise of a difference:
- **combined:** `sqrt(sd_a² + sd_b²)`, treating all runs as independent;
- **paired:** subtract seed by seed, then take the std of those three differences. Seed i shares batch order
  and head initialisation across methods, so some noise cancels.

We divide the mean difference by each. **Clear:** at least 2 both ways. **Tie:** below 2 both ways (this means
"not separated", not "equal"). **Borderline:** the two disagree.

Worked example, `sp_mixed` vs the student at r = 11. The means are 82.38 and 81.50, so the difference is 0.88.
- combined: `sqrt(0.27² + 0.54²) = 0.61`, ratio 1.4;
- paired: std of the differences 0.32, ratio 2.8;
- verdict: borderline. Do not claim it; more seeds would settle it.

This convention is a heuristic, fixed after the results were known. It is not a significance test.

## 9. Results and what they mean

### 9.1 The problem is real
Ordinary model, asymmetric mAP: full 84.45; low resolution 192x96 81.38, 160x80 75.81, 128x64 60.13; gradual
merging r = 11: 77.45; 3-stage merging at half the tokens: 68.56.

### 9.2 How you cut matters
At nearly the same cost the ordinary model scores 68.55 (3-stage, 1.46 GFLOPs) and 77.45 (gradual merging,
1.41). Low resolution, interpolated, gives about 79.0 at 1.41, only about one std above gradual merging (std
1.39). Removing tokens early is what hurts: halving only at block 4 costs 3.8 mAP, only at block 10 costs 0.2.
Our first method looked +13 mAP better because it was compared with the worst schedule. Against the 216-epoch
baseline + linear map it is better under merging (+2.1 at r = 11) and not better at low resolution.

### 9.3 The method (`sp_mixed`) vs baselines (asymmetric mAP, 3 seeds)
| Query | GFLOPs | 216-ep baseline, best map | Frozen-gallery student | `sp_mixed` | vs baseline | vs student |
| --- | --- | --- | --- | --- | --- | --- |
| full | 2.93 | 85.11 | 84.45 | 83.95 | −1.16, clear | −0.50, tie |
| 192x96 | 1.62 | 82.96 | 83.68 | 83.60 | +0.64, borderline | −0.07, tie |
| merge r = 11 | 1.41 | 79.70 | 81.50 | 82.38 | +2.68, clear | +0.88, borderline |
| 160x80 | 1.12 | 79.16 | 82.78 | 83.01 | +3.85, clear | +0.23, tie |
| 128x64 | 0.72 | 68.88 | 80.48 | 81.50 | +12.61, clear | +1.02, borderline |

How to read it:
- At 1.41 GFLOPs and below, our method clearly beats the corrected baseline. At 1.62 the win is borderline.
- At higher budgets the advantage is gone: a tie at 1.94, borderline behind at 2.22, clearly 1.16 behind at
  full cost.
- It is **not clearly better than the frozen-gallery student**: ties, plus two borderline leads of about 1 mAP.
  The practical difference is one set of weights against two.
- "Best map" is picked on test, which favours the baseline. Non-linear maps were not tested on merging.
- Training budgets are only similar (Section 11.3).

### 9.4 Latency (measured, not estimated)
On the GPU with one image at a time, every resolution costs about 6 ms: the GPU is waiting on overheads, not on
arithmetic. Gradual merging is even **2.3x slower** there (its matching step runs in every block). Savings
appear with large batches and on the CPU: 160x80 is 2.67x faster at GPU batch 256 and 1.69x on CPU batch 1;
128x64 3.9x and 2.0x; 192x96 only 1.9x and 1.4x. Resolution is mostly faster than merging at similar GFLOPs. In
accuracy it is ahead at about 1.41 GFLOPs and equal at 1.94. Between two sessions CPU timings varied 9-17 % at
batch 1 and 17-46 % at batch 8, so treat the batch-8 numbers as rough.

### 9.5 ViT-B (one seed, indicative)
Same pattern, larger full-cost deficit (2.7 mAP), and worse accuracy per GFLOP than ViT-S. ViT-S stays.

## 10. Per-query budget selection (the cascade)

Full numbers: `REPORT_ITERATION1.md` Section 6.9 and figure `runs/adaptive.png`.

**Result in brief.** The cascade loses to giving every query one intermediate resolution at the same average
cost (our method: −0.07 to −0.68 mAP), because an escalated query pays twice and the fixed-budget curve is
concave (it rises steeply, then flattens). The routing signal itself is better than random (e.g. baseline from
160x80 at p = 0.3: 80.91 vs random 78.32, drift oracle 80.80), but the idea is prior work (DaRe, CVPR 2018). It is
reported as a negative result; reusing the cheap pass is possible iteration-2 work.

The idea in brief:
- **Intuition:** most queries are easy; spend full compute only on the hard ones.
- **Cascade:** embed every query cheaply (e.g. 128x64) and search. Compute a confidence score from that search
  (the cheap query's highest cosine similarity to the gallery, `top-1`). Re-embed the fraction p of queries
  with the lowest confidence at full cost and search again.
- **Average cost:** `g_cheap + p · g_full` (the cheap pass is already paid for the escalated queries).
- **Controls:** random escalation (does the score carry information?) and a drift oracle that escalates the
  queries whose cheap embedding moved most. It needs the full embedding and is **not an upper bound**: it can be
  beaten (80.91 vs 80.80 above), because the most-moved query is not always the one that gains most.
- **Limit:** p is applied to the ranking of all test queries at once, so the rule assumes they are known together.
- **Prior work:** DaRe (CVPR 2018) routes re-ID queries by nearest-gallery distance, the same signal. RANet
  (CVPR 2020) and DVT (NeurIPS 2021) use cascades for classification with softmax confidence.

## 11. Cost: GFLOPs, latency, GPU-hours

### 11.1 GFLOPs (analytic, `vit_gflops` in `common.py`)
Counted as multiply-accumulates. Per block with n tokens, width d = 384, MLP width h = 1536:
`4 n d²` (q, k, v and output projections) + `2 n² d` (attention scores and weighted sum) + `2 n d h` (MLP).
For n = 129: 76.1 M + 12.8 M + 152.2 M = 241 M per block; x 12 = 2.89 G; plus the patch embedding
(128 · 768 · 384 = 37.7 M) = **2.93 GFLOPs**. For merging, n changes from block to block (we record the actual
token counts of the forward pass). For low resolution, n and the patch-embedding term shrink.

### 11.2 Latency
`bench_latency.py`: random input, warm-up, median of repeated runs, GPU (mixed precision) at batch 1 and 256,
CPU (8 threads) at batch 1 and 8, with nothing else running. Report latency per regime, because GFLOPs do not
predict it.

### 11.3 Training cost
GPU-hours from the training log; peak VRAM from PyTorch.

| Run | GPU-h per run | Peak allocated VRAM |
| --- | --- | --- |
| Baseline, 120 epochs | 0.22-0.23 | 1.81 GB |
| Baseline, 216 epochs | 0.36-0.40 (mean 0.38) | 1.81 GB |
| `sp_mixed` | 0.35-0.44 (mean 0.40) | 3.17 GB |
| Frozen-gallery student | 0.23-0.24, plus 0.22 for its teacher | 1.83 GB |

Identical configurations vary by up to about 25 % in time. The project total is 50 real runs, 13.38 GPU-hours.

## 12. Negative results and limitations

Negative results (keep them in the report; they are part of the contribution):
1. No clear win over the prior-work-style frozen-gallery student.
2. Worse than the similar-GPU-hour baseline at full cost; borderline worse at 2.22 GFLOPs.
3. Our first method (3-stage TBSD) did not beat that baseline + linear map at low resolution.
4. No low-data regularisation effect at 10/25/50 % of identities.
5. Class-attention token selection (EViT rule) is below random for cross-budget matching: clear in most models,
   a tie on the baselines at keep 0.7. Cause unknown.
6. Merging gives no single-image GPU speed-up.
7. The per-query cascade is below fixed budgets.
8. A non-linear post-hoc map does not improve on a linear one.
9. Per-budget BatchNorm statistics are not the cause of the full-cost deficit.

Limitations:
1. One dataset, natively 128x64, so low resolution loses little real information here; results may not
   transfer to higher-resolution data.
2. No validation split: the method and budgets were chosen on test results.
3. Missing baselines: single-pass multi-resolution augmentation, a small model at equal GFLOPs, a student of
   the 216-epoch teacher.
4. Missing ablations: `sp_mixed` without alignment, and with separate classifiers.
5. One backbone with 3 seeds (ViT-S); ViT-B one seed. Three seeds cannot settle borderline results.
6. Merging is not official ToMe.
7. Narrow novelty: a combination of existing techniques (credited).
8. No re-implementation of SFSC, PrunNet or D3still; the frozen-gallery student is an adaptation of Suma and
   Tolias, not their method.
9. Training budgets only similar; GFLOPs analytic; latency on one machine.

## 13. Related work in one page

| Paper | One-line summary | Relation to us |
| --- | --- | --- |
| **DaRe (CVPR 2018)** | Budgeted person re-ID: queries exit early by nearest-gallery distance | Already did our per-query idea |
| **MatFormer (NeurIPS 2024)** | One ViT; smaller sub-models query a gallery from the largest | Already did one-ViT cheap-query retrieval (width, not re-ID) |
| **BTNet (BMVC 2023)** | Resolution-specific branches with a shared trunk and compatible embeddings | Already linked resolution, compute and compatibility (faces) |
| VPFA (ICML 2026) | Post-hoc MLP from low- to high-resolution features | Basis of our non-linear-map baseline |
| ThinkingViT (CVPR 2026) | Nested ViT that reuses the first stage when it escalates | Template for iteration 2 |
| SFSC (CVPR 2023) | One CNN with width-sliced sub-models whose features are mutually compatible; Market-1501, MSMT17 | Same goal, CNN width instead of input size |
| D3still (CVPR 2024) | Small query network distilled to rank like a large gallery network | Our alignment loss is in the family of its feature term |
| Suma and Tolias (WACV 2023) | Frozen gallery network; same-architecture student for small-resolution queries, label-free | Our frozen-gallery baseline is adapted from it |
| ResFormer (CVPR 2023) | Multi-resolution ViT training; detached high-resolution class token teaches low resolution | Mechanism of our low-resolution pass; no retrieval |
| PrunNet (CVPR 2025) | One network, pruned sub-networks compatible with the dense one, incl. ViT-S and re-ID | One-network multi-budget compatibility, but weights pruned, no measured speed-up |
| DVT (NeurIPS 2021) / RANet (CVPR 2020) | Cascades of increasing tokens / resolution with confidence-based early exit | Classification relatives of our cascade |
| ToMe (ICLR 2023), EViT (ICLR 2022) | Token merging; attention-based token keeping | Our token-reduction tools |
| Cross-resolution re-ID (e.g. RFD, 2021) | Low-resolution queries vs high-resolution gallery as a degradation; query upsampled | Same geometry, but no compute saving |

## 14. Implementation map

| File | What it does |
| --- | --- |
| `common.py` | Market-1501 parsing, PK sampler, `ReIDViT` (ViT + BNNeck + classifier, token reduction modes, low-resolution input), analytic GFLOPs, triplet loss, mAP/CMC |
| `train.py` | Training. `--sd-mode mixed` = our method; `--teacher` = frozen-gallery student; checkpoints every epoch, STOP file to pause, `--resume` |
| `evaluate.py` | Token-reduction sweep (symmetric and asymmetric mAP vs GFLOPs) |
| `mechanism.py` | Drift diagnostics, linear-map correction, low-resolution queries; `--gallery-ckpt` for a separate gallery model |
| `adaptive.py` | Per-query cascade evaluation |
| `bench_latency.py` | Latency and throughput (`--extra` adds low-resolution and gradual-merge rows) |
| `aggregate_mech.py` | Mean ± std over seeds |
| `nonlinear_map.py`, `aggregate_nonlinear.py` | VPFA-inspired map baseline |
| `mech_diag.py`, `aggregate_mech_diag.py` | Mechanism diagnostic |
| `run_iter2.sh`, `run_iter3.sh`, `run_iter4.sh` | Past experiment queues. **They commit and push automatically** |

Key code points: the cheap pass is chosen in `train.py` (`--sd-mode mixed` picks `lowres` or `merge_r` per
step); the alignment term is `(1 - F.cosine_similarity(bn_s, bn.detach())).mean()`; low resolution in
`ReIDViT.backbone(..., hw=...)` resizes the image and the position embedding.

## 15. How to reproduce (Git Bash, from the project folder)

**Every output below goes to a new `repro` path on purpose.** `train.py` refuses to overwrite an existing run, also
with `--resume`, but the other scripts have no guard: pointing their `--out` at an existing file replaces it.
Do not copy commands with original paths from the logs.

```bash
# our method, seed 0
.venv/Scripts/python.exe train.py --data C:/Users/user/data/Market-1501-v15.09.15 --out runs/repro_sp_mixed_s0 --seed 0 --sd-mode mixed
# diagnostics (asymmetric mAP for every cheap setting)
.venv/Scripts/python.exe mechanism.py --data C:/Users/user/data/Market-1501-v15.09.15 --ckpts runs/repro_sp_mixed_s0/last.pth --out runs/repro_sp_mixed_s0/mech.csv
# frozen-gallery student on the existing baseline of the same seed (the teacher is only read)
.venv/Scripts/python.exe train.py --data C:/Users/user/data/Market-1501-v15.09.15 --out runs/repro_fg_mixed_s0 --seed 0 --teacher runs/base_s0/last.pth --sd-mode mixed --epochs 96
# its evaluation: the gallery comes from the teacher
.venv/Scripts/python.exe mechanism.py --data C:/Users/user/data/Market-1501-v15.09.15 --ckpts runs/repro_fg_mixed_s0/last.pth --gallery-ckpt runs/base_s0/last.pth --out runs/repro_fg_mixed_s0/mech.csv
# cascade
.venv/Scripts/python.exe adaptive.py --data C:/Users/user/data/Market-1501-v15.09.15 --ckpts runs/repro_sp_mixed_s0/last.pth --out runs/repro_adaptive.csv
# latency (machine idle)
.venv/Scripts/python.exe bench_latency.py --extra --out runs/repro_latency_extra.csv
```
Run each line once; a second run overwrites the first `repro` CSVs (training refuses). The `aggregate*.py` scripts write to the
original summary files by default, so do not run them just to look at results.

Expected: about 0.4 GPU-hours per two-pass run on an RTX 5080; full-cost mAP around 84. Expect close, not
identical, numbers: GPU training is not bit-exact, and merge evaluation alone varies by up to 0.05 mAP. These
commands have not been re-run end to end.

## 16. Glossary
- **AMP:** automatic mixed precision; 16-bit arithmetic where safe, for speed.
- **Asymmetric retrieval:** query and gallery embedded by different (here: cheaper vs full) computations.
- **BNNeck:** BatchNorm after the feature; its output is the retrieval embedding.
- **Budget:** how much compute a query may use (GFLOPs); set by resolution or merge rate.
- **Cascade:** cheap first, more expensive only if needed.
- **Class token ([CLS]):** extra token whose final state is the image representation.
- **Compatibility:** embeddings from different computations can be compared directly.
- **Cosine similarity:** dot product of length-normalised vectors.
- **Drift:** difference between an image's cheap and full embeddings.
- **Similar-GPU-hour baseline:** a baseline given about as much training time as the method.
- **Clear / borderline / tie:** our three verdicts on a difference (Section 8.1).
- **GFLOPs:** billions of multiply-accumulates per image (analytic).
- **Gallery / query:** the searched collection / the image we search with.
- **Label smoothing:** soft target distribution for cross-entropy.
- **Linear (ridge) map:** a matrix fitted by least squares with a penalty, mapping cheap to full embeddings.
- **mAP, Rank-1:** Section 2.3.
- **Oracle:** a control that uses information unavailable at test time. Our drift oracle is not an upper bound.
- **PK sampling:** P identities x K images per batch.
- **Seed:** the random-number starting point; different seeds = independent repetitions.
- **Stop-gradient (sg):** treat a value as a constant in back-propagation.
- **Token:** one vector per image patch (plus the class token).
- **Token merging (ToMe-style):** averaging similar tokens to reduce their number; ours differs from official ToMe.
- **Triplet loss:** pulls same-person embeddings together, pushes others apart by a margin.
- **ViT-S/16:** Vision Transformer, small (width 384, 12 blocks, 6 heads), 16x16 patches.
