# Study guide: cheap queries against a full-cost gallery in ViT person re-identification

Written 2026-11-17 for the author of this project. It explains, from the beginning, everything needed to
understand, reproduce and defend iteration 1. Read it top to bottom once; afterwards use it as a reference.
Numbers are from our own runs (Market-1501, ViT-S/16, mean of 3 seeds unless stated); sources are given in
`REPORT_ITERATION1.md`. Words in **bold** are defined where they first appear.

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
the expensive way. If it is not, search quality collapses.

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
- **Identity loss:** cross-entropy on the 751 logits with **label smoothing** 0.1 (the target is 0.9 on the
  true person and 0.1 spread over the rest; reduces over-confidence).
- **Triplet loss (batch-hard):** for each image (the *anchor*), take the furthest same-person image (hardest
  positive) and the closest other-person image (hardest negative); loss = max(0, d(anchor, pos) −
  d(anchor, neg) + 0.3). It pushes same-person images together and others apart by a margin of 0.3.
- **Optimiser:** SGD with momentum 0.9, weight decay 1e-4, learning rate 0.008, 5 warm-up epochs then cosine
  decay; 120 epochs x 202 steps; mixed precision (AMP) on an RTX 5080. One run takes about 0.22 GPU-hours.
- Our ordinary model ("baseline") reaches **84.45 ± 0.49 mAP** at full cost. Trained for 216 epochs (the
  **equal-GPU-hour baseline**) it reaches 85.11.

## 5. Making a query cheaper: tokens and resolution

The cost of a ViT grows with the number of tokens n: the linear layers cost about n·d², attention about n²·d.
Two ways to reduce n without changing the weights:

### 5.1 Token merging (ToMe, Bolya et al., ICLR 2023)
In each block, after attention, the tokens are split alternately into two sets A and B. For each token in A
find its most similar token in B (cosine of the attention keys). Merge the r most similar pairs by averaging
them (weighted by how many original patches each already represents). After 12 blocks with r = 11, the 128
patch tokens shrink gradually and the cost falls to 1.41 GFLOPs (48 % of 2.93). This is **gradual merging**
(`merge_r` in our code). Our earlier **3-stage** schedule removed half the tokens at once after blocks 4, 7 and
10; that turned out to be much worse (Section 9.2).

### 5.2 Lower input resolution
Resize the image before the network: 192x96 gives 12x6 = 72 patches, 160x80 gives 50, 128x64 gives 32. The
position embedding (learned for a 16x8 grid) is resized to the new grid by bicubic interpolation
(`resample_abs_pos_embed` from timm). Costs: 1.62, 1.12 and 0.72 GFLOPs.

### 5.3 Other selection rules we tested
- **Random:** keep a random subset of tokens.
- **Class-attention top-k (EViT, ICLR 2022):** keep the tokens the class token attends to most; fuse the rest
  into one token. Surprisingly, this was *worse than random* for our purpose (Section 12).

## 6. Asymmetric retrieval and "drift"

- **Symmetric:** query and gallery computed the same way. **Asymmetric:** query cheap, gallery full cost.
  Asymmetric is the realistic setting: the gallery is computed once, at full quality.
- **Drift:** the cheap embedding of an image is not exactly its full-cost embedding. We measure it with
  `q_cos = cos(f(x; cheap), f(x; full))` averaged over queries (1 = no drift).
- An ordinary model drifts a lot under abrupt reduction: at 128x64, `q_cos` ≈ 0.61 and mAP drops from 84.45 to
  60.13. The model was never trained to give the same answer from fewer tokens.
- We found the drift is **per image**, not a shared shift: subtracting an average offset fixes almost nothing
  (offset fraction 0.03-0.18). A closed-form linear map (ridge regression from cheap to full embeddings,
  fitted on training images) recovers part of it (+1 to +4 mAP).

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
1. `L_id(b) = CE(W f(x; b), y)` with label smoothing. Because W is **shared** between the two passes, both
   embeddings must place a person on the same side of the same class boundaries: this alone creates a lot of
   compatibility (the idea of BCT, Shen et al. CVPR 2020).
2. `L_tri(b)`: batch-hard triplet on the pre-BNNeck feature, as in the baseline.
3. `L_align = mean_i [1 − cos(f(x_i; b), sg f(x_i; full))]`. `sg` = stop-gradient: the full embedding is treated
   as a fixed target in this term, so the cheap pass moves towards the full one and not the reverse. This is
   the standard feature term of asymmetric retrieval (D3still's `L_f`, Suma and Tolias's `l_abs`).
Total: `L = L_id(full) + L_tri(full) + L_id(b) + L_tri(b) + λ L_align`, λ = 1.

### 7.4 Why "mixed"
We tried three versions (3 seeds each): cheap pass always low resolution (`sp_lowres`), always gradual merging
(`sp_merge_r`), or either at random (`sp_mixed`). Each specialist only becomes compatible on its own knob;
`sp_mixed` is compatible on both, at almost no cost on either (Section 9.3). So `sp_mixed` is our method.

### 7.5 What is NOT new (say this clearly)
Two passes through shared weights with a shared classifier (SFSC's BCT-S baseline), the alignment loss
(D3still, Suma and Tolias), multi-resolution training with a detached higher-resolution teacher (ResFormer),
token merging (ToMe). Our contribution is applying and combining them for **input-side compute knobs** in
compatible re-ID retrieval, and evaluating the result against equal-budget baselines with measured latency.
The revised project rule (2026-11-17) accepts a combination of existing techniques if prior work is credited.

## 8. The baselines, and why each one is there

| Baseline | What it tests |
| --- | --- |
| Ordinary model, 120 epochs, cheap queries without retraining | How bad the problem is without any fix |
| **Equal-GPU-hour baseline**: ordinary model trained 216 epochs | Our method trains about 1.6-2x longer; is the gain just "more training"? |
| Same + **linear map** from cheap to full embeddings | A nearly free training-free correction |
| 3-stage TBSD probe (our first attempt) | Our own earlier method, kept for honesty |
| Multi-budget training without the alignment term | Is the alignment loss needed? |
| **Frozen-gallery student** (`fg_mixed`) in the style of Suma and Tolias | The closest prior-work scheme: freeze the gallery model, train a copy for cheap queries without labels |

Rule used throughout: 3 seeds per configuration, mean ± standard deviation; a difference smaller than about
two standard deviations is not called a result.

## 9. Results and what they mean

### 9.1 The problem is real
Ordinary model, asymmetric mAP: full 84.45; low resolution 192x96 81.38, 160x80 75.81, 128x64 60.13; gradual
merging r = 11: 77.45; 3-stage merging at half the tokens: 68.56.

### 9.2 How you cut matters more than how much
At about the same cost (1.4-1.5 GFLOPs) the ordinary model scores 68.6 (3-stage), 77.5 (gradual merging) or
about 79.6 (low resolution, interpolated). Removing tokens early (block 4) is what hurts: halving at block 4
only costs 3.8 mAP, at block 10 only 0.2. Our first method looked +13 mAP better only because it was compared
with the worst schedule; against the best training-free option the gain was about +2.4.

### 9.3 The method (`sp_mixed`) vs baselines (asymmetric mAP, 3 seeds)
| Query | GFLOPs | 216-ep baseline + linear map | `sp_mixed` | frozen-gallery student |
| --- | --- | --- | --- | --- |
| full | 2.93 | 85.11 | 83.95 | 84.45 |
| 192x96 | 1.62 | 82.80 | 83.60 | 83.68 |
| merge r=11 | 1.41 | 79.70 | 82.38 | 81.50 |
| 160x80 | 1.12 | 78.96 | 83.01 | 82.78 |
| 128x64 | 0.72 | not measured | 81.50 | 80.48 |

How to read it:
- Below about 1.7 GFLOPs our method beats the equal-budget baseline by +0.8 to +4.1 mAP.
- At full cost it is 1.16 below: one set of weights serving all budgets costs some full-cost accuracy.
- The prior-work-style frozen-gallery student ties it on low resolution and has no full-cost loss, but needs
  two sets of weights. Our method is about 1 mAP better under merging and at 128x64 (suggestive, 3 seeds).
- Training-time budgets are only approximately equal (stated in the report).

### 9.4 Latency (measured, not estimated)
On the GPU with one image at a time, every setting costs about 6 ms: the GPU is waiting on overheads, not on
arithmetic. Gradual merging is even **2.3x slower** there (its matching step runs in every block). Savings
appear with large batches and on the CPU: 160x80 is 2.67x faster at GPU batch 256 and 1.69x on CPU batch 1;
128x64 3.9x and 2.0x. On Market-1501, resolution beats merging in both speed and accuracy.

### 9.5 ViT-B (one seed, indicative)
Same pattern, larger full-cost deficit (2.7 mAP), and worse accuracy per GFLOP than ViT-S. ViT-S stays.

## 10. Per-query budget selection (the cascade)

Full numbers: `REPORT_ITERATION1.md` Section 6.9 and figure `runs/adaptive.png`.

**Result in two sentences.** The confidence score works: the cheap query's top-1 gallery similarity picks the
queries that need escalation as well as an oracle that knows the true drift, and far better than random
(e.g. baseline from 160x80 at p = 0.3: 80.91 vs random 78.32, oracle 80.80). But the cascade still loses to
simply giving every query one intermediate resolution at the same average cost (our method: −0.1 to −0.7 mAP),
because an escalated query pays twice and the fixed-budget curve is concave (it rises steeply at first, then
flattens). So the cascade is reported as a preliminary negative result and becomes iteration-2 work, where the
cheap pass would be reused instead of paid twice.

The idea in brief:
- **Intuition:** most queries are easy; spend full compute only on the hard ones.
- **Cascade:** embed every query cheaply (e.g. 128x64) and search. Compute a confidence score from that search
  (the cheap query's highest cosine similarity to the gallery, `top-1`). Re-embed the fraction p of queries
  with the lowest confidence at full cost and search again.
- **Average cost:** `g_cheap + p · g_full` (the cheap pass is already paid for the escalated queries).
- **Controls:** random escalation (does the score carry information?) and an oracle that escalates the queries
  whose cheap embedding drifted most (needs the full embedding; upper bound only).
- **Prior work:** RANet (CVPR 2020, resolution cascade) and DVT (NeurIPS 2021, token-count cascade) do this
  for classification with softmax confidence. Using retrieval similarity as the confidence is a
  query-performance-prediction idea (iQPP benchmark, 2023). Ours is an adaptation, evaluation-only.

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
GPU-hours from the training log; peak VRAM from PyTorch. Baseline 0.22 h, 216-epoch baseline 0.36-0.38 h,
`sp_mixed` 0.35-0.44 h, frozen-gallery student 0.23 h on top of its 0.22 h teacher.

## 12. Negative results and limitations

Negative results (keep them in the report; they are part of the contribution):
1. Our first method (3-stage TBSD) did not beat the equal-GPU-hour baseline with a linear map.
2. No low-data regularisation effect at 10/25/50 % of identities.
3. Class-attention token selection (EViT rule) is worse than random selection for cross-budget matching. Cause
   unknown.
4. Token merging is dominated by lower resolution on Market-1501 (accuracy and latency).
5. The prior-work frozen-gallery student matches our method on low resolution.
6. Per-budget BatchNorm statistics are not the cause of the full-cost deficit.

Limitations:
1. One dataset, natively 128x64, so low resolution loses little real information here; results may not
   transfer to higher-resolution data. A second dataset (CUHK03-NP) is future work.
2. One backbone with 3 seeds (ViT-S); ViT-B one seed.
3. Full-cost deficit of about 1.2 mAP.
4. Narrow novelty: a combination of existing techniques (credited).
5. No re-implementation of SFSC, PrunNet or D3still; frozen-gallery baseline implements only the absolute
   term of Suma and Tolias.
6. Training budgets only approximately equal; GFLOPs analytic.

## 13. Related work in one page

| Paper | One-line summary | Relation to us |
| --- | --- | --- |
| SFSC (CVPR 2023) | One CNN with width-sliced sub-models whose features are mutually compatible; Market-1501, MSMT17 | Same goal, CNN width instead of input size |
| D3still (CVPR 2024) | Small query network distilled to rank like a large gallery network | Our alignment loss is its feature term |
| Suma and Tolias (WACV 2023) | Frozen gallery network; same-architecture student for small-resolution queries, label-free | Our frozen-gallery baseline |
| ResFormer (CVPR 2023) | Multi-resolution ViT training; detached high-resolution class token teaches low resolution | Mechanism of our low-resolution pass; no retrieval |
| PrunNet (arXiv 2025) | One network, pruned sub-networks compatible with the dense one, incl. ViT-S and re-ID | One-network multi-budget compatibility, but weights pruned, no measured speed-up |
| DVT (NeurIPS 2021) / RANet (CVPR 2020) | Cascades of increasing tokens / resolution with confidence-based early exit | Basis of our per-query cascade |
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
| `run_iter3.sh`, `run_iter4.sh` | Experiment queues (resumable, auto-commit) |

Key code points: the cheap pass is chosen in `train.py` (`--sd-mode mixed` picks `lowres` or `merge_r` per
step); the alignment term is `(1 - F.cosine_similarity(bn_s, bn.detach())).mean()`; low resolution in
`ReIDViT.backbone(..., hw=...)` resizes the image and the position embedding.

## 15. How to reproduce (Git Bash, from the project folder)
```bash
# our method, seed 0
.venv/Scripts/python.exe train.py --data C:/Users/user/data/Market-1501-v15.09.15 --out runs/sp_mixed_s0 --seed 0 --sd-mode mixed
# diagnostics (asymmetric mAP for every cheap setting)
.venv/Scripts/python.exe mechanism.py --data C:/Users/user/data/Market-1501-v15.09.15 --ckpts runs/sp_mixed_s0/last.pth --out runs/sp_mixed_s0/mech.csv
# frozen-gallery student on the baseline of the same seed
.venv/Scripts/python.exe train.py --data C:/Users/user/data/Market-1501-v15.09.15 --out runs/fg_mixed_s0 --seed 0 --teacher runs/base_s0/last.pth --sd-mode mixed --epochs 96
# cascade
.venv/Scripts/python.exe adaptive.py --data C:/Users/user/data/Market-1501-v15.09.15 --ckpts runs/sp_mixed_s0/last.pth --out runs/adaptive.csv
# latency (machine idle)
.venv/Scripts/python.exe bench_latency.py --extra --out runs/latency_extra.csv
```
Expected: about 0.4 GPU-hours per two-pass run on an RTX 5080; full-cost mAP around 84.

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
- **Equal-GPU-hour baseline:** a baseline given as much training time as the method.
- **GFLOPs:** billions of multiply-accumulates per image (analytic).
- **Gallery / query:** the searched collection / the image we search with.
- **Label smoothing:** soft target distribution for cross-entropy.
- **Linear (ridge) map:** a matrix fitted by least squares with a penalty, mapping cheap to full embeddings.
- **mAP, Rank-1:** Section 2.3.
- **Oracle:** an unrealistic control that uses information unavailable at test time, as an upper bound.
- **PK sampling:** P identities x K images per batch.
- **Seed:** the random-number starting point; different seeds = independent repetitions.
- **Stop-gradient (sg):** treat a value as a constant in back-propagation.
- **Token:** one vector per image patch (plus the class token).
- **Token merging (ToMe):** averaging similar tokens to reduce their number.
- **Triplet loss:** pulls same-person embeddings together, pushes others apart by a margin.
- **ViT-S/16:** Vision Transformer, small (width 384, 12 blocks, 6 heads), 16x16 patches.
