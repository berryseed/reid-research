# RESEARCH_LOG.md — explained record of the project

Purpose: everything needed to reproduce and defend this research, written to be read from the top by someone
new to the area. `PROGRESS.md` is the short status, `PROGRESS_LOG.md` the terse chronological record (exact
commands, full tables). This file explains the *why*: concepts, maths, the reasoning behind each experiment,
what was expected, what happened, and what it means. Newest entries are at the bottom.

All numbers are from runs on this machine (RTX 5080) unless marked as an estimate.

---

## Part 1. The concepts, from the ground up

### 1.1 What person re-identification is
A camera network sees the same person from different cameras. Re-identification (re-ID) is a *retrieval*
task: given one cropped image of a person (the **query**), rank a large set of other crops (the **gallery**) so
that crops of the same person from *other cameras* come first.

The network is not a classifier at test time, because test identities were never seen in training. Instead it
maps each image to a vector (an **embedding**, 384 numbers for our model). Two images are compared by the
cosine similarity of their embeddings: `cos(a, b) = a·b / (|a||b|)`. Training has to make embeddings of the
same person close and of different people far apart.

### 1.2 Dataset: Market-1501
- 1,501 identities, 6 cameras. Train: 12,936 images of 751 identities. Test: 3,368 queries of 750 other
  identities, and 19,732 gallery images (15,913 after dropping "junk" detections; "distractor" images stay).
- Protocol: for each query, gallery images of the same person **from the same camera** are ignored (finding the
  same frame sequence is trivial and not what re-ID is about).

### 1.3 Metrics
- **Rank-1 (R1):** percentage of queries whose top-ranked gallery image is the right person.
- **mAP (mean average precision):** for one query, walk down the ranked list; each time a correct image
  appears at position `p` as the `i`-th correct one, record the precision `i / p`; the average of these is the
  AP of that query. mAP is the mean over all queries. Rank-1 only cares about the first hit; mAP rewards
  ranking *all* images of the person early, so it is the stricter and more informative number. We report mAP
  as the main metric.
- **Cost metrics** (the thesis of this project is accuracy *per unit of cost*): GFLOPs (billions of
  multiply-adds for one image), measured latency, training GPU-hours, peak GPU memory, fraction of labelled
  identities used.

### 1.4 The model: a Vision Transformer (ViT-S/16)
- The 256x128 image is cut into 16x16-pixel patches: 16 rows x 8 columns = **128 patch tokens**. Each patch is
  linearly projected to a 384-dimensional vector. A learned **class token** (CLS) is prepended: 129 tokens.
- 12 transformer blocks follow. Each block does (a) **self-attention**, where every token looks at every
  other token, and (b) an **MLP** applied to each token separately.
- The output CLS vector is the image feature `f`.
- Cost. With `N` tokens of width `D` (384) and MLP hidden width `4D`, one block costs about
  `4·N·D²  +  2·N²·D  +  2·N·D·4D` multiply-adds (query/key/value/output projections; the two attention matrix
  products; the MLP). Summed over 12 blocks plus the patch projection this gives **2.93 GFLOPs** at 129 tokens.
  This is the formula in `common.py: vit_gflops`. It is an analytic estimate, not a profiler measurement.
  The key point: cost falls at least linearly with `N`. Fewer tokens = cheaper network.
- Weights start from an ImageNet-pretrained checkpoint (timm `vit_small_patch16_224`, augreg, ImageNet-21k then
  1k). No person-specific pretraining, no external teacher: that is the "no extra data" constraint.

### 1.5 The re-ID head and the losses (standard "bag of tricks" recipe)
- **BNNeck:** `f` goes through a BatchNorm layer giving `b = BN(f)`. A bias-free linear classifier over the 751
  training identities sits on `b`. At test time `b` (L2-normalised) is the embedding. The reason for the neck:
  the two losses below want different geometry (triplet wants free Euclidean distances, cross-entropy wants
  features on a normalised hypersphere); putting BN between them lets each loss act on its preferred space.
- **Identity loss** (on the classifier output): cross-entropy with label smoothing 0.1,
  `L_id = -Σ_c q_c log p_c`, where `q` puts 0.9 on the true identity and spreads 0.1 over all classes. Smoothing
  stops the network becoming over-confident on 751 identities it will never see again.
- **Batch-hard triplet loss** (on `f`): for each image `a` in the batch take its *farthest* same-identity image
  `p` and its *closest* different-identity image `n`: `L_tri = mean( max(0, d(a,p) - d(a,n) + 0.3) )`, `d` =
  Euclidean distance, margin 0.3. It directly trains the ranking the test needs.
- **PK sampling:** each batch has P=16 identities x K=4 images = 64 images, so every image has positives and
  negatives in its batch (required by the triplet loss).

### 1.6 Token reduction
Many patches are redundant (background, repeated clothing texture). Token reduction removes tokens part-way
through the network so later blocks are cheaper. We reduce at three places, inside blocks 4, 7 and 10, after
attention and before the MLP. `keep` is the fraction of patch tokens kept at each of the three points:

| keep | patch tokens after blocks 4 / 7 / 10 | GFLOPs | saving |
| --- | --- | --- | --- |
| 1.0 | 128 / 128 / 128 | 2.93 | 0% |
| 0.8 | 103 / 83 / 67 | 2.22 | 24% |
| 0.7 | 90 / 63 / 45 | 1.92 | 34% |
| 0.6 | 77 / 47 / 29 | 1.68 | 43% |
| 0.5 | 64 / 32 / 16 | 1.46 | 50% |

Rules implemented in `common.py` (all our own implementations):
- **merge** (ToMe-style bipartite soft matching). Split the patch tokens alternately into sets A and B. For
  every A token find its most similar B token (cosine similarity of the attention *key* vectors, which
  summarise what the token is "about"). Take the `r` most similar pairs and average each pair into one token,
  weighted by how many original patches each already represents (its `size`). Information is blended, not
  thrown away. Afterwards attention adds `log(size)` to its logits ("proportional attention") so a token
  standing for 4 patches counts 4 times.
- **cls_attn** (EViT-style): keep the patches the class token attends to most. `cls_attn_fuse` additionally
  appends one token that is the attention-weighted average of the dropped ones (EViT's "fused token").
- **random**: keep a random subset. A control: any clever rule must beat it.
- **cls_attn_low**: keep the *least* attended patches. A diagnostic only.

### 1.7 Symmetric vs asymmetric retrieval, and "drift"
- **Symmetric:** query and gallery both encoded with reduced tokens.
- **Asymmetric:** the gallery is encoded **once, with all tokens** (done offline, cost amortised), and only the
  query, which must be encoded live and possibly on a weak device, uses reduced tokens.
- For asymmetric retrieval to work, the reduced-token embedding of an image must land near its full-token
  embedding. We measure this as **q_cos**: the mean cosine similarity between each query's reduced and
  full-token embeddings. 1.0 = identical. Low q_cos = the embedding has *drifted* into a different part of
  the space, and comparisons against a full-token gallery degrade.

### 1.8 The proposed method: token-budget self-distillation (TBSD)
One network trained so that its embedding at *any* token budget stays in the same space as its full-token
embedding. Every training step runs the same batch twice through the same weights:
1. full tokens → `f, b, logits`; losses `L_id + L_tri`;
2. merged tokens with `keep` drawn at random from {0.5, 0.6, 0.7, 0.8, 0.9} → `f', b', logits'`; losses
   `L_id' + L_tri'` plus a distillation term.

Total loss: `L = L_id + L_tri + L_id' + L_tri' + λ · mean(1 − cos(b', stopgrad(b)))`, `λ = 1`.
- "Self-distillation": the teacher is the network's own full-token pass; there is no second model.
- `stopgrad` (`.detach()` in code): gradient flows only into the reduced pass, so the cheap embedding is pulled
  to the expensive one and not the other way round.
- The distillation is on the BNNeck embedding `b`, not on class logits, because `b` is what retrieval uses and
  because prior work (PAT, ICCV 2023) reports logit-level self-distillation hurting re-ID generalisation.
- Cost: two forward/backward passes per step, about 1.85x the training time and 3.1 GB instead of 1.8 GB VRAM.
  Inference cost is unchanged: one network, any budget.

### 1.9 Training recipe (identical for every run unless stated)
ViT-S/16, input 256x128, batch 64 (P16xK4), 202 iterations per epoch, **120 epochs = 24,240 iterations**.
SGD, momentum 0.9, peak learning rate 0.008, weight decay 1e-4. Learning rate rises linearly from 1% over the
first 5 epochs (warm-up, protects the pretrained weights from large early gradients), then follows a cosine
curve down to 0.2%. Augmentation: horizontal flip, pad 10 + random crop, random erasing (p = 0.5). Mixed
precision (16-bit floats where safe) for speed and memory. Evaluation every 20 epochs with full tokens.
Seeds: every configuration that carries a claim is run with seeds 0, 1, 2; we report mean ± standard
deviation, and a difference smaller than the seed spread is not treated as a result.

---

## Part 2. Experiments so far, in order: question → expectation → result → what we learned

mAP in %. "full gal." = asymmetric (full-token gallery); "red. gal." = symmetric.

### E1. Baseline and its token-reduction sweep (3 seeds)
- **Question:** how good is the plain model, and how fast does accuracy fall when tokens are removed from a
  model that was never trained for it?
- **Expectation:** mAP in the 80s; graceful decline with merging.
- **Result:** 84.45 ± 0.49 at full tokens. Merge keep 0.8: 83.70; 0.7: 82.50; 0.6: 79.99; **0.5: 68.56 ± 1.14**
  (full gal.), 59.60 (red. gal.). q_cos at 0.5 = 0.733.
- **Learned:** (a) the kit is sound (sanity range met). (b) The original hypothesis H1 ("asymmetric retrieval is
  the hard case") was wrong as stated: the asymmetric number is *higher* than the symmetric one at every budget,
  because a full-token gallery is simply a better gallery. The real problem is drift of the query embedding
  (q_cos falls to 0.73). H1 was reframed accordingly. (c) Below keep 0.6 an untrained-for-it model collapses.

### E2. TBSD (3 seeds)
- **Question:** does TBSD remove the drift without hurting full-token accuracy?
- **Result:** full tokens 83.95 ± 0.13 (baseline 84.45); keep 0.5: **81.98 ± 0.18** full gal. (+13.4),
  80.26 ± 0.13 red. gal. (+20.7); q_cos 0.944. Training 0.41 GPU-h vs 0.22.
- **Learned:** the robustness gain is far outside seed noise. It costs about 0.5 mAP at full tokens (lower in
  all three seeds) and 1.85x training time. One set of weights now serves every budget from 2.93 down to
  1.46 GFLOPs with at most a 2-point loss.

### E3. Equal-GPU-hour baseline (216 epochs; seed 0 done, seeds 1–2 in phase B)
- **Why:** TBSD uses 1.85x the training compute. A fair comparison gives the baseline the same compute
  (216 epochs ≈ 0.36–0.40 GPU-h). Otherwise any gain could be "just trained longer".
- **Result (seed 0):** 85.28 at full tokens (+1.3 over TBSD), but 69.02 at keep 0.5.
- **Learned:** longer training buys full-token accuracy, not budget robustness. Honest framing: at matched
  compute TBSD gives up ~1.3 mAP at full tokens and gains ~13 at half the compute.

### E4. Is it the distillation, or just training at several budgets? (3 seeds, `--sd-weight 0`)
- **Why:** an ablation removes one ingredient to measure its contribution. Here `λ = 0`: the reduced pass still
  has its own identity and triplet losses, but nothing explicitly ties `b'` to `b`.
- **Result:** keep 0.5: 80.56 ± 0.06 (full gal.) vs TBSD 81.98 ± 0.18; red. gal. 78.31 vs 80.26; q_cos 0.902 vs
  0.944. Full tokens 83.71 ± 0.20 vs 83.95 ± 0.13.
- **Learned:** multi-budget training alone delivers +12.0 of the +13.4. The distillation term adds a real but
  modest +1.4 / +1.9 at the lowest budget (many standard deviations) at no extra cost. **The paper cannot claim
  distillation as the main source of the gain.** The defensible claim is: training one ViT across token
  budgets yields mutually compatible retrieval embeddings, and embedding self-distillation tightens it further.

### E5. Why is TBSD ~0.5 lower at full tokens? BNNeck hypothesis (3 seeds, `--sd-bn nostat`)
- **Background:** BatchNorm normalises with the statistics of the current batch during training and keeps a
  running average of them for test time. In TBSD both passes go through the same BN, so its running average
  mixes full-token and reduced-token features, while the test uses full tokens only. A mismatch between
  training and test statistics is a classic source of small accuracy losses.
- **Test:** during the reduced pass set the BN momentum to 0, so the running average is built from the
  full-token pass only (`train.py`, `--sd-bn nostat`). Verified on synthetic data that the running mean is then
  bit-identical before and after the reduced pass.
- **Result:** 83.83 ± 0.20 vs 83.95 ± 0.13 at full tokens; identical within noise at every budget.
- **Learned (negative result):** stat mixing is not the cause. Since the no-distillation variant shows the same
  deficit (83.71), the cost comes from the reduced pass's own identity/triplet gradients competing for the same
  weights: a capacity trade-off, not a bug. Negative results are kept, not hidden.

### E6. Which tokens matter? Selection diagnostics (3 seeds, evaluation only)
- **Why:** on seed 0, keeping the most-attended tokens (the standard EViT idea) was *worse* than keeping random
  tokens. That is surprising enough to be either a bug or a finding, so it had to be tested.
- **Three checks:** (1) 3 seeds instead of 1; (2) add EViT's fused token, which our first version lacked;
  (3) invert the rule: if attention were meaningless, keeping the least-attended tokens would do equally well.
- **Result (baseline checkpoints, keep 0.7, full gal.):** random 77.99 ± 0.61; cls_attn 72.66 ± 3.29;
  cls_attn_fuse 73.82 ± 2.99; cls_attn_low 37.62 ± 6.26. With a reduced gallery, cls_attn and random are within
  noise at keep 0.8 and 0.7.
- **Learned:** not a bug (inverse is far worse, so attention is informative); the fused token explains only
  ~1 mAP; attention-based selection still loses to random for a full-token gallery and has 3–8x the seed
  variance. A plausible reading: attention keeps a *biased* subset (a few salient parts), which changes the
  embedding more than an unbiased sample of the whole body, and drift is what hurts against a full-token
  gallery. This is our own implementation, not the official EViT code, and the mechanism is unproven.

### E7. Does the GFLOPs saving turn into real speed? (`bench_latency.py`)
- **Why:** GFLOPs are a paper quantity; the thesis is about real cost.
- **Result (merge keep 0.5 vs full, ms per image):** GPU batch 256: 0.062 vs 0.111 (1.78x faster). CPU batch 8:
  6.2 vs 11.1 (1.78x). CPU batch 1: 11.9 vs 16.3 (1.37x). **GPU batch 1: 8.5 vs 6.0 (slower).**
- **Learned:** a ViT-S on this GPU at batch 1 is limited by fixed per-operation overhead, not arithmetic, and
  the matching step of merging adds operations. Token reduction pays off for throughput (large batches) and on
  CPU, which is the low-resource deployment case. The paper must report this and not imply GFLOPs = speed.

### E8. Low data, 25% of identities (seed 0 only so far; 3 seeds at 10/25/50% in phase C)
- **Question (H3):** does TBSD act as a regulariser when labels are scarce?
- **Result:** full tokens 71.6 (baseline) vs 70.0 (TBSD); keep 0.5: 51.2 vs 66.4.
- **Learned so far:** no sign of a regularisation benefit; budget robustness survives with few labels.

### Novelty position after two rounds of search (abstract-level, about 20 searches)
Closest prior work: **SFSC (CVPR 2023)** trains one CNN whose sub-networks of different *width* give mutually
compatible features; asymmetric-retrieval papers (CSD 2022, D3still 2024) distil a small query network toward a
large gallery network (two models); SUReID, AMG, SRE-Merge, TE-VMamba reduce tokens in re-ID at one budget with
query and gallery treated the same; SiT/ToCom train multi-budget ViTs for *classification*. No paper was found
that trains one ViT across *token* budgets for cross-budget-compatible *retrieval* embeddings and evaluates
asymmetric re-ID. Novelty is rated moderate: a new instance of a known idea (self-compatible budgets), with new
evidence (drift analysis, cost accounting, ablations).

---

## Part 3. Session notes (chronological, explained)

### 2026-11-15, power outage
At about 12:33 the power failed during `tbsd_bnns_s0`. `train.py` writes a checkpoint after every epoch by
saving to `ckpt.pth.tmp` and then renaming it over `ckpt.pth`. A rename is atomic: the file is either the old
complete checkpoint or the new complete one, never half-written. That is why the checkpoint at epoch 94 was
valid. It stores the weights, the optimizer's momentum buffers (SGD's running average of past gradients;
without them the first steps after resume would behave differently), the mixed-precision loss scale, the
iteration counter (which determines the learning rate on the schedule) and all random-number-generator states
(so batches and augmentations continue the same sequence). Resuming re-ran only epochs 95–120. Loss: under
12 seconds of training. The queue script now also commits and pushes each run's metrics as soon as it finishes.

### 2026-11-15, phase B (in progress)
See the entries that follow.
