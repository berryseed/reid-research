# Likely supervisor questions, with answers (iteration 1)

Prepared 2026-11-17. Answers use our own measured numbers (Market-1501, ViT-S/16, mean ± std over 3 seeds
unless stated). Details: `REPORT_ITERATION1.md`; explanations from the ground up: `STUDY_GUIDE.md`.
Answer honestly; most of the strength of this iteration is in what it measured carefully, including what
did not work.

---

## A. Problem and motivation

**A1. What problem are you solving, in one sentence?**
How to make the *query* side of a ViT person re-ID system cheaper (smaller input or fewer tokens) while its
embedding stays directly comparable with a gallery that was computed once at full cost.

**A2. Why does that matter?**
In deployment the gallery is indexed once on a server, but queries arrive continuously and may be computed on
edge devices. Re-computing the gallery for every new model is expensive (re-indexing). So we want a cheap query
encoder that is *compatible* with the existing full-cost gallery. This fits the project thesis: accuracy per
unit of compute, labels and training time.

**A3. Why not just use a smaller network for the query?**
That is asymmetric retrieval with two networks (D3still, CVPR 2024; Suma and Tolias, WACV 2023). It works but
needs a second model trained to imitate the first. We study the knobs of a *single* ViT: input resolution and
token count, which can be changed per query at run time without loading another model. We also implemented the
two-network idea (frozen-gallery student) as a baseline and report that it ties our method on low resolution.

**A4. Why ViT and not a CNN?**
A ViT's cost is set by its number of tokens, and both knobs (resolution, merging) act on the token count without
changing the weights. CNNs have width or resolution knobs; width is what SFSC studies. ViT-S/16 also gives a
strong baseline (84.45 mAP) at 2.93 GFLOPs.

## B. Novelty and related work

**B1. What is new here?**
Honestly: the *combination and the evaluation*, not the components. Each component is prior work and we cite
it: two passes through shared weights with a shared classifier (SFSC's BCT-S baseline), the cosine alignment
loss (D3still's feature term, Suma and Tolias's absolute term), multi-resolution ViT training with a detached
high-resolution teacher (ResFormer), token merging (ToMe). What we did not find in the papers read is the
combination for *input-side compute knobs* of one ViT re-ID model, compatible with its own full-cost gallery
under two knobs at once, evaluated against an equal-GPU-hour baseline with measured latency. The iteration-1
rules accept a credited combination of existing techniques.

**B2. How is this different from SFSC and PrunNet?**
Both train one network whose cheaper sub-models stay compatible. SFSC (CVPR 2023) slices CNN channel width;
PrunNet (arXiv 2025) prunes weights, including a ViT-S, and also evaluates Market-1501. We make the *input*
cheaper. Practical difference: their sub-models need sparse or sliced kernels to run faster (PrunNet's authors
state their unstructured sparsity is hard to accelerate), whereas a smaller input runs faster on ordinary
hardware, which we measured (2.7x throughput at 160x80). We did not re-implement either, so we cannot compare
accuracy directly; that is a stated limitation.

**B3. Isn't low-resolution query matching just cross-resolution re-ID?**
The geometry is the same, but the goal differs. In cross-resolution re-ID (MLR-Market-1501, CAVIAR) low
resolution is a degradation of the captured image; methods super-resolve or upsample the query back to the full
network input (in the paper we read in full, RFD, training images are upsampled to 384x128), so no compute is
saved. We start from a full-resolution image and choose a smaller one to save compute.

**B4. Isn't your training just ResFormer?**
The low-resolution half of it is mechanically close: ResFormer trains a ViT on several resolutions with the
detached high-resolution class token teaching the low-resolution one. ResFormer is for classification and never
matches a low-resolution embedding against high-resolution embeddings of other images. We use the mechanism for
retrieval compatibility, add the merging knob and the shared re-ID losses, and evaluate asymmetric retrieval.

**B5. Which 2025-2026 work did you check?**
PrunNet (arXiv 2504.11879, April 2025) in full; QuARI (NeurIPS 2025, query-adaptive retrieval) and ACD (CVPR
2026 Findings, asymmetric distillation) and ThinkingViT (CVPR 2026, elastic ViT) at search or abstract level
only, and I say so. Abstract-level reading is never used to claim a paper lacks something.

**B6. How thorough was the literature search?**
About 40 searches; nine papers read in full (four of them through a summarising tool: SFSC, Suma and Tolias,
PrunNet, ResFormer, RFD; D3still and SUReID directly; ToMe and EViT through their code). The matrix with reading
depth for each entry is `LITERATURE_MATRIX.md`. "Not found" means not found in what was read.

## C. Method

**C1. Describe the method precisely.**
Each training step runs a batch twice through the same ViT: once at full cost and once at a cheap budget b
drawn at random (a low resolution in {224x112, 192x96, 160x80, 128x64} or gradual token merging with
r in {3, 5, 7, 9, 11}). Loss:
`L = CE(full) + Tri(full) + CE(b) + Tri(b) + λ · mean(1 − cos(e_b, sg(e_full)))`, λ = 1, sharing the classifier.
At test time the gallery is full cost; a query uses whatever budget the device affords.

**C2. Why the stop-gradient?**
The gallery is computed at full cost, so the full embedding is the reference. Without stop-gradient the
alignment term would also pull the full embedding towards the cheap one and degrade the gallery.

**C3. Why share the classifier?**
A shared classifier forces both passes to put each identity on the same side of the same decision boundaries.
Our ablation shows this does most of the work: multi-budget training *without* the alignment term already
reaches 80.56 at 3-stage keep 0.5, against 81.99 with it.

**C4. Why mix two knobs instead of one?**
Measured: a model trained on one knob is not compatible on the other (merge-trained: 73.0 mAP at 128x64;
resolution-trained: 79.6 at merge r = 11). The mixed model is compatible on both (81.5 and 82.4) at almost no
cost on either.

**C5. How did you choose λ, the budgets, the learning rate?**
λ = 1 and the baseline recipe (lr 0.008, 120 epochs, standard re-ID losses) were not tuned for the method, on
purpose: tuning would be extra budget the baselines did not get. The budget sets span roughly 25-75 % of full
cost. This is a limitation as well as a fairness choice.

## D. Experiments and fairness

**D1. Is the comparison fair?**
Main safeguards: (1) an **equal-GPU-hour baseline**, the ordinary model trained 216 epochs (0.36-0.38 GPU-h)
against our 0.35-0.44 GPU-h; (2) that baseline also gets a training-free linear map from cheap to full
embeddings; (3) 3 seeds everywhere; (4) the closest prior-work scheme (frozen-gallery student) implemented.
Not perfectly equal: the merge-only variant used about 15 % more GPU time and the frozen-gallery
student plus teacher about 20 % more; both are stated.

**D2. What are the headline numbers?**
Asymmetric mAP of our method (`sp_mixed`) vs the equal-GPU-hour baseline with linear map:
192x96 (1.62 GFLOPs) 83.60 vs 82.80; merge r = 11 (1.41) 82.38 vs 79.70; 160x80 (1.12) 83.01 vs 78.96;
128x64 (0.72) 81.50 (baseline 60.13 without map). Full cost: 83.95 vs 85.11.

**D3. So your method is worse at full cost?**
Yes, by 1.16 mAP against the equal-GPU-hour baseline and 0.5 against the 120-epoch baseline. One set of weights
serving all budgets costs some full-cost accuracy. The frozen-gallery student avoids this (84.45 at full cost)
but needs two sets of weights. It is a trade-off and we report it as such.

**D4. Why only Market-1501?**
Time and access: CUHK03 and MSMT17 require the author's own sign-up. It is the main limitation, especially
because Market-1501 crops are natively about 128x64, so a 128x64 query loses little real information. The
resolution results must be confirmed on higher-resolution data (future work).

**D5. Why are three seeds enough?**
It is the project standard and the minimum to estimate a spread. Differences we call results are at least
about 3 standard deviations (e.g. +4.1 mAP at 1.12 GFLOPs with std 0.15-0.18); differences of about 2 std
(e.g. 0.9 mAP between our method and the frozen-gallery student under merging) are called suggestive.

**D6. How did you check your token-reduction code is correct?**
`check_reduction.py` compares our merging and top-k selection with a transcription of the official ToMe and EViT
functions on random inputs (difference 0). The official code was read, not executed, because executing
downloaded code is blocked in this environment. Sanity checks: keep = 1.0 equals no reduction; full-cost GFLOPs
2.93 match the hand calculation.

**D7. Why ViT-S and not ViT-B?**
ViT-B (one seed) reached 86.38 at full cost but 11.34 GFLOPs, 3.9x ViT-S for +1.9 mAP; its cheapened version
(82.4 at 5.65 GFLOPs) is below a plain ViT-S at 2.93 GFLOPs (84.45). For accuracy per compute ViT-S is the
better operating point. The two-pass probe also lost more at full cost on ViT-B (2.7 mAP).

## E. Efficiency

**E1. Are GFLOPs savings real speed-ups?**
Not always; we measured. On the GPU at batch 1 every setting costs about 6 ms (fixed overhead), and gradual
merging is 2.3x *slower* (its matching step runs in every block). At GPU batch 256 and on the CPU the savings
appear: 160x80 is 2.67x faster at GPU batch 256, 1.69x at CPU batch 1, 2.35x at CPU batch 8.

**E2. Then why keep token merging at all?**
On Market-1501, resolution beats merging in accuracy and speed, and the report says so. Merging may matter where
downsizing destroys detail that merging keeps (higher-resolution data); that is untested. The mixed model costs
nothing extra to keep both options.

**E3. What is the training cost?**
Our method 0.35-0.44 GPU-hours per run on an RTX 5080 (two passes per step), peak VRAM about 4.9 GB; baseline
0.22 h; frozen-gallery student 0.23 h on top of its 0.22 h teacher, 1.8 GB VRAM.

## F. Negative results

**F1. What did not work?**
(1) Our first method, 3-stage TBSD, looked +13 mAP better but only against a weak schedule; against the
equal-GPU-hour baseline with a linear map it was not better. (2) No low-data regularisation effect at 10/25/50 %
of identities. (3) Class-attention token selection (EViT rule) is worse than random selection for cross-budget
matching. (4) Merging lost to resolution. (5) Per-query budget selection (cascade) lost to fixed budgets.

**F2. Why did the cascade not work?**
The confidence signal works: the cheap query's top-1 similarity to the gallery chooses which queries to
escalate as well as an oracle that knows the true drift, and clearly better than random. But an escalated query
pays the cheap pass *plus* the full pass, and the fixed-budget curve is concave: a single intermediate
resolution for all queries already keeps most of the accuracy at about half the cost. So every cascade point is
dominated by some fixed budget. Making it work needs reuse of the cheap pass (as DVT reuses features) or data
where low resolution fails more often. Proposed for iteration 2.

**F3. Why is class-attention selection worse than random?**
Unexplained. We ruled out our fusion implementation (exact EViT fusion gives the same result) and a systematic
embedding offset. Hypothesis, untested: attention concentrates on a few discriminative regions, so the kept
tokens cover the body unevenly and the cheap embedding moves further from the full one than with uniformly
spread random tokens. A spatial-coverage test is planned.

**F4. Why does two-pass training cost full-cost accuracy?**
Not established. Per-budget BatchNorm statistics are ruled out. A likely cause is gradient conflict between the
two passes (SFSC and PrunNet report and handle this); the gradient-cosine diagnostic is future work.

## G. Future work

**G1. What is the plan for iteration 2?**
(1) A second dataset with higher native resolution (CUHK03-NP); (2) per-query budget selection with feature
reuse, so escalation is not paid twice; (3) reduce the full-cost deficit (gradient-conflict handling as in
PrunNet, or a frozen-gallery student on the stronger 216-epoch teacher); (4) re-implement one prior-work
baseline (SFSC-style or PrunNet-style) for a direct accuracy comparison; (5) more ViT-B seeds.

**G2. If you had one more week, what single experiment would you run?**
The main table on CUHK03-NP. It decides whether the resolution results are a property of Market-1501's small
native crops, and whether token merging has any role.

## H. Reproducibility

**H1. Can someone reproduce this?**
Yes: the code, configs, logs (`log.json`, `stdout.txt`), per-run CSVs and aggregates are in the GitHub
repository; checkpoints are local. Exact commands are in `PROGRESS_LOG.md` and `STUDY_GUIDE.md` Section 15.
Every training run writes a resumable checkpoint each epoch. Hardware: one RTX 5080. Total training time of
all 57 runs of the project: 13.4 GPU-hours (sum of `train_hours` in the `log.json` files; evaluation and
latency runs not included).

**H2. Did you tune anything on the test set?**
No hyperparameters were tuned for the method. Model selection uses the last epoch. For the cascade, the whole
cost-accuracy curve is reported without choosing a threshold. The linear map is fitted on training images. Our
earlier choice of budgets and variants was guided by test results across iterations, which is a mild form of
test-set reuse we should acknowledge; a validation split is planned.
