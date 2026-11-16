# Literature comparison matrix

Started 2026-11-15 (session 2). Purpose: decide whether "cross-token-budget embedding compatibility in person
re-ID" is a real gap, and what the closest published methods already do.

**Reading depth codes.** `FULL` = whole paper text read by the assistant (PDF converted to text).
`FULL-S` = full text read through a summarising fetch tool with targeted questions (equations reported by the
tool, not checked line by line). `ABS` = arXiv abstract page only. `CODE` = official source code read.
`PK` = assistant's prior knowledge, not re-read this session.
**Rule:** "not stated" means not found in the text that was read. It does not mean the paper lacks the feature.
Entries at `ABS`/`PK` depth must not be used to claim that a paper lacks something.

"Ours" below means the exploratory method TBSD: one ViT, a full-token pass and a reduced-token pass per step
through shared weights, identity + triplet losses on both, plus `mean(1 - cos(bn_reduced, sg(bn_full)))`.

---

## A. Closest work, read in full

### A1. SFSC — Switchable Representation Learning Framework with Self-compatibility
- **Bibliographic:** Shengsen Wu, Yan Bai, Yihang Lou, Xiongkun Linghu, Jianzhong He, Ling-Yu Duan. CVPR 2023.
  arXiv 2206.08289 (v1 16 Jun 2022, v4 23 Mar 2023). No code link on the arXiv page. Depth: `FULL-S` + keyword
  search of the PDF text.
- **Objective:** one training run that yields several sub-models of different capacity whose features are
  mutually comparable (many-to-many compatibility), for deployment on platforms with different budgets.
- **Methodology:** sub-models are channel-width slices (e.g. 0.25x, 0.5x, 0.75x) of one CNN sharing weights;
  a separate BatchNorm per sub-model. Compatibility comes from all sub-models sharing one classifier, with an
  evidential (Dirichlet) classification loss per sub-model: `alpha = e + 1`, `S = sum(alpha)`,
  uncertainty `u = C / S`, loss `sum_c y_c (psi(S) - psi(alpha_c)) + lambda * KL(Dir(alpha) || Dir(1))`,
  `lambda = 0.2`. Gradient conflicts between sub-models are handled in magnitude (uncertainty re-weighting) and
  direction (project a gradient onto the normal plane of another when their cosine is negative, PCGrad-style).
- **Novel contribution (as claimed):** self-compatible switchable framework; conflict handling by magnitude
  and direction.
- **Token reduction mechanism:** none found. Backbones are ResNet-18/50, MobileNetV2, ShuffleNet. Tokens and
  vision transformers are not mentioned in the text (LeViT appears only as a citation).
- **Embedding compatibility:** yes, the central topic.
- **Asymmetric retrieval:** yes. Reports mAP/R1 for every (query sub-model, gallery sub-model) pair.
- **Costs:** parameter ratios only (about gamma^2). No FLOPs, latency or training time found.
- **Datasets:** Market-1501, MSMT17 (person re-ID), VeRi-776 (vehicle re-ID).
- **Limitations (ours, the paper has no limitations section in the text read):** CNN width only; no cost
  measurements; per-sub-model BN means the sub-models are not strictly one set of weights.
- **Its own baselines:** "BCT-S" and "Asymmetric-S" = BCT-style shared-classifier loss, or an embedding
  alignment loss, summed over sub-models. **These are structurally what our multi-budget control and TBSD do.**
- **Relative to ours:** same goal (one training run, several cost levels, compatible re-ID embeddings, same
  datasets). Different elastic dimension (channel width of a CNN vs token count of a ViT). SFSC has conflict
  handling that we lack; we have cost measurements, token-specific diagnostics and a single shared BNNeck.
  **Consequence: "one model, many budgets, compatible re-ID embeddings" is not an open problem in general.**

### A2. D3still — Decoupled Differential Distillation for Asymmetric Image Retrieval
- **Bibliographic:** Yi Xie, Yihong Lin, Wenjie Cai, Xuemiao Xu, Huaidong Zhang, Yong Du, Shengfeng He.
  CVPR 2024, pp. 17181-17190, DOI 10.1109/CVPR52733.2024.01626. Code: github.com/SCY-X/D3still. Depth: `FULL`.
- **Objective:** make a lightweight query network rank a gallery encoded by a large network in the same order
  as the large network would.
- **Methodology:** gallery network pretrained and fixed; query network trained by distillation only.
  Feature term `L_f = (1/n) * sqrt(sum_i (S^q_ii - 1)^2)` with `S^q_ij = cos(q_i, g_j)`: pull the query
  embedding of an image onto the gallery embedding of the same image. Ranking term on similarity differences
  over the top-k (k = 10) gallery neighbours: `((S^q_ij - S^q_il) - (S^g_ij - S^g_il)) / (m + |S^g_ij - S^g_il|)`,
  m = 0.1, split into rank-inconsistent and rank-consistent parts. Total `100 L_f + 0.2 L_irpd + 0.1 L_crpd`.
- **Novel contribution (as claimed):** distil differences of pairwise similarities (retrieval order), and
  decouple the feature term so that it dominates.
- **Token reduction mechanism:** none. Asymmetry is network (ResNet-101 or SwinV2-T gallery; ResNet-18 or
  MobileNetV2 query) plus resolution (256x256 gallery, 64x64 query).
- **Embedding compatibility / asymmetric retrieval:** yes / yes.
- **Costs:** inference FLOPs (12.99 G gallery vs 0.25 G query). Training cost not stated.
- **Datasets:** In-Shop, SOP, MSMT17 (person re-ID).
- **Reported numbers worth remembering:** MSMT17 mAP 59.81 (ResNet-101 both sides) vs 18.64 (D3still,
  ResNet-18 at 64x64 query). Pure relational distillation (RKD, PKT) collapses to near 0 in the asymmetric
  setting; CSD and ROP fall below plain feature distillation (FitNet) in their re-implementation.
- **Limitations (stated):** retrieval only; assumes a large capacity gap between the two networks.
- **Relative to ours:** our distillation term is `L_f` with shared weights and a mean instead of a root-sum.
  **Our loss is therefore not new.** D3still needs two networks and a pretrained teacher, and accepts a large
  accuracy loss for a 50x FLOPs cut; we stay within one network and a 2x cut. The ranking-difference term is
  a candidate improvement for us to test, not a contribution of ours.

### A3. SUReID — A Deep Hierarchical Feature Sparse Framework for Occluded Person Re-Identification
- **Bibliographic:** Yihu Song, Shuaishi Liu (arXiv listing; the PDF header shows Yihu Song). arXiv 2401.07469,
  15 Jan 2024, v1 only, no venue listed. No code link. Depth: `FULL`.
  (Earlier project notes called this paper "SUReID, 2024" without its real title.)
- **Objective:** occluded person re-ID without auxiliary pose/parsing models, with faster inference.
- **Methodology:** student DeiT-S (ViT for the small occluded sets) with DynamicViT-style learned token
  pruning: an MLP predicts keep/drop per token, Gumbel-softmax, attention masking during training, ratio loss
  `L_ratio` (MSE to target ratios). Pruning stages at layers 4, 7, 10 with ratios p, p^2, p^3. A class-token
  attention re-weighting module (training only). Distillation from a separate, frozen ViT teacher pretrained
  on the same dataset: `L_KD = 0.1 * KL(logits) + || f_s - I(f_t) ||_2^2`, I = interpolation to match
  dimensions. Total `alpha (L_KD + 2 L_ratio) + beta (L_cls + L_tri)`. Occlusion-patch augmentation (NODA).
- **Novel contribution (as claimed):** token sparsification against occlusion, interpolation-based feature KD,
  real-world occlusion augmentation.
- **Token reduction mechanism:** learned hard pruning (no merging, no fused token found).
- **Embedding compatibility:** not addressed as such; the feature KD aligns the student to the teacher's
  feature, but no cross-model or cross-budget retrieval is reported in the text.
- **Asymmetric retrieval:** not found in the text. One keep ratio per trained model (Table V lists p = 0.5-0.9
  as separately trained settings).
- **Costs:** throughput (img/s, RTX 2080 Ti, batch 32): 1074 (DeiT-S) to 1585 at p = 0.7, 2269 at p = 0.5.
  No FLOPs, no single-image latency, no training cost. All tokens are computed during training (masking), and
  the teacher runs too, so training is more expensive than the baseline.
- **Datasets:** Occluded-DukeMTMC, Occluded-REID, Partial-REID, Market-1501 (87.2 mAP / 94.5 R1),
  DukeMTMC-reID.
- **Limitations (stated):** fixed keep ratios for all images; depends on the teacher's quality.
- **Relative to ours:** same backbone family, same pruning positions and ratio schedule. Differs in: separate
  frozen teacher vs shared weights; one budget per model vs all budgets in one model; learned pruning vs
  merging; no cross-budget evaluation found.

### A4. CSD — Contextual Similarity Distillation for Asymmetric Image Retrieval
- **Bibliographic:** Hui Wu, Min Wang, Wengang Zhou, Houqiang Li, Qi Tian. CVPR 2022, pp. 9489-9498,
  DOI 10.1109/CVPR52688.2022.00927. No arXiv version found. Depth: PDF text keyword-searched plus D3still's
  description of it (not a line-by-line read).
- **Objective / method:** train a small query model, without labels, to keep each image's similarity
  distribution to its neighbours as computed by a frozen large gallery model.
- **Token reduction:** none found. Networks: ResNet-101 gallery; MobileNetV2, EfficientNet-B3 query.
- **Compatibility / asymmetric:** yes / yes. **Costs:** GFLOPs and parameters of each network.
- **Datasets:** Revisited Oxford and Paris, GLDv2, SfM120k (landmarks). No re-ID dataset found in the text.
- **Relative to ours:** label-free neighbour-structure distillation is a possible ingredient; in D3still's
  re-implementation it underperformed plain feature distillation.

### A5. Suma and Tolias — Large-to-small Image Resolution Asymmetry in Deep Metric Learning (added 2026-11-16)
- **Bibliographic:** Pavel Suma, Giorgos Tolias (CTU Prague). WACV 2023 (venue from the earlier abstract-level
  entry; not shown on the HTML page read). arXiv 2210.05463. Code: github.com/pavelsuma/raml.
  Depth: `FULL-S` (ar5iv HTML through a summarising fetch tool; equations as reported by the tool).
- **Objective:** cheap query extraction by using a small query resolution against a database indexed at a
  large resolution.
- **Methodology:** teacher = the database network, trained with labels (triplet loss), frozen. Student = same
  architecture, **initialised from the teacher's weights**, trained **without labels** on low-resolution
  inputs. Loss `L = L_abs + lambda_t L_rel-ts + lambda_s L_rel-ss`, with
  `l_abs = (1 - cos(S(x_low), T(x_high)))^2`, `l_rel-ts = (T(y)^T T(z) - T(y)^T S(z))^2`,
  `l_rel-ss = (T(y)^T T(z) - S(y)^T S(z))^2` over pairs of augmentations y, z of one image. "Coupled
  augmentations": the same random augmentation for the teacher and student inputs (separate ones hurt).
  8 augmentations per image; random resized crop, colour distortion, flip, mixup.
- **Token reduction mechanism:** none. ResNet-50 / ResNet-18 + GeM. No vision transformer or tokens in the text.
- **Embedding compatibility / asymmetric retrieval:** yes / yes (student queries vs teacher database at 448).
- **One model for several budgets?** No: Table 2 trains a different student per query resolution (317/224/158).
  A single student for several resolutions is not stated.
- **Costs:** query GFLOPs in figures (values not in the text). No latency. Training time only qualitatively
  (grows about linearly with the number of augmentations).
- **Datasets:** CUB200, Cars196, SOP. No person re-ID dataset.
- **Numbers (mAP / R@1, query 224 vs database 448):** untrained teacher asymmetric 34.3/61.0 (CUB), 25.7/61.0
  (Cars), 51.3/74.3 (SOP); trained student asymmetric 40.8/70.5, 38.1/83.4, 59.7/80.8; teacher symmetric at
  448: 42.7/73.7, 41.6/87.9, 62.6/82.1.
- **Future work named by the authors:** combine network and resolution asymmetry; drop the fixed-database
  assumption and optimise both networks jointly.
- **Relative to ours:** (1) **Candidate C1 (frozen gallery, same-network student at a reduced budget) is this
  method** with a different knob; "student starts aligned because it is the same network" is their idea, not
  ours. (2) Their untrained-teacher control is our low-resolution-query baseline. (3) Not found in the text:
  one set of weights serving several budgets; token count as the knob; more than one knob; re-ID; latency or
  training-cost accounting. Our shared-weight two-pass training is one instance of the joint optimisation they
  leave as future work, and it pays for it with a full-token deficit that their frozen teacher avoids.

---

## B. Token reduction in vision transformers

| Work (depth) | Verified metadata | Mechanism | Trained or training-free | Compatibility / asymmetric retrieval | Relation to ours |
| --- | --- | --- | --- | --- | --- |
| ToMe (`CODE` + `ABS`) | "Token Merging: Your ViT But Faster", Bolya, Fu, Dai, Zhang, Feichtenhofer, Hoffman. ICLR 2023 (oral). arXiv 2210.09461 | Bipartite soft matching on head-averaged attention keys; a constant `r` tokens merged in **every** block; size-weighted average; proportional attention | Both (off-the-shelf and trained) | Classification, video, audio. Retrieval compatibility not stated in abstract | Our `merge` uses the same matching (verified numerically against a transcription of the official function, difference 0) but a **different schedule**: 3 large reductions instead of 12 small ones |
| EViT (`CODE` + `ABS`) | "Not All Patches are What You Need: Expediting Vision Transformers via Token Reorganizations", Liang, Ge, Tong, Song, Wang, Xie. ICLR 2022 (spotlight). arXiv 2202.07800 | Keep top-k by class-token attention at layers 4/7/10; fuse the rest into one token = attention-weighted **sum** | Trained with token reduction | Classification | Our `cls_attn` selection is identical. Our fused token is a weighted **average** (same direction, larger norm): not exactly EViT |
| SiT (`FULL-S`) | "Self-slimmed Vision Transformer", Zong, Li, Song, Wang, Qiao, Leng, Liu. ECCV 2022. arXiv 2111.12624 | Learned soft aggregation (token slimming module), 3 stages | Trained; un-slimmed ViT as teacher; token-level MSE through a reverse module + logit KL | ImageNet classification only; retrieval not stated; ratios appear to be separate runs | Self-distillation from full to slimmed model, but for logits/tokens, per ratio |
| ToCom (`ABS` via search) | "Token Compensator: Altering Inference Cost of Vision Transformer without Re-Tuning", Jie, Tang, Guo, Deng, Han, Wang. ECCV 2024. arXiv 2408.06798. Code github.com/JieShibo/ToCom | Plug-in module compensating a mismatch between training and inference compression degree | Self-distilled plug-in | Classification (CIFAR-100, FGVC, VTAB-1k) | Addresses train/test budget mismatch, the classification analogue of our drift |
| Haurum et al. (`ABS`) | "Which Tokens to Use? Investigating Token Reduction in Vision Transformers", Haurum, Escalera, Taylor, Moeslund. ICCV 2023 NIVT Workshop. arXiv 2308.04657 | Comparison of 10 reduction methods | n/a | Classification | Reports top-k pruning as a strong baseline. The abstract does not report random selection |
| AMG (`ABS`, iteration 1) | arXiv 2304.01452, "Attention Map Guided Transformer Pruning for Edge Device" | Attention-guided token and head pruning | Trained | Re-ID (Occluded-Duke, Market-1501) | Not re-read this session |
| FPC (`ABS`) | "Dynamic Feature Pruning and Consolidation for Occluded Person Re-Identification", Ye, Zhou, Cai, Gao, Zhang, Wang, Hu, Yu, Yang. AAAI 2024. arXiv 2211.14742 | Class-attention token dropping, then consolidation from gallery neighbours | Trained | Re-ID, occluded; compatibility not stated in abstract | Uses pruning against occlusion, not for a cost/compatibility trade-off |
| SRE-Merge, TE-VMamba (`ABS`, iteration 1) | ICML 2026; CVPR 2026 (Huang et al.) | Training-free merging; pruning + merging in VMamba | — | Re-ID, in-domain (abstract level) | Not re-read this session |

**Removed:** "MergeTok" (arXiv 2605.30904) was recorded earlier as training with random merge ratios. Its
abstract shows it is a visual tokenizer for image generation. It is not relevant and the earlier note was wrong.

## C. Compatible and asymmetric retrieval (not token-based)

| Work (depth) | Verified metadata | What is asymmetric | Mechanism | Datasets |
| --- | --- | --- | --- | --- |
| BCT (`ABS` + `PK`) | "Towards Backward-Compatible Representation Learning", Shen, Xiong, Xia, Soatto. CVPR 2020 (oral). arXiv 2003.11942 | New model vs old model | `PK`: the old model's classifier is applied to the new embeddings as an extra loss | Face recognition |
| Resolution asymmetry (`ABS`) | "Large-to-small Image Resolution Asymmetry in Deep Metric Learning", Suma, Tolias. WACV 2023. arXiv 2210.05463 | Query at small resolution, database at large resolution | Label-free per-image distillation from a fixed teacher with coupled augmentations | CUB200, Cars196, SOP |
| AML, HVS/CMC, ROP, AFF (`PK` / cited in D3still) | Budnik and Avrithis CVPR 2021; Duggal et al. CVPR 2021; Wu et al. ICLR 2023; Wu et al. CVPR 2023 | Small query network, large gallery network | Regression/contrastive, classifier sharing, rank preserving, feature fusion | Mostly landmarks |
| Asymmetric VPR (`ABS`) | "Towards Test-time Efficient Visual Place Recognition via Asymmetric Query Processing", Kim, Cho, Yoon. AAAI 2026. arXiv 2512.13055 | Lightweight query network, large gallery model | Geographical memory bank + implicit embedding augmentation | Place recognition |
| ACD (`search result only`) | "Asymmetric Collaborative Distillation for Asymmetric Image Retrieval", Xie, Zhang, Luo, Zhou, He. CVPR 2026 Findings | Small query, large gallery (ResNet-101 at 256x256 in the supplement) | Not read | Not read |
| MatMamba (`FULL-S`, retrieval section) | "MatMamba: A Matryoshka State Space Model", Shukla, Vemprala, Kusupati, Kapoor. arXiv 2410.06718 (9 Oct 2024), no venue listed | Gallery from the largest model, queries from smaller nested **width** sub-models of the same network (depth fixed; tokens not mentioned) | Matryoshka (nested) training | ImageNet-1K 1-NN accuracy; reports 55% less compute for < 0.5% loss; independently trained small models are not compatible |
| ThinkingViT (`ABS`) | "ThinkingViT: Matryoshka Thinking Vision Transformer for Elastic Inference", Hojjat, Haberer, Pirk, Landsiedel. CVPR 2026. arXiv 2507.10800 | Nested attention heads + early exit by confidence | Progressive stages with token recycling | ImageNet-1K classification, segmentation; no retrieval mentioned |
| Wang et al. (`search summary`) | "When Token Pruning is Worse than Random: Understanding Visual Token Information in VLLMs", Yahong Wang et al. CVPR 2026. arXiv 2512.07580 | n/a (vision-language models) | Finds training-free pruning no better than random in deep layers ("vanishing token information") | VLLM benchmarks. **Analogy only** for our class-attention result; earlier notes cited it too loosely |

**Important for ViTs:** a lower input resolution also means fewer tokens. Resolution asymmetry is therefore
the nearest existing form of "cheap query, expensive gallery" for a ViT, and must be a baseline for any
token-budget method.

---

## D. Gap reassessment (2026-11-15, provisional)

Established by the reading above:
1. Many-budget compatible re-ID embeddings from one training run exist (SFSC), on Market-1501 and MSMT17.
2. Asymmetric re-ID retrieval with a cheap query encoder exists (SFSC, D3still on MSMT17).
3. The embedding-alignment loss we use is the standard asymmetric-retrieval feature loss.
4. Token reduction + distillation for re-ID exists (SUReID), with a separate teacher and one budget per model.

Not found in ~25 searches and the texts read (absence of evidence, not proof):
- token count of a single ViT as the asymmetric/elastic dimension for retrieval embeddings;
- any measurement of how token reduction moves a re-ID embedding relative to the full-token one;
- cost accounting (training GPU-hours, latency regimes) in the compatible-retrieval papers read.

Honest reading: the remaining gap is **narrow**. It is "the SFSC/asymmetric-retrieval problem, with token
count instead of width/network/resolution as the knob". A paper on that alone would be judged incremental
unless it shows (a) token count is a *better* knob than width or resolution at equal cost, or (b) something
about the mechanism that the width/resolution literature does not have.

Still to read in full before the gap statement is final: ACD (CVPR 2026 Findings), ThinkingViT, MatMamba
(retrieval section), ToCom, AMG, SRE-Merge, TE-VMamba, BCT, AML. (Suma and Tolias: done 2026-11-16, entry A5.)

## E. Gap reassessment after session 3 (2026-11-16)

Papers read in full today: one (Suma and Tolias, A5, depth `FULL-S`). No other entry changed depth.

What changed:
1. **Candidate C1 is prior work.** Frozen gallery model + same-architecture student initialised from it +
   label-free alignment on a cheaper input is the Suma and Tolias method. We now run it as a baseline
   (`train.py --teacher`, runs `fg_mixed_s*`; absolute term only, so "in the style of", not a re-implementation:
   their two relational terms and their 8-augmentation scheme are not implemented).
2. **Our own measurements narrow the gap further.** On Market-1501 the resolution knob (which A5 studies) beats
   the token-merging knob in accuracy and in measured latency (`runs/latency_extra.csv`). "Token count as the
   cheap-query knob" is therefore not a defensible headline on this dataset.
3. **What is still not found in the texts read** (absence of evidence only): a single set of weights that stays
   compatible with its own full-cost gallery under several budgets and under more than one knob (resolution and
   token merging); person re-ID in the resolution-asymmetry setting; a cost account of compatibility (training
   GPU-hours against an equal-budget baseline, full-cost accuracy lost, measured latency per regime).
4. A5 trains one student per resolution; our frozen-gallery student and `sp_mixed` each cover four resolutions
   and five merge rates with one set of weights. Whether that is "new" depends on the unread papers above
   (ACD, ToCom, MatMamba, ThinkingViT in particular) and must not be claimed before they are read.

Open literature questions for the next session:
- Does any asymmetric-retrieval paper train one query model for several budgets (ACD, AML, ROP)?
- Is there a re-ID paper on resolution-asymmetric or low-resolution-query retrieval with a compatible gallery
  (cross-resolution re-ID is a known topic; **not searched yet**, and it could close most of the remaining gap)?
