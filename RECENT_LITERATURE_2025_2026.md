# Recent literature 2025-2026: positioning of iteration 1 and directions for iteration 2

**Written 2026-11-21 (session 5).** Literature only: no training, no new experiments, nothing committed.
Scope: papers published in 2025-2026, 2026 preferred. Our own numbers come from `REPORT_ITERATION1.md`
(Market-1501, ViT-S/16, 3 seeds) and are not re-measured here.

## 0. How this review was done, and how deep each paper was read

- **Searches** (WebSearch, standard mode, 12 queries) covered:
  - asymmetric retrieval with distillation, 2026;
  - ThinkingViT;
  - token pruning and merging for re-ID;
  - backward-compatible learning, 2025-2026;
  - cross-resolution re-ID;
  - ACD;
  - elastic and Matryoshka retrieval;
  - knowledge distillation for lightweight re-ID;
  - compatible lightweight query models;
  - dynamic or early-exit re-ID;
  - arXiv copies of ACD, SRE-Merge and TE-VMamba;
  - the venue of PrunNet.
- **Full reads** went through a fetching tool that summarises pages (`FULL-S`: whole text read; figures not
  visible; equations as reported by the tool). Some papers were only readable in a browser pane at the
  abstract level (`ABS`).
- **Access limits.**
  - CVF Open Access PDFs and HTML returned HTTP 403 to the fetching tool. In the browser the abstract pages
    opened, but the PDFs were downloads, not pages, so they were not read.
  - OpenReview showed a browser-verification (bot-check) page, which was not bypassed.
  - **ACD, TE-VMamba and SRE-Merge are therefore abstract-level only.** For ACD, the official code README was
    also read.
- **Rules.**
  - "Not reported" means not found in what was read; it never means the paper lacks it.
  - Abstract-level entries are never used for a claim of absence.
  - Benchmark numbers from different protocols are not compared with each other or with ours.

| Code | Paper | Venue (verified on) | Depth |
| --- | --- | --- | --- |
| P1 | ThinkingViT | CVPR 2026 poster (cvpr.thecvf.com/virtual/2026/poster/36340); arXiv 2507.10800 v3, 26 Mar 2026 | `FULL-S` (arXiv v3, all sections and appendices; figures not visible) |
| P2 | ACD | CVPR 2026 **Findings**, pp. 6706-6716 (CVF Open Access abstract page) | `ABS` + official code README |
| P3 | TE-VMamba | CVPR 2026 main, pp. 4363-4372 (CVF Open Access abstract page) | `ABS` |
| P4 | SRE-Merge | ICML 2026 poster (icml.cc/virtual/2026/poster/64967); PMLR v306 entry not located | `ABS` |
| P5 | VPFA | ICML 2026, PMLR v306 pp. 79598-79614 (proceedings.mlr.press/v306/liu26fr.html); arXiv 2510.00936 v1 1 Oct 2025, v2 28 May 2026 | `FULL-S` (v2 main text; appendix tables only as a summary) |
| P6 | PrunNet | **CVPR 2025**, pp. 15255-15264 (CVF Open Access listing); arXiv 2504.11879 | `FULL-S` (read 2026-11-17, see LITERATURE_MATRIX.md F1) |
| P7 | OSKT | ICCV 2025 (arXiv comments and journal reference); arXiv 2511.06016 v1, 8 Nov 2025 | `FULL-S` (arXiv HTML; appendix absent from HTML) |

Five of the seven are from 2026 (P1 to P5). The two from 2025 were kept because they are the only
2025 papers found that train **one** re-ID network for several budgets: P6 with compatibility, P7 without.

---

## 1. Comparison table

Our method's row uses our own protocol (Market-1501, asymmetric mAP, full-cost gallery, 3 seeds).

| | What is made cheap | One set of weights, several budgets | Cheap query vs full-cost gallery (compatibility) | Per-input budget choice | Backbone | Re-ID data | Costs reported |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **P1 ThinkingViT** (CVPR 2026) | Attention heads and width (nested, 3 then 6 heads) | Yes (nested slices) | No (classification) | **Yes**: softmax entropy threshold, second stage **reuses** first-stage tokens | DeiT-T/S sizes, Swin-S | None | GMACs, throughput (A100, batch 512), params |
| **P2 ACD** (CVPR 2026 Findings) | Separate query network (capacity and resolution, e.g. ResNet-18 at 64x64 vs ResNet-101 at 256x256 in a config name) | No (two networks) | **Yes**: the gallery network is adapted to a **frozen** query network | No | ResNet (CNN) | MSMT17 (also VeRi-776, CUB, In-Shop, SOP) | FLOPs tool in the code (`ptflops`); numbers not read |
| **P3 TE-VMamba** (CVPR 2026) | Tokens (pruning shallow, merging deep), learned per-layer thresholds | Not stated in the abstract | Not stated in the abstract | Learned thresholds (per input? not stated) | VMamba (state-space model) | Market-1501 | "over 60 %" fewer FLOPs (abstract) |
| **P4 SRE-Merge** (ICML 2026) | Tokens (training-free merging) | Training-free, so it applies to any trained ViT | Not stated in the abstract | No | ViT | Person re-ID benchmarks (unnamed in abstract) | about 24 % fewer GFLOPs (abstract) |
| **P5 VPFA** (ICML 2026) | Nothing: the low-resolution query is a degradation | One frozen backbone + residual MLP on query features | **Yes**: low-resolution query vs high-resolution gallery | No | TransReID (ViT); CLIP-ReID in appendix | MLR-Market-1501, MLR-CUHK03, MLR-VIPeR, CAVIAR | Throughput of the MLP alone; params 24.14 M |
| **P6 PrunNet** (CVPR 2025) | Weights (unstructured pruning, 20-80 %) | Yes | Yes (subnetworks vs dense) | No | ResNet, MobileNet, ViT-S | Market-1501, MSMT17 (appendix, ResNet-18) | FLOPs and storage in a figure; no latency |
| **P7 OSKT** (ICCV 2025) | Width (students generated from one "weight chain") | One chain generates several students (each fine-tuned) | **No** (not evaluated) | No | ResNet-50, ViT-S, ViT-B (PASS) | Market-1501, MSMT17, CUHK03 | Epoch counts; params only for OSNet / MSINet |
| **Ours `sp_mixed`** | Input resolution **and** token count (ToMe merging) | Yes | Yes (own full-cost gallery) | Tested (cascade, E17): signal good, cascade dominated | ViT-S/16 | Market-1501 | GFLOPs, measured latency in 4 regimes, training GPU-hours, equal-budget baseline |

---

## 2. Paper-by-paper analysis

### P1. ThinkingViT: Matryoshka Thinking Vision Transformer for Elastic Inference

- **Bibliographic.**
  - Authors: Ali Hojjat, Janek Haberer, Sören Pirk, Olaf Landsiedel (Kiel University, TUHH, UNU-INWEH).
  - Venue: CVPR 2026 poster (poster page title verified). An earlier ICML 2025 workshop version used the
    title "Nested Thinking ViT".
  - URLs: https://arxiv.org/abs/2507.10800 (v3, 26 Mar 2026); code github.com/ds-kiel/ThinkingViT.
  - Depth: `FULL-S`.
- **Problem.** Nested ("Matryoshka") ViTs such as MatFormer, HydraViT and SortedNet run at several sizes but
  spend the same compute on every input. Separate routers are costly. Simply looping a ViT gives little: on
  DeiT-Tiny, 72.2 becomes 74.0 at 2x the cost.
- **Architecture.**
  - A ViT whose subnetworks V_{d_i,h_i} use the first d_i embedding dimensions and the first h_i heads, with
    nested subnetworks V_{d1,h1} ⊂ V_{d2,h2} ⊂ ... All parts are sliced: embedding, attention, MLP and norm.
  - Main setting: two stages, 3 heads then 6 heads (the second stage is DeiT-S-sized).
  - **Token Recycling (Eq. 4):** `E_fused = α · Proj_{d_i→d_j}(z_L) + E^{d_j}(x)`.
    - z_L is the final-layer token output of the previous stage.
    - E^{d_j}(x) is the fresh patch embedding at the new width.
    - α is a learnable scalar (initialised to 0; it converges to −0.19 for 3H→6H).
    - The projection is learnable.
  - Ablation: recycling the final-layer tokens beats layer-wise snapshots, memory tokens and KV caching.
- **Training.**
  - Loss (Eq. 5): `L = Σ_i λ_i · L_cls(V_{d_i,h_i}(x), y)`, with λ = [0.5, 0.5].
  - All stages run in one forward pass. The sandwich rule or stochastic sampling is used when there are many
    stages.
  - Data: ImageNet-1K at 224x224, DeiT recipe, initialised from DeiT-Tiny.
  - Training cost: about 20 min/epoch, against 7 for DeiT-T and 13 for DeiT-S (their hardware).
- **Inference.**
  - Halt after stage k if the entropy of the softmax output, `H(f_k) = −Σ_c f_k^(c) log f_k^(c)`, is ≤ τ.
    The main text says "lower than", Algorithm 1 says "≤".
  - Uncertain images are re-batched and sent to the next stage.
  - Average cost: `GMACs_τ = Σ_i p_i · Σ_{j≤i} GMACs_j`, where p_i is the fraction of images that stop at
    stage i. **This is the same cost model as our cascade** (Section 4.5 of the report).
- **Results** (ImageNet-1K; reported numbers; 22.01 M params in every row).

  | Setting | Top-1 (%) | GMACs | Throughput (img/s) |
  | --- | --- | --- | --- |
  | 3H only | 73.58 | 1.25 | 10,048 |
  | 3H→6H, τ = 0 (always escalate) | 81.44 | 5.85 | 3,157 |
  | τ = 0.5 | 81.39 | 3.98 | 4,381 |
  | τ = 1.1 | 79.99 | 2.72 | 5,919 |

  - Reference points: DeiT-S 79.9 at 4.6 GMACs; DeiT-Base is only 0.36 above ThinkingViT at 86.6 M params.
  - Up to +2.0 points at equal throughput and +2.9 at equal GMACs (from a figure).
  - About 2 % of images are correct at stage 1 and wrong at stage 2 ("overthinking").
- **Strengths.**
  - Escalation is **not paid twice in information**: the second stage reuses the first.
  - Its stage-wise accuracy-cost curve is reported in full.
  - Throughput is measured, not only GMACs.
- **Limitations.**
  - Stated by the authors: higher training cost of nested models; "jumping too far" (3H→12H) is worse than
    going through 9H.
  - From our reading:
    - Classification only, with no retrieval or compatibility constraint.
    - The stage-1 compute is still spent again in FLOPs terms: GMACs add up, as in our cascade. The gain comes
      from the stage-2 accuracy that recycling buys.
    - The latency cost of re-batching is not measured.
- **Relative to `sp_mixed`.**
  - **Similar:** one set of weights trained jointly on several budgets with a sum of per-budget losses (our
    two-pass objective without the alignment term is Eq. 5 with a re-ID loss).
  - **Similar:** a per-input confidence decides escalation, with the same average-cost formula.
  - **Different:**
    - Its knob is width and heads, ours is input resolution and tokens.
    - It has no compatibility requirement: a re-ID gallery embedded at stage 2 must be comparable with stage-1
      queries.
    - Its confidence is softmax entropy; ours is the top-1 gallery similarity, which needs no classifier and
      works for unseen identities.
  - **Most important for us:** token recycling is exactly the missing piece identified by our E17 cascade
    result (Report Section 6.9: "an escalated query pays the cheap pass *and* the full pass").

### P2. ACD: Asymmetric Collaborative Distillation for Asymmetric Image Retrieval

- **Bibliographic.**
  - Authors: Yi Xie, Huaidong Zhang, Xuandi Luo, Yan Zhou, Shengfeng He (South China University of
    Technology, Foshan University, Singapore Management University; the same group as D3still).
  - Venue: CVPR 2026 Findings, pp. 6706-6716.
  - URLs: https://openaccess.thecvf.com/content/CVPR2026F/html/Xie_Asymmetric_Collaborative_Distillation_for_Asymmetric_Image_Retrieval_CVPRF_2026_paper.html;
    code github.com/SCY-X/Asymmetric_Collaborative_Distillation.
  - Depth: `ABS` + README. **No equations were read.**
- **Problem.** Asymmetric retrieval pairs networks of different capacity **and input resolution**. One-way
  distillation (query imitates gallery) ignores that both networks take part in inference.
- **Method (abstract and README).**
  - The direction is reversed: a well-trained **query network is frozen** and the **gallery network is
    optimised** for compatibility in a "collaborative logit space". The README says the gallery network is
    learnable while the query network is frozen, initialised from AIR-Distiller student weights.
  - A search snippet of the supplement mentions a KL-divergence alignment weighted by β (not verified).
  - RAHO (Resolution-Aware Hierarchical Optimisation) gives module groups of the gallery network adaptive
    learning rates according to their contribution to cross-resolution performance.
  - ACD is an add-on to existing distillers: FitNet, CC, CSD, RAML, ROP, D3still and UGD (Neural Networks
    2025).
- **Data and results.**
  - Abstract: four benchmarks; +2.25 mAP and +3.78 R1 on SOP.
  - README, corrected MSMT17 table (ResNet-101-IBN gallery): D3still 40.90 → 42.43 mAP with ACD, UGD 41.98 →
    41.99.
  - A config name in the README, `ResNet101_256x256_ResNet18_64x64`, shows the query runs at 64x64 against a
    256x256 gallery.
  - Different protocol from ours (separate CNNs, MSMT17), so not comparable.
- **Strengths.**
  - It recognises resolution asymmetry as part of asymmetric retrieval in 2026.
  - It is a plug-in for seven distillers, with public code.
- **Limitations (our reading).**
  - It needs a gallery re-embedding whenever the gallery network changes.
  - Two networks.
  - CNN only, as far as the README shows.
  - The authors themselves report an error in their Table 2 (UGD gains).
  - Gains of 0.01-2.35 mAP on MSMT17.
- **Relative to `sp_mixed`.**
  - It is the 2026 state of the asymmetric-distillation line that our alignment loss belongs to (D3still's
    feature term).
  - Reversing which side is trained is the counterpart of our two baselines: `fg_mixed` freezes the gallery
    and trains the query; ACD freezes the query and trains the gallery; `sp_mixed` trains both jointly as one
    network.
  - It does not use one network for both sides or input-side knobs of a ViT (not stated in what was read).

### P3. TE-VMamba: SSM-Aware Token-Efficient VMamba via Adaptive Patch Pruning and Merging for Person Re-ID

- **Bibliographic.**
  - Authors: Huiyuan Huang, Sang Min Yoon.
  - Venue: CVPR 2026, pp. 4363-4372.
  - URL: https://openaccess.thecvf.com/content/CVPR2026/html/Huang_SSM-Aware_Token-Efficient_VMamba_via_Adaptive_Patch_Pruning_and_Merging_for_CVPR_2026_paper.html.
  - Depth: `ABS` (PDF not readable here).
- **Problem.** Even linear-complexity state-space models waste compute on dense tokens in re-ID.
- **Method (abstract).**
  - Layer-adaptive **pruning** of low-importance tokens in shallow layers.
  - Depth-aware **merging** of similar tokens in deep layers.
  - **Learnable per-layer thresholds** that balance accuracy and cost.
- **Results (abstract).** More than 60 % fewer FLOPs on Market-1501 with "competitive accuracy". No mAP was
  read.
- **Strengths.** A re-ID-specific token reduction at a main venue, with prune and merge combined by depth.
- **Limitations.**
  - Not readable beyond the abstract.
  - Whether queries and gallery are reduced asymmetrically, and whether latency was measured, is unknown.
- **Relative to `sp_mixed`.**
  - It confirms that token reduction for re-ID is an active 2026 topic: our use of merging as a knob is not
    new.
  - Its schedule (prune early, merge late) differs from our gradual ToMe.
  - Our measured latency shows that merging can be *slower* at GPU batch 1. If TE-VMamba reports only FLOPs,
    the same caveat may apply to it (hypothesis, not verified).

### P4. SRE-Merge: Spatially-Regularized Entropy for Discriminative Token Merging in Fine-Grained Re-ID

- **Bibliographic.**
  - Authors: Shangze Li, Yifan Xu, Jingmiao Liang, Yongfei Zhang, Yuzhuo Ma, Yingbo Qu.
  - Venue: ICML 2026 poster.
  - URLs: https://icml.cc/virtual/2026/poster/64967; OpenReview forum id IJ2C56ra1a (bot check, not read).
  - Depth: `ABS`.
- **Problem.** Generic token pruning or merging criteria, designed for classification, smooth away the fine
  local details that re-ID needs.
- **Method (abstract).** Training-free merging with three parts:
  - SES-Assess scores tokens by "Spatial-Entropic Mass", combining spatial structure with local attention
    entropy;
  - HCA-Match chooses merge pairs from feature similarity plus that mass;
  - EPW-Fuse weights the fusion by the mass to counter variance shrinkage.
- **Results (abstract).** About 24 % fewer GFLOPs on a base ViT with competitive accuracy. No numbers were
  read.
- **Relative to `sp_mixed`.**
  - It is a drop-in replacement candidate for ToMe's bipartite matching in our merge knob.
  - It does not address compatibility (not stated in the abstract).
  - Its claimed 24 % saving is small next to our resolution knob (−62 % GFLOPs at 160x80 with −0.9 mAP), on
    our data only.

### P5. VPFA: Resolution as a Direction: Vector-Panning Feature Alignment for Cross-Resolution Re-ID

- **Bibliographic.**
  - Authors: Zanwu Liu, Chao Yuan, Bo Li, Xiaowei Zhang, Guanglin Niu (Beihang University, Qingdao
    University).
  - Venue: ICML 2026, PMLR 306:79598-79614.
  - URLs: https://arxiv.org/abs/2510.00936 and https://proceedings.mlr.press/v306/liu26fr.html; code
    github.com/ashmentlzw/VPFA (not checked).
  - Depth: `FULL-S` (main text; appendix only as a summary).
- **Problem.** In cross-resolution re-ID, low-resolution (LR) queries meet a high-resolution (HR) gallery.
  Super-resolution is costly and does not target the embedding mismatch.
- **Observation.** The identity-averaged HR−LR feature difference vectors point in almost the same direction
  across identities: cosine 0.991-0.998 on Market and CUHK03 at 2x, 3x and 4x. Resolution looks like a
  "direction" in feature space.
- **Method.** A residual MLP on the features of a **frozen** backbone:
  - (2) `ẑ_LR = z_LR + Gate(VP(z_LR))`;
  - (3) `VP(z) = W3 σ(W2 σ(W1 z))`;
  - (4) `Gate = tanh(W4 ·)`;
  - (5) `L_VPL = ‖ẑ_LR − z_HR‖²`, expanded as `r² + R² − 2rR cos θ`.
  - The MLP has a 3840-d input, two hidden blocks of width 2048 (linear, LayerNorm, ReLU), and a tanh output
    back to 3840.
- **Training.**
  - 5,000 paired (LR, HR) features, 120 epochs, Adam (lr 2e-4), batch 32.
  - No identity labels.
  - The backbone and the gallery are unchanged.
- **Data.** MLR-Market-1501, MLR-CUHK03, MLR-VIPeR: synthetic downsampling by r ∈ {2, 3, 4}. CAVIAR: real low
  resolution.
- **Results** (Rank-1; the main text reports no mAP).
  - MLR-Market 94.1, against a TransReID baseline of 90.3.
  - MLR-CUHK03 94.7; MLR-VIPeR 50.5; CAVIAR 66.6.
- **Costs.** The MLP alone runs at millions of samples/s and has 24.14 M params. No end-to-end FLOPs or
  latency.
- **Not stated in what was read:** whether LR queries enter the ViT at native low resolution or are resized
  to the full input size. **This decides whether any compute is saved.** Also not stated: how the training
  pairs are generated.
- **Limitations (stated).**
  - It depends on the frozen backbone.
  - It does not prove that a single direction exists.
  - It relies mainly on synthetic downsampling.
  - It does not restore detail.
- **Relative to `sp_mixed`.**
  - It is the 2026 version of our "baseline + linear (ridge) map" baseline. VPFA's map is non-linear, residual
    and gated, and label-free. Our ridge map recovered only part of the loss: 78.96 against 83.01 for
    `sp_mixed` at 160x80.
  - **A VPFA-style MLP on our frozen baseline is the fairest 2026 post-hoc baseline we have not run.**
  - Its direction finding is a testable hypothesis on our own drift data (`mechanism.py` already computes the
    cheap-minus-full drift).

### P6. PrunNet: Learning Compatible Multi-Prize Subnetworks for Asymmetric Retrieval

- **Bibliographic.**
  - Authors: Yushuai Sun, Zikun Zhou, Dongmei Jiang, Yaowei Wang, Jun Yu, Guangming Lu, Wenjie Pei.
  - Venue: **CVPR 2025**, pp. 15255-15264, DOI 10.1109/CVPR52734.2025.01421. This corrects our earlier
    "arXiv, venue not stated".
  - URL: arXiv 2504.11879.
  - Depth: `FULL-S` (details: LITERATURE_MATRIX.md F1).
- **Method.**
  - One dense network. Subnetworks keep the top c % of connections per layer by a learned score.
  - Compatibility with the dense network comes through a shared classifier (BCT- or SFSC-style).
  - Conflicting gradients between dense and subnetworks are projected and re-weighted.
- **Data and results.** ViT-S on CUB; ResNet-18 on Market-1501 and MSMT17 (appendix). On VeRi-776 it beats
  SFSC.
- **Limitation (stated).** Unstructured sparsity is hard to accelerate in hardware. No latency is reported.
- **Relative to `sp_mixed`.**
  - The closest in methodology among 2025-2026 papers: one network, several compatible budgets, shared
    classifier, ViT-S among its backbones, re-ID data.
  - It differs in the knob (weights vs input) and in measured speed-up.

### P7. OSKT: One-Shot Knowledge Transfer for Scalable Person Re-Identification

- **Bibliographic.**
  - Authors: Longhua Li, Lei Qi, Xin Geng (Southeast University).
  - Venue: ICCV 2025 (arXiv comment and journal reference).
  - URL: https://arxiv.org/abs/2511.06016. No code link found.
  - Depth: `FULL-S` (the appendix with student sizes is absent from the HTML).
- **Problem.** Edge re-ID needs models of many sizes, and pruning or distilling each size separately is
  costly.
- **Method.**
  - A "weight chain" keeps the teacher's depth with fewer rows per layer, initialised by clustering the
    teacher rows.
  - Training: the teacher and the smallest student are trained jointly.
  - Loss: `L = L_T + L_S + α L_ref`, where `L_T = L_id + L_tri` for the teacher and likewise for the student.
    `L_ref` keeps each chain row close to the teacher rows of its cluster. α = iter/n_iter for ViTs.
  - Students of any intermediate width are generated by stacking chain rows in proportion, with no training,
    and then **fine-tuned** separately (120 epochs each for ViT).
- **Results** (Market-1501, each student evaluated **alone**, mAP/R1).

  | Student | DepGraph | OSKT |
  | --- | --- | --- |
  | ViT-S-S1 | 56.5 / 74.1 | 74.2 / 87.1 |
  | ViT-S-S2 | 69.0 / 84.2 | 77.2 / 89.0 |
  | ViT-B-S2 | 81.5 / 91.7 | 82.9 / 92.4 |

- **Compatibility.** Cross-model retrieval (student query vs teacher gallery) is **not evaluated**.
- **Relative to `sp_mixed`.**
  - Same goal of serving several budgets with re-ID ViTs from one training.
  - But it gives separate fine-tuned models with incompatible embeddings, so a gallery must be re-embedded per
    size. This is exactly the cost our setting avoids.
  - Its numbers are symmetric (query and gallery from the same student) and use PASS pre-training: not
    comparable with ours.

### Found but not selected (lead level, not read)

- **Outside the 2025-2026 window:** HashReID (WACV 2024) uses early exits for re-ID with binary codes. The
  snippet says more than 70 % of Market-1501 samples exit early. It is relevant to the per-query budget claim
  and must be read before iteration 2.
- **Search-result level only, 2025-2026:**
  - DCAH (AAAI 2026, asymmetric retrieval with quantisation);
  - UGD (Neural Networks 2025, asymmetric distillation);
  - compatible-learning preprints (arXiv 2606.12488 stationary representations; 2610.29550 hyperspherical
    simplex);
  - MM-Matryoshka (arXiv 2606.07654) and Tevatron-Elastic (SIGIR-AP 2026), budget-elastic document
    retrieval;
  - asymmetric visual place recognition (arXiv 2512.13055);
  - ReIDMamba (arXiv 2511.07948).

---

## 3. The closest papers in detail, compared with `sp_mixed`

The four closest papers are P6 PrunNet, P1 ThinkingViT, P5 VPFA and P2 ACD. Our objective, for reference
(Report Section 4.4):

`L = CE(full) + Tri(full) + CE(b) + Tri(b) + λ · mean(1 − cos(e_b, sg(e_full)))`, λ = 1, with the cheap budget
b drawn from {4 resolutions} ∪ {5 merge rates}, one ViT-S, and a classifier shared between the passes.

| Aspect | PrunNet | ThinkingViT | VPFA | ACD | `sp_mixed` |
| --- | --- | --- | --- | --- | --- |
| Number of networks | 1 | 1 | 1 frozen + MLP | 2 | 1 |
| Which side is trained for compatibility | Subnetworks and dense, jointly | (none needed) | Query-side map only | **Gallery** (query frozen) | Both passes jointly |
| Compatibility signal | Shared classifier + gradient projection | — | MSE to the HR feature | Collaborative logit space (KL, unverified) | Shared classifier + cosine to the detached full embedding |
| Budget knob | Weights | Heads and width | (degradation, not a knob) | Separate network + resolution | Input resolution + tokens |
| Labels needed | Yes | Yes | No | Yes (not verified) | Yes |
| Gallery re-embedding needed after training | No | n/a | No | **Yes** | No |
| Measured speed-up | No | Throughput | MLP only | Not read | Yes (4 regimes) |
| Per-input adaptivity | No | Yes, with reuse | No | No | Tested, dominated |

**Verified similarities.**
1. Our joint multi-budget loss has the same structure as ThinkingViT's Eq. 5 and PrunNet's dense plus
   subnetwork training: a sum of task losses over budgets through shared weights.
2. Our shared classifier is PrunNet's compatibility mechanism (BCT), applied to input budgets.
3. Our alignment term is the feature term of the D3still / ACD line. VPFA's loss (MSE to the HR feature) is
   the same quantity up to normalisation: `‖a − b‖² = 2 − 2cos` for unit vectors.

**Verified differences.**
1. None of the seven uses input resolution **and** token count of one ViT as compute knobs with compatibility
   to the same model's full-cost gallery.
2. ACD and VPFA treat low resolution as given by the deployment (a degradation, or a query device), not as a
   knob chosen to save compute in the same network.
3. Only ThinkingViT (throughput) and we report a measured speed. Only we report a configuration where the
   cheaper setting is *slower* (merging at GPU batch 1).

**Hypotheses, not verified.**
1. ThinkingViT-style token recycling could make our cascade non-dominated (Section 5, D1).
2. A VPFA-style non-linear map on a frozen baseline could close much of the gap to `sp_mixed`. That would
   weaken the case for joint training, as `fg_mixed` already partly does.
3. SRE-Merge or TE-VMamba criteria could make merging competitive with resolution in accuracy, but not
   necessarily in latency.

---

## 4. Is our preliminary gap still valid?

**Verdict: valid but must be narrowed. There is no substantial overlap with the core claim, but one of the
four gap bullets of Report Section 3 is now largely covered.**

| Gap bullet (Report Section 3) | Status after this review | Reason |
| --- | --- | --- |
| Input-side query budgets of one ViT re-ID model as a compute knob, compatible with its own full-cost gallery | **Still not found** (in what was read) | PrunNet: weights; OSKT: width, no compatibility; ACD: two networks; VPFA: resolution as degradation, frozen backbone; TE-VMamba and SRE-Merge: compatibility not stated (abstract only, so **not** an absence claim) |
| One set of weights compatible under two input knobs at once | Still not found | No paper combines resolution and token knobs |
| Cost account (equal-GPU-hour baseline, full-cost deficit, measured latency) | Still not found as a combination; partly in ThinkingViT (throughput) | Most 2026 re-ID efficiency papers read report FLOPs only (abstract level for P3 and P4) |
| Retrieval-side confidence to choose the query budget per query | **Narrow it** | ThinkingViT (CVPR 2026) does per-input escalation with reuse inside one nested ViT, for classification; HashReID (2024, unread) does early exit for re-ID. What remains: a *gallery-similarity* signal (top-1 cosine, no classifier) under a compatibility constraint. Our own E17 result shows that this alone is not enough without reuse. |

**Refined gap statement (proposed for iteration 2):**
> One ViT re-ID model whose *input-side* query budgets (resolution, tokens) remain compatible with its own
> full-cost gallery, evaluated by measured latency and equal-budget accounting, with per-query escalation
> that reuses the cheap pass (adapting ThinkingViT-style recycling to retrieval under a compatibility
> constraint).

**What must not be claimed:**
- that token reduction for re-ID is new (TE-VMamba, SRE-Merge, 2026);
- that resolution-asymmetric retrieval is new (ACD 2026, Suma and Tolias 2023, VPFA 2026);
- that one network with several compatible budgets is new (PrunNet 2025, SFSC 2023);
- that confidence-based elastic inference for ViTs is new (ThinkingViT 2026).

---

## 5. Three directions for iteration 2 (ranked)

The ranking weighs feasibility on one RTX 5080, research value for the thesis "accuracy per unit of compute",
and compute. GPU-hour figures are **estimates** from our iteration-1 run times (one `sp_mixed` run about
0.4 GPU-h).

### D1. Cascade with cheap-pass reuse (ThinkingViT-style token recycling for compatible retrieval)

Rank 1: highest value, feasible.

- **What.** If the cheap query (e.g. 128x64) has a low top-1 gallery similarity, run a second pass at higher
  resolution. Its input is `E(x_high) + α · Proj(upsample(z_cheap))`, with α initialised to 0. Train jointly
  with the existing `sp_mixed` losses; the escalated embedding must still align with the full-cost gallery
  embedding.
- **Why.**
  - It addresses the exact cause of our negative E17 result (the escalated query pays twice and gains nothing
    from the first pass).
  - It is the one bullet of our gap that 2026 work narrowed. Adapting ThinkingViT to retrieval with a
    compatibility constraint and a gallery-similarity signal is a credited, defensible iteration-2 mechanism.
- **Expected outcome (hypothesis).** The cascade becomes competitive with fixed budgets. It is not dominated
  if the recycled second pass gains more than about 0.5-0.7 mAP over a plain second pass at equal GFLOPs
  (that is the E17 gap).
- **Risks.**
  - The token grids differ between resolutions; this needs interpolating the recycled tokens.
  - The gallery may need the recycled path too.
  - Latency: two passes at GPU batch 1 cost about 2 x 6 ms.
- **Compute (estimate).** Implementation 1-2 days; 3 seeds × about 0.5 GPU-h ≈ 1.5-2 GPU-h; evaluation reuses
  `adaptive.py`.

### D2. Fair 2026 baselines: post-hoc non-linear map (VPFA-style) and gallery-side adaptation (ACD-style)

Rank 2: most feasible, cheapest, needed for fairness.

- **What.**
  - (a) On the frozen 120-epoch baseline, train VPFA's residual gated MLP from cheap-query features to
    full-cost features (label-free, minutes of GPU). Compare with our ridge map and `sp_mixed` at every budget.
  - (b) Optionally, an ACD-style variant: freeze a cheap-query model and fine-tune the gallery side. This needs
    gallery re-embedding, a cost to account for.
  - Also test VPFA's "resolution is a direction" observation on our drift vectors (evaluation only).
- **Why.**
  - Our strongest result (+0.8 to +4.1 mAP over the equal-budget baseline with a *linear* map) could shrink
    against a non-linear map.
  - The project rules require equal-budget baselines, and VPFA is now the 2026 standard for post-hoc
    correction.
- **Expected outcome.** Unknown. If the MLP closes most of the gap, joint training's value reduces to "one
  model, no extra module". That is still a valid but smaller claim and must be reported.
- **Compute (estimate).** Under 0.5 GPU-h in total (features cached; 3 seeds of a small MLP). Code: an
  extension of `mechanism.py`.

### D3. Native-resolution second dataset and a third knob (width or heads, as in ThinkingViT / PrunNet)

Rank 3: high value for generality, least feasible now.

- **What.**
  - Repeat the main comparison on a dataset whose crops are not natively 128x64: CUHK03, or an MLR protocol
    (MLR-CUHK03, as used by VPFA) for real cross-resolution.
  - Optionally add a nested-width knob and test whether input-side and width knobs are complementary in
    measured latency.
- **Why.**
  - Our main limitation is that Market-1501 crops are natively 128x64, so the low-resolution knob may be
    flattered.
  - Reviewers of 2026 re-ID efficiency papers (TE-VMamba, SRE-Merge) will expect more than one dataset.
- **Blocker.** CUHK03 needs the user's own sign-up (CLAUDE.md: downloads need explicit OK).
- **Compute (estimate).** 3-6 GPU-h for baselines plus `sp_mixed` on one new dataset. A width knob needs new
  slicing code and more.

---

## 6. Unresolved questions

1. How TE-VMamba (CVPR 2026) and SRE-Merge (ICML 2026) evaluate: are queries and gallery reduced alike? What
   mAP do they report on Market-1501, and do they measure latency? This needs the PDFs, which the user can
   download from CVF Open Access and OpenReview.
2. ACD's exact losses and its query and gallery architectures across its four benchmarks. This needs the PDF
   and the supplement.
3. Whether VPFA feeds low-resolution queries at native size (which would save compute) or resized to full
   size. The text read is silent; the code repository could settle it.
4. Whether HashReID (WACV 2024) or a 2025-2026 successor already does per-query early exit with a
   compatibility constraint in re-ID.
5. Whether ThinkingViT's recycling transfers across *resolutions*: its stages share the input size and change
   width, while ours change the token grid.
