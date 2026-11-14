# PROJECT_CONTEXT.md — Low-resource person re-identification research project

Last updated: Saturday 2026-11-14, 18:45 Asia/Dhaka (UTC+06:00).
Purpose of this file: let a new Claude Code session continue the project with no access to the
original conversation. Read sections 1, 2, 9, 10 and 11 first if short on time.

---

## 1. Original project objectives

- **Field:** person re-identification (re-ID).
- **Ultimate goal:** find a genuine research gap, propose an original methodology that addresses it,
  and publish a research paper.
- **First iteration goal:** survey the current state of the art and methodologies, identify gaps, and
  propose the user's own methodology.
- **Recommended techniques (from the user's instructor):** self-distillation and/or token pruning.
- **Hard requirement:** novelty. The user said: "we are not trying to copy others' work, rather finding
  a novel idea." Finding gaps and working on them is mandatory.

## 2. Instructions and requirements given by the user

1. **Resource philosophy from the instructor: "no money, no computation, no data."** Not literally
   achievable, but the project must emphasise how well a method performs on limited resources.
   This turned efficiency from an add-on into the thesis of the paper.
2. Novel idea required; do not replicate existing papers.
3. Use self-distillation and/or token pruning where sensible.
4. **Deadline:** the user must show some part of iteration 1 to the instructor within 16 hours of
   Sat 2026-11-14 18:24 Dhaka time, i.e. by about **Sun 2026-11-15 10:24 Dhaka time**.
5. The user wants the project continued in Claude Code from here.
6. Hardware is the user's own PC (section 3); other hardware can be arranged "if necessary", but the
   plan assumes none is needed.

Working rules adopted for the project (assistant-proposed, user did not object):

- Report trade-offs, not single accuracy numbers (section 6.4).
- Hold baselines to the same compute and data budget as the proposed method.
- 3 seeds per configuration; report mean and spread. Gains inside the noise are not results.
- Log GPU-hours, peak VRAM and wall-clock for every run from day one.
- Never present unverified numbers as verified. Be explicit about what has and has not been run.

## 3. Hardware and software constraints

| Item | Detail |
| --- | --- |
| GPU | NVIDIA RTX 5080, 16 GB VRAM |
| CPU | AMD Ryzen 7 7800X3D (also the "no-GPU deployment" latency proxy) |
| RAM | 16 GB DDR5. **This is the bottleneck, not the GPU.** |
| OS | Not confirmed. Probably Windows; scripts are written to be Windows-safe (`if __name__ == "__main__"`, modest worker counts). Confirm with the user. |
| Budget | No paid compute, no extra pretraining data, no external teacher models |

Practical consequences:

- Data loader workers: 4 by default; drop to 2 if RAM runs short. Do not cache datasets in memory.
- The RTX 50 series needs a recent PyTorch/CUDA build. Install from the selector on pytorch.org.
  Old re-ID repos (TransReID, CLIP-ReID) pin old versions and will need porting; that is why the
  project uses its own small kit instead of those codebases for now.
- ViT-B training fits with mixed precision. ViT-L and dual foundation encoders are tight on 16 GB
  and out of scope.
- The assistant's cloud sandbox had no GPU and no datasets. The kit was smoke-tested there on CPU
  with torch 2.15.0 and timm 1.0.31.

Rough, **unverified** time estimates for the 5080 (time the first run and rescale):
ViT-S on Market-1501 well under an hour; ViT-B on Market-1501 about an hour; MSMT17 a few hours;
inference-only sweeps take minutes.

## 4. Decisions made and their reasoning

| Decision | Reasoning |
| --- | --- |
| Do not chase accuracy on Market-1501 | A July 2026 survey reports standard benchmarks are near saturation. A +0.3 mAP paper will struggle. |
| Reframe the research question as accuracy per unit of compute, labels and training time | Matches the instructor's "no money, no computation, no data" emphasis and explains why self-distillation (no external teacher) and token pruning (less compute) were recommended. |
| Setting: supervised image re-ID, evaluated in-domain and cross-dataset; occluded re-ID as a second test bed | Fits the hardware; cross-dataset evaluation is inference-only. |
| Out of scope: video re-ID, ViT-L, dual foundation-model encoders (e.g. DINO+CLIP), large-scale person pretraining | Too heavy for the budget or contradict "no extra data". |
| Main backbone ViT-S/16; ViT-B/16 and OSNet as references | ViT-S fits the low-resource thesis. **Open question left as a comment in the report: would the instructor prefer ViT-B?** |
| Always compare against a small CNN (OSNet) at equal cost | Reviewers will ask whether a pruned ViT beats simply using a small CNN. |
| Datasets: Market-1501 first; then MSMT17, CUHK03, Occluded-REID | DukeMTMC was withdrawn by its creators and some reviewers object to its derivatives (including Occluded-Duke). Check the target venue's policy before using them. |
| Write a small self-contained kit instead of porting TransReID tonight | Porting old repos to the 5080's PyTorch build would eat the 16-hour window. |
| Distil embeddings, not class logits | The Part-Aware Transformer paper (ICCV 2023) reports plain logit-level self-distillation hurt cross-domain re-ID. |
| "Token pruning + distillation for re-ID" is not claimed as novel | AMG (2023), SUReID (2024), SRE-Merge (ICML 2026), TE-VMamba (CVPR 2026) already exist. Novelty must come from a gap inside that space. |
| "Pruning cuts training cost" (H4) demoted to a measured side benefit | Token dropping to speed training is well known (MAE etc.); too generic to be the main claim. |

## 5. Research findings and references

Basis: about 11 web searches on 2026-11-14, reading abstracts and search summaries, **not full
papers**. Items marked * are from the assistant's prior knowledge and were not re-checked.
Everything here needs confirmation by full reads in the weeks 2 to 3 literature pass.

### 5.1 State of the art in brief

- **Saturation.** Standard supervised benchmarks are close to their ceiling; current work competes
  on generalization, robustness, modality and cost.
- **Backbones.** ViTs became the default after TransReID* (ICCV 2021). Self-supervised person
  pretraining (TransReID-SSL, SOLIDER*, PersonViT) raised accuracy further; PersonViT's authors
  flag the high pretraining cost.
- **Foundation models.** CLIP-ReID* (AAAI 2023) is a common baseline. DRFormer (Feb 2026) fuses
  DINO (local texture) and CLIP (global semantics). A 2026 benchmark (11 models, 9 datasets)
  reports fine-tuned language-aligned models generalize best and that scale gives only marginal
  cross-domain gains (CLIP-L/14 barely better than CLIP-B/16).
- **Efficiency.** Token reduction (AMG, SUReID, SRE-Merge, TE-VMamba); lightweight architectures
  and compression (OSNet*, ADLN, TE-TransReID, OSKT); cloud-to-edge distillation (MLLMEmbed-ReID).
- **Self-distillation.** Used as a regulariser or between branches. Logit-level self-distillation
  hurt domain-generalizable re-ID in the PAT paper; they used part-level soft labels instead.
- **Pattern in the efficiency papers (from abstracts, to be confirmed):** evaluated in-domain,
  query and gallery at the same token budget, full training sets, inference cost reported but not
  training cost.

### 5.2 Literature matrix (21 works so far; target 30 to 40 with full reads)

| Work | Area | What it does | Cost evidence | Gap for this project | Link |
| --- | --- | --- | --- | --- | --- |
| TransReID*, ICCV 2021 | Backbone | Pure ViT, jigsaw patches, side-information embeddings | none | Full-token ViT-B baseline | arXiv 2102.04378 (ID from memory) |
| CLIP-ReID*, AAAI 2023 | Foundation model | Two-stage identity prompt learning on CLIP | none | Full tokens; needs CLIP pretraining | arXiv 2211.13977 (ID from memory) |
| OSNet*, ICCV 2019 | Lightweight CNN | Omni-scale CNN, a few million parameters | params, FLOPs | Low-cost baseline a pruned ViT must beat | arXiv 1905.00953 (ID from memory) |
| PersonViT, 2024 | Pretraining | MIM + DINO on large unlabeled person data | authors note high pretraining overhead | Outside a no-extra-data budget | https://arxiv.org/abs/2408.05398 |
| DRFormer, 2026 | Foundation model | Bidirectional fusion of DINO and CLIP | none in abstract | Two encoders roughly double cost | https://arxiv.org/abs/2602.01059 |
| Person Re-ID in 2025: what works?, 2026 | Benchmark | 11 models on 9 datasets, cross-domain | model sizes | Scale alone does not buy generalization | https://arxiv.org/abs/2601.20598 |
| Part-Aware Transformer (PAT), ICCV 2023 | Generalization | Cross-ID part similarity proxy task + part-guided self-distillation | none | Logit self-distillation hurt; no efficiency angle | https://arxiv.org/abs/2308.03322 |
| AMG, 2023 | Token pruning | Attention-map-guided token and head pruning | 29.4% FLOPs saved on ViT-B, -0.2 Rank-1, +0.4 mAP (abstract) | In-domain only (Occluded-Duke, Market-1501) | https://arxiv.org/abs/2304.01452 |
| SUReID, 2024 | Token pruning + KD | Hierarchical token sparsification, non-parametric feature-alignment KD, noise occlusion augmentation | inference speed | Uses a teacher; one budget; symmetric retrieval | https://arxiv.org/abs/2401.07469 |
| SRE-Merge, ICML 2026 | Token merging | Training-free merging with spatial-entropy saliency | about 24% GFLOPs saved | Training-free, so no cross-budget alignment | https://icml.cc/virtual/2026/poster/64967 |
| TE-VMamba, CVPR 2026 | Pruning + merging | Layer-adaptive pruning and depth-aware merging in VMamba | over 60% FLOPs saved on Market-1501 | Different backbone; in-domain | https://openaccess.thecvf.com/content/CVPR2026/html/Huang_SSM-Aware_Token-Efficient_VMamba_via_Adaptive_Patch_Pruning_and_Merging_for_CVPR_2026_paper.html |
| OSKT, 2025 | Compression | One-shot knowledge transfer into a "weight chain" expandable to any model size | one-time transfer | Scales model size, not tokens; needs a teacher | https://arxiv.org/abs/2511.06016 |
| ADLN, IEEE Access 2024 | Lightweight + self-distillation | Attention module, deep-to-shallow self-distillation | params, latency | CNN; Market and Duke only | https://doaj.org/article/2b828e526d9146ab9e3cee7ca6836b79 |
| TE-TransReID, Sensors 2025 | Lightweight hybrid | Two-branch CNN + light transformer, 27.5 M params | params | Architecture change, no token reduction | https://www.mdpi.com/1424-8220/25/17/5461/html |
| MLLMEmbed-ReID, 2026 | Cloud-to-edge KD | Distils a multimodal LLM (Qwen2-VL) into an edge student | none in abstract | Needs a very large teacher | https://arxiv.org/abs/2602.12936 |
| Bi-C2R, TPAMI 2026 | Compatible features | Lifelong re-ID without re-indexing old gallery features | none | Compatibility across model updates, not compute budgets | https://arxiv.org/abs/2512.25000 |
| Cross-model compatible ensembles, 2023 | Asymmetric retrieval | Large gallery model, small query model, shared space | query-side budget | Face/image retrieval; two separate models | https://arxiv.org/abs/2303.17531 |
| Self-slimmed ViT (SiT), ECCV 2022 | Token slimming + self-distillation | Full model teaches its token-slimmed version | 1.7x speed-up, negligible drop | ImageNet classification only | https://arxiv.org/abs/2111.12624 |
| Token Compensator (ToCom), ECCV 2024 | Elastic token budget | Self-distilled plugin bridges compression degrees between training and inference | none in abstract | Classification; no retrieval compatibility | https://mlanthology.org/eccv/2024/jie2024eccv-token/ |
| Token condensation as adaptation (TCA), ICCV 2025 | Tokens under domain shift | Training-free token condensation improves CLIP cross-dataset accuracy | 12.2% to 48.9% GFLOPs saved | Zero-shot classification, not re-ID | https://arxiv.org/abs/2410.14729 |
| LUPerson, CVPR 2021 | Low-data protocol | Person pretraining; defines "small-scale" (fewer IDs) and "few-shot" (fewer images per ID) splits at 10% to 90% | none | Solves low data with extra data; protocol reusable | https://arxiv.org/abs/2012.03753 |

Generic token-reduction baselines implemented in the kit: ToMe* (ICLR 2023, arXiv 2210.09461) and
EViT* (ICLR 2022, arXiv 2202.07800); arXiv IDs from memory.

Other sources:

- Survey, The Visual Computer, July 2026: https://elmi.hbku.edu.qa/en/publications/recent-advances-in-person-re-identification-a-comprehensive-surve/
- Survey, single- to multi-modal re-ID, arXiv 2607.14821: https://papers.cool/arxiv/2607.14821
- Review of recent techniques, arXiv 2509.22690: https://arxiv.org/abs/2509.22690
- CSCH asymmetric hashing, 2024: https://link.springer.com/article/10.1007/s41095-024-0428-y
- ReViT, adaptive token length with self-distillation, arXiv 2307.02092: https://arxiv.org/abs/2307.02092
- AdaPerceiver, adaptive width/depth/tokens, arXiv 2511.18105: https://arxiv.org/abs/2511.18105
- ElastiFormer, self-distilled elastic transformers, arXiv 2411.15281
- MaskedKD, masking teacher tokens in ViT distillation, arXiv 2302.10494

### 5.3 Gap analysis

"Open" means no matching re-ID paper appeared in about ten searches. It is **not** an exhaustive
novelty check.

| ID | Hypothesis | Closest prior work | Status | Cheapest test |
| --- | --- | --- | --- | --- |
| H1 | Asymmetric token budgets: gallery encoded once with all tokens, queries with few tokens | Asymmetric retrieval with two models (face/image retrieval); Bi-C2R; SUReID prunes both sides equally | Open for re-ID | Sweep on a trained baseline: symmetric vs asymmetric mAP |
| H2 | Token reduction under domain shift | TCA (CLIP classification improves cross-dataset); re-ID pruning papers are in-domain | Open for re-ID; direction of effect unknown | Same sweep on an unseen dataset |
| H3 | Token reduction + self-distillation as a low-data regulariser | SiT on ImageNet; LUPerson low-data protocol solved with extra data | Partly explored in classification, not found in re-ID | Train on 25% of identities with and without self-distillation |
| H4 | Token reduction to cut training cost | MAE-style token dropping; suggested by PersonViT authors | Weak as a claim; report as side benefit | Training hours to reach equal mAP |

**Caution on novelty.** Training one network to work at several token budgets with
self-distillation already exists for ImageNet classification (SiT, ToCom, ReViT, AdaPerceiver,
ElastiFormer). The contribution must be retrieval-specific: embeddings from different budgets that
stay mutually comparable, tested under asymmetric retrieval, low data and domain shift. The
assistant rated novelty as **moderate, not strong**, pending diagnostics and a full novelty check.

## 6. Methodology and technical details

### 6.1 Proposed direction: token-budget self-distillation (working name TBSD)

One ViT trained so its embeddings at any token budget stay in the same space as its full-token
embeddings. Each training step:

1. Full-token forward pass; identity (cross-entropy, label smoothing 0.1) + batch-hard triplet
   (margin 0.3) losses.
2. Second forward pass of the same batch at a reduced token budget (keep ratio sampled per step
   from {0.5, 0.6, 0.7, 0.8, 0.9}, or fixed).
3. Same ID + triplet losses on the reduced pass, plus a distillation term
   `mean(1 - cos(bn_reduced, bn_full.detach()))` on the BNNeck embeddings, weight 1.0.

No second network, external teacher or extra data. Expected benefits if it works: (a) cheap
queries match a gallery indexed once at full quality (H1); (b) one set of weights serves every
budget; (c) possible regularisation when identities are scarce (H3).

Differences from closest work: vs SUReID, no separate teacher, many budgets, cross-budget
retrieval; vs SiT/ToCom, retrieval embeddings rather than logits and re-ID-specific evaluation;
vs two-model asymmetric retrieval, one network so spaces start aligned; vs SRE-Merge, training-time
alignment that can wrap any merging rule.

Risks: training-free merging may already preserve compatibility (then H1 has no problem to solve;
the baseline sweep reveals this); two passes roughly double cost per step; novelty moderate.

### 6.2 Model and token reduction (as implemented in `common.py`)

- `ReIDViT`: `timm.create_model(name, pretrained, num_classes=0, img_size=(256,128))`, default
  `vit_small_patch16_224` (dim 384, 12 blocks, 128 patch tokens + class token = 129).
  Class-token feature, then BNNeck (`BatchNorm1d`, bias frozen), then bias-free linear classifier
  (init std 0.001). `forward(x, mode, keep)` returns `(feat, bn, logits)`; `logits` is `None` in
  eval mode. Triplet uses `feat`; retrieval and distillation use `bn`.
- Normalisation mean/std taken from the timm model's `pretrained_cfg`.
- Reduction happens inside blocks with index 3, 6 and 9 (blocks 4, 7, 10), **after attention and
  before the MLP**. `keep` is the fraction of patch tokens kept at each of the three points:
  keep 0.7 leaves 45 patch tokens (46 total), keep 0.5 leaves 16 (17 total).
- Modes:
  - `none`: full tokens, uses timm's own fused block forward.
  - `merge`: ToMe-style bipartite soft matching on mean attention keys, size-weighted averaging,
    proportional attention (`log(size)` added to attention logits) in all later blocks. Own
    implementation with an EViT-style schedule, so numbers will not match official ToMe.
  - `cls_attn`: EViT-style, keep the patch tokens with the highest class-token attention (mean over
    heads). No fused "inattentive" token.
  - `random`: keep random patch tokens (control).
- `attention_with_maps` re-implements timm `Attention.forward` to expose attention maps and keys;
  checked equal to timm's output to 4e-8.
- `vit_gflops`: analytic estimate per block `4*N*D^2 + 2*N^2*D` (attention) `+ 2*N*D*hidden` (MLP)
  plus patch embedding. ViT-S at 256x128: 2.930 GFLOPs full, 1.921 at keep 0.7, 1.463 at keep 0.5.
  It is an estimate of the backbone, not a profiler measurement.

### 6.3 Training recipe (`train.py` defaults)

Input 256x128; augmentation: bicubic resize, horizontal flip, pad 10 + random crop, random erasing
p=0.5. PK sampling P=16, K=4 (batch 64), one epoch about one pass over the training images.
SGD, momentum 0.9, lr 0.008, weight decay 1e-4, 5 warm-up epochs then cosine, 120 epochs
(TransReID-like settings, without its stride-12 patches, SIE or JPM). Mixed precision on CUDA.
Evaluates every 20 epochs with full tokens. Writes `log.json` (per-epoch losses, mAP, Rank-1,
cumulative `train_hours`, `peak_vram_gb`, `gflops_full`) and `last.pth`
(`{"model", "args", "num_classes"}`).

Flags: `--id-fraction` (random subset of identities, seeded), `--train-mode/--train-keep` (reduce
tokens in the primary pass, for H4), `--sd-mode/--sd-keep/--sd-weight` (TBSD), `--model`,
`--no-pretrained`, `--seed`, `--workers`, `--no-amp`.

### 6.4 Evaluation (`evaluate.py`)

Standard Market-1501 protocol: gallery images with the query's identity **and** camera are ignored;
junk (pid -1) dropped; distractors (pid 0) kept. Features are L2-normalised BNNeck outputs, cosine
similarity. For each mode and keep ratio (default modes `merge cls_attn random`, keeps
`0.9 0.8 0.7 0.6 0.5`) it reports:

- `sym_mAP`, `sym_R1`: query and gallery both reduced.
- `asym_mAP`, `asym_R1`: gallery full-token (encoded once), query reduced.
- `q_cos`: mean cosine between reduced and full-token query embeddings.
- `gflops` (analytic), `final_tokens`, `img_s` (measured forward throughput; batch 64 on GPU, 8 on CPU).

Outputs a CSV and a two-panel PNG (symmetric, asymmetric; mAP against query-side GFLOPs).
`--cpu` runs on CPU for CPU throughput. Cross-dataset test: point `--data` at another
Market-format dataset.

Headline metrics for the paper: mAP/Rank-1 vs GFLOPs and latency (GPU and CPU); mAP vs training
GPU-hours; mAP vs fraction of identities (10/25/50/100%); mean and spread over 3 seeds.

## 7. Files generated so far

Experiment kit (delivered to the user as downloads in the chat; keep all five in one folder):

| File | Purpose |
| --- | --- |
| `common.py` | Dataset parsing (Market-1501 file names), PK sampler, transforms, `ReIDViT`, token reduction, analytic GFLOPs, triplet loss, feature extraction, mAP/CMC evaluation, throughput timing |
| `train.py` | Baseline training, low-data option, reduced-token training, token-budget self-distillation |
| `evaluate.py` | Token-reduction sweep with symmetric and asymmetric retrieval, CSV + plot |
| `make_fake_market.py` | Creates a tiny random Market-format dataset to check the install |
| `README.md` | Setup, tonight's run order, what to look at, sanity checks, limitations |

Report (cloud document, not a local file): "Person Re-ID under limited resources: Iteration 1
report", https://claude.ai/artifact/AsT8BcEmxPPYbCMvi28h7E . Sections: Summary; Scope and
constraints; State of the art; Literature matrix; Gap analysis; Proposed direction; Experiment plan
and status; References. Claude Code cannot edit it directly; its content is reproduced in sections
4 to 6 and 9 to 11 of this file. It contains one open comment asking whether ViT-B should replace
ViT-S as the main backbone.

This file: `PROJECT_CONTEXT.md`.

## 8. Current progress

Done:

- Six-week first-iteration workflow defined (section 11).
- Targeted literature search, 21-work matrix, gap analysis, proposed direction.
- Iteration 1 report written.
- Experiment kit written and checked:
  - manual attention path equals timm's attention (max diff 4e-8);
  - token counts and GFLOPs per mode as expected;
  - merge conserves token mass and total size;
  - mAP/CMC verified on a hand-made case (expected AP 58.33, Rank-1 0; got exactly that);
  - `train.py` (baseline, TBSD, reduced-token, low-data) and `evaluate.py` ran end to end on CPU
    with the synthetic dataset.

Not done:

- **No run on Market-1501, no run on a GPU, no real accuracy numbers.** As of this file the user
  had not reported starting any run or having Market-1501 downloaded.
- No full paper reads; matrix entries are abstract-level.
- No full novelty check.

## 9. Unresolved problems and open questions

1. The kit is untested on real data. The first Market-1501 run is its first real test. Baseline
   ViT-S mAP is **expected** (not verified) to land in the 80s after 120 epochs; far below that
   indicates a bug or setup issue, not a finding.
2. Possible first-run issues to watch: PyTorch/CUDA build for the 5080; `torch.load` defaults
   (`weights_only`) across versions; Windows data-loader worker behaviour; RAM pressure with
   16 GB; timm API differences (`_pos_embed`, `patch_drop`, `norm_pre`, attention attributes)
   if the installed timm differs much from 1.0.31.
3. Only Market-1501-style file names are parsed. MSMT17, CUHK03 and Occluded-REID need loaders.
4. OSNet and CLIP-ReID baselines are not implemented in the kit.
5. In the synthetic smoke test the distillation loss was near zero (untrained network on noise
   images gives nearly identical embeddings at all budgets). It is non-zero with gradients on
   random input (0.03), but its behaviour on real data is unknown.
6. TBSD costs about two forward passes per step. Fair comparison needs an equal GPU-hour baseline
   (e.g. baseline trained for twice the epochs).
7. `merge` differs from official ToMe (schedule and details); `cls_attn` omits EViT's fused token.
   For the paper, compare against official implementations too.
8. Novelty is moderate. Whether H1 is a real problem is unknown until the baseline sweep is run.
9. Open questions for the user/instructor: ViT-S or ViT-B as main backbone; target venue (affects
   dataset policy on DukeMTMC derivatives); operating system; whether Market-1501 is downloaded.
10. arXiv IDs marked "from memory" and all abstract-level claims need verification before citing.

## 10. Remaining tasks in execution order

### A. Before the instructor meeting (deadline about Sun 2026-11-15 10:24 Dhaka)

1. Confirm OS, install PyTorch (CUDA build for the 5080), `pip install timm pillow numpy matplotlib`.
2. Get Market-1501 (folder with `bounding_box_train/`, `query/`, `bounding_box_test/`).
3. Check the install:
   `python make_fake_market.py fake_market` then
   `python train.py --data fake_market --out runs/smoke --epochs 1 --P 4 --K 2 --eval-every 1`
4. Baseline: `python train.py --data DATA --out runs/base_s0 --seed 0`
5. Sweep: `python evaluate.py --data DATA --ckpt runs/base_s0/last.pth --out runs/base_s0/sweep.csv`
6. TBSD: `python train.py --data DATA --out runs/tbsd_s0 --seed 0 --sd-mode merge --sd-keep random`
7. Sweep: `python evaluate.py --data DATA --ckpt runs/tbsd_s0/last.pth --out runs/tbsd_s0/sweep.csv`
8. If time: low-data pair
   `python train.py --data DATA --out runs/base_id25_s0 --id-fraction 0.25` and
   `python train.py --data DATA --out runs/tbsd_id25_s0 --id-fraction 0.25 --sd-mode merge --sd-keep random`
9. Interpret and add results to the report:
   - asymmetric mAP well below symmetric mAP on the baseline means H1 is a real problem;
   - TBSD closing that gap without losing full-token mAP means the direction passes its first test;
   - `merge`/`cls_attn` no better than `random` means selection criteria are the gap to work on;
   - compare `train_hours` between baseline and TBSD.
10. Show the instructor: the report, the kit, and whichever curves finished.

### B. After the meeting (rest of the six-week iteration)

11. Fix any bugs found; add a short results section and plots to the report.
12. Repeat runs with 3 seeds; add an equal-GPU-hour baseline for TBSD.
13. Add loaders for MSMT17, CUHK03, Occluded-REID; run cross-dataset sweeps (H2).
14. Add OSNet and ViT-B references; later a ported CLIP-ReID baseline.
15. Low-data runs at 10/25/50% of identities, 3 seeds (H3).
16. Reduced-token training runs for training-cost curves (H4, side benefit).
17. CPU latency sweep with `evaluate.py --cpu`.
18. Literature matrix to 30 to 40 papers with full reads; columns: backbone and pretraining data,
    setting, datasets, mAP/Rank-1, FLOPs/latency, training GPU-hours, external teacher or auxiliary
    model, code availability, stated limitations.
19. Full novelty check: arXiv and Google Scholar, including papers citing AMG, SUReID, SRE-Merge,
    SiT, ToCom; search specifically for elastic/any-budget token models in retrieval or re-ID.
20. Decide the method; write the one-page proposal: problem, diagnostic evidence, method, three
    closest papers and differences, fixed resource budget ("one consumer GPU, under N GPU-hours,
    no extra pretraining data").

## 11. Six-week first-iteration workflow (reference)

- **Week 1:** scope, environment, datasets, logging template.
- **Weeks 2 to 3:** literature matrix (30 to 40 papers, 2023 to 2026; CVPR, ICCV, ECCV, NeurIPS,
  ICML, AAAI, TPAMI, TIP).
- **Weeks 3 to 4:** baselines (OSNet, ViT-S, TransReID ViT-B, CLIP-ReID) on Market-1501 and
  MSMT17, 3 seeds; training-free pruning/merging at several keep ratios; accuracy-vs-cost plots.
- **Week 5:** diagnostics for H1 to H4, a few GPU-hours each.
- **Week 6:** gap statement, novelty check, one-page proposal.
- **Deliverables:** literature matrix; baseline table and trade-off curves with cost logs;
  diagnostics for at least two hypotheses; one-page proposal.

## 12. Notes for the next Claude Code session

- Start by asking the user what has been run since this file was written and for any `log.json`
  and sweep CSV files.
- Keep the honesty rules in section 2: label unverified numbers, separate "ran" from "wrote".
- Look up current documentation rather than trusting memory for PyTorch/timm details on the 5080.
- The user's instructor values results under limited resources over raw accuracy; keep every
  experiment and claim tied to cost.
