# Iteration 1: supervisor brief

**Budget-compatible queries for ViT person re-identification.** Prepared 2026-11-22.
Full report: [REPORT_ITERATION1.md](REPORT_ITERATION1.md). Audit and full disclosure list:
[SUPERVISOR_REVIEW_AUDIT.md](SUPERVISOR_REVIEW_AUDIT.md).

## In three sentences

We train one ViT-S so that a cheap query (smaller image or merged tokens) can be matched against a gallery
indexed once at full cost. On Market-1501 it keeps 83.0 mAP at 38 % of the query compute, where a
similar-training-budget baseline with the best tested post-hoc correction reaches 79.2. It costs 1.2 mAP at full
compute and is **not clearly better than a prior-work-style frozen-gallery student**, so the contribution so far is a
credited combination plus a cost-accounted evaluation, not a new method.

## 1. Research problem

Re-ID galleries are indexed once, while queries arrive continuously and may run on weak devices. Can one model
make queries cheaper without re-indexing the gallery and without losing much accuracy? The thesis is accuracy
per unit of compute, labels and training time.

## 2. Preliminary gap

In the papers read we did not find one ViT person re-ID model that uses **both input resolution and token
merging** as query budgets, stays compatible with its own full-cost gallery, and reports **training cost,
full-cost loss and measured latency**.

The search was not exhaustive. This is a gap of application and evaluation. These claims are **not** open:
- one network with compatible budgets (SFSC, PrunNet);
- one ViT with cheap queries against its full gallery ([MatFormer](https://arxiv.org/abs/2310.07707), NeurIPS 2024);
- resolution-dependent compute with compatible embeddings ([BTNet](https://arxiv.org/abs/2310.09563), BMVC 2023, faces);
- per-query budgets from gallery distance in re-ID ([DaRe](https://arxiv.org/abs/1805.08805), CVPR 2018).

## 3. Six closest papers

| Paper | What it does | How we differ |
| --- | --- | --- |
| [SFSC](https://arxiv.org/abs/2206.08289), Wu et al., CVPR 2023 | Compatible re-ID sub-models of several CNN widths | We reduce the input, not the width |
| [PrunNet](https://arxiv.org/abs/2504.11879), Sun et al., CVPR 2025 | Compatible pruned sub-networks of one network, including ViT-S | We reduce the input, not the weights, and measure latency |
| [D3still](https://openaccess.thecvf.com/content/CVPR2024/html/Xie_D3still_Decoupled_Differential_Distillation_for_Asymmetric_Image_Retrieval_CVPR_2024_paper.html), Xie et al., CVPR 2024 | Small query network distilled against a large gallery network | We use one network. Our alignment loss is in the family of its feature term |
| [Suma and Tolias](https://arxiv.org/abs/2210.05463), WACV 2023 | Frozen gallery model and a low-resolution student | We use one set of weights. An adaptation of theirs is our strongest baseline |
| [ResFormer](https://arxiv.org/abs/2212.00776), Tian et al., CVPR 2023 | Multi-resolution ViT training, high resolution teaches low | We apply the mechanism to retrieval compatibility |
| [VPFA](https://proceedings.mlr.press/v306/liu26fr.html), Liu et al., ICML 2026 | Post-hoc MLP maps low-resolution features to high-resolution ones | We train the backbone. An adaptation of theirs is a baseline |

### Results published by those papers (their protocols; not comparable with ours or with each other)

| Paper | Protocol | Reported |
| --- | --- | --- |
| D3still | MSMT17, ResNet-18 at 64x64 query, ResNet-101 gallery | 18.64 mAP (gallery model alone 59.81) |
| PrunNet | VeRi-776 vehicles, self-test | 67.82 mAP dense, 64.45 at 25 % capacity |
| Suma and Tolias | CUB200, query 224, database 448 | 40.8 mAP |
| VPFA | MLR-Market-1501, TransReID features | 94.1 Rank-1 (its baseline 90.3) |
| SFSC, ResFormer | CNN widths; ImageNet classification | Not quoted, no matching setting |

## 4. Proposed methodology (`sp_mixed`)

Each training step runs the same batch twice through shared weights: once at full cost, once at a random budget
b (224x112, 192x96, 160x80, 128x64, or ToMe-style merging of r = 3, 5, 7, 9, 11 tokens per block).

`L = CE(full) + Triplet(full) + CE(b) + Triplet(b) + mean(1 − cos(e_b, stopgrad(e_full)))`

The classifier is shared. At test time the gallery is full cost and each query uses any budget. The design
follows the instructor's suggestion of self-distillation and token reduction. Every component is prior work.

## 5. Dataset and protocol

- [Market-1501](https://openaccess.thecvf.com/content_iccv_2015/html/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.html)
  (Zheng et al., ICCV 2015): 12,936 training images of 751 identities; 3,368 queries; 15,913 evaluated gallery
  images. Train and test identities are disjoint.
- ViT-S/16 at 256x128, ImageNet pretrained, RTX 5080, 3 seeds, last epoch, cheap query vs full gallery.
- 50 training runs, 13.38 GPU-hours in total.
- Differences are judged against 2 std in two ways, combined and seed-paired. Clear = passes both, tie = passes
  neither, borderline = passes one. This is a heuristic fixed after seeing the results, not a significance test.

## 6. Benchmark comparison: our own runs only

mAP (%), mean ± std over 3 seeds. "-style" and "-inspired" mean our adaptation, not the authors' method.

| Method | GPU-h | Full, 2.93 G | 192x96, 1.62 G | Merge r=11, 1.41 G | 160x80, 1.12 G | 128x64, 0.72 G |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline, 120 epochs | 0.22 | 84.45 ± 0.49 | 81.38 ± 0.36 | 77.45 ± 1.39 | 75.81 ± 0.43 | 60.13 ± 2.68 |
| Baseline, 216 epochs | 0.38 | 85.11 ± 0.15 | 82.06 ± 0.19 | 78.97 ± 0.12 | 75.72 ± 0.51 | 57.45 ± 2.34 |
| + linear map (λ = 1) | 0.38 + fit | 85.11 ± 0.15 | 82.80 ± 0.23 | 79.70 ± 0.13 | 78.96 ± 0.18 | 68.88 ± 0.70 |
| + linear map (λ tuned) | 0.38 + fit | 85.11 ± 0.15 | 82.96 ± 0.22 | not run | 79.16 ± 0.21 | 68.84 ± 0.69 |
| + MLP, VPFA-inspired | 0.38 + fit | 85.11 ± 0.15 | 82.77 ± 0.31 | not run | 78.61 ± 0.33 | 66.88 ± 0.22 |
| Resolution-only training | 0.35 | 83.09 ± 0.25 | 82.85 ± 0.17 | 79.59 ± 0.37 | 82.36 ± 0.19 | 80.96 ± 0.20 |
| Merge-only training | 0.44 | 84.27 ± 0.07 | 82.68 ± 0.10 | 82.90 ± 0.13 | 79.89 ± 0.18 | 73.04 ± 0.37 |
| Frozen-gallery student, Suma-Tolias-style | 0.46 | 84.45 ± 0.49 | 83.68 ± 0.49 | 81.50 ± 0.54 | 82.78 ± 0.55 | 80.48 ± 0.69 |
| **`sp_mixed` (proposed)** | 0.40 | 83.95 ± 0.15 | **83.60 ± 0.14** | **82.38 ± 0.27** | **83.01 ± 0.15** | **81.50 ± 0.22** |

Evidence: `runs/base_s*/`, `runs/mechanism.csv`, `runs/base_e216_s*/mech.csv`, `runs/sp_*_s*/mech.csv`,
`runs/fg_mixed_s*/mech.csv`, `runs/nonlinear_map/nonlinear_map.csv`; GPU-hours from `runs/*/log.json`; full-cost
rows from `runs/*/sweep.csv`.

## 7. Findings

| | Finding | Evidence |
| --- | --- | --- |
| + | Against the best tested correction of the 216-epoch baseline: +3.85 at 1.12 G and +12.61 at 0.72 G; +0.64 at 1.62 G is borderline | `runs/nonlinear_map/agg.md` |
| + | One model works on both knobs, while single-knob models carry over only partly | `runs/sp_mechanism_agg.md` |
| + | 160x80 is 2.67x faster at GPU batch 256 and 1.69x on CPU for one image | `runs/latency_extra.csv` |
| + | Alignment adds about 1.4 mAP (tested on the older 3-stage schedule only) | `runs/tbsd_nosd_s*/sweep.csv` |
| − | Not clearly better than the frozen-gallery student: ties at four cheap budgets, borderline ahead at two (+0.88 at r = 11, +1.02 at 128x64) | `runs/fg_mechanism_agg.md` |
| − | 1.16 worse than the 216-epoch baseline at full cost, borderline worse at 2.22 G (−0.43), tie at 1.94 G | `runs/base_e216_s*/mech.csv` |
| − | No single-image GPU speed-up; gradual merging is 2.3x slower there | `runs/latency_extra.csv` |
| − | No low-data benefit at 10, 25 or 50 % of identities (tested on the earlier probe) | `runs/iter2_summary.csv` |
| − | Per-query cascade is below fixed budgets at equal cost | `runs/adaptive_agg.md` |
| − | Class-attention token selection is below random (clear in most models, a tie on the baselines at keep 0.7); unexplained | `runs/mechanism.csv` |
| − | A non-linear post-hoc map does not improve on a linear one | `runs/nonlinear_map/agg.md` |

## 8. Limitations

- One dataset, whose crops are natively 128x64, so the low-resolution results are probably flattered.
- No validation split: variants and budgets were chosen on test results.
- Missing baselines: single-pass multi-resolution augmentation, and a small model at equal GFLOPs.
- `sp_mixed` was never run without alignment or with separate classifiers.
- Three seeds and many comparisons. ViT-B has one seed and was not trained with `sp_mixed`.
- GFLOPs are analytic. Latency comes from one machine.

## 9. What I need to disclose

1. **Merging is not official ToMe.** An earlier check claimed identity but tested the wrong code path. The
   numbers are valid for our variant, and the effect on mAP is unknown.
2. **Earlier drafts overclaimed.** The "+1 mAP over the frozen-gallery student" is borderline (it depends
   on how std is defined), and the per-query "gap" was already published (DaRe).
3. **The test set guided design choices**, including a look at test mAP during the smoke test of the MLP
   baseline.
4. **Re-implementations are adaptations.** No official code of a compared method was run.
5. **AI assistance.** An AI assistant (Claude Code) wrote the code, ran the experiments, processed the papers,
   some only in part or through summaries, and drafted these documents. I still need to read the closest papers
   myself.
6. **The mechanism explanation is a hypothesis.** The diagnostic cannot separate information loss from
   incompatibility.

## 10. Proposed iteration 2

1. **Validity first:** validation split, the two missing baselines, the λ = 0 ablation, a student of the
   216-epoch teacher, more seeds, corrected merging.
2. **Generality:** CUHK03-NP or MSMT17 (dataset access needed).
3. **Mechanism, only if 1 and 2 hold:** escalation that reuses the cheap pass
   ([ThinkingViT](https://arxiv.org/abs/2507.10800), CVPR 2026) under a gallery-compatibility constraint, after
   reading DaRe, MatFormer and HashReID in full.

**Decisions requested:** Is an empirical-study framing acceptable, or is a new mechanism required? Which second
dataset? Is ViT-S acceptable as the main backbone?
