# Likely supervisor questions, with answers (iteration 1)

Prepared 2026-11-17; revised 2026-11-22 to match the audited `REPORT_ITERATION1.md`. Answers use our own measured
numbers (Market-1501, ViT-S/16, mean ± std over 3 seeds unless stated). Explanations from the ground up:
`STUDY_GUIDE.md`. Every known issue: `SUPERVISOR_REVIEW_AUDIT.md`.
Answer honestly; most of the strength of this iteration is in what it measured carefully, including what
did not work. Only say "I read" or "I did" where that is true of you (see B6 and H3).

Three words are used precisely below (explained in D5): **clear**, **borderline**, **tie**.

**If you remember four sentences:**
1. One ViT-S gives 83.01 mAP at 38 % of the query compute against its own full-cost gallery; the best
   post-hoc-corrected baseline of similar training budget gives 79.16.
2. It pays 1.16 mAP at full cost.
3. It is not clearly better than a prior-work-style frozen-gallery student.
4. The components are prior work; the contribution is the combination and the cost-accounted evaluation.

---

## A. Problem and motivation

**A1. What problem are you solving, in one sentence?**
How to make the *query* side of a ViT person re-ID system cheaper (smaller input or fewer tokens) while its
embedding stays directly comparable with a gallery that was computed once at full cost.

**A2. Why does that matter?**
In deployment the gallery is indexed once on a server, but queries arrive continuously and may be computed on
edge devices. Re-computing the gallery for every new model is expensive (re-indexing). So we want a cheap query
encoder that is *compatible* with the existing full-cost gallery. This fits the project thesis: accuracy per
unit of compute, labels and training time. Of these, this iteration supports only the compute part.

**A3. Why not just use a smaller network for the query?**
We have not shown that a smaller network would be worse. A small model (ViT-Tiny, OSNet) at equal GFLOPs is a
missing baseline. The known route is asymmetric retrieval with two networks (D3still, CVPR 2024; Suma and
Tolias, WACV 2023). We study one ViT whose cost changes per query without loading another model. Our adaptation
of the two-network idea, the frozen-gallery student, is about as accurate as our method (D8).

**A4. Why ViT and not a CNN?**
A ViT's cost is set by its number of tokens, and both knobs (resolution, merging) act on the token count without
changing the weights. ViT-S/16 also gives a strong baseline (84.45 mAP) at 2.93 GFLOPs. We did not compare with
a CNN.

## B. Novelty and related work

**B1. What is new here?**
The combination and the evaluation, not the components. What we did not find in the papers read is one ViT
re-ID model that uses input resolution *and* token merging as query budgets, stays compatible with its own
full-cost gallery, and reports training cost, full-cost loss and measured latency. The search was not
exhaustive. The components are prior work:
- shared classifier across budgets: BCT, SFSC;
- alignment to the full embedding: the same family as D3still's feature term and Suma and Tolias's absolute
  term (the functional forms differ);
- multi-resolution ViT training with a detached high-resolution target: ResFormer;
- token merging: ToMe (our variant, D6).

Do **not** claim as new: one network with compatible budgets (SFSC, PrunNet), one ViT with cheap queries against
its full gallery (MatFormer), resolution-dependent compute with compatible embeddings (BTNet), or per-query
budgets in re-ID (DaRe).

**B2. How is this different from SFSC and PrunNet?**
They make the weights cheaper; we make the input cheaper. SFSC (CVPR 2023) slices CNN channel width. PrunNet
(CVPR 2025) prunes weights, with ViT-S among its backbones and Market-1501 in an appendix with ResNet-18. Sliced
widths also run faster on ordinary hardware; PrunNet's authors say their unstructured sparsity is hard to
accelerate. We re-implemented neither, so we cannot say which is more accurate.

**B3. Isn't low-resolution query matching just cross-resolution re-ID?**
The geometry is the same and the goal differs. There, low resolution is a degradation, and in the method we
processed in full (RFD) it is upsampled, so no compute is saved. Here it is a choice that saves compute. One
honest caveat: Market-1501 crops are natively 128x64, so on this dataset our "reduction" mostly undoes the
upsampling.

**B4. Isn't your training just ResFormer?**
The low-resolution half is mechanically close, and we credit it. ResFormer is evaluated on classification and
other non-retrieval tasks; we found no cross-resolution matching in it. We add the retrieval losses, the merging
knob and asymmetric retrieval evaluation.

**B5. Which 2025-2026 work did you check?**
- Processed in full through a summarising tool: PrunNet (CVPR 2025), OSKT (ICCV 2025), ThinkingViT (CVPR 2026),
  VPFA (ICML 2026).
- Abstract level only: ACD (CVPR 2026 Findings; also its code README), TE-VMamba (CVPR 2026), SRE-Merge (ICML
  2026), QuARI (NeurIPS 2025). These are never used to claim a paper lacks something.

Our VPFA-inspired map was tested as a baseline. It did not improve on a linear map.

**B6. How thorough was the literature search?**
Targeted, not systematic. The papers were processed by an AI assistant:
- whole text directly: D3still, SUReID;
- whole text through a summarising tool: SFSC, Suma and Tolias, PrunNet, ResFormer, RFD, ThinkingViT, VPFA,
  OSKT;
- relevant sections only: DaRe, MatFormer, BTNet;
- official code read, not run: ToMe, EViT.

Say which of these you have read yourself. DaRe, MatFormer and BTNet were missed until the audit of 2026-11-22,
which shows the search can miss close work.

**B7. Didn't DaRe already do per-query budgets in re-ID?**
Yes. DaRe (CVPR 2018) exits queries early by nearest-gallery distance on Market-1501, with shared computation
between stages. An earlier draft listed this as a gap; that claim is withdrawn. Our cascade applies the same
signal to our knobs, and it lost to fixed budgets.

## C. Method

**C1. Describe the method precisely.**
Each training step runs a batch twice through the same ViT: once at full cost and once at a cheap budget b
drawn at random (a low resolution in {224x112, 192x96, 160x80, 128x64} or gradual ToMe-style merging with
r in {3, 5, 7, 9, 11}). Loss:
`L = CE(full) + Tri(full) + CE(b) + Tri(b) + λ · mean(1 − cos(e_b, sg(e_full)))`, λ = 1, sharing the classifier.
At test time the gallery is full cost; a query uses whatever budget the device affords.

**C2. Why the stop-gradient?**
The full embedding is the reference, because the gallery uses it. Stop-gradient blocks the direct pull of the
full embedding towards the cheap one. The weights are shared, so the term still changes the full embedding
indirectly. We did not test training without stop-gradient.

**C3. Why share the classifier?**
So both passes place each identity on the same side of the same class boundaries (the BCT idea). We did not
isolate its effect. What we measured, on the older 3-stage schedule: two-pass training with a shared classifier
and no alignment reaches 80.56 at keep 0.5, and adding alignment gives 81.99. "Two-pass training does most of
the work" is supported; "the shared classifier does most of the work" is not.

**C4. Why mix two knobs instead of one?**
Single-knob models carry over only partly. The merge-trained model gives 73.04 at 128x64 and the
resolution-trained one 79.59 at merge r = 11; the mixed model gives 81.50 and 82.38. The price: under merging it
is at or slightly below the merge specialist (−0.52 at r = 11, a tie; −0.37 at r = 7, clear).

**C5. How did you choose λ, the budgets, the learning rate?**
λ = 1 and the training recipe (lr 0.008, 120 epochs, standard re-ID losses) were not tuned. The budget sets span
about 25-85 % of full cost. The variants and budget sets were chosen after seeing test results (H2).

## D. Experiments and fairness

**D1. Is the comparison fair?**
Partly. In its favour:
- the baseline gets a similar training budget: 216 epochs, 0.36-0.40 GPU-h (mean 0.38), against our 0.35-0.44
  (mean 0.40);
- that baseline also gets linear and non-linear post-hoc maps, and we compare with the best one picked on test;
- 3 seeds per main configuration;
- a frozen-gallery student in the style of prior work.

Against it:
- no single-pass multi-resolution-augmentation baseline and no small-model baseline;
- no validation split;
- unequal budgets: merge-only +14 % GPU time, student plus teacher +21 %, and timings of identical runs vary up
  to about 25 %.

**D2. What are the headline numbers?**
Asymmetric mAP, our own runs:

| Query | GFLOPs | 216-epoch baseline, best map | Frozen-gallery student | `sp_mixed` | vs baseline | vs student |
| --- | --- | --- | --- | --- | --- | --- |
| full | 2.93 | 85.11 | 84.45 | 83.95 | −1.16, clear | −0.50, tie |
| 192x96 | 1.62 | 82.96 | 83.68 | 83.60 | +0.64, borderline | −0.07, tie |
| merge r = 11 | 1.41 | 79.70 | 81.50 | 82.38 | +2.68, clear | +0.88, borderline |
| 160x80 | 1.12 | 79.16 | 82.78 | 83.01 | +3.85, clear | +0.23, tie |
| 128x64 | 0.72 | 68.88 | 80.48 | 81.50 | +12.61, clear | +1.02, borderline |

Non-linear maps were not tested on merging queries. At 224x112 the baseline is ahead (borderline).

**D3. So your method is worse at full cost?**
Yes: 1.16 mAP below the 216-epoch baseline, a clear difference. Against the 120-epoch baseline it is −0.50, a
tie. The student keeps its teacher's 84.45, which is not separated from either, but needs two sets of weights.

**D4. Why only Market-1501?**
Access and time: CUHK03 and MSMT17 need a sign-up or request. It is a main limitation, because Market-1501 crops
are natively about 128x64, so a 128x64 query loses little real information.

**D5. Why are three seeds enough?**
They are the minimum to see a spread, and not enough for close calls. Each difference is judged against two
standard deviations in two ways:
- combined, `sqrt(sd_a² + sd_b²)`;
- paired, the std of the per-seed differences.

Clear passes both, a tie passes neither, borderline passes one. Example: +0.88 over the student at r = 11 is 1.4
combined std and 2.8 paired std, so borderline. This is a heuristic, fixed after the results were known, with no
correction for many comparisons. Borderline results need more seeds before they are claimed.

**D6. How did you check your token-reduction code is correct?**
- mAP / Rank-1 code: equal to a brute-force implementation on 200 random cases.
- EViT top-k selection: equal to a transcription of the official function.
- **Merging: not identical to official ToMe.** An earlier check reported a difference of 0 but tested the wrong
  code path (no class token). On random input, one merge step at r = 11 gives the same tokens as official ToMe
  in 10 of 64 samples. Our numbers are valid for our variant; the effect on mAP is unknown.

**D7. Why ViT-S and not ViT-B?**
ViT-B (one seed) reached 86.38 at 11.34 GFLOPs: 3.9x the cost of ViT-S for +1.9 mAP. Its reduced version (82.37
at 5.65 GFLOPs) is below a plain ViT-S at 2.93 (84.45). `sp_mixed` was not run on ViT-B.

**D8. Does your method beat the closest prior-work scheme?**
Not clearly. Against our adaptation of Suma and Tolias's frozen-gallery student it ties at four of the six cheap
trained budgets evaluated. It is ahead in all three seeds at r = 11 (+0.88) and 128x64 (+1.02), but both are
borderline. The practical difference is one set of weights against two. The student is an adaptation (different
loss form, augmentations, backbone, one student for all budgets), so this says nothing about their published
method.

## E. Efficiency

**E1. Are GFLOPs savings real speed-ups?**
Not always; we measured. For one image on the GPU every resolution costs about 6 ms, and gradual merging is 2.3x
*slower* than full tokens. At 160x80 we get 2.67x at GPU batch 256, 1.69x at CPU batch 1 and roughly 2.35x at
CPU batch 8. The last is rough: that regime varied 17-46 % between two sessions.

**E2. Then why keep token merging at all?**
On this dataset it is hard to justify. For `sp_mixed`, resolution is ahead at about 1.41 GFLOPs (about 83.4
interpolated vs 82.38) and equal at 1.94, and it is mostly faster. Merging may matter where downsizing destroys
real detail; that is untested.

**E3. What is the training cost?**
`sp_mixed` 0.35-0.44 GPU-h per run, peak allocated VRAM 3.17 GB; 120-epoch baseline 0.22 h, 1.81 GB;
frozen-gallery student 0.23-0.24 h on top of its 0.22 h teacher, 1.83 GB.

## F. Negative results

**F1. What did not work?**
1. No clear win over the frozen-gallery student.
2. Worse than the similar-budget baseline at full cost.
3. Our first method (3-stage TBSD) did not beat baseline + linear map at low resolution, though it did under
   merging. Its apparent +13 came from a weak schedule.
4. No low-data benefit at 10/25/50 % of identities.
5. Class-attention token selection was below random.
6. Merging gave no single-image GPU speed-up.
7. The per-query cascade lost to fixed budgets.
8. The non-linear post-hoc map did not improve on a linear one.

**F2. Why did the cascade not work?**
An escalated query pays the cheap pass *and* the full pass, while one intermediate resolution for every query
already keeps most of the accuracy, because the fixed-budget curve is concave. The routing signal itself selects
better than random, as in DaRe, and is within 0.15 mAP of a drift oracle. That oracle is not an upper bound: it
knows which embeddings moved, not which queries would gain. Reusing the cheap pass is untested.

**F3. Why is class-attention selection worse than random?**
Unexplained. It is lower on average in every model evaluated, clearly in most, but a tie on the baselines at
keep 0.7. EViT's exact fusion does not remove it, and a shared embedding offset does not explain it. Untested
hypothesis: attention concentrates on a few regions, so the kept tokens cover the body unevenly. We never
trained a model with EViT, so this says nothing about EViT as published.

**F4. Why does two-pass training cost full-cost accuracy?**
Not established. Per-budget BatchNorm statistics are ruled out. Higher hardest-negative similarity is not a
sufficient explanation: the model without alignment has the deficit with lower hardest-negative similarity.
Gradient conflict between passes (handled by SFSC and PrunNet) is an untested hypothesis.

**F5. Why does the method work at low resolution?**
We know what changes, not why. The cheap embeddings become closer to the full ones (`q_cos` 0.953 vs 0.770 for
the baseline at 160x80) and more discriminative on their own. Our diagnostic cannot say whether the baseline
fails through lost information or through incompatibility, because same-budget retrieval also degrades the
gallery.

## G. Future work

**G1. What is the plan for iteration 2?**
1. Validity: a validation split, multi-resolution-augmentation and small-model baselines, `sp_mixed` without
   alignment, a student of the 216-epoch teacher, more seeds for borderline results, merging aligned with ToMe.
2. Generality: a second dataset with higher native resolution (CUHK03-NP or MSMT17).
3. Mechanism, only if 1 and 2 hold: escalation that reuses the cheap pass, after reading DaRe, MatFormer and
   HashReID in full.
4. One published multi-budget method re-implemented (SFSC- or PrunNet-style).

**G2. If you had one more week, what single experiment would you run?**
Single-pass multi-resolution-augmentation training at the same GPU budget, selected on a validation split. It is
the simplest competitor: if it matches `sp_mixed`, two-pass training is unnecessary. With dataset access, the
main table on CUHK03-NP comes next.

## H. Reproducibility and integrity

**H1. Can someone reproduce this?**
The code, per-run logs and CSVs are in the repository; confirm that the 21-22 November files are committed
before saying "all". Checkpoints are local. There were 50 real training runs, 13.38 GPU-hours in total
(evaluation, map fitting and latency not included). Re-training has not been tested for reproducing the numbers.
Three runs were resumed and are not bit-identical to uninterrupted runs. Use the commands in `STUDY_GUIDE.md`
Section 15: evaluation commands copied from the older logs overwrite the original CSVs.

**H2. Did you tune anything on the test set?**
Yes, indirectly. No numeric hyperparameter of the method was searched and the last epoch was always used, but
variants and budget sets were chosen after seeing test results. The smoke test of the MLP baseline also printed
test mAP before its candidate list was final. There was no validation split for training or design. This is
test-set reuse; a split is the first item of iteration 2.

**H3. How much of this did you do yourself, and how much did the AI do?**
Answer this in your own words. The repository's record is that an AI assistant (Claude Code) wrote the code, ran
and monitored the experiments, processed the papers, and drafted the documents, under your instructions and
priorities. Add what you personally read, checked, decided or re-ran, and nothing more. Check your institution's
rules on AI assistance.

**H4. Were there errors in earlier versions?**
Yes, found in an audit on 2026-11-22 and corrected in the report. The most important: merging was wrongly
described as identical to official ToMe, the per-query gap was prior work, and several differences were stated
more firmly than three seeds allow, in both directions. The list is in `SUPERVISOR_REVIEW_AUDIT.md`
(Sections 3 and 5).
