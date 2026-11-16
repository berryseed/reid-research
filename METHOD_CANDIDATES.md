# Candidate methodology directions (preliminary, 2026-11-15)

Status: hypotheses. None is a claimed contribution. Each is tied to evidence already measured
(`runs/mechanism.csv`, `runs/iter2_summary.csv`) and to prior work in `LITERATURE_MATRIX.md`.
TBSD (two passes, 3-stage merge, cosine alignment) is the exploratory probe that produced the evidence.

Notation: `f(x; b)` = L2-normalised BNNeck embedding of image `x` computed at budget `b` (b = full, a token
schedule, or an input resolution). Gallery is always `f(x; full)`. `sg` = stop-gradient.
Asymmetric mAP = queries at budget b against the full gallery. All numbers: Market-1501, ViT-S/16, 3 seeds.

## Evidence the candidates rest on
| # | Observation | Numbers |
| --- | --- | --- |
| E1 | An ordinary model's embedding moves when tokens are reduced; the movement is per-image, not a shared shift | offset fraction 0.03-0.16; mean/affine correction +0.1 to +1.5 mAP |
| E2 | A closed-form linear map recovers part of it | 3-stage merge 68.55 -> 72.70; gradual merge 77.45 -> 78.65 |
| E3 | The schedule matters as much as the rule | 3-stage 68.55 @1.46 G vs gradual ToMe 77.45 @1.41 G |
| E4 | Early reduction is the expensive part | halve at block 4 / 7 / 10 only: -3.8 / -1.2 / -0.2 mAP |
| E5 | Low-resolution queries need no training and beat untrained merging | 192x96: 81.38 @1.62 G; 160x80: 75.81 @1.12 G. Caveat: Market-1501 is natively 128x64 |
| E6 | Two-pass training gives robustness that transfers to a knob it never saw | TBSD at 160x80: 78.26 vs baseline 75.81 |
| E7 | Two-pass training costs full-token accuracy; per-budget BN statistics are not the cause | 83.95 vs 84.45 (120 ep), vs 85.11 at equal GPU-hours; nostat ablation 83.83 |
| E8 | The alignment term is a minor contributor; sharing the classifier across passes does most of the work | 80.56 without it vs 81.98 with it, at keep 0.5 |
| E9 | Class-attention top-k (exact EViT fusion included) is worse than random for cross-budget matching | 72.93 ± 3.28 vs 78.06 ± 0.63 at keep 0.7 |
| E10 | Against the best training-free option at equal cost, TBSD's gain is about +2.4 mAP, not +13.4 | 81.99 vs ~79.6 (interpolated low-res) at 1.46 G |

## C1. Retrofit query encoder: frozen gallery model, trained cheap-query copy
- **Problem addressed:** E7 and E10. Two-pass training changes the full-token model (lower gallery quality,
  gallery must be re-indexed) to buy a modest gain.
- **Intuition:** the gallery side never needs to change. Keep the best full-token model frozen as the gallery
  encoder and as the teacher. Train only a query-side encoder (initialised from the same weights) that runs at
  a reduced budget and lands on the frozen model's embedding. Full-token accuracy is preserved by
  construction; an existing deployed index gains cheap queries without re-indexing.
- **Formulation:** teacher `T` frozen, student `S` initialised as `T`. Per step, sample budget `b`:
  `L = mean_i (1 - cos(S(x_i; b), sg T(x_i; full)))` (the AML / D3still feature term), optionally
  `+ CE(W_T · S(x_i; b), y_i)` with the teacher's frozen classifier `W_T` (the BCT mechanism), optionally a
  D3still ranking-difference term. Trainable part of `S`: all weights, the last k blocks, or low-rank adapters.
- **Difference from prior work:** AML/CSD/D3still use a different, smaller network as student; here the
  student is the same network at a reduced token budget or resolution, so the two spaces start aligned
  (cosine 0.73-0.95 before training instead of unrelated). SUReID distils from a frozen teacher but evaluates
  the pruned student against a pruned gallery in the text read. **Honest view: this is asymmetric-retrieval
  distillation with a new kind of student; the novelty is in the setting and the cost accounting.**
- **Correction 2026-11-16 (after reading Suma and Tolias in full, LITERATURE_MATRIX.md A5):** a frozen teacher
  with a student of the same architecture initialised from the teacher's weights, trained label-free on a
  cheaper input, is exactly their method (resolution knob, CNNs, CUB/Cars/SOP). The "same network, aligned at
  the start" point above is therefore **not** a difference from prior work. C1 is a baseline/component for us
  (new only in: token knob, re-ID, one student for several budgets, cost accounting), not a contribution.
- **Expected advantages:** zero full-token deficit; no re-indexing; one forward+backward of the cheap pass per
  step plus a no-grad teacher pass, so cheaper than TBSD; label-free variant possible.
- **Limitations:** two sets of weights (or adapters) to store; one student may not cover all budgets.
- **Compute:** estimated about 0.6x TBSD per step (unverified); can start from an existing checkpoint.
- **Validating experiments:** (a) full fine-tune vs last-k blocks vs adapters; (b) with/without frozen-classifier
  term; (c) compare to TBSD and to the 216-epoch baseline at equal total GPU-hours **including** the teacher's
  training; (d) label-free version on 100% of images with 0% labels (ties to the "no data" thesis).

## C2. Knob-agnostic compatibility: train against several cheap-query families at once
- **Problem addressed:** E5 and E6. Token merging and resolution are competing knobs; a deployment may want
  either (resolution is faster at batch 1, merging keeps fine detail).
- **Intuition:** E6 shows robustness to one knob transfers partly to another. Sampling the cheap view from a
  family of perturbations (resolution, gradual merging, random dropping) may give an embedding that is
  compatible under any of them.
- **Formulation:** as TBSD, with the cheap view `b ~ Uniform(B_res ∪ B_merge)`:
  `L = L_id(full) + L_tri(full) + L_id(b) + L_tri(b) + λ (1 - cos(f(x;b), sg f(x;full)))`.
- **Difference from prior work:** resolution asymmetry (Suma and Tolias) and width switching (SFSC) each fix
  one knob. Multi-knob compatibility was not found in the texts read (absence of evidence only).
- **Risks:** could be "just augmentation"; may enlarge the full-token deficit.
- **Compute:** same as TBSD or lower (low-resolution passes are cheaper).
- **Validating experiments:** running now: `sp_lowres`, `sp_merge_r`, `sp_mixed`, 3 seeds each. Decisive
  comparison: accuracy-vs-GFLOPs frontier of each trained model on each knob, and full-token mAP.

## C3. Protect the full-token objective: asymmetric gradient projection
- **Problem addressed:** E7. SFSC attributes the same kind of loss to gradient conflict between sub-models.
- **Intuition:** the gallery is full-token, so the full-token objective should have priority. When the cheap
  pass's gradient opposes it, remove only the opposing component of the cheap gradient.
- **Formulation:** `g_f = ∇L_full`, `g_c = ∇L_cheap`. Update with
  `g = g_f + g_c - min(0, <g_c, g_f>) / ||g_f||² · g_f`. SFSC/PCGrad project symmetrically in random order;
  here only `g_c` is projected.
- **Not yet measured:** whether `cos(g_f, g_c)` is actually negative during TBSD training. **This diagnostic
  comes first; if the gradients rarely conflict, drop C3.**
- **Compute:** two backward passes per step (estimated +30-50% time, unverified).
- **Baseline to include:** symmetric PCGrad as in SFSC.

## C4. Drift-calibrated reduction schedule (training-free)
- **Problem addressed:** E3 and E4. Where tokens are removed matters more than how.
- **Intuition:** pick per-block reduction amounts that minimise embedding drift for a GFLOPs target, using
  unlabeled images and forward passes only.
- **Formulation:** choose `r = (r_1..r_12)` to minimise `E_x[1 - cos(f(x; r), f(x; full))]` subject to
  `GFLOPs(r) ≤ G`; greedy or coordinate search over a small grid, each evaluation a forward pass on a few
  hundred images.
- **Prior work to read first (prior knowledge, unverified):** DiffRate (ICCV 2023) learns per-layer
  compression rates for classification; ToMe reports decreasing schedules. Novelty would be limited to the
  retrieval-drift criterion and the zero-training cost.
- **Compute:** minutes of inference. **Role:** probably a strong baseline and a component, not a paper.

## What I would not pursue
- Class-attention selection as the reduction rule (E9), unless its failure is explained.
- "Self-distillation as a low-data regulariser" (H3 not supported at 10/25/50% of identities).
- Claiming the alignment loss or the two-pass scheme as new (D3still's `L_f`; SFSC's BCT-S/Asymmetric-S).

## Comparisons a paper would need (plan)
- Baselines at equal training GPU-hours: longer-trained full model; SFSC-style gradient projection;
  D3still loss terms on our setting; resolution asymmetry; gradual ToMe untrained and trained.
- A small CNN (OSNet) at equal query GFLOPs, as a separate-network asymmetric baseline.
- Datasets: Market-1501 plus at least one with higher native resolution (CUHK03-NP or MSMT17) and one
  cross-domain test (Occluded-REID, as in SUReID). Needs the user's sign-up/agreement.
- Efficiency reported separately: analytic GFLOPs; throughput at batch 256; single-image latency on GPU and
  CPU (merging was slower than full tokens at GPU batch 1); training GPU-hours; peak VRAM; gallery
  re-indexing cost.
