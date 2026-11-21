# Nonlinear (VPFA-inspired) mapping baseline

**Supplementary baseline experiment, 2026-11-21 (session 5).** Evaluation-time only: the ViT backbone is frozen
and never retrained. **This is our own VPFA-inspired implementation, not an official reproduction of VPFA**
(Liu et al., ICML 2026; the official code at github.com/ashmentlzw/VPFA was neither read nor run). The
iteration-1 report (`REPORT_ITERATION1.md`) is unchanged; no earlier number is replaced. Nothing is committed.

## 1. Motivation

- **Our claim.** `sp_mixed` beats the equal-GPU-hour baseline by +0.8 to +4.1 mAP below about 1.7 GFLOPs
  (Report Sections 6.4 and 7).
- **The limitation.** That comparison gives the baseline only a **linear** post-hoc correction (a ridge map).
  A stronger, non-linear post-hoc map could close part of the gap without any retraining, which would weaken
  the case for joint multi-budget training.
- **Why VPFA.** VPFA (ICML 2026) is the 2026 representative of this idea: a label-free residual gated MLP that
  maps low-resolution query features of a frozen re-ID ViT onto high-resolution features
  (`RECENT_LITERATURE_2025_2026.md`, P5).
- **Question.** Does a VPFA-style non-linear map on the frozen baseline change the conclusion that `sp_mixed`
  is better at low query budgets?

## 2. Methodology and equations

**Notation.**
- f(x) is the frozen baseline's BNNeck embedding, L2-normalised (d = 384).
- For a training image x, y = f(x) is its full-resolution embedding (256x128, 129 tokens).
- z = f_r(x) is its embedding at query resolution r (192x96, 160x80 or 128x64), with the position embedding
  resampled bicubically, as in the report.
- The gallery is always embedded at full resolution by the same frozen model and is never changed.
- A map g is applied only to the query: score(q, i) = cos(g(z_q), y_i).

**Maps compared.**
1. **raw:** g(z) = z.
2. **ridge_report:** the report's map (`mechanism.py`, `asym_ridge`), recomputed:
   - g(z) = z W, with W = (XᵀX + λI)⁻¹ XᵀY and λ = 1;
   - X and Y are cheap and full embeddings of every 2nd training image (6,468 images).
3. **ridge_cv:** the same closed form, fitted on the FIT identities (with horizontal flips), with
   λ ∈ {0.1, 1, 10, 100} chosen on VAL. This is a linear map tuned as carefully as the MLP.
4. **mlp (VPFA-inspired):** the residual gated MLP of VPFA Eqs. 2-4, with the loss of VPFA Eq. 5:

       VP(z)  = W3 · ReLU(LN(W2 · ReLU(LN(W1 z + b1)) + b2)) + b3
       g(z)   = z + tanh(W4 · VP(z) + b4)
       L      = mean over pairs of ‖g(z) − y‖²

   - Widths: 384 → h → h → 384, then a 384 → 384 gate.
   - Initialisation: N(0, 10⁻³) weights and zero biases, so g starts at the identity.
   - Optimiser: Adam with weight decay.
   - The output is L2-normalised before retrieval.

**Leakage control.**
- Every map is fitted on images of `bounding_box_train` only. The 751 training identities are disjoint from
  the 750 test identities (Market-1501 protocol).
- No identity labels are used by any fit, only pairs (z, y) of the same image.
- The test query and gallery sets are used **once**, after all choices are fixed.

**Model selection (overfitting control).**
- The 751 training identities are split once (seed 0) into 601 FIT and 150 VAL identities: 10,576 and 2,360
  images. FIT pairs are doubled with horizontal flips (21,152 pairs).
- Every candidate is fitted on FIT. The ridge λ, the MLP configuration and the MLP early-stopping epoch are
  chosen by the **VAL loss** ‖normalise(g(z)) − y‖² = 2 − 2cos, which is checked every 5 epochs.
- The model at the best VAL epoch is kept.
- Retrieval mAP on VAL identities (VAL cheap queries against VAL full embeddings) was the first criterion
  tried, but it is **saturated at 99.98 mAP**: the frozen ViT was trained on all training identities. It is
  logged but not used.
- The selection never sees the test split.

**MLP candidates** (chosen per seed and per resolution on VAL).

| Name | hidden h | Adam lr | weight decay | batch | max epochs | Params |
| --- | --- | --- | --- | --- | --- | --- |
| `vpfa` | 2048 | 2e-4 | 1e-5 | 32 | 30 | 5.93 M |
| `h2048_wd1e-5` | 2048 | 1e-3 | 1e-5 | 256 | 120 | 5.93 M |
| `h512_wd1e-5` | 512 | 1e-3 | 1e-5 | 256 | 120 | 0.81 M |

- `vpfa` uses the optimiser settings stated in the VPFA paper. Its 30 epochs over 21,152 pairs give about the
  paper's number of optimiser steps (5,000 pairs / 32 × 120 epochs ≈ 18.8k).
- VPFA's hidden width (2048) is kept, although its input is 3840-d and ours is 384-d.
- A fourth candidate (h = 512, weight decay 1e-3) was dropped after the smoke test, for the GPU budget: its
  VAL loss was within 10⁻³ of `h512_wd1e-5`.

**Differences from VPFA** (it is a re-implementation from the paper text, and adapted):
- 384-d BNNeck embeddings (theirs 3840-d);
- L2-normalised targets;
- 21k image pairs (theirs 5,000);
- our own selection protocol;
- a frozen *ViT-S trained by us* instead of TransReID;
- synthetic resolution from bicubic resizing of 256x128 crops instead of MLR-style downsampling.

## 3. Implementation

| File | Role |
| --- | --- |
| `nonlinear_map.py` (new) | Embeds query, gallery and training images with a frozen checkpoint; fits raw / ridge_report / ridge_cv / mlp per resolution; evaluates on the test split; times the MLP |
| `aggregate_nonlinear.py` (new) | Mean ± std over seeds, paired differences, cost table; adds `sp_mixed` mAP and R1 from `runs/adaptive.csv` and backbone latency from `runs/latency_extra.csv` |

**Reused code.**
- `mechanism.py`: `cache_images`, `embed`, `forward_res`, `gflops_res`.
- `common.py`: `evaluate_rank` (standard Market-1501 protocol: junk and same-camera matches ignored) and
  `vit_gflops`.
- Low-resolution queries are produced exactly as in the report: bicubic antialiased resize of the 256x128
  tensor, plus bicubic resampling of the position embedding.

**Checkpoints and embeddings.**
- Checkpoints: `runs/base_e216_s{0,1,2}/last.pth` (the equal-GPU-hour baseline of Report Section 6.4), loaded
  read-only.
- No embeddings had been saved before, so they were recomputed: a few seconds per setting.
- Nothing was trained except the maps. No checkpoint was written or changed.

**Sanity checks.**
- `ridge_report` reproduces the report's linear-map numbers exactly: 82.80 / 78.96 / 68.88 at 192x96 / 160x80
  / 128x64. The per-seed values equal those in `runs/base_e216_s*/mech.csv`.
- The full-resolution raw mAP equals the report's 85.11.

**Outputs.**
- `runs/nonlinear_map/nonlinear_map.csv` (per seed);
- `runs/nonlinear_map/nonlinear_map_log.json` (VAL curves of every candidate);
- `runs/nonlinear_map/agg.md` (aggregate);
- `runs/nonlinear_map/stdout.txt`;
- smoke-test output in `runs/nonlinear_smoke/`.

## 4. Experimental settings

- **Data.** Market-1501: 3,368 queries; gallery 19,732 files, of which 15,913 are evaluated.
- **Frozen model.** ViT-S/16 baseline trained 216 epochs (3 seeds). Gallery and VAL targets are at full
  resolution (2.93 GFLOPs).
- **Query resolutions.** 192x96 (1.62 GFLOPs), 160x80 (1.12) and 128x64 (0.72).
- **Map seeds.** One MLP initialisation seed (0) per backbone seed and resolution, so the seed variation is the
  backbone's, as for `sp_mixed`.
- **Commands.**
  - Smoke test: `nonlinear_map.py ... --ckpts runs/base_e216_s0/last.pth --res 160x80 --max-epochs 3 --eval-every 1`.
  - Main run:

        .venv/Scripts/python.exe nonlinear_map.py --data C:/Users/user/data/Market-1501-v15.09.15 \
            --ckpts runs/base_e216_s0/last.pth runs/base_e216_s1/last.pth runs/base_e216_s2/last.pth \
            --out runs/nonlinear_map/nonlinear_map.csv

    Defaults: `--workers 2 --val-frac 0.2 --eval-every 5`.
  - Aggregate: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe aggregate_nonlinear.py`.
- **GPU time.** The main run took 750 s (12.5 min) and the two smoke tests about 2.2 min, so about **15
  GPU-minutes** of the 30-minute budget. These wall-clock times include CPU image loading; the machine was
  otherwise idle.

## 5. Results

Setup:
- Asymmetric retrieval: the cheap query (after the map) is matched against the full-resolution gallery of the
  same frozen model.
- Values are mean ± sample std over 3 seeds (`runs/nonlinear_map/agg.md`).
- `sp_mixed` rows are the existing iteration-1 results (`runs/adaptive.csv`, fixed budgets, same evaluation
  code), not re-run.

**mAP (%)**

| Query | GFLOPs | Baseline, raw | + linear map (report, λ = 1) | + linear map (λ tuned on VAL) | + **nonlinear MLP (VPFA-inspired)** | **`sp_mixed`** |
| --- | --- | --- | --- | --- | --- | --- |
| full | 2.93 | 85.11 ± 0.15 | — | — | — | 83.95 ± 0.15 |
| 192x96 | 1.62 | 82.06 ± 0.19 | 82.80 ± 0.23 | 82.96 ± 0.22 | 82.77 ± 0.31 | **83.60 ± 0.14** |
| 160x80 | 1.12 | 75.72 ± 0.51 | 78.96 ± 0.18 | 79.16 ± 0.21 | 78.61 ± 0.33 | **83.01 ± 0.15** |
| 128x64 | 0.72 | 57.45 ± 2.34 | 68.88 ± 0.70 | 68.84 ± 0.69 | 66.88 ± 0.22 | **81.50 ± 0.22** |

**Rank-1 (%)**

| Query | Baseline, raw | + linear (report) | + linear (tuned) | + nonlinear MLP | **`sp_mixed`** |
| --- | --- | --- | --- | --- | --- |
| full | 94.08 ± 0.39 | — | — | — | 93.74 ± 0.17 (`sweep.csv`) |
| 192x96 | 92.39 ± 0.55 | 92.79 ± 0.66 | 92.82 ± 0.62 | 92.85 ± 0.51 | **93.59 ± 0.45** |
| 160x80 | 88.22 ± 0.29 | 90.44 ± 0.35 | 90.46 ± 0.41 | 89.99 ± 0.32 | **92.95 ± 0.27** |
| 128x64 | 70.67 ± 2.71 | 82.35 ± 1.13 | 82.23 ± 1.05 | 80.38 ± 0.36 | **92.09 ± 0.02** |

Rank-1 of the linear map was not reported in iteration 1. It is new here and comes from the same fit.

**Paired differences** (per seed, then mean ± std).

| Query | MLP − linear (report), mAP | MLP − linear (tuned), mAP | `sp_mixed` − best baseline map, mAP | `sp_mixed` − best baseline map, R1 |
| --- | --- | --- | --- | --- |
| 192x96 | −0.02 ± 0.09 | −0.19 ± 0.10 | +0.64 ± 0.35 | +0.69 ± 0.66 |
| 160x80 | −0.35 ± 0.15 | −0.55 ± 0.12 | +3.85 ± 0.30 | +2.45 ± 0.22 |
| 128x64 | −2.00 ± 0.69 | −1.96 ± 0.66 | +12.61 ± 0.71 | +9.74 ± 1.14 |

- "Best baseline map" is the per-seed maximum of the three maps **on the test set**. That choice is
  deliberately optimistic for the baseline (oracle selection).
- `sp_mixed` and the baseline are independent runs; pairing seed i with seed i is bookkeeping only.

**Hyperparameter selection** (on VAL, never the test split).
- Tuned λ: 0.1 at 192x96 and 160x80, and 1 at 128x64, in all 3 seeds.
- Selected MLP: `h512_wd1e-5` or `h2048_wd1e-5` (lr 1e-3), with the best epoch at 10-20 of 120.
- The VPFA-setting candidate (`vpfa`) was never selected: its VAL loss was the highest of the three MLPs in
  every case.
- VAL curves are in `nonlinear_map_log.json`.

**Overfitting check: lower held-out loss, lower test mAP.**
- The selected MLP has a *lower* VAL loss than the tuned ridge in 8 of 9 cases:

  | Query | MLP VAL loss | Ridge VAL loss |
  | --- | --- | --- |
  | 192x96 | 0.122 | 0.124 |
  | 160x80 | 0.264 | 0.266 |
  | 128x64 | 0.566 | 0.572 |

- Yet its test mAP is *lower*. The MLP fits the cheap→full mapping on held-out *training* identities slightly
  better than the ridge, but this does not transfer to the 750 test identities.
- Early stopping chose 10-20 epochs, after which the VAL loss rose again.
- **Hypothesis, not tested:** the frozen ViT was trained on every training identity, so their embeddings form
  tight clusters. A non-linear map can exploit that structure, which test identities lack. A linear map has
  too little capacity to overfit it.

**Cost of the map** (MLP alone; backbone latency from `runs/latency_extra.csv`, same machine).

| | MLP, h = 512 | MLP, h = 2048 | Linear map | Backbone at 160x80 (for scale) |
| --- | --- | --- | --- | --- |
| Parameters | 0.81 M | 5.93 M | 0.15 M | about 21.7 M |
| Extra compute | 0.0008 GMACs | 0.0059 GMACs | 0.00015 GMACs | 1.12 GFLOPs |
| Fit time per resolution and seed | about 18 s (GPU) | about 18 s (GPU) | under 1 s (closed form) | — |
| GPU batch 1 | 0.21 ms | 0.22 ms | not timed | 6.16 ms |
| GPU batch 256 | about 0.001 ms/img | about 0.001 ms/img | not timed | 0.042 ms/img |
| CPU batch 1 | 0.13-0.29 ms | 0.61 ms | not timed | 10.6 ms |

- Each map adds at most 3-6 % latency at batch 1, so no conclusion about speed changes.
- The ViT-S parameter count is the standard figure for the architecture, not re-counted here.

## 6. Limitations

1. **Not an official VPFA reproduction.** This is a re-implementation of VPFA's equations from the paper text.
   It differs in:
   - embedding size (384 vs 3840) and L2-normalised targets;
   - backbone (our ViT-S, not TransReID);
   - number of pairs;
   - selection protocol;
   - synthetic resolution (bicubic resize of 256x128, not MLR downsampling).

   VPFA's own gains (+3.8 Rank-1 over TransReID on MLR-Market) come from a different protocol and are not
   comparable.
2. **Small search.** Three MLP configurations and four λ values, with one map initialisation per seed. A larger
   search could do better, and was not tested within the 30-minute budget. Candidates include other losses
   (cosine, or relational terms as in Suma and Tolias) and more pairs (augmentations beyond flips).
3. **Selection signal.** VAL identities were seen by the frozen backbone, so neither VAL loss nor VAL retrieval
   fully represents unseen identities. VAL retrieval mAP was saturated (99-100), so the regression loss was
   used instead. The loss-versus-test mismatch is itself a finding (Section 5).
4. **One baseline.** Only the 216-epoch baseline was mapped. The 120-epoch baseline was not, and `fg_mixed`
   has its own frozen-gallery comparison (Report Section 6.8).
5. **One dataset,** whose crops are natively 128x64 (Report Section 8). The large margins at 128x64 in
   particular should not be generalised.
6. **Low-resolution queries only.** Token-merging queries were not mapped. The report's linear map at merge
   r = 11 (79.70) remains the only post-hoc number there.
7. **Pairing.** `sp_mixed`-minus-baseline differences pair independent runs. The ± values describe seed
   spread, not a formal paired test.

## 7. Implications for iteration 1

- **The main conclusion holds, and the report's linear-map baseline was not a weak strawman.**
  - The VPFA-inspired non-linear map does **not** beat a linear map on this setting: −0.02 to −2.0 mAP
    against the report's map, and −0.2 to −2.0 against a tuned linear map.
  - The non-linear map adds 0.8-5.9 M parameters.
- **Against the best of all three post-hoc maps** (chosen on the test set, so optimistic for the baseline),
  `sp_mixed` stays ahead:
  - +0.64 ± 0.35 mAP at 192x96 (about 2 std; the report said +0.8 against its linear map);
  - +3.85 ± 0.30 at 160x80 (report: +4.05);
  - +12.6 at 128x64.
- **Suggested wording for a later report revision** (not applied: the report stays unchanged, as requested):
  - "Below about 1.7 GFLOPs `sp_mixed` beats the equal-GPU-hour baseline equipped with linear or
    VPFA-inspired non-linear post-hoc maps, by +0.6 (192x96, marginal) to +3.9 (160x80) mAP."
  - The 192x96 margin should be called small.
  - The full-cost price is unchanged: 83.95 vs 85.11 mAP.
- **Tuned linear map.** It is slightly better than the report's λ = 1 map (+0.16 / +0.20 mAP at 192x96 /
  160x80). A revised report should keep its numbers and mention this, or use the tuned values with a note. It
  must not silently replace them.
- **Direction D2 of the literature review** (`RECENT_LITERATURE_2025_2026.md`) is answered for its non-linear
  part: on this data a post-hoc map does not match joint training. The ACD-style gallery-side variant remains
  untested.
