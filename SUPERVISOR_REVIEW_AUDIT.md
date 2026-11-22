# Supervisor-review audit, 2026-11-22

This file lists what was checked, what was corrected in `REPORT_ITERATION1.md`, and what remains unresolved.
No training, checkpoint evaluation or dataset experiment was run. Nothing was committed or pushed.

## 1. What was checked, and how

| Check | Method | Result |
| --- | --- | --- |
| Benchmark numbers | Recomputed mean and sample std from per-seed CSVs, independently of `aggregate*.py` | Core mAP and Rank-1 values reproduce. The errors were in derived ranges, counts and wording (Section 3) |
| Same measurement in different files | Compared sweep, mech, cascade, map and diagnostic CSVs seed by seed | Differences of at most 0.05 mAP, the largest under merging |
| Training logs | `log.json` against `stdout.txt` and `sweep.csv` for every real run | Consistent |
| Checkpoints | Loaded on CPU, weights hashed, saved arguments read | 50 real checkpoints, all distinct; arguments of the headline runs match |
| Data split | Counted files and identities, MD5 of every image | 12,936 / 3,368 / 15,913; no identity or identical-file overlap |
| mAP / Rank-1 code | Brute-force implementation, 200 random cases | Exact (below 1e-14) |
| Merging code | CPU test on random tensors against a transcription of official ToMe | **Not identical** (D1) |
| Saved diagnostic arrays | Recomputed Rank-1 and `q_cos` from 18 `.npz` files | Match the CSV |
| Citations | Online search for the six closest papers, plus DaRe, MatFormer, BTNet and the ThinkingViT venue | Venues and pages confirmed as cited. Three gap-narrowing papers had been missing from the report |
| Published numbers | D3still and SUReID against saved paper text | Match. The PrunNet, Suma-Tolias and VPFA numbers come from earlier tool-mediated reads and were **not** re-checked |

Not checked:
- whether re-running training reproduces the numbers;
- whether the checkpoints reproduce their CSVs (this would need evaluation runs);
- the latency measurements, beyond reading their CSVs;
- the contents of ACD, TE-VMamba, SRE-Merge and OSKT.

## 2. Unresolved issues to disclose

### High priority

**D1. Merging is not official ToMe.**
- `check_reduction.py` compared against the `class_token=False` path. Official ToMe passes `class_token=True`
  for a ViT with a class token (`tome/patch/timm.py`), which changes both the A/B partition and the token order.
- On random input, one step gave the same token set in 64/64 samples at r = 1, 56/64 at 3, 29/64 at 7, 10/64
  at 11 and 0/64 at 38.
- All merge results, and every training run with a merge pass, use our variant. The effect on mAP is unknown.
  The code was left unchanged so that it still matches the checkpoints.

**D2. The proposed method does not clearly beat the strongest baseline.**
- `sp_mixed` minus the frozen-gallery student: +0.88 at r = 11 (1.4 combined std, 2.8 paired) and +1.02 at
  128x64 (1.4 combined, 2.1 paired). Both are positive in all seeds and both are borderline. The other cheap
  trained budgets evaluated are ties.
- The std convention was chosen after the results were known, and these two verdicts depend on it.
- The only clear lead in the main table is on the 3-stage schedule, which neither model was trained on (+2.42).

**D3. The gap is narrower than previously written.**
- DaRe (CVPR 2018) already routes re-ID queries by nearest-gallery distance on Market-1501, with a random
  control and shared computation. It was in `PROGRESS_LOG.md` as a lead but never reached the report.
- MatFormer (NeurIPS 2024) already uses one ViT with sub-model queries against a largest-model gallery.
- BTNet (BMVC 2023) already links resolution, compute and compatible embeddings, for faces.
- The search was not exhaustive, so more such papers may exist.

**D4. No validation split.**
- Variants, budget sets and the choice of method followed test results.
- Test mAP was logged six times per 120-epoch run (the last epoch was always used).
- The MLP smoke test printed test mAP at 23:02:27 and `nonlinear_map.py` was last edited at 23:02:58. One
  candidate was dropped after that smoke test. `NONLINEAR_MAPPING_BASELINE.md` says the test set was used once, which is inaccurate.

**D5. Decisive baselines are missing.**
- Single-pass multi-resolution augmentation, the simplest competitor.
- ViT-Tiny or OSNet at equal GFLOPs. `PROJECT_CONTEXT.md` says to always compare against a small CNN; this was
  never done.
- A low-resolution model with a re-indexed gallery.
- A frozen-gallery student of the 216-epoch teacher.

**D6. One dataset with native 128x64 crops.** Upsampling and then downsizing loses little, so the
low-resolution findings may not transfer.

**D7. AI assistance and reading depth.** The assistant wrote the code, ran the experiments, processed the papers
and drafted the documents. "Read in full" in the older documents means read by the assistant, often through a
summarising tool. Check your institution's rules, say this to your supervisor, and read the closest papers
before presenting them as read.

### Medium priority

**D8. Causal claims are weaker than written.**
- The shared classifier was never isolated.
- Alignment was ablated only on the 3-stage schedule.
- The "information loss" explanation is not identified by the diagnostic. Baseline same-budget Rank-1 is higher
  than asymmetric at 128x64 (73.15 vs 70.67, borderline).
- The "hardest-negative" explanation of the full-cost deficit is contradicted by `tbsd_nosd`, which has the
  deficit with lower hardest-negative similarity.

**D9. Training budgets are only similar.**
- `sp_mixed` mean 0.40 GPU-h against 0.38; merge-only +14 %; student plus teacher +21 %.
- Identical configurations vary by up to about 25 % in time.

**D10. "Label-free student" needs a qualifier.** The loss uses no labels, but the PK sampler does. It also
differs from Suma and Tolias in loss form, augmentations, backbone and budget set.

**D11. Three runs were resumed** (`tbsd_bnns_s0`, `sp_mixed_s1`, `vitb_tbsd_s0`). They are not bit-identical to
uninterrupted runs, and `sp_mixed_s1` is one of the three headline seeds.

**D12. Merge evaluation is not deterministic.** Two evaluations of `base_e216_s0` at r = 11 gave 78.84 and 78.88
mAP, 90.88 and 91.00 Rank-1, and 152 and 148 broken queries (`mech_diag_check.csv` vs `mech_diag.csv`).
`MECHANISM_ANALYSIS.md` attributes such differences to rounding, which does not explain them.

**D13. The original objectives are only partly met.** Low-data benefit (H3) is not supported, cross-dataset
behaviour (H2) is untested, and the original novelty requirement was relaxed on 2026-11-17. Confirm that your
supervisor accepts that relaxation.

**D14. The post-hoc baseline search was small.** The tuned ridge λ sits at the grid edge for two of three
resolutions, so the baseline could be slightly stronger, and the margin at 192x96 is already borderline.

### Repository state (your Git decisions)

**D15. Evidence is uncommitted or ignored.**
- `MECHANISM_ANALYSIS.md`, `mech_diag.py`, `aggregate_mech_diag.py`, `runs/mech_diag/` and
  `runs/nonlinear_smoke/` are untracked.
- `runs/mech_diag/*.npz` are git-ignored, so the histogram cannot be rebuilt from a clone.
- `runs/nonlinear_smoke/` is the evidence for D4, so keep it.
- Checkpoints are local only.

**D17. Documented commands overwrite results.**
- **Fixed for `train.py` on 2026-11-22.** It used to replace `log.json` and `last.pth` in an existing `--out`,
  also with `--resume` on a finished run. It now refuses (`refuse_overwrite`); the training logic is unchanged.
  The queue scripts inherit this, but still append to a run's `stdout.txt` before the refusal.
- `mechanism.py`, `evaluate.py`, `adaptive.py` and `bench_latency.py` still overwrite their `--out`, as do `nonlinear_map.py` and `mech_diag.py`.
- `PROGRESS.md` and `PROGRESS_LOG.md` contain such commands with the original paths, and so do the
  `run_iter*.sh` queues (they skip a run only while its `last.pth` exists, so not in a fresh clone).
  `README.md` did too; since 2026-11-22 its commands use new paths and skip existing outputs.
  `STUDY_GUIDE.md` Section 15 did too and now uses new paths.
- Checkpoints are not in Git, so an overwrite cannot be undone.

**D16. The queue scripts (`run_iter2.sh`, `run_iter3.sh`, `run_iter4.sh`) commit and push by themselves.** Do
not run them while you want to control Git.

## 3. Corrections made to the main report

No measurement was altered. These are corrections to how measurements were quoted or interpreted.

| # | Earlier report | Evidence says |
| --- | --- | --- |
| C1 | 57 runs | 50 real runs + 7 synthetic smoke runs |
| C2 | Peak VRAM 4.9 GB (two-pass) | 3.17 GB allocated (`log.json`); the source of 4.9 is not logged, likely `nvidia-smi` |
| C3 | 216-epoch baseline 0.36-0.38 GPU-h | 0.36-0.40, mean 0.38 |
| C4 | Merging verified against official ToMe, difference 0 | Wrong code path tested (D1) |
| C5 | About 1 mAP over the student, about 2 std | Borderline: 1.4 combined std, 2.1-2.8 paired (see Section 5) |
| C6 | Student has no full-cost deficit | Equals its teacher: 0.66 below the 216-epoch baseline on average, a tie |
| C7 | "+0.8 to +4.1 below 1.7 G" | Held against the λ = 1 map and omitted 0.72 G (+12.61). Against the best map: +0.64, +3.85, +12.61 |
| C8 | Baseline cells blank at 224x112 and 128x64 | 84.28 (above `sp_mixed`: tie raw, borderline with correction) and 57.45; both were already measured |
| C9 | +0.5 to +0.75 over the resolution specialist, about 3 std | +0.54 to +0.84; clear at three resolutions, borderline at 128x64 |
| C10 | Specialists are not compatible on the other knob | They carry over partly |
| C11 | TBSD not better than baseline + map | Better under merging (+2.07 at r = 11), not at low resolution |
| C12 | Offset 3-16 %; mean/affine +0.1 to +1.5; linear +1 to +4 | 3-18 %; 0 to +3.8; −0.1 to +8.9 |
| C13 | Low resolution 79.6 vs merging 77.5 at "1.4-1.5 G" | At the same 1.41 G: about 79.0 vs 77.45 ± 1.39 |
| C14 | Resolution speed-up 1.7-3.9x | Only for 160x80 and 128x64; 192x96 is 1.4-1.9x |
| C15 | Cascade signal within 0.1 of the oracle; up to +5.9 over random | Within 0.15; up to +6.13; the drift oracle is not an upper bound |
| C16 | Alternative signal no better than random | Mostly worse (to −1.21), better in one setting (to +0.94) |
| C17 | Baseline cascade gain −1 to −8 | −0.31 to −8.39 |
| C18 | No-alignment full cost 83.63 (seed 0) | 83.71 ± 0.20 over 3 seeds |
| C19 | Shared classifier does most of the work | Two-pass training does; the classifier was not isolated |
| C20 | Alignment loss equals the D3still and Suma-Tolias terms | Same family, different functional form |
| C21 | SUReID 87.2 "at full cost" | With its token pruning active |
| C22 | Best method below 1.5 GFLOPs | Not clearly separated from the student; merge specialist is higher at r = 11 (a tie) |
| C23 | DVT, read at abstract level, among the six closest papers, with absence claims | Replaced by VPFA; DVT moved to other work |
| C24 | Per-query gallery confidence claimed as gap and contribution | Withdrawn (DaRe) |
| A1 | Missing from the report | Non-linear map baseline (6.10), mechanism diagnostic (6.11), Rank-1, low-data table, fixed-budget values, run inventory (6.12) |

## 4. Companion documents that still contain superseded claims

`SUPERVISOR_QA.md` and `STUDY_GUIDE.md` were brought up to date later on 2026-11-22. The files below were not
edited. **Do not hand them to the supervisor as they are.** The revised report takes precedence.

| File | Superseded content |
| --- | --- |
| `MECHANISM_ANALYSIS.md` | "Rounding" explanation (Section 3); 5.1 "dominant failure" and "mainly fixes"; 5.4 hypothesis; Section 7 first bullet |
| `NONLINEAR_MAPPING_BASELINE.md` | "Test used once" (Section 2); "report unchanged" and "nothing committed" notes |
| `RECENT_LITERATURE_2025_2026.md` | Section 4: per-query gap and "still not found" verdicts predate DaRe, MatFormer and BTNet; D2 is now done in part |
| `LITERATURE_MATRIX.md` | ToMe "verified, difference 0"; gap sections D-G; no entries for DaRe, MatFormer, BTNet |
| `README.md`, `METHOD_CANDIDATES.md`, older sections of `PROGRESS.md`, `RESEARCH_LOG.md`, `PROGRESS_LOG.md` | Historical records with the earlier claims |
| `CLAUDE.md` | State notes repeat C3, C5 and C7. It is your instruction file, so it was left to you |

## 5. Corrections to this audit (2026-11-22, second pass)

The first version of this audit, and the report and brief revised with it, contained these errors. All are now
corrected in those files.

| # | First pass said | Correct |
| --- | --- | --- |
| E1 | The two leads over the frozen-gallery student are ties under "the project's own rule" | The rule never defined std. The first pass chose combined std, which hid paired ratios of 2.8 and 2.1, yet used the paired std to downgrade the 192x96 win. Both readings are now applied everywhere: borderline |
| E2 | The baseline is better at 224x112 | Borderline with a correction, a tie without |
| E3 | `sp_mixed` ties the resolution specialist at 128x64 | Borderline in favour of `sp_mixed` |
| E4 | Class attention vs random is a tie only on the 216-epoch baseline at keep 0.7 | Also a tie on the 120-epoch baseline and the no-alignment model at keep 0.7; borderline on the 216-epoch baseline at keep 0.5 |
| E5 | Low-data full-cost differences are all within noise | Borderline at 50 % (TBSD lower) |
| E6 | CPU timings differ 10-16 % between sessions | 9-17 % at batch 1, 17-46 % at batch 8 |
| E7 | The student's 0.66 deficit against the 216-epoch baseline, stated as a finding | A tie |
| E8 | Same-budget Rank-1 is higher at 128x64, stated as clear | Borderline |
| E9 | Pointed to the study guide's reproduction commands | Those commands overwrote results (D17) |
