# Low-resource re-ID experiment kit (iteration 1)

Four files, no framework: `common.py` (data, model, token reduction, evaluation), `train.py`, `evaluate.py`,
and `make_fake_market.py` (tiny random dataset to check the install).

**Status:** unit-checked; run end to end on CPU and (2026-11-14) on the RTX 5080 with a synthetic dataset.
It has not been run on Market-1501. Treat the first real run as the test: see "Sanity checks" below.
Project venv: `.venv` (torch 2.15.0+cu132). One-command queue: `bash run_iter1.sh <Market-1501 root>`.
History and current state: `PROGRESS_LOG.md`.

## Setup

1. Install a current PyTorch build with CUDA support for the RTX 5080 from the selector on pytorch.org.
2. `pip install timm pillow numpy matplotlib`
3. Download Market-1501. The folder must contain `bounding_box_train/`, `query/` and `bounding_box_test/`.
4. Check the install: `python make_fake_market.py fake_market` then
   `python train.py --data fake_market --out runs/smoke --epochs 1 --P 4 --K 2 --eval-every 1`

On Windows keep `--workers 4`. If RAM runs short (16 GB), lower it to 2.

## Runs for tonight, in order

Replace `DATA` with your Market-1501 path.

```
# 1. Baseline ViT-S/16, full tokens
python train.py --data DATA --out runs/base_s0 --seed 0

# 2. Sweep token reduction on the baseline (inference only, minutes)
python evaluate.py --data DATA --ckpt runs/base_s0/last.pth --out runs/base_s0/sweep.csv

# 3. Token-budget self-distillation (same network: full-token pass teaches a merged-token pass)
python train.py --data DATA --out runs/tbsd_s0 --seed 0 --sd-mode merge --sd-keep random

# 4. Same sweep on the self-distilled model
python evaluate.py --data DATA --ckpt runs/tbsd_s0/last.pth --out runs/tbsd_s0/sweep.csv

# 5. If time remains: low-data versions of 1 and 3
python train.py --data DATA --out runs/base_id25_s0 --id-fraction 0.25
python train.py --data DATA --out runs/tbsd_id25_s0 --id-fraction 0.25 --sd-mode merge --sd-keep random
```

Each training run writes `log.json` (loss, mAP, Rank-1, training hours, peak VRAM) and `last.pth`.
Each sweep writes a CSV and a PNG with two panels: symmetric and asymmetric retrieval.

## What to look at

- **Baseline sweep, symmetric panel:** how fast mAP falls as tokens are removed, and whether `merge` and
  `cls_attn` beat the `random` control. If they do not, attention-based criteria are not finding identity cues.
- **Baseline sweep, asymmetric panel:** the gap between symmetric and asymmetric mAP at the same keep ratio.
  A large gap means reduced-token embeddings drift out of the full-token embedding space.
- **Self-distilled model against baseline:** does the asymmetric curve move up at low keep ratios, and does
  full-token mAP stay the same? That is the first test of the proposed direction.
- **Training cost:** compare `train_hours` in the two `log.json` files. Self-distillation runs two forward
  passes per step, so it should cost more per epoch; the question is what it buys.

## Sanity checks

- Baseline mAP on Market-1501 after 120 epochs should land somewhere in the 80s. A result far below that
  means a bug or a setup problem, not a finding; send me the log.
- `keep=1.0` and the `none` row must give identical numbers.
- The analytic GFLOPs for ViT-S at 256x128 with full tokens should print about 2.93.

## Options worth knowing

| Flag | Meaning |
| --- | --- |
| `--id-fraction 0.25` | train on a random 25% of identities (low-data setting) |
| `--train-mode merge --train-keep 0.7` | reduce tokens in the main training pass (cheaper training) |
| `--sd-mode {merge,cls_attn,random}` | reduction used in the distilled pass |
| `--sd-keep random` or `0.7` | keep ratio per reduction layer; `random` samples from 0.5 to 0.9 each step |
| `--model vit_base_patch16_224` | larger backbone |
| `evaluate.py --cpu` | run the sweep on CPU to get CPU throughput |

Tokens are reduced after the attention of blocks 4, 7 and 10; `keep` is the fraction of patch tokens kept
at each of those three points, so `keep=0.7` leaves about 34% of patch tokens at the end.

## Pausing and resuming training

`train.py` saves a resumable `<out>/ckpt.pth` after every epoch (`--ckpt-every N` to change; written
atomically). It holds model, optimizer, AMP scaler, epoch, step (the LR schedule is a function of step),
elapsed time, the metrics log, args and all RNG states (Python, NumPy, torch CPU/CUDA, PK sampler).
It is deleted once `last.pth` is written, unless `--keep-ckpt`.

- **Pause gracefully:** create an empty file `<out>/STOP` (e.g. `touch runs/tbsd_s0/STOP`). The run finishes
  the current epoch, saves `ckpt.pth` and exits with code 3. Queue scripts stop when they see exit code 3.
- **Resume:** rerun the *same* command with `--resume` (or rerun the queue script, which always passes it).
  Key args (`--epochs`, `--lr`, `--seed`, modes...) must match the checkpoint or it refuses to resume.
- Verified on CPU: pause after epoch 1 + resume gives bit-identical weights to an uninterrupted run.
  On GPU (cuDNN/AMP nondeterminism) and with `--workers > 0` (worker augmentation streams restart) the
  resumed run is statistically equivalent but not bit-identical.
- A hard kill (Ctrl-C, shutdown) loses at most the current epoch; resume from the last `ckpt.pth`.

## Known limitations

- `merge` follows the published ToMe idea (bipartite matching on attention keys, proportional attention)
  but is my own implementation with an EViT-style three-stage schedule, so its numbers will not match the
  official ToMe code exactly.
- GFLOPs are an analytic estimate of the backbone, not a profiler measurement.
- Only Market-1501-style file names are parsed. MSMT17 and others need a small loader added.
