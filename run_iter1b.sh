#!/usr/bin/env bash
# Iteration-1b: seeds 1-2 for baseline and TBSD, plus fairness controls (seed 0).
# Usage (Git Bash): bash run_iter1b.sh /c/Users/user/data/Market-1501-v15.09.15
set -u
DATA="${1:?pass the Market-1501 root}"
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1

train() {  # name, args...
  local name=$1; shift
  if [ -f "runs/$name/last.pth" ]; then echo "skip train $name"; return; fi
  mkdir -p "runs/$name"
  echo "=== train $name $(date)"
  $PY train.py --data "$DATA" --out "runs/$name" --resume "$@" >> "runs/$name/stdout.txt" 2>&1
  local rc=$?
  if [ $rc -eq 3 ]; then echo "PAUSED $name (resume: rerun this script)"; exit 3; fi
  [ $rc -eq 0 ] || { echo "FAILED train $name"; return 1; }
  tail -2 "runs/$name/stdout.txt"
}
sweep() {  # name, extra evaluate args...
  local name=$1; shift
  if [ -f "runs/$name/sweep.csv" ]; then echo "skip sweep $name"; return; fi
  echo "=== sweep $name $(date)"
  $PY evaluate.py --data "$DATA" --ckpt "runs/$name/last.pth" --out "runs/$name/sweep.csv" "$@" \
    > "runs/$name/sweep_stdout.txt" 2>&1 || { echo "FAILED sweep $name"; return 1; }
}

TBSD="--sd-mode merge --sd-keep random"
for s in 1 2; do
  train base_s$s --seed $s && sweep base_s$s
  train tbsd_s$s --seed $s $TBSD && sweep tbsd_s$s
done
# Equal GPU-hour baseline: TBSD took 0.40 h vs 0.22 h for 120 epochs -> ~216 epochs.
train base_e216_s0 --seed 0 --epochs 216 && sweep base_e216_s0 --modes merge
# Multi-budget training without the distillation term (isolates the distillation loss).
train tbsd_nosd_s0 --seed 0 $TBSD --sd-weight 0 && sweep tbsd_nosd_s0 --modes merge
echo "=== all done $(date)"
