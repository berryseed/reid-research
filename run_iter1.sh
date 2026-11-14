#!/usr/bin/env bash
# Iteration-1 run queue (PROJECT_CONTEXT.md §10 A, steps 4-8). Skips any run whose last.pth / sweep.csv exists.
# Usage (Git Bash):  bash run_iter1.sh /c/data/Market-1501-v15.09.15
set -u
DATA="${1:?pass the Market-1501 root (contains bounding_box_train/, query/, bounding_box_test/)}"
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1

train() {  # name, extra args...
  local name=$1; shift
  if [ -f "runs/$name/last.pth" ]; then echo "skip train $name"; return; fi
  mkdir -p "runs/$name"
  echo "=== train $name $(date)"
  $PY train.py --data "$DATA" --out "runs/$name" --seed 0 --resume "$@" >> "runs/$name/stdout.txt" 2>&1
  local rc=$?
  if [ $rc -eq 3 ]; then echo "PAUSED $name (resume: rerun this script)"; exit 3; fi
  [ $rc -eq 0 ] || { echo "FAILED train $name (see runs/$name/stdout.txt)"; return 1; }
  tail -2 "runs/$name/stdout.txt"
}
sweep() {
  local name=$1
  if [ -f "runs/$name/sweep.csv" ]; then echo "skip sweep $name"; return; fi
  echo "=== sweep $name $(date)"
  $PY evaluate.py --data "$DATA" --ckpt "runs/$name/last.pth" --out "runs/$name/sweep.csv" \
    > "runs/$name/sweep_stdout.txt" 2>&1 || { echo "FAILED sweep $name"; return 1; }
}

train base_s0 && sweep base_s0
train tbsd_s0 --sd-mode merge --sd-keep random && sweep tbsd_s0
train base_id25_s0 --id-fraction 0.25 && sweep base_id25_s0
train tbsd_id25_s0 --id-fraction 0.25 --sd-mode merge --sd-keep random && sweep tbsd_id25_s0
echo "=== all done $(date)"
