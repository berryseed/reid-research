#!/usr/bin/env bash
# Iteration 4, phase F: frozen-gallery student baseline (in the style of Suma & Tolias, WACV 2023).
# Teacher = the finished 120-epoch baseline of the same seed (frozen; it also encodes the gallery).
# Student = a copy of it, trained WITHOUT labels for 96 epochs so that its cheap pass (low resolution or
# gradual merging, as in sp_mixed) reproduces the teacher's full-token embedding.
# 120 + 96 = 216 epoch-equivalents, the budget of the equal-GPU-hour baseline.
# Finished runs are skipped; interrupted runs resume from ckpt.pth; metrics are committed and pushed.
# Usage (Git Bash): bash run_iter4.sh /c/Users/user/data/Market-1501-v15.09.15 >> runs/queue_iter4.txt 2>&1
set -u
DATA="${1:?pass the Market-1501 root}"
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 PYTHONUTF8=1

echo "##### phase F $(date)"
for s in 0 1 2; do
  n=fg_mixed_s$s
  t=runs/base_s$s/last.pth
  if [ ! -f "runs/$n/last.pth" ]; then
    mkdir -p "runs/$n"
    echo "=== train $n (teacher $t) $(date)"
    $PY train.py --data "$DATA" --out "runs/$n" --resume --seed $s --teacher "$t" --sd-mode mixed --epochs 96 \
      >> "runs/$n/stdout.txt" 2>&1
    rc=$?
    if [ $rc -eq 3 ]; then echo "PAUSED $n (resume: rerun this script)"; exit 3; fi
    if [ $rc -ne 0 ]; then echo "FAILED train $n (rc $rc)"; tail -5 "runs/$n/stdout.txt"; continue; fi
    tail -2 "runs/$n/stdout.txt"
  else
    echo "skip train $n"
  fi
  if [ ! -f "runs/$n/mech.csv" ]; then
    echo "=== mech $n (gallery from $t) $(date)"
    $PY mechanism.py --data "$DATA" --ckpts "runs/$n/last.pth" --gallery-ckpt "$t" --out "runs/$n/mech.csv" \
      > "runs/$n/mech_stdout.txt" 2>&1 || { echo "FAILED mech $n"; tail -5 "runs/$n/mech_stdout.txt"; continue; }
  else
    echo "skip mech $n"
  fi
  git add "runs/$n" runs/queue_iter4.txt 2>/dev/null
  git commit -q -m "Results: $n (auto-commit from run_iter4.sh after the run and its evaluation finished)" \
    && git push -q && echo "pushed $n" || echo "WARN: commit/push of $n did not complete (will be picked up later)"
done
echo "=== all done $(date)"
