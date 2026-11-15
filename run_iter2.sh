#!/usr/bin/env bash
# Iteration 2: remaining Market-1501 experiments. Finished runs are skipped; interrupted runs resume.
# Usage (Git Bash): bash run_iter2.sh /c/Users/user/data/Market-1501-v15.09.15 S A B C D
#   S  selection diagnostics (eval only) + latency benchmark
#   A  no-distillation control seeds 1-2; BNNeck running-stat ablation (--sd-bn nostat), 3 seeds
#   B  fixed-budget training control (merge keep 0.5), 3 seeds; equal-GPU-hour baseline seeds 1-2
#   C  low-data: 10/25/50% of identities, baseline and TBSD, 3 seeds
#   D  ViT-B/16 reference: baseline and TBSD, seed 0; ViT-B latency
set -u
DATA="${1:?pass the Market-1501 root}"; shift
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 PYTHONUTF8=1

train() {  # name, args...
  local name=$1; shift
  if [ -f "runs/$name/last.pth" ]; then echo "skip train $name"; return; fi
  mkdir -p "runs/$name"
  echo "=== train $name $(date)"
  $PY train.py --data "$DATA" --out "runs/$name" --resume "$@" >> "runs/$name/stdout.txt" 2>&1
  local rc=$?
  if [ $rc -eq 3 ]; then echo "PAUSED $name (resume: rerun this script)"; exit 3; fi
  [ $rc -eq 0 ] || { echo "FAILED train $name (rc $rc)"; tail -5 "runs/$name/stdout.txt"; return 1; }
  tail -2 "runs/$name/stdout.txt"
}
sweep() {  # name, csv basename, extra evaluate args...
  local name=$1 out=$2; shift 2
  if [ -f "runs/$name/$out.csv" ]; then echo "skip $out $name"; return; fi
  echo "=== $out $name $(date)"
  $PY evaluate.py --data "$DATA" --ckpt "runs/$name/last.pth" --out "runs/$name/$out.csv" "$@" \
    > "runs/$name/${out}_stdout.txt" 2>&1 || { echo "FAILED $out $name"; tail -5 "runs/$name/${out}_stdout.txt"; return 1; }
  save "$name"
}
save() {  # commit + push the metrics of a finished run so a power cut loses at most the run in progress
  git add "runs/$1" runs/queue_iter2.txt runs/latency*.csv runs/latency*_stdout.txt 2>/dev/null
  git commit -q -m "Results: $1 (auto-commit from run_iter2.sh after the run and its sweep finished)"     && git push -q && echo "pushed $1" || echo "WARN: commit/push of $1 did not complete (will be picked up later)"
}

TBSD="--sd-mode merge --sd-keep random"
for phase in "$@"; do
  echo "##### phase $phase $(date)"
  case $phase in
  S)
    for r in base_s0 base_s1 base_s2 tbsd_s0 tbsd_s1 tbsd_s2; do
      sweep $r sweep_sel --modes cls_attn cls_attn_fuse cls_attn_low random --keeps 0.8 0.7 0.5
    done
    [ -f runs/latency.csv ] || $PY bench_latency.py --out runs/latency.csv > runs/latency_stdout.txt 2>&1 || echo "FAILED latency"
    ;;
  A)
    for s in 1 2; do train tbsd_nosd_s$s --seed $s $TBSD --sd-weight 0 && sweep tbsd_nosd_s$s sweep --modes merge; done
    for s in 0 1 2; do train tbsd_bnns_s$s --seed $s $TBSD --sd-bn nostat && sweep tbsd_bnns_s$s sweep --modes merge; done
    ;;
  B)
    for s in 0 1 2; do train fix50_s$s --seed $s --train-mode merge --train-keep 0.5 && sweep fix50_s$s sweep --modes merge; done
    for s in 1 2; do train base_e216_s$s --seed $s --epochs 216 && sweep base_e216_s$s sweep --modes merge; done
    ;;
  C)
    for f in 25 50 10; do for s in 0 1 2; do
      train base_id${f}_s$s --seed $s --id-fraction 0.$f && sweep base_id${f}_s$s sweep --modes merge
      train tbsd_id${f}_s$s --seed $s --id-fraction 0.$f $TBSD && sweep tbsd_id${f}_s$s sweep --modes merge
    done; done
    ;;
  D)
    VB="--model vit_base_patch16_224"
    train vitb_base_s0 --seed 0 $VB && sweep vitb_base_s0 sweep --modes merge
    train vitb_tbsd_s0 --seed 0 $VB $TBSD && sweep vitb_tbsd_s0 sweep --modes merge
    [ -f runs/latency_vitb.csv ] || $PY bench_latency.py $VB --out runs/latency_vitb.csv > runs/latency_vitb_stdout.txt 2>&1 || echo "FAILED latency vitb"
    ;;
  *) echo "unknown phase $phase";;
  esac
done
echo "=== all done $(date)"
