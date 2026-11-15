#!/usr/bin/env bash
# Iteration 3: which "cheap query" knob should the second training pass use? Finished runs are skipped;
# interrupted runs resume from ckpt.pth. Every finished run is committed and pushed.
# Usage (Git Bash): bash run_iter3.sh /c/Users/user/data/Market-1501-v15.09.15 M E D >> runs/queue_iter3.txt 2>&1
#   M  mechanism diagnostics (eval only) on the remaining iteration-2 controls
#   E  second-pass variants, 3 seeds each: lowres (smaller image), merge_r (gradual ToMe), mixed (either)
#   D  ViT-B/16 reference: baseline and TBSD (3-stage merge), seed 0; ViT-B latency
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
mech() {  # name: asymmetric diagnostics (merge, gradual ToMe, selection rules, low resolution, corrections)
  local name=$1
  if [ -f "runs/$name/mech.csv" ]; then echo "skip mech $name"; return; fi
  echo "=== mech $name $(date)"
  $PY mechanism.py --data "$DATA" --ckpts "runs/$name/last.pth" --out "runs/$name/mech.csv" \
    > "runs/$name/mech_stdout.txt" 2>&1 || { echo "FAILED mech $name"; tail -5 "runs/$name/mech_stdout.txt"; return 1; }
}
sweep() {  # name, csv basename, extra evaluate args...
  local name=$1 out=$2; shift 2
  if [ -f "runs/$name/$out.csv" ]; then echo "skip $out $name"; else
    echo "=== $out $name $(date)"
    $PY evaluate.py --data "$DATA" --ckpt "runs/$name/last.pth" --out "runs/$name/$out.csv" "$@" \
      > "runs/$name/${out}_stdout.txt" 2>&1 || { echo "FAILED $out $name"; tail -5 "runs/$name/${out}_stdout.txt"; return 1; }
  fi
  save "$name"
}
save() {  # commit + push the metrics of a finished run so a power cut loses at most the run in progress
  git add "runs/$1" runs/queue_iter3.txt runs/latency*.csv runs/latency*_stdout.txt 2>/dev/null
  git commit -q -m "Results: $1 (auto-commit from run_iter3.sh after the run and its evaluation finished)" \
    && git push -q && echo "pushed $1" || echo "WARN: commit/push of $1 did not complete (will be picked up later)"
}

for phase in "$@"; do
  echo "##### phase $phase $(date)"
  case $phase in
  M)
    for r in base_e216_s0 base_e216_s1 base_e216_s2 tbsd_nosd_s0 tbsd_nosd_s1 tbsd_nosd_s2; do
      mech $r && save $r
    done
    ;;
  E)
    for s in 0 1 2; do for m in lowres merge_r mixed; do
      n=sp_${m}_s$s
      train $n --seed $s --sd-mode $m && mech $n && sweep $n sweep --modes merge
    done; done
    ;;
  D)
    VB="--model vit_base_patch16_224"
    train vitb_base_s0 --seed 0 $VB && sweep vitb_base_s0 sweep --modes merge
    train vitb_tbsd_s0 --seed 0 $VB --sd-mode merge --sd-keep random && sweep vitb_tbsd_s0 sweep --modes merge
    [ -f runs/latency_vitb.csv ] || $PY bench_latency.py $VB --out runs/latency_vitb.csv > runs/latency_vitb_stdout.txt 2>&1 || echo "FAILED latency vitb"
    ;;
  *) echo "unknown phase $phase";;
  esac
done
echo "=== all done $(date)"
