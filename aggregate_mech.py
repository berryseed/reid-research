"""Mean ± std over seeds of mechanism.py output, per group (run-name prefix before _s<seed>).

Run: python aggregate_mech.py [runs/mechanism.csv]
"""
import csv
import re
import sys
from collections import defaultdict

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "runs/mechanism.csv"
cols = ["q_cos", "offset_frac", "asym_raw", "asym_mean", "asym_affine", "asym_ridge"]
data, gfl, order = defaultdict(list), {}, []
for r in csv.DictReader(open(path)):
    group = re.sub(r"_s\d+$", "", r["ckpt"])
    setting = re.sub(r"\(r=\d+\)", "", r["setting"])  # r may differ by one between checkpoints
    key = (group, setting)
    if key not in data:
        order.append(key)
    data[key].append([float(r[c]) for c in cols])
    gfl.setdefault(key, []).append(float(r["gflops"]))

print("| group | setting | n | GFLOPs | " + " | ".join(cols) + " |")
print("| --- | --- | --- | --- | " + " | ".join("---" for _ in cols) + " |")
for key in order:
    a = np.array(data[key])
    m, s = a.mean(0), a.std(0, ddof=1) if len(a) > 1 else np.zeros(a.shape[1])
    cells = [f"{m[i]:.3f}" if i < 2 else f"{m[i]:.2f} ± {s[i]:.2f}" for i in range(len(cols))]
    print(f"| {key[0]} | {key[1]} | {len(a)} | {np.mean(gfl[key]):.2f} | " + " | ".join(cells) + " |")
