"""Summary figure for iteration 1: baseline vs TBSD, mAP against query-side GFLOPs (merge rule).

  python plot_results.py            -> runs/iter1_summary.png
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAIRS = [("Market-1501, all 751 IDs", "runs/base_s0", "runs/tbsd_s0"),
         ("Market-1501, 25% of IDs (188)", "runs/base_id25_s0", "runs/tbsd_id25_s0")]


def load(run, mode="merge"):
    rows = list(csv.DictReader(open(os.path.join(run, "sweep.csv"))))
    rows = [r for r in rows if r["mode"] in ("none", mode)]
    return sorted(rows, key=lambda r: float(r["gflops"]))


fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, (title, base, tbsd) in zip(axes, PAIRS):
    for run, name, color in ((base, "baseline", "#888888"), (tbsd, "TBSD (ours)", "#1f6fb4")):
        rows = load(run)
        g = [float(r["gflops"]) for r in rows]
        ax.plot(g, [float(r["asym_mAP"]) for r in rows], "o-", color=color, label=f"{name}, full-token gallery")
        ax.plot(g, [float(r["sym_mAP"]) for r in rows], "s--", color=color, alpha=0.6, label=f"{name}, reduced gallery")
    ax.set_title(title)
    ax.set_xlabel("Query-side backbone GFLOPs (analytic; token merging)")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("mAP (%)")
axes[0].legend(fontsize=8, loc="lower right")
fig.suptitle("ViT-S/16, one seed (seed 0). Single set of weights evaluated at several token budgets.", fontsize=9)
fig.tight_layout()
fig.savefig("runs/iter1_summary.png", dpi=150)
print("saved runs/iter1_summary.png")
