"""Aggregate adaptive.py output over seeds; compare each cascade point with the fixed-budget curve at equal cost.

  python aggregate_adaptive.py runs/adaptive.csv runs/adaptive_fg.csv --out runs/adaptive_agg.md --png runs/adaptive.png
gain = cascade mAP - fixed-budget mAP of the same model linearly interpolated at the cascade's average GFLOPs
(fixed curve: 128x64, 160x80, 192x96, 224x112, full). Cascade points above 2.93 GFLOPs have no fixed
counterpart (gain left empty).
"""
import argparse
import csv
import re
from collections import defaultdict

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("csvs", nargs="+")
p.add_argument("--out", default="runs/adaptive_agg.md")
p.add_argument("--png", default="runs/adaptive.png")
args = p.parse_args()

rows = [r for f in args.csvs for r in csv.DictReader(open(f))]
for r in rows:
    r["group"] = re.sub(r"_s\d+$", "", r["ckpt"])
    r["gflops"], r["mAP"], r["p"] = float(r["gflops"]), float(r["mAP"]), float(r["p"])

# fixed curves per checkpoint
fixed = defaultdict(list)
for r in rows:
    if r["kind"] == "fixed":
        fixed[r["ckpt"]].append((r["gflops"], r["mAP"]))
for k in fixed:
    fixed[k].sort()


def fixed_at(ckpt, g):
    xs, ys = zip(*fixed[ckpt])
    return float(np.interp(g, xs, ys)) if g <= xs[-1] + 1e-9 else None


agg = defaultdict(list)
for r in rows:
    if r["kind"] != "cascade":
        continue
    f = fixed_at(r["ckpt"], r["gflops"])
    agg[(r["group"], r["cheap"], r["signal"], r["p"])].append((r["gflops"], r["mAP"], None if f is None else r["mAP"] - f))

fx = defaultdict(list)
for r in rows:
    if r["kind"] == "fixed":
        fx[(r["group"], r["cheap"])].append((r["gflops"], r["mAP"]))

out = ["# Per-query cascade, aggregated over seeds (mean ± std)", "",
       "## Fixed budgets (every query at one setting)", "",
       "| model | setting | n | GFLOPs | mAP |", "| --- | --- | --- | --- | --- |"]
for (g, c), v in sorted(fx.items(), key=lambda kv: (kv[0][0], -np.mean([x[0] for x in kv[1]]))):
    a = np.array(v)
    out.append(f"| {g} | {c} | {len(a)} | {a[:, 0].mean():.2f} | {a[:, 1].mean():.2f} ± {a[:, 1].std(ddof=1) if len(a) > 1 else 0:.2f} |")
out += ["", "## Cascades: cheap stage, signal, escalated fraction p", "",
        "gain = cascade mAP minus fixed-budget mAP at the same average GFLOPs (interpolated), mean ± std over seeds.", "",
        "| model | cheap | signal | p | GFLOPs | mAP | gain vs fixed |", "| --- | --- | --- | --- | --- | --- | --- |"]
for key in sorted(agg):
    a = agg[key]
    g = np.mean([x[0] for x in a])
    m = np.array([x[1] for x in a])
    gains = [x[2] for x in a if x[2] is not None]
    gs = (f"{np.mean(gains):+.2f} ± {np.std(gains, ddof=1) if len(gains) > 1 else 0:.2f}" if len(gains) == len(a) else "")
    out.append(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]:.2f} | {g:.2f} | {m.mean():.2f} ± "
               f"{m.std(ddof=1) if len(m) > 1 else 0:.2f} | {gs} |")
open(args.out, "w", encoding="utf-8").write("\n".join(out) + "\n")
print("\n".join(out[:4]))
print(f"saved {args.out}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = sorted({k[0] for k in agg})
    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4), squeeze=False)
    styles = {"top1": ("C0", "-", "cascade, top-1 signal"), "random": ("C7", ":", "cascade, random"),
              "oracle": ("C3", "--", "cascade, oracle (upper bound)")}
    for ax, grp in zip(axes[0], groups):
        pts = sorted((np.mean([x[0] for x in v]), np.mean([x[1] for x in v])) for (g, c), v in fx.items() if g == grp)
        ax.plot(*zip(*pts), "k-o", label="fixed budget (all queries)")
        for sig, (col, ls, lab) in styles.items():
            for cheap, mk in (("128x64", "s"), ("160x80", "^")):
                ks = sorted(k for k in agg if k[0] == grp and k[1] == cheap and k[2] == sig)
                xy = [(np.mean([x[0] for x in agg[k]]), np.mean([x[1] for x in agg[k]])) for k in ks]
                ax.plot(*zip(*xy), color=col, ls=ls, marker=mk, ms=3,
                        label=f"{lab}, from {cheap}" if sig == "top1" or cheap == "128x64" else None)
        ax.set_title(grp)
        ax.set_xlabel("average query GFLOPs")
        ax.set_ylabel("asymmetric mAP (%)")
        ax.set_xlim(0.6, 3.0)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.png, dpi=130)
    print(f"saved {args.png}")
except Exception as e:  # the table is the result; the figure is optional
    print("figure skipped:", e)
