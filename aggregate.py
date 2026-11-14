"""Aggregate sweeps over seeds and compare against controls.

  python aggregate.py   -> prints markdown tables, writes runs/iter1_seeds.csv and runs/iter1_seeds.png
"""
import csv
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GROUPS = {"baseline": ["runs/base_s0", "runs/base_s1", "runs/base_s2"],
          "TBSD": ["runs/tbsd_s0", "runs/tbsd_s1", "runs/tbsd_s2"]}
CONTROLS = {"baseline 216 ep (equal GPU-h)": "runs/base_e216_s0",
            "TBSD w/o distillation (sd-weight 0)": "runs/tbsd_nosd_s0"}
KEYS = ("sym_mAP", "asym_mAP", "q_cos")


def sweep(run):
    p = os.path.join(run, "sweep.csv")
    if not os.path.exists(p):
        return None
    return {(r["mode"], float(r["keep"])): r for r in csv.DictReader(open(p))}


def hours(run):
    p = os.path.join(run, "log.json")
    return json.load(open(p))["epochs"][-1]["train_hours"] if os.path.exists(p) else float("nan")


def ms(vals):
    return (st.mean(vals), st.stdev(vals) if len(vals) > 1 else 0.0)


data = {g: [s for s in map(sweep, runs) if s] for g, runs in GROUPS.items()}
cells = sorted({k for g in data.values() for s in g for k in s}, key=lambda k: (k[0] != "none", k[0], -k[1]))
out = []
print("| mode | keep | n | " + " | ".join(f"{g} {k}" for k in KEYS for g in GROUPS) + " |")
print("|" + " --- |" * (3 + len(KEYS) * len(GROUPS)))
for c in cells:
    row = {"mode": c[0], "keep": c[1]}
    parts = []
    for k in KEYS:
        for g in GROUPS:
            vals = [float(s[c][k]) for s in data[g] if c in s]
            m, sd = ms(vals) if vals else (float("nan"), 0)
            row[f"{g}_{k}_mean"], row[f"{g}_{k}_std"], row[f"{g}_n"] = m, sd, len(vals)
            parts.append(f"{m:.3f}±{sd:.3f}" if k == "q_cos" else f"{m:.2f}±{sd:.2f}")
    out.append(row)
    print(f"| {c[0]} | {c[1]} | {min(row[f'{g}_n'] for g in GROUPS)} | " + " | ".join(parts) + " |")

print("\nTraining GPU-hours: " + ", ".join(
    f"{g} {st.mean(hours(r) for r in runs if os.path.exists(r)):.2f}" for g, runs in GROUPS.items()))

print("\nControls (seed 0, merge rule): asym mAP / sym mAP, train hours")
for name, run in CONTROLS.items():
    s = sweep(run)
    if s is None:
        print(f"  {name}: not run yet")
        continue
    pts = ", ".join(f"keep {k[1]}: {float(s[k]['asym_mAP']):.2f}/{float(s[k]['sym_mAP']):.2f}"
                    for k in sorted(s, key=lambda k: -k[1]) if k[0] in ("none", "merge"))
    print(f"  {name} ({hours(run):.2f} h): {pts}")

with open("runs/iter1_seeds.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)

fig, ax = plt.subplots(figsize=(6.5, 4.2))
styles = {"baseline": "#888888", "TBSD": "#1f6fb4"}
for g, color in styles.items():
    rows = sorted((r for r in out if r["mode"] in ("none", "merge")), key=lambda r: r["keep"])
    keeps = [r["keep"] for r in rows]
    gfl = [float(data[g][0][("none", 1.0) if r["mode"] == "none" else ("merge", r["keep"])]["gflops"]) for r in rows]
    for k, ls in (("asym_mAP", "o-"), ("sym_mAP", "s--")):
        ax.errorbar(gfl, [r[f"{g}_{k}_mean"] for r in rows], yerr=[r[f"{g}_{k}_std"] for r in rows],
                    fmt=ls, color=color, capsize=3, alpha=1 if k == "asym_mAP" else 0.6,
                    label=f"{g}, {'full-token' if k == 'asym_mAP' else 'reduced'} gallery")
for name, run, mk in ((*list(CONTROLS.items())[0], "^"), (*list(CONTROLS.items())[1], "v")):
    s = sweep(run)
    if s:
        pts = sorted((float(r["gflops"]), float(r["asym_mAP"])) for k, r in s.items() if k[0] in ("none", "merge"))
        ax.plot(*zip(*pts), mk + ":", label=name + ", full-token gallery", alpha=0.8)
n = min(len(v) for v in data.values())
ax.set_title(f"Market-1501, ViT-S/16, token merging; mean ± std over {n} seed(s)", fontsize=9)
ax.set_xlabel("Query-side backbone GFLOPs (analytic)")
ax.set_ylabel("mAP (%)")
ax.grid(alpha=0.3)
ax.legend(fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig("runs/iter1_seeds.png", dpi=150)
print("\nsaved runs/iter1_seeds.csv, runs/iter1_seeds.png")
