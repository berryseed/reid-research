"""Iteration-2 aggregate: mean ± std over the seeds that exist for every configuration.

  python aggregate2.py  -> prints markdown tables, writes runs/iter2_summary.csv, runs/iter2_lowdata.png,
                           runs/iter2_controls.png
Only finished runs (sweep.csv present) are used; the seed count n is printed for every row.
"""
import csv
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEEDS = (0, 1, 2)
GROUPS = {  # label -> run-name pattern
    "Baseline (120 ep)": "base_s{}",
    "Baseline 216 ep (equal GPU-h)": "base_e216_s{}",
    "Fixed-budget training (merge 0.5)": "fix50_s{}",
    "Multi-budget, no distillation": "tbsd_nosd_s{}",
    "TBSD": "tbsd_s{}",
    "TBSD, BNNeck stats from full pass only": "tbsd_bnns_s{}",
}
LOWDATA = {f: {"Baseline": f"base_id{f}_s{{}}", "TBSD": f"tbsd_id{f}_s{{}}"} for f in (10, 25, 50)}
VITB = {"ViT-B baseline": "vitb_base_s{}", "ViT-B TBSD": "vitb_tbsd_s{}"}
KEEPS = (1.0, 0.8, 0.7, 0.6, 0.5)


def load(run):
    p, lg = os.path.join("runs", run, "sweep.csv"), os.path.join("runs", run, "log.json")
    if not (os.path.exists(p) and os.path.exists(lg)):
        return None
    sw = {(r["mode"], float(r["keep"])): r for r in csv.DictReader(open(p))}
    ep = json.load(open(lg))["epochs"][-1]
    return {"sweep": sw, "hours": ep["train_hours"], "vram": ep.get("peak_vram_gb", float("nan"))}


def cell(runs, keep, key):
    k = ("none", 1.0) if keep == 1.0 else ("merge", keep)
    return [float(r["sweep"][k][key]) for r in runs if k in r["sweep"]]


def fmt(vals, nd=2):
    if not vals:
        return "n/a"
    return f"{st.mean(vals):.{nd}f} ± {st.stdev(vals):.{nd}f}" if len(vals) > 1 else f"{vals[0]:.{nd}f}"


def table(title, groups, out_rows, tag):
    print(f"\n### {title}\nmAP (%), merge rule: full-token gallery / reduced gallery; mean ± std over n seeds\n")
    print("| Configuration | n | GPU-h | VRAM GB | " + " | ".join(f"keep {k}" for k in KEEPS) + " | q_cos @0.5 |")
    print("|" + " --- |" * (5 + len(KEEPS)))
    res = {}
    for label, pat in groups.items():
        runs = [r for r in (load(pat.format(s)) for s in SEEDS) if r]
        if not runs:
            print(f"| {label} | 0 | not run |" + " |" * (len(KEEPS) + 2))
            continue
        res[label] = runs
        cols = []
        for k in KEEPS:
            a, s_ = cell(runs, k, "asym_mAP"), cell(runs, k, "sym_mAP")
            cols.append(fmt(a) if k == 1.0 else f"{fmt(a)} / {fmt(s_)}")
            out_rows.append({"table": tag, "config": label, "n": len(runs), "keep": k,
                             "asym_mean": st.mean(a) if a else "", "asym_std": st.stdev(a) if len(a) > 1 else 0,
                             "sym_mean": st.mean(s_) if s_ else "", "sym_std": st.stdev(s_) if len(s_) > 1 else 0,
                             "gpu_h": st.mean(r["hours"] for r in runs)})
        print(f"| {label} | {len(runs)} | {st.mean(r['hours'] for r in runs):.2f} | "
              f"{max(r['vram'] for r in runs):.2f} | " + " | ".join(cols) + f" | {fmt(cell(runs, 0.5, 'q_cos'), 3)} |")
    return res


rows = []
main = table("Market-1501, ViT-S/16, all identities", GROUPS, rows, "main")
low = {f: table(f"Low data: {f}% of identities", g, rows, f"id{f}") for f, g in LOWDATA.items()}
vitb = table("ViT-B/16 reference (seed 0)", VITB, rows, "vitb")

if main.get("TBSD") and main.get("Multi-budget, no distillation"):
    print("\n### Distillation term: paired per-seed difference (TBSD minus no-distillation), full-token gallery mAP")
    for k in KEEPS:
        a, b = cell(main["TBSD"], k, "asym_mAP"), cell(main["Multi-budget, no distillation"], k, "asym_mAP")
        n = min(len(a), len(b))
        d = [a[i] - b[i] for i in range(n)]
        print(f"- keep {k}: " + ", ".join(f"{x:+.2f}" for x in d) + f"  (mean {st.mean(d):+.2f})")

with open("runs/iter2_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

# figure 1: controls, full-token-gallery mAP against query GFLOPs
fig, ax = plt.subplots(figsize=(6.8, 4.4))
ref = next(iter(main.values()))[0]["sweep"]
gfl = [float(ref[("none", 1.0) if k == 1.0 else ("merge", k)]["gflops"]) for k in KEEPS]
for label, runs in main.items():
    m = [st.mean(cell(runs, k, "asym_mAP")) for k in KEEPS]
    e = [st.stdev(cell(runs, k, "asym_mAP")) if len(runs) > 1 else 0 for k in KEEPS]
    ax.errorbar(gfl, m, yerr=e, marker="o", capsize=3, label=f"{label} (n={len(runs)})")
ax.set_xlabel("Query-side backbone GFLOPs (analytic)")
ax.set_ylabel("mAP (%), full-token gallery")
ax.set_title("Market-1501, ViT-S/16, token merging: method and controls", fontsize=9)
ax.grid(alpha=0.3)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig("runs/iter2_controls.png", dpi=150)

# figure 2: low data
fracs = [f for f in (10, 25, 50) if low[f]] + ([100] if main.get("TBSD") else [])
if len(fracs) > 1:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, keep, title in zip(axes, (1.0, 0.5), ("Full tokens (2.93 GFLOPs)", "Merge keep 0.5 query (1.46 GFLOPs), full-token gallery")):
        for name, color in (("Baseline", "#888888"), ("TBSD", "#1f6fb4")):
            xs, ms, es = [], [], []
            for f in fracs:
                runs = (main.get("Baseline (120 ep)" if name == "Baseline" else "TBSD") if f == 100 else low[f].get(name))
                if not runs:
                    continue
                v = cell(runs, keep, "asym_mAP")
                xs.append(f); ms.append(st.mean(v)); es.append(st.stdev(v) if len(v) > 1 else 0)
            ax.errorbar(xs, ms, yerr=es, marker="o", capsize=3, color=color, label=name)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Training identities used (%)")
        ax.set_xscale("log"); ax.set_xticks(fracs); ax.set_xticklabels([str(f) for f in fracs])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mAP (%)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig("runs/iter2_lowdata.png", dpi=150)
print("\nsaved runs/iter2_summary.csv, runs/iter2_controls.png" + (", runs/iter2_lowdata.png" if len(fracs) > 1 else ""))
