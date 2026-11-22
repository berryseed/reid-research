"""Aggregate runs/mech_diag/mech_diag.csv over seeds; tables (markdown) and figure runs/mech_diag/mech_diag.png.

Usage: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe aggregate_mech_diag.py
"""
import csv
import glob
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC, OUT_MD, OUT_PNG = "runs/mech_diag/mech_diag.csv", "runs/mech_diag/agg.md", "runs/mech_diag/mech_diag.png"
GROUPS = ["base_e216", "sp_lowres", "sp_merge_r", "sp_mixed", "tbsd_nosd", "tbsd"]
BUDGETS = ["full", "lowres192x96", "merge_r11", "lowres160x80", "lowres128x64"]
COLS = ["gflops", "asym_mAP", "sym_mAP", "compat_loss", "asym_R1", "sym_R1", "q_cos", "g_cos", "pos_mean",
        "neg_mean", "dprime", "s_pos_max", "s_neg_max", "margin", "top1_broken", "top1_fixed"]
# reference palette (dataviz skill), light mode: slot 1 blue, slot 2 orange
C_BASE, C_SP = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

rows = list(csv.DictReader(open(SRC)))
agg = defaultdict(lambda: defaultdict(list))
for r in rows:
    g = r["ckpt"].rsplit("_s", 1)[0]
    for c in COLS:
        agg[(g, r["budget"])][c].append(float(r[c]))


def ms(v, nd=2):
    v = np.asarray(v, float)
    return f"{v.mean():.{nd}f} ± {v.std(ddof=1):.{nd}f}" if len(v) > 1 else f"{v[0]:.{nd}f}"


out = ["# Mechanism diagnostics (evaluation only)", "", f"Source `{SRC}`; mean ± sample std over seeds.", ""]
out += ["## A. Asymmetric vs same-budget (symmetric) retrieval", "",
        "| model | budget | GFLOPs | asym mAP | sym mAP | sym − asym | asym R1 | sym R1 | q_cos | g_cos |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
for g in GROUPS:
    for b in BUDGETS:
        d = agg.get((g, b))
        if not d:
            continue
        out.append(f"| {g} | {b} | {np.mean(d['gflops']):.2f} | {ms(d['asym_mAP'])} | {ms(d['sym_mAP'])} | "
                   f"{ms(d['compat_loss'])} | {ms(d['asym_R1'])} | {ms(d['sym_R1'])} | {ms(d['q_cos'], 3)} | "
                   f"{ms(d['g_cos'], 3)} |")
out += ["", "## B. Similarity structure of the asymmetric setting (cheap query vs full gallery)", "",
        "| model | budget | mean pos sim | mean neg sim | d′ | mean best-pos | mean hardest-neg | mean margin | top-1 broken | top-1 fixed |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
for g in GROUPS:
    for b in BUDGETS:
        d = agg.get((g, b))
        if not d:
            continue
        out.append(f"| {g} | {b} | {ms(d['pos_mean'], 3)} | {ms(d['neg_mean'], 3)} | {ms(d['dprime'])} | "
                   f"{ms(d['s_pos_max'], 3)} | {ms(d['s_neg_max'], 3)} | {ms(d['margin'], 3)} | "
                   f"{ms(d['top1_broken'], 0)} | {ms(d['top1_fixed'], 0)} |")
open(OUT_MD, "w", encoding="utf-8").write("\n".join(out) + "\n")
print("\n".join(out))

# ---- figure: (a) asym vs sym mAP over budgets; (b) margin histograms at 160x80 (seed 0) ----
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), facecolor="#fcfcfb")
lowres = ["lowres128x64", "lowres160x80", "lowres192x96", "full"]
for g, col, lab in (("base_e216", C_BASE, "baseline (216 ep)"), ("sp_mixed", C_SP, "sp_mixed")):
    x = [np.mean(agg[(g, b)]["gflops"]) for b in lowres]
    for key, ls, mk, suffix in (("asym_mAP", "-", "o", "cheap query vs full gallery"),
                                ("sym_mAP", "--", "s", "cheap query vs cheap gallery")):
        y = [np.mean(agg[(g, b)][key]) for b in lowres]
        ax1.plot(x, y, ls, color=col, lw=2, marker=mk, ms=8, mec="#fcfcfb", mew=2, label=f"{lab}: {suffix}")
ax1.set_xlabel("query GFLOPs (input resolution 128x64 → 256x128)")
ax1.set_ylabel("mAP (%)")
ax1.set_title("Asymmetric vs same-budget retrieval (3 seeds)", color=INK, loc="left")
ax1.grid(axis="y", color=GRID, lw=0.8)
ax1.legend(frameon=False, fontsize=7.5, loc="lower right", handlelength=4)

bins = np.linspace(-0.4, 0.6, 51)
for g, col, lab in (("base_e216", C_BASE, "baseline (216 ep)"), ("sp_mixed", C_SP, "sp_mixed")):
    f = glob.glob(f"runs/mech_diag/mech_diag_{g}_s0_lowres160x80.npz")
    if not f:
        continue
    z = np.load(f[0])
    m = z["s_pos_max"] - z["s_neg_max"]
    ax2.hist(m, bins=bins, histtype="step", lw=2, color=col,
             label=f"{lab}: {100 * (m > 0).mean():.1f} % > 0")
ax2.axvline(0, color=INK2, lw=1, ls=":")
ax2.set_xlabel("margin = best same-ID cos − hardest other-ID cos\n(160x80 query, full-resolution gallery)")
ax2.set_ylabel("queries")
ax2.set_title("Per-query margin, seed 0 (> 0 ⇔ rank-1 correct)", color=INK, loc="left")
ax2.grid(axis="y", color=GRID, lw=0.8)
ax2.legend(frameon=False, fontsize=7.5, loc="upper left", handlelength=3)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150, facecolor=fig.get_facecolor())
print("saved", OUT_PNG)
