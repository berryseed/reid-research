"""Aggregate runs/nonlinear_map/nonlinear_map.csv (3 seeds of the 216-epoch baseline) and compare with sp_mixed.

sp_mixed numbers are read, not recomputed: mAP and R1 of its fixed budgets from runs/adaptive.csv (E17, the same
evaluation code path). Backbone latency per query resolution from runs/latency_extra.csv.
Usage: python aggregate_nonlinear.py [--csv runs/nonlinear_map/nonlinear_map.csv] [--out runs/nonlinear_map/agg.md]
"""
import argparse
import csv
from collections import defaultdict

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--csv", default="runs/nonlinear_map/nonlinear_map.csv")
p.add_argument("--out", default="runs/nonlinear_map/agg.md")
args = p.parse_args()

rows = list(csv.DictReader(open(args.csv)))
by = defaultdict(lambda: defaultdict(list))  # (setting, map) -> field -> [per seed]
for r in rows:
    for k in ("mAP", "R1", "val_loss", "params", "train_sec", "gpu_b1_ms", "gpu_b256_ms_per_img", "cpu_b1_ms",
              "gmacs", "best_epoch", "lam"):
        if r.get(k):
            by[(r["setting"], r["map_name"])][k].append(float(r[k]))
    if r.get("config"):
        by[(r["setting"], r["map_name"])]["config"].append(r["config"])
    by[(r["setting"], r["map_name"])]["ckpt"].append(r["ckpt"])

sp = defaultdict(lambda: defaultdict(list))
for r in csv.DictReader(open("runs/adaptive.csv")):
    if r["ckpt"].startswith("sp_mixed") and r["kind"] == "fixed":
        sp["lowres" + r["cheap"]]["mAP"].append(float(r["mAP"]))
        sp["lowres" + r["cheap"]]["R1"].append(float(r["R1"]))
lat = {r["mode"]: r for r in csv.DictReader(open("runs/latency_extra.csv"))}


def ms(v):
    v = np.array(v, dtype=float)
    return f"{v.mean():.2f} ± {v.std(ddof=1):.2f}" if len(v) > 1 else (f"{v[0]:.2f}" if len(v) else "—")


out = ["# Nonlinear (VPFA-inspired) map vs linear map vs sp_mixed",
       "",
       f"Source: `{args.csv}`; sp_mixed from `runs/adaptive.csv` (fixed budgets). Mean ± sample std over 3 seeds.",
       "Baseline = 216-epoch (equal-GPU-hour) baseline, frozen. All maps fitted on training images only, no labels.",
       ""]
settings = [s for s in ("full", "lowres192x96", "lowres160x80", "lowres128x64") if any(k[0] == s for k in by)]
out += ["| Query | GFLOPs | map | mAP | R1 | VAL loss | selected |", "| --- | --- | --- | --- | --- | --- | --- |"]
g_of = {r["setting"]: r["gflops"] for r in rows}
for s in settings:
    for m in ("raw", "ridge_report", "ridge_cv", "mlp"):
        d = by.get((s, m))
        if not d or "mAP" not in d:
            continue
        sel = ""
        if m == "ridge_cv":
            sel = "lambda " + "/".join(f"{x:g}" for x in d["lam"])
        if m == "mlp":
            sel = "/".join(d["config"]) + "; epoch " + "/".join(f"{x:.0f}" for x in d["best_epoch"])
        out.append(f"| {s} | {g_of[s]} | {m} | {ms(d['mAP'])} | {ms(d['R1'])} | "
                   f"{ms(d['val_loss']) if d.get('val_loss') else '—'} | {sel} |")
    if s in sp:
        out.append(f"| {s} | {g_of[s]} | **sp_mixed (raw)** | {ms(sp[s]['mAP'])} | {ms(sp[s]['R1'])} | — | — |")

out += ["", "## Paired differences (per seed, then mean ± std)", "",
        "| Query | mlp − ridge_report (mAP) | mlp − ridge_cv (mAP) | sp_mixed − best baseline map (mAP) | sp_mixed − best map (R1) |",
        "| --- | --- | --- | --- | --- |"]
for s in settings[1:]:
    mlp, rr, rc = by[(s, "mlp")], by[(s, "ridge_report")], by[(s, "ridge_cv")]
    best = np.maximum.reduce([np.array(mlp["mAP"]), np.array(rr["mAP"]), np.array(rc["mAP"])])
    best_r1 = np.maximum.reduce([np.array(mlp["R1"]), np.array(rr["R1"]), np.array(rc["R1"])])
    d1 = np.array(mlp["mAP"]) - np.array(rr["mAP"])
    d2 = np.array(mlp["mAP"]) - np.array(rc["mAP"])
    spd = (ms(np.array(sp[s]["mAP"]) - best), ms(np.array(sp[s]["R1"]) - best_r1)) if s in sp else ("—", "—")
    out.append(f"| {s} | {ms(d1)} | {ms(d2)} | {spd[0]} | {spd[1]} |")
out += ["", "sp_mixed − best map pairs seed i of sp_mixed with seed i of the baseline (independent runs; the pairing "
        "is only a bookkeeping convenience). 'best map' = per-seed maximum over the three maps on the TEST set, "
        "an optimistic (oracle-selected) baseline.", ""]

out += ["## Cost of the map", "",
        "| Query | backbone GFLOPs | MLP GMACs | MLP params | MLP fit time (s) | MLP GPU b1 (ms) | MLP GPU b256 (ms/img) | MLP CPU b1 (ms) | backbone GPU b1 / b256 / CPU b1 (ms) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
for s in settings[1:]:
    d = by[(s, "mlp")]
    L = lat.get(s, {})
    bb = (f"{float(L['cuda_b1_ms_per_img']):.2f} / {float(L['cuda_b256_ms_per_img']):.3f} / "
          f"{float(L['cpu_b1_ms_per_img']):.2f}") if L else "—"
    out.append(f"| {s} | {g_of[s]} | {ms(d['gmacs'])} | {ms(d['params'])} | {ms(d['train_sec'])} | "
               f"{ms(d['gpu_b1_ms'])} | {np.mean(d['gpu_b256_ms_per_img']):.5f} | {ms(d['cpu_b1_ms'])} | {bb} |")
out.append("")
open(args.out, "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out))
