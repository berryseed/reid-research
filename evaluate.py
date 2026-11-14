"""Sweep token reduction on a trained checkpoint and report accuracy against cost.

For each (mode, keep ratio) it reports
  sym_*    query and gallery both encoded with reduced tokens
  asym_*   gallery encoded once with full tokens, query encoded with reduced tokens
  q_cos    mean cosine similarity between reduced-token and full-token query embeddings
  gflops   analytic estimate for the backbone,  img_s  measured forward throughput

Examples
  python evaluate.py --data D:/data/Market-1501-v15.09.15 --ckpt runs/vits_base/last.pth --out runs/vits_base/sweep.csv
  Cross-dataset (model trained on Market, tested on another Market-format dataset):
  python evaluate.py --data D:/data/OtherDataset --ckpt runs/vits_base/last.pth --out runs/vits_base/sweep_other.csv
"""
import argparse
import csv

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import (ImageSet, ReIDViT, build_transforms, evaluate_rank, extract_features,
                    measure_throughput, parse_market_dir, vit_gflops)
import os


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="sweep.csv")
    p.add_argument("--modes", nargs="+", default=["merge", "cls_attn", "random"])
    p.add_argument("--keeps", nargs="+", type=float, default=[0.9, 0.8, 0.7, 0.6, 0.5])
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cpu", action="store_true", help="run everything on CPU (also gives CPU throughput)")
    p.add_argument("--tp-batch", type=int, default=None,
                   help="batch for throughput timing (default 256 GPU, 8 CPU; at 64 a ViT-S on a 5080 is "
                        "launch-overhead bound and token reduction shows little speed-up)")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    a = ck["args"]
    img_size = (a["height"], a["width"])
    model = ReIDViT(a["model"], ck["num_classes"], img_size, pretrained=False)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    cfg = model.vit.pretrained_cfg
    _, test_tf = build_transforms(img_size, cfg.get("mean", (0.5,) * 3), cfg.get("std", (0.5,) * 3))
    kw = dict(batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    q_loader = DataLoader(ImageSet(parse_market_dir(os.path.join(args.data, "query")), test_tf), **kw)
    g_loader = DataLoader(ImageSet(parse_market_dir(os.path.join(args.data, "bounding_box_test")), test_tf), **kw)
    tp_batch = args.tp_batch or (256 if device.type == "cuda" else 8)

    rows = []

    def record(mode, keep, qf, gf, qf_full, gf_full, meta):
        qp, gp, qc, gc = meta
        sym_map, sym_cmc = evaluate_rank(qf, gf, qp, gp, qc, gc, device)
        asym_map, asym_cmc = evaluate_rank(qf, gf_full, qp, gp, qc, gc, device)
        row = {
            "mode": mode, "keep": keep,
            "final_tokens": model.last_counts[-1][1],
            "gflops": round(vit_gflops(model), 3),
            "img_s": round(measure_throughput(model, device, img_size, mode, keep, batch=tp_batch), 1),
            "sym_mAP": round(sym_map, 2), "sym_R1": round(float(sym_cmc[0]), 2),
            "asym_mAP": round(asym_map, 2), "asym_R1": round(float(asym_cmc[0]), 2),
            "q_cos": round(float(F.cosine_similarity(qf, qf_full, dim=1).mean()), 4),
        }
        rows.append(row)
        print(row, flush=True)

    # full-token reference
    gf_full, gp, gc = extract_features(model, g_loader, device)
    qf_full, qp, qc = extract_features(model, q_loader, device)
    meta = (qp, gp, qc, gc)
    record("none", 1.0, qf_full, gf_full, qf_full, gf_full, meta)

    for mode in args.modes:
        for keep in args.keeps:
            torch.manual_seed(0)  # makes the 'random' control repeatable
            gf, _, _ = extract_features(model, g_loader, device, mode, keep)
            qf, _, _ = extract_features(model, q_loader, device, mode, keep)
            record(mode, keep, qf, gf, qf_full, gf_full, meta)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {args.out}")

    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed, skipping plot")
            return
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, key, title in zip(axes, ("sym_mAP", "asym_mAP"),
                                  ("Symmetric: query and gallery reduced", "Asymmetric: full gallery, reduced query")):
            for mode in args.modes:
                pts = sorted((r["gflops"], r[key]) for r in rows if r["mode"] in (mode, "none"))
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=mode)
            ax.set_title(title)
            ax.set_xlabel("Query-side GFLOPs (analytic estimate)")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("mAP (%)")
        axes[0].legend(title="reduction")
        fig.tight_layout()
        png = os.path.splitext(args.out)[0] + ".png"
        fig.savefig(png, dpi=150)
        print(f"saved {png}")


if __name__ == "__main__":
    main()
