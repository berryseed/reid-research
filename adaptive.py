"""Per-query budget selection (preliminary, evaluation only): a two-stage cascade on existing checkpoints.

Stage 1: every query is embedded cheaply (low resolution) and searched against the full-cost gallery.
Stage 2: the fraction p of queries whose cheap search looks least reliable is re-embedded at full cost
(2.93 GFLOPs) and searched again. Average query cost = g_cheap + p * g_full (the cheap pass is already paid).

Decision signals (computed from the cheap search only, no labels, no full-cost pass):
  top1 : highest cosine similarity of the cheap query to the gallery (low = ambiguous)
  gap  : top-1 similarity minus the mean similarity of ranks 2-10
Controls: random escalation (same p), and an oracle that escalates the queries whose cheap embedding is
furthest from their own full-cost embedding (needs the full pass, so an upper bound only).
The whole curve over p is reported; no threshold is chosen on the test set.

This is an adaptation of confidence-based cascades for classification (RANet, CVPR 2020; DVT, NeurIPS 2021)
to compatible retrieval, with a retrieval-side confidence instead of a softmax score.

  python adaptive.py --data C:/Users/user/data/Market-1501-v15.09.15 --ckpts runs/sp_mixed_s0/last.pth ... \
      --out runs/adaptive.csv [--gallery-ckpts runs/base_s0/last.pth ...]
"""
import argparse
import csv
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import ImageSet, ReIDViT, build_transforms, evaluate_rank, parse_market_dir, vit_gflops
from mechanism import cache_images, embed, forward_res, gflops_res

PS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
CHEAP = ((128, 64), (160, 80))
FIXED = ((224, 112), (192, 96), (160, 80), (128, 64))


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--ckpts", nargs="+", required=True, help="query models")
    p.add_argument("--gallery-ckpts", nargs="+", default=None,
                   help="optional gallery models, one per --ckpts entry (frozen-gallery students)")
    p.add_argument("--out", default="runs/adaptive.csv")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0, help="for the random-escalation control")
    return p.parse_args()


def load(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    a = ck["args"]
    m = ReIDViT(a["model"], ck["num_classes"], (a["height"], a["width"]), pretrained=False)
    m.load_state_dict(ck["model"])
    return m.to(device).eval()


@torch.no_grad()
def signals(q, gf, device, chunk=1024):
    """top-1 similarity and top-1 minus mean of ranks 2-10, per query."""
    t1, gap = [], []
    g = gf.to(device)
    for s in range(0, q.shape[0], chunk):
        top = (q[s:s + chunk].to(device) @ g.T).topk(10, dim=1).values
        t1.append(top[:, 0].cpu())
        gap.append((top[:, 0] - top[:, 1:].mean(1)).cpu())
    return torch.cat(t1).numpy(), torch.cat(gap).numpy()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    rows, cached = [], None
    galleries = args.gallery_ckpts or [None] * len(args.ckpts)
    assert len(galleries) == len(args.ckpts)
    for path, gpath in zip(args.ckpts, galleries):
        model = load(path, device)
        ref = load(gpath, device) if gpath else model
        if cached is None:
            cfg = model.vit.pretrained_cfg
            _, tf = build_transforms((256, 128), cfg.get("mean", (0.5,) * 3), cfg.get("std", (0.5,) * 3))
            kw = dict(batch_size=args.batch, shuffle=False, num_workers=args.workers)
            q_items = parse_market_dir(os.path.join(args.data, "query"))
            g_items = parse_market_dir(os.path.join(args.data, "bounding_box_test"))
            cached = (cache_images(DataLoader(ImageSet(q_items, tf), **kw)),
                      cache_images(DataLoader(ImageSet(g_items, tf), **kw)))
            print(f"cached {cached[0][0].shape[0]} queries, {cached[1][0].shape[0]} gallery images", flush=True)
        (qx, qp, qc), (gx, gp, gc) = cached
        full_ref = lambda im: ref(im, "none", 1.0)[1]
        gf = embed(full_ref, gx, device, args.batch)                  # gallery: full cost, gallery model
        qf = embed(lambda im: model(im, "none", 1.0)[1], qx, device, args.batch)  # query model, full cost
        g_full = vit_gflops(model)
        name = os.path.basename(os.path.dirname(path))
        ev = lambda q: evaluate_rank(q, gf, qp, gp, qc, gc, device)

        def add(kind, cheap, signal, p, gflops, q):
            mAP, cmc = ev(q)
            rows.append({"ckpt": name, "kind": kind, "cheap": cheap, "signal": signal, "p": p,
                         "gflops": round(gflops, 3), "mAP": round(mAP, 2), "R1": round(float(cmc[0]), 2)})

        add("fixed", "full", "-", 1.0, g_full, qf)
        for hw in FIXED:
            g, _ = gflops_res(model, hw)
            add("fixed", f"{hw[0]}x{hw[1]}", "-", 0.0, g, embed(lambda im, hw=hw: forward_res(model, im, hw),
                                                                qx, device, args.batch))
        for hw in CHEAP:
            g_c, _ = gflops_res(model, hw)
            qc_ = embed(lambda im, hw=hw: forward_res(model, im, hw), qx, device, args.batch)
            t1, gap = signals(qc_, gf, device)
            drift = 1 - (qc_ * qf).sum(1).numpy()
            orders = {"top1": np.argsort(t1),          # least confident first
                      "gap": np.argsort(gap),
                      "random": rng.permutation(len(t1)),
                      "oracle": np.argsort(-drift)}    # largest drift first (upper bound)
            for sig, order in orders.items():
                for p in PS:
                    k = int(round(p * len(order)))
                    q = qc_.clone()
                    if k:
                        idx = torch.as_tensor(order[:k])
                        q[idx] = qf[idx]
                    add("cascade", f"{hw[0]}x{hw[1]}", sig, p, g_c + p * g_full, q)
            print(f"{name} cheap {hw}: done", flush=True)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:  # rewritten after every checkpoint
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"{name}: saved {args.out}", flush=True)
        del model, ref
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
