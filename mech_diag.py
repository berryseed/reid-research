"""Why does sp_mixed work? Compatibility-vs-information decomposition and similarity diagnostics (eval only).

For each frozen checkpoint and query budget b (full, 192x96, 160x80, 128x64, gradual merge r = 11):
  asym      mAP / R1: cheap query  vs FULL-budget gallery        (the deployment setting; matches earlier results)
  sym       mAP / R1: cheap query  vs gallery at the SAME budget  (information kept by the cheap embedding alone)
  info_loss = full mAP - sym mAP        compat_loss = sym mAP - asym mAP
  q_cos, g_cos: mean cos(cheap, full) embedding of the same query / gallery image (drift)
  per-query (asymmetric): s_pos_max = best same-ID different-camera similarity, s_neg_max = hardest different-ID
    similarity, margin = s_pos_max - s_neg_max (R1 correct <=> margin > 0)
  pair-level (asymmetric): mean/std of all positive and all negative pair similarities, d' separability
  top-1 transitions vs the full query: correct->wrong ("broken"), wrong->correct ("fixed")
No training. Uses common.evaluate_rank (Market-1501 protocol).

Example
  python mech_diag.py --data C:/Users/user/data/Market-1501-v15.09.15 --ckpts runs/sp_mixed_s0/last.pth \
      --out runs/mech_diag/mech_diag.csv
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import ImageSet, ReIDViT, build_transforms, evaluate_rank, parse_market_dir
from mechanism import cache_images, embed, forward_res, gflops_res
from common import vit_gflops


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--out", default="runs/mech_diag/mech_diag.csv")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--save-hist", default="lowres160x80",
                   help="budget whose per-query similarity arrays are saved to <out>_<ckpt>.npz")
    return p.parse_args()


@torch.no_grad()
def pair_stats(q, g, qp, gp, qc, gc, device):
    """Asymmetric similarity statistics. Junk (same pid AND same camera) excluded as in evaluate_rank."""
    gd = g.to(device)
    gp_t, gc_t = torch.as_tensor(gp, device=device), torch.as_tensor(gc, device=device)
    s_pos_max, s_neg_max, s_pos_mean = [], [], []
    acc = {"pos_sum": 0.0, "pos_sq": 0.0, "pos_n": 0, "neg_sum": 0.0, "neg_sq": 0.0, "neg_n": 0}
    for s in range(0, q.shape[0], 512):
        sim = q[s:s + 512].to(device) @ gd.T
        pq = torch.as_tensor(qp[s:s + 512], device=device)[:, None]
        cq = torch.as_tensor(qc[s:s + 512], device=device)[:, None]
        same = gp_t[None] == pq
        pos = same & (gc_t[None] != cq)
        neg = ~same
        has = pos.any(1)
        sp = torch.where(pos, sim, torch.full_like(sim, -9)).max(1).values
        sn = torch.where(neg, sim, torch.full_like(sim, -9)).max(1).values
        spm = (sim * pos).sum(1) / pos.sum(1).clamp(min=1)
        s_pos_max.append(sp[has].cpu()), s_neg_max.append(sn[has].cpu()), s_pos_mean.append(spm[has].cpu())
        acc["pos_sum"] += float(sim[pos].sum()); acc["pos_sq"] += float(sim[pos].pow(2).sum()); acc["pos_n"] += int(pos.sum())
        acc["neg_sum"] += float(sim[neg].sum()); acc["neg_sq"] += float(sim[neg].pow(2).sum()); acc["neg_n"] += int(neg.sum())
    sp, sn, spm = torch.cat(s_pos_max), torch.cat(s_neg_max), torch.cat(s_pos_mean)
    mu_p, mu_n = acc["pos_sum"] / acc["pos_n"], acc["neg_sum"] / acc["neg_n"]
    sd_p = (acc["pos_sq"] / acc["pos_n"] - mu_p ** 2) ** 0.5
    sd_n = (acc["neg_sq"] / acc["neg_n"] - mu_n ** 2) ** 0.5
    return {"pos_mean": mu_p, "pos_sd": sd_p, "neg_mean": mu_n, "neg_sd": sd_n,
            "dprime": (mu_p - mu_n) / ((sd_p ** 2 + sd_n ** 2) / 2) ** 0.5,
            "s_pos_max": float(sp.mean()), "s_neg_max": float(sn.mean()), "margin": float((sp - sn).mean()),
            "margin_pos_frac": float((sp > sn).float().mean())}, (sp.numpy(), sn.numpy(), spm.numpy())


@torch.no_grad()
def top1_correct(q, g, qp, gp, qc, gc, device):
    """Top-1 correctness per query (same exclusion as evaluate_rank); queries without a valid match -> -1."""
    gd = g.to(device)
    gp_t, gc_t = torch.as_tensor(gp, device=device), torch.as_tensor(gc, device=device)
    out = []
    for s in range(0, q.shape[0], 512):
        sim = q[s:s + 512].to(device) @ gd.T
        pq = torch.as_tensor(qp[s:s + 512], device=device)[:, None]
        cq = torch.as_tensor(qc[s:s + 512], device=device)[:, None]
        junk = (gp_t[None] == pq) & (gc_t[None] == cq)
        valid = ((gp_t[None] == pq) & ~junk).any(1)
        top = torch.where(junk, torch.full_like(sim, -9), sim).argmax(1)
        c = (gp_t[top] == pq[:, 0]).long()
        out.append(torch.where(valid, c, torch.full_like(c, -1)).cpu())
    return torch.cat(out).numpy()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, cached = [], None
    t0 = time.time()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for path in args.ckpts:
        ck = torch.load(path, map_location="cpu", weights_only=True)
        a = ck["args"]
        img_size = (a["height"], a["width"])
        model = ReIDViT(a["model"], ck["num_classes"], img_size, pretrained=False)
        model.load_state_dict(ck["model"])
        model.to(device).eval()
        tag = os.path.basename(os.path.dirname(path))
        if cached is None:
            cfg = model.vit.pretrained_cfg
            _, tf = build_transforms(img_size, cfg.get("mean", (0.5,) * 3), cfg.get("std", (0.5,) * 3))
            kw = dict(batch_size=args.batch, shuffle=False, num_workers=args.workers)
            cached = [cache_images(DataLoader(ImageSet(parse_market_dir(os.path.join(args.data, d)), tf), **kw))
                      for d in ("query", "bounding_box_test")]
            print(f"cached {cached[0][0].shape[0]} query, {cached[1][0].shape[0]} gallery images "
                  f"({time.time() - t0:.0f}s)", flush=True)
        (qx, qp, qc), (gx, gp, gc) = cached
        t_ck = time.time()

        budgets = [("full", lambda im: model(im, "none", 1.0)[1], 2.93)]
        for hw in ((192, 96), (160, 80), (128, 64)):
            budgets.append((f"lowres{hw[0]}x{hw[1]}", lambda im, hw=hw: forward_res(model, im, hw),
                            gflops_res(model, hw)[0]))
        budgets.append(("merge_r11", lambda im: model(im, "merge_r", 1.0)[1], None))
        emb = {}
        for name, fn, g in budgets:
            if name == "merge_r11":
                model.tome_r = 11
            emb[name] = (embed(fn, qx, device, args.batch), embed(fn, gx, device, args.batch))
            if g is None:
                g = vit_gflops(model)
            emb[name] = emb[name] + (g,)
        qf, gf, _ = emb["full"]
        c_full = top1_correct(qf, gf, qp, gp, qc, gc, device)
        m_full = evaluate_rank(qf, gf, qp, gp, qc, gc, device)[0]
        for name, (q, g, gfl) in emb.items():
            ma, ca = evaluate_rank(q, gf, qp, gp, qc, gc, device)
            ms, cs = evaluate_rank(q, g, qp, gp, qc, gc, device)
            st, arrs = pair_stats(q, gf, qp, gp, qc, gc, device)
            c_b = top1_correct(q, gf, qp, gp, qc, gc, device)
            v = (c_full >= 0) & (c_b >= 0)
            row = {"ckpt": tag, "budget": name, "gflops": round(gfl, 3),
                   "asym_mAP": round(ma, 2), "asym_R1": round(float(ca[0]), 2),
                   "sym_mAP": round(ms, 2), "sym_R1": round(float(cs[0]), 2),
                   "info_loss": round(m_full - ms, 2), "compat_loss": round(ms - ma, 2),
                   "q_cos": round(float(F.cosine_similarity(q, qf, dim=1).mean()), 4),
                   "g_cos": round(float(F.cosine_similarity(g, gf, dim=1).mean()), 4),
                   **{k: round(v_, 4) for k, v_ in st.items()},
                   "top1_broken": int(((c_full == 1) & (c_b == 0) & v).sum()),
                   "top1_fixed": int(((c_full == 0) & (c_b == 1) & v).sum()),
                   "n_valid": int(v.sum())}
            rows.append(row)
            print(row, flush=True)
            if name == args.save_hist:
                np.savez(os.path.splitext(args.out)[0] + f"_{tag}_{name}.npz", s_pos_max=arrs[0], s_neg_max=arrs[1],
                         s_pos_mean=arrs[2], q_cos=F.cosine_similarity(q, qf, dim=1).numpy())
        print(f"== {tag} done in {time.time() - t_ck:.0f}s (total {time.time() - t0:.0f}s)", flush=True)
        with open(args.out, "w", newline="") as f:  # rewritten after every checkpoint
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        del model, emb
        torch.cuda.empty_cache()
    print(f"saved {args.out}; wall {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
