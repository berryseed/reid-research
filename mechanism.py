"""Mechanism diagnostics for token-budget embedding drift (eval only, no training).

For each checkpoint and each query-side reduction setting, with a FULL-TOKEN gallery:
  q_cos        mean cosine(reduced query embedding, full-token query embedding)
  offset_frac  ||mean(d)||^2 / mean(||d||^2), d = f_red - f_full : 1 = one shared offset, 0 = per-image drift
  asym_raw     asymmetric mAP, no correction
  asym_mean    + mean-shift correction            (fitted on unlabeled training images)
  asym_affine  + per-dimension mean/std matching  (fitted on unlabeled training images)
  asym_ridge   + closed-form linear map W = (X'X + lam I)^-1 X'Y, X = reduced, Y = full training embeddings
Settings: 3-stage merge (ours), gradual ToMe (constant r per block, GFLOPs-matched), random, class-attention
(no fusion / averaged fusion / official EViT summed fusion), single-stage reduction, low-resolution queries.

Example
  python mechanism.py --data D:/data/Market-1501-v15.09.15 --ckpts runs/base_s0/last.pth --out runs/mechanism.csv
"""
import argparse
import csv
import os

import torch
import torch.nn.functional as F
from timm.layers import resample_abs_pos_embed
from torch.utils.data import DataLoader

from common import ImageSet, ReIDViT, build_transforms, evaluate_rank, parse_market_dir, vit_gflops


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--out", default="runs/mechanism.csv")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--train-step", type=int, default=2, help="use every n-th training image for the fits")
    p.add_argument("--lam", type=float, default=1.0, help="ridge strength for the linear map")
    return p.parse_args()


@torch.no_grad()
def cache_images(loader):
    """Preprocessed images as one fp16 CPU tensor (so each setting costs seconds, not a disk pass)."""
    xs, pids, cams = [], [], []
    for imgs, pid, cam in loader:
        xs.append(imgs.half())
        pids.append(torch.as_tensor(pid))
        cams.append(torch.as_tensor(cam))
    return torch.cat(xs), torch.cat(pids).numpy(), torch.cat(cams).numpy()


@torch.no_grad()
def embed(fn, x, device, batch):
    out = []
    for s in range(0, x.shape[0], batch):
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            bn = fn(x[s:s + batch].to(device).float())
        out.append(F.normalize(bn.float(), dim=1).cpu())
    return torch.cat(out)


def forward_res(model, imgs, hw):
    """Full-token forward at a lower input resolution (position embedding bicubically resampled)."""
    v = model.vit
    x = F.interpolate(imgs, size=hw, mode="bicubic", antialias=True, align_corners=False)
    x = v.patch_embed.proj(x)
    gh, gw = x.shape[-2:]
    x = v.patch_embed.norm(x.flatten(2).transpose(1, 2))
    pos = resample_abs_pos_embed(v.pos_embed, (gh, gw), old_size=v.patch_embed.grid_size,
                                 num_prefix_tokens=v.num_prefix_tokens)
    x = torch.cat([v.cls_token.expand(x.shape[0], -1, -1), x], dim=1) + pos.to(x.dtype)
    x = v.norm_pre(x)
    for blk in v.blocks:
        x = blk(x)
    return model.bnneck(v.norm(x)[:, 0])


def gflops_res(model, hw):
    ph, pw = model.vit.patch_embed.patch_size
    v = model.vit
    n = (hw[0] // ph) * (hw[1] // pw)
    g = vit_gflops(model, [(n + 1, n + 1)] * len(v.blocks))
    g -= (v.patch_embed.num_patches - n) * (ph * pw * 3) * v.embed_dim / 1e9  # patch embedding of n patches only
    return g, n


def tome_counts(n_patch, r, depth):
    counts, n = [], n_patch
    for _ in range(depth):
        ri = min(r, n // 2)
        counts.append((n + 1, n - ri + 1))
        n -= ri
    return counts


def match_r(model, target):
    v = model.vit
    best = min(range(1, 33), key=lambda r: abs(vit_gflops(model, tome_counts(v.patch_embed.num_patches, r,
                                                                                 len(v.blocks))) - target))
    return best


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    cached = None
    for path in args.ckpts:
        ck = torch.load(path, map_location="cpu", weights_only=True)
        a = ck["args"]
        img_size = (a["height"], a["width"])
        model = ReIDViT(a["model"], ck["num_classes"], img_size, pretrained=False)
        model.load_state_dict(ck["model"])
        model.to(device).eval()
        if cached is None:
            cfg = model.vit.pretrained_cfg
            _, tf = build_transforms(img_size, cfg.get("mean", (0.5,) * 3), cfg.get("std", (0.5,) * 3))
            kw = dict(batch_size=args.batch, shuffle=False, num_workers=args.workers)
            q_items = parse_market_dir(os.path.join(args.data, "query"))
            g_items = parse_market_dir(os.path.join(args.data, "bounding_box_test"))
            t_items = parse_market_dir(os.path.join(args.data, "bounding_box_train"))[::args.train_step]
            cached = [cache_images(DataLoader(ImageSet(it, tf), **kw)) for it in (q_items, t_items)]
            g_loader = DataLoader(ImageSet(g_items, tf), **kw)
            print(f"cached {cached[0][0].shape[0]} query and {cached[1][0].shape[0]} training images", flush=True)
        (qx, qp, qc), (tx, _, _) = cached

        full = lambda im: model(im, "none", 1.0)[1]
        gfs, gps, gcs = [], [], []
        with torch.no_grad():
            for imgs, pid, cam in g_loader:
                gfs.append(embed(full, imgs, device, args.batch))
                gps.append(torch.as_tensor(pid))
                gcs.append(torch.as_tensor(cam))
        gf, gp, gc = torch.cat(gfs), torch.cat(gps).numpy(), torch.cat(gcs).numpy()
        qf, tf_full = embed(full, qx, device, args.batch), embed(full, tx, device, args.batch)
        base_map = evaluate_rank(qf, gf, qp, gp, qc, gc, device)[0]
        g_full = vit_gflops(model)
        print(f"== {path}: full-token mAP {base_map:.2f}, {g_full:.3f} GFLOPs", flush=True)

        def run(name, fn, gflops):
            q, t = embed(fn, qx, device, args.batch), embed(fn, tx, device, args.batch)
            d = q - qf
            mu_r, mu_f = t.mean(0), tf_full.mean(0)
            sd_r, sd_f = t.std(0).clamp(min=1e-6), tf_full.std(0)
            W = torch.linalg.solve(t.double().T @ t.double() + args.lam * torch.eye(t.shape[1], dtype=torch.double),
                                   t.double().T @ tf_full.double()).float()
            fixes = {
                "asym_raw": q,
                "asym_mean": F.normalize(q + (mu_f - mu_r), dim=1),
                "asym_affine": F.normalize((q - mu_r) / sd_r * sd_f + mu_f, dim=1),
                "asym_ridge": F.normalize(q @ W, dim=1),
            }
            row = {"ckpt": os.path.basename(os.path.dirname(path)), "setting": name, "gflops": round(gflops, 3),
                   "q_cos": round(float(F.cosine_similarity(q, qf, dim=1).mean()), 4),
                   "offset_frac": round(float(d.mean(0).pow(2).sum() / d.pow(2).sum(1).mean()), 4)}
            for k, v in fixes.items():
                row[k] = round(evaluate_rank(v, gf, qp, gp, qc, gc, device)[0], 2)
            rows.append(row)
            print(row, flush=True)

        def red(mode, keep):
            def fn(im):
                return model(im, mode, keep)[1]
            return fn

        def cost(mode, keep):
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                model(qx[:2].to(device).float(), mode, keep)
            return vit_gflops(model)

        for keep in (0.7, 0.5):
            for mode in ("merge", "random", "cls_attn", "cls_attn_fuse", "cls_attn_evit"):
                run(f"{mode}@{keep}", red(mode, keep), cost(mode, keep))
            model.tome_r = match_r(model, cost("merge", keep))
            run(f"merge_r(r={model.tome_r})~{keep}", red("merge_r", 1.0), cost("merge_r", 1.0))
        stages = model.reduce_layers
        for st in stages:  # one large reduction (keep 0.5) at a single stage
            model.reduce_layers = (st,)
            run(f"merge@0.5,stage{st}only", red("merge", 0.5), cost("merge", 0.5))
        model.reduce_layers = stages
        for hw in ((224, 112), (192, 96), (160, 80), (128, 64)):
            g, n = gflops_res(model, hw)
            run(f"lowres{hw[0]}x{hw[1]}({n}tok)", lambda im, hw=hw: forward_res(model, im, hw), g)

        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:  # rewritten after every checkpoint
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        del model
        torch.cuda.empty_cache()
    print("saved", args.out)


if __name__ == "__main__":
    main()
