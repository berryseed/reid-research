"""VPFA-inspired nonlinear post-hoc map (evaluation-time baseline; the ViT is frozen, never retrained).

For one frozen re-ID checkpoint and each low-resolution query setting:
  X = cheap (low-resolution) query embeddings, Y = full-resolution embeddings of the SAME training images
  (Market-1501 bounding_box_train, no identity labels used in any fit). Train identities are disjoint from the
  test identities (Market-1501 protocol), so no test identity is seen by any map.
  Model selection: training identities split 80/20 (fixed seed) into FIT and VAL identities. Every map is fitted
  on FIT only; hyperparameters (ridge lambda; MLP config) and the early-stopping epoch are chosen by the
  regression loss ||map(x) - y||^2 on VAL identities (identities never seen by the map). VAL retrieval mAP is
  also logged but cannot select: the frozen ViT was trained on all training identities, so it is saturated
  (about 99.98 in the smoke test). The test query/gallery split is used once, for the final numbers.

Maps compared (all outputs L2-normalised before retrieval):
  raw          no correction
  ridge_report the report's linear map: W = (X'X + I)^-1 X'Y on every 2nd training image (as mechanism.py)
  ridge_cv     same closed form, fitted on FIT identities (+ flips), lambda chosen on VAL
  mlp          VPFA-style residual gated MLP:  z_hat = z + tanh(W4 VP(z)),  VP(z) = W3 relu(LN(W2 relu(LN(W1 z)))),
               loss  ||z_hat - y||^2  (VPFA Eq. 5); config + epoch chosen on VAL

Example
  python nonlinear_map.py --data C:/Users/user/data/Market-1501-v15.09.15 \
      --ckpts runs/base_e216_s0/last.pth --out runs/nonlinear_map.csv
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import ImageSet, ReIDViT, build_transforms, evaluate_rank, parse_market_dir, vit_gflops
from mechanism import cache_images, embed, forward_res, gflops_res

RES = ((192, 96), (160, 80), (128, 64))
# (name, hidden, weight decay, lr, batch, max epochs). "vpfa" = the optimiser settings stated in the VPFA paper
# (Adam 2e-4, wd 1e-5, batch 32); 30 epochs on our ~21k FIT pairs = about the paper's number of optimiser steps
# (5,000 pairs / 32 x 120 epochs = 18.8k steps). A 4th config (h512, wd 1e-3) was dropped after the smoke test
# for the 30-min GPU budget (VAL loss within 1e-3 of h512_wd1e-5).
GRID = [("vpfa", 2048, 1e-5, 2e-4, 32, 30),
        ("h2048_wd1e-5", 2048, 1e-5, 1e-3, 256, 120),
        ("h512_wd1e-5", 512, 1e-5, 1e-3, 256, 120)]
LAMS = (0.1, 1.0, 10.0, 100.0)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--out", default="runs/nonlinear_map.csv")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--eval-every", type=int, default=5, help="epochs between VAL-mAP checks (early stopping)")
    p.add_argument("--max-epochs", type=int, default=None, help="override for smoke tests")
    p.add_argument("--res", default="192x96,160x80,128x64")
    return p.parse_args()


class VPMap(nn.Module):
    """Residual gated MLP after VPFA (Liu et al., ICML 2026), scaled to our 384-d BNNeck embedding."""

    def __init__(self, d, h):
        super().__init__()
        self.vp = nn.Sequential(nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU(),
                                nn.Linear(h, h), nn.LayerNorm(h), nn.ReLU(), nn.Linear(h, d))
        self.gate = nn.Linear(d, d)
        for m in self.modules():  # VPFA initialisation: N(0, 1e-3) weights, zero biases (std 1e-3 assumed)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 1e-3)
                nn.init.zeros_(m.bias)

    def forward(self, z):
        return z + torch.tanh(self.gate(self.vp(z)))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def map_macs(d, h):  # multiply-accumulates of one mapped embedding (same convention as vit_gflops)
    return d * h + h * h + h * d + d * d


def ridge(x, y, lam):
    x, y = x.double(), y.double()
    return torch.linalg.solve(x.T @ x + lam * torch.eye(x.shape[1], dtype=torch.double), x.T @ y).float()


def val_map(q, g, p, c, device):
    return evaluate_rank(F.normalize(q, dim=1), g, p, p, c, c, device)[0]


def val_loss(z, y):
    return float((F.normalize(z, dim=1) - y).pow(2).sum(1).mean())  # on normalised outputs (= 2 - 2 cos)


def train_mlp(cfg, xf, yf, xv, yv, pv, cv, device, eval_every, max_epochs, seed):
    name, h, wd, lr, bs, ep = cfg
    ep = max_epochs or ep
    torch.manual_seed(seed)
    m = VPMap(xf.shape[1], h).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    xf_d, yf_d, xv_d = xf.to(device), yf.to(device), xv.to(device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    best = (float("inf"), 0, None, 0.0)
    hist = []
    for e in range(1, ep + 1):
        m.train()
        perm = torch.randperm(xf.shape[0], generator=gen).to(device)
        tot = 0.0
        for s in range(0, xf.shape[0], bs):
            i = perm[s:s + bs]
            loss = (m(xf_d[i]) - yf_d[i]).pow(2).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * i.numel()
        if e % eval_every == 0 or e == ep:
            m.eval()
            with torch.no_grad():
                zv = m(xv_d)
                vloss = val_loss(zv, yv.to(device))
                vm = val_map(zv.cpu(), yv, pv, cv, device)
            hist.append({"epoch": e, "train_loss": round(tot / xf.shape[0], 5), "val_loss": round(vloss, 5),
                         "val_map": round(vm, 2)})
            if vloss < best[0]:
                best = (vloss, e, {k: v.detach().clone() for k, v in m.state_dict().items()}, vm)
    m.load_state_dict(best[2])
    m.eval()
    return m, best[0], best[1], best[3], hist


@torch.no_grad()
def time_map(m, d, device, batch, reps=200):
    x = torch.randn(batch, d, device=device)
    for _ in range(20):
        m(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        m(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1e3  # ms per call


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = [tuple(int(v) for v in r.split("x")) for r in args.res.split(",")]
    rows, logs, cached = [], {}, None
    t_start = time.time()
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
            q_items = parse_market_dir(os.path.join(args.data, "query"))
            g_items = parse_market_dir(os.path.join(args.data, "bounding_box_test"))
            t_items = parse_market_dir(os.path.join(args.data, "bounding_box_train"))
            cached = [cache_images(DataLoader(ImageSet(it, tf), **kw)) for it in (q_items, t_items)]
            g_loader = DataLoader(ImageSet(g_items, tf), **kw)
            tp = cached[1][1]
            ids = np.unique(tp)
            rng = np.random.default_rng(0)
            val_ids = set(rng.choice(ids, size=int(round(len(ids) * args.val_frac)), replace=False).tolist())
            is_val = np.array([p in val_ids for p in tp])
            print(f"cached {cached[0][0].shape[0]} query, {len(tp)} training images; "
                  f"{len(ids) - len(val_ids)} FIT / {len(val_ids)} VAL training identities "
                  f"({(~is_val).sum()} / {is_val.sum()} images)", flush=True)
        (qx, qp, qc), (tx, tp, tc) = cached
        t_ck = time.time()

        full = lambda im: model(im, "none", 1.0)[1]
        gfs, gps, gcs = [], [], []
        with torch.no_grad():
            for imgs, pid, cam in g_loader:
                gfs.append(embed(full, imgs, device, args.batch))
                gps.append(torch.as_tensor(pid))
                gcs.append(torch.as_tensor(cam))
        gf, gp, gc = torch.cat(gfs), torch.cat(gps).numpy(), torch.cat(gcs).numpy()
        flip = lambda fn: (lambda im: fn(torch.flip(im, dims=[3])))
        t_full, t_full_f = embed(full, tx, device, args.batch), embed(flip(full), tx, device, args.batch)
        qf = embed(full, qx, device, args.batch)
        m_full, cmc_full = evaluate_rank(qf, gf, qp, gp, qc, gc, device)
        print(f"== {tag}: full mAP {m_full:.2f} R1 {cmc_full[0]:.2f}", flush=True)
        rows.append({"ckpt": tag, "setting": "full", "gflops": round(vit_gflops(model), 3), "map_name": "raw",
                     "mAP": round(m_full, 2), "R1": round(float(cmc_full[0]), 2)})

        fit, val = torch.from_numpy(~is_val), torch.from_numpy(is_val)
        for hw in res:
            g, n = gflops_res(model, hw)
            fn = lambda im, hw=hw: forward_res(model, im, hw)
            t_c, t_c_f = embed(fn, tx, device, args.batch), embed(flip(fn), tx, device, args.batch)
            q = embed(fn, qx, device, args.batch)
            # FIT: original + flipped training images of FIT identities; VAL: original images of VAL identities
            xf, yf = torch.cat([t_c[fit], t_c_f[fit]]), torch.cat([t_full[fit], t_full_f[fit]])
            xv, yv, pv, cv = t_c[val], t_full[val], tp[is_val], tc[is_val]
            setting = f"lowres{hw[0]}x{hw[1]}"
            out = {}

            out["raw"] = (q, {})
            W = ridge(t_c[::2], t_full[::2], 1.0)  # exactly the report's fit (mechanism.py, --train-step 2, lam 1)
            out["ridge_report"] = (q @ W, {"lam": 1.0, "fit_images": int(t_c[::2].shape[0])})
            lam_scores = {lam: val_loss(xv @ ridge(xf, yf, lam), yv) for lam in LAMS}
            lam = min(lam_scores, key=lam_scores.get)
            out["ridge_cv"] = (q @ ridge(xf, yf, lam), {"lam": lam, "val_loss": round(lam_scores[lam], 5),
                                                        "val_by_lam": {str(k): round(v, 5) for k, v in lam_scores.items()}})
            raw_val = val_loss(xv, yv)

            best, cand = None, {}
            for cfg in GRID:
                t0 = time.time()
                mlp, vl, be, vm, hist = train_mlp(cfg, xf, yf, xv, yv, pv, cv, device, args.eval_every,
                                                  args.max_epochs, seed=0)
                sec = time.time() - t0
                with torch.no_grad():
                    vl = val_loss(mlp(xv.to(device)).cpu(), yv)
                cand[cfg[0]] = {"val_loss": round(vl, 5), "val_map": round(vm, 2), "best_epoch": be,
                                "train_sec": round(sec, 1), "hist": hist}
                print(f"  {setting} {cfg[0]}: VAL loss {vl:.5f} (raw {raw_val:.5f}, ridge {lam_scores[lam]:.5f}) "
                      f"at epoch {be}, VAL mAP {vm:.2f} ({sec:.0f}s)", flush=True)
                if best is None or vl < best[1]:
                    best = (cfg, vl, be, mlp, sec, vm)
            cfg, vl, be, mlp, sec, vm = best
            with torch.no_grad():
                qm = mlp(q.to(device)).cpu()
            d, h = q.shape[1], cfg[1]
            lat = {"gpu_b1_ms": round(time_map(mlp, d, device, 1), 4),
                   "gpu_b256_ms_per_img": round(time_map(mlp, d, device, 256) / 256, 5)}
            mlp_cpu = VPMap(d, h).eval()
            mlp_cpu.load_state_dict({k: v.cpu() for k, v in mlp.state_dict().items()})
            lat["cpu_b1_ms"] = round(time_map(mlp_cpu, d, torch.device("cpu"), 1, reps=100), 4)
            out["raw"][1]["val_loss"] = round(raw_val, 5)
            out["mlp"] = (qm, {"config": cfg[0], "hidden": h, "val_loss": round(vl, 5), "val_map": round(vm, 2),
                               "best_epoch": be,
                               "params": n_params(mlp), "gmacs": round(map_macs(d, h) / 1e9, 5),
                               "train_sec": round(sec, 1), **lat})
            out["mlp_vpfa_cfg"] = (None, cand["vpfa"])  # VAL record of the paper's own hyperparameters

            for name, (emb, info) in out.items():
                row = {"ckpt": tag, "setting": setting, "gflops": round(g, 3), "map_name": name}
                if emb is not None:
                    mp, cmc = evaluate_rank(F.normalize(emb, dim=1), gf, qp, gp, qc, gc, device)
                    row.update({"mAP": round(mp, 2), "R1": round(float(cmc[0]), 2)})
                row.update({k: v for k, v in info.items() if k != "hist"})
                rows.append(row)
                print({k: v for k, v in row.items() if k not in ("val_by_lam",)}, flush=True)
            logs[f"{tag}/{setting}"] = {"grid": cand, "ridge_val": out["ridge_cv"][1]}

        print(f"== {tag} done in {time.time() - t_ck:.0f}s (total {time.time() - t_start:.0f}s)", flush=True)
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:  # rewritten after every checkpoint
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        with open(os.path.splitext(args.out)[0] + "_log.json", "w") as f:
            json.dump(logs, f, indent=1)
        del model
        torch.cuda.empty_cache()
    print(f"saved {args.out}; wall {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
