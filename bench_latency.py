"""Forward-pass latency of the ViT re-ID encoder at several token budgets (random input, weights irrelevant).

Reports ms per image and images/s for GPU (batch 1 and 256, AMP) and CPU (batch 1 and 8, fp32).
Run it only when no training is using the GPU/CPU, otherwise the numbers are contaminated.

  python bench_latency.py --out runs/latency.csv
  python bench_latency.py --model vit_base_patch16_224 --out runs/latency_vitb.csv
"""
import argparse
import csv
import time

import torch

from common import ReIDViT, vit_gflops


@torch.no_grad()
def time_forward(model, device, img_size, mode, keep, batch, iters, warmup=5):
    x = torch.randn(batch, 3, *img_size, device=device)
    amp = device.type == "cuda"

    def run():
        with torch.autocast(device_type=device.type, enabled=amp):
            model(x, mode, keep)
        if amp:
            torch.cuda.synchronize()

    for _ in range(warmup):
        run()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        run()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]  # median seconds per batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="vit_small_patch16_224")
    p.add_argument("--out", default="runs/latency.csv")
    p.add_argument("--modes", nargs="+", default=["merge", "random"])
    p.add_argument("--keeps", nargs="+", type=float, default=[0.8, 0.7, 0.6, 0.5])
    p.add_argument("--no-cpu", action="store_true")
    args = p.parse_args()
    img_size = (256, 128)
    model = ReIDViT(args.model, 751, img_size, pretrained=False).eval()
    settings = [("none", 1.0)] + [(m, k) for m in args.modes for k in args.keeps]
    plans = []
    if torch.cuda.is_available():
        plans += [("cuda", 1, 200), ("cuda", 256, 20)]
    if not args.no_cpu:
        plans += [("cpu", 1, 30), ("cpu", 8, 10)]
    rows = {s: {"mode": s[0], "keep": s[1]} for s in settings}
    for dev, batch, iters in plans:
        device = torch.device(dev)
        model.to(device)
        for mode, keep in settings:
            sec = time_forward(model, device, img_size, mode, keep, batch, iters)
            r = rows[(mode, keep)]
            r["gflops"] = round(vit_gflops(model), 3)
            r[f"{dev}_b{batch}_ms_per_img"] = round(1000 * sec / batch, 4)
            r[f"{dev}_b{batch}_img_s"] = round(batch / sec, 1)
            print(dev, batch, mode, keep, r[f"{dev}_b{batch}_ms_per_img"], "ms/img", flush=True)
    rows = list(rows.values())
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"cpu threads: {torch.get_num_threads()}; saved {args.out}")


if __name__ == "__main__":
    main()
