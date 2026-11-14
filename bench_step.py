"""Time training steps on random input to estimate wall-clock per epoch on this GPU.

Not a substitute for a real run: no data loading, no augmentation. Use it to size runs.
  python bench_step.py                       # ViT-S baseline and TBSD steps
  python bench_step.py --model vit_base_patch16_224
"""
import argparse
import time

import torch
import torch.nn.functional as F

from common import ReIDViT, batch_hard_triplet

p = argparse.ArgumentParser()
p.add_argument("--model", default="vit_small_patch16_224")
p.add_argument("--P", type=int, default=16)
p.add_argument("--K", type=int, default=4)
p.add_argument("--iters", type=int, default=30)
p.add_argument("--steps-per-epoch", type=int, default=202, help="Market-1501: 12936 images // 64")
args = p.parse_args()

dev = torch.device("cuda")
model = ReIDViT(args.model, 751, (256, 128), pretrained=False).to(dev).train()
opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
scaler = torch.amp.GradScaler()
B = args.P * args.K
x = torch.randn(B, 3, 256, 128, device=dev)
y = torch.arange(args.P, device=dev).repeat_interleave(args.K)


def step(sd):
    with torch.autocast("cuda"):
        feat, bn, logits = model(x)
        loss = F.cross_entropy(logits.float(), y, label_smoothing=0.1) + batch_hard_triplet(feat, y)
        if sd:
            fs, bs, ls = model(x, "merge", 0.7)
            loss = loss + F.cross_entropy(ls.float(), y) + batch_hard_triplet(fs, y) \
                + (1 - F.cosine_similarity(bs.float(), bn.detach().float())).mean()
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()


for sd in (False, True):
    torch.cuda.reset_peak_memory_stats()
    for _ in range(5):
        step(sd)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        step(sd)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / args.iters
    print(f"{args.model} {'TBSD(merge,0.7)' if sd else 'baseline'}: {dt * 1000:.1f} ms/step, "
          f"~{dt * args.steps_per_epoch / 60:.2f} min/epoch (compute only), "
          f"120 epochs ~{dt * args.steps_per_epoch * 120 / 3600:.2f} h, "
          f"peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
