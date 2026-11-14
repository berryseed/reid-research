"""Train a ViT re-ID baseline on a Market-1501-format dataset, optionally with
token-budget self-distillation (full-token pass teaches a reduced-token pass of the same network).

Examples
  Baseline:
    python train.py --data D:/data/Market-1501-v15.09.15 --out runs/vits_base
  Token-budget self-distillation (random budget per step, merging):
    python train.py --data ... --out runs/vits_tbsd --sd-mode merge --sd-keep random
  Low-data run (25% of identities):
    python train.py --data ... --out runs/vits_base_id25 --id-fraction 0.25
  Cheaper training (primary pass itself uses 70% of tokens at each reduction layer):
    python train.py --data ... --out runs/vits_train70 --train-mode merge --train-keep 0.7
"""
import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import (MODES, ImageSet, PKSampler, ReIDViT, batch_hard_triplet, build_transforms,
                    evaluate_rank, extract_features, load_market, set_seed, vit_gflops)

SD_KEEPS = (0.5, 0.6, 0.7, 0.8, 0.9)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="dataset root with bounding_box_train/, query/, bounding_box_test/")
    p.add_argument("--out", default="runs/exp")
    p.add_argument("--model", default="vit_small_patch16_224")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--P", type=int, default=16, help="identities per batch")
    p.add_argument("--K", type=int, default=4, help="images per identity")
    p.add_argument("--lr", type=float, default=0.008)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--margin", type=float, default=0.3)
    p.add_argument("--label-smooth", type=float, default=0.1)
    p.add_argument("--id-fraction", type=float, default=1.0, help="fraction of training identities to use")
    p.add_argument("--train-mode", choices=MODES, default="none", help="token reduction in the primary pass")
    p.add_argument("--train-keep", type=float, default=1.0)
    p.add_argument("--sd-mode", choices=MODES, default="none", help="token reduction in the distilled pass")
    p.add_argument("--sd-keep", default="random", help="keep ratio per reduction layer, or 'random'")
    p.add_argument("--sd-weight", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    img_size = (args.height, args.width)

    train_items, query, gallery, num_classes = load_market(args.data, args.id_fraction, args.seed)
    model = ReIDViT(args.model, num_classes, img_size, pretrained=not args.no_pretrained).to(device)
    cfg = model.vit.pretrained_cfg
    train_tf, test_tf = build_transforms(img_size, cfg.get("mean", (0.5,) * 3), cfg.get("std", (0.5,) * 3))

    pin = device.type == "cuda"
    train_loader = DataLoader(ImageSet(train_items, train_tf),
                              batch_sampler=PKSampler(train_items, args.P, args.K, args.seed),
                              num_workers=args.workers, pin_memory=pin, persistent_workers=args.workers > 0)
    test_kw = dict(batch_size=256, shuffle=False, num_workers=args.workers, pin_memory=pin)
    q_loader = DataLoader(ImageSet(query, test_tf), **test_kw)
    g_loader = DataLoader(ImageSet(gallery, test_tf), **test_kw)

    optim = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                            momentum=0.9, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    def lr_at(step):
        if step < warmup_steps:
            return args.lr * (0.01 + 0.99 * step / max(1, warmup_steps))
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return args.lr * (0.002 + 0.998 * 0.5 * (1 + math.cos(math.pi * t)))

    scaler = torch.amp.GradScaler(enabled=amp)
    use_sd = args.sd_mode != "none"
    log = {"args": vars(args), "num_classes": num_classes, "train_images": len(train_items), "epochs": []}
    print(f"device={device} ids={num_classes} train_images={len(train_items)} steps/epoch={steps_per_epoch} sd={use_sd}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    train_seconds, step = 0.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        sums = {"loss": 0.0, "id": 0.0, "tri": 0.0, "sd": 0.0}
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            for g in optim.param_groups:
                g["lr"] = lr_at(step)
            with torch.autocast(device_type=device.type, enabled=amp):
                feat, bn, logits = model(imgs, args.train_mode, args.train_keep)
                l_id = F.cross_entropy(logits.float(), labels, label_smoothing=args.label_smooth)
                l_tri = batch_hard_triplet(feat, labels, args.margin)
                loss = l_id + l_tri
                l_sd = torch.zeros((), device=device)
                if use_sd:
                    keep = random.choice(SD_KEEPS) if args.sd_keep == "random" else float(args.sd_keep)
                    feat_s, bn_s, logits_s = model(imgs, args.sd_mode, keep)
                    l_id_s = F.cross_entropy(logits_s.float(), labels, label_smoothing=args.label_smooth)
                    l_tri_s = batch_hard_triplet(feat_s, labels, args.margin)
                    # the reduced-token embedding is pulled onto the (detached) full-token embedding
                    l_sd = (1 - F.cosine_similarity(bn_s.float(), bn.detach().float(), dim=1)).mean()
                    loss = loss + l_id_s + l_tri_s + args.sd_weight * l_sd
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            step += 1
            for k, v in zip(sums, (loss, l_id, l_tri, l_sd)):
                sums[k] += float(v.detach())
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_seconds += time.perf_counter() - t0
        rec = {"epoch": epoch, "train_hours": train_seconds / 3600,
               **{k: v / steps_per_epoch for k, v in sums.items()}}
        if device.type == "cuda":
            rec["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
        msg = (f"epoch {epoch:3d}/{args.epochs} loss {rec['loss']:.3f} id {rec['id']:.3f} tri {rec['tri']:.3f} "
               f"sd {rec['sd']:.3f} | {train_seconds / 60:.1f} min")
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            qf, qp, qc = extract_features(model, q_loader, device, amp=amp)
            gf, gp, gc = extract_features(model, g_loader, device, amp=amp)
            mAP, cmc = evaluate_rank(qf, gf, qp, gp, qc, gc, device)
            rec.update(mAP=mAP, rank1=float(cmc[0]))
            msg += f" | mAP {mAP:.1f} R1 {cmc[0]:.1f}"
        print(msg, flush=True)
        log["epochs"].append(rec)
        with open(os.path.join(args.out, "log.json"), "w") as f:
            json.dump(log, f, indent=1)

    with torch.no_grad():
        model.eval()
        model(torch.zeros(1, 3, *img_size, device=device))
    log["gflops_full"] = vit_gflops(model)
    with open(os.path.join(args.out, "log.json"), "w") as f:
        json.dump(log, f, indent=1)
    torch.save({"model": model.state_dict(), "args": vars(args), "num_classes": num_classes},
               os.path.join(args.out, "last.pth"))
    print(f"done. train time {train_seconds / 3600:.2f} h, checkpoint: {os.path.join(args.out, 'last.pth')}")


if __name__ == "__main__":
    main()
