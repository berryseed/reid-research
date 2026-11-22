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

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import (MODES, ImageSet, PKSampler, ReIDViT, batch_hard_triplet, build_transforms,
                    evaluate_rank, extract_features, load_market, set_seed, vit_gflops)

SD_KEEPS = (0.5, 0.6, 0.7, 0.8, 0.9)
SD_RES = ((224, 112), (192, 96), (160, 80), (128, 64))  # --sd-mode lowres: 98 / 72 / 50 / 32 patch tokens
SD_RS = (3, 5, 7, 9, 11)                                 # --sd-mode merge_r: tokens merged per block (ToMe)


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
    p.add_argument("--sd-mode", choices=MODES + ("lowres", "mixed"), default="none",
                   help="cheap second pass: a token-reduction mode; 'merge_r' = gradual ToMe with random r; "
                        "'lowres' = full tokens of a randomly downsized image; 'mixed' = lowres or merge_r per step")
    p.add_argument("--sd-keep", default="random", help="keep ratio per reduction layer, or 'random'")
    p.add_argument("--sd-weight", type=float, default=1.0)
    p.add_argument("--sd-bn", choices=("shared", "nostat"), default="shared",
                   help="BNNeck in the reduced pass: 'shared' also updates the running statistics (original "
                        "behaviour); 'nostat' normalises with batch statistics but leaves the running "
                        "statistics to the full-token pass only")
    p.add_argument("--teacher", default=None,
                   help="frozen-gallery student mode: path to a trained last.pth. The model starts as a copy of it "
                        "and is trained WITHOUT labels so that its cheap pass (--sd-mode, required) reproduces the "
                        "frozen teacher's full-token BNNeck embedding: loss = mean(1 - cos(student_cheap, "
                        "teacher_full)). In the style of Suma & Tolias (WACV 2023), absolute term only.")
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="continue from <out>/ckpt.pth if it exists; start fresh only if <out> has no results")
    p.add_argument("--ckpt-every", type=int, default=1,
                   help="save the resumable <out>/ckpt.pth every N epochs (0 = never)")
    p.add_argument("--keep-ckpt", action="store_true",
                   help="keep ckpt.pth after the run completes (default: removed once last.pth is saved)")
    return p.parse_args()


PAUSE_EXIT_CODE = 3  # returned when a run stops at an epoch boundary because <out>/STOP exists


def refuse_overwrite(out, resume):
    """Exit (code 1) instead of overwriting an earlier run in `out`.

    The only run allowed into a folder that has results is the resume of an unfinished one (ckpt.pth, no last.pth).
    log.json alone also counts as results: it is all that a fresh clone has of a finished run.
    """
    found = [f for f in ("last.pth", "ckpt.pth", "log.json") if os.path.exists(os.path.join(out, f))]
    if not found or (resume and "ckpt.pth" in found and "last.pth" not in found):
        return
    hint = ("It is an unfinished run: add --resume to continue it, or choose a new --out."
            if "ckpt.pth" in found and "last.pth" not in found else "Choose a new --out.")
    raise SystemExit(f"refusing to overwrite {out} (found {', '.join(found)}). {hint}")


def save_resume_ckpt(path, **state):
    """Write atomically so an interruption mid-write never corrupts the previous checkpoint."""
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def main():
    args = get_args()
    refuse_overwrite(args.out, args.resume)  # before anything is written
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    img_size = (args.height, args.width)

    train_items, query, gallery, num_classes = load_market(args.data, args.id_fraction, args.seed)
    model = ReIDViT(args.model, num_classes, img_size, pretrained=not args.no_pretrained).to(device)
    teacher = None
    if args.teacher:
        if args.sd_mode == "none":
            raise ValueError("--teacher needs a cheap pass: set --sd-mode (e.g. mixed)")
        tck = torch.load(args.teacher, map_location="cpu", weights_only=False)  # our own file
        if tck["num_classes"] != num_classes:
            raise ValueError("teacher was trained on a different identity set")
        model.load_state_dict(tck["model"])                      # student starts as an exact copy
        teacher = ReIDViT(args.model, num_classes, img_size, pretrained=False).to(device)
        teacher.load_state_dict(tck["model"])
        teacher.eval().requires_grad_(False)                     # the gallery encoder never changes
    cfg = model.vit.pretrained_cfg
    train_tf, test_tf = build_transforms(img_size, cfg.get("mean", (0.5,) * 3), cfg.get("std", (0.5,) * 3))

    pin = device.type == "cuda"
    sampler = PKSampler(train_items, args.P, args.K, args.seed)
    train_loader = DataLoader(ImageSet(train_items, train_tf), batch_sampler=sampler,
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
    train_seconds, step, start_epoch = 0.0, 0, 1
    ckpt_path = os.path.join(args.out, "ckpt.pth")
    stop_path = os.path.join(args.out, "STOP")
    if args.resume and os.path.exists(ckpt_path):
        # our own file; weights_only=False is needed for the Python/NumPy RNG states
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        for k in ("model", "lr", "epochs", "P", "K", "seed", "sd_mode", "train_mode", "id_fraction",
                  "sd_weight", "sd_bn", "sd_keep", "train_keep", "teacher"):
            if ck["args"].get(k) != vars(args).get(k):
                raise ValueError(f"--{k} differs from the checkpoint ({ck['args'].get(k)} vs {vars(args).get(k)})")
        model.load_state_dict(ck["model"])
        optim.load_state_dict(ck["optim"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, step, train_seconds, log = ck["epoch"] + 1, ck["step"], ck["train_seconds"], ck["log"]
        sampler.rng.setstate(ck["rng"]["sampler"])
        random.setstate(ck["rng"]["python"])
        np.random.set_state(ck["rng"]["numpy"])
        torch.set_rng_state(ck["rng"]["torch"])
        if device.type == "cuda" and ck["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
        print(f"resumed from {ckpt_path}: epoch {ck['epoch']} done, step {step}, "
              f"{train_seconds / 3600:.2f} h trained", flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        if teacher is not None:
            model.bnneck.eval()  # student embedding computed as at test time (teacher's running statistics)
        t0 = time.perf_counter()
        sums = {"loss": 0.0, "id": 0.0, "tri": 0.0, "sd": 0.0}
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            for g in optim.param_groups:
                g["lr"] = lr_at(step)
            with torch.autocast(device_type=device.type, enabled=amp):
                if teacher is not None:
                    # frozen-gallery student: no labels, only the cheap pass has gradients
                    with torch.no_grad():
                        bn_t = teacher(imgs)[1]                       # full-token target (same augmented image)
                    sd_mode, keep, sd_hw = args.sd_mode, 1.0, None
                    if sd_mode == "mixed":
                        sd_mode = random.choice(("lowres", "merge_r"))
                    if sd_mode == "lowres":
                        sd_mode, sd_hw = "none", random.choice(SD_RES)
                    elif sd_mode == "merge_r":
                        model.tome_r = random.choice(SD_RS)
                    else:
                        keep = random.choice(SD_KEEPS) if args.sd_keep == "random" else float(args.sd_keep)
                    bn_s = model(imgs, sd_mode, keep, sd_hw)[1]
                    l_sd = (1 - F.cosine_similarity(bn_s.float(), bn_t.float(), dim=1)).mean()
                    l_id = l_tri = torch.zeros((), device=device)
                    loss = l_sd
                else:
                    feat, bn, logits = model(imgs, args.train_mode, args.train_keep)
                    l_id = F.cross_entropy(logits.float(), labels, label_smoothing=args.label_smooth)
                    l_tri = batch_hard_triplet(feat, labels, args.margin)
                    loss = l_id + l_tri
                    l_sd = torch.zeros((), device=device)
                if use_sd and teacher is None:
                    keep = random.choice(SD_KEEPS) if args.sd_keep == "random" else float(args.sd_keep)
                    sd_mode, sd_hw = args.sd_mode, None
                    if sd_mode == "mixed":
                        sd_mode = random.choice(("lowres", "merge_r"))
                    if sd_mode == "lowres":      # same network, smaller image, all of its tokens
                        sd_mode, keep, sd_hw = "none", 1.0, random.choice(SD_RES)
                    elif sd_mode == "merge_r":   # official ToMe schedule, r tokens merged in every block
                        model.tome_r, keep = random.choice(SD_RS), 1.0
                    bn_mom = model.bnneck.momentum
                    if args.sd_bn == "nostat":
                        model.bnneck.momentum = 0.0  # running stats unchanged by this pass
                    feat_s, bn_s, logits_s = model(imgs, sd_mode, keep, sd_hw)
                    model.bnneck.momentum = bn_mom
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
        pause = os.path.exists(stop_path) and epoch < args.epochs
        if args.ckpt_every and (epoch % args.ckpt_every == 0 or pause) and epoch < args.epochs:
            save_resume_ckpt(
                ckpt_path, model=model.state_dict(), optim=optim.state_dict(), scaler=scaler.state_dict(),
                epoch=epoch, step=step, train_seconds=train_seconds, log=log, args=vars(args),
                num_classes=num_classes,
                rng={"sampler": sampler.rng.getstate(), "python": random.getstate(),
                     "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                     "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None})
        if pause:
            os.remove(stop_path)
            print(f"paused after epoch {epoch} (STOP file). Resume with the same command plus --resume. "
                  f"Checkpoint: {ckpt_path}", flush=True)
            raise SystemExit(PAUSE_EXIT_CODE)

    with torch.no_grad():
        model.eval()
        model(torch.zeros(1, 3, *img_size, device=device))
    log["gflops_full"] = vit_gflops(model)
    with open(os.path.join(args.out, "log.json"), "w") as f:
        json.dump(log, f, indent=1)
    torch.save({"model": model.state_dict(), "args": vars(args), "num_classes": num_classes},
               os.path.join(args.out, "last.pth"))
    if os.path.exists(ckpt_path) and not args.keep_ckpt:
        os.remove(ckpt_path)  # final weights are in last.pth; the resumable state is no longer needed
    print(f"done. train time {train_seconds / 3600:.2f} h, checkpoint: {os.path.join(args.out, 'last.pth')}")


if __name__ == "__main__":
    main()
