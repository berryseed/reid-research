"""Shared code for the low-resource re-ID experiment kit.

Contents
  - Market-1501-format dataset parsing, PK sampler, transforms
  - ReIDViT: timm ViT + BNNeck, with optional token reduction inside the forward pass
      mode = 'none'      full tokens
             'merge'     ToMe-style bipartite soft matching on attention keys (own implementation)
             'cls_attn'  EViT-style: keep the patch tokens the class token attends to most
             'random'    keep random patch tokens (control)
  - analytic FLOPs estimate, batch-hard triplet loss, Market-1501 evaluation (mAP / CMC)
"""
import glob
import math
import os
import random
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, Sampler

import timm
import torchvision.transforms as T
from timm.layers import resample_abs_pos_embed

# cls_attn_fuse: EViT with its fused "inattentive" token (dropped tokens averaged, weighted by class attention)
# cls_attn_low:  diagnostic only, keeps the patch tokens the class token attends to LEAST
MODES = ("none", "merge", "cls_attn", "random", "cls_attn_fuse", "cls_attn_low",
         "cls_attn_evit",   # official EViT fused token: attention-weighted SUM of the dropped tokens
         "merge_r")         # official ToMe schedule: merge model.tome_r tokens in EVERY block (keep is ignored)


# --------------------------------------------------------------------------- data
def parse_market_dir(dir_path):
    """Return [(path, pid, camid)] for a Market-1501-style folder (e.g. 0002_c1s1_000451_03.jpg)."""
    pattern = re.compile(r"([-\d]+)_c(\d+)")
    paths = sorted(glob.glob(os.path.join(dir_path, "*.jpg")) + glob.glob(os.path.join(dir_path, "*.png")))
    items = []
    for p in paths:
        m = pattern.search(os.path.basename(p))
        if m is None:
            continue
        pid, cam = int(m.group(1)), int(m.group(2))
        if pid == -1:  # junk detections
            continue
        items.append((p, pid, cam))
    if not items:
        raise FileNotFoundError(f"No Market-format images found in {dir_path}")
    return items


def load_market(root, id_fraction=1.0, seed=0):
    """root must contain bounding_box_train/, query/ and bounding_box_test/."""
    train = parse_market_dir(os.path.join(root, "bounding_box_train"))
    query = parse_market_dir(os.path.join(root, "query"))
    gallery = parse_market_dir(os.path.join(root, "bounding_box_test"))
    pids = sorted({pid for _, pid, _ in train})
    if id_fraction < 1.0:  # low-data setting: keep a random subset of identities
        rng = random.Random(seed)
        n = max(2, int(round(len(pids) * id_fraction)))
        pids = sorted(rng.sample(pids, n))
    pid2label = {pid: i for i, pid in enumerate(pids)}
    train = [(p, pid2label[pid], cam) for p, pid, cam in train if pid in pid2label]
    return train, query, gallery, len(pids)


class ImageSet(Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, pid, cam = self.items[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), pid, cam


class PKSampler(Sampler):
    """Batches of P identities x K images. One epoch ~ one pass over the number of training images."""

    def __init__(self, items, p, k, seed=0):
        self.p, self.k = p, k
        self.index = {}
        for i, (_, pid, _) in enumerate(items):
            self.index.setdefault(pid, []).append(i)
        self.pids = list(self.index)
        if len(self.pids) < p:
            raise ValueError(f"Need at least P={p} identities, got {len(self.pids)}")
        self.n_batches = max(1, len(items) // (p * k))
        self.rng = random.Random(seed)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        stream = []
        for _ in range(self.n_batches):
            while len(stream) < self.p:
                perm = self.pids[:]
                self.rng.shuffle(perm)
                stream.extend(perm)
            chosen, stream = stream[: self.p], stream[self.p:]
            if len(set(chosen)) < self.p:  # rare duplicate across a permutation boundary
                chosen = self.rng.sample(self.pids, self.p)
            batch = []
            for pid in chosen:
                idx = self.index[pid]
                if len(idx) >= self.k:
                    batch.extend(self.rng.sample(idx, self.k))
                else:
                    batch.extend(self.rng.choices(idx, k=self.k))
            yield batch


def build_transforms(img_size, mean, std):
    train_tf = T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(0.5),
        T.Pad(10),
        T.RandomCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean, std),
        T.RandomErasing(p=0.5, value=0),
    ])
    test_tf = T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return train_tf, test_tf


# --------------------------------------------------------------------------- token reduction
def attention_with_maps(attn, x, size=None):
    """Re-implements timm's Attention.forward but also returns the attention map and mean keys.

    size: (B, N, 1) number of original patches each token represents (proportional attention), or None.
    """
    B, N, C = x.shape
    h = attn.num_heads
    qkv = attn.qkv(x).reshape(B, N, 3, h, C // h).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q = getattr(attn, "q_norm", nn.Identity())(q)
    k = getattr(attn, "k_norm", nn.Identity())(k)
    a = (q * attn.scale) @ k.transpose(-2, -1)
    if size is not None:
        a = a + size.clamp(min=1e-6).log()[:, None, None, :, 0].to(a.dtype)
    a = a.softmax(dim=-1)
    a_drop = attn.attn_drop(a)
    y = (a_drop @ v).transpose(1, 2).reshape(B, N, C)
    y = getattr(attn, "norm", nn.Identity())(y)
    y = attn.proj_drop(attn.proj(y))
    return y, a, k.mean(1)


def attention_fast(attn, x, size=None, need_cls=False):
    """Same output as attention_with_maps, but uses fused scaled_dot_product_attention.

    Returns (y, cls_attn, mean_key). cls_attn is the class-token attention row (B, heads, N) when
    need_cls is True, else None; only that row is computed explicitly, so the full map is never stored.
    """
    B, N, C = x.shape
    h = attn.num_heads
    qkv = attn.qkv(x).reshape(B, N, 3, h, C // h).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q = getattr(attn, "q_norm", nn.Identity())(q)
    k = getattr(attn, "k_norm", nn.Identity())(k)
    bias = None
    if size is not None:  # proportional attention: log(size) added to the logits of every query
        bias = size.clamp(min=1e-6).log()[:, None, None, :, 0].to(q.dtype)
    p = attn.attn_drop.p if attn.training else 0.0
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=p, scale=attn.scale)
    cls = None
    if need_cls:
        a0 = (q[:, :, :1] * attn.scale) @ k.transpose(-2, -1)
        if bias is not None:
            a0 = a0 + bias
        cls = a0.float().softmax(dim=-1)[:, :, 0]
    y = y.transpose(1, 2).reshape(B, N, C)
    y = getattr(attn, "norm", nn.Identity())(y)
    y = attn.proj_drop(attn.proj(y))
    return y, cls, k.mean(1)


def bipartite_merge(x, size, metric, r, n_prefix=1):
    """Merge r patch tokens into their most similar partner (ToMe-style bipartite soft matching).

    Prefix tokens (class token) are never merged. Returns the merged tokens and updated sizes.
    """
    B, N, C = x.shape
    if size is None:
        size = x.new_ones(B, N, 1)
    px, ps = x[:, n_prefix:], size[:, n_prefix:]
    m = F.normalize(metric[:, n_prefix:].float(), dim=-1)
    a, b = m[:, ::2], m[:, 1::2]
    r = min(r, a.shape[1])
    if r <= 0 or b.shape[1] == 0:
        return x, size
    scores = a @ b.transpose(-1, -2)
    node_max, node_idx = scores.max(dim=-1)
    edge_idx = node_max.argsort(dim=-1, descending=True)
    merged_idx, kept_idx = edge_idx[:, :r], edge_idx[:, r:]
    dst_idx = node_idx.gather(1, merged_idx)

    def merge(t):
        c = t.shape[-1]
        ta, tb = t[:, ::2], t[:, 1::2]
        kept = ta.gather(1, kept_idx[..., None].expand(-1, -1, c))
        src = ta.gather(1, merged_idx[..., None].expand(-1, -1, c))
        tb = tb.scatter_add(1, dst_idx[..., None].expand(-1, -1, c), src)
        return torch.cat([kept, tb], dim=1)

    new_s = merge(ps)
    new_x = merge(px * ps) / new_s
    return torch.cat([x[:, :n_prefix], new_x], 1), torch.cat([size[:, :n_prefix], new_s], 1)


def select_tokens(x, score, n_keep, n_prefix=1):
    """Keep the n_keep patch tokens with the highest score (B, N_patch)."""
    idx = score.topk(n_keep, dim=1).indices.sort(dim=1).values
    patches = x[:, n_prefix:].gather(1, idx[..., None].expand(-1, -1, x.shape[-1]))
    return torch.cat([x[:, :n_prefix], patches], dim=1)


def select_tokens_fuse(x, score, n_keep, n_prefix=1, normalize=True):
    """EViT: keep the top n_keep patch tokens and append one token fused from the dropped ones
    (n_keep + 1 patch tokens come out). normalize=True: score-weighted AVERAGE (our variant);
    normalize=False: score-weighted SUM, as in the official EViT code."""
    kept = select_tokens(x, score, n_keep, n_prefix)
    idx = score.topk(n_keep, dim=1).indices
    w = score.scatter(1, idx, 0.0).clamp(min=0)            # weights of the dropped tokens only
    fused = (x[:, n_prefix:].float() * w[..., None]).sum(1, keepdim=True)
    if normalize:
        fused = fused / w.sum(1).clamp(min=1e-6)[:, None, None]
    return torch.cat([kept, fused.to(x.dtype)], dim=1)


# --------------------------------------------------------------------------- model
class ReIDViT(nn.Module):
    def __init__(self, name="vit_small_patch16_224", num_classes=751, img_size=(256, 128),
                 pretrained=True, reduce_layers=(3, 6, 9)):
        super().__init__()
        self.vit = timm.create_model(name, pretrained=pretrained, num_classes=0, img_size=img_size)
        assert getattr(self.vit, "num_prefix_tokens", 1) >= 1, "model needs a class token"
        self.dim = self.vit.embed_dim
        self.reduce_layers = tuple(reduce_layers)
        self.bnneck = nn.BatchNorm1d(self.dim)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)
        self.fast_attn = True  # fused SDPA in reduced/merged blocks; False = explicit maps (reference path)
        self.tome_r = 0        # tokens merged per block in mode "merge_r" (official ToMe schedule)
        self.last_counts = []  # [(tokens in attention, tokens in MLP)] per block, from the last forward

    def backbone(self, x, mode="none", keep=1.0, hw=None):
        """hw: optional (height, width) to which the input is resized first (fewer patch tokens);
        the position embedding is bicubically resampled to the new grid. None = unchanged path."""
        assert mode in MODES, mode
        v = self.vit
        npre = v.num_prefix_tokens
        if hw is not None and tuple(hw) != tuple(x.shape[-2:]):
            x = F.interpolate(x, size=tuple(hw), mode="bicubic", antialias=True, align_corners=False)
            x = v.patch_embed.proj(x)
            grid = x.shape[-2:]
            x = v.patch_embed.norm(x.flatten(2).transpose(1, 2))
            pos = resample_abs_pos_embed(v.pos_embed, tuple(grid), old_size=v.patch_embed.grid_size,
                                         num_prefix_tokens=npre)
            x = torch.cat([v.cls_token.expand(x.shape[0], -1, -1), x], dim=1) + pos.to(x.dtype)
            x = v.pos_drop(x)
        else:
            x = v.patch_embed(x)
            x = v._pos_embed(x)
        x = v.patch_drop(x)
        x = v.norm_pre(x)
        every = mode == "merge_r" and self.tome_r > 0
        reduce = (mode != "none" and keep < 1.0 and mode != "merge_r") or every
        size, counts = None, []
        for i, blk in enumerate(v.blocks):
            n_in = x.shape[1]
            if reduce and (every or i in self.reduce_layers):
                if self.fast_attn:
                    y, cls_row, key = attention_fast(blk.attn, blk.norm1(x), size,
                                                     need_cls=mode.startswith("cls_attn"))
                else:
                    y, attn, key = attention_with_maps(blk.attn, blk.norm1(x), size)
                    cls_row = attn[:, :, 0]
                x = x + blk.drop_path1(blk.ls1(y))
                n_patch = x.shape[1] - npre
                n_keep = max(1, math.ceil(n_patch * keep))
                if every:  # ToMe: constant r per block, at most half of the tokens
                    n_keep = n_patch - min(self.tome_r, n_patch // 2)
                if mode in ("merge", "merge_r"):
                    x, size = bipartite_merge(x, size, key, n_patch - n_keep, npre)
                elif mode == "cls_attn_evit":
                    x = select_tokens_fuse(x, cls_row[:, :, npre:].mean(1).float(), n_keep, npre, normalize=False)
                elif mode == "cls_attn":
                    x = select_tokens(x, cls_row[:, :, npre:].mean(1).float(), n_keep, npre)
                elif mode == "cls_attn_low":
                    x = select_tokens(x, -cls_row[:, :, npre:].mean(1).float(), n_keep, npre)
                elif mode == "cls_attn_fuse":
                    x = select_tokens_fuse(x, cls_row[:, :, npre:].mean(1).float(), n_keep, npre)
                else:
                    x = select_tokens(x, torch.rand(x.shape[0], n_patch, device=x.device), n_keep, npre)
                x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))
            elif size is not None:  # after merging, attention must account for token sizes
                attn_fn = attention_fast if self.fast_attn else attention_with_maps
                y, _, _ = attn_fn(blk.attn, blk.norm1(x), size)
                x = x + blk.drop_path1(blk.ls1(y))
                x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))
            else:
                x = blk(x)
            counts.append((n_in, x.shape[1]))
        self.last_counts = counts
        return v.norm(x)[:, 0]

    def forward(self, x, mode="none", keep=1.0, hw=None):
        feat = self.backbone(x, mode, keep, hw)   # used by the triplet loss
        bn = self.bnneck(feat)                # used for retrieval and distillation
        logits = self.classifier(bn) if self.training else None
        return feat, bn, logits


def vit_gflops(model, counts=None):
    """Analytic multiply-accumulate estimate (GFLOPs) of the backbone for the given token counts."""
    v = model.vit
    d = v.embed_dim
    ph, pw = v.patch_embed.patch_size
    n_patches = v.patch_embed.num_patches
    mlp_hidden = v.blocks[0].mlp.fc1.out_features
    counts = counts or model.last_counts
    total = n_patches * (ph * pw * 3) * d
    for n_attn, n_mlp in counts:
        total += 4 * n_attn * d * d + 2 * n_attn * n_attn * d   # qkv + proj, QK^T and AV
        total += 2 * n_mlp * d * mlp_hidden                      # MLP
    return total / 1e9


# --------------------------------------------------------------------------- losses
def batch_hard_triplet(feat, labels, margin=0.3):
    feat = feat.float()
    dist = torch.cdist(feat, feat)
    same = labels[:, None] == labels[None, :]
    hardest_pos = (dist - 1e6 * (~same)).max(dim=1).values
    hardest_neg = (dist + 1e6 * same).min(dim=1).values
    return F.relu(hardest_pos - hardest_neg + margin).mean()


# --------------------------------------------------------------------------- evaluation
@torch.no_grad()
def extract_features(model, loader, device, mode="none", keep=1.0, amp=True):
    model.eval()
    feats, pids, cams = [], [], []
    for imgs, pid, cam in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            _, bn, _ = model(imgs, mode, keep)
        feats.append(F.normalize(bn.float(), dim=1).cpu())
        pids.append(torch.as_tensor(pid))
        cams.append(torch.as_tensor(cam))
    return torch.cat(feats), torch.cat(pids).numpy(), torch.cat(cams).numpy()


def evaluate_rank(qf, gf, q_pids, g_pids, q_cams, g_cams, device=None, chunk=512, max_rank=10):
    """Standard Market-1501 protocol: gallery images with the query's pid AND camera are ignored."""
    device = device or torch.device("cpu")
    gf_d = gf.to(device)
    aps, cmc_sum, valid = [], np.zeros(max_rank), 0
    for s in range(0, qf.shape[0], chunk):
        sim = qf[s:s + chunk].to(device) @ gf_d.T
        order = sim.argsort(dim=1, descending=True).cpu().numpy()
        for j in range(order.shape[0]):
            qi = s + j
            o = order[j]
            keep = ~((g_pids[o] == q_pids[qi]) & (g_cams[o] == q_cams[qi]))
            matches = (g_pids[o] == q_pids[qi])[keep]
            if not matches.any():
                continue
            valid += 1
            hits = np.flatnonzero(matches)
            aps.append(np.mean((np.arange(len(hits)) + 1) / (hits + 1)))
            first = hits[0]
            if first < max_rank:
                cmc_sum[first:] += 1
    if valid == 0:
        raise RuntimeError("No valid query (every query needs a match under another camera)")
    return float(np.mean(aps)) * 100, (cmc_sum / valid) * 100


@torch.no_grad()
def measure_throughput(model, device, img_size, mode="none", keep=1.0, batch=64, iters=20, amp=True):
    """Images per second for the forward pass on random input."""
    model.eval()
    x = torch.randn(batch, 3, *img_size, device=device)
    def run():
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            model(x, mode, keep)
    for _ in range(3):
        run()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        run()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return batch * iters / (time.perf_counter() - t0)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
