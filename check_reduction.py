"""Checks of our token-reduction code against the official ToMe / EViT algorithms.

The reference functions below are TRANSCRIPTIONS (made by reading the official source on 2026-11-15) of
  - facebookresearch/ToMe  tome/merge.py: bipartite_soft_matching + merge_wavg  (class_token=False path)
  - youweiliang/evit       evit.py: Attention.forward top-k selection and Block.forward fused token
They are not the official files themselves, so a transcription error is possible.

Run: .venv/Scripts/python.exe check_reduction.py
"""
import torch

import common


def tome_reference(x, size, metric, r):
    """Official ToMe merge on a sequence without protected tokens; returns (x, size)."""
    t = metric.shape[1]
    r = min(r, t // 2)
    metric = metric / metric.norm(dim=-1, keepdim=True)
    a, b = metric[..., ::2, :], metric[..., 1::2, :]
    scores = a @ b.transpose(-1, -2)
    node_max, node_idx = scores.max(dim=-1)
    edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
    unm_idx, src_idx = edge_idx[..., r:, :], edge_idx[..., :r, :]
    dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    def merge(v):
        src, dst = v[..., ::2, :], v[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce="sum")
        return torch.cat([unm, dst], dim=1)

    xs, s = merge(x * size), merge(size)
    return xs / s, s


def evit_reference(x, cls_attn, n_keep, fuse):
    """Official EViT: x (B, 1+N, C) after the attention residual, cls_attn (B, N) head-averaged."""
    B, N1, C = x.shape
    _, idx = torch.topk(cls_attn, n_keep, dim=1, largest=True, sorted=True)
    non_cls = x[:, 1:]
    kept = torch.gather(non_cls, 1, idx.unsqueeze(-1).expand(-1, -1, C))
    if not fuse:
        return torch.cat([x[:, :1], kept], dim=1)
    mask = torch.ones(B, N1 - 1, dtype=torch.bool).scatter(1, idx, False)
    compl = mask.nonzero()[:, 1].view(B, -1)
    non_topk = torch.gather(non_cls, 1, compl.unsqueeze(-1).expand(-1, -1, C))
    w = torch.gather(cls_attn, 1, compl)
    extra = torch.sum(non_topk * w.unsqueeze(-1), dim=1, keepdim=True)  # weighted SUM, not average
    return torch.cat([x[:, :1], kept, extra], dim=1)


def main():
    torch.manual_seed(0)
    B, N, C = 4, 128, 384
    worst = 0.0
    for _ in range(20):
        x = torch.randn(B, N + 1, C, dtype=torch.float64)
        metric = torch.randn(B, N + 1, 64, dtype=torch.float64)
        size = torch.randint(1, 5, (B, N + 1, 1)).double()
        size[:, 0] = 1
        for r in (1, 13, 38, 64):
            ox, osz = common.bipartite_merge(x, size, metric, r, 1)
            rx, rsz = tome_reference(x[:, 1:], size[:, 1:], metric[:, 1:], r)
            assert ox.shape[1] == N + 1 - r
            worst = max(worst, (ox[:, 1:] - rx).abs().max().item(), (osz[:, 1:] - rsz).abs().max().item())
    print(f"ToMe merge, same partition: max |ours - reference| = {worst:.2e}")

    w_sel = w_fuse_dir = w_fuse_scale = 0.0
    for _ in range(20):
        x = torch.randn(B, N + 1, C, dtype=torch.float64)
        attn = torch.rand(B, N, dtype=torch.float64).softmax(-1) * 0.8  # class token keeps 0.2 of its attention
        for n_keep in (103, 90, 64):
            ours = common.select_tokens(x, attn, n_keep, 1)
            ref = evit_reference(x, attn, n_keep, fuse=False)
            # same SET of tokens (order differs: ours spatial, EViT by attention); compare order-free
            w_sel = max(w_sel, (ours[:, 1:].sort(1).values - ref[:, 1:].sort(1).values).abs().max().item())
            of = common.select_tokens_fuse(x, attn, n_keep, 1)[:, -1]
            rf = evit_reference(x, attn, n_keep, fuse=True)[:, -1]
            cos = torch.nn.functional.cosine_similarity(of, rf, dim=-1)
            w_fuse_dir = max(w_fuse_dir, (1 - cos).abs().max().item())
            w_fuse_scale = max(w_fuse_scale, (of.norm(dim=-1) / rf.norm(dim=-1)).max().item())
    print(f"EViT top-k selection: max |ours - reference| (order-free) = {w_sel:.2e}")
    print(f"EViT fused token: max (1 - cosine) = {w_fuse_dir:.2e}; max norm ratio ours/reference = {w_fuse_scale:.1f}")


if __name__ == "__main__":
    main()
